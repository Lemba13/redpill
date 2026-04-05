"""
bandit.py — Bandit-based dimension selection for the query planning pipeline.

Implements the two-pool (explore/exploit) selection layer that sits on top of
the dimension registry built in Steps 2–4.

Pool states (dimension_registry.pool):
    explore  — newly registered; selected by coverage-gap scoring
    exploit  — promoted after consistent value; selected by UCB + saturation penalty
    retired  — persistently unproductive; excluded from all selection

Public API:
    check_promotions(conn, topic, k=3) -> list[str]
    check_retirements(conn, topic) -> None
    get_exploit_pool(conn, topic) -> list[sqlite3.Row]
    get_explore_pool(conn, topic) -> list[sqlite3.Row]
    compute_budget_split(max_queries, conn, topic) -> tuple[int, int]
    compute_ucb_scores(exploit_dims, conn, topic, alpha=1.0) -> dict[str, float]
    compute_saturation_penalty(dim_id, conn, decay_days=7, penalty_weight=0.3) -> float
    select_exploit_dims(n_slots, conn, topic, alpha=1.0) -> list[sqlite3.Row]
    select_explore_dims(n_slots, conn, topic) -> list[sqlite3.Row]
    update_rewards(run_results, conn, topic) -> None
    mmr_filter(proposed_dims, conn, topic, lambda_val=None) -> list[dict]
    compute_lambda(conn, topic, base_lambda=0.5, floor=0.3) -> float
"""

import logging
import math
from datetime import date
from typing import TYPE_CHECKING

from redpill.registry import get_dimension_axis_tags
from redpill.state import deserialize_embedding

if TYPE_CHECKING:
    import sqlite3

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------

def get_exploit_pool(conn: "sqlite3.Connection", topic: str) -> list:
    """Return all exploit-pool dimensions for *topic*."""
    return conn.execute(
        """
        SELECT dim_id, canonical_name, alpha, beta, run_count, last_seen
        FROM dimension_registry
        WHERE topic = ? AND pool = 'exploit'
        """,
        (topic,),
    ).fetchall()


def get_explore_pool(conn: "sqlite3.Connection", topic: str) -> list:
    """Return all explore-pool dimensions for *topic*."""
    return conn.execute(
        """
        SELECT dim_id, canonical_name, alpha, beta, run_count, last_seen
        FROM dimension_registry
        WHERE topic = ? AND pool = 'explore'
        """,
        (topic,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Pool transitions
# ---------------------------------------------------------------------------

def check_promotions(conn: "sqlite3.Connection", topic: str, k: int = 3) -> list[str]:
    """Promote explore-pool dimensions that have k consecutive successful runs.

    A successful run is: kept_items > 0 AND avg_relevance_score >= 3.0.
    Returns list of dim_ids promoted this call.
    """
    explore_dims = get_explore_pool(conn, topic)
    promoted: list[str] = []

    for dim in explore_dims:
        rows = conn.execute(
            """
            SELECT kept_items, avg_relevance_score
            FROM query_log
            WHERE dim_id = ? AND topic = ?
            ORDER BY run_date DESC
            LIMIT ?
            """,
            (dim["dim_id"], topic, k),
        ).fetchall()

        if len(rows) < k:
            continue

        all_success = all(
            r["kept_items"] > 0 and (r["avg_relevance_score"] or 0.0) >= 3.0
            for r in rows
        )

        if all_success:
            conn.execute(
                "UPDATE dimension_registry SET pool = 'exploit' WHERE dim_id = ?",
                (dim["dim_id"],),
            )
            promoted.append(dim["dim_id"])
            log.info(
                "Promoted %r (%s) to exploit pool",
                dim["canonical_name"], dim["dim_id"],
            )

    return promoted


def check_retirements(conn: "sqlite3.Connection", topic: str) -> None:
    """Retire exploit-pool dimensions with persistently poor performance.

    Criteria: beta >= 10 AND alpha <= 1 AND run_count >= 20.
    Logs retiring dimensions before executing.
    """
    candidates = conn.execute(
        """
        SELECT dim_id, canonical_name, alpha, beta, run_count
        FROM dimension_registry
        WHERE topic = ? AND pool = 'exploit'
          AND beta >= 10 AND alpha <= 1 AND run_count >= 20
        """,
        (topic,),
    ).fetchall()

    for row in candidates:
        log.info(
            "Retiring %r (%s): alpha=%d beta=%d run_count=%d",
            row["canonical_name"], row["dim_id"],
            row["alpha"], row["beta"], row["run_count"],
        )

    conn.execute(
        """
        UPDATE dimension_registry
        SET pool = 'retired'
        WHERE topic = ? AND pool = 'exploit'
          AND beta >= 10 AND alpha <= 1 AND run_count >= 20
        """,
        (topic,),
    )


# ---------------------------------------------------------------------------
# Budget split
# ---------------------------------------------------------------------------

def _is_exploit_reward_declining(
    conn: "sqlite3.Connection",
    topic: str,
    lookback: int = 7,
) -> bool:
    """Return True if average exploit-pool success rate has declined recently."""
    rows = conn.execute(
        """
        SELECT run_date,
               AVG(CASE WHEN kept_items > 0 AND avg_relevance_score >= 3
                        THEN 1.0 ELSE 0.0 END) as avg_success
        FROM query_log q
        JOIN dimension_registry r ON q.dim_id = r.dim_id
        WHERE q.topic = ? AND r.pool = 'exploit'
          AND q.dim_id NOT IN ('dim_fallback', 'dim_base')
        GROUP BY run_date
        ORDER BY run_date DESC
        LIMIT ?
        """,
        (topic, lookback + 1),
    ).fetchall()

    if len(rows) < lookback + 1:
        return False

    recent = [r["avg_success"] for r in rows[:lookback]]
    baseline = rows[lookback]["avg_success"]
    recent_avg = sum(recent) / len(recent) if recent else 0.0
    return recent_avg < baseline


def compute_budget_split(
    max_queries: int,
    conn: "sqlite3.Connection",
    topic: str,
) -> tuple[int, int]:
    """Return (n_exploit_slots, n_explore_slots).

    Always reserves at least 1 explore slot. Increases explore allocation
    when exploit pool reward is declining.
    """
    available = max_queries - 1  # subtract base topic slot

    exploit_declining = _is_exploit_reward_declining(conn, topic)

    if available <= 2:
        return (available - 1, 1)

    n_explore = 2 if exploit_declining else 1
    n_exploit = available - n_explore
    return (n_exploit, n_explore)


# ---------------------------------------------------------------------------
# Saturation penalty
# ---------------------------------------------------------------------------

def compute_saturation_penalty(
    dim_id: str,
    conn: "sqlite3.Connection",
    decay_days: int = 7,
    penalty_weight: float = 0.3,
) -> float:
    """Return a penalty in [0, penalty_weight] based on days since last kept item.

    Approaches penalty_weight asymptotically. Returns to 0 when content is found.
    """
    row = conn.execute(
        """
        SELECT MAX(run_date) as last_kept_date
        FROM query_log
        WHERE dim_id = ? AND kept_items > 0
        """,
        (dim_id,),
    ).fetchone()

    if not row or not row["last_kept_date"]:
        return penalty_weight

    last_kept = date.fromisoformat(row["last_kept_date"])
    days_since = (date.today() - last_kept).days
    return penalty_weight * (1 - math.exp(-days_since / decay_days))


# ---------------------------------------------------------------------------
# UCB scoring
# ---------------------------------------------------------------------------

def compute_ucb_scores(
    exploit_dims: list,
    conn: "sqlite3.Connection",
    topic: str,
    alpha: float = 1.0,
    saturation_decay_days: int = 7,
    saturation_penalty_weight: float = 0.3,
) -> dict[str, float]:
    """Return {dim_id: adjusted_ucb_score} for all exploit pool dims.

    Unrun dims get float('inf') — always selected first.
    Score = mean_reward + alpha * sqrt(log(total_runs) / runs_for_dim) - saturation_penalty.
    """
    dim_ids = [d["dim_id"] for d in exploit_dims]
    if not dim_ids:
        return {}

    rows = conn.execute(
        """
        SELECT dim_id, COUNT(*) as runs_for_dim,
               AVG(CASE WHEN kept_items > 0 AND avg_relevance_score >= 3.0
                        THEN 1.0 ELSE 0.0 END) as mean_reward
        FROM query_log
        WHERE topic = ?
          AND run_date >= date('now', '-30 days')
          AND dim_id IN ({})
        GROUP BY dim_id
        """.format(",".join("?" * len(dim_ids))),
        [topic] + dim_ids,
    ).fetchall()

    total_runs_row = conn.execute(
        """
        SELECT COUNT(DISTINCT run_date) as total_runs
        FROM query_log
        WHERE topic = ?
          AND run_date >= date('now', '-30 days')
          AND dim_id NOT IN ('dim_fallback', 'dim_base')
        """,
        (topic,),
    ).fetchone()

    total_runs = total_runs_row["total_runs"] if total_runs_row else 1
    stats = {r["dim_id"]: r for r in rows}

    scores: dict[str, float] = {}
    for dim in exploit_dims:
        dim_id = dim["dim_id"]

        if dim_id not in stats or stats[dim_id]["runs_for_dim"] == 0:
            scores[dim_id] = float("inf")
            continue

        s = stats[dim_id]
        exploration_bonus = alpha * math.sqrt(
            math.log(max(total_runs, 1)) / s["runs_for_dim"]
        )
        ucb = (s["mean_reward"] or 0.0) + exploration_bonus
        penalty = compute_saturation_penalty(
            dim_id, conn,
            decay_days=saturation_decay_days,
            penalty_weight=saturation_penalty_weight,
        )
        scores[dim_id] = ucb - penalty

    return scores


# ---------------------------------------------------------------------------
# Exploit selection
# ---------------------------------------------------------------------------

def select_exploit_dims(
    n_slots: int,
    conn: "sqlite3.Connection",
    topic: str,
    alpha: float = 1.0,
    saturation_decay_days: int = 7,
    saturation_penalty_weight: float = 0.3,
) -> list:
    """Select the top n_slots exploit-pool dims by UCB score."""
    if n_slots == 0:
        return []

    exploit_dims = get_exploit_pool(conn, topic)
    if not exploit_dims:
        return []

    scores = compute_ucb_scores(
        exploit_dims, conn, topic, alpha=alpha,
        saturation_decay_days=saturation_decay_days,
        saturation_penalty_weight=saturation_penalty_weight,
    )

    ranked = sorted(
        exploit_dims,
        key=lambda d: scores.get(d["dim_id"], 0),
        reverse=True,
    )
    return ranked[:n_slots]


# ---------------------------------------------------------------------------
# Explore selection
# ---------------------------------------------------------------------------

def _get_exploit_axis_counts(conn: "sqlite3.Connection", topic: str) -> dict[str, int]:
    """Count exploit-pool dims per primary axis for coverage-gap scoring."""
    exploit_dims = get_exploit_pool(conn, topic)
    counts: dict[str, int] = {}
    for dim in exploit_dims:
        axis_info = get_dimension_axis_tags(dim["dim_id"], topic, conn)
        axis = axis_info.get("primary_axis", "unknown")
        counts[axis] = counts.get(axis, 0) + 1
    return counts


def select_explore_dims(
    n_slots: int,
    conn: "sqlite3.Connection",
    topic: str,
) -> list:
    """Select the top n_slots explore-pool dims by coverage-gap score.

    Coverage gap = axes underrepresented in the exploit pool score higher.
    Tiebreak: oldest last_seen first.
    """
    if n_slots == 0:
        return []

    explore_dims = get_explore_pool(conn, topic)
    if not explore_dims:
        return []

    exploit_axis_counts = _get_exploit_axis_counts(conn, topic)

    scored: list[tuple] = []
    for dim in explore_dims:
        axis_info = get_dimension_axis_tags(dim["dim_id"], topic, conn)
        axis = axis_info.get("primary_axis", "unknown")
        axis_count = exploit_axis_counts.get(axis, 0)
        # Negative count so ascending sort gives highest-gap first
        scored.append((dim, -axis_count, dim["last_seen"] or "1970-01-01"))

    scored.sort(key=lambda x: (x[1], x[2]))
    return [s[0] for s in scored[:n_slots]]


# ---------------------------------------------------------------------------
# Reward update
# ---------------------------------------------------------------------------

def update_rewards(
    run_results: list[dict],
    conn: "sqlite3.Connection",
    topic: str,
) -> None:
    """Update alpha/beta in dimension_registry from this run's results.

    run_results: list of {dim_id, kept_items, avg_relevance_score}
    query_log rows must already be written before calling this.
    """
    for result in run_results:
        dim_id = result["dim_id"]
        success = (
            result.get("kept_items", 0) > 0
            and (result.get("avg_relevance_score") or 0.0) >= 3.0
        )

        if success:
            conn.execute(
                "UPDATE dimension_registry SET alpha = alpha + 1 WHERE dim_id = ?",
                (dim_id,),
            )
        else:
            conn.execute(
                "UPDATE dimension_registry SET beta = beta + 1 WHERE dim_id = ?",
                (dim_id,),
            )


# ---------------------------------------------------------------------------
# MMR diversity filter
# ---------------------------------------------------------------------------

def compute_lambda(
    conn: "sqlite3.Connection",
    topic: str,
    base_lambda: float = 0.5,
    floor: float = 0.3,
) -> float:
    """Adaptive lambda for MMR. Decreases as registry grows or reward declines."""
    lambda_val = base_lambda

    registry_size = conn.execute(
        """
        SELECT COUNT(*) as n FROM dimension_registry
        WHERE topic = ? AND dim_id NOT IN ('dim_fallback', 'dim_base')
        """,
        (topic,),
    ).fetchone()["n"]

    density_factor = min(registry_size / 20.0, 1.0)
    lambda_val -= 0.1 * density_factor

    declining = _is_exploit_reward_declining(conn, topic, lookback=5)
    if declining:
        lambda_val -= 0.1

    lambda_val = max(lambda_val, floor)
    log.info(
        "MMR lambda=%.3f (registry_size=%d, declining=%s)",
        lambda_val, registry_size, "yes" if declining else "no",
    )
    return lambda_val


def _find_pool_replacement(
    pool: str,
    already_selected: list[dict],
    conn: "sqlite3.Connection",
    topic: str,
    dim_embeddings: dict[str, "any"],
) -> "dict | None":
    """Find the most diverse replacement from *pool* not already in selected."""
    selected_ids = {d["dim_id"] for d in already_selected}

    candidates = conn.execute(
        "SELECT dim_id, canonical_name FROM dimension_registry WHERE topic = ? AND pool = ?",
        (topic, pool),
    ).fetchall()

    best_candidate = None
    best_max_sim = float("inf")

    for row in candidates:
        cid = row["dim_id"]
        if cid in selected_ids or cid not in dim_embeddings:
            continue

        c_vec = dim_embeddings[cid]
        sims = [
            float(c_vec @ dim_embeddings[s["dim_id"]])
            for s in already_selected
            if s["dim_id"] in dim_embeddings
        ]
        max_sim = max(sims) if sims else 0.0

        if max_sim < best_max_sim:
            best_max_sim = max_sim
            best_candidate = dict(row)

    return best_candidate


def mmr_filter(
    proposed_dims: list[dict],
    conn: "sqlite3.Connection",
    topic: str,
    lambda_val: "float | None" = None,
    mmr_lambda_floor: float = 0.3,
) -> list[dict]:
    """Reorder (and possibly swap) proposed_dims for pairwise diversity.

    MMR iteratively selects the candidate that maximizes:
        lambda * sim(d, topic) - (1 - lambda) * max_sim(d, already_selected)

    If a selected dim is above 0.75 cosine similarity with an already-selected
    dim, it is swapped for the next best candidate from the same pool.

    Returns reordered list of same length as input (unembedded dims appended last).
    """
    if lambda_val is None:
        lambda_val = compute_lambda(conn, topic, floor=mmr_lambda_floor)

    # Load embeddings for all proposed dims
    dim_embeddings: dict[str, any] = {}
    for dim in proposed_dims:
        row = conn.execute(
            "SELECT embedding FROM dimension_registry WHERE dim_id = ?",
            (dim["dim_id"],),
        ).fetchone()
        if row and row["embedding"]:
            try:
                dim_embeddings[dim["dim_id"]] = deserialize_embedding(row["embedding"])
            except Exception as exc:
                log.warning("mmr_filter: failed to deserialize embedding for %s: %s", dim["dim_id"], exc)

    # Load topic embedding
    topic_row = conn.execute(
        "SELECT embedding FROM topic_embeddings WHERE topic = ?", (topic,)
    ).fetchone()

    if not topic_row or not dim_embeddings:
        return proposed_dims

    try:
        topic_vec = deserialize_embedding(topic_row["embedding"])
    except Exception as exc:
        log.warning("mmr_filter: failed to deserialize topic embedding: %s", exc)
        return proposed_dims

    selected: list[dict] = []
    remaining = list(proposed_dims)

    while remaining and len(selected) < len(proposed_dims):
        best_dim = None
        best_score = float("-inf")

        for dim in remaining:
            dim_id = dim["dim_id"]
            if dim_id not in dim_embeddings:
                continue

            d_vec = dim_embeddings[dim_id]
            relevance = float(d_vec @ topic_vec)

            if selected:
                max_sim = max(
                    float(d_vec @ dim_embeddings[s["dim_id"]])
                    for s in selected
                    if s["dim_id"] in dim_embeddings
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_val * relevance - (1 - lambda_val) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_dim = dim

        if best_dim is None:
            break

        # Check pairwise threshold against already selected
        if selected:
            best_vec = dim_embeddings.get(best_dim["dim_id"])
            max_pairwise = (
                max(
                    float(best_vec @ dim_embeddings[s["dim_id"]])
                    for s in selected
                    if s["dim_id"] in dim_embeddings
                )
                if best_vec is not None
                else 0.0
            )

            if max_pairwise > 0.75:
                pool = best_dim.get("pool", "explore")
                replacement = _find_pool_replacement(
                    pool, selected + [best_dim], conn, topic, dim_embeddings
                )
                if replacement:
                    # Add pool to replacement from best_dim's pool
                    replacement["pool"] = pool
                    remaining.remove(best_dim)
                    # replacement may not be in remaining (it's from DB, not proposed_dims)
                    # so just append it directly
                    selected.append(replacement)
                    log.debug(
                        "MMR swapped %r (sim=%.3f) for %r",
                        best_dim.get("canonical_name", best_dim["dim_id"]),
                        max_pairwise,
                        replacement.get("canonical_name", replacement["dim_id"]),
                    )
                    continue

        remaining.remove(best_dim)
        selected.append(best_dim)

    # Append any dims that couldn't be embedded (pass through unchanged)
    unembedded = [d for d in proposed_dims if d["dim_id"] not in dim_embeddings]
    return selected + unembedded
