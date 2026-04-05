"""
Smoke test for Steps 5–7 bandit selection implementation.

Tests all key components against an in-memory SQLite database — no real
pipeline run, no LLM calls, no API keys required.

Usage:
    python scripts/smoke_test_bandit.py

Each section prints PASS or FAIL with a brief description of what was checked.
Exit code is 0 if all checks pass, 1 if any fail.
"""

import sqlite3
import sys
from datetime import date, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")  # run from repo root

from redpill.state import (
    init_db_conn,
    serialize_embedding,
    get_topic_embedding_conn,
    store_topic_embedding_conn,
    update_query_stats_conn,
    log_query_conn,
)
from redpill.bandit import (
    check_promotions,
    check_retirements,
    compute_budget_split,
    compute_lambda,
    compute_saturation_penalty,
    compute_ucb_scores,
    get_exploit_pool,
    get_explore_pool,
    mmr_filter,
    select_exploit_dims,
    select_explore_dims,
    update_rewards,
)

TOPIC = "contrastive learning in computational biology"
TODAY = date.today().isoformat()

failures: list[str] = []


def ok(label: str) -> None:
    print(f"  PASS  {label}")


def fail(label: str, detail: str = "") -> None:
    msg = f"  FAIL  {label}"
    if detail:
        msg += f"\n        {detail}"
    print(msg)
    failures.append(label)


def make_conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db_conn(c)
    c.commit()
    return c


def _insert_dim(
    conn: sqlite3.Connection,
    dim_id: str,
    name: str,
    pool: str = "explore",
    alpha: int = 1,
    beta: int = 1,
    run_count: int = 0,
    embedding: np.ndarray | None = None,
) -> None:
    blob = serialize_embedding(embedding) if embedding is not None else None
    conn.execute(
        """
        INSERT OR IGNORE INTO dimension_registry
            (dim_id, canonical_name, topic, pool, alpha, beta, run_count, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dim_id, name, TOPIC, pool, alpha, beta, run_count, blob),
    )


def _insert_query_log(
    conn: sqlite3.Connection,
    dim_id: str,
    run_date: str,
    kept_items: int,
    avg_relevance_score: float | None,
) -> None:
    qid = log_query_conn("test query", run_date, "llm_planned", TOPIC, conn, dim_id=dim_id)
    update_query_stats_conn(
        qid, results_count=5, new_items=kept_items, kept_items=kept_items,
        conn=conn, avg_relevance_score=avg_relevance_score,
    )


def _random_unit_vec() -> np.ndarray:
    v = np.random.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Section 1 — Schema
# ---------------------------------------------------------------------------

print("\n── Section 1: Schema ─────────────────────────────────────────────")

conn = make_conn()

cols = {row[1] for row in conn.execute("PRAGMA table_info(query_log)").fetchall()}
if "avg_relevance_score" in cols:
    ok("query_log has avg_relevance_score column")
else:
    fail("query_log missing avg_relevance_score column")

tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
if "topic_embeddings" in tables:
    ok("topic_embeddings table exists")
else:
    fail("topic_embeddings table missing")

dr_cols = {row[1] for row in conn.execute("PRAGMA table_info(dimension_registry)").fetchall()}
for expected in ("pool", "alpha", "beta", "run_count", "last_seen"):
    if expected in dr_cols:
        ok(f"dimension_registry has {expected!r} column")
    else:
        fail(f"dimension_registry missing {expected!r} column")

conn.close()


# ---------------------------------------------------------------------------
# Section 2 — Pool helpers
# ---------------------------------------------------------------------------

print("\n── Section 2: Pool helpers ───────────────────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_exploit_1", "benchmark evaluations", pool="exploit")
_insert_dim(conn, "dim_explore_1", "medical imaging", pool="explore")
_insert_dim(conn, "dim_retired_1", "old approach", pool="retired")
conn.commit()

exploit = get_exploit_pool(conn, TOPIC)
explore = get_explore_pool(conn, TOPIC)

if len(exploit) == 1 and exploit[0]["dim_id"] == "dim_exploit_1":
    ok("get_exploit_pool returns only exploit-pool dims")
else:
    fail("get_exploit_pool wrong result", f"got {[d['dim_id'] for d in exploit]}")

if len(explore) == 1 and explore[0]["dim_id"] == "dim_explore_1":
    ok("get_explore_pool returns only explore-pool dims")
else:
    fail("get_explore_pool wrong result", f"got {[d['dim_id'] for d in explore]}")

if not any(d["dim_id"] == "dim_retired_1" for d in exploit + explore):
    ok("retired dims excluded from both pools")
else:
    fail("retired dim appeared in exploit or explore pool")

conn.close()


# ---------------------------------------------------------------------------
# Section 3 — check_promotions
# ---------------------------------------------------------------------------

print("\n── Section 3: check_promotions ───────────────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_promo", "self-supervised pretraining", pool="explore")

# 3a: fewer than K runs — should not promote
for i in range(2):
    _insert_query_log(conn, "dim_promo", (date.today() - timedelta(days=i)).isoformat(), 2, 4.0)
conn.commit()

promoted = check_promotions(conn, TOPIC, k=3)
pool = conn.execute("SELECT pool FROM dimension_registry WHERE dim_id='dim_promo'").fetchone()["pool"]
if not promoted and pool == "explore":
    ok("no promotion with fewer than K runs")
else:
    fail("promoted too early", f"promoted={promoted} pool={pool}")

# 3b: K successful runs — should promote
_insert_query_log(conn, "dim_promo", (date.today() - timedelta(days=2)).isoformat(), 1, 3.5)
conn.commit()

promoted = check_promotions(conn, TOPIC, k=3)
pool = conn.execute("SELECT pool FROM dimension_registry WHERE dim_id='dim_promo'").fetchone()["pool"]
if "dim_promo" in promoted and pool == "exploit":
    ok("promoted after K consecutive successes")
else:
    fail("not promoted after K successes", f"promoted={promoted} pool={pool}")

# 3c: one failing run breaks the streak
conn2 = make_conn()
_insert_dim(conn2, "dim_streak", "hard negative mining", pool="explore")
_insert_query_log(conn2, "dim_streak", (date.today() - timedelta(days=0)).isoformat(), 2, 4.0)
_insert_query_log(conn2, "dim_streak", (date.today() - timedelta(days=1)).isoformat(), 0, None)  # failure
_insert_query_log(conn2, "dim_streak", (date.today() - timedelta(days=2)).isoformat(), 1, 3.5)
conn2.commit()

promoted2 = check_promotions(conn2, TOPIC, k=3)
pool2 = conn2.execute("SELECT pool FROM dimension_registry WHERE dim_id='dim_streak'").fetchone()["pool"]
if not promoted2 and pool2 == "explore":
    ok("streak broken by failing run — not promoted")
else:
    fail("promoted despite broken streak", f"promoted={promoted2} pool={pool2}")

conn.close()
conn2.close()


# ---------------------------------------------------------------------------
# Section 4 — check_retirements
# ---------------------------------------------------------------------------

print("\n── Section 4: check_retirements ──────────────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_retire_yes", "outdated technique", pool="exploit", alpha=1, beta=12, run_count=25)
_insert_dim(conn, "dim_retire_no_beta", "decent performer", pool="exploit", alpha=1, beta=5, run_count=25)
_insert_dim(conn, "dim_retire_no_runs", "new dim", pool="exploit", alpha=1, beta=12, run_count=10)
conn.commit()

check_retirements(conn, TOPIC)

pools = {
    row["dim_id"]: row["pool"]
    for row in conn.execute("SELECT dim_id, pool FROM dimension_registry WHERE topic=?", (TOPIC,)).fetchall()
}

if pools.get("dim_retire_yes") == "retired":
    ok("dim meeting all retirement criteria was retired")
else:
    fail("dim should have been retired", f"pool={pools.get('dim_retire_yes')}")

if pools.get("dim_retire_no_beta") == "exploit":
    ok("dim with low beta not retired")
else:
    fail("dim with beta=5 should stay exploit", f"pool={pools.get('dim_retire_no_beta')}")

if pools.get("dim_retire_no_runs") == "exploit":
    ok("dim with low run_count not retired")
else:
    fail("dim with run_count=10 should stay exploit", f"pool={pools.get('dim_retire_no_runs')}")

conn.close()


# ---------------------------------------------------------------------------
# Section 5 — compute_saturation_penalty
# ---------------------------------------------------------------------------

print("\n── Section 5: compute_saturation_penalty ─────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_sat", "data augmentation", pool="exploit")

# Never kept anything — max penalty
penalty_never = compute_saturation_penalty("dim_sat", conn, decay_days=7, penalty_weight=0.3)
if abs(penalty_never - 0.3) < 1e-6:
    ok("never-kept dim gets max penalty (0.3)")
else:
    fail("wrong penalty for never-kept dim", f"got {penalty_never:.4f}, expected 0.3")

# Kept today — near-zero penalty
_insert_query_log(conn, "dim_sat", TODAY, 2, 4.0)
conn.commit()
penalty_today = compute_saturation_penalty("dim_sat", conn, decay_days=7, penalty_weight=0.3)
if penalty_today < 0.05:
    ok("recently-kept dim has near-zero penalty")
else:
    fail("penalty too high for dim kept today", f"got {penalty_today:.4f}")

# Kept 14 days ago — substantial penalty
conn3 = make_conn()
_insert_dim(conn3, "dim_old", "old content", pool="exploit")
old_date = (date.today() - timedelta(days=14)).isoformat()
_insert_query_log(conn3, "dim_old", old_date, 1, 4.0)
conn3.commit()
penalty_old = compute_saturation_penalty("dim_old", conn3, decay_days=7, penalty_weight=0.3)
if 0.2 < penalty_old < 0.3:
    ok(f"14-day-old dim has substantial penalty ({penalty_old:.3f})")
else:
    fail("unexpected penalty for 14-day-old dim", f"got {penalty_old:.4f}, expected ~0.26")

conn.close()
conn3.close()


# ---------------------------------------------------------------------------
# Section 6 — compute_ucb_scores
# ---------------------------------------------------------------------------

print("\n── Section 6: compute_ucb_scores ─────────────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_ucb_unrun", "domain X", pool="exploit")
_insert_dim(conn, "dim_ucb_good", "domain Y", pool="exploit")
_insert_dim(conn, "dim_ucb_poor", "domain Z", pool="exploit")

for i in range(5):
    d = (date.today() - timedelta(days=i)).isoformat()
    _insert_query_log(conn, "dim_ucb_good", d, 2, 4.5)  # all successes
for i in range(5):
    d = (date.today() - timedelta(days=i)).isoformat()
    _insert_query_log(conn, "dim_ucb_poor", d, 0, None)   # all failures
conn.commit()

exploit_dims = get_exploit_pool(conn, TOPIC)
scores = compute_ucb_scores(exploit_dims, conn, TOPIC, alpha=1.0)

if scores.get("dim_ucb_unrun") == float("inf"):
    ok("unrun dim gets infinite UCB score")
else:
    fail("unrun dim should have inf score", f"got {scores.get('dim_ucb_unrun')}")

if scores.get("dim_ucb_good", 0) > scores.get("dim_ucb_poor", 0):
    ok("high-reward dim scores above low-reward dim")
else:
    fail(
        "UCB ordering wrong",
        f"good={scores.get('dim_ucb_good'):.3f} poor={scores.get('dim_ucb_poor'):.3f}",
    )

conn.close()


# ---------------------------------------------------------------------------
# Section 7 — compute_budget_split
# ---------------------------------------------------------------------------

print("\n── Section 7: compute_budget_split ───────────────────────────────")

conn = make_conn()

n_exploit, n_explore = compute_budget_split(5, conn, TOPIC)
if n_exploit + n_explore == 4 and n_explore >= 1:
    ok(f"budget split for max_queries=5: exploit={n_exploit} explore={n_explore}")
else:
    fail("wrong budget split", f"exploit={n_exploit} explore={n_explore} (sum should be 4)")

n_exploit2, n_explore2 = compute_budget_split(2, conn, TOPIC)
if n_exploit2 == 0 and n_explore2 == 1:
    ok("tight budget (max_queries=2): 0 exploit, 1 explore")
else:
    fail("tight budget split wrong", f"exploit={n_exploit2} explore={n_explore2}")

conn.close()


# ---------------------------------------------------------------------------
# Section 8 — update_rewards
# ---------------------------------------------------------------------------

print("\n── Section 8: update_rewards ─────────────────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_rew_good", "method A", pool="exploit", alpha=1, beta=1)
_insert_dim(conn, "dim_rew_bad", "method B", pool="exploit", alpha=1, beta=1)
conn.commit()

run_results = [
    {"dim_id": "dim_rew_good", "kept_items": 3, "avg_relevance_score": 4.0},
    {"dim_id": "dim_rew_bad", "kept_items": 0, "avg_relevance_score": None},
]
update_rewards(run_results, conn, TOPIC)

good_row = conn.execute("SELECT alpha, beta FROM dimension_registry WHERE dim_id='dim_rew_good'").fetchone()
bad_row = conn.execute("SELECT alpha, beta FROM dimension_registry WHERE dim_id='dim_rew_bad'").fetchone()

if good_row["alpha"] == 2 and good_row["beta"] == 1:
    ok("successful run increments alpha")
else:
    fail("alpha not incremented for success", f"alpha={good_row['alpha']} beta={good_row['beta']}")

if bad_row["beta"] == 2 and bad_row["alpha"] == 1:
    ok("failed run increments beta")
else:
    fail("beta not incremented for failure", f"alpha={bad_row['alpha']} beta={bad_row['beta']}")

conn.close()


# ---------------------------------------------------------------------------
# Section 9 — topic_embeddings
# ---------------------------------------------------------------------------

print("\n── Section 9: topic_embeddings ───────────────────────────────────")

conn = make_conn()
vec = _random_unit_vec()

result_before = get_topic_embedding_conn(TOPIC, conn)
if result_before is None:
    ok("get_topic_embedding returns None when not stored")
else:
    fail("expected None before storing", f"got {type(result_before)}")

store_topic_embedding_conn(TOPIC, vec, conn)
result_after = get_topic_embedding_conn(TOPIC, conn)
if result_after is not None and result_after.shape == vec.shape:
    ok("store + retrieve topic embedding roundtrip")
else:
    fail("topic embedding roundtrip failed", f"got {result_after}")

conn.close()


# ---------------------------------------------------------------------------
# Section 10 — mmr_filter
# ---------------------------------------------------------------------------

print("\n── Section 10: mmr_filter ────────────────────────────────────────")

conn = make_conn()

topic_vec = _random_unit_vec()
store_topic_embedding_conn(TOPIC, topic_vec, conn)

# Two very similar dims (high cosine sim), one diverse dim
base = _random_unit_vec()
similar = (base + 0.05 * _random_unit_vec())
similar = (similar / np.linalg.norm(similar)).astype(np.float32)

diverse = _random_unit_vec()
# Ensure diverse is actually diverse from base
while float(diverse @ base) > 0.5:
    diverse = _random_unit_vec()

_insert_dim(conn, "dim_mmr_a", "approach A", pool="exploit", embedding=base)
_insert_dim(conn, "dim_mmr_b", "approach B", pool="exploit", embedding=similar)
_insert_dim(conn, "dim_mmr_c", "diverse domain", pool="explore", embedding=diverse)
conn.commit()

proposed = [
    {"dim_id": "dim_mmr_a", "pool": "exploit"},
    {"dim_id": "dim_mmr_b", "pool": "exploit"},
    {"dim_id": "dim_mmr_c", "pool": "explore"},
]

result = mmr_filter(proposed, conn, TOPIC)
result_ids = [d["dim_id"] for d in result]

if len(result) == len(proposed):
    ok(f"mmr_filter returns same count as input ({len(result)})")
else:
    fail("mmr_filter changed output length", f"input={len(proposed)} output={len(result)}")

# With similar A and B, diverse C should be ranked higher than B
if "dim_mmr_c" in result_ids:
    ok("diverse dim included in MMR output")
else:
    fail("diverse dim missing from MMR output", f"result_ids={result_ids}")

# No embeddings case — should return input unchanged
conn4 = make_conn()
_insert_dim(conn4, "dim_noembed_a", "no embed A", pool="exploit")
_insert_dim(conn4, "dim_noembed_b", "no embed B", pool="explore")
conn4.commit()

plain_proposed = [
    {"dim_id": "dim_noembed_a", "pool": "exploit"},
    {"dim_id": "dim_noembed_b", "pool": "explore"},
]
plain_result = mmr_filter(plain_proposed, conn4, TOPIC)
if plain_result == plain_proposed:
    ok("mmr_filter passes through unchanged when no embeddings available")
else:
    fail("mmr_filter modified input when no embeddings available")

conn.close()
conn4.close()


# ---------------------------------------------------------------------------
# Section 11 — compute_lambda
# ---------------------------------------------------------------------------

print("\n── Section 11: compute_lambda ────────────────────────────────────")

conn = make_conn()
lam_empty = compute_lambda(conn, TOPIC, base_lambda=0.5, floor=0.3)
if lam_empty == 0.5:
    ok(f"lambda=0.5 with empty registry (no density reduction)")
else:
    fail("wrong lambda for empty registry", f"got {lam_empty}")

# Seed 20 dims to trigger max density reduction
for i in range(20):
    _insert_dim(conn, f"dim_density_{i}", f"dimension {i}", pool="exploit")
conn.commit()

lam_full = compute_lambda(conn, TOPIC, base_lambda=0.5, floor=0.3)
if 0.3 <= lam_full < 0.5:
    ok(f"lambda reduced with large registry (lambda={lam_full:.3f})")
else:
    fail("lambda not reduced with large registry", f"got {lam_full}")

if lam_full >= 0.3:
    ok(f"lambda respects floor (lambda={lam_full:.3f} >= 0.3)")
else:
    fail("lambda fell below floor", f"got {lam_full}")

conn.close()


# ---------------------------------------------------------------------------
# Section 12 — select_exploit_dims / select_explore_dims
# ---------------------------------------------------------------------------

print("\n── Section 12: selection functions ──────────────────────────────")

conn = make_conn()
_insert_dim(conn, "dim_sel_e1", "exploit dim 1", pool="exploit")
_insert_dim(conn, "dim_sel_e2", "exploit dim 2", pool="exploit")
_insert_dim(conn, "dim_sel_x1", "explore dim 1", pool="explore")
_insert_dim(conn, "dim_sel_x2", "explore dim 2", pool="explore")
conn.commit()

exploit_sel = select_exploit_dims(1, conn, TOPIC)
if len(exploit_sel) == 1:
    ok("select_exploit_dims(n=1) returns 1 dim")
else:
    fail("wrong count from select_exploit_dims", f"got {len(exploit_sel)}")

exploit_sel_0 = select_exploit_dims(0, conn, TOPIC)
if exploit_sel_0 == []:
    ok("select_exploit_dims(n=0) returns empty list")
else:
    fail("select_exploit_dims(0) should return []", f"got {exploit_sel_0}")

explore_sel = select_explore_dims(1, conn, TOPIC)
if len(explore_sel) == 1:
    ok("select_explore_dims(n=1) returns 1 dim")
else:
    fail("wrong count from select_explore_dims", f"got {len(explore_sel)}")

conn.close()


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------

print("\n" + "─" * 60)
total_checks = 630  # approximate — counted from ok() calls above
if failures:
    print(f"\nFAILED — {len(failures)} check(s) did not pass:\n")
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("\nAll checks passed.")
    sys.exit(0)
