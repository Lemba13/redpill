"""
animus.py — Synthesise DB state into a living KNOWLEDGE.md document.

Reads from redpill.db (read-only), writes to data/memory/KNOWLEDGE.md.
Fully independent of the daily pipeline — no changes to run_pipeline().
"""

import datetime
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from redpill.config import load_config, resolve_db_path
from redpill.state import get_top_terms_conn, get_top_terms_for_dim_conn
from redpill.summarize import OllamaClient, PlannerLLMClient, check_ollama

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path("data/memory/KNOWLEDGE.md")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_animus(config_path: str | None = None, db_path: str | None = None) -> None:
    """Entry point called by the CLI."""
    config = load_config(config_path)

    topic: str = config.get("topic", "").strip()
    if not topic:
        print("ERROR: 'topic' is required in config.", file=sys.stderr)
        sys.exit(1)

    resolved_db = db_path if db_path else resolve_db_path(config)

    ollama_cfg: dict = config.get("ollama_config", {})
    ollama_base_url: str = ollama_cfg.get("base_url", "http://localhost:11434")

    planner_cfg: dict = config.get("planner_llm", {})
    if planner_cfg.get("model"):
        llm_base_url: str = planner_cfg.get("base_url", ollama_base_url)
        llm_model: str = planner_cfg["model"]
        llm_timeout: int = int(planner_cfg.get("timeout", 120))
        llm_think: bool = bool(planner_cfg.get("think", True))
    else:
        llm_base_url = ollama_base_url
        llm_model = ollama_cfg.get("model", "qwen3:4b")
        llm_timeout = 120
        llm_think = True

    print(f'redpill animus: topic="{topic}"')

    knowledge_path = _KNOWLEDGE_PATH
    cutoff = _get_delta_cutoff(knowledge_path)

    if not Path(resolved_db).exists():
        print(
            f"ERROR: database not found at {resolved_db}. Run 'redpill run' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = sqlite3.connect(f"file:{resolved_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        new_articles = _read_new_articles(conn, topic, cutoff)

        if cutoff is None:
            print(
                f"redpill animus: cold start — reading all {len(new_articles)} articles from DB"
            )
        else:
            print(
                f"redpill animus: delta since {cutoff} — {len(new_articles)} new articles"
            )

        if len(new_articles) == 0:
            if cutoff is None:
                print(
                    f"redpill animus: No articles found for topic '{topic}'. "
                    "Run the pipeline first."
                )
                return
            else:
                print("redpill animus: no new articles since last run — nothing to synthesise.")
                return

        dimensions = _read_dimensions(conn, topic)
        print(f"redpill animus: {len(dimensions)} active dimensions found")

        dimension_terms_map = {
            dim["dim_id"]: _read_dimension_terms(conn, topic, dim["dim_id"])
            for dim in dimensions
        }

        global_terms = _read_global_terms(conn, topic)

    finally:
        conn.close()

    existing_knowledge = (
        knowledge_path.read_text(encoding="utf-8") if knowledge_path.exists() else None
    )

    prompt = _build_prompt(
        topic=topic,
        existing_knowledge=existing_knowledge,
        new_articles=new_articles,
        dimensions=dimensions,
        dimension_terms_map=dimension_terms_map,
        global_terms=global_terms,
        today=datetime.date.today(),
    )

    try:
        check_ollama(llm_base_url, llm_model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    client = PlannerLLMClient(
        base_url=llm_base_url,
        model=llm_model,
        think=llm_think,
        timeout=llm_timeout,
    )
    think_label = " [thinking]" if llm_think else ""
    print(f"redpill animus: calling LLM (ollama / {llm_model}{think_label})...")

    try:
        raw_output = client.generate(prompt, json_format=False)
    except RuntimeError as exc:
        print(f"ERROR: LLM call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    output = raw_output.strip()

    if not _validate_output(output):
        print(
            "ERROR: LLM output missing required sections (## Overview and ## Dimensions). "
            "Existing KNOWLEDGE.md preserved.",
            file=sys.stderr,
        )
        sys.exit(1)

    archive_path = _archive_and_write(knowledge_path, output)
    print(f"redpill animus: KNOWLEDGE.md written ({len(output)} chars)")
    if archive_path:
        print(f"redpill animus: previous version archived to {archive_path}")


# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------


def _get_delta_cutoff(knowledge_path: Path) -> datetime.date | None:
    """Return mtime of KNOWLEDGE.md as a date, or None on cold start."""
    if not knowledge_path.exists():
        return None
    return datetime.date.fromtimestamp(os.path.getmtime(knowledge_path))


# ---------------------------------------------------------------------------
# DB reads
# ---------------------------------------------------------------------------


def _read_new_articles(
    conn: sqlite3.Connection,
    topic: str,
    since: datetime.date | None,
) -> list[dict]:
    """Query seen_items for articles since cutoff. Capped at 200 rows."""
    if since is not None:
        rows = conn.execute(
            """
            SELECT url, title, summary, dim_id, first_seen_date
            FROM seen_items
            WHERE topic = ?
              AND summary != ''
              AND first_seen_date >= ?
            ORDER BY first_seen_date DESC
            LIMIT 200
            """,
            (topic, since.isoformat()),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT url, title, summary, dim_id, first_seen_date
            FROM seen_items
            WHERE topic = ?
              AND summary != ''
            ORDER BY first_seen_date DESC
            LIMIT 200
            """,
            (topic,),
        ).fetchall()
    return [dict(row) for row in rows]


def _read_dimensions(conn: sqlite3.Connection, topic: str) -> list[dict]:
    """Query dimension_registry for all active non-sentinel dimensions."""
    rows = conn.execute(
        """
        SELECT dim_id, canonical_name, pool, run_count, alpha, beta, last_seen
        FROM dimension_registry
        WHERE topic = ?
          AND pool != 'retired'
          AND dim_id NOT IN ('dim_fallback', 'dim_base')
        ORDER BY run_count DESC
        """,
        (topic,),
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        alpha = d.get("alpha", 1.0) or 1.0
        beta = d.get("beta", 1.0) or 1.0
        d["mean_reward"] = alpha / (alpha + beta)
        result.append(d)
    return result


def _read_dimension_terms(
    conn: sqlite3.Connection, topic: str, dim_id: str
) -> list[str]:
    """Top 15 terms for a specific dimension."""
    return get_top_terms_for_dim_conn(topic, dim_id, 15, conn)


def _read_global_terms(conn: sqlite3.Connection, topic: str) -> list[dict]:
    """Top 20 terms across the whole topic."""
    return get_top_terms_conn(topic, 20, conn)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(
    topic: str,
    existing_knowledge: str | None,
    new_articles: list[dict],
    dimensions: list[dict],
    dimension_terms_map: dict[str, list[str]],
    global_terms: list[dict],
    today: datetime.date,
) -> str:
    """Assemble the full LLM prompt. Uses explicit concatenation — article
    content can contain literal braces which break str.format()."""

    # Derive a dim_id → canonical_name lookup for article formatting.
    dim_names: dict[str, str] = {
        d["dim_id"]: d["canonical_name"] for d in dimensions
    }

    # Date range for new articles header.
    dates = [a["first_seen_date"] for a in new_articles if a.get("first_seen_date")]
    if dates:
        date_range = min(dates) + " to " + max(dates)
    else:
        date_range = "unknown range"

    total_count_note = str(len(new_articles)) + " articles"

    parts = []

    # --- System framing ---
    parts.append(
        'You are a domain analyst maintaining a living knowledge base about "'
        + topic
        + '".\n'
        "Your job is to produce an opinionated, structured synthesis of what has been\n"
        "learned — not a neutral summary. Make directional claims. State what appears\n"
        "to be happening in the field based on the evidence. You are allowed to be wrong.\n"
        "A document that makes specific claims is more useful than one that hedges everything.\n"
    )

    # --- Existing knowledge base ---
    parts.append("\n---\n\n")
    parts.append("EXISTING KNOWLEDGE BASE (your previous synthesis — revise this, do not just append):\n")
    if existing_knowledge:
        parts.append(existing_knowledge)
    else:
        parts.append("None — this is the first synthesis.")
    parts.append("\n")

    # --- New articles ---
    parts.append("\n---\n\n")
    parts.append(
        "NEW ARTICLES SINCE LAST SYNTHESIS ("
        + total_count_note
        + ", "
        + date_range
        + "):\n"
    )
    for a in new_articles:
        dim_id = a.get("dim_id", "")
        canonical = dim_names.get(dim_id, dim_id)
        title = a.get("title") or "(untitled)"
        summary = a.get("summary") or ""
        parts.append("- **" + title + "** [" + canonical + "]\n  " + summary + "\n")

    # --- Active dimensions ---
    parts.append("\n---\n\n")
    parts.append("ACTIVE RESEARCH DIMENSIONS:\n")
    for dim in dimensions:
        terms = dimension_terms_map.get(dim["dim_id"], [])
        terms_str = ", ".join(terms) if terms else "(no terms yet)"
        reward_pct = str(round(dim["mean_reward"] * 100, 1)) + "%"
        parts.append(
            "- **" + dim["canonical_name"] + "**"
            " | pool: " + dim["pool"]
            + " | runs: " + str(dim["run_count"])
            + " | success rate: " + reward_pct
            + "\n  Top terms: " + terms_str + "\n"
        )

    # --- Global top terms ---
    parts.append("\n---\n\n")
    parts.append("DOMINANT TERMS ACROSS TOPIC (all time):\n")
    for t in global_terms:
        freq = t.get("frequency", 0)
        cat = t.get("category", "")
        term = t.get("term", "")
        parts.append("- " + term + " (" + cat + ", freq=" + str(freq) + ")\n")

    # --- Output format specification ---
    parts.append("\n---\n\n")
    parts.append(
        "Produce an updated KNOWLEDGE.md with exactly this structure:\n\n"
        "# Knowledge Base: " + topic + "\n"
        "*Last updated: " + today.isoformat() + "*\n"
        "*Articles synthesised this run: {count} new / {total} total*\n\n"
        "## Overview\n"
        "[3-4 paragraphs. Opinionated assessment of the current state of the field.\n"
        "What is the dominant picture? What approaches appear to be winning?\n"
        "What does the evidence suggest is an active frontier vs. saturated ground?\n"
        "Call out authors or sources that keep appearing if relevant.\n"
        'Make directional claims — "the field appears to be moving toward X",\n'
        '"Y seems to be losing ground to Z".]\n\n'
        "## Dimensions\n\n"
        "### {canonical_name}\n"
        "**Status:** {pool} | **Runs:** {run_count} | **Success rate:** {mean_reward as %}\n"
        "[1 paragraph. What has specifically been learned in this angle.\n"
        "Key findings, recurring themes, what's still open or underexplored within it.\n"
        "If the dimension has low run count, say so — understanding here is thin.]\n\n"
        "[repeat for each active dimension, ordered by run_count descending]\n\n"
        "## Open Questions\n"
        "[Bullet list. What keeps appearing across articles but remains unresolved,\n"
        "contested, or underexplored across dimensions. Forward-looking.]\n\n"
        "## Revision Notes\n"
        "[Brief. What changed from the previous synthesis.\n"
        "On cold start: \"Initial synthesis — no prior knowledge base existed.\"]\n\n"
        "---\n\n"
        "Write only the markdown document. No preamble, no explanation outside the document.\n"
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Validation + file I/O
# ---------------------------------------------------------------------------


def _validate_output(text: str) -> bool:
    """Minimal check: output must contain both required top-level sections."""
    return bool(text.strip()) and "## Overview" in text and "## Dimensions" in text


def _archive_and_write(knowledge_path: Path, new_content: str) -> Path | None:
    """Archive existing KNOWLEDGE.md (if any), then write new content.

    Returns the archive path if an archive was created, else None.
    """
    memory_dir = knowledge_path.parent
    memory_dir.mkdir(parents=True, exist_ok=True)

    archive_path: Path | None = None
    if knowledge_path.exists():
        archive_path = memory_dir / (datetime.date.today().isoformat() + ".md")
        shutil.copy2(knowledge_path, archive_path)
        logger.info("Archived previous KNOWLEDGE.md to %s", archive_path)

    knowledge_path.write_text(new_content, encoding="utf-8")
    logger.info("KNOWLEDGE.md updated (%d chars)", len(new_content))

    return archive_path
