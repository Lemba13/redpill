"""
query_planner.py — Phase 3: generate search queries from term history.

The planner is the feedback loop that makes each pipeline run smarter than
the last. It reads the extracted_terms table, asks the LLM to propose targeted
queries, and falls back to a deterministic term-expansion strategy when the
LLM is unavailable or produces unusable output.

Guarantee: the base topic is ALWAYS the first query regardless of LLM output.

Public API:
    plan_queries(topic, db_path, llm_client, max_queries=5) -> list[dict]
        Full planner: try LLM-planned queries, fall back to deterministic.

    plan_queries_fallback(topic, db_path, max_queries=5) -> list[dict]
        Deterministic fallback: expand top terms into "{topic} {term}" queries.
        Also called when LLM planning is disabled or fails.

Each returned dict has:
    query     — the search query string
    source    — "base" | "extracted_term" | "llm_planned"
    reasoning — human-readable explanation of why this query was chosen
"""

import logging
from typing import TYPE_CHECKING

from redpill.llm_utils import extract_json
from redpill.state import get_recent_terms_conn, get_top_terms_conn

if TYPE_CHECKING:
    import sqlite3

    from redpill.summarize import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research assistant that generates targeted search queries.
You return only valid JSON — no explanation, no markdown, no preamble.
"""

# How many days of term history to show the LLM.
_TERM_HISTORY_DAYS = 30

# How many terms to include in the LLM prompt.
_MAX_TERMS_IN_PROMPT = 20


def _base_query(topic: str) -> dict:
    return {
        "query": topic,
        "source": "base",
        "reasoning": "Base topic query — always included.",
    }


def _build_planner_prompt(topic: str, terms: list[dict], n_queries: int) -> str:
    """Build the LLM prompt that asks for *n_queries* additional queries."""
    if terms:
        term_lines = "\n".join(
            f"  - {t['term']} (category: {t.get('category', 'keyword')}, "
            f"seen {t['frequency']} time(s))"
            for t in terms[:_MAX_TERMS_IN_PROMPT]
        )
        term_section = f"Recently extracted terms from prior runs:\n{term_lines}"
    else:
        term_section = "No term history available yet."

    return f"""\
Topic: {topic}

{term_section}

Generate {n_queries} search queries that would find new, relevant research on \
the topic. Use the term history above to make queries more specific than the \
base topic. Prefer combining the topic with specific techniques, authors, \
datasets, or frameworks. Avoid generic queries that would return the same \
results as the base topic.

Return a JSON array with exactly {n_queries} elements. Each element must have:
  "query"     — the search query string
  "reasoning" — one sentence explaining why this query is useful

Example:
[
  {{"query": "{topic} MoCo contrastive loss", "reasoning": "Targets a specific technique seen frequently in prior runs."}},
  {{"query": "{topic} benchmark ImageNet 2025", "reasoning": "Focuses on recent benchmark results."}}
]
"""


def _parse_llm_queries(raw: str, topic: str, n_expected: int) -> list[dict]:
    """Parse and validate the LLM's query array response.

    Returns a list of validated query dicts (source="llm_planned"). Returns []
    if parsing fails or the output doesn't contain usable queries.
    """
    parsed = extract_json(raw)

    if parsed is None:
        logger.warning("plan_queries: could not parse JSON from LLM response")
        return []

    if not isinstance(parsed, list):
        logger.warning(
            "plan_queries: expected JSON array, got %s", type(parsed).__name__
        )
        return []

    base_lower = topic.lower()
    results: list[dict] = []
    seen_queries: set[str] = {base_lower}

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query", "")
        reasoning = entry.get("reasoning", "")
        if not query or not isinstance(query, str):
            continue
        query = query.strip()
        if not query:
            continue
        # Deduplicate (case-insensitive) and skip if same as base topic.
        query_lower = query.lower()
        if query_lower in seen_queries:
            logger.debug("plan_queries: dropping duplicate query %r", query)
            continue
        seen_queries.add(query_lower)
        results.append(
            {
                "query": query,
                "source": "llm_planned",
                "reasoning": reasoning.strip() if isinstance(reasoning, str) else "",
            }
        )

    logger.debug("plan_queries: LLM produced %d valid query/ies", len(results))
    return results


def plan_queries_fallback(
    topic: str,
    conn: "sqlite3.Connection",
    max_queries: int = 5,
) -> list[dict]:
    """Deterministic fallback: expand top terms into targeted queries.

    Always starts with the base topic query, then adds "{topic} {term}" for
    each of the top terms by frequency until *max_queries* is reached.

    Parameters
    ----------
    topic:
        The research topic from config.
    conn:
        An open SQLite connection (in-memory or file-backed).
    max_queries:
        Maximum number of queries to return (including the base query).

    Returns
    -------
    A list of query dicts with ``source`` values of "base" or "extracted_term".
    """
    queries = [_base_query(topic)]

    if max_queries <= 1:
        return queries

    n_extra = max_queries - 1
    top_terms = get_top_terms_conn(topic, limit=n_extra, conn=conn)

    for term_dict in top_terms:
        term = term_dict.get("term", "")
        if not term:
            continue
        queries.append(
            {
                "query": f"{topic} {term}",
                "source": "extracted_term",
                "reasoning": (
                    f"High-frequency term '{term}' "
                    f"(seen {term_dict.get('frequency', 1)} time(s))."
                ),
            }
        )

    logger.info(
        "plan_queries_fallback: %d query/ies (%d term-expanded)",
        len(queries),
        len(queries) - 1,
    )
    return queries


def plan_queries(
    topic: str,
    conn: "sqlite3.Connection",
    llm_client: "LLMClient",
    max_queries: int = 5,
) -> list[dict]:
    """Generate search queries using term history and LLM planning.

    Strategy:
    1. Base topic query is always first.
    2. Fetch recent term history from the DB.
    3. Ask the LLM to propose (max_queries - 1) additional queries.
    4. If the LLM returns parseable, non-duplicate queries, use them.
    5. Fall back to plan_queries_fallback on any LLM failure.

    Parameters
    ----------
    topic:
        The research topic from config.
    conn:
        An open SQLite connection.
    llm_client:
        Any LLMClient-compatible object (used for LLM planning).
    max_queries:
        Maximum number of queries to return (including the base query).

    Returns
    -------
    A list of query dicts, each with keys: query, source, reasoning.
    The first element is always the base topic query.
    """
    if max_queries <= 1:
        return [_base_query(topic)]

    n_extra = max_queries - 1

    # Fetch term history for the prompt.
    recent_terms = get_recent_terms_conn(topic, _TERM_HISTORY_DAYS, conn)

    if not recent_terms:
        logger.info(
            "plan_queries: no term history — using fallback for topic %r", topic
        )
        return plan_queries_fallback(topic, conn, max_queries)

    prompt = _build_planner_prompt(topic, recent_terms, n_extra)

    try:
        raw = llm_client.generate(prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning(
            "plan_queries: LLM call failed (%s) — using fallback", exc
        )
        return plan_queries_fallback(topic, conn, max_queries)

    llm_queries = _parse_llm_queries(raw, topic, n_extra)

    if not llm_queries:
        logger.warning("plan_queries: LLM returned no usable queries — using fallback")
        return plan_queries_fallback(topic, conn, max_queries)

    # Cap at n_extra and prepend the base query.
    result = [_base_query(topic)] + llm_queries[:n_extra]
    logger.info(
        "plan_queries: %d query/ies planned (%d LLM, 1 base)",
        len(result),
        len(result) - 1,
    )
    return result
