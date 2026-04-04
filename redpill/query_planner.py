"""
query_planner.py — Autonomous research planning via two-stage LLM decomposition.

Architecture
------------
The planner has two distinct paths:

  Two-stage path (PlannerLLMClient with think=True):
    Stage 1 — decompose_topic(): reasoning LLM produces a structured research
              plan with dimensions, priorities, and coverage assessments.
    Stage 2 — synthesize_queries(): deterministic conversion of the plan
              into concrete search queries, prioritized by dimension priority
              and coverage gap.
    These are orchestrated by plan_queries() which also saves the plan to DB.

  Single-stage path (standard LLMClient):
    The LLM is prompted directly for a JSON array of query strings.
    Used when a PlannerLLMClient is not available (e.g. no reasoning model).

  Deterministic fallback (no LLM):
    plan_queries_fallback() expands the top extracted terms from prior runs
    into "{topic} {term}" queries.  Always succeeds.

Guarantee: the base topic is ALWAYS the first query regardless of LLM output.

Public API:
    decompose_topic(topic, conn, planner_llm, max_dimensions) -> dict
        Stage 1: produce a structured research plan using the reasoning LLM.

    synthesize_queries(plan, topic, max_queries) -> list[dict]
        Stage 2: deterministically convert a research plan to search queries.

    plan_queries(topic, conn, llm_client, max_queries) -> list[dict]
        Orchestrator: two-stage → single-stage → fallback, in priority order.

    plan_queries_fallback(topic, conn, max_queries) -> list[dict]
        Deterministic fallback: expand top terms into "{topic} {term}" queries.

Each returned query dict has:
    query     — the search query string
    source    — "base" | "extracted_term" | "llm_planned" | "fallback"
    reasoning — human-readable explanation of why this query was chosen
"""

import hashlib
import json
import logging
from datetime import date as _date
from typing import TYPE_CHECKING

from redpill.llm_utils import extract_json
from redpill.state import (
    get_query_performance_conn,
    get_recent_terms_conn,
    get_top_terms_conn,
    save_research_plan_conn,
)

if TYPE_CHECKING:
    import sqlite3

    from redpill.summarize import LLMClient, PlannerLLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research assistant that generates targeted search queries.
You return only valid JSON — no explanation, no markdown, no preamble.
"""

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a research strategist. You return only valid JSON — no explanation,
no markdown, no preamble.
"""

# How many days of term history to show the LLM.
_TERM_HISTORY_DAYS = 30

# How many terms to include in the LLM prompt.
_MAX_TERMS_IN_PROMPT = 20

# How many days of query performance history to include in the planner prompt.
_QUERY_PERF_HISTORY_DAYS = 14


def _compute_dim_id(canonical_name: str, topic: str) -> str:
    """Return a stable dim_id for a named dimension within a topic.

    Format: "dim_" + first 12 hex chars of SHA-256(canonical_name + topic).
    Deterministic across runs — two runs with the same dimension name and topic
    produce the same dim_id, allowing reward history to accumulate.
    """
    raw = (canonical_name + topic).encode("utf-8")
    return "dim_" + hashlib.sha256(raw).hexdigest()[:12]


def _base_query(topic: str) -> dict:
    return {
        "query": topic,
        "source": "base",
        "reasoning": "Base topic query — always included.",
        "dim_id": "dim_base",
    }


def _build_planner_prompt(topic: str, terms: list[dict], n_queries: int) -> str:
    """Build the single-stage LLM prompt that asks for *n_queries* additional queries."""
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


def _format_feedback_section(signals: dict) -> str:
    """Render the feedback signals section for the decomposition prompt.

    Called only when ``signals["has_feedback"]`` is True and
    ``signals["vote_count"]`` meets the caller's minimum threshold.
    """
    period = signals.get("period_days", 30)
    vote_count = signals.get("vote_count", 0)
    engagement = signals.get("engagement", {})
    engagement_pct = round(engagement.get("engagement_rate", 0.0) * 100, 1)

    lines: list[str] = [
        f"\nUser feedback signals (last {period} days, {vote_count} votes):\n"
    ]

    dim_prefs: list[dict] = signals.get("dimension_preferences", [])
    if dim_prefs:
        lines.append("Dimension preferences (approval rate from user votes):")
        for dp in dim_prefs:
            lines.append(
                f"  - {dp['dimension']!r}: "
                f"{dp['up']} up, {dp['down']} down "
                f"(approval: {dp['approval']:.0%}, shown: {dp['shown']})"
            )

    src_prefs: list[dict] = signals.get("source_preferences", [])
    if src_prefs:
        lines.append("\nSource preferences:")
        for sp in src_prefs:
            lines.append(
                f"  - {sp['domain']}: "
                f"{sp['up']} up, {sp['down']} down "
                f"(approval: {sp['approval']:.0%}, shown: {sp['shown']})"
            )

    lines.append(f"\nOverall engagement rate: {engagement_pct}% of delivered items received a vote.")

    term_sentiment: list[dict] | None = signals.get("term_sentiment")
    if term_sentiment:
        lines.append("\nTerm sentiment (from source queries of voted items):")
        for ts in term_sentiment[:10]:
            lines.append(
                f"  - {ts['term']!r}: {ts['sentiment']} "
                f"({ts['up']} up, {ts['down']} down)"
            )

    lines.append(
        "\nUse these signals to adjust dimension priorities:\n"
        "- High-approval dimensions should get more query budget\n"
        "- Low-approval dimensions should be deprioritized (but not eliminated "
        "— keep at least one exploratory query)\n"
        "- Prefer sources the user engages with when multiple options exist\n"
        "- Weight these signals proportionally to the engagement rate — a 5% "
        "engagement rate means weak signal, treat cautiously. A 50%+ rate means "
        "strong signal, lean into it.\n"
        "- IMPORTANT: Always keep at least one exploratory dimension regardless "
        "of feedback. Do not create a filter bubble."
    )
    return "\n".join(lines)


def _build_decompose_prompt(
    topic: str,
    today: str,
    previous_plan: dict | None,
    recent_terms: list[dict],
    top_terms: list[dict],
    query_perf: list[dict],
    max_dimensions: int,
    feedback_signals: dict | None = None,
    min_votes_for_signals: int = 5,
) -> str:
    """Build the two-stage decomposition prompt for the reasoning LLM.

    This is the most important prompt in the system. It gives the reasoning
    model all available context and asks it to produce a structured research
    plan with dimensions, priorities, and coverage assessments.

    Parameters
    ----------
    feedback_signals:
        When provided and ``signals["has_feedback"]`` is True and vote_count
        meets *min_votes_for_signals*, a formatted feedback section is
        appended to the prompt.
    min_votes_for_signals:
        Minimum total votes before feedback signals are included.
    """
    # Format previous plan
    if previous_plan:
        prev_plan_text = json.dumps(previous_plan, indent=2)
    else:
        prev_plan_text = "No previous plan — this is the first run."

    # Format recent terms
    if recent_terms:
        recent_terms_text = "\n".join(
            f"  - {t['term']} (category: {t.get('category', 'keyword')}, "
            f"seen {t['frequency']} time(s), last seen: {t['last_seen']})"
            for t in recent_terms[:_MAX_TERMS_IN_PROMPT]
        )
    else:
        recent_terms_text = "  (none yet)"

    # Format top all-time terms
    if top_terms:
        top_terms_text = ", ".join(
            f"{t['term']} ({t['frequency']}x)" for t in top_terms[:10]
        )
    else:
        top_terms_text = "(none yet)"

    # Format query performance — show which queries found content vs which found nothing
    if query_perf:
        perf_lines = []
        for qp in query_perf[:10]:
            kept = qp.get("kept_items", 0)
            status = "productive" if kept > 0 else "unproductive (0 kept)"
            perf_lines.append(
                f"  - {qp['query_text']!r}: {kept} kept item(s) — {status}"
            )
        query_perf_text = "\n".join(perf_lines)
    else:
        query_perf_text = "  (no query history yet)"

    base_prompt = f"""\
You are a research strategist planning the next search run for a daily research digest.

Topic: {topic}
Today's date: {today}

Previous research plan (from last run):
{prev_plan_text}

Domain terms extracted from recent articles (last 14 days):
{recent_terms_text}

All-time top terms (by frequency):
  {top_terms_text}

Query performance from recent runs:
{query_perf_text}
"""

    # Append feedback signals when available and meaningful.
    if (
        feedback_signals is not None
        and feedback_signals.get("has_feedback")
        and feedback_signals.get("vote_count", 0) >= min_votes_for_signals
    ):
        base_prompt += _format_feedback_section(feedback_signals)
    else:
        base_prompt += (
            "\nNo user feedback available yet. "
            "Base priorities on topic analysis and query performance only.\n"
        )

    base_prompt += f"""
Your task: Produce an updated research plan with {max_dimensions} dimensions \
(subtopics/angles) to investigate.

For each dimension:
- "name": short label (3-6 words)
- "description": one sentence explaining what this covers
- "priority": "high" | "medium" | "low" — based on how much new content is likely to exist
- "coverage": "under-explored" | "partially-covered" | "well-covered" — based on what prior runs have already found
- "suggested_queries": list of 1-2 realistic web search strings (3-8 words each, the kind you'd type into Google)

Also include:
- "dropped_dimensions": list of areas from the previous plan you are removing, each with a "name" and "reason"
- "new_directions": list of strings — any emerging angles suggested by recent findings not in the previous plan

Rules:
- Always keep one dimension for the base topic itself (broad catch-all)
- Always keep at least one "exploratory" dimension probing an adjacent area
- Deprioritize dimensions where recent queries found zero new content
- Prioritize dimensions where extracted terms are trending (appearing with increasing frequency)
- Search queries should be realistic web search strings, not academic jargon dumps
- Good query: "contrastive learning hard negatives 2026" — specific, likely to find content
- Bad query: "contrastive self-supervised representation learning methodologies" — too dense

Respond with valid JSON only. No preamble, no markdown, no explanation outside the JSON.
The top-level JSON object must have exactly these keys: "dimensions", "dropped_dimensions", "new_directions".
"""
    return base_prompt


def _parse_llm_queries(raw: str, topic: str, n_expected: int) -> list[dict]:
    """Parse and validate the single-stage LLM's query array response.

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
                "dim_id": "dim_fallback",
            }
        )

    logger.debug("plan_queries: LLM produced %d valid query/ies", len(results))
    return results


def _parse_research_plan(raw: str) -> dict | None:
    """Parse the research plan JSON from the reasoning LLM's response.

    Returns the plan dict if parsing succeeds and the structure is valid,
    or None on any failure. A valid plan must have a "dimensions" key
    containing a list with at least one element.
    """
    parsed = extract_json(raw)

    if parsed is None:
        logger.warning("decompose_topic: could not parse JSON from planner LLM response")
        return None

    if not isinstance(parsed, dict):
        logger.warning(
            "decompose_topic: expected JSON object, got %s", type(parsed).__name__
        )
        return None

    dimensions = parsed.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        logger.warning(
            "decompose_topic: 'dimensions' must be a non-empty list, got: %r", dimensions
        )
        return None

    return parsed


def decompose_topic(
    topic: str,
    conn: "sqlite3.Connection",
    planner_llm: "PlannerLLMClient",
    max_dimensions: int = 6,
    feedback_signals: "dict | None" = None,
    min_votes_for_signals: int = 5,
) -> dict:
    """Stage 1: use the reasoning LLM to produce a structured research plan.

    Gathers all available context from the DB (previous plan, recent terms,
    top terms, query performance) and feeds it to the reasoning model. The
    model produces a plan with dimensions, priorities, and coverage assessments.

    Parameters
    ----------
    topic:
        The research topic from config.
    conn:
        An open SQLite connection with all tables initialized.
    planner_llm:
        A PlannerLLMClient instance (reasoning model with think=True).
    max_dimensions:
        Maximum number of research dimensions to include in the plan.
    feedback_signals:
        Optional dict from FeedbackReader.compute_preference_signals().
        When provided and has_feedback is True with enough votes, user
        preference signals are included in the planning prompt.
    min_votes_for_signals:
        Minimum vote count before feedback signals are included.

    Returns
    -------
    The validated research plan dict.

    Raises
    ------
    RuntimeError
        If the LLM call fails or returns an unparseable/invalid plan.
        Callers should catch this and fall back to plan_queries_fallback().
    """
    today = _date.today().isoformat()

    # Gather context from DB.
    recent_terms = get_recent_terms_conn(topic, 14, conn)
    top_terms = get_top_terms_conn(topic, 20, conn)
    query_perf = get_query_performance_conn(topic, _QUERY_PERF_HISTORY_DAYS, conn)

    # Load previous plan if available.
    from redpill.state import get_latest_research_plan_conn
    prev_plan_row = get_latest_research_plan_conn(topic, conn)
    previous_plan: dict | None = None
    if prev_plan_row:
        try:
            previous_plan = json.loads(prev_plan_row["plan_json"])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("decompose_topic: could not load previous plan: %s", exc)

    prompt = _build_decompose_prompt(
        topic=topic,
        today=today,
        previous_plan=previous_plan,
        recent_terms=recent_terms,
        top_terms=top_terms,
        query_perf=query_perf,
        max_dimensions=max_dimensions,
        feedback_signals=feedback_signals,
        min_votes_for_signals=min_votes_for_signals,
    )

    logger.info(
        "decompose_topic: calling reasoning LLM (model=%r) for topic %r ...",
        getattr(planner_llm, "_model", "unknown"),
        topic,
    )

    raw = planner_llm.generate(prompt, system=_DECOMPOSE_SYSTEM_PROMPT)

    plan = _parse_research_plan(raw)
    if plan is None:
        raise RuntimeError(
            "decompose_topic: reasoning LLM returned an invalid research plan"
        )

    logger.info(
        "decompose_topic: plan has %d dimension(s) for topic %r",
        len(plan.get("dimensions", [])),
        topic,
    )
    return plan


def synthesize_queries(
    plan: dict,
    topic: str,
    max_queries: int = 5,
) -> list[dict]:
    """Stage 2: deterministically convert a research plan to search queries.

    Priority ordering for query selection:
    1. Base topic anchor (always first, from outside this function).
    2. High priority + under-explored dimensions (first choice).
    3. High priority + partially-covered dimensions.
    4. Medium priority + under-explored dimensions.
    5. Medium priority + partially-covered dimensions.
    6. At least one query from new_directions if present and budget remains.
    7. Any remaining dimensions in natural order.

    Dimensions with coverage="well-covered" and priority="low" are included
    last, only if query budget allows.

    Each dimension contributes its suggested_queries until the budget is filled.
    Duplicate queries (case-insensitive) are dropped.

    Parameters
    ----------
    plan:
        The research plan dict produced by decompose_topic(). Must have a
        "dimensions" key with a list of dimension dicts.
    topic:
        The base research topic — used to detect and de-duplicate the anchor.
    max_queries:
        Total query budget including the base topic anchor.

    Returns
    -------
    A list of query dicts (source="llm_planned"). The base topic anchor is
    NOT included — callers prepend it. Returns [] if the plan has no usable
    suggested_queries.
    """
    budget = max_queries - 1  # base topic is added by the caller
    if budget <= 0:
        return []

    dimensions: list[dict] = plan.get("dimensions", [])
    new_directions: list[str] = plan.get("new_directions", [])

    # Priority ordering: (priority_rank, coverage_rank)
    # Lower rank numbers = higher precedence.
    _PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
    _COVERAGE_RANK = {"under-explored": 0, "partially-covered": 1, "well-covered": 2}

    def _dim_sort_key(dim: dict) -> tuple[int, int]:
        p = _PRIORITY_RANK.get(dim.get("priority", "low"), 2)
        c = _COVERAGE_RANK.get(dim.get("coverage", "well-covered"), 2)
        return (p, c)

    sorted_dims = sorted(dimensions, key=_dim_sort_key)

    seen: set[str] = {topic.lower()}  # prevent base topic re-inclusion
    results: list[dict] = []

    def _try_add(query: str, reasoning: str, dim_id: str) -> bool:
        """Attempt to add a query; return True if added, False if duplicate/budget."""
        q = query.strip()
        if not q:
            return False
        if q.lower() in seen:
            return False
        if len(results) >= budget:
            return False
        seen.add(q.lower())
        results.append({
            "query": q,
            "source": "llm_planned",
            "reasoning": reasoning,
            "dim_id": dim_id,
        })
        return True

    # Add queries from sorted dimensions.
    for dim in sorted_dims:
        if len(results) >= budget:
            break
        name = dim.get("name", "")
        priority = dim.get("priority", "medium")
        coverage = dim.get("coverage", "partially-covered")
        suggested = dim.get("suggested_queries", [])
        if not isinstance(suggested, list):
            continue
        d_id = _compute_dim_id(name, topic)
        for sq in suggested:
            if not isinstance(sq, str):
                continue
            reasoning = (
                f"Dimension '{name}': priority={priority}, coverage={coverage}."
            )
            _try_add(sq, reasoning, d_id)
            if len(results) >= budget:
                break

    # Add at least one new_directions query if budget allows.
    for direction in new_directions:
        if len(results) >= budget:
            break
        if not isinstance(direction, str) or not direction.strip():
            continue
        # new_directions are narrative strings, not raw query strings.
        # Extract the first quoted phrase or use the direction as-is (truncated).
        import re as _re
        m = _re.search(r'"([^"]{5,60})"', direction)
        if m:
            query_candidate = m.group(1)
        else:
            # Use the first 60 chars as a rough query hint.
            query_candidate = direction.strip()[:60]
        _try_add(
            query_candidate,
            f"New direction from research plan: {direction[:80]}",
            "dim_fallback",
        )

    logger.debug(
        "synthesize_queries: %d query/ies from plan for topic %r",
        len(results), topic,
    )
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
                "dim_id": "dim_fallback",
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
    feedback_signals: "dict | None" = None,
    min_votes_for_signals: int = 5,
) -> list[dict]:
    """Generate search queries using the best available strategy.

    Strategy (in priority order):

    1. Two-stage path (PlannerLLMClient):
       If *llm_client* is a PlannerLLMClient, call decompose_topic() to get a
       structured research plan, save it to DB, then synthesize_queries() to
       extract query strings.  The reasoning trace is saved alongside the plan.

    2. Single-stage path (standard LLMClient):
       If term history exists, ask the LLM for a JSON array of query strings
       directly. This is the v2 path — used when a PlannerLLMClient is not
       available.

    3. Deterministic fallback:
       Called when no term history exists, when the LLM fails, or when the
       LLM produces unusable output. Also saves a minimal fallback plan to DB.

    Guarantee: the base topic is always the first query.

    Parameters
    ----------
    topic:
        The research topic from config.
    conn:
        An open SQLite connection with all tables initialized.
    llm_client:
        A PlannerLLMClient (two-stage) or any LLMClient (single-stage).
    max_queries:
        Maximum number of queries to return (including the base query).
    feedback_signals:
        Optional preference signals from FeedbackReader.compute_preference_signals().
        Passed through to decompose_topic() for the two-stage path only.
    min_votes_for_signals:
        Minimum vote count before feedback signals affect the planning prompt.

    Returns
    -------
    A list of query dicts, each with keys: query, source, reasoning.
    The first element is always the base topic query.
    """
    from redpill.summarize import PlannerLLMClient as _PlannerLLMClient

    if max_queries <= 1:
        return [_base_query(topic)]

    today = _date.today().isoformat()


    # ------------------------------------------------------------------
    # Path 1: Two-stage decomposition (PlannerLLMClient with think=True)
    # ------------------------------------------------------------------
    if isinstance(llm_client, _PlannerLLMClient):
        try:
            plan = decompose_topic(
                topic,
                conn,
                llm_client,
                feedback_signals=feedback_signals,
                min_votes_for_signals=min_votes_for_signals,
            )
        except Exception as exc:
            logger.warning(
                "plan_queries: decompose_topic failed (%s) — falling back to single-stage",
                exc,
            )
        else:
            # Save the plan and its reasoning trace.
            reasoning_trace = getattr(llm_client, "last_thinking", None)
            try:
                save_research_plan_conn(
                    topic=topic,
                    run_date=today,
                    plan=plan,
                    conn=conn,
                    reasoning_trace=reasoning_trace,
                    source="llm",
                )
            except Exception as exc:
                logger.warning(
                    "plan_queries: failed to save research plan: %s", exc
                )

            synth = synthesize_queries(plan, topic, max_queries)
            if synth:
                result = [_base_query(topic)] + synth
                logger.info(
                    "plan_queries (two-stage): %d query/ies for topic %r",
                    len(result), topic,
                )
                return result

            logger.warning(
                "plan_queries: synthesize_queries produced no queries — "
                "falling back to single-stage"
            )

    # ------------------------------------------------------------------
    # Path 2: Single-stage LLM (standard LLMClient)
    # ------------------------------------------------------------------
    n_extra = max_queries - 1
    recent_terms = get_recent_terms_conn(topic, _TERM_HISTORY_DAYS, conn)

    if not recent_terms:
        logger.info(
            "plan_queries: no term history — using deterministic fallback for topic %r",
            topic,
        )
        _save_fallback_plan(topic, today, conn)
        return plan_queries_fallback(topic, conn, max_queries)

    prompt = _build_planner_prompt(topic, recent_terms, n_extra)

    try:
        raw = llm_client.generate(prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning(
            "plan_queries: LLM call failed (%s) — using deterministic fallback", exc
        )
        _save_fallback_plan(topic, today, conn)
        return plan_queries_fallback(topic, conn, max_queries)

    llm_queries = _parse_llm_queries(raw, topic, n_extra)

    if not llm_queries:
        logger.warning(
            "plan_queries: LLM returned no usable queries — using deterministic fallback"
        )
        _save_fallback_plan(topic, today, conn)
        return plan_queries_fallback(topic, conn, max_queries)

    # Cap at n_extra and prepend the base query.
    result = [_base_query(topic)] + llm_queries[:n_extra]
    logger.info(
        "plan_queries (single-stage): %d query/ies planned (%d LLM, 1 base)",
        len(result),
        len(result) - 1,
    )
    return result


def _save_fallback_plan(
    topic: str,
    run_date: str,
    conn: "sqlite3.Connection",
) -> None:
    """Persist a minimal fallback plan record to the research_plans table.

    Called when the LLM path fails so there is still a record of what happened
    on this run. Best-effort — exceptions are logged and suppressed.
    """
    fallback_plan = {
        "dimensions": [
            {
                "name": topic,
                "description": "Base topic — deterministic fallback used.",
                "priority": "high",
                "coverage": "under-explored",
                "suggested_queries": [topic],
            }
        ],
        "dropped_dimensions": [],
        "new_directions": [],
    }
    try:
        save_research_plan_conn(
            topic=topic,
            run_date=run_date,
            plan=fallback_plan,
            conn=conn,
            reasoning_trace=None,
            source="fallback",
        )
    except Exception as exc:
        logger.debug("_save_fallback_plan: could not save fallback plan: %s", exc)
