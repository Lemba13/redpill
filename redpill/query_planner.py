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

import json
import logging
from datetime import date as _date
from typing import TYPE_CHECKING

from redpill.llm_utils import extract_json
from redpill.registry import compute_dim_id, get_all_registry_dims_for_prompt
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
    registry_dims: list[dict] | None = None,
    scaffold: dict | None = None,
    registry_size: int = 0,
    scaffold_registry_min_size: int = 5,
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
    registry_dims:
        All registered dimensions for this topic with axis annotations.
        Used to build the coverage map section.
    scaffold:
        Topic coverage scaffold from generate_topic_scaffold(). Used to
        list known families when the registry is still sparse.
    registry_size:
        Number of non-system rows in dimension_registry. Controls whether
        to show the scaffold (sparse) or the full coverage map (populated).
    scaffold_registry_min_size:
        Registry must have at least this many entries before the full
        coverage map replaces the scaffold in the prompt.
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

    # ------------------------------------------------------------------
    # Section 1 — Intent framing (prepended before everything)
    # ------------------------------------------------------------------
    intent_framing = (
        "Your goal is to generate research dimensions that MAXIMIZE COVERAGE of the\n"
        'topic "' + topic + '", not to deepen what is already well-covered.\n\n'
        "A research dimension defines a specific angle, community, or application area\n"
        "to search for new content. The system has a known failure mode: it repeatedly\n"
        "generates dimensions that are semantically similar to what it has already\n"
        "explored, creating an echo chamber where the same research communities are\n"
        "surfaced repeatedly.\n\n"
        "Your job is to identify what is MISSING from the current coverage and generate\n"
        "dimensions in those gaps.\n\n"
    )

    # ------------------------------------------------------------------
    # Section 2 — Coverage map or scaffold (after intent, before body)
    # ------------------------------------------------------------------
    if registry_size < scaffold_registry_min_size:
        # Registry is still sparse — show the scaffold families instead.
        if scaffold:
            scaffold_lines: list[str] = [
                "## Topic coverage scaffold (registry is still sparse)\n",
                "The following major families exist within this topic. Ensure your\n"
                "dimensions span multiple families rather than clustering in one area:\n",
            ]
            for axis, entries in scaffold.items():
                if entries:
                    scaffold_lines.append(f"  {axis}:")
                    for entry in entries[:6]:
                        scaffold_lines.append(f"    - {entry}")
            coverage_section = "\n".join(scaffold_lines) + "\n\n"
        else:
            coverage_section = ""
    else:
        # Registry is populated — show the full annotated coverage map.
        if registry_dims:
            dim_lines: list[str] = []
            for d in registry_dims:
                tags_str = ", ".join(d.get("tags", [])[:3]) or "—"
                dim_lines.append(
                    f'  - "{d["canonical_name"]}" '
                    f'[axis: {d["primary_axis"]}] '
                    f'[coverage: {d["coverage"]}] '
                    f'[terms: {tags_str}]'
                )

            # Summarise which axes are well-covered vs sparse.
            axis_counts: dict[str, int] = {}
            for d in registry_dims:
                ax = d.get("primary_axis", "unknown")
                axis_counts[ax] = axis_counts.get(ax, 0) + 1
            well_covered_axes = [ax for ax, cnt in axis_counts.items() if cnt >= 2]

            sparse_axes: list[str] = []
            if scaffold:
                for ax in scaffold:
                    if axis_counts.get(ax, 0) < 2:
                        sparse_axes.append(ax)

            coverage_section = (
                "## Current coverage map\n\n"
                "The following dimensions have already been explored. Do NOT generate\n"
                "dimensions that are semantically similar to these.\n\n"
                + "\n".join(dim_lines) + "\n\n"
            )
            if well_covered_axes:
                coverage_section += (
                    "Coverage summary:\n"
                    "  Well-covered axes: " + ", ".join(well_covered_axes) + "\n"
                )
            if sparse_axes:
                coverage_section += (
                    "  Sparse or absent axes: " + ", ".join(sparse_axes) + "\n"
                )
            coverage_section += "\n"
        else:
            coverage_section = ""

    # ------------------------------------------------------------------
    # Body — existing context sections
    # ------------------------------------------------------------------
    body = (
        "You are a research strategist planning the next search run for a daily research digest.\n\n"
        "Topic: " + topic + "\n"
        "Today's date: " + today + "\n\n"
        "Previous research plan (from last run):\n"
        + prev_plan_text + "\n\n"
        "Domain terms extracted from recent articles (last 14 days):\n"
        + recent_terms_text + "\n\n"
        "All-time top terms (by frequency):\n"
        "  " + top_terms_text + "\n\n"
        "Query performance from recent runs:\n"
        + query_perf_text + "\n"
    )

    # Append feedback signals when available and meaningful.
    if (
        feedback_signals is not None
        and feedback_signals.get("has_feedback")
        and feedback_signals.get("vote_count", 0) >= min_votes_for_signals
    ):
        body += _format_feedback_section(feedback_signals)
    else:
        body += (
            "\nNo user feedback available yet. "
            "Base priorities on topic analysis and query performance only.\n"
        )

    # ------------------------------------------------------------------
    # Section 3 — Gap analysis instruction (before the task)
    # ------------------------------------------------------------------
    gap_analysis = (
        "\n## Your task: gap analysis first, then generate\n\n"
        "Before proposing any dimensions, reason through the following questions:\n"
        "1. Which methodological families from the scaffold are absent or sparse in\n"
        "   the current coverage map?\n"
        "2. Which application domains or research communities are unrepresented?\n"
        "3. Which evaluation paradigms or theoretical angles have not been explored?\n"
        "4. What adjacent fields apply similar techniques to different problems?\n\n"
        "Write your gap analysis in your thinking. Only then propose dimensions.\n\n"
    )

    # ------------------------------------------------------------------
    # Section 4 — Task + constrained generation
    # ------------------------------------------------------------------
    min_axes = min(3, max_dimensions)
    task = (
        f"Your task: Produce an updated research plan with {max_dimensions} dimensions "
        "(subtopics/angles) to investigate.\n\n"
        "For each dimension:\n"
        '- "name": short label (3-6 words)\n'
        '- "description": one sentence explaining what this covers\n'
        '- "priority": "high" | "medium" | "low" — based on how much new content is likely to exist\n'
        '- "coverage": "under-explored" | "partially-covered" | "well-covered" — based on what prior runs have already found\n'
        '- "type": "orthogonal" | "adjacent" — orthogonal means an axis absent from the coverage map; adjacent means a new angle on an existing axis\n'
        '- "suggested_queries": list of 1-2 realistic web search strings (3-8 words each, the kind you\'d type into Google)\n\n'
        "Also include:\n"
        '- "dropped_dimensions": list of areas from the previous plan you are removing, each with a "name" and "reason"\n'
        '- "new_directions": list of strings — any emerging angles suggested by recent findings not in the previous plan\n\n'
        "Rules:\n"
        "- Always keep one dimension for the base topic itself (broad catch-all)\n"
        "- Always keep at least one \"exploratory\" dimension probing an adjacent area\n"
        "- Deprioritize dimensions where recent queries found zero new content\n"
        "- Prioritize dimensions where extracted terms are trending (appearing with increasing frequency)\n"
        "- Search queries should be realistic web search strings, not academic jargon dumps\n"
        '- Good query: "contrastive learning hard negatives 2026" — specific, likely to find content\n'
        '- Bad query: "contrastive self-supervised representation learning methodologies" — too dense\n\n'
        "Additional constraints — ALL must be satisfied:\n"
        "1. Each dimension must be semantically distinct from every dimension in the coverage map above\n"
        f"2. At least 1 dimension must be ORTHOGONAL — covering an axis entirely absent from the coverage map\n"
        "3. At most 2 dimensions may be ADJACENT — a new angle on an existing axis\n"
        f"4. Dimensions must span at least {min_axes} distinct research axes\n\n"
        "Respond with valid JSON only. No preamble, no markdown, no explanation outside the JSON.\n"
        'The top-level JSON object must have exactly these keys: "dimensions", "dropped_dimensions", "new_directions".\n'
    )

    return intent_framing + coverage_section + body + gap_analysis + task


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
    scaffold_registry_min_size: int = 5,
) -> dict:
    """Stage 1: use the reasoning LLM to produce a structured research plan.

    Gathers all available context from the DB (previous plan, recent terms,
    top terms, query performance, registry coverage map, topic scaffold) and
    feeds it to the reasoning model. The model produces a plan with dimensions,
    priorities, coverage assessments, and orthogonal/adjacent type labels.

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
    scaffold_registry_min_size:
        Registry must have at least this many entries before showing the
        full coverage map instead of the scaffold.

    Returns
    -------
    The validated research plan dict.

    Raises
    ------
    RuntimeError
        If the LLM call fails or returns an unparseable/invalid plan.
        Callers should catch this and fall back to plan_queries_fallback().
    """
    from redpill.registry import generate_topic_scaffold, get_registry_size
    from redpill.state import get_latest_research_plan_conn

    today = _date.today().isoformat()

    # Gather context from DB.
    recent_terms = get_recent_terms_conn(topic, 14, conn)
    top_terms = get_top_terms_conn(topic, 20, conn)
    query_perf = get_query_performance_conn(topic, _QUERY_PERF_HISTORY_DAYS, conn)

    # Load previous plan if available.
    prev_plan_row = get_latest_research_plan_conn(topic, conn)
    previous_plan: dict | None = None
    if prev_plan_row:
        try:
            previous_plan = json.loads(prev_plan_row["plan_json"])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("decompose_topic: could not load previous plan: %s", exc)

    # Gather registry context for the coverage map / scaffold sections.
    registry_size = get_registry_size(conn)
    registry_dims = get_all_registry_dims_for_prompt(topic, conn)

    scaffold: dict | None = None
    try:
        scaffold = generate_topic_scaffold(topic, planner_llm, conn)
    except Exception as exc:
        logger.warning(
            "decompose_topic: scaffold generation failed (%s) — proceeding without it",
            exc,
        )

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
        registry_dims=registry_dims,
        scaffold=scaffold,
        registry_size=registry_size,
        scaffold_registry_min_size=scaffold_registry_min_size,
    )

    logger.info(
        "decompose_topic: calling reasoning LLM (model=%r) for topic %r "
        "(registry_size=%d, scaffold=%s) ...",
        getattr(planner_llm, "_model", "unknown"),
        topic,
        registry_size,
        "yes" if scaffold else "no",
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
        d_id = dim.get("dim_id") or compute_dim_id(name, topic)
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


def _ensure_topic_embedding(
    topic: str,
    llm_client: "PlannerLLMClient",
    conn: "sqlite3.Connection",
) -> None:
    """Generate and store a topic HyDE embedding if not already present."""
    from redpill.registry import embed_hyde_abstract, generate_hyde_abstract
    from redpill.state import get_topic_embedding_conn, store_topic_embedding_conn

    try:
        if get_topic_embedding_conn(topic, conn) is not None:
            return
        abstract = generate_hyde_abstract(topic, topic, llm_client)
        embedding = embed_hyde_abstract(abstract)
        store_topic_embedding_conn(topic, embedding, conn)
        logger.debug("_ensure_topic_embedding: stored embedding for topic=%r", topic)
    except Exception as exc:
        logger.warning("_ensure_topic_embedding: failed for topic=%r: %s", topic, exc)


def plan_queries(
    topic: str,
    conn: "sqlite3.Connection",
    llm_client: "LLMClient",
    max_queries: int = 5,
    feedback_signals: "dict | None" = None,
    min_votes_for_signals: int = 5,
    registry_resolution_threshold: float = 0.88,
    hyde_abstracts_per_dim: int = 3,
    scaffold_registry_min_size: int = 5,
    ucb_alpha: float = 1.0,
    promotion_k: int = 3,
    mmr_lambda_floor: float = 0.3,
    saturation_decay_days: int = 7,
    saturation_penalty_weight: float = 0.3,
) -> list[dict]:
    """Generate search queries using the best available strategy.

    Strategy (in priority order):

    1. Two-stage path (PlannerLLMClient):
       If *llm_client* is a PlannerLLMClient, call decompose_topic() to get a
       structured research plan, save it to DB, resolve each dimension against
       the registry, then synthesize_queries() to extract query strings.

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
    registry_resolution_threshold:
        Cosine similarity threshold for merging a new candidate into an
        existing registry dimension (default 0.88).
    hyde_abstracts_per_dim:
        Number of HyDE abstracts to generate when registering a new dimension.
    scaffold_registry_min_size:
        Registry must have at least this many entries before the full
        coverage map is shown instead of the scaffold.

    Returns
    -------
    A list of query dicts, each with keys: query, source, reasoning, dim_id.
    The first element is always the base topic query.
    """
    from redpill.registry import resolve_or_register
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
                scaffold_registry_min_size=scaffold_registry_min_size,
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

            # Ensure topic HyDE embedding exists for MMR.
            _ensure_topic_embedding(topic, llm_client, conn)

            # Run pool transitions at start of planning (uses prior run data).
            from redpill.bandit import (
                check_promotions,
                check_retirements,
                compute_budget_split,
                mmr_filter,
                select_exploit_dims,
                select_explore_dims,
            )
            try:
                check_promotions(conn, topic, k=promotion_k)
                check_retirements(conn, topic)
            except Exception as exc:
                logger.warning("plan_queries: pool transition failed: %s", exc)

            # Resolve each dimension against the registry before bandit selection.
            for dim in plan.get("dimensions", []):
                name = dim.get("name", "")
                if not name:
                    dim["dim_id"] = "dim_fallback"
                    continue
                try:
                    dim["dim_id"] = resolve_or_register(
                        name,
                        topic,
                        llm_client,
                        conn,
                        threshold=registry_resolution_threshold,
                        n_abstracts=hyde_abstracts_per_dim,
                    )
                except Exception as exc:
                    logger.warning(
                        "plan_queries: resolve_or_register failed for %r: %s — using hash",
                        name,
                        exc,
                    )
                    dim["dim_id"] = compute_dim_id(name, topic)

            # Bandit selection: budget split → pool selections → MMR filter.
            try:
                n_exploit, n_explore = compute_budget_split(max_queries, conn, topic)
                exploit_sels = select_exploit_dims(
                    n_exploit, conn, topic, alpha=ucb_alpha,
                    saturation_decay_days=saturation_decay_days,
                    saturation_penalty_weight=saturation_penalty_weight,
                )
                explore_sels = select_explore_dims(n_explore, conn, topic)

                for d in exploit_sels:
                    d = dict(d)
                for d in explore_sels:
                    d = dict(d)

                # Build proposed list as dicts with pool tag.
                proposed: list[dict] = []
                for d in exploit_sels:
                    proposed.append({**dict(d), "pool": "exploit"})
                for d in explore_sels:
                    proposed.append({**dict(d), "pool": "explore"})

                if proposed:
                    final_dims = mmr_filter(
                        proposed, conn, topic, mmr_lambda_floor=mmr_lambda_floor
                    )
                    # Restrict plan dimensions to bandit-selected set, preserving order.
                    selected_ids = [d["dim_id"] for d in final_dims]
                    dim_id_to_plan_dim = {
                        d.get("dim_id"): d for d in plan.get("dimensions", [])
                    }
                    plan["dimensions"] = [
                        dim_id_to_plan_dim[did]
                        for did in selected_ids
                        if did in dim_id_to_plan_dim
                    ]
                    logger.info(
                        "plan_queries: bandit selected %d dims (exploit=%d explore=%d)",
                        len(plan["dimensions"]), n_exploit, n_explore,
                    )
            except Exception as exc:
                logger.warning(
                    "plan_queries: bandit selection failed (%s) — using all plan dims", exc
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
