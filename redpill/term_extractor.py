"""
term_extractor.py — Step 2b: extract named terms from summarized articles.

After summarization, each item has a relevance_score and may have content or
a snippet.  This module calls the LLM once per qualifying item and asks it to
identify domain-specific terms worth tracking for future query planning.

Terms are persisted to the ``extracted_terms`` table in state.py; the query
planner (Phase 3) reads them back to generate targeted search queries.

Public API:
    extract_terms(item, topic, llm_client) -> list[dict]
        Extract terms from a single item dict.  Returns [] on LLM failure.

    extract_terms_batch(items, topic, llm_client) -> list[dict]
        Filter qualifying items, call extract_terms per item, return all
        extracted term dicts with source_url/source_title attached.

Item filtering (applied in extract_terms_batch):
    - extraction_success must be True   (snippet-only items have too little signal)
    - relevance_score must be >= 3      (low-relevance articles waste LLM calls)

Each returned term dict has:
    term, category, source_url, source_title, topic, first_seen, last_seen

Categories (as used in the LLM prompt):
    subtopic | technique | author | dataset | framework | keyword
"""

import logging
from datetime import date as _date
from typing import TYPE_CHECKING

from redpill.llm_utils import extract_json

if TYPE_CHECKING:
    from redpill.summarize import LLMClient

logger = logging.getLogger(__name__)

# Items with relevance_score below this threshold are skipped.
MIN_RELEVANCE_SCORE = 3

# Terms with relevance (as judged by the LLM, 1-5) below this are dropped.
MIN_TERM_RELEVANCE = 3

_SYSTEM_PROMPT = """\
You are a research assistant that extracts domain-specific terms from article text.
You return only valid JSON — no explanation, no markdown, no preamble.
"""

_TERM_CATEGORIES = "subtopic | technique | author | dataset | framework | keyword"


def _build_extraction_prompt(content: str, topic: str) -> str:
    """Build the extraction prompt for a single article."""
    # Truncate to avoid overflowing small local model context windows.
    truncated = content[:4000] if len(content) > 4000 else content
    return f"""\
Topic: {topic}

Article text:
\"\"\"
{truncated}
\"\"\"

Extract up to 10 domain-specific terms from the article that are relevant to \
the topic above. Skip generic terms like "research", "paper", "study", "results", \
"method", "approach", "model", "data", or any term that is too broad to be a \
useful search query.

Return a JSON array. Each element must have exactly these keys:
  "term"      — the term or phrase (string)
  "category"  — one of: {_TERM_CATEGORIES}
  "relevance" — integer 1-5 (5 = highly specific and useful for future searches)

Example:
[
  {{"term": "MoCo", "category": "technique", "relevance": 5}},
  {{"term": "Yann LeCun", "category": "author", "relevance": 4}}
]

Return [] if no specific terms are found.
"""


def extract_terms(
    item: dict,
    topic: str,
    llm_client: "LLMClient",
    db_path: str | None = None,
) -> list[dict]:
    """Extract domain-specific terms from a single summarized item.

    Parameters
    ----------
    item:
        A merged/summarized item dict. Must have a ``content`` key (the full
        article text). ``url`` and ``title`` are used for attribution.
    topic:
        The research topic — used to judge term relevance in the prompt.
    llm_client:
        Any object satisfying the LLMClient Protocol.
    db_path:
        When provided, the raw LLM response is written to llm_call_log in
        this database.  Pass None to skip logging (default).

    Returns
    -------
    A list of term dicts. Each has: ``term``, ``category``, ``source_url``,
    ``source_title``.  Returns ``[]`` on LLM failure or bad JSON.
    """
    content: str = item.get("content") or item.get("snippet") or ""
    url: str = item.get("url", "")
    title: str = item.get("title", "")

    if not content.strip():
        logger.debug("extract_terms: skipping item with empty content url=%r", url)
        return []

    prompt = _build_extraction_prompt(content, topic)
    try:
        raw = llm_client.generate(prompt, system=_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning("extract_terms: LLM call failed for url=%r: %s", url, exc)
        return []

    # Persist raw response for debugging / analysis.
    if db_path is not None:
        from redpill.state import log_llm_call
        log_llm_call(
            call_site="extract_terms",
            raw_response=getattr(llm_client, "last_raw_response", None) or raw,
            db_path=db_path,
            model=getattr(llm_client, "_model", None),
            topic=topic,
            prompt_len=len(prompt),
            thinking=getattr(llm_client, "last_thinking", None),
        )

    parsed = extract_json(raw)

    if parsed is None:
        logger.warning("extract_terms: could not parse JSON for url=%r", url)
        return []

    if not isinstance(parsed, list):
        logger.warning(
            "extract_terms: expected list, got %s for url=%r", type(parsed).__name__, url
        )
        return []

    today = _date.today().isoformat()
    results: list[dict] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term", "")
        category = entry.get("category", "keyword")
        relevance = entry.get("relevance", 0)

        if not term or not isinstance(term, str):
            continue
        try:
            if int(relevance) < MIN_TERM_RELEVANCE:
                logger.debug("extract_terms: dropping low-relevance term %r (score=%s)", term, relevance)
                continue
        except (TypeError, ValueError):
            continue

        results.append(
            {
                "term": term.strip(),
                "category": category if isinstance(category, str) else "keyword",
                "source_url": url or None,
                "source_title": title or None,
                "topic": topic,
                "first_seen": today,
                "last_seen": today,
            }
        )

    logger.debug("extract_terms: %d term(s) extracted from url=%r", len(results), url)
    return results


def extract_terms_batch(
    items: list[dict],
    topic: str,
    llm_client: "LLMClient",
    db_path: str | None = None,
) -> list[dict]:
    """Extract terms from a batch of summarized items.

    Only processes items that pass both filters:
    - ``extraction_success`` is True (real content, not just a snippet)
    - ``relevance_score`` >= MIN_RELEVANCE_SCORE (worth the LLM call)

    Parameters
    ----------
    items:
        Summarized item dicts from the pipeline (post summarize_item).
    topic:
        Research topic passed through to extract_terms.
    llm_client:
        Any LLMClient-compatible object.
    db_path:
        When provided, each raw LLM response is logged to llm_call_log.
        Passed through to extract_terms.  Pass None to skip logging (default).

    Returns
    -------
    Flat list of all extracted term dicts across all qualifying items.
    """
    all_terms: list[dict] = []
    skipped = 0

    for item in items:
        if not item.get("extraction_success", False):
            skipped += 1
            logger.debug(
                "extract_terms_batch: skipping item (extraction_success=False) url=%r",
                item.get("url"),
            )
            continue

        try:
            score = int(item.get("relevance_score", 0))
        except (TypeError, ValueError):
            score = 0

        if score < MIN_RELEVANCE_SCORE:
            skipped += 1
            logger.debug(
                "extract_terms_batch: skipping item (relevance_score=%d) url=%r",
                score, item.get("url"),
            )
            continue

        terms = extract_terms(item, topic, llm_client, db_path=db_path)
        all_terms.extend(terms)

    logger.info(
        "extract_terms_batch: %d item(s) processed, %d skipped, %d term(s) total",
        len(items) - skipped,
        skipped,
        len(all_terms),
    )
    return all_terms
