"""
search.py — Step 1: Query Tavily with multiple query variations.

Public API:
    search(queries: list[str], max_results: int) -> list[dict]
        Each result: {url, title, snippet, published_date}
        Merges results from all queries and deduplicates by URL.
        Retries on API errors (max 3, exponential backoff).
"""

import logging
import os
import time
from typing import Optional

from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    TimeoutError,
    UsageLimitExceededError,
)

logger = logging.getLogger(__name__)

# Errors that are permanent — no point retrying. Re-raised immediately.
_FATAL_ERRORS = (InvalidAPIKeyError, ForbiddenError, BadRequestError)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds; delay = base ** attempt (1s, 2s, 4s)


def _make_client(api_key: Optional[str] = None) -> TavilyClient:
    """Create a TavilyClient, pulling the key from the environment if not given."""
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. Add it to your .env file or pass api_key explicitly."
        )
    return TavilyClient(api_key=key)


def _search_one(client: TavilyClient, query: str, max_results: int) -> list[dict]:
    """
    Run a single Tavily query with up to _MAX_RETRIES attempts.

    Returns a list of normalised result dicts:
        {url, title, snippet, published_date, source_query}

    Raises the last exception if all retries are exhausted, except for fatal
    errors (bad key, forbidden, bad request) which are re-raised immediately.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            logger.debug("Querying Tavily: %r (attempt %d/%d)", query, attempt + 1, _MAX_RETRIES)
            response = client.search(query=query, max_results=max_results)
            raw_results: list[dict] = response.get("results", [])
            logger.debug("Query %r returned %d raw results", query, len(raw_results))
            return [_normalise(r, source_query=query) for r in raw_results]

        except _FATAL_ERRORS as exc:
            # No point retrying — configuration or auth error.
            logger.error("Fatal Tavily error for query %r: %s", query, exc)
            raise

        except Exception as exc:
            last_exc = exc
            delay = _BACKOFF_BASE ** attempt
            logger.warning(
                "Tavily request failed for query %r (attempt %d/%d): %s — retrying in %.1fs",
                query,
                attempt + 1,
                _MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)

    logger.error("All %d attempts exhausted for query %r", _MAX_RETRIES, query)
    raise last_exc  # type: ignore[misc]  # guaranteed non-None here


def _normalise(raw: dict, source_query: str = "") -> dict:
    """
    Map a raw Tavily result dict to the project's canonical shape.

    Tavily uses 'content' for the snippet text. We rename it to 'snippet'
    for clarity and to decouple callers from Tavily's naming conventions.

    Parameters
    ----------
    raw:
        A single result dict from the Tavily API response.
    source_query:
        The search query string that produced this result.  Attached to the
        result so downstream code (e.g. the sidecar writer) can attribute
        each item back to its originating query.
    """
    return {
        "url": raw.get("url", ""),
        "title": raw.get("title", ""),
        "snippet": raw.get("content", ""),
        "published_date": raw.get("published_date"),  # None when absent — intentional
        "source_query": source_query,
    }


def search(
    queries: list[str],
    max_results: int,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Run all query variations against Tavily, merge, and deduplicate by URL.

    Parameters
    ----------
    queries:
        One or more search query strings. All will be executed.
    max_results:
        Maximum results to request *per query*. Tavily's hard ceiling is 20.
    api_key:
        Tavily API key. Falls back to the TAVILY_API_KEY environment variable.

    Returns
    -------
    A list of result dicts, each with keys:
        url, title, snippet, published_date, source_query.
    ``source_query`` is the query string that produced the result.
    Results are deduplicated by URL (first occurrence wins).
    """
    if not queries:
        logger.warning("search() called with empty queries list — returning []")
        return []

    client = _make_client(api_key)
    seen_urls: set[str] = set()
    merged: list[dict] = []

    for query in queries:
        try:
            results = _search_one(client, query, max_results)
        except _FATAL_ERRORS:
            # Auth and config errors affect all queries equally — re-raise so
            # the caller knows the search could not run at all, rather than
            # silently returning an empty list that looks like a valid result.
            raise
        except Exception as exc:
            # Transient failures on one query should not abort the whole run.
            # Log and continue; callers will work with whatever we collected.
            logger.error("Skipping query %r after repeated failures: %s", query, exc)
            continue

        added = 0
        for result in results:
            url = result["url"]
            if not url:
                logger.debug("Dropping result with empty URL from query %r", query)
                continue
            if url in seen_urls:
                logger.debug("Deduplicating URL (already seen): %s", url)
                continue
            seen_urls.add(url)
            merged.append(result)
            added += 1

        logger.info("Query %r: %d new results (total so far: %d)", query, added, len(merged))

    logger.info(
        "search() complete — %d queries, %d unique results",
        len(queries),
        len(merged),
    )
    return merged
