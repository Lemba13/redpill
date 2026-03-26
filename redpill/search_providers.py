"""
search_providers.py — SearchProvider protocol, SearchResult dataclass, and
concrete provider implementations (Tavily, Serper, FanOut).

Public API
----------
SearchResult               dataclass — canonical result shape for all providers
SearchProvider             Protocol  — synchronous search interface
TavilySearchProvider       wraps existing _make_client / _search_one logic
SerperSearchProvider       POST to https://google.serper.dev/search via httpx
FanOutSearchProvider       sequences multiple providers, merges + deduplicates
create_search_provider     factory function keyed by provider name string
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

from redpill.search import _make_client, _search_one

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"
_SERPER_MAX_RESULTS = 10  # Serper free-tier cap


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Canonical result shape produced by every search provider."""

    url: str
    title: str
    snippet: str
    source: str                      # "tavily" | "serper" — for attribution
    published_date: str | None = None
    source_query: str = ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SearchProvider(Protocol):
    """Synchronous search interface all concrete providers must satisfy."""

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        ...


# ---------------------------------------------------------------------------
# TavilySearchProvider
# ---------------------------------------------------------------------------

class TavilySearchProvider:
    """Thin wrapper around the existing Tavily search internals in search.py.

    Delegates to ``_make_client`` and ``_search_one`` so all retry logic,
    backoff, and fatal-error handling live in exactly one place.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Defer key resolution to first call so unit tests can patch
        # _make_client without constructing a real TavilyClient.
        self._api_key = api_key

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        client = _make_client(self._api_key)
        raw_dicts = _search_one(client, query, max_results)
        return [
            SearchResult(
                url=d["url"],
                title=d["title"],
                snippet=d["snippet"],
                source="tavily",
                published_date=d.get("published_date"),
                source_query=d.get("source_query", query),
            )
            for d in raw_dicts
        ]


# ---------------------------------------------------------------------------
# SerperSearchProvider
# ---------------------------------------------------------------------------

class SerperSearchProvider:
    """Calls the Serper Google SERP API and maps results to SearchResult.

    Raises
    ------
    EnvironmentError
        At construction time if SERPER_API_KEY is not in the environment and
        no explicit *api_key* was passed.
    """

    def __init__(self, api_key: str | None = None) -> None:
        resolved = api_key or os.environ.get("SERPER_API_KEY")
        if not resolved:
            raise EnvironmentError(
                "SERPER_API_KEY is not set. "
                "Add it to your .env file or pass api_key explicitly."
            )
        self._api_key = resolved

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """POST *query* to Serper and return normalised results.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Desired number of results.  Clamped to 10 (Serper free-tier cap).
        """
        num = min(max_results, _SERPER_MAX_RESULTS)
        logger.debug("Querying Serper: %r (num=%d)", query, num)

        with httpx.Client() as client:
            response = client.post(
                _SERPER_URL,
                headers={
                    "X-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num},
            )

        if not response.is_success:
            logger.error(
                "Serper returned HTTP %d for query %r: %s",
                response.status_code,
                query,
                response.text[:200],
            )
            response.raise_for_status()

        data = response.json()
        results: list[SearchResult] = []
        for item in data.get("organic", []):
            url = item.get("link", "")
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    source="serper",
                    published_date=None,
                    source_query=query,
                )
            )

        logger.debug("Serper query %r returned %d result(s)", query, len(results))
        return results


# ---------------------------------------------------------------------------
# FanOutSearchProvider
# ---------------------------------------------------------------------------

class FanOutSearchProvider:
    """Sequences multiple providers, merges results, and deduplicates by URL.

    URL deduplication is first-seen wins.  The order of *providers* therefore
    controls priority: pass Tavily first if you want Tavily results to win on
    collisions.

    If a provider raises, the error is logged and the loop continues with the
    remaining providers.  The caller receives whatever was collected.
    """

    def __init__(self, providers: list[SearchProvider]) -> None:
        if not providers:
            raise ValueError("FanOutSearchProvider requires at least one provider")
        self._providers = providers

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        seen_urls: set[str] = set()
        merged: list[SearchResult] = []

        for provider in self._providers:
            provider_name = type(provider).__name__
            try:
                results = provider.search(query, max_results)
            except Exception as exc:
                logger.error(
                    "Provider %s failed for query %r: %s — continuing with other providers",
                    provider_name,
                    query,
                    exc,
                )
                continue

            added = 0
            for result in results:
                if not result.url:
                    continue
                if result.url in seen_urls:
                    logger.debug(
                        "FanOut: deduplicating URL (already seen from earlier provider): %s",
                        result.url,
                    )
                    continue
                seen_urls.add(result.url)
                merged.append(result)
                added += 1

            logger.debug(
                "FanOut: provider %s added %d new result(s) for query %r",
                provider_name,
                added,
                query,
            )

        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_search_provider(provider_name: str) -> SearchProvider:
    """Return the correct SearchProvider instance for *provider_name*.

    Parameters
    ----------
    provider_name:
        One of ``"tavily"``, ``"serper"``, or ``"both"``.

    Returns
    -------
    A ``SearchProvider`` instance ready to use.

    Raises
    ------
    ValueError
        If *provider_name* is not one of the three valid options.
    """
    match provider_name:
        case "tavily":
            return TavilySearchProvider()
        case "serper":
            return SerperSearchProvider()
        case "both":
            return FanOutSearchProvider([
                TavilySearchProvider(),
                SerperSearchProvider(),
            ])
        case _:
            raise ValueError(
                f"Unknown search provider {provider_name!r}. "
                "Must be one of: 'tavily', 'serper', 'both'."
            )
