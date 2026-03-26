"""
tests/test_search_providers.py — Unit tests for redpill.search_providers

All network I/O and Tavily client calls are mocked.  No real HTTP requests
are made.

Test structure:
    TestSearchResult              — dataclass field defaults
    TestTavilySearchProvider      — delegates to _make_client / _search_one
    TestSerperSearchProvider      — httpx.Client.post mocking, field mapping,
                                    num clamping, error cases
    TestFanOutSearchProvider      — merging, URL dedup, per-provider failure isolation
    TestCreateSearchProvider      — factory: correct types, ValueError on bad name
"""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from redpill.search_providers import (
    FanOutSearchProvider,
    SearchResult,
    SerperSearchProvider,
    TavilySearchProvider,
    create_search_provider,
)


# ---------------------------------------------------------------------------
# TestSearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_required_fields(self):
        r = SearchResult(url="https://a.com", title="T", snippet="S", source="tavily")
        assert r.url == "https://a.com"
        assert r.title == "T"
        assert r.snippet == "S"
        assert r.source == "tavily"

    def test_optional_fields_default_to_none_and_empty(self):
        r = SearchResult(url="https://a.com", title="T", snippet="S", source="tavily")
        assert r.published_date is None
        assert r.source_query == ""

    def test_optional_fields_can_be_set(self):
        r = SearchResult(
            url="https://a.com",
            title="T",
            snippet="S",
            source="tavily",
            published_date="2026-03-01",
            source_query="my query",
        )
        assert r.published_date == "2026-03-01"
        assert r.source_query == "my query"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result_dict(
    url: str = "https://example.com/1",
    title: str = "Title",
    snippet: str = "Snippet",
    published_date: str | None = "2026-03-01",
    source_query: str = "test query",
) -> dict:
    """Dict shape that _search_one returns (from search.py internals)."""
    return {
        "url": url,
        "title": title,
        "snippet": snippet,
        "published_date": published_date,
        "source_query": source_query,
    }


def _make_serper_response(
    organic: list[dict] | None = None,
) -> MagicMock:
    """Return a mock httpx.Response with the given organic results."""
    if organic is None:
        organic = [
            {
                "link": "https://example.com/serper1",
                "title": "Serper Title",
                "snippet": "Serper snippet",
                "position": 1,
            }
        ]
    response = MagicMock(spec=httpx.Response)
    response.is_success = True
    response.json.return_value = {"organic": organic}
    return response


# ---------------------------------------------------------------------------
# TestTavilySearchProvider
# ---------------------------------------------------------------------------


class TestTavilySearchProvider:
    def test_delegates_to_make_client_and_search_one(self):
        raw = _make_search_result_dict(url="https://a.com", snippet="text")
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client) as m_make,
            patch("redpill.search_providers._search_one", return_value=[raw]) as m_one,
        ):
            provider = TavilySearchProvider(api_key="fake-key")
            results = provider.search("test query", max_results=5)

        m_make.assert_called_once_with("fake-key")
        m_one.assert_called_once_with(mock_client, "test query", 5)
        assert len(results) == 1

    def test_returns_search_result_objects(self):
        raw = _make_search_result_dict(url="https://a.com")
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client),
            patch("redpill.search_providers._search_one", return_value=[raw]),
        ):
            provider = TavilySearchProvider(api_key="fake")
            results = provider.search("q", max_results=5)

        assert all(isinstance(r, SearchResult) for r in results)

    def test_sets_source_to_tavily(self):
        raw = _make_search_result_dict(url="https://a.com")
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client),
            patch("redpill.search_providers._search_one", return_value=[raw]),
        ):
            provider = TavilySearchProvider(api_key="fake")
            results = provider.search("q", max_results=5)

        assert results[0].source == "tavily"

    def test_preserves_published_date(self):
        raw = _make_search_result_dict(published_date="2026-01-15")
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client),
            patch("redpill.search_providers._search_one", return_value=[raw]),
        ):
            provider = TavilySearchProvider(api_key="fake")
            results = provider.search("q", max_results=5)

        assert results[0].published_date == "2026-01-15"

    def test_published_date_none_when_absent(self):
        raw = _make_search_result_dict(published_date=None)
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client),
            patch("redpill.search_providers._search_one", return_value=[raw]),
        ):
            provider = TavilySearchProvider(api_key="fake")
            results = provider.search("q", max_results=5)

        assert results[0].published_date is None

    def test_propagates_source_query(self):
        raw = _make_search_result_dict(source_query="ml papers")
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client),
            patch("redpill.search_providers._search_one", return_value=[raw]),
        ):
            provider = TavilySearchProvider(api_key="fake")
            results = provider.search("ml papers", max_results=5)

        assert results[0].source_query == "ml papers"

    def test_empty_results_returns_empty_list(self):
        mock_client = MagicMock()

        with (
            patch("redpill.search_providers._make_client", return_value=mock_client),
            patch("redpill.search_providers._search_one", return_value=[]),
        ):
            provider = TavilySearchProvider(api_key="fake")
            results = provider.search("q", max_results=5)

        assert results == []


# ---------------------------------------------------------------------------
# TestSerperSearchProvider
# ---------------------------------------------------------------------------


class TestSerperSearchProvider:
    def test_raises_environment_error_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="SERPER_API_KEY"):
                SerperSearchProvider()

    def test_uses_env_api_key_when_not_passed_explicitly(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "env-key"}):
            provider = SerperSearchProvider()
        assert provider._api_key == "env-key"

    def test_explicit_api_key_takes_precedence_over_env(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "env-key"}):
            provider = SerperSearchProvider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    def _make_provider(self) -> SerperSearchProvider:
        return SerperSearchProvider(api_key="test-api-key")

    def test_posts_to_correct_url(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            provider.search("test query", max_results=5)

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://google.serper.dev/search"

    def test_sends_correct_headers(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            provider.search("test query", max_results=5)

        _, kwargs = mock_client.post.call_args
        headers = kwargs["headers"]
        assert headers["X-API-KEY"] == "test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_sends_query_in_body(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            provider.search("contrastive learning", max_results=5)

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["q"] == "contrastive learning"

    def test_clamps_num_to_10_when_max_results_exceeds_limit(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            provider.search("q", max_results=20)

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["num"] == 10

    def test_num_not_clamped_when_within_limit(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            provider.search("q", max_results=7)

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["num"] == 7

    def test_maps_organic_results_to_search_result_objects(self):
        organic = [
            {"link": "https://a.com", "title": "Article A", "snippet": "Snippet A"},
            {"link": "https://b.com", "title": "Article B", "snippet": "Snippet B"},
        ]
        mock_response = _make_serper_response(organic)
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            results = provider.search("q", max_results=5)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].url == "https://a.com"
        assert results[0].title == "Article A"
        assert results[0].snippet == "Snippet A"
        assert results[1].url == "https://b.com"

    def test_sets_source_to_serper(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            results = provider.search("q", max_results=5)

        assert all(r.source == "serper" for r in results)

    def test_published_date_is_always_none(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            results = provider.search("q", max_results=5)

        assert all(r.published_date is None for r in results)

    def test_source_query_set_to_query_string(self):
        mock_response = _make_serper_response()
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            results = provider.search("deep learning", max_results=5)

        assert all(r.source_query == "deep learning" for r in results)

    def test_skips_items_with_missing_link(self):
        organic = [
            {"link": "", "title": "No URL", "snippet": "Snippet"},
            {"link": "https://valid.com", "title": "Valid", "snippet": "Valid snippet"},
        ]
        mock_response = _make_serper_response(organic)
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            results = provider.search("q", max_results=5)

        assert len(results) == 1
        assert results[0].url == "https://valid.com"

    def test_empty_organic_returns_empty_list(self):
        mock_response = _make_serper_response(organic=[])
        provider = self._make_provider()

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = mock_response

            results = provider.search("q", max_results=5)

        assert results == []

    def test_raises_on_http_error(self):
        """Non-2xx responses should propagate as httpx.HTTPStatusError."""
        provider = self._make_provider()
        error_response = MagicMock(spec=httpx.Response)
        error_response.is_success = False
        error_response.status_code = 401
        error_response.text = "Unauthorized"
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=error_response,
        )

        with patch("redpill.search_providers.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = error_response

            with pytest.raises(httpx.HTTPStatusError):
                provider.search("q", max_results=5)


# ---------------------------------------------------------------------------
# TestFanOutSearchProvider
# ---------------------------------------------------------------------------


def _make_mock_provider(
    results: list[SearchResult] | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    """Build a mock SearchProvider."""
    mock = MagicMock()
    if raises is not None:
        mock.search.side_effect = raises
    else:
        mock.search.return_value = results or []
    return mock


class TestFanOutSearchProvider:
    def test_calls_all_providers(self):
        r1 = SearchResult(url="https://a.com", title="A", snippet="s", source="tavily")
        r2 = SearchResult(url="https://b.com", title="B", snippet="s", source="serper")
        p1 = _make_mock_provider([r1])
        p2 = _make_mock_provider([r2])

        fanout = FanOutSearchProvider([p1, p2])
        results = fanout.search("q", max_results=5)

        p1.search.assert_called_once_with("q", 5)
        p2.search.assert_called_once_with("q", 5)
        assert len(results) == 2

    def test_deduplicates_by_url_first_provider_wins(self):
        shared_url = "https://same.com/article"
        r1 = SearchResult(url=shared_url, title="From Tavily", snippet="s", source="tavily")
        r2 = SearchResult(url=shared_url, title="From Serper", snippet="s", source="serper")
        r3 = SearchResult(url="https://unique.com", title="Unique", snippet="s", source="serper")

        p1 = _make_mock_provider([r1])
        p2 = _make_mock_provider([r2, r3])

        fanout = FanOutSearchProvider([p1, p2])
        results = fanout.search("q", max_results=5)

        urls = [r.url for r in results]
        assert urls.count(shared_url) == 1
        # First-seen (tavily) wins
        dupe = next(r for r in results if r.url == shared_url)
        assert dupe.source == "tavily"
        assert len(results) == 2

    def test_one_provider_failing_does_not_abort_other(self):
        r2 = SearchResult(url="https://b.com", title="B", snippet="s", source="serper")
        p1 = _make_mock_provider(raises=RuntimeError("network error"))
        p2 = _make_mock_provider([r2])

        fanout = FanOutSearchProvider([p1, p2])
        results = fanout.search("q", max_results=5)

        assert len(results) == 1
        assert results[0].url == "https://b.com"

    def test_all_providers_failing_returns_empty_list(self):
        p1 = _make_mock_provider(raises=RuntimeError("down"))
        p2 = _make_mock_provider(raises=RuntimeError("also down"))

        fanout = FanOutSearchProvider([p1, p2])
        results = fanout.search("q", max_results=5)

        assert results == []

    def test_results_with_empty_url_are_skipped(self):
        r1 = SearchResult(url="", title="No URL", snippet="s", source="tavily")
        r2 = SearchResult(url="https://valid.com", title="Valid", snippet="s", source="tavily")
        p1 = _make_mock_provider([r1, r2])

        fanout = FanOutSearchProvider([p1])
        results = fanout.search("q", max_results=5)

        assert len(results) == 1
        assert results[0].url == "https://valid.com"

    def test_single_provider(self):
        r = SearchResult(url="https://a.com", title="A", snippet="s", source="tavily")
        p = _make_mock_provider([r])

        fanout = FanOutSearchProvider([p])
        results = fanout.search("q", max_results=5)

        assert len(results) == 1

    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError, match="at least one provider"):
            FanOutSearchProvider([])

    def test_preserves_order_within_each_provider(self):
        results_p1 = [
            SearchResult(url=f"https://a{i}.com", title=f"A{i}", snippet="s", source="tavily")
            for i in range(3)
        ]
        p1 = _make_mock_provider(results_p1)
        fanout = FanOutSearchProvider([p1])
        results = fanout.search("q", max_results=5)

        assert [r.url for r in results] == [f"https://a{i}.com" for i in range(3)]


# ---------------------------------------------------------------------------
# TestCreateSearchProvider
# ---------------------------------------------------------------------------


class TestCreateSearchProvider:
    def test_tavily_returns_tavily_provider(self):
        provider = create_search_provider("tavily")
        assert isinstance(provider, TavilySearchProvider)

    def test_serper_returns_serper_provider(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test-key"}):
            provider = create_search_provider("serper")
        assert isinstance(provider, SerperSearchProvider)

    def test_both_returns_fanout_provider(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test-key"}):
            provider = create_search_provider("both")
        assert isinstance(provider, FanOutSearchProvider)

    def test_both_fanout_contains_tavily_then_serper(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test-key"}):
            provider = create_search_provider("both")
        assert isinstance(provider, FanOutSearchProvider)
        providers = provider._providers
        assert len(providers) == 2
        assert isinstance(providers[0], TavilySearchProvider)
        assert isinstance(providers[1], SerperSearchProvider)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown search provider"):
            create_search_provider("bing")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            create_search_provider("")
