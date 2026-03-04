"""
tests/test_search.py — Unit tests for redpill.search

All Tavily I/O is mocked; no network access occurs.
"""

import pytest
from unittest.mock import MagicMock, call, patch
from tavily.errors import (
    BadRequestError,
    InvalidAPIKeyError,
    TimeoutError,
    UsageLimitExceededError,
)

from redpill.search import _normalise, _search_one, search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tavily_result(
    url: str = "https://example.com/1",
    title: str = "Example Title",
    content: str = "A short snippet.",
    published_date: str | None = "2026-03-01",
) -> dict:
    """Build a raw Tavily result dict as the API would return it."""
    result = {"url": url, "title": title, "content": content}
    if published_date is not None:
        result["published_date"] = published_date
    return result


def _make_tavily_response(results: list[dict]) -> dict:
    """Wrap raw results in the Tavily response envelope."""
    return {"results": results}


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_maps_content_to_snippet(self):
        raw = _make_tavily_result(content="The snippet text.")
        result = _normalise(raw)
        assert result["snippet"] == "The snippet text."
        assert "content" not in result

    def test_preserves_url_title_published_date(self):
        raw = _make_tavily_result(
            url="https://example.com",
            title="My Title",
            published_date="2026-01-15",
        )
        result = _normalise(raw)
        assert result["url"] == "https://example.com"
        assert result["title"] == "My Title"
        assert result["published_date"] == "2026-01-15"

    def test_published_date_none_when_absent(self):
        raw = {"url": "https://example.com", "title": "T", "content": "S"}
        result = _normalise(raw)
        assert result["published_date"] is None

    def test_missing_fields_default_to_empty_string(self):
        result = _normalise({})
        assert result["url"] == ""
        assert result["title"] == ""
        assert result["snippet"] == ""


# ---------------------------------------------------------------------------
# _search_one
# ---------------------------------------------------------------------------

class TestSearchOne:
    def _make_client(self, return_value: dict) -> MagicMock:
        client = MagicMock()
        client.search.return_value = return_value
        return client

    def test_returns_normalised_results(self):
        raw = _make_tavily_result(url="https://a.com", content="snippet a")
        client = self._make_client(_make_tavily_response([raw]))
        results = _search_one(client, "test query", max_results=5)
        assert len(results) == 1
        assert results[0]["url"] == "https://a.com"
        assert results[0]["snippet"] == "snippet a"
        assert "content" not in results[0]

    def test_passes_query_and_max_results_to_client(self):
        client = self._make_client(_make_tavily_response([]))
        _search_one(client, "ml papers", max_results=12)
        client.search.assert_called_once_with(query="ml papers", max_results=12)

    def test_empty_results_key_returns_empty_list(self):
        client = self._make_client({"results": []})
        assert _search_one(client, "q", max_results=5) == []

    def test_missing_results_key_returns_empty_list(self):
        client = self._make_client({})
        assert _search_one(client, "q", max_results=5) == []

    def test_retries_on_transient_error_then_succeeds(self):
        raw = _make_tavily_result()
        client = MagicMock()
        client.search.side_effect = [
            UsageLimitExceededError("rate limit"),
            _make_tavily_response([raw]),
        ]
        with patch("redpill.search.time.sleep"):
            results = _search_one(client, "q", max_results=5)
        assert len(results) == 1
        assert client.search.call_count == 2

    def test_retries_on_timeout_then_succeeds(self):
        raw = _make_tavily_result()
        client = MagicMock()
        client.search.side_effect = [
            TimeoutError(60),
            _make_tavily_response([raw]),
        ]
        with patch("redpill.search.time.sleep"):
            results = _search_one(client, "q", max_results=5)
        assert len(results) == 1

    def test_raises_after_max_retries_exhausted(self):
        client = MagicMock()
        client.search.side_effect = Exception("service unavailable")
        with patch("redpill.search.time.sleep"):
            with pytest.raises(Exception, match="service unavailable"):
                _search_one(client, "q", max_results=5)
        assert client.search.call_count == 3  # _MAX_RETRIES

    def test_fatal_error_raises_immediately_no_retry(self):
        client = MagicMock()
        client.search.side_effect = InvalidAPIKeyError("bad key")
        with pytest.raises(InvalidAPIKeyError):
            _search_one(client, "q", max_results=5)
        assert client.search.call_count == 1  # no retry

    def test_bad_request_raises_immediately_no_retry(self):
        client = MagicMock()
        client.search.side_effect = BadRequestError("bad request")
        with pytest.raises(BadRequestError):
            _search_one(client, "q", max_results=5)
        assert client.search.call_count == 1

    def test_exponential_backoff_delays(self):
        client = MagicMock()
        client.search.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            Exception("fail 3"),
        ]
        with patch("redpill.search.time.sleep") as mock_sleep:
            with pytest.raises(Exception):
                _search_one(client, "q", max_results=5)
        # attempts 0, 1, 2 → delays 2^0=1.0, 2^1=2.0, 2^2=4.0
        assert mock_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]


# ---------------------------------------------------------------------------
# search (public API)
# ---------------------------------------------------------------------------

class TestSearch:
    def _patch_client(self, responses: list[dict]):
        """
        Patch _make_client to return a mock whose .search yields one response
        per call (one response per query).
        """
        client = MagicMock()
        client.search.side_effect = responses
        return patch("redpill.search._make_client", return_value=client), client

    def test_single_query_returns_normalised_results(self):
        response = _make_tavily_response([
            _make_tavily_result(url="https://a.com"),
            _make_tavily_result(url="https://b.com"),
        ])
        patcher, _ = self._patch_client([response])
        with patcher:
            results = search(["query one"], max_results=10, api_key="fake")
        assert len(results) == 2
        assert {r["url"] for r in results} == {"https://a.com", "https://b.com"}

    def test_multiple_queries_merges_results(self):
        resp1 = _make_tavily_response([_make_tavily_result(url="https://a.com")])
        resp2 = _make_tavily_response([_make_tavily_result(url="https://b.com")])
        patcher, _ = self._patch_client([resp1, resp2])
        with patcher:
            results = search(["q1", "q2"], max_results=5, api_key="fake")
        assert len(results) == 2

    def test_deduplicates_same_url_across_queries(self):
        duplicate_url = "https://same.com/article"
        resp1 = _make_tavily_response([_make_tavily_result(url=duplicate_url, title="First")])
        resp2 = _make_tavily_response([_make_tavily_result(url=duplicate_url, title="Second")])
        patcher, _ = self._patch_client([resp1, resp2])
        with patcher:
            results = search(["q1", "q2"], max_results=5, api_key="fake")
        assert len(results) == 1
        # First occurrence wins
        assert results[0]["title"] == "First"

    def test_deduplicates_same_url_within_single_query(self):
        duplicate_url = "https://same.com/article"
        resp = _make_tavily_response([
            _make_tavily_result(url=duplicate_url, title="First"),
            _make_tavily_result(url=duplicate_url, title="Second"),
        ])
        patcher, _ = self._patch_client([resp])
        with patcher:
            results = search(["q1"], max_results=5, api_key="fake")
        assert len(results) == 1

    def test_empty_queries_returns_empty_list(self):
        with patch("redpill.search._make_client") as mock_make:
            results = search([], max_results=10, api_key="fake")
        assert results == []
        mock_make.assert_not_called()

    def test_results_with_empty_url_are_dropped(self):
        resp = _make_tavily_response([
            _make_tavily_result(url=""),
            _make_tavily_result(url="https://valid.com"),
        ])
        patcher, _ = self._patch_client([resp])
        with patcher:
            results = search(["q"], max_results=5, api_key="fake")
        assert len(results) == 1
        assert results[0]["url"] == "https://valid.com"

    def test_failed_query_is_skipped_others_still_run(self):
        """A permanently failing query should not abort the whole search."""
        resp_good = _make_tavily_response([_make_tavily_result(url="https://good.com")])
        client = MagicMock()
        # First query always fails; second succeeds.
        client.search.side_effect = [
            Exception("always fails"),
            Exception("always fails"),
            Exception("always fails"),
            resp_good,
        ]
        with patch("redpill.search._make_client", return_value=client):
            with patch("redpill.search.time.sleep"):
                results = search(["bad query", "good query"], max_results=5, api_key="fake")
        assert len(results) == 1
        assert results[0]["url"] == "https://good.com"

    def test_fatal_error_propagates_immediately(self):
        """InvalidAPIKeyError should not be swallowed — it's a configuration problem."""
        client = MagicMock()
        client.search.side_effect = InvalidAPIKeyError("bad key")
        with patch("redpill.search._make_client", return_value=client):
            with pytest.raises(InvalidAPIKeyError):
                search(["q"], max_results=5, api_key="fake")

    def test_missing_api_key_raises_environment_error(self):
        # clear=True wipes the entire environment for the duration of the block,
        # so TAVILY_API_KEY is guaranteed absent without any additional pop().
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="TAVILY_API_KEY"):
                search(["q"], max_results=5)

    def test_published_date_is_none_when_absent(self):
        raw = {"url": "https://nodatearticle.com", "title": "T", "content": "S"}
        resp = _make_tavily_response([raw])
        patcher, _ = self._patch_client([resp])
        with patcher:
            results = search(["q"], max_results=5, api_key="fake")
        assert results[0]["published_date"] is None

    def test_result_dict_has_required_keys(self):
        resp = _make_tavily_response([_make_tavily_result()])
        patcher, _ = self._patch_client([resp])
        with patcher:
            results = search(["q"], max_results=5, api_key="fake")
        assert len(results) == 1
        assert set(results[0].keys()) == {"url", "title", "snippet", "published_date"}
