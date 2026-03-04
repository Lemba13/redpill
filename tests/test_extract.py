"""
tests/test_extract.py — Unit tests for redpill.extract

All HTTP I/O is mocked; no real network calls are made.

Test strategy:
- extract() is tested by mocking _fetch_html() and trafilatura.extract().
  This keeps tests fast, deterministic, and focused on our logic rather than
  trafilatura's internals.
- extract_batch() is tested by mocking _extract_one() so we can validate
  ordering, concurrency semantics, and error handling independently of the
  single-URL path.
- Integration-style tests for _extract_one() use real HTML fixtures and mock
  only _fetch_html(), verifying that trafilatura parses content correctly.
- _is_pdf_url() and _parse_extraction() are tested as pure units.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import pytest

from redpill.extract import (
    _extract_one,
    _fetch_html,
    _is_pdf_url,
    _parse_extraction,
    extract,
    extract_batch,
)


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------
# trafilatura requires at least 250 chars of body text to return a result.
# These fixtures are deliberately wordy to stay above that threshold.

_ARTICLE_TITLE = "AI Research Digest: Key Findings This Week"

_ARTICLE_HTML = f"""
<html>
<head>
  <title>{_ARTICLE_TITLE}</title>
</head>
<body>
  <article>
    <h1>{_ARTICLE_TITLE}</h1>
    <p>
      Researchers at several leading institutions have published findings suggesting that
      contrastive learning methods continue to outperform supervised baselines on a wide
      range of downstream tasks, particularly when labelled data is scarce.
    </p>
    <p>
      The key insight from the latest batch of papers is that data augmentation strategy
      matters more than model architecture in the self-supervised setting, a finding that
      has significant implications for practitioners who rely on off-the-shelf encoders.
    </p>
    <p>
      Further work is needed to understand the interaction between augmentation strength
      and model capacity, but early results are promising and the community is converging
      on a set of best practices that should lower the barrier to entry considerably.
    </p>
  </article>
</body>
</html>
"""

_EMPTY_HTML = "<html><body></body></html>"

_BOILERPLATE_ONLY_HTML = """
<html>
<head><title>Site</title></head>
<body>
  <nav><a href="/">Home</a><a href="/about">About</a></nav>
  <footer>Copyright 2026</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# _is_pdf_url
# ---------------------------------------------------------------------------

class TestIsPdfUrl:
    def test_pdf_extension_detected(self):
        assert _is_pdf_url("https://arxiv.org/pdf/2401.00001.pdf") is True

    def test_pdf_extension_case_insensitive(self):
        assert _is_pdf_url("https://example.com/paper.PDF") is True
        assert _is_pdf_url("https://example.com/paper.Pdf") is True

    def test_html_url_not_flagged(self):
        assert _is_pdf_url("https://example.com/article") is False

    def test_url_with_pdf_in_path_segment_not_flagged(self):
        # "pdf" appears in a path component but the final segment doesn't end in .pdf
        assert _is_pdf_url("https://example.com/pdfs/report") is False

    def test_url_with_pdf_query_param_not_flagged(self):
        # Query string should not affect the check — only the path is examined
        assert _is_pdf_url("https://example.com/view?format=pdf") is False

    def test_empty_string_not_flagged(self):
        assert _is_pdf_url("") is False


# ---------------------------------------------------------------------------
# _parse_extraction
# ---------------------------------------------------------------------------

class TestParseExtraction:
    def _make_json(self, text: str | None = "body text", title: str | None = "Page Title") -> str:
        return json.dumps({"text": text, "title": title})

    def test_returns_text_and_title(self):
        raw = self._make_json(text="body text", title="Page Title")
        text, title = _parse_extraction(raw)
        assert text == "body text"
        assert title == "Page Title"

    def test_none_title_returned_as_none(self):
        raw = self._make_json(text="body text", title=None)
        text, title = _parse_extraction(raw)
        assert text == "body text"
        assert title is None

    def test_empty_string_title_coerced_to_none(self):
        # Empty string is falsy — we normalize it to None so callers
        # don't need to distinguish "" from None.
        raw = self._make_json(text="body text", title="")
        text, title = _parse_extraction(raw)
        assert title is None

    def test_none_text_returned_as_none(self):
        raw = self._make_json(text=None, title="Title")
        text, title = _parse_extraction(raw)
        assert text is None

    def test_empty_string_text_coerced_to_none(self):
        raw = self._make_json(text="", title="Title")
        text, title = _parse_extraction(raw)
        assert text is None

    def test_invalid_json_returns_none_none(self):
        text, title = _parse_extraction("this is not json {{{")
        assert text is None
        assert title is None

    def test_missing_keys_returns_none_none(self):
        raw = json.dumps({"author": "someone"})
        text, title = _parse_extraction(raw)
        assert text is None
        assert title is None


# ---------------------------------------------------------------------------
# _fetch_html (unit: mock requests)
# ---------------------------------------------------------------------------

class TestFetchHtml:
    def test_returns_html_on_success(self):
        mock_response = MagicMock()
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status.return_value = None
        with patch("redpill.extract.requests.get", return_value=mock_response) as mock_get:
            result = _fetch_html("https://example.com")
        assert result == "<html>content</html>"
        mock_get.assert_called_once_with(
            "https://example.com",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; redpill-bot/0.1)"},
        )

    def test_returns_none_on_timeout(self):
        import requests as req
        with patch("redpill.extract.requests.get", side_effect=req.exceptions.Timeout):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_http_error(self):
        import requests as req
        with patch("redpill.extract.requests.get", side_effect=req.exceptions.HTTPError("404")):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_connection_error(self):
        import requests as req
        with patch(
            "redpill.extract.requests.get",
            side_effect=req.exceptions.ConnectionError("refused"),
        ):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_raise_for_status_is_called(self):
        """raise_for_status() must be called so 4xx/5xx responses are treated as errors."""
        import requests as req
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.exceptions.HTTPError("403 Forbidden")
        with patch("redpill.extract.requests.get", return_value=mock_response):
            result = _fetch_html("https://example.com/paywall")
        assert result is None

    def test_custom_timeout_is_passed(self):
        mock_response = MagicMock()
        mock_response.text = "<html/>"
        mock_response.raise_for_status.return_value = None
        with patch("redpill.extract.requests.get", return_value=mock_response) as mock_get:
            _fetch_html("https://example.com", timeout=5)
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 5


# ---------------------------------------------------------------------------
# extract() — public single-URL API
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_text_on_success(self):
        with patch("redpill.extract._fetch_html", return_value=_ARTICLE_HTML):
            result = extract("https://example.com/article")
        assert result is not None
        assert "contrastive learning" in result

    def test_returns_none_for_pdf_url(self):
        with patch("redpill.extract._fetch_html") as mock_fetch:
            result = extract("https://example.com/paper.pdf")
        assert result is None
        mock_fetch.assert_not_called()

    def test_returns_none_when_fetch_fails(self):
        with patch("redpill.extract._fetch_html", return_value=None):
            result = extract("https://example.com/down")
        assert result is None

    def test_returns_none_when_trafilatura_finds_no_content(self):
        # A page with only nav/footer — trafilatura returns None
        with patch("redpill.extract._fetch_html", return_value=_EMPTY_HTML):
            result = extract("https://example.com/empty")
        assert result is None

    def test_returns_none_when_trafilatura_returns_none(self):
        with patch("redpill.extract._fetch_html", return_value=_ARTICLE_HTML):
            with patch("redpill.extract.trafilatura.extract", return_value=None):
                result = extract("https://example.com/article")
        assert result is None

    def test_returns_none_when_extraction_json_has_no_text(self):
        """trafilatura returned JSON but with null text — treat as failure."""
        empty_result = json.dumps({"text": None, "title": "Some Title"})
        with patch("redpill.extract._fetch_html", return_value=_ARTICLE_HTML):
            with patch("redpill.extract.trafilatura.extract", return_value=empty_result):
                result = extract("https://example.com/article")
        assert result is None

    def test_trafilatura_called_with_metadata_and_json_format(self):
        """Verify we always request metadata + JSON output from trafilatura."""
        with patch("redpill.extract._fetch_html", return_value=_ARTICLE_HTML):
            with patch("redpill.extract.trafilatura.extract", return_value=None) as mock_extract:
                extract("https://example.com/article")
        mock_extract.assert_called_once_with(
            _ARTICLE_HTML,
            with_metadata=True,
            output_format="json",
        )


# ---------------------------------------------------------------------------
# _extract_one — per-worker helper
# ---------------------------------------------------------------------------

class TestExtractOne:
    """
    These tests use real HTML fixtures and mock only _fetch_html.
    This validates the full extraction path (requests -> trafilatura -> dict)
    without hitting the network.
    """

    def test_success_returns_correct_dict_shape(self):
        with patch("redpill.extract._fetch_html", return_value=_ARTICLE_HTML):
            result = _extract_one("https://example.com/article")
        assert result["url"] == "https://example.com/article"
        assert result["extraction_success"] is True
        assert result["title"] == _ARTICLE_TITLE
        assert "contrastive learning" in result["content"]

    def test_required_keys_always_present(self):
        with patch("redpill.extract._fetch_html", return_value=None):
            result = _extract_one("https://example.com/down")
        assert set(result.keys()) == {"url", "title", "content", "extraction_success"}

    def test_failure_sets_extraction_success_false(self):
        with patch("redpill.extract._fetch_html", return_value=None):
            result = _extract_one("https://example.com/down")
        assert result["extraction_success"] is False

    def test_failure_sets_content_to_none(self):
        with patch("redpill.extract._fetch_html", return_value=None):
            result = _extract_one("https://example.com/down")
        assert result["content"] is None

    def test_failure_sets_title_to_empty_string(self):
        with patch("redpill.extract._fetch_html", return_value=None):
            result = _extract_one("https://example.com/down")
        assert result["title"] == ""

    def test_pdf_url_returns_failure_without_fetching(self):
        with patch("redpill.extract._fetch_html") as mock_fetch:
            result = _extract_one("https://example.com/paper.pdf")
        assert result["extraction_success"] is False
        mock_fetch.assert_not_called()

    def test_page_without_extractable_content_returns_failure(self):
        with patch("redpill.extract._fetch_html", return_value=_EMPTY_HTML):
            result = _extract_one("https://example.com/empty")
        assert result["extraction_success"] is False
        assert result["content"] is None

    def test_page_with_no_title_uses_empty_string(self):
        """When trafilatura finds content but no title, title should be "" not None."""
        no_title_json = json.dumps({"text": "Some article content here.", "title": None})
        with patch("redpill.extract._fetch_html", return_value=_ARTICLE_HTML):
            with patch("redpill.extract.trafilatura.extract", return_value=no_title_json):
                result = _extract_one("https://example.com/no-title")
        assert result["title"] == ""
        assert result["extraction_success"] is True


# ---------------------------------------------------------------------------
# extract_batch — public batch API
# ---------------------------------------------------------------------------

class TestExtractBatch:
    def _make_success(self, url: str, title: str = "Title", content: str = "text") -> dict:
        return {"url": url, "title": title, "content": content, "extraction_success": True}

    def _make_failure(self, url: str) -> dict:
        return {"url": url, "title": "", "content": None, "extraction_success": False}

    def test_empty_input_returns_empty_list(self):
        with patch("redpill.extract._extract_one") as mock_one:
            result = extract_batch([])
        assert result == []
        mock_one.assert_not_called()

    def test_single_url_returns_one_result(self):
        url = "https://example.com/a"
        with patch("redpill.extract._extract_one", return_value=self._make_success(url)):
            results = extract_batch([url])
        assert len(results) == 1
        assert results[0]["url"] == url

    def test_results_preserve_input_order(self):
        """Order must match input list, not completion order (as_completed is arbitrary)."""
        urls = [f"https://example.com/{i}" for i in range(10)]

        def fake_extract(url):
            return self._make_success(url)

        with patch("redpill.extract._extract_one", side_effect=fake_extract):
            results = extract_batch(urls)

        assert [r["url"] for r in results] == urls

    def test_all_results_have_required_keys(self):
        urls = ["https://example.com/a", "https://example.com/b"]
        side_effects = [
            self._make_success(urls[0]),
            self._make_failure(urls[1]),
        ]
        with patch("redpill.extract._extract_one", side_effect=side_effects):
            results = extract_batch(urls)
        for result in results:
            assert set(result.keys()) == {"url", "title", "content", "extraction_success"}

    def test_one_failed_url_does_not_abort_batch(self):
        urls = ["https://ok.com/a", "https://down.com/b", "https://ok.com/c"]
        side_effects = [
            self._make_success(urls[0]),
            self._make_failure(urls[1]),
            self._make_success(urls[2]),
        ]
        with patch("redpill.extract._extract_one", side_effect=side_effects):
            results = extract_batch(urls)
        assert len(results) == 3
        assert results[0]["extraction_success"] is True
        assert results[1]["extraction_success"] is False
        assert results[2]["extraction_success"] is True

    def test_unexpected_exception_from_worker_produces_failure_dict(self):
        """_extract_one is designed not to raise, but extract_batch must handle it anyway."""
        url = "https://example.com/boom"
        with patch("redpill.extract._extract_one", side_effect=RuntimeError("unexpected")):
            results = extract_batch([url])
        assert len(results) == 1
        assert results[0]["extraction_success"] is False
        assert results[0]["url"] == url
        assert results[0]["content"] is None

    def test_uses_thread_pool_executor(self):
        """extract_batch must use ThreadPoolExecutor, not sequential calls."""
        urls = ["https://example.com/a", "https://example.com/b"]
        with patch("redpill.extract._extract_one", side_effect=lambda u: self._make_success(u)):
            with patch("redpill.extract.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as mock_pool:
                extract_batch(urls)
        mock_pool.assert_called_once_with(max_workers=5)

    def test_returns_list_of_dicts_with_success_and_failure_mixed(self):
        """Smoke test on real HTML to verify the full path end-to-end."""
        urls = ["https://a.com/article", "https://b.com/404"]

        def fake_fetch(url, **kwargs):
            if "a.com" in url:
                return _ARTICLE_HTML
            return None  # simulate failed fetch for b.com

        with patch("redpill.extract._fetch_html", side_effect=fake_fetch):
            results = extract_batch(urls)

        assert results[0]["url"] == "https://a.com/article"
        assert results[0]["extraction_success"] is True
        assert results[0]["content"] is not None

        assert results[1]["url"] == "https://b.com/404"
        assert results[1]["extraction_success"] is False
        assert results[1]["content"] is None

    def test_max_workers_constant_is_five(self):
        """Verify the module-level max_workers constant matches the spec."""
        from redpill.extract import _MAX_WORKERS
        assert _MAX_WORKERS == 5
