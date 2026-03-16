"""
tests/test_term_extractor.py — Unit tests for redpill.term_extractor

All LLM calls are mocked. No network or Ollama access occurs.
"""

import json
from unittest.mock import MagicMock

import pytest

from redpill.term_extractor import (
    MIN_RELEVANCE_SCORE,
    MIN_TERM_RELEVANCE,
    extract_terms,
    extract_terms_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(response: str) -> MagicMock:
    """Return a mock LLMClient whose generate() returns *response*."""
    client = MagicMock()
    client.generate.return_value = response
    return client


def _item(
    url: str = "https://example.com/paper",
    title: str = "Test Paper",
    content: str = "This paper introduces SimCLR, a contrastive learning framework by Geoffrey Hinton.",
    snippet: str = "",
    extraction_success: bool = True,
    relevance_score: int = 4,
) -> dict:
    return {
        "url": url,
        "title": title,
        "content": content,
        "snippet": snippet,
        "extraction_success": extraction_success,
        "relevance_score": relevance_score,
    }


def _terms_json(*terms) -> str:
    """Return a JSON array of term dicts."""
    return json.dumps(list(terms))


def _term_dict(term="SimCLR", category="technique", relevance=5) -> dict:
    return {"term": term, "category": category, "relevance": relevance}


TOPIC = "contrastive learning"


# ---------------------------------------------------------------------------
# extract_terms — happy path
# ---------------------------------------------------------------------------

class TestExtractTerms:
    def test_returns_list_of_dicts(self):
        raw = _terms_json(_term_dict("SimCLR"))
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert isinstance(result, list)
        assert all(isinstance(t, dict) for t in result)

    def test_term_fields_present(self):
        raw = _terms_json(_term_dict("SimCLR"))
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert len(result) == 1
        t = result[0]
        assert set(t.keys()) == {
            "term", "category", "source_url", "source_title", "topic", "first_seen", "last_seen"
        }

    def test_term_value(self):
        raw = _terms_json(_term_dict("MoCo", category="framework", relevance=5))
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result[0]["term"] == "MoCo"
        assert result[0]["category"] == "framework"

    def test_source_url_attached(self):
        raw = _terms_json(_term_dict())
        item = _item(url="https://src.example.com")
        result = extract_terms(item, TOPIC, _mock_client(raw))
        assert result[0]["source_url"] == "https://src.example.com"

    def test_source_title_attached(self):
        raw = _terms_json(_term_dict())
        item = _item(title="My Paper")
        result = extract_terms(item, TOPIC, _mock_client(raw))
        assert result[0]["source_title"] == "My Paper"

    def test_topic_attached(self):
        raw = _terms_json(_term_dict())
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result[0]["topic"] == TOPIC

    def test_first_seen_and_last_seen_are_iso_dates(self):
        from datetime import date
        raw = _terms_json(_term_dict())
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        today = date.today().isoformat()
        assert result[0]["first_seen"] == today
        assert result[0]["last_seen"] == today

    def test_multiple_terms_returned(self):
        raw = _terms_json(
            _term_dict("SimCLR"),
            _term_dict("MoCo", category="framework"),
            _term_dict("Geoffrey Hinton", category="author"),
        )
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert len(result) == 3

    def test_term_whitespace_stripped(self):
        raw = _terms_json({"term": "  SimCLR  ", "category": "technique", "relevance": 5})
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result[0]["term"] == "SimCLR"

    def test_empty_array_from_llm_returns_empty_list(self):
        result = extract_terms(_item(), TOPIC, _mock_client("[]"))
        assert result == []

    def test_uses_content_over_snippet(self):
        """When content is present, it should be used (not tested via LLM call
        content, but verifiable via the generate call being made at all)."""
        raw = _terms_json(_term_dict())
        item = _item(content="Full article text", snippet="Short snippet")
        client = _mock_client(raw)
        extract_terms(item, TOPIC, client)
        # The prompt passed to generate should contain the full content
        prompt_arg = client.generate.call_args[0][0]
        assert "Full article text" in prompt_arg

    def test_falls_back_to_snippet_when_content_empty(self):
        raw = _terms_json(_term_dict())
        item = _item(content="", snippet="Short snippet text")
        client = _mock_client(raw)
        extract_terms(item, TOPIC, client)
        prompt_arg = client.generate.call_args[0][0]
        assert "Short snippet text" in prompt_arg


# ---------------------------------------------------------------------------
# extract_terms — filtering by relevance
# ---------------------------------------------------------------------------

class TestExtractTermsRelevanceFiltering:
    def test_drops_terms_below_min_relevance(self):
        low = _term_dict("generic method", relevance=MIN_TERM_RELEVANCE - 1)
        high = _term_dict("SimCLR", relevance=MIN_TERM_RELEVANCE)
        raw = _terms_json(low, high)
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        terms = [t["term"] for t in result]
        assert "SimCLR" in terms
        assert "generic method" not in terms

    def test_keeps_terms_at_exactly_min_relevance(self):
        raw = _terms_json(_term_dict("BoundaryTerm", relevance=MIN_TERM_RELEVANCE))
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert any(t["term"] == "BoundaryTerm" for t in result)

    def test_drops_entry_with_missing_relevance(self):
        raw = json.dumps([{"term": "NoScore", "category": "keyword"}])
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result == []

    def test_drops_entry_with_non_numeric_relevance(self):
        raw = json.dumps([{"term": "BadScore", "category": "keyword", "relevance": "high"}])
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result == []

    def test_drops_entry_with_missing_term(self):
        raw = json.dumps([{"category": "keyword", "relevance": 5}])
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result == []

    def test_drops_entry_with_empty_term(self):
        raw = json.dumps([{"term": "", "category": "keyword", "relevance": 5}])
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result == []


# ---------------------------------------------------------------------------
# extract_terms — failure / edge cases
# ---------------------------------------------------------------------------

class TestExtractTermsFailures:
    def test_llm_exception_returns_empty_list(self):
        client = MagicMock()
        client.generate.side_effect = RuntimeError("Ollama down")
        result = extract_terms(_item(), TOPIC, client)
        assert result == []

    def test_invalid_json_returns_empty_list(self):
        result = extract_terms(_item(), TOPIC, _mock_client("not json at all"))
        assert result == []

    def test_json_object_instead_of_array_returns_empty_list(self):
        result = extract_terms(_item(), TOPIC, _mock_client('{"term": "x"}'))
        assert result == []

    def test_empty_content_and_snippet_returns_empty_without_llm_call(self):
        client = MagicMock()
        item = _item(content="", snippet="")
        result = extract_terms(item, TOPIC, client)
        assert result == []
        client.generate.assert_not_called()

    def test_non_dict_entries_in_array_are_skipped(self):
        raw = json.dumps(["not a dict", _term_dict("SimCLR"), 42])
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        terms = [t["term"] for t in result]
        assert terms == ["SimCLR"]

    def test_missing_category_defaults_to_keyword(self):
        raw = json.dumps([{"term": "NoCat", "relevance": 5}])
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert result[0]["category"] == "keyword"

    def test_llm_wraps_json_in_markdown_fences(self):
        """extract_json should strip ```json fences before parsing."""
        inner = _terms_json(_term_dict("SimCLR"))
        raw = f"```json\n{inner}\n```"
        result = extract_terms(_item(), TOPIC, _mock_client(raw))
        assert any(t["term"] == "SimCLR" for t in result)

    def test_content_truncated_at_4000_chars(self):
        long_content = "x" * 6000
        item = _item(content=long_content)
        client = _mock_client("[]")
        extract_terms(item, TOPIC, client)
        prompt = client.generate.call_args[0][0]
        # The 4000-char content appears truncated in the prompt
        assert "x" * 4000 in prompt
        assert "x" * 4001 not in prompt


# ---------------------------------------------------------------------------
# extract_terms_batch
# ---------------------------------------------------------------------------

class TestExtractTermsBatch:
    def test_returns_flat_list(self):
        raw = _terms_json(_term_dict("SimCLR"))
        items = [_item(), _item(url="https://b.com")]
        result = extract_terms_batch(items, TOPIC, _mock_client(raw))
        assert isinstance(result, list)

    def test_skips_failed_extractions(self):
        client = MagicMock()
        client.generate.return_value = _terms_json(_term_dict())
        items = [
            _item(extraction_success=False),
            _item(url="https://good.com", extraction_success=True),
        ]
        result = extract_terms_batch(items, TOPIC, client)
        # Only the successful item should trigger a generate call
        assert client.generate.call_count == 1

    def test_skips_low_relevance_items(self):
        client = MagicMock()
        client.generate.return_value = _terms_json(_term_dict())
        items = [
            _item(relevance_score=MIN_RELEVANCE_SCORE - 1),
            _item(url="https://good.com", relevance_score=MIN_RELEVANCE_SCORE),
        ]
        result = extract_terms_batch(items, TOPIC, client)
        assert client.generate.call_count == 1

    def test_keeps_items_at_exactly_min_relevance_score(self):
        client = _mock_client(_terms_json(_term_dict()))
        items = [_item(relevance_score=MIN_RELEVANCE_SCORE)]
        extract_terms_batch(items, TOPIC, client)
        client.generate.assert_called_once()

    def test_aggregates_terms_from_multiple_items(self):
        def generate_side_effect(prompt, system=None):
            if "item-a" in prompt:
                return _terms_json(_term_dict("TermA"))
            return _terms_json(_term_dict("TermB"))

        client = MagicMock()
        client.generate.side_effect = generate_side_effect
        items = [
            _item(url="https://a.com", content="item-a content"),
            _item(url="https://b.com", content="item-b content"),
        ]
        result = extract_terms_batch(items, TOPIC, client)
        term_names = [t["term"] for t in result]
        assert "TermA" in term_names
        assert "TermB" in term_names

    def test_empty_items_list_returns_empty(self):
        result = extract_terms_batch([], TOPIC, _mock_client("[]"))
        assert result == []

    def test_all_items_filtered_out_returns_empty(self):
        client = MagicMock()
        items = [
            _item(extraction_success=False),
            _item(relevance_score=0),
        ]
        result = extract_terms_batch(items, TOPIC, client)
        assert result == []
        client.generate.assert_not_called()

    def test_invalid_relevance_score_treated_as_zero(self):
        client = MagicMock()
        item = _item()
        item["relevance_score"] = "not a number"
        result = extract_terms_batch([item], TOPIC, client)
        assert result == []
        client.generate.assert_not_called()
