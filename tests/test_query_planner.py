"""
tests/test_query_planner.py — Unit tests for redpill.query_planner

All LLM calls and DB interactions are controlled. No disk I/O or Ollama
access occurs — the *_conn functions from state.py are called directly
via in-memory SQLite connections.
"""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from redpill.query_planner import (
    _base_query,
    _build_planner_prompt,
    _parse_llm_queries,
    plan_queries,
    plan_queries_fallback,
)
from redpill.state import init_db_conn, store_extracted_terms_conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with all tables initialised."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db_conn(c)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def conn_with_terms(conn) -> sqlite3.Connection:
    """DB pre-populated with a handful of extracted terms."""
    from datetime import date
    today = date.today().isoformat()
    terms = [
        {"term": "SimCLR", "topic": TOPIC, "category": "technique",
         "first_seen": today, "last_seen": today},
        {"term": "MoCo", "topic": TOPIC, "category": "technique",
         "first_seen": today, "last_seen": today},
        {"term": "Geoffrey Hinton", "topic": TOPIC, "category": "author",
         "first_seen": today, "last_seen": today},
    ]
    # Give SimCLR the highest frequency via repeated upserts
    for _ in range(3):
        store_extracted_terms_conn(
            [{"term": "SimCLR", "topic": TOPIC, "category": "technique",
              "first_seen": today, "last_seen": today}],
            conn,
        )
    store_extracted_terms_conn(terms[1:], conn)
    conn.commit()
    return conn


TOPIC = "contrastive learning"


def _mock_client(response: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = response
    return client


def _llm_queries_json(*queries) -> str:
    return json.dumps(list(queries))


def _llm_query(query="contrastive learning SimCLR", reasoning="Targets SimCLR technique.") -> dict:
    return {"query": query, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# _base_query
# ---------------------------------------------------------------------------

class TestBaseQuery:
    def test_query_is_topic(self):
        q = _base_query(TOPIC)
        assert q["query"] == TOPIC

    def test_source_is_base(self):
        assert _base_query(TOPIC)["source"] == "base"

    def test_has_reasoning(self):
        assert _base_query(TOPIC)["reasoning"]


# ---------------------------------------------------------------------------
# _build_planner_prompt
# ---------------------------------------------------------------------------

class TestBuildPlannerPrompt:
    def test_contains_topic(self):
        prompt = _build_planner_prompt(TOPIC, [], 3)
        assert TOPIC in prompt

    def test_contains_n_queries_count(self):
        prompt = _build_planner_prompt(TOPIC, [], 4)
        assert "4" in prompt

    def test_no_terms_shows_placeholder(self):
        prompt = _build_planner_prompt(TOPIC, [], 3)
        assert "No term history" in prompt

    def test_terms_appear_in_prompt(self):
        from datetime import date
        today = date.today().isoformat()
        terms = [{"term": "SimCLR", "category": "technique", "frequency": 3,
                   "first_seen": today, "last_seen": today}]
        prompt = _build_planner_prompt(TOPIC, terms, 3)
        assert "SimCLR" in prompt

    def test_term_frequency_shown(self):
        from datetime import date
        today = date.today().isoformat()
        terms = [{"term": "MoCo", "category": "framework", "frequency": 5,
                   "first_seen": today, "last_seen": today}]
        prompt = _build_planner_prompt(TOPIC, terms, 2)
        assert "5" in prompt


# ---------------------------------------------------------------------------
# _parse_llm_queries
# ---------------------------------------------------------------------------

class TestParseLlmQueries:
    def test_valid_response_parsed(self):
        raw = _llm_queries_json(_llm_query("contrastive learning SimCLR"))
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert len(result) == 1
        assert result[0]["query"] == "contrastive learning SimCLR"

    def test_source_set_to_llm_planned(self):
        raw = _llm_queries_json(_llm_query("contrastive learning MoCo"))
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert result[0]["source"] == "llm_planned"

    def test_reasoning_preserved(self):
        raw = _llm_queries_json(_llm_query(reasoning="Good reason."))
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert result[0]["reasoning"] == "Good reason."

    def test_base_topic_duplicate_dropped(self):
        raw = _llm_queries_json(_llm_query(query=TOPIC))
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert result == []

    def test_base_topic_case_insensitive_dropped(self):
        raw = _llm_queries_json(_llm_query(query=TOPIC.upper()))
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert result == []

    def test_duplicate_queries_deduplicated(self):
        raw = _llm_queries_json(
            _llm_query("contrastive learning SimCLR"),
            _llm_query("contrastive learning SimCLR"),
        )
        result = _parse_llm_queries(raw, TOPIC, 3)
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        result = _parse_llm_queries("not json", TOPIC, 2)
        assert result == []

    def test_json_object_not_array_returns_empty(self):
        result = _parse_llm_queries('{"query": "x"}', TOPIC, 2)
        assert result == []

    def test_entry_missing_query_skipped(self):
        raw = json.dumps([{"reasoning": "No query field."}])
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert result == []

    def test_non_dict_entries_skipped(self):
        raw = json.dumps(["not a dict", _llm_query("contrastive learning MoCo")])
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert len(result) == 1

    def test_empty_array_returns_empty(self):
        result = _parse_llm_queries("[]", TOPIC, 2)
        assert result == []

    def test_whitespace_stripped_from_query(self):
        raw = _llm_queries_json({"query": "  contrastive learning SimCLR  ", "reasoning": "ok"})
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert result[0]["query"] == "contrastive learning SimCLR"

    def test_think_block_stripped(self):
        inner = _llm_queries_json(_llm_query("contrastive learning SimCLR"))
        raw = f"<think>let me think</think>{inner}"
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert len(result) == 1

    def test_markdown_fences_stripped(self):
        inner = _llm_queries_json(_llm_query("contrastive learning SimCLR"))
        raw = f"```json\n{inner}\n```"
        result = _parse_llm_queries(raw, TOPIC, 2)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# plan_queries_fallback
# ---------------------------------------------------------------------------

class TestPlanQueriesFallback:
    def test_first_query_is_base_topic(self, conn):
        result = plan_queries_fallback(TOPIC, conn, max_queries=3)
        assert result[0]["query"] == TOPIC
        assert result[0]["source"] == "base"

    def test_empty_db_returns_only_base(self, conn):
        result = plan_queries_fallback(TOPIC, conn, max_queries=5)
        assert len(result) == 1
        assert result[0]["source"] == "base"

    def test_terms_expand_into_queries(self, conn_with_terms):
        result = plan_queries_fallback(TOPIC, conn_with_terms, max_queries=4)
        sources = [q["source"] for q in result]
        assert "extracted_term" in sources

    def test_query_format_is_topic_plus_term(self, conn_with_terms):
        result = plan_queries_fallback(TOPIC, conn_with_terms, max_queries=4)
        expanded = [q for q in result if q["source"] == "extracted_term"]
        for q in expanded:
            assert q["query"].startswith(TOPIC)

    def test_max_queries_respected(self, conn_with_terms):
        result = plan_queries_fallback(TOPIC, conn_with_terms, max_queries=2)
        assert len(result) == 2

    def test_max_queries_one_returns_only_base(self, conn_with_terms):
        result = plan_queries_fallback(TOPIC, conn_with_terms, max_queries=1)
        assert len(result) == 1
        assert result[0]["source"] == "base"

    def test_source_is_extracted_term(self, conn_with_terms):
        result = plan_queries_fallback(TOPIC, conn_with_terms, max_queries=5)
        for q in result[1:]:
            assert q["source"] == "extracted_term"

    def test_reasoning_mentions_term(self, conn_with_terms):
        result = plan_queries_fallback(TOPIC, conn_with_terms, max_queries=3)
        expanded = [q for q in result if q["source"] == "extracted_term"]
        assert all(q["reasoning"] for q in expanded)

    def test_filters_by_topic(self, conn):
        from datetime import date
        today = date.today().isoformat()
        store_extracted_terms_conn(
            [{"term": "OtherTopic term", "topic": "other topic",
              "first_seen": today, "last_seen": today}],
            conn,
        )
        result = plan_queries_fallback(TOPIC, conn, max_queries=5)
        # Only base query — no terms for this topic
        assert len(result) == 1


# ---------------------------------------------------------------------------
# plan_queries — LLM path
# ---------------------------------------------------------------------------

class TestPlanQueriesLlm:
    def test_first_query_always_base_topic(self, conn_with_terms):
        raw = _llm_queries_json(_llm_query("contrastive learning SimCLR"))
        result = plan_queries(TOPIC, conn_with_terms, _mock_client(raw), max_queries=3)
        assert result[0]["query"] == TOPIC
        assert result[0]["source"] == "base"

    def test_llm_queries_follow_base(self, conn_with_terms):
        raw = _llm_queries_json(
            _llm_query("contrastive learning SimCLR"),
            _llm_query("contrastive learning MoCo benchmark"),
        )
        result = plan_queries(TOPIC, conn_with_terms, _mock_client(raw), max_queries=3)
        sources = [q["source"] for q in result]
        assert sources[0] == "base"
        assert all(s == "llm_planned" for s in sources[1:])

    def test_max_queries_respected(self, conn_with_terms):
        raw = _llm_queries_json(*[_llm_query(f"query {i}") for i in range(10)])
        result = plan_queries(TOPIC, conn_with_terms, _mock_client(raw), max_queries=3)
        assert len(result) == 3

    def test_llm_failure_falls_back(self, conn_with_terms):
        client = MagicMock()
        client.generate.side_effect = RuntimeError("Ollama down")
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        # fallback: base + extracted_term queries
        sources = {q["source"] for q in result}
        assert "base" in sources
        assert "llm_planned" not in sources

    def test_bad_llm_json_falls_back(self, conn_with_terms):
        result = plan_queries(TOPIC, conn_with_terms, _mock_client("not json"), max_queries=4)
        sources = {q["source"] for q in result}
        assert "llm_planned" not in sources

    def test_no_term_history_uses_fallback_without_llm_call(self, conn):
        client = MagicMock()
        result = plan_queries(TOPIC, conn, client, max_queries=3)
        # No terms → fallback path → no LLM call
        client.generate.assert_not_called()
        assert result[0]["source"] == "base"

    def test_max_queries_one_returns_only_base(self, conn_with_terms):
        client = MagicMock()
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=1)
        assert len(result) == 1
        client.generate.assert_not_called()

    def test_llm_returning_empty_array_falls_back(self, conn_with_terms):
        result = plan_queries(TOPIC, conn_with_terms, _mock_client("[]"), max_queries=4)
        sources = {q["source"] for q in result}
        assert "llm_planned" not in sources

    def test_llm_query_duplicating_base_dropped(self, conn_with_terms):
        raw = _llm_queries_json(_llm_query(query=TOPIC))
        result = plan_queries(TOPIC, conn_with_terms, _mock_client(raw), max_queries=3)
        # Only the base query remains — LLM's duplicate is dropped, triggers fallback
        sources = [q["source"] for q in result]
        assert "llm_planned" not in sources
