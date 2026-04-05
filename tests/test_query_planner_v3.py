"""
tests/test_query_planner_v3.py — Unit tests for the v3 additions to
redpill.query_planner: decompose_topic(), synthesize_queries(), and the
two-stage plan_queries() path via PlannerLLMClient.

All LLM calls are mocked.  No network or Ollama access occurs.
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from redpill.query_planner import (
    _parse_research_plan,
    _save_fallback_plan,
    decompose_topic,
    plan_queries,
    plan_queries_fallback,
    synthesize_queries,
)
from redpill.state import (
    get_latest_research_plan_conn,
    init_db_conn,
    store_extracted_terms_conn,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

TOPIC = "contrastive learning"


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db_conn(c)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def conn_with_terms(conn) -> sqlite3.Connection:
    from datetime import date
    today = date.today().isoformat()
    terms = [
        {"term": "SimCLR", "topic": TOPIC, "category": "technique",
         "first_seen": today, "last_seen": today},
        {"term": "MoCo", "topic": TOPIC, "category": "framework",
         "first_seen": today, "last_seen": today},
    ]
    store_extracted_terms_conn(terms, conn)
    conn.commit()
    return conn


def _make_planner_client(response: str, thinking: str | None = None) -> MagicMock:
    """Return a mock PlannerLLMClient that generate() returns *response*."""
    from redpill.summarize import PlannerLLMClient
    client = MagicMock(spec=PlannerLLMClient)
    client.generate.return_value = response
    client.last_thinking = thinking
    return client


def _make_standard_client(response: str) -> MagicMock:
    """Return a mock standard LLMClient."""
    client = MagicMock()
    client.generate.return_value = response
    return client


_VALID_PLAN_JSON = json.dumps({
    "dimensions": [
        {
            "name": "hard negative mining",
            "description": "Informative negative sample strategies.",
            "priority": "high",
            "coverage": "under-explored",
            "suggested_queries": ["hard negative mining contrastive 2026", "negative sampling strategies"],
        },
        {
            "name": "benchmark comparisons",
            "description": "Latest benchmark results.",
            "priority": "medium",
            "coverage": "partially-covered",
            "suggested_queries": ["contrastive learning benchmark ImageNet 2026"],
        },
    ],
    "dropped_dimensions": [{"name": "basic tutorials", "reason": "well-covered"}],
    "new_directions": ["Audio contrastive learning is emerging."],
})


# ---------------------------------------------------------------------------
# _parse_research_plan
# ---------------------------------------------------------------------------

class TestParseResearchPlan:
    def test_valid_plan_parsed(self):
        result = _parse_research_plan(_VALID_PLAN_JSON)
        assert result is not None
        assert "dimensions" in result

    def test_dimensions_preserved(self):
        result = _parse_research_plan(_VALID_PLAN_JSON)
        assert len(result["dimensions"]) == 2

    def test_invalid_json_returns_none(self):
        assert _parse_research_plan("not json") is None

    def test_json_array_returns_none(self):
        assert _parse_research_plan("[1, 2, 3]") is None

    def test_missing_dimensions_key_returns_none(self):
        assert _parse_research_plan('{"other": []}') is None

    def test_empty_dimensions_list_returns_none(self):
        result = _parse_research_plan(json.dumps({"dimensions": []}))
        assert result is None

    def test_non_list_dimensions_returns_none(self):
        result = _parse_research_plan(json.dumps({"dimensions": "not a list"}))
        assert result is None

    def test_extra_keys_preserved(self):
        plan = {"dimensions": [{"name": "x"}], "dropped_dimensions": [], "new_directions": ["y"]}
        result = _parse_research_plan(json.dumps(plan))
        assert "dropped_dimensions" in result
        assert "new_directions" in result

    def test_think_block_stripped(self):
        inner = json.dumps({"dimensions": [{"name": "x"}]})
        raw = f"<think>reasoning here</think>{inner}"
        result = _parse_research_plan(raw)
        assert result is not None

    def test_markdown_fences_stripped(self):
        inner = json.dumps({"dimensions": [{"name": "x"}]})
        raw = f"```json\n{inner}\n```"
        result = _parse_research_plan(raw)
        assert result is not None


# ---------------------------------------------------------------------------
# decompose_topic
# ---------------------------------------------------------------------------

class TestDecomposeTopic:
    def test_returns_plan_dict(self, conn):
        client = _make_planner_client(_VALID_PLAN_JSON)
        result = decompose_topic(TOPIC, conn, client)
        assert isinstance(result, dict)
        assert "dimensions" in result

    def test_generates_called_at_least_once(self, conn):
        # decompose_topic calls generate at least once for the plan itself.
        # It also calls generate_topic_scaffold (which calls generate once more
        # on a cold cache), so the total count is >= 1.
        client = _make_planner_client(_VALID_PLAN_JSON)
        decompose_topic(TOPIC, conn, client)
        assert client.generate.call_count >= 1

    def test_prompt_contains_topic(self, conn):
        client = _make_planner_client(_VALID_PLAN_JSON)
        decompose_topic(TOPIC, conn, client)
        prompt_arg = client.generate.call_args[0][0]
        assert TOPIC in prompt_arg

    def test_prompt_contains_date(self, conn):
        from datetime import date
        client = _make_planner_client(_VALID_PLAN_JSON)
        decompose_topic(TOPIC, conn, client)
        prompt_arg = client.generate.call_args[0][0]
        today = date.today().isoformat()
        assert today in prompt_arg

    def test_prompt_contains_terms_when_available(self, conn_with_terms):
        client = _make_planner_client(_VALID_PLAN_JSON)
        decompose_topic(TOPIC, conn_with_terms, client)
        prompt_arg = client.generate.call_args[0][0]
        assert "SimCLR" in prompt_arg

    def test_raises_on_llm_error(self, conn):
        client = _make_planner_client("")
        client.generate.side_effect = RuntimeError("Ollama down")
        with pytest.raises(RuntimeError):
            decompose_topic(TOPIC, conn, client)

    def test_raises_on_invalid_plan(self, conn):
        client = _make_planner_client("not valid json")
        with pytest.raises(RuntimeError):
            decompose_topic(TOPIC, conn, client)

    def test_uses_previous_plan_when_available(self, conn):
        from redpill.state import save_research_plan_conn
        from datetime import date
        prev_plan = {"dimensions": [{"name": "prior dimension"}]}
        save_research_plan_conn(TOPIC, date.today().isoformat(), prev_plan, conn)
        conn.commit()

        client = _make_planner_client(_VALID_PLAN_JSON)
        decompose_topic(TOPIC, conn, client)
        prompt_arg = client.generate.call_args[0][0]
        assert "prior dimension" in prompt_arg

    def test_shows_no_previous_plan_message_on_first_run(self, conn):
        client = _make_planner_client(_VALID_PLAN_JSON)
        decompose_topic(TOPIC, conn, client)
        prompt_arg = client.generate.call_args[0][0]
        assert "first run" in prompt_arg


# ---------------------------------------------------------------------------
# synthesize_queries
# ---------------------------------------------------------------------------

class TestSynthesizeQueries:
    def _make_plan(self, dimensions: list, new_directions: list | None = None) -> dict:
        return {
            "dimensions": dimensions,
            "dropped_dimensions": [],
            "new_directions": new_directions or [],
        }

    def test_returns_list_of_dicts(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["query one"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert isinstance(result, list)
        assert all(isinstance(q, dict) for q in result)

    def test_source_is_llm_planned(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["query one"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert all(q["source"] == "llm_planned" for q in result)

    def test_max_queries_respected(self):
        # 3 dimensions, 2 queries each = 6 candidates, but budget=2 (max_queries=3 - 1 for base)
        plan = self._make_plan([
            {"name": f"d{i}", "priority": "high", "coverage": "under-explored",
             "suggested_queries": [f"query {i}a", f"query {i}b"]}
            for i in range(3)
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert len(result) <= 2

    def test_base_topic_not_duplicated(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored",
             "suggested_queries": [TOPIC]},  # same as base topic
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        # The base topic duplicate must be dropped
        assert all(q["query"].lower() != TOPIC.lower() for q in result)

    def test_high_under_explored_before_medium(self):
        plan = self._make_plan([
            {"name": "medium", "priority": "medium", "coverage": "under-explored",
             "suggested_queries": ["medium query"]},
            {"name": "high", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["high query"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert len(result) >= 1
        # First result should come from the high priority dimension
        assert result[0]["query"] == "high query"

    def test_duplicate_suggested_queries_dropped(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["duplicate query", "duplicate query"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=5)
        queries = [q["query"] for q in result]
        assert queries.count("duplicate query") == 1

    def test_empty_dimensions_returns_empty(self):
        plan = self._make_plan([])
        result = synthesize_queries(plan, TOPIC, max_queries=5)
        assert result == []

    def test_max_queries_one_returns_empty(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["some query"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=1)
        assert result == []

    def test_non_string_suggested_queries_skipped(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored",
             "suggested_queries": [None, 42, "valid query"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert len(result) == 1
        assert result[0]["query"] == "valid query"

    def test_new_directions_included_when_budget_allows(self):
        # Only 1 dimension query fills the first slot; new_directions fills the next
        plan = self._make_plan(
            dimensions=[
                {"name": "d1", "priority": "high", "coverage": "under-explored",
                 "suggested_queries": ["first query"]},
            ],
            new_directions=['"audio contrastive learning"'],
        )
        result = synthesize_queries(plan, TOPIC, max_queries=4)
        queries = [q["query"] for q in result]
        assert "audio contrastive learning" in queries

    def test_reasoning_contains_dimension_name(self):
        plan = self._make_plan([
            {"name": "hard negatives", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["hard negative mining 2026"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert "hard negatives" in result[0]["reasoning"]

    def test_missing_suggested_queries_key_skipped(self):
        plan = self._make_plan([
            {"name": "d1", "priority": "high", "coverage": "under-explored"},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=3)
        assert result == []

    def test_well_covered_low_dimension_included_last(self):
        plan = self._make_plan([
            {"name": "well covered low", "priority": "low", "coverage": "well-covered",
             "suggested_queries": ["low query"]},
            {"name": "high priority", "priority": "high", "coverage": "under-explored",
             "suggested_queries": ["high query"]},
        ])
        result = synthesize_queries(plan, TOPIC, max_queries=4)
        assert result[0]["query"] == "high query"


# ---------------------------------------------------------------------------
# plan_queries — two-stage path (PlannerLLMClient)
# ---------------------------------------------------------------------------

class TestPlanQueriesTwoStage:
    def test_uses_two_stage_when_planner_client(self, conn_with_terms):
        """When a PlannerLLMClient is passed, decompose_topic() should be called."""
        client = _make_planner_client(_VALID_PLAN_JSON)
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        # decompose_topic calls generate() — verify it was called
        assert client.generate.call_count >= 1
        assert result[0]["source"] == "base"

    def test_first_query_always_base_topic(self, conn_with_terms):
        client = _make_planner_client(_VALID_PLAN_JSON)
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        assert result[0]["query"] == TOPIC
        assert result[0]["source"] == "base"

    def test_plan_saved_to_db_on_success(self, conn_with_terms):
        client = _make_planner_client(_VALID_PLAN_JSON, thinking="I thought.")
        plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        conn_with_terms.commit()
        row = get_latest_research_plan_conn(TOPIC, conn_with_terms)
        assert row is not None
        assert row["source"] == "llm"

    def test_reasoning_trace_saved_when_present(self, conn_with_terms):
        client = _make_planner_client(_VALID_PLAN_JSON, thinking="Detailed reasoning.")
        plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        conn_with_terms.commit()
        row = get_latest_research_plan_conn(TOPIC, conn_with_terms)
        assert row is not None
        assert row["reasoning_trace"] == "Detailed reasoning."

    def test_two_stage_falls_back_on_llm_error(self, conn_with_terms):
        client = _make_planner_client("")
        client.generate.side_effect = RuntimeError("Ollama down")
        # Should not raise — falls back to deterministic or single-stage
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        assert result[0]["source"] == "base"

    def test_two_stage_falls_back_on_bad_json(self, conn_with_terms):
        client = _make_planner_client("garbage output")
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=4)
        assert result[0]["source"] == "base"
        sources = {q["source"] for q in result}
        # No llm_planned queries from the two-stage path when plan is invalid
        assert "llm_planned" not in sources

    def test_max_queries_respected(self, conn_with_terms):
        client = _make_planner_client(_VALID_PLAN_JSON)
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=3)
        assert len(result) <= 3

    def test_standard_client_does_not_trigger_two_stage(self, conn_with_terms):
        """A standard MagicMock without PlannerLLMClient spec goes through the
        single-stage path, not the two-stage path."""
        llm_response = json.dumps([
            {"query": f"{TOPIC} SimCLR benchmark", "reasoning": "SimCLR is frequent."},
        ])
        client = _make_standard_client(llm_response)
        result = plan_queries(TOPIC, conn_with_terms, client, max_queries=3)
        # Single-stage: LLM queries present
        sources = {q["source"] for q in result}
        assert "llm_planned" in sources


# ---------------------------------------------------------------------------
# _save_fallback_plan
# ---------------------------------------------------------------------------

class TestSaveFallbackPlan:
    def test_saves_plan_with_fallback_source(self, conn):
        from datetime import date
        _save_fallback_plan(TOPIC, date.today().isoformat(), conn)
        conn.commit()
        row = get_latest_research_plan_conn(TOPIC, conn)
        assert row is not None
        assert row["source"] == "fallback"

    def test_plan_json_is_valid(self, conn):
        from datetime import date
        _save_fallback_plan(TOPIC, date.today().isoformat(), conn)
        conn.commit()
        row = get_latest_research_plan_conn(TOPIC, conn)
        parsed = json.loads(row["plan_json"])
        assert "dimensions" in parsed

    def test_does_not_raise_on_db_error(self, conn):
        """_save_fallback_plan must be resilient — close the conn to force an error."""
        conn.close()
        # Should not raise even with a closed connection
        try:
            _save_fallback_plan(TOPIC, "2026-03-24", conn)
        except Exception:
            pytest.fail("_save_fallback_plan raised unexpectedly on DB error")


# ---------------------------------------------------------------------------
# Integration: plan_queries fallback saves plan record
# ---------------------------------------------------------------------------

class TestPlanQueriesFallbackSavesPlan:
    def test_fallback_plan_saved_when_no_terms(self, conn):
        """When no term history exists, plan_queries falls back and should
        save a fallback plan record for auditability."""
        standard_client = _make_standard_client("[]")
        result = plan_queries(TOPIC, conn, standard_client, max_queries=3)
        conn.commit()
        row = get_latest_research_plan_conn(TOPIC, conn)
        # A fallback plan record should exist after the run
        assert row is not None
        assert row["source"] == "fallback"

    def test_base_query_always_first_in_fallback(self, conn):
        standard_client = _make_standard_client("[]")
        result = plan_queries(TOPIC, conn, standard_client, max_queries=3)
        assert result[0]["query"] == TOPIC
        assert result[0]["source"] == "base"
