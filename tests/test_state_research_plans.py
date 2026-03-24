"""
tests/test_state_research_plans.py — Unit tests for the research_plans table
and its helper methods in redpill.state.

All tests use an in-memory SQLite connection — no disk I/O occurs.
"""

import json
import sqlite3
from datetime import date, timedelta

import pytest

from redpill.state import (
    get_latest_research_plan_conn,
    init_db_conn,
    save_research_plan_conn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn() -> sqlite3.Connection:
    """Fresh in-memory SQLite connection with all tables initialised."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db_conn(c)
    c.commit()
    yield c
    c.close()


TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()
TOPIC = "contrastive learning"

_SAMPLE_PLAN = {
    "dimensions": [
        {
            "name": "hard negative mining",
            "description": "Methods for selecting informative negatives.",
            "priority": "high",
            "coverage": "under-explored",
            "suggested_queries": ["hard negative mining contrastive 2026"],
        },
        {
            "name": "self-supervised benchmarks",
            "description": "Latest benchmark comparisons.",
            "priority": "medium",
            "coverage": "partially-covered",
            "suggested_queries": ["self-supervised learning benchmark results"],
        },
    ],
    "dropped_dimensions": [],
    "new_directions": ["Emerging interest in contrastive learning for audio."],
}


# ---------------------------------------------------------------------------
# init_db_conn — research_plans table
# ---------------------------------------------------------------------------

class TestInitDbResearchPlans:
    def test_creates_research_plans_table(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        row = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='research_plans'"
        ).fetchone()
        assert row is not None

    def test_research_plans_columns(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        info = c.execute("PRAGMA table_info(research_plans)").fetchall()
        cols = {row[1] for row in info}
        assert cols == {"id", "topic", "run_date", "plan_json", "reasoning_trace", "source"}

    def test_idempotent_second_call(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        init_db_conn(c)  # must not raise


# ---------------------------------------------------------------------------
# save_research_plan_conn
# ---------------------------------------------------------------------------

class TestSaveResearchPlan:
    def test_returns_integer_id(self, conn):
        row_id = save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_ids_are_unique(self, conn):
        id1 = save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        id2 = save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        assert id1 != id2

    def test_plan_json_stored_correctly(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        row = conn.execute("SELECT plan_json FROM research_plans").fetchone()
        assert row is not None
        parsed = json.loads(row["plan_json"])
        assert parsed == _SAMPLE_PLAN

    def test_topic_stored_correctly(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        row = conn.execute("SELECT topic FROM research_plans").fetchone()
        assert row["topic"] == TOPIC

    def test_run_date_stored_correctly(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        row = conn.execute("SELECT run_date FROM research_plans").fetchone()
        assert row["run_date"] == TODAY

    def test_default_source_is_llm(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        row = conn.execute("SELECT source FROM research_plans").fetchone()
        assert row["source"] == "llm"

    def test_source_fallback_stored(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn, source="fallback")
        row = conn.execute("SELECT source FROM research_plans").fetchone()
        assert row["source"] == "fallback"

    def test_reasoning_trace_stored(self, conn):
        trace = "I thought about this deeply."
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn, reasoning_trace=trace)
        row = conn.execute("SELECT reasoning_trace FROM research_plans").fetchone()
        assert row["reasoning_trace"] == trace

    def test_reasoning_trace_null_by_default(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        row = conn.execute("SELECT reasoning_trace FROM research_plans").fetchone()
        assert row["reasoning_trace"] is None

    def test_multiple_plans_for_same_topic_allowed(self, conn):
        save_research_plan_conn(TOPIC, YESTERDAY, _SAMPLE_PLAN, conn)
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        count = conn.execute("SELECT COUNT(*) FROM research_plans").fetchone()[0]
        assert count == 2

    def test_unicode_in_plan_json_preserved(self, conn):
        plan_with_unicode = {"dimensions": [{"name": "café research", "emoji": "✓"}]}
        save_research_plan_conn(TOPIC, TODAY, plan_with_unicode, conn)
        row = conn.execute("SELECT plan_json FROM research_plans").fetchone()
        parsed = json.loads(row["plan_json"])
        assert parsed["dimensions"][0]["name"] == "café research"


# ---------------------------------------------------------------------------
# get_latest_research_plan_conn
# ---------------------------------------------------------------------------

class TestGetLatestResearchPlan:
    def test_returns_none_when_no_plans(self, conn):
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert result is None

    def test_returns_dict_when_plan_exists(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert isinstance(result, dict)

    def test_result_has_expected_keys(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert set(result.keys()) == {
            "id", "topic", "run_date", "plan_json", "reasoning_trace", "source"
        }

    def test_plan_json_is_string(self, conn):
        """plan_json is stored as a JSON string — callers must deserialize it."""
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn)
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert isinstance(result["plan_json"], str)
        # But it must be valid JSON
        parsed = json.loads(result["plan_json"])
        assert parsed == _SAMPLE_PLAN

    def test_returns_most_recent_plan(self, conn):
        old_plan = {"dimensions": [{"name": "old", "priority": "low"}]}
        new_plan = {"dimensions": [{"name": "new", "priority": "high"}]}
        save_research_plan_conn(TOPIC, YESTERDAY, old_plan, conn)
        save_research_plan_conn(TOPIC, TODAY, new_plan, conn)
        conn.commit()
        result = get_latest_research_plan_conn(TOPIC, conn)
        parsed = json.loads(result["plan_json"])
        assert parsed["dimensions"][0]["name"] == "new"

    def test_filters_by_topic(self, conn):
        save_research_plan_conn("other topic", TODAY, {"dimensions": []}, conn)
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert result is None

    def test_reasoning_trace_returned(self, conn):
        trace = "I reasoned about this."
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn, reasoning_trace=trace)
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert result["reasoning_trace"] == trace

    def test_source_returned(self, conn):
        save_research_plan_conn(TOPIC, TODAY, _SAMPLE_PLAN, conn, source="fallback")
        result = get_latest_research_plan_conn(TOPIC, conn)
        assert result["source"] == "fallback"

    def test_latest_by_id_when_same_date(self, conn):
        """When two plans share the same run_date, the higher id wins."""
        plan_a = {"dimensions": [{"name": "first"}]}
        plan_b = {"dimensions": [{"name": "second"}]}
        save_research_plan_conn(TOPIC, TODAY, plan_a, conn)
        save_research_plan_conn(TOPIC, TODAY, plan_b, conn)
        conn.commit()
        result = get_latest_research_plan_conn(TOPIC, conn)
        parsed = json.loads(result["plan_json"])
        assert parsed["dimensions"][0]["name"] == "second"
