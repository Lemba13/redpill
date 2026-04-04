"""
tests/test_state_v2.py — Tests for the v2 state.py additions.

Covers: extracted_terms table, query_log table, and all new *_conn methods.
All tests use an in-memory SQLite connection; no disk I/O occurs.
"""

import sqlite3
from datetime import date, timedelta

import pytest

from redpill.state import (
    get_query_performance_conn,
    get_recent_terms_conn,
    get_top_terms_conn,
    init_db_conn,
    log_query_conn,
    store_extracted_terms_conn,
    update_query_stats_conn,
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
LAST_WEEK = (date.today() - timedelta(days=7)).isoformat()
LONG_AGO = "2020-01-01"


def _term(
    term: str = "SimCLR",
    topic: str = "contrastive learning",
    category: str = "technique",
    source_url: str | None = "https://example.com/paper",
    source_title: str | None = "SimCLR Paper",
    first_seen: str = TODAY,
    last_seen: str = TODAY,
) -> dict:
    return {
        "term": term,
        "topic": topic,
        "category": category,
        "source_url": source_url,
        "source_title": source_title,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


# ---------------------------------------------------------------------------
# init_db_conn — new tables
# ---------------------------------------------------------------------------

class TestInitDbNewTables:
    def test_creates_extracted_terms_table(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        row = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extracted_terms'"
        ).fetchone()
        assert row is not None

    def test_creates_query_log_table(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        row = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'"
        ).fetchone()
        assert row is not None

    def test_extracted_terms_columns(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        info = c.execute("PRAGMA table_info(extracted_terms)").fetchall()
        cols = {row[1] for row in info}
        assert cols == {
            "id", "term", "source_url", "source_title", "topic",
            "category", "first_seen", "frequency", "last_seen",
        }

    def test_query_log_columns(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        info = c.execute("PRAGMA table_info(query_log)").fetchall()
        cols = {row[1] for row in info}
        assert cols == {
            "id", "query_text", "run_date", "source", "topic",
            "results_count", "new_items", "kept_items", "dim_id",
        }

    def test_idempotent_on_existing_db(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        init_db_conn(c)  # must not raise

    def test_seen_items_still_created(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        row = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='seen_items'"
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# store_extracted_terms_conn
# ---------------------------------------------------------------------------

class TestStoreExtractedTerms:
    def test_inserts_new_term(self, conn):
        store_extracted_terms_conn([_term()], conn)
        row = conn.execute("SELECT * FROM extracted_terms WHERE term = 'SimCLR'").fetchone()
        assert row is not None
        assert row["frequency"] == 1
        assert row["topic"] == "contrastive learning"
        assert row["category"] == "technique"

    def test_upsert_increments_frequency(self, conn):
        store_extracted_terms_conn([_term()], conn)
        store_extracted_terms_conn([_term(last_seen=TODAY)], conn)
        row = conn.execute("SELECT frequency FROM extracted_terms WHERE term = 'SimCLR'").fetchone()
        assert row["frequency"] == 2

    def test_upsert_updates_last_seen(self, conn):
        store_extracted_terms_conn([_term(last_seen=YESTERDAY)], conn)
        store_extracted_terms_conn([_term(last_seen=TODAY)], conn)
        row = conn.execute("SELECT last_seen FROM extracted_terms WHERE term = 'SimCLR'").fetchone()
        assert row["last_seen"] == TODAY

    def test_first_seen_not_overwritten_on_upsert(self, conn):
        store_extracted_terms_conn([_term(first_seen=YESTERDAY)], conn)
        store_extracted_terms_conn([_term(first_seen=TODAY)], conn)
        row = conn.execute("SELECT first_seen FROM extracted_terms WHERE term = 'SimCLR'").fetchone()
        # first_seen was set on first insert — upsert must not change it
        assert row["first_seen"] == YESTERDAY

    def test_same_term_different_topic_creates_separate_rows(self, conn):
        store_extracted_terms_conn([_term(topic="topic-a")], conn)
        store_extracted_terms_conn([_term(topic="topic-b")], conn)
        count = conn.execute("SELECT COUNT(*) FROM extracted_terms").fetchone()[0]
        assert count == 2

    def test_source_url_preserved_when_new_is_null(self, conn):
        store_extracted_terms_conn([_term(source_url="https://original.com")], conn)
        store_extracted_terms_conn([_term(source_url=None)], conn)
        row = conn.execute("SELECT source_url FROM extracted_terms WHERE term = 'SimCLR'").fetchone()
        assert row["source_url"] == "https://original.com"

    def test_multiple_terms_inserted(self, conn):
        terms = [_term("A"), _term("B"), _term("C")]
        store_extracted_terms_conn(terms, conn)
        count = conn.execute("SELECT COUNT(*) FROM extracted_terms").fetchone()[0]
        assert count == 3

    def test_term_missing_topic_is_skipped(self, conn):
        bad = {"term": "orphan", "first_seen": TODAY, "last_seen": TODAY}
        store_extracted_terms_conn([bad], conn)
        count = conn.execute("SELECT COUNT(*) FROM extracted_terms").fetchone()[0]
        assert count == 0

    def test_empty_list_is_noop(self, conn):
        store_extracted_terms_conn([], conn)
        count = conn.execute("SELECT COUNT(*) FROM extracted_terms").fetchone()[0]
        assert count == 0

    def test_frequency_accumulates_over_many_upserts(self, conn):
        for _ in range(5):
            store_extracted_terms_conn([_term()], conn)
        row = conn.execute("SELECT frequency FROM extracted_terms WHERE term = 'SimCLR'").fetchone()
        assert row["frequency"] == 5


# ---------------------------------------------------------------------------
# get_recent_terms_conn
# ---------------------------------------------------------------------------

class TestGetRecentTerms:
    def _populate(self, conn):
        store_extracted_terms_conn([_term("Recent", last_seen=TODAY)], conn)
        store_extracted_terms_conn([_term("Old", last_seen=LONG_AGO, first_seen=LONG_AGO)], conn)

    def test_returns_recent_terms(self, conn):
        self._populate(conn)
        results = get_recent_terms_conn("contrastive learning", 30, conn)
        terms = [r["term"] for r in results]
        assert "Recent" in terms

    def test_excludes_old_terms(self, conn):
        self._populate(conn)
        results = get_recent_terms_conn("contrastive learning", 30, conn)
        terms = [r["term"] for r in results]
        assert "Old" not in terms

    def test_zero_days_returns_only_today(self, conn):
        store_extracted_terms_conn([_term("Today", last_seen=TODAY)], conn)
        store_extracted_terms_conn([_term("Yday", last_seen=YESTERDAY, first_seen=YESTERDAY)], conn)
        results = get_recent_terms_conn("contrastive learning", 0, conn)
        terms = [r["term"] for r in results]
        assert "Today" in terms
        assert "Yday" not in terms

    def test_filters_by_topic(self, conn):
        store_extracted_terms_conn([_term("A", topic="topic-a")], conn)
        store_extracted_terms_conn([_term("B", topic="topic-b")], conn)
        results = get_recent_terms_conn("topic-a", 30, conn)
        assert all(r["topic"] == "topic-a" for r in results)

    def test_sorted_by_frequency_desc(self, conn):
        store_extracted_terms_conn([_term("Low")], conn)
        # Give "High" a higher frequency by upserting 3 times
        for _ in range(3):
            store_extracted_terms_conn([_term("High")], conn)
        results = get_recent_terms_conn("contrastive learning", 30, conn)
        assert results[0]["term"] == "High"

    def test_empty_db_returns_empty_list(self, conn):
        assert get_recent_terms_conn("contrastive learning", 30, conn) == []

    def test_result_dicts_have_expected_keys(self, conn):
        store_extracted_terms_conn([_term()], conn)
        results = get_recent_terms_conn("contrastive learning", 30, conn)
        assert len(results) == 1
        assert set(results[0].keys()) == {
            "id", "term", "source_url", "source_title", "topic",
            "category", "first_seen", "frequency", "last_seen",
        }


# ---------------------------------------------------------------------------
# get_top_terms_conn
# ---------------------------------------------------------------------------

class TestGetTopTerms:
    def _populate(self, conn):
        """Insert 5 terms with different frequencies."""
        for i, name in enumerate(["E", "D", "C", "B", "A"], start=1):
            for _ in range(i):
                store_extracted_terms_conn([_term(name)], conn)

    def test_returns_terms_sorted_by_frequency(self, conn):
        self._populate(conn)
        results = get_top_terms_conn("contrastive learning", 5, conn)
        freqs = [r["frequency"] for r in results]
        assert freqs == sorted(freqs, reverse=True)

    def test_limit_is_respected(self, conn):
        self._populate(conn)
        results = get_top_terms_conn("contrastive learning", 3, conn)
        assert len(results) == 3

    def test_filters_by_topic(self, conn):
        store_extracted_terms_conn([_term("X", topic="topic-x")], conn)
        store_extracted_terms_conn([_term("Y", topic="topic-y")], conn)
        results = get_top_terms_conn("topic-x", 50, conn)
        assert all(r["topic"] == "topic-x" for r in results)

    def test_empty_db_returns_empty_list(self, conn):
        assert get_top_terms_conn("contrastive learning", 10, conn) == []

    def test_includes_old_terms_unlike_recent(self, conn):
        store_extracted_terms_conn([_term("Ancient", last_seen=LONG_AGO, first_seen=LONG_AGO)], conn)
        results = get_top_terms_conn("contrastive learning", 50, conn)
        assert any(r["term"] == "Ancient" for r in results)


# ---------------------------------------------------------------------------
# log_query_conn
# ---------------------------------------------------------------------------

class TestLogQuery:
    def test_returns_integer_id(self, conn):
        qid = log_query_conn("contrastive learning", TODAY, "base", "contrastive learning", conn)
        assert isinstance(qid, int)
        assert qid > 0

    def test_ids_are_unique(self, conn):
        id1 = log_query_conn("query one", TODAY, "base", "topic", conn)
        id2 = log_query_conn("query two", TODAY, "llm_planned", "topic", conn)
        assert id1 != id2

    def test_row_stored_correctly(self, conn):
        qid = log_query_conn("test query", TODAY, "extracted_term", "ml", conn)
        row = conn.execute("SELECT * FROM query_log WHERE id = ?", (qid,)).fetchone()
        assert row["query_text"] == "test query"
        assert row["run_date"] == TODAY
        assert row["source"] == "extracted_term"
        assert row["topic"] == "ml"
        assert row["results_count"] == 0
        assert row["new_items"] == 0
        assert row["kept_items"] == 0

    def test_multiple_queries_same_day(self, conn):
        ids = [
            log_query_conn(f"query {i}", TODAY, "base", "topic", conn)
            for i in range(3)
        ]
        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# update_query_stats_conn
# ---------------------------------------------------------------------------

class TestUpdateQueryStats:
    def test_updates_all_stat_fields(self, conn):
        qid = log_query_conn("q", TODAY, "base", "topic", conn)
        update_query_stats_conn(qid, results_count=10, new_items=5, kept_items=3, conn=conn)
        row = conn.execute("SELECT * FROM query_log WHERE id = ?", (qid,)).fetchone()
        assert row["results_count"] == 10
        assert row["new_items"] == 5
        assert row["kept_items"] == 3

    def test_zero_stats(self, conn):
        qid = log_query_conn("q", TODAY, "base", "topic", conn)
        update_query_stats_conn(qid, 0, 0, 0, conn)
        row = conn.execute("SELECT * FROM query_log WHERE id = ?", (qid,)).fetchone()
        assert row["results_count"] == 0
        assert row["new_items"] == 0
        assert row["kept_items"] == 0

    def test_does_not_affect_other_rows(self, conn):
        id1 = log_query_conn("q1", TODAY, "base", "topic", conn)
        id2 = log_query_conn("q2", TODAY, "base", "topic", conn)
        update_query_stats_conn(id1, 10, 5, 2, conn)
        row2 = conn.execute("SELECT * FROM query_log WHERE id = ?", (id2,)).fetchone()
        assert row2["results_count"] == 0


# ---------------------------------------------------------------------------
# get_query_performance_conn
# ---------------------------------------------------------------------------

class TestGetQueryPerformance:
    def _log_and_update(self, conn, query, run_date, results, new, kept, topic="topic"):
        qid = log_query_conn(query, run_date, "base", topic, conn)
        update_query_stats_conn(qid, results, new, kept, conn)
        return qid

    def test_returns_entries_within_days(self, conn):
        self._log_and_update(conn, "recent query", TODAY, 10, 5, 3)
        results = get_query_performance_conn("topic", 14, conn)
        assert any(r["query_text"] == "recent query" for r in results)

    def test_excludes_old_entries(self, conn):
        self._log_and_update(conn, "old query", LONG_AGO, 10, 5, 3)
        results = get_query_performance_conn("topic", 14, conn)
        assert not any(r["query_text"] == "old query" for r in results)

    def test_filters_by_topic(self, conn):
        self._log_and_update(conn, "q-a", TODAY, 5, 2, 1, topic="topic-a")
        self._log_and_update(conn, "q-b", TODAY, 5, 2, 1, topic="topic-b")
        results = get_query_performance_conn("topic-a", 14, conn)
        assert all(r["topic"] == "topic-a" for r in results)

    def test_ordered_most_recent_first(self, conn):
        self._log_and_update(conn, "old", YESTERDAY, 1, 0, 0)
        self._log_and_update(conn, "new", TODAY, 1, 0, 0)
        results = get_query_performance_conn("topic", 14, conn)
        dates = [r["run_date"] for r in results]
        assert dates == sorted(dates, reverse=True)

    def test_empty_db_returns_empty_list(self, conn):
        assert get_query_performance_conn("topic", 14, conn) == []

    def test_result_dicts_have_expected_keys(self, conn):
        self._log_and_update(conn, "q", TODAY, 5, 2, 1)
        results = get_query_performance_conn("topic", 14, conn)
        assert len(results) == 1
        assert set(results[0].keys()) == {
            "id", "query_text", "run_date", "source", "topic",
            "results_count", "new_items", "kept_items", "dim_id",
        }

    def test_stat_values_correct(self, conn):
        self._log_and_update(conn, "q", TODAY, 15, 8, 4)
        results = get_query_performance_conn("topic", 14, conn)
        r = results[0]
        assert r["results_count"] == 15
        assert r["new_items"] == 8
        assert r["kept_items"] == 4
