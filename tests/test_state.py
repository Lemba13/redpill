"""
tests/test_state.py — Unit tests for redpill.state

All tests use an in-memory SQLite connection; no disk I/O occurs.
The internal *_conn() functions are tested directly so that a single
sqlite3.connect(":memory:") connection is shared across calls within a test.
"""

import sqlite3
import struct

import numpy as np
import pytest

from redpill.state import (
    _deserialize_embedding,
    _serialize_embedding,
    add_item_conn,
    get_all_embeddings_conn,
    get_items_since_conn,
    init_db_conn,
    is_url_seen_conn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn() -> sqlite3.Connection:
    """Fresh in-memory SQLite connection with seen_items table initialised."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db_conn(c)
    c.commit()
    yield c
    c.close()


def _make_embedding(values: list[float] | None = None, dtype: str = "float32") -> np.ndarray:
    """Build a small test embedding array."""
    if values is None:
        values = [0.1, 0.2, 0.3, 0.4]
    return np.array(values, dtype=dtype)


def _add(
    conn: sqlite3.Connection,
    url: str = "https://example.com/article",
    title: str = "Test Article",
    content_hash: str = "abc123",
    embedding: np.ndarray | None = None,
    summary: str = "A test summary.",
    topic: str = "ml",
    first_seen_date: str = "2026-03-01",
) -> None:
    """Convenience wrapper around add_item_conn with sensible defaults."""
    if embedding is None:
        embedding = _make_embedding()
    add_item_conn(
        url, title, content_hash, embedding, summary, topic, conn,
        first_seen_date=first_seen_date,
    )


# ---------------------------------------------------------------------------
# Embedding serialization round-trips
# ---------------------------------------------------------------------------

class TestEmbeddingSerialization:
    def test_float32_roundtrip(self):
        arr = _make_embedding([0.1, 0.2, 0.3], dtype="float32")
        blob = _serialize_embedding(arr)
        result = _deserialize_embedding(blob)
        np.testing.assert_array_equal(arr, result)
        assert result.dtype == arr.dtype

    def test_float64_roundtrip(self):
        arr = _make_embedding([1.0, 2.0, 3.0], dtype="float64")
        blob = _serialize_embedding(arr)
        result = _deserialize_embedding(blob)
        np.testing.assert_array_equal(arr, result)
        assert result.dtype == np.float64

    def test_2d_array_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        blob = _serialize_embedding(arr)
        result = _deserialize_embedding(blob)
        np.testing.assert_array_equal(arr, result)
        assert result.shape == (2, 2)

    def test_1d_384_dim_array(self):
        """Sentence-transformer typical output size."""
        arr = np.random.default_rng(42).random(384).astype("float32")
        blob = _serialize_embedding(arr)
        result = _deserialize_embedding(blob)
        np.testing.assert_array_almost_equal(arr, result, decimal=6)

    def test_dtype_is_preserved_not_coerced(self):
        """Deserialization must not silently upcast float32 to float64."""
        arr = np.array([1.0, 2.0], dtype="float32")
        blob = _serialize_embedding(arr)
        result = _deserialize_embedding(blob)
        assert result.dtype == np.float32

    def test_blob_is_bytes(self):
        arr = _make_embedding()
        blob = _serialize_embedding(arr)
        assert isinstance(blob, bytes)

    def test_corrupted_blob_raises(self):
        with pytest.raises((struct.error, Exception)):
            _deserialize_embedding(b"\x00\x00")


# ---------------------------------------------------------------------------
# init_db_conn
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_seen_items_table(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        row = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='seen_items'"
        ).fetchone()
        assert row is not None

    def test_idempotent_second_call_does_not_raise(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        # Should not raise due to CREATE TABLE IF NOT EXISTS
        init_db_conn(c)

    def test_table_has_expected_columns(self):
        c = sqlite3.connect(":memory:")
        init_db_conn(c)
        info = c.execute("PRAGMA table_info(seen_items)").fetchall()
        column_names = {row[1] for row in info}
        assert column_names == {
            "id", "url", "title", "content_hash",
            "embedding", "summary", "first_seen_date", "topic", "dim_id",
        }


# ---------------------------------------------------------------------------
# is_url_seen_conn
# ---------------------------------------------------------------------------

class TestIsUrlSeen:
    def test_returns_false_for_unknown_url(self, conn):
        assert is_url_seen_conn("https://never-seen.com", conn) is False

    def test_returns_true_after_item_added(self, conn):
        _add(conn, url="https://seen.com")
        assert is_url_seen_conn("https://seen.com", conn) is True

    def test_url_match_is_exact_not_prefix(self, conn):
        _add(conn, url="https://example.com/article")
        assert is_url_seen_conn("https://example.com", conn) is False
        assert is_url_seen_conn("https://example.com/article/extra", conn) is False

    def test_url_match_is_case_sensitive(self, conn):
        _add(conn, url="https://example.com/Article")
        assert is_url_seen_conn("https://example.com/article", conn) is False

    def test_empty_url_returns_false_when_not_stored(self, conn):
        assert is_url_seen_conn("", conn) is False


# ---------------------------------------------------------------------------
# add_item_conn
# ---------------------------------------------------------------------------

class TestAddItem:
    def test_item_is_retrievable_after_insert(self, conn):
        _add(conn, url="https://a.com", title="A", first_seen_date="2026-03-01")
        row = conn.execute(
            "SELECT url, title FROM seen_items WHERE url = ?", ("https://a.com",)
        ).fetchone()
        assert row is not None
        assert row["title"] == "A"

    def test_all_fields_stored_correctly(self, conn):
        emb = _make_embedding([0.5, 0.6])
        _add(
            conn,
            url="https://b.com",
            title="B Title",
            content_hash="deadbeef",
            embedding=emb,
            summary="B summary",
            topic="nlp",
            first_seen_date="2026-02-15",
        )
        row = conn.execute(
            "SELECT * FROM seen_items WHERE url = ?", ("https://b.com",)
        ).fetchone()
        assert row["title"] == "B Title"
        assert row["content_hash"] == "deadbeef"
        assert row["summary"] == "B summary"
        assert row["topic"] == "nlp"
        assert row["first_seen_date"] == "2026-02-15"

    def test_embedding_survives_roundtrip_through_db(self, conn):
        emb = _make_embedding([0.1, 0.2, 0.3, 0.4])
        _add(conn, url="https://c.com", embedding=emb)
        blob = conn.execute(
            "SELECT embedding FROM seen_items WHERE url = ?", ("https://c.com",)
        ).fetchone()["embedding"]
        from redpill.state import _deserialize_embedding
        recovered = _deserialize_embedding(blob)
        np.testing.assert_array_almost_equal(emb, recovered, decimal=6)

    def test_duplicate_url_is_ignored_not_raised(self, conn):
        _add(conn, url="https://dup.com", title="First")
        # Second insert should silently do nothing (INSERT OR IGNORE)
        _add(conn, url="https://dup.com", title="Second")
        rows = conn.execute(
            "SELECT title FROM seen_items WHERE url = ?", ("https://dup.com",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["title"] == "First"

    def test_multiple_distinct_urls_all_stored(self, conn):
        for i in range(5):
            _add(conn, url=f"https://example.com/{i}")
        count = conn.execute("SELECT COUNT(*) FROM seen_items").fetchone()[0]
        assert count == 5

    def test_first_seen_date_defaults_to_today(self, conn):
        from datetime import date
        # Explicitly pass first_seen_date=None to exercise the default path.
        add_item_conn(
            "https://today.com",
            "T",
            "h",
            _make_embedding(),
            "s",
            "t",
            conn,
            first_seen_date=None,
        )
        row = conn.execute(
            "SELECT first_seen_date FROM seen_items WHERE url = ?",
            ("https://today.com",),
        ).fetchone()
        assert row["first_seen_date"] == date.today().isoformat()


# ---------------------------------------------------------------------------
# get_all_embeddings_conn
# ---------------------------------------------------------------------------

class TestGetAllEmbeddings:
    def test_empty_db_returns_empty_list(self, conn):
        result = get_all_embeddings_conn(conn)
        assert result == []

    def test_returns_id_and_array_tuple(self, conn):
        emb = _make_embedding([1.0, 2.0])
        _add(conn, url="https://e1.com", embedding=emb)
        result = get_all_embeddings_conn(conn)
        assert len(result) == 1
        item_id, arr = result[0]
        assert isinstance(item_id, int)
        assert isinstance(arr, np.ndarray)

    def test_embedding_values_match(self, conn):
        emb = _make_embedding([0.11, 0.22, 0.33])
        _add(conn, url="https://e2.com", embedding=emb)
        _, arr = get_all_embeddings_conn(conn)[0]
        np.testing.assert_array_almost_equal(emb, arr, decimal=6)

    def test_returns_all_embeddings(self, conn):
        for i in range(3):
            _add(conn, url=f"https://emb{i}.com", embedding=_make_embedding([float(i)]))
        result = get_all_embeddings_conn(conn)
        assert len(result) == 3

    def test_ids_are_unique(self, conn):
        for i in range(4):
            _add(conn, url=f"https://uid{i}.com")
        result = get_all_embeddings_conn(conn)
        ids = [r[0] for r in result]
        assert len(ids) == len(set(ids))

    def test_null_embedding_rows_are_skipped(self, conn):
        """A row with NULL embedding should not appear in the output."""
        conn.execute(
            """
            INSERT INTO seen_items (url, title, content_hash, summary, first_seen_date, topic)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("https://no-emb.com", "T", "h", "s", "2026-03-01", "t"),
        )
        conn.commit()
        result = get_all_embeddings_conn(conn)
        assert result == []


# ---------------------------------------------------------------------------
# get_items_since_conn
# ---------------------------------------------------------------------------

class TestGetItemsSince:
    def _populate(self, conn: sqlite3.Connection) -> None:
        """Insert three items with different dates."""
        _add(conn, url="https://jan.com", first_seen_date="2026-01-15")
        _add(conn, url="https://feb.com", first_seen_date="2026-02-20")
        _add(conn, url="https://mar.com", first_seen_date="2026-03-01")

    def test_returns_items_on_or_after_date(self, conn):
        self._populate(conn)
        results = get_items_since_conn("2026-02-20", conn)
        urls = {r["url"] for r in results}
        assert urls == {"https://feb.com", "https://mar.com"}

    def test_excludes_items_before_date(self, conn):
        self._populate(conn)
        results = get_items_since_conn("2026-02-20", conn)
        assert all(r["url"] != "https://jan.com" for r in results)

    def test_exact_date_match_is_inclusive(self, conn):
        _add(conn, url="https://exact.com", first_seen_date="2026-03-04")
        results = get_items_since_conn("2026-03-04", conn)
        assert any(r["url"] == "https://exact.com" for r in results)

    def test_empty_db_returns_empty_list(self, conn):
        assert get_items_since_conn("2026-01-01", conn) == []

    def test_future_date_returns_empty(self, conn):
        self._populate(conn)
        results = get_items_since_conn("2099-01-01", conn)
        assert results == []

    def test_result_dicts_have_expected_keys(self, conn):
        _add(conn, url="https://keys.com", first_seen_date="2026-03-01")
        results = get_items_since_conn("2026-03-01", conn)
        assert len(results) == 1
        assert set(results[0].keys()) == {
            "id", "url", "title", "content_hash", "summary", "first_seen_date", "topic", "dim_id"
        }

    def test_embedding_blob_not_exposed(self, conn):
        """Raw embedding bytes should NOT appear in the returned dicts."""
        _add(conn, url="https://noblob.com", first_seen_date="2026-03-01")
        results = get_items_since_conn("2026-03-01", conn)
        assert "embedding" not in results[0]

    def test_results_ordered_by_date_then_id(self, conn):
        # Insert out of order
        _add(conn, url="https://z.com", first_seen_date="2026-03-02")
        _add(conn, url="https://a.com", first_seen_date="2026-03-01")
        _add(conn, url="https://m.com", first_seen_date="2026-03-01")
        results = get_items_since_conn("2026-03-01", conn)
        dates = [r["first_seen_date"] for r in results]
        assert dates == sorted(dates)
        # Items on the same date should be ordered by id (insertion order)
        same_date = [r for r in results if r["first_seen_date"] == "2026-03-01"]
        assert same_date[0]["id"] < same_date[1]["id"]

    def test_returned_values_match_inserted_data(self, conn):
        emb = _make_embedding()
        _add(
            conn,
            url="https://data-check.com",
            title="My Title",
            content_hash="cafebabe",
            embedding=emb,
            summary="My summary",
            topic="robotics",
            first_seen_date="2026-03-03",
        )
        results = get_items_since_conn("2026-03-03", conn)
        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://data-check.com"
        assert r["title"] == "My Title"
        assert r["content_hash"] == "cafebabe"
        assert r["summary"] == "My summary"
        assert r["topic"] == "robotics"
        assert r["first_seen_date"] == "2026-03-03"
