"""
tests/test_dedup.py — Unit tests for redpill.dedup

No real embedding model is ever loaded. All tests patch redpill.dedup._get_model
to return a mock whose .encode() returns a controlled numpy array.

Test structure:
    TestComputeEmbedding        — compute_embedding(): truncation + dtype
    TestIsSemanticDuplicate     — pure similarity logic with hand-crafted vectors
    TestFilterNewItems          — full pipeline using in-memory SQLite + mocked model
"""

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from redpill.dedup import compute_embedding, filter_new_items, is_semantic_duplicate
from redpill.state import add_item_conn, init_db_conn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _unit_vec(*components: float) -> np.ndarray:
    """Return a float32 unit vector with the given components."""
    arr = np.array(components, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _fake_model(return_vec: np.ndarray) -> MagicMock:
    """Build a mock SentenceTransformer whose encode() always returns return_vec."""
    m = MagicMock()
    m.encode.return_value = return_vec
    return m


def _in_memory_conn() -> sqlite3.Connection:
    """Open an in-memory SQLite connection with the seen_items table ready."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db_conn(conn)
    conn.commit()
    return conn


def _seed_item(
    conn: sqlite3.Connection,
    url: str,
    embedding: np.ndarray,
    title: str = "Seeded Article",
) -> None:
    """Insert one item into the DB with the given URL and embedding."""
    add_item_conn(
        url=url,
        title=title,
        content_hash="hash",
        embedding=embedding,
        summary="summary",
        topic="test",
        conn=conn,
        first_seen_date="2026-01-01",
    )
    conn.commit()


def _make_candidate(
    url: str = "https://example.com/new",
    content: str | None = "some article content about machine learning",
    snippet: str = "ML snippet",
) -> dict:
    return {"url": url, "title": "Test", "content": content, "snippet": snippet}


# ---------------------------------------------------------------------------
# compute_embedding
# ---------------------------------------------------------------------------


class TestComputeEmbedding:
    def test_returns_float32_array(self):
        """Output must be float32 regardless of what the model returns."""
        mock_model = _fake_model(np.array([0.1, 0.2, 0.3], dtype=np.float64))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            result = compute_embedding("hello world")
        assert result.dtype == np.float32

    def test_returns_ndarray(self):
        mock_model = _fake_model(np.array([1.0, 2.0], dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            result = compute_embedding("hello")
        assert isinstance(result, np.ndarray)

    def test_model_encode_called_with_text(self):
        """compute_embedding must forward the (possibly truncated) text to encode()."""
        mock_model = _fake_model(np.zeros(4, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            compute_embedding("hello world")
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == "hello world"

    def test_text_under_512_tokens_passed_unchanged(self):
        """Short text must reach encode() verbatim."""
        short_text = " ".join(f"word{i}" for i in range(100))
        mock_model = _fake_model(np.zeros(4, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            compute_embedding(short_text)
        actual_text = mock_model.encode.call_args[0][0]
        assert actual_text == short_text

    def test_text_over_512_tokens_is_truncated(self):
        """Text with more than 512 whitespace-delimited tokens must be truncated."""
        long_text = " ".join(f"token{i}" for i in range(600))
        mock_model = _fake_model(np.zeros(4, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            compute_embedding(long_text)
        actual_text = mock_model.encode.call_args[0][0]
        actual_tokens = actual_text.split()
        assert len(actual_tokens) == 512

    def test_truncation_keeps_first_512_tokens(self):
        """Truncation must keep the *first* 512 tokens, not the last."""
        tokens = [f"t{i}" for i in range(600)]
        long_text = " ".join(tokens)
        mock_model = _fake_model(np.zeros(4, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            compute_embedding(long_text)
        actual_text = mock_model.encode.call_args[0][0]
        assert actual_text == " ".join(tokens[:512])

    def test_exactly_512_tokens_not_truncated(self):
        """Exactly 512 tokens should be passed through unchanged."""
        text_512 = " ".join(f"w{i}" for i in range(512))
        mock_model = _fake_model(np.zeros(4, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            compute_embedding(text_512)
        actual_text = mock_model.encode.call_args[0][0]
        assert len(actual_text.split()) == 512

    def test_model_called_with_convert_to_numpy(self):
        """encode() must be called with convert_to_numpy=True."""
        mock_model = _fake_model(np.zeros(2, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            compute_embedding("test")
        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("convert_to_numpy") is True


# ---------------------------------------------------------------------------
# is_semantic_duplicate
# ---------------------------------------------------------------------------


class TestIsSemanticDuplicate:
    def test_empty_existing_returns_not_dup(self):
        emb = _unit_vec(1.0, 0.0)
        is_dup, closest_id, score = is_semantic_duplicate(emb, [], threshold=0.85)
        assert is_dup is False
        assert closest_id is None
        assert score is None

    def test_identical_vector_is_duplicate(self):
        """Cosine similarity of 1.0 must exceed any reasonable threshold."""
        v = _unit_vec(1.0, 0.0, 0.0)
        existing = [(1, v.copy())]
        is_dup, closest_id, score = is_semantic_duplicate(v, existing, threshold=0.85)
        assert is_dup is True
        assert closest_id == 1
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_are_not_duplicate(self):
        """Cosine similarity of 0.0 must not exceed a positive threshold."""
        v1 = _unit_vec(1.0, 0.0)
        v2 = _unit_vec(0.0, 1.0)
        existing = [(42, v2)]
        is_dup, closest_id, score = is_semantic_duplicate(v1, existing, threshold=0.85)
        assert is_dup is False
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_returns_closest_id_of_highest_similarity(self):
        """When multiple existing embeddings are present, closest_id must be the best match."""
        candidate = _unit_vec(1.0, 0.0, 0.0)
        near = _unit_vec(0.99, 0.1, 0.0)   # high similarity
        far = _unit_vec(0.0, 1.0, 0.0)     # low similarity
        existing = [(10, far), (20, near)]
        _, closest_id, score = is_semantic_duplicate(candidate, existing, threshold=0.99)
        assert closest_id == 20
        assert score > 0.9

    def test_threshold_boundary_below_is_not_dup(self):
        """Score exactly below threshold: not a duplicate."""
        # Build two vectors with a known similarity just under 0.85.
        # cos(angle) = v1·v2. We want ~0.84.
        import math
        angle = math.acos(0.84)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        existing = [(1, v2)]
        is_dup, _, score = is_semantic_duplicate(v1, existing, threshold=0.85)
        assert is_dup is False
        assert score == pytest.approx(0.84, abs=1e-4)

    def test_threshold_boundary_at_is_dup(self):
        """Score exactly at threshold must be treated as a duplicate (>=)."""
        import math
        angle = math.acos(0.85)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        existing = [(1, v2)]
        is_dup, _, score = is_semantic_duplicate(v1, existing, threshold=0.85)
        assert is_dup is True
        assert score == pytest.approx(0.85, abs=1e-4)

    def test_zero_candidate_vector_is_not_dup(self):
        """A zero embedding has undefined cosine similarity; must not crash."""
        zero = np.zeros(4, dtype=np.float32)
        existing = [(1, _unit_vec(1.0, 0.0, 0.0, 0.0))]
        is_dup, closest_id, score = is_semantic_duplicate(zero, existing, threshold=0.5)
        assert is_dup is False
        assert closest_id is None
        assert score is None

    def test_all_existing_zero_vectors_returns_not_dup(self):
        """If every existing embedding is a zero vector, treat as no comparison."""
        candidate = _unit_vec(1.0, 0.0)
        existing = [(1, np.zeros(2, dtype=np.float32)), (2, np.zeros(2, dtype=np.float32))]
        is_dup, closest_id, score = is_semantic_duplicate(candidate, existing, threshold=0.5)
        assert is_dup is False
        assert closest_id is None
        assert score is None

    def test_return_types_are_correct(self):
        v = _unit_vec(1.0, 0.0)
        existing = [(7, v.copy())]
        is_dup, closest_id, score = is_semantic_duplicate(v, existing, threshold=0.5)
        assert isinstance(is_dup, bool)
        assert isinstance(closest_id, int)
        assert isinstance(score, float)

    def test_negative_similarity_not_dup(self):
        """Antiparallel vectors have cosine similarity -1; must never be a dup."""
        v1 = _unit_vec(1.0, 0.0)
        v2 = _unit_vec(-1.0, 0.0)
        existing = [(1, v2)]
        is_dup, _, score = is_semantic_duplicate(v1, existing, threshold=0.0)
        # threshold=0.0: score (-1.0) < 0.0, so NOT a dup
        assert is_dup is False
        assert score == pytest.approx(-1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# filter_new_items — uses in-memory SQLite + mocked _get_model
# ---------------------------------------------------------------------------


class TestFilterNewItems:
    """
    Strategy: filter_new_items opens its own sqlite3.connect(db_path) connection.
    We use a real on-disk temp file (via tmp_path fixture) pre-seeded via a
    separate in-memory-style connection. This is simpler than mocking sqlite3
    and gives us real DB behaviour.

    The embedding model is patched throughout.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @pytest.fixture()
    def db_path(self, tmp_path) -> str:
        """Return path to a fresh SQLite file with seen_items table."""
        path = str(tmp_path / "test.db")
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        init_db_conn(conn)
        conn.commit()
        conn.close()
        return path

    @pytest.fixture()
    def seeded_db(self, tmp_path) -> tuple[str, np.ndarray]:
        """Return (db_path, existing_embedding) with one item already stored."""
        path = str(tmp_path / "seeded.db")
        existing_emb = _unit_vec(1.0, 0.0, 0.0, 0.0)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        init_db_conn(conn)
        _seed_item(conn, "https://already-seen.com", existing_emb)
        conn.close()
        return path, existing_emb

    # ------------------------------------------------------------------
    # Empty / trivial cases
    # ------------------------------------------------------------------

    def test_empty_candidates_returns_empty(self, db_path):
        with patch("redpill.dedup._get_model", return_value=_fake_model(np.zeros(4, dtype=np.float32))):
            result = filter_new_items([], db_path, threshold=0.85)
        assert result == []

    # ------------------------------------------------------------------
    # Pass 1: URL exact match
    # ------------------------------------------------------------------

    def test_url_seen_candidate_is_dropped(self, seeded_db):
        db_path, _ = seeded_db
        candidate = _make_candidate(url="https://already-seen.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(np.zeros(4, dtype=np.float32))):
            result = filter_new_items([candidate], db_path, threshold=0.85)
        assert result == []

    def test_url_unseen_candidate_survives_pass1(self, seeded_db):
        """A new URL must not be dropped by the URL pass."""
        db_path, existing_emb = seeded_db
        # Give the new candidate a different embedding so it passes Pass 2 too.
        new_emb = _unit_vec(0.0, 1.0, 0.0, 0.0)  # orthogonal to existing
        candidate = _make_candidate(url="https://brand-new.com")
        mock_model = _fake_model(new_emb)
        with patch("redpill.dedup._get_model", return_value=mock_model):
            result = filter_new_items([candidate], db_path, threshold=0.85)
        assert len(result) == 1
        assert result[0]["url"] == "https://brand-new.com"

    def test_mix_of_seen_and_unseen_urls(self, seeded_db):
        """Seen URL is dropped; unseen URL passes (given low similarity)."""
        db_path, existing_emb = seeded_db
        new_emb = _unit_vec(0.0, 1.0, 0.0, 0.0)
        candidates = [
            _make_candidate(url="https://already-seen.com"),
            _make_candidate(url="https://fresh.com"),
        ]
        with patch("redpill.dedup._get_model", return_value=_fake_model(new_emb)):
            result = filter_new_items(candidates, db_path, threshold=0.85)
        assert len(result) == 1
        assert result[0]["url"] == "https://fresh.com"

    # ------------------------------------------------------------------
    # Pass 2: Semantic similarity
    # ------------------------------------------------------------------

    def test_high_similarity_candidate_is_dropped(self, seeded_db):
        """Candidate embedding nearly identical to an existing one must be dropped."""
        db_path, existing_emb = seeded_db
        # near-duplicate: cosine similarity ~0.9999
        near_dup_emb = existing_emb + np.array([0.001, 0.0, 0.0, 0.0], dtype=np.float32)
        candidate = _make_candidate(url="https://new-but-similar.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(near_dup_emb)):
            result = filter_new_items([candidate], db_path, threshold=0.85)
        assert result == []

    def test_low_similarity_candidate_is_kept(self, seeded_db):
        """Candidate embedding very different from existing must be kept."""
        db_path, existing_emb = seeded_db
        orthogonal_emb = _unit_vec(0.0, 1.0, 0.0, 0.0)
        candidate = _make_candidate(url="https://genuinely-new.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(orthogonal_emb)):
            result = filter_new_items([candidate], db_path, threshold=0.85)
        assert len(result) == 1

    def test_no_existing_embeddings_all_candidates_kept(self, db_path):
        """When the DB is empty, every candidate passes both checks."""
        emb = _unit_vec(1.0, 0.0)
        candidates = [
            _make_candidate(url="https://a.com"),
            _make_candidate(url="https://b.com"),
        ]
        with patch("redpill.dedup._get_model", return_value=_fake_model(emb)):
            result = filter_new_items(candidates, db_path, threshold=0.85)
        assert len(result) == 2

    # ------------------------------------------------------------------
    # Text fallback: content vs snippet
    # ------------------------------------------------------------------

    def test_snippet_used_when_content_is_none(self, db_path):
        """When content is None, compute_embedding must receive the snippet."""
        emb = _unit_vec(1.0, 0.0)
        candidate = {"url": "https://no-content.com", "title": "T", "content": None, "snippet": "the snippet text"}
        mock_model = _fake_model(emb)
        with patch("redpill.dedup._get_model", return_value=mock_model):
            filter_new_items([candidate], db_path, threshold=0.85)
        mock_model.encode.assert_called_once()
        encoded_text = mock_model.encode.call_args[0][0]
        assert encoded_text == "the snippet text"

    def test_content_preferred_over_snippet(self, db_path):
        """When content is non-empty, it must be used instead of snippet."""
        emb = _unit_vec(1.0, 0.0)
        candidate = {"url": "https://has-content.com", "title": "T", "content": "full article body", "snippet": "the snippet"}
        mock_model = _fake_model(emb)
        with patch("redpill.dedup._get_model", return_value=mock_model):
            filter_new_items([candidate], db_path, threshold=0.85)
        encoded_text = mock_model.encode.call_args[0][0]
        assert encoded_text == "full article body"

    def test_snippet_used_when_content_is_empty_string(self, db_path):
        """Empty string content counts as absent; snippet should be used."""
        emb = _unit_vec(1.0, 0.0)
        candidate = {"url": "https://empty-content.com", "title": "T", "content": "", "snippet": "fallback snippet"}
        mock_model = _fake_model(emb)
        with patch("redpill.dedup._get_model", return_value=mock_model):
            filter_new_items([candidate], db_path, threshold=0.85)
        encoded_text = mock_model.encode.call_args[0][0]
        assert encoded_text == "fallback snippet"

    def test_item_with_no_text_is_excluded_with_warning(self, db_path, caplog):
        """An item with neither content nor snippet must be excluded and warned about."""
        import logging
        candidate = {"url": "https://no-text.com", "title": "T", "content": None, "snippet": ""}
        emb = _unit_vec(1.0, 0.0)
        with patch("redpill.dedup._get_model", return_value=_fake_model(emb)):
            with caplog.at_level(logging.WARNING, logger="redpill.dedup"):
                result = filter_new_items([candidate], db_path, threshold=0.85)
        assert result == []
        assert any("no text" in record.message.lower() or "SKIPPED" in record.message for record in caplog.records)

    def test_item_with_whitespace_only_content_and_snippet_excluded(self, db_path):
        """Whitespace-only content and snippet should be treated as absent."""
        candidate = {"url": "https://ws.com", "title": "T", "content": "   ", "snippet": "  "}
        emb = _unit_vec(1.0, 0.0)
        with patch("redpill.dedup._get_model", return_value=_fake_model(emb)):
            result = filter_new_items([candidate], db_path, threshold=0.85)
        assert result == []

    # ------------------------------------------------------------------
    # Logging decisions
    # ------------------------------------------------------------------

    def test_kept_item_logs_kept(self, db_path, caplog):
        import logging
        emb = _unit_vec(1.0, 0.0)
        candidate = _make_candidate(url="https://new.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(emb)):
            with caplog.at_level(logging.INFO, logger="redpill.dedup"):
                filter_new_items([candidate], db_path, threshold=0.85)
        assert any("KEPT" in r.message for r in caplog.records)

    def test_url_dropped_item_logs_dropped_url_match(self, seeded_db, caplog):
        import logging
        db_path, _ = seeded_db
        candidate = _make_candidate(url="https://already-seen.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(np.zeros(4, dtype=np.float32))):
            with caplog.at_level(logging.INFO, logger="redpill.dedup"):
                filter_new_items([candidate], db_path, threshold=0.85)
        assert any("url_match" in r.message for r in caplog.records)

    def test_semantic_dropped_item_logs_score(self, seeded_db, caplog):
        import logging
        db_path, existing_emb = seeded_db
        near_dup = existing_emb.copy()
        candidate = _make_candidate(url="https://near-dup.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(near_dup)):
            with caplog.at_level(logging.INFO, logger="redpill.dedup"):
                filter_new_items([candidate], db_path, threshold=0.85)
        assert any("semantic" in r.message and "score" in r.message for r in caplog.records)

    # ------------------------------------------------------------------
    # Return value shape
    # ------------------------------------------------------------------

    def test_returned_dicts_are_original_candidate_objects(self, db_path):
        """filter_new_items must return the original candidate dicts, not copies."""
        emb = _unit_vec(1.0, 0.0)
        candidate = _make_candidate(url="https://orig.com")
        with patch("redpill.dedup._get_model", return_value=_fake_model(emb)):
            result = filter_new_items([candidate], db_path, threshold=0.85)
        assert result[0] is candidate

    def test_order_of_kept_items_is_preserved(self, db_path):
        """Items must come back in the same order they were passed in."""
        emb = _unit_vec(0.5, 0.5)
        candidates = [
            _make_candidate(url=f"https://item{i}.com") for i in range(5)
        ]
        with patch("redpill.dedup._get_model", return_value=_fake_model(emb)):
            result = filter_new_items(candidates, db_path, threshold=0.85)
        # DB starts empty so all should be kept (same embedding — but no prior
        # stored embeddings to compare against, so all pass Pass 2).
        assert [r["url"] for r in result] == [c["url"] for c in candidates]

    # ------------------------------------------------------------------
    # model is NOT called when all candidates are dropped by URL
    # ------------------------------------------------------------------

    def test_model_not_called_when_all_dropped_by_url(self, seeded_db):
        """If every candidate fails the URL check, encode() must never be called."""
        db_path, _ = seeded_db
        candidate = _make_candidate(url="https://already-seen.com")
        mock_model = _fake_model(np.zeros(4, dtype=np.float32))
        with patch("redpill.dedup._get_model", return_value=mock_model):
            filter_new_items([candidate], db_path, threshold=0.85)
        mock_model.encode.assert_not_called()
