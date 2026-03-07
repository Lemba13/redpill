"""
dedup.py — Deduplication engine using URL exact-match + semantic similarity.

Embedding model: sentence-transformers all-MiniLM-L6-v2, loaded once via a
lazy module-level singleton. Tests patch _get_model() to avoid loading the
real model.

Public API:
    compute_embedding(text: str) -> np.ndarray
        Computes an embedding for the given text, truncating to the first 512
        whitespace-delimited tokens before encoding.

    is_semantic_duplicate(
        embedding: np.ndarray,
        existing: list[tuple[int, np.ndarray]],
        threshold: float,
    ) -> tuple[bool, int | None, float | None]
        Returns (is_dup, closest_id, score). `score` is the cosine similarity
        to the closest existing embedding (higher == more similar).

    filter_new_items(
        candidates: list[dict],
        db_path: str,
        threshold: float,
    ) -> list[dict]
        Pass 1 — URL exact match (cheap, avoids computing any embeddings).
        Pass 2 — semantic similarity against all embeddings already in the DB.
        Logs every decision: KEPT / DROPPED (url_match) / DROPPED (semantic, score=X).
"""

import logging
import sqlite3
from typing import Optional

import numpy as np

import redpill.state as state

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Lazy singleton for the embedding model
# ---------------------------------------------------------------------------

_model: Optional[object] = None  # SentenceTransformer at runtime


def _get_model():  # -> SentenceTransformer
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is None:
        # Import deferred so that importing this module is fast and tests can
        # patch _get_model before the real import ever happens.
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        logger.info("Loading embedding model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_embedding(text: str) -> np.ndarray:
    """Return a normalized embedding vector for *text*.

    The input is truncated to the first ``_MAX_TOKENS`` whitespace-delimited
    tokens before encoding. This is a fast, model-agnostic guard — the
    underlying tokenizer still applies its own limits, but capping here avoids
    sending absurdly long strings to the model.

    Returns a 1-D float32 numpy array.
    """
    tokens = text.split()
    if len(tokens) > _MAX_TOKENS:
        text = " ".join(tokens[:_MAX_TOKENS])

    model = _get_model()
    # sentence-transformers returns a numpy array by default; convert
    # explicitly so callers can rely on the type.
    embedding = model.encode(text, convert_to_numpy=True)
    return np.array(embedding, dtype=np.float32)


def is_semantic_duplicate(
    embedding: np.ndarray,
    existing: list[tuple[int, np.ndarray]],
    threshold: float,
) -> tuple[bool, int | None, float | None]:
    """Compare *embedding* against every existing embedding via cosine similarity.

    Cosine similarity is computed as the dot product of L2-normalised vectors.
    This is numerically equivalent to scipy.spatial.distance.cosine but avoids
    a heavy dependency and an extra allocation.

    Returns:
        (is_dup, closest_id, score) where:
            is_dup     — True if any existing embedding exceeds *threshold*.
            closest_id — DB id of the most similar item, or None if *existing* is empty.
            score      — cosine similarity to the closest item, or None if *existing* is empty.
    """
    if not existing:
        return False, None, None

    # Normalise the candidate once.
    candidate_norm = np.linalg.norm(embedding)
    if candidate_norm == 0.0:
        # A zero vector has undefined cosine similarity; treat as non-duplicate.
        return False, None, None
    candidate_unit = embedding / candidate_norm

    best_id: int | None = None
    best_score: float = -float("inf")

    for item_id, existing_embedding in existing:
        existing_norm = np.linalg.norm(existing_embedding)
        if existing_norm == 0.0:
            continue
        existing_unit = existing_embedding / existing_norm
        score = float(np.dot(candidate_unit, existing_unit))
        if score > best_score:
            best_score = score
            best_id = item_id

    if best_id is None:
        # All existing embeddings were zero vectors — nothing to compare against.
        return False, None, None

    is_dup = best_score >= threshold
    return is_dup, best_id, best_score


def filter_new_items(
    candidates: list[dict],
    db_path: str,
    threshold: float = 0.85,
) -> list[dict]:
    """Filter *candidates* to only those not already seen in the database.

    Two-pass strategy:
        Pass 1 (cheap)     — exact URL match via state.is_url_seen_conn().
        Pass 2 (expensive) — semantic similarity via compute_embedding() and
                             is_semantic_duplicate() against all stored embeddings.

    Every decision is logged at INFO level with a clear KEPT / DROPPED label.

    Each candidate dict must have at minimum:
        url     (str)  — used for Pass 1 and logging.
        title   (str)  — used for logging.
        snippet (str)  — fallback text for embedding if content is absent.
        content (str | None) — preferred text for embedding.

    Items with no usable text (both content and snippet are empty/None) are
    skipped with a WARNING and excluded from the output.

    Returns the filtered list of candidate dicts that passed both checks.
    """
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        state.init_db_conn(conn)
        conn.commit()

        # --- Pass 1: URL exact match -------------------------------------------
        url_survivors: list[dict] = []
        for item in candidates:
            url = item.get("url", "")
            if state.is_url_seen_conn(url, conn):
                logger.info("DROPPED (url_match): %s", url)
            else:
                url_survivors.append(item)

        if not url_survivors:
            logger.info("filter_new_items: all candidates dropped by URL match")
            return []

        # --- Pass 2: Semantic similarity ---------------------------------------
        existing_embeddings: list[tuple[int, np.ndarray]] = (
            state.get_all_embeddings_conn(conn)
        )

        kept: list[dict] = []
        for item in url_survivors:
            url = item.get("url", "")
            content = item.get("content") or ""
            snippet = item.get("snippet") or ""
            text = content if content.strip() else snippet

            if not text.strip():
                logger.warning(
                    "SKIPPED (no text): url=%s has neither content nor snippet; "
                    "cannot compute embedding — excluding from output",
                    url,
                )
                continue

            embedding = compute_embedding(text)
            is_dup, closest_id, score = is_semantic_duplicate(
                embedding, existing_embeddings, threshold
            )

            if is_dup:
                logger.info(
                    "DROPPED (semantic, score=%.4f, closest_id=%s): %s",
                    score,
                    closest_id,
                    url,
                )
            else:
                if score is not None:
                    logger.info(
                        "KEPT (best_score=%.4f, closest_id=%s): %s",
                        score,
                        closest_id,
                        url,
                    )
                else:
                    logger.info("KEPT (no existing embeddings to compare): %s", url)
                kept.append(item)

        return kept

    finally:
        if conn is not None:
            conn.close()
