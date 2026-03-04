"""
dedup.py — Deduplication engine using URL exact-match + semantic similarity.

Embedding model: sentence-transformers all-MiniLM-L6-v2 (loaded once as lazy singleton).

Public API:
    compute_embedding(text: str) -> np.ndarray
        Truncates input to first 512 tokens.

    is_semantic_duplicate(
        embedding: np.ndarray,
        existing: list[tuple[int, np.ndarray]],
        threshold: float,
    ) -> tuple[bool, int | None, float | None]
        Returns (is_dup, closest_id, score).

    filter_new_items(
        candidates: list[dict],
        db: StateDB,
        threshold: float,
    ) -> list[dict]
        Pass 1 — URL exact match (cheap).
        Pass 2 — semantic similarity (expensive).
        Logs every decision: KEPT / DROPPED (url_match) / DROPPED (semantic, score=X).
"""
