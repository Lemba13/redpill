"""
state.py — SQLite-backed state management for seen items.

Table: seen_items
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    url             TEXT UNIQUE
    title           TEXT
    content_hash    TEXT   (SHA256 of extracted content)
    embedding       BLOB   (serialized numpy array)
    summary         TEXT
    first_seen_date TEXT   (ISO format)
    topic           TEXT

Public API:
    init_db(db_path: str)
    is_url_seen(url: str) -> bool
    get_all_embeddings() -> list[tuple[int, np.ndarray]]
    add_item(url, title, content_hash, embedding, summary, topic)
    get_items_since(date: str) -> list[dict]
"""
