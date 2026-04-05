"""
state.py — SQLite-backed state management for seen items, extracted terms,
query logs, and research plans.

Tables
------
seen_items:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    url             TEXT UNIQUE
    title           TEXT
    content_hash    TEXT   (SHA256 of extracted content)
    embedding       BLOB   (serialized numpy array; self-describing binary format)
    summary         TEXT
    first_seen_date TEXT   (ISO format)
    topic           TEXT

extracted_terms:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    term            TEXT NOT NULL
    source_url      TEXT
    source_title    TEXT
    topic           TEXT NOT NULL
    category        TEXT        -- subtopic|technique|author|dataset|framework|keyword
    first_seen      DATE NOT NULL
    frequency       INTEGER DEFAULT 1
    last_seen       DATE NOT NULL
    UNIQUE(term, topic)

query_log:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    query_text      TEXT NOT NULL
    run_date        DATE NOT NULL
    source          TEXT NOT NULL   -- base|extracted_term|llm_planned|fallback|static
    topic           TEXT NOT NULL
    results_count   INTEGER DEFAULT 0
    new_items       INTEGER DEFAULT 0
    kept_items      INTEGER DEFAULT 0

research_plans:
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    topic           TEXT NOT NULL
    run_date        DATE NOT NULL
    plan_json       TEXT NOT NULL       -- full research plan as JSON
    reasoning_trace TEXT               -- reasoning model's chain of thought (if available)
    source          TEXT NOT NULL DEFAULT 'llm'  -- "llm" or "fallback"

Embedding serialization format (all fields packed via struct.pack):
    [4 bytes: dtype_len (uint32 big-endian)]
    [dtype_len bytes: dtype string, e.g. b"float32"]
    [4 bytes: ndim (uint32 big-endian)]
    [ndim * 8 bytes: shape dimensions (uint64 big-endian each)]
    [remaining bytes: ndarray.tobytes() raw data]

Public API:
    init_db(db_path: str) -> None
    is_url_seen(url: str, db_path: str) -> bool
    get_all_embeddings(db_path: str) -> list[tuple[int, np.ndarray]]
    add_item(url, title, content_hash, embedding, summary, topic, db_path, *, dim_id) -> None
    get_items_since(date: str, db_path: str) -> list[dict]
    store_extracted_terms(terms: list[dict], db_path: str) -> None
    get_recent_terms(topic: str, db_path: str, days: int = 30) -> list[dict]
    get_top_terms(topic: str, db_path: str, limit: int = 50) -> list[dict]
    log_query(query_text: str, run_date: str, source: str, topic: str, db_path: str, *, dim_id) -> int
    update_query_stats(query_id: int, results_count: int, new_items: int, kept_items: int, db_path: str) -> None
    get_query_performance(topic: str, db_path: str, days: int = 14) -> list[dict]
    save_research_plan(topic: str, run_date: str, plan: dict, db_path: str, reasoning_trace: str | None, source: str) -> int
    get_latest_research_plan(topic: str, db_path: str) -> dict | None

Internal (for testing with an in-memory connection):
    init_db_conn(conn) -> None
    is_url_seen_conn(url, conn) -> bool
    get_all_embeddings_conn(conn) -> list[tuple[int, np.ndarray]]
    add_item_conn(url, title, content_hash, embedding, summary, topic, conn, *, dim_id) -> None
    get_items_since_conn(date, conn) -> list[dict]
    store_extracted_terms_conn(terms, conn) -> None
    get_recent_terms_conn(topic, days, conn) -> list[dict]
    get_top_terms_conn(topic, limit, conn) -> list[dict]
    log_query_conn(query_text, run_date, source, topic, conn, *, dim_id) -> int
    update_query_stats_conn(query_id, results_count, new_items, kept_items, conn) -> None
    get_query_performance_conn(topic, days, conn) -> list[dict]
    save_research_plan_conn(topic, run_date, plan, conn, reasoning_trace, source) -> int
    get_latest_research_plan_conn(topic, conn) -> dict | None
"""

import logging
import sqlite3
import struct
from contextlib import contextmanager
from datetime import date as _date, timedelta
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/redpill.db"

_CREATE_SEEN_ITEMS_SQL = """
CREATE TABLE IF NOT EXISTS seen_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL DEFAULT '',
    embedding       BLOB,
    summary         TEXT NOT NULL DEFAULT '',
    first_seen_date TEXT NOT NULL,
    topic           TEXT NOT NULL DEFAULT '',
    dim_id          TEXT
)
"""

_CREATE_EXTRACTED_TERMS_SQL = """
CREATE TABLE IF NOT EXISTS extracted_terms (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    term            TEXT NOT NULL,
    source_url      TEXT,
    source_title    TEXT,
    topic           TEXT NOT NULL,
    category        TEXT,
    first_seen      DATE NOT NULL,
    frequency       INTEGER NOT NULL DEFAULT 1,
    last_seen       DATE NOT NULL,
    UNIQUE(term, topic)
)
"""

_CREATE_QUERY_LOG_SQL = """
CREATE TABLE IF NOT EXISTS query_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text      TEXT NOT NULL,
    run_date        DATE NOT NULL,
    source          TEXT NOT NULL,
    topic           TEXT NOT NULL,
    results_count   INTEGER NOT NULL DEFAULT 0,
    new_items       INTEGER NOT NULL DEFAULT 0,
    kept_items      INTEGER NOT NULL DEFAULT 0,
    dim_id          TEXT
)
"""

_CREATE_DIMENSION_REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS dimension_registry (
    dim_id          TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    topic           TEXT NOT NULL,
    embedding       BLOB,
    hyde_abstract   TEXT,
    pool            TEXT NOT NULL DEFAULT 'explore',
    alpha           INTEGER NOT NULL DEFAULT 1,
    beta            INTEGER NOT NULL DEFAULT 1,
    run_count       INTEGER NOT NULL DEFAULT 0,
    last_seen       DATE
)
"""

_CREATE_RESEARCH_PLANS_SQL = """
CREATE TABLE IF NOT EXISTS research_plans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT NOT NULL,
    run_date        DATE NOT NULL,
    plan_json       TEXT NOT NULL,
    reasoning_trace TEXT,
    source          TEXT NOT NULL DEFAULT 'llm'
)
"""

_CREATE_TOPIC_SCAFFOLD_SQL = """
CREATE TABLE IF NOT EXISTS topic_scaffold (
    topic       TEXT PRIMARY KEY,
    scaffold    TEXT NOT NULL,
    created_at  DATE NOT NULL
)
"""

# Keep the old name as an alias so existing callers (init_db_conn) are not broken
_CREATE_TABLE_SQL = _CREATE_SEEN_ITEMS_SQL


# ---------------------------------------------------------------------------
# Embedding serialization
# ---------------------------------------------------------------------------

def serialize_embedding(arr: np.ndarray) -> bytes:
    """Public alias for _serialize_embedding. Use this from other modules."""
    return _serialize_embedding(arr)


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Public alias for _deserialize_embedding. Use this from other modules."""
    return _deserialize_embedding(blob)


def _serialize_embedding(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to a self-describing byte sequence.

    Format:
        uint32 (big-endian): length of dtype string
        bytes:               dtype name (e.g. b"float32"), endian-agnostic
        uint32 (big-endian): number of dimensions
        uint64[] (big-endian): shape tuple, one per dimension
        bytes:               raw array data via ndarray.tobytes()

    dtype.name is used (not dtype.str) so the stored name is endian-agnostic
    (e.g. "float32", not "<f4"). The raw bytes are always written in native
    order via tobytes(), which is correct for single-machine use.
    """
    dtype_bytes = arr.dtype.name.encode("ascii")
    dtype_len = len(dtype_bytes)
    ndim = arr.ndim
    header = struct.pack(
        f">I{dtype_len}sI{ndim}Q",
        dtype_len,
        dtype_bytes,
        ndim,
        *arr.shape,
    )
    return header + arr.tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Inverse of _serialize_embedding. Reconstructs the numpy array exactly."""
    offset = 0

    # Read dtype string length
    (dtype_len,) = struct.unpack_from(">I", blob, offset)
    offset += 4

    # Read dtype string
    (dtype_bytes,) = struct.unpack_from(f">{dtype_len}s", blob, offset)
    offset += dtype_len
    dtype = np.dtype(dtype_bytes.decode("ascii"))

    # Read number of dimensions
    (ndim,) = struct.unpack_from(">I", blob, offset)
    offset += 4

    # Read shape
    shape = struct.unpack_from(f">{ndim}Q", blob, offset)
    offset += ndim * 8

    # Read raw array data
    raw = blob[offset:]
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@contextmanager
def _open_conn(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection with sensible defaults, close it on exit.

    Row factory is set so rows behave like dicts (via sqlite3.Row).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Connection-based internal implementations
# ---------------------------------------------------------------------------

def init_db_conn(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not already exist.

    Safe to call on an existing database — all statements use
    CREATE TABLE IF NOT EXISTS, so no data is dropped on re-init.
    Does not commit — callers (or _open_conn) are responsible for commit/rollback.
    """
    conn.execute(_CREATE_SEEN_ITEMS_SQL)
    conn.execute(_CREATE_EXTRACTED_TERMS_SQL)
    conn.execute(_CREATE_QUERY_LOG_SQL)
    conn.execute(_CREATE_RESEARCH_PLANS_SQL)
    conn.execute(_CREATE_DIMENSION_REGISTRY_SQL)
    conn.execute(_CREATE_TOPIC_SCAFFOLD_SQL)

    # Safe migrations for existing databases — silently skip if column exists.
    for _stmt in (
        "ALTER TABLE query_log ADD COLUMN dim_id TEXT",
        "ALTER TABLE seen_items ADD COLUMN dim_id TEXT",
    ):
        try:
            conn.execute(_stmt)
        except sqlite3.OperationalError:
            pass  # column already exists

    # Permanent sentinel rows — always present regardless of LLM output.
    conn.execute(
        """
        INSERT OR IGNORE INTO dimension_registry
            (dim_id, canonical_name, topic, pool, alpha, beta, run_count)
        VALUES
            ('dim_fallback', 'deterministic fallback', '__system__', 'exploit', 0, 0, 0),
            ('dim_base', 'base topic anchor', '__system__', 'exploit', 0, 0, 0)
        """
    )

    logger.debug("All tables initialised (or already exist)")


def is_url_seen_conn(url: str, conn: sqlite3.Connection) -> bool:
    """Return True if the given URL is already recorded in seen_items."""
    row = conn.execute(
        "SELECT 1 FROM seen_items WHERE url = ? LIMIT 1", (url,)
    ).fetchone()
    return row is not None


def get_all_embeddings_conn(
    conn: sqlite3.Connection,
) -> list[tuple[int, np.ndarray]]:
    """Return all (id, embedding) pairs from seen_items that have an embedding.

    Rows with a NULL embedding blob are silently skipped.
    """
    rows = conn.execute(
        "SELECT id, embedding FROM seen_items WHERE embedding IS NOT NULL"
    ).fetchall()

    result: list[tuple[int, np.ndarray]] = []
    for row in rows:
        item_id: int = row["id"]
        blob: bytes = row["embedding"]
        try:
            arr = _deserialize_embedding(blob)
        except Exception as exc:
            logger.warning(
                "Failed to deserialize embedding for id=%d: %s", item_id, exc
            )
            continue
        result.append((item_id, arr))

    logger.debug("Loaded %d embeddings from database", len(result))
    return result


def add_item_conn(
    url: str,
    title: str,
    content_hash: str,
    embedding: np.ndarray,
    summary: str,
    topic: str,
    conn: sqlite3.Connection,
    first_seen_date: str | None = None,
    dim_id: str | None = None,
) -> None:
    """Insert a new item using INSERT OR IGNORE for idempotency.

    If the URL already exists the row is left unchanged and no error is raised.
    `first_seen_date` defaults to today's date in ISO format if not provided.
    """
    if first_seen_date is None:
        first_seen_date = _date.today().isoformat()

    blob = _serialize_embedding(embedding)

    conn.execute(
        """
        INSERT OR IGNORE INTO seen_items
            (url, title, content_hash, embedding, summary, first_seen_date, topic, dim_id)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (url, title, content_hash, blob, summary, first_seen_date, topic, dim_id),
    )
    logger.debug("add_item: url=%r (INSERT OR IGNORE)", url)


def get_items_since_conn(
    date: str, conn: sqlite3.Connection
) -> list[dict]:
    """Return all items first seen on or after `date` (ISO format string).

    The returned dicts include all columns except the raw embedding BLOB.
    Rows are ordered by first_seen_date ascending, then id ascending.
    """
    rows = conn.execute(
        """
        SELECT id, url, title, content_hash, summary, first_seen_date, topic, dim_id
        FROM seen_items
        WHERE first_seen_date >= ?
        ORDER BY first_seen_date ASC, id ASC
        """,
        (date,),
    ).fetchall()

    items = [dict(row) for row in rows]
    logger.debug(
        "get_items_since(%r): returned %d items", date, len(items)
    )
    return items


# ---------------------------------------------------------------------------
# Public API (db_path-based, opens and closes its own connection)
# ---------------------------------------------------------------------------

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the SQLite database and all tables if they do not exist."""
    with _open_conn(db_path) as conn:
        init_db_conn(conn)


def is_url_seen(url: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Return True if `url` has already been processed and stored."""
    with _open_conn(db_path) as conn:
        return is_url_seen_conn(url, conn)


def get_all_embeddings(
    db_path: str = DEFAULT_DB_PATH,
) -> list[tuple[int, np.ndarray]]:
    """Return all (id, embedding) pairs for use in similarity search."""
    with _open_conn(db_path) as conn:
        return get_all_embeddings_conn(conn)


def add_item(
    url: str,
    title: str,
    content_hash: str,
    embedding: np.ndarray,
    summary: str,
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
    first_seen_date: str | None = None,
    dim_id: str | None = None,
) -> None:
    """Persist a new item. Safe to call more than once with the same URL."""
    with _open_conn(db_path) as conn:
        add_item_conn(
            url, title, content_hash, embedding, summary, topic, conn,
            first_seen_date=first_seen_date,
            dim_id=dim_id,
        )


def get_items_since(
    date: str, db_path: str = DEFAULT_DB_PATH
) -> list[dict]:
    """Return all items first seen on or after the given ISO date string."""
    with _open_conn(db_path) as conn:
        return get_items_since_conn(date, conn)


# ---------------------------------------------------------------------------
# extracted_terms — internal implementations
# ---------------------------------------------------------------------------

def store_extracted_terms_conn(
    terms: list[dict],
    conn: sqlite3.Connection,
) -> None:
    """Upsert a list of extracted term dicts into the extracted_terms table.

    Each dict must have: ``term``, ``topic``, ``first_seen`` (ISO date string),
    ``last_seen`` (ISO date string).  Optional fields: ``source_url``,
    ``source_title``, ``category``.

    On conflict (same term + topic already exists): increment frequency and
    update last_seen.  The source_url and source_title are updated only when
    the new values are non-NULL (preserves the earliest source on re-runs).
    """
    today = _date.today().isoformat()
    for term_dict in terms:
        term = term_dict.get("term", "")
        topic = term_dict.get("topic", "")
        if not term or not topic:
            logger.warning("store_extracted_terms_conn: skipping term with missing term or topic: %r", term_dict)
            continue
        conn.execute(
            """
            INSERT INTO extracted_terms
                (term, source_url, source_title, topic, category, first_seen, frequency, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(term, topic) DO UPDATE SET
                frequency    = frequency + 1,
                last_seen    = excluded.last_seen,
                source_url   = COALESCE(excluded.source_url, source_url),
                source_title = COALESCE(excluded.source_title, source_title)
            """,
            (
                term,
                term_dict.get("source_url"),
                term_dict.get("source_title"),
                topic,
                term_dict.get("category"),
                term_dict.get("first_seen", today),
                term_dict.get("last_seen", today),
            ),
        )
    logger.debug("store_extracted_terms_conn: upserted %d term(s)", len(terms))


def get_recent_terms_conn(
    topic: str,
    days: int,
    conn: sqlite3.Connection,
) -> list[dict]:
    """Return terms for *topic* seen within the last *days* days.

    Results are sorted by frequency descending, then term ascending.
    """
    cutoff = (_date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT id, term, source_url, source_title, topic, category,
               first_seen, frequency, last_seen
        FROM extracted_terms
        WHERE topic = ? AND last_seen >= ?
        ORDER BY frequency DESC, term ASC
        """,
        (topic, cutoff),
    ).fetchall()
    result = [dict(row) for row in rows]
    logger.debug("get_recent_terms_conn(topic=%r, days=%d): %d term(s)", topic, days, len(result))
    return result


def get_top_terms_conn(
    topic: str,
    limit: int,
    conn: sqlite3.Connection,
) -> list[dict]:
    """Return the *limit* most frequent terms for *topic* across all time.

    Results are sorted by frequency descending, then term ascending.
    """
    rows = conn.execute(
        """
        SELECT id, term, source_url, source_title, topic, category,
               first_seen, frequency, last_seen
        FROM extracted_terms
        WHERE topic = ?
        ORDER BY frequency DESC, term ASC
        LIMIT ?
        """,
        (topic, limit),
    ).fetchall()
    result = [dict(row) for row in rows]
    logger.debug("get_top_terms_conn(topic=%r, limit=%d): %d term(s)", topic, limit, len(result))
    return result


# ---------------------------------------------------------------------------
# query_log — internal implementations
# ---------------------------------------------------------------------------

def log_query_conn(
    query_text: str,
    run_date: str,
    source: str,
    topic: str,
    conn: sqlite3.Connection,
    dim_id: str | None = None,
) -> int:
    """Insert a new query log entry and return its row id.

    Parameters
    ----------
    query_text:
        The search query string.
    run_date:
        ISO date of the pipeline run (e.g. "2026-03-15").
    source:
        One of: "base", "extracted_term", "llm_planned".
    topic:
        The topic this query belongs to (used for per-topic filtering).
    dim_id:
        The dimension identifier for this query (e.g. "dim_base", "dim_fallback",
        or a hashed dim id from the planner). NULL for legacy rows.

    Returns
    -------
    The ``id`` of the newly inserted row (sqlite3 lastrowid).
    """
    cursor = conn.execute(
        """
        INSERT INTO query_log (query_text, run_date, source, topic, dim_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (query_text, run_date, source, topic, dim_id),
    )
    row_id: int = cursor.lastrowid  # type: ignore[assignment]
    logger.debug("log_query_conn: logged query %r → id=%d", query_text, row_id)
    return row_id


def update_query_stats_conn(
    query_id: int,
    results_count: int,
    new_items: int,
    kept_items: int,
    conn: sqlite3.Connection,
) -> None:
    """Update the stats columns for an existing query_log row.

    Parameters
    ----------
    query_id:
        The id returned by log_query_conn.
    results_count:
        Total raw results the query returned from Tavily.
    new_items:
        How many survived deduplication.
    kept_items:
        How many made it into the final digest.
    """
    conn.execute(
        """
        UPDATE query_log
        SET results_count = ?, new_items = ?, kept_items = ?
        WHERE id = ?
        """,
        (results_count, new_items, kept_items, query_id),
    )
    logger.debug(
        "update_query_stats_conn: id=%d results=%d new=%d kept=%d",
        query_id, results_count, new_items, kept_items,
    )


def get_query_performance_conn(
    topic: str,
    days: int,
    conn: sqlite3.Connection,
) -> list[dict]:
    """Return query log entries for *topic* within the last *days* days.

    Ordered by run_date descending, then id descending (most recent first).
    """
    cutoff = (_date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT id, query_text, run_date, source, topic,
               results_count, new_items, kept_items, dim_id
        FROM query_log
        WHERE topic = ? AND run_date >= ?
        ORDER BY run_date DESC, id DESC
        """,
        (topic, cutoff),
    ).fetchall()
    result = [dict(row) for row in rows]
    logger.debug(
        "get_query_performance_conn(topic=%r, days=%d): %d entries", topic, days, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# research_plans — internal implementations
# ---------------------------------------------------------------------------

def save_research_plan_conn(
    topic: str,
    run_date: str,
    plan: dict,
    conn: sqlite3.Connection,
    reasoning_trace: str | None = None,
    source: str = "llm",
) -> int:
    """Persist a research plan for *topic* on *run_date*.

    The plan is stored as serialized JSON in ``plan_json``. Each call inserts
    a new row — the table is append-only; callers can retrieve the most recent
    plan via get_latest_research_plan_conn.

    Parameters
    ----------
    topic:
        The research topic this plan belongs to.
    run_date:
        ISO date string of the pipeline run (e.g. "2026-03-15").
    plan:
        The research plan dict produced by decompose_topic() or the fallback.
    conn:
        An open SQLite connection.
    reasoning_trace:
        The reasoning model's chain-of-thought text, if available.
    source:
        "llm" when the plan came from the reasoning model, "fallback" otherwise.

    Returns
    -------
    The ``id`` of the newly inserted row.
    """
    import json as _json

    plan_json = _json.dumps(plan, ensure_ascii=False)
    cursor = conn.execute(
        """
        INSERT INTO research_plans (topic, run_date, plan_json, reasoning_trace, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (topic, run_date, plan_json, reasoning_trace, source),
    )
    row_id: int = cursor.lastrowid  # type: ignore[assignment]
    logger.debug(
        "save_research_plan_conn: saved plan for topic=%r run_date=%r source=%r id=%d",
        topic, run_date, source, row_id,
    )
    return row_id


def get_latest_research_plan_conn(
    topic: str,
    conn: sqlite3.Connection,
) -> dict | None:
    """Return the most recent research plan for *topic*, or None.

    The returned dict includes all columns: id, topic, run_date, plan_json
    (as a string), reasoning_trace, source. Callers are responsible for
    deserializing plan_json via json.loads().
    """
    row = conn.execute(
        """
        SELECT id, topic, run_date, plan_json, reasoning_trace, source
        FROM research_plans
        WHERE topic = ?
        ORDER BY run_date DESC, id DESC
        LIMIT 1
        """,
        (topic,),
    ).fetchone()

    if row is None:
        logger.debug("get_latest_research_plan_conn: no plan found for topic=%r", topic)
        return None

    result = dict(row)
    logger.debug(
        "get_latest_research_plan_conn: found plan id=%d run_date=%r for topic=%r",
        result["id"], result["run_date"], topic,
    )
    return result


# ---------------------------------------------------------------------------
# extracted_terms + query_log + research_plans — public API (db_path-based)
# ---------------------------------------------------------------------------

def store_extracted_terms(
    terms: list[dict],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Upsert extracted terms into the database. See store_extracted_terms_conn."""
    with _open_conn(db_path) as conn:
        store_extracted_terms_conn(terms, conn)


def get_recent_terms(
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
    days: int = 30,
) -> list[dict]:
    """Return terms for *topic* seen in the last *days* days."""
    with _open_conn(db_path) as conn:
        return get_recent_terms_conn(topic, days, conn)


def get_top_terms(
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 50,
) -> list[dict]:
    """Return the top *limit* most frequent terms for *topic*."""
    with _open_conn(db_path) as conn:
        return get_top_terms_conn(topic, limit, conn)


def log_query(
    query_text: str,
    run_date: str,
    source: str,
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
    dim_id: str | None = None,
) -> int:
    """Insert a query log entry and return its id."""
    with _open_conn(db_path) as conn:
        return log_query_conn(query_text, run_date, source, topic, conn, dim_id=dim_id)


def update_query_stats(
    query_id: int,
    results_count: int,
    new_items: int,
    kept_items: int,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Update stats on an existing query_log row."""
    with _open_conn(db_path) as conn:
        update_query_stats_conn(query_id, results_count, new_items, kept_items, conn)


def get_query_performance(
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
    days: int = 14,
) -> list[dict]:
    """Return query log entries for *topic* within the last *days* days."""
    with _open_conn(db_path) as conn:
        return get_query_performance_conn(topic, days, conn)


def save_research_plan(
    topic: str,
    run_date: str,
    plan: dict,
    db_path: str = DEFAULT_DB_PATH,
    reasoning_trace: str | None = None,
    source: str = "llm",
) -> int:
    """Persist a research plan for *topic* on *run_date*. Returns the row id."""
    with _open_conn(db_path) as conn:
        return save_research_plan_conn(
            topic, run_date, plan, conn,
            reasoning_trace=reasoning_trace,
            source=source,
        )


def get_latest_research_plan(
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """Return the most recent research plan for *topic*, or None.

    The returned dict includes all columns including plan_json (as a string).
    Callers must deserialize plan_json via json.loads() to get the plan dict.
    """
    with _open_conn(db_path) as conn:
        return get_latest_research_plan_conn(topic, conn)
