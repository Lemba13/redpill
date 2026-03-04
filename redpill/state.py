"""
state.py — SQLite-backed state management for seen items.

Table: seen_items
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    url             TEXT UNIQUE
    title           TEXT
    content_hash    TEXT   (SHA256 of extracted content)
    embedding       BLOB   (serialized numpy array; self-describing binary format)
    summary         TEXT
    first_seen_date TEXT   (ISO format)
    topic           TEXT

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
    add_item(url, title, content_hash, embedding, summary, topic, db_path) -> None
    get_items_since(date: str, db_path: str) -> list[dict]

Internal (for testing with an in-memory connection):
    init_db_conn(conn: sqlite3.Connection) -> None
    is_url_seen_conn(url: str, conn: sqlite3.Connection) -> bool
    get_all_embeddings_conn(conn: sqlite3.Connection) -> list[tuple[int, np.ndarray]]
    add_item_conn(url, title, content_hash, embedding, summary, topic, conn) -> None
    get_items_since_conn(date: str, conn: sqlite3.Connection) -> list[dict]
"""

import logging
import sqlite3
import struct
from contextlib import contextmanager
from datetime import date as _date
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/redpill.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS seen_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL DEFAULT '',
    embedding       BLOB,
    summary         TEXT NOT NULL DEFAULT '',
    first_seen_date TEXT NOT NULL,
    topic           TEXT NOT NULL DEFAULT ''
)
"""


# ---------------------------------------------------------------------------
# Embedding serialization
# ---------------------------------------------------------------------------

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
    """Create the seen_items table if it does not already exist.

    Does not commit — callers (or _open_conn) are responsible for commit/rollback.
    """
    conn.execute(_CREATE_TABLE_SQL)
    logger.debug("seen_items table initialised (or already exists)")


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
            (url, title, content_hash, embedding, summary, first_seen_date, topic)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
        """,
        (url, title, content_hash, blob, summary, first_seen_date, topic),
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
        SELECT id, url, title, content_hash, summary, first_seen_date, topic
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
    """Create the SQLite database and seen_items table if they do not exist."""
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
) -> None:
    """Persist a new item. Safe to call more than once with the same URL."""
    with _open_conn(db_path) as conn:
        add_item_conn(
            url, title, content_hash, embedding, summary, topic, conn,
            first_seen_date=first_seen_date,
        )


def get_items_since(
    date: str, db_path: str = DEFAULT_DB_PATH
) -> list[dict]:
    """Return all items first seen on or after the given ISO date string."""
    with _open_conn(db_path) as conn:
        return get_items_since_conn(date, conn)
