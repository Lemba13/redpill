"""
feedback/db.py — Manages feedback.db: a completely separate SQLite database
from redpill.db.

Schema
------
digest_items:
    item_id        TEXT NOT NULL
    digest_date    DATE NOT NULL
    title          TEXT
    summary        TEXT
    url            TEXT
    domain         TEXT
    key_insight    TEXT
    relevance_score INTEGER
    source_query   TEXT
    plan_dimension TEXT
    dim_id         TEXT
    topic          TEXT
    PRIMARY KEY (item_id, digest_date)

votes:
    id         INTEGER PRIMARY KEY AUTOINCREMENT
    item_id    TEXT NOT NULL
    vote       TEXT NOT NULL CHECK (vote IN ('up', 'down'))
    voted_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    UNIQUE (item_id)          -- enforces last-vote-wins via INSERT OR REPLACE

Design notes
------------
- No persistent connection.  Every public method opens its own sqlite3.connect()
  and closes it on exit.  This is safe for use from FastAPI where multiple
  requests may arrive concurrently on different threads.
- Last-vote-wins is enforced by UNIQUE(item_id) on votes combined with
  INSERT OR REPLACE, which atomically removes any previous row and inserts the
  new one.  No DELETE + INSERT transaction needed.
- No FOREIGN KEY on votes (to keep the schema simple and avoid SQLite FK
  pragma dance).
"""

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_DIGEST_ITEMS_SQL = """
CREATE TABLE IF NOT EXISTS digest_items (
    item_id         TEXT NOT NULL,
    digest_date     DATE NOT NULL,
    title           TEXT,
    summary         TEXT,
    url             TEXT,
    domain          TEXT,
    key_insight     TEXT,
    relevance_score INTEGER,
    source_query    TEXT,
    plan_dimension  TEXT,
    dim_id          TEXT,
    topic           TEXT,
    PRIMARY KEY (item_id, digest_date)
)
"""

_CREATE_VOTES_SQL = """
CREATE TABLE IF NOT EXISTS votes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id   TEXT NOT NULL,
    vote      TEXT NOT NULL CHECK (vote IN ('up', 'down')),
    voted_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (item_id)
)
"""

_CREATE_BOOKMARKS_SQL = """
CREATE TABLE IF NOT EXISTS bookmarks (
    item_id       TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    summary       TEXT,
    key_insight   TEXT,
    digest_date   TEXT NOT NULL,
    bookmarked_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _open(db_path: str) -> sqlite3.Connection:
    """Open db_path, set row_factory, create tables if needed, return conn."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_DIGEST_ITEMS_SQL)
    conn.execute(_CREATE_VOTES_SQL)
    conn.execute(_CREATE_BOOKMARKS_SQL)
    # Safe migration for existing databases.
    try:
        conn.execute("ALTER TABLE digest_items ADD COLUMN dim_id TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    return conn


class FeedbackDB:
    """Thin wrapper around feedback.db.

    Every method opens and closes its own connection.  Pass ``db_path`` once
    at construction time; it is stored and reused per-call.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # Ensure the parent directory exists so the first _open() succeeds.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Digest ingestion
    # ------------------------------------------------------------------

    def ingest_digest(self, sidecar_path: str) -> int:
        """Read a JSON sidecar file and insert items into digest_items.

        Items that already exist (same item_id + digest_date) are skipped
        (INSERT OR IGNORE), making this call idempotent.

        Parameters
        ----------
        sidecar_path:
            Absolute or relative path to the YYYY-MM-DD.json sidecar.

        Returns
        -------
        Count of newly inserted rows.

        Raises
        ------
        FileNotFoundError
            If *sidecar_path* does not exist.
        ValueError
            If the JSON is malformed or missing required fields.
        """
        p = Path(sidecar_path)
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

        try:
            data: dict = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in sidecar {sidecar_path}: {exc}") from exc

        digest_date: str = data.get("digest_date", "")
        topic: str = data.get("topic", "")
        items: list[dict] = data.get("items", [])

        if not digest_date:
            raise ValueError(f"Sidecar {sidecar_path} is missing 'digest_date'")

        inserted = 0
        conn = _open(self._db_path)
        try:
            for item in items:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO digest_items
                        (item_id, digest_date, title, summary, url, domain,
                         key_insight, relevance_score, source_query, plan_dimension,
                         dim_id, topic)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.get("item_id", ""),
                        digest_date,
                        item.get("title"),
                        item.get("summary"),
                        item.get("url"),
                        item.get("domain"),
                        item.get("key_insight"),
                        item.get("relevance_score"),
                        item.get("source_query"),
                        item.get("plan_dimension"),
                        item.get("dim_id"),
                        topic,
                    ),
                )
                if cursor.rowcount:
                    inserted += 1
            conn.commit()
        finally:
            conn.close()

        logger.info(
            "ingest_digest: %s — %d item(s) inserted, %d skipped",
            sidecar_path,
            inserted,
            len(items) - inserted,
        )
        return inserted

    def is_digest_ingested(self, digest_date: str) -> bool:
        """Return True if at least one item for *digest_date* exists in digest_items."""
        conn = _open(self._db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM digest_items WHERE digest_date = ? LIMIT 1",
                (digest_date,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def record_vote(self, item_id: str, vote: str) -> dict:
        """Insert or replace a vote for *item_id*.

        Uses INSERT OR REPLACE to implement last-vote-wins atomically —
        the UNIQUE(item_id) constraint ensures at most one vote per item.

        Parameters
        ----------
        item_id:
            The item being voted on.
        vote:
            Either "up" or "down".

        Returns
        -------
        A dict with keys: item_id, vote, voted_at (ISO datetime string).

        Raises
        ------
        ValueError
            If *vote* is not "up" or "down".
        LookupError
            If *item_id* does not exist in digest_items.
        """
        if vote not in ("up", "down"):
            raise ValueError(f"vote must be 'up' or 'down', got {vote!r}")

        conn = _open(self._db_path)
        try:
            # Confirm item exists before recording the vote.
            row = conn.execute(
                "SELECT 1 FROM digest_items WHERE item_id = ? LIMIT 1",
                (item_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"item_id {item_id!r} not found in digest_items")

            conn.execute(
                """
                INSERT OR REPLACE INTO votes (item_id, vote)
                VALUES (?, ?)
                """,
                (item_id, vote),
            )
            conn.commit()

            voted_at_row = conn.execute(
                "SELECT voted_at FROM votes WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            voted_at: str = voted_at_row["voted_at"] if voted_at_row else ""
        finally:
            conn.close()

        logger.debug("record_vote: item_id=%r vote=%r", item_id, vote)
        return {"item_id": item_id, "vote": vote, "voted_at": voted_at}

    def get_vote_for_item(self, item_id: str) -> str | None:
        """Return the current vote ('up', 'down', or None) for *item_id*."""
        conn = _open(self._db_path)
        try:
            row = conn.execute(
                "SELECT vote FROM votes WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            return row["vote"] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_digest_items(self, digest_date: str) -> list[dict]:
        """Return all items for *digest_date* with their current vote status.

        Each returned dict has all digest_items columns plus a ``vote`` key
        (the current vote string, or None if not voted).  Items are ordered
        by relevance_score descending.
        """
        conn = _open(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    di.item_id,
                    di.digest_date,
                    di.title,
                    di.summary,
                    di.url,
                    di.domain,
                    di.key_insight,
                    di.relevance_score,
                    di.source_query,
                    di.plan_dimension,
                    di.dim_id,
                    di.topic,
                    v.vote
                FROM digest_items di
                LEFT JOIN votes v ON di.item_id = v.item_id
                WHERE di.digest_date = ?
                ORDER BY di.relevance_score DESC
                """,
                (digest_date,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_all_votes(self, days: int = 30) -> list[dict]:
        """Return all votes from the last *days* days, joined with item metadata.

        Used by the pipeline to compute preference signals.  Ordered by
        voted_at descending (most recent first).
        """
        conn = _open(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    v.item_id,
                    v.vote,
                    v.voted_at,
                    di.digest_date,
                    di.title,
                    di.url,
                    di.domain,
                    di.source_query,
                    di.plan_dimension,
                    di.dim_id,
                    di.topic,
                    di.relevance_score
                FROM votes v
                JOIN digest_items di ON v.item_id = di.item_id
                WHERE v.voted_at >= datetime('now', '-' || ? || ' days')
                ORDER BY v.voted_at DESC
                """,
                (days,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_distinct_domains(self) -> list[str]:
        """Return all distinct non-empty domains in digest_items, sorted."""
        conn = _open(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT domain
                FROM digest_items
                WHERE domain IS NOT NULL AND domain != ''
                ORDER BY domain ASC
                """,
            ).fetchall()
            return [row["domain"] for row in rows]
        finally:
            conn.close()

    def get_history_items(
        self,
        *,
        q: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        domain: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        """Return a page of digest items with vote status, filtered by optional criteria.

        Returns {"items": list[dict], "has_next": bool}.
        Fetches page_size+1 rows to detect whether a next page exists.
        Items are ordered by digest_date DESC, relevance_score DESC.
        """
        conditions: list[str] = []
        params: list = []

        if q:
            conditions.append("(di.title LIKE ? OR di.summary LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])
        if from_date:
            conditions.append("di.digest_date >= ?")
            params.append(from_date)
        if to_date:
            conditions.append("di.digest_date <= ?")
            params.append(to_date)
        if domain:
            conditions.append("di.domain = ?")
            params.append(domain)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        offset = (page - 1) * page_size
        params.extend([page_size + 1, offset])

        conn = _open(self._db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT
                    di.item_id,
                    di.digest_date,
                    di.title,
                    di.summary,
                    di.url,
                    di.domain,
                    di.topic,
                    di.key_insight,
                    di.relevance_score,
                    v.vote
                FROM digest_items di
                LEFT JOIN votes v ON di.item_id = v.item_id
                {where}
                ORDER BY di.digest_date DESC, di.relevance_score DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        items = [dict(row) for row in rows]
        has_next = len(items) > page_size
        return {"items": items[:page_size], "has_next": has_next}

    # ------------------------------------------------------------------
    # Bookmarks
    # ------------------------------------------------------------------

    def toggle_bookmark(self, item_id: str) -> dict:
        """Toggle a bookmark for *item_id*.

        If the item is already bookmarked, delete it (unbookmark).
        If not, copy fields from digest_items and insert into bookmarks.

        Returns
        -------
        {"bookmarked": True}  after inserting
        {"bookmarked": False} after deleting

        Raises
        ------
        LookupError
            If *item_id* does not exist in digest_items.
        """
        conn = _open(self._db_path)
        try:
            existing = conn.execute(
                "SELECT 1 FROM bookmarks WHERE item_id = ?",
                (item_id,),
            ).fetchone()

            if existing:
                conn.execute("DELETE FROM bookmarks WHERE item_id = ?", (item_id,))
                conn.commit()
                return {"bookmarked": False}

            row = conn.execute(
                """
                SELECT item_id, title, url, summary, key_insight, digest_date
                FROM digest_items
                WHERE item_id = ?
                LIMIT 1
                """,
                (item_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"item_id {item_id!r} not found in digest_items")

            conn.execute(
                """
                INSERT INTO bookmarks (item_id, title, url, summary, key_insight, digest_date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["item_id"],
                    row["title"] or "",
                    row["url"] or "",
                    row["summary"],
                    row["key_insight"],
                    row["digest_date"],
                ),
            )
            conn.commit()
            return {"bookmarked": True}
        finally:
            conn.close()

    def get_bookmarked_ids(self) -> set:
        """Return the set of all bookmarked item_ids."""
        conn = _open(self._db_path)
        try:
            rows = conn.execute("SELECT item_id FROM bookmarks").fetchall()
            return {row["item_id"] for row in rows}
        finally:
            conn.close()

    def get_all_bookmarks(self) -> list[dict]:
        """Return all bookmarks ordered by bookmarked_at descending."""
        conn = _open(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT item_id, title, url, summary, key_insight, digest_date, bookmarked_at
                FROM bookmarks
                ORDER BY bookmarked_at DESC
                """,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_available_digests(self) -> list[dict]:
        """Return digests that have been ingested, newest first.

        Each entry is a dict with keys: date, item_count, vote_count.
        A single aggregation query computes all three in one pass.
        """
        conn = _open(self._db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    di.digest_date                              AS date,
                    COUNT(DISTINCT di.item_id)                  AS item_count,
                    COUNT(DISTINCT v.item_id)                   AS vote_count
                FROM digest_items di
                LEFT JOIN votes v ON di.item_id = v.item_id
                GROUP BY di.digest_date
                ORDER BY di.digest_date DESC
                """,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
