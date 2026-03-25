"""
feedback_reader.py — Read-only access to feedback.db from the pipeline.

This module is part of the redpill pipeline, NOT the feedback service.  It
reads feedback.db in read-only mode via SQLite's URI syntax to ensure the
pipeline never accidentally writes to the feedback database.

Key constraint: this module must NOT import anything from feedback.* .
It talks to feedback.db directly via sqlite3.

Public API
----------
FeedbackReader
    Context manager.  Open with:

        with FeedbackReader("data/feedback.db") as reader:
            signals = reader.compute_preference_signals(topic)

    compute_preference_signals(topic, days) -> dict
        Return a structured dict of preference signals derived from vote
        history.  The dict always has ``has_feedback`` (bool) and
        ``vote_count`` (int).  When has_feedback is False, all other signal
        keys are empty/None and the planner should skip feedback entirely.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum total votes (up + down) on a dimension/domain before it is included
# in the preference signal.  Avoids noisy signals from sparse data.
_MIN_VOTES_FOR_DIMENSION = 3
_MIN_VOTES_FOR_DOMAIN = 3

# Minimum total votes in the whole period before term sentiment is computed.
_MIN_VOTES_FOR_TERM_SENTIMENT = 20


class FeedbackReader:
    """Read-only access to feedback.db from the pipeline.

    Usage
    -----
    Always use as a context manager::

        with FeedbackReader(path) as reader:
            signals = reader.compute_preference_signals(topic)

    The connection is opened in read-only URI mode so SQLite will raise an
    error rather than silently creating the file if it doesn't exist.

    Parameters
    ----------
    feedback_db_path:
        Path to feedback.db.  The file must exist (or be created by the
        feedback service) before this can be opened.
    """

    def __init__(self, feedback_db_path: str) -> None:
        self._path = feedback_db_path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "FeedbackReader":
        self._conn = sqlite3.connect(
            f"file:{self._path}?mode=ro",
            uri=True,
        )
        self._conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        return None  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "FeedbackReader must be used as a context manager "
                "(use 'with FeedbackReader(...) as reader')"
            )
        return self._conn

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def compute_preference_signals(
        self,
        topic: str,
        days: int = 30,
    ) -> dict:
        """Compute preference signals from vote history for *topic*.

        Parameters
        ----------
        topic:
            The research topic to filter on.
        days:
            How many days of vote history to consider.

        Returns
        -------
        A dict with the following keys:

        has_feedback : bool
            False when no votes exist; all other keys are empty/None.
        vote_count : int
            Total number of votes in the period.
        period_days : int
            The *days* parameter, for the planner prompt.
        dimension_preferences : list[dict]
            Per-dimension approval signals (only for dimensions with at
            least _MIN_VOTES_FOR_DIMENSION total votes).
            Each dict: dimension, shown, up, down, approval.
        source_preferences : list[dict]
            Per-domain approval signals (only for domains with at least
            _MIN_VOTES_FOR_DOMAIN total votes).
            Each dict: domain, shown, up, down, approval.
        engagement : dict
            items_delivered, items_voted, engagement_rate.
        term_sentiment : list[dict] | None
            Per-term sentiment (only computed when total votes >= 20).
            Each dict: term, appearances_in_voted, up, down, sentiment.
            None when insufficient data.
        """
        # ------------------------------------------------------------------
        # Total votes in period (quick check — bail early if none)
        # ------------------------------------------------------------------
        vote_count_row = self._db.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM votes v
            JOIN digest_items di ON v.item_id = di.item_id
            WHERE di.topic = ?
              AND v.voted_at >= datetime('now', '-' || ? || ' days')
            """,
            (topic, days),
        ).fetchone()
        vote_count: int = vote_count_row["cnt"] if vote_count_row else 0

        if vote_count == 0:
            logger.info(
                "FeedbackReader: no votes for topic=%r in last %d days", topic, days
            )
            return {
                "has_feedback": False,
                "vote_count": 0,
                "period_days": days,
                "dimension_preferences": [],
                "source_preferences": [],
                "engagement": {
                    "items_delivered": 0,
                    "items_voted": 0,
                    "engagement_rate": 0.0,
                },
                "term_sentiment": None,
            }

        # ------------------------------------------------------------------
        # Dimension preferences
        # ------------------------------------------------------------------
        dim_rows = self._db.execute(
            """
            SELECT
                di.plan_dimension                       AS dimension,
                COUNT(DISTINCT di.item_id)              AS shown,
                SUM(CASE WHEN v.vote = 'up'   THEN 1 ELSE 0 END) AS up,
                SUM(CASE WHEN v.vote = 'down' THEN 1 ELSE 0 END) AS down
            FROM digest_items di
            JOIN votes v ON di.item_id = v.item_id
            WHERE di.topic = ?
              AND v.voted_at >= datetime('now', '-' || ? || ' days')
              AND di.plan_dimension IS NOT NULL
              AND di.plan_dimension != ''
            GROUP BY di.plan_dimension
            HAVING (SUM(CASE WHEN v.vote = 'up' THEN 1 ELSE 0 END)
                  + SUM(CASE WHEN v.vote = 'down' THEN 1 ELSE 0 END)) >= ?
            ORDER BY (CAST(SUM(CASE WHEN v.vote='up' THEN 1 ELSE 0 END) AS REAL)
                      / MAX(SUM(CASE WHEN v.vote='up' THEN 1 ELSE 0 END)
                          + SUM(CASE WHEN v.vote='down' THEN 1 ELSE 0 END), 1)) DESC
            """,
            (topic, days, _MIN_VOTES_FOR_DIMENSION),
        ).fetchall()

        dimension_preferences: list[dict] = []
        for row in dim_rows:
            up: int = row["up"] or 0
            down: int = row["down"] or 0
            total_voted = up + down
            approval = up / total_voted if total_voted > 0 else 0.0
            dimension_preferences.append(
                {
                    "dimension": row["dimension"],
                    "shown": row["shown"],
                    "up": up,
                    "down": down,
                    "approval": round(approval, 2),
                }
            )

        # ------------------------------------------------------------------
        # Source (domain) preferences
        # ------------------------------------------------------------------
        domain_rows = self._db.execute(
            """
            SELECT
                di.domain                               AS domain,
                COUNT(DISTINCT di.item_id)              AS shown,
                SUM(CASE WHEN v.vote = 'up'   THEN 1 ELSE 0 END) AS up,
                SUM(CASE WHEN v.vote = 'down' THEN 1 ELSE 0 END) AS down
            FROM digest_items di
            JOIN votes v ON di.item_id = v.item_id
            WHERE di.topic = ?
              AND v.voted_at >= datetime('now', '-' || ? || ' days')
              AND di.domain IS NOT NULL
              AND di.domain != ''
            GROUP BY di.domain
            HAVING (SUM(CASE WHEN v.vote = 'up' THEN 1 ELSE 0 END)
                  + SUM(CASE WHEN v.vote = 'down' THEN 1 ELSE 0 END)) >= ?
            ORDER BY (CAST(SUM(CASE WHEN v.vote='up' THEN 1 ELSE 0 END) AS REAL)
                      / MAX(SUM(CASE WHEN v.vote='up' THEN 1 ELSE 0 END)
                          + SUM(CASE WHEN v.vote='down' THEN 1 ELSE 0 END), 1)) DESC
            """,
            (topic, days, _MIN_VOTES_FOR_DOMAIN),
        ).fetchall()

        source_preferences: list[dict] = []
        for row in domain_rows:
            up = row["up"] or 0
            down = row["down"] or 0
            total_voted = up + down
            approval = up / total_voted if total_voted > 0 else 0.0
            source_preferences.append(
                {
                    "domain": row["domain"],
                    "shown": row["shown"],
                    "up": up,
                    "down": down,
                    "approval": round(approval, 2),
                }
            )

        # ------------------------------------------------------------------
        # Engagement rate
        # ------------------------------------------------------------------
        engagement_row = self._db.execute(
            """
            SELECT
                COUNT(DISTINCT di.item_id)                     AS items_delivered,
                COUNT(DISTINCT v.item_id)                      AS items_voted
            FROM digest_items di
            LEFT JOIN votes v
                   ON di.item_id = v.item_id
                  AND v.voted_at >= datetime('now', '-' || ? || ' days')
            WHERE di.topic = ?
              AND di.digest_date >= date('now', '-' || ? || ' days')
            """,
            (days, topic, days),
        ).fetchone()

        items_delivered: int = engagement_row["items_delivered"] if engagement_row else 0
        items_voted: int = engagement_row["items_voted"] if engagement_row else 0
        engagement_rate = items_voted / items_delivered if items_delivered > 0 else 0.0

        engagement = {
            "items_delivered": items_delivered,
            "items_voted": items_voted,
            "engagement_rate": round(engagement_rate, 2),
        }

        # ------------------------------------------------------------------
        # Term sentiment (only when sufficient data)
        # ------------------------------------------------------------------
        term_sentiment: list[dict] | None = None
        if vote_count >= _MIN_VOTES_FOR_TERM_SENTIMENT:
            term_sentiment = self._compute_term_sentiment(topic, days)

        logger.info(
            "FeedbackReader: topic=%r days=%d vote_count=%d dimensions=%d sources=%d",
            topic,
            days,
            vote_count,
            len(dimension_preferences),
            len(source_preferences),
        )

        return {
            "has_feedback": True,
            "vote_count": vote_count,
            "period_days": days,
            "dimension_preferences": dimension_preferences,
            "source_preferences": source_preferences,
            "engagement": engagement,
            "term_sentiment": term_sentiment,
        }

    def _compute_term_sentiment(self, topic: str, days: int) -> list[dict]:
        """Compute per-term sentiment from voted items.

        Checks which voted items' source queries contain known extracted terms.
        This is approximate (substring match) but good enough for the planner
        prompt — the planner treats these as soft signals, not hard rules.

        Note: this accesses ``digest_items.source_query`` in feedback.db.
        Cross-DB joins with redpill.db are not performed here; the pipeline
        caller may pass additional context if needed.
        """
        # Pull all voted items with their votes and source queries.
        voted_rows = self._db.execute(
            """
            SELECT
                di.source_query,
                v.vote
            FROM votes v
            JOIN digest_items di ON v.item_id = di.item_id
            WHERE di.topic = ?
              AND v.voted_at >= datetime('now', '-' || ? || ' days')
              AND di.source_query IS NOT NULL
              AND di.source_query != ''
            """,
            (topic, days),
        ).fetchall()

        if not voted_rows:
            return []

        # Build a rough term -> {up, down} counter from source query substrings.
        # We extract individual words (3+ chars, alpha only) from source queries.
        import re

        term_counts: dict[str, dict[str, int]] = {}
        for row in voted_rows:
            query: str = row["source_query"] or ""
            vote: str = row["vote"]
            # Extract meaningful words from the query string.
            words = re.findall(r"\b[a-zA-Z]{3,}\b", query)
            # Deduplicate within a single query to avoid double-counting.
            for word in set(words):
                word_lower = word.lower()
                if word_lower not in term_counts:
                    term_counts[word_lower] = {"up": 0, "down": 0}
                term_counts[word_lower][vote] += 1

        result: list[dict] = []
        for term, counts in sorted(
            term_counts.items(),
            key=lambda kv: kv[1]["up"] + kv[1]["down"],
            reverse=True,
        ):
            up = counts["up"]
            down = counts["down"]
            total = up + down
            if total < 2:
                continue
            sentiment = "positive" if up > down else "negative" if down > up else "neutral"
            result.append(
                {
                    "term": term,
                    "appearances_in_voted": total,
                    "up": up,
                    "down": down,
                    "sentiment": sentiment,
                }
            )

        # Cap at 20 entries — the planner prompt has a token budget.
        return result[:20]
