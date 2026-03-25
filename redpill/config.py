"""
config.py — Load and validate configuration from config.yaml / config.example.yaml.
API keys are loaded from .env via python-dotenv.
"""

_FEEDBACK_DEFAULTS: dict = {
    "enabled": False,
    "base_url": "http://localhost:8080",
    "port": 8080,
    "db_path": "data/feedback.db",
    "min_votes_for_signals": 5,
    "signal_lookback_days": 30,
}


def get_feedback_config(config: dict) -> dict:
    """Return a fully-populated feedback config dict.

    Merges the ``feedback`` block from *config* (if present) with
    ``_FEEDBACK_DEFAULTS``.  Missing keys fall back to defaults so callers
    can always do ``cfg["enabled"]`` without a guard.

    Parameters
    ----------
    config:
        The top-level application config dict loaded from config.yaml.

    Returns
    -------
    A dict with keys: enabled, base_url, port, db_path,
    min_votes_for_signals, signal_lookback_days.
    """
    raw: dict = config.get("feedback", {}) or {}
    return {**_FEEDBACK_DEFAULTS, **raw}
