"""
config.py — Load and validate configuration from config.yaml / config.example.yaml.
API keys are loaded from .env via python-dotenv.
"""

_VALID_SEARCH_PROVIDERS = {"tavily", "serper", "both"}

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


def get_search_provider(config: dict) -> str:
    """Return the validated search_provider string from config.

    Defaults to ``"tavily"`` when the key is absent so existing deployments
    that upgrade without a ``SERPER_API_KEY`` continue to work unchanged.

    Parameters
    ----------
    config:
        The top-level application config dict loaded from config.yaml.

    Returns
    -------
    One of ``"tavily"``, ``"serper"``, or ``"both"``.

    Raises
    ------
    ValueError
        If the value present in config is not a recognised provider name.
    """
    value = config.get("search_provider", "tavily")
    if value not in _VALID_SEARCH_PROVIDERS:
        raise ValueError(
            f"Invalid search_provider {value!r}. "
            f"Must be one of: {sorted(_VALID_SEARCH_PROVIDERS)}"
        )
    return value
