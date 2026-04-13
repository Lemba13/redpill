"""
config.py — Load and validate configuration from config.yaml / config.example.yaml.
API keys are loaded from .env via python-dotenv.
"""

import logging
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_CANDIDATES = ("config.yaml", "config.example.yaml")


def load_config(config_path: str | None = None) -> dict:
    """Load and return the YAML config as a dict.

    If *config_path* is given, that file is used exclusively.
    Otherwise the function tries ``config.yaml`` then ``config.example.yaml``
    in the current working directory, in that order.

    Raises
    ------
    SystemExit
        If the file cannot be found or parsed.  Callers should not catch this —
        it is meant to terminate the process with a user-readable message.
    """
    candidates: list[str]
    if config_path is not None:
        candidates = [config_path]
    else:
        candidates = list(_CONFIG_CANDIDATES)

    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                with p.open(encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)
                if not isinstance(config, dict):
                    print(
                        f"ERROR: {path} did not parse to a mapping — "
                        "check that the file is valid YAML.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                logger.info("Loaded config from %s", p.resolve())
                return config
            except yaml.YAMLError as exc:
                print(f"ERROR: Failed to parse {path}: {exc}", file=sys.stderr)
                sys.exit(1)

    tried = ", ".join(candidates)
    print(
        f"ERROR: No config file found. Tried: {tried}\n"
        "Copy config.example.yaml to config.yaml and edit it.",
        file=sys.stderr,
    )
    sys.exit(1)

_VALID_SEARCH_PROVIDERS = {"tavily", "serper", "both"}

_DEFAULT_DB_PATH = "data/redpill.db"


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text


def resolve_db_path(config: dict) -> str:
    """Return the resolved SQLite database path for this config.

    Resolution order (first match wins):
    1. ``db_path`` key present → use as-is.
    2. ``db_dir`` key present → construct ``<db_dir>/redpill_<slug>.db``
       where ``<slug>`` is derived from the ``topic`` key.
    3. Neither → fall back to ``data/redpill.db``.

    Raises
    ------
    ValueError
        If ``db_dir`` is set but ``topic`` is absent or empty.
    """
    if "db_path" in config:
        return str(config["db_path"])

    if "db_dir" in config:
        topic: str = config.get("topic", "").strip()
        if not topic:
            raise ValueError(
                "'db_dir' requires 'topic' to be set so the filename can be derived."
            )
        return str(Path(config["db_dir"]) / f"redpill_{_slugify(topic)}.db")

    return _DEFAULT_DB_PATH

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
