import pytest

from redpill.config import resolve_db_path


def test_explicit_db_path():
    config = {"db_path": "custom/my.db", "topic": "anything", "db_dir": "data"}
    assert resolve_db_path(config) == "custom/my.db"


def test_db_dir_constructs_slug():
    config = {"db_dir": "data", "topic": "contrastive learning"}
    assert resolve_db_path(config) == "data/redpill_contrastive_learning.db"


def test_db_dir_slugify_punctuation():
    config = {"db_dir": "/var/db", "topic": "ML & AI: the future!"}
    assert resolve_db_path(config) == "/var/db/redpill_ml_ai_the_future.db"


def test_db_dir_missing_topic_raises():
    with pytest.raises(ValueError, match="topic"):
        resolve_db_path({"db_dir": "data"})


def test_db_dir_empty_topic_raises():
    with pytest.raises(ValueError, match="topic"):
        resolve_db_path({"db_dir": "data", "topic": "  "})


def test_fallback_default():
    assert resolve_db_path({}) == "data/redpill.db"


def test_production_topic_slug_stability():
    # Pin the slug for the production topic so a typo fix doesn't silently
    # abandon the DB.
    config = {"db_dir": "data", "topic": "contrastive learning"}
    assert resolve_db_path(config) == "data/redpill_contrastive_learning.db"
