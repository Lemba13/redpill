from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from redpill.config import _CONFIG_CANDIDATES, load_config, resolve_db_path

_SAMPLE_CONFIG = {
    "topic": "contrastive learning",
    "ollama_config": {"base_url": "http://localhost:11434", "model": "qwen3:4b"},
}


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_config_yaml_when_present(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(_SAMPLE_CONFIG), encoding="utf-8")
        result = load_config(str(cfg_file))
        assert result["topic"] == "contrastive learning"

    def test_falls_back_to_example_config(self, tmp_path: Path, monkeypatch) -> None:
        example = tmp_path / "config.example.yaml"
        example.write_text(yaml.dump({"topic": "fallback"}), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with patch(
            "redpill.config._CONFIG_CANDIDATES",
            ("config.yaml", str(example)),
        ):
            result = load_config()
        assert result["topic"] == "fallback"

    def test_explicit_path_is_used_exclusively(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.yaml"
        custom.write_text(yaml.dump({"topic": "custom"}), encoding="utf-8")
        result = load_config(str(custom))
        assert result["topic"] == "custom"

    def test_exits_1_when_no_file_found(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("redpill.config._CONFIG_CANDIDATES", ("nonexistent.yaml",)):
            with pytest.raises(SystemExit) as exc_info:
                load_config()
        assert exc_info.value.code == 1

    def test_exits_1_on_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("{broken: yaml: :", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            load_config(str(bad))
        assert exc_info.value.code == 1

    def test_exits_1_when_yaml_not_a_mapping(self, tmp_path: Path) -> None:
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            load_config(str(list_yaml))
        assert exc_info.value.code == 1

    def test_returns_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump(_SAMPLE_CONFIG), encoding="utf-8")
        result = load_config(str(cfg))
        assert isinstance(result, dict)


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
