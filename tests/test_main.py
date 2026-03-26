"""
tests/test_main.py — Unit tests for redpill.main

All external modules (search, extract, dedup, summarize, deliver, state) are
mocked.  No network I/O, no SQLite files, no Ollama connections are made.

Test structure:
    TestLoadConfig          — YAML loading, fallback order, error paths
    TestMergeSearchExtract  — merging of search + extraction result dicts
    TestContentHash         — SHA-256 hashing helper
    TestRunPipeline         — full pipeline: happy path, nothing-new, dry-run,
                              per-item failure resilience, state persistence
    TestCmdHistory          — history command: file discovery, --last N, edge cases
    TestCmdStats            — stats command: domain aggregation, output format
    TestBuildParser         — argparse structure and defaults
    TestCli                 — wiring of logging setup + load_dotenv
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Module imports — these must succeed even without the real dependencies
# because all heavy objects are mocked below.
# ---------------------------------------------------------------------------
from redpill.main import (
    _build_parser,
    _cmd_history,
    _cmd_queries,
    _cmd_run,
    _cmd_stats,
    _cmd_terms,
    _content_hash,
    _load_config,
    _maybe_deliver_nothing_new,
    _merge_search_and_extract,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "topic": "contrastive learning",
    "search_queries": ["contrastive learning 2026"],
    "max_results_per_query": 5,
    "dedup_similarity_threshold": 0.85,
    "delivery_method": "markdown",
    "output_dir": "data/digests",
    "search_provider": "tavily",
    "ollama_config": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:4b",
    },
}

SAMPLE_SEARCH_RESULTS = [
    {
        "url": "https://example.com/paper1",
        "title": "Paper One",
        "snippet": "A great paper about contrastive learning.",
        "published_date": "2026-03-07",
    },
    {
        "url": "https://example.com/paper2",
        "title": "Paper Two",
        "snippet": "Another paper.",
        "published_date": None,
    },
]

SAMPLE_EXTRACTED = [
    {
        "url": "https://example.com/paper1",
        "title": "Paper One (extracted)",
        "content": "Full body of paper one.",
        "extraction_success": True,
    },
    {
        "url": "https://example.com/paper2",
        "title": "",
        "content": None,
        "extraction_success": False,
    },
]

SAMPLE_SUMMARIZED = [
    {
        "url": "https://example.com/paper1",
        "title": "Paper One (LLM)",
        "summary": "A summary.",
        "key_insight": "Key insight here.",
        "relevance_score": 4,
    },
]


def _make_namespace(**kwargs) -> argparse.Namespace:
    defaults = {"config": None, "dry_run": False, "last": 5}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_config_yaml_when_present(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(SAMPLE_CONFIG), encoding="utf-8")
        with patch("redpill.main._CONFIG_CANDIDATES", (str(cfg_file),)):
            # _load_config("config.yaml") by passing the path explicitly
            result = _load_config(str(cfg_file))
        assert result["topic"] == "contrastive learning"

    def test_falls_back_to_example_config(self, tmp_path: Path, monkeypatch) -> None:
        example = tmp_path / "config.example.yaml"
        example.write_text(yaml.dump({"topic": "fallback"}), encoding="utf-8")
        # Patch candidate order so config.yaml does not exist but example does.
        monkeypatch.chdir(tmp_path)
        with patch(
            "redpill.main._CONFIG_CANDIDATES",
            ("config.yaml", str(example)),
        ):
            result = _load_config()
        assert result["topic"] == "fallback"

    def test_explicit_path_is_used_exclusively(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.yaml"
        custom.write_text(yaml.dump({"topic": "custom"}), encoding="utf-8")
        result = _load_config(str(custom))
        assert result["topic"] == "custom"

    def test_exits_1_when_no_file_found(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("redpill.main._CONFIG_CANDIDATES", ("nonexistent.yaml",)):
            with pytest.raises(SystemExit) as exc_info:
                _load_config()
        assert exc_info.value.code == 1

    def test_exits_1_on_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("{broken: yaml: :", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            _load_config(str(bad))
        assert exc_info.value.code == 1

    def test_exits_1_when_yaml_not_a_mapping(self, tmp_path: Path) -> None:
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            _load_config(str(list_yaml))
        assert exc_info.value.code == 1

    def test_returns_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump(SAMPLE_CONFIG), encoding="utf-8")
        result = _load_config(str(cfg))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestMergeSearchExtract
# ---------------------------------------------------------------------------


class TestMergeSearchExtract:
    def test_prefers_extraction_title(self) -> None:
        merged = _merge_search_and_extract(
            SAMPLE_SEARCH_RESULTS[:1], SAMPLE_EXTRACTED[:1]
        )
        assert merged[0]["title"] == "Paper One (extracted)"

    def test_falls_back_to_search_title_when_extraction_empty(self) -> None:
        merged = _merge_search_and_extract(
            SAMPLE_SEARCH_RESULTS[1:], SAMPLE_EXTRACTED[1:]
        )
        assert merged[0]["title"] == "Paper Two"

    def test_content_none_when_extraction_failed(self) -> None:
        merged = _merge_search_and_extract(
            SAMPLE_SEARCH_RESULTS[1:], SAMPLE_EXTRACTED[1:]
        )
        assert merged[0]["content"] is None

    def test_content_present_when_extraction_succeeded(self) -> None:
        merged = _merge_search_and_extract(
            SAMPLE_SEARCH_RESULTS[:1], SAMPLE_EXTRACTED[:1]
        )
        assert merged[0]["content"] == "Full body of paper one."

    def test_snippet_preserved_from_search(self) -> None:
        merged = _merge_search_and_extract(
            SAMPLE_SEARCH_RESULTS[:1], SAMPLE_EXTRACTED[:1]
        )
        assert merged[0]["snippet"] == "A great paper about contrastive learning."

    def test_output_length_matches_search_results(self) -> None:
        merged = _merge_search_and_extract(SAMPLE_SEARCH_RESULTS, SAMPLE_EXTRACTED)
        assert len(merged) == len(SAMPLE_SEARCH_RESULTS)

    def test_urls_preserved(self) -> None:
        merged = _merge_search_and_extract(SAMPLE_SEARCH_RESULTS, SAMPLE_EXTRACTED)
        assert merged[0]["url"] == "https://example.com/paper1"
        assert merged[1]["url"] == "https://example.com/paper2"

    def test_missing_extract_entry_produces_none_content(self) -> None:
        """URL in search but not in extract list — content must be None."""
        merged = _merge_search_and_extract(SAMPLE_SEARCH_RESULTS[:1], [])
        assert merged[0]["content"] is None
        assert merged[0]["extraction_success"] is False

    def test_order_preserved(self) -> None:
        merged = _merge_search_and_extract(SAMPLE_SEARCH_RESULTS, SAMPLE_EXTRACTED)
        assert [m["url"] for m in merged] == [r["url"] for r in SAMPLE_SEARCH_RESULTS]


# ---------------------------------------------------------------------------
# TestContentHash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_returns_64_char_hex_string(self) -> None:
        h = _content_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_none_input_same_as_empty_string(self) -> None:
        assert _content_hash(None) == _content_hash("")

    def test_different_content_different_hash(self) -> None:
        assert _content_hash("abc") != _content_hash("xyz")

    def test_same_content_same_hash(self) -> None:
        assert _content_hash("deterministic") == _content_hash("deterministic")

    def test_known_hash(self) -> None:
        import hashlib

        expected = hashlib.sha256(b"test").hexdigest()
        assert _content_hash("test") == expected


# ---------------------------------------------------------------------------
# TestRunPipeline — mocking all external dependencies
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """run_pipeline integrates many moving parts.  We mock at module boundaries."""

    def _patch_all(
        self,
        *,
        search_return=None,
        extract_return=None,
        filter_return=None,
        summarize_return=None,
        config: dict | None = None,
    ):
        """Return a context-manager stack that patches every external call."""
        if search_return is None:
            search_return = SAMPLE_SEARCH_RESULTS
        if extract_return is None:
            extract_return = SAMPLE_EXTRACTED
        if filter_return is None:
            # filter_new_items must return the merged items, not the raw search
            # items, so we produce a reasonable merged result here.
            filter_return = [
                {
                    "url": "https://example.com/paper1",
                    "title": "Paper One (extracted)",
                    "snippet": "A great paper about contrastive learning.",
                    "content": "Full body of paper one.",
                    "extraction_success": True,
                }
            ]
        if summarize_return is None:
            summarize_return = {
                "url": "https://example.com/paper1",
                "title": "Paper One (LLM)",
                "summary": "A summary.",
                "key_insight": "Key insight here.",
                "relevance_score": 4,
            }
        if config is None:
            config = SAMPLE_CONFIG

        return {
            "load_config": patch("redpill.main._load_config", return_value=config),
            "init_db": patch("redpill.main.init_db"),
            "check_ollama": patch("redpill.main.check_ollama"),
            "OllamaClient": patch("redpill.main.OllamaClient"),
            "search": patch("redpill.main.search", return_value=search_return),
            "extract_batch": patch(
                "redpill.main.extract_batch", return_value=extract_return
            ),
            "filter_new_items": patch(
                "redpill.main.filter_new_items", return_value=filter_return
            ),
            "summarize_item": patch(
                "redpill.main.summarize_item", return_value=summarize_return
            ),
            "generate_digest": patch(
                "redpill.main.generate_digest", return_value="# digest\n"
            ),
            "deliver": patch("redpill.main.deliver", return_value=None),
            "add_item": patch("redpill.main.add_item"),
            "compute_embedding": patch(
                "redpill.main.compute_embedding",
                return_value=np.zeros(384, dtype=np.float32),
            ),
        }

    def test_happy_path_calls_all_steps(self, tmp_path: Path) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"] as m_cfg,
            patches["init_db"] as m_init,
            patches["check_ollama"] as m_chk,
            patches["OllamaClient"],
            patches["search"] as m_search,
            patches["extract_batch"] as m_extract,
            patches["filter_new_items"] as m_filter,
            patches["summarize_item"] as m_sum,
            patches["generate_digest"] as m_digest,
            patches["deliver"] as m_deliver,
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            run_pipeline()

        m_init.assert_called_once()
        m_chk.assert_called_once()
        m_search.assert_called_once()
        m_extract.assert_called_once()
        m_filter.assert_called_once()
        m_sum.assert_called_once()
        m_digest.assert_called_once()
        m_deliver.assert_called_once()
        m_add.assert_called_once()

    def test_dry_run_skips_deliver_and_add_item(
        self, tmp_path: Path, capsys
    ) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"] as m_deliver,
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            run_pipeline(dry_run=True)

        m_deliver.assert_not_called()
        m_add.assert_not_called()

    def test_dry_run_prints_digest_to_stdout(self, capsys) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"] as m_digest,
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
        ):
            m_digest.return_value = "DIGEST_CONTENT"
            run_pipeline(dry_run=True)

        captured = capsys.readouterr()
        assert "DIGEST_CONTENT" in captured.out

    def test_nothing_new_after_dedup_delivers_empty_digest(self) -> None:
        patches = self._patch_all(filter_return=[])
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"] as m_sum,
            patches["generate_digest"] as m_digest,
            patches["deliver"] as m_deliver,
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            run_pipeline()

        m_sum.assert_not_called()
        m_add.assert_not_called()
        # deliver should still be called with the "nothing new" digest
        m_deliver.assert_called_once()

    def test_empty_search_results_delivers_empty_digest(self) -> None:
        patches = self._patch_all(search_return=[])
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"] as m_extract,
            patches["filter_new_items"] as m_filter,
            patches["summarize_item"] as m_sum,
            patches["generate_digest"] as m_digest,
            patches["deliver"] as m_deliver,
            patches["add_item"],
            patches["compute_embedding"],
        ):
            run_pipeline()

        m_extract.assert_not_called()
        m_filter.assert_not_called()
        m_sum.assert_not_called()
        m_deliver.assert_called_once()

    def test_ollama_check_failure_exits_1(self) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"] as m_chk,
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
        ):
            m_chk.side_effect = RuntimeError("Ollama not running")
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_delivery_failure_exits_1(self) -> None:
        from redpill.deliver import DeliveryError

        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"] as m_deliver,
            patches["add_item"],
            patches["compute_embedding"],
        ):
            m_deliver.side_effect = DeliveryError("disk full")
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_summarize_item_exception_skips_item_does_not_abort(self) -> None:
        """An unexpected exception from summarize_item must not abort the pipeline."""
        two_items = [
            {
                "url": "https://example.com/paper1",
                "title": "Paper One",
                "snippet": "snippet",
                "content": "content",
                "extraction_success": True,
            },
            {
                "url": "https://example.com/paper2",
                "title": "Paper Two",
                "snippet": "snippet",
                "content": "content",
                "extraction_success": True,
            },
        ]
        patches = self._patch_all(filter_return=two_items)
        # First call raises; second returns a valid result.
        ok_result = {
            "url": "https://example.com/paper2",
            "title": "Paper Two (LLM)",
            "summary": "Summary.",
            "key_insight": "Insight.",
            "relevance_score": 3,
        }
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patch(
                "redpill.main.summarize_item",
                side_effect=[RuntimeError("LLM died"), ok_result],
            ) as m_sum,
            patches["generate_digest"] as m_digest,
            patches["deliver"],
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            run_pipeline()

        # Both items were attempted
        assert m_sum.call_count == 2
        # generate_digest called with only the one successful result
        digest_items = m_digest.call_args[0][0]
        assert len(digest_items) == 1
        # Only the successful item is persisted
        m_add.assert_called_once()

    def test_state_persistence_failure_does_not_abort(self) -> None:
        """A failure in add_item must not abort the whole pipeline."""
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            m_add.side_effect = Exception("DB locked")
            # Must not raise — just log a warning and continue
            run_pipeline()

    def test_missing_topic_in_config_exits_1(self) -> None:
        bad_config = {**SAMPLE_CONFIG, "topic": ""}
        patches = self._patch_all(config=bad_config)
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_empty_search_queries_exits_1(self) -> None:
        bad_config = {**SAMPLE_CONFIG, "search_queries": []}
        patches = self._patch_all(config=bad_config)
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_add_item_called_with_correct_url(self) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            run_pipeline()

        _, kwargs = m_add.call_args
        assert kwargs["url"] == "https://example.com/paper1"

    def test_add_item_called_with_topic_from_config(self) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"] as m_add,
            patches["compute_embedding"],
        ):
            run_pipeline()

        _, kwargs = m_add.call_args
        assert kwargs["topic"] == "contrastive learning"

    def test_search_exception_exits_1(self) -> None:
        patches = self._patch_all()
        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"] as m_search,
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
        ):
            m_search.side_effect = Exception("API down")
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_serper_provider_without_api_key_exits_1(self) -> None:
        """Fail fast when search_provider requires SERPER_API_KEY but it is absent."""
        serper_config = {**SAMPLE_CONFIG, "search_provider": "serper"}
        patches = self._patch_all(config=serper_config)
        import os

        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
            patch.dict(os.environ, {}, clear=True),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_both_provider_without_api_key_exits_1(self) -> None:
        """Fail fast when search_provider is 'both' but SERPER_API_KEY is absent."""
        both_config = {**SAMPLE_CONFIG, "search_provider": "both"}
        patches = self._patch_all(config=both_config)
        import os

        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
            patch.dict(os.environ, {}, clear=True),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_invalid_search_provider_exits_1(self) -> None:
        """An unrecognised search_provider value must exit 1 with a clear message."""
        bad_config = {**SAMPLE_CONFIG, "search_provider": "bing"}
        patches = self._patch_all(config=bad_config)

        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
        assert exc_info.value.code == 1

    def test_tavily_provider_with_key_present_does_not_exit(self) -> None:
        """Default tavily provider must not require SERPER_API_KEY."""
        patches = self._patch_all()  # SAMPLE_CONFIG has search_provider: "tavily"
        import os

        with (
            patches["load_config"],
            patches["init_db"],
            patches["check_ollama"],
            patches["OllamaClient"],
            patches["search"],
            patches["extract_batch"],
            patches["filter_new_items"],
            patches["summarize_item"],
            patches["generate_digest"],
            patches["deliver"],
            patches["add_item"],
            patches["compute_embedding"],
            patch.dict(os.environ, {}, clear=True),
        ):
            # Should complete without SystemExit even with no env vars
            run_pipeline()


# ---------------------------------------------------------------------------
# TestCmdHistory
# ---------------------------------------------------------------------------


class TestCmdHistory:
    def _write_digests(self, output_dir: Path, names: list[str]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (output_dir / name).write_text(f"# {name}\n", encoding="utf-8")

    def test_prints_last_n_digests(self, tmp_path: Path, capsys) -> None:
        output_dir = tmp_path / "digests"
        self._write_digests(output_dir, ["2026-03-05.md", "2026-03-06.md", "2026-03-07.md"])
        config = {**SAMPLE_CONFIG, "output_dir": str(output_dir)}
        args = _make_namespace(last=2)
        with patch("redpill.main._load_config", return_value=config):
            _cmd_history(args)
        captured = capsys.readouterr()
        # The two most recent files should appear; the oldest should not.
        # (exact content depends on mtime order, which may be creation order)
        assert "---" in captured.out

    def test_no_digests_prints_message(self, tmp_path: Path, capsys) -> None:
        output_dir = tmp_path / "digests"
        output_dir.mkdir()
        config = {**SAMPLE_CONFIG, "output_dir": str(output_dir)}
        args = _make_namespace(last=5)
        with patch("redpill.main._load_config", return_value=config):
            _cmd_history(args)
        captured = capsys.readouterr()
        assert "No digests" in captured.out

    def test_missing_output_dir_prints_message(self, tmp_path: Path, capsys) -> None:
        config = {**SAMPLE_CONFIG, "output_dir": str(tmp_path / "nonexistent")}
        args = _make_namespace(last=5)
        with patch("redpill.main._load_config", return_value=config):
            _cmd_history(args)
        captured = capsys.readouterr()
        assert "No digests" in captured.out

    def test_separator_between_digests(self, tmp_path: Path, capsys) -> None:
        output_dir = tmp_path / "digests"
        self._write_digests(output_dir, ["2026-03-06.md", "2026-03-07.md"])
        config = {**SAMPLE_CONFIG, "output_dir": str(output_dir)}
        args = _make_namespace(last=2)
        with patch("redpill.main._load_config", return_value=config):
            _cmd_history(args)
        captured = capsys.readouterr()
        assert "\n---\n" in captured.out

    def test_single_digest_no_separator(self, tmp_path: Path, capsys) -> None:
        output_dir = tmp_path / "digests"
        self._write_digests(output_dir, ["2026-03-07.md"])
        config = {**SAMPLE_CONFIG, "output_dir": str(output_dir)}
        args = _make_namespace(last=1)
        with patch("redpill.main._load_config", return_value=config):
            _cmd_history(args)
        captured = capsys.readouterr()
        # Only one digest → no separator
        assert "\n---\n" not in captured.out


# ---------------------------------------------------------------------------
# TestCmdStats
# ---------------------------------------------------------------------------


class TestCmdStats:
    def _make_db(self, tmp_path: Path) -> str:
        """Create a minimal populated SQLite DB and return its path."""
        db_path = str(tmp_path / "redpill.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE seen_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                embedding BLOB,
                summary TEXT NOT NULL DEFAULT '',
                first_seen_date TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.executemany(
            "INSERT INTO seen_items (url, title, first_seen_date, topic) VALUES (?, ?, ?, ?)",
            [
                ("https://arxiv.org/abs/1", "A", "2026-03-01", "ml"),
                ("https://arxiv.org/abs/2", "B", "2026-03-02", "ml"),
                ("https://openai.com/blog/x", "C", "2026-03-03", "ml"),
            ],
        )
        conn.commit()
        conn.close()
        return db_path

    def test_prints_total_items(self, tmp_path: Path, capsys) -> None:
        db_path = self._make_db(tmp_path)
        config = {**SAMPLE_CONFIG, "db_path": db_path}
        args = _make_namespace()
        with patch("redpill.main._load_config", return_value=config):
            _cmd_stats(args)
        captured = capsys.readouterr()
        assert "3" in captured.out

    def test_prints_top_sources(self, tmp_path: Path, capsys) -> None:
        db_path = self._make_db(tmp_path)
        config = {**SAMPLE_CONFIG, "db_path": db_path}
        args = _make_namespace()
        with patch("redpill.main._load_config", return_value=config):
            _cmd_stats(args)
        captured = capsys.readouterr()
        assert "arxiv.org" in captured.out

    def test_prints_avg_per_day(self, tmp_path: Path, capsys) -> None:
        db_path = self._make_db(tmp_path)
        config = {**SAMPLE_CONFIG, "db_path": db_path}
        args = _make_namespace()
        with patch("redpill.main._load_config", return_value=config):
            _cmd_stats(args)
        captured = capsys.readouterr()
        assert "Avg" in captured.out or "avg" in captured.out.lower()

    def test_missing_db_prints_message(self, tmp_path: Path, capsys) -> None:
        config = {**SAMPLE_CONFIG, "db_path": str(tmp_path / "nonexistent.db")}
        args = _make_namespace()
        with patch("redpill.main._load_config", return_value=config):
            _cmd_stats(args)
        captured = capsys.readouterr()
        assert "No database" in captured.out

    def test_empty_db_prints_message(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE seen_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT DEFAULT '',
                content_hash TEXT DEFAULT '',
                embedding BLOB,
                summary TEXT DEFAULT '',
                first_seen_date TEXT NOT NULL,
                topic TEXT DEFAULT ''
            )
            """
        )
        conn.commit()
        conn.close()
        config = {**SAMPLE_CONFIG, "db_path": db_path}
        args = _make_namespace()
        with patch("redpill.main._load_config", return_value=config):
            _cmd_stats(args)
        captured = capsys.readouterr()
        assert "No items" in captured.out

    def test_top_sources_limited_to_five(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "big.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE seen_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT DEFAULT '',
                content_hash TEXT DEFAULT '',
                embedding BLOB,
                summary TEXT DEFAULT '',
                first_seen_date TEXT NOT NULL,
                topic TEXT DEFAULT ''
            )
            """
        )
        rows = [
            (f"https://site{i}.com/article", f"T{i}", "2026-03-01", "ml")
            for i in range(10)
        ]
        conn.executemany(
            "INSERT INTO seen_items (url, title, first_seen_date, topic) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        config = {**SAMPLE_CONFIG, "db_path": db_path}
        args = _make_namespace()
        with patch("redpill.main._load_config", return_value=config):
            _cmd_stats(args)
        captured = capsys.readouterr()
        # At most 5 domain lines should appear (each starts with spaces + count)
        domain_lines = [
            line
            for line in captured.out.splitlines()
            if line.strip() and line.strip().split()[0].isdigit()
        ]
        assert len(domain_lines) <= 5


# ---------------------------------------------------------------------------
# TestBuildParser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_run_subcommand_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_run_dry_run_flag_defaults_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.dry_run is False

    def test_run_dry_run_flag_set(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--dry-run"])
        assert args.dry_run is True

    def test_run_config_defaults_to_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.config is None

    def test_run_config_can_be_set(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--config", "myconfig.yaml"])
        assert args.config == "myconfig.yaml"

    def test_history_subcommand_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["history"])
        assert args.command == "history"

    def test_history_last_defaults_to_5(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["history"])
        assert args.last == 5

    def test_history_last_can_be_set(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["history", "--last", "10"])
        assert args.last == 10

    def test_stats_subcommand_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_no_subcommand_exits_nonzero(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_run_has_func_attribute(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert callable(args.func)

    def test_history_has_func_attribute(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["history"])
        assert callable(args.func)

    def test_stats_has_func_attribute(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["stats"])
        assert callable(args.func)

    def test_plan_subcommand_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plan"])
        assert args.command == "plan"

    def test_plan_max_queries_defaults_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plan"])
        assert args.max_queries is None

    def test_plan_max_queries_can_be_set(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["plan", "--max-queries", "7"])
        assert args.max_queries == 7

    def test_queries_subcommand_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queries"])
        assert args.command == "queries"

    def test_queries_last_defaults_14(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queries"])
        assert args.last == 14

    def test_queries_last_can_be_set(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queries", "--last", "30"])
        assert args.last == 30

    def test_queries_has_func_attribute(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queries"])
        assert callable(args.func)

    def test_terms_subcommand_exists(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["terms"])
        assert args.command == "terms"

    def test_terms_top_defaults_20(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["terms"])
        assert args.top == 20

    def test_terms_top_can_be_set(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["terms", "--top", "50"])
        assert args.top == 50

    def test_terms_recent_flag_defaults_30_when_no_value(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["terms", "--recent"])
        assert args.recent == 30

    def test_terms_recent_accepts_explicit_days(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["terms", "--recent", "7"])
        assert args.recent == 7

    def test_terms_top_and_recent_mutually_exclusive(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["terms", "--top", "10", "--recent", "7"])

    def test_terms_has_func_attribute(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["terms"])
        assert callable(args.func)


# ---------------------------------------------------------------------------
# TestCmdQueries
# ---------------------------------------------------------------------------


class TestCmdQueries:
    def _make_args(self, config: str, last: int = 14) -> argparse.Namespace:
        return argparse.Namespace(config=config, last=last)

    def test_no_db_prints_message(self, tmp_path: Path, capsys) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": str(tmp_path / "missing.db")}))
        _cmd_queries(self._make_args(str(cfg)))
        out = capsys.readouterr().out
        assert "No database found" in out

    def test_empty_history_prints_message(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "test.db")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": db_path}))
        # Create the DB so the "no db" check passes, but leave query_log empty.
        from redpill.state import init_db
        init_db(db_path)
        _cmd_queries(self._make_args(str(cfg)))
        out = capsys.readouterr().out
        assert "No query history" in out

    def test_shows_query_rows(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "test.db")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": db_path}))
        from redpill.state import init_db, log_query, update_query_stats
        init_db(db_path)
        from datetime import date
        today = date.today().isoformat()
        qid = log_query("contrastive learning SimCLR", today, "llm_planned",
                        "contrastive learning", db_path=db_path)
        update_query_stats(qid, 10, 4, 2, db_path=db_path)
        _cmd_queries(self._make_args(str(cfg)))
        out = capsys.readouterr().out
        assert "contrastive learning SimCLR" in out
        assert "llm_planned" in out

    def test_missing_topic_exits(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"search_queries": ["q"]}))
        with pytest.raises(SystemExit):
            _cmd_queries(self._make_args(str(cfg)))


# ---------------------------------------------------------------------------
# TestCmdTerms
# ---------------------------------------------------------------------------


class TestCmdTerms:
    def _make_args(self, config: str, top: int = 20, recent=None) -> argparse.Namespace:
        return argparse.Namespace(config=config, top=top, recent=recent)

    def test_no_db_prints_message(self, tmp_path: Path, capsys) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": str(tmp_path / "missing.db")}))
        _cmd_terms(self._make_args(str(cfg)))
        out = capsys.readouterr().out
        assert "No database found" in out

    def test_empty_terms_prints_message(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "test.db")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": db_path}))
        from redpill.state import init_db
        init_db(db_path)
        _cmd_terms(self._make_args(str(cfg)))
        out = capsys.readouterr().out
        assert "No terms found" in out

    def test_shows_top_terms(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "test.db")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": db_path}))
        from datetime import date
        from redpill.state import init_db, store_extracted_terms
        init_db(db_path)
        today = date.today().isoformat()
        store_extracted_terms([
            {"term": "SimCLR", "topic": "contrastive learning", "category": "technique",
             "first_seen": today, "last_seen": today},
        ], db_path=db_path)
        _cmd_terms(self._make_args(str(cfg), top=5))
        out = capsys.readouterr().out
        assert "SimCLR" in out
        assert "technique" in out

    def test_recent_flag_uses_get_recent_terms(self, tmp_path: Path, capsys) -> None:
        db_path = str(tmp_path / "test.db")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({**SAMPLE_CONFIG, "db_path": db_path}))
        from datetime import date
        from redpill.state import init_db, store_extracted_terms
        init_db(db_path)
        today = date.today().isoformat()
        store_extracted_terms([
            {"term": "MoCo", "topic": "contrastive learning", "category": "framework",
             "first_seen": today, "last_seen": today},
        ], db_path=db_path)
        _cmd_terms(self._make_args(str(cfg), recent=30))
        out = capsys.readouterr().out
        assert "MoCo" in out
        assert "last 30 day" in out

    def test_missing_topic_exits(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"search_queries": ["q"]}))
        with pytest.raises(SystemExit):
            _cmd_terms(self._make_args(str(cfg)))


# ---------------------------------------------------------------------------
# TestCli
# ---------------------------------------------------------------------------


class TestCli:
    def test_cli_calls_load_dotenv(self) -> None:
        from redpill.main import cli

        with (
            patch("redpill.main.load_dotenv") as m_dotenv,
            patch("redpill.main._build_parser") as m_parser,
        ):
            mock_args = MagicMock()
            mock_args.func = MagicMock()
            m_parser.return_value.parse_args.return_value = mock_args
            cli()

        m_dotenv.assert_called_once()

    def test_cli_calls_args_func(self) -> None:
        from redpill.main import cli

        with (
            patch("redpill.main.load_dotenv"),
            patch("redpill.main._build_parser") as m_parser,
        ):
            mock_args = MagicMock()
            m_parser.return_value.parse_args.return_value = mock_args
            cli()

        mock_args.func.assert_called_once_with(mock_args)
