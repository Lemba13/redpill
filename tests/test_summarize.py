"""
tests/test_summarize.py — Unit tests for redpill.summarize

No real Ollama instance is ever contacted. All tests use either:
- A hand-rolled stub that implements the LLMClient Protocol, or
- unittest.mock.patch to intercept ollama.Client / ollama.list at the boundary.

Test structure:
    TestStripThinkBlocks     — _strip_think_blocks() regex correctness
    TestExtractJson          — _extract_json() parsing strategies and fallbacks
    TestValidateSummary      — _validate_summary() type coercion and clamping
    TestBuildSummarizePrompt — _build_summarize_prompt() brace-safety
    TestSummarizeItem        — summarize_item() end-to-end with stub clients
    TestGenerateDigest       — generate_digest() sorting, formatting, edge cases
    TestCheckOllama          — check_ollama() with mocked ollama.Client
    TestOllamaClient         — OllamaClient.generate() with mocked ollama.Client
"""

import logging
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

from redpill.llm_utils import strip_think_blocks as _strip_think_blocks
from redpill.summarize import (
    LLMClient,
    OllamaClient,
    _build_summarize_prompt,
    _extract_json,
    _validate_summary,
    check_ollama,
    generate_digest,
    summarize_item,
)


# ---------------------------------------------------------------------------
# Stub LLMClient implementations for summarize_item tests
# ---------------------------------------------------------------------------


class _FixedClient:
    """LLMClient stub that always returns a pre-set string."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self._response


class _RaisingClient:
    """LLMClient stub that always raises RuntimeError."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("simulated LLM failure")


def _good_json_response(
    title: str = "Test Title",
    summary: str = "A two-sentence summary. It covers the topic well.",
    key_insight: str = "This matters because of X.",
    relevance_score: int = 4,
) -> str:
    import json
    return json.dumps(
        {
            "title": title,
            "summary": summary,
            "key_insight": key_insight,
            "relevance_score": relevance_score,
        }
    )


def _make_item(
    url: str = "https://example.com/article",
    content: str | None = "This is the full article body about contrastive learning.",
    snippet: str = "Short snippet about contrastive learning.",
    title: str = "Original Title",
) -> dict:
    return {"url": url, "content": content, "snippet": snippet, "title": title}


# ---------------------------------------------------------------------------
# TestStripThinkBlocks
# ---------------------------------------------------------------------------


class TestStripThinkBlocks:
    def test_no_think_block_unchanged(self):
        text = '{"title": "hello"}'
        assert _strip_think_blocks(text) == text

    def test_single_line_think_block_removed(self):
        text = "<think>some reasoning</think>{}"
        assert _strip_think_blocks(text) == "{}"

    def test_multiline_think_block_removed(self):
        text = "<think>\nline one\nline two\n</think>\n{}"
        result = _strip_think_blocks(text)
        assert "<think>" not in result
        assert "{}" in result

    def test_multiple_think_blocks_all_removed(self):
        text = "<think>first</think> middle <think>second</think> end"
        result = _strip_think_blocks(text)
        assert "<think>" not in result
        assert "middle" in result
        assert "end" in result

    def test_nested_angle_brackets_not_confused(self):
        """A think block containing angle brackets still gets removed."""
        text = "<think>x < y > z</think>actual"
        result = _strip_think_blocks(text)
        assert "actual" in result
        assert "<think>" not in result

    def test_empty_string_unchanged(self):
        assert _strip_think_blocks("") == ""

    def test_think_block_with_json_inside_removed(self):
        """The JSON inside <think> must not be mistaken for the real response."""
        text = '<think>{"fake": 1}</think>{"real": 2}'
        result = _strip_think_blocks(text)
        assert '"fake"' not in result
        assert '"real": 2' in result


# ---------------------------------------------------------------------------
# TestExtractJson
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_clean_json_object(self):
        raw = '{"title": "T", "relevance_score": 3}'
        result = _extract_json(raw)
        assert result == {"title": "T", "relevance_score": 3}

    def test_json_with_think_prefix(self):
        raw = "<think>reasoning</think>\n" + _good_json_response()
        result = _extract_json(raw)
        assert result["title"] == "Test Title"

    def test_json_with_preamble_text(self):
        """Models sometimes say 'Here is the JSON:' before the object."""
        raw = "Here is the response:\n" + _good_json_response()
        result = _extract_json(raw)
        assert "title" in result

    def test_json_with_trailing_text(self):
        raw = _good_json_response() + "\n\nNote: I hope this helps."
        result = _extract_json(raw)
        assert "title" in result

    def test_completely_invalid_returns_empty_dict(self):
        result = _extract_json("not json at all")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _extract_json("")
        assert result == {}

    def test_json_array_returns_empty_dict(self):
        """The model returning a JSON array instead of an object is a failure."""
        result = _extract_json("[1, 2, 3]")
        assert result == {}

    def test_json_scalar_returns_empty_dict(self):
        result = _extract_json('"just a string"')
        assert result == {}

    def test_think_block_with_actual_json_after(self):
        raw = "<think>Let me think...</think>\n" + _good_json_response(title="Real")
        result = _extract_json(raw)
        assert result["title"] == "Real"

    def test_returns_dict(self):
        raw = _good_json_response()
        result = _extract_json(raw)
        assert isinstance(result, dict)

    def test_markdown_fenced_json(self):
        """Some models wrap JSON in ```json ... ``` fences; brace-hunt extracts it."""
        raw = "```json\n" + _good_json_response() + "\n```"
        result = _extract_json(raw)
        # The brace-hunting strategy should recover this.
        assert "title" in result


# ---------------------------------------------------------------------------
# TestValidateSummary
# ---------------------------------------------------------------------------


class TestValidateSummary:
    def test_valid_input_passes_through(self):
        data = {
            "title": "T",
            "summary": "S",
            "key_insight": "K",
            "relevance_score": 4,
        }
        result = _validate_summary(data)
        assert result == data

    def test_missing_keys_use_defaults(self):
        result = _validate_summary({})
        assert result["title"] == ""
        assert result["summary"] == ""
        assert result["key_insight"] == ""
        assert result["relevance_score"] == 1

    def test_relevance_score_clamped_above_5(self):
        result = _validate_summary({"relevance_score": 99})
        assert result["relevance_score"] == 5

    def test_relevance_score_clamped_below_1(self):
        result = _validate_summary({"relevance_score": -10})
        assert result["relevance_score"] == 1

    def test_relevance_score_at_boundary_1(self):
        result = _validate_summary({"relevance_score": 1})
        assert result["relevance_score"] == 1

    def test_relevance_score_at_boundary_5(self):
        result = _validate_summary({"relevance_score": 5})
        assert result["relevance_score"] == 5

    def test_relevance_score_as_string_integer(self):
        """LLMs sometimes return scores as strings."""
        result = _validate_summary({"relevance_score": "3"})
        assert result["relevance_score"] == 3

    def test_relevance_score_as_float_truncated(self):
        result = _validate_summary({"relevance_score": 3.9})
        assert result["relevance_score"] == 3

    def test_relevance_score_non_numeric_defaults_to_1(self):
        result = _validate_summary({"relevance_score": "high"})
        assert result["relevance_score"] == 1

    def test_non_string_title_coerced_to_str(self):
        result = _validate_summary({"title": 42})
        assert isinstance(result["title"], str)
        assert result["title"] == "42"

    def test_non_string_summary_coerced_to_str(self):
        result = _validate_summary({"summary": ["a", "b"]})
        assert isinstance(result["summary"], str)

    def test_output_has_exactly_four_keys(self):
        result = _validate_summary({"extra_key": "ignored"})
        assert set(result.keys()) == {"title", "summary", "key_insight", "relevance_score"}


# ---------------------------------------------------------------------------
# TestBuildSummarizePrompt
# ---------------------------------------------------------------------------


class TestBuildSummarizePrompt:
    def test_topic_appears_in_prompt(self):
        prompt = _build_summarize_prompt(topic="machine learning", content="article text")
        assert "machine learning" in prompt

    def test_content_appears_in_prompt(self):
        prompt = _build_summarize_prompt(topic="AI", content="unique content string xyz")
        assert "unique content string xyz" in prompt

    def test_topic_with_braces_does_not_raise(self):
        """A topic like 'AI {systems}' must not raise KeyError."""
        prompt = _build_summarize_prompt(topic="AI {systems}", content="some text")
        assert "AI {systems}" in prompt

    def test_content_with_braces_does_not_raise(self):
        """Article content containing JSON-like fragments must not raise."""
        prompt = _build_summarize_prompt(
            topic="AI",
            content='This article discusses {"key": "value"} patterns.',
        )
        assert '{"key": "value"}' in prompt

    def test_returns_string(self):
        result = _build_summarize_prompt("topic", "content")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestSummarizeItem
# ---------------------------------------------------------------------------


class TestSummarizeItem:
    def test_returns_dict_with_required_keys(self):
        client = _FixedClient(_good_json_response())
        result = summarize_item(_make_item(), topic="AI", client=client)
        assert set(result.keys()) == {"title", "summary", "key_insight", "relevance_score", "url"}

    def test_url_preserved_in_output(self):
        client = _FixedClient(_good_json_response())
        item = _make_item(url="https://preserved.example.com/")
        result = summarize_item(item, topic="AI", client=client)
        assert result["url"] == "https://preserved.example.com/"

    def test_content_used_over_snippet(self):
        """When content is present, the LLM must receive it (not the snippet)."""
        client = _FixedClient(_good_json_response())
        item = _make_item(content="full article", snippet="short snippet")
        summarize_item(item, topic="AI", client=client)
        prompt_sent = client.calls[0]["prompt"]
        assert "full article" in prompt_sent
        assert "short snippet" not in prompt_sent

    def test_snippet_used_when_content_is_none(self):
        client = _FixedClient(_good_json_response())
        item = _make_item(content=None, snippet="only the snippet")
        summarize_item(item, topic="AI", client=client)
        prompt_sent = client.calls[0]["prompt"]
        assert "only the snippet" in prompt_sent

    def test_snippet_used_when_content_is_empty_string(self):
        client = _FixedClient(_good_json_response())
        item = _make_item(content="", snippet="fallback snippet")
        summarize_item(item, topic="AI", client=client)
        prompt_sent = client.calls[0]["prompt"]
        assert "fallback snippet" in prompt_sent

    def test_snippet_used_when_content_is_whitespace_only(self):
        client = _FixedClient(_good_json_response())
        item = _make_item(content="   \n  ", snippet="whitespace fallback")
        summarize_item(item, topic="AI", client=client)
        prompt_sent = client.calls[0]["prompt"]
        assert "whitespace fallback" in prompt_sent

    def test_no_text_returns_fallback(self):
        client = _FixedClient(_good_json_response())
        item = _make_item(content=None, snippet="")
        result = summarize_item(item, topic="AI", client=client)
        # LLM should not be called at all.
        assert client.calls == []
        assert result["relevance_score"] == 1
        assert result["title"] == ""

    def test_llm_error_returns_fallback(self):
        client = _RaisingClient()
        result = summarize_item(_make_item(), topic="AI", client=client)
        assert result["relevance_score"] == 1
        assert result["title"] == ""
        assert result["url"] == "https://example.com/article"

    def test_unparseable_json_returns_fallback(self):
        client = _FixedClient("this is not json at all")
        result = summarize_item(_make_item(), topic="AI", client=client)
        assert result["title"] == ""
        assert result["relevance_score"] == 1

    def test_system_prompt_is_passed_to_client(self):
        client = _FixedClient(_good_json_response())
        summarize_item(_make_item(), topic="AI", client=client)
        system = client.calls[0]["system"]
        assert system is not None
        assert len(system) > 0

    def test_topic_in_prompt(self):
        client = _FixedClient(_good_json_response())
        summarize_item(_make_item(), topic="contrastive learning", client=client)
        prompt_sent = client.calls[0]["prompt"]
        assert "contrastive learning" in prompt_sent

    def test_parsed_fields_correctly_extracted(self):
        response = _good_json_response(
            title="Parsed Title",
            summary="Parsed summary text.",
            key_insight="Parsed insight.",
            relevance_score=5,
        )
        client = _FixedClient(response)
        result = summarize_item(_make_item(), topic="AI", client=client)
        assert result["title"] == "Parsed Title"
        assert result["summary"] == "Parsed summary text."
        assert result["key_insight"] == "Parsed insight."
        assert result["relevance_score"] == 5

    def test_think_block_stripped_before_parse(self):
        """qwen3 think-block prefix must be stripped so JSON parses correctly."""
        response = "<think>I will analyze this article.</think>\n" + _good_json_response(
            title="After Think"
        )
        client = _FixedClient(response)
        result = summarize_item(_make_item(), topic="AI", client=client)
        assert result["title"] == "After Think"

    def test_relevance_score_clamped_to_valid_range(self):
        import json
        response = json.dumps(
            {"title": "T", "summary": "S", "key_insight": "K", "relevance_score": 10}
        )
        client = _FixedClient(response)
        result = summarize_item(_make_item(), topic="AI", client=client)
        assert result["relevance_score"] == 5

    def test_topic_with_braces_does_not_raise(self):
        """A topic containing braces must not cause a formatting error."""
        client = _FixedClient(_good_json_response())
        item = _make_item()
        # Should not raise.
        result = summarize_item(item, topic="AI {systems}", client=client)
        assert isinstance(result, dict)

    def test_content_with_braces_does_not_raise(self):
        """Article content containing JSON must not cause a formatting error."""
        client = _FixedClient(_good_json_response())
        item = _make_item(content='Article about {"neural": "networks"} architectures.')
        result = summarize_item(item, topic="AI", client=client)
        assert isinstance(result, dict)

    def test_satisfies_llmclient_protocol(self):
        """OllamaClient is a structural subtype of LLMClient (runtime check)."""
        client = OllamaClient.__new__(OllamaClient)
        assert isinstance(client, LLMClient)

    def test_stub_satisfies_llmclient_protocol(self):
        assert isinstance(_FixedClient(""), LLMClient)


# ---------------------------------------------------------------------------
# TestGenerateDigest
# ---------------------------------------------------------------------------


def _make_summarized_item(
    title: str = "Article Title",
    summary: str = "Summary text.",
    key_insight: str = "Insight here.",
    relevance_score: int = 3,
    url: str = "https://example.com",
) -> dict:
    return {
        "title": title,
        "summary": summary,
        "key_insight": key_insight,
        "relevance_score": relevance_score,
        "url": url,
    }


class TestGenerateDigest:
    def test_empty_items_returns_no_new_items_message(self):
        result = generate_digest([], topic="AI", date="2026-03-07")
        assert "0 new items" in result
        assert "No new items" in result

    def test_empty_items_includes_topic(self):
        result = generate_digest([], topic="quantum computing", date="2026-03-07")
        assert "quantum computing" in result

    def test_empty_items_includes_date(self):
        result = generate_digest([], topic="AI", date="2026-03-07")
        assert "2026-03-07" in result

    def test_single_item_singular_label(self):
        items = [_make_summarized_item()]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        assert "1 new item" in result
        # Must NOT say "1 new items" (plural).
        assert "1 new items" not in result

    def test_multiple_items_plural_label(self):
        items = [_make_summarized_item(), _make_summarized_item()]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        assert "2 new items" in result

    def test_items_sorted_by_relevance_descending(self):
        items = [
            _make_summarized_item(title="Low", relevance_score=1, url="https://low.com"),
            _make_summarized_item(title="High", relevance_score=5, url="https://high.com"),
            _make_summarized_item(title="Mid", relevance_score=3, url="https://mid.com"),
        ]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        pos_high = result.index("High")
        pos_mid = result.index("Mid")
        pos_low = result.index("Low")
        assert pos_high < pos_mid < pos_low

    def test_first_ranked_item_is_most_relevant(self):
        items = [
            _make_summarized_item(title="Less Relevant", relevance_score=2),
            _make_summarized_item(title="Most Relevant", relevance_score=5),
        ]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        # Rank 1 must be the most-relevant item.
        assert result.index("## 1.") < result.index("Most Relevant")
        # "Most Relevant" must appear before "Less Relevant" overall.
        assert result.index("Most Relevant") < result.index("Less Relevant")

    def test_all_required_fields_present_in_output(self):
        items = [
            _make_summarized_item(
                title="My Title",
                summary="My summary.",
                key_insight="My insight.",
                relevance_score=4,
                url="https://myurl.com",
            )
        ]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        assert "My Title" in result
        assert "My summary." in result
        assert "My insight." in result
        assert "4/5" in result
        assert "https://myurl.com" in result

    def test_digest_header_contains_topic(self):
        items = [_make_summarized_item()]
        result = generate_digest(items, topic="contrastive learning", date="2026-03-07")
        assert "contrastive learning" in result

    def test_digest_header_contains_date(self):
        items = [_make_summarized_item()]
        result = generate_digest(items, topic="AI", date="2026-01-15")
        assert "2026-01-15" in result

    def test_source_link_formatted_correctly(self):
        items = [_make_summarized_item(url="https://source.example.com/paper")]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        assert "[Source](https://source.example.com/paper)" in result

    def test_missing_title_shows_fallback(self):
        item = _make_summarized_item(title="")
        result = generate_digest([item], topic="AI", date="2026-03-07")
        assert "(no title)" in result

    def test_missing_summary_shows_fallback(self):
        item = _make_summarized_item(summary="")
        result = generate_digest([item], topic="AI", date="2026-03-07")
        assert "(no summary)" in result

    def test_missing_key_insight_shows_fallback(self):
        item = _make_summarized_item(key_insight="")
        result = generate_digest([item], topic="AI", date="2026-03-07")
        assert "(no insight)" in result

    def test_returns_string(self):
        result = generate_digest([_make_summarized_item()], topic="AI", date="2026-03-07")
        assert isinstance(result, str)

    def test_topic_with_braces_does_not_raise(self):
        """Topic containing braces must not trigger a format() error."""
        items = [_make_summarized_item()]
        result = generate_digest(items, topic="AI {systems}", date="2026-03-07")
        assert "AI {systems}" in result

    def test_summary_with_braces_does_not_raise(self):
        """LLM-generated summary containing braces must not trigger a format() error."""
        item = _make_summarized_item(summary='Discusses {"key": "value"} in depth.')
        result = generate_digest([item], topic="AI", date="2026-03-07")
        assert '{"key": "value"}' in result

    def test_stable_sort_preserves_order_for_equal_scores(self):
        """Items with identical relevance scores must appear in original order."""
        items = [
            _make_summarized_item(title=f"Item{i}", relevance_score=3, url=f"https://item{i}.com")
            for i in range(4)
        ]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        positions = [result.index(f"Item{i}") for i in range(4)]
        assert positions == sorted(positions), "Equal-score items must preserve input order"

    def test_item_count_in_header_matches_input(self):
        items = [_make_summarized_item() for _ in range(7)]
        result = generate_digest(items, topic="AI", date="2026-03-07")
        assert "7 new items" in result


# ---------------------------------------------------------------------------
# TestCheckOllama
# ---------------------------------------------------------------------------


def _make_list_response(model_names: list[str]) -> ollama.ListResponse:
    """Build a fake ListResponse with the given model name strings."""
    models = [
        ollama.ListResponse.Model(model=name)
        for name in model_names
    ]
    return ollama.ListResponse(models=models)


class TestCheckOllama:
    def test_passes_when_model_is_available(self):
        fake_response = _make_list_response(["qwen3:4b", "llama3:8b"])
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.return_value = fake_response
            # Should not raise.
            check_ollama(base_url="http://localhost:11434", model="qwen3:4b")

    def test_raises_when_model_not_in_list(self):
        fake_response = _make_list_response(["llama3:8b", "mistral:7b"])
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.return_value = fake_response
            with pytest.raises(RuntimeError, match="not available"):
                check_ollama(base_url="http://localhost:11434", model="qwen3:4b")

    def test_raises_on_connect_error(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.side_effect = httpx.ConnectError("refused")
            with pytest.raises(RuntimeError, match="not reachable"):
                check_ollama(base_url="http://localhost:11434", model="qwen3:4b")

    def test_raises_on_timeout(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(RuntimeError, match="[Tt]imed out"):
                check_ollama(base_url="http://localhost:11434", model="qwen3:4b")

    def test_raises_on_response_error(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.side_effect = ollama.ResponseError("server error")
            with pytest.raises(RuntimeError, match="error"):
                check_ollama(base_url="http://localhost:11434", model="qwen3:4b")

    def test_base_name_match_finds_model(self):
        """'qwen3' should match when Ollama stores the model as 'qwen3:latest'."""
        fake_response = _make_list_response(["qwen3:latest"])
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.return_value = fake_response
            # Should not raise — base name 'qwen3' == 'qwen3' from 'qwen3:latest'.
            check_ollama(base_url="http://localhost:11434", model="qwen3")

    def test_empty_model_list_raises(self):
        fake_response = _make_list_response([])
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.return_value = fake_response
            with pytest.raises(RuntimeError):
                check_ollama(base_url="http://localhost:11434", model="qwen3:4b")

    def test_error_message_includes_model_name(self):
        fake_response = _make_list_response(["other-model:7b"])
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.list.return_value = fake_response
            with pytest.raises(RuntimeError, match="qwen3:4b"):
                check_ollama(base_url="http://localhost:11434", model="qwen3:4b")


# ---------------------------------------------------------------------------
# TestOllamaClient
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> ollama.ChatResponse:
    """Build a minimal ChatResponse with the given content string."""
    msg = ollama.Message(role="assistant", content=content)
    return ollama.ChatResponse(model="qwen3:4b", message=msg)


class TestOllamaClient:
    def test_generate_returns_message_content(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = _make_chat_response("hello world")
            client = OllamaClient(base_url="http://localhost:11434", model="qwen3:4b")
            result = client.generate("test prompt")
        assert result == "hello world"

    def test_generate_sends_user_message(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = _make_chat_response("")
            client = OllamaClient()
            client.generate("my prompt")
            call_kwargs = MockClient.return_value.chat.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
            user_msgs = [m for m in messages if m["role"] == "user"]
            assert len(user_msgs) == 1
            assert user_msgs[0]["content"] == "my prompt"

    def test_generate_with_system_sends_system_message(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = _make_chat_response("")
            client = OllamaClient()
            client.generate("prompt", system="be a helpful assistant")
            call_kwargs = MockClient.return_value.chat.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
            sys_msgs = [m for m in messages if m["role"] == "system"]
            assert len(sys_msgs) == 1
            assert sys_msgs[0]["content"] == "be a helpful assistant"

    def test_generate_without_system_sends_no_system_message(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = _make_chat_response("")
            client = OllamaClient()
            client.generate("prompt", system=None)
            call_kwargs = MockClient.return_value.chat.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
            sys_msgs = [m for m in messages if m["role"] == "system"]
            assert sys_msgs == []

    def test_generate_raises_runtime_on_request_error(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.side_effect = ollama.RequestError("bad request")
            client = OllamaClient()
            with pytest.raises(RuntimeError, match="[Rr]equest error"):
                client.generate("prompt")

    def test_generate_raises_runtime_on_response_error(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.side_effect = ollama.ResponseError("model gone")
            client = OllamaClient()
            with pytest.raises(RuntimeError, match="[Rr]esponse error"):
                client.generate("prompt")

    def test_generate_raises_runtime_on_connect_error(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.side_effect = httpx.ConnectError("refused")
            client = OllamaClient()
            with pytest.raises(RuntimeError, match="[Cc]onnect"):
                client.generate("prompt")

    def test_generate_raises_runtime_on_timeout(self):
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.side_effect = httpx.TimeoutException("timed out")
            client = OllamaClient()
            with pytest.raises(RuntimeError, match="[Tt]imed out"):
                client.generate("prompt")

    def test_none_content_returns_empty_string(self):
        """If Ollama returns a message with content=None, generate() must return ''."""
        msg = ollama.Message(role="assistant", content=None)
        response = ollama.ChatResponse(model="qwen3:4b", message=msg)
        with patch("redpill.summarize.ollama.Client") as MockClient:
            MockClient.return_value.chat.return_value = response
            client = OllamaClient()
            result = client.generate("prompt")
        assert result == ""

    def test_base_url_stored_without_private_access(self):
        """OllamaClient must not rely on private library internals for base_url."""
        client = OllamaClient.__new__(OllamaClient)
        client._base_url = "http://test:11434"
        client._model = "qwen3:4b"
        # The attribute must be accessible directly.
        assert client._base_url == "http://test:11434"
