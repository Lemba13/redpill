"""
tests/test_planner_llm_client.py — Unit tests for PlannerLLMClient in
redpill.summarize.

All Ollama calls are mocked.  No real Ollama instance is required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from redpill.summarize import PlannerLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str, thinking: str | None = None) -> MagicMock:
    """Build a mock ollama.ChatResponse with .message.content and .message.thinking."""
    msg = MagicMock()
    msg.content = content
    msg.thinking = thinking
    response = MagicMock()
    response.message = msg
    return response


def _make_client(content: str, thinking: str | None = None) -> PlannerLLMClient:
    """Return a PlannerLLMClient whose Ollama client is mocked."""
    client = PlannerLLMClient(
        base_url="http://localhost:11434",
        model="qwen3.5:4b",
        think=True,
        timeout=120,
    )
    client._client = MagicMock()
    client._client.chat.return_value = _make_response(content, thinking)
    return client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPlannerLLMClientConstruction:
    def test_default_model(self):
        client = PlannerLLMClient()
        assert client._model == "qwen3.5:4b"

    def test_custom_model(self):
        client = PlannerLLMClient(model="qwen3:4b")
        assert client._model == "qwen3:4b"

    def test_think_enabled_by_default(self):
        client = PlannerLLMClient()
        assert client._think is True

    def test_think_can_be_disabled(self):
        client = PlannerLLMClient(think=False)
        assert client._think is False

    def test_last_thinking_initially_none(self):
        client = PlannerLLMClient()
        assert client.last_thinking is None

    def test_timeout_stored(self):
        client = PlannerLLMClient(timeout=60)
        assert client._timeout == 60

    def test_num_ctx_stored(self):
        client = PlannerLLMClient(num_ctx=4096)
        assert client._num_ctx == 4096


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------

class TestPlannerLLMClientGenerate:
    def test_returns_string(self):
        client = _make_client('{"result": "ok"}')
        result = client.generate("test prompt")
        assert isinstance(result, str)

    def test_returns_content(self):
        client = _make_client('{"result": "ok"}')
        result = client.generate("test prompt")
        assert '{"result": "ok"}' in result

    def test_captures_thinking_from_message_attribute(self):
        client = _make_client('{"result": "ok"}', thinking="My reasoning.")
        client.generate("test prompt")
        assert client.last_thinking == "My reasoning."

    def test_last_thinking_none_when_model_returns_none(self):
        client = _make_client('{"result": "ok"}', thinking=None)
        client.generate("test prompt")
        assert client.last_thinking is None

    def test_last_thinking_reset_before_each_call(self):
        client = _make_client('{"result": "ok"}', thinking="first")
        client.generate("prompt 1")
        assert client.last_thinking == "first"

        # Second call with no thinking
        client._client.chat.return_value = _make_response('{}', thinking=None)
        client.generate("prompt 2")
        assert client.last_thinking is None

    def test_think_param_passed_to_ollama_when_enabled(self):
        client = _make_client('{}')
        client.generate("prompt")
        call_kwargs = client._client.chat.call_args[1]
        assert call_kwargs.get("think") is True

    def test_think_param_not_passed_when_disabled(self):
        client = PlannerLLMClient(think=False)
        client._client = MagicMock()
        client._client.chat.return_value = _make_response("{}")
        client.generate("prompt")
        call_kwargs = client._client.chat.call_args[1]
        assert "think" not in call_kwargs

    def test_format_json_passed(self):
        client = _make_client('{}')
        client.generate("prompt")
        call_kwargs = client._client.chat.call_args[1]
        assert call_kwargs.get("format") == "json"

    def test_num_ctx_in_options(self):
        client = PlannerLLMClient(num_ctx=4096)
        client._client = MagicMock()
        client._client.chat.return_value = _make_response("{}")
        client.generate("prompt")
        call_kwargs = client._client.chat.call_args[1]
        assert call_kwargs["options"]["num_ctx"] == 4096

    def test_system_prompt_included_when_provided(self):
        client = _make_client('{}')
        client.generate("user prompt", system="system message")
        call_kwargs = client._client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert any(m["role"] == "system" and m["content"] == "system message" for m in messages)

    def test_no_system_prompt_when_not_provided(self):
        client = _make_client('{}')
        client.generate("user prompt")
        call_kwargs = client._client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)

    def test_user_message_always_present(self):
        client = _make_client('{}')
        client.generate("the prompt")
        call_kwargs = client._client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert any(m["role"] == "user" and m["content"] == "the prompt" for m in messages)


# ---------------------------------------------------------------------------
# generate() — thinking extracted from content block as fallback
# ---------------------------------------------------------------------------

class TestPlannerLLMClientThinkingFallback:
    def test_extracts_thinking_from_content_when_attribute_absent(self):
        """Some Ollama versions return thinking inside the content block
        rather than as a separate attribute."""
        content_with_think = '<think>my reasoning</think>{"result": "ok"}'

        client = PlannerLLMClient()
        client._client = MagicMock()
        # No .thinking attribute on message — must have thinking as None
        response = _make_response(content_with_think, thinking=None)
        # Remove the thinking attribute entirely to simulate older Ollama
        del response.message.thinking
        client._client.chat.return_value = response

        result = client.generate("prompt")
        # The think block should be stripped from the returned content
        assert "<think>" not in result
        assert "my reasoning" not in result
        # But the JSON answer remains
        assert '{"result": "ok"}' in result
        # And the trace was captured
        assert client.last_thinking == "my reasoning"


# ---------------------------------------------------------------------------
# generate() — error handling
# ---------------------------------------------------------------------------

class TestPlannerLLMClientErrors:
    def test_request_error_raises_runtime_error(self):
        import ollama
        client = PlannerLLMClient()
        client._client = MagicMock()
        client._client.chat.side_effect = ollama.RequestError("bad request")
        with pytest.raises(RuntimeError, match="request error"):
            client.generate("prompt")

    def test_response_error_raises_runtime_error(self):
        import ollama
        client = PlannerLLMClient()
        client._client = MagicMock()
        client._client.chat.side_effect = ollama.ResponseError("model error")
        with pytest.raises(RuntimeError, match="response error"):
            client.generate("prompt")

    def test_connect_error_raises_runtime_error(self):
        import httpx
        client = PlannerLLMClient()
        client._client = MagicMock()
        client._client.chat.side_effect = httpx.ConnectError("refused")
        with pytest.raises(RuntimeError, match="cannot connect"):
            client.generate("prompt")

    def test_timeout_raises_runtime_error(self):
        import httpx
        client = PlannerLLMClient()
        client._client = MagicMock()
        client._client.chat.side_effect = httpx.TimeoutException("timed out")
        with pytest.raises(RuntimeError, match="timed out"):
            client.generate("prompt")

    def test_last_thinking_reset_to_none_before_failed_call(self):
        import ollama
        client = PlannerLLMClient()
        client._client = MagicMock()
        client.last_thinking = "previous thinking"
        client._client.chat.side_effect = ollama.RequestError("error")
        with pytest.raises(RuntimeError):
            client.generate("prompt")
        # last_thinking must have been reset to None before the (failed) call
        assert client.last_thinking is None


# ---------------------------------------------------------------------------
# PlannerLLMClient is an LLMClient
# ---------------------------------------------------------------------------

class TestPlannerLLMClientProtocol:
    def test_satisfies_llm_client_protocol(self):
        from redpill.summarize import LLMClient
        client = PlannerLLMClient()
        # Protocol check
        assert isinstance(client, LLMClient)
