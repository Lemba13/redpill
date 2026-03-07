"""
summarize.py — LLM-powered summarization via an LLMClient abstraction.

LLMClient interface (Protocol):
    generate(prompt: str, system: str | None = None) -> str

Concrete implementations:
    OllamaClient(base_url, model)  — local Ollama instance via the ollama library

Public API:
    check_ollama(base_url: str, model: str) -> None
        Raises RuntimeError if Ollama is unreachable or the model is not pulled.

    summarize_item(item: dict, topic: str, client: LLMClient) -> dict
        Returns: {title, summary, key_insight, relevance_score, url}
        Prompts the LLM for JSON; falls back gracefully on parse failure.

    generate_digest(items: list[dict], topic: str, date: str) -> str
        Produces a markdown digest sorted by relevance_score descending.
        Each item must already have been through summarize_item() so it carries
        the summarized fields alongside the original 'url' key.
"""

import json
import logging
import re
from typing import Protocol, runtime_checkable

import httpx
import ollama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_BASE_URL = "http://localhost:11434"

# Matches <think>...</think> blocks including newlines (non-greedy).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a focused research assistant. Your job is to read articles and "
    "extract the most important signal for a researcher tracking a specific "
    "topic. You are precise, concise, and never hallucinate. You always "
    "respond with valid JSON and nothing else — no markdown fences, no "
    "preamble, no commentary."
)

def _build_summarize_prompt(topic: str, content: str) -> str:
    """Build the summarization prompt by explicit concatenation.

    We deliberately avoid str.format() here because both *topic* and *content*
    come from untrusted external sources and may contain literal brace
    characters (e.g. code snippets, JSON fragments). A stray '{' in the
    article text would cause str.format() to raise KeyError or ValueError.
    """
    return (
        "Topic: " + topic + "\n\n"
        "Article content:\n" + content + "\n\n"
        "Respond with a JSON object containing exactly these four keys:\n"
        '- "title": a concise, descriptive title for the article (string)\n'
        '- "summary": a 2-3 sentence summary of the article (string)\n'
        '- "key_insight": one sentence explaining why this matters for someone '
        'tracking "' + topic + '" (string)\n'
        '- "relevance_score": an integer from 1 to 5 indicating how relevant '
        'this article is to "' + topic + '" '
        "(1 = barely related, 5 = highly relevant)\n\n"
        "Return only the JSON object. No markdown, no explanation."
    )

# ---------------------------------------------------------------------------
# LLMClient protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface that all LLM backend implementations must satisfy.

    Having a single generate() method keeps the abstraction honest — callers
    only need a string in, string out. System prompt is optional because not
    every provider or use-case needs one.
    """

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send *prompt* to the model and return the raw response string.

        Parameters
        ----------
        prompt:
            The user-facing prompt text.
        system:
            An optional system prompt that sets the model's role / behaviour.
            Pass None to omit it entirely.

        Raises
        ------
        RuntimeError
            If the underlying provider is unreachable or returns an error that
            cannot be handled (e.g. model not loaded, connection refused).
        """
        ...


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------


class OllamaClient:
    """LLMClient implementation backed by a local Ollama instance.

    Parameters
    ----------
    base_url:
        URL of the Ollama HTTP API (default: http://localhost:11434).
    model:
        Name of the pulled Ollama model to use (default: qwen3:4b).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client = ollama.Client(host=base_url)

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Call ollama.chat() and return the assistant message content.

        Both RequestError (bad request) and ResponseError (Ollama API error)
        are caught and re-raised as RuntimeError so callers have a single
        exception type to handle. Connection failures surface as RuntimeError
        too, wrapping the underlying httpx.ConnectError.

        Parameters
        ----------
        prompt:
            User message to send to the model.
        system:
            Optional system prompt. When provided, it is sent as the first
            message with role="system".
        """
        messages: list[dict] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "OllamaClient.generate: model=%r, system=%s, prompt_len=%d",
            self._model,
            "yes" if system else "no",
            len(prompt),
        )

        try:
            response: ollama.ChatResponse = self._client.chat(
                model=self._model,
                messages=messages,
            )
        except ollama.RequestError as exc:
            raise RuntimeError(f"Ollama request error: {exc}") from exc
        except ollama.ResponseError as exc:
            raise RuntimeError(f"Ollama response error: {exc}") from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Ollama request timed out: {exc}") from exc

        content: str = response.message.content or ""
        logger.debug("OllamaClient.generate: received %d chars", len(content))
        return content


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_ollama(base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL) -> None:
    """Verify Ollama is running and *model* is available.

    This should be called once at pipeline startup, before any summarization
    begins, so failures are caught early with a clear error message rather than
    buried inside a batch loop.

    Parameters
    ----------
    base_url:
        URL of the Ollama HTTP API.
    model:
        Name of the model that must be present in the pulled model list.

    Raises
    ------
    RuntimeError
        If Ollama is unreachable, or if *model* is not in the pulled model list.
    """
    client = ollama.Client(host=base_url)
    logger.debug("check_ollama: contacting Ollama at %s", base_url)

    try:
        list_response: ollama.ListResponse = client.list()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Timed out connecting to Ollama at {base_url}: {exc}"
        ) from exc
    except ollama.ResponseError as exc:
        raise RuntimeError(f"Ollama returned an error during health check: {exc}") from exc

    available_models = [m.model for m in list_response.models if m.model]

    # Ollama model names can include a tag (e.g. "qwen3:4b"). We check for
    # an exact match first, then fall back to base-name matching so that
    # "qwen3:4b" is found even when Ollama stores it as "qwen3:4b:latest".
    model_found = model in available_models
    if not model_found:
        # Normalise: strip trailing ":latest" noise for comparison.
        def _base(name: str) -> str:
            return name.split(":")[0]

        model_found = any(_base(m) == _base(model) for m in available_models)

    if not model_found:
        raise RuntimeError(
            f"Model {model!r} is not available in Ollama. "
            f"Pull it first: `ollama pull {model}`. "
            f"Available models: {available_models}"
        )

    logger.info("check_ollama: OK — model %r is available", model)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> sections that qwen3 models emit before their
    actual response.

    The regex is non-greedy and DOTALL so it handles multi-line think blocks
    correctly. Multiple blocks are all removed.
    """
    return _THINK_BLOCK_RE.sub("", text)


def _extract_json(raw: str) -> dict:
    """Parse JSON from a raw LLM response string.

    Strategy:
    1. Strip any <think>...</think> blocks.
    2. Strip leading/trailing whitespace.
    3. Attempt direct json.loads().
    4. If that fails, look for the first '{' and last '}' and try again —
       handles models that prepend/append text outside the JSON object.

    Returns the parsed dict, or an empty dict on complete failure (so callers
    can decide on a fallback without catching exceptions).
    """
    text = _strip_think_blocks(raw).strip()

    # Attempt 1: clean JSON — also guard against the model returning a JSON
    # array or scalar instead of an object.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Attempt 2: locate outermost braces — handles models that prepend/append
    # stray text around an otherwise valid JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("_extract_json: failed to parse JSON from LLM response: %r", text[:200])
    return {}


_FALLBACK_SUMMARY: dict = {
    "title": "",
    "summary": "",
    "key_insight": "",
    "relevance_score": 1,
}


def _validate_summary(data: dict) -> dict:
    """Ensure the parsed dict has all required keys with correct types.

    Missing or wrongly-typed fields are replaced with safe defaults. This
    means summarize_item() always returns a dict with a consistent shape,
    regardless of what the LLM actually produced.
    """
    title = data.get("title", "")
    if not isinstance(title, str):
        title = str(title)

    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    key_insight = data.get("key_insight", "")
    if not isinstance(key_insight, str):
        key_insight = str(key_insight)

    raw_score = data.get("relevance_score", 1)
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 1
    # Clamp to the valid 1–5 range.
    score = max(1, min(5, score))

    return {
        "title": title,
        "summary": summary,
        "key_insight": key_insight,
        "relevance_score": score,
    }


# ---------------------------------------------------------------------------
# Public summarization API
# ---------------------------------------------------------------------------


def summarize_item(item: dict, topic: str, client: LLMClient) -> dict:
    """Summarize one article item using the provided LLM client.

    The input *item* dict must have at minimum:
        url     (str)         — preserved in the output for digest rendering
        content (str | None)  — full extracted article text; preferred
        snippet (str)         — short search-result snippet; fallback
        title   (str)         — existing title from extraction (may be empty)

    The LLM is asked to return JSON with: title, summary, key_insight,
    relevance_score. On any parse or LLM failure the function returns a
    safe fallback dict rather than raising, so one bad article does not
    abort the full pipeline.

    Parameters
    ----------
    item:
        Candidate dict as produced by the extract / dedup pipeline.
    topic:
        The research topic (used in the prompt so the LLM can score relevance).
    client:
        Any LLMClient implementation.

    Returns
    -------
    Dict with keys: title, summary, key_insight, relevance_score, url.
    """
    url: str = item.get("url", "")
    content: str | None = item.get("content")
    snippet: str = item.get("snippet", "")

    # Use full content when available; fall back to the search snippet.
    text = content if content and content.strip() else snippet

    if not text.strip():
        logger.warning(
            "summarize_item: no usable text for url=%r — returning fallback", url
        )
        return {**_FALLBACK_SUMMARY, "url": url}

    prompt = _build_summarize_prompt(topic=topic, content=text)

    logger.debug("summarize_item: sending %d chars to LLM for url=%r", len(text), url)

    try:
        raw = client.generate(prompt=prompt, system=_SYSTEM_PROMPT)
    except RuntimeError as exc:
        logger.error("summarize_item: LLM call failed for url=%r: %s", url, exc)
        return {**_FALLBACK_SUMMARY, "url": url}

    parsed = _extract_json(raw)
    if not parsed:
        logger.warning(
            "summarize_item: JSON parse failed for url=%r — returning fallback", url
        )
        return {**_FALLBACK_SUMMARY, "url": url}

    validated = _validate_summary(parsed)
    result = {**validated, "url": url}

    logger.info(
        "summarize_item: url=%r title=%r relevance=%d/5",
        url,
        result["title"][:60] if result["title"] else "(empty)",
        result["relevance_score"],
    )
    return result


# ---------------------------------------------------------------------------
# Digest generation
# ---------------------------------------------------------------------------


def _render_digest_header(topic: str, date: str, n: int) -> str:
    """Render the digest header block.

    Built by concatenation rather than str.format() because *topic* is
    user-supplied and may contain brace characters.
    """
    plural = "" if n == 1 else "s"
    return (
        "# RedPill Digest — " + topic + "\n"
        "**" + date + "** | " + str(n) + " new item" + plural + "\n"
        "\n"
        "---\n"
    )


def _render_digest_item(
    rank: int,
    title: str,
    summary: str,
    key_insight: str,
    relevance_score: int,
    url: str,
) -> str:
    """Render one digest item block.

    Built by concatenation rather than str.format() because title, summary,
    key_insight, and url all originate from LLM output or scraped content
    and may contain brace characters.
    """
    return (
        "## " + str(rank) + ". " + title + "\n"
        + summary + "\n"
        "\n"
        "**Key insight:** " + key_insight + "\n"
        "**Relevance:** " + str(relevance_score) + "/5 | [Source](" + url + ")\n"
        "\n"
        "---\n"
    )


def generate_digest(items: list[dict], topic: str, date: str) -> str:
    """Produce a formatted markdown digest from a list of summarized items.

    Items are sorted by relevance_score descending (highest first). Items
    with the same score retain their original relative order (stable sort).

    Parameters
    ----------
    items:
        List of dicts as returned by summarize_item(). Each must have:
        title, summary, key_insight, relevance_score, url.
    topic:
        The research topic, used in the digest header.
    date:
        ISO date string (e.g. "2026-03-07") for the digest header.

    Returns
    -------
    A multi-line markdown string ready to be written to a file or emailed.
    If *items* is empty, returns a minimal "nothing new" digest.
    """
    if not items:
        logger.info("generate_digest: no items — producing empty digest")
        return (
            "# RedPill Digest — " + topic + "\n"
            "**" + date + "** | 0 new items\n"
            "\n"
            "---\n"
            "\n"
            "_No new items found for today._\n"
        )

    sorted_items = sorted(items, key=lambda x: x.get("relevance_score", 1), reverse=True)

    n = len(sorted_items)
    parts: list[str] = [_render_digest_header(topic=topic, date=date, n=n)]

    for rank, item in enumerate(sorted_items, start=1):
        parts.append(
            _render_digest_item(
                rank=rank,
                title=item.get("title") or "(no title)",
                summary=item.get("summary") or "(no summary)",
                key_insight=item.get("key_insight") or "(no insight)",
                relevance_score=int(item.get("relevance_score", 1)),
                url=item.get("url", ""),
            )
        )

    digest = "\n".join(parts)
    logger.info("generate_digest: produced digest with %d item(s) for topic=%r", n, topic)
    return digest
