"""
llm_utils.py — Shared helpers for parsing LLM responses.

Public API:
    strip_think_blocks(text: str) -> str
        Remove <think>...</think> sections emitted by reasoning models (e.g. qwen3).

    extract_json(raw: str) -> dict | list | None
        Parse JSON from a raw LLM response. Returns None on complete failure.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Matches <think>...</think> blocks including newlines (non-greedy).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> sections that reasoning models emit before
    their actual response.  Multiple blocks are all removed.
    """
    return _THINK_BLOCK_RE.sub("", text)


def extract_json(raw: str) -> dict | list | None:
    """Parse JSON from a raw LLM response string.

    Strategy:
    1. Strip any <think>...</think> blocks.
    2. Strip leading/trailing whitespace.
    3. Attempt direct json.loads().
    4. If that fails, locate the first ``[`` or ``{`` and last matching ``]``
       or ``}`` and try again — handles models that wrap JSON in preamble or
       markdown fences.

    Returns the parsed value (dict or list), or ``None`` on complete failure.
    Callers must check the type if they need a specific JSON structure.
    """
    text = strip_think_blocks(raw).strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: locate outermost JSON container
    # Try object first, then array
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start: end + 1])
            except json.JSONDecodeError:
                pass

    # Attempt 3: repair missing commas between adjacent JSON objects/arrays
    # LLMs sometimes emit `} {` or `] [` without the separating comma.
    repaired = re.sub(r'}\s*\{', '}, {', text)
    repaired = re.sub(r']\s*\[', '], [', repaired)
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = repaired.find(open_char)
        end = repaired.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(repaired[start: end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning("extract_json: failed to parse JSON from LLM response: %r", text[:200])
    return None
