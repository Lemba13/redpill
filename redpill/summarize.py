"""
summarize.py — LLM-powered summarization via an LLMClient abstraction.

LLMClient interface:
    generate(prompt: str, system: str | None) -> str

Implementations (to be added):
    OllamaClient   — local Ollama instance
    AnthropicClient — Anthropic API (future)
    OpenAIClient    — OpenAI API (future)

Public API:
    summarize_item(content: str, topic: str, client: LLMClient) -> dict
        Returns: {title, summary, key_insight, relevance_score}
        LLM is prompted to respond in JSON.

    generate_digest(items: list[dict], topic: str, date: str, client: LLMClient) -> str
        Produces a formatted markdown digest sorted by relevance_score desc.
"""
