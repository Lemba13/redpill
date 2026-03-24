# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**redpill** is an AI-powered autonomous web research digest agent. It crawls the web daily for a given topic, deduplicates content against previous runs using semantic similarity, summarizes findings with a local LLM (Ollama), and delivers a markdown digest or email.

## Setup

```bash
pip install -e ".[dev]"     # Install with dev dependencies
cp config.example.yaml config.yaml
cp .env.example .env        # Add TAVILY_API_KEY
```

Requires Ollama running locally (`ollama serve`) with a model pulled (e.g. `ollama pull llama3.1:8b`).

## Commands

```bash
# Run
redpill run                          # Full pipeline
redpill run --dry-run                # Search + summarize to stdout, no state changes
redpill run --config FILE            # Custom config

# Inspect state
redpill history --last N             # Last N digests
redpill stats                        # DB statistics
redpill queries --last DAYS          # Query performance history
redpill terms --top N                # Top N terms all-time
redpill plan                         # Preview next run's queries

# Tests
pytest                               # Unit tests
python smoke_test.py                 # Full integration test (no real API calls)
python smoke_test.py -v              # Verbose
```

## Architecture

The pipeline runs sequentially in `redpill/main.py:run_pipeline()`:

1. Load YAML config → validate required fields
2. Initialize SQLite DB (3 tables: `seen_items`, `extracted_terms`, `query_log`)
3. Health-check local Ollama
4. **Query planning** — LLM generates queries from term history (or use static `search_queries` from config)
5. **Web search** — Tavily API, deduplicated by URL
6. **Content extraction** — parallel (5 workers, 10s timeout) via trafilatura
7. **Deduplication** — URL exact-match, then semantic cosine similarity on sentence-transformer embeddings
8. **Summarization** — Ollama LLM → JSON (title, summary, key_insight, relevance_score)
9. **Term extraction** — LLM extracts domain terms from high-relevance articles
10. **Digest generation** — Markdown sorted by relevance descending
11. **Delivery** — Write markdown file or send SMTP email
12. **State persistence** — URLs, embeddings, terms, query logs → SQLite

Dry-run mode skips steps 9–12.

## Key Modules

| Module | Role |
|---|---|
| `main.py` | CLI (Click) + pipeline orchestrator |
| `search.py` | Tavily search with 3× exponential backoff retry |
| `extract.py` | Parallel content extraction (ThreadPoolExecutor) |
| `dedup.py` | Two-pass dedup: URL hash then cosine similarity |
| `summarize.py` | Ollama LLM client, health checks, digest generation |
| `deliver.py` | Markdown file write or SMTP multipart email |
| `state.py` | All SQLite CRUD, embedding serialization/deserialization |
| `query_planner.py` | LLM-driven query generation with deterministic fallback |
| `term_extractor.py` | Per-article LLM term extraction, filters by relevance |
| `llm_utils.py` | Robust JSON parsing from LLM output (multi-strategy + strip `<think>` blocks) |

## Design Patterns

- **LLMClient Protocol** — abstract interface; swapping Ollama for another provider requires only a new class implementing the protocol
- **Embedding serialization** — self-describing binary format (dtype + shape header + raw bytes) in `state.py`
- **Two-pass dedup** — cheap URL match first, expensive embedding similarity second
- **Error isolation** — modules return results with success flags rather than raising; individual failures don't abort the run
- **Graceful fallbacks** — LLM failures fall back to deterministic query expansion or partial results

## Configuration

See `config.example.yaml` for all options. Key fields:

```yaml
topic: "contrastive learning"
search_queries: [...]               # Static; ignored if query_planning.enabled = true
max_results_per_query: 10
dedup_similarity_threshold: 0.85
db_path: "data/redpill.db"
delivery_method: "markdown"         # or "email"
ollama_config:
  base_url: "http://localhost:11434"
  model: "llama3.1:8b"
embedding_model: "all-MiniLM-L6-v2"
query_planning:
  enabled: false
  max_queries: 5
```

Environment variables (`.env`): `TAVILY_API_KEY` (required), `SMTP_PASSWORD` (email only).

## Testing Notes

- Unit tests use `pytest-mock` and in-memory SQLite (`:memory:`)
- `smoke_test.py` exercises the full pipeline end-to-end without real API calls
- Each module has its own test file in `tests/`
