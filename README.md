# RedPill

An autonomous agent that crawls the web daily for a given topic, deduplicates against previously seen content, summarizes new findings with a local LLM, and delivers a clean digest.

## How it works

1. **Search** — Queries Tavily with multiple query variations for broader coverage
2. **Extract** — Fetches each URL and strips it down to main article text
3. **Deduplicate** — Filters out previously seen articles by URL and semantic similarity
4. **Summarize** — Sends new articles to a local Ollama LLM for structured summaries
5. **Deliver** — Writes a markdown digest or sends it via email
6. **Persist** — Saves seen URLs and embeddings to SQLite so tomorrow's run stays fresh

## Setup

```bash
git clone <repo>
cd redpill
pip install -e ".[dev]"
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `config.yaml` with your topic and queries. Add your `TAVILY_API_KEY` to `.env`.

## Requirements

- Python 3.11+
- [Tavily API key](https://tavily.com) (free tier: 1000 searches/month)
- [Ollama](https://ollama.com) running locally with a model pulled (e.g. `ollama pull llama3.1`)

## Usage

```bash
redpill run              # full pipeline
redpill run --dry-run    # skip delivery and state update
redpill history --last 7 # show last 7 digests
redpill stats            # summary stats
```

## Scheduling

Add to cron to run daily at 7 AM:

```
0 7 * * * cd /path/to/redpill && redpill run
```

Or use the included GitHub Actions workflow for cloud scheduling.

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Search (search.py) | ✅ Done |
| 2 | Content extraction (extract.py) | ✅ Done |
| 3 | State management (state.py) | 🔲 Pending |
| 4 | Deduplication (dedup.py) | 🔲 Pending |
| 5 | Summarization (summarize.py) | 🔲 Pending |
| 6 | Delivery (deliver.py) | 🔲 Pending |
| 7 | Orchestrator + CLI (main.py) | 🔲 Pending |
