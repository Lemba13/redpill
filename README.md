# RedPill

An autonomous agent that crawls the web daily for a given topic, deduplicates against previously seen content, summarizes new findings with a local LLM, and delivers a clean digest.

## How it works

1. **Search** — Queries Tavily with multiple query variations for broader coverage
2. **Extract** — Fetches each URL and strips it down to main article text
3. **Deduplicate** — Filters out previously seen articles by URL and semantic similarity
4. **Summarize** — Sends new articles to a local Ollama LLM for structured summaries
5. **Deliver** — Writes a markdown digest or sends it via email
6. **Persist** — Saves seen URLs and embeddings to SQLite so tomorrow's run stays fresh

## Requirements

- Python 3.11+
- [Tavily API key](https://tavily.com) (free tier: 1000 searches/month)
- [Ollama](https://ollama.com) running locally with a model pulled (e.g. `ollama pull qwen3:4b`)

## Setup

```bash
git clone <repo>
cd redpill
pip install -e ".[dev]"
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `config.yaml` with your topic, queries, and delivery settings. Add your `TAVILY_API_KEY` (and optionally `SMTP_PASSWORD`) to `.env`.

## Configuration

| Key | Description |
|-----|-------------|
| `topic` | What you want to track (e.g. `"contrastive learning"`) |
| `search_queries` | 2-3 query variations for broader search coverage |
| `max_results_per_query` | Results per query, max 20 (Tavily limit) |
| `dedup_similarity_threshold` | Semantic similarity cutoff (0.85 recommended) |
| `delivery_method` | `"markdown"` saves a file, `"email"` sends to your inbox |
| `ollama_config.model` | Ollama model to use (e.g. `qwen3:4b`, `mistral`) |
| `email_config` | SMTP settings — only needed if `delivery_method: "email"` |

### Email delivery (Gmail example)

```yaml
delivery_method: "email"
email_config:
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  sender: "you@gmail.com"
  recipient: "you@gmail.com"
```

Add your Gmail App Password to `.env`:
```
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
```

Generate an App Password at: Google Account → Security → 2-Step Verification → App passwords.

## Usage

```bash
redpill run                   # full pipeline
redpill run --dry-run         # search → extract → dedup → summarize, print to stdout only
redpill run --config my.yaml  # use a custom config file
redpill history --last 7      # show last 7 digests
redpill stats                 # total seen, avg per day, top sources
```

## Smoke tests

```bash
python scripts/smoke_test_search.py  # test Tavily API connection
python scripts/smoke_test_email.py   # send a test email
```

## Scheduling

Add to cron to run daily at 7 AM:

```
0 7 * * * cd /path/to/redpill && redpill run
```
