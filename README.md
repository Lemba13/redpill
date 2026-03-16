# RedPill

An autonomous agent that crawls the web daily for a given topic, deduplicates against previously seen content, summarizes new findings with a local LLM, and delivers a clean digest. Each run extracts domain-specific terms and uses them to plan smarter queries for the next run.

## How it works

1. **Plan** — Uses term history from prior runs to generate targeted search queries via LLM (or falls back to deterministic term expansion). The base topic is always included.
2. **Search** — Queries Tavily with the planned queries for broad coverage
3. **Extract** — Fetches each URL and strips it down to main article text
4. **Deduplicate** — Filters out previously seen articles by URL and semantic similarity
5. **Summarize** — Sends new articles to a local Ollama LLM for structured summaries
6. **Extract terms** — Pulls domain-specific terms (techniques, authors, datasets, frameworks) from high-relevance articles to feed the next planning cycle
7. **Deliver** — Writes a markdown digest or sends it via email
8. **Persist** — Saves seen URLs, embeddings, extracted terms, and query stats to SQLite

## Requirements

- Python 3.11+
- [Tavily API key](https://tavily.com) (free tier: 1000 searches/month)
- [Ollama](https://ollama.com) running locally with a model pulled (e.g. `ollama pull llama3.1:8b`)

## Setup

```bash
git clone <repo>
cd redpill
pip install -e ".[dev]"
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `config.yaml` with your topic and delivery settings. Fill in `.env` with your secrets — `.env.example` documents all available variables including `TAVILY_API_KEY`, `SMTP_PASSWORD`, and `HUGGINGFACE_HUB_VERBOSITY` (pre-set to silence a cosmetic HuggingFace auth warning for the public embedding model).

## Configuration

| Key | Description |
|-----|-------------|
| `topic` | What you want to track (e.g. `"contrastive learning"`) |
| `search_queries` | Static query list — used when `query_planning.enabled: false` |
| `max_results_per_query` | Results per query, max 20 (Tavily limit) |
| `dedup_similarity_threshold` | Semantic similarity cutoff (0.85 recommended) |
| `delivery_method` | `"markdown"` saves a file, `"email"` sends to your inbox |
| `ollama_config.model` | Ollama model to use (e.g. `llama3.1:8b`, `qwen3:4b`) |
| `email_config` | SMTP settings — only needed if `delivery_method: "email"` |
| `query_planning.enabled` | Enable LLM-driven query planning (default: `false`) |
| `query_planning.max_queries` | Max queries per run including base topic (default: `5`) |

### Query planning

When enabled, each run reads recently extracted terms from the database and asks the LLM to propose targeted queries. On the first run (empty term history) it falls back to deterministic term expansion. With `enabled: false` (the default), `search_queries` from config is used as-is.

```yaml
query_planning:
  enabled: true
  max_queries: 5
```

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
# Pipeline
redpill run                    # full pipeline
redpill run --dry-run          # search → summarize → print to stdout, no state changes
redpill run --config my.yaml   # use a custom config file

# Inspect digests
redpill history --last 7       # show last 7 digests

# Database views
redpill stats                  # total seen, avg per day, top sources
redpill queries                # query performance history (last 14 days)
redpill queries --last 30      # extend lookback window
redpill terms                  # top 20 extracted terms (all time)
redpill terms --top 50         # top 50 terms
redpill terms --recent         # terms seen in the last 30 days
redpill terms --recent 7       # terms seen in the last 7 days

# Query planner (dry-run)
redpill plan                   # show what queries would be planned next run
redpill plan --max-queries 8   # override query count
```

## Smoke test

Verifies the full v2 feedback loop without any real API calls:

```bash
python smoke_test.py      # compact output
python smoke_test.py -v   # verbose (show every check)
```

## Scheduling

Add to cron to run daily at 7 AM:

```
0 7 * * * cd /path/to/redpill && redpill run
```
