# RedPill

An autonomous agent that crawls the web daily for a given topic, deduplicates against previously seen content, summarizes new findings with a local LLM, and delivers a clean digest. Each run extracts domain-specific terms and uses them to plan smarter queries for the next run. A lightweight feedback service lets you vote on digest items, and those signals feed back into the planner to prioritize what you actually find useful.

![Architecture](docs/architecture.png)

## How it works

1. **Plan** — Uses term history and user feedback signals from prior runs to generate targeted search queries via LLM (or falls back to deterministic term expansion). The base topic is always included.
2. **Search** — Queries Tavily and/or Serper (Google) with the planned queries; fan-out mode hits both providers and merges results for broader coverage
3. **Extract** — Fetches each URL and strips it down to main article text
4. **Deduplicate** — Filters out previously seen articles by URL and semantic similarity
5. **Summarize** — Sends new articles to a local Ollama LLM for structured summaries
6. **Extract terms** — Pulls domain-specific terms (techniques, authors, datasets, frameworks) from high-relevance articles to feed the next planning cycle
7. **Deliver** — Writes a markdown digest or sends it via email (with a feedback link when the feedback service is enabled)
8. **Persist** — Saves seen URLs, embeddings, extracted terms, and query stats to SQLite
9. **Feedback** — Reads votes from `feedback.db` and incorporates preference signals (dimension approval, source preference, engagement rate) into the next planning cycle

## Requirements

- Python 3.11+
- [Tavily API key](https://tavily.com) (free tier: 1000 searches/month) — required unless `search_provider: "serper"`
- [Serper API key](https://serper.dev) (free tier: 2500 queries one-time) — required when `search_provider` is `"serper"` or `"both"`
- [Ollama](https://ollama.com) running locally with models pulled:
  - `ollama pull llama3.1:8b` (or your preferred summarization model)
  - `ollama pull deepseek-r1:7b` (reasoning model for query planning — required if `query_planning.enabled: true`)

## Setup

```bash
git clone <repo>
cd redpill
pip install -e ".[dev]"
cp config.example.yaml config.yaml
cp .env.example .env
```

Edit `config.yaml` with your topic and delivery settings. Fill in `.env` with your secrets — `.env.example` documents all available variables including `TAVILY_API_KEY`, `SERPER_API_KEY`, `SMTP_PASSWORD`, and `HUGGINGFACE_HUB_VERBOSITY` (pre-set to silence a cosmetic HuggingFace auth warning for the public embedding model).

To use the feedback service, install its optional dependencies:

```bash
pip install -e ".[feedback]"
```

## Configuration

| Key | Description |
|-----|-------------|
| `topic` | What you want to track (e.g. `"contrastive learning"`) |
| `search_queries` | Static query list — used when `query_planning.enabled: false` |
| `search_provider` | `"tavily"` (default), `"serper"`, or `"both"` (fan-out to both, merged results) |
| `max_results_per_query` | Results per query — max 20 for Tavily, capped at 10 for Serper |
| `dedup_similarity_threshold` | Semantic similarity cutoff (0.85 recommended) |
| `delivery_method` | `"markdown"` saves a file, `"email"` sends to your inbox |
| `ollama_config.model` | Ollama model to use (e.g. `llama3.1:8b`, `qwen3:4b`) |
| `email_config` | SMTP settings — only needed if `delivery_method: "email"` |
| `query_planning.enabled` | Enable LLM-driven query planning (default: `false`) |
| `query_planning.max_queries` | Max queries per run including base topic (default: `5`) |
| `query_planning.max_dimensions` | Max research dimensions in the two-stage plan (default: `6`) |
| `query_planning.registry_resolution_threshold` | Cosine similarity threshold for merging dimensions into the registry (default: `0.88`) |
| `query_planning.hyde_abstracts_per_dim` | Number of HyDE abstracts generated per new dimension (default: `3`) |
| `query_planning.scaffold_registry_min_size` | Registry entries needed before switching from scaffold to full coverage map (default: `5`) |
| `planner_llm.model` | Reasoning model for topic decomposition (e.g. `deepseek-r1:7b`) |
| `planner_llm.think` | Enable extended chain-of-thought reasoning (default: `true`) |
| `planner_llm.timeout` | Max seconds to wait for the planner LLM response (default: `600`) |
| `feedback.enabled` | Write JSON sidecars and embed feedback links in emails (default: `false`) |
| `feedback.base_url` | URL where the feedback service is reachable (default: `http://localhost:8080`) |
| `feedback.db_path` | Path to the feedback SQLite database (default: `data/feedback.db`) |
| `feedback.min_votes_for_signals` | Minimum votes before feedback influences planning (default: `5`) |
| `feedback.signal_lookback_days` | How many days of vote history to consider (default: `30`) |

### Query planning

When enabled, the pipeline runs a two-stage planner each run:

1. **Stage 1 — Topic decomposition**: a reasoning LLM analyzes the base topic, previous research plan, extracted terms, query performance, and a coverage map of already-explored dimensions to produce a structured research plan. The prompt explicitly instructs the model to identify gaps and generate orthogonal dimensions rather than deepening already-covered areas.
2. **Stage 2 — Query synthesis**: deterministic code converts the plan into concrete search queries, always anchoring on the base topic and prioritizing high-priority, under-explored dimensions.

**Dimension registry** — each planned dimension is embedded using HyDE (Hypothetical Document Embeddings): the planner LLM writes a synthetic abstract for the dimension, which is encoded by `all-MiniLM-L6-v2` and stored in `dimension_registry`. On subsequent runs, new candidates are resolved against the registry by cosine similarity — dimensions that are semantically close enough (`registry_resolution_threshold`) reuse the existing entry rather than spawning a duplicate. This lets the system accumulate a stable identity for each research arm across runs.

**Topic scaffold** — on first run, the planner generates a coverage scaffold that maps the topic into five axes (methodological, domain, evaluation, theoretical, application). This is cached in SQLite and used as a guide when the registry is still sparse, ensuring early runs span the full topic space rather than clustering in one corner.

The research plan and reasoning trace are stored in SQLite and fed back into the next run, so the planner learns over time which angles yield new content and which are saturated.

On the first run (no term history) or if the LLM fails, the system falls back to deterministic query expansion using top extracted terms. With `enabled: false`, `search_queries` from config is used as-is.

```yaml
query_planning:
  enabled: true
  max_queries: 5
  max_dimensions: 6
  registry_resolution_threshold: 0.88
  hyde_abstracts_per_dim: 3
  scaffold_registry_min_size: 5

planner_llm:
  base_url: "http://localhost:11434"
  model: "deepseek-r1:7b"
  think: true
  timeout: 600
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

# Embedding visualizer
redpill viz                    # 3D scatter of all seen articles (default db from config)
redpill viz --db data/foo.db   # visualize a specific db
```

### Embedding visualizer

`redpill viz` reduces every seen article's embedding to 2D with UMAP and renders a self-contained interactive HTML file. X/Y axes are the semantic projection (similar articles cluster together); Z axis is `first_seen_date` so you can see how the corpus evolved over time. Points are colored by vote status — green for upvoted, red for downvoted, gray for not yet voted.

Requires the `viz` optional dependency group and a `viz/` output directory:

```bash
pip install -e ".[viz]"
mkdir -p viz
redpill viz
```

Output files are written to `viz/embeddings-YYYYMMDD-HHMMSS.html` and accumulate over time (each run produces a new file). The `viz/` directory is gitignored. Open any file in a browser — no server required.

### Feedback service

When `feedback.enabled: true`, each pipeline run writes a JSON sidecar to `data/digests/YYYY-MM-DD.json` alongside the digest. The feedback service reads these sidecars and serves an interactive page where you can vote on items.

Start the service (keep it running persistently):

```bash
redpill-feedback
# or in the background:
nohup redpill-feedback > data/feedback.log 2>&1 &
```

Then open `http://localhost:8080` after a pipeline run. Vote on items — once you reach `min_votes_for_signals` votes, the planner will start factoring in your preferences: boosting dimensions you upvote, deprioritizing ones you downvote, and preferring sources you engage with.

The service exposes four pages:

| Path | Description |
|------|-------------|
| `/` | List of all digests with item and vote counts |
| `/digest/YYYY-MM-DD` | Individual digest — vote on items here (this is the link in digest emails) |
| `/history` | All articles across every digest in a compact searchable view — filter by keyword, date range, or domain |
| `/bookmarks` | Reading list of all bookmarked articles across every digest |

Any article can be bookmarked from the digest or history views. Bookmarks are stored in `feedback.db` and toggled via `POST /api/bookmark` — bookmarking an already-bookmarked item removes it.

The feedback service and the pipeline are fully decoupled — they share only the filesystem (`data/digests/*.json`) and `feedback.db`. The pipeline reads `feedback.db` in read-only mode. If the service is down, the pipeline continues unaffected.

## Directory structure

```
redpill/
├── data/
│   ├── redpill.db                  # main SQLite db (seen items, embeddings, terms, query log)
│   │   or redpill_<topic>.db       # per-topic db when db_dir is set in config
│   ├── feedback.db                 # votes and bookmarks (written by the feedback service)
│   └── digests/
│       └── YYYY-MM-DD.json         # feedback sidecars (one per pipeline run)
├── viz/                            # gitignored — created manually
│   └── embeddings-YYYYMMDD-HHMMSS.html  # viz output files (accumulate over time)
├── sisyphus/                       # scheduling scripts
├── redpill/                        # pipeline source
├── feedback/                       # feedback service source
└── config.yaml                     # your local config (gitignored)
```

`data/` is created automatically on first run. `viz/` must be created manually before using `redpill viz`.

## Smoke test

Verifies the full pipeline (including v3 planning loop) without any real API calls:

```bash
python smoke_test.py      # compact output
python smoke_test.py -v   # verbose (show every check)
```

## Scheduling

Two helper scripts in `sisyphus/` handle scheduling. They load `.env` automatically so API keys are available to cron, and use absolute paths throughout.

**`sisyphus/run_pipeline.sh`** — runs the daily pipeline, logs to `data/redpill.log`.

**`sisyphus/start_feedback.sh`** — starts the feedback service in the background via `nohup`, logs to `data/feedback.log`. Safe to call multiple times — skips if already running.

Add both to your user crontab (`crontab -e`):

```
# Feedback service — start on every boot (60s delay for network to come up)
@reboot sleep 60 && /path/to/redpill/sisyphus/start_feedback.sh

# Daily pipeline — runs at 10 AM
0 10 * * * /path/to/redpill/sisyphus/run_pipeline.sh
```

Replace `/path/to/redpill` with your actual project path. No `sudo` required.

To restart the feedback service after making changes:

```bash
pkill -f redpill-feedback && ./sisyphus/start_feedback.sh
```

### Tailscale

The feedback service binds to `0.0.0.0:8080`, so it's reachable over Tailscale from any device on your tailnet. Set `feedback.base_url` in `config.yaml` to your machine's Tailscale address so vote links in digest emails point to the right place:

```yaml
feedback:
  base_url: "http://<tailscale-ip>:8080"
```

Using your Tailscale DNS hostname (`<machine>.tail*.ts.net`) is more resilient than a raw IP — the IP can change if the device is re-registered, but the hostname stays stable:

```yaml
feedback:
  base_url: "http://<machine>.tail*.ts.net:8080"
```
