# RedPill — TODO

> An autonomous AI agent that crawls the web daily for a given topic, deduplicates against previously seen content, summarizes new findings, and delivers a clean digest.

---

## Project Setup

- [ ] Initialize Python project with `pyproject.toml` (Python 3.11+)
- [ ] Set up project structure:

  ```
  redpill/
  ├── redpill/
  │   ├── __init__.py
  │   ├── search.py        # Web search / discovery
  │   ├── extract.py       # Content extraction from URLs
  │   ├── dedup.py          # Deduplication engine
  │   ├── summarize.py      # LLM summarization
  │   ├── deliver.py        # Digest delivery (email / markdown)
  │   ├── state.py          # SQLite state management
  │   ├── config.py         # Configuration loading
  │   └── main.py           # Orchestrator — runs the full pipeline
  ├── data/                  # SQLite DB and output digests live here
  ├── config.example.yaml    # Example config with comments
  ├── .env.example           # API keys template
  ├── tests/
  ├── TODO.md
  └── README.md
  ```

- [ ] Create `config.example.yaml` with the following fields:
  - `topic`: string (e.g. "contrastive learning")
  - `search_queries`: list of query variations for broader coverage
  - `max_results_per_query`: int (default 10)
  - `dedup_similarity_threshold`: float (default 0.85)
  - `delivery_method`: "markdown" | "email"
  - `ollama_config`: block with `base_url` (default "<http://localhost:11434>"), `model` (e.g. "llama3.1", "mistral", "qwen2.5")
  - `email_config`: optional block with smtp host, port, sender, recipient
  - `llm_provider`: "ollama" (default) | "anthropic" | "openai" (future placeholders)
  - `embedding_model`: "all-MiniLM-L6-v2" (local, zero cost)
- [ ] Create `.env.example` with placeholders for `TAVILY_API_KEY` and optional `SMTP_PASSWORD`
- [ ] Install core dependencies: `tavily-python`, `trafilatura`, `sentence-transformers`, `ollama`, `pyyaml`, `python-dotenv`

---

## Phase 1 — Discovery (search.py) ✅

- [x] Implement `search(queries: list[str], max_results: int) -> list[dict]` using Tavily API
  - Each result dict should have: `url`, `title`, `snippet`, `published_date` (if available)
  - Run all query variations and merge results
  - Deduplicate by URL at this stage (cheap exact match before we do anything expensive)
- [x] Handle Tavily API errors gracefully with retries (max 3, exponential backoff)
- [x] Add logging throughout (use Python `logging` module, not print statements)
- [x] Write tests with mocked Tavily responses

---

## Phase 2 — Content Extraction (extract.py) ✅

- [x] Implement `extract(url: str) -> str | None` using trafilatura
  - Return the main text content of the page, stripped of boilerplate
  - Return `None` if extraction fails (page down, paywall, etc.)
  - Set a timeout of 10 seconds per request
- [x] Implement `extract_batch(urls: list[str]) -> list[dict]` that processes URLs concurrently
  - Use `concurrent.futures.ThreadPoolExecutor` with max 5 workers
  - Return list of dicts: `url`, `title`, `content`, `extraction_success`
- [x] Skip URLs that are PDFs for now (flag for future support)
- [x] Write tests with sample HTML fixtures

---

## Phase 3 — State Management (state.py)

- [ ] Create SQLite database with a single table `seen_items`:
  - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
  - `url`: TEXT UNIQUE
  - `title`: TEXT
  - `content_hash`: TEXT (SHA256 of extracted content)
  - `embedding`: BLOB (serialized numpy array)
  - `summary`: TEXT
  - `first_seen_date`: TEXT (ISO format)
  - `topic`: TEXT
- [ ] Implement `init_db(db_path: str)` — creates table if not exists
- [ ] Implement `is_url_seen(url: str) -> bool` — exact URL match
- [ ] Implement `get_all_embeddings() -> list[tuple[int, np.ndarray]]` — returns id + embedding pairs for similarity search
- [ ] Implement `add_item(url, title, content_hash, embedding, summary)` — insert new row
- [ ] Implement `get_items_since(date: str) -> list[dict]` — for retrieving past digests
- [ ] Write tests using an in-memory SQLite database

---

## Phase 4 — Deduplication Engine (dedup.py)

- [ ] Implement `compute_embedding(text: str) -> np.ndarray` using sentence-transformers `all-MiniLM-L6-v2`
  - Load model once at module level (lazy singleton)
  - Truncate input text to first 512 tokens to stay within model limits
- [ ] Implement `is_semantic_duplicate(embedding: np.ndarray, existing_embeddings: list[tuple[id, np.ndarray]], threshold: float) -> bool`
  - Compute cosine similarity against all existing embeddings
  - Return True if any similarity exceeds threshold
  - Also return the closest match ID and score for logging purposes
- [ ] Implement `filter_new_items(candidates: list[dict], db: StateDB, threshold: float) -> list[dict]`
  - First pass: filter out already-seen URLs (cheap)
  - Second pass: compute embeddings for remaining candidates, filter out semantic duplicates (expensive)
  - Return only genuinely new items
- [ ] Log every dedup decision: "KEPT: {title}" or "DROPPED (url_match): {title}" or "DROPPED (semantic, score={score}): {title} ~ {matched_title}"
- [ ] Write tests with known duplicate and non-duplicate pairs

---

## Phase 5 — Summarization (summarize.py)

- [ ] Implement `summarize_item(content: str, topic: str) -> dict` using Ollama Python client
  - Connect to Ollama at configured `base_url` with configured `model`
  - Prompt should ask for: `title`, `summary` (2-3 sentences), `key_insight` (one line on why it matters), `relevance_score` (1-5 how relevant to the topic)
  - Return structured dict parsed from LLM response (instruct the model to respond in JSON)
  - Use a system prompt that establishes the agent's role as a research assistant
- [ ] Implement `generate_digest(items: list[dict], topic: str, date: str) -> str`
  - Send all items to Ollama in a single call
  - Ask it to produce a formatted daily digest in markdown
  - Include: date header, item count, then each item with title, summary, key insight, and source link
  - Sort by relevance_score descending
- [ ] Add a health check: verify Ollama is running and the configured model is pulled before starting
- [ ] Handle Ollama connection errors and timeouts gracefully
- [ ] Abstract LLM calls behind a simple interface (`LLMClient` with a `generate` method) so swapping providers later is just a new implementation
- [ ] Write tests with mocked API responses

---

## Phase 6 — Delivery (deliver.py)

- [ ] Implement `deliver_markdown(digest: str, output_dir: str, date: str)` — writes digest to `data/digests/YYYY-MM-DD.md`
- [ ] Implement `deliver_email(digest: str, config: dict)` — sends digest via SMTP
  - Convert markdown to simple HTML for the email body
  - Subject line: "RedPill Digest: {topic} — {date}"
- [ ] Implement `deliver(digest: str, config: dict)` — dispatcher that calls the right method based on config
- [ ] Write tests (mock SMTP, verify markdown file creation)

---

## Phase 7 — Orchestrator (main.py)

- [ ] Implement `run_pipeline(config_path: str)` that executes the full flow:
  1. Load config and initialize DB
  2. Run search queries → get candidate URLs
  3. Extract content from URLs
  4. Filter through dedup engine
  5. If no new items, deliver a "nothing new today" message and exit
  6. Summarize new items and generate digest
  7. Deliver digest
  8. Update state DB with all new items
- [ ] Add CLI interface using `argparse`:
  - `redpill run` — execute the full pipeline once
  - `redpill run --dry-run` — do everything except deliver and update state (useful for testing)
  - `redpill history --last N` — show the last N digests
  - `redpill stats` — show total items seen, items per day average, top sources
- [ ] Add proper error handling: if any step fails, log the error and continue with remaining items
- [ ] Make the pipeline idempotent — safe to run multiple times in a day without duplicating

---

## Phase 8 — Scheduling & Deployment

- [ ] Create a `run.sh` wrapper script that activates venv and runs the pipeline
- [ ] Document cron setup: `0 7 * * * /path/to/run.sh` (daily at 7 AM)
- [ ] Alternative: create a GitHub Actions workflow for scheduled runs
  - Workflow file: `.github/workflows/daily-digest.yml`
  - Trigger: `schedule: cron: '0 7 * * *'`
  - Store SQLite DB as a workflow artifact or use a persistent branch for state
- [ ] Add a Dockerfile for containerized deployment (optional)

---

## Phase 9 — Post-MVP Enhancements (Future)

- [ ] **Multi-topic support**: extend config to accept a list of topics, run pipeline per topic
- [ ] **Source preferences**: add a `sources` config block where users can block/boost specific domains
- [ ] **PDF support**: add PDF extraction in extract.py (use `pymupdf` or `pdfplumber`)
- [ ] **ArXiv integration**: add a dedicated ArXiv search in search.py using their API for academic topics
- [ ] **Development-level dedup**: cluster multiple articles about the same underlying paper/event into a single digest entry
- [ ] **Web dashboard**: simple Flask/FastAPI app to browse past digests and manage topics
- [ ] **Feedback loop**: let user mark items as "relevant" or "not useful" to tune future results
- [ ] **RSS feed output**: generate an RSS feed of digests for feed reader consumption

---

## Notes for Claude Code

- Always load config from `config.yaml` in project root (fall back to `config.example.yaml`)
- All API keys come from `.env` file via `python-dotenv` (only Tavily needs a key; Ollama is local)
- Ollama must be running locally with the configured model already pulled (`ollama pull <model>`)
- LLM calls should go through an `LLMClient` abstraction so adding Anthropic/OpenAI later is just a new class
- Use type hints everywhere
- Use `logging` module with format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Each module should be independently testable — no hidden global state
- SQLite DB path defaults to `data/redpill.db`
- Digest output defaults to `data/digests/`
- When in doubt about a design decision, optimize for simplicity over cleverness
