"""
main.py — Orchestrator and CLI entry point.

Pipeline (run_pipeline):
    1. Load config + init DB
    2. Search → candidate URLs
    3. Extract content
    4. Deduplicate
    5. If nothing new → deliver "nothing new today" and exit
    6. Summarize + generate digest
    7. Deliver digest
    8. Update state DB

CLI (via argparse):
    redpill run                  — full pipeline
    redpill run --dry-run        — skip deliver + state update
    redpill history --last N     — show last N digests
    redpill stats                — total seen, avg per day, top sources
"""

import argparse
import hashlib
import logging
import os
import sqlite3
import sys
from datetime import date as _date
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from redpill.config import get_feedback_config, get_search_provider
from redpill.dedup import compute_embedding, filter_new_items
from redpill.deliver import DeliveryError, deliver, generate_item_id, write_digest_sidecar
from redpill.extract import extract_batch
from redpill.feedback_reader import FeedbackReader
from redpill.query_planner import plan_queries, plan_queries_fallback
from redpill.search import search
from redpill.state import (
    DEFAULT_DB_PATH,
    add_item,
    get_query_performance,
    get_recent_terms,
    get_top_terms,
    init_db,
    log_query,
    store_extracted_terms,
    update_query_stats,
)
from redpill.summarize import OllamaClient, PlannerLLMClient, check_ollama, generate_digest, summarize_item
from redpill.term_extractor import extract_terms_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_CANDIDATES = ("config.yaml", "config.example.yaml")


def _load_config(config_path: str | None = None) -> dict:
    """Load and return the YAML config as a dict.

    If *config_path* is given, that file is used exclusively.
    Otherwise the function tries ``config.yaml`` then ``config.example.yaml``
    in the current working directory, in that order.

    Raises
    ------
    SystemExit
        If the file cannot be found or parsed.  Callers should not catch this —
        it is meant to terminate the process with a user-readable message.
    """
    candidates: list[str]
    if config_path is not None:
        candidates = [config_path]
    else:
        candidates = list(_CONFIG_CANDIDATES)

    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                with p.open(encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)
                if not isinstance(config, dict):
                    print(
                        f"ERROR: {path} did not parse to a mapping — "
                        "check that the file is valid YAML.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                logger.info("Loaded config from %s", p.resolve())
                return config
            except yaml.YAMLError as exc:
                print(f"ERROR: Failed to parse {path}: {exc}", file=sys.stderr)
                sys.exit(1)

    tried = ", ".join(candidates)
    print(
        f"ERROR: No config file found. Tried: {tried}\n"
        "Copy config.example.yaml to config.yaml and edit it.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: merge search results with extraction results
# ---------------------------------------------------------------------------

def _merge_search_and_extract(
    search_results: list[dict],
    extracted: list[dict],
) -> list[dict]:
    """Combine per-URL fields from search and extract into one dict per item.

    Search results carry: url, title, snippet, published_date.
    Extract results carry: url, title, content, extraction_success.

    The merged dict carries all fields.  The extraction title is preferred over
    the search title when it is non-empty (trafilatura tends to be more
    accurate).  Content is None when extraction failed.

    Order is preserved (same as *search_results* order).
    """
    extract_by_url: dict[str, dict] = {r["url"]: r for r in extracted}
    merged: list[dict] = []
    for sr in search_results:
        url = sr["url"]
        er = extract_by_url.get(url, {})
        title = er.get("title") or sr.get("title") or ""
        merged.append(
            {
                "url": url,
                "title": title,
                "snippet": sr.get("snippet") or "",
                "published_date": sr.get("published_date"),
                "content": er.get("content"),  # None when extraction failed
                "extraction_success": er.get("extraction_success", False),
                "source_query": sr.get("source_query") or "",
            }
        )
    return merged


# ---------------------------------------------------------------------------
# Helper: compute content hash
# ---------------------------------------------------------------------------

def _content_hash(content: str | None) -> str:
    """Return the SHA-256 hex digest of *content*, or of an empty string."""
    text = content or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str | None = None, dry_run: bool = False) -> None:
    """Execute the full redpill pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.  ``None`` triggers the default search
        order (``config.yaml`` then ``config.example.yaml``).
    dry_run:
        When True, the pipeline runs search → extract → dedup → summarize and
        prints the digest to stdout, but skips delivery and state updates
        (including query logging and term extraction).
        Useful for iterating on prompts and config without polluting the DB or
        sending emails.
    """
    today = _date.today().isoformat()

    # ------------------------------------------------------------------
    # Step 1: Load config + initialize DB
    # ------------------------------------------------------------------
    config = _load_config(config_path)

    topic: str = config.get("topic", "")
    static_queries: list[str] = config.get("search_queries", [])
    max_results: int = int(config.get("max_results_per_query", 10))
    threshold: float = float(config.get("dedup_similarity_threshold", 0.85))
    db_path: str = config.get("db_path", DEFAULT_DB_PATH)

    qp_cfg: dict = config.get("query_planning", {})
    use_planner: bool = bool(qp_cfg.get("enabled", False))
    max_queries: int = int(qp_cfg.get("max_queries", 5))
    registry_resolution_threshold: float = float(qp_cfg.get("registry_resolution_threshold", 0.88))
    hyde_abstracts_per_dim: int = int(qp_cfg.get("hyde_abstracts_per_dim", 3))
    scaffold_registry_min_size: int = int(qp_cfg.get("scaffold_registry_min_size", 5))

    ollama_cfg: dict = config.get("ollama_config", {})
    ollama_base_url: str = ollama_cfg.get("base_url", "http://localhost:11434")
    ollama_model: str = ollama_cfg.get("model", "qwen3:4b")

    planner_cfg: dict = config.get("planner_llm", {})

    # Feedback config — fully merged with defaults so all keys are present.
    fb_cfg: dict = get_feedback_config(config)
    feedback_enabled: bool = bool(fb_cfg["enabled"])
    feedback_base_url: str = str(fb_cfg["base_url"])
    feedback_db_path: str = str(fb_cfg["db_path"])
    min_votes_for_signals: int = int(fb_cfg["min_votes_for_signals"])
    signal_lookback_days: int = int(fb_cfg["signal_lookback_days"])

    if not topic:
        print("ERROR: 'topic' is required in config.", file=sys.stderr)
        sys.exit(1)

    if not use_planner and not static_queries:
        print(
            "ERROR: 'search_queries' must be a non-empty list in config "
            "(or set query_planning.enabled: true).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve and validate the search provider.  Fail fast here (before any
    # network or DB I/O) when SERPER_API_KEY is required but missing.
    try:
        provider_name: str = get_search_provider(config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if provider_name in ("serper", "both") and "SERPER_API_KEY" not in os.environ:
        print(
            f"ERROR: search_provider is {provider_name!r} but SERPER_API_KEY is not set. "
            "Add it to your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure the data directory exists before init_db tries to create the file.
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    init_db(db_path)
    logger.info("DB initialised at %s", db_path)

    # ------------------------------------------------------------------
    # Ollama health check (done early so failures surface before I/O)
    # ------------------------------------------------------------------
    try:
        check_ollama(ollama_base_url, ollama_model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    llm_client = OllamaClient(base_url=ollama_base_url, model=ollama_model)

    planner_client: OllamaClient | PlannerLLMClient
    if use_planner and planner_cfg.get("model"):
        planner_client = PlannerLLMClient(
            base_url=planner_cfg.get("base_url", ollama_base_url),
            model=planner_cfg["model"],
            think=bool(planner_cfg.get("think", True)),
            timeout=int(planner_cfg.get("timeout", 120)),
        )
        logger.info(
            "Using PlannerLLMClient (model=%r, think=%s) for query planning",
            planner_cfg["model"],
            planner_cfg.get("think", True),
        )
    else:
        planner_client = llm_client

    # ------------------------------------------------------------------
    # Step 1b: Load feedback signals (read-only, best-effort)
    # ------------------------------------------------------------------
    feedback_signals: dict | None = None
    if feedback_enabled and use_planner:
        fb_db = Path(feedback_db_path)
        if not fb_db.exists():
            logger.info(
                "Feedback DB not found at %s — cold start, skipping signals",
                feedback_db_path,
            )
        else:
            try:
                with FeedbackReader(feedback_db_path) as reader:
                    feedback_signals = reader.compute_preference_signals(
                        topic, days=signal_lookback_days
                    )
                logger.info(
                    "Loaded feedback signals: has_feedback=%s vote_count=%d",
                    feedback_signals.get("has_feedback"),
                    feedback_signals.get("vote_count", 0),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load feedback signals from %s: %s — continuing without",
                    feedback_db_path,
                    exc,
                )

    # ------------------------------------------------------------------
    # Step 2: Plan queries
    # ------------------------------------------------------------------
    if use_planner:
        logger.info("Query planning enabled (max_queries=%d) ...", max_queries)
        planner_conn = sqlite3.connect(db_path)
        planner_conn.row_factory = sqlite3.Row
        try:
            planned_queries = plan_queries(
                topic,
                planner_conn,
                planner_client,
                max_queries=max_queries,
                feedback_signals=feedback_signals,
                min_votes_for_signals=min_votes_for_signals,
                registry_resolution_threshold=registry_resolution_threshold,
                hyde_abstracts_per_dim=hyde_abstracts_per_dim,
                scaffold_registry_min_size=scaffold_registry_min_size,
                ucb_alpha=float(qp_cfg.get("ucb_alpha", 1.0)),
                promotion_k=int(qp_cfg.get("promotion_k", 3)),
                mmr_lambda_floor=float(qp_cfg.get("mmr_lambda_floor", 0.3)),
                saturation_decay_days=int(qp_cfg.get("saturation_decay_days", 7)),
                saturation_penalty_weight=float(qp_cfg.get("saturation_penalty_weight", 0.3)),
            )
        except Exception as exc:
            logger.warning("Query planner raised unexpectedly (%s) — using fallback", exc)
            planned_queries = plan_queries_fallback(topic, planner_conn, max_queries=max_queries)
        finally:
            planner_conn.close()
    else:
        # Backward-compat: wrap static queries as plain base-source dicts.
        planned_queries = [
            {"query": q, "source": "base", "reasoning": "Static query from config.", "dim_id": "dim_base"}
            for q in static_queries
        ]

    query_strings: list[str] = [pq["query"] for pq in planned_queries]

    # Build lookups from query string → plan_dimension name and dim_id.
    query_to_dimension: dict[str, str] = {}
    query_to_dim_id: dict[str, str] = {}
    for pq in planned_queries:
        q = pq["query"]
        # The dimension name lives in "reasoning" for llm_planned queries
        # from synthesize_queries() which formats it as "Dimension 'name': ...".
        # For all other sources we leave it blank — it's optional sidecar metadata.
        if pq.get("source") == "llm_planned":
            reasoning: str = pq.get("reasoning", "")
            import re as _re
            m = _re.search(r"Dimension\s+'([^']+)'", reasoning)
            query_to_dimension[q] = m.group(1) if m else ""
        else:
            query_to_dimension[q] = ""
        query_to_dim_id[q] = pq.get("dim_id") or "dim_fallback"

    logger.info(
        "Running %d search quer%s for topic %r ...",
        len(query_strings),
        "y" if len(query_strings) == 1 else "ies",
        topic,
    )

    # Log planned queries to the DB (skipped in dry-run).
    query_ids: list[int] = []
    if not dry_run:
        for pq in planned_queries:
            try:
                qid = log_query(
                    pq["query"], today, pq["source"], topic,
                    db_path=db_path,
                    dim_id=pq.get("dim_id"),
                )
                query_ids.append(qid)
            except Exception as exc:
                logger.warning("Failed to log query %r: %s", pq["query"], exc)
                query_ids.append(-1)

    # ------------------------------------------------------------------
    # Step 3: Search
    # ------------------------------------------------------------------
    try:
        candidates = search(query_strings, max_results=max_results, provider=provider_name)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        print(f"ERROR: Search step failed: {exc}", file=sys.stderr)
        sys.exit(1)

    logger.info("Search returned %d candidate URL(s)", len(candidates))

    if not candidates:
        logger.info("No candidates returned from search — nothing to do.")
        _maybe_deliver_nothing_new(config, topic, today, dry_run)
        return

    # ------------------------------------------------------------------
    # Step 4: Extract content
    # ------------------------------------------------------------------
    urls = [c["url"] for c in candidates]
    logger.info("Extracting content from %d URL(s) ...", len(urls))
    extracted = extract_batch(urls)

    merged = _merge_search_and_extract(candidates, extracted)

    # Annotate each merged item with source_query (from search.py's passthrough),
    # plan_dimension (derived from query_to_dimension lookup), and dim_id.
    for item in merged:
        sq: str = item.get("source_query") or ""
        item["plan_dimension"] = query_to_dimension.get(sq, "")
        item["dim_id"] = query_to_dim_id.get(sq, "dim_fallback")

    # ------------------------------------------------------------------
    # Step 5: Deduplicate
    # ------------------------------------------------------------------
    logger.info("Running deduplication (threshold=%.2f) ...", threshold)
    new_items = filter_new_items(merged, db_path=db_path, threshold=threshold)
    logger.info("%d new item(s) after dedup", len(new_items))

    # Update query log stats after dedup (best-effort).
    if not dry_run and query_ids:
        n_new = len(new_items)
        n_results = len(candidates)
        for qid in query_ids:
            if qid < 0:
                continue
            try:
                update_query_stats(qid, results_count=n_results, new_items=n_new, kept_items=0, db_path=db_path)
            except Exception as exc:
                logger.warning("Failed to update query stats for id=%d: %s", qid, exc)

    # ------------------------------------------------------------------
    # Step 6: Nothing new?
    # ------------------------------------------------------------------
    if not new_items:
        logger.info("All items were duplicates — nothing new today.")
        _maybe_deliver_nothing_new(config, topic, today, dry_run)
        return

    # ------------------------------------------------------------------
    # Step 7: Summarize
    # ------------------------------------------------------------------
    logger.info("Summarizing %d item(s) ...", len(new_items))
    summarized: list[dict] = []
    for item in new_items:
        try:
            result = summarize_item(
                item,
                topic=topic,
                client=llm_client,
                db_path=db_path if not dry_run else None,
            )
            # summarize_item always returns; it uses a fallback on LLM errors.
            # Attach the original item fields we still need for state persistence
            # and sidecar annotation.
            result["_content"] = item.get("content")
            result["_snippet"] = item.get("snippet") or ""
            result["source_query"] = item.get("source_query") or ""
            result["plan_dimension"] = item.get("plan_dimension") or ""
            result["dim_id"] = item.get("dim_id") or "dim_fallback"
            summarized.append(result)
        except Exception as exc:
            # Belt-and-suspenders: summarize_item is designed not to raise,
            # but defend here anyway.
            logger.warning(
                "Unexpected error summarizing url=%r: %s — skipping item",
                item.get("url"),
                exc,
            )

    if not summarized:
        logger.warning("All items failed summarization — nothing to deliver.")
        _maybe_deliver_nothing_new(config, topic, today, dry_run)
        return

    digest = generate_digest(summarized, topic=topic, date=today)

    # ------------------------------------------------------------------
    # Step 7b: Extract terms (skipped in dry-run)
    # ------------------------------------------------------------------
    # Build a lookup from URL back to the merged item (needed both here and
    # in the persist step below).
    merged_by_url: dict[str, dict] = {m["url"]: m for m in merged}

    if not dry_run:
        # Combine relevance_score from summarized items with content/
        # extraction_success from merged items so the batch filter works.
        items_for_extraction: list[dict] = []
        for s_item in summarized:
            url = s_item.get("url", "")
            m = merged_by_url.get(url, {})
            items_for_extraction.append(
                {
                    **m,
                    "relevance_score": s_item.get("relevance_score", 0),
                }
            )

        logger.info("Running term extraction on %d summarized item(s) ...", len(items_for_extraction))
        try:
            extracted_terms = extract_terms_batch(
                items_for_extraction, topic, llm_client, db_path=db_path
            )
        except Exception as exc:
            logger.warning("Term extraction failed: %s — continuing without terms", exc)
            extracted_terms = []

        if extracted_terms:
            logger.info("Storing %d extracted term(s) ...", len(extracted_terms))
            try:
                store_extracted_terms(extracted_terms, db_path=db_path)
            except Exception as exc:
                logger.warning("Failed to store extracted terms: %s", exc)

    # ------------------------------------------------------------------
    # Step 8: Deliver (skipped in dry-run mode)
    # ------------------------------------------------------------------
    if dry_run:
        logger.info("Dry-run mode: printing digest to stdout, skipping delivery.")
        print(digest)
    else:
        logger.info("Delivering digest ...")
        try:
            out = deliver(
                digest,
                topic=topic,
                date=today,
                config=config,
                feedback_base_url=feedback_base_url if feedback_enabled else "",
            )
            if out is not None:
                logger.info("Digest written to %s", out)
        except (DeliveryError, ValueError) as exc:
            logger.error("Delivery failed: %s", exc)
            print(f"ERROR: Delivery failed: {exc}", file=sys.stderr)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Step 8b: Write feedback sidecar (best-effort, after delivery)
        # ------------------------------------------------------------------
        if feedback_enabled:
            try:
                sidecar_path = write_digest_sidecar(
                    items=summarized,
                    topic=topic,
                    date=today,
                    feedback_base_url=feedback_base_url,
                )
                logger.info("Feedback sidecar written to %s", sidecar_path)
            except Exception as exc:
                logger.warning("Failed to write feedback sidecar: %s — continuing", exc)

        # ------------------------------------------------------------------
        # Step 9: Persist state
        # ------------------------------------------------------------------
        logger.info("Persisting %d new item(s) to state DB ...", len(summarized))
        n_kept = 0
        for s_item in summarized:
            url: str = s_item.get("url", "")
            original = merged_by_url.get(url, {})
            content = s_item.pop("_content", None) or original.get("content")
            snippet = s_item.pop("_snippet", "") or original.get("snippet", "")
            # Pop sidecar-only fields so they don't interfere with state persistence.
            s_item.pop("source_query", None)
            s_item.pop("plan_dimension", None)
            dim_id_val: str = s_item.pop("dim_id", None) or "dim_fallback"

            title: str = s_item.get("title") or original.get("title") or ""
            summary: str = s_item.get("summary") or ""
            chash = _content_hash(content)
            embed_text = content if content and content.strip() else snippet
            try:
                embedding = compute_embedding(embed_text if embed_text else url)
                add_item(
                    url=url,
                    title=title,
                    content_hash=chash,
                    embedding=embedding,
                    summary=summary,
                    topic=topic,
                    db_path=db_path,
                    first_seen_date=today,
                    dim_id=dim_id_val,
                )
                n_kept += 1
                logger.debug("Persisted: %r", url)
            except Exception as exc:
                logger.warning(
                    "Failed to persist item url=%r: %s — continuing",
                    url,
                    exc,
                )

        # Final query stats update — per-dim kept_items and avg_relevance_score.
        # Note: dim_id and source_query are popped from s_item in the persist loop
        # above, so re-derive dim_id from source_query via merged_by_url.
        kept_per_dim_id: dict[str, int] = {}
        relevance_per_dim_id: dict[str, list[float]] = {}
        for s_item in summarized:
            url = s_item.get("url", "")
            original = merged_by_url.get(url, {})
            sq = original.get("source_query") or ""
            did = query_to_dim_id.get(sq, "dim_fallback")
            kept_per_dim_id[did] = kept_per_dim_id.get(did, 0) + 1
            rel = s_item.get("relevance_score")
            if rel is not None:
                relevance_per_dim_id.setdefault(did, []).append(float(rel))

        # Build query_id → dim_id mapping for per-dim stats write.
        qid_to_dim: dict[int, str] = {
            qid: pq.get("dim_id", "dim_fallback")
            for qid, pq in zip(query_ids, planned_queries)
            if qid >= 0
        }

        for qid in query_ids:
            if qid < 0:
                continue
            try:
                did = qid_to_dim.get(qid, "dim_fallback")
                rel_list = relevance_per_dim_id.get(did)
                avg_rel = sum(rel_list) / len(rel_list) if rel_list else None
                update_query_stats(
                    qid,
                    results_count=len(candidates),
                    new_items=len(new_items),
                    kept_items=kept_per_dim_id.get(did, 0),
                    avg_relevance_score=avg_rel,
                    db_path=db_path,
                )
            except Exception as exc:
                logger.warning("Failed to finalize query stats for id=%d: %s", qid, exc)

        # End-of-run bandit update: rewards, run_count, pool transitions.
        if use_planner:
            try:
                from redpill.bandit import (
                    check_promotions as _check_promotions,
                    check_retirements as _check_retirements,
                    update_rewards as _update_rewards,
                )
                bandit_conn = sqlite3.connect(db_path)
                bandit_conn.row_factory = sqlite3.Row
                try:
                    run_results = [
                        {
                            "dim_id": did,
                            "kept_items": kept_per_dim_id.get(did, 0),
                            "avg_relevance_score": (
                                sum(relevance_per_dim_id[did]) / len(relevance_per_dim_id[did])
                                if relevance_per_dim_id.get(did) else 0.0
                            ),
                        }
                        for did in {pq.get("dim_id", "dim_fallback") for pq in planned_queries}
                        if did not in ("dim_fallback", "dim_base")
                    ]
                    _update_rewards(run_results, bandit_conn, topic)
                    bandit_conn.execute(
                        """
                        UPDATE dimension_registry
                        SET run_count = run_count + 1, last_seen = date('now')
                        WHERE dim_id IN (
                            SELECT DISTINCT dim_id FROM query_log
                            WHERE run_date = ? AND topic = ?
                        )
                        """,
                        (today, topic),
                    )
                    _check_promotions(bandit_conn, topic, k=int(qp_cfg.get("promotion_k", 3)))
                    _check_retirements(bandit_conn, topic)
                    bandit_conn.commit()
                except Exception as exc:
                    logger.warning("Bandit end-of-run update failed: %s", exc)
                    bandit_conn.rollback()
                finally:
                    bandit_conn.close()
            except ImportError:
                pass

    logger.info("Pipeline complete.")


def _maybe_deliver_nothing_new(
    config: dict,
    topic: str,
    today: str,
    dry_run: bool,
) -> None:
    """Generate and deliver (or print) a 'nothing new today' digest."""
    nothing_digest = generate_digest([], topic=topic, date=today)
    if dry_run:
        print(nothing_digest)
        return
    try:
        deliver(nothing_digest, topic=topic, date=today, config=config)
    except (DeliveryError, ValueError) as exc:
        logger.error("Failed to deliver 'nothing new' message: %s", exc)
        print(f"ERROR: Delivery failed: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    """Handler for: redpill run [--config PATH] [--dry-run]."""
    run_pipeline(config_path=args.config, dry_run=args.dry_run)


def _cmd_history(args: argparse.Namespace) -> None:
    """Handler for: redpill history [--config PATH] [--last N]."""
    config = _load_config(args.config)
    output_dir = Path(config.get("output_dir", "data/digests"))

    if not output_dir.exists():
        print(f"No digests directory found at {output_dir.resolve()}")
        return

    md_files = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
    last_n = md_files[-args.last :] if args.last else md_files

    if not last_n:
        print("No digests found.")
        return

    parts: list[str] = []
    for path in last_n:
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Could not read digest file %s: %s", path, exc)

    print("\n---\n".join(parts))


def _cmd_plan(args: argparse.Namespace) -> None:
    """Handler for: redpill plan [--config PATH] [--max-queries N]."""
    config = _load_config(args.config)

    topic: str = config.get("topic", "")
    if not topic:
        print("ERROR: 'topic' is required in config.", file=sys.stderr)
        sys.exit(1)

    db_path: str = config.get("db_path", DEFAULT_DB_PATH)
    qp_cfg: dict = config.get("query_planning", {})
    max_queries: int = args.max_queries or int(qp_cfg.get("max_queries", 5))
    ucb_alpha: float = float(qp_cfg.get("ucb_alpha", 1.0))

    ollama_cfg: dict = config.get("ollama_config", {})
    ollama_base_url: str = ollama_cfg.get("base_url", "http://localhost:11434")
    ollama_model: str = ollama_cfg.get("model", "qwen3:4b")

    # Ensure DB exists (may not if user hasn't run the pipeline yet).
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    init_db(db_path)

    try:
        check_ollama(ollama_base_url, ollama_model)
        llm_client = OllamaClient(base_url=ollama_base_url, model=ollama_model)
    except RuntimeError as exc:
        logger.warning("Ollama unavailable (%s) — using deterministic fallback.", exc)
        llm_client = None  # type: ignore[assignment]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if llm_client is not None:
            queries = plan_queries(topic, conn, llm_client, max_queries=max_queries)
        else:
            queries = plan_queries_fallback(topic, conn, max_queries=max_queries)

        # Show bandit pool state if available.
        try:
            from redpill.bandit import (
                compute_budget_split,
                compute_saturation_penalty,
                compute_ucb_scores,
                get_exploit_pool,
                get_explore_pool,
            )
            from redpill.registry import get_dimension_axis_tags

            exploit_dims = get_exploit_pool(conn, topic)
            explore_dims = get_explore_pool(conn, topic)

            print(f"\nPool state for topic: {topic!r}")
            print(f"  Exploit pool: {len(exploit_dims)} dim(s)")
            print(f"  Explore pool: {len(explore_dims)} dim(s)\n")

            if exploit_dims:
                scores = compute_ucb_scores(exploit_dims, conn, topic, alpha=ucb_alpha)
                print("Exploit pool (UCB selection):")
                for dim in sorted(exploit_dims, key=lambda d: scores.get(d["dim_id"], 0), reverse=True):
                    did = dim["dim_id"]
                    score = scores.get(did, 0)
                    penalty = compute_saturation_penalty(did, conn)
                    sat_note = "  ← saturated" if penalty > 0.15 else ""
                    score_str = "inf" if score == float("inf") else f"{score:.3f}"
                    print(
                        f"  {did}  {dim['canonical_name']!r:<40} "
                        f"score={score_str}  runs={dim['run_count']}  penalty={penalty:.2f}{sat_note}"
                    )

            if explore_dims:
                print("\nExplore pool (coverage-gap selection):")
                for dim in explore_dims:
                    did = dim["dim_id"]
                    axis_info = get_dimension_axis_tags(did, topic, conn)
                    print(
                        f"  {did}  {dim['canonical_name']!r:<40} "
                        f"axis={axis_info.get('primary_axis', 'unknown')}  "
                        f"last_seen={dim['last_seen'] or 'never'}"
                    )

            n_exploit, n_explore = compute_budget_split(max_queries, conn, topic)
            print(f"\nBudget split: {n_exploit} exploit + {n_explore} explore "
                  f"+ 1 base = {n_exploit + n_explore + 1} total\n")
        except Exception as exc:
            logger.debug("Could not show bandit pool state: %s", exc)

    finally:
        conn.close()

    print(f"Planned {len(queries)} query/ies for topic: {topic!r}\n")
    for i, q in enumerate(queries, start=1):
        print(f"  {i}. [{q['source']}] {q['query']}")
        if q.get("reasoning"):
            print(f"     → {q['reasoning']}")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Handler for: redpill stats [--config PATH]."""
    config = _load_config(args.config)
    db_path = config.get("db_path", DEFAULT_DB_PATH)

    if not Path(db_path).exists():
        print(f"No database found at {db_path}. Run 'redpill run' first.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Total items
        total = conn.execute("SELECT COUNT(*) FROM seen_items").fetchone()[0]
        if total == 0:
            print("No items in database yet. Run 'redpill run' first.")
            return

        # Date range for average calculation
        row = conn.execute(
            "SELECT MIN(first_seen_date), MAX(first_seen_date) FROM seen_items"
        ).fetchone()
        first_date_str: str = row[0]
        last_date_str: str = row[1]

        first_date = _date.fromisoformat(first_date_str)
        last_date = _date.fromisoformat(last_date_str)
        days_span = (last_date - first_date).days + 1  # inclusive
        avg_per_day = total / days_span

        # Top 5 sources by domain
        urls = conn.execute("SELECT url FROM seen_items").fetchall()
        domain_counts: dict[str, int] = {}
        for (url,) in urls:
            try:
                domain = urlparse(url).netloc or url
            except Exception:
                domain = url
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        top_sources = sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    finally:
        conn.close()

    print(f"Total items seen:        {total}")
    print(f"First seen date:         {first_date_str}")
    print(f"Last seen date:          {last_date_str}")
    print(f"Days tracked:            {days_span}")
    print(f"Avg items / day:         {avg_per_day:.1f}")
    print()
    print("Top sources:")
    for domain, count in top_sources:
        print(f"  {count:>4}  {domain}")


def _cmd_queries(args: argparse.Namespace) -> None:
    """Handler for: redpill queries [--config PATH] [--last N]."""
    config = _load_config(args.config)
    db_path: str = config.get("db_path", DEFAULT_DB_PATH)
    topic: str = config.get("topic", "")

    if not topic:
        print("ERROR: 'topic' is required in config.", file=sys.stderr)
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"No database found at {db_path}. Run 'redpill run' first.")
        return

    rows = get_query_performance(topic, db_path=db_path, days=args.last)

    if not rows:
        print(f"No query history found for topic {topic!r} in the last {args.last} day(s).")
        return

    print(f"Query history for topic {topic!r} (last {args.last} day(s)):\n")
    print(f"  {'Date':<12}  {'Source':<14}  {'Results':>7}  {'New':>5}  {'Kept':>5}  Query")
    print(f"  {'-'*12}  {'-'*14}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*40}")
    for r in rows:
        print(
            f"  {r['run_date']:<12}  {r['source']:<14}  "
            f"{r['results_count']:>7}  {r['new_items']:>5}  {r['kept_items']:>5}  "
            f"{r['query_text']}"
        )


def _cmd_terms(args: argparse.Namespace) -> None:
    """Handler for: redpill terms [--config PATH] [--top N | --recent [DAYS]]."""
    config = _load_config(args.config)
    db_path: str = config.get("db_path", DEFAULT_DB_PATH)
    topic: str = config.get("topic", "")

    if not topic:
        print("ERROR: 'topic' is required in config.", file=sys.stderr)
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"No database found at {db_path}. Run 'redpill run' first.")
        return

    if args.recent is not None:
        days = args.recent
        rows = get_recent_terms(topic, db_path=db_path, days=days)
        header = f"Terms seen in the last {days} day(s) for topic {topic!r}:"
    else:
        limit = args.top
        rows = get_top_terms(topic, db_path=db_path, limit=limit)
        header = f"Top {limit} terms for topic {topic!r} (all time):"

    if not rows:
        print(f"No terms found. Run 'redpill run' a few times to build up term history.")
        return

    print(f"{header}\n")
    print(f"  {'Freq':>5}  {'Category':<12}  {'Last seen':<12}  Term")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*40}")
    for r in rows:
        print(
            f"  {r['frequency']:>5}  {(r['category'] or 'keyword'):<12}  "
            f"{r['last_seen']:<12}  {r['term']}"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redpill",
        description="Daily AI-powered research digest agent.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # --- redpill run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Execute the full pipeline once.",
    )
    run_parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: config.yaml → config.example.yaml).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run everything but skip delivery and state updates.",
    )
    run_parser.set_defaults(func=_cmd_run)

    # --- redpill history ---
    hist_parser = subparsers.add_parser(
        "history",
        help="Show the last N digests.",
    )
    hist_parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: config.yaml → config.example.yaml).",
    )
    hist_parser.add_argument(
        "--last",
        type=int,
        default=5,
        metavar="N",
        help="Number of most recent digests to show (default: 5).",
    )
    hist_parser.set_defaults(func=_cmd_history)

    # --- redpill stats ---
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show database statistics.",
    )
    stats_parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: config.yaml → config.example.yaml).",
    )
    stats_parser.set_defaults(func=_cmd_stats)

    # --- redpill plan ---
    plan_parser = subparsers.add_parser(
        "plan",
        help="Show what search queries the planner would generate (dry-run).",
    )
    plan_parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: config.yaml → config.example.yaml).",
    )
    plan_parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help="Override max queries (default: query_planning.max_queries from config or 5).",
    )
    plan_parser.set_defaults(func=_cmd_plan)

    # --- redpill queries ---
    queries_parser = subparsers.add_parser(
        "queries",
        help="Show query performance history.",
    )
    queries_parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: config.yaml → config.example.yaml).",
    )
    queries_parser.add_argument(
        "--last",
        type=int,
        default=14,
        metavar="DAYS",
        help="How many days back to look (default: 14).",
    )
    queries_parser.set_defaults(func=_cmd_queries)

    # --- redpill terms ---
    terms_parser = subparsers.add_parser(
        "terms",
        help="Browse the extracted term database.",
    )
    terms_parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: config.yaml → config.example.yaml).",
    )
    _terms_group = terms_parser.add_mutually_exclusive_group()
    _terms_group.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Show the N most frequent terms of all time (default: 20).",
    )
    _terms_group.add_argument(
        "--recent",
        type=int,
        nargs="?",
        const=30,
        metavar="DAYS",
        help="Show terms seen in the last DAYS days (default: 30 when flag is given).",
    )
    terms_parser.set_defaults(func=_cmd_terms)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    load_dotenv()  # Must run before any config reading or os.getenv() calls.
                   # override=False (default) means real env vars beat .env —
                   # local dev and CI both work without special-casing.

    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
