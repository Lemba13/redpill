#!/usr/bin/env python3
"""
smoke_test.py — End-to-end smoke test for the v2 pipeline.

Tests the full feedback loop without any real API calls or disk state:
  1. State.py — new tables and APIs
  2. LLM utils — JSON parsing
  3. Term extractor — extraction + filtering
  4. Query planner — LLM path, fallback path, no-history path
  5. Full pipeline integration — two consecutive "runs" via an in-memory DB,
     verifying that terms extracted in run 1 influence queries in run 2

Run with:
    python smoke_test.py
    python smoke_test.py -v      # verbose output
"""

import json
import sqlite3
import sys
import textwrap
from datetime import date
from unittest.mock import MagicMock

VERBOSE = "-v" in sys.argv

TODAY = date.today().isoformat()
TOPIC = "contrastive learning"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

_results: list[tuple[bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((condition, name))
    icon = PASS if condition else FAIL
    msg = f"  {icon} {name}"
    if not condition and detail:
        msg += f"\n      → {detail}"
    if VERBOSE or not condition:
        print(msg)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _fresh_conn() -> sqlite3.Connection:
    """Open a fresh in-memory SQLite connection with all tables."""
    from redpill.state import init_db_conn
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db_conn(c)
    c.commit()
    return c


def _mock_llm(response: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = response
    return client


# ─────────────────────────────────────────────────────────────────────────────
# 1. State — new tables and APIs
# ─────────────────────────────────────────────────────────────────────────────

section("1. State — new tables and APIs")

from redpill.state import (
    get_query_performance_conn,
    get_recent_terms_conn,
    get_top_terms_conn,
    init_db_conn,
    log_query_conn,
    store_extracted_terms_conn,
    update_query_stats_conn,
)

conn = _fresh_conn()

# Tables exist
for tbl in ("seen_items", "extracted_terms", "query_log"):
    row = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl}'"
    ).fetchone()
    check(f"table '{tbl}' exists", row is not None)

# Term upsert
terms = [
    {"term": "SimCLR", "topic": TOPIC, "category": "technique",
     "first_seen": TODAY, "last_seen": TODAY},
    {"term": "MoCo", "topic": TOPIC, "category": "framework",
     "first_seen": TODAY, "last_seen": TODAY},
]
store_extracted_terms_conn(terms, conn)
store_extracted_terms_conn(terms, conn)  # second upsert increments frequency
conn.commit()

count = conn.execute("SELECT COUNT(*) FROM extracted_terms").fetchone()[0]
check("upsert does not create duplicates", count == 2, f"count={count}")

freq = conn.execute(
    "SELECT frequency FROM extracted_terms WHERE term = 'SimCLR'"
).fetchone()["frequency"]
check("frequency increments on re-upsert", freq == 2, f"freq={freq}")

recent = get_recent_terms_conn(TOPIC, 30, conn)
check("get_recent_terms returns both terms", len(recent) == 2, f"len={len(recent)}")

top = get_top_terms_conn(TOPIC, 1, conn)
check("get_top_terms limit respected", len(top) == 1)

# Query log
qid = log_query_conn("test query", TODAY, "base", TOPIC, conn)
check("log_query returns int id", isinstance(qid, int) and qid > 0, f"qid={qid!r}")

update_query_stats_conn(qid, results_count=10, new_items=4, kept_items=2, conn=conn)
conn.commit()

perf = get_query_performance_conn(TOPIC, 14, conn)
check("query performance entry found", len(perf) == 1)
r = perf[0]
check("query stats stored correctly",
      r["results_count"] == 10 and r["new_items"] == 4 and r["kept_items"] == 2,
      f"got {r['results_count']}/{r['new_items']}/{r['kept_items']}")

conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM utils — JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

section("2. LLM utils — JSON parsing")

from redpill.llm_utils import extract_json, strip_think_blocks

check("strip_think_blocks removes single block",
      strip_think_blocks("<think>reasoning</think>actual") == "actual")

check("strip_think_blocks removes multiple blocks",
      "<think>" not in strip_think_blocks("<think>a</think>x<think>b</think>y"))

check("extract_json parses clean object",
      extract_json('{"k": 1}') == {"k": 1})

check("extract_json parses clean array",
      extract_json('[{"term": "SimCLR"}]') == [{"term": "SimCLR"}])

check("extract_json strips think blocks",
      extract_json('<think>ignore</think>{"k": 2}') == {"k": 2})

check("extract_json strips markdown fences",
      extract_json('```json\n{"k": 3}\n```') == {"k": 3})

check("extract_json returns None on garbage",
      extract_json("definitely not json") is None)

check("extract_json finds object inside preamble",
      isinstance(extract_json('Here is the result: {"a": 1}'), dict))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Term extractor
# ─────────────────────────────────────────────────────────────────────────────

section("3. Term extractor")

from redpill.term_extractor import MIN_RELEVANCE_SCORE, extract_terms, extract_terms_batch

good_term = json.dumps([
    {"term": "SimCLR", "category": "technique", "relevance": 5},
    {"term": "Geoffrey Hinton", "category": "author", "relevance": 4},
])

item = {
    "url": "https://example.com/paper",
    "title": "SimCLR Paper",
    "content": "This paper introduces SimCLR by Geoffrey Hinton.",
    "snippet": "",
    "extraction_success": True,
    "relevance_score": 4,
}

terms = extract_terms(item, TOPIC, _mock_llm(good_term))
check("extract_terms returns list", isinstance(terms, list))
check("extract_terms correct count", len(terms) == 2, f"len={len(terms)}")
check("term has required fields",
      all(k in terms[0] for k in ("term", "category", "source_url", "source_title", "topic", "first_seen", "last_seen")))
check("source_url attached", terms[0]["source_url"] == "https://example.com/paper")
check("topic attached", terms[0]["topic"] == TOPIC)

# Low-relevance term filtered
low = json.dumps([{"term": "research", "category": "keyword", "relevance": 1}])
result = extract_terms(item, TOPIC, _mock_llm(low))
check("low-relevance term filtered out", result == [], f"got {result}")

# LLM failure
failing_client = MagicMock()
failing_client.generate.side_effect = RuntimeError("Ollama down")
check("extract_terms returns [] on LLM error", extract_terms(item, TOPIC, failing_client) == [])

# Batch filtering
items = [
    {**item, "extraction_success": False},           # filtered: no content
    {**item, "relevance_score": 0},                  # filtered: low relevance
    {**item, "url": "https://good.com", "relevance_score": 4, "extraction_success": True},
]
client = _mock_llm(good_term)
batch_result = extract_terms_batch(items, TOPIC, client)
check("batch skips failed/low-relevance items",
      client.generate.call_count == 1,
      f"generate called {client.generate.call_count} time(s)")
check("batch returns terms from qualifying item", len(batch_result) > 0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Query planner
# ─────────────────────────────────────────────────────────────────────────────

section("4. Query planner")

from redpill.query_planner import plan_queries, plan_queries_fallback

# Fallback with no term history
conn = _fresh_conn()
result = plan_queries_fallback(TOPIC, conn, max_queries=3)
check("fallback: empty DB → only base query", len(result) == 1)
check("fallback: base query is topic", result[0]["query"] == TOPIC)
check("fallback: base source correct", result[0]["source"] == "base")

# Populate terms
terms_to_insert = [
    {"term": "SimCLR", "topic": TOPIC, "category": "technique", "first_seen": TODAY, "last_seen": TODAY},
    {"term": "MoCo", "topic": TOPIC, "category": "framework", "first_seen": TODAY, "last_seen": TODAY},
]
store_extracted_terms_conn(terms_to_insert, conn)
conn.commit()

result = plan_queries_fallback(TOPIC, conn, max_queries=3)
check("fallback: base + 2 term-expanded queries", len(result) == 3, f"len={len(result)}")
check("fallback: expanded queries start with topic",
      all(q["query"].startswith(TOPIC) for q in result[1:]))
check("fallback: expanded source = extracted_term",
      all(q["source"] == "extracted_term" for q in result[1:]))

# LLM planner — success path
llm_response = json.dumps([
    {"query": "contrastive learning SimCLR benchmark", "reasoning": "Specific to SimCLR."},
    {"query": "contrastive learning MoCo v3 2025", "reasoning": "Recent MoCo work."},
])
result = plan_queries(TOPIC, conn, _mock_llm(llm_response), max_queries=3)
check("planner: first query is always base", result[0]["source"] == "base")
check("planner: LLM queries follow", result[1]["source"] == "llm_planned")
check("planner: max_queries respected", len(result) == 3, f"len={len(result)}")

# LLM planner — failure falls back
failing = MagicMock()
failing.generate.side_effect = RuntimeError("down")
result = plan_queries(TOPIC, conn, failing, max_queries=3)
sources = {q["source"] for q in result}
check("planner: LLM error → fallback (no llm_planned)", "llm_planned" not in sources)

# LLM planner — bad JSON falls back
result = plan_queries(TOPIC, conn, _mock_llm("not json at all"), max_queries=3)
check("planner: bad JSON → fallback", "llm_planned" not in {q["source"] for q in result})

# No term history → no LLM call
conn_empty = _fresh_conn()
no_llm_client = MagicMock()
plan_queries(TOPIC, conn_empty, no_llm_client, max_queries=3)
check("planner: no terms → LLM not called", not no_llm_client.generate.called)

conn.close()
conn_empty.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Full feedback loop — two simulated runs
# ─────────────────────────────────────────────────────────────────────────────

section("5. Full feedback loop — two simulated pipeline runs")

from redpill.state import get_recent_terms_conn, get_top_terms_conn, log_query_conn

conn = _fresh_conn()

# ── Run 1: no term history → only base query planned ──
print("  [Run 1] Planning queries (no history) ...")

result_run1 = plan_queries_fallback(TOPIC, conn, max_queries=5)
check("run 1: only base query (no terms yet)", len(result_run1) == 1)

# Simulate search + summarization → extract terms
simulated_items = [
    {
        "url": f"https://paper{i}.com",
        "title": f"Paper {i}",
        "content": f"This paper discusses SimCLR, MoCo, and BYOL in contrastive learning context {i}.",
        "snippet": "",
        "extraction_success": True,
        "relevance_score": 4,
    }
    for i in range(3)
]

terms_run1 = json.dumps([
    {"term": "SimCLR", "category": "technique", "relevance": 5},
    {"term": "BYOL", "category": "technique", "relevance": 4},
    {"term": "MoCo", "category": "framework", "relevance": 4},
])
client_run1 = _mock_llm(terms_run1)
extracted_run1 = extract_terms_batch(simulated_items, TOPIC, client_run1)
check("run 1: terms extracted", len(extracted_run1) > 0, f"len={len(extracted_run1)}")

store_extracted_terms_conn(extracted_run1, conn)
conn.commit()

stored = get_recent_terms_conn(TOPIC, 30, conn)
check("run 1: terms persisted to DB", len(stored) >= 3, f"len={len(stored)}")

# ── Run 2: term history exists → planner uses it ──
print("  [Run 2] Planning queries (with history) ...")

llm_plan_run2 = json.dumps([
    {"query": f"{TOPIC} SimCLR benchmark 2025", "reasoning": "SimCLR is high-frequency."},
    {"query": f"{TOPIC} BYOL self-supervised", "reasoning": "BYOL appeared in run 1."},
    {"query": f"{TOPIC} MoCo v3", "reasoning": "MoCo framework follow-up."},
    {"query": f"{TOPIC} linear evaluation protocol", "reasoning": "Common eval method."},
])
result_run2 = plan_queries(TOPIC, conn, _mock_llm(llm_plan_run2), max_queries=5)

check("run 2: more than 1 query (terms influenced plan)", len(result_run2) > 1, f"len={len(result_run2)}")
check("run 2: base query still first", result_run2[0]["source"] == "base")
check("run 2: LLM queries present", any(q["source"] == "llm_planned" for q in result_run2))

# Log run 2 queries
for pq in result_run2:
    qid = log_query_conn(pq["query"], TODAY, pq["source"], TOPIC, conn)
    update_query_stats_conn(qid, results_count=8, new_items=3, kept_items=2, conn=conn)
conn.commit()

perf = get_query_performance_conn(TOPIC, 14, conn)
check(
    "run 2: all queries logged",
    len(perf) == len(result_run2),
    f"logged={len(perf)}, planned={len(result_run2)}",
)
check("run 2: stats recorded", all(r["results_count"] == 8 for r in perf))

conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI commands — queries and terms
# ─────────────────────────────────────────────────────────────────────────────

section("6. CLI commands — queries and terms")

import argparse
import io
from contextlib import redirect_stdout
from redpill.main import _build_parser, _cmd_queries, _cmd_terms

parser = _build_parser()

# Parser structure
for sub in ("run", "history", "stats", "plan", "queries", "terms"):
    args = parser.parse_args([sub])
    check(f"parser: '{sub}' subcommand registered", args.command == sub)

args = parser.parse_args(["queries", "--last", "30"])
check("queries --last parses correctly", args.last == 30)

args = parser.parse_args(["terms", "--top", "10"])
check("terms --top parses correctly", args.top == 10)

args = parser.parse_args(["terms", "--recent"])
check("terms --recent defaults to 30", args.recent == 30)

args = parser.parse_args(["terms", "--recent", "7"])
check("terms --recent 7 parses correctly", args.recent == 7)

# _cmd_queries output — wire against in-memory state via temp file
import tempfile, os, yaml as _yaml

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, "test.db")
    cfg_path = os.path.join(tmpdir, "config.yaml")
    cfg = {"topic": TOPIC, "search_queries": [TOPIC], "db_path": db_path,
           "ollama_config": {"base_url": "http://localhost:11434", "model": "qwen3:4b"}}
    with open(cfg_path, "w") as f:
        _yaml.dump(cfg, f)

    from redpill.state import init_db, log_query, update_query_stats, store_extracted_terms
    init_db(db_path)

    qid = log_query(f"{TOPIC} SimCLR", TODAY, "llm_planned", TOPIC, db_path=db_path)
    update_query_stats(qid, results_count=8, new_items=3, kept_items=2, db_path=db_path)

    ns = argparse.Namespace(config=cfg_path, last=14)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_queries(ns)
    out = buf.getvalue()
    check("queries cmd: shows query text", f"{TOPIC} SimCLR" in out, f"output: {out[:200]!r}")
    check("queries cmd: shows source", "llm_planned" in out)
    check("queries cmd: shows stats", "8" in out)

    store_extracted_terms([
        {"term": "SimCLR", "topic": TOPIC, "category": "technique",
         "first_seen": TODAY, "last_seen": TODAY},
        {"term": "BYOL", "topic": TOPIC, "category": "technique",
         "first_seen": TODAY, "last_seen": TODAY},
    ], db_path=db_path)

    # --top
    ns_top = argparse.Namespace(config=cfg_path, top=10, recent=None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_terms(ns_top)
    out = buf.getvalue()
    check("terms --top: shows terms", "SimCLR" in out and "BYOL" in out)
    check("terms --top: shows category", "technique" in out)
    check("terms --top: header mentions 'all time'", "all time" in out)

    # --recent
    ns_recent = argparse.Namespace(config=cfg_path, top=20, recent=30)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _cmd_terms(ns_recent)
    out = buf.getvalue()
    check("terms --recent: shows terms", "SimCLR" in out)
    check("terms --recent: header mentions days", "30 day" in out)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

n_pass = sum(1 for ok, _ in _results if ok)
n_fail = sum(1 for ok, _ in _results if not ok)
total = len(_results)

print(f"\n{'═' * 60}")
print(f"  Results: {n_pass}/{total} passed", end="")
if n_fail:
    failed_names = [name for ok, name in _results if not ok]
    print(f"  ({n_fail} FAILED)")
    print(f"\n  Failed checks:")
    for name in failed_names:
        print(f"    {FAIL} {name}")
else:
    print(f"  — all good ✓")
print(f"{'═' * 60}\n")

sys.exit(0 if n_fail == 0 else 1)
