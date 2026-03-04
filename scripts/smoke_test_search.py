"""
Smoke test for Phase 1 (search.py) against the real Tavily API.

Usage:
    python scripts/smoke_test_search.py

Requires TAVILY_API_KEY in your .env file or environment.
"""

from dotenv import load_dotenv

load_dotenv()

from redpill.search import search

results = search(["contrastive learning 2026"], max_results=5)

for r in results:
    print(r["url"], "|", r["title"])
