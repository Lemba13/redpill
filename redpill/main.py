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

import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def cli() -> None:
    load_dotenv()  # Must run before any config reading or os.getenv() calls.
                   # override=False (default) means real env vars beat .env —
                   # local dev and CI both work without special-casing.
