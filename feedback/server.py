"""
feedback/server.py — FastAPI feedback service.

Endpoints
---------
GET  /                  — Digest listing page (index.html)
GET  /digest/{date}     — Interactive digest page (digest.html)
POST /api/vote          — Record a vote for a digest item
GET  /health            — Health check

Running
-------
    redpill-feedback            # via pyproject.toml entry point
    python -m feedback.server   # directly

The service discovers JSON sidecars in data/digests/ and lazily ingests them
into feedback.db on first access.  The pipeline writes sidecars; this service
only reads them.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from feedback.db import FeedbackDB
from feedback.models import VoteRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RedPill Feedback")

# Resolve templates directory relative to *this file* so the service works
# regardless of the working directory the process was launched from.
_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_DB_PATH = "data/feedback.db"
_SIDECAR_DIR = Path("data/digests")

db = FeedbackDB(_DB_PATH)


# ---------------------------------------------------------------------------
# Helper: discover sidecar files on disk
# ---------------------------------------------------------------------------

def _discover_sidecar_dates() -> list[str]:
    """Return ISO date strings for all JSON sidecars in data/digests/, newest first."""
    if not _SIDECAR_DIR.exists():
        return []
    dates = sorted(
        (p.stem for p in _SIDECAR_DIR.glob("*.json")),
        reverse=True,
    )
    return dates


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Digest listing page.

    Shows all digests that have been ingested into feedback.db, merged with
    any sidecar files on disk that have not yet been ingested (so newly
    written sidecars appear in the list immediately without requiring a
    manual ingest step).
    """
    ingested: list[dict] = db.get_available_digests()
    ingested_dates: set[str] = {d["date"] for d in ingested}

    # Merge in sidecar dates that haven't been ingested yet.
    disk_dates = _discover_sidecar_dates()
    for date_str in disk_dates:
        if date_str not in ingested_dates:
            ingested.append({"date": date_str, "item_count": 0, "vote_count": 0})

    # Re-sort by date descending (stable across merged sources).
    digests = sorted(ingested, key=lambda d: d["date"], reverse=True)

    return templates.TemplateResponse(
        request,
        "index.html",
        {"digests": digests},
    )


@app.get("/digest/{date}", response_class=HTMLResponse)
async def digest_page(date: str, request: Request) -> HTMLResponse:
    """Interactive digest page.

    Lazily ingests the JSON sidecar on first visit.  Returns 404 if the
    sidecar file does not exist.
    """
    sidecar_path = _SIDECAR_DIR / f"{date}.json"

    if not sidecar_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No digest found for {date}. The pipeline may not have run yet.",
        )

    # Lazy ingestion: import sidecar into feedback.db if not yet done.
    if not db.is_digest_ingested(date):
        try:
            count = db.ingest_digest(str(sidecar_path))
            logger.info("Lazily ingested %d item(s) for digest %s", count, date)
        except (ValueError, OSError) as exc:
            logger.error("Failed to ingest sidecar for %s: %s", date, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load digest data for {date}: {exc}",
            ) from exc

    items = db.get_digest_items(date)

    # Derive topic from the first item (all items share the same topic).
    topic = items[0]["topic"] if items else date

    return templates.TemplateResponse(
        request,
        "digest.html",
        {
            "digest_date": date,
            "topic": topic,
            "items": items,
        },
    )


@app.post("/api/vote")
async def record_vote(vote_request: VoteRequest) -> dict:
    """Record a vote for a digest item.

    Uses last-vote-wins semantics: if the user votes on the same item twice,
    the new vote replaces the old one.

    Returns
    -------
    JSON: {"status": "ok", "item_id": "...", "vote": "up"|"down"}

    Raises
    ------
    404 if item_id does not exist in feedback.db.
    422 if the request body is malformed (handled by FastAPI/Pydantic).
    """
    try:
        result = db.record_vote(
            item_id=vote_request.item_id,
            vote=vote_request.vote,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "status": "ok",
        "item_id": result["item_id"],
        "vote": result["vote"],
    }


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the feedback service via uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
