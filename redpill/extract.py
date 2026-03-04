"""
extract.py — Step 2: Fetch URLs and extract main article text via trafilatura.

Public API:
    extract(url: str) -> str | None
        Returns main text, or None on failure (timeout, paywall, etc.).
        Timeout: 10s per request. Skips PDFs.

    extract_batch(urls: list[str]) -> list[dict]
        Concurrent extraction (ThreadPoolExecutor, max 5 workers).
        Each result: {url, title, content, extraction_success}

Design note — why we use requests + trafilatura.extract() instead of trafilatura.fetch_url():
    trafilatura.fetch_url() caches its urllib3 pool globally after the first call.
    That means the DOWNLOAD_TIMEOUT config only takes effect when the pool is first
    created — making per-call timeout control unreliable. By fetching with requests
    (which exposes an explicit per-call timeout parameter) and passing the raw HTML
    to trafilatura.extract(), we get a deterministic 10-second timeout on every
    single request without relying on global mutable state.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urlparse

import requests
import trafilatura

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT: int = 10  # seconds per HTTP request
_MAX_WORKERS: int = 5


def _is_pdf_url(url: str) -> bool:
    """Return True if the URL path ends with .pdf (case-insensitive).

    PDF support is flagged for future implementation (see TODO.md Phase 9).
    This is a heuristic — content-type sniffing is not performed.
    """
    path = urlparse(url).path
    return path.lower().endswith(".pdf")


def _fetch_html(url: str, timeout: int = _REQUEST_TIMEOUT) -> Optional[str]:
    """Fetch raw HTML from a URL using requests with an explicit timeout.

    Returns the response text on success, or None on any network/HTTP error.
    Caller is responsible for deciding what to do with a None return.
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; redpill-bot/0.1)"},
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout:
        logger.warning("Request timed out after %ds: %s", timeout, url)
    except requests.exceptions.HTTPError as exc:
        logger.warning("HTTP error fetching %s: %s", url, exc)
    except requests.exceptions.RequestException as exc:
        logger.warning("Network error fetching %s: %s", url, exc)
    return None


def _parse_extraction(raw_json: str) -> tuple[Optional[str], Optional[str]]:
    """Parse trafilatura's JSON output and return (text, title).

    Returns (None, None) if the JSON is malformed or missing required fields.
    This is a defensive wrapper — trafilatura's JSON schema is stable but we
    don't want a parse error to propagate as an unhandled exception.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse trafilatura JSON output: %s", exc)
        return None, None

    text: Optional[str] = data.get("text") or None
    title: Optional[str] = data.get("title") or None
    return text, title


def extract(url: str) -> Optional[str]:
    """Fetch a URL and return its main article text, stripped of boilerplate.

    Returns None when:
    - The URL points to a PDF (skipped; future work)
    - The HTTP request fails (timeout, 4xx/5xx, network error)
    - trafilatura cannot identify extractable main content

    Parameters
    ----------
    url:
        The page URL to fetch and extract.

    Returns
    -------
    Extracted article text as a plain string, or None on any failure.
    """
    if _is_pdf_url(url):
        logger.info("Skipping PDF URL (not yet supported): %s", url)
        return None

    logger.debug("Fetching: %s", url)
    html = _fetch_html(url)
    if html is None:
        return None

    raw_json = trafilatura.extract(html, with_metadata=True, output_format="json")
    if raw_json is None:
        logger.info("No extractable content found: %s", url)
        return None

    text, _ = _parse_extraction(raw_json)
    if text is None:
        logger.info("Extraction returned empty text: %s", url)
        return None

    logger.debug("Extracted %d chars from: %s", len(text), url)
    return text


def _extract_one(url: str) -> dict:
    """Extract content from a single URL and return a result dict.

    This is the per-worker function for extract_batch. It always returns a dict
    with all required keys — failures are represented as extraction_success=False
    rather than raising exceptions, so one bad URL doesn't abort the batch.
    """
    if _is_pdf_url(url):
        logger.info("Skipping PDF URL (not yet supported): %s", url)
        return {
            "url": url,
            "title": "",
            "content": None,
            "extraction_success": False,
        }

    logger.debug("Fetching: %s", url)
    html = _fetch_html(url)
    if html is None:
        return {
            "url": url,
            "title": "",
            "content": None,
            "extraction_success": False,
        }

    raw_json = trafilatura.extract(html, with_metadata=True, output_format="json")
    if raw_json is None:
        logger.info("No extractable content found: %s", url)
        return {
            "url": url,
            "title": "",
            "content": None,
            "extraction_success": False,
        }

    text, title = _parse_extraction(raw_json)
    if text is None:
        logger.info("Extraction returned empty text: %s", url)
        return {
            "url": url,
            "title": "",
            "content": None,
            "extraction_success": False,
        }

    logger.info("Extracted %d chars from: %s", len(text), url)
    return {
        "url": url,
        "title": title or "",
        "content": text,
        "extraction_success": True,
    }


def extract_batch(urls: list[str]) -> list[dict]:
    """Extract content from multiple URLs concurrently.

    Uses a ThreadPoolExecutor with up to _MAX_WORKERS (5) threads. Results are
    returned in the same order as the input URL list, regardless of which
    fetches complete first.

    Parameters
    ----------
    urls:
        List of page URLs to fetch and extract.

    Returns
    -------
    List of dicts, one per input URL, with keys:
        url              — the original URL
        title            — page title extracted by trafilatura, or "" if unavailable
        content          — extracted article text, or None on failure
        extraction_success — True only when content was successfully extracted
    """
    if not urls:
        logger.debug("extract_batch() called with empty URL list — returning []")
        return []

    logger.info("Starting batch extraction of %d URLs (%d workers)", len(urls), _MAX_WORKERS)

    # Pre-allocate results indexed by URL so we can reassemble in input order.
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_url = {executor.submit(_extract_one, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as exc:
                # _extract_one is designed to not raise, but defend against
                # unexpected exceptions (e.g. from a buggy trafilatura version).
                logger.error("Unexpected error extracting %s: %s", url, exc)
                results[url] = {
                    "url": url,
                    "title": "",
                    "content": None,
                    "extraction_success": False,
                }

    ordered = [results[url] for url in urls]

    succeeded = sum(1 for r in ordered if r["extraction_success"])
    logger.info(
        "Batch extraction complete: %d/%d succeeded", succeeded, len(urls)
    )
    return ordered
