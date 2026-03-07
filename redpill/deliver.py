"""
deliver.py — Digest delivery: markdown file or email.

Public API:
    DeliveryError
        Custom exception raised when delivery fails. Callers (e.g. main.py)
        should catch this rather than smtplib or OSError directly.

    deliver_markdown(digest: str, output_dir: str, date: str) -> Path
        Writes digest to {output_dir}/{date}.md.
        Creates the directory tree if needed. Overwrites with a warning if
        the file already exists.

    deliver_email(digest: str, topic: str, date: str, config: dict) -> None
        Sends a multipart/alternative email (text/plain + text/html) via
        SMTP with STARTTLS on port 587.
        config keys: smtp_host, smtp_port, sender, recipient.
        SMTP_PASSWORD is read from os.environ["SMTP_PASSWORD"].

    deliver(digest: str, topic: str, date: str, config: dict) -> Path | None
        Dispatcher. Validates config, then calls deliver_markdown or
        deliver_email based on config["delivery_method"].
        Returns the written Path for markdown delivery, None for email.
        Raises DeliveryError on any failure.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import markdown as md_lib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DeliveryError(Exception):
    """Raised when digest delivery fails.

    Wraps lower-level exceptions (OSError, smtplib exceptions, missing config)
    behind a single exception type so the Phase 7 orchestrator has one thing
    to catch.
    """


# ---------------------------------------------------------------------------
# Markdown file delivery
# ---------------------------------------------------------------------------


def deliver_markdown(digest: str, output_dir: str, date: str) -> Path:
    """Write *digest* to ``{output_dir}/{date}.md``.

    Parameters
    ----------
    digest:
        The formatted markdown string produced by generate_digest().
    output_dir:
        Directory to write into. Created (including parents) if it does not
        exist. Relative paths are resolved against the current working
        directory.
    date:
        ISO date string used as the filename stem (e.g. "2026-03-07").

    Returns
    -------
    The resolved Path of the written file.

    Raises
    ------
    DeliveryError
        If the directory cannot be created or the file cannot be written.
    """
    out_dir = Path(output_dir).resolve()

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DeliveryError(
            f"deliver_markdown: cannot create output directory {out_dir!r}: {exc}"
        ) from exc

    out_path = out_dir / f"{date}.md"

    if out_path.exists():
        logger.warning(
            "deliver_markdown: %s already exists — overwriting", out_path
        )

    try:
        out_path.write_text(digest, encoding="utf-8")
    except OSError as exc:
        raise DeliveryError(
            f"deliver_markdown: cannot write digest to {out_path!r}: {exc}"
        ) from exc

    logger.info("deliver_markdown: digest written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Email delivery helpers
# ---------------------------------------------------------------------------


def _markdown_to_html(digest: str) -> str:
    """Convert a markdown digest string to an HTML document string.

    Wraps the converted fragment in a minimal HTML skeleton with a basic
    inline style so the output is readable across email clients. We keep
    styling conservative (font-family, max-width, colours) because anything
    beyond simple inline CSS is broken by most email clients.

    Uses the ``markdown`` library (PyPI: markdown) which handles the subset
    of CommonMark that generate_digest() produces: ATX headings, bold,
    horizontal rules, and inline links.
    """
    html_body: str = md_lib.markdown(digest, extensions=["nl2br"])
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset=\"utf-8\">"
        "<style>"
        "body{font-family:Georgia,serif;max-width:720px;margin:40px auto;"
        "color:#222;line-height:1.6;padding:0 20px}"
        "h1{font-size:1.4em;border-bottom:2px solid #e0e0e0;padding-bottom:8px}"
        "h2{font-size:1.1em;margin-top:1.6em}"
        "hr{border:none;border-top:1px solid #e0e0e0;margin:1.5em 0}"
        "a{color:#1a6eb5}"
        "strong{font-weight:600}"
        "</style>"
        "</head><body>"
        + html_body
        + "</body></html>"
    )


def _build_email(
    digest: str,
    topic: str,
    date: str,
    sender: str,
    recipient: str,
) -> MIMEMultipart:
    """Construct the multipart/alternative MIME message.

    The message has two parts in the correct MIME order: text/plain first
    (fallback), text/html second (preferred). RFC 2046 §5.1.4 specifies that
    the last part is the one clients should prefer, so HTML goes last.

    Parameters
    ----------
    digest:
        Raw markdown string — used as the plain-text body verbatim.
    topic:
        Research topic, used in the subject line.
    date:
        ISO date string, used in the subject line.
    sender:
        From address.
    recipient:
        To address.
    """
    msg = MIMEMultipart("alternative")
    # Subject line uses an em dash (—) as specified. Built by concatenation
    # because topic is user-supplied and may contain format characters.
    msg["Subject"] = "RedPill Digest: " + topic + " \u2014 " + date
    msg["From"] = sender
    msg["To"] = recipient

    plain_part = MIMEText(digest, "plain", "utf-8")
    html_part = MIMEText(_markdown_to_html(digest), "html", "utf-8")

    # Attach plain first, then HTML — recipients see the HTML version unless
    # their client explicitly prefers plain text.
    msg.attach(plain_part)
    msg.attach(html_part)

    return msg


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------


def deliver_email(digest: str, topic: str, date: str, config: dict) -> None:
    """Send *digest* as a multipart email via SMTP with STARTTLS.

    Parameters
    ----------
    digest:
        The formatted markdown string produced by generate_digest().
    topic:
        Research topic — appears in the email subject line.
    date:
        ISO date string — appears in the email subject line.
    config:
        Dict with keys: smtp_host (str), smtp_port (int), sender (str),
        recipient (str). SMTP_PASSWORD is read from the environment, not
        from this dict.

    Raises
    ------
    DeliveryError
        If SMTP_PASSWORD is not set, the SMTP connection fails, auth fails,
        or the message cannot be sent.
    """
    smtp_host: str = config["smtp_host"]
    smtp_port: int = int(config["smtp_port"])
    sender: str = config["sender"]
    recipient: str = config["recipient"]

    password = os.environ.get("SMTP_PASSWORD")
    if not password:
        raise DeliveryError(
            "deliver_email: SMTP_PASSWORD is not set in the environment. "
            "Add it to your .env file."
        )

    msg = _build_email(
        digest=digest,
        topic=topic,
        date=date,
        sender=sender,
        recipient=recipient,
    )

    logger.info(
        "deliver_email: connecting to %s:%d as %s", smtp_host, smtp_port, sender
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise DeliveryError(
            f"deliver_email: SMTP authentication failed for {sender!r}. "
            "Check SMTP_PASSWORD and the sender address."
        ) from exc
    except smtplib.SMTPConnectError as exc:
        raise DeliveryError(
            f"deliver_email: cannot connect to SMTP server {smtp_host}:{smtp_port}: {exc}"
        ) from exc
    except smtplib.SMTPException as exc:
        raise DeliveryError(f"deliver_email: SMTP error: {exc}") from exc
    except OSError as exc:
        # Covers socket-level errors: host unreachable, connection refused, timeout.
        raise DeliveryError(
            f"deliver_email: network error connecting to {smtp_host}:{smtp_port}: {exc}"
        ) from exc

    logger.info("deliver_email: digest sent to %s", recipient)


# ---------------------------------------------------------------------------
# Config validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_EMAIL_KEYS: tuple[str, ...] = ("smtp_host", "smtp_port", "sender", "recipient")


def _validate_email_config(config: dict) -> None:
    """Raise ValueError if any required email_config key is missing or blank.

    Called by the dispatcher before attempting delivery so failures are
    caught early with clear, actionable messages.
    """
    email_cfg = config.get("email_config")
    if not email_cfg:
        raise ValueError(
            "deliver: delivery_method is 'email' but 'email_config' is "
            "missing from config. Add an email_config block to config.yaml."
        )
    missing = [k for k in _REQUIRED_EMAIL_KEYS if not email_cfg.get(k)]
    if missing:
        raise ValueError(
            "deliver: email_config is missing required key(s): "
            + ", ".join(repr(k) for k in missing)
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def deliver(digest: str, topic: str, date: str, config: dict) -> Path | None:
    """Dispatch digest delivery based on ``config["delivery_method"]``.

    Parameters
    ----------
    digest:
        The formatted markdown string produced by generate_digest().
    topic:
        Research topic. Passed to deliver_email for the subject line; not
        used by deliver_markdown (which only needs output_dir and date).
    date:
        ISO date string (e.g. "2026-03-07"). Used as the markdown filename
        and in the email subject line.
    config:
        Full application config dict. Must contain "delivery_method".
        For email delivery must also contain "email_config".
        For markdown delivery must also contain "output_dir" (optional —
        defaults to "data/digests").

    Returns
    -------
    Path
        The written file Path, when delivery_method is "markdown".
    None
        When delivery_method is "email".

    Raises
    ------
    ValueError
        If delivery_method is unknown or required config keys are absent.
        Raised *before* any I/O so callers get fast, clear feedback.
    DeliveryError
        If the underlying delivery operation fails.
    """
    method: str = config.get("delivery_method", "")

    if method == "markdown":
        output_dir: str = config.get("output_dir", "data/digests")
        return deliver_markdown(digest=digest, output_dir=output_dir, date=date)

    if method == "email":
        _validate_email_config(config)
        deliver_email(
            digest=digest,
            topic=topic,
            date=date,
            config=config["email_config"],
        )
        return None

    raise ValueError(
        f"deliver: unknown delivery_method {method!r}. "
        "Expected 'markdown' or 'email'."
    )
