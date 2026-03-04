"""
deliver.py — Digest delivery: markdown file or email.

Public API:
    deliver_markdown(digest: str, output_dir: str, date: str)
        Writes to {output_dir}/YYYY-MM-DD.md.

    deliver_email(digest: str, config: dict)
        Sends via SMTP. Converts markdown to simple HTML.
        Subject: "RedPill Digest: {topic} — {date}"

    deliver(digest: str, config: dict)
        Dispatcher — calls the right method based on config["delivery_method"].
"""
