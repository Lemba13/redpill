"""
Smoke test for Phase 6 (deliver.py) — sends a real test email.

Usage:
    python scripts/smoke_test_email.py

Requires SMTP_PASSWORD in your .env file and a valid email_config in config.yaml.
"""

from dotenv import load_dotenv

load_dotenv()

import yaml
from redpill.deliver import deliver_email

with open("config.yaml") as f:
    config = yaml.safe_load(f)

DUMMY_DIGEST = """# RedPill Digest — Test
**2026-03-07** | 1 new item

---

## 1. This is a test email
If you're reading this, email delivery is working correctly.

**Key insight:** The pipeline can deliver digests to your inbox.
**Relevance:** 5/5 | [Source](https://github.com/Lemba13/redpill)

---
"""

print("Sending test email...")
deliver_email(
    digest=DUMMY_DIGEST,
    topic=config["topic"],
    date="2026-03-07",
    config=config["email_config"],
)
print("Done! Check your inbox.")
