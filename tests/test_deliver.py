"""
tests/test_deliver.py — Unit tests for redpill.deliver

No real filesystem paths outside of tmp_path are touched. No real SMTP
connection is ever made. All network I/O is intercepted via unittest.mock.

Test structure:
    TestDeliverMarkdown     — file creation, overwrite behaviour, return value,
                              directory creation, error handling
    TestMarkdownToHtml      — HTML conversion correctness and structure
    TestBuildEmail          — MIME message structure and headers
    TestDeliverEmail        — SMTP interaction, env var handling, error mapping
    TestValidateEmailConfig — config validation logic
    TestDeliver             — dispatcher routing, config validation, error contract
"""

import os
import smtplib
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from redpill.deliver import (
    DeliveryError,
    _build_email,
    _markdown_to_html,
    _validate_email_config,
    deliver,
    deliver_email,
    deliver_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures and shared helpers
# ---------------------------------------------------------------------------

SAMPLE_DIGEST = (
    "# RedPill Digest — contrastive learning\n"
    "**2026-03-07** | 2 new items\n\n"
    "---\n\n"
    "## 1. Great Paper\n"
    "A summary sentence. Another sentence.\n\n"
    "**Key insight:** This matters.\n"
    "**Relevance:** 5/5 | [Source](https://example.com/paper)\n\n"
    "---\n"
)

SAMPLE_EMAIL_CONFIG = {
    "smtp_host": "smtp.example.com",
    "smtp_port": 587,
    "sender": "redpill@example.com",
    "recipient": "you@example.com",
}

SAMPLE_CONFIG_MARKDOWN = {
    "delivery_method": "markdown",
    "output_dir": "data/digests",
}

SAMPLE_CONFIG_EMAIL = {
    "delivery_method": "email",
    "email_config": SAMPLE_EMAIL_CONFIG,
}


# ---------------------------------------------------------------------------
# TestDeliverMarkdown
# ---------------------------------------------------------------------------


class TestDeliverMarkdown:
    def test_creates_file_at_correct_path(self, tmp_path: Path) -> None:
        result = deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")
        assert result == tmp_path / "2026-03-07.md"
        assert result.exists()

    def test_file_content_matches_digest(self, tmp_path: Path) -> None:
        deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")
        written = (tmp_path / "2026-03-07.md").read_text(encoding="utf-8")
        assert written == SAMPLE_DIGEST

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        result = deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        result = deliver_markdown(SAMPLE_DIGEST, str(nested), "2026-03-07")
        assert result.exists()
        assert result.parent == nested.resolve()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "2026-03-07.md"
        path.write_text("old content", encoding="utf-8")
        deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")
        assert path.read_text(encoding="utf-8") == SAMPLE_DIGEST

    def test_overwrite_emits_warning(self, tmp_path: Path, caplog) -> None:
        path = tmp_path / "2026-03-07.md"
        path.write_text("old content", encoding="utf-8")
        import logging
        with caplog.at_level(logging.WARNING, logger="redpill.deliver"):
            deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")
        assert any("already exists" in r.message for r in caplog.records)

    def test_no_warning_on_first_write(self, tmp_path: Path, caplog) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="redpill.deliver"):
            deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")
        assert not any("already exists" in r.message for r in caplog.records)

    def test_filename_uses_date_as_stem(self, tmp_path: Path) -> None:
        result = deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2025-12-31")
        assert result.name == "2025-12-31.md"

    def test_raises_delivery_error_on_write_failure(self, tmp_path: Path) -> None:
        # Make the output path a directory so write_text fails.
        bad_path = tmp_path / "2026-03-07.md"
        bad_path.mkdir()
        with pytest.raises(DeliveryError, match="cannot write"):
            deliver_markdown(SAMPLE_DIGEST, str(tmp_path), "2026-03-07")

    def test_file_encoded_as_utf8(self, tmp_path: Path) -> None:
        digest_with_unicode = "# RedPill Digest — résumé\n**2026-03-07**\n"
        deliver_markdown(digest_with_unicode, str(tmp_path), "2026-03-07")
        raw_bytes = (tmp_path / "2026-03-07.md").read_bytes()
        assert raw_bytes == digest_with_unicode.encode("utf-8")


# ---------------------------------------------------------------------------
# TestMarkdownToHtml
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    def test_returns_string(self) -> None:
        result = _markdown_to_html("# Hello")
        assert isinstance(result, str)

    def test_contains_html_boilerplate(self) -> None:
        result = _markdown_to_html("# Hello")
        assert "<!DOCTYPE html>" in result
        assert "<html>" in result
        assert "</html>" in result

    def test_h1_converted(self) -> None:
        result = _markdown_to_html("# My Title")
        assert "<h1>" in result
        assert "My Title" in result

    def test_h2_converted(self) -> None:
        result = _markdown_to_html("## Section")
        assert "<h2>" in result

    def test_bold_converted(self) -> None:
        result = _markdown_to_html("**bold text**")
        assert "<strong>bold text</strong>" in result

    def test_link_converted(self) -> None:
        result = _markdown_to_html("[Source](https://example.com)")
        assert 'href="https://example.com"' in result
        assert "Source" in result

    def test_content_preserved(self) -> None:
        result = _markdown_to_html(SAMPLE_DIGEST)
        assert "Great Paper" in result
        assert "contrastive learning" in result

    def test_has_charset_meta(self) -> None:
        result = _markdown_to_html("x")
        assert 'charset="utf-8"' in result or "charset=utf-8" in result.lower()


# ---------------------------------------------------------------------------
# TestBuildEmail
# ---------------------------------------------------------------------------


class TestBuildEmail:
    def _make_msg(self, topic: str = "AI", date: str = "2026-03-07"):
        return _build_email(
            digest=SAMPLE_DIGEST,
            topic=topic,
            date=date,
            sender="from@example.com",
            recipient="to@example.com",
        )

    def test_subject_contains_topic(self) -> None:
        msg = self._make_msg(topic="contrastive learning")
        assert "contrastive learning" in msg["Subject"]

    def test_subject_contains_date(self) -> None:
        msg = self._make_msg(date="2026-03-07")
        assert "2026-03-07" in msg["Subject"]

    def test_subject_format(self) -> None:
        msg = self._make_msg(topic="AI", date="2026-03-07")
        # Em dash U+2014
        assert "RedPill Digest: AI \u2014 2026-03-07" == msg["Subject"]

    def test_from_header_set(self) -> None:
        msg = _build_email(
            SAMPLE_DIGEST, "AI", "2026-03-07", "from@example.com", "to@example.com"
        )
        assert msg["From"] == "from@example.com"

    def test_to_header_set(self) -> None:
        msg = _build_email(
            SAMPLE_DIGEST, "AI", "2026-03-07", "from@example.com", "to@example.com"
        )
        assert msg["To"] == "to@example.com"

    def test_is_multipart_alternative(self) -> None:
        msg = self._make_msg()
        assert msg.get_content_type() == "multipart/alternative"

    def test_has_two_parts(self) -> None:
        msg = self._make_msg()
        assert len(msg.get_payload()) == 2

    def test_first_part_is_plain_text(self) -> None:
        msg = self._make_msg()
        plain_part = msg.get_payload()[0]
        assert plain_part.get_content_type() == "text/plain"

    def test_second_part_is_html(self) -> None:
        msg = self._make_msg()
        html_part = msg.get_payload()[1]
        assert html_part.get_content_type() == "text/html"

    def test_plain_part_contains_raw_markdown(self) -> None:
        msg = self._make_msg()
        plain_part = msg.get_payload()[0]
        # Decode the payload (it's base64 or quoted-printable due to utf-8).
        decoded = plain_part.get_payload(decode=True).decode("utf-8")
        assert "# RedPill Digest" in decoded

    def test_html_part_contains_converted_html(self) -> None:
        msg = self._make_msg()
        html_part = msg.get_payload()[1]
        decoded = html_part.get_payload(decode=True).decode("utf-8")
        assert "<h1>" in decoded

    def test_topic_with_braces_does_not_raise(self) -> None:
        # Topic is user-supplied; must not blow up f-strings.
        msg = _build_email(SAMPLE_DIGEST, "AI {systems}", "2026-03-07", "a@b.com", "c@d.com")
        assert "AI {systems}" in msg["Subject"]


# ---------------------------------------------------------------------------
# TestDeliverEmail
# ---------------------------------------------------------------------------


class TestDeliverEmail:
    def _call(self, env: dict | None = None, smtp_mock: MagicMock | None = None):
        """Helper: call deliver_email with SMTP_PASSWORD set in the environment."""
        env = env or {"SMTP_PASSWORD": "secret"}
        with patch.dict(os.environ, env, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                if smtp_mock:
                    MockSMTP.return_value.__enter__ = lambda s: smtp_mock
                    MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                else:
                    mock_server = MagicMock()
                    MockSMTP.return_value.__enter__ = lambda s: mock_server
                    MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                deliver_email(
                    digest=SAMPLE_DIGEST,
                    topic="AI",
                    date="2026-03-07",
                    config=SAMPLE_EMAIL_CONFIG,
                )
                return MockSMTP

    def test_raises_delivery_error_when_password_missing(self) -> None:
        # Temporarily remove SMTP_PASSWORD from the environment.
        env_without_password = {k: v for k, v in os.environ.items() if k != "SMTP_PASSWORD"}
        with patch.dict(os.environ, env_without_password, clear=True):
            with pytest.raises(DeliveryError, match="SMTP_PASSWORD"):
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)

    def test_raises_delivery_error_when_password_empty_string(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": ""}, clear=False):
            with pytest.raises(DeliveryError, match="SMTP_PASSWORD"):
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)

    def test_connects_to_correct_host_and_port(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)
                MockSMTP.assert_called_once_with("smtp.example.com", 587, timeout=30)

    def test_calls_starttls(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)
                mock_server.starttls.assert_called_once()

    def test_calls_login_with_sender_and_password(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "mysecret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)
                mock_server.login.assert_called_once_with("redpill@example.com", "mysecret")

    def test_calls_sendmail_with_correct_addresses(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)
                args = mock_server.sendmail.call_args
                from_addr = args[0][0]
                to_addrs = args[0][1]
                assert from_addr == "redpill@example.com"
                assert "you@example.com" in to_addrs

    def test_raises_delivery_error_on_auth_failure(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "wrong"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad")
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                with pytest.raises(DeliveryError, match="[Aa]uth"):
                    deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)

    def test_raises_delivery_error_on_connect_error(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                MockSMTP.side_effect = smtplib.SMTPConnectError(421, b"Service unavailable")
                with pytest.raises(DeliveryError, match="[Cc]onnect"):
                    deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)

    def test_raises_delivery_error_on_generic_smtp_exception(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                mock_server.sendmail.side_effect = smtplib.SMTPException("unknown error")
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                with pytest.raises(DeliveryError, match="SMTP"):
                    deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)

    def test_raises_delivery_error_on_os_error(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                MockSMTP.side_effect = OSError("connection refused")
                with pytest.raises(DeliveryError, match="[Nn]etwork"):
                    deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)

    def test_smtp_port_coerced_to_int(self) -> None:
        """smtp_port supplied as a string (e.g. from YAML) must be handled."""
        config = {**SAMPLE_EMAIL_CONFIG, "smtp_port": "587"}
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", config)
                MockSMTP.assert_called_once_with("smtp.example.com", 587, timeout=30)

    def test_returns_none(self) -> None:
        with patch.dict(os.environ, {"SMTP_PASSWORD": "secret"}, clear=False):
            with patch("redpill.deliver.smtplib.SMTP") as MockSMTP:
                mock_server = MagicMock()
                MockSMTP.return_value.__enter__ = lambda s: mock_server
                MockSMTP.return_value.__exit__ = MagicMock(return_value=False)
                result = deliver_email(SAMPLE_DIGEST, "AI", "2026-03-07", SAMPLE_EMAIL_CONFIG)
        assert result is None


# ---------------------------------------------------------------------------
# TestValidateEmailConfig
# ---------------------------------------------------------------------------


class TestValidateEmailConfig:
    def test_valid_config_does_not_raise(self) -> None:
        config = {"email_config": SAMPLE_EMAIL_CONFIG}
        _validate_email_config(config)  # must not raise

    def test_missing_email_config_key_raises(self) -> None:
        with pytest.raises(ValueError, match="email_config"):
            _validate_email_config({"delivery_method": "email"})

    def test_none_email_config_raises(self) -> None:
        with pytest.raises(ValueError, match="email_config"):
            _validate_email_config({"email_config": None})

    def test_missing_smtp_host_raises(self) -> None:
        cfg = {**SAMPLE_EMAIL_CONFIG}
        del cfg["smtp_host"]
        with pytest.raises(ValueError, match="smtp_host"):
            _validate_email_config({"email_config": cfg})

    def test_missing_smtp_port_raises(self) -> None:
        cfg = {**SAMPLE_EMAIL_CONFIG}
        del cfg["smtp_port"]
        with pytest.raises(ValueError, match="smtp_port"):
            _validate_email_config({"email_config": cfg})

    def test_missing_sender_raises(self) -> None:
        cfg = {**SAMPLE_EMAIL_CONFIG}
        del cfg["sender"]
        with pytest.raises(ValueError, match="sender"):
            _validate_email_config({"email_config": cfg})

    def test_missing_recipient_raises(self) -> None:
        cfg = {**SAMPLE_EMAIL_CONFIG}
        del cfg["recipient"]
        with pytest.raises(ValueError, match="recipient"):
            _validate_email_config({"email_config": cfg})

    def test_empty_smtp_host_raises(self) -> None:
        cfg = {**SAMPLE_EMAIL_CONFIG, "smtp_host": ""}
        with pytest.raises(ValueError, match="smtp_host"):
            _validate_email_config({"email_config": cfg})


# ---------------------------------------------------------------------------
# TestDeliver
# ---------------------------------------------------------------------------


class TestDeliver:
    def test_markdown_method_calls_deliver_markdown(self, tmp_path: Path) -> None:
        config = {"delivery_method": "markdown", "output_dir": str(tmp_path)}
        result = deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)
        assert result == tmp_path / "2026-03-07.md"
        assert result.exists()

    def test_markdown_method_returns_path(self, tmp_path: Path) -> None:
        config = {"delivery_method": "markdown", "output_dir": str(tmp_path)}
        result = deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)
        assert isinstance(result, Path)

    def test_markdown_uses_default_output_dir_when_omitted(self, tmp_path: Path) -> None:
        """When output_dir is absent, the dispatcher should default gracefully."""
        config = {"delivery_method": "markdown"}
        with patch("redpill.deliver.deliver_markdown") as mock_md:
            mock_md.return_value = tmp_path / "2026-03-07.md"
            deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)
            mock_md.assert_called_once_with(
                digest=SAMPLE_DIGEST, output_dir="data/digests", date="2026-03-07"
            )

    def test_email_method_returns_none(self) -> None:
        config = {"delivery_method": "email", "email_config": SAMPLE_EMAIL_CONFIG}
        with patch("redpill.deliver.deliver_email") as mock_email:
            result = deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)
        assert result is None

    def test_email_method_calls_deliver_email_with_correct_args(self) -> None:
        config = {"delivery_method": "email", "email_config": SAMPLE_EMAIL_CONFIG}
        with patch("redpill.deliver.deliver_email") as mock_email:
            deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)
            mock_email.assert_called_once_with(
                digest=SAMPLE_DIGEST,
                topic="AI",
                date="2026-03-07",
                config=SAMPLE_EMAIL_CONFIG,
                feedback_base_url="",
            )

    def test_unknown_method_raises_value_error(self) -> None:
        config = {"delivery_method": "carrier_pigeon"}
        with pytest.raises(ValueError, match="unknown delivery_method"):
            deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)

    def test_missing_delivery_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown delivery_method"):
            deliver(SAMPLE_DIGEST, "AI", "2026-03-07", {})

    def test_email_with_missing_email_config_raises_value_error(self) -> None:
        config = {"delivery_method": "email"}
        with pytest.raises(ValueError, match="email_config"):
            deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)

    def test_email_with_incomplete_email_config_raises_value_error(self) -> None:
        config = {
            "delivery_method": "email",
            "email_config": {"smtp_host": "smtp.example.com"},
        }
        with pytest.raises(ValueError):
            deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)

    def test_delivery_error_propagates_from_deliver_markdown(self, tmp_path: Path) -> None:
        config = {"delivery_method": "markdown", "output_dir": str(tmp_path)}
        with patch("redpill.deliver.deliver_markdown") as mock_md:
            mock_md.side_effect = DeliveryError("disk full")
            with pytest.raises(DeliveryError, match="disk full"):
                deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)

    def test_delivery_error_propagates_from_deliver_email(self) -> None:
        config = {"delivery_method": "email", "email_config": SAMPLE_EMAIL_CONFIG}
        with patch("redpill.deliver.deliver_email") as mock_email:
            mock_email.side_effect = DeliveryError("SMTP refused")
            with pytest.raises(DeliveryError, match="SMTP refused"):
                deliver(SAMPLE_DIGEST, "AI", "2026-03-07", config)

    def test_topic_passed_to_deliver_email(self) -> None:
        config = {"delivery_method": "email", "email_config": SAMPLE_EMAIL_CONFIG}
        with patch("redpill.deliver.deliver_email") as mock_email:
            deliver(SAMPLE_DIGEST, "contrastive learning", "2026-03-07", config)
            assert mock_email.call_args.kwargs["topic"] == "contrastive learning"

    def test_date_passed_to_deliver_email(self) -> None:
        config = {"delivery_method": "email", "email_config": SAMPLE_EMAIL_CONFIG}
        with patch("redpill.deliver.deliver_email") as mock_email:
            deliver(SAMPLE_DIGEST, "AI", "2025-11-15", config)
            assert mock_email.call_args.kwargs["date"] == "2025-11-15"

    def test_date_passed_to_deliver_markdown(self, tmp_path: Path) -> None:
        config = {"delivery_method": "markdown", "output_dir": str(tmp_path)}
        with patch("redpill.deliver.deliver_markdown") as mock_md:
            mock_md.return_value = tmp_path / "2025-11-15.md"
            deliver(SAMPLE_DIGEST, "AI", "2025-11-15", config)
            assert mock_md.call_args.kwargs["date"] == "2025-11-15"
