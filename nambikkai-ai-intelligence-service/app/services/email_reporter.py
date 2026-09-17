"""
Email reporter — no-op stub that only sends when fully configured.

This module MUST NOT send any email when:
  - APP_ENV == "test"
  - EMAIL_ENABLED is False (default)
  - Any of SMTP_HOST / SMTP_USER / SMTP_PASSWORD / OWNER_EMAIL are empty

All SMTP credentials and addresses come exclusively from the Settings object
(env vars).  No addresses, credentials, or host names are ever hardcoded.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_email_status() -> dict:
    """Return email service configuration status without exposing credentials."""
    s = get_settings()
    if s.APP_ENV == "test":
        return {
            "status": "TEST_MODE_DISABLED",
            "configured": False,
            "message": "Email sending is disabled in test mode.",
        }
    if not s.EMAIL_ENABLED:
        return {
            "status": "EMAIL_CONFIGURATION_PENDING",
            "configured": False,
            "message": "EMAIL_ENABLED is false. Email reporting disabled by default in development.",
        }
    required = [("SMTP_HOST", s.SMTP_HOST), ("SMTP_USER", s.SMTP_USER),
                ("SMTP_PASSWORD", s.SMTP_PASSWORD), ("OWNER_EMAIL", s.OWNER_EMAIL)]
    missing = [name for name, val in required if not val or not val.strip()]
    if missing:
        return {
            "status": "EMAIL_CONFIGURATION_PENDING",
            "configured": False,
            "missing_fields": missing,
            "message": "SMTP configuration incomplete. Configure via environment variables.",
        }
    return {
        "status": "CONFIGURED",
        "configured": True,
        "smtp_host": s.SMTP_HOST,
        "smtp_port": s.SMTP_PORT,
    }


def _is_sending_allowed() -> bool:
    """Return True only when email is fully configured and not in test mode."""
    status = get_email_status()
    return status.get("configured", False)


def send_scan_report(summaries: List[dict]) -> bool:
    """
    Send a bulk-scan summary email.

    Parameters
    ----------
    summaries:
        List of dicts, each containing at least 'platform', 'actionable',
        'scanned', 'xgboost_status'.

    Returns
    -------
    True if email was sent, False if skipped or failed.
    """
    if not _is_sending_allowed():
        logger.info("[EmailReporter] Sending skipped: EMAIL_CONFIGURATION_PENDING (unconfigured or disabled)")
        return False

    s = get_settings()
    try:
        subject = "Nambikkai AI — Daily Scan Report"
        body_lines = ["<h2>Nambikkai AI Daily Scan Report</h2>", "<table border='1' cellpadding='4'>"]
        body_lines.append(
            "<tr><th>Platform</th><th>Scanned</th><th>Actionable</th>"
            "<th>Errors</th><th>XGBoost Status</th></tr>"
        )
        for smry in summaries:
            body_lines.append(
                f"<tr>"
                f"<td>{smry.get('platform','')}</td>"
                f"<td>{smry.get('scanned',0)}</td>"
                f"<td>{smry.get('actionable',0)}</td>"
                f"<td>{smry.get('errors',0)}</td>"
                f"<td>{smry.get('xgboost_status','')}</td>"
                f"</tr>"
            )
        body_lines.append("</table>")
        body = "\n".join(body_lines)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = s.SMTP_USER
        msg["To"] = s.OWNER_EMAIL
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(s.SMTP_HOST, s.SMTP_PORT) as server:
            server.starttls()
            server.login(s.SMTP_USER, s.SMTP_PASSWORD)
            server.sendmail(s.SMTP_USER, [s.OWNER_EMAIL], msg.as_string())

        logger.info("[EmailReporter] Scan report sent to configured owner email.")
        return True
    except Exception as exc:
        logger.error("[EmailReporter] Failed to send email: %s", exc)
        return False

