"""Gmail SMTP alerter for critical system events.

Reads credentials from two env vars:
  - GMAIL_USER: the sending address (e.g. you@gmail.com)
  - GMAIL_APP_PASSWORD: a 16-char app password from Google's security page
    (NOT your regular Gmail password — you must have 2FA enabled and
    create an app-specific password)

If either env var is missing, the service silently no-ops — the system
keeps running without alerts. This lets paper deployments skip the setup.

Designed to fail gracefully: a failed email never raises up to the caller.
Logs the failure and returns False so the scanner can keep going.
"""
import os
import smtplib
import asyncio
from email.message import EmailMessage
from loguru import logger


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SUBJECT_PREFIX = "[Trading Agent Intraday]"


class EmailService:
    def __init__(self):
        self.user = os.getenv("GMAIL_USER", "").strip()
        self.password = os.getenv("GMAIL_APP_PASSWORD", "").strip().replace(" ", "")
        self.recipient = self.user  # send to self
        self.enabled = bool(self.user and self.password)
        if not self.enabled:
            logger.info("EmailService disabled (GMAIL_USER / GMAIL_APP_PASSWORD not set)")

    def _send_sync(self, subject: str, body: str) -> bool:
        msg = EmailMessage()
        msg["Subject"] = f"{SUBJECT_PREFIX} {subject}"
        msg["From"] = self.user
        msg["To"] = self.recipient
        msg.set_content(body)
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
            logger.info(f"Alert email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed ({subject}): {type(e).__name__}: {e}")
            return False

    async def send_alert(self, subject: str, body: str) -> bool:
        if not self.enabled:
            logger.debug(f"Email skipped ({subject}): service disabled")
            return False
        return await asyncio.to_thread(self._send_sync, subject, body)
