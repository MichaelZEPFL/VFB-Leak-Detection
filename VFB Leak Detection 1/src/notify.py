"""
Notification utilities: Slack webhook + optional SMTP email.

Slack is preferred if SLACK_WEBHOOK_URL env var is set, or if config notify.slack.webhook_url is present.
"""
from __future__ import annotations

import json
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


@dataclass
class SlackConfig:
    enabled: bool
    webhook_url: str


@dataclass
class EmailConfig:
    enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_addr: str
    to_addrs: List[str]
    use_tls: bool = True


class Notifier:
    def __init__(self, slack: SlackConfig, email: EmailConfig) -> None:
        self.slack = slack
        self.email = email

    def notify(self, title: str, message: str) -> None:
        """
        Best-effort: try Slack first if configured, else email if enabled.
        Failures should not crash the monitoring loop.
        """
        if self.slack.enabled and self.slack.webhook_url:
            try:
                self._send_slack(title, message)
                return
            except Exception:
                # Fall back to email if configured.
                pass

        if self.email.enabled:
            try:
                self._send_email(title, message)
            except Exception:
                pass

    def _send_slack(self, title: str, message: str) -> None:
        payload = {
            "text": f"*{title}*\n{message}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.slack.webhook_url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            _ = resp.read()

    def _send_email(self, subject: str, body: str) -> None:
        if not self.email.from_addr:
            raise ValueError("Email from_addr is empty in config.")
        if not self.email.to_addrs:
            raise ValueError("Email to_addrs is empty in config.")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.email.from_addr
        msg["To"] = ", ".join(self.email.to_addrs)
        msg.set_content(body)

        with smtplib.SMTP(self.email.smtp_host, self.email.smtp_port, timeout=15) as server:
            if self.email.use_tls:
                server.starttls()
            if self.email.smtp_user:
                server.login(self.email.smtp_user, self.email.smtp_password)
            server.send_message(msg)
