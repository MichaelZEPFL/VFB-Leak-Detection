"""
Structured JSONL logging utilities.

Writes one JSON object per line to logs/events.jsonl.

Notes:
- Avoid logging secrets. This module includes basic redaction for keys that look sensitive.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


SENSITIVE_KEY_PATTERNS = ("password", "secret", "webhook", "token", "key")


def _looks_sensitive(key: str) -> bool:
    k = key.lower()
    return any(pat in k for pat in SENSITIVE_KEY_PATTERNS)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _looks_sensitive(str(k)):
                out[k] = "[REDACTED]"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


class JsonlLogger:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        event = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "type": event_type,
            **fields,
        }
        safe_event = _redact(event)
        line = json.dumps(safe_event, ensure_ascii=False)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
