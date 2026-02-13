"""Message parsing, writing, and listing for inbox/outbox."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Matches YYYYMMDD_HHMMSS anywhere in a filename.
_TS_RE = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


@dataclass
class Message:
    """A parsed inbox or outbox message."""

    sender: str
    date: str
    title: str
    body: str
    path: Path


def parse_filename_date(name: str) -> datetime | None:
    """Parse YYYYMMDD_HHMMSS from a filename."""
    m = _TS_RE.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, s)


def parse_message(path: Path) -> Message:
    """Parse an email-format message file."""
    text = path.read_text()
    lines = text.split("\n")
    headers: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()
    body = "\n".join(lines[body_start:]).strip()
    return Message(
        sender=headers.get("From", ""),
        date=headers.get("Date", ""),
        title=headers.get("Title", ""),
        body=body,
        path=path,
    )


def write_message(path: Path, sender: str, title: str, body: str) -> None:
    """Write a message in email format with Date set to now."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"From: {sender}\nDate: {now}\nTitle: {title}\n\n{body}\n")


def list_messages(inbox_dir: Path, before: datetime | None = None) -> list[Message]:
    """List .txt files sorted by filename (oldest first), filtered by date.

    Only messages with filename-date strictly before *before* are included.
    If *before* is None, all messages are returned.
    """
    if not inbox_dir.is_dir():
        return []
    files = sorted(inbox_dir.glob("*.txt"))
    result: list[Message] = []
    for f in files:
        if before is not None:
            ts = parse_filename_date(f.name)
            if ts is not None and ts >= before:
                continue
        result.append(parse_message(f))
    return result


def count_new(messages: list[Message], since: datetime | None) -> int:
    """Count messages whose filename-date is after *since*.

    If *since* is None, all messages are new.
    """
    if since is None:
        return len(messages)
    count = 0
    for msg in messages:
        ts = parse_filename_date(msg.path.name)
        if ts is not None and ts > since:
            count += 1
    return count
