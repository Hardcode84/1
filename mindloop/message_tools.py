"""Agent-facing message tools: message_list, message_read, message_send."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from mindloop.messages import (
    Message,
    count_new,
    list_messages,
    parse_filename_date,
    parse_message,
    write_message,
)


class MessageTools:
    """Inbox/outbox tools bound to a session's message directories."""

    def __init__(
        self,
        inbox_dir: Path,
        outbox_dir: Path,
        instance: int,
        before: datetime | None = None,
        since: datetime | None = None,
        stats: dict[Any, Any] | None = None,
    ) -> None:
        self._inbox_dir = inbox_dir
        self._outbox_dir = outbox_dir
        self._instance = instance
        self._since = since
        # Freeze inbox list at init.
        self._messages = list_messages(inbox_dir, before=before)
        self._stats = stats if stats is not None else {}

    @property
    def new_count(self) -> int:
        """Number of new messages since previous instance."""
        return count_new(self._messages, self._since)

    @property
    def new_message_note(self) -> str:
        """Human-readable note about new messages."""
        n = self.new_count
        return (
            f"You have {n} new message{'s' if n != 1 else ''} since last instance. "
            f"Use message_list to see them."
        )

    def _track(self, tool: str) -> None:
        """Increment call count for a message tool."""
        counts = self._stats.setdefault("messages", {})
        counts[tool] = counts.get(tool, 0) + 1

    def _is_new(self, msg: Message) -> bool:
        """Check if a message is new (after previous instance start)."""
        if self._since is None:
            return True
        ts = parse_filename_date(msg.path.name)
        return ts is not None and ts > self._since

    def _outbox_messages(self) -> list[Message]:
        """Read outbox live (not frozen) so agent sees its own sends."""
        if not self._outbox_dir.is_dir():
            return []
        return [parse_message(f) for f in sorted(self._outbox_dir.glob("*.txt"))]

    _VALID_BOXES = ("inbox", "outbox")

    def _get_messages(self, box: str) -> list[Message] | str:
        """Return message list for the given box, or error string."""
        if box not in self._VALID_BOXES:
            return f"Error: box must be 'inbox' or 'outbox', got '{box}'."
        if box == "outbox":
            return self._outbox_messages()
        return self._messages

    def message_list(
        self, box: str = "inbox", count: int = 10, starting: int = 0
    ) -> str:
        """List messages, newest first, paginated."""
        self._track("message_list")
        result = self._get_messages(box)
        if isinstance(result, str):
            return result
        messages = result
        total = len(messages)
        if total == 0:
            return f"No {box} messages."
        # Reverse for newest-first display, then paginate.
        reversed_msgs = list(reversed(list(enumerate(messages, 1))))
        page = reversed_msgs[starting : starting + count]
        lines: list[str] = []
        for idx, msg in page:
            new = " (new)" if box == "inbox" and self._is_new(msg) else ""
            lines.append(f'#{idx}{new} {msg.date} "{msg.sender}", "{msg.title}"')
        remaining = total - starting - len(page)
        if remaining > 0:
            lines.append(f"... {remaining} more (use starting={starting + count})")
        return "\n".join(lines)

    def message_read(self, id: int, box: str = "inbox") -> str:
        """Return full message content by 1-based ID."""
        self._track("message_read")
        result = self._get_messages(box)
        if isinstance(result, str):
            return result
        messages = result
        if id < 1 or id > len(messages):
            return (
                f"Error: invalid {box} message id {id}. Valid range: 1-{len(messages)}."
            )
        msg = messages[id - 1]
        return (
            f"From: {msg.sender}\n"
            f"Date: {msg.date}\n"
            f"Title: {msg.title}\n\n"
            f"{msg.body}"
        )

    def message_send(self, to: str, title: str, text: str) -> str:
        """Write a message to the outbox."""
        self._track("message_send")
        self._outbox_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self._instance:03d}_{ts}"
        path = self._outbox_dir / f"{base}.txt"
        # Avoid collision if multiple sends within the same second.
        seq = 1
        while path.exists():
            path = self._outbox_dir / f"{base}_{seq}.txt"
            seq += 1
        write_message(path, sender="Agent", title=title, body=f"To: {to}\n\n{text}")
        return f"Message sent to {to}: {path.name}"
