"""Tests for messages.py and message_tools.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mindloop.messages import (
    count_new,
    list_messages,
    parse_filename_date,
    parse_message,
    write_message,
)
from mindloop.message_tools import MessageTools


def _write_raw(path: Path, sender: str, date: str, title: str, body: str) -> None:
    """Write a message file with an explicit date string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"From: {sender}\nDate: {date}\nTitle: {title}\n\n{body}\n")


# --- messages.py ---


def test_parse_filename_date() -> None:
    dt = parse_filename_date("20250213_143022.txt")
    assert dt == datetime(2025, 2, 13, 14, 30, 22)


def test_parse_filename_date_with_prefix() -> None:
    dt = parse_filename_date("001_20250213_143022.txt")
    assert dt == datetime(2025, 2, 13, 14, 30, 22)


def test_parse_filename_date_no_match() -> None:
    assert parse_filename_date("readme.txt") is None


def test_parse_message(tmp_path: Path) -> None:
    path = tmp_path / "msg.txt"
    _write_raw(path, "Alice", "2025-02-13 14:30:22", "Question about X", "Hello there")
    msg = parse_message(path)
    assert msg.sender == "Alice"
    assert msg.date == "2025-02-13 14:30:22"
    assert msg.title == "Question about X"
    assert msg.body == "Hello there"
    assert msg.path == path


def test_write_message(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    write_message(path, sender="Bob", title="Reply", body="Thanks!")
    msg = parse_message(path)
    assert msg.sender == "Bob"
    assert msg.title == "Reply"
    assert msg.body == "Thanks!"
    # Date should be parseable.
    datetime.strptime(msg.date, "%Y-%m-%d %H:%M:%S")


def test_list_messages_sorted(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_raw(inbox / "20250213_100000.txt", "B", "d", "Second", "b")
    _write_raw(inbox / "20250212_090000.txt", "A", "d", "First", "a")
    _write_raw(inbox / "20250214_110000.txt", "C", "d", "Third", "c")
    msgs = list_messages(inbox)
    assert [m.sender for m in msgs] == ["A", "B", "C"]


def test_list_messages_before_filter(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_raw(inbox / "20250212_090000.txt", "A", "d", "Old", "a")
    _write_raw(inbox / "20250213_100000.txt", "B", "d", "Current", "b")
    _write_raw(inbox / "20250214_110000.txt", "C", "d", "Future", "c")
    before = datetime(2025, 2, 13, 10, 0, 0)
    msgs = list_messages(inbox, before=before)
    assert len(msgs) == 1
    assert msgs[0].sender == "A"


def test_list_messages_empty_dir(tmp_path: Path) -> None:
    assert list_messages(tmp_path / "nonexistent") == []


def test_count_new(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_raw(inbox / "20250212_090000.txt", "A", "d", "Old", "a")
    _write_raw(inbox / "20250213_100000.txt", "B", "d", "New", "b")
    msgs = list_messages(inbox)
    since = datetime(2025, 2, 12, 12, 0, 0)
    assert count_new(msgs, since) == 1


def test_count_new_no_previous(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_raw(inbox / "20250212_090000.txt", "A", "d", "t", "a")
    _write_raw(inbox / "20250213_100000.txt", "B", "d", "t", "b")
    msgs = list_messages(inbox)
    assert count_new(msgs, None) == 2


# --- message_tools.py ---


def _make_tools(
    tmp_path: Path,
    filenames: list[str] | None = None,
    since: datetime | None = None,
    before: datetime | None = None,
) -> MessageTools:
    """Create a MessageTools with some inbox messages."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    inbox.mkdir(exist_ok=True)
    outbox.mkdir(exist_ok=True)
    for fn in filenames or []:
        _write_raw(inbox / fn, "Alice", "2025-02-13", f"Re: {fn}", f"Body of {fn}")
    return MessageTools(
        inbox_dir=inbox,
        outbox_dir=outbox,
        instance=1,
        before=before,
        since=since,
    )


def test_message_list_tool(tmp_path: Path) -> None:
    mt = _make_tools(
        tmp_path,
        filenames=["20250212_090000.txt", "20250213_100000.txt"],
        since=datetime(2025, 2, 12, 12, 0, 0),
    )
    result = mt.message_list()
    assert "#2 (new)" in result
    assert "#1" in result
    assert "(new)" not in result.split("\n")[1]  # First msg is old.


def test_message_list_empty(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path)
    assert mt.message_list() == "No inbox messages."


def test_message_read_tool(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path, filenames=["20250213_100000.txt"])
    result = mt.message_read(id=1)
    assert "From: Alice" in result
    assert "Body of 20250213_100000.txt" in result


def test_message_read_invalid_id(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path, filenames=["20250213_100000.txt"])
    result = mt.message_read(id=0)
    assert "Error" in result
    result = mt.message_read(id=99)
    assert "Error" in result


def test_message_send_tool(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path)
    result = mt.message_send(to="Bob", title="Hello", text="Hi Bob!")
    assert "Message sent to Bob" in result
    outbox = tmp_path / "outbox"
    files = list(outbox.glob("*.txt"))
    assert len(files) == 1
    msg = parse_message(files[0])
    assert msg.sender == "Agent"
    assert msg.title == "Hello"
    assert "Hi Bob!" in msg.body
    # Filename should start with instance number.
    assert files[0].name.startswith("001_")


def test_message_list_outbox(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path)
    assert mt.message_list(box="outbox") == "No outbox messages."
    mt.message_send(to="Bob", title="First", text="Hello")
    mt.message_send(to="Carol", title="Second", text="Hi")
    result = mt.message_list(box="outbox")
    lines = result.strip().split("\n")
    assert len(lines) == 2
    # Newest first.
    assert '"Second"' in lines[0]
    assert '"First"' in lines[1]


def test_message_read_outbox(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path)
    mt.message_send(to="Bob", title="Hello", text="Hi Bob!")
    result = mt.message_read(id=1, box="outbox")
    assert "From: Agent" in result
    assert "Hi Bob!" in result


def test_message_list_invalid_box(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path)
    result = mt.message_list(box="junk")
    assert "Error" in result
    assert "'inbox' or 'outbox'" in result


def test_message_read_invalid_box(tmp_path: Path) -> None:
    mt = _make_tools(tmp_path)
    result = mt.message_read(id=1, box="junk")
    assert "Error" in result
    assert "'inbox' or 'outbox'" in result


def test_message_list_pagination(tmp_path: Path) -> None:
    fns = [f"202502{i:02d}_120000.txt" for i in range(1, 6)]
    mt = _make_tools(tmp_path, filenames=fns)
    result = mt.message_list(count=2, starting=0)
    lines = result.strip().split("\n")
    # 2 messages + "... N more" line.
    assert len(lines) == 3
    assert "more" in lines[-1]
