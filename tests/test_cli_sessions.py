"""Tests for mindloop.cli.sessions."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mindloop.cli.sessions import _gather_sessions, main


def _make_log(log_dir: Path, name: str, stop_content: str | None) -> None:
    """Create a minimal JSONL log file."""
    lines = [json.dumps({"role": "assistant", "content": "hello"})]
    if stop_content is not None:
        lines.append(json.dumps({"role": "system", "content": stop_content}))
    (log_dir / name).write_text("\n".join(lines) + "\n")


def _make_session(
    sessions_dir: Path,
    name: str,
    logs: list[tuple[str, str | None]],
    *,
    notes: bool = False,
) -> None:
    """Create a fake session with logs and optionally notes."""
    log_dir = sessions_dir / name / "logs"
    log_dir.mkdir(parents=True)
    for fname, stop in logs:
        _make_log(log_dir, fname, stop)
    ws = sessions_dir / name / "workspace"
    ws.mkdir(exist_ok=True)
    if notes:
        (ws / "_notes.md").write_text("some notes")


def test_gather_basic(tmp_path: Path) -> None:
    """Collects session metadata correctly."""
    _make_session(
        tmp_path,
        "alpha",
        [
            ("001_agent_20260115_100000.jsonl", "[stop] model finished (1k)"),
            ("002_agent_20260213_143000.jsonl", "[stop] model finished (2k)"),
        ],
        notes=True,
    )
    rows = _gather_sessions(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "alpha"
    assert row["instances"] == "2"
    assert row["started"] == "2026-01-15 10:00:00"
    assert row["last_run"] == "2026-02-13 14:30:00"
    assert row["status"] == "clean"
    assert row["notes"] == "yes"


def test_gather_no_prefix(tmp_path: Path) -> None:
    """Logs without NNN_ prefix are still picked up."""
    _make_session(
        tmp_path,
        "old_style",
        [("agent_20260212_110304.jsonl", "[stop] model finished (1k)")],
    )
    rows = _gather_sessions(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session"] == "old_style"
    assert rows[0]["started"] == "2026-02-12 11:03:04"
    assert rows[0]["status"] == "clean"


def test_gather_crashed(tmp_path: Path) -> None:
    """Crashed session (no [stop] line) shows crashed status."""
    _make_session(
        tmp_path,
        "broken",
        [("001_agent_20260212_090000.jsonl", None)],
    )
    rows = _gather_sessions(tmp_path)
    assert rows[0]["status"] == "crashed"
    assert rows[0]["notes"] == "no"


def test_gather_tokens(tmp_path: Path) -> None:
    """Token budget exceeded shows tokens status."""
    _make_session(
        tmp_path,
        "budget",
        [("001_agent_20260212_090000.jsonl", "[stop] token budget exceeded (50k)")],
    )
    rows = _gather_sessions(tmp_path)
    assert rows[0]["status"] == "tokens"


def test_gather_iterations(tmp_path: Path) -> None:
    """Max iterations shows iterations status."""
    _make_session(
        tmp_path,
        "loops",
        [("001_agent_20260212_090000.jsonl", "[stop] max iterations reached (200)")],
    )
    rows = _gather_sessions(tmp_path)
    assert rows[0]["status"] == "iterations"


def test_gather_unknown_reason(tmp_path: Path) -> None:
    """Unrecognized stop reason includes the raw message."""
    _make_session(
        tmp_path,
        "weird",
        [("001_agent_20260212_090000.jsonl", "[stop] solar flare detected")],
    )
    rows = _gather_sessions(tmp_path)
    assert rows[0]["status"].startswith("unknown")
    assert "solar flare" in rows[0]["status"]


def test_gather_sorted_by_last_run(tmp_path: Path) -> None:
    """Sessions are sorted by last run timestamp."""
    _make_session(
        tmp_path,
        "newer",
        [("001_agent_20260213_120000.jsonl", "[stop] model finished (1k)")],
    )
    _make_session(
        tmp_path,
        "older",
        [("001_agent_20260211_080000.jsonl", "[stop] model finished (1k)")],
    )
    rows = _gather_sessions(tmp_path)
    assert [r["session"] for r in rows] == ["older", "newer"]


def test_gather_skips_non_session(tmp_path: Path) -> None:
    """Directories without logs/ are skipped."""
    (tmp_path / "random_dir").mkdir()
    (tmp_path / "also_not_a_session" / "workspace").mkdir(parents=True)
    rows = _gather_sessions(tmp_path)
    assert rows == []


def test_gather_skips_empty_logs(tmp_path: Path) -> None:
    """Sessions with logs/ but no JSONL files are skipped."""
    (tmp_path / "empty" / "logs").mkdir(parents=True)
    rows = _gather_sessions(tmp_path)
    assert rows == []


def test_main_empty_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Main prints message when no sessions exist."""
    with patch("sys.argv", ["mindloop-sessions", "--dir", str(tmp_path)]):
        main()
    captured = capsys.readouterr()
    assert "No sessions found" in captured.out


def test_main_missing_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Main prints message when sessions dir does not exist."""
    missing = tmp_path / "nonexistent"
    with patch("sys.argv", ["mindloop-sessions", "--dir", str(missing)]):
        main()
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_main_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Main outputs a formatted table."""
    _make_session(
        tmp_path,
        "demo",
        [("001_agent_20260213_120000.jsonl", "[stop] model finished (1k)")],
    )
    with patch("sys.argv", ["mindloop-sessions", "--dir", str(tmp_path)]):
        main()
    captured = capsys.readouterr()
    assert "Session" in captured.out
    assert "demo" in captured.out
    assert "clean" in captured.out
