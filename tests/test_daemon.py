"""Tests for mindloop.daemon."""

import sys
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from mindloop.daemon import _build_cmd, _next_time, _run_once, run_daemon


def test_build_cmd_minimal() -> None:
    """Base command with only session name."""
    cmd = _build_cmd("mysession", None, None, 1, None)
    assert cmd == [sys.executable, "-m", "mindloop.cli.agent", "--session", "mysession"]


def test_build_cmd_full() -> None:
    """All flags included when options are set."""
    cmd = _build_cmd(
        "s1", "deepseek/deepseek-v3.2", "meta-llama/llama-3", 4, 50000, allow_exec=True
    )
    assert "--model" in cmd
    assert "deepseek/deepseek-v3.2" in cmd
    assert "--summarizer-model" in cmd
    assert "meta-llama/llama-3" in cmd
    assert "--n-experts" in cmd
    assert "4" in cmd
    assert "--max-tokens" in cmd
    assert "50000" in cmd
    assert "--allow-exec" in cmd


def test_build_cmd_no_allow_exec() -> None:
    """--allow-exec is absent when not requested."""
    cmd = _build_cmd("s1", None, None, 1, None)
    assert "--allow-exec" not in cmd


def test_next_time() -> None:
    """Next trigger for '* * * * *' should be within 60 seconds."""
    nxt = _next_time("* * * * *")
    delta = nxt - datetime.now()
    assert timedelta(0) < delta <= timedelta(seconds=60)


def test_invalid_schedule() -> None:
    """Invalid cron expression raises on startup."""
    with pytest.raises((ValueError, KeyError)):
        run_daemon("test", "bad-cron-expression")


def test_run_once() -> None:
    """Successful subprocess returns exit code 0."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    shutdown = threading.Event()
    with patch("mindloop.daemon.subprocess.run", return_value=mock_result) as mock_run:
        code = _run_once(["echo", "hello"], shutdown)
    assert code == 0
    mock_run.assert_called_once_with(["echo", "hello"])


def test_run_once_crash() -> None:
    """Subprocess exception returns exit code 1 (doesn't crash daemon)."""
    shutdown = threading.Event()
    with patch("mindloop.daemon.subprocess.run", side_effect=OSError("boom")):
        code = _run_once(["bad"], shutdown)
    assert code == 1


def test_daemon_run_now() -> None:
    """With --run-now, _run_once is called before scheduling."""
    calls: list[str] = []

    def fake_run_once(cmd: list[str], shutdown: threading.Event) -> int:
        calls.append("run")
        # Shut down after first run so daemon exits.
        shutdown.set()
        return 0

    with patch("mindloop.daemon._run_once", side_effect=fake_run_once):
        run_daemon("test", "0 9 * * *", run_now=True)

    assert calls == ["run"]


def test_daemon_shutdown_during_sleep() -> None:
    """Daemon exits cleanly when shutdown fires during sleep."""
    runs: list[str] = []

    def fake_run_once(cmd: list[str], shutdown: threading.Event) -> int:
        runs.append("run")
        return 0

    def start_and_stop() -> None:
        # Give daemon time to enter the wait, then signal shutdown.
        import time

        time.sleep(0.1)
        # Signal is handled by the daemon's signal handler, but in tests
        # we need to set the event directly. We patch _run_once to track
        # calls, and use _next_time to return a far-future time so the
        # daemon is sleeping when we fire shutdown.
        stop_event.set()

    stop_event = threading.Event()

    def patched_signal(sig: int, handler: object) -> None:
        pass  # Don't install real signal handlers in tests.

    far_future = datetime.now() + timedelta(hours=1)

    with (
        patch("mindloop.daemon._run_once", side_effect=fake_run_once),
        patch("mindloop.daemon._next_time", return_value=far_future),
        patch("mindloop.daemon.signal.signal", side_effect=patched_signal),
    ):
        # Replace the shutdown event inside run_daemon.
        def make_event() -> threading.Event:
            return stop_event

        with patch("mindloop.daemon.threading.Event", side_effect=make_event):
            stopper = threading.Thread(target=start_and_stop)
            stopper.start()
            run_daemon("test", "0 0 1 1 *")  # Far-future schedule.
            stopper.join(timeout=5)

    # No sessions should have run.
    assert runs == []


def test_daemon_stops_after_max_failures() -> None:
    """Daemon exits after consecutive failures hit the limit."""
    calls: list[int] = []

    def fake_run_once(cmd: list[str], shutdown: threading.Event) -> int:
        calls.append(1)
        return 1  # Always fail.

    near = datetime.now() + timedelta(seconds=0.01)
    with (
        patch("mindloop.daemon._run_once", side_effect=fake_run_once),
        patch("mindloop.daemon._next_time", return_value=near),
        patch("mindloop.daemon.signal.signal"),
    ):
        run_daemon("test", "* * * * *", max_failures=3)

    assert len(calls) == 3


def test_daemon_resets_failures_on_success() -> None:
    """A successful run resets the consecutive failure counter."""
    codes = iter([1, 1, 0, 1, 1, 0])
    calls: list[int] = []

    def fake_run_once(cmd: list[str], shutdown: threading.Event) -> int:
        code = next(codes, None)
        if code is None:
            shutdown.set()
            return 0
        calls.append(code)
        return code

    near = datetime.now() + timedelta(seconds=0.01)
    with (
        patch("mindloop.daemon._run_once", side_effect=fake_run_once),
        patch("mindloop.daemon._next_time", return_value=near),
        patch("mindloop.daemon.signal.signal"),
    ):
        run_daemon("test", "* * * * *", max_failures=3)

    # 6 calls: fail, fail, success, fail, fail, success — never hits 3.
    assert calls == [1, 1, 0, 1, 1, 0]
