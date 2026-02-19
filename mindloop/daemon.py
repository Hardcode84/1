"""Session daemon: run an agent session on a cron schedule."""

import logging
import signal
import subprocess
import sys
import threading
from datetime import datetime

from croniter import croniter

log = logging.getLogger(__name__)


def _build_cmd(
    session: str,
    model: str | None,
    summarizer_model: str | None,
    n_experts: int,
) -> list[str]:
    """Build the mindloop-agent command."""
    cmd = [sys.executable, "-m", "mindloop.cli.agent", "--session", session]
    if model:
        cmd += ["--model", model]
    if summarizer_model:
        cmd += ["--summarizer-model", summarizer_model]
    if n_experts > 1:
        cmd += ["--n-experts", str(n_experts)]
    return cmd


def _next_time(schedule: str) -> datetime:
    """Compute the next run time from a cron expression."""
    return croniter(schedule, datetime.now()).get_next(datetime)


def _run_once(cmd: list[str], shutdown: threading.Event) -> int:
    """Run one agent session. Returns exit code."""
    log.info("Starting session instance...")
    start = datetime.now()
    try:
        proc = subprocess.run(cmd)
        code = proc.returncode
    except Exception:
        log.exception("Session crashed with exception.")
        code = 1
    elapsed = datetime.now() - start
    log.info("Session finished (exit code %d, %s).", code, elapsed)
    return code


_DEFAULT_MAX_FAILURES = 5


def run_daemon(
    session: str,
    schedule: str,
    *,
    model: str | None = None,
    summarizer_model: str | None = None,
    n_experts: int = 1,
    run_now: bool = False,
    max_failures: int = _DEFAULT_MAX_FAILURES,
) -> None:
    """Main daemon loop. Blocks until SIGINT/SIGTERM."""
    # Validate cron expression early.
    croniter(schedule)

    shutdown = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: shutdown.set())

    cmd = _build_cmd(session, model, summarizer_model, n_experts)
    log.info("Daemon started for session '%s', schedule: '%s'", session, schedule)
    log.info("Command: %s", " ".join(cmd))

    consecutive_failures = 0

    if run_now:
        log.info("Running immediately (--run-now)...")
        code = _run_once(cmd, shutdown)
        if code != 0:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

    while not shutdown.is_set():
        if consecutive_failures >= max_failures:
            log.error("Stopping after %d consecutive failures.", consecutive_failures)
            break
        nxt = _next_time(schedule)
        log.info("Next run: %s", nxt.strftime("%Y-%m-%d %H:%M:%S"))
        wait_secs = (nxt - datetime.now()).total_seconds()
        if wait_secs > 0:
            log.debug("Sleeping %.1f seconds...", wait_secs)
            if shutdown.wait(timeout=wait_secs):
                break
        if shutdown.is_set():
            break
        log.info("Triggering scheduled run.")
        code = _run_once(cmd, shutdown)
        if code != 0:
            consecutive_failures += 1
            log.warning(
                "Consecutive failures: %d / %d.",
                consecutive_failures,
                max_failures,
            )
        else:
            consecutive_failures = 0

    log.info("Daemon shutting down.")
