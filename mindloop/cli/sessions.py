"""List sessions and their metadata."""

import argparse
import re
from pathlib import Path

from mindloop.cli.agent import _session_exit_reason

_SESSIONS_DIR = Path("sessions")

# Matches both NNN_agent_YYYYMMDD_HHMMSS.jsonl and agent_YYYYMMDD_HHMMSS.jsonl.
_TS_RE = re.compile(r"agent_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.jsonl$")


def _parse_timestamp(name: str) -> str | None:
    """Extract a human-readable timestamp from a JSONL filename."""
    m = _TS_RE.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}:{s}"


def _exit_label(jsonl_path: Path) -> str:
    """Return a short status label from the exit reason."""
    reason = _session_exit_reason(jsonl_path)
    if reason is None:
        return "clean"
    if "abruptly" in reason:
        return "crashed"
    if "tokens" in reason:
        return "tokens"
    if "iteration" in reason:
        return "iterations"
    return f"unknown: {reason}"


def _gather_sessions(sessions_dir: Path) -> list[dict[str, str]]:
    """Collect metadata for each session directory."""
    rows: list[dict[str, str]] = []
    for entry in sessions_dir.iterdir():
        log_dir = entry / "logs"
        if not log_dir.is_dir():
            continue
        logs = sorted(log_dir.glob("*agent_*.jsonl"))
        count = len(logs)
        if count == 0:
            continue

        started = _parse_timestamp(logs[0].name) or "?"
        last_run = _parse_timestamp(logs[-1].name) or "?"
        status = _exit_label(logs[-1])
        has_notes = "yes" if (entry / "workspace" / "_notes.md").is_file() else "no"

        rows.append(
            {
                "session": entry.name,
                "instances": str(count),
                "started": started,
                "last_run": last_run,
                "status": status,
                "notes": has_notes,
            }
        )
    rows.sort(key=lambda r: r["last_run"])
    return rows


def _print_table(rows: list[dict[str, str]]) -> None:
    """Print rows as a formatted table."""
    headers = {
        "session": "Session",
        "instances": "Instances",
        "started": "Started",
        "last_run": "Last run",
        "status": "Status",
        "notes": "Notes",
    }
    widths = {k: len(v) for k, v in headers.items()}
    for row in rows:
        for k, v in row.items():
            widths[k] = max(widths[k], len(v))

    fmt = "  ".join(f"{{:<{widths[k]}}}" for k in headers)
    print(fmt.format(*headers.values()))
    for row in rows:
        print(fmt.format(*(row[k] for k in headers)))


def main() -> None:
    parser = argparse.ArgumentParser(description="List sessions and metadata.")
    parser.add_argument(
        "--dir",
        default=str(_SESSIONS_DIR),
        help=f"Sessions directory (default: {_SESSIONS_DIR}).",
    )
    args = parser.parse_args()

    sessions_dir = Path(args.dir)
    if not sessions_dir.is_dir():
        print(f"Sessions directory not found: {sessions_dir}")
        return

    rows = _gather_sessions(sessions_dir)
    if not rows:
        print("No sessions found.")
        return

    _print_table(rows)


if __name__ == "__main__":
    main()
