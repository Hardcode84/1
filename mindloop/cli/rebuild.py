"""Rebuild memory DB from JSONL session logs using current extraction logic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mindloop.client import API_KEY, DEFAULT_MODEL
from mindloop.extractor import extract_window
from mindloop.memory import MemoryStore
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunk
from mindloop.util import DEDUP_THRESHOLD, noop

# Fallback window size when no boundary markers are found.
_FALLBACK_WINDOW = 20

# Boundary marker strings.
_NUDGE_MARKER = "Continue autonomously"
_REFLECT_MARKER = "You've been using tools for a while"


def _load_messages(path: Path) -> list[dict[str, object]]:
    """Load messages from a JSONL log, stripping log-only fields."""
    messages: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        entry.pop("timestamp", None)
        entry.pop("usage", None)
        messages.append(entry)
    return messages


def _is_boundary(msg: dict[str, object]) -> bool:
    """Check if a message is a nudge/reflect extraction boundary."""
    content = str(msg.get("content", ""))
    role = msg.get("role", "")
    if role == "user" and _NUDGE_MARKER in content:
        return True
    if role == "system" and _REFLECT_MARKER in content:
        return True
    return False


def _has_boundaries(messages: list[dict[str, object]]) -> bool:
    """Check if log contains any nudge/reflect markers."""
    return any(_is_boundary(m) for m in messages)


def _parse_remember_calls(
    msg: dict[str, object],
) -> list[tuple[str, str]]:
    """Extract (text, abstract) pairs from remember tool calls in a message."""
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    results: list[tuple[str, str]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        if not isinstance(func, dict):
            continue
        if func.get("name") != "remember":
            continue
        raw_args = func.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            continue
        text = args.get("text", "")
        abstract = args.get("abstract", "")
        if text and abstract:
            results.append((text, abstract))
    return results


def _save_remember(
    store: MemoryStore,
    text: str,
    abstract: str,
    model: str,
    dry_run: bool,
    log: Callable[[str], None],
) -> bool:
    """Save an explicit remember call. Returns True if saved."""
    from datetime import datetime

    from mindloop.chunker import Chunk, Turn

    if dry_run:
        log(f"  [remember] would save: {abstract}")
        return True

    chunk = Chunk(turns=[Turn(timestamp=datetime.now(), role="memory", text=text)])
    log(f"  [remember] summarizing: {abstract}")
    cs = summarize_chunk(chunk, model=model)
    save_memory(
        store,
        text=text,
        abstract=abstract,
        summary=cs.summary,
        model=model,
        log=log,
    )
    return True


def _save_extracted(
    store: MemoryStore,
    facts: list[dict[str, str]],
    model: str,
    dry_run: bool,
    log: Callable[[str], None],
) -> tuple[int, int]:
    """Dedup-check and save extracted facts. Returns (saved, skipped)."""
    saved = 0
    skipped = 0
    for fact in facts:
        text = fact["text"]
        abstract = fact["abstract"]
        summary = fact.get("summary", abstract)

        if dry_run:
            log(f"  [extract] would save: {abstract}")
            saved += 1
            continue

        # Dedup gate.
        results = store.search(text, top_k=1)
        if results and results[0].cosine_score >= DEDUP_THRESHOLD:
            log(
                f"  [extract] duplicate (cosine={results[0].cosine_score:.2f}),"
                f" skipping: {abstract}"
            )
            skipped += 1
            continue

        save_memory(
            store,
            text=text,
            abstract=abstract,
            summary=summary,
            model=model,
            log=log,
        )
        saved += 1

    return saved, skipped


def rebuild_from_log(
    messages: list[dict[str, object]],
    store: MemoryStore,
    model: str,
    dry_run: bool,
    log: Callable[[str], None],
) -> tuple[int, int, int]:
    """Single-pass rebuild from a message log.

    Walks messages in order:
    - On remember tool call: save immediately.
    - On boundary (or fallback interval): extract window, dedup, save.

    Returns (remembers, facts_saved, facts_skipped).
    """
    use_boundaries = _has_boundaries(messages)
    window: list[dict[str, Any]] = []
    remembers = 0
    facts_saved = 0
    facts_skipped = 0

    def _flush_window() -> None:
        nonlocal facts_saved, facts_skipped
        if not window:
            return
        log(f"  [extract] extracting window ({len(window)} messages)...")
        facts = extract_window(window, model=model)
        s, sk = _save_extracted(store, facts, model, dry_run, log)
        facts_saved += s
        facts_skipped += sk
        window.clear()

    for msg in messages:
        # Check for remember tool calls.
        for text, abstract in _parse_remember_calls(msg):
            remembers += 1
            _save_remember(store, text, abstract, model, dry_run, log)

        # Add to window buffer.
        window.append(msg)

        # Check for extraction boundary.
        if use_boundaries:
            if _is_boundary(msg):
                _flush_window()
        else:
            # Fallback: extract every N messages.
            if len(window) >= _FALLBACK_WINDOW:
                _flush_window()

    # Flush remaining window at end of log.
    _flush_window()

    return remembers, facts_saved, facts_skipped


def main() -> None:
    """CLI entry point. Collect logs, create store, process in order."""
    parser = argparse.ArgumentParser(
        description="Rebuild memory DB from JSONL session logs."
    )
    parser.add_argument(
        "logfiles",
        nargs="*",
        type=Path,
        help="Explicit JSONL log files.",
    )
    parser.add_argument(
        "--session",
        metavar="NAME",
        help="Rebuild from all logs in sessions/<NAME>/logs/.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild from all sessions.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Database path (default: session's memory.db or memory.db).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model for extraction + summarization LLM calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be saved/skipped without writing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each step.")
    args = parser.parse_args()

    # Validate: exactly one source mode.
    modes = bool(args.session) + bool(args.all) + bool(args.logfiles)
    if modes == 0:
        parser.error("Specify --session NAME, --all, or LOGFILE [LOGFILE ...].")
    if modes > 1:
        parser.error(
            "--session, --all, and positional logfiles are mutually exclusive."
        )

    if not API_KEY:
        print("Set OPENROUTER_API_KEY for extraction.", file=sys.stderr)
        sys.exit(1)

    log: Callable[[str], None] = print if args.verbose else noop
    sessions_dir = Path("sessions")

    # Collect log files.
    log_files: list[Path] = []
    db_path: Path

    if args.session:
        session_dir = sessions_dir / args.session
        log_dir = session_dir / "logs"
        if not log_dir.is_dir():
            print(f"Session log dir not found: {log_dir}", file=sys.stderr)
            sys.exit(1)
        log_files = sorted(log_dir.glob("*_agent_*.jsonl"))
        db_path = args.db or (session_dir / "memory.db")
    elif args.all:
        if not sessions_dir.is_dir():
            print(f"Sessions dir not found: {sessions_dir}", file=sys.stderr)
            sys.exit(1)
        for session in sorted(sessions_dir.iterdir()):
            log_dir = session / "logs"
            if log_dir.is_dir():
                log_files.extend(sorted(log_dir.glob("*_agent_*.jsonl")))
        db_path = args.db or Path("memory.db")
    else:
        log_files = [p for p in args.logfiles if p.suffix == ".jsonl"]
        db_path = args.db or Path("memory.db")

    if not log_files:
        print("No JSONL log files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} log files, writing to {db_path}")
    if args.dry_run:
        print("(dry run — nothing will be saved)")

    # Wipe and recreate DB unless dry-run.
    if not args.dry_run:
        if db_path.exists():
            db_path.unlink()
            print(f"Removed existing {db_path}")

    t0 = time.monotonic()
    store = MemoryStore(db_path=db_path)
    total_remembers = 0
    total_saved = 0
    total_skipped = 0

    for i, log_file in enumerate(log_files, 1):
        print(f"\n[{i}/{len(log_files)}] {log_file}")
        messages = _load_messages(log_file)
        if not messages:
            print("  (empty)")
            continue

        r, s, sk = rebuild_from_log(messages, store, args.model, args.dry_run, log)
        total_remembers += r
        total_saved += s
        total_skipped += sk
        print(f"  remembers={r}, extracted={s}, skipped={sk}")

    store.close()
    elapsed = time.monotonic() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    print(
        f"\nDone in {minutes}m{seconds:02d}s. remembers={total_remembers},"
        f" extracted={total_saved}, skipped={total_skipped}"
    )
    if not args.dry_run:
        check = MemoryStore(db_path=db_path)
        print(f"Active memories in DB: {check.count()}")
        check.close()


if __name__ == "__main__":
    main()
