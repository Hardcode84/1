"""Generate a session recap from a JSONL agent log."""

import argparse
import json
import sys
from pathlib import Path

from mindloop.client import API_KEY
from mindloop.recap import generate_recap, save_recap
from mindloop.util import DEFAULT_WORKERS


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a session recap from an agent JSONL log."
    )
    parser.add_argument("logfile", type=Path, help="Path to the JSONL log file.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for summarization (default: summarizer default).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="Token budget for the recap (default: 1000).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write recap to file instead of stdout.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel summarization workers.",
    )
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"File not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    if not API_KEY:
        print("Set OPENROUTER_API_KEY for summarization.", file=sys.stderr)
        sys.exit(1)

    messages = _load_messages(args.logfile)
    if not messages:
        print("No messages found in log file.", file=sys.stderr)
        sys.exit(1)

    recap = generate_recap(
        messages,
        model=args.model,
        token_budget=args.budget,
        log=print,
        workers=args.workers,
    )
    if not recap:
        print("No recap generated (too few messages?).", file=sys.stderr)
        sys.exit(1)

    if args.output:
        save_recap(args.output, recap)
        print(f"Recap written to {args.output}")
    else:
        print(recap)


if __name__ == "__main__":
    main()
