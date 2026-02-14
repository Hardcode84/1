"""Extract memories from a JSONL agent log into the memory database."""

import argparse
import json
import sys
from pathlib import Path

from mindloop.client import API_KEY
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
        description="Extract memories from an agent JSONL log."
    )
    parser.add_argument("logfile", type=Path, help="Path to the JSONL log file.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("memory.db"),
        help="Database path (default: memory.db).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for extraction + summarization LLM calls.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel extraction workers.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each step.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and print facts without saving to DB.",
    )
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"File not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    if not API_KEY:
        print("Set OPENROUTER_API_KEY for extraction.", file=sys.stderr)
        sys.exit(1)

    messages = _load_messages(args.logfile)
    if not messages:
        print("No messages found in log file.", file=sys.stderr)
        sys.exit(1)

    log = print if args.verbose else (lambda _msg: None)

    if args.dry_run:
        from mindloop.chunker import chunk_turns, compact_chunks, merge_chunks
        from mindloop.client import get_embeddings
        from mindloop.extractor import extract_facts
        from mindloop.recap import collapse_messages

        turns = collapse_messages(messages)
        chunks = compact_chunks(chunk_turns(turns))
        if len(chunks) >= 2:
            embeddings = get_embeddings([c.text for c in chunks])
            chunks = merge_chunks(chunks, embeddings, log=log)

        total = 0
        for i, chunk in enumerate(chunks):
            context = chunks[i - 1].text[-200:] if i > 0 else None
            facts = extract_facts(chunk.text, context=context, model=args.model)
            for fact in facts:
                print(f"  [{fact['abstract']}] {fact['text']}")
            total += len(facts)
        print(f"\nExtracted {total} facts from {len(chunks)} chunks.")
    else:
        from mindloop.extractor import extract_session
        from mindloop.memory import MemoryStore

        store = MemoryStore(db_path=args.db)
        saved = extract_session(
            messages, store, model=args.model, log=log, workers=args.workers
        )
        print(f"Extracted {saved} facts from session.")


if __name__ == "__main__":
    main()
