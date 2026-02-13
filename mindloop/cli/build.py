"""Build semantic database from files."""

import argparse
import glob
from pathlib import Path

from mindloop.chunker import chunk_turns, compact_chunks, parse_turns, parse_turns_md
from mindloop.memory import MemoryStore
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunks
from mindloop.util import DEFAULT_WORKERS


def _truncate(text: str, limit: int = 120) -> str:
    """Truncate text to a single line for display."""
    line = text.replace("\n", " ")
    if len(line) > limit:
        return line[:limit] + "..."
    return line


def process_file(
    path: Path,
    store: MemoryStore,
    model: str,
    verbose: bool = False,
    workers: int = 1,
) -> int:
    """Run the full pipeline on a single file. Returns number of chunks saved."""
    if path.suffix == ".md":
        turns = parse_turns_md(path)
    else:
        turns = parse_turns(path)

    if not turns:
        return 0

    chunks = compact_chunks(chunk_turns(turns))

    if verbose:
        print(f"  Original chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"    [{i}] {_truncate(chunk.text)}")

    # Summarize all chunks (parallel when workers > 1).
    summaries = summarize_chunks(
        chunks, model=model, log=print if verbose else lambda _: None, workers=workers
    )

    # Save sequentially (SQLite writes can't be parallelized safely).
    _log = print if verbose else lambda _s: None
    for i, cs in enumerate(summaries, 1):
        if verbose:
            print(f"  Summarized [{i}]: {cs.abstract}")

        save_memory(
            store,
            cs.chunk.text,
            cs.abstract,
            cs.summary,
            model=model,
            log=_log,
        )

        if verbose:
            print(f"  Saved [{i}]")

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic database from files.")
    parser.add_argument("pattern", help="Glob pattern, e.g. docs/**/*.md")
    parser.add_argument("--db", default="memory.db", help="Database path.")
    parser.add_argument(
        "--model", default="openrouter/free", help="Model for LLM calls."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print each pipeline stage."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel summarization workers.",
    )
    args = parser.parse_args()

    paths = sorted(
        p
        for p in (Path(m) for m in glob.glob(args.pattern, recursive=True))
        if p.is_file()
    )

    if not paths:
        print(f"No files match pattern: {args.pattern}")
        return

    store = MemoryStore(db_path=Path(args.db))
    try:
        for path in paths:
            print(f"{path}")
            n = process_file(
                path, store, args.model, verbose=args.verbose, workers=args.workers
            )
            print(f"  → {n} chunks saved")
    finally:
        store.close()


if __name__ == "__main__":
    main()
