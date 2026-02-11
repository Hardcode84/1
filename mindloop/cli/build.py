"""Build semantic database from files."""

import argparse
import glob
from pathlib import Path

from mindloop.chunker import chunk_turns, compact_chunks, parse_turns, parse_turns_md
from mindloop.memory import MemoryStore
from mindloop.merge_llm import MergeResult
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunk


def _truncate(text: str, limit: int = 120) -> str:
    """Truncate text to a single line for display."""
    line = text.replace("\n", " ")
    if len(line) > limit:
        return line[:limit] + "..."
    return line


def process_file(
    path: Path, store: MemoryStore, model: str, verbose: bool = False
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

    for i, chunk in enumerate(chunks, 1):
        cs = summarize_chunk(chunk, model=model)

        if verbose:
            print(f"  Summarized [{i}]: {cs.abstract}")

        def _on_merge(mr: MergeResult, idx: int = i) -> None:
            print(f"  Merged [{idx}]: {_truncate(mr.text)}")

        save_memory(
            store,
            chunk.text,
            cs.abstract,
            cs.summary,
            model=model,
            on_merge=_on_merge if verbose else None,
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
            n = process_file(path, store, args.model, verbose=args.verbose)
            print(f"  → {n} chunks saved")
    finally:
        store.close()


if __name__ == "__main__":
    main()
