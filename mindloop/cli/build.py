"""Build semantic database from files."""

import argparse
import glob
from pathlib import Path

from mindloop.chunker import chunk_turns, compact_chunks, parse_turns, parse_turns_md
from mindloop.memory import MemoryStore
from mindloop.semantic_memory import save_memory
from mindloop.summarizer import summarize_chunk


def process_file(path: Path, store: MemoryStore, model: str) -> int:
    """Run the full pipeline on a single file. Returns number of chunks saved."""
    if path.suffix == ".md":
        turns = parse_turns_md(path)
    else:
        turns = parse_turns(path)

    if not turns:
        return 0

    chunks = compact_chunks(chunk_turns(turns))
    for chunk in chunks:
        cs = summarize_chunk(chunk, model=model)
        save_memory(store, chunk.text, cs.abstract, cs.summary, model=model)

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic database from files.")
    parser.add_argument("pattern", help="Glob pattern, e.g. docs/**/*.md")
    parser.add_argument("--db", default="memory.db", help="Database path.")
    parser.add_argument(
        "--model", default="openrouter/free", help="Model for LLM calls."
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
            n = process_file(path, store, args.model)
            print(f"{path} → {n} chunks → saved")
    finally:
        store.close()


if __name__ == "__main__":
    main()
