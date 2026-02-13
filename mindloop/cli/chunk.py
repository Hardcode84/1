"""Chat log chunker CLI."""

import argparse
from pathlib import Path

from mindloop.chunker import (
    Chunk,
    chunk_turns,
    compact_chunks,
    cosine_similarities,
    merge_chunks,
    parse_turns,
    parse_turns_md,
)
from mindloop.client import API_KEY, Embeddings, get_embeddings
from mindloop.util import DEFAULT_WORKERS


def print_chunks(
    chunks: list[Chunk],
    embeddings: Embeddings | None = None,
    show_timestamps: bool = True,
) -> None:
    similarities = (
        cosine_similarities(embeddings)
        if embeddings is not None and len(embeddings) > 1
        else None
    )

    for i, chunk in enumerate(chunks):
        header = f"--- Chunk {i + 1}"
        if show_timestamps:
            header += f" [{chunk.time_range}]"
        header += f" ({len(chunk.turns)} turns) ---"
        print(header)
        for turn in chunk.turns:
            if show_timestamps:
                ts = turn.timestamp.strftime("%H:%M:%S")
                print(f"  {ts} {turn.role}: {turn.text}")
            else:
                print(f"  {turn.text}")
        if similarities is not None and i < len(similarities):
            print(f"  Similarity to next chunk: {similarities[i]:.4f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a chat log into chunks.")
    parser.add_argument("logfile", type=Path, help="Path to the log file.")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Fetch embeddings for each chunk via OpenRouter.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize each chunk via OpenRouter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel summarization workers.",
    )
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"File not found: {args.logfile}")
        return

    is_md = args.logfile.suffix == ".md"
    if is_md:
        turns = parse_turns_md(args.logfile)
    else:
        turns = parse_turns(args.logfile)
    if not turns:
        print("No turns found in log file.")
        return

    chunks = compact_chunks(chunk_turns(turns))

    if args.embed:
        if not API_KEY:
            print("Set OPENROUTER_API_KEY for embeddings.")
            return

        if len(chunks) >= 2:
            embeddings = get_embeddings([c.text for c in chunks])
            print("=== Before merging ===\n")
            print_chunks(chunks, embeddings, show_timestamps=not is_md)

            chunks = merge_chunks(chunks, embeddings)
            print(f"\n=== After merging ({len(chunks)} chunks) ===\n")

    print_chunks(chunks, show_timestamps=not is_md)

    if args.summarize:
        if not API_KEY:
            print("Set OPENROUTER_API_KEY for summarization.")
            return

        from mindloop.summarizer import summarize_chunks

        print("=== Summaries ===\n")
        summaries = summarize_chunks(chunks, log=print, workers=args.workers)
        for i, summary in enumerate(summaries, 1):
            print(f"--- Chunk {i} ---")
            print(f"  Abstract: {summary.abstract}")
            print(f"  Summary:  {summary.summary}")
            print()


if __name__ == "__main__":
    main()
