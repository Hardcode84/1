"""Chat log chunker CLI."""

import argparse
from pathlib import Path

from mindloop.chunker import (
    DEFAULT_GAP_THRESHOLD,
    Chunk,
    chunk_turns,
    cosine_similarities,
    merge_chunks,
    parse_turns,
)
from mindloop.client import API_KEY, get_embeddings


def print_chunks(chunks: list[Chunk], embeddings: list[list[float]] | None = None):
    similarities = (
        cosine_similarities(embeddings) if embeddings and len(embeddings) > 1 else None
    )

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} [{chunk.time_range}] ({len(chunk.turns)} turns) ---")
        for turn in chunk.turns:
            ts = turn.timestamp.strftime("%H:%M:%S")
            print(f"  {ts} {turn.role}: {turn.text}")
        if similarities and i < len(similarities):
            print(f"  Similarity to next chunk: {similarities[i]:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Split a chat log into chunks.")
    parser.add_argument("logfile", type=Path, help="Path to the log file.")
    parser.add_argument(
        "--gap",
        type=int,
        default=DEFAULT_GAP_THRESHOLD,
        help=f"Time gap in seconds to split chunks (default: {DEFAULT_GAP_THRESHOLD}).",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Fetch embeddings for each chunk via OpenRouter.",
    )
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"File not found: {args.logfile}")
        return

    turns = parse_turns(args.logfile)
    if not turns:
        print("No turns found in log file.")
        return

    chunks = chunk_turns(turns, args.gap)

    if not args.embed:
        print_chunks(chunks)
        return

    if not API_KEY:
        print("Set OPENROUTER_API_KEY for embeddings.")
        return

    embeddings = get_embeddings([c.text for c in chunks])
    if len(chunks) < 2:
        print_chunks(chunks)
        return

    sims = cosine_similarities(embeddings)
    print("=== Before merging ===\n")
    print_chunks(chunks, embeddings)

    merged = merge_chunks(chunks, sims)
    print(f"=== After merging ({len(chunks)} -> {len(merged)} chunks) ===\n")
    print_chunks(merged)


if __name__ == "__main__":
    main()
