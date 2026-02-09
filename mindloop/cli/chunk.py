"""Chat log chunker CLI."""

import argparse
from pathlib import Path

from mindloop.chunker import (
    Chunk,
    chunk_turns,
    cosine_similarities,
    merge_chunks,
    parse_turns,
)
from mindloop.client import API_KEY, Embeddings, get_embeddings


def print_chunks(chunks: list[Chunk], embeddings: Embeddings | None = None) -> None:
    similarities = (
        cosine_similarities(embeddings)
        if embeddings is not None and len(embeddings) > 1
        else None
    )

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} [{chunk.time_range}] ({len(chunk.turns)} turns) ---")
        for turn in chunk.turns:
            ts = turn.timestamp.strftime("%H:%M:%S")
            print(f"  {ts} {turn.role}: {turn.text}")
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
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"File not found: {args.logfile}")
        return

    turns = parse_turns(args.logfile)
    if not turns:
        print("No turns found in log file.")
        return

    chunks = chunk_turns(turns)

    if args.embed:
        if not API_KEY:
            print("Set OPENROUTER_API_KEY for embeddings.")
            return

        if len(chunks) >= 2:
            embeddings = get_embeddings([c.text for c in chunks])
            sims = cosine_similarities(embeddings)
            print("=== Before merging ===\n")
            print_chunks(chunks, embeddings)

            chunks = merge_chunks(chunks, sims)
            print(f"=== After merging ({len(chunks)} chunks) ===\n")

    print_chunks(chunks)

    if args.summarize:
        if not API_KEY:
            print("Set OPENROUTER_API_KEY for summarization.")
            return

        from mindloop.summarizer import summarize_chunks

        print("=== Summaries ===\n")
        summaries = summarize_chunks(chunks)
        for i, summary in enumerate(summaries, 1):
            print(f"--- Chunk {i} ---")
            print(f"  Abstract: {summary.abstract}")
            print(f"  Summary:  {summary.summary}")
            print()


if __name__ == "__main__":
    main()
