"""Query the semantic memory database."""

import argparse
from pathlib import Path

from mindloop.memory import MemoryStore, SearchResult


def _format_sources(result: SearchResult) -> str:
    """Format source lineage as a short string."""
    parts = [s for s in (result.source_a, result.source_b) if s is not None]
    if not parts:
        return "original"
    return "merged from " + ", ".join(f"#{s}" for s in parts)


def print_result(result: SearchResult, rank: int, verbose: bool = False) -> None:
    """Print a single search result."""
    cs = result.chunk_summary
    print(
        f"#{result.id}  [{rank}]  score={result.score:.4f}  ({_format_sources(result)})"
    )
    print(f"  Abstract: {cs.abstract}")
    print(f"  Summary:  {cs.summary}")
    if verbose:
        print("  Text:")
        for line in cs.chunk.text.splitlines():
            print(f"    {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the semantic memory database.")
    parser.add_argument("query", help="Search query text.")
    parser.add_argument("--db", default="memory.db", help="Database path.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show full chunk text."
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    store = MemoryStore(db_path=db_path)
    try:
        results = store.search(args.query, top_k=args.top_k)
        if not results:
            print("No results found.")
            return

        print(f"{len(results)} result(s) for: {args.query}\n")
        for rank, result in enumerate(results, 1):
            print_result(result, rank, verbose=args.verbose)
    finally:
        store.close()


if __name__ == "__main__":
    main()
