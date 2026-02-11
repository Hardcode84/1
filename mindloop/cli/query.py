"""Query the semantic memory database."""

import argparse
from pathlib import Path

from mindloop.memory import LineageNode, MemoryStore, SearchResult


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


def _node_label(node: LineageNode) -> str:
    """Format a single lineage node as a one-line label."""
    status = "active" if node.active else "leaf" if node.is_leaf else "intermediate"
    return f'#{node.id} ({status}) "{node.abstract}"'


def print_tree(node: LineageNode, prefix: str = "", is_last: bool = True) -> None:
    """Print a merge lineage tree with box-drawing connectors."""
    connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
    print(f"{prefix}{connector}{_node_label(node)}" if prefix else _node_label(node))

    children = [c for c in (node.source_a, node.source_b) if c is not None]
    child_prefix = prefix + ("    " if is_last else "\u2502   ")
    for i, child in enumerate(children):
        print_tree(child, child_prefix, is_last=(i == len(children) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the semantic memory database.")
    parser.add_argument("query", nargs="?", help="Search query text.")
    parser.add_argument("--db", default="memory.db", help="Database path.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show full chunk text."
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Search original (unmerged) chunks instead of merged ones.",
    )
    parser.add_argument(
        "--tree",
        type=int,
        metavar="ID",
        help="Show the full merge lineage tree for a chunk ID.",
    )
    args = parser.parse_args()

    if args.tree is None and args.query is None:
        parser.error("either query or --tree ID is required")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    store = MemoryStore(db_path=db_path)
    try:
        if args.tree is not None:
            node = store.lineage(args.tree)
            if node is None:
                print(f"Chunk #{args.tree} not found.")
                return
            print_tree(node)
            return

        results = store.search(
            args.query, top_k=args.top_k, original_only=args.original
        )
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
