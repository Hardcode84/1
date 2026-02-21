"""Dry-run should_merge on previously-rejected pairs in a session database.

Reads related_to edges between active chunks and re-evaluates each pair
with the current should_merge prompt. Reports which pairs flip to "yes".

Usage:
    python scripts/test_merge_prompt.py sessions/af2d06a5/memory.db
    python scripts/test_merge_prompt.py DB --cluster 77,122,140,145
    python scripts/test_merge_prompt.py DB --model deepseek/deepseek-v3.2
"""

import argparse
import sqlite3
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test merge prompt on previously-rejected pairs."
    )
    parser.add_argument("db", help="Path to memory.db")
    parser.add_argument(
        "--cluster", help="Comma-separated chunk IDs to restrict to"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-haiku",
        help="Model for should_merge calls (default: anthropic/claude-haiku)",
    )
    args = parser.parse_args()

    # Import after argparse so --help works without OPENROUTER_API_KEY.
    from mindloop.merge_llm import should_merge

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT id, text, abstract FROM chunks WHERE active = 1"
    ).fetchall()
    text_by_id = {r[0]: r[1] for r in rows}
    abstract_by_id = {r[0]: r[2] for r in rows}

    edges = conn.execute(
        "SELECT source_id, target_id, score FROM chunk_edges "
        "WHERE edge_type = 'related_to'"
    ).fetchall()
    conn.close()

    # Filter to active-active pairs.
    active = set(text_by_id)
    pairs = [(s, t, sc) for s, t, sc in edges if s in active and t in active]

    if args.cluster:
        cluster = {int(x) for x in args.cluster.split(",")}
        pairs = [(s, t, sc) for s, t, sc in pairs if s in cluster and t in cluster]

    pairs.sort(key=lambda x: -x[2])

    print(f"Testing {len(pairs)} previously-rejected pairs (model: {args.model})\n")
    print(f"{'Pair':<12} {'Cosine':>7} {'Old':>5} {'New':>5}  Abstracts")
    print("-" * 80)

    flipped = 0
    for src, tgt, score in pairs:
        try:
            result = should_merge(text_by_id[src], text_by_id[tgt], model=args.model)
        except Exception as e:
            print(f"#{src:<4}↔#{tgt:<4}  {score:>6.3f}    no   ERR   {e}")
            continue

        new = "yes" if result else "no"
        marker = " ←" if result else ""
        if result:
            flipped += 1

        abs_src = abstract_by_id[src][:30]
        abs_tgt = abstract_by_id[tgt][:30]
        print(
            f"#{src:<4}↔#{tgt:<4}  {score:>6.3f}    no   {new:>3}"
            f"   {abs_src} | {abs_tgt}{marker}"
        )
        time.sleep(0.1)

    print("-" * 80)
    print(f"\nFlipped: {flipped}/{len(pairs)} pairs would now merge.")


if __name__ == "__main__":
    main()
