"""Dump the entire memory database to text."""

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump memory database to text.")
    parser.add_argument("--db", default="memory.db", help="Database path.")
    parser.add_argument("-o", "--output", help="Output file (default: stdout).")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, text, abstract, summary, active, source_a, source_b, created_at "
        "FROM chunks ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        print("Database is empty.")
        return

    # Build reverse index: chunk id -> list of merge ids that used it as source.
    merged_into: dict[int, list[int]] = {}
    for row in rows:
        cid, _text, _abs, _sum, _act, src_a, src_b, _ts = row
        for src in (src_a, src_b):
            if src is not None:
                merged_into.setdefault(src, []).append(cid)

    lines: list[str] = []
    for row in rows:
        cid, text, abstract, summary, active, src_a, src_b, created = row
        status = "active" if active else "inactive"
        sources = ""
        if src_a is not None or src_b is not None:
            parts = [f"#{s}" for s in (src_a, src_b) if s is not None]
            sources = f"  sources: {', '.join(parts)}"
        merges = ""
        if cid in merged_into:
            merges = f"  merged_into: {', '.join(f'#{m}' for m in merged_into[cid])}"

        lines.append(f"--- #{cid} ({status}){sources}{merges}  [{created}] ---")
        lines.append(f"Abstract: {abstract}")
        lines.append(f"Summary: {summary}")
        lines.append(text)
        lines.append("")

    output = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Dumped {len(rows)} chunks to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
