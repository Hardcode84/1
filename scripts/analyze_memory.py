"""Analyze a memory.db for anomalies and redundancy.

Reports structural integrity, redundancy clusters, low-information chunks,
potential contradictions, and edge graph statistics.

Usage:
    python scripts/analyze_memory.py sessions/af2d06a5/memory.db
"""

import argparse
import math
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations

from mindloop.voting import compression_score, ngram_diversity

JACCARD_THRESHOLD = 0.50
JACCARD_CLUSTER_MIN = 3
JACCARD_PAIR_THRESHOLD = 0.35
SHORT_TEXT_THRESHOLD = 60
TFIDF_COSINE_THRESHOLD = 0.45
META_KEYWORDS = [
    "restraint", "token efficiency", "completing tasks",
    "self-referential", "meta-cognitive", "being concise",
    "saving tokens", "avoid unnecessary", "be brief",
    "minimize output", "reduce verbosity", "busywork",
    "not invent work", "genuine request", "avoid the discomfort",
    "should not verify", "don't repeat", "stop working",
    "self-analysis", "navel-gazing",
]

NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "doesn't",
    "don't", "isn't", "aren't", "wasn't", "weren't", "won't",
    "can't", "cannot", "shouldn't", "wouldn't", "couldn't",
    "without", "lacks", "unlike", "instead", "however",
    "except", "rather", "false", "incorrect", "wrong",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "and", "but", "or", "nor",
    "not", "so", "if", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "them", "his", "her", "their", "which", "who", "whom", "what", "when",
    "where", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "also", "about", "up", "s", "t", "re", "ve", "ll",
    "d", "m",
}


def _word_set(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    words = []
    for w in text.lower().split():
        w = w.strip(".,;:!?()[]\"'-/")
        if w and w not in STOPWORDS and len(w) > 2:
            words.append(w)
    return words


def _tfidf_vectors(docs: dict[int, str]) -> dict[int, dict[str, float]]:
    """Compute TF-IDF vectors for a collection of documents."""
    n = len(docs)
    df: Counter[str] = Counter()
    tf_per_doc: dict[int, Counter[str]] = {}
    for doc_id, text in docs.items():
        tokens = _tokenize(text)
        tf = Counter(tokens)
        tf_per_doc[doc_id] = tf
        for term in set(tokens):
            df[term] += 1

    vectors: dict[int, dict[str, float]] = {}
    for doc_id, tf in tf_per_doc.items():
        vec: dict[str, float] = {}
        max_tf = max(tf.values()) if tf else 1
        for term, count in tf.items():
            idf = math.log(n / (1 + df[term]))
            vec[term] = (0.5 + 0.5 * count / max_tf) * idf
        vectors[doc_id] = vec
    return vectors


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _connected_components(
    adj: dict[int, set[int]], min_size: int
) -> list[set[int]]:
    """Extract connected components with at least *min_size* members."""
    visited: set[int] = set()
    clusters: list[set[int]] = []
    for node in adj:
        if node in visited:
            continue
        component: set[int] = set()
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            queue.extend(adj[n] - visited)
        if len(component) >= min_size:
            clusters.append(component)
    return clusters


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze memory.db for anomalies.")
    parser.add_argument("db", help="Path to memory.db")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # ── Basic stats ─────────────────────────────────────────
    _section("BASIC STATS")

    total = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    active = conn.execute("SELECT COUNT(*) c FROM chunks WHERE active=1").fetchone()["c"]
    inactive = total - active
    print(f"  Total chunks:   {total}")
    print(f"  Active:         {active}")
    print(f"  Inactive:       {inactive}")
    print(f"  Merge ratio:    {inactive}/{total} = {inactive / total:.1%}" if total else "")

    # Merge tree depth.
    all_rows = conn.execute(
        "SELECT id, text, active, source_a, source_b FROM chunks"
    ).fetchall()
    parent_map = {r["id"]: (r["source_a"], r["source_b"]) for r in all_rows}
    all_ids = {r["id"] for r in all_rows}

    def depth(cid: int, seen: set[int] | None = None) -> int:
        if seen is None:
            seen = set()
        if cid in seen:
            return 0
        seen.add(cid)
        sa, sb = parent_map.get(cid, (None, None))
        if sa is None and sb is None:
            return 0
        da = depth(sa, seen) if sa is not None else 0
        db = depth(sb, seen) if sb is not None else 0
        return 1 + max(da, db)

    active_rows = conn.execute(
        "SELECT id, text, abstract, summary, source_a, source_b "
        "FROM chunks WHERE active=1"
    ).fetchall()
    active_ids = [r["id"] for r in active_rows]
    active_id_set = set(active_ids)

    print("\n  Merge tree depth distribution (active chunks):")
    depth_dist: dict[int, int] = defaultdict(int)
    for cid in active_ids:
        depth_dist[depth(cid)] += 1
    for d in sorted(depth_dist):
        print(f"    depth {d}: {depth_dist[d]} chunks")

    print("\n  Edge counts by type:")
    for row in conn.execute(
        "SELECT edge_type, COUNT(*) c FROM chunk_edges GROUP BY edge_type ORDER BY c DESC"
    ):
        print(f"    {row['edge_type']:20s} {row['c']}")

    lengths = sorted(len(r["text"]) for r in active_rows)
    print(
        f"\n  Text length: min={lengths[0]}, "
        f"median={lengths[len(lengths) // 2]}, max={lengths[-1]}, "
        f"avg={sum(lengths) / len(lengths):.0f}"
    )

    comp_scores = [compression_score(r["text"]) for r in active_rows]
    div_scores = [ngram_diversity(r["text"]) for r in active_rows]
    print(
        f"  Compression:  min={min(comp_scores):.3f}, "
        f"avg={sum(comp_scores) / len(comp_scores):.3f}, max={max(comp_scores):.3f}"
    )
    print(
        f"  Ngram div:    min={min(div_scores):.3f}, "
        f"avg={sum(div_scores) / len(div_scores):.3f}, max={max(div_scores):.3f}"
    )

    # ── Prepare lookup maps ────────────────────────────────
    text_map = {r["id"]: r["text"] for r in active_rows}
    abstract_map = {r["id"]: r["abstract"] for r in active_rows}
    word_sets = {cid: _word_set(txt) for cid, txt in text_map.items()}

    # ── Dense clusters (Jaccard) ───────────────────────────
    _section(f"DENSE CLUSTERS (word Jaccard > {JACCARD_THRESHOLD})")

    adj: dict[int, set[int]] = defaultdict(set)
    for (id_a, ws_a), (id_b, ws_b) in combinations(word_sets.items(), 2):
        if _jaccard(ws_a, ws_b) > JACCARD_THRESHOLD:
            adj[id_a].add(id_b)
            adj[id_b].add(id_a)

    clusters = _connected_components(adj, JACCARD_CLUSTER_MIN)
    if not clusters:
        print("  None found.")
    for i, cluster in enumerate(clusters, 1):
        print(f"\n  Cluster {i} ({len(cluster)} chunks):")
        for cid in sorted(cluster):
            cs = compression_score(text_map[cid])
            nd = ngram_diversity(text_map[cid])
            print(f"    [id={cid}] comp={cs:.3f} ndiv={nd:.3f}")
            print(f"      {text_map[cid][:200]}")

    # Near-duplicate pairs outside clusters.
    print(f"\n  Near-duplicate pairs (word Jaccard > {JACCARD_PAIR_THRESHOLD}):")
    near_dup: list[tuple[int, int, float]] = []
    for (id_a, ws_a), (id_b, ws_b) in combinations(word_sets.items(), 2):
        j = _jaccard(ws_a, ws_b)
        if j > JACCARD_PAIR_THRESHOLD:
            if not any(id_a in c and id_b in c for c in clusters):
                near_dup.append((id_a, id_b, j))
    near_dup.sort(key=lambda x: -x[2])
    if not near_dup:
        print("    None found.")
    for id_a, id_b, j in near_dup[:20]:
        print(f"\n    ({id_a}, {id_b}) Jaccard={j:.3f}")
        print(f"      A: {text_map[id_a][:180]}")
        print(f"      B: {text_map[id_b][:180]}")

    # ── TF-IDF paraphrase detection ────────────────────────
    _section(f"TF-IDF NEAR-DUPLICATES (cosine > {TFIDF_COSINE_THRESHOLD})")

    vectors = _tfidf_vectors(text_map)
    tfidf_pairs: list[tuple[int, int, float]] = []
    for (id_a, va), (id_b, vb) in combinations(vectors.items(), 2):
        sim = _cosine_sim(va, vb)
        if sim > TFIDF_COSINE_THRESHOLD:
            tfidf_pairs.append((id_a, id_b, sim))
    tfidf_pairs.sort(key=lambda x: -x[2])

    if not tfidf_pairs:
        print("  None found.")
    for id_a, id_b, sim in tfidf_pairs[:25]:
        print(f"\n  ({id_a}, {id_b}) cosine={sim:.3f}")
        print(f"    A: {text_map[id_a][:200]}")
        print(f"    B: {text_map[id_b][:200]}")

    tfidf_adj: dict[int, set[int]] = defaultdict(set)
    for id_a, id_b, _ in tfidf_pairs:
        tfidf_adj[id_a].add(id_b)
        tfidf_adj[id_b].add(id_a)
    tfidf_clusters = _connected_components(tfidf_adj, 3)
    if tfidf_clusters:
        print(f"\n  TF-IDF dense clusters (>=3 members):")
        for i, cluster in enumerate(tfidf_clusters, 1):
            print(f"\n  -- Cluster {i} ({len(cluster)} chunks) --")
            for cid in sorted(cluster):
                print(f"    [id={cid}] {text_map[cid][:180]}")

    # ── Abstract similarity ────────────────────────────────
    _section("SEMANTIC BLOAT (abstract Jaccard > 0.50)")

    abstract_ws = {cid: _word_set(ab) for cid, ab in abstract_map.items()}
    paraphrase: list[tuple[int, int, float]] = []
    for (id_a, ws_a), (id_b, ws_b) in combinations(abstract_ws.items(), 2):
        j = _jaccard(ws_a, ws_b)
        if j > 0.50:
            paraphrase.append((id_a, id_b, j))
    paraphrase.sort(key=lambda x: -x[2])
    if not paraphrase:
        print("  None found.")
    for id_a, id_b, j in paraphrase[:20]:
        print(f"\n  ({id_a}, {id_b}) J(abstract)={j:.3f}")
        print(f"    A: {abstract_map[id_a][:200]}")
        print(f"    B: {abstract_map[id_b][:200]}")

    # ── Low-information chunks ─────────────────────────────
    _section("LOW-INFORMATION CHUNKS")

    low_info: list[tuple[int, str, str]] = []
    for r in active_rows:
        txt = r["text"]
        lower = txt.lower()
        reason = ""
        if len(txt) < SHORT_TEXT_THRESHOLD:
            reason = "very short"
        for kw in META_KEYWORDS:
            if kw in lower:
                reason = f"meta keyword '{kw}'"
                break
        cs = compression_score(txt)
        if cs < 0.25:
            reason = f"highly compressible (comp={cs:.3f})"
        if reason:
            low_info.append((r["id"], reason, txt))

    if not low_info:
        print("  None found.")
    for cid, reason, txt in low_info:
        print(f"\n  [id={cid}] ({reason})")
        print(f"    {txt[:600]}")

    # ── Contradictions ─────────────────────────────────────
    _section("POTENTIAL CONTRADICTIONS (overlap + asymmetric negation)")

    contradiction_pairs: list[tuple[int, int, float, set[str]]] = []
    for (id_a, ws_a), (id_b, ws_b) in combinations(word_sets.items(), 2):
        j = _jaccard(ws_a, ws_b)
        if j < 0.25:
            continue
        diff = (ws_a & NEGATION_WORDS).symmetric_difference(ws_b & NEGATION_WORDS)
        if diff:
            contradiction_pairs.append((id_a, id_b, j, diff))
    contradiction_pairs.sort(key=lambda x: -x[2])
    if not contradiction_pairs:
        print("  None found.")
    for id_a, id_b, j, diff in contradiction_pairs[:15]:
        print(f"\n  ({id_a}, {id_b}) J={j:.3f}  neg_diff={diff}")
        print(f"    A: {text_map[id_a][:200]}")
        print(f"    B: {text_map[id_b][:200]}")

    # ── Structural integrity ───────────────────────────────
    _section("STRUCTURAL INTEGRITY")

    edges = conn.execute(
        "SELECT source_id, target_id, edge_type, score FROM chunk_edges"
    ).fetchall()

    orphaned_missing = [
        e for e in edges
        if e["source_id"] not in all_ids or e["target_id"] not in all_ids
    ]
    orphaned_inactive = [
        e for e in edges
        if e["source_id"] in all_ids and e["target_id"] in all_ids
        and (e["source_id"] not in active_id_set or e["target_id"] not in active_id_set)
    ]

    print(f"  Edges to non-existent chunks: {len(orphaned_missing)}")
    print(f"  Edges involving inactive chunks: {len(orphaned_inactive)}")
    for e in orphaned_inactive[:10]:
        sides = []
        if e["source_id"] not in active_id_set:
            sides.append(f"src={e['source_id']}")
        if e["target_id"] not in active_id_set:
            sides.append(f"tgt={e['target_id']}")
        print(f"    {e['source_id']} -> {e['target_id']}  ({e['edge_type']})  [inactive: {', '.join(sides)}]")
    if len(orphaned_inactive) > 10:
        print(f"    ... and {len(orphaned_inactive) - 10} more")

    broken_merge = []
    for r in all_rows:
        bad = []
        if r["source_a"] is not None and r["source_a"] not in all_ids:
            bad.append(f"source_a={r['source_a']}")
        if r["source_b"] is not None and r["source_b"] not in all_ids:
            bad.append(f"source_b={r['source_b']}")
        if bad:
            broken_merge.append((r["id"], bad))

    if not broken_merge:
        print("  Merge tree pointers: OK")
    for cid, fields in broken_merge:
        print(f"  [id={cid}] missing: {', '.join(fields)}")

    asymmetric = [
        r for r in all_rows
        if (r["source_a"] is None) != (r["source_b"] is None)
    ]
    print(f"  Asymmetric merge pointers: {len(asymmetric)}")
    for r in asymmetric[:5]:
        print(f"    [id={r['id']}] source_a={r['source_a']} source_b={r['source_b']}")

    # ── Edge graph ─────────────────────────────────────────
    _section("EDGE GRAPH")

    degree: dict[int, int] = defaultdict(int)
    for e in edges:
        degree[e["source_id"]] += 1
        degree[e["target_id"]] += 1
    active_degrees = sorted(degree.get(cid, 0) for cid in active_ids)
    isolated = sum(1 for d in active_degrees if d == 0)
    max_possible = len(active_ids) * (len(active_ids) - 1) // 2
    print(
        f"  Degree: min={active_degrees[0]}, "
        f"median={active_degrees[len(active_degrees) // 2]}, "
        f"max={active_degrees[-1]}, avg={sum(active_degrees) / len(active_degrees):.1f}"
    )
    print(f"  Isolated (degree 0): {isolated}")
    print(f"  Edges: {len(edges)} / {max_possible} possible ({len(edges) / max_possible:.4f})")

    print("\n  Highest-degree active chunks:")
    top_deg = sorted(
        ((cid, degree.get(cid, 0)) for cid in active_ids), key=lambda t: -t[1]
    )[:10]
    for cid, deg in top_deg:
        print(f"    [id={cid}] degree={deg}  {text_map[cid][:120]}")

    # ── Thematic overview ──────────────────────────────────
    _section("TOP TERMS IN ABSTRACTS")

    word_freq: Counter[str] = Counter()
    for ab in abstract_map.values():
        for w in _tokenize(ab):
            word_freq[w] += 1
    for w, count in word_freq.most_common(30):
        print(f"    {w:25s} {count}")

    # ── Summary ────────────────────────────────────────────
    _section("SUMMARY")
    print(f"  Total / Active / Inactive:      {total} / {active} / {inactive}")
    print(f"  Dense clusters (J>{JACCARD_THRESHOLD}):        {len(clusters)}")
    print(f"  Near-duplicate pairs (J>{JACCARD_PAIR_THRESHOLD}):   {len(near_dup)}")
    print(f"  TF-IDF near-dup pairs:           {len(tfidf_pairs)}")
    print(f"  TF-IDF dense clusters:           {len(tfidf_clusters)}")
    print(f"  Semantic bloat (abstract):        {len(paraphrase)}")
    print(f"  Low-information chunks:           {len(low_info)}")
    print(f"  Potential contradictions:         {len(contradiction_pairs)}")
    print(f"  Orphaned edges (missing):         {len(orphaned_missing)}")
    print(f"  Orphaned edges (inactive):        {len(orphaned_inactive)}")
    print(f"  Broken merge pointers:            {len(broken_merge)}")
    print(f"  Isolated active chunks:           {isolated}")

    conn.close()


if __name__ == "__main__":
    main()
