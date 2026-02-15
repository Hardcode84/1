# Idea: Graph-Based Memory (Typed Edges Between Chunks)

**Status: not implemented.**

Keep chunks as separate nodes, connect them with typed edges instead of merging into mega-chunks. Merging conflates two operations — connecting related knowledge and compressing it. Graph edges separate these: related-but-distinct facts stay as individual nodes linked by relations.

## Problem with current merge approach

The merge loop snowballs. Two similar chunks merge, the result is similar to a third, that merges too, and after 5-7 rounds you get a 2000-char blob covering 5 unrelated topics. Guardrails (faithfulness, neighbor score) help but fight symptoms rather than the root cause: merging is the only way to express "these are related."

Observed failure modes:
- **Mega-chunks**: depth-7 merge chains producing unfocused catch-all entries.
- **Information loss**: specific facts diluted in merged philosophical frameworks.
- **Redundancy**: same topic in both a mega-chunk and a smaller standalone entry, too different to dedup.
- **Stale facts**: old version merged into a chunk, new version arrives, both active, contradicting each other.

## Prior art

### A-MEM (NeurIPS 2025)

Zettelkasten-inspired. Memories are structured notes (description, keywords, tags) that dynamically link to form an evolving network. New memories trigger updates to related ones. Already cited in `docs/ideas/automatic_extraction.md`.

Key insight: **memory evolution** — new information reshapes existing context, not just appends.

### Zep / Graphiti (2025)

Temporal knowledge graph with bi-temporal model: when an event occurred vs when it was learned. Every edge has validity intervals. Hybrid retrieval (cosine + BM25 + BFS graph traversal). Real-time incremental updates without batch recomputation.

Key insight: **supersedes** relation — new facts explicitly replace old ones while preserving history.

### HippoRAG (NeurIPS 2024)

Hippocampus-inspired. LLM extracts entities and relations into a schemaless knowledge graph. Retrieval via Personalized PageRank. 10-30x cheaper than iterative retrieval, 20% better on multi-hop QA.

Key insight: **graph traversal for retrieval** — follow edges rather than relying solely on embedding similarity.

### MAGMA (2026)

Four orthogonal graph views per memory: semantic, temporal, causal, entity. Query-adaptive selection of which view to traverse. Up to 45% better reasoning accuracy.

Key insight: **multiple relation types** serve different retrieval needs.

## Proposed design for mindloop

### Edge types

Minimal set to start:

| Edge | Meaning | When created |
|---|---|---|
| `similar_to` | Topically related | On save, when merge gate *rejects* a merge (failed faithfulness, neighbor score too high) but similarity is high |
| `supersedes` | Newer version of same fact | On save, when new chunk covers same topic with updated info. Old chunk deactivated, edge preserves lineage |
| `elaborates` | Adds detail to existing chunk | On save, when new chunk extends an existing one without replacing it |

`contradicts` is tempting but hard to detect reliably without an LLM call. Better to let `supersedes` handle the common case (fact changed) and leave genuine contradictions for later.

### Schema change

```sql
CREATE TABLE chunk_edges (
    source_id INTEGER NOT NULL REFERENCES chunks(id),
    target_id INTEGER NOT NULL REFERENCES chunks(id),
    edge_type TEXT NOT NULL,  -- 'similar_to', 'supersedes', 'elaborates'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, target_id, edge_type)
);
CREATE INDEX idx_edges_source ON chunk_edges(source_id);
CREATE INDEX idx_edges_target ON chunk_edges(target_id);
```

Lightweight — no graph database needed, plain SQLite.

### Changes to save_memory()

Current flow:
1. Search for similar chunks.
2. If similar enough, merge into one chunk. Deactivate absorbed chunk.
3. Repeat until fixed point.

Proposed flow:
1. Search for similar chunks.
2. If highly similar and both small → merge (keep current behavior for genuine duplicates).
3. If similar but merge loses focus (faithfulness/neighbor rejection) → add `similar_to` edge, keep both.
4. If new chunk supersedes existing fact → add `supersedes` edge, deactivate old chunk.
5. No merge loop beyond 1-2 rounds.

The key change: the merge loop becomes a **classify and link** step. Most relations become edges, not merges. Merging is reserved for near-duplicates where two chunks are genuinely saying the same thing.

### Changes to search()

Current: flat cosine + BM25 hybrid search.

Proposed: after top-k retrieval, expand results by following edges 1 hop.

```
results = search(query, top_k=5)
expanded = set(results)
for r in results:
    for edge in get_edges(r.id):
        expanded.add(edge.target)
return rank(expanded, query)
```

This gives the agent richer context without requiring the LLM to stuff everything into one chunk. Cost: one extra SQL query per search.

### Changes to recall tool output

Current format: flat ranked list.

Proposed: show relations.

```
[1] #42 (score=0.87) "Python prefers explicit over implicit"
    ├── similar_to #38 "Zen of Python principles"
    └── supersedes #29 "Python style is implicit"
```

### Supersedes detection

When `should_merge()` returns true but the chunks have low source similarity (cosine(a, b) < threshold), it's likely a supersede rather than a merge. The new chunk is an updated version, not additional detail.

Heuristic:
- High similarity + small chunks → merge (dedup).
- High similarity + merge rejected by gate → `similar_to` edge.
- LLM says merge + low source similarity → `supersedes` edge, deactivate old.

### Migration

The `source_a`/`source_b` columns on chunks already encode merge lineage. These become `supersedes` edges in the new schema. Existing merged chunks stay as-is; new saves use the edge system.

## Interaction with other ideas

- **Memory activation** (`memory_activation.md`): activation scores work on nodes regardless of edges. Edges could boost activation spreading — retrieving one node boosts connected ones (Hebbian).
- **Intrusive recall** (`intrusive_recall.md`): graph traversal could surface not just the recalled memory but its neighbors, providing richer context in nudges.
- **Importance scoring** (`importance_scoring.md`): graph centrality (degree, PageRank) becomes a natural importance signal — well-connected memories are more central to the knowledge base.

## Implementation order

1. Add `chunk_edges` table and migration.
2. Emit `similar_to` edges when merge is rejected (faithfulness failure, neighbor score) but similarity was above `sim_low`.
3. Add `supersedes` edge type and detection heuristic.
4. Expand search results by 1-hop edge traversal.
5. Update recall tool to show relations.
6. Reduce merge loop `max_rounds` from 10 to 2-3.

Steps 1-3 are low-risk additive changes. Steps 4-5 change retrieval behavior. Step 6 is the payoff — less merging, more linking.

## Open questions

- Should `similar_to` be symmetric? (A similar_to B implies B similar_to A.) Simplifies queries but doubles storage.
- Edge weight: should `similar_to` store the cosine score? Useful for ranking expanded results.
- Pruning: do we ever remove edges? Or let activation decay handle irrelevance?
- How does the agent discover the graph? Should recall_detail show the full neighborhood, or should there be a dedicated `explore_memory` tool?
