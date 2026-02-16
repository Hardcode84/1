# Memory Discoverability

Over-merging is not the core problem. The merge loop preserves all originals in the database via `source_a`/`source_b` lineage. The real problem is that deactivated chunks are invisible to `search()` (`WHERE active = 1`), so specific facts buried inside mega-chunks are unreachable without manual multi-hop traversal.

Two gaps addressed:
- **Vertical (depth):** original facts inside a merge tree hidden behind the active mega-chunk.
- **Horizontal (breadth):** relationship evidence from rejected merges was discarded.

## 1. `original_only` search

`recall` exposes an `original_only` flag that passes through to `store.search(original_only=True)`, searching leaf chunks (those with no `source_a`/`source_b`) regardless of active status. The agent can bypass mega-chunks and search specific original facts directly.

## 2. Merge depth and leaf count in `recall`

Each `recall` result shows merge tree metadata when the chunk is not a leaf:

```
[1] #42 (score=0.87, depth=3, sources=4) "Python coding style preferences"
    Python follows explicit-over-implicit philosophy...
```

`depth` = merge tree height (0 for originals, omitted). `sources` = leaf count (1 for originals, omitted). Computed by `merge_stats()` which walks the `source_a`/`source_b` tree with caching.

## 3. Leaf abstracts in `recall_detail`

When the agent requests detail on a merged chunk, `recall_detail` shows a flat list of original leaf abstracts instead of the intermediate merge tree:

```
Sources (4 originals):
  #12: "Python is explicit"
  #18: "List comprehensions"
  #27: "PEP 8 rules"
  #9:  "Type hints"
```

Collapses a multi-call traversal into one. Uses `_collect_leaves()` on the lineage tree.

## 4. Rejection-based edges

When the merge loop rejects a merge, the relationship evidence is recorded as a `related_to` edge. Edges are emitted at all three rejection points:

- **LLM says "no merge"** in the middle similarity zone (0.2-0.9).
- **Leaf faithfulness failure:** merge drifted too far from original leaves.
- **Neighbor score rejection:** merge was faithful but too generic.

Schema: `chunk_edges` table with `(source_id, target_id, edge_type, score)` and `INSERT OR IGNORE` dedup.

## 5. Leaf faithfulness

`leaf_faithfulness()` checks `sim(merged, leaf_i)` for ALL original leaf texts in the merge tree (both the incoming chain and the existing chunk's lineage). An `incoming_leaves` accumulator grows across merge rounds, avoiding re-traversal. This naturally caps depth — the more leaves, the harder to stay faithful to every one.

**Edge inheritance:** when chunks merge, `inherit_edges()` moves their `related_to` edges onto the merged chunk with cosine re-scoring (one batched `get_embeddings` call). Old edges on absorbed chunks are deleted. Each cascade round inherits from the previous, so edges propagate transitively. See `docs/design/memory.md` for details.

## Exposing connections to the agent

Two mechanisms, chosen for simplicity:

**Hint in `recall`.** Edge count per result: `[1] #42 (score=0.87, +3 related) "Python style"`. The agent knows there is more to explore. Computed by `edge_counts()`.

**Show in `recall_detail`.** When the agent requests detail, edges are listed with abstracts:

```
Related (3):
  #45 (related_to): "abstract text"
  #38 (related_to): "abstract text"
```

## Remaining work

- **Supersedes detection** in `should_merge`. Deactivate old fact, keep new one, no merge. Prevents contradictory blobs from stale facts. The rest of the richer `should_merge` output (`related`, `distinct`) is already covered by three-tier similarity + edge recording.
- **Invisible search expansion** (follow edges 1-hop after top-k retrieval, re-rank). Would improve recall quality silently, but not yet implemented.

## Interaction with graph_memory.md

This is a pragmatic subset of the graph memory idea. Reuses the `chunk_edges` table but starts with a single edge type (`related_to`) from rejection events, rather than the full classify-and-link paradigm. The `graph_memory.md` proposal remains the long-term direction.
