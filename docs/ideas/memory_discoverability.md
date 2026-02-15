# Idea: Memory Discoverability

**Status: not implemented.**

Over-merging is not the core problem. The merge loop preserves all originals in the database via `source_a`/`source_b` lineage. The real problem is that deactivated chunks are invisible to `search()` (`WHERE active = 1`), so specific facts buried inside mega-chunks are unreachable without manual multi-hop traversal through `recall_detail`.

Two gaps:
- **Vertical (depth):** original facts inside a merge tree are hidden behind the active mega-chunk. The agent must call `recall_detail`, read the tree, then call `recall_detail` again on each leaf. 3-4 tool calls to reach one specific fact.
- **Horizontal (breadth):** when the merge loop rejects a merge (faithfulness failure, neighbor score rejection), the relationship evidence is discarded. Chunks that are related but were never merged have no connection.

## 1. Expose `original_only` search to the agent

`search()` already has an `original_only` parameter that searches leaf chunks regardless of active status. Nothing in the agent tooling uses it.

Add an `original_only` boolean to `recall` (or a separate `recall_originals` tool) that passes through to `store.search(original_only=True)`. One new `Param`, one parameter passthrough.

## 2. Show merge depth and leaf count in `recall` output

The agent currently cannot distinguish a leaf chunk from a depth-7 mega-chunk in `recall` output. Add two numbers:

```
[1] #42 (score=0.87, depth=3, sources=4) "Python coding style preferences"
    Python follows explicit-over-implicit philosophy...
```

`depth` = merge tree height (0 for originals). `sources` = leaf count. The agent immediately knows whether to drill down. Costs one recursive query or a cached column, adds ~20 characters per result.

## 3. Show leaf abstracts in `recall_detail`

Replace the intermediate-node tree rendering with a flat list of leaf abstracts and summaries:

```
Sources (4 originals):
  #12: "Python is explicit" -- Zen of Python principle, explicit imports.
  #18: "List comprehensions" -- Prefer over map/filter for simple cases.
  #27: "PEP 8 rules" -- 4-space indent, 79-char lines, snake_case.
  #9:  "Type hints" -- Use for public APIs, Optional for nullable.
```

The agent does not care about the merge tree structure (which intermediate merged what). It cares about the original facts. Collapses a 4-call traversal into one.

## 4. Record rejection-based edges

When the merge loop rejects a merge, it has already proven the chunks are related. Record this as an edge instead of discarding it.

Emit `related_to` edges at:
- **Faithfulness rejection:** LLM merged the texts but result drifted too far from a source. Chunks are related, merge is lossy.
- **Neighbor score rejection:** merge was faithful but result is too generic. Chunks are related, merging would lose focus.
- **LLM says "no merge"** in the middle similarity zone (0.2-0.9): weaker signal, still worth recording.

Schema: same `chunk_edges` table from `graph_memory.md`. Single edge type for now. ~50 lines of code, no changes to merge logic itself.

## 5. Anchor faithfulness to the original leaf

Currently `faithfulness()` checks `sim(merged, source_a)` and `sim(merged, source_b)` where `source_a` is the result of the previous merge. Each step is faithful to its parent but the chain drifts.

Fix: preserve the original input text and add a check against it at each merge round. If `sim(merged_round_N, original_input) < threshold`, the loop stops.

This naturally caps merge depth without an arbitrary `max_rounds` or char limit. The deeper the chain, the harder it is to stay faithful to the original, so the loop self-terminates. Cost: one extra embedding call per merge round (3 texts), cheap compared to the LLM merge call.

## 6. Two-tier merge: dedup vs summary

Currently `merge_texts()` always tries to preserve all facts from both sources. This is correct for near-duplicates but wrong for related-but-distinct topics.

Distinguish two modes:
- **Dedup merge** (rounds 1-2): current behavior. Combine near-identical chunks, discard redundancy.
- **Summary merge** (round 3+): produce a short topic summary / index node instead of a faithful combination. The merged text becomes a navigational hub: "Knowledge about Python style: Zen of Python, PEP 8, type annotations, list comprehensions." Originals hold the specifics.

Summary nodes are short and abstract, so they do not match aggressively with the next incoming fact. This stops the snowball naturally.

## 7. Richer `should_merge` output

`should_merge()` currently returns `bool`. The LLM already sees both chunks. Change the prompt to return one of `{merge, related, supersedes, distinct}` at no extra cost (same single LLM call).

- `merge`: proceed as today.
- `related`: skip merge, emit `related_to` edge.
- `supersedes`: deactivate old chunk, activate new one, no combined text. Handles the stale-facts problem (old version merged with new version into contradictory blob).
- `distinct`: skip entirely.

This extracts more information from a call we are already making. Even before implementing the full edge table, `supersedes` alone would prevent contradictory merges.

## Implementation order

1. **Expose `original_only`** -- trivial wiring, immediate value.
2. **Depth/leaf count in recall** -- cheap signal, agent can make informed drill-down decisions.
3. **Leaf abstracts in recall_detail** -- collapse multi-hop traversal.
4. **Record rejection edges** -- capture horizontal relationships already discovered by the merge loop.
5. **Anchor faithfulness to original leaf** -- natural depth cap, no arbitrary limits.
6. **Two-tier merge** -- bigger change, high payoff.
7. **Richer should_merge output** -- supersedes detection, more edge types.

Steps 1-3 are retrieval-only changes, no risk to the save path. Step 4 is additive (new table, inserts at rejection points). Steps 5-7 modify the merge loop.

## Exposing connections to the agent

Four options, not mutually exclusive:

**A. Invisible search expansion.** After top-k retrieval, follow edges 1-hop, re-rank the expanded set, return the best results. The agent sees better results without knowing why. Zero tool changes, zero extra tokens in output. Handles the common case where the agent just needs relevant facts.

**B. Hint in `recall`.** Add a connection count per result: `[1] #42 (score=0.87, +3 related) "Python style"`. The agent knows there is more to explore but has to drill down. Cheap — one number per result.

**C. Show in `recall_detail`.** When the agent requests detail on a chunk, list its connections alongside the merge tree / leaf abstracts. Natural extension — `recall_detail` is already the "tell me more" tool. No new tools to learn.

**D. New `recall_related` tool.** Explicit graph traversal. Most flexible, but adds cognitive load — the agent has to decide when to use it, and each tool call costs tokens and a decision.

**Recommendation: A + C.** Search expansion handles the common case silently; `recall_detail` shows connections when the agent explicitly investigates. Agents don't think in graphs — they think "I need to know X." Option A gives that for free. Option D is overkill unless the agent needs to explore connections independently of a search query.

## Interaction with graph_memory.md

This proposal is a pragmatic subset of the graph memory idea. It reuses the `chunk_edges` table from that design but starts with a single edge type (`related_to`) emitted from existing rejection events, rather than the full classify-and-link paradigm. The graph_memory.md proposal remains the long-term direction; this is the incremental path to get there.

Key difference: graph_memory.md proposes replacing the merge loop with classify-and-link. This proposal keeps merging but makes it safe by ensuring originals are discoverable and relationships are recorded.
