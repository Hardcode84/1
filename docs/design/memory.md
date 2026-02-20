# Semantic Memory

Three-layer system for persistent agent knowledge.

## Layers

1. **MemoryStore** (`memory.py`): SQLite storage. Stores chunks with text, abstract, summary, and merge lineage (source_a/source_b). Embeddings are computed on demand via `get_embeddings()` rather than stored — this avoids stale vectors when the embedding model changes.
2. **Semantic memory** (`semantic_memory.py`): `save_memory()` runs a fixed-point merge loop — find most similar chunk, merge if appropriate, repeat until the merge gate rejects or no more candidates. Does not handle embeddings directly; `MemoryStore.search()` and `neighbor_score()` compute them internally.
3. **Memory tools** (`memory_tools.py`): Agent-facing `remember`, `recall`, `recall_detail`.

## Search

Hybrid search via Reciprocal Rank Fusion (K=60):
- Embedding cosine similarity (dense retrieval). Chunk texts are embedded on the fly via `get_embeddings()`, which caches per-session by `(model, text)`.
- FTS5 BM25 keyword matching (sparse retrieval).

## Merge decisions (`merge_llm.py`)

Three-tier similarity thresholds:
- `> 0.9`: always merge (no LLM call).
- `< 0.2`: never merge.
- Middle zone: LLM decides. Temperature=0, seed=42 for reproducibility.

## Merge gate

Two-stage check prevents over-merging after each merge:

1. **Leaf faithfulness** (`leaf_faithfulness()` in `memory.py`): embeds the merged text and ALL original leaf texts from both sides of the merge, checks that cosine similarity to each leaf meets a threshold (default 0.7). An `incoming_leaves` accumulator grows across merge rounds so leaf texts are never re-traversed. This naturally caps merge depth — the more leaves, the harder to stay faithful to every one, so the loop self-terminates. Runs before any deactivation — cheap bail-out if the merge drifted from any original fact.
2. **Relative density** (`MemoryStore.neighbor_score()`): deactivates the absorbed chunk, then computes the mean cosine of top-k neighbors for both the candidate's original text and the proposed merged text against the same ambient set. If the merged text's neighbor score exceeds the candidate's by more than a margin (default 0.05), the merge made the neighborhood denser than it already was and is rejected. The absorbed chunk is reactivated on rejection. This adapts to local cluster density — in a dense cluster the candidate already has a high neighbor score, so the merged result having a similarly high score is expected and allowed.

## Edge inheritance

When chunks are absorbed during a merge, their `related_to` edges would become orphaned (pointing at inactive chunks). Instead, `inherit_edges()` repoints them onto the new merged chunk with a re-scored cosine similarity (one batched `get_embeddings` call per merge round). Internal edges between the absorbed chunks are dropped. In cascading merges, each round inherits from the previous round's result, so edges propagate transitively without deep lineage walks.

### Why mean neighbor score

The old `specificity()` computed `1 - neighbor_count / total_count` using a fixed cosine similarity threshold. This had two problems:

- **O(n) embedding cost.** Every call embedded all active chunks to count neighbors above the threshold. As memory grew, this became the dominant cost of each merge iteration.
- **Brittle threshold.** A single cosine cutoff decided "neighbor or not" — small embedding model differences could flip chunks across the boundary, producing unstable merge behavior.

`neighbor_score()` replaces both with a single `search()` call that already exists in the hot path. It reuses the hybrid cosine + BM25 RRF pipeline, so the score reflects both semantic and keyword similarity. Taking the mean of top-k results (default k=3) gives a smooth, continuous signal. Rather than comparing against an absolute threshold (which breaks in dense clusters), the merge gate compares the merged text's neighbor score against the candidate's — only the *increase* matters. A margin of 0.05 means merging is only rejected when the result is notably more generic than the input it replaced, regardless of how dense the surrounding cluster already was.

## Deduplication

Exact-text duplicates are detected via `find_exact()` before saving. The stored text form (including role prefix) is used for comparison.
