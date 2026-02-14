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

1. **Faithfulness** (`faithfulness()` in `memory.py`): embeds the merged text and both source texts, checks that cosine similarity to each source meets a threshold (default 0.7). Runs before any deactivation — cheap bail-out if the merge drifted from a source.
2. **Neighbor score** (`MemoryStore.neighbor_score()`): deactivates the absorbed chunk, then calls `search()` and returns the mean score of the top-k results. If the score exceeds a threshold (default 0.6), the merged text is too close to existing memories and the merge is rejected. The absorbed chunk is reactivated on rejection.

### Why mean neighbor score

The old `specificity()` computed `1 - neighbor_count / total_count` using a fixed cosine similarity threshold. This had two problems:

- **O(n) embedding cost.** Every call embedded all active chunks to count neighbors above the threshold. As memory grew, this became the dominant cost of each merge iteration.
- **Brittle threshold.** A single cosine cutoff decided "neighbor or not" — small embedding model differences could flip chunks across the boundary, producing unstable merge behavior.

`neighbor_score()` replaces both with a single `search()` call that already exists in the hot path. It reuses the hybrid cosine + BM25 RRF pipeline, so the score reflects both semantic and keyword similarity. Taking the mean of top-k results (default k=3) gives a smooth, continuous signal: a high mean means the merged chunk is surrounded by close duplicates and further merging would produce something too generic. The threshold (default 0.6) compares directly against the RRF score scale that search already normalizes to [0, 1], so it's easy to reason about and tune.

## Deduplication

Exact-text duplicates are detected via `find_exact()` before saving. The stored text form (including role prefix) is used for comparison.
