# Semantic Memory

Three-layer system for persistent agent knowledge.

## Layers

1. **MemoryStore** (`memory.py`): SQLite + numpy embeddings. Stores chunks with text, abstract, summary, embedding, and merge lineage (source_a/source_b).
2. **Semantic memory** (`semantic_memory.py`): `save_memory()` runs a fixed-point merge loop — find most similar chunk, merge if appropriate, repeat until no more merges or specificity drops too low.
3. **Memory tools** (`memory_tools.py`): Agent-facing `remember`, `recall`, `recall_detail`.

## Search

Hybrid search via Reciprocal Rank Fusion (K=60):
- Embedding cosine similarity (dense retrieval).
- FTS5 BM25 keyword matching (sparse retrieval).

## Merge decisions (`merge_llm.py`)

Three-tier similarity thresholds:
- `> 0.9`: always merge (no LLM call).
- `< 0.2`: never merge.
- Middle zone: LLM decides. Temperature=0, seed=42 for reproducibility.

## Specificity

`specificity = 1 - neighbor_count / total_count` where neighbors are chunks above a cosine similarity threshold. Prevents over-merging: if a merged chunk would be too generic (below threshold 0.3), the merge is rejected.

## Deduplication

Exact-text duplicates are detected via `find_exact()` before saving. The stored text form (including role prefix) is used for comparison.
