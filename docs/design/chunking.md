# Chunking

Turns raw text (JSONL logs, markdown files) into semantically coherent chunks for summarization and memory storage.

## Data model

```
Turn(timestamp, role, text)   — one message or reasoning step.
Chunk(turns: list[Turn])      — a group of related turns.
  .text                       — "role: text\n..." for all turns.
  .time_range                 — "HH:MM:SS - HH:MM:SS".
```

## Pipeline stages

### 1. Parse

Two input formats, same output type (`list[Turn]`).

**JSONL logs** (`parse_turns`): each line becomes a turn. Reasoning tokens emit a separate `"Bot thinking"` turn before the main content. Role mapping: `user → You`, `assistant → Bot`, `tool → Tool`.

**Markdown files** (`parse_turns_md`): split on headings (`# ...`). Each section becomes a `"Doc"` turn. File name is emitted as the first turn for context.

### 2. Chunk (`chunk_turns`)

Group turns into chunks by splitting on blank lines (`\n\n`) within turn text.

- First paragraph extends the current chunk.
- Each subsequent paragraph starts a new chunk.
- Turns without blank lines stay in the current chunk.

This produces many small, topically focused chunks. The assumption is that authors use blank lines to separate ideas.

### 3. Compact (`compact_chunks`)

Merge undersized chunks into neighbors. Default minimum: 80 characters.

- Forward pass: if the previous chunk is below threshold, absorb the current chunk into it.
- Trailing pass: if the last chunk is still small, merge it back into the previous.

This eliminates fragments (short paragraphs, section breaks) that are too small to embed or summarize meaningfully.

### 4. Embed + merge (`merge_chunks`)

Optional stage requiring an embedding API call. Used by `mindloop-chunk --embed` and the recap pipeline.

**Embedding**: one batched `get_embeddings` call for all chunk texts. Cached per-text to avoid redundant API calls on re-runs.

**Merge algorithm**: fixed-point iteration over adjacent pairs.

Each pass:
1. Compute cosine similarity between consecutive chunk embeddings.
2. Set threshold: `mean(similarities) - 0.5 * std(similarities)`.
3. Walk left to right. Merge adjacent chunks if:
   - similarity >= threshold, OR similarity >= 0.8 (always merge), AND
   - merged text would not exceed `max_chunk_chars` (default 8192, ~2048 tokens).
4. Update embeddings: incremental weighted average of merged chunks.
5. Repeat until no merges occur (fixed point).

**Properties**:
- Only considers adjacent pairs (O(n) per pass, not all-pairs).
- Adaptive threshold: adjusts to the similarity distribution of each document.
- The 0.8 hard floor ensures obviously related chunks always merge regardless of distribution.
- Size cap prevents runaway merges into single mega-chunks.

### 5. Summarize (`summarize_chunks`)

Each chunk is summarized by an LLM into:
- **Abstract**: one-sentence TL;DR.
- **Summary**: 2–4 sentence expanded overview.

Not part of `chunker.py` but always follows chunking in practice. See `summarizer.py`.

## Pipeline variations by caller

| Caller | Parse | Compact | Embed+Merge | Summarize |
|--------|-------|---------|-------------|-----------|
| `mindloop-chunk` | yes | yes | optional (`--embed`) | optional (`--summarize`) |
| `mindloop-build` | yes | yes | no | yes (then save to memory) |
| `mindloop-recap` | collapse_messages | yes | yes | yes |

The build pipeline skips chunk-level merging because `semantic_memory.py` performs its own merge loop at storage time (see `docs/design/memory.md`).

The recap pipeline always embeds+merges because collapsed tool calls produce many small turns that would otherwise generate too many summarization calls.
