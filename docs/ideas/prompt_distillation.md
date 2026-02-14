# Prompt Distillation from Text Corpora

## Goal

Convert a corpus of texts (session logs, books, articles, any written material) into a pool of diverse reflective prompts for the synthetic diversity injection mechanism (see `synthetic_diversity.md`).

## Pipeline

### 1. Extractive pass (free, no LLM calls)

Pull content directly from the corpus:

- **Questions**: sentences ending in `?` that are open-ended (filter out yes/no).
- **Strong claims**: sentences with opinion markers ("I believe", "the key insight", "surprisingly", "the problem is").
- **Diverse excerpts**: embed all sentences, cluster via k-means, pick the centroid sentence from each cluster.

Reuses existing infrastructure: `chunk_turns`, `compact_chunks`, `get_embeddings`.

### 2. LLM distillation pass (one-time batch cost)

For chunks not covered by extraction:

1. Chunk the corpus using existing chunker.
2. One-shot LLM call per chunk with a prompt like: "Generate 3 open-ended reflective prompts inspired by this text. Prompts should provoke thought, not summarize. Write as questions or challenges directed at a thinking agent."
3. Reuses the `summarize_chunks` pattern with a different system prompt.

### 3. Deduplication and diversity selection

1. Embed all candidate prompts.
2. Greedy selection: pick the prompt furthest from already-selected set, repeat until pool is full.
3. This maximizes minimum pairwise distance — the pool covers the broadest possible range.

### 4. Output

Static JSON file shipped with the agent:

```json
[
  "What assumptions are you making right now?",
  "What would change if you approached this from the opposite direction?",
  "What have you learned that surprised you?",
  ...
]
```

## Source Material

The corpus does not need to be topically related to the agent's tasks. Less related material produces better diversity injection. Good sources:

- Agent session logs (captures the agent's own patterns, helps break out of them).
- Philosophy, science writing, fiction (maximally diverse from typical agent tasks).
- Collections of aphorisms, koans, or thought experiments.

## CLI Tool (future)

```bash
mindloop-distill corpus.txt --output prompts.json --pool-size 100
mindloop-distill session_logs/ --format jsonl --output prompts.json
```

Options:
- `--extract-only` — skip LLM pass, extractive only.
- `--pool-size N` — target number of prompts after diversity selection.
- `--format` — input format (txt, md, jsonl).
- `--model` — model for LLM distillation pass.

## Open Questions

- Optimal pool size? Too small = repetitive, too large = diluted quality. Likely 50-200.
- Should the pool be static or grow over time as the agent encounters new material?
- Weight prompts by effectiveness? Track which prompts actually break loops vs get ignored.
