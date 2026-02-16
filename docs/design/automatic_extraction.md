# Automatic Memory Extraction

Hybrid approach: explicit `remember` for high-signal moments, automatic extraction as a safety net that captures everything else.

## Problem

Agents must explicitly call `remember` to persist knowledge. In practice this is unreliable: inconsistent use, lost-in-the-middle attention bias, competing priorities during focused work, and no coverage guarantee.

## Two extraction paths

### Mid-session extraction (periodic)

`Extractor` in `agent.py` tracks a checkpoint into the message list. At each reflection interval (`_REFLECT_INTERVAL = 5` tool turns) and on text-only nudges, it sends `messages[checkpoint:]` to the `on_extract` callback and advances the checkpoint.

`MidSessionExtractor` in `cli/agent.py` receives those windows:

1. Drains the previous extraction (waits for commit so memories are available for recall within the session).
2. Submits `extract_window(messages, model)` to a single-thread `ThreadPoolExecutor`.
3. On drain: for each fact, checks `_is_duplicate()` (cosine search, `DEDUP_THRESHOLD = 0.7`) before calling `save_memory()`.

The agent never sees the extraction call — no token cost in its context window. Each window is short, sidestepping lost-in-the-middle bias.

### Standalone extraction (offline)

Two CLI tools for batch extraction from JSONL logs:

- **`mindloop-extract`** (`cli/extract.py`): Runs `extract_session()` on a single log file. Supports `--dry-run`, `--workers`, `--verbose`.
- **`mindloop-rebuild`** (`cli/rebuild.py`): Full DB rebuild — replays both `remember` calls and extraction from logs. Supports `--session`, `--all`, `--dry-run`.

There is no automatic post-session extraction at shutdown. The `finally` block in `cli/agent.py` calls `mid_extract.finish()` (drains in-flight extraction) then generates the session recap. Full-session extraction is a manual step via the CLI tools.

## Extraction pipeline

### Short windows (`extract_window`)

Used by mid-session extraction and `mindloop-rebuild`:

```
messages (window since checkpoint)
    → collapse_messages()      # collapse tool calls
    → join turns as text
    → extract_facts()          # single LLM call
    → list of {text, abstract, summary}
```

### Full sessions (`extract_session`)

Used by `mindloop-extract`:

```
messages (full session)
    → collapse_messages()
    → chunk_turns() → compact_chunks()
    → merge_chunks()           # embed + merge similar chunks
    → extract_facts() per chunk (parallelized, with cross-chunk context)
    → save_memory() per fact
```

Cross-chunk context: the tail (`CONTEXT_CHARS = 200`) of each chunk is passed to the next extraction call so the LLM can resolve references spanning chunk boundaries.

## Extraction prompt

The prompt (`_SYSTEM_PROMPT` in `extractor.py`) writes from the assistant's perspective in first person. Returns a JSON array of `{text, abstract, summary}` objects.

Extracts: decisions and rationale, user preferences, discoveries about tools/APIs/code, conclusions useful in a different session.

Skips: greetings, filler, meta-talk, session-specific state (token counts, file listings), facts obtainable by reading files, vague observations, self-referential statements about remembering.

Deduplicates within the window: "if multiple parts describe the same fact, produce ONE consolidated entry." Prefers fewer, higher-quality entries. Returns empty array when nothing is worth keeping.

On malformed JSON, retries once by feeding the bad output back. Strips markdown code fences before parsing.

## Interaction with explicit `remember`

Explicit `remember` stays for high-confidence, agent-curated moments. Automatic extraction is the safety net. Deduplication is two-layered:

1. `MidSessionExtractor._is_duplicate()`: cosine search with `DEDUP_THRESHOLD = 0.7` before saving.
2. `save_memory()`: `find_exact()` for identical text, then merge gate (faithfulness + neighbor score) for similar content.

## Prior art

- **Mem0 (2025)**: Extract facts, don't compress conversations. 26% accuracy uplift over OpenAI's memory.
- **A-Mem (NeurIPS 2025)**: Zettelkasten-inspired structured notes with memory evolution.
- **MemGPT / Letta (2023)**: OS-inspired tiered memory. Context as scarce resource, but still agent-managed.
- **Generative Agents (Park et al., 2023)**: Store everything, score later. Retrieval scored by recency * importance * relevance.

## Open questions

- Should extracted memories be tagged as "auto-extracted" vs "agent-curated"? Would let retrieval weight them differently.
- How does this interact with intrusive recall? Auto-extracted memories surfaced in reflection nudges could create a feedback loop.

## References

- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023). https://arxiv.org/abs/2307.03172
- Xu et al., "A-Mem: Agentic Memory for LLM Agents" (NeurIPS 2025). https://arxiv.org/abs/2502.12110
- Mem0, "Building Production-Ready AI Agents with Scalable Long-Term Memory" (2025). https://arxiv.org/abs/2504.19413
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023). https://arxiv.org/abs/2310.08560
- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023). https://arxiv.org/abs/2304.03442
