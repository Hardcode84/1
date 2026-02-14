# Idea: Automatic Memory Extraction

**Status: not implemented.**

Replace the fully agent-driven `remember` workflow with a hybrid approach: keep explicit `remember` for high-signal moments, add automatic extraction as a safety net that captures everything else.

## Problem

Agents must explicitly call `remember` to persist knowledge. In practice this is unreliable:

- **Inconsistent use.** One agent used `remember` exclusively for end-of-session reports and never during active work. Others never used it at all.
- **Lost in the middle.** LLMs exhibit a U-shaped attention curve — strong attention to the beginning and end of context, weak in the middle (Liu et al., 2023; MIT 2025). By turn 50, events from turns 15-35 are in a blind spot. The agent literally cannot notice important things that happened mid-session.
- **Competing priorities.** The agent is both the worker and the curator of its own knowledge. During focused task execution, pausing to formulate a `remember` call is a distraction that costs tokens and breaks flow.
- **No coverage guarantee.** Even a diligent agent will miss things. There is no fallback — if `remember` isn't called, the knowledge is lost when the session ends.

## Prior art

### Mem0 (2025)

Selective fact extraction beats conversation summarization. Their pipeline: (1) ingest latest exchange + rolling summary + recent messages, (2) LLM extracts candidate memories, (3) each candidate is compared to existing memory via vector search, (4) LLM picks one of four operations: add / update / delete / noop. Reports 26% accuracy uplift over OpenAI's memory, 90% token reduction vs full-context.

Key insight: **extract facts, don't compress conversations.** Summarization loses specificity; extraction preserves actionable detail.

### A-Mem (NeurIPS 2025)

Zettelkasten-inspired structured notes with context, keywords, tags. When a new memory arrives, it triggers updates to related existing memories — the network refines itself. ~1200 tokens per operation.

Key insight: **memory evolution.** New information doesn't just get added — it reshapes the context of what's already stored.

### MemGPT / Letta (2023)

OS-inspired tiered memory with explicit paging between context and archival storage. The LLM manages its own memory via tools.

Key insight: **context as scarce resource.** But still relies on the agent to manage it — same fundamental problem we're trying to solve.

### Generative Agents (Park et al., 2023)

All observations are stored automatically. Retrieval is scored by `recency * importance * relevance`. Importance is LLM-scored at creation time.

Key insight: **store everything, score later.** Extraction is not selective — filtering happens at retrieval time.

## Proposed design

### Two extraction triggers

**1. Mid-session extraction (periodic).**

Runs every N tool turns (piggyback on the existing reflection interval, currently 5). A non-agentic LLM call reads the last K turns and extracts candidate facts.

- The agent never sees this call — no token cost in its context window.
- Each extraction window is short (K turns), sidestepping the lost-in-the-middle problem. Position bias doesn't matter when the context is small.
- Candidates are fed into `save_memory`, which handles deduplication and merging with existing knowledge.

**2. Post-session extraction (on shutdown).**

Runs after the agent calls `done` or hits its token budget — in the `finally` block of `cli/agent.py`, alongside the existing `_generate_session_recap`. Processes the full conversation in windowed chunks.

- Reuses the existing recap infrastructure: `collapse_messages` → `chunk_turns` → `compact_chunks` → summarize.
- But instead of writing a text file, each chunk summary is fed into `save_memory`.
- The recap file can still be generated as before (for the next instance's system prompt), but now the knowledge also enters the persistent semantic memory.

### Extraction pipeline

```
conversation turns (window of K turns)
    │
    ▼
collapse_messages()          # existing: collapse tool calls
    │
    ▼
LLM extraction call          # new: extract candidate facts
    │
    ▼
candidate facts (list of text + abstract pairs)
    │
    ▼
for each candidate:
    save_memory()            # existing: dedup, merge, faithfulness, neighbor score
```

The extraction LLM call is the only new component. Everything downstream exists.

### Extraction prompt

The extraction call should produce discrete facts, not summaries. Each fact should be self-contained (understandable without the surrounding conversation).

```
Given this conversation excerpt, extract facts worth remembering long-term.
Each fact should be a single self-contained statement.
Skip procedural details (tool calls, file listings) — focus on conclusions,
decisions, discoveries, preferences, and open questions.

Respond as a JSON array of objects:
[{"text": "...", "abstract": "one-sentence label"}, ...]

Return an empty array if nothing is worth extracting.
```

The "return empty array" instruction is critical — most windows will have nothing worth saving. The pipeline must be comfortable producing nothing.

### Interaction with explicit `remember`

Explicit `remember` stays. It serves a different purpose:

- **Explicit** = high-confidence, agent-curated. The agent recognized something important in real time.
- **Automatic** = safety net, system-curated. Catches what the agent missed.

Deduplication is handled by `save_memory` — if the agent already remembered a fact, the automatic extraction will either find an exact match (`find_exact`) or merge with the existing memory (high similarity → auto-merge). No special coordination needed.

### Cost model

| Trigger | Frequency | Input size | Estimated cost |
|---------|-----------|------------|----------------|
| Mid-session | Every 5 tool turns | ~K turns (~2-4K tokens) | ~$0.001-0.003 per call |
| Post-session | Once | Full session in windows | ~$0.01-0.05 per session |

Mid-session extraction uses a cheap model (same as summarizer). Post-session can afford a slightly better model since it runs once.

### Configuration

```python
# In cli/agent.py or a config object:
extract_interval: int = 5       # Every N tool turns (0 = disabled).
extract_window: int = 15        # Last K turns per extraction.
extract_model: str = "..."      # Model for extraction calls.
post_session_extract: bool = True  # Extract on shutdown.
```

## Where it hooks in

### Mid-session

In `agent.py`, the reflection nudge fires at `consecutive_tool_turns % _REFLECT_INTERVAL == 0`. Extraction runs at the same cadence but is a separate, non-agentic call. It reads from `messages` (which the agent loop already maintains) but does not append to it.

The extraction call must run in the background or be fast enough not to block the agent loop. Options:

- **Synchronous, cheap model.** Simplest. A small model on a short window takes <1s.
- **Background thread.** Non-blocking but adds complexity (thread-safe `save_memory` access). Since SQLite with WAL mode supports concurrent reads, and writes are serialized through `MemoryStore.transaction()`, this should work.

### Post-session

In `cli/agent.py:main()`, the `finally` block already calls `_generate_session_recap`. Add `_extract_session_memories` at the same level:

```python
finally:
    mt.close()
    _generate_session_recap(paths, jsonl_path, summarizer_model)
    _extract_session_memories(paths, jsonl_path, summarizer_model)
```

Note: `mt.close()` runs first, so `_extract_session_memories` needs its own `MemoryStore` instance. This is fine — SQLite handles sequential access cleanly.

## Mitigating lost-in-the-middle

The windowed extraction approach directly addresses the U-shaped attention problem:

- Each extraction call sees at most K turns (~15). Position bias is negligible in short contexts.
- Mid-session extraction slides the window forward, so every part of the conversation gets a turn near the beginning of some extraction window.
- Post-session extraction chunks the full conversation and processes each chunk independently, same principle.

This is more robust than asking the agent to remember things from the middle of its own long context.

## Risks and mitigations

### Noise

Automatic extraction may save low-value facts that the agent would have filtered out. The merge gate (faithfulness + neighbor score) provides some defense — generic or redundant extractions get rejected. But the extraction prompt must aggressively filter ("return empty array if nothing is worth extracting").

### Token cost

Mid-session extraction adds ~$0.001-0.003 per interval. For a 100-turn session with interval=5, that's ~$0.02-0.06. Modest, but not free. The `extract_interval=0` option disables it entirely.

### Stale context

Mid-session extraction sees raw turns, not the agent's internal reasoning about them. It may extract a fact that the agent later corrected or abandoned. Mitigation: post-session extraction sees the full arc and can produce higher-quality facts that supersede mid-session ones (via merge).

### Memory pollution

Over many sessions, automatic extraction could inflate the memory store with marginal facts, degrading search quality. Mitigations:
- Importance scoring at extraction time (see `importance_scoring.md`).
- Memory activation / decay (see `memory_activation.md`) — unused extractions naturally fade.
- Aggressive neighbor score threshold — if an extraction is too similar to existing memories, it's rejected.

## Open questions

- Should mid-session extraction share the same `MemoryStore` instance as the agent's explicit `remember` calls, or use a separate connection? Sharing is simpler but requires thread safety.
- Should extracted memories be tagged as "auto-extracted" vs "agent-curated"? This would let retrieval weight them differently, but adds schema complexity.
- Should the extraction prompt be customizable per agent personality, or is one universal prompt sufficient?
- How does this interact with intrusive recall (`intrusive_recall.md`)? Auto-extracted memories could be surfaced back to the agent in reflection nudges, creating a feedback loop where the system extracts a fact → surfaces it → the agent acts on it → new facts are extracted. This could be powerful or pathological.

## Related

- **Intrusive recall** (`intrusive_recall.md`): surfaces memories during reflection. Automatic extraction fills the memory store that intrusive recall draws from.
- **Importance scoring** (`importance_scoring.md`): could score extraction candidates before saving, filtering low-value ones.
- **Memory activation** (`memory_activation.md`): decay model for auto-extracted memories that turn out to be unused.
- **Coherence detection** (`coherence_detection.md`): could detect contradictions between auto-extracted facts and existing memories.

## References

- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023). https://arxiv.org/abs/2307.03172
- Xu et al., "A-Mem: Agentic Memory for LLM Agents" (NeurIPS 2025). https://arxiv.org/abs/2502.12110
- Mem0, "Building Production-Ready AI Agents with Scalable Long-Term Memory" (2025). https://arxiv.org/abs/2504.19413
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023). https://arxiv.org/abs/2310.08560
- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023). https://arxiv.org/abs/2304.03442
- "Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization" (2024). https://arxiv.org/abs/2406.16008
