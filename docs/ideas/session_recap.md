# Idea: Session Recap

**Status: not implemented.**

Automatically generate a ranked summary of the previous session to inject into the next instance's context.

## Problem

Procedural context (what was I doing, my plan, unfinished work) is lost between instances. Semantic memory captures declarative knowledge but not in-progress state.

## Proposed pipeline

1. Chunk the previous session's JSONL log (existing chunker).
2. Summarize each chunk (existing summarizer).
3. Score each chunk for importance (see below).
4. Build recap from top-scored chunks up to a token budget.
5. Inject into system prompt; write full recap to `_handoff.md` in workspace.

## Scoring: novelty x centrality x recency

```
score = novelty(chunk, memories) * centrality(chunk, session_chunks) * recency(position)
```

- **Novelty**: embedding distance from nearest existing memory. Filters out what's already remembered.
- **Centrality**: TextRank-style connectedness within session chunks. Filters out noise.
- **Recency**: positional weight biasing toward end of session (unfinished work).

### Action-based signal

Tool calls as importance multiplier:
- `write`, `edit`, `remember` = consequential (changed state).
- `read`, `ls`, `recall` = observational.
- `action_weight = 1 + alpha * consequential_tool_count`.

## Open questions

- Token budget for system prompt recap? (500? 1000?)
- Agent-written handoff (richer) vs system-generated (reliable)?
- Store recap as a tagged memory entry or just a workspace file?
- Handle Ctrl+C interruptions (no time for summarization)?
