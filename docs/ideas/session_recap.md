# Idea: Session Recap

**Status: not implemented.**

Automatically generate a ranked summary of the previous session to inject into the next instance's context.

## Problem

Procedural context (what was I doing, my plan, unfinished work) is lost between instances. Semantic memory captures declarative knowledge but not in-progress state.

## Proposed pipeline

1. **Pre-process** the JSONL log: collapse tool calls + results into concise natural language.
2. Chunk the processed log (existing chunker).
3. Summarize each chunk (existing summarizer).
4. Score each chunk for importance (see below).
5. Build recap from top-scored chunks up to a token budget.
6. Inject into system prompt; write full recap to `_handoff.md` in workspace.

## Pre-processing: tool call collapsing

Raw JSONL logs contain structured tool calls and verbose results (file contents, recall outputs). These are noisy for the summarizer and waste tokens. Before chunking, collapse each tool call + result pair into a concise natural language line.

### Why pre-process rather than post-filter

Alternative: chunk raw logs and down-weight tool-heavy chunks at scoring time. This is simpler but the summarizer still wastes effort compressing file contents that are irrelevant to the recap. Pre-processing gives the summarizer a clean narrative to work with.

### Collapsing rules by tool type

| Tool | Category | Collapsed form |
|------|----------|----------------|
| `read` | Observational | "Read foo.txt (15 lines)." |
| `ls` | Observational | "Listed directory: foo/, bar.txt" |
| `recall` | Observational | "Recalled 3 memories about X." (from query) |
| `recall_detail` | Observational | "Retrieved full text of memory #N." |
| `edit` | Consequential | "Edited foo.txt: replaced X with Y." |
| `write` | Consequential | "Wrote bar.txt (20 lines)." |
| `remember` | Consequential | "Remembered: 'one-sentence abstract'." (keep full abstract) |
| `ask` | Consequential | "Asked user: 'question'. Response: 'answer'." |
| `status` | Meta | Drop or "Checked status." |
| `done` | Meta | "Finished: 'summary'." |

### Principles

- **Consequential tools** (write, edit, remember, ask) keep more detail — they changed state.
- **Observational tools** (read, ls, recall) collapse to action + brief outcome.
- **Meta tools** (status, done) are minimal or dropped.
- **Tool results** are mostly discarded. The fact that the agent read a file matters; the file contents don't.
- **Assistant reasoning** passes through unchanged — high signal, the summarizer handles it well.

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
