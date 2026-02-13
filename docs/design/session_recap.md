# Session Recap

Implementation: `mindloop/recap.py`, `mindloop/cli/recap.py`.

Automatically generate a ranked summary of the previous session to inject into the next instance's context.

## Problem

Procedural context (what was I doing, my plan, unfinished work) is lost between instances. Semantic memory captures declarative knowledge but not in-progress state.

## Implemented pipeline

1. **Pre-process** the JSONL log: collapse tool calls + results into concise natural language (`collapse_messages`).
2. **Chunk** the processed log (`chunk_turns` + `compact_chunks`).
3. **Embed + merge** semantically similar adjacent chunks (`get_embeddings` + `merge_chunks`). Typically reduces ~100+ raw chunks to ~5–15.
4. **Summarize** each merged chunk (`summarize_chunks`).
5. **Score** by recency: linear positional bias toward end of session.
6. **Select** top-scored summaries up to a token budget (default 1000 tokens).
7. **Inject** into system prompt and write to `_recap.md` in workspace.

### Timing

- **Shutdown**: generated in the `finally` block after the agent loop ends (including Ctrl+C). Written to `sessions/<name>/workspace/_recap.md`.
- **Startup**: loaded from `_recap.md` if it exists. If not (first instance or crash), generated on the fly from the latest JSONL log.

### CLI

`mindloop-recap <logfile>` generates a recap on demand. Supports `--budget`, `--model`, `-o` flags.

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
| `edit` | Consequential | "Edited foo.txt." |
| `write` | Consequential | "Wrote bar.txt (20 lines)." |
| `remember` | Consequential | "Remembered: 'one-sentence abstract'." (keep full abstract) |
| `ask` | Consequential | "Asked user: 'question'. Response: 'answer'." |
| `status` | Meta | Dropped. |
| `done` | Meta | "Finished: 'summary'." |

### Principles

- **Consequential tools** (write, edit, remember, ask) keep more detail — they changed state.
- **Observational tools** (read, ls, recall) collapse to action + brief outcome.
- **Meta tools** (status, done) are minimal or dropped.
- **Tool results** are mostly discarded. The fact that the agent read a file matters; the file contents don't.
- **Assistant reasoning** passes through unchanged — high signal, the summarizer handles it well.

## Scoring: v1 vs future

### Current (v1): recency only

Linear positional bias: `score = (i + 1) / len(summaries)`. Simple, reliable, biases toward unfinished work at end of session.

### Future: novelty x centrality x recency

```
score = novelty(chunk, memories) * centrality(chunk, session_chunks) * recency(position)
```

- **Novelty**: embedding distance from nearest existing memory. Filters out what's already remembered.
- **Centrality**: TextRank-style connectedness within session chunks. Filters out noise.
- **Recency**: positional weight biasing toward end of session (unfinished work).

### Future: action-based signal

Tool calls as importance multiplier:
- `write`, `edit`, `remember` = consequential (changed state).
- `read`, `ls`, `recall` = observational.
- `action_weight = 1 + alpha * consequential_tool_count`.

## Why not store recaps in semantic memory

Recaps are kept as workspace files, not saved into semantic memory. Reasons:

- **Specificity**. Recaps are broad summaries with many semantic neighbors. They'd score low on the specificity metric and risk being rejected by `save_memory` or diluting recall results.
- **Merge pollution**. The `save_memory` merge loop could blend procedural state ("I was editing chunker.py") with declarative knowledge about chunker.py's architecture. Those are different kinds of knowledge.
- **Staleness**. Procedural context ("in the middle of refactoring X") goes stale once X is done. Declarative memories don't have this problem.
- **Redundancy**. The agent already calls `remember` during the session for anything it considers important. The recap mostly captures transient procedural state.

If cross-session recall becomes valuable (recalling what happened N sessions ago, not just the previous one), revisit this. Possible approaches: store chunk summaries individually with a `[session N]` tag, or use `store.save()` directly with no merge loop.

## Resolved questions

- **Token budget**: 1000 tokens default, configurable via `--budget`.
- **Generation method**: system-generated (reliable, automatic).
- **Storage**: workspace file (`_recap.md`), not a memory entry.
- **Ctrl+C**: handled via `finally` block in agent CLI.
