# Session Context Transfer

How to make transitions between agent instances feel seamless.

## Types of context

- **Declarative** — facts, conclusions, reflections. Handled well by semantic memory (remember/recall). Survives merging.
- **Procedural** — what was I doing, my plan, unfinished work. Not captured by memory today. This is the main gap.
- **Episodic** — what happened, in what order. Logs have it but too verbose to inject.

## Current mechanisms

| Mechanism | Persists | Cost to next instance |
|-----------|----------|-----------------------|
| Semantic memory | Declarative knowledge | Recall queries (cheap) |
| Workspace files | Artifacts | ls + read (moderate) |
| JSONL resume | Full conversation | Full replay (expensive) |
| System prompt | Instance number | Zero |

## Proposed: session recap

Chunk and summarize the previous session's log, then rank chunks by importance. Inject the top chunks into the next instance's system prompt (token-budgeted) and write the rest to a handoff file.

### Pipeline

1. Chunk the session log (existing chunker).
2. Summarize each chunk (existing summarizer).
3. Score each chunk for importance.
4. Build recap from top-scored chunks up to a token budget.
5. Inject into system prompt; write full recap to `_handoff.md` in workspace.

## Scoring: novelty x centrality

Two independent signals, both principled:

### Novelty (relative to existing memory)

How much new information does this chunk add beyond what's already in long-term memory?

- **Information-theoretic basis**: `I(x) = -log P(x)`. Less probable = more informative. IDF in BM25 is this for terms.
- **Embedding approximation**: `novelty = 1 - max(cos_sim(chunk, memory_i))`. Nearest-neighbor density estimate in embedding space. Low density = high surprise.
- **Bayesian surprise**: KL(posterior || prior). How much does this chunk shift the knowledge distribution?
- **Compression-based**: NCD (Normalized Compression Distance). Novel if it can't be compressed given existing knowledge.

Simplest practical version: embedding distance from nearest memory. Already an approximation of information content via density estimation.

### Centrality (within the session)

How representative is this chunk of the session's main threads?

- **TextRank/LexRank**: Build similarity graph of chunks, run PageRank. Central chunks are hubs connected to many others.
- **MMR (Maximal Marginal Relevance)**: Iteratively pick chunks maximizing `lambda * relevance - (1 - lambda) * redundancy`.
- **TF-IDF within session**: Terms frequent here but rare globally characterize what this session was about.

### Combined score

```
score = novelty(chunk, memories) * centrality(chunk, session_chunks) * recency(position)
```

- **Novelty** filters out what memory already has.
- **Centrality** filters out noise and tangents.
- **Recency** biases toward the end of session (what was just happening, unfinished work).

## Action-based signals

Tool calls carry importance weight:
- `write`, `edit`, `remember` = consequential (changed state).
- `read`, `ls`, `recall` = observational.
- Chunks with more consequential actions are likely more important.

Can be a multiplier: `action_weight = 1 + alpha * consequential_tool_count`.

## Open questions

- How large should the system prompt recap budget be? (500 tokens? 1000?)
- Should the handoff be agent-written (richer) or system-generated (reliable)?
- Should we store the recap as a special memory entry (tagged, searchable) or just a file?
- How to handle Ctrl+C interruptions — no time for summarization?
