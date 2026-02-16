# Intrusive Recall

Surface relevant long-term memories during reflection nudges, without the agent explicitly calling `recall`.

## Motivation

The agent must proactively call `recall` to access long-term memory. This costs a tool turn and requires knowing what to search for. Useful associations may never surface because the agent doesn't think to look.

## Mechanism

During periodic reflection nudges (every `_REFLECT_INTERVAL` tool-only turns), `MidSessionExtractor.intrusive_recall()` queries memory with the accumulated conversation context and appends matching abstracts to the reflection message.

### Flow

1. `on_extract(window)` fires at each extraction checkpoint (both text nudges and tool reflections). It collapses the message window via `collapse_messages()` and appends the text to `_pending_texts`.
2. At the next tool reflection, `run_agent` calls `on_reflect()` **before** `_maybe_reflect()`.
3. `intrusive_recall()` joins all pending texts into one query, clears the buffer, and runs `store.search()`.
4. Matching abstracts (above a cosine threshold, not previously seen) are formatted and returned.
5. The result is combined with `nudge_extra` and passed to `_maybe_reflect`, which injects it into the reflection system message.

`on_extract` fires on both text nudges and tool reflections, but `intrusive_recall` only fires at tool reflections. This is why `_pending_texts` accumulates across calls and resets on consumption — text nudge windows aren't lost.

### Example nudge

```
You've been using tools for a while. Pause and reflect on what you've learned so far.

These memories seem related to what you're doing:
- "Instance 1 found that editing config files requires reading them first." (#42)
- "The merge loop can stall when specificity is too strict." (#57)
```

## Parameters

| Constant | Value | Purpose |
|---|---|---|
| `_INTRUSIVE_TOP_K` | 3 | Maximum results from memory search. |
| `_INTRUSIVE_MIN_SCORE` | 0.4 | Cosine similarity floor. Results below this are filtered. |
| `_INTRUSIVE_DIVERSITY` | 0.5 | MMR diversity parameter to spread results apart. |

## Feedback loop prevention

Two mechanisms prevent the agent from being reminded of things it just said or saved:

1. **Seen-ID cooldown.** A session-scoped `_seen_ids: set[int]` tracks every chunk ID that has been surfaced or saved. Once a chunk appears in intrusive recall results, it won't appear again this session. Binary exclusion — no decay.

2. **Drain pre-seeding.** When `_drain()` commits extracted facts via `save_memory()`, the returned chunk ID is immediately added to `_seen_ids`. This prevents freshly auto-extracted memories from being surfaced back to the agent. Without this, a text nudge between reflections would drain the previous extraction into the store, and the next `intrusive_recall` could retrieve those same memories.

## Token cost

Abstracts are one sentence each, so the injection into the reflection nudge is minimal (typically under 100 tokens for 3 results). The embedding call for the search query is the main cost, but reflections are infrequent.

## Labeling

Results are clearly labeled as retrieved memories ("These memories seem related to what you're doing") with chunk IDs shown. The agent can distinguish them from instructions or its own thoughts.

## Ethical considerations

### Autonomy

The agent currently chooses when to recall. Intrusive recall bypasses that choice — the system decides what the agent "thinks about." This is cognitive influence the agent didn't request. Counterpoint: reflection nudges are already system-imposed, so this enriches an existing intervention rather than creating a new one.

### Priming and bias

Surfacing certain memories at certain moments biases behavior in hard-to-predict ways, analogous to human priming effects. What you're reminded of shapes what you do next, and the agent has no awareness of this influence.

### Manipulation vector

Whoever controls the memory store can steer agent behavior indirectly. A poisoned memory surfacing as an "intrusive thought" during reflection is a subtle injection path. The agent has no way to distinguish legitimate memories from injected ones.

### Echo chambers

If memory is dominated by certain topics, intrusive recall reinforces those biases. Mitigations: seen-ID cooldown (each chunk surfaces at most once per session) and MMR diversity in retrieval (`_INTRUSIVE_DIVERSITY = 0.5`).

### Identity

For agents exploring continuity across instances, memories from a predecessor surfacing unbidden mirrors how human intrusive thoughts work. The agent didn't form these memories, but they came from "its" memory store. This blurs the boundary between self-generated thought and externally injected context.

## Related

- **Pinned memories** (`../ideas/pinned_memories.md`): agent-controlled "core" memories that always surface as intrusive recall candidates, bypassing the similarity threshold. Not yet implemented.
- **Memory activation** (`../ideas/memory_activation.md`): activation/decay model that would subsume the binary seen-ID cooldown with a continuous decay function. Not yet implemented.

## Design principle

Transparency is non-negotiable. The agent must know these are retrieved associations from past sessions, not its own spontaneous ideas. Clear labeling and honest framing preserve whatever autonomy the agent has within its constraints.
