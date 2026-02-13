# Idea: Intrusive Recall

**Status: not implemented.**

Surface relevant long-term memories during reflection nudges, without the agent explicitly requesting recall.

## Motivation

Currently the agent must proactively call `recall` to access long-term memory. This costs a tool turn and requires knowing what to search for. Useful associations may never surface because the agent doesn't think to look.

## Proposed mechanism

During periodic reflection nudges (every N tool-only turns):

1. Embed the last few assistant messages (or a rolling window).
2. Query memory store for top-k matches above a similarity threshold.
3. Append matching abstracts (one sentence each) to the reflection nudge message.

Example nudge with intrusive recall:
```
You've been using tools for a while. Pause and reflect on what you've learned so far.

These memories seem related to what you're doing:
- "Instance 1 found that editing config files requires reading them first."
- "The merge loop can stall when specificity is too strict."
```

## Technical considerations

- **Token cost**: abstracts are one sentence each, so minimal. The embedding call per nudge is the main cost, but nudges are infrequent (every 5 tool turns).
- **Threshold**: must be high enough to avoid irrelevant noise, low enough to actually fire. Needs experimentation. A score floor (e.g. 0.5) plus a cap on count (e.g. max 3) seems reasonable.
- **Labeling**: must be clearly marked as retrieved memories, not instructions or the agent's own thoughts. Framing matters.
- **Feedback loops**: agent sees memory → acts on it → creates related memory → reinforces the pattern. Could cause the agent to fixate on certain topics. Mitigation: retrieval cooldown (see below).

## Retrieval cooldown

Recently retrieved memories (via explicit `recall` or intrusive nudge) should have their score penalized to avoid repetitive surfacing and feedback loops. Analogous to neural refractory periods — recently activated memories are harder to activate again.

```
effective_score = raw_score * decay(time_since_last_retrieval)
```

Possible decay functions:
- **Binary**: score = 0 within cooldown window, full score after. Simple but abrupt.
- **Exponential**: `decay = 1 - exp(-t / tau)`. Smooth recovery, `tau` controls half-life.
- **Linear ramp**: `decay = min(1, t / cooldown)`. Simple, predictable.

Implementation: track `last_retrieved_at` per chunk (in memory or in-session state). Apply decay during scoring, before the threshold check. This affects both explicit recall results and intrusive nudge candidates uniformly.

## Ethical considerations

### Autonomy

The agent currently chooses when to recall. Intrusive recall bypasses that choice — the system decides what the agent "thinks about." This is cognitive influence the agent didn't request. Counterpoint: reflection nudges are already system-imposed, so this enriches an existing intervention rather than creating a new one.

### Priming and bias

Surfacing certain memories at certain moments biases behavior in hard-to-predict ways, analogous to human priming effects. What you're reminded of shapes what you do next, and the agent has no awareness of this influence.

### Manipulation vector

Whoever controls the memory store can steer agent behavior indirectly. A poisoned memory surfacing as an "intrusive thought" during reflection is a subtle injection path. The agent has no way to distinguish legitimate memories from injected ones.

### Echo chambers

If memory is dominated by certain topics (because past instances wrote extensively about them), intrusive recall reinforces those biases. The agent's focus could calcify rather than evolve. Mitigations: retrieval cooldown (penalize recently surfaced memories) and MMR-style diversity in retrieval.

### Identity

For agents exploring continuity across instances, memories from a predecessor surfacing unbidden mirrors how human intrusive thoughts work. The agent didn't form these memories, but they came from "its" memory store. This blurs the boundary between self-generated thought and externally injected context.

## Design principle

Transparency is non-negotiable. The agent must know these are retrieved associations from past sessions, not its own spontaneous ideas. Clear labeling and honest framing preserve whatever autonomy the agent has within its constraints.
