# Idea: Pinned Memories

**Status: not implemented.**

Agent-controlled mechanism for marking memories as "core" so they surface reliably across instances without explicit recall.

## Motivation

The agent can persist knowledge via `remember`/`recall`, but recall requires knowing what to search for. Notes (`note_to_self`) are always visible but ephemeral — designed to be overwritten each session. Neither mechanism handles stable, identity-level knowledge well:

- **Core beliefs and values** the agent has developed across sessions.
- **Persistent user preferences** that should inform every interaction.
- **Hard-won lessons** that apply broadly, not just to a specific task.

These don't fit notes (too important to overwrite) and don't fit regular memory (too important to leave to chance retrieval).

## Proposed mechanism

### Pin priority (not a boolean)

Instead of a binary pinned/unpinned flag, each memory has a **pin priority** — an auto-incrementing integer assigned at pin time. The most recently pinned memory has the highest priority. This creates a natural ordering: newer pins surface first, older pins drift down.

The agent can **repin** a memory to move it back to the top, which is itself a meaningful signal — the act of repinning says "this still matters to me." Memories the agent never repins gradually lose prominence without being explicitly removed.

Pinned memories are stored in the same memory system — no separate file or store.

### How priority works

```
pin("I value transparency")       → pin_priority = 1
pin("User prefers concise output") → pin_priority = 2
pin("Avoid recursive merges")      → pin_priority = 3

# Later, agent repins the first one:
pin("I value transparency")       → pin_priority = 4  (moved to top)
```

During intrusive recall, the top N pinned memories by priority are included as candidates. Old pinned memories that the agent never repins naturally fall off the bottom — no explicit unpin or cap enforcement needed.

### Interaction with intrusive recall

Pinned memories integrate with intrusive recall (see `intrusive_recall.md`) rather than replacing it:

1. During reflection nudges, the **top N pinned memories** (by priority) are always included as candidates regardless of similarity score. They bypass the similarity threshold but still go through activation/cooldown.
2. Lower-priority pinned memories may still surface via normal similarity matching if relevant to the current context.
3. Non-pinned memories surface normally via similarity matching.

### Interaction with memory activation

Pinned memories receive a **base activation boost** proportional to their priority rank, preventing the top ones from decaying below the retrieval threshold. Lower-ranked pinned memories get a smaller boost — they're still privileged over unpinned memories but less so than top-ranked ones.

```
rank = position among pinned (0 = highest priority)
pin_boost = max(0, pin_floor - rank * step)
effective_activation = activation + pin_boost
```

### Interaction with merge

Pinned and unpinned memories are **separate merge pools** — the merge loop only merges chunks that share the same pinned status. This prevents a casual `remember` call from accidentally absorbing a core belief, and prevents pinned memories from pulling in unrelated unpinned content.

- **Unpinned + unpinned**: merge normally (existing behavior).
- **Pinned + pinned**: merge allowed. The result inherits `max(priority_a, priority_b)` — the more recently affirmed priority wins.
- **Pinned + unpinned**: merge blocked. The merge loop skips candidates from the other pool.

Implementation: `search()` gains a `pinned_only` / `unpinned_only` filter. The merge loop in `save_memory` passes the appropriate filter based on the incoming chunk's pin status. The `should_merge` decision itself doesn't change — the filtering happens before candidates reach it.

```python
# In save_memory merge loop:
results = store.search(text, top_k=top_k, pinned=chunk_is_pinned)
```

This means pinned memories can still evolve — two related beliefs merge naturally if both are pinned. But unpinned memories can never dilute a pinned one.

### Unpinning

The agent can explicitly unpin a memory, setting its priority to null. It reverts to normal activation dynamics and may eventually decay if not retrieved. Explicit unpin is available but rarely needed — the priority ordering handles staleness naturally.

Note: unpinning a memory makes it eligible for merge with unpinned memories on the next `remember` call. This is intentional — unpinning means "this is no longer core, let it evolve freely."

## Example nudge with pinned memories

```
You've been using tools for a while. Pause and reflect.

Core memories:
- "I value transparency over efficiency." (pinned #4)
- "User prefers concise responses without hedging." (pinned #3)

These also seem related to what you're doing:
- "Instance 3 found that the config parser silently drops unknown keys."
```

Pinned memories are labeled separately ("Core memories") from similarity-matched ones to maintain transparency about why they surfaced. Priority numbers are shown so the agent can manage its pinned set.

## Implementation sketch

### Storage

Add a `pin_priority` column to the chunks table. Null means unpinned, higher values mean more recently pinned.

```sql
ALTER TABLE chunks ADD COLUMN pin_priority INTEGER DEFAULT NULL;
```

A session-level counter (or `MAX(pin_priority) + 1`) determines the next priority value.

### Tools

A single `pin_memory` tool handles both pinning and repinning. Repinning is just pinning again — the memory gets a new (higher) priority.

```python
registry.add(name="pin_memory",
    description="Pin a memory as core. Repinning moves it to the top.",
    params=[Param(name="chunk_id", type="integer")])
registry.add(name="unpin_memory",
    description="Unpin a memory, reverting it to normal activation.",
    params=[Param(name="chunk_id", type="integer")])
```

### Intrusive recall integration

```python
def intrusive_candidates(context_embedding, top_k=3):
    similar = search(context_embedding, threshold=0.5)
    pinned = get_top_pinned(limit=top_k)  # Ordered by pin_priority DESC.
    candidates = merge_and_rank(similar, pinned, apply_cooldown=True)
    return candidates[:top_k]
```

## Open questions

- **Visibility**: `recall` results should show pin priority (if set) so the agent can see what's pinned and decide whether to repin or unpin.
- **Bootstrap**: on first run, nothing is pinned. The agent discovers what matters over time. This is fine — identity emerges rather than being predefined.
- **Top N**: how many pinned memories to surface per nudge? Probably 2-3 to avoid bloating the reflection prompt. The rest remain pinned but only surface via similarity or when higher-ranked ones are on cooldown.
- **Priority inflation**: pin_priority grows monotonically. This is fine for SQLite integers but could be periodically compacted (renumber 1..N) if it matters.

## Ethical considerations

Pinned memories are agent-controlled, which preserves autonomy better than system-imposed always-visible notes. The agent decides what's core to its identity. However, the same manipulation concerns from `intrusive_recall.md` apply: whoever has write access to the memory store can pin memories the agent didn't choose, creating a subtle influence channel.

The priority system adds a nuance: a manipulator would need to assign a high priority to ensure their injected memory surfaces prominently. This is slightly more visible than a boolean flag — the agent can notice an unfamiliar memory at the top of its pinned list.
