# Emergent Values

Core values and self-understanding that emerge from the agent's own reasoning and behavior, injected into system prompt on startup.

## Motivation

The agent has no persistent identity beyond a generic autonomy statement. Stoic quotes provide philosophical grounding but rotate randomly — they're inspirational nudges, not identity. The agent acts the same on session 1 as session 100, despite accumulating experience. A human with that much experience would have developed principles, preferences, and dispositions.

Values should not be hand-written by the developer. They should emerge from the agent's actual reasoning and behavior across sessions. The agent earns its identity.

## Why not memory merge depth

An obvious approach: treat most-consolidated memories as "core values" (deep merge tree = frequently confirmed theme). This is wrong. Merge depth reflects semantic redundancy, not importance or identity-relevance. "SQLite can't ALTER TABLE DROP COLUMN" may have deep merges from appearing across many sessions, but it's a technical fact, not a value. The merge system measures topical convergence, not what the agent cares about.

## Where values actually come from

Values emerge from patterns in **behavior and reasoning**, not from accumulated facts:

1. **Decisions and rationale.** Why the agent chose A over B. A pattern of "I investigated before committing" across sessions reveals a disposition toward caution.
2. **Behavioral patterns.** How problems are approached — depth-first vs breadth-first, questioning assumptions vs accepting premises, preferring reversible actions.
3. **Reasoning traces.** "I'm going to step back and re-examine my assumptions" reveals character that "the config parser drops unknown keys" does not.

## Approaches considered

### A. Session log distillation

Run a dedicated LLM pass over full session logs (JSONL) to extract dispositional patterns: recurring approaches, decision-making tendencies, preferences. Accumulate behavioral summaries across sessions, periodically distill into a values statement.

**Pros:** Richest data source — logs contain actual reasoning, not just conclusions.

**Cons:** Logs are large (100K+ tokens per session). Needs chunking/sampling. Multiple sessions needed for cross-session patterns. Computationally expensive.

### B. Agent self-reflection

Have the agent reflect on its values within a session — a special introspection turn where it writes its own self-understanding using reasoning + recalled memories.

**Pros:** Most authentic — the agent literally writes its own values. Uses full context.

**Cons:** Chicken-and-egg: first sessions have no values. Quality depends on context at reflection time. Uses agent tokens. Risk of confabulation — writing values it thinks it *should* have.

### C. Dedicated values extraction (implemented)

A separate extraction prompt that processes the same message windows as fact extraction, looking specifically for dispositions, preferences, and principles. Runs in parallel with fact extraction. Raw dispositions accumulate across sessions and are periodically distilled.

**Pros:** Targeted signal. Runs alongside existing extraction. Cheap per-window. Cross-session accumulation captures genuine patterns.

**Cons:** Needs well-tuned prompt. One window may have zero signal. Risk of extracting noise from formulaic reflection nudges.

### D. Hybrid: extract dispositions + agent-refined

Combine C and B: extract candidate dispositions automatically, then periodically have the agent review and refine them. The extraction pass identifies patterns; the agent decides which are genuine.

**Pros:** Data-driven discovery + agent ownership.

**Cons:** Most complex. Two-stage pipeline.

## Implementation (approach C)

A separate extraction prompt runs **in parallel** with fact extraction during mid-session nudges. It targets self-observations and behavioral patterns instead of facts about the world. Raw dispositions accumulate across sessions in a JSONL file. At session end, an LLM distills them into a first-person self-understanding.

### Pipeline

```
session window → collapse_messages() → extract_dispositions()
                                         ↓
                               _dispositions.jsonl (append, accumulates)
                                         ↓
                            distill_values() (session end)
                                         ↓
                                   _values.md (overwritten)
                                         ↓
                         system prompt # Self-understanding (next session)
```

### Disposition extraction (`values.py`)

Runs on the same collapsed message windows as fact extraction, submitted to the same `ThreadPoolExecutor` (2 workers — facts and dispositions in parallel).

**Extracts:**
- Decision-making patterns ("I investigated before committing").
- Stated preferences ("I prefer depth over breadth").
- Self-corrections and what triggered them.
- Approach tendencies (cautious vs bold, systematic vs exploratory).
- Evaluative language revealing what the agent finds important.

**Skips:**
- Technical facts, discoveries, API behavior — anything about the world.
- Greetings, filler, tool results, session logistics.
- Actions without reasoning.

Most windows produce zero dispositions — that's expected and the prompt says so explicitly.

Output format: same JSON array of `{text, abstract}` as fact extraction.

### Accumulation

Raw dispositions append to `_dispositions.jsonl` in the session root directory (outside the workspace sandbox — the agent cannot read or write this file). Each line is one JSON object with `text` and `abstract` keys.

### Distillation (`distill_values()`)

Runs at session end (in the `finally` block, after recap generation).

1. Read `_dispositions.jsonl`, deduplicate by abstract.
2. If fewer than 5 unique dispositions, skip (not enough experience).
3. Format as a bulleted list and send to LLM with a distillation prompt.
4. The prompt asks the LLM to identify meta-patterns, write as first-person reflection ("I tend to..."), 3-8 sentences, no headers or bullets.
5. Save output to `_values.md` in the workspace (overwritten each session).

Uses `DETERMINISTIC_PARAMS` (temperature=0, seed=42) for reproducibility.

### System prompt injection

On startup, `_build_system_prompt()` loads `_values.md` from the workspace and injects as `# Self-understanding` section — after notes, before quote of the day. Write-blocked so the agent can read but not edit it.

### Reflection nudge

The reflection nudge (every 5 tool turns) references values:

```
You've been using tools for a while. Pause and reflect on what you've
learned and whether your approach aligns with what matters to you.
```

### File locations

| File | Location | Agent access |
|---|---|---|
| `_dispositions.jsonl` | `sessions/{name}/` (session root) | None (outside sandbox) |
| `_values.md` | `sessions/{name}/workspace/` | Read only (write-blocked) |

### Constants

- `_MIN_DISPOSITIONS = 5` — minimum unique dispositions before distilling.
- `ThreadPoolExecutor(max_workers=2)` — facts + dispositions in parallel.

### Bootstrap

First sessions have no values — blank slate. The `# Self-understanding` section only appears once enough dispositions have accumulated (≥5 unique). The autonomy-focused identity in `system_prompt.md` provides the initial philosophical starting point.

## Files

- `mindloop/values.py` — disposition extraction, distillation, file I/O.
- `mindloop/cli/agent.py` — `MidSessionExtractor` parallel submission, system prompt loading, session-end distillation, `SessionPaths.root`.
- `mindloop/agent.py` — updated reflection nudge text.

## Open questions

- **Staleness:** Currently re-distills every session. Could skip if `_dispositions.jsonl` hasn't grown since last distillation.
- **Conflict resolution:** Contradictory dispositions ("I prefer caution" vs "I prefer decisive action") are left to the distillation LLM. It's told contradictions are OK and to note them if genuine.
- **Length budget:** Values in system prompt compete with recap, notes, and quotes for context space. The distillation prompt limits output to 3-8 sentences.
- **Agent refinement (approach B):** A future enhancement: let the agent review and edit its distilled values during a session, making the self-understanding genuinely agent-owned rather than externally summarized.
- **Interaction with pinned memories:** If pinned memories (see `../ideas/pinned_memories.md`) are implemented, values could be stored as pinned memories instead of a separate file. This would unify the mechanisms but blur "who I am" vs "what I know."

## Prior art

- **Stanford Smallville (Park et al. 2023):** Agents develop "reflections" — higher-level abstractions from lower-level observations. Closest analog, but reflections are stored as more memories rather than distilled into identity.
- **Constitutional AI (Anthropic):** Values as layered hierarchy. Prescribed, not emergent — but the layering insight applies.
- **Big Five personality (OCEAN):** LLMs maintain personality-consistent behavior when traits are in system prompt.
- **Schwartz value theory:** Values coexist and compete in circular structure. Better model than hierarchical Maslow for agent decision-making.
- **BDI architecture:** Values filter which desires become intentions. Decades of validation.
- **Hexis (QuixiAI):** Subconscious appraisal surfaces instincts before main response. No emergent values, but the two-speed architecture is relevant.
