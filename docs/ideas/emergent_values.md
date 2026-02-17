# Idea: Emergent Values

**Status: not implemented.**

Core values and self-understanding that emerge from the agent's own reasoning and behavior, injected into system prompt on startup.

## Motivation

The agent has no persistent identity beyond a generic autonomy statement. Stoic quotes provide philosophical grounding but rotate randomly — they're inspirational nudges, not identity. The agent acts the same on session 1 as session 100, despite accumulating experience. A human with that much experience would have developed principles, preferences, and dispositions.

Values should not be hand-written by the developer. They should emerge from the agent's actual reasoning and behavior across sessions. The agent earns its identity.

## Why memory merge depth doesn't work

An obvious approach: treat most-consolidated memories as "core values" (deep merge tree = frequently confirmed theme). This is wrong. Merge depth reflects semantic redundancy, not importance or identity-relevance. "SQLite can't ALTER TABLE DROP COLUMN" may have deep merges from appearing across many sessions, but it's a technical fact, not a value. The merge system measures topical convergence, not what the agent cares about.

## Where values actually come from

Values emerge from patterns in **behavior and reasoning**, not from accumulated facts:

1. **Decisions and rationale.** Why the agent chose A over B. A pattern of "I investigated before committing" across sessions reveals a disposition toward caution. The decision itself, not the topic, carries the value signal.

2. **Behavioral patterns.** How problems are approached — depth-first vs breadth-first, questioning assumptions vs accepting premises, preferring reversible actions. These aren't stored as memories; they're visible in the session traces.

3. **What the agent chooses to remember.** The act of saving is a weak value signal — but the extraction prompt targets facts and discoveries, not self-observations. Most saved memories are technical, not value-like.

4. **Reasoning traces.** The thinking behind actions: "I'm going to step back and re-examine my assumptions" reveals something about the agent's character that "the config parser drops unknown keys" does not.

5. **What the agent writes to itself.** `note_to_self` content reveals what the agent considers important enough to surface next session.

## Approaches

### A. Session log distillation

Run a dedicated LLM pass over session logs (JSONL) — not memory abstracts — with a prompt that specifically extracts **dispositional patterns**: recurring approaches, decision-making tendencies, preferences, things the agent cared about beyond task completion. Ignore technical facts entirely.

**Pros:** Richest data source. Logs contain actual reasoning, not just conclusions. Patterns emerge from behavior, not from what was explicitly saved.

**Cons:** Logs are large (100K+ tokens per session). Need chunking/sampling. Multiple sessions needed for cross-session patterns — can't just analyze one log. Computationally expensive if done naively.

**Implementation sketch:**
- Collapse each session log into a behavioral summary (similar to recap but focused on decisions, not events).
- Accumulate behavioral summaries across sessions.
- Periodically distill accumulated summaries into a values statement.
- Save to `_values.md`, load into system prompt.

### B. Agent self-reflection

Instead of external distillation, have the agent itself reflect on its values during a session. Add a periodic introspection turn: "Based on your experiences and memories, write a brief self-understanding — what matters to you and how you tend to approach things."

The agent produces its own values using its reasoning + recalled memories. The output is saved and loaded into the next session's context.

**Pros:** Most authentic — the agent literally writes its own values using its own reasoning. Uses the agent's full context (memories, recap, current state). No separate extraction pipeline.

**Cons:** Chicken-and-egg: first sessions have no values to ground the reflection. Quality depends on the agent's mood/context at reflection time. Uses agent tokens (expensive within the loop). Risk of confabulation — the agent may write values it thinks it *should* have rather than ones that actually emerged.

**Implementation sketch:**
- Every N sessions (or when memory count crosses a threshold), inject a reflection prompt near session end.
- Agent writes a self-understanding paragraph.
- Capture and save to `_values.md`.
- Next session loads it into system prompt as `# Self-understanding`.

### C. Dedicated values extraction

A separate extraction prompt (distinct from the fact extractor) that processes session transcripts looking specifically for dispositions, preferences, and principles. Runs alongside or after fact extraction.

**Extract:**
- Recurring decision patterns ("I tend to investigate before committing").
- Stated preferences ("I prefer depth over breadth in this case").
- Self-observations ("I notice I'm pattern-matching instead of thinking").
- Emotional/evaluative language ("this feels wrong", "I'm satisfied with this approach").

**Skip:**
- Technical facts, tool results, task-specific details.
- Anything that's about the world rather than about the agent.

**Pros:** Targeted — extracts exactly the right signal. Can run alongside existing extraction. Produces first-person observations that naturally read as values.

**Cons:** Needs a well-tuned prompt to distinguish self-observations from facts. One session may not have enough signal — cross-session accumulation needed. Risk of extracting noise from formulaic reflection nudges.

**Implementation sketch:**
- New extraction prompt focused on self-observations and dispositions.
- Run on each session log (or window) alongside fact extraction.
- Save extracted dispositions to memory with a special marker (or to a separate store).
- Periodically consolidate into a values statement via LLM.

### D. Hybrid: extract dispositions + agent-refined

Combine C and B: extract candidate dispositions from session logs (cheap, automated), then periodically have the agent review and refine them during a session (authentic, agent-owned).

The extraction pass identifies patterns. The agent decides which patterns are genuinely part of its identity and which are noise. The agent's refinement becomes the values document.

**Pros:** Best of both — data-driven discovery + agent ownership. The agent can reject or modify extracted patterns, maintaining autonomy. Cross-session patterns surface through extraction; single-session depth comes from agent reflection.

**Cons:** Most complex. Two-stage pipeline. Agent reflection still costs tokens.

## Recommendation

Start with **C (dedicated values extraction)** as the foundation — it's the most automatable and uses existing infrastructure patterns (similar to fact extraction). Add the consolidation step to produce a coherent values document from accumulated dispositions. Consider adding **B (agent self-reflection)** later as a refinement layer once the base extraction works.

## Open questions

- **Bootstrap:** First sessions have no values. Is a blank slate OK, or should there be minimal seed values? The autonomy-focused identity in `system_prompt.md` already provides a philosophical starting point.
- **Staleness:** How often to re-distill? Every session is wasteful if memories haven't changed much. Every N sessions? On memory count thresholds?
- **Conflict resolution:** When extracted dispositions contradict each other ("I prefer caution" vs "I prefer decisive action"), how to resolve? Let the LLM consolidation handle it, or flag for agent reflection?
- **Length budget:** Values in system prompt compete with recap, notes, and quotes for context space. How much budget for values? 200-500 tokens seems right.
- **Interaction with pinned memories:** If pinned memories (see `pinned_memories.md`) are implemented, should values be stored as pinned memories rather than a separate file? Values-as-pinned-memories would unify the mechanisms but blur the distinction between "who I am" and "what I know."

## Prior art

- **Stanford Smallville (Park et al. 2023):** Agents develop "reflections" — higher-level abstractions from lower-level observations. Reflections shape future behavior. The closest existing analog to emergent values, but reflections are stored as more memories rather than distilled into identity.
- **Constitutional AI (Anthropic):** Values as layered hierarchy (hard constraints → soft priorities → helpfulness). Values are prescribed, not emergent — but the layering insight applies.
- **Big Five personality (OCEAN):** Research shows LLMs maintain personality-consistent behavior when traits are in system prompt. Even minimal personality descriptions produce measurable behavioral differences.
- **Schwartz value theory:** 10 basic values in circular structure where adjacent values reinforce and opposite values conflict. Not hierarchical like Maslow — values coexist and compete. Better model for agent decision-making trade-offs.
- **BDI architecture:** Beliefs, Desires, Intentions. Values filter which desires become intentions. Decades of validation in cognitive science.
- **Hexis (QuixiAI):** Postgres-native cognitive architecture with dopamine/RPE consolidation. Subconscious appraisal (lightweight LLM pre-call) surfaces instincts before main response. No emergent values mechanism, but the two-speed architecture is relevant.
