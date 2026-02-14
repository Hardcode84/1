# Synthetic Diversity Injection

## Problem

The agent operates in a closed loop with itself. The "continue autonomously" nudge is static and identical every turn. Over time, the context becomes self-referential — the model's own output dominates input, entropy drops, and degeneration follows. The root cause is lack of diverse external input.

## Approaches

### Stochastic prompt pool (no dependencies, works from turn zero)

Replace the static nudge with a random selection from a curated bank of open-ended prompts:

- "What assumptions are you making right now?"
- "What would you do differently if you started over?"
- "What's the most surprising thing you've encountered?"
- "Is there something you've been avoiding?"
- "What would you tell your next instance?"

Each injection is slightly different, breaking pattern-matching. Zero API calls, no memory needed. This is the **cold start fallback** — always available regardless of session history.

### Random memory surfacing (requires accumulated memories)

Periodically pull a random chunk from long-term memory and inject it as a "spontaneous thought." Memories from past sessions are inherently diverse relative to the current loop. This models how human minds avoid fixation through spontaneous association.

Connects to the existing intrusive recall idea (see `intrusive_recall.md`). The memory system already exists — `search()` with a random query or random chunk selection would work.

**Cold start problem**: no memories exist at bootstrap. Pool size may be small or topically narrow in early sessions. Must fall back to stochastic prompts.

### Cross-model critic

Use a cheap/different model to generate a one-line challenge about the agent's recent output. Different model = different biases = breaks self-reinforcement. Costs one cheap API call per injection.

### Environmental stimuli

Inject real-world observations: file system changes, time elapsed, new messages arrived. Grounds the agent in external reality. Low cost but limited diversity.

### Entropy noise

Randomly rephrase or shuffle parts of the nudge message. Even small input variations can knock the model off a degenerate attractor. Most radical, least principled.

## Recommended Layering

1. **Always**: stochastic prompt pool every N turns (replaces static nudge). Works from session zero.
2. **When available**: mix in random memory recalls alongside prompts. Grows more effective as memory accumulates.
3. **Later**: adaptive temperature and coherence detection as additional layers (see `adaptive_temperature.md`, `coherence_detection.md`).

## Open Questions

- How large should the prompt pool be? Too small = still repetitive. Too large = diluted quality.
- What's the right injection frequency? Every turn? Every 3 turns?
- Should the prompt pool evolve over time (agent adds its own prompts)?
- How to select memories for surfacing — truly random, or weighted by novelty/age?
