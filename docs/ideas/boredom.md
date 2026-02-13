# Boredom / Topic Saturation

## Problem

Agents can fixate on a single task or topic for too long, exhausting it without switching to other productive work. A "boredom" mechanism would create intrinsic pressure to diversify attention over time.

## Prior Art

### Curiosity-driven exploration (RL)

Schmidhuber's artificial curiosity (1990s), Pathak et al. ICM (2017), Burda et al. RND (2018). Agents get intrinsic reward from prediction error or learning progress. When the agent stops learning in an area, reward drops and it moves on.

### Information foraging theory

Pirolli & Card. Borrowed from optimal foraging in ecology. Agents should switch tasks when the rate of useful discovery ("information scent") drops below a threshold, like an animal leaving a depleted food patch.

### Habituation

The biological boredom mechanism. Repeated exposure to a stimulus decreases response. Simple and well-understood.

### Explore-exploit tradeoff

Multi-armed bandit literature (UCB, Thompson sampling). Each task is an "arm." The longer you pull one arm without novel reward, the more attractive unexplored arms become.

## Proposed Implementation

**Embedding-based topic saturation.** Embed recent turns, track how long the agent stays in the same embedding neighborhood.

1. Maintain a sliding window of recent turn embeddings.
2. Compute average pairwise similarity within the window.
3. When similarity exceeds a threshold for N consecutive turns, inject a nudge: "You've been focused on this for a while. What else could you explore?"
4. Escalate progressively — gentle suggestion first, then stronger nudge.

This reuses existing embeddings infrastructure (`get_embeddings` in `client.py`). The saturation signal could also feed into task prioritization if we add explicit task management later.

## Open Questions

- What's the right window size? Too short = jittery, too long = slow to react.
- Should boredom be per-session or persist across sessions (via memory)?
- How to balance boredom against legitimate deep work that requires sustained focus?
