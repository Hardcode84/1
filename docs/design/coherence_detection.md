# Coherence Detection

## Problem

Agents can degenerate into feedback loops where their own output dominates context, producing repetitive or incomprehensible text (e.g. legalese). This poisons workspace files and notes, carrying over across sessions.

## Approaches

### Compression ratio

Degenerate text compresses much better than meaningful text. Run `zlib.compress()` on recent responses and check `compressed_size / original_size`. Below a threshold = repetitive garbage. Zero API calls, essentially free.

### N-gram repetition

Count unique n-grams vs total n-grams in recent output. Degenerate text reuses phrases heavily, dropping the ratio. Also cheap and local.

### Embedding drift

Track cosine similarity between consecutive agent responses. If the last N responses cluster too tightly (high pairwise similarity), the agent is looping. Reuses existing embeddings infrastructure.

### External LLM judge

Ask a cheap model "is this output coherent and on-task?" Most accurate but adds latency and cost per turn.

## Recommendation

Compression ratio is the best starting point — free, catches both repetitive loops and formulaic text, needs minimal tuning. Can combine with the text-turn cap (see TODO.md) as a lightweight guard: if ratio drops below threshold, force-stop early.
