# Idea: Memory Activation (Decay & Strengthening)

**Status: not implemented.**

Neuroscience-inspired activation model for memory chunks: memories strengthen with use and decay with time.

## Prior art

### Cognitive architectures

- **ACT-R** (Anderson): base-level activation `B_i = ln(Σ t_j^(-d))` where `t_j` is time since the j-th retrieval and `d` is the decay rate. Each retrieval adds a term to the sum. Below an activation threshold, memory becomes inaccessible. The most theoretically grounded model — unifies decay and strengthening in one formula.
- **Soar**: episodic and semantic memory with activation, plus chunking (analogous to our merge system).
- **LIDA**: based on Global Workspace Theory, includes explicit decay and activation spreading.

### LLM agent systems

- **Generative Agents** (Park et al., 2023): memories scored by `recency × importance × relevance`. Recency is exponential decay since last access. Importance is LLM-scored at creation (1-10). Relevance is embedding similarity to current context. Closest modern analog to what we're building.
- **MemGPT** (Packer et al., 2023): tiered memory (context / archival / recall) inspired by OS virtual memory. Frequently needed memories stay "hot."

## Neuroscience mapping

| Neuroscience concept | Mindloop analog |
|---|---|
| Hebbian strengthening | Retrieval boosts chunk activation |
| Ebbinghaus forgetting curve | Activation decays with time since last access |
| Spaced repetition (spacing effect) | Retrievals across sessions strengthen more than clustered ones |
| Memory consolidation | Chunks surviving multiple sessions/merges gain permanence |
| Reconsolidation | Retrieved chunks become candidates for re-merge with new context |
| Long-term depression | Never-retrieved chunks fall below activation threshold |
| Interference | Merge loop already handles this — related memories combine |

## Proposed design

### Activation field

Add `activation` (float) and `retrieval_history` (list of timestamps) to each chunk.

### Scoring options

**Simple (Generative Agents style):**
```
score = similarity * recency * importance
```
- `similarity`: embedding cosine sim (existing).
- `recency`: exponential decay since last retrieval, `exp(-lambda * t)`.
- `importance`: could be LLM-scored at creation, or derived from merge count (more merges = more important).

**Principled (ACT-R style):**
```
activation = ln(Σ t_j^(-d)) + similarity * relevance_weight
```
- Each retrieval adds a `t_j^(-d)` term. Recent retrievals contribute more.
- Naturally decays: old, unretrieved memories have low activation.
- Naturally strengthens: frequently retrieved memories accumulate terms.
- `d ≈ 0.5` is the standard ACT-R decay parameter.

### Soft forgetting

Memories below an activation threshold are excluded from search results but remain in the database. They can be "rediscovered" if a very strong similarity match overrides the threshold, or reactivated if the threshold is lowered.

This is more humane than hard deletion and mirrors how human memories become inaccessible rather than truly erased.

### Consolidation

Memories that persist across multiple sessions without being merged or contradicted gain a base activation bonus. Like sleep consolidation in neuroscience — surviving the passage of time is itself evidence of importance.

Possible implementation: `base_activation = ln(1 + sessions_survived)`.

### Interaction with existing features

- **Merge loop**: merged chunks inherit the combined retrieval history of their sources. A merge is itself a form of consolidation.
- **Specificity**: low-activation memories could be excluded from the neighbor count, so they don't inflate specificity for active memories.
- **Intrusive recall** (see `intrusive_recall.md`): activation naturally handles cooldown — a just-retrieved memory has high activation but can be penalized for the intrusive path specifically. Subsumes the retrieval cooldown idea.
- **Pinned memories** (see `pinned_memories.md`): pinned chunks receive an activation boost proportional to their pin priority rank, preventing top-ranked ones from decaying below the retrieval threshold.
- **Hybrid search**: activation becomes a third factor in scoring alongside embedding similarity and BM25.

## Implementation sketch

```python
# In MemoryStore:
# New columns: activation REAL, retrieval_ts TEXT (JSON list of ISO timestamps).

def record_retrieval(self, chunk_id: int) -> None:
    """Record a retrieval event, boosting activation."""
    # Append current timestamp to retrieval history.
    # Recompute activation.

def compute_activation(self, retrieval_ts: list[float], d: float = 0.5) -> float:
    """ACT-R base-level activation."""
    now = time.time()
    terms = [(now - t) ** (-d) for t in retrieval_ts if now > t]
    return math.log(sum(terms)) if terms else -float('inf')

def search(self, query_embedding, ..., min_activation: float = -2.0):
    """Exclude chunks below activation threshold."""
    # Filter by activation before or after similarity ranking.
```

## Open questions

- ACT-R vs Generative Agents approach? ACT-R is more principled but adds complexity. Could start with the simpler model.
- Should importance be LLM-scored at creation time or derived from structural signals (merge count, retrieval frequency)?
- How does activation interact with the merge decision? Should low-activation chunks be less likely to participate in merges (they're "fading") or more likely (consolidate before forgetting)?
- Activation threshold: fixed or adaptive (e.g. based on total memory count)?
- Should the agent be aware of activation scores? Showing them in recall results could help the agent understand which memories are fading.
