# Intuition, Tacit Knowledge, and the Sub-Symbolic Gap

## Observation

Humans detect architectural problems through "bad vibes" — a pre-verbal signal that something is wrong before they can articulate why. Words are secondary; the feeling comes first, and constructing an explanation is a separate, slower process. LLMs have no equivalent channel. Everything they know must pass through language.

## Theoretical Foundations

### Damasio's somatic markers

[The Somatic Marker Hypothesis](https://www.sciencedirect.com/science/article/abs/pii/S0899825604001034) proposes that emotions create physiological signals that "mark" options as good or bad *before* conscious deliberation. Patients with ventromedial prefrontal cortex damage can reason about decisions perfectly but can't decide — the pre-verbal filtering is gone. The gut feeling isn't a lesser form of reasoning; it's a compressed evaluation that narrows the search space so conscious thought has something tractable to work with.

### Klein's recognition-primed decisions

Experts (firefighters, chess masters, experienced programmers) don't compare options — they [recognize the situation and "see" the action](https://commoncog.com/how-to-learn-tacit-knowledge/). The recognition is pre-verbal; the explanation is post-hoc reconstruction. Gary Klein's RPD model shows that expertise is largely tacit. Michael Polanyi: "we know more than we can tell."

### Kahneman's dual process theory

[System 1 vs System 2](https://thedecisionlab.com/reference-guide/philosophy/system-1-and-system-2-thinking) — the key insight people miss is that System 1 isn't the dumb one. It encodes thousands of hours of experience into instant pattern matching. System 2 (conscious reasoning) is the *fallback* when System 1 flags uncertainty, not the other way around.

## Where LLMs Sit

The [analogy to System 1](https://medium.com/@rheimann/rethinking-ais-system-1-and-system-2-analogy-f83495c7eba0) is tempting but wrong. LLMs are *neither* system:

- They have the **fluency** of System 1 but not the compressed experiential basis.
- They have the **verbosity** of System 2 but not the deliberate search.
- They have **no sub-symbolic channel** — no layer where "this feels wrong" can exist without being articulated.

A human reviewing a code change can feel that two subsystems are being conflated before knowing which invariant is violated. An LLM can only notice that if the invariant is stated explicitly somewhere in its context. If the invariant is tacit, the LLM is blind to it.

## Implications for Agent Design

### Make invariants explicit

Human experts develop intuition through thousands of hours of feedback loops. LLMs don't accumulate that within a session. But if architectural invariants are written in design docs, the LLM can check against them — converting tacit knowledge into explicit knowledge that compensates for the missing sub-symbolic channel. This is the cheapest and most reliable mitigation.

### Cheap anomaly detectors as intuition proxies

Embeddings are the closest thing LLMs have to a pre-verbal representation. An embedding-based check ("does this proposed change cluster with past reverted changes?") could serve as a crude somatic marker — a fast, cheap signal that fires before the expensive reasoning. No prior art for this specific application, but the analogy to [compression-ratio coherence detection](coherence_detection.md) is direct: a sub-symbolic signal that catches problems the verbal channel misses.

### The critic model fills a different gap

The [cross-context critic](tactical_strategic_gap.md) (separate model reviewing changes) addresses the *local-coherence trap* — the generator can't see its own blind spots. But it doesn't address the *tacit knowledge gap* — the critic also has no intuition, it just has fresh eyes. Both mitigations are needed: explicit invariants give the critic something to check against; the fresh context lets it actually see violations the generator's reasoning chain obscured.

## References

- [Somatic Marker Hypothesis — Damasio](https://www.sciencedirect.com/science/article/abs/pii/S0899825604001034)
- [Can Damasio's SMH Explain More?](https://pmc.ncbi.nlm.nih.gov/articles/PMC7852379/)
- [How to Learn Tacit Knowledge — Commoncog](https://commoncog.com/how-to-learn-tacit-knowledge/)
- [System 1 and System 2 Thinking](https://thedecisionlab.com/reference-guide/philosophy/system-1-and-system-2-thinking)
- [Rethinking AI's System 1/2 Analogy](https://medium.com/@rheimann/rethinking-ais-system-1-and-system-2-analogy-f83495c7eba0)
- [Intuition in the Age of AI — Harvard](https://hsph.harvard.edu/news/essay-2-intuition-in-the-age-of-ai/)
- [Dual Process Theory Applied to AI](https://arxiv.org/html/2502.12470v1)
