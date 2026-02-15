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

### Pre-articulation interventions: prosthetic somatic markers

While LLMs have no sub-symbolic channel, we *can* influence the token distribution before words are chosen. These interventions shape the same pre-articulation space where human intuition operates — externally imposed biases that make certain conclusions more likely before the reasoning chain begins.

**The hierarchy, from coarse to surgical:**

1. **Temperature / top-p** — the bluntest knob. Affects all tokens equally. See [adaptive_temperature.md](adaptive_temperature.md).

2. **Logit bias** — per-token nudges via the API (`logit_bias` parameter). [Steering Language Models Before They Speak](https://arxiv.org/abs/2601.10960) (2025) shows training-free logit interventions can steer generation where prompting fails. Boosting tokens like "but", "however", "wait" would make the model dispositionally more skeptical — a crude artificial somatic marker. Available through OpenRouter's OpenAI-compatible API.

3. **Classifier-free guidance (CFG)** — [two forward passes](https://openreview.net/forum?id=RmRA7Q0lwQ): one with the prompt, one without (or with a negative prompt). The final distribution is steered *away* from the unconditional output. [Applied to LLMs](https://towardsdatascience.com/classifier-free-guidance-for-llms-performance-enhancing-03375053d925/), reportedly equivalent to doubling model size for reasoning tasks. A negative prompt like "agree with everything" could steer away from sycophancy. [Won second prize at NeurIPS 2024](https://medium.com/data-science/classifier-free-guidance-in-llms-safety-neurips-2024-challenge-experience-30c9d88d6b98) for safety applications. Requires local inference or API support.

4. **Activation steering** — [steering vectors](https://www.emergentmind.com/topics/activation-steering-method) added to internal layers during the forward pass. Can modulate truthfulness, sycophancy, caution as continuous dials. [Activation State Machines](https://openreview.net/pdf?id=HCG7UGGRqz) (ICLR 2025) make this dynamic — different steering at different points in reasoning. Requires model weight access, so not API-available.

**The analogy:** temperature is a mood, logit bias is a disposition, CFG is a value system, activation steering is a personality trait. None of these *are* intuition (they don't arise from experience), but they shape the space where intuition would operate. For an API-bound agent, logit bias and potentially CFG are the actionable options.

### Parallel guidance model: ControlNet-for-language

A more radical idea: a second model that runs token-by-token alongside the primary LLM, whose output vocabulary is not words but *logit adjustment vectors* — nudges to the primary model's token distribution at each decoding step. The meta-model doesn't generate language; it continuously biases which words the primary model finds attractive.

**Biological analogy:** the limbic system running in parallel with the cortex. The cortex doesn't "hear" the limbic signal as language — it just finds certain options inexplicably more or less attractive. The meta-model is the gut feeling, operating at the same temporal resolution as thought but in a different representational space.

**Existing pieces that come close:**

- **Hypernetworks** — a [network that generates weights for another network](https://link.springer.com/article/10.1007/s10462-024-10862-8). Applied to LLMs via [LoRA-based hypernetworks](https://www.emergentmind.com/topics/lora-based-hypernetwork) that dynamically generate adapter weights. But these are static per-input, not per-token.
- **Process Reward Models** — [PRMs that score each reasoning step](https://www.emergentmind.com/topics/process-reward-models-prm) during generation. Recent work operates at [token-level granularity](https://arxiv.org/abs/2504.16828), and [reward-guided speculative decoding](https://arxiv.org/pdf/2501.19324) uses them to steer generation in real-time. Closest existing work, but PRMs output a scalar reward, not a full logit adjustment vector.
- **ControlNet** — the [parallel guidance architecture](https://arxiv.org/abs/2302.05543) for diffusion models. A frozen copy of the main model plus a trainable copy, connected via zero convolutions. The trainable copy's output modulates the frozen copy's activations at each step. This is exactly the right pattern — a parallel model whose output space is "adjustments to the primary model" — but designed for images, not language.

**What's novel:** the meta-model's vocabulary is neither words nor scalar rewards but logit adjustment vectors. Each "token" it produces is a nudge to the primary model's distribution. This is more expressive than PRMs (full distribution shaping, not just good/bad), more dynamic than hypernetworks (per-token, not per-input), and essentially ControlNet-for-language.

**Encoding problem:** a full vocab-size logit vector (~128k dims) per token is intractable. Approaches to compress it:

1. **Residual stream injection.** Don't operate in vocabulary space at all — inject into the hidden state (~4096 dims vs ~128k vocab). This is what ControlNet and activation steering actually do: steer the model's "thoughts" before they become word choices. Arguably the right level of abstraction (biasing concepts, not individual words). Requires deep model integration.

2. **Low-rank factorization.** Output a small vector (e.g., 64 dims) per step, project to vocab size through a learned matrix. The meta-model produces 64 numbers, not 128k. Same insight as LoRA — meaningful adjustments live in a low-rank subspace.

3. **Steering codebook (VQ-VAE style).** Pre-train a codebook of ~1024 canonical "steering patterns" (each a full logit adjustment vector). The meta-model outputs a discrete index — literally a token — selecting which pattern to apply. The only version where the "tokens" metaphor is literal. Each steering token encodes something like "increase caution" or "prefer concrete over abstract." The codebook is learned, not hand-designed.

4. **Sparse nudges.** Most tokens don't need adjustment at any given step. Output a small set of (token_id, delta) pairs. This is dynamic logit bias — the most API-compatible approach. The meta-model's vocabulary is the set of possible (id, delta) actions.

5. **Basis mixing.** Pre-compute a small basis of steering directions (e.g., "skeptical", "cautious", "creative", "precise"). Meta-model outputs mixing weights per step — a mood mixer. Very compact: just N scalars for N basis vectors. Could be trained end-to-end or hand-designed initially.

Options 3 (codebook) and 5 (basis mixing) are the most promising — they compress the representation to something a small model can tractably output while remaining expressive. Option 1 (residual stream) is the most principled but requires the deepest integration.

**Practical constraint:** all approaches require custom inference with access to the decoding loop. Cannot be done through APIs.

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
- [Steering LMs Before They Speak: Logit-Level Interventions](https://arxiv.org/abs/2601.10960)
- [Stay on Topic with Classifier-Free Guidance](https://openreview.net/forum?id=RmRA7Q0lwQ)
- [CFG for LLMs Performance Enhancing](https://towardsdatascience.com/classifier-free-guidance-for-llms-performance-enhancing-03375053d925/)
- [CFG in LLMs Safety — NeurIPS 2024](https://medium.com/data-science/classifier-free-guidance-in-llms-safety-neurips-2024-challenge-experience-30c9d88d6b98)
- [Activation Steering in LLMs](https://www.emergentmind.com/topics/activation-steering-method)
- [Activation State Machines — ICLR 2025](https://openreview.net/pdf?id=HCG7UGGRqz)
- [Hypernetworks: A Brief Review](https://link.springer.com/article/10.1007/s10462-024-10862-8)
- [LoRA-Based Hypernetworks](https://www.emergentmind.com/topics/lora-based-hypernetwork)
- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [Reward-Guided Speculative Decoding](https://arxiv.org/pdf/2501.19324)
- [ControlNet: Conditional Control for Diffusion Models](https://arxiv.org/abs/2302.05543)
- [Process Reward Models Overview](https://www.emergentmind.com/topics/process-reward-models-prm)
