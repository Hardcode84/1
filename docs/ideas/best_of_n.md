# Best-of-N Sampling: Inference-Time Compute Scaling

## Problem

A single LLM sample is a lottery ticket. The model may hallucinate, pick a suboptimal tool, or reason poorly on a hard step. Scaling model size helps but is expensive. An alternative: generate N candidates from the same prompt and select the best one.

## Prior Art

### Self-Consistency (Wang et al., 2022)

Sample multiple chain-of-thought paths, take majority vote on the final answer. No reward model needed. Gives +10-15% on math/reasoning benchmarks (GSM8K, ARC). Free selection signal.

### Process Reward Models (Lightman et al., 2023)

"Let's Verify Step by Step." Train a verifier that scores each reasoning step, not just the final answer. Use it to rank best-of-N candidates. Substantially outperforms outcome-only reward models.

### Inference-Time Compute Scaling (Snell et al., 2024)

Adaptive allocation of test-time compute (more samples for harder problems) can match or beat a 14x larger model. The scaling curve is real and roughly logarithmic in N.

### Large Language Monkeys (Brown et al., 2024)

With enough samples, even weaker models solve hard problems. On SWE-bench, going from 1 to 250 samples dramatically increased solve rate. Bottleneck shifts from generation to selection.

### AlphaCode (DeepMind)

Generated ~1M candidate programs, filtered via clustering and test execution. Achieved competitive programming performance through massive overgeneration + strong selection.

## Scoring Functions

The technique is only as good as the selector. Options ranked by reliability:

### Execution-based (strongest, when available)

- Run code against test cases (binary pass/fail or partial).
- Symbolic verification (SymPy, constraint solvers).
- Cross-validation: one sample writes tests, others are judged against them.
- Tool call validity: does it parse? Are args well-formed? Does the target exist?

### Self-consistency (no extra cost)

- Cluster N responses by final answer or chosen action, pick majority.
- For agents: if 4/5 samples pick `read` and 1 picks `write`, go with `read`.
- Variant: within the majority cluster, pick the most representative response (closest to centroid by embedding).

### Log-probability based (if provider exposes logprobs)

- Mean token log-prob: higher average confidence, fewer hallucinations.
- Length-normalized log-prob: avoids short-response bias.
- Entropy of token distribution: low entropy = high model confidence.
- OpenRouter exposes logprobs for some models but not all.

### LLM-as-judge

- Ask the same or a cheaper model to score/rank candidates.
- Pairwise comparison is more reliable than absolute scoring.
- Cost: 1 extra call per candidate, or log(N) with tournament bracket.
- Risk: shares blind spots with the generator.

### Reward models

- General-purpose: trained on human preference data (RLHF-style).
- Domain-specific: fine-tune a small model on task-specific good/bad examples.
- Most practical at scale; overkill for small N.

### Heuristic (cheap, often underrated)

- Format/structure compliance (required sections, valid JSON, length bounds).
- Reasoning-action coherence: does the reasoning mention the tool it calls?
- Context grounding: does the response reference retrieved chunks?

### Coherence metrics (smooth, cheap, domain-neutral)

Reuses the signals from [coherence detection](../design/coherence_detection.md) as continuous scoring inputs rather than binary degeneration guards.

- **Compression ratio** (`zlib`): `compressed_size / original_size`. Ranges from ~0.1 (degenerate repetition) to ~0.5+ (information-dense). Higher = more informational content. Zero API calls.
- **N-gram diversity**: `unique_ngrams / total_ngrams`, continuous 0-1. Higher = richer language. Catches phrase-level reuse that compression ratio may miss.
- **Embedding drift**: cosine distance from recent context centroid. Inverted-U shape — too close means looping, too far means off-topic. Score as distance from a target band center.

These bridge the gap between binary validity checks ("does the tool call parse?") and expensive judges ("is this good reasoning?"). They provide a smooth gradient for ranking candidates with no extra API calls (except embedding drift, which reuses existing infrastructure).

### Composite

Best results combine signals. A smooth scoring function using coherence:

```
coherence = w1 * compression_ratio + w2 * ngram_diversity
score = majority_cluster_membership * 10 + coherence
```

Self-consistency (cluster membership) remains the dominant signal. Coherence breaks ties within clusters and filters garbage before voting.

### Recommended pipeline

1. **Filter**: discard candidates with compression ratio below a floor (clearly degenerate).
2. **Cluster**: group by structured action (tool + args) for self-consistency vote.
3. **Rank**: within the majority cluster, pick the candidate with highest coherence score.

Each stage is cheap. The function degrades gracefully — even if coherence scoring adds no value on easy turns, it costs almost nothing. On hard turns where multiple candidates look plausible, it differentiates.

## Recommended Scoring: Self-Consistency

For domain-neutral general reasoning, **self-consistency** is the strongest choice.

**Why it wins**: correct reasoning paths converge on the same answer more often than incorrect ones. This is a statistical property of the model's distribution, not a domain-specific heuristic. No verifier, no reward model, no logprobs needed.

**Why alternatives are weaker here**:

- **Log-probs**: high confidence != correct reasoning. Models are confidently wrong routinely. Measures fluency more than correctness.
- **LLM-as-judge**: shares blind spots with the generator. If the model can't distinguish right from wrong, judging N wrong answers won't help.
- **Reward models**: require domain-specific training data.

**For agents**: vote on the structured action (tool + args), not the full reasoning text. Two samples that call `read("config.py")` with different reasoning are still in agreement. Clustering becomes trivial exact-match on the structured output.

**Limitation**: degrades when the correct answer has low probability (very hard problems where the model almost never gets it right). No majority to find. But for the general case it's the best cost/quality tradeoff.

## Diversity Strategy: Prompt Experts over Temperature

### Temperature diversity is tempting but flawed

The idea: low temperature exploits (reliable, consensus answers), high temperature explores (non-obvious solutions). AlphaCode used this successfully — but AlphaCode had a strong execution-based verifier (test cases).

The problem for self-consistency: it assumes samples are comparable quality. High-temperature samples are noisier and pollute the vote. Weighting votes by temperature adds complexity and tuning.

### Prompt diversity is stronger

Same moderate temperature (0.7), different reasoning strategies per expert:

- Expert A: "Think step by step."
- Expert B: "First identify what could go wrong, then solve."
- Expert C: "Work backwards from the desired outcome."
- Expert D: "List your assumptions before reasoning."

**Why this works better**:

- Temperature only changes sampling noise — the model follows the same reasoning pattern with more randomness.
- Prompt diversity changes *what the model attends to*. Different framings activate different knowledge and failure modes.
- Errors become less correlated across experts, which is exactly what makes ensemble methods powerful.

**For an agent loop**: vary the instruction for *how to decide the next action* — cautious vs. exploratory vs. goal-focused vs. diagnostic. Then majority-vote on the action.

**Recipe**: uniform moderate temperature (0.7), diverse system prompts, self-consistency on structured output.

## Applicability to Mindloop

### Where it fits

- **Tool selection**: sample N next-actions, majority-vote on which tool to call. Low cost, high impact on hard decisions.
- **Memory recall ranking**: generate N recall queries, union the results, re-rank.
- **Reflection quality**: sample N reflections, pick the most specific/actionable one (LLM-as-judge or length heuristic).

### Practical considerations

- **KV cache sharing**: OpenRouter caches prefixes for some models. All N requests share the cached prompt, paying only for divergent tokens.
- **Temperature**: need `temperature > 0` for diversity. Sweet spot: 0.6-1.0. Too low = identical samples.
- **Diminishing returns**: gains are ~logarithmic. 1->4 helps a lot, 4->16 some, 16->64 less.
- **Latency**: requests are parallel, so wall-clock time = single request (if provider handles concurrency).
- **Cost**: N requests cost Nx tokens. Amortized by cache sharing and by only applying to high-stakes decisions.

### Proposed approach

Selective best-of-N: not every turn, only when stakes are high.

1. Normal turns: single sample (temperature=0 or low).
2. High-stakes triggers: unfamiliar task, contradictory context, tool call with side effects.
3. On trigger: sample N=4 at temperature=0.7, select by self-consistency on action + validity check.
4. Fallback: if no majority, escalate to LLM-as-judge tiebreaker.

## Open Questions

- How to detect "high-stakes" turns automatically? Entropy of first sample? Semantic distance from recent context?
- Should N be adaptive (start small, increase if no consensus)?
- Is self-consistency on tool choice sufficient, or do we need to also compare arguments?
- Cost/benefit threshold: at what per-turn cost does best-of-4 stop being worthwhile?
