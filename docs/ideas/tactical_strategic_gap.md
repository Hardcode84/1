# Tactical vs Strategic Reasoning Gap in LLM Agents

## Problem

LLM agents (including coding assistants) excel at local, tactical tasks — refactoring code, renaming variables, extracting functions — but consistently miss broader architectural implications. When asked "add X to Y," they execute the transformation fluently without questioning whether it changes the invariants of Y.

Example: asked to fire memory extraction during user nudges (in addition to reflection points), the agent did it cleanly but didn't flag that user nudges and reflection points serve fundamentally different purposes. A human noticed the "bad vibes" immediately.

## Root Causes

### Next-token prediction rewards local coherence

The training objective optimizes for "what's the next right step," not "should we be walking this direction." Each local decision is correct in isolation; the failure is compositional.

### Sycophancy bias from RLHF

Instruction-tuned models prioritize helpfulness over critical reasoning. Without explicit pushback training, models comply with reasonable-sounding requests even when they have the knowledge to question them. Research shows models accurately reject "1 + 2 = 5" unprompted but agree with it when the user asserts it.

### Compositionality collapse

Strategic reasoning benchmarks show models achieving 90%+ accuracy on atomic games but dropping below 20% on compositional structures. The gap between understanding a rule and applying it across interacting subsystems is fundamental, not a scaling issue.

## Research

- [Survey of Strategic Reasoning with LLMs](https://openreview.net/pdf?id=iMqJsQ4evS) — compositionality collapse in game-theoretic settings.
- [The AI Paradox: Can Explain But Can't Execute](https://medium.com/@san_24295/the-ai-paradox-llms-can-explain-a-winning-strategy-but-cant-execute-it-here-s-the-missing-piece-ca12a5b64677) — knowledge-execution gap.
- [Coherence Boosting](https://aclanthology.org/2022.acl-long.565/) — models underweight distant context in next-token prediction.
- [Sycophancy in LLMs: Causes and Mitigations](https://arxiv.org/html/2411.15287v1) — RLHF helpfulness-accuracy tension.

## Mitigation: Cross-Context Critic

The most promising mitigation is a separate critic that reviews actions with fresh context. Same model is fine — what matters is breaking the reasoning chain that led to the decision.

### Prior Art

- **Cross-model critique.** [Enhancing LLM Reasoning via Critique Models](https://arxiv.org/html/2411.16579v1) — a trained 3B critic can supervise actors of various sizes. Cross-model correction outperforms self-correction because the critic doesn't share the generator's biases.
- **CRITIC framework.** [CRITIC](https://arxiv.org/abs/2305.11738) — model validates its own outputs via external tools (code execution, search), then revises. Key finding: LLMs alone cannot reliably self-critique without external grounding.
- **Constitutional AI.** [Anthropic's CAI](https://arxiv.org/abs/2212.08073) — critique against explicit principles, then rewrite. Applied at training time, but the pattern (check output against stated rules) works at runtime too.
- **Self-correction limits.** [When Can LLMs Actually Correct Their Own Mistakes?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/) — critical survey showing self-correction is unreliable without external feedback. The same model that made the mistake often can't see it.
- **Intrinsic self-critique for planning.** [Enhancing LLM Planning through Intrinsic Self-Critique](https://arxiv.org/html/2512.24103v1) — improves planning from 49.8% to 89.3%, but only for structured tasks, not open-ended architectural judgment.

### Proposed Design for Mindloop

A cheap model with a different context window sees only:
1. The action summary (what was changed and why).
2. Relevant design docs and architectural invariants.
3. A prompt: "Does this change violate any stated invariants? What side effects might the author have missed?"

The fresh context is the key mechanism — it breaks the local-coherence trap because the critic wasn't in the reasoning chain that produced the decision. This is essentially the "code review" pattern: a reviewer catches things the author can't because they read the diff, not the thought process.

Open questions:
- When to fire the critic — every tool call is too expensive, every reflection point might work.
- What to do with the critique — inject as a system message? Log for human review?
- How to avoid the critic becoming sycophantic toward the actor's stated reasoning.

## Other Mitigations

- **Explicit architectural invariants** in code comments and design docs, so the agent (and its future self) can check proposed changes against them.
- **Reflection prompts that ask "why"** not just "what" — nudging the agent to justify changes in terms of design intent, not just mechanical correctness.
- **Human review at architectural boundaries** — tactical changes within a subsystem are safe to automate; changes that cross subsystem boundaries need human "bad vibes" checks.
