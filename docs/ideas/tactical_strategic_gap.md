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

## Implications for Mindloop

When building an autonomous agent framework, this gap matters doubly — the agent *building* the framework has it, and the agent *running in* the framework has it. Mitigations to consider:

- **Explicit architectural invariants** in code comments and design docs, so the agent (and its future self) can check proposed changes against them.
- **Reflection prompts that ask "why"** not just "what" — nudging the agent to justify changes in terms of design intent, not just mechanical correctness.
- **Human review at architectural boundaries** — tactical changes within a subsystem are safe to automate; changes that cross subsystem boundaries need human "bad vibes" checks.
