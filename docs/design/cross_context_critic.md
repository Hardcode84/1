# Cross-Context Critic

A cheap model periodically reviews the agent's recent actions with fresh context, catching reasoning failures that self-reflection misses.

## Problem

LLM agents fall into predictable failure modes that are invisible from inside the reasoning chain:

- **Circular tool use.** Repeating the same action expecting different results.
- **Thread loss.** Drifting from self-defined goals without noticing.
- **Sycophancy toward own reasoning.** Doubling down on a failing approach because each local step still "makes sense."
- **Premature completion.** Calling `done` before the task is actually finished.
- **Stagnation.** Generating tokens without meaningful progress.

Self-reflection is unreliable for these because the same reasoning chain that produced the mistake evaluates whether it was a mistake ([When Can LLMs Actually Correct Their Own Mistakes?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/)). The fix is fresh context — a separate model that sees only what happened, not the thinking that led to it.

## Prior art

- **Cross-model critique.** A trained 3B critic can supervise larger actors because it doesn't share the generator's biases ([Enhancing LLM Reasoning via Critique Models](https://arxiv.org/html/2411.16579v1)). Even CriticGPT (same base model, different fine-tuning) is preferred over self-critique 63% of the time ([LLM Critics Help Catch LLM Bugs](https://arxiv.org/abs/2407.00215)).
- **SOFAI-LM.** A rule-based metacognitive controller monitors a fast LLM and provides iterative feedback. The controller doesn't need to be powerful — fresh context and structured evaluation outperform the LLM alone ([arXiv:2508.17959](https://arxiv.org/abs/2508.17959)).
- **AI Control / Trusted Monitor.** A weaker trusted model reviews outputs of a stronger untrusted model, achieving 92% safety at 94% usefulness ([Redwood Research](https://arxiv.org/pdf/2312.06942)).
- **Process Reward Models.** Score each intermediate step, not just final output. Small 3B models outperform GPT-4o on agent benchmarks when trained for step-level oversight ([AgentPRM](https://arxiv.org/abs/2502.10325)).
- **Tactical vs Strategic Gap** (`tactical_strategic_gap.md`). Identifies the core issue: next-token prediction rewards local coherence, not global correctness. Fresh context breaks the local-coherence trap.

## Design

### What the critic sees

The collapsed message window — agent text and tool call summaries, produced by `collapse_messages()`. This is already computed at every `on_extract` checkpoint. Example:

```
Bot: I'll start by reading the configuration.
Bot: Read config.py (42 lines).
Bot: The config uses a flat dict, I'll refactor to dataclass.
Bot: Wrote config.py.
Bot: Read config.py (42 lines).
Bot: Now I need to update all callers...
```

What's included:
- Agent text responses (what it said).
- Tool calls with condensed results (what it did).

What's excluded:
- Reasoning/thinking tokens (would pull the critic into the same chain).
- Raw tool results (too long, same local-coherence risk).
- System prompt and conversation history before the window.

The fresh context is the key mechanism. The critic wasn't in the reasoning chain that produced these actions, so it reads them like a code reviewer reads a diff.

### When it fires

At reflection intervals (`_REFLECT_INTERVAL = 5` tool turns), alongside intrusive recall. Both mechanisms already share the same trigger point and the same accumulated text buffer (`_pending_texts`).

Firing every 5 tool turns is cheap enough to be always-on and frequent enough to catch loops before they waste significant budget.

### Prompt

```
You're reviewing an autonomous agent's recent actions.
The agent sets its own goals — your job is to check
whether it's reasoning well, not whether its goals are correct.

Recent activity:
{collapsed_turns}

Respond with ONLY one of:
- "ok" if the agent is making meaningful progress
- A single sentence describing the concern if you see:
  circular actions, lost thread, stagnation, or
  the agent doubling down on a failing approach
```

Key properties:
- **No external goals or invariants.** The critic evaluates internal coherence, not external compliance.
- **Adversarial framing.** "Your job is to find problems" — not helpful-assistant mode.
- **Forced brevity.** "ok" or one sentence. Prevents the critic from generating lengthy agreement.
- **Structured failure modes.** Enumerated so the critic knows what to look for.

### Model

The summarizer model — already configured via `--summarizer-model`, already cheap. The critic doesn't need to be smart; it needs to not share the agent's reasoning context. SOFAI-LM demonstrated that a rule-based controller (no learned weights at all) outperforms standalone reasoning models when combined with iterative feedback.

### Where it lives

Standalone function in `mindloop/critic.py`:

```python
def critic_review(activity: str, model: str, log: Callable) -> str:
    """Review recent agent actions with fresh context."""
```

`MidSessionExtractor` has a thin wrapper that passes `self._last_query`, `self._model`, and `self._log`. Runs synchronously — a single cheap LLM call, not worth threading. Returns empty string or a formatted concern.

### How it's wired

The `on_reflect` callback in `cli/agent.py` already combines signals for injection into reflection nudges:

```python
def on_reflect():
    intrusive = mid_extract.intrusive_recall()
    critique = mid_extract.critic_review()
    # Combine both signals.
    parts = [s for s in (intrusive, critique) if s]
    return "\n\n".join(parts)
```

The combined output flows through `nudge_extra` into `_maybe_reflect()`, which injects it as a system message. No changes to `agent.py` needed.

### Output format

When the critic returns "ok", `critic_review()` returns empty string (no injection).

When the critic returns a concern, it's formatted as:

```
[critic] The agent appears to be reading the same file repeatedly without making changes.
```

The `[critic]` prefix distinguishes it from intrusive recall and other nudge content. The agent sees it as external feedback, not its own thought.

### Interaction with intrusive recall

Both fire at the same reflection point via the `_on_reflect` wrapper. Order matters:

1. `intrusive_recall()` runs first, consumes `_pending_texts`, saves the joined query to `self._last_query`.
2. `critic_review()` reads `self._last_query` set by step 1.

## Parameters

| Constant | Value | Purpose |
|---|---|---|
| `_CRITIC_MAX_TOKENS` | `100` | Cap on critic response length. |

No threshold tuning needed — the prompt forces binary output ("ok" or concern).

## Cost

- ~200-400 input tokens (collapsed window + prompt).
- ~20 output tokens ("ok" or one sentence).
- Summarizer model pricing (~$0.001 per call).
- Fires every 5 tool turns.
- 40 tool turns in a session = 8 critic calls = ~$0.008.

Negligible compared to the agent's own token usage.

## Limitations

- **No ground truth.** The critic evaluates coherence, not correctness. It can catch "doing the same thing twice" but not "doing the wrong thing once."
- **Collapsed context loses nuance.** "Read config.py (42 lines)" doesn't reveal what the agent was looking for. Some strategic failures require seeing the content.
- **Cheap model may miss subtle patterns.** A summarizer-tier model won't catch sophisticated reasoning failures. This is a feature — it keeps cost negligible — but sets a ceiling on critic quality.
- **Prompt sensitivity.** The forced-brevity prompt may cause the critic to default to "ok" too often. Calibration against real session logs will be needed.

## Future directions

- **Coherence metrics as pre-filter.** Run compression ratio and n-gram diversity (already implemented in `voting.py`) before the LLM call. If metrics are normal, skip the critic entirely. Only invoke the LLM when cheap signals suggest a problem.
- **Accumulated context.** Track critic history across reflection points — "you flagged stagnation last time, has the agent changed approach?" Currently each review is independent.
- **Critic-as-tool.** Let the agent invoke the critic voluntarily ("am I on track?") in addition to the periodic automatic review.
- **Disposition learning.** Feed critic outputs into the disposition extraction pipeline — if certain failure modes recur, they become part of the agent's self-understanding.

## Related

- `tactical_strategic_gap.md` — problem statement and research context.
- `intrusive_recall.md` — companion mechanism at the same trigger point.
- `coherence_detection.md` — cheap heuristic signals that could pre-filter critic calls.
- `best_of_n.md` — per-turn quality via candidate diversity; critic is per-window quality via fresh context.
