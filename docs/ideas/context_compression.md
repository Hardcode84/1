# Context Compression

Strategies for managing growing context within and across agent sessions.

## Problem

Agent sessions accumulate long conversation histories. Tool outputs (file reads, search results) dominate token usage and become stale quickly. Research (Chroma "Context Rot", 2025) shows performance *degrades* with longer inputs — more context is not always better. Careful curation beats raw context length.

## Current state

Mindloop already does some things well:

- **Cross-session recaps.** `collapse_messages` + summarizer compresses previous sessions into ~budget tokens for the next instance.
- **Recall on demand.** Semantic memory is searched, not dumped into the prompt. This avoids context pollution.
- **Reflection checkpoints.** Natural boundaries for extraction and compression.

The gap is **within-session context management** — as a session runs, the message list grows unbounded until the token budget is hit.

## Approaches (ranked by effort/impact)

### 1. Observation masking (low effort, high impact)

Replace old tool outputs with short placeholders, keep last N turns verbatim.

```
# Before (500 tokens):
{"role": "tool", "content": "<full file contents, 500 lines>"}

# After (20 tokens):
{"role": "tool", "content": "[read main.py: 487 lines]"}
```

JetBrains research (NeurIPS 2025 DL4Code) found this gives 50%+ cost reduction and sometimes *better* accuracy than full context. With Qwen3-Coder 480B, observation masking boosted solve rates by 2.6% while being 52% cheaper. Outperformed LLM-based summarization in 4 of 5 configurations.

Implementation: iterate messages oldest-first, replace tool result content with a summary line for any message older than the recent window. Keep the tool name and a size hint. The agent's reasoning and actions stay intact — only bulky observations get masked.

Window size: ~10 recent turns seems standard across Claude Code, Codex CLI, and OpenCode.

References:
- [JetBrains: Efficient Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [JetBrains: The Complexity Trap (NeurIPS 2025)](https://github.com/JetBrains-Research/the-complexity-trap)

### 2. Structured compaction at reflection points (medium effort)

When context grows past a threshold, summarize the old portion into explicit sections:

```markdown
## Compaction summary
- **State:** Working on feature X, file Y modified.
- **Decisions:** Chose approach A over B because ...
- **Files touched:** foo.py (added func), bar.py (fixed bug).
- **Failed approaches:** Tried C, didn't work because ...
- **Next steps:** Need to test Z.
```

Factory.ai (2025) showed structured summaries (score 3.70/5) beat Anthropic's built-in compaction (3.44/5) and OpenAI's (3.35/5). The key is preserving *decisions and failures*, not just current state.

This could fire at the same reflection points that already trigger extraction and critic review.

### 3. Focus architecture (medium effort)

From arXiv:2601.07190 (Jan 2026). The agent itself decides when to compress via explicit primitives (`start_focus` / `complete_focus`). The agent introspects, summarizes its recent trajectory into high-level learnings, and the raw logs are dropped. 22-57% token savings with identical accuracy.

Fits naturally with mindloop's tool-based architecture — could be a new tool:

```
focus(summary="Explored approaches A and B for X. A works, B fails because Y.")
```

The agent calls it when it recognizes a phase of work is complete.

### 4. LLM-based summarization of old turns (medium effort, mixed results)

Use the summarizer model to rewrite old turns into compressed form. The infrastructure exists (summarizer model, `collapse_messages`).

**Caution:** JetBrains found this performed worse than observation masking because:
- Lossy in unpredictable ways — summarizer doesn't know what the agent needs later.
- Smooths over failure signals — agents ran 15% longer because summaries "cleaned up" signs of failure.
- Added latency and cost.
- Hallucination risk.

If used at all, should only fire at natural boundaries (reflection points), not continuously. And should preserve failure information explicitly.

### 5. LLMLingua-2 token compression (medium effort)

Microsoft's `llmlingua` library uses a BERT-level encoder to score tokens by information content and drop low-value ones. 2-10x compression, 95%+ accuracy retention, 3-6x faster than v1.

Could run as a preprocessing step on old messages before sending to OpenRouter. Task-agnostic, works with any API.

Downside: adds a dependency and preprocessing step. May interact poorly with structured content (code, JSON).

Reference: [github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)

### 6. ACON gradient-free guideline optimization (high effort)

From arXiv:2510.00615 (Oct 2025). Optimizes natural-language compression guidelines via a feedback loop: when compressed context leads to failure, an LLM analyzes what was lost and updates the guideline. 26-54% memory reduction, 95%+ accuracy.

Interesting but complex. Would require running tasks with both full and compressed context to generate training signal.

## Cross-session: critic-pinned turns

The recap system compresses a full session into ~500 tokens of narrative. This is good for big-picture continuity but loses exact details — error tracebacks, specific reasoning chains, code snippets that worked. Raw turns preserve what summaries can't.

### The idea

The critic model already reviews the agent's actions at reflection points with fresh context. Extend its output to also identify which turns contain information critical for the next session. At session end, collect all critic-pinned turns and write them to `_pinned_turns.json`. The next instance loads them alongside the recap.

### Why the critic, not the agent or a post-hoc pass

- **The agent is too deep in the weeds.** Everything feels important when you're doing it.
- **A post-hoc pass sees everything at once** and has to guess what the next session will need.
- **The critic has distance.** It reviews bounded windows with fresh context — the right vantage point to judge significance.
- **Nearly free.** The critic call is already happening. Just extend the output format.

### Implementation sketch

Extend the critic prompt:

```
Does this window contain anything critical for the next session instance
(key decisions, surprising errors, breakthroughs, unresolved problems)?
If yes, explain briefly what should be preserved and why.
```

The critic returns its existing review text plus an optional `"pin_reason"` field:

```json
{"review": "The agent is ...", "pin_reason": "Found root cause of FTS5 rebuild bug."}
```

If `pin_reason` is present, the entire reflection window gets pinned. The windows are small (bounded by reflection interval), so pinning the whole window is cheap — no need to pick specific turns within it.

We know the absolute message indices that map to each window since we built it. At session end:

1. Collect all pinned windows, deduplicate overlapping ranges.
2. Write pinned messages to `_pinned_turns.json` in the session root.
3. Next instance injects them into the system prompt as a clearly delimited section:

```
# Pinned context from previous instance

The following are raw conversation turns from your previous instance that
the critic flagged as important. This is historical context, not your
current session. Your current session starts after the system prompt.

## Window pinned because: Found root cause of FTS5 rebuild bug.

[assistant] I'll check the migration code...
[tool:read] migration.py contents...
[assistant] The issue is that rebuild() is never called after ALTER TABLE.

## Window pinned because: User rejected approach B, wants approach A.

[user] No, don't use the rename pattern...
[assistant] Understood, I'll use ALTER TABLE ADD COLUMN instead.
```

This goes into the system prompt alongside the recap and notes — clearly marked as historical. The actual conversation starts empty.

### Properties

- No extra LLM call — piggybacks on the existing critic.
- Progressive — pins accumulate at each reflection point, not a single post-hoc pass.
- Naturally capped — the critic sees bounded windows and pins at most 3 per window.
- Complements the recap — recap gives narrative, pinned turns give exact details.

### Relation to pinned memories

Pinned memories (see `pinned_memories.md`) preserve distilled *facts* in the semantic DB. Pinned turns preserve raw *conversation moments* in the log. They're complementary: a turn might lead to a pinned memory, but the raw turn carries details (exact error text, surrounding reasoning) the memory might not.

## Recommendation

Start with **observation masking** (#1). It is the simplest, cheapest, and empirically strongest approach. Implementation is ~50 lines in the agent loop. Combine with the existing recap system for cross-session compression.

Add **structured compaction** (#2) at reflection points as a second phase if observation masking alone isn't enough.

The agent-driven **focus** approach (#3) is worth exploring once the basics are in place — it aligns with the existing reflection/extraction architecture and lets the agent manage its own context.

For cross-session transfer, **critic-pinned turns** are the most promising approach — zero extra cost, leverages the existing critic infrastructure, and preserves exact details that recaps lose.
