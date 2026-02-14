# Prompt Distillation from Text Corpora

## Goal

Convert a corpus of texts into a pool of diverse reflective prompts for the synthetic diversity injection mechanism (see `synthetic_diversity.md`).

## Source Material

### Ready-made: storopoli/stoic-quotes (CC0)

[github.com/storopoli/stoic-quotes](https://github.com/storopoli/stoic-quotes) — 246 stoic quotes, CC0 1.0 (public domain), clean JSON with `text` + `author` fields. 8 authors (Marcus Aurelius, Epictetus, Seneca, Zeno, Musonius Rufus, Diogenes, etc.). Ready to use as-is — no processing pipeline needed for the direct quotes portion. Eliminates the need for Gutenberg download + extraction for quotes.

### Optional: Project Gutenberg texts for distilled prompts

If we want rephrased reflective prompts on top of direct quotes, source from public domain texts (~80k words total):

- Marcus Aurelius, *Meditations* (~45k words) — self-directed notes, almost every passage is a prompt.
- Epictetus, *Enchiridion* (~10k words) — 53 short chapters of direct imperatives. Densest source.
- La Rochefoucauld, *Maxims* (~20k words) — ~500 one-line aphorisms. Nearly zero processing needed.
- Lao Tzu, *Tao Te Ching* (~5k words) — 81 short poems. Maximally different worldview.

All Apache 2.0 compatible (public domain, strip PG headers).

## Pipeline

This is a one-off process, tailored per book. No need for a reusable CLI tool.

### 1. Prep (bash, per book)

Strip PG headers/footers, split into passages. Each book has its own natural structure — numbered passages, chapters, aphorisms, verses. Use `sed`/`awk` to extract, one passage per line or per file. Tailored per book, not a generic tool.

```bash
# Example: strip PG header/footer.
sed -n '/\*\*\* START/,/\*\*\* END/p' meditations.txt | tail -n +2 | head -n -1 > clean.txt
# Split on paragraph breaks, filter short lines, etc.
```

### 2. Extractive pass (grep/awk)

Pull high-value passages directly:

- Questions: `grep '?$'`
- Imperatives: lines starting with a verb.
- Strong claims: lines with universals ("always", "never", "the nature of").

Stoic texts are dense with these — high yield expected.

### 3. LLM distillation (one-time, ~$1-2)

For remaining passages, one-shot LLM call per chunk:

> "Rephrase as a reflective prompt for an autonomous agent. Write as a question or challenge. One line only."

~300-400 cheap model calls using existing `chat()`. Can run in parallel with `ThreadPoolExecutor`.

### 4. Diversity selection (Python script)

1. Embed all candidates via `get_embeddings()`.
2. Greedy max-min distance: pick prompt furthest from already-selected set, repeat.
3. Select ~200 prompts. Use no-repeat sampling at runtime (reset across sessions) so 200 = 200 guaranteed unique turns.
4. If pool needs to grow later: parameterized templates or runtime composition of two prompts (~20k combos from 200).

### 5. Output

Static JSON shipped with mindloop. Two types of entries:

**Direct quotes** — preserve original voice and author attribution. Triggers associations with the author's tradition, gives the diversity injection more semantic depth. Cheaper to produce (pure extractive).

**Distilled prompts** — rephrased as questions/challenges. More directly actionable, less anchored in a specific voice.

Mix both in the pool, roughly 50/50.

```json
[
  "Quote of the day: 'Waste no more time arguing about what a good man should be. Be one.' — Marcus Aurelius",
  "Quote of the day: 'The tao that can be told is not the eternal Tao.' — Lao Tzu",
  "What assumptions are you making right now?",
  "What would change if you approached this from the opposite direction?"
]
```

## Open Questions

- Should the pool grow over time as the agent encounters new material?
- Weight prompts by effectiveness? Track which prompts actually break loops vs get ignored.
- Optimal ratio of direct quotes to distilled prompts? Start 50/50, tune based on observed effect.

## Quote of the Day

In addition to using the pool for nudge diversity, inject a daily quote into the system prompt at startup. Select deterministically using the current date as seed (`random.seed(date.today())`), so all instances on the same day see the same quote. Changes daily. Append to system prompt as `# Quote of the day\n"..." — Author`.
