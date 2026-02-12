# CLAUDE.md

## Project

mindloop — autonomous AI agent framework with persistent semantic memory. Python 3.10+, OpenRouter API.

## Structure

```
agent.py           Agent loop: tool calls until done. Token budget, reflection nudges.
tools.py           Tool registry (OpenAI function-calling format) + built-in fs tools.
memory.py          SQLite + numpy storage. Hybrid search (cosine + FTS5 BM25, RRF).
semantic_memory.py Save with iterative merge loop. Specificity threshold prevents over-merging.
memory_tools.py    Agent-facing: remember, recall, recall_detail.
merge_llm.py       LLM-based merge decisions (three-tier: auto/LLM/never by similarity).
chunker.py         Parse JSONL/markdown into chunks, compact, merge by similarity.
summarizer.py      LLM-generated abstract + summary per chunk.
client.py          OpenRouter API wrapper (chat, embeddings, streaming, retry).
cli/               Six entry points: agent, chat, chunk, build, query, dump.
```

## Commands

```bash
pip install -e ".[dev]"
pytest
pre-commit run --all-files    # black + ruff + mypy strict
```

## Code style

- Comments end in full stop.
- Tool config lives on `ToolRegistry` instances, not global state.
- One-shot LLM calls use `cache_messages=False` (cache system prompt only).
- Filesystem sandboxing via `root_dir` + `blocked_dirs` on registry.

## Design docs

- `docs/design/` — current architecture.
- `docs/ideas/` — future proposals, not yet implemented.

## Environment

- `OPENROUTER_API_KEY` required.
