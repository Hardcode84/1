# mindloop

A lightweight agent loop with tool use, streaming, and token budgeting. Built on top of the OpenRouter API.

## Setup

```bash
pip install -e .
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

### Chat

```bash
mindloop-chat
# or
python -m mindloop.cli.chat
```

### Agent

```bash
mindloop-agent
# or
python -m mindloop.cli.agent
```

Runs an autonomous agent loop with tool use, streaming, and token budgeting.

### Log chunker

```bash
# Structural chunking only.
mindloop-chunk logs/20260208_171554.log

# With embedding-based merging.
mindloop-chunk logs/20260208_171554.log --embed

# Custom time gap threshold (seconds).
mindloop-chunk logs/20260208_171554.log --gap 60

# Summarize each chunk.
mindloop-chunk logs/20260208_171554.log --summarize

# Embed, merge, then summarize.
mindloop-chunk logs/20260208_171554.log --embed --summarize
```

### Build semantic database

```bash
# Index all markdown files into memory.db.
mindloop-build "docs/**/*.md"

# Custom database path and model.
mindloop-build "logs/*.jsonl" --db knowledge.db --model deepseek/deepseek-v3.2
```

Processes each file through the full pipeline: parse, chunk, summarize, and save into the semantic database with automatic merge. Use `-v` for verbose output showing each stage.

### Query semantic database

```bash
# Search for relevant memories.
mindloop-query "how does chunking work"

# Top 3 results from a custom database.
mindloop-query "merge algorithm" --db knowledge.db -k 3

# Show full chunk text.
mindloop-query "embeddings" -v
```

Results are ranked using hybrid search (embedding cosine similarity + BM25 keyword matching via FTS5), combined with Reciprocal Rank Fusion.

Use `--original` to search original (unmerged) chunks, or `--tree ID` to show the full merge lineage tree for a chunk.

### Dump database

```bash
# Print all chunks to stdout.
mindloop-dump

# Write to file from a custom database.
mindloop-dump --db knowledge.db -o dump.txt
```

Dumps every chunk (active and inactive) with its abstract, summary, full text, source lineage, and timestamp.

### Session recap

```bash
# Generate recap from an agent log and print to stdout.
mindloop-recap logs/001_agent_20260210_120000.jsonl

# Write to file.
mindloop-recap logs/001_agent_20260210_120000.jsonl -o _recap.md

# Custom token budget and model.
mindloop-recap logs/001_agent_20260210_120000.jsonl --budget 500 --model deepseek/deepseek-v3.2
```

Collapses tool calls into concise natural-language descriptions, chunks and summarizes the session, then selects the most relevant summaries with recency bias within a token budget. Recaps are also generated automatically at session shutdown and injected into the next instance's system prompt.

### List sessions

```bash
# List all sessions with metadata.
mindloop-sessions

# Custom sessions directory.
mindloop-sessions --dir path/to/sessions
```

Shows each session's name, instance count, date range, last exit status (clean/crashed/tokens/iterations), and whether notes exist.

### Messaging

```bash
# Send a message to a session's inbox.
mindloop-message send --session myagent --from "Alice" --title "Status?" "How is the task going?"

# List inbox messages.
mindloop-message list --session myagent

# List outbox (agent replies).
mindloop-message list --session myagent --outbox
```

The agent sees new messages on startup and can browse with `message_list`, read with `message_read`, and reply with `message_send`. Inbox is frozen at session start; outbox updates live.

### Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=mindloop
```

### As a library

```python
from mindloop.client import chat, get_embeddings
from mindloop.chunker import parse_turns, chunk_turns, merge_chunks
```
