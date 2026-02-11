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

Returns ranked results with cosine similarity scores, abstracts, summaries, and merge lineage.

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
