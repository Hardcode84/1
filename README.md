# mindloop

Autonomous agent toolkit powered by OpenRouter.

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

Runs an autonomous agent loop driven by the built-in system prompt. The agent explores the working directory until it produces a final response.

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
