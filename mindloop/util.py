"""Shared utilities."""

import uuid
from pathlib import Path

# Approximate characters per token for budget estimation.
CHARS_PER_TOKEN = 4

# Default thread pool size for the global executor.
DEFAULT_WORKERS = 8

# Cosine-similarity threshold for dedup gating during extraction and rebuild.
DEDUP_THRESHOLD = 0.7

# System message prefixes to filter from logs and recaps.
SKIP_PREFIXES = ("[stop]", "[stats]", "Warning:")


def generate_session_name(sessions_dir: Path) -> str:
    """Generate a unique random session name (8-char hex)."""
    name = uuid.uuid4().hex[:8]
    while (sessions_dir / name).exists():
        name = uuid.uuid4().hex[:8]
    return name


def noop(_msg: str) -> None:
    """No-op log callback."""
    pass
