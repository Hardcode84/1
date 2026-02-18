"""Shared utilities."""

# Approximate characters per token for budget estimation.
CHARS_PER_TOKEN = 4

# Default thread pool size for the global executor.
DEFAULT_WORKERS = 8

# Cosine-similarity threshold for dedup gating during extraction and rebuild.
DEDUP_THRESHOLD = 0.7

# System message prefixes to filter from logs and recaps.
SKIP_PREFIXES = ("[stop]", "[stats]", "Warning:")


def noop(_msg: str) -> None:
    """No-op log callback."""
    pass
