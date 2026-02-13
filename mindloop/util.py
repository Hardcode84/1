"""Shared utilities."""

# Approximate characters per token for budget estimation.
CHARS_PER_TOKEN = 4

# Default thread pool size for parallel summarization.
DEFAULT_WORKERS = 4

# System message prefixes to filter from logs and recaps.
SKIP_PREFIXES = ("[stop]", "[stats]", "Warning:")


def noop(_msg: str) -> None:
    """No-op log callback."""
    pass
