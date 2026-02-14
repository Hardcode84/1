"""Quote loading and diversity nudges for the agent loop."""

import json
import random
from datetime import date
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_QUOTE_FILES = ["stoic-quotes.json", "enchiridion_prompts.json"]

_cached_quotes: list[dict[str, str]] | None = None


def load_quotes() -> list[dict[str, str]]:
    """Load and merge all quote JSON files from data/. Cached after first call."""
    global _cached_quotes
    if _cached_quotes is not None:
        return _cached_quotes
    quotes: list[dict[str, str]] = []
    for name in _QUOTE_FILES:
        path = _DATA_DIR / name
        if path.exists():
            quotes.extend(json.loads(path.read_text()))
    _cached_quotes = quotes
    return quotes


def _format(entry: dict[str, str]) -> str:
    """Format a quote entry as a readable string."""
    text = entry["text"]
    author = entry.get("author", "")
    source = entry.get("source", "")
    attribution = ", ".join(p for p in [author, source] if p)
    if attribution:
        return f'"{text}" — {attribution}'
    return f'"{text}"'


def quote_of_the_day() -> str:
    """Return a deterministic quote for the current calendar day."""
    quotes = load_quotes()
    if not quotes:
        return ""
    rng = random.Random(date.today().toordinal())
    return _format(rng.choice(quotes))


class NudgePool:
    """Session-scoped sampler that cycles through quotes without repeats."""

    def __init__(self, quotes: list[dict[str, str]] | None = None) -> None:
        self._quotes = quotes if quotes is not None else load_quotes()
        self._indices: list[int] = []
        self._pos = 0
        self._rng = random.Random()
        self._shuffle()

    def _shuffle(self) -> None:
        self._indices = list(range(len(self._quotes)))
        self._rng.shuffle(self._indices)
        self._pos = 0

    def next(self) -> str:
        """Return the next formatted quote, reshuffling when exhausted."""
        if not self._quotes:
            return ""
        if self._pos >= len(self._indices):
            self._shuffle()
        entry = self._quotes[self._indices[self._pos]]
        self._pos += 1
        return _format(entry)
