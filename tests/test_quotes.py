"""Tests for mindloop.quotes."""

from datetime import date
from unittest.mock import patch

from mindloop.quotes import NudgePool, _format, load_quotes, quote_of_the_day


def test_load_quotes() -> None:
    """load_quotes returns non-empty list with text keys."""
    quotes = load_quotes()
    assert len(quotes) > 0
    for q in quotes:
        assert "text" in q


def test_quote_of_the_day_deterministic() -> None:
    """Same date produces the same quote; different dates may differ."""
    d1 = date(2025, 6, 15)
    d2 = date(2025, 12, 25)

    with patch("mindloop.quotes.date", wraps=date) as mock_date:
        mock_date.today.return_value = d1
        q1 = quote_of_the_day()
        q2 = quote_of_the_day()
    assert q1 == q2

    with patch("mindloop.quotes.date", wraps=date) as mock_date:
        mock_date.today.return_value = d2
        q3 = quote_of_the_day()
    # Different dates should (almost certainly) give a different quote.
    # With 246 quotes the collision chance is ~0.4%.
    assert isinstance(q3, str) and len(q3) > 0


def test_nudge_pool_no_repeat() -> None:
    """Drawing N quotes from a pool of N produces all unique values."""
    quotes = [{"text": f"q{i}", "author": f"a{i}"} for i in range(5)]
    pool = NudgePool(quotes)
    drawn = [pool.next() for _ in range(5)]
    assert len(set(drawn)) == 5


def test_nudge_pool_reshuffles() -> None:
    """After exhausting the pool, continues returning quotes."""
    quotes = [{"text": f"q{i}", "author": f"a{i}"} for i in range(3)]
    pool = NudgePool(quotes)
    # Draw more than pool size.
    drawn = [pool.next() for _ in range(9)]
    assert all(isinstance(q, str) and len(q) > 0 for q in drawn)
    # Each cycle of 3 should have all unique quotes.
    for start in range(0, 9, 3):
        cycle = drawn[start : start + 3]
        assert len(set(cycle)) == 3


def test_nudge_pool_format_stoic() -> None:
    """Stoic-quote format (text + author) produces valid string."""
    entry = {"text": "Be tolerant.", "author": "Marcus Aurelius"}
    result = _format(entry)
    assert "Be tolerant." in result
    assert "Marcus Aurelius" in result


def test_nudge_pool_format_enchiridion() -> None:
    """Enchiridion format (text + author + source + type) produces valid string."""
    entry = {
        "type": "quote",
        "text": "Demand not that events happen as you wish.",
        "author": "Epictetus",
        "source": "Enchiridion VIII",
    }
    result = _format(entry)
    assert "Demand not" in result
    assert "Epictetus" in result
    assert "Enchiridion VIII" in result


def test_format_distilled() -> None:
    """Distilled prompts render as bare text with parenthetical source."""
    entry = {
        "type": "distilled",
        "text": "What truly lies within your control?",
        "source": "Enchiridion I",
    }
    result = _format(entry)
    assert result == "What truly lies within your control? (Enchiridion I)"
    assert '"' not in result


def test_format_distilled_no_source() -> None:
    """Distilled prompt without source renders as bare text."""
    entry = {"type": "distilled", "text": "Reflect on your nature."}
    assert _format(entry) == "Reflect on your nature."


def test_format_no_author() -> None:
    """Entry with no author or type still formats correctly."""
    entry = {"text": "Anonymous wisdom."}
    result = _format(entry)
    assert "Anonymous wisdom." in result
    assert "—" not in result


def test_nudge_pool_empty() -> None:
    """Empty quote list returns empty strings without error."""
    pool = NudgePool([])
    assert pool.next() == ""
