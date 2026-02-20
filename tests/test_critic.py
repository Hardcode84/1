"""Unit tests for mindloop.critic response parsing."""

from mindloop.critic import _parse_critic_response


def test_ok_no_pin() -> None:
    """Both ok and no_pin → both fields empty."""
    result = _parse_critic_response("ok\nno_pin")
    assert result.concern == ""
    assert result.pin_reason == ""


def test_concern_no_pin() -> None:
    """Concern line with no_pin → concern only."""
    result = _parse_critic_response("The agent is looping.\nno_pin")
    assert result.concern == "[critic] The agent is looping."
    assert result.pin_reason == ""


def test_ok_with_pin() -> None:
    """Ok review with pin reason → pin only."""
    result = _parse_critic_response("ok\nKey decision about architecture.")
    assert result.concern == ""
    assert result.pin_reason == "Key decision about architecture."


def test_concern_with_pin() -> None:
    """Both concern and pin reason."""
    result = _parse_critic_response("Stuck in a loop.\nKey decision.")
    assert result.concern == "[critic] Stuck in a loop."
    assert result.pin_reason == "Key decision."


def test_single_line_ok() -> None:
    """Single line 'ok' → graceful, no pin."""
    result = _parse_critic_response("ok")
    assert result.concern == ""
    assert result.pin_reason == ""


def test_single_line_concern() -> None:
    """Single line concern → no pin."""
    result = _parse_critic_response("Agent is stuck.")
    assert result.concern == "[critic] Agent is stuck."
    assert result.pin_reason == ""


def test_empty_string() -> None:
    """Empty string → both empty."""
    result = _parse_critic_response("")
    assert result.concern == ""
    assert result.pin_reason == ""


def test_whitespace_between_lines() -> None:
    """Blank lines between content lines are ignored."""
    result = _parse_critic_response("ok\n\n\nImportant discovery.")
    assert result.concern == ""
    assert result.pin_reason == "Important discovery."
