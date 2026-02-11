"""Tests for mindloop.merge_llm."""

from unittest.mock import patch

from mindloop.merge_llm import should_merge


def test_high_similarity_skips_llm() -> None:
    with patch("mindloop.merge_llm.chat") as mock_chat:
        assert should_merge("a", "b", cosine_sim=0.95) is True
    mock_chat.assert_not_called()


def test_low_similarity_skips_llm() -> None:
    with patch("mindloop.merge_llm.chat") as mock_chat:
        assert should_merge("a", "b", cosine_sim=0.1) is False
    mock_chat.assert_not_called()


def test_borderline_calls_llm_yes() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": "yes"}) as mock_chat:
        assert should_merge("a", "b", cosine_sim=0.5) is True
    mock_chat.assert_called_once()


def test_borderline_calls_llm_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": "no"}) as mock_chat:
        assert should_merge("a", "b", cosine_sim=0.5) is False
    mock_chat.assert_called_once()


def test_unexpected_response_treated_as_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": "maybe"}):
        assert should_merge("a", "b", cosine_sim=0.5) is False


def test_empty_response_treated_as_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": ""}):
        assert should_merge("a", "b", cosine_sim=0.5) is False


def test_none_content_treated_as_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={}):
        assert should_merge("a", "b", cosine_sim=0.5) is False


def test_custom_thresholds() -> None:
    with patch("mindloop.merge_llm.chat") as mock_chat:
        # Above custom high — no LLM call.
        assert should_merge("a", "b", cosine_sim=0.75, high=0.7) is True
        mock_chat.assert_not_called()
        # Below custom low — no LLM call.
        assert should_merge("a", "b", cosine_sim=0.3, low=0.4) is False
        mock_chat.assert_not_called()
