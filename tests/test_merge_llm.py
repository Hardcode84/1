"""Tests for mindloop.merge_llm."""

from unittest.mock import patch

from mindloop.merge_llm import merge_texts, should_merge


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


# --- merge_texts ---


def test_merge_texts_two_calls() -> None:
    responses = [
        {"content": "Merged content here."},
        {"content": "ABSTRACT: A short summary.\nSUMMARY: A longer explanation."},
    ]
    with patch("mindloop.merge_llm.chat", side_effect=responses) as mock_chat:
        result = merge_texts("chunk a", "chunk b")
    assert mock_chat.call_count == 2
    assert result.text == "Merged content here."
    assert result.abstract == "A short summary."
    assert result.summary == "A longer explanation."


def test_merge_texts_second_call_has_conversation_context() -> None:
    responses = [
        {"content": "merged"},
        {"content": "ABSTRACT: abs\nSUMMARY: sum"},
    ]
    with patch("mindloop.merge_llm.chat", side_effect=responses) as mock_chat:
        merge_texts("a", "b")
    # Second call should have 3 messages: user, assistant (merged), user (summarize).
    second_call_messages = mock_chat.call_args_list[1][0][0]
    assert len(second_call_messages) == 3
    assert second_call_messages[1]["role"] == "assistant"
    assert second_call_messages[1]["content"] == "merged"


def test_merge_texts_prefer_a() -> None:
    responses = [{"content": "merged"}, {"content": "ABSTRACT: a\nSUMMARY: s"}]
    with patch("mindloop.merge_llm.chat", side_effect=responses) as mock_chat:
        merge_texts("a", "b", prefer="a")
    system = mock_chat.call_args_list[0][1]["system_prompt"]
    assert "Chunk A is the primary source" in system


def test_merge_texts_prefer_b() -> None:
    responses = [{"content": "merged"}, {"content": "ABSTRACT: a\nSUMMARY: s"}]
    with patch("mindloop.merge_llm.chat", side_effect=responses) as mock_chat:
        merge_texts("a", "b", prefer="b")
    system = mock_chat.call_args_list[0][1]["system_prompt"]
    assert "Chunk B is the primary source" in system


def test_merge_texts_unparseable_summary() -> None:
    responses = [
        {"content": "merged text"},
        {"content": "The model did something unexpected."},
    ]
    with patch("mindloop.merge_llm.chat", side_effect=responses):
        result = merge_texts("a", "b")
    assert result.text == "merged text"
    assert result.abstract == ""
    assert result.summary == ""
