"""Tests for mindloop.merge_llm."""

from unittest.mock import patch

from mindloop.merge_llm import merge_texts, should_merge


def test_llm_yes() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": "yes"}) as mock_chat:
        assert should_merge("a", "b", model="test-model") is True
    mock_chat.assert_called_once()


def test_llm_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": "no"}) as mock_chat:
        assert should_merge("a", "b", model="test-model") is False
    mock_chat.assert_called_once()


def test_unexpected_response_treated_as_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": "maybe"}):
        assert should_merge("a", "b", model="test-model") is False


def test_empty_response_treated_as_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={"content": ""}):
        assert should_merge("a", "b", model="test-model") is False


def test_none_content_treated_as_no() -> None:
    with patch("mindloop.merge_llm.chat", return_value={}):
        assert should_merge("a", "b", model="test-model") is False


# --- merge_texts ---


def test_merge_texts_single_call() -> None:
    resp = {
        "content": "MERGED: Merged content here.\nABSTRACT: A short summary.\nSUMMARY: A longer explanation."
    }
    with patch("mindloop.merge_llm.chat", return_value=resp) as mock_chat:
        result = merge_texts("chunk a", "chunk b", model="test-model")
    mock_chat.assert_called_once()
    assert result.text == "Merged content here."
    assert result.abstract == "A short summary."
    assert result.summary == "A longer explanation."


def test_merge_texts_prefer_a() -> None:
    resp = {"content": "MERGED: m\nABSTRACT: a\nSUMMARY: s"}
    with patch("mindloop.merge_llm.chat", return_value=resp) as mock_chat:
        merge_texts("a", "b", prefer="a", model="test-model")
    system = mock_chat.call_args[1]["system_prompt"]
    assert "Chunk A is the primary source" in system


def test_merge_texts_prefer_b() -> None:
    resp = {"content": "MERGED: m\nABSTRACT: a\nSUMMARY: s"}
    with patch("mindloop.merge_llm.chat", return_value=resp) as mock_chat:
        merge_texts("a", "b", prefer="b", model="test-model")
    system = mock_chat.call_args[1]["system_prompt"]
    assert "Chunk B is the primary source" in system


def test_merge_texts_retries_on_malformed() -> None:
    """Malformed first response triggers retry with correction prompt."""
    responses = [
        {"content": "The model did something unexpected."},
        {"content": "MERGED: fixed\nABSTRACT: abs\nSUMMARY: sum"},
    ]
    with patch("mindloop.merge_llm.chat", side_effect=responses) as mock_chat:
        result = merge_texts("a", "b", model="test-model")
    assert mock_chat.call_count == 2
    assert result.text == "fixed"
    assert result.abstract == "abs"
    # Second call should include the bad output and retry prompt.
    second_messages = mock_chat.call_args_list[1][0][0]
    assert second_messages[-1]["content"].startswith("Wrong format")


def test_merge_texts_retry_sends_bad_output_back() -> None:
    """The bad output is included in the retry conversation."""
    bad = "no format at all"
    responses = [
        {"content": bad},
        {"content": "MERGED: ok\nABSTRACT: a\nSUMMARY: s"},
    ]
    with patch("mindloop.merge_llm.chat", side_effect=responses) as mock_chat:
        merge_texts("a", "b", model="test-model")
    second_messages = mock_chat.call_args_list[1][0][0]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["content"] == bad


def test_merge_texts_exhausted_retries() -> None:
    """All retries fail — returns raw text with empty abstract/summary."""
    with patch("mindloop.merge_llm.chat", return_value={"content": "garbage"}):
        result = merge_texts("a", "b", model="test-model")
    assert result.text == "garbage"
    assert result.abstract == ""
    assert result.summary == ""
