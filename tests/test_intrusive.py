"""Tests for intrusive recall and cross-context critic in MidSessionExtractor."""

from typing import Any
from unittest.mock import MagicMock, patch

from mindloop.agent import run_agent
from mindloop.cli.agent import (
    MidSessionExtractor,
    _INTRUSIVE_MIN_SCORE,
)
from mindloop.memory import SearchResult
from mindloop.summarizer import ChunkSummary
from mindloop.tools import Param, ToolRegistry


def _make_search_result(chunk_id: int, abstract: str, cosine: float) -> SearchResult:
    """Build a SearchResult with a stub ChunkSummary."""
    chunk = MagicMock()
    chunk.text = f"text {chunk_id}"
    summary = ChunkSummary(chunk=chunk, abstract=abstract, summary="summary")
    return SearchResult(
        id=chunk_id, chunk_summary=summary, score=1.0, cosine_score=cosine
    )


def test_intrusive_recall_empty_before_first_window() -> None:
    """No prior window means empty string."""
    store = MagicMock()
    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")
    assert ext.intrusive_recall() == ""
    store.search.assert_not_called()


@patch("mindloop.cli.agent.collapse_messages")
def test_intrusive_recall_surfaces_matches(mock_collapse: MagicMock) -> None:
    """After on_extract, intrusive_recall returns formatted matches."""
    store = MagicMock()
    store.search.return_value = [
        _make_search_result(42, "Cats like fish.", 0.7),
        _make_search_result(57, "Dogs like bones.", 0.6),
    ]

    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")

    # Simulate on_extract: mock collapse_messages to return turns.
    turn = MagicMock()
    turn.role = "Bot"
    turn.text = "discussing pets"
    mock_collapse.return_value = [turn]

    ext.on_extract([{"role": "assistant", "content": "hi"}])

    result = ext.intrusive_recall()
    assert "Cats like fish." in result
    assert "Dogs like bones." in result
    assert "#42" in result
    assert "#57" in result
    assert result.startswith("These memories seem related")


@patch("mindloop.cli.agent.collapse_messages")
def test_intrusive_recall_cooldown(mock_collapse: MagicMock) -> None:
    """Second call returns empty because IDs are in the seen set."""
    store = MagicMock()
    store.search.return_value = [
        _make_search_result(42, "Cats like fish.", 0.7),
    ]

    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")

    turn = MagicMock()
    turn.role = "Bot"
    turn.text = "discussing pets"
    mock_collapse.return_value = [turn]
    ext.on_extract([{"role": "assistant", "content": "hi"}])

    first = ext.intrusive_recall()
    assert "#42" in first

    # Feed another window so the buffer is non-empty, but same search results.
    ext.on_extract([{"role": "assistant", "content": "more"}])
    second = ext.intrusive_recall()
    assert second == ""


@patch("mindloop.cli.agent.collapse_messages")
def test_intrusive_recall_filters_low_score(mock_collapse: MagicMock) -> None:
    """Results below the minimum cosine score are excluded."""
    store = MagicMock()
    store.search.return_value = [
        _make_search_result(10, "Low score.", _INTRUSIVE_MIN_SCORE - 0.01),
        _make_search_result(11, "High score.", _INTRUSIVE_MIN_SCORE + 0.1),
    ]

    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")

    turn = MagicMock()
    turn.role = "Bot"
    turn.text = "working"
    mock_collapse.return_value = [turn]
    ext.on_extract([{"role": "assistant", "content": "hi"}])

    result = ext.intrusive_recall()
    assert "#10" not in result
    assert "#11" in result


@patch("mindloop.cli.agent.collapse_messages")
def test_intrusive_recall_accumulates_windows(mock_collapse: MagicMock) -> None:
    """Multiple on_extract calls accumulate text until intrusive_recall consumes it."""
    store = MagicMock()
    store.search.return_value = [
        _make_search_result(10, "Accumulated fact.", 0.8),
    ]

    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")

    turn1 = MagicMock()
    turn1.role = "Bot"
    turn1.text = "first window"
    mock_collapse.return_value = [turn1]
    ext.on_extract([{"role": "assistant", "content": "a"}])

    turn2 = MagicMock()
    turn2.role = "Bot"
    turn2.text = "second window"
    mock_collapse.return_value = [turn2]
    ext.on_extract([{"role": "assistant", "content": "b"}])

    result = ext.intrusive_recall()
    assert "#10" in result

    # Verify search was called with both windows joined.
    query = store.search.call_args[0][0]
    assert "first window" in query
    assert "second window" in query

    # Buffer is now empty.
    assert ext.intrusive_recall() == ""


@patch("mindloop.cli.agent.collapse_messages")
@patch("mindloop.semantic_memory.save_memory")
@patch("mindloop.extractor.verify_facts", side_effect=lambda facts, *a, **kw: facts)
@patch("mindloop.extractor.extract_window")
def test_drain_excludes_saved_ids_from_recall(
    mock_extract: MagicMock,
    mock_verify: MagicMock,
    mock_save: MagicMock,
    mock_collapse: MagicMock,
) -> None:
    """Chunks saved by _drain are pre-seeded into _seen_ids."""
    store = MagicMock()
    store.search.side_effect = [
        [],  # _is_duplicate check.
        [_make_search_result(99, "Just saved.", 0.9)],  # intrusive_recall query.
    ]
    mock_extract.return_value = [
        {"text": "fact", "abstract": "abs", "summary": "sum"},
    ]
    mock_save.return_value = 99  # save_memory returns chunk ID 99.

    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")

    # First on_extract: no drain (no future), accumulates text.
    turn = MagicMock()
    turn.role = "Bot"
    turn.text = "first"
    mock_collapse.return_value = [turn]
    ext.on_extract([{"role": "assistant", "content": "a"}])

    # Second on_extract: drains first future, saves chunk 99.
    ext.on_extract([{"role": "assistant", "content": "b"}])

    # ID 99 should now be in _seen_ids from drain.
    assert 99 in ext._seen_ids

    # Intrusive recall should filter it out.
    result = ext.intrusive_recall()
    assert result == ""


# --- Agent-level integration test ---


def _echo_registry() -> ToolRegistry:
    """Registry with a single echo tool for testing."""
    reg = ToolRegistry()
    reg.add(
        name="echo",
        description="Echo input.",
        params=[Param(name="text", description="Text.")],
        func=lambda text: f"echoed: {text}",
    )
    return reg


def _make_tool_response(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _make_tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_done_response(call_id: str, summary: str) -> dict[str, Any]:
    return _make_tool_response(
        [_make_tool_call(call_id, "done", f'{{"summary": "{summary}"}}')]
    )


@patch("mindloop.agent._REFLECT_INTERVAL", 3)
@patch("mindloop.agent.chat_best_of_n")
def test_on_reflect_injects_into_nudge(mock_chat: MagicMock) -> None:
    """on_reflect output appears in the reflection system message."""
    tool_resp = _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')])
    mock_chat.side_effect = [
        tool_resp,
        tool_resp,
        tool_resp,  # 3rd → reflection fires.
        _make_done_response("c2", "done"),
    ]
    on_msg = MagicMock()

    def fake_reflect() -> str:
        return (
            'These memories seem related to what you\'re doing:\n- "Remember X." (#1)'
        )

    run_agent(
        "prompt",
        registry=_echo_registry(),
        model="test-model",
        on_message=on_msg,
        on_reflect=fake_reflect,
    )

    # Find the reflection system message.
    reflect_msgs = [
        c.args[0]
        for c in on_msg.call_args_list
        if c.args[0].get("role") == "system"
        and "reflect" in c.args[0].get("content", "").lower()
    ]
    assert len(reflect_msgs) == 1
    assert "Remember X." in reflect_msgs[0]["content"]


@patch("mindloop.agent._REFLECT_INTERVAL", 3)
@patch("mindloop.agent.chat_best_of_n")
def test_on_reflect_empty_does_not_alter_nudge(mock_chat: MagicMock) -> None:
    """Empty on_reflect return does not change the reflection message."""
    tool_resp = _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')])
    mock_chat.side_effect = [
        tool_resp,
        tool_resp,
        tool_resp,
        _make_done_response("c2", "done"),
    ]
    on_msg = MagicMock()

    run_agent(
        "prompt",
        registry=_echo_registry(),
        model="test-model",
        on_message=on_msg,
        on_reflect=lambda: "",
    )

    reflect_msgs = [
        c.args[0]
        for c in on_msg.call_args_list
        if c.args[0].get("role") == "system"
        and "reflect" in c.args[0].get("content", "").lower()
    ]
    assert len(reflect_msgs) == 1
    # Should not contain "memories" since on_reflect returned empty.
    assert "memories" not in reflect_msgs[0]["content"]


# --- Cross-context critic tests ---


def test_critic_review_empty_before_first_window() -> None:
    """No prior window means empty string, no chat call."""
    store = MagicMock()
    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")
    with patch("mindloop.critic.chat") as mock_chat:
        assert ext.critic_review() == ""
        mock_chat.assert_not_called()


@patch("mindloop.critic.chat")
def test_critic_review_returns_concern(mock_chat: MagicMock) -> None:
    """Critic concern is returned with [critic] prefix."""
    mock_chat.return_value = {
        "content": "The agent is reading the same file repeatedly.\nno_pin"
    }
    ext = MidSessionExtractor(
        MagicMock(), model="m", log=lambda _: None, agent_model="m"
    )
    ext._last_query = "Bot: Read config.py\nBot: Read config.py"
    result = ext.critic_review()
    assert result == "[critic] The agent is reading the same file repeatedly."


@patch("mindloop.critic.chat")
def test_critic_review_ok_returns_empty(mock_chat: MagicMock) -> None:
    """Critic returning 'ok' means no injection."""
    mock_chat.return_value = {"content": "ok\nno_pin"}
    ext = MidSessionExtractor(
        MagicMock(), model="m", log=lambda _: None, agent_model="m"
    )
    ext._last_query = "Bot: Making progress on the task."
    assert ext.critic_review() == ""


@patch("mindloop.critic.chat")
@patch("mindloop.cli.agent.collapse_messages")
def test_critic_review_uses_last_query(
    mock_collapse: MagicMock, mock_chat: MagicMock
) -> None:
    """Critic receives the query text saved by intrusive_recall."""
    store = MagicMock()
    store.search.return_value = []
    mock_chat.return_value = {"content": "ok\nno_pin"}

    turn = MagicMock()
    turn.role = "Bot"
    turn.text = "working on feature X"
    mock_collapse.return_value = [turn]

    ext = MidSessionExtractor(store, model="m", log=lambda _: None, agent_model="m")
    ext.on_extract([{"role": "assistant", "content": "hi"}])
    ext.intrusive_recall()
    ext.critic_review()

    call_args = mock_chat.call_args
    prompt_text = call_args[0][0][0]["content"]
    assert "working on feature X" in prompt_text


@patch("mindloop.critic.chat")
def test_critic_review_accumulates_pinned_window(mock_chat: MagicMock) -> None:
    """Pin reason causes window to be accumulated."""
    mock_chat.return_value = {"content": "ok\nKey decision about retry logic."}
    ext = MidSessionExtractor(
        MagicMock(), model="m", log=lambda _: None, agent_model="m"
    )
    ext._last_query = "Bot: decided to retry with backoff"
    ext.critic_review()
    assert len(ext.pinned_windows) == 1
    assert ext.pinned_windows[0]["reason"] == "Key decision about retry logic."
    assert ext.pinned_windows[0]["text"] == "Bot: decided to retry with backoff"


@patch("mindloop.critic.chat")
def test_critic_review_no_pin_does_not_accumulate(mock_chat: MagicMock) -> None:
    """no_pin response does not accumulate a pinned window."""
    mock_chat.return_value = {"content": "ok\nno_pin"}
    ext = MidSessionExtractor(
        MagicMock(), model="m", log=lambda _: None, agent_model="m"
    )
    ext._last_query = "Bot: routine work"
    ext.critic_review()
    assert ext.pinned_windows == []


@patch("mindloop.critic.chat")
def test_critic_review_empty_query_skips_pin(mock_chat: MagicMock) -> None:
    """Empty _last_query skips pin even with a pin reason."""
    mock_chat.return_value = {"content": "ok\nImportant discovery."}
    ext = MidSessionExtractor(
        MagicMock(), model="m", log=lambda _: None, agent_model="m"
    )
    # _last_query is empty — critic_review returns early without chat call.
    ext._last_query = ""
    ext.critic_review()
    assert ext.pinned_windows == []


@patch("mindloop.agent._REFLECT_INTERVAL", 3)
@patch("mindloop.agent.chat_best_of_n")
def test_on_reflect_combines_intrusive_and_critic(mock_chat: MagicMock) -> None:
    """Both intrusive recall and critic output appear in the reflection message."""
    tool_resp = _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')])
    mock_chat.side_effect = [
        tool_resp,
        tool_resp,
        tool_resp,  # 3rd → reflection fires.
        _make_done_response("c2", "done"),
    ]
    on_msg = MagicMock()

    def fake_reflect() -> str:
        return (
            'These memories seem related to what you\'re doing:\n- "Remember X." (#1)'
            "\n\n[critic] The agent appears stuck in a loop."
        )

    run_agent(
        "prompt",
        registry=_echo_registry(),
        model="test-model",
        on_message=on_msg,
        on_reflect=fake_reflect,
    )

    reflect_msgs = [
        c.args[0]
        for c in on_msg.call_args_list
        if c.args[0].get("role") == "system"
        and "reflect" in c.args[0].get("content", "").lower()
    ]
    assert len(reflect_msgs) == 1
    content = reflect_msgs[0]["content"]
    assert "Remember X." in content
    assert "[critic] The agent appears stuck in a loop." in content
