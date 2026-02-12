"""Tests for mindloop.agent."""

from typing import Any
from unittest.mock import MagicMock, patch

from mindloop.agent import _USER_UNAVAILABLE, run_agent
from mindloop.tools import Param, ToolRegistry


def _make_tool_response(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a mock assistant message with tool_calls."""
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _make_tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _make_final_response(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _make_done_response(call_id: str, summary: str) -> dict[str, Any]:
    """Build a response that calls the done tool."""
    return _make_tool_response(
        [_make_tool_call(call_id, "done", f'{{"summary": "{summary}"}}')]
    )


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


@patch("mindloop.agent.chat")
def test_tool_call_then_done(mock_chat: MagicMock) -> None:
    """Model calls a tool, gets result, then calls done."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        _make_done_response("c2", "finished"),
    ]
    run_agent("prompt", registry=_echo_registry())
    assert mock_chat.call_count == 2


@patch("mindloop.agent.chat")
def test_nudge_injects_user_unavailable(mock_chat: MagicMock) -> None:
    """Text-only response triggers a user-unavailable nudge."""
    mock_chat.side_effect = [
        _make_final_response("waiting for input"),
        _make_done_response("c1", "done"),
    ]
    run_agent("prompt", registry=_echo_registry())

    # After the first text response, a user message should be injected.
    messages = mock_chat.call_args_list[1][0][0]
    user_msgs = [m for m in messages if m.get("content") == _USER_UNAVAILABLE]
    assert len(user_msgs) == 1


@patch("mindloop.agent.chat")
def test_done_tool_terminates(mock_chat: MagicMock) -> None:
    """Calling the done tool terminates the loop."""
    mock_chat.side_effect = [
        _make_done_response("c1", "all done"),
    ]
    on_msg = MagicMock()
    run_agent("prompt", registry=_echo_registry(), on_message=on_msg)
    assert mock_chat.call_count == 1

    # Verify stop message was emitted.
    stop_msgs = [
        c.args[0]
        for c in on_msg.call_args_list
        if c.args[0].get("role") == "system"
        and "[stop]" in c.args[0].get("content", "")
    ]
    assert len(stop_msgs) == 1
    assert "model finished" in stop_msgs[0]["content"]


@patch("mindloop.agent.chat")
def test_max_iterations_safety_valve(mock_chat: MagicMock) -> None:
    """Loop stops after max_iterations even if model keeps calling tools."""
    mock_chat.return_value = _make_tool_response(
        [_make_tool_call("c1", "echo", '{"text": "loop"}')]
    )
    run_agent("prompt", registry=_echo_registry(), max_iterations=3)
    assert mock_chat.call_count == 3


@patch("mindloop.agent.chat")
def test_multiple_tool_calls_in_single_response(mock_chat: MagicMock) -> None:
    """Model issues multiple tool calls in one response."""
    mock_chat.side_effect = [
        _make_tool_response(
            [
                _make_tool_call("c1", "echo", '{"text": "a"}'),
                _make_tool_call("c2", "echo", '{"text": "b"}'),
            ]
        ),
        _make_done_response("c3", "both done"),
    ]
    run_agent("prompt", registry=_echo_registry())

    # Verify both echo tool results were appended.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    echo_msgs = [
        m
        for m in second_call_messages
        if m["role"] == "tool" and m["content"].startswith("echoed:")
    ]
    assert len(echo_msgs) == 2
    assert echo_msgs[0]["content"] == "echoed: a"
    assert echo_msgs[1]["content"] == "echoed: b"


@patch("mindloop.agent.chat")
def test_model_kwarg_passed_through(mock_chat: MagicMock) -> None:
    """Custom model parameter is forwarded to chat()."""
    mock_chat.side_effect = [
        _make_done_response("c1", "ok"),
    ]
    run_agent("prompt", registry=_echo_registry(), model="test-model")
    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "test-model"


@patch("mindloop.agent.chat")
def test_nudge_then_tool_call_continues(mock_chat: MagicMock) -> None:
    """After nudge, model can resume tool use instead of finishing."""
    mock_chat.side_effect = [
        _make_final_response("hmm let me think"),
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "go"}')]),
        _make_done_response("c2", "found it"),
    ]
    run_agent("prompt", registry=_echo_registry())
    assert mock_chat.call_count == 3


@patch("mindloop.agent.chat")
def test_malformed_tool_call_arguments(mock_chat: MagicMock) -> None:
    """Malformed JSON arguments are reported to the model and sanitized."""
    bad_args = '{"path": "./"README.md"}'
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "read", bad_args)]),
        _make_done_response("c2", "sorry"),
    ]
    run_agent("prompt", registry=_echo_registry())

    # Verify the tool result reports the malformed arguments.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "malformed arguments" in tool_msg["content"]
    assert bad_args in tool_msg["content"]

    # Verify the arguments field was sanitized for the API.
    assistant_msg = second_call_messages[0]
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == "{}"


@patch("mindloop.agent.chat")
def test_empty_arguments_treated_as_empty_object(mock_chat: MagicMock) -> None:
    """Empty string arguments are normalised to {} and the tool executes."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", "")]),
        _make_done_response("c2", "ok"),
    ]
    run_agent("prompt", registry=_echo_registry())

    second_call_messages = mock_chat.call_args_list[1][0][0]
    assistant_msg = second_call_messages[0]
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == "{}"

    # Tool was called (not rejected as malformed).
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "malformed" not in tool_msg["content"]


@patch("mindloop.agent.chat")
def test_ask_tool_returns_user_response(mock_chat: MagicMock) -> None:
    """Ask tool passes message to callback and returns user's response."""
    mock_chat.side_effect = [
        _make_tool_response(
            [_make_tool_call("c1", "ask", '{"message": "what next?"}')]
        ),
        _make_done_response("c2", "got answer"),
    ]
    on_ask = MagicMock(return_value="do nothing")
    run_agent("prompt", registry=_echo_registry(), on_ask=on_ask)

    on_ask.assert_called_once_with(message="what next?")
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert tool_msg["content"] == "do nothing"


@patch("mindloop.agent.chat")
def test_ask_tool_default_unavailable(mock_chat: MagicMock) -> None:
    """Without on_ask callback, ask tool returns unavailable message."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "ask", '{"message": "hello?"}')]),
        _make_done_response("c2", "ok"),
    ]
    run_agent("prompt", registry=_echo_registry())

    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "unavailable" in tool_msg["content"].lower()


@patch("mindloop.agent.chat")
def test_on_confirm_denied(mock_chat: MagicMock) -> None:
    """Denied tool calls return an error to the model without executing."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        _make_done_response("c2", "ok denied"),
    ]
    deny_echo = MagicMock(side_effect=lambda name, _args: name != "echo")
    run_agent("prompt", registry=_echo_registry(), on_confirm=deny_echo)

    # Model should see the denial message, not the tool result.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "denied" in tool_msg["content"]


@patch("mindloop.agent.chat")
def test_on_confirm_approved(mock_chat: MagicMock) -> None:
    """Approved tool calls execute normally."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        _make_done_response("c2", "got it"),
    ]
    approve_all = MagicMock(return_value=True)
    run_agent("prompt", registry=_echo_registry(), on_confirm=approve_all)

    # Tool should have actually executed.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "echoed: hi" in tool_msg["content"]


@patch("mindloop.agent._REFLECT_INTERVAL", 3)
@patch("mindloop.agent.chat")
def test_reflect_nudge_after_consecutive_tools(mock_chat: MagicMock) -> None:
    """System reflect message injected after N consecutive tool-only turns."""
    tool_resp = _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')])
    mock_chat.side_effect = [
        tool_resp,
        tool_resp,
        tool_resp,  # 3rd consecutive → reflect nudge.
        _make_done_response("c2", "thinking now"),
    ]
    on_msg = MagicMock()
    run_agent("prompt", registry=_echo_registry(), on_message=on_msg)

    # Find system reflect messages among all on_message calls.
    system_msgs = [
        c.args[0]
        for c in on_msg.call_args_list
        if c.args[0].get("role") == "system"
        and "reflect" in c.args[0].get("content", "").lower()
    ]
    assert len(system_msgs) == 1


@patch("mindloop.agent._REFLECT_INTERVAL", 3)
@patch("mindloop.agent.chat")
def test_reflect_counter_resets_on_text(mock_chat: MagicMock) -> None:
    """Consecutive tool counter resets when model produces text."""
    tool_resp = _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')])
    mock_chat.side_effect = [
        tool_resp,
        tool_resp,
        # Text response resets the counter.
        _make_final_response("let me think"),
        tool_resp,
        tool_resp,
        # Only 2 after reset — no nudge. Then done.
        _make_done_response("c2", "ok"),
    ]
    on_msg = MagicMock()
    run_agent("prompt", registry=_echo_registry(), on_message=on_msg)

    system_msgs = [
        c.args[0]
        for c in on_msg.call_args_list
        if c.args[0].get("role") == "system"
        and "reflect" in c.args[0].get("content", "").lower()
    ]
    assert len(system_msgs) == 0


@patch("mindloop.agent.chat")
def test_token_budget_stops_loop(mock_chat: MagicMock) -> None:
    """Loop stops when cumulative token usage exceeds max_tokens."""
    response_with_usage = {
        **_make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    mock_chat.return_value = response_with_usage
    run_agent("prompt", registry=_echo_registry(), max_tokens=200)
    # First call: 150 tokens (under 200), continues. Second call: 300 total (over 200), stops.
    assert mock_chat.call_count == 2


@patch("mindloop.agent.chat")
def test_token_budget_estimated_when_no_usage(mock_chat: MagicMock) -> None:
    """Token budget works via estimation when API returns no usage."""
    # ~400 chars of content -> ~100 estimated tokens per response.
    long_text = "x" * 400
    mock_chat.return_value = _make_tool_response(
        [_make_tool_call("c1", "echo", '{"text": "hi"}')]
    )
    mock_chat.return_value["content"] = long_text
    # No "usage" key in response — estimation should kick in.
    run_agent("prompt", registry=_echo_registry(), max_tokens=150)
    # First call: ~100 estimated (under 150). Second call: prompt grows, estimate
    # exceeds 150. Loop stops.
    assert mock_chat.call_count == 2


@patch("mindloop.agent.chat")
def test_initial_messages_seeded(mock_chat: MagicMock) -> None:
    """Initial messages are included in the first chat call."""
    mock_chat.side_effect = [
        _make_done_response("c1", "resumed"),
    ]
    history = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "continue"},
    ]
    run_agent("prompt", registry=_echo_registry(), initial_messages=history)
    first_call_messages = mock_chat.call_args_list[0][0][0]
    assert first_call_messages[0] == history[0]
    assert first_call_messages[1] == history[1]
