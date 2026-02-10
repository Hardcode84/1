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
def test_tool_call_then_final_response(mock_chat: MagicMock) -> None:
    """Model calls a tool, gets result, then gives final text after nudge."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        # First text response triggers a nudge.
        _make_final_response("done"),
        # After nudge, model responds again — now it's final.
        _make_final_response("really done"),
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "really done"
    assert mock_chat.call_count == 3


@patch("mindloop.agent.chat")
def test_nudge_injects_user_unavailable(mock_chat: MagicMock) -> None:
    """First text-only response triggers a user-unavailable nudge."""
    mock_chat.side_effect = [
        _make_final_response("waiting for input"),
        _make_final_response("ok continuing"),
    ]
    run_agent("prompt", registry=_echo_registry())

    # After the first text response, a user message should be injected.
    messages = mock_chat.call_args_list[1][0][0]
    user_msgs = [m for m in messages if m.get("content") == _USER_UNAVAILABLE]
    assert len(user_msgs) == 1


@patch("mindloop.agent.chat")
def test_double_text_response_terminates(mock_chat: MagicMock) -> None:
    """After nudge, second text-only response is treated as final."""
    mock_chat.side_effect = [
        _make_final_response("question?"),
        _make_final_response("ok done"),
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "ok done"
    assert mock_chat.call_count == 2


@patch("mindloop.agent.chat")
def test_max_iterations_safety_valve(mock_chat: MagicMock) -> None:
    """Loop stops after max_iterations even if model keeps calling tools."""
    mock_chat.return_value = _make_tool_response(
        [_make_tool_call("c1", "echo", '{"text": "loop"}')]
    )
    run_agent("prompt", registry=_echo_registry(), max_iterations=3)
    # Should have called chat exactly 3 times then stopped.
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
        _make_final_response("both done"),
        _make_final_response("final"),
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "final"

    # Verify both tool results were appended.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "echoed: a"
    assert tool_msgs[1]["content"] == "echoed: b"


@patch("mindloop.agent.chat")
def test_model_kwarg_passed_through(mock_chat: MagicMock) -> None:
    """Custom model parameter is forwarded to chat()."""
    mock_chat.side_effect = [
        _make_final_response("ask"),
        _make_final_response("ok"),
    ]
    run_agent("prompt", registry=_echo_registry(), model="test-model")
    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "test-model"


@patch("mindloop.agent.chat")
def test_nudge_then_tool_call_continues(mock_chat: MagicMock) -> None:
    """After nudge, model can resume tool use instead of finishing."""
    mock_chat.side_effect = [
        _make_final_response("hmm let me think"),
        # After nudge, model decides to use a tool.
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "go"}')]),
        _make_final_response("found it"),
        _make_final_response("done"),
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "done"
    assert mock_chat.call_count == 4


@patch("mindloop.agent.chat")
def test_malformed_tool_call_arguments(mock_chat: MagicMock) -> None:
    """Malformed JSON arguments are reported to the model and sanitized."""
    bad_args = '{"path": "./"README.md"}'
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "read", bad_args)]),
        _make_final_response("sorry"),
        _make_final_response("done"),
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "done"

    # Verify the tool result reports the malformed arguments.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "malformed arguments" in tool_msg["content"]
    assert bad_args in tool_msg["content"]

    # Verify the arguments field was sanitized for the API.
    assistant_msg = second_call_messages[0]
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == "{}"


@patch("mindloop.agent.chat")
def test_on_confirm_denied(mock_chat: MagicMock) -> None:
    """Denied tool calls return an error to the model without executing."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        _make_final_response("ok denied"),
        _make_final_response("done"),
    ]
    deny_all = MagicMock(return_value=False)
    result = run_agent("prompt", registry=_echo_registry(), on_confirm=deny_all)
    assert result == "done"

    # Confirm callback was called with the tool name and arguments.
    deny_all.assert_called_once_with("echo", '{"text": "hi"}')

    # Model should see the denial message, not the tool result.
    second_call_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "denied" in tool_msg["content"]


@patch("mindloop.agent.chat")
def test_on_confirm_approved(mock_chat: MagicMock) -> None:
    """Approved tool calls execute normally."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        _make_final_response("got it"),
        _make_final_response("done"),
    ]
    approve_all = MagicMock(return_value=True)
    result = run_agent("prompt", registry=_echo_registry(), on_confirm=approve_all)
    assert result == "done"

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
        _make_final_response("thinking now"),
        _make_final_response("done"),
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
        # Only 2 after reset — no nudge. Then final.
        _make_final_response("ok"),
        _make_final_response("done"),
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
