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
