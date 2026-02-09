"""Tests for mindloop.agent."""

from typing import Any
from unittest.mock import MagicMock, patch

from mindloop.agent import run_agent
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
    """Model calls a tool, gets result, then produces final text."""
    mock_chat.side_effect = [
        _make_tool_response([_make_tool_call("c1", "echo", '{"text": "hi"}')]),
        _make_final_response("done"),
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "done"
    assert mock_chat.call_count == 2

    # Verify tool result was fed back. Messages list is shared, so after the
    # second chat() call the final assistant response is appended too.
    final_messages = mock_chat.call_args_list[1][0][0]
    tool_msg = final_messages[-2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "echoed: hi"


@patch("mindloop.agent.chat")
def test_immediate_final_response(mock_chat: MagicMock) -> None:
    """Model produces final text without calling any tools."""
    mock_chat.return_value = _make_final_response("hello")
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "hello"
    assert mock_chat.call_count == 1


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
    ]
    result = run_agent("prompt", registry=_echo_registry())
    assert result == "both done"

    # Verify both tool results were appended.
    final_messages = mock_chat.call_args_list[1][0][0]
    tool_msgs = [m for m in final_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "echoed: a"
    assert tool_msgs[1]["content"] == "echoed: b"


@patch("mindloop.agent.chat")
def test_model_kwarg_passed_through(mock_chat: MagicMock) -> None:
    """Custom model parameter is forwarded to chat()."""
    mock_chat.return_value = _make_final_response("ok")
    run_agent("prompt", registry=_echo_registry(), model="test-model")
    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "test-model"
