"""Agentic loop: chat with tool use until the model produces a final text response."""

import json
from collections.abc import Callable

from mindloop.client import Message, chat
from mindloop.tools import ToolRegistry, default_registry

DEFAULT_MAX_ITERATIONS = 20
_USER_UNAVAILABLE = (
    "User is unavailable. Continue autonomously using the tools provided."
)


def _noop(_msg: str) -> None:
    pass


def _noop_message(_msg: Message) -> None:
    pass


# Rough chars-per-token ratio for estimation.
_CHARS_PER_TOKEN = 4

DEFAULT_MAX_TOKENS = 200_000


def _estimate_tokens(messages: list[Message], response: Message) -> int:
    """Estimate total tokens for a call from character counts."""
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    response_chars = len(str(response.get("content", "")))
    response_chars += len(str(response.get("reasoning", "")))
    return (prompt_chars + response_chars) // _CHARS_PER_TOKEN


def run_agent(
    system_prompt: str,
    registry: ToolRegistry = default_registry,
    model: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_step: Callable[[str], None] = _noop,
    on_thinking: Callable[[str], None] | None = None,
    on_message: Callable[[Message], None] = _noop_message,
    reasoning_effort: str = "high",
) -> str:
    """Run the agent loop driven by system_prompt alone. Returns the final text."""
    from mindloop.client import DEFAULT_MODEL

    messages: list[Message] = []
    effective_model = model if model is not None else DEFAULT_MODEL
    total_tokens = 0

    for _ in range(max_iterations):
        response = chat(
            messages,
            model=effective_model,
            system_prompt=system_prompt,
            tools=registry.definitions(),
            stream=True,
            on_token=on_step,
            on_thinking=on_thinking,
            reasoning_effort=reasoning_effort,
        )
        usage = response.get("usage")
        if usage:
            total_tokens += int(usage.get("total_tokens", 0))
        else:
            total_tokens += _estimate_tokens(messages, response)
        messages.append(response)
        on_message(response)

        if total_tokens >= max_tokens:
            on_step(f"\n[budget] {total_tokens} tokens used, limit {max_tokens}")
            break

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            # If the previous message was already our nudge, the model is done.
            if len(messages) >= 2 and messages[-2].get("content") == _USER_UNAVAILABLE:
                return str(response.get("content", ""))
            # Otherwise, nudge the model to keep going.
            nudge: Message = {"role": "user", "content": _USER_UNAVAILABLE}
            messages.append(nudge)
            on_message(nudge)
            continue

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            on_step(f"[tool] {name}({arguments})")
            # Sanitize malformed JSON so it doesn't poison future API calls.
            try:
                json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                call["function"]["arguments"] = "{}"
                tool_result = f"Error: malformed arguments: {arguments}"
            else:
                tool_result = registry.execute(name, arguments)
            on_step(f"[result] {tool_result}")
            tool_msg: Message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": tool_result,
            }
            messages.append(tool_msg)
            on_message(tool_msg)

    # Safety valve: return whatever the model last said.
    return str(messages[-1].get("content", "")) if messages else ""
