"""Agentic loop: chat with tool use until the model produces a final text response."""

from collections.abc import Callable

from mindloop.client import Message, chat
from mindloop.tools import ToolRegistry, default_registry

DEFAULT_MAX_ITERATIONS = 20


def _noop(_msg: str) -> None:
    pass


def run_agent(
    system_prompt: str,
    registry: ToolRegistry = default_registry,
    model: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_step: Callable[[str], None] = _noop,
    on_thinking: Callable[[str], None] | None = None,
    reasoning_effort: str = "high",
) -> str:
    """Run the agent loop driven by system_prompt alone. Returns the final text."""
    from mindloop.client import DEFAULT_MODEL

    messages: list[Message] = []
    effective_model = model if model is not None else DEFAULT_MODEL

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
        messages.append(response)

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return str(response.get("content", ""))

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            on_step(f"[tool] {name}({arguments})")
            tool_result = registry.execute(name, arguments)
            on_step(f"[result] {tool_result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result,
                }
            )

    # Safety valve: return whatever the model last said.
    return str(messages[-1].get("content", "")) if messages else ""
