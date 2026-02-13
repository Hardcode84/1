"""Agentic loop: chat with tool use until the model produces a final text response."""

import json
from collections.abc import Callable
from datetime import datetime

from mindloop.client import Message, chat
from mindloop.tools import Param, ToolRegistry, create_default_registry
from mindloop.util import CHARS_PER_TOKEN, noop

DEFAULT_MAX_ITERATIONS = 1000
_USER_UNAVAILABLE = "Continue autonomously. Use the ask tool if you need user input."


def _noop_message(_msg: Message) -> None:
    pass


DEFAULT_MAX_TOKENS = 200_000 * 5
_BUDGET_WARNING_THRESHOLDS = (0.5, 0.8)
_REFLECT_INTERVAL = 5


def _inject_budget_warnings(
    total_tokens: int,
    max_tokens: int,
    warned: set[float],
    messages: list[Message],
    on_message: Callable[[Message], None],
) -> None:
    """Inject system warnings when token usage crosses thresholds."""
    for threshold in _BUDGET_WARNING_THRESHOLDS:
        if threshold not in warned and total_tokens >= max_tokens * threshold:
            warned.add(threshold)
            pct = int(threshold * 100)
            warn: Message = {
                "role": "system",
                "content": f"Warning: {pct}% of token budget used"
                f" ({total_tokens} / {max_tokens}).",
            }
            messages.append(warn)
            on_message(warn)


def _estimate_tokens(messages: list[Message], response: Message) -> int:
    """Estimate total tokens for a call from character counts."""
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    response_chars = len(str(response.get("content", "")))
    response_chars += len(str(response.get("reasoning", "")))
    return (prompt_chars + response_chars) // CHARS_PER_TOKEN


def _auto_confirm(_name: str, _arguments: str) -> bool:
    return True


def _no_ask(message: str) -> str:
    return "User is unavailable."


def run_agent(
    system_prompt: str,
    registry: ToolRegistry | None = None,
    model: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_step: Callable[[str], None] = noop,
    on_thinking: Callable[[str], None] | None = None,
    on_message: Callable[[Message], None] = _noop_message,
    on_confirm: Callable[[str, str], bool] | None = None,
    on_ask: Callable[[str], str] = _no_ask,
    reasoning_effort: str = "high",
    initial_messages: list[Message] | None = None,
    instance: int = 0,
    nudge_extra: str = "",
) -> str:
    """Run the agent loop driven by system_prompt alone. Returns the final text."""
    from mindloop.client import DEFAULT_MODEL

    if on_confirm is None:
        on_confirm = _auto_confirm
    if registry is None:
        registry = create_default_registry()

    messages: list[Message] = list(initial_messages) if initial_messages else []
    effective_model = model if model is not None else DEFAULT_MODEL
    total_tokens = 0
    warned_thresholds: set[float] = set()
    consecutive_tool_turns = 0

    # Register runtime tools directly on the registry.

    def _status() -> str:
        now = datetime.now().isoformat(timespec="seconds")
        lines = [f"time: {now}", f"tokens used: {total_tokens} / {max_tokens}"]
        if instance:
            lines.append(f"instance: {instance}")
        return "\n".join(lines)

    registry.add(
        name="status",
        description="Query system info: current time, token usage and limit.",
        params=[],
        func=_status,
    )

    finished = False

    def _done(summary: str) -> str:
        nonlocal finished
        finished = True
        return summary

    registry.add(
        name="done",
        description="Call when you are finished. Terminates the session.",
        params=[
            Param(name="summary", description="Brief summary of what was accomplished.")
        ],
        func=_done,
    )

    registry.add(
        name="ask",
        description="Ask the user a question. User may not always be immediately available for an answer.",
        params=[Param(name="message", description="Message to show the user.")],
        func=on_ask,
    )

    def _stop(reason: str) -> str:
        """Log termination reason and return the last model content."""
        stop_msg: Message = {
            "role": "system",
            "content": f"[stop] {reason} ({total_tokens} tokens used)",
        }
        on_message(stop_msg)
        on_step(f"\n[stop] {reason} ({total_tokens} tokens used)")
        if registry.stats:
            stats_text = json.dumps(registry.stats)
            stats_msg: Message = {
                "role": "system",
                "content": f"[stats] {stats_text}",
            }
            on_message(stats_msg)
            on_step(f"\n[stats] {stats_text}")
        last = messages[-1].get("content", "") if messages else ""
        return str(last)

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
            return _stop(f"token budget exceeded (limit {max_tokens})")

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            consecutive_tool_turns = 0
            # Nudge the model to keep going.
            nudge: Message = {"role": "user", "content": _USER_UNAVAILABLE}
            messages.append(nudge)
            on_message(nudge)
            _inject_budget_warnings(
                total_tokens, max_tokens, warned_thresholds, messages, on_message
            )
            continue

        for call in tool_calls:
            name = call["function"]["name"].strip()
            arguments = call["function"]["arguments"]
            on_step(f"[tool] {name}({arguments})")
            # Treat empty arguments as {} (models often omit args for no-param tools).
            if not arguments or not arguments.strip():
                arguments = "{}"
                call["function"]["arguments"] = arguments

            # Reject truly malformed JSON so the model learns its mistake.
            try:
                json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                call["function"]["arguments"] = "{}"
                tool_result = f"Error: malformed arguments: {arguments}"
            else:
                if not on_confirm(name, arguments):
                    tool_result = f"Error: {name} was denied by the user."
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

        if finished:
            return _stop("model finished")

        consecutive_tool_turns += 1

        # Warn the model after all tool results are appended.
        _inject_budget_warnings(
            total_tokens, max_tokens, warned_thresholds, messages, on_message
        )

        # Nudge reflection after consecutive tool-only turns.
        if consecutive_tool_turns % _REFLECT_INTERVAL == 0:
            reflect_text = (
                "You've been using tools for a while. "
                "Pause and reflect on what you've learned so far."
            )
            if nudge_extra:
                reflect_text += "\n\n" + nudge_extra
            reflect: Message = {
                "role": "system",
                "content": reflect_text,
            }
            messages.append(reflect)
            on_message(reflect)

    return _stop(f"max iterations reached ({max_iterations})")
