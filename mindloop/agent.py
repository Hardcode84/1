"""Agentic loop: chat with tool use until the model produces a final text response."""

import json
from collections.abc import Callable
from datetime import datetime

from mindloop.client import Message, Stats, _ANTHROPIC_PREFIXES
from mindloop.voting import chat_best_of_n
from mindloop.quotes import NudgePool
from mindloop.tools import Param, ToolRegistry, create_default_registry
from mindloop.util import noop

DEFAULT_MAX_ITERATIONS = 1000
_USER_UNAVAILABLE = "Continue autonomously. Use the ask tool if you need user input."


def _noop_message(_msg: Message) -> None:
    pass


DEFAULT_MAX_TOKENS = 100_000
_MAX_OUTPUT_TOKENS = 16_384
_BUDGET_WARNING_THRESHOLDS = (0.5, 0.8)
_REFLECT_INTERVAL = 5


def _inject_budget_warnings(
    total_tokens: int,
    max_tokens: int,
    warned: set[float],
    messages: list[Message],
    on_message: Callable[[Message], None],
    on_step: Callable[[str], None],
) -> None:
    """Inject system warnings when token usage crosses thresholds."""
    for threshold in _BUDGET_WARNING_THRESHOLDS:
        if threshold not in warned and total_tokens >= max_tokens * threshold:
            warned.add(threshold)
            pct = int(threshold * 100)
            text = (
                f"Warning: {pct}% of token budget used"
                f" ({total_tokens} / {max_tokens})."
            )
            warn: Message = {"role": "system", "content": text}
            messages.append(warn)
            on_message(warn)
            on_step(f"\n[budget] {text}")


def _auto_confirm(_name: str, _arguments: str) -> bool:
    return True


def _no_ask(message: str) -> str:
    return "User is unavailable."


class Extractor:
    """Tracks checkpoint and fires mid-session extraction callback."""

    def __init__(
        self,
        on_extract: Callable[[list[Message]], None] | None,
        start: int = 0,
    ) -> None:
        self._on_extract = on_extract
        self._checkpoint = start

    def advance(self, messages: list[Message]) -> None:
        """Send new messages to the extraction callback and advance checkpoint."""
        if self._on_extract is not None:
            self._on_extract(messages[self._checkpoint :])
            self._checkpoint = len(messages)


def _nudge_continue(
    messages: list[Message],
    extractor: Extractor,
    total_tokens: int,
    max_tokens: int,
    warned_thresholds: set[float],
    on_message: Callable[[Message], None],
    on_step: Callable[[str], None],
    nudge_pool: NudgePool | None,
) -> None:
    """Inject a user-role nudge after a text-only model response."""
    extractor.advance(messages)
    nudge_text = _USER_UNAVAILABLE
    if nudge_pool:
        nudge_text += "\n\n" + nudge_pool.next()
    nudge: Message = {"role": "user", "content": nudge_text}
    messages.append(nudge)
    on_message(nudge)
    on_step(f"\n[nudge] {nudge_text}")
    _inject_budget_warnings(
        total_tokens, max_tokens, warned_thresholds, messages, on_message, on_step
    )


def _maybe_reflect(
    messages: list[Message],
    extractor: Extractor,
    on_message: Callable[[Message], None],
    on_step: Callable[[str], None],
    nudge_extra: str,
    nudge_pool: NudgePool | None,
) -> None:
    """Run mid-session extraction and inject a reflection nudge."""
    extractor.advance(messages)

    reflect_text = (
        "You've been using tools for a while. "
        "Pause and reflect on what you've learned and whether your approach "
        "aligns with what matters to you."
    )
    if nudge_extra:
        reflect_text += "\n\n" + nudge_extra
    if nudge_pool:
        reflect_text += "\n\n" + nudge_pool.next()
    reflect: Message = {"role": "system", "content": reflect_text}
    messages.append(reflect)
    on_message(reflect)
    on_step(f"\n[reflect] {reflect_text}")


def run_agent(
    system_prompt: str,
    model: str,
    registry: ToolRegistry | None = None,
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
    nudge_pool: NudgePool | None = None,
    on_extract: Callable[[list[Message]], None] | None = None,
    on_reflect: Callable[[], str] | None = None,
    n_experts: int = 1,
) -> str:
    """Run the agent loop driven by system_prompt alone. Returns the final text."""
    if on_confirm is None:
        on_confirm = _auto_confirm
    if registry is None:
        registry = create_default_registry()

    messages: list[Message] = list(initial_messages) if initial_messages else []
    with Stats() as stats:

        def total_tokens() -> int:
            return stats.per_model[model].output_tokens

        def total_cost() -> float:
            return stats.counters.cost

        warned_thresholds: set[float] = set()
        tool_turns_since_reflect = 0
        extractor = Extractor(on_extract, start=len(messages))

        # Register runtime tools directly on the registry.

        def _status() -> str:
            now = datetime.now().isoformat(timespec="seconds")
            lines = [
                f"time: {now}",
                f"output tokens used: {total_tokens()} / {max_tokens}",
            ]
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
                Param(
                    name="summary",
                    description="Brief summary of what was accomplished.",
                )
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
            extractor.advance(messages)
            cost_str = f", ${total_cost():.4f}" if total_cost() else ""
            stop_msg: Message = {
                "role": "system",
                "content": f"[stop] {reason} ({total_tokens()} out tokens{cost_str})",
            }
            on_message(stop_msg)
            on_step(f"\n[stop] {reason} ({total_tokens()} out tokens{cost_str})")
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
            response = chat_best_of_n(
                messages,
                model=model,
                n=n_experts,
                system_prompt=system_prompt,
                tools=registry.definitions(),
                stream=True,
                on_token=on_step,
                on_thinking=on_thinking,
                reasoning_effort=reasoning_effort,
                max_tokens=_MAX_OUTPUT_TOKENS,
            )
            messages.append(response)
            on_message(response)

            if total_tokens() >= max_tokens:
                return _stop(f"token budget exceeded (limit {max_tokens})")

            tool_calls = response.get("tool_calls")
            if not tool_calls:
                _nudge_continue(
                    messages,
                    extractor,
                    total_tokens(),
                    max_tokens,
                    warned_thresholds,
                    on_message,
                    on_step,
                    nudge_pool,
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

            tool_turns_since_reflect += 1

            # Warn the model after all tool results are appended.
            _inject_budget_warnings(
                total_tokens(),
                max_tokens,
                warned_thresholds,
                messages,
                on_message,
                on_step,
            )

            # Periodic reflection + mid-session extraction.
            if tool_turns_since_reflect >= _REFLECT_INTERVAL:
                tool_turns_since_reflect = 0
                combined_extra = nudge_extra
                if on_reflect is not None:
                    intrusive = on_reflect()
                    if intrusive:
                        combined_extra = (
                            (nudge_extra + "\n\n" + intrusive).strip()
                            if nudge_extra
                            else intrusive
                        )
                _maybe_reflect(
                    messages,
                    extractor,
                    on_message=on_message,
                    on_step=on_step,
                    nudge_extra=combined_extra,
                    nudge_pool=nudge_pool,
                )

            # Anthropic via OpenRouter only returns reasoning when the last
            # message is user-role. Inject a continue after tool results.
            if model.startswith(_ANTHROPIC_PREFIXES):
                cont: Message = {"role": "user", "content": _USER_UNAVAILABLE}
                messages.append(cont)
                on_message(cont)

    return _stop(f"max iterations reached ({max_iterations})")
