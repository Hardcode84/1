"""CLI entry point for the autonomous agent loop."""

import json
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

from mindloop.agent import run_agent
from mindloop.client import API_KEY

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "system_prompt.md"


def _print_step(text: str) -> None:
    """Print streaming tokens inline, tool steps on their own line."""
    if text.startswith("[tool]") or text.startswith("[result]"):
        print(f"\n{text}", flush=True)
    else:
        print(text, end="", flush=True)


def _print_thinking(text: str) -> None:
    """Print reasoning tokens dimmed."""
    print(f"\033[2m{text}\033[0m", end="", flush=True)


def _format_message(message: dict[str, Any]) -> str:
    """Format a message dict as a human-readable string."""
    role = message.get("role", "unknown")
    content = message.get("content") or ""
    reasoning = message.get("reasoning")
    tool_calls = message.get("tool_calls")
    parts: list[str] = []
    if reasoning:
        parts.append(f"[{role} thinking] {reasoning}")
    if role == "tool":
        call_id = message.get("tool_call_id", "")
        parts.append(f"[tool result {call_id}] {content}")
    elif tool_calls:
        calls = [
            f"  {tc['function']['name']}({tc['function']['arguments']})"
            for tc in tool_calls
        ]
        parts.append(f"[{role}] tool calls:\n" + "\n".join(calls))
    else:
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _make_logger(jsonl_path: Path, log_path: Path) -> Callable[[dict[str, Any]], None]:
    """Return a callback that logs each message in both formats."""

    def _log(message: dict[str, Any]) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        # Structured JSONL log.
        entry = {"timestamp": ts, **message}
        with jsonl_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        # Human-readable log.
        with log_path.open("a") as f:
            f.write(f"{ts}  {_format_message(message)}\n")

    return _log


def main() -> None:
    if not API_KEY:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = log_dir / f"agent_{timestamp}.jsonl"
    log_path = log_dir / f"agent_{timestamp}.log"

    system_prompt = _PROMPT_PATH.read_text().strip()
    print(f"Starting agent... (logging to {log_path})\n")
    run_agent(
        system_prompt,
        on_step=_print_step,
        on_thinking=_print_thinking,
        on_message=_make_logger(jsonl_path, log_path),
    )
    print()


if __name__ == "__main__":
    main()
