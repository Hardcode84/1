"""CLI entry point for the autonomous agent loop."""

import argparse
import json
import select
import sys
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

from mindloop.agent import run_agent
from mindloop.client import API_KEY
from mindloop.tools import add_memory_tools, create_default_registry

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


_DESTRUCTIVE_TOOLS = {"edit", "write"}


def _confirm_tool(name: str, arguments: str) -> bool:
    """Prompt user for confirmation on destructive tools."""
    if name not in _DESTRUCTIVE_TOOLS:
        return True
    print(f"\n\033[33m[confirm] {name}({arguments})\033[0m")
    reply = input("Allow? [y/N] ").strip().lower()
    return reply in ("y", "yes")


_ASK_TIMEOUT = 60


def _ask_user(message: str) -> str:
    """Print agent's message and wait for user input with timeout."""
    print(f"\n\033[36m[ask] {message}\033[0m")
    print(f"(waiting {_ASK_TIMEOUT}s for response)")
    ready, _, _ = select.select([sys.stdin], [], [], _ASK_TIMEOUT)
    if ready:
        return sys.stdin.readline().rstrip("\n")
    print("(no response, continuing)")
    return "User is unavailable."


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


_DEFAULT_MODEL = "deepseek/deepseek-v3.2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous agent.")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Model to use (default: {_DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = log_dir / f"agent_{timestamp}.jsonl"
    log_path = log_dir / f"agent_{timestamp}.log"
    model: str = args.model

    system_prompt = _PROMPT_PATH.read_text().strip()

    mem_dir = Path("memory")
    mem_dir.mkdir(exist_ok=True)
    db_path = mem_dir / "memory.db"
    registry = create_default_registry()
    mt = add_memory_tools(registry, db_path=db_path, model=model)

    print(f"Starting agent... (logging to {log_path}, memory: {db_path})\n")
    try:
        run_agent(
            system_prompt,
            registry=registry,
            on_step=_print_step,
            model=model,
            on_thinking=_print_thinking,
            on_message=_make_logger(jsonl_path, log_path),
            on_confirm=_confirm_tool,
            on_ask=_ask_user,
        )
    finally:
        mt.close()
    print()


if __name__ == "__main__":
    main()
