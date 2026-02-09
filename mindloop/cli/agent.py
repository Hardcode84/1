"""CLI entry point for the autonomous agent loop."""

from pathlib import Path

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


def main() -> None:
    if not API_KEY:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    system_prompt = _PROMPT_PATH.read_text().strip()
    print("Starting agent...\n")
    run_agent(system_prompt, on_step=_print_step, on_thinking=_print_thinking)
    print()


if __name__ == "__main__":
    main()
