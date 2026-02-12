"""CLI entry point for the autonomous agent loop."""

import argparse
import json
import select
import shutil
import sys
import uuid
from dataclasses import dataclass, field
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


_ASK_TIMEOUT = 120


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

# System messages injected by the agent loop that shouldn't be replayed.
_SKIP_PREFIXES = ("[stop]", "[stats]", "Warning:")


def _load_messages(path: Path) -> list[dict[str, Any]]:
    """Load messages from a JSONL log, stripping metadata and agent-loop noise."""
    messages: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        # Strip log-only fields.
        entry.pop("timestamp", None)
        entry.pop("usage", None)
        # Skip system messages injected by the agent loop.
        if entry.get("role") == "system":
            content = entry.get("content", "")
            if any(content.startswith(p) for p in _SKIP_PREFIXES):
                continue
        messages.append(entry)
    return messages


def _latest_jsonl(log_dir: Path) -> Path | None:
    """Find the most recent JSONL file in a directory."""
    files = sorted(log_dir.glob("*_agent_*.jsonl"))
    return files[-1] if files else None


# --- Session path setup ---

_SESSIONS_DIR = Path("sessions")
_TEMPLATE_DIR = Path("workspace_template")
_ISOLATED_BLOCKED = [
    Path("logs"),
    Path("artifacts"),
    Path("memory"),
    _SESSIONS_DIR,
]


@dataclass
class SessionPaths:
    """Resolved paths for a single agent run."""

    log_dir: Path
    db_path: Path
    name: str | None = None
    instance: int = 0
    workspace: Path | None = None
    blocked_dirs: list[Path] = field(default_factory=list)


def _setup_session(session: str | None, isolated: bool, timestamp: str) -> SessionPaths:
    """Resolve log, memory, and workspace paths."""
    if isolated and not session:
        session = f"isolated_{timestamp}"

    if session:
        root = _SESSIONS_DIR / session
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        workspace = root / "workspace"
        fresh = not workspace.exists()
        workspace.mkdir(exist_ok=True)
        if fresh and _TEMPLATE_DIR.is_dir():
            shutil.copytree(_TEMPLATE_DIR, workspace, dirs_exist_ok=True)
        instance = len(list(log_dir.glob("*_agent_*.jsonl"))) + 1
        blocked = list(_ISOLATED_BLOCKED) if isolated else []
        return SessionPaths(
            log_dir=log_dir,
            db_path=root / "memory.db",
            name=session,
            instance=instance,
            workspace=workspace,
            blocked_dirs=blocked,
        )

    # Default: shared logs + memory.
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    mem_dir = Path("memory")
    mem_dir.mkdir(exist_ok=True)
    return SessionPaths(log_dir=log_dir, db_path=mem_dir / "memory.db")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous agent.")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Model to use (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--session",
        metavar="NAME",
        help="Run inside a named session (sessions/<NAME>/).",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Create a new session with an auto-generated name.",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run with fresh memory and no access to logs/artifacts/memory/sessions.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        metavar="JSONL",
        help="Resume from a JSONL log. With --session, auto-finds the latest log.",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("Set OPENROUTER_API_KEY environment variable first.")
        print("Get a free key at https://openrouter.ai/keys")
        return

    # Validate flag combinations.
    if args.session and args.new_session:
        parser.error("--session and --new-session are mutually exclusive.")
    if args.new_session and args.resume is not None:
        parser.error("--new-session and --resume are mutually exclusive.")
    if args.isolated and args.resume is not None and not args.session:
        parser.error(
            "--isolated --resume requires --session to identify which session to resume."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model: str = args.model
    session_name: str | None = args.session
    if args.new_session:
        session_name = uuid.uuid4().hex[:8]
        while (_SESSIONS_DIR / session_name).exists():
            session_name = uuid.uuid4().hex[:8]
    paths = _setup_session(session_name, args.isolated, timestamp)

    if paths.instance:
        log_prefix = f"{paths.instance:03d}_agent_{timestamp}"
    else:
        log_prefix = f"agent_{timestamp}"
    jsonl_path = paths.log_dir / f"{log_prefix}.jsonl"
    log_path = paths.log_dir / f"{log_prefix}.log"

    system_prompt = _PROMPT_PATH.read_text().strip()
    if paths.instance:
        system_prompt += f"\n\nYou are instance {paths.instance}."
    registry = create_default_registry(
        blocked_dirs=paths.blocked_dirs or None,
        root_dir=paths.workspace,
    )
    mt = add_memory_tools(registry, db_path=paths.db_path, model=model)

    # Handle --resume: explicit path or auto-find latest in session.
    initial_messages: list[dict[str, Any]] | None = None
    if args.resume is not None:
        if args.resume is True:
            # Auto-find latest JSONL in session log dir.
            resume_path = _latest_jsonl(paths.log_dir)
            if resume_path is None:
                print(f"No JSONL logs found in {paths.log_dir}")
                return
        else:
            resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Resume file not found: {resume_path}")
            return
        initial_messages = _load_messages(resume_path)
        print(
            f"Resuming from {resume_path} ({len(initial_messages)} messages)\n"
            f"  logging to {log_path}, memory: {paths.db_path}\n"
        )
    else:
        label = f"session: {paths.name}, " if paths.name else ""
        print(
            f"Starting agent... ({label}logging to {log_path},"
            f" memory: {paths.db_path})\n"
        )

    # Log the final system prompt.
    logger = _make_logger(jsonl_path, log_path)
    logger({"role": "system", "content": system_prompt})
    print(f"\033[2m[system] {system_prompt}\033[0m\n")

    # Session workspace is isolated; skip confirmation for file operations.
    confirm = _confirm_tool if paths.workspace is None else None

    try:
        run_agent(
            system_prompt,
            registry=registry,
            on_step=_print_step,
            model=model,
            on_thinking=_print_thinking,
            on_message=logger,
            on_confirm=confirm,
            on_ask=_ask_user,
            initial_messages=initial_messages,
            instance=paths.instance,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        mt.close()

    if paths.name:
        model_flag = f" --model {model}" if model != _DEFAULT_MODEL else ""
        print(
            f"\nTo resume:    mindloop-agent --session {paths.name} --resume{model_flag}"
        )
        print(f"New instance: mindloop-agent --session {paths.name}{model_flag}")


if __name__ == "__main__":
    main()
