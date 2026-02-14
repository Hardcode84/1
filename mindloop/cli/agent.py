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
from mindloop.quotes import NudgePool, quote_of_the_day
from mindloop.recap import generate_recap, load_recap, save_recap
from mindloop.messages import parse_filename_date
from mindloop.tools import (
    Param,
    add_memory_tools,
    add_message_tools,
    create_default_registry,
)
from mindloop.util import SKIP_PREFIXES

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "system_prompt.md"
_NOTES_MAX_CHARS = 2000


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
_DEFAULT_SUMMARIZER_MODEL = "deepseek/deepseek-v3.2"


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
            if any(content.startswith(p) for p in SKIP_PREFIXES):
                continue
        messages.append(entry)
    return messages


def _session_exit_reason(jsonl_path: Path) -> str | None:
    """Extract how a session ended from its JSONL log.

    Returns a human-readable reason string, or None for a clean ``done`` exit.
    """
    stop_line: str | None = None
    for line in reversed(jsonl_path.read_text().splitlines()):
        if not line.strip():
            continue
        entry = json.loads(line)
        content = entry.get("content", "")
        if entry.get("role") == "system" and content.startswith("[stop]"):
            stop_line = content
            break
    if stop_line is None:
        return "Previous session terminated abruptly (crash or interrupt)."
    if "model finished" in stop_line:
        return None  # Clean exit.
    if "token budget exceeded" in stop_line:
        return "Previous session ran out of tokens before finishing."
    if "max iterations reached" in stop_line:
        return "Previous session hit the iteration limit before finishing."
    return f"Previous session ended unexpectedly: {stop_line}"


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
        help=f"Model to use for the agent (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--summarizer-model",
        default=_DEFAULT_SUMMARIZER_MODEL,
        help=f"Model for summaries, merges, and recaps (default: {_DEFAULT_SUMMARIZER_MODEL}).",
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
    summarizer_model: str = args.summarizer_model
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
        system_prompt += f"\n\nYou are instance {paths.instance} (1-based)."
    registry = create_default_registry(
        blocked_dirs=paths.blocked_dirs or None,
        root_dir=paths.workspace,
    )
    mt = add_memory_tools(
        registry, db_path=paths.db_path, model=summarizer_model, log=_print_step
    )

    # Register note_to_self tool when a workspace exists.
    notes_path: Path | None = None
    if paths.workspace:
        notes_path = paths.workspace / "_notes.md"
        # Block direct write/edit; reads are allowed.
        registry.write_blocked[notes_path.resolve()] = "use note_to_self tool instead"

        def _note_to_self(content: str) -> str:
            assert notes_path is not None
            if len(content) > _NOTES_MAX_CHARS:
                return (
                    f"Error: content is {len(content)} chars, "
                    f"max is {_NOTES_MAX_CHARS}. Trim and retry."
                )
            previous = notes_path.read_text() if notes_path.is_file() else None
            notes_path.write_text(content)
            result = f"Saved {len(content)} chars to notes."
            if previous:
                result += f"\n\n--- Previous notes (overwritten) ---\n{previous}"
            return result

        registry.add(
            name="note_to_self",
            description=(
                "Write notes for your next instance. Overwrites previous notes. "
                "Use for directives, user preferences, and task status. "
                f"Max {_NOTES_MAX_CHARS} chars — curate, don't append."
            ),
            params=[
                Param(
                    name="content",
                    description=f"Markdown content (max {_NOTES_MAX_CHARS} chars).",
                )
            ],
            func=_note_to_self,
        )

    # Set up inbox/outbox messaging when a workspace (session) exists.
    msg_tools = None
    if paths.workspace:
        session_root = paths.workspace.parent
        inbox_dir = session_root / "_inbox"
        outbox_dir = session_root / "_outbox"
        inbox_dir.mkdir(exist_ok=True)
        outbox_dir.mkdir(exist_ok=True)

        # Current session start — messages at or after this are invisible.
        before = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

        # Previous session start — messages after this are "new".
        since: datetime | None = None
        prev_logs = sorted(paths.log_dir.glob("*_agent_*.jsonl"))
        if len(prev_logs) >= 1:
            # The current log hasn't been created yet, so the last entry
            # in prev_logs is the previous instance's log.
            prev_ts = parse_filename_date(prev_logs[-1].name)
            if prev_ts is not None:
                since = prev_ts

        msg_tools = add_message_tools(
            registry,
            inbox_dir,
            outbox_dir,
            paths.instance,
            before=before,
            since=since,
        )
        # Block direct writes to inbox.
        registry.write_blocked[inbox_dir.resolve()] = ""

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

    # Load or generate recap from previous instance.
    # _recap.md is overwritten (not appended) at session end. If an instance
    # crashes before generating a new recap, the next instance still sees
    # the previous one as a fallback.
    if paths.workspace and initial_messages is None:
        prev_log = _latest_jsonl(paths.log_dir)
        if prev_log and prev_log != jsonl_path:
            exit_reason = _session_exit_reason(prev_log)
            if exit_reason:
                system_prompt += f"\n\n# Warning\n{exit_reason}"

        recap_path = paths.workspace / "_recap.md"
        registry.write_blocked[recap_path.resolve()] = "autogenerated between sessions"
        recap = load_recap(recap_path)
        if recap is None:
            if prev_log and prev_log != jsonl_path:
                prev_msgs = _load_messages(prev_log)
                if prev_msgs:
                    recap = generate_recap(prev_msgs, model=summarizer_model, log=print)
        if recap:
            system_prompt += f"\n\n# Previous session recap\n{recap}"

    # Load notes from previous instance.
    if notes_path and notes_path.is_file():
        notes = load_recap(notes_path)
        if notes:
            system_prompt += f"\n\n# Notes from previous instance\n{notes}"

    # Append quote of the day to system prompt.
    system_prompt += f"\n\n# Quote of the day\n{quote_of_the_day()}"

    # Append message count to system prompt.
    nudge_extra = ""
    if msg_tools is not None:
        nudge_extra = msg_tools.new_message_note
        system_prompt += f"\n\n# Messages\n{nudge_extra}"

    # Log the final system prompt.
    logger = _make_logger(jsonl_path, log_path)
    logger({"role": "system", "content": system_prompt})
    print(f"\033[2m[system] {system_prompt}\033[0m\n")

    # Session workspace is isolated; skip confirmation for file operations.
    confirm = _confirm_tool if paths.workspace is None else None

    nudge_pool = NudgePool()

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
            nudge_extra=nudge_extra,
            nudge_pool=nudge_pool,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        mt.close()
        print("\n")
        # Generate recap for the next instance.
        if paths.workspace and jsonl_path.exists():
            try:
                msgs = _load_messages(jsonl_path)
                if msgs:
                    recap = generate_recap(msgs, model=summarizer_model, log=print)
                    save_recap(paths.workspace / "_recap.md", recap)
            except Exception:
                pass  # Don't crash cleanup on recap failure.

    if paths.name:
        model_flag = f" --model {model}" if model != _DEFAULT_MODEL else ""
        summ_flag = (
            f" --summarizer-model {summarizer_model}"
            if summarizer_model != _DEFAULT_SUMMARIZER_MODEL
            else ""
        )
        flags = model_flag + summ_flag
        print(f"\nTo resume:    mindloop-agent --session {paths.name} --resume{flags}")
        print(f"New instance: mindloop-agent --session {paths.name}{flags}")


if __name__ == "__main__":
    main()
