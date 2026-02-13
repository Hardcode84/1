"""CLI for humans to send and list messages in a session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mindloop.messages import parse_message, write_message

_SESSIONS_DIR = Path("sessions")


def _send(args: argparse.Namespace) -> None:
    """Send a message to a session's inbox."""
    from datetime import datetime

    inbox = _SESSIONS_DIR / args.session / "_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = inbox / f"{ts}.txt"
    body: str = args.body
    if body is None:
        if sys.stdin.isatty():
            print("Enter message body (Ctrl+D to finish):")
        body = sys.stdin.read().strip()
        if not body:
            print("Empty body, aborting.")
            return
    write_message(path, sender=args.sender, title=args.title, body=body)
    print(f"Sent: {path}")


def _list(args: argparse.Namespace) -> None:
    """List messages in a session's inbox or outbox."""
    session_root = _SESSIONS_DIR / args.session
    target = session_root / ("_outbox" if args.outbox else "_inbox")
    if not target.is_dir():
        print(f"Directory not found: {target}")
        return
    files = sorted(target.glob("*.txt"))
    if not files:
        print("No messages.")
        return
    for i, f in enumerate(files, 1):
        msg = parse_message(f)
        print(f'#{i} {msg.date} "{msg.sender}", "{msg.title}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Send and list session messages.")
    sub = parser.add_subparsers(dest="command")

    send_p = sub.add_parser("send", help="Send a message to a session's inbox.")
    send_p.add_argument("--session", required=True, help="Session name.")
    send_p.add_argument(
        "--from", dest="sender", default="Admin", help="Sender name (default: Admin)."
    )
    send_p.add_argument("--title", required=True, help="Message subject.")
    send_p.add_argument(
        "body",
        nargs="?",
        default=None,
        help="Message body text (reads from stdin if omitted).",
    )

    list_p = sub.add_parser("list", help="List messages in a session.")
    list_p.add_argument("--session", required=True, help="Session name.")
    list_p.add_argument("--outbox", action="store_true", help="List outbox instead.")

    args = parser.parse_args()
    if args.command == "send":
        _send(args)
    elif args.command == "list":
        _list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
