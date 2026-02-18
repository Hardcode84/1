"""Manage virtual symlinks for a session."""

import argparse
import json
from pathlib import Path

from mindloop.cli.agent import _SESSIONS_DIR, _SYMLINKS_FILE


def _symlinks_path(session: str) -> Path:
    """Return the symlinks config path for a session."""
    return _SESSIONS_DIR / session / _SYMLINKS_FILE


def _load(path: Path) -> dict[str, str]:
    """Load existing symlinks or return empty dict."""
    if path.is_file():
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    return {}


def _save(path: Path, data: dict[str, str]) -> None:
    """Write symlinks config."""
    path.write_text(json.dumps(data, indent=2) + "\n")


def _cmd_add(args: argparse.Namespace) -> None:
    """Add a virtual symlink."""
    target = Path(args.path).resolve()
    if not target.is_dir():
        print(f"Error: {args.path} is not a directory.")
        return
    path = _symlinks_path(args.session)
    data = _load(path)
    data[args.name] = str(target)
    _save(path, data)
    print(f"Added: {args.name} -> {target}")


def _cmd_remove(args: argparse.Namespace) -> None:
    """Remove a virtual symlink."""
    path = _symlinks_path(args.session)
    data = _load(path)
    if args.name not in data:
        print(f"Error: symlink '{args.name}' not found.")
        return
    del data[args.name]
    if data:
        _save(path, data)
    else:
        path.unlink()
    print(f"Removed: {args.name}")


def _cmd_list(args: argparse.Namespace) -> None:
    """List virtual symlinks."""
    path = _symlinks_path(args.session)
    data = _load(path)
    if not data:
        print("No symlinks configured.")
        return
    for name, target in sorted(data.items()):
        print(f"{name} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage session virtual symlinks.")
    parser.add_argument("session", help="Session name.")
    sub = parser.add_subparsers(dest="command")

    add_p = sub.add_parser("add", help="Add a symlink.")
    add_p.add_argument("name", help="Symlink name visible to the agent.")
    add_p.add_argument("path", help="Absolute or relative path to mount.")

    rm_p = sub.add_parser("remove", help="Remove a symlink.")
    rm_p.add_argument("name", help="Symlink name to remove.")

    sub.add_parser("list", help="List configured symlinks.")

    args = parser.parse_args()
    if args.command is None:
        # Default to list.
        args.command = "list"
        _cmd_list(args)
    elif args.command == "add":
        _cmd_add(args)
    elif args.command == "remove":
        _cmd_remove(args)
    elif args.command == "list":
        _cmd_list(args)


if __name__ == "__main__":
    main()
