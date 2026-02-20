"""Sandboxed command execution via bubblewrap (bwrap)."""

import shutil
import subprocess
import sys
from pathlib import Path

_OUTPUT_LIMIT = 8000
_DEFAULT_TIMEOUT = 30

# System paths to bind read-only (skipped if they don't exist on the host).
_SYSTEM_RO_BINDS = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/alternatives"]


def _bwrap_available() -> bool:
    """Return True if bwrap is on PATH."""
    return shutil.which("bwrap") is not None


def run_sandboxed(
    command: str,
    workspace: Path,
    symlinks: dict[str, Path] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> str:
    """Execute *command* inside a bwrap sandbox rooted at *workspace*.

    Returns combined stdout/stderr, truncated to ``_OUTPUT_LIMIT`` chars.
    """
    ws = str(workspace.resolve())

    cmd: list[str] = ["bwrap"]

    # Read-only system directories.
    for p in _SYSTEM_RO_BINDS:
        if Path(p).exists():
            cmd += ["--ro-bind", p, p]

    # Read-only Python environment.
    cmd += ["--ro-bind", sys.prefix, sys.prefix]

    # Writable workspace.
    cmd += ["--bind", ws, ws]

    # Minimal system.
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]

    # Isolation.
    cmd += ["--unshare-net", "--unshare-pid", "--die-with-parent"]

    # Working directory.
    cmd += ["--chdir", ws]

    # Virtual symlinks as read-only bind mounts.
    if symlinks:
        for name, target in symlinks.items():
            mount_point = str((workspace.resolve() / name))
            cmd += ["--ro-bind", str(target.resolve()), mount_point]

    # Inject .env variables.
    env_file = workspace.resolve() / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            cmd += ["--setenv", key.strip(), value.strip()]

    # Command via sh -c to support pipes and chaining.
    cmd += ["--", "sh", "-c", command]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"

    output = proc.stdout
    if proc.stderr:
        output += "\n--- stderr ---\n" + proc.stderr
    if proc.returncode != 0:
        output += f"\n[exit code {proc.returncode}]"
    return output[:_OUTPUT_LIMIT]
