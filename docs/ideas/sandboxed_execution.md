# Sandboxed Execution

Allow the agent to run programs (Python scripts, shell commands) inside its workspace sandbox, isolated from the host system via bubblewrap (`bwrap`).

## Motivation

Agents currently have filesystem tools (read, write, edit, ls, mv) but cannot execute code. Running scripts enables data processing, computation, and self-testing — but must be isolated to prevent host damage.

## Approach: bubblewrap

`bwrap` is a lightweight sandbox using Linux user namespaces. Single static binary, no daemon, no root. Used by Flatpak.

Key properties:
- Only the workspace directory is writable.
- Python environment (`sys.prefix`) is bind-mounted read-only — interpreter, stdlib, and all installed packages are available.
- Network disabled (`--unshare-net`).
- Separate PID namespace (`--unshare-pid`).
- Child killed if parent dies (`--die-with-parent`).
- Timeout via `subprocess.run(timeout=...)`.

Install: `apt install bubblewrap`.

## Tool sketch

```python
def _run(reg: ToolRegistry, command: str, timeout: int = 30) -> str:
    """Execute a command in a sandboxed environment."""
    import subprocess, sys, shlex

    workspace = str(reg.root_dir.resolve())
    python_prefix = sys.prefix

    cmd = [
        "bwrap",
        # Read-only system libs.
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        # Read-only Python environment (interpreter + all packages).
        "--ro-bind", python_prefix, python_prefix,
        # Writable workspace only.
        "--bind", workspace, workspace,
        # Minimal system.
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # Isolation.
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        "--chdir", workspace,
        "--",
        *shlex.split(command),
    ]

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    output = proc.stdout
    if proc.stderr:
        output += "\n--- stderr ---\n" + proc.stderr
    if proc.returncode != 0:
        output += f"\n[exit code {proc.returncode}]"
    return output[:8000]
```

## Registration

```python
registry.add("run", "Execute a command in the sandbox.", {
    "command": {"type": "string", "description": "Shell command to execute."},
    "timeout": {"type": "integer", "description": "Max seconds.", "default": 30},
}, partial(_run, registry))
```

## Symlinks as bind-mounts

Current virtual symlinks (`_symlinks.json`) only exist in the tool layer's Python logic — a subprocess inside bwrap wouldn't see them. Two options:

1. **Real symlinks on session start, remove on shutdown.** Simple, but cleanup is fragile if the agent crashes mid-session.
2. **Pass as `--ro-bind` args to bwrap.** Each virtual symlink becomes a read-only mount point inside the sandbox. No filesystem changes, automatic cleanup when bwrap exits.

Option 2 is better. The `_run` tool builder iterates `reg.symlinks` and appends bind-mount args:

```python
if reg.symlinks:
    for name, target in reg.symlinks.items():
        mount_point = str(reg.root_dir.resolve() / name)
        cmd += ["--ro-bind", str(target.resolve()), mount_point]
```

The agent sees symlinked dirs as subdirectories of its workspace (e.g. `/workspace/docs/`), read-only, and they vanish when the process ends. This replaces the current virtual symlink resolution for sandboxed commands while keeping the existing tool-layer resolution for `read`/`ls`/etc.

## Environment variables

Each bwrap invocation is a separate process — env vars set inside one run die with it. To persist env vars across runs, the `_run` tool reads a `.env` file from the workspace and injects each entry via `--setenv`:

```python
env_file = reg.root_dir.resolve() / ".env"
if env_file.is_file():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        cmd += ["--setenv", key.strip(), value.strip()]
```

The agent can create and update `.env` with its existing `write` tool — no new mechanism needed.

## Open questions

- **Network access.** Some tasks need it (web scraping, API calls). Could add a per-session or per-tool `allow_network` flag. Default off.
- **Timeout default.** 30s may be too short for data processing. Consider 60s or configurable per-session.
- **Per-session opt-in.** Maybe a `_config.json` flag or CLI `--allow-exec` rather than always-on.
- **bwrap availability.** Fail gracefully with a clear error if not installed. Could check at tool registration time.
- **Output cap.** 8000 chars prevents context flooding, but long outputs may need truncation strategy (head + tail).
