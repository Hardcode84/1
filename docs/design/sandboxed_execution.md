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

## Module structure

Core sandbox logic lives in `mindloop/sandbox.py`:

- `_bwrap_available()` — checks if bwrap is on PATH. Called at tool registration time; prints a warning and skips registration if missing.
- `run_sandboxed(command, workspace, symlinks, timeout)` — builds the bwrap command and runs it via `subprocess.run`. Commands are executed via `sh -c` to support pipes and chaining.

Tool registration happens in `mindloop/cli/agent.py` via `_register_run_tool()`, gated behind the `--allow-exec` CLI flag.

## CLI integration

The `run` tool is opt-in via `--allow-exec`:

```bash
mindloop-agent --session mysession --allow-exec
mindloop-daemon --session mysession --schedule "0 9 * * *" --allow-exec
```

Both `mindloop/cli/agent.py` and `mindloop/cli/daemon.py` accept the flag. The daemon plumbs it through `_build_cmd` / `run_daemon` to the agent subprocess.

## Symlinks as bind-mounts

Virtual symlinks (`_symlinks.json`) are passed as `--ro-bind` args to bwrap. Each virtual symlink becomes a read-only mount point inside the sandbox:

```python
if symlinks:
    for name, target in symlinks.items():
        mount_point = str(workspace.resolve() / name)
        cmd += ["--ro-bind", str(target.resolve()), mount_point]
```

The agent sees symlinked dirs as subdirectories of its workspace, read-only, and they vanish when the process ends.

## Environment variables

Each bwrap invocation is a separate process — env vars set inside one run die with it. To persist env vars across runs, `run_sandboxed` reads a `.env` file from the workspace and injects each entry via `--setenv`. Comments and blank lines are skipped.

The agent can create and update `.env` with its existing `write` tool — no new mechanism needed.

## Output handling

- stdout and stderr are combined (stderr under a `--- stderr ---` header).
- Non-zero exit codes append `[exit code N]`.
- `TimeoutExpired` produces `[timed out after Ns]`.
- Output is truncated to 8000 chars to prevent context flooding.

## Resolved decisions

- **Per-session opt-in:** `--allow-exec` CLI flag (not always-on).
- **bwrap availability:** Checked at registration time; warning printed, tool skipped if missing.
- **Command execution:** Uses `sh -c command` to support pipes, redirects, and chaining.
- **Timeout default:** 30s, configurable per-call via the `timeout` tool parameter.
- **Network access:** Disabled by default. Future: could add `allow_network` flag.
