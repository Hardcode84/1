"""Tests for mindloop.sandbox."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


from mindloop.sandbox import (
    _DEFAULT_TIMEOUT,
    _OUTPUT_LIMIT,
    _bwrap_available,
    run_sandboxed,
)


def test_bwrap_available() -> None:
    """Detects bwrap when it is on PATH."""
    with patch("mindloop.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        assert _bwrap_available() is True


def test_bwrap_not_available() -> None:
    """Returns False when bwrap is not on PATH."""
    with patch("mindloop.sandbox.shutil.which", return_value=None):
        assert _bwrap_available() is False


def _make_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


def test_run_success(tmp_path: Path) -> None:
    """Successful command returns stdout."""
    result = _make_result(stdout="hello\n")
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        out = run_sandboxed("echo hello", tmp_path)
    assert out == "hello\n"
    # Verify sh -c is used.
    cmd = mock_run.call_args[0][0]
    assert cmd[-3:] == ["sh", "-c", "echo hello"]


def test_run_failure(tmp_path: Path) -> None:
    """Non-zero exit code is reported."""
    result = _make_result(stdout="oops\n", returncode=1)
    with patch("mindloop.sandbox.subprocess.run", return_value=result):
        out = run_sandboxed("false", tmp_path)
    assert "[exit code 1]" in out
    assert "oops" in out


def test_run_stderr(tmp_path: Path) -> None:
    """Stderr is included under a header."""
    result = _make_result(stdout="ok\n", stderr="warn\n", returncode=0)
    with patch("mindloop.sandbox.subprocess.run", return_value=result):
        out = run_sandboxed("cmd", tmp_path)
    assert "--- stderr ---" in out
    assert "warn" in out


def test_run_timeout(tmp_path: Path) -> None:
    """TimeoutExpired produces a timed-out message."""
    with patch(
        "mindloop.sandbox.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="bwrap", timeout=10),
    ):
        out = run_sandboxed("sleep 999", tmp_path, timeout=10)
    assert "[timed out after 10s]" in out


def test_output_truncation(tmp_path: Path) -> None:
    """Output longer than _OUTPUT_LIMIT is truncated."""
    long_output = "x" * (_OUTPUT_LIMIT + 1000)
    result = _make_result(stdout=long_output)
    with patch("mindloop.sandbox.subprocess.run", return_value=result):
        out = run_sandboxed("cmd", tmp_path)
    assert len(out) == _OUTPUT_LIMIT


def test_symlinks_as_ro_bind(tmp_path: Path) -> None:
    """Virtual symlinks become --ro-bind args."""
    target = Path("/some/target")
    symlinks = {"docs": target}
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("ls", tmp_path, symlinks=symlinks)
    cmd = mock_run.call_args[0][0]
    # Find the --ro-bind pair for the symlink.
    ro_binds = [
        (cmd[i + 1], cmd[i + 2]) for i in range(len(cmd)) if cmd[i] == "--ro-bind"
    ]
    mount_point = str(tmp_path.resolve() / "docs")
    assert (str(target.resolve()), mount_point) in ro_binds


def test_env_file_injection(tmp_path: Path) -> None:
    """A .env file in the workspace injects --setenv args."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n# comment\n\nBAZ=qux\n")
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("env", tmp_path)
    cmd = mock_run.call_args[0][0]
    setenvs = [
        (cmd[i + 1], cmd[i + 2]) for i in range(len(cmd)) if cmd[i] == "--setenv"
    ]
    assert ("FOO", "bar") in setenvs
    assert ("BAZ", "qux") in setenvs
    # Comments and blank lines should not appear.
    keys = [k for k, _ in setenvs]
    assert "#" not in "".join(keys)


def test_python_prefix_bound(tmp_path: Path) -> None:
    """sys.prefix is bound read-only."""
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("python --version", tmp_path)
    cmd = mock_run.call_args[0][0]
    ro_binds = [
        (cmd[i + 1], cmd[i + 2]) for i in range(len(cmd)) if cmd[i] == "--ro-bind"
    ]
    assert (sys.prefix, sys.prefix) in ro_binds


def test_workspace_is_writable_bind(tmp_path: Path) -> None:
    """Workspace is mounted with --bind (writable), not --ro-bind."""
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("touch test", tmp_path)
    cmd = mock_run.call_args[0][0]
    ws = str(tmp_path.resolve())
    # Find --bind pairs (writable mounts).
    binds = [(cmd[i + 1], cmd[i + 2]) for i in range(len(cmd)) if cmd[i] == "--bind"]
    assert (ws, ws) in binds
    # Should NOT appear in --ro-bind.
    ro_binds = [
        (cmd[i + 1], cmd[i + 2]) for i in range(len(cmd)) if cmd[i] == "--ro-bind"
    ]
    assert (ws, ws) not in ro_binds


def test_default_timeout(tmp_path: Path) -> None:
    """Default timeout is passed to subprocess.run."""
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("echo hi", tmp_path)
    assert mock_run.call_args[1]["timeout"] == _DEFAULT_TIMEOUT
