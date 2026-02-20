"""Tests for mindloop.sandbox."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mindloop.sandbox as sandbox_mod
from mindloop.sandbox import (
    _DEFAULT_TIMEOUT,
    _OUTPUT_LIMIT,
    _bwrap_available,
    _can_unshare_net,
    run_sandboxed,
)


@pytest.fixture(autouse=True)
def _reset_net_cache() -> None:
    """Reset the cached --unshare-net probe between tests."""
    sandbox_mod._net_unshare_ok = None


@pytest.fixture()
def _net_ok() -> None:
    """Pretend --unshare-net works."""
    sandbox_mod._net_unshare_ok = True


@pytest.fixture()
def _net_fail() -> None:
    """Pretend --unshare-net fails."""
    sandbox_mod._net_unshare_ok = False


def test_bwrap_available() -> None:
    """Detects bwrap when it is on PATH."""
    with patch("mindloop.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        assert _bwrap_available() is True


def test_bwrap_not_available() -> None:
    """Returns False when bwrap is not on PATH."""
    with patch("mindloop.sandbox.shutil.which", return_value=None):
        assert _bwrap_available() is False


# --- _can_unshare_net probe ---


def test_can_unshare_net_success() -> None:
    """Probe returns True when bwrap --unshare-net succeeds."""
    ok = MagicMock(returncode=0)
    with patch("mindloop.sandbox.subprocess.run", return_value=ok):
        assert _can_unshare_net() is True


def test_can_unshare_net_failure() -> None:
    """Probe returns False when bwrap --unshare-net fails."""
    fail = MagicMock(returncode=1)
    with patch("mindloop.sandbox.subprocess.run", return_value=fail):
        assert _can_unshare_net() is False


def test_can_unshare_net_cached() -> None:
    """Probe result is cached after the first call."""
    ok = MagicMock(returncode=0)
    with patch("mindloop.sandbox.subprocess.run", return_value=ok) as mock_run:
        _can_unshare_net()
        _can_unshare_net()
    mock_run.assert_called_once()


# --- run_sandboxed ---


def _make_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


@pytest.mark.usefixtures("_net_ok")
def test_run_success(tmp_path: Path) -> None:
    """Successful command returns stdout."""
    result = _make_result(stdout="hello\n")
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        out = run_sandboxed("echo hello", tmp_path)
    assert out == "hello\n"
    # Verify sh -c is used.
    cmd = mock_run.call_args[0][0]
    assert cmd[-3:] == ["sh", "-c", "echo hello"]


@pytest.mark.usefixtures("_net_ok")
def test_run_failure(tmp_path: Path) -> None:
    """Non-zero exit code is reported."""
    result = _make_result(stdout="oops\n", returncode=1)
    with patch("mindloop.sandbox.subprocess.run", return_value=result):
        out = run_sandboxed("false", tmp_path)
    assert "[exit code 1]" in out
    assert "oops" in out


@pytest.mark.usefixtures("_net_ok")
def test_run_stderr(tmp_path: Path) -> None:
    """Stderr is included under a header."""
    result = _make_result(stdout="ok\n", stderr="warn\n", returncode=0)
    with patch("mindloop.sandbox.subprocess.run", return_value=result):
        out = run_sandboxed("cmd", tmp_path)
    assert "--- stderr ---" in out
    assert "warn" in out


@pytest.mark.usefixtures("_net_ok")
def test_run_timeout(tmp_path: Path) -> None:
    """TimeoutExpired produces a timed-out message."""
    with patch(
        "mindloop.sandbox.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="bwrap", timeout=10),
    ):
        out = run_sandboxed("sleep 999", tmp_path, timeout=10)
    assert "[timed out after 10s]" in out


@pytest.mark.usefixtures("_net_ok")
def test_output_truncation(tmp_path: Path) -> None:
    """Output longer than _OUTPUT_LIMIT is truncated."""
    long_output = "x" * (_OUTPUT_LIMIT + 1000)
    result = _make_result(stdout=long_output)
    with patch("mindloop.sandbox.subprocess.run", return_value=result):
        out = run_sandboxed("cmd", tmp_path)
    assert len(out) == _OUTPUT_LIMIT


@pytest.mark.usefixtures("_net_ok")
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


@pytest.mark.usefixtures("_net_ok")
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


@pytest.mark.usefixtures("_net_ok")
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


@pytest.mark.usefixtures("_net_ok")
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


@pytest.mark.usefixtures("_net_ok")
def test_default_timeout(tmp_path: Path) -> None:
    """Default timeout is passed to subprocess.run."""
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("echo hi", tmp_path)
    assert mock_run.call_args[1]["timeout"] == _DEFAULT_TIMEOUT


@pytest.mark.usefixtures("_net_ok")
def test_unshare_net_included(tmp_path: Path) -> None:
    """--unshare-net is in the command when the probe succeeds."""
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("ls", tmp_path)
    cmd = mock_run.call_args[0][0]
    assert "--unshare-net" in cmd


@pytest.mark.usefixtures("_net_fail")
def test_unshare_net_omitted(tmp_path: Path) -> None:
    """--unshare-net is omitted when the probe fails."""
    result = _make_result()
    with patch("mindloop.sandbox.subprocess.run", return_value=result) as mock_run:
        run_sandboxed("ls", tmp_path)
    cmd = mock_run.call_args[0][0]
    assert "--unshare-net" not in cmd
