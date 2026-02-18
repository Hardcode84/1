"""Tests for mindloop.pool."""

from __future__ import annotations

from mindloop.pool import Executor, submit


def test_submit_no_context() -> None:
    """Without an Executor context, submit runs inline and returns a resolved future."""
    fut = submit(lambda x: x * 2, 21)
    assert fut.done()
    assert fut.result() == 42


def test_submit_with_context() -> None:
    """Inside an Executor context, submit delegates to the thread pool."""
    with Executor(workers=2):
        fut = submit(lambda x: x + 1, 9)
        assert fut.result() == 10


def test_nesting() -> None:
    """Inner Executor takes precedence; outer is restored on exit."""
    with Executor(workers=4):
        with Executor(workers=1):
            # Inner pool is active.
            fut = submit(lambda: "inner")
            assert fut.result() == "inner"
        # Outer pool restored.
        fut = submit(lambda: "outer")
        assert fut.result() == "outer"
    # No pool active — inline fallback.
    fut = submit(lambda: "none")
    assert fut.done()
    assert fut.result() == "none"


def test_submit_exception_no_context() -> None:
    """Exceptions are captured in the future when running inline."""
    fut = submit(int, "not_a_number")
    assert fut.done()
    try:
        fut.result()
        raise AssertionError("Expected ValueError")  # pragma: no cover
    except ValueError:
        pass


def test_submit_exception_with_context() -> None:
    """Exceptions propagate through the future inside an Executor context."""
    with Executor(workers=1):
        fut = submit(int, "bad")
        try:
            fut.result()
            raise AssertionError("Expected ValueError")  # pragma: no cover
        except ValueError:
            pass
