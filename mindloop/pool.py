"""Global thread pool with arena-style context management."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, TypeVar

from mindloop.util import DEFAULT_WORKERS

T = TypeVar("T")

_current: ThreadPoolExecutor | None = None


class Executor:
    """Context manager that creates a shared thread pool.

    Nesting is stack-based: inner executor takes precedence, outer is
    restored on exit.  ``submit()`` delegates to the innermost active
    executor, or runs inline when none is active.
    """

    def __init__(self, workers: int = DEFAULT_WORKERS) -> None:
        self._workers = workers
        self._pool: ThreadPoolExecutor | None = None
        self._prev: ThreadPoolExecutor | None = None

    def __enter__(self) -> Executor:
        global _current  # noqa: PLW0603
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._prev = _current
        _current = self._pool
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        global _current  # noqa: PLW0603
        _current = self._prev
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def submit(fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
    """Submit *fn* to the active executor, or run it inline.

    When no ``Executor`` context is active the function executes
    synchronously in the calling thread and a resolved future is returned.
    This keeps tests working without any pool setup.
    """
    if _current is not None:
        return _current.submit(fn, *args, **kwargs)
    fut: Future[Any] = Future()
    try:
        fut.set_result(fn(*args, **kwargs))
    except Exception as exc:
        fut.set_exception(exc)
    return fut
