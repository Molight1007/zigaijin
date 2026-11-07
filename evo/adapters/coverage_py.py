from __future__ import annotations

import contextvars
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Collection, Iterator, Mapping


class CoverageError(RuntimeError):
    """Line coverage cannot be collected or mapped."""


_TOOL_NAME = "evo-coverage"


class LineCollector:
    """Collect executed line numbers per task with sys.monitoring.

    Attribution uses a context variable, so concurrent asyncio trials in one
    process stay separate as long as each run is wrapped in :meth:`capture`.
    """

    def __init__(self, roots: Collection[Path | str]):
        self._roots = tuple(Path(root).resolve() for root in roots)
        self._context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"{_TOOL_NAME}-task", default=None
        )
        self._traces: dict[str, dict[str, set[int]]] = {}
        self._lock = threading.Lock()
        self._filename_cache: dict[str, str | None] = {}
        self._tool_id: int | None = None

    @property
    def active(self) -> bool:
        return self._tool_id is not None

    def _scope_path(self, filename: str) -> str | None:
        if filename in self._filename_cache:
            return self._filename_cache[filename]
        try:
            resolved = Path(filename).resolve()
        except (OSError, ValueError):
            scoped = None
        else:
            scoped = None
            if any(resolved == root or resolved.is_relative_to(root) for root in self._roots):
                scoped = str(resolved)
        self._filename_cache[filename] = scoped
        return scoped

    def _on_line(self, code, line: int) -> None:
        task = self._context.get()
        if task is None:
            return
        path = self._scope_path(code.co_filename)
        if path is None:
            return
        with self._lock:
            files = self._traces.setdefault(task, {})
            files.setdefault(path, set()).add(line)

    def start(self) -> None:
        if self._tool_id is not None:
            raise CoverageError("collector already started")
        tool_id = sys.monitoring.COVERAGE_ID
        if sys.monitoring.get_tool(tool_id) is not None:
            raise CoverageError(f"monitoring tool id {tool_id} is already in use")
        sys.monitoring.use_tool_id(tool_id, _TOOL_NAME)
        try:
            sys.monitoring.register_callback(
                tool_id, sys.monitoring.events.LINE, self._on_line
            )
            sys.monitoring.set_events(tool_id, sys.monitoring.events.LINE)
        except BaseException:
            sys.monitoring.set_events(tool_id, 0)
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, None)
            sys.monitoring.free_tool_id(tool_id)
            raise
        self._tool_id = tool_id

    def stop(self) -> None:
        tool_id, self._tool_id = self._tool_id, None
        if tool_id is None:
            return
        sys.monitoring.set_events(tool_id, 0)
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, None)
        sys.monitoring.free_tool_id(tool_id)

    @contextmanager
    def capture(self, task_id: str) -> Iterator[None]:
        if self._tool_id is None:
            raise CoverageError("collector not started")
        if not task_id:
            raise CoverageError("task id must not be empty")
        with self._lock:
            self._traces.setdefault(task_id, {})
        token = self._context.set(task_id)
        try:
            yield
        finally:
            self._context.reset(token)

    def traces(self) -> dict[str, dict[str, frozenset[int]]]:
        with self._lock:
            return {
                task: {path: frozenset(lines) for path, lines in files.items()}
                for task, files in self._traces.items()
            }


def relativize(
    traces: Mapping[str, Mapping[str, Collection[int]]], root: Path | str
) -> dict[str, dict[str, frozenset[int]]]:
    base = Path(root).resolve()
    relative: dict[str, dict[str, frozenset[int]]] = {}
    for task, files in traces.items():
        out: dict[str, frozenset[int]] = {}
        for path, lines in files.items():
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(base):
                raise CoverageError(f"traced file outside root {base}: {path}")
            out[resolved.relative_to(base).as_posix()] = frozenset(lines)
        relative[task] = out
    return relative
