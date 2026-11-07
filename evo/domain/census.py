from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping


class CensusError(ValueError):
    """Coverage traces cannot be mapped to source units."""


@dataclass(frozen=True)
class Unit:
    path: str
    qualname: str
    name: str
    def_line: int
    span: tuple[int, int]
    full_span: tuple[int, int]

    @property
    def body_lines(self) -> range:
        return range(self.span[0], self.span[1] + 1)


@dataclass(frozen=True)
class UnitTouch:
    unit: Unit
    tasks: int
    total: int
    task_ids: tuple[str, ...]


def _units_in(
    node: ast.AST, path: str, prefix: str = "", seen: set[str] | None = None
) -> list[Unit]:
    seen = seen if seen is not None else set()
    units: list[Unit] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{prefix}{child.name}"
            if qualname in seen:
                raise CensusError(f"duplicate unit {qualname} in {path}")
            seen.add(qualname)
            first = min((stmt.lineno for stmt in child.body), default=child.lineno)
            end = child.end_lineno or first
            decorators = [decorator.lineno for decorator in child.decorator_list]
            units.append(
                Unit(
                    path=path,
                    qualname=qualname,
                    name=child.name,
                    def_line=child.lineno,
                    span=(first, end),
                    full_span=(min([child.lineno, *decorators]), end),
                )
            )
            units.extend(_units_in(child, path, f"{qualname}.", seen))
        elif isinstance(child, ast.ClassDef):
            units.extend(_units_in(child, path, f"{prefix}{child.name}.", seen))
        else:
            units.extend(_units_in(child, path, prefix, seen))
    return units


def function_units(sources: Mapping[str, str]) -> tuple[Unit, ...]:
    units: list[Unit] = []
    for path, source in sources.items():
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            raise CensusError(f"cannot parse {path}: {exc}") from exc
        units.extend(_units_in(tree, path))
    return tuple(units)


def touch_report(
    units: Collection[Unit],
    traces: Mapping[str, Mapping[str, Collection[int]]],
    task_ids: Collection[str],
) -> tuple[UnitTouch, ...]:
    tasks = tuple(task_ids)
    if not tasks:
        raise CensusError("task_ids must not be empty")
    if len(set(tasks)) != len(tasks):
        raise CensusError(f"duplicate task ids: {tasks}")
    unknown = sorted(set(traces) - set(tasks))
    if unknown:
        raise CensusError(f"traces contain unknown tasks: {unknown}")

    report = []
    for unit in units:
        touching = []
        for task in tasks:
            lines = traces.get(task, {}).get(unit.path, ())
            if any(unit.span[0] <= line <= unit.span[1] for line in lines):
                touching.append(task)
        report.append(
            UnitTouch(
                unit=unit,
                tasks=len(touching),
                total=len(tasks),
                task_ids=tuple(touching),
            )
        )
    return tuple(report)


def zero_touch(report: Collection[UnitTouch]) -> tuple[UnitTouch, ...]:
    return tuple(row for row in report if row.tasks == 0)


@dataclass(frozen=True)
class TraceSet:
    subject: str
    root: str
    task_ids: tuple[str, ...]
    tasks: Mapping[str, Mapping[str, frozenset[int]]]


_TRACE_KEYS = {"subject", "root", "task_ids", "tasks"}


def _positive_lines(path: str, task_id: str, values: object) -> frozenset[int]:
    if not isinstance(values, list):
        raise CensusError(
            f"trace lines for task {task_id!r} path {path!r} must be a list"
        )
    lines = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CensusError(
                f"line numbers must be positive integers in task {task_id!r} path {path!r}: {value!r}"
            )
        lines.add(value)
    return frozenset(lines)


def load_traces(path: Path | str) -> TraceSet:
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CensusError(f"cannot read traces {source}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CensusError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CensusError("traces payload must be a JSON object")
    unknown = sorted(set(payload) - _TRACE_KEYS)
    if unknown:
        raise CensusError(f"unknown trace keys: {unknown}")
    missing = sorted(_TRACE_KEYS - set(payload))
    if missing:
        raise CensusError(f"missing trace keys: {missing}")

    subject = payload["subject"]
    root = payload["root"]
    if not isinstance(subject, str) or not subject:
        raise CensusError("subject must be a non-empty string")
    if not isinstance(root, str) or not root:
        raise CensusError("root must be a non-empty string")

    raw_task_ids = payload["task_ids"]
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        raise CensusError("task_ids must be a non-empty list")
    if not all(isinstance(task, str) and task for task in raw_task_ids):
        raise CensusError("task_ids must be non-empty strings")
    task_ids = tuple(raw_task_ids)
    if len(set(task_ids)) != len(task_ids):
        raise CensusError(f"duplicate task ids: {list(task_ids)}")

    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, dict):
        raise CensusError("tasks must be a JSON object")
    missing_tasks = sorted(set(task_ids) - set(raw_tasks))
    extra_tasks = sorted(set(raw_tasks) - set(task_ids))
    if missing_tasks or extra_tasks:
        raise CensusError(
            f"tasks do not match task_ids: missing={missing_tasks}, extra={extra_tasks}"
        )

    tasks: dict[str, dict[str, frozenset[int]]] = {}
    for task_id in task_ids:
        files = raw_tasks[task_id]
        if not isinstance(files, dict):
            raise CensusError(f"task {task_id!r} trace must be a JSON object")
        normalized: dict[str, frozenset[int]] = {}
        for trace_path, values in files.items():
            if not isinstance(trace_path, str) or not trace_path:
                raise CensusError("trace paths must be non-empty strings")
            normalized[trace_path] = _positive_lines(trace_path, task_id, values)
        tasks[task_id] = normalized

    return TraceSet(subject=subject, root=root, task_ids=task_ids, tasks=tasks)
