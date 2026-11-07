from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


class TraceMergeError(ValueError):
    """Coverage trace payloads cannot be merged into one task set."""


def _lines(path: str, task: str, values: object) -> set[int]:
    if not isinstance(values, list):
        raise TraceMergeError(f"lines for {task!r} {path!r} must be a list")
    lines: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TraceMergeError(
                f"line numbers for {task!r} {path!r} must be positive integers"
            )
        lines.add(value)
    return lines


def merge_trace_payloads(payloads: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Union per-trial traces into one payload keyed by benchmark task id."""
    if not payloads:
        raise TraceMergeError("at least one trace payload is required")
    subject: str | None = None
    root: str | None = None
    order: list[str] = []
    tasks: dict[str, dict[str, set[int]]] = {}
    for index, payload in enumerate(payloads):
        if not isinstance(payload, Mapping):
            raise TraceMergeError(f"trace payload {index} must be an object")
        current_subject = payload.get("subject")
        current_root = payload.get("root")
        if not isinstance(current_subject, str) or not current_subject:
            raise TraceMergeError(f"trace payload {index} has no subject")
        if not isinstance(current_root, str) or not current_root:
            raise TraceMergeError(f"trace payload {index} has no root")
        if subject is None:
            subject, root = current_subject, current_root
        elif current_subject != subject:
            raise TraceMergeError(
                f"trace payload {index} subject {current_subject!r} != {subject!r}"
            )
        elif current_root != root:
            raise TraceMergeError(
                f"trace payload {index} root {current_root!r} != {root!r}"
            )
        task_ids = payload.get("task_ids")
        raw_tasks = payload.get("tasks")
        if not isinstance(task_ids, list) or not isinstance(raw_tasks, Mapping):
            raise TraceMergeError(f"trace payload {index} has malformed tasks")
        for task in task_ids:
            if not isinstance(task, str) or not task:
                raise TraceMergeError(f"trace payload {index} has an invalid task id")
            if task not in tasks:
                order.append(task)
                tasks[task] = {}
            files = raw_tasks.get(task)
            if not isinstance(files, Mapping):
                raise TraceMergeError(f"trace payload {index} has no files for {task!r}")
            for path, values in files.items():
                if not isinstance(path, str) or not path:
                    raise TraceMergeError(f"trace payload {index} has an invalid path")
                tasks[task].setdefault(path, set()).update(
                    _lines(path, task, values)
                )
    return {
        "subject": subject,
        "root": root,
        "task_ids": order,
        "tasks": {
            task: {path: sorted(lines) for path, lines in sorted(files.items())}
            for task, files in tasks.items()
        },
    }


def merge_trace_dir(directory: str | Path) -> dict[str, object]:
    base = Path(directory)
    if not base.is_dir():
        raise TraceMergeError(f"trace directory does not exist: {base}")
    payloads = []
    for path in sorted(base.glob("*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TraceMergeError(f"cannot read trace {path}: {exc}") from exc
    if not payloads:
        raise TraceMergeError(f"trace directory has no JSON files: {base}")
    return merge_trace_payloads(payloads)
