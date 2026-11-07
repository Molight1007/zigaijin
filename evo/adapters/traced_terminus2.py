from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from evo.adapters.coverage_py import LineCollector, relativize


_TRACE_DIR = (
    Path(os.environ["EVO_TRACE_DIR"]).resolve()
    if os.environ.get("EVO_TRACE_DIR")
    else None
)
_SUBJECT_ROOT = (
    Path(os.environ["EVO_SUBJECT_ROOT"]).resolve()
    if os.environ.get("EVO_SUBJECT_ROOT")
    else None
)
_LOCK = threading.Lock()
_COLLECTOR: LineCollector | None = None


def trial_name(environment: Any) -> str:
    """Return the stable Harbor trial name used as the coverage task key."""
    paths = getattr(environment, "trial_paths", None)
    trial_dir = getattr(paths, "trial_dir", None)
    if trial_dir is not None:
        name = Path(trial_dir).name
        if name:
            return name
    session = str(getattr(environment, "session_id", "") or "")
    return session.removesuffix("__env") or "trial"


def collector() -> LineCollector | None:
    global _COLLECTOR
    if _TRACE_DIR is None or _SUBJECT_ROOT is None:
        return None
    with _LOCK:
        if _COLLECTOR is None:
            try:
                _COLLECTOR = LineCollector([_SUBJECT_ROOT])
                _COLLECTOR.start()
            except Exception as exc:  # noqa: BLE001 - coverage must not kill the trial
                print(f"coverage disabled: {exc}", file=sys.stderr)
                _COLLECTOR = None
                return None
            atexit.register(_flush_all)
        return _COLLECTOR


def coverage_task_id(trial: str) -> str:
    """Map a Harbor trial name to the benchmark task id used by census."""
    return trial.split("__", 1)[0] or trial


def write_trial(
    trace_collector: LineCollector,
    subject_root: Path,
    out_dir: Path,
    trial: str,
) -> Path | None:
    """Write one trial's relative line traces to its own JSON artifact."""
    traces = relativize(trace_collector.traces(), subject_root)
    files = traces.get(trial, {})
    if not files:
        return None
    task_id = coverage_task_id(trial)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": Path(subject_root).name,
        "root": str(Path(subject_root).resolve()),
        "task_ids": [task_id],
        "tasks": {
            task_id: {path: sorted(lines) for path, lines in files.items()},
        },
    }
    target = out / f"{trial}.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def safe_write_trial(
    trace_collector: LineCollector,
    subject_root: Path,
    out_dir: Path,
    trial: str,
) -> Path | None:
    """Write traces without letting a coverage bug fail the agent trial."""
    try:
        return write_trial(trace_collector, subject_root, out_dir, trial)
    except Exception as exc:  # noqa: BLE001 - coverage is best-effort
        print(f"coverage write failed for {trial}: {exc}", file=sys.stderr)
        return None


def _flush_all() -> None:
    global _COLLECTOR
    with _LOCK:
        current = _COLLECTOR
        _COLLECTOR = None
        if current is None:
            return
        try:
            if _SUBJECT_ROOT is not None and _TRACE_DIR is not None:
                for trial in current.traces():
                    safe_write_trial(current, _SUBJECT_ROOT, _TRACE_DIR, trial)
        finally:
            current.stop()


try:
    from terminus2 import AgentHarness as _BaseAgentHarness
except Exception:  # pragma: no cover - only when Harbor has not added vendor/
    _BaseAgentHarness = None


if _BaseAgentHarness is None:
    class TracedAgentHarness:
        """Placeholder used when the subject package is not importable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "terminus2 is not importable; add its parent directory to PYTHONPATH"
            )
else:
    class TracedAgentHarness(_BaseAgentHarness):
        """Terminus-2 wrapper that records per-trial line coverage."""

        async def run(self, instruction, environment, context):
            trace_collector = collector()
            if trace_collector is None:
                return await super().run(instruction, environment, context)
            trial = trial_name(environment)
            with trace_collector.capture(trial):
                try:
                    return await super().run(instruction, environment, context)
                finally:
                    safe_write_trial(
                        trace_collector, _SUBJECT_ROOT, _TRACE_DIR, trial
                    )
