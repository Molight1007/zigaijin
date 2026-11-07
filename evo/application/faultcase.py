from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from evo.application.diagnose import (
    CATEGORIES,
    DiagnosisCase,
    DiagnosisComponent,
)


class FaultCaseError(ValueError):
    """A failing trial cannot be turned into a diagnosis case."""


@dataclass(frozen=True)
class TrialFailure:
    trial_id: str
    task_id: str
    status: str
    reward: float | None
    trace: str

    @property
    def failed(self) -> bool:
        return self.status != "completed" or self.reward is None or self.reward < 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "status": self.status,
            "reward": self.reward,
            "trace": self.trace,
        }


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FaultCaseError(f"{name} is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FaultCaseError(f"{name} cannot be read: {path}: {exc}") from exc


def _reward(payload: dict[str, object]) -> float | None:
    verifier = payload.get("verifier_result")
    if not isinstance(verifier, dict):
        return None
    rewards = verifier.get("rewards")
    value = rewards.get("reward") if isinstance(rewards, dict) else verifier.get("reward")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FaultCaseError(f"verifier reward must be a number: {value!r}")
    return float(value)


def read_failure(
    trial_dir: str | Path,
    *,
    max_steps: int = 10,
    max_chars: int = 8_000,
    verifier_chars: int = 1_500,
    step_chars: int = 1_200,
) -> TrialFailure:
    for name, value in (
        ("max_steps", max_steps),
        ("max_chars", max_chars),
        ("verifier_chars", verifier_chars),
        ("step_chars", step_chars),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FaultCaseError(f"{name} must be a positive integer: {value!r}")
    base = Path(trial_dir)
    if not base.is_dir():
        raise FaultCaseError(f"trial directory is not a directory: {trial_dir}")

    result = _read_json(base / "result.json", "result.json")
    if not isinstance(result, dict):
        raise FaultCaseError(f"result.json must be a JSON object: {trial_dir}")
    task_id = result.get("task_name")
    trial_id = result.get("trial_name")
    if not isinstance(task_id, str) or not task_id.strip():
        raise FaultCaseError(f"result.json has no task_name: {base}")
    if not isinstance(trial_id, str) or not trial_id.strip():
        raise FaultCaseError(f"result.json has no trial_name: {base}")
    reward = _reward(result)
    exception = result.get("exception_info")
    if exception is None:
        status = "completed" if reward is not None else "agent_error"
    elif isinstance(exception, dict):
        status = str(exception.get("exception_type") or "error")
    else:
        status = "error"

    trajectory_path = base / "agent" / "trajectory.json"
    if not trajectory_path.is_file():
        raise FaultCaseError(f"trajectory is missing: {trajectory_path}")
    trajectory = _read_json(trajectory_path, "trajectory")
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if not isinstance(steps, list) or not steps:
        raise FaultCaseError(f"trajectory has no steps: {trajectory_path}")

    lines = [
        f"trial {trial_id} | task {task_id} | status {status} | reward {reward}",
        "",
    ]
    for step in steps[-max_steps:]:
        if not isinstance(step, dict):
            raise FaultCaseError(f"trajectory has a malformed step: {trajectory_path}")
        message = str(step.get("message") or "")
        if len(message) > step_chars:
            message = message[:step_chars] + " ...[step truncated]"
        lines.append(f"[{step.get('source') or 'unknown'}] {message}")
    verifier_path = base / "verifier" / "test-stdout.txt"
    if verifier_path.is_file():
        try:
            output = verifier_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise FaultCaseError(f"cannot read {verifier_path}: {exc}") from exc
        lines += ["", "--- verifier ---", output[-verifier_chars:].rstrip()]
    trace = "\n".join(lines).rstrip() + "\n"
    if len(trace) > max_chars:
        trace = "...[truncated]\n" + trace[-(max_chars):]
    return TrialFailure(
        trial_id=trial_id,
        task_id=task_id,
        status=status,
        reward=reward,
        trace=trace,
    )


def build_case(
    case_id: str,
    failure: TrialFailure,
    *,
    culprit: str,
    components: Sequence[DiagnosisComponent],
    category: str = "harness",
) -> DiagnosisCase:
    if not isinstance(failure, TrialFailure):
        raise FaultCaseError(f"failure must be a TrialFailure: {failure!r}")
    if not failure.failed:
        raise FaultCaseError(
            f"trial {failure.trial_id!r} did not fail; a fault case needs a failing run"
        )
    if category not in CATEGORIES:
        raise FaultCaseError(
            f"category must be one of {', '.join(CATEGORIES)}: {category!r}"
        )
    ids = [component.id for component in components]
    if culprit not in ids:
        raise FaultCaseError(f"culprit {culprit!r} is not in the component list")
    return DiagnosisCase(
        case_id=case_id,
        failure=failure.trace,
        components=tuple(components),
        culprit=culprit,
        category=category,
    )


def cases_payload(cases: Sequence[DiagnosisCase]) -> dict[str, object]:
    return {"schema": 1, "cases": [case.as_dict() for case in cases]}
