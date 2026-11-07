from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


class StateError(ValueError):
    """A run state transition or payload is invalid."""


STEPS = ("prestudy", "baseline", "arm", "diagnosis", "report")
PHASES = (
    "created",
    *(
        f"{step}_{suffix}"
        for step in STEPS
        for suffix in ("pending", "running", "done")
    ),
    "blocked_subject",
    "blocked_docker",
    "invalid_measurement",
    "crashed",
)
BLOCKED = frozenset(
    {"blocked_subject", "blocked_docker", "invalid_measurement", "crashed"}
)
MANUAL_BLOCKS = frozenset({"blocked_subject", "invalid_measurement"})
TERMINAL = frozenset({"report_done"})
ACTORS = ("runner", "human")


def _timestamp(at: str | None) -> str:
    if at is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not isinstance(at, str) or not at.strip():
        raise StateError(f"at must be a non-empty string: {at!r}")
    return at


@dataclass(frozen=True)
class RunState:
    run_id: str
    phase: str
    resume: str | None = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise StateError(f"run_id must be a non-empty string: {self.run_id!r}")
        if self.phase not in PHASES:
            raise StateError(f"phase must be one of {PHASES}: {self.phase!r}")
        if self.resume is not None and self.resume not in PHASES:
            raise StateError(f"resume must be a phase: {self.resume!r}")
        if not isinstance(self.updated_at, str):
            raise StateError("updated_at must be a string")

    @classmethod
    def initial(cls, run_id: str, *, at: str | None = None) -> "RunState":
        return cls(run_id=run_id, phase="created", resume=None, updated_at=_timestamp(at))

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "resume": self.resume,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "RunState":
        if not isinstance(payload, Mapping):
            raise StateError("state must be a JSON object")
        run_id = payload.get("run_id")
        phase = payload.get("phase")
        if not isinstance(run_id, str) or not run_id.strip():
            raise StateError("state has no run_id")
        if not isinstance(phase, str) or phase not in PHASES:
            raise StateError(f"state has an unknown phase: {phase!r}")
        resume = payload.get("resume")
        if resume is not None and (not isinstance(resume, str) or resume not in PHASES):
            raise StateError(f"state has an unknown resume phase: {resume!r}")
        updated_at = payload.get("updated_at", "")
        if not isinstance(updated_at, str):
            raise StateError("state updated_at must be a string")
        return cls(
            run_id=run_id,
            phase=phase,
            resume=resume,
            updated_at=updated_at or _timestamp(None),
        )


def _expected(phase: str) -> str:
    if phase == "created":
        return "prestudy_pending"
    for index, step in enumerate(STEPS):
        if phase == f"{step}_pending":
            return f"{step}_running"
        if phase == f"{step}_running":
            return f"{step}_done"
        if phase == f"{step}_done":
            return (
                f"{STEPS[index + 1]}_pending"
                if index + 1 < len(STEPS)
                else "report_done"
            )
    raise StateError(f"phase has no successor: {phase!r}")


def advance(
    state: RunState,
    to: str,
    *,
    artifact: str | None = None,
    actor: str = "runner",
    at: str | None = None,
) -> RunState:
    if not isinstance(state, RunState):
        raise StateError("state must be a RunState")
    if to not in PHASES:
        raise StateError(f"unknown target phase: {to!r}")
    if actor not in ACTORS:
        raise StateError(f"actor must be one of {ACTORS}: {actor!r}")
    if artifact is not None and (
        not isinstance(artifact, str) or not artifact.strip()
    ):
        raise StateError(f"artifact must be a non-empty string: {artifact!r}")
    current = state.phase
    timestamp = _timestamp(at)
    if current in TERMINAL:
        raise StateError(f"{current} is terminal; no transition is allowed")
    if to in BLOCKED:
        if current in BLOCKED:
            raise StateError(f"already blocked in {current!r}")
        return RunState(state.run_id, to, current, timestamp)
    if current in BLOCKED:
        if to != state.resume:
            raise StateError(
                f"resume must return to {state.resume!r}, not {to!r}"
            )
        if current in MANUAL_BLOCKS and actor != "human":
            raise StateError(f"{current} can only be cleared by a human")
        return RunState(state.run_id, to, None, timestamp)
    expected = _expected(current)
    if to != expected:
        raise StateError(f"cannot move from {current} to {to}; expected {expected}")
    if to.endswith("_done") and artifact is None:
        raise StateError(f"{to} requires an artifact")
    return RunState(state.run_id, to, None, timestamp)
