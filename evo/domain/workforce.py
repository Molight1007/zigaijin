from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from evo.contracts.types import KVector


LAYERS = ("L1", "L2", "L3", "L4")
PLAN_MODES = ("delete", "edit")
VERDICTS = ("keep", "delete", "undecidable", "undeletable")
SCREEN_STATUSES = ("pass", "reject")


class WorkforceError(ValueError):
    """A workforce contract or artifact is malformed."""


@dataclass(frozen=True)
class RoleSpec:
    name: str
    question: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    source: str
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "question": self.question,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "source": self.source,
            "depends_on": list(self.depends_on),
        }


ROLE_SPECS = (
    RoleSpec(
        name="monitor",
        question="What happened in the tests, and which trials are weak?",
        inputs=("reference outcomes", "traces", "coverage"),
        outputs=("monitor.json",),
        source="RRSI analyst/digester (observation); deterministic monitoring here",
    ),
    RoleSpec(
        name="architect",
        question="What is the code architecture, and where are removable units?",
        inputs=("subject root", "boundary spec", "census/ablation artifacts"),
        outputs=("architecture.json",),
        source="Our addition; RRSI has static components but no architect employee",
    ),
    RoleSpec(
        name="summarizer",
        question="Given test evidence and the code map, which problems matter and what is the plan?",
        inputs=("monitor report", "architecture map", "history", "budget"),
        outputs=("plan.json",),
        source="RRSI proposer (planning half)",
        depends_on=("monitor", "architect"),
    ),
    RoleSpec(
        name="executor",
        question="Can the plan be applied exactly as specified?",
        inputs=("plan",),
        outputs=("execution.json", "diff.patch"),
        source="RRSI proposer (execution half); split from planning for auditability",
        depends_on=("summarizer",),
    ),
    RoleSpec(
        name="reviewer",
        question="Is the executor's output safe and worth accepting?",
        inputs=("execution", "verification", "evaluation", "gate inputs"),
        outputs=("review.json", "ledger.jsonl"),
        source="RRSI critic (pre-evaluation) + selection (post-evaluation) + this project's gate",
        depends_on=("executor",),
    ),
)

ROLE_BY_NAME = {spec.name: spec for spec in ROLE_SPECS}


@dataclass(frozen=True)
class PlanResult:
    mode: str
    candidate_id: str
    layer: str
    summary: str
    problems: tuple[Mapping[str, object], ...] = ()
    batch: tuple[Mapping[str, object], ...] = ()
    edits: tuple[Mapping[str, object], ...] = ()
    target_k_ast: int | None = None
    diff: str = ""
    staged_root: Path | None = None
    brief: str = ""
    directions: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in PLAN_MODES:
            raise WorkforceError(
                f"plan mode must be one of {', '.join(PLAN_MODES)}: {self.mode!r}"
            )
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise WorkforceError("plan candidate_id must be a non-empty string")
        if self.layer not in LAYERS:
            raise WorkforceError(
                f"plan layer must be one of {', '.join(LAYERS)}: {self.layer!r}"
            )
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise WorkforceError("plan summary must be a non-empty string")
        for name in ("problems", "batch", "edits"):
            rows = getattr(self, name)
            if any(not isinstance(row, Mapping) for row in rows):
                raise WorkforceError(f"plan {name} rows must be mappings")
        if self.target_k_ast is not None and (
            isinstance(self.target_k_ast, bool)
            or not isinstance(self.target_k_ast, int)
            or self.target_k_ast <= 0
        ):
            raise WorkforceError("target_k_ast must be a positive integer or None")
        if not isinstance(self.diff, str):
            raise WorkforceError("plan diff must be a string")
        if not isinstance(self.brief, str):
            raise WorkforceError("plan brief must be a string")
        if any(not isinstance(row, Mapping) for row in self.directions):
            raise WorkforceError("plan directions must be mappings")
        if self.staged_root is not None:
            staged = Path(self.staged_root)
            if not staged.is_dir():
                raise WorkforceError(f"plan staged_root is not a directory: {staged}")
            object.__setattr__(self, "staged_root", staged)
        if (
            self.mode == "edit"
            and not self.diff.strip()
            and self.staged_root is None
            and not self.brief.strip()
        ):
            raise WorkforceError("edit plans need a diff, a staged_root, or a brief")

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "candidate_id": self.candidate_id,
            "layer": self.layer,
            "summary": self.summary,
            "problems": [dict(row) for row in self.problems],
            "batch": [dict(row) for row in self.batch],
            "edits": [dict(row) for row in self.edits],
            "target_k_ast": self.target_k_ast,
            "diff": self.diff,
            "staged_root": str(self.staged_root) if self.staged_root is not None else None,
            "brief": self.brief,
            "directions": [dict(row) for row in self.directions],
        }


@dataclass(frozen=True)
class ExecutionResult:
    plan_id: str
    root: Path
    changed: tuple[str, ...]
    diff: str
    k_before: KVector | None = None
    k_after: KVector | None = None
    protection: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    summary: str = "applied"

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise WorkforceError("execution plan_id must be a non-empty string")
        root = Path(self.root)
        if not root.is_dir():
            raise WorkforceError(f"execution root is not a directory: {root}")
        if any(not isinstance(path, str) or not path for path in self.changed):
            raise WorkforceError("execution changed paths must be non-empty strings")
        if not isinstance(self.diff, str):
            raise WorkforceError("execution diff must be a string")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise WorkforceError("execution summary must be a non-empty string")
        for name in ("k_before", "k_after"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, KVector):
                raise WorkforceError(f"execution {name} must be a KVector or None")
        if any(not isinstance(hit, str) or not hit for hit in self.protection):
            raise WorkforceError("execution protection hits must be non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.components):
            raise WorkforceError("execution components must be non-empty strings")
        object.__setattr__(self, "root", root)

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "root": str(self.root),
            "changed": list(self.changed),
            "diff": self.diff,
            "k_before": self.k_before.as_dict() if self.k_before is not None else None,
            "k_after": self.k_after.as_dict() if self.k_after is not None else None,
            "protection": list(self.protection),
            "components": list(self.components),
            "summary": self.summary,
        }


def find_task_leakage(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return task ids or test-path markers that appear in a candidate diff."""
    if not isinstance(text, str):
        raise WorkforceError("leakage text must be a string")
    found: list[str] = []
    for term in terms:
        if term and term in text and term not in found:
            found.append(term)
    has_tests = "/tests/" in text or "\\tests\\" in text
    if has_tests:
        found.append("/tests/")
    return tuple(found)
