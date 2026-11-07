from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from evo.application.baseline import BaselineError, TrialOutcome, build_baseline
from evo.contracts.types import KVector
from evo.domain.arms import ArmError, arm_config


ARM_POOLS = ("smoke", "evolution", "holdout", "all")
K_FIELDS = ("k_lines_nb", "k_ast", "k_chars", "k_bytes", "k_gzip", "k_cc")
_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
)


class ArmRunError(ValueError):
    """An arm evaluation violates the frozen protocol."""


@dataclass(frozen=True)
class ArmRunPlan:
    arm: str
    pool: str
    task_ids: tuple[str, ...]
    k_reps: int
    subject: str
    subject_root: str
    subject_hash: str
    k: KVector

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "pool": self.pool,
            "task_ids": list(self.task_ids),
            "k_reps": self.k_reps,
            "subject": self.subject,
            "subject_root": self.subject_root,
            "subject_hash": self.subject_hash,
            "k": self.k.as_dict(),
        }


def subject_digest(root: str | Path) -> str:
    base = Path(root)
    if not base.is_dir():
        raise ArmRunError(f"subject root is not a directory: {root}")
    files = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        relative = path.relative_to(base)
        if any(part in _SKIP_DIRS for part in relative.parts[:-1]):
            continue
        files.append(relative.as_posix())
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((base / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def pool_ids(pool: str, pools: Mapping[str, object]) -> tuple[str, ...]:
    if pool not in ARM_POOLS:
        raise ArmRunError(
            f"unknown pool {pool!r}; expected one of {', '.join(ARM_POOLS)}"
        )
    if not isinstance(pools, Mapping):
        raise ArmRunError(f"pools must be a mapping: {type(pools).__name__}")
    names = ("smoke", "evolution", "holdout") if pool == "all" else (pool,)
    ids: list[str] = []
    for name in names:
        values = pools.get(name)
        if values is None:
            if pool == "all":
                continue
            raise ArmRunError(f"pools has no task pool {name!r}")
        if not isinstance(values, (list, tuple)):
            raise ArmRunError(f"pool {name!r} must be a list of task ids")
        if not values and pool != "all":
            raise ArmRunError(f"pool {pool!r} is empty")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ArmRunError(f"pool {name!r} has a non-string task id: {value!r}")
            ids.append(value)
    if not ids:
        raise ArmRunError(f"pool {pool!r} is empty")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for task in ids:
        if task in seen:
            duplicates.add(task)
        seen.add(task)
    if duplicates:
        raise ArmRunError(
            f"pools contain duplicate tasks: {', '.join(sorted(duplicates))}"
        )
    return tuple(ids)


def plan_arm_run(
    *,
    arm: str,
    pool: str,
    pools: Mapping[str, object],
    k_reps: int,
    subject: str,
    subject_root: str | Path,
    k: KVector,
) -> ArmRunPlan:
    try:
        arm_config(arm)
    except ArmError as exc:
        raise ArmRunError(str(exc)) from exc
    task_ids = pool_ids(pool, pools)
    if isinstance(k_reps, bool) or not isinstance(k_reps, int) or k_reps <= 0:
        raise ArmRunError(f"k_reps must be a positive integer: {k_reps!r}")
    if not isinstance(subject, str) or not subject.strip():
        raise ArmRunError(f"subject must be a non-empty string: {subject!r}")
    if not isinstance(k, KVector):
        raise ArmRunError(f"k must be a KVector: {k!r}")
    base = Path(subject_root)
    if not base.is_dir():
        raise ArmRunError(f"subject root is not a directory: {subject_root}")
    return ArmRunPlan(
        arm=arm,
        pool=pool,
        task_ids=task_ids,
        k_reps=k_reps,
        subject=subject,
        subject_root=str(base.resolve()),
        subject_hash=subject_digest(base),
        k=k,
    )


def check_frozen_subject(
    arm: str, measured: KVector, frozen_k: Mapping[str, object] | None
) -> None:
    if arm != "H0":
        return
    if not isinstance(frozen_k, Mapping):
        raise ArmRunError("H0 requires the frozen k vector from run.json")
    missing = [name for name in K_FIELDS if name not in frozen_k]
    if missing:
        raise ArmRunError(
            f"frozen k vector is missing fields: {', '.join(missing)}"
        )
    changes = {
        name: (frozen_k[name], getattr(measured, name))
        for name in K_FIELDS
        if frozen_k[name] != getattr(measured, name)
    }
    if changes:
        detail = ", ".join(
            f"{name} {before} -> {after}" for name, (before, after) in changes.items()
        )
        raise ArmRunError(
            f"H0 subject does not match the frozen baseline: {detail}"
        )


def build_arm_payload(
    plan: ArmRunPlan, outcomes: Sequence[TrialOutcome]
) -> dict[str, object]:
    try:
        report = build_baseline(
            outcomes,
            tasks=plan.task_ids,
            k=plan.k_reps,
            k_vector=plan.k,
        )
    except BaselineError as exc:
        raise ArmRunError(str(exc)) from exc
    return {
        "schema": 1,
        "arm": plan.arm,
        "arm_config": arm_config(plan.arm).as_dict(),
        "pool": plan.pool,
        "task_ids": list(plan.task_ids),
        "k_reps": plan.k_reps,
        "subject": plan.subject,
        "subject_root": plan.subject_root,
        "subject_hash": plan.subject_hash,
        **report,
        "outcomes": [asdict(outcome) for outcome in outcomes],
    }
