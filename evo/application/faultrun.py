from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from evo.contracts.types import Component
from evo.domain.faults import FaultMutation


class FaultRunError(ValueError):
    """A fault plan cannot be executed against the given subject."""


@dataclass(frozen=True)
class FaultRunJob:
    index: int
    job_name: str
    mutation_id: str
    path: str
    line: int
    operator: str
    component: str
    workdir: str
    subject_root: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "job_name": self.job_name,
            "mutation_id": self.mutation_id,
            "path": self.path,
            "line": self.line,
            "operator": self.operator,
            "component": self.component,
            "workdir": self.workdir,
            "subject_root": self.subject_root,
        }


def _component_for(components: Sequence[Component], path: str) -> str:
    matches = [component.id for component in components if path in component.paths]
    if not matches:
        raise FaultRunError(
            f"mutated path {path!r} is not covered by any component"
        )
    if len(matches) > 1:
        raise FaultRunError(
            f"mutated path {path!r} belongs to several components: {', '.join(matches)}"
        )
    return matches[0]


def plan_fault_runs(
    mutations: Sequence[FaultMutation],
    *,
    subject_root: str | Path,
    work_root: str | Path,
    job_prefix: str,
    components: Sequence[Component],
) -> tuple[FaultRunJob, ...]:
    if not mutations:
        raise FaultRunError("fault plan is empty")
    base = Path(subject_root)
    if not base.is_dir():
        raise FaultRunError(f"subject root is not a directory: {subject_root}")
    if not base.name.isidentifier():
        raise FaultRunError(
            f"subject directory must be a valid module name: {base.name!r}"
        )
    if (
        not isinstance(job_prefix, str)
        or not job_prefix.strip()
        or "/" in job_prefix
        or "\\" in job_prefix
    ):
        raise FaultRunError(f"job_prefix must be a plain name: {job_prefix!r}")
    work = Path(work_root)
    jobs = []
    for index, mutation in enumerate(mutations, start=1):
        if not isinstance(mutation, FaultMutation):
            raise FaultRunError(f"mutation must be a FaultMutation: {mutation!r}")
        directory = (work / f"{index:03d}").resolve()
        jobs.append(
            FaultRunJob(
                index=index,
                job_name=f"{job_prefix}-{index:03d}",
                mutation_id=mutation.id,
                path=mutation.path,
                line=mutation.line,
                operator=mutation.operator,
                component=_component_for(components, mutation.path),
                workdir=str(directory),
                subject_root=str(directory / base.name),
            )
        )
    return tuple(jobs)
