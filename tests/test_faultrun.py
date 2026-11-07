from __future__ import annotations

import pytest

from evo.application.faultrun import FaultRunError, plan_fault_runs
from evo.contracts.types import Component
from evo.domain.faults import FaultEdit, FaultMutation


COMPONENTS = [
    Component(id="terminus2:agent", name="agent", paths=("agent.py",)),
    Component(id="terminus2:mech", name="mech", paths=("mechanisms.py",)),
]


def mutation(path="agent.py", identifier="m1"):
    return FaultMutation(
        id=identifier,
        path=path,
        line=1,
        operator="flip_bool",
        symbol="FLAG",
        edits=(FaultEdit(start=0, end=1, before="x", after="y"),),
    )


def test_plan_maps_each_mutation_to_its_component(tmp_path):
    subject = tmp_path / "terminus2"
    subject.mkdir()
    jobs = plan_fault_runs(
        [mutation(), mutation("mechanisms.py", "m2")],
        subject_root=subject,
        work_root=tmp_path / "work",
        job_prefix="fault-a",
        components=COMPONENTS,
    )
    assert [job.job_name for job in jobs] == ["fault-a-001", "fault-a-002"]
    assert [job.component for job in jobs] == ["terminus2:agent", "terminus2:mech"]
    assert jobs[0].workdir == str((tmp_path / "work" / "001").resolve())
    assert jobs[0].subject_root == str(
        (tmp_path / "work" / "001" / "terminus2").resolve()
    )
    assert jobs[0].mutation_id == "m1"


def test_plan_rejects_a_mutation_outside_every_component(tmp_path):
    subject = tmp_path / "terminus2"
    subject.mkdir()
    with pytest.raises(FaultRunError, match="component"):
        plan_fault_runs(
            [mutation("other.py")],
            subject_root=subject,
            work_root=tmp_path / "work",
            job_prefix="fault-a",
            components=COMPONENTS,
        )


def test_plan_rejects_a_bad_job_prefix_and_empty_plan(tmp_path):
    subject = tmp_path / "terminus2"
    subject.mkdir()
    with pytest.raises(FaultRunError, match="job_prefix"):
        plan_fault_runs(
            [mutation()],
            subject_root=subject,
            work_root=tmp_path / "work",
            job_prefix="a/b",
            components=COMPONENTS,
        )
    with pytest.raises(FaultRunError, match="empty"):
        plan_fault_runs(
            [],
            subject_root=subject,
            work_root=tmp_path / "work",
            job_prefix="fault-a",
            components=COMPONENTS,
        )


def test_plan_rejects_a_subject_directory_that_is_not_a_module(tmp_path):
    subject = tmp_path / "not-a-module"
    subject.mkdir()
    with pytest.raises(FaultRunError, match="module"):
        plan_fault_runs(
            [mutation()],
            subject_root=subject,
            work_root=tmp_path / "work",
            job_prefix="fault-a",
            components=COMPONENTS,
        )
