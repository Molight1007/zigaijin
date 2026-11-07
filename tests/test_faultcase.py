from __future__ import annotations

import json

import pytest

from evo.application.diagnose import DiagnosisComponent, parse_cases
from evo.application.faultcase import (
    FaultCaseError,
    build_case,
    cases_payload,
    read_failure,
)


def write_trial(
    root,
    trial="fix-git__abc",
    *,
    reward=0.0,
    exception=None,
    steps=3,
    message=None,
    test_stdout=None,
):
    trial_dir = root / trial
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    result = {
        "task_name": "terminal-bench/fix-git",
        "trial_name": trial,
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "exception_info": exception,
    }
    (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": "s1",
        "agent": {"name": "rrsi-terminus2"},
        "steps": [
            {
                "step_id": number,
                "timestamp": "2026-11-04T00:00:00+00:00",
                "source": "agent" if number % 2 else "user",
                "message": message if message is not None else f"step {number}",
            }
            for number in range(1, steps + 1)
        ],
        "final_metrics": {},
    }
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(trajectory), encoding="utf-8"
    )
    if test_stdout is not None:
        (trial_dir / "verifier" / "test-stdout.txt").write_text(
            test_stdout, encoding="utf-8"
        )
    return trial_dir


def test_read_failure_keeps_the_last_steps(tmp_path):
    trial = write_trial(tmp_path, steps=5)
    failure = read_failure(trial, max_steps=2)
    assert failure.trial_id == "fix-git__abc"
    assert failure.task_id == "terminal-bench/fix-git"
    assert failure.status == "completed"
    assert failure.reward == 0.0
    assert "step 4" in failure.trace
    assert "step 5" in failure.trace
    assert "step 1" not in failure.trace
    assert "[user]" in failure.trace or "[agent]" in failure.trace


def test_read_failure_includes_the_verifier_output(tmp_path):
    trial = write_trial(
        tmp_path, test_stdout="FAILED test_cancel.py::test_pending\nassert 1 == 0\n"
    )
    failure = read_failure(trial)
    assert "test_cancel.py::test_pending" in failure.trace
    assert "--- verifier ---" in failure.trace


def test_read_failure_truncates_from_the_front(tmp_path):
    trial = write_trial(tmp_path, steps=2, message="x" * 20_000)
    failure = read_failure(trial, max_chars=1_000)
    assert len(failure.trace) <= 1_100
    assert failure.trace.startswith("...[truncated]")
    assert "x" * 100 in failure.trace


def test_read_failure_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FaultCaseError, match="not a directory"):
        read_failure(tmp_path / "nope")


def test_read_failure_rejects_a_malformed_trajectory(tmp_path):
    trial = write_trial(tmp_path)
    (trial / "agent" / "trajectory.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FaultCaseError, match="trajectory"):
        read_failure(trial)


def test_read_failure_rejects_a_trajectory_without_steps(tmp_path):
    trial = write_trial(tmp_path)
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    with pytest.raises(FaultCaseError, match="steps"):
        read_failure(trial)


def test_read_failure_marks_timeouts_as_failed(tmp_path):
    trial = write_trial(
        tmp_path,
        reward=None,
        exception={"exception_type": "AgentTimeoutError", "exception_message": "late"},
    )
    failure = read_failure(trial)
    assert failure.status == "AgentTimeoutError"
    assert failure.failed


COMPONENTS = (
    DiagnosisComponent(id="terminus2:terminus_2", description="main loop", k_lines=900),
    DiagnosisComponent(id="terminus2:mechanisms", description="patches", k_lines=300),
)


def test_build_case_carries_the_injected_component(tmp_path):
    failure = read_failure(write_trial(tmp_path))
    case = build_case(
        "fault-1",
        failure,
        culprit="terminus2:mechanisms",
        components=COMPONENTS,
    )
    assert case.case_id == "fault-1"
    assert case.culprit == "terminus2:mechanisms"
    assert case.category == "harness"
    assert case.failure == failure.trace
    payload = cases_payload([case])
    parsed = parse_cases(payload)
    assert parsed[0].culprit == "terminus2:mechanisms"


def test_build_case_rejects_a_trial_that_did_not_fail(tmp_path):
    failure = read_failure(write_trial(tmp_path, reward=1.0))
    with pytest.raises(FaultCaseError, match="did not fail"):
        build_case("c", failure, culprit=COMPONENTS[0].id, components=COMPONENTS)


def test_build_case_rejects_an_unknown_culprit(tmp_path):
    failure = read_failure(write_trial(tmp_path))
    with pytest.raises(FaultCaseError, match="culprit"):
        build_case("c", failure, culprit="terminus2:nope", components=COMPONENTS)


def test_build_case_rejects_an_unknown_category(tmp_path):
    failure = read_failure(write_trial(tmp_path))
    with pytest.raises(FaultCaseError, match="category"):
        build_case(
            "c", failure, culprit=COMPONENTS[0].id, components=COMPONENTS, category="x"
        )
