from __future__ import annotations

import json

import pytest

from evo.application.runstate import (
    RunStateError,
    advance_run,
    init_run,
    load_state,
)
from evo.domain.state import PHASES, RunState, StateError, advance


AT = "2026-01-01T00:00:00+00:00"


def start():
    return RunState.initial("r1", at=AT)


def step(state, to, **kwargs):
    kwargs.setdefault("at", AT)
    return advance(state, to, **kwargs)


def test_initial_state_is_created():
    state = start()
    assert state.phase == "created"
    assert state.run_id == "r1"
    assert state.resume is None


def test_happy_path_requires_an_artifact_before_done():
    state = step(start(), "prestudy_pending")
    state = step(state, "prestudy_running")
    with pytest.raises(StateError, match="artifact"):
        step(state, "prestudy_done")
    state = step(state, "prestudy_done", artifact="prestudy.json")
    state = step(state, "baseline_pending")
    assert state.phase == "baseline_pending"


def test_report_done_is_terminal():
    state = start()
    for phase in (
        "prestudy_pending",
        "prestudy_running",
        "prestudy_done",
        "baseline_pending",
        "baseline_running",
        "baseline_done",
        "arm_pending",
        "arm_running",
        "arm_done",
        "diagnosis_pending",
        "diagnosis_running",
        "diagnosis_done",
        "report_pending",
        "report_running",
    ):
        artifact = phase if phase.endswith("_done") else None
        state = step(state, phase, artifact=artifact)
    state = step(state, "report_done", artifact="tables/t1.csv")
    assert state.phase == "report_done"
    with pytest.raises(StateError, match="terminal"):
        step(state, "blocked_docker")


def test_invalid_edges_are_rejected():
    with pytest.raises(StateError, match="expected"):
        step(start(), "baseline_pending")
    state = step(start(), "prestudy_pending")
    with pytest.raises(StateError, match="expected"):
        step(state, "prestudy_done", artifact="x")


def test_blocked_docker_resumes_automatically():
    state = step(start(), "prestudy_pending")
    state = step(state, "prestudy_running")
    blocked = step(state, "blocked_docker")
    assert blocked.resume == "prestudy_running"
    resumed = step(blocked, "prestudy_running")
    assert resumed.phase == "prestudy_running"
    assert resumed.resume is None


def test_crashed_resumes_automatically():
    state = step(start(), "prestudy_pending")
    crashed = step(state, "crashed")
    assert crashed.resume == "prestudy_pending"
    assert step(crashed, "prestudy_pending").resume is None


def test_blocked_subject_needs_a_human():
    state = step(start(), "prestudy_pending")
    blocked = step(state, "blocked_subject")
    assert blocked.resume == "prestudy_pending"
    with pytest.raises(StateError, match="human"):
        step(blocked, "prestudy_pending", actor="runner")
    resumed = step(blocked, "prestudy_pending", actor="human")
    assert resumed.phase == "prestudy_pending"


def test_invalid_measurement_needs_a_human():
    state = step(start(), "prestudy_pending")
    blocked = step(state, "invalid_measurement")
    with pytest.raises(StateError, match="human"):
        step(blocked, "prestudy_pending")
    assert step(blocked, "prestudy_pending", actor="human").phase == "prestudy_pending"


def test_resume_must_target_the_recorded_phase():
    state = step(start(), "prestudy_pending")
    blocked = step(state, "blocked_docker")
    with pytest.raises(StateError, match="resume"):
        step(blocked, "baseline_pending")


def test_state_round_trips_json_and_rejects_unknown_phases():
    state = step(start(), "prestudy_pending")
    assert RunState.from_dict(state.as_dict()) == state
    with pytest.raises(StateError, match="phase"):
        RunState.from_dict({"run_id": "r1", "phase": "nope"})


def test_actor_is_validated():
    state = step(start(), "prestudy_pending")
    with pytest.raises(StateError, match="actor"):
        step(state, "blocked_docker", actor="robot")


def test_phase_list_covers_every_transition_target():
    assert "created" in PHASES
    assert "report_done" in PHASES
    assert len(PHASES) == len(set(PHASES))


def test_init_run_writes_state_and_is_idempotent(tmp_path):
    state = init_run(tmp_path, "r1", at=AT)
    assert state.phase == "created"
    assert load_state(tmp_path) == state
    assert init_run(tmp_path, "r1", at=AT) == state
    with pytest.raises(RunStateError, match="run"):
        init_run(tmp_path, "r2", at=AT)


def test_advance_run_requires_the_artifact_to_exist(tmp_path):
    init_run(tmp_path, "r1", at=AT)
    advance_run(tmp_path, "prestudy_pending", at=AT)
    advance_run(tmp_path, "prestudy_running", at=AT)
    with pytest.raises(RunStateError, match="artifact"):
        advance_run(tmp_path, "prestudy_done", artifact="prestudy.json", at=AT)
    (tmp_path / "prestudy.json").write_text("{}", encoding="utf-8")
    state, event = advance_run(
        tmp_path, "prestudy_done", artifact="prestudy.json", at=AT
    )
    assert state.phase == "prestudy_done"
    assert event["from"] == "prestudy_running"
    assert event["to"] == "prestudy_done"
    assert event["artifact"] == "prestudy.json"
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1] == event
    assert load_state(tmp_path) == state


def test_advance_run_rejects_an_artifact_outside_the_root(tmp_path):
    init_run(tmp_path, "r1", at=AT)
    advance_run(tmp_path, "prestudy_pending", at=AT)
    advance_run(tmp_path, "prestudy_running", at=AT)
    outside = tmp_path.parent / "prestudy.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(RunStateError, match="artifact"):
        advance_run(tmp_path, "prestudy_done", artifact="../prestudy.json", at=AT)


def test_load_state_returns_none_when_missing(tmp_path):
    assert load_state(tmp_path / "missing") is None


def test_advance_run_needs_an_existing_state(tmp_path):
    with pytest.raises(RunStateError, match="state"):
        advance_run(tmp_path, "prestudy_pending", at=AT)
