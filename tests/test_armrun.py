from __future__ import annotations

import pytest

from evo.application.armrun import (
    ArmRunError,
    build_arm_payload,
    check_frozen_subject,
    plan_arm_run,
    subject_digest,
)
from evo.application.baseline import TrialOutcome
from evo.contracts.types import KVector


POOLS = {
    "smoke": ["terminal-bench/a", "terminal-bench/b"],
    "evolution": ["terminal-bench/c"],
    "holdout": ["terminal-bench/d"],
}


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def plan(tmp_path, *, arm="H0", pool="smoke", k_reps=2, k=None):
    return plan_arm_run(
        arm=arm,
        pool=pool,
        pools=POOLS,
        k_reps=k_reps,
        subject="terminus2",
        subject_root=tmp_path,
        k=k or KVector(k_lines_nb=10, k_ast=20, k_chars=100, k_bytes=100, k_gzip=50, k_cc=2),
    )


def test_subject_digest_is_stable_and_content_sensitive(tmp_path):
    write(tmp_path, "terminus2/a.py", "x = 1\n")
    write(tmp_path, "terminus2/templates/prompt.txt", "hello\n")
    write(tmp_path, "terminus2/__pycache__/a.cpython-311.pyc", "junk")
    first = subject_digest(tmp_path)
    assert subject_digest(tmp_path) == first
    write(tmp_path, "terminus2/a.py", "x = 2\n")
    second = subject_digest(tmp_path)
    assert second != first
    write(tmp_path, "terminus2/__pycache__/a.cpython-311.pyc", "other junk")
    assert subject_digest(tmp_path) == second


def test_subject_digest_rejects_a_missing_root(tmp_path):
    with pytest.raises(ArmRunError, match="not a directory"):
        subject_digest(tmp_path / "nope")


def test_plan_selects_the_requested_pool(tmp_path):
    result = plan(tmp_path)
    assert result.arm == "H0"
    assert result.pool == "smoke"
    assert result.task_ids == ("terminal-bench/a", "terminal-bench/b")
    assert result.k_reps == 2
    assert result.subject == "terminus2"
    assert result.subject_root == str(tmp_path.resolve())
    assert result.k.k_ast == 20


def test_plan_pool_all_concatenates_smoke_evolution_holdout(tmp_path):
    result = plan(tmp_path, pool="all")
    assert result.task_ids == (
        "terminal-bench/a",
        "terminal-bench/b",
        "terminal-bench/c",
        "terminal-bench/d",
    )


def test_plan_rejects_an_unknown_pool(tmp_path):
    with pytest.raises(ArmRunError, match="pool"):
        plan(tmp_path, pool="nope")


def test_plan_rejects_an_empty_pool(tmp_path):
    with pytest.raises(ArmRunError, match="empty"):
        plan_arm_run(
            arm="H0",
            pool="smoke",
            pools={"smoke": [], "evolution": [], "holdout": []},
            k_reps=1,
            subject="terminus2",
            subject_root=tmp_path,
            k=KVector(k_lines_nb=1),
        )


def test_plan_rejects_duplicate_tasks(tmp_path):
    with pytest.raises(ArmRunError, match="duplicate"):
        plan_arm_run(
            arm="H0",
            pool="all",
            pools={"smoke": ["t1"], "evolution": ["t1"], "holdout": []},
            k_reps=1,
            subject="terminus2",
            subject_root=tmp_path,
            k=KVector(k_lines_nb=1),
        )


def test_plan_rejects_a_bad_repeat_count(tmp_path):
    for value in (0, -1, True, 1.5):
        with pytest.raises(ArmRunError, match="k_reps"):
            plan(tmp_path, k_reps=value)


def test_plan_rejects_an_unknown_arm(tmp_path):
    with pytest.raises(ArmRunError, match="unknown arm"):
        plan(tmp_path, arm="H9")


def test_plan_rejects_a_missing_subject_root(tmp_path):
    with pytest.raises(ArmRunError, match="not a directory"):
        plan_arm_run(
            arm="H0",
            pool="smoke",
            pools=POOLS,
            k_reps=1,
            subject="terminus2",
            subject_root=tmp_path / "missing",
            k=KVector(k_lines_nb=1),
        )


def test_h0_subject_must_match_the_frozen_k():
    measured = KVector(k_lines_nb=10, k_ast=20)
    frozen = measured.as_dict()
    check_frozen_subject("H0", measured, frozen)
    with pytest.raises(ArmRunError, match="frozen"):
        check_frozen_subject("H0", KVector(k_lines_nb=9, k_ast=20), frozen)


def test_h0_check_requires_every_frozen_field():
    measured = KVector(k_lines_nb=10, k_ast=20)
    with pytest.raises(ArmRunError, match="frozen"):
        check_frozen_subject("H0", measured, {"k_lines_nb": 10})


def test_non_reference_arms_may_differ_from_the_frozen_k():
    check_frozen_subject("SIM", KVector(k_lines_nb=5), {"k_lines_nb": 10})


def test_h0_check_requires_a_frozen_k_vector():
    with pytest.raises(ArmRunError, match="frozen"):
        check_frozen_subject("H0", KVector(k_lines_nb=10), None)


def _outcomes():
    return [
        TrialOutcome(task_id="terminal-bench/a", trial_id="a1", reward=1.0, tokens_in=10, tokens_out=2),
        TrialOutcome(task_id="terminal-bench/a", trial_id="a2", reward=1.0, tokens_in=10, tokens_out=2),
        TrialOutcome(task_id="terminal-bench/b", trial_id="b1", reward=0.0, tokens_in=10, tokens_out=2),
        TrialOutcome(task_id="terminal-bench/b", trial_id="b2", reward=1.0, tokens_in=10, tokens_out=2),
    ]


def test_build_arm_payload_carries_the_baseline_report(tmp_path):
    result = plan(tmp_path)
    payload = build_arm_payload(result, _outcomes())
    assert payload["schema"] == 1
    assert payload["arm"] == "H0"
    assert payload["arm_config"]["simplify"] == "off"
    assert payload["task_ids"] == ["terminal-bench/a", "terminal-bench/b"]
    assert payload["k_reps"] == 2
    assert payload["subject_hash"] == result.subject_hash
    assert payload["k"]["k_lines_nb"] == 10
    assert payload["aggregate"]["mean_reward"] == pytest.approx(0.75)
    assert payload["decision"] == "baseline_valid"
    assert len(payload["outcomes"]) == 4
    assert payload["outcomes"][0]["trial_id"] == "a1"


def test_build_arm_payload_rejects_outcomes_outside_the_pool(tmp_path):
    with pytest.raises(ArmRunError, match="unknown task"):
        build_arm_payload(
            plan(tmp_path),
            [TrialOutcome(task_id="terminal-bench/zzz", trial_id="z1", reward=1.0)],
        )


def test_build_arm_payload_rejects_more_than_k_trials(tmp_path):
    extra = _outcomes() + [
        TrialOutcome(task_id="terminal-bench/a", trial_id="a3", reward=1.0)
    ]
    with pytest.raises(ArmRunError, match="exceeds"):
        build_arm_payload(plan(tmp_path), extra)
