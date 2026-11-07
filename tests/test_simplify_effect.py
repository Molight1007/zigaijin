from __future__ import annotations

from evo.application.baseline import TrialOutcome
from evo.application.simplify_effect import build_simplify_effect
from evo.contracts.types import KVector


TASKS = ("t1", "t2", "t3", "t4", "t5", "t6")
HOLDOUT = ("t5", "t6")


def outcomes(rewards, *, prefix: str = "r") -> list[TrialOutcome]:
    return [
        TrialOutcome(
            task_id=task,
            trial_id=f"{prefix}-{task}",
            status="completed",
            reward=rewards[task],
            tokens_in=100,
            tokens_out=10,
            cost_usd=0.001,
        )
        for task in TASKS
    ]


def k_before() -> KVector:
    return KVector(
        k_lines_nb=1000,
        k_ast=5000,
        k_chars=10_000,
        k_bytes=12_000,
        k_gzip=3000,
        k_cc=100,
    )


def k_after() -> KVector:
    return KVector(
        k_lines_nb=900,
        k_ast=4500,
        k_chars=9000,
        k_bytes=10_800,
        k_gzip=2700,
        k_cc=90,
    )


BATCH = (
    {
        "id": "c1",
        "layer": "L1",
        "delta": {
            "k_lines_nb": 60,
            "k_ast": 300,
            "k_chars": 600,
            "k_bytes": 700,
            "k_gzip": 200,
            "k_cc": 6,
        },
    },
    {
        "id": "c2",
        "layer": "L2",
        "delta": {
            "k_lines_nb": 40,
            "k_ast": 200,
            "k_chars": 400,
            "k_bytes": 500,
            "k_gzip": 100,
            "k_cc": 4,
        },
    },
)


def _effect(candidate=None, *, after=None, batch=BATCH):
    rewards = {task: 1.0 for task in TASKS}
    return build_simplify_effect(
        outcomes(rewards, prefix="ref"),
        outcomes(candidate or rewards, prefix="cand"),
        tasks=TASKS,
        holdout=HOLDOUT,
        k=1,
        k_before=k_before(),
        k_after=after or k_after(),
        batch=batch,
        margin=0.10,
        holdout_margin=0.10,
        n_boot=400,
        seed=7,
    )


def test_effective_simplification_reports_k_score_and_layers():
    effect = _effect()

    assert effect["verdict"] == "simplification_effective"
    assert effect["k_gate"]["status"] == "pass"
    assert effect["k"]["reduction_pct"]["k_ast"] == 10.0
    assert effect["score"]["non_inferior"] is True
    assert effect["score"]["holdout"]["non_inferior"] is True
    assert effect["layers"]["L1"]["count"] == 1
    assert effect["layers"]["L2"]["count"] == 1
    assert effect["layers"]["L1"]["delta_k_ast"] == 300
    assert effect["claim"].startswith("K_ast")


def test_score_regression_blocks_the_simplification_claim():
    candidate = {task: 1.0 for task in TASKS}
    candidate["t1"] = 0.0
    effect = _effect(candidate)

    assert effect["verdict"] == "not_supported"
    assert any("non-inferior" in reason for reason in effect["reasons"])


def test_insufficient_k_drop_blocks_the_simplification_claim():
    effect = _effect(after=k_before())

    assert effect["verdict"] == "not_supported"
    assert effect["k_gate"]["status"] == "fail"
    assert any("K" in reason for reason in effect["reasons"])


def test_diagnosis_before_after_is_reported_when_present():
    rewards = {task: 1.0 for task in TASKS}
    effect = build_simplify_effect(
        outcomes(rewards, prefix="ref"),
        outcomes(rewards, prefix="cand"),
        tasks=TASKS,
        holdout=HOLDOUT,
        k=1,
        k_before=k_before(),
        k_after=k_after(),
        batch=BATCH,
        diagnosis_before={"top1": 0.2, "top3": 0.5},
        diagnosis_after={"top1": 0.4, "top3": 0.7},
        n_boot=400,
        seed=7,
    )

    assert effect["diagnosis"]["delta_top1"] == 0.2
    assert effect["diagnosis"]["delta_top3"] == 0.2
    assert effect["diagnosis"]["improved"] is True
