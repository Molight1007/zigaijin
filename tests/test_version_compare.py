from __future__ import annotations

from evo.application.baseline import TrialOutcome
from evo.application.version_compare import build_version_comparison
from evo.contracts.types import KVector


TASKS = ("t1", "t2", "t3", "t4", "t5", "t6")
HOLDOUT = ("t5", "t6")


def outcomes(rewards, prefix: str):
    return [
        TrialOutcome(
            task_id=task,
            trial_id=f"{prefix}-{task}",
            status="completed",
            reward=rewards[task],
            tokens_in=100,
            tokens_out=10,
            cost_usd=0.01,
        )
        for task in TASKS
    ]


def rewards(values):
    return dict(zip(TASKS, values))


def test_version_compare_detects_improvement_over_previous():
    payload = build_version_comparison(
        {
            "H0": outcomes(rewards([1, 0, 1, 0, 1, 0]), "h0"),
            "previous": outcomes(rewards([1, 0, 1, 0, 1, 0]), "prev"),
            "candidate": outcomes(rewards([1, 1, 1, 1, 1, 0]), "cand"),
        },
        tasks=TASKS,
        holdout=HOLDOUT,
        k=1,
        n_boot=400,
        seed=7,
    )

    assert payload["per_task"]["t2"]["candidate"]["mean_reward"] == 1.0
    assert payload["per_task"]["t2"]["previous"]["mean_reward"] == 0.0
    pair = payload["comparisons"]["previous->candidate"]
    assert pair["wins"] == 2
    assert pair["losses"] == 0
    assert pair["mean_delta"] > 0
    assert payload["verdict"] == "candidate_improved"


def test_version_compare_flags_regression():
    payload = build_version_comparison(
        {
            "H0": outcomes(rewards([1, 0, 1, 0, 1, 0]), "h0"),
            "previous": outcomes(rewards([1, 0, 1, 0, 1, 0]), "prev"),
            "candidate": outcomes(rewards([0, 0, 1, 0, 1, 0]), "cand"),
        },
        tasks=TASKS,
        holdout=HOLDOUT,
        k=1,
        n_boot=400,
        seed=7,
    )

    assert payload["comparisons"]["previous->candidate"]["losses"] == 1
    assert payload["verdict"] == "candidate_regressed"


def test_version_compare_reports_k_reduction():
    payload = build_version_comparison(
        {
            "H0": outcomes(rewards([1, 0, 1, 0, 1, 0]), "h0"),
            "previous": outcomes(rewards([1, 0, 1, 0, 1, 0]), "prev"),
            "candidate": outcomes(rewards([1, 0, 1, 0, 1, 0]), "cand"),
        },
        tasks=TASKS,
        holdout=HOLDOUT,
        k=1,
        k_vectors={
            "previous": KVector(k_lines_nb=1000, k_ast=5000),
            "candidate": KVector(k_lines_nb=900, k_ast=4500),
        },
        n_boot=400,
        seed=7,
    )

    assert payload["k"]["reduction_pct"]["k_ast"] == 10.0
    assert payload["k"]["reduction_pct"]["k_lines_nb"] == 10.0


def test_version_compare_per_task_has_all_arms():
    payload = build_version_comparison(
        {
            "H0": outcomes(rewards([1, 0, 1, 0, 1, 0]), "h0"),
            "previous": outcomes(rewards([1, 0, 1, 0, 1, 0]), "prev"),
            "candidate": outcomes(rewards([1, 0, 1, 0, 1, 0]), "cand"),
        },
        tasks=TASKS,
        holdout=HOLDOUT,
        k=1,
        n_boot=400,
        seed=7,
    )

    row = payload["per_task"]["t1"]
    assert set(row) == {"H0", "previous", "candidate"}
    assert row["H0"]["mean_reward"] == 1.0
