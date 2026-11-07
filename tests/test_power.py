from __future__ import annotations

import pytest

from evo.application.baseline import TrialOutcome
from evo.application.power import PowerError, power_report


TASKS = ("t1", "t2", "t3", "t4")


def outcomes(rewards, *, prefix):
    return [
        TrialOutcome(
            task_id=task,
            trial_id=f"{prefix}-{task}",
            reward=rewards[task],
            tokens_in=90,
            tokens_out=10,
            cost_usd=0.001,
        )
        for task in rewards
    ]


def test_power_report_summarizes_mde_and_required_pairs():
    reference = outcomes({task: 0.5 for task in TASKS}, prefix="ref")
    candidate = outcomes(
        dict(zip(TASKS, (0.8, 0.6, 0.5, 0.4))), prefix="cand"
    )
    report = power_report(reference, candidate, tasks=TASKS, k=1, margin=0.05)

    assert report["n_pairs"] == 4
    assert report["mean_delta"] == pytest.approx(0.075)
    assert report["sd"] == pytest.approx(0.1708, rel=0.01)
    assert report["mde"] == pytest.approx(0.2391, rel=0.01)
    assert report["required_pairs"] == 92
    assert report["enough"] is False
    assert report["margin"] == pytest.approx(0.05)


def test_a_noise_free_gap_needs_one_pair():
    reference = outcomes({task: 0.5 for task in TASKS}, prefix="ref")
    candidate = outcomes({task: 0.6 for task in TASKS}, prefix="cand")
    report = power_report(reference, candidate, tasks=TASKS, k=1, margin=0.05)

    assert report["sd"] == 0.0
    assert report["required_pairs"] == 1
    assert report["enough"] is True


def test_power_requires_paired_coverage():
    reference = outcomes({task: 0.5 for task in TASKS}, prefix="ref")
    candidate = outcomes({task: 0.6 for task in TASKS[:2]}, prefix="cand")
    with pytest.raises(PowerError, match="missing"):
        power_report(reference, candidate, tasks=TASKS, k=1, margin=0.05)
