from __future__ import annotations

import pytest

from evo.application.baseline import TrialOutcome
from evo.application.compare import CompareError, compare_arms
from evo.contracts.types import KVector


TASKS = ("t1", "t2", "t3", "t4", "t5", "t6")
REWARDS = {"t1": 1.0, "t2": 0.0, "t3": 1.0, "t4": 0.0, "t5": 1.0, "t6": 0.0}


def outcomes(rewards, *, prefix="r", tokens_in=100, tokens_out=10):
    return [
        TrialOutcome(
            task_id=task,
            trial_id=f"{prefix}-{task}",
            status="completed",
            reward=rewards[task],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.001,
        )
        for task in TASKS
    ]


def compare(reference=None, candidate=None, **overrides):
    kwargs = {
        "tasks": TASKS,
        "k": 1,
        "margin": 0.10,
        "holdout_margin": 0.10,
        "n_boot": 400,
        "seed": 7,
    }
    kwargs.update(overrides)
    return compare_arms(
        reference if reference is not None else outcomes(REWARDS, prefix="ref"),
        candidate if candidate is not None else outcomes(REWARDS, prefix="cand"),
        **kwargs,
    )


def test_identical_arms_are_non_inferior():
    payload = compare()
    aggregate = payload["aggregate"]
    assert aggregate["n_pairs"] == 6
    assert aggregate["mean_delta"] == 0.0
    assert aggregate["ci"]["low"] <= 0.0 <= aggregate["ci"]["high"]
    assert aggregate["non_inferior"] is True
    assert aggregate["worst_task"]["delta"] == 0.0
    assert payload["tasks"]["t1"] == {
        "reference": 1.0,
        "candidate": 1.0,
        "delta": 0.0,
        "holdout": False,
    }


def test_a_large_task_regression_fails_non_inferiority():
    candidate = dict(REWARDS)
    candidate["t1"] = 0.0
    payload = compare(candidate=outcomes(candidate, prefix="cand"))
    aggregate = payload["aggregate"]
    assert aggregate["mean_delta"] == pytest.approx(-1 / 6)
    assert aggregate["ci"]["low"] < -0.10
    assert aggregate["non_inferior"] is False
    assert aggregate["worst_task"] == {"task": "t1", "delta": -1.0}


def test_holdout_is_scored_separately():
    candidate = dict(REWARDS)
    candidate["t5"] = 0.0
    payload = compare(
        candidate=outcomes(candidate, prefix="cand"),
        holdout=("t5", "t6"),
    )
    holdout = payload["aggregate"]["holdout"]
    assert holdout["n_pairs"] == 2
    assert holdout["mean_delta"] == pytest.approx(-0.5)
    assert holdout["non_inferior"] is False
    assert payload["tasks"]["t5"]["holdout"] is True
    assert payload["tasks"]["t1"]["holdout"] is False


def test_tokens_and_cost_are_summed_per_arm():
    reference = outcomes(REWARDS, prefix="ref", tokens_in=100)
    candidate = outcomes(REWARDS, prefix="cand", tokens_in=120)
    payload = compare(reference=reference, candidate=candidate)
    tokens = payload["aggregate"]["tokens"]
    assert tokens["reference"]["in"] == 600
    assert tokens["candidate"]["in"] == 720
    assert tokens["candidate"]["total"] == 780
    assert tokens["candidate"]["ratio"] == pytest.approx(780 / 660)
    assert payload["aggregate"]["cost"]["candidate"] == pytest.approx(0.006)


def test_k_vectors_are_compared_when_given():
    before = KVector(
        k_lines_nb=200, k_ast=400, k_chars=1000, k_bytes=1000, k_gzip=100, k_cc=10
    )
    after = KVector(
        k_lines_nb=180, k_ast=360, k_chars=900, k_bytes=900, k_gzip=90, k_cc=9
    )
    payload = compare(k_before=before, k_after=after)
    k = payload["k"]
    assert k["before"]["k_lines_nb"] == 200
    assert k["after"]["k_lines_nb"] == 180
    assert k["delta"]["k_lines_nb"] == 20
    assert k["reduction_pct"]["k_lines_nb"] == 10.0
    assert k["reduction_pct"]["k_ast"] == 10.0


def test_missing_paired_rewards_are_refused():
    reference = [row for row in outcomes(REWARDS, prefix="ref") if row.task_id != "t2"]
    with pytest.raises(CompareError, match="t2"):
        compare(reference=reference)


def test_infra_error_is_scored_as_zero_in_paired_comparison():
    reference = outcomes(REWARDS, prefix="ref")
    reference[1] = TrialOutcome(
        task_id="t2",
        trial_id="ref-t2",
        status="infra_error",
        reward=None,
    )
    payload = compare(reference=reference)
    assert payload["tasks"]["t2"]["reference"] == 0.0


def test_holdout_must_be_a_subset_of_the_tasks():
    with pytest.raises(CompareError, match="holdout"):
        compare(holdout=("t7",))


def test_duplicate_tasks_are_refused():
    with pytest.raises(CompareError, match="duplicate"):
        compare(tasks=("t1", "t1"))


def test_margin_must_be_a_fraction():
    with pytest.raises(CompareError, match="margin"):
        compare(margin=1.5)


def test_k_vectors_must_come_in_pairs():
    before = KVector(
        k_lines_nb=200, k_ast=400, k_chars=1000, k_bytes=1000, k_gzip=100, k_cc=10
    )
    with pytest.raises(CompareError, match="k_before"):
        compare(k_before=before)


def test_family_size_adjusts_the_confidence_alpha():
    payload = compare(family_size=5)
    assert payload["aggregate"]["family"] == {
        "size": 5,
        "alpha": pytest.approx(0.01),
        "method": "bonferroni",
    }
    assert payload["aggregate"]["ci"]["alpha"] == pytest.approx(0.01)


def test_family_size_is_recorded_even_when_unadjusted():
    payload = compare()
    assert payload["aggregate"]["family"] == {
        "size": 1,
        "alpha": pytest.approx(0.05),
        "method": "none",
    }
