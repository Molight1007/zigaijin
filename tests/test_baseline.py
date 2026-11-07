import pytest

from evo.application.baseline import BaselineError, TrialOutcome, build_baseline
from evo.contracts.types import KVector


K = KVector(
    k_lines_nb=100,
    k_ast=200,
    k_chars=1000,
    k_bytes=4000,
    k_gzip=1000,
    k_cc=10,
)


def outcome(
    task_id="t0",
    trial_id="r0",
    status="completed",
    reward=1.0,
    tokens_in=100,
    tokens_out=10,
    cost_usd=0.01,
    failure_cause=None,
):
    return TrialOutcome(
        task_id=task_id,
        trial_id=trial_id,
        status=status,
        reward=reward,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        failure_cause=failure_cause,
    )


def test_rejects_unknown_status():
    with pytest.raises(BaselineError, match="status"):
        outcome(status="timeout")


def test_completed_trial_requires_a_reward():
    with pytest.raises(BaselineError, match="reward"):
        outcome(reward=None)


def test_error_trial_may_omit_reward_and_tokens():
    record = outcome(status="agent_error", reward=None, tokens_in=None, tokens_out=None)
    assert record.status == "agent_error"


def test_rejects_negative_tokens():
    with pytest.raises(BaselineError, match="tokens_in"):
        outcome(tokens_in=-1)


def test_rejects_non_finite_reward():
    with pytest.raises(BaselineError, match="reward"):
        outcome(reward=float("nan"))


def test_rejects_empty_task_id():
    with pytest.raises(BaselineError, match="task_id"):
        outcome(task_id="")


def test_rejects_unknown_task():
    with pytest.raises(BaselineError, match="unknown task"):
        build_baseline([outcome(task_id="t9")], tasks=["t0"], k=1, k_vector=K)


def test_rejects_duplicate_trial_ids():
    with pytest.raises(BaselineError, match="duplicate"):
        build_baseline(
            [outcome(trial_id="r0"), outcome(trial_id="r0")],
            tasks=["t0"],
            k=2,
            k_vector=K,
        )


def test_rejects_more_trials_than_the_protocol_allows():
    with pytest.raises(BaselineError, match="exceeds"):
        build_baseline(
            [outcome(trial_id="r0"), outcome(trial_id="r1")],
            tasks=["t0"],
            k=1,
            k_vector=K,
        )


def test_rejects_non_positive_k():
    with pytest.raises(BaselineError, match="k must be"):
        build_baseline([], tasks=["t0"], k=0, k_vector=K)


def test_aggregates_per_task_and_equal_weight_mean():
    outcomes = [
        outcome(task_id="t0", trial_id="r0", reward=1.0),
        outcome(task_id="t0", trial_id="r1", reward=0.0),
        outcome(task_id="t1", trial_id="r2", reward=0.5),
        outcome(task_id="t1", trial_id="r3", reward=0.5),
    ]
    report = build_baseline(outcomes, tasks=["t0", "t1"], k=2, k_vector=K)
    assert report["tasks"]["t0"]["mean_reward"] == pytest.approx(0.5)
    assert report["tasks"]["t1"]["mean_reward"] == pytest.approx(0.5)
    assert report["aggregate"]["mean_reward"] == pytest.approx(0.5)
    assert report["aggregate"]["valid_trials"] == 4
    assert report["decision"] == "baseline_valid"


def test_task_means_are_weighted_equally_not_by_trial_count():
    outcomes = [
        outcome(task_id="t0", trial_id="r0", reward=1.0),
        outcome(task_id="t0", trial_id="r1", reward=1.0),
        outcome(task_id="t1", trial_id="r2", reward=0.0),
    ]
    report = build_baseline(outcomes, tasks=["t0", "t1"], k=2, k_vector=K)
    assert report["aggregate"]["mean_reward"] == pytest.approx(0.5)
    assert report["decision"] == "baseline_invalid"


def test_missing_task_counts_against_coverage():
    report = build_baseline(
        [outcome(task_id="t0", trial_id="r0")],
        tasks=["t0", "t1"],
        k=2,
        k_vector=K,
    )
    validity = report["validity"]
    assert validity["missing_frac"] == pytest.approx(0.75)
    assert validity["complete"] is False
    assert report["decision"] == "baseline_invalid"
    assert report["aggregate"]["tasks_with_valid_trials"] == 1


def test_infra_errors_are_counted_separately():
    outcomes = [
        outcome(task_id="t0", trial_id="r0"),
        outcome(task_id="t0", trial_id="r1", status="infra_error", reward=None),
        outcome(task_id="t1", trial_id="r2"),
        outcome(task_id="t1", trial_id="r3"),
    ]
    report = build_baseline(outcomes, tasks=["t0", "t1"], k=2, k_vector=K)
    validity = report["validity"]
    assert validity["missing_frac"] == pytest.approx(0.0)
    assert validity["infra_frac"] == pytest.approx(0.25)
    assert validity["complete"] is True
    assert report["decision"] == "baseline_invalid"


def test_token_coverage_counts_reported_trials():
    outcomes = [
        outcome(task_id="t0", trial_id="r0", tokens_in=None, tokens_out=None),
        outcome(task_id="t0", trial_id="r1"),
        outcome(task_id="t1", trial_id="r2"),
        outcome(task_id="t1", trial_id="r3"),
    ]
    report = build_baseline(outcomes, tasks=["t0", "t1"], k=2, k_vector=K)
    assert report["validity"]["token_coverage"] == pytest.approx(0.75)


def test_tokens_complete_is_reported_per_baseline():
    report = build_baseline(
        [
            outcome(task_id="t0", trial_id="r0", tokens_in=None, tokens_out=None),
            outcome(task_id="t0", trial_id="r1"),
        ],
        tasks=["t0"],
        k=2,
        k_vector=K,
    )
    assert report["aggregate"]["tokens_complete"] is False


def test_zero_valid_trials_score_zero_not_null():
    report = build_baseline(
        [
            outcome(task_id="t0", trial_id="r0", status="agent_error", reward=None),
            outcome(task_id="t0", trial_id="r1", status="agent_error", reward=None),
        ],
        tasks=["t0"],
        k=2,
        k_vector=K,
    )
    assert report["tasks"]["t0"]["mean_reward"] == pytest.approx(0.0)
    assert report["tasks"]["t0"]["valid_mean_reward"] is None
    assert report["aggregate"]["mean_reward"] == pytest.approx(0.0)
    assert report["aggregate"]["valid_mean_reward"] is None
    assert report["validity"]["token_coverage"] == 0.0
    assert report["decision"] == "baseline_invalid"


def test_strict_pass_at_1_counts_agent_errors_as_zero():
    outcomes = [
        outcome(task_id="t0", trial_id="r0", reward=1.0),
        outcome(task_id="t0", trial_id="r1", status="agent_error", reward=None),
    ]
    report = build_baseline(outcomes, tasks=["t0"], k=2, k_vector=K)
    task = report["tasks"]["t0"]
    assert task["mean_reward"] == pytest.approx(0.5)
    assert task["valid_mean_reward"] == pytest.approx(1.0)
    assert task["agent_errors"] == 1
    assert report["validity"]["agent_fail_frac"] == pytest.approx(0.5)
    assert report["validity"]["missing_frac"] == pytest.approx(0.0)


def test_baseline_with_half_agent_errors_is_invalid():
    report = build_baseline(
        [
            outcome(task_id="t0", trial_id="r0", reward=1.0),
            outcome(task_id="t0", trial_id="r1", status="agent_error", reward=None),
        ],
        tasks=["t0"],
        k=2,
        k_vector=K,
    )
    assert report["validity"]["agent_fail_frac"] == pytest.approx(0.5)
    assert report["decision"] == "baseline_invalid"


def test_strict_pass_at_1_counts_infra_errors_as_zero():
    outcomes = [
        outcome(task_id="t0", trial_id="r0", reward=1.0),
        outcome(task_id="t0", trial_id="r1", status="infra_error", reward=None),
    ]
    report = build_baseline(outcomes, tasks=["t0"], k=2, k_vector=K)
    task = report["tasks"]["t0"]
    assert task["mean_reward"] == pytest.approx(0.5)
    assert task["valid_mean_reward"] == pytest.approx(1.0)
    assert report["validity"]["infra_frac"] == pytest.approx(0.5)
    assert report["validity"]["missing_frac"] == pytest.approx(0.0)


def test_strict_pass_at_1_counts_missing_records_as_zero():
    report = build_baseline(
        [outcome(task_id="t0", trial_id="r0", reward=1.0)],
        tasks=["t0"],
        k=2,
        k_vector=K,
    )
    task = report["tasks"]["t0"]
    assert task["mean_reward"] == pytest.approx(0.5)
    assert task["valid_mean_reward"] == pytest.approx(1.0)
    assert task["missing_trials"] == 1
    assert report["validity"]["missing_frac"] == pytest.approx(0.5)
    assert report["validity"]["complete"] is False


def test_token_and_cost_totals_only_count_valid_trials():
    outcomes = [
        outcome(task_id="t0", trial_id="r0", tokens_in=100, tokens_out=10, cost_usd=0.01),
        outcome(task_id="t0", trial_id="r1", status="infra_error", reward=None, tokens_in=None, tokens_out=None, cost_usd=None),
    ]
    report = build_baseline(outcomes, tasks=["t0"], k=2, k_vector=K)
    task = report["tasks"]["t0"]
    assert task["tokens_in"] == 100
    assert task["tokens_out"] == 10
    assert task["cost_usd"] == pytest.approx(0.01)
    assert task["trials_with_tokens"] == 1


def test_missing_cost_marks_the_baseline_incomplete():
    report = build_baseline(
        [
            outcome(task_id="t0", trial_id="r0", cost_usd=None),
            outcome(task_id="t0", trial_id="r1", cost_usd=0.01),
        ],
        tasks=["t0"],
        k=2,
        k_vector=K,
    )
    assert report["tasks"]["t0"]["cost_complete"] is False
    assert report["aggregate"]["cost_complete"] is False


def test_complete_costs_are_reported_as_complete():
    report = build_baseline(
        [outcome(task_id="t0", trial_id="r0"), outcome(task_id="t0", trial_id="r1")],
        tasks=["t0"],
        k=2,
        k_vector=K,
    )
    assert report["tasks"]["t0"]["cost_complete"] is True
    assert report["aggregate"]["cost_complete"] is True


def test_report_carries_k_vector_and_repetition_count():
    report = build_baseline([outcome()], tasks=["t0"], k=1, k_vector=K)
    assert report["k"] == K.as_dict()
    assert report["k_reps"] == 1


def test_trial_order_does_not_change_the_result():
    first = [
        outcome(task_id="t0", trial_id="r0", reward=1.0),
        outcome(task_id="t1", trial_id="r1", reward=0.0),
    ]
    second = list(reversed(first))
    report_a = build_baseline(first, tasks=["t0", "t1"], k=1, k_vector=K)
    report_b = build_baseline(second, tasks=["t0", "t1"], k=1, k_vector=K)
    assert report_a == report_b
