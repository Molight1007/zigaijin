from __future__ import annotations

import math

import pytest

from evo.domain.stats import (
    StatsError,
    bootstrap_mean_ci,
    mde_paired,
    paired_bootstrap_diff,
    per_task_worst,
    required_pairs,
    tost_paired,
    worst_case_ci_halfwidth,
)


def test_bootstrap_ci_of_constant_sample_is_degenerate():
    interval = bootstrap_mean_ci([0.5] * 20)
    assert interval.estimate == 0.5
    assert interval.low == interval.high == 0.5


def test_bootstrap_is_deterministic_for_a_seed():
    values = [0, 1, 0, 1, 1, 0, 0, 1]
    first = bootstrap_mean_ci(values, seed=7)
    second = bootstrap_mean_ci(values, seed=7)
    assert first == second


def test_paired_diff_is_candidate_minus_reference():
    reference = [0.2, 0.4, 0.6]
    candidate = [0.3, 0.4, 0.5]
    interval = paired_bootstrap_diff(reference, candidate)
    assert interval.estimate == pytest.approx(0.0)
    assert interval.low <= interval.estimate <= interval.high


def test_paired_diff_ci_brackets_known_effect():
    reference = [0.0] * 30
    candidate = [0.1] * 30
    interval = paired_bootstrap_diff(reference, candidate)
    assert interval.estimate == pytest.approx(0.1)
    assert interval.low == pytest.approx(0.1)
    assert interval.high == pytest.approx(0.1)


def test_tost_accepts_equivalent_arms():
    result = tost_paired([0.5] * 40, [0.5] * 40, margin=0.1)
    assert result.equivalent
    assert result.p_value < 0.05
    assert result.diff == 0.0


def test_tost_rejects_clear_degradation():
    result = tost_paired([0.8] * 40, [0.5] * 40, margin=0.1)
    assert not result.equivalent
    assert result.diff == pytest.approx(-0.3)


def test_tost_rejects_clear_improvement_against_upper_bound():
    result = tost_paired([0.5] * 40, [0.8] * 40, margin=0.1)
    assert not result.equivalent
    assert result.diff == pytest.approx(0.3)


def test_tost_requires_positive_margin():
    with pytest.raises(StatsError):
        tost_paired([0.5], [0.5], margin=0)


def test_mde_grows_with_variance_and_shrinks_with_n():
    low_noise = mde_paired([0.0, 0.01, -0.01, 0.0] * 10)
    high_noise = mde_paired([0.0, 0.5, -0.5, 0.0] * 10)
    assert high_noise > low_noise
    assert mde_paired([0.0] * 100) == 0.0


def test_mde_requires_two_observations():
    assert math.isinf(mde_paired([0.1]))


def test_required_pairs_is_monotone_in_margin():
    assert required_pairs(0.05, 0.5) > required_pairs(0.1, 0.5)
    assert required_pairs(0.1, 0.0) == 1


def test_worst_case_halfwidth_matches_known_value():
    assert worst_case_ci_halfwidth(180) == pytest.approx(0.98 / math.sqrt(180), rel=0.02)


def test_per_task_worst_returns_minimum_and_task():
    assert per_task_worst({"t1": 0.0, "t2": -0.3, "t3": 0.2}) == ("t2", -0.3)
    with pytest.raises(StatsError):
        per_task_worst({})


def test_length_mismatch_and_small_bootstrap_are_rejected():
    with pytest.raises(StatsError):
        paired_bootstrap_diff([0.1], [0.1, 0.2])
    with pytest.raises(StatsError):
        bootstrap_mean_ci([0.1], n_boot=10)


def test_holm_controls_the_family_with_a_step_down():
    from evo.domain.stats import holm

    result = holm([0.01, 0.04, 0.03])
    assert result["adjusted"] == pytest.approx([0.03, 0.06, 0.06])
    assert result["reject"] == [True, False, False]
    assert result["method"] == "holm"


def test_holm_rejects_bad_input():
    from evo.domain.stats import holm

    with pytest.raises(StatsError):
        holm([])
    with pytest.raises(StatsError):
        holm([0.5, 1.2])
    with pytest.raises(StatsError):
        holm([0.01], alpha=0.0)


def test_bonferroni_alpha_shrinks_with_family_size():
    from evo.domain.stats import bonferroni_alpha

    assert bonferroni_alpha(0.05, 5) == pytest.approx(0.01)
    assert bonferroni_alpha(0.05, 1) == pytest.approx(0.05)
    with pytest.raises(StatsError):
        bonferroni_alpha(0.05, 0)
