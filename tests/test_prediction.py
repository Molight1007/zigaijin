from __future__ import annotations

import pytest

from evo.domain.prediction import PredictionError, verify_prediction


def test_all_predicted_tasks_improving_is_verified():
    result = verify_prediction(
        ["t1", "t2"],
        {"t1": 0.0, "t2": 0.5},
        {"t1": 1.0, "t2": 0.6},
        k=1,
    )
    assert result["prediction_verified"] is True
    assert result["prediction_falsified"] is False
    assert result["hit_rate"] == 1.0
    assert result["predicted_miss"] == []


def test_one_predicted_task_missing_falsifies_the_prediction():
    result = verify_prediction(
        ["t1", "t2"],
        {"t1": 0.0, "t2": 0.5},
        {"t1": 1.0, "t2": 0.5},
        k=2,
    )
    assert result["prediction_verified"] is False
    assert result["prediction_falsified"] is True
    assert result["hit_rate"] == 0.5
    assert result["predicted_miss"] == ["t2"]


def test_no_eligible_prediction_is_not_verified_or_falsified():
    result = verify_prediction(
        ["t9"],
        {"t1": 0.0},
        {"t1": 1.0},
        k=1,
    )
    assert result["prediction_verified"] is None
    assert result["prediction_falsified"] is None
    assert result["hit_rate"] is None
    assert result["n_eligible"] == 0


def test_unpredicted_regression_is_reported():
    result = verify_prediction(
        ["t1"],
        {"t1": 0.0, "t2": 1.0},
        {"t1": 1.0, "t2": 0.0},
        k=1,
    )
    assert result["unpredicted_regressions"] == ["t2"]
    assert result["prediction_verified"] is True


def test_custom_threshold_flags_continuous_regression():
    result = verify_prediction([], {"t1": 0.8}, {"t1": 0.1}, k=1, threshold=0.05)
    assert result["unpredicted_regressions"] == ["t1"]


def test_missing_means_are_not_counted_as_eligible():
    result = verify_prediction(
        ["t1"],
        {"t1": None},
        {"t1": 1.0},
        k=1,
    )
    assert result["n_eligible"] == 0
    assert result["prediction_verified"] is None


def test_negative_threshold_is_rejected():
    with pytest.raises(PredictionError, match="threshold"):
        verify_prediction(["t1"], {"t1": 0.0}, {"t1": 1.0}, k=1, threshold=-0.1)


def test_non_positive_k_is_rejected():
    with pytest.raises(PredictionError, match="k"):
        verify_prediction(["t1"], {"t1": 0.0}, {"t1": 1.0}, k=0)
