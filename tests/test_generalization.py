from __future__ import annotations

import pytest

from evo.domain.generalization import GeneralizationError, direction_consistency


def test_same_direction_is_consistent_and_informative():
    result = direction_consistency(0.20, 0.10, delta=0.01)
    assert result["computed"] is True
    assert result["consistent"] is True
    assert result["informative"] is True
    assert result["feedback_direction"] == "up"
    assert result["heldout_direction"] == "up"


def test_opposite_direction_is_inconsistent():
    result = direction_consistency(-0.20, 0.10, delta=0.01)
    assert result["computed"] is True
    assert result["consistent"] is False
    assert result["informative"] is True


def test_both_flat_counts_as_consistent_but_not_informative():
    result = direction_consistency(0.005, -0.005, delta=0.01)
    assert result["computed"] is True
    assert result["consistent"] is True
    assert result["informative"] is False
    assert result["reason"] == "both_flat"


def test_one_flat_side_is_not_claimed_as_inconsistent():
    result = direction_consistency(0.20, 0.005, delta=0.01)
    assert result["computed"] is True
    assert result["consistent"] is None
    assert result["informative"] is False
    assert result["reason"] == "flat_one_side"


def test_missing_side_is_not_computed_and_not_counted_as_inconsistent():
    result = direction_consistency(None, 0.10, delta=0.01)
    assert result["computed"] is False
    assert result["consistent"] is None
    assert result["informative"] is False
    assert result["reason"] == "missing_side"


def test_negative_delta_is_rejected():
    with pytest.raises(GeneralizationError):
        direction_consistency(0.1, 0.1, delta=-0.01)


def test_non_numeric_delta_is_rejected():
    with pytest.raises(GeneralizationError):
        direction_consistency(0.1, 0.1, delta="0.1")  # type: ignore[arg-type]
