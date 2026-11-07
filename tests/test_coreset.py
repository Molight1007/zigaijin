from __future__ import annotations

import pytest

from evo.domain.coreset import CoreTask, CoresetError, select


def _tasks() -> list[CoreTask]:
    return [
        CoreTask("a", 1.0, "alpha beta gamma delta"),
        CoreTask("b", 1.0, "alpha beta gamma delta"),
        CoreTask("c", 0.9, "epsilon zeta eta theta"),
        CoreTask("d", 0.9, "iota kappa lambda mu"),
        CoreTask("e", 0.8, "nu xi omicron pi"),
    ]


def test_select_returns_k_unique_tasks():
    result = select(_tasks(), 3, theta=0.7)
    assert len(result.picks) == 3
    assert len(set(result.picks)) == 3
    assert result.padded == ()


def test_duplicate_fingerprint_is_not_picked_twice_when_diversity_matters():
    result = select(_tasks(), 3, theta=0.7)
    assert "a" in result.picks
    assert "b" not in result.picks
    assert set(result.picks) == {"a", "c", "d"}


def test_rank_collapse_is_filled_by_difficulty_and_marked_padded():
    tasks = [
        CoreTask("a", 1.0, "same fingerprint"),
        CoreTask("b", 0.8, "same fingerprint"),
        CoreTask("c", 0.6, "same fingerprint"),
    ]
    result = select(tasks, 3, theta=0.7)
    assert result.picks == ("a", "b", "c")
    assert result.padded == ("b", "c")
    assert all(row.get("padded") is True for row in result.trace[1:])


def test_k_larger_than_pool_returns_the_whole_pool():
    result = select(_tasks(), 10, theta=0.7)
    assert len(result.picks) == 5
    assert set(result.picks) == {"a", "b", "c", "d", "e"}


def test_selection_is_deterministic_for_the_same_input():
    first = select(_tasks(), 3, theta=0.7)
    second = select(_tasks(), 3, theta=0.7)
    assert first.picks == second.picks
    assert first.trace == second.trace


def test_theta_one_falls_back_to_difficulty_order():
    result = select(_tasks(), 2, theta=1.0)
    assert result.picks[0] == "a"


def test_duplicate_task_ids_are_rejected():
    tasks = [CoreTask("a", 1.0, "x"), CoreTask("a", 0.5, "y")]
    with pytest.raises(CoresetError, match="duplicate"):
        select(tasks, 1)


def test_empty_pool_is_rejected():
    with pytest.raises(CoresetError, match="empty"):
        select([], 1)


def test_non_positive_k_is_rejected():
    with pytest.raises(CoresetError, match="k"):
        select(_tasks(), 0)


def test_difficulty_out_of_range_is_rejected():
    with pytest.raises(CoresetError, match="difficulty"):
        CoreTask("a", 1.2, "x")


def test_theta_out_of_range_is_rejected():
    with pytest.raises(CoresetError, match="theta"):
        select(_tasks(), 1, theta=1.1)


def test_score_floor_out_of_range_is_rejected():
    with pytest.raises(CoresetError, match="score_floor"):
        select(_tasks(), 1, score_floor=0.0)
