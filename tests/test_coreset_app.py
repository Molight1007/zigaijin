from __future__ import annotations

import pytest

from evo.application.baseline import TrialOutcome
from evo.application.coreset import select_core_tasks
from evo.domain.coreset import CoresetError


def _outcome(
    task_id: str,
    reward: float | None,
    *,
    trial_id: str = "trial",
    status: str = "completed",
) -> TrialOutcome:
    return TrialOutcome(
        task_id=task_id,
        trial_id=f"{trial_id}-{task_id}",
        status=status,
        reward=reward,
        tokens_in=10,
        tokens_out=5,
    )


def test_select_core_tasks_uses_failure_difficulty():
    tasks = ("t1", "t2", "t3", "t4")
    outcomes = [
        _outcome("t1", 1.0),
        _outcome("t2", 1.0),
        _outcome("t3", 0.0),
        _outcome("t4", 0.0),
    ]
    result = select_core_tasks(outcomes, tasks, 2, trials_per_task=1)
    assert len(result.picks) == 2
    assert set(result.picks) == {"t3", "t4"}


def test_select_core_tasks_ignores_holdout_rows():
    tasks = ("t1", "t2")
    outcomes = [
        _outcome("t1", 0.0),
        _outcome("t2", 1.0),
        _outcome("h1", 0.0),
    ]
    result = select_core_tasks(outcomes, tasks, 1, trials_per_task=1)
    assert result.picks == ("t1",)


def test_select_core_tasks_rejects_empty_tasks():
    with pytest.raises(CoresetError, match="empty"):
        select_core_tasks([], (), 1, trials_per_task=1)


def test_incomplete_trials_are_not_selectable():
    rows = [
        _outcome("t1", 1.0),  # only 1 of 2
        _outcome("t2", 1.0, trial_id="a"),
        _outcome("t2", 1.0, trial_id="b"),
    ]
    with pytest.raises(CoresetError, match="complete"):
        select_core_tasks(rows, ("t1", "t2"), 2, trials_per_task=2)


def test_select_core_tasks_skips_a_harder_incomplete_task():
    rows = [
        _outcome("t1", 0.0),  # hardest, but only 1 of 2 trials
        _outcome("t2", 1.0, trial_id="a"),
        _outcome("t2", 1.0, trial_id="b"),
        _outcome("t3", 1.0, trial_id="a"),
        _outcome("t3", 1.0, trial_id="b"),
    ]
    result = select_core_tasks(rows, ("t1", "t2", "t3"), 2, trials_per_task=2)
    assert set(result.picks) == {"t2", "t3"}


def test_select_core_tasks_skips_agent_and_infra_errors():
    rows = [
        _outcome("t1", 1.0, trial_id="a"),
        _outcome("t1", None, trial_id="b", status="agent_error"),
        _outcome("t2", 1.0, trial_id="a"),
        _outcome("t2", None, trial_id="b", status="infra_error"),
        _outcome("t3", 1.0, trial_id="a"),
        _outcome("t3", 1.0, trial_id="b"),
        _outcome("t4", 1.0, trial_id="a"),
        _outcome("t4", 1.0, trial_id="b"),
    ]
    result = select_core_tasks(rows, ("t1", "t2", "t3", "t4"), 2, trials_per_task=2)
    assert set(result.picks) == {"t3", "t4"}
