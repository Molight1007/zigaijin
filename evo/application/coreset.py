from __future__ import annotations

from typing import Sequence

from evo.application.baseline import TrialOutcome, summarize_trials
from evo.domain.coreset import CoreTask, CoresetError, CoresetResult, select


def select_core_tasks(
    outcomes: Sequence[TrialOutcome],
    tasks: Sequence[str],
    k: int,
    *,
    trials_per_task: int,
    theta: float = 0.7,
) -> CoresetResult:
    """Select a coreset from measured trials; difficulty is 1 - strict pass rate.

    Only complete tasks (all k trials, no agent/infra/missing) qualify; <k raises.
    Diagnostic or budget-shaping only. Anchor: run_workforce_round.
    """
    task_list = tuple(tasks)
    if not task_list:
        raise CoresetError("tasks must not be empty")
    known = set(task_list)
    rows = [row for row in outcomes if row.task_id in known]
    summary = summarize_trials(rows, tasks=task_list, k=trials_per_task)
    eligible: list[str] = []
    for task in task_list:
        row = summary["tasks"][task]
        if int(row["valid_trials"]) != trials_per_task:
            continue
        if (
            int(row["agent_errors"])
            or int(row["infra_errors"])
            or int(row["missing_trials"])
        ):
            continue
        eligible.append(task)
    if len(eligible) < min(k, len(task_list)):
        raise CoresetError(
            f"not enough complete tasks for k={k}: {len(eligible)} eligible"
        )
    core = [
        CoreTask(
            task_id=task,
            difficulty=1.0 - float(summary["tasks"][task]["mean_reward"]),
            fingerprint=task,
        )
        for task in eligible
    ]
    return select(core, k, theta=theta)
