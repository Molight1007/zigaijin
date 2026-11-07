from __future__ import annotations

import statistics
from typing import Sequence

from evo.application.baseline import BaselineError, TrialOutcome, summarize_trials
from evo.domain.stats import StatsError, mde_paired, required_pairs


class PowerError(ValueError):
    """The paired outcomes cannot support a power analysis."""


def _tasks(values: Sequence[str]) -> tuple[str, ...]:
    tasks = tuple(values)
    if not tasks:
        raise PowerError("tasks must not be empty")
    if any(not isinstance(task, str) or not task for task in tasks):
        raise PowerError(f"task ids must be non-empty strings: {tasks!r}")
    if len(set(tasks)) != len(tasks):
        raise PowerError("tasks contain a duplicate")
    return tasks


def power_report(
    reference: Sequence[TrialOutcome],
    candidate: Sequence[TrialOutcome],
    *,
    tasks: Sequence[str],
    k: int,
    margin: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, object]:
    """Paired design check: observed spread, MDE and the pairs a margin needs."""
    task_ids = _tasks(tasks)
    if margin <= 0:
        raise PowerError(f"margin must be positive: {margin}")
    try:
        ref = summarize_trials(list(reference), tasks=task_ids, k=k)
        cand = summarize_trials(list(candidate), tasks=task_ids, k=k)
    except BaselineError as exc:
        raise PowerError(str(exc)) from exc

    diffs: list[float] = []
    missing: list[str] = []
    for task in task_ids:
        ref_row = ref["tasks"][task]
        cand_row = cand["tasks"][task]
        if int(ref_row["trials"]) == 0 or int(cand_row["trials"]) == 0:
            missing.append(task)
            continue
        before = ref_row["mean_reward"]
        after = cand_row["mean_reward"]
        if before is None or after is None:
            missing.append(task)
            continue
        diffs.append(float(after) - float(before))
    if missing:
        raise PowerError(f"missing paired rewards for tasks: {', '.join(missing)}")
    if len(diffs) < 2:
        raise PowerError("power analysis needs at least two paired tasks")

    spread = statistics.stdev(diffs)
    try:
        mde = mde_paired(diffs, alpha=alpha, power=power)
        needed = required_pairs(margin, spread, alpha=alpha, power=power)
    except StatsError as exc:
        raise PowerError(str(exc)) from exc
    worst_task, worst_delta = min(
        zip(task_ids, diffs), key=lambda row: (row[1], row[0])
    )
    observed = statistics.fmean(diffs)
    return {
        "schema": 1,
        "n_pairs": len(diffs),
        "mean_delta": observed,
        "sd": spread,
        "mde": mde,
        "margin": margin,
        "alpha": alpha,
        "power": power,
        "required_pairs": needed,
        "enough": mde <= margin,
        "worst_task": {"task": worst_task, "delta": worst_delta},
    }
