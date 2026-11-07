from __future__ import annotations

import math
from typing import Sequence

from evo.application.baseline import (
    BaselineError,
    TrialOutcome,
    summarize_trials,
)
from evo.contracts.types import KVector
from evo.domain.stats import (
    Interval,
    StatsError,
    bonferroni_alpha,
    paired_bootstrap_diff,
)


class CompareError(ValueError):
    """Two arms cannot be compared under the pre-registered protocol."""


def _fraction(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompareError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise CompareError(f"{name} must be in [0, 1]: {value!r}")
    return number


def _mean(rows: dict[str, dict[str, object]], task: str, arm: str) -> float:
    row = rows[task]
    if int(row["trials"]) == 0:
        raise CompareError(f"missing paired rewards for task {task!r} in the {arm} arm")
    value = row["mean_reward"]
    if value is None:
        raise CompareError(f"missing paired rewards for task {task!r} in the {arm} arm")
    return float(value)


def _block(
    tasks: Sequence[str],
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    margin: float,
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    try:
        interval: Interval = paired_bootstrap_diff(
            reference, candidate, n_boot=n_boot, alpha=alpha, seed=seed
        )
    except StatsError as exc:
        raise CompareError(str(exc)) from exc
    deltas = [cand - ref for ref, cand in zip(reference, candidate)]
    worst_task = min(zip(tasks, deltas), key=lambda row: (row[1], row[0]))
    return {
        "n_pairs": len(tasks),
        "reference_mean": sum(reference) / len(reference),
        "candidate_mean": sum(candidate) / len(candidate),
        "mean_delta": interval.estimate,
        "ci": {
            "estimate": interval.estimate,
            "low": interval.low,
            "high": interval.high,
            "alpha": alpha,
            "n_boot": n_boot,
        },
        "margin": margin,
        "non_inferior": interval.low >= -margin,
        "worst_task": {"task": worst_task[0], "delta": worst_task[1]},
    }


def _ratio(candidate: float, reference: float) -> float | None:
    if reference == 0:
        return None if candidate == 0 else math.inf
    return candidate / reference


def _k_block(before: KVector, after: KVector) -> dict[str, object]:
    reduction = {}
    for name in ("k_lines_nb", "k_ast", "k_chars", "k_cc"):
        start = getattr(before, name)
        end = getattr(after, name)
        reduction[name] = round(100 * (start - end) / start, 4) if start else 0.0
    return {
        "before": before.as_dict(),
        "after": after.as_dict(),
        "delta": (before - after).as_dict(),
        "reduction_pct": reduction,
    }


def compare_arms(
    reference: Sequence[TrialOutcome],
    candidate: Sequence[TrialOutcome],
    *,
    tasks: Sequence[str],
    k: int,
    holdout: Sequence[str] = (),
    margin: float = 0.10,
    holdout_margin: float = 0.10,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    family_size: int = 1,
    k_before: KVector | None = None,
    k_after: KVector | None = None,
) -> dict[str, object]:
    task_ids = list(tasks)
    if not task_ids:
        raise CompareError("tasks must not be empty")
    if any(not isinstance(task, str) or not task.strip() for task in task_ids):
        raise CompareError(f"task ids must be non-empty strings: {task_ids!r}")
    if len(set(task_ids)) != len(task_ids):
        raise CompareError("tasks contain a duplicate")
    holdout_ids = list(holdout)
    if len(set(holdout_ids)) != len(holdout_ids):
        raise CompareError("holdout contains a duplicate")
    unknown = sorted(set(holdout_ids) - set(task_ids))
    if unknown:
        raise CompareError(f"holdout tasks are not in the task list: {', '.join(unknown)}")
    margin = _fraction("margin", margin)
    holdout_margin = _fraction("holdout_margin", holdout_margin)
    try:
        adjusted_alpha = bonferroni_alpha(alpha, family_size)
    except StatsError as exc:
        raise CompareError(str(exc)) from exc
    if (k_before is None) != (k_after is None):
        raise CompareError("k_before and k_after must be given together")
    if k_before is not None:
        if not isinstance(k_before, KVector) or not isinstance(k_after, KVector):
            raise CompareError("k_before and k_after must be KVector instances")

    try:
        ref = summarize_trials(reference, tasks=task_ids, k=k)
        cand = summarize_trials(candidate, tasks=task_ids, k=k)
    except BaselineError as exc:
        raise CompareError(str(exc)) from exc

    ref_means = [_mean(ref["tasks"], task, "reference") for task in task_ids]
    cand_means = [_mean(cand["tasks"], task, "candidate") for task in task_ids]
    holdout_set = set(holdout_ids)

    payload: dict[str, object] = {
        "tasks": {
            task: {
                "reference": ref_mean,
                "candidate": cand_mean,
                "delta": cand_mean - ref_mean,
                "holdout": task in holdout_set,
            }
            for task, ref_mean, cand_mean in zip(task_ids, ref_means, cand_means)
        },
        "aggregate": {
            **_block(
                task_ids,
                ref_means,
                cand_means,
                margin=margin,
                n_boot=n_boot,
                alpha=adjusted_alpha,
                seed=seed,
            ),
            "family": {
                "size": family_size,
                "alpha": adjusted_alpha,
                "method": "bonferroni" if family_size > 1 else "none",
            },
            "tokens": {
                "reference": {
                    "in": ref["aggregate"]["tokens_in"],
                    "out": ref["aggregate"]["tokens_out"],
                    "total": ref["aggregate"]["tokens_in"]
                    + ref["aggregate"]["tokens_out"],
                },
                "candidate": {
                    "in": cand["aggregate"]["tokens_in"],
                    "out": cand["aggregate"]["tokens_out"],
                    "total": cand["aggregate"]["tokens_in"]
                    + cand["aggregate"]["tokens_out"],
                },
            },
            "cost": {
                "reference": ref["aggregate"]["cost_usd"],
                "candidate": cand["aggregate"]["cost_usd"],
            },
        },
        "validity": {"reference": ref["validity"], "candidate": cand["validity"]},
        "arms": {"reference": ref, "candidate": cand},
    }
    tokens = payload["aggregate"]["tokens"]
    tokens["candidate"]["ratio"] = _ratio(
        tokens["candidate"]["total"], tokens["reference"]["total"]
    )
    payload["aggregate"]["cost"]["ratio"] = _ratio(
        payload["aggregate"]["cost"]["candidate"],
        payload["aggregate"]["cost"]["reference"],
    )

    if holdout_ids:
        held_ref = [
            mean for task, mean in zip(task_ids, ref_means) if task in holdout_set
        ]
        held_cand = [
            mean for task, mean in zip(task_ids, cand_means) if task in holdout_set
        ]
        payload["aggregate"]["holdout"] = _block(
            sorted(holdout_set),
            held_ref,
            held_cand,
            margin=holdout_margin,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
    if k_before is not None:
        payload["k"] = _k_block(k_before, k_after)
    return payload
