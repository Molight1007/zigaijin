from __future__ import annotations

from typing import Mapping, Sequence

from evo.application.baseline import (
    BaselineError,
    TrialOutcome,
    summarize_trials,
)
from evo.application.compare import CompareError, compare_arms
from evo.contracts.types import KVector


class VersionCompareError(ValueError):
    """Version-comparison inputs are malformed or not comparable."""


def _k_block(
    vectors: Mapping[str, KVector],
    previous: str,
    candidate: str,
) -> dict[str, object] | None:
    if previous not in vectors or candidate not in vectors:
        return None
    before = vectors[previous]
    after = vectors[candidate]
    reduction = {}
    for name in ("k_lines_nb", "k_ast", "k_chars", "k_cc"):
        start = getattr(before, name)
        end = getattr(after, name)
        reduction[name] = round(100 * (start - end) / start, 4) if start else 0.0
    return {
        "arms": {name: vector.as_dict() for name, vector in vectors.items()},
        "previous": before.as_dict(),
        "candidate": after.as_dict(),
        "delta": (before - after).as_dict(),
        "reduction_pct": reduction,
    }


def _task_means(summary: Mapping[str, object], tasks: Sequence[str]) -> dict[str, float]:
    rows = summary.get("tasks")
    if not isinstance(rows, Mapping):
        raise VersionCompareError("summary has no task rows")
    values: dict[str, float] = {}
    for task in tasks:
        row = rows.get(task)
        if not isinstance(row, Mapping):
            raise VersionCompareError(f"summary has no row for task {task!r}")
        values[task] = float(row["mean_reward"])
    return values


def build_version_comparison(
    arms: Mapping[str, Sequence[TrialOutcome]],
    *,
    tasks: Sequence[str],
    holdout: Sequence[str],
    k: int,
    reference: str = "H0",
    previous: str = "previous",
    candidate: str = "candidate",
    margin: float = 0.10,
    holdout_margin: float = 0.10,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    k_vectors: Mapping[str, KVector] | None = None,
) -> dict[str, object]:
    """Compare H0 / previous / candidate on the exact same task set and k."""
    if not isinstance(arms, Mapping) or not arms:
        raise VersionCompareError("arms must be a non-empty mapping")
    if reference not in arms or candidate not in arms:
        raise VersionCompareError(
            f"arms must include {reference!r} and {candidate!r}"
        )
    task_ids = tuple(tasks)
    if not task_ids:
        raise VersionCompareError("tasks must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise VersionCompareError("tasks contain a duplicate")
    holdout_ids = tuple(holdout)
    if not set(holdout_ids) <= set(task_ids):
        raise VersionCompareError("holdout tasks must be a subset of tasks")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise VersionCompareError("k must be a positive integer")

    summaries: dict[str, dict[str, object]] = {}
    for name, outcomes in arms.items():
        if not isinstance(name, str) or not name:
            raise VersionCompareError("arm name must be a non-empty string")
        try:
            summaries[name] = summarize_trials(outcomes, tasks=task_ids, k=k)
        except BaselineError as exc:
            raise VersionCompareError(f"arm {name!r}: {exc}") from exc

    per_task: dict[str, dict[str, object]] = {}
    for task in task_ids:
        row: dict[str, object] = {}
        for name, summary in summaries.items():
            task_row = summary["tasks"][task]
            row[name] = {
                "mean_reward": task_row["mean_reward"],
                "valid_trials": task_row["valid_trials"],
                "trials": task_row["trials"],
                "tokens_in": task_row["tokens_in"],
                "tokens_out": task_row["tokens_out"],
                "cost_usd": task_row["cost_usd"],
            }
        per_task[task] = row

    means = {
        name: _task_means(summary, task_ids)
        for name, summary in summaries.items()
    }
    names = list(arms)
    pairs: list[tuple[str, str]] = []
    if previous in arms:
        pairs.append((reference, previous))
        pairs.append((previous, candidate))
    pairs.append((reference, candidate))
    comparisons: dict[str, dict[str, object]] = {}
    for left, right in pairs:
        try:
            comparison = compare_arms(
                list(arms[left]),
                list(arms[right]),
                tasks=task_ids,
                k=k,
                holdout=holdout_ids,
                margin=margin,
                holdout_margin=holdout_margin,
                n_boot=n_boot,
                alpha=alpha,
                seed=seed,
            )
        except CompareError as exc:
            raise VersionCompareError(f"{left}->{right}: {exc}") from exc
        aggregate = comparison["aggregate"]
        wins = losses = ties = 0
        for task in task_ids:
            delta = means[right][task] - means[left][task]
            if delta > 1e-9:
                wins += 1
            elif delta < -1e-9:
                losses += 1
            else:
                ties += 1
        comparisons[f"{left}->{right}"] = {
            "reference": left,
            "candidate": right,
            "mean_delta": aggregate["mean_delta"],
            "ci": aggregate["ci"],
            "non_inferior": aggregate["non_inferior"],
            "worst_task": aggregate["worst_task"],
            "holdout": aggregate.get("holdout"),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "validity": comparison["validity"],
        }

    verdict = "candidate_unchanged"
    pair = comparisons.get(f"{previous}->{candidate}") if previous in arms else None
    if pair is not None:
        if not pair["non_inferior"]:
            verdict = "candidate_regressed"
        elif pair["wins"] > pair["losses"] and float(pair["mean_delta"]) > 0:
            verdict = "candidate_improved"
        elif pair["wins"] == pair["losses"] and abs(float(pair["mean_delta"])) <= 1e-9:
            verdict = "candidate_unchanged"
        else:
            verdict = "candidate_non_inferior"
    held = pair.get("holdout") if isinstance(pair, Mapping) else None
    if isinstance(held, Mapping) and not held.get("non_inferior"):
        verdict = "candidate_holdout_regressed"

    return {
        "schema": 1,
        "tasks": list(task_ids),
        "holdout": list(holdout_ids),
        "k_reps": k,
        "arms": names,
        "per_task": per_task,
        "comparisons": comparisons,
        "validity": {
            name: summary["validity"] for name, summary in summaries.items()
        },
        "k": _k_block(k_vectors or {}, previous, candidate),
        "verdict": verdict,
    }
