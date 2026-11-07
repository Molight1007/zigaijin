from __future__ import annotations

import math
from typing import Mapping, Sequence


class PredictionError(ValueError):
    """A prediction-verification input is malformed."""


def _means(name: str, values: Mapping[str, float | None]) -> dict[str, float | None]:
    if not isinstance(values, Mapping):
        raise PredictionError(f"{name} must be a mapping")
    result: dict[str, float | None] = {}
    for task, value in values.items():
        if not isinstance(task, str) or not task:
            raise PredictionError(f"{name} has an invalid task id: {task!r}")
        if value is None:
            result[task] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PredictionError(f"{name}[{task!r}] must be a number or None")
        number = float(value)
        if not math.isfinite(number):
            raise PredictionError(f"{name}[{task!r}] must be finite")
        result[task] = number
    return result


def verify_prediction(
    predicted_affected: Sequence[str],
    incumbent_means: Mapping[str, float | None],
    candidate_means: Mapping[str, float | None],
    *,
    k: int,
    threshold: float | None = None,
) -> dict[str, object]:
    """Check a proposal's predicted tasks against measured per-task deltas.

    Only tasks with both means are eligible; missing data is not falsification.
    Verified means every eligible prediction improved; any miss falsifies it.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise PredictionError(f"k must be a positive integer: {k!r}")
    if threshold is None:
        threshold = 1.0 / max(1, k)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise PredictionError(f"threshold must be a number: {threshold!r}")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise PredictionError(f"threshold must be finite and non-negative: {threshold!r}")
    if not isinstance(predicted_affected, Sequence) or isinstance(
        predicted_affected, (str, bytes)
    ):
        raise PredictionError("predicted_affected must be a sequence")
    predicted: list[str] = []
    for task in predicted_affected:
        if not isinstance(task, str) or not task:
            raise PredictionError(f"predicted task must be a non-empty string: {task!r}")
        if task not in predicted:
            predicted.append(task)
    incumbent = _means("incumbent_means", incumbent_means)
    candidate = _means("candidate_means", candidate_means)
    eligible = [
        task
        for task in predicted
        if incumbent.get(task) is not None and candidate.get(task) is not None
    ]
    hits = [
        task
        for task in eligible
        if float(candidate[task]) > float(incumbent[task])  # type: ignore[arg-type]
    ]
    misses = [task for task in eligible if task not in hits]
    drops = [
        task
        for task in incumbent
        if incumbent[task] is not None
        and candidate.get(task) is not None
        and float(candidate[task]) - float(incumbent[task]) <= -threshold  # type: ignore[arg-type]
    ]
    return {
        "n_predicted": len(predicted),
        "n_eligible": len(eligible),
        "predicted_hit": hits,
        "predicted_miss": misses,
        "hit_rate": round(len(hits) / len(eligible), 2) if eligible else None,
        "unpredicted_regressions": [
            task for task in drops if task not in predicted
        ][:12],
        "prediction_verified": (not misses) if eligible else None,
        "prediction_falsified": bool(misses) if eligible else None,
        "threshold": threshold,
    }
