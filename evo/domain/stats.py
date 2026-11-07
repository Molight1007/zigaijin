from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping


class StatsError(ValueError):
    """Inputs cannot support the requested statistical procedure."""


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class TostResult:
    equivalent: bool
    p_value: float
    margin: float
    diff: float
    ci: Interval


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise StatsError(f"alpha must be in (0, 1): {alpha}")


def _validate_boot(n_boot: int) -> None:
    if n_boot < 100:
        raise StatsError(f"n_boot must be at least 100: {n_boot}")


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise StatsError("cannot take a percentile of an empty sample")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap_means(values: list[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    ]


def _interval(samples: list[float], alpha: float, estimate: float) -> Interval:
    ordered = sorted(samples)
    return Interval(
        estimate=estimate,
        low=_percentile(ordered, alpha / 2),
        high=_percentile(ordered, 1 - alpha / 2),
    )


def bootstrap_mean_ci(
    values: Iterable[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    data = list(values)
    if not data:
        raise StatsError("values must not be empty")
    _validate_boot(n_boot)
    _validate_alpha(alpha)
    return _interval(
        _bootstrap_means(data, n_boot, seed), alpha, statistics.fmean(data)
    )


def paired_bootstrap_diff(
    reference: Iterable[float],
    candidate: Iterable[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    ref = list(reference)
    cand = list(candidate)
    if not ref or not cand:
        raise StatsError("paired samples must not be empty")
    if len(ref) != len(cand):
        raise StatsError(f"paired samples differ in length: {len(ref)} != {len(cand)}")
    _validate_boot(n_boot)
    _validate_alpha(alpha)
    diffs = [b - a for a, b in zip(ref, cand)]
    return _interval(
        _bootstrap_means(diffs, n_boot, seed), alpha, statistics.fmean(diffs)
    )


def tost_paired(
    reference: Iterable[float],
    candidate: Iterable[float],
    margin: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> TostResult:
    if margin <= 0:
        raise StatsError(f"margin must be positive: {margin}")
    _validate_alpha(alpha)
    if alpha >= 0.5:
        raise StatsError("alpha must be below 0.5 for a two-sided interval")
    ref = list(reference)
    cand = list(candidate)
    if not ref or not cand:
        raise StatsError("paired samples must not be empty")
    if len(ref) != len(cand):
        raise StatsError(f"paired samples differ in length: {len(ref)} != {len(cand)}")
    _validate_boot(n_boot)

    diffs = [b - a for a, b in zip(ref, cand)]
    boot = _bootstrap_means(diffs, n_boot, seed)
    p_low = (sum(1 for value in boot if value <= -margin) + 1) / (n_boot + 1)
    p_high = (sum(1 for value in boot if value >= margin) + 1) / (n_boot + 1)
    p_value = max(p_low, p_high)
    ordered = sorted(boot)
    ci = Interval(
        estimate=statistics.fmean(diffs),
        low=_percentile(ordered, alpha),
        high=_percentile(ordered, 1 - alpha),
    )
    return TostResult(
        equivalent=p_value < alpha,
        p_value=p_value,
        margin=margin,
        diff=statistics.fmean(diffs),
        ci=ci,
    )


def _z(probability: float) -> float:
    return statistics.NormalDist().inv_cdf(probability)


def mde_paired(
    diffs: Iterable[float], alpha: float = 0.05, power: float = 0.8
) -> float:
    data = list(diffs)
    _validate_alpha(alpha)
    if not 0 < power < 1:
        raise StatsError(f"power must be in (0, 1): {power}")
    if len(data) < 2:
        return math.inf
    spread = statistics.stdev(data)
    if spread == 0:
        return 0.0
    return (_z(1 - alpha / 2) + _z(power)) * spread / math.sqrt(len(data))


def required_pairs(
    margin: float, sd: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    if margin <= 0:
        raise StatsError(f"margin must be positive: {margin}")
    if sd < 0:
        raise StatsError(f"sd must be non-negative: {sd}")
    _validate_alpha(alpha)
    if not 0 < power < 1:
        raise StatsError(f"power must be in (0, 1): {power}")
    if sd == 0:
        return 1
    return max(
        1,
        math.ceil(((_z(1 - alpha / 2) + _z(power)) * sd / margin) ** 2),
    )


def worst_case_ci_halfwidth(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        raise StatsError(f"n must be positive: {n}")
    _validate_alpha(alpha)
    return _z(1 - alpha / 2) * 0.5 / math.sqrt(n)


def per_task_worst(deltas: Mapping[str, float]) -> tuple[str, float]:
    if not deltas:
        raise StatsError("per-task deltas must not be empty")
    return min(deltas.items(), key=lambda item: (item[1], item[0]))


def bonferroni_alpha(alpha: float, family_size: int) -> float:
    """Per-comparison alpha that controls the family-wise error rate."""
    _validate_alpha(alpha)
    if (
        isinstance(family_size, bool)
        or not isinstance(family_size, int)
        or family_size < 1
    ):
        raise StatsError(f"family_size must be a positive integer: {family_size!r}")
    return alpha / family_size


def holm(p_values: Iterable[float], alpha: float = 0.05) -> dict[str, object]:
    """Holm step-down adjusted p-values, returned in the input order."""
    _validate_alpha(alpha)
    values = list(p_values)
    if not values:
        raise StatsError("p_values must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StatsError(f"p value must be a number: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise StatsError(f"p value must be in [0, 1]: {value!r}")
    m = len(values)
    order = sorted(range(m), key=lambda index: (float(values[index]), index))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * float(values[index])
        running = min(1.0, max(running, candidate))
        adjusted[index] = running
    return {
        "method": "holm",
        "alpha": alpha,
        "family_size": m,
        "adjusted": adjusted,
        "reject": [value <= alpha for value in adjusted],
    }
