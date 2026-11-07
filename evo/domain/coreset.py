from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

_EPS = 1e-10


class CoresetError(ValueError):
    """A coreset input is malformed."""


@dataclass(frozen=True)
class CoreTask:
    task_id: str
    difficulty: float
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise CoresetError(f"task_id must be a non-empty string: {self.task_id!r}")
        if isinstance(self.difficulty, bool) or not isinstance(
            self.difficulty, (int, float)
        ):
            raise CoresetError(f"difficulty must be a number: {self.difficulty!r}")
        value = float(self.difficulty)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise CoresetError(
                f"difficulty must be finite and in [0, 1]: {self.difficulty!r}"
            )
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise CoresetError(
                f"fingerprint must be a non-empty string: {self.fingerprint!r}"
            )


@dataclass(frozen=True)
class CoresetResult:
    picks: tuple[str, ...]
    gains: tuple[float, ...]
    trace: tuple[dict[str, object], ...]
    padded: tuple[str, ...]
    theta: float

    def as_dict(self) -> dict[str, object]:
        return {
            "picks": list(self.picks),
            "gains": list(self.gains),
            "trace": [dict(row) for row in self.trace],
            "padded": list(self.padded),
            "theta": self.theta,
        }


def _grams(text: str) -> Counter[str]:
    normalized = " ".join(text.lower().split())
    if len(normalized) < 3:
        return Counter({normalized: 1}) if normalized else Counter()
    return Counter(normalized[i : i + 3] for i in range(len(normalized) - 2))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right[key] for key in left.keys() & right.keys())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _greedy_map(kernel: Sequence[Sequence[float]], k: int) -> tuple[list[int], list[float]]:
    size = len(kernel)
    if size == 0 or k <= 0:
        return [], []
    d2 = [float(kernel[i][i]) for i in range(size)]
    c: list[list[float]] = [[] for _ in range(size)]
    picked = [False] * size
    first = max(range(size), key=lambda index: (d2[index], -index))
    if d2[first] <= _EPS:
        return [], []
    picks = [first]
    gains = [math.log(d2[first])]
    picked[first] = True
    while len(picks) < min(k, size):
        last = picks[-1]
        norm = math.sqrt(max(d2[last], 0.0))
        if norm <= 0.0:
            break
        for index in range(size):
            if picked[index]:
                continue
            if c[last]:
                dot = sum(
                    c[last][position] * c[index][position]
                    for position in range(len(c[last]))
                )
            else:
                dot = 0.0
            residual = (kernel[last][index] - dot) / norm
            c[index].append(residual)
            d2[index] = max(d2[index] - residual * residual, 0.0)
        candidates = [index for index in range(size) if not picked[index]]
        if not candidates:
            break
        nxt = max(candidates, key=lambda index: (d2[index], -index))
        if d2[nxt] <= _EPS:
            break
        picks.append(nxt)
        gains.append(math.log(d2[nxt]))
        picked[nxt] = True
    return picks, gains


def _validate_parameters(k: int, theta: float, score_floor: float) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise CoresetError(f"k must be a positive integer: {k!r}")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise CoresetError(f"theta must be a number: {theta!r}")
    if not math.isfinite(float(theta)) or not 0.0 <= float(theta) <= 1.0:
        raise CoresetError(f"theta must be in [0, 1]: {theta!r}")
    if isinstance(score_floor, bool) or not isinstance(score_floor, (int, float)):
        raise CoresetError(f"score_floor must be a number: {score_floor!r}")
    if not math.isfinite(float(score_floor)) or not 0.0 < float(score_floor) <= 1.0:
        raise CoresetError(f"score_floor must be in (0, 1]: {score_floor!r}")


def select(
    tasks: Sequence[CoreTask],
    k: int,
    *,
    theta: float = 0.7,
    score_floor: float = 0.1,
) -> CoresetResult:
    """Select a difficulty-diverse coreset with a pure-Python DPP greedy.

    Mirrors RHO's Diag(r) S Diag(r) kernel without numpy. When the kernel
    collapses (duplicate fingerprints), pad by difficulty to keep the budget k.
    """
    _validate_parameters(k, theta, score_floor)
    rows = list(tasks)
    if not rows:
        raise CoresetError("tasks must not be empty")
    ids = [row.task_id for row in rows]
    if len(set(ids)) != len(ids):
        raise CoresetError("tasks contain a duplicate task_id")
    target = min(k, len(rows))
    floored = [max(row.difficulty, float(score_floor)) for row in rows]
    scale = max(floored)
    raw = [value / scale for value in floored]

    if float(theta) == 1.0:
        order = sorted(range(len(rows)), key=lambda index: (-rows[index].difficulty, ids[index]))
        picks = order[:target]
        gains = [0.0] * len(picks)
        trace = tuple(
            {
                "step": step,
                "picked_id": ids[index],
                "log_det_gain": 0.0,
                "score": rows[index].difficulty,
                "padded": False,
                "mode": "difficulty",
            }
            for step, index in enumerate(picks)
        )
        return CoresetResult(
            picks=tuple(ids[index] for index in picks),
            gains=tuple(gains),
            trace=trace,
            padded=(),
            theta=float(theta),
        )

    alpha = float(theta) / (2.0 * max(1.0 - float(theta), 1e-6))
    weights = [value**alpha for value in raw]
    grams = [_grams(row.fingerprint) for row in rows]
    kernel = [
        [weights[i] * _cosine(grams[i], grams[j]) * weights[j] for j in range(len(rows))]
        for i in range(len(rows))
    ]
    picks, gains = _greedy_map(kernel, target)
    picked_ids = [ids[index] for index in picks]
    padded: list[str] = []
    remaining = sorted(
        (index for index in range(len(rows)) if index not in picks),
        key=lambda index: (-rows[index].difficulty, ids[index]),
    )
    trace: list[dict[str, object]] = [
        {
            "step": step,
            "picked_id": ids[index],
            "log_det_gain": gain,
            "score": rows[index].difficulty,
            "padded": False,
            "mode": "dpp",
        }
        for step, (index, gain) in enumerate(zip(picks, gains))
    ]
    for index in remaining:
        if len(picked_ids) >= target:
            break
        picked_ids.append(ids[index])
        padded.append(ids[index])
        gains.append(0.0)
        trace.append(
            {
                "step": len(trace),
                "picked_id": ids[index],
                "log_det_gain": 0.0,
                "score": rows[index].difficulty,
                "padded": True,
                "mode": "difficulty",
            }
        )
    return CoresetResult(
        picks=tuple(picked_ids),
        gains=tuple(gains),
        trace=tuple(trace),
        padded=tuple(padded),
        theta=float(theta),
    )
