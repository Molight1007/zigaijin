from __future__ import annotations

import math
from dataclasses import dataclass


L3_YIELD_THRESHOLD = 0.20
K_AST_PCT_THRESHOLD = 1.0
MIN_CENSUS_TASKS = 10
DECISIONS = ("proceed", "switch_subject", "pending_l3", "pending_l2")


class PrestudyError(ValueError):
    """A prestudy input or decision configuration is invalid."""


def _pct(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrestudyError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise PrestudyError(f"{name} must be in [0, 100]: {value!r}")
    return number


def _yield(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrestudyError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise PrestudyError(f"{name} must be in [0, 1]: {value!r}")
    return number


@dataclass(frozen=True)
class PrestudyDecision:
    decision: str
    combined_pct: float
    l3_yield: float | None
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "combined_pct": self.combined_pct,
            "l3_yield": self.l3_yield,
            "reasons": list(self.reasons),
        }


def decide_prestudy(
    l1_pct: float,
    l2_pct: float,
    l3_yield: float | None = None,
    *,
    census_tasks: int | None = None,
) -> PrestudyDecision:
    l1 = _pct("l1_pct", l1_pct)
    l2 = _pct("l2_pct", l2_pct)
    if census_tasks is not None:
        if isinstance(census_tasks, bool) or not isinstance(census_tasks, int) or census_tasks < 0:
            raise PrestudyError(f"census_tasks must be a non-negative integer: {census_tasks!r}")
    combined = l1 + l2
    reach_reason = (
        f"static+census K_ast reach {combined:.3f}% "
        f"{'>=' if combined >= K_AST_PCT_THRESHOLD else '<'} {K_AST_PCT_THRESHOLD}%"
    )
    if census_tasks is not None and census_tasks < MIN_CENSUS_TASKS:
        return PrestudyDecision(
            "pending_l2",
            combined,
            None,
            (
                f"census covers {census_tasks} < {MIN_CENSUS_TASKS} tasks; "
                "zero-touch mass is provisional",
                reach_reason,
            ),
        )
    if l3_yield is None:
        if combined >= K_AST_PCT_THRESHOLD:
            return PrestudyDecision("proceed", combined, None, (reach_reason,))
        return PrestudyDecision(
            "pending_l3", combined, None, (reach_reason, "L3 yield not measured yet")
        )

    measured = _yield("l3_yield", l3_yield)
    yield_reason = (
        f"L3 yield {measured:.1%} "
        f"{'>=' if measured >= L3_YIELD_THRESHOLD else '<'} {L3_YIELD_THRESHOLD:.0%}"
    )
    if measured < L3_YIELD_THRESHOLD and combined < K_AST_PCT_THRESHOLD:
        return PrestudyDecision(
            "switch_subject", combined, measured, (yield_reason, reach_reason)
        )
    reason = yield_reason if measured >= L3_YIELD_THRESHOLD else reach_reason
    return PrestudyDecision("proceed", combined, measured, (reason,))
