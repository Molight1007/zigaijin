from __future__ import annotations

import math


class GeneralizationError(ValueError):
    """A direction-consistency input is malformed."""


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeneralizationError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise GeneralizationError(f"{name} must be finite: {value!r}")
    return number


def _optional_number(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _number(name, value)


def _direction(value: float | None, delta: float) -> str:
    if value is None:
        return "missing"
    if abs(value) <= delta:
        return "flat"
    return "up" if value > 0 else "down"


def direction_consistency(
    delta_feedback: float | None,
    delta_heldout: float | None,
    *,
    delta: float,
) -> dict[str, object]:
    """Compare feedback and held-out deltas without claiming direction from noise.

    A missing side is not inconsistent; a flat side has no direction. Only two
    non-flat sides can be informative. Diagnostic only; never feeds the gate.
    """
    threshold = _number("delta", delta)
    if threshold < 0:
        raise GeneralizationError(f"delta must be non-negative: {delta!r}")
    feedback = _optional_number("delta_feedback", delta_feedback)
    heldout = _optional_number("delta_heldout", delta_heldout)
    feedback_direction = _direction(feedback, threshold)
    heldout_direction = _direction(heldout, threshold)

    if feedback is None or heldout is None:
        return {
            "computed": False,
            "consistent": None,
            "informative": False,
            "reason": "missing_side",
            "delta_feedback": feedback,
            "delta_heldout": heldout,
            "delta": threshold,
            "feedback_direction": feedback_direction,
            "heldout_direction": heldout_direction,
        }
    if feedback_direction == "flat" and heldout_direction == "flat":
        return {
            "computed": True,
            "consistent": True,
            "informative": False,
            "reason": "both_flat",
            "delta_feedback": feedback,
            "delta_heldout": heldout,
            "delta": threshold,
            "feedback_direction": feedback_direction,
            "heldout_direction": heldout_direction,
        }
    if feedback_direction == "flat" or heldout_direction == "flat":
        return {
            "computed": True,
            "consistent": None,
            "informative": False,
            "reason": "flat_one_side",
            "delta_feedback": feedback,
            "delta_heldout": heldout,
            "delta": threshold,
            "feedback_direction": feedback_direction,
            "heldout_direction": heldout_direction,
        }
    consistent = feedback_direction == heldout_direction
    return {
        "computed": True,
        "consistent": consistent,
        "informative": True,
        "reason": "same_direction" if consistent else "opposite_direction",
        "delta_feedback": feedback,
        "delta_heldout": heldout,
        "delta": threshold,
        "feedback_direction": feedback_direction,
        "heldout_direction": heldout_direction,
    }
