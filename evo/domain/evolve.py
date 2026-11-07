from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class EvolveError(ValueError):
    """An evolution input is malformed or violates the pre-registered method."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvolveError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EvolveError(f"{name} must be finite: {value!r}")
    return number


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvolveError(f"{name} must be a non-negative integer: {value!r}")
    return value


def _positive_int(name: str, value: object) -> int:
    number = _non_negative_int(name, value)
    if number == 0:
        raise EvolveError(f"{name} must be positive: {value!r}")
    return number


def _component(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvolveError(f"{name} must be a non-empty string: {value!r}")
    return value


def _components(values: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        name = _component("component", value)
        if name in seen:
            raise EvolveError(f"duplicate component: {name!r}")
        seen.append(name)
    return tuple(seen)


def edit_budget(t: int, rounds: int, *, b_min: int = 1, b_max: int = 4) -> int:
    """b_t of Eq. (anneal): b_min + (b_max - b_min) * (1 + cos(pi t / T)) / 2."""
    _non_negative_int("t", t)
    _positive_int("rounds", rounds)
    _positive_int("b_min", b_min)
    _positive_int("b_max", b_max)
    if b_min > b_max:
        raise EvolveError(f"b_min {b_min} exceeds b_max {b_max}")
    clamped = min(t, rounds)
    value = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * clamped / rounds))
    return int(math.ceil(round(value, 9)))


def stall_flag(
    trajectory: Sequence[float], t: int, *, window: int, delta: float
) -> bool:
    """sigma_t = 1[S_t - S_{t-w} <= delta]; false while fewer than w rounds exist."""
    _non_negative_int("t", t)
    _positive_int("window", window)
    threshold = _finite("delta", delta)
    if threshold < 0:
        raise EvolveError(f"delta must be non-negative: {delta!r}")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise EvolveError("trajectory must be a sequence of scores")
    scores = [_finite("trajectory score", value) for value in trajectory]
    if t >= len(scores):
        raise EvolveError(f"t {t} is outside a trajectory of length {len(scores)}")
    if t < window:
        return False
    return bool(scores[t] - scores[t - window] <= threshold)


def exploration(
    components: Sequence[str],
    tried: Iterable[str],
    *,
    stalled: bool,
    m_draft: int,
) -> dict[str, object]:
    """E_t = (sigma_t, U_t, m_draft) plus the text handed to the proposer."""
    vocabulary = _components(components)
    if not isinstance(stalled, bool):
        raise EvolveError(f"stalled must be a bool: {stalled!r}")
    slots = _non_negative_int("m_draft", m_draft)
    tried_set: set[str] = set()
    for value in tried:
        name = _component("tried component", value)
        if name not in vocabulary:
            raise EvolveError(f"unknown tried component: {name!r}")
        tried_set.add(name)
    untried = [name for name in vocabulary if name not in tried_set]
    reserved = min(slots, len(untried)) if stalled and untried else 0
    if stalled and untried:
        text = (
            f"STALL: the incumbent has not moved by more than the noise band. "
            f"{reserved} candidate slot(s) are RESERVED for edits on components "
            f"the run has never exercised: {untried}."
        )
    elif untried:
        text = (
            f"Components not yet exercised in this run: {untried}. Not mandatory "
            f"this round (no stall), but evidence about them is still missing."
        )
    else:
        text = "Every component in the vocabulary has been exercised at least once."
    return {
        "sigma": stalled,
        "untried": untried,
        "m_draft": slots,
        "reserved": reserved,
        "text": text,
    }


def novelty(components: Iterable[str], accepted_counts: Mapping[str, int]) -> int:
    """nu(l'): edited components with no accepted edit in the incumbent."""
    if not isinstance(accepted_counts, Mapping):
        raise EvolveError("accepted_counts must be a mapping")
    unique = {_component("component", value) for value in components}
    return sum(1 for name in unique if int(accepted_counts.get(name, 0)) == 0)


def _record_round(record: Mapping[str, object], index: int) -> int:
    value = record.get("t")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvolveError(f"history record {index} has an invalid t: {value!r}")
    return value


def tried_components(records: Iterable[Mapping[str, object]]) -> set[str]:
    """T_t: components with at least one measured edit."""
    tried: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EvolveError(f"history record {index} is not an object")
        delta = record.get("delta_S")
        if delta is None:
            continue
        _finite(f"history record {index} delta_S", delta)
        component = record.get("component")
        if component is None:
            continue
        tried.add(_component(f"history record {index} component", component))
    return tried


def prune_set(
    records: Iterable[Mapping[str, object]],
    *,
    t: int,
    n_prune: int,
) -> list[dict[str, object]]:
    """B_t: tried components whose best recent Delta S is non-positive."""
    _non_negative_int("t", t)
    _non_negative_int("n_prune", n_prune)
    best: dict[str, float] = {}
    accepted: dict[str, list[dict[str, object]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EvolveError(f"history record {index} is not an object")
        component = record.get("component")
        if component is None:
            continue
        name = _component(f"history record {index} component", component)
        if record.get("accepted") is True:
            accepted.setdefault(name, []).append(
                {
                    "t": _record_round(record, index),
                    "edit_id": record.get("edit_id"),
                    "hypothesis": record.get("hypothesis"),
                    "delta_S": record.get("delta_S"),
                }
            )
        delta = record.get("delta_S")
        if delta is None:
            continue
        gain = _finite(f"history record {index} delta_S", delta)
        if t - _record_round(record, index) <= n_prune:
            best[name] = max(best.get(name, -math.inf), gain)
        else:
            best.setdefault(name, -math.inf)
        best.setdefault(name, -math.inf)
    rows = []
    for name in sorted(best):
        gain = best[name]
        if gain <= 0:
            rows.append(
                {
                    "component": name,
                    "recent_best_gain": None if gain == -math.inf else round(gain, 5),
                    "accepted_edits_in_incumbent": accepted.get(name, []),
                }
            )
    return rows


@dataclass(frozen=True)
class EvolvePolicy:
    beta0: float = 0.10
    beta1: float = 40.0
    w_s: float = 100.0
    w_c: float = 15.0
    w_n: float = 0.5

    def __post_init__(self) -> None:
        for name in ("beta0", "beta1", "w_s", "w_c", "w_n"):
            value = _finite(name, getattr(self, name))
            if value < 0:
                raise EvolveError(f"{name} must be non-negative: {value!r}")
        if self.w_s <= 0:
            raise EvolveError(f"w_s must be positive: {self.w_s!r}")

    def as_dict(self) -> dict[str, float]:
        return {
            "beta0": self.beta0,
            "beta1": self.beta1,
            "w_s": self.w_s,
            "w_c": self.w_c,
            "w_n": self.w_n,
        }


@dataclass(frozen=True)
class EvolveMeasurement:
    variant: str
    score: float | None
    tokens_per_trial: float | None = None
    valid: bool = True
    gate_failure: str | None = None
    components: tuple[str, ...] = ()
    k_lines: int | None = None
    k_limit: int | None = None

    def __post_init__(self) -> None:
        _component("variant", self.variant)
        if not isinstance(self.valid, bool):
            raise EvolveError(f"valid must be a bool: {self.valid!r}")
        if self.score is not None:
            _finite("score", self.score)
        if self.tokens_per_trial is not None:
            tokens = _finite("tokens_per_trial", self.tokens_per_trial)
            if tokens < 0:
                raise EvolveError(
                    f"tokens_per_trial must be non-negative: {self.tokens_per_trial!r}"
                )
        if self.gate_failure is not None:
            _component("gate_failure", self.gate_failure)
        for value in self.components:
            _component("component", value)
        if self.k_lines is not None:
            _non_negative_int("k_lines", self.k_lines)
        if self.k_limit is not None:
            _non_negative_int("k_limit", self.k_limit)


@dataclass(frozen=True)
class EvolveDecision:
    variant: str
    admissible: bool
    reason: str
    score: float | None = None
    tokens_per_trial: float | None = None
    delta_score: float | None = None
    delta_tokens: float | None = None
    novelty: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "admissible": self.admissible,
            "reason": self.reason,
            "score": self.score,
            "tokens_per_trial": self.tokens_per_trial,
            "delta_score": self.delta_score,
            "delta_tokens": self.delta_tokens,
            "novelty": self.novelty,
        }


def cost_rule(
    delta_score: float,
    delta_tokens: float,
    novelty_count: int,
    delta: float,
    policy: EvolvePolicy,
) -> tuple[bool, str]:
    """c of Algorithm 2, line 5."""
    gain = _finite("delta_score", delta_score)
    cost = _finite("delta_tokens", delta_tokens)
    count = _non_negative_int("novelty", novelty_count)
    threshold = _finite("delta", delta)
    if threshold < 0:
        raise EvolveError(f"delta must be non-negative: {delta!r}")
    if not isinstance(policy, EvolvePolicy):
        raise EvolveError("policy must be an EvolvePolicy")
    if gain > threshold:
        budget = policy.beta0 + policy.beta1 * gain
        ok = cost <= budget
        return ok, (
            f"gain {gain:+.4f} > delta {threshold:.4f}; cost change "
            f"{cost:+.3f} {'<=' if ok else '>'} budget {budget:.3f} "
            f"(beta0 {policy.beta0} + beta1 {policy.beta1} * dS)"
        )
    shaped = policy.w_s * gain - policy.w_c * cost + policy.w_n * count
    ok = shaped > 0
    return ok, (
        f"gain {gain:+.4f} within delta {threshold:.4f}; shaped "
        f"{policy.w_s}*dS - {policy.w_c}*dC + {policy.w_n}*nu = {shaped:+.4f} "
        f"{'>' if ok else '<='} 0 (nu={count})"
    )


def judge(
    measurement: EvolveMeasurement,
    *,
    incumbent_score: float,
    incumbent_tokens: float | None,
    s_star: float,
    delta: float,
    policy: EvolvePolicy,
    accepted_counts: Mapping[str, int],
) -> EvolveDecision:
    if not isinstance(measurement, EvolveMeasurement):
        raise EvolveError("measurement must be an EvolveMeasurement")
    base_score = _finite("incumbent_score", incumbent_score)
    star = _finite("s_star", s_star)
    threshold = _finite("delta", delta)
    if threshold < 0:
        raise EvolveError(f"delta must be non-negative: {delta!r}")
    if incumbent_tokens is not None:
        tokens = _finite("incumbent_tokens", incumbent_tokens)
        if tokens < 0:
            raise EvolveError(f"incumbent_tokens must be non-negative: {incumbent_tokens!r}")
    if not isinstance(policy, EvolvePolicy):
        raise EvolveError("policy must be an EvolvePolicy")

    if measurement.gate_failure is not None:
        return EvolveDecision(
            measurement.variant,
            False,
            f"candidate failed before measurement: {measurement.gate_failure}",
        )
    if not measurement.valid:
        return EvolveDecision(
            measurement.variant, False, "invalid measurement: candidate evaluation did not pass validity"
        )
    if measurement.score is None:
        return EvolveDecision(measurement.variant, False, "no score was reported")

    score = float(measurement.score)
    floor = star - threshold
    if score < floor:
        return EvolveDecision(
            measurement.variant,
            False,
            f"below noise-adjusted floor: S' {score:.4f} < S* {star:.4f} "
            f"- delta {threshold:.4f}",
            score=score,
            tokens_per_trial=measurement.tokens_per_trial,
        )

    if (
        measurement.k_limit is not None
        and measurement.k_lines is not None
        and measurement.k_lines > measurement.k_limit
    ):
        return EvolveDecision(
            measurement.variant,
            False,
            f"k growth exceeds limit: {measurement.k_lines} > {measurement.k_limit}",
            score=score,
            tokens_per_trial=measurement.tokens_per_trial,
        )

    gain = score - base_score
    if measurement.tokens_per_trial is None:
        return EvolveDecision(
            measurement.variant,
            False,
            "token data missing; cost rule cannot be evaluated",
            score=score,
        )
    if incumbent_tokens is None:
        return EvolveDecision(
            measurement.variant,
            False,
            "incumbent token data missing; cost rule cannot be evaluated",
            score=score,
        )
    if incumbent_tokens > 0:
        cost = (float(measurement.tokens_per_trial) - float(incumbent_tokens)) / float(
            incumbent_tokens
        )
    else:
        cost = 0.0
    count = novelty(measurement.components, accepted_counts)
    ok, why = cost_rule(gain, cost, count, threshold, policy)
    return EvolveDecision(
        measurement.variant,
        ok,
        f"admissible: {why}" if ok else f"cost rule failed: {why}",
        score=score,
        tokens_per_trial=measurement.tokens_per_trial,
        delta_score=gain,
        delta_tokens=cost,
        novelty=count,
    )


def select_round(
    measurements: Sequence[EvolveMeasurement],
    *,
    incumbent_score: float,
    incumbent_tokens: float | None,
    s_star: float,
    delta: float,
    policy: EvolvePolicy,
    accepted_counts: Mapping[str, int],
) -> tuple[EvolveMeasurement | None, list[EvolveDecision]]:
    """Judge every candidate, return (winner or None, decisions)."""
    seen: set[str] = set()
    decisions: list[EvolveDecision] = []
    for item in measurements:
        if not isinstance(item, EvolveMeasurement):
            raise EvolveError("every candidate must be an EvolveMeasurement")
        if item.variant in seen:
            raise EvolveError(f"duplicate candidate variant: {item.variant!r}")
        seen.add(item.variant)
        decisions.append(
            judge(
                item,
                incumbent_score=incumbent_score,
                incumbent_tokens=incumbent_tokens,
                s_star=s_star,
                delta=delta,
                policy=policy,
                accepted_counts=accepted_counts,
            )
        )
    admissible = [
        (item, decision)
        for item, decision in zip(measurements, decisions)
        if decision.admissible
    ]
    if not admissible:
        return None, decisions
    winner, _ = min(
        admissible,
        key=lambda pair: (
            -float(pair[0].score),  # type: ignore[arg-type]
            pair[0].tokens_per_trial if pair[0].tokens_per_trial is not None else math.inf,
            pair[0].k_lines if pair[0].k_lines is not None else math.inf,
            pair[0].variant,
        ),
    )
    return winner, decisions
