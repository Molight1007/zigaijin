from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from evo.contracts.types import KVector
from evo.domain.stats import paired_bootstrap_diff


VERDICTS = ("invalid", "accept", "reject")
_FRACTION = 1_000_000


class GateError(ValueError):
    """A gate input is malformed or violates the pre-registered protocol."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise GateError(f"{name} must be finite: {value!r}")
    return number


def _fraction(name: str, value: object) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise GateError(f"{name} must be in [0, 1]: {value!r}")
    return number


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{name} must be a non-negative integer: {value!r}")
    return value


def _positive(name: str, value: object) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise GateError(f"{name} must be positive: {value!r}")
    return number


def _validate_kvector(name: str, vector: object) -> KVector:
    if not isinstance(vector, KVector):
        raise GateError(f"{name} must be a KVector: {vector!r}")
    for field_name, value in vector.as_dict().items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(
                f"{name}.{field_name} must be a non-negative integer: {value!r}"
            )
    return vector


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reasons: tuple[str, ...] = ()
    data: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "reasons": list(self.reasons),
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class GateResult:
    verdict: str
    checks: tuple[Check, ...]
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "checks": [check.as_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ValidityInput:
    missing_frac: float
    infra_frac: float
    token_coverage: float
    complete: bool = True
    agent_fail_frac: float = 0.0

    def __post_init__(self) -> None:
        _fraction("missing_frac", self.missing_frac)
        _fraction("infra_frac", self.infra_frac)
        _fraction("token_coverage", self.token_coverage)
        _fraction("agent_fail_frac", self.agent_fail_frac)
        if not isinstance(self.complete, bool):
            raise GateError(f"complete must be a bool: {self.complete!r}")


@dataclass(frozen=True)
class ScoreInput:
    tasks: tuple[str, ...]
    reference: tuple[float, ...]
    candidate: tuple[float, ...]
    margin: float

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        reference = tuple(self.reference)
        candidate = tuple(self.candidate)
        if not tasks:
            raise GateError("score tasks must not be empty")
        if any(not isinstance(task, str) or not task for task in tasks):
            raise GateError("score task ids must be non-empty strings")
        if len(set(tasks)) != len(tasks):
            raise GateError("score task ids must be unique")
        if len(reference) != len(tasks) or len(candidate) != len(tasks):
            raise GateError(
                f"paired scores must align with {len(tasks)} tasks: "
                f"{len(reference)} reference vs {len(candidate)} candidate"
            )
        for value in reference:
            _finite("reference reward", value)
        for value in candidate:
            _finite("candidate reward", value)
        _positive("margin", self.margin)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "candidate", candidate)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(self.candidate) - statistics.fmean(self.reference)


@dataclass(frozen=True)
class TokenInput:
    before: Mapping[str, int]
    after: Mapping[str, int]
    tolerance: float = 0.10
    before_missing: tuple[str, ...] = ()
    after_missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, mapping in (("before", self.before), ("after", self.after)):
            if not isinstance(mapping, Mapping) or not mapping:
                raise GateError(f"token {name} must be a non-empty mapping")
            for task, value in mapping.items():
                if not isinstance(task, str) or not task:
                    raise GateError(f"token {name} has an invalid task id: {task!r}")
                _non_negative_int(f"token {name}[{task!r}]", value)
        tolerance = _finite("tolerance", self.tolerance)
        if tolerance < 0:
            raise GateError(f"tolerance must be non-negative: {self.tolerance!r}")
        for name in ("before_missing", "after_missing"):
            raw = getattr(self, name)
            if isinstance(raw, str):
                raise GateError(
                    f"{name} must be a tuple of task ids, not a string: {raw!r}"
                )
            values = tuple(raw)
            if any(not isinstance(value, str) or not value for value in values):
                raise GateError(f"{name} must contain non-empty task ids")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class KGateInput:
    before: KVector
    after: KVector
    min_lines: int = 10
    min_pct: float = 0.005

    def __post_init__(self) -> None:
        _validate_kvector("before", self.before)
        _validate_kvector("after", self.after)
        _non_negative_int("min_lines", self.min_lines)
        pct = _finite("min_pct", self.min_pct)
        if not 0 < pct < 1:
            raise GateError(f"min_pct must be in (0, 1): {self.min_pct!r}")


@dataclass(frozen=True)
class ProtectionInput:
    hits: tuple[str, ...] = ()
    frozen_patch: bool = False
    public_signature_changed: bool = False

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        if any(not isinstance(hit, str) or not hit for hit in hits):
            raise GateError(f"protection hits must be non-empty strings: {self.hits!r}")
        for name in ("frozen_patch", "public_signature_changed"):
            if not isinstance(getattr(self, name), bool):
                raise GateError(f"{name} must be a bool")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True)
class GateRequest:
    validity: ValidityInput
    scores: ScoreInput
    holdout: ScoreInput
    k: KGateInput
    tokens: TokenInput
    protection: ProtectionInput
    round: int
    protection_until: int
    task_margin: float
    n_boot: int = 10_000
    alpha: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        for name, expected in (
            ("validity", ValidityInput),
            ("scores", ScoreInput),
            ("holdout", ScoreInput),
            ("k", KGateInput),
            ("tokens", TokenInput),
            ("protection", ProtectionInput),
        ):
            if not isinstance(getattr(self, name), expected):
                raise GateError(f"{name} must be a {expected.__name__}")
        _non_negative_int("round", self.round)
        _non_negative_int("protection_until", self.protection_until)
        _positive("task_margin", self.task_margin)
        if isinstance(self.n_boot, bool) or not isinstance(self.n_boot, int) or self.n_boot < 100:
            raise GateError(f"n_boot must be at least 100: {self.n_boot!r}")
        alpha = _finite("alpha", self.alpha)
        if not 0 < alpha < 1:
            raise GateError(f"alpha must be in (0, 1): {self.alpha!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise GateError(f"seed must be an integer: {self.seed!r}")


def check_validity(
    missing_frac: float,
    infra_frac: float,
    token_coverage: float,
    complete: bool = True,
    agent_fail_frac: float = 0.0,
    max_agent_fail_frac: float = 0.10,
) -> Check:
    missing = _fraction("missing_frac", missing_frac)
    infra = _fraction("infra_frac", infra_frac)
    coverage = _fraction("token_coverage", token_coverage)
    agent = _fraction("agent_fail_frac", agent_fail_frac)
    limit = _fraction("max_agent_fail_frac", max_agent_fail_frac)
    if not isinstance(complete, bool):
        raise GateError(f"complete must be a bool: {complete!r}")
    reasons: list[str] = []
    if missing > 0.10:
        reasons.append(f"missing_frac {missing:.3f} > 0.10")
    if infra > 0.10:
        reasons.append(f"infra_frac {infra:.3f} > 0.10")
    if coverage < 0.90:
        reasons.append(f"token_coverage {coverage:.3f} < 0.90")
    if agent > limit:
        reasons.append(f"agent_fail_frac {agent:.3f} > {limit:.3f}")
    if not complete:
        reasons.append("repetitions incomplete for at least one task")
    return Check(
        "validity",
        "pass" if not reasons else "fail",
        tuple(reasons),
        {
            "missing_frac": missing,
            "infra_frac": infra,
            "token_coverage": coverage,
            "agent_fail_frac": agent,
            "complete": complete,
        },
    )


def _score_check(name: str, score: ScoreInput, n_boot: int, alpha: float, seed: int) -> Check:
    interval = paired_bootstrap_diff(
        score.reference,
        score.candidate,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    reasons: tuple[str, ...] = ()
    if interval.low < -score.margin:
        reasons = (
            f"ci_low {interval.low:.4f} < -{score.margin} (diff {interval.estimate:.4f})",
        )
    return Check(
        name,
        "pass" if not reasons else "fail",
        reasons,
        {
            "diff": interval.estimate,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "margin": score.margin,
            "n_pairs": len(score.tasks),
        },
    )


def check_non_inferiority(
    score: ScoreInput,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Check:
    if not isinstance(score, ScoreInput):
        raise GateError("score must be a ScoreInput")
    return _score_check("non_inferiority", score, n_boot, alpha, seed)


def check_k(
    before: KVector,
    after: KVector,
    *,
    min_lines: int = 10,
    min_pct: float = 0.005,
) -> Check:
    _validate_kvector("before", before)
    _validate_kvector("after", after)
    min_lines = _non_negative_int("min_lines", min_lines)
    pct = _finite("min_pct", min_pct)
    if not 0 < pct < 1:
        raise GateError(f"min_pct must be in (0, 1): {min_pct!r}")
    keep = round((1 - pct) * _FRACTION)
    reasons: list[str] = []
    if after.k_ast * _FRACTION > before.k_ast * keep:
        reasons.append(
            f"k_ast did not drop by {pct:.3%}: {before.k_ast} -> {after.k_ast}"
        )
    if after.k_lines_nb * _FRACTION > before.k_lines_nb * keep:
        reasons.append(
            f"k_lines_nb did not drop by {pct:.3%}: "
            f"{before.k_lines_nb} -> {after.k_lines_nb}"
        )
    removed_lines = before.k_lines_nb - after.k_lines_nb
    if removed_lines < min_lines:
        reasons.append(
            f"k_lines_nb drop {removed_lines} < required {min_lines}"
        )
    if after.k_chars > before.k_chars:
        reasons.append(f"k_chars rose: {before.k_chars} -> {after.k_chars}")
    if after.k_cc > before.k_cc:
        reasons.append(f"k_cc rose: {before.k_cc} -> {after.k_cc}")
    data = {
        "before": before.as_dict(),
        "after": after.as_dict(),
        "delta": (before - after).as_dict(),
        "removed_lines": removed_lines,
        "min_lines": min_lines,
        "min_pct": pct,
        "reduction_pct": {
            name: (round(100 * (getattr(before, name) - getattr(after, name)) / getattr(before, name), 4)
                   if getattr(before, name) else 0.0)
            for name in ("k_lines_nb", "k_ast", "k_chars", "k_cc")
        },
    }
    return Check("k", "pass" if not reasons else "fail", tuple(reasons), data)


def check_tokens(
    before: Mapping[str, int],
    after: Mapping[str, int],
    *,
    tolerance: float = 0.10,
    before_missing: tuple[str, ...] = (),
    after_missing: tuple[str, ...] = (),
) -> Check:
    token = TokenInput(
        before=before,
        after=after,
        tolerance=tolerance,
        before_missing=before_missing,
        after_missing=after_missing,
    )
    reasons: list[str] = []
    if token.before_missing:
        reasons.append("token data missing before: " + ", ".join(token.before_missing))
    if token.after_missing:
        reasons.append("token data missing after: " + ", ".join(token.after_missing))
    missing = sorted(set(token.before) - set(token.after))
    extra = sorted(set(token.after) - set(token.before))
    if missing:
        reasons.append(f"token missing for task(s): {', '.join(missing)}")
    if extra:
        reasons.append(f"token reported for unknown task(s): {', '.join(extra)}")
    before_total = sum(token.before.values())
    after_total = sum(token.after.values())
    if after_total > before_total:
        reasons.append(f"token total increased: {before_total} -> {after_total}")
    common = sorted(set(token.before) & set(token.after))
    ratios = {
        task: (
            token.after[task] / token.before[task]
            if token.before[task]
            else (1.0 if token.after[task] == 0 else math.inf)
        )
        for task in common
    }
    worst_task = min(ratios, key=lambda task: (-ratios[task], task)) if ratios else None
    worst_ratio = ratios[worst_task] if worst_task is not None else None
    if worst_ratio is not None and worst_ratio > 1 + tolerance:
        reasons.append(
            f"token_task_increase: {worst_task} {worst_ratio:.3f} > {1 + tolerance:.3f}"
        )
    return Check(
        "tokens",
        "pass" if not reasons else "fail",
        tuple(reasons),
        {
            "before_total": before_total,
            "after_total": after_total,
            "worst_task": worst_task,
            "worst_ratio": worst_ratio,
            "tolerance": token.tolerance,
        },
    )


def check_protection(
    hits: Sequence[str] = (),
    *,
    frozen_patch: bool = False,
    public_signature_changed: bool = False,
) -> Check:
    protection = ProtectionInput(
        hits=tuple(hits),
        frozen_patch=frozen_patch,
        public_signature_changed=public_signature_changed,
    )
    reasons: list[str] = []
    for hit in protection.hits:
        reasons.append(f"protection hit: {hit}")
    if protection.frozen_patch:
        reasons.append("frozen compatibility patch must not be edited")
    if protection.public_signature_changed:
        reasons.append("public signature changed")
    return Check(
        "protection",
        "pass" if not reasons else "fail",
        tuple(reasons),
        {
            "hits": list(protection.hits),
            "frozen_patch": protection.frozen_patch,
            "public_signature_changed": protection.public_signature_changed,
        },
    )


def check_t4(protection_until: int, round: int) -> Check:
    until = _non_negative_int("protection_until", protection_until)
    current = _non_negative_int("round", round)
    reasons: tuple[str, ...] = ()
    if until > current:
        reasons = (f"component protected until round {until} > {current}",)
    return Check(
        "t4",
        "pass" if not reasons else "fail",
        reasons,
        {"protection_until": until, "round": current},
    )


def check_pareto(delta_score: float, delta_k_ast: int) -> Check:
    score = _finite("delta_score", delta_score)
    k_ast = _finite("delta_k_ast", delta_k_ast)
    reasons: tuple[str, ...] = ()
    if score <= 0 and k_ast >= 0:
        reasons = (
            f"no improvement on either axis: dS {score:.4f} <= 0 and dK_ast {k_ast:.0f} >= 0",
        )
    return Check(
        "pareto",
        "pass" if not reasons else "fail",
        reasons,
        {"delta_score": score, "delta_k_ast": k_ast},
    )


def check_worst_task(deltas: Mapping[str, float], margin: float) -> Check:
    if not isinstance(deltas, Mapping) or not deltas:
        raise GateError("per-task deltas must be a non-empty mapping")
    values: dict[str, float] = {}
    for task, value in deltas.items():
        if not isinstance(task, str) or not task:
            raise GateError(f"invalid task id in deltas: {task!r}")
        values[task] = _finite(f"delta[{task!r}]", value)
    limit = _positive("margin", margin)
    worst_task = min(values, key=lambda task: (values[task], task))
    worst_delta = values[worst_task]
    reasons: tuple[str, ...] = ()
    if worst_delta < -limit:
        reasons = (f"worst task {worst_task} dS {worst_delta:.4f} < -{limit}",)
    return Check(
        "worst_task",
        "pass" if not reasons else "fail",
        reasons,
        {"worst_task": worst_task, "worst_delta": worst_delta, "margin": limit},
    )


def simplification_gate(request: GateRequest) -> GateResult:
    validity = check_validity(
        request.validity.missing_frac,
        request.validity.infra_frac,
        request.validity.token_coverage,
        complete=request.validity.complete,
        agent_fail_frac=request.validity.agent_fail_frac,
    )
    if validity.status == "fail":
        return GateResult("invalid", (validity,), validity.reasons)

    per_task = {
        task: candidate - reference
        for task, reference, candidate in zip(
            request.scores.tasks, request.scores.reference, request.scores.candidate
        )
    }
    delta_k_ast = request.k.after.k_ast - request.k.before.k_ast
    checks = (
        validity,
        _score_check("non_inferiority", request.scores, request.n_boot, request.alpha, request.seed),
        _score_check("holdout", request.holdout, request.n_boot, request.alpha, request.seed),
        check_k(
            request.k.before,
            request.k.after,
            min_lines=request.k.min_lines,
            min_pct=request.k.min_pct,
        ),
        check_tokens(
            request.tokens.before,
            request.tokens.after,
            tolerance=request.tokens.tolerance,
            before_missing=request.tokens.before_missing,
            after_missing=request.tokens.after_missing,
        ),
        check_protection(
            request.protection.hits,
            frozen_patch=request.protection.frozen_patch,
            public_signature_changed=request.protection.public_signature_changed,
        ),
        check_t4(request.protection_until, request.round),
        check_pareto(request.scores.mean_delta, delta_k_ast),
        check_worst_task(per_task, request.task_margin),
    )
    reasons = tuple(reason for check in checks for reason in check.reasons)
    verdict = "accept" if all(check.status == "pass" for check in checks) else "reject"
    return GateResult(verdict, checks, reasons)
