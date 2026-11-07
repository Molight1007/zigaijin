from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from evo.contracts.types import Component, KResult, KVector
from evo.domain.stats import Interval, paired_bootstrap_diff


ABLATION_VERDICTS = ("zero_contribution", "contributes", "inconclusive", "invalid")

_K_FIELDS = ("k_lines_nb", "k_ast", "k_chars", "k_bytes", "k_gzip", "k_cc")
_ID_RE = re.compile(r"^[a-z][a-z0-9_\-]*$")
_SUBJECT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_FLAG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCALAR_TYPES = (bool, int, float, str, type(None))
_TOP_KEYS = {"subject", "levers"}
_LEVER_KEYS = {"id", "flag", "default", "off", "components", "description"}


class AblationError(ValueError):
    """An ablation plan or decision input is invalid."""


def _is_scalar(value: object) -> bool:
    if not isinstance(value, _SCALAR_TYPES):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return True


@dataclass(frozen=True)
class Lever:
    id: str
    flag: str
    default_value: object
    off_value: object
    components: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.match(self.id):
            raise AblationError(f"invalid lever id: {self.id!r}")
        if not isinstance(self.flag, str) or not _FLAG_RE.match(self.flag):
            raise AblationError(f"invalid flag: {self.flag!r}")
        if not _is_scalar(self.default_value):
            raise AblationError(
                f"default value must be a finite JSON scalar for lever {self.id!r}"
            )
        if not _is_scalar(self.off_value):
            raise AblationError(
                f"off value must be a finite JSON scalar for lever {self.id!r}"
            )
        if self.default_value == self.off_value:
            raise AblationError(
                f"off value must differ from the default for lever {self.id!r}"
            )
        components = tuple(self.components)
        if not components:
            raise AblationError(
                f"lever {self.id!r} needs at least one component for K attribution"
            )
        seen: set[str] = set()
        for name in components:
            if not isinstance(name, str) or not name:
                raise AblationError(f"invalid component name in lever {self.id!r}")
            if name in seen:
                raise AblationError(
                    f"duplicate component {name!r} in lever {self.id!r}"
                )
            seen.add(name)
        if not isinstance(self.description, str):
            raise AblationError(f"description must be a string for lever {self.id!r}")
        object.__setattr__(self, "components", components)


@dataclass(frozen=True)
class AblationPlan:
    subject: str
    levers: tuple[Lever, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not _SUBJECT_RE.match(self.subject):
            raise AblationError(f"invalid subject: {self.subject!r}")
        levers = tuple(self.levers)
        if not levers:
            raise AblationError("ablation plan needs at least one lever")
        ids: set[str] = set()
        flags: set[str] = set()
        for lever in levers:
            if not isinstance(lever, Lever):
                raise AblationError("levers must be Lever instances")
            if lever.id in ids:
                raise AblationError(f"duplicate lever id: {lever.id!r}")
            if lever.flag in flags:
                raise AblationError(
                    f"duplicate flag {lever.flag!r}: two levers would confound each other"
                )
            ids.add(lever.id)
            flags.add(lever.flag)
        object.__setattr__(self, "levers", levers)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(lever.id for lever in self.levers)

    def get(self, lever_id: str) -> Lever:
        for lever in self.levers:
            if lever.id == lever_id:
                return lever
        raise AblationError(f"unknown lever: {lever_id!r}")

    def kwargs_for(self, lever_id: str) -> dict[str, object]:
        lever = self.get(lever_id)
        return {lever.flag: lever.off_value}


def load_ablation_plan(path: str | Path) -> AblationPlan:
    plan_path = Path(path)
    if not plan_path.is_file():
        raise AblationError(f"ablation plan is not a file: {plan_path}")
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AblationError(f"invalid JSON in ablation plan: {exc}") from exc
    if not isinstance(payload, dict):
        raise AblationError("ablation plan must be a JSON object")
    unknown = sorted(set(payload) - _TOP_KEYS)
    if unknown:
        raise AblationError(f"unknown key(s) in ablation plan: {unknown}")
    for key in ("subject", "levers"):
        if key not in payload:
            raise AblationError(f"ablation plan is missing {key!r}")
    raw_levers = payload["levers"]
    if not isinstance(raw_levers, list):
        raise AblationError("levers must be a list")
    levers: list[Lever] = []
    for index, item in enumerate(raw_levers):
        if not isinstance(item, dict):
            raise AblationError(f"lever {index} must be a JSON object")
        unknown = sorted(set(item) - _LEVER_KEYS)
        if unknown:
            raise AblationError(f"unknown key(s) in lever {index}: {unknown}")
        for key in ("id", "flag", "default", "off", "components"):
            if key not in item:
                raise AblationError(f"lever {index} is missing {key!r}")
        components = item["components"]
        if not isinstance(components, list):
            raise AblationError(f"components must be a list in lever {index}")
        levers.append(
            Lever(
                id=item["id"],
                flag=item["flag"],
                default_value=item["default"],
                off_value=item["off"],
                components=tuple(components),
                description=item.get("description", ""),
            )
        )
    return AblationPlan(subject=payload["subject"], levers=tuple(levers))


def attributed_k(
    lever: Lever, components: Sequence[Component], k: KResult
) -> KVector:
    """Upper bound on the K mass a lever's deletion can reach."""
    by_name: dict[str, Component] = {}
    for component in components:
        if component.name in by_name:
            raise AblationError(f"duplicate component name: {component.name!r}")
        by_name[component.name] = component
    paths: list[str] = []
    seen: set[str] = set()
    for name in lever.components:
        component = by_name.get(name)
        if component is None:
            raise AblationError(
                f"unknown component {name!r} for lever {lever.id!r}"
            )
        for path in component.paths:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    totals = {field: 0 for field in _K_FIELDS}
    for path in paths:
        vector = k.files.get(path)
        if vector is None:
            raise AblationError(f"component file missing from K result: {path}")
        for field, value in vector.as_dict().items():
            totals[field] += value
    return KVector(**totals)


@dataclass(frozen=True)
class Validity:
    ok: bool
    reasons: tuple[str, ...] = ()


def _check_fraction(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AblationError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise AblationError(f"{name} must be in [0, 1]: {value!r}")
    return number


def validity_gate(
    missing_frac: float, infra_frac: float, token_coverage: float
) -> Validity:
    missing = _check_fraction("missing_frac", missing_frac)
    infra = _check_fraction("infra_frac", infra_frac)
    coverage = _check_fraction("token_coverage", token_coverage)
    reasons: list[str] = []
    if missing > 0.10:
        reasons.append(f"missing_frac {missing:.3f} > 0.10")
    if infra > 0.10:
        reasons.append(f"infra_frac {infra:.3f} > 0.10")
    if coverage < 0.90:
        reasons.append(f"token_coverage {coverage:.3f} < 0.90")
    return Validity(ok=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class AblationDecision:
    verdict: str
    delta: Interval | None
    margin: float
    n_pairs: int
    token_ratio: float | None
    worst_token_task: tuple[str, float] | None
    reasons: tuple[str, ...]


def _numbers(name: str, values: Sequence[float], *, integer: bool = False) -> tuple:
    numbers = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            kind = "token" if integer else "reward"
            raise AblationError(f"{kind} at index {index} is not a number: {value!r}")
        if not math.isfinite(value):
            kind = "token" if integer else "reward"
            raise AblationError(f"{kind} at index {index} is not finite: {value!r}")
        if integer and (value < 0 or int(value) != value):
            raise AblationError(f"token at index {index} is invalid: {value!r}")
        numbers.append(int(value) if integer else float(value))
    return tuple(numbers)


def _token_stats(
    tasks: Sequence[str],
    on_tokens: tuple[int, ...],
    off_tokens: tuple[int, ...],
    tolerance: float,
) -> tuple[float, tuple[str, float], int, int]:
    on_total = sum(on_tokens)
    off_total = sum(off_tokens)
    if on_total == 0:
        ratio = 1.0 if off_total == 0 else math.inf
    else:
        ratio = off_total / on_total
    per_on: dict[str, int] = {}
    per_off: dict[str, int] = {}
    for task, before, after in zip(tasks, on_tokens, off_tokens):
        per_on[task] = per_on.get(task, 0) + before
        per_off[task] = per_off.get(task, 0) + after
    ratios = {
        task: (per_off[task] / before if before else (1.0 if per_off[task] == 0 else math.inf))
        for task, before in per_on.items()
    }
    worst = min(ratios.items(), key=lambda item: (-item[1], item[0]))
    return ratio, worst, on_total, off_total


def decide_ablation(
    task_ids: Sequence[str],
    on_rewards: Sequence[float],
    off_rewards: Sequence[float],
    *,
    margin: float,
    validity: Validity,
    on_tokens: Sequence[int] | None = None,
    off_tokens: Sequence[int] | None = None,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
    token_tolerance: float = 0.10,
) -> AblationDecision:
    """Compare a lever-on arm with the same lever forced off (`off - on`)."""
    tasks = tuple(task_ids)
    if not isinstance(validity, Validity):
        raise AblationError("validity must come from validity_gate()")
    if not tasks:
        raise AblationError("task_ids must not be empty")
    if any(not isinstance(task, str) or not task for task in tasks):
        raise AblationError("task ids must be non-empty strings")
    if len(on_rewards) != len(off_rewards):
        raise AblationError(
            f"paired reward samples differ in length: {len(on_rewards)} != {len(off_rewards)}"
        )
    if len(tasks) != len(on_rewards):
        raise AblationError(
            f"task_ids length {len(tasks)} != sample length {len(on_rewards)}"
        )
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        raise AblationError(f"margin must be a number: {margin!r}")
    if not math.isfinite(margin) or margin <= 0:
        raise AblationError(f"margin must be positive and finite: {margin!r}")
    if not isinstance(token_tolerance, (int, float)) or isinstance(token_tolerance, bool):
        raise AblationError(f"token_tolerance must be a number: {token_tolerance!r}")
    if not math.isfinite(token_tolerance) or token_tolerance < 0:
        raise AblationError(
            f"token_tolerance must be non-negative and finite: {token_tolerance!r}"
        )
    if not 0 < alpha < 1:
        raise AblationError(f"alpha must be in (0, 1): {alpha!r}")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 100:
        raise AblationError(f"n_boot must be at least 100: {n_boot!r}")

    on = _numbers("reward", on_rewards)
    off = _numbers("reward", off_rewards)
    if (on_tokens is None) != (off_tokens is None):
        raise AblationError("token lists must be provided together or not at all")
    tokens: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    if on_tokens is not None and off_tokens is not None:
        if len(on_tokens) != len(tasks) or len(off_tokens) != len(tasks):
            raise AblationError("token samples must align with the reward samples")
        tokens = (
            _numbers("token", on_tokens, integer=True),
            _numbers("token", off_tokens, integer=True),
        )

    if not validity.ok:
        return AblationDecision(
            verdict="invalid",
            delta=None,
            margin=float(margin),
            n_pairs=0,
            token_ratio=None,
            worst_token_task=None,
            reasons=validity.reasons,
        )

    delta = paired_bootstrap_diff(
        on, off, n_boot=n_boot, alpha=alpha, seed=seed
    )
    token_ratio: float | None = None
    worst_token_task: tuple[str, float] | None = None
    on_total = off_total = 0
    if tokens is not None:
        token_ratio, worst_token_task, on_total, off_total = _token_stats(
            tasks, tokens[0], tokens[1], token_tolerance
        )

    if delta.high < -margin:
        return AblationDecision(
            verdict="contributes",
            delta=delta,
            margin=float(margin),
            n_pairs=len(tasks),
            token_ratio=token_ratio,
            worst_token_task=worst_token_task,
            reasons=(f"s_harmful: upper CI {delta.high:.3f} < -{margin}",),
        )
    if delta.low < -margin:
        return AblationDecision(
            verdict="inconclusive",
            delta=delta,
            margin=float(margin),
            n_pairs=len(tasks),
            token_ratio=token_ratio,
            worst_token_task=worst_token_task,
            reasons=(
                f"s_inconclusive: CI [{delta.low:.3f}, {delta.high:.3f}] straddles -{margin}",
            ),
        )
    if token_ratio is None:
        return AblationDecision(
            verdict="inconclusive",
            delta=delta,
            margin=float(margin),
            n_pairs=len(tasks),
            token_ratio=None,
            worst_token_task=None,
            reasons=("token_gate_closed: no paired token data",),
        )
    if off_total > on_total:
        return AblationDecision(
            verdict="contributes",
            delta=delta,
            margin=float(margin),
            n_pairs=len(tasks),
            token_ratio=token_ratio,
            worst_token_task=worst_token_task,
            reasons=(f"token_total_increased: {off_total} > {on_total}",),
        )
    assert worst_token_task is not None
    if worst_token_task[1] > 1.0 + token_tolerance:
        task, ratio = worst_token_task
        return AblationDecision(
            verdict="contributes",
            delta=delta,
            margin=float(margin),
            n_pairs=len(tasks),
            token_ratio=token_ratio,
            worst_token_task=worst_token_task,
            reasons=(
                f"token_task_increase: {task} {ratio:.3f} > {1.0 + token_tolerance:.3f}",
            ),
        )
    return AblationDecision(
        verdict="zero_contribution",
        delta=delta,
        margin=float(margin),
        n_pairs=len(tasks),
        token_ratio=token_ratio,
        worst_token_task=worst_token_task,
        reasons=(),
    )
