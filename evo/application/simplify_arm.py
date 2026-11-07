from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from evo.adapters.store_fs import RunStore
from evo.application.baseline import (
    BaselineError,
    TrialOutcome,
    summarize_trials,
)
from evo.application.round import RoundError, decide_from_artifacts
from evo.application.simplify import CandidateDecision, decide_candidate, record_decision
from evo.contracts.types import KVector
from evo.domain.arms import ArmError, arm_config
from evo.domain.census import CensusError, function_units
from evo.domain.kmetrics import KMeasurementError, measure_source
from evo.domain.ledger import Ledger
from evo.domain.protection import ProtectionError, ProtectionPolicy, check_protection
from evo.domain.scan import Candidate
from evo.domain.verify import VerifyReport


class SimplifyArmError(ValueError):
    """A simplify-arm plan or batch is malformed."""


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    delta: KVector


@dataclass(frozen=True)
class SimplifySpec:
    arm: str
    mode: str
    k: int
    run_id: str = "simplify"
    round: int = 0
    margin: float = 0.10
    holdout_margin: float = 0.10
    task_margin: float = 0.10
    n_boot: int = 10_000
    alpha: float = 0.05
    seed: int = 0
    target_k_ast: int | None = None
    batch_seed: int = 0
    levers: tuple[Mapping[str, object], ...] = ()
    agent_kwargs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            config = arm_config(self.arm)
        except ArmError as exc:
            raise SimplifyArmError(str(exc)) from exc
        allowed = {
            "SIM": ("static", "census", "full"),
            "PLACEBO": ("random",),
            "SHARP": ("sharp",),
        }
        if self.arm not in allowed:
            raise SimplifyArmError(f"arm {self.arm!r} is not a simplify arm")
        if self.mode not in allowed[self.arm]:
            raise SimplifyArmError(
                f"mode {self.mode!r} is not valid for arm {self.arm!r}"
            )
        if config.simplify == "off":
            raise SimplifyArmError(f"arm {self.arm!r} has simplify=off")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise SimplifyArmError("k must be a positive integer")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise SimplifyArmError("run_id must be a non-empty string")
        if isinstance(self.round, bool) or not isinstance(self.round, int) or self.round < 0:
            raise SimplifyArmError("round must be a non-negative integer")
        for name in ("margin", "holdout_margin", "task_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SimplifyArmError(f"{name} must be a number")
            if not 0 <= float(value) <= 1:
                raise SimplifyArmError(f"{name} must be in [0, 1]")
        if not 0 < float(self.alpha) < 1:
            raise SimplifyArmError("alpha must be in (0, 1)")
        if isinstance(self.n_boot, bool) or not isinstance(self.n_boot, int) or self.n_boot < 100:
            raise SimplifyArmError("n_boot must be at least 100")
        if self.target_k_ast is not None and (
            isinstance(self.target_k_ast, bool)
            or not isinstance(self.target_k_ast, int)
            or self.target_k_ast <= 0
        ):
            raise SimplifyArmError("target_k_ast must be a positive integer or None")
        if isinstance(self.batch_seed, bool) or not isinstance(self.batch_seed, int):
            raise SimplifyArmError("batch_seed must be an integer")
        if any(not isinstance(row, Mapping) for row in self.levers):
            raise SimplifyArmError("levers must be mappings")
        if any(not isinstance(arg, str) or "=" not in arg for arg in self.agent_kwargs):
            raise SimplifyArmError("agent_kwargs must be flag=value strings")


@dataclass(frozen=True)
class SimplifyRuntime:
    apply: Callable[[Path, tuple[ScoredCandidate, ...]], Iterable[str]]
    verify: Callable[[Path, tuple[str, ...]], VerifyReport]
    measure: Callable[[Path], KVector]
    evaluate: Callable[[Path, tuple[str, ...], int], Sequence[TrialOutcome]]


def _overlaps(first: Candidate, second: Candidate) -> bool:
    return first.path == second.path and not (
        first.span[1] < second.span[0] or second.span[1] < first.span[0]
    )


def _place(items: Iterable[ScoredCandidate], selected: list[ScoredCandidate]) -> None:
    for item in items:
        if all(not _overlaps(item.candidate, other.candidate) for other in selected):
            selected.append(item)


def _static(items: Sequence[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
    ordered = sorted(
        items,
        key=lambda item: (
            -item.delta.k_ast,
            item.candidate.path,
            item.candidate.span,
            item.candidate.id,
        ),
    )
    selected: list[ScoredCandidate] = []
    _place(ordered, selected)
    return tuple(selected)


def _random(
    items: Sequence[ScoredCandidate], target_k_ast: int, seed: int
) -> tuple[ScoredCandidate, ...]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    selected: list[ScoredCandidate] = []
    total = 0
    for item in shuffled:
        if total >= target_k_ast:
            break
        if all(not _overlaps(item.candidate, other.candidate) for other in selected):
            selected.append(item)
            total += item.delta.k_ast
    return tuple(selected)


def _sharp(
    items: Sequence[ScoredCandidate], target_k_ast: int
) -> tuple[ScoredCandidate, ...]:
    groups: dict[str, list[ScoredCandidate]] = {}
    for item in items:
        groups.setdefault(item.candidate.component, []).append(item)
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            -sum(item.delta.k_ast for item in group),
            group[0].candidate.component,
        ),
    )
    selected: list[ScoredCandidate] = []
    total = 0
    for group in ordered:
        if total >= target_k_ast:
            break
        before = len(selected)
        _place(group, selected)
        total += sum(item.delta.k_ast for item in selected[before:])
    return tuple(selected)


def ddmin(
    items: Sequence[ScoredCandidate],
    still_fails: Callable[[Sequence[ScoredCandidate]], bool],
) -> tuple[ScoredCandidate, ...]:
    """Remove candidates one by one while the failing property survives."""
    current = list(items)
    changed = True
    while changed:
        changed = False
        for item in list(current):
            trial = [row for row in current if row is not item]
            if still_fails(trial):
                current = trial
                changed = True
                break
    return tuple(current)


def plan_batch(
    items: Sequence[ScoredCandidate],
    *,
    mode: str,
    target_k_ast: int | None = None,
    seed: int = 0,
) -> tuple[ScoredCandidate, ...]:
    """Select a non-overlapping batch for one simplify arm."""
    rows = tuple(items)
    if not rows:
        raise SimplifyArmError("candidate pool must not be empty")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, ScoredCandidate):
            raise SimplifyArmError("items must be ScoredCandidate instances")
        if row.candidate.blocked:
            raise SimplifyArmError(
                f"candidate {row.candidate.id!r} is protected"
            )
        if row.candidate.id in seen:
            raise SimplifyArmError(f"duplicate candidate id: {row.candidate.id!r}")
        seen.add(row.candidate.id)
        if not isinstance(row.delta, KVector):
            raise SimplifyArmError("candidate delta must be a KVector")
    if mode == "static":
        return _static(rows)
    if mode not in ("random", "sharp"):
        raise SimplifyArmError(f"unknown batch mode: {mode!r}")
    if isinstance(target_k_ast, bool) or not isinstance(target_k_ast, int):
        raise SimplifyArmError("target_k_ast must be an integer")
    if target_k_ast <= 0:
        raise SimplifyArmError("target_k_ast must be positive")
    if mode == "random":
        return _random(rows, target_k_ast, seed)
    return _sharp(rows, target_k_ast)


def target_k_from_candidates(items: Sequence[ScoredCandidate]) -> int:
    """Return the K_ast mass of the static non-overlapping batch."""
    return sum(row.delta.k_ast for row in plan_batch(items, mode="static"))


def select_zero_contribution_levers(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Filter an ablation report to L3 levers with a zero-contribution verdict."""
    levers = payload.get("levers")
    if not isinstance(levers, list):
        raise SimplifyArmError("ablation report has no levers list")
    selected: list[dict[str, object]] = []
    for index, row in enumerate(levers):
        if not isinstance(row, Mapping):
            raise SimplifyArmError(f"ablation lever {index} is not an object")
        if row.get("verdict") == "zero_contribution":
            selected.append(dict(row))
    return tuple(selected)


def lever_agent_kwargs(
    levers: Iterable[Mapping[str, object]],
) -> tuple[str, ...]:
    """Collect unique flag=value overrides from selected L3 levers."""
    args: list[str] = []
    for row in levers:
        raw = row.get("agent_kwargs") or []
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and "=" in item and item not in args:
                args.append(item)
    return tuple(args)


def apply_batch(
    root: str | Path, batch: Sequence[ScoredCandidate]
) -> tuple[str, ...]:
    """Delete every candidate span from a copied subject, bottom-up per file."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise SimplifyArmError(f"subject root is not a directory: {root}")
    if not batch:
        raise SimplifyArmError("batch must not be empty")
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row in batch:
        candidate = row.candidate
        raw = Path(candidate.path)
        if not candidate.path or raw.is_absolute() or ".." in raw.parts:
            raise SimplifyArmError(f"candidate path escapes the subject: {candidate.path!r}")
        target = (base / raw).resolve()
        if not target.is_relative_to(base):
            raise SimplifyArmError(f"candidate path escapes the subject: {candidate.path!r}")
        if not target.is_file():
            raise SimplifyArmError(f"candidate file does not exist: {candidate.path!r}")
        grouped.setdefault(raw.as_posix(), []).append(candidate.span)

    changed: list[str] = []
    for relative, spans in sorted(grouped.items()):
        target = base / relative
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        ordered = sorted(spans, key=lambda span: (span[0], span[1]))
        for start, end in ordered:
            if start < 1 or end < start or end > len(lines):
                raise SimplifyArmError(
                    f"span {start}-{end} is outside {relative} (1..{len(lines)})"
                )
        for previous, following in zip(ordered, ordered[1:]):
            if following[0] <= previous[1]:
                raise SimplifyArmError(
                    f"overlapping spans in {relative}: {previous} and {following}"
                )
        for start, end in sorted(ordered, reverse=True):
            del lines[start - 1 : end]
        target.write_text("".join(lines), encoding="utf-8", newline="\n")
        changed.append(relative)
    return tuple(changed)


def _span_delta(source: str, span: tuple[int, int]) -> KVector | None:
    lines = source.splitlines(keepends=True)
    start, end = span
    if start < 1 or end < start or end > len(lines):
        return None
    after = "".join(lines[: start - 1] + lines[end:])
    before_vector = measure_source(source)
    after_vector = measure_source(after)
    return before_vector - after_vector


def all_unit_candidates(
    root: str | Path,
    *,
    policy: ProtectionPolicy | None = None,
    components: Mapping[str, str] | None = None,
) -> tuple[ScoredCandidate, ...]:
    """Return every unblocked function unit, including touched functions."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise SimplifyArmError(f"subject root is not a directory: {root}")
    skip = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
    sources: dict[str, str] = {}
    for path in sorted(base.rglob("*.py")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if any(part in skip for part in relative.parts[:-1]):
            continue
        try:
            sources[relative.as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SimplifyArmError(f"cannot read {relative}: {exc}") from exc
    try:
        units = function_units(sources)
    except CensusError as exc:
        raise SimplifyArmError(str(exc)) from exc
    protection = policy or ProtectionPolicy()
    rows: list[ScoredCandidate] = []
    for index, unit in enumerate(units, start=1):
        source = sources[unit.path]
        try:
            hits = check_protection(source, unit.path, protection, unit.full_span)
            delta = _span_delta(source, unit.full_span)
        except ProtectionError as exc:
            raise SimplifyArmError(str(exc)) from exc
        except KMeasurementError as exc:
            raise SimplifyArmError(str(exc)) from exc
        if hits or delta is None:
            continue
        rows.append(
            ScoredCandidate(
                candidate=Candidate(
                    id=f"unit-{index:04d}",
                    layer="L2",
                    kind="unit_candidate",
                    component=(components or {}).get(unit.path, unit.path),
                    path=unit.path,
                    span=unit.full_span,
                    symbol=unit.qualname,
                ),
                delta=delta,
            )
        )
    return tuple(rows)


def load_scored_candidates(path: str | Path) -> tuple[ScoredCandidate, ...]:
    """Load an unblocked candidate pool from a census JSON artifact."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SimplifyArmError(f"cannot read candidates {source}: {exc}") from exc
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise SimplifyArmError("candidate payload must contain a candidates list")
    selected: list[ScoredCandidate] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SimplifyArmError(f"candidate {index} must be an object")
        if row.get("blocked"):
            continue
        delta = row.get("delta")
        if not isinstance(delta, dict):
            continue
        candidate_id = row.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise SimplifyArmError(f"candidate {index} has no id")
        if candidate_id in seen:
            raise SimplifyArmError(f"duplicate candidate id: {candidate_id!r}")
        seen.add(candidate_id)
        span = row.get("span")
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
        ):
            raise SimplifyArmError(f"candidate {candidate_id} has an invalid span")
        try:
            vector = KVector(**delta)
        except TypeError as exc:
            raise SimplifyArmError(f"candidate {candidate_id} has an invalid delta") from exc
        selected.append(
            ScoredCandidate(
                candidate=Candidate(
                    id=candidate_id,
                    layer=str(row.get("layer") or "L2"),
                    kind=str(row.get("kind") or "unknown"),
                    component=str(row.get("component") or "unknown"),
                    path=str(row.get("path") or ""),
                    span=(span[0], span[1]),
                    symbol=str(row.get("symbol") or candidate_id),
                ),
                delta=vector,
            )
        )
    if not selected:
        raise SimplifyArmError("candidate pool has no unblocked candidates with deltas")
    return tuple(selected)


def _prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SimplifyArmError(f"{path} already exists; pass force=True")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copy_subject(source: Path, target: Path) -> None:
    if target.exists():
        raise SimplifyArmError(f"candidate workspace already exists: {target}")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".pytest_cache"),
    )


def _batch_candidate(spec: SimplifySpec) -> Candidate:
    return Candidate(
        id=f"batch-{spec.arm.lower()}-{spec.round:02d}",
        layer="L2",
        kind="batch_delete",
        component="batch",
        path="",
        span=(1, 1),
        symbol="batch",
    )


def _payload(
    spec: SimplifySpec,
    batch: tuple[ScoredCandidate, ...],
    before: KVector,
    after: KVector | None,
    report: VerifyReport | None,
    decision: CandidateDecision,
    target: int,
    evaluated: Sequence[TrialOutcome] = (),
) -> dict[str, object]:
    return {
        "schema": 1,
        "spec": asdict(spec),
        "batch": [
            {
                "id": row.candidate.id,
                "component": row.candidate.component,
                "layer": row.candidate.layer,
                "path": row.candidate.path,
                "span": list(row.candidate.span),
                "delta": row.delta.as_dict(),
            }
            for row in batch
        ],
        "target_k_ast": target,
        "before": before.as_dict(),
        "after": after.as_dict() if after is not None else None,
        "delta": (before - after).as_dict() if after is not None else None,
        "verify": report.as_dict() if report is not None else None,
        "decision": decision.as_dict(),
        "evaluated": [asdict(row) for row in evaluated],
    }


def run_simplify_arm(
    spec: SimplifySpec,
    runtime: SimplifyRuntime,
    *,
    candidates: Sequence[ScoredCandidate],
    subject_root: str | Path,
    reference_outcomes: Sequence[TrialOutcome],
    tasks: Sequence[str],
    holdout_tasks: Sequence[str],
    work_dir: str | Path,
    out_dir: str | Path,
    force: bool = False,
) -> dict[str, object]:
    """Apply one simplify batch, verify it, evaluate it, and record the verdict."""
    if not isinstance(spec, SimplifySpec):
        raise SimplifyArmError("spec must be a SimplifySpec")
    if not isinstance(runtime, SimplifyRuntime):
        raise SimplifyArmError("runtime must be a SimplifyRuntime")
    source = Path(subject_root)
    if not source.is_dir():
        raise SimplifyArmError(f"subject root is not a directory: {subject_root}")
    task_ids = tuple(tasks)
    holdout_ids = tuple(holdout_tasks)
    if not task_ids or not holdout_ids:
        raise SimplifyArmError("tasks and holdout_tasks must both be non-empty")
    overlap = sorted(set(task_ids) & set(holdout_ids))
    if overlap:
        raise SimplifyArmError(f"holdout overlaps search tasks: {', '.join(overlap)}")

    try:
        for label, ids in (("search", task_ids), ("holdout", holdout_ids)):
            subset = [row for row in reference_outcomes if row.task_id in ids]
            summary = summarize_trials(subset, tasks=ids, k=spec.k)
            if summary["validity"]["status"] != "pass":
                reasons = "; ".join(summary["validity"]["reasons"])
                raise SimplifyArmError(
                    f"reference {label} outcomes are invalid: {reasons}"
                )
    except BaselineError as exc:
        raise SimplifyArmError(f"reference outcomes: {exc}") from exc

    if spec.target_k_ast is not None:
        target = spec.target_k_ast
    else:
        if spec.arm != "SIM":
            raise SimplifyArmError(
                f"{spec.arm} requires an explicit target_k_ast"
            )
        smart = plan_batch(candidates, mode="static")
        target = sum(row.delta.k_ast for row in smart)
    batch_mode = "static" if spec.mode in ("static", "census", "full") else spec.mode
    batch = plan_batch(
        candidates,
        mode=batch_mode,
        target_k_ast=None if batch_mode == "static" else target,
        seed=spec.batch_seed,
    )
    if not batch:
        raise SimplifyArmError("batch selection returned no candidates")

    work = Path(work_dir)
    out = Path(out_dir)
    _prepare(work, force)
    _prepare(out, force)
    candidate_root = work / "batch" / source.name
    _copy_subject(source, candidate_root)
    ledger = Ledger(out / "ledger.jsonl", run_id=spec.run_id)
    store = RunStore(out)
    batch_candidate = _batch_candidate(spec)

    try:
        before = runtime.measure(candidate_root)
    except Exception as exc:  # noqa: BLE001 - a broken baseline is an error
        raise SimplifyArmError(f"before measurement failed: {exc}") from exc
    if not isinstance(before, KVector):
        raise SimplifyArmError("measure returned a non-KVector")

    try:
        changed = tuple(str(path) for path in runtime.apply(candidate_root, batch))
    except Exception as exc:  # noqa: BLE001 - apply failure is undeletable
        decision = CandidateDecision(
            batch_candidate.id, "L2", "undeletable", f"apply failed: {exc}", "unknown"
        )
        record_decision(
            ledger,
            decision,
            round=spec.round,
            evidence={"arm": spec.arm, "mode": spec.mode, "error": str(exc)},
        )
        payload = _payload(spec, batch, before, None, None, decision, target)
        store.write_json("simplify.json", payload)
        store.write_json("decision.json", decision.as_dict())
        return payload

    try:
        report = runtime.verify(candidate_root, changed)
    except Exception as exc:  # noqa: BLE001 - verify failure is undecidable
        decision = CandidateDecision(
            batch_candidate.id, "L2", "undecidable", f"verify failed: {exc}", "unknown"
        )
        record_decision(
            ledger,
            decision,
            round=spec.round,
            evidence={"arm": spec.arm, "mode": spec.mode, "error": str(exc)},
        )
        payload = _payload(spec, batch, before, None, None, decision, target)
        store.write_json("simplify.json", payload)
        store.write_json("decision.json", decision.as_dict())
        return payload
    if not isinstance(report, VerifyReport):
        raise SimplifyArmError("verify must return a VerifyReport")

    if report.verdict != "pass":
        decision = decide_candidate(batch_candidate.id, "L2", report, None)
        record_decision(
            ledger,
            decision,
            round=spec.round,
            evidence={"arm": spec.arm, "mode": spec.mode, "paths": list(changed)},
        )
        payload = _payload(spec, batch, before, None, report, decision, target)
        store.write_json("simplify.json", payload)
        store.write_json("decision.json", decision.as_dict())
        store.write_json("applied.json", {"paths": list(changed), "verify": report.as_dict()})
        return payload

    try:
        after = runtime.measure(candidate_root)
    except Exception as exc:  # noqa: BLE001 - a broken measurement is undecidable
        decision = CandidateDecision(
            batch_candidate.id, "L2", "undecidable", f"after measurement failed: {exc}", report.verdict
        )
        record_decision(
            ledger,
            decision,
            round=spec.round,
            evidence={"arm": spec.arm, "mode": spec.mode, "error": str(exc)},
        )
        payload = _payload(spec, batch, before, None, report, decision, target)
        store.write_json("simplify.json", payload)
        store.write_json("decision.json", decision.as_dict())
        return payload
    if not isinstance(after, KVector):
        raise SimplifyArmError("measure returned a non-KVector")

    try:
        evaluated = tuple(runtime.evaluate(candidate_root, task_ids + holdout_ids, spec.k))
    except Exception as exc:  # noqa: BLE001 - a failed evaluation is undecidable
        decision = CandidateDecision(
            batch_candidate.id, "L2", "undecidable", f"evaluation failed: {exc}", report.verdict
        )
        record_decision(
            ledger,
            decision,
            round=spec.round,
            evidence={"arm": spec.arm, "mode": spec.mode, "error": str(exc)},
        )
        payload = _payload(spec, batch, before, after, report, decision, target)
        store.write_json("simplify.json", payload)
        store.write_json("decision.json", decision.as_dict())
        return payload

    applied = {
        "before": before.as_dict(),
        "after": after.as_dict(),
        "verify": report.as_dict(),
        "undeletable": False,
        "undeletable_reason": "",
    }
    pools = {"evolution": task_ids, "holdout": holdout_ids}
    try:
        decision = decide_from_artifacts(
            pools=pools,
            candidate=batch_candidate,
            applied=applied,
            reference=reference_outcomes,
            evaluated=evaluated,
            k=spec.k,
            scores_pool="evolution",
            holdout_pool="holdout",
            round=spec.round,
            task_margin=spec.task_margin,
            margin=spec.margin,
            holdout_margin=spec.holdout_margin,
            n_boot=spec.n_boot,
            alpha=spec.alpha,
            seed=spec.seed,
        )
    except RoundError as exc:
        raise SimplifyArmError(f"cannot decide the batch: {exc}") from exc

    record_decision(
        ledger,
        decision,
        round=spec.round,
        evidence={
            "arm": spec.arm,
            "mode": spec.mode,
            "paths": list(changed),
            "batch": [row.candidate.id for row in batch],
        },
    )
    payload = _payload(
        spec, batch, before, after, report, decision, target, evaluated
    )
    store.write_json("simplify.json", payload)
    store.write_json("decision.json", decision.as_dict())
    store.write_json("applied.json", applied)
    store.write_json(
        "candidate.json",
        {
            "arm": spec.arm,
            "mode": spec.mode,
            "target_k_ast": target,
            "candidates": payload["batch"],
        },
    )
    return payload
