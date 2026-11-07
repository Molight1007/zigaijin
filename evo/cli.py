from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from evo import __version__
from evo.adapters.deepseek import DEEPSEEK_FLASH, ModelError, ModelProfile
from evo.adapters.git_worktree import WorktreeError, create_worktree, remove_worktree
from evo.adapters.llm import ChatClient, LLMError
from evo.adapters.harbor_tb21 import (
    AGENT,
    DATASET,
    HarborError,
    build_run,
    read_job,
    run_harbor,
)
from evo.adapters.store_fs import RunStore, content_hash
from evo.application.ablate import AblateRunError, run_ablations
from evo.application.apply import ApplyError, delete_candidate, load_observations
from evo.application.armrun import (
    ARM_POOLS,
    ArmRunError,
    build_arm_payload,
    check_frozen_subject,
    plan_arm_run,
    pool_ids,
)
from evo.application.baseline import BaselineError, TrialOutcome, build_baseline
from evo.application.compare import CompareError, compare_arms
from evo.application.diagnose import (
    DIAGNOSE_SYSTEM,
    DiagnoseError,
    DiagnosisComponent,
    diagnose,
    merge_scores,
    parse_cases,
)
from evo.application.faultcase import (
    FaultCaseError,
    build_case,
    cases_payload,
    read_failure,
)
from evo.application.faultrun import FaultRunError, plan_fault_runs
from evo.application.evolve import EvolveRoundError, evolve_round
from evo.application.freeze import FreezeError, freeze_run, tree_hash
from evo.application.power import PowerError, power_report
from evo.application.prestudy import PrestudyInputError, load_prestudy
from evo.application.propose import (
    ProposeError,
    propose,
    snapshot_tree,
    tree_diff,
)
from evo.application.report import (
    TABLES as REPORT_TABLES,
    ReportError,
    build_tables,
    render_table,
)
from evo.application.round import RoundError, decide_from_artifacts
from evo.application.runstate import (
    RunStateError,
    advance_run,
    init_run,
    load_state,
)
from evo.application.simplify import record_decision
from evo.application.simplify_effect import (
    SimplifyEffectError,
    build_simplify_effect,
)
from evo.application.simplify_arm import (
    ScoredCandidate,
    SimplifyArmError,
    SimplifyRuntime,
    SimplifySpec,
    all_unit_candidates,
    apply_batch,
    lever_agent_kwargs,
    load_scored_candidates,
    run_simplify_arm,
    select_zero_contribution_levers,
    target_k_from_candidates,
)
from evo.application.traces import TraceMergeError, merge_trace_dir
from evo.application.version_compare import (
    VersionCompareError,
    build_version_comparison,
)
from evo.application.workforce import (
    WorkforceSpec,
    default_runtime,
    run_workforce_loop,
    run_workforce_round,
)
from evo.contracts.types import Boundary, BoundarySpec, Component, KVector
from evo.domain.ablation import AblationError, attributed_k, load_ablation_plan
from evo.domain.arms import ARMS, arm_config
from evo.domain.census import (
    CensusError,
    function_units,
    load_traces,
    touch_report,
    zero_touch,
)
from evo.domain.components import ComponentError, derive_components, measure_components
from evo.domain.coreset import CoresetError
from evo.domain.faults import (
    FaultError,
    apply_mutation,
    find_mutations,
    parse_mutations,
    sample_mutations,
    verify_sources,
)
from evo.domain.ai_cleanup import AICleanupError, find_ai_cleanup
from evo.domain.incidents import (
    Incident,
    IncidentError,
    append_incident,
    read_incidents,
)
from evo.domain.kmetrics import KMeasurementError, measure_paths, measure_source
from evo.domain.ledger import Ledger, LedgerError
from evo.domain.prestudy import PrestudyError
from evo.domain.protection import ProtectionHit, ProtectionPolicy, check_protection
from evo.domain.protocol import Protocol, ProtocolError
from evo.domain.scan import Candidate, ScanError, scan_sources
from evo.domain.state import ACTORS, PHASES
from evo.domain.workforce import WorkforceError
from evo.domain.verify import (
    CheckOutcome,
    VerifyError,
    aggregate,
    compile_files,
    differential_check,
    import_check,
    replay_check,
    required_checks,
)


DEFAULT_EXCLUDES = ("**/__pycache__/**", ".venv", ".git", ".pytest_cache")
COVERAGE_AGENT = "evo.adapters.traced_terminus2:TracedAgentHarness"


def _coverage_options(coverage_dir: str | Path, subject_root: str | Path):
    target = Path(coverage_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    env = {
        "EVO_TRACE_DIR": str(target),
        "EVO_SUBJECT_ROOT": str(Path(subject_root).resolve()),
    }
    return COVERAGE_AGENT, env, str(Path(__file__).resolve().parents[1])


class CLIError(ValueError):
    pass


def _py_files(root: Path) -> list[str]:
    if not root.is_dir():
        raise CLIError(f"root is not a directory: {root}")
    files = []
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        if any(part in {".venv", ".git", ".pytest_cache", "__pycache__"} or part.startswith(".") for part in parts):
            continue
        files.append(path.relative_to(root).as_posix())
    return sorted(files)


def _boundaries(values: list[str] | None, subject: str) -> tuple[BoundarySpec, list[Boundary]]:
    if values:
        boundaries = []
        for value in values:
            name, separator, patterns = value.partition("=")
            if not separator or not name or not patterns:
                raise CLIError(f"boundary must look like name=pattern[,pattern]: {value!r}")
            boundaries.append(
                Boundary(name=name, patterns=tuple(part.strip() for part in patterns.split(",")))
            )
        return BoundarySpec(subject, tuple(boundaries), DEFAULT_EXCLUDES), boundaries
    boundary = Boundary("all", ("**/*.py",))
    return BoundarySpec(subject, (boundary,), DEFAULT_EXCLUDES), [boundary]


def _read_sources(root: Path, component: Component) -> dict[str, str]:
    base = root.resolve()
    sources: dict[str, str] = {}
    for rel in component.paths:
        target = (base / rel).resolve()
        if not target.is_relative_to(base):
            raise CLIError(f"component path escapes root: {rel}")
        try:
            sources[rel] = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CLIError(f"cannot read {rel}: {exc}") from exc
    return sources


def _span_without(source: str, span: tuple[int, int]) -> str | None:
    lines = source.splitlines(keepends=True)
    start, end = span
    if start < 1 or end < start or end > len(lines):
        return None
    return "".join(lines[: start - 1] + lines[end:])


def _delta_for(source: str, span: tuple[int, int]) -> KVector | None:
    updated = _span_without(source, span)
    if updated is None:
        return None
    try:
        return measure_source(source) - measure_source(updated)
    except KMeasurementError:
        return None


def _candidate_dict(
    candidate: Candidate, delta: KVector | None = None, include_delta: bool = False
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": candidate.id,
        "layer": candidate.layer,
        "kind": candidate.kind,
        "component": candidate.component,
        "path": candidate.path,
        "span": list(candidate.span),
        "symbol": candidate.symbol,
        "evidence": dict(candidate.evidence),
        "blocked": candidate.blocked,
        "protection": [
            {"kind": hit.kind, "start": hit.start, "end": hit.end, "reason": hit.reason}
            for hit in candidate.protection
        ],
    }
    if include_delta:
        payload["applicable"] = delta is not None
        payload["delta"] = delta.as_dict() if delta is not None else None
    return payload


def _sum_additive(vectors: list[KVector]) -> KVector:
    return KVector(
        k_lines_nb=sum(vector.k_lines_nb for vector in vectors),
        k_ast=sum(vector.k_ast for vector in vectors),
        k_chars=sum(vector.k_chars for vector in vectors),
        k_bytes=sum(vector.k_bytes for vector in vectors),
        k_gzip=0,
        k_cc=sum(vector.k_cc for vector in vectors),
    )


def _prestudy_payload(root: Path, spec: BoundarySpec) -> dict[str, object]:
    components = derive_components(root, spec)
    measured = measure_components(root, components)
    all_paths = sorted({path for component in components for path in component.paths})
    totals = measure_paths(root, all_paths)

    candidate_rows: list[dict[str, object]] = []
    eligible_with_delta: list[tuple[str, tuple[int, int], KVector]] = []
    component_rows = []
    for component in components:
        sources = _read_sources(root, component)
        candidates = scan_sources(component, sources, ProtectionPolicy())
        for candidate in candidates:
            delta = _delta_for(sources[candidate.path], candidate.span)
            candidate_rows.append(_candidate_dict(candidate, delta, include_delta=True))
            if not candidate.blocked and delta is not None:
                eligible_with_delta.append((candidate.path, candidate.span, delta))
        component_rows.append(
            {
                "id": component.id,
                "paths": list(component.paths),
                "k": measured[component.id].as_dict(),
            }
        )

    eligible_with_delta.sort(key=lambda item: (item[0], item[1]))
    selected: list[KVector] = []
    last_end: dict[str, int] = {}
    for path, span, delta in eligible_with_delta:
        if span[0] <= last_end.get(path, 0):
            continue
        selected.append(delta)
        last_end[path] = span[1]

    removed = _sum_additive(selected)
    reduction_pct = {}
    for name in ("k_lines_nb", "k_ast", "k_chars", "k_cc"):
        before = getattr(totals, name)
        reduction_pct[name] = round(100 * getattr(removed, name) / before, 4) if before else 0.0

    total_count = len(candidate_rows)
    blocked_count = sum(bool(row["blocked"]) for row in candidate_rows)
    applicable_count = sum(bool(row["applicable"]) for row in candidate_rows)
    return {
        "subject": spec.subject,
        "root": str(root.resolve()),
        "static_only": True,
        "boundaries": {
            "subject": spec.subject,
            "boundaries": [
                {"name": boundary.name, "patterns": list(boundary.patterns)}
                for boundary in spec.boundaries
            ],
            "exclude": list(spec.exclude),
        },
        "components": sorted(component_rows, key=lambda row: row["id"]),
        "totals": totals.as_dict(),
        "l1": {
            "candidate_count": total_count,
            "blocked_count": blocked_count,
            "eligible_count": total_count - blocked_count,
            "applicable_count": applicable_count,
            "selected_count": len(selected),
            "delta": removed.as_dict(),
            "reduction_pct": reduction_pct,
            "candidates": candidate_rows,
        },
        "decision": "pending_l2_l3",
    }


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"evo {__version__}")
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    root = Path(args.root)
    result = measure_paths(root, _py_files(root))
    payload = {
        "root": str(root.resolve()),
        "files": len(result.files),
        "total": result.as_dict(),
        "per_file": {path: vector.as_dict() for path, vector in result.files.items()},
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"files: {payload['files']}")
        for name, value in payload["total"].items():
            print(f"{name}: {value}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    root = Path(args.root)
    spec, _ = _boundaries(args.boundary, args.subject)
    components = derive_components(root, spec)
    candidates = []
    for component in components:
        sources = _read_sources(root, component)
        candidates.extend(scan_sources(component, sources, ProtectionPolicy()))
    candidates.sort(key=lambda candidate: (candidate.component, candidate.path, candidate.span, candidate.kind))
    blocked = sum(candidate.blocked for candidate in candidates)
    payload = {
        "subject": args.subject,
        "components": [component.id for component in components],
        "candidates": [_candidate_dict(candidate) for candidate in candidates],
        "counts": {
            "total": len(candidates),
            "blocked": blocked,
            "eligible": len(candidates) - blocked,
        },
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"components: {len(components)}, candidates: {len(candidates)}, eligible: {len(candidates) - blocked}")
        for candidate in candidates:
            state = "blocked" if candidate.blocked else "eligible"
            print(f"{candidate.id} {candidate.kind:16s} {candidate.path}:{candidate.span[0]}-{candidate.span[1]} {candidate.symbol} [{state}]")
    return 0


def _cmd_ai_cleanup(args: argparse.Namespace) -> int:
    root = Path(args.root)
    findings = find_ai_cleanup(root)
    payload = {
        "root": str(root.resolve()),
        "count": len(findings),
        "findings": [row.as_dict() for row in findings],
    }
    if args.out:
        out = Path(args.out)
        RunStore(out.parent).write_json(out.name, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"findings: {len(findings)}")
        for row in findings:
            print(f"{row.kind:20s} {row.path}:{row.span[0]} {row.symbol} [{row.action}]")
    return 0


def _cmd_prestudy_static(args: argparse.Namespace) -> int:
    root = Path(args.root)
    spec, _ = _boundaries(args.boundary, args.subject)
    payload = _prestudy_payload(root, spec)
    store = RunStore(Path(args.out))
    store.write_json("prestudy.static.json", payload)
    print(str(Path(args.out) / "prestudy.static.json"))
    return 0


def _cmd_prestudy_decide(args: argparse.Namespace) -> int:
    payload = load_prestudy(
        Path(args.static),
        Path(args.census),
        Path(args.l3) if args.l3 else None,
    )
    store = RunStore(Path(args.out))
    store.write_json("prestudy.json", payload)
    print(str(Path(args.out) / "prestudy.json"))
    return 0


def _load_outcomes(path: Path) -> list[TrialOutcome]:
    outcome_path = Path(path)
    if not outcome_path.exists():
        raise CLIError(f"outcomes file does not exist: {outcome_path}")
    records = []
    for number, line in enumerate(
        outcome_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CLIError(f"{outcome_path} line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise CLIError(f"{outcome_path} line {number}: expected a JSON object")
        try:
            records.append(
                TrialOutcome(
                    task_id=row.get("task_id"),
                    trial_id=row.get("trial_id"),
                    status=row.get("status", "completed"),
                    reward=row.get("reward"),
                    tokens_in=row.get("tokens_in"),
                    tokens_out=row.get("tokens_out"),
                    cost_usd=row.get("cost_usd"),
                    failure_cause=row.get("failure_cause"),
                )
            )
        except BaselineError as exc:
            raise CLIError(f"{outcome_path} line {number}: {exc}") from exc
    return records


def _cmd_baseline(args: argparse.Namespace) -> int:
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise CLIError("--tasks must list at least one task")
    records = _load_outcomes(Path(args.outcomes))
    root = Path(args.root)
    spec, _ = _boundaries(args.boundary, args.subject)
    components = derive_components(root, spec)
    paths = sorted({path for component in components for path in component.paths})
    if not paths:
        raise CLIError(f"no Python files under subject {args.subject!r}")
    report = build_baseline(
        records,
        tasks=tasks,
        k=args.reps,
        k_vector=measure_paths(root, paths),
    )
    payload = {
        "subject": args.subject,
        "arm": "H0",
        "root": str(root.resolve()),
        "task_ids": tasks,
        **report,
    }
    store = RunStore(Path(args.out))
    store.write_json("baseline.json", payload)
    print(str(Path(args.out) / "baseline.json"))
    return 0


CANDIDATE_FIELDS = ("id", "layer", "kind", "component", "path", "span", "symbol")


def _candidate_text(payload: dict[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value.strip():
        raise CLIError(f"candidate field {name!r} must be a non-empty string")
    return value


def _candidate_int(row: dict[str, object], name: str, where: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CLIError(f"{where} field {name!r} must be an integer")
    return value


def _load_candidate(path: Path) -> tuple[Candidate, dict[str, object]]:
    if not path.is_file():
        raise CLIError(f"candidate file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"candidate file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise CLIError("candidate file must contain a JSON object")
    missing = [name for name in CANDIDATE_FIELDS if name not in payload]
    if missing:
        raise CLIError(f"candidate is missing fields: {', '.join(missing)}")

    span = payload["span"]
    if (
        not isinstance(span, (list, tuple))
        or len(span) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
    ):
        raise CLIError(f"span must be a [start, end] integer pair: {span!r}")
    start, end = span
    if start < 1 or end < start:
        raise CLIError(f"span must be positive with start <= end: {span!r}")

    evidence = payload.get("evidence", {})
    if not isinstance(evidence, dict):
        raise CLIError("candidate evidence must be a JSON object")
    rows = payload.get("protection", [])
    if not isinstance(rows, list):
        raise CLIError("candidate protection must be a JSON array")
    hits = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CLIError(f"candidate protection[{index}] must be a JSON object")
        hits.append(
            ProtectionHit(
                kind=_candidate_text(row, "kind"),
                path=_candidate_text(row, "path"),
                start=_candidate_int(row, "start", f"protection[{index}]"),
                end=_candidate_int(row, "end", f"protection[{index}]"),
                reason=_candidate_text(row, "reason"),
            )
        )
    if payload.get("blocked") and not hits:
        raise CLIError("candidate is marked blocked but has no protection hits")

    candidate = Candidate(
        id=_candidate_text(payload, "id"),
        layer=_candidate_text(payload, "layer"),
        kind=_candidate_text(payload, "kind"),
        component=_candidate_text(payload, "component"),
        path=_candidate_text(payload, "path"),
        span=(start, end),
        symbol=_candidate_text(payload, "symbol"),
        evidence=evidence,
        protection=tuple(hits),
    )
    return candidate, payload


def _cmd_apply(args: argparse.Namespace) -> int:
    candidate, raw = _load_candidate(Path(args.candidate))
    paths = list(args.path or ())
    modules = list(args.module or ())
    if not paths:
        raise CLIError("--path must list at least one subject-relative file")
    if not modules:
        raise CLIError("--module must list at least one importable module")
    worktree = create_worktree(
        Path(args.repo),
        Path(args.worktree),
        base=args.base,
        include=tuple(args.include or ()),
    )
    out = Path(args.out)
    extra_checks = ()
    if args.observations:
        observations = load_observations(Path(args.observations))
        extra_checks = (
            replay_check(observations, minimum=10),
            differential_check(observations),
        )
    try:
        result = delete_candidate(
            worktree,
            candidate,
            paths=paths,
            modules=modules,
            subject_root=args.subject_root,
            import_root=args.import_root,
            python=args.python,
            extra_checks=extra_checks,
        )
        delta = (
            result.before - result.after if result.after is not None else None
        )
        store = RunStore(out)
        store.write_text("diff.patch", result.diff)
        store.write_json(
            "applied.json",
            {
                "repo": str(Path(args.repo).resolve()),
                "worktree": str(Path(args.worktree).resolve()),
                "base": worktree.base,
                "candidate": raw,
                "subject_root": args.subject_root,
                "import_root": args.import_root,
                "paths": paths,
                "modules": modules,
                "baseline_import": result.baseline_import,
                "verify": result.verify.as_dict(),
                "before": result.before.as_dict(),
                "after": result.after.as_dict() if result.after is not None else None,
                "delta": delta.as_dict() if delta is not None else None,
                "undeletable": result.undeletable,
                "undeletable_reason": result.undeletable_reason,
                "observations": args.observations,
                "diff": "diff.patch",
            },
        )
    finally:
        if not args.keep:
            remove_worktree(worktree)
    print(str(out / "applied.json"))
    return 0


def _read_task_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise CLIError(f"tasks file does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CLIError(f"cannot read tasks file {path}: {exc}") from exc
    ids = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not ids:
        raise CLIError(f"task list is empty: {path}")
    return ids


def _cmd_freeze(args: argparse.Namespace) -> int:
    ids = _read_task_ids(Path(args.tasks_file))
    protocol = Protocol(
        smoke=args.smoke,
        evolution=args.evolution,
        holdout=args.holdout,
        iteration_k=args.iteration_k,
    )
    root = Path(args.root)
    files = _py_files(root)
    if not files:
        raise CLIError(f"no Python files under subject root {root}")
    payload = freeze_run(
        ids,
        dataset=args.dataset,
        subject=args.subject,
        k_vector=measure_paths(root, files),
        subject_hash=tree_hash(root, files),
        protocol=protocol,
        seed=args.seed,
        margin=args.margin,
        holdout_margin=args.holdout_margin,
    )
    store = RunStore(Path(args.out))
    store.write_json("run.json", payload)
    print(str(Path(args.out) / "run.json"))
    return 0


def _comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _pool_tasks(path: Path, pool: str) -> list[str]:
    if not path.is_file():
        raise CLIError(f"run file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"run file is not valid JSON: {exc.msg}") from exc
    pools = payload.get("pools") if isinstance(payload, dict) else None
    if not isinstance(pools, dict):
        raise CLIError(f"run file has no pools: {path}")
    if pool == "all":
        ids: list[str] = []
        for name in ("smoke", "evolution", "holdout"):
            ids.extend(pools.get(name) or [])
    else:
        ids = pools.get(pool)
    if not isinstance(ids, list) or not ids:
        raise CLIError(f"run file has no task pool {pool!r}: {path}")
    return ids


def _k_from_applied(path: Path) -> tuple[KVector, KVector]:
    if not path.is_file():
        raise CLIError(f"applied file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = KVector(**payload["before"])
        after = KVector(**payload["after"])
    except json.JSONDecodeError as exc:
        raise CLIError(f"applied file is not valid JSON: {exc.msg}") from exc
    except (KeyError, TypeError) as exc:
        raise CLIError(f"applied file has no usable K vectors: {exc}") from exc
    return before, after


def _cmd_compare(args: argparse.Namespace) -> int:
    if args.family_size <= 0:
        raise CLIError(f"--family-size must be positive: {args.family_size}")
    if bool(args.tasks) == bool(args.run):
        raise CLIError("either --tasks or --run must be given, not both")
    if args.tasks:
        tasks = _comma_list(args.tasks)
    else:
        tasks = _pool_tasks(Path(args.run), args.pool)
    if not tasks:
        raise CLIError("task list is empty")
    k_before = k_after = None
    if args.applied:
        k_before, k_after = _k_from_applied(Path(args.applied))
    result = compare_arms(
        _load_outcomes(Path(args.reference)),
        _load_outcomes(Path(args.candidate)),
        tasks=tasks,
        k=args.k,
        holdout=_comma_list(args.holdout),
        margin=args.margin,
        holdout_margin=args.holdout_margin,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
        family_size=args.family_size,
        k_before=k_before,
        k_after=k_after,
    )
    payload = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "tasks": tasks,
        "family_size": args.family_size,
        **result,
    }
    store = RunStore(Path(args.out))
    store.write_json("comparison.json", payload)
    print(str(Path(args.out) / "comparison.json"))
    return 0


def _load_run(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise CLIError(f"run file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"run file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise CLIError("run file must contain a JSON object")
    if not isinstance(payload.get("pools"), dict):
        raise CLIError(f"run file has no pools: {path}")
    if not isinstance(payload.get("prereg"), dict):
        raise CLIError(f"run file has no pre-registration: {path}")
    return payload


def _load_json_object(path: Path, name: str) -> dict[str, object]:
    if not path.is_file():
        raise CLIError(f"{name} file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"{name} file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise CLIError(f"{name} file must contain a JSON object")
    return payload


def _load_json_array(path: Path, name: str) -> list[object]:
    if not path.is_file():
        raise CLIError(f"{name} file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"{name} file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise CLIError(f"{name} file must contain a JSON array")
    return payload


def _prereg_number(
    value: float | None, prereg: dict[str, object], name: str
) -> float:
    frozen = prereg.get(name)
    if isinstance(frozen, bool) or not isinstance(frozen, (int, float)):
        raise CLIError(f"run pre-registration has no numeric {name!r}")
    if value is not None and float(value) != float(frozen):
        raise CLIError(
            f"--{name.replace('_', '-')} {value!r} does not match "
            f"the frozen pre-registration {frozen!r}"
        )
    return float(frozen)


def _prereg_int(value: int | None, prereg: dict[str, object], name: str) -> int:
    frozen = prereg.get(name)
    if isinstance(frozen, bool) or not isinstance(frozen, int) or frozen <= 0:
        raise CLIError(f"run pre-registration has no positive integer {name!r}")
    if value is not None and value != frozen:
        raise CLIError(
            f"--{name} {value!r} does not match "
            f"the frozen pre-registration {frozen!r}"
        )
    return frozen


def _cmd_round(args: argparse.Namespace) -> int:
    run = _load_run(Path(args.run))
    prereg = run["prereg"]
    margin = _prereg_number(args.margin, prereg, "margin")
    holdout_margin = _prereg_number(args.holdout_margin, prereg, "holdout_margin")
    task_margin = args.task_margin if args.task_margin is not None else margin
    k = _prereg_int(args.k, prereg, "k")
    candidate, raw = _load_candidate(Path(args.candidate))
    applied = _load_json_object(Path(args.applied), "applied")
    decision = decide_from_artifacts(
        pools=run["pools"],
        candidate=candidate,
        applied=applied,
        reference=_load_outcomes(Path(args.reference)),
        evaluated=_load_outcomes(Path(args.evaluated)),
        k=k,
        scores_pool=args.pool,
        holdout_pool=args.holdout_pool,
        round=args.round,
        protection_until=args.protection_until,
        task_margin=task_margin,
        margin=margin,
        holdout_margin=holdout_margin,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )
    run_id = args.run_id or Path(args.run).resolve().parent.name
    ledger = Ledger(Path(args.ledger), run_id=run_id)
    record_decision(
        ledger,
        decision,
        round=args.round,
        evidence={
            "run": str(Path(args.run)),
            "pool": args.pool,
            "holdout_pool": args.holdout_pool,
            "reference": str(Path(args.reference)),
            "evaluated": str(Path(args.evaluated)),
            "applied": str(Path(args.applied)),
        },
    )
    out = Path(args.out)
    store = RunStore(out)
    store.write_json(
        "decision.json",
        {
            "run": str(Path(args.run)),
            "run_id": run_id,
            "round": args.round,
            "pool": args.pool,
            "holdout_pool": args.holdout_pool,
            "candidate": raw,
            "decision": decision.as_dict(),
        },
    )
    print(str(out / "decision.json"))
    return 0


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CLIError(f"{path} line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise CLIError(f"{path} line {number}: expected a JSON object")
        rows.append(row)
    return rows


def _variant_paths(values: list[str], flag: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in values or ():
        variant, separator, path = item.partition("=")
        variant = variant.strip()
        path = path.strip()
        if not separator or not variant or not path:
            raise CLIError(f"{flag} expects VARIANT=PATH: {item!r}")
        if variant in paths:
            raise CLIError(f"duplicate variant {variant!r} for {flag}")
        paths[variant] = Path(path)
    return paths


def _load_edits(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise CLIError(f"edits file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"edits file is not valid JSON: {exc.msg}") from exc
    edits = payload.get("edits") if isinstance(payload, dict) else payload
    if not isinstance(edits, list) or any(not isinstance(item, dict) for item in edits):
        raise CLIError(f"edits file must contain a JSON array of edit objects: {path}")
    return edits


def _load_trajectory(path: Path) -> list[float]:
    if not path.is_file():
        raise CLIError(f"trajectory file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"trajectory file is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise CLIError("trajectory file must contain a JSON array")
    values = []
    for item in payload:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CLIError(f"trajectory contains a non-number: {item!r}")
        values.append(float(item))
    return values


def _variant_ints(values: list[str], flag: str) -> dict[str, int]:
    numbers: dict[str, int] = {}
    for item in values or ():
        variant, separator, value = item.partition("=")
        variant = variant.strip()
        value = value.strip()
        if not separator or not variant or not value:
            raise CLIError(f"{flag} expects VARIANT=INTEGER: {item!r}")
        try:
            number = int(value)
        except ValueError:
            raise CLIError(f"{flag} expects VARIANT=INTEGER: {item!r}") from None
        if number < 0:
            raise CLIError(f"{flag} must be non-negative: {item!r}")
        if variant in numbers:
            raise CLIError(f"duplicate variant {variant!r} for {flag}")
        numbers[variant] = number
    return numbers


def _cmd_evolve_round(args: argparse.Namespace) -> int:
    run = _load_run(Path(args.run))
    tasks = run["pools"].get("evolution") if isinstance(run.get("pools"), dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise CLIError(f"run file has no evolution pool: {args.run}")
    k = _prereg_int(args.k, run["prereg"], "k")
    candidate_paths = _variant_paths(args.candidate, "--candidate")
    if not candidate_paths:
        raise CLIError("--candidate must be given at least once")
    edits: dict[str, list[dict[str, object]]] = {}
    for variant, path in _variant_paths(args.edits, "--edits").items():
        if variant not in candidate_paths:
            raise CLIError(f"--edits names unknown variant {variant!r}")
        edits[variant] = _load_edits(path)
    k_lines = _variant_ints(args.k_lines, "--k-lines")
    unknown_k = sorted(set(k_lines) - set(candidate_paths))
    if unknown_k:
        raise CLIError(f"--k-lines names unknown variants: {', '.join(unknown_k)}")
    if k_lines and args.incumbent_k_lines is None:
        raise CLIError("--incumbent-k-lines is required when --k-lines is used")
    trajectory = _load_trajectory(Path(args.trajectory)) if args.trajectory else []
    history_path = Path(args.history)
    existing = _read_jsonl_rows(history_path)
    prior = [row for row in existing if row.get("t") == args.round]
    if prior and not args.force:
        raise CLIError(
            f"round {args.round} is already recorded in {history_path}; "
            "pass --force to replace it"
        )
    payload = evolve_round(
        incumbent=_load_outcomes(Path(args.incumbent)),
        candidates={
            variant: _load_outcomes(path)
            for variant, path in candidate_paths.items()
        },
        tasks=tasks,
        k=k,
        delta=args.delta,
        round=args.round,
        total_rounds=args.total_rounds,
        trajectory=trajectory,
        components=_comma_list(args.components),
        history=existing,
        edits=edits,
        s_star=args.s_star,
        k_lines=k_lines,
        incumbent_k_lines=args.incumbent_k_lines,
        max_k_growth=args.max_k_growth,
        n_prune=args.n_prune,
        stall_window=args.stall_window,
        m_draft=args.m_draft,
        b_min=args.b_min,
        b_max=args.b_max,
    )
    out = Path(args.out)
    store = RunStore(out)
    store.write_json(
        "decisions.json",
        {
            "run": str(Path(args.run)),
            "inputs": {
                "incumbent": str(Path(args.incumbent)),
                "candidates": {
                    variant: str(path) for variant, path in candidate_paths.items()
                },
                "edits": {
                    variant: str(path)
                    for variant, path in _variant_paths(args.edits, "--edits").items()
                },
                "history": str(history_path),
                "trajectory": args.trajectory,
            },
            **payload,
        },
    )
    if prior:
        kept = [row for row in existing if row.get("t") != args.round]
        text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in kept
        )
        RunStore(history_path.parent).write_text(history_path.name, text)
    history_store = RunStore(history_path.parent)
    for record in payload["history"]:
        history_store.append_jsonl(history_path.name, record)
    print(str(out / "decisions.json"))
    return 0


def _cmd_propose(args: argparse.Namespace) -> int:
    if bool(args.root) == bool(args.repo):
        raise CLIError("provide exactly one of --root or --repo")
    if args.repo and not args.worktree:
        raise CLIError("--repo requires --worktree")
    extensions = tuple(
        item.strip() for item in args.extensions.split(",") if item.strip()
    )
    if not extensions:
        raise CLIError("--extensions must list at least one suffix")
    components = tuple(
        item.strip() for item in (args.components or "").split(",") if item.strip()
    )
    profile = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    profile.api_key()
    evidence: list[object] = []
    for item in args.evidence or ():
        evidence.extend(_load_json_array(Path(item), "evidence"))
    history = _read_jsonl_rows(Path(args.history)) if args.history else []
    exploration = (
        _load_json_object(Path(args.exploration), "exploration")
        if args.exploration
        else {}
    )
    prune_set = (
        _load_json_array(Path(args.prune_set), "prune-set")
        if args.prune_set
        else []
    )
    worktree = None
    if args.repo:
        worktree = create_worktree(
            Path(args.repo),
            Path(args.worktree),
            base=args.base,
            include=tuple(args.include or ()),
        )
        root = worktree.path
    else:
        root = Path(args.root)
    out = Path(args.out)
    try:
        before = snapshot_tree(root, extensions=extensions)
        client = ChatClient(profile, timeout=args.timeout)

        def call(prompt: str, system: str) -> str:
            return client.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
            )

        proposal = propose(
            root=root,
            model=call,
            budget=args.budget,
            components=components,
            evidence=evidence,
            history=history,
            exploration=exploration,
            prune_set=prune_set,
            reserved=args.reserved,
            task_brief=args.task_brief or "",
            max_turns=args.max_turns,
            max_file_edits=args.max_file_edits,
            extensions=extensions,
        )
        after = snapshot_tree(root, extensions=extensions)
        patch = tree_diff(before, after)
        changed = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
        store = RunStore(out)
        store.write_text("diff.patch", patch)
        store.write_json(
            "proposal.json",
            {
                "run": out.name,
                "root": str(Path(root).resolve()),
                "repo": str(Path(args.repo).resolve()) if args.repo else None,
                "base": worktree.base if worktree is not None else None,
                "model": profile.redacted(),
                "budget": args.budget,
                "components": list(components),
                "reserved": bool(args.reserved),
                "status": proposal.status,
                "summary": proposal.summary,
                "edits": list(proposal.edits),
                "n_file_edits": proposal.n_file_edits,
                "files": {
                    "changed": changed,
                    "added": sorted(set(after) - set(before)),
                    "deleted": sorted(set(before) - set(after)),
                },
                "diff": "diff.patch",
                "transcript": list(proposal.transcript),
            },
        )
    finally:
        if worktree is not None and not args.keep:
            remove_worktree(worktree)
    print(str(out / "proposal.json"))
    return 0


def _cmd_version_compare(args: argparse.Namespace) -> int:
    tasks = _comma_list(args.tasks)
    if not tasks:
        raise CLIError("--tasks must list at least one task")
    holdout = _comma_list(args.holdout)
    vectors = {}
    if args.previous_applied:
        vectors["previous"] = _load_k_pair(Path(args.previous_applied))[1]
    if args.candidate_applied:
        vectors["candidate"] = _load_k_pair(Path(args.candidate_applied))[1]
    payload = build_version_comparison(
        {
            "H0": _load_outcomes(Path(args.h0)),
            "previous": _load_outcomes(Path(args.previous)),
            "candidate": _load_outcomes(Path(args.candidate)),
        },
        tasks=tasks,
        holdout=holdout,
        k=args.k,
        margin=args.margin,
        holdout_margin=args.holdout_margin,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
        k_vectors=vectors,
    )
    out = Path(args.out)
    store = RunStore(out.parent)
    store.write_json(out.name, payload)
    if args.csv:
        arms = payload["arms"]
        lines = ["task," + ",".join(f"{arm}_mean_reward" for arm in arms)]
        for task in tasks:
            row = payload["per_task"][task]
            values = ",".join(str(row[arm]["mean_reward"]) for arm in arms)
            lines.append(f"{task},{values}")
        csv_path = Path(args.csv)
        RunStore(csv_path.parent).write_text(csv_path.name, "\n".join(lines) + "\n")
    print(str(out))
    print(payload["verdict"])
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise CLIError("--tasks must list at least one task")
    if args.k <= 0:
        raise CLIError(f"--k must be positive: {args.k}")
    report = power_report(
        _load_outcomes(Path(args.reference)),
        _load_outcomes(Path(args.candidate)),
        tasks=tasks,
        k=args.k,
        margin=args.margin,
        alpha=args.alpha,
        power=args.power,
    )
    store = RunStore(Path(args.out))
    store.write_json("power.json", report)
    print(str(Path(args.out) / "power.json"))
    return 0


def _cmd_ablate_run(args: argparse.Namespace) -> int:
    manifest = _load_run(Path(args.run))
    prereg = manifest["prereg"]
    k_reps = prereg.get("k")
    if isinstance(k_reps, bool) or not isinstance(k_reps, int) or k_reps <= 0:
        raise CLIError(f"run.json prereg.k must be a positive integer: {k_reps!r}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise CLIError(f"run.json has no dataset: {args.run}")
    subject = manifest.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        raise CLIError(f"run.json has no subject: {args.run}")
    root = Path(args.subject_root)
    if root.name != AGENT.split(":", 1)[0]:
        raise CLIError(
            f"subject root must be a directory named {AGENT.split(':', 1)[0]!r}: {root}"
        )
    files = _py_files(root)
    if not files:
        raise CLIError(f"no Python files under subject root {root}")
    ablation_plan = load_ablation_plan(Path(args.plan))
    if args.subject is not None and args.subject != ablation_plan.subject:
        raise CLIError(
            f"subject mismatch: plan is {ablation_plan.subject!r}, got {args.subject!r}"
        )
    tasks = pool_ids(args.pool, manifest["pools"])
    reference = _load_outcomes(Path(args.reference))
    profile = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    profile.api_key()
    out = Path(args.out)
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else out / "jobs"
    log_dir = Path(args.log_dir) if args.log_dir else out / "logs"

    def evaluate_lever(lever):
        job_name = f"{args.job_prefix}-{lever.id}"
        plan = plan_arm_run(
            arm=args.arm,
            pool=args.pool,
            pools=manifest["pools"],
            k_reps=k_reps,
            subject=subject,
            subject_root=root,
            k=measure_paths(root, files),
        )
        overrides = tuple(
            f"{flag}={json.dumps(value)}"
            for flag, value in ablation_plan.kwargs_for(lever.id).items()
        )
        pythonpath = [str(Path(root).resolve().parent)]
        if args.harbor_src:
            pythonpath.append(str(Path(args.harbor_src).resolve()))
        pythonpath.extend(args.pythonpath or ())
        command = build_run(
            python=args.python,
            dataset=dataset,
            task_ids=plan.task_ids,
            model=profile,
            jobs_dir=str(jobs_dir),
            job_name=job_name,
            reps=plan.k_reps,
            concurrency=args.concurrency,
            config=args.config,
            pythonpath=pythonpath,
            environment_build_timeout_multiplier=getattr(
                args, "environment_build_timeout_multiplier", None
            ),
            kwargs=("record_terminal_session=false", *overrides),
        )
        run_harbor(
            command.argv,
            log_path=log_dir / f"{job_name}.log",
            env=command.env,
            timeout=args.timeout,
        )
        return [record.to_outcome() for record in read_job(jobs_dir, job_name)]

    payload = run_ablations(
        ablation_plan,
        evaluate_lever,
        reference=reference,
        tasks=tasks,
        k=k_reps,
        margin=args.margin,
        alpha=args.alpha,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    store = RunStore(out)
    store.write_json("ablation.json", payload)
    print(str(out / "ablation.json"))
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    path = Path(args.cases)
    if not path.is_file():
        raise CLIError(f"cases file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CLIError(f"cases file is not valid JSON: {exc.msg}") from exc
    cases = parse_cases(payload)
    profile = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    profile.api_key()
    client = ChatClient(profile, timeout=args.timeout)

    def rank(prompt: str) -> str:
        return client.complete(
            [
                {"role": "system", "content": DIAGNOSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
        )

    result = diagnose(
        cases,
        rank,
        repeats=args.repeats,
        seed=args.shuffle_seed,
        max_trace_chars=args.max_trace_chars,
    )
    out = Path(args.out)
    store = RunStore(out)
    store.write_json("score.json", {args.phase: result["metrics"]})
    store.write_json(
        "details.json",
        {
            **result,
            "phase": args.phase,
            "seed": args.shuffle_seed,
            "cases": str(path.resolve()),
            "model": profile.redacted(),
        },
    )
    print(str(out / "details.json"))
    return 0


def _cmd_diagnose_merge(args: argparse.Namespace) -> int:
    before = _load_json_object(Path(args.before), "before")
    after = _load_json_object(Path(args.after), "after")
    payload = merge_scores(before, after)
    out = Path(args.out)
    RunStore(out.parent).write_json(out.name, payload)
    print(str(out))
    return 0


def _cmd_faults(args: argparse.Namespace) -> int:
    base = Path(args.root).resolve()
    if not base.is_dir():
        raise CLIError(f"root is not a directory: {base}")
    paths = list(args.path or ())
    if not paths:
        raise CLIError("--path must list at least one file")
    if args.per_file <= 0:
        raise CLIError("--per-file must be positive")
    sources: dict[str, str] = {}
    sampled = []
    for value in paths:
        raw = str(value).replace("\\", "/")
        pure = PurePosixPath(raw)
        if (
            not raw
            or pure.is_absolute()
            or Path(raw).is_absolute()
            or ".." in pure.parts
        ):
            raise CLIError(f"path must stay inside the root: {value!r}")
        target = (base / Path(*pure.parts)).resolve()
        if not target.is_relative_to(base):
            raise CLIError(f"path escapes the root: {value!r}")
        if not target.is_file():
            raise CLIError(f"path does not exist: {value!r}")
        try:
            source = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CLIError(f"cannot read {value!r}: {exc}") from exc
        relative = pure.as_posix()
        mutations = find_mutations(source, path=relative)
        sampled.extend(
            sample_mutations(mutations, count=args.per_file, seed=args.seed)
        )
        sources[relative] = content_hash(target.read_bytes())
    if not sampled:
        raise CLIError("no mutable sites found in the requested files")
    sampled.sort(key=lambda m: (m.path, m.line, m.edits[0].start))
    store = RunStore(Path(args.out))
    store.write_json(
        "faults.json",
        {
            "schema": 1,
            "root": str(base),
            "seed": args.seed,
            "per_file": args.per_file,
            "sources": sources,
            "mutations": [mutation.as_dict() for mutation in sampled],
        },
    )
    print(str(Path(args.out) / "faults.json"))
    return 0


def _read_ledger_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise CLIError(f"ledger does not exist: {path}")
    rows: list[dict[str, object]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CLIError(f"{path} line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise CLIError(f"{path} line {number}: expected a JSON object")
        rows.append(row)
    return rows


def _comparison_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in values or ():
        arm, separator, path = item.partition("=")
        arm = arm.strip()
        path = path.strip()
        if not separator or not arm or not path:
            raise CLIError(f"--comparison expects ARM=PATH: {item!r}")
        if arm not in ARMS:
            raise CLIError(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
        if arm in paths:
            raise CLIError(f"duplicate comparison arm {arm!r}")
        paths[arm] = Path(path)
    return paths


def _input_record(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise CLIError(f"cannot read input {resolved}: {exc}") from exc
    return {"path": str(resolved), "sha256": content_hash(data)}


def _cmd_fault_cases(args: argparse.Namespace) -> int:
    base = Path(args.jobs_dir)
    if args.job_name:
        job_dirs = [base / args.job_name]
    else:
        job_dirs = sorted({path.parent.parent for path in base.glob("*/*/result.json")})
        if not job_dirs:
            raise CLIError(f"no job results found under {base}")
    for job in job_dirs:
        if not job.is_dir():
            raise CLIError(f"job directory not found: {job}")
    truth = _load_json_object(Path(args.truth), "truth")
    if not truth:
        raise CLIError("truth map is empty")
    for trial, culprit in truth.items():
        if not isinstance(trial, str) or not trial.strip():
            raise CLIError(f"truth trial name must be a string: {trial!r}")
        if not isinstance(culprit, str) or not culprit.strip():
            raise CLIError(f"truth culprit for {trial!r} must be a string: {culprit!r}")
    root = Path(args.subject_root)
    spec, _ = _boundaries(args.boundary, args.subject)
    components = derive_components(root, spec)
    measured = measure_components(root, components)
    diagnosis = [
        DiagnosisComponent(
            id=component.id,
            description=component.description
            or f"{component.name}: {', '.join(component.paths)}",
            k_lines=measured[component.id].k_lines_nb,
        )
        for component in components
    ]
    ids = {component.id for component in diagnosis}
    for culprit in truth.values():
        if culprit not in ids:
            raise CLIError(f"culprit {culprit!r} is not in the components: {sorted(ids)}")

    trials: dict[str, Path] = {}
    for job in job_dirs:
        for result_path in sorted(job.glob("*/result.json")):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CLIError(f"cannot read {result_path}: {exc}") from exc
            name = payload.get("trial_name") if isinstance(payload, dict) else None
            if isinstance(name, str) and name.strip():
                trials[name] = result_path.parent

    cases = []
    skipped = []
    for trial, culprit in truth.items():
        if trial not in trials:
            skipped.append({"trial": trial, "reason": "trial not found"})
            continue
        failure = read_failure(
            trials[trial], max_steps=args.max_steps, max_chars=args.max_chars
        )
        if not failure.failed:
            skipped.append({"trial": trial, "reason": "trial did not fail"})
            continue
        cases.append(
            build_case(
                trial,
                failure,
                culprit=culprit,
                components=diagnosis,
                category=args.category,
            )
        )
    if not cases:
        raise CLIError("no failure cases were built; every trial was skipped")
    payload = {
        **cases_payload(cases),
        "subject": args.subject,
        "subject_root": str(root.resolve()),
        "skipped": skipped,
    }
    store = RunStore(Path(args.out))
    store.write_json("cases.json", payload)
    print(str(Path(args.out) / "cases.json"))
    return 0


def _cmd_fault_run(args: argparse.Namespace) -> int:
    payload = _load_json_object(Path(args.faults), "faults")
    mutations = parse_mutations(payload)
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise CLIError("fault plan has no sources")
    subject_root = Path(args.subject_root)
    verify_sources(subject_root, sources)
    unverified = sorted({mutation.path for mutation in mutations} - set(sources))
    if unverified:
        raise CLIError(f"mutation path was not checked against the source: {', '.join(unverified)}")
    spec, _ = _boundaries(args.boundary, args.subject)
    components = derive_components(subject_root, spec)
    jobs = plan_fault_runs(
        mutations,
        subject_root=subject_root,
        work_root=Path(args.work),
        job_prefix=args.job_prefix,
        components=components,
    )
    tasks = _comma_list(args.tasks)
    if not tasks:
        raise CLIError("--tasks must list at least one task")
    model = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    commands = []
    for job in jobs:
        pythonpath = [job.workdir]
        if args.harbor_src:
            pythonpath.append(str(Path(args.harbor_src).resolve()))
        pythonpath.extend(args.pythonpath or ())
        commands.append(
            (
                job,
                build_run(
                    python=args.python,
                    dataset=args.dataset,
                    task_ids=tasks,
                    model=model,
                    jobs_dir=args.jobs_dir,
                    job_name=job.job_name,
                    reps=1,
                    concurrency=args.concurrency,
                    config=args.config,
                    pythonpath=pythonpath,
            environment_build_timeout_multiplier=getattr(
                args, "environment_build_timeout_multiplier", None
            ),
                ),
            )
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "jobs": [
                        {
                            **job.as_dict(),
                            "argv": list(command.argv),
                            "env": dict(command.env),
                        }
                        for job, command in commands
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    for job, _ in commands:
        job_path = Path(args.jobs_dir) / job.job_name
        if job_path.exists():
            raise CLIError(f"job directory already exists: {job_path}")
        if Path(job.subject_root).parent.exists():
            raise CLIError(f"work directory already exists: {Path(job.subject_root).parent}")
    model.api_key()
    by_id = {mutation.id: mutation for mutation in mutations}
    truth: dict[str, str] = {}
    runs: list[dict[str, object]] = []
    store = RunStore(Path(args.out))
    for job, command in commands:
        dest = Path(job.subject_root)
        dest.parent.mkdir(parents=True, exist_ok=False)
        shutil.copytree(
            subject_root,
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        target = (dest / Path(job.path)).resolve()
        if not target.is_relative_to(dest):
            raise CLIError(f"mutation path escapes the subject copy: {job.path!r}")
        try:
            source = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CLIError(f"cannot read {job.path!r}: {exc}") from exc
        try:
            target.write_text(
                apply_mutation(source, by_id[job.mutation_id]),
                encoding="utf-8",
                newline="\n",
            )
        except OSError as exc:
            raise CLIError(f"cannot write {job.path!r}: {exc}") from exc
        run_harbor(
            command.argv,
            log_path=Path(args.log_dir) / f"{job.job_name}.log",
            env=command.env,
            timeout=args.timeout,
        )
        records = read_job(Path(args.jobs_dir), job.job_name)
        if not records:
            raise CLIError(f"job {job.job_name} produced no trials")
        missing = sorted({task for task in tasks} - {record.task_id for record in records})
        if missing:
            raise CLIError(
                f"job {job.job_name} is missing trials for: {', '.join(missing)}"
            )
        for record in records:
            if record.trial_id in truth and truth[record.trial_id] != job.component:
                raise CLIError(
                    f"trial {record.trial_id!r} already has a different culprit"
                )
            truth[record.trial_id] = job.component
        runs.append(
            {
                **job.as_dict(),
                "trials": [
                    {
                        "trial_id": record.trial_id,
                        "task_id": record.task_id,
                        "status": record.status,
                        "reward": record.reward,
                    }
                    for record in records
                ],
            }
        )
        store.write_json("truth.json", truth)
        store.write_json("fault-runs.json", {"schema": 1, "jobs": runs})
    print(str(Path(args.out) / "truth.json"))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run = _load_run(Path(args.run))
    comparison_paths = _comparison_paths(args.comparison)
    tables = build_tables(
        run=run,
        prestudy=_load_json_object(Path(args.prestudy), "prestudy")
        if args.prestudy
        else None,
        baseline=_load_json_object(Path(args.baseline), "baseline")
        if args.baseline
        else None,
        comparisons={
            arm: _load_json_object(path, f"comparison {arm}")
            for arm, path in comparison_paths.items()
        },
        ledger=_read_ledger_rows(Path(args.ledger)) if args.ledger else (),
        diagnosis=_load_json_object(Path(args.diagnosis), "diagnosis")
        if args.diagnosis
        else None,
    )
    inputs: dict[str, object] = {"run": _input_record(Path(args.run))}
    if args.prestudy:
        inputs["prestudy"] = _input_record(Path(args.prestudy))
    if args.baseline:
        inputs["baseline"] = _input_record(Path(args.baseline))
    if comparison_paths:
        inputs["comparisons"] = {
            arm: _input_record(path) for arm, path in sorted(comparison_paths.items())
        }
    if args.ledger:
        inputs["ledger"] = _input_record(Path(args.ledger))
    if args.diagnosis:
        inputs["diagnosis"] = _input_record(Path(args.diagnosis))
    out = Path(args.out)
    store = RunStore(out)
    manifest: dict[str, object] = {
        "schema": 1,
        "run": {
            "dataset": run.get("dataset"),
            "subject": run.get("subject"),
            "pool_hash": run.get("pool_hash"),
        },
        "inputs": inputs,
        "tables": {},
    }
    for name in REPORT_TABLES:
        table = tables[name]
        rel = f"tables/{name}.csv"
        store.write_text(rel, render_table(table))
        manifest["tables"][name] = {
            "file": rel,
            "status": table["status"],
            "rows": len(table["rows"]),
            "source": table["source"],
        }
    store.write_json("tables.json", manifest)
    print(str(out / "tables.json"))
    return 0


def _check_input_hash(
    name: str,
    record: object,
    issues: list[str],
    checked: dict[str, object],
) -> None:
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
    ):
        raise CLIError(f"tables.json input {name!r} is malformed")
    path = Path(record["path"])
    try:
        data = path.read_bytes()
    except OSError as exc:
        issues.append(f"{name}: cannot read {path}: {exc}")
        checked[name] = {"match": False, "path": str(path)}
        return
    digest = content_hash(data)
    match = digest == record["sha256"]
    if not match:
        issues.append(f"{name}: content changed since the report was written: {path}")
    checked[name] = {"match": match, "path": str(path), "sha256": digest}


def _input_path(record: object, name: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise CLIError(f"tables.json input {name!r} is malformed")
    return Path(record["path"])


def _cmd_verify_evidence(args: argparse.Namespace) -> int:
    root = Path(args.report)
    manifest = _load_json_object(root / "tables.json", "tables manifest")
    inputs = manifest.get("inputs")
    tables_meta = manifest.get("tables")
    if not isinstance(inputs, dict) or not isinstance(tables_meta, dict):
        raise CLIError("tables.json has no inputs or tables section")
    if "run" not in inputs:
        raise CLIError("tables.json has no run input")

    issues: list[str] = []
    checked: dict[str, object] = {}
    for name, record in sorted(inputs.items()):
        if name == "comparisons":
            if not isinstance(record, dict):
                raise CLIError("tables.json comparisons inputs are malformed")
            for arm, item in sorted(record.items()):
                _check_input_hash(f"comparisons[{arm}]", item, issues, checked)
        else:
            _check_input_hash(name, record, issues, checked)

    comparisons: dict[str, dict[str, object]] = {}
    raw_comparisons = inputs.get("comparisons", {})
    if not isinstance(raw_comparisons, dict):
        raise CLIError("tables.json comparisons inputs are malformed")
    for arm in sorted(raw_comparisons):
        comparisons[arm] = _load_json_object(
            _input_path(raw_comparisons[arm], f"comparisons[{arm}]"),
            f"comparison {arm}",
        )
    ledger = (
        _read_ledger_rows(_input_path(inputs["ledger"], "ledger"))
        if "ledger" in inputs
        else ()
    )
    tables = build_tables(
        run=_load_json_object(_input_path(inputs["run"], "run"), "run"),
        prestudy=(
            _load_json_object(_input_path(inputs["prestudy"], "prestudy"), "prestudy")
            if "prestudy" in inputs
            else None
        ),
        baseline=(
            _load_json_object(_input_path(inputs["baseline"], "baseline"), "baseline")
            if "baseline" in inputs
            else None
        ),
        comparisons=comparisons,
        ledger=ledger,
        diagnosis=(
            _load_json_object(_input_path(inputs["diagnosis"], "diagnosis"), "diagnosis")
            if "diagnosis" in inputs
            else None
        ),
    )

    results: dict[str, object] = {}
    for name in REPORT_TABLES:
        rel = f"tables/{name}.csv"
        try:
            stored = (root / rel).read_text(encoding="utf-8")
        except OSError as exc:
            issues.append(f"{name}: cannot read {rel}: {exc}")
            results[name] = {"file": rel, "match": False}
            continue
        match = stored == render_table(tables[name])
        results[name] = {"file": rel, "match": match}
        if not match:
            issues.append(f"{name}: table does not match a recomputation from its inputs")

    status = "pass" if not issues else "fail"
    payload = {
        "schema": 1,
        "status": status,
        "tables": results,
        "inputs": checked,
        "issues": issues,
    }
    store = RunStore(root)
    store.write_json("evidence.json", payload)
    print(str(root / "evidence.json"))
    return 0 if status == "pass" else 1


def _cmd_harbor_results(args: argparse.Namespace) -> int:
    records = read_job(Path(args.jobs_dir), args.job_name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(asdict(record.to_outcome()), ensure_ascii=False) + "\n"
            )
    print(str(out))
    return 0


def _cmd_harbor_run(args: argparse.Namespace) -> int:
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise CLIError("--tasks must list at least one task")
    model = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    pythonpath = [str(Path(args.subject_root).resolve().parent)]
    pythonpath.extend(args.pythonpath or ())
    agent = AGENT
    extra_env = None
    if args.coverage_dir:
        agent, extra_env, project_root = _coverage_options(
            args.coverage_dir, args.subject_root
        )
        pythonpath.append(project_root)
    command = build_run(
        python=args.python,
        dataset=args.dataset,
        task_ids=tasks,
        model=model,
        jobs_dir=args.jobs_dir,
        job_name=args.job_name,
        reps=args.reps,
        concurrency=args.concurrency,
        config=args.config,
        pythonpath=pythonpath,
            environment_build_timeout_multiplier=getattr(
                args, "environment_build_timeout_multiplier", None
            ),
        agent=agent,
        extra_env=extra_env,
    )
    if args.dry_run:
        print(
            json.dumps(
                {"argv": list(command.argv), "env": dict(command.env)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    job_path = Path(args.jobs_dir) / args.job_name
    if job_path.exists():
        raise CLIError(f"job directory already exists: {job_path}")
    model.api_key()
    run_harbor(
        command.argv,
        log_path=args.log,
        env=command.env,
        timeout=args.timeout,
    )
    records = read_job(Path(args.jobs_dir), args.job_name)
    print(
        json.dumps(
            {
                "job": args.job_name,
                "trials": len(records),
                "completed": sum(record.status == "completed" for record in records),
                "infra_errors": sum(record.status == "infra_error" for record in records),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    run_path = Path(args.run)
    manifest = _load_run(run_path)
    prereg = manifest["prereg"]
    k_reps = prereg.get("k")
    if isinstance(k_reps, bool) or not isinstance(k_reps, int) or k_reps <= 0:
        raise CLIError(f"run.json prereg.k must be a positive integer: {k_reps!r}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise CLIError(f"run.json has no dataset: {run_path}")
    subject = args.subject if args.subject is not None else manifest.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        raise CLIError(f"run.json has no subject: {run_path}")
    subject_root = Path(args.subject_root)
    module = AGENT.split(":", 1)[0]
    if subject_root.name != module:
        raise CLIError(
            f"subject root must be a directory named {module!r}: {subject_root}"
        )
    files = _py_files(subject_root)
    if not files:
        raise CLIError(f"no Python files under subject root {subject_root}")
    k = measure_paths(subject_root, files)
    plan = plan_arm_run(
        arm=args.arm,
        pool=args.pool,
        pools=manifest["pools"],
        k_reps=k_reps,
        subject=subject,
        subject_root=subject_root,
        k=k,
    )
    check_frozen_subject(args.arm, k, manifest.get("k"))
    pythonpath = [str(Path(subject_root).resolve().parent)]
    agent = AGENT
    extra_env = None
    if args.coverage_dir:
        agent, extra_env, project_root = _coverage_options(
            args.coverage_dir, subject_root
        )
        pythonpath.append(project_root)
    if args.harbor_src:
        pythonpath.append(str(Path(args.harbor_src).resolve()))
    pythonpath.extend(args.pythonpath or ())
    model = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    command = build_run(
        python=args.python,
        dataset=dataset,
        task_ids=plan.task_ids,
        model=model,
        jobs_dir=args.jobs_dir,
        job_name=args.job_name,
        reps=plan.k_reps,
        concurrency=args.concurrency,
        config=args.config,
        pythonpath=pythonpath,
            environment_build_timeout_multiplier=getattr(
                args, "environment_build_timeout_multiplier", None
            ),
        agent=agent,
        extra_env=extra_env,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "argv": list(command.argv),
                    "env": dict(command.env),
                    "plan": plan.as_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    job_path = Path(args.jobs_dir) / args.job_name
    if job_path.exists():
        raise CLIError(f"job directory already exists: {job_path}")
    model.api_key()
    run_harbor(
        command.argv,
        log_path=args.log,
        env=command.env,
        timeout=args.timeout,
    )
    records = read_job(Path(args.jobs_dir), args.job_name)
    outcomes = [record.to_outcome() for record in records]
    payload = build_arm_payload(plan, outcomes)
    store = RunStore(Path(args.out))
    store.write_json("arm.json", payload)
    store.write_text(
        "outcomes.jsonl",
        "".join(
            json.dumps(asdict(outcome), ensure_ascii=False) + "\n"
            for outcome in outcomes
        ),
    )
    print(str(Path(args.out) / "arm.json"))
    return 0


def _static_scored_candidates(root: Path) -> tuple[ScoredCandidate, ...]:
    subject = AGENT.split(":", 1)[0]
    spec, _ = _boundaries(None, subject)
    components = derive_components(root, spec)
    rows: list[ScoredCandidate] = []
    for component in components:
        sources = _read_sources(root, component)
        for candidate in scan_sources(component, sources, ProtectionPolicy()):
            if candidate.blocked:
                continue
            delta = _delta_for(sources[candidate.path], candidate.span)
            if delta is not None:
                rows.append(ScoredCandidate(candidate=candidate, delta=delta))
    return tuple(rows)


def _cmd_simplify(args: argparse.Namespace) -> int:
    manifest = _load_run(Path(args.run))
    prereg = manifest["prereg"]
    k_reps = prereg.get("k")
    if isinstance(k_reps, bool) or not isinstance(k_reps, int) or k_reps <= 0:
        raise CLIError(f"run.json prereg.k must be a positive integer: {k_reps!r}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise CLIError(f"run.json has no dataset: {args.run}")
    subject = manifest.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        raise CLIError(f"run.json has no subject: {args.run}")
    subject_root = Path(args.subject_root)
    if subject_root.name != AGENT.split(":", 1)[0]:
        raise CLIError(
            f"subject root must be a directory named {AGENT.split(':', 1)[0]!r}: {subject_root}"
        )
    pools = manifest["pools"]
    tasks = pool_ids(args.pool, pools)
    holdout = pool_ids(args.holdout_pool, pools)
    if args.smoke_tasks:
        smoke = tuple(
            task.strip() for task in args.smoke_tasks.split(",") if task.strip()
        )
    else:
        smoke = pool_ids("smoke", pools) if pools.get("smoke") else ()
    effective_pools = {**pools, "smoke": list(smoke)}
    reference = _load_outcomes(Path(args.reference_outcomes))

    pool: list[ScoredCandidate] = []
    if args.candidates:
        pool.extend(load_scored_candidates(Path(args.candidates)))
    if args.all_units:
        try:
            pool.extend(all_unit_candidates(subject_root))
        except SimplifyArmError as exc:
            raise CLIError(f"cannot build the all-unit pool: {exc}") from exc
    if args.mode in ("static", "full") and not args.all_units:
        try:
            pool.extend(_static_scored_candidates(subject_root))
        except (ComponentError, ScanError, KMeasurementError) as exc:
            raise CLIError(f"static scan failed: {exc}") from exc
    unique: list[ScoredCandidate] = []
    seen: set[str] = set()
    for row in pool:
        if row.candidate.id in seen:
            continue
        seen.add(row.candidate.id)
        unique.append(row)
    if not unique:
        raise CLIError(
            "no candidates: pass --candidates from `evo census`, "
            "use --all-units, or use --mode static/full"
        )

    target_k_ast = args.target_k_ast
    if target_k_ast is None and args.target_candidates:
        try:
            target_rows = load_scored_candidates(Path(args.target_candidates))
            target_k_ast = target_k_from_candidates(target_rows)
        except SimplifyArmError as exc:
            raise CLIError(f"cannot compute target K: {exc}") from exc
    if args.arm in ("PLACEBO", "SHARP") and target_k_ast is None:
        raise CLIError(
            "PLACEBO/SHARP require --target-candidates or --target-k-ast"
        )
    selected_levers: tuple[dict[str, object], ...] = ()
    lever_kwargs: tuple[str, ...] = ()
    if args.l3:
        if args.mode != "full":
            raise CLIError("--l3 is only valid with --mode full")
        l3_payload = _load_json_object(Path(args.l3), "l3")
        selected_levers = select_zero_contribution_levers(l3_payload)
        lever_kwargs = lever_agent_kwargs(selected_levers)
        if not selected_levers:
            print("warning: no zero_contribution levers found", file=sys.stderr)

    spec = SimplifySpec(
        arm=args.arm,
        mode=args.mode,
        k=k_reps,
        run_id=Path(args.run).resolve().parent.name,
        margin=args.margin,
        holdout_margin=args.holdout_margin,
        task_margin=args.task_margin,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
        target_k_ast=target_k_ast,
        batch_seed=args.batch_seed,
        levers=selected_levers,
        agent_kwargs=lever_kwargs,
    )
    profile = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    profile.api_key()
    out = Path(args.out)
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else out / "jobs"
    log_dir = Path(args.log_dir) if args.log_dir else out / "logs"
    counter = {"jobs": 0}

    def run_pool(root: Path, name: str, ids: tuple[str, ...], reps: int):
        files = _py_files(root)
        if not files:
            raise CLIError(f"no Python files under candidate root {root}")
        counter["jobs"] += 1
        job_name = f"{args.job_prefix}-{counter['jobs']:03d}"
        plan = plan_arm_run(
            arm=args.arm,
            pool=name,
            pools=effective_pools,
            k_reps=reps,
            subject=subject,
            subject_root=root,
            k=measure_paths(root, files),
        )
        if list(plan.task_ids) != list(ids):
            raise CLIError(
                f"pool {name!r} resolved to different tasks than the simplify runner expected"
            )
        pythonpath = [str(root.resolve().parent)]
        agent = AGENT
        extra_env = None
        if args.coverage_dir:
            agent, extra_env, project_root = _coverage_options(
                args.coverage_dir, root
            )
            pythonpath.append(project_root)
        if args.harbor_src:
            pythonpath.append(str(Path(args.harbor_src).resolve()))
        pythonpath.extend(args.pythonpath or ())
        command = build_run(
            python=args.python,
            dataset=dataset,
            task_ids=plan.task_ids,
            model=profile,
            jobs_dir=str(jobs_dir),
            job_name=job_name,
            reps=reps,
            concurrency=args.concurrency,
            config=args.config,
            pythonpath=pythonpath,
            environment_build_timeout_multiplier=getattr(
                args, "environment_build_timeout_multiplier", None
            ),
            agent=agent,
            extra_env=extra_env,
            kwargs=("record_terminal_session=false", *spec.agent_kwargs),
        )
        run_harbor(
            command.argv,
            log_path=log_dir / f"{job_name}.log",
            env=command.env,
            timeout=args.timeout,
        )
        return [record.to_outcome() for record in read_job(jobs_dir, job_name)]

    def evaluate_fn(root: Path, ids: tuple[str, ...], reps: int):
        selected = set(ids)
        rows = []
        for name, pool_ids_ in (("evolution", tasks), ("holdout", holdout), ("smoke", smoke)):
            if pool_ids_ and set(pool_ids_) <= selected:
                rows.extend(run_pool(root, name, pool_ids_, reps))
                selected -= set(pool_ids_)
        if selected:
            raise CLIError(f"evaluation requested tasks outside the pools: {sorted(selected)}")
        return rows

    def verify_fn(root: Path, changed: tuple[str, ...]):
        checks = [compile_files(root, changed)]
        modules = tuple(args.verify_module or (subject_root.name,))
        checks.append(import_check(root.parent, modules))
        if smoke:
            smoke_rows = evaluate_fn(root, smoke, 1)
            if any(row.status == "infra_error" for row in smoke_rows):
                smoke_status = "unknown"
            elif all(row.status == "completed" for row in smoke_rows):
                smoke_status = "pass"
            else:
                smoke_status = "fail"
            checks.append(
                CheckOutcome("smoke", smoke_status, f"{len(smoke_rows)} smoke trials")
            )
        else:
            checks.append(CheckOutcome("smoke", "unknown", "no smoke pool"))
        return aggregate(checks, required_checks("L2"))

    def measure_fn(root: Path) -> KVector:
        return measure_paths(root, _py_files(root))

    runtime = SimplifyRuntime(
        apply=apply_batch,
        verify=verify_fn,
        measure=measure_fn,
        evaluate=evaluate_fn,
    )
    try:
        payload = run_simplify_arm(
            spec,
            runtime,
            candidates=tuple(unique),
            subject_root=subject_root,
            reference_outcomes=reference,
            tasks=tasks,
            holdout_tasks=holdout,
            work_dir=args.work,
            out_dir=out,
            force=args.force,
        )
    except SimplifyArmError as exc:
        raise CLIError(str(exc)) from exc
    (out / "smoke.json").write_text(
        json.dumps({"tasks": list(smoke)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(out / "simplify.json"))
    print(payload["decision"]["verdict"])
    return 0


def _cmd_workforce(args: argparse.Namespace) -> int:
    if args.variants > 1 and not args.loop:
        raise CLIError("--variants requires --loop")
    manifest = _load_run(Path(args.run))
    prereg = manifest["prereg"]
    k_reps = prereg.get("k")
    if isinstance(k_reps, bool) or not isinstance(k_reps, int) or k_reps <= 0:
        raise CLIError(f"run.json prereg.k must be a positive integer: {k_reps!r}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        raise CLIError(f"run.json has no dataset: {args.run}")
    subject = AGENT.split(":", 1)[0]
    subject_root = Path(args.subject_root)
    if subject_root.name != subject:
        raise CLIError(f"subject root must be a directory named {subject!r}: {subject_root}")
    tasks = tuple(pool_ids(args.pool, manifest["pools"]))
    holdout = tuple(pool_ids(args.holdout_pool, manifest["pools"]))
    if args.smoke_tasks:
        smoke = tuple(
            task.strip() for task in args.smoke_tasks.split(",") if task.strip()
        )
    else:
        smoke = tuple(manifest["pools"].get("smoke") or ())
    reference = _load_outcomes(Path(args.reference_outcomes))
    observations = (
        load_observations(Path(args.observations)) if args.observations else ()
    )
    history = _read_jsonl_rows(Path(args.history)) if args.history else []
    trajectory = _load_trajectory(Path(args.trajectory)) if args.trajectory else []
    assist = args.deepseek_assist
    analysis = args.analysis or assist
    llm_summarizer = args.llm_summarizer or assist
    semantic_review = not args.no_semantic_review or assist
    deepseek_native = args.deepseek_native or assist
    analysis_digests = args.analysis_digests
    if assist and analysis_digests == 0:
        analysis_digests = 8
    profile = ModelProfile(
        name=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    profile.api_key()
    client = ChatClient(profile, timeout=args.llm_timeout)

    def call(prompt: str, system: str) -> str:
        return client.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
        )
    out = Path(args.out)
    jobs_dir = Path(args.jobs_dir)
    log_dir = Path(args.log_dir) if args.log_dir else out / "logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    counter = {"jobs": 0}

    def run_pool(root: Path, ids: tuple[str, ...], reps: int):
        counter["jobs"] += 1
        job_name = f"{args.job_prefix}-{args.round:03d}-{counter['jobs']:03d}"
        pythonpath = [str(Path(root).resolve().parent)]
        if args.harbor_src:
            pythonpath.append(str(Path(args.harbor_src).resolve()))
        pythonpath.extend(args.pythonpath or ())
        command = build_run(
            python=args.python,
            dataset=dataset,
            task_ids=ids,
            model=profile,
            jobs_dir=str(jobs_dir),
            job_name=job_name,
            reps=reps,
            concurrency=args.concurrency,
            config=args.config,
            pythonpath=pythonpath,
            environment_build_timeout_multiplier=args.environment_build_timeout_multiplier,
        )
        run_harbor(
            command.argv,
            log_path=log_dir / f"{job_name}.log",
            env=command.env,
            timeout=args.timeout,
        )
        return [record.to_outcome() for record in read_job(jobs_dir, job_name)]

    def evaluate(root: Path, ids: tuple[str, ...], reps: int):
        return run_pool(root, ids, reps)

    def smoke_evaluate(root: Path, ids: tuple[str, ...], reps: int):
        return run_pool(root, ids, reps)

    runtime = default_runtime(
        evaluate=evaluate,
        smoke_evaluate=smoke_evaluate,
        candidates_path=args.candidates,
        ablation_path=args.ablation,
        trace_dir=args.traces,
        model=call,
        analysis=analysis,
        analysis_digests=analysis_digests,
        llm_summarizer=llm_summarizer,
        deepseek_native=deepseek_native,
    )
    if semantic_review and runtime.semantic is None:
        raise CLIError(
            "semantic review is on by default but runtime.semantic is None; "
            "pass --no-semantic-review to disable it"
        )
    margin = _prereg_number(args.margin, prereg, "margin")
    holdout_margin = _prereg_number(args.holdout_margin, prereg, "holdout_margin")
    task_margin = args.task_margin if args.task_margin is not None else margin
    spec = WorkforceSpec(
        run_id=args.run_id or Path(args.run).resolve().parent.name,
        round=args.round,
        arm=args.arm,
        mode=args.mode,
        subject=subject,
        tasks=tasks,
        holdout_tasks=holdout,
        k=k_reps,
        coreset_k=args.coreset_k,
        coreset_theta=args.coreset_theta,
        margin=margin,
        holdout_margin=holdout_margin,
        task_margin=task_margin,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
        layer=args.layer,
        batch_mode=args.batch_mode,
        target_k_ast=args.target_k_ast,
        import_modules=tuple(args.verify_module or (subject,)),
        smoke_tasks=smoke,
        forbidden_terms=tasks + holdout + smoke,
        observations=observations,
        protection_until=args.protection_until,
        rounds=args.total_rounds,
        delta=args.delta,
        s_star=args.s_star,
        stall_window=args.stall_window,
        history=tuple(history),
        trajectory=tuple(trajectory),
        b_min=args.b_min,
        b_max=args.b_max,
        m_draft=args.m_draft,
        n_prune=args.n_prune,
        budget=args.budget,
        task_brief=args.task_brief or "",
        max_turns=args.max_turns,
        max_file_edits=args.max_file_edits,
        reserved=args.reserved,
        semantic_review=semantic_review,
        repair_rounds=args.repair_rounds,
        variants=args.variants,
        max_total_trials=args.max_total_trials,
        max_cost_usd=args.max_cost_usd,
    )
    if args.loop:
        payload = run_workforce_loop(
            spec,
            runtime,
            subject_root=subject_root,
            reference=reference,
            work_dir=args.work,
            out_dir=out,
            ledger_path=args.ledger,
            force=args.force,
            rounds=args.total_rounds,
        )
        print(str(out / "loop.json"))
        print(payload["final"]["score"])
        return 0
    payload = run_workforce_round(
        spec,
        runtime,
        subject_root=subject_root,
        reference=reference,
        work_dir=args.work,
        out_dir=out,
        ledger_path=args.ledger,
        force=args.force,
    )
    print(str(out / "round.json"))
    print(payload["verdict"])
    return 0


def _load_k_pair(path: Path) -> tuple[KVector, KVector]:
    payload = _load_json_object(path, "applied")
    before = payload.get("before", payload.get("k_before"))
    after = payload.get("after", payload.get("k_after"))
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise CLIError(f"applied file has no usable K before/after: {path}")
    try:
        return KVector(**before), KVector(**after)
    except TypeError as exc:
        raise CLIError(f"applied file has an invalid K vector: {exc}") from exc


def _phase_scores(path: Path, phase: str) -> dict[str, object]:
    payload = _load_json_object(path, phase)
    nested = payload.get(phase)
    if isinstance(nested, dict):
        return dict(nested)
    return payload


def _cmd_simplify_effect(args: argparse.Namespace) -> int:
    tasks = _comma_list(args.tasks)
    if not tasks:
        raise CLIError("--tasks must list at least one task")
    holdout = _comma_list(args.holdout)
    k_before, k_after = _load_k_pair(Path(args.applied))
    batch: list[object] = []
    if args.plan:
        plan = _load_json_object(Path(args.plan), "plan")
        raw_batch = plan.get("batch", [])
        if isinstance(raw_batch, list):
            batch = raw_batch
    payload = build_simplify_effect(
        _load_outcomes(Path(args.reference)),
        _load_outcomes(Path(args.candidate)),
        tasks=tasks,
        holdout=holdout,
        k=args.k,
        k_before=k_before,
        k_after=k_after,
        batch=batch,
        diagnosis_before=(
            _phase_scores(Path(args.diagnosis_before), "before")
            if args.diagnosis_before
            else None
        ),
        diagnosis_after=(
            _phase_scores(Path(args.diagnosis_after), "after")
            if args.diagnosis_after
            else None
        ),
        placebo=(
            _load_json_object(Path(args.placebo), "placebo")
            if args.placebo
            else None
        ),
        sharp=(
            _load_json_object(Path(args.sharp), "sharp")
            if args.sharp
            else None
        ),
        margin=args.margin,
        holdout_margin=args.holdout_margin,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )
    out = Path(args.out)
    RunStore(out.parent).write_json(out.name, payload)
    print(str(out))
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    plan = load_ablation_plan(Path(args.plan))
    if args.subject is not None and args.subject != plan.subject:
        raise CLIError(
            f"subject mismatch: plan is {plan.subject!r}, got {args.subject!r}"
        )
    spec, _ = _boundaries(args.boundary, plan.subject)
    components = derive_components(root, spec)
    measured = measure_components(root, components)
    totals = measure_paths(root, _py_files(root))

    lever_rows = []
    for lever in plan.levers:
        upper_bound = attributed_k(lever, components, totals)
        reduction_pct = {}
        for name in ("k_lines_nb", "k_ast", "k_chars", "k_cc"):
            before = getattr(totals, name)
            reduction_pct[name] = (
                round(100 * getattr(upper_bound, name) / before, 4) if before else 0.0
            )
        lever_rows.append(
            {
                "id": lever.id,
                "flag": lever.flag,
                "default": lever.default_value,
                "off": lever.off_value,
                "components": list(lever.components),
                "description": lever.description,
                "agent_kwargs": plan.kwargs_for(lever.id),
                "attributed_k": upper_bound.as_dict(),
                "reduction_pct": reduction_pct,
            }
        )

    payload = {
        "subject": plan.subject,
        "root": str(root.resolve()),
        "static_only": True,
        "boundaries": {
            "subject": spec.subject,
            "boundaries": [
                {"name": boundary.name, "patterns": list(boundary.patterns)}
                for boundary in spec.boundaries
            ],
            "exclude": list(spec.exclude),
        },
        "components": [
            {
                "id": component.id,
                "paths": list(component.paths),
                "k": measured[component.id].as_dict(),
            }
            for component in components
        ],
        "totals": totals.as_dict(),
        "levers": lever_rows,
        "decision": "pending_runs",
    }
    store = RunStore(Path(args.out))
    store.write_json("prestudy.ablate-plan.json", payload)
    print(str(Path(args.out) / "prestudy.ablate-plan.json"))
    return 0


def _cmd_census(args: argparse.Namespace) -> int:
    root = Path(args.root)
    spec, _ = _boundaries(args.boundary, args.subject)
    components = derive_components(root, spec)
    component_by_path = {
        path: component.id for component in components for path in component.paths
    }
    sources: dict[str, str] = {}
    for component in components:
        sources.update(_read_sources(root, component))

    units = function_units(sources)
    traces = load_traces(Path(args.traces))
    if traces.subject != args.subject:
        raise CLIError(
            f"traces subject {traces.subject!r} does not match {args.subject!r}"
        )
    traced_paths = {path for task in traces.tasks.values() for path in task}
    matched_paths = traced_paths & set(component_by_path)
    if not matched_paths:
        raise CLIError("no traced paths match subject components")

    report = touch_report(units, traces.tasks, traces.task_ids)
    dead = zero_touch(report)
    policy = ProtectionPolicy()
    ordered = sorted(
        dead, key=lambda row: (row.unit.path, row.unit.full_span, row.unit.qualname)
    )

    candidates = []
    selected: list[KVector] = []
    picked: list[tuple[str, tuple[int, int], KVector]] = []
    for index, row in enumerate(ordered, start=1):
        unit = row.unit
        source = sources[unit.path]
        hits = tuple(check_protection(source, unit.path, policy, unit.full_span))
        delta = _delta_for(source, unit.full_span)
        candidate = Candidate(
            id=f"l2-{index:04d}",
            layer="L2",
            kind="zero_touch_function",
            component=component_by_path[unit.path],
            path=unit.path,
            span=unit.full_span,
            symbol=unit.qualname,
            evidence={
                "tasks_touched": 0,
                "tasks_total": row.total,
                "body_span": list(unit.span),
                "qualname": unit.qualname,
            },
            protection=hits,
        )
        candidates.append(
            _candidate_dict(candidate, delta, include_delta=True)
        )
        if not hits and delta is not None:
            picked.append((unit.path, unit.full_span, delta))

    last_end: dict[str, int] = {}
    for path, span, delta in sorted(picked, key=lambda item: (item[0], item[1])):
        if span[0] <= last_end.get(path, 0):
            continue
        selected.append(delta)
        last_end[path] = span[1]
    removed = _sum_additive(selected)
    totals = measure_paths(root, sorted(sources))
    reduction_pct = {}
    for name in ("k_lines_nb", "k_ast", "k_chars", "k_cc"):
        before = getattr(totals, name)
        reduction_pct[name] = (
            round(100 * getattr(removed, name) / before, 4) if before else 0.0
        )

    unblocked = [row for row in candidates if not row["blocked"] and row["applicable"]]
    payload = {
        "subject": args.subject,
        "root": str(root.resolve()),
        "layer": "L2",
        "tasks": list(traces.task_ids),
        "units": len(report),
        "touched": len(report) - len(dead),
        "zero_touch": len(dead),
        "trace_paths_outside_components": sorted(traced_paths - set(component_by_path)),
        "candidates": candidates,
        "l2": {
            "candidate_count": len(candidates),
            "blocked_count": sum(bool(row["blocked"]) for row in candidates),
            "applicable_count": sum(bool(row["applicable"]) for row in candidates),
            "selected_count": len(selected),
            "delta": removed.as_dict(),
            "reduction_pct": reduction_pct,
        },
        "decision": "candidates_found" if unblocked else "no_unblocked_candidates",
    }
    store = RunStore(Path(args.out))
    store.write_json("census.l2.json", payload)
    print(str(Path(args.out) / "census.l2.json"))
    return 0


def _cmd_traces_merge(args: argparse.Namespace) -> int:
    try:
        payload = merge_trace_dir(Path(args.dir))
    except TraceMergeError as exc:
        raise CLIError(str(exc)) from exc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(str(out))
    return 0


def _cmd_incident_open(args: argparse.Namespace) -> int:
    incident = Incident(
        id=args.id,
        version=args.version,
        phase=args.phase,
        location=args.location,
        symptom=args.symptom,
        consequence=args.consequence,
        component=args.component or "",
        task_id=args.task or "",
        trial_id=args.trial or "",
        cause=args.cause or "",
        fix=args.fix or "",
        status=args.status,
        evidence=tuple(args.evidence or ()),
    )
    path = Path(args.path)
    append_incident(path, incident)
    print(str(path))
    return 0


def _cmd_incident_list(args: argparse.Namespace) -> int:
    rows = read_incidents(Path(args.path))
    if args.status:
        rows = [row for row in rows if row.get("status") == args.status]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(f"incidents: {len(rows)}")
        for row in rows:
            print(
                f"{row.get('id')} [{row.get('status')}] {row.get('phase')} "
                f"{row.get('location')} :: {row.get('consequence')}"
            )
    return 0


def _cmd_incident_show(args: argparse.Namespace) -> int:
    rows = [row for row in read_incidents(Path(args.path)) if row.get("id") == args.id]
    if not rows:
        raise CLIError(f"incident not found: {args.id}")
    print(json.dumps(rows[-1], ensure_ascii=False, indent=2))
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        raise CLIError(f"ledger does not exist: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        payload = {"run_id": args.run_id or "", "entries": 0, "summary": {}, "latest": {}}
    else:
        run_id = args.run_id or records[0]["run_id"]
        ledger = Ledger(path, run_id=run_id)
        entries = ledger.entries()
        payload = {
            "run_id": run_id,
            "entries": len(entries),
            "summary": ledger.summary(),
            "latest": {
                candidate_id: {
                    "verdict": entry.verdict,
                    "reason": entry.reason,
                    "round": entry.round,
                }
                for candidate_id, entry in ledger.latest().items()
            },
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"run: {payload['run_id']}, entries: {payload['entries']}")
        for verdict, count in sorted(payload["summary"].items()):
            print(f"{verdict}: {count}")
    return 0


def _state_path(root: str) -> Path:
    return Path(root).resolve() / "state.json"


def _cmd_state_init(args: argparse.Namespace) -> int:
    init_run(Path(args.root), args.run_id, at=args.at)
    print(str(_state_path(args.root)))
    return 0


def _cmd_state_show(args: argparse.Namespace) -> int:
    state = load_state(Path(args.root))
    if state is None:
        raise CLIError(
            f"{_state_path(args.root)} does not exist; call state init first"
        )
    if args.json:
        print(json.dumps(state.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"run: {state.run_id}")
        print(f"phase: {state.phase}")
        print(f"resume: {state.resume or '-'}")
        print(f"updated: {state.updated_at}")
    return 0


def _cmd_state_advance(args: argparse.Namespace) -> int:
    advance_run(
        Path(args.root),
        args.to,
        artifact=args.artifact,
        actor=args.actor,
        at=args.at,
    )
    print(str(_state_path(args.root)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evo")
    subparsers = parser.add_subparsers(dest="command")

    version = subparsers.add_parser("version")
    version.set_defaults(func=_cmd_version)

    measure = subparsers.add_parser("measure")
    measure.add_argument("--root", required=True)
    measure.add_argument("--json", action="store_true")
    measure.set_defaults(func=_cmd_measure)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--root", required=True)
    scan.add_argument("--subject", required=True)
    scan.add_argument("--boundary", action="append")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=_cmd_scan)

    ai_cleanup = subparsers.add_parser("ai-cleanup")
    ai_cleanup.add_argument("--root", required=True)
    ai_cleanup.add_argument("--json", action="store_true")
    ai_cleanup.add_argument("--out")
    ai_cleanup.set_defaults(func=_cmd_ai_cleanup)

    prestudy = subparsers.add_parser("prestudy")
    prestudy_stages = prestudy.add_subparsers(dest="prestudy_stage", required=True)

    prestudy_static = prestudy_stages.add_parser("static")
    prestudy_static.add_argument("--root", required=True)
    prestudy_static.add_argument("--subject", required=True)
    prestudy_static.add_argument("--out", required=True)
    prestudy_static.add_argument("--boundary", action="append")
    prestudy_static.set_defaults(func=_cmd_prestudy_static)

    prestudy_decide = prestudy_stages.add_parser("decide")
    prestudy_decide.add_argument("--static", required=True)
    prestudy_decide.add_argument("--census", required=True)
    prestudy_decide.add_argument("--l3")
    prestudy_decide.add_argument("--out", required=True)
    prestudy_decide.set_defaults(func=_cmd_prestudy_decide)

    ablate = subparsers.add_parser("ablate")
    ablate.add_argument("--root", required=True)
    ablate.add_argument("--plan", required=True)
    ablate.add_argument("--subject")
    ablate.add_argument("--out", required=True)
    ablate.add_argument("--boundary", action="append")
    ablate.set_defaults(func=_cmd_ablate)

    ablate_run = subparsers.add_parser("ablate-run")
    ablate_run.add_argument("--run", required=True)
    ablate_run.add_argument("--plan", required=True)
    ablate_run.add_argument("--reference", required=True)
    ablate_run.add_argument("--pool", default="evolution", choices=ARM_POOLS)
    ablate_run.add_argument("--arm", default="SHARP", choices=ARMS)
    ablate_run.add_argument("--subject")
    ablate_run.add_argument("--subject-root", default="vendor/terminus2")
    ablate_run.add_argument("--jobs-dir")
    ablate_run.add_argument("--log-dir")
    ablate_run.add_argument("--job-prefix", default="ablate")
    ablate_run.add_argument("--python", default=sys.executable)
    ablate_run.add_argument("--harbor-src")
    ablate_run.add_argument("--pythonpath", action="append")
    ablate_run.add_argument("--environment-build-timeout-multiplier", type=float)
    ablate_run.add_argument("--config")
    ablate_run.add_argument("--concurrency", type=int, default=1)
    ablate_run.add_argument("--timeout", type=float)
    ablate_run.add_argument("--model", default=DEEPSEEK_FLASH.name)
    ablate_run.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    ablate_run.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    ablate_run.add_argument("--margin", type=float, default=0.10)
    ablate_run.add_argument("--alpha", type=float, default=0.05)
    ablate_run.add_argument("--n-boot", type=int, default=10_000)
    ablate_run.add_argument("--seed", type=int, default=0)
    ablate_run.add_argument("--out", required=True)
    ablate_run.set_defaults(func=_cmd_ablate_run)

    census = subparsers.add_parser("census")
    census.add_argument("--root", required=True)
    census.add_argument("--subject", required=True)
    census.add_argument("--traces", required=True)
    census.add_argument("--out", required=True)
    census.add_argument("--boundary", action="append")
    census.set_defaults(func=_cmd_census)

    traces_merge = subparsers.add_parser("traces-merge")
    traces_merge.add_argument("--dir", required=True)
    traces_merge.add_argument("--out", required=True)
    traces_merge.set_defaults(func=_cmd_traces_merge)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--outcomes", required=True)
    baseline.add_argument("--tasks", required=True)
    baseline.add_argument("--reps", type=int, required=True)
    baseline.add_argument("--root", required=True)
    baseline.add_argument("--subject", required=True)
    baseline.add_argument("--out", required=True)
    baseline.add_argument("--boundary", action="append")
    baseline.set_defaults(func=_cmd_baseline)

    harbor_results = subparsers.add_parser("harbor-results")
    harbor_results.add_argument("--jobs-dir", required=True)
    harbor_results.add_argument("--job-name", required=True)
    harbor_results.add_argument("--out", required=True)
    harbor_results.set_defaults(func=_cmd_harbor_results)

    harbor_run = subparsers.add_parser("harbor-run")
    harbor_run.add_argument("--dataset", default=DATASET)
    harbor_run.add_argument("--tasks", required=True)
    harbor_run.add_argument("--reps", type=int, default=1)
    harbor_run.add_argument("--concurrency", type=int, default=1)
    harbor_run.add_argument("--jobs-dir", required=True)
    harbor_run.add_argument("--job-name", required=True)
    harbor_run.add_argument("--python", default=sys.executable)
    harbor_run.add_argument("--pythonpath", action="append")
    harbor_run.add_argument("--subject-root", default="vendor/terminus2")
    harbor_run.add_argument("--coverage-dir")
    harbor_run.add_argument("--environment-build-timeout-multiplier", type=float)
    harbor_run.add_argument("--config")
    harbor_run.add_argument("--log", required=True)
    harbor_run.add_argument("--timeout", type=float)
    harbor_run.add_argument("--model", default=DEEPSEEK_FLASH.name)
    harbor_run.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    harbor_run.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    harbor_run.add_argument("--dry-run", action="store_true")
    harbor_run.set_defaults(func=_cmd_harbor_run)

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--run", required=True)
    run_cmd.add_argument("--arm", required=True, choices=ARMS)
    run_cmd.add_argument("--pool", required=True, choices=ARM_POOLS)
    run_cmd.add_argument("--subject")
    run_cmd.add_argument("--subject-root", default="vendor/terminus2")
    run_cmd.add_argument("--jobs-dir", required=True)
    run_cmd.add_argument("--job-name", required=True)
    run_cmd.add_argument("--log", required=True)
    run_cmd.add_argument("--python", default=sys.executable)
    run_cmd.add_argument("--harbor-src")
    run_cmd.add_argument("--pythonpath", action="append")
    run_cmd.add_argument("--coverage-dir")
    run_cmd.add_argument("--environment-build-timeout-multiplier", type=float)
    run_cmd.add_argument("--config")
    run_cmd.add_argument("--concurrency", type=int, default=1)
    run_cmd.add_argument("--timeout", type=float)
    run_cmd.add_argument("--model", default=DEEPSEEK_FLASH.name)
    run_cmd.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    run_cmd.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    run_cmd.add_argument("--dry-run", action="store_true")
    run_cmd.add_argument("--out", required=True)
    run_cmd.set_defaults(func=_cmd_run)

    simplify_cmd = subparsers.add_parser("simplify")
    simplify_cmd.add_argument("--run", required=True)
    simplify_cmd.add_argument("--arm", required=True, choices=("SIM", "PLACEBO", "SHARP"))
    simplify_cmd.add_argument(
        "--mode",
        required=True,
        choices=("static", "census", "full", "random", "sharp"),
    )
    simplify_cmd.add_argument("--candidates")
    simplify_cmd.add_argument("--l3")
    simplify_cmd.add_argument("--all-units", action="store_true")
    simplify_cmd.add_argument("--subject-root", default="vendor/terminus2")
    simplify_cmd.add_argument("--reference-outcomes", required=True)
    simplify_cmd.add_argument("--pool", default="evolution", choices=ARM_POOLS)
    simplify_cmd.add_argument("--holdout-pool", default="holdout", choices=ARM_POOLS)
    simplify_cmd.add_argument("--smoke-tasks")
    simplify_cmd.add_argument("--work", required=True)
    simplify_cmd.add_argument("--out", required=True)
    simplify_cmd.add_argument("--jobs-dir")
    simplify_cmd.add_argument("--log-dir")
    simplify_cmd.add_argument("--job-prefix", default="simplify")
    simplify_cmd.add_argument("--force", action="store_true")
    simplify_cmd.add_argument("--target-k-ast", type=int)
    simplify_cmd.add_argument("--target-candidates")
    simplify_cmd.add_argument("--batch-seed", type=int, default=0)
    simplify_cmd.add_argument("--margin", type=float, default=0.10)
    simplify_cmd.add_argument("--holdout-margin", type=float, default=0.10)
    simplify_cmd.add_argument("--task-margin", type=float, default=0.10)
    simplify_cmd.add_argument("--n-boot", type=int, default=10_000)
    simplify_cmd.add_argument("--alpha", type=float, default=0.05)
    simplify_cmd.add_argument("--seed", type=int, default=0)
    simplify_cmd.add_argument("--python", default=sys.executable)
    simplify_cmd.add_argument("--harbor-src")
    simplify_cmd.add_argument("--pythonpath", action="append")
    simplify_cmd.add_argument("--coverage-dir")
    simplify_cmd.add_argument("--environment-build-timeout-multiplier", type=float)
    simplify_cmd.add_argument("--config")
    simplify_cmd.add_argument("--concurrency", type=int, default=1)
    simplify_cmd.add_argument("--timeout", type=float)
    simplify_cmd.add_argument("--model", default=DEEPSEEK_FLASH.name)
    simplify_cmd.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    simplify_cmd.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    simplify_cmd.add_argument("--verify-module", action="append")
    simplify_cmd.set_defaults(func=_cmd_simplify)

    workforce = subparsers.add_parser("workforce")
    workforce.add_argument("--run", required=True)
    workforce.add_argument("--run-id")
    workforce.add_argument("--arm", default="SIM", choices=ARMS)
    workforce.add_argument("--mode", default="delete", choices=("delete", "edit", "auto"))
    workforce.add_argument("--pool", default="evolution", choices=ARM_POOLS)
    workforce.add_argument("--holdout-pool", default="holdout", choices=ARM_POOLS)
    workforce.add_argument("--smoke-tasks")
    workforce.add_argument("--subject-root", default="vendor/terminus2")
    workforce.add_argument("--reference-outcomes", required=True)
    workforce.add_argument("--candidates")
    workforce.add_argument("--ablation")
    workforce.add_argument("--traces")
    workforce.add_argument("--observations")
    workforce.add_argument("--history")
    workforce.add_argument("--trajectory")
    workforce.add_argument("--round", type=int, default=0)
    workforce.add_argument("--total-rounds", type=int, default=10)
    workforce.add_argument("--delta", type=float, default=0.10)
    workforce.add_argument("--s-star", type=float)
    workforce.add_argument("--stall-window", type=int, default=3)
    workforce.add_argument("--b-min", type=int, default=1)
    workforce.add_argument("--b-max", type=int, default=4)
    workforce.add_argument("--m-draft", type=int, default=1)
    workforce.add_argument("--n-prune", type=int, default=4)
    workforce.add_argument("--budget", type=int)
    workforce.add_argument("--task-brief")
    workforce.add_argument("--max-turns", type=int, default=24)
    workforce.add_argument("--max-file-edits", type=int, default=40)
    workforce.add_argument("--reserved", action="store_true")
    workforce.add_argument("--semantic-review", action="store_true")
    workforce.add_argument("--no-semantic-review", action="store_true")
    workforce.add_argument("--llm-summarizer", action="store_true")
    workforce.add_argument("--deepseek-assist", action="store_true")
    workforce.add_argument("--deepseek-native", action="store_true")
    workforce.add_argument("--repair-rounds", type=int, default=0)
    workforce.add_argument("--variants", type=int, default=1)
    workforce.add_argument("--max-total-trials", type=int)
    workforce.add_argument("--max-cost-usd", type=float)
    workforce.add_argument("--analysis", action="store_true")
    workforce.add_argument("--analysis-digests", type=int, default=0)
    workforce.add_argument("--llm-timeout", type=float, default=180.0)
    workforce.add_argument("--temperature", type=float, default=0.0)
    workforce.add_argument("--layer", default="L2", choices=("L1", "L2", "L3", "L4"))
    workforce.add_argument("--batch-mode", default="static", choices=("static", "random", "sharp"))
    workforce.add_argument("--target-k-ast", type=int)
    workforce.add_argument("--coreset-k", type=int, default=0)
    workforce.add_argument("--coreset-theta", type=float, default=0.7)
    workforce.add_argument("--verify-module", action="append")
    workforce.add_argument("--protection-until", type=int, default=0)
    workforce.add_argument("--margin", type=float)
    workforce.add_argument("--holdout-margin", type=float)
    workforce.add_argument("--task-margin", type=float)
    workforce.add_argument("--n-boot", type=int, default=10_000)
    workforce.add_argument("--alpha", type=float, default=0.05)
    workforce.add_argument("--seed", type=int, default=0)
    workforce.add_argument("--work", required=True)
    workforce.add_argument("--out", required=True)
    workforce.add_argument("--ledger")
    workforce.add_argument("--force", action="store_true")
    workforce.add_argument("--loop", action="store_true")
    workforce.add_argument("--jobs-dir", required=True)
    workforce.add_argument("--log-dir")
    workforce.add_argument("--job-prefix", default="workforce")
    workforce.add_argument("--dataset", default=DATASET)
    workforce.add_argument("--python", default=sys.executable)
    workforce.add_argument("--harbor-src")
    workforce.add_argument("--pythonpath", action="append")
    workforce.add_argument("--environment-build-timeout-multiplier", type=float)
    workforce.add_argument("--config")
    workforce.add_argument("--concurrency", type=int, default=1)
    workforce.add_argument("--timeout", type=float)
    workforce.add_argument("--model", default=DEEPSEEK_FLASH.name)
    workforce.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    workforce.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    workforce.set_defaults(func=_cmd_workforce)

    simplify_effect = subparsers.add_parser("simplify-effect")
    simplify_effect.add_argument("--reference", required=True)
    simplify_effect.add_argument("--candidate", required=True)
    simplify_effect.add_argument("--tasks", required=True)
    simplify_effect.add_argument("--holdout")
    simplify_effect.add_argument("--k", type=int, default=1)
    simplify_effect.add_argument("--applied", required=True)
    simplify_effect.add_argument("--plan")
    simplify_effect.add_argument("--diagnosis-before")
    simplify_effect.add_argument("--diagnosis-after")
    simplify_effect.add_argument("--placebo")
    simplify_effect.add_argument("--sharp")
    simplify_effect.add_argument("--margin", type=float, default=0.10)
    simplify_effect.add_argument("--holdout-margin", type=float, default=0.10)
    simplify_effect.add_argument("--n-boot", type=int, default=10_000)
    simplify_effect.add_argument("--alpha", type=float, default=0.05)
    simplify_effect.add_argument("--seed", type=int, default=0)
    simplify_effect.add_argument("--out", required=True)
    simplify_effect.set_defaults(func=_cmd_simplify_effect)

    incident = subparsers.add_parser("incident")
    incident_stages = incident.add_subparsers(dest="incident_stage", required=True)

    incident_open = incident_stages.add_parser("open")
    incident_open.add_argument("--path", default="runs/incidents.jsonl")
    incident_open.add_argument("--id", required=True)
    incident_open.add_argument("--version", required=True)
    incident_open.add_argument("--phase", required=True)
    incident_open.add_argument("--location", required=True)
    incident_open.add_argument("--symptom", required=True)
    incident_open.add_argument("--consequence", required=True)
    incident_open.add_argument("--component")
    incident_open.add_argument("--task")
    incident_open.add_argument("--trial")
    incident_open.add_argument("--cause")
    incident_open.add_argument("--fix")
    incident_open.add_argument("--status", default="open", choices=("open", "fixed", "verified", "wontfix"))
    incident_open.add_argument("--evidence", action="append")
    incident_open.set_defaults(func=_cmd_incident_open)

    incident_list = incident_stages.add_parser("list")
    incident_list.add_argument("--path", default="runs/incidents.jsonl")
    incident_list.add_argument("--status", choices=("open", "fixed", "verified", "wontfix"))
    incident_list.add_argument("--json", action="store_true")
    incident_list.set_defaults(func=_cmd_incident_list)

    incident_show = incident_stages.add_parser("show")
    incident_show.add_argument("--path", default="runs/incidents.jsonl")
    incident_show.add_argument("--id", required=True)
    incident_show.set_defaults(func=_cmd_incident_show)

    ledger = subparsers.add_parser("ledger")
    ledger.add_argument("--path", required=True)
    ledger.add_argument("--run-id")
    ledger.add_argument("--json", action="store_true")
    ledger.set_defaults(func=_cmd_ledger)

    state = subparsers.add_parser("state")
    state_stages = state.add_subparsers(dest="state_stage", required=True)

    state_init = state_stages.add_parser("init")
    state_init.add_argument("--root", required=True)
    state_init.add_argument("--run-id", required=True)
    state_init.add_argument("--at")
    state_init.set_defaults(func=_cmd_state_init)

    state_show = state_stages.add_parser("show")
    state_show.add_argument("--root", required=True)
    state_show.add_argument("--json", action="store_true")
    state_show.set_defaults(func=_cmd_state_show)

    state_advance = state_stages.add_parser("advance")
    state_advance.add_argument("--root", required=True)
    state_advance.add_argument("--to", required=True, choices=PHASES)
    state_advance.add_argument("--artifact")
    state_advance.add_argument("--actor", choices=ACTORS, default="runner")
    state_advance.add_argument("--at")
    state_advance.set_defaults(func=_cmd_state_advance)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--tasks-file", required=True)
    freeze.add_argument("--subject", required=True)
    freeze.add_argument("--root", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--dataset", default=DATASET)
    freeze.add_argument("--smoke", type=int, default=2)
    freeze.add_argument("--evolution", type=int, default=60)
    freeze.add_argument("--holdout", type=int, default=27)
    freeze.add_argument("--iteration-k", type=int, default=3)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument("--margin", type=float, default=0.10)
    freeze.add_argument("--holdout-margin", type=float, default=0.10)
    freeze.set_defaults(func=_cmd_freeze)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--tasks")
    compare.add_argument("--run")
    compare.add_argument("--pool", default="all")
    compare.add_argument("--holdout")
    compare.add_argument("--k", type=int, default=1)
    compare.add_argument("--margin", type=float, default=0.10)
    compare.add_argument("--holdout-margin", type=float, default=0.10)
    compare.add_argument("--n-boot", type=int, default=10_000)
    compare.add_argument("--alpha", type=float, default=0.05)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--family-size", type=int, default=1)
    compare.add_argument("--applied")
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=_cmd_compare)

    power_cmd = subparsers.add_parser("power")
    power_cmd.add_argument("--reference", required=True)
    power_cmd.add_argument("--candidate", required=True)
    power_cmd.add_argument("--tasks", required=True)
    power_cmd.add_argument("--k", type=int, default=1)
    power_cmd.add_argument("--margin", type=float, default=0.10)
    power_cmd.add_argument("--alpha", type=float, default=0.05)
    power_cmd.add_argument("--power", type=float, default=0.8)
    power_cmd.add_argument("--out", required=True)
    power_cmd.set_defaults(func=_cmd_power)

    version_compare = subparsers.add_parser("version-compare")
    version_compare.add_argument("--h0", required=True)
    version_compare.add_argument("--previous", required=True)
    version_compare.add_argument("--candidate", required=True)
    version_compare.add_argument("--tasks", required=True)
    version_compare.add_argument("--holdout")
    version_compare.add_argument("--k", type=int, default=3)
    version_compare.add_argument("--previous-applied")
    version_compare.add_argument("--candidate-applied")
    version_compare.add_argument("--margin", type=float, default=0.10)
    version_compare.add_argument("--holdout-margin", type=float, default=0.10)
    version_compare.add_argument("--n-boot", type=int, default=10_000)
    version_compare.add_argument("--alpha", type=float, default=0.05)
    version_compare.add_argument("--seed", type=int, default=0)
    version_compare.add_argument("--csv")
    version_compare.add_argument("--out", required=True)
    version_compare.set_defaults(func=_cmd_version_compare)

    round_cmd = subparsers.add_parser("round")
    round_cmd.add_argument("--run", required=True)
    round_cmd.add_argument("--candidate", required=True)
    round_cmd.add_argument("--applied", required=True)
    round_cmd.add_argument("--reference", required=True)
    round_cmd.add_argument("--evaluated", required=True)
    round_cmd.add_argument("--pool", default="evolution")
    round_cmd.add_argument("--holdout-pool", default="holdout")
    round_cmd.add_argument("--k", type=int)
    round_cmd.add_argument("--round", type=int, default=0)
    round_cmd.add_argument("--protection-until", type=int, default=0)
    round_cmd.add_argument("--margin", type=float)
    round_cmd.add_argument("--holdout-margin", type=float)
    round_cmd.add_argument("--task-margin", type=float)
    round_cmd.add_argument("--n-boot", type=int, default=10_000)
    round_cmd.add_argument("--alpha", type=float, default=0.05)
    round_cmd.add_argument("--seed", type=int, default=0)
    round_cmd.add_argument("--ledger", required=True)
    round_cmd.add_argument("--run-id")
    round_cmd.add_argument("--out", required=True)
    round_cmd.set_defaults(func=_cmd_round)

    evolve_round_cmd = subparsers.add_parser("evolve-round")
    evolve_round_cmd.add_argument("--run", required=True)
    evolve_round_cmd.add_argument("--incumbent", required=True)
    evolve_round_cmd.add_argument("--candidate", action="append", required=True)
    evolve_round_cmd.add_argument("--edits", action="append")
    evolve_round_cmd.add_argument("--k-lines", action="append")
    evolve_round_cmd.add_argument("--incumbent-k-lines", type=int)
    evolve_round_cmd.add_argument("--max-k-growth", type=int, default=0)
    evolve_round_cmd.add_argument("--trajectory")
    evolve_round_cmd.add_argument("--components")
    evolve_round_cmd.add_argument("--s-star", type=float)
    evolve_round_cmd.add_argument("--delta", type=float, required=True)
    evolve_round_cmd.add_argument("--round", type=int, default=0)
    evolve_round_cmd.add_argument("--total-rounds", type=int, required=True)
    evolve_round_cmd.add_argument("--k", type=int)
    evolve_round_cmd.add_argument("--n-prune", type=int, default=4)
    evolve_round_cmd.add_argument("--stall-window", type=int, default=3)
    evolve_round_cmd.add_argument("--m-draft", type=int, default=1)
    evolve_round_cmd.add_argument("--b-min", type=int, default=1)
    evolve_round_cmd.add_argument("--b-max", type=int, default=4)
    evolve_round_cmd.add_argument("--history", required=True)
    evolve_round_cmd.add_argument("--out", required=True)
    evolve_round_cmd.add_argument("--force", action="store_true")
    evolve_round_cmd.set_defaults(func=_cmd_evolve_round)

    propose_cmd = subparsers.add_parser("propose")
    propose_cmd.add_argument("--root")
    propose_cmd.add_argument("--repo")
    propose_cmd.add_argument("--worktree")
    propose_cmd.add_argument("--base", default="HEAD")
    propose_cmd.add_argument("--include", action="append")
    propose_cmd.add_argument("--model", default=DEEPSEEK_FLASH.name)
    propose_cmd.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    propose_cmd.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    propose_cmd.add_argument("--temperature", type=float, default=0.0)
    propose_cmd.add_argument("--timeout", type=float, default=180.0)
    propose_cmd.add_argument("--budget", type=int, required=True)
    propose_cmd.add_argument("--components")
    propose_cmd.add_argument("--evidence", action="append")
    propose_cmd.add_argument("--history")
    propose_cmd.add_argument("--exploration")
    propose_cmd.add_argument("--prune-set")
    propose_cmd.add_argument("--reserved", action="store_true")
    propose_cmd.add_argument("--task-brief")
    propose_cmd.add_argument("--max-turns", type=int, default=24)
    propose_cmd.add_argument("--max-file-edits", type=int, default=40)
    propose_cmd.add_argument("--extensions", default=".py")
    propose_cmd.add_argument("--keep", action="store_true")
    propose_cmd.add_argument("--out", required=True)
    propose_cmd.set_defaults(func=_cmd_propose)

    diagnose_cmd = subparsers.add_parser("diagnose")
    diagnose_cmd.add_argument("--cases", required=True)
    diagnose_cmd.add_argument(
        "--phase", choices=("before", "after"), default="before"
    )
    diagnose_cmd.add_argument("--repeats", type=int, default=3)
    diagnose_cmd.add_argument("--shuffle-seed", type=int)
    diagnose_cmd.add_argument("--max-trace-chars", type=int, default=8_000)
    diagnose_cmd.add_argument("--model", default=DEEPSEEK_FLASH.name)
    diagnose_cmd.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    diagnose_cmd.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    diagnose_cmd.add_argument("--temperature", type=float, default=0.0)
    diagnose_cmd.add_argument("--timeout", type=float, default=180.0)
    diagnose_cmd.add_argument("--out", required=True)
    diagnose_cmd.set_defaults(func=_cmd_diagnose)

    diagnose_merge = subparsers.add_parser("diagnose-merge")
    diagnose_merge.add_argument("--before", required=True)
    diagnose_merge.add_argument("--after", required=True)
    diagnose_merge.add_argument("--out", required=True)
    diagnose_merge.set_defaults(func=_cmd_diagnose_merge)

    faults_cmd = subparsers.add_parser("faults")
    faults_cmd.add_argument("--root", required=True)
    faults_cmd.add_argument("--path", action="append", required=True)
    faults_cmd.add_argument("--seed", type=int, default=0)
    faults_cmd.add_argument("--per-file", type=int, default=5)
    faults_cmd.add_argument("--out", required=True)
    faults_cmd.set_defaults(func=_cmd_faults)

    fault_cases = subparsers.add_parser("fault-cases")
    fault_cases.add_argument("--jobs-dir", required=True)
    fault_cases.add_argument("--job-name")
    fault_cases.add_argument("--truth", required=True)
    fault_cases.add_argument("--subject", default="terminus2")
    fault_cases.add_argument("--subject-root", required=True)
    fault_cases.add_argument("--boundary", action="append")
    fault_cases.add_argument(
        "--category", choices=("harness", "environment"), default="harness"
    )
    fault_cases.add_argument("--max-steps", type=int, default=10)
    fault_cases.add_argument("--max-chars", type=int, default=8_000)
    fault_cases.add_argument("--out", required=True)
    fault_cases.set_defaults(func=_cmd_fault_cases)

    fault_run = subparsers.add_parser("fault-run")
    fault_run.add_argument("--faults", required=True)
    fault_run.add_argument("--subject-root", required=True)
    fault_run.add_argument("--tasks", required=True)
    fault_run.add_argument("--work", required=True)
    fault_run.add_argument("--jobs-dir", required=True)
    fault_run.add_argument("--job-prefix", required=True)
    fault_run.add_argument("--log-dir", required=True)
    fault_run.add_argument("--python", default=sys.executable)
    fault_run.add_argument("--harbor-src")
    fault_run.add_argument("--pythonpath", action="append")
    fault_run.add_argument("--config")
    fault_run.add_argument("--dataset", default=DATASET)
    fault_run.add_argument("--concurrency", type=int, default=1)
    fault_run.add_argument("--timeout", type=float)
    fault_run.add_argument("--subject", default="terminus2")
    fault_run.add_argument("--boundary", action="append")
    fault_run.add_argument("--model", default=DEEPSEEK_FLASH.name)
    fault_run.add_argument("--base-url", default=DEEPSEEK_FLASH.base_url)
    fault_run.add_argument("--api-key-env", default=DEEPSEEK_FLASH.api_key_env)
    fault_run.add_argument("--dry-run", action="store_true")
    fault_run.add_argument("--out", required=True)
    fault_run.set_defaults(func=_cmd_fault_run)

    report = subparsers.add_parser("report")
    report.add_argument("--run", required=True)
    report.add_argument("--prestudy")
    report.add_argument("--baseline")
    report.add_argument("--comparison", action="append")
    report.add_argument("--ledger")
    report.add_argument("--diagnosis")
    report.add_argument("--out", required=True)
    report.set_defaults(func=_cmd_report)

    verify_ev = subparsers.add_parser("verify-evidence")
    verify_ev.add_argument("--report", required=True)
    verify_ev.set_defaults(func=_cmd_verify_evidence)

    apply = subparsers.add_parser("apply")
    apply.add_argument("--repo", required=True)
    apply.add_argument("--worktree", required=True)
    apply.add_argument("--candidate", required=True)
    apply.add_argument("--path", action="append", required=True)
    apply.add_argument("--module", action="append", required=True)
    apply.add_argument("--subject-root", default=".")
    apply.add_argument("--import-root", default=".")
    apply.add_argument("--include", action="append")
    apply.add_argument("--base", default="HEAD")
    apply.add_argument("--python")
    apply.add_argument("--observations")
    apply.add_argument("--out", required=True)
    apply.add_argument("--keep", action="store_true")
    apply.set_defaults(func=_cmd_apply)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except (
        CLIError,
        AblateRunError,
        AICleanupError,
        AblationError,
        ApplyError,
        ArmRunError,
        BaselineError,
        CensusError,
        CompareError,
        ComponentError,
        CoresetError,
        DiagnoseError,
        FaultCaseError,
        FaultError,
        FaultRunError,
        FreezeError,
        HarborError,
        IncidentError,
        KMeasurementError,
        LedgerError,
        LLMError,
        ModelError,
        PowerError,
        PrestudyError,
        ProposeError,
        PrestudyInputError,
        ProtocolError,
        ReportError,
        RoundError,
        RunStateError,
        EvolveRoundError,
        ScanError,
        SimplifyEffectError,
        VerifyError,
        VersionCompareError,
        WorkforceError,
        WorktreeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
