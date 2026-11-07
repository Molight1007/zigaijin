from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from evo.adapters.git_worktree import Worktree
from evo.contracts.types import KVector
from evo.domain.kmetrics import measure_paths
from evo.domain.scan import Candidate
from evo.domain.verify import (
    CheckOutcome,
    Observation,
    VerifyError,
    VerifyReport,
    aggregate,
    compile_files,
    import_check,
    required_checks,
)


BASE_CHECKS = ("compile", "import")


class ApplyError(ValueError):
    """A candidate cannot be applied to the worktree."""


@dataclass(frozen=True)
class DeletionResult:
    candidate_id: str
    path: str
    span: tuple[int, int]
    symbol: str
    before: KVector
    after: KVector | None
    verify: VerifyReport
    diff: str
    baseline_import: str
    undeletable: bool = False
    undeletable_reason: str = ""


def _read_source(root: Path, relative: str) -> str:
    path = Path(relative)
    if path.is_absolute():
        raise ApplyError(f"candidate path must be relative: {relative!r}")
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ApplyError(f"cannot read candidate file {relative!r}: {exc}") from exc


def _delete_span(source: str, span: tuple[int, int]) -> str:
    start, end = span
    lines = source.splitlines(keepends=True)
    if not 1 <= start <= end <= len(lines):
        raise ApplyError(
            f"span {span} is outside the file (1..{len(lines)})"
        )
    return "".join(lines[: start - 1] + lines[end:])


def _inside(worktree: Worktree, relative: str, label: str) -> Path:
    root = Path(relative)
    if root.is_absolute():
        raise ApplyError(f"{label} must be relative: {relative!r}")
    base = (worktree.path / root).resolve()
    if not base.is_relative_to(worktree.path.resolve()):
        raise ApplyError(f"{label} escapes the worktree: {relative!r}")
    if not base.is_dir():
        raise ApplyError(f"{label} is not a directory: {relative!r}")
    return base


def _diff(path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def _is_residual_reference(detail: str, symbol: str) -> bool:
    return symbol in detail and (
        detail.startswith("NameError") or detail.startswith("AttributeError")
    )


def _validate_extra_checks(extra: Iterable[CheckOutcome]) -> tuple[CheckOutcome, ...]:
    checks = tuple(extra)
    names = [check.name for check in checks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    overlap = sorted(set(names) & set(BASE_CHECKS))
    if duplicates:
        raise ApplyError(f"duplicate check names: {', '.join(duplicates)}")
    if overlap:
        raise ApplyError(f"extra checks repeat base checks: {', '.join(overlap)}")
    return checks


def load_observations(path: str | Path) -> tuple[Observation, ...]:
    """Load recorded before/after cases for replay and differential checks."""
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApplyError(f"cannot read observations {target}: {exc}") from exc
    if not isinstance(payload, list):
        raise ApplyError("observations file must contain a JSON array")
    observations: list[Observation] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ApplyError(f"observation {index} must be a JSON object")
        try:
            observations.append(
                Observation(
                    case_id=row.get("case_id"),
                    recorded=row.get("recorded"),
                    reference=row.get("reference"),
                    candidate=row.get("candidate"),
                )
            )
        except VerifyError as exc:
            raise ApplyError(f"observation {index}: {exc}") from exc
    return tuple(observations)


def delete_candidate(
    worktree: Worktree,
    candidate: Candidate,
    *,
    paths: Sequence[str],
    modules: Sequence[str],
    subject_root: str = ".",
    import_root: str = ".",
    python: str | None = None,
    extra_checks: Iterable[CheckOutcome] = (),
) -> DeletionResult:
    if not isinstance(worktree, Worktree):
        raise ApplyError("worktree must be a Worktree")
    if not isinstance(candidate, Candidate):
        raise ApplyError("candidate must be a Candidate")
    if candidate.blocked:
        raise ApplyError(
            f"candidate {candidate.id!r} is protected: "
            + "; ".join(hit.reason for hit in candidate.protection)
        )
    path_list = [str(path) for path in paths]
    if not path_list:
        raise ApplyError("paths must not be empty")
    module_list = [str(name) for name in modules]
    if not module_list:
        raise ApplyError("modules must not be empty")
    checks = _validate_extra_checks(extra_checks)

    subject = _inside(worktree, subject_root, "subject root")
    import_base = _inside(worktree, import_root, "import root")
    source = _read_source(subject, candidate.path)
    after_source = _delete_span(source, candidate.span)

    before = measure_paths(subject, path_list)
    baseline_import = import_check(import_base, module_list, python=python)

    target = subject / candidate.path
    target.write_text(after_source, encoding="utf-8", newline="\n")

    compile_outcome = compile_files(subject, [candidate.path])
    if compile_outcome.status == "fail":
        verify = aggregate(
            [compile_outcome, CheckOutcome("import", "unknown", "compile failed"), *checks],
            required_checks(candidate.layer),
        )
        return DeletionResult(
            candidate_id=candidate.id,
            path=candidate.path,
            span=candidate.span,
            symbol=candidate.symbol,
            before=before,
            after=None,
            verify=verify,
            diff=_diff(candidate.path, source, after_source),
            baseline_import=baseline_import.status,
            undeletable=True,
            undeletable_reason=f"deletion did not compile: {compile_outcome.detail}",
        )

    after_import = import_check(import_base, module_list, python=python)
    verify = aggregate(
        [compile_outcome, after_import, *checks], required_checks(candidate.layer)
    )
    undeletable = False
    reason = ""
    if baseline_import.status == "pass" and after_import.status == "fail":
        undeletable = True
        if _is_residual_reference(after_import.detail, candidate.symbol):
            reason = (
                f"residual call site for {candidate.symbol}: {after_import.detail}"
            )
        else:
            reason = f"deletion broke import: {after_import.detail}"
    return DeletionResult(
        candidate_id=candidate.id,
        path=candidate.path,
        span=candidate.span,
        symbol=candidate.symbol,
        before=before,
        after=measure_paths(subject, path_list),
        verify=verify,
        diff=_diff(candidate.path, source, after_source),
        baseline_import=baseline_import.status,
        undeletable=undeletable,
        undeletable_reason=reason,
    )
