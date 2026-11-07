from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


STATUSES = ("pass", "fail", "unknown")

REQUIRED_BY_LAYER = {
    "L1": ("compile", "import"),
    "L2": ("compile", "import", "smoke"),
    "L3": ("compile", "import", "smoke", "replay"),
    "L4": ("compile", "import", "smoke", "replay", "differential"),
}

_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_IMPORT_SCRIPT = (
    "import importlib, os\n"
    "for name in os.environ['EVO_VERIFY_MODULES'].split(','):\n"
    "    importlib.import_module(name)\n"
)


class VerifyError(ValueError):
    """A verification input is malformed or cannot be evaluated."""


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    status: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise VerifyError(f"check name must be a non-empty string: {self.name!r}")
        if self.status not in STATUSES:
            raise VerifyError(
                f"unknown status {self.status!r}; expected one of {', '.join(STATUSES)}"
            )
        if not isinstance(self.detail, str):
            raise VerifyError(f"detail must be a string: {self.detail!r}")

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class Observation:
    case_id: str
    recorded: bool
    reference: object = None
    candidate: object = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise VerifyError("case_id must be a non-empty string")
        if not isinstance(self.recorded, bool):
            raise VerifyError("recorded must be a bool")


@dataclass(frozen=True)
class VerifyReport:
    outcomes: tuple[CheckOutcome, ...]
    required: tuple[str, ...]

    def __post_init__(self) -> None:
        names = [outcome.name for outcome in self.outcomes]
        if len(set(names)) != len(names):
            raise VerifyError("duplicate check outcome names")
        if not self.required:
            raise VerifyError("required checks must not be empty")
        if len(set(self.required)) != len(self.required):
            raise VerifyError("duplicate required check names")
        for check in self.outcomes:
            if not isinstance(check, CheckOutcome):
                raise VerifyError("outcomes must be CheckOutcome instances")

    @property
    def verdict(self) -> str:
        if any(outcome.status == "fail" for outcome in self.outcomes):
            return "fail"
        by_name = {outcome.name: outcome for outcome in self.outcomes}
        for name in self.required:
            outcome = by_name.get(name)
            if outcome is None or outcome.status == "unknown":
                return "unknown"
        return "pass"

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "required": list(self.required),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }


def required_checks(layer: str) -> tuple[str, ...]:
    try:
        return REQUIRED_BY_LAYER[layer]
    except KeyError:
        raise VerifyError(
            f"unknown layer {layer!r}; expected one of {', '.join(REQUIRED_BY_LAYER)}"
        ) from None


def aggregate(
    outcomes: Iterable[CheckOutcome], required: Sequence[str]
) -> VerifyReport:
    required = tuple(required)
    if not required:
        raise VerifyError("required checks must not be empty")
    return VerifyReport(tuple(outcomes), required)


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return repr(value)


def _ordered(observations: Iterable[Observation]) -> tuple[Observation, ...]:
    ordered = tuple(observations)
    seen: set[str] = set()
    for observation in ordered:
        if not isinstance(observation, Observation):
            raise VerifyError("observations must be Observation instances")
        if observation.case_id in seen:
            raise VerifyError(f"duplicate observation id: {observation.case_id!r}")
        seen.add(observation.case_id)
    return ordered


def compile_files(root: Path | str, paths: Iterable[str]) -> CheckOutcome:
    base = Path(root).resolve()
    failures: list[str] = []
    for relative in paths:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise VerifyError(f"path must be relative to root: {relative!r}")
        resolved = (base / candidate).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise VerifyError(f"path escapes the root: {relative!r}") from None
        label = candidate.as_posix()
        try:
            source = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{label}: cannot read: {exc}")
            continue
        try:
            compile(source, label, "exec")
        except SyntaxError as exc:
            location = f"{label}:{exc.lineno}" if exc.lineno else label
            failures.append(f"{location}: {exc.msg}")
    if failures:
        shown = "; ".join(failures[:5])
        if len(failures) > 5:
            shown += f"; (+{len(failures) - 5} more)"
        return CheckOutcome("compile", "fail", shown)
    return CheckOutcome("compile", "pass")


def import_check(
    root: Path | str,
    modules: Iterable[str],
    *,
    python: str | None = None,
    timeout: float = 60.0,
) -> CheckOutcome:
    names = tuple(modules)
    if not names:
        raise VerifyError("modules must not be empty")
    for name in names:
        if not isinstance(name, str) or not _MODULE_RE.match(name):
            raise VerifyError(f"invalid module name: {name!r}")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise VerifyError(f"timeout must be positive: {timeout!r}")
    base = Path(root).resolve()
    if not base.is_dir():
        raise VerifyError(f"root is not a directory: {base}")

    env = dict(os.environ)
    env["EVO_VERIFY_MODULES"] = ",".join(names)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(base), existing]) if existing else str(base)
    try:
        completed = subprocess.run(
            [python or sys.executable, "-c", _IMPORT_SCRIPT],
            cwd=base,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CheckOutcome("import", "unknown", f"import timed out after {timeout}s")
    except OSError as exc:
        return CheckOutcome("import", "unknown", f"cannot launch interpreter: {exc}")

    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        detail = lines[-1] if lines else f"interpreter exited with {completed.returncode}"
        return CheckOutcome("import", "fail", detail)
    return CheckOutcome("import", "pass")


def replay_check(
    observations: Iterable[Observation], *, minimum: int = 10
) -> CheckOutcome:
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise VerifyError(f"minimum must be a non-negative integer: {minimum!r}")
    ordered = _ordered(observations)
    for observation in ordered:
        if observation.recorded and _canonical(observation.reference) != _canonical(
            observation.candidate
        ):
            return CheckOutcome(
                "replay",
                "fail",
                f"case {observation.case_id} changed behavior: "
                f"{_canonical(observation.reference)} -> {_canonical(observation.candidate)}",
            )
    if len(ordered) < minimum:
        return CheckOutcome(
            "replay",
            "unknown",
            f"only {len(ordered)} of {minimum} required cases available",
        )
    for observation in ordered:
        if not observation.recorded:
            return CheckOutcome(
                "replay", "unknown", f"case {observation.case_id} has no recorded behavior"
            )
    return CheckOutcome("replay", "pass")


def differential_check(observations: Iterable[Observation]) -> CheckOutcome:
    ordered = _ordered(observations)
    if not ordered:
        return CheckOutcome(
            "differential", "unknown", "no recorded cases; differential needs captured inputs"
        )
    for observation in ordered:
        if observation.recorded and _canonical(observation.reference) != _canonical(
            observation.candidate
        ):
            return CheckOutcome(
                "differential",
                "fail",
                f"case {observation.case_id} changed output: "
                f"{_canonical(observation.reference)} -> {_canonical(observation.candidate)}",
            )
    for observation in ordered:
        if not observation.recorded:
            return CheckOutcome(
                "differential",
                "unknown",
                f"case {observation.case_id} has no recorded input",
            )
    return CheckOutcome("differential", "pass")
