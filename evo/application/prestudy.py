from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from evo.domain.prestudy import K_AST_PCT_THRESHOLD, L3_YIELD_THRESHOLD, decide_prestudy


_REDUCTION_METRICS = ("k_lines_nb", "k_ast", "k_chars", "k_cc")
_L3_VERDICTS = ("zero_contribution", "contributes", "inconclusive", "invalid")


class PrestudyInputError(ValueError):
    """A prestudy artifact is missing, malformed or inconsistent."""


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise PrestudyInputError(f"{label} is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PrestudyInputError(f"invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrestudyInputError(f"{label} must be a JSON object: {path}")
    return payload


def _subject(payload: Mapping, label: str) -> str:
    subject = payload.get("subject")
    if not isinstance(subject, str) or not subject:
        raise PrestudyInputError(f"{label} has no subject")
    return subject


def _reduction(payload: Mapping, section: str, label: str) -> dict[str, float]:
    block = payload.get(section)
    if not isinstance(block, dict):
        raise PrestudyInputError(f"{label} has no {section!r} section")
    reduction = block.get("reduction_pct")
    if not isinstance(reduction, dict):
        raise PrestudyInputError(f"{label} has no {section}.reduction_pct")
    values: dict[str, float] = {}
    for metric in _REDUCTION_METRICS:
        value = reduction.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PrestudyInputError(
                f"{label} has no numeric {section}.reduction_pct.{metric}"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise PrestudyInputError(
                f"{label} has an invalid {section}.reduction_pct.{metric}: {value!r}"
            )
        values[metric] = number
    return values


def _count(payload: Mapping, key: str, label: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrestudyInputError(f"{label} has an invalid {key}: {value!r}")
    return value


def _l3_summary(payload: Mapping, label: str) -> tuple[float, int, int]:
    levers = payload.get("levers")
    if not isinstance(levers, list) or not levers:
        raise PrestudyInputError(f"{label} has no levers")
    evaluated = 0
    zero_contribution = 0
    for index, lever in enumerate(levers):
        if not isinstance(lever, dict):
            raise PrestudyInputError(f"{label} lever {index} is not an object")
        lever_id = lever.get("id")
        if not isinstance(lever_id, str) or not lever_id:
            raise PrestudyInputError(f"{label} lever {index} has no id")
        verdict = lever.get("verdict")
        if verdict not in _L3_VERDICTS:
            raise PrestudyInputError(
                f"{label} lever {lever_id!r} has an unknown verdict: {verdict!r}"
            )
        if verdict == "invalid":
            continue
        evaluated += 1
        if verdict == "zero_contribution":
            zero_contribution += 1
    if evaluated == 0:
        raise PrestudyInputError(f"{label} has no valid L3 verdicts")
    return zero_contribution / evaluated, zero_contribution, evaluated


def _census_tasks(payload: Mapping, label: str) -> int | None:
    tasks = payload.get("tasks")
    if tasks is None:
        return None
    if not isinstance(tasks, list):
        raise PrestudyInputError(f"{label} tasks must be a list")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, str) or not task:
            raise PrestudyInputError(f"{label} has an invalid task id: {task!r}")
        if task in seen:
            raise PrestudyInputError(f"{label} has a duplicate task id: {task!r}")
        seen.add(task)
    return len(tasks)


def build_prestudy(
    static: Mapping, census: Mapping, l3: Mapping | None = None
) -> dict[str, object]:
    static_subject = _subject(static, "static prestudy")
    census_subject = _subject(census, "census")
    if static_subject != census_subject:
        raise PrestudyInputError(
            f"subject mismatch: static is {static_subject!r}, census is {census_subject!r}"
        )
    subject = static_subject
    static_reduction = _reduction(static, "l1", "static prestudy")
    census_reduction = _reduction(census, "l2", "census")

    l3_block: dict[str, object] | None = None
    l3_yield: float | None = None
    if l3 is not None:
        l3_subject = _subject(l3, "L3 results")
        if l3_subject != subject:
            raise PrestudyInputError(
                f"subject mismatch: L3 results is {l3_subject!r}, expected {subject!r}"
            )
        l3_yield, zero_contribution, evaluated = _l3_summary(l3, "L3 results")
        l3_block = {
            "yield": l3_yield,
            "zero_contribution": zero_contribution,
            "evaluated": evaluated,
        }

    decision = decide_prestudy(
        static_reduction["k_ast"],
        census_reduction["k_ast"],
        l3_yield,
        census_tasks=_census_tasks(census, "census"),
    )
    l1_block = static.get("l1")
    static_candidates = (
        _count(l1_block, "candidate_count", "static prestudy")
        if isinstance(l1_block, dict)
        else None
    )
    report: dict[str, object] = {
        "subject": subject,
        "k_metric": "k_ast",
        "static": {
            "reduction_pct": static_reduction,
            "candidate_count": static_candidates,
        },
        "census": {
            "reduction_pct": census_reduction,
            "tasks": list(census.get("tasks") or []),
            "zero_touch": _count(census, "zero_touch", "census"),
        },
        "l3": l3_block,
        "decision": decision.decision,
        "combined_pct": decision.combined_pct,
        "thresholds": {
            "l3_yield": L3_YIELD_THRESHOLD,
            "k_ast_pct": K_AST_PCT_THRESHOLD,
        },
        "reasons": list(decision.reasons),
    }
    return report


def load_prestudy(
    static_path: Path | str,
    census_path: Path | str,
    l3_path: Path | str | None = None,
) -> dict[str, object]:
    static = _read_json(Path(static_path), "static prestudy")
    census = _read_json(Path(census_path), "census")
    l3 = _read_json(Path(l3_path), "L3 results") if l3_path is not None else None
    return build_prestudy(static, census, l3)
