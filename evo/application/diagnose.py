from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


class DiagnoseError(ValueError):
    """A diagnosis case, answer or ranking is malformed."""


CATEGORIES = ("harness", "environment")

DIAGNOSE_SYSTEM = """You are a failure attribution engineer for an LLM-agent
harness. You receive a failed run's trace and a frozen list of harness
components. Decide which components caused the failure and whether the
failure was caused by the harness or by the environment/task.

Rules:
- Rank at most three components, most likely first, using ONLY the listed ids.
- Attribute to the harness when a harness bug, prompt or state handling caused
  the failure; choose environment when the task, the grader or the sandbox
  failed.
- Base the answer on the trace and the component descriptions; do not invent
  components.

Reply with EXACTLY one JSON object and nothing else:
{"top3": ["component-id", "..."], "cause": "harness" | "environment"}"""


@dataclass(frozen=True)
class DiagnosisComponent:
    id: str
    description: str = ""
    k_lines: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise DiagnoseError(f"component id must be a non-empty string: {self.id!r}")
        if not isinstance(self.description, str):
            raise DiagnoseError(
                f"component {self.id!r} description must be a string"
            )
        if self.k_lines is not None and (
            isinstance(self.k_lines, bool)
            or not isinstance(self.k_lines, int)
            or self.k_lines < 0
        ):
            raise DiagnoseError(
                f"component {self.id!r} k_lines must be a non-negative integer"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "k_lines": self.k_lines,
        }


@dataclass(frozen=True)
class DiagnosisCase:
    case_id: str
    failure: str
    components: tuple[DiagnosisComponent, ...]
    culprit: str | None
    category: str = "harness"

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise DiagnoseError(f"case_id must be a non-empty string: {self.case_id!r}")
        if not isinstance(self.failure, str) or not self.failure.strip():
            raise DiagnoseError(
                f"case {self.case_id!r} failure must be a non-empty string"
            )
        if not isinstance(self.components, tuple) or not self.components:
            raise DiagnoseError(
                f"case {self.case_id!r} must list at least one component"
            )
        ids = [component.id for component in self.components]
        if len(ids) != len(set(ids)):
            raise DiagnoseError(
                f"case {self.case_id!r} has duplicate component ids"
            )
        if self.culprit is not None and self.culprit not in ids:
            raise DiagnoseError(
                f"case {self.case_id!r} culprit {self.culprit!r} is not in components"
            )
        if self.category not in CATEGORIES:
            raise DiagnoseError(
                f"case {self.case_id!r} category must be one of {CATEGORIES}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "failure": self.failure,
            "components": [component.as_dict() for component in self.components],
            "culprit": self.culprit,
            "category": self.category,
        }


def rank_prompt(
    case: DiagnosisCase,
    *,
    seed: int | None = None,
    max_trace_chars: int = 8_000,
) -> str:
    if isinstance(max_trace_chars, bool) or not isinstance(max_trace_chars, int):
        raise DiagnoseError("max_trace_chars must be an integer")
    if max_trace_chars <= 0:
        raise DiagnoseError("max_trace_chars must be positive")
    components = list(case.components)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DiagnoseError("seed must be an integer")
        random.Random(seed).shuffle(components)
    trace = case.failure
    if len(trace) > max_trace_chars:
        trace = trace[:max_trace_chars] + "\n...[truncated]"
    listing = json.dumps(
        [component.as_dict() for component in components],
        ensure_ascii=False,
        indent=1,
    )
    return "\n\n".join(
        [
            "=== FAILURE TRACE ===",
            trace,
            "=== HARNESS COMPONENTS ===",
            listing,
            "=== ANSWER ===",
            "Return exactly one JSON object with keys top3 and cause.",
        ]
    )


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_ranking(text: str) -> tuple[list[str], str]:
    if not isinstance(text, str) or not text.strip():
        raise DiagnoseError("ranking answer must be non-empty text")
    try:
        payload = json.loads(_strip_fence(text))
    except json.JSONDecodeError as exc:
        raise DiagnoseError(f"ranking answer is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise DiagnoseError("ranking answer must be one JSON object")
    top3 = payload.get("top3")
    if not isinstance(top3, list) or not top3:
        raise DiagnoseError("ranking answer has no non-empty top3 list")
    if len(top3) > 3:
        raise DiagnoseError("ranking answer lists more than three components")
    ranking: list[str] = []
    for item in top3:
        if not isinstance(item, str) or not item.strip():
            raise DiagnoseError("top3 entries must be non-empty component ids")
        if item in ranking:
            raise DiagnoseError(f"top3 repeats component {item!r}")
        ranking.append(item)
    cause = payload.get("cause")
    if cause not in CATEGORIES:
        raise DiagnoseError(
            f"ranking answer cause must be one of {CATEGORIES}: {cause!r}"
        )
    return ranking, str(cause)


def score_case(
    case: DiagnosisCase, ranking: Sequence[str], cause: str
) -> dict[str, bool | None]:
    if cause not in CATEGORIES:
        raise DiagnoseError(f"cause must be one of {CATEGORIES}: {cause!r}")
    binary = cause == case.category
    if case.culprit is None:
        return {"top1": None, "top3": None, "binary": binary}
    ranked = list(ranking)
    return {
        "top1": bool(ranked) and ranked[0] == case.culprit,
        "top3": case.culprit in ranked[:3],
        "binary": binary,
    }


def _majority(values: Sequence[bool]) -> bool:
    return sum(1 for value in values if value) * 2 > len(values)


def _mean(values: list[bool]) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 6)


def diagnose(
    cases: Sequence[DiagnosisCase],
    model: Callable[[str], str],
    *,
    repeats: int = 3,
    seed: int | None = None,
    max_trace_chars: int = 8_000,
) -> dict[str, object]:
    if not callable(model):
        raise DiagnoseError("model must be callable")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise DiagnoseError(f"repeats must be a positive integer: {repeats!r}")
    case_list = list(cases)
    if not case_list:
        raise DiagnoseError("diagnose needs at least one case")

    rows: list[dict[str, object]] = []
    top1_values: list[bool] = []
    top3_values: list[bool] = []
    binary_values: list[bool] = []
    for case in case_list:
        answers: list[dict[str, object]] = []
        top1_repeats: list[bool] = []
        top3_repeats: list[bool] = []
        binary_repeats: list[bool] = []
        for _ in range(repeats):
            prompt = rank_prompt(case, seed=seed, max_trace_chars=max_trace_chars)
            raw = model(prompt)
            try:
                ranking, cause = parse_ranking(raw)
            except DiagnoseError as exc:
                answers.append({"error": str(exc), "raw": str(raw)[:500]})
                if case.culprit is not None:
                    top1_repeats.append(False)
                    top3_repeats.append(False)
                binary_repeats.append(False)
                continue
            scored = score_case(case, ranking, cause)
            answers.append({"ranking": ranking, "cause": cause})
            if scored["top1"] is not None:
                top1_repeats.append(bool(scored["top1"]))
                top3_repeats.append(bool(scored["top3"]))
            binary_repeats.append(bool(scored["binary"]))
        has_culprit = case.culprit is not None
        top1 = _majority(top1_repeats) if has_culprit else None
        top3 = _majority(top3_repeats) if has_culprit else None
        binary = _majority(binary_repeats)
        if top1 is not None:
            top1_values.append(top1)
            top3_values.append(bool(top3))
        binary_values.append(binary)
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "culprit": case.culprit,
                "top1": top1,
                "top3": top3,
                "binary": binary,
                "answers": answers,
            }
        )

    return {
        "schema": 1,
        "n_cases": len(case_list),
        "repeats": repeats,
        "metrics": {
            "n_cases": len(case_list),
            "n_ranked": len(top1_values),
            "top1": _mean(top1_values),
            "top3": _mean(top3_values),
            "binary": _mean(binary_values),
        },
        "per_case": rows,
    }


def parse_cases(payload: object) -> list[DiagnosisCase]:
    if isinstance(payload, Mapping):
        raw = payload.get("cases")
    else:
        raw = payload
    if not isinstance(raw, list):
        raise DiagnoseError("cases must be a JSON array or an object with 'cases'")
    if not raw:
        raise DiagnoseError("cases must contain at least one case")
    cases: list[DiagnosisCase] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise DiagnoseError(f"case {index} is not an object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DiagnoseError(f"case {index} has no case_id")
        if case_id in seen:
            raise DiagnoseError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        failure = item.get("failure")
        if not isinstance(failure, str) or not failure.strip():
            raise DiagnoseError(f"case {case_id!r} has no failure text")
        raw_components = item.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise DiagnoseError(
                f"case {case_id!r} must list at least one component"
            )
        components: list[DiagnosisComponent] = []
        for position, component in enumerate(raw_components):
            if not isinstance(component, Mapping):
                raise DiagnoseError(
                    f"case {case_id!r} component {position} is not an object"
                )
            components.append(
                DiagnosisComponent(
                    id=component.get("id"),  # type: ignore[arg-type]
                    description=component.get("description", ""),  # type: ignore[arg-type]
                    k_lines=component.get("k_lines"),  # type: ignore[arg-type]
                )
            )
        culprit = item.get("culprit")
        if culprit is not None and not isinstance(culprit, str):
            raise DiagnoseError(f"case {case_id!r} culprit must be a string or null")
        category = item.get("category", "harness")
        cases.append(
            DiagnosisCase(
                case_id=case_id,
                failure=failure,
                components=tuple(components),
                culprit=culprit,
                category=category if isinstance(category, str) else str(category),
            )
        )
    return cases


SCORE_PHASES = ("before", "after")


def merge_scores(*documents: Mapping[str, object]) -> dict[str, object]:
    """Merge before/after score documents into one report input."""
    merged: dict[str, object] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise DiagnoseError("score document must be a JSON object")
        for phase, metrics in document.items():
            if phase not in SCORE_PHASES:
                raise DiagnoseError(f"unknown diagnosis phase: {phase!r}")
            if phase in merged:
                raise DiagnoseError(f"duplicate diagnosis phase: {phase!r}")
            if not isinstance(metrics, Mapping):
                raise DiagnoseError(f"diagnosis phase {phase!r} must be an object")
            merged[phase] = dict(metrics)
    if not merged:
        raise DiagnoseError("no diagnosis phases were provided")
    return merged
