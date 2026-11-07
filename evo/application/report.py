from __future__ import annotations

import csv
import io
import math
from typing import Mapping, Sequence


TABLES = (
    "t1_subject",
    "t2_arms",
    "t3_deletions",
    "t4_baselines",
    "t5_diagnosis",
    "t6_cost",
)

K_COLUMNS = ("k_lines_nb", "k_ast", "k_chars", "k_bytes", "k_gzip", "k_cc")
CONTROL_ARMS = ("PLACEBO", "SHARP")
DIAGNOSIS_PHASES = ("before", "after")

T1_COLUMNS = ["component", "paths", *K_COLUMNS]
T2_COLUMNS = [
    "arm",
    "source",
    "n_pairs",
    "reference_mean",
    "candidate_mean",
    "mean_delta",
    "ci_low",
    "ci_high",
    "margin",
    "non_inferior",
    "validity",
    "worst_task",
    "worst_delta",
    "tokens_reference",
    "tokens_candidate",
    "cost_reference",
    "cost_candidate",
]
T3_COLUMNS = ["round", "candidate_id", "verdict", "reason", "layer", "verify"]
T4_COLUMNS = [
    "arm",
    "mean_delta",
    "ci_low",
    "ci_high",
    "margin",
    "non_inferior",
    "worst_task",
    "worst_delta",
    "tokens_reference",
    "tokens_candidate",
    "cost_reference",
    "cost_candidate",
]
T5_COLUMNS = ["phase", "metric", "value"]
T6_COLUMNS = [
    "arm",
    "k_lines_pct",
    "k_ast_pct",
    "trials",
    "tokens_total",
    "cost_total",
    "trials_per_1pct_k",
    "tokens_per_1pct_k",
    "cost_per_1pct_k",
]


class ReportError(ValueError):
    """A report artifact is malformed and cannot be turned into a table."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{name} must be a mapping: {value!r}")
    return value


def _number(block: Mapping[str, object], key: str, where: str) -> float:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{where}.{key} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(f"{where}.{key} must be finite: {value!r}")
    return number


def _count(block: Mapping[str, object], key: str, where: str) -> int:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportError(f"{where}.{key} must be a non-negative integer: {value!r}")
    return value


def _text(block: Mapping[str, object], key: str, where: str) -> str:
    value = block.get(key)
    if not isinstance(value, str) or not value:
        raise ReportError(f"{where}.{key} must be a non-empty string: {value!r}")
    return value


def _table(
    columns: Sequence[str],
    rows: list[list[object]],
    source: Sequence[str],
) -> dict[str, object]:
    return {
        "status": "ok" if rows else "not_run",
        "columns": list(columns),
        "rows": rows,
        "source": list(source) if rows else [],
    }


def _subject_table(prestudy: Mapping[str, object] | None) -> dict[str, object]:
    if prestudy is None:
        return _table(T1_COLUMNS, [], [])
    document = _mapping(prestudy, "prestudy")
    components = document.get("components")
    if not isinstance(components, list):
        raise ReportError("prestudy.components must be a list")
    rows: list[list[object]] = []
    for index, component in enumerate(components):
        where = f"prestudy.components[{index}]"
        record = _mapping(component, where)
        paths = record.get("paths")
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not path for path in paths
        ):
            raise ReportError(f"{where}.paths must be a list of strings")
        k = _mapping(record.get("k"), f"{where}.k")
        row: list[object] = [_text(record, "id", where), ";".join(paths)]
        row.extend(_count(k, name, f"{where}.k") for name in K_COLUMNS)
        rows.append(row)
    rows.sort(key=lambda row: row[0])
    return _table(T1_COLUMNS, rows, ["prestudy.json"])


def _baseline_row(baseline: Mapping[str, object]) -> list[object]:
    document = _mapping(baseline, "baseline")
    aggregate = _mapping(document.get("aggregate"), "baseline.aggregate")
    validity = _mapping(document.get("validity"), "baseline.validity")
    status = validity.get("status")
    if status not in ("pass", "fail"):
        raise ReportError(f"baseline.validity.status is unknown: {status!r}")
    mean = aggregate.get("mean_reward")
    if mean is None:
        mean_cell: object = ""
    else:
        mean_cell = _number(aggregate, "mean_reward", "baseline.aggregate")
    tokens = _count(aggregate, "tokens_in", "baseline.aggregate") + _count(
        aggregate, "tokens_out", "baseline.aggregate"
    )
    return [
        "H0",
        "baseline",
        _count(aggregate, "tasks_with_valid_trials", "baseline.aggregate"),
        mean_cell,
        "",
        "",
        "",
        "",
        "",
        "",
        status,
        "",
        "",
        tokens,
        "",
        _number(aggregate, "cost_usd", "baseline.aggregate"),
        "",
    ]


def _comparison_parts(arm: str, payload: Mapping[str, object]) -> tuple[
    Mapping[str, object], Mapping[str, object], Mapping[str, object]
]:
    where = f"comparison[{arm}]"
    body = _mapping(payload, where)
    aggregate = _mapping(body.get("aggregate"), f"{where}.aggregate")
    ci = _mapping(aggregate.get("ci"), f"{where}.aggregate.ci")
    worst = _mapping(aggregate.get("worst_task"), f"{where}.aggregate.worst_task")
    return aggregate, ci, worst


def _tokens_and_cost(
    arm: str, aggregate: Mapping[str, object]
) -> dict[str, float | int]:
    where = f"comparison[{arm}].aggregate"
    tokens = _mapping(aggregate.get("tokens"), f"{where}.tokens")
    reference = _mapping(tokens.get("reference"), f"{where}.tokens.reference")
    candidate = _mapping(tokens.get("candidate"), f"{where}.tokens.candidate")
    cost = _mapping(aggregate.get("cost"), f"{where}.cost")
    return {
        "tokens_reference": _count(reference, "total", f"{where}.tokens.reference"),
        "tokens_candidate": _count(candidate, "total", f"{where}.tokens.candidate"),
        "cost_reference": _number(cost, "reference", f"{where}.cost"),
        "cost_candidate": _number(cost, "candidate", f"{where}.cost"),
    }


def _candidate_validity(arm: str, payload: Mapping[str, object]) -> str:
    where = f"comparison[{arm}]"
    body = _mapping(payload, where)
    validity = _mapping(body.get("validity"), f"{where}.validity")
    candidate = _mapping(validity.get("candidate"), f"{where}.validity.candidate")
    status = candidate.get("status")
    if status not in ("pass", "fail"):
        raise ReportError(f"{where}.validity.candidate.status is unknown: {status!r}")
    return status


def _comparison_row(
    arm: str, payload: Mapping[str, object]
) -> list[object]:
    aggregate, ci, worst = _comparison_parts(arm, payload)
    where = f"comparison[{arm}].aggregate"
    money = _tokens_and_cost(arm, aggregate)
    non_inferior = aggregate.get("non_inferior")
    if not isinstance(non_inferior, bool):
        raise ReportError(f"{where}.non_inferior must be a bool: {non_inferior!r}")
    return [
        arm,
        "comparison",
        _count(aggregate, "n_pairs", where),
        _number(aggregate, "reference_mean", where),
        _number(aggregate, "candidate_mean", where),
        _number(aggregate, "mean_delta", where),
        _number(ci, "low", f"{where}.ci"),
        _number(ci, "high", f"{where}.ci"),
        _number(aggregate, "margin", where),
        non_inferior,
        _candidate_validity(arm, payload),
        _text(worst, "task", f"{where}.worst_task"),
        _number(worst, "delta", f"{where}.worst_task"),
        money["tokens_reference"],
        money["tokens_candidate"],
        money["cost_reference"],
        money["cost_candidate"],
    ]


def _control_row(arm: str, payload: Mapping[str, object]) -> list[object]:
    aggregate, ci, worst = _comparison_parts(arm, payload)
    where = f"comparison[{arm}].aggregate"
    money = _tokens_and_cost(arm, aggregate)
    non_inferior = aggregate.get("non_inferior")
    if not isinstance(non_inferior, bool):
        raise ReportError(f"{where}.non_inferior must be a bool: {non_inferior!r}")
    return [
        arm,
        _number(aggregate, "mean_delta", where),
        _number(ci, "low", f"{where}.ci"),
        _number(ci, "high", f"{where}.ci"),
        _number(aggregate, "margin", where),
        non_inferior,
        _text(worst, "task", f"{where}.worst_task"),
        _number(worst, "delta", f"{where}.worst_task"),
        money["tokens_reference"],
        money["tokens_candidate"],
        money["cost_reference"],
        money["cost_candidate"],
    ]


def _cost_row(arm: str, payload: Mapping[str, object]) -> list[object] | None:
    body = _mapping(payload, f"comparison[{arm}]")
    k = body.get("k")
    if k is None:
        return None
    aggregate, _, _ = _comparison_parts(arm, payload)
    k_block = _mapping(k, f"comparison[{arm}].k")
    reduction = _mapping(k_block.get("reduction_pct"), f"comparison[{arm}].k.reduction_pct")
    lines_pct = _number(reduction, "k_lines_nb", f"comparison[{arm}].k.reduction_pct")
    ast_pct = _number(reduction, "k_ast", f"comparison[{arm}].k.reduction_pct")
    money = _tokens_and_cost(arm, aggregate)
    arms = _mapping(body.get("arms"), f"comparison[{arm}].arms")
    reference = _mapping(arms.get("reference"), f"comparison[{arm}].arms.reference")
    candidate = _mapping(arms.get("candidate"), f"comparison[{arm}].arms.candidate")
    ref_agg = _mapping(reference.get("aggregate"), f"comparison[{arm}].arms.reference.aggregate")
    cand_agg = _mapping(candidate.get("aggregate"), f"comparison[{arm}].arms.candidate.aggregate")
    trials = _count(ref_agg, "valid_trials", f"comparison[{arm}].arms.reference.aggregate") + _count(
        cand_agg, "valid_trials", f"comparison[{arm}].arms.candidate.aggregate"
    )
    tokens_total = money["tokens_reference"] + money["tokens_candidate"]
    cost_total = money["cost_reference"] + money["cost_candidate"]
    if lines_pct > 0:
        trials_per = round(trials / lines_pct, 4)
        tokens_per = round(tokens_total / lines_pct, 4)
        cost_per = round(cost_total / lines_pct, 4)
    else:
        trials_per = tokens_per = cost_per = ""
    return [
        arm,
        lines_pct,
        ast_pct,
        trials,
        tokens_total,
        cost_total,
        trials_per,
        tokens_per,
        cost_per,
    ]


def _arms_table(
    baseline: Mapping[str, object] | None,
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows: list[list[object]] = []
    source: list[str] = []
    if baseline is not None:
        rows.append(_baseline_row(baseline))
        source.append("baseline.json")
    for arm in sorted(comparisons):
        rows.append(_comparison_row(arm, comparisons[arm]))
        source.append(f"comparison:{arm}")
    return _table(T2_COLUMNS, rows, source)


def _controls_table(
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    arms = [arm for arm in sorted(comparisons) if arm in CONTROL_ARMS]
    rows = [_control_row(arm, comparisons[arm]) for arm in arms]
    return _table(T4_COLUMNS, rows, [f"comparison:{arm}" for arm in arms])


def _deletions_table(ledger: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows: list[list[object]] = []
    for index, entry in enumerate(ledger):
        where = f"ledger[{index}]"
        record = _mapping(entry, where)
        round_number = record.get("round")
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            raise ReportError(f"{where}.round must be an integer: {round_number!r}")
        evidence = record.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise ReportError(f"{where}.evidence must be a mapping: {evidence!r}")
        rows.append(
            [
                round_number,
                _text(record, "candidate_id", where),
                _text(record, "verdict", where),
                _text(record, "reason", where),
                evidence.get("layer") or "",
                evidence.get("verify") or "",
            ]
        )
    return _table(T3_COLUMNS, rows, ["ledger.jsonl"] if rows else [])


def _diagnosis_table(diagnosis: Mapping[str, object] | None) -> dict[str, object]:
    if diagnosis is None:
        return _table(T5_COLUMNS, [], [])
    document = _mapping(diagnosis, "diagnosis")
    unknown = sorted(set(document) - set(DIAGNOSIS_PHASES))
    if unknown:
        raise ReportError(f"diagnosis has unknown phase(s): {', '.join(unknown)}")
    rows: list[list[object]] = []
    for phase in sorted(document):
        block = _mapping(document[phase], f"diagnosis.{phase}")
        for metric in sorted(block):
            rows.append([phase, metric, _number(block, metric, f"diagnosis.{phase}")])
    return _table(T5_COLUMNS, rows, ["score.json"] if rows else [])


def _cost_table(
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows: list[list[object]] = []
    source: list[str] = []
    for arm in sorted(comparisons):
        row = _cost_row(arm, comparisons[arm])
        if row is not None:
            rows.append(row)
            source.append(f"comparison:{arm}")
    return _table(T6_COLUMNS, rows, source)


def build_tables(
    *,
    run: Mapping[str, object] | None = None,
    prestudy: Mapping[str, object] | None = None,
    baseline: Mapping[str, object] | None = None,
    comparisons: Mapping[str, Mapping[str, object]] | None = None,
    ledger: Sequence[Mapping[str, object]] = (),
    diagnosis: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    if run is not None:
        _mapping(run, "run")
    if comparisons is None:
        comparisons = {}
    if not isinstance(comparisons, Mapping):
        raise ReportError(f"comparisons must be a mapping: {comparisons!r}")
    for arm in comparisons:
        if not isinstance(arm, str) or not arm:
            raise ReportError(f"comparison arm must be a non-empty string: {arm!r}")
    if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
        raise ReportError(f"ledger must be a sequence: {ledger!r}")
    return {
        "t1_subject": _subject_table(prestudy),
        "t2_arms": _arms_table(baseline, comparisons),
        "t3_deletions": _deletions_table(ledger),
        "t4_baselines": _controls_table(comparisons),
        "t5_diagnosis": _diagnosis_table(diagnosis),
        "t6_cost": _cost_table(comparisons),
    }


def render_table(table: Mapping[str, object]) -> str:
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ReportError(f"table must have column and row lists: {table!r}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        if not isinstance(row, list):
            raise ReportError(f"table row must be a list: {row!r}")
        writer.writerow(row)
    return buffer.getvalue()
