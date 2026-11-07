from __future__ import annotations

from typing import Callable, Mapping, Sequence

from evo.application.baseline import BaselineError, TrialOutcome, summarize_trials
from evo.domain.ablation import (
    AblationDecision,
    AblationError,
    AblationPlan,
    Lever,
    Validity,
    decide_ablation,
)


class AblateRunError(ValueError):
    """An ablation run cannot be assembled from the supplied measurements."""


def _tokens_by_task(
    rows: Sequence[TrialOutcome], tasks: Sequence[str]
) -> dict[str, int] | None:
    totals = {task: 0 for task in tasks}
    seen = {task: 0 for task in tasks}
    for row in rows:
        if row.task_id not in totals:
            continue
        seen[row.task_id] += 1
        if row.tokens_in is None or row.tokens_out is None:
            return None
        totals[row.task_id] += int(row.tokens_in) + int(row.tokens_out)
    if any(count == 0 for count in seen.values()):
        return None
    return totals


def _means(summary: Mapping[str, object], tasks: Sequence[str], name: str) -> list[float]:
    values = []
    for task in tasks:
        value = summary["tasks"][task]["mean_reward"]
        if value is None:
            raise AblateRunError(f"{name} has no paired reward for task {task!r}")
        values.append(float(value))
    return values


def run_ablations(
    plan: AblationPlan,
    evaluate: Callable[[Lever], Sequence[TrialOutcome]],
    *,
    reference: Sequence[TrialOutcome],
    tasks: Sequence[str],
    k: int,
    margin: float,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Force each lever off, measure, and record its attribution verdict."""
    if not isinstance(plan, AblationPlan):
        raise AblateRunError("plan must be an AblationPlan")
    task_ids = tuple(tasks)
    if not task_ids:
        raise AblateRunError("tasks must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise AblateRunError("tasks contain a duplicate")
    try:
        on_summary = summarize_trials(list(reference), tasks=task_ids, k=k)
    except BaselineError as exc:
        raise AblateRunError(f"reference outcomes: {exc}") from exc

    rows: list[dict[str, object]] = []
    for lever in plan.levers:
        try:
            measured = list(evaluate(lever))
        except Exception as exc:  # noqa: BLE001 - a failed job is an invalid lever
            rows.append(
                {
                    "lever": lever.id,
                    "flag": lever.flag,
                    "components": list(lever.components),
                    "description": lever.description,
                    "agent_kwargs": plan.kwargs_for(lever.id),
                    "verdict": "invalid",
                    "reasons": [f"evaluation failed: {exc}"],
                    "decision": None,
                }
            )
            continue
        for row in measured:
            if not isinstance(row, TrialOutcome):
                raise AblateRunError(f"lever {lever.id}: non-TrialOutcome measurement")
        try:
            off_summary = summarize_trials(measured, tasks=task_ids, k=k)
        except BaselineError as exc:
            raise AblateRunError(f"lever {lever.id}: {exc}") from exc

        on_validity = on_summary["validity"]
        off_validity = off_summary["validity"]
        reasons = list(on_validity["reasons"]) + [
            f"off: {reason}" for reason in off_validity["reasons"]
        ]
        validity = Validity(
            ok=on_validity["status"] == "pass" and off_validity["status"] == "pass",
            reasons=tuple(reasons),
        )
        try:
            on_rewards = _means(on_summary, task_ids, f"reference of {lever.id}")
            off_rewards = _means(off_summary, task_ids, f"lever {lever.id}")
            on_tokens = _tokens_by_task(list(reference), task_ids)
            off_tokens = _tokens_by_task(measured, task_ids)
        except AblateRunError as exc:
            validity = Validity(ok=False, reasons=(str(exc),))

        decision: AblationDecision | None = None
        if validity.ok:
            try:
                decision = decide_ablation(
                    task_ids,
                    on_rewards,
                    off_rewards,
                    margin=margin,
                    validity=validity,
                    on_tokens=list(on_tokens.values()) if on_tokens else None,
                    off_tokens=list(off_tokens.values()) if off_tokens else None,
                    alpha=alpha,
                    n_boot=n_boot,
                    seed=seed,
                )
            except AblationError as exc:
                raise AblateRunError(f"lever {lever.id}: {exc}") from exc

        rows.append(
            {
                "lever": lever.id,
                "flag": lever.flag,
                "components": list(lever.components),
                "description": lever.description,
                "agent_kwargs": plan.kwargs_for(lever.id),
                "verdict": "invalid" if decision is None else decision.verdict,
                "reasons": list(validity.reasons)
                if decision is None
                else list(decision.reasons),
                "decision": None if decision is None else _decision_dict(decision),
                "on": _arm_dict(on_summary),
                "off": _arm_dict(off_summary),
            }
        )
    return {
        "schema": 1,
        "subject": plan.subject,
        "tasks": list(task_ids),
        "k": k,
        "margin": margin,
        "alpha": alpha,
        "n_boot": n_boot,
        "levers": rows,
    }


def _arm_dict(summary: Mapping[str, object]) -> dict[str, object]:
    aggregate = summary["aggregate"]
    return {
        "score": aggregate["mean_reward"],
        "tokens_in": aggregate["tokens_in"],
        "tokens_out": aggregate["tokens_out"],
        "valid_trials": aggregate["valid_trials"],
        "validity": dict(summary["validity"]),
    }


def _decision_dict(decision: AblationDecision) -> dict[str, object]:
    delta = decision.delta
    return {
        "verdict": decision.verdict,
        "delta": None
        if delta is None
        else {"estimate": delta.estimate, "low": delta.low, "high": delta.high},
        "margin": decision.margin,
        "n_pairs": decision.n_pairs,
        "token_ratio": decision.token_ratio,
        "worst_token_task": None
        if decision.worst_token_task is None
        else {
            "task": decision.worst_token_task[0],
            "ratio": decision.worst_token_task[1],
        },
        "reasons": list(decision.reasons),
    }
