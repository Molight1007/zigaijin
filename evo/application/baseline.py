from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from evo.contracts.types import KVector
from evo.domain.gate import check_validity


STATUSES = ("completed", "agent_error", "infra_error")
DECISIONS = ("baseline_valid", "baseline_invalid")


class BaselineError(ValueError):
    """A baseline input violates the pre-registered protocol."""


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BaselineError(f"{field} must be a non-empty string: {value!r}")
    return value


def _optional_number(value: object, field: str, *, integer: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"{field} must be a number or None: {value!r}")
    if not math.isfinite(float(value)):
        raise BaselineError(f"{field} must be finite: {value!r}")
    if float(value) < 0:
        raise BaselineError(f"{field} must be non-negative: {value!r}")
    if integer and isinstance(value, float) and not value.is_integer():
        raise BaselineError(f"{field} must be an integer: {value!r}")


@dataclass(frozen=True)
class TrialOutcome:
    task_id: str
    trial_id: str
    status: str = "completed"
    reward: float | None = None
    tokens_in: int | None = None  # includes cache tokens (Harbor semantics)
    tokens_out: int | None = None
    cost_usd: float | None = None
    failure_cause: str | None = None

    def __post_init__(self) -> None:
        _name(self.task_id, "task_id")
        _name(self.trial_id, "trial_id")
        if self.status not in STATUSES:
            raise BaselineError(
                f"status must be one of {STATUSES}: {self.status!r}"
            )
        if self.status == "completed" and self.reward is None:
            raise BaselineError("completed trial requires a reward")
        _optional_number(self.reward, "reward")
        _optional_number(self.tokens_in, "tokens_in", integer=True)
        _optional_number(self.tokens_out, "tokens_out", integer=True)
        _optional_number(self.cost_usd, "cost_usd")
        if self.failure_cause is not None:
            _name(self.failure_cause, "failure_cause")


def _task_row(records: Sequence[TrialOutcome], k: int) -> dict[str, object]:
    valid = [record for record in records if record.status == "completed"]
    rewards = [float(record.reward) for record in valid if record.reward is not None]
    tokens_in = sum(record.tokens_in or 0 for record in valid)
    tokens_out = sum(record.tokens_out or 0 for record in valid)
    cost = sum(record.cost_usd or 0.0 for record in valid)
    with_tokens = sum(
        1
        for record in valid
        if record.tokens_in is not None and record.tokens_out is not None
    )
    with_costs = sum(1 for record in valid if record.cost_usd is not None)
    agent_errors = sum(1 for record in records if record.status == "agent_error")
    infra_errors = sum(1 for record in records if record.status == "infra_error")
    missing = k - len(records)
    return {
        "trials": len(records),
        "valid_trials": len(valid),
        "scored_trials": len(valid) + agent_errors,
        "agent_errors": agent_errors,
        "infra_errors": infra_errors,
        "missing_trials": missing,
        # Strict pass@1 over the pre-registered k trials: every non-pass
        # outcome (agent error, infra error, absent record) contributes 0.
        "mean_reward": sum(rewards) / k,
        "valid_mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "trials_with_tokens": with_tokens,
        "tokens_complete": with_tokens == len(valid),
        "cost_complete": with_costs == len(valid),
    }


def summarize_trials(
    outcomes: Sequence[TrialOutcome],
    *,
    tasks: Sequence[str],
    k: int,
) -> dict[str, object]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise BaselineError(f"k must be a positive integer: {k!r}")
    task_ids = list(tasks)
    if not task_ids:
        raise BaselineError("tasks must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise BaselineError("tasks contain a duplicate")
    for task_id in task_ids:
        _name(task_id, "task id")

    by_task: dict[str, list[TrialOutcome]] = {task_id: [] for task_id in task_ids}
    seen: set[str] = set()
    for record in outcomes:
        if record.task_id not in by_task:
            raise BaselineError(f"unknown task {record.task_id!r}")
        if record.trial_id in seen:
            raise BaselineError(f"duplicate trial id {record.trial_id!r}")
        seen.add(record.trial_id)
        by_task[record.task_id].append(record)

    planned = len(task_ids) * k
    rows: dict[str, dict[str, object]] = {}
    valid_total = 0
    scored_total = 0
    agent_total = 0
    infra_total = 0
    missing_total = 0
    reported_total = 0
    task_means: list[float] = []
    valid_means: list[float] = []
    tokens_in_total = 0
    tokens_out_total = 0
    cost_total = 0.0

    for task_id in task_ids:
        records = by_task[task_id]
        if len(records) > k:
            raise BaselineError(
                f"task {task_id!r} exceeds the pre-registered k={k}: {len(records)} trials"
            )
        row = _task_row(records, k)
        rows[task_id] = row
        valid_total += int(row["valid_trials"])
        scored_total += int(row["scored_trials"])
        agent_total += int(row["agent_errors"])
        infra_total += int(row["infra_errors"])
        missing_total += int(row["missing_trials"])
        reported_total += int(row["trials_with_tokens"])
        task_means.append(float(row["mean_reward"]))
        if row["valid_mean_reward"] is not None:
            valid_means.append(float(row["valid_mean_reward"]))
        tokens_in_total += int(row["tokens_in"])
        tokens_out_total += int(row["tokens_out"])
        cost_total += float(row["cost_usd"])

    missing_frac = missing_total / planned
    infra_frac = infra_total / planned
    agent_fail_frac = agent_total / planned
    token_coverage = (reported_total / valid_total) if valid_total else 0.0
    complete = missing_total == 0
    validity = check_validity(
        missing_frac=missing_frac,
        infra_frac=infra_frac,
        token_coverage=token_coverage,
        complete=complete,
        agent_fail_frac=agent_fail_frac,
    )
    tokens_complete = all(bool(row["tokens_complete"]) for row in rows.values())
    cost_complete = all(bool(row["cost_complete"]) for row in rows.values())

    return {
        "tasks": rows,
        "aggregate": {
            "tasks": len(task_ids),
            "tasks_with_valid_trials": len(valid_means),
            "planned_trials": planned,
            "valid_trials": valid_total,
            "scored_trials": scored_total,
            "mean_reward": sum(task_means) / len(task_means),
            "valid_mean_reward": (
                sum(valid_means) / len(valid_means) if valid_means else None
            ),
            "tokens_in": tokens_in_total,
            "tokens_out": tokens_out_total,
            "cost_usd": cost_total,
            "tokens_complete": tokens_complete,
            "cost_complete": cost_complete,
        },
        "validity": {
            "status": validity.status,
            "missing_frac": missing_frac,
            "infra_frac": infra_frac,
            "agent_fail_frac": agent_fail_frac,
            "token_coverage": token_coverage,
            "complete": complete,
            "reasons": list(validity.reasons),
        },
    }


def build_baseline(
    outcomes: Sequence[TrialOutcome],
    *,
    tasks: Sequence[str],
    k: int,
    k_vector: KVector,
) -> dict[str, object]:
    if not isinstance(k_vector, KVector):
        raise BaselineError(f"k_vector must be a KVector: {k_vector!r}")
    summary = summarize_trials(outcomes, tasks=tasks, k=k)
    decision = (
        "baseline_valid"
        if summary["validity"]["status"] == "pass"
        else "baseline_invalid"
    )
    return {
        **summary,
        "k": k_vector.as_dict(),
        "k_reps": k,
        "decision": decision,
    }
