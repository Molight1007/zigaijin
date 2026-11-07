from __future__ import annotations

import pytest

from evo.application.ablate import AblateRunError, run_ablations
from evo.application.baseline import TrialOutcome
from evo.domain.ablation import AblationPlan, Lever


TASKS = ("t1", "t2")


def outcomes(rewards, *, prefix, tokens_in=90, tokens_out=10, tasks=TASKS):
    rows = []
    for task in tasks:
        rows.append(
            TrialOutcome(
                task_id=task,
                trial_id=f"{prefix}-{task}",
                reward=rewards[task],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=0.001,
            )
        )
    return rows


def flat(value):
    return {task: value for task in TASKS}


def plan():
    return AblationPlan(
        subject="terminus2",
        levers=(
            Lever(
                id="no_memory",
                flag="memory",
                default_value=True,
                off_value=False,
                components=("memory",),
            ),
            Lever(
                id="no_tools",
                flag="tools",
                default_value=True,
                off_value=False,
                components=("tools",),
            ),
        ),
    )


def reference():
    return outcomes(flat(0.5), prefix="on")


def test_zero_contribution_and_contributing_levers_are_separated():
    def evaluate(lever):
        if lever.id == "no_memory":
            return outcomes(flat(0.5), prefix="memory", tokens_in=70, tokens_out=10)
        return outcomes(flat(0.0), prefix="tools")

    payload = run_ablations(
        plan(),
        evaluate,
        reference=reference(),
        tasks=TASKS,
        k=1,
        margin=0.10,
        n_boot=400,
    )

    rows = {row["lever"]: row for row in payload["levers"]}
    assert rows["no_memory"]["verdict"] == "zero_contribution"
    assert rows["no_memory"]["decision"]["token_ratio"] == pytest.approx(0.8)
    assert rows["no_tools"]["verdict"] == "contributes"
    assert rows["no_tools"]["decision"]["delta"]["high"] == pytest.approx(-0.5)
    assert payload["subject"] == "terminus2"
    assert rows["no_tools"]["agent_kwargs"] == {"tools": False}


def test_missing_off_task_data_makes_the_lever_invalid():
    def evaluate(lever):
        return outcomes(flat(0.5), prefix="partial", tasks=TASKS[:1])

    payload = run_ablations(
        plan(), evaluate, reference=reference(), tasks=TASKS, k=1, margin=0.10, n_boot=400
    )
    rows = {row["lever"]: row for row in payload["levers"]}
    assert rows["no_memory"]["verdict"] == "invalid"


def test_missing_token_data_fails_validity_at_the_first_gate():
    def evaluate(lever):
        return outcomes(flat(0.5), prefix="notokens", tokens_in=None, tokens_out=None)

    payload = run_ablations(
        plan(), evaluate, reference=reference(), tasks=TASKS, k=1, margin=0.10, n_boot=400
    )
    rows = {row["lever"]: row for row in payload["levers"]}
    assert rows["no_memory"]["verdict"] == "invalid"
    assert any("token_coverage" in reason for reason in rows["no_memory"]["reasons"])


def test_failed_evaluation_is_recorded_not_raised():
    def evaluate(lever):
        raise RuntimeError("docker is down")

    payload = run_ablations(
        plan(), evaluate, reference=reference(), tasks=TASKS, k=1, margin=0.10, n_boot=400
    )
    rows = {row["lever"]: row for row in payload["levers"]}
    assert rows["no_memory"]["verdict"] == "invalid"
    assert "docker is down" in rows["no_memory"]["reasons"][0]


def test_duplicate_tasks_are_rejected():
    with pytest.raises(AblateRunError, match="duplicate"):
        run_ablations(
            plan(),
            lambda lever: outcomes(flat(0.5), prefix="x"),
            reference=reference(),
            tasks=("t1", "t1"),
            k=1,
            margin=0.10,
        )
