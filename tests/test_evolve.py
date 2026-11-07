from __future__ import annotations

import pytest

from evo.application.baseline import TrialOutcome
from evo.application.evolve import EvolveRoundError, _attribution, evolve_round
from evo.domain.evolve import EvolvePolicy


TASKS = ("t1", "t2", "t3", "t4")


def outcomes(rewards, *, prefix, k=1, tokens_in=90, tokens_out=10):
    rows = []
    for task in TASKS:
        for rep in range(k):
            rows.append(
                TrialOutcome(
                    task_id=task,
                    trial_id=f"{prefix}-{task}-{rep}",
                    reward=rewards[task],
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=0.001,
                )
            )
    return rows


def flat(value=0.5):
    return {task: value for task in TASKS}


def run(**overrides):
    kwargs = {
        "incumbent": outcomes(flat(), prefix="inc"),
        "candidates": {
            "A": outcomes(flat(0.6), prefix="a"),
            "B": outcomes(flat(0.55), prefix="b"),
        },
        "tasks": TASKS,
        "k": 1,
        "delta": 0.01,
        "round": 0,
        "total_rounds": 10,
        "incumbent_k_lines": None,
        "max_k_growth": 0,
    }
    kwargs.update(overrides)
    return evolve_round(**kwargs)


def test_selects_the_best_admissible_variant():
    result = run()
    assert result["winner"] == "A"
    decisions = {row["variant"]: row for row in result["decisions"]}
    assert decisions["A"]["admissible"] is True
    assert decisions["B"]["admissible"] is True
    outcomes_by_variant = {
        row["variant"]: row["outcome"] for row in result["history"]
    }
    assert outcomes_by_variant == {"A": "ACCEPTED", "B": "LOST"}


def test_cost_rule_can_reject_a_high_token_gain():
    result = run(
        incumbent=outcomes(flat(), prefix="inc", tokens_in=90, tokens_out=10),
        candidates={
            "A": outcomes(flat(0.60), prefix="a", tokens_in=590, tokens_out=10),
            "B": outcomes(flat(0.61), prefix="b", tokens_in=90, tokens_out=10),
        },
        delta=0.005,
    )
    decisions = {row["variant"]: row for row in result["decisions"]}
    assert decisions["A"]["admissible"] is False
    assert "cost rule failed" in decisions["A"]["reason"]
    assert result["winner"] == "B"


def test_missing_tokens_make_a_candidate_inadmissible():
    # Brief 的 4 题版本 token_coverage = 3/4 < 0.90，先被 validity 判无效、
    # 到不了成本判定；用 10 题（覆盖率 0.90）才真正走到 token 缺失分支。
    tasks = tuple(f"t{i}" for i in range(10))
    incumbent = [
        TrialOutcome(task_id=task, trial_id=f"inc-{task}", reward=0.5,
                     tokens_in=90, tokens_out=10)
        for task in tasks
    ]
    result = run(
        incumbent=incumbent,
        tasks=tasks,
        candidates={"A": [
            TrialOutcome(task_id=task, trial_id=f"a-{task}",
                         reward=0.5,
                         tokens_in=None if task == "t1" else 90,
                         tokens_out=None if task == "t1" else 10)
            for task in tasks
        ]},
    )
    decision = result["decisions"][0]
    assert decision["admissible"] is False
    assert "token" in decision["reason"].lower()


def test_candidate_below_the_floor_is_rejected():
    result = run(
        candidates={
            "A": outcomes(flat(0.64), prefix="a"),
        },
        s_star=0.70,
        delta=0.05,
        incumbent=outcomes(flat(0.60), prefix="inc"),
    )
    assert result["winner"] is None
    assert "floor" in result["decisions"][0]["reason"]


def test_incomplete_candidate_is_invalid_and_unmeasured():
    result = run(
        incumbent=outcomes(flat(), prefix="inc", k=2),
        candidates={
            "A": outcomes(flat(0.9), prefix="a", k=1),
        },
        k=2,
    )
    decision = result["decisions"][0]
    assert decision["admissible"] is False
    assert "invalid measurement" in decision["reason"]
    record = result["history"][0]
    assert record["outcome"] == "eval_invalid"
    assert record["delta_S"] is None


def test_invalid_incumbent_raises():
    with pytest.raises(EvolveRoundError, match="incumbent"):
        run(k=2)


def test_history_has_one_record_per_edit_with_the_bundle_measurement():
    edits = {
        "A": [
            {"id": "C1", "component": "core", "hypothesis": "tighten retries"},
            {"id": "C2", "component": "parser", "hypothesis": "drop dead branch"},
        ]
    }
    result = run(edits=edits)
    rows = [row for row in result["history"] if row["variant"] == "A"]
    assert len(rows) == 2
    assert {row["bundle"] for row in rows} == {2}
    assert {row["component"] for row in rows} == {"core", "parser"}
    assert all(row["accepted"] is True for row in rows)
    assert rows[0]["delta_S"] == pytest.approx(0.1)


def test_budget_stall_and_exploration_are_reported():
    result = run(
        round=1,
        total_rounds=10,
        trajectory=[0.5, 0.5],
        components=("core", "parser", "session"),
        history=[{"t": 0, "component": "core", "delta_S": 0.01}],
        stall_window=1,
    )
    assert result["budget"] == 4
    assert result["stall"] is True
    assert result["exploration"]["untried"] == ["parser", "session"]
    assert result["exploration"]["reserved"] == 1


def test_edit_budget_is_enforced():
    # b_t(9, 10) 默认 b_max=4 退火到 2（ceil），把 b_max 收紧到 1 才能复现 brief
    # 的 budget=1 场景：2 个 edit 超预算，必须在测量/选择之前抛错。
    edits = [
        {
            "id": f"E{i}",
            "component": "core",
            "hypothesis": "x",
            "predicted_affected": ["t1"],
        }
        for i in range(2)
    ]
    with pytest.raises(EvolveRoundError, match="budget"):
        run(round=9, total_rounds=10, b_max=1, edits={"A": edits})


def test_prune_set_comes_from_history():
    history = [
        {
            "t": 0,
            "component": "core",
            "delta_S": -0.02,
            "accepted": True,
            "edit_id": "C9",
            "hypothesis": "old mechanism",
        }
    ]
    result = run(round=2, trajectory=[0.5, 0.5, 0.5], history=history)
    assert result["prune_set"][0]["component"] == "core"
    assert result["prune_set"][0]["accepted_edits_in_incumbent"][0]["edit_id"] == "C9"


def test_ties_break_on_lower_tokens():
    result = run(
        candidates={
            "A": outcomes(flat(0.6), prefix="a", tokens_in=190, tokens_out=10),
            "B": outcomes(flat(0.6), prefix="b", tokens_in=90, tokens_out=10),
        }
    )
    assert result["winner"] == "B"


def test_predicted_affected_hits_are_recorded():
    inc = flat(0.5)
    cand = dict(inc)
    cand["t1"] = 0.9
    cand["t2"] = 0.0
    result = run(
        incumbent=outcomes(inc, prefix="inc", k=2),
        candidates={"A": outcomes(cand, prefix="a", k=2)},
        k=2,
        edits={
            "A": [
                {
                    "id": "C1",
                    "component": "core",
                    "hypothesis": "fix t1",
                    "predicted_affected": ["t1", "t3"],
                }
            ]
        },
    )
    record = result["history"][0]
    assert record["predicted_hit"] == ["t1"]
    assert record["hit_rate"] == 0.5
    assert record["unpredicted_regressions"] == ["t2"]
    assert record["prediction_verified"] is False
    assert record["prediction_falsified"] is True


def test_history_flags_continuous_regression_beyond_delta():
    inc = flat(0.8)
    cand = dict(inc)
    cand["t1"] = 0.1
    result = run(
        incumbent=outcomes(inc, prefix="inc"),
        candidates={"A": outcomes(cand, prefix="a")},
        delta=0.05,
        edits={
            "A": [
                {
                    "id": "C1",
                    "component": "core",
                    "hypothesis": "x",
                    "predicted_affected": ["t2"],
                }
            ]
        },
    )
    record = result["history"][0]
    assert record["unpredicted_regressions"] == ["t1"]
    assert record["threshold"] == pytest.approx(0.05)


def test_attribution_without_threshold_keeps_one_over_k_default():
    result = _attribution({"predicted_affected": []}, {"t1": 0.8}, {"t1": 0.1}, 2)
    assert result["threshold"] == pytest.approx(0.5)
    assert result["unpredicted_regressions"] == ["t1"]


def test_s_star_is_carried_forward():
    result = run(s_star=0.8)
    assert result["s_star"] == 0.8
    assert result["winner"] is None


def test_historical_s_star_blocks_cumulative_drift():
    history = [{"t": 0, "variant": "A", "score": 0.9, "accepted": True}]
    result = run(
        history=history,
        round=1,
        total_rounds=5,
        delta=0.05,
        incumbent=outcomes(flat(0.8), prefix="inc"),
        candidates={"A": outcomes(flat(0.81), prefix="a")},
        edits={"A": [{"id": "C1", "component": "core", "hypothesis": "x"}]},
    )
    assert result["s_star"] == pytest.approx(0.9)
    assert result["history"][0]["s_star"] == pytest.approx(0.9)
    assert result["winner"] is None


def test_accepted_scores_ignore_the_current_round():
    # a re-run of round 1 must not use that same round's accepted 0.99 to lift S*
    history = [{"t": 1, "variant": "A", "score": 0.99, "accepted": True}]
    result = run(
        history=history,
        round=1,
        total_rounds=5,
        delta=0.05,
        incumbent=outcomes(flat(0.8), prefix="inc"),
        candidates={"A": outcomes(flat(0.85), prefix="a")},
        edits={"A": [{"id": "C1", "component": "core", "hypothesis": "x"}]},
    )
    assert result["s_star"] == pytest.approx(0.85)
    assert result["winner"] == "A"


def test_trajectory_is_rebuilt_from_history_for_stall():
    history = [{"t": 0, "variant": "A", "score": 0.9, "accepted": True}]
    result = run(
        history=history,
        round=1,
        total_rounds=5,
        stall_window=1,
        delta=0.01,
        incumbent=outcomes(flat(0.9), prefix="inc"),
        candidates={"A": outcomes(flat(0.9), prefix="a")},
        edits={"A": [{"id": "C1", "component": "core", "hypothesis": "x"}]},
    )
    assert result["stall"] is True


def test_bad_k_raises():
    with pytest.raises(EvolveRoundError, match="k"):
        run(k=0)


def test_duplicate_task_raises():
    with pytest.raises(EvolveRoundError, match="duplicate"):
        run(tasks=("t1", "t1"))


def test_gate_failure_rejects_a_candidate_before_measurement():
    result = run(gate_failures={"A": "k gate: k_lines_nb did not drop by 0.500%"})
    decisions = {row["variant"]: row for row in result["decisions"]}
    assert decisions["A"]["admissible"] is False
    assert "failed before measurement" in decisions["A"]["reason"]
    assert "k_lines_nb did not drop" in decisions["A"]["reason"]
    assert result["winner"] == "B"
    outcomes_by_variant = {row["variant"]: row["outcome"] for row in result["history"]}
    assert outcomes_by_variant["A"] == "eval_invalid"


def test_gate_failure_for_an_unknown_variant_is_rejected():
    with pytest.raises(EvolveRoundError, match="unknown variants"):
        run(gate_failures={"Z": "k gate: not evaluated"})


def test_candidate_cannot_grow_k_beyond_limit():
    result = run(
        k_lines={"A": 120},
        incumbent_k_lines=100,
        max_k_growth=0,
        edits={"A": [{"id": "C1", "component": "core", "hypothesis": "x"}]},
    )
    decision = result["decisions"][0]
    assert decision["admissible"] is False
    assert "k growth" in decision["reason"]


@pytest.mark.parametrize(
    "override",
    [
        {"incumbent_k_lines": True},
        {"incumbent_k_lines": -1},
        {"max_k_growth": -1},
        {"max_k_growth": 1.5},
    ],
)
def test_k_limit_inputs_are_validated(override):
    with pytest.raises(EvolveRoundError, match="k_lines|k_growth"):
        run(**override)
