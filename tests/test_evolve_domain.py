from __future__ import annotations

import math

import pytest

from evo.domain.evolve import (
    EvolveError,
    EvolveMeasurement,
    EvolvePolicy,
    cost_rule,
    edit_budget,
    exploration,
    judge,
    novelty,
    prune_set,
    select_round,
    stall_flag,
    tried_components,
)


def measurement(variant, score, *, tokens=100.0, valid=True, failure=None,
                components=(), k_lines=None):
    return EvolveMeasurement(
        variant=variant,
        score=score,
        tokens_per_trial=tokens,
        valid=valid,
        gate_failure=failure,
        components=tuple(components),
        k_lines=k_lines,
    )


def test_edit_budget_anneals_from_max_to_min():
    budgets = [edit_budget(t, 10, b_min=1, b_max=4) for t in range(11)]
    assert budgets[0] == 4
    assert budgets[-1] == 1
    assert all(a >= b for a, b in zip(budgets, budgets[1:]))


def test_edit_budget_rejects_bad_bounds():
    with pytest.raises(EvolveError, match="b_min"):
        edit_budget(0, 4, b_min=0, b_max=2)
    with pytest.raises(EvolveError, match="b_max"):
        edit_budget(0, 4, b_min=3, b_max=2)
    with pytest.raises(EvolveError, match="rounds"):
        edit_budget(0, 0, b_min=1, b_max=2)


def test_stall_flag_fires_when_window_gain_within_delta():
    trajectory = [0.50, 0.52, 0.505]
    assert stall_flag(trajectory, 2, window=2, delta=0.01) is True
    assert stall_flag(trajectory, 2, window=2, delta=0.001) is False
    assert stall_flag(trajectory, 1, window=2, delta=0.01) is False


def test_exploration_reserves_slots_only_when_stalled():
    stalled = exploration(("a", "b", "c"), {"a"}, stalled=True, m_draft=1)
    assert stalled["untried"] == ["b", "c"]
    assert stalled["reserved"] == 1
    calm = exploration(("a", "b", "c"), {"a"}, stalled=False, m_draft=1)
    assert calm["reserved"] == 0


def test_exploration_rejects_unknown_tried_component():
    with pytest.raises(EvolveError, match="unknown"):
        exploration(("a", "b"), {"zzz"}, stalled=False, m_draft=1)


def test_novelty_counts_components_without_accepted_edits():
    assert novelty(("a", "b", "a"), {"a": 2}) == 1
    assert novelty(("a",), {}) == 1


def test_cost_rule_allows_a_gain_to_buy_cost():
    policy = EvolvePolicy()
    ok, reason = cost_rule(0.05, 0.50, 0, 0.02, policy)
    assert ok is True
    assert "budget" in reason


def test_cost_rule_rejects_excessive_cost():
    policy = EvolvePolicy(beta0=0.10, beta1=40.0)
    ok, reason = cost_rule(0.05, 3.0, 0, 0.02, policy)
    assert ok is False
    assert "cost change" in reason


def test_inside_noise_band_needs_shaped_utility():
    policy = EvolvePolicy()
    ok, _ = cost_rule(0.01, 0.0, 0, 0.02, policy)
    assert ok is True
    ok, _ = cost_rule(0.0, 0.0, 0, 0.02, policy)
    assert ok is False
    ok, _ = cost_rule(0.0, 0.0, 1, 0.02, policy)
    assert ok is True


def test_floor_rejects_below_s_star_minus_delta():
    decision = judge(
        measurement("A", 0.63),
        incumbent_score=0.60,
        incumbent_tokens=100.0,
        s_star=0.70,
        delta=0.05,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert decision.admissible is False
    assert "floor" in decision.reason


def test_invalid_measurement_is_not_admissible():
    decision = judge(
        measurement("A", None, valid=False),
        incumbent_score=0.6,
        incumbent_tokens=100.0,
        s_star=0.6,
        delta=0.01,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert decision.admissible is False
    assert "invalid" in decision.reason


def test_unknown_incumbent_tokens_are_inadmissible():
    # Cost rule compares against the incumbent; an unknown denominator must
    # not silently become dC = 0.0.
    decision = judge(
        measurement("A", 0.62),
        incumbent_score=0.60,
        incumbent_tokens=None,
        s_star=0.60,
        delta=0.01,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert decision.admissible is False
    assert "incumbent token data missing" in decision.reason


def test_failed_candidate_reports_the_screening_failure():
    decision = judge(
        measurement("A", None, failure="critic_reject"),
        incumbent_score=0.6,
        incumbent_tokens=100.0,
        s_star=0.6,
        delta=0.01,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert decision.admissible is False
    assert decision.reason == "candidate failed before measurement: critic_reject"


def test_winner_is_max_score_among_admissible():
    candidates = [
        measurement("A", 0.62),
        measurement("B", 0.66),
        measurement("C", 0.64, valid=False),
    ]
    winner, decisions = select_round(
        candidates,
        incumbent_score=0.60,
        incumbent_tokens=100.0,
        s_star=0.60,
        delta=0.01,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert winner is not None and winner.variant == "B"
    assert [d.admissible for d in decisions] == [True, True, False]


def test_ties_break_on_lower_tokens_then_lower_k_lines():
    candidates = [
        measurement("A", 0.66, tokens=120.0, k_lines=30),
        measurement("B", 0.66, tokens=90.0, k_lines=40),
        measurement("C", 0.66, tokens=90.0, k_lines=20),
    ]
    winner, _ = select_round(
        candidates,
        incumbent_score=0.60,
        incumbent_tokens=100.0,
        s_star=0.60,
        delta=0.01,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert winner is not None and winner.variant == "C"


def test_tried_components_counts_only_measured_edits():
    records = [
        {"t": 0, "component": "a", "delta_S": 0.01},
        {"t": 0, "component": "b", "delta_S": None},
        {"t": 1, "component": "c", "delta_S": -0.02},
    ]
    assert tried_components(records) == {"a", "c"}


def test_prune_set_lists_components_with_nonpositive_recent_yield():
    records = [
        {"t": 4, "component": "a", "delta_S": -0.01, "accepted": True, "edit_id": "C1"},
        {"t": 5, "component": "b", "delta_S": 0.02, "accepted": True, "edit_id": "C2"},
        {"t": 1, "component": "c", "delta_S": 0.03, "accepted": False, "edit_id": "C3"},
    ]
    pruned = prune_set(records, t=5, n_prune=2)
    by_component = {row["component"]: row for row in pruned}
    assert "b" not in by_component
    assert by_component["a"]["recent_best_gain"] == -0.01
    assert by_component["a"]["accepted_edits_in_incumbent"][0]["edit_id"] == "C1"
    assert by_component["c"]["recent_best_gain"] is None


def test_policy_rejects_non_finite_weights():
    with pytest.raises(EvolveError, match="beta0"):
        EvolvePolicy(beta0=math.nan)
    with pytest.raises(EvolveError, match="w_s"):
        EvolvePolicy(w_s=-1.0)


def test_k_limit_is_enforced():
    item = EvolveMeasurement(
        variant="A",
        score=0.6,
        tokens_per_trial=100.0,
        k_lines=120,
        k_limit=100,
    )
    decision = judge(
        item,
        incumbent_score=0.5,
        incumbent_tokens=100.0,
        s_star=0.5,
        delta=0.01,
        policy=EvolvePolicy(),
        accepted_counts={},
    )
    assert decision.admissible is False
    assert "k growth" in decision.reason
