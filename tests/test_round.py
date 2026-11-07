from __future__ import annotations

import pytest

from evo.application.baseline import TrialOutcome
from evo.application.round import RoundError, decide_from_artifacts
from evo.domain.scan import Candidate


POOLS = {"evolution": ("t1", "t2", "t3", "t4"), "holdout": ("h1", "h2", "h3")}


def outcomes(tasks, reward, *, prefix, tokens=100):
    return [
        TrialOutcome(
            task_id=task,
            trial_id=f"{prefix}-{task}",
            reward=reward[task],
            tokens_in=tokens,
            tokens_out=10,
            cost_usd=0.001,
        )
        for task in tasks
    ]


def candidate(layer="L1", protection=()):
    return Candidate(
        id="l1-0001",
        layer=layer,
        kind="dead_function",
        component="demo",
        path="pkg/mod.py",
        span=(1, 2),
        symbol="_dead",
        protection=protection,
    )


def applied(**overrides):
    payload = {
        "before": {
            "k_lines_nb": 200,
            "k_ast": 400,
            "k_chars": 1000,
            "k_bytes": 1000,
            "k_gzip": 100,
            "k_cc": 10,
        },
        "after": {
            "k_lines_nb": 180,
            "k_ast": 360,
            "k_chars": 900,
            "k_bytes": 900,
            "k_gzip": 90,
            "k_cc": 9,
        },
        "verify": {
            "required": ["compile", "import"],
            "outcomes": [
                {"name": "compile", "status": "pass", "detail": ""},
                {"name": "import", "status": "pass", "detail": ""},
            ],
        },
        "undeletable": False,
        "undeletable_reason": "",
    }
    payload.update(overrides)
    return payload


def decide(reference=None, evaluated=None, **overrides):
    rewards = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    kwargs = {
        "pools": POOLS,
        "candidate": candidate(),
        "applied": applied(),
        "reference": reference
        if reference is not None
        else outcomes(rewards, rewards, prefix="ref"),
        "evaluated": evaluated
        if evaluated is not None
        else outcomes(rewards, rewards, prefix="sim"),
        "k": 1,
        "n_boot": 400,
        "seed": 3,
    }
    kwargs.update(overrides)
    return decide_from_artifacts(**kwargs)


def test_full_pass_deletes_the_candidate():
    decision = decide()
    assert decision.verdict == "delete"
    assert decision.reason == "all gates pass"
    assert decision.verify_status == "pass"
    assert decision.gate.verdict == "accept"


def test_agent_error_invalidates_the_candidate():
    rewards = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    reference = outcomes(rewards, rewards, prefix="ref")
    evaluated = outcomes(rewards, rewards, prefix="sim")
    broken = evaluated[0]
    evaluated[0] = TrialOutcome(
        task_id=broken.task_id,
        trial_id=broken.trial_id,
        status="agent_error",
        reward=None,
    )
    decision = decide(reference=reference, evaluated=evaluated)
    assert decision.verdict == "undecidable"
    assert decision.gate is not None
    assert decision.gate.verdict == "invalid"
    assert "agent_fail_frac" in " ".join(decision.gate.reasons)


def test_a_reward_regression_keeps_the_candidate():
    rewards = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    regressed = dict(rewards)
    regressed["t1"] = 0.0
    decision = decide(evaluated=outcomes(regressed, regressed, prefix="sim"))
    assert decision.verdict == "keep"
    assert "gate rejected" in decision.reason


def test_unknown_required_checks_make_the_candidate_undecidable():
    payload = applied(
        verify={
            "required": ["compile", "import", "smoke", "replay"],
            "outcomes": [
                {"name": "compile", "status": "pass", "detail": ""},
                {"name": "import", "status": "pass", "detail": ""},
                {"name": "smoke", "status": "pass", "detail": ""},
            ],
        }
    )
    decision = decide(candidate=candidate(layer="L3"), applied=payload)
    assert decision.verdict == "undecidable"
    assert "replay" in decision.reason


def test_undeletable_takes_precedence_over_the_gate():
    payload = applied(
        undeletable=True,
        undeletable_reason="residual call site for _dead",
    )
    decision = decide(applied=payload)
    assert decision.verdict == "undeletable"
    assert "_dead" in decision.reason


def test_incomplete_holdout_evidence_is_refused():
    rewards = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    partial = {task: 1.0 for task in POOLS["evolution"]}
    with pytest.raises(RoundError, match="incomplete"):
        decide(
            reference=outcomes(rewards, rewards, prefix="ref"),
            evaluated=outcomes(partial, partial, prefix="sim"),
        )


def test_unknown_pool_is_refused():
    with pytest.raises(RoundError, match="pool"):
        decide(scores_pool="nope")


def test_missing_k_vectors_are_refused():
    with pytest.raises(RoundError, match="K"):
        decide(applied=applied(after=None))


def test_one_task_may_grow_when_the_total_falls():
    rewards = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    reference = outcomes(rewards, rewards, prefix="ref", tokens=100)
    evaluated = outcomes(rewards, rewards, prefix="sim", tokens=90)
    grown = evaluated[0]
    evaluated[0] = TrialOutcome(
        task_id=grown.task_id,
        trial_id=grown.trial_id,
        reward=grown.reward,
        tokens_in=105,
        tokens_out=grown.tokens_out,
        cost_usd=grown.cost_usd,
    )
    decision = decide(reference=reference, evaluated=evaluated)
    assert decision.verdict == "delete"


def test_tokens_beyond_tolerance_keep_the_candidate():
    rewards = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    reference = outcomes(rewards, rewards, prefix="ref", tokens=100)
    evaluated = outcomes(rewards, rewards, prefix="sim", tokens=150)
    decision = decide(reference=reference, evaluated=evaluated)
    assert decision.verdict == "keep"
    assert "token" in decision.reason.lower()


def test_missing_tokens_keep_the_candidate(tmp_path):
    tasks = tuple(f"t{i}" for i in range(10))
    pools = {"evolution": tasks, "holdout": ("h1",)}
    rewards = {task: 1.0 for task in tasks + ("h1",)}
    reference = [TrialOutcome(task, f"r-{task}", reward=1.0, tokens_in=100, tokens_out=10) for task in tasks]
    reference += [TrialOutcome("h1", "r-h1", reward=1.0, tokens_in=100, tokens_out=10)]
    evaluated = [TrialOutcome(task, f"e-{task}", reward=1.0,
                              tokens_in=None if task == "t0" else 100,
                              tokens_out=None if task == "t0" else 10) for task in tasks]
    evaluated += [TrialOutcome("h1", "e-h1", reward=1.0, tokens_in=100, tokens_out=10)]
    decision = decide_from_artifacts(
        pools=pools,
        candidate=candidate(),
        applied=applied(),
        reference=reference,
        evaluated=evaluated,
        k=1,
        n_boot=200,
        seed=0,
    )
    assert decision.verdict != "delete"
    assert "token" in decision.reason.lower()


def test_direction_is_recorded_when_feedback_and_holdout_agree():
    rewards = {task: 0.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    improved = {task: 1.0 for task in POOLS["evolution"] + POOLS["holdout"]}
    decision = decide(
        reference=outcomes(rewards, rewards, prefix="ref"),
        evaluated=outcomes(improved, improved, prefix="sim"),
        direction_delta=0.01,
    )
    assert decision.direction is not None
    assert decision.direction["computed"] is True
    assert decision.direction["consistent"] is True
    assert decision.as_dict()["direction"]["consistent"] is True


def test_direction_is_recorded_when_holdout_moves_the_other_way():
    reference_rewards = {
        **{task: 0.0 for task in POOLS["evolution"]},
        **{task: 1.0 for task in POOLS["holdout"]},
    }
    candidate_rewards = {
        **{task: 1.0 for task in POOLS["evolution"]},
        **{task: 0.0 for task in POOLS["holdout"]},
    }
    decision = decide(
        reference=outcomes(reference_rewards, reference_rewards, prefix="ref"),
        evaluated=outcomes(candidate_rewards, candidate_rewards, prefix="sim"),
        direction_delta=0.01,
    )
    assert decision.direction is not None
    assert decision.direction["computed"] is True
    assert decision.direction["consistent"] is False
