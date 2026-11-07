from __future__ import annotations

import math

import pytest

from evo.contracts.types import KVector
from evo.domain.gate import (
    GateError,
    GateRequest,
    KGateInput,
    ProtectionInput,
    ScoreInput,
    TokenInput,
    ValidityInput,
    check_k,
    check_pareto,
    check_protection,
    check_t4,
    check_tokens,
    check_validity,
    check_worst_task,
    simplification_gate,
)


def scores(
    reference: list[float], candidate: list[float], margin: float = 0.10
) -> ScoreInput:
    tasks = tuple(f"t{index}" for index in range(len(reference)))
    return ScoreInput(
        tasks=tasks,
        reference=tuple(reference),
        candidate=tuple(candidate),
        margin=margin,
    )


def good_k(before_lines: int = 200, after_lines: int = 150) -> KGateInput:
    ratio = after_lines / before_lines
    return KGateInput(
        before=KVector(
            k_lines_nb=before_lines,
            k_ast=1000,
            k_chars=8000,
            k_bytes=8000,
            k_gzip=2000,
            k_cc=80,
        ),
        after=KVector(
            k_lines_nb=after_lines,
            k_ast=int(1000 * ratio),
            k_chars=int(8000 * ratio),
            k_bytes=int(8000 * ratio),
            k_gzip=2000,
            k_cc=int(80 * ratio),
        ),
    )


def good_tokens() -> TokenInput:
    return TokenInput(
        before={"t0": 1000, "t1": 2000, "t2": 3000},
        after={"t0": 1000, "t1": 1900, "t2": 3000},
    )


def good_protection() -> ProtectionInput:
    return ProtectionInput()


def good_request(**overrides) -> GateRequest:
    payload = dict(
        validity=ValidityInput(missing_frac=0.0, infra_frac=0.0, token_coverage=1.0),
        scores=scores([0.8, 0.6, 0.7], [0.8, 0.6, 0.7]),
        holdout=scores([0.5, 0.4], [0.5, 0.4]),
        k=good_k(),
        tokens=good_tokens(),
        protection=good_protection(),
        round=3,
        protection_until=0,
        task_margin=0.10,
        n_boot=200,
    )
    payload.update(overrides)
    return GateRequest(**payload)


def failed_names(result) -> set[str]:
    return {check.name for check in result.checks if check.status == "fail"}


# ---------------------------------------------------------------- validity


def test_validity_passes_at_the_documented_thresholds():
    check = check_validity(0.10, 0.10, 0.90)
    assert check.status == "pass"
    assert check.reasons == ()


@pytest.mark.parametrize(
    ("missing", "infra", "coverage", "complete", "needle"),
    [
        (0.11, 0.0, 1.0, True, "missing_frac"),
        (0.0, 0.11, 1.0, True, "infra_frac"),
        (0.0, 0.0, 0.89, True, "token_coverage"),
        (0.0, 0.0, 1.0, False, "repetitions"),
    ],
)
def test_validity_fails_each_documented_condition(missing, infra, coverage, complete, needle):
    check = check_validity(missing, infra, coverage, complete=complete)
    assert check.status == "fail"
    assert any(needle in reason for reason in check.reasons)


def test_validity_rejects_agent_failures_above_threshold():
    check = check_validity(0.0, 0.0, 1.0, agent_fail_frac=0.5)
    assert check.status == "fail"
    assert "agent_fail_frac" in " ".join(check.reasons)


def test_validity_accepts_agent_failures_at_exactly_the_limit():
    check = check_validity(0.0, 0.0, 1.0, agent_fail_frac=0.10)
    assert check.status == "pass"
    assert check.reasons == ()


def test_validity_rejects_agent_failures_just_above_the_limit():
    check = check_validity(0.0, 0.0, 1.0, agent_fail_frac=0.101)
    assert check.status == "fail"
    assert check.reasons == ("agent_fail_frac 0.101 > 0.100",)


def test_validity_rejects_agent_failures_outside_zero_to_one():
    for value in (-0.01, 1.01, math.nan):
        with pytest.raises(GateError):
            check_validity(0.0, 0.0, 1.0, agent_fail_frac=value)
        with pytest.raises(GateError):
            ValidityInput(0.0, 0.0, 1.0, agent_fail_frac=value)


def test_validity_rejects_out_of_range_inputs():
    with pytest.raises(GateError):
        check_validity(-0.1, 0.0, 1.0)
    with pytest.raises(GateError):
        check_validity(0.0, 0.0, 1.2)
    with pytest.raises(GateError):
        check_validity(0.0, 0.0, math.nan)


def test_gate_returns_invalid_and_runs_no_judgement_when_validity_fails():
    request = good_request(
        validity=ValidityInput(missing_frac=0.3, infra_frac=0.0, token_coverage=1.0)
    )
    result = simplification_gate(request)
    assert result.verdict == "invalid"
    assert [check.name for check in result.checks] == ["validity"]
    assert "missing_frac" in " ".join(result.reasons)


# ---------------------------------------------------------- non-inferiority


def test_gate_accepts_when_every_check_passes():
    result = simplification_gate(good_request())
    assert result.verdict == "accept"
    assert result.reasons == ()
    assert [check.name for check in result.checks] == [
        "validity",
        "non_inferiority",
        "holdout",
        "k",
        "tokens",
        "protection",
        "t4",
        "pareto",
        "worst_task",
    ]


def test_non_inferiority_rejects_when_ci_low_falls_below_negative_margin():
    request = good_request(
        scores=scores([1.0, 1.0, 1.0], [0.85, 0.85, 0.85], margin=0.10)
    )
    result = simplification_gate(request)
    assert result.verdict == "reject"
    assert "non_inferiority" in failed_names(result)


def test_non_inferiority_accepts_loss_within_margin():
    request = good_request(
        scores=scores([1.0, 1.0, 1.0], [0.92, 0.92, 0.92], margin=0.10)
    )
    assert simplification_gate(request).verdict == "accept"


def test_holdout_gate_rejects_even_when_evolution_pool_is_flat():
    request = good_request(
        holdout=scores([0.9, 0.9, 0.9], [0.4, 0.4, 0.4], margin=0.10)
    )
    result = simplification_gate(request)
    assert result.verdict == "reject"
    assert "holdout" in failed_names(result)


def test_non_inferiority_requires_aligned_pairs():
    with pytest.raises(GateError):
        ScoreInput(tasks=("t0",), reference=(1.0, 0.9), candidate=(1.0,), margin=0.1)
    with pytest.raises(GateError):
        ScoreInput(tasks=("t0", "t0"), reference=(1.0, 1.0), candidate=(1.0, 1.0), margin=0.1)
    with pytest.raises(GateError):
        ScoreInput(tasks=("t0",), reference=(1.0,), candidate=(1.0,), margin=0.0)
    with pytest.raises(GateError):
        ScoreInput(tasks=("t0",), reference=(math.inf,), candidate=(1.0,), margin=0.1)


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    reference = [0.8, 0.7, 0.9, 0.6, 0.75]
    candidate = [0.79, 0.71, 0.88, 0.61, 0.74]
    first = simplification_gate(
        good_request(scores=scores(reference, candidate), seed=7)
    )
    second = simplification_gate(
        good_request(scores=scores(reference, candidate), seed=7)
    )
    assert first.checks[1].data["ci_low"] == second.checks[1].data["ci_low"]


# ------------------------------------------------------------------- K gate


def test_k_gate_accepts_documented_reduction():
    check = check_k(good_k().before, good_k().after)
    assert check.status == "pass"
    assert check.data["reduction_pct"]["k_ast"] > 0.5


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("k_ast", 1000, "k_ast"),
        ("k_lines_nb", 195, "k_lines_nb"),
        ("k_chars", 9000, "k_chars"),
        ("k_cc", 90, "k_cc"),
    ],
)
def test_k_gate_rejects_each_documented_failure(field, value, needle):
    k = good_k()
    after = KVector(**{**k.after.as_dict(), field: value})
    check = check_k(k.before, after)
    assert check.status == "fail"
    assert any(needle in reason for reason in check.reasons)


def test_k_gate_requires_at_least_ten_net_lines():
    before = KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10)
    after = KVector(k_lines_nb=95, k_ast=400, k_chars=900, k_cc=9)
    check = check_k(before, after)
    assert check.status == "fail"
    assert any("10" in reason for reason in check.reasons)


def test_k_gate_accepts_the_exact_half_percent_ast_boundary():
    before = KVector(k_lines_nb=1000, k_ast=1000, k_chars=1000, k_cc=10)
    after = KVector(k_lines_nb=900, k_ast=995, k_chars=1000, k_cc=10)
    assert check_k(before, after).status == "pass"


def test_k_gate_rejects_negative_vectors():
    before = KVector(k_lines_nb=-1, k_ast=10, k_chars=10, k_cc=1)
    with pytest.raises(GateError):
        check_k(before, KVector(k_lines_nb=0, k_ast=0, k_chars=0, k_cc=0))


def test_gate_rejects_when_k_gate_fails():
    k = good_k()
    flat = KGateInput(before=k.before, after=KVector(**k.before.as_dict()))
    result = simplification_gate(good_request(k=flat))
    assert result.verdict == "reject"
    assert "k" in failed_names(result)


# -------------------------------------------------------------- token gate


def test_token_gate_rejects_total_increase():
    tokens = TokenInput(
        before={"t0": 1000, "t1": 1000}, after={"t0": 1000, "t1": 1100}
    )
    check = check_tokens(tokens.before, tokens.after)
    assert check.status == "fail"
    assert any("total" in reason for reason in check.reasons)


def test_token_gate_rejects_a_single_task_beyond_ten_percent():
    tokens = TokenInput(
        before={"t0": 1000, "t1": 1000, "t2": 1000},
        after={"t0": 1000, "t1": 1000, "t2": 1200},
    )
    check = check_tokens(tokens.before, tokens.after)
    assert check.status == "fail"
    assert check.data["worst_task"] == "t2"
    assert check.data["worst_ratio"] == pytest.approx(1.2)


def test_token_gate_passes_at_exactly_ten_percent():
    check = check_tokens({"t0": 1000, "t1": 1000}, {"t0": 900, "t1": 1100})
    assert check.status == "pass"
    assert check.data["worst_ratio"] == pytest.approx(1.1)


def test_token_gate_rejects_a_missing_task():
    check = check_tokens({"t0": 1000, "t1": 1000}, {"t0": 1000})
    assert check.status == "fail"
    assert any("missing" in reason for reason in check.reasons)


def test_token_gate_rejects_invalid_inputs():
    with pytest.raises(GateError):
        TokenInput(before={"t0": -1}, after={"t0": 0})


def test_token_gate_rejects_unknown_task_tokens():
    check = check_tokens(
        {"t0": 1000, "t1": 1000},
        {"t0": 1000, "t1": 1000},
        after_missing=("t1",),
    )
    assert check.status == "fail"
    assert "missing" in " ".join(check.reasons)


def test_token_gate_rejects_a_bare_string_for_missing_tasks():
    # A bare string would be split into characters, mislabelling the evidence.
    with pytest.raises(GateError, match="before_missing"):
        TokenInput(before={"t0": 1}, after={"t0": 1}, before_missing="t0")
    with pytest.raises(GateError, match="after_missing"):
        check_tokens({"t0": 1}, {"t0": 1}, after_missing="t0")


# ------------------------------------------------------- protection and T4


@pytest.mark.parametrize(
    ("protection", "needle"),
    [
        (ProtectionInput(hits=("decorator",)), "decorator"),
        (ProtectionInput(frozen_patch=True), "frozen"),
        (ProtectionInput(public_signature_changed=True), "signature"),
    ],
)
def test_protection_gate_rejects_each_hard_line(protection, needle):
    check = check_protection(
        protection.hits,
        frozen_patch=protection.frozen_patch,
        public_signature_changed=protection.public_signature_changed,
    )
    assert check.status == "fail"
    assert any(needle in reason for reason in check.reasons)


def test_t4_gate_rejects_while_the_component_is_protected():
    assert check_t4(protection_until=5, round=3).status == "fail"
    assert check_t4(protection_until=3, round=3).status == "pass"


def test_gate_rejects_a_protected_component():
    result = simplification_gate(good_request(protection_until=10, round=3))
    assert result.verdict == "reject"
    assert "t4" in failed_names(result)


# ------------------------------------------------------- pareto and worst


def test_pareto_rejects_flat_on_both_axes():
    assert check_pareto(delta_score=0.0, delta_k_ast=0).status == "fail"
    assert check_pareto(delta_score=-0.01, delta_k_ast=0).status == "fail"


def test_pareto_accepts_an_improvement_on_either_axis():
    assert check_pareto(delta_score=0.01, delta_k_ast=0).status == "pass"
    assert check_pareto(delta_score=0.0, delta_k_ast=-1).status == "pass"


def test_worst_task_gate_rejects_a_single_task_beyond_its_margin():
    check = check_worst_task({"t0": 0.0, "t1": -0.2}, margin=0.10)
    assert check.status == "fail"
    assert check.data["worst_task"] == "t1"


def test_worst_task_gate_passes_within_margin():
    assert check_worst_task({"t0": 0.0, "t1": -0.10}, margin=0.10).status == "pass"


def test_gate_rejects_a_bad_worst_task():
    request = good_request(
        scores=scores([1.0, 1.0, 1.0], [1.0, 1.0, 0.7]),
        task_margin=0.10,
    )
    result = simplification_gate(request)
    assert result.verdict == "reject"
    assert "worst_task" in failed_names(result)


# ------------------------------------------------------------------ report


def test_gate_reports_every_failure_not_just_the_first():
    request = good_request(
        scores=scores([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], margin=0.10),
        protection=ProtectionInput(hits=("assert",)),
        protection_until=9,
    )
    result = simplification_gate(request)
    assert result.verdict == "reject"
    assert {"non_inferiority", "protection", "t4"} <= failed_names(result)
    assert len(result.reasons) >= 3


def test_gate_result_serializes_to_plain_json():
    result = simplification_gate(good_request())
    payload = result.as_dict()
    assert payload["verdict"] == "accept"
    assert payload["checks"][0]["name"] == "validity"
    assert payload["checks"][1]["data"]["margin"] == 0.10


def test_gate_rejects_bad_round_inputs():
    with pytest.raises(GateError):
        good_request(round=-1)
    with pytest.raises(GateError):
        good_request(task_margin=0.0)
