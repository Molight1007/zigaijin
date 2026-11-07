from __future__ import annotations

from dataclasses import replace

import pytest

from evo.application.simplify import (
    CandidateDecision,
    SimplifyError,
    decide_candidate,
    record_decision,
)
from evo.contracts.types import KVector
from evo.domain.gate import (
    GateRequest,
    KGateInput,
    ProtectionInput,
    ScoreInput,
    TokenInput,
    ValidityInput,
)
from evo.domain.ledger import Ledger
from evo.domain.verify import CheckOutcome, aggregate, required_checks


K_BEFORE = KVector(
    k_lines_nb=200,
    k_ast=400,
    k_chars=4000,
    k_bytes=8000,
    k_gzip=2000,
    k_cc=20,
)
K_AFTER = KVector(
    k_lines_nb=180,
    k_ast=360,
    k_chars=3900,
    k_bytes=7800,
    k_gzip=1950,
    k_cc=18,
)


def request(**overrides) -> GateRequest:
    base = GateRequest(
        validity=ValidityInput(missing_frac=0.0, infra_frac=0.0, token_coverage=1.0),
        scores=ScoreInput(
            tasks=("t1", "t2", "t3", "t4"),
            reference=(1.0, 0.0, 1.0, 0.0),
            candidate=(1.0, 0.0, 1.0, 0.0),
            margin=0.10,
        ),
        holdout=ScoreInput(
            tasks=("h1", "h2"),
            reference=(1.0, 0.0),
            candidate=(1.0, 0.0),
            margin=0.10,
        ),
        k=KGateInput(before=K_BEFORE, after=K_AFTER),
        tokens=TokenInput(
            before={"t1": 100, "t2": 100, "t3": 100, "t4": 100},
            after={"t1": 100, "t2": 100, "t3": 100, "t4": 100},
        ),
        protection=ProtectionInput(),
        round=1,
        protection_until=0,
        task_margin=0.10,
        n_boot=200,
    )
    return replace(base, **overrides)


def report(layer: str, *outcomes: CheckOutcome):
    return aggregate(outcomes, required_checks(layer))


def passing_report(layer: str = "L2"):
    return report(layer, *(CheckOutcome(name, "pass") for name in required_checks(layer)))


def test_accepts_a_clean_candidate():
    decision = decide_candidate("c001", "L2", passing_report(), request())
    assert decision.verdict == "delete"
    assert decision.reason == "all gates pass"
    assert decision.verify_status == "pass"
    assert decision.gate is not None and decision.gate.verdict == "accept"
    payload = decision.as_dict()
    assert payload["candidate_id"] == "c001"
    assert payload["layer"] == "L2"
    assert payload["verify"] == "pass"
    assert payload["gate"]["verdict"] == "accept"


def test_failed_verification_keeps_the_code():
    broken = report(
        "L2",
        CheckOutcome("compile", "pass"),
        CheckOutcome("import", "pass"),
        CheckOutcome("smoke", "fail", "task t1 crashed"),
    )
    decision = decide_candidate("c002", "L2", broken, request())
    assert decision.verdict == "keep"
    assert "smoke" in decision.reason
    assert decision.gate is None


def test_unknown_required_check_is_undecidable():
    unknown = report(
        "L2",
        CheckOutcome("compile", "pass"),
        CheckOutcome("import", "pass"),
        CheckOutcome("smoke", "unknown"),
    )
    decision = decide_candidate("c003", "L2", unknown, request())
    assert decision.verdict == "undecidable"
    assert "smoke" in decision.reason


def test_missing_verification_is_undecidable():
    decision = decide_candidate("c004", "L1", None, request())
    assert decision.verdict == "undecidable"
    assert decision.verify_status == "unknown"
    assert "verification" in decision.reason


def test_missing_evaluation_is_undecidable():
    decision = decide_candidate("c005", "L1", passing_report("L1"), None)
    assert decision.verdict == "undecidable"
    assert "evaluation" in decision.reason


def test_undeletable_short_circuits_and_needs_a_reason():
    decision = decide_candidate(
        "c006",
        "L3",
        passing_report("L3"),
        request(),
        undeletable=True,
        undeletable_reason="call site remains in agent.py",
    )
    assert decision.verdict == "undeletable"
    assert "call site" in decision.reason
    with pytest.raises(SimplifyError, match="reason"):
        decide_candidate("c006", "L3", passing_report("L3"), request(), undeletable=True)


def test_gate_reject_maps_to_keep():
    weak = replace(
        request().k,
        after=KVector(
            k_lines_nb=196,
            k_ast=398,
            k_chars=3990,
            k_bytes=7990,
            k_gzip=1995,
            k_cc=20,
        ),
    )
    decision = decide_candidate("c007", "L2", passing_report(), request(k=weak))
    assert decision.verdict == "keep"
    assert decision.gate is not None and decision.gate.verdict == "reject"
    assert decision.reason.startswith("gate rejected:")


def test_invalid_measurement_maps_to_undecidable():
    invalid = replace(
        request().validity,
        missing_frac=0.5,
    )
    decision = decide_candidate("c008", "L2", passing_report(), request(validity=invalid))
    assert decision.verdict == "undecidable"
    assert decision.gate is not None and decision.gate.verdict == "invalid"
    assert decision.reason.startswith("invalid measurement:")


def test_l4_with_unknown_differential_is_undecidable():
    outcomes = [
        CheckOutcome(name, "unknown" if name == "differential" else "pass")
        for name in required_checks("L4")
    ]
    decision = decide_candidate("c009", "L4", report("L4", *outcomes), request())
    assert decision.verdict == "undecidable"
    assert "differential" in decision.reason


def test_unknown_layer_is_rejected():
    with pytest.raises(SimplifyError, match="layer"):
        decide_candidate("c010", "L9", passing_report(), request())


def test_empty_candidate_id_is_rejected():
    with pytest.raises(SimplifyError, match="candidate_id"):
        decide_candidate("", "L1", passing_report("L1"), request())


def test_record_decision_appends_evidence(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", run_id="run-1")
    decision = decide_candidate("c011", "L2", passing_report(), request())
    entry = record_decision(ledger, decision, round=3)
    assert entry.verdict == "delete"
    assert entry.round == 3
    stored = ledger.entries()
    assert len(stored) == 1
    assert stored[0].candidate_id == "c011"
    assert stored[0].evidence["layer"] == "L2"
    assert stored[0].evidence["verify"] == "pass"
    assert stored[0].evidence["gate"]["verdict"] == "accept"


def test_record_decision_keeps_caller_evidence(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", run_id="run-1")
    decision = CandidateDecision(
        candidate_id="c012",
        layer="L1",
        verdict="keep",
        reason="gate rejected: k",
        verify_status="pass",
    )
    entry = record_decision(ledger, decision, evidence={"diff": "candidates/c012/diff.patch"})
    assert entry.evidence["diff"] == "candidates/c012/diff.patch"
    assert entry.evidence["layer"] == "L1"
    assert entry.evidence["gate"] is None


def test_record_decision_keeps_direction_diagnostic(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", run_id="run-1")
    decision = CandidateDecision(
        candidate_id="c013",
        layer="L2",
        verdict="keep",
        reason="gate rejected: holdout",
        verify_status="pass",
        direction={"computed": True, "consistent": False},
    )
    entry = record_decision(ledger, decision)
    assert entry.evidence["direction"]["consistent"] is False
