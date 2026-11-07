from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from evo.domain.gate import GateRequest, GateResult, simplification_gate
from evo.domain.ledger import Ledger, LedgerEntry
from evo.domain.verify import VerifyReport


LAYERS = ("L1", "L2", "L3", "L4")
VERDICTS = ("keep", "delete", "undecidable", "undeletable")


class SimplifyError(ValueError):
    """A simplification decision input is malformed."""


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    layer: str
    verdict: str
    reason: str
    verify_status: str
    gate: GateResult | None = None
    direction: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "layer": self.layer,
            "verdict": self.verdict,
            "reason": self.reason,
            "verify": self.verify_status,
            "gate": self.gate.as_dict() if self.gate is not None else None,
            "direction": dict(self.direction) if self.direction is not None else None,
        }


def _unknown_checks(report: VerifyReport) -> list[str]:
    by_name = {outcome.name: outcome for outcome in report.outcomes}
    return [
        name
        for name in report.required
        if name not in by_name or by_name[name].status == "unknown"
    ]


def _failed_checks(report: VerifyReport) -> list[str]:
    return [outcome.name for outcome in report.outcomes if outcome.status == "fail"]


def decide_candidate(
    candidate_id: str,
    layer: str,
    verify_report: VerifyReport | None,
    request: GateRequest | None,
    *,
    undeletable: bool = False,
    undeletable_reason: str = "",
    direction: Mapping[str, object] | None = None,
) -> CandidateDecision:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SimplifyError(f"candidate_id must be a non-empty string: {candidate_id!r}")
    if layer not in LAYERS:
        raise SimplifyError(
            f"unknown layer {layer!r}; expected one of {', '.join(LAYERS)}"
        )
    if verify_report is not None and not isinstance(verify_report, VerifyReport):
        raise SimplifyError("verify_report must be a VerifyReport or None")
    if request is not None and not isinstance(request, GateRequest):
        raise SimplifyError("request must be a GateRequest or None")
    if not isinstance(undeletable, bool):
        raise SimplifyError(f"undeletable must be a bool: {undeletable!r}")
    if undeletable and not undeletable_reason.strip():
        raise SimplifyError("undeletable candidates require a reason")
    if direction is not None and not isinstance(direction, Mapping):
        raise SimplifyError("direction must be a mapping or None")
    diagnostic = dict(direction) if direction is not None else None

    status = verify_report.verdict if verify_report is not None else "unknown"

    if undeletable:
        return CandidateDecision(
            candidate_id, layer, "undeletable", undeletable_reason.strip(), status,
            direction=diagnostic,
        )
    if verify_report is None:
        return CandidateDecision(
            candidate_id, layer, "undecidable", "no verification report", status,
            direction=diagnostic,
        )
    if status == "fail":
        failed = ", ".join(_failed_checks(verify_report))
        return CandidateDecision(
            candidate_id, layer, "keep", f"verification failed: {failed}", status,
            direction=diagnostic,
        )
    if status == "unknown":
        unknown = ", ".join(_unknown_checks(verify_report))
        return CandidateDecision(
            candidate_id, layer, "undecidable", f"required checks unknown: {unknown}", status,
            direction=diagnostic,
        )
    if request is None:
        return CandidateDecision(
            candidate_id, layer, "undecidable", "no evaluation evidence", status,
            direction=diagnostic,
        )

    gate = simplification_gate(request)
    if gate.verdict == "invalid":
        reason = "invalid measurement: " + "; ".join(gate.reasons)
        return CandidateDecision(
            candidate_id, layer, "undecidable", reason, status, gate, diagnostic
        )
    if gate.verdict == "reject":
        reason = "gate rejected: " + "; ".join(gate.reasons)
        return CandidateDecision(
            candidate_id, layer, "keep", reason, status, gate, diagnostic
        )
    return CandidateDecision(
        candidate_id, layer, "delete", "all gates pass", status, gate, diagnostic
    )


def record_decision(
    ledger: Ledger,
    decision: CandidateDecision,
    *,
    round: int = 0,
    evidence: Mapping[str, object] | None = None,
) -> LedgerEntry:
    if not isinstance(ledger, Ledger):
        raise SimplifyError("ledger must be a Ledger")
    if not isinstance(decision, CandidateDecision):
        raise SimplifyError("decision must be a CandidateDecision")
    payload: dict[str, object] = dict(evidence or {})
    payload["layer"] = decision.layer
    payload["verify"] = decision.verify_status
    payload["gate"] = decision.gate.as_dict() if decision.gate is not None else None
    payload["direction"] = decision.as_dict()["direction"]
    return ledger.append(
        decision.candidate_id,
        decision.verdict,
        decision.reason,
        round=round,
        evidence=payload,
    )
