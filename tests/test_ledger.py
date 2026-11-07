from __future__ import annotations

import json

import pytest

from evo.domain.ledger import Ledger, LedgerEntry, LedgerError


FIXED_NOW = "2026-11-04T00:00:00+00:00"


def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl", run_id="r1", clock=lambda: FIXED_NOW)


def test_append_and_read_roundtrip(tmp_path):
    log = ledger(tmp_path)
    entry = log.append("c001", "delete", "all gates pass", round=1, evidence={"dS": 0.0})
    assert entry.timestamp == FIXED_NOW
    assert log.entries() == [entry]


def test_file_is_append_only_jsonl(tmp_path):
    log = ledger(tmp_path)
    log.append("c001", "keep", "score drop", round=1)
    log.append("c002", "delete", "no drop", round=1)
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["candidate_id"] == "c001"
    assert json.loads(lines[1])["candidate_id"] == "c002"


def test_all_four_verdicts_are_allowed(tmp_path):
    log = ledger(tmp_path)
    for index, verdict in enumerate(("keep", "delete", "undecidable", "undeletable")):
        log.append(f"c{index}", verdict, "reason", round=1)
    assert log.summary() == {"delete": 1, "keep": 1, "undecidable": 1, "undeletable": 1}


def test_untested_is_not_a_verdict(tmp_path):
    with pytest.raises(LedgerError, match="verdict"):
        ledger(tmp_path).append("c001", "untested", "not measured")


def test_undeletable_requires_a_reason(tmp_path):
    with pytest.raises(LedgerError, match="reason"):
        ledger(tmp_path).append("c001", "undeletable", "")


def test_all_verdicts_require_a_reason(tmp_path):
    with pytest.raises(LedgerError, match="reason"):
        ledger(tmp_path).append("c001", "keep", "   ")


def test_latest_returns_last_entry_per_candidate(tmp_path):
    log = ledger(tmp_path)
    log.append("c001", "undecidable", "n too small", round=1)
    log.append("c001", "delete", "n increased", round=2)
    assert log.latest()["c001"].verdict == "delete"
    assert log.latest()["c001"].round == 2


def test_round_must_be_non_negative(tmp_path):
    with pytest.raises(LedgerError, match="round"):
        ledger(tmp_path).append("c001", "keep", "reason", round=-1)


def test_non_serializable_evidence_is_rejected(tmp_path):
    with pytest.raises(LedgerError):
        ledger(tmp_path).append("c001", "keep", "reason", evidence={"bad": {1, 2}})


def test_malformed_line_raises(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"candidate_id": "c001"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(LedgerError):
        Ledger(path, run_id="r1").entries()


def test_entry_requires_matching_run_id(tmp_path):
    path = tmp_path / "ledger.jsonl"
    log = Ledger(path, run_id="r1")
    log.append("c001", "keep", "reason")
    with pytest.raises(LedgerError, match="run_id"):
        Ledger(path, run_id="r2").entries()
