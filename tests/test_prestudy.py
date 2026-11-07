from __future__ import annotations

import json
import math

import pytest

from evo.application.prestudy import (
    PrestudyInputError,
    build_prestudy,
    load_prestudy,
)
from evo.domain.prestudy import (
    K_AST_PCT_THRESHOLD,
    L3_YIELD_THRESHOLD,
    MIN_CENSUS_TASKS,
    PrestudyError,
    decide_prestudy,
)


def static_payload(k_ast: float = 0.2, subject: str = "demo") -> dict:
    return {
        "subject": subject,
        "l1": {
            "reduction_pct": {
                "k_lines_nb": k_ast,
                "k_ast": k_ast,
                "k_chars": k_ast,
                "k_cc": k_ast,
            }
        },
    }


def census_payload(k_ast: float = 0.4, subject: str = "demo", tasks: int = 10) -> dict:
    return {
        "subject": subject,
        "tasks": [f"t{index}" for index in range(tasks)],
        "zero_touch": 3,
        "l2": {
            "reduction_pct": {
                "k_lines_nb": k_ast,
                "k_ast": k_ast,
                "k_chars": k_ast,
                "k_cc": k_ast,
            }
        },
    }


def l3_payload(verdicts, subject: str = "demo") -> dict:
    return {
        "subject": subject,
        "levers": [{"id": f"lever-{index}", "verdict": verdict} for index, verdict in enumerate(verdicts)],
    }


# ------------------------------------------------------------------ domain


def test_proceed_when_l3_yield_reaches_the_threshold():
    decision = decide_prestudy(0.2, 0.4, 0.35)
    assert decision.decision == "proceed"
    assert decision.l3_yield == 0.35


def test_switch_subject_when_both_conditions_hold():
    decision = decide_prestudy(0.2, 0.4, 0.10)
    assert decision.decision == "switch_subject"
    assert len(decision.reasons) == 2


def test_proceed_when_static_and_census_reach_one_percent():
    decision = decide_prestudy(0.8, 0.5, 0.05)
    assert decision.decision == "proceed"
    assert decision.combined_pct == pytest.approx(1.3)


def test_proceed_without_l3_when_static_and_census_reach_one_percent():
    decision = decide_prestudy(0.8, 0.5)
    assert decision.decision == "proceed"
    assert decision.l3_yield is None


def test_pending_when_l3_is_missing_and_reach_is_below_one_percent():
    decision = decide_prestudy(0.2, 0.4)
    assert decision.decision == "pending_l3"
    assert decision.l3_yield is None


def test_pending_l2_when_the_census_is_provisional():
    decision = decide_prestudy(0.2, 0.4, None, census_tasks=2)
    assert decision.decision == "pending_l2"
    assert any("2" in reason for reason in decision.reasons)


def test_census_task_count_does_not_block_at_the_minimum():
    decision = decide_prestudy(0.2, 0.4, None, census_tasks=MIN_CENSUS_TASKS)
    assert decision.decision == "pending_l3"


def test_l3_yield_boundary_is_inclusive():
    assert decide_prestudy(0.2, 0.4, L3_YIELD_THRESHOLD).decision == "proceed"


def test_combined_boundary_is_inclusive():
    decision = decide_prestudy(0.5, 0.5, 0.10)
    assert decision.combined_pct == pytest.approx(K_AST_PCT_THRESHOLD)
    assert decision.decision == "proceed"


@pytest.mark.parametrize(
    "call",
    [
        lambda: decide_prestudy(-0.1, 0.0),
        lambda: decide_prestudy(0.0, math.inf),
        lambda: decide_prestudy(0.0, 0.0, 1.5),
        lambda: decide_prestudy(0.0, 0.0, -0.1),
        lambda: decide_prestudy(0.0, 0.0, math.nan),
    ],
)
def test_domain_rejects_invalid_inputs(call):
    with pytest.raises(PrestudyError):
        call()


def test_decision_serializes():
    payload = decide_prestudy(0.2, 0.4, 0.10).as_dict()
    assert payload["decision"] == "switch_subject"
    assert payload["combined_pct"] == pytest.approx(0.6)
    assert payload["l3_yield"] == 0.10
    assert isinstance(payload["reasons"], list)


# ------------------------------------------------------------- application


def test_build_prestudy_assembles_the_three_stages():
    report = build_prestudy(
        static_payload(), census_payload(), l3_payload(["zero_contribution", "contributes", "inconclusive"])
    )
    assert report["subject"] == "demo"
    assert report["k_metric"] == "k_ast"
    assert report["static"]["reduction_pct"]["k_ast"] == pytest.approx(0.2)
    assert report["census"]["zero_touch"] == 3
    assert report["l3"] == {"yield": 1 / 3, "zero_contribution": 1, "evaluated": 3}
    assert report["decision"] == "proceed"


def test_build_prestudy_switches_subject_when_both_signals_are_low():
    report = build_prestudy(
        static_payload(0.2), census_payload(0.3), l3_payload(["contributes", "inconclusive"])
    )
    assert report["decision"] == "switch_subject"


def test_build_prestudy_refuses_a_provisional_census():
    report = build_prestudy(
        static_payload(2.0), census_payload(41.0, tasks=2), l3_payload(["zero_contribution"])
    )
    assert report["decision"] == "pending_l2"
    assert report["census"]["tasks"] == ["t0", "t1"]


def test_build_prestudy_without_l3_is_pending_or_early_proceed():
    report = build_prestudy(static_payload(0.2), census_payload(0.4))
    assert report["l3"] is None
    assert report["decision"] == "pending_l3"


def test_build_prestudy_rejects_subject_mismatch():
    with pytest.raises(PrestudyInputError):
        build_prestudy(static_payload(), census_payload(subject="other"), None)
    with pytest.raises(PrestudyInputError):
        build_prestudy(static_payload(), census_payload(), l3_payload([], subject="other"))


def test_build_prestudy_rejects_missing_reduction_fields():
    static = static_payload()
    del static["l1"]["reduction_pct"]
    with pytest.raises(PrestudyInputError):
        build_prestudy(static, census_payload(), None)


def test_build_prestudy_rejects_an_empty_l3_file():
    with pytest.raises(PrestudyInputError):
        build_prestudy(static_payload(), census_payload(), l3_payload(["invalid"]))


def test_build_prestudy_counts_only_valid_l3_measurements():
    report = build_prestudy(
        static_payload(),
        census_payload(),
        l3_payload(["zero_contribution", "contributes", "invalid"]),
    )
    assert report["l3"] == {"yield": 0.5, "zero_contribution": 1, "evaluated": 2}


def test_load_prestudy_reads_files_from_disk(tmp_path):
    static = tmp_path / "prestudy.static.json"
    census = tmp_path / "census.l2.json"
    l3 = tmp_path / "l3.json"
    static.write_text(json.dumps(static_payload()), encoding="utf-8")
    census.write_text(json.dumps(census_payload()), encoding="utf-8")
    l3.write_text(json.dumps(l3_payload(["contributes"])), encoding="utf-8")
    report = load_prestudy(static, census, l3)
    assert report["decision"] == "switch_subject"
    assert report["l3"]["evaluated"] == 1


def test_load_prestudy_reports_a_missing_file(tmp_path):
    with pytest.raises(PrestudyInputError) as excinfo:
        load_prestudy(tmp_path / "absent.json", tmp_path / "also.json", None)
    assert "absent.json" in str(excinfo.value)


def test_load_prestudy_reports_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(PrestudyInputError) as excinfo:
        load_prestudy(bad, bad, None)
    assert "JSON" in str(excinfo.value)
