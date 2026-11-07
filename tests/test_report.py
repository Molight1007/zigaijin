from __future__ import annotations

import json

import pytest

from evo.application.report import ReportError, build_tables, render_table


RUN = {
    "dataset": "terminal-bench/terminal-bench-2-1",
    "subject": "terminus2",
    "pool_hash": "abc123",
}

PRESTUDY = {
    "subject": "terminus2",
    "components": [
        {
            "id": "terminus2:agent.py",
            "paths": ["agent.py"],
            "k": {
                "k_lines_nb": 100,
                "k_ast": 200,
                "k_chars": 1000,
                "k_bytes": 1000,
                "k_gzip": 100,
                "k_cc": 5,
            },
        }
    ],
}

BASELINE = {
    "aggregate": {
        "tasks_with_valid_trials": 3,
        "valid_trials": 9,
        "planned_trials": 9,
        "mean_reward": 0.5,
        "tokens_in": 300,
        "tokens_out": 30,
        "cost_usd": 0.03,
    },
    "validity": {"status": "pass"},
}

K_BLOCK = {
    "before": {
        "k_lines_nb": 100,
        "k_ast": 200,
        "k_chars": 1000,
        "k_bytes": 1000,
        "k_gzip": 100,
        "k_cc": 5,
    },
    "after": {
        "k_lines_nb": 90,
        "k_ast": 180,
        "k_chars": 900,
        "k_bytes": 900,
        "k_gzip": 90,
        "k_cc": 4,
    },
    "reduction_pct": {"k_lines_nb": 10.0, "k_ast": 10.0},
}


def comparison(*, k=True):
    payload = {
        "aggregate": {
            "n_pairs": 3,
            "reference_mean": 0.5,
            "candidate_mean": 0.5,
            "mean_delta": 0.0,
            "ci": {"low": -0.1, "high": 0.1},
            "margin": 0.1,
            "non_inferior": True,
            "worst_task": {"task": "t1", "delta": 0.0},
            "tokens": {
                "reference": {"total": 330},
                "candidate": {"total": 320},
            },
            "cost": {"reference": 0.03, "candidate": 0.028},
        },
        "validity": {
            "reference": {"status": "pass"},
            "candidate": {"status": "pass"},
        },
        "arms": {
            "reference": {"aggregate": {"valid_trials": 9}},
            "candidate": {"aggregate": {"valid_trials": 9}},
        },
    }
    if k:
        payload["k"] = K_BLOCK
    return payload


LEDGER = [
    {
        "round": 1,
        "candidate_id": "l1-0001",
        "verdict": "delete",
        "reason": "all gates pass",
        "evidence": {"layer": "L1", "verify": "pass"},
    },
    {
        "round": 1,
        "candidate_id": "l1-0002",
        "verdict": "keep",
        "reason": "gate rejected: ci_low < -0.1",
        "evidence": {"layer": "L1", "verify": "pass"},
    },
]

DIAGNOSIS = {"before": {"top1": 0.28, "top3": 0.44}, "after": {"top1": 0.51, "top3": 0.66}}


def tables(**overrides):
    kwargs = {
        "run": RUN,
        "prestudy": PRESTUDY,
        "baseline": BASELINE,
        "comparisons": {"SIM": comparison()},
        "ledger": LEDGER,
        "diagnosis": DIAGNOSIS,
    }
    kwargs.update(overrides)
    return build_tables(**kwargs)


def test_six_tables_are_built_from_artifacts():
    built = tables()
    assert set(built) == {
        "t1_subject",
        "t2_arms",
        "t3_deletions",
        "t4_baselines",
        "t5_diagnosis",
        "t6_cost",
    }
    t1 = built["t1_subject"]
    assert t1["status"] == "ok"
    assert t1["rows"] == [
        ["terminus2:agent.py", "agent.py", 100, 200, 1000, 1000, 100, 5]
    ]
    t2 = built["t2_arms"]
    assert t2["status"] == "ok"
    assert [row[0] for row in t2["rows"]] == ["H0", "SIM"]
    sim = t2["rows"][1]
    assert sim[1] == "comparison"
    assert sim[3] == 0.5 and sim[4] == 0.5
    t3 = built["t3_deletions"]
    assert t3["status"] == "ok"
    assert [row[2] for row in t3["rows"]] == ["delete", "keep"]
    assert t3["rows"][0][5] == "pass"
    assert built["t4_baselines"]["status"] == "not_run"
    t5 = built["t5_diagnosis"]
    assert t5["status"] == "ok"
    assert t5["rows"] == [
        ["after", "top1", 0.51],
        ["after", "top3", 0.66],
        ["before", "top1", 0.28],
        ["before", "top3", 0.44],
    ]
    t6 = built["t6_cost"]
    assert t6["status"] == "ok"
    row = t6["rows"][0]
    assert row[0] == "SIM"
    assert row[1] == 10.0
    assert row[3] == 18
    assert row[4] == 650
    assert row[6] == pytest.approx(1.8)


def test_t4_only_reports_control_arms():
    built = tables(
        comparisons={
            "SIM": comparison(),
            "PLACEBO": comparison(),
            "SHARP": comparison(),
        }
    )
    t4 = built["t4_baselines"]
    assert t4["status"] == "ok"
    assert [row[0] for row in t4["rows"]] == ["PLACEBO", "SHARP"]


def test_missing_artifacts_mark_tables_not_run():
    built = build_tables(run=RUN)
    for name, table in built.items():
        assert table["status"] == "not_run", name
        assert table["rows"] == []


def test_comparison_without_k_skips_the_cost_table():
    built = tables(comparisons={"SIM": comparison(k=False)})
    assert built["t2_arms"]["status"] == "ok"
    assert built["t6_cost"]["status"] == "not_run"


def test_malformed_comparison_is_refused():
    payload = comparison()
    del payload["aggregate"]["ci"]
    with pytest.raises(ReportError, match="ci"):
        tables(comparisons={"SIM": payload})


def test_ledger_entries_without_evidence_are_allowed():
    built = tables(
        ledger=[
            {
                "round": 0,
                "candidate_id": "l1-0009",
                "verdict": "undecidable",
                "reason": "required checks unknown: replay",
            }
        ]
    )
    assert built["t3_deletions"]["rows"] == [
        [0, "l1-0009", "undecidable", "required checks unknown: replay", "", ""]
    ]


def test_render_table_writes_a_stable_csv():
    text = render_table(tables()["t2_arms"])
    lines = text.splitlines()
    assert lines[0] == (
        "arm,source,n_pairs,reference_mean,candidate_mean,mean_delta,"
        "ci_low,ci_high,margin,non_inferior,validity,worst_task,worst_delta,"
        "tokens_reference,tokens_candidate,cost_reference,cost_candidate"
    )
    assert lines[1].startswith("H0,baseline,3,0.5")
    assert lines[2].startswith("SIM,comparison,3,0.5,0.5,0.0,-0.1,0.1,0.1,True,pass")
    assert text.endswith("\n")


def test_report_payload_keeps_run_identity():
    built = tables()
    assert built["t1_subject"]["source"] == ["prestudy.json"]
    assert built["t2_arms"]["source"] == ["baseline.json", "comparison:SIM"]
    derived = json.loads(json.dumps(built))
    assert derived["t1_subject"]["rows"][0][0] == "terminus2:agent.py"
