from __future__ import annotations

import json
import os
import subprocess

import pytest

import evo.cli as cli_module
from evo.cli import main
from evo.domain.ledger import Ledger


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def output_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_compare_cli_accepts_family_size(tmp_path, capsys):
    def write_outcomes(name, prefix):
        lines = []
        for task in ("t1", "t2"):
            lines.append(
                json.dumps(
                    {
                        "task_id": task,
                        "trial_id": f"{prefix}-{task}",
                        "status": "completed",
                        "reward": 1.0,
                        "tokens_in": 10,
                        "tokens_out": 2,
                    }
                )
            )
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    reference = write_outcomes("reference.jsonl", "ref")
    candidate = write_outcomes("candidate.jsonl", "cand")
    out = tmp_path / "out"
    assert (
        main(
            [
                "compare",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--tasks",
                "t1,t2",
                "--family-size",
                "2",
                "--n-boot",
                "200",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["family_size"] == 2
    assert payload["aggregate"]["family"]["size"] == 2


def test_compare_cli_rejects_a_non_positive_family_size(tmp_path, capsys):
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "trial_id": "ref-1",
                "status": "completed",
                "reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "compare",
                "--reference",
                str(reference),
                "--candidate",
                str(reference),
                "--tasks",
                "t1",
                "--family-size",
                "0",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "family-size" in capsys.readouterr().err


def test_diagnose_merge_writes_a_before_after_document(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    out = tmp_path / "merged.json"
    before.write_text(json.dumps({"before": {"top1": 0.2, "top3": 0.5}}), encoding="utf-8")
    after.write_text(json.dumps({"after": {"top1": 0.4, "top3": 0.7}}), encoding="utf-8")

    assert (
        main(
            [
                "diagnose-merge",
                "--before",
                str(before),
                "--after",
                str(after),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "before": {"top1": 0.2, "top3": 0.5},
        "after": {"top1": 0.4, "top3": 0.7},
    }


def test_workforce_command_is_registered():
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "workforce",
            "--run",
            "run.json",
            "--reference-outcomes",
            "outcomes.jsonl",
            "--work",
            "work",
            "--out",
            "out",
            "--jobs-dir",
            "jobs",
        ]
    )
    assert args.command == "workforce"
    assert args.mode == "delete"
    assert args.layer == "L2"
    assert args.batch_mode == "static"


def test_workforce_parser_defaults_to_semantic_review():
    parser = cli_module.build_parser()
    args = parser.parse_args([
        "workforce", "--run", "run.json", "--reference-outcomes", "o.jsonl",
        "--work", "w", "--out", "o", "--jobs-dir", "j",
    ])
    assert args.no_semantic_review is False


def test_workforce_parser_accepts_no_semantic_review():
    parser = cli_module.build_parser()
    args = parser.parse_args([
        "workforce", "--run", "run.json", "--reference-outcomes", "o.jsonl",
        "--work", "w", "--out", "o", "--jobs-dir", "j", "--no-semantic-review",
    ])
    assert args.no_semantic_review is True


def test_workforce_parser_accepts_cost_budgets():
    parser = cli_module.build_parser()
    base = [
        "workforce", "--run", "run.json", "--reference-outcomes", "o.jsonl",
        "--work", "w", "--out", "o", "--jobs-dir", "j",
    ]
    defaults = parser.parse_args(base)
    assert defaults.max_total_trials is None
    assert defaults.max_cost_usd is None
    args = parser.parse_args(
        base + ["--max-total-trials", "40", "--max-cost-usd", "1.5"]
    )
    assert args.max_total_trials == 40
    assert args.max_cost_usd == 1.5


def test_workforce_coreset_error_exits_two_without_traceback(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    subject = tmp_path / "terminus2"
    subject.mkdir()
    (subject / "a.py").write_text("x = 1\n", encoding="utf-8")
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "dataset": "terminal-bench/terminal-bench-2-1",
                "prereg": {"k": 2, "margin": 0.10, "holdout_margin": 0.10},
                "pools": {"evolution": ["t1", "t2"], "holdout": ["h1"]},
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task,
                    "trial_id": f"r-{task}",
                    "status": "completed",
                    "reward": 1.0,
                }
            )
            for task in ("t1", "t2", "h1")
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "workforce",
            "--run",
            str(run),
            "--reference-outcomes",
            str(reference),
            "--subject-root",
            str(subject),
            "--work",
            str(tmp_path / "work"),
            "--out",
            str(tmp_path / "out"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--coreset-k",
            "1",
            "--loop",
            "--total-rounds",
            "1",
            # the command creates out/logs before the loop, so --force is required
            "--force",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "not enough complete tasks" in captured.err
    assert "Traceback" not in captured.err


def test_simplify_effect_writes_an_effect_report(tmp_path, capsys):
    def outcomes(name, prefix):
        lines = []
        for index in range(1, 7):
            task = f"t{index}"
            lines.append(
                json.dumps(
                    {
                        "task_id": task,
                        "trial_id": f"{prefix}-{task}",
                        "status": "completed",
                        "reward": 1.0,
                        "tokens_in": 100,
                        "tokens_out": 10,
                    }
                )
            )
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    reference = outcomes("reference.jsonl", "ref")
    candidate = outcomes("candidate.jsonl", "cand")
    applied = tmp_path / "applied.json"
    applied.write_text(
        json.dumps(
            {
                "before": {
                    "k_lines_nb": 1000,
                    "k_ast": 5000,
                    "k_chars": 10000,
                    "k_bytes": 12000,
                    "k_gzip": 3000,
                    "k_cc": 100,
                },
                "after": {
                    "k_lines_nb": 900,
                    "k_ast": 4500,
                    "k_chars": 9000,
                    "k_bytes": 10800,
                    "k_gzip": 2700,
                    "k_cc": 90,
                },
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "batch": [
                    {
                        "id": "c1",
                        "layer": "L1",
                        "delta": {
                            "k_lines_nb": 60,
                            "k_ast": 300,
                            "k_chars": 600,
                            "k_bytes": 700,
                            "k_gzip": 200,
                            "k_cc": 6,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "simplification.json"

    assert (
        main(
            [
                "simplify-effect",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--tasks",
                "t1,t2,t3,t4,t5,t6",
                "--holdout",
                "t5,t6",
                "--applied",
                str(applied),
                "--plan",
                str(plan),
                "--n-boot",
                "200",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "simplification_effective"
    assert payload["layers"]["L1"]["count"] == 1


def test_ai_cleanup_reports_wrappers(tmp_path, capsys):
    write(tmp_path, "pkg/a.py", "def helper(x):\n    return x + 1\n\ndef _wrapped(x):\n    return helper(x)\n")

    assert main(["ai-cleanup", "--root", str(tmp_path), "--json"]) == 0
    payload = output_json(capsys)

    assert any(
        row["kind"] == "trivial_wrapper" and row["symbol"] == "_wrapped"
        for row in payload["findings"]
    )


def test_version_compare_cli_writes_json_and_csv(tmp_path, capsys):
    def write_outcomes(name, prefix, rewards):
        lines = []
        for task, reward in rewards.items():
            lines.append(
                json.dumps(
                    {
                        "task_id": task,
                        "trial_id": f"{prefix}-{task}",
                        "status": "completed",
                        "reward": reward,
                        "tokens_in": 10,
                        "tokens_out": 2,
                    }
                )
            )
        path = tmp_path / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    h0 = write_outcomes("h0.jsonl", "h0", {"t1": 0.0, "t2": 1.0})
    previous = write_outcomes("previous.jsonl", "prev", {"t1": 0.0, "t2": 1.0})
    candidate = write_outcomes("candidate.jsonl", "cand", {"t1": 1.0, "t2": 1.0})
    out = tmp_path / "comparison.json"
    csv_path = tmp_path / "per_task.csv"

    assert (
        main(
            [
                "version-compare",
                "--h0",
                str(h0),
                "--previous",
                str(previous),
                "--candidate",
                str(candidate),
                "--tasks",
                "t1,t2",
                "--holdout",
                "t2",
                "--k",
                "1",
                "--n-boot",
                "200",
                "--csv",
                str(csv_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "candidate_improved"
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == (
        "task,H0_mean_reward,previous_mean_reward,candidate_mean_reward"
    )


def test_incident_open_and_list(tmp_path, capsys):
    path = tmp_path / "incidents.jsonl"
    assert (
        main(
            [
                "incident",
                "open",
                "--path",
                str(path),
                "--id",
                "inc-1",
                "--version",
                "v1",
                "--phase",
                "evaluation",
                "--location",
                "vendor/terminus2/terminus_2.py:837",
                "--symptom",
                "timeout after fixed wait",
                "--consequence",
                "reward 0; tokens +40%",
                "--component",
                "terminus2:summarize",
                "--task",
                "terminal-bench/example",
                "--trial",
                "trial-1",
                "--evidence",
                "runs/versions/v1/outcomes.jsonl",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["incident", "list", "--path", str(path), "--json"]) == 0
    payload = output_json(capsys)
    assert payload[0]["id"] == "inc-1"
    assert payload[0]["location"].endswith("terminus_2.py:837")
    assert payload[0]["consequence"].startswith("reward 0")


def test_version(capsys):
    assert main(["version"]) == 0
    assert "evo" in capsys.readouterr().out


def test_measure_json(tmp_path, capsys):
    write(tmp_path, "a.py", "x = 1\n")
    write(tmp_path, "b.py", "y = 2\n\n")
    assert main(["measure", "--root", str(tmp_path), "--json"]) == 0
    payload = output_json(capsys)
    assert payload["files"] == 2
    assert payload["total"]["k_lines_nb"] == 2
    assert set(payload["per_file"]) == {"a.py", "b.py"}


def test_scan_json(tmp_path, capsys):
    write(tmp_path, "a.py", "import os\n\ndef _dead():\n    return 1\n")
    assert main(["scan", "--root", str(tmp_path), "--subject", "demo", "--json"]) == 0
    payload = output_json(capsys)
    symbols = {candidate["symbol"] for candidate in payload["candidates"]}
    assert symbols == {"os", "_dead"}
    assert payload["counts"]["eligible"] == 2


def test_scan_custom_boundaries(tmp_path, capsys):
    write(tmp_path, "a/x.py", "import os\n")
    write(tmp_path, "b/y.py", "import sys\n")
    assert (
        main(
            [
                "scan",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--boundary",
                "a=a",
                "--boundary",
                "b=b",
                "--json",
            ]
        )
        == 0
    )
    payload = output_json(capsys)
    assert {candidate["component"] for candidate in payload["candidates"]} == {
        "demo:a",
        "demo:b",
    }


def test_prestudy_static_writes_json(tmp_path, capsys):
    write(tmp_path, "pkg/a.py", "import os\n\ndef _dead():\n    return 1\n\ndef run():\n    return 2\n")
    out = tmp_path / "run"
    assert (
        main(
            [
                "prestudy",
                "static",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "prestudy.static.json").read_text(encoding="utf-8"))
    assert payload["subject"] == "demo"
    assert payload["l1"]["candidate_count"] == 2
    assert payload["l1"]["eligible_count"] == 2
    assert payload["l1"]["delta"]["k_lines_nb"] > 0
    assert payload["decision"] == "pending_l2_l3"


def test_prestudy_is_deterministic(tmp_path, capsys):
    write(tmp_path, "pkg/a.py", "import os\n")
    out = tmp_path / "run"
    args = [
        "prestudy",
        "static",
        "--root",
        str(tmp_path),
        "--subject",
        "demo",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    first = (out / "prestudy.static.json").read_bytes()
    assert main(args) == 0
    second = (out / "prestudy.static.json").read_bytes()
    assert first == second


def test_prestudy_requires_a_stage(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "prestudy",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(tmp_path / "run"),
            ]
        )
    assert "static" in capsys.readouterr().err


def test_ledger_summary_cli(tmp_path, capsys):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, run_id="r1", clock=lambda: "2026-11-04T00:00:00+00:00")
    ledger.append("c001", "delete", "ok", round=1)
    ledger.append("c002", "keep", "drop", round=1)
    assert main(["ledger", "--path", str(path), "--json"]) == 0
    payload = output_json(capsys)
    assert payload["run_id"] == "r1"
    assert payload["summary"] == {"delete": 1, "keep": 1}


def test_error_exit_code(tmp_path, capsys):
    assert main(["measure", "--root", str(tmp_path / "missing"), "--json"]) == 2
    assert "error" in capsys.readouterr().err


def write_plan(path, subject="demo", components=("recording",)):
    path.write_text(
        json.dumps(
            {
                "subject": subject,
                "levers": [
                    {
                        "id": "terminal_recording",
                        "flag": "record_terminal_session",
                        "default": True,
                        "off": False,
                        "components": list(components),
                        "description": "asciinema capture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ablate_plan_writes_attributed_k(tmp_path, capsys):
    write(tmp_path, "recording.py", "x = 1\ny = 2\n")
    write(tmp_path, "agent.py", "def run():\n    return 1\n")
    plan = write_plan(tmp_path / "ablation.json")
    out = tmp_path / "run"
    assert (
        main(
            [
                "ablate",
                "--root",
                str(tmp_path),
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--boundary",
                "recording=recording.py",
                "--boundary",
                "agent=agent.py",
            ]
        )
        == 0
    )
    payload = json.loads((out / "prestudy.ablate-plan.json").read_text(encoding="utf-8"))
    assert payload["subject"] == "demo"
    assert payload["decision"] == "pending_runs"
    lever = payload["levers"][0]
    assert lever["id"] == "terminal_recording"
    assert lever["attributed_k"]["k_lines_nb"] == 2
    assert lever["agent_kwargs"] == {"record_terminal_session": False}
    assert lever["reduction_pct"]["k_lines_nb"] > 0
    assert {row["id"] for row in payload["components"]} == {
        "demo:recording",
        "demo:agent",
    }


def test_ablate_ratios_use_the_whole_subject_k(tmp_path, capsys):
    write(tmp_path, "recording.py", "x = 1\ny = 2\n")
    write(tmp_path, "agent.py", "".join(f"a{index} = {index}\n" for index in range(8)))
    plan = write_plan(tmp_path / "ablation.json")
    out = tmp_path / "run"
    assert (
        main(
            [
                "ablate",
                "--root",
                str(tmp_path),
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--boundary",
                "recording=recording.py",
            ]
        )
        == 0
    )
    payload = json.loads((out / "prestudy.ablate-plan.json").read_text(encoding="utf-8"))
    assert payload["totals"]["k_lines_nb"] == 10
    assert payload["levers"][0]["reduction_pct"]["k_lines_nb"] == pytest.approx(20.0)


def test_ablate_subject_defaults_to_plan(tmp_path, capsys):
    write(tmp_path, "recording.py", "x = 1\n")
    plan = write_plan(tmp_path / "ablation.json")
    out = tmp_path / "run"
    assert (
        main(
            [
                "ablate",
                "--root",
                str(tmp_path),
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--boundary",
                "recording=recording.py",
            ]
        )
        == 0
    )
    payload = json.loads((out / "prestudy.ablate-plan.json").read_text(encoding="utf-8"))
    assert payload["subject"] == "demo"


def test_ablate_rejects_subject_mismatch(tmp_path, capsys):
    write(tmp_path, "recording.py", "x = 1\n")
    plan = write_plan(tmp_path / "ablation.json")
    assert (
        main(
            [
                "ablate",
                "--root",
                str(tmp_path),
                "--plan",
                str(plan),
                "--subject",
                "other",
                "--out",
                str(tmp_path / "run"),
                "--boundary",
                "recording=recording.py",
            ]
        )
        == 2
    )
    assert "subject" in capsys.readouterr().err


def test_ablate_rejects_unknown_component(tmp_path, capsys):
    write(tmp_path, "recording.py", "x = 1\n")
    plan = write_plan(tmp_path / "ablation.json")
    assert (
        main(
            [
                "ablate",
                "--root",
                str(tmp_path),
                "--plan",
                str(plan),
                "--out",
                str(tmp_path / "run"),
                "--boundary",
                "elsewhere=recording.py",
            ]
        )
        == 2
    )
    assert "unknown component" in capsys.readouterr().err


def test_ablate_requires_existing_plan(tmp_path, capsys):
    write(tmp_path, "recording.py", "x = 1\n")
    assert (
        main(
            [
                "ablate",
                "--root",
                str(tmp_path),
                "--plan",
                str(tmp_path / "missing.json"),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "file" in capsys.readouterr().err


def write_traces(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def census_subject(tmp_path):
    write(
        tmp_path,
        "pkg/a.py",
        "def used():\n    return 1\n\ndef _dead():\n    return 2\n",
    )
    write(
        tmp_path,
        "pkg/b.py",
        "@wrapper\ndef _decorated():\n    return 3\n",
    )


def traces_payload(subject="demo", task_ids=("t1",), tasks=None):
    return {
        "subject": subject,
        "root": ".",
        "task_ids": list(task_ids),
        "tasks": tasks if tasks is not None else {"t1": {"pkg/a.py": [2]}},
    }


def test_census_reports_zero_touch_candidates(tmp_path, capsys):
    census_subject(tmp_path)
    traces = write_traces(tmp_path / "traces.json", traces_payload())
    out = tmp_path / "run"
    assert (
        main(
            [
                "census",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--traces",
                str(traces),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "census.l2.json").read_text(encoding="utf-8"))
    symbols = {candidate["symbol"] for candidate in payload["candidates"]}
    assert symbols == {"_dead", "_decorated"}
    assert payload["tasks"] == ["t1"]
    assert payload["zero_touch"] == 2
    assert payload["touched"] == 1
    dead = next(c for c in payload["candidates"] if c["symbol"] == "_dead")
    assert dead["blocked"] is False
    assert dead["delta"]["k_lines_nb"] == 2
    decorated = next(c for c in payload["candidates"] if c["symbol"] == "_decorated")
    assert decorated["blocked"] is True
    assert {hit["kind"] for hit in decorated["protection"]} == {"decorator"}


def test_census_requires_known_subject(tmp_path, capsys):
    census_subject(tmp_path)
    traces = write_traces(tmp_path / "traces.json", traces_payload(subject="other"))
    assert (
        main(
            [
                "census",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--traces",
                str(traces),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )


def test_census_rejects_traces_that_match_nothing(tmp_path, capsys):
    census_subject(tmp_path)
    traces = write_traces(
        tmp_path / "traces.json",
        traces_payload(tasks={"t1": {"elsewhere/x.py": [1]}}),
    )
    assert (
        main(
            [
                "census",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--traces",
                str(traces),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )


def test_census_requires_traces_file(tmp_path, capsys):
    census_subject(tmp_path)
    assert (
        main(
            [
                "census",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--traces",
                str(tmp_path / "absent.json"),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )


def test_prestudy_decide_writes_json(tmp_path, capsys):
    census_subject(tmp_path)
    traces = write_traces(tmp_path / "traces.json", traces_payload())
    out = tmp_path / "run"
    assert (
        main(
            [
                "prestudy",
                "static",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "census",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--traces",
                str(traces),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "prestudy",
                "decide",
                "--static",
                str(out / "prestudy.static.json"),
                "--census",
                str(out / "census.l2.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "prestudy.json").read_text(encoding="utf-8"))
    assert payload["subject"] == "demo"
    assert payload["k_metric"] == "k_ast"
    assert payload["decision"] == "pending_l2"
    assert payload["l3"] is None
    assert payload["census"]["zero_touch"] == 2
    assert payload["census"]["tasks"] == ["t1"]
    assert payload["static"]["reduction_pct"]["k_ast"] > 0


def test_prestudy_decide_accepts_l3_results(tmp_path, capsys):
    census_subject(tmp_path)
    traces = write_traces(tmp_path / "traces.json", traces_payload())
    out = tmp_path / "run"
    main(["prestudy", "static", "--root", str(tmp_path), "--subject", "demo", "--out", str(out)])
    capsys.readouterr()
    main(
        [
            "census",
            "--root",
            str(tmp_path),
            "--subject",
            "demo",
            "--traces",
            str(traces),
            "--out",
            str(out),
        ]
    )
    capsys.readouterr()
    l3 = tmp_path / "l3.json"
    l3.write_text(
        json.dumps(
            {
                "subject": "demo",
                "levers": [
                    {"id": "a", "verdict": "zero_contribution"},
                    {"id": "b", "verdict": "contributes"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "prestudy",
                "decide",
                "--static",
                str(out / "prestudy.static.json"),
                "--census",
                str(out / "census.l2.json"),
                "--l3",
                str(l3),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "prestudy.json").read_text(encoding="utf-8"))
    assert payload["l3"] == {"yield": 0.5, "zero_contribution": 1, "evaluated": 2}


def test_prestudy_decide_reports_missing_inputs(tmp_path, capsys):
    static = tmp_path / "prestudy.static.json"
    static.write_text(
        json.dumps(
            {
                "subject": "demo",
                "l1": {
                    "reduction_pct": {
                        "k_lines_nb": 0.0,
                        "k_ast": 0.0,
                        "k_chars": 0.0,
                        "k_cc": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "prestudy",
                "decide",
                "--static",
                str(static),
                "--census",
                str(tmp_path / "absent.json"),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "census" in capsys.readouterr().err


def test_prestudy_decide_rejects_subject_mismatch(tmp_path, capsys):
    static = tmp_path / "prestudy.static.json"
    census = tmp_path / "census.l2.json"
    reduction = {"k_lines_nb": 0.0, "k_ast": 0.0, "k_chars": 0.0, "k_cc": 0.0}
    static.write_text(
        json.dumps({"subject": "demo", "l1": {"reduction_pct": reduction}}),
        encoding="utf-8",
    )
    census.write_text(
        json.dumps({"subject": "other", "l2": {"reduction_pct": reduction}}),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "prestudy",
                "decide",
                "--static",
                str(static),
                "--census",
                str(census),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "subject mismatch" in capsys.readouterr().err


def _outcomes_file(tmp_path, rows):
    path = tmp_path / "outcomes.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_baseline_writes_artifact(tmp_path, capsys):
    write(tmp_path, "a.py", "x = 1\n")
    outcomes = _outcomes_file(
        tmp_path,
        [
            {"task_id": "t0", "trial_id": "r0", "reward": 1.0, "tokens_in": 10, "tokens_out": 2},
            {"task_id": "t1", "trial_id": "r1", "reward": 0.0, "tokens_in": 20, "tokens_out": 4},
        ],
    )
    assert (
        main(
            [
                "baseline",
                "--outcomes",
                str(outcomes),
                "--tasks",
                "t0,t1",
                "--reps",
                "1",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    artifact = tmp_path / "run" / "baseline.json"
    assert str(artifact) in capsys.readouterr().out
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["arm"] == "H0"
    assert payload["subject"] == "demo"
    assert set(payload["tasks"]) == {"t0", "t1"}
    assert payload["decision"] == "baseline_valid"
    assert payload["aggregate"]["mean_reward"] == 0.5
    assert payload["k"]["k_lines_nb"] == 1


def test_baseline_rejects_bad_json_line(tmp_path, capsys):
    write(tmp_path, "a.py", "x = 1\n")
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text('{"task_id": "t0", "trial_id": "r0", "reward": 1.0}\nnot-json\n', encoding="utf-8")
    assert (
        main(
            [
                "baseline",
                "--outcomes",
                str(outcomes),
                "--tasks",
                "t0",
                "--reps",
                "1",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "line 2" in capsys.readouterr().err


def test_baseline_rejects_unknown_task(tmp_path, capsys):
    write(tmp_path, "a.py", "x = 1\n")
    outcomes = _outcomes_file(
        tmp_path, [{"task_id": "t9", "trial_id": "r0", "reward": 1.0}]
    )
    assert (
        main(
            [
                "baseline",
                "--outcomes",
                str(outcomes),
                "--tasks",
                "t0",
                "--reps",
                "1",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "unknown task" in capsys.readouterr().err


def _write_trial_result(tmp_path, trial="fix-git__abc", reward=1.0):
    trial_dir = tmp_path / "jobs" / "job1" / trial
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "trial_name": trial,
                "trial_uri": f"file:///jobs/{trial}",
                "agent_result": {"n_input_tokens": 10, "n_output_tokens": 2, "cost_usd": 0.1},
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )


def test_harbor_results_writes_outcomes_jsonl(tmp_path, capsys):
    _write_trial_result(tmp_path)
    out = tmp_path / "outcomes.jsonl"
    assert (
        main(
            [
                "harbor-results",
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--job-name",
                "job1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert str(out) in capsys.readouterr().out
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "task_id": "terminal-bench/fix-git",
            "trial_id": "fix-git__abc",
            "status": "completed",
            "reward": 1.0,
            "tokens_in": 10,
            "tokens_out": 2,
            "cost_usd": 0.1,
            "failure_cause": None,
        }
    ]


def test_harbor_results_rejects_a_missing_job(tmp_path, capsys):
    assert (
        main(
            [
                "harbor-results",
                "--jobs-dir",
                str(tmp_path),
                "--job-name",
                "nope",
                "--out",
                str(tmp_path / "out.jsonl"),
            ]
        )
        == 2
    )
    assert "job directory" in capsys.readouterr().err


def test_harbor_run_dry_run_prints_the_command(tmp_path, capsys):
    assert (
        main(
            [
                "harbor-run",
                "--tasks",
                "terminal-bench/fix-git",
                "--jobs-dir",
                str(tmp_path),
                "--job-name",
                "job1",
                "--log",
                str(tmp_path / "run.log"),
                "--concurrency",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = output_json(capsys)
    assert payload["argv"][1:3] == ["-m", "harbor.cli.main"]
    assert "terminus2:AgentHarness" in payload["argv"]
    assert "deepseek-v4-flash" in payload["argv"]
    assert payload["argv"][payload["argv"].index("-n") + 1] == "2"
    assert payload["env"]["PYTHONUTF8"] == "1"


def test_harbor_run_dry_run_does_not_need_the_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert (
        main(
            [
                "harbor-run",
                "--tasks",
                "terminal-bench/fix-git",
                "--jobs-dir",
                str(tmp_path),
                "--job-name",
                "job1",
                "--log",
                str(tmp_path / "run.log"),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()


def test_harbor_run_requires_the_api_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert (
        main(
            [
                "harbor-run",
                "--tasks",
                "terminal-bench/fix-git",
                "--jobs-dir",
                str(tmp_path),
                "--job-name",
                "job1",
                "--log",
                str(tmp_path / "run.log"),
            ]
        )
        == 2
    )
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err


def test_harbor_run_refuses_to_reuse_an_existing_job(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    (tmp_path / "job1").mkdir()
    assert (
        main(
            [
                "harbor-run",
                "--tasks",
                "terminal-bench/fix-git",
                "--jobs-dir",
                str(tmp_path),
                "--job-name",
                "job1",
                "--log",
                str(tmp_path / "run.log"),
            ]
        )
        == 2
    )
    assert "already exists" in capsys.readouterr().err


def write_run_fixture(tmp_path, *, text="x = 1\n", k_reps=1, tasks=None):
    subject = tmp_path / "vendor" / "terminus2"
    write(subject, "agent.py", text)
    files = cli_module._py_files(subject)
    k = cli_module.measure_paths(subject, files).as_dict()
    tasks = tasks or ["terminal-bench/fix-git"]
    run = tmp_path / "frozen" / "run.json"
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text(
        json.dumps(
            {
                "schema": 1,
                "dataset": "terminal-bench/terminal-bench-2-1",
                "subject": "terminus2",
                "pools": {"smoke": tasks, "evolution": ["terminal-bench/evo"], "holdout": ["terminal-bench/hold"]},
                "prereg": {"k": k_reps, "n": len(tasks), "margin": 0.1, "holdout_margin": 0.1},
                "k": k,
            }
        ),
        encoding="utf-8",
    )
    return subject, run


def run_argv(run, subject, tmp_path, *extra):
    return [
        "run",
        "--run",
        str(run),
        "--arm",
        "H0",
        "--pool",
        "smoke",
        "--subject-root",
        str(subject),
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--job-name",
        "h0-smoke",
        "--log",
        str(tmp_path / "h0.log"),
        "--harbor-src",
        str(tmp_path / "harbor" / "src"),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def test_run_dry_run_prints_the_command_and_plan(tmp_path, capsys):
    subject, run = write_run_fixture(tmp_path, k_reps=2)
    assert main(run_argv(run, subject, tmp_path, "--dry-run")) == 0
    payload = output_json(capsys)
    argv = payload["argv"]
    assert "-a" in argv and argv[argv.index("-a") + 1] == "terminus2:AgentHarness"
    assert argv[argv.index("-k") + 1] == "2"
    assert argv[argv.index("-i") + 1] == "terminal-bench/fix-git"
    expected = [
        str(subject.resolve().parent),
        str((tmp_path / "harbor" / "src").resolve()),
    ]
    assert payload["env"]["PYTHONPATH"].split(os.pathsep) == expected
    assert payload["plan"]["arm"] == "H0"
    assert payload["plan"]["task_ids"] == ["terminal-bench/fix-git"]
    assert not (tmp_path / "out").exists()


def test_run_requires_the_h0_subject_to_match_the_frozen_k(tmp_path, capsys):
    subject, run = write_run_fixture(tmp_path)
    write(subject, "agent.py", "x = 1\ny = 2\n")
    assert main(run_argv(run, subject, tmp_path, "--dry-run")) == 2
    assert "frozen" in capsys.readouterr().err


def test_run_allows_a_non_reference_arm_to_change_the_subject(tmp_path, capsys):
    subject, run = write_run_fixture(tmp_path)
    write(subject, "agent.py", "x = 1\ny = 2\n")
    argv = run_argv(run, subject, tmp_path, "--dry-run")
    argv[argv.index("--arm") + 1] = "SIM"
    assert main(argv) == 0
    capsys.readouterr()


def test_run_writes_arm_json_and_outcomes(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    subject, run = write_run_fixture(tmp_path)
    captured = {}

    def fake_run(argv, *, log_path, env=None, cwd=None, timeout=None):
        captured["argv"] = list(argv)
        trial = tmp_path / "jobs" / "h0-smoke" / "fix-git__t1"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "terminal-bench/fix-git",
                    "trial_name": "fix-git__t1",
                    "agent_result": {
                        "n_input_tokens": 10,
                        "n_output_tokens": 2,
                        "cost_usd": 0.1,
                    },
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": None,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(cli_module, "run_harbor", fake_run)
    assert main(run_argv(run, subject, tmp_path)) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "out" / "arm.json")
    payload = json.loads((tmp_path / "out" / "arm.json").read_text(encoding="utf-8"))
    assert payload["arm"] == "H0"
    assert payload["decision"] == "baseline_valid"
    assert payload["aggregate"]["mean_reward"] == 1.0
    assert payload["subject_hash"]
    assert [row["trial_id"] for row in payload["outcomes"]] == ["fix-git__t1"]
    lines = (tmp_path / "out" / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["task_id"] == "terminal-bench/fix-git"
    assert "terminus2:AgentHarness" in captured["argv"]


def test_run_refuses_to_reuse_an_existing_job(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    subject, run = write_run_fixture(tmp_path)
    (tmp_path / "jobs" / "h0-smoke").mkdir(parents=True)
    assert main(run_argv(run, subject, tmp_path)) == 2
    assert "already exists" in capsys.readouterr().err


def git_repo(root):
    run = subprocess.run
    run(["git", "init", "-q", "-b", "main"], cwd=root, capture_output=True, check=True)
    run(["git", "add", "."], cwd=root, capture_output=True, check=True)
    run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init"],
        cwd=root,
        capture_output=True,
        check=True,
    )


def candidate_json(root, **overrides):
    payload = {
        "id": "l1-0001",
        "layer": "L1",
        "kind": "dead_function",
        "component": "demo",
        "path": "pkg/mod.py",
        "span": [1, 2],
        "symbol": "_dead",
        "evidence": {},
        "blocked": False,
        "protection": [],
    }
    payload.update(overrides)
    path = root / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def apply_argv(repo, worktree, candidate, out, *extra):
    return [
        "apply",
        "--repo",
        str(repo),
        "--worktree",
        str(worktree),
        "--candidate",
        str(candidate),
        "--out",
        str(out),
        *extra,
    ]


@pytest.fixture
def apply_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    write(repo, "pkg/__init__.py", "")
    write(repo, "pkg/mod.py", "def _dead():\n    return 1\n\n\ndef keep():\n    return 2\n")
    git_repo(repo)
    return repo


def test_apply_writes_deletion_artifacts(apply_repo, tmp_path, capsys):
    candidate = candidate_json(tmp_path)
    out = tmp_path / "run"
    worktree = tmp_path / "wt"
    assert (
        main(
            apply_argv(
                apply_repo,
                worktree,
                candidate,
                out,
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
            )
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(out / "applied.json")
    payload = json.loads((out / "applied.json").read_text(encoding="utf-8"))
    assert payload["candidate"]["id"] == "l1-0001"
    assert payload["baseline_import"] == "pass"
    assert payload["verify"]["verdict"] == "pass"
    assert payload["undeletable"] is False
    assert payload["after"]["k_lines_nb"] < payload["before"]["k_lines_nb"]
    assert payload["delta"]["k_lines_nb"] == 2
    assert payload["diff"] == "diff.patch"
    patch = (out / "diff.patch").read_text(encoding="utf-8")
    assert "--- a/pkg/mod.py" in patch
    assert "-def _dead():" in patch
    assert not worktree.exists()


def test_apply_keep_leaves_the_worktree(apply_repo, tmp_path, capsys):
    candidate = candidate_json(tmp_path)
    worktree = tmp_path / "wt"
    assert (
        main(
            apply_argv(
                apply_repo,
                worktree,
                candidate,
                tmp_path / "run",
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
                "--keep",
            )
        )
        == 0
    )
    text = (worktree / "pkg/mod.py").read_text(encoding="utf-8")
    assert "_dead" not in text
    assert "def keep():" in text


def test_apply_reports_residual_call_sites(apply_repo, tmp_path, capsys):
    write(apply_repo, "pkg/mod.py", "def _dead():\n    return 1\n\n\nVALUE = _dead()\n")
    subprocess.run(["git", "add", "."], cwd=apply_repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "two"],
        cwd=apply_repo,
        capture_output=True,
        check=True,
    )
    candidate = candidate_json(tmp_path)
    out = tmp_path / "run"
    assert (
        main(
            apply_argv(
                apply_repo,
                tmp_path / "wt",
                candidate,
                out,
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
            )
        )
        == 0
    )
    payload = json.loads((out / "applied.json").read_text(encoding="utf-8"))
    assert payload["undeletable"] is True
    assert "_dead" in payload["undeletable_reason"]
    assert payload["verify"]["verdict"] == "fail"
    assert not (tmp_path / "wt").exists()


def test_apply_refuses_a_protected_candidate(apply_repo, tmp_path, capsys):
    candidate = candidate_json(
        tmp_path,
        blocked=True,
        protection=[{"kind": "assert", "path": "pkg/mod.py", "start": 1, "end": 2, "reason": "assert statement"}],
    )
    assert (
        main(
            apply_argv(
                apply_repo,
                tmp_path / "wt",
                candidate,
                tmp_path / "run",
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
            )
        )
        == 2
    )
    assert "protected" in capsys.readouterr().err
    assert not (tmp_path / "wt").exists()


def test_apply_rejects_a_malformed_span(apply_repo, tmp_path, capsys):
    candidate = candidate_json(tmp_path, span=[2])
    assert (
        main(
            apply_argv(
                apply_repo,
                tmp_path / "wt",
                candidate,
                tmp_path / "run",
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
            )
        )
        == 2
    )
    assert "span" in capsys.readouterr().err


def test_apply_rejects_a_missing_field(apply_repo, tmp_path, capsys):
    payload = {"id": "l1-0001", "layer": "L1"}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        main(
            apply_argv(
                apply_repo,
                tmp_path / "wt",
                candidate,
                tmp_path / "run",
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
            )
        )
        == 2
    )
    assert "missing" in capsys.readouterr().err


def test_apply_rejects_a_repository_root_that_is_not_a_repo(tmp_path, capsys):
    bare = tmp_path / "plain"
    bare.mkdir()
    write(bare, "pkg/mod.py", "def _dead():\n    return 1\n")
    candidate = candidate_json(tmp_path)
    assert (
        main(
            apply_argv(
                bare,
                tmp_path / "wt",
                candidate,
                tmp_path / "run",
                "--path",
                "pkg/mod.py",
                "--module",
                "pkg.mod",
            )
        )
        == 2
    )
    assert "not the root" in capsys.readouterr().err


def test_apply_copies_untracked_subjects_through_include(apply_repo, tmp_path, capsys):
    write(apply_repo, "vendor/agent/__init__.py", "")
    write(apply_repo, "vendor/agent/mod.py", "def _dead():\n    return 1\n")
    candidate = candidate_json(tmp_path, path="agent/mod.py")
    out = tmp_path / "run"
    assert (
        main(
            apply_argv(
                apply_repo,
                tmp_path / "wt",
                candidate,
                out,
                "--subject-root",
                "vendor",
                "--import-root",
                "vendor",
                "--path",
                "agent/mod.py",
                "--module",
                "agent.mod",
                "--include",
                "vendor",
                "--keep",
            )
        )
        == 0
    )
    payload = json.loads((out / "applied.json").read_text(encoding="utf-8"))
    assert payload["subject_root"] == "vendor"
    assert payload["verify"]["verdict"] == "pass"
    assert payload["undeletable"] is False
    assert "_dead" not in (tmp_path / "wt/vendor/agent/mod.py").read_text(encoding="utf-8")
    assert "_dead" in (apply_repo / "vendor/agent/mod.py").read_text(encoding="utf-8")


def outcome_line(task, trial, reward, tokens_in=100, tokens_out=10, cost=0.001):
    return {
        "task_id": task,
        "trial_id": trial,
        "status": "completed",
        "reward": reward,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "failure_cause": None,
    }


def write_outcomes(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def compare_argv(reference, candidate, out, *extra):
    return [
        "compare",
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--out",
        str(out),
        *extra,
    ]


def test_compare_cli_writes_a_comparison(tmp_path, capsys):
    reference = write_outcomes(
        tmp_path / "ref.jsonl",
        [outcome_line("t1", "r1", 1.0), outcome_line("t2", "r2", 0.0)],
    )
    candidate = write_outcomes(
        tmp_path / "cand.jsonl",
        [outcome_line("t1", "c1", 1.0), outcome_line("t2", "c2", 1.0)],
    )
    out = tmp_path / "run"
    assert (
        main(
            compare_argv(
                reference,
                candidate,
                out,
                "--tasks",
                "t1,t2",
                "--n-boot",
                "200",
            )
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(out / "comparison.json")
    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["aggregate"]["n_pairs"] == 2
    assert payload["aggregate"]["mean_delta"] == pytest.approx(0.5)
    assert payload["tasks"]["t2"]["delta"] == 1.0


def test_compare_cli_uses_a_frozen_pool(tmp_path, capsys):
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "pools": {
                    "smoke": ["t1", "t2"],
                    "evolution": ["t3"],
                    "holdout": ["t4"],
                }
            }
        ),
        encoding="utf-8",
    )
    reference = write_outcomes(
        tmp_path / "ref.jsonl",
        [outcome_line("t1", "r1", 1.0), outcome_line("t2", "r2", 1.0)],
    )
    candidate = write_outcomes(
        tmp_path / "cand.jsonl",
        [outcome_line("t1", "c1", 1.0), outcome_line("t2", "c2", 1.0)],
    )
    assert (
        main(
            compare_argv(
                reference,
                candidate,
                tmp_path / "out",
                "--run",
                str(run),
                "--pool",
                "smoke",
            )
        )
        == 0
    )
    payload = json.loads((tmp_path / "out/comparison.json").read_text(encoding="utf-8"))
    assert sorted(payload["tasks"]) == ["t1", "t2"]


def test_compare_cli_requires_exactly_one_task_source(tmp_path, capsys):
    reference = write_outcomes(tmp_path / "ref.jsonl", [outcome_line("t1", "r1", 1.0)])
    candidate = write_outcomes(tmp_path / "cand.jsonl", [outcome_line("t1", "c1", 1.0)])
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"pools": {"smoke": ["t1"]}}), encoding="utf-8")
    assert (
        main(
            compare_argv(
                reference,
                candidate,
                tmp_path / "out",
                "--tasks",
                "t1",
                "--run",
                str(run),
            )
        )
        == 2
    )
    assert "either" in capsys.readouterr().err


def test_compare_cli_rejects_an_unknown_pool(tmp_path, capsys):
    reference = write_outcomes(tmp_path / "ref.jsonl", [outcome_line("t1", "r1", 1.0)])
    candidate = write_outcomes(tmp_path / "cand.jsonl", [outcome_line("t1", "c1", 1.0)])
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"pools": {"smoke": ["t1"]}}), encoding="utf-8")
    assert (
        main(
            compare_argv(
                reference,
                candidate,
                tmp_path / "out",
                "--run",
                str(run),
                "--pool",
                "nope",
            )
        )
        == 2
    )
    assert "pool" in capsys.readouterr().err


def test_compare_cli_reads_k_from_an_applied_deletion(tmp_path, capsys):
    reference = write_outcomes(tmp_path / "ref.jsonl", [outcome_line("t1", "r1", 1.0)])
    candidate = write_outcomes(tmp_path / "cand.jsonl", [outcome_line("t1", "c1", 1.0)])
    applied = tmp_path / "applied.json"
    applied.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            compare_argv(
                reference,
                candidate,
                tmp_path / "out",
                "--tasks",
                "t1",
                "--applied",
                str(applied),
            )
        )
        == 0
    )
    payload = json.loads((tmp_path / "out/comparison.json").read_text(encoding="utf-8"))
    assert payload["k"]["reduction_pct"]["k_lines_nb"] == 10.0


def frozen_run(root, *, margin=0.10, holdout_margin=0.10, k=1):
    payload = {
        "schema": 1,
        "dataset": "terminal-bench/terminal-bench-2-1",
        "subject": "terminus2",
        "pools": {
            "smoke": [],
            "evolution": ["t1", "t2"],
            "holdout": ["h1"],
        },
        "prereg": {
            "primary_endpoint": "reward",
            "margin": margin,
            "holdout_margin": holdout_margin,
            "n": 3,
            "k": k,
            "stopping_rule": "stop at the first rejected round",
        },
    }
    path = root / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def applied_json(root, **overrides):
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
    path = root / "applied.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def round_argv(run, candidate, applied, reference, evaluated, ledger, out, *extra):
    return [
        "round",
        "--run",
        str(run),
        "--candidate",
        str(candidate),
        "--applied",
        str(applied),
        "--reference",
        str(reference),
        "--evaluated",
        str(evaluated),
        "--ledger",
        str(ledger),
        "--out",
        str(out),
        *extra,
    ]


def round_inputs(tmp_path):
    run = frozen_run(tmp_path)
    candidate = candidate_json(tmp_path)
    applied = applied_json(tmp_path)
    tasks = ("t1", "t2", "h1")
    reference = write_outcomes(
        tmp_path / "ref.jsonl",
        [outcome_line(task, f"r-{task}", 1.0) for task in tasks],
    )
    evaluated = write_outcomes(
        tmp_path / "sim.jsonl",
        [outcome_line(task, f"s-{task}", 1.0) for task in tasks],
    )
    return run, candidate, applied, reference, evaluated


def test_round_cli_writes_a_decision_and_a_ledger_entry(tmp_path, capsys):
    run, candidate, applied, reference, evaluated = round_inputs(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "round1"
    assert (
        main(
            round_argv(
                run,
                candidate,
                applied,
                reference,
                evaluated,
                ledger,
                out,
                "--n-boot",
                "200",
            )
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(out / "decision.json")
    payload = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert payload["decision"]["verdict"] == "delete"
    assert payload["decision"]["gate"]["verdict"] == "accept"
    entries = Ledger(ledger, run_id=payload["run_id"]).entries()
    assert [entry.verdict for entry in entries] == ["delete"]
    assert entries[0].candidate_id == "l1-0001"


def test_round_cli_refuses_a_margin_that_drifts_from_prereg(tmp_path, capsys):
    run, candidate, applied, reference, evaluated = round_inputs(tmp_path)
    rc = main(
        round_argv(
            run,
            candidate,
            applied,
            reference,
            evaluated,
            tmp_path / "ledger.jsonl",
            tmp_path / "round1",
            "--margin",
            "0.2",
            "--n-boot",
            "200",
        )
    )
    assert rc == 2
    assert "pre-registration" in capsys.readouterr().err


def test_round_cli_refuses_an_unknown_pool(tmp_path, capsys):
    run, candidate, applied, reference, evaluated = round_inputs(tmp_path)
    rc = main(
        round_argv(
            run,
            candidate,
            applied,
            reference,
            evaluated,
            tmp_path / "ledger.jsonl",
            tmp_path / "round1",
            "--pool",
            "nope",
            "--n-boot",
            "200",
        )
    )
    assert rc == 2
    assert "pool" in capsys.readouterr().err


def report_comparison():
    return {
        "aggregate": {
            "n_pairs": 2,
            "reference_mean": 1.0,
            "candidate_mean": 1.0,
            "mean_delta": 0.0,
            "ci": {"low": -0.05, "high": 0.05},
            "margin": 0.1,
            "non_inferior": True,
            "worst_task": {"task": "t1", "delta": 0.0},
            "tokens": {"reference": {"total": 220}, "candidate": {"total": 210}},
            "cost": {"reference": 0.002, "candidate": 0.0019},
        },
        "validity": {
            "reference": {"status": "pass"},
            "candidate": {"status": "pass"},
        },
        "arms": {
            "reference": {"aggregate": {"valid_trials": 2}},
            "candidate": {"aggregate": {"valid_trials": 2}},
        },
        "k": {
            "before": {"k_lines_nb": 200, "k_ast": 400, "k_chars": 1000,
                       "k_bytes": 1000, "k_gzip": 100, "k_cc": 10},
            "after": {"k_lines_nb": 180, "k_ast": 360, "k_chars": 900,
                      "k_bytes": 900, "k_gzip": 90, "k_cc": 9},
            "reduction_pct": {"k_lines_nb": 10.0, "k_ast": 10.0},
        },
    }


def test_report_cli_writes_six_tables(tmp_path, capsys):
    run = frozen_run(tmp_path)
    prestudy = tmp_path / "prestudy.json"
    prestudy.write_text(
        json.dumps(
            {
                "subject": "terminus2",
                "components": [
                    {
                        "id": "terminus2:agent.py",
                        "paths": ["agent.py"],
                        "k": {"k_lines_nb": 200, "k_ast": 400, "k_chars": 1000,
                              "k_bytes": 1000, "k_gzip": 100, "k_cc": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(report_comparison()), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "tasks_with_valid_trials": 2,
                    "valid_trials": 2,
                    "planned_trials": 2,
                    "mean_reward": 1.0,
                    "tokens_in": 200,
                    "tokens_out": 20,
                    "cost_usd": 0.002,
                },
                "validity": {"status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "round": 1,
                "candidate_id": "l1-0001",
                "verdict": "delete",
                "reason": "all gates pass",
                "evidence": {"layer": "L1", "verify": "pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"
    assert (
        main(
            [
                "report",
                "--run",
                str(run),
                "--prestudy",
                str(prestudy),
                "--baseline",
                str(baseline_path),
                "--comparison",
                f"SIM={comparison}",
                "--ledger",
                str(ledger),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(out / "tables.json")
    manifest = json.loads((out / "tables.json").read_text(encoding="utf-8"))
    assert set(manifest["tables"]) == {
        "t1_subject", "t2_arms", "t3_deletions",
        "t4_baselines", "t5_diagnosis", "t6_cost",
    }
    assert manifest["tables"]["t1_subject"]["status"] == "ok"
    assert manifest["tables"]["t2_arms"]["status"] == "ok"
    assert manifest["tables"]["t3_deletions"]["status"] == "ok"
    assert manifest["tables"]["t4_baselines"]["status"] == "not_run"
    assert manifest["tables"]["t6_cost"]["status"] == "ok"
    t2 = (out / "tables/t2_arms.csv").read_text(encoding="utf-8")
    assert t2.splitlines()[2].startswith("SIM,comparison,2,1.0,1.0")
    t3 = (out / "tables/t3_deletions.csv").read_text(encoding="utf-8")
    assert "l1-0001,delete" in t3


def test_report_cli_marks_missing_inputs_not_run(tmp_path, capsys):
    run = frozen_run(tmp_path)
    out = tmp_path / "report"
    assert main(["report", "--run", str(run), "--out", str(out)]) == 0
    manifest = json.loads((out / "tables.json").read_text(encoding="utf-8"))
    assert all(
        table["status"] == "not_run" for table in manifest["tables"].values()
    )
    assert (out / "tables/t5_diagnosis.csv").read_text(encoding="utf-8").strip() == (
        "phase,metric,value"
    )


def test_report_cli_refuses_a_bad_comparison_flag(tmp_path, capsys):
    run = frozen_run(tmp_path)
    rc = main(
        ["report", "--run", str(run), "--comparison", "SIM", "--out", str(tmp_path / "r")]
    )
    assert rc == 2
    assert "ARM=PATH" in capsys.readouterr().err


def build_report_dir(tmp_path):
    run = frozen_run(tmp_path)
    prestudy = tmp_path / "prestudy.json"
    prestudy.write_text(
        json.dumps(
            {
                "subject": "terminus2",
                "components": [
                    {
                        "id": "terminus2:agent.py",
                        "paths": ["agent.py"],
                        "k": {"k_lines_nb": 200, "k_ast": 400, "k_chars": 1000,
                              "k_bytes": 1000, "k_gzip": 100, "k_cc": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(report_comparison()), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "tasks_with_valid_trials": 2,
                    "valid_trials": 2,
                    "planned_trials": 2,
                    "mean_reward": 1.0,
                    "tokens_in": 200,
                    "tokens_out": 20,
                    "cost_usd": 0.002,
                },
                "validity": {"status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "round": 1,
                "candidate_id": "l1-0001",
                "verdict": "delete",
                "reason": "all gates pass",
                "evidence": {"layer": "L1", "verify": "pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report"
    assert (
        main(
            [
                "report",
                "--run",
                str(run),
                "--prestudy",
                str(prestudy),
                "--baseline",
                str(baseline_path),
                "--comparison",
                f"SIM={comparison}",
                "--ledger",
                str(ledger),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    return out, comparison


def test_verify_evidence_passes_on_an_untouched_report(tmp_path, capsys):
    out, _ = build_report_dir(tmp_path)
    capsys.readouterr()
    assert main(["verify-evidence", "--report", str(out)]) == 0
    payload = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["issues"] == []
    assert all(item["match"] for item in payload["tables"].values())


def test_verify_evidence_detects_a_tampered_table(tmp_path, capsys):
    out, _ = build_report_dir(tmp_path)
    table = out / "tables/t2_arms.csv"
    table.write_text(table.read_text(encoding="utf-8") + "FAKE,row\n", encoding="utf-8")
    capsys.readouterr()
    assert main(["verify-evidence", "--report", str(out)]) == 1
    payload = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["tables"]["t2_arms"]["match"] is False
    assert payload["tables"]["t1_subject"]["match"] is True


def test_verify_evidence_detects_an_edited_input(tmp_path, capsys):
    out, comparison = build_report_dir(tmp_path)
    document = json.loads(comparison.read_text(encoding="utf-8"))
    document["aggregate"]["reference_mean"] = 0.9
    comparison.write_text(json.dumps(document), encoding="utf-8")
    capsys.readouterr()
    assert main(["verify-evidence", "--report", str(out)]) == 1
    payload = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert any("comparisons[SIM]" in issue for issue in payload["issues"])
    assert payload["tables"]["t2_arms"]["match"] is False


def evolve_inputs(root, *, candidate_reward=0.6):
    run = frozen_run(root)
    tasks = ("t1", "t2")
    incumbent = write_outcomes(
        root / "inc.jsonl",
        [outcome_line(task, f"i-{task}", 0.5) for task in tasks],
    )
    candidate_a = write_outcomes(
        root / "a.jsonl",
        [outcome_line(task, f"a-{task}", candidate_reward) for task in tasks],
    )
    candidate_b = write_outcomes(
        root / "b.jsonl",
        [outcome_line(task, f"b-{task}", 0.55) for task in tasks],
    )
    return run, incumbent, candidate_a, candidate_b


def evolve_argv(run, incumbent, candidate_a, candidate_b, history, out, *extra):
    return [
        "evolve-round",
        "--run",
        str(run),
        "--incumbent",
        str(incumbent),
        "--candidate",
        f"A={candidate_a}",
        "--candidate",
        f"B={candidate_b}",
        "--delta",
        "0.01",
        "--round",
        "0",
        "--total-rounds",
        "5",
        "--history",
        str(history),
        "--out",
        str(out),
        *extra,
    ]


def test_evolve_round_cli_writes_decisions_and_history(tmp_path, capsys):
    run, incumbent, candidate_a, candidate_b = evolve_inputs(tmp_path)
    history = tmp_path / "history.jsonl"
    out = tmp_path / "r0"
    assert main(evolve_argv(run, incumbent, candidate_a, candidate_b, history, out)) == 0
    printed = capsys.readouterr().out.strip()
    assert printed == str(out / "decisions.json")
    payload = json.loads((out / "decisions.json").read_text(encoding="utf-8"))
    assert payload["winner"] == "A"
    assert payload["round"] == 0
    assert payload["budget"] == 4
    rows = [
        json.loads(line)
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"A", "B"}
    assert [row["outcome"] for row in rows if row["variant"] == "A"] == ["ACCEPTED"]


def test_evolve_round_cli_refuses_duplicate_round_history(tmp_path, capsys):
    run, incumbent, candidate_a, candidate_b = evolve_inputs(tmp_path)
    history = tmp_path / "history.jsonl"
    out = tmp_path / "r0"
    assert main(evolve_argv(run, incumbent, candidate_a, candidate_b, history, out)) == 0
    capsys.readouterr()
    assert main(evolve_argv(run, incumbent, candidate_a, candidate_b, history, out)) == 2
    assert "already" in capsys.readouterr().err


def test_evolve_round_cli_loads_tagged_edits(tmp_path, capsys):
    run, incumbent, candidate_a, candidate_b = evolve_inputs(tmp_path)
    edits = tmp_path / "a.edits.json"
    edits.write_text(
        json.dumps(
            [
                {
                    "id": "C1",
                    "component": "core",
                    "hypothesis": "tighten retries",
                    "predicted_affected": ["t1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    history = tmp_path / "history.jsonl"
    out = tmp_path / "r0"
    assert (
        main(
            evolve_argv(
                run,
                incumbent,
                candidate_a,
                candidate_b,
                history,
                out,
                "--edits",
                f"A={edits}",
            )
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tagged = [row for row in rows if row["variant"] == "A"]
    assert tagged[0]["component"] == "core"
    assert tagged[0]["predicted_hit"] == ["t1"]


def test_evolve_round_cli_rejects_edits_for_unknown_variant(tmp_path, capsys):
    run, incumbent, candidate_a, candidate_b = evolve_inputs(tmp_path)
    edits = tmp_path / "z.edits.json"
    edits.write_text("[]", encoding="utf-8")
    assert (
        main(
            evolve_argv(
                run,
                incumbent,
                candidate_a,
                candidate_b,
                tmp_path / "history.jsonl",
                tmp_path / "r0",
                "--edits",
                f"Z={edits}",
            )
        )
        == 2
    )
    assert "unknown" in capsys.readouterr().err


def test_evolve_round_cli_refuses_a_k_that_drifts_from_prereg(tmp_path, capsys):
    run, incumbent, candidate_a, candidate_b = evolve_inputs(tmp_path)
    assert (
        main(
            evolve_argv(
                run,
                incumbent,
                candidate_a,
                candidate_b,
                tmp_path / "history.jsonl",
                tmp_path / "r0",
                "--k",
                "2",
            )
        )
        == 2
    )
    assert "pre-registration" in capsys.readouterr().err


def fake_chat(monkeypatch, answers):
    prompts = []

    class Fake:
        def __init__(self, profile, **kwargs):
            self.profile = profile
            self.answers = list(answers)

        def complete(self, messages, **kwargs):
            prompts.append(messages[-1]["content"])
            if not self.answers:
                raise AssertionError("fake chat ran out of answers")
            return self.answers.pop(0)

    monkeypatch.setattr(cli_module, "ChatClient", Fake)
    return prompts


def propose_answers():
    return [
        json.dumps(
            {
                "action": "edit_file",
                "path": "agent.py",
                "old": "MAX_TURNS = 10",
                "new": "MAX_TURNS = 20",
            }
        ),
        json.dumps(
            {
                "action": "done",
                "summary": "bound retries",
                "edits": [
                    {
                        "id": "C1",
                        "component": "core",
                        "hypothesis": "cap turns so runs terminate",
                        "predicted_affected": ["t1"],
                    }
                ],
            }
        ),
    ]


def test_propose_cli_edits_in_place_and_writes_evidence(tmp_path, capsys, monkeypatch):
    root = tmp_path / "harness"
    root.mkdir()
    (root / "agent.py").write_text(
        "MAX_TURNS = 10\n\ndef run(task):\n    return task\n", encoding="utf-8"
    )
    fake_chat(monkeypatch, propose_answers())
    monkeypatch.setenv("TEST_API_KEY", "unit-test-secret")
    out = tmp_path / "out"
    assert (
        main(
            [
                "propose",
                "--root",
                str(root),
                "--out",
                str(out),
                "--budget",
                "1",
                "--components",
                "core,parser",
                "--api-key-env",
                "TEST_API_KEY",
                "--max-turns",
                "4",
            ]
        )
        == 0
    )
    payload = json.loads((out / "proposal.json").read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["n_file_edits"] >= 1
    assert payload["files"]["changed"] == ["agent.py"]
    assert payload["edits"][0]["component"] == "core"
    assert "MAX_TURNS = 20" in (root / "agent.py").read_text(encoding="utf-8")
    patch = (out / "diff.patch").read_text(encoding="utf-8")
    assert "-MAX_TURNS = 10" in patch
    assert "+MAX_TURNS = 20" in patch
    assert "unit-test-secret" not in json.dumps(payload)


def test_propose_cli_passes_history_and_evidence_to_the_prompt(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "harness"
    root.mkdir()
    (root / "agent.py").write_text("MAX_TURNS = 10\n", encoding="utf-8")
    prompts = fake_chat(monkeypatch, propose_answers())
    monkeypatch.setenv("TEST_API_KEY", "x")
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps({"component": "core", "delta_S": 0.02, "outcome": "ACCEPTED"})
        + "\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps([{"task": "t1", "score": 0.0, "cause": "timeout"}]),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert (
        main(
            [
                "propose",
                "--root",
                str(root),
                "--out",
                str(out),
                "--budget",
                "1",
                "--components",
                "core",
                "--history",
                str(history),
                "--evidence",
                str(evidence),
                "--api-key-env",
                "TEST_API_KEY",
            ]
        )
        == 0
    )
    assert "timeout" in prompts[0]
    assert '"delta_S": 0.02' in prompts[0]


@pytest.fixture
def git_harness(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"], cwd=root, check=True
    )
    (root / "agent.py").write_text("MAX_TURNS = 10\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "agent.py"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "-m",
            "one",
        ],
        check=True,
    )
    return root


def test_propose_cli_runs_in_a_worktree_and_keeps_it_on_request(
    tmp_path, capsys, monkeypatch, git_harness
):
    fake_chat(monkeypatch, propose_answers())
    monkeypatch.setenv("TEST_API_KEY", "x")
    worktree = tmp_path / "wt"
    out = tmp_path / "out"
    assert (
        main(
            [
                "propose",
                "--repo",
                str(git_harness),
                "--worktree",
                str(worktree),
                "--out",
                str(out),
                "--budget",
                "1",
                "--components",
                "core",
                "--api-key-env",
                "TEST_API_KEY",
                "--keep",
            ]
        )
        == 0
    )
    assert "MAX_TURNS = 20" in (worktree / "agent.py").read_text(encoding="utf-8")
    assert "MAX_TURNS = 10" in (git_harness / "agent.py").read_text(encoding="utf-8")
    payload = json.loads((out / "proposal.json").read_text(encoding="utf-8"))
    assert payload["root"] == str(worktree.resolve())
    head = subprocess.run(
        ["git", "-C", str(git_harness), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert payload["base"] == head


def test_propose_cli_removes_the_worktree_by_default(
    tmp_path, capsys, monkeypatch, git_harness
):
    fake_chat(monkeypatch, propose_answers())
    monkeypatch.setenv("TEST_API_KEY", "x")
    worktree = tmp_path / "wt"
    assert (
        main(
            [
                "propose",
                "--repo",
                str(git_harness),
                "--worktree",
                str(worktree),
                "--out",
                str(tmp_path / "out"),
                "--budget",
                "1",
                "--components",
                "core",
                "--api-key-env",
                "TEST_API_KEY",
            ]
        )
        == 0
    )
    assert not worktree.exists()


def test_propose_cli_requires_exactly_one_root_or_repo(tmp_path, capsys):
    assert (
        main(
            [
                "propose",
                "--out",
                str(tmp_path / "out"),
                "--budget",
                "1",
            ]
        )
        == 2
    )
    assert "exactly one" in capsys.readouterr().err


def test_propose_cli_fails_early_without_the_api_key(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "harness"
    root.mkdir()
    (root / "agent.py").write_text("MAX_TURNS = 10\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    out = tmp_path / "out"
    assert (
        main(
            [
                "propose",
                "--root",
                str(root),
                "--out",
                str(out),
                "--budget",
                "1",
                "--components",
                "core",
            ]
        )
        == 2
    )
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err
    assert not (out / "proposal.json").exists()


def diagnose_case(case_id, culprit, components=("core", "parser")):
    return {
        "case_id": case_id,
        "failure": "TimeoutError while parsing the reply",
        "components": [
            {"id": name, "description": f"{name} component", "k_lines": 10}
            for name in components
        ],
        "culprit": culprit,
    }


def test_diagnose_cli_writes_report_ready_scores(tmp_path, capsys, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            {
                "cases": [
                    diagnose_case("f01", "core"),
                    diagnose_case("f02", "parser"),
                ]
            }
        ),
        encoding="utf-8",
    )
    fake_chat(
        monkeypatch,
        [
            json.dumps({"top3": ["core", "parser"], "cause": "harness"}),
            json.dumps({"top3": ["parser"], "cause": "harness"}),
        ],
    )
    monkeypatch.setenv("TEST_API_KEY", "unit-test-secret")
    out = tmp_path / "out"
    assert (
        main(
            [
                "diagnose",
                "--cases",
                str(cases),
                "--out",
                str(out),
                "--phase",
                "after",
                "--repeats",
                "1",
                "--api-key-env",
                "TEST_API_KEY",
            ]
        )
        == 0
    )
    score = json.loads((out / "score.json").read_text(encoding="utf-8"))
    assert list(score) == ["after"]
    assert score["after"]["top1"] == 1.0
    assert score["after"]["top3"] == 1.0
    assert score["after"]["binary"] == 1.0
    details = json.loads((out / "details.json").read_text(encoding="utf-8"))
    assert details["n_cases"] == 2
    assert details["per_case"][0]["case_id"] == "f01"
    assert details["model"]["name"] == cli_module.DEEPSEEK_FLASH.name
    assert "unit-test-secret" not in json.dumps(details)


def test_diagnose_cli_rejects_a_culprit_outside_the_component_list(
    tmp_path, capsys, monkeypatch
):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([diagnose_case("f01", "missing")]), encoding="utf-8"
    )
    monkeypatch.setenv("TEST_API_KEY", "x")
    out = tmp_path / "out"
    assert (
        main(
            [
                "diagnose",
                "--cases",
                str(cases),
                "--out",
                str(out),
                "--repeats",
                "1",
                "--api-key-env",
                "TEST_API_KEY",
            ]
        )
        == 2
    )
    assert "culprit" in capsys.readouterr().err
    assert not (out / "details.json").exists()


def test_diagnose_cli_fails_early_without_the_api_key(
    tmp_path, capsys, monkeypatch
):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([diagnose_case("f01", "core")]), encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    out = tmp_path / "out"
    assert (
        main(["diagnose", "--cases", str(cases), "--out", str(out)])
        == 2
    )
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err
    assert not (out / "details.json").exists()


def test_faults_cli_writes_a_deterministic_seeded_plan(tmp_path, capsys):
    root = tmp_path / "harness"
    root.mkdir()
    (root / "agent.py").write_text(
        "def f(x):\n    if x > 1:\n        return 1\n    return 0\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    argv = [
        "faults",
        "--root",
        str(root),
        "--path",
        "agent.py",
        "--seed",
        "3",
        "--per-file",
        "2",
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    first = json.loads((out / "faults.json").read_text(encoding="utf-8"))
    assert first["seed"] == 3
    assert 1 <= len(first["mutations"]) <= 2
    assert all(m["path"] == "agent.py" for m in first["mutations"])
    assert main(argv) == 0
    second = json.loads((out / "faults.json").read_text(encoding="utf-8"))
    assert second["mutations"] == first["mutations"]


def test_faults_cli_rejects_a_path_outside_the_root(tmp_path, capsys):
    root = tmp_path / "harness"
    root.mkdir()
    (tmp_path / "evil.py").write_text("x = 1\n", encoding="utf-8")
    assert (
        main(
            [
                "faults",
                "--root",
                str(root),
                "--path",
                "../evil.py",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "path" in capsys.readouterr().err


def write_fault_trial(tmp_path, trial="fault__t1", *, reward=0.0, job="fault-a"):
    trial_dir = tmp_path / "jobs" / job / trial
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "trial_name": trial,
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"step_id": 1, "source": "user", "message": "solve it"},
                    {"step_id": 2, "source": "agent", "message": "commands failed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return trial_dir


def fault_cases_argv(tmp_path, truth, *extra, job_name="fault-a"):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    argv = [
        "fault-cases",
        "--jobs-dir",
        str(tmp_path / "jobs"),
    ]
    if job_name is not None:
        argv += ["--job-name", job_name]
    argv += [
        "--truth",
        str(truth_path),
        "--subject",
        "terminus2",
        "--subject-root",
        str(tmp_path / "subject"),
        "--out",
        str(tmp_path / "cases"),
        *extra,
    ]
    return argv


def test_fault_cases_cli_writes_cases_json(tmp_path, capsys):
    write_fault_trial(tmp_path)
    subject = tmp_path / "subject"
    write(subject, "agent.py", "x = 1\n")
    truth = {"fault__t1": "terminus2:all"}
    assert main(fault_cases_argv(tmp_path, truth)) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "cases" / "cases.json")
    payload = json.loads((tmp_path / "cases" / "cases.json").read_text(encoding="utf-8"))
    assert payload["skipped"] == []
    case = payload["cases"][0]
    assert case["case_id"] == "fault__t1"
    assert case["culprit"] == "terminus2:all"
    assert case["category"] == "harness"
    assert "commands failed" in case["failure"]


def test_fault_cases_cli_skips_a_trial_that_did_not_fail(tmp_path, capsys):
    write_fault_trial(tmp_path, reward=1.0)
    subject = tmp_path / "subject"
    write(subject, "agent.py", "x = 1\n")
    assert main(fault_cases_argv(tmp_path, {"fault__t1": "terminus2:all"})) == 2
    assert "no failure cases" in capsys.readouterr().err


def test_fault_cases_cli_uses_file_boundaries(tmp_path, capsys):
    write_fault_trial(tmp_path)
    subject = tmp_path / "subject"
    write(subject, "agent.py", "x = 1\n")
    write(subject, "mechanisms.py", "y = 2\n")
    argv = fault_cases_argv(
        tmp_path,
        {"fault__t1": "terminus2:mechanisms"},
        "--boundary",
        "agent=agent.py",
        "--boundary",
        "mechanisms=mechanisms.py",
    )
    assert main(argv) == 0
    capsys.readouterr()
    payload = json.loads((tmp_path / "cases" / "cases.json").read_text(encoding="utf-8"))
    assert payload["cases"][0]["culprit"] == "terminus2:mechanisms"
    assert {c["id"] for c in payload["cases"][0]["components"]} == {
        "terminus2:agent",
        "terminus2:mechanisms",
    }


def test_fault_cases_cli_rejects_an_unknown_culprit(tmp_path, capsys):
    write_fault_trial(tmp_path)
    write(tmp_path / "subject", "agent.py", "x = 1\n")
    assert main(fault_cases_argv(tmp_path, {"fault__t1": "terminus2:nope"})) == 2
    assert "culprit" in capsys.readouterr().err


def test_fault_cases_cli_collects_every_job(tmp_path, capsys):
    write_fault_trial(tmp_path, "fix-git__a", job="fault-a-001")
    write_fault_trial(tmp_path, "fix-git__b", job="fault-a-002")
    subject = tmp_path / "subject"
    write(subject, "agent.py", "x = 1\n")
    truth = {"fix-git__a": "terminus2:all", "fix-git__b": "terminus2:all"}
    assert main(fault_cases_argv(tmp_path, truth, job_name=None)) == 0
    capsys.readouterr()
    payload = json.loads((tmp_path / "cases" / "cases.json").read_text(encoding="utf-8"))
    assert payload["skipped"] == []
    assert {case["case_id"] for case in payload["cases"]} == {
        "fix-git__a",
        "fix-git__b",
    }


def make_fault_plan(tmp_path, capsys):
    subject = tmp_path / "vendor" / "terminus2"
    write(subject, "agent.py", "FLAG = True\n")
    plan = tmp_path / "plan"
    assert (
        main(
            [
                "faults",
                "--root",
                str(subject),
                "--path",
                "agent.py",
                "--seed",
                "0",
                "--per-file",
                "1",
                "--out",
                str(plan),
            ]
        )
        == 0
    )
    capsys.readouterr()
    return subject, plan / "faults.json"


def fault_run_argv(tmp_path, subject, faults, *extra):
    return [
        "fault-run",
        "--faults",
        str(faults),
        "--subject-root",
        str(subject),
        "--tasks",
        "terminal-bench/fix-git",
        "--work",
        str(tmp_path / "work"),
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--job-prefix",
        "fault-a",
        "--log-dir",
        str(tmp_path / "logs"),
        "--harbor-src",
        str(tmp_path / "harbor" / "src"),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def test_fault_run_dry_run_prints_the_mutated_jobs(tmp_path, capsys):
    subject, faults = make_fault_plan(tmp_path, capsys)
    assert main(fault_run_argv(tmp_path, subject, faults, "--dry-run")) == 0
    payload = output_json(capsys)
    assert len(payload["jobs"]) == 1
    job = payload["jobs"][0]
    assert job["job_name"] == "fault-a-001"
    assert job["component"] == "terminus2:all"
    assert job["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
        (tmp_path / "work" / "001").resolve()
    )
    assert job["argv"][job["argv"].index("-k") + 1] == "1"
    assert not (tmp_path / "work").exists()


def test_fault_run_applies_the_mutation_and_records_truth(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    subject, faults = make_fault_plan(tmp_path, capsys)

    def fake_run(argv, *, log_path, env=None, cwd=None, timeout=None):
        trial = tmp_path / "jobs" / "fault-a-001" / "fix-git__t1"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "terminal-bench/fix-git",
                    "trial_name": "fix-git__t1",
                    "agent_result": {},
                    "verifier_result": {"rewards": {"reward": 0.0}},
                    "exception_info": None,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(cli_module, "run_harbor", fake_run)
    assert main(fault_run_argv(tmp_path, subject, faults)) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "out" / "truth.json")
    mutated = (
        tmp_path / "work" / "001" / "terminus2" / "agent.py"
    ).read_text(encoding="utf-8")
    assert "FLAG = False" in mutated
    truth = json.loads((tmp_path / "out" / "truth.json").read_text(encoding="utf-8"))
    assert truth == {"fix-git__t1": "terminus2:all"}
    summary = json.loads(
        (tmp_path / "out" / "fault-runs.json").read_text(encoding="utf-8")
    )
    assert summary["jobs"][0]["trials"][0]["reward"] == 0.0


def test_fault_run_rejects_a_changed_source(tmp_path, capsys):
    subject, faults = make_fault_plan(tmp_path, capsys)
    write(subject, "agent.py", "FLAG = False\n")
    assert main(fault_run_argv(tmp_path, subject, faults, "--dry-run")) == 2
    assert "changed" in capsys.readouterr().err


def test_baseline_keeps_the_task_list_alongside_the_rows(tmp_path, capsys):
    write(tmp_path, "a.py", "x = 1\n")
    outcomes = _outcomes_file(
        tmp_path,
        [
            {"task_id": "t0", "trial_id": "r0", "reward": 1.0},
            {"task_id": "t1", "trial_id": "r1", "reward": 0.0},
        ],
    )
    assert (
        main(
            [
                "baseline",
                "--outcomes",
                str(outcomes),
                "--tasks",
                "t0,t1",
                "--reps",
                "1",
                "--root",
                str(tmp_path),
                "--subject",
                "demo",
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    payload = json.loads((tmp_path / "run" / "baseline.json").read_text(encoding="utf-8"))
    assert payload["task_ids"] == ["t0", "t1"]
    assert payload["tasks"]["t0"]["trials"] == 1


def test_state_init_and_show(tmp_path, capsys):
    root = tmp_path / "run"
    assert main(["state", "init", "--root", str(root), "--run-id", "r1"]) == 0
    assert capsys.readouterr().out.strip() == str(root / "state.json")
    assert main(["state", "show", "--root", str(root), "--json"]) == 0
    payload = output_json(capsys)
    assert payload["run_id"] == "r1"
    assert payload["phase"] == "created"
    assert payload["resume"] is None


def test_state_advance_requires_an_artifact(tmp_path, capsys):
    root = tmp_path / "run"
    assert main(["state", "init", "--root", str(root), "--run-id", "r1"]) == 0
    capsys.readouterr()
    assert main(["state", "advance", "--root", str(root), "--to", "prestudy_pending"]) == 0
    assert capsys.readouterr().out.strip() == str(root / "state.json")
    assert main(["state", "advance", "--root", str(root), "--to", "prestudy_running"]) == 0
    assert main(["state", "advance", "--root", str(root), "--to", "prestudy_done"]) == 2
    assert "artifact" in capsys.readouterr().err
    write(root, "prestudy.json", "{}")
    assert (
        main(
            [
                "state",
                "advance",
                "--root",
                str(root),
                "--to",
                "prestudy_done",
                "--artifact",
                "prestudy.json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["to"] == "prestudy_done"
    assert events[-1]["artifact"] == "prestudy.json"


def test_state_blocked_subject_requires_a_human(tmp_path, capsys):
    root = tmp_path / "run"
    assert main(["state", "init", "--root", str(root), "--run-id", "r1"]) == 0
    assert main(["state", "advance", "--root", str(root), "--to", "prestudy_pending"]) == 0
    assert main(["state", "advance", "--root", str(root), "--to", "blocked_subject"]) == 0
    assert main(["state", "advance", "--root", str(root), "--to", "prestudy_pending"]) == 2
    assert "human" in capsys.readouterr().err
    assert (
        main(
            [
                "state",
                "advance",
                "--root",
                str(root),
                "--to",
                "prestudy_pending",
                "--actor",
                "human",
            ]
        )
        == 0
    )


def test_state_show_reports_a_missing_state(tmp_path, capsys):
    assert main(["state", "show", "--root", str(tmp_path / "run")]) == 2
    assert "state.json" in capsys.readouterr().err


def test_power_reports_the_design_gap(tmp_path, capsys):
    reference = tmp_path / "ref.jsonl"
    candidate = tmp_path / "cand.jsonl"
    tasks = ("t1", "t2", "t3", "t4")
    rewards = (0.5, 0.6, 0.5, 0.4)
    for path, base in ((reference, 0.5), (candidate, None)):
        rows = []
        for index, task in enumerate(tasks):
            rows.append(
                json.dumps(
                    {
                        "task_id": task,
                        "trial_id": f"{task}-0",
                        "status": "completed",
                        "reward": base if base is not None else rewards[index],
                        "tokens_in": 90,
                        "tokens_out": 10,
                        "cost_usd": 0.001,
                    }
                )
            )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "power"
    assert (
        main(
            [
                "power",
                "--reference", str(reference),
                "--candidate", str(candidate),
                "--tasks", ",".join(tasks),
                "--k", "1",
                "--margin", "0.05",
                "--out", str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "power.json").read_text(encoding="utf-8"))
    assert payload["n_pairs"] == 4
    assert payload["enough"] is False
    assert payload["required_pairs"] > payload["n_pairs"]


def test_static_scored_candidates_finds_dead_private_function(tmp_path):
    root = tmp_path / "terminus2"
    write(
        root,
        "mod.py",
        "def live():\n    return 1\n\n\ndef _dead():\n    return 2\n",
    )

    rows = cli_module._static_scored_candidates(root)

    assert [row.candidate.symbol for row in rows] == ["_dead"]
    assert rows[0].delta.k_ast > 0


def test_coverage_options_point_at_the_traced_agent(tmp_path):
    agent, env, project_root = cli_module._coverage_options(
        tmp_path / "traces", tmp_path / "vendor" / "terminus2"
    )

    assert agent == "evo.adapters.traced_terminus2:TracedAgentHarness"
    assert env["EVO_TRACE_DIR"] == str((tmp_path / "traces").resolve())
    assert env["EVO_SUBJECT_ROOT"] == str(
        (tmp_path / "vendor" / "terminus2").resolve()
    )
    assert (tmp_path / "traces").is_dir()
    assert project_root
