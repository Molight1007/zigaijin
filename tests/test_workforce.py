from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evo.application.baseline import TrialOutcome, summarize_trials
from evo.application.workforce import (
    _provisional_decision,
    _round_history,
    WorkforceRuntime,
    WorkforceSpec,
    analyze_evidence,
    default_runtime,
    execute_deletion_plan,
    execute_deepseek_plan,
    execute_edit_plan,
    find_task_leakage,
    judge_execution,
    map_architecture,
    monitor_tests,
    repair_edit_plan,
    review_semantics,
    run_workforce_loop,
    run_workforce_round,
    screen_execution,
    summarize_deletion_plan,
    summarize_deletion_plan_llm,
    summarize_direction,
    summarize_edit_plan,
)
from evo.contracts.types import KVector
from evo.domain.verify import CheckOutcome, aggregate
from evo.domain.workforce import ExecutionResult, PlanResult, ROLE_SPECS, WorkforceError


def _outcome(task_id: str, trial_id: str, reward: float) -> TrialOutcome:
    return TrialOutcome(
        task_id=task_id,
        trial_id=trial_id,
        reward=reward,
        tokens_in=10,
        tokens_out=5,
    )


def test_analyze_evidence_parses_the_three_lenses():
    evidence = {
        "tasks": ["t1"],
        "per_task": {"t1": {"mean_reward": 0.0}},
        "failures": [{"task_id": "t1", "mean_reward": 0.0}],
        "traces": {},
    }

    def model(prompt: str, system: str) -> str:
        assert "failure_modes" in prompt
        return json.dumps(
            {
                "failure_modes": [{"mode": "fixed_wait", "n_tasks": 1}],
                "capability_gaps": [],
                "success_habits": [{"habit": "poll_builds"}],
            }
        )

    analysis = analyze_evidence(evidence, model=model)

    assert analysis["failure_modes"][0]["mode"] == "fixed_wait"
    assert analysis["success_habits"][0]["habit"] == "poll_builds"


def test_analyze_evidence_uses_a_digester_when_requested(tmp_path: Path):
    trace = tmp_path / "t1.txt"
    trace.write_text("step 1: command timed out after fixed wait", encoding="utf-8")
    evidence = {
        "tasks": ["t1"],
        "per_task": {"t1": {"mean_reward": 0.0}},
        "failures": [{"task_id": "t1", "mean_reward": 0.0}],
        "traces": {"t1": str(trace)},
    }

    def model(prompt: str, system: str) -> str:
        if "read-only trajectory digester" in system:
            return json.dumps(
                {
                    "task_id": "t1",
                    "lens": "failure",
                    "summary": "fixed wait on a long command",
                    "failure_mechanism": "fixed_wait",
                    "anchors": ["step 1"],
                    "needed_instead": "poll the command",
                }
            )
        return json.dumps(
            {"failure_modes": [], "capability_gaps": [], "success_habits": []}
        )

    analysis = analyze_evidence(evidence, model=model, max_digests=1)

    assert analysis["digests"][0]["summary"] == "fixed wait on a long command"


def test_workforce_spec_rejects_h0():
    with pytest.raises(WorkforceError, match="baseline"):
        WorkforceSpec(
            run_id="r1",
            round=0,
            arm="H0",
            mode="delete",
            tasks=("t1",),
        )


def test_workforce_spec_rejects_placebo_static_mode():
    with pytest.raises(WorkforceError, match="random"):
        WorkforceSpec(
            run_id="r1",
            round=0,
            arm="PLACEBO",
            mode="delete",
            batch_mode="static",
            target_k_ast=100,
            tasks=("t1",),
        )


def test_llm_summarizer_selects_a_candidate_subset(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    candidate_id = architecture["candidates"][0]["id"]
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
    )

    def model(prompt: str, system: str) -> str:
        assert candidate_id in prompt
        return json.dumps(
            {
                "selected_ids": [candidate_id],
                "rationale": "keep the highest-K removable unit",
                "problems": [{"kind": "failure", "task_id": "t1"}],
            }
        )

    plan = summarize_deletion_plan_llm(architecture, spec, {}, model)

    assert plan.mode == "delete"
    assert len(plan.batch) == 1
    assert plan.batch[0]["id"] == candidate_id
    assert "llm:" in plan.summary


def test_llm_summarizer_falls_back_on_an_unknown_id(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
    )

    def model(prompt: str, system: str) -> str:
        return json.dumps({"selected_ids": ["not-a-real-id"], "rationale": "bad"})

    plan = summarize_deletion_plan_llm(architecture, spec, {}, model)

    assert plan.batch
    assert plan.candidate_id.startswith("llm-fallback")


def test_default_runtime_wires_the_llm_summarizer(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    candidate_id = architecture["candidates"][0]["id"]
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
    )

    def model(prompt: str, system: str) -> str:
        return json.dumps({"selected_ids": [candidate_id], "rationale": "pick one"})

    runtime = default_runtime(
        evaluate=lambda root, tasks, k: [],
        model=model,
        llm_summarizer=True,
    )
    plan = runtime.summarize(subject, {}, architecture, tmp_path / "work", spec)

    assert plan.candidate_id.startswith("llm-batch")


def test_summarize_direction_returns_a_brief_without_editing(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="COMBO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
    )

    def model(prompt: str, system: str) -> str:
        return json.dumps(
            {
                "problem_summary": "the agent guesses fixed waits",
                "directions": [
                    {
                        "direction": "poll long commands",
                        "target_components": ["demo:all"],
                        "expected_effect": "fewer timeouts",
                        "risk": "longer wall clock",
                    }
                ],
                "task_brief": "replace the fixed wait with polling",
                "constraints": ["no task-specific data"],
            }
        )

    plan = summarize_direction({}, architecture, spec, model)

    assert plan.brief == "replace the fixed wait with polling"
    assert plan.directions[0]["direction"] == "poll long commands"
    assert plan.staged_root is None
    assert "def _dead" in (subject / "pkg" / "a.py").read_text(encoding="utf-8")


def test_execute_deepseek_plan_edits_the_harness(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="COMBO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
        rounds=2,
    )
    plan = PlanResult(
        mode="edit",
        candidate_id="brief-1",
        layer="L2",
        summary="brief",
        brief="change the live return",
    )
    calls = {"n": 0}

    def model(prompt: str, system: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "edit_file",
                    "path": "pkg/a.py",
                    "old": "return 2",
                    "new": "return 3",
                }
            )
        return json.dumps(
            {
                "action": "done",
                "summary": "changed the live return",
                "edits": [
                    {
                        "id": "C1",
                        "component": "demo:all",
                        "hypothesis": "exercise the executor",
                        "targets_mode": "test",
                        "predicted_affected": [],
                        "regression_risk": "none",
                    }
                ],
            }
        )

    execution = execute_deepseek_plan(
        subject,
        plan,
        tmp_path / "exec",
        spec,
        {},
        architecture,
        model,
    )

    assert "return 3" in execution.diff
    assert "return 3" in (execution.root / "pkg" / "a.py").read_text(encoding="utf-8")
    assert execution.components == ("demo:all",)


def test_semantic_reviewer_returns_revise_suggestions(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="brief-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        components=("pkg",),
    )
    plan = PlanResult(
        mode="edit",
        candidate_id="brief-1",
        layer="L2",
        summary="brief",
        brief="improve the harness",
    )

    def model(prompt: str, system: str) -> str:
        return json.dumps(
            {
                "verdict": "revise",
                "reasons": ["incomplete fix"],
                "risk_notes": [],
                "suggestions": ["also handle parse errors"],
            }
        )

    review = review_semantics(execution, plan, {}, {}, _spec(), model)

    assert review["verdict"] == "revise"
    assert review["suggestions"] == ["also handle parse errors"]


def test_semantic_reviewer_rejects_deterministic_leak_without_calling_model(
    tmp_path: Path,
):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="leak-1",
        root=candidate_root,
        changed=("a.py",),
        diff="+api_key = 'sk-abcdefghijklmnopqrstuvwx'\n",
        components=("pkg",),
    )
    plan = PlanResult(
        mode="edit",
        candidate_id="leak-1",
        layer="L2",
        summary="leak",
        brief="improve the harness",
    )
    called: list[str] = []

    def model(prompt: str, system: str) -> str:
        called.append(prompt)
        return json.dumps({"verdict": "accept", "reasons": [], "risk_notes": []})

    review = review_semantics(execution, plan, {}, {}, _spec(), model)

    assert review["verdict"] == "reject"
    assert any("precheck" in reason for reason in review["reasons"])
    assert called == []


def test_semantic_review_rejects_oversized_diff_without_model(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="big", root=candidate_root, changed=("a.py",),
        diff="+x\n" * 200_000,
    )
    plan = PlanResult(mode="edit", candidate_id="big", layer="L2",
                      summary="big", brief="x")
    called: list[str] = []

    def model(prompt: str, system: str) -> str:
        called.append(prompt)
        return '{"verdict":"accept"}'

    spec = replace(_spec(), max_semantic_diff_chars=1000)
    review = review_semantics(execution, plan, {}, {}, spec, model)

    assert review["verdict"] == "reject"
    assert "diff too large" in " ".join(review["reasons"])
    assert called == []


def test_semantic_review_marks_provider_error_as_unknown(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="err", root=candidate_root, changed=("a.py",),
        diff="+x = 1\n",
    )
    plan = PlanResult(mode="edit", candidate_id="err", layer="L2",
                      summary="err", brief="x")

    def model(prompt: str, system: str) -> str:
        raise RuntimeError("timeout")

    review = review_semantics(execution, plan, {}, {}, _spec(), model)

    assert review["verdict"] == "unknown"
    assert "timeout" in " ".join(review["reasons"])


def test_semantic_review_calls_model_at_exactly_the_diff_limit(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="edge", root=candidate_root, changed=("a.py",),
        diff="+123456789",
    )
    plan = PlanResult(mode="edit", candidate_id="edge", layer="L2",
                      summary="edge", brief="x")
    called: list[str] = []

    def model(prompt: str, system: str) -> str:
        called.append(prompt)
        return '{"verdict":"accept","reasons":[]}'

    review = review_semantics(
        execution, plan, {}, {}, replace(_spec(), max_semantic_diff_chars=10), model
    )

    assert review["verdict"] == "accept"
    assert len(called) == 1


def test_default_runtime_wires_the_deepseek_native_chain(tmp_path: Path):
    subject = _subject(tmp_path)
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="COMBO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
        rounds=2,
    )
    edits = {"n": 0}

    def model(prompt: str, system: str) -> str:
        if "architecture advisor" in system:
            return json.dumps(
                {
                    "summary": "the live path has one removable helper",
                    "hotspots": [{"module": "demo:all", "why": "small surface"}],
                    "opportunities": [{"target": "demo:all", "change": "inline"}],
                    "risks": ["keep imports valid"],
                }
            )
        if "summarizer of a harness-improvement" in system:
            return json.dumps(
                {
                    "problem_summary": "simplify the live path",
                    "directions": [{"direction": "change the return", "target_components": ["demo:all"]}],
                    "task_brief": "change the live return",
                    "constraints": [],
                }
            )
        edits["n"] += 1
        if edits["n"] == 1:
            return json.dumps(
                {
                    "action": "edit_file",
                    "path": "pkg/a.py",
                    "old": "return 2",
                    "new": "return 3",
                }
            )
        return json.dumps(
            {
                "action": "done",
                "summary": "executor changed the return",
                "edits": [
                    {
                        "id": "C1",
                        "component": "demo:all",
                        "hypothesis": "exercise the native chain",
                        "targets_mode": "test",
                        "predicted_affected": [],
                        "regression_risk": "none",
                    }
                ],
            }
        )

    runtime = default_runtime(
        evaluate=lambda root, tasks, k: [],
        model=model,
        deepseek_native=True,
    )
    architecture = runtime.architect(subject, "demo")
    assert architecture["advice"]["summary"]
    plan = runtime.summarize(subject, {}, architecture, tmp_path / "brief", spec)
    assert plan.brief == "change the live return"
    execution = runtime.execute(
        subject, plan, tmp_path / "exec", spec, {}, architecture
    )

    assert "return 3" in execution.diff
    assert execution.components == ("demo:all",)


def test_workforce_has_five_employees():
    assert [spec.name for spec in ROLE_SPECS] == [
        "monitor",
        "architect",
        "summarizer",
        "executor",
        "reviewer",
    ]
    by_name = {spec.name: spec for spec in ROLE_SPECS}
    assert by_name["monitor"].depends_on == ()
    assert by_name["architect"].depends_on == ()
    assert by_name["summarizer"].depends_on == ("monitor", "architect")
    assert by_name["executor"].depends_on == ("summarizer",)
    assert by_name["reviewer"].depends_on == ("executor",)


def test_monitor_flags_low_scoring_tasks():
    evidence = monitor_tests(
        [
            _outcome("t1", "t1-a", 1.0),
            _outcome("t2", "t2-a", 0.0),
            _outcome("t2", "t2-b", 0.5),
        ],
        tasks=("t1", "t2"),
        k=2,
    )

    assert evidence["per_task"]["t2"]["mean_reward"] == 0.25
    assert evidence["failures"][0]["task_id"] == "t2"
    assert evidence["summary"]["mean_reward"] == 0.375


def test_find_task_leakage_flags_task_ids_and_tests():
    assert find_task_leakage("if task == 'fix-git':", ("fix-git",)) == ("fix-git",)
    assert find_task_leakage("open('/tests/test_x.py')", ()) == ("/tests/",)


def _subject(tmp_path: Path) -> Path:
    subject = tmp_path / "subject"
    package = subject / "pkg"
    package.mkdir(parents=True)
    (package / "a.py").write_text(
        "def _dead():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def _live():\n"
        "    return 2\n"
        "\n"
        "\n"
        "_live()\n",
        encoding="utf-8",
    )
    return subject


def test_architect_maps_components_and_candidates(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")

    assert architecture["subject"] == "demo"
    assert architecture["totals"]["k_lines_nb"] > 0
    assert architecture["components"]
    assert any(row["path"] == "pkg/a.py" for row in architecture["candidates"])


def test_summarizer_selects_a_plan_without_touching_the_subject(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
    )

    plan = summarize_deletion_plan(architecture, spec)

    assert plan.mode == "delete"
    assert plan.batch
    assert "def _dead" in (subject / "pkg" / "a.py").read_text(encoding="utf-8")


def test_executor_applies_the_plan_and_reports_diff(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
    )
    plan = summarize_deletion_plan(architecture, spec)

    execution = execute_deletion_plan(subject, plan, tmp_path / "work", spec)

    text = (execution.root / "pkg" / "a.py").read_text(encoding="utf-8")
    assert "def _dead" not in text
    assert "def _live" in text
    assert execution.changed == ("pkg/a.py",)
    assert "def _dead" in execution.diff


def test_summarizer_builds_an_edit_plan_with_a_model(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="EVO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
        rounds=2,
    )
    calls: list[str] = []

    def model(prompt: str, system: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(
                {
                    "action": "edit_file",
                    "path": "pkg/a.py",
                    "old": "return 2",
                    "new": "return 3",
                }
            )
        return json.dumps(
            {
                "action": "done",
                "summary": "change the live return",
                "edits": [
                    {
                        "id": "C1",
                        "component": "demo:all",
                        "hypothesis": "exercise the edit path",
                        "targets_mode": "test",
                        "predicted_affected": [],
                        "regression_risk": "none",
                    }
                ],
            }
        )

    plan = summarize_edit_plan(subject, {}, architecture, tmp_path / "work", spec, model)

    assert plan.mode == "edit"
    assert plan.staged_root is not None
    assert "return 3" in plan.diff
    assert "return 2" in (subject / "pkg" / "a.py").read_text(encoding="utf-8")


def test_executor_materializes_an_edit_plan(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="EVO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
        rounds=2,
    )
    calls: list[str] = []

    def model(prompt: str, system: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(
                {
                    "action": "edit_file",
                    "path": "pkg/a.py",
                    "old": "return 2",
                    "new": "return 3",
                }
            )
        return json.dumps(
            {
                "action": "done",
                "summary": "change the live return",
                "edits": [
                    {
                        "id": "C1",
                        "component": "demo:all",
                        "hypothesis": "exercise the edit path",
                        "targets_mode": "test",
                        "predicted_affected": [],
                        "regression_risk": "none",
                    }
                ],
            }
        )

    plan = summarize_edit_plan(subject, {}, architecture, tmp_path / "plan", spec, model)
    execution = execute_edit_plan(subject, plan, tmp_path / "work", spec)

    text = (execution.root / "pkg" / "a.py").read_text(encoding="utf-8")
    assert "return 3" in text
    assert "return 3" in execution.diff
    assert execution.changed == ("pkg/a.py",)


def test_repair_edit_plan_reapplies_the_model_on_the_staged_root(tmp_path: Path):
    subject = _subject(tmp_path)
    architecture = map_architecture(subject, "demo")
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="EVO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
        rounds=2,
    )

    calls = {"n": 0}

    def first_model(prompt: str, system: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "action": "edit_file",
                    "path": "pkg/a.py",
                    "old": "return 2",
                    "new": "return 3",
                }
            )
        if calls["n"] == 3:
            return json.dumps(
                {
                    "action": "edit_file",
                    "path": "pkg/a.py",
                    "old": "return 3",
                    "new": "return 4",
                }
            )
        return json.dumps(
            {
                "action": "done",
                "summary": "edit",
                "edits": [
                    {
                        "id": "C1",
                        "component": "demo:all",
                        "hypothesis": "edit",
                        "targets_mode": "test",
                        "predicted_affected": [],
                        "regression_risk": "none",
                    }
                ],
            }
        )

    plan = summarize_edit_plan(subject, {}, architecture, tmp_path / "plan", spec, first_model)
    repaired = repair_edit_plan(
        subject,
        plan,
        {"reasons": ["undeclared bundling"], "risk_notes": []},
        {},
        architecture,
        tmp_path / "repair",
        spec,
        first_model,
    )

    assert "return 4" in repaired.diff
    assert "return 4" in (repaired.staged_root / "pkg" / "a.py").read_text(encoding="utf-8")


def test_runner_repairs_a_semantic_rejection_then_evaluates(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "a.py").write_text("x = 1\n", encoding="utf-8")
    semantic_calls = {"n": 0}

    def summarize(subject, evidence, architecture, work, spec):
        calls.append("summarize")
        return PlanResult(
            mode="edit",
            candidate_id="edit-1",
            layer="L2",
            summary="test edit",
            diff="-x = 0\n+x = 1\n",
            staged_root=staged,
        )

    def repair(subject, plan, critic, evidence, architecture, work, spec):
        calls.append("repair")
        return PlanResult(
            mode="edit",
            candidate_id="repair-1",
            layer="L2",
            summary="repaired",
            diff="-x = 0\n+x = 2\n",
            staged_root=staged,
        )

    def semantic(execution, plan, evidence, architecture, spec):
        calls.append("semantic")
        semantic_calls["n"] += 1
        if semantic_calls["n"] == 1:
            return {"verdict": "reject", "reasons": ["leak"], "risk_notes": []}
        return {"verdict": "accept", "reasons": [], "risk_notes": []}

    runtime = replace(runtime, summarize=summarize, repair=repair, semantic=semantic)
    spec = replace(
        _spec(),
        mode="edit",
        semantic_review=True,
        repair_rounds=1,
    )

    payload = run_workforce_round(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert "repair" in calls
    assert "evaluate" in calls
    assert payload["review"]["verdict"] == "delete"
    assert (tmp_path / "out" / "repair.json").exists()


def test_workforce_loop_runs_two_rounds_and_writes_loop_json(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)
    spec = replace(_spec(), rounds=2)

    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "loop-work",
        out_dir=tmp_path / "loop-out",
        rounds=2,
    )

    assert len(payload["rounds"]) == 2
    assert all(row["accepted"] for row in payload["rounds"])
    assert payload["stopped"] == "rounds"
    assert payload["budget"]["trials_used"] == 2
    assert calls.count("evaluate") == 2
    assert (tmp_path / "loop-out" / "loop.json").exists()
    assert (tmp_path / "loop-out" / "round-00" / "round.json").exists()
    assert (tmp_path / "loop-out" / "round-01" / "round.json").exists()


def test_loop_selects_one_coreset_and_reuses_it(tmp_path: Path):
    tasks = ("t1", "t2", "t3", "t4")
    reference = [_outcome(task, f"r-{task}", 1.0) for task in tasks]
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls, reference=reference)
    seen: list[tuple[str, ...]] = []
    base_monitor = runtime.monitor

    def monitor(root, records, current_tasks, k):
        seen.append(tuple(current_tasks))
        return base_monitor(root, records, current_tasks, k)

    def evaluate(root, current_tasks, k):
        wanted = set(current_tasks)
        return [row for row in reference if row.task_id in wanted]

    runtime = replace(runtime, monitor=monitor, evaluate=evaluate)
    spec = replace(
        _spec(),
        tasks=tasks,
        coreset_k=2,
        coreset_theta=0.7,
        rounds=2,
        variants=1,
    )
    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "coreset-work",
        out_dir=tmp_path / "coreset-out",
    )

    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert len(seen[0]) == 2
    assert (tmp_path / "coreset-out" / "coreset.json").exists()
    assert payload["spec"]["tasks"] == list(seen[0])
    assert payload["spec"]["coreset_k"] == 0
    # the requested k and the actual pick stay auditable in loop.json
    assert payload["requested_coreset_k"] == 2
    assert payload["coreset"]["picks"] == list(seen[0])


def test_workforce_loop_runs_multiple_variants(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)
    spec = replace(_spec(), rounds=1, variants=2)

    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "variant-work",
        out_dir=tmp_path / "variant-out",
        rounds=1,
    )

    report = payload["rounds"][0]
    assert len(report["variants"]) == 2
    assert report["winner"] == "A"
    assert calls.count("evaluate") == 2


def test_loop_stops_at_trial_budget(tmp_path: Path):
    tasks = ("t1",)
    reference = [_outcome("t1", "r-t1", 1.0)]
    calls: list[str] = []
    runtime, _ = _runtime(tmp_path, calls, reference=reference)
    spec = replace(
        _spec(), tasks=tasks, rounds=5, variants=1, max_total_trials=1
    )
    payload = run_workforce_loop(
        spec, runtime, subject_root=tmp_path / "subject",
        reference=reference, work_dir=tmp_path / "work", out_dir=tmp_path / "out",
    )
    assert payload["stopped"] == "trial_budget"
    assert len(payload["rounds"]) <= 1
    assert payload["budget"]["trials_used"] == 1


def test_loop_stops_at_cost_budget_with_round_granularity(tmp_path: Path):
    reference = [_outcome("t1", "r-t1", 1.0)]
    calls: list[str] = []

    def evaluate(root, tasks, k):
        calls.append("evaluate")
        return [replace(row, cost_usd=0.4) for row in reference]

    runtime, _ = _runtime(tmp_path, calls, reference=reference)
    runtime = replace(runtime, evaluate=evaluate)
    spec = replace(_spec(), tasks=("t1",), rounds=5, max_cost_usd=1.0)

    payload = run_workforce_loop(
        spec, runtime, subject_root=tmp_path / "subject",
        reference=reference, work_dir=tmp_path / "work", out_dir=tmp_path / "out",
    )

    # 0.4 USD per round: the gate fires only after crossing 1.0, so 3 rounds
    assert payload["stopped"] == "cost_budget"
    assert len(payload["rounds"]) == 3
    assert payload["budget"]["trials_used"] == 3
    assert payload["budget"]["cost_used"] == pytest.approx(1.2)
    assert payload["budget"]["cost_complete"] is True
    assert payload["spec"]["max_cost_usd"] == 1.0
    assert payload["spec"]["max_total_trials"] is None


def test_loop_fails_closed_when_costs_are_missing(tmp_path: Path):
    reference = [_outcome("t1", "r-t1", 1.0)]
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls, reference=reference)
    # _outcome leaves cost_usd unset, so every evaluation has an unknown cost
    spec = replace(_spec(), tasks=("t1",), rounds=5, max_cost_usd=0.01)

    payload = run_workforce_loop(
        spec, runtime, subject_root=tmp_path / "subject",
        reference=reference, work_dir=tmp_path / "work", out_dir=tmp_path / "out",
    )

    # missing cost must not read as 0.0, or the 0.01 budget never fires
    assert payload["stopped"] == "cost_unknown"
    assert payload["budget"]["cost_complete"] is False
    assert len(payload["rounds"]) == 1


def test_round_history_uses_variant_score_and_prediction(tmp_path: Path):
    reference = [_outcome("t1", "r-t1", 0.0), _outcome("t2", "r-t2", 1.0)]
    outcomes = [_outcome("t1", "c-t1", 1.0), _outcome("t2", "c-t2", 0.0)]
    spec = replace(_spec(), tasks=("t1", "t2"), k=1, delta=0.01)
    payload = {
        "plan": {"edits": [{"id": "C1", "component": "core",
                            "hypothesis": "x", "predicted_affected": ["t1", "t2"]}]},
        "review": {"selection": {"score": 0.5, "delta_score": 0.0}},
        "outcomes": [asdict(row) for row in outcomes],
    }
    rows = _round_history(
        spec, payload, accepted=False, score=0.9,
        variant="A", reference=reference,
    )
    assert rows[0]["score"] == 0.5
    assert rows[0]["prediction_verified"] is False
    assert rows[0]["prediction_falsified"] is True
    assert rows[0]["predicted_hit"] == ["t1"]
    assert rows[0]["predicted_miss"] == ["t2"]


def test_loop_history_verifies_predictions_against_the_round_incumbent(tmp_path: Path):
    reference = [_outcome("t1", "ref-t1", 1.0)]
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls, reference=reference)
    runtime = replace(
        runtime,
        summarize=lambda subject, evidence, architecture, work, spec: PlanResult(
            mode="delete",
            candidate_id="c1",
            layer="L2",
            summary="test plan",
            batch=({"id": "c1", "component": "core"},),
        ),
        evaluate=lambda root, tasks, k: [
            TrialOutcome(task_id=task, trial_id=f"cand-{task}", reward=0.0)
            for task in tasks
        ],
    )
    spec = replace(_spec(), rounds=1)
    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "history-work",
        out_dir=tmp_path / "history-out",
        rounds=1,
    )

    row = payload["history"][0]
    assert row["predicted_affected"] == []
    assert row["unpredicted_regressions"] == ["t1"]


def test_final_holdout_rechecks_h0_in_the_same_window(tmp_path: Path):
    tasks = ("t1",)
    holdout = ("h1",)
    reference = [_outcome("t1", "r-t1", 1.0), _outcome("h1", "r-h1", 1.0)]

    def evaluate(root, current_tasks, k):
        # H0 的新窗口分数是 0；旧 reference 是 1。若仍用旧 reference，会误判失败。
        reward = 0.0 if "baseline" in str(root) else 0.0
        return [
            TrialOutcome(task_id=task, trial_id=f"{root.name}-{task}", reward=reward)
            for task in current_tasks
        ]

    runtime, _ = _runtime(tmp_path, [], reference=reference)
    runtime = replace(runtime, evaluate=evaluate)
    spec = replace(
        _spec(),
        tasks=tasks,
        holdout_tasks=holdout,
        rounds=1,
        require_holdout=True,
        recheck_baseline=True,
    )
    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert payload["final"]["holdout"]["non_inferior"] is True
    assert payload["final"]["holdout"]["baseline"] == "same_window"
    assert payload["final"]["rolled_back"] is False
    assert payload["final"]["subject"] == payload["final"]["candidate_subject"]
    # final score follows the surviving subject, not the other arm
    assert payload["final"]["score"] == payload["final"]["candidate_score"]
    assert (tmp_path / "work" / "baseline" / "subject" / "a.py").exists()


def test_final_holdout_failure_rolls_back_to_h0(tmp_path: Path):
    tasks = ("t1",)
    holdout = ("h1",)
    reference = [_outcome("t1", "r-t1", 1.0), _outcome("h1", "r-h1", 1.0)]

    def evaluate(root, current_tasks, k):
        reward = 1.0 if "baseline" in str(root) else 0.0
        return [
            TrialOutcome(task_id=task, trial_id=f"{root.name}-{task}", reward=reward)
            for task in current_tasks
        ]

    runtime, _ = _runtime(tmp_path, [], reference=reference)
    runtime = replace(runtime, evaluate=evaluate)
    spec = replace(
        _spec(),
        tasks=tasks,
        holdout_tasks=holdout,
        rounds=1,
        require_holdout=True,
        recheck_baseline=True,
    )
    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert payload["final"]["holdout"]["non_inferior"] is False
    assert payload["final"]["rolled_back"] is True
    assert payload["final"]["subject"] == str(tmp_path / "work" / "baseline" / "subject")
    assert payload["final"]["candidate_subject"] != payload["final"]["subject"]
    assert payload["final"]["verdict"] == "rolled_back"
    # the final score must be the subject's (baseline) value, not the candidate's
    assert payload["final"]["score"] == payload["final"]["baseline_score"]
    assert payload["final"]["candidate_score"] != payload["final"]["baseline_score"]


def test_rollback_reports_the_baseline_score_not_the_candidate(tmp_path: Path):
    tasks = ("t1",)
    holdout = ("h1",)
    reference = [_outcome("t1", "r-t1", 0.0), _outcome("h1", "r-h1", 1.0)]

    def evaluate(root, current_tasks, k):
        is_baseline = "baseline" in str(root)
        if "h1" in current_tasks:
            reward = 1.0 if is_baseline else 0.0
        else:
            reward = 0.0 if is_baseline else 1.0
        return [
            TrialOutcome(task_id=task, trial_id=f"{root.name}-{task}", reward=reward)
            for task in current_tasks
        ]

    runtime, _ = _runtime(tmp_path, [], reference=reference)
    runtime = replace(runtime, evaluate=evaluate)
    spec = replace(
        _spec(),
        tasks=tasks,
        holdout_tasks=holdout,
        rounds=1,
        require_holdout=True,
        recheck_baseline=True,
    )
    payload = run_workforce_loop(
        spec, runtime, subject_root=tmp_path / "subject",
        reference=reference, work_dir=tmp_path / "work", out_dir=tmp_path / "out",
    )

    final = payload["final"]
    assert final["rolled_back"] is True
    assert final["subject"] == str(tmp_path / "work" / "baseline" / "subject")
    # baseline t1 is 0.0, the accepted candidate scored 1.0
    assert final["score"] == pytest.approx(0.0)
    assert final["s_star"] == pytest.approx(0.0)
    assert final["candidate_score"] == pytest.approx(1.0)
    assert final["candidate_s_star"] == pytest.approx(1.0)


def test_final_holdout_error_rolls_back_instead_of_claiming_a_pass(tmp_path: Path):
    tasks = ("t1",)
    holdout = ("h1",)
    reference = [_outcome("t1", "r-t1", 1.0), _outcome("h1", "r-h1", 1.0)]

    def evaluate(root, current_tasks, k):
        if "baseline" in str(root):
            raise RuntimeError("harbor unavailable")
        return [
            TrialOutcome(task_id=task, trial_id=f"cand-{task}", reward=0.0)
            for task in current_tasks
        ]

    runtime, _ = _runtime(tmp_path, [], reference=reference)
    runtime = replace(runtime, evaluate=evaluate)
    spec = replace(
        _spec(),
        tasks=tasks,
        holdout_tasks=holdout,
        rounds=1,
        require_holdout=True,
        recheck_baseline=True,
    )
    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert payload["final"]["holdout"]["non_inferior"] is False
    assert "harbor unavailable" in payload["final"]["holdout"]["error"]
    assert payload["final"]["holdout_ok"] is False
    assert payload["final"]["rolled_back"] is True
    assert payload["final"]["verdict"] == "rolled_back"


def test_final_holdout_uses_the_reference_when_recheck_is_disabled(tmp_path: Path):
    tasks = ("t1",)
    holdout = ("h1",)
    reference = [_outcome("t1", "r-t1", 1.0), _outcome("h1", "r-h1", 1.0)]
    evaluated: list[str] = []

    def evaluate(root, current_tasks, k):
        evaluated.append(str(root))
        return [
            TrialOutcome(task_id=task, trial_id=f"cand-{task}", reward=0.0)
            for task in current_tasks
        ]

    runtime, _ = _runtime(tmp_path, [], reference=reference)
    runtime = replace(runtime, evaluate=evaluate)
    spec = replace(
        _spec(),
        tasks=tasks,
        holdout_tasks=holdout,
        rounds=1,
        require_holdout=True,
        recheck_baseline=False,
    )
    payload = run_workforce_loop(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert payload["final"]["holdout"]["baseline"] == "reference_file"
    assert payload["final"]["holdout"]["non_inferior"] is False
    assert payload["final"]["rolled_back"] is True
    assert all("baseline" not in root for root in evaluated)


def test_recheck_baseline_must_be_a_bool():
    with pytest.raises(WorkforceError, match="recheck_baseline"):
        replace(_spec(), recheck_baseline=1)


def _runtime(
    tmp_path: Path,
    calls: list[str],
    *,
    screen: dict | None = None,
    barrier: threading.Barrier | None = None,
    reference: list[TrialOutcome] | None = None,
):
    reference = reference or [_outcome("t1", "ref-a", 1.0)]
    subject = tmp_path / "subject"
    subject.mkdir()
    (subject / "a.py").write_text("x = 1\n", encoding="utf-8")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")

    def monitor(root, records, tasks, k):
        calls.append("monitor")
        if barrier is not None:
            barrier.wait(timeout=2)
        return {
            "tasks": list(tasks),
            "per_task": {},
            "failures": [],
            "summary": {"mean_reward": 1.0},
        }

    def architect(root, subject_name):
        calls.append("architect")
        if barrier is not None:
            barrier.wait(timeout=2)
        return {
            "subject": subject_name,
            "root": str(root),
            "components": [],
            "totals": {},
            "candidates": [],
        }

    def summarize(subject, evidence, architecture, work, spec):
        calls.append("summarize")
        return PlanResult(
            mode="delete",
            candidate_id="c1",
            layer="L2",
            summary="test plan",
            batch=({"id": "c1"},),
        )

    def execute(subject_root, plan, work_dir, spec, evidence=None, architecture=None):
        calls.append("execute")
        return ExecutionResult(
            plan_id=plan.candidate_id,
            root=candidate_root,
            changed=("a.py",),
            diff="-x = 0\n+x = 1\n",
            summary="test execution",
        )

    def verify(root, changed, spec):
        calls.append("verify")
        return aggregate(
            [
                CheckOutcome("compile", "pass"),
                CheckOutcome("import", "pass"),
                CheckOutcome("smoke", "pass"),
            ],
            ("compile", "import", "smoke"),
        )

    def evaluate(root, tasks, k):
        calls.append("evaluate")
        return reference

    def screen_fn(execution, verify_report, evidence, architecture, spec):
        calls.append("screen")
        return screen or {"status": "pass"}

    def judge(execution, verify_report, evidence, architecture, ref, outcomes, spec):
        calls.append("judge")
        return {
            "verdict": "delete",
            "reason": "all gates pass",
            "gate": {},
            "verify": verify_report.as_dict(),
        }

    return (
        WorkforceRuntime(
            monitor=monitor,
            architect=architect,
            summarize=summarize,
            execute=execute,
            verify=verify,
            evaluate=evaluate,
            screen=screen_fn,
            judge=judge,
        ),
        reference,
    )


def _spec() -> WorkforceSpec:
    return WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
    )


def test_workforce_coreset_selects_a_subset_before_monitoring(tmp_path: Path):
    calls: list[str] = []
    reference = [
        _outcome("t1", "ref-t1", 1.0),
        _outcome("t2", "ref-t2", 1.0),
        _outcome("t3", "ref-t3", 0.0),
        _outcome("t4", "ref-t4", 0.0),
    ]
    runtime, reference = _runtime(tmp_path, calls, reference=reference)
    seen: dict[str, tuple[str, ...]] = {}
    original_monitor = runtime.monitor

    def monitor(root, records, tasks, k):
        seen["tasks"] = tuple(tasks)
        return original_monitor(root, records, tasks, k)

    runtime = replace(runtime, monitor=monitor)
    spec = replace(
        _spec(),
        tasks=("t1", "t2", "t3", "t4"),
        coreset_k=2,
    )

    run_workforce_round(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert len(seen["tasks"]) == 2
    assert set(seen["tasks"]) == {"t3", "t4"}


def test_runner_orders_five_employees_and_writes_artifacts(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)

    payload = run_workforce_round(
        _spec(),
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert set(calls[:2]) == {"monitor", "architect"}
    assert calls[2:] == ["summarize", "execute", "screen", "verify", "evaluate", "judge"]
    for name in (
        "monitor.json",
        "architecture.json",
        "plan.json",
        "execution.json",
        "verify.json",
        "evaluation.jsonl",
        "review.json",
        "round.json",
        "state.json",
        "events.jsonl",
    ):
        assert (tmp_path / "out" / name).exists(), name
    state = json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))
    assert state["stage"] == "done"
    ledger_rows = (tmp_path / "out" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(ledger_rows[0])["verdict"] == "delete"
    assert payload["roles"] == ["monitor", "architect", "summarizer", "executor", "reviewer"]


def test_monitor_and_architect_run_in_parallel(tmp_path: Path):
    calls: list[str] = []
    barrier = threading.Barrier(2)
    runtime, reference = _runtime(tmp_path, calls, barrier=barrier)

    run_workforce_round(
        _spec(),
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert set(calls[:2]) == {"monitor", "architect"}
    assert calls[2:] == ["summarize", "execute", "screen", "verify", "evaluate", "judge"]


def test_runner_stops_before_evaluation_when_semantic_critic_rejects(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)
    runtime = replace(
        runtime,
        semantic=lambda execution, plan, evidence, architecture, spec: {
            "verdict": "reject",
            "reasons": ["task-specific entity in diff"],
            "risk_notes": [],
        },
    )
    spec = replace(_spec(), semantic_review=True)

    payload = run_workforce_round(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert "evaluate" not in calls
    assert "verify" not in calls
    assert payload["review"]["verdict"] == "keep"
    assert payload["review"]["stage"] == "semantic"
    state = json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))
    assert state["stage"] == "rejected"


def test_runner_rejects_without_repair_when_semantic_is_unknown(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "a.py").write_text("x = 1\n", encoding="utf-8")

    def summarize(subject, evidence, architecture, work, spec):
        calls.append("summarize")
        return PlanResult(
            mode="edit",
            candidate_id="edit-1",
            layer="L2",
            summary="test edit",
            diff="-x = 0\n+x = 1\n",
            staged_root=staged,
        )

    def repair(subject, plan, critic, evidence, architecture, work, spec):
        calls.append("repair")
        raise AssertionError("repair must not run while the critic is unavailable")

    def semantic(execution, plan, evidence, architecture, spec):
        calls.append("semantic")
        return {
            "verdict": "unknown",
            "reasons": ["semantic critic unavailable: timeout"],
            "risk_notes": [],
        }

    runtime = replace(runtime, summarize=summarize, repair=repair, semantic=semantic)
    spec = replace(_spec(), mode="edit", semantic_review=True, repair_rounds=1)

    payload = run_workforce_round(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert "repair" not in calls
    assert "verify" not in calls
    assert "evaluate" not in calls
    assert payload["review"]["verdict"] == "keep"
    assert payload["review"]["stage"] == "semantic"
    assert "semantic unavailable" in payload["review"]["reason"]
    assert "verify" not in payload["review"]
    assert "verify.json" not in payload["artifacts"]
    review = json.loads((tmp_path / "out" / "review.json").read_text(encoding="utf-8"))
    assert review["semantic"]["verdict"] == "unknown"


def test_runner_stops_before_evaluation_when_screen_rejects(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(
        tmp_path,
        calls,
        screen={"status": "reject", "verdict": "keep", "reason": "task leakage"},
    )

    payload = run_workforce_round(
        _spec(),
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert set(calls[:2]) == {"monitor", "architect"}
    assert calls[2:] == ["summarize", "execute", "screen"]
    assert "verify" not in calls
    assert payload["review"]["verdict"] == "keep"
    state = json.loads((tmp_path / "out" / "state.json").read_text(encoding="utf-8"))
    assert state["stage"] == "rejected"
    ledger_rows = (tmp_path / "out" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(ledger_rows[0])["verdict"] == "keep"


def test_screen_runs_before_semantic_and_verify(tmp_path: Path):
    calls: list[str] = []
    runtime, reference = _runtime(tmp_path, calls)
    original_verify = runtime.verify
    original_screen = runtime.screen
    original_semantic = runtime.semantic

    def verify(root, changed, spec):
        calls.append("verify")
        return original_verify(root, changed, spec)

    def screen(execution, report, evidence, architecture, spec):
        calls.append("screen")
        return {"status": "pass"}

    def semantic(execution, plan, evidence, architecture, spec):
        calls.append("semantic")
        return {"verdict": "accept", "reasons": [], "risk_notes": []}

    runtime = replace(runtime, verify=verify, screen=screen, semantic=semantic)
    spec = replace(_spec(), semantic_review=True)
    run_workforce_round(
        spec,
        runtime,
        subject_root=tmp_path / "subject",
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )
    assert calls.index("screen") < calls.index("semantic") < calls.index("verify")


def test_spec_rejects_smoke_overlap(tmp_path: Path):
    with pytest.raises(WorkforceError, match="smoke"):
        replace(_spec(), tasks=("t1", "t2"), smoke_tasks=("t2",))


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_total_trials", 0),
        ("max_total_trials", -2),
        ("max_total_trials", True),
        ("max_total_trials", 1.5),
        ("max_cost_usd", 0),
        ("max_cost_usd", -0.5),
        ("max_cost_usd", False),
        ("max_cost_usd", float("nan")),
        ("max_cost_usd", float("inf")),
    ],
)
def test_spec_rejects_invalid_cost_budgets(field: str, value: object):
    with pytest.raises(WorkforceError, match=field):
        replace(_spec(), **{field: value})


def test_spec_as_dict_carries_cost_budgets():
    payload = replace(_spec(), max_total_trials=12, max_cost_usd=2.5).as_dict()
    assert payload["max_total_trials"] == 12
    assert payload["max_cost_usd"] == 2.5


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1000"])
def test_spec_rejects_invalid_max_semantic_diff_chars(value: object):
    with pytest.raises(WorkforceError, match="max_semantic_diff_chars"):
        replace(_spec(), max_semantic_diff_chars=value)


def test_spec_as_dict_carries_max_semantic_diff_chars():
    assert _spec().as_dict()["max_semantic_diff_chars"] == 120_000
    payload = replace(_spec(), max_semantic_diff_chars=50).as_dict()
    assert payload["max_semantic_diff_chars"] == 50


def test_screen_without_verify_report_still_blocks_leak(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="c1", root=candidate_root, changed=("a.py",),
        diff="+api_key = 'sk-abcdefghijklmnopqrstuvwx'\n",
    )
    result = screen_execution(execution, None, {}, {}, _spec())
    assert result["status"] == "reject"
    assert "precheck" in result["reason"]
    clean = replace(execution, diff="-x = 0\n+x = 1\n")
    assert screen_execution(clean, None, {}, {}, _spec()) == {
        "status": "pass",
        "reasons": [],
    }


def test_default_runtime_runs_a_real_delete_round(tmp_path: Path):
    subject = tmp_path / "subject"
    (subject / "pkg").mkdir(parents=True)
    (subject / "__init__.py").write_text("", encoding="utf-8")
    (subject / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    body = "\n".join(f"    v{index} = {index}" for index in range(1, 13))
    (subject / "pkg" / "a.py").write_text(
        "def _dead():\n"
        f"{body}\n"
        "    return v1 + v12\n"
        "\n"
        "\n"
        "def _live():\n"
        "    return 2\n"
        "\n"
        "\n"
        "_live()\n",
        encoding="utf-8",
    )
    reference = [_outcome("t1", "ref-t1", 1.0), _outcome("t2", "ref-t2", 1.0)]
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="subject",
        tasks=("t1",),
        holdout_tasks=("t2",),
        k=1,
        layer="L2",
        import_modules=("subject",),
        smoke_tasks=("s1",),
        forbidden_terms=("t1", "t2"),
    )
    runtime = default_runtime(
        evaluate=lambda root, tasks, k: reference,
        smoke_evaluate=lambda root, tasks, reps: reference,
    )

    payload = run_workforce_round(
        spec,
        runtime,
        subject_root=subject,
        reference=reference,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )

    assert payload["verdict"] == "delete"
    review = json.loads((tmp_path / "out" / "review.json").read_text(encoding="utf-8"))
    assert review["gate"]["verdict"] == "accept"
    effect = json.loads((tmp_path / "out" / "effect.json").read_text(encoding="utf-8"))
    assert effect["verdict"] == "simplification_effective"
    assert effect["k_gate"]["status"] == "pass"


def test_judge_execution_applies_the_rsSI_floor_for_evo(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="edit-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=10, k_ast=20),
        k_after=KVector(k_lines_nb=10, k_ast=21),
        components=("pkg",),
    )
    verify = aggregate(
        [
            CheckOutcome("compile", "pass"),
            CheckOutcome("import", "pass"),
            CheckOutcome("smoke", "pass"),
        ],
        ("compile", "import", "smoke"),
    )
    reference = [_outcome("t1", "ref-t1", 1.0), _outcome("t2", "ref-t2", 1.0)]
    outcomes = [_outcome("t1", "cand-t1", 0.5), _outcome("t2", "cand-t2", 1.0)]
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="EVO",
        mode="edit",
        subject="demo",
        tasks=("t1",),
        holdout_tasks=("t2",),
        k=1,
        delta=0.10,
    )

    review = judge_execution(execution, verify, {}, {}, reference, outcomes, spec)

    assert review["verdict"] == "keep"
    assert review["selection"]["admissible"] is False
    assert "floor" in review["reason"]


def test_judge_execution_provisional_gate_accepts_without_holdout(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="batch-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    verify = aggregate(
        [
            CheckOutcome("compile", "pass"),
            CheckOutcome("import", "pass"),
            CheckOutcome("smoke", "pass"),
        ],
        ("compile", "import", "smoke"),
    )
    reference = [_outcome("t1", "ref-t1", 1.0)]
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=("t1",),
        k=1,
        require_holdout=False,
    )

    review = judge_execution(
        execution, verify, {}, {}, reference, reference, spec
    )

    assert review["verdict"] == "delete"
    assert review["provisional"] is True
    assert review["selection"]["admissible"] is True


def test_judge_execution_keeps_a_candidate_whose_tokens_are_missing(tmp_path: Path):
    tasks = tuple(f"t{index}" for index in range(10))
    holdout = ("h1",)
    reference = [_outcome(task, f"ref-{task}", 1.0) for task in tasks + holdout]
    outcomes = [
        TrialOutcome(
            task_id=task,
            trial_id=f"cand-{task}",
            reward=1.0,
            tokens_in=None if task == "t0" else 10,
            tokens_out=None if task == "t0" else 5,
        )
        for task in tasks + holdout
    ]
    execution = ExecutionResult(
        plan_id="batch-1",
        root=tmp_path,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    verify = aggregate(
        [
            CheckOutcome("compile", "pass"),
            CheckOutcome("import", "pass"),
            CheckOutcome("smoke", "pass"),
        ],
        ("compile", "import", "smoke"),
    )
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=tasks,
        holdout_tasks=holdout,
        k=1,
    )

    review = judge_execution(execution, verify, {}, {}, reference, outcomes, spec)

    assert review["verdict"] != "delete"
    assert "token" in review["reason"].lower()


def test_judge_execution_keeps_a_candidate_when_incumbent_tokens_are_missing(
    tmp_path: Path,
):
    tasks = tuple(f"t{index}" for index in range(10))
    holdout = ("h1",)
    reference = [
        TrialOutcome(
            task_id=task,
            trial_id=f"ref-{task}",
            reward=1.0,
            tokens_in=None if task == "t0" else 10,
            tokens_out=None if task == "t0" else 5,
        )
        for task in tasks + holdout
    ]
    outcomes = [_outcome(task, f"cand-{task}", 1.0) for task in tasks + holdout]
    execution = ExecutionResult(
        plan_id="batch-1",
        root=tmp_path,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    verify = aggregate(
        [
            CheckOutcome("compile", "pass"),
            CheckOutcome("import", "pass"),
            CheckOutcome("smoke", "pass"),
        ],
        ("compile", "import", "smoke"),
    )
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=tasks,
        holdout_tasks=holdout,
        k=1,
    )

    review = judge_execution(execution, verify, {}, {}, reference, outcomes, spec)

    assert review["verdict"] != "delete"
    assert "incumbent token data missing" in review["reason"]


def test_provisional_gate_keeps_a_candidate_whose_tokens_are_missing(tmp_path: Path):
    tasks = tuple(f"t{index}" for index in range(10))
    reference = summarize_trials(
        [_outcome(task, f"ref-{task}", 1.0) for task in tasks], tasks=tasks, k=1
    )
    candidates = summarize_trials(
        [
            TrialOutcome(
                task_id=task,
                trial_id=f"cand-{task}",
                reward=1.0,
                tokens_in=None if task == "t0" else 10,
                tokens_out=None if task == "t0" else 5,
            )
            for task in tasks
        ],
        tasks=tasks,
        k=1,
    )
    execution = ExecutionResult(
        plan_id="batch-1",
        root=tmp_path,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=tasks,
        k=1,
        require_holdout=False,
    )

    review = _provisional_decision(execution, reference, candidates, spec)

    token_check = {check["name"]: check for check in review["gate"]["checks"]}["tokens"]
    assert token_check["status"] == "fail"
    assert "missing" in " ".join(token_check["reasons"])
    assert review["verdict"] == "keep"


def test_provisional_gate_is_invalid_when_a_candidate_trial_is_an_agent_error(
    tmp_path: Path,
):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="batch-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    tasks = ("t1", "t2")
    ref_records = [_outcome("t1", "ref-t1", 1.0), _outcome("t2", "ref-t2", 1.0)]
    cand_records = [
        TrialOutcome(task_id="t1", trial_id="cand-t1", status="agent_error"),
        _outcome("t2", "cand-t2", 1.0),
    ]
    reference = summarize_trials(ref_records, tasks=tasks, k=1)
    candidates = summarize_trials(cand_records, tasks=tasks, k=1)
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=tasks,
        k=1,
        require_holdout=False,
    )

    review = _provisional_decision(execution, reference, candidates, spec)

    validity_check = review["gate"]["checks"][0]
    assert validity_check["name"] == "validity"
    assert validity_check["status"] == "fail"
    assert "agent_fail_frac" in " ".join(validity_check["reasons"])
    assert review["verdict"] == "undecidable"

    judged = judge_execution(
        execution,
        aggregate(
            [
                CheckOutcome("compile", "pass"),
                CheckOutcome("import", "pass"),
                CheckOutcome("smoke", "pass"),
            ],
            ("compile", "import", "smoke"),
        ),
        {},
        {},
        ref_records,
        cand_records,
        spec,
    )

    assert judged["verdict"] != "delete"
    assert "invalid measurement" in judged["reason"]


def test_provisional_gate_is_invalid_when_a_reference_trial_is_an_agent_error(
    tmp_path: Path,
):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="batch-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    tasks = ("t1", "t2")
    ref_records = [
        TrialOutcome(task_id="t1", trial_id="ref-t1", status="agent_error"),
        _outcome("t2", "ref-t2", 1.0),
    ]
    cand_records = [_outcome("t1", "cand-t1", 1.0), _outcome("t2", "cand-t2", 1.0)]
    reference = summarize_trials(ref_records, tasks=tasks, k=1)
    candidates = summarize_trials(cand_records, tasks=tasks, k=1)
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=tasks,
        k=1,
        require_holdout=False,
    )

    review = _provisional_decision(execution, reference, candidates, spec)

    validity_check = review["gate"]["checks"][0]
    assert validity_check["name"] == "validity"
    assert validity_check["status"] == "fail"
    assert "agent_fail_frac" in " ".join(validity_check["reasons"])
    assert review["verdict"] == "undecidable"


def test_holdout_gate_is_invalid_when_a_reference_trial_is_an_agent_error(
    tmp_path: Path,
):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="batch-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    tasks = ("t1", "t2")
    holdout = ("h1", "h2")
    ref_records = [
        _outcome("t1", "ref-t1", 1.0),
        _outcome("t2", "ref-t2", 1.0),
        TrialOutcome(task_id="h1", trial_id="ref-h1", status="agent_error"),
        _outcome("h2", "ref-h2", 1.0),
    ]
    cand_records = [
        _outcome("t1", "cand-t1", 1.0),
        _outcome("t2", "cand-t2", 1.0),
        _outcome("h1", "cand-h1", 1.0),
        _outcome("h2", "cand-h2", 1.0),
    ]
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="SIM",
        mode="delete",
        subject="demo",
        tasks=tasks,
        holdout_tasks=holdout,
        k=1,
    )
    verify = aggregate(
        [
            CheckOutcome("compile", "pass"),
            CheckOutcome("import", "pass"),
            CheckOutcome("smoke", "pass"),
        ],
        ("compile", "import", "smoke"),
    )

    judged = judge_execution(
        execution, verify, {}, {}, ref_records, cand_records, spec
    )

    assert judged["verdict"] != "delete"
    assert judged["selection"]["admissible"] is True
    assert judged["gate"]["verdict"] == "invalid"
    validity_check = judged["gate"]["checks"][0]
    assert validity_check["name"] == "validity"
    assert validity_check["status"] == "fail"
    assert "agent_fail_frac" in " ".join(validity_check["reasons"])


def test_evo_selection_is_invalid_when_a_reference_trial_is_an_agent_error(
    tmp_path: Path,
):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="evo-1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        k_before=KVector(k_lines_nb=100, k_ast=500, k_chars=1000, k_cc=10),
        k_after=KVector(k_lines_nb=80, k_ast=400, k_chars=800, k_cc=8),
        components=("pkg",),
    )
    tasks = ("t1", "t2")
    ref_records = [
        TrialOutcome(task_id="t1", trial_id="ref-t1", status="agent_error"),
        _outcome("t2", "ref-t2", 1.0),
    ]
    cand_records = [_outcome("t1", "cand-t1", 1.0), _outcome("t2", "cand-t2", 1.0)]
    verify = aggregate(
        [CheckOutcome("compile", "pass"), CheckOutcome("import", "pass")],
        ("compile", "import"),
    )
    spec = WorkforceSpec(
        run_id="r1",
        round=0,
        arm="EVO",
        mode="delete",
        subject="demo",
        tasks=tasks,
        k=1,
        require_holdout=False,
    )

    judged = judge_execution(
        execution, verify, {}, {}, ref_records, cand_records, spec
    )

    # EVO has no separate gate: a broken reference must block the candidate here
    assert judged["selection"]["admissible"] is False
    assert judged["verdict"] == "keep"
    assert "invalid measurement" in judged["reason"]


def test_screen_execution_rejects_failed_verification(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="c1",
        root=candidate_root,
        changed=("a.py",),
        diff="-x = 0\n+x = 1\n",
        summary="test execution",
    )
    verify = aggregate([CheckOutcome("compile", "fail", "syntax")], ("compile",))

    result = screen_execution(execution, verify, {}, {}, _spec())

    assert result["status"] == "reject"
    assert result["verdict"] == "keep"
    assert "compile" in result["reason"]


def test_screen_execution_rejects_deterministic_leak(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "a.py").write_text("x = 1\n", encoding="utf-8")
    execution = ExecutionResult(
        plan_id="c1",
        root=candidate_root,
        changed=("a.py",),
        diff="+api_key = 'sk-abcdefghijklmnopqrstuvwx'\n",
        summary="test execution",
    )
    verify = aggregate([CheckOutcome("compile", "pass")], ("compile",))

    result = screen_execution(execution, verify, {}, {}, _spec())

    assert result["status"] == "reject"
    assert result["verdict"] == "keep"
    assert "precheck" in result["reason"]
