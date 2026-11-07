from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evo.application.baseline import TrialOutcome
from evo.application.simplify_arm import (
    ScoredCandidate,
    SimplifyArmError,
    SimplifyRuntime,
    SimplifySpec,
    all_unit_candidates,
    apply_batch,
    ddmin,
    lever_agent_kwargs,
    load_scored_candidates,
    plan_batch,
    run_simplify_arm,
    select_zero_contribution_levers,
    target_k_from_candidates,
)
from evo.contracts.types import KVector
from evo.domain.protection import ProtectionHit
from evo.domain.scan import Candidate
from evo.domain.verify import CheckOutcome, aggregate, required_checks


def candidate(
    candidate_id: str,
    *,
    path: str = "pkg/mod.py",
    span: tuple[int, int] = (1, 2),
    component: str = "core",
) -> Candidate:
    return Candidate(
        id=candidate_id,
        layer="L2",
        kind="zero_touch_function",
        component=component,
        path=path,
        span=span,
        symbol=candidate_id,
    )


def scored(
    candidate_id: str,
    *,
    k_ast: int,
    path: str = "pkg/mod.py",
    span: tuple[int, int] = (1, 2),
    component: str = "core",
) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=candidate(candidate_id, path=path, span=span, component=component),
        delta=KVector(k_lines_nb=k_ast // 2, k_ast=k_ast, k_chars=k_ast * 4),
    )


def test_select_zero_contribution_levers_filters_and_collects_kwargs():
    payload = {
        "levers": [
            {
                "id": "recording",
                "verdict": "zero_contribution",
                "agent_kwargs": ["record_terminal_session=false"],
            },
            {
                "id": "retry",
                "verdict": "contributes",
                "agent_kwargs": ["retry=false"],
            },
        ]
    }

    selected = select_zero_contribution_levers(payload)

    assert [row["id"] for row in selected] == ["recording"]
    assert lever_agent_kwargs(selected) == ("record_terminal_session=false",)


def test_simplify_spec_records_l3_levers():
    selected = (
        {
            "id": "recording",
            "verdict": "zero_contribution",
            "agent_kwargs": ["record_terminal_session=false"],
        },
    )
    spec = SimplifySpec(
        arm="SIM",
        mode="full",
        k=1,
        levers=selected,
        agent_kwargs=("record_terminal_session=false",),
    )

    assert spec.levers == selected
    assert spec.agent_kwargs == ("record_terminal_session=false",)


def test_static_batch_takes_large_non_overlapping_candidates():
    items = [
        scored("small", k_ast=10),
        scored("large", k_ast=100, span=(1, 10)),
        scored("overlap", k_ast=50, span=(5, 6)),
        scored("next", k_ast=80, span=(12, 20)),
    ]

    selected = plan_batch(items, mode="static")

    assert [item.candidate.id for item in selected] == ["large", "next"]


def test_random_batch_is_seeded_and_reaches_the_target():
    items = [
        scored("a", k_ast=40, span=(1, 2)),
        scored("b", k_ast=40, span=(4, 5)),
        scored("c", k_ast=40, span=(7, 8)),
        scored("d", k_ast=40, span=(10, 11)),
    ]

    first = plan_batch(items, mode="random", target_k_ast=80, seed=7)
    second = plan_batch(items, mode="random", target_k_ast=80, seed=7)

    assert [item.candidate.id for item in first] == [
        item.candidate.id for item in second
    ]
    assert sum(item.delta.k_ast for item in first) >= 80
    assert len({item.candidate.id for item in first}) == len(first)


def test_sharp_batch_takes_whole_components_largest_first():
    items = [
        scored("a1", k_ast=60, span=(1, 2), component="alpha"),
        scored("a2", k_ast=60, span=(4, 5), component="alpha"),
        scored("b1", k_ast=100, span=(1, 2), path="pkg/beta.py", component="beta"),
        scored("c1", k_ast=30, span=(1, 2), path="pkg/gamma.py", component="gamma"),
    ]

    selected = plan_batch(items, mode="sharp", target_k_ast=100)

    assert [item.candidate.id for item in selected] == ["a1", "a2"]


def test_plan_batch_rejects_a_protected_candidate():
    item = scored("protected", k_ast=10)
    protected = replace(
        item.candidate,
        protection=(ProtectionHit("assert", "pkg/mod.py", 1, 2, "assert"),),
    )

    with pytest.raises(SimplifyArmError, match="protected"):
        plan_batch([ScoredCandidate(protected, item.delta)], mode="static")


def test_target_k_uses_the_non_overlapping_static_batch():
    items = [
        scored("large", k_ast=100, span=(1, 10)),
        scored("overlap", k_ast=50, span=(5, 6)),
        scored("next", k_ast=80, span=(12, 20)),
    ]

    assert target_k_from_candidates(items) == 180


def test_ddmin_returns_a_minimal_failing_subset():
    items = [
        scored("a", k_ast=10),
        scored("b", k_ast=10, span=(4, 5)),
        scored("c", k_ast=10, span=(7, 8)),
        scored("d", k_ast=10, span=(10, 11)),
    ]

    def still_fails(subset):
        return any(item.candidate.id == "c" for item in subset)

    reduced = ddmin(items, still_fails)

    assert [item.candidate.id for item in reduced] == ["c"]


TASKS = ("t1", "t2")
HOLDOUT = ("h1",)


def outcome_rows(rewards, tasks, prefix, *, k=1):
    return [
        TrialOutcome(
            task_id=task,
            trial_id=f"{prefix}-{task}-{rep}",
            reward=rewards[task],
            tokens_in=90,
            tokens_out=10,
            cost_usd=0.001,
        )
        for task in tasks
        for rep in range(k)
    ]


def reference_rows():
    ids = TASKS + HOLDOUT
    return outcome_rows({task: 0.5 for task in ids}, ids, "ref")


def report(layer, status):
    checks = required_checks(layer)
    return aggregate([CheckOutcome(name, status) for name in checks], checks)


def make_subject(tmp_path):
    root = tmp_path / "harness" / "terminus2"
    root.mkdir(parents=True)
    (root / "agent.py").write_text(
        "".join(f"x{index} = {index}\n" for index in range(100)), encoding="utf-8"
    )
    return root


def fake_runtime(*, verify_status="pass", apply_error=False):
    calls = {"apply": 0, "evaluate": 0}

    def apply(root, batch):
        calls["apply"] += 1
        if apply_error:
            raise RuntimeError("cannot apply batch")
        path = root / "agent.py"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-10]), encoding="utf-8")
        return ("agent.py",)

    def verify(root, files):
        return report("L2", verify_status)

    def measure(root):
        text = (root / "agent.py").read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        return KVector(
            k_lines_nb=len(lines),
            k_ast=len(lines) * 2,
            k_chars=len(text),
            k_bytes=len(text.encode("utf-8")),
            k_gzip=len(text),
            k_cc=1,
        )

    def evaluate(root, tasks, k):
        calls["evaluate"] += 1
        ids = tuple(tasks)
        return outcome_rows({task: 0.5 for task in ids}, ids, "cand", k=k)

    return (
        SimplifyRuntime(
            apply=apply,
            verify=verify,
            measure=measure,
            evaluate=evaluate,
        ),
        calls,
    )


def run_runner(
    tmp_path,
    *,
    verify_status="pass",
    apply_error=False,
    reference=None,
    arm="SIM",
    mode="full",
    target_k_ast=None,
    levers=(),
    agent_kwargs=(),
):
    subject = make_subject(tmp_path)
    runtime, calls = fake_runtime(
        verify_status=verify_status, apply_error=apply_error
    )
    spec = SimplifySpec(
        arm=arm,
        mode=mode,
        k=1,
        n_boot=400,
        target_k_ast=target_k_ast,
        levers=levers,
        agent_kwargs=agent_kwargs,
    )
    payload = run_simplify_arm(
        spec,
        runtime,
        candidates=(scored("dead", k_ast=20, span=(1, 20)),),
        subject_root=subject,
        reference_outcomes=reference if reference is not None else reference_rows(),
        tasks=TASKS,
        holdout_tasks=HOLDOUT,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )
    return payload, calls, tmp_path / "out"


def test_runner_accepts_a_clean_batch(tmp_path):
    payload, calls, out = run_runner(tmp_path)

    assert payload["decision"]["verdict"] == "delete"
    assert calls == {"apply": 1, "evaluate": 1}
    assert (out / "simplify.json").is_file()
    assert (out / "decision.json").is_file()
    assert (out / "ledger.jsonl").is_file()


def test_runner_records_l3_levers_in_the_plan(tmp_path):
    lever = {
        "id": "recording",
        "verdict": "zero_contribution",
        "agent_kwargs": ["record_terminal_session=false"],
    }
    payload, _, _ = run_runner(
        tmp_path,
        levers=(lever,),
        agent_kwargs=("record_terminal_session=false",),
    )

    assert payload["spec"]["levers"][0]["id"] == "recording"
    assert payload["spec"]["agent_kwargs"] == ("record_terminal_session=false",)


def test_runner_keeps_the_code_when_verification_fails(tmp_path):
    payload, calls, _ = run_runner(tmp_path, verify_status="fail")

    assert payload["decision"]["verdict"] == "keep"
    assert calls["evaluate"] == 0


def test_runner_marks_an_apply_failure_undeletable(tmp_path):
    payload, calls, _ = run_runner(tmp_path, apply_error=True)

    assert payload["decision"]["verdict"] == "undeletable"
    assert calls["evaluate"] == 0


def test_placebo_requires_an_explicit_target(tmp_path):
    with pytest.raises(SimplifyArmError, match="target"):
        run_runner(
            tmp_path,
            arm="PLACEBO",
            mode="random",
        )


def test_reference_missing_holdout_is_rejected_before_apply(tmp_path):
    reference = outcome_rows(
        {task: 0.5 for task in TASKS}, TASKS, "ref"
    )
    with pytest.raises(SimplifyArmError, match="holdout"):
        run_runner(tmp_path, reference=reference)


def test_apply_batch_deletes_spans_bottom_up(tmp_path):
    root = tmp_path / "subject"
    root.mkdir()
    (root / "mod.py").write_text(
        "".join(f"line{index}\n" for index in range(1, 11)), encoding="utf-8"
    )
    batch = (
        scored("a", k_ast=10, path="mod.py", span=(2, 3)),
        scored("b", k_ast=10, path="mod.py", span=(6, 7)),
    )

    changed = apply_batch(root, batch)

    assert changed == ("mod.py",)
    assert (root / "mod.py").read_text(encoding="utf-8") == (
        "line1\nline4\nline5\nline8\nline9\nline10\n"
    )


def test_apply_batch_rejects_overlapping_spans(tmp_path):
    root = tmp_path / "subject"
    root.mkdir()
    (root / "mod.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    batch = (
        scored("a", k_ast=10, path="mod.py", span=(1, 2)),
        scored("b", k_ast=10, path="mod.py", span=(2, 3)),
    )

    with pytest.raises(SimplifyArmError, match="overlap"):
        apply_batch(root, batch)


def census_payload():
    return {
        "subject": "terminus2",
        "candidates": [
            {
                "id": "l2-0001",
                "layer": "L2",
                "kind": "zero_touch_function",
                "component": "terminus2:all",
                "path": "pkg/a.py",
                "span": [2, 4],
                "symbol": "_dead",
                "blocked": False,
                "delta": {"k_lines_nb": 2, "k_ast": 4, "k_chars": 10},
            },
            {
                "id": "l2-0002",
                "layer": "L2",
                "kind": "zero_touch_function",
                "component": "terminus2:all",
                "path": "pkg/b.py",
                "span": [1, 2],
                "symbol": "_protected",
                "blocked": True,
                "delta": {"k_lines_nb": 1, "k_ast": 2, "k_chars": 8},
            },
            {
                "id": "l2-0003",
                "layer": "L2",
                "kind": "zero_touch_function",
                "component": "terminus2:all",
                "path": "pkg/c.py",
                "span": [1, 2],
                "symbol": "_unknown",
                "blocked": False,
                "delta": None,
            },
        ],
    }


def test_all_unit_candidates_include_touched_private_functions(tmp_path):
    root = tmp_path / "terminus2"
    root.mkdir()
    (root / "mod.py").write_text(
        "def live():\n"
        "    return _helper()\n"
        "\n"
        "\n"
        "def _helper():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def _dead():\n"
        "    return 2\n",
        encoding="utf-8",
    )

    rows = all_unit_candidates(root)

    names = {row.candidate.symbol for row in rows}
    assert {"_helper", "_dead"} <= names
    assert "live" not in names


def test_load_scored_candidates_skips_blocked_and_missing_deltas(tmp_path):
    path = tmp_path / "census.json"
    path.write_text(__import__("json").dumps(census_payload()), encoding="utf-8")

    rows = load_scored_candidates(path)

    assert [row.candidate.id for row in rows] == ["l2-0001"]
    assert rows[0].candidate.path == "pkg/a.py"
    assert rows[0].candidate.span == (2, 4)
    assert rows[0].delta.k_ast == 4
