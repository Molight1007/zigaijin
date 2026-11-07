from __future__ import annotations

import json

import pytest

from evo.application.propose import (
    Proposal,
    ProposeError,
    Workspace,
    propose,
    snapshot_tree,
    tree_diff,
)


class Scripted:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []
        self.systems = []

    def __call__(self, prompt, system):
        self.prompts.append(prompt)
        self.systems.append(system)
        if not self.answers:
            raise AssertionError("scripted model ran out of answers")
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer)


def edit_meta(**overrides):
    payload = {
        "id": "C1",
        "component": "core",
        "hypothesis": "make retries bounded",
        "targets_mode": "retry storm",
        "predicted_affected": ["t1"],
    }
    payload.update(overrides)
    return payload


def done(*edits, summary="first candidate"):
    return {"action": "done", "summary": summary, "edits": list(edits)}


def setup(tmp_path):
    root = tmp_path / "harness"
    root.mkdir()
    (root / "agent.py").write_text(
        "MAX_TURNS = 10\n\ndef run(task):\n    return task\n", encoding="utf-8"
    )
    return root


def call(root, model, **overrides):
    kwargs = {
        "root": root,
        "model": model,
        "budget": 2,
        "components": ("core", "parser", "session"),
        "evidence": [{"task": "t1", "score": 0.0, "cause": "timeout"}],
        "history": [],
        "exploration": {"sigma": False, "untried": [], "reserved": 0, "text": ""},
        "prune_set": [],
        "max_turns": 12,
    }
    kwargs.update(overrides)
    return propose(**kwargs)


def test_propose_edits_the_harness_and_finishes(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {"action": "list_files"},
        {"action": "read_file", "path": "agent.py"},
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 20",
        },
        done(edit_meta()),
    )
    proposal = call(root, model)
    assert isinstance(proposal, Proposal)
    assert proposal.status == "done"
    assert proposal.n_file_edits == 1
    assert "MAX_TURNS = 20" in (root / "agent.py").read_text(encoding="utf-8")
    assert len(proposal.transcript) == 4
    assert proposal.edits[0]["id"] == "C1"
    assert proposal.edits[0]["predicted_affected"] == ["t1"]
    assert "core" in model.systems[0]


def test_done_without_file_changes_is_bounced(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        done(edit_meta()),
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "return task",
            "new": "return task or None",
        },
        done(edit_meta()),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert proposal.n_file_edits == 1
    assert "ZERO file changes" in model.prompts[-1]


def test_over_budget_edits_are_bounced(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(edit_meta(id="C1"), edit_meta(id="C2"), edit_meta(id="C3")),
        done(edit_meta(id="C1")),
    )
    proposal = call(root, model, budget=1)
    assert proposal.status == "done"
    assert len(proposal.edits) == 1
    assert "budget" in model.prompts[-1]


def test_reserved_slot_requires_an_untried_component(tmp_path):
    root = setup(tmp_path)
    exploration = {
        "sigma": True,
        "untried": ["parser"],
        "reserved": 1,
        "text": "STALL",
    }
    model = Scripted(
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(edit_meta(component="core")),
        done(edit_meta(component="core", id="C2"), edit_meta(component="parser", id="C3")),
    )
    proposal = call(root, model, exploration=exploration)
    assert proposal.status == "done"
    assert {edit["component"] for edit in proposal.edits} == {"core", "parser"}
    assert "RESERVED" in model.prompts[-1]


def test_a_path_escape_is_refused(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {"action": "read_file", "path": "../../secret.txt"},
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(edit_meta()),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert any("ERROR" in row["result"] for row in proposal.transcript)


def test_edit_requires_a_unique_old_string(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {"action": "edit_file", "path": "agent.py", "old": "absent", "new": "x"},
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(edit_meta()),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert "not found" in proposal.transcript[0]["result"]


def test_write_refuses_to_overwrite_and_bad_extensions(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {"action": "write_file", "path": "agent.py", "content": "x = 1\n"},
        {"action": "write_file", "path": "notes.txt", "content": "x = 1\n"},
        {"action": "write_file", "path": "helpers.py", "content": "x = 1\n"},
        done(edit_meta(component="core")),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert "exists" in proposal.transcript[0]["result"]
    assert "extension" in proposal.transcript[1]["result"]
    assert (root / "helpers.py").exists()
    assert proposal.n_file_edits == 1


def test_unknown_component_is_bounced(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(edit_meta(component="nowhere")),
        done(edit_meta(component="core")),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert "component" in model.prompts[-1]


def test_empty_edits_are_bounced(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(),
        done(edit_meta()),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert len(proposal.edits) == 1


def test_malformed_json_is_bounced(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        "not json",
        {
            "action": "edit_file",
            "path": "agent.py",
            "old": "MAX_TURNS = 10",
            "new": "MAX_TURNS = 12",
        },
        done(edit_meta()),
    )
    proposal = call(root, model)
    assert proposal.status == "done"
    assert "JSON" in proposal.transcript[0]["result"]


def test_max_turns_returns_a_failed_proposal(tmp_path):
    root = setup(tmp_path)
    model = Scripted(*[{"action": "list_files"} for _ in range(3)])
    proposal = call(root, model, max_turns=3)
    assert proposal.status == "max_turns"
    assert proposal.edits == ()
    assert proposal.n_file_edits == 0


def test_abort_is_bounced_once_then_reported(tmp_path):
    root = setup(tmp_path)
    model = Scripted(
        {"action": "abort", "reason": "no good idea"},
        {"action": "abort", "reason": "still none"},
    )
    proposal = call(root, model)
    assert proposal.status == "abort"
    assert "no abort" in proposal.transcript[0]["result"]


def test_workspace_rejects_absolute_and_parent_paths(tmp_path):
    root = setup(tmp_path)
    workspace = Workspace(root, extensions=(".py",))
    with pytest.raises(ProposeError, match="relative"):
        workspace.read("/etc/passwd")
    with pytest.raises(ProposeError, match="escape"):
        workspace.read("../agent.py")
    with pytest.raises(ProposeError, match="extension"):
        workspace.write("agent.txt", "x")


def test_snapshot_collects_only_editable_files(tmp_path):
    root = setup(tmp_path)
    (root / "notes.md").write_text("x\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("y = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config.py").write_text("ignored\n", encoding="utf-8")
    snapshot = snapshot_tree(root, extensions=(".py",))
    assert set(snapshot) == {"agent.py", "pkg/mod.py"}
    assert "MAX_TURNS" in snapshot["agent.py"]


def test_tree_diff_reports_changed_added_and_deleted_files():
    before = {"agent.py": "a\nb\n", "old.py": "bye\n"}
    after = {"agent.py": "a\nc\n", "new.py": "hi\n"}
    patch = tree_diff(before, after)
    assert "--- a/agent.py" in patch
    assert "+++ b/agent.py" in patch
    assert "-b" in patch
    assert "+c" in patch
    assert "--- /dev/null" in patch
    assert "+++ b/new.py" in patch
    assert "--- a/old.py" in patch
    assert "+++ /dev/null" in patch
    assert "+++ b/agent.py" in patch


def test_tree_diff_is_empty_when_nothing_changed():
    snapshot = {"agent.py": "a\n"}
    assert tree_diff(snapshot, dict(snapshot)) == ""
