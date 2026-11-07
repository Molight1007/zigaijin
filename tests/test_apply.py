from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evo.adapters.git_worktree import create_worktree
from evo.application.apply import (
    ApplyError,
    delete_candidate,
    load_observations,
)
from evo.domain.kmetrics import measure_paths
from evo.domain.scan import Candidate
from evo.domain.verify import CheckOutcome, Observation


MOD = """\
import os


def _dead():
    return 1


def used():
    return 2
"""

MODULE_FILES = {
    "pkg/__init__.py": "",
    "pkg/mod.py": MOD,
}

MAIN = """\
from pkg.mod import used

print(used())
"""


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for rel, text in {**MODULE_FILES, "main.py": MAIN}.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return create_worktree(repo, tmp_path / "wt")


def dead_candidate(**overrides) -> Candidate:
    base = Candidate(
        id="c001",
        layer="L1",
        kind="zero_ref_function",
        component="pkg",
        path="pkg/mod.py",
        span=(4, 5),
        symbol="_dead",
        evidence={"static_refs": 0},
    )
    from dataclasses import replace

    return replace(base, **overrides)


def test_deletes_a_zero_ref_function(worktree):
    result = delete_candidate(
        worktree,
        dead_candidate(),
        paths=["pkg/mod.py", "main.py"],
        modules=["pkg.mod", "main"],
    )
    text = (worktree.path / "pkg/mod.py").read_text(encoding="utf-8")
    assert "_dead" not in text
    assert "def used" in text
    assert result.undeletable is False
    assert result.verify.verdict == "pass"
    assert result.baseline_import == "pass"
    assert result.before.k_lines_nb > result.after.k_lines_nb
    assert result.after.k_lines_nb == result.before.k_lines_nb - 2
    assert result.diff.startswith("--- a/pkg/mod.py")
    assert "-def _dead():" in result.diff


def test_before_matches_the_untouched_worktree(worktree):
    expected = measure_paths(worktree.path, ["pkg/mod.py", "main.py"])
    result = delete_candidate(
        worktree,
        dead_candidate(),
        paths=["pkg/mod.py", "main.py"],
        modules=["pkg.mod", "main"],
    )
    assert result.before == expected


def test_protected_candidate_is_refused(worktree):
    from evo.domain.protection import ProtectionHit

    hit = ProtectionHit("assert", "pkg/mod.py", 4, 5, "assert statement")
    candidate = dead_candidate(protection=(hit,))
    with pytest.raises(ApplyError, match="protected"):
        delete_candidate(
            worktree,
            candidate,
            paths=["pkg/mod.py"],
            modules=["pkg.mod"],
        )
    assert "_dead" in (worktree.path / "pkg/mod.py").read_text(encoding="utf-8")


def test_residual_call_site_is_undeletable(worktree):
    (worktree.path / "pkg/mod.py").write_text(
        "def _gone():\n    return 1\n\n\nVALUE = _gone()\n",
        encoding="utf-8",
    )
    candidate = dead_candidate(span=(1, 2), symbol="_gone")
    result = delete_candidate(
        worktree,
        candidate,
        paths=["pkg/mod.py"],
        modules=["pkg.mod"],
    )
    assert result.undeletable is True
    assert "_gone" in result.undeletable_reason
    assert result.verify.verdict == "fail"


def test_baseline_import_failure_is_not_attributed_to_the_deletion(worktree):
    (worktree.path / "pkg/mod.py").write_text(
        "import missing_dependency_xyz\n\n\ndef _dead():\n    return 1\n",
        encoding="utf-8",
    )
    result = delete_candidate(
        worktree,
        dead_candidate(span=(4, 5)),
        paths=["pkg/mod.py"],
        modules=["pkg.mod"],
    )
    assert result.baseline_import == "fail"
    assert result.undeletable is False
    assert result.verify.verdict == "fail"


def test_compile_failure_is_undeletable(worktree):
    (worktree.path / "pkg/mod.py").write_text(
        "def _dead():\n    return 1\n", encoding="utf-8"
    )
    candidate = dead_candidate(span=(2, 2), symbol="_dead")
    result = delete_candidate(
        worktree,
        candidate,
        paths=["pkg/mod.py"],
        modules=["pkg.mod"],
    )
    assert result.undeletable is True
    assert "compile" in result.undeletable_reason
    assert result.verify.verdict == "fail"
    assert result.after is None


def test_smoke_and_replay_evidence_reach_the_report(worktree):
    result = delete_candidate(
        worktree,
        dead_candidate(layer="L3"),
        paths=["pkg/mod.py", "main.py"],
        modules=["pkg.mod", "main"],
        extra_checks=(
            CheckOutcome("smoke", "pass"),
            CheckOutcome("replay", "pass"),
        ),
    )
    assert result.verify.verdict == "pass"
    assert {outcome.name for outcome in result.verify.outcomes} == {
        "compile",
        "import",
        "smoke",
        "replay",
    }


def test_out_of_range_span_is_refused(worktree):
    with pytest.raises(ApplyError, match="span"):
        delete_candidate(
            worktree,
            dead_candidate(span=(4, 99)),
            paths=["pkg/mod.py"],
            modules=["pkg.mod"],
        )


def test_duplicate_check_names_are_refused(worktree):
    with pytest.raises(ApplyError, match="duplicate"):
        delete_candidate(
            worktree,
            dead_candidate(),
            paths=["pkg/mod.py"],
            modules=["pkg.mod"],
            extra_checks=(CheckOutcome("smoke", "pass"), CheckOutcome("smoke", "fail")),
        )


def test_missing_candidate_file_is_refused(worktree):
    with pytest.raises(ApplyError, match="cannot read"):
        delete_candidate(
            worktree,
            dead_candidate(path="pkg/nope.py"),
            paths=["pkg/mod.py"],
            modules=["pkg.mod"],
        )


def test_import_root_can_differ_from_the_k_root(tmp_path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    files = {
        "vendor/pkg/__init__.py": "",
        "vendor/pkg/mod.py": "def _dead():\n    return 1\n",
    }
    for rel, text in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    worktree = create_worktree(repo, tmp_path / "wt2")
    candidate = dead_candidate(path="pkg/mod.py", span=(1, 2), symbol="_dead")
    result = delete_candidate(
        worktree,
        candidate,
        paths=["pkg/mod.py"],
        modules=["pkg.mod"],
        subject_root="vendor",
        import_root="vendor",
    )
    assert result.verify.verdict == "pass"
    assert result.baseline_import == "pass"


def test_import_root_must_stay_inside_the_worktree(worktree):
    with pytest.raises(ApplyError, match="import"):
        delete_candidate(
            worktree,
            dead_candidate(),
            paths=["pkg/mod.py"],
            modules=["pkg.mod"],
            import_root="..",
        )


def test_empty_paths_are_refused(worktree):
    with pytest.raises(ApplyError, match="paths"):
        delete_candidate(worktree, dead_candidate(), paths=[], modules=["pkg.mod"])


def test_empty_modules_are_refused(worktree):
    with pytest.raises(ApplyError, match="modules"):
        delete_candidate(worktree, dead_candidate(), paths=["pkg/mod.py"], modules=[])


def test_load_observations_roundtrips(tmp_path):
    path = tmp_path / "observations.json"
    path.write_text(
        '[{"case_id": "c1", "recorded": true, "reference": 1, "candidate": 1}]',
        encoding="utf-8",
    )

    observations = load_observations(path)

    assert observations == (Observation("c1", True, 1, 1),)


def test_load_observations_rejects_a_non_list(tmp_path):
    path = tmp_path / "observations.json"
    path.write_text('{"case_id": "c1"}', encoding="utf-8")

    with pytest.raises(ApplyError, match="array"):
        load_observations(path)
