from __future__ import annotations

import subprocess

import pytest

from evo.adapters.git_worktree import (
    WorktreeError,
    create_worktree,
    remove_worktree,
)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "add", "app.py")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "one")
    (root / "app.py").write_text("x = 2\n", encoding="utf-8")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-am", "two")
    return root


def test_create_checks_out_head(repo, tmp_path):
    worktree = create_worktree(repo, tmp_path / "wt")
    assert worktree.path == (tmp_path / "wt").resolve()
    assert worktree.base == git(repo, "rev-parse", "HEAD")
    assert (worktree.path / "app.py").read_text(encoding="utf-8") == "x = 2\n"


def test_create_at_an_older_commit(repo, tmp_path):
    head_parent = git(repo, "rev-parse", "HEAD~1")
    worktree = create_worktree(repo, tmp_path / "wt", base="HEAD~1")
    assert worktree.base == head_parent
    assert (worktree.path / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_worktree_edits_do_not_touch_the_main_tree(repo, tmp_path):
    worktree = create_worktree(repo, tmp_path / "wt")
    (worktree.path / "app.py").write_text("x = 99\n", encoding="utf-8")
    assert (repo / "app.py").read_text(encoding="utf-8") == "x = 2\n"


def test_remove_deletes_the_worktree(repo, tmp_path):
    worktree = create_worktree(repo, tmp_path / "wt")
    remove_worktree(worktree)
    assert not worktree.path.exists()
    listing = git(repo, "worktree", "list", "--porcelain")
    assert str(worktree.path) not in listing


def test_remove_is_not_idempotent(repo, tmp_path):
    worktree = create_worktree(repo, tmp_path / "wt")
    remove_worktree(worktree)
    with pytest.raises(WorktreeError, match="worktree"):
        remove_worktree(worktree)


def test_includes_untracked_paths(repo, tmp_path):
    subject = repo / "vendor" / "pkg"
    subject.mkdir(parents=True)
    (subject / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (subject / "__pycache__").mkdir()
    (subject / "__pycache__" / "junk.pyc").write_bytes(b"junk")
    worktree = create_worktree(repo, tmp_path / "wt", include=["vendor/pkg"])
    copied = worktree.path / "vendor" / "pkg"
    assert (copied / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (copied / "__pycache__").exists()
    assert worktree.included == ("vendor/pkg",)
    (copied / "mod.py").write_text("y = 2\n", encoding="utf-8")
    assert (subject / "mod.py").read_text(encoding="utf-8") == "x = 1\n"


def test_include_rejects_unknown_paths(repo, tmp_path):
    with pytest.raises(WorktreeError, match="include"):
        create_worktree(repo, tmp_path / "wt", include=["vendor/missing"])


def test_include_rejects_escapes(repo, tmp_path):
    with pytest.raises(WorktreeError, match="include"):
        create_worktree(repo, tmp_path / "wt", include=["../outside"])
    assert not (tmp_path / "wt").exists()


def test_rejects_a_non_repository(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorktreeError, match="not the root"):
        create_worktree(plain, tmp_path / "wt")


def test_rejects_an_existing_target(repo, tmp_path):
    target = tmp_path / "wt"
    target.mkdir()
    (target / "keep.txt").write_text("mine\n", encoding="utf-8")
    with pytest.raises(WorktreeError):
        create_worktree(repo, target)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "mine\n"


def test_rejects_an_unknown_base(repo, tmp_path):
    with pytest.raises(WorktreeError, match="base"):
        create_worktree(repo, tmp_path / "wt", base="does-not-exist")
    assert not (tmp_path / "wt").exists()
