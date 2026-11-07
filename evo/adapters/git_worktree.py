from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class WorktreeError(RuntimeError):
    """A git worktree operation failed or the input is not a repository."""


@dataclass(frozen=True)
class Worktree:
    repo: Path
    path: Path
    base: str
    included: tuple[str, ...] = ()


def _git(repo: Path, *args: str) -> str:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(f"cannot run git: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise WorktreeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _copy_include(repo: Path, path: Path, includes: Sequence[str]) -> tuple[str, ...]:
    copied = []
    for relative in includes:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise WorktreeError(f"include must be relative: {relative!r}")
        source = (repo / candidate).resolve()
        if not source.is_relative_to(repo):
            raise WorktreeError(f"include escapes the repository: {relative!r}")
        if not source.exists():
            raise WorktreeError(f"include does not exist: {relative!r}")
        target = path / candidate
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied.append(candidate.as_posix())
    return tuple(copied)


def create_worktree(
    repo: Path,
    path: Path,
    *,
    base: str = "HEAD",
    include: Sequence[str] = (),
) -> Worktree:
    repo = Path(repo).resolve()
    path = Path(path).resolve()
    try:
        toplevel = _git(repo, "rev-parse", "--show-toplevel")
    except WorktreeError as exc:
        raise WorktreeError(f"{repo} is not a git work tree: {exc}") from exc
    if Path(toplevel).resolve() != repo:
        raise WorktreeError(f"{repo} is not the root of a git work tree")
    try:
        commit = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    except WorktreeError as exc:
        raise WorktreeError(f"unknown base {base!r}: {exc}") from exc
    if path.exists():
        raise WorktreeError(f"worktree target already exists: {path}")
    _git(repo, "worktree", "add", "--detach", str(path), commit)
    try:
        included = _copy_include(repo, path, include)
    except BaseException:
        _git(repo, "worktree", "remove", "--force", str(path))
        raise
    return Worktree(repo=repo, path=path, base=commit, included=included)


def remove_worktree(worktree: Worktree) -> None:
    if not isinstance(worktree, Worktree):
        raise WorktreeError("worktree must be a Worktree")
    _git(worktree.repo, "worktree", "remove", "--force", str(worktree.path))
