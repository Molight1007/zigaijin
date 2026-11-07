from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence


class ProposeError(ValueError):
    """A proposal workspace or protocol input is malformed."""


DEFAULT_SYSTEM = """You are a harness engineer agent. You edit the source of an
LLM-agent scaffold (the "harness"). The policy model is frozen and is a
DIFFERENT model from you: improve the harness from ITS perspective, using the
task evidence as evidence of how it actually behaves. The ONLY thing you may
change is source code in the working directory; the tool environment, the
grader and the task set are frozen.

COMPONENT VOCABULARY: {components}

Rules:
- No task-specific entities, names, ids or values anywhere in code or prompts.
- Keep the public interface and entry points unchanged; the code runs
  unattended on every task, so an unhandled exception kills the whole candidate.
- One edit is ONE independent, attributable change that works on its own.
- Every edit names a component from the vocabulary and states the mechanism.
- Do not weaken error handling, assertions or safety checks to gain score.

You interact through a strict JSON protocol: reply with EXACTLY ONE action
object per turn, nothing else.
  {{"action": "list_files"}}
  {{"action": "read_file", "path": "relative/path.py"}}
  {{"action": "edit_file", "path": "p", "old": "exact substring",
   "new": "replacement"}}
      (old must occur EXACTLY ONCE in the file)
  {{"action": "write_file", "path": "p", "content": "full file content"}}
      (new files only; never overwrite an existing file)
  {{"action": "done", "summary": "one-line summary", "edits": [
     {{"id": "C1", "component": "one of the vocabulary",
      "hypothesis": "the mechanism and WHY it should move the score",
      "targets_mode": "failure mode or habit it targets",
      "predicted_affected": ["task id", "..."],
      "regression_risk": "what could break outside predicted_affected"}}]}}

There is no abort action. A round that ships nothing tests nothing: ship the
most defensible mechanism you can build within the edit budget."""


@dataclass
class Workspace:
    root: Path
    extensions: tuple[str, ...] = (".py",)
    max_read_bytes: int = 200_000

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ProposeError(f"workspace root is not a directory: {self.root}")
        if not self.extensions:
            raise ProposeError("extensions must not be empty")
        for extension in self.extensions:
            if not extension.startswith("."):
                raise ProposeError(f"extension must start with a dot: {extension!r}")

    def _target(self, rel: object) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise ProposeError(f"path must be a non-empty string: {rel!r}")
        raw = rel.replace("\\", "/")
        pure = PurePosixPath(raw)
        if pure.is_absolute() or Path(raw).is_absolute():
            raise ProposeError(f"path must be relative to the harness: {rel!r}")
        parts = list(pure.parts)
        if not parts or parts == ["."]:
            raise ProposeError(f"path must name a file: {rel!r}")
        if ".." in parts:
            raise ProposeError(f"path escapes the harness: {rel!r}")
        if parts[0] == self.root.name:
            parts = parts[1:]
        if not parts:
            raise ProposeError(f"path must name a file: {rel!r}")
        target = (self.root / Path(*parts)).resolve()
        if not target.is_relative_to(self.root):
            raise ProposeError(f"path escapes the harness: {rel!r}")
        return target

    def _check_extension(self, target: Path, rel: str) -> None:
        if target.suffix not in self.extensions:
            raise ProposeError(
                f"extension {target.suffix!r} is not editable (allowed: "
                f"{', '.join(self.extensions)}): {rel!r}"
            )

    def list_files(self) -> str:
        rows = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix == ".pyc":
                continue
            rows.append(
                f"{path.relative_to(self.root).as_posix()} ({path.stat().st_size}B)"
            )
        return "\n".join(rows) if rows else "(no files)"

    def read(self, rel: str) -> str:
        target = self._target(rel)
        if not target.is_file():
            raise ProposeError(f"no such file: {rel!r}")
        self._check_extension(target, rel)
        if target.stat().st_size > self.max_read_bytes:
            raise ProposeError(
                f"file too large to read ({target.stat().st_size}B > "
                f"{self.max_read_bytes}B): {rel!r}"
            )
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ProposeError(f"cannot read {rel!r}: {exc}") from exc

    def edit(self, rel: str, old: str, new: str) -> str:
        target = self._target(rel)
        self._check_extension(target, rel)
        if not target.is_file():
            raise ProposeError(f"no such file: {rel!r}")
        if not isinstance(old, str) or not old:
            raise ProposeError("old must be a non-empty string")
        if not isinstance(new, str):
            raise ProposeError("new must be a string")
        try:
            source = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ProposeError(f"cannot read {rel!r}: {exc}") from exc
        count = source.count(old)
        if count == 0:
            raise ProposeError(
                "old string not found (must match exactly, including whitespace)"
            )
        if count > 1:
            raise ProposeError(
                f"old string occurs {count} times; provide a longer unique context"
            )
        target.write_text(source.replace(old, new, 1), encoding="utf-8", newline="\n")
        return f"OK: edited {rel}"

    def write(self, rel: str, content: str) -> str:
        target = self._target(rel)
        self._check_extension(target, rel)
        if target.exists():
            raise ProposeError(f"{rel!r} exists; use edit_file")
        if not isinstance(content, str):
            raise ProposeError("content must be a string")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return f"OK: created {rel}"


@dataclass(frozen=True)
class Proposal:
    status: str
    summary: str
    edits: tuple[dict[str, object], ...] = ()
    n_file_edits: int = 0
    transcript: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "edits": [dict(edit) for edit in self.edits],
            "n_file_edits": self.n_file_edits,
            "transcript": [dict(row) for row in self.transcript],
        }


def _load_action(raw: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise ProposeError("model reply must be a string")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        action = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposeError(f"not valid JSON: {exc.msg}") from exc
    if isinstance(action, list):
        action = next((item for item in action if isinstance(item, dict)), None)
    if not isinstance(action, dict):
        raise ProposeError("reply must be exactly one JSON action object")
    return action


def _problems(
    action: dict[str, object],
    *,
    budget: int,
    components: Sequence[str],
    untried: Sequence[str],
    reserved: bool,
    n_file_edits: int,
) -> list[str]:
    problems: list[str] = []
    raw_edits = action.get("edits")
    if raw_edits is None:
        raw_edits = action.get("candidates")
    edits = raw_edits if raw_edits is not None else []
    if not isinstance(edits, list):
        return ["edits must be a JSON array"]
    if n_file_edits == 0 and edits:
        problems.append(
            "you declared edits but made ZERO file changes. Implement them with "
            "edit_file/write_file, then call done"
        )
    if not edits:
        problems.append("edits array is empty; ship at least one edit")
    if len(edits) > budget:
        problems.append(f"{len(edits)} edits exceed the budget b_t = {budget}")
    seen: set[str] = set()
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            problems.append(f"edit {index} is not an object")
            continue
        edit_id = edit.get("id")
        if not isinstance(edit_id, str) or not edit_id.strip():
            problems.append(f"edit {index} has no id")
        elif edit_id in seen:
            problems.append(f"duplicate edit id {edit_id!r}")
        else:
            seen.add(edit_id)
        component = edit.get("component")
        if not isinstance(component, str) or not component.strip():
            problems.append(f"edit {edit_id or index} has no component")
        elif components and component not in components:
            problems.append(
                f"edit {edit_id or index} component {component!r} is not in the "
                f"vocabulary {list(components)}"
            )
        hypothesis = edit.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            problems.append(f"edit {edit_id or index} has no hypothesis")
        predicted = edit.get("predicted_affected")
        if predicted is not None and (
            not isinstance(predicted, list)
            or any(not isinstance(task, str) or not task for task in predicted)
        ):
            problems.append(
                f"edit {edit_id or index} predicted_affected must be a list of task ids"
            )
    if reserved and untried and edits and not any(
        isinstance(edit, dict) and edit.get("component") in untried for edit in edits
    ):
        problems.append(
            "this variant holds a RESERVED EXPLORATION SLOT: at least one edit "
            f"must be on a never-exercised component from {list(untried)}"
        )
    return problems


def _context(
    *,
    workspace: Workspace,
    budget: int,
    components: Sequence[str],
    evidence: Sequence[Mapping[str, object]],
    history: Sequence[Mapping[str, object]],
    scoreboard: Sequence[Mapping[str, object]] = (),
    exploration: Mapping[str, object],
    prune_set: Sequence[Mapping[str, object]],
    reserved: bool,
    task_brief: str,
    variant_brief: str = "",
    repair_brief: Mapping[str, object] | None = None,
) -> str:
    untried = list(exploration.get("untried") or [])
    parts = []
    if variant_brief:
        parts.extend(["=== THIS VARIANT'S BRIEF ===", variant_brief])
    if task_brief:
        parts.extend(["=== TASK BRIEF ===", task_brief])
    parts.extend(
        [
            "=== THIS ROUND'S EDIT BUDGET b_t ===",
            f"You may ship AT MOST {budget} independent edit(s) in this candidate. "
            "Ship fewer if the evidence supports fewer.",
            "=== EDIT HISTORY L_t (component, hypothesis, Delta S, accepted) ===",
            json.dumps(list(history), ensure_ascii=False, indent=1)
            if history
            else "(no measured edits yet)",
            "=== ATTRIBUTION SCOREBOARD (measured edit outcomes) ===",
            json.dumps(list(scoreboard), ensure_ascii=False, indent=1)
            if scoreboard
            else "(no measured attribution yet)",
            "=== EXPLORATION DIRECTIVES E_t ===",
            str(exploration.get("text") or "(none)"),
            "=== COMPONENTS TO PRUNE B_t ===",
            json.dumps(list(prune_set), ensure_ascii=False, indent=1)
            if prune_set
            else "(none)",
            "=== TASK EVIDENCE ===",
            json.dumps(list(evidence), ensure_ascii=False, indent=1)
            if evidence
            else "(no evidence)",
            "=== CURRENT HARNESS FILES H_t ===",
            workspace.list_files(),
            "=== COMPONENT VOCABULARY ===",
            " | ".join(components) if components else "(none declared)",
        ]
    )
    if reserved and untried:
        parts.extend(
            [
                "=== RESERVED EXPLORATION SLOT ===",
                "At least one edit MUST be on a never-exercised component from: "
                f"{untried}",
            ]
        )
    if repair_brief:
        parts.extend(
            [
                "=== REPAIR ROUND ===",
                "Your previous edits for this candidate are already in the working "
                "tree (reflected in CURRENT HARNESS SOURCE above). The reviewer "
                "raised the objections below. Fix ONLY what the objections require "
                "with minimal additional edits, then call done again with the "
                "corrected edits array.",
                json.dumps(dict(repair_brief), ensure_ascii=False, indent=1),
            ]
        )
    parts.append("First action:")
    return "\n\n".join(parts)


def propose(
    *,
    root: Path,
    model: Callable[[str, str], str],
    budget: int,
    components: Sequence[str] = (),
    evidence: Sequence[Mapping[str, object]] = (),
    history: Sequence[Mapping[str, object]] = (),
    scoreboard: Sequence[Mapping[str, object]] = (),
    exploration: Mapping[str, object] | None = None,
    prune_set: Sequence[Mapping[str, object]] = (),
    reserved: bool = False,
    task_brief: str = "",
    variant_brief: str = "",
    max_turns: int = 24,
    max_file_edits: int = 40,
    extensions: Sequence[str] = (".py",),
    repair_brief: Mapping[str, object] | None = None,
) -> Proposal:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ProposeError(f"budget must be a positive integer: {budget!r}")
    if not callable(model):
        raise ProposeError("model must be callable")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise ProposeError(f"max_turns must be a positive integer: {max_turns!r}")
    if (
        isinstance(max_file_edits, bool)
        or not isinstance(max_file_edits, int)
        or max_file_edits <= 0
    ):
        raise ProposeError(
            f"max_file_edits must be a positive integer: {max_file_edits!r}"
        )
    vocabulary = tuple(dict.fromkeys(str(item) for item in components))
    explore = dict(exploration or {})
    untried = [str(item) for item in (explore.get("untried") or [])]
    reserve = bool(reserved or explore.get("reserved", 0))
    workspace = Workspace(Path(root), extensions=tuple(extensions))
    system = DEFAULT_SYSTEM.format(
        components=" | ".join(vocabulary) if vocabulary else "(none declared)"
    )
    context = _context(
        workspace=workspace,
        budget=budget,
        components=vocabulary,
        evidence=evidence,
        history=history,
        scoreboard=scoreboard,
        exploration=explore,
        prune_set=prune_set,
        reserved=reserve,
        task_brief=task_brief,
        variant_brief=variant_brief,
        repair_brief=repair_brief,
    )
    transcript: list[dict[str, object]] = []
    log: list[str] = []
    n_file_edits = 0
    aborts = 0
    for _turn in range(max_turns):
        prompt = (
            "=== ROUND CONTEXT ===\n"
            + context
            + "\n\n=== INTERACTION LOG ===\n"
            + ("\n".join(log) if log else "(empty)")
            + "\n\nReply with exactly one JSON action object."
        )
        raw = model(prompt, system)
        try:
            action = _load_action(raw)
        except ProposeError as exc:
            log.append(f"[you] {str(raw)[:500]}\n[result] ERROR: {exc}")
            transcript.append({"action": None, "result": f"ERROR: {exc}"})
            continue
        name = action.get("action")
        if name == "done":
            problems = _problems(
                action,
                budget=budget,
                components=vocabulary,
                untried=untried,
                reserved=reserve,
                n_file_edits=n_file_edits,
            )
            if problems:
                message = "ERROR: " + "; ".join(problems) + ". Call done again fixed."
                log.append(f"[you] done\n[result] {message}")
                transcript.append({"action": action, "result": message})
                continue
            raw_edits = action.get("edits")
            if raw_edits is None:
                raw_edits = action.get("candidates")
            normalized = []
            for edit in raw_edits:
                normalized.append(
                    {
                        "id": str(edit["id"]),
                        "component": str(edit["component"]),
                        "hypothesis": str(edit["hypothesis"]),
                        "targets_mode": str(edit.get("targets_mode") or ""),
                        "predicted_affected": [
                            str(task) for task in (edit.get("predicted_affected") or [])
                        ],
                        "regression_risk": str(edit.get("regression_risk") or ""),
                    }
                )
            transcript.append(
                {"action": {"action": "done"}, "result": f"OK: {len(normalized)} edit(s)"}
            )
            return Proposal(
                status="done",
                summary=str(action.get("summary") or ""),
                edits=tuple(normalized),
                n_file_edits=n_file_edits,
                transcript=tuple(transcript),
            )
        if name == "abort":
            aborts += 1
            if aborts > 1:
                transcript.append(
                    {
                        "action": action,
                        "result": "aborted: there is no abort action",
                    }
                )
                return Proposal(
                    status="abort",
                    summary=str(action.get("reason") or ""),
                    transcript=tuple(transcript),
                )
            message = (
                "ERROR: there is no abort action. Ship the most defensible "
                "mechanism you can build within the rules, then call done."
            )
            log.append(f"[you] abort\n[result] {message}")
            transcript.append({"action": action, "result": message})
            continue
        try:
            if name == "list_files":
                result = workspace.list_files()
            elif name == "read_file":
                result = workspace.read(action.get("path"))
            elif name == "edit_file":
                if n_file_edits >= max_file_edits:
                    result = "ERROR: file edit limit reached; call done"
                else:
                    result = workspace.edit(
                        action.get("path"), action.get("old"), action.get("new")
                    )
                    n_file_edits += result.startswith("OK")
            elif name == "write_file":
                if n_file_edits >= max_file_edits:
                    result = "ERROR: file edit limit reached; call done"
                else:
                    result = workspace.write(
                        action.get("path"), action.get("content")
                    )
                    n_file_edits += result.startswith("OK")
            else:
                result = f"ERROR: unknown action {name!r}"
        except ProposeError as exc:
            result = f"ERROR: {exc}"
        cap = 60_000 if name in ("read_file", "list_files") else 4_000
        text = str(result)
        if len(text) > cap:
            text = text[:cap] + f"\n...[capped at {cap} chars]"
        log.append(f"[you] {json.dumps(action)[:1500]}\n[result] {text}")
        transcript.append({"action": action, "result": text})
    return Proposal(
        status="max_turns",
        summary="",
        n_file_edits=n_file_edits,
        transcript=tuple(transcript),
    )


_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules",
     ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


def snapshot_tree(
    root: Path, *, extensions: Sequence[str] = (".py",)
) -> dict[str, str]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ProposeError(f"workspace root is not a directory: {root}")
    allowed = tuple(extensions)
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in allowed:
            continue
        relative = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in relative.parts[:-1]):
            continue
        try:
            snapshot[relative.as_posix()] = path.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            raise ProposeError(f"cannot read {relative.as_posix()}: {exc}") from exc
    return snapshot


def tree_diff(
    before: Mapping[str, str], after: Mapping[str, str]
) -> str:
    chunks: list[str] = []
    for path in sorted(set(before) | set(after)):
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                (old or "").splitlines(keepends=True),
                (new or "").splitlines(keepends=True),
                fromfile=f"a/{path}" if old is not None else "/dev/null",
                tofile=f"b/{path}" if new is not None else "/dev/null",
                n=3,
            )
        )
    patch = "".join(chunks)
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch
