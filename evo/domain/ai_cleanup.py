from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}


class AICleanupError(ValueError):
    """An AI-cleanup scan input is malformed."""


@dataclass(frozen=True)
class AICleanupFinding:
    kind: str
    path: str
    span: tuple[int, int]
    symbol: str
    reason: str
    action: str
    confidence: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "span": list(self.span),
            "symbol": self.symbol,
            "reason": self.reason,
            "action": self.action,
            "confidence": self.confidence,
        }


def _function_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
    return start, node.end_lineno or node.lineno


def _trivial_wrapper(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
        return False
    value = node.body[0].value
    if isinstance(value, ast.Await):
        value = value.value
    return isinstance(value, ast.Call)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _called_names(tree: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _body_dump(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = ast.Module(body=node.body, type_ignores=[])
    return ast.dump(body, include_attributes=False)


def _arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    args = node.args
    return tuple(
        [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        + [arg.arg for arg in (args.vararg, args.kwarg) if arg is not None]
    )


def _unread_cli_options(path: str, tree: ast.AST, source: str) -> list[AICleanupFinding]:
    used: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            used.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "args"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                used.add(node.args[1].value)

    findings: list[AICleanupFinding] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        option = node.args[0]
        if not (
            isinstance(option, ast.Constant)
            and isinstance(option.value, str)
            and option.value.startswith("--")
        ):
            continue
        dest = option.value[2:].replace("-", "_")
        for keyword in node.keywords:
            if (
                keyword.arg == "dest"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                dest = keyword.value.value
        if dest and dest not in used:
            findings.append(
                AICleanupFinding(
                    kind="unread_cli_option",
                    path=path,
                    span=(node.lineno, node.lineno),
                    symbol=dest,
                    reason=f"option {option.value!r} is declared but args.{dest} is never read",
                    action="delete_option",
                    confidence="medium",
                )
            )
    return findings


def _scan_file(path: Path, relative: str) -> list[AICleanupFinding]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    findings: list[AICleanupFinding] = []
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    wrappers = {
        node.name
        for node in functions
        if node.name.startswith("_") and _trivial_wrapper(node)
    }
    for node in functions:
        if node.name in wrappers:
            findings.append(
                AICleanupFinding(
                    kind="trivial_wrapper",
                    path=relative,
                    span=_function_span(node),
                    symbol=node.name,
                    reason="private function only forwards a call; inline it at the call site",
                    action="inline",
                    confidence="medium",
                )
            )

    calls = _called_names(tree)
    for node in functions:
        name = node.name
        if (
            name.startswith("_")
            and not (name.startswith("__") and name.endswith("__"))
            and name not in wrappers
            and calls.get(name, 0) == 1
        ):
            findings.append(
                AICleanupFinding(
                    kind="single_call_helper",
                    path=relative,
                    span=_function_span(node),
                    symbol=name,
                    reason="private helper has exactly one call site; inline it before deleting",
                    action="inline_or_delete",
                    confidence="low",
                )
            )

    groups: dict[tuple[tuple[str, ...], str], list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in functions:
        if len(node.body) < 3:
            continue
        groups.setdefault((_arg_names(node), _body_dump(node)), []).append(node)
    for group in groups.values():
        if len(group) < 2:
            continue
        names = ", ".join(node.name for node in group)
        for node in group:
            findings.append(
                AICleanupFinding(
                    kind="duplicate_body",
                    path=relative,
                    span=_function_span(node),
                    symbol=node.name,
                    reason=f"identical body to sibling function(s): {names}",
                    action="merge",
                    confidence="medium",
                )
            )

    findings.extend(_unread_cli_options(relative, tree, source))
    return findings


def find_ai_cleanup(
    root: str | Path, files: Sequence[str] | None = None
) -> tuple[AICleanupFinding, ...]:
    """Find AI-code cleanup candidates: wrappers, one-call helpers, duplicates, dead options."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise AICleanupError(f"root is not a directory: {base}")
    if files is None:
        paths = sorted(
            path
            for path in base.rglob("*.py")
            if path.is_file()
            and not any(part in SKIP_PARTS for part in path.relative_to(base).parts)
        )
    else:
        paths = [(base / Path(rel)).resolve() for rel in files]
    findings: list[AICleanupFinding] = []
    global_calls: dict[str, int] = {}
    for path in paths:
        if not path.is_relative_to(base) or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for name, count in _called_names(tree).items():
            global_calls[name] = global_calls.get(name, 0) + count
    for path in paths:
        if not path.is_relative_to(base) or not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        findings.extend(_scan_file(path, relative))
    findings = [
        row
        for row in findings
        if row.kind != "single_call_helper"
        or global_calls.get(row.symbol, 0) == 1
    ]
    return tuple(
        sorted(findings, key=lambda row: (row.path, row.span, row.kind, row.symbol))
    )
