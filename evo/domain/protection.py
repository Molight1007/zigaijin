from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Mapping


class ProtectionError(ValueError):
    """Source cannot be checked for protected constructs."""


@dataclass(frozen=True)
class ProtectionHit:
    kind: str
    path: str
    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class ProtectionPolicy:
    frozen_files: Mapping[str, str] = field(default_factory=dict)
    protected_names: tuple[str, ...] = ()


def _parse(source: str) -> ast.AST:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise ProtectionError(f"cannot parse source: {exc}") from exc


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, *test.comparators]
    has_name = any(isinstance(side, ast.Name) and side.id == "__name__" for side in sides)
    has_main = any(
        isinstance(side, ast.Constant) and side.value == "__main__" for side in sides
    )
    return has_name and has_main


def _signature_span(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[int, int]:
    if node.body:
        end = max(node.lineno, node.body[0].lineno - 1)
    else:
        end = node.lineno
    return node.lineno, end


def _assignment_targets_all(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets)


def _hits_for(source: str, path: str, policy: ProtectionPolicy) -> list[ProtectionHit]:
    tree = _parse(source)
    hits: list[ProtectionHit] = []

    def add(kind: str, start: int, end: int, reason: str) -> None:
        hits.append(ProtectionHit(kind, path, start, end, reason))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            add("assert", node.lineno, node.end_lineno or node.lineno, "assertion would be removed")
        elif isinstance(node, ast.Raise):
            add("raise", node.lineno, node.end_lineno or node.lineno, "raise would be removed")
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            add(
                "try",
                node.lineno,
                node.end_lineno or node.lineno,
                "exception structure would be touched",
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = _signature_span(node)
            name = node.name
            if node.decorator_list:
                decorator_start = min(item.lineno for item in node.decorator_list)
                add(
                    "decorator",
                    decorator_start,
                    node.lineno,
                    "decorated definition would be touched",
                )
            if name.startswith("__") and name.endswith("__"):
                add("dunder", start, end, "dunder/framework hook would be touched")
            elif not name.startswith("_"):
                add("public_signature", start, end, "public signature would be changed")
            if name == "main":
                add("entry_point", node.lineno, node.end_lineno or node.lineno, "entry point would be touched")
            if name in policy.protected_names:
                add("protected_name", start, end, "name is protected by policy")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if _assignment_targets_all(node):
                add(
                    "public_api",
                    node.lineno,
                    node.end_lineno or node.lineno,
                    "__all__ would be changed",
                )
        elif isinstance(node, ast.If) and _is_main_guard(node):
            add(
                "entry_point",
                node.lineno,
                node.end_lineno or node.lineno,
                "module entry guard would be removed",
            )

    return hits


def check_protection(
    source: str,
    path: str,
    policy: ProtectionPolicy,
    span: tuple[int, int] | None = None,
) -> list[ProtectionHit]:
    if path in policy.frozen_files:
        line_count = max(1, source.count("\n") + (0 if source.endswith("\n") else 1))
        reason = policy.frozen_files[path]
        frozen = ProtectionHit("frozen_patch", path, 1, line_count, f"frozen patch: {reason}")
        if span is None or _intersects((1, line_count), span):
            return [frozen]
        return []

    hits = _hits_for(source, path, policy)
    if span is not None:
        if span[0] > span[1]:
            raise ProtectionError(f"invalid span: {span}")
        hits = [hit for hit in hits if _intersects((hit.start, hit.end), span)]
    return sorted(hits, key=lambda hit: (hit.path, hit.start, hit.end, hit.kind))


def _intersects(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]
