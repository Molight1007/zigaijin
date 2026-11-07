from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from evo.contracts.types import Component
from evo.domain.protection import ProtectionHit, ProtectionPolicy, check_protection


class ScanError(ValueError):
    """A component source cannot be scanned."""


@dataclass(frozen=True)
class Candidate:
    id: str
    layer: str
    kind: str
    component: str
    path: str
    span: tuple[int, int]
    symbol: str
    evidence: Mapping[str, object] = field(default_factory=dict)
    protection: tuple[ProtectionHit, ...] = ()

    @property
    def blocked(self) -> bool:
        return bool(self.protection)


def _parse_sources(sources: Mapping[str, str]) -> dict[str, ast.AST]:
    trees: dict[str, ast.AST] = {}
    for path, source in sources.items():
        try:
            trees[path] = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            raise ScanError(f"cannot parse {path}: {exc}") from exc
    return trees


def _string_tokens(source: str, path: str) -> list[str]:
    try:
        return [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.STRING
        ]
    except (tokenize.TokenError, IndentationError) as exc:
        raise ScanError(f"cannot tokenize {path}: {exc}") from exc


def _mentioned(strings: list[str], symbol: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(symbol)}(?!\w)")
    return any(pattern.search(text) for text in strings)


def _load_counts(tree: ast.AST) -> Counter[str]:
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            counts[node.id] += 1
    return counts


def _reference_counts(tree: ast.AST) -> Counter[str]:
    counts = _load_counts(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            counts[node.attr] += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                counts[alias.asname or alias.name.split(".")[0]] += 1
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    counts[alias.asname or alias.name] += 1
    return counts


def _has_star_import(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    )


def _is_type_checking_file(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
        for node in ast.walk(tree)
    )


def _is_dynamic(tree: ast.AST) -> bool:
    if _has_star_import(tree):
        return True
    dynamic_calls = {"globals", "locals", "vars", "eval", "exec", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "__getattr__",
            "__getattribute__",
        }:
            return True
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in dynamic_calls:
            return True
        if isinstance(func, ast.Name) and func.id == "getattr":
            if len(node.args) < 2 or not (
                isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
            ):
                return True
        if isinstance(func, ast.Attribute) and func.attr == "import_module":
            return True
    return False


def _all_names(tree: ast.AST) -> tuple[set[str], bool]:
    names: set[str] = set()
    opaque = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
            value = getattr(node, "value", None)
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.add(element.value)
                else:
                    opaque = True
        else:
            opaque = True
    return names, opaque


def _bound_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names]
    return [alias.asname or alias.name for alias in node.names]


def _import_candidates(
    path: str,
    node: ast.Import | ast.ImportFrom,
    loads: Counter[str],
    string_tokens: list[str],
    all_names: set[str],
    all_opaque: bool,
    dynamic: bool,
    type_checking: bool,
) -> list[tuple[str, tuple[int, int], str, dict[str, object]]]:
    if Path(path).name == "__init__.py":
        return []
    if type_checking or dynamic or all_opaque:
        return []
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
        return []
    if any(alias.name == "*" for alias in node.names):
        return []
    bound = _bound_names(node)
    if not bound or any(name in all_names for name in bound):
        return []
    if any(loads[name] > 0 or _mentioned(string_tokens, name) for name in bound):
        return []
    span = (node.lineno, node.end_lineno or node.lineno)
    return [
        (
            "unused_import",
            span,
            ", ".join(bound),
            {"bound_names": bound, "load_refs": 0, "scope": "file"},
        )
    ]


def _zero_ref_candidates(
    path: str,
    tree: ast.AST,
    refs: Counter[str],
    strings: list[str],
    all_names: set[str],
    all_opaque: bool,
    skip: bool,
    protection_policy: ProtectionPolicy,
    source: str,
) -> list[tuple[str, tuple[int, int], str, dict[str, object], tuple[ProtectionHit, ...]]]:
    if skip or all_opaque:
        return []
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if not name.startswith("_") or (name.startswith("__") and name.endswith("__")):
            continue
        if name in all_names:
            continue
        ref_count = refs[name] + (1 if _mentioned(strings, name) else 0)
        if ref_count:
            continue
        start = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        end = node.end_lineno or node.lineno
        hits = tuple(check_protection(source, path, protection_policy, (start, end)))
        found.append(
            (
                "zero_ref_function",
                (start, end),
                name,
                {"static_refs": 0, "scope": "component"},
                hits,
            )
        )
    return found


def _body_lists(tree: ast.AST) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                bodies.append(value)
    return bodies


def _unreachable_candidates(
    path: str,
    tree: ast.AST,
    source: str,
    protection_policy: ProtectionPolicy,
    skip: bool,
) -> list[tuple[str, tuple[int, int], str, dict[str, object], tuple[ProtectionHit, ...]]]:
    if skip:
        return []
    found = []
    terminators = (ast.Return, ast.Raise, ast.Break, ast.Continue)
    for body in _body_lists(tree):
        for index, statement in enumerate(body[:-1]):
            if not isinstance(statement, terminators):
                continue
            rest = body[index + 1 :]
            start = rest[0].lineno
            end = max(item.end_lineno or item.lineno for item in rest)
            hits = tuple(check_protection(source, path, protection_policy, (start, end)))
            found.append(
                (
                    "unreachable",
                    (start, end),
                    "<unreachable>",
                    {"after": type(statement).__name__.lower(), "statements": len(rest)},
                    hits,
                )
            )
            break
    return found


def scan_sources(
    component: Component,
    sources: Mapping[str, str],
    policy: ProtectionPolicy,
) -> list[Candidate]:
    trees = _parse_sources(sources)
    loads = {path: _load_counts(tree) for path, tree in trees.items()}
    refs = {path: _reference_counts(tree) for path, tree in trees.items()}
    strings = {path: _string_tokens(source, path) for path, source in sources.items()}
    dynamic_files = {path for path, tree in trees.items() if _is_dynamic(tree)}
    type_checking_files = {path for path, tree in trees.items() if _is_type_checking_file(tree)}
    all_info = {path: _all_names(tree) for path, tree in trees.items()}

    component_refs: Counter[str] = Counter()
    for counts in refs.values():
        component_refs.update(counts)
    component_strings = [text for values in strings.values() for text in values]
    any_dynamic = bool(dynamic_files)
    any_type_checking = bool(type_checking_files)

    rows: list[
        tuple[str, tuple[int, int], str, str, dict[str, object], tuple[ProtectionHit, ...]]
    ] = []

    for path, tree in trees.items():
        source = sources[path]
        all_names, all_opaque = all_info[path]
        file_skip = path in type_checking_files
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                span = (node.lineno, node.end_lineno or node.lineno)
                hits = tuple(check_protection(source, path, policy, span))
                for kind, row_span, symbol, evidence in _import_candidates(
                    path,
                    node,
                    loads[path],
                    strings[path],
                    all_names,
                    all_opaque,
                    path in dynamic_files,
                    file_skip,
                ):
                    rows.append((path, row_span, kind, symbol, evidence, hits))

        for kind, span, symbol, evidence, hits in _zero_ref_candidates(
            path,
            tree,
            component_refs,
            component_strings,
            all_names,
            all_opaque,
            any_dynamic or any_type_checking,
            policy,
            source,
        ):
            rows.append((path, span, kind, symbol, evidence, hits))

        for kind, span, symbol, evidence, hits in _unreachable_candidates(
            path, tree, source, policy, file_skip
        ):
            rows.append((path, span, kind, symbol, evidence, hits))

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [
        Candidate(
            id=f"l1-{index:04d}",
            layer="L1",
            kind=kind,
            component=component.id,
            path=path,
            span=span,
            symbol=symbol,
            evidence=evidence,
            protection=hits,
        )
        for index, (path, span, kind, symbol, evidence, hits) in enumerate(rows, start=1)
    ]
