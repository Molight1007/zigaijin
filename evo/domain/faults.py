from __future__ import annotations

import ast
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


class FaultError(ValueError):
    """A source file cannot be parsed or a mutation cannot be applied."""


@dataclass(frozen=True)
class FaultEdit:
    start: int
    end: int
    before: str
    after: str

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "before": self.before,
            "after": self.after,
        }


@dataclass(frozen=True)
class FaultMutation:
    id: str
    path: str
    line: int
    operator: str
    symbol: str
    edits: tuple[FaultEdit, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": self.path,
            "line": self.line,
            "operator": self.operator,
            "symbol": self.symbol,
            "edits": [edit.as_dict() for edit in self.edits],
        }


def _line_starts(source: str) -> list[int]:
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line.encode("utf-8")))
    return starts


def _span(
    data: bytes, starts: Sequence[int], node: ast.AST
) -> tuple[int, int, str]:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in (lineno, end_lineno, col, end_col):
        raise FaultError("AST node has no source position")
    start = starts[lineno - 1] + col
    end = starts[end_lineno - 1] + end_col
    return start, end, data[start:end].decode("utf-8")


def _range_span(
    data: bytes, starts: Sequence[int], first: ast.AST, last: ast.AST
) -> tuple[int, int, str]:
    start, _, _ = _span(data, starts, first)
    _, end, _ = _span(data, starts, last)
    return start, end, data[start:end].decode("utf-8")


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _symbol(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return "<module>"


def _inside_fstring(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.JoinedStr, ast.FormattedValue)):
            return True
        current = parents.get(current)
    return False


def _flipped(value: bool) -> str:
    return "False" if value else "True"


def _shifted(value: int | float) -> str:
    if isinstance(value, int):
        return str(value + 1)
    return repr(value + 1.0)


def find_mutations(source: str, *, path: str) -> tuple[FaultMutation, ...]:
    if not isinstance(path, str) or not path.strip():
        raise FaultError(f"path must be a non-empty string: {path!r}")
    if not isinstance(source, str):
        raise FaultError("source must be text")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise FaultError(
            f"cannot parse {path}: {exc.msg} (line {exc.lineno})"
        ) from exc
    data = source.encode("utf-8")
    starts = _line_starts(source)
    parents = _parents(tree)
    mutations: list[FaultMutation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            symbol = _symbol(node, parents)
            test_start, test_end, test_text = _span(data, starts, node.test)
            already_negated = isinstance(node.test, ast.UnaryOp) and isinstance(
                node.test.op, ast.Not
            )
            if not already_negated:
                mutations.append(
                    FaultMutation(
                        id=f"negate_condition:{path}:{node.lineno}:{test_start}",
                        path=path,
                        line=node.lineno,
                        operator="negate_condition",
                        symbol=symbol,
                        edits=(
                            FaultEdit(
                                start=test_start,
                                end=test_end,
                                before=test_text,
                                after=f"not ({test_text})",
                            ),
                        ),
                    )
                )
            body, orelse = node.body, node.orelse
            is_elif = len(orelse) == 1 and isinstance(orelse[0], ast.If)
            if body and orelse and not is_elif:
                body_start, body_end, body_text = _range_span(
                    data, starts, body[0], body[-1]
                )
                else_start, else_end, else_text = _range_span(
                    data, starts, orelse[0], orelse[-1]
                )
                if body_text != else_text:
                    mutations.append(
                        FaultMutation(
                            id=f"swap_branches:{path}:{node.lineno}:{body_start}",
                            path=path,
                            line=node.lineno,
                            operator="swap_branches",
                            symbol=symbol,
                            edits=(
                                FaultEdit(
                                    start=body_start,
                                    end=body_end,
                                    before=body_text,
                                    after=else_text,
                                ),
                                FaultEdit(
                                    start=else_start,
                                    end=else_end,
                                    before=else_text,
                                    after=body_text,
                                ),
                            ),
                        )
                    )
        elif isinstance(node, ast.Constant):
            if _inside_fstring(node, parents):
                continue
            value = node.value
            if isinstance(value, bool):
                operator, after = "flip_bool", _flipped(value)
            elif isinstance(value, int):
                operator, after = "change_constant", _shifted(value)
            elif isinstance(value, float) and math.isfinite(value):
                operator, after = "change_constant", _shifted(value)
            else:
                continue
            start, end, before = _span(data, starts, node)
            if before == after:
                continue
            mutations.append(
                FaultMutation(
                    id=f"{operator}:{path}:{node.lineno}:{start}",
                    path=path,
                    line=node.lineno,
                    operator=operator,
                    symbol=_symbol(node, parents),
                    edits=(
                        FaultEdit(start=start, end=end, before=before, after=after),
                    ),
                )
            )

    mutations.sort(key=lambda mutation: (mutation.line, mutation.edits[0].start, mutation.operator))
    return tuple(mutations)


def sample_mutations(
    mutations: Sequence[FaultMutation], *, count: int, seed: int
) -> tuple[FaultMutation, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise FaultError(f"count must be a non-negative integer: {count!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FaultError(f"seed must be an integer: {seed!r}")
    pool = list(mutations)
    if count == 0:
        return ()
    if count >= len(pool):
        return tuple(pool)
    return tuple(random.Random(seed).sample(pool, count))


def apply_mutation(source: str, mutation: FaultMutation) -> str:
    if not isinstance(source, str):
        raise FaultError("source must be text")
    if not isinstance(mutation, FaultMutation):
        raise FaultError("mutation must be a FaultMutation")
    data = source.encode("utf-8")
    for edit in sorted(mutation.edits, key=lambda item: item.start, reverse=True):
        actual = data[edit.start : edit.end].decode("utf-8")
        if actual != edit.before:
            raise FaultError(
                f"mutation {mutation.id} does not match the source at "
                f"byte {edit.start}: expected {edit.before!r}, found {actual!r}"
            )
        data = data[: edit.start] + edit.after.encode("utf-8") + data[edit.end :]
    return data.decode("utf-8")


def _mutation_field(payload: Mapping[str, object], name: str) -> object:
    if name not in payload:
        raise FaultError(f"mutation is missing {name!r}")
    return payload[name]


def parse_mutations(payload: object) -> tuple[FaultMutation, ...]:
    if not isinstance(payload, Mapping):
        raise FaultError(f"fault plan must be a JSON object: {type(payload).__name__}")
    raw = payload.get("mutations")
    if not isinstance(raw, list):
        raise FaultError("fault plan has no mutations list")
    mutations = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise FaultError(f"mutation must be a JSON object: {entry!r}")
        identifier = _mutation_field(entry, "id")
        path = _mutation_field(entry, "path")
        line = _mutation_field(entry, "line")
        operator = _mutation_field(entry, "operator")
        symbol = _mutation_field(entry, "symbol")
        edits_raw = _mutation_field(entry, "edits")
        if not isinstance(identifier, str) or not identifier.strip():
            raise FaultError(f"mutation id must be a non-empty string: {identifier!r}")
        if identifier in seen:
            raise FaultError(f"duplicate mutation id: {identifier!r}")
        seen.add(identifier)
        if not isinstance(path, str) or not path.strip():
            raise FaultError(f"mutation path must be a non-empty string: {path!r}")
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise FaultError(f"mutation line must be a positive integer: {line!r}")
        if not isinstance(operator, str) or not operator.strip():
            raise FaultError(f"mutation operator must be a non-empty string: {operator!r}")
        if not isinstance(symbol, str):
            raise FaultError(f"mutation symbol must be a string: {symbol!r}")
        if not isinstance(edits_raw, list) or not edits_raw:
            raise FaultError(f"mutation {identifier!r} has no edit")
        edits = []
        for edit_raw in edits_raw:
            if not isinstance(edit_raw, Mapping):
                raise FaultError(f"mutation {identifier!r} has a malformed edit")
            start = edit_raw.get("start")
            end = edit_raw.get("end")
            before = edit_raw.get("before")
            after = edit_raw.get("after")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or isinstance(end, bool)
                or not isinstance(end, int)
                or end <= start
            ):
                raise FaultError(
                    f"mutation {identifier!r} edit has a bad span: {start!r}..{end!r}"
                )
            if not isinstance(before, str) or not isinstance(after, str):
                raise FaultError(
                    f"mutation {identifier!r} edit must carry before/after strings"
                )
            edits.append(FaultEdit(start=start, end=end, before=before, after=after))
        mutations.append(
            FaultMutation(
                id=identifier,
                path=path,
                line=line,
                operator=operator,
                symbol=symbol,
                edits=tuple(edits),
            )
        )
    return tuple(mutations)


def verify_sources(root: str | Path, sources: Mapping[str, str]) -> None:
    base = Path(root)
    if not base.is_dir():
        raise FaultError(f"subject root is not a directory: {root}")
    if not isinstance(sources, Mapping):
        raise FaultError(f"sources must be a mapping: {type(sources).__name__}")
    for raw_path, expected in sources.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise FaultError(f"source path must be a non-empty string: {raw_path!r}")
        pure = PurePosixPath(raw_path.replace("\\", "/"))
        if not raw_path or pure.is_absolute() or ".." in pure.parts or Path(raw_path).is_absolute():
            raise FaultError(f"source path must stay inside the root: {raw_path!r}")
        target = (base / Path(*pure.parts)).resolve()
        if not target.is_relative_to(base.resolve()):
            raise FaultError(f"source path must stay inside the root: {raw_path!r}")
        if not target.is_file():
            raise FaultError(f"source is missing: {raw_path}")
        if not isinstance(expected, str) or not expected.strip():
            raise FaultError(f"source hash must be a string: {raw_path}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise FaultError(
                f"source changed since the fault plan was generated: {raw_path}"
            )
