from __future__ import annotations

import ast
import gzip
import io
import tokenize
from pathlib import Path

from radon.complexity import cc_visit_ast

from evo.contracts.types import KResult, KVector


class KMeasurementError(ValueError):
    """The subject source cannot be measured as Python."""


_SKIP_TOKENS = {
    tokenize.ENCODING,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.COMMENT,
}


def _parse(source: str) -> ast.AST:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise KMeasurementError(f"cannot parse source: {exc}") from exc


def _standalone_string_spans(tree: ast.AST) -> list[tuple[int, int]]:
    spans = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            spans.append((node.lineno, node.end_lineno or node.lineno))
    return spans


def count_lines_nb(source: str) -> int:
    tree = _parse(source)
    string_spans = _standalone_string_spans(tree)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise KMeasurementError(f"cannot tokenize source: {exc}") from exc

    counted: set[int] = set()
    for tok in tokens:
        if tok.type in _SKIP_TOKENS:
            continue
        if tok.type == tokenize.STRING and any(
            start <= tok.start[0] and tok.end[0] <= end for start, end in string_spans
        ):
            continue
        counted.update(range(tok.start[0], tok.end[0] + 1))
    return len(counted)


def count_ast_nodes(source: str) -> int:
    tree = _parse(source)
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            excluded.add(id(node))
            excluded.add(id(node.value))
    return sum(
        1
        for node in ast.walk(tree)
        if not isinstance(node, (ast.Module, ast.TypeIgnore))
        and id(node) not in excluded
    )


def count_cc(source: str) -> int:
    """Classes count as 1; every function, method and closure counts its McCabe CC."""
    tree = _parse(source)
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            total += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            blocks = cc_visit_ast(node)
            if blocks:
                total += int(blocks[0].complexity)
    return total


def _gzip_size(data: bytes) -> int:
    if not data:
        return 0
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def measure_source(source: str) -> KVector:
    if not isinstance(source, str):
        raise TypeError("source must be str")
    raw = source.encode("utf-8")
    return KVector(
        k_lines_nb=count_lines_nb(source),
        k_ast=count_ast_nodes(source),
        k_chars=len(source),
        k_bytes=len(raw),
        k_gzip=_gzip_size(raw),
        k_cc=count_cc(source),
    )


def measure_sources(sources: dict[str, str]) -> KResult:
    files = {path: measure_source(source) for path, source in sources.items()}
    if not files:
        return KResult()
    joined = b"\n".join(sources[path].encode("utf-8") for path in sorted(sources))
    return KResult(
        k_lines_nb=sum(v.k_lines_nb for v in files.values()),
        k_ast=sum(v.k_ast for v in files.values()),
        k_chars=sum(v.k_chars for v in files.values()),
        k_bytes=sum(v.k_bytes for v in files.values()),
        k_gzip=_gzip_size(joined),
        k_cc=sum(v.k_cc for v in files.values()),
        files=files,
    )


def measure_paths(root: Path, paths: list[str]) -> KResult:
    base = root.resolve()
    sources: dict[str, str] = {}
    for raw_path in paths:
        rel = Path(raw_path)
        if rel.is_absolute():
            raise KMeasurementError(f"path must be relative to subject root: {raw_path}")
        target = (base / rel).resolve()
        if not target.is_relative_to(base):
            raise KMeasurementError(f"path escapes subject root: {raw_path}")
        try:
            sources[rel.as_posix()] = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise KMeasurementError(f"cannot read {raw_path}: {exc}") from exc
    return measure_sources(sources)
