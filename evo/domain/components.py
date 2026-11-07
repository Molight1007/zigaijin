from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath

from evo.contracts.types import BoundarySpec, Component, KResult
from evo.domain.kmetrics import measure_paths


class ComponentError(ValueError):
    """Boundary declaration cannot be turned into disjoint components."""


_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
_MAGIC = set("*?[")


def _normalize(pattern: str) -> str:
    normalized = pattern.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _validate_name(name: str) -> None:
    if not _NAME.match(name):
        raise ComponentError(f"invalid boundary name: {name!r}")


def _validate_pattern(pattern: str) -> str:
    normalized = _normalize(pattern)
    if not normalized:
        raise ComponentError("empty boundary pattern")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ComponentError(f"pattern escapes subject root: {pattern}")
    return normalized


def _has_magic(pattern: str) -> bool:
    return any(ch in pattern for ch in _MAGIC)


def _is_excluded(rel: str, patterns: tuple[str, ...]) -> bool:
    for raw in patterns:
        pattern = _normalize(raw)
        if not pattern:
            continue
        if fnmatch.fnmatchcase(rel, pattern):
            return True
        if not _has_magic(pattern) and (rel == pattern or rel.startswith(pattern + "/")):
            return True
    return False


def _relative_files(root: Path, pattern: str) -> list[str]:
    base = root.resolve()
    if _has_magic(pattern):
        matches = list(root.glob(pattern))
        if not matches:
            raise ComponentError(f"boundary does not exist or matches nothing: {pattern}")
        candidates = [path for path in matches if path.is_file() and path.suffix == ".py"]
    else:
        target = root / pattern
        if not target.exists():
            raise ComponentError(f"boundary does not exist: {pattern}")
        if target.is_dir():
            candidates = [path for path in target.rglob("*.py") if path.is_file()]
        else:
            candidates = [target] if target.suffix == ".py" else []
    files: list[str] = []
    for path in candidates:
        resolved = path.resolve()
        if not resolved.is_relative_to(base):
            raise ComponentError(f"boundary path escapes subject root: {path}")
        rel = resolved.relative_to(base).as_posix()
        if not _is_excluded(rel, ()):
            files.append(rel)
    return sorted(set(files))


def derive_components(root: Path, spec: BoundarySpec) -> list[Component]:
    if not root.is_dir():
        raise ComponentError(f"subject root is not a directory: {root}")
    _validate_name(spec.subject)
    seen_names: set[str] = set()
    seen_paths: dict[str, str] = {}
    components: list[Component] = []

    for boundary in spec.boundaries:
        _validate_name(boundary.name)
        if boundary.name in seen_names:
            raise ComponentError(f"duplicate boundary name: {boundary.name}")
        seen_names.add(boundary.name)
        if not boundary.patterns:
            raise ComponentError(f"boundary has no patterns: {boundary.name}")

        paths: set[str] = set()
        for raw_pattern in boundary.patterns:
            pattern = _validate_pattern(raw_pattern)
            for rel in _relative_files(root, pattern):
                if not _is_excluded(rel, spec.exclude):
                    paths.add(rel)

        if not paths:
            raise ComponentError(f"boundary has no Python files after exclusions: {boundary.name}")

        for rel in paths:
            if rel in seen_paths:
                raise ComponentError(
                    f"overlap: {rel} is in both {seen_paths[rel]} and {boundary.name}"
                )
            seen_paths[rel] = boundary.name

        components.append(
            Component(
                id=f"{spec.subject}:{boundary.name}",
                name=boundary.name,
                paths=tuple(sorted(paths)),
                description=boundary.description,
            )
        )

    return sorted(components, key=lambda component: component.id)


def measure_components(root: Path, components: list[Component]) -> dict[str, KResult]:
    return {
        component.id: measure_paths(root, list(component.paths))
        for component in components
    }
