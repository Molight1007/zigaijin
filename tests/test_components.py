from __future__ import annotations

import pytest

from evo.contracts.types import Boundary, BoundarySpec
from evo.domain.components import (
    ComponentError,
    derive_components,
    measure_components,
)


def write(root, rel, text="x = 1\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_derives_file_lists_from_directories_files_and_globs(tmp_path):
    write(tmp_path, "pkg/a.py")
    write(tmp_path, "pkg/sub/b.py")
    write(tmp_path, "pkg/notes.txt", "not python\n")
    write(tmp_path, "other/c.py")
    spec = BoundarySpec(
        subject="demo",
        boundaries=(
            Boundary("core", ("pkg",)),
            Boundary("other", ("other/c.py",)),
        ),
    )
    components = derive_components(tmp_path, spec)
    by_id = {component.id: component for component in components}
    assert by_id["demo:core"].paths == ("pkg/a.py", "pkg/sub/b.py")
    assert by_id["demo:other"].paths == ("other/c.py",)


def test_glob_pattern_expands_recursively(tmp_path):
    write(tmp_path, "pkg/a.py")
    write(tmp_path, "pkg/sub/b.py")
    spec = BoundarySpec("demo", (Boundary("globbed", ("pkg/**/*.py",)),))
    components = derive_components(tmp_path, spec)
    assert components[0].paths == ("pkg/a.py", "pkg/sub/b.py")


def test_components_have_stable_ids_and_sorted_paths(tmp_path):
    write(tmp_path, "pkg/b.py")
    write(tmp_path, "pkg/a.py")
    write(tmp_path, "other/c.py")
    spec = BoundarySpec(
        subject="demo",
        boundaries=(
            Boundary("z_other", ("other/c.py",)),
            Boundary("core", ("pkg",)),
        ),
    )
    components = derive_components(tmp_path, spec)
    assert [c.id for c in components] == ["demo:core", "demo:z_other"]
    assert components[0].paths == ("pkg/a.py", "pkg/b.py")


def test_boundary_order_does_not_change_output(tmp_path):
    write(tmp_path, "a/x.py")
    write(tmp_path, "b/y.py")
    first = derive_components(
        tmp_path,
        BoundarySpec(
            "demo",
            (Boundary("a", ("a",)), Boundary("b", ("b",))),
        ),
    )
    second = derive_components(
        tmp_path,
        BoundarySpec(
            "demo",
            (Boundary("b", ("b",)), Boundary("a", ("a",))),
        ),
    )
    assert first == second


def test_exclude_patterns_remove_files_and_prefixes(tmp_path):
    write(tmp_path, "pkg/a.py")
    write(tmp_path, "pkg/skip.py")
    write(tmp_path, "pkg/gen/z.py")
    spec = BoundarySpec(
        subject="demo",
        boundaries=(Boundary("core", ("pkg",)),),
        exclude=("pkg/skip.py", "pkg/gen"),
    )
    components = derive_components(tmp_path, spec)
    assert components[0].paths == ("pkg/a.py",)


def test_overlapping_boundaries_are_rejected(tmp_path):
    write(tmp_path, "pkg/a.py")
    spec = BoundarySpec(
        subject="demo",
        boundaries=(
            Boundary("all", ("pkg",)),
            Boundary("one", ("pkg/a.py",)),
        ),
    )
    with pytest.raises(ComponentError, match="overlap"):
        derive_components(tmp_path, spec)


def test_missing_boundary_path_is_rejected(tmp_path):
    spec = BoundarySpec("demo", (Boundary("missing", ("nope",)),))
    with pytest.raises(ComponentError, match="does not exist"):
        derive_components(tmp_path, spec)


def test_boundary_with_no_python_files_is_rejected(tmp_path):
    write(tmp_path, "pkg/readme.md", "text\n")
    spec = BoundarySpec("demo", (Boundary("docs", ("pkg",)),))
    with pytest.raises(ComponentError, match="no Python files"):
        derive_components(tmp_path, spec)


def test_duplicate_boundary_names_are_rejected(tmp_path):
    write(tmp_path, "a/x.py")
    write(tmp_path, "b/y.py")
    spec = BoundarySpec(
        subject="demo",
        boundaries=(Boundary("core", ("a",)), Boundary("core", ("b",))),
    )
    with pytest.raises(ComponentError, match="duplicate"):
        derive_components(tmp_path, spec)


def test_invalid_boundary_name_is_rejected(tmp_path):
    write(tmp_path, "a/x.py")
    spec = BoundarySpec("demo", (Boundary("bad:name", ("a",)),))
    with pytest.raises(ComponentError, match="name"):
        derive_components(tmp_path, spec)


def test_path_escape_is_rejected(tmp_path):
    write(tmp_path, "../outside.py")
    spec = BoundarySpec("demo", (Boundary("bad", ("../outside.py",)),))
    with pytest.raises(ComponentError, match="escapes"):
        derive_components(tmp_path, spec)


def test_measure_components_returns_k_per_component(tmp_path):
    write(tmp_path, "a/x.py", "x = 1\ny = 2\n")
    write(tmp_path, "b/z.py", "z = 3\n")
    components = derive_components(
        tmp_path,
        BoundarySpec(
            "demo",
            (Boundary("a", ("a",)), Boundary("b", ("b",))),
        ),
    )
    measured = measure_components(tmp_path, components)
    assert measured["demo:a"].k_lines_nb == 2
    assert measured["demo:b"].k_lines_nb == 1
