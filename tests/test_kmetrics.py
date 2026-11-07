from __future__ import annotations

import pytest

from evo.domain.kmetrics import (
    KMeasurementError,
    count_ast_nodes,
    count_lines_nb,
    measure_paths,
    measure_source,
    measure_sources,
)


SOURCE = '''"""Module docstring.

Second line.
"""
import os  # inline

def foo(a,
        b):
    """Function docstring."""
    # comment
    if a:
        return 1
    return 2


def bar():
    pass


def baz():
    text = """assigned
string"""
    return text
'''


def test_lines_nb_excludes_blank_comments_and_docstrings():
    assert measure_source(SOURCE).k_lines_nb == 12


def test_lines_nb_counts_physical_lines_of_multiline_statement():
    assert count_lines_nb("x = (1 +\n     2)\n") == 2
    assert count_lines_nb("x = 1; y = 2\n") == 1


def test_lines_nb_of_comment_only_and_docstring_only_sources_is_zero():
    assert count_lines_nb("# only a comment\n\n") == 0
    assert count_lines_nb('"""only a docstring"""\n') == 0


def test_lines_nb_treats_standalone_string_expression_as_noise():
    src = "def f():\n    'no-op'\n    return 1\n"
    assert count_lines_nb(src) == 2


def test_lines_nb_rejects_invalid_syntax():
    with pytest.raises(KMeasurementError):
        count_lines_nb("def broken(:\n")


def test_ast_nodes_ignore_formatting_comments_and_names():
    compact = count_ast_nodes("x = 1 + 2\n")
    commented = count_ast_nodes("# comment\nx = 1 + 2\n")
    wrapped = count_ast_nodes("x = (1 +\n     2)\n")
    renamed = count_ast_nodes("a_much_longer_name = 1 + 2\n")
    assert compact == commented == wrapped == renamed


def test_ast_nodes_exclude_docstring_nodes():
    assert count_ast_nodes('"""module doc"""\n') == 0
    src = 'def f():\n    """doc"""\n    return 1\n'
    assert count_ast_nodes(src) == 4


def test_ast_nodes_fall_when_code_is_removed():
    before = count_ast_nodes("def f():\n    return 1\n\ndef g():\n    return 2\n")
    after = count_ast_nodes("def f():\n    return 1\n")
    assert after < before


def test_chars_and_bytes_are_distinct_for_non_ascii():
    vector = measure_source("中文\n")
    assert vector.k_chars == 3
    assert vector.k_bytes == 7


def test_cc_counts_branches_and_nested_blocks():
    src = '''class A:
    def m(self, x):
        if x:
            return 1
        return 0

def outer(x):
    def inner(y):
        if y:
            return 1
        return 0
    return inner(x)
'''
    # class 1 + method m 2 + outer 1 + closure inner 2
    assert measure_source(src).k_cc == 6


def test_gzip_is_deterministic_and_order_independent():
    one = measure_sources({"b.py": "x = 1\n", "a.py": "y = 2\n"})
    two = measure_sources({"a.py": "y = 2\n", "b.py": "x = 1\n"})
    assert one == two
    assert one.k_gzip == two.k_gzip
    assert one.files["a.py"] == measure_source("y = 2\n")


def test_aggregate_sums_additive_metrics_and_counts_each_file_once():
    result = measure_sources({"a.py": "x = 1\n", "b.py": "y = 2\n    \n"})
    assert result.k_lines_nb == 2
    assert result.k_chars == len("x = 1\n") + len("y = 2\n    \n")
    assert result.k_bytes == result.k_chars
    assert sorted(result.files) == ["a.py", "b.py"]


def test_empty_sources_are_all_zero():
    result = measure_sources({})
    assert result.k_lines_nb == 0
    assert result.k_ast == 0
    assert result.k_chars == 0
    assert result.k_bytes == 0
    assert result.k_gzip == 0
    assert result.k_cc == 0


def test_measure_paths_reads_utf8_relative_to_root(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    result = measure_paths(tmp_path, ["pkg/mod.py"])
    assert result.k_lines_nb == 1
    assert list(result.files) == ["pkg/mod.py"]


def test_vector_delta_reports_all_fields():
    before = measure_source("def f():\n    return 1\n")
    after = measure_source("")
    delta = before - after
    assert delta.k_lines_nb == 2
    assert delta.k_ast > 0
    assert delta.k_chars > 0
