from __future__ import annotations

import pytest

from evo.contracts.types import Component
from evo.domain.protection import ProtectionPolicy
from evo.domain.scan import ScanError, scan_sources


COMPONENT = Component(id="demo:pkg", name="pkg", paths=("a.py", "b.py"))


def scan(sources, policy=None, paths=("a.py",)):
    component = Component(id="demo:pkg", name="pkg", paths=tuple(paths))
    return scan_sources(component, dict(sources), policy or ProtectionPolicy())


def symbols(candidates, kind=None):
    return {
        candidate.symbol
        for candidate in candidates
        if kind is None or candidate.kind == kind
    }


def test_unused_import_is_a_candidate_and_used_import_is_not():
    src = "import os\nimport sys\n\nprint(sys)\n"
    candidates = scan({"a.py": src})
    assert symbols(candidates, "unused_import") == {"os"}


def test_partially_used_import_statement_is_skipped():
    src = "import os, sys\n\nprint(os)\n"
    assert scan({"a.py": src}) == []


def test_from_import_and_alias_binding():
    src = "from m import x\nfrom n import y as z\n\nprint(z)\n"
    assert symbols(scan({"a.py": src}), "unused_import") == {"x"}


def test_dunder_init_is_never_flagged_for_reexports():
    component = Component(id="demo:pkg", name="pkg", paths=("__init__.py",))
    candidates = scan_sources(component, {"__init__.py": "from m import x\n"}, ProtectionPolicy())
    assert candidates == []


def test_all_reexport_is_not_an_unused_import():
    src = 'from m import x\n\n__all__ = ["x"]\n'
    assert scan({"a.py": src}) == []


def test_star_import_disables_unused_import_detection():
    assert scan({"a.py": "from m import *\n"}) == []


def test_type_checking_file_is_skipped():
    src = "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from m import X\n"
    assert scan({"a.py": src}) == []


def test_import_inside_try_is_reported_but_blocked_by_protection():
    src = "try:\n    import os\nexcept ImportError:\n    os = None\n"
    candidates = scan({"a.py": src})
    assert symbols(candidates, "unused_import") == {"os"}
    assert candidates[0].protection
    assert candidates[0].blocked


def test_zero_reference_private_module_function():
    src = "def _dead():\n    return 1\n\ndef _used():\n    return 2\n\ndef run():\n    return _used()\n"
    assert symbols(scan({"a.py": src}), "zero_ref_function") == {"_dead"}


def test_private_methods_are_not_l1_candidates():
    src = "class A:\n    def _m(self):\n        return 1\n"
    assert scan({"a.py": src}) == []


def test_dunder_functions_are_not_l1_candidates():
    src = "def __helper__():\n    return 1\n"
    assert scan({"a.py": src}) == []


def test_decorated_zero_ref_function_is_reported_and_blocked():
    src = "@retry\ndef _dead():\n    return 1\n"
    candidates = scan({"a.py": src})
    assert symbols(candidates, "zero_ref_function") == {"_dead"}
    assert "decorator" in {hit.kind for hit in candidates[0].protection}


def test_cross_file_reference_prevents_zero_ref_candidate():
    sources = {
        "a.py": "def _f():\n    return 1\n",
        "b.py": "from a import _f\n\n_f()\n",
    }
    assert symbols(scan(sources, paths=("a.py", "b.py")), "zero_ref_function") == set()


def test_dynamic_lookup_disables_zero_ref_detection_for_component():
    sources = {
        "a.py": "def _dead():\n    return 1\n",
        "b.py": "def lookup(obj, name):\n    return getattr(obj, name)\n",
    }
    assert symbols(scan(sources, paths=("a.py", "b.py")), "zero_ref_function") == set()


def test_literal_getattr_counts_as_a_reference():
    src = "def _dead():\n    return 1\n\ndef lookup(obj):\n    return getattr(obj, '_dead')\n"
    assert "_dead" not in symbols(scan({"a.py": src}), "zero_ref_function")


def test_unreachable_statement_after_return_is_a_candidate():
    src = "def _f():\n    return 1\n    x = 2\n"
    unreachable = [c for c in scan({"a.py": src}) if c.kind == "unreachable"]
    assert symbols(unreachable) == {"<unreachable>"}
    assert unreachable[0].span == (3, 3)


def test_unreachable_code_with_assert_is_blocked():
    src = "def _f():\n    return 1\n    assert False\n"
    unreachable = [c for c in scan({"a.py": src}) if c.kind == "unreachable"]
    assert unreachable and unreachable[0].blocked


def test_no_candidates_for_clean_source():
    src = "def _f():\n    return 1\n\ndef run():\n    return _f()\n"
    assert scan({"a.py": src}) == []


def test_candidate_ids_are_deterministic_and_sorted():
    src = "import os\nimport sys\n\ndef _dead():\n    return 1\n"
    first = scan({"a.py": src})
    second = scan({"a.py": src})
    assert [c.id for c in first] == [c.id for c in second]
    assert first == second
    assert [c.id for c in first] == sorted(c.id for c in first)


def test_syntax_error_raises_scan_error():
    with pytest.raises(ScanError):
        scan({"a.py": "def broken(:\n"})
