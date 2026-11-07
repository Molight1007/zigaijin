from __future__ import annotations

import pytest

from evo.domain.protection import (
    ProtectionError,
    ProtectionPolicy,
    check_protection,
)


def kinds(source, path="m.py", span=None, policy=None):
    hits = check_protection(source, path, policy or ProtectionPolicy(), span)
    return {hit.kind for hit in hits}


def test_clean_private_function_has_no_hits():
    assert kinds("def _f(x):\n    return x + 1\n") == set()


def test_assert_and_raise_are_protected():
    assert "assert" in kinds("def _f(x):\n    assert x\n    return x\n")
    assert "raise" in kinds("def _f():\n    raise ValueError('no')\n")


def test_try_structure_is_protected():
    src = "def _f():\n    try:\n        return 1\n    except ValueError:\n        return 0\n"
    assert "try" in kinds(src)


def test_decorated_private_function_is_protected():
    src = "@retry\ndef _f():\n    return 1\n"
    assert "decorator" in kinds(src)


def test_public_function_and_class_are_protected():
    assert "public_signature" in kinds("def f():\n    return 1\n")
    assert "public_signature" in kinds("class A:\n    pass\n")
    assert "public_signature" in kinds("class A:\n    def m(self):\n        return 1\n")
    assert kinds("def _f():\n    return 1\n") == set()
    assert kinds("class _A:\n    pass\n") == set()


def test_async_public_function_is_protected():
    assert "public_signature" in kinds("async def f():\n    return 1\n")


def test_all_assignment_is_protected():
    assert "public_api" in kinds('__all__ = ["_f"]\n')


def test_main_guard_and_main_function_are_protected():
    guard = 'if __name__ == "__main__":\n    run()\n'
    assert "entry_point" in kinds(guard)
    assert "entry_point" in kinds("def main():\n    return 1\n")


def test_protected_names_from_policy_are_hits():
    policy = ProtectionPolicy(protected_names=("_keep",))
    assert "protected_name" in kinds("def _keep():\n    return 1\n", policy=policy)


def test_frozen_file_is_protected_regardless_of_span():
    policy = ProtectionPolicy(frozen_files={"m.py": "MODEL_CONNECTION patch"})
    hits = check_protection("x = 1\n", "m.py", policy, span=(1, 1))
    assert [hit.kind for hit in hits] == ["frozen_patch"]
    assert "MODEL_CONNECTION" in hits[0].reason


def test_span_filter_only_returns_intersecting_hits():
    src = "def _f(x):\n    assert x\n    y = x + 1\n    return y\n"
    assert "assert" not in kinds(src, span=(3, 4))
    assert "assert" in kinds(src, span=(2, 4))
    assert "assert" in kinds(src, span=(1, 4))


def test_invalid_source_raises():
    with pytest.raises(ProtectionError):
        check_protection("def broken(:\n", "m.py", ProtectionPolicy())


def test_hits_carry_path_and_line_numbers():
    hits = check_protection("def f():\n    return 1\n", "pkg/m.py", ProtectionPolicy())
    assert hits
    assert all(hit.path == "pkg/m.py" for hit in hits)
    assert all(hit.start >= 1 and hit.end >= hit.start for hit in hits)
