from __future__ import annotations

import pytest

from evo.domain.critic import CriticError, precheck


def test_credential_is_rejected():
    assert precheck("api_key = 'sk-abcdefghijklmnopqrstuvwx'") == (
        "credential in diff",
    )


def test_google_key_is_rejected():
    text = "key = 'AIza" + "a" * 35 + "'"
    assert "credential in diff" in precheck(text)


def test_test_path_is_rejected():
    assert "test path in diff" in precheck("path = '/tests/test_fix.py'")


def test_grader_artifact_is_rejected():
    assert "grader artifact in diff" in precheck("open('test-stdout.txt')")


def test_grader_directory_is_rejected():
    assert "grader path in diff" in precheck("path = 'verifier/output.json'")


def test_explicit_task_term_is_rejected():
    assert "task term in diff: fix-git" in precheck(
        "if task == 'fix-git':", ("fix-git",)
    )


def test_clean_diff_passes():
    assert precheck("def solve(x):\n    return x + 1\n") == ()


def test_non_string_diff_is_rejected():
    with pytest.raises(CriticError):
        precheck(None)  # type: ignore[arg-type]


def test_custom_pattern_is_supported():
    reasons = precheck("SECRET_MARKER", patterns=((r"SECRET_MARKER", "custom leak"),))
    assert reasons == ("custom leak",)
