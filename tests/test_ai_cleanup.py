from __future__ import annotations

import json

from evo.domain.ai_cleanup import find_ai_cleanup


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_trivial_wrapper_is_reported(tmp_path):
    write(
        tmp_path,
        "pkg/a.py",
        "def helper(x):\n"
        "    return x + 1\n"
        "\n"
        "\n"
        "def _wrapped(x):\n"
        "    return helper(x)\n",
    )

    findings = find_ai_cleanup(tmp_path)

    assert any(row.kind == "trivial_wrapper" and row.symbol == "_wrapped" for row in findings)


def test_single_call_helper_is_reported(tmp_path):
    write(
        tmp_path,
        "pkg/a.py",
        "def _only_once(x):\n"
        "    y = x + 1\n"
        "    return y\n"
        "\n"
        "\n"
        "def public(x):\n"
        "    return _only_once(x)\n",
    )

    findings = find_ai_cleanup(tmp_path)

    assert any(
        row.kind == "single_call_helper" and row.symbol == "_only_once"
        for row in findings
    )


def test_single_call_helper_is_not_flagged_when_called_from_another_file(tmp_path):
    write(
        tmp_path,
        "pkg/a.py",
        "def _shared(x):\n"
        "    y = x + 1\n"
        "    return y\n",
    )
    write(tmp_path, "pkg/b.py", "from pkg.a import _shared\n\ndef use(x):\n    return _shared(x)\n")

    findings = find_ai_cleanup(tmp_path)

    assert not any(
        row.kind == "single_call_helper" and row.symbol == "_shared"
        for row in findings
    )


def test_duplicate_function_bodies_are_reported(tmp_path):
    body = (
        "    a = value + 1\n"
        "    b = a * 2\n"
        "    return a + b\n"
    )
    write(
        tmp_path,
        "pkg/a.py",
        f"def first(value):\n{body}\n\ndef second(value):\n{body}",
    )

    findings = find_ai_cleanup(tmp_path)

    assert any(row.kind == "duplicate_body" for row in findings)


def test_unread_cli_option_is_reported(tmp_path):
    write(
        tmp_path,
        "cli.py",
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--used-flag')\n"
        "parser.add_argument('--unused-flag')\n"
        "args = parser.parse_args()\n"
        "print(args.used_flag)\n",
    )

    findings = find_ai_cleanup(tmp_path)

    assert any(
        row.kind == "unread_cli_option" and row.symbol == "unused_flag"
        for row in findings
    )
    assert not any(
        row.kind == "unread_cli_option" and row.symbol == "used_flag"
        for row in findings
    )


def test_findings_are_json_serializable(tmp_path):
    write(tmp_path, "pkg/a.py", "def _wrapped(x):\n    return helper(x)\n")

    payload = [row.as_dict() for row in find_ai_cleanup(tmp_path)]

    assert json.loads(json.dumps(payload)) == payload
