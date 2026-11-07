from __future__ import annotations

import sys
import textwrap

import pytest

from evo.domain.verify import (
    CheckOutcome,
    Observation,
    VerifyError,
    aggregate,
    compile_files,
    differential_check,
    import_check,
    replay_check,
    required_checks,
)


def write(root, name: str, source: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


# ----------------------------------------------------------------- compile


def test_compile_passes_for_importable_sources(tmp_path):
    write(tmp_path, "pkg/a.py", "def add(a, b):\n    return a + b\n")
    write(tmp_path, "pkg/b.py", "from .a import add\n")
    outcome = compile_files(tmp_path, ["pkg/a.py", "pkg/b.py"])
    assert outcome.status == "pass"


def test_compile_reports_a_syntax_error(tmp_path):
    write(tmp_path, "pkg/broken.py", "def add(a, b):\nreturn a + b\n")
    outcome = compile_files(tmp_path, ["pkg/broken.py"])
    assert outcome.status == "fail"
    assert "broken.py" in outcome.detail


def test_compile_reports_every_broken_file(tmp_path):
    write(tmp_path, "a.py", "def broken(:\n")
    write(tmp_path, "b.py", "class X(:\n")
    outcome = compile_files(tmp_path, ["a.py", "b.py"])
    assert outcome.status == "fail"
    assert "a.py" in outcome.detail and "b.py" in outcome.detail


def test_compile_fails_on_a_missing_file(tmp_path):
    outcome = compile_files(tmp_path, ["missing.py"])
    assert outcome.status == "fail"
    assert "missing.py" in outcome.detail


def test_compile_fails_on_undecodable_bytes(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00")
    assert compile_files(tmp_path, ["bad.py"]).status == "fail"


def test_compile_rejects_absolute_paths_outside_the_root(tmp_path):
    with pytest.raises(VerifyError):
        compile_files(tmp_path, [str(tmp_path.parent / "elsewhere.py")])


# ------------------------------------------------------------------ import


def test_import_passes_for_a_real_package(tmp_path):
    write(
        tmp_path,
        "probe/__init__.py",
        """
        VALUE = 42

        def identity(value):
            return value
        """,
    )
    outcome = import_check(tmp_path, ["probe"])
    assert outcome.status == "pass"


def test_import_fails_for_a_missing_module(tmp_path):
    outcome = import_check(tmp_path, ["definitely_not_a_module"])
    assert outcome.status == "fail"
    assert "ModuleNotFoundError" in outcome.detail


def test_import_fails_when_the_module_raises(tmp_path):
    write(tmp_path, "boom/__init__.py", "raise RuntimeError('broken at import')\n")
    outcome = import_check(tmp_path, ["boom"])
    assert outcome.status == "fail"
    assert "broken at import" in outcome.detail


def test_import_is_unknown_when_the_interpreter_is_missing(tmp_path):
    outcome = import_check(tmp_path, ["probe"], python="definitely-missing-python-xyz")
    assert outcome.status == "unknown"


def test_import_is_unknown_on_timeout(tmp_path):
    write(tmp_path, "slow/__init__.py", "import time\ntime.sleep(10)\n")
    outcome = import_check(tmp_path, ["slow"], timeout=0.5)
    assert outcome.status == "unknown"


def test_import_rejects_malformed_module_names(tmp_path):
    with pytest.raises(VerifyError):
        import_check(tmp_path, ["os; os.system('echo pwned')"])
    with pytest.raises(VerifyError):
        import_check(tmp_path, [""])


# ------------------------------------------------------------------ replay


def case(name: str, reference, candidate, recorded: bool = True) -> Observation:
    return Observation(case_id=name, recorded=recorded, reference=reference, candidate=candidate)


def test_replay_is_unknown_below_the_required_case_count():
    cases = tuple(case(f"task-{index}", {"reward": 0}, {"reward": 0}) for index in range(9))
    outcome = replay_check(cases, minimum=10)
    assert outcome.status == "unknown"
    assert "9" in outcome.detail


def test_replay_is_unknown_when_a_case_has_no_record():
    cases = tuple(case(f"task-{index}", {"reward": 0}, {"reward": 0}) for index in range(9))
    cases += (case("task-9", None, None, recorded=False),)
    outcome = replay_check(cases, minimum=10)
    assert outcome.status == "unknown"
    assert "task-9" in outcome.detail


def test_replay_passes_when_all_cases_reproduce():
    cases = tuple(case(f"task-{index}", {"reward": 0}, {"reward": 0}) for index in range(10))
    assert replay_check(cases, minimum=10).status == "pass"


def test_replay_fails_when_behavior_changes():
    cases = tuple(case(f"task-{index}", {"reward": 0}, {"reward": 0}) for index in range(9))
    cases += (case("task-9", {"reward": 0}, {"reward": 1}),)
    outcome = replay_check(cases, minimum=10)
    assert outcome.status == "fail"
    assert "task-9" in outcome.detail


# ------------------------------------------------------------ differential


def test_differential_is_unknown_without_recorded_inputs():
    outcome = differential_check(())
    assert outcome.status == "unknown"
    assert "recorded" in outcome.detail


def test_differential_is_unknown_when_a_case_was_not_captured():
    outcome = differential_check((case("op-1", None, None, recorded=False),))
    assert outcome.status == "unknown"
    assert "op-1" in outcome.detail


def test_differential_passes_on_identical_json_outputs():
    outcome = differential_check(
        (
            case("op-1", {"text": "a", "n": 1}, {"n": 1, "text": "a"}),
            case("op-2", [1, 2, 3], [1, 2, 3]),
        )
    )
    assert outcome.status == "pass"


def test_differential_fails_on_a_changed_output():
    outcome = differential_check((case("op-1", {"text": "a"}, {"text": "b"}),))
    assert outcome.status == "fail"
    assert "op-1" in outcome.detail


def test_differential_distinguishes_bool_from_int():
    assert differential_check((case("op-1", 1, True),)).status == "fail"


def test_observations_reject_duplicate_ids():
    with pytest.raises(VerifyError):
        differential_check((case("op-1", 1, 1), case("op-1", 1, 1)))


# --------------------------------------------------------------- aggregate


def outcome(name: str, status: str) -> CheckOutcome:
    return CheckOutcome(name=name, status=status)


def test_required_checks_per_layer():
    assert required_checks("L1") == ("compile", "import")
    assert required_checks("L2") == ("compile", "import", "smoke")
    assert required_checks("L3") == ("compile", "import", "smoke", "replay")
    assert required_checks("L4") == ("compile", "import", "smoke", "replay", "differential")
    with pytest.raises(VerifyError):
        required_checks("L9")


def test_aggregate_passes_only_when_every_required_check_passes():
    report = aggregate(
        [outcome("compile", "pass"), outcome("import", "pass"), outcome("smoke", "pass")],
        ("compile", "import", "smoke"),
    )
    assert report.verdict == "pass"


def test_aggregate_is_unknown_when_a_required_check_is_missing():
    report = aggregate([outcome("compile", "pass")], ("compile", "import"))
    assert report.verdict == "unknown"


def test_aggregate_is_unknown_when_a_required_check_is_unknown():
    report = aggregate(
        [outcome("compile", "pass"), outcome("replay", "unknown")],
        ("compile", "replay"),
    )
    assert report.verdict == "unknown"


def test_aggregate_fail_wins_over_unknown():
    report = aggregate(
        [outcome("compile", "fail"), outcome("replay", "unknown")],
        ("compile", "replay"),
    )
    assert report.verdict == "fail"


def test_aggregate_fails_on_an_extra_failed_check():
    report = aggregate(
        [outcome("compile", "pass"), outcome("differential", "fail")],
        ("compile",),
    )
    assert report.verdict == "fail"


def test_aggregate_ignores_extra_unknown_checks():
    report = aggregate(
        [outcome("compile", "pass"), outcome("differential", "unknown")],
        ("compile",),
    )
    assert report.verdict == "pass"


def test_aggregate_rejects_bad_inputs():
    with pytest.raises(VerifyError):
        CheckOutcome(name="compile", status="maybe")
    with pytest.raises(VerifyError):
        aggregate([outcome("compile", "pass")], ())
    with pytest.raises(VerifyError):
        aggregate([outcome("compile", "pass"), outcome("compile", "fail")], ("compile",))


def test_report_serializes_for_the_cli():
    report = aggregate(
        [outcome("compile", "pass"), outcome("smoke", "unknown")],
        ("compile", "smoke"),
    )
    payload = report.as_dict()
    assert payload["verdict"] == "unknown"
    assert [item["name"] for item in payload["outcomes"]] == ["compile", "smoke"]


def test_compile_success_carries_no_detail(tmp_path):
    write(tmp_path, "pkg/a.py", "import sys\nprint(sys.version)\n")
    assert compile_files(tmp_path, ["pkg/a.py"]).detail == ""


def test_import_uses_the_requested_interpreter(tmp_path):
    write(tmp_path, "probe2/__init__.py", "VALUE = 2\n")
    assert import_check(tmp_path, ["probe2"], python=sys.executable).status == "pass"
