import asyncio
import importlib.util
import json
import sys
import textwrap

import pytest

from evo.adapters.coverage_py import CoverageError, LineCollector, relativize
from evo.domain.census import (
    CensusError,
    function_units,
    load_traces,
    touch_report,
    zero_touch,
)


def write_module(tmp_path, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


SUBJECT = """
def called():
    first = 1
    second = 2
    return first + second


def never_called():
    hidden = 3
    return hidden


class Tool:
    def method(self):
        value = 4
        return value


async def async_called():
    await _sleep()
    return 5


async def _sleep():
    import asyncio

    await asyncio.sleep(0)
"""


# ---------------------------------------------------------------- census


def test_function_units_qualified_names_and_body_spans(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    units = function_units({str(path): path.read_text(encoding="utf-8")})

    by_name = {unit.qualname: unit for unit in units}
    assert set(by_name) == {"called", "never_called", "Tool.method", "async_called", "_sleep"}

    called = by_name["called"]
    assert called.path == str(path)
    assert called.span == (3, 5)
    assert called.def_line == 2

    assert by_name["Tool.method"].span == (15, 16)
    assert by_name["async_called"].span == (20, 21)


def test_function_units_nested_qualname(tmp_path):
    source = """
def outer():
    def inner():
        return 1
    return inner
"""
    path = write_module(tmp_path, "nested", source)
    units = function_units({str(path): path.read_text(encoding="utf-8")})
    assert [unit.qualname for unit in units] == ["outer", "outer.inner"]


def test_function_units_full_span_includes_decorators(tmp_path):
    source = """
@decorator
@other(arg=1)
def decorated():
    return 1
"""
    path = write_module(tmp_path, "deco", source)
    unit = function_units({str(path): path.read_text(encoding="utf-8")})[0]
    assert unit.full_span == (2, 5)
    assert unit.span == (5, 5)
    assert unit.def_line == 4


def test_function_units_syntax_error_names_path(tmp_path):
    path = write_module(tmp_path, "broken", "def nope(:\n    pass\n")
    with pytest.raises(CensusError, match="broken.py"):
        function_units({str(path): path.read_text(encoding="utf-8")})


def test_touch_report_counts_only_body_lines(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    key = str(path)
    units = function_units({key: path.read_text(encoding="utf-8")})

    traces = {
        "t1": {key: {2}},  # def line only: not a body touch
        "t2": {key: {3, 4}},
    }
    report = {row.unit.qualname: row for row in touch_report(units, traces, ("t1", "t2"))}

    assert report["called"].tasks == 1
    assert report["called"].task_ids == ("t2",)
    assert report["never_called"].tasks == 0


def test_touch_report_span_end_is_inclusive(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    key = str(path)
    units = function_units({key: path.read_text(encoding="utf-8")})
    report = {row.unit.qualname: row for row in touch_report(units, {"t1": {key: {5}}}, ("t1",))}
    assert report["called"].tasks == 1


def test_touch_report_lines_between_functions_ignored(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    key = str(path)
    units = function_units({key: path.read_text(encoding="utf-8")})
    report = {row.unit.qualname: row for row in touch_report(units, {"t1": {key: {1, 7, 8}}}, ("t1",))}
    assert all(row.tasks == 0 for row in report.values())


def test_touch_report_rejects_duplicate_tasks(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    units = function_units({str(path): path.read_text(encoding="utf-8")})
    with pytest.raises(CensusError, match="duplicate"):
        touch_report(units, {}, ("t1", "t1"))


def test_touch_report_rejects_empty_tasks(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    units = function_units({str(path): path.read_text(encoding="utf-8")})
    with pytest.raises(CensusError, match="task"):
        touch_report(units, {}, ())


def test_touch_report_rejects_unknown_trace_task(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    units = function_units({str(path): path.read_text(encoding="utf-8")})
    with pytest.raises(CensusError, match="unknown"):
        touch_report(units, {"ghost": {}}, ("t1",))


def test_zero_touch_preserves_unit_order(tmp_path):
    path = write_module(tmp_path, "mod", SUBJECT)
    key = str(path)
    units = function_units({key: path.read_text(encoding="utf-8")})
    report = touch_report(units, {"t1": {key: {3, 4, 14, 15}}}, ("t1",))
    dead = zero_touch(report)
    assert [row.unit.qualname for row in dead] == ["never_called", "async_called", "_sleep"]


def test_one_line_function_counts_definition_as_touch(tmp_path):
    source = "def inline(): return 1\n\ndef multi():\n    return 2\n"
    path = write_module(tmp_path, "one_line", source)
    key = str(path)
    units = function_units({key: source})
    report = {row.unit.qualname: row for row in touch_report(units, {"t1": {key: {1}}}, ("t1",))}
    assert report["inline"].tasks == 1


# ------------------------------------------------------------- traces


def write_traces(tmp_path, payload):
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_traces_payload():
    return {
        "subject": "terminus2",
        "root": "vendor/terminus2",
        "task_ids": ["t1", "t2"],
        "tasks": {
            "t1": {"agent.py": [3, 4]},
            "t2": {"agent.py": [4], "tmux.py": [10]},
        },
    }


def test_load_traces_valid(tmp_path):
    traces = load_traces(write_traces(tmp_path, valid_traces_payload()))
    assert traces.subject == "terminus2"
    assert traces.root == "vendor/terminus2"
    assert traces.task_ids == ("t1", "t2")
    assert traces.tasks["t1"]["agent.py"] == frozenset({3, 4})
    assert traces.tasks["t2"]["tmux.py"] == frozenset({10})


def test_load_traces_missing_file(tmp_path):
    with pytest.raises(CensusError, match="cannot read"):
        load_traces(tmp_path / "absent.json")


def test_load_traces_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CensusError, match="JSON"):
        load_traces(path)


def test_load_traces_unknown_top_key(tmp_path):
    payload = valid_traces_payload()
    payload["extra"] = 1
    with pytest.raises(CensusError, match="unknown"):
        load_traces(write_traces(tmp_path, payload))


def test_load_traces_missing_task_ids(tmp_path):
    payload = valid_traces_payload()
    del payload["task_ids"]
    with pytest.raises(CensusError, match="task_ids"):
        load_traces(write_traces(tmp_path, payload))


def test_load_traces_task_set_mismatch(tmp_path):
    payload = valid_traces_payload()
    payload["tasks"]["t3"] = {}
    with pytest.raises(CensusError, match="task"):
        load_traces(write_traces(tmp_path, payload))


def test_load_traces_duplicate_task_ids(tmp_path):
    payload = valid_traces_payload()
    payload["task_ids"] = ["t1", "t1"]
    payload["tasks"] = {"t1": {}}
    with pytest.raises(CensusError, match="duplicate"):
        load_traces(write_traces(tmp_path, payload))


def test_load_traces_rejects_bad_lines(tmp_path):
    for bad in (0, -3, 2.5, True, "4"):
        payload = valid_traces_payload()
        payload["tasks"]["t1"]["agent.py"] = [bad]
        with pytest.raises(CensusError, match="line"):
            load_traces(write_traces(tmp_path, payload))


def test_load_traces_empty_task_ids(tmp_path):
    payload = valid_traces_payload()
    payload["task_ids"] = []
    payload["tasks"] = {}
    with pytest.raises(CensusError, match="task"):
        load_traces(write_traces(tmp_path, payload))


# --------------------------------------------------------------- collector


def test_collector_attributes_lines_per_task(tmp_path):
    path = write_module(tmp_path, "subject_collect", SUBJECT)
    module = import_module(path, "subject_collect")
    collector = LineCollector(roots=[tmp_path])
    collector.start()
    try:
        with collector.capture("t1"):
            module.called()
        with collector.capture("t2"):
            module.never_called()
        traces = collector.traces()
    finally:
        collector.stop()

    assert set(traces) == {"t1", "t2"}
    assert 3 in traces["t1"][str(path)]
    assert 3 not in traces["t2"][str(path)]
    assert 10 in traces["t2"][str(path)]


def test_collector_filters_out_of_scope_files(tmp_path):
    scope = tmp_path / "scope"
    scope.mkdir()
    subject = write_module(scope, "inside", "def f():\n    return 1\n")
    other = write_module(tmp_path, "outside", "def g():\n    return 2\n")
    inside = import_module(subject, "inside_mod")
    outside = import_module(other, "outside_mod")

    collector = LineCollector(roots=[scope])
    collector.start()
    try:
        with collector.capture("t1"):
            inside.f()
            outside.g()
        traces = collector.traces()
    finally:
        collector.stop()

    assert str(subject) in traces["t1"]
    assert str(other) not in traces["t1"]


def test_collector_async_concurrency_keeps_tasks_separate(tmp_path):
    source = """
import asyncio

async def slow_a():
    await asyncio.sleep(0.01)
    marker_a = 1
    return marker_a

async def slow_b():
    await asyncio.sleep(0.005)
    marker_b = 2
    return marker_b
"""
    path = write_module(tmp_path, "async_subject", source)
    module = import_module(path, "async_subject")
    collector = LineCollector(roots=[tmp_path])
    collector.start()

    async def main():
        async def run(task_id, fn):
            with collector.capture(task_id):
                await fn()

        await asyncio.gather(run("a", module.slow_a), run("b", module.slow_b))

    try:
        asyncio.run(main())
        traces = collector.traces()
    finally:
        collector.stop()

    assert 6 in traces["a"][str(path)]
    assert 6 not in traces["b"][str(path)]
    assert 11 in traces["b"][str(path)]
    assert 11 not in traces["a"][str(path)]


def test_collector_records_task_without_executed_lines(tmp_path):
    path = write_module(tmp_path, "idle_subject", "def f():\n    return 1\n")
    import_module(path, "idle_subject")
    collector = LineCollector(roots=[tmp_path])
    collector.start()
    try:
        with collector.capture("idle"):
            pass
        traces = collector.traces()
    finally:
        collector.stop()
    assert traces == {"idle": {}}


def test_collector_capture_requires_start(tmp_path):
    collector = LineCollector(roots=[tmp_path])
    with pytest.raises(CoverageError, match="not started"):
        with collector.capture("t1"):
            pass


def test_collector_snapshot_is_immutable(tmp_path):
    path = write_module(tmp_path, "snap_subject", "def f():\n    return 1\n")
    module = import_module(path, "snap_subject")
    collector = LineCollector(roots=[tmp_path])
    collector.start()
    try:
        with collector.capture("t1"):
            module.f()
        first = collector.traces()
        with collector.capture("t1"):
            module.f()
        second = collector.traces()
    finally:
        collector.stop()

    assert isinstance(first["t1"][str(path)], frozenset)
    assert first is not second


def test_collector_restarts_after_stop(tmp_path):
    path = write_module(tmp_path, "restart_subject", "def f():\n    return 1\n")
    module = import_module(path, "restart_subject")
    collector = LineCollector(roots=[tmp_path])
    collector.start()
    with collector.capture("t1"):
        module.f()
    collector.stop()

    importlib.invalidate_caches()
    collector.start()
    try:
        with collector.capture("t2"):
            module.f()
        traces = collector.traces()
    finally:
        collector.stop()

    assert 2 in traces["t2"][str(path)]


def test_collector_reports_tool_conflict():
    occupied = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(occupied, "other-tool")
    try:
        with pytest.raises(CoverageError, match="tool"):
            LineCollector(roots=[]).start()
    finally:
        sys.monitoring.free_tool_id(occupied)


def test_relativize_uses_posix_separators(tmp_path):
    scope = tmp_path / "scope"
    scope.mkdir()
    file = scope / "pkg" / "mod.py"
    file.parent.mkdir()
    file.write_text("x = 1\n", encoding="utf-8")
    traces = {"t1": {str(file): frozenset({1})}}

    relative = relativize(traces, scope)
    assert relative == {"t1": {"pkg/mod.py": frozenset({1})}}


def test_relativize_rejects_outside_root(tmp_path):
    scope = tmp_path / "scope"
    scope.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(CoverageError, match="outside"):
        relativize({"t1": {str(outside): frozenset({1})}}, scope)
