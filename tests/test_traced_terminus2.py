from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import evo.adapters.traced_terminus2 as traced
from evo.adapters.traced_terminus2 import (
    coverage_task_id,
    safe_write_trial,
    trial_name,
    write_trial,
)


class FakeCollector:
    def __init__(self, traces):
        self._traces = traces

    def traces(self):
        return self._traces


def test_trial_name_prefers_the_trial_directory():
    environment = SimpleNamespace(
        trial_paths=SimpleNamespace(
            trial_dir=Path("runs/job/chess-best-move__abc123")
        ),
        session_id="ignored__env",
    )

    assert trial_name(environment) == "chess-best-move__abc123"


def test_trial_name_falls_back_to_the_session_id():
    environment = SimpleNamespace(session_id="fix-git__abc123__env")

    assert trial_name(environment) == "fix-git__abc123"


def test_write_trial_writes_relative_paths(tmp_path):
    root = tmp_path / "terminus2"
    root.mkdir()
    source = root / "pkg" / "mod.py"
    source.parent.mkdir()
    source.write_text("a = 1\n", encoding="utf-8")
    collector = FakeCollector({"task__abc": {str(source): {1, 2}}})

    target = write_trial(collector, root, tmp_path / "traces", "task__abc")

    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["task_ids"] == ["task"]
    assert payload["tasks"]["task"] == {"pkg/mod.py": [1, 2]}


def test_coverage_task_id_strips_the_trial_hash():
    assert coverage_task_id("chess-best-move__EyxMeKb") == "chess-best-move"


def test_write_trial_returns_none_for_an_unseen_trial(tmp_path):
    root = tmp_path / "terminus2"
    root.mkdir()
    collector = FakeCollector({})

    assert write_trial(collector, root, tmp_path / "traces", "missing") is None


class BrokenCollector:
    def traces(self):
        raise RuntimeError("boom")


def test_safe_write_trial_does_not_raise(tmp_path, capsys):
    root = tmp_path / "terminus2"
    root.mkdir()

    assert safe_write_trial(BrokenCollector(), root, tmp_path / "traces", "t") is None
    assert "coverage write failed" in capsys.readouterr().err


def test_collector_failure_disables_tracing(monkeypatch, tmp_path, capsys):
    class BrokenLineCollector:
        def __init__(self, roots):
            pass

        def start(self):
            raise RuntimeError("monitoring unavailable")

    monkeypatch.setattr(traced, "_TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(traced, "_SUBJECT_ROOT", tmp_path / "terminus2")
    monkeypatch.setattr(traced, "_COLLECTOR", None)
    monkeypatch.setattr(traced, "LineCollector", BrokenLineCollector)

    assert traced.collector() is None
    assert "coverage disabled" in capsys.readouterr().err
