from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo.domain.incidents import (
    Incident,
    IncidentError,
    append_incident,
    read_incidents,
)


def incident(**overrides):
    payload = {
        "id": "inc-001",
        "version": "v1",
        "phase": "evaluation",
        "location": "vendor/terminus2/terminus_2.py:837",
        "component": "terminus2:summarize",
        "task_id": "terminal-bench/example",
        "trial_id": "trial-1",
        "symptom": "timeout after fixed wait",
        "consequence": "reward 0; tokens +40%",
        "cause": "fixed duration instead of polling",
        "fix": "poll with bounded retries",
        "status": "open",
        "evidence": ("runs/versions/v1/outcomes.jsonl",),
    }
    payload.update(overrides)
    return Incident(**payload)


def test_incident_roundtrip(tmp_path):
    path = tmp_path / "incidents.jsonl"
    append_incident(path, incident())

    rows = read_incidents(path)

    assert len(rows) == 1
    assert rows[0]["id"] == "inc-001"
    assert rows[0]["location"].endswith("terminus_2.py:837")
    assert rows[0]["consequence"].startswith("reward 0")


def test_incident_requires_location_and_consequence():
    with pytest.raises(IncidentError, match="location"):
        incident(location="")
    with pytest.raises(IncidentError, match="consequence"):
        incident(consequence="")


def test_duplicate_incident_id_is_rejected(tmp_path):
    path = tmp_path / "incidents.jsonl"
    append_incident(path, incident())

    with pytest.raises(IncidentError, match="duplicate"):
        append_incident(path, incident())
