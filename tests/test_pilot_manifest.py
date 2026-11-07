from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "specs" / "pilot_v1.json"
FROZEN = ROOT / "runs" / "frozen" / "tb21-h0" / "run.json"


@pytest.mark.skipif(not FROZEN.is_file(), reason="frozen run manifest is not present")
def test_pilot_tasks_are_disjoint_from_holdout():
    pilot = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run = json.loads(FROZEN.read_text(encoding="utf-8"))
    evolution = set(run["pools"]["evolution"])
    holdout = set(run["pools"]["holdout"])
    smoke = set(run["pools"]["smoke"])
    dev = set(pilot["pilot_dev"])
    validation = set(pilot["pilot_validation"])
    sensitivity = set(pilot["sensitivity_smoke"])

    assert dev <= evolution
    assert validation <= evolution
    assert sensitivity <= smoke
    assert not dev & validation
    assert not dev & holdout
    assert not validation & holdout
    assert not sensitivity & dev
    assert not sensitivity & validation
    assert len(holdout) == pilot["holdout"]["size"]
