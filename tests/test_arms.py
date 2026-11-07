from __future__ import annotations

import pytest

from evo.domain.arms import (
    ARM_CONFIGS,
    ARMS,
    ArmConfig,
    ArmError,
    arm_config,
    arms_table,
)


def test_the_six_arms_match_the_preregistered_table():
    assert ARMS == ("H0", "EVO", "SIM", "COMBO", "PLACEBO", "SHARP")
    assert set(ARM_CONFIGS) == set(ARMS)
    assert arm_config("H0").evolve is False
    assert arm_config("H0").simplify == "off"
    assert arm_config("EVO").evolve is True
    assert arm_config("EVO").simplify == "off"
    assert arm_config("SIM").simplify == "full"
    assert arm_config("COMBO").evolve is True
    assert arm_config("COMBO").simplify == "full"
    assert arm_config("PLACEBO").simplify == "random"
    assert arm_config("SHARP").simplify == "sharp"


def test_only_evo_and_combo_evolve():
    evolving = [name for name, config in ARM_CONFIGS.items() if config.evolve]
    assert evolving == ["EVO", "COMBO"]


def test_order_is_preregistered_the_same_for_every_arm():
    assert {config.order for config in ARM_CONFIGS.values()} == {"evolve_then_simplify"}


def test_unknown_arm_is_rejected():
    with pytest.raises(ArmError):
        arm_config("BASELINE")


def test_arm_config_rejects_invalid_fields():
    with pytest.raises(ArmError):
        ArmConfig(name="X", evolve=False, simplify="magic", order="evolve_then_simplify")
    with pytest.raises(ArmError):
        ArmConfig(name="X", evolve="yes", simplify="off", order="evolve_then_simplify")
    with pytest.raises(ArmError):
        ArmConfig(name="X", evolve=False, simplify="off", order="simplify_then_evolve")
    with pytest.raises(ArmError):
        ArmConfig(name="", evolve=False, simplify="off", order="evolve_then_simplify")


def test_arm_config_serializes():
    payload = arm_config("SIM").as_dict()
    assert set(payload) == {
        "arm",
        "evolve",
        "simplify",
        "order",
        "diagnose",
        "description",
    }
    assert payload["arm"] == "SIM"
    assert payload["simplify"] == "full"
    assert payload["diagnose"] is False
    assert payload["description"]


def test_arms_table_preserves_the_canonical_order():
    rows = arms_table()
    assert [row["arm"] for row in rows] == list(ARMS)
    assert all(isinstance(row["description"], str) for row in rows)
