from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ARMS = ("H0", "EVO", "SIM", "COMBO", "PLACEBO", "SHARP")
SIMPLIFY_MODES = ("off", "static", "census", "ablate", "full", "random", "sharp")
ORDER = "evolve_then_simplify"


class ArmError(ValueError):
    """An arm configuration violates the pre-registered design."""


@dataclass(frozen=True)
class ArmConfig:
    name: str
    evolve: bool
    simplify: str
    order: str = ORDER
    diagnose: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ArmError(f"arm name must be a non-empty string: {self.name!r}")
        if not isinstance(self.evolve, bool):
            raise ArmError(f"evolve must be a bool: {self.evolve!r}")
        if self.simplify not in SIMPLIFY_MODES:
            raise ArmError(
                f"unknown simplify mode {self.simplify!r}; "
                f"expected one of {', '.join(SIMPLIFY_MODES)}"
            )
        if self.order != ORDER:
            raise ArmError(
                f"order is pre-registered as {ORDER!r}, not {self.order!r}"
            )
        if not isinstance(self.diagnose, bool):
            raise ArmError(f"diagnose must be a bool: {self.diagnose!r}")
        if not isinstance(self.description, str):
            raise ArmError(f"description must be a string: {self.description!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.name,
            "evolve": self.evolve,
            "simplify": self.simplify,
            "order": self.order,
            "diagnose": self.diagnose,
            "description": self.description,
        }


ARM_CONFIGS: Mapping[str, ArmConfig] = {
    "H0": ArmConfig(
        name="H0",
        evolve=False,
        simplify="off",
        description="基线：不演化、不简化",
    ),
    "EVO": ArmConfig(
        name="EVO",
        evolve=True,
        simplify="off",
        description="自演化对照：只演化，复现 harness 膨胀",
    ),
    "SIM": ArmConfig(
        name="SIM",
        evolve=False,
        simplify="full",
        description="主臂：L1+L2+L3 简化，不演化",
    ),
    "COMBO": ArmConfig(
        name="COMBO",
        evolve=True,
        simplify="full",
        description="组合臂：演化 + 简化",
    ),
    "PLACEBO": ArmConfig(
        name="PLACEBO",
        evolve=False,
        simplify="random",
        description="安慰剂：随机删等量代码",
    ),
    "SHARP": ArmConfig(
        name="SHARP",
        evolve=False,
        simplify="sharp",
        description="同层对手：按模块消融删减",
    ),
}


def arm_config(name: str) -> ArmConfig:
    try:
        return ARM_CONFIGS[name]
    except KeyError:
        raise ArmError(f"unknown arm {name!r}; expected one of {', '.join(ARMS)}") from None


def arms_table() -> list[dict[str, object]]:
    return [ARM_CONFIGS[name].as_dict() for name in ARMS]
