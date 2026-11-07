from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping


ARMS = ("H0", "EVO", "SIM", "COMBO", "PLACEBO", "SHARP")


class ProtocolError(ValueError):
    """A protocol, pool split or pre-registration is invalid."""


@dataclass(frozen=True)
class Protocol:
    smoke: int = 2
    evolution: int = 60
    holdout: int = 27
    iteration_k: int = 3
    failure_repro_k: int = 10
    final_k: int = 5

    def __post_init__(self) -> None:
        for name in ("smoke", "evolution", "holdout", "iteration_k", "failure_repro_k", "final_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer: {value!r}")
        if self.total <= 0:
            raise ProtocolError("protocol must contain at least one task")
        if self.iteration_k == 0 or self.failure_repro_k == 0 or self.final_k == 0:
            raise ProtocolError("repetition counts must be positive")

    @property
    def total(self) -> int:
        return self.smoke + self.evolution + self.holdout


@dataclass(frozen=True)
class Pools:
    smoke: tuple[str, ...]
    evolution: tuple[str, ...]
    holdout: tuple[str, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for name in ("smoke", "evolution", "holdout"):
            for task_id in getattr(self, name):
                if not task_id:
                    raise ProtocolError(f"{name} contains an empty task id")
                if task_id in seen:
                    raise ProtocolError(f"pools overlap at task {task_id!r}")
                seen.add(task_id)

    def all_ids(self) -> tuple[str, ...]:
        return self.smoke + self.evolution + self.holdout


def split_pools(
    task_ids: Iterable[str], seed: int = 0, protocol: Protocol = Protocol()
) -> Pools:
    ids = list(task_ids)
    if len(ids) != protocol.total:
        raise ProtocolError(
            f"task count {len(ids)} does not match protocol total {protocol.total}"
        )
    if any(not task_id for task_id in ids):
        raise ProtocolError("task ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ProtocolError("task ids contain a duplicate")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    smoke_end = protocol.smoke
    evolution_end = smoke_end + protocol.evolution
    return Pools(
        smoke=tuple(shuffled[:smoke_end]),
        evolution=tuple(shuffled[smoke_end:evolution_end]),
        holdout=tuple(shuffled[evolution_end:]),
    )


def freeze_hash(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value cannot be frozen: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pool_hash(pools: Pools) -> str:
    return freeze_hash(
        {
            "smoke": sorted(pools.smoke),
            "evolution": sorted(pools.evolution),
            "holdout": sorted(pools.holdout),
        }
    )


@dataclass(frozen=True)
class Prereg:
    arm: str
    hypothesis: str
    primary_endpoint: str
    margin: float
    n: int
    k: int
    stopping_rule: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ProtocolError(f"unknown arm {self.arm!r}")
        for name in ("hypothesis", "primary_endpoint", "stopping_rule"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProtocolError(f"{name} must be a non-empty string")
        if not isinstance(self.margin, (int, float)) or isinstance(self.margin, bool):
            raise ProtocolError(f"margin must be numeric: {self.margin!r}")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ProtocolError(f"margin must be finite and non-negative: {self.margin}")
        for name in ("n", "k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ProtocolError(f"{name} must be a positive integer: {value!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "hypothesis": self.hypothesis,
            "primary_endpoint": self.primary_endpoint,
            "margin": self.margin,
            "n": self.n,
            "k": self.k,
            "stopping_rule": self.stopping_rule,
        }
