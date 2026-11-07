from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from evo.contracts.types import KVector
from evo.domain.arms import ORDER, arms_table
from evo.domain.protocol import (
    Protocol,
    ProtocolError,
    pool_hash,
    split_pools,
)


DEFAULT_MARGIN = 0.10
DEFAULT_HOLDOUT_MARGIN = 0.10
PRIMARY_ENDPOINT = "reward"
STOPPING_RULE = (
    "stop after the pre-registered number of rounds or at the first rejected round"
)


class FreezeError(ValueError):
    """A run configuration cannot be frozen."""


def _margin(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeError(f"{name} must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise FreezeError(f"{name} must be in [0, 1]: {value!r}")
    return number


def tree_hash(root: str | Path, paths: Iterable[str]) -> str:
    """Hash sorted subject-relative paths and their bytes into one digest."""
    base = Path(root).resolve()
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise FreezeError(f"subject path must stay inside the root: {relative!r}")
        target = (base / path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise FreezeError(f"subject path is not a file: {relative!r}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(target.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_run(
    task_ids: Iterable[str],
    *,
    dataset: str,
    subject: str,
    k_vector: KVector,
    subject_hash: str = "",
    protocol: Protocol = Protocol(),
    seed: int = 0,
    margin: float = DEFAULT_MARGIN,
    holdout_margin: float = DEFAULT_HOLDOUT_MARGIN,
    created_at: str | None = None,
) -> dict[str, object]:
    if not isinstance(dataset, str) or not dataset.strip():
        raise FreezeError(f"dataset must be a non-empty string: {dataset!r}")
    if not isinstance(subject, str) or not subject.strip():
        raise FreezeError(f"subject must be a non-empty string: {subject!r}")
    if not isinstance(k_vector, KVector):
        raise FreezeError(f"k_vector must be a KVector: {k_vector!r}")
    if not isinstance(subject_hash, str):
        raise FreezeError(f"subject_hash must be a string: {subject_hash!r}")
    if not isinstance(protocol, Protocol):
        raise FreezeError(f"protocol must be a Protocol: {protocol!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FreezeError(f"seed must be an integer: {seed!r}")
    margin = _margin("margin", margin)
    holdout_margin = _margin("holdout_margin", holdout_margin)
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not isinstance(created_at, str) or not created_at.strip():
        raise FreezeError(f"created_at must be a non-empty string: {created_at!r}")

    ids = tuple(task_ids)
    try:
        pools = split_pools(ids, seed=seed, protocol=protocol)
    except ProtocolError as exc:
        raise FreezeError(str(exc)) from exc

    return {
        "schema": 1,
        "dataset": dataset,
        "subject": subject,
        "subject_hash": subject_hash,
        "n_tasks": len(ids),
        "seed": seed,
        "protocol": asdict(protocol),
        "pools": {
            "smoke": list(pools.smoke),
            "evolution": list(pools.evolution),
            "holdout": list(pools.holdout),
        },
        "pool_hash": pool_hash(pools),
        "k": k_vector.as_dict(),
        "arms": arms_table(),
        "order": ORDER,
        "prereg": {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "margin": margin,
            "holdout_margin": holdout_margin,
            "n": len(ids),
            "k": protocol.iteration_k,
            "stopping_rule": STOPPING_RULE,
        },
        "created_at": created_at,
    }
