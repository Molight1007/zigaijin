from __future__ import annotations

import pytest

from evo.domain.protocol import (
    Prereg,
    Pools,
    Protocol,
    ProtocolError,
    freeze_hash,
    pool_hash,
    split_pools,
)


def test_default_protocol_matches_stage_one():
    protocol = Protocol()
    assert protocol.total == 89
    assert protocol.iteration_k == 3
    assert protocol.failure_repro_k == 10
    assert protocol.final_k == 5


def test_split_pools_are_disjoint_and_complete():
    tasks = [f"t{i:03d}" for i in range(89)]
    pools = split_pools(tasks, seed=11)
    assert len(pools.smoke) == 2
    assert len(pools.evolution) == 60
    assert len(pools.holdout) == 27
    assert set(pools.all_ids()) == set(tasks)
    assert len(set(pools.all_ids())) == 89


def test_split_is_deterministic_and_seed_sensitive():
    tasks = [f"t{i:03d}" for i in range(89)]
    assert split_pools(tasks, seed=3) == split_pools(tasks, seed=3)
    assert split_pools(tasks, seed=3) != split_pools(tasks, seed=4)


def test_custom_protocol_sizes():
    protocol = Protocol(smoke=1, evolution=3, holdout=2)
    pools = split_pools([f"t{i}" for i in range(6)], seed=0, protocol=protocol)
    assert (len(pools.smoke), len(pools.evolution), len(pools.holdout)) == (1, 3, 2)


def test_duplicate_task_ids_are_rejected():
    with pytest.raises(ProtocolError, match="duplicate"):
        split_pools(["a", "a"], seed=0, protocol=Protocol(smoke=1, evolution=1, holdout=0))


def test_task_count_must_match_protocol():
    with pytest.raises(ProtocolError, match="count"):
        split_pools(["a"], seed=0, protocol=Protocol(smoke=1, evolution=1, holdout=0))


def test_pools_reject_overlap():
    with pytest.raises(ProtocolError, match="overlap"):
        Pools(smoke=("a",), evolution=("a",), holdout=())


def test_pool_hash_ignores_order_but_not_membership():
    one = Pools(("a", "b"), ("c", "d"), ("e",))
    reordered = Pools(("b", "a"), ("d", "c"), ("e",))
    changed = Pools(("a", "b"), ("c", "e"), ("d",))
    assert pool_hash(one) == pool_hash(reordered)
    assert pool_hash(one) != pool_hash(changed)


def test_freeze_hash_is_canonical_and_sensitive():
    first = {"b": [1, 2], "a": {"x": 1, "y": 2}}
    second = {"a": {"y": 2, "x": 1}, "b": [1, 2]}
    assert freeze_hash(first) == freeze_hash(second)
    assert freeze_hash(first) != freeze_hash({"a": {"x": 1, "y": 3}, "b": [1, 2]})


def test_freeze_hash_rejects_non_serializable_values():
    with pytest.raises(ProtocolError):
        freeze_hash({"bad": {1, 2}})


def valid_prereg():
    return Prereg(
        arm="SIM",
        hypothesis="L1 deletion does not reduce S",
        primary_endpoint="S",
        margin=0.10,
        n=180,
        k=3,
        stopping_rule="stop after 20 rounds or two invalid rounds",
    )


def test_prereg_roundtrip_and_required_fields():
    prereg = valid_prereg()
    data = prereg.as_dict()
    assert data["arm"] == "SIM"
    assert data["margin"] == 0.10
    assert data["n"] == 180


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", "UNKNOWN"),
        ("hypothesis", ""),
        ("primary_endpoint", " "),
        ("margin", -0.1),
        ("n", 0),
        ("k", 0),
        ("stopping_rule", ""),
    ],
)
def test_prereg_validation_rejects_bad_values(field, value):
    values = valid_prereg().as_dict()
    values[field] = value
    with pytest.raises(ProtocolError):
        Prereg(**values)
