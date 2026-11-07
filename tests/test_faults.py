from __future__ import annotations

import pytest

from evo.domain.faults import (
    FaultError,
    apply_mutation,
    find_mutations,
    sample_mutations,
)


def mutate(source, mutation):
    return apply_mutation(source, mutation)


def operators(mutations):
    return {mutation.operator for mutation in mutations}


def test_negate_condition_flips_the_branch():
    source = (
        "def f(x):\n"
        "    if x > 1:\n"
        "        return 'big'\n"
        "    return 'small'\n"
    )
    mutations = find_mutations(source, path="m.py")
    negated = [m for m in mutations if m.operator == "negate_condition"]
    assert len(negated) == 1
    mutated = mutate(source, negated[0])
    assert "if not (x > 1):" in mutated
    namespace = {}
    exec(mutated, namespace)
    assert namespace["f"](0) == "big"
    assert namespace["f"](2) == "small"


def test_swap_branches_swaps_the_bodies():
    source = (
        "def f(x):\n"
        "    if x:\n"
        "        return 'yes'\n"
        "    else:\n"
        "        return 'no'\n"
    )
    mutations = find_mutations(source, path="m.py")
    swaps = [m for m in mutations if m.operator == "swap_branches"]
    assert len(swaps) == 1
    mutated = mutate(source, swaps[0])
    namespace = {}
    exec(mutated, namespace)
    assert namespace["f"](True) == "no"
    assert namespace["f"](False) == "yes"


def test_an_elif_chain_is_never_swapped_through_the_parent():
    source = (
        "def f(x):\n"
        "    if x == 1:\n"
        "        return 'one'\n"
        "    elif x == 2:\n"
        "        return 'two'\n"
        "    else:\n"
        "        return 'other'\n"
    )
    mutations = find_mutations(source, path="m.py")
    swaps = [m for m in mutations if m.operator == "swap_branches"]
    assert len(swaps) == 1
    mutated = mutate(source, swaps[0])
    compile(mutated, "m.py", "exec")
    namespace = {}
    exec(mutated, namespace)
    assert namespace["f"](1) == "one"


def test_bool_and_number_constants_are_mutated():
    source = "FLAG = True\nRETRIES = 2\n\ndef f():\n    return FLAG, RETRIES\n"
    mutations = find_mutations(source, path="m.py")
    assert {"flip_bool", "change_constant"} <= operators(mutations)
    flip = next(m for m in mutations if m.operator == "flip_bool")
    assert "FLAG = False" in mutate(source, flip)
    number = next(m for m in mutations if m.operator == "change_constant")
    mutated = mutate(source, number)
    assert "RETRIES = 3" in mutated or "RETRIES = 1" in mutated


def test_mutations_are_deterministic_and_have_unique_ids():
    source = "def f(x):\n    if x and True:\n        return 1\n    return 0\n"
    first = find_mutations(source, path="m.py")
    second = find_mutations(source, path="m.py")
    assert [m.id for m in first] == [m.id for m in second]
    assert len({m.id for m in first}) == len(first)
    for mutation in first:
        assert mutation.path == "m.py"
        assert mutation.line >= 1
        assert mutation.symbol == "f"


def test_sample_is_seeded_and_bounded():
    source = "\n".join(
        f"def f{i}(x):\n    if x > {i}:\n        return {i}\n    return 0"
        for i in range(5)
    )
    mutations = find_mutations(source, path="m.py")
    assert len(mutations) > 4
    first = sample_mutations(mutations, count=3, seed=7)
    second = sample_mutations(mutations, count=3, seed=7)
    assert [m.id for m in first] == [m.id for m in second]
    assert len(first) == 3
    assert {m.id for m in first} <= {m.id for m in mutations}
    assert len(sample_mutations(mutations, count=99, seed=7)) == len(mutations)
    assert sample_mutations(mutations, count=0, seed=7) == ()


def test_apply_rejects_a_stale_source():
    source = "def f(x):\n    if x > 1:\n        return 1\n    return 0\n"
    mutation = next(
        m for m in find_mutations(source, path="m.py") if m.operator == "negate_condition"
    )
    with pytest.raises(FaultError, match="does not match"):
        apply_mutation(source.replace("x > 1", "x > 2"), mutation)


def test_syntax_errors_are_rejected():
    with pytest.raises(FaultError, match="parse"):
        find_mutations("def f(:\n", path="m.py")


def test_a_file_without_mutable_sites_yields_nothing():
    assert find_mutations('NAME = "hello"\n', path="m.py") == ()


def test_unicode_before_a_constant_does_not_break_offsets():
    source = 'NAME = "中文"\nFLAG = True\n'
    flip = next(m for m in find_mutations(source, path="m.py") if m.operator == "flip_bool")
    mutated = apply_mutation(source, flip)
    assert 'NAME = "中文"' in mutated
    assert "FLAG = False" in mutated


def test_path_and_arguments_are_validated():
    with pytest.raises(FaultError, match="path"):
        find_mutations("x = 1\n", path="")
    with pytest.raises(FaultError, match="count"):
        sample_mutations((), count=-1, seed=0)


def test_parse_mutations_roundtrips_as_dict():
    from evo.domain.faults import parse_mutations

    source = "def f(x):\n    if x > 1:\n        return 1\n    return 0\n"
    mutations = find_mutations(source, path="m.py")
    payload = {
        "schema": 1,
        "root": ".",
        "sources": {"m.py": "x"},
        "mutations": [mutation.as_dict() for mutation in mutations],
    }
    parsed = parse_mutations(payload)
    assert [mutation.as_dict() for mutation in parsed] == [
        mutation.as_dict() for mutation in mutations
    ]
    for original, copy in zip(mutations, parsed):
        assert apply_mutation(source, copy) == apply_mutation(source, original)


def test_parse_mutations_rejects_bad_payloads():
    from evo.domain.faults import parse_mutations

    with pytest.raises(FaultError, match="mutations"):
        parse_mutations({})
    with pytest.raises(FaultError, match="duplicate"):
        parse_mutations(
            {
                "mutations": [
                    {"id": "a", "path": "m.py", "line": 1, "operator": "flip_bool", "symbol": "f", "edits": [{"start": 0, "end": 1, "before": "a", "after": "b"}]},
                    {"id": "a", "path": "m.py", "line": 1, "operator": "flip_bool", "symbol": "f", "edits": [{"start": 0, "end": 1, "before": "a", "after": "b"}]},
                ]
            }
        )
    with pytest.raises(FaultError, match="edit"):
        parse_mutations(
            {
                "mutations": [
                    {"id": "a", "path": "m.py", "line": 1, "operator": "flip_bool", "symbol": "f", "edits": []}
                ]
            }
        )


def test_verify_sources_rejects_a_changed_file(tmp_path):
    from evo.domain.faults import verify_sources

    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256((tmp_path / "m.py").read_bytes()).hexdigest()
    verify_sources(tmp_path, {"m.py": digest})
    (tmp_path / "m.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(FaultError, match="changed"):
        verify_sources(tmp_path, {"m.py": digest})


def test_verify_sources_rejects_a_missing_or_escaping_path(tmp_path):
    from evo.domain.faults import verify_sources

    with pytest.raises(FaultError, match="missing"):
        verify_sources(tmp_path, {"nope.py": "0" * 64})
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(FaultError, match="inside"):
        verify_sources(tmp_path, {"../ok.py": "0" * 64})
