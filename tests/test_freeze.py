from __future__ import annotations

import json

import pytest

from evo.application.freeze import FreezeError, freeze_run, tree_hash
from evo.cli import main
from evo.contracts.types import KVector
from evo.domain.protocol import Protocol, pool_hash, split_pools


def task_ids(count=89):
    return tuple(f"terminal-bench/task-{index:03d}" for index in range(count))


def k_vector():
    return KVector(
        k_lines_nb=2893,
        k_ast=16299,
        k_chars=120000,
        k_bytes=130000,
        k_gzip=30000,
        k_cc=1200,
    )


def freeze(**overrides):
    payload = {
        "task_ids": task_ids(),
        "dataset": "terminal-bench/terminal-bench-2-1",
        "subject": "vendor/terminus2",
        "k_vector": k_vector(),
        "created_at": "2026-11-04T00:00:00+00:00",
    }
    payload.update(overrides)
    return freeze_run(**payload)


def test_freeze_splits_the_default_protocol():
    payload = freeze()
    pools = payload["pools"]
    assert len(pools["smoke"]) == 2
    assert len(pools["evolution"]) == 60
    assert len(pools["holdout"]) == 27
    seen = pools["smoke"] + pools["evolution"] + pools["holdout"]
    assert sorted(seen) == sorted(task_ids())
    assert payload["n_tasks"] == 89


def test_freeze_records_the_pool_hash_of_the_split():
    payload = freeze()
    expected = pool_hash(
        split_pools(task_ids(), seed=0, protocol=Protocol())
    )
    assert payload["pool_hash"] == expected


def test_freeze_is_reproducible():
    assert freeze() == freeze()


def test_freeze_hash_ignores_the_timestamp():
    first = freeze(created_at="2026-11-04T00:00:00+00:00")
    second = freeze(created_at="2026-11-05T12:00:00+00:00")
    assert first["pool_hash"] == second["pool_hash"]
    assert first["pools"] == second["pools"]


def test_freeze_changes_pools_with_the_seed():
    first = freeze(seed=0)
    second = freeze(seed=1)
    assert first["pools"] != second["pools"]
    assert first["pool_hash"] != second["pool_hash"]


def test_freeze_records_arms_prereg_and_k():
    payload = freeze()
    assert [row["arm"] for row in payload["arms"]] == [
        "H0",
        "EVO",
        "SIM",
        "COMBO",
        "PLACEBO",
        "SHARP",
    ]
    assert payload["order"] == "evolve_then_simplify"
    prereg = payload["prereg"]
    assert prereg["primary_endpoint"] == "reward"
    assert prereg["margin"] == 0.10
    assert prereg["holdout_margin"] == 0.10
    assert prereg["n"] == 89
    assert prereg["k"] == 3
    assert payload["k"]["k_lines_nb"] == 2893
    assert payload["subject"] == "vendor/terminus2"


def test_freeze_rejects_a_wrong_task_count():
    with pytest.raises(FreezeError, match="89"):
        freeze(task_ids=task_ids(88))


def test_freeze_rejects_duplicate_tasks():
    ids = list(task_ids())
    ids[1] = ids[0]
    with pytest.raises(FreezeError, match="duplicate"):
        freeze(task_ids=tuple(ids))


def test_freeze_rejects_a_missing_margin():
    with pytest.raises(FreezeError, match="margin"):
        freeze(margin=-0.1)


def test_freeze_rejects_a_non_kvector():
    with pytest.raises(FreezeError, match="KVector"):
        freeze(k_vector={"k_lines_nb": 1})


def test_freeze_rejects_empty_subject():
    with pytest.raises(FreezeError, match="subject"):
        freeze(subject="  ")


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def test_tree_hash_is_stable_and_content_sensitive(tmp_path):
    root = tmp_path / "subject"
    write(root, "a.py", "x = 1\n")
    write(root, "b.py", "y = 2\n")
    first = tree_hash(root, ("a.py", "b.py"))
    assert first == tree_hash(root, ("a.py", "b.py"))
    write(root, "a.py", "x = 2\n")
    assert tree_hash(root, ("a.py", "b.py")) != first


def test_freeze_cli_writes_run_json(tmp_path, capsys):
    root = tmp_path / "subject"
    write(root, "pkg/a.py", "x = 1\n")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("\n".join(task_ids(5)) + "\n", encoding="utf-8")
    out = tmp_path / "run"
    assert (
        main(
            [
                "freeze",
                "--tasks-file",
                str(tasks),
                "--subject",
                "demo",
                "--root",
                str(root),
                "--out",
                str(out),
                "--smoke",
                "1",
                "--evolution",
                "2",
                "--holdout",
                "2",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(out / "run.json")
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "terminal-bench/terminal-bench-2-1"
    assert payload["prereg"]["n"] == 5
    assert payload["k"]["k_lines_nb"] == 1
    assert len(payload["subject_hash"]) == 64
    assert len(payload["pools"]["smoke"]) == 1
    assert len(payload["pools"]["evolution"]) == 2
    assert len(payload["pools"]["holdout"]) == 2


def test_freeze_cli_rejects_a_missing_task_file(tmp_path, capsys):
    root = tmp_path / "subject"
    write(root, "a.py", "x = 1\n")
    assert (
        main(
            [
                "freeze",
                "--tasks-file",
                str(tmp_path / "nope.txt"),
                "--subject",
                "demo",
                "--root",
                str(root),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "tasks" in capsys.readouterr().err


def test_freeze_cli_rejects_an_empty_task_file(tmp_path, capsys):
    root = tmp_path / "subject"
    write(root, "a.py", "x = 1\n")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("\n\n", encoding="utf-8")
    assert (
        main(
            [
                "freeze",
                "--tasks-file",
                str(tasks),
                "--subject",
                "demo",
                "--root",
                str(root),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "task" in capsys.readouterr().err
