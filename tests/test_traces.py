from __future__ import annotations

import json

import pytest

from evo.application.traces import (
    TraceMergeError,
    merge_trace_dir,
    merge_trace_payloads,
)


def payload(task, lines, *, subject="terminus2", root="/subject"):
    return {
        "subject": subject,
        "root": root,
        "task_ids": [task],
        "tasks": {task: {"pkg/mod.py": lines}},
    }


def test_merge_unions_lines_for_the_same_task():
    merged = merge_trace_payloads(
        [payload("t1", [1, 2]), payload("t1", [2, 3]), payload("t2", [4])]
    )

    assert merged["task_ids"] == ["t1", "t2"]
    assert merged["tasks"]["t1"]["pkg/mod.py"] == [1, 2, 3]
    assert merged["tasks"]["t2"]["pkg/mod.py"] == [4]


def test_merge_rejects_different_subjects():
    with pytest.raises(TraceMergeError, match="subject"):
        merge_trace_payloads(
            [payload("t1", [1]), payload("t1", [1], subject="other")]
        )


def test_merge_dir_reads_each_json_file(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(payload("t1", [1])), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(
        json.dumps(payload("t1", [2])), encoding="utf-8"
    )

    merged = merge_trace_dir(tmp_path)

    assert merged["tasks"]["t1"]["pkg/mod.py"] == [1, 2]
