from __future__ import annotations

import json

import pytest

from evo.adapters.store_fs import RunStore, StoreError, content_hash


def test_json_roundtrip_and_parent_creation(tmp_path):
    store = RunStore(tmp_path / "run")
    store.write_json("rounds/01/prereg.json", {"arm": "SIM", "k": 3})
    assert store.read_json("rounds/01/prereg.json") == {"arm": "SIM", "k": 3}


def test_text_is_written_with_lf(tmp_path):
    store = RunStore(tmp_path / "run")
    store.write_text("notes.txt", "a\r\nb\r\n")
    assert store.read_text("notes.txt") == "a\nb\n"


def test_jsonl_append_and_read(tmp_path):
    store = RunStore(tmp_path / "run")
    store.append_jsonl("events.jsonl", {"n": 1})
    store.append_jsonl("events.jsonl", {"n": 2})
    assert store.read_jsonl("events.jsonl") == [{"n": 1}, {"n": 2}]


def test_atomic_write_leaves_no_temp_files(tmp_path):
    store = RunStore(tmp_path / "run")
    store.write_json("state.json", {"state": "created"})
    leftovers = [p.name for p in (tmp_path / "run").rglob("*.tmp")]
    assert leftovers == []


def test_failed_serialization_does_not_replace_existing_file(tmp_path):
    store = RunStore(tmp_path / "run")
    store.write_json("state.json", {"state": "created"})
    with pytest.raises(StoreError):
        store.write_json("state.json", {"bad": {1, 2}})
    assert store.read_json("state.json") == {"state": "created"}


@pytest.mark.parametrize("bad", ["../escape.json", "a/../../escape.json", "/absolute.json"])
def test_paths_cannot_escape_root(tmp_path, bad):
    store = RunStore(tmp_path / "run")
    with pytest.raises(StoreError):
        store.write_json(bad, {})


def test_missing_file_raises_store_error(tmp_path):
    store = RunStore(tmp_path / "run")
    with pytest.raises(StoreError):
        store.read_json("missing.json")


def test_content_hash_is_stable_and_sensitive():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


def test_hash_text_matches_hash_file(tmp_path):
    store = RunStore(tmp_path / "run")
    store.write_text("a.txt", "中文\n")
    assert store.hash_text("中文\n") == store.hash_file("a.txt")


def test_exists(tmp_path):
    store = RunStore(tmp_path / "run")
    assert not store.exists("a.json")
    store.write_json("a.json", {})
    assert store.exists("a.json")
