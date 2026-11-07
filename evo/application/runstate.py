from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator

from filelock import FileLock, Timeout

from evo.adapters.store_fs import RunStore, StoreError
from evo.domain.state import RunState, StateError, advance


STATE_FILE = "state.json"
EVENTS_FILE = "events.jsonl"
LOCK_FILE = ".state.lock"


class RunStateError(ValueError):
    """A run state file or transition cannot be used."""


def _store(root: str | Path) -> RunStore:
    try:
        return RunStore(Path(root))
    except OSError as exc:
        raise RunStateError(f"cannot create run root {root}: {exc}") from exc


def load_state(root: str | Path) -> RunState | None:
    path = Path(root) / STATE_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunStateError(f"cannot read {path}: {exc}") from exc
    try:
        return RunState.from_dict(payload)
    except StateError as exc:
        raise RunStateError(f"invalid {path}: {exc}") from exc


@contextmanager
def _locked(root: Path, timeout: float) -> Iterator[None]:
    lock = FileLock(str(root / LOCK_FILE))
    try:
        with lock.acquire(timeout=timeout):
            yield
    except Timeout as exc:
        raise RunStateError(f"could not lock {root / LOCK_FILE}") from exc


def init_run(
    root: str | Path, run_id: str, *, at: str | None = None, timeout: float = 30.0
) -> RunState:
    store = _store(root)
    with _locked(store.root, timeout):
        existing = load_state(store.root)
        if existing is not None:
            if existing.run_id == run_id:
                return existing
            raise RunStateError(
                f"run root already holds run {existing.run_id!r}: {store.root}"
            )
        try:
            state = RunState.initial(run_id, at=at)
        except StateError as exc:
            raise RunStateError(str(exc)) from exc
        store.write_json(STATE_FILE, state.as_dict())
        store.append_jsonl(
            EVENTS_FILE,
            {
                "at": state.updated_at,
                "from": None,
                "to": state.phase,
                "actor": "runner",
                "artifact": None,
                "resume": None,
            },
        )
    return state


def _artifact(store: RunStore, artifact: str | Path | None) -> str | None:
    if artifact is None:
        return None
    raw = str(artifact).replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or Path(raw).is_absolute() or ".." in pure.parts:
        raise RunStateError(f"artifact must stay inside the run root: {raw!r}")
    target = (store.root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(store.root):
        raise RunStateError(f"artifact escapes the run root: {raw!r}")
    if not target.is_file():
        raise RunStateError(f"artifact does not exist: {raw!r}")
    return pure.as_posix()


def advance_run(
    root: str | Path,
    to: str,
    *,
    artifact: str | Path | None = None,
    actor: str = "runner",
    at: str | None = None,
    timeout: float = 30.0,
) -> tuple[RunState, dict[str, object]]:
    store = _store(root)
    with _locked(store.root, timeout):
        state = load_state(store.root)
        if state is None:
            raise RunStateError(
                f"{store.root / STATE_FILE} does not exist; call init_run first"
            )
        relative = _artifact(store, artifact)
        try:
            updated = advance(
                state, to, artifact=relative, actor=actor, at=at
            )
        except StateError as exc:
            raise RunStateError(str(exc)) from exc
        event = {
            "at": updated.updated_at,
            "from": state.phase,
            "to": updated.phase,
            "actor": actor,
            "artifact": relative,
            "resume": updated.resume,
        }
        store.write_json(STATE_FILE, updated.as_dict())
        store.append_jsonl(EVENTS_FILE, event)
    return updated, event
