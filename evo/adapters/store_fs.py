from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4


class StoreError(ValueError):
    """A path or record cannot be handled by the run store."""


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _target(self, rel: str | Path) -> Path:
        raw = str(rel).replace("\\", "/")
        pure = PurePosixPath(raw)
        if not raw or pure.is_absolute() or ".." in pure.parts or Path(raw).is_absolute():
            raise StoreError(f"path must stay inside the run root: {rel!r}")
        target = (self.root / Path(*pure.parts)).resolve()
        if not target.is_relative_to(self.root):
            raise StoreError(f"path escapes the run root: {rel!r}")
        return target

    def _atomic_write(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with temp.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except OSError as exc:
            raise StoreError(f"cannot write {target}: {exc}") from exc
        finally:
            if temp.exists():
                temp.unlink()

    def write_text(self, rel: str | Path, text: str) -> None:
        self._atomic_write(self._target(rel), _normalize_newlines(text).encode("utf-8"))

    def read_text(self, rel: str | Path) -> str:
        target = self._target(rel)
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StoreError(f"cannot read {rel}: {exc}") from exc

    def write_json(self, rel: str | Path, value: object) -> None:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        except (TypeError, ValueError) as exc:
            raise StoreError(f"value is not JSON serializable: {exc}") from exc
        self.write_text(rel, text)

    def read_json(self, rel: str | Path) -> object:
        try:
            return json.loads(self.read_text(rel))
        except json.JSONDecodeError as exc:
            raise StoreError(f"invalid JSON in {rel}: {exc}") from exc

    def append_jsonl(self, rel: str | Path, value: object) -> None:
        target = self._target(rel)
        try:
            line = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise StoreError(f"value is not JSON serializable: {exc}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise StoreError(f"cannot append {rel}: {exc}") from exc

    def read_jsonl(self, rel: str | Path) -> list[object]:
        records = []
        for number, line in enumerate(self.read_text(rel).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise StoreError(f"invalid JSONL at {rel}:{number}: {exc}") from exc
        return records

    def exists(self, rel: str | Path) -> bool:
        return self._target(rel).is_file()

    def hash_file(self, rel: str | Path) -> str:
        target = self._target(rel)
        try:
            return content_hash(target.read_bytes())
        except OSError as exc:
            raise StoreError(f"cannot hash {rel}: {exc}") from exc

    def hash_text(self, text: str) -> str:
        return content_hash(text.encode("utf-8"))
