from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


VERDICTS = ("keep", "delete", "undecidable", "undeletable")


class LedgerError(ValueError):
    """A ledger record is malformed or uses an invalid verdict."""


@dataclass(frozen=True)
class LedgerEntry:
    run_id: str
    candidate_id: str
    verdict: str
    reason: str
    round: int
    timestamp: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "round": self.round,
            "timestamp": self.timestamp,
            "evidence": dict(self.evidence),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(entry: LedgerEntry) -> str:
    try:
        return json.dumps(
            entry.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"evidence is not JSON serializable: {exc}") from exc


class Ledger:
    def __init__(
        self,
        path: Path,
        run_id: str,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if not run_id:
            raise LedgerError("run_id must not be empty")
        self.path = Path(path)
        self.run_id = run_id
        self._clock = clock

    def append(
        self,
        candidate_id: str,
        verdict: str,
        reason: str,
        round: int = 0,
        evidence: Mapping[str, object] | None = None,
    ) -> LedgerEntry:
        if not candidate_id:
            raise LedgerError("candidate_id must not be empty")
        if verdict not in VERDICTS:
            raise LedgerError(
                f"unknown verdict {verdict!r}; expected one of {', '.join(VERDICTS)}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError("reason must be a non-empty string")
        if not isinstance(round, int) or isinstance(round, bool) or round < 0:
            raise LedgerError(f"round must be a non-negative integer: {round!r}")

        entry = LedgerEntry(
            run_id=self.run_id,
            candidate_id=candidate_id,
            verdict=verdict,
            reason=reason,
            round=round,
            timestamp=self._clock(),
            evidence=dict(evidence or {}),
        )
        line = _serialize(entry) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise LedgerError(f"cannot read ledger: {exc}") from exc
        entries = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"malformed ledger line {number}: {exc}") from exc
            entries.append(self._entry_from_record(record, number))
        return entries

    def _entry_from_record(self, record: object, number: int) -> LedgerEntry:
        if not isinstance(record, dict):
            raise LedgerError(f"ledger line {number} is not an object")
        required = ("run_id", "candidate_id", "verdict", "reason", "round", "timestamp")
        missing = [key for key in required if key not in record]
        if missing:
            raise LedgerError(f"ledger line {number} missing fields: {', '.join(missing)}")
        if record["run_id"] != self.run_id:
            raise LedgerError(
                f"ledger line {number} has run_id {record['run_id']!r}, expected {self.run_id!r}"
            )
        if record["verdict"] not in VERDICTS:
            raise LedgerError(f"ledger line {number} has unknown verdict {record['verdict']!r}")
        if not isinstance(record["reason"], str) or not record["reason"].strip():
            raise LedgerError(f"ledger line {number} has an empty reason")
        evidence = record.get("evidence", {})
        if not isinstance(evidence, dict):
            raise LedgerError(f"ledger line {number} evidence is not an object")
        return LedgerEntry(
            run_id=record["run_id"],
            candidate_id=record["candidate_id"],
            verdict=record["verdict"],
            reason=record["reason"],
            round=record["round"],
            timestamp=record["timestamp"],
            evidence=evidence,
        )

    def latest(self) -> dict[str, LedgerEntry]:
        latest: dict[str, LedgerEntry] = {}
        for entry in self.entries():
            latest[entry.candidate_id] = entry
        return latest

    def summary(self) -> dict[str, int]:
        return dict(Counter(entry.verdict for entry in self.entries()))
