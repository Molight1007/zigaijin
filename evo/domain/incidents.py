from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


STATUSES = ("open", "fixed", "verified", "wontfix")


class IncidentError(ValueError):
    """An incident record is malformed or conflicts with an existing record."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IncidentError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Incident:
    id: str
    version: str
    phase: str
    location: str
    symptom: str
    consequence: str
    component: str = ""
    task_id: str = ""
    trial_id: str = ""
    cause: str = ""
    fix: str = ""
    status: str = "open"
    evidence: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        for name in ("id", "version", "phase", "location", "symptom", "consequence"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        for name in ("component", "task_id", "trial_id", "cause", "fix"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise IncidentError(f"{name} must be a string")
        if self.status not in STATUSES:
            raise IncidentError(
                f"status must be one of {', '.join(STATUSES)}: {self.status!r}"
            )
        if any(not isinstance(item, str) or not item for item in self.evidence):
            raise IncidentError("evidence must contain non-empty strings")
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "phase": self.phase,
            "location": self.location,
            "component": self.component,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
            "symptom": self.symptom,
            "consequence": self.consequence,
            "cause": self.cause,
            "fix": self.fix,
            "status": self.status,
            "evidence": list(self.evidence),
            "created_at": self.created_at,
        }


def read_incidents(path: str | Path) -> list[dict[str, object]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, object]] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IncidentError(f"{target} line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise IncidentError(f"{target} line {number}: expected a JSON object")
        rows.append(payload)
    return rows


def append_incident(path: str | Path, incident: Incident) -> None:
    if not isinstance(incident, Incident):
        raise IncidentError("incident must be an Incident")
    target = Path(path)
    existing = {str(row.get("id")) for row in read_incidents(target)}
    if incident.id in existing:
        raise IncidentError(f"duplicate incident id: {incident.id!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(incident.as_dict(), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def latest_incidents(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for row in rows:
        identifier = row.get("id")
        if isinstance(identifier, str):
            latest[identifier] = row
    return latest
