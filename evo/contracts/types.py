from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class KVector:
    k_lines_nb: int = 0
    k_ast: int = 0
    k_chars: int = 0
    k_bytes: int = 0
    k_gzip: int = 0
    k_cc: int = 0

    def __sub__(self, other: "KVector") -> "KVector":
        return KVector(
            k_lines_nb=self.k_lines_nb - other.k_lines_nb,
            k_ast=self.k_ast - other.k_ast,
            k_chars=self.k_chars - other.k_chars,
            k_bytes=self.k_bytes - other.k_bytes,
            k_gzip=self.k_gzip - other.k_gzip,
            k_cc=self.k_cc - other.k_cc,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "k_lines_nb": self.k_lines_nb,
            "k_ast": self.k_ast,
            "k_chars": self.k_chars,
            "k_bytes": self.k_bytes,
            "k_gzip": self.k_gzip,
            "k_cc": self.k_cc,
        }


@dataclass(frozen=True)
class KResult(KVector):
    files: Mapping[str, KVector] = field(default_factory=dict)


@dataclass(frozen=True)
class Boundary:
    name: str
    patterns: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class BoundarySpec:
    subject: str
    boundaries: tuple[Boundary, ...]
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    paths: tuple[str, ...]
    description: str = ""
