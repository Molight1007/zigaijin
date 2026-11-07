from __future__ import annotations

import re
from typing import Sequence

_GENERIC = (
    (r"AIza[0-9A-Za-z_-]{35}", "credential in diff"),
    (r"sk-[A-Za-z0-9]{20,}", "credential in diff"),
    (r"api_key\s*=\s*[\"'][^\"']{8,}", "credential in diff"),
    (r"test-stdout\.txt", "grader artifact in diff"),
    (r"(?:^|[/\\'\"\s])(?:verifier|grader)[/\\]", "grader path in diff"),
)


class CriticError(ValueError):
    """A critic input is malformed."""


def precheck(
    diff: str,
    terms: Sequence[str] = (),
    patterns: Sequence[tuple[str, str]] = (),
) -> tuple[str, ...]:
    """Run deterministic leak checks before spending an LLM or eval call.

    Generic patterns cover credentials, grader artifacts and grader paths;
    terms cover task ids supplied by the caller. Reasons are deduplicated.
    """
    if not isinstance(diff, str):
        raise CriticError(f"diff must be a string: {diff!r}")
    reasons: list[str] = []
    for pattern, reason in _GENERIC + tuple(patterns):
        if re.search(pattern, diff):
            reasons.append(reason)
    for term in terms:
        if not isinstance(term, str):
            raise CriticError(f"term must be a string: {term!r}")
        if term and term in diff:
            reasons.append(f"task term in diff: {term}")
    if "/tests/" in diff or "\\tests\\" in diff:
        reasons.append("test path in diff")
    return tuple(dict.fromkeys(reasons))
