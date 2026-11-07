"""Render one MathPilot trial for the analyst / digester roles.

The raw result record already carries everything the analysis stage needs:
the question, the final answer, the gold answer, the error class, wall-clock
cost, and a `diag` object with one entry per pipeline stage (blueprint nodes,
skeleton review, DAG review, subgoal trace, subgoal stats, formal gaps ...).
So rendering is a formatting job, not a join across files.

detail=True keeps long field values (used by the digester); the default trims
them so a whole evolve set still fits in one prompt.
"""

from __future__ import annotations

import json

_MAX_SHORT = 320
_MAX_LONG = 2000


def _clip(value, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …"


def render_full(rec: dict, detail: bool = False) -> str:
    raw = rec.get("raw") or {}
    limit = _MAX_LONG if detail else _MAX_SHORT
    out: list[str] = []

    out.append(f"=== TASK {rec.get('task_id')} ===")
    out.append(
        "OUTCOME: reward={rw} elapsed={sec}s error_class={err} empty_answer={empty}".format(
            rw=rec.get("reward"), sec=rec.get("elapsed_sec"),
            err=rec.get("error_class") or "-", empty=rec.get("empty_answer")))

    question = raw.get("question") or ""
    out.append("QUESTION: " + _clip(question, limit * 2))
    out.append("PREDICTED: " + _clip(raw.get("predicted") or "", limit))
    out.append("GOLD: " + _clip(raw.get("gold") or "", limit))

    diag = raw.get("diag") or {}
    if isinstance(diag, dict) and diag:
        out.append(f"--- diag ({len(diag)} keys) ---")
        for key, value in diag.items():
            out.append(f"DIAG.{key}: {_clip(value, limit)}")

    if detail:
        out.append("RAW_KEYS: " + ", ".join(sorted(raw)))
    return "\n".join(out)
