from __future__ import annotations

from typing import Mapping, Sequence

from evo.application.baseline import (
    BaselineError,
    TrialOutcome,
    summarize_trials,
)
from evo.application.simplify import (
    CandidateDecision,
    SimplifyError,
    decide_candidate,
)
from evo.contracts.types import KVector
from evo.domain.gate import (
    GateError,
    GateRequest,
    KGateInput,
    ProtectionInput,
    ScoreInput,
    TokenInput,
    ValidityInput,
)
from evo.domain.generalization import GeneralizationError, direction_consistency
from evo.domain.scan import Candidate
from evo.domain.verify import CheckOutcome, VerifyError, VerifyReport, aggregate


K_FIELDS = ("k_lines_nb", "k_ast", "k_chars", "k_bytes", "k_gzip", "k_cc")


class RoundError(ValueError):
    """Round artifacts are missing or mutually inconsistent."""


def _pool_tasks(pools: Mapping[str, Sequence[str]], name: str) -> tuple[str, ...]:
    if not isinstance(pools, Mapping):
        raise RoundError(f"pools must be a mapping: {pools!r}")
    if name not in pools:
        known = ", ".join(sorted(str(key) for key in pools)) or "<none>"
        raise RoundError(f"unknown pool {name!r}; expected one of {known}")
    tasks = tuple(pools[name])
    if not tasks:
        raise RoundError(f"pool {name!r} is empty")
    for task in tasks:
        if not isinstance(task, str) or not task:
            raise RoundError(f"pool {name!r} has an invalid task id: {task!r}")
    if len(set(tasks)) != len(tasks):
        raise RoundError(f"pool {name!r} contains a duplicate task id")
    return tasks


def _kvector(applied: Mapping[str, object], name: str) -> KVector:
    value = applied.get(name)
    if value is None:
        raise RoundError(f"applied artifacts must include the {name} K vector")
    if not isinstance(value, Mapping):
        raise RoundError(f"applied {name} K vector must be a mapping: {value!r}")
    missing = [field for field in K_FIELDS if field not in value]
    extra = [field for field in value if field not in K_FIELDS]
    if missing or extra:
        raise RoundError(
            f"applied {name} K vector has the wrong fields: "
            f"missing {missing}, extra {extra}"
        )
    for field in K_FIELDS:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RoundError(
                f"applied {name}.{field} must be a non-negative integer: {item!r}"
            )
    return KVector(**{field: value[field] for field in K_FIELDS})


def _verify(applied: Mapping[str, object]) -> VerifyReport | None:
    payload = applied.get("verify")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise RoundError(f"applied verify must be a mapping: {payload!r}")
    required = payload.get("required")
    outcomes = payload.get("outcomes")
    if not isinstance(required, (list, tuple)) or not isinstance(
        outcomes, (list, tuple)
    ):
        raise RoundError("applied verify must contain required and outcomes lists")
    checks: list[CheckOutcome] = []
    for row in outcomes:
        if not isinstance(row, Mapping):
            raise RoundError(f"verification outcome must be a mapping: {row!r}")
        try:
            checks.append(
                CheckOutcome(
                    row.get("name", ""),
                    row.get("status", ""),
                    row.get("detail", ""),
                )
            )
        except VerifyError as exc:
            raise RoundError(f"invalid verification outcome: {exc}") from exc
    try:
        return aggregate(checks, tuple(required))
    except VerifyError as exc:
        raise RoundError(f"invalid verification report: {exc}") from exc


def _means(
    summary: Mapping[str, object], tasks: Sequence[str], arm: str
) -> tuple[float, ...]:
    rows = summary["tasks"]
    values: list[float] = []
    missing: list[str] = []
    for task in tasks:
        row = rows[task]
        mean = row["mean_reward"]
        if int(row["trials"]) == 0 or mean is None:
            missing.append(task)
        else:
            values.append(float(mean))
    if missing:
        raise RoundError(
            f"{arm} evidence is incomplete: "
            f"no completed trial for task(s) {', '.join(missing)}"
        )
    return tuple(values)


def token_evidence(
    summary: Mapping[str, object], tasks: Sequence[str]
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Per-task token totals plus tasks whose trials lack token data.

    A task with no completed trial or with any unscored trial is "missing",
    so callers can fail closed instead of treating absent tokens as zero.
    """
    rows = summary["tasks"]
    values: dict[str, int] = {}
    missing: list[str] = []
    for task in tasks:
        row = rows[task]
        values[task] = int(row["tokens_in"]) + int(row["tokens_out"])
        if int(row["trials"]) == 0 or int(row["trials_with_tokens"]) < int(row["trials"]):
            missing.append(task)
    return values, tuple(missing)


def merged_validity(
    reference: Mapping[str, object], evaluated: Mapping[str, object]
) -> ValidityInput:
    """Merge both arms conservatively: worst fraction, least coverage.

    Any arm's agent/infra/missing failures invalidate the whole measurement,
    so a reference-only failure cannot be hidden by a clean candidate.
    """
    ref = reference["validity"]
    cand = evaluated["validity"]
    return ValidityInput(
        missing_frac=max(float(ref["missing_frac"]), float(cand["missing_frac"])),
        infra_frac=max(float(ref["infra_frac"]), float(cand["infra_frac"])),
        token_coverage=min(
            float(ref["token_coverage"]), float(cand["token_coverage"])
        ),
        complete=bool(ref["complete"] and cand["complete"]),
        agent_fail_frac=max(
            float(ref["agent_fail_frac"]), float(cand["agent_fail_frac"])
        ),
    )


def _hit_text(hit: object) -> str:
    if not all(hasattr(hit, field) for field in ("kind", "path", "start", "end", "reason")):
        raise RoundError(f"candidate protection hit is malformed: {hit!r}")
    return f"{hit.kind} {hit.path}:{hit.start}-{hit.end} ({hit.reason})"


def decide_from_artifacts(
    *,
    pools: Mapping[str, Sequence[str]],
    candidate: Candidate,
    applied: Mapping[str, object],
    reference: Sequence[TrialOutcome],
    evaluated: Sequence[TrialOutcome],
    k: int,
    scores_pool: str = "evolution",
    holdout_pool: str = "holdout",
    round: int = 0,
    protection_until: int = 0,
    task_margin: float = 0.10,
    margin: float = 0.10,
    holdout_margin: float = 0.10,
    direction_delta: float = 0.0,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CandidateDecision:
    if not isinstance(candidate, Candidate):
        raise RoundError(f"candidate must be a Candidate: {candidate!r}")
    if not isinstance(applied, Mapping):
        raise RoundError(f"applied artifacts must be a mapping: {applied!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RoundError(f"k must be a positive integer: {k!r}")
    scores_tasks = _pool_tasks(pools, scores_pool)
    holdout_tasks = _pool_tasks(pools, holdout_pool)
    overlap = sorted(set(scores_tasks) & set(holdout_tasks))
    if overlap:
        raise RoundError(
            f"score and holdout pools overlap at task(s): {', '.join(overlap)}"
        )
    tasks = scores_tasks + holdout_tasks
    for arm, outcomes in (("reference", reference), ("evaluated", evaluated)):
        for outcome in outcomes:
            if not isinstance(outcome, TrialOutcome):
                raise RoundError(
                    f"{arm} outcomes must be TrialOutcome instances: {outcome!r}"
                )

    try:
        ref_summary = summarize_trials(reference, tasks=tasks, k=k)
        cand_summary = summarize_trials(evaluated, tasks=tasks, k=k)
    except BaselineError as exc:
        raise RoundError(f"invalid outcome artifacts: {exc}") from exc

    scores_reference = _means(ref_summary, scores_tasks, "reference")
    scores_evaluated = _means(cand_summary, scores_tasks, "evaluated")
    holdout_reference = _means(ref_summary, holdout_tasks, "reference")
    holdout_evaluated = _means(cand_summary, holdout_tasks, "evaluated")

    feedback_delta = sum(scores_evaluated) / len(scores_evaluated) - sum(
        scores_reference
    ) / len(scores_reference)
    heldout_delta = sum(holdout_evaluated) / len(holdout_evaluated) - sum(
        holdout_reference
    ) / len(holdout_reference)
    try:
        direction = direction_consistency(
            feedback_delta, heldout_delta, delta=direction_delta
        )
    except GeneralizationError as exc:
        raise RoundError(f"invalid direction inputs: {exc}") from exc

    before = _kvector(applied, "before")
    after = _kvector(applied, "after")
    verify = _verify(applied)

    undeletable = applied.get("undeletable", False)
    if not isinstance(undeletable, bool):
        raise RoundError(f"applied undeletable must be a bool: {undeletable!r}")
    undeletable_reason = applied.get("undeletable_reason", "")
    if not isinstance(undeletable_reason, str):
        raise RoundError(
            f"applied undeletable_reason must be a string: {undeletable_reason!r}"
        )

    hits = tuple(_hit_text(hit) for hit in candidate.protection)
    token_before, before_missing = token_evidence(ref_summary, tasks)
    token_after, after_missing = token_evidence(cand_summary, tasks)
    request_tokens = TokenInput(
        before=token_before,
        after=token_after,
        before_missing=before_missing,
        after_missing=after_missing,
    )

    try:
        request = GateRequest(
            validity=merged_validity(ref_summary, cand_summary),
            scores=ScoreInput(
                tasks=scores_tasks,
                reference=scores_reference,
                candidate=scores_evaluated,
                margin=margin,
            ),
            holdout=ScoreInput(
                tasks=holdout_tasks,
                reference=holdout_reference,
                candidate=holdout_evaluated,
                margin=holdout_margin,
            ),
            k=KGateInput(before=before, after=after),
            tokens=request_tokens,
            protection=ProtectionInput(hits=hits),
            round=round,
            protection_until=protection_until,
            task_margin=task_margin,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
    except GateError as exc:
        raise RoundError(f"gate inputs are invalid: {exc}") from exc

    try:
        return decide_candidate(
            candidate.id,
            candidate.layer,
            verify,
            request,
            undeletable=undeletable,
            undeletable_reason=undeletable_reason,
            direction=direction,
        )
    except SimplifyError as exc:
        raise RoundError(f"cannot decide candidate: {exc}") from exc
