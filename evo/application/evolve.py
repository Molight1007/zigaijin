from __future__ import annotations

import math
from typing import Mapping, Sequence

from evo.application.baseline import BaselineError, TrialOutcome, summarize_trials
from evo.domain.evolve import (
    EvolveError,
    EvolveMeasurement,
    EvolvePolicy,
    edit_budget,
    exploration,
    prune_set,
    select_round,
    stall_flag,
    tried_components,
)
from evo.domain.prediction import PredictionError, verify_prediction


class EvolveRoundError(ValueError):
    """An evolution round cannot be assembled under the pre-registered method."""


def _outcomes(value: object, name: str) -> list[TrialOutcome]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvolveRoundError(f"{name} must be a sequence of trial outcomes")
    rows = list(value)
    for row in rows:
        if not isinstance(row, TrialOutcome):
            raise EvolveRoundError(f"{name} contains a non-TrialOutcome: {row!r}")
    return rows


def _summary(outcomes: Sequence[TrialOutcome], tasks: Sequence[str], k: int, name: str) -> dict:
    try:
        return summarize_trials(outcomes, tasks=tasks, k=k)
    except BaselineError as exc:
        raise EvolveRoundError(f"{name}: {exc}") from exc


def _tokens_per_trial(summary: Mapping[str, object]) -> float | None:
    aggregate = summary["aggregate"]
    if not aggregate.get("tokens_complete", False):
        return None
    valid = int(aggregate["valid_trials"])
    if valid == 0:
        return None
    total = int(aggregate["tokens_in"]) + int(aggregate["tokens_out"])
    return total / valid


def _per_task(summary: Mapping[str, object]) -> dict[str, float | None]:
    return {
        task: row["mean_reward"]
        for task, row in summary["tasks"].items()
    }


def _accepted_scores(
    history: Sequence[Mapping[str, object]], current_round: int
) -> dict[int, float]:
    """Best accepted score per earlier round; feeds the S* floor, never S_t (T1).

    Rows stamped with the current round are ignored so a re-run cannot self-lift.
    """
    scores: dict[int, float] = {}
    for row in history:
        t = row.get("t")
        score = row.get("score")
        if (
            isinstance(t, int)
            and not isinstance(t, bool)
            and t < current_round
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and row.get("accepted") is True
        ):
            scores[t] = max(scores.get(t, -math.inf), float(score))
    return scores


def _attribution(
    edit: Mapping[str, object],
    incumbent_means: Mapping[str, float | None],
    candidate_means: Mapping[str, float | None],
    k: int,
    threshold: float | None = None,
) -> dict[str, object]:
    predicted = [str(task) for task in (edit.get("predicted_affected") or [])]
    try:
        return verify_prediction(
            predicted, incumbent_means, candidate_means, k=k, threshold=threshold
        )
    except PredictionError as exc:
        raise EvolveRoundError(f"attribution: {exc}") from exc


def _history_rows(
    *,
    round_number: int,
    variant: str,
    edits: Sequence[Mapping[str, object]],
    decision: dict[str, object],
    measured: bool,
    outcome: str,
    accepted: bool,
    k: int,
    incumbent_means: Mapping[str, float | None],
    candidate_means: Mapping[str, float | None],
    s_star: float,
    threshold: float | None = None,
) -> list[dict[str, object]]:
    rows = edits or [{"id": "C1", "component": None, "hypothesis": None}]
    records = []
    for edit in rows:
        record = {
            "t": round_number,
            "variant": variant,
            "edit_id": edit.get("id"),
            "component": edit.get("component"),
            "hypothesis": edit.get("hypothesis"),
            "targets_mode": edit.get("targets_mode"),
            "predicted_affected": list(edit.get("predicted_affected") or []),
            "delta_S": decision["delta_score"] if measured else None,
            "delta_C": decision["delta_tokens"] if measured else None,
            "accepted": accepted,
            "outcome": outcome,
            "score": decision["score"],
            "s_star": s_star,
            "tokens_per_trial": decision["tokens_per_trial"],
            "bundle": len(edits) if edits else 1,
            "detail": str(decision["reason"])[:600],
            "prediction_verified": None,
            "prediction_falsified": None,
        }
        if measured:
            record.update(
                _attribution(edit, incumbent_means, candidate_means, k, threshold)
            )
        records.append(record)
    return records


def evolve_round(
    *,
    incumbent: Sequence[TrialOutcome],
    candidates: Mapping[str, Sequence[TrialOutcome]],
    tasks: Sequence[str],
    k: int,
    delta: float,
    round: int,
    total_rounds: int,
    policy: EvolvePolicy = EvolvePolicy(),
    trajectory: Sequence[float] = (),
    components: Sequence[str] = (),
    history: Sequence[Mapping[str, object]] = (),
    edits: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    gate_failures: Mapping[str, str] | None = None,
    s_star: float | None = None,
    k_lines: Mapping[str, int] | None = None,
    incumbent_k_lines: int | None = None,
    max_k_growth: int = 0,
    n_prune: int = 4,
    stall_window: int = 3,
    m_draft: int = 1,
    b_min: int = 1,
    b_max: int = 4,
) -> dict[str, object]:
    """Judge one evolution round from stored measurements (Algorithm 2).

    Rejects bundles over b_t; restores s_star and the stall trajectory from
    history; caps candidates at incumbent_k_lines + max_k_growth. Floors are
    S* - delta, never S_t - delta (T1).
    """
    if not isinstance(policy, EvolvePolicy):
        raise EvolveRoundError("policy must be an EvolvePolicy")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise EvolveRoundError(f"k must be a positive integer: {k!r}")
    if isinstance(total_rounds, bool) or not isinstance(total_rounds, int) or total_rounds <= 0:
        raise EvolveRoundError(
            f"total_rounds must be a positive integer: {total_rounds!r}"
        )
    if isinstance(round, bool) or not isinstance(round, int) or round < 0:
        raise EvolveRoundError(f"round must be a non-negative integer: {round!r}")
    if round >= total_rounds:
        raise EvolveRoundError(
            f"round {round} is outside the pre-registered horizon of {total_rounds}"
        )
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        raise EvolveRoundError(f"delta must be a number: {delta!r}")
    threshold = float(delta)
    if not math.isfinite(threshold) or threshold < 0:
        raise EvolveRoundError(f"delta must be finite and non-negative: {delta!r}")

    incumbent_rows = _outcomes(incumbent, "incumbent")
    if not isinstance(candidates, Mapping):
        raise EvolveRoundError("candidates must be a mapping of variant to outcomes")
    if not isinstance(edits, Mapping) and edits is not None:
        raise EvolveRoundError("edits must be a mapping of variant to edit records")
    edit_map = edits or {}
    unknown_edits = sorted(set(edit_map) - set(candidates))
    if unknown_edits:
        raise EvolveRoundError(
            f"edits were supplied for unknown variants: {', '.join(unknown_edits)}"
        )
    if not isinstance(gate_failures, Mapping) and gate_failures is not None:
        raise EvolveRoundError(
            "gate_failures must be a mapping of variant to failure reason"
        )
    gate_map = dict(gate_failures or {})
    unknown_gates = sorted(set(gate_map) - set(candidates))
    if unknown_gates:
        raise EvolveRoundError(
            f"gate failures were supplied for unknown variants: {', '.join(unknown_gates)}"
        )
    for variant, reason in gate_map.items():
        if not isinstance(reason, str) or not reason.strip():
            raise EvolveRoundError(
                f"gate failure for {variant!r} must be a non-empty string: {reason!r}"
            )
    if not isinstance(k_lines, Mapping) and k_lines is not None:
        raise EvolveRoundError("k_lines must be a mapping of variant to line counts")
    line_map = k_lines or {}
    if incumbent_k_lines is not None:
        if (
            isinstance(incumbent_k_lines, bool)
            or not isinstance(incumbent_k_lines, int)
            or incumbent_k_lines < 0
        ):
            raise EvolveRoundError(
                "incumbent_k_lines must be a non-negative integer or None"
            )
    if (
        isinstance(max_k_growth, bool)
        or not isinstance(max_k_growth, int)
        or max_k_growth < 0
    ):
        raise EvolveRoundError("max_k_growth must be a non-negative integer")
    k_limit = None if incumbent_k_lines is None else incumbent_k_lines + max_k_growth

    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        raise EvolveRoundError("trajectory must be a sequence of scores")
    scores = []
    for value in trajectory:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvolveRoundError(f"trajectory contains a non-number: {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise EvolveRoundError(f"trajectory contains a non-finite score: {value!r}")
        scores.append(number)
    if scores and len(scores) != round + 1:
        raise EvolveRoundError(
            f"trajectory has {len(scores)} scores for round {round}; expected {round + 1}"
        )

    history_rows = list(history)
    for index, record in enumerate(history_rows):
        if not isinstance(record, Mapping):
            raise EvolveRoundError(f"history record {index} is not an object")

    incumbent_summary = _summary(incumbent_rows, tasks, k, "incumbent")
    validity = incumbent_summary["validity"]
    if validity["status"] != "pass":
        reasons = "; ".join(validity["reasons"]) or "unknown validity failure"
        raise EvolveRoundError(f"incumbent evaluation is invalid: {reasons}")
    incumbent_score = incumbent_summary["aggregate"]["mean_reward"]
    if incumbent_score is None:
        raise EvolveRoundError("incumbent evaluation has no score")
    incumbent_score = float(incumbent_score)
    incumbent_tokens = _tokens_per_trial(incumbent_summary)
    if s_star is not None and (
        isinstance(s_star, bool) or not isinstance(s_star, (int, float))
    ):
        raise EvolveRoundError(f"s_star must be a number: {s_star!r}")
    prior_scores = _accepted_scores(history_rows, round)
    if s_star is None:
        star = max([incumbent_score, *prior_scores.values()])
    else:
        star = float(s_star)
    if not math.isfinite(star):
        source = "s_star" if s_star is not None else "history accepted score"
        raise EvolveRoundError(f"{source} must be finite: {star!r}")
    if not scores:
        rebuilt = [prior_scores[t] for t in range(round) if t in prior_scores]
        if len(rebuilt) == round:
            scores = [*rebuilt, incumbent_score]

    try:
        budget = edit_budget(round, total_rounds, b_min=b_min, b_max=b_max)
    except EvolveError as exc:
        raise EvolveRoundError(f"edit budget: {exc}") from exc
    for variant, declared in edit_map.items():
        count = len(list(declared))
        if count > budget:
            raise EvolveRoundError(
                f"candidate {variant}: {count} edits exceed the "
                f"pre-registered budget b_t={budget}"
            )

    measurements: list[EvolveMeasurement] = []
    summaries: dict[str, dict] = {}
    for variant in sorted(set(candidates) | set(gate_map)):
        variant_edits = list(edit_map.get(variant, []))
        for index, edit in enumerate(variant_edits):
            if not isinstance(edit, Mapping):
                raise EvolveRoundError(
                    f"candidate {variant} edit {index} is not an object"
                )
        summary = None
        if variant in candidates and variant not in gate_map:
            rows = _outcomes(candidates[variant], f"candidate {variant}")
            summary = _summary(rows, tasks, k, f"candidate {variant}")
            summaries[variant] = summary
        measurements.append(
            EvolveMeasurement(
                variant=variant,
                score=None if summary is None else summary["aggregate"]["mean_reward"],
                tokens_per_trial=(
                    None if summary is None else _tokens_per_trial(summary)
                ),
                valid=True if summary is None else summary["validity"]["status"] == "pass",
                gate_failure=gate_map.get(variant),
                components=tuple(
                    str(edit["component"])
                    for edit in variant_edits
                    if edit.get("component")
                ),
                k_lines=line_map.get(variant),
                k_limit=k_limit,
            )
        )

    tried = tried_components(history_rows)
    accepted_counts: dict[str, int] = {}
    for record in history_rows:
        if record.get("accepted") is True and record.get("component"):
            name = str(record["component"])
            accepted_counts[name] = accepted_counts.get(name, 0) + 1

    stalled = (
        stall_flag(scores, round, window=stall_window, delta=threshold)
        if scores
        else False
    )
    if components:
        try:
            explore = exploration(
                components, tried, stalled=stalled, m_draft=m_draft
            )
        except EvolveError as exc:
            raise EvolveRoundError(f"exploration: {exc}") from exc
    else:
        explore = {
            "sigma": stalled,
            "untried": [],
            "m_draft": m_draft,
            "reserved": 0,
            "text": "no component vocabulary was declared",
        }
    try:
        prune = prune_set(history_rows, t=round, n_prune=n_prune)
    except EvolveError as exc:
        raise EvolveRoundError(f"prune set: {exc}") from exc

    try:
        winner, decisions = select_round(
            measurements,
            incumbent_score=incumbent_score,
            incumbent_tokens=incumbent_tokens,
            s_star=star,
            delta=threshold,
            policy=policy,
            accepted_counts=accepted_counts,
        )
    except EvolveError as exc:
        raise EvolveRoundError(f"selection: {exc}") from exc

    incumbent_means = _per_task(incumbent_summary)
    records: list[dict[str, object]] = []
    for measurement, decision in zip(measurements, decisions):
        summary = summaries.get(measurement.variant)
        candidate_means = _per_task(summary) if summary is not None else {}
        measured = (
            measurement.valid
            and measurement.score is not None
            and measurement.gate_failure is None
        )
        if not measured:
            outcome = "eval_invalid"
        elif decision.admissible:
            outcome = "ACCEPTED" if winner is measurement else "LOST"
        else:
            outcome = "REJECTED"
        records.extend(
            _history_rows(
                round_number=round,
                variant=measurement.variant,
                edits=list(edit_map.get(measurement.variant, [])),
                decision=decision.as_dict(),
                measured=measured,
                outcome=outcome,
                accepted=winner is measurement,
                k=k,
                incumbent_means=incumbent_means,
                candidate_means=candidate_means,
                s_star=star,
                threshold=threshold,
            )
        )

    new_star = star
    if winner is not None and winner.score is not None:
        new_star = max(star, float(winner.score))
    return {
        "schema": 1,
        "round": round,
        "total_rounds": total_rounds,
        "delta": threshold,
        "k": k,
        "k_limit": k_limit,
        "policy": policy.as_dict(),
        "budget": budget,
        "stall": stalled,
        "exploration": explore,
        "prune_set": prune,
        "incumbent": {
            "score": incumbent_score,
            "tokens_per_trial": incumbent_tokens,
            "valid_trials": incumbent_summary["aggregate"]["valid_trials"],
        },
        "s_star": new_star,
        "winner": None if winner is None else winner.variant,
        "decisions": [decision.as_dict() for decision in decisions],
        "history": records,
    }
