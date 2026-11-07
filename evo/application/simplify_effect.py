from __future__ import annotations

from typing import Mapping, Sequence

from evo.application.baseline import TrialOutcome
from evo.application.compare import CompareError, compare_arms
from evo.contracts.types import KVector


K_METRICS = ("k_lines_nb", "k_ast", "k_chars", "k_bytes", "k_gzip", "k_cc")


class SimplifyEffectError(ValueError):
    """A simplification-effect input is malformed."""


def _k_gate(before: KVector, after: KVector) -> dict[str, object]:
    checks = {
        "k_ast_drop_0_5pct": after.k_ast <= before.k_ast * 0.995,
        "k_lines_nb_drop_0_5pct": after.k_lines_nb <= before.k_lines_nb * 0.995,
        "k_lines_nb_drop_10": before.k_lines_nb - after.k_lines_nb >= 10,
        "k_chars_not_up": after.k_chars <= before.k_chars,
        "k_cc_not_up": after.k_cc <= before.k_cc,
    }
    reasons = [name for name, ok in checks.items() if not ok]
    return {
        "status": "pass" if not reasons else "fail",
        "checks": checks,
        "reasons": reasons,
    }


def _layers(batch: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    layers: dict[str, dict[str, object]] = {}
    for row in batch:
        if not isinstance(row, Mapping):
            raise SimplifyEffectError("batch rows must be mappings")
        layer = row.get("layer")
        name = str(layer) if isinstance(layer, str) and layer else "unknown"
        delta = row.get("delta")
        if not isinstance(delta, Mapping):
            delta = {}
        entry = layers.setdefault(
            name,
            {
                "count": 0,
                "ids": [],
                "delta_k_lines_nb": 0,
                "delta_k_ast": 0,
                "delta_k_chars": 0,
                "delta_k_cc": 0,
            },
        )
        entry["count"] = int(entry["count"]) + 1
        entry["ids"].append(row.get("id"))
        for metric in ("k_lines_nb", "k_ast", "k_chars", "k_cc"):
            value = delta.get(metric, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            entry[f"delta_{metric}"] = int(entry[f"delta_{metric}"]) + int(value)
    return layers


def _diagnosis(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if before is None or after is None:
        return None
    values: dict[str, object] = {}
    for metric in ("top1", "top3"):
        start = before.get(metric)
        end = after.get(metric)
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            continue
        values[f"before_{metric}"] = round(float(start), 6)
        values[f"after_{metric}"] = round(float(end), 6)
        values[f"delta_{metric}"] = round(float(end) - float(start), 6)
    if "delta_top1" not in values:
        return None
    values["improved"] = bool(
        values.get("delta_top1", 0) >= 0 and values.get("delta_top3", 0) >= 0
    )
    return values


def _control(name: str, payload: Mapping[str, object] | None) -> dict[str, object] | None:
    if payload is None:
        return None
    aggregate = payload.get("aggregate")
    k_block = payload.get("k")
    if not isinstance(aggregate, Mapping):
        return None
    out: dict[str, object] = {
        "non_inferior": aggregate.get("non_inferior"),
        "mean_delta": aggregate.get("mean_delta"),
    }
    if isinstance(k_block, Mapping):
        reduction = k_block.get("reduction_pct")
        if isinstance(reduction, Mapping):
            out["k_ast_reduction_pct"] = reduction.get("k_ast")
            out["k_lines_nb_reduction_pct"] = reduction.get("k_lines_nb")
    return {"arm": name, **out}


def build_simplify_effect(
    reference: Sequence[TrialOutcome],
    candidate: Sequence[TrialOutcome],
    *,
    tasks: Sequence[str],
    holdout: Sequence[str],
    k: int,
    k_before: KVector,
    k_after: KVector,
    batch: Sequence[Mapping[str, object]] = (),
    diagnosis_before: Mapping[str, object] | None = None,
    diagnosis_after: Mapping[str, object] | None = None,
    placebo: Mapping[str, object] | None = None,
    sharp: Mapping[str, object] | None = None,
    margin: float = 0.10,
    holdout_margin: float = 0.10,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Measure whether a simplification actually reduces K without losing S."""
    if not isinstance(k_before, KVector) or not isinstance(k_after, KVector):
        raise SimplifyEffectError("k_before and k_after must be KVector instances")
    try:
        comparison = compare_arms(
            reference,
            candidate,
            tasks=tasks,
            k=k,
            holdout=holdout,
            margin=margin,
            holdout_margin=holdout_margin,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
            k_before=k_before,
            k_after=k_after,
        )
    except CompareError as exc:
        raise SimplifyEffectError(str(exc)) from exc

    aggregate = comparison["aggregate"]
    k_block = comparison["k"]
    score = {
        "reference_mean": aggregate["reference_mean"],
        "candidate_mean": aggregate["candidate_mean"],
        "mean_delta": aggregate["mean_delta"],
        "ci": aggregate["ci"],
        "margin": aggregate["margin"],
        "non_inferior": aggregate["non_inferior"],
        "worst_task": aggregate["worst_task"],
        "holdout": aggregate.get("holdout"),
    }
    k_gate = _k_gate(k_before, k_after)
    input_valid = (
        comparison["validity"]["reference"]["status"] == "pass"
        and comparison["validity"]["candidate"]["status"] == "pass"
    )
    reasons: list[str] = []
    if not input_valid:
        reasons.append("measurement validity failed")
    if k_gate["status"] != "pass":
        reasons.append("K gate failed: " + ", ".join(k_gate["reasons"]))
    if not score["non_inferior"]:
        reasons.append("score non-inferiority failed")
    held = score["holdout"]
    if isinstance(held, Mapping) and not held.get("non_inferior"):
        reasons.append("holdout non-inferiority failed")
    tokens = aggregate["tokens"]
    if tokens["candidate"]["total"] > tokens["reference"]["total"]:
        reasons.append("token total increased")
    worst = score["worst_task"]
    if isinstance(worst, Mapping):
        delta = worst.get("delta")
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            if float(delta) < -margin:
                reasons.append(f"worst task {worst.get('task')} regressed by {delta}")

    reduction = k_block["reduction_pct"]
    ci = aggregate["ci"]
    claim = (
        f"K_ast {reduction['k_ast']:.2f}% lower and K_lines_nb "
        f"{reduction['k_lines_nb']:.2f}% lower; score delta "
        f"{aggregate['mean_delta']:+.4f} (CI {ci['low']:+.4f}, {ci['high']:+.4f}); "
        f"holdout "
        + (
            f"delta {held['mean_delta']:+.4f} (CI {held['ci']['low']:+.4f}, "
            f"{held['ci']['high']:+.4f})"
            if isinstance(held, Mapping)
            else "not measured"
        )
        + f"; token ratio {tokens['candidate']['ratio']}"
    )
    return {
        "schema": 1,
        "verdict": "simplification_effective" if not reasons else "not_supported",
        "reasons": reasons,
        "claim": claim,
        "k": k_block,
        "k_gate": k_gate,
        "score": score,
        "tokens": tokens,
        "cost": aggregate["cost"],
        "layers": _layers(batch),
        "diagnosis": _diagnosis(diagnosis_before, diagnosis_after),
        "controls": [
            row
            for row in (_control("PLACEBO", placebo), _control("SHARP", sharp))
            if row is not None
        ],
        "validity": comparison["validity"],
    }
