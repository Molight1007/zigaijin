import json
import math

import pytest

from evo.contracts.types import Component, KResult, KVector
from evo.domain.ablation import (
    AblationDecision,
    AblationError,
    AblationPlan,
    Lever,
    attributed_k,
    decide_ablation,
    load_ablation_plan,
    validity_gate,
)


def make_lever(**overrides) -> Lever:
    fields = {
        "id": "terminal_recording",
        "flag": "record_terminal_session",
        "default_value": True,
        "off_value": False,
        "components": ("recording",),
        "description": "asciinema capture",
    }
    fields.update(overrides)
    return Lever(**fields)


def write_plan(tmp_path, payload) -> "object":
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_ablation_plan(path)


def valid_payload() -> dict:
    return {
        "subject": "terminus2",
        "levers": [
            {
                "id": "terminal_recording",
                "flag": "record_terminal_session",
                "default": True,
                "off": False,
                "components": ["recording"],
                "description": "asciinema capture",
            }
        ],
    }


# ---- Lever / AblationPlan -------------------------------------------------


def test_plan_exposes_constructor_kwargs():
    plan = AblationPlan(subject="terminus2", levers=(make_lever(),))
    assert plan.ids == ("terminal_recording",)
    assert plan.kwargs_for("terminal_recording") == {"record_terminal_session": False}


def test_plan_rejects_duplicate_lever_ids():
    with pytest.raises(AblationError, match="duplicate lever id"):
        AblationPlan(subject="terminus2", levers=(make_lever(), make_lever()))


def test_plan_rejects_two_levers_on_the_same_flag():
    second = make_lever(id="recording_alias")
    with pytest.raises(AblationError, match="duplicate flag"):
        AblationPlan(subject="terminus2", levers=(make_lever(), second))


def test_lever_must_actually_change_the_flag():
    with pytest.raises(AblationError, match="off value"):
        make_lever(default_value=True, off_value=True)


def test_lever_rejects_zero_off_value_against_false_default():
    with pytest.raises(AblationError, match="off value"):
        make_lever(default_value=False, off_value=0)


def test_lever_rejects_invalid_id():
    with pytest.raises(AblationError, match="lever id"):
        make_lever(id="Terminal Recording")


def test_lever_rejects_invalid_flag():
    with pytest.raises(AblationError, match="flag"):
        make_lever(flag="record-terminal")


def test_lever_requires_component_attribution():
    with pytest.raises(AblationError, match="component"):
        make_lever(components=())


def test_lever_rejects_duplicate_components():
    with pytest.raises(AblationError, match="duplicate component"):
        make_lever(components=("recording", "recording"))


def test_lever_rejects_non_scalar_off_value():
    with pytest.raises(AblationError, match="scalar"):
        make_lever(off_value=["off"])


def test_kwargs_for_unknown_lever_fails():
    plan = AblationPlan(subject="terminus2", levers=(make_lever(),))
    with pytest.raises(AblationError, match="unknown lever"):
        plan.kwargs_for("nope")


def test_plan_rejects_invalid_subject():
    with pytest.raises(AblationError, match="subject"):
        AblationPlan(subject="terminus 2", levers=(make_lever(),))


# ---- load_ablation_plan ---------------------------------------------------


def test_load_plan_roundtrips(tmp_path):
    plan = write_plan(tmp_path, valid_payload())
    assert plan.subject == "terminus2"
    assert plan.ids == ("terminal_recording",)
    assert plan.kwargs_for("terminal_recording") == {"record_terminal_session": False}
    assert plan.levers[0].description == "asciinema capture"


def test_load_plan_description_is_optional(tmp_path):
    payload = valid_payload()
    del payload["levers"][0]["description"]
    plan = write_plan(tmp_path, payload)
    assert plan.levers[0].description == ""


def test_load_plan_rejects_malformed_json(tmp_path):
    path = tmp_path / "ablation.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(AblationError, match="JSON"):
        load_ablation_plan(path)


def test_load_plan_rejects_unknown_top_level_key(tmp_path):
    payload = valid_payload()
    payload["extra"] = 1
    with pytest.raises(AblationError, match="unknown key"):
        write_plan(tmp_path, payload)


def test_load_plan_rejects_unknown_lever_key(tmp_path):
    payload = valid_payload()
    payload["levers"][0]["surprise"] = True
    with pytest.raises(AblationError, match="unknown key"):
        write_plan(tmp_path, payload)


def test_load_plan_rejects_missing_levers(tmp_path):
    with pytest.raises(AblationError, match="levers"):
        write_plan(tmp_path, {"subject": "terminus2"})


def test_load_plan_rejects_non_list_levers(tmp_path):
    with pytest.raises(AblationError, match="levers"):
        write_plan(tmp_path, {"subject": "terminus2", "levers": {}})


def test_load_plan_rejects_non_object_lever(tmp_path):
    with pytest.raises(AblationError, match="object"):
        write_plan(tmp_path, {"subject": "terminus2", "levers": ["x"]})


def test_load_plan_rejects_missing_off(tmp_path):
    payload = valid_payload()
    del payload["levers"][0]["off"]
    with pytest.raises(AblationError, match="off"):
        write_plan(tmp_path, payload)


def test_load_plan_rejects_non_file(tmp_path):
    with pytest.raises(AblationError, match="file"):
        load_ablation_plan(tmp_path)


# ---- attributed_k ---------------------------------------------------------


def sample_components() -> tuple[Component, ...]:
    return (
        Component(
            id="terminus2:recording",
            name="recording",
            paths=("asciinema_handler.py",),
        ),
        Component(
            id="terminus2:tmux",
            name="tmux",
            paths=("tmux_session.py",),
        ),
    )


def sample_k() -> KResult:
    return KResult(
        k_lines_nb=30,
        k_ast=100,
        k_chars=900,
        k_bytes=900,
        k_gzip=200,
        k_cc=5,
        files={
            "asciinema_handler.py": KVector(
                k_lines_nb=10,
                k_ast=40,
                k_chars=300,
                k_bytes=300,
                k_gzip=100,
                k_cc=2,
            ),
            "tmux_session.py": KVector(
                k_lines_nb=20,
                k_ast=60,
                k_chars=600,
                k_bytes=600,
                k_gzip=100,
                k_cc=3,
            ),
        },
    )


def test_attributed_k_sums_only_named_components():
    result = attributed_k(make_lever(), sample_components(), sample_k())
    assert result.as_dict() == {
        "k_lines_nb": 10,
        "k_ast": 40,
        "k_chars": 300,
        "k_bytes": 300,
        "k_gzip": 100,
        "k_cc": 2,
    }


def test_attributed_k_deduplicates_shared_files():
    lever = make_lever(components=("recording", "tmux"))
    result = attributed_k(lever, sample_components(), sample_k())
    assert result.k_lines_nb == 30


def test_attributed_k_unknown_component_fails():
    with pytest.raises(AblationError, match="unknown component"):
        attributed_k(
            make_lever(components=("ghost",)), sample_components(), sample_k()
        )


def test_attributed_k_missing_file_fails():
    k = KResult(
        k_lines_nb=10,
        files={"tmux_session.py": KVector(k_lines_nb=10)},
    )
    with pytest.raises(AblationError, match="missing"):
        attributed_k(make_lever(), sample_components(), k)


# ---- validity_gate --------------------------------------------------------


def test_validity_gate_accepts_exact_limits():
    report = validity_gate(missing_frac=0.10, infra_frac=0.10, token_coverage=0.90)
    assert report.ok
    assert report.reasons == ()


def test_validity_gate_rejects_too_much_missing():
    report = validity_gate(missing_frac=0.11, infra_frac=0.0, token_coverage=1.0)
    assert not report.ok
    assert any("missing_frac" in reason for reason in report.reasons)


def test_validity_gate_rejects_too_much_infra():
    report = validity_gate(missing_frac=0.0, infra_frac=0.11, token_coverage=1.0)
    assert not report.ok
    assert any("infra_frac" in reason for reason in report.reasons)


def test_validity_gate_rejects_low_token_coverage():
    report = validity_gate(missing_frac=0.0, infra_frac=0.0, token_coverage=0.89)
    assert not report.ok
    assert any("token_coverage" in reason for reason in report.reasons)


def test_validity_gate_reports_all_failures_in_order():
    report = validity_gate(missing_frac=0.5, infra_frac=0.5, token_coverage=0.0)
    assert [reason.split()[0] for reason in report.reasons] == [
        "missing_frac",
        "infra_frac",
        "token_coverage",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"missing_frac": -0.01, "infra_frac": 0.0, "token_coverage": 1.0},
        {"missing_frac": 1.01, "infra_frac": 0.0, "token_coverage": 1.0},
        {"missing_frac": math.nan, "infra_frac": 0.0, "token_coverage": 1.0},
    ],
)
def test_validity_gate_rejects_invalid_fractions(kwargs):
    with pytest.raises(AblationError):
        validity_gate(**kwargs)


# ---- decide_ablation ------------------------------------------------------


def test_identical_arms_are_zero_contribution():
    tasks = ("a", "b", "c")
    decision = decide_ablation(
        tasks,
        on_rewards=(1.0, 0.0, 1.0),
        off_rewards=(1.0, 0.0, 1.0),
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(100, 200, 100),
        off_tokens=(100, 200, 100),
    )
    assert decision.verdict == "zero_contribution"
    assert decision.delta is not None
    assert decision.delta.low == 0.0
    assert decision.delta.high == 0.0
    assert decision.token_ratio == 1.0
    assert decision.n_pairs == 3
    assert decision.reasons == ()


def test_clear_loss_means_the_lever_contributes():
    tasks = ("a", "b", "c", "d")
    decision = decide_ablation(
        tasks,
        on_rewards=(1.0, 1.0, 1.0, 1.0),
        off_rewards=(0.0, 0.0, 0.0, 0.0),
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(100, 100, 100, 100),
        off_tokens=(100, 100, 100, 100),
    )
    assert decision.verdict == "contributes"
    assert decision.delta is not None
    assert decision.delta.high < -0.10


def test_straddling_interval_is_inconclusive():
    tasks = tuple(f"t{i}" for i in range(10))
    on = (1.0,) * 10
    off = (1.0,) * 8 + (0.0,) * 2
    decision = decide_ablation(
        tasks,
        on_rewards=on,
        off_rewards=off,
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(100,) * 10,
        off_tokens=(100,) * 10,
    )
    assert decision.verdict == "inconclusive"
    assert decision.delta is not None
    assert decision.delta.low < -0.10
    assert decision.delta.high >= -0.10


def test_invalid_measurement_short_circuits():
    decision = decide_ablation(
        ("a",),
        on_rewards=(1.0,),
        off_rewards=(0.0,),
        margin=0.10,
        validity=validity_gate(0.5, 0.0, 1.0),
        on_tokens=(100,),
        off_tokens=(100,),
    )
    assert decision.verdict == "invalid"
    assert decision.delta is None
    assert any("missing_frac" in reason for reason in decision.reasons)


def test_token_total_increase_blocks_deletion():
    decision = decide_ablation(
        ("a", "b"),
        on_rewards=(1.0, 1.0),
        off_rewards=(1.0, 1.0),
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(100, 100),
        off_tokens=(110, 110),
    )
    assert decision.verdict == "contributes"
    assert decision.token_ratio == pytest.approx(1.10)
    assert any("token_total_increased" in reason for reason in decision.reasons)


def test_per_task_token_spike_blocks_deletion_even_if_total_is_flat():
    decision = decide_ablation(
        ("a", "b"),
        on_rewards=(1.0, 1.0),
        off_rewards=(1.0, 1.0),
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(100, 100),
        off_tokens=(112, 88),
    )
    assert decision.verdict == "contributes"
    assert decision.worst_token_task == ("a", pytest.approx(1.12))


def test_token_gate_is_closed_without_token_data():
    decision = decide_ablation(
        ("a",),
        on_rewards=(1.0,),
        off_rewards=(1.0,),
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
    )
    assert decision.verdict == "inconclusive"
    assert decision.token_ratio is None
    assert any("token_gate_closed" in reason for reason in decision.reasons)


def test_one_sided_token_data_is_a_programming_error():
    with pytest.raises(AblationError, match="token"):
        decide_ablation(
            ("a",),
            on_rewards=(1.0,),
            off_rewards=(1.0,),
            margin=0.10,
            validity=validity_gate(0.0, 0.0, 1.0),
            on_tokens=(100,),
        )


def test_negative_tokens_are_rejected():
    with pytest.raises(AblationError, match="token"):
        decide_ablation(
            ("a",),
            on_rewards=(1.0,),
            off_rewards=(1.0,),
            margin=0.10,
            validity=validity_gate(0.0, 0.0, 1.0),
            on_tokens=(-1,),
            off_tokens=(100,),
        )


def test_mismatched_sample_lengths_are_rejected():
    with pytest.raises(AblationError, match="length"):
        decide_ablation(
            ("a", "b"),
            on_rewards=(1.0,),
            off_rewards=(1.0, 1.0),
            margin=0.10,
            validity=validity_gate(0.0, 0.0, 1.0),
        )


def test_non_finite_rewards_are_rejected():
    with pytest.raises(AblationError, match="reward"):
        decide_ablation(
            ("a",),
            on_rewards=(math.nan,),
            off_rewards=(1.0,),
            margin=0.10,
            validity=validity_gate(0.0, 0.0, 1.0),
        )


def test_margin_must_be_positive():
    with pytest.raises(AblationError, match="margin"):
        decide_ablation(
            ("a",),
            on_rewards=(1.0,),
            off_rewards=(1.0,),
            margin=0.0,
            validity=validity_gate(0.0, 0.0, 1.0),
        )


def test_empty_samples_are_rejected():
    with pytest.raises(AblationError, match="empty"):
        decide_ablation(
            (),
            on_rewards=(),
            off_rewards=(),
            margin=0.10,
            validity=validity_gate(0.0, 0.0, 1.0),
        )


def test_decision_is_deterministic_for_a_fixed_seed():
    kwargs = dict(
        task_ids=("a", "b", "c", "d", "e"),
        on_rewards=(1.0, 0.0, 1.0, 1.0, 0.0),
        off_rewards=(1.0, 1.0, 0.0, 1.0, 0.0),
        margin=0.20,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(10, 20, 30, 40, 50),
        off_tokens=(10, 20, 31, 40, 50),
        seed=7,
    )
    first = decide_ablation(**kwargs)
    second = decide_ablation(**kwargs)
    assert first == second


def test_on_total_zero_uses_defined_token_ratio():
    decision = decide_ablation(
        ("a",),
        on_rewards=(1.0,),
        off_rewards=(1.0,),
        margin=0.10,
        validity=validity_gate(0.0, 0.0, 1.0),
        on_tokens=(0,),
        off_tokens=(0,),
    )
    assert decision.token_ratio == 1.0
