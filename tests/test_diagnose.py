from __future__ import annotations

import json

import pytest

from evo.application.diagnose import (
    DIAGNOSE_SYSTEM,
    DiagnoseError,
    DiagnosisCase,
    DiagnosisComponent,
    diagnose,
    merge_scores,
    parse_cases,
    parse_ranking,
    rank_prompt,
    score_case,
)


class Scripted:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("scripted model ran out of answers")
        answer = self.answers.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer)


def components():
    return (
        DiagnosisComponent("core", "turn loop", 120),
        DiagnosisComponent("parser", "output parsing", 80),
        DiagnosisComponent("session", "session state", 40),
    )


def case(**overrides):
    payload = {
        "case_id": "f01",
        "failure": "TimeoutError raised while parsing the model reply",
        "components": components(),
        "culprit": "parser",
        "category": "harness",
    }
    payload.update(overrides)
    return DiagnosisCase(**payload)


def answer(top3, cause="harness"):
    return {"top3": list(top3), "cause": cause}


def test_rank_prompt_lists_components_and_demands_json():
    prompt = rank_prompt(case())
    assert "parser" in prompt
    assert "output parsing" in prompt
    assert "80" in prompt
    assert "top3" in prompt
    assert "TimeoutError" in prompt
    assert DIAGNOSE_SYSTEM


def test_merge_scores_builds_a_before_after_document():
    merged = merge_scores(
        {"before": {"top1": 0.2, "top3": 0.5}},
        {"after": {"top1": 0.4, "top3": 0.7}},
    )
    assert merged == {
        "before": {"top1": 0.2, "top3": 0.5},
        "after": {"top1": 0.4, "top3": 0.7},
    }


def test_merge_scores_rejects_duplicate_phases():
    with pytest.raises(DiagnoseError):
        merge_scores({"before": {"top1": 0.2}}, {"before": {"top1": 0.4}})


def test_rank_prompt_preserves_order_without_a_seed():
    prompt = rank_prompt(case())
    assert prompt.index("core") < prompt.index("parser") < prompt.index("session")


def test_rank_prompt_shuffle_is_deterministic_for_a_seed():
    first = rank_prompt(case(), seed=7)
    second = rank_prompt(case(), seed=7)
    assert first == second
    for component in components():
        assert component.id in first


def test_parse_ranking_accepts_a_strict_json_answer():
    ranking, cause = parse_ranking(
        '{"top3": ["parser", "core", "session"], "cause": "harness"}'
    )
    assert ranking == ["parser", "core", "session"]
    assert cause == "harness"


def test_parse_ranking_tolerates_a_json_fence():
    ranking, _ = parse_ranking(
        '```json\n{"top3": ["parser"], "cause": "environment"}\n```'
    )
    assert ranking == ["parser"]


def test_parse_ranking_rejects_malformed_or_empty_answers():
    with pytest.raises(DiagnoseError, match="JSON"):
        parse_ranking("not json")
    with pytest.raises(DiagnoseError, match="top3"):
        parse_ranking('{"cause": "harness"}')
    with pytest.raises(DiagnoseError, match="top3"):
        parse_ranking('{"top3": [], "cause": "harness"}')
    with pytest.raises(DiagnoseError, match="cause"):
        parse_ranking('{"top3": ["parser"], "cause": "maybe"}')


def test_score_case_counts_top1_top3_and_binary():
    scored = score_case(case(), ["parser", "core"], "harness")
    assert scored == {"top1": True, "top3": True, "binary": True}
    scored = score_case(case(), ["core", "session"], "environment")
    assert scored == {"top1": False, "top3": False, "binary": False}


def test_environment_cases_have_no_component_ground_truth():
    environment_case = case(
        case_id="e01", culprit=None, category="environment"
    )
    scored = score_case(environment_case, ["core"], "environment")
    assert scored["top1"] is None
    assert scored["top3"] is None
    assert scored["binary"] is True


def test_diagnose_takes_the_majority_over_repeats():
    model = Scripted(
        answer(["parser", "core"]),
        answer(["core", "parser"]),
        answer(["parser", "session"]),
    )
    result = diagnose([case()], model, repeats=3)
    assert result["metrics"]["top1"] == 1.0
    assert result["metrics"]["top3"] == 1.0
    assert result["metrics"]["binary"] == 1.0
    assert len(model.prompts) == 3
    assert result["per_case"][0]["answers"][0]["ranking"] == ["parser", "core"]


def test_diagnose_scores_malformed_answers_as_wrong():
    model = Scripted("not json", answer(["parser"]), answer(["parser"]))
    result = diagnose([case()], model, repeats=3)
    assert result["metrics"]["top1"] == 1.0
    assert result["per_case"][0]["answers"][0]["error"]
    assert result["per_case"][0]["top1"] is True


def test_diagnose_aggregates_only_cases_with_a_culprit():
    cases = [
        case(),
        case(case_id="e01", culprit=None, category="environment"),
    ]
    model = Scripted(
        answer(["core"], "harness"),
        answer(["core"], "harness"),
    )
    result = diagnose(cases, model, repeats=1)
    assert result["metrics"]["n_cases"] == 2
    assert result["metrics"]["n_ranked"] == 1
    assert result["metrics"]["top1"] == 0.0
    assert result["metrics"]["binary"] == 0.5


def test_parse_cases_validates_ids_culprits_and_categories():
    good = {
        "cases": [
            {
                "case_id": "f01",
                "failure": "boom",
                "components": [{"id": "core"}, {"id": "parser"}],
                "culprit": "core",
                "category": "harness",
            }
        ]
    }
    parsed = parse_cases(good)
    assert parsed[0].components[1].id == "parser"

    with pytest.raises(DiagnoseError, match="culprit"):
        parse_cases(
            {
                "cases": [
                    {
                        "case_id": "f01",
                        "failure": "boom",
                        "components": [{"id": "core"}],
                        "culprit": "missing",
                    }
                ]
            }
        )
    with pytest.raises(DiagnoseError, match="duplicate"):
        parse_cases(
            {
                "cases": [
                    {
                        "case_id": "f01",
                        "failure": "boom",
                        "components": [{"id": "core"}],
                        "culprit": "core",
                    },
                    {
                        "case_id": "f01",
                        "failure": "bang",
                        "components": [{"id": "core"}],
                        "culprit": "core",
                    },
                ]
            }
        )
    with pytest.raises(DiagnoseError, match="category"):
        parse_cases(
            {
                "cases": [
                    {
                        "case_id": "f01",
                        "failure": "boom",
                        "components": [{"id": "core"}],
                        "culprit": "core",
                        "category": "weather",
                    }
                ]
            }
        )


def test_empty_case_list_is_rejected():
    with pytest.raises(DiagnoseError, match="at least one"):
        parse_cases([])
    with pytest.raises(DiagnoseError, match="at least one"):
        diagnose([], Scripted(), repeats=1)


def test_repeats_must_be_positive():
    with pytest.raises(DiagnoseError, match="repeats"):
        diagnose([case()], Scripted(), repeats=0)
