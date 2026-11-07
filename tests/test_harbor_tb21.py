import json
import os

import pytest

from evo.adapters.deepseek import ModelProfile
from evo.adapters.harbor_tb21 import (
    HarborError,
    build_run,
    parse_result,
    read_job,
    run_harbor,
)

MODEL = ModelProfile(
    name="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
)


def success_payload(**overrides):
    payload = {
        "task_name": "terminal-bench/fix-git",
        "trial_name": "fix-git__abc123",
        "trial_uri": "file:///runs/fix-git__abc123",
        "agent_result": {
            "n_input_tokens": 120,
            "n_cache_tokens": 20,
            "n_output_tokens": 30,
            "cost_usd": 0.002,
        },
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
        "step_results": None,
    }
    payload.update(overrides)
    return payload


def build(**overrides):
    kwargs = dict(
        python="python.exe",
        dataset="terminal-bench/terminal-bench-2-1",
        task_ids=("terminal-bench/fix-git",),
        model=MODEL,
        jobs_dir="runs/live",
        job_name="job1",
        reps=1,
    )
    kwargs.update(overrides)
    return build_run(**kwargs)


def test_command_targets_terminus2_with_the_model():
    command = build()
    argv = list(command.argv)
    assert "-a" in argv
    assert argv[argv.index("-a") + 1] == "terminus2:AgentHarness"
    model_flag = [index for index, value in enumerate(argv) if value == "-m"][-1]
    assert argv[model_flag + 1] == "deepseek-v4-flash"
    assert argv[argv.index("--base-url") + 1] == "https://api.deepseek.com/v1"
    assert argv[argv.index("--api-key-name") + 1] == "DEEPSEEK_API_KEY"
    assert argv[argv.index("-d") + 1] == "terminal-bench/terminal-bench-2-1"


def test_command_sets_concurrency():
    argv = list(build(concurrency=3).argv)
    assert argv[argv.index("-n") + 1] == "3"


def test_command_rejects_non_positive_concurrency():
    with pytest.raises(HarborError, match="concurrency"):
        build(concurrency=0)


def test_command_sets_repetitions_and_job_paths():
    command = build(reps=3, task_ids=("terminal-bench/a", "terminal-bench/b"))
    argv = list(command.argv)
    assert argv[argv.index("-k") + 1] == "3"
    includes = [argv[i + 1] for i, value in enumerate(argv) if value == "-i"]
    assert includes == ["terminal-bench/a", "terminal-bench/b"]
    assert argv[argv.index("--jobs-dir") + 1] == "runs/live"
    assert argv[argv.index("--job-name") + 1] == "job1"


def test_command_disables_terminal_recording_by_default():
    argv = list(build().argv)
    assert "record_terminal_session=false" in argv


def test_command_includes_the_config_file_when_given():
    argv = list(build(config=".tmp/offline_uv.yaml").argv)
    assert argv[argv.index("-c") + 1] == ".tmp/offline_uv.yaml"


def test_command_sets_pythonpath_with_the_platform_separator():
    command = build(pythonpath=("a", "b"))
    assert command.env["PYTHONPATH"] == os.pathsep.join(("a", "b"))
    assert command.env["PYTHONUTF8"] == "1"


def test_command_uses_a_custom_agent():
    argv = list(
        build(agent="evo.adapters.traced_terminus2:TracedAgentHarness").argv
    )
    assert argv[argv.index("-a") + 1] == (
        "evo.adapters.traced_terminus2:TracedAgentHarness"
    )


def test_command_merges_extra_environment():
    command = build(extra_env={"EVO_TRACE_DIR": "traces"})
    assert command.env["EVO_TRACE_DIR"] == "traces"


def test_command_sets_environment_build_timeout_multiplier():
    argv = list(build(environment_build_timeout_multiplier=10).argv)
    flag = argv.index("--environment-build-timeout-multiplier")
    assert argv[flag + 1] == "10.0"


def test_command_rejects_an_empty_task_set():
    with pytest.raises(HarborError, match="task"):
        build(task_ids=())


def test_command_rejects_duplicate_tasks():
    with pytest.raises(HarborError, match="duplicate"):
        build(task_ids=("terminal-bench/a", "terminal-bench/a"))


def test_command_rejects_non_positive_reps():
    with pytest.raises(HarborError, match="reps"):
        build(reps=0)


def test_parse_success_records_reward_tokens_and_status():
    record = parse_result(success_payload())
    assert record.task_id == "terminal-bench/fix-git"
    assert record.trial_id == "fix-git__abc123"
    assert record.reward == 1.0
    assert record.status == "completed"
    assert record.tokens_in == 120
    assert record.tokens_out == 30
    assert record.cost_usd == 0.002
    assert record.failure_cause is None


def test_parse_single_reward_key_without_the_name_reward():
    payload = success_payload(verifier_result={"rewards": {"pytest": 0.5}})
    assert parse_result(payload).reward == 0.5


def test_parse_multiple_reward_keys_prefers_reward():
    payload = success_payload(
        verifier_result={"rewards": {"pytest": 0.5, "reward": 0.75}}
    )
    assert parse_result(payload).reward == 0.75


def test_parse_missing_verifier_marks_the_trial_as_agent_error():
    payload = success_payload(verifier_result=None)
    record = parse_result(payload)
    assert record.reward is None
    assert record.status == "agent_error"
    assert record.failure_cause == "missing_verifier"


def test_parse_sandbox_exception_marks_the_trial_as_infra_error():
    payload = success_payload(
        verifier_result=None,
        exception_info={
            "exception_type": "EnvironmentStartTimeoutError",
            "exception_message": "docker compose failed: environment did not start",
            "occurred_at": "2026-11-04T00:00:00Z",
        },
    )
    record = parse_result(payload)
    assert record.status == "infra_error"
    assert record.failure_cause == "EnvironmentStartTimeoutError"


def test_parse_verifier_timeout_is_agent_error():
    payload = success_payload(
        verifier_result=None,
        exception_info={
            "exception_type": "VerifierTimeoutError",
            "exception_message": "verifier exceeded its budget",
            "occurred_at": "2026-11-04T00:00:00Z",
        },
    )
    assert parse_result(payload).status == "agent_error"


def test_parse_generic_exception_is_agent_error():
    payload = success_payload(
        verifier_result=None,
        exception_info={
            "exception_type": "ValueError",
            "exception_message": "environment did not start",
            "occurred_at": "2026-11-04T00:00:00Z",
        },
    )
    assert parse_result(payload).status == "agent_error"


def test_parse_agent_budget_exception_marks_the_trial_as_agent_error():
    payload = success_payload(
        verifier_result=None,
        exception_info={
            "exception_type": "AgentTimeoutError",
            "exception_message": "budget exhausted",
            "occurred_at": "2026-11-04T00:00:00Z",
        },
    )
    assert parse_result(payload).status == "agent_error"


def test_parse_command_timeout_marks_the_trial_as_infra_error():
    payload = success_payload(
        verifier_result=None,
        exception_info={
            "exception_type": "RuntimeError",
            "exception_message": "Command timed out after 120 seconds",
            "occurred_at": "2026-11-04T00:00:00Z",
        },
    )
    assert parse_result(payload).status == "infra_error"


def test_parse_sums_tokens_from_step_results():
    payload = success_payload(
        agent_result=None,
        step_results=[
            {"agent_result": {"n_input_tokens": 10, "n_output_tokens": 1, "cost_usd": 0.1}},
            {"agent_result": {"n_input_tokens": 20, "n_output_tokens": 2, "cost_usd": 0.2}},
        ],
    )
    record = parse_result(payload)
    assert record.tokens_in == 30
    assert record.tokens_out == 3
    assert record.cost_usd == pytest.approx(0.3)


def test_parse_without_token_reporting_leaves_tokens_none():
    payload = success_payload(agent_result={})
    record = parse_result(payload)
    assert record.tokens_in is None
    assert record.tokens_out is None


def test_parse_rejects_a_payload_without_a_trial_name():
    with pytest.raises(HarborError, match="trial_name"):
        parse_result(success_payload(trial_name=""))


def test_to_outcome_carries_the_measurement_fields():
    record = parse_result(success_payload())
    outcome = record.to_outcome()
    assert outcome.task_id == "terminal-bench/fix-git"
    assert outcome.trial_id == "fix-git__abc123"
    assert outcome.status == "completed"
    assert outcome.reward == 1.0
    assert outcome.tokens_in == 120
    assert outcome.tokens_out == 30


def test_read_job_collects_trial_results_in_order(tmp_path):
    job = tmp_path / "job1"
    for name, reward in (("b__2", 0.5), ("a__1", 1.0)):
        trial = job / name
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(success_payload(trial_name=name, verifier_result={"rewards": {"reward": reward}})),
            encoding="utf-8",
        )
    records = read_job(tmp_path, "job1")
    assert [record.trial_id for record in records] == ["a__1", "b__2"]
    assert [record.reward for record in records] == [1.0, 0.5]


def test_read_job_reports_a_malformed_result(tmp_path):
    trial = tmp_path / "job1" / "bad__1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(HarborError, match="bad__1"):
        read_job(tmp_path, "job1")


def test_read_job_rejects_a_missing_job(tmp_path):
    with pytest.raises(HarborError, match="job1"):
        read_job(tmp_path, "job1")


def test_run_harbor_writes_the_log_and_returns_the_code(tmp_path):
    log = tmp_path / "run.log"
    code = run_harbor(["python.exe", "-c", "print('hello')"], log_path=log)
    assert code == 0
    assert "hello" in log.read_text(encoding="utf-8")


def test_run_harbor_raises_on_a_non_zero_exit(tmp_path):
    with pytest.raises(HarborError, match="exit"):
        run_harbor(
            ["python.exe", "-c", "import sys; sys.exit(3)"],
            log_path=tmp_path / "run.log",
        )
