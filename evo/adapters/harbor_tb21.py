from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from evo.adapters.deepseek import ModelProfile
from evo.application.baseline import TrialOutcome


AGENT = "terminus2:AgentHarness"
DATASET = "terminal-bench/terminal-bench-2-1"

# RRSI's coding adapter treats agent/verifier timeouts as valid failures and
# only Docker/registry faults as infrastructure. Kept verbatim so our pass rate
# uses the same denominator as the paper we compare against.
_INFRA_MESSAGE = re.compile(
    r"docker|compose|image|pull|registry|no space left|connection refused|"
    r"daemon|network .* (not found|already exists)",
    re.IGNORECASE,
)


def infra_failure(exception_type: str, message: str) -> bool:
    """Return True when a trial never got a working environment.

    AgentTimeoutError and VerifierTimeoutError count as failures, matching
    third_party/rrsi/domains/coding/adapter.py:83-95.
    """
    if exception_type in ("AgentTimeoutError", "VerifierTimeoutError"):
        return False
    return bool(_INFRA_MESSAGE.search(message)) or exception_type in (
        "RuntimeError",
        "OSError",
    )


class HarborError(RuntimeError):
    """A Harbor invocation or result file could not be used."""


@dataclass(frozen=True)
class HarborRun:
    argv: tuple[str, ...]
    env: Mapping[str, str]


def build_run(
    *,
    python: str,
    dataset: str = DATASET,
    task_ids: Sequence[str],
    model: ModelProfile,
    jobs_dir: str | Path,
    job_name: str,
    reps: int,
    concurrency: int = 1,
    config: str | Path | None = None,
    kwargs: Sequence[str] = ("record_terminal_session=false",),
    pythonpath: Sequence[str] = (),
    agent: str = AGENT,
    extra_env: Mapping[str, str] | None = None,
    environment_build_timeout_multiplier: float | None = None,
) -> HarborRun:
    tasks = tuple(task_ids)
    if not tasks:
        raise HarborError("task_ids must contain at least one task")
    if any(not isinstance(task, str) or not task.strip() for task in tasks):
        raise HarborError(f"task ids must be non-empty strings: {tasks!r}")
    if len(set(tasks)) != len(tasks):
        raise HarborError(f"task ids contain a duplicate: {tasks!r}")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
        raise HarborError(f"reps must be a positive integer: {reps!r}")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise HarborError(f"concurrency must be a positive integer: {concurrency!r}")
    if not isinstance(model, ModelProfile):
        raise HarborError(f"model must be a ModelProfile: {model!r}")
    if not str(job_name).strip():
        raise HarborError(f"job_name must be a non-empty string: {job_name!r}")
    if not isinstance(agent, str) or not agent.strip():
        raise HarborError(f"agent must be a non-empty string: {agent!r}")
    if extra_env is not None and not isinstance(extra_env, Mapping):
        raise HarborError("extra_env must be a mapping")
    if environment_build_timeout_multiplier is not None:
        value = environment_build_timeout_multiplier
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise HarborError(
                "environment_build_timeout_multiplier must be positive"
            )

    argv = [python, "-m", "harbor.cli.main", "run"]
    if config is not None:
        argv += ["-c", str(config)]
    argv += [
        "-d",
        dataset,
        "-a",
        agent,
        "-m",
        model.name,
    ]
    if model.base_url:
        argv += ["--base-url", model.base_url]
    argv += ["--api-key-name", model.api_key_env]
    for value in kwargs:
        argv += ["--ak", value]
    argv += [
        "--agent-setup-timeout-multiplier",
        "2",
    ]
    if environment_build_timeout_multiplier is not None:
        argv += [
            "--environment-build-timeout-multiplier",
            str(float(environment_build_timeout_multiplier)),
        ]
    argv += [
        "-k",
        str(reps),
        "-n",
        str(concurrency),
    ]
    for task in tasks:
        argv += ["-i", task]
    argv += ["--jobs-dir", str(jobs_dir), "--job-name", str(job_name)]

    env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return HarborRun(tuple(argv), env)


def _primary_reward(verifier: object) -> float | None:
    if not isinstance(verifier, Mapping):
        return None
    rewards = verifier.get("rewards")
    if not isinstance(rewards, Mapping) or not rewards:
        return None
    if "reward" in rewards:
        value = rewards["reward"]
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if len(rewards) == 1:
        value = next(iter(rewards.values()))
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return None


def _sum_field(contexts: Sequence[Mapping[str, object]], name: str) -> float | int | None:
    values = [context[name] for context in contexts if context.get(name) is not None]
    if not values:
        return None
    return sum(values)


def _token_totals(
    payload: Mapping[str, object],
) -> tuple[int | None, int | None, float | None]:
    agent_result = payload.get("agent_result")
    if isinstance(agent_result, Mapping):
        contexts = [agent_result]
    else:
        contexts = [
            step["agent_result"]
            for step in payload.get("step_results") or []
            if isinstance(step, Mapping) and isinstance(step.get("agent_result"), Mapping)
        ]
    if not contexts:
        return None, None, None
    tokens_in = _sum_field(contexts, "n_input_tokens")
    tokens_out = _sum_field(contexts, "n_output_tokens")
    cost = _sum_field(contexts, "cost_usd")
    return (
        int(tokens_in) if tokens_in is not None else None,
        int(tokens_out) if tokens_out is not None else None,
        float(cost) if cost is not None else None,
    )


@dataclass(frozen=True)
class TrialRecord:
    task_id: str
    trial_id: str
    reward: float | None
    status: str
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    failure_cause: str | None
    trial_uri: str = ""

    def to_outcome(self) -> TrialOutcome:
        return TrialOutcome(
            task_id=self.task_id,
            trial_id=self.trial_id,
            status=self.status,
            reward=self.reward,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost_usd,
            failure_cause=self.failure_cause,
        )


def parse_result(payload: object) -> TrialRecord:
    if not isinstance(payload, Mapping):
        raise HarborError(f"result must be a JSON object: {type(payload).__name__}")
    task_id = payload.get("task_name")
    trial_id = payload.get("trial_name")
    if not isinstance(task_id, str) or not task_id.strip():
        raise HarborError(f"result has no task_name: {task_id!r}")
    if not isinstance(trial_id, str) or not trial_id.strip():
        raise HarborError(f"result has no trial_name: {trial_id!r}")

    reward = _primary_reward(payload.get("verifier_result"))
    tokens_in, tokens_out, cost = _token_totals(payload)
    exception = payload.get("exception_info")
    if exception is None:
        status = "completed" if reward is not None else "agent_error"
        failure_cause = None if reward is not None else "missing_verifier"
    else:
        if not isinstance(exception, Mapping):
            raise HarborError(f"exception_info must be an object: {exception!r}")
        exception_type = str(exception.get("exception_type") or "unknown")
        message = str(exception.get("exception_message") or "")
        infra = infra_failure(exception_type, message)
        status = "infra_error" if infra else "agent_error"
        failure_cause = exception_type

    return TrialRecord(
        task_id=task_id,
        trial_id=trial_id,
        reward=reward,
        status=status,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        failure_cause=failure_cause,
        trial_uri=str(payload.get("trial_uri") or ""),
    )


def read_job(jobs_dir: str | Path, job_name: str) -> list[TrialRecord]:
    job = Path(jobs_dir) / job_name
    if not job.is_dir():
        raise HarborError(f"job directory not found: {job}")
    records = []
    for result_path in sorted(job.glob("*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarborError(
                f"{result_path.parent.name}: cannot read result.json: {exc}"
            ) from exc
        try:
            records.append(parse_result(payload))
        except HarborError as exc:
            raise HarborError(f"{result_path.parent.name}: {exc}") from exc
    return records


def run_harbor(
    argv: Sequence[str],
    *,
    log_path: str | Path,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = None,
) -> int:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = {**os.environ, **(env or {})}
    try:
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                list(argv),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=merged,
                cwd=str(cwd) if cwd is not None else None,
                timeout=timeout,
                check=False,
            )
    except FileNotFoundError as exc:
        raise HarborError(f"cannot start harbor: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HarborError(f"harbor timed out; see {log}") from exc
    if completed.returncode != 0:
        raise HarborError(
            f"harbor exited with code {completed.returncode}; see {log}"
        )
    return completed.returncode
