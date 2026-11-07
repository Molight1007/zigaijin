"""Math instance: the MathPilot official-112 bank through its own evaluator.

Evaluate = `python run_eval.py --test_file <ids>.jsonl --output <job>__t<i>.jsonl`,
run with cwd = the candidate's harness root (MATHPILOT_ROOT, or
third_party/mathpilot inside this repo). One run_<i>.jsonl per trial; a task's
reward is 1.0 iff `correct` is true after symbolic answer matching.

Cost: this project's result jsonl carries elapsed_sec but NO token counts, so
the cost signal is wall-clock seconds per trial (budget_kind in rrsi.json).
`TaskResult.tokens` therefore holds seconds, and beta1 reads as "extra seconds
budgeted per additional solved task". Do not compare these numbers with the
paper's token costs.

A trial is MISSING (not zero) only when the evaluator produced no record for
that task: crashed batch, killed process, task absent from the output file.
A record with correct=false is a genuine zero.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
import briefs                            # noqa: E402
import render                            # noqa: E402

CFG = json.loads((HERE / "rrsi.json").read_text())
PYBIN = os.environ.get("MATHPILOT_PYTHON", sys.executable)
BANK_REL = Path("题库") / "official112_本地测试题库" / "official112_full.jsonl"

# Kept out of the child environment on purpose: a stale local proxy makes every
# request fail with 502 / WSAECONNREFUSED before the API is ever reached.
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
               "http_proxy", "https_proxy", "all_proxy")


def _harness_root() -> Path:
    """Where the editable MathPilot tree lives (can sit outside this repo)."""
    env = os.environ.get("MATHPILOT_ROOT")
    if env:
        return Path(env).resolve()
    return (HERE.parent.parent / "third_party" / "mathpilot").resolve()


def _bank_path() -> Path:
    env = os.environ.get("MATHPILOT_BANK")
    return Path(env).resolve() if env else (_harness_root() / BANK_REL)


def _load_bank() -> dict[str, dict]:
    p = _bank_path()
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[str(rec.get("id"))] = rec
    return out


def _reward(rec: dict) -> float:
    """Symbolic match is deterministic, so correct is a clean 0/1."""
    return 1.0 if rec.get("correct") else 0.0


def _cost(rec: dict) -> int | None:
    sec = rec.get("elapsed_sec")
    if not isinstance(sec, (int, float)):
        return None
    return int(round(sec))


def _error_class(rec: dict) -> str:
    return str(rec.get("error_class") or "")


def _is_empty_answer(rec: dict) -> bool:
    """The formatter prints this when the candidate pool was emptied upstream."""
    pred = str(rec.get("predicted") or "").strip()
    return pred in ("", "未给出有效解答。", "未给出有效解答")


class MathDomain(Domain):
    name = "math"
    harness_path = "third_party/mathpilot"
    source_exts = {".py", ".txt", ".md", ".json", ".toml", ".yaml"}
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}
    critic_patterns = [
        (r"official112[-_]\d+", "task id in diff"),
        (r"number_theory[-_]\d+", "task slug in diff"),
        (r"\bgold\b\s*[:=]", "gold answer wired into code or prompt"),
        (r"题库/", "task bank path in diff"),
        (r"answer[s]?_match\s*\(", "editing the scorer"),
        (r"def\s+_reward\b", "editing the reward mapping"),
        (r"official112_full\.jsonl", "bank file reference in diff"),
    ]
    component_signals = [
        ("memory",          [r"LESSONS\b", r"经验库", r"retrieve_lessons"]),
        ("skill",           [r"skills/", r"SKILL\.md"]),
        ("client_tool",     [r"tools/agent_toolkit", r"def\s+\w+_tool\(", r"register_tool"]),
        ("context_mgmt",    [r"\[\s*:\s*\d+\s*\]", r"truncat", r"max_tokens",
                             r"compact", r"summariz"]),
        ("control_flow",    [r"orchestrator", r"_run_main_stages", r"stage",
                             r"rewrite_gate", r"math_check_gate"]),
        ("config",          [r"SWITCHES", r"AgentConfig", r"=\s*\d+(\.\d+)?\s*(#|$)",
                             r"get_(bool|num|str)\("]),
        ("prompt",          [r"prompts/", r"_SYSTEM\b", r"_USER\b", r"TEMPLATE",
                             r"system_prompt", r"instruction"]),
    ]

    def __init__(self):
        bank = _load_bank()
        self._ids = sorted(bank)
        split = HERE / "data"
        self._evolve = self._read_ids(split / "evolve_ids.txt") or self._ids
        self._heldout = self._read_ids(split / "heldout_ids.txt")
        self._smoke = self._read_ids(split / "smoke_ids.txt") or CFG.get("smoke_tasks", [])

    @staticmethod
    def _read_ids(p: Path) -> list[str]:
        if not p.exists():
            return []
        return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return list(self._evolve)

    def heldout_ids(self) -> list[str]:
        return list(self._heldout)

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        return list(self._smoke)

    # ---- harness location --------------------------------------------------
    def harness_dir(self, root: Path) -> Path:
        """Overridden: the MathPilot tree may live outside this repository."""
        return _harness_root()

    # ---- Evaluate ----------------------------------------------------------
    def _env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in _PROXY_VARS}
        return env

    def _write_subset(self, target: Path, ids: list[str]) -> int:
        bank = _load_bank()
        rows = [bank[i] for i in ids if i in bank]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    def _job_file(self, runs_dir: Path, job: str, trial: int) -> Path:
        """Trial file for this job; falls back to the project's legacy naming
        (`<job>.jsonl`, no trial suffix) so historical runs can be re-scored."""
        p = Path(runs_dir) / f"{job}__t{trial}.jsonl"
        if p.exists():
            return p
        legacy = Path(runs_dir) / f"{job}.jsonl"
        if trial == 0 and legacy.exists():
            return legacy
        return p

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        """Run k independent passes of the evaluator over `ids`."""
        root, runs_dir = Path(root), Path(runs_dir)
        harness = self.harness_dir(root)
        if not (harness / "run_eval.py").exists():
            raise RuntimeError(f"run_eval.py not found under {harness}; "
                               "set MATHPILOT_ROOT to the MathPilot checkout")
        runs_dir.mkdir(parents=True, exist_ok=True)
        subset = runs_dir / f"{job}__ids.jsonl"
        n = self._write_subset(subset, ids)
        if n == 0:
            raise RuntimeError(f"no task from ids found in bank {_bank_path()}")
        for trial in range(k):
            out = self._job_file(runs_dir, job, trial)
            if out.exists() and out.stat().st_size > 0:
                continue                     # resume-safe: keep finished trials
            cmd = [PYBIN, "-u", "run_eval.py",
                   "--test_file", str(subset),
                   "--output", str(out),
                   "--concurrency", str(CFG.get("concurrency", 1)),
                   "--max_total_calls", str(CFG.get("max_total_calls", 40))]
            base = os.environ.get("MATHPILOT_BASE_URL") or os.environ.get("DEEPSEEK_API_BASE")
            model = os.environ.get("MATHPILOT_MODEL") or os.environ.get("DEEPSEEK_MODEL")
            if base:
                cmd += ["--base_url", base]
            if model:
                cmd += ["--model", model]
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(harness), env=self._env(),
                                  capture_output=True, text=True)
            (runs_dir / f"{job}__t{trial}.log").write_text(
                f"# cmd: {' '.join(cmd)}\n# rc={proc.returncode} "
                f"elapsed={time.time() - t0:.0f}s\n"
                f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
                f"--- stderr tail ---\n{proc.stderr[-4000:]}\n",
                encoding="utf-8")

    def score(self, runs_dir, job, ids, k):
        per = {}
        total = 0.0
        for t in ids:
            rewards, costs, missing = [], [], 0
            for trial in range(k):
                f = self._job_file(Path(runs_dir), job, trial)
                rec = _find_record(f, t)
                if rec is None:
                    missing += 1
                    rewards.append(0.0)
                    costs.append(None)
                    continue
                rw = _reward(rec)
                rewards.append(rw)
                costs.append(_cost(rec))
                total += rw
            per[t] = TaskResult(rewards=rewards, tokens=costs, missing=missing,
                                extra={"empty_answer": bool(
                                    _find_record(self._job_file(Path(runs_dir), job, 0), t)
                                    and _is_empty_answer(_find_record(
                                        self._job_file(Path(runs_dir), job, 0), t)))})
        n_trials = len(ids) * k
        return per, {"total_passes": total, "n_trials": n_trials,
                     "pass_rate": total / max(1, n_trials),
                     "budget_kind": CFG.get("budget_kind", "elapsed_sec")}

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        rec = _find_record(self._job_file(Path(runs_dir), job, trial), task_id)
        if rec is None:
            return None
        return {"task_id": task_id, "reward": _reward(rec),
                "elapsed_sec": _cost(rec),
                "error_class": _error_class(rec),
                "empty_answer": _is_empty_answer(rec),
                "predicted": str(rec.get("predicted") or "")[:200],
                "gold": str(rec.get("gold") or "")[:200],
                "raw": rec}

    def render_trace(self, rec, detail=False):
        return render.render_full(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        return (f"{task_id} | rewards={tr.rewards if tr else '?'} | "
                f"elapsed={rec.get('elapsed_sec')}s | "
                f"error_class={rec.get('error_class') or '-'} | "
                f"empty={rec.get('empty_answer')}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root), Path(runs_dir)
        harness = self.harness_dir(root)
        comp = subprocess.run([PYBIN, "-m", "compileall", "-q", str(harness / "agent")],
                              capture_output=True, text=True, env=self._env())
        if comp.returncode != 0:
            return False, {"stage": "compile",
                           "detail": (comp.stdout + comp.stderr)[-1500:]}
        try:
            self.run(root, runs_dir, f"{job}-smoke", list(ids),
                     k=1, log_prefix="smoke")
        except Exception as exc:                       # noqa: BLE001
            return False, {"stage": "run", "detail": str(exc)[:800]}
        per, extra = self.score(runs_dir, f"{job}-smoke", list(ids), 1)
        got = sum(1 for t in ids if per.get(t) and per[t].missing == 0)
        return got > 0, {"stage": "eval", "answered": got, "asked": len(ids),
                         "pass_rate": extra.get("pass_rate")}


def _find_record(path: Path, task_id: str) -> dict | None:
    """Stream a result jsonl for one task id (files can be large)."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("id")) == str(task_id):
                return rec
    return None


DOMAIN = MathDomain()
