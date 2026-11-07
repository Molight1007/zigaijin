# -*- coding: utf-8 -*-
"""单题运行并导出 trace 时间线（定位 D5 冒烟「时间被吃掉」的问题）。"""
from __future__ import annotations
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def load_env_debug_one_trace() -> None:
    for line in open(os.path.join(_ROOT, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
    for d, s in [("OPENAI_API_KEY", "INTERN_API_KEY"),
                 ("OPENAI_BASE_URL", "INTERN_API_BASE"),
                 ("LLM_MODEL", "INTERN_MODEL")]:
        if os.environ.get(s):
            os.environ[d] = os.environ[s]


def main() -> int:
    load_env_debug_one_trace()
    import run_eval
    from utils.llm_client import LLMClient
    from user_agent import ReasoningAgent

    sample = os.path.join(_ROOT, "sample_data", "IMO-AnswerBench_smoke2.jsonl")
    with open(sample, encoding="utf-8") as fh:
        r = json.loads(fh.readline())

    client = LLMClient()
    ag = ReasoningAgent(client, **dict(run_eval.DEFAULT_AGENT_OVERRIDES))
    t0 = time.time()
    res = ag.solve(r["question"], {})
    print("TOTAL %.0fs" % (time.time() - t0), flush=True)
    tr = res.get("trace", [])
    print("--- trace (%d 条) ---" % len(tr), flush=True)
    base = tr[0].get("t", 0) if tr else 0
    for e in tr:
        print("  %6.0fs  %-24s %s" % (e.get("t", 0) - base,
                                      e.get("event", "?"),
                                      str(e.get("detail", ""))[:70]),
              flush=True)
    print("final:", str(res.get("final_response", ""))[:120], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
