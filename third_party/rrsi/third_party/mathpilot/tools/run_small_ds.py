# -*- coding: utf-8 -*-
"""小规模测试启动器 —— 用 DeepSeek 顶替书生（INTERN_*）跑求解主链。

为什么需要它：
  `run_eval.py` **不读 .env**（只有 run_research.py 自带 loader），
  主链只认 `--api_key/--base_url/--model` 或进程环境变量 `OPENAI_*`。
  本脚本把 `.env.ds` 里的 DeepSeek 配置注入环境变量后再调 run_eval，
  从而**不必修改 .env**，不影响既有的错题分析流水线。

用法：
    D:/python/python.exe tools/run_small_ds.py --test_file <题单> --output <结果> [run_eval 的其它参数]

    # 只看会传什么、不真跑：
    D:/python/python.exe tools/run_small_ds.py --dry-run

    # 用别的模型（默认取 .env.ds 的 LLM_MODEL）：
    D:/python/python.exe tools/run_small_ds.py --ds_model deepseek-chat --test_file ...

★ 实测提醒（2026-10-01）：reasoning 类模型（deepseek-v4-flash / deepseek-reasoner）
  在 max_tokens 过小时会**把全部预算花在 reasoning 上、正文返回空串**。
  本脚本默认把 max_answer_tokens 抬到 8192 以上；若你显式传入更小的值，会给出告警。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DS = os.path.join(REPO, ".env.ds")
RUN_EVAL = os.path.join(REPO, "run_eval.py")

REASONING_MODELS = ("deepseek-v4-flash", "deepseek-reasoner", "deepseek-r1")


def load_env_file_run_small_ds(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--ds_model", default="", help="覆盖 .env.ds 的 LLM_MODEL")
    ap.add_argument("--dry-run", action="store_true", help="只打印将注入的环境与命令，不执行")
    args, rest = ap.parse_known_args()

    cfg = load_env_file_run_small_ds(ENV_DS)
    if not cfg:
        print(f"!! 未找到 {ENV_DS} —— 请先创建并填入 OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL")
        return 2

    key = cfg.get("OPENAI_API_KEY", "")
    base = cfg.get("OPENAI_BASE_URL", "")
    model = args.ds_model or cfg.get("LLM_MODEL", "")

    if not key or not base or not model:
        print("!! .env.ds 缺 OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL 之一")
        return 2

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = key
    env["OPENAI_BASE_URL"] = base
    env["LLM_MODEL"] = model

    # reasoning 模型需给足输出预算，否则正文会被截断成空串
    if any(m in model for m in REASONING_MODELS):
        if not any(a.startswith("--max_answer_tokens") for a in rest):
            rest += ["--max_answer_tokens", "8192"]
            note = "（已自动加 --max_answer_tokens 8192）"
        else:
            note = ""
        print(f"★ 模型 {model} 属 reasoning 类，务必保证 max_answer_tokens 充足{note}")
    else:
        print(f"模型 {model} 无 reasoning 开销")

    cmd = [sys.executable, RUN_EVAL] + rest
    print(f"[小测试] LLM_MODEL={model}")
    print(f"[小测试] OPENAI_BASE_URL={base}")
    print(f"[小测试] OPENAI_API_KEY=…{key[-6:]}（已注入，不落盘）")
    print(f"[小测试] 命令: {' '.join(cmd)}")
    if args.dry_run:
        print("[dry-run] 未执行")
        return 0

    print("-" * 70)
    return subprocess.call(cmd, cwd=REPO, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
