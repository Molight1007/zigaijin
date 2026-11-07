# -*- coding: utf-8 -*-
"""从历史评测题集重建本地基准题集（Phase 0 诊断用）。

背景：sample_data/ 下原只有 ab_test.jsonl + dev.jsonl（共 1.5KB），
IMO-AnswerBench.jsonl 缺失 → `run_eval.py --bank IMO-AnswerBench` 无法运行。
本项目从 测试结果/原始问题/ 下的 8 份历史 json 重建，按 id 去重。

字段映射：源文件的 reference_answer → run_eval.py 读取的 answer。
（run_eval.py:316 `gold = test.get("answer", "")`）

用法：
    python tools/build_sample_bank.py                    # 全部题库
    python tools/build_sample_bank.py --only IMO         # 仅 IMO-AnswerBench
    python tools/build_sample_bank.py --out sample_data/IMO-AnswerBench.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# 项目根目录（本文件在 <root>/tools/ 下）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_GLOB = os.path.join(_ROOT, "测试结果", "原始问题", "*.json")

DEFAULT_OUT = os.path.join(_ROOT, "sample_data", "IMO-AnswerBench.jsonl")


def collect(src_glob: str = SRC_GLOB, only: str = "") -> list[dict]:
    """按 id 去重收集题目，字段规范化为 run_eval.py 期望的格式。"""
    seen: set[str] = set()
    out: list[dict] = []
    for f in sorted(glob.glob(src_glob)):
        base = os.path.basename(f)
        if only and only.lower() not in base.lower():
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] 跳过 {base}: {e}", file=sys.stderr)
            continue
        for r in data:
            pid = r.get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            out.append({
                "id": pid,
                "question": r.get("question", ""),
                "domain": r.get("domain", ""),
                # run_eval.py 读的是 "answer"，源文件字段是 "reference_answer"
                "answer": r.get("reference_answer", ""),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="重建本地基准题集")
    ap.add_argument("--src", default=SRC_GLOB, help="源 json 通配符")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 jsonl 路径")
    ap.add_argument("--only", default="", help="只收文件名含该关键字的批次，如 IMO")
    args = ap.parse_args()

    items = collect(args.src, args.only)
    if not items:
        print("[error] 未收集到任何题目，检查 --src 路径", file=sys.stderr)
        return 1

    # 统计领域分布，便于诊断阶段做 per_domain 分析
    by_domain: dict[str, int] = {}
    empty_ans = 0
    for it in items:
        by_domain[it["domain"] or "unknown"] = by_domain.get(it["domain"] or "unknown", 0) + 1
        if not str(it["answer"]).strip():
            empty_ans += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"written {len(items)} -> {args.out}")
    for d, n in sorted(by_domain.items(), key=lambda kv: -kv[1]):
        print(f"  {d}: {n}")
    if empty_ans:
        print(f"[warn] {empty_ans} 条 answer 为空，评测时会计为判错", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
