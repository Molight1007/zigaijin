#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验组 vs 对照组逐题对比（2026-09-11）。

用法：
  D:/python/python.exe tools/compare_groups.py results/eval10_v9_0911.jsonl results/eval10_ctrl_0911.jsonl
  ... [--baseline results/official112_local_0910_rejudged.jsonl] [--md 报告.md]

背景：单次运行差异无法区分「优化导致」与「LLM 采样随机」，故跑实验组（优化全开）
与对照组（优化全关）同题对比；本脚本输出逐题对错、阶段耗时结构与优化触发情况。
"""
from __future__ import annotations

import argparse
import json
import os


def _load(p: str) -> dict:
    if not p or not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r.get("id")] = r
    return out


def _fmt(rows: dict, key: str) -> str:
    r = rows.get(key)
    if not r:
        return "—"
    return "✅" if r.get("correct") is True else ("❌" if r.get("correct") is False else "—")


def _stages(r: dict) -> str:
    st = ((r.get("diag") or {}).get("stage_timers") or {}) if r else {}
    keys = ["2.7_subgoal_main", "3_solve", "3.3_improve", "3.6_audit_filter", "4_verify"]
    return " ".join(f"{k.split('_')[0]}={round(st.get(k, 0) or 0)}" for k in keys if (st.get(k) or 0) > 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("ctrl")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    exp, ctrl = _load(args.exp), _load(args.ctrl)
    base = _load(args.baseline) if args.baseline else {}
    ids = sorted(set(exp) | set(ctrl))

    L: list[str] = []
    A = L.append
    A("# 实验组 vs 对照组 逐题对比（同题同模型）")
    A("")
    A(f"- 实验组（优化全开）：`{os.path.basename(args.exp)}`")
    A(f"- 对照组（优化全关）：`{os.path.basename(args.ctrl)}`")
    if base:
        A(f"- 历史基线：`{os.path.basename(args.baseline)}`")
    A("")
    A("| 题号 | 实验组 | 对照组 | 基线 | 实验组耗时 | 实验组阶段结构 | 对照组阶段结构 |")
    A("|---|---|---|---|---|---|---|")
    for i in ids:
        e, c, b = exp.get(i), ctrl.get(i), base.get(i)
        et = f"{round(float(e.get('elapsed_sec') or 0))}s" if e else "—"
        A(f"| {i} | {_fmt(exp, i)} | {_fmt(ctrl, i)} | {_fmt(base, i)} | {et} | "
          f"{_stages(e)} | {_stages(c)} |")
    A("")

    def _score(rows: dict) -> str:
        s = [r for r in rows.values() if r.get("correct") is not None]
        ok = sum(1 for r in s if r.get("correct"))
        return f"{ok}/{len(s)}"

    A("## 汇总")
    A("")
    A(f"- 实验组正确：**{_score(exp)}**")
    A(f"- 对照组正确：**{_score(ctrl)}**")
    if base:
        A(f"- 历史基线正确：**{_score(base)}**")
    A("")
    # 触发情况
    for label, rows in (("实验组", exp), ("对照组", ctrl)):
        trig = {"P1": 0, "P2": 0, "P5": 0, "P4": 0}
        for r in rows.values():
            d = r.get("diag") or {}
            if any("P1 硬信号" in str(x) for x in (d.get("revise") or [])):
                trig["P1"] += 1
            if d.get("numericize_events"):
                trig["P2"] += 1
            if d.get("objective_check_events"):
                trig["P5"] += 1
            if d.get("phase_budget_events"):
                trig["P4"] += 1
        A(f"- {label}触发：P1={trig['P1']} P2={trig['P2']} P5={trig['P5']} P4={trig['P4']}")
    A("")

    md = args.md or os.path.join(os.path.dirname(args.exp) or ".", "两组对比_0911.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"实验组 {_score(exp)}｜对照组 {_score(ctrl)}" + (f"｜基线 {_score(base)}" if base else ""))
    print(f"报告：{os.path.abspath(md)}")


if __name__ == "__main__":
    main()
