#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""机制运行检查（2026-09-11）——回答「优化到底跑了没有」。

用法：
  D:/python/python.exe tools/check_mechanisms.py results/eval10_xxx.jsonl

设计：只关心**机制是否触发**（不评效果），逐项列出触发题号与证据。
适用：每次 10 题左右的小测（用户 9/11 指示：主要看优化有没有运行）。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def _load(p: str) -> list:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_jsonl")
    args = ap.parse_args()
    rows = _load(args.result_jsonl)

    def hit(pred) -> list:
        return [r.get("id") for r in rows if pred((r.get("diag") or {}))]

    def _n(s) -> str:
        """去掉 markdown 星号再匹配 —— 埋点文案里含 `重解**启动**` 这类强调符，
        直接子串匹配会漏检（2026-09-12 实测被误报为"0 触发"）。"""
        return str(s).replace("*", "")

    print(f"=== 机制运行检查：{os.path.basename(args.result_jsonl)}（{len(rows)} 题）===")
    print()

    checks = [
        ("P4  阶段预算（2.7/3_solve cap）", lambda d: bool(d.get("phase_budget_events"))),
        ("P1  硬信号→强制重解", lambda d: any("P1 硬信号" in str(x) for x in (d.get("revise_events") or []))),
        ("#3  ★重解**启动**（Bug 3 修复生效）",
         lambda d: any("重解启动" in _n(x) for x in (d.get("revise_events") or []))),
        ("#3' 重解**未启动**（候选满/槽位 0）",
         lambda d: any("重解未启动" in _n(x) for x in (d.get("revise_events") or []))),
        ("#2  候选满→腾位（A 修复路径）",
         lambda d: any("腾位" in _n(x) for x in (d.get("revise_events") or []))),
        ("P2  强制数值化", lambda d: bool(d.get("numericize_events"))),
        ("P5  客观题自检", lambda d: bool(d.get("objective_check_events"))),
        ("#9  Lean 逐候选时间护栏(time_budget)",
         lambda d: any(isinstance(g, dict) and g.get("degraded") == "time_budget"
                       for g in (d.get("lean_gate") or []))),
        ("#13 preverify 已执行（verdict 非空）",
         lambda d: bool((d.get("preverify_trace") or {}).get("verdict"))),
        ("P1' 未触发原因已记录（p1_check）",
         lambda d: bool(d.get("p1_check_events"))),
    ]
    for label, pred in checks:
        ids = hit(pred)
        print(f"[{'✓' if ids else '✗'}] {label:<44} 触发 {len(ids)} 题"
              + (f"  → {', '.join(ids[:6])}" + (" …" if len(ids) > 6 else "") if ids else ""))

    # 附加统计
    print()
    print("--- 附加观测 ---")
    pv = Counter((r.get("diag") or {}).get("preverify_trace", {}).get("verdict")
                 for r in rows if isinstance((r.get("diag") or {}).get("preverify_trace"), dict))
    print(f"preverify 判定分布: {dict(pv)}")
    lg_deg = Counter(g.get("degraded") for r in rows
                     for g in ((r.get("diag") or {}).get("lean_gate") or [])
                     if isinstance(g, dict) and g.get("degraded"))
    print(f"lean_gate degraded 分布: {dict(lg_deg)}")
    rr = Counter((r.get("diag") or {}).get("revise_round") for r in rows)
    print(f"revise_round 分布: {dict(rr)}")
    bvf = sum(1 for r in rows if (r.get("diag") or {}).get("blueprint_value_false"))
    print(f"数值攻击证伪信号（blueprint_value_false）: {bvf} 题")
    ok = sum(1 for r in rows if r.get("correct") is True)
    print(f"（参考）本轮正确: {ok}/{len(rows)}")

    # ---- P1 未触发原因（2026-09-12 新增）----
    # 目的：把"P1 0 触发"拆成「**没信号**（机制无燃料）」与「**有时间但被拦**」
    # 两类，此前两者都表现为"日志里什么都没有"。
    p1c = [str(x) for r in rows
           for x in ((r.get("diag") or {}).get("p1_check_events") or [])]
    if p1c:
        cc = Counter()
        for x in p1c:
            if "无硬信号" in x:
                cc["无硬信号（机制无燃料）"] += 1
            if "≤ 150s" in x or "150s 门槛" in x:
                cc["剩余时间不足 150s"] += 1
            if "emergency" in x:
                cc["emergency 状态"] += 1
            if "不在 deep/standard" in x:
                cc["档位不符"] += 1
        print()
        print(f"--- P1 未触发原因（{len(p1c)} 条记录）---")
        for k, v in cc.most_common():
            print(f"  {k}: {v}")
        print(f"  样例: {p1c[0][:160]}")

    # ---- mcp 后端判据（2026-09-11 新增；从同名 .log 解析 A9–A10）----
    lg = os.path.splitext(args.result_jsonl)[0] + ".log"
    if os.path.exists(lg):
        txt = open(lg, encoding="utf-8", errors="replace").read()
        print()
        print("--- mcp 后端判据（来自日志）---")
        for label, key in (
            ("A9  mcp 命中（走 mcp 后端）", "★ 走 mcp 后端"),
            ("A9  mcp 成功返回", "★ mcp 返回成功"),
            ("A10 axiom 检查通过", "★ mcp axiom 检查通过"),
            ("A10 axiom 发现问题(sorryAx)", "发现 sorryAx"),
            ("⚠ 回落 bridge", "回落 bridge"),
        ):
            n = txt.count(key)
            mark = "✗" if key == "回落 bridge" else "✓"
            print(f"[{mark if n else '·'}] {label:<32} {n} 次")
        # A11：B1 子目标数值核验（日志中 sgnum 临时文件）
        print(f"[{'✓' if 'sgnum_' in txt else '·'}] A11 B1 子目标数值核验(sgnum)      "
              f"{txt.count('sgnum_')} 次")



if __name__ == "__main__":
    main()
