#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐项审查本轮优化效果（2026-09-11）。

用法：
  D:/python/python.exe tools/review_optimizations.py results/official112_p4p1_0911.jsonl
  ... [--baseline results/official112_local_0910_rejudged.jsonl] [--md 报告.md]

设计（对应用户要求"一次评测即可逐项审查"）：所有优化均在 diag 留埋点，
本脚本一次把每项的触发情况、影响题号、与基线的对错翻转全部列出。
"""
from __future__ import annotations

import argparse
import json
import os


def _load(p: str) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r.get("id")] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_jsonl")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    cur = _load(args.result_jsonl)
    base = _load(args.baseline) if args.baseline and os.path.exists(args.baseline) else {}

    L: list[str] = []
    A = L.append
    A(f"# 逐项优化审查 — {os.path.basename(args.result_jsonl)}")
    A("")

    # ---------- 总览 ----------
    scored = [r for r in cur.values() if r.get("correct") is not None]
    ok = sum(1 for r in scored if r.get("correct"))
    A("## 〇、总览")
    A("")
    A(f"- 题量：{len(cur)}（参与判分 {len(scored)}）")
    A(f"- 正确率：**{ok}/{len(scored)} = {ok / max(1, len(scored)) * 100:.1f}%**")
    if base:
        bs = [r for r in base.values() if r.get("correct") is not None]
        bok = sum(1 for r in bs if r.get("correct"))
        A(f"- 基线：{bok}/{len(bs)} = {bok / max(1, len(bs)) * 100:.1f}%（{os.path.basename(args.baseline)}）")
        A(f"- 差值：**{(ok - bok) / max(1, len(bs)) * 100:+.1f}pp**")
    A("")

    # ---------- 逐项审查 ----------
    A("## 一、P1 硬信号 → 强制重解")
    A("")
    p1_rows = [r for r in cur.values()
               if any("P1 硬信号" in str(x) for x in ((r.get("diag") or {}).get("revise") or []))]
    A(f"- 触发题数：**{len(p1_rows)}**")
    for r in p1_rows:
        d = r.get("diag") or {}
        A(f"  - `{r.get('id')}` revise_round={d.get('revise_round')} correct={r.get('correct')}")
    A("")

    A("## 二、P2 强制数值化")
    A("")
    p2_rows = [r for r in cur.values() if ((r.get("diag") or {}).get("numericize_events"))]
    A(f"- 触发题数：**{len(p2_rows)}**")
    p2_flip = 0
    for r in p2_rows:
        d = r.get("diag") or {}
        b = base.get(r.get("id")) if base else None
        flip = ""
        if b is not None and b.get("correct") != r.get("correct"):
            flip = "  ← 对错翻转"
            p2_flip += 1
        A(f"  - `{r.get('id')}` correct={r.get('correct')}{flip}")
        for ev in (d.get("numericize_events") or [])[:2]:
            A(f"    - {ev}")
    A("")

    A("## 三、P5 客观题自检")
    A("")
    p5_rows = [r for r in cur.values() if ((r.get("diag") or {}).get("objective_check_events"))]
    A(f"- 触发题数：**{len(p5_rows)}**")
    for r in p5_rows:
        d = r.get("diag") or {}
        A(f"  - `{r.get('id')}` correct={r.get('correct')}")
        for ev in (d.get("objective_check_events") or [])[:2]:
            A(f"    - {ev}")
    A("")

    A("## 四、P4 阶段预算")
    A("")
    p4_rows = [r for r in cur.values() if ((r.get("diag") or {}).get("phase_budget_events"))]
    A(f"- 出现预算记录的题数：**{len(p4_rows)}**")
    two7, sol = [], []
    for r in cur.values():
        st = (r.get("diag") or {}).get("stage_timers") or {}
        if st.get("2.7_subgoal_main"):
            two7.append(st["2.7_subgoal_main"])
        if st.get("3_solve"):
            sol.append(st["3_solve"])
    if two7:
        A(f"- `2.7` 实耗：均 {sum(two7) / len(two7):.0f}s / 最大 {max(two7):.0f}s")
    if sol:
        A(f"- `3_solve` 实耗：均 {sum(sol) / len(sol):.0f}s / 最大 {max(sol):.0f}s")
    A("")

    A("## 五、P0 判分口径（已单独验证 +1.8pp）")
    A("")
    A("- 见 `official112_local_0910_rejudged.jsonl` 与离线重判记录（升 2 降 0）")
    A("")

    # ---------- 对错翻转总表 ----------
    if base:
        A("## 六、与基线的逐题翻转")
        A("")
        up = [i for i in cur if i in base and base[i].get("correct") is False and cur[i].get("correct") is True]
        down = [i for i in cur if i in base and base[i].get("correct") is True and cur[i].get("correct") is False]
        A(f"- 由错变对（**升**）：{len(up)} 题 → {', '.join(sorted(up)) or '无'}")
        A(f"- 由对变错（**降**）：{len(down)} 题 → {', '.join(sorted(down)) or '无'}")
        A("")

    md = args.md or os.path.join(os.path.dirname(args.result_jsonl) or ".",
                                 os.path.splitext(os.path.basename(args.result_jsonl))[0] + "_优化审查.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"题量 {len(cur)}｜正确率 {ok}/{len(scored)} = {ok / max(1, len(scored)) * 100:.1f}%")
    print(f"P1 触发 {len(p1_rows)}｜P2 触发 {len(p2_rows)}｜P5 触发 {len(p5_rows)}｜P4 记录 {len(p4_rows)}")
    if base:
        print(f"翻转：升 {len(up)} / 降 {len(down)}")
    print(f"报告：{os.path.abspath(md)}")


if __name__ == "__main__":
    main()
