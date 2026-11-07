#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""18 题评测步骤级耗时分析：stage_timers 分解 + 压缩点定位。
用法: D:/python/python.exe tools/analyze_stage_timers.py <results.jsonl>
"""
import json, sys, collections

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval_ab/ab18_full_0904.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"共 {len(rows)} 题\n")
    # 阶段聚合 key（把细粒度 stage 归到主阶段）
    def group(k):
        if k.startswith("2.6"): return "preverify"
        if k.startswith("2.7"): return "subgoal_main"
        if k.startswith("3_solve") or k == "3_solve": return "solve"
        if "lean_candidate" in k or "lean_filter" in k or "3.6" in k or "3.5.1" in k: return "lean_candidate_filter"
        if k.startswith("3.2"): return "complete"
        if k.startswith("4_"): return "verify"
        if k.startswith("1_") or k.startswith("2.5"): return "classify_diff"
        if k.startswith("5") or k.startswith("6"): return "revise_format_gate"
        return "other"
    agg_rows = []
    print("=" * 100)
    print(f"{'题':<28}{'正确':<7}{'总耗时':>7}{'截断':>5} | " + " | ".join(f"{g:>14}" for g in
          ["preverify","subgoal_main","solve","lean_cand_filter","complete","verify","classify","其他"]))
    print("=" * 100)
    colsum = collections.defaultdict(float); colcnt = collections.Counter()
    for r in sorted(rows, key=lambda x: x["id"]):
        st = (r.get("diag") or {}).get("stage_timers") or {}
        g = collections.defaultdict(float)
        for k, v in st.items():
            if isinstance(v, (int, float)): g[group(k)] += v
        trunc = "⏰" if r["elapsed_sec"] >= 1190 else ""
        corr = "✅" if r.get("correct") else "❌"
        order = ["preverify","subgoal_main","solve","lean_candidate_filter","complete","verify","classify_diff","other"]
        print(f"{r['id']:<28}{corr:<7}{r['elapsed_sec']:>7.0f}{trunc:>5} | " +
              " | ".join(f"{g.get(o,0):>14.0f}" for o in order))
        for o in order:
            colsum[o] += g.get(o, 0)
        colcnt[r["id"]] = 1
    n = len(rows) or 1
    print("=" * 100)
    print(f"{'均值(18题)':<28}{'':<7}{'':>7}{'':>5} | " +
          " | ".join(f"{colsum[o]/n:>14.0f}" for o in
          ["preverify","subgoal_main","solve","lean_candidate_filter","complete","verify","classify_diff","other"]))
    total = sum(colsum.values())
    print(f"\n阶段总耗时占比（跨 {n} 题）: 总 {total:.0f}s")
    for o in ["preverify","subgoal_main","solve","lean_candidate_filter","complete","verify","classify_diff","other"]:
        print(f"  {o:<22}{colsum[o]:>8.0f}s  {colsum[o]/total*100:>5.1f}%")
    # 正确率
    scored = [r for r in rows if r.get("correct") is not None]
    corr = sum(1 for r in scored if r["correct"])
    print(f"\n正确率: {corr}/{len(scored)} = {corr/max(1,len(scored))*100:.1f}%")
    truncs = [r for r in rows if r["elapsed_sec"] >= 1190]
    print(f"超 1200s 被截断: {len(truncs)}/{len(rows)} 题")

if __name__ == "__main__":
    main()
