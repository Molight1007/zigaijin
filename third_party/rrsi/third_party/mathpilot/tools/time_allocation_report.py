# -*- coding: utf-8 -*-
"""时间分配诊断报告（official112）。

用途：112 全量跑完后，一条命令给出「时间花在哪、该往哪里重新分配」的数据依据。

    D:/python/python.exe tools/time_allocation_report.py results/official112_local_0910.jsonl

输出：同目录 <results 同名>_时间分配报告.md，并打印摘要到屏幕。

设计原则（对应用户要求「根据 112 题结果分析后重新制定分配方向」）：
  1) 不猜——每个结论后面都带数字；
  2) 区分「生成侧吃时间」与「验证侧被饿死」两类问题；
  3) 显式标出被 1200s 硬墙挡掉的题（Lean / 闸门在哪些题上根本没机会跑）。
"""
from __future__ import annotations

import io
import json
import os
import sys
from collections import Counter, defaultdict


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


WALL = 1200.0

# 生成侧 / 验证侧分组（用于「该往哪边分配时间」的判断）
# ★ 2026-10-02 移除 `1_classify` / `2.5_difficulty`（阶段已删）；此前报告仍含之。
GEN_STAGES = ("2.6_pre_audit", "2.7_subgoal_main",
              "3_solve", "3.2_complete", "3.3_improve", "3.4_collab",
              "3.5_subgoal_sup")
VERIFY_STAGES = ("3.6_audit_filter", "4_verify", "4.5_oracle", "4.6_adv",
                 "5_revise_or_fallback", "5.5_low_conf", "6_format",
                 "6.5_audit_gate")


def _render_lean_budget(w, rows, n) -> int:
    """D 段：Lean 触点是否被时间饿死（2026-10-08 从 main 逐字搬出）；返回 tc。"""
    # ---------- D Lean 触点预算 ----------
    w("## 三、Lean 触点是否被时间饿死")
    w()
    tc = 0
    cand_level = 0
    for r in rows:
        g = [e for e in ((r.get("diag") or {}).get("lean_gate") or []) if isinstance(e, dict)]
        if any("id" in e for e in g):
            cand_level += 1
        fg = [e for e in g if e.get("step") == "final_gate"]
        if fg and fg[-1].get("degraded") == "time_critical":
            tc += 1
    w(f"- 3.6 候选级 Lean 真跑过的题：**{cand_level}**")
    w(f"- 6.5 最终闸门因 **剩余<2s 被跳过** 的题：**{tc}**（{tc/n*100:.1f}%）")
    w()
    w("| 触点 | 中位剩余预算 | <2s(被跳过) | <20s | >120s |")
    w("|---|---|---|---|---|")
    for stage, label in [("3.6_audit_filter", "3.6 候选级"),
                         ("6.5_audit_gate", "6.5 最终闸门")]:
        rems = []
        for r in rows:
            st = (r.get("diag") or {}).get("stage_timers") or {}
            if not isinstance(st, dict) or stage not in st:
                continue
            ks = list(st.keys())
            try:
                rems.append(WALL - sum(float(st[k]) for k in ks[:ks.index(stage)]))
            except (TypeError, ValueError):
                pass
        if rems:
            w(f"| {label} | {_pct(rems,0.5):.0f}s | {sum(1 for x in rems if x<2)} | "
              f"{sum(1 for x in rems if x<20)} | {sum(1 for x in rems if x>120)} |")
    w()
    return tc


def main():
    # 2026-09-29：原先硬编码 `D:\挑战杯`（工作区已迁走）⇒ 改为由本文件位置
    # 反推仓库根，保证任何机器/盘符下都能跑。
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _repo_root, "results", "official112_local_0910.jsonl")
    if not os.path.isfile(p):
        print(f"结果文件不存在: {p}")
        sys.exit(1)
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    n = len(rows)
    out_md = os.path.splitext(p)[0] + "_时间分配报告.md"
    L: list[str] = []

    def w(s=""):
        L.append(s)

    # ---------- A 总览 ----------
    ok = [r for r in rows if r.get("correct")]
    el = [float(r.get("elapsed_sec") or 0) for r in rows]
    tiers = Counter((r.get("diag") or {}).get("tier", "(缺)") for r in rows)
    w("# official112 时间分配诊断报告")
    w()
    w(f"- 样本：**{n} 题**（正确 {len(ok)} 题，{len(ok)/n*100:.1f}%）")
    w(f"- 耗时：均值 **{_mean(el):.0f}s** / 中位 **{_pct(el,0.5):.0f}s** / "
      f"p90 **{_pct(el,0.9):.0f}s** / 最大 **{max(el):.0f}s**")
    w(f"- 硬墙 {WALL:.0f}s：**贴墙(>1150s) {sum(1 for x in el if x > 1150)} 题**"
      f"（{sum(1 for x in el if x > 1150)/n*100:.1f}%）")
    w(f"- 档位分布：{dict(tiers)}")
    w()

    # ---------- B 阶段耗时分解 ----------
    stage_sum = defaultdict(float)
    stage_each = defaultdict(list)
    for r in rows:
        st = (r.get("diag") or {}).get("stage_timers") or {}
        if not isinstance(st, dict):
            continue
        for k, v in st.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            stage_sum[k] += v
            stage_each[k].append(v)
    total_tracked = sum(stage_sum.values()) or 1.0
    w("## 一、阶段耗时分解（谁在吃时间）")
    w()
    w("| 阶段 | 总耗时(s) | 占比 | 中位(s) | p90(s) | 最大(s) |")
    w("|---|---|---|---|---|---|")
    for k in sorted(stage_sum, key=lambda x: -stage_sum[x]):
        w(f"| `{k}` | {stage_sum[k]:.0f} | {stage_sum[k]/total_tracked*100:.1f}% | "
          f"{_pct(stage_each[k],0.5):.1f} | {_pct(stage_each[k],0.9):.1f} | "
          f"{max(stage_each[k]):.1f} |")
    w()
    gen = sum(stage_sum.get(k, 0) for k in GEN_STAGES)
    ver = sum(stage_sum.get(k, 0) for k in VERIFY_STAGES)
    w(f"**生成侧 vs 验证侧**：{gen:.0f}s（{gen/total_tracked*100:.1f}%）"
      f" vs {ver:.0f}s（{ver/total_tracked*100:.1f}%）")
    w()

    # ---------- C 时间 → 正确率 ----------
    w("## 二、时间与正确率的关系（延长时间到底有没有用）")
    w()
    buckets = [(0, 450), (450, 700), (700, 1000), (1000, 1150), (1150, 10**9)]
    w("| 耗时区间 | 题数 | 正确 | 正确率 |")
    w("|---|---|---|---|")
    for lo, hi in buckets:
        g = [r for r in rows if lo <= float(r.get("elapsed_sec") or 0) < hi]
        if not g:
            continue
        gok = sum(1 for r in g if r.get("correct"))
        label = f"{lo}–{hi}s" if hi < 10**9 else f">{lo}s(贴墙)"
        w(f"| {label} | {len(g)} | {gok} | {gok/len(g)*100:.1f}% |")
    w()
    fast = [r for r in rows if float(r.get("elapsed_sec") or 0) < 700]
    slow = [r for r in rows if float(r.get("elapsed_sec") or 0) >= 1000]
    if fast and slow:
        fa = sum(1 for r in fast if r.get("correct")) / len(fast) * 100
        sa = sum(1 for r in slow if r.get("correct")) / len(slow) * 100
        w(f"→ 快题(<700s)正确率 {fa:.1f}% vs 慢题(≥1000s)正确率 {sa:.1f}%；"
          f"**{'慢题更差，说明时间被烧在了死路上' if sa < fa else '慢题不差，时间投入有效'}**")
    w()

    tc = _render_lean_budget(w, rows, n)

    # ---------- E 候选数 / 返工 / 降级 ----------
    w("## 四、预算消耗的可疑来源")
    w()
    nc = Counter((r.get("diag") or {}).get("n_candidates") for r in rows)
    rv = Counter((r.get("diag") or {}).get("revise_round") for r in rows)
    bs = [int((r.get("diag") or {}).get("budget_skips") or 0) for r in rows]
    dg = [int((r.get("diag") or {}).get("degraded_flags") or 0) for r in rows]
    ph = sum(1 for r in rows if (r.get("diag") or {}).get("placeholder"))
    tr = sum(1 for r in rows if not (r.get("diag") or {}).get("answer_complete", True))
    w(f"- 候选数分布：{dict(sorted(nc.items(), key=lambda x:(x[0] is None, x[0])))}")
    w(f"- revise_round 分布：{dict(sorted(rv.items(), key=lambda x:(x[0] is None, x[0])))}")
    w(f"- budget_skips 合计 {sum(bs)}（>0 的题 {sum(1 for x in bs if x)}）")
    w(f"- degraded_flags 合计 {sum(dg)}（>0 的题 {sum(1 for x in dg if x)}）")
    w(f"- 出现占位符「子目标求解失败」的题：**{ph}**")
    w(f"- 答案被截断的题：**{tr}**")
    w()
    sg = [((r.get("diag") or {}).get("subgoal_stats") or {}) for r in rows]
    nsg = [s.get("n_subgoals") for s in sg if isinstance(s, dict) and s.get("n_subgoals")]
    if nsg:
        w(f"- 子目标数：均值 {_mean(nsg):.1f} / 中位 {_pct(nsg,0.5)} / 最大 {max(nsg)}")
        w()

    # ---------- F 结论与分配方向 ----------
    w("## 五、数据结论与「时间分配」优化方向")
    w()
    top = sorted(stage_sum, key=lambda x: -stage_sum[x])[:3]
    w(f"1. **时间大户**：{', '.join('`%s`' % t for t in top)} —— "
      f"合计占 {sum(stage_sum[t] for t in top)/total_tracked*100:.1f}%。"
      "优化优先级应从这里入手，而不是去调验证侧的小项。")
    w(f"2. **验证侧占比 {ver/total_tracked*100:.1f}%**"
      f"{'（偏低 → 生成侧挤占了验证预算，先减生成冗余步骤）' if ver/total_tracked < 0.15 else ''}")
    if tc:
        w(f"3. **{tc} 题（{tc/n*100:.0f}%）的 6.5 闸门被时间墙跳过** —— "
          "这是「时间分配错误」的直接证据：应在生成侧提前收口，为终审留出预算。")
    if ph:
        w(f"4. **{ph} 题出现「子目标求解失败」占位符** —— 这些题的时间基本是白烧的，"
          "应加提前熔断（失败即转兜底），把时间让给能解的题。")
    if tr:
        w(f"5. **{tr} 题答案被截断** —— 截断比耗时更丢分，token/时间上限需与截断率一起权衡。")
    w()
    w("> 本报告由 `tools/time_allocation_report.py` 生成；数据齐备后重跑即可。")

    md = "\n".join(L) + "\n"
    with io.open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[已写出] {out_md}")


if __name__ == "__main__":
    main()
