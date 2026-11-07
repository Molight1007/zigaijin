#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cloud_health_report.py — MathPilot 云端评测「一站式体检」

把 10 类检测项一次出齐，避免每次巡检临时拼脚本。

用法
    python tools/cloud_health_report.py <run.jsonl> [--log <run.log>]
        [--monitor <monitor.csv>] [--top-k 10] [--out <x.md>]

检测项（对应「用户要求检测的数据」清单）
    A 进度与正确率                B 分阶段耗时（含过短/过长标记）
    C 机制触发普查                D Lean 通道健康（★ 本轮重点）
    E 截断台账                    F LeanSearch 需求侧
    G MCP 工具使用                H 参数实测 vs 上限
    I 错误分类                    J 资源曲线

设计约束
    · 只读；不写仓库内任何被跟踪文件（报告默认打印到 stdout）
    · md 输出**禁围栏代码块**、正文竖线换 `∣`（生成器约定）
    · 老结果缺新字段时**显式说明**，不把缺失当 0（防比率的假阴性）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- 阶段名映射
STAGE_LABEL = {
    # ★ 2026-10-02 移除 `0_paper_pacer` / `1_classify` / `2.5_difficulty`（阶段已删）；此前报告仍含之。
    "2.6_pre_audit": "前置审计",
    # ★ 2026-10-02 移除 `2.65_calc_prewarm`（calc 板块删除时漏网残留）；此前报告仍含之。
    "2.7_subgoal_main": "子目标主循环",
    "3_solve": "主体求解",
    "3.2_complete": "答案补全",
    "3.3_improve": "自改进",
    "3.4_collab": "协作轮",
    "3.5_subgoal_sup": "子目标补充",
    "3.6_audit_filter": "审计过滤",
    "4_verify": "验证器",
    "4.5_oracle": "Oracle 复核",
    "4.6_adv": "对抗验证",
    "5_revise_or_fallback": "修订/回退",
    "5.5_low_conf": "低置信处理",
    "6_format": "格式化",
    "6.5_audit_gate": "闸门审计",
}

# 机制触发相关字段（diag 里的 *_events / list 型）
MECH_FIELDS = [
    "lean_gate", "audit_gate", "control_events", "revise_events",
    "skeleton_review_events", "dag_replan_events", "subgoal_trace",
    "answer_form_events", "p1_check_events",
    "phase_budget_events", "toolcall_text_detected", "revise_feedback",
    "toolcall_mode", "toolcall_exec", "value_attack", "exhaust_diag", "deep_review",
    "formal_gaps", "lemma_repo",
    "numericize_events", "objective_check_events", "expression_eval_events",
    "final_postprocess_change_events", "lean_feedback_revise_events",
    "subgoal_recover_events", "subgoal_replan_needed_events",
    "symbolic_crosscheck_events", "symbolic_solve_events",
    "dag_review", "preverify_trace", "sketch_audit", "subgoal_ctx_inject_chars",
]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def stat_line(name, vals, unit="", cap=None):
    if not vals:
        return "  %-32s %s" % (name, "(无数据)")
    v = sorted(vals)
    s = "  %-32s n=%-4d 中位=%-9.4g 均=%-9.4g max=%-9.4g%s" % (
        name, len(v), st.median(v), sum(v) / len(v), v[-1],
        (" " + unit) if unit else "")
    if cap is not None:
        hit = sum(1 for x in v if x >= cap)
        s += "  上限=%-8g 撞顶=%d/%d" % (cap, hit, len(v))
    return s


def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- A
def sec_a(rows):
    sec("A) 进度与正确率")
    n = len(rows)
    ok = sum(1 for r in rows if r.get("correct"))
    print("  完成题数 %d   正确 %d   正确率 %.2f%%" % (
        n, ok, 100.0 * ok / n if n else 0))
    print("  tier 分布:", dict(Counter((r.get("diag") or {}).get("tier")
                                      for r in rows)))
    print("  域分布  :", dict(Counter(r.get("domain") for r in rows)))


# ---------------------------------------------------------------- B
def sec_b(rows):
    sec("B) 分阶段耗时（秒）—— 关注「异常短(没跑)」「异常长(卡住)」")
    acc = defaultdict(list)
    for r in rows:
        for k, v in ((r.get("diag") or {}).get("stage_timers") or {}).items():
            x = num(v)
            if x is not None:
                acc[k].append(x)
    tot = [num(r.get("elapsed_sec")) for r in rows]
    tot = [x for x in tot if x]
    tmed = st.median(tot) if tot else 1.0
    print("  总耗时: 中位 %.1fs  均 %.1fs  max %.1fs" % (
        tmed, sum(tot) / len(tot) if tot else 0, max(tot) if tot else 0))
    print()
    print("  %-18s %9s %9s %9s %8s  %s" % ("阶段", "中位", "均", "max",
                                           "占比", "标记"))
    for k in sorted(acc, key=lambda k: -st.median(acc[k])):
        v = acc[k]
        med = st.median(v)
        zero = sum(1 for x in v if x <= 0.01)
        mark = []
        if zero == len(v):
            mark.append("★全程未跑")
        elif zero > len(v) * 0.5:
            mark.append("多数未跑(%d/%d)" % (zero, len(v)))
        if med > 60:
            mark.append("长")
        print("  %-18s %9.1f %9.1f %9.1f %7.1f%%  %s" % (
            STAGE_LABEL.get(k, k)[:18], med, sum(v) / len(v), max(v),
            100 * med / tmed, " ".join(mark)))


# ---------------------------------------------------------------- C
def sec_c(rows):
    sec("C) 机制触发普查（0 触发 = 机制没起作用）")
    print("  %-34s %8s %8s %8s %8s" % ("字段", "有数据", "0触发", "中位", "max"))
    for f in MECH_FIELDS:
        vals = []
        for r in rows:
            d = (r.get("diag") or {}).get(f)
            if d is None:
                vals.append(None)
            elif isinstance(d, (list, dict)):
                vals.append(len(d))
            else:
                vals.append(num(d, 0))
        got = [v for v in vals if v is not None]
        if not got:
            print("  %-34s %8s  (该轮无此字段)" % (f, "-"))
            continue
        zero = sum(1 for v in got if not v)
        print("  %-34s %8d %8d %8.4g %8.4g%s" % (
            f, len(got), zero, st.median(got), max(got),
            "  ★全为0" if zero == len(got) else ""))


# ---------------------------------------------------------------- D
def sec_d(rows, log=None):
    sec("D) ★ Lean 通道健康（本轮修复重点）")
    cvs = [num((r.get("mathlib_usage_stats") or {}).get("compile_valid"))
           for r in rows]
    cvs = [c for c in cvs if c is not None]
    if cvs:
        print("  compile_valid：中位 %g  均 %.2f  max %g  非零题数 %d/%d" % (
            st.median(cvs), sum(cvs) / len(cvs), max(cvs),
            sum(1 for c in cvs if c), len(cvs)))
        if all(c == 0 for c in cvs):
            print("    ★ 恒为 0 ⇒ Lean 编译从未成功（典型：闭包残缺 / import 失败）")
    else:
        print("  compile_valid：该轮无此字段")

    # ★ Lean 结果在 diag.lean_gate 里（**不是**顶层 verdicts！顶层 verdicts 是
    #   答案投票聚合 {answer, confidence, correct_votes, total_votes}）。
    lg_verdict = Counter()
    lg_valid = Counter()
    lg_degraded = Counter()
    lg_vreason = Counter()
    for r in rows:
        for e in ((r.get("diag") or {}).get("lean_gate") or []):
            if not isinstance(e, dict):
                continue
            if "verdict" in e:
                lg_verdict[str(e["verdict"])] += 1
            if "lean_valid" in e:
                lg_valid[str(bool(e["lean_valid"]))] += 1
            if e.get("degraded"):
                lg_degraded[str(e["degraded"])] += 1
            if e.get("verdict_reason"):
                lg_vreason[str(e["verdict_reason"])] += 1
    if lg_verdict:
        tot = sum(lg_verdict.values())
        print("  lean_gate.verdict：" + "  ".join(
            "%s=%d(%.1f%%)" % (k, c, 100.0 * c / tot)
            for k, c in lg_verdict.most_common()))
        if lg_verdict.get("unknown", 0) / tot > 0.9:
            print("    ★ unknown 占比 >90% ⇒ Lean 结果基本未被采纳")
    if lg_valid:
        tot = sum(lg_valid.values())
        print("  lean_gate.lean_valid：" + "  ".join(
            "%s=%d" % (k, c) for k, c in lg_valid.most_common()))
    if lg_degraded:
        tot = sum(lg_degraded.values())
        print("  lean_gate.degraded：" + "  ".join(
            "%s=%d(%.0f%%)" % (k, c, 100.0 * c / tot)
            for k, c in lg_degraded.most_common()))
        if lg_degraded.get("verify_stop"):
            print("    ★ verify_stop ⇒「连续 2 个候选 unknown 即止损跳过整题 verify」"
                  "，是 Lean 失效的**连锁反应**")
    if lg_vreason:
        print("  lean_gate.verdict_reason：" + "  ".join(
            "%s=%d" % (k, c) for k, c in lg_vreason.most_common(5)))

    # 顶层 verdicts = 候选答案投票聚合
    conf, votes = [], Counter()
    for r in rows:
        for v in (r.get("verdicts") or []):
            if isinstance(v, dict):
                c = num(v.get("confidence"))
                if c is not None:
                    conf.append(c)
                tv = v.get("total_votes")
                if tv is not None:
                    votes[int(tv)] += 1
    if conf:
        print("  候选答案投票：候选数/题中位 %.0f，confidence 中位 %.2f，"
              "零置信候选 %d/%d（%.0f%%）" % (
                  len(conf) / len(rows), st.median(conf),
                  sum(1 for c in conf if c == 0), len(conf),
                  100.0 * sum(1 for c in conf if c == 0) / len(conf)))
    if votes:
        print("  total_votes 分布:", dict(sorted(votes.items())))

    mu = [num((r.get("mathlib_usage_stats") or {}).get("search_calls"))
          for r in rows]
    mu = [c for c in mu if c is not None]
    if mu:
        print("  mathlib search_calls：中位 %g  max %g" % (
            st.median(mu), max(mu)))

    tc = Counter()
    for r in rows:
        for k, v in ((r.get("tool_calls") or {})).items():
            if isinstance(v, dict):
                tc[k + ".calls"] += int(v.get("calls") or 0)
    if tc:
        print("  tool_calls：" + "  ".join("%s=%d" % kv for kv in
                                          sorted(tc.items())))

    if log and os.path.isfile(log):
        print()
        print("  --- 日志侧（MCP 通道） ---")
        txt = open(log, encoding="utf-8", errors="replace").read()
        for pat, label in (
            (r"mcp 返回成功：ok=True", "Lean 编译成功"),
            (r"mcp 返回成功：ok=False", "Lean 编译失败"),
            (r"走 mcp 后端", "走 mcp 后端"),
            (r"回落 bridge", "回落 bridge（应尽量为 0）"),
            (r"axiom 检查通过", "axiom 检查通过"),
            (r"axiom 检查发现 sorryAx", "★ axiom 检出 sorryAx"),
            (r"axiom 检查改用 example 改写副本", "★ example 改写副本（新修复）"),
            (r"multi_attempt 找到可用策略", "multi_attempt 找到策略"),
            (r"hover 查证", "hover 查证"),
            (r"goal 定位失败", "goal 定位失败"),
        ):
            n = len(re.findall(pat, txt))
            print("    %-38s %d" % (label, n))


# ---------------------------------------------------------------- E
def sec_e(rows):
    sec("E) 截断台账（max_answer_tokens 的直接证据）")
    tr = [(r.get("llm_calls") or {}).get("truncated") for r in rows]
    got = [t for t in tr if isinstance(t, (int, float))]
    if not got:
        print("  该轮无 truncated 字段（老格式）")
        return
    print("  截断题数 %d / %d（%.0f%%）  截断总次数 %d" % (
        sum(1 for x in got if x), len(got),
        100.0 * sum(1 for x in got if x) / len(got), sum(got)))
    print("  分布:", dict(sorted(Counter(int(x) for x in got).items())))
    calls = [num((r.get("llm_calls") or {}).get("calls")) for r in rows]
    calls = [c for c in calls if c is not None]
    if calls:
        print("  llm_calls 总数：中位 %g  max %g（对照 max_total_calls）" % (
            st.median(calls), max(calls)))


# ---------------------------------------------------------------- F
def sec_f(rows):
    sec("F) LeanSearch 需求侧（要找多少条定理）")
    ls = [(r.get("diag") or {}).get("leansearch") or {} for r in rows]
    calls = [num(x.get("calls")) for x in ls]
    calls = [c for c in calls if c is not None]
    hits = [num(x.get("hits")) for x in ls]
    hits = [c for c in hits if c is not None]
    uniq = [num(x.get("unique")) for x in ls]
    uniq = [c for c in uniq if c is not None]
    if calls:
        print("  calls/题 ：" + stat_line("", calls).strip())
    if hits:
        print("  hits/题  ：" + stat_line("", hits).strip())
        print("  hits 分布:", dict(sorted(Counter(int(x) for x in hits).items())))
    if uniq:
        print("  unique/题：" + stat_line("", uniq).strip())
    tkd = [(r.get("diag") or {}).get("leansearch") or {} for r in rows]
    if tkd and any("top_k_capped" in x for x in tkd):
        got = [x["top_k_capped"] for x in tkd if "top_k_capped" in x]
        print("  top_k_capped（真字段）：%d/%d = %.0f%%" % (
            sum(1 for g in got if g), len(got), 100.0 * sum(1 for g in got if g) / len(got)))
    else:
        print("  top_k_capped：该轮无此字段（老格式）⇒ 需新版重跑才能判撞顶")
    roots = Counter(x.get("root") for x in ls if x.get("root"))
    if roots:
        print("  检索源:", dict(roots))


# ---------------------------------------------------------------- G
def sec_g(rows):
    sec("G) 参数实测 vs 上限（diag.param_usage）")
    have = [r for r in rows if (r.get("diag") or {}).get("param_usage")]
    if not have:
        print("  该轮无 param_usage 字段 ⇒ 需本轮之后的代码跑出的结果才有。")
        print("  （可用 cli/param_usage_report.py 对新结果出详细表）")
        return
    agg = defaultdict(lambda: {"cap": [], "used": [], "capped": 0, "n": 0})
    for r in have:
        for it in ((r.get("diag") or {})["param_usage"].get("items") or []):
            a = agg[it.get("name")]
            if it.get("cap") is not None:
                a["cap"].append(it["cap"])
            if it.get("used") is not None:
                a["used"].append(it["used"])
            if it.get("capped"):
                a["capped"] += 1
            a["n"] += 1
    print("  %-34s %8s %9s %8s %8s" % ("参数", "上限", "用量中位", "用量max",
                                        "撞顶率"))
    for name in sorted(agg):
        a = agg[name]
        cap = st.median(a["cap"]) if a["cap"] else None
        used = a["used"]
        print("  %-34s %8s %9s %8s %7.0f%%" % (
            name, ("%g" % cap) if cap is not None else "-",
            ("%g" % st.median(used)) if used else "-",
            ("%g" % max(used)) if used else "-",
            100.0 * a["capped"] / a["n"] if a["n"] else 0))


# ---------------------------------------------------------------- H
def sec_h(rows):
    sec("H) 错误分类")
    ec = Counter(r.get("error_class") or "(空/答对)" for r in rows)
    tot = sum(ec.values())
    for k, c in ec.most_common():
        print("  %-24s %4d  %5.1f%%" % (k, c, 100.0 * c / tot))
    et = Counter()
    for r in rows:
        for k, v in ((r.get("diag") or {}).get("error_types") or {}).items():
            if isinstance(v, (int, float)) and v:
                et[k] += v
    if et:
        print("  diag.error_types 汇总:", dict(et.most_common(10)))


# ---------------------------------------------------------------- I
def sec_i(monitor):
    sec("I) 资源曲线（monitor CSV）")
    if not monitor or not os.path.isfile(monitor):
        print("  未提供 --monitor，跳过")
        return
    mem, swap, lean = [], [], []
    with open(monitor, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            for key, box in (("mem_used_pct", mem), ("swap_used_mb", swap),
                             ("lean_procs", lean)):
                v = num(row.get(key))
                if v is not None:
                    box.append(v)
    if mem:
        print("  内存占用 %%：中位 %.0f  max %.0f" % (st.median(mem), max(mem)))
    if swap:
        print("  swap 使用 MB：中位 %.0f  max %.0f%s" % (
            st.median(swap), max(swap),
            "  ★ 有 swap ⇒ 内存吃紧" if max(swap) > 512 else "  （全程几乎无 swap）"))
    if lean:
        print("  lean 进程数：中位 %.0f  max %.0f" % (st.median(lean), max(lean)))


def main() -> int:
    ap = argparse.ArgumentParser(description="MathPilot 云端评测一站式体检")
    ap.add_argument("jsonl")
    ap.add_argument("--log", default="")
    ap.add_argument("--monitor", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.isfile(args.jsonl):
        print("错误：结果文件不存在：%s" % args.jsonl)
        return 2
    rows = load_jsonl(args.jsonl)
    if not rows:
        print("错误：解析出 0 条记录")
        return 2

    import io
    buf = io.StringIO()
    _stdout = sys.stdout
    sys.stdout = buf
    try:
        print("结果文件 %s（%d 题）" % (os.path.basename(args.jsonl), len(rows)))
        if args.log:
            print("日志     %s" % args.log)
        sec_a(rows)
        sec_b(rows)
        sec_c(rows)
        sec_d(rows, args.log or None)
        sec_e(rows)
        sec_f(rows)
        sec_g(rows)
        sec_h(rows)
        sec_i(args.monitor)
    finally:
        sys.stdout = _stdout

    text = buf.getvalue()
    text = text.replace("|", "\u2223")          # 竖线换 ∣
    text = re.sub(r"^\s*```.*$", "", text, flags=re.M)   # 禁围栏
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
