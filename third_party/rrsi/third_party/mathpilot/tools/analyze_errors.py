# -*- coding: utf-8 -*-
"""
错题逐步归因分析工具（2026-09-02 用户要求：45 题正确率低时，要能定位错在哪个环节）
====================================================================================

输入：评测结果 JSONL（run_eval.py 输出，结果行含 orchestrator 打包的 diag 字段）
输出：
  1. 控制台汇总：正确率 + 各环节异常计数（找系统性 bug 信号）
  2. HTML 体检报告（逐题归因 + 阶段状态 + trace 关键事件）

归因环节划分：
  ① 理解  preverify/formal_gaps        —— 题目理解错 → 原 Lean 前置验证 fail / gaps 多
  ② 规划  蓝图节点/骨架评审/DAG评审    —— 子目标不适定 → skeleton_review replan/ill_posed
  ③ 求解  子目标 trace/占位符          —— 子目标算错 → placeholder / result 标记失败
  ④ 验证  audit_gate/verdicts          —— AuditGate（程序/反例/rubric）拒错解 / 漏检错解
  ⑤ 预算  budget_skips/degraded/tier   —— 时间不够被截断
  ⑥ 输出  判分 error_class             —— pred/gold 不匹配（真错或判分问题）

用法：
  D:/python/python.exe tools/analyze_errors.py eval_dag_ab45_current.jsonl [--html report.html]
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter

# ---------------------------------------------------------------------------
# 归因规则
# ---------------------------------------------------------------------------

def attribute_question(row: dict) -> list[str]:
    """对单题做环节归因，返回疑点列表（可能多个，按严重度排序）。"""
    d = row.get("diag") or {}
    flags: list[str] = []
    if not d:
        return ["[无诊断数据：旧版结果或链路提前失败]"]

    # ⑤ 预算 / 截断
    bs = int(d.get("budget_skips", 0) or 0)
    if d.get("placeholder"):
        flags.append("⚠ 子目标求解失败占位符（时间/LLM失败截断主链路）")
    if bs >= 10:
        flags.append(f"⚠ 预算跳过 {bs} 次（单题 deadline 内未跑完所有步骤）")

    # ③ 求解环节
    sub = d.get("subgoal_trace") or []
    sub_fail = []
    for s in sub:
        if not isinstance(s, dict):
            continue
        res = str(s.get("result", ""))
        if res.startswith("[子目标") or not res.strip():
            sub_fail.append(s)
    if sub_fail:
        flags.append(f"✗ 子目标求解失败 {len(sub_fail)}/{len(sub)}："
                     + ", ".join(str(s.get("title", "?"))[:30] for s in sub_fail[:3]))
    elif sub:
        # 子目标都有结果但整体错 → 某步算错（无法自动定位，标红让 trace 复核）
        flags.append(f"? 子目标 {len(sub)} 步均有输出但结论错（需人工复核 trace 定位错步）")

    # ② 规划环节
    sr = d.get("skeleton_review") or {}
    if sr:
        if sr.get("degraded"):
            flags.append("⚠ 骨架评审降级放行（degraded）")
        vd = sr.get("verdicts") or {}
        ill = [k for k, v in vd.items() if isinstance(v, dict) and v.get("verdict") == "ill_posed"]
        ns = [k for k, v in vd.items() if isinstance(v, dict) and v.get("verdict") == "not_simplifying"]
        if ill:
            flags.append(f"? 骨架评审判 ill_posed {len(ill)} 节点（{', '.join(str(x) for x in ill[:3])}）")
        if ns:
            flags.append(f"? 骨架评审判 not_simplifying {len(ns)} 节点")
        if sr.get("overall") == "replan":
            flags.append("? 骨架评审触发 replan 后仍产出（规划质量存疑）")
    dr = d.get("dag_review") or {}
    if dr.get("degraded"):
        flags.append("⚠ DAG 评审降级")
    if not sr and not dr and d.get("blueprint_nodes", 0) == 0:
        flags.append("? 无蓝图 DAG（未走子目标分解路径）")

    # ① 理解环节
    pt = d.get("preverify_trace") or {}
    if isinstance(pt, dict) and pt.get("verdict") == "fail":
        flags.append("? Lean 前置验证失败（理解环节异常）")
    gaps = d.get("formal_gaps") or []
    if gaps:
        flags.append(f"? 前置验证发现 {len(gaps)} 个形式化缺口")

    # ④ AuditGate 客观审核（旧日志兼容 lean_gate 键）
    lg = d.get("audit_gate") or d.get("lean_gate") or []
    if lg:
        rejects = sum(1 for g in lg if isinstance(g, dict)
                      and g.get("verdict") == "reject")
        flags.append(f"i AuditGate 审核 {len(lg)} 条，拒绝 {rejects} 条")

    # ⑥ 判分
    if row.get("correct") is False:
        ec = row.get("error_class") or ""
        if ec in ("extract_failed", "format_unresolved"):
            flags.append(f"? 判分侧 {ec}（答案格式未识别，可能判分器可救）")
    if not flags:
        flags.append("（无环节异常标记，疑为推理本身错误）")
    return flags


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    scored = [r for r in rows if r.get("correct") is not None]
    ok = sum(1 for r in scored if r.get("correct"))
    diag_rows = [r for r in rows if r.get("diag")]
    agg = {
        "total": total, "scored": len(scored), "correct": ok,
        "accuracy": ok / len(scored) if scored else 0.0,
        "diag_count": len(diag_rows),
        # 环节异常计数（找系统性 bug）
        "placeholder": sum(1 for r in diag_rows if r["diag"].get("placeholder")),
        "budget_skips_ge10": sum(1 for r in diag_rows
                                 if int(r["diag"].get("budget_skips", 0) or 0) >= 10),
        "skeleton_degraded": sum(1 for r in diag_rows
                                 if (r["diag"].get("skeleton_review") or {}).get("degraded")),
        "skeleton_replan": sum(1 for r in diag_rows
                               if (r["diag"].get("skeleton_review") or {}).get("overall") == "replan"),
        "dag_degraded": sum(1 for r in diag_rows
                            if (r["diag"].get("dag_review") or {}).get("degraded")),
        "preverify_fail": sum(1 for r in diag_rows
                              if (r["diag"].get("preverify_trace") or {}).get("verdict") == "fail"),
        "no_blueprint": sum(1 for r in diag_rows
                            if int(r["diag"].get("blueprint_nodes", 0) or 0) == 0),
        "sub_fail_any": 0,
        "sub_total": 0,
    }
    for r in diag_rows:
        sub = r["diag"].get("subgoal_trace") or []
        agg["sub_total"] += len(sub)
        if any(str(s.get("result", "")).startswith("[子目标") for s in sub if isinstance(s, dict)):
            agg["sub_fail_any"] += 1
    agg["error_dist"] = Counter(r.get("error_class") or "correct" for r in scored)
    return agg


def _fmt_stage(d: dict) -> list[tuple[str, str]]:
    """抽取各阶段一句话状态，供 HTML 展示。"""
    out = []
    # 理解
    pt = d.get("preverify_trace") or {}
    pv = pt.get("verdict") if isinstance(pt, dict) else None
    out.append(("① 理解", f"preverify={pv or '未跑'} gaps={len(d.get('formal_gaps') or [])}"))
    # 规划
    bn = d.get("blueprint_nodes", 0)
    sr = d.get("skeleton_review") or {}
    dr = d.get("dag_review") or {}
    sr_s = sr.get("overall", "-") if sr else "未评审"
    out.append(("② 规划", f"蓝图节点={bn} 骨架评审={sr_s}"
                          + (f" DAG评审={dr.get('verdict', '-')}" if dr else "")))
    # 求解
    sub = d.get("subgoal_trace") or []
    if sub:
        n_fail = sum(1 for s in sub if isinstance(s, dict)
                     and str(s.get("result", "")).startswith("[子目标"))
        out.append(("③ 求解", f"子目标 {len(sub)} 步，失败 {n_fail}"))
    else:
        out.append(("③ 求解", "无子目标轨迹（未走 DAG 路径）"))
    # AuditGate 客观审核（旧日志兼容 lean_gate 键）
    lg = d.get("audit_gate") or d.get("lean_gate") or []
    rejects = sum(1 for g in lg if isinstance(g, dict)
                  and g.get("verdict") == "reject")
    out.append(("④ 审核", f"AuditGate {len(lg)} 条，拒绝 {rejects} 条"))
    # 预算
    out.append(("⑤ 预算", f"档位={d.get('tier', '-')} 软预算={d.get('soft_budget', '-')}s "
                          f"跳过={d.get('budget_skips', 0)}次"))
    return out


def _trace_events(trace: list) -> list[dict]:
    """从完整 trace 抽关键事件（budget_skip/degraded/失败类 + 阶段边界）。"""
    if not trace:
        return []
    keys = []
    stage_boundary = {"classify", "paper_pacer", "lean_preverify", "blueprint",
                      "skeleton_review", "dag_review", "subgoal", "merge",
                      "verify", "revise", "self_improve", "lean_gate",
                      "audit_gate", "budget_skip"}
    for t in trace:
        if not isinstance(t, dict):
            continue
        step = str(t.get("step", ""))
        content = str(t.get("content", ""))
        agent = str(t.get("agent", ""))
        low = (step + " " + content).lower()
        is_key = step in stage_boundary or any(k in low for k in
            ("degrad", "fail", "占位", "跳过", "replan", "解析失败", "timeout",
             "超时", "非法", "拒绝", "exception", "异常", "截断", "修正"))
        if is_key:
            keys.append({"agent": agent, "step": step, "content": content[:150]})
    return keys


def build_html(rows: list[dict], agg: dict, title: str) -> str:
    P = []
    P.append(f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             "<style>body{font-family:'Segoe UI',sans-serif;margin:24px;background:#f7f8fa;color:#222}"
             "h1{font-size:20px}h2{font-size:16px;border-left:4px solid #4a6cf7;padding-left:8px;margin-top:28px}"
             "table{border-collapse:collapse;width:100%;margin:8px 0;background:#fff;font-size:13px}"
             "th,td{border:1px solid #dde;padding:5px 8px;text-align:left;vertical-align:top}"
             "th{background:#eef1fb}.ok{color:#0a7d33;font-weight:700}.bad{color:#c00;font-weight:700}"
             ".warn{color:#b85c00}.stage{color:#555;font-size:12px}.flag{color:#b85c00;font-size:12px}"
             "pre{background:#fff;border:1px solid #ddd;padding:8px;font-size:12px;overflow-x:auto}"
             ".card{background:#fff;border:1px solid #e2e5ef;border-radius:8px;padding:12px;margin:10px 0}</style>"
             "</head><body>")
    P.append(f"<h1>{html.escape(title)}</h1>")
    # 汇总
    P.append(f"<div class='card'><b>正确率 {agg['accuracy']:.1%} "
             f"（{agg['correct']}/{agg['scored']}）</b> · 总 {agg['total']} 题 · 诊断 {agg['diag_count']} 题"
             f"<br>error 分布：{html.escape(str(dict(agg['error_dist'])))}</div>")
    P.append("<h2>环节健康汇总（找系统性 bug 信号）</h2>")
    P.append("<table><tr><th>信号</th><th>数量</th><th>说明</th></tr>")
    sig = [
        ("占位符子目标失败", agg["placeholder"], "主链路被截断（时间/LLM失败）——需查预算或超时"),
        ("预算跳过≥10次", agg["budget_skips_ge10"], "单题 deadline 内步骤未跑完"),
        ("骨架评审 degraded", agg["skeleton_degraded"], "评审降级放行（质量门失效信号）"),
        ("骨架评审 replan", agg["skeleton_replan"], "初版蓝图规划质量差"),
        ("DAG 评审 degraded", agg["dag_degraded"], "DAG 评审降级"),
        ("前置验证 fail", agg["preverify_fail"], "题目理解环节失败"),
        ("无蓝图（未走DAG）", agg["no_blueprint"], "子目标路径未触发（直接求解/快车道）"),
        ("子目标含失败", agg["sub_fail_any"], "求解环节有子目标失败"),
    ]
    for name, n, note in sig:
        cls = "bad" if n and n / max(1, agg["diag_count"]) > 0.2 else ""
        P.append(f"<tr><td>{name}</td><td class='{cls}'>{n}</td><td>{note}</td></tr>")
    P.append("</table>")
    # 逐题
    P.append("<h2>逐题归因</h2>")
    P.append("<table><tr><th>题号</th><th>对错</th><th>耗时</th><th>阶段状态</th><th>归因疑点</th></tr>")
    for r in sorted(rows, key=lambda x: str(x.get("id", ""))):
        qid = str(r.get("id", "?")).split("-")[-1]
        ok = r.get("correct")
        cls = "ok" if ok else ("bad" if ok is False else "")
        tag = "✅" if ok else ("❌" if ok is False else "—")
        stages = _fmt_stage(r.get("diag") or {}) if r.get("diag") else []
        stage_html = "<br>".join(f"<span class='stage'>{s}: {c}</span>" for s, c in stages)
        flags = attribute_question(r)
        flag_html = "<br>".join(f"<span class='flag'>{html.escape(f)}</span>" for f in flags)
        P.append(f"<tr><td>{qid}</td><td class='{cls}'>{tag} {html.escape(str(r.get('error_class') or ''))}</td>"
                 f"<td>{r.get('elapsed_sec', '')}s</td><td>{stage_html}</td><td>{flag_html}</td></tr>")
    P.append("</table>")
    # 错题 trace 关键事件（前 10 错题）
    wrongs = [r for r in rows if r.get("correct") is False and r.get("diag")]
    if wrongs:
        P.append("<h2>错题 trace 关键事件（前 10）</h2>")
        for r in sorted(wrongs, key=lambda x: str(x.get("id", "")))[:10]:
            qid = str(r.get("id", "?")).split("-")[-1]
            evs = _trace_events(r.get("trace") or [])
            if not evs:
                P.append(f"<div class='card'><b>{qid}</b>：无 trace（旧数据或链路未记录）</div>")
                continue
            lines = "".join(
                f"<pre>{html.escape(e['agent'])} | {html.escape(e['step'])} | "
                f"{html.escape(e['content'][:120])}</pre>" for e in evs[-12:])
            P.append(f"<div class='card'><b>{qid}</b>（关键事件 {len(evs)} 条，最近 12 条）<br>{lines}</div>")
    P.append("</body></html>")
    return "".join(P)


def main():
    ap = argparse.ArgumentParser(description="错题逐步归因分析")
    ap.add_argument("result_file", help="评测结果 jsonl")
    ap.add_argument("--html", default="", help="输出 HTML 报告路径")
    args = ap.parse_args()
    rows = []
    for line in open(args.result_file, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    agg = summarize(rows)
    print(f"正确率 {agg['accuracy']:.1%} ({agg['correct']}/{agg['scored']})  总 {agg['total']}  诊断 {agg['diag_count']}")
    print(f"error: {dict(agg['error_dist'])}")
    print(f"占位符 {agg['placeholder']} | 预算跳过≥10 {agg['budget_skips_ge10']} | "
          f"骨架degraded {agg['skeleton_degraded']} | replan {agg['skeleton_replan']} | "
          f"前置fail {agg['preverify_fail']} | 无蓝图 {agg['no_blueprint']} | 子目标失败 {agg['sub_fail_any']}")
    print("\n=== 逐题归因 ===")
    for r in sorted(rows, key=lambda x: str(x.get("id", ""))):
        qid = str(r.get("id", "?")).split("-")[-1]
        ok = r.get("correct")
        tag = "✅" if ok else ("❌" if ok is False else "—")
        print(f"{tag} {qid} | {str(r.get('error_class') or ''):20s} | {str(r.get('elapsed_sec', '')):>6}s")
        for f in attribute_question(r):
            print(f"      {f}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(build_html(rows, agg, args.result_file))
        print(f"\nHTML 报告已写: {args.html}")


if __name__ == "__main__":
    main()
