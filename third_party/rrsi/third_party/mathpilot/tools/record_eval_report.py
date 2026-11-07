#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测结果记录器：把 run_eval.py 的 JSONL 结果 → 「答案错在哪 + 耗时」Markdown 报告。

用法:
  D:/python/python.exe tools/record_eval_report.py results/official112_local_0910.jsonl
  D:/python/python.exe tools/record_eval_report.py <results.jsonl> --md <out.md> --top 15

输入字段（run_eval.py 输出行）:
  id / domain / correct / elapsed_sec / error_class / gold / predicted / diag

输出:
  1) 控制台：总览 + 错因分布 + 耗时分布
  2) Markdown 报告（默认 results/<stem>_报告.md）
"""
# 2026-10-01 去同名：_stage_group -> _stage_group_en（它与 diagnosis_report 的同名函数逻辑相同但**返回英文标签**，加 _en 区分）
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

# 判分器错误分类的中文解释（口径见 run_eval._classify_error）
ERROR_CLASS_CN = {
    "empty_output": "空输出/只剩定界符（解析或截断 bug）",
    "extract_failed": "答案抽取失败（混入推理过程、LaTeX 截断、只吐标题）",
    "format_unresolved": "答案未定型（含未求值符号或条件式）",
    "value_wrong": "真算错（两边都是裸数却不等）",
    "expr_wrong": "表达式错（推理解答错）",
}

# 阶段分组：把细粒度 stage 归到主阶段
def _stage_group_en(k: str) -> str:
    if k.startswith("2.6"):
        return "preverify"
    if k.startswith("2.7"):
        return "subgoal_main"
    if k.startswith("3_solve"):
        return "solve"
    if "lean_candidate" in k or "lean_filter" in k or "3.6" in k or "3.5.1" in k:
        return "lean_candidate_filter"
    if k.startswith("3.2"):
        return "complete"
    if k.startswith("4_"):
        return "verify"
    if k.startswith("1_") or k.startswith("2.5"):
        return "classify_diff"
    if k.startswith("5") or k.startswith("6"):
        return "revise_format_gate"
    return "other"


STAGE_ORDER = ["classify_diff", "preverify", "subgoal_main", "solve",
               "lean_candidate_filter", "complete", "verify", "revise_format_gate", "other"]


def _clip(s, n=220) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    return s if len(s) <= n else s[:n] + " …"


def _clues(row: dict) -> list[str]:
    """从 diag 提取结构化归因线索（人话描述，避免打印原始 JSON 片段）。"""
    d = row.get("diag") or {}
    out: list[str] = []
    if not d:
        return ["无诊断数据"]

    # 预算
    bs = int(d.get("budget_skips", 0) or 0)
    if bs:
        out.append(f"预算跳过 {bs} 次（单题时限内未跑完所有步骤）")
    if d.get("placeholder"):
        out.append("出现子目标占位符（求解失败/被截断）")
    tr = d.get("tier")
    if tr:
        out.append(f"预算档位 = {tr}")

    # 理解环节
    pv = d.get("preverify_trace") or {}
    if isinstance(pv, dict) and pv.get("verdict"):
        out.append(f"Lean 前置验证判定 = {pv.get('verdict')}")
    for x in (d.get("formal_gaps") or [])[:2]:
        if isinstance(x, dict):
            out.append(f"形式化缺口[{x.get('kind')}]：{_clip(x.get('detail'), 130)}")
        else:
            out.append(f"形式化缺口：{_clip(x, 130)}")

    # 验证环节：Lean 最终闸门（结构化）
    for g in (d.get("lean_gate") or []):
        if isinstance(g, dict) and g.get("step") == "final_gate":
            bits = [f"判定={g.get('verdict')}"]
            if g.get("degraded"):
                bits.append(f"降级={g['degraded']}")
            if g.get("feedback"):
                bits.append(f"反馈={_clip(g['feedback'], 110)}")
            out.append("Lean 最终闸门：" + "；".join(bits))

    # 验证环节：AuditGate 汇总
    ag = [g for g in (d.get("audit_gate") or [])
          if isinstance(g, dict) and g.get("step") == "candidate_audit"]
    if ag:
        vc = Counter((g.get("verdict") or "none") for g in ag)
        out.append(f"AuditGate 审核 {len(ag)} 条：" +
                   "，".join(f"{k}={v}" for k, v in vc.most_common()))

    # 数值攻击（直接给反例内容）
    va = d.get("value_attack")
    if va:
        for x in (va if isinstance(va, list) else [va])[:2]:
            out.append(f"数值攻击：{_clip(x, 150)}")

    # 求解环节
    st = d.get("subgoal_trace") or []
    if st:
        fails = [s for s in st if isinstance(s, dict) and
                 (not str(s.get("result", "")).strip() or str(s.get("result", "")).startswith("[子目标"))]
        if fails:
            out.append(f"子目标 {len(st)} 步，其中 **{len(fails)} 步求解失败**（空/占位）")
        else:
            out.append(f"子目标 {len(st)} 步均有输出，但最终结论错（错误在中间推理步骤）")

    # 修订反馈（含错误定位）
    rf = " ".join(str(x) for x in (d.get("revise_feedback") or [])).strip()
    if rf:
        out.append(f"修订反馈：{_clip(rf, 260)}")
    rr = int(d.get("revise_round") or 0)
    if rr:
        out.append(f"触发修订轮数 = {rr}")
    elif va:
        out.append("⚠ 已检测到问题，但**未触发修订**（revise_round=0）")

    df = d.get("degraded_flags")
    if df:
        out.append(f"降级标记 = {df}")
    return out


def _bucket(sec: float) -> str:
    if sec < 120:
        return "<120s"
    if sec < 540:
        return "120-540s"
    if sec < 1200:
        return "540-1200s"
    return ">=1200s(截断)"


def main() -> None:
    ap = argparse.ArgumentParser(description="评测结果「错因+耗时」记录器")
    ap.add_argument("result_jsonl")
    ap.add_argument("--md", default="", help="Markdown 输出路径（默认 results/<stem>_报告.md）")
    ap.add_argument("--top", type=int, default=15, help="耗时 TopN（默认 15）")
    args = ap.parse_args()

    path = args.result_jsonl
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        print("结果文件为空")
        return

    stem = os.path.splitext(os.path.basename(path))[0]
    md_path = args.md or os.path.join(os.path.dirname(path) or ".", f"{stem}_报告.md")

    scored = [r for r in rows if r.get("correct") is not None]
    n_correct = sum(1 for r in scored if r.get("correct"))
    acc = n_correct / len(scored) * 100 if scored else 0.0
    elapsed = [float(r.get("elapsed_sec") or 0.0) for r in rows]
    wrong = [r for r in rows if r.get("correct") is False]

    # 错因分布
    ec = Counter((r.get("error_class") or "未分类") for r in wrong)
    # 分桶
    buckets = defaultdict(lambda: {"n": 0, "correct": 0, "t": 0.0})
    for r in rows:
        t = float(r.get("elapsed_sec") or 0.0)
        b = buckets[_bucket(t)]
        b["n"] += 1
        b["t"] += t
        if r.get("correct") is True:
            b["correct"] += 1
    # 阶段耗时
    stage_sum = defaultdict(float)
    for r in rows:
        st = (r.get("diag") or {}).get("stage_timers") or {}
        for k, v in st.items():
            if isinstance(v, (int, float)):
                stage_sum[_stage_group_en(k)] += float(v)
    stage_total = sum(stage_sum.values()) or 1.0

    lines: list[str] = []
    A = lines.append
    A(f"# 评测结果记录 — {stem}")
    A("")
    A(f"- 结果文件：`{os.path.abspath(path)}`")
    A(f"- 题量：**{len(rows)}** 题（参与判分 {len(scored)} 题，未判分 {len(rows) - len(scored)} 题）")
    A(f"- 正确率：**{n_correct}/{len(scored)} = {acc:.1f}%**")
    A(f"- 单题耗时合计：**{sum(elapsed) / 3600:.2f} h**（各题耗时累加，非墙钟）；单题均值 {statistics.mean(elapsed):.0f}s / 中位 {statistics.median(elapsed):.0f}s / 最大 {max(elapsed):.0f}s")
    A("")
    A("## 一、耗时情况")
    A("")
    A("### 1.1 分桶（耗时 × 正确率）")
    A("")
    A("| 耗时桶 | 题数 | 占比 | 平均耗时 | 正确数 | 桶内正确率 |")
    A("|---|---|---|---|---|---|")
    for k in ["<120s", "120-540s", "540-1200s", ">=1200s(截断)"]:
        b = buckets.get(k)
        if not b or not b["n"]:
            continue
        A(f"| {k} | {b['n']} | {b['n'] / len(rows) * 100:.0f}% | {b['t'] / b['n']:.0f}s | {b['correct']} | {b['correct'] / b['n'] * 100:.0f}% |")
    A("")
    A("### 1.2 阶段耗时占比")
    A("")
    A("| 阶段 | 累计耗时 | 占比 |")
    A("|---|---|---|")
    for k in STAGE_ORDER:
        if stage_sum.get(k):
            A(f"| {k} | {stage_sum[k]:.0f}s | {stage_sum[k] / stage_total * 100:.1f}% |")
    A("")
    A(f"### 1.3 单题耗时 Top {args.top}")
    A("")
    A("| 题号 | domain | 对错 | 耗时 | 判分 | 错误分类 |")
    A("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(float(x.get("elapsed_sec") or 0)))[: args.top]:
        corr = "✅" if r.get("correct") is True else ("❌" if r.get("correct") is False else "—")
        A(f"| {r.get('id')} | {r.get('domain', '')} | {corr} | {float(r.get('elapsed_sec') or 0):.0f}s | "
          f"{'截断' if float(r.get('elapsed_sec') or 0) >= 1200 else '—'} | {r.get('error_class') or ''} |")
    A("")
    A("### 1.4 全题耗时明细")
    A("")
    A("| 题号 | domain | 对错 | 耗时(s) |")
    A("|---|---|---|---|")
    for r in rows:
        corr = "✅" if r.get("correct") is True else ("❌" if r.get("correct") is False else "—")
        A(f"| {r.get('id')} | {r.get('domain', '')} | {corr} | {float(r.get('elapsed_sec') or 0):.0f} |")
    A("")
    A("## 二、错题记录（错在哪）")
    A("")
    A(f"错题共 **{len(wrong)}** 题。错误分类分布：")
    A("")
    A("| 错误分类 | 题数 | 占比 | 含义 |")
    A("|---|---|---|---|")
    for k, v in ec.most_common():
        A(f"| {k} | {v} | {v / max(1, len(wrong)) * 100:.0f}% | {ERROR_CLASS_CN.get(k, '—')} |")
    A("")
    for i, r in enumerate(wrong, 1):
        A(f"### 二.{i} `{r.get('id')}` ｜ {r.get('domain', '')} ｜ {float(r.get('elapsed_sec') or 0):.0f}s ｜ {r.get('error_class') or '未分类'}")
        A("")
        A(f"- **错误分类**：{r.get('error_class') or '未分类'} —— {ERROR_CLASS_CN.get(r.get('error_class'), '')}")
        A(f"- **标准答案**：`{_clip(r.get('gold'), 160)}`")
        A(f"- **模型输出**：`{_clip(r.get('predicted'), 300)}`")
        cl = _clues(r)
        if cl:
            A("- **归因线索**：")
            for c in cl:
                A(f"  - {c}")
        A("")

    text = "\n".join(lines) + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 控制台汇总
    print(f"共 {len(rows)} 题，参与判分 {len(scored)} 题")
    print(f"正确率 {n_correct}/{len(scored)} = {acc:.1f}%")
    print(f"总墙钟 {sum(elapsed)/3600:.2f} h；单题均值 {statistics.mean(elapsed):.0f}s / 中位 {statistics.median(elapsed):.0f}s / 最大 {max(elapsed):.0f}s")
    print("错因分布：" + ("，".join(f"{k}={v}" for k, v in ec.most_common()) or "无错题"))
    print("耗时桶：" + "；".join(
        f"{k} {v['n']}题/{v['correct']}对" for k, v in sorted(buckets.items(), key=lambda kv: kv[0])))
    print(f"报告已写出：{os.path.abspath(md_path)}")


if __name__ == "__main__":
    main()
