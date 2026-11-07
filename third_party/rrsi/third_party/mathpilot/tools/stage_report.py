# -*- coding: utf-8 -*-
"""逐环节报告：把一次 run 的产物拆成「每个环节发生了什么」。

为什么单独做一个（2026-10-23 用户要求）
----------------------------------------
「这次必定出现 bug，你要准备记录每个环节的信息。」
上一轮的教训是：跑完 4 小时才发现某个环节没跑，而且看不出**是没跑、跑了没用、还是跑了但全失败**。
本工具把三件事摆在一张表里：

    ① 这个环节**耗时多少**
    ② 它的**输入规模 / 产出规模**
    ③ 它的**判定分布与失败原因**（含日志里的 WARNING）

用法
----
    python tools/stage_report.py --result results/xxx.jsonl [--log results/xxx.log]
    python tools/stage_report.py --result ... --md out.md     # 存成 Markdown

不依赖网络、不改主链、只读产物。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

# 环节 → (读哪个 diag 键, 说明)
STAGE_SPEC = [
    ("2.6_pre_audit", ("preverify_trace",), "预审"),
    ("2.7_subgoal_main", ("subgoal_stats", "subgoal_gaps", "subgoal_delivery"),
     "子目标主链"),
    ("3_solve", ("n_candidates", "answer_normalize"), "求解产候选"),
    ("3.4_collab", ("subgoal_collab",), "三角色协作"),
    ("4.0_normalize", ("answer_normalize",), "答案规范化"),
    ("4.1_self_audit", ("self_audit",), "AI 自审"),
    ("4.2_falsify", ("adversarial_gate",), "对抗证伪"),
    ("4.3_lean_check", ("lean_gate", "lean_active"), "Lean 代回"),
    ("4.4_math_check", ("math_check",), "数学工具检测"),
    ("4.5_vote", ("vote_gate",), "投票"),
    ("4.6_rewrite", ("rewrite_gate",), "池空重写"),
]


def load_result(path: str) -> dict:
    for line in io.open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line:
            return json.loads(line)
    return {}


def scan_log(path: str) -> dict:
    """从日志里抽三类信号：WARNING/ERROR 计数与样例、各机制触发次数。"""
    out = {"warn": [], "error": [], "counts": {}}
    if not path or not os.path.isfile(path):
        return out
    keys = ("subgoal_selfcheck", "subgoal_lean", "mcp run_code",
            "编译失败", "Lean 环境不可用", "finish_reason=length",
            "池空", "重写", "truncated")
    for line in io.open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"\b(WARNING|ERROR)\b", line)
        if m:
            txt = line.strip()[:200]
            (out["warn"] if m.group(1) == "WARNING" else out["error"]).append(txt)
        for k in keys:
            if k in line:
                out["counts"][k] = out["counts"].get(k, 0) + 1
    return out


def _fmt(v, n=120) -> str:
    if v is None:
        return "-"
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    return s[:n]


def build_rows(res: dict, log: dict) -> list[dict]:
    g = res.get("diag") or {}
    timers = g.get("stage_timers") or {}
    rows = []
    for stage, keys, desc in STAGE_SPEC:
        row = {"阶段": stage, "说明": desc,
               "耗时(s)": round(float(timers.get(stage, 0) or 0), 1)}
        for k in keys:
            if k in g:
                row[k] = g[k]
        rows.append(row)
    return rows


def render(res: dict, log: dict) -> str:
    lines = []
    gold = res.get("gold")
    pred = res.get("predicted")
    lines.append("# 逐环节报告\n")
    lines.append("- 题目：%s" % res.get("id"))
    lines.append("- gold：`%s`" % str(gold)[:80])
    lines.append("- predicted：`%s`" % str(pred)[:120])
    lines.append("- correct：**%s**" % res.get("correct"))
    lines.append("- 总耗时：%.0f s（%.2f h）" % (
        res.get("elapsed_sec") or 0, (res.get("elapsed_sec") or 0) / 3600))
    lines.append("")

    lines.append("## 一、各环节耗时与产出\n")
    lines.append("| 阶段 | 说明 | 耗时(s) | 占比 | 关键产物 |")
    lines.append("|---|---|---|---|---|")
    tot = float(res.get("elapsed_sec") or 1) or 1
    g = res.get("diag") or {}
    for stage, keys, desc in STAGE_SPEC:
        t = float((g.get("stage_timers") or {}).get(stage, 0) or 0)
        prod = _fmt(g.get(keys[0]) if keys else None, 90)
        lines.append("| `%s` | %s | %.0f | %.1f%% | %s |"
                     % (stage, desc, t, 100 * t / tot, prod))
    lines.append("")

    lines.append("## 二、机制触发情况（来自日志）\n")
    if log.get("counts"):
        lines.append("| 关键词 | 出现次数 |")
        lines.append("|---|---|")
        for k, v in sorted(log["counts"].items(), key=lambda x: -x[1]):
            lines.append("| %s | %d |" % (k, v))
    else:
        lines.append("（日志里没有关键词命中，或未提供 --log）")
    lines.append("")

    lines.append("## 三、判定分布\n")
    for key, label in (("self_audit", "4.1 自审"),
                       ("adversarial_gate", "4.2 证伪"),
                       ("rewrite_gate", "4.6 重写")):
        v = g.get(key)
        if v:
            lines.append("- **%s**：%s" % (label, _fmt(v, 260)))
    lg = g.get("lean_gate") or []
    if lg:
        lines.append("- **4.3 Lean**（逐候选）：")
        for e in lg[:10]:
            if isinstance(e, dict):
                lines.append("  - id=%s verdict=%s failure_kind=%s degraded=%s"
                             % (e.get("id"), e.get("verdict"),
                                e.get("failure_kind") or "-",
                                e.get("degraded") or "-"))
    else:
        lines.append("- **4.3 Lean**：（空）lean_active=%s" % g.get("lean_active"))
    sub = (g.get("metadata") or {}).get("subgoal_suspects") if isinstance(
        g.get("metadata"), dict) else None
    lines.append("- **子目标自检**：%s" % (_fmt(sub, 200) if sub else "（无记录）"))
    lines.append("")

    lines.append("## 四、异常与警告（前 12 条）\n")
    for w in (log.get("warn") or [])[:12]:
        lines.append("- ⚠ %s" % w)
    for e in (log.get("error") or [])[:6]:
        lines.append("- ❌ %s" % e)
    if not (log.get("warn") or log.get("error")):
        lines.append("（无）")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="逐环节报告")
    ap.add_argument("--result", required=True, help="结果 jsonl")
    ap.add_argument("--log", default="", help="同名的 .log（可选）")
    ap.add_argument("--md", default="", help="输出 Markdown 路径（可选）")
    args = ap.parse_args(argv)

    res = load_result(args.result)
    log = scan_log(args.log)
    text = render(res, log)
    print(text)
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        io.open(args.md, "w", encoding="utf-8").write(text)
        print("\n已写入：%s" % args.md, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
