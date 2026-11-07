# -*- coding: utf-8 -*-
"""编辑账本 —— 一次原子改动 = 一行记录（R13 七字段）。

为什么需要它
============
现有 `results/_research/ledger.jsonl` 回答的是「**这道题为什么错**」（错题归因），
回答不了「**我上次改的那一处到底有没有用**」（编辑记录）。后者才是自动化的燃料：
提案器要读历史才知道哪些改动该重提、哪些单元从没碰过、哪些改了从没涨分。

七字段（对齐《智能体要求》R13）
==============================
    组件 / 假设 / 实际改动 / 指标变化 / 成本变化 / 是否采纳 / 拒绝原因
本 schema 另加 `stage` `dim` `eu_id`（定位到三级）与 `commit` `evidence`（证据链）。

★ 2026-10-25 增补三字段（抄自 RRSI `propose.py::done()` 契约）
============================================================
RRSI 的每个 edit 必填 8 项，其中三项我们原账本没有，恰好是最能防"乱改"的：
    `why_not_lower_lever`  —— 为什么不用更省力的改法（防提示词塌缩 R11）
    `predicted_affected`   —— 事前预测会影响哪些题（事后可核 = 可归因的强实现）
    `retroactive_check`    —— 事前写清"怎么验证它真有效"
默认**可选但会警告**；加 `--strict` 则必填。理由：历史记录没有这三项，
一刀切必填会让老账不可读。`stats` 里报"三字段完整率"作为质量指标。

三条派生判据（对应"哪里有用/哪里没用"）
======================================
    ① 消费率 = 0        —— 由 `tools/prune_scan.py` 提供，提案前直接拒
    ② 零正贡献           —— 窗口内该 EU 的最大 Δ ≤ 0 ⇒ 剪枝候选（R14：关闭不删）
    ③ 从未被改过         —— 账本无记录 ⇒ 探索配额（R11 防塌缩）

用法
====
    python tools/edit_ledger.py add --stage 2.7_subgoal_main --dim flow \\
        --eu flow.blueprint_multi_gen_rounds --hypothesis "3→2 遍减少锁值" \\
        --diff "3 -> 2" --delta-metric 3.0 --delta-cost 0.12 --accepted

    python tools/edit_ledger.py add ... --reject-reason "低于噪声带 δ"
    python tools/edit_ledger.py list --stage 2.7
    python tools/edit_ledger.py stats

⚠ 写入 `results/_edit_ledger/ledger.jsonl`（追加，不覆盖）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SCHEMA = "edit_ledger/1"
DIMS = ("prompt", "flow", "tool", "memory", "context")
_DIR = os.path.join(_ROOT, "results", "_edit_ledger")
PATH = os.path.join(_DIR, "ledger.jsonl")


def load() -> list:
    if not os.path.isfile(PATH):
        return []
    out = []
    for line in io.open(PATH, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def append(rec: dict) -> None:
    os.makedirs(_DIR, exist_ok=True)
    with io.open(PATH, "a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate(rec: dict, strict: bool = False) -> tuple:
    """返回 (错误列表, 警告列表)；错误非空 = 拒写，警告只提示。

    `strict=True` 时把"缺三字段"的警告升级为错误（正式流程用）。
    """
    errs, warns = [], []
    for k in ("stage", "dim", "eu_id", "hypothesis", "diff"):
        if not str(rec.get(k) or "").strip():
            errs.append("缺少 %s" % k)
    if rec.get("dim") not in DIMS:
        errs.append("dim 必须是 %s 之一" % "/".join(DIMS))
    if rec.get("accepted") and rec.get("delta_metric") is None:
        errs.append("未测指标就标 accepted（违反 R7：报分必须带度量）")
    if not rec.get("accepted") and not str(rec.get("reject_reason") or "").strip():
        errs.append("未采纳但无 reject_reason（违反 R2：禁静默）")
    # RRSI done() 三字段
    missing = [k for k in ("why_not_lower_lever", "predicted_affected", "retroactive_check")
               if not str(rec.get(k) or "").strip()]
    if missing:
        msg = "缺 RRSI 三字段：%s（--strict 下必填）" % ", ".join(missing)
        (errs if strict else warns).append(msg)
    return errs, warns


def cmd_add(args) -> int:
    rec = {
        "schema": SCHEMA,
        "id": "%s-%04d" % (time.strftime("%Y%m%d%H%M%S"), len(load()) + 1),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": args.stage, "dim": args.dim, "eu_id": args.eu,
        "hypothesis": args.hypothesis, "diff": args.diff,
        "delta_metric": args.delta_metric, "delta_cost": args.delta_cost,
        "accepted": bool(args.accepted), "reject_reason": args.reject_reason or "",
        "commit": args.commit or "", "evidence": args.evidence or "",
        "why_not_lower_lever": args.why_not_lower_lever or "",
        "predicted_affected": args.predicted_affected or "",
        "retroactive_check": args.retroactive_check or "",
    }
    errs, warns = validate(rec, strict=bool(args.strict))
    for w in warns:
        print("[警告] %s" % w, file=sys.stderr)
    if errs:
        print("拒绝写入，缺 %d 项：" % len(errs), file=sys.stderr)
        for e in errs:
            print("  - %s" % e, file=sys.stderr)
        return 2
    append(rec)
    print("已记账：%s  [%s/%s]  accepted=%s" % (rec["id"], rec["stage"], rec["eu_id"], rec["accepted"]))
    return 0


def cmd_list(args) -> int:
    rows = load()
    if args.stage:
        rows = [r for r in rows if r.get("stage", "").startswith(args.stage)]
    if args.dim:
        rows = [r for r in rows if r.get("dim") == args.dim]
    if not rows:
        print("（无记录）")
        return 0
    print("| id | 环节 | 维 | 单元 | Δ指标 | Δ成本 | 采纳 |")
    print("|---|---|---|---|---:|---:|:--:|")
    for r in rows[-args.limit:]:
        print("| %s | %s | %s | `%s` | %s | %s | %s |" % (
            r.get("id", ""), r.get("stage", ""), r.get("dim", ""), r.get("eu_id", ""),
            "-" if r.get("delta_metric") is None else r["delta_metric"],
            "-" if r.get("delta_cost") is None else r["delta_cost"],
            "✅" if r.get("accepted") else "❌"))
    return 0


def _all_eu_ids() -> set:
    """借 eu_inventory 拿全量 EU（失败则返回空集，不阻塞 stats）。"""
    try:
        sys.path.insert(0, os.path.join(_ROOT, "tools"))
        import eu_inventory as EUI  # noqa: PLC0415
        EUI._load_corpus()
        items = []
        for fn in (EUI.scan_prompts, EUI.scan_flow, EUI.scan_tools, EUI.scan_memory, EUI.scan_context):
            try:
                items.extend(fn())
            except Exception:  # noqa: BLE001
                pass
        return {x["id"] for x in items}
    except Exception:  # noqa: BLE001
        return set()


def cmd_stats(args) -> int:
    rows = load()
    n = len(rows)
    if n == 0:
        print("账本为空。先记第一条：`python tools/edit_ledger.py add --help`")
        return 0
    acc = [r for r in rows if r.get("accepted")]
    print("# 编辑账本统计（%d 条）\n" % n)
    print("| 指标 | 数值 |")
    print("|---|---:|")
    print("| 接受率 | %d/%d = %.1f%% |" % (len(acc), n, len(acc) / n * 100))
    print("| 接受率参考区间（S11） | 10%–15% 属正常 |")
    rrsi3 = ("why_not_lower_lever", "predicted_affected", "retroactive_check")
    full = sum(1 for r in rows if all(str(r.get(k) or "").strip() for k in rrsi3))
    print("| RRSI 三字段完整 | %d/%d = %.1f%%（缺则无法证明改动可归因） |"
          % (full, n, full / n * 100))
    print()
    # 分布
    from collections import Counter
    for key, label in (("stage", "环节"), ("dim", "五维")):
        c = Counter(r.get(key) or "(空)" for r in rows)
        print("## 按%s分布\n" % label)
        print("| %s | 条数 |" % label)
        print("|---|---:|")
        for k, v in c.most_common():
            print("| %s | %d |" % (k, v))
        print()
    # 判据②：零正贡献
    zero, seen = [], set()
    for r in rows:
        eu = r.get("eu_id")
        if not eu:
            continue
        d = r.get("delta_metric")
        if d is None:
            continue
        if d <= 0:
            zero.append((eu, d, r.get("id")))
        seen.add(eu)
    print("## 判据② 零正贡献（窗口内 Δ ≤ 0 ⇒ 剪枝候选，R14 关闭不删）\n")
    if zero:
        print("| 单元 | 最大Δ | 记录 |")
        print("|---|---:|---|")
        for eu, d, rid in zero:
            print("| `%s` | %s | %s |" % (eu, d, rid))
    else:
        print("（无）")
    print()
    # 判据③：从未被改过
    allids = _all_eu_ids()
    if allids:
        untried = sorted(allids - seen)
        print("## 判据③ 从未被改过的单元（探索空间，R11 防塌缩）\n")
        print("共 **%d / %d** 个 EU 从未被改动过（%.1f%%）。前 15 个：\n"
              % (len(untried), len(allids), len(untried) / len(allids) * 100))
        for x in untried[:15]:
            print("- `%s`" % x)
        print("\n> 完整清单：`python tools/eu_inventory.py --json out.json` 后与本账本取差集。")
    else:
        print("## 判据③ 未探索单元\n\n（EU 清单加载失败，跳过）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="编辑账本（R13 七字段）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="追加一次改动")
    a.add_argument("--stage", required=True)
    a.add_argument("--dim", required=True, choices=DIMS)
    a.add_argument("--eu", required=True, help="单元 ID")
    a.add_argument("--hypothesis", required=True, help="一句话：改了什么行为、为何该有效（R15）")
    a.add_argument("--diff", required=True, help="实际改动")
    a.add_argument("--delta-metric", type=float, default=None, help="ΔS（配对差值）")
    a.add_argument("--delta-cost", type=float, default=None, help="ΔC（相对成本变化）")
    a.add_argument("--accepted", action="store_true")
    a.add_argument("--reject-reason", default="")
    a.add_argument("--commit", default="")
    a.add_argument("--evidence", default="")
    a.add_argument("--why-not-lower-lever", default="",
                   help="为什么不用更省力的改法（防提示词塌缩 R11）")
    a.add_argument("--predicted-affected", default="",
                   help="事前预测会影响哪些题（事后可核）")
    a.add_argument("--retroactive-check", default="",
                   help="怎么验证它真有效")
    a.add_argument("--strict", action="store_true",
                   help="三字段必填（正式流程用；历史记录无此三字段）")
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="列出记录")
    l.add_argument("--stage", default="")
    l.add_argument("--dim", default="")
    l.add_argument("--limit", type=int, default=30)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("stats", help="统计（三条判据）")
    s.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
