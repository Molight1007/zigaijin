# -*- coding: utf-8 -*-
"""定理注入 A/B 实验 —— 回答用户核心问题：「定理对大模型的推理效果如何？」

用户原话
--------
> 我们要判断它是哪个领域的题目，会用到什么定理……判断大模型或 leansearch
> 最后有没有找到正确的定理。以及**定理对大模型的推理效果如何**？

本脚本只做一件事：把「**注入定理线索**（A+）vs **不注入**（A−）」做成
**同题配对**实验，并给出可直接引用的配对统计结论。它**不启动评测**
（评测是长跑、要用户授权），只负责：

  ① `plan`  —— 打印实验设计（题号、两组命令、判据），供人审阅后再执行；
  ② `analyze` —— 读两轮已完成的结果，做**配对分析**（McNemar / 提升量 / 归因）。

为什么必须配对
--------------
本项目的既有教训是「不同轮次的正确率不可直接相减」：题库难度分布不均、
轮次间还有配额故障轮干扰。同题配对把「题目难度」这个最大方差源消掉，
剩下的差异才能归因到注入本身。因此两组**必须用同一份题号清单**。

判据（沿用项目铁律）
--------------------
  · 配对差异 ≥ 5pp **且** net ≥ 3 题，且 McNemar 显著 → 才认为有效；
  · 否则判「无显著差异」，不要据此改动其它环节。

用法
----
  # 1) 打印实验设计（默认官方 112 题，可换清单）
  python tools/theorem_hint_ab.py plan
  python tools/theorem_hint_ab.py plan --ids-file data/xxx.txt

  # 2) 两轮跑完后做配对分析
  python tools/theorem_hint_ab.py analyze \
      --off run_xxx_no/answers.jsonl --on run_yyy_yes/answers.jsonl

★ 本脚本**不会**自己发起评测。发起前必须先获得用户授权（项目铁律：
  「测试开始 / 停止 / 重开前必须先询问用户」）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# 读取评测结果（对多种可能的结果文件名保持宽容，避免脚本因命名变化而失效）
# ---------------------------------------------------------------------------
_CANDIDATE_FILES = [
    "answers.jsonl", "results.jsonl", "eval_results.jsonl",
    "answer_results.jsonl", "predictions.jsonl",
]


def _load_results(path: str) -> dict:
    """读一个评测结果文件 → {题号: 是否答对}。

    宽容策略：字段名在项目里有多种写法（correct / is_correct / score），
    这里按优先级探测；读不到的题**跳过并在报告里点名**（不静默丢）。
    """
    if os.path.isdir(path):
        for name in _CANDIDATE_FILES:
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                path = cand
                break
        else:
            raise SystemExit(
                f"目录 {path} 下找不到结果文件（试过 {_CANDIDATE_FILES}）")
    if not os.path.exists(path):
        raise SystemExit(f"结果文件不存在：{path}")

    out: dict = {}
    skipped: list = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                skipped.append(line[:60])
                continue
            qid = (row.get("id") or row.get("question_id")
                   or row.get("qid") or row.get("problem_id"))
            if qid is None:
                skipped.append("(无 id 字段)")
                continue
            verdict = None
            for key in ("correct", "is_correct", "matched", "reference_matched"):
                if key in row:
                    verdict = bool(row[key])
                    break
            if verdict is None and "score" in row:
                try:
                    verdict = float(row["score"]) >= 0.999
                except Exception:  # noqa: BLE001
                    verdict = None
            if verdict is None:
                skipped.append(f"{qid}(无判分字段)")
                continue
            out[str(qid)] = verdict
    if skipped:
        # ★ 不静默：点名被跳过的行，否则「样本变少」会被误读成「题目变少」
        print(f"[warn] {path}: 跳过 {len(skipped)} 行无法解析："
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    return out


# ---------------------------------------------------------------------------
# 配对统计
# ---------------------------------------------------------------------------
def _mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）p 值。

    b = 仅 B 组对（A−对/B+错 之外的 "off错→on对" 格）
    c = 仅 A 组对
    用二项分布精确计算，样本量小（本项目 n≈112）时比卡方可靠。
    """
    n = b + c
    if n == 0:
        return 1.0
    # 双侧 p = 2 * P(X <= min(b,c)), X ~ Binomial(n, 0.5)
    k = min(b, c)
    tail = sum(_binom_pmf(i, n) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def _binom_pmf(k: int, n: int) -> float:
    from math import comb
    return comb(n, k) * (0.5 ** n)


def analyze(off_path: str, on_path: str, threshold_pp: float = 5.0,
            min_net: int = 3) -> dict:
    off = _load_results(off_path)
    on = _load_results(on_path)

    common = sorted(set(off) & set(on))
    if not common:
        raise SystemExit("两组没有共同题号 —— 无法做配对分析（检查 id 口径）")

    both = {"off_only": [], "on_only": [], "both_ok": [], "both_bad": []}
    for qid in common:
        a, b_ = off[qid], on[qid]
        if a and b_:
            both["both_ok"].append(qid)
        elif (not a) and (not b_):
            both["both_bad"].append(qid)
        elif (not a) and b_:
            both["on_only"].append(qid)   # A−错 → A+对：注入的**正**贡献
        else:
            both["off_only"].append(qid)  # A−对 → A+错：注入的**负**贡献

    n = len(common)
    acc_off = sum(off[q] for q in common) / n
    acc_on = sum(on[q] for q in common) / n
    b, c = len(both["on_only"]), len(both["off_only"])
    net = b - c
    delta_pp = (acc_on - acc_off) * 100.0
    p = _mcnemar_exact(b, c)

    verdict = ("有效（注入带来配对显著提升）"
               if (delta_pp >= threshold_pp and net >= min_net and p < 0.05)
               else ("负向（注入有害，需查原因）"
                     if (delta_pp <= -threshold_pp and -net >= min_net and p < 0.05)
                     else "无显著差异（不得据此改动其它环节）"))

    return {
        "n_paired": n,
        "unpaired": {"off_only_ids": sorted(set(off) - set(on))[:20],
                     "on_only_ids": sorted(set(on) - set(off))[:20]},
        "acc_off": round(acc_off, 4), "acc_on": round(acc_on, 4),
        "delta_pp": round(delta_pp, 2),
        "flip_off2on": {"n": b, "ids": both["on_only"]},
        "flip_on2off": {"n": c, "ids": both["off_only"]},
        "net": net, "mcnemar_p": round(p, 5),
        "both_ok": len(both["both_ok"]), "both_bad": len(both["both_bad"]),
        "verdict": verdict,
    }


def _print_report(r: dict, threshold_pp: float, min_net: int) -> None:
    print("=" * 68)
    print("定理注入 A/B 配对分析（同题配对，消掉题目难度方差）")
    print("=" * 68)
    print(f"配对题数            : {r['n_paired']}")
    print(f"未配对（仅一组有）  : off={len(r['unpaired']['off_only_ids'])} "
          f"on={len(r['unpaired']['on_only_ids'])}")
    print(f"正确率 A−(不注入)   : {r['acc_off']:.2%}")
    print(f"正确率 A+(注入)     : {r['acc_on']:.2%}")
    print(f"提升 Δ              : {r['delta_pp']:+.2f} pp")
    print("-" * 68)
    print(f"A−错→A+对（注入立功）: {r['flip_off2on']['n']} 题")
    if r["flip_off2on"]["ids"]:
        print(f"    {r['flip_off2on']['ids'][:15]}")
    print(f"A−对→A+错（注入致错）: {r['flip_on2off']['n']} 题")
    if r["flip_on2off"]["ids"]:
        print(f"    {r['flip_on2off']['ids'][:15]}")
    print(f"net（净翻转）        : {r['net']:+d}")
    print(f"McNemar 精确 p       : {r['mcnemar_p']}")
    print(f"两轮皆对 / 皆错      : {r['both_ok']} / {r['both_bad']}")
    print("-" * 68)
    print(f"判据                 : Δ ≥ {threshold_pp:.0f}pp 且 net ≥ {min_net} 且 p < 0.05")
    print(f"结论                 : {r['verdict']}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# plan：打印实验设计（不执行评测）
# ---------------------------------------------------------------------------
# 官方 112 题题库的实际落点（2026-10-14 校）：`data/` 目录在本仓并不存在，
# 题库统一在 `题库/` 下。按优先级探测，避免 plan 打出「题数 0」的空设计。
_BANK_CANDIDATES = [
    "题库/official112_本地测试题库/official112_full.jsonl",
    "题库/official112_题库包/official112_full.jsonl",
    "题库/official112_题库包/official112_questions.jsonl",
    "data/official112.jsonl",
]

# `run_eval.py` 的入参是 `--test_file`（**没有** `--problems` 这个旗标）；旧脚本
# 里的 `--problems` 属过期写法，会直接报 unknown argument，故一并改正。
_BANK_PROBLEMS_FLAG = "--test_file"


def _resolve_bank_path() -> str:
    """返回实际存在的官方 112 题题库路径（找不到则回退首个候选名）。"""
    for rel in _BANK_CANDIDATES:
        if os.path.exists(os.path.join(_SRC, rel)):
            return rel
    return _BANK_CANDIDATES[0]


def _default_ids(limit: int = 0) -> list:
    """默认取官方 112 题 id 清单。

    优先读题意标注文件（`data/theorem_ground_truth/*.jsonl`，含定理 ground
    truth）；**该文件不存在时**回退到题库本身的 `id` 字段（只取题号，够做配对
    实验的题号清单）——否则 plan 会打出「题数 0」的空设计。
    """
    gt = os.path.join(_SRC, "data", "theorem_ground_truth",
                      "official112_theorems.jsonl")
    ids: list = []
    if os.path.exists(gt):
        with open(gt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    ids.append(str(json.loads(line).get("id")))
                except Exception:  # noqa: BLE001
                    pass
    if not ids:
        bank = os.path.join(_SRC, _resolve_bank_path())
        if os.path.exists(bank):
            with open(bank, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        _qid = json.loads(line).get("id")
                    except Exception:  # noqa: BLE001
                        continue
                    if _qid is not None:
                        ids.append(str(_qid))
    if limit:
        ids = ids[:limit]
    return ids


def cmd_plan(args) -> None:
    ids = []
    if args.ids_file and os.path.exists(args.ids_file):
        with open(args.ids_file, encoding="utf-8") as f:
            ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not ids:
        ids = _default_ids(args.limit)
    bank = _resolve_bank_path()

    print("=" * 68)
    print("定理注入 A/B 实验设计")
    print("=" * 68)
    print(f"题号来源 : {args.ids_file or bank}")
    print(f"题库文件 : {bank}")
    print(f"题数     : {len(ids)}")
    print(f"题号前 10: {ids[:10]}")
    print()
    print("★ 设计要点")
    print("  · **同题配对**：两组用完全相同的题号清单，只改一个变量；")
    print("  · 只改 `--enable_theorem_hint`，其余参数**逐字相同**（含档位/预算/并发）；")
    print("  · 串行执行（项目铁律：不得并行跑评测），两轮之间不夹带其它改动；")
    print("  · 中间不夹带任何代码改动 —— 否则差异无法归因到注入。")
    print()
    print("★ 两条命令（**需先获得用户授权再执行**）")
    print("-" * 68)
    base = ("D:/python/python.exe run_eval.py "
            "%s %s "
            "--enable_theorem_hint {v}" % (_BANK_PROBLEMS_FLAG, bank))
    print("A−（对照组，不注入定理）:")
    print("   " + base.format(v="false"))
    print("A+（实验组，注入定理线索）:")
    print("   " + base.format(v="true"))
    print("-" * 68)
    print(f"⚠ 两轮必须用**同一**题库文件（当前：{bank}）；")
    print("  若要只跑子集，两轮都加 `--test_file <同一子集文件>`。")
    print(f"   当前题数 {len(ids)} 已由 `_default_ids` 解析，可直接落为子集清单。")
    print()
    print("★ 分析")
    print("   python tools/theorem_hint_ab.py analyze \\")
    print("       --off <A−的输出目录> --on <A+的输出目录>")
    print()
    print("★ 配套的「检索是否找对定理」（不依赖评测，可先跑）")
    print("   D:/python/python.exe cli/theorem_probe.py --report")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="定理注入 A/B：设计打印 + 同题配对分析（不自动发起评测）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan", help="打印实验设计")
    p1.add_argument("--ids-file", default=None, help="题号清单（每行一个）")
    p1.add_argument("--limit", type=int, default=0, help="只取前 N 题（0=全部）")
    p1.set_defaults(func=cmd_plan)

    p2 = sub.add_parser("analyze", help="读两轮结果做配对分析")
    p2.add_argument("--off", required=True, help="A−（不注入）结果文件或目录")
    p2.add_argument("--on", required=True, help="A+（注入）结果文件或目录")
    p2.add_argument("--threshold-pp", type=float, default=5.0)
    p2.add_argument("--min-net", type=int, default=3)
    p2.set_defaults(func=lambda a: _print_report(
        analyze(a.off, a.on, a.threshold_pp, a.min_net),
        a.threshold_pp, a.min_net))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
