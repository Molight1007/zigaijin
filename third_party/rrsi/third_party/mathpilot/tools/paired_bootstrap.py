# -*- coding: utf-8 -*-
"""配对显著性 + 噪声带校准（RRSI 内化，2026-10-22）。

为什么需要这个工具
==================
项目原判据是「配对差异 ≥5pp 且 net≥3（+McNemar）」，其中 **5pp 是拍的**。
RRSI（arXiv:2609.24972）明确指出：噪声带 δ 必须**实测**——
「用未变的 harness 反复评测得到的离散度」，并且「δ 是唯一绝不能调的值，
调低它等于用噪声换分数」。

本工具做两件互不替代的事：

1. ``--mode compare``：两个 arm 的逐题正确性 → 净胜题数、任务级配对 bootstrap
   95% CI、McNemar 精确 p 值。**重采样单位是「题目」而非「trial」**——
   同一道题的多次尝试高度相关，按 trial 重采样会低估区间（RRSI §E.2）。

2. ``--mode delta``：**同一个 harness 的两次评测**（未做任何改动）→
   观测差异分布，取高分位作为噪声带 δ。之后判定「提升是否显著」应以
   ``Δ > δ`` 为准，而不是拍一个 5pp。

只依赖标准库（tools 层不得反向依赖 agent）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys


def load_verdicts(path: str) -> dict:
    """读评测 jsonl → {题目 id: 是否正确}。

    ★ 只认 ``id`` + ``correct`` 两个键；缺 `correct` 的行按「未跑」跳过，
    绝不当作「答错」——把「没跑到」算成「失败了」是本项目踩过的坑。
    """
    out: dict = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not isinstance(d, dict) or d.get("id") is None:
                continue
            if "correct" not in d:
                continue
            out[str(d["id"])] = bool(d.get("correct"))
    return out


def compare(a: dict, b: dict) -> dict:
    """逐题配对比较（同一题集，两个 arm）。"""
    common = sorted(set(a) & set(b))
    a_only = [t for t in common if a[t] and not b[t]]
    b_only = [t for t in common if b[t] and not a[t]]
    both = [t for t in common if a[t] and b[t]]
    neither = [t for t in common if not a[t] and not b[t]]
    n = len(common)
    return {
        "n_tasks": n,
        "correct_a": sum(1 for t in common if a[t]),
        "correct_b": sum(1 for t in common if b[t]),
        "a_only": a_only,
        "b_only": b_only,
        "n_both": len(both),
        "n_neither": len(neither),
        # 净胜题数：B 相对 A 多答对的题数（这是小样本集上唯一有意义的计数口径）
        "net": len(b_only) - len(a_only),
        "delta_pp": (100.0 * (len(b_only) - len(a_only)) / n) if n else 0.0,
    }


def paired_bootstrap(a: dict, b: dict, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """任务级配对 bootstrap：重采样**题目**，给出 Δ（B−A）的 95% CI。

    做法（RRSI §E.2）：以**同一批题目**为总体，有放回地重抽 n_tasks 道，
    两个 arm 用同一组重抽结果（配对），算每次的 Δ，取 2.5/97.5 百分位。
    配对的好处是：题目难度的方差在差值里抵消。
    """
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        return {"n_tasks": 0, "ci_low": 0.0, "ci_high": 0.0, "mean": 0.0}
    # 逐题差值（以 pp 为单位）：+100 / -100 / 0
    diffs = []
    for t in common:
        diffs.append((int(b[t]) - int(a[t])) * 100.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(max(1, int(n_boot))):
        s = 0.0
        for _i in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return {
        "n_tasks": n,
        "n_boot": len(stats),
        "mean": sum(stats) / len(stats),
        "ci_low": lo,
        "ci_high": hi,
        "significant": (lo > 0.0 or hi < 0.0),
    }


def mcnemar_exact(a_only: list, b_only: list) -> float:
    """McNemar 精确检验（双侧）p 值。

    H0：两种不一致方向等概率。b = |a_only|，c = |b_only|，
    X ~ Bin(b+c, 0.5)，p = 2·P(X ≤ min(b,c))（上限 1）。
    小样本（<25 个不一致对）用它比卡方近似可靠 —— 我们的题集本来就小。
    """
    b, c = len(a_only), len(b_only)
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m)
    return min(1.0, 2.0 * tail)


def calibrate_delta(x: dict, y: dict) -> dict:
    """**噪声带 δ 校准**：同一个 harness 的两次未改动评测。

    返回逐题观测差异的统计（以 pp 计）。δ 取 ``p95_abs``：
    「不做任何改动时，两次评测的差最多能有多大」。判定提升时应要求
    ``Δ > δ``，否则无法区分真实改进与评测抖动。
    """
    common = sorted(set(x) & set(y))
    n = len(common)
    if n == 0:
        return {"n_tasks": 0, "warning": "两次评测没有共同题目"}
    diffs = [abs(int(y[t]) - int(x[t])) * 100.0 for t in common]
    diffs_sorted = sorted(diffs)
    flips = sum(1 for d in diffs if d > 0)
    mean_abs = sum(diffs) / n
    # ★ 口径说明：逐题结果是二值的，单题「差异」非 0 即 100pp，取 max 没有意义。
    #   有意义的两个量是：
    #     · net_diff_pp —— 整体正确率偏移 |Δcorrect|/n（可能被正负抵消）；
    #     · flip_pp     —— 翻转率 flips/n（总抖动，是净偏移的上界，**更保守**）。
    #   δ 取 flip_pp：它保证「不做改动时也可能出现的差异」不会被当成提升。
    p95 = diffs_sorted[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    net_diff = abs(sum(1 for t in common if y[t]) - sum(1 for t in common if x[t]))
    net_diff_pp = 100.0 * net_diff / n
    flip_pp = 100.0 * flips / n
    return {
        "n_tasks": n,
        "n_flipped_tasks": flips,
        "mean_abs_pp": round(mean_abs, 2),
        "p95_abs_pp": p95,
        "net_diff_pp": round(net_diff_pp, 2),
        "flip_pp": round(flip_pp, 2),
        # 可判定的最小非零差异：1 道题 / n 题
        "resolution_pp": round(100.0 / n, 2),
        "delta_pp": round(max(flip_pp, 100.0 / n), 2),
        "note": ("δ 必须由「未改动的 harness 重复评测」测出，不得调低；"
                 "n 题集的最小分辨率是 100/n pp。本值只覆盖观测到的抖动，"
                 "样本越少越可能低估。"),
    }


def _fmt(res: dict, title: str) -> str:
    lines = ["=== %s ===" % title]
    for k, v in res.items():
        if isinstance(v, list):
            lines.append("  %-16s %d 项 %s" % (k, len(v), v[:8]))
        elif isinstance(v, float):
            lines.append("  %-16s %.2f" % (k, v))
        else:
            lines.append("  %-16s %s" % (k, v))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="配对显著性（compare）与噪声带校准（delta）")
    ap.add_argument("--a", required=True, help="基线 arm 的 jsonl")
    ap.add_argument("--b", required=True, help="对照 arm 的 jsonl（delta 模式=重复评测）")
    ap.add_argument("--mode", choices=("compare", "delta", "both"),
                    default="compare")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args(argv)

    for p in (args.a, args.b):
        if not os.path.isfile(p):
            print(json.dumps({"error": "文件不存在: %s" % p}, ensure_ascii=False))
            return 2
    a, b = load_verdicts(args.a), load_verdicts(args.b)
    out: dict = {}
    if args.mode in ("compare", "both"):
        cmp_res = compare(a, b)
        boot = paired_bootstrap(a, b, args.n_boot, args.seed)
        p = mcnemar_exact(cmp_res["a_only"], cmp_res["b_only"])
        out["compare"] = dict(cmp_res, bootstrap=boot, mcnemar_p=p)
    if args.mode in ("delta", "both"):
        out["delta"] = calibrate_delta(a, b)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if "compare" in out:
        c = out["compare"]
        print(_fmt({k: v for k, v in c.items() if k != "bootstrap"},
                   "配对比较（B 相对 A）"))
        print(_fmt(c["bootstrap"], "任务级配对 bootstrap（%d 次）" % c["bootstrap"].get("n_boot", 0)))
        print("  McNemar 精确 p = %.4f" % c["mcnemar_p"])
    if "delta" in out:
        print(_fmt(out["delta"], "噪声带 δ 校准（同一 harness 两次评测）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
