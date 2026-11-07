# -*- coding: utf-8 -*-
"""标注表去污（特异性过滤）—— 纯后处理，**不重跑检索**。

背景（2026-09-29 实测暴露的严重设计缺陷）
-----------------------------------------
`build_theorem_gt.py` 最初把「检索结果前 N 条」直接当作 required
（"解本题必须用到的定理"）。但检索是**关键词泛化匹配**：
题干含 `card` / `choose` / `permutation` 就会命中 `card_perms_of_finset` 之类。

实测后果（official112 全量）：

| 定理 | 出现在 required 的题数 |
|---|---|
| `card_perms_of_finset` | **49 / 112** |
| `card_sym_fin_eq_multichoose` | **36 / 112** |
| `permsOfFinset` | **30 / 112** |

三条合计占 448 个 required 槽位的 **26%**。

★ 为什么这是致命缺陷
--------------------
这等于**用检索结果定义标准答案** ⇒ 评测侧与标注侧**同源**，
「命中率 86.6%」变成**循环自证**，不具判别力。
（此前只做了"标注 3 query vs 评测 2 query"的差异，那只防住了
 query 数相同的自证，**防不住同一条定理被两边共同命中**。）

★ 本脚本做什么
--------------
把横跨题数 > 阈值（默认 5）的定理从 `required` **降级到 `optional`**：
  · 不删除 —— 它可能确实是某些题会用到的（保留线索）；
  · 但不再计入"必中"基准 —— 因为它**不区分题目**，谁都能命中。
这样 `hit_rate` 才恢复为**题目特异定理的召回率**，而非泛化匹配率。

阈值是启发式（112 题里横跨 >5 题 ≈ >4.5%）。用 `--max-span` 可调。

用法
----
  # 预览（只报告，不改文件）
  D:/python/python.exe tools/theorem_gt_dedup.py --dry-run

  # 执行（原地改，先自动备份 .bak）
  D:/python/python.exe tools/theorem_gt_dedup.py
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_GT = os.path.join(_SRC, "data", "theorem_ground_truth",
                          "official112_theorems.jsonl")


def _load(path: str) -> tuple:
    """返回 (注释头行列表, 数据行列表)。注释头原样保留。"""
    header, rows = [], []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.startswith("#"):
                header.append(s)
            elif s.strip():
                rows.append(json.loads(s))
    return header, rows


def analyze(rows: list, max_span: int) -> dict:
    """统计跨度并给出降级清单（**纯函数，便于单测**）。"""
    cnt: Counter = Counter()
    for r in rows:
        for nm in set(r.get("required") or []):
            cnt[nm] += 1
    too_common = {nm for nm, c in cnt.items() if c > max_span}
    n_slots = sum(len(r.get("required") or []) for r in rows)
    polluted = sum(c for nm, c in cnt.items() if nm in too_common)
    return {
        "n_rows": len(rows),
        "n_required_names": len(cnt),
        "n_required_slots": n_slots,
        "too_common": sorted(too_common, key=lambda x: -cnt[x]),
        "span_of": {nm: cnt[nm] for nm in too_common},
        "polluted_slots": polluted,
        "polluted_ratio": (polluted / n_slots) if n_slots else 0.0,
    }


def demote(rows: list, max_span: int) -> tuple:
    """执行降级。返回 (被改动的行数, 降级处数, 降级清单)。"""
    info = analyze(rows, max_span)
    too_common = set(info["too_common"])
    changed = moved = 0
    for r in rows:
        req = list(r.get("required") or [])
        kept = [nm for nm in req if nm not in too_common]
        dropped = [nm for nm in req if nm in too_common]
        if not dropped:
            continue
        r["required"] = kept
        opt = list(r.get("optional") or [])
        for nm in dropped:
            if nm not in opt:
                opt.append(nm)
        r["optional"] = opt
        changed += 1
        moved += len(dropped)
    return changed, moved, info


def main() -> int:
    ap = argparse.ArgumentParser(description="标注表特异性过滤（纯后处理）")
    ap.add_argument("--gt", default=DEFAULT_GT, help="标注表 jsonl")
    ap.add_argument("--max-span", type=int, default=5,
                    help="横跨超过 N 题即降级（默认 5）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只报告不写文件")
    ap.add_argument("--no-backup", action="store_true",
                    help="不生成 .bak（默认生成）")
    args = ap.parse_args()

    if not os.path.isfile(args.gt):
        print(f"标注表不存在：{args.gt}")
        return 1

    header, rows = _load(args.gt)
    print(f"读入 {args.gt}")
    print(f"  题数 {len(rows)}")

    changed, moved, info = demote(rows, args.max_span)

    print(f"\n跨度阈值：required 中一条定理横跨 > {args.max_span} 题即降级")
    print(f"  降级定理数    : {len(info['too_common'])}")
    print(f"  受影响题数    : {changed}")
    print(f"  降级处数      : {moved}")
    print(f"  污染槽位占比  : {info['polluted_ratio']:.1%} "
          f"（{info['polluted_slots']}/{info['n_required_slots']}）")
    if info["too_common"]:
        print("\n  降级清单（原 required 跨度）：")
        for nm in info["too_common"]:
            print(f"    {nm:<40s} {info['span_of'][nm]:3d} 题")

    if args.dry_run:
        print("\n[dry-run] 未写文件。去掉 --dry-run 即执行。")
        return 0

    if not args.no_backup:
        bak = args.gt + ".bak"
        shutil.copy2(args.gt, bak)
        print(f"\n已备份 → {bak}")

    with io.open(args.gt, "w", encoding="utf-8") as f:
        for h in header:
            f.write(h + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 复核：降级后跨度分布
    after = Counter()
    for r in rows:
        for nm in set(r.get("required") or []):
            after[nm] += 1
    worst = after.most_common(3)
    print(f"已写出 → {args.gt}")
    print(f"  降级后 required 去重定理数 : {len(after)}")
    print(f"  降级后最大跨度 Top3        : "
          + "、".join(f"{n}({c}题)" for n, c in worst))
    print("\n⚠ 复核提示：跨度最大值应显著下降；若仍 >20 题，"
          "说明阈值偏松或还有别的泛化词，重跑本脚本并调低 --max-span。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
