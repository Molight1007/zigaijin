# -*- coding: utf-8 -*-
"""标注表重建（2026-09-29 第二轮，回应「每题才 2-4 个定理是不是太少了」）。

## 结论：真正的病根是「截断」，不是「泛化」

用户一眼看穿的表象是「每题才 2-4 条定理，太少了」。追下去发现两层原因：

1. **第一轮我用错了药**：`theorem_gt_dedup.py` 拿「横跨 >5 题」当泛化判据，
   一刀切降级 —— 24 题被削到只剩 1 条 required，基准失去区分度。
2. **更根本的是截断**：`build_theorem_gt.py` 的 `req = hits[:top_n]`，
   `top_n` 默认 4。而**检索实际命中 5-10 条**（题干越长命中越多：
   72 字的题命中 5 条，1312 字的题命中 10 条）。基准被硬砍成 4 条，
   与其说是"必须用到的定理"，不如说是"前 4 条"。

## 那 `used_theorems` 能不能当独立第二信号？—— 不能，实测否决

我本轮先尝试过这条路，**被自己的历史代码打脸**。
`agent/orchestrator.py:2838-2839` 早就写明：

    实测 `used_theorems` 的条数恒等于 `search_hits` 恒等于 `top_k`
    ⇒ **该字段实为"检索结果"，不是"实际使用"**（字段名误导）。

复核确认：`results/.../run_2026-09-23c_0924_0002.jsonl` 里，
`used_theorems` **恒为 10 条**（= 那次 top_k），而 `adoption_rate` 多为 0.0。
即它**不是**独立信号，拿它做基准等于再犯一次同源错误。已放弃。

同理，`trace` 里的 `lean_gate` 只记「通过 N/8 候选」，**不含定理名**，也无法用作来源。

## 本脚本最终做法：**不截断 + 只剔通用词 + 分层保留**

    required         = 检索命中中剔除通用词后的**全部**有效条目
    optional         = 通用词（card / permsOfFinset 之类，仅作线索）
    evidence         = 每条命中的来源 query 与位次，供人工校正

即：**恢复检索的真实召回宽度**（5-10 条），只把实测横跨 >30% 题目的
通用词剥出去。这既回应了"太少"，又不引入新的同源污染。

同时**记录 `span`（该定理在 112 题里被多少题命中）**，
让人工校正时能一眼分辨「题目特异定理」与「领域通用定理」。

## 用法

    python tools/rebuild_theorem_gt.py --dry-run      # 只看统计，不写文件
    python tools/rebuild_theorem_gt.py                # 执行（自动 .bak2）
    python tools/rebuild_theorem_gt.py --universal-span-ratio 0.20   # 更激进地剔通用词
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_GT = os.path.join(_SRC, "data", "theorem_ground_truth",
                          "official112_theorems.jsonl")
DEFAULT_PROBE = os.path.join(_SRC, "data", "theorem_ground_truth",
                             "probe_official112.json")
# ⚠ 本脚本**不消费历史评测结果**。曾有设想拿 `used_theorems` 当独立的
#   第二信号源做双源交集，实测否决（见文件头"实测否决"一节）：
#   该字段恒等于检索结果本身，用它等于再犯一次同源错误。故不引入。

# 通用词黑名单（按全名末段短名匹配）：Mathlib 里最基础的函数/概念名，
# 任何题都可能被关键词匹配到。判据以**实测跨度**为主，本表只兜底
# 那些因检索参数差异没被 span 抓到、但显然是通用的名字。
UNIVERSAL = {
    "card", "card_perm", "card_finset_len", "permsOfFinset",
    "card_perms_of_finset", "card_sym_fin_eq_multichoose", "fintypePerm",
    "instInhabited", "map", "state",
}

# ★ 非定理噪音：Lean **实现层**的声明，不是数学定理。
#   实测案例（official112-002，题干含 "choose four distinct numbers"）：
#     `Tactic.Choose.ChooseArg.name`、`Mathlib.Tactic.Choose.elabChoose`
#     —— 这是 `choose` 战术的**实现代码**，与"选四个数"毫无数学关系。
#   检索是纯关键词匹配 ⇒ 题干里有哪个英文词，就可能匹到同名战术/工具声明。
#   这些必须剔除：它们既不能被"用于证明"，也不该出现在定理基准里。
#   判据用**路径片段**（不是末段短名），因为实现层声明散落在这些命名空间下。
NON_THEOREM_MARKERS = (
    ".Tactic.",
    "Tactic.",
    ".Meta.",
    ".Elab.",
    ".Simp.",
    ".NormNum.",
    ".FieldSimp.",
    ".NormCast.",
    ".LibraryNote",
    ".MinImports",
    ".Syntax.",
    ".Parser.",
    "Tactic.Choose",
    ".Basic.inst",
)


def is_non_theorem(full: str) -> bool:
    """该声明是否为 Lean 实现层（战术/元编程/语法），而非数学定理。"""
    s = full or ""
    return any(k in s for k in NON_THEOREM_MARKERS)

# ★ 不再截断 required。实测检索命中 5-10 条（随题干长度分层），
#   硬砍到 4 条正是"每题才 2-4 个定理"的直接原因。保留全部有效命中。


def _load_jsonl_rebuild_theorem_gt(path: str) -> list:
    if not os.path.isfile(path):
        return []
    out = []
    for l in io.open(path, encoding="utf-8", errors="replace"):
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(json.loads(s))
        except Exception:  # noqa: BLE001
            continue
    return out


def c_span(name: str, st: dict) -> int:
    """从 stats 里取某条命中的跨度（仅用于打印，取不到返回 0）。"""
    return int((st.get("span_full") or {}).get(name, 0))


def last_segment(full: str) -> str:
    """Mathlib 全名 -> **末段**短名。仅供黑名单匹配使用。

    ⚠ 不要用它做展示或去重：`...Nat.card_perm` 与 `...Fintype.card_perm`
      的末段都是 `card_perm`，显示出来无法区分。
      需要展示时走 `tools/theorem_review_sheet.py` 的 `disambiguate()`。

    本脚本里它只有一个用途：把 `UNIVERSAL` 黑名单（写的是末段短名）
    套到全名上，判断"这条命中是不是 card 这类通用词"。
    """
    return (full or "").split(".")[-1]


def analyze(rows: list, probe: dict, universal_span_ratio: float) -> dict:
    """纯函数：按「不截断 + 只剔通用词」重建每题 required。

    返回 {"rows_out": [...], "stats": {...}}

    设计要点（为什么是这个形状）
    ---------------------------
    1. **不截断**：检索命中几条就保留几条。实测 5-10 条，硬砍到 4 条
       是"每题才 2-4 个定理"的直接原因。
    2. **只剔通用词**：判据是**实测跨度**（出现在 >X% 题目里），
       外加一份人工黑名单（card 之类 Mathlib 最基础的名字）。
       不做"横跨>5题就降级"那种一刀切 —— 那会误伤领域定理。
    3. **记录 span**：每条命中在 112 题里被多少题命中，供人工判断
       "这是题目特异定理还是领域通用定理"。
    """
    n_q = len(rows)
    max_span = max(1, int(n_q * universal_span_ratio))

    # ---- 1) 从 probe 拿「检索命中」（评测侧的真实返回） ----
    hits_by_q: dict = {}
    for pid, p in probe.items():
        hits_by_q[str(pid)] = [str(h) for h in (p.get("hits") or [])]

    # ---- 2) 全局跨度：识别通用词（按**全名**统计） ----
    span = Counter()
    for qid, hs in hits_by_q.items():
        for h in set(hs):
            span[h] += 1
    auto_universal = {h for h, c in span.items() if c > max_span}
    # 黑名单按末段短名匹配（card 会以各种全名形式出现）
    explicit_hit = {h for h in span if last_segment(h) in UNIVERSAL}
    universal = explicit_hit | auto_universal

    # ---- 3) 逐题重建（不再截断） ----
    out_rows: list = []
    n_non_theorem = 0
    for r in rows:
        qid = str(r.get("id") or "")
        hs = hits_by_q.get(qid) or []

        req, opt = [], []
        for nm in hs:
            if is_non_theorem(nm):
                n_non_theorem += 1
                continue          # 实现层声明：直接从两侧都丢掉
            (opt if nm in universal else req).append(nm)

        out_rows.append({
            "id": qid,
            "domain": r.get("domain") or "",
            "required": req,
            "optional": opt,
            "span": {nm: span.get(nm, 0) for nm in hs},
            "source": "rebuilt:leansearch-nouniversal",
            "evidence": {
                "n_retrieved": len(hs),
                "n_required": len(req),
                "n_universal": len(opt),
                "n_dropped_non_theorem": len(hs) - len(req) - len(opt),
            },
            "note": "",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    stats = {
        "n_rows": n_q,
        "universal_span_ratio": universal_span_ratio,
        "max_span": max_span,
        "n_universal": len(universal),
        "universal_explicit": sorted(explicit_hit, key=lambda x: -span[x]),
        "universal_auto": sorted(auto_universal - explicit_hit,
                                 key=lambda x: -span[x]),
        "span_top": span.most_common(12),
        "span_full": dict(span),
        "n_required": sum(len(x["required"]) for x in out_rows),
        "n_opt": sum(len(x["optional"]) for x in out_rows),
        "n_dropped_non_theorem": n_non_theorem,
        "n_q_with_required": sum(1 for x in out_rows if x["required"]),
        "n_q_without_required": sum(1 for x in out_rows if not x["required"]),
    }
    return {"rows_out": out_rows, "stats": stats}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="标注表重建（不截断 + 只剔通用词）")
    ap.add_argument("--gt", default=DEFAULT_GT, help="标注表 jsonl")
    ap.add_argument("--probe", default=DEFAULT_PROBE, help="probe 结果 json")
    ap.add_argument("--out", default="", help="输出路径（默认覆盖 --gt）")
    ap.add_argument("--universal-span-ratio", type=float, default=0.30,
                    help="出现在超过此比例题目里的检索命中视为通用词（默认 0.30）")
    ap.add_argument("--dry-run", action="store_true", help="只看统计不写文件")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak2")
    args = ap.parse_args()

    rows = _load_jsonl_rebuild_theorem_gt(args.gt)
    if not rows:
        print(f"标注表为空: {args.gt}")
        return 1
    if not os.path.isfile(args.probe):
        print(f"probe 结果不存在: {args.probe}")
        print("请先运行: python cli/theorem_probe.py --resume")
        return 1
    with io.open(args.probe, encoding="utf-8") as f:
        probe = (json.load(f).get("per_question") or {})
    probe_map = {str(x.get("id")): x for x in probe}

    print(f"标注表 {len(rows)} 题 | probe {len(probe_map)} 题")

    res = analyze(rows, probe_map, args.universal_span_ratio)
    st = res["stats"]
    out_rows = res["rows_out"]


    print("\n" + "=" * 66)
    print("重建统计")
    print("=" * 66)
    print(f"  通用词阈值      >{args.universal_span_ratio:.0%} 题 "
          f"(={st['max_span']} 题) 即判为通用")
    print(f"  识别出通用词    {st['n_universal']} 条")
    for nm in st["universal_explicit"][:10]:
        print(f"      [黑名单] {c_span(nm, st):3d} 题  {nm}")
    for nm in st["universal_auto"][:10]:
        print(f"      [实测]   {c_span(nm, st):3d} 题  {nm}")
    print(f"\n  required 总数   {st['n_required']} 条"
          f"（{st['n_q_with_required']} 题有，{st['n_q_without_required']} 题无）")
    if st["n_q_with_required"]:
        print(f"  required 平均   "
              f"{st['n_required']/st['n_q_with_required']:.2f} 条/题")
    print(f"  optional 总数   {st['n_opt']} 条（通用词，仅作线索）")
    print(f"  剔除实现层声明  {st['n_dropped_non_theorem']} 条"
          f"（Tactic/Meta/Syntax 等，非数学定理）")
    print(f"  合计平均        "
          f"{(st['n_required']+st['n_opt'])/st['n_rows']:.2f} 条/题")

    print("\n  检索命中跨度 Top12（前几个就是泛化噪声）：")
    for nm, c in st["span_top"]:
        tag = "通用" if nm in set(st["universal_auto"]) | set(st["universal_explicit"]) else "    "
        print(f"    {tag} {c:3d} 题  {nm}")

    if args.dry_run:
        print("\n[dry-run] 未写文件。去掉 --dry-run 即执行。")
        return 0

    out_path = args.out or args.gt
    if not args.no_backup and os.path.isfile(out_path):
        bak = out_path + ".bak2"
        with io.open(bak, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n已备份 → {bak}")

    header = [
        "# MathPilot 题库定理标注表（重建版 2026-09-29）",
        "# 每行一题：{id, domain, required, optional, span, evidence, source, note}",
        "# required  = LeanSearch 检索命中，剔除通用词后的**全部**条目（不截断）",
        "#             这是「本题应当用到的 Mathlib 定理」候选，人工校正后作评测基准",
        "# optional  = 实测跨 >30% 题目的通用词（card 之类），仅作线索、不计分",
        "# span      = 该定理在 112 题里被多少题命中（越小 = 越题目特异，越可信）",
        "# ★ 人工校正指引：优先保留 span 小（题目特异）的条目；",
        "#   若某条明显与题意无关，直接删；若缺了关键定理，手动补上。",
        "# ⚠ 本项目定理基准**尚未人工校正**（source 全为 auto），命中率仅作自洽性参考。",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        for h in header:
            f.write(h + "\n")
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写出 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
