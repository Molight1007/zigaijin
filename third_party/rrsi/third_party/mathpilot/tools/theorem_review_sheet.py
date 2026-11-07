# -*- coding: utf-8 -*-
"""定理标注表 → 人工审核清单（Markdown）。

用途
----
标注表 `data/theorem_ground_truth/official112_theorems.jsonl` 目前是
**纯自动生成**（`source` 全为 `rebuilt:leansearch-nouniversal`），
它与被评测的检索**同源** ⇒ 命中率恒为 100%，无判别力。

要让「命中率」变成可信指标，必须由人（老师/同学）逐题确认：
    · 哪些定理**确实是**解这道题会用到的  → 保留在 required
    · 哪些**明显无关**                    → 划掉
    · 有没有**关键定理漏了**              → 手工补

本脚本把标注表渲染成**便于人工过审**的 Markdown：
每题给出题干、领域、标准答案，以及**按跨度升序**排列的定理清单
（跨度小 = 题目特异 = 更可能是真需要的；跨度大 = 越可能是泛化噪声）。

用法
----
    python tools/theorem_review_sheet.py --limit 10        # 先看前 10 题
    python tools/theorem_review_sheet.py                   # 全量 112 题
    python tools/theorem_review_sheet.py --out review.md   # 指定输出
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_GT = os.path.join(_SRC, "data", "theorem_ground_truth",
                          "official112_theorems.jsonl")
DEFAULT_BANK = os.path.join(
    _SRC, "题库", "official112_本地测试题库", "official112_full.jsonl")
DEFAULT_OUT = os.path.join(_SRC, "data", "theorem_ground_truth",
                           "review_sheet.md")

# 跨度分档：用于给审核者一个快速判断线索
SPAN_TIERS = (
    (2, "★ 题目特异", "跨 ≤2 题，几乎可以肯定与本题强相关"),
    (10, "  领域相关", "跨 3-10 题，同领域题可能共用，需看题意"),
    (10 ** 9, "⚠ 偏泛化", "跨 >10 题，优先怀疑是关键词泛化命中"),
)


def _load_jsonl_theorem_review_sheet(path: str) -> list:
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


def tier_of(span: int) -> str:
    for thr, tag, _ in SPAN_TIERS:
        if span <= thr:
            return tag
    return "  未知"


def short_of(full: str) -> str:
    """Mathlib 全名 -> **可区分**的短名。

    只取末段会产生歧义：`Mathlib.Data.Finite.Perm.Nat.card_perm` 与
    `Mathlib.Data.Fintype.Perm.Fintype.card_perm` 的末段都是 `card_perm`，
    在清单里显示成两行一模一样的名字，审核者无法判断该删哪个。
    ⇒ 取末两段（如 `Nat.card_perm` / `Fintype.card_perm`），仍冲突则取末三段。
    """
    parts = [p for p in (full or "").split(".") if p]
    if len(parts) <= 3:
        return ".".join(parts)
    return ".".join(parts[-4:]) if len(parts) >= 4 else ".".join(parts[-2:])


def disambiguate(names: list, span: dict) -> list:
    """在**同一题内部**消除显示名冲突：冲突的显示名逐步加长前缀。

    返回 [(显示名, 全名)]，保持输入顺序。
    """
    if not names:
        return []
    out = []
    for full in names:
        parts = [p for p in (full or "").split(".") if p]
        disp = short_of(full)
        # 与已知显示名冲突则加长
        k = 4
        while any(disp == d for d, _ in out) and k <= len(parts):
            disp = ".".join(parts[-k:])
            k += 1
        out.append((disp, full))
    return out


def build_theorem_review_sheet(gt_rows: list, bank: dict, limit: int = 0) -> list:
    """纯函数：产出 Markdown 行列表。"""
    L: list = []
    L.append("# 定理标注表 —— 人工审核清单\n")
    L.append("## 怎么用这份清单\n")
    L.append("本表是**自动生成**的「每题应当用到的 Mathlib 定理」候选，")
    L.append("尚未经人工校正。审核时请对每条定理做三选一：\n")
    L.append("| 判断 | 怎么做 | 含义 |")
    L.append("|---|---|---|")
    L.append("| ✓ 保留 | 不动 | 确认解这道题确实会用到 |")
    L.append("| ✗ 删除 | 从 `required` 移除 | 与本题无关（多为关键词泛化命中） |")
    L.append("| ＋ 补充 | 手工加进 `required` | 应有但检索漏了的关键定理 |")
    L.append("\n**判断线索**：跨度（这道定理在 112 题里被多少题命中）——")
    L.append("跨度越小越可能是本题特有；越大越可能是通用噪声。")
    L.append("表内已按跨度**升序**排列，最可疑的排在最后。\n")
    L.append(f"- 待审题数：{len(gt_rows) if not limit else min(limit, len(gt_rows))}")
    L.append(f"- 数据来源：`{os.path.relpath(DEFAULT_GT, _SRC)}`\n")
    L.append("---\n")

    rows = gt_rows[:limit] if limit else gt_rows
    for i, r in enumerate(rows, 1):
        qid = str(r.get("id") or "")
        b = bank.get(qid) or {}
        q = (b.get("question") or "").strip()
        req = list(r.get("required") or [])
        opt = list(r.get("optional") or [])
        span = r.get("span") or {}

        L.append(f"## {i}. {qid}　`{r.get('domain') or '—'}`\n")
        if b.get("answer") is not None:
            L.append(f"**标准答案**：`{b.get('answer')}`\n")
        L.append("**题干**\n")
        L.append("> " + (q.replace("\n", "\n> ") if q else "（题干缺失）"))
        L.append("")
        L.append(f"**候选定理 {len(req)} 条**（按跨度升序，从上往下越来越可疑）\n")
        L.append("| # | 定理 | 跨度 | 判断线索 |")
        L.append("|---|---|---|---|")
        _sorted = sorted(req, key=lambda x: span.get(x, 0))
        for j, (disp, _full) in enumerate(disambiguate(_sorted, span), 1):
            nm_full = _sorted[j - 1]
            L.append(f"| {j} | `{disp}` | {span.get(nm_full, 0)} 题 | "
                     f"{tier_of(span.get(nm_full, 0))} |")
        if opt:
            L.append("")
            L.append(f"**已自动剔除的通用词 {len(opt)} 条**"
                     f"（不计分，仅列出供参考）\n")
            L.append("| 定理 | 跨度 |")
            L.append("|---|---|")
            _osorted = sorted(opt, key=lambda x: -span.get(x, 0))
            for _disp, _full in disambiguate(_osorted, span):
                L.append(f"| `{_disp}` | {span.get(_full, 0)} 题 |")
        L.append("")
        L.append("**审核结论**：□ 全部合适　□ 有删除　□ 有补充")
        L.append("　删除：________________　补充：________________\n")
        L.append("---\n")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="定理标注表 → 人工审核清单")
    ap.add_argument("--gt", default=DEFAULT_GT, help="标注表 jsonl")
    ap.add_argument("--bank", default=DEFAULT_BANK, help="题库 jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 markdown")
    ap.add_argument("--limit", type=int, default=0, help="只出前 N 题（0=全量）")
    args = ap.parse_args()

    gt_rows = _load_jsonl_theorem_review_sheet(args.gt)
    if not gt_rows:
        print(f"标注表为空: {args.gt}")
        return 1
    bank = {str(x.get("id")): x for x in _load_jsonl_theorem_review_sheet(args.bank)}

    lines = build_theorem_review_sheet(gt_rows, bank, args.limit)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    n = args.limit or len(gt_rows)
    print(f"已写出 {n} 题审核清单 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
