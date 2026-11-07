# -*- coding: utf-8 -*-
"""按题型过滤测试题库 —— **只保留解答题与证明题**。

★★ 2026-10-07 用户确立口径（原话）：
    「以后测试不要测试选择题和客观题，只测试解答题与证明题。」

## 为什么要过滤（不只是"省事"）
1. **口径统一**：选择题/判断题/填空题走的是完全不同的判分与验证路径
   （选项比对 vs 表达式等价；`AuditGate` 对解答题结构性无输出、`LeanGate` 不适用），
   混在一起会让"环节是否生效"的统计被污染。
2. **实测代价**（run_2026-10-06_smoke10 的 087/088 两道选择题）：
   - 各有 **7 个阶段 <0.01 秒**（多个环节结构性零输出）
   - `6.5_audit_gate` 计时 1.08e-03 / 7.36e-04，`gate_tried=[]`
   - `preverify_trace` 缺失、`lean_gate n=0`、定理检索「题型为选择题，跳过」
   ⇒ 这些"未触发"被计入指标分母，会把解答题上的真实表现**稀释掉**。

## 用法
    python tools/filter_testset.py <输入.jsonl> [-o <输出.jsonl>] [--dry-run]

不传 `-o` 时输出到同目录 `<原名>_subj.jsonl`。
`--dry-run` 只统计不写文件。

## 题型来源
题库 `official112_*.jsonl` 里**没有题型字段**（只有 `domain`＝学科）。
题型由运行时判定并落在结果的 `diag.question_type`。
⇒ 本脚本对**已有题库**用"题面特征"判题型；对**评测结果**优先读 `diag.question_type`。
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

# 排除的题型（用户口径：选择题与客观题不测）
EXCLUDED = ("选择题", "判断题", "填空题", "客观题")
# 保留的题型
KEPT = ("解答题", "证明题")

# —— 无题型字段时，按题面特征推断 ——
# 选择题特征：出现 A/B/C/D 选项块，且题面含"选出/以下/正确的是"
_CHOICE_RE = re.compile(
    r"(?:^|\n)\s*[（(]?\s*[A-D]\s*[）)]\s*[\.、]|"
    r"(?:选项|选择|选出|下列|以下).{0,12}(?:正确|错误|一项|哪)", re.M)
# 证明题特征
_PROOF_RE = re.compile(r"(?:证明|求证|试证|prove)\s*[:：]?")
# 填空题特征
_FILL_RE = re.compile(r"_{3,}|填空|填上")


def infer_type(rec: dict) -> str:
    """优先读运行时题型；否则按题面特征推断。"""
    d = rec.get("diag") or {}
    qt = str(d.get("question_type") or rec.get("question_type") or "").strip()
    if qt:
        return qt
    txt = str(rec.get("question") or rec.get("problem") or "")
    if not txt:
        return "未知"
    if _CHOICE_RE.search(txt):
        return "选择题"
    if _FILL_RE.search(txt):
        return "填空题"
    if _PROOF_RE.search(txt):
        return "证明题"
    return "解答题"      # 默认（本题库绝大多数是解答题）


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        return 2
    src = pathlib.Path(args[0])
    if not src.exists():
        print("找不到输入文件：%s" % src)
        return 2
    dry = "--dry-run" in sys.argv
    out = None
    if "-o" in sys.argv:
        out = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
    elif not dry:
        out = src.with_name(src.stem + "_subj.jsonl")

    rows, kept, dropped = [], [], []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append(rec)
        qt = infer_type(rec)
        (dropped if qt in EXCLUDED else kept).append((rec, qt))

    from collections import Counter
    print("输入：%s（%d 题）" % (src.name, len(rows)))
    print("题型分布：%s" % dict(Counter(qt for _, qt in kept + dropped)))
    print("保留 %d 题 / 排除 %d 题" % (len(kept), len(dropped)))
    if dropped:
        print("  被排除的：")
        for rec, qt in dropped:
            print("    [%s] %s" % (qt, rec.get("id")))
    if dry:
        print("\n(--dry-run，未写文件)")
        return 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec, _ in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n已写出：%s（%d 题，仅解答题/证明题）" % (out, len(kept)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
