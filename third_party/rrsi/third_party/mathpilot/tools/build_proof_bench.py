# -*- coding: utf-8 -*-
"""从本地题库重建 LEAP 证明题基准 CSV（lean_proof_bench.csv）。

背景：superhuman/imobench/lean_proof_bench.csv 已随 superhuman/ 目录丢失，
且该文件从未进过 git（git 里只有 answerbench.csv / gradingbench.csv），
重新克隆整个 superhuman 仓库只为拿一个 csv 不划算。
本项目从 题库/question_bank.db 的 IMO-ProofBench（60 题）重建，列结构对齐
leap_eval.load_bench() 的读取口径：
    Problem ID / Problem / Lean Statement / Category / Level

已知差异（重要）：
  题库里只有**自然语言**参考答案，没有 Lean 形式化陈述，
  因此 Lean Statement 列留空。leap_eval.py:266 对该列是"若有才用作形式化基线"，
  留空不影响三阶段流程，只是少了一个可选的先验。

用法：
    python tools/build_proof_bench.py
    python tools/build_proof_bench.py --bank IMO-ProofBench --out superhuman/imobench/lean_proof_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(_ROOT, "题库", "question_bank.db")
DEFAULT_OUT = os.path.join(_ROOT, "superhuman", "imobench", "lean_proof_bench.csv")

FIELDS = ["Problem ID", "Problem", "Lean Statement", "Category", "Level"]


def _level_of(pid: str) -> str:
    """从题号推断难度档：PB-Basic-* → Basic，PB-Advanced-* → Advanced。"""
    low = pid.lower()
    if "advanced" in low:
        return "Advanced"
    if "basic" in low:
        return "Basic"
    return "Unknown"


def build(db_path: str, bank: str) -> list[dict]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"题库不存在: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT problem_id, question, domain, reference_answer "
            "FROM problems WHERE bank_name = ? ORDER BY problem_id",
            (bank,),
        )
        rows = []
        for r in cur.fetchall():
            pid = (r["problem_id"] or "").strip()
            q = (r["question"] or "").strip()
            if not pid or not q:
                continue
            rows.append({
                "Problem ID": pid,
                "Problem": q,
                # 题库无 Lean 形式化陈述，留空（leap_eval 中该项为可选基线）
                "Lean Statement": "",
                "Category": (r["domain"] or "").strip(),
                "Level": _level_of(pid),
            })
        return rows
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 LEAP 证明题基准 CSV")
    ap.add_argument("--db", default=DEFAULT_DB, help="题库 sqlite 路径")
    ap.add_argument("--bank", default="IMO-ProofBench", help="题库名称")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 CSV 路径")
    args = ap.parse_args()

    try:
        rows = build(args.db, args.bank)
    except (FileNotFoundError, sqlite3.Error) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    if not rows:
        print(f"[error] 题库 {args.bank} 中无有效题目", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # 必须写 utf-8 而非 utf-8-sig：leap_eval.load_bench() 用 encoding="utf-8" 读取，
    # 带 BOM 时首列名会变成 "\ufeffProblem ID"，DictReader 取不到值 → 加载 0 题。
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    levels: dict[str, int] = {}
    cats: dict[str, int] = {}
    for r in rows:
        levels[r["Level"]] = levels.get(r["Level"], 0) + 1
        cats[r["Category"]] = cats.get(r["Category"], 0) + 1
    print(f"written {len(rows)} -> {args.out}")
    print(f"  Level: {levels}")
    print(f"  Category: {cats}")
    print("[note] Lean Statement 列留空：题库无 Lean 形式化陈述，"
          "leap_eval 会自动跳过形式化基线（不影响三阶段流程）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
