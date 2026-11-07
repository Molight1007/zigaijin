# -*- coding: utf-8 -*-
"""泄漏筛查（RRSI 内化，2026-10-22）。

为什么需要它
============
RRSI（arXiv:2609.24972 §3.3）把「把 evolve 集的答案/题面特征编码进 harness」
称为 **leakage**，并要求在**评测之前**由 critic 读 diff 直接拒掉这类改动——
理由是一旦这类候选拿到虚高的 evolve 分，后面几轮都会被它吸引。

本项目已有一份同构的实证：K11「蓝图/子目标锁死具体值」占错题根因 41/99，
本质就是把某道题的答案写进了通用机制里。以前只能靠事后逐题归因发现，
本工具把它变成**提交前的一次静态扫描**。

判据（四类命中）
----------------
1. ``TASK_ID``  —— 出现具体题号（official112-073 / imo-bench-…-075 / q<8hex>）
2. ``GOLD``     —— 出现某题标准答案的原文（长度 ≥6，避免 "1"/"x" 这类通用值）
3. ``CONST``    —— 出现只属于极少数题目的长常量（≥4 位数字）
4. ``BRANCH``   —— 形如 ``if <题号>`` / ``== "<题号>"`` 的单题分支（惰性机械）

用法
----
    # 扫工作区相对某提交的改动（默认 HEAD，即未提交改动）
    python tools/leakage_scan.py --base HEAD

    # 扫指定文件（提交前自查）
    python tools/leakage_scan.py --files agent/math_check_gate.py

    # CI 用：命中即非零退出
    python tools/leakage_scan.py --base origin/ma --strict

只依赖标准库（tools 层不得反向依赖 agent）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "题库", "official112_本地测试题库", "official112_full.jsonl")

# 题号形态：覆盖本项目用过的全部命名
_TASK_ID_RES = (
    (re.compile(r"official112-\d{3}"), "official112 题号"),
    (re.compile(r"imo-bench-[A-Za-z_]+-\d{3}"), "imo-bench 题号"),
    (re.compile(r"number_theory-\d{3}"), "number_theory 题号"),
    (re.compile(r"\bq[0-9a-f]{8}\b"), "题号哈希"),
)

# 单题分支：if/elif 条件里直接写题号
_BRANCH_RE = re.compile(
    r"(?:if|elif)\b[^\n]{0,80}?(official112-\d{3}|q[0-9a-f]{8}|number_theory-\d{3})")

# 常量：≥4 位数字（年份类如 2026 常见，单独降级为提示）
_CONST_RE = re.compile(r"(?<![\d.])(\d{4,})(?![\d.])")
_BENIGN_CONSTS = {"2025", "2026", "2027", "1000", "1024", "2048", "4096",
                  "8192", "3600", "1200", "1800", "9999"}

MAX_HITS_PER_KIND = 20

# ============================================================
# 分级扫描（2026-10-24 审计层 P2）
# ------------------------------------------------------------
# 旧 `scan()` 只看文本，无法区分「机制里写死某题」与「注释/测试样本/台账里
# 提到某题」—— 两者都判 LEAK，结果是告警疲劳。分级把命中拆成四档：
#
#   LEAK     主链代码行（会进 prompt / 参与分支）  ← 唯一需要阻塞的
#   COMMENT  注释行或 docstring（运行时不读，但会污染后续改代码的人与 critic）
#   SAMPLE   tests/ 下的样本库（单测需要 gold 才能断言，属合法）
#   INFO     results/ docs/ 台账与报告（本来就是记录）
#
# 排序：路径优先于行内位置 —— tests/ 里的代码行仍是 SAMPLE。
# ============================================================
_PATHS_SAMPLE = ("tests/", "test/")
_PATHS_INFO = ("results/", "docs/", "output/", "测试结果/", "题库备份/")
_GRADE_ORDER = ("LEAK", "COMMENT", "SAMPLE", "INFO")


def _norm_path(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def _path_grade(path: str) -> str:
    p = _norm_path(path)
    if p.startswith(_PATHS_SAMPLE) or "/tests/" in p:
        return "SAMPLE"
    for pre in _PATHS_INFO:
        if p.startswith(pre):
            return "INFO"
    return ""      # 主链路径：交给行内判定


def _docstring_lines(path: str, rev: str = None) -> set:
    """返回该 .py 文件里属于 **docstring** 的行号集合（1-based）。

    只认模块/类/函数的首个 Expr 常量 —— 不能把所有字符串都当安全：
    prompt 模板本身就是字符串，里面写死 gold 才是真泄漏。
    取不到源码（历史提交缺文件、非 py）时返回空集，退化为「按代码行判」。
    """
    try:
        if rev:
            out = subprocess.run(
                ["git", "-c", "core.quotepath=false", "show",
                 "%s:%s" % (rev, path)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30)
            src = out.stdout if out.returncode == 0 else ""
        else:
            abs_p = path if os.path.isabs(path) else os.path.join(ROOT, path)
            with open(abs_p, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        if not src:
            return set()
        import ast
        tree = ast.parse(src)
    except Exception:  # noqa: BLE001
        return set()
    lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            start = first.value.lineno
            end = getattr(first.value, "end_lineno", start) or start
            lines.update(range(start, end + 1))
    return lines


def _line_grade(path: str, text: str, col: int, doc_lines: set, lineno: int) -> str:
    """单行单处命中的分级。``col`` 是命中在行内的起始列（-1 表示未知）。"""
    g = _path_grade(path)
    if g:
        return g
    if lineno in doc_lines:
        return "COMMENT"
    # 行内 `#` 之后出现 ⇒ 是注释（不是代码）
    hash_pos = text.find("#")
    if hash_pos >= 0 and (col < 0 or col > hash_pos):
        return "COMMENT"
    return "LEAK"


def _collect_diff_lines(base: str = "HEAD", files: list = None,
                        rev: str = None) -> list:
    """取待检查的新增行：``[(path, lineno, text)]``。

    · ``rev``   给了就扫**单次提交**的 diff（`git show`），用于事后复核；
    · ``files`` 给了就扫整份文件（行号 1..n），用于提交前自查；
    · 否则扫工作区相对 ``base`` 的未提交改动。
    """
    if files:
        out_lines = []
        for p in files:
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, raw in enumerate(fh, 1):
                        out_lines.append((p, i, raw.rstrip("\r\n")))
            except OSError:
                continue
        return out_lines

    if rev:
        cmd = ["git", "-c", "core.quotepath=false", "show",
               "--unified=0", "--format=", rev]
    else:
        cmd = ["git", "-c", "core.quotepath=false", "diff",
               "--unified=0", base, "--", "."]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=180)
    except Exception as e:  # noqa: BLE001
        print("# git 失败: %s" % e, file=sys.stderr)
        return []

    rows = []
    path = ""
    new_no = 0
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("+++ b/"):
            path = ln[6:]
            continue
        if ln.startswith("+++ ") or ln.startswith("--- "):
            continue
        if ln.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,\d+)?", ln)
            new_no = int(m.group(1)) if m else 0
            continue
        if ln.startswith("+"):
            rows.append((path, new_no, ln[1:]))
            new_no += 1
        elif ln.startswith(" "):
            new_no += 1
        # "-" 行与 "\ No newline" 不占新文件行号
    return rows


def scan_graded(rows: list, golds: list, const_freq: Counter,
                rev: str = None) -> dict:
    """按行分级扫描，返回 ``{档位: [命中明细]}``（LEAK/COMMENT/SAMPLE/INFO）。"""
    hits = {g: [] for g in _GRADE_ORDER}
    by_path: dict = {}
    for path, lineno, text in rows:
        by_path.setdefault(path, []).append((lineno, text))

    for path, items in by_path.items():
        doc_lines = _docstring_lines(path, rev) if path.endswith(".py") else set()
        blob = "\n".join(t for _n, t in items)

        # 1) 题号 / 单题分支 / 单题长常量
        found = []
        for rx, label in _TASK_ID_RES:
            for m in rx.finditer(blob):
                found.append((m.group(0), m.start(), "TASK_ID", label))
        for m in _BRANCH_RE.finditer(blob):
            found.append((m.group(1), m.start(), "BRANCH", m.group(0)[:100]))
        for m in _CONST_RE.finditer(blob):
            c = m.group(1)
            if c in _BENIGN_CONSTS or len(c) < 5 or const_freq.get(c, 0) != 1:
                continue
            found.append((c, m.start(), "CONST", "单题常量"))
        # 2) 答案原文（用 find 拿位置，便于判注释）
        for tid, ans in golds:
            pos = blob.find(ans)
            while pos >= 0:
                found.append((ans, pos, "GOLD", tid))
                pos = blob.find(ans, pos + 1)

        # blob 偏移 → (lineno, 行内列)
        offsets, acc = [], 0
        for lineno, text in items:
            offsets.append((acc, lineno, text))
            acc += len(text) + 1

        for match, pos, kind, extra in found:
            lineno, col, cur_text = items[-1][0], -1, items[-1][1]
            for start, ln_no, text in offsets:
                if start <= pos <= start + len(text):
                    lineno, col, cur_text = ln_no, pos - start, text
                    break
            grade = _line_grade(path, cur_text, col, doc_lines, lineno)
            if len(hits[grade]) >= MAX_HITS_PER_KIND:
                continue
            hits[grade].append({
                "kind": kind, "match": str(match)[:80],
                "path": path, "line": lineno,
                "detail": str(extra)[:80],
            })
    return hits


def verdict_graded(hits: dict) -> tuple:
    """分级结论：只有 LEAK 档阻塞；其余档位如实计数但不拦。"""
    n_leak = len(hits.get("LEAK", []))
    n_comment = len(hits.get("COMMENT", []))
    n_sample = len(hits.get("SAMPLE", []))
    n_info = len(hits.get("INFO", []))
    if n_leak:
        return "LEAK", ("命中 %d 处主链代码泄漏（题号/单题分支/答案原文）；"
                        "另有 COMMENT %d / SAMPLE %d / INFO %d"
                        % (n_leak, n_comment, n_sample, n_info))
    if n_comment:
        return "CLEAN", ("无主链代码泄漏；%d 处注释/docstring 提到答案"
                         "（不影响运行，但会污染后续改动判断）" % n_comment)
    return "CLEAN", "无主链代码泄漏（SAMPLE %d / INFO %d）" % (n_sample, n_info)


def _load_bank(path: str = BANK):
    """读题库 → (gold 答案列表, {常量: 覆盖题数})。

    只收**足够长**的 gold（≥6 字符）：``1`` / ``x`` 这类答案在任何代码里都会
    偶然出现，拿它们做泄漏判据只会制造假阳性。
    """
    golds: list = []
    const_freq: Counter = Counter()
    if not os.path.isfile(path):
        return golds, const_freq
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ans = str(d.get("answer") or "").strip()
            # ★ 只收「足够独特」的 gold：含 4 位以上数字，或长到 20 字符以上。
            #   否则 `$2-2m$` / `$\frac{1}{2}$` 这类通用符号式会误报（实测会命中
            #   任何含分数的代码）。
            if re.search(r"\d{4,}", ans) or len(ans) >= 20:
                golds.append((str(d.get("id") or "?"), ans))
            for c in set(_CONST_RE.findall(str(d.get("question") or ""))):
                const_freq[c] += 1
    return golds, const_freq


def _collect_diff(base: str, files: list) -> str:
    """取待检查的文本：git diff（只取新增行，删掉的代码不会造成泄漏）。"""
    if files:
        chunks = []
        for p in files:
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    chunks.append(fh.read())
            except OSError as e:
                chunks.append("# 读取失败 %s: %s" % (p, e))
        return "\n".join(chunks)
    try:
        out = subprocess.run(
            ["git", "diff", "--unified=0", base, "--", "."],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120)
    except Exception as e:  # noqa: BLE001
        return "# git diff 失败: %s" % e
    # 只保留新增行（以 + 开头且不是 +++ 头）
    added = [ln[1:] for ln in (out.stdout or "").splitlines()
             if ln.startswith("+") and not ln.startswith("+++")]
    return "\n".join(added)


def scan(text: str, golds: list, const_freq: Counter) -> dict:
    """对文本做四类扫描，返回 {命中类别: [明细]}。"""
    hits: dict = {"TASK_ID": [], "GOLD": [], "CONST": [], "BRANCH": []}
    if not text.strip():
        return hits

    for rx, label in _TASK_ID_RES:
        for m in rx.finditer(text):
            hits["TASK_ID"].append({"match": m.group(0), "kind": label})

    for m in _BRANCH_RE.finditer(text):
        hits["BRANCH"].append({"match": m.group(1),
                               "context": m.group(0)[:100]})

    norm = text.replace("\r", "")
    for tid, ans in golds:
        if ans in norm:
            hits["GOLD"].append({"task": tid, "answer": ans[:60]})

    for m in _CONST_RE.finditer(text):
        c = m.group(1)
        if c in _BENIGN_CONSTS or len(c) < 5:
            continue
        # 只属于**唯一**一道题的长常量才算「单题特有」。
        # （初期用 ≤2 时误报严重：注释里的示例数字 `3000` 恰好出现在 2 道题里。）
        if const_freq.get(c, 0) == 1:
            hits["CONST"].append({"match": c, "tasks_containing": 1})

    for k in hits:
        hits[k] = hits[k][:MAX_HITS_PER_KIND]
    return hits


def verdict(hits: dict) -> tuple:
    """返回 (等级, 说明)。

    ★ 只有三类**硬泄漏**决定等级：题号 / 单题分支 / 答案原文。
      ``CONST``（单题特有常量）只作提示 —— 实测假阳性高（注释里的示例数字、
      通用配置常量都会命中），让它决定等级会逼人忽略告警。
    """
    hard = len(hits["TASK_ID"]) + len(hits["BRANCH"]) + len(hits["GOLD"])
    if hard:
        return "LEAK", "命中 %d 处硬泄漏（题号/单题分支/答案原文）" % hard
    if hits["CONST"]:
        return "CLEAN", "未发现硬泄漏（%d 处单题常量仅作提示）" % len(hits["CONST"])
    return "CLEAN", "未发现泄漏迹象"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="改动泄漏筛查（题号/答案/单题常量）")
    ap.add_argument("--base", default="HEAD", help="对比的提交（默认 HEAD）")
    ap.add_argument("--files", nargs="*", default=None, help="直接扫这些文件")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--strict", action="store_true", help="LEAK 时非零退出")
    ap.add_argument("--bank", default=BANK, help="题库 jsonl 路径")
    ap.add_argument("--grade", action="store_true",
                    help="按路径/注释分级：LEAK / COMMENT / SAMPLE / INFO"
                         "（只有 LEAK 阻塞；不加此参数时行为与旧版一致）")
    ap.add_argument("--rev", default=None,
                    help="扫单次提交的 diff（git show <rev>），用于事后复核")
    args = ap.parse_args(argv)

    golds, const_freq = _load_bank(args.bank)

    # ---- 分级模式（新增，默认关闭）----
    if args.grade or args.rev:
        rows = _collect_diff_lines(args.base, args.files, args.rev)
        ghits = scan_graded(rows, golds, const_freq, args.rev)
        level, msg = verdict_graded(ghits)
        if args.json:
            print(json.dumps({"level": level, "message": msg,
                              "graded": ghits, "n_rows": len(rows),
                              "n_gold_loaded": len(golds)},
                             ensure_ascii=False, indent=2))
        else:
            print("泄漏筛查（分级）：%s —— %s" % (level, msg))
            print("  扫描新增行 %d 条" % len(rows))
            for g in _GRADE_ORDER:
                items = ghits.get(g) or []
                if not items:
                    continue
                print("  [%s] %d 处" % (g, len(items)))
                for h in items[:6]:
                    print("    - %s:%s  %s  %s  (%s)" % (
                        h["path"], h["line"], h["kind"], h["match"], h["detail"]))
        if args.strict and level == "LEAK":
            return 1
        return 0

    text = _collect_diff(args.base, args.files)
    hits = scan(text, golds, const_freq)
    level, msg = verdict(hits)

    if args.json:
        print(json.dumps({"level": level, "message": msg, "hits": hits,
                          "n_gold_loaded": len(golds)},
                         ensure_ascii=False, indent=2))
    else:
        print("泄漏筛查：%s —— %s" % (level, msg))
        for k in ("TASK_ID", "BRANCH", "GOLD", "CONST"):
            if not hits[k]:
                continue
            print("  [%s] %d 处" % (k, len(hits[k])))
            for h in hits[k][:6]:
                print("    - %s" % json.dumps(h, ensure_ascii=False)[:120])
    if args.strict and level == "LEAK":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
