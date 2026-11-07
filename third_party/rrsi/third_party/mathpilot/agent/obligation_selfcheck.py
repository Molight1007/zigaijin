# -*- coding: utf-8 -*-
"""义务账本机械自检：在审核链用它之前，先查账本自身是否可信。

为什么需要
----------
2026-10-23 分阶段探针复现出一起**上游错误**：义务 O5 断言

    (m+n)·D ≤ 777²

代入 gold（D = 777²）得 m+n ≤ 1，而 m<n 是正整数 ⇒ m+n ≥ 3 —— **断言自相矛盾**。
但 4.1 自审按「候选必须满足义务」判，于是把**正确指出该矛盾**的候选淘汰掉
（即用户指出的「把正确的判断为错误的」）。

根源在账本：它把「差的上界 `|x_n²−x_m²| ≤ 777²`」误写成「D 的上界 `(m+n)D ≤ 777²`」。
账本一旦错，下游所有审核都建立错误前提上，且越审越"自洽"。

本模块做什么
------------
只做**能机械判定**的检查，不做语义理解（宁可漏，不可乱判）：

  C1 `target_squeezed`  —— 待求量被「随变量增长的因子」压死
      形态：`(<a>+<b>)·<待求量> ≤ <常数>` 或 `≤ <常数>/(<a>+<b>)`
      判据：该因子随 a、b 增大 ⇒ 待求量上界 → 0。若题目的待求量是**正常数**
            （题面没给"可以为 0/负数"的迹象），这条断言实际在说「待求量 ≤ 0」⇒ 可疑。

  C2 `empty_statement` —— 义务正文为空 / 过短

  C3 `duplicate`       —— 两条义务归一化后完全相同（编号冗余）

输出只标注「可疑」，**不阻断**：调用方把它写进 diag 与错误库，也可作为
「候选反驳该义务时不直接判错」的依据。

用法
----
    from agent.obligation_selfcheck import check_ledger
    suspects = check_ledger(ledger)          # list[dict]
    if suspects: ...                          # 记 diag / 记错误库
"""
from __future__ import annotations

import re

# `(m+n)D ≤ C` / `(a+b)·D \\le C` 这类"求和因子乘待求量"
_SUM_MUL_RE = re.compile(
    r"\(\s*([A-Za-z])\s*\+\s*([A-Za-z])\s*\)\s*[·*]?\s*"
    r"([A-Za-z])\s*(?:≤|\\le|<=|<)")
# `D ≤ C/(m+n)` 这类"待求量被求和因子除"
_DIV_SQ_RE = re.compile(
    r"([A-Za-z])\s*(?:≤|\\le|<=|<)\s*[^,;]{0,20}?/\s*\(\s*"
    r"([A-Za-z])\s*\+\s*([A-Za-z])\s*\)")
_LEQ_RE = re.compile(r"≤|\\le|<=|<")


def _is_probably_positive_target(statement: str) -> bool:
    """题面没明说待求量可 <=0 时，按"正常数"处理（宽松：只排除明显负数语境）。"""
    return not any(w in statement for w in ("≤ 0", "<= 0", "< 0", "负数", "可为负"))


def check_ledger(ledger: dict | None) -> list[dict]:
    """返回可疑义务清单；账本为空/异常时返回空列表（不阻断主链）。"""
    out: list[dict] = []
    if not isinstance(ledger, dict):
        return out
    obls = ledger.get("obligations") or []
    seen_norm: dict[str, str] = {}
    for o in obls:
        if not isinstance(o, dict):
            continue
        oid = str(o.get("oid") or "")
        stmt = str(o.get("statement") or "").strip()

        # C2 空断言
        if len(stmt) < 6:
            out.append({"oid": oid, "kind": "empty_statement",
                        "statement": stmt, "why": "义务正文为空或过短"})
            continue

        # C3 重复
        norm = re.sub(r"\s+", "", stmt.lower())
        if norm in seen_norm:
            out.append({"oid": oid, "kind": "duplicate",
                        "statement": stmt[:120],
                        "why": "与 %s 完全一致" % seen_norm[norm]})
            continue
        seen_norm[norm] = oid

        # C1 待求量被求和因子压死
        # ⚠ 两个正则的**分组语义不同**：乘法式是 (a,b,target)，除式是 (target,a,b)
        target = a = b = ""
        m_mul = _SUM_MUL_RE.search(stmt)
        if m_mul:
            a, b, target = m_mul.group(1), m_mul.group(2), m_mul.group(3)
        else:
            m_div = _DIV_SQ_RE.search(stmt)
            if m_div:
                target, a, b = m_div.group(1), m_div.group(2), m_div.group(3)
        if target and _LEQ_RE.search(stmt) and _is_probably_positive_target(stmt):
            out.append({
                "oid": oid, "kind": "target_squeezed",
                "statement": stmt[:160],
                "why": ("待求量 %s 与求和因子 (%s+%s) %s后不超过常数 ⇒ "
                        "%s 的上界随 (%s+%s) 增大趋 0；若 %s 应为正常数，"
                        "该断言实际等价于 %s ≤ 0"
                        % (target, a, b,
                           "相乘" if m_mul else "相除",
                           target, a, b, target, target)),
            })
    return out


def summarize(suspects: list[dict]) -> str:
    """一行摘要，便于写进 diag / 日志。"""
    if not suspects:
        return ""
    by_kind: dict[str, int] = {}
    for s in suspects:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    oids = ",".join(s["oid"] for s in suspects[:6])
    return "义务自检：%d 条可疑（%s）%s" % (
        len(suspects),
        " ".join("%s=%d" % (k, v) for k, v in sorted(by_kind.items())),
        ("，含 " + oids) if oids else "")
