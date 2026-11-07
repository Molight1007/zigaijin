# -*- coding: utf-8 -*-
"""终答选择的「答案归一 / 等价判定 / 脏答案 / LLM 输出解析」纯函数簇
============================================================

2026-10-08 批 F1：从 agent/final_selector.py **逐字搬出**（模块级纯函数，缩进不变）。
final_selector.py 原位 re-export 同名符号 ⇒ 外部 `from agent.final_selector import
<name>` 与内部裸调用契约逐字不变。

本模块**自洽**：只依赖 stdlib（json/re）+ 函数内惰性 import（answer_oracle /
formatter），**不 import** final_selector.py（无循环）。
"""
from __future__ import annotations

import json
import re
from typing import Optional


# 拒绝/占位答案（与 formatter._REFUSAL_RE 同源窄口径：这些不可能当终答）
_REFUSAL_RE = re.compile(
    r"无法求解|无法解决|不能解决|无法解答|我无法|子目标求解失败|"
    r"生成失败|调用受限|拒绝回答|未给出有效解答|无法作答",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# 工具：归一化 / 脏答案 / 解析
# ----------------------------------------------------------------------
def _norm(a: str) -> str:
    """归一化答案（剥 \\boxed{} 外壳 + 去空白 + 小写）—— 与全项目等价口径同源。"""
    s = str(a or "")
    try:
        from agent.answer_oracle import AnswerOracle
        s = AnswerOracle.strip_wrappers(s)
    except Exception:  # noqa: BLE001
        s = s.strip()
    return re.sub(r"\s+", "", s).lower()


def _items_equiv(a: str, b: str) -> bool:
    """两个**单项**答案是否等价（SymPy 精确优先，失败回退 `_norm`）。

    ★ 2026-10-04 修（D9-a）：第②层去重原先只用 `_norm`（剥壳 + 去空白 + 小写），
    于是 `0.5` 与 `\\frac12`、`1/2` 与 `0.5` 会被当成**两个不同答案**同时输出，
    而判分器按 `,` 拆成多项与 gold 比对 ⇒ **必错**（与 D2 同族）。
    复用全项目唯一的等价入口 `AnswerOracle.answers_equivalent`。
    """
    if _norm(a) == _norm(b):
        return True
    try:
        from agent.answer_oracle import AnswerOracle
        return bool(AnswerOracle.answers_equivalent(a, b))
    except Exception:  # noqa: BLE001
        return False


def _is_bad_answer(a: str) -> bool:
    """空 / 拒绝词 / 结构化非答案 ⇒ 不能当终答候选。"""
    s = str(a or "").strip()
    if not s:
        return True
    if _REFUSAL_RE.search(s):
        return True
    try:
        from agent.formatter import _looks_like_non_answer
        return bool(_looks_like_non_answer(s))
    except Exception:  # noqa: BLE001
        return False

def _parse_multiple(text) -> Optional[bool]:
    """解析步②输出：{'multiple': true/false}。判不出返回 None（⇒ 按"非多答案"继续）。"""
    t = str(text or "")
    m = re.search(r"\{[^{}]*\}", t)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "multiple" in obj:
                v = obj["multiple"]
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    return v.strip().lower() in ("true", "yes", "1", "是", "可能")
        except Exception:  # noqa: BLE001
            pass
    if re.search(r'"multiple"\s*:\s*true', t, re.IGNORECASE):
        return True
    if re.search(r'"multiple"\s*:\s*false', t, re.IGNORECASE):
        return False
    return None


def _parse_compare_pick(text, n: int) -> Optional[int]:
    """解析步④输出：首个有效行里的 `候选 N`。判不出/无法判定 ⇒ None。"""
    for raw in str(text or "").splitlines():
        line = raw.strip().strip("`*` ").strip()
        if not line:
            continue
        m = re.search(r"候选\s*[#：:]?\s*(\d+)", line)
        if m:
            i = int(m.group(1)) - 1
            return i if 0 <= i < n else None
        if re.search(r"无法判定|无法判断|无法确定|不确定|unknown|都可以|均无法",
                     line, re.IGNORECASE):
            return None
        # 只看第一个有效行
        return None
    return None
