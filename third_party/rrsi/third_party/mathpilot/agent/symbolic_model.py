"""L2 独立符号建模复核：解析与求值（2026-09-10）。

纯函数、零 LLM、零副作用，便于单测与复用：
- parse_symbolic_payload：从模型输出里稳健地取出结构化建模结果
- normalize_payload：规整成 (变量映射, 表达式)
- evaluate_payload：交 utils/math_eval 精确代入求值 → (精确值, 说明)

设计原则：解析失败一律返回 None / (None, 原因)，由调用方降级放行，
**绝不因解析问题影响主链**。
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("MathPilot.Symbolic")


def _first_json_object(text: str) -> str | None:
    """抠出文本里第一个平衡的 `{...}` 块（跳过字符串内部的括号与转义）。"""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


_EXPR_RE = re.compile(r'"(?:answer_)?expr"\s*:\s*"([^"]*)"')
_VARS_BLOCK_RE = re.compile(r'"vars"\s*:\s*\{([^{}]*)\}')
_VAR_PAIR_RE = re.compile(
    r'"([A-Za-z][A-Za-z0-9]*)"\s*:\s*"?(-?\d+(?:\.\d+)?(?:/\d+)?)"?')
_VAR_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_BARE_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:/\d+)?$")


def parse_symbolic_payload(text: str) -> dict | None:
    """从模型输出里稳健解析出建模 JSON；失败返回 None。

    三级兜底：①整体 json.loads ②抠第一个平衡 `{}` 再 loads
    ③正则抽 expr / vars。任何异常都吞掉返回 None（不阻断主链）。
    """
    try:
        if not text or not str(text).strip():
            return None
        t = str(text).strip()
        if t.startswith("```"):                       # 去 markdown 围栏
            t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
            t = re.sub(r"\s*```\s*$", "", t)
        for cand in (t, _first_json_object(t)):
            if not cand:
                continue
            try:
                obj = json.loads(cand)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and any(
                    k in obj for k in ("expr", "answer_expr", "vars")):
                return obj
        m = _EXPR_RE.search(t)
        if m:
            vars_: dict = {}
            mb = _VARS_BLOCK_RE.search(t)
            if mb:
                for k, v in _VAR_PAIR_RE.findall(mb.group(1)):
                    vars_[k] = v
            return {"vars": vars_, "expr": m.group(1)}
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("[symbolic] payload 解析异常: %s", str(exc)[:100])
        return None


def normalize_payload(payload) -> tuple[dict | None, str | None]:
    """payload → (变量映射, 表达式)；表达式缺失/为空 → (None, None)。

    变量映射**可为空 dict**（表示纯数值算式、无变量）。变量名只保留
    「字母开头 + 字母数字」的键（长度不限）；数值合法性交给 safe_eval_subst。
    """
    if not isinstance(payload, dict):
        return None, None
    expr = payload.get("expr", payload.get("answer_expr", ""))
    if not isinstance(expr, str) or not expr.strip():
        return None, None
    raw_vars = payload.get("vars")
    mapping: dict = {}
    if isinstance(raw_vars, dict):
        for key, val in raw_vars.items():
            name = str(key).strip()
            if _VAR_NAME_RE.match(name):
                mapping[name] = str(val).strip()
    return mapping, expr.strip()


def _uses_any(expr: str, names) -> bool:
    """表达式是否真正用到 names 里至少一个变量（词边界匹配）。"""
    for name in names:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", expr):
            return True
    return False


def evaluate_payload(payload) -> tuple[str | None, str]:
    """payload → (精确值串, 说明)。求不出精确值时值为 None。

    两条"防偷算"护栏（宁漏勿误：命中即弃权，而不是给出可能错的"真值"）：
      1) expr 是**裸数值**（如 "4"）→ 说明模型已经自己算出了结果，不是关系式；
      2) 声明了变量、但 expr 完全没用到任何一个 → 同上（它已把数值代进去了）。
    """
    mapping, expr = normalize_payload(payload)
    if expr is None:
        return None, "无可建模的表达式"
    if _BARE_NUMBER_RE.match(expr):
        return None, f"表达式是裸数值 {expr!r}（疑似已自行计算），弃权"
    if mapping and not _uses_any(expr, mapping):
        return None, f"表达式 {expr!r} 未使用所声明的变量，疑似已自行代入，弃权"
    try:
        from utils.math_eval import safe_eval_subst, to_exact_number
    except ImportError:  # 模块不可用时降级（绝不阻断）
        return None, "math_eval 不可用"
    got = safe_eval_subst(expr, mapping)
    if got.startswith(("WARN:", "ERROR:")):
        return None, f"代入求值未成功: {got[:80]}"
    if to_exact_number(got) is None:
        return None, f"代入结果非精确数值: {got[:60]}"
    return got, f"{expr} 代入 {mapping} → {got}"
