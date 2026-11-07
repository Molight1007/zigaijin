# -*- coding: utf-8 -*-
"""工具④：Lean 语言转化（自然语言 → Lean 4 代码 · LLM 翻译）。

职责（用户定稿）：「大模型的语言不会是 lean 语言，所以就需要大模型将要发给
lean 的东西翻译为 lean 语言来给 lean 工具使用。」

实现：**由调用方注入 LLM client**（依赖注入 ⇒ 本板块不反向依赖 agent）。
输出为「带 sorry 占位」的 Lean 声明骨架 —— 只保证**陈述的形式化**，
证明留给审核环节（工具①）。
"""
from __future__ import annotations

import re

_SYS_HAMMER = (
    "你是 Lean 4 / Mathlib 形式化翻译器。把用户给出的数学陈述翻译为**一个** "
    "Lean 4 声明（theorem / lemma / def），要求：\n"
    "1) 只输出代码块，不要解释；\n"
    "2) 声明体写**自动化证明尝试链**（不要手写分步证明，交给 Lean 策略）：\n"
    "   `:= by first | omega | linarith | nlinarith | norm_num | ring | positivity "
    "| decide | tauto | sorry`\n"
    "   （按顺序尝试，全部失败时以 `sorry` 兜底——系统会识别 `sorry` 并判「未证明」）；\n"
    "3) 使用 `import Mathlib` 级别的可用记号；拿不准的记号改用 Mathlib 里"
    "语义最接近的现成定义；\n"
    "4) 第一行给一行注释 `-- 翻译注记：<不确定点，没有就写 无>`。"
)

_SYS_LEGACY = (
    "你是 Lean 4 / Mathlib 形式化翻译器。把用户给出的数学陈述翻译为**一个** "
    "Lean 4 声明（theorem / lemma / def），要求：\n"
    "1) 只输出代码块，不要解释；\n"
    "2) 声明体用 `by sorry` 占位（你的任务是翻译陈述，不是补证明）；\n"
    "3) 使用 `import Mathlib` 级别的可用记号；拿不准的记号改用 Mathlib 里"
    "语义最接近的现成定义；\n"
    "4) 第一行给一行注释 `-- 翻译注记：<不确定点，没有就写 无>`。"
)


def _choose_sys() -> str:
    """★ 2026-10-18：`lean_hammer` 开关（默认 True）——开=先试自动化策略链
    （零成本，实测可解决过半目标），关=旧行为（`by sorry` 只验陈述）。
    读取失败按默认开处理（与注册表默认一致）。"""
    try:
        from contracts.switch_registry import get_bool as _g
        return _SYS_HAMMER if _g("lean_hammer") else _SYS_LEGACY
    except Exception:  # noqa: BLE001
        return _SYS_HAMMER


_USER = """【题目/上下文】
{context}

【要翻译的陈述】
{statement}

请翻译为 Lean 4 声明。"""

_FENCE_RE = re.compile(r"```(?:lean4?|)\s*([\s\S]*?)```", re.IGNORECASE)
_DECL_RE = re.compile(r"^\s*(?:noncomputable\s+)?(?:private\s+|protected\s+)?"
                      r"(theorem|lemma|def|example)\s+([A-Za-z_][\w'.]*)?")


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def to_lean(statement: str, context: str = "", client=None,
            max_tokens: int = 8192) -> dict:
    """自然语言陈述 → Lean 4 代码。

    参数
    ----
    statement : 要翻译的数学陈述（自然语言 / LaTeX 均可）
    context   : 题目上下文（可选，帮助消歧）
    client    : 具备 ``chat(messages=..., temperature=..., max_tokens=...)`` 的
                LLM 客户端（由调用方注入；缺省 ⇒ 返回错误，不猜测）
    max_tokens: 翻译输出上限

    返回
    ----
    {"ok": bool, "lean_code": str, "decl_name": str, "notes": str, "error": str}
    """
    if not isinstance(statement, str) or not statement.strip():
        return {"ok": False, "lean_code": "", "decl_name": "", "notes": "",
                "error": "empty-statement"}
    if client is None:
        return {"ok": False, "lean_code": "", "decl_name": "", "notes": "",
                "error": "llm-client-missing（翻译需调用方注入 client）"}
    try:
        resp = client.chat(
            messages=[{"role": "system", "content": _choose_sys()},
                      {"role": "user", "content": _USER.format(
                          context=(context or "")[:4000],
                          statement=statement[:6000])}],
            temperature=0.1, max_tokens=max_tokens)
        text = resp if isinstance(resp, str) else str(resp)
    except Exception as exc:  # noqa: BLE001  翻译失败不抛（调用方按 ok=False 处理）
        return {"ok": False, "lean_code": "", "decl_name": "", "notes": "",
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}

    code = _strip_fence(text)
    if not code:
        return {"ok": False, "lean_code": "", "decl_name": "", "notes": "",
                "error": "empty-translation"}
    # 翻译注记（首行注释）
    notes = ""
    m = re.search(r"--\s*翻译注记[:：]?\s*(.+)", code)
    if m:
        notes = m.group(1).strip()[:300]
    # 声明名（theorem/lemma/def 后的第一个名字；example 无名）
    decl = ""
    for ln in code.splitlines():
        dm = _DECL_RE.match(ln)
        if dm:
            decl = dm.group(2) or ""
            break
    return {"ok": True, "lean_code": code, "decl_name": decl, "notes": notes,
            "error": ""}
