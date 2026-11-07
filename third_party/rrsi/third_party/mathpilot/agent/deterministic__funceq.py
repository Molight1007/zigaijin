# -*- coding: utf-8 -*-
"""确定性验证通道的「函数方程回代」职责簇
========================================

2026-10-08 批 D1：从 agent/deterministic.py **逐字搬出**。搬出后形如模块级函数：
实例方法保留 `self` 首参（原方法体逐字不变），静态方法去 `@staticmethod`（普通函数）。
deterministic.py 在 `DeterministicChecker` 类内保留**跳板**
（`_x = _det_funceq._x` / `_x = staticmethod(_det_funceq._x)`），故
`self._x(...)` / `DeterministicChecker._x(...)` 契约逐字不变（与
agent/sub_goal_solver__lean.py 批2-D 同口径）。

一并搬入的**解析前置工具**（本簇自洽所需）：`_MATH_CHARS` / `_math_trim` /
`_normalize_answer` / `_latex_prep` / `_FUNC_SYMBOL_BLACKLIST`；其中
`_math_trim` / `_normalize_answer` 被 deterministic.py 主链复用，由该文件原位
re-export（`from .deterministic__funceq import ...`）。

本模块**自洽**：只依赖 stdlib（re）+ SymPy，**不 import** deterministic.py（无循环）。
"""
from __future__ import annotations

import re

try:
    import sympy as sp
    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - 环境缺依赖时的降级路径
    sp = None
    _HAS_SYMPY = False


# 数学主体字符集（用于修剪等式两侧的中文叙述前缀/后缀）
_MATH_CHARS = r"0-9a-zA-Zα-ωΑ-Ω\\\^\(\)\+\-\*/=.,，≤≥<>!%"


def _math_trim(text: str) -> str:
    """去掉表达式两侧的中文叙述（如"解方程："），保留数学主体。"""
    if not text:
        return ""
    t = text.strip()
    m = re.search(rf"[{_MATH_CHARS}]", t)
    if m:
        t = t[m.start():]
    m = re.search(rf"[{_MATH_CHARS}]+$", t)
    if m:
        t = t[:m.end()]
    return t.strip()

def _normalize_answer(text: str) -> str:
    """解析前归一答案：剥外表壳（``\\boxed{}``/``$``/``\\(\\)``/``\\[\\]``）+ 折叠常见 LaTeX。

    ★ 这是"解答题 Level0 结构性无输出"的第一真因修复。实测 `_safe_parse('\\boxed{-4}')`
    → None ⇒ `check_answer` 在 ``method="parse"`` 即返回 unknown，7/10 题命中。
    """
    s = (text or "").strip()
    if not s:
        return ""
    try:
        from .answer_oracle import AnswerOracle as _AO
        s = _AO.strip_wrappers(s) or s
    except Exception:  # noqa: BLE001  提交包/导入异常时不动原文
        pass
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ ", "", s)
    s = re.sub(r"\\times|\\cdot", "*", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^(\1)", s)
    s = s.replace("\\{", "{").replace("\\}", "}")
    return s.replace("$", "").strip()

def _latex_prep(text: str) -> str:
    """函数方程解析专用的 LaTeX 折叠（更激进：把 `^{}` 变 `**()`、去花括号）。

    与 `_normalize_answer` 的分工：那条用于"单表达式"解析（保留 `^` 交给 `_safe_parse`
    的预处理），这条用于把**方程两侧**交给 `sympy.parsing.sympy_parser.parse_expr`。
    """
    t = (text or "").strip().replace("$", "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\times|\\cdot", "*", t)
    t = re.sub(r"\\left|\\right", "", t)
    t = re.sub(r"\\!|\\,|\\;|\\ ", "", t)
    t = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", t)
    return t.replace("{", "").replace("}", "")

# 解析函数方程时需排除的已知数学函数名（避免把 sin/cos 之类误当未知函数符号）
_FUNC_SYMBOL_BLACKLIST = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "exp", "log", "ln", "lg", "sqrt", "abs", "min",
    "max", "gcd", "lcm", "floor", "ceil", "round", "det", "mod", "arg",
    "dim", "deg", "root", "frac", "atan", "asin", "acos",
})

# ------------------------------------------------------------------
# 2026-10-07 新增：函数方程回代（解答题可局部客观判定的子类）
# ------------------------------------------------------------------
def _detect_function_symbols(problem: str) -> set:
    """从题目中检出"作为函数使用"的单字母符号（出现 ≥2 次 `X(` 形态）。"""
    counts: dict = {}
    try:
        for m in re.finditer(r"(?<![A-Za-z0-9\\])([A-Za-z])\s*\(", problem or ""):
            name = m.group(1)
            if name.lower() in _FUNC_SYMBOL_BLACKLIST:
                continue
            counts[name] = counts.get(name, 0) + 1
    except Exception:  # noqa: BLE001
        return set()
    return {s for s, n in counts.items() if n >= 2}

def _extract_functional_equation(problem: str, fsym: str) -> str | None:
    """从题目里提取一条"两侧都含该函数应用"的单个 `=` 等式。"""
    try:
        segs = re.findall(r"\$\$(.+?)\$\$", problem, re.S)
        segs += re.findall(r"\$(.+?)\$", problem, re.S)
        segs += re.split(r"[。\n]", problem)
        for seg in segs:
            if seg.count("=") != 1:
                continue
            l, r = seg.split("=", 1)
            ls, rs = l.replace(" ", ""), r.replace(" ", "")
            if (fsym + "(") in ls and (fsym + "(") in rs:
                return seg.strip()
        return None
    except Exception:  # noqa: BLE001
        return None

def _parse_with_function(self, text: str, fsym: str):
    """把含 `fsym(...)` 的 LaTeX 片段解析为 SymPy 表达式（失败返回 None）。"""
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        local = {fsym: sp.Function(fsym)}
        trans = standard_transformations + (
            convert_xor, implicit_multiplication_application)
        return parse_expr(_latex_prep(text), local_dict=local,
                          transformations=trans, evaluate=False)
    except Exception:  # noqa: BLE001
        return None

def _extract_function_definitions(answer: str, fsym: str) -> list:
    """从答案里提取函数定义 `fsym(变量)=表达式`（返回 [(变量, 表达式串), ...]）。"""
    s = _normalize_answer(answer)
    pat = re.compile(rf"{re.escape(fsym)}\s*\(\s*([A-Za-z])\s*\)\s*=\s*")
    defs = []
    try:
        for m in pat.finditer(s):
            var = m.group(1)
            start = m.end()
            nxt = pat.search(s, start)
            end = nxt.start() if nxt else len(s)
            body = s[start:end].strip().rstrip(".,;\\ ")
            if body:
                defs.append((var, body))
    except Exception:  # noqa: BLE001
        return defs
    return defs

def _check_one_function(self, eq, var: str, body: str, fsym: str):
    """把候选函数 `fsym(var)=body` 代入方程做恒等判定。

        Returns:
            True = 满足；False = 明确不满足；None = 判不了（宁可 unknown 绝不误杀）。
        """
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        trans = standard_transformations + (
            convert_xor, implicit_multiplication_application)
        cand = parse_expr(_latex_prep(body), transformations=trans,
                          evaluate=False)
        v = sp.Symbol(var)
        substituted = eq.replace(
            lambda e: isinstance(e, sp.Function) and e.func.__name__ == fsym,
            lambda e: cand.subs(v, e.args[0]))
        # 仍有未替换的函数应用 ⇒ 判不了
        if substituted.atoms(sp.Function):
            return None
        val = sp.simplify(substituted)
        if val == 0:
            return True
        # 残式仍含其它不确定项 ⇒ 判不了
        if val.atoms(sp.Function):
            return None
        # 数值抽样复核：化简未归零但**数值处处为 0** ⇒ 视为化简不足，判不了
        if self._numerically_zero(val):
            return None
        return False
    except Exception:  # noqa: BLE001
        return None

def _numerically_zero(self, expr, samples: int = 12, tol: float = 1e-6) -> bool:
    """对表达式随机取有理数点，判断是否处处 ≈ 0（防 `simplify` 漏化简导致误杀）。"""
    try:
        syms = sorted(expr.free_symbols, key=str)
        if not syms:
            n = self._num(expr)
            return n is not None and abs(n) <= tol
        for _ in range(samples):
            subs = {s: self._rng.uniform(-3.0, 3.0) for s in syms}
            n = self._num(expr.subs(subs))
            if n is None:
                continue
            if abs(n) > tol:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False

def _verify_functional_equations(self, problem: str, answer: str) -> dict:
    """函数方程回代：题目给 `F(...)=...`、答案给 `F(x)=expr` → 符号代入恒等判定。

        错误候选函数代入化简后会留下非零残式（且不含函数符号）；正确候选化简为 0。
        宁 unknown 不误杀：解析不出、残式仍含函数符号、或化简不足 → unknown。
        """
    if not _HAS_SYMPY or not problem or not answer:
        return {"verdict": "unknown", "evidence": ""}
    try:
        for fsym in sorted(self._detect_function_symbols(problem)):
            eq_text = self._extract_functional_equation(problem, fsym)
            if not eq_text:
                continue
            defs = self._extract_function_definitions(answer, fsym)
            if not defs:
                continue
            lhs_s, rhs_s = eq_text.split("=", 1)
            lhs = self._parse_with_function(lhs_s, fsym)
            rhs = self._parse_with_function(rhs_s, fsym)
            if lhs is None or rhs is None:
                continue
            eq = lhs - rhs
            n_ok = 0
            for var, body in defs:
                r = self._check_one_function(eq, var, body, fsym)
                if r is None:
                    n_ok = -1
                    break
                if not r:
                    return {
                        "verdict": "fail", "confidence": 1.0,
                        "evidence": f"函数方程回代失败: {fsym}({var})={body} "
                                    f"不满足 {_math_trim(eq_text)[:80]}",
                        "method": "functional_substitution"}
                n_ok += 1
            if n_ok > 0:
                return {
                    "verdict": "pass", "confidence": 0.8,
                    "evidence": f"函数方程回代通过: {n_ok} 个候选函数均满足 "
                                f"{_math_trim(eq_text)[:80]}",
                    "method": "functional_substitution"}
        return {"verdict": "unknown", "evidence": ""}
    except Exception as e:  # noqa: BLE001
        return {"verdict": "unknown",
                "evidence": f"函数方程回代异常: {str(e)[:80]}"}
