# -*- coding: utf-8 -*-
"""工具③：数学计算验证器（逐条机械复算 · 2026-10-15 用户定稿）。

职责（用户原话）：「就是对大模型的每一个数学计算进行机械检测，这需要大模型
将它的数学计算都发给这个工具，它来利用工具计算一遍，确保大模型不会算错；
要是算错就将算错的地方和正确答案返回。」

实现：SymPy 独立复算（与 Lean 数值核验互为独立交叉校验——见 config 注释）。
本工具是**纯计算**：不做逻辑审核（那是工具①），不做定理检索（那是工具②）。

安全防护：仅接受数学表达式字符串（禁 `__` / `import` / 换行 / 超长），
表达式解析失败按"无法验证"处理（不误判为错）。
"""
from __future__ import annotations

_EXPR_MAX = 800
_FORBIDDEN = ("__", "import", "open(", "exec", "eval", "lambda", "\\n")


def _parse(s: str):
    """字符串 → sympy 表达式；不可解析抛异常。"""
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor)
    txt = str(s or "").strip()
    if not txt or len(txt) > _EXPR_MAX:
        raise ValueError("empty-or-too-long")
    low = txt.lower()
    for bad in _FORBIDDEN:
        if bad in low:
            raise ValueError("forbidden-token: %s" % bad)
    # ★ 2026-10-15（审核 S10）：幂塔/超大指数防护 —— `9**9**9`（右结合
    #   ⇒ 9**387420489）级表达式会拖垮 SymPy。拒收（按"无法验证"处理）。
    import re as _re_e
    if _re_e.search(r"\*\*\s*\(?\s*\d{6,}", txt):
        raise ValueError("exponent-too-large")
    # ★ 2026-10-17 修正：原正则过宽，把 `a**2 + 3*b**2`（两个独立幂）也误判为
    #   幂塔（实测 017 型不等式因此直接"无法解析"）。现只抓**直接嵌套**：
    #   `**` 右操作数是标识符/数字且**紧跟另一个 `**`**（如 9**9**9 / a**2**3）。
    if _re_e.search(r"\*\*\s*[\w.]*\s*\*\*", txt) or \
            _re_e.search(r"\^\s*[\w.]*\s*\^", txt):
        raise ValueError("power-tower")
    tr = (standard_transformations
          + (implicit_multiplication_application, convert_xor))
    return parse_expr(txt, transformations=tr, evaluate=True,
                      local_dict={"pi": sympy.pi, "e": sympy.E,
                                  "sqrt": sympy.sqrt, "inf": sympy.oo})


def _equal(a, b, tol: float) -> "bool | None":
    """数学等价判定：数值按容差、符号按 simplify(a-b)==0。"""
    import sympy
    try:
        d = sympy.simplify(a - b)
        if d == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        da, db = complex(a.evalf()), complex(b.evalf())
        scale = max(1.0, abs(da), abs(db))
        return abs(da - db) <= tol * scale
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-15（审核 S10）：简化非零且数值不可比 —— 含自由符号时**无法
    #   判定**（返回 None，调用方计 n_err），绝不武断"算错"。
    try:
        if getattr(a, "free_symbols", None) or getattr(b, "free_symbols", None):
            return None
    except Exception:  # noqa: BLE001
        pass
    return False


def _render(v) -> str:
    """把 sympy 值渲染为可读字符串（数值保留足够精度）。"""
    try:
        if v.is_Number:
            if v.is_Integer:
                return str(int(v))
            if v.is_Rational:
                f = float(v)
                return str(v) if abs(f - round(f)) < 1e-12 else "%.15g" % f
            return "%.15g" % float(v.evalf())
    except Exception:  # noqa: BLE001
        pass
    return str(v)


def _prepare_constraints(constraints) -> list:
    """预解析前提为 `[(差式, op)]` —— 代值点要反复比对，不能每点重解析。"""
    out = []
    for c in constraints or []:
        try:
            out.append((_parse(str(c.get("expr") or "")) -
                        _parse(str(c.get("claimed") or "")),
                        str(c.get("op") or "ge").lower()))
        except Exception:  # noqa: BLE001  坏约束忽略（不放宽也不收紧）
            continue
    return out


def _constraints_hold(sub: dict, prepared, tol: float = 1e-9) -> bool:
    """代值点是否满足全部**题目前提**（如 `d>=2`、`0<x<=1`）。

    ★ 2026-10-21：没有前提就分不清「恒假断言」和「有条件才真的命题」——
      `x^(d-1) >= x` 在 d=1 时假，但题目说 `d>=2`，它是真命题（实测误杀 3 次）。
      该点未覆盖的符号 ⇒ 该约束不作数（不能因信息不全就判越界）。
    """
    for diff, op in prepared:
        try:
            d = diff.subs(sub)
            if d.free_symbols:
                continue
            v = float(d)
        except Exception:  # noqa: BLE001
            continue
        ok = {"ge": v >= -tol, "le": v <= tol,
              "gt": v > tol, "lt": v < -tol}.get(op, True)
        if not ok:
            return False
    return True


def _verify_inequality(lhs, rhs, op: str, tol: float = 1e-9,
                       constraints=None) -> "bool | None":
    """判定 ``lhs op rhs``（op: ge/le/gt/lt）—— True 成立 / False 有反例 / None 无法判定。

    ★ 2026-10-17（用户要求「让工具检错计算过程」）：原工具只验**数值等式**，
      017 类错误（``a^2+3b^2 >= 4|a||b|^3`` 为假）根本不在检测范围。
      本函数三重判定：
        ① 常数化简（simplify(lhs-rhs) 无自由符号 ⇒ 直接判）；
        ② 符号性质（sympy 的 is_nonnegative / is_nonpositive）；
        ③ **代值找反例**（正整数区确定性枚举，命中即确证 False）——抓 017 的关键。
      任一步不能定 ⇒ None（调用方计 n_err，绝不误杀）。
    """
    import sympy as _sympy                 # ★ 局部导入（原代码未定义 _sympy，
                                           #   异常被 except 吞掉 ⇒ 恒返回 None）
    try:
        a = _parse(str(lhs))
        b = _parse(str(rhs))
        d_raw = a - b                     # 代值用原始差式（simplify 可能变 Piecewise）
        d = _sympy.simplify(d_raw)
    except Exception:
        return None

    def _judge(v: float) -> bool:
        if op == "ge":
            return v >= -tol
        if op == "le":
            return v <= tol
        if op == "gt":
            return v > tol
        return v < -tol

    # ① 常数
    if not getattr(d, "free_symbols", None):
        try:
            return _judge(float(d))
        except Exception:
            pass
    # ② 符号性质（先原式、再因式分解——AM-GM 类 `a^2+b^2-2ab` 分解成
    #   `(a-b)^2` 后才能被 sympy 判为非负）
    try:
        for _e in (d, _sympy.factor(d)):
            if op in ("ge", "gt"):
                _p = _e.is_positive if op == "gt" else _e.is_nonnegative
            else:
                _p = _e.is_negative if op == "lt" else _e.is_nonpositive
            if _p is True:
                return True
            if _p is False:
                return False
    except Exception:
        pass
    # ③ 代值找反例（只用于确证 False；找不到不等价于成立）
    #   ★ 用 `d_raw`（未 simplify 的原始差）：simplify 可能引入 Piecewise，
    #     其 subs/float 会失败导致全部跳过（实测 017 型因此漏判）。
    #   ★ 2026-10-17：分「全正区 / 负区」，负区只记疑 —— 形如 `x+1/x>=2`
    #     （前提 x>0）的真命题在负数区必出"反例"，但那不构成命题为假
    #     （工具不知道前提）。
    #   ★ 2026-10-21 修正（条件不等式误杀）：原「全正区」取样是
    #     `Rational(randint(1,5), randint(1,3))` ⇒ 含 1/2 这类**小于 1 的正
    #     有理数**。对前提为 `q>=1` 的真命题 `q^2-1 >= 0`，q=1/2 时得 -0.75
    #     ⇒ 直接判 False ⇒ 候选被淘汰（实测误杀）。现分三区：
    #       · **正整数区**：**确定性枚举**（不随机 —— 同一候选两次运行必须同
    #         结论；1..5 足够覆盖 017 型反例），出现反例 ⇒ 确证 False；
    #       · **正分数区 / 负混合区**：只记疑 ⇒ None（前提未知，反例不作数）。
    #   ★ 前提预解析放在 try 之外：它自身已带异常保护，若放在 try 内，
    #     任何 NameError 都会被下面的 `except Exception: pass` 吞掉 ⇒ 静默退化成
    #     「全部无法判定」（实测踩到，所有不等式一律 None）。
    _prep = _prepare_constraints(constraints)
    try:
        import itertools
        import random
        syms = sorted(d_raw.free_symbols, key=lambda x: str(x))[:3]

        def _sample(pos: bool, den: int = 1) -> dict:
            _s = {}
            for sy in syms:
                _v = random.randint(1, 5) * (1 if pos else -1)
                _s[sy] = _sympy.Rational(_v, den)
            return _s

        # ① 正整数区：可判 False（017 型在此被抓）
        for _combo in itertools.product((1, 2, 3, 4, 5), repeat=len(syms)):
            _sub = {sy: _sympy.Rational(v) for sy, v in zip(syms, _combo)}
            if not _constraints_hold(_sub, _prep, tol):
                continue                 # 越出题目前提的点不构成反例
            try:
                v = float(_sympy.N(d_raw.subs(_sub)))
            except Exception:
                continue
            if not _judge(v):
                return False
        # ② 正分数区：只记疑（`1/x >= 1` 类命题前提是 0<x<=1，此处反例不作数）
        for _ in range(10):
            _sub = _sample(True, random.randint(2, 3))
            if not _constraints_hold(_sub, _prep, tol):
                continue
            try:
                v = float(_sympy.N(d_raw.subs(_sub)))
            except Exception:
                continue
            if not _judge(v):
                return None
        # ③ 负/混合区：只记疑
        for _ in range(20):
            _sub = _sample(False, random.randint(1, 3))
            if not _constraints_hold(_sub, _prep, tol):
                continue
            try:
                v = float(_sympy.N(d_raw.subs(_sub)))
            except Exception:
                continue
            if not _judge(v):
                return None
    except Exception:
        pass
    return None


def calc_verify(computations, tol: float = 1e-9, constraints=None) -> dict:
    """逐条机械复算。

    参数
    ----
    computations : list[dict]（[{"expr": "19.9 + (10 - sqrt(9.9))**2", "claimed": "20"}]）
                   或单条 dict / 单个字符串（视为 expr，claimed 留空 ⇒ 只求真值）
    tol          : 数值容差（相对）

    返回
    ----
    {"ok": bool,           # 全部通过（含"无法验证"条目？否——err 计入不通过语义由调用方定，
                           #   本处 ok=False 当且仅当存在 ok=False 的条目）
     "n_ok": int, "n_bad": int, "n_err": int,
     "results": [{"expr","claimed","actual","ok","detail"}]}
        · ok=True  → 一致（或 claimed 留空时的求值成功）
        · ok=False → 不一致（detail 给正确答案）或无法验证（detail 说明原因）
    """
    if isinstance(computations, (str, dict)):
        computations = [computations]
    results, n_ok, n_bad, n_err = [], 0, 0, 0
    for item in (computations or []):
        if isinstance(item, str):
            item = {"expr": item, "claimed": ""}
        if not isinstance(item, dict):
            continue
        expr_s = str(item.get("expr") or "")
        claimed_s = str(item.get("claimed") or "")
        rec = {"expr": expr_s, "claimed": claimed_s, "actual": "",
               "ok": False, "detail": ""}
        try:
            val = _parse(expr_s)
        except Exception as exc:  # noqa: BLE001
            rec["detail"] = "表达式无法解析（未做判定）：%s" % str(exc)[:120]
            n_err += 1
            results.append(rec)
            continue
        rec["actual"] = _render(val)
        if not claimed_s.strip():
            rec["ok"] = True
            rec["detail"] = "已复算（无声称值可比）"
            n_ok += 1
            results.append(rec)
            continue
        _op = str(item.get("op") or "").strip().lower()
        if _op in ("ge", "le", "gt", "lt") and claimed_s.strip():
            try:
                claimed = _parse(claimed_s)
            except Exception as exc:  # noqa: BLE001
                rec["detail"] = "声称值无法解析（未做判定）：%s" % str(exc)[:120]
                n_err += 1
                results.append(rec)
                continue
            _iv = _verify_inequality(val, claimed, _op, tol, constraints)
            _op_txt = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<"}[_op]
            if _iv is True:
                rec["ok"] = True
                rec["detail"] = "不等式成立"
                n_ok += 1
            elif _iv is None:
                rec["detail"] = "不等式无法判定（符号/代值均未定，未下结论）"
                n_err += 1
            else:
                rec["detail"] = ("**不等式不成立**：%s %s %s 有反例（已由随机代值确证）"
                                 % (_render(val), _op_txt, claimed_s.strip()[:60]))
                n_bad += 1
            results.append(rec)
            continue
        try:
            claimed = _parse(claimed_s)
        except Exception as exc:  # noqa: BLE001
            rec["detail"] = "声称值无法解析（未做判定）：%s" % str(exc)[:120]
            n_err += 1
            results.append(rec)
            continue
        _eq = _equal(val, claimed, tol)
        if _eq is True:
            rec["ok"] = True
            rec["detail"] = "一致"
            n_ok += 1
        elif _eq is None:
            # 含自由符号的差异 ⇒ 无法判定（计 n_err，不误报"算错"）
            rec["detail"] = "无法判定（含自由符号的符号差异，未下结论）"
            n_err += 1
        else:
            rec["detail"] = "**算错**：正确结果是 %s（声称 %s）" % (
                _render(val), claimed_s.strip()[:60])
            n_bad += 1
        results.append(rec)
    return {"ok": n_bad == 0 and n_err == 0, "n_ok": n_ok, "n_bad": n_bad,
            "n_err": n_err, "results": results}
