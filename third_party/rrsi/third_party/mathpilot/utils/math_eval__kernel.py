"""math_eval 求值引擎（自 utils/math_eval.py 逐字搬出）。

2026-10-08 批 D-me1：`utils/math_eval.py` 超过 800 行上限，本模块承载其
**求值引擎**（Fraction 精确算术 + ast 白名单 + float 近似 + SymPy 符号/根式），
均为纯函数 / 求值器类，**逐字搬、语义零变更**：
  · 精确路径：_BINOPS/_UNARYOPS/白名单函数/_Evaluator
  · 近似路径：_FLOAT_BINOPS/_FLOAT_FUNCS/_FloatEvaluator
  · 符号/根式/积分/求和：_symbolic_eval(+_core)/_split_call_args + 安全字符集

对外契约：`from utils.math_eval import <X>` 仍可用（主文件跳板式 re-export，同一对象 is）。
本模块**不导入** agent/tools（守 utils 分层 §2.4），也不反向导入主文件（无环）。
"""
from __future__ import annotations

import ast
import math
import re
from fractions import Fraction

# ============================================================
# 精确路径（Fraction 精确算术 + ast 白名单）
# ============================================================

# 允许的二元运算符 → (ast 节点类型, Fraction 运算)
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,          # Fraction / Fraction → 精确分数
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: _safe_pow(a, b),
}

_UNARYOPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


# 白名单函数：名称 → (实现, 参数个数限制说明)
def _fact(n: Fraction) -> Fraction:
    if n.denominator != 1 or n < 0:
        raise ValueError("fact 仅支持非负整数")
    return Fraction(math.factorial(int(n)))


def _comb(n: Fraction, k: Fraction) -> Fraction:
    if n.denominator != 1 or k.denominator != 1:
        raise ValueError("comb 参数必须为整数")
    nn, kk = int(n), int(k)
    if kk < 0 or kk > nn:
        raise ValueError(f"comb({nn},{kk}) 参数越界")
    return Fraction(math.comb(nn, kk))


def _perm(n: Fraction, k: Fraction) -> Fraction:
    if n.denominator != 1 or k.denominator != 1:
        raise ValueError("perm 参数必须为整数")
    nn, kk = int(n), int(k)
    if kk < 0 or kk > nn:
        raise ValueError(f"perm({nn},{kk}) 参数越界")
    return Fraction(math.perm(nn, kk))


def _gcd(a: Fraction, b: Fraction) -> Fraction:
    if a.denominator != 1 or b.denominator != 1:
        raise ValueError("gcd 仅支持整数")
    return Fraction(math.gcd(int(a), int(b)))


def _lcm(a: Fraction, b: Fraction) -> Fraction:
    if a.denominator != 1 or b.denominator != 1:
        raise ValueError("lcm 仅支持整数")
    ia, ib = int(a), int(b)
    if ia == 0 or ib == 0:
        # lcm(0, x) 数学上为 0。原实现会走 math.gcd(0, 0) = 0 触发
        # ZeroDivisionError，并经上层显示成"除数为零"（对调用方是误导性报错）。
        return Fraction(0)
    return Fraction(abs(ia * ib) // math.gcd(ia, ib))


def _floor(fr: Fraction) -> Fraction:
    return Fraction(math.floor(fr))


def _ceil(fr: Fraction) -> Fraction:
    return Fraction(math.ceil(fr))


def _min(*args: Fraction) -> Fraction:
    return min(args)


def _max(*args: Fraction) -> Fraction:
    return max(args)


def _sqrt_exact(fr: Fraction) -> Fraction | None:
    """完全平方开方 → 精确 Fraction；否则 None（需走近似路径）。"""
    if fr < 0:
        raise ValueError("sqrt 参数不能为负")
    n, d = fr.numerator, fr.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn == n and rd * rd == d:
        return Fraction(rn, rd)
    return None


def _square_part(n: int) -> int:
    """n 的最大平方因子部分（√n = √(s·b) 中的 s）。无 → 1。

    上限保护：n > 4e10 时不判定（视为无因子，走 ≈ 近似兜底），
    防 fact/comb 巨值把试除循环拖死（中学题 sqrt 参数远小于此）。
    """
    if n <= 1:
        return 1
    lim = math.isqrt(n)
    if lim > 200000:
        return 1
    best, p = 1, 2
    while p <= lim:
        p2 = p * p
        if n % p2 == 0:
            best *= p2
            n //= p2
            lim = math.isqrt(n)
        else:
            p = p + 1 if p == 2 else p + 2   # 2,3,5,7…（跳过偶数）
    return best


def _has_square_factor(fr: Fraction) -> bool:
    """sqrt(fr) 是否能开方化简（分子或分母含 >1 的平方因子）。"""
    for x in (fr.numerator, fr.denominator):
        if x > 1 and _square_part(x) > 1:
            return True
    return False


def _sqrt(fr: Fraction) -> Fraction:
    """精确路径的 sqrt：完全平方 → 精确；可提平方因子 → _SqrtSurd（走精确
    根式路径）；否则 _NeedsApprox（走 float 近似路径）。"""
    v = _sqrt_exact(fr)
    if v is not None:
        return v
    if _has_square_factor(fr):
        raise _SqrtSurd(f"sqrt({fr}) 含平方因子，可开方化简")
    raise _NeedsApprox(f"sqrt({fr}) 非完全平方")


def _ln(fr: Fraction) -> Fraction:
    raise _NeedsApprox(f"ln({fr}) 为超越数")


def _exp(fr: Fraction) -> Fraction:
    raise _NeedsApprox(f"exp({fr}) 为超越数")


# 精确路径函数表
_FUNCS = {
    "fact": _fact,
    "comb": _comb,
    "perm": _perm,
    "gcd": _gcd,
    "lcm": _lcm,
    "abs": abs,
    "binomial": _comb,   # 别名（部分模型输出 binomial 而非 comb）
    "sqrt": _sqrt,
    "floor": _floor,
    "ceil": _ceil,
    "min": _min,
    "max": _max,
    # ln/log/exp 在精确路径直接抛 _NeedsApprox（由 safe_eval 捕获后切换
    # float 近似路径求值），避免"不支持函数"误伤——它们不是断链，是能力标注为近似。
    "ln": _ln,
    "log": _ln,          # 数学书写中 log 默认自然对数
    "exp": _exp,
}

_FUNC_ARITY = {
    "fact": 1, "comb": 2, "perm": 2, "gcd": 2, "lcm": 2,
    "abs": 1, "binomial": 2,
    "sqrt": 1, "floor": 1, "ceil": 1,
    "min": -1, "max": -1,      # -1 = 变参（≥1）
    "ln": 1, "log": 1, "exp": 1,
}

# 计算结果上限保护：防止 fact(10**9) 之类的爆炸（内存/时间）
_MAX_RESULT_DIGITS = 20000
_MAX_INT_ARG = 10**6


class _NeedsApprox(Exception):
    """表达式含无理/超越函数（sqrt 无平方因子 / ln / log / exp），需切 float 近似路径。"""


class _SqrtSurd(Exception):
    """sqrt 参数含平方因子（√45 → 3√5），需切 SymPy 精确根式路径。"""


class _SymbolicError(Exception):
    """表达式含符号变量（Name 节点），计算器仅支持数值表达式。"""


class _UnsupportedFuncError(Exception):
    """调用白名单之外的函数（sin/cos/…），工具能力外。"""


def _safe_pow(a: Fraction, b: Fraction) -> Fraction:
    """幂运算保护：指数为整数；先预估结果位数，超限直接拒绝（不实际计算）。"""
    if b.denominator != 1 or abs(int(b)) > 100000:
        # 小数/分数指数给替代路径（10^0.5 → sqrt(10)）
        raise ValueError(
            "幂指数必须为 |n|≤100000 的整数；小数/分数指数请用 sqrt 或根式"
            "（如 10^0.5 → sqrt(10)）")
    e = int(b)
    if a.denominator != 1 and e < 0:
        raise ValueError("分数负指数不支持")
    # 位数预估：整数底数 ≈ len(str(a)) * e；分数取分子（更保守）
    digits = max(len(str(a.numerator)), len(str(a.denominator))) * e
    if digits > _MAX_RESULT_DIGITS:
        raise ValueError("计算结果过大（超过 20000 位）")
    return a ** e


class _Evaluator(ast.NodeVisitor):
    """ast 白名单求值器：任何未识别节点抛 ValueError。"""

    def __init__(self) -> None:
        self._result_digits = 0

    def _guard_digits(self, v: Fraction) -> None:
        self._result_digits = max(self._result_digits, len(str(v.numerator)) + len(str(v.denominator)))
        if self._result_digits > _MAX_RESULT_DIGITS:
            raise ValueError("计算结果过大（超过 20000 位）")

    def visit_Expression(self, node: ast.Expression) -> Fraction:  # noqa: N802
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:  # noqa: N802
        if isinstance(node.value, int):
            if abs(node.value) > _MAX_INT_ARG:
                raise ValueError("整数字面量过大")
            return Fraction(node.value)
        if isinstance(node.value, float):
            # <calc>0.5</calc> 这类小数写法：十进制字符串精确转 Fraction
            # （0.5→1/2，无二进制误差）；科学计数/非有限拒绝走 ERROR。
            if not math.isfinite(node.value):
                raise ValueError("不支持非有限浮点常量")
            try:
                return Fraction(repr(node.value))
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"不支持的浮点常量: {node.value!r}") from e
        raise ValueError(f"不支持的常量: {node.value!r}")

    def visit_Name(self, node: ast.Name) -> Fraction:  # noqa: N802
        # 符号变量（x/n/k/a…）→ WARN 降级（工具能力外），不是语法 ERROR。
        raise _SymbolicError(
            f"含符号变量 {node.id!r}，计算器仅支持数值表达式——"
            "请代入具体数值自检，或把代数断言写成 ```lean example``` "
            "交本地编译器核验，勿把符号式硬算当结论")

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:  # noqa: N802
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        a = self.visit(node.left)
        b = self.visit(node.right)
        v = op_fn(a, b)
        self._guard_digits(v)
        return v

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:  # noqa: N802
        op_fn = _UNARYOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op_fn(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Fraction:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持白名单函数调用")
        fname = node.func.id
        if fname not in _FUNCS:
            # 未知函数（sin/cos/tan 等常见越界）→ WARN 降级，文案给
            # "为什么不可算 + 替代路径"（不再只列函数名）
            _tri_hint = ("。三角/反三角/双曲无通用精确值：特殊角请写精确式"
                         "（如 sin(pi/6) 直接给 1/2、cos(pi/4) 给 sqrt(2)/2），"
                         "一般角请用 <check> 或 lean example 断言交编译器验算，"
                         "勿把心算近似当精确结论") if fname in (
                             "sin", "cos", "tan", "cot", "sec", "csc",
                             "asin", "acos", "atan", "sinh", "cosh", "tanh") else ""
            raise _UnsupportedFuncError(
                f"不支持函数 {fname}()；可用: "
                + "/".join(sorted(_FUNCS)) + _tri_hint)
        fn = _FUNCS[fname]
        want = _FUNC_ARITY[fname]
        if want == -1:
            if len(node.args) < 1:
                raise ValueError(f"{fname} 至少需要 1 个参数")
        elif len(node.args) != want:
            raise ValueError(f"{fname} 需要 {want} 个参数")
        for kw in node.keywords:
            raise ValueError("不支持关键字参数")
        args = [self.visit(a) for a in node.args]
        v = Fraction(fn(*args))
        self._guard_digits(v)
        return v

    def generic_visit(self, node: ast.AST) -> Fraction:  # noqa: N802
        raise ValueError(f"不允许的语法节点: {type(node).__name__}")


# ============================================================
# float 近似求值器：表达式含 sqrt 非平方 / ln / log / exp 时走此路径。
# 输出 15 位有效数字并带 ≈ 前缀，不可当精确结论用。
# ============================================================

_FLOAT_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: math.pow(a, b),
}


def _f_int(f: float, what: str) -> int:
    if not math.isfinite(f) or not f.is_integer():
        raise ValueError(f"{what} 参数必须为整数")
    return int(f)


def _f_fact(f: float) -> float:
    n = _f_int(f, "fact")
    if n < 0 or n > 100000:
        raise ValueError("fact 参数越界")
    return float(math.factorial(n))


def _f_comb(fa: float, fb: float) -> float:
    a, b = _f_int(fa, "comb"), _f_int(fb, "comb")
    if b < 0 or b > a:
        raise ValueError(f"comb({a},{b}) 参数越界")
    return float(math.comb(a, b))


def _f_perm(fa: float, fb: float) -> float:
    a, b = _f_int(fa, "perm"), _f_int(fb, "perm")
    if b < 0 or b > a:
        raise ValueError(f"perm({a},{b}) 参数越界")
    return float(math.perm(a, b))


def _f_gcd(fa: float, fb: float) -> float:
    return float(math.gcd(_f_int(fa, "gcd"), _f_int(fb, "gcd")))


def _f_lcm(fa: float, fb: float) -> float:
    return float(abs(_f_int(fa, "lcm") * _f_int(fb, "lcm"))
                 // math.gcd(_f_int(fa, "lcm"), _f_int(fb, "lcm")))


_FLOAT_FUNCS = {
    "fact": _f_fact, "comb": _f_comb, "perm": _f_perm,
    "gcd": _f_gcd, "lcm": _f_lcm, "binomial": _f_comb,
    "abs": abs,
    "sqrt": math.sqrt,
    "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max,
    "ln": math.log, "log": math.log, "exp": math.exp,
}


class _FloatEvaluator(ast.NodeVisitor):
    """float 域近似求值器（仅由 safe_eval 在 _NeedsApprox 后调用）。"""

    def __init__(self) -> None:
        self._symbolic = False

    def _guard(self, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("结果过大或未定义（inf/nan）")
        return v

    def visit_Expression(self, node: ast.Expression) -> float:  # noqa: N802
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:  # noqa: N802
        if isinstance(node.value, int):
            return float(node.value)
        if isinstance(node.value, float):
            return node.value
        raise ValueError(f"不支持的常量: {node.value!r}")

    def visit_Name(self, node: ast.Name) -> float:  # noqa: N802
        raise _SymbolicError(f"含符号变量 {node.id!r}，计算器仅支持数值表达式")

    def visit_BinOp(self, node: ast.BinOp) -> float:  # noqa: N802
        op_fn = _FLOAT_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        try:
            return self._guard(op_fn(self.visit(node.left), self.visit(node.right)))
        except ZeroDivisionError as e:
            raise ZeroDivisionError("除零") from e
        except OverflowError as e:
            raise ValueError("结果过大") from e

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:  # noqa: N802
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError(f"不支持的运算符: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call) -> float:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持白名单函数调用")
        fname = node.func.id
        if fname not in _FLOAT_FUNCS:
            _tri_hint = (
                "。三角/反三角/双曲无通用精确值：特殊角请写精确式（如 sin(pi/6)"
                "= 1/2），一般角请用 <check> 或 lean example 断言验算") if fname in (
                    "sin", "cos", "tan", "asin", "acos", "atan",
                    "sinh", "cosh", "tanh") else ""
            raise _UnsupportedFuncError(
                f"不支持函数 {fname}()；可用: "
                + "/".join(sorted(_FUNCS)) + _tri_hint)
        fn = _FLOAT_FUNCS[fname]
        want = _FUNC_ARITY[fname]
        if want == -1:
            if len(node.args) < 1:
                raise ValueError(f"{fname} 至少需要 1 个参数")
        elif len(node.args) != want:
            raise ValueError(f"{fname} 需要 {want} 个参数")
        for kw in node.keywords:
            raise ValueError("不支持关键字参数")
        args = [self.visit(a) for a in node.args]
        return self._guard(float(fn(*args)))

    def generic_visit(self, node: ast.AST) -> float:  # noqa: N802
        raise ValueError(f"不允许的语法节点: {type(node).__name__}")


# ============================================================
# SymPy 符号/根式计算：
# 符号变量、可开方化简的根式、integral() 积分、sum() 求和 → 本地 SymPy
# 精确化简/求值。安全：先过字符集白名单（无引号/下划线/中文 → sympify 无
# RCE 面），^ 幂记号仅在本分支内转 **（数值表达式不经过这里）；import 失败/
# 化简失败/超长 → 一律返回 None，调用方回落旧行为（WARN 或 ≈），绝不阻断。
# ============================================================

# 允许的数学字符集：数字/单字母符号/运算/括号/逗号/小数点/空白。
# 有意排除：引号 ' "、下划线 _（禁 dunder 属性链）、方括号 []（禁下标）、
# % //（符号下语义歧义）。无引号+无下标+无下划线 → sympify 无代码执行面。
_SYM_SAFE_RE = re.compile(r"^[0-9a-zA-Z*/().,\-+ ]+$")
_INTEGRAL_CALL_RE = re.compile(r"\bintegral\s*\(", re.IGNORECASE)
# 求和协议 sum(f, x, a, b)（SymPy summation；对齐 integral 语法）
_SUM_CALL_RE = re.compile(r"\bsum\s*\(", re.IGNORECASE)
# 符号分支超时护栏（sympy 极端输入防卡主线程）
_SYM_TIMEOUT_SEC = 3.0
_SYM_EXECUTOR = None
_POW_INT_RE = re.compile(r"\*\*\s*(\d+)")
_SQRT_NUM_RE = re.compile(r"sqrt\(\s*(\d+)(?:\s*/\s*(\d+))?\s*\)")


def _max_int_exponent(expr: str) -> int:
    """表达式里最大的数字幂指数（**N），用于决定是否安全展开。"""
    mx = 0
    for m in _POW_INT_RE.finditer(expr):
        try:
            mx = max(mx, int(m.group(1)))
        except ValueError:
            pass
    return mx


def _expr_has_simplifiable_sqrt(expr: str) -> bool:
    """表达式含可开方化简的 sqrt(数字)（√45 → 3√5 型）。

    只扫纯数字参数的 sqrt（复合参数如 sqrt(2*15) 罕见，覆盖不到时走 ≈）。
    """
    for m in _SQRT_NUM_RE.finditer(expr):
        try:
            num = int(m.group(1))
            den = int(m.group(2)) if m.group(2) else 1
        except ValueError:
            continue
        if den and _has_square_factor(Fraction(num, den)):
            return True
    return False


def _symbolic_eval(expr: str) -> str | None:
    """符号/根式/积分/求和表达式 → SymPy 化简，返回结果串；失败返回 None。

    外包线程池 + 3s 超时——sympy 对极端输入（深嵌套/病态化简）可能秒级卡死，
    不能占用调用方主线程。超时线程仍会跑完（占 1 个池 worker），卡死极罕见
    可接受；超时按"化简失败"回落（WARN/≈），不阻断。
    """
    try:
        import concurrent.futures as _cf
    except Exception:  # noqa: BLE001
        return None
    global _SYM_EXECUTOR
    if _SYM_EXECUTOR is None:
        _SYM_EXECUTOR = _cf.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="calc_sym")
    try:
        fut = _SYM_EXECUTOR.submit(_symbolic_eval_core, expr)
        return fut.result(timeout=_SYM_TIMEOUT_SEC)
    except Exception:  # noqa: BLE001  超时/化简失败一律回落，不阻断
        return None


def _symbolic_eval_core(expr: str) -> str | None:
    """符号分支实际执行（见 _symbolic_eval 外层说明）。"""
    if not expr.strip():
        return None
    s = expr.replace("^", "**")                  # 兼容数学幂记号（仅本分支）
    if not _SYM_SAFE_RE.match(s):
        return None
    if "//" in s or "%" in s:                    # 符号整除/取模语义歧义 → 能力外
        return None
    try:
        import sympy as sp              # noqa: PLC0415  # 延迟导入
    except ImportError:
        return None
    try:
        res = None
        # ---- 求和协议 sum(f, x, a, b)（对齐 integral 拆参）----
        msum = _SUM_CALL_RE.search(s)
        if msum and _INTEGRAL_CALL_RE.search(s) is None:
            iargs = _split_call_args(s, msum.end() - 1)
            # 支持显式变量 sum(f, x, a, b) 与省略变量 sum(f, a, b)（自动取被加式
            # 唯一自由符号；多符号歧义 → None 回落 WARN）
            if iargs is None or len(iargs) not in (3, 4):
                return None
            f_expr = sp.sympify(iargs[0])
            if len(iargs) == 4:
                var = sp.Symbol(iargs[1].strip())
                a, b = sp.sympify(iargs[2]), sp.sympify(iargs[3])
            else:
                _fs = list(f_expr.free_symbols)
                if len(_fs) != 1:
                    return None
                var = _fs[0]
                a, b = sp.sympify(iargs[1]), sp.sympify(iargs[2])
            res = sp.summation(f_expr, (var, a, b))
            if isinstance(res, sp.Sum):          # 求不出闭合和 → 能力外
                return None
        else:
            # ---- 积分协议 integral(f, x[, a, b])：手动拆参（支持定积分）----
            m = _INTEGRAL_CALL_RE.search(s)
            if m:
                iargs = _split_call_args(s, m.end() - 1)   # 平衡括号取参
                if iargs is None or len(iargs) not in (2, 4):
                    return None
                f_expr = sp.sympify(iargs[0])
                var = sp.Symbol(iargs[1].strip())
                if len(iargs) == 2:
                    res = sp.integrate(f_expr, var)
                else:
                    a, b = sp.sympify(iargs[2]), sp.sympify(iargs[3])
                    res = sp.integrate(f_expr, (var, a, b))
                if isinstance(res, sp.Integral):   # 仍积分不出 → 能力外
                    return None
            else:
                # ---- 常规符号/根式化简 ----
                parsed = sp.sympify(s)
                if _max_int_exponent(expr) > 50:
                    # 大指数（如 (1-x)**2024）：simplify 会走慢路径（秒级）→ 跳过，
                    # sympify 本身已规范化；此时也不展开（防爆炸）。
                    res = parsed
                else:
                    res = sp.simplify(parsed)
                    # 数字幂指数 ≤12 时补一次展开（simplify 常保持积形式 k*(1-k)，
                    # expand 给出展开式 -k**2+k——更有"运算发生"；展开后**不再
                    # simplify**：simplify 会把展开式重新因子化回去）
                    if _max_int_exponent(expr) <= 12:
                        _exp = sp.expand(res)
                        if len(sp.sstr(_exp)) <= 8 * len(expr) + 200:  # 防爆炸兜底
                            res = _exp
        if res is None:
            return None
        # Piecewise 可能带外层系数（如 -Piecewise(...)）→ 递归取主分支
        res = res.replace(
            lambda e: isinstance(e, sp.Piecewise),
            lambda pw: pw.args[0][0])
        out = sp.sstr(res)
        if len(out) > 2000 or "\n" in out:
            return None
        # 纯数学常量（pi/e 等 NumberSymbol、无自由变量）→ 给 15 位数值近似
        # （≈ 前缀），防 `pi` 被当普通符号结果回填。精确根式（3*sqrt(5)）不含
        # NumberSymbol → 不受影响，仍返回精确式。
        try:
            if not getattr(res, "free_symbols", None) and res.atoms(sp.NumberSymbol):
                return f"≈ {sp.N(res, 15)}"
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:                               # 化简失败一律回落，不阻断
        return None


def _split_call_args(s: str, open_idx: int) -> list[str] | None:
    """从 '(' 位置开始按**顶层逗号**切分调用参数（含嵌套括号）。

    返回参数串列表（strip 后）；括号不闭合 → None。
    """
    depth, start, parts = 0, open_idx + 1, []
    i = open_idx
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                parts.append(s[start:i])
                return [p.strip() for p in parts]
        elif c == "," and depth == 1:
            parts.append(s[start:i])
            start = i + 1
        i += 1
    return None                                   # 未闭合
