"""utils/math_eval.py —— 确定性精确计算内核（2026-10-01 从 agent/calc_tool.py 迁出）。

本模块只保留**纯数学求值工具函数**（原 `agent/calc_tool.py` 的 B 类）：
给定一个数学表达式字符串 → 返回精确 / 符号 / 近似 / WARN 结果串。

三种求值能力（安全边界一致：**绝不 eval 任意 Python**）：
  1. `safe_eval` —— 单表达式求值：Fraction 精确算术 + ast 节点白名单；
     无理/超越（sqrt 无平方因子、ln、log、exp）→ float 近似（"≈ …"）；
     符号/根式/积分/求和 → 本地 SymPy 精确化简；
  2. `safe_eval_subst` —— 把给定变量的值**精确代入**符号表达式求真值
     （用于把"关系式答案"与"givens"接起来做核验）；
  3. `to_exact_number` —— 答案串 → 精确 `Fraction`（非纯数值一律 None）。

表达式语言：
  - 四则 + - * /（/ 产生精确分数），幂 **（兼容数学记号 ^），取模 %，整除 //
  - 括号 ( )；隐式乘 2(x+1) / 2x / (a+b)(c+d) 自动识别为显式 *
  - 函数：fact(n)、comb(n,k)、perm(n,k)、gcd(a,b)、lcm(a,b)、abs(x)、
          sqrt(x)、floor(x)/ceil(x)、min(...)/max(...)、
          ln(x)/log(x)（自然对数）、exp(x)、
          integral(f, x) / integral(f, x, a, b) 积分、
          sum(f, x, a, b)（可省略变量）求和
  - 符号变量与多项式/根式 → SymPy 精确化简（如 (x-1)*(x+1) → x**2 - 1）
  - 示例：1/2 + 1/3 → 5/6 · sqrt(45) → 3*sqrt(5) · sqrt(15) → ≈ 3.87298…

结果形态：
  - 精确值（整数 / 分数 / 精确根式如 3*sqrt(5)）—— 可直接信任
  - 符号化简式（含变量，如 1/(n + 1)、x**2 - 1）—— SymPy 化简的恒等式
  - "≈ x.xxx" —— 含无理/超越函数且无法开方化简的近似值（15 位有效数字），
    只能核对量级/接近性，**不能当作精确结论**
  - "WARN: ..." —— 能力外（未知函数 sin/cos 等、含符号的整除/取模、
    表达式过长、SymPy 化简失败）。**不是错误消息**，语义是"该步未经工具确认"
  - "ERROR: ..." —— 语法/数值非法

安全：符号/根式走 SymPy 前先过字符集白名单（无引号/下划线/方括号/中文 →
sympify 无 RCE 面）；数值表达式走 ast.parse + 节点白名单，任何未列出的
节点/函数一律拒绝；SymPy 分支外包线程池 + 3s 超时，避免病态输入卡死主线程。
import sympy 失败/化简失败 → 回落 WARN 或 ≈，绝不阻断。
"""

from __future__ import annotations

import ast
import re
import threading
from fractions import Fraction

# 2026-10-08 批 D-me1：求值引擎（精确/近似/符号分支）已逐字搬到
# utils/math_eval__kernel.py。跳板式 re-export ⇒ `from utils.math_eval
# import <X>` 零改动 + **同一对象 is**。⚠ 勿按「未使用」清理（契约面）。
from .math_eval__kernel import (  # noqa: F401
    _BINOPS, _UNARYOPS, _fact,
    _comb, _perm, _gcd,
    _lcm, _floor, _ceil,
    _min, _max, _sqrt_exact,
    _square_part, _has_square_factor, _sqrt,
    _ln, _exp, _FUNCS,
    _FUNC_ARITY, _MAX_RESULT_DIGITS, _MAX_INT_ARG,
    _NeedsApprox, _SqrtSurd, _SymbolicError,
    _UnsupportedFuncError, _safe_pow, _Evaluator,
    _FLOAT_BINOPS, _f_int, _f_fact,
    _f_comb, _f_perm, _f_gcd,
    _f_lcm, _FLOAT_FUNCS, _FloatEvaluator,
    _SYM_SAFE_RE, _INTEGRAL_CALL_RE, _SUM_CALL_RE,
    _SYM_TIMEOUT_SEC, _POW_INT_RE, _SQRT_NUM_RE,
    _max_int_exponent, _expr_has_simplifiable_sqrt, _symbolic_eval,
    _symbolic_eval_core, _split_call_args,
)


# ============================================================
# 数学隐式乘预处理（2(x+1) / (a+b)(c+d) / 2x → 显式 *）
# ============================================================
_IMPLICIT_DIGIT_LPAREN = re.compile(r"(?<=[0-9])\(")       # 2(  → 2*(
_IMPLICIT_RPAREN_TOKEN = re.compile(r"\)(?=[0-9a-zA-Z(])")  # )( )x )2 → )*…
_IMPLICIT_DIGIT_LETTER = re.compile(r"([0-9])([a-zA-Z])")   # 2x → 2*x（分母位除外）


def _prev_nonspace_char(s: str, pos: int) -> str:
    j = pos - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    return s[j] if j >= 0 else ""


def _preprocess_implicit_mul(s: str) -> str:
    """把确定无歧义的数学隐式乘转为显式 `*`（ast 解析前调用）。

    规则（宁漏勿误，绝不静默改义）：
      - `2(x+1)` → `2*(x+1)`（数字紧跟左括号）
      - `(a+b)(c+d)` / `)x` / `)2` → `(a+b)*(c+d)`（右括号后接数字/字母/左括号）
      - `2x+1` → `2*x+1`（数字紧跟字母；**分母位跳过**：`1/2x` 数学上有
        1/(2x) 歧义，保留原样 → 解析失败走 WARN 引导，不静默按 (1/2)*x 算错）
    函数名后括号（sqrt(…)/comb(…)/integral(…)）与 `^`/`**` 后括号不转。
    """
    s = _IMPLICIT_DIGIT_LPAREN.sub("*(", s)
    s = _IMPLICIT_RPAREN_TOKEN.sub(")*", s)
    s = _IMPLICIT_DIGIT_LETTER.sub(
        lambda m: m.group(1)
        + ("*" if _prev_nonspace_char(m.string, m.start()) != "/" else "")
        + m.group(2),
        s)
    return s


def _err_msg(e: Exception) -> str:
    """统一错误文案：除零专名，其余用异常消息原文。"""
    if isinstance(e, ZeroDivisionError):
        return "除数为零（表达式非法，检查分母）"
    return str(e) or type(e).__name__


# 净化用正则：剥离非 ASCII（中文/全角标点/全角乘除号）与 = 尾巴
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_TRAIL_TAIL_RE = re.compile(r"[=＝].*$")          # 截断 "3-1=2" 的 = 尾巴
_WS_RE = re.compile(r"\s+")




def safe_eval(expr: str) -> str:
    """安全求值一个数学表达式，返回精确/符号/近似/WARN 多态结果字符串。

    成功（精确）→ "1960000" / "5/6" / "3*sqrt(5)"（精确根式）/
                  "x**2 - 1" / "1/(n + 1)"（SymPy 符号化简式）
    近似 → "≈ 3.14159265358979"（sqrt 无平方因子 / ln / log / exp，15 位）
    能力外 → "WARN: ..."（未知函数 / 含符号的整除取模 / SymPy 化简失败
             ——**不是错误**，是降级标记）
    失败 → "ERROR: ..."（语法/数值非法）
    """
    expr = expr.strip()
    if not expr:
        return "ERROR: 空表达式"
    if len(expr) > 500:
        return "ERROR: 表达式过长"
    return _eval_core(expr)


def _eval_core(expr: str) -> str:
    """白名单求值核心：单个数学表达式 → 结果串。

    `^`→`**` 归一 → 隐式乘预处理 → ast 白名单求值 → 精确 / 符号根式 / float
    近似。**不 eval 任意 Python**：未列出的 ast 节点一律拒绝（参数安全边界）。
    """
    # Python ast 中 ^ 是 BitXor，精确/近似求值器均不识 → 统一在解析前
    # ^→**（对已含 ** 的串幂等；_symbolic_eval 内自带 ^→** 亦幂等）。
    expr = expr.replace("^", "**")
    # 数学隐式乘（2(x+1)/2x/(a+b)(c+d)）→ 显式 *，再解析
    expr = _preprocess_implicit_mul(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        # 语法错给可行动引导：运算符显式、分式分母带变量写法
        return (f"ERROR: 表达式语法错误（{e.msg}）。"
                "请检查：运算符需显式（如 2*x 勿写 2x）、括号配对、"
                "分式分母含变量请写 1/(2*x) 形式。")
    # 1) 精确路径（纯数值；sqrt 完全平方→精确）
    try:
        v = _Evaluator().visit(tree)
        if v.denominator == 1:
            return str(v.numerator)
        return f"{v.numerator}/{v.denominator}"
    except _SqrtSurd:
        # 1a) sqrt 可开方化简（√45 → 3*sqrt(5)）：SymPy 精确根式，失败回落 ≈
        _sym = _symbolic_eval(expr)
        if _sym is not None:
            return _sym
    except _NeedsApprox:
        # 1b) 表达式里含其他可开方化简的 sqrt（√15+√20 型）→ 根式化简优先
        if _expr_has_simplifiable_sqrt(expr):
            _sym = _symbolic_eval(expr)
            if _sym is not None:
                return _sym
        # 1c) 纯无理无平方因子 → 走 float 近似（下方第 2 段）
    except _SymbolicError as e:
        # 1d) 含符号变量 → SymPy 符号化简；失败才退回 WARN 引导
        _sym = _symbolic_eval(expr)
        if _sym is not None:
            return _sym
        return f"WARN: {e}"
    except _UnsupportedFuncError as e:
        # integral()/sum() 是符号分支协议函数（精确求值器不识）→ 先试 SymPy
        if _INTEGRAL_CALL_RE.search(expr) or _SUM_CALL_RE.search(expr):
            _sym = _symbolic_eval(expr)
            if _sym is not None:
                return _sym
        return f"WARN: {e}"
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError) as e:
        return f"ERROR: {_err_msg(e)}"
    # 2) 近似路径（仅含超越/无理函数时到达；符号/未知函数同样 WARN）
    try:
        v = _FloatEvaluator().visit(tree)
    except _SymbolicError as e:
        return f"WARN: {e}"
    except _UnsupportedFuncError as e:
        return f"WARN: {e}"
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError) as e:
        return f"ERROR: {_err_msg(e)}"
    return f"≈ {v:.15g}"


# ============================================================
# 符号代入求值 + 答案数值化
# ------------------------------------------------------------
# 动机：模型给出的**符号表达式答案**（如 "2*x+y"、"n*(n+1)/2"）无法被程序
# 直接核对——判分器只比字符串。补上"把 givens 代入 expr 求精确值"这一环后，
# 才能把「关系式」与「给定的具体数值」接起来做核验。
# ============================================================
_SUBST_VAR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_SUBST_VAL_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:/\d+)?$")
# 答案数值化：只接受可**精确**判等的形式（整数/小数/分数），
# 近似值（"≈ …"）与符号式（"3*sqrt(5)"、"2*x+y"）一律弃权——宁漏勿误。
_NUM_ONLY_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:/\d+)?$")
_BOXED_PREFIXES = ("\\boxed", "\\fbox")
_ANSWER_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
# 代入求值的超时护栏：每次调用起一个 daemon 线程，join 超时即放弃
# （见 _subst_with_timeout）。已放弃（卡死）线程计数有上限，防无界建线程。
_SUBST_HUNG = 0
_SUBST_HUNG_MAX = 4


def _rename_vars(expr: str, names: list) -> tuple[str, dict]:
    """把表达式里的变量名整体改名为内部安全记号 q0/q1/...。

    为什么必须改名：SymPy 里 `E / I / N / O / S / pi / sin / sqrt ...` 是
    **内建常量或函数**。直接 sympify("E*x") 会把 E 当自然常数 → 代入被静默
    忽略、算出错值（`{"E": 5, "x": 2}` 的 "E*x" 会得 2e 而不是 10）。
    改名为绝不与内建冲突的记号后，代入语义才确定。

    词边界替换：`n` 不会破坏 `n2`（后随数字 → 前瞻失败），故名字顺序无关。
    返回 (改名后的表达式, {原名: 记号})。
    """
    out = expr
    token_of: dict = {}
    used = set(re.findall(r"[A-Za-z0-9]+", expr))
    idx = 0
    for name in names:
        token = f"q{idx}"
        while token in used:
            idx += 1
            token = f"q{idx}"
        used.add(token)
        idx += 1
        pat = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])"
        out = re.sub(pat, token, out)
        token_of[name] = token
    return out, token_of


def _subst_eval_core(expr: str, pairs: list) -> str:
    """safe_eval_subst 的实际执行体（外层有 3s 超时护栏）。"""
    import sympy as sp                      # noqa: PLC0415  延迟导入
    subs = {}
    for name, val in pairs:
        # 统一转精确有理数："0.5" → 1/2（否则 sympify 出 Float，结果退化成
        # "0.250000000000000" 这种浮点串，违背"精确代入"的初衷）。
        fr = Fraction(val)
        subs[sp.Symbol(name)] = sp.Rational(fr.numerator, fr.denominator)
    parsed = sp.sympify(expr)
    # 表达式里的十进制字面量（如 "x**0.5"）同样转有理数，保持全程精确
    try:
        parsed = sp.nsimplify(parsed, rational=True)
    except Exception:  # noqa: BLE001
        pass
    res = parsed.subs(subs)
    free = getattr(res, "free_symbols", None) or set()
    if free:
        return "WARN: 仍有未代入变量 " + ", ".join(sorted(str(s) for s in free))
    try:
        if res.is_real is False:            # zoo / nan / 复数 → 能力外
            return f"WARN: 代入结果不是实数（{sp.sstr(res)[:30]}）"
    except Exception:  # noqa: BLE001
        pass
    # 1) 精确有理数（含 Fraction 化的整数/小数/分数）
    if getattr(res, "is_rational", False):
        num, den = int(sp.numer(res)), int(sp.denom(res))
        return str(num) if den == 1 else f"{num}/{den}"
    # 2) 含 pi/e 等常量 → 15 位近似（与 _symbolic_eval 口径一致）
    try:
        if res.atoms(sp.NumberSymbol):
            return f"≈ {sp.N(res, 15)}"
    except Exception:  # noqa: BLE001
        pass
    # 3) 精确无理式（如 sqrt(5)、3*sqrt(5)）→ 化简后原样返回
    out = sp.sstr(sp.simplify(res))
    if not out or len(out) > 300 or "\n" in out:
        return f"≈ {sp.N(res, 15)}"
    return out


def safe_eval_subst(expr: str, mapping: dict) -> str:
    """把 mapping 里的变量代入 expr 后求**精确**值（自洽核验内核）。

    - expr 可含变量（"2*x+y"、"n*(n+1)/2"、"length*width"、"x^2-1"）；
      变量名须为「字母开头 + 字母数字」；**空 mapping 也合法**（纯数值算式）；
    - mapping 形如 {"x": 10, "y": "5/6"}（值可为数值/数值串）；非法名忽略；
    - 变量名会先整体改名为内部记号再代入（避免 E/I/N/pi/sin 等 SymPy
      内建名把代入静默吃掉 → 算出错值，见 _rename_vars）；
    - 返回与 safe_eval 同构的结果串：
        "25" / "5/6" / "3*sqrt(5)"（精确）· "≈ 3.14…"（含 pi/e）
        "WARN: …"（仍有未代入变量 / 能力外，**不是错误**）
        "ERROR: …"（语法非法 / 取值非数值）

    安全边界与 _symbolic_eval 一致：先过字符白名单（无引号/下划线/方括号
    → sympify 无代码执行面），再交给 3s 超时线程，超时/异常一律回落 WARN。
    """
    expr = "" if expr is None else str(expr)
    expr = expr.strip()
    if not expr:
        return "ERROR: 空表达式"
    if len(expr) > 500:
        return "ERROR: 表达式过长"
    s = _preprocess_implicit_mul(expr.replace("^", "**"))
    if "//" in s or "%" in s:
        return "WARN: 符号整除/取模语义歧义，计算器不支持"
    if not _SYM_SAFE_RE.match(s):
        return "WARN: 表达式含非数学字符（仅支持字母/数字/运算/括号）"
    names, vals = [], {}
    if isinstance(mapping, dict):
        for key, val in mapping.items():
            name = str(key).strip()
            if not _SUBST_VAR_RE.match(name):
                continue           # 非法名忽略（真被用到时会成自由符号 → WARN）
            text = str(val).strip()
            if not _SUBST_VAL_RE.match(text):
                return f"ERROR: 变量 {name} 的取值 {text!r} 不是数值"
            if name not in vals:
                names.append(name)
                vals[name] = text
    # 空 mapping 也合法：纯数值算式（如 "1/2+1/3"）直接精确求值；
    # 表达式里仍有未提供取值的符号时，内核的 free_symbols 检查会回落 WARN。
    renamed, token_of = _rename_vars(s, names)
    pairs = [(token_of[n], vals[n]) for n in names]
    return _subst_with_timeout(renamed, pairs)


def _subst_with_timeout(expr: str, pairs: list) -> str:
    """在 **daemon 线程**里跑代入求值并施加 3s 超时。

    - 用每次调用一个 **daemon** 线程：卡死既不堵后续调用（ThreadPoolExecutor
      的非 daemon worker 卡死会堵死整个池、静默丢失全部后续能力），也不阻塞
      进程退出（非 daemon 线程会让 `_python_exit` join 挂死正常退出）；
    - 已放弃线程数超过上限即整体降级（返回 WARN），避免无界建线程；
    - 异常兜底回到"一律回落 WARN"的契约。
    """
    global _SUBST_HUNG
    if _SUBST_HUNG >= _SUBST_HUNG_MAX:
        return "WARN: 代入求值已降级（历史超时过多）"
    box: dict = {}

    def _run() -> None:
        try:
            box["v"] = _subst_eval_core(expr, pairs)
        except Exception as exc:  # noqa: BLE001  线程内异常回传，不逃逸
            box["e"] = exc

    th = threading.Thread(target=_run, name="calc_subst", daemon=True)
    th.start()
    th.join(_SYM_TIMEOUT_SEC)
    if th.is_alive():                      # 卡死 → 放弃该线程（daemon，不影响退出）
        _SUBST_HUNG += 1
        return "WARN: 代入求值超时"
    if "e" in box:
        return f"WARN: 代入求值未成功（{type(box['e']).__name__}）"
    return box.get("v") or "WARN: 代入求值无结果"


def to_exact_number(text) -> Fraction | None:
    """答案串 → 精确 Fraction；非纯数值一律返回 None（宁漏勿误）。

    接受：  "25" / "-3" / "5/6" / "0.5" / "\\boxed{25}" / "\\frac{1}{2}" / "$25$"
    拒绝：  含变量（"2*x+y"）、精确根式（"3*sqrt(5)"）、近似（"≈ 3.14"）、
            百分数、带单位或文字的长答案 —— 这些无法**精确**判等。
    """
    if text is None:
        return None
    t = str(text).strip()
    if not t or t.startswith("≈"):
        return None
    t = _ANSWER_FRAC_RE.sub(r"\1/\2", t)
    # 只去首尾空白与 $，**不去内部空白**——否则 "1 2" 会被误读成 12
    t = t.strip().strip("$").strip()
    for prefix in _BOXED_PREFIXES:
        if t.startswith(prefix):
            i = t.find("{")
            if i >= 0 and t.endswith("}"):
                t = t[i + 1:-1]
            break
    t = t.strip()
    # 残留反斜杠 = 没被上面规则识别的 LaTeX（如无括号的 "\dfrac12"）→ 直接弃权。
    # 否则剥掉命令后会把 "\dfrac12" 错读成 12。
    if "\\" in t:
        return None
    if not _NUM_ONLY_RE.match(t):
        return None
    try:
        return Fraction(t)
    except Exception:  # noqa: BLE001
        return None

