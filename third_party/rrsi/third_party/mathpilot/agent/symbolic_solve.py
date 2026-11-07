"""符号化方程求解通道：数值剥离 → 符号建模 → 硬校验 → 工具求解（2026-09-12）。

用户需求（9/11）：大模型根据题目逻辑推导出**方程式类型**的答案，具体数值由
智能体调用工具执行计算 —— 智能体把数据存下来，给大模型的是未知数类型，
让大模型推导求解方程式，然后根据公式计算答案。

四段链路（前三段零 LLM、纯本地）：
  ① strip_given_numbers  题面里「显式给定的具体数值」→ 符号 P1,P2…，
     数值存本地参数表；结构性常数（指数 ^2、分数 1/2、倍数、序数）**保留**，
     防止题面被改写成谜语；
  ② 模型只输出 EQUATIONS / TARGET（禁止任何数值结果）—— 见 prompts/symbolic_model；
  ③ validate_payload     三层硬闸门的代码层（非空 / 必含等号 / 字符白名单 /
     target 不得是裸数值）；
  ④ solve_with_tool      数值回代 → SymPy 解方程（组）→ 代入 TARGET → 精确值。

设计原则（与项目既有纪律一致）：
- **纯函数、零 LLM、零副作用**：解析/校验/求解全在本地，毫秒~秒级；
- 失败一律返回 None / (None, 原因)，由调用方**弃权放行**，绝不阻断主链；
- **宁漏勿误**：多解且目标值不唯一、结果非精确数值、超时 → 全部弃权；
- 安全边界与 utils/math_eval._SYM_SAFE_RE 同口径（无引号/下划线/方括号 → sympify
  无代码执行面），并复用 daemon 线程 + 超时护栏（防病态输入卡死主线程）。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field

# 2026-10-01 去重：原此处另有一份与 agent/symbolic_model.py 逐字相同的
# _first_json_object（29 行）。本模块存在两种导入方式（
# 与 ），故用双路径 try/except。
from .symbolic_model import _first_json_object

logger = logging.getLogger("MathPilot.SymbolicSolve")

# --------------------------------------------------------------------------
# 常量与安全边界
# --------------------------------------------------------------------------
# 与 utils/math_eval._SYM_SAFE_RE 同口径：字母/数字/四则/括号/逗号/空格/点；
# **不含**引号 ' "、下划线 _（禁 dunder 属性链）、方括号 []（禁下标）、
# % //（语义歧义）。^ 先规范化为 ** 再检查。
_SAFE_EXPR_RE = re.compile(r"^[0-9a-zA-Z*/().,\-+ ]+$")
_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

_SOLVE_TIMEOUT_SEC = 5.0
_HUNG_MAX = 20
_HUNG = 0                      # 累计超时次数；超上限整体降级（防无界建线程）

# ① 赋值型「显式给定参数」：设 n = 100 / 令 x=5 / 已知 k=3 …
_ASSIGN_RE = re.compile(
    r"(?:设|令|已知|其中|当|取|若|且)\s*"
    r"(?P<var>[A-Za-z][A-Za-z0-9]{0,3})\s*(?:=|＝)\s*"
    r"(?P<num>-?\d+(?:\.\d+)?(?:/\d+)?)")

# ② 数量型「显式给定数据」：60 个 / 2024 元 / 3 组 / 1.5 米
_UNITS = ("分钟|小时|千米|公里|厘米|毫米|毫升|千克|平方米|立方米|"
          "个|件|元|条|组|次|种|名|人|岁|米|秒|天|年|页|本|张|颗|箱|袋|升|克|吨|度")
_QUANT_RE = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?P<unit>" + _UNITS + r")")

# ③ 中文叙述型「显式给定数据」：「首项为 3」「公差是 4」「半径等于 2」
# 要求"为/是/等于"后面紧跟数字 —— 故"当 n 为奇数时"（无数字）不会被误伤。
_COPULA_RE = re.compile(
    r"(?:为|是|等于)\s*(?P<num>-?\d+(?:\.\d+)?(?:/\d+)?)")

# ④ 无前置词赋值：`n=2024` / `\(x=5\)` —— 数学题里"给定参数"的最常见形态。
# 变量名限 1~3 位（避免吞掉正文里的普通词），要求前面不是字母数字。
_BARE_ASSIGN_RE = re.compile(
    r"(?P<var>(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{0,2})\s*(?:=|＝)\s*"
    r"(?P<num>-?\d+(?:\.\d+)?(?:/\d+)?)")

# ⑤ 裸大数（整数部分 ≥3 位）：`2024` / `10000` —— 数学题里的"数据"。
# 排除"结构位"：指数（前邻 ^ ** {）、分母/乘数（前邻 / *）、小数内部（前邻数字或点）；
# 排除小数尾部（后邻数字或点）——如 "0.01" 不会被切成 "01"。
_BIG_NUM_RE = re.compile(r"(?<![\d.\^/*{])(?P<num>\d{3,})(?![\d.])")

# ③ 模型输出协议
_UNSUPPORTED_RE = re.compile(r"\bUNSUPPORTED\b", re.I)
_EQ_BLOCK_RE = re.compile(
    r"EQUATIONS?\s*[:：]\s*(.*?)(?=\n\s*TARGET\s*[:：]|\Z)", re.S | re.I)
_TARGET_RE = re.compile(r"TARGET\s*[:：]\s*(.+)", re.I)

_BARE_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:/\d+)?$")
# 整数形式的 ≥3 位字面量（"0.125" 因前邻 . 不算，"1.5" 仅有 1 位）
_LITERAL_RE = re.compile(r"(?<![\d.])\d{3,}(?!\d)")


@dataclass
class StripResult:
    """数值剥离结果。"""

    problem: str                    # 符号化题面（数值已被 P1,P2… 替换）
    params: dict = field(default_factory=dict)   # P1 -> "100"（本地保存，不给模型）
    n_hits: int = 0
    note: str = ""


# ==========================================================================
# ① 数值剥离器
# ==========================================================================
def _left_context_bad(text: str, pos: int) -> bool:
    """裸赋值左侧是否是**数学结构内部**（说明它不是"给定参数"）。

    两类必须拒绝：
      ① 方程式内部：`py^3 + qy^2 + ry + s = 0` 里的 `s = 0` —— 跳过空白后
         左邻是运算符，属于方程右端定义，**绝不能**把 0 剥离（会破坏方程语义）；
      ② 数学结构内部（2026-09-12 修）：`\\sum_{j=1}^n` / `\\sum_{i,j=1}^n` /
         `\\int_{0}^{1}` / `\\{x : x=1\\}` 里的下标、上下限、集合构造 ——
         左邻是 `{` `[` `,` `:` 之一。**097 实况**：`\\sum_{i,j=1}^n` 的求和下标
         `1` 曾被剥成 `P1`，直接把题目的算子定义改坏。
    """
    i = pos - 1
    while i >= 0 and text[i] in " \t\n\r":
        i -= 1
    return i >= 0 and text[i] in "+-*/^({[,:"


def strip_given_numbers(problem, *, max_params: int = 12) -> StripResult | None:
    """把题面「显式给定的具体数值」换成符号 P1,P2…，数值存本地参数表。

    只替换**属于题目数据**的数字：
      - 赋值式：`设 n = 100` / `令 x=5`（前置词）
      - 数量式：`60 个` / `2024 元` / `3 组`（量词）
      - 叙述式：`首项为 3` / `公差是 4`（"为/是/等于"+数字）
      - 裸赋值：`n=2024` / `\\(x=5\\)`（数学题最常见形态；方程式内部除外）
      - 裸大数：整数部分 ≥3 位（`2024`、`10000`）
    结构性常数（指数 `^2`/`^{19}`、分数 `1/2`、乘数、序数、方程右端 0）**保留**
    —— 否则题面会被改写成谜语，模型必然建错模。

    同一数值复用同一符号（避免同值异名造成建模歧义）。
    命中数 > max_params → 判定不适用。无命中/异常 → 返回 None（调用方弃权放行）。
    """
    try:
        if not problem or not str(problem).strip():
            return None
        text = str(problem)

        spans: list[tuple[int, int, str]] = []
        for pat in (_ASSIGN_RE, _QUANT_RE, _COPULA_RE, _BIG_NUM_RE):
            for m in pat.finditer(text):
                spans.append((m.start("num"), m.end("num"), m.group("num")))
        for m in _BARE_ASSIGN_RE.finditer(text):
            if _left_context_bad(text, m.start("var")):
                continue                       # 方程式内部 → 不是给定参数
            spans.append((m.start("num"), m.end("num"), m.group("num")))
        if not spans:
            return None

        # 去重叠（按起点排序，先到先得）
        spans.sort()
        picked: list[tuple[int, int, str]] = []
        last_end = -1
        for s, e, num in spans:
            if s < last_end:
                continue
            picked.append((s, e, num))
            last_end = e

        if len(picked) > max_params:
            logger.debug("[symbolic_solve] 可剥离数值 %d 个 > %d，判不适用",
                         len(picked), max_params)
            return None

        # 同一数值复用同一符号（倒序替换，避免破坏前面的偏移）
        num_to_name: dict = {}
        for _s, _e, num in picked:
            if num in num_to_name:
                continue
            name = f"P{len(num_to_name) + 1}"
            while re.search(r"(?<![A-Za-z0-9])" + name + r"(?![A-Za-z0-9])", text):
                name = "Q" + name
            num_to_name[num] = name
        params: dict = {name: num for num, name in num_to_name.items()}
        out = text
        for s, e, num in reversed(picked):
            out = out[:s] + num_to_name[num] + out[e:]

        return StripResult(problem=out, params=params, n_hits=len(picked))
    except Exception as exc:  # noqa: BLE001  任何异常都不阻断主链
        logger.debug("[symbolic_solve] 数值剥离异常: %s", str(exc)[:100])
        return None


# ==========================================================================
# ② 协议解析
# ==========================================================================


def _split_equations(block: str) -> list:
    """块内容 → 方程列表（按换行与中文/英文分号切分，保留含等号者）。"""
    parts: list[str] = []
    for line in str(block).splitlines():
        for piece in re.split(r"[;；]", line):
            piece = piece.strip().strip(",").strip()
            if piece:
                parts.append(piece)
    return [p for p in parts if "=" in p]


def parse_symbolic_solve(text) -> dict | None:
    """模型输出 → {"equations": [...], "target": str, "unsupported": bool}。

    容错优先级：①整段 JSON ②抠平衡 `{}` 再 loads ③紧凑块
    `EQUATIONS: … TARGET: …` ④行扫描（含等号的行）。
    异常一律吞掉返回 None（不阻断主链）。
    """
    try:
        if text is None or not str(text).strip():
            return None
        t = str(text).strip()
        if t.startswith("```"):                       # 去 markdown 围栏
            t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
            t = re.sub(r"\s*```\s*$", "", t)

        # 弃权出口：整段只有 UNSUPPORTED
        if _UNSUPPORTED_RE.search(t) and "=" not in t:
            return {"equations": [], "target": "", "unsupported": True}

        # ① / ② JSON
        for cand in (t, _first_json_object(t)):
            if not cand or not cand.lstrip().startswith("{"):
                continue
            try:
                obj = json.loads(cand)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and any(
                    k in obj for k in ("equations", "target", "unsupported")):
                eqs = obj.get("equations") or []
                if isinstance(eqs, str):
                    eqs = _split_equations(eqs)
                elif isinstance(eqs, list):
                    eqs = [str(x).strip() for x in eqs if str(x).strip()]
                else:
                    eqs = []
                return {
                    "equations": eqs,
                    "target": str(obj.get("target") or "").strip(),
                    "unsupported": bool(obj.get("unsupported")),
                }

        # ③ 紧凑块
        em = _EQ_BLOCK_RE.search(t)
        tm = _TARGET_RE.search(t)
        if em or tm:
            eqs = _split_equations(em.group(1)) if em else []
            target = tm.group(1).strip().strip("。.") if tm else ""
            if eqs or target:
                return {"equations": eqs, "target": target,
                        "unsupported": False}

        # ④ 行扫描兜底
        eqs = [ln.strip() for ln in t.splitlines()
               if "=" in ln and not ln.strip().upper().startswith("TARGET")]
        if eqs:
            return {"equations": eqs, "target": "", "unsupported": False}
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("[symbolic_solve] 协议解析异常: %s", str(exc)[:100])
        return None


# ==========================================================================
# ③ 三层硬闸门的代码层
# ==========================================================================
def _expr_safe(expr: str) -> bool:
    """字符白名单检查（^ 先规范化为 **）。"""
    s = str(expr or "").replace("^", "**").strip()
    return bool(s) and bool(_SAFE_EXPR_RE.match(s))


def validate_payload(payload, params=None) -> tuple[bool, str]:
    """校验建模结果是否合格 → (ok, 具体违规原因)。

    「强硬关键词」的**代码层**：不合格必须给出可回传的具体原因，
    由调用方带原因重试一次，而不是放行。
    """
    if not isinstance(payload, dict):
        return False, "未解析出结构化建模结果"
    if payload.get("unsupported"):
        return False, "模型声明该题无法符号建模（UNSUPPORTED）"

    eqs = payload.get("equations") or []
    target = str(payload.get("target") or "").strip()
    if not eqs and not target:
        return False, "EQUATIONS 与 TARGET 均为空"

    # 「模型不参与计算」的代码级保证：题面里被剥离的数值都存进本地参数表，
    # 因此模型输出的方程**不该出现题面未给定的大数**（≥3 位）。出现即视为
    # 自行代入/心算，直接打回。结构性常数（2、10、1/2 等 1-2 位数）不受影响。
    allowed = {str(v).strip() for v in (params or {}).values()}

    def _bad_literal(line: str) -> str | None:
        for num in _LITERAL_RE.findall(line):
            if num not in allowed:
                return num
        return None

    for i, line in enumerate(eqs, 1):
        line = str(line).strip()
        if not line:
            continue
        if line.count("=") != 1:
            return False, (f"第 {i} 个方程必须且只能含一个等号，"
                           f"实际为 {line!r}")
        left, right = line.split("=", 1)
        if not _expr_safe(left) or not _expr_safe(right):
            return False, (f"第 {i} 个方程含非数学字符或为空：{line!r}"
                           "（只允许字母/数字/四则/括号）")
        if _BARE_NUMBER_RE.match(left.strip()) and _BARE_NUMBER_RE.match(
                right.strip()):
            return False, f"第 {i} 个方程两侧都是常量，不是关系式：{line!r}"
        bad = _bad_literal(line)
        if bad:
            return False, (f"第 {i} 个方程出现题面未给定的数值 {bad}"
                           "——题目数据必须用符号（P1、P2…）代替，"
                           "禁止自行计算或代入数值")

    if target:
        if not _expr_safe(target):
            return False, f"TARGET 含非数学字符：{target!r}"
        if _BARE_NUMBER_RE.match(target):
            return False, (f"TARGET 是裸数值 {target!r}"
                           "——疑似已自行计算，禁止；请给出表达式或变量名")
        bad = _bad_literal(target)
        if bad:
            return False, (f"TARGET 出现题面未给定的数值 {bad}"
                           "——请用符号表达目标量")

    return True, "ok"


# ==========================================================================
# ④ 本地工具求解（SymPy）
# ==========================================================================
def _solve_core(equations: list, target: str, params: dict) -> tuple[str | None, str]:
    """实际求解（外层负责超时）。返回 (精确值串, 说明)。"""
    import sympy as sp                                  # 延迟导入

    # 断言 + TARGET 里出现的所有标识符 → 强制映射为普通 Symbol
    # （否则 E/pi/sin 等 SymPy 内建名会把符号静默吃掉，见 math_eval._rename_vars 同款教训）
    idents: set = set()
    for line in equations:
        idents.update(_IDENT_RE.findall(str(line)))
    idents.update(_IDENT_RE.findall(target))
    ns = {name: sp.Symbol(name) for name in idents}

    subs = {}
    for key, val in (params or {}).items():
        name = str(key).strip()
        text = str(val).strip()
        if not _IDENT_RE.fullmatch(name) or not _BARE_NUMBER_RE.match(text):
            return None, f"参数 {name}={text!r} 非法"
        subs[sp.Symbol(name)] = sp.Rational(text)

    eqs = []
    for line in equations:
        line = str(line).strip().replace("^", "**")
        if not line:
            continue
        if "=" in line:
            left, right = line.split("=", 1)
            expr = sp.sympify(f"({left})-({right})", locals=ns)
        else:
            expr = sp.sympify(line, locals=ns)
        eqs.append(sp.expand(expr).subs(subs))

    if not target:
        return None, "无 TARGET，无法确定目标量"
    tgt = sp.sympify(target.replace("^", "**"), locals=ns).subs(subs)

    unknowns = sorted({s for e in (eqs + [tgt]) for s in e.free_symbols},
                      key=str)
    if not unknowns:
        val = sp.simplify(tgt)
    else:
        if not eqs:
            return None, "缺少方程，无法求解未知量"
        sols = sp.solve(eqs, unknowns, dict=True)
        if not sols:
            return None, "方程组无解"
        if len(sols) > 1:
            # 多解：仅当各解下 TARGET 取值唯一才可用，否则弃权（宁弃不猜）
            vals = set()
            for s in sols:
                try:
                    vals.add(sp.simplify(tgt.subs(s)))
                except Exception:  # noqa: BLE001
                    return None, f"多解（{len(sols)} 组）且目标值无法比较"
            if len(vals) != 1:
                return None, f"多解（{len(sols)} 组）且目标值不唯一"
            val = vals.pop()
        else:
            val = sp.simplify(tgt.subs(sols[0]))

    text = sp.sstr(val)
    if getattr(val, "free_symbols", None):
        return None, f"结果仍含符号：{text}"

    exact = None
    try:
        from utils.math_eval import to_exact_number
    except ImportError:
        to_exact_number = None

    if to_exact_number is not None and to_exact_number(text) is not None:
        exact = text
    else:
        try:
            frac = sp.Rational(val)                     # 无理数会抛异常
            exact = str(frac.p) if frac.q == 1 else f"{frac.p}/{frac.q}"
        except Exception:  # noqa: BLE001
            return None, f"结果非精确数值：{text}"

    detail = ("；".join(equations) if equations else target)
    return exact, f"{detail}（代入 {params}）→ {exact}"


def solve_with_tool(payload, params) -> tuple[str | None, str]:
    """数值回代 → SymPy 解方程（组）→ 代入 TARGET → 精确值串。

    返回 (精确值, 说明)；失败返回 (None, 原因)。全程 daemon 线程 + 超时护栏。
    """
    global _HUNG
    if _HUNG >= _HUNG_MAX:
        return None, "符号求解已降级（历史超时过多）"
    if not isinstance(payload, dict):
        return None, "建模结果结构异常"
    try:
        equations = [str(x).strip() for x in (payload.get("equations") or [])
                     if str(x).strip()]
        target = str(payload.get("target") or "").strip()
    except Exception:  # noqa: BLE001
        return None, "建模结果结构异常"

    box: dict = {}

    def _run() -> None:
        try:
            box["r"] = _solve_core(equations, target, params or {})
        except Exception as exc:  # noqa: BLE001  线程内异常回传，不逃逸
            box["e"] = exc

    th = threading.Thread(target=_run, name="sym_solve", daemon=True)
    th.start()
    th.join(_SOLVE_TIMEOUT_SEC)
    if th.is_alive():                    # 卡死 → 放弃该线程（daemon 不阻塞退出）
        _HUNG += 1
        return None, "符号求解超时"
    # 2026-09-12 修复：成功时让累计超时计数**衰减**。原实现只增不减，跨题累计
    # 满 _HUNG_MAX(20) 后该通道对本进程**永久降级**——后段题目静默失去符号求解
    # 能力，且没有任何日志可察（属"跨题状态污染"，与并发=3 的批次场景叠加后果更重）。
    if _HUNG:
        _HUNG -= 1
    if "e" in box:
        return None, f"符号求解异常（{type(box['e']).__name__}）"
    return box.get("r") or (None, "符号求解无结果")
