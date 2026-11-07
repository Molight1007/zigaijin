# -*- coding: utf-8 -*-
"""4.4 数学工具计算检测 —— 五道审核链第 4 道（2026-10-14 用户定案）。

用户口径（原话）：「第四道就是利用数学工具进行计算检测，看看计算步骤有没有问题」。

实现口径（加模块优先，复用既有数学工具内核）：
  · 后端 = `agent/deterministic.DeterministicChecker.check_answer`（SymPy
    精确判定：答案可解析性 / 常量等式全量回带 / 函数方程回代 / 反例搜索
    —— 均为**可机检**判定，非 LLM 自评），**不新写第二个计算器**；
  · 输入优先取 4.0 的 `answer_norm`（规范化的意义就在于此）；无则回退原文。

判定纪律（沿用 AuditGate Level0 既有实证口径）：
  · 单候选 `verdict == "fail"` ⇒ 计算检测不通过 ⇒ **淘汰**（机器判定，非自评）；
  · `unknown`（不可解析 / 超时 / sympy 不可用）⇒ **放行**（宁 unknown 不误杀）；
  · **全拒** ⇒ 回退保留原候选池（与 `AuditGate.audit_candidates` 的
    「全部候选被否 → 回退保留（宁 unknown 不误杀）」逐字一致），显式留痕。

观测：trace step=``math_check``，diag 键 ``math_check`` / ``math_check_events``。

五道链位置：4.0 规范化 → 4.1 自审 → 4.2 对抗证伪 → 4.3 Lean 代回 →
**4.4 计算检测** → ⑤投票。
"""
from __future__ import annotations

import re

from .switch_registry import get_bool as _sw_get_bool

def _switch_enabled() -> bool:
    """读「五道链·第 4 道计算检测」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_math_check")
    except Exception:  # noqa: BLE001
        return True


def _calcverify_enabled() -> bool:
    """读「4.4 工具③等式复算观测通道」开关（默认 False，云端 A/B 时开）。

    ★ 2026-10-15（工具化定稿）：开启后对**存活候选的推理**提取"数值等式"
    （如 ``19.9 + (10 - sqrt(9.9))**2 = 20``），逐条调工具③
    ``tools/agent_toolkit.calc_verify`` 机械复算；算错的条目写 diag
    （``ctx.math_calcverify``）与 trace —— **观测模式：暂不改淘汰**。
    """
    try:
        return _sw_get_bool("five_gate_math_check_calcverify")
    except Exception:  # noqa: BLE001
        return False


# ============================================================ 提取层（2026-10-21 重写）
# ★ 2026-10-21（mc075_r1 归因，五条原文全部在真实文本上复现）：4.4 把**正确答案**
#   `\boxed{P(x)=x-4}` 判死的根因全在提取层，不在复算器、更不在阈值：
#     ① 链式等式 `a=b=c=d` 被切成错位片段 —— `$D(1)=0!-6(-1)^0=1-6=-5$`
#        被提成 `('6(-1)^0','1-6')` ⇒ 6 ≠ -5 ⇒ 判「算错」。数学写作常态，主因；
#     ② XML 标签被当不等式 —— `</check>` 里的 `<` 被当小于号；
#     ③ LaTeX 命令截断右值 —— `12=2^2\cdot3` 只取到 `2`（`\` 不在字符类）。
#   修法（真实实现，非降级）：先把数学文本归一化成纯算式，再**以关系运算符为
#   锚点双向扩段**。链式 `a=b=c=d` 的每个 `=` 天然构成一对相邻段 ⇒ 变成逐对
#   判等（等价于整链验证）。既不跳过链式（跳过会丢掉证明里最容易算错的算术），
#   也不动淘汰阈值（放宽门槛只会连带放过真错误）。
_OP_NORMALIZE = (("×", "*"), ("·", "*"), ("⋅", "*"), ("÷", "/"),
                 ("−", "-"), ("–", "-"), ("—", "-"))
_ALLOWED_FUNCS = ("sqrt", "abs", "floor", "ceil", "log", "exp", "sin",
                  "cos", "tan", "pi", "ln", "pow", "e", "factorial", "Abs")

# 只剥**已知标签名**：`a<b>c` 这类真不等式不该被当标签。
_XML_TAGS = ("check", "calc", "answer", "step", "solution", "proof", "result",
             "note", "math", "text", "boxed", "b", "i", "em", "strong", "code")
_XML_TAG_RE = re.compile(r"</?(?:%s)\b[^<>]*?/?>" % "|".join(_XML_TAGS), re.I)
# 中文/全角标点 → 哨兵 `\x00`（**不是空格**：空格是段内合法字符，换成空格会把
# 两个中文句子之间的算式粘成一段 —— 旧实现跨标点粘连正是这么来的）。
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+")
# ★ 2026-10-22：**否定语境**标记（与中文哨兵 `\x00` 分开）。
#   mc075 实测原文 `$1!=-6(-1)=6$ 不成立` —— 模型在**否定**这条等式，
#   字面真假方向与机械复算相反。旧实现把它当"声称成立"判 bad ⇒ 误杀。
#   机械复算无法表达否定语义，故打标记后由 `_relations` 丢弃该段
#   （宁可少一条证据，不可误判 —— 与 `_looks_numeric` 拒收同一哲学）。
_NEG_MARK = "\x01"
_NEG_RE = re.compile(r"不成立|不相等|矛盾|显然错误|不可能|不满足|不成立")
_LATEX_SIMPLE = ((r"\left", " "), (r"\right", " "), (r"\!", " "), (r"\,", " "),
                 (r"\;", " "), (r"\:", " "), (r"\quad", " "), (r"\qquad", " "),
                 (r"\cdot", "*"), (r"\times", "*"), (r"\ast", "*"),
                 (r"\div", "/"), (r"\leq", "<="), (r"\le", "<="),
                 (r"\geq", ">="), (r"\ge", ">="), (r"\neq", "<>"),
                 (r"\ne", "<>"), (r"\pi", "pi"), (r"\infty", "inf"),
                 (r"\ln", "log"))


def _normalize_math(text: str) -> str:
    """把含 LaTeX / XML / 中文的推理文本归一化成「只剩算式与关系符」的串。"""
    s = str(text or "")
    s = _XML_TAG_RE.sub(" ", s)                      # ① 剥 <check>…</check>
    s = s.replace("$", " ")
    for _dl in ("\\(", "\\)", "\\[", "\\]"):
        s = s.replace(_dl, " ")
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"(\1)", s)
    s = re.sub(r"\\text\s*\{[^{}]*\}", " ", s)
    for _a, _b in _LATEX_SIMPLE:
        s = s.replace(_a, _b)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)       # ② 2^{10} → 2^(10)
    s = re.sub(r"_[a-zA-Z0-9]+", "", s)              # ③ 下标当拼接：a_1 → a
    # ③' `!` 的二义性（2026-10-22）：
    #    `(k-1)!=6` / `2!=2` / `0!=1` 是「阶乘 + 等号」，ASCII `!=` 才是真不等号。
    #    `\neq` / `\ne` 已在上一步归一成 `<>`，不再污染 `!`。
    #    ★ 旧实现把 `\neq` 归一为 `!=`，而 `!` 属于段字符集 ⇒
    #      `6\neq 1!=1` 的右值被扩成 `6!`（多吞一个阶乘号），把正确的
    #      `6(-1)^2=6` 判成 `6(-1)^2=6!` ⇒ 6 vs 720 误杀整条候选。
    s = re.sub(r"(?<=[\d)])!=", "! =", s)             # 阶乘后紧接等号 → 拆开
    s = s.replace("!=", "<>")                         # 其余 `!=` 才是不等号
    s = re.sub(r"(\d+)!", r"factorial(\1)", s)       # ④ 0! → factorial(0)
    for _u, _a in _OP_NORMALIZE:
        s = s.replace(_u, _a)                         # ⑤ × ÷ − 等 → ASCII
    s = re.sub(r"\|([^|]{1,40})\|", r"Abs(\1)", s)    # ⑥ |q-k| → Abs(q-k)
    s = _NEG_RE.sub(_NEG_MARK, s)                     # ⑦ 否定词 → `\x01`（先于中文折叠）
    s = _CJK_RE.sub("\x00", s)                        # ⑧ 中文 → 隔断哨兵
    # ⑨ 否定标记与四周中文折叠成一个分隔符：`=6 \x00\x01\x00 ；` → `=6 \x01`。
    #    这样 `\x01` 紧贴被否定的那句，`_neg_ranges` 才能圈出整句范围。
    return re.sub(r"\x00*\x01\x00*", _NEG_MARK, s)


# 段字符集：空格在内（`1 - 6` 合法）；中文已变哨兵，故不会跨句。
_OPERAND_CHARS = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._+-*/^()|! ")
_REL_RE = re.compile(r">=|<=|<>|!=|==|≥|≤|=|<|>")
_REL_OPS = {">=": "ge", "<=": "le", "≥": "ge", "≤": "le",
            ">": "gt", "<": "lt", "=": "eq", "==": "eq"}
_LEAD_BAD = "*/^)!>="      # 不能作段首（`-` 可以：负数）
_TAIL_BAD = "+-*/^<(="     # 不能作段尾（`!` `)` 可以：阶乘/右括号）


def _operand_left(s: str, end: int) -> str:
    i = end - 1
    while i >= 0 and s[i] in _OPERAND_CHARS:
        i -= 1
    return s[i + 1:end]


def _operand_right(s: str, start: int) -> str:
    i = start
    while i < len(s) and s[i] in _OPERAND_CHARS:
        i += 1
    return s[start:i]


def _trim_operand(seg: str) -> str:
    """削掉段首尾的悬挂运算符（`= 1 +` / `* 3`），保留负号与前导括号。"""
    s = (seg or "").strip()
    while s and s[0] in _LEAD_BAD:
        s = s[1:].lstrip()
    while s and s[-1] in _TAIL_BAD:
        s = s[:-1].rstrip()
    return s


def _neg_ranges(s: str) -> list:
    """否定语境覆盖的字符区间 `[(start, end)]`（`end` 为 `\\x01` 的位置）。

    ★ 否定作用于**整句**而非单个关系符：`1!=-6(-1)=6 不成立` 里链式被拆成
      两对（`1!=-6(-1)` / `-6(-1)=6`），两对都该丢弃。只判"右值后紧跟否定"
      会漏掉链中间那对（2026-10-22 实测：修完仍残留 `factorial(1)=-6(-1)`）。
    ★ 区间左界取上一个分隔符（`\\x00` 或 `\\x01`），即"被否定的那句话"。
    """
    out = []
    for m in re.finditer(re.escape(_NEG_MARK), s):
        pos = m.start()
        l = max(s.rfind("\x00", 0, pos), s.rfind(_NEG_MARK, 0, pos))
        out.append((l + 1, pos))
    return out


def _relations(text: str) -> list:
    """扫描所有关系运算符，返回 [(lhs, op, rhs)]，op ∈ {eq, ge, le, gt, lt}。

    ★ 这是整条提取链的**唯一分段点**：段由运算符向两侧扩展得到，因此
      `a=b=c=d` 天然产出 `(a,b) (b,c) (c,d)` 三对相邻段 —— 链式等式变成
      逐对判等，不需要任何「链式特判」。
    ★ 2026-10-22：处在**否定语境**的关系（右值后紧接 `\x01`）整条丢弃，
      原因见 `_NEG_MARK` 注释。丢弃会少一条证据，但不会误判算错。
    """
    s = _normalize_math(text)
    out = []
    _negs = _neg_ranges(s)
    for m in _REL_RE.finditer(s):
        op = _REL_OPS.get(m.group(0))
        if not op:
            continue                 # `<>` / `!=` 不是可复算关系（且会被误当 `=`）
        if any(l <= m.start() < r for l, r in _negs):
            continue                 # 否定语境：字面真假相反，机械复算无法表达
        lhs = _trim_operand(_operand_left(s, m.start()))
        rhs = _trim_operand(_operand_right(s, m.end()))
        if lhs and rhs:
            out.append((lhs, op, rhs))
    return out


def _looks_numeric(s: str) -> bool:
    """去掉白名单函数名后不得残留裸字母（否则疑似含变量 ⇒ 拒收）。"""
    t = s
    for fn in sorted(_ALLOWED_FUNCS, key=len, reverse=True):
        t = t.replace(fn, " ")
    return not re.search(r"[A-Za-z]", t)


def _extract_numeric_equations(text: str) -> list:
    """从推理文本提取数值等式 [(lhs, rhs), ...]（最多 8 条）。

    链式 `a=b=c=d` 的每个 `=` 各成一对 ⇒ 相邻段逐对判等（= 整链验证）。
    含裸变量的段（`x` / `D(1)`）按 `_looks_numeric` 拒收 —— `x = 3` 是赋值，
    不是可机械复算的算术。**拒收只会少一条证据，不会误判算错。**
    """
    out, seen = [], set()
    for lhs, op, rhs in _relations(text):
        if op != "eq":
            continue
        if not (_looks_numeric(lhs) and _looks_numeric(rhs)):
            continue
        key = (lhs, rhs)
        if key in seen:
            continue
        seen.add(key)
        out.append((lhs, rhs))
        if len(out) >= 16:
            break
    return out


_LOOKS_STRONG = re.compile(r"[\^*/]|sqrt|abs|frac|factorial", re.I)
_CALL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\s*\(")


def _has_undef_call(seg: str) -> bool:
    """段内是否含**未定义函数应用**（`P(q)` / `f(x)` / `g(a+b)`）。

    ★ 2026-10-22（mc075 实测）：`Abs(P(q)) > q` 被提取后，sympy 把 `P(q)`
      解析成乘法 `P*q`，代值必然找出"反例"⇒ 判不等式不成立 ⇒ 误杀
      正确候选。`P` 是函数符号，乘法语义完全不同，代值结论无意义。
    ★ 白名单函数（`sqrt/Abs/factorial/...`）先中立化再查，故
      `Abs(q+b)`、`factorial(7)` 不受影响；`x^2-1 >= 0` 无调用 ⇒ 保留
      （017 型符号不等式的检出能力不降）。
    """
    t = seg
    for fn in sorted(_ALLOWED_FUNCS, key=len, reverse=True):
        # ★ 不加 `\b`：`_normalize_math` 会把 `4|a||b|^3` 折成 `4Abs(a)Abs(b)^3`，
        #   `4` 与 `A` 之间没有词边界，`\bAbs` 匹配不到 ⇒ 白名单失效 ⇒
        #   017 型符号不等式被误判成"未定义函数应用"而拒收（2026-10-22 实测）。
        t = re.sub(r"%s\s*\(" % re.escape(fn), " (", t)
    return bool(_CALL_RE.search(t))


def _extract_inequalities(text: str) -> list:
    """从推理文本提取**可机器复算的不等式** [(lhs, rhs, op), ...]（最多 8 条）。

    ★ 2026-10-17（用户要求「计算错误要靠工具检」）：原工具只验数值等式，
      017 型错误（``a^2+3b^2 >= 4|a||b|^3`` 为假）不在检测范围。
    ★ 安全边界（防误杀题面条件）：**任一侧须含强运算符/函数**
      （`^ * / sqrt abs frac factorial`）；`n >= 2` / `x >= -1` 这类纯条件
      两侧都没有 ⇒ 不提取。
    ★ 2026-10-21：段改由 `_relations` 双向扩展得到，不再跨标点粘连。旧实现在
      `'… 2 + 2 = 5，另 x^2-1 >= 0'` 上会把上一句右值粘成 `('5  x^2-1','0','ge')`
      —— 那是把两个独立命题焊成一个假命题。
    """
    out = []
    for lhs, op, rhs in _relations(text):
        if op == "eq":
            continue
        if len(lhs) > 70 or len(rhs) > 70:
            continue
        # 含未知函数应用（`P(q)`）⇒ sympy 会当乘法，代值结论无意义
        if _has_undef_call(lhs) or _has_undef_call(rhs):
            continue
        # 任一侧含强运算符/函数即可（另一侧允许纯数字/变量，如 `x^2-1 >= 0` 的 `0`）
        if not (_LOOKS_STRONG.search(lhs) or _LOOKS_STRONG.search(rhs)):
            continue
        # 两侧都没有强运算符时还限长，防止把一整句话当式子
        if not (_LOOKS_STRONG.search(lhs) and _LOOKS_STRONG.search(rhs)):
            if len(lhs) > 24 or len(rhs) > 24:
                continue
        out.append((lhs, rhs, op))
        if len(out) >= 16:
            break
    return out


# --------------------------------------------------------------- 题目前提
# ★ 2026-10-21：没有前提就分不清「恒假断言」和「有条件才真的命题」。
#   `q^2-1 >= 0`（前提 q>=1）会被 q=1/2 判成假；`x^(d-1) >= x`（前提 d>=2）
#   会被 d=1 判成假 —— 两者在 mc075 真实文本里都出现过（后者 3 次）。
#   前提的形态恰好是 `_extract_inequalities` **拒收**的那类：两侧都没有
#   强运算符（`^ * / sqrt`）的纯条件（`d >= 2` / `0 < x <= 1`）。
_COND_MAX = 12
_COND_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_COND_ID_RE = re.compile(r"^[A-Za-z]$")


def _cond_side(seg: str, side: str) -> str:
    """把条件一侧修剪成**单个数学符号**（单字母变量或数字）。

    ★ 英文题干的段会把散文一起吞进来（实测 `let $d \\ge 2$` 提成
      `('let  d', '2  be its degree')`）—— 约束就永远不生效。数学里前提的
      操作数几乎都是单字母/数字，故只认这两种形态；`d-1` 这类带运算的段
      会被拒（那是命题，不是前提）。
    """
    s = (seg or "").strip()
    if not s:
        return ""
    if _COND_NUM_RE.match(s) or _COND_ID_RE.match(s):
        return s
    if side == "lhs":
        m = re.search(r"(?:^|[\s(,])(-?\d+(?:\.\d+)?|[A-Za-z])$", s)
        return m.group(1) if m else ""
    m = re.match(r"(-?\d+(?:\.\d+)?|[A-Za-z])(?=$|[\s),.])", s)
    return m.group(1) if m else ""


def _extract_conditions(text: str) -> list:
    """从题干提取**数值前提** [{"expr","op","claimed"}]，供代值时过滤样本点。

    与 `_extract_inequalities` 严格互斥：那边收「带运算的命题」，这边收
    「无运算的条件」。`0 < x <= 1` 会被切成 `0<x` 与 `x<=1` 两条（同时满足）。
    """
    out = []
    for lhs, op, rhs in _relations(text):
        if op == "eq":
            continue
        if _LOOKS_STRONG.search(lhs) or _LOOKS_STRONG.search(rhs):
            continue                      # 带运算 ⇒ 是命题，不是前提
        a, b = _cond_side(lhs, "lhs"), _cond_side(rhs, "rhs")
        if not a or not b:
            continue
        out.append({"expr": a, "op": op, "claimed": b})
        if len(out) >= _COND_MAX:
            break
    return out


# --------------------------------------------------------------- 证据快照
# ★ 2026-10-21：med2_r1 上 4.4 把 13 个候选全清空，但 trace 只留判词、
#   candidate.reasoning 不落盘 ⇒ 事后无法区分「真算错」和「提取误伤」。
#   本快照把**提取到的每一条式子 + 逐条复算结论**落进 diag，让下一轮能直接归因。
_SNAP_ITEMS_MAX = 16         # 每候选取前 N 条（等式上限 8 + 不等式上限 8）
_SNAP_TEXT_MAX = 100         # 单条表达式/实际值截断
_SNAP_DETAIL_MAX = 160       # 单条判词截断
_SNAP_CANDS_MAX = 24         # 单题最多落盘候选数（防 diag 爆）


def _build_calc_snapshot(cid, kind, items, rv) -> dict:
    """把一次工具③复算收敛成可落盘快照（**只观测，不改判定**）。

    三个计数语义（与 `tools/agent_toolkit/calc_verify` 严格一致）：
      · `n_ok`  —— 一致；
      · `n_bad` —— **真实不一致**（唯一触发淘汰的计数）；
      · `n_err` —— 无法判定（解析失败 / 含自由符号）—— **不淘汰**。
    """
    rows = []
    for i, r in enumerate(rv.get("results") or []):
        it = items[i] if i < len(items) else {}
        rows.append({
            "expr": str(it.get("expr") or "")[:_SNAP_TEXT_MAX],
            "claimed": str(it.get("claimed") or "")[:_SNAP_TEXT_MAX],
            "op": str(it.get("op") or ""),
            "ok": bool(r.get("ok")),
            "actual": str(r.get("actual") or "")[:_SNAP_TEXT_MAX],
            "detail": str(r.get("detail") or "")[:_SNAP_DETAIL_MAX],
        })
    # ★ 2026-10-22：裁剪**保语义** —— 超出上限时优先保留非 ok 项（它们是淘汰
    #   依据），再按原序补齐。旧实现直接 `rows[:8]`，mc075 实测 n_bad=8 但
    #   落盘只有 3 条非 ok，其余 5 条坏证据被截掉 ⇒ 事后无法归因。
    if len(rows) > _SNAP_ITEMS_MAX:
        keep = {i for i, r in enumerate(rows) if not r["ok"]}
        for i in range(len(rows)):
            if len(keep) >= _SNAP_ITEMS_MAX:
                break
            keep.add(i)
        rows = [rows[i] for i in sorted(keep)][:_SNAP_ITEMS_MAX]
    return {
        "id": cid, "kind": kind, "n_items": len(items),
        "n_eq": sum(1 for x in items if not x.get("op")),
        "n_ineq": sum(1 for x in items if x.get("op")),
        "n_ok": int(rv.get("n_ok") or 0),
        "n_bad": int(rv.get("n_bad") or 0),
        "n_err": int(rv.get("n_err") or 0),
        "items": rows[:_SNAP_ITEMS_MAX],
    }


def math_check_stage(self, ctx) -> None:
    """4.4 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例）。

    注入点：``getattr(self, "math_check_det", None) or DeterministicChecker()``
    （与 4.3/4.2 同款，便于单测注入桩）。
    """
    try:
        if not _switch_enabled():
            self.record(ctx, "math_check",
                        "4.4 计算检测：开关关闭（five_gate_math_check=0）")
            return
        cands = list(getattr(ctx, "candidates", None) or [])
        if not cands:
            self.record(ctx, "math_check", "4.4 计算检测：候选池为空，跳过")
            return

        from .deterministic import DeterministicChecker
        _det = getattr(self, "math_check_det", None) or DeterministicChecker()
        problem = str(getattr(ctx, "problem", "") or "")
        domain = str(getattr(ctx, "domain", "") or "")
        # ★ 2026-10-21：题干里的数值前提（`d>=2` / `q>=1`）交给代值过滤器 ——
        #   否则「有条件才为真」的不等式会被越界样本判成假（实测误杀）。
        _conds = _extract_conditions(problem)

        kept, rejected, n_fail, n_unknown = [], [], 0, 0
        _snaps, _n_no_extract, _n_cv_rejected = [], 0, 0
        for c in cands:
            cid = getattr(c, "id", "?")
            kind = str(getattr(c, "answer_kind", "") or "")
            ans = str(getattr(c, "answer_norm", None)
                      or getattr(c, "answer", "") or "")
            det = {"verdict": "unknown", "evidence": "", "method": ""}
            _cv_pass = False    # 逐式复算是否全一致（与 det.verdict 分开记）
            try:
                det = _det.check_answer(ctx, problem, ans, domain or None) or det
            except Exception as _e_d:  # noqa: BLE001  异常 ⇒ 放行（不误杀）
                self.record(ctx, "math_check",
                            "cid=%s 计算检测异常（放行）: %s: %s"
                            % (cid, type(_e_d).__name__, str(_e_d)[:120]))
            # ★ 2026-10-17（用户要求「计算过程交给工具检」）：工具③逐式复算
            #   推理里的等式 + 不等式；**确有反例/算错 ⇒ 淘汰**（带证据）；
            #   无法判定（n_err）⇒ 放行（不误杀）。
            try:
                from tools.agent_toolkit.calc_verify import calc_verify as _cv
                _txt = str(getattr(c, "reasoning", "") or "")
                _eqs = _extract_numeric_equations(_txt)
                _ineq = _extract_inequalities(_txt)
                _items = [{"expr": a, "claimed": b} for a, b in _eqs[:16]]
                _items += [{"expr": a, "claimed": b, "op": op}
                           for a, b, op in _ineq[:16]]
                if _items:
                    _rv = _cv(_items, constraints=_conds)
                    _nb = int(_rv.get("n_bad") or 0)
                    _snap = _build_calc_snapshot(cid, kind, _items, _rv)
                    _snap["n_conditions"] = len(_conds)
                    if len(_snaps) < _SNAP_CANDS_MAX:
                        _snaps.append(_snap)
                    if _nb > 0:
                        # ★ 2026-10-17 审核：原写法 `A and B or C` 因优先级等价于
                        #   `(A and B) or C`——C 命中时即使 ok=True 也会被选为证据。
                        # ★ 2026-10-21：判词不匹配关键词时退回首条非 ok 项，
                        #   避免「淘汰了但证据为空」（留痕比好看重要）。
                        _r0 = next((x for x in (_rv.get("results") or [])
                                    if (not x.get("ok"))
                                    and ("不成立" in str(x.get("detail"))
                                         or "算错" in str(x.get("detail")))), {})
                        if not _r0:
                            _r0 = next((x for x in (_rv.get("results") or [])
                                        if not x.get("ok")), {})
                        _n_cv_rejected += 1
                        n_fail += 1
                        rejected.append({
                            "id": cid, "kind": kind, "answer": ans[:160],
                            "method": "calc_verify",
                            "evidence": str(_r0.get("detail") or "")[:300]})
                        self.record(ctx, "math_check",
                                    "cid=%s 工具③：提取 %d 条（等式%d/不等式%d）"
                                    "⇒ 不一致 %d、无法判定 %d ⇒ 淘汰；"
                                    "样例 `%s` = %s → 实际 %s"
                                    % (cid, _snap["n_items"], _snap["n_eq"],
                                       _snap["n_ineq"], _nb,
                                       int(_rv.get("n_err") or 0),
                                       str(_r0.get("expr") or "")[:60],
                                       str(_r0.get("claimed") or "")[:24],
                                       str(_r0.get("actual") or "")[:24]))
                        continue
                    # ★ 2026-10-22：逐式复算全部一致 —— 这是「算术层已核」，
                    #   与「答案被机器确认」（det.verdict=="pass"）不是一回事，
                    #   但也不能算 unknown（否则 4.4 的产出永远无法被下游区分）。
                    _cv_pass = True
                else:
                    _n_no_extract += 1
            except Exception as _e_cv:  # noqa: BLE001
                self.record(ctx, "math_check",
                            "cid=%s 工具③复算异常（放行）: %s: %s"
                            % (cid, type(_e_cv).__name__, str(_e_cv)[:100]))
            _v = str(det.get("verdict") or "unknown")
            if _v == "fail":
                n_fail += 1
                rejected.append({
                    "id": cid, "kind": kind, "answer": ans[:160],
                    "method": str(det.get("method") or ""),
                    "evidence": str(det.get("evidence") or "")[:300]})
                self.record(ctx, "math_check",
                            "cid=%s **计算检测不通过**（%s）⇒ 淘汰：%s"
                            % (cid, det.get("method") or "-",
                               str(det.get("evidence") or "")[:200]))
            else:
                if _v == "unknown":
                    n_unknown += 1
                # ★ 2026-10-22（五道链审计）：给候选挂**验证状态**，让下游能区分
                #   「机器确认通过」与「根本没判定」——旧实现两者在 4.5 投票里同权，
                #   等于 4.4 白做。
                try:
                    c.verify_math = ("pass" if (_v == "pass" or _cv_pass)
                                     else "unknown")
                except Exception:  # noqa: BLE001  只读候选对象异常不影响主链
                    pass
                kept.append(c)
                self.record(ctx, "math_check",
                            "cid=%s 通过/放行（verdict=%s、逐式复算=%s%s）"
                            % (cid, _v, "all_ok" if _cv_pass else "none",
                               "、kind=%s" % kind if kind else ""))

        # ★ 2026-10-21：淘汰来源**自动分组**（不写死 method 枚举）——
        #   4.2 曾因只统计两种 by 而漏掉 cross_model，同一类坑不重犯。
        _n_by_method: dict = {}
        for _r in rejected:
            _k = str(_r.get("method") or "unknown")
            _n_by_method[_k] = _n_by_method.get(_k, 0) + 1
        # ★ 2026-10-21：工具③证据快照（只落盘，不参与判定）。
        # ★ 2026-10-22：**累积而非覆盖** —— 4.4 可能被调用多次（首轮全淘汰后
        #   4.6 重写再进 4.4）。旧实现每次整体覆盖，mc075 实测第一轮 2 条候选
        #   的证据被第二轮 1 条冲掉，事后只剩 1/3 的淘汰依据。
        _prev = getattr(ctx, "math_calcverify_gate", None) or {}
        _prev_by = dict(_prev.get("n_by_method") or {})
        for _k, _v in _n_by_method.items():
            _prev_by[_k] = _prev_by.get(_k, 0) + int(_v)
        ctx.math_calcverify_gate = {
            "n_rounds": int(_prev.get("n_rounds") or 0) + 1,
            "n_cand": int(_prev.get("n_cand") or 0) + len(cands),
            "n_no_extract": int(_prev.get("n_no_extract") or 0) + _n_no_extract,
            "n_checked": int(_prev.get("n_checked") or 0) + len(_snaps),
            "n_rejected": int(_prev.get("n_rejected") or 0) + _n_cv_rejected,
            "n_by_method": _prev_by,
            "candidates": (list(_prev.get("candidates") or []) + _snaps)
                          [-_SNAP_CANDS_MAX:],
        }

        if rejected and not kept:
            # ★ 2026-10-18（p0b5_r4 实测 3 题：010/013/017 均受害）：原
            #   「全拒 ⇒ 回退保留原候选池（宁 unknown 不误杀）」是**降级**——
            #   工具③已明确算出候选算错（010 的正确结果是 2 = 标准答案），
            #   回退等于把**已证伪的错答案**放回去让投票选中。
            #   用户口径「不要隐藏错误」⇒ 删除回退：全拒即清空，交由 4.6 重写。
            ctx.candidates = []
            ctx.audit_reject_feedback = list(
                getattr(ctx, "audit_reject_feedback", None) or []) + [
                    "[计算检测] cid=%s: %s（%s）"
                    % (r["id"], r["evidence"], r["method"]) for r in rejected]
            ctx.math_check_rejected = list(
                getattr(ctx, "math_check_rejected", None) or []) + rejected
            ctx.math_check = {
                "n": len(cands), "n_rejected": len(rejected), "n_fail": n_fail,
                "n_unknown": n_unknown, "all_rejected_fallback": False,
                "n_by_method": dict(_n_by_method),
                "n_cv_no_extract": _n_no_extract,
                "rejected_ids": [r["id"] for r in rejected],
                "rejected_evidence": [str(r.get("evidence"))[:240]
                                      for r in rejected][:4],
            }
            self.record(ctx, "math_check",
                        "4.4 计算检测：%d 个候选**全部算错** ⇒ 清空候选池"
                        "（不回退；交由 4.6 重写）" % len(cands))
        else:
            if rejected:
                ctx.candidates = kept
                ctx.audit_reject_feedback = list(
                    getattr(ctx, "audit_reject_feedback", None) or []) + [
                        "[计算检测] cid=%s: %s（%s）"
                        % (r["id"], r["evidence"], r["method"])
                        for r in rejected]
                ctx.math_check_rejected = list(
                    getattr(ctx, "math_check_rejected", None) or []) + rejected
            ctx.math_check = {
                "n": len(cands), "n_rejected": len(rejected), "n_fail": n_fail,
                "n_unknown": n_unknown, "all_rejected_fallback": False,
                "n_by_method": dict(_n_by_method),
                "n_cv_no_extract": _n_no_extract,
                "rejected_ids": [r["id"] for r in rejected],
            }
            self.record(ctx, "math_check",
                        "4.4 计算检测：%d 候选 ⇒ 淘汰 %d / 保留 %d"
                        "（无法判定 %d、工具③未提取 %d）"
                        % (len(cands), len(rejected), len(kept), n_unknown,
                           _n_no_extract))

        # ★ 2026-10-15（工具③接线 · 观测模式）：对最终存活候选的推理做
        #   "等式逐条机械复算"（工具③ calc_verify）——算错条目写 diag/record，
        #   不影响淘汰（等云端验证后再升级为判定通道）。
        if _calcverify_enabled() and kept:
            try:
                from tools.agent_toolkit import calc_verify as _cv
                _n_eq, _bad = 0, []
                for _c in kept:
                    _eqs = _extract_numeric_equations(
                        str(getattr(_c, "reasoning", "") or ""))
                    if not _eqs:
                        continue
                    _n_eq += len(_eqs)
                    _r = _cv([{"expr": a, "claimed": b}
                              for a, b in _eqs[:16]])
                    for _x in (_r.get("results") or []):
                        if not _x.get("ok"):
                            _bad.append({
                                "id": getattr(_c, "id", None),
                                "expr": str(_x.get("expr", ""))[:80],
                                "claimed": str(_x.get("claimed", ""))[:40],
                                "actual": str(_x.get("actual", ""))[:40],
                                "detail": str(_x.get("detail", ""))[:140]})
                ctx.math_calcverify = {"n_checked": _n_eq, "n_bad": len(_bad),
                                       "bad": _bad[:8]}
                if _bad:
                    _x0 = _bad[0]
                    self.record(ctx, "math_check",
                                "4.4（工具③观测）：复算 %d 条等式，发现 %d 条"
                                "**不一致**；样例 cid=%s: %s = %s → 实际 %s"
                                % (_n_eq, len(_bad), _x0["id"], _x0["expr"],
                                   _x0["claimed"], _x0["actual"]))
                else:
                    self.record(ctx, "math_check",
                                "4.4（工具③观测）：复算 %d 条等式，全部一致"
                                % _n_eq)
            except Exception as _e_cv:  # noqa: BLE001  观测通道绝不阻断主链
                self.record(ctx, "math_check",
                            "4.4（工具③观测）异常（跳过）: %s: %s"
                            % (type(_e_cv).__name__, str(_e_cv)[:100]))

        try:
            from .orchestrator__verify import _pool_empty_fallback as _pef
            _pef(self, ctx, "4.4_math_check")
        except Exception:  # noqa: BLE001  兜底检查失败不影响主流程
            pass
    except Exception as _e:  # noqa: BLE001  增益环节绝不阻断主链
        self.record(ctx, "math_check",
                    f"4.4 计算检测异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
