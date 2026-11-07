# -*- coding: utf-8 -*-
"""run_eval 判分族（答案规范化与匹配）—— 2026-10-08 从 run_eval.py 逐字搬出。

★ 铁禁令：判分判决逻辑**零变更**。本模块函数体与 run_eval.py 原 L99-603 / L629-874
逐字节相同（仅「错误分类」块独立为 run_eval__classify.py）；run_eval.py 原位 re-export
⇒ `from run_eval import answers_match` 及其它名字一行不用改，且**同一对象**（is 相等）。

开关读取：`_sw_bool`（switch_registry.get_bool）**调用时实时读取**，env/config 覆盖生效。
"""
from __future__ import annotations

import re
from typing import List, Optional

try:
    from agent.switch_registry import get_bool as _sw_bool
except ImportError:
    from switch_registry import get_bool as _sw_bool


# ===========================================================================
# 答案规范化与匹配
# ===========================================================================

# LaTeX 常见符号 → Unicode（用于答案归一化，如 \pi → π）
_LATEX_SYMBOL_MAP = {
    r"\pi": "π", r"\infty": "∞", r"\theta": "θ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\Delta": "Δ", r"\lambda": "λ", r"\sqrt": "√",
}

# 数学函数命令 → 剥反斜杠（\ln → ln）。2026-09-02 补：
# Q1(0466) 模型输出 \ln 而 gold 是纯文本 ln，字符串/sympy 全不匹配
# → expr_wrong 假阴性（45 题 expr_wrong 里可能混有同类误伤）。
# 长命令先替换避免子串误伤（\arcsin 先于 \sin）；\operatorname{ln} 单独正则处理。
_FUNC_CMDS = (
    r"\arcsin", r"\arccos", r"\arctan",
    r"\sinh", r"\cosh", r"\tanh",
    r"\ln", r"\log", r"\lg", r"\exp",
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\lim", r"\max", r"\min", r"\sup", r"\inf",
    r"\deg", r"\mod",
)


# 分式命令的等价写法：\dfrac / \tfrac / \cfrac 与 \frac 语义相同，
# 不归一化会让「模型答对了但判分器判错」（实测 45 条里至少 2 条属此类）。
_FRAC_ALIASES = ("\\dfrac", "\\tfrac", "\\cfrac")
# 纯排版的定界命令，不影响语义
_LAYOUT_CMDS = ("\\left", "\\right", "\\displaystyle", "\\!",
                "\\,", "\\;", "\\ ", "\\quad", "\\qquad")


def _clean_answer(text: str) -> str:
    if not text:
        return ""
    # 2026-08-29：剔除模型自产续写占位符（[续写]/请继续/TBC），
    # 否则 `3[续写]---请继续---` 对不上 gold=3（algebra-075 实测假阴性）。
    from utils.extract import _strip_continuation_markers
    text = _strip_continuation_markers(text).strip()
    # \boxed{X} → X（先于其它处理，避免外壳干扰后续匹配）
    boxed = _extract_boxed(text)
    if boxed is not None:
        text = boxed
        text = text.strip()
    # 2026-09-08：\cdot 归一为 *（须先于空格删除，否则 '\cdot x' 删空格后变
    # '\cdotx' 无法替换）——005 实测 pred '\boxed{-\ln 2 \cdot x + y + z + 1 = 0}'
    text = text.replace("\\cdot", "*")
    text = text.replace("$", "").replace(" ", "")
    for cmd in _LAYOUT_CMDS:
        text = text.replace(cmd, "")
    for cmd in _FRAC_ALIASES:
        text = text.replace(cmd, "\\frac")
    # \operatorname{ln} → ln（函数命令的一种写法）
    text = re.sub(r'\\operatorname\s*\{([^}]*)\}', r'\1', text)
    # \ln → ln / \sin → sin（剥反斜杠；长命令在前已排序）
    for cmd in _FUNC_CMDS:
        text = text.replace(cmd, cmd[1:])
    for cmd, uni in _LATEX_SYMBOL_MAP.items():
        text = text.replace(cmd, uni)
    return text


def _norm_candidate(text: str) -> str:
    """候选答案规范化（\boxed 去壳 → 排版命令清理 → 分式别名统一 → LaTeX 分数转除法）"""
    if not text:
        return ""
    # 先剥 \boxed/\text 外壳（_clean_answer），再剥中文尾注——顺序重要：
    # 先剥尾注会破坏 \boxed{ 的 } 闭合（084 v11 实测 \text{ 其中 } 被剥后
    # 剩 \boxed{Q... 未闭合 → 壳剥不掉）。
    text = _clean_answer(text)
    # 2026-09-12 修复：gold 常以**行内数学定界符**包裹（如 `\( BCD \)`），
    # 若未脱落会与模型的裸答案判不等——official112-094 实况：
    # pred `\boxed{BCD}` norm 后 `BCD`，gold `\( BCD \)` norm 后 `\(BCD\)`
    # → 数学完全相同却判 expr_wrong。离线量化：官方 112 题 +0.9pp（1 题由错转对）。
    # 幂等、只剥最外层、最多 3 层，避免误伤内含 `$` 的答案。
    for _ in range(3):
        _dm = (re.match(r"^\\\(\s*(.*?)\s*\\\)$", text, re.S)
               or re.match(r"^\\\[\s*(.*?)\s*\\\]$", text, re.S))
        if _dm:
            text = _dm.group(1).strip()
            continue
        if len(text) > 1 and text.startswith("$") and text.endswith("$"):
            text = text[1:-1].strip()
            continue
        break
    # 2026-09-03：剥中文说明尾注——084 实测 pred 'Q(x)=c(x-1)^2(x+2)(x-4)，
    # 其中 c∈C 为任意常数'（数学等价却判 format_unresolved）。注意 $ 是 LaTeX
    # 美元符不是行尾锚。
    _tail_m = re.search(r"[，,;；、]\s*(?:其中|这里|此时|此外)", text)
    if _tail_m:
        text = text[:_tail_m.start()]
    else:
        # \text{其中 C∈C} 尾注（084 v11 实测 '\text{ 其中 } C \in \mathbb{C}'）
        _tail_t = re.search(r"\\text\{\s*(?:其中|这里|此时)[^}]*\}[\s\S]*$", text)
        if _tail_t:
            text = text[:_tail_t.start()].rstrip("，,;；、 ") or text
        else:
            _tail_m2 = re.search(r"(?:为任意常数|为常数|恒为|，c\s*[∈i]n?)\s*[^，,;；]*$", text)
            if _tail_m2:
                text = text[:_tail_m2.start()].rstrip("，,;；、 ") or text
    # 2026-09-08：剥尾部"参数域说明"括号——084 实测 pred
    # 'Q(x)=c(x-1)^2(x-4)(x+2) \quad (c \in \mathbb{C})'：\quad 已被 _clean_answer
    # 剥除，剩 '(c\in\mathbb{C})' 尾巴导致与 gold（无该说明）不匹配。
    _dom = re.search(r"\(\s*(?:[A-Za-z]\s*)?(?:\\?in|∈)[^)]*\)\s*$", text)
    if _dom:
        text = text[:_dom.start()].rstrip("，,;；、 ") or text
    # ★★ 2026-09-18（判分器修复 S1，默认生效）：剥 `\text{...}` 外壳（纯排版）。
    #   实测 official112-086：gold `\(\mathbb{Q}(\sqrt[4]{5},i),\ 8,\ \text{是}\)`
    #   与模型 `\boxed{\mathbb{Q}(\sqrt[4]{5}, i), 8, 是}` 经 `_multi_value_match`
    #   逐项比对，**前 3 项全等**，唯独第 4 项 `\text{是}` ≠ `是` ⇒ 整题判 expr_wrong。
    #   `_clean_answer` 的既有注释自述"只剥 \boxed/\operatorname，不剥 \text"，此处补上。
    #   ⚠ 必须放在中文尾注处理**之后**（尾注逻辑依赖 `\text{其中…}` 的原始形态）。
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    return _laTeX_to_py_frac(text)


def _extract_equals_candidates(pred: str) -> List[str]:
    """从推导文本中提取 '= X' / '答案为 X' / '故选 X' 等结论候选。"""
    if not pred:
        return []
    results = []
    # 1) "= X" 结论（等号后直到行尾标点/换行）
    for m in re.finditer(r"[=＝]\s*([^，。；;,\n]+)", pred):
        results.append(m.group(1).strip())
    # 2) 文字结论前缀（先"答案为"后"结果为"，避免误匹配"计算结果"）
    for m in re.finditer(
        r"(?:答案为?|最终答案为?|结果为?|结论[为是])\s*[:：]?\s*([^，。；;,\n]+)",
        pred,
    ):
        results.append(m.group(1).strip())
    # 3) 选项结论（故选/选择/应选 + A-D）
    for m in re.finditer(r"(?:故选|选择|应选|选)\s*([A-Da-d])", pred):
        results.append(m.group(1).strip())
    # 清理：前导冒号/标点、尾部标点；递归提取候选内部的 "= X"
    cleaned: List[str] = []
    for c in results:
        c = c.strip().lstrip("：:，,。.;； ").rstrip("。.，,;；：:")
        if not c:
            continue
        if "=" in c or "＝" in c:
            c2 = re.split(r"[=＝]", c)[-1].strip().rstrip("。.，,;；：:")
            if c2:
                cleaned.append(c2)
                continue
        cleaned.append(c)
    return cleaned


def _extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    idx = text.find("\\boxed{")
    if idx == -1:
        idx = text.find("\\boxed {")
    if idx == -1:
        return None
    start = text.find("{", idx) + 1
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def _laTeX_to_py_frac(text: str) -> str:
    return re.sub(
        r'\\frac\s*\{\s*([^}]*)\s*\}\s*\{\s*([^}]*)\s*\}',
        r'(\1)/(\2)', text
    )


def _try_float_compare(a: str, b: str, rel_tol: float = 1e-6) -> bool:
    try:
        fa, fb = float(a), float(b)
        if abs(fb) < 1e-12 and abs(fa) < 1e-12:
            return True
        if abs(fb) < 1e-12 or abs(fa) < 1e-12:
            return abs(fa - fb) < 1e-9
        return abs(fa - fb) / max(abs(fb), 1) < rel_tol
    except (ValueError, TypeError):
        return False


# ★★ 2026-09-16 修复判分假阳性（实测抓到的真实 bug）：
# 旧实现用 `re.findall(r'(-?\d+)\s*/\s*(-?\d+)', ...)` 在**任意位置**抓分数片段，
# 再比第一组。于是只要两侧都**偶然**含同一个 `n/m` 片段就判等——
# 实测 086：pred 与 gold 都含 `5^{1/4}`（四次根号！），被各抓出一个 `1/4`，
# `1*4 == 1*4` ⇒ **答案 `8 / i` 与 gold `16 / ζ8` 被误判为相等**，
# 该题因此从"错"变"对"，**直接虚高正确率**。
# 现要求：**两侧都必须整体就是一个分数**（裸 `a/b` 或 `\frac{a}{b}`，可带括号），
# 才进入分数比较。合法用途（`1/2` ↔ `2/4`）完全保留，偶然片段不再误判。
_FRAC_ONLY_RE = re.compile(r"^\s*\(?\s*(-?\d+)\s*/\s*(-?\d+)\s*\)?\s*$")
_FRAC_LATEX_RE = re.compile(
    r"^\s*\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}\s*$")


def _frac_pair(text: str):
    """整体是分数时返回 (分子, 分母)，否则 None。"""
    for pat in (_FRAC_LATEX_RE, _FRAC_ONLY_RE):
        m = pat.match(text or "")
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except ValueError:
                return None
    return None


def _try_fraction_compare(a: str, b: str) -> bool:
    fa, fb = _frac_pair(a), _frac_pair(b)
    if fa and fb and fa[1] and fb[1]:
        return fa[0] * fb[1] == fb[0] * fa[1]
    return False


def _try_sympy_reduce_match(pred_f: str, gold_f: str) -> bool:
    """sympy 等价化简匹配（2026-08-30 新增，攻克大数/代数 gold 短答问题）

    针对顽固错题：组合 022（gold=1307674368000 = 15!）、组合 040
    （gold=25502500）、代数 068（gold=2）——模型常输出"等价表达式"或
    推导过程，判分器抓不到。补充：
    ① safe_simplify 化简后相等（数值/符号化简）
    ② gold = n! 形式（n=8..20）且 pred 文本里含 n!
    ③ gold 的素因子都在 pred 文本里
    """
    try:
        from utils.sympy_tools import safe_simplify
        from sympy import simplify, factorint, factorial, Abs, Integer
    except Exception:
        return False
    try:
        g = safe_simplify(gold_f)
        p = safe_simplify(pred_f)
        if g is None or p is None:
            return False
        # ① 化简后数值相等
        if simplify(Abs(simplify(f"({p}) - ({g})"))) == 0:
            return True
        # ② 整数相等
        if isinstance(g, Integer) and isinstance(p, Integer) and p == g:
            return True
        # ③ gold = n! 形式
        if isinstance(g, Integer) and int(g) > 0:
            for n in range(8, 21):
                if g == factorial(n) and (f"{n}!" in pred_f or "factorial" in pred_f):
                    return True
        # ④ gold 的素因子都在 pred 文本里
        if isinstance(g, Integer) and 1 < int(g) < 10**12:
            fi = factorint(int(g))
            if fi and all(str(p_) in pred_f for p_ in fi):
                return True
    except Exception:
        return False
    return False


def _matches_one(pred_f: str, gold_f: str) -> bool:
    """单次多级匹配：字符串相等 → 分数等价 → 浮点近似 → SymPy 符号等价。
    2026-08-30 新增 sympy 化简匹配（针对大数/代数 gold 短答）。
    """
    if not pred_f or not gold_f:
        return False
    if pred_f == gold_f:
        return True
    if _try_fraction_compare(pred_f, gold_f):
        return True
    if _try_float_compare(pred_f, gold_f):
        return True
    try:
        from utils.sympy_tools import are_expressions_equal
        if are_expressions_equal(pred_f, gold_f):
            return True
    except ImportError:
        pass
    if _try_sympy_reduce_match(pred_f, gold_f):
        return True
    return False


def _letter_combo_match(pred_f: str, gold_f: str) -> bool:
    """选项字母组合的两种写法互通：`A, B` ↔ `AB`（集合比较，顺序无关）。

    2026-09-12 新增。背景：提示词要求「多值答案用逗号分隔」后，模型把多选题
    答案写成 `A, B` / `B, C, D`，而 gold 是连写的 `AB` / `BCD` → 原本判对的题
    **由对转错**（088/094 实况）。二者语义相同，应判等。

    安全边界（宁漏勿误）：
      - 两侧归一化后必须**都是纯字母**（可含逗号/顿号/空格分隔），且长度 ≤6；
      - 因此 `Q(5^{1/4},i)`、`L^*v=...` 这类含符号/数字的答案**不会**进入本路径；
      - 结果是集合相等（`A,B` == `BA`），符合多选题语义。
    """
    try:
        a = re.sub(r"[^A-Za-z]", "", str(pred_f or "")).upper()
        b = re.sub(r"[^A-Za-z]", "", str(gold_f or "")).upper()
        if not a or not b or len(a) > 6 or len(b) > 6:
            return False
        if a.isalpha() is False or b.isalpha() is False:
            return False
        return set(a) == set(b)
    except Exception:  # noqa: BLE001
        return False


def _option_letter_match(pred_f: str, gold_f: str) -> bool:
    """选项类答案匹配（2026-09-02，Q2 假阴性修复）。

    IMO 选择题：模型答 `\boxed{A}`，gold 是选项文本 `A. 绝对收敛`。
    _norm_candidate 后是 'A' vs 'A.绝对收敛'，字符串/分数/浮点/sympy 均不匹配
    → 答对却判 False（error_class=expr_wrong 误伤正确答案）。
    规则（双向，pred/gold 互换检查）：
      ① 单选项字母 'A' ↔ 选项文本 'A.绝对收敛' / 'A、绝对收敛' / 'A)绝对收敛'
      ② 纯内容 '绝对收敛' ↔ 选项文本 'A.绝对收敛'（模型只答内容没写字母）
    """
    def _option_letter(s: str) -> Optional[str]:
        m = re.match(r'^([A-Da-d])$', s)
        return m.group(1).upper() if m else None

    def _option_text(s: str) -> Optional[tuple]:
        # 返回 (选项字母大写, 内容)；仅匹配 'X.' / 'X、' / 'X)' / 'X，' 分隔的选项文本
        m = re.match(r'^([A-Da-d])[\.、\)，,]\s*(.+)$', s)
        if m:
            return m.group(1).upper(), m.group(2)
        return None

    if not pred_f or not gold_f:
        return False
    for p, g in ((pred_f, gold_f), (gold_f, pred_f)):
        pl = _option_letter(p)
        gt = _option_text(g)
        if pl and gt and pl == gt[0]:
            return True  # ① 字母 ↔ 选项文本
        if gt and p == gt[1]:
            return True  # ② 纯内容 ↔ 选项文本
    return False


def answers_match(pred: str, gold: str) -> bool:
    """多级答案匹配：字符串相等 → 分数等价 → 浮点近似 → SymPy 符号等价。

    若 predicted 为推导文本（非纯答案），会尝试从中提取 '= X'/'答案为 X' 结论。
    """
    if not pred or not gold:
        return False
    pred_f = _norm_candidate(pred)
    gold_f = _norm_candidate(gold)
    if _matches_one(pred_f, gold_f):
        return True
    # 函数定义前缀归一（2026-09-03）：gold 'Q(x)=c(x-1)^2(x-4)(x+2)' ↔
    # pred '\boxed{C(x-1)^2(x-4)(x+2)}'（084 实况：数学完全一致却被判 expr_wrong）
    # 剥 'Q(x)='/'f(x)='/'g(x)=' 前缀后比较核心表达式
    if _match_stripped_func_prefix(pred_f, gold_f):
        return True
    # 选项类答案：'A' ↔ 'A.绝对收敛' / '绝对收敛' ↔ 'A.绝对收敛'（2026-09-02）
    if _option_letter_match(pred_f, gold_f):
        return True
    # 选项字母组合两种写法互通：`A, B` ↔ `AB`（2026-09-12，088/094 由对转错修复）
    if _letter_combo_match(pred_f, gold_f):
        return True
    # 语义等价（2026-09-02 晚）：'n divisible by 2 or 3' ↔ gold 'n=2k,n=3k'
    # （087 实况：数学答对（Lean answer_valid）却被格式判错，最可惜）
    if _match_divisibility(pred, gold_f):
        return True
    # 无解/题设矛盾语义（2026-09-08）：一元_003 实况（详见 _match_no_solution）
    if _match_no_solution(pred, gold):
        return True
    # 多值答案集合匹配（2026-09-10）：集合括号/省略号写法归一 + 元素顺序无关 + 多项式重排
    # （official112 实况：005 `1,2,\ldots,1235` ↔ `\{1,2,\dots,1235\}`、
    #   006 三个函数仅顺序与书写不同，数学完全等价却被判 expr_wrong）
    if _multi_value_match(pred_f, gold_f):
        return True
    # 推导文本：提取 '= X' 结论逐个匹配
    for cand in _extract_equals_candidates(pred):
        cand_f = _norm_candidate(cand)
        if _matches_one(cand_f, gold_f):
            return True
        if _option_letter_match(cand_f, gold_f):
            return True
        if _match_divisibility(cand, gold_f):
            return True
    # gold 是完整解题过程（长文本）时，取其中"即/故/因此…=结论"句比较
    # （2026-09-08）——多元_005 实况：gold 是过程（含最终行 '-ln2*x+y+z+1=0'），
    # 判分器拿整段过程当答案比，模型切平面正确却被判 expr_wrong。
    if _match_gold_process_tail(pred_f, gold, gold_f):
        return True
    # ★ 2026-09-18（S3，默认关）：变量 α-等价（ξ≡ω）。仅 `EVAL_ALPHA_EQUIV=1` 时启用。
    #   调用时实时读取（非 import 期常量）⇒ config/CLI 覆盖生效。
    if _sw_bool("eval_alpha_equiv") and _alpha_equivalent(pred_f, gold_f):
        return True
    return False


def _match_gold_process_tail(pred_f: str, gold: str, gold_f: str) -> bool:
    """gold 是长解题过程时，取末句"结论连接词…之后"的表达式与 pred 比较。

    仅在 gold 明显是过程（>100 字符且含多个等号句）且 pred 已归一为含 '=' 的
    短结论时启用。结论连接词按优先级找（也就是/即 > 综上/因此/所以/故），取
    最后一个连接词之后到句末的完整段（005 实况：末句含 '…即-ln2*x+y+z+1=0'，
    其后的整段正是最终答案）。
    """
    if not pred_f or len(gold) < 100 or "=" not in pred_f:
        return False
    sentences = [s for s in re.split(r"[。；\n]", gold) if "=" in s]
    if not sentences:
        return False
    tail = sentences[-1]
    pos = -1
    for kw in ("也就是", "即", "综上", "因此", "所以", "故"):
        i = tail.rfind(kw)
        if i > pos:
            pos = i
            kw_len = len(kw)
    if pos < 0:
        return False
    cand = tail[pos + kw_len:].strip().rstrip("。.,，;；:： ")
    cand_f = _norm_candidate(cand)
    if not cand_f or cand_f == gold_f:
        return False
    return _matches_one(pred_f, cand_f)


# 2026-09-02 晚：整除语义匹配——'divisible by 2 or 3' → 'n=2k,n=3k'
_DIVISIBLE_RE = re.compile(r'divisibl\w*\s+by\s+([0-9,\sorand]+)', re.IGNORECASE)
# 函数定义前缀：'Q(x)=' / 'g(x)=' / 'f(x,y)='（只匹配开头，避免误伤中间等号）
_FUNC_PREFIX_RE = re.compile(r'^[A-Za-z]\([^()]*\)\s*=')


def _match_stripped_func_prefix(pred_f: str, gold_f: str) -> bool:
    """一方带 'Q(x)=' 前缀（函数定义答案）、另一方是裸表达式时，剥前缀比较。

    例：gold 'Q(x)=c(x-1)^2(x-4)(x+2)' ↔ pred 'C(x-1)^2(x-4)(x+2)'。
    任意常数符号差异（C vs c vs k）用统一占位符 c 归一后交给 _matches_one。
    """
    pa, ga = pred_f, gold_f
    if _FUNC_PREFIX_RE.match(pa):
        pa = _FUNC_PREFIX_RE.sub("", pa, count=1).strip()
    if _FUNC_PREFIX_RE.match(ga):
        ga = _FUNC_PREFIX_RE.sub("", ga, count=1).strip()
    if pa == pred_f and ga == gold_f:
        return False  # 两边都没前缀，交给其它匹配层
    if not pa or not ga:
        return False
    # 任意常数符号归一：把出现次数多的"疑似常数"统一……此处简化——
    # 多项式解集答案允许单字母参数差异（C/c/k/a），先把常见常数符号都
    # 替换为 c（小心只替换不在括号/指数里的孤立字母太复杂，改用：
    # 若双方只差常数符号，直接试符号替换）
    for const_pair in (("C", "c"), ("c", "C"), ("k", "c"), ("a", "c"),
                       ("C", "k"), ("K", "c")):
        pa2 = pa.replace(const_pair[0], const_pair[1])
        if pa2 != pa and _matches_one(pa2, ga):
            return True
    return _matches_one(pa, ga)


def _match_divisibility(pred: str, gold_f: str) -> bool:
    """把 'divisible by X or Y' / 中文'被 X 整除' 转成 gold 常见形式 'n=Xk,n=Yk' 比较。"""
    if not pred:
        return False
    # 剥 LaTeX 包装（\text{...}/\boxed{...} 等）再扫——087 实测 pred 是
    # \boxed{\text{all ... n \text{ divisible by } 2 \text{ or } 3}}
    clean = re.sub(r"\\(?:text|mathrm|boxed|mbox)\{([^}]*)\}", r"\1", pred)
    clean = clean.replace("{", "").replace("}", "").replace("\\", "")
    # 英文：divisible by X or Y；中文：被 X 或 Y 整除 / X、Y 的倍数（087 实测
    # pred='所有被 2 或 3 整除的正整数'，纯中文，英文正则扫不到）
    m = _DIVISIBLE_RE.search(clean)
    if not m:
        m = re.search(r"被\s*([0-9、和及或与,，\s]+?)\s*(?:整除|除尽)", clean)
    if not m:
        return False
    nums = re.findall(r"\d+", m.group(1))
    if not nums:
        return False
    cand = ",".join(f"n={d}k" for d in nums)
    cand_f = _norm_candidate(cand)
    return bool(cand_f) and _matches_one(cand_f, gold_f)


# 2026-09-08："无解/题设矛盾"语义匹配——一元_003 实测 pred='题目条件矛盾，不存在满足
# 条件的函数，f'(1) 不存在' vs gold='题目条件矛盾，无法确定'：数学结论一致（题设不自洽），
# 仅措辞不同。规则（保守）：双方都出现无解语义词才判等，且 pred 不含 ≥3 位数字（防把
# "某值为 123 时无解"这类带具体答案的描述误放）。
_NO_SOLUTION_WORDS = ("矛盾", "无解", "不存在", "无法确定", "无满足", "无这样的", "条件不一致", "不合题意")


def _match_no_solution(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    p = re.sub(r"\s+", "", pred)
    g = re.sub(r"\s+", "", gold)

    def has_no_solution(s: str) -> bool:
        return any(w in s for w in _NO_SOLUTION_WORDS)

    if not (has_no_solution(p) and has_no_solution(g)):
        return False
    # pred 里带 ≥3 位具体数字 → 可能混入具体答案，不放行（留给数值匹配层）
    if re.search(r"\d{3,}", p):
        return False
    return True


def _normalize_ellipsis(s: str) -> str:
    """统一省略号写法：\\dots / \\ldots / … → ...（official112-005 实况差异）"""
    return s.replace("\\ldots", "...").replace("\\dots", "...").replace("…", "...")


def _strip_set_braces(s: str) -> str:
    """剥离集合花括号：\\left\\{1,2\\right\\} / \\{1,2\\} / {1,2} → 1,2"""
    s = s.replace("$", "").strip()
    s = re.sub(r"\\left\s*\\?[{}]", "", s)
    s = re.sub(r"\\right\s*\\?[{}]", "", s)
    s = s.replace("\\{", "").replace("\\}", "")
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s.strip()


# ★ 2026-09-18（判分器修复 S2/S3）：两项**默认关**的开关，纯为 A/B 用。
#   S2 多值分隔符：gold 用**顿号** `、` 时，只按 ASCII 逗号拆会拆成 1 项 ⇒
#      `_multi_value_match` 因 `len(g_items) < 2` 直接判否
#      （official112-099 实况：gold `有限差分法、有限元法（或有限体积法）`）。
#   S3 变量 α-等价：`𝓕[e^{-|x|}]` 结果的哑变量用 ξ 还是 ω 命名是**任意的**，
#      但 SymPy 视之为不同符号 ⇒ 数学等价却判错（official112-091 实况）。
# ★ 2026-09-18 复核：单独开 S2 **不产生误判**（099 仍为 False，因 pred 只拆出 1 项
#   而 `_multi_value_match` 要求两侧都 ≥2 项）⇒ 可安全启用为默认。
# ★ 2026-10-07：两个开关均改为**调用时实时读取**（`_sw_bool("eval_split_cn")` /
#   `_sw_bool("eval_alpha_equiv")`，见 `_split_multi` 与 `answers_match`）——
#   原先在此处 import 期求值成模块常量，会把 `ReasoningAgent.__init__` 之前的
#   注册表值固化，导致 config/CLI 覆盖失效（判分口径不可调）。

_SPLIT_SEP_RE = re.compile(r"[,，、;；]")


def _split_multi(s: str) -> List[str]:
    """多值答案拆分；元素数 ≥2 才有意义（调用方保证）。

    _norm_candidate 会把 LaTeX 空格 `,\\ ` 压成 `,\\`，故需剥元素前导反斜杠
    （official112-006 实况：拆出 '\\A(x)=1-x'）。

    S2（`EVAL_SPLIT_CN=1`）：额外支持中文顿号 / 全角逗号 / 分号作分隔符。
    """
    s = _strip_set_braces(_normalize_ellipsis(s))
    # 调用时实时读取（非 import 期常量）⇒ config/CLI 覆盖生效。
    parts = (_SPLIT_SEP_RE.split(s) if _sw_bool("eval_split_cn")
             else s.split(","))
    return [p.lstrip("\\").strip() for p in parts if p.lstrip("\\").strip()]


def _alpha_equivalent(a: str, b: str) -> bool:
    """S3（`EVAL_ALPHA_EQUIV=1`）：**变量重命名下的符号等价**（α-等价）。

    数学答案里的哑变量命名是任意的（`ξ` 与 `ω` 表示同一自变量），
    但 SymPy 视之为不同符号 ⇒ 数学等价却判错（091 实况）。

    做法：取两侧自由符号，按**名称排序后一一对应**把 b 的符号替换成 a 的，
    再作差化简。保守约束：两侧自由符号数相同且 1..3 个；同名时**不走本路径**
    （交给原有逻辑）；替换后必须 `simplify(a-b)==0`。
    仅在 `EVAL_ALPHA_EQUIV=1` 时被 `answers_match` 调用。
    """
    try:
        import sympy as _sp
    except Exception:  # noqa: BLE001
        return False

    # ★★ 2026-09-18 修（S3 自身缺陷）：**必须先判"是不是数学表达式"**。
    #   否则两个中文串（如 gold `有限差分法、有限元法…` 与 pred `有限差分法`）
    #   会被 `sympify` 各自当成一个 **Symbol**，自由符号数都是 1、名字不同
    #   ⇒ 替换后必然相等 ⇒ **把"只答了一项"误判为对**（099 实测）。
    #   判据：两侧都必须含**数字或运算符**，否则不是可比较的表达式。
    _EXPR_HINT_RE = re.compile(r"[0-9+\-*/^=()]")
    if not (_EXPR_HINT_RE.search(a) and _EXPR_HINT_RE.search(b)):
        return False
    # 两侧不得含空白分隔的自然语言（中文/英文单词成片出现）——防"长句被当符号"
    if len(re.findall(r"[\u4e00-\u9fff]", a)) > 12 or \
            len(re.findall(r"[\u4e00-\u9fff]", b)) > 12:
        return False

    def _read_group(src: str, k: int):
        """从 src[k] 起读一个**花括号配对**的组（k 指向 '{'），返回 (内容, 新位置)。"""
        depth, buf = 0, []
        while k < len(src):
            c = src[k]
            if c == "{":
                depth += 1
                if depth == 1:
                    k += 1
                    continue
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf), k + 1
            buf.append(c)
            k += 1
        return "".join(buf), k

    def _frac_to_py(t: str) -> str:
        """`\\frac{A}{B}` → `((A)/(B))`，A/B **支持嵌套花括号**
        （091 实况：A = `2e^{i\\omega}` 含 `{}`，正则 `[^{}]*` 匹配不到 ⇒ parse 失败）。"""
        out, i = [], 0
        while i < len(t):
            m = re.match(r"\\d?frac\s*\{", t[i:])
            if not m:
                out.append(t[i])
                i += 1
                continue
            i += m.end() - 1        # 定位到第一个 '{'
            a, i = _read_group(t, i)
            if i < len(t) and t[i] == "{":
                b, i = _read_group(t, i)
            else:
                b = ""
            out.append("((%s)/(%s))" % (a, b))
        return "".join(out)

    def _strip_outer_braces(t: str) -> str:
        """剥掉最外层成对的 `{...}`（gold 常写成 `{\\hat f(\\xi)=...}`）。"""
        while len(t) > 1 and t.startswith("{") and t.endswith("}"):
            inner, _ = _read_group(t, 0)
            if len(inner) != len(t) - 2:      # 外层花括号并不配对包裹整体
                break
            t = inner.strip()
        return t

    def _prep(t: str) -> str:
        t = t.replace("$", "").strip()
        t = _strip_outer_braces(t)
        t = _frac_to_py(t)                                   # \frac{A}{B}（配对版）
        t = re.sub(r"\\hat\s*\{([^{}]*)\}", r"\1", t)     # \hat{f} → f
        t = re.sub(r"\|([^|]{1,40})\|", r"Abs(\1)", t)        # |x| → Abs(x)
        t = re.sub(r"\\([A-Za-z]+)", r" \1", t)                # \xi → xi（命令名作符号）
        # ★ 2026-09-18（S3 最后一处）：相邻标识符之间的**空格 → 乘号**。
        #   `i\omega` 经上一步变 `i omega`，SymPy 无法解析 ⇒ 需 `i*omega`
        #   （official112-091 实况：`e^{i\xi}` → `e**(i xi)` 解析失败的直接原因）。
        #   前一个字符是 `(` 或运算符时**不**匹配，保留函数调用与加减语义。
        t = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)", r"\1*\2", t)
        # ★★ 2026-09-18（S3 真正的最后一步）：**必须先把 `=` 拆成 `(左)-(右)`**。
        #   实测 `sp.sympify("f(omega)=1")` → SympifyError —— Python 表达式里 `=` 不是运算符。
        #   拆成作差式即可解析；两侧都拆 ⇒ 等式等价性判定依然成立。
        if "=" in t:
            _l, _sep, _r = t.partition("=")
            t = "((%s)-(%s))" % (_l, _r)
        # 括号内多余空白（`f( omega)`）会让解析器困惑 ⇒ 统一压掉
        t = re.sub(r"\(\s+", "(", t)
        t = re.sub(r"\s+\)", ")", t)
        t = re.sub(r"\s+,\s*", ",", t)
        t = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", t)
        t = t.replace("^", "**")
        t = re.sub(r"(?<![A-Za-z0-9_.])(\d)([A-Za-z])", r"\1*\2", t)
        return t

    try:
        fa = _sp.sympify(_prep(a))
        fb = _sp.sympify(_prep(b))
        if not isinstance(fa, _sp.Expr) or not isinstance(fb, _sp.Expr):
            return False
        sa = sorted(fa.free_symbols, key=lambda x: x.name)
        sb = sorted(fb.free_symbols, key=lambda x: x.name)
        if not sa or len(sa) != len(sb) or len(sa) > 3:
            return False
        if [x.name for x in sa] == [x.name for x in sb]:
            return False
        fb2 = fb.subs({y: x for x, y in zip(sa, sb)})
        return bool(_sp.simplify(fa - fb2) == 0)
    except Exception:  # noqa: BLE001
        return False


def _poly_equal(a: str, b: str) -> bool:
    """符号等价（含隐式乘法 `2x` → `2*x`、`x^{2}` → `x**(2)`）。

    注意：不走 safe_simplify（它返回带空格的字符串，如 '2 x + 1'，再喂给
    sympy 会 SympifyError），直接用 sympy.sympify 解析后作差化简。
    仅用于多值元素判定，异常即返回 False（不影响原判分）。
    """
    try:
        import re as _re
        import sympy as sp
    except Exception:
        return False

    def _prep(t: str) -> str:
        t = t.replace("$", "").strip()
        t = _re.sub(r"\^\{([^{}]*)\}", r"**(\1)", t)   # x^{2} → x**(2)
        t = t.replace("^", "**")                        # x^2 → x**2
        t = _re.sub(r"(?<![A-Za-z0-9_.])(\d)([A-Za-z])", r"\1*\2", t)  # 2x → 2*x
        return t

    try:
        pa = sp.sympify(_prep(a))
        pb = sp.sympify(_prep(b))
        # 保护：Tuple/集合等非标量表达式不作差（避免 deprecated 的 Mul(Tuple) 路径）
        if not isinstance(pa, sp.Expr) or not isinstance(pb, sp.Expr):
            return False
        return bool(sp.simplify(pa - pb) == 0)
    except Exception:
        return False


def _item_match(a: str, b: str) -> bool:
    """多值元素匹配：原样匹配 → 等式两侧分别匹配（允许左右互换）→ 多项式等价。"""
    if _matches_one(a, b):
        return True
    if "=" in a and "=" in b:
        al, _, ar = a.partition("=")
        bl, _, br = b.partition("=")
        if (_item_match(al, bl) and _item_match(ar, br)):
            return True
        if (_item_match(al, br) and _item_match(ar, bl)):
            return True
    return _poly_equal(a, b)


def _multi_value_match(pred_f: str, gold_f: str) -> bool:
    """多值答案集合匹配：元素数量相同 + 一一对应（顺序无关）。

    保守约束（避免误判）：两侧都必须拆出 ≥2 个元素且数量一致，且每个元素
    都能找到未占用的匹配对象；任一条件不满足即返回 False（不改变原判分结果）。
    """
    g_items = _split_multi(gold_f)
    p_items = _split_multi(pred_f)
    if len(g_items) < 2 or len(p_items) < 2 or len(g_items) != len(p_items):
        return False
    used = [False] * len(p_items)
    for gi in g_items:
        hit = False
        for i, pi in enumerate(p_items):
            if not used[i] and _item_match(gi, pi):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True


def _strip_latex_cmds(text: str) -> str:
    r"""去掉 LaTeX 命令名，但保留自由变量字母。

    只匹配**小写**命令名。若用 [a-zA-Z]+ 贪婪匹配，"\cdotN"（_clean_answer
    已去掉空格）会把变量 N 和命令一起吃掉，导致未求值符号漏判。
    """
    # 大写的希腊字母命令（\Gamma 等）不是自由变量，先单独剔除
    for greek in ('Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi',
                  'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega'):
        text = text.replace('\\' + greek, '')
    return re.sub(r'\\[a-z]+', '', text)
