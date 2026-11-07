"""非答案形态判据（2026-10-08 从 agent/formatter.py 逐字搬出）。

簇①：`_looks_like_non_answer` 及其**专用**正则/常量族（工具标签 / 步骤标签 /
推理连接词 / 过程叙述 / 段首叙述 / Python 代码 / 英文检索词）。这些正则
**仅**被 `_looks_like_non_answer` 使用，故整段随迁。

formatter.py 原位 re-export（`from .formatter__nonanswer import
_looks_like_non_answer`）⇒ 外部 `from agent.formatter import
_looks_like_non_answer`（orchestrator / orchestrator__verify / final_selector /
tools/lean_local/lean_gate / 多处测试）契约逐字不变。

⚠ 本模块**不** import formatter 主模块（避免循环导入）；`_looks_like_non_answer`
内部的 `utils.extract._looks_like_reasoning_fragment` 仍为函数内惰性 import。
"""
from __future__ import annotations

import re


# ★ 2026-09-17 新增：工具调用 XML 残留标签（用于判定"这不是答案"）。
# 实测 official112-025：6 个候选里 5 个 answer 就是字面量 `</tool_call>`；
# 而原 `_REFUSAL_RE` / `_MISSING_ANSWER_RE` / `_INCOMPLETE_RE` 全是"拒绝词
# 词表"，对这类结构残留**全部漏检**，脏文本被当合法答案直接提交。
_TOOL_TAG_RE = re.compile(
    r"</?tool_calls?>|</?function\b|<parameter=|</parameter>",
    re.IGNORECASE,
)

# ★★ 2026-09-17 补强：实测发现上面的判据**只拦得住工具标签** —— 配套的
# `utils.extract._looks_like_reasoning_fragment` 对下列全部 5 类真实脏答案
# **都返回 False**（已逐例实测）。故此处按**实测形态**补三类结构信号：
#   · `步骤8：重新思考——正确的下界构造`（002 候选）→ 步骤/思考标签
#   · `从 $N$ 倒推：若从位置 $k`（032 最终答案）→ 推理连接词
#   · `a_wins = analyze_game(N)`（032 候选）→ Python 代码
#   · `subset sum approximation bounded integers ...`（013 候选）→ 英文检索词
# ⚠ 已知未覆盖（有意）：`509040-2*169680` 这类**未求值表达式** —— 它与
#   `\binom{2k}{k}^2` 等合法表达式答案形态难分，判错代价高于收益，故不拦。
_STEP_LABEL_RE = re.compile(
    r"^\s*(?:步骤\s*\d|第\s*[0-9一二三四五六七八九十]+\s*步|Step\s*\d"
    r"|\d+\s*[.、)]\s*【)"
    r"|重新思考|让我们(?:来)?(?:看|想|计算|考虑)",
    re.IGNORECASE,
)
# 推理连接词：正常"答案"不会带"倒推 / 由此 / 综上"这类**过程**措辞。
# 实测 032 提交的最终答案就是 `从 $N$ 倒推：若从位置 $k` 这样的推理碎片。
# 放在 has_math 闸门**之外**：该碎片含 `$`，会被"疑似数学式"跳过。
_REASONING_CONNECTIVE_RE = re.compile(
    r"倒推|由此可见|同理可得|可以看出|注意到|综上|我们需要")
_PY_ASSIGN_CALL_RE = re.compile(r"^[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\(")
_PY_KEYWORD_RE = re.compile(
    r"(?<![\w.])(?:def|import|return|assert|print|lambda|while|elif)\b")
# 英文检索词：足够长、足够多空格、纯 ASCII 且只含词/数字/连字符/斜杠/加号。
_ENGLISH_QUERY_MIN_LEN = 50
_ENGLISH_QUERY_MIN_SPACES = 7
_ENGLISH_QUERY_ALLOWED = re.compile(r"^[A-Za-z0-9\s\-+/]+$")
# ★★ 2026-09-18 再补强：实测**最终答案**本身是"过程叙述"——模型把
#   "我接下来要做什么"当成答案提交，且因句子含 `$` 而穿过 has_math 闸门：
#     · `搜索已知结论：这个问题看起来像是一个已知的组合数学问题。也许答案是
#        $2^{k+1}$ 或 $2 \\cdot 4^{k-1}$ 之类的。让我用 web_search 查找类似问题。`（020）
#     · `继续找规律，目前 type B：2, 8, 10。`（032）
#     · `步骤11：搜索已知结论`（025）
#     · `让我搜索 $x^4 + 5$ 的分裂域次数。`（086 候选）
#   ⚠ 原 `_STEP_LABEL_RE` 只写了 `让我们`，**漏了 `让我`** —— 这一字之差是本轮
#     020/025/086 三题漏检的直接原因。
#   与 `_REASONING_CONNECTIVE_RE` 同样放在 has_math 闸门**之外**执行。
_PROCESS_NARRATIVE_RE = re.compile(
    r"让我(?:们)?(?:先|再|来)?(?:搜索|用|查|看看|想|计算|考虑|尝试|重新审视|重新思考)"
    r"|搜索(?:一下|已知结论|相关(?:数学)?结论|类似)"
    r"|继续(?:找规律|寻找|搜索)"
    r"|目前\s*(?:type|类型|得到|发现|只)"
    r"|文本提到"
    r"|也许答案(?:是|可能)"
    r"|这个(?:问题|题)看起来"
    r"|看起来像是(?:一个)?已知"
    # ★ 2026-09-18（审计发现，086 实测仍漏）：
    r"|在答案中[，,]?\s*通常接受"
    r"|但通常这类题的标准答案"
    r"|通常这类题的?标准答案"
    r"|对于[^，。]{0,12}类似",
    re.IGNORECASE,
)

# ★ 2026-09-21 新增：**段首叙述形态**（受控、锚定串首，避免误杀答案中的连接词）。
#   实测 0921 arm2c2t：6.5 重做循环把过程叙述推成最终答案并提交 ——
#     · `因此a_{2025} = 2026。`（003；含 `{}` ⇒ 触发 has_math 跳过代码/英文判定，
#        而「因此」不在 `_REASONING_CONNECTIVE_RE` 的 7 个词里 ⇒ 全链漏检）
#     · `最终正确的上界证明：考虑 $2|F|$ 个"端点"（…`（004，187 字长段）
#     · `所以对手（选5个数的人）会试图构造5个数…`（002）
#   规则 = 段首连接词 ∨ 段首"结论文档标签"（…证明： / 结论： / 思路： …）。
#   ⚠ 只锚定**串首**（无 re.MULTILINE）⇒ 答案中段出现「因此」不受影响。
#   ⚠ 实测对照集 16 条合法答案（各题 gold + 正常候选 + `解：x=1` / `答：42` /
#     `无解` / 区间 / 分数 / 中文列举）**零误杀**。
#   ⚠ 刻意**不**拦纯数值（`20`）与数学碎片（`p=1：f(f(d)) = f(2) + 1。`）——
#     形态上与合法答案难分，判错代价高于收益；这两类由 orchestrator 6.5 的
#     「未过审核 → 回滚到重做前答案」（2026-09-21）兜住。
_LEAD_NARRATIVE_RE = re.compile(
    r"^\s*(?:因此|所以|于是|从而|因而|由此|进而|继而|这样一来|这表明|这说明"
    r"|也就是说|换句话说|综上)"
    r"|^\s*.{0,14}(?:证明|论证|推导|结论|思路|分析|讨论)\s*[:：]")
# 含这些记号即视为"疑似数学式"，**不再**按代码/英文词串判定，避免误杀。
_MATH_MARKERS = ("\\", "{", "}", "$", "^")


def _looks_like_non_answer(text: str) -> bool:
    """2026-09-17 新增：结构化"是否像合法答案"检查（拒绝词表之外的脏答案）。

    背景：本文件原只有三条**拒绝词**正则，实测对下列脏答案全部漏检：
      - `</tool_call>`（official112-025：5/6 候选）；
      - `a_wins = analyze_game(N)`（032：代码片段）；
      - `从 $N$ 倒推：若从位置 $k`（032：推理碎片）；
      - `步骤8：重新思考——正确的下界构造`（002：步骤标签）；
      - `subset sum approximation bounded integers ...`（013：英文搜索词）。
    这里只做"明显不是答案"的**结构**判定：空串 / 工具标签 / 步骤与思考标签 /
    推理连接词 / Python 代码 / 英文检索词串 / 推理碎片。
    刻意**不**使用 `is_valid_final_answer` —— 它会误杀合法中文答案
    （如 gold `有限差分法、有限元法`）。import 惰性化并 try/except 兜底。
    """
    if not text or not str(text).strip():
        return True
    t = str(text).strip()
    try:
        if _TOOL_TAG_RE.search(t):
            return True
    except Exception:  # noqa: BLE001
        pass
    # —— 步骤 / 思考标签 / 推理连接词 / 过程叙述：推理过程的措辞，不可能是答案 ——
    #    ★ 2026-09-21 增加 `_LEAD_NARRATIVE_RE`（段首连接词 / 结论文档标签）。
    try:
        if (_STEP_LABEL_RE.search(t) or _REASONING_CONNECTIVE_RE.search(t)
                or _PROCESS_NARRATIVE_RE.search(t)
                or _LEAD_NARRATIVE_RE.search(t)):
            return True
    except Exception:  # noqa: BLE001
        pass
    # —— 含 LaTeX 记号（\ { } $ ^）时视为疑似数学式，跳过代码/英文词串判定 ——
    has_math = any(ch in t for ch in _MATH_MARKERS)
    if not has_math:
        try:
            if _PY_ASSIGN_CALL_RE.match(t) or _PY_KEYWORD_RE.search(t):
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            if (len(t) >= _ENGLISH_QUERY_MIN_LEN
                    and t.count(" ") >= _ENGLISH_QUERY_MIN_SPACES
                    and _ENGLISH_QUERY_ALLOWED.match(t)):
                return True
        except Exception:  # noqa: BLE001
            pass
    try:
        from utils.extract import _looks_like_reasoning_fragment
        if _looks_like_reasoning_fragment(t):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False
