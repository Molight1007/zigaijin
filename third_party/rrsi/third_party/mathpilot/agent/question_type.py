from __future__ import annotations
"""题型识别与差异化策略（v2.6）。

基于关键词识别题型：证明题 / 选择题 / 判断题 / 填空题 / 解答题。
不同题型采用差异化解题策略：
- 选择题：利用选项逆推验证；
- 判断题：不确定时合理猜测；
- 证明题：对每一步证明步骤反复校验；
- 解答题：附带答案进行结果检测。
"""

import re

# ---------------------------------------------------------------------------
# 题型名称常量
# ---------------------------------------------------------------------------
QT_PROOF = "证明题"
QT_CHOICE = "选择题"
QT_JUDGE = "判断题"
QT_FILL = "填空题"
QT_SOLUTION = "解答题"


# ---------------------------------------------------------------------------
# 关键词信号表
# ---------------------------------------------------------------------------
# 选项标记：A. / (A) / A、/ A） / 【A】 等（选择题强信号）
# 2026-09-01 修复（系统根因）：原正则含 [\s:：]|$ 分隔符，把数学正文里的
# (b - a)、(a, b)、 a 、(a) 误判成选项 → PB 证明题 39/60 被判选择题，
# 模型被误导只输出选项字母，且 Lean 门禁（is_proof）被挡在门外。
# 新规则：① 字母必须紧跟标点（. 、 ) 】 等），空格不算分隔符；
#         ② 命中标记数 ≥2 才算选择题（杜绝孤立 (a) / (b-a) 误判）；
#         ③ 选项关键词（选择/选项/选出…）仍为强信号，1 个即判选择题。
_OPTION_MARK_RE = re.compile(
    r"(?:^|[\s(（\[【])([A-Da-d])(?:[\.、\)）\]】]|[：:])",
    re.MULTILINE,
)

# 数学正文信号：标记后紧跟这些字符 → 是数学表达式不是选项
_MATH_AFTER = ("\\", "$", ",", "，", "-", "=", "\\in", "\\neq", "\\leq", "\\geq")


def _count_option_marks(text: str) -> int:
    """统计真实选项标记数（排除数学正文误判）。

    数学正文里的 (b - a) / (a, b) / (a)f / \\item[(a)] / K (\\neq D) 等
    会被过滤：标记后紧跟数学符号、或前导是 \\item 分点列表 → 不算选项。
    """
    n = 0
    for m in _OPTION_MARK_RE.finditer(text):
        # 1) 前导 \\item[(a)] 分点列表（LaTeX enumerate）
        pre = text[max(0, m.start() - 12):m.start()]
        if "item[" in pre or "\\item" in pre:
            continue
        # 2) 标记后紧跟数学符号/公式起始 → 数学正文
        after = text[m.end():m.end() + 4].lstrip()
        if after.startswith(_MATH_AFTER):
            continue
        # 3) 括号式标记 (a) 后紧跟字母/数字/括号/右括号（如 f(a))、(b-a)f(f(a))）
        #    → 数学调用/嵌套，不是选项
        # 2026-09-10：补闭合括号 ) ] } ）】——LaTeX 分式 $\frac{g(a)-g(b)}{a-b}$
        # 里 "g(b)}" 的 "}" 原不在集合内，导致 official112-009 被误判选择题。
        if m.group(0).strip().endswith((")", "）", "]", "】")):
            nxt = text[m.end():m.end() + 1]
            if nxt and (nxt.isalnum() or nxt in "([{（)]}）】"):
                continue
        n += 1
    return n

# ---------------------------------------------------------------------------
# 客观题选项检测 v2（2026-09-12，按 official112 客观题标定）
# ---------------------------------------------------------------------------
# 背景：v1 的 `_count_option_marks` 依赖「标记前导必须是空白/左括号」+
# 「标记后不能跟数学符号」，在比赛卷上漏检 6 道真选择题（087/098/102/103/
# 106/107，实测全部被判成"解答题"，特化策略与选项清单都没生效）：
#   · 中文连排选项 `A.长期趋势B.季节变动C.循环变动D.不规则变动E.随机变动`
#     —— 选项之间无空白，B/C/D/E 的前导是汉字 → 全部漏检（102/103/106/107）；
#   · LaTeX 列表选项 `\item[A.] … \item[B.] …`（087）→ 被"\item 分点"规则整段丢弃；
#   · 选项后紧跟数学式 `A. $\kappa(A)=\sqrt{…}$`（098）→ 被"反斜杠开头即数学"
#     规则过滤，计数不足 3。
#
# v2 判据换成「有序字母序列」，比逐标记过滤鲁棒得多：
#   ① 只认**大写** A–E（小写 (a)/(b) 是数学正文与分点枚举，直接排除）；
#   ② 分隔符只认 `.` `．` `、`——右括号 `)` 形态（数学调用 `\kappa(A)`、
#      `f(B)`）因此天然排除，不需要额外的数学符号白名单；
#   ③ 同一字母重复出现 → 视为题面重复列项，取最新位置而不重开序列；
#   ④ 字母必须**严格递增**且相邻标记间距 < 400 字符（避免跨段落误配）；
#   ⑤ 与 v1 取 max（只增不减）：v2 只可能"补回"漏检，不会让原本命中的选题
#      丢掉命中，把误判风险限制在新增路径上，便于回归定位。
_OPTION_MARK_V2_RE = re.compile(r"(?<![A-Za-z0-9])([A-E])\s*([\.．、])")

# 同一选项序列内相邻标记允许的最大字符间距
_OPTION_SEQ_MAX_GAP = 400


def _option_sequence_marks(text: str) -> list[tuple[int, str]]:
    """返回最长「有序选项序列」的 ``[(位置, 字母), ...]``；无序列返回 ``[]``。

    序列 = 字母严格递增、相邻间距 < ``_OPTION_SEQ_MAX_GAP`` 的最长标记段。
    例：`A.甲B.乙C.丙D.丁` → `[(0,'A'),(4,'B'),(8,'C'),(12,'D')]`；
    `\\kappa(A)=\\sqrt{|A|}` → `[]`（右括号形态不认，孤立大写字母也不成序列）。
    """
    marks = [(m.start(), m.group(1)) for m in _OPTION_MARK_V2_RE.finditer(text or "")]
    if not marks:
        return []
    best: list[tuple[int, str]] = []
    cur: list[tuple[int, str]] = []
    for pos, lab in marks:
        if not cur:
            cur = [(pos, lab)]
            continue
        prev_pos, prev_lab = cur[-1]
        if lab == prev_lab:
            cur[-1] = (pos, lab)          # 重复列项，跟随最新位置
            continue
        if lab > prev_lab and pos - prev_pos <= _OPTION_SEQ_MAX_GAP:
            cur.append((pos, lab))
            continue
        if len(cur) > len(best):
            best = cur
        cur = [(pos, lab)]
    if len(cur) > len(best):
        best = cur
    return best


def _option_sequence_length(text: str) -> tuple[int, list[str]]:
    """返回 ``(去重后选项字母数, 有序字母序列)``。"""
    best = _option_sequence_marks(text)
    labels = [lab for _pos, lab in best]
    return len(set(labels)), labels


# 填空题空白占位：___ / ＿ / 【空】 / () / \((\quad)\)
# 2026-09-12：补 `\(\quad\)` 形态——official112-086 的三个空写成 `$(\quad)$`，
# 原判据不命中 → 被判"解答题"，特化的"按空序逗号分隔"输出规范没生效。
_UNDERSCORE_BLANK_RE = re.compile(r"_{2,}|＿{2,}|\[空\]|【空】|\\underline\{\s*\}")
# `(\quad)` / `（\quad）` 形态（可带 $ 包裹）；**不可**裸认 `\quad`——该命令是
# 排版间隔符，正文里极常见（084 的 `\quad (c \in \mathbb{C})`），裸认会大面积误判。
_QUAD_BLANK_RE = re.compile(r"\$?\s*[（(]\s*\\quad\s*[）)]\s*\$?")

# ---------------------------------------------------------------------------
# 主任务信号（2026-09-10，按 official112 比赛卷标定）
# ---------------------------------------------------------------------------
# 背景：原实现用「子串命中」判证明题（"prove" in low / "proof" in low），
# 在比赛卷上会系统性误判——比赛题常以
#     "determine, with proof, the largest number n ..."   (official112-028)
#     "... determine the moment ... or prove that such a moment does not
#      necessarily exist"                                  (official112-034)
#     "find (with proof) the amount that Winnie wins"      (official112-061)
# 这类**求值题附带证明要求**的形态出现，"proof / prove" 只是修饰语，主任务
# 仍是求一个结果。误判后果不小：is_proof 直接决定 Lean 走「整题形式化证明」
# 还是「答案级核验」，判错 = 走错验证通道（实测这 3 题被判证明题后触发
# 6 候选整题 verify，全 unknown 后 verify_stop 止损，白烧约 2 分钟预算）。
#
# 新规则：证明题 = 命中「证明型主任务句式」**且**未命中「求值型主任务」。
# 注意禁止用裸词 proof / prove —— 那正是本次误判的根因。

# 求值型主任务信号：题面在要一个结果（数值 / 集合 / 表达式 / 区间）。
# 中文用 求(?!证) 排除「求证」（否则所有中文求证题会被否决）。
_SOLVE_TASK_RE = re.compile(
    r"\b(?:find|compute|determine|calculate|evaluate|what\s+is)\b"
    r"|求(?!证)|计算|求解|等于多少",
    re.I,
)

# 证明型主任务信号：必须是「要求证明」的句式，而非 proof 这类裸词。
_PROOF_TASK_RE = re.compile(
    r"证明(?!题)|求证|试证"
    r"|\bprove\s+(?:that|the\s+following|or\s+disprove|it)\b"
    r"|\bshow\s+that\b|\bdemonstrate\s+that\b",
    re.I,
)


def classify_question_type(problem: str) -> str:
    """基于关键词识别题型，返回 证明题/选择题/判断题/填空题/解答题 之一。

    优先级（信号强度由强到弱）：
        选择题 > 证明题 > 判断题 > 填空题 > 解答题(兜底)
    """
    text = problem or ""
    low = text.lower()

    # 1) 选择题：选项标记 ≥3 个 / 选项关键词（信号最专一）
    # 2026-09-01：标记数 ≥2 才判选择题。数学正文里的 (b-a) / (a, b) / (a)f /
    # \item[(a)] 均被排除（见 _is_option_mark），杜绝 PB 证明题 39/60 误判。
    # 2026-09-10（official112 标定）：阈值 2 → 3。实测 2 个标记仍会被数学
    # 正文/步骤枚举触发两例误判：
    #   official112-009 的 $\frac{g(a)-g(b)}{a-b}$ —— "g(b)}" 后接右花括号，
    #                    旧过滤只查"后接字母/左括号"，漏掉闭合括号；
    #   official112-045 的两步说明 "(a) Bob ... (b) Ali ..."，是步骤枚举不是选项。
    # 比赛卷真选择题均为 3~4 选项（094/109/110 等实测 ≥3），≥3 可完全区分；
    # 显式选项关键词（选择/选项/选出…）路径不受影响。
    # 2026-09-12（official112 客观题专项）：补 v2「有序字母序列」判据，与 v1 取
    # max（只增不减）。v2 修回 6 道被漏判为"解答题"的真选择题（087/098/102/103/
    # 106/107），详见 _OPTION_MARK_V2_RE 上方注释；v1 原判据与阈值一字未动，
    # 历史防误判收益（PB 证明题 39/60 误判）完整保留。
    _v2_n, _v2_labels = _option_sequence_length(text)
    _opt_n = max(_count_option_marks(text), _v2_n)
    if _opt_n >= 3 or any(
        k in low for k in ("选择", "选项", "选出", "单选", "多选", "下列选项中", "正确的选项")
    ):
        return QT_CHOICE

    # 2) 证明题（任务型判定，2026-09-10 按 official112 标定重写）
    #    原实现：any(k in low for k in ("证明","求证","试证","prove","proof",
    #    "show that","verify that")) —— 子串命中即判证明题。在比赛卷上会误判
    #    "with proof" / "or prove that ..." 形态的**求值题**（028/034/061）。
    #    新实现：命中证明型主任务句式，且未被求值型主任务否决。
    if _PROOF_TASK_RE.search(text) and not _SOLVE_TASK_RE.search(text):
        return QT_PROOF

    # 3) 判断题
    if any(k in low for k in ("判断", "对错", "正误", "对还是错", "true or false", "是否正确")):
        return QT_JUDGE

    # 4) 填空题 / 多空题
    # 2026-09-12：补 `(\quad)` 空形态——official112-086 是"三个空 + 一个是否判断"
    # 的三段型客观题（gold 形态 `域, 次数, 是`），原判据不命中被判"解答题"，
    # 按空序逗号分隔的输出规范因此没生效。
    if (_UNDERSCORE_BLANK_RE.search(text) or _QUAD_BLANK_RE.search(text)
            or any(k in low for k in ("填空", "填入", "填上", "blank"))):
        return QT_FILL

    # 5) 解答题（兜底）
    return QT_SOLUTION


# ---------------------------------------------------------------------------
# Lean 适用性判定（2026-09-10，按 official112 比赛卷标定）
# ---------------------------------------------------------------------------
# 用户要求：**所有题都要用到 Lean 检测机制，除非非常简单的题**。
# 但有一类题 Lean 结构上无从核验——答案是"非数学对象"：
#   · 选项字母：答案是 A / AB / BCD / A,B,C,D,E
#   · 判断题值：答案是 正确 / 错误
#   · 概念文字：答案是 "标准差" / "移动平均法、时间序列分解法"
# 对这类题，Lean 只能产出 unknown，白烧一次编译 + 翻译（约 21s+）。
#
# official112 全 112 题标定实测：
#   豁免 18 道（选项 12 / 判断 2 / 概念文字 4），Lean 覆盖其余 94 道；
#   误伤（把该跑的题豁免掉）0 道；漏判 1 道（099：题面含偏微分方程，
#   被判为可形式化——**宁可多跑一道也不错杀数学题**，代价仅一次失败编译）。
#
# 判据与 classify_question_type 的证明题否决分开维护：这里的求值信号更宽
# （含 how many / number of / for which / 是多少 等），只用于"要不要上 Lean"，
# 不参与题型归类，避免把带这些措辞的证明题误判成解答题。

# 求值型信号（宽口径）：题面要求一个结果 → 答案应为数学对象。
_ASK_RESULT_RE = re.compile(
    r"\b(?:find|compute|determine|calculate|evaluate|what\s+is|what\s+are"
    r"|how\s+many|how\s+much|number\s+of|value\s+of|for\s+which|which\s+values?"
    r"|give)\b"
    r"|求(?!证)|计算|求解|等于多少|是多少|有多少|多少个",
    re.I,
)

# 明确数学运算符（**不含裸数字**：列表编号「10．」这种会误触发）
_MATH_OP_RE = re.compile(r"[\^=]|\\frac|\\sqrt|\\int|\\sum|\\cdot|\\times|[+*/<>]")

# 宽松选项标记：A. / B、 / C）...(选项连排无空格也能识别）
_LOOSE_OPTION_RE = re.compile(r"[A-E][\.、．)）]\s*\S")

# 客观题 / 概念题豁免阈值：选项标记 ≥3 个即视为客观题（比赛卷真选择题均 ≥3）
_OBJECTIVE_OPTION_MIN = 3


def lean_applicable(problem: str, question_type: str = "") -> tuple[bool, str]:
    """判断本题是否适用 Lean 形式化核验，返回 ``(是否适用, 不适用原因)``。

    不适用（答案不是数学对象，Lean 无从形式化）：
        1. 题型为选择题 / 判断题；
        2. 题面含 ≥3 个选项标记（含选项连排的无空格形态）；
        3. 题面既不要求一个结果、也不含任何明确数学运算符（概念/描述题）。
    其余一律适用 —— 即"除非常简单的题之外，都要用 Lean"。
    """
    text = problem or ""
    # ① 客观题：答案必然是字母或判断值
    if question_type in (QT_CHOICE, QT_JUDGE):
        return False, "objective_type"
    # ② 选项连排（如 A.甲B.乙C.丙D.丁）——宽松计数；2026-09-12 起并入 v2
    #    有序序列判据，二者取大（同理只增不减：只会让更多客观题被豁免 Lean，
    #    不会让原本豁免的题跑回 Lean 白烧编译时间）。
    _v2_n, _ = _option_sequence_length(text)
    if max(len(_LOOSE_OPTION_RE.findall(text)), _v2_n) >= _OBJECTIVE_OPTION_MIN:
        return False, "objective_options"
    # ③ 要求一个结果 → 答案是数学对象
    if _ASK_RESULT_RE.search(text):
        return True, ""
    # ④ 虽无求值措辞，但题面含明确数学运算符 → 仍可能是数学题（不俗杀）
    if _MATH_OP_RE.search(text):
        return True, ""
    # ⑤ 纯文字题（概念/描述/填空），答案不是数学对象
    return False, "non_math"


# ---------------------------------------------------------------------------
# 差异化策略提示（注入求解 prompt）
# ---------------------------------------------------------------------------
_TYPE_HINTS: dict[str, str] = {
    # 2026-09-12 客观题特化（official112 实测：15 道客观题基线仅对 4 道）。
    # 三条纪律直接对应实测错因：
    #   · 漏项 —— 093 只答 C（漏 E）、103 漏 E：根因是"选项连排/被截断"，
    #     检测层已修（E 项现能解析出），这里再强制"清点选项总数 + 判完复核"；
    #   · 单选惯性 —— 题面问"哪些是正确的"（多选）却只给一个字母；
    #   · 判真无依据 —— 只给结论不给反例，复查时无法自纠。
    QT_CHOICE: (
        "\n\n**【客观题特化解法 · 选择题 · 必须逐步执行】**\n"
        "1. **先清点选项**：从 A 起连续数到最后一个字母，逐一确认共几项"
        "（选项可能连排在同一行、或写在 LaTeX 列表里，**E 项最容易被漏读**）；\n"
        "2. **逐项独立判真**：对**每一个选项**单独判定真假。判为假时必须给出"
        "**具体反例或构造**（反例优先于文字论证）；判为真时给出可核验的理由"
        "（定义/定理原文，或代入计算）；\n"
        "3. **★ 角色 / 方向核对（最易错，必须列映射表）**：当选项形如“X 变为 Y”"
        "“由 A 得到 B”“增大↔减小”“左端↔右端”“系数↔常数项”“约束↔目标”"
        "“前件↔后件”等**对应关系**时，**先把原对象的每个元素与变换后元素列成"
        "映射表**（逐项对应），再核对该选项的表述是否与映射表一致；"
        "**严禁凭“大体相关 / 看起来对”判定为正确**；\n"
        "4. **按问法定项数**：题面问“哪些 / 正确的是”时，**所有**为真的选项都要"
        "选上——判完全部选项后**回头复核一遍**，禁止答出第一个正确项就收手；"
        "只有问“哪一项”时才单选；\n"
        "5. **★ 定稿前机械核对（必须执行，禁止跳过）**：写出最终答案之前，"
        "对**每一个选项字母**（从 A 起连续数到最后一项）逐一回答三问并记录结论：\n"
        "   ① 我**是否已经对它做过独立判定**？（**没判过的必须补判**；"
        "实测漏项（103 漏 E）的根因是“没判过”，而不是“判错了”——"
        "绝不允许把没判过的项默认排除）\n"
        "   ② 我判它为**真**还是**假**？\n"
        "   ③ 目前写下的答案里**是否包含**它？\n"
        "   **校验等式：答案字母集合 ≡ 判为真的选项集合**（既不多也不少）。"
        "不满足就必须改答案，改完再核对一次；\n"
        "6. **输出格式**：最终答案只输出选项字母，**连写**、不要逗号/空格/顿号、"
        "不要任何解释（正确示例：`AB`、`BCD`、`ABCDE`），写在 \\boxed{} 内。"
    ),
    QT_JUDGE: (
        "\n\n**【客观题特化解法 · 判断题 · 必须逐步执行】**\n"
        "1. **先提取核心断言**：明确被判断的对象、属性/关系、以及量词范围"
        "（全部还是存在）；\n"
        "2. **绝对化措辞 → 优先找反例**：断言含“一定 / 必然 / 都 / 只能 / 唯一 / "
        "任何 / 所有 / 不可能 / 必定”时，先尝试构造反例；**找到反例即判“错误”**"
        "（反例要具体，不能只写“不一定”）；\n"
        "3. **无反例则回定义核对**：逐条比对教材定义 / 定理原文，不要凭直觉、"
        "印象或类比下结论；特别注意“方向性”陷阱"
        "（增大↔减小、有偏↔无偏、有效↔一致 之间不可互换）；\n"
        "4. **★ 定稿前机械核对（必须执行，禁止跳过）**：写下最终判定之前，"
        "把断言拆成「对象 + 属性/关系 + 量词范围」三部分，逐部分与定义原文比对；"
        "若断言含方向性措辞（增大/减小、有偏/无偏、有效/一致、前件/后件），"
        "**必须先写出正确方向再对答案**，不得凭语感判断；"
        "最后自问一句「如果我判错了，最可能错在哪一部分？」\n"
        "5. **输出格式**：最终答案只输出 **正确** 或 **错误** 两个词之一"
        "（不要写 √ × T F 对 错 是 否 真 假），写在 \\boxed{} 内。"
    ),
    QT_PROOF: (
        "\n[题型策略] 本题为证明题。请逐步严格证明，每一步标注依据，"
        "并对关键步骤进行反复校验，确保逻辑严密、无跳步。"
    ),
    QT_FILL: (
        "\n\n**【客观题特化解法 · 填空/多空题 · 必须逐步执行】**\n"
        "1. **先数空**：从题面逐一清点共有几个空（或几问、几个 `(\\quad)`、"
        "几组“（ ）”），把总数 N 记下来；\n"
        "2. 按题面给出的空（或分问）的**先后顺序**逐一求解，每个空只给结果本身；\n"
        "3. 多个空的结果**用逗号在同一行依次分隔**（示例：`X, 8, 是`）；"
        "问“是 / 否”的空就直接写“是”或“否”；\n"
        "4. **★ 定稿前机械核对（必须执行）**：把最终答案按逗号切开，"
        "**数一数是否恰好 N 段**，再与第 1 步记下的 N 个空**逐位核对**"
        "（第 i 段是否确实是第 i 空的答案？有无错位、漏空、多写？）。"
        "段数不符或错位 → 必须修正后再提交；\n"
        "5. 不要写“设…=”“所以”“因此”之类的推导前缀，也不要重复题目已给的符号说明；\n"
        "6. **输出格式**：全部空的结果写在同一行，逗号分隔，整体放进 \\boxed{}。"
    ),
    QT_SOLUTION: (
        "\n[题型策略] 本题为解答题。请完整求解，"
        "并在【最终答案】给出简洁结果以供结果检测。"
    ),
}


def get_question_type_hint(qtype: str) -> str:
    """获取某题型的差异化策略提示片段；未知题型返回空串。"""
    return _TYPE_HINTS.get(qtype, "")


# ---------------------------------------------------------------------------
# 答案形态要求（B0 批次，2026-09-13）
# ---------------------------------------------------------------------------
# ★ 为什么**必须前置注入到生成阶段**（而不是事后闸门重问）：
#   事后闸门挂在 formatter（流程**最后一个阶段**），实测 003/015/074 三题
#   全部落到"时间不足一次重问，跳过" —— 因为 deadline 被 tier_budget 收紧到
#   1800s，而题目跑到 formatter 时已耗 1700–2400s，剩余必然 < 120s。
#   ⇒ 末端重问在时间上不可行（比赛 1200s 硬限更不可能）。
#   ⇒ 形态要求应在**规划/求解时**就写进 prompt，零额外时间成本。
_ASKS_ALL_RE = re.compile(
    r"find\s+all|determine\s+all|all\s+possible|求.{0,12}所有|求.{0,12}全部|"
    r"所有\s*可能|全部\s*可能|"
    # ★ 2026-09-15 补：「哪些 / 哪几种 / 有哪些 / 列举」同样是**穷尽性问法**。
    #   实测依据：099 题面为「常用的离散化方法有哪些」，正解是**多个方法**
    #   （有限差分法、有限元法/有限体积法），但旧正则只认「所有/全部」
    #   ⇒ 完备性要求**根本没注入**，模型只答了一个方法（漏项，与 103 漏选 E 同类）。
    #   ⚠ 选择题/证明题在 `answer_form_requirement` 开头即 return ""，
    #   不受本扩展影响（不会把选择题的 `ABCD` 要求成"具体数值"）。
    r"哪些|哪几[种个项条]|有哪些|列举|枚举|罗列",
    re.IGNORECASE)
_ASKS_RANGE_RE = re.compile(
    r"范围|区间|取值范围|充要条件|必要条件|充分条件", re.IGNORECASE)
# 极值类题（2026-09-14 B2 批次）：实测 016/066 都是"差一"失分——
# 只验证了 n 可行、没验证 n±1 不可行。112 题中约 48 道含极值措辞。
_EXTREMUM_RE = re.compile(
    r"smallest|largest|minimum|maximum|least\s+possible|greatest\s+possible"
    r"|minimal|maximal|最小|最大", re.IGNORECASE)


_ANSWER_FMT_DIRECTIVE_RE = re.compile(
    r"(Remember to put your final answer within\s*\\boxed\{\}\s*\.?|"
    r"Put your final answer within\s*\\boxed\{\}\s*\.?|"
    r"Please put your final answer within\s*\\boxed\{\}\s*\.?|"
    r"请将最终答案(?:放|写)在\s*\\boxed\{\}\s*(?:内|中)[。.]?)",
    re.IGNORECASE)


def strip_answer_format_directive(problem: str) -> str:
    """剥离题面里「把最终答案放进 \\boxed{}」这句格式指令。

    ★ 2026-09-14 实测：**112/112 题全部带这句**（题库统一附加）。

    为何必须剥离：它会随题面进入**每一个中间子目标**的 prompt ⇒
      ① 每个选项判定子目标都输出 `\\boxed{A}`/`\\boxed{B}`…，
         而不是要求的「结论：正确/错误」；
      ② **最后一个选项**子目标直接吐**最终答案**（#103 的「判定选项 E」
         就输出了 `\\boxed{ABCD}`）⇒ 聚合层认不出它是 E 的判定 ⇒ **丢项**。
    该指令只对**最终汇总**步骤有效，故在子目标 prompt 中剥离（merge 仍保留）。
    """
    if not problem:
        return problem or ""
    _t = _ANSWER_FMT_DIRECTIVE_RE.sub("", problem)
    return re.sub(r"\n{3,}", "\n\n", _t).strip()


def asks_all_values(problem: str) -> bool:
    """题面是否要求「所有 / 全部」取值（穷尽性搜索机制的触发条件）。

    2026-09-14：003（漏 2030）与 074（漏取整解族）的失败**不是形态问题**——
    模型自信地认为"只有一个解"，形态要求（必须枚举）对它无效。
    需要的是**穷尽性搜索**：强制按解族分类穷举。本函数供该机制判定是否触发。
    """
    return bool(_ASKS_ALL_RE.search(problem or ""))


def answer_form_requirement(problem: str, qtype: str = "") -> str:
    """按题面推导「答案形态要求」文本（无要求时返回空串）。

    覆盖台账 B0 两个可救模式：
      ① 求所有 → **必须枚举**（003 只给 `2026` 漏 `2030`）
      ② 求具体值 → **禁止条件式**（064 给 `m is even` 而非 `4`）

    ⚠ 选择题**直接返回空**：其答案形态（字母连写、选项数对齐）由
    `objective_injection`（客观题特化）负责，两套要求同时注入会互相干扰
    —— 例如把 093 的答案 `CE` 要求成"具体数值"。
    """
    t = problem or ""
    if not t:
        return ""
    if qtype in (QT_CHOICE, QT_PROOF):
        # 选择题：答案形态由 `objective_injection`（客观题特化）负责，
        #   两套要求同时注入会互相干扰（曾把 093 的 `CE` 要求成"具体数值"）。
        # 证明题：答案是**证明过程**而非数值/对象 ⇒ 注入"必须给具体值"有害。
        return ""
    _parts: list = []
    if _ASKS_ALL_RE.search(t):
        _parts.append(
            "\n\n**【答案形态要求 · 必须遵守】**\n"
            "本题问『所有 / 全部』的取值：最终答案**必须逐项枚举出全部解**，"
            "用逗号分隔（正确示例：`2026, 2030`）。\n"
            "· **禁止**只给出其中一个值；\n"
            "· **禁止**用 `≥ / ≤ / 任意 / for all` 等条件式代替枚举；\n"
            "· 收尾前必须自问一句『是否存在第二个解族 / 另一支解？』"
            "并确认已穷尽所有可能。"
        )
    elif not _ASKS_RANGE_RE.search(t):
        _parts.append(
            "\n\n**【答案形态要求 · 必须遵守】**\n"
            "本题要求**具体数值或具体对象**（不是取值范围、不是充要条件）：\n"
            "· 最终答案必须是一个明确的值，或明确的对象列表；\n"
            "· **禁止**用 `≥ / ≤ / is even / for all / 任意` 等条件式"
            "或性质描述代替具体值。"
        )
    # 2026-09-14 B2 批次：极值类题必须验证**边界严格性**。
    # 实测 016（答 20 / 正解 21）、066（答 3 / 正解 4）都是同一种失分：
    # **只验证了 n 可行，没验证 n±1 不可行** ⇒ 普遍性地"差一"。
    if _EXTREMUM_RE.search(t):
        _parts.append(
            "\n\n**【极值题要求 · 必须遵守】**\n"
            "本题求**极值**（最小/最大）。给出答案 n 时必须同时给出**两项**论证：\n"
            "· **可行性**：给出达到 n 的**显式构造**（不能只说『存在』）；\n"
            "· **严格性**：证明**相邻值**（求最小则 n−1，求最大则 n+1）"
            "**不可能达到** —— 这是最常被漏掉的一步。\n"
            "两项都成立才能提交 n；若发现 n 不可行或相邻值可行，必须修正答案。"
        )
    return "".join(_parts)


# ---------------------------------------------------------------------------
# 选择题选项提取（用于逆推验证）
# ---------------------------------------------------------------------------
def extract_options(problem: str) -> list[tuple[str, str]]:
    """从选择题题干提取选项文本，返回 ``[(标签, 内容), ...]``。

    2026-09-12 重写：改用与题型判定同源的「有序字母序列」定位选项，修掉三类
    实测漏检（这些漏检直接造成模型"漏选 E"与选项碎片）：

      · **连排无分隔**：`A.长期趋势B.季节变动C.循环变动D.不规则变动E.随机变动`
        —— 旧实现依赖 `选项/下列` 关键词定位起点，起点之后才切分，E 项也因
        正则写死 `[A-D]` 而被丢（103/106/107）；
      · **选项连排且无换行**：内容切分到"行尾/分号"截止，E 项被并进 D 的内容里
        （103 实测 `D.不规则变动E.随机变动`）；
      · **选项内嵌数学**：旧 `_OPTION_SPLIT_RE` 会把 `\\kappa(A)` 这类调用当成
        `(A)` 选项标记，098 实测切出 8 个碎片。

    支持 5 个选项（A–E）。提取失败返回空列表（不阻塞主流程，模型仍能看到完整题干）。
    """
    text = problem or ""
    if not text:
        return []
    marks = _option_sequence_marks(text)
    if len(marks) < 2:
        return []
    opts: list[tuple[str, str]] = []
    for i, (pos, label) in enumerate(marks):
        m = _OPTION_MARK_V2_RE.match(text, pos)
        content_start = m.end() if m else pos + 1
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        content = text[content_start:end]
        # LaTeX 列表残留（`\item[A.]` 剥掉标记后多一个 `]`）
        content = content.lstrip("]）) \t")
        # 到行尾/分号截止（同行连排时无换行，保留完整内容）
        content = re.split(r"[\n；;]", content)[0].strip().rstrip("，,；;")
        if content:
            opts.append((label, content))
    return opts


def format_options(problem: str) -> str:
    """把选择题选项格式化为清单文本，注入求解 prompt；无选项返回空串。"""
    opts = extract_options(problem)
    if not opts:
        return ""
    lines = ["\n[已知选项]"]
    for label, content in opts:
        lines.append(f"{label}. {content}")
    return "\n".join(lines)


def objective_injection(problem: str, qtype: str, tactic_on: bool = True) -> str:
    """客观题注入文本（选项清单 + 特化解法纪律）；非客观题返回空串。

    统一入口：**三条生成路径共用**——主求解（solver）、子目标主路径 2.7 的规划、
    子目标逐步求解与 merge 合并（sub_goal_solver）。子目标主路径对所有档位默认
    先行（``enable_subgoal_main_path`` 默认 True），客观题的答案完全可能由该路径
    产出，只在 solver 里注入等于没注入（"机制触发 ≠ 效果提升"的老坑）。

    注入位置约定：**必须拼在"题目"占位符内部**，不得追加到提示词末尾——历史
    教训（algebra-075）显示末尾追加会破坏提示词收尾结构，让模型进入续写模式并
    泄漏 `[续写]` 占位符，把答对的题变成错的。
    """
    if qtype not in (QT_CHOICE, QT_JUDGE, QT_FILL):
        return ""
    parts: list[str] = []
    if qtype == QT_CHOICE:                 # 选项清单属"输入信息补全"，始终注入
        opts = format_options(problem)
        if opts:
            parts.append(opts)
    if tactic_on:                          # 特化解法纪律受开关控制
        hint = get_question_type_hint(qtype)
        if hint:
            parts.append(hint)
    return "".join(parts)

# ---------------------------------------------------------------------------
# §3 方案 A（2026-10-13）：题型 LLM 判题 + 交叉校验 + 路径分流表
#   ★ 2026-10-14（任务列表 #1）：本块产出的判题结论**已并入 2.6 题意理解**（不再是
#     独立阶段 1.1_type_understanding）；同时修复了"中文题型 ⇄ 类别键"从未互转
#     导致交叉校验与下游消费整体失效的缺陷（见 `QT_TO_KEY` / `merge_type_decision`）。
# ---------------------------------------------------------------------------

TYPE_LLM_SYSTEM = """你是数学题**题型判定**专家。请判断题目属于下列哪一类，并只输出一个 JSON 对象。

类别（必须选其一）：
- "proof"      ：证明题（要求证明某命题）
- "solution"   ：解答题（求解某个结果：数值/表达式/最值/个数/构造）
- "choice"     ：选择题（给了 A/B/C/D 等选项）
- "boolean"    ：判断题（要求判断对错/是否成立）
- "fill_blank" ：填空题（要求填写数值或表达式）

输出格式（严格）：
{"type": "<类别>", "confidence": 0.0~1.0, "evidence": "判定依据（引用题面特征，一句）"}

⚠ 只输出 JSON，不要解释、不要 markdown。拿不准时 confidence 给低值（<0.6）。"""

TYPE_LLM_USER = """## 题目
{problem}

## 规则基线的判定（供参考，可能不准）
{rule_guess}

请输出题型判定 JSON。"""

# 路径分流表（§3.4）：不同题型走不同路径。
# ★ 保守设计：未知题型一律按"全走"（与接线前一致），**不得因判题而降低任何题型的处理质量**。
TYPE_PATH_PLAN = {
    "proof": {"lean": True, "options": False, "answer_form": "proof"},
    "solution": {"lean": True, "options": False, "answer_form": "value"},
    "choice": {"lean": False, "options": True, "answer_form": "option"},
    "boolean": {"lean": False, "options": False, "answer_form": "yesno"},
    "fill_blank": {"lean": False, "options": False, "answer_form": "value"},
}
_TYPE_PATH_DEFAULT = {"lean": True, "options": False, "answer_form": ""}

# ---------------------------------------------------------------------------
# 两套题型命名空间的显式互转（★ 2026-10-14 修复；随任务 #1「题型判断并入题意理解」）
# ---------------------------------------------------------------------------
# 本项目里的题型有**两套命名**，此前从未互转，导致 §3 交叉校验整体失效
# （机理见 `merge_type_decision` 的长注释）：
#   · **中文题型常量**（`QT_*`）：全仓 130+ 处消费点按它判定
#     （`objective_injection` / `answer_form_requirement` / `lean_applicable` /
#      `audit_gate` / `solver__prompt` …）；
#   · **LLM 类别键**（`TYPE_LLM_SYSTEM` 要求模型输出的 "proof"/"solution"/
#     "choice"/"boolean"/"fill_blank"）：`TYPE_PATH_PLAN` 按它索引。
# ⇒ 铁律：凡进入 `ctx.question_type` 的值**必须是中文题型**（既有契约）；类别键
#   只用于查 `TYPE_PATH_PLAN`。两者一律经 `type_key()` / `qt_of_key()` 互转，
#   **禁止裸比较或裸写入**。
QT_TO_KEY = {
    QT_PROOF: "proof",
    QT_SOLUTION: "solution",
    QT_CHOICE: "choice",
    QT_JUDGE: "boolean",
    QT_FILL: "fill_blank",
}
KEY_TO_QT = {_k: _q for _q, _k in QT_TO_KEY.items()}

# 规则基线的**强信号**（§3.3）：规则判为选择题/判断题时视为"规则可信"，LLM 不得覆盖
# —— 这两类的形态标记（选项/判断措辞）在题面上极其专一，且规则判据已背书上位
# 七轮误判修复（PB 39/60、009、045、086、087/098/102/103/106/107）。
# ⚠ 用**类别键**表达（原写成中文标记串 `_STRONG_SIGNAL_MARKERS`，与比较用的英文
#   命名空间不同源 ⇒ 守卫恒假，见 `merge_type_decision`）。
_STRONG_SIGNAL_KEYS = ("choice", "boolean")


def type_key(qtype) -> str:
    """题型（中文常量**或**类别键、任意大小写）→ 统一类别键；未知/空 → ``""``。

    ★ 这是两套命名空间之间**唯一**的入口，任何"比较题型"的地方都必须先经它归一。
    """
    s = str(qtype or "").strip()
    if not s:
        return ""
    if s in QT_TO_KEY:                     # 中文题型常量
        return QT_TO_KEY[s]
    _k = s.lower()
    return _k if _k in TYPE_PATH_PLAN else ""


def qt_of_key(key) -> str:
    """类别键（或中文题型）→ 中文题型常量；未知/空 → ``""``（**不猜**）。"""
    return KEY_TO_QT.get(type_key(key), "")


_ANSWER_FORM_HINT = {
    "proof": "本题是**证明题**：必须给出完整、逐步的证明，不得只给结论或写「显然」。",
    "value": "本题要求**求出具体结果**：结论写成可核验对象（数值/表达式/构造），并说明推导。",
    "option": "本题是**选择题**：最终必须明确给出所选**选项字母**，并简述排除依据。",
    "yesno": "本题是**判断题**：先明确「成立 / 不成立」，再给理由（若不成立须给反例）。",
}


def render_type_path_hint(type_path) -> str:
    """把路径分流表渲染成「答题形态提示」（§3.4 的**消费者**，2026-10-13）。

    ★ 设计纪律：分流 = **针对性增强**，**不是裁剪** —— 项目铁律
    「『不研究某类』≠『降低其处理质量』」⇒ 本提示只**增加**针对性要求
    （如选择题必须给选项字母），**绝不**用题型去关闭任何验证 / 检索环节。

    返回空串表示无提示（未知题型/表为空）⇒ 下游零成本。
    """
    if not isinstance(type_path, dict):
        return ""
    _h = _ANSWER_FORM_HINT.get(str(type_path.get("answer_form") or ""))
    return ("【答题形态（依题型分流）】" + _h) if _h else ""


def type_path_plan(qtype: str) -> dict:
    """题型 → 路径计划（§3.4）。未知题型按默认（全走）⇒ 行为与接线前一致。

    ★ 2026-10-14：入参**中文题型常量与类别键都接受**（经 `type_key()` 归一）——
    此前只认英文键，而调用方传的是中文 `ctx.question_type` ⇒ 除 LLM 覆盖那条路径
    外**恒返回默认计划**（"分流表形同虚设"）。
    """
    return dict(TYPE_PATH_PLAN.get(type_key(qtype), _TYPE_PATH_DEFAULT))


def parse_type_json(raw) -> dict:
    """解析 LLM 判题输出（容错）。返回 {} 表示无法解析（调用方回退规则）。

    ★ 2026-10-14：`type` 字段一律归一为**类别键**（`type_key()`）—— 模型偶尔会
      回中文题型（"证明题"），归一后与英文键同源，下游不必再分支。
    """
    if not raw:
        return {}
    txt = str(raw).strip()
    try:
        import json as _json_t
        _m = txt[txt.find("{"):txt.rfind("}") + 1] if "{" in txt else ""
        if _m:
            d = _json_t.loads(_m)
            if isinstance(d, dict):
                _k = type_key(d.get("type"))
                if _k:
                    d["type"] = _k
                    return d
    except Exception:  # noqa: BLE001
        pass
    low = txt.lower()
    for k in TYPE_PATH_PLAN:                     # 英文类别键兜底
        if k in low:
            return {"type": k, "confidence": 0.5,
                    "evidence": "正则兜底（JSON 不可解析）"}
    for _qt, _key in QT_TO_KEY.items():          # 中文题型兜底
        if _qt in txt:
            return {"type": _key, "confidence": 0.5,
                    "evidence": "正则兜底（JSON 不可解析，命中中文题型）"}
    return {}


def merge_type_decision(rule_type: str, llm_type: str,
                        llm_conf: float = 0.0) -> dict:
    """**交叉校验**（§3.3）：规则强信号优先；否则采纳 LLM，但记录分歧。

    为什么必须保守：`question_type` 被 130+ 处读取（含 7 次以上历史误判修复：
    PB 39/60、009、045、086、087/098/102/103/106/107）⇒ 让 LLM 无条件覆盖会**误伤**
    本来能答对的题（code-audit 路线 C：「不研究某类」≠「降低其处理质量」）。

    返回字段：
      · ``final``     —— **中文题型常量**，与 `ctx.question_type` **同一命名空间**
        （既有契约；**不得**写类别键，否则 `objective_injection` /
        `answer_form_requirement` / `lean_applicable` / `solver__prompt` /
        `audit_gate` 等按中文常量判定的下游模块会**整段失效**）；
      · ``final_key`` —— 类别键（英文），供 `type_path_plan` 索引 `TYPE_PATH_PLAN`；
      · ``rule`` / ``llm`` —— 归一后的**中文题型**（报告可读，未识别时保留原串）；
        ``rule_key`` / ``llm_key`` 为对应类别键；
      · ``overrides`` 非空 = 发生了"LLM 覆盖规则"或"规则否决 LLM"，供 diag 审计。

    ⚠ `final` **严格限定在中文题型常量或空串**：未识别的输入一律回落空串
      （调用方据此不覆盖 `ctx.question_type`，由惰性属性给出规则判定），
      **不得**把原始串/类别键透传进契约域。

    ★ 2026-10-14（随任务 #1「题型判断并入题意理解」的随附修复）：
      原实现**直接拿中文规则值与英文 LLM 值比较**，在同一行代码里造成三重失效：
        ① `l == r` 永不成立（"解答题" ≠ "solution"）⇒ `reason` 恒为"LLM 覆盖"，
           低置信/一致等分支全部不可达；
        ② `if r in ("choice", "boolean")` 中的 `r` 是中文常量（"选择题"/"判断题"）
           ⇒ **"规则强信号优先"守卫恒为假** —— §3.3 的核心保护**从未生效过**；
        ③ 返回的 `final` 是英文类别键 ⇒ 写进 `ctx.question_type` 后**污染中文契约**：
           证明题变成 `"proof"` 后 `answer_form_requirement` 不再认它是证明题，
           于是反过来给它注入"必须给出具体数值"（与 028/034/061 的标定结论相悖）；
           选择题变成 `"choice"` 后 `objective_injection` 直接返回空串
           ⇒ **选项清单与客观题特化解法整段丢失**。
      现改为：两侧先经 `type_key()` 归一为**类别键**再比较（比较域统一），
      `final` 再经 `qt_of_key()` 回转**中文**（契约域统一）。比较用键、契约用中文。
    """
    _rk = type_key(rule_type)
    _lk = type_key(llm_type)
    _r_qt = qt_of_key(_rk)                     # 未识别 → ""（**不把原始串塞进契约域**）
    _l_qt = qt_of_key(_lk)
    out = {"final": _r_qt, "final_key": _rk,
           "rule": _r_qt or str(rule_type or "").strip(), "rule_key": _rk,
           "llm": _l_qt or str(llm_type or "").strip(), "llm_key": _lk,
           "confidence": float(llm_conf or 0.0), "overrides": [], "reason": ""}
    if not _lk or _lk == _rk:
        out["reason"] = "一致或无 LLM 结果"
        return out
    # 规则**强信号**（选择题/判断题的形态标记）⇒ 规则优先，拒绝 LLM 覆盖
    if _rk in _STRONG_SIGNAL_KEYS:
        out["overrides"].append({"by": "rule", "from": _lk, "to": _rk,
                                 "reason": "规则命中强信号（选项/判断形态）"})
        out["reason"] = "规则强信号优先"
        return out
    # 其余：采纳 LLM（但低置信不覆盖，避免抖动）
    if out["confidence"] >= 0.6:
        out["final"] = qt_of_key(_lk)      # ★ 契约字段必须是中文题型
        out["final_key"] = _lk
        out["overrides"].append({"by": "llm", "from": _rk, "to": _lk,
                                 "reason": "置信 %.2f ≥ 0.6" % out["confidence"]})
        out["reason"] = "LLM 覆盖（高置信）"
    else:
        out["reason"] = "LLM 置信不足（<0.6）⇒ 保留规则判定"
    return out
