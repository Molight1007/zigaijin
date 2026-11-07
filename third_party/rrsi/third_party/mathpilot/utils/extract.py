from __future__ import annotations
"""答案提取与格式化工具"""

import re

# 2026-10-08 批 D-ex1：抽取内核 + 救援兜底簇已逐字搬到 utils/extract__core.py。
# 跳板式 re-export ⇒ `from utils.extract import <X>` 零改动 + **同一对象 is**。
# ⚠ 勿按「未使用」清理（契约面）。
from .extract__core import (  # noqa: F401
    _META_PATTERNS, _META_RE, _is_meta_line,
    _is_incomplete_answer, _SHELL_ONLY, _TITLE_ONLY,
    _ANSWER_CONTENT, _has_answer_content, _extract_last_valid_answer,
    extract_final_answer, _CALC_BLOCK_RE, _CALC_STRAG_RE,
    _TOOLCALL_BLOCK_RE, _TOOLCALL_OPEN_RE, _TOOLCALL_STRAG_RE,
    _strip_toolcall_markers, _strip_calc_markers, _extract_final_answer_impl,
    _CONTINUATION_MARKERS, _strip_continuation_markers, clean_answer,
    _extract_boxed_nested, _ANSWER_PATTERNS, _WEAK_ASSIGN_PATTERN,
    _extract_strong_pattern, _extract_tail_fallback, _REASONING_FRAGMENT_WORDS,
    _has_unbalanced_math_delims, _clean_extracted_answer, _looks_like_reasoning_fragment,
    rescue_final_answer,
)


def format_response(answer: str) -> str:
    """确保 final_response 非空且可序列化"""
    if answer is None:
        return ""
    answer = str(answer).strip()
    return answer


def _looks_like_answer(s: str) -> bool:
    """判断一段文本**是否长得像一个最终答案**（而不是推理句 / 元话语）。

    ★★ 2026-10-06 新增（**候选池被"非答案"污染** —— 上游根因，实测驱动）：
    两轮 32 个候选实测：**59% 的 `answer` 字段不是答案**，而是
      · 推理句：`'因此，似乎没有满足条件的路径，答案为 0。'`
      · 元话语：`'让我搜索这个问题的标准解法'`、`'步骤 10：搜索相关结果'`
      · 英文推理片段：`'If there are three or more non-zero numbers, consider the su'`
      · 半截标题：`'步骤23：关键洞察'`

    根因：`smart_fallback_answer` 的「策略 1」用**关键词**匹配行
    （正则含 `答案|故选|正确`），于是**含"答案"二字的叙述句被整句收下**。
    后果：候选池 59% 是垃圾 ⇒ **检错器验垃圾、投票投垃圾、修正无从下手**
    （题 002 的 6 个候选里 4 个逐字相同，就是同一份错答案被复制）。

    判据**保守、宁缺勿滥**（宁可返回空串让调用方重试，也不塞一句叙述句）：
      1. 长度 ≤ 120 字（真答案通常很短：`20460` / `$2-2m$` / `\\frac12` / `B`）；
      2. 不含**元话语**（让我/我需要/重新考虑/搜索/似乎/也许/但是…）；
      3. **不以句末标点结尾**（`。`/`？`/`！`/`.` 结尾基本是叙述句）；
      4. 含**可判分的实质**（数字、LaTeX、`\\boxed`、选项字母、等号）。
    """
    t = " ".join(str(s or "").split())
    if not t or len(t) > 120:
        return False
    # ★ 2026-10-06 补：**步骤标题**不是答案（实测 `'步骤23：关键洞察'` 含数字、
    #   无元话语、也不以句号结尾，前三条判据都放行 ⇒ 必须单独拦）。
    if _STEP_HEADER_RE.match(t):
        return False
    if _META_ANSWER_RE.search(t):
        return False
    if t[-1] in "。？！.；;":
        return False
    # 必须含"可判分的实质"：数字 / LaTeX / 选项 / 等号 / 答案提示词
    return bool(re.search(r"[\d$\\=]|\bboxed\b|^[A-E]$|答案", t, re.I))


# 步骤/章节标题形态（`步骤23：…` / `第 3 步` / `Step 13:`）—— 它们是进度标记，不是答案
_STEP_HEADER_RE = re.compile(
    r"^\s*(步骤\s*\d*|第\s*\d+\s*步|step\s*\d*|phase\s*\d+)\s*[：:.\s]",
    re.IGNORECASE)


# 元话语 / 推理语气（出现在"答案"里就说明它不是答案）
_META_ANSWER_RE = re.compile(
    r"(让我|我们来|我们需要|需要重新|重新考虑|重新审|搜索|查询|确认|尝试|"
    r"似乎|也许|可能|但是|不过|然而|注意到|这说明|这意味着|"
    r"let\s+me|let'?s|we\s+need|perhaps|maybe|however|search|wait\b)",
    re.IGNORECASE)


def _salvage_value_from_line(line: str) -> str:
    """从一句**叙述句**里抢救出"可判分的值"，而不是把整句当答案。

    ★ 2026-10-06 新增（配合入池门槛；由回归测试暴露的必要细分）：
    必须区分两类"句子形态"，不能一刀切：
      · `'所以最小 d 是 48。'`            → **有明确最终值** ⇒ 应抽出 `48`
        （`tests/test_tool_residue_fallback_paths.py` 锁定了这个行为：
         "正文里没有 boxed 时，尾部行兜底仍要给出正文（而不是空）"）
      · `'因此，似乎没有满足条件的路径，答案为 0。'` → 含 **`似乎`**（不确定语气）
        ⇒ 拒绝（它自己也说"没有路径"，值不可信）
      · `'因此水平边确实是有向的：形成一个有向环 0→1→2→…→19→0。'`
        → 无 `是/为/=` 引导的句末值 ⇒ 抽不出 ⇒ 返回空串（正确）

    判据：**先排除不确定语气**，再尝试 `\\boxed{}` 或「是/为/等于/= <值>」句末形态。
    """
    t = " ".join(str(line or "").split())
    if not t:
        return ""
    if _META_ANSWER_RE.search(t):
        return ""
    m = re.search(r"\\boxed\s*\{([^{}]*)\}", t)
    if m:
        v = m.group(1).strip()
        return v
    # 句末「是/为/等于/= <值>」，值不含标点
    m = re.search(r"(?:是|为|等于|=)\s*\$?([^，。；;、\s]{1,40})\$?\s*[。.]?$", t)
    if m:
        v = m.group(1).strip(" $。.")
        if v and re.fullmatch(r"[-\d./^()a-zA-Z\\{}]*\d[-\d./^()a-zA-Z\\{}]*", v):
            return v
    return ""


# ---------------------------------------------------------------------------
# 候选**入池判据 v2**（2026-10-07 新增；`pool_gate_v2` 开关消费）
# ---------------------------------------------------------------------------
# 背景（smoke10 2026-10-06，`run_2026-10-06_smoke10_...jsonl` 实测）：
#   上一轮的入池门槛用 `_looks_like_answer`，实测**两头都错**：
#   · **漏网**（拦住的是垃圾判成答案）：4 条"规划/设定句"入池并被选为终答，
#       题 002 `'证明这个最小值就是所求的 $T$，并验证其最优性'`（成了唯一幸存
#         候选、被 final_select③ 直接采用、`predicted` = 该句、置信度 0.00）；
#       题 000 `'子问题4：计算满足条件的路径总数'`（自改进"保留原版"落盘）、
#         `'{(x,y) | 0 ≤ x ≤ 19, 0 ≤ y ≤ 2}，共 20 × 3 = 60 个元素'`（revise 入池）；
#       题 007 `'- 子问题4：求和并验证结果'`。
#       根因：`_looks_like_answer` 只拦 `步骤N`（`_STEP_HEADER_RE`），不拦
#         `子问题N/子目标N`；且只要含数字/`$`/`=` 就放行，于是"描述性句子"漏网。
#   · **误杀**（真答案判成垃圾）：题 087 的正确答案 `BD`、题 088 的 `AB`
#       （多字母选项答案）被判 False 而**拒绝入池** —— trace 实证
#       087「候选#1..#4 不含最终答案 → 不入池（answer 前 40 字：'BD'）」共 4 次、
#       088 同型共 5 次（gold 分别为 `B, D` / `AB`）。`_looks_like_answer` 的
#       选项判据 `^[A-E]$` 只认**单字母**，多选答案必然误杀。
#       另经 official112 全量 gold 扫描（`looks_like_pool_answer(gold)`）：
#       上一轮判据对 **8/112** 条 gold 判 False（判误杀面 ≈ 7.1%）——除 087/088
#       的选项答案外，还有 101/111 的判断题 `正确`/`错误`。本次修掉其中**两类
#       无争议形态**（多字母/带分隔符的选项、判断题判定词）；其余 5 条为
#       填空题/论述题的**自然语言 gold**（037 `taking the empty card`、
#       095/108 长文本、104 `移动平均法、时间序列分解法`、105 `标准差`）。
#       ★ 2026-10-07（B）：这 5 条已由**下游 v3 自然语言容忍层**修复 ——
#         默认口径去掉"必含数值"白名单（修回 037/104/105），显式传
#         `question_type` 时再放宽句末标点/长度（修回 108/095）。
#       ★ 2026-10-07（ledger2-B2）**复验补修**：095（in-scope 解答题，gold 为两段
#         `\[…\]`）在**默认口径**下也被长度闸门误杀 ⇒ 追加"含显示数学块则不按长度拒"
#         的例外（零新增误收，见下）。108（统计论述句）仍需 tolerant 口径，**不默认放开**。
#         详见下方「入池判据 v3」段（含"题型无法自动区分"的关键实测）。
#
# v2 判据 = 先补**两条确定性修正**（修误杀，不扩大误杀面），再修漏网，最后复用 v1：
#   ① 选项答案：单/多字母 + 可选逗号/顿号/斜杠分隔（`BD` / `B, D` / `A,C`）⇒ 答案。
#      ⚠ 2026-10-07 复验修正：初版**要求分隔符** ⇒ 连写的 `BD`（题 087 模型实际输出、
#      实测被 v1 拒 4 次）仍被误杀。改为分隔符可选、上限 4 字母（`BD`/`AB`/`ABD`）。
#      代价：`CAB`/`BED` 这类 ≤4 的 A-E 字母串会被放行为答案（误收）；权衡下远优于
#      误杀真答案（多收一个候选由下游 final_selector/verifier 兜住，误杀则直接丢正确答案）。
#   ② 判断题判定词（`正确`/`错误`/`对`/`错`/`真`/`假`/`是`/`否`）⇒ 答案；
#   ③ `\boxed{...}` 显式形态 ⇒ 直接视为答案（模型显式定界，不拦）；
#   ④ 规划/设定描述句 ⇒ 不是答案（修漏网），见 `_looks_like_plan_or_setup`；
#   ⑤ 其余交回 `_looks_like_answer`（口径不变）。
_OPTION_ANSWER_RE = re.compile(r"^[A-E](?:\s*[,，、/]?\s*[A-E]){0,3}$")
_VERDICT_ANSWER_RE = re.compile(r"^(?:正确|错误|对|错|真|假|是|否)$")
# 规划/步骤**标题**（`子问题4：…` / `- 子目标3: …` / `第 3 步` / `Step 13:`…）——
# v1 只认 `步骤N`，此处补齐题干里最常见的 `子问题/子目标/子任务` 变体。
_PLAN_HEADER_RE = re.compile(
    r"^\s*[-*•·]?\s*[（(【\[]?\s*\**\s*"
    r"(?:子问题|子目标|子任务|任务|步骤|第\s*\d+\s*步|step|phase)\s*\d*\s*[：:、.]",
    re.IGNORECASE)
# 祈使/任务动词开头（`证明…` / `计算…` / `求所有…`）⇒ 描述"要做什么"，不是"结果是多少"。
_TASK_VERB_HEAD_RE = re.compile(
    r"^\s*[（(【\[]?\s*\**\s*"
    r"(?:证明|求证|验证|求出?|计算|推导|化简|说明|讨论|分析|构造|判断|确定|列举|列出|给出)"
    r"\s*[：:，,、]?")
# 描述性从句：`，共 20 × 3 = 60 个元素`（设定/计数说明）
_DESC_COUNT_CLAUSE_RE = re.compile(r"[，,]\s*共\s*\d")
# 描述性从句：`…，并验证其最优性`（把两个任务用顿号并列，仍是任务描述）
_DESC_CONJ_VERB_RE = re.compile(r"[，,]\s*并\s*(?:验证|证明|求出?|计算|说明|讨论|构造)")


# 列表项 + 推导引导词（" - 依赖：…" / "• 因此…"）⇒ 推导步骤描述
_LIST_ITEM_META_RE = re.compile(
    r"^\s*[-•*·]\s*(依赖|所以|因此|故|需要|构造|证明|由|根据|利用|考虑|将|令|设)")


def _looks_like_plan_or_setup(s: str) -> bool:
    """该文本是否是**规划/设定描述句**（任务描述、步骤标题、计数说明）而非答案。

    这是 v2 相对 v1 的**漏网修正层**：只收"确定不是答案"的形态（标题/祈使句/
    描述性从句），不碰任何可能为真答案的形态 —— 避免扩大误杀。
    """
    t = " ".join(str(s or "").split())
    if not t:
        return True
    if _PLAN_HEADER_RE.match(t):
        return True
    if _TASK_VERB_HEAD_RE.match(t):
        return True
    if _DESC_COUNT_CLAUSE_RE.search(t):
        return True
    if _DESC_CONJ_VERB_RE.search(t):
        return True
    # ★ 2026-10-24（工具测试台 bench:normalize_meta_speech 抓到）：
    #   列表项形态 + 引导词 ⇒ 是推导步骤的描述，不是答案。
    #   实例：`- 依赖：子问题1的上界`、`- 因此需要构造序列`——它们曾被判为
    #   "可用答案"入池（实测 6 个候选里有 3 个是这种），污染投票与检错。
    #   只收"确定不是答案"的形态：**必须以列表符开头**且紧跟引导词，
    #   避免误杀 `-5`、`—1/2` 这类负数值答案。
    if _LIST_ITEM_META_RE.match(t):
        return True
    return False


# ---------------------------------------------------------------------------
# 入池判据 v3（2026-10-07 新增）：**自然语言答案容忍层**（修 B：5/112 gold 误杀）
# ---------------------------------------------------------------------------
# 背景：v2 对 official112 全量 gold 扫描后仍误杀 5 条**自然语言 gold** ——
#   037 `taking the empty card`、095 LZ 长文本、104 `移动平均法、时间序列分解法`、
#   105 `标准差`、108 论述句。根因：v1/v2 的**白名单**要求答案含"可判分实质"
#   （数字 / LaTeX / `\boxed` / 等号 / 选项），而这几条 gold 是自然语言描述。
#
# ★ 关键实测（决定本条的口径）：
#   · `classify_question_type` 与 `tools/filter_testset.py::infer_type` 把这 5 条
#     **全部判为 `解答题`**（且 filter 会**保留**它们，107/112）⇒ 它们与题
#     000/002/007（gold 为数值、其被拦候选是"规划句"）**同题型**
#     ⇒ **"题型感知"无法自动区分**，只能做成"调用方可选参数"。
#   · 故改为**黑名单替代白名单**：默认不再要求"含数值"，只要求"不像推理句"。
#   · 泄漏量化（现存 538 条唯一候选 + 112 条 gold 全扫；`_scan_*.py` 实测）：
#       - 默认口径（去掉"必含数值"+ 结构兜底）⇒ 新增入池 **0**（原 4 条候选
#         `</tool_call>` / `【修正过程】` / `【错误分析】` / `正确的Boris策略：`
#         被结构兜底全兜住）；修复 037/104/105。
#       - 容忍口径（再放宽"句末标点 / 长度"）⇒ 新增 **44 条**（多为中文推理句）
#         ⇒ **不可默认放开**，故置于 `question_type` 参数之下；修复 108/095。
#   ★ 2026-10-07（ledger2-B2）复验：在 **537 条真实候选**语料上重测，容忍口径新增
#     **45** 条（全是 `但…。` / `So the condition…` 这类推理句）、并**丢失 1 条**
#     （`$$\sum…$$` 真答案）⇒ 证实"容忍口径是有害的，不能开在生产调用点"。
#     故 095 改由**默认口径的最小放宽**（仅长度闸门对显示数学块例外）修复：
#     同语料新增 0 / 丢失 0；108 保留在 tolerant 之下（调用方不传 `question_type` ⇒
#     生产链路**不触达**）。
# ---------------------------------------------------------------------------
_POOL_TOOL_TAG_RE = re.compile(
    r"</?tool_calls?>|</?function\b|<parameter=|</parameter>", re.IGNORECASE)
# 段落标题壳（`【修正过程】` / `【错误分析】` / `[结论]`）—— 只有标签、没有内容
_POOL_SECTION_ONLY_RE = re.compile(
    r"^\s*[（(【\[]\s*[^（()）\[\]】]{0,16}"
    r"(?:过程|分析|检查|矛盾|修正|结论|说明|步骤|思路|推导|证明|计算|答案|结果)"
    r"\s*[）)】\]]\s*$")
# 尾部冒号标签（`正确的Boris策略：`）—— 标题而非内容
_POOL_TRAIL_LABEL_RE = re.compile(r"[:：]\s*$")
# 英文推理起始（`If there are…` / `We can…` / `Consider…`）—— 推理句不是答案
_POOL_EN_REASONING_RE = re.compile(
    r"^(?:if|we|let|consider|assume|suppose|since|because|now|then|first|next|"
    r"note|observe|here|thus|hence|after|before|however|therefore)\b", re.IGNORECASE)
# 中文推理起始（锚串首）—— 刻意**不含** `标准`/`即`/`但`/`覆盖` 等会误伤真答案的
# 裸词（实测 `标准差` 会被裸 `^标准` 误杀 ⇒ 精确到 `标准(结果|答案|…)`）。
_POOL_CN_REASONING_RE = re.compile(
    r"^(?:实际上|经过|继续|目前|在答案中|通常|现在|假设|关键|由|因此|所以|同时|"
    r"然后|缺少|正确的分析|这与我|这个(?:问题|题|推理|分析)|我们|考虑|首先|接下来|"
    r"由此|综上|注意到|由于|已知|结论(?:无效|是|为)?[:：]|对于本题|对于这种|对于固定|"
    r"这种在|覆盖格子|若第|最后一步|最终的正确|以上|上述|一切|综上所|总而言之)")
_POOL_SENT_TERM = "。！？!?"
# 显示数学块标记（`\[…\]` / `$$…$$` / `\begin{…}`）—— 答案里出现整块 LaTeX 排版时，
# 文本自然变长（题 095 的 gold 是两段 `\[…\]`），此时**不应再按"长度"当推理句拒掉**。
# 实测（537 条真实候选 + 112 条 gold）：放宽后**零新增误收**，只修回 095。
_POOL_DISPLAY_MATH_RE = re.compile(r"\\\[|\$\$|\\begin\{")


def _looks_like_hard_noise(t: str) -> bool:
    """**硬黑名单**：容忍口径下的兜底拦截层（只收"确定不是答案"的形态）。

    维度：工具调用残留 / 段落标题壳 / 尾部冒号标签 / 步骤标题 / 元话语 /
    英文·中文推理起始 / **多句叙述**（≥2 个句末标点）。
    刻意不碰"可能为真答案"的形态（避免扩大误杀）；实测可把容忍口径的新增收
    从 147 压到 44（详见上方 v3 背景）。
    """
    if not t:
        return True
    if _POOL_TOOL_TAG_RE.search(t):
        return True
    if _POOL_SECTION_ONLY_RE.match(t):
        return True
    if _POOL_TRAIL_LABEL_RE.search(t):
        return True
    if _STEP_HEADER_RE.match(t):
        return True
    if _META_ANSWER_RE.search(t):
        return True
    if _POOL_EN_REASONING_RE.search(t) or _POOL_CN_REASONING_RE.search(t):
        return True
    if sum(t.count(c) for c in _POOL_SENT_TERM) >= 2:
        return True
    try:
        if _looks_like_reasoning_fragment(t):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


# 自然语言答案**合法**的题型（这些题型的 gold 可能是自然语言，不要求含数值）。
# ⚠ 仅当调用方**显式传入** `question_type` 时才启用（默认 None ⇒ 严口径，零行为变化）；
#   `选择题` 不在内（其答案形态是选项字母，由 ① 直接放行）。
_NL_ANSWER_QUESTION_TYPES = frozenset({
    "解答题", "证明题", "填空题", "判断题", "论述题", "简答题",
})


def _nl_answer_tolerant(question_type) -> bool:
    """调用方传入的题型是否允许"自然语言答案"（进而放宽白名单 + 句末标点 + 长度）。"""
    return str(question_type or "").strip() in _NL_ANSWER_QUESTION_TYPES


def _looks_like_answer_v3(t: str, tolerant: bool) -> bool:
    """v3 判据：**白名单 → 黑名单**。

    · 结构兜底（宽严共用）：工具残留 / 标题壳 / 尾部标签 ⇒ 一律拦；
    · `tolerant=False`（默认）：= v2 的 `_looks_like_answer` **去掉"必含数值"白名单**，
      其余（长度 ≤120 / 非步骤标题 / 无元话语 / 不以句末标点结尾）逐条保留 ⇒
      实测零新增误收，修复 037/104/105；
      ★ 2026-10-07（ledger2-B2）：长度闸门**加例外** —— 含 LaTeX 显示数学块
        （方括号 LaTeX / 双美元 / 环境）的文本不因超长被拒，修回 in-scope 题 095
        （LZ 编码 gold 为两段方括号 LaTeX）；实测 537 条真实候选零新增误收。
        108（统计论述句，以句末句号结尾）**不在此列**。
    · `tolerant=True`：只要求"不是推理句"（`_looks_like_hard_noise`），
      连长度/句末标点也放宽 ⇒ 修复 108（095 已由默认口径的长度例外修复）。
    """
    if _POOL_TOOL_TAG_RE.search(t):
        return False
    if _POOL_SECTION_ONLY_RE.match(t):
        return False
    if _POOL_TRAIL_LABEL_RE.search(t):
        return False
    if tolerant:
        return not _looks_like_hard_noise(t)
    if len(t) > 120 and not _POOL_DISPLAY_MATH_RE.search(t):
        return False
    if _STEP_HEADER_RE.match(t):
        return False
    if _META_ANSWER_RE.search(t):
        return False
    if t[-1] in "。？！.；;":
        return False
    return True


def looks_like_pool_answer(s: str, question_type=None) -> bool:
    """候选**入池**判据 v2 + v3 自然语言容忍层。True = 像一个可入池的最终答案。

    ⚠ 与 `_looks_like_answer` 的关系：**只做增量修正**（多字母选项 / boxed 形态 /
    规划描述句 / 自然语言答案），其余仍交回 v1/v2 口径 ⇒ 是"补漏 + 修误杀"。

    ★ 2026-10-07（B）：新增可选 `question_type`。
      · 默认 `None` ⇒ 相对 v2 **仅去掉"必含数值"白名单**（白名单 → 硬黑名单），
        实测对现存 538 条候选**零新增误收**，并把 5 条 gold 中的 037/104/105 修回；
      · 传入自然语言容忍题型（`解答题/证明题/填空题/判断题/论述题/简答题`）⇒ 进一步
        放宽"句末标点 / 长度"，把 108/095 也修回（调用方按题面选择；见交付说明）。
      · 传 `选择题` / 未知题型 ⇒ 走默认口径。
    既修误杀，又把"重新放开漏网面"的风险**收敛到调用方可控的显式参数**。
    """
    t = " ".join(str(s or "").split())
    if not t:
        return False
    # ① 选项答案（修 087/088 误杀：BD / B, D / A,C…）
    if _OPTION_ANSWER_RE.match(t):
        return True
    # ② 判断题判定词（修 101/111 误杀：正确 / 错误…）
    if _VERDICT_ANSWER_RE.match(t):
        return True
    # ③ 显式 boxed ⇒ 模型自己定界了答案，不拦（防误杀）
    if re.search(r"\\boxed\s*\{", t):
        return True
    # ④ 规划/设定描述句 ⇒ 不是答案（修 000/002/007 漏网）
    if _looks_like_plan_or_setup(t):
        return False
    # ⑤ v3 自然语言容忍层（白名单 → 黑名单）
    return _looks_like_answer_v3(t, _nl_answer_tolerant(question_type))


def normalize_pool_answer(answer, question_type=None) -> tuple:
    """候选入池前的"规范化 + 校验"**纯函数内核**（入池判据的单一来源）。

    返回 `(规范化后的 answer, 是否可用)`。调用方（solver / sub_goal_solver /
    collaborative_solver）各自决定留痕与入池策略，判据本身只此一处，防口径漂移。
    `question_type` 见 `looks_like_pool_answer`（默认 None ⇒ 严口径）。

    先抢救、再判定（沿用 v1 的修正）：句子形态但含明确最终值时，**把 answer 换成
    那个值**（如 `'所以最小 d 是 48。'` → `'48'`），而不是丢候选、也不是留整句。
    """
    a = str(answer or "").strip()
    if looks_like_pool_answer(a, question_type):
        return a, True
    _sal = _salvage_value_from_line(a)
    if _sal and looks_like_pool_answer(_sal, question_type):
        return _sal, True
    return a, False


def smart_fallback_answer(text: str) -> str:
    """
    当 extract_final_answer 返回空或不理想时的智能回退。
    从文本尾部找最后一个有实质内容（数学/答案关键词）的行，
    优于盲目的 [-500:] 截取——避免长 CoT 中取到验证/总结文字而非答案。

    ★★ 2026-09-17 修复（official112-013 实况）：与 `rescue_final_answer` 同源缺陷
    ——本函数原先也不剥工具调用 XML。实测 013 的 `</tool_call>` 残留正是从这条
    兜底路径落盘成候选答案的（策略 2「第一个有实质内容的非空行」命中标签本身）。
    ⇒ 入口统一剥壳；剥完为空则直接返回空串，让调用方的兜底链继续往下走。
    """
    if not text or not text.strip():
        return ""
    # ★ 先剥工具调用残留 + calc 标记（与 extract_final_answer 口径一致）
    text = _strip_toolcall_markers(_strip_calc_markers(text)).strip()
    if not text:
        return ""
    text = _strip_continuation_markers(text).strip()
    if not text:
        return ""

    # 先试 extract_final_answer，有时它内部的多级策略能命中
    # ★★ 2026-10-07（单字符闸门残留 · A）：原判据 `len(ans) > 1` 与 `is_valid_final_answer`
    #   已修口径冲突。实测 `smart_fallback_answer("推理\n7")` 因本行把单字符 `7` 拒掉
    #   ⇒ 再往下策略 1（要求 `\d{2,}` 等）与策略 2（原要求 `len>2`）也捞不回 ⇒ 返回空。
    #   改为 `>= 1`（`ans != text.strip()` 仍保留，避免把整段原文当答案返回）。
    ans = extract_final_answer(text)
    if ans and len(ans) >= 1 and not _is_incomplete_answer(ans) and ans != text.strip():
        return ans

    lines = text.split('\n')

    # 策略 1：从后往前找第一个含数学符号/答案关键词的非元行
    # ★ 2026-10-06 收紧：**"含关键词"不够，还必须"长得像答案"**。
    #   原实现在此把 `'因此，似乎没有满足条件的路径，答案为 0。'`（含"答案"二字）
    #   与 `'让我搜索这个问题的标准解法'` 整句收下 ⇒ 污染候选池 59%。
    #   但**不能一刀切**：`'所以最小 d 是 48。'` 这类"句子形态答案"要**抽出值**。
    for i in range(len(lines) - 1, max(len(lines) - 30, -1), -1):
        line = lines[i].strip()
        if not line or _is_meta_line(line):
            continue
        if re.search(r'[$\\=]|\d{2,}|答案|故选|boxed|正确|选项', line):
            cand = clean_answer(line)
            if _looks_like_answer(cand):
                return cand
            sal = _salvage_value_from_line(cand)
            if sal:
                return sal

    # 策略 2：从后往前找第一个有实质内容的非空行
    # ★ 2026-10-06 同样收紧（原实现无条件收下任意一行）。
    # ★★ 2026-10-07（单字符闸门残留 · A）：原 `len(line) > 2` 把**单字符/双字符答案行**
    #   （选项字母 `A`、个位数 `7`、`AB`）整行跳过 —— 实测 `smart_fallback_answer("A")`
    #   与 `("3")` 均返回空串（答案被丢）。改为 `>= 1`：单字符经下方
    #   `_looks_like_answer` 闸门（仅放行数字 / 单字母 A-E / 含 LaTeX 等实质内容），
    #   中文单字仍被 `_looks_like_answer` 的"必含可判分实质"判否 ⇒ 不会放进噪声。
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line and len(line) >= 1 and not _is_meta_line(line):
            cand = clean_answer(line)[:500]
            if _looks_like_answer(cand):
                return cand
            sal = _salvage_value_from_line(cand)
            if sal:
                return sal

    # 兜底：**不再取尾部 500 字**。
    # ★ 2026-10-06 改：原实现 `return clean_answer(text[-500:])` 会把
    #   **一整段推理的尾部**当成答案返回（实测正是"推理句当答案"的最后一道口子）。
    #   现在只在**最后一行能抢救出可判分的值**时返回该值，否则返回空串 ——
    #   让调用方知道"这个候选没有可用答案"，去做更正确的处置
    #   （标记无效 / 不入池），**而不是用垃圾填满候选池**。
    tail_lines = [l for l in (text.split("\n")[-3:]) if l.strip()]
    for line in reversed(tail_lines):
        sal = _salvage_value_from_line(line)
        if sal:
            return sal
    return ""


def safe_json_serialize(obj: dict) -> dict:
    """
    安全地将字典转为 JSON 可序列化格式。
    递归处理所有值，将不可序列化的对象转为字符串。
    """
    result = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            result[key] = safe_json_serialize(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                safe_json_serialize(v) if isinstance(v, dict)
                else str(v) if not isinstance(v, (str, int, float, bool, type(None)))
                else v
                for v in value
            ]
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        else:
            result[key] = str(value)
    return result


# ============================================================
# 答案有效性校验（BUG-12 修复）
# ============================================================

_REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:无法|不能|没办法|不擅长|抱歉|对不起|sorry|cannot|unable)", re.IGNORECASE),
    re.compile(r"(?:as an AI|I cannot|I don't have)", re.IGNORECASE),
    re.compile(r"^(?:\s|[{[(（])*$"),  # 纯空白/括号
    re.compile(r"^(?:未知|unknow|n/?a|none)\s*$", re.IGNORECASE),
    re.compile(r"^(?:\[|【|\\begin).{0,20}(?:答案|最终答案).{0,20}(?:\]|】|\\end)"),
]


_INCOMPLETE_TAIL_RE = re.compile(
    r"(?:最终答案|答案|结果|故|解集|通解|特解)[：:是为\s]*$"
    r"|(?:^|[；;。！!？?，,])\s*解[：:是为\s]*$")
_LATEX_HALF_RE = re.compile(r"[\\^]\s*\{?\s*$")
_LATEX_BEGIN_RE = re.compile(r"\\begin\{[^}]+\}\s*$")


def is_truncated_answer(text: str) -> bool:
    """判断答案字符串是否明显截断/不完整（2026-09-02 老师需求）。

    观测用，不修改内容。识别 LaTeX 半截（\\xxx{、^{）、花括号不配对、
    中文前缀词结尾、占位符。识别为 True 时答案几乎必然判错。
    """
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    if _LATEX_HALF_RE.search(s) or _LATEX_BEGIN_RE.search(s):
        return True
    if _INCOMPLETE_TAIL_RE.search(s):
        return True
    # 花括号不配对（最稳的 LaTeX 截断信号）：`{` 多于 `}` ⇒ 中间有未闭合块
    if s.count("{") > s.count("}"):
        return True
    if "[子目标求解失败]" in s or "未给出有效解答" in s:
        return True
    return False


def is_valid_final_answer(text: str) -> bool:
    """
    检查 final_answer 是否为有效的数学解答（非拒绝语、非空模板、非占位符）。
    BUG-12 修复：供 formatter 最终输出前校验，不合法则回退下一候选。
    """
    if not text or not text.strip():
        return False
    s = text.strip()
    # 2026-09-20 修复：删除 `if len(s) < 2: return False` 闸门。
    # 单字符是**合法且完整**的答案（选项字母 A/B/C/D、判断题值、个位数），
    # 这与 formatter.py 已修口径一致（见该文件 run() 中"单字符是合法且完整
    # 的答案，只在空或纯空白时才回退"）。原判据使 is_valid_final_answer("A")
    # 返回 False，触发 solver.py 的 `answer = raw[-500:]` 兜底，把正确选项
    # 换成推理尾部散文 —— 客观题必然判错。
    # 真正的非答案形态（纯空白、纯括号、拒绝语）由下方 _REFUSAL_PATTERNS 负责。
    for pat in _REFUSAL_PATTERNS:
        if pat.search(s):
            return False
    return True


# ============================================================
# 答案规范化管道 (Normalization Pipeline)
# ============================================================

# 单位后缀模式（按优先级排序，长模式优先）
_UNIT_PATTERNS = [
    re.compile(r"\s*(?:厘米|cm|毫米|mm|米|m|千米|km|公里)\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:千克|kg|克|g|吨|吨|斤|磅|lb)\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:秒|s|分钟|min|小时|h|小时|天|d|年|year)\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:度|°|°C|°F|开尔文|K|弧度|rad)\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:元|美元|USD|欧元|EUR|日元|JPY|英镑|GBP)\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:人|个|次|倍|%|％|百分比)\s*$", re.IGNORECASE),
]



