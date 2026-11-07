"""
子目标求解智能体（SubGoalSolverAgent）
===================================

将数学问题分解为有序子目标，逐步求解后合并结论。

与 SolverAgent 的关系：
- SolverAgent：一次性生成多个候选解答（含蓝图分解在内的一步式求解）
- SubGoalSolverAgent：先规划子目标树，再逐步求解每个子目标，最后合并

流程：
  1. 子目标规划 (plan)  →  结构化 JSON 子目标列表
  2. 逐步求解 (step)    →  按依赖顺序依次求解每个子目标
  3. 结论合并 (merge)   →  组装最终答案并输出 Candidate
"""

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
from agent.switch_registry import (
    get_bool as _sw_bool)
# 2026-10-02 中间结果存储层（老师：「每个中间结果都要存」）：子目标结果**即时落盘**。
# 只加不改：写盘失败静默（append_ctx 永不抛异常），不影响求解与合并。
# ⚠ 2026-10-08 批1-B增：`_artifact_append` 的直接消费方已迁到
#   `agent/sub_goal_solver__loop.py`（函数内 `from .sub_goal_solver import
#   _artifact_append` 惰性引用）⇒ **本导入须保留**（单一来源，勿按"未使用"清理）。
from agent.artifact_store import append_ctx as _artifact_append  # noqa: F401  跳板 re-export（subgoal_collab/loop 惰性引用）
import logging
import re
import time

# 2026-10-07：候选入池门槛 v2 开关改为自 base 引用（去重，单一来源）。
# ⚠ 2026-10-08 批1第三增量：`_pool_gate_v2_on` 的直接消费方已迁到
#   `agent/sub_goal_solver__pool.py::apply_pool_gate`（函数内
#   `from .sub_goal_solver import _pool_gate_v2_on` 惰性引用）⇒ **本导入须保留**
#   （单一来源 = 本模块，勿按"未使用"清理 —— 与 `_artifact_append` 同口径）。
from .base import (
    BaseAgent, TaskContext, Candidate, next_candidate_id,
    _pool_gate_v2_on)  # noqa: F401  _pool_gate_v2_on 跳板 re-export（sub_goal_solver__pool 惰性引用）

# 2026-10-08 批1：run 的两段已抽到同包模块（逐字搬）。
from .sub_goal_solver__stage import wire_subgoal_stage_budget  # noqa: F401
from .sub_goal_solver__exhaust import append_exhaust_subgoal  # noqa: F401
from .sub_goal_solver__plan import (  # noqa: F401
    strip_and_mask_plan, finalize_plan_stats, build_plan_summary)
from .sub_goal_solver__loop import solve_subgoals  # noqa: F401
from .sub_goal_solver__pool import apply_pool_gate  # noqa: F401
from . import sub_goal_solver__lean as _sgs_lean  # noqa: F401
from . import sub_goal_solver__planner as _sgs_planner  # noqa: F401
from . import sub_goal_solver__parse as _sgs_parse  # noqa: F401
from . import sub_goal_solver__format as _sgs_format  # noqa: F401
from . import sub_goal_solver__merge as _sgs_merge  # noqa: F401
from . import sub_goal_solver__contract as _sgs_contract  # noqa: F401
from . import sub_goal_solver__solve as _sgs_solve  # noqa: F401

# 2026-10-08 批3：prompts.sub_goal / utils.prefill / utils.extract 的导入已随
#   全部消费方（_plan*/_solve*/_call_step/_prepare_merge_prompt/_llm_crosscheck/
#   _extract_json/_parse_subgoal_plan 等）逐字搬入 `agent/sub_goal_solver__*.py`
#   ⇒ 本模块不再直接消费，相关 try/except 导入块整体删除（各新模块自带同口径导入）。

# 2026-10-01：`<calc>` 计算工具板块（含子目标链的 <calc> 纪律/回填/裸算打回）
# 已按用户决策整体删除；纯数学求值内核迁至 `utils/math_eval.py`。

logger = logging.getLogger("MathPilot")


# ★★ 2026-09-17（P-20）：规划结论**不得当作已证实事实**注入 merge。
# 实测 10 题中 **5 题的蓝图结论值即与 gold 不符**：
#   025「故最小值为 4」(gold 506) / 013「prove minimal d=50」(gold 48) /
#   070「1/2 is achievable」(gold 5/8) / 002「T=1 is sufficient」(gold 1/2) /
#   086「次数 8」(gold 16)。
# 而 merge 提示词同时注入【蓝图规划结论】与【合并策略】(后者即 blueprint_merge)，
# 二者都内嵌该数值 ⇒ **等于指令模型输出该值**，把蓝图的错误结论锚进最终答案。
# 处理分两层：
#   ① 本函数剥掉最露骨的「答案形态」（\boxed / 「最终答案: …」）；
#   ② 模板侧声明二者均为「待验证假设」（见 prompts/sub_goal.py 的合并纪律段）。
# ⚠ 只收「**明确的形态标记**」，不收自然语句：
#   · `\boxed{…}` 与「最终答案: …」是**答案形态标记**，必剥；
#   · 「答案为 X」属**自然语句**（如规划结论「证明最终答案为 42」），
#     **不得剥** —— 首版纳入该模式导致既有测试失败（合法结论被削掉）。
_ANSWER_FORM_RES = (
    re.compile(r"\\boxed\s*\{[^{}]*\}"),
    re.compile(r"最终答案\s*[:：][^\n]*"),
    re.compile(r"the\s+answer\s+is\b[^\n]{0,80}", re.IGNORECASE),
)


def _strip_answer_forms(text: str) -> str:
    """剥掉文本中的「结论形态」片段，只留策略 / 规划描述（P-20，2026-09-17）。

    **只删最露骨的形态，不删数值本身** —— 策略里出现 `d=50` 属正常规划描述，
    靠模板侧的「待验证假设」声明约束；正则误删会丢失策略信息，故不扩大到数值。
    """
    t = str(text or "")
    if not t:
        return ""
    for pat in _ANSWER_FORM_RES:
        t = pat.sub("", t)
    t = re.sub(r"[（(]\s*[)）]", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# ★★ 2026-09-22（用户口径）：**蓝图不得锁死答案 —— 蓝图只构建数据与信息**。
#   P-20（09-17）只剥"答案形态标记"（`\boxed{}` /「最终答案:」），
#   **故意保留自然语句里的数值**。但实测 32 题自动检错显示：
#   真正把蓝图错值锚进终答的是**自然语句断言**（K11 命中 18/32 = 56%），例如
#     · "Combine n3's value with n4's proof to output single value 2024."
#     · "证明最小值为 4030，故答案为 4030"
#   —— 这些句子会被原样注入下游 merge/求解提示词，等于**指令模型输出该值**。
#
#   本函数与 P-20 的分工：
#     · P-20  `_strip_answer_forms`：删「答案形态标记」（整段形态，不可改）；
#     · 本函数：**不删句**，只把**被断言的数值**换成占位符 ——
#       句子的结构/策略信息完整保留，被剥离的只有"未经证实的那个数"。
#   为什么不像 P-20 首版那样删整句：09-17 实测删「答案为 X」会误杀合法结论
#   （既有测试失败）。只换数不删句，则策略可读性不受损。
_ASSERTED_VALUE_RES = (
    # 中文断言：最小/最大/最终/答案/结果/取值 … + 为/是/等于 + 数值
    re.compile(r"(?:最小|最大|最终|答案|结果|取值|次数|个数|总数为?)"
               r"\s*值?\s*(?:为|是|等于|即为|＝|=)\s*[-+]?\d[\d,]*(?:\.\d+)?"),
    # 英文断言 A：output(s) … value X / minimum is X / equals X
    re.compile(r"(?:outputs?|equals|value\s+is|"
               r"minimum\s+\w*\s*is|maximum\s+\w*\s*is)[^\n]{0,20}?"
               r"[-+]?\b\d[\d,]*(?:\.\d+)?",
               re.IGNORECASE),
    # 英文断言 D：N is achievable and maximal（数量词在前，实测 041 原句）
    re.compile(r"[-+]?\b\d[\d,]*(?:\.\d+)?\s+is\s+(?:achievable\s+and\s+maximal|"
               r"maximal|minimal|optimal)",
               re.IGNORECASE),
    # 英文断言 B：X of N（实测原句 "derive the minimal query count of 324"）
    re.compile(r"(?:count|value|number|answer|total)\s+of\s*[-+]?\b\d[\d,]*(?:\.\d+)?",
               re.IGNORECASE),
    # 英文断言 C：conclude(s) … N
    re.compile(r"concludes?[^\n]{0,24}?[-+]?\b\d[\d,]*(?:\.\d+)?", re.IGNORECASE),
)
_MASK_LOCKED = "[待求解值：由子目标结果现场确定]"


def _mask_asserted_values(text: str) -> tuple:
    """把策略文本里**被断言的数值**换成占位符（不删句）。返回 (新文本, 替换处数)。

    只用于 `merge_strategy` 这类**策略文本**（`problem_analysis` 已于 2026-10-14
    随题意理解收敛删除，改由 2.6 的 `ctx.problem_goal` 单一承载）；
    节点 `statement` 不在此处改（由提示词纪律约束，代码不越权改写题目陈述）。
    """
    t = str(text or "")
    if not t:
        return "", 0
    n = 0
    for pat in _ASSERTED_VALUE_RES:
        t, k = pat.subn(_MASK_LOCKED, t)
        n += k
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip(), n


def _conclusion_contract_on() -> bool:
    """子目标**结论契约**开关（`subgoal_conclusion_contract`，默认 True）。

    打开时：step 提示词追加「结论契约」块（prompts/sub_goal.py::
    SUBGOAL_CONCLUSION_CONTRACT），强制每步产出 `本步结论: 值/等式/判定`
    三形态之一 + `本步依据` 行 + 验算段；`_call_step` 解析并把结构化结论、
    推导长度、「是否只有结论」标记写进 trace，供客观判定（B）与
    主求解警示（C）消费。=0 回退旧行为（只给自由文本结论）。
    """
    try:
        return bool(_sw_bool("subgoal_conclusion_contract"))
    except Exception:  # noqa: BLE001  未注册/读取异常 ⇒ 保持修好后的行为
        return True


def _objective_judge_on() -> bool:
    """子目标结论**客观判定**开关（`subgoal_objective_judge`，默认 True）。

    打开时：每个子目标的结论都走一次 0-LLM 客观判定（SymPy 判等式真伪 /
    多候选符号多数 / 判定真假的留痕），并把 `objective_verdict` /
    `objective_reason` / `candidates_verified_index` 写进 trace。
    实测背景：`n_judged = 0/73`（一个都没判定过）——原判定**只在多 agent
    路径**且要求全部候选可解析为数学表达式，结论是自然语言时静默返回 -1。
    =0 回退旧行为（只在多 agent 路径判定、判不出不给原因）。
    """
    try:
        return bool(_sw_bool("subgoal_objective_judge"))
    except Exception:  # noqa: BLE001
        return True


def _k11_mask_on() -> bool:
    """K11 注入点掩码开关（`k11_mask_statement`，默认 True）。读取失败按 True 处理。

    ★ 用 try/except 包裹是刻意的：`switch_registry.get_bool()` 在字段未注册时
    会 `KeyError`，且**不接受默认参数**；而本开关由另一路工人并行登记
    （`user_agent.py` + `agent/switch_registry.py` + `agent/param_usage.py`）。
    本函数必须在**注册完成前后都能跑**，且默认行为一致（都 = 守卫生效）。
    """
    try:
        from .switch_registry import get_bool as _swb
        return bool(_swb("k11_mask_statement"))
    except Exception:  # noqa: BLE001  开关未注册/读取异常 ⇒ 保持守卫生效（研究期安全侧）
        return True


def _itemwise_plan_on() -> bool:
    """子目标**规划**阶段的客观题逐项策略开关（`objective_itemwise_plan`，默认 True）。

    ★ 2026-10-07（fix-tail-gaps）：原实现为裸读 `OBJECTIVE_ITEMWISE_PLAN` 并**取反**
    （`os.environ.get(..., "1") == "0"` 才回退 LLM 规划）。改走注册制后语义为
    `not _sw_bool("objective_itemwise_plan")` —— 未设/`"1"` ⇒ 启用逐项规划（同旧默认），
    `"0"` ⇒ 关（同旧行为）。三态与原 `== "0"` 逐条一致；仅 `""`/`"false"` 等非标准值
    由 registry 统一口径判为「关」，属已文档化的修正。=0 一键回退到 LLM 规划。
    """
    try:
        return bool(_sw_bool("objective_itemwise_plan"))
    except Exception:  # noqa: BLE001  未注册/读取异常 ⇒ 保持修好后的行为（默认启用）
        return True


def _check_protocol_guard_on() -> bool:
    """`<check>` **协议守卫**开关（`check_protocol_guard`，默认 True）。

    针对问题 2.9：实测题 000 大量子目标把**中文句子**塞进 `<check>`（如
    `<check>图 G 是弱连通的</check>`）⇒ 客观判定**一条都出不来**
    （`is_parseable=False`，`n_judged = 0/73` 的直接原因之一）。打开后：
      ① 判定"该 `<check>` 内容是否为**可核验对象**"（纯数学等式）；
      ② 违反 ⇒ **留痕**（trace + sg 字段）并**触发一次定向重问**；
      ③ 重问仍不合规 ⇒ 保留并留痕（绝不静默落 undecided）。
    =0 回退旧行为（只静默跳过、不留痕、不重问）做 A/B。
    """
    try:
        return bool(_sw_bool("check_protocol_guard"))
    except Exception:  # noqa: BLE001  未注册/读取异常 ⇒ 保持修好后的行为
        return True


# ---------------------------------------------------------------------------
# `<check>` 协议校验（2026-10-07，问题 2.9）
# ---------------------------------------------------------------------------
# 协议（`prompts/sub_goal.py::SUBGOAL_STEP_SYSTEM` 第 6 条）要求 `<check>` **只装
# 一个纯数学等式**：仅含数字/变量/运算符（`+ - * / ^ ( )`），恰含一个 `=`，无中文。
# 违反协议（装中文句子/文字/非等式）时**不是"没给出断言"，而是"给了不可核验的东西"**
# ⇒ 必须与"未给断言"区分：前者留痕 + 定向重问，后者正常走 undecided。
_CHECK_INNER_RE = re.compile(r"<check>(.*?)</check>", re.DOTALL)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# ⚠ 2026-10-07 实测修正：初版字符类漏了 `|` ⇒ 把 `|D_8| = 8`（二面体群阶，实测出现
#   8 次）误判为"违反协议"（假阳）。这里补上数学常用符号（`|` 绝对值/阶、`,` 元组、
#   `{}[]` 集合/区间、`!` 阶乘、`'` 导数、`%`、`~`、`\` LaTeX 命令、`<>`）。仍要求
#   "恰含一个 `=`"且不含中文 ⇒ 只拦真正的破坏形态（`<check>图 G 是弱连通的</check>`）。
_CHECK_BODY_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_+\-*/^().=\s|,{}\[\]<>!%'~\\]+$")


def _is_verifiable_check_body(body: str) -> bool:
    """`<check>` 的 inner 是否是**可核验对象**（一个纯数学等式）。

    ⚠ 本函数只判**语法形态**（"是不是一个纯数学等式"），**不判语义可验性**：
    像 `x = 5` / `s = 265/247` 这类"赋值式"语法上合规（协议第 6 条虽提示不要用，
    但实测模型大量使用），语义"不可验"由 `_sympy_judge` 返回 unknown 放行处理
    —— 若在此也判违规，会对大量正常步骤触发无谓重问（成本风险 > 收益）。
    故本守卫专注拦截**协议真正的破坏形态**：`<check>` 里装中文/文字/非等式
    （实测 `'图 G 是弱连通的'` 就是这一类）。
    """
    b = " ".join(str(body or "").split())
    if not b:
        return False
    if _CJK_RE.search(b):          # 含中文 ⇒ 句子，不是可核验等式
        return False
    if b.count("=") != 1:          # 必须恰含一个 =
        return False
    L, R = (p.strip() for p in b.split("="))
    if not L or not R:             # 两侧都要有内容
        return False
    if not _CHECK_BODY_ALLOWED_RE.match(b):   # 只允许数学字符
        return False
    return True


def _check_protocol_violations(text) -> list:
    """返回**违反 `<check>` 协议**的 inner 文本列表（空 = 无 `<check>` 或全部合规）。

    与 `_extract_check_asserts` 的区别：后者"宁缺毋滥"——不合规的 `<check>`
    **静默 skip**（于是违反协议不可见）；本函数把**同一批 skip 掉的**显式收出来，
    供留痕与定向重问。
    """
    if not text or "<check>" not in str(text):
        return []
    out = []
    for m in _CHECK_INNER_RE.finditer(str(text)):
        body = m.group(1).strip()
        if not _is_verifiable_check_body(body):
            out.append(body or "(空)")
    # 去重保序
    seen, uniq = set(), []
    for b in out:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq[:6]


# ★ 2026-09-17（P-9）：穷尽性检查的**固定段落**清单。
# ⚠ 与 `orchestrator._parse_exhaust_result` 的口径必须一致 ——
#   由 tests/test_exhaust_sections_consistency.py 断言，防止两处漂移。
_EXHAUST_SECTIONS = ("解族清单", "逐族判定", "前序结论复核")
_EXHAUST_VERDICT_RE = re.compile(r"EXHAUSTIVE\s*[:：]\s*(yes|no)", re.IGNORECASE)


def _exhaust_missing_sections(text: str) -> list:
    """返回「解族穷尽性检查」输出中缺失的段落名（空列表 = 齐备）。"""
    t = str(text or "")
    miss = [n for n in _EXHAUST_SECTIONS if n not in t]
    if not _EXHAUST_VERDICT_RE.search(t):
        miss.append("EXHAUSTIVE")
    return miss

# 单选信号（2026-09-13）：「单选 / 多选」口径。
# ★ 2026-10-14（任务列表 #3）：判定本身**上移到 2.6 题意理解** ——
#   正则与判定函数现居 `agent/problem_goal.py`（`SINGLE_CHOICE_RE` /
#   `judge_n_correct()`），结论落在 `ctx.problem_goal["n_correct"]`，
#   2.7 只读结论 ⇒ 此处**不再保留副本**（原 `_SINGLE_CHOICE_RE` 已删）。
# ⚠ 判定措辞的实测教训（随正则一并迁移）：不可用"下列正确的是"判断单选 ——
#   087/093/094 都是该措辞却为**多选**，096 也是该措辞却为**单选**；
#   只有"哪一项 / 最合适 / 定义是"这类措辞才可靠。



class SubGoalSolverAgent(BaseAgent):
    """子目标逐步求解智能体"""

    name = "SubGoalSolver"

    # ---------- JSON 提取 ----------
    _extract_json = staticmethod(_sgs_parse._extract_json)

    # ---------- P2 子目标类型路由（2026-09-09 老师：计算/推理类型化）----------
    # calc_kind: "terminal"=整体计算型子目标（期望输出=单个数值/表达式 → 专用
    # 模板只翻译表达式，工具回填即结论，模型零计算面）；"inline"=推理/证明型
    # （步骤级 <calc> 截断）。规则启发先行，宁 inline 勿误 terminal（防把推理
    # 子目标截成表达式翻译）。字段总是打标进 trace（0 成本可观测分布）；
    # ★ 2026-10-04 修注释（D32）：原文写「专用模板/校验由 config.subgoal_calc_router
    #   控制（默认关待 A/B）」—— 该配置项**已随 `<calc>` 计算工具板块一并删除**
    #   （见 user_agent.py 的删除留痕），`calc_kind` 现为**纯观测字段**（只进 trace/diag，
    #   不驱动任何分支）。原注释会误导后人以为它仍在生效。
    _TERM_ACTION_KW = ("代入", "求和", "化简", "展开", "求值", "积分",
                       "乘积", "计算", "算得")

    _calc_kind_of = staticmethod(_sgs_parse._calc_kind_of)

    _parse_subgoal_plan = staticmethod(_sgs_parse._parse_subgoal_plan)

    # ---------- 主流程 ----------
    def run(self, ctx: TaskContext) -> TaskContext:
        """执行子目标规划 → 逐步求解 → 结论合并 全流程，结果追加到 ctx.candidates"""
        # 预算闸门：连规划所需的 1 次 LLM 调用都负担不起时，整体跳过、不追加候选
        # 2026-09-06：升级 gen_time_up（生成侧软截止，未设时回退 is_time_critical）
        if ctx.gen_time_up():
            self.record(ctx, "subgoal", "预算耗尽，跳过子目标求解")
            return ctx

        # 2026-09-04 子目标阶段预算（装配细节、依据与取舍见
        # agent/sub_goal_solver__stage.py::wire_subgoal_stage_budget；
        # 2026-10-08 批1：逐字搬，调用点改一行委托）。
        wire_subgoal_stage_budget(self, ctx)

        def _stage_left() -> float:
            """子目标阶段剩余秒数（0 = 未启用阶段预算，返回 inf）"""
            if not ctx._subgoal_stage_end:
                return float("inf")
            return ctx._subgoal_stage_end - time.time()

        # ★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
        #   检索在**本阶段体首**发起 —— 早于 `_plan_subgoals`（蓝图构建在其内），
        #   也早于逐步求解与合并。这样定理线索对蓝图（3 处读
        #   `theorem_hint_block`）、子目标规划、merge/solve、3_solve（3 处）
        #   全部可见；同时顺带修掉 #1 遗留的顺序代价（此前检索跑在 2.6 判题
        #   之前，读到的题型是规则判定）。
        #   ⚠ 增益型动作：`run_hint_for` 内部全程 try，失败绝不阻断主链。
        #   ⚠ 保留原 `theorem_hint_stage` 的**两道门控**，搬移不得改变语义：
        #       ① master 开关 `enable_theorem_hint`（默认 True）——任务 #10 的
        #          A/B 把手（`--enable_theorem_hint 0/1`），漏搬即 A/B 失效；
        #       ② 紧急态守卫 `ctx.state.emergency` 时不检索。
        try:
            from .theorem_hint import run_hint_for
            if (getattr(self.config, 'enable_theorem_hint', True)
                    and not ctx.state.emergency):
                run_hint_for(self, ctx)
        except Exception as _e_th:  # noqa: BLE001
            self.record(ctx, "theorem_hint",
                        "定理检索启动失败（跳过，不阻断）: %s: %s"
                        % (type(_e_th).__name__, str(_e_th)[:100]))

        # 阶段一：子目标规划
        plan_data = self._plan_subgoals(ctx)
        if plan_data is None:
            self.record(ctx, "subgoal", "子目标规划失败，回退到标准求解")
            return ctx

        subgoals = plan_data.get("subgoals", [])
        # ★ 2026-09-17（Z8）：规划结果为空 ⇒ 后续的逐步求解与 merge 都无从下手，
        #   原实现仍会跑一遍 merge/建候选（纯空转）。此处直接早退。
        if not subgoals:
            self.record(ctx, "subgoal", "子目标规划为空 → 跳过求解与合并（Z8 2026-09-17）")
            return ctx
        # ★ 2026-10-14（任务列表 #3）：第二项不再是「蓝图自己产的问题分析」，而是
        #   2.6 的**唯一题目目标** `ctx.problem_goal`（`strip_and_mask_plan` 只负责
        #   merge_strategy 的形态剥离与断言值掩码）。
        merge_strategy, problem_goal = strip_and_mask_plan(self, ctx, plan_data)
        # ---- 2026-09-14 穷尽性搜索机制（B0 遗留课题）----
        # 2026-10-08 批1：本段逐字搬到 agent/sub_goal_solver__exhaust.py::
        # append_exhaust_subgoal，调用点改一行委托（merge_strategy 由之返回）。
        merge_strategy = append_exhaust_subgoal(self, ctx, subgoals, merge_strategy)

        finalize_plan_stats(self, ctx, subgoals, merge_strategy)

        # 阶段二：逐步求解每个子目标
        subgoal_plan_summary = build_plan_summary(self, ctx, subgoals, merge_strategy)
        results_map = {}  # subgoal_id → result_text
        # ★ 2026-10-15（用户定案 · 三角色协作）：开关 subgoal_collab_mode 开启时，
        #   逐步求解走 agent/subgoal_collab.py（求解/审核/规划三角色合作 +
        #   3 遍「在现有基础上再生成」+ 规划预取 + 蓝图回传）；默认 Off（可 A/B）。
        _collab_on = False
        try:
            from .subgoal_collab import _enabled as _collab_enabled
            _collab_on = _collab_enabled(self)
        except Exception:  # noqa: BLE001
            _collab_on = False
        if _collab_on:
            from .subgoal_collab import solve_subgoals_collab as _collab_solve
            self.record(ctx, "subgoal",
                        "逐步求解走三角色协作（subgoal_collab_mode=1）")
            _ctx_inject_chars = _collab_solve(
                self, ctx, subgoals, subgoal_plan_summary, results_map)
        else:
            _ctx_inject_chars = solve_subgoals(
                self, ctx, subgoals, subgoal_plan_summary, results_map)

        # 阶段三：结论合并
        # 2026-09-04 保护：预算已尽且一个子目标都没解出（典型：或chestrator 后续
        # 调用 run() 时预算早已耗尽）→ 空 merge 只产垃圾，直接 return 不追加候选。
        if not results_map and _stage_left() <= 0:
            self.record(ctx, "subgoal",
                        "子目标阶段预算已尽且无已解子目标，跳过空 merge")
            return ctx
        if ctx.gen_time_up():
            # 生成侧软截止/全局逼近 deadline（2026-09-06 升级 gen_time_up；
            # 正常 750s stage_budget 结束于 ~760s < gen_deadline，merge 仍真跑，
            # 只有跑超到 gen_deadline 才兜底）→ 连 1 次 merge LLM 调用都挤不出
            # 时，用最后一个子目标结果兜底（不空手返回）。
            self.record(ctx, "subgoal", "全局预算不足，跳过合并阶段")
            # 使用最后一个子目标的结果作为最终答案
            final_answer = self._fallback_from_last_subgoal(subgoals, results_map)
        else:
            # 2026-09-04：即使子目标阶段预算已用尽（_stage_left()<=0），
            # 只要全局未 critical 就**必须真 merge**——这是"强制收尾产出"的关键：
            # 此前阶段预算用尽直接 return / 或 merge 被全局 critical 挤掉，
            # 答案停在子目标枚举（060: S(1)..S(12)）而没合成最终结论。
            final_answer = self._merge_results(ctx, subgoals, subgoal_plan_summary,
                                               results_map, merge_strategy)

        # v2.9：结构化输出最终整合方案
        ctx.subgoal_merge_plan = (
            f"合并策略: {merge_strategy or '将各子目标结果按逻辑顺序组合'}\n"
            f"最终答案: {final_answer}"
        )

        final_answer, _skip_pool = apply_pool_gate(self, ctx, final_answer)
        if _skip_pool:
            return
        # 构造 Candidate（`full_reasoning` 在门槛**之后**构建 ⇒ 与最终入池的
        #   `final_answer` 保持一致；避免抢救替换值后 reasoning 仍写原句）。
        full_reasoning = self._build_full_reasoning(
            subgoals, problem_goal, final_answer, results_map)
        candidate = Candidate(
            id=next_candidate_id(ctx.candidates),
            answer=final_answer,
            reasoning=full_reasoning,
            revised=False,
            # 2026-09-14：选择题走「逐项判定」链路 ⇒ 标记来源，供选取阶段优先采纳。
            # 依据：实测 #107 逐项判定正确得出 D，却被候选池里 5 个整体求解的 C
            # 以 5:1 多数投票淹没。逐项判定是**有依据**的结论，不应输给数量。
            # 置 OBJECTIVE_ITEMWISE_PRIORITY=0 可回退（项目惯例：每项优化一开关）。
            origin=(
                "itemwise"
                if (getattr(ctx, "question_type", "") == "选择题"
                    and _sw_bool("objective_itemwise_priority"))
                else ""
            ),
        )
        ctx.candidates.append(candidate)
        self.record(ctx, "subgoal", "子目标求解完成，已生成候选解答",
                    subgoal_stats=self._subgoal_stats(subgoals),
                    ctx_inject_chars=_ctx_inject_chars)
        # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：
        #   原 `#34 阶段四：DAG 评审 + 整树重生成`（`enable_dag_replan`，默认 False）
        #   已**整块撤除** —— 机制从未在生产启用（默认关、仅 arm2 复测臂引用），
        #   连同 `_review_and_maybe_replan` / `_review_dag_prepare` / `_note_dag_replan`
        #   与其宿主模块 `sub_goal_solver__review.py` 一并删除。
        return ctx

    _is_value_choice = staticmethod(_sgs_planner._is_value_choice)

    _plan_value_choice = _sgs_planner._plan_value_choice

    _plan_choice_by_options = _sgs_planner._plan_choice_by_options

    # ---------- 阶段一：规划 ----------
    _subgoal_cap = _sgs_planner._subgoal_cap

    _plan_subgoals = _sgs_planner._plan_subgoals

    # ---------- Blueprint DAG 规划（#27）----------
    _plan_from_blueprint = _sgs_planner._plan_from_blueprint

    _inject_mathlib_theorems = _sgs_planner._inject_mathlib_theorems

    _inject_anticipatory_lemmas = _sgs_planner._inject_anticipatory_lemmas

    # ★ 2026-10-15：`_skeleton_review_loop` / `_dag_replan_gate(_prepare)` /
    #   `_lean_dag_logic_check` / `_parse_lean_expr_map` / `_map_lean_errors` /
    #   `_merge_lean_rejects` 跳板随三个默认关闭的门一并撤除
    #   （`sub_goal_solver__review.py` 已整文件删除）。

    # ---------- 阶段二：逐步求解 ----------
    _solve_subgoal = _sgs_solve._solve_subgoal
    # ★ 2026-10-13：本类用**显式装配清单**把 `sub_goal_solver__*` 的模块级函数挂到类上
    #   ⇒ **新增函数必须在此登记**，否则类上不存在（实测：漏登记导致
    #     `AttributeError: 'SubGoalSolverAgent' object has no attribute
    #      '_improve_subgoal_result'`，被集成测试当场抓到）。
    _improve_subgoal_result = _sgs_solve._improve_subgoal_result

    _prepare_step_prompt = _sgs_solve._prepare_step_prompt

    _use_lemma = _sgs_solve._use_lemma

    _accumulate_lemma = _sgs_solve._accumulate_lemma

    _strip_tool_residue = staticmethod(_sgs_solve._strip_tool_residue)

    _call_step = _sgs_solve._call_step

    # ------------------------------------------------------------------
    # 多 agent 子目标求解（2026-09-30，截图 #8）
    # ------------------------------------------------------------------
    _multi_agent_n = _sgs_solve._multi_agent_n

    _sample_step_consensus = _sgs_solve._sample_step_consensus

    _pick_consensus_sample = staticmethod(_sgs_solve._pick_consensus_sample)

    # ==================================================================
    # 子目标**结论契约**：解析 / 客观判定 / 落 trace（2026-10-06 重写 A/B/C）
    # ==================================================================
    # 契约（见 prompts/sub_goal.py::SUBGOAL_CONCLUSION_CONTRACT）要求【本步结果】
    # 段内头两行固定为：
    #   本步结论: 值 = <expr> | 等式 = <L> = <R> | 判定 = 真|假|不确定
    #   本步依据: <前序编号 / 题设 / 本步推导>
    # 之后 ≤300 字为关键验算（= "有推导" 的证据）。
    _CONCL_LINE_RE = re.compile(
        r"本步结论\s*[:：]\s*(值|等式|判定)\s*=\s*(.+)")
    _SUPPORT_LINE_RE = re.compile(r"本步依据\s*[:：]\s*(.+)")
    # 「有推导」的最小验算篇幅（字符）：低于此视为"只给结论、未给推导"
    _MIN_DERIVATION_CHARS = 20

    _parse_step_contract = classmethod(_sgs_contract._parse_step_contract)

    _extract_derivation_block = staticmethod(_sgs_contract._extract_derivation_block)

    _tail_after_conclusion = classmethod(_sgs_contract._tail_after_conclusion)

    _judge_subgoal_objective = _sgs_contract._judge_subgoal_objective

    _apply_conclusion_contract = _sgs_contract._apply_conclusion_contract

    # 决策缓存需一起跨 run() 复用的契约字段（见 `_build_trace_item` 同名键）。
    _CONTRACT_KEYS = (
        "conclusion", "conclusion_kind", "conclusion_support",
        "conclusion_contract_ok", "has_derivation", "derivation_chars",
        "objective_verdict", "objective_reason",
        "candidates", "candidates_picked_index", "candidate_answers",
        "candidates_verified_index",
    )

    _contract_snapshot = classmethod(_sgs_contract._contract_snapshot)

    _restore_contract = classmethod(_sgs_contract._restore_contract)

    _answer_from_raw = _sgs_contract._answer_from_raw

    _extract_step_result = staticmethod(_sgs_contract._extract_step_result)

    _verify_candidates_objective = staticmethod(_sgs_contract._verify_candidates_objective)

    _oracle_check_step = staticmethod(_sgs_contract._oracle_check_step)

    # ------------------------------------------------------------------
    # S1-lite：子目标级 0-LLM 校验前移（2026-09-06 老师建议）
    # ------------------------------------------------------------------
    # 老师：lean 校验服务很多要排优先级；子目标宜独立、上下文少 → 校验可前移。
    # 落地形态（全部 0-LLM、秒级~21s、异常/环境缺失一律放行，绝不阻断）：
    #   L0  截断/空结果检查（全档，0 成本）——治"答案腰斩"型错误（L2 截断修复下沉）；
    #   L0P 占位/空转检测（全档，0 成本）——2026-09-07 禁网冒烟实证：PB-002 SG#2
    #       ='（无）'、SG#4='（子目标4 已求解）'、alg-060 SG#1='[计算] 1 = 1'，
    #       LLM 空转输出既非空也非截断、不带 lean 代码 → 旧 L0/L1 全放行流入 merge。
    #       短结果里出现占位词 / 重言式回填 → 视为未真正求解，带反馈重解。
    #   L1 Lean 代码片编译校验（仅 deep 档 + lean 环境可用，0 LLM 5-21s）——
    #     子目标结果若自带 lean 代码（```lean ... ``` 或 import 块），本地编译
    #     抓语法/类型错，禁网下纯本地（lean-lsp-mcp 禁网已验证 0 出网）。
    #     注：2026-09-07 冒烟实测子目标结果多为短结论行、极少带 lean fence，
    #     L1 触发面天然小；价值主要在 L0P 的占位拦截，L1 仅作兜底保留。
    # 不做：LLM 翻译整段子目标结果去 lean 化（翻译成本与整题 6.5 校验重复，列赛后）。

    _LEAN_FENCE = re.compile(
        r"```(?:lean4?|lean)?\s*\n(.*?)```", re.DOTALL)

    # L0P：短结果占位词（≤50 字符内命中才算，长结果里的行文不判）。
    # 只收高置信空转形态（宁漏勿误伤：如"方程无解""待定系数"是真结论）。
    # 2026-09-08：`[计算] X = X` 型**平凡 calc 回填**单列判据（见下）——
    # 旧注释担心 `[计算] 191/192 = 191/192` 是真分数验证无法与空转区分；
    # 但回填格式是 `[计算] expr = result`，**result 与 expr 逐字相同**意味着
    # 模型只是把结论原样丢给 <calc>（<calc>191/192</calc> 无任何运算发生），
    # 属 alg-060 SG#1='[计算] 1 = 1' 同型空转，可可靠判定。真运算（如
    # <calc>comb(50,3)</calc> → [计算] comb(50,3) = 19600）result≠expr 不受影响。
    _PLACEHOLDER_RE = re.compile(
        r"（无）|（略）|暂无|占位|空转|未求解|未给出"
        r"|子目标\s*\d+\s*已(?:求解|完成|给出)"
        r"|^同上$|^见上$"
    )
    # `[计算] X = X` 切分（左右各自归一后比较，见 _looks_placeholder）。
    # 2026-09-08 二次升级：SymPy 符号化简回填带空格（`1/(n + 1)`）或幂记号
    # 差异，旧 `\1` 逐字反引用会漏判"符号复读空转"（模型把 1/(n+1) 原样丢给
    # <calc> 当自证）；改为切出左右两侧后**去空白归一比较**——真运算
    # （comb(50,3) → 19600、integral 结果、sqrt(45) → 3*sqrt(5)）两侧不同，
    # 不受影响；无运算的复读（1/(n+1)=1/(n+1)、0=0）可靠拦截。
    _TRIVIAL_CALC_RE = re.compile(
        r"^\[计算\]\s*(?P<left>.+?)\s*=\s*(?P<right>.+?)\s*$", re.DOTALL)

    _looks_placeholder = classmethod(_sgs_contract._looks_placeholder)

    _extract_lean_code = classmethod(_sgs_contract._extract_lean_code)

    _subgoal_light_check = _sgs_contract._subgoal_light_check

    # ---------- L2 数值/代数断言 Lean 验证（2026-09-08）----------
    _L2_EXPR_CHARS = re.compile(r"^[0-9a-zA-Z^*/+\-() ]+$")

    _extract_lean_assert_pairs = _sgs_lean._extract_lean_assert_pairs

    _fix_lean_mul = staticmethod(_sgs_lean._fix_lean_mul)

    _numeric_lean_verify = _sgs_lean._numeric_lean_verify

    # ---------- L3 <check> 协议断言验证（2026-09-08 协议版）----------
    # 模型在【本步结果】用 <check>LHS = RHS</check> 显式声明待验证等式 →
    # 提取零歧义（解决提取式 2.8% 覆盖率问题）。判定：sympy 精确判定为主
    # （expand/simplify，确定性无浮点误差）→ 判"不成立"的再交 Lean
    # norm_num/ring 编译背书（mcp 后端）双确认 → 反馈带标准化简结果
    # （可行动），杜绝"只知道错、不知道对什么"的裸诊断。
    _CHECK_RE = re.compile(r"<check>(.*?)</check>", re.DOTALL)

    _extract_check_asserts = _sgs_lean._extract_check_asserts

    _strip_check_tags = staticmethod(_sgs_contract._strip_check_tags)

    _pow_to_python = staticmethod(_sgs_contract._pow_to_python)

    _sympy_judge = _sgs_lean._sympy_judge

    _check_assert_verify = _sgs_lean._check_assert_verify

    # P4（2026-09-09 老师"描述即完成/零代码"）：疑似错断言 → 生成**多条策略
    # 尝试**声明（norm_num/norm_num1/ring/ring_nf/nlinarith/omega/grind/aesop
    # 按形态选择），一次编译，组内任一策略闭合 → 剔除（防 sympy 误报）；
    # 组内全部失败 → 双确认该断言不成立。
    _SYNTAX_ERR_KW = ("unknown identifier", "unexpected token",
                      "unknown constant", "invalid")

    _build_auto_proof = _sgs_lean._build_auto_proof

    _lean_backup_fails = _sgs_lean._lean_backup_fails

    # ---------- ★ 2026-10-15 蓝图 tool 指派的兑现层 ----------
    # 蓝图在节点上声明 `tool`（lean/sympy/numeric/llm），这里把它接成实例方法，
    # 供 `sub_goal_solver__loop.sg_tool_exec` 调用（`self._tool_declared_verify`）。
    _tool_declared_verify = _sgs_lean._tool_declared_verify
    _sympy_verify_pairs = _sgs_lean._sympy_verify_pairs
    _lean_block_verify = _sgs_lean._lean_block_verify

    # ---------- ③ 子目标交叉核对（2026-09-08 老师建议3）----------
    # 老师建议："子目标交叉核对机制"——不同子目标对同一量给出互相矛盾的结论
    # 时，系统要能发现并打回，而不是让矛盾静默流入最终合并（如 alg-060 型）。
    # 落地两方向（独立开关，可单开/同开做 A/B）：
    #   A 确定性冲突闸（0 LLM，subgoal_conflict_gate）：
    #     汇总全题子目标结果里的 <calc> 回填（[计算] X = v）+ <check> 协议断言
    #     + L2 已提取等式断言，同名单变量 LHS 出现**可判定不同的常数**值
    #     （x=1 vs x=2）→ 判显式矛盾，merge 提示词强制裁决后才许合并。
    #     已知盲区：隐含矛盾（alg-060：xy=25 与 D=2 不共用同一变量名，纯规则
    #     必漏）；非单变量 LHS（x^2=4 含 ± 分支）、近似舍入差异 → 宁漏勿误报。
    #   B LLM 交叉核对轮（merge 前 +1 次小调用，subgoal_crosscheck_llm）：
    #     把各子目标结论行单独抽出交给 LLM 集中裁决矛盾——能看隐含矛盾
    #     （alg-060 这型有机会看出，不保证）。预算不足自动跳过，不阻断 merge。
    _LHS_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    # 行内 [计算] 回填（允许前缀自然语言，如 "解得 [计算] x = 1"）：
    # RHS 截断到行尾或中英文句读/分号，防吞掉后续文字。
    _CALC_INLINE_RE = re.compile(
        r"\[计算\]\s*([^=\n，。；;,]+?)\s*=\s*([^\n，。；;,]+)")
    # 松散 ident = const 片段（纯文本结论 "解得 x = 1"）：前 2 个非空字符
    # 命中情形/条件标记（当/若/情形/情况/分支…）→ 跳过，防分支讨论误报。
    _LOOSE_EQ_RE = re.compile(
        r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s，。；;,+)(]+)")
    _CASE_MARK = set("当若情形况分")

    _lookback_case_mark = staticmethod(_sgs_merge._lookback_case_mark)

    _collect_claims = _sgs_merge._collect_claims

    _rhs_const = staticmethod(_sgs_merge._rhs_const)

    _cross_conflicts = _sgs_merge._cross_conflicts

    _llm_crosscheck = _sgs_merge._llm_crosscheck


    # ---------- 阶段三：合并 ----------
    _merge_results = _sgs_merge._merge_results

    _prepare_merge_prompt = _sgs_merge._prepare_merge_prompt

    _blueprint_conclusion = staticmethod(_sgs_format._blueprint_conclusion)

    # ---------- 辅助方法 ----------
    _build_trace_item = staticmethod(_sgs_format._build_trace_item)

    _subgoal_stats = staticmethod(_sgs_format._subgoal_stats)

    _format_plan_summary = staticmethod(_sgs_format._format_plan_summary)

    _format_previous_results = staticmethod(_sgs_format._format_previous_results)

    _format_dep_results = staticmethod(_sgs_format._format_dep_results)

    # 选项判定结论解析（2026-09-14 实测 #103/#107/#110 暴露的问题）。
    # 模型**经常不遵守**"开头写『结论：正确/错误』"的格式，实际输出的形态包括：
    #   「结论：正确 依据：…」/「选项 D：正确」/「X = Y」（等价判断，暗示正确）/
    #   直接复述基准（**无任何结论**）。
    # 聚合层若不能识别，就会**丢项**（#103 丢 E）或**误采纳**（#107 判 D 对却输出 C）。
    _VERDICT_PATTERNS = (
        (re.compile(r"结论\s*[:：]\s*(正确|成立|符合|是正确|对的)"), "正确"),
        (re.compile(r"结论\s*[:：]\s*(错误|不成立|不符合|不正确|是错)"), "错误"),
        (re.compile(r"选项\s*[A-E]\s*[:：]\s*(正确|成立|符合)"), "正确"),
        (re.compile(r"选项\s*[A-E]\s*[:：]\s*(错误|不成立|不符合)"), "错误"),
        (re.compile(r"^\s*(正确|成立|符合)\s*$", re.M), "正确"),
        (re.compile(r"^\s*(错误|不成立|不符合)\s*$", re.M), "错误"),
    )

    _extract_option_verdict = classmethod(_sgs_format._extract_option_verdict)

    _format_all_results = staticmethod(_sgs_format._format_all_results)

    _fallback_from_last_subgoal = staticmethod(_sgs_format._fallback_from_last_subgoal)

    _build_full_reasoning = staticmethod(_sgs_format._build_full_reasoning)
