# -*- coding: utf-8 -*-
"""纯数据结构（Candidate / Verdict / RunState / RoundState / TaskContext）
—— 从 agent/base.py 逐字搬出（2026-10-08 拆分）。

对外契约不变（同一对象）：``from agent.base import TaskContext, Candidate, Verdict,
RunState, RoundState`` 仍可用（base.py 原位 re-export）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from contracts.types import Budget  # noqa: F401  TaskContext.budget 注解用

# ============================================================
# 数据结构
# ============================================================
@dataclass
class Candidate:
    """一个候选解答"""
    id: int
    answer: str                 # 提取出的简洁最终答案
    reasoning: str              # 完整推理过程（含文本）
    revised: bool = False       # 是否由自纠错回环产生
    # 2026-09-14：来源标记。选择题的「逐项判定」候选标 `itemwise` ——
    # 它是**逐个选项验证过**的结论，可靠性高于"整体求解"的候选；
    # 实测 #107 逐项判定得出 D（正确），却因候选池里 5 个 C 而被多数
    # 投票淹没 ⇒ 选择题选取时应对 itemwise 候选优先。
    origin: str = ""

    def __post_init__(self):
        # 2026-09-17：候选答案统一出口剥壳。
        # 实测 official112-025：6 个候选里有 **5 个** answer 字段是字面量
        # `</tool_call>`（而 reasoning 是完整的 1000+ 字符推理）；032 的
        # `从 $N$ 倒推：若从位置 $k`、013 的英文搜索词等，也都是"工具标签 /
        # 推理碎片被当成最终答案"。根因是多条生成路径漏剥壳、项目内无统一收敛点。
        # 在本数据类的构造出口集中剥一次，一处即覆盖 20+ 个 Candidate 写入点。
        # 惰性 import 避免与 utils.extract 的循环依赖；剥壳失败绝不阻断构造。
        if isinstance(self.answer, str) and self.answer:
            try:
                from utils.extract import (
                    _strip_calc_markers,
                    _strip_toolcall_markers,
                )
                self.answer = _strip_toolcall_markers(
                    _strip_calc_markers(self.answer)).strip()
            except Exception:  # noqa: BLE001  剥壳失败不阻断构造
                pass





@dataclass
class Verdict:
    """一个候选解答的验证结果"""
    id: int = 0
    answer: str = ""
    reasoning: str = ""
    confidence: float = 0.0     # 置信度 = 正确票数 / 总票数
    correct_votes: int = 0
    total_votes: int = 0
    feedback: str = ""          # 失败时由验证器提取的错误原因
    correct: bool = False       # 本次投票结果是否正确（verifier 使用）
    raw: str = ""               # 原始投票返回文本（verifier 使用）
    score: dict | None = None   # 评分模式的详细分数（verifier 使用）
    deterministic: dict | None = None  # 确定性验证旁证/否决证据（verifier 使用，0 LLM 预算）
    abstain: bool = False       # 弃权票（verifier 使用）：LLM 调用失败/输出无法解析
                                # → 属"基础设施/格式故障"，不是"答案错误"。
                                # 聚合时剔出 total_votes 分母，避免把故障当反证。
    # ★ 2026-09-15 新增（用户要求"测试时把错误暴露得更具体"）：
    # 从 `VERDICT: B（错误类型）` 里解析出的**结构化错误类型**（逗号分隔的标签，
    # 如 "方法不适用, 前提不成立"）。此前只保留 verdict（对/错），错误类型
    # 被丢弃 ⇒ 事后归因只能靠人工重读全文。现按标签落盘，便于逐题统计。
    error_type: str = ""


# ★ 2026-10-07：`Budget`（纯数据）已下沉到 `contracts/types.py`，顶部已 re-export。


# ------------------------------------------------------------
# P3：Finding / BugReport（step 级分级验证报告）
# ★ 2026-10-07：Budget / Finding / BugReport 三个**纯数据类**已下沉到
#   `contracts/types.py`（零依赖），文件顶部已 re-export ⇒ 对外契约不变。
#   原因：它们被 `tools/lean_local/{lean_bridge,lean_gate}.py` 反向依赖，
#   是 `agent ↔ tools` 包级环的一环；下沉后 tools/ 只依赖 contracts/。
# ------------------------------------------------------------


# ------------------------------------------------------------
# B6 运行时覆盖态 + P4 AcceptGate 门控（v2.8，自 sq 增量移植）
# ------------------------------------------------------------
@dataclass
class RunState:
    """运行期可变状态（承载运行时覆盖值，不冻结 config）。

    每题为独立 TaskContext，天然并发隔离；orchestrator 不再改写共享 config，
    消除并发=3 时跨题污染（时间预算自律核心）。
    """
    emergency: bool = False              # 应急模式：候选/投票降至 1，跳过续写/复算
    playoff_enabled: bool = False        # 是否启用 playoff 确定性复算
    voting_times: Optional[int] = None   # 生效投票数（None=用 config 默认）
    sample_times: Optional[int] = None   # 生效采样数（None=用 config 默认）
    verify_only: bool = False            # L1：剩余时间不足时只验证、不再生成新候选


@dataclass
class RoundState:
    """AcceptGate 门控状态（P4）。

    契约：
    - 连续通过 >= 5 → ACCEPT；
    - 连续重大缺陷 >= 10 → REJECT；
    - 复位逻辑：一次失败复位 pass 计数，一次通过复位 major_defect 计数；
    - ``update(is_pass, has_major_defect) -> str`` 返回最新 decision。
    """
    consecutive_pass: int = 0
    consecutive_major_defect: int = 0
    decision: str = "HOLD"     # HOLD | ACCEPT | REJECT
    rounds: int = 0
    ACCEPT_PASS_THRESHOLD = 5
    REJECT_DEFECT_THRESHOLD = 10

    def update(self, is_pass: bool, has_major_defect: bool = False) -> str:
        """按一轮结果更新门控，返回最新 decision。"""
        self.rounds += 1
        if is_pass:
            self.consecutive_pass += 1
            self.consecutive_major_defect = 0
        else:
            self.consecutive_pass = 0
            if has_major_defect:
                self.consecutive_major_defect += 1
        if self.consecutive_pass >= self.ACCEPT_PASS_THRESHOLD:
            self.decision = "ACCEPT"
        elif self.consecutive_major_defect >= self.REJECT_DEFECT_THRESHOLD:
            self.decision = "REJECT"
        else:
            self.decision = "HOLD"
        return self.decision


@dataclass
class TaskContext:
    """黑板：所有 Agent 共享的推理上下文"""
    problem: str
    metadata: dict
    domain: Optional[str] = None               # ClassifierAgent 写入
    # ★ 2026-10-02：`1_classify` 阶段已删除（老师建议「聚焦推导」）⇒ 题型改**惰性求值**。
    #   原字段 `question_type` 改名 `_question_type`，对外由下方 @property 顶替 ——
    #   读写语法**完全不变**，下游 130+ 处 `getattr(ctx,"question_type","")` 读取点零改动。
    _question_type: Optional[str] = None       # 惰性题型（None=未计算；证明题/选择题/判断题/填空题/解答题）
    _enable_question_type: bool = True         # 题型识别开关（False → 属性直接返回 ""；消融实验用）
    candidates: list = field(default_factory=list)   # SolverAgent 写入
    verdicts: list = field(default_factory=list)     # VerifierAgent 写入
    revise_feedback: list = field(default_factory=list)  # 回传给 Solver 的错误原因
    trace: list = field(default_factory=list)        # 全程决策轨迹
    budget: Optional[Budget] = None            # 预算控制器
    revise_round: int = 0                       # 已触发的自纠错轮数
    final_response: str = ""
    state: RunState = field(default_factory=RunState)          # 运行期覆盖态（B6）
    round_state: RoundState = field(default_factory=RoundState)  # AcceptGate 门控态（P4）
    lemma_repo: list[str] = field(default_factory=list)  # 已验证的子结论（引理积累）
    # ---- 硬验证门禁字段（v2.5+LeanBridge；2026-09-06 起主链路由 AuditGate 承担）----
    lean_gate: list = field(default_factory=list)  # 遗留：Lean 验证诊断记录（本地证据链/旧日志兼容保留）
    audit_reject_feedback: list = field(default_factory=list)  # 被客观审核淘汰候选的反馈（供 revise）
    # ---- AuditGate 审核记录（2026-09-06 去 Lean 化：平台审核由 AuditGate 承担）----
    audit_gate: list = field(default_factory=list)      # 每候选/最终答案 AuditGate 审核诊断记录

    # ---- Lean 前置形式化验证 + 子目标细化字段（v2.9）----
    # ★ 2026-10-14：**主流水线的 2.6 前置形式化调用点已撤除**（三条原始理由全失效；
    #   详见 orchestrator__prep.py 内撤除说明）⇒ 下列三字段在**主流水线恒为空**，
    #   只有独立入口 `cli/leap_eval.py` 会写（与 sketch_tree / refine_result 同纪律）。
    formal_spec: str = ""                           # 题目前置形式化描述（已知条件/结论，LeanPreVerifier 写入）
    formal_gaps: list = field(default_factory=list) # Lean 形式化编译发现的「缺口」（缺失定义/引理/模块/类型），供子目标规划优先拆解
    # AI 实际检索/引用过的 Mathlib 定理名（2026-08-29 新增）
    # 老师 #1/#2 的证据链：AI 解答 → Lean 形式化验证，用了哪些定理要有记录。
    # leansearch 检索命中、lean_gate 编译、lean_refiner 补全都追加到这里（去重）。
    used_theorems: list = field(default_factory=list)
    # Mathlib 使用统计（2026-08-29 新增，回应"调用频繁但定理不多"的观察）
    # search_calls: leansearch 调用次数；search_hits: 检索命中的定理条数（未去重）；
    # compile_valid: lean_gate 编译通过的候选数；distinct_theorems 由 used_theorems 推出。
    # 三者的关系 = "检索了 N 次 → 命中 M 条 → 实际编译通过 K 个候选"，
    # 让"检索多但有效定理少"这个现象有数据可查。
    mathlib_usage_stats: dict = field(default_factory=lambda: {
        "search_calls": 0, "search_hits": 0, "compile_valid": 0})
    preverify_trace: dict = field(default_factory=dict)  # 前置验证轨迹（通过/失败/修正轮次/lean声明）
    subgoal_trace: list = field(default_factory=list)    # 每步子目标过程与中间结果（结构化输出）
    subgoal_merge_plan: str = ""                    # 最终整合方案

    # ---- 骨架(sketch) 阶段 Lean 审核字段（#28）----
    # ★ 2026-10-14：同上前置形式化 —— 主流水线不写（恒空），仅 `cli/leap_eval.py` 写。
    sketch: str = ""                                 # 书生生成的题目骨架/Proof Body Outline（LeanPreVerifier 写入）
    sketch_audit: dict = field(default_factory=dict) # 骨架 Lean 语法审核结果（verdict/gaps/lean_code/formal_spec）

    # ---- Blueprint DAG（LEAP Stage 1，#27）----
    blueprint: dict = field(default_factory=dict)    # AND-OR DAG 序列化（BlueprintPlannerAgent 写入）
    blueprint_plan: dict = field(default_factory=dict)  # DAG 转出的子目标规划（兼容 SubGoalSolver）

    # ---- 单一「题目目标」（任务列表 #3，2026-10-14）----
    # 题意理解的**唯一产物**：`agent/problem_goal.py::build_problem_goal` 从既有产物
    # （question_type / type_path / understanding_chain / extract_options）派生，
    # **零 LLM 调用**。字段：question_type / task / options / n_correct /
    # answer_form / terminal_state / answer_slot / goal_statement。
    # ⚠ 原先「理解题意、抽目标」在**三处**各做一遍（蓝图 LLM 的 problem_analysis、
    #   子目标 LLM 的 problem_analysis、选择题两处代码构造的 problem_analysis）
    #   —— 那些副本已随 #3 删除，蓝图/子目标/merge 渲染一律改读本字段。
    problem_goal: dict = field(default_factory=dict)

    # ---- 整树 Lean 搭桥（LEAP Stage 2，#26/#28）----
    # ⚠ 2026-09-30 注释订正（原注释「LeanTranslatorAgent 写入」在主流水线里是**错的**）：
    #   写入方 `LeanTranslatorAgent` 确实存在（`tools/lean_local/lean_translator.py`），
    #   但它**已从主流水线（orchestrator）摘除** —— `sub_goal_solver.py:1602-1605`
    #   刻意移除调用，理由："平台无 Lean，整树翻译+编译只空转"。
    #   ⇒ **主流水线跑完本字段恒为空**；只有独立入口 `cli/leap_eval.py`
    #     才会写它（该入口自带测试，仍可用）。
    #   判读纪律：不得因本字段为空就声称"Lean 没检查过"——主线的 Lean 检查走
    #   `lean_gate`（见下）。
    sketch_tree: dict = field(default_factory=dict)

    # ---- Stage 3 迭代精炼（#29/#30/#32/#33）----
    # ⚠ 2026-09-30 注释订正：同上，「LeanRefinerAgent 写入」仅在独立入口成立。
    #   主流水线不调用 ⇒ **本字段恒为空**，故 `gap_analyzer.extract_gaps()`
    #   （消费本字段）在主链路下**恒返回 "无 refine_result"**；
    #   现役缺口分析走 `extract_subgoal_gaps()`（subgoal_trace）
    #   （Lean 侧 `extract_lean_dag_gaps()` 已随求解前门一并撤除）。
    #   详见 gap_analyzer.ORPHAN_NOTE。
    refine_result: dict = field(default_factory=dict)

    # ---- DAG 动态评审（LEAP 5.3，#34，2026-09-01）----
    # DagReviewerAgent 写入：评审 DAG 分解质量，建议触发整树重生成
    dag_review_report: dict = field(default_factory=dict)

    # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：
    #   原 `lean_dag_fails` / `lean_dag_checked`（求解前 Lean 逻辑检查落点）随
    #   `_lean_dag_logic_check` 一并撤除 —— 其唯一调用者是默认关闭的
    #   `dag_replan_gate`，字段在生产中恒为空。

    # ---- 领域 → 定理检索（2026-09-29 新增，截图 #3+#4）----
    # 由 orchestrator 的 1.6) 阶段写入（实现在 agent/theorem_hint.py）。
    # 语义：把"本题该用的 Mathlib 定理"检索出来，供 2.6 前置理解 / 3_solve
    # 提示词按需引用；trace 供离线统计命中率与耗时。
    # ★ 缺省为空 ⇒ 不检索时下游行为与改动前**完全一致**。
    theorem_hints: list = field(default_factory=list)   # list[str] Mathlib 全名
    theorem_hint_block: str = ""        # 注入提示词的文本块（空=不注入）
    theorem_hint_trace: dict = field(default_factory=dict)  # 完整埋点

    # ---- 统一档位标识（2026-09-29 起：不再分档，恒为 "deep"）----
    # 历史：fast / standard / deep 三档，由 DifficultyRouter 按难度写入。
    # 2026-09-29 用户决策：研究目标 = 用**全部方法**提升正确率，按时间/档位
    # 区分遂违背初心 ⇒ 删档位路由，**一律取原 deep 档最强配置**。
    # 本字段保留仅为观测/诊断兼容（诊断表仍按 tier 聚合），**无任何分支作用**。
    tier: str = "deep"                          # 常量标识，恒 "deep"
    tier_evidence: dict = field(default_factory=dict)  # 档位判定依据（统一档后恒为空）
    soft_budget: float = 0.0                    # 单题设计预算帽（秒），来自 tier_budget["deep"]
    # 2026-10-02：原「单题剩余时间」诊断字段随 `0_paper_pacer` 阶段删除（唯一写入点已移除）。

    # ---- 壁钟时间追踪（适配竞赛新规则：单题≤20分钟，总计≤6小时）----
    start_time: float = 0.0                    # 单题壁钟启动时间 (time.time())
    deadline: float = 0.0                      # 单题绝对截止时间戳
    total_start_time: float = 0.0              # Agent总启动时间
    total_deadline: float = 0.0                # Agent总截止时间戳
    # 进入"时间紧急"的尾部阈值（秒）：剩余不足该值则跳过可选步骤。
    # 2026-08-28：原先硬编码 300s，对 deep 档（1200s 预算）意味着最后 25%
    # 的时间直接放弃协作验证/ revise / Lean，"难题用满 20 分钟"根本做不到。
    # 改为可配置，默认 120s（由 orchestrator 按档位从 config 写入）。
    critical_tail_seconds: float = 120.0
    # 生成类阶段软截止（2026-09-06 超时修复）：orchestrator 在档位确定后
    # 设 _gen_deadline = deadline - verify_reserve，凡"追加生成"类模块
    # （solver/improve/collab/sub_goal 的生成循环）在循环边界用 gen_time_up()
    # 判断是否停手，把 verify_reserve 秒强制留给 4_verify 投票 + 6.5 审核。
    # 未设置（测试 fixture / 其他调用方）= 0.0 → gen_time_up() 回退 is_time_critical，
    # 行为与旧版完全一致。
    _gen_deadline: float = 0.0

    # ------------------------------------------------------------
    # 题型（惰性求值属性，2026-10-02 起）
    # ------------------------------------------------------------
    # `1_classify` 阶段已删除 ⇒ 题型不再预填、不缓存进 ctx；改为**首次访问时按需分类**
    # 并缓存，对下游读取点完全透明（契约不变）。
    # `enable_question_type=False`（消融）时直接返回 ""，不触发任何分类。
    @property
    def question_type(self) -> str:
        """题型（惰性：首次访问时按需分类，不再由 `1_classify` 阶段预填）。

        2026-10-02：`1_classify` 阶段已删除（老师建议「聚焦推导」）。
        分类改为**按需计算**，对下游读取点完全透明（契约不变）。
        """
        if not getattr(self, "_enable_question_type", True):
            return ""
        if self._question_type is None:
            try:
                from .question_type import classify_question_type
                self._question_type = classify_question_type(self.problem)
            except Exception:  # noqa: BLE001  分类失败 → 空串（与旧默认一致，绝不阻断）
                self._question_type = ""
        return self._question_type

    @question_type.setter
    def question_type(self, value) -> None:
        # 兼容既有**写入点**（`agent/classifier.py` 的 classifier.run、单测显式赋值）：
        # 显式赋值即缓存该值；置 `None` 表示"重新惰性求值"。
        self._question_type = value

    # ---- 时间检查方法（适配竞赛新规则）----
    def time_remaining(self) -> float:
        """返回当前题目剩余壁钟时间（秒），负值=已超时。

        2026-09-03 审核统一口径：``BaseAgent.llm`` 早已把 ``deadline < 1e8``
        视为「测试 fixture 的伪 epoch 值」按无限剩余处理，而本方法没有，
        导致同一个 ctx 在 llm() 看是"时间充裕"、在 is_time_critical() 看是
        "已超时"——测试里表现为子目标求解被误报"预算耗尽"整体跳过。
        两处口径统一，避免这类自相矛盾的时间判断。
        """
        if self.deadline == 0.0 or self.deadline < 10**8:
            return float("inf")
        import time
        return self.deadline - time.time()

    def is_time_critical(self) -> bool:
        """距当前题目超时不足 critical_tail_seconds → 跳过所有可选步骤。

        阈值历史：120 → 300（P1 修复）→ 改为可配置（默认 120）。
        硬编码 300 会让 deep 档（1200s）最后 1/4 时间白白放弃协作验证与
        自纠错，与"难题用满 20 分钟"的策略直接冲突。
        """
        return self.time_remaining() < self.critical_tail_seconds

    def is_timed_out(self) -> bool:
        """当前题目是否已超时"""
        return self.time_remaining() <= 0.0

    def gen_time_up(self) -> bool:
        """生成类阶段是否应停手（生成侧软截止已到）。

        2026-09-06 超时修复：生成类模块的循环边界统一改查本方法——
        若 orchestrator 设了 _gen_deadline（单题 deadline 前 verify_reserve 秒），
        则到点即停，避免"最后一段生成把剩余预算烧穿 → 验证投票全跳、
        答案零验证裸提交"（冒烟 4/5 题实证）。未设 _gen_deadline 时回退
        原 is_time_critical 语义，测试/其他调用方行为零变化。
        """
        gd = getattr(self, "_gen_deadline", 0.0) or 0.0
        if gd and gd >= 10**8 and self.deadline and self.deadline >= 10**8:
            import time
            return (time.time() > gd) or self.is_time_critical()
        return self.is_time_critical()

    def total_time_remaining(self) -> float:
        """返回Agent总剩余时间（秒）"""
        # 2026-09-12 定型前审核：与 time_remaining() 统一「伪 epoch」护栏。
        # 原先只挡 `== 0.0`，未挡测试 fixture 的伪 epoch（如 9999.0）→
        # is_total_timed_out() 恒 True，而同一个 ctx 上 time_remaining()
        # 却返回 inf，属"同一实体内两套矛盾口径"（会上演'某机制永远触发'）。
        if self.total_deadline == 0.0 or self.total_deadline < 10**8:
            return float("inf")
        import time
        return self.total_deadline - time.time()

    def is_total_timed_out(self) -> bool:
        """Agent总时间是否已用尽"""
        return self.total_time_remaining() <= 0.0
