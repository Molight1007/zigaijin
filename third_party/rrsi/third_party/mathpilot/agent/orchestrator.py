from __future__ import annotations


# 2026-10-02 中间结果存储层（李平老师架构建议 #5，老师点名的「耦合过紧」根因）：
# 把题意理解 / 蓝图 / 子目标 / 候选 / verdict / Lean / 终答**多存一份**到
# results/<run_id>/<qid>/。只加不改：不删分支、不改判定、不动返回值；
# 每一处调用都**永不抛异常**（失败只落一条 trace，绝不阻断主链）。
from agent.artifact_store import (
    put_ctx as _artifact_put, attach as _artifact_attach)
"""
编排器（Orchestrator）—— 简化版
================================

借鉴 ss-main 的简洁流水线，不做复杂回环，每道题 LLM 调用控制在 7 次以内：

    Classifier → Solver → Verifier → Formatter
    (1次LLM)   (3次并行)  (3次投票)  (无LLM)

弱化改动：
- 蓝图分解默认开启（`use_blueprint=True`；2026-10-01 研究期按用户决策默认开）
- 不设自纠错回环（直接用聚类选最优候选）
- 不设完整性审核链（省去 3+ 次 LLM 确认与续写）
- Symbol 快车道仍在（可确定性求解时短路）
"""

import logging
import time
import re as _re

# 2026-09-13 晚：拒绝/占位符答案的正则。与 `agent/formatter.py` 同源，此处独立
# 一份是为了避免 orchestrator↔formatter 的导入顺序纠缠（formatter 已被本文件
# import，反向引用成环）。两处**必须同步修改**。
# （2026-10-15：user_agent 那份已随降级兜底删除）
# ⚠ 口径刻意收窄：**不含** `无解` / `暂无` —— 它们是**合法答案**（「该方程无解」就是
#   正确答案）。`formatter._REFUSAL_RE` 里保留这两个词是 HEAD 既有语义（用于"换个
#   候选试试"），但这里决定的是"有没有可用候选"，误判会让 `_fallback_direct`
#   用直答**覆盖正确解**（独立验证者实测：唯一候选 answer='无解' → 被覆盖成 '7'）。
_DEGRADED_ANSWER_RE = _re.compile(
    r"生成失败|调用受限|拒绝回答|未给出有效解答|无法作答|子目标求解失败|"
    r"我无法|无法求解|无法解决|不能解决|无法解答",
    _re.IGNORECASE,
)


from .base import BaseAgent, TaskContext, Budget
from .classifier import ClassifierAgent, _KNOWN_DOMAINS
from .solver import SolverAgent
from .sub_goal_solver import SubGoalSolverAgent
# ★ 2026-10-06：`_clip_review_prose` **直接 import 复用**（不复制正则）——
#   复核清单与 verifier 深复核原文需同一道清洗口径，两处各写一份必然分叉
#   （仓库既有教训）。verifier 不 import orchestrator ⇒ 无循环导入。
from .verifier import VerifierAgent
# ★ 2026-10-07（问题总账 §2.6「择优逻辑口径分叉」）：终答兜底择优**复用** formatter
#   的统一排序键 `_rank_key`（票数→独立复现数→置信度→答案形态→答案更短），
#   以及配套的 `_attach_support`（把"独立复现数"挂到候选上）。**只读引用，不改
#   formatter.py**。formatter 不 import orchestrator ⇒ 无循环导入。
from .formatter import (
    FormatterAgent)
from .collaborative_solver import CollaborativeSolver
from .adversarial_verifier import AdversarialVerifier
from .audit_gate import AuditGate
from utils.extract import safe_json_serialize
# 2026-10-07 阶段C1：`collect_param_usage` 已随 _collect_diag 移到
# agent/diag_export.py；此处**原位 re-export** 以保契约（
# tests/test_param_usage_0921.py 断言 orchestrator 可导入该符号）。
from .param_usage import collect_param_usage  # noqa: F401
# 2026-10-07 阶段C1：diag 专用汇总函数抽到 agent/diag_metrics.py；此处 re-export，
# 既有 `from agent.orchestrator import _merge_error_types` 等调用点/测试零改动。
from .diag_metrics import (  # noqa: F401
    _merge_error_types, _sum_reject_votes, _compute_subgoal_gaps,
    _switch_snapshot, _tool_gateway_summary,
    _parse_exhaust_result, _diag_completeness, _summarize_deep_review,
    _summarize_leansearch)
from .diag_export import collect_diag  # noqa: F401  （供 _collect_diag 委托）
# 2026-10-07 阶段C2：6.5_audit_gate 段已抽到 agent/orchestrator__final_gate.py。
# ★ 2026-10-14：6.5_audit_gate 阶段已删（五道链定案）；本 import 仅剩 `format_stage`
#   （6_format 段体，含终答后处理：数值化 + 客观题自检）。
from .orchestrator__final_gate import format_stage  # noqa: F401
# ★ 2026-10-14：3.6_audit_filter 段已删（职责并入 4.3/4.4）⇒ 不再 import。
from .answer_normalizer import normalize_stage  # noqa: F401
from .self_auditor import self_audit_stage  # noqa: F401
from .adversarial_gate import adversarial_gate_stage  # noqa: F401
from .lean_backcheck import lean_backcheck_stage  # noqa: F401
from .math_check_gate import math_check_stage  # noqa: F401
from .vote_gate import vote_stage  # noqa: F401
from .rewrite_gate import rewrite_stage  # noqa: F401
# ★ 2026-10-14：5.5_low_conf 段已删（五道链定案）⇒ 不再 import。
# 2026-10-07 阶段C3c：3.4_collab 段已抽到 agent/orchestrator__collab.py，
# 由 run 一行委托调用（阶段打点原位保留，契约不变）。
from .orchestrator__collab import collaborative_deep_improve  # noqa: F401
# 2026-10-07 阶段C4a：2.7_subgoal_main 段已抽到 agent/orchestrator__solve.py，
# 由 run 一行委托调用（阶段打点原位保留，契约不变）。
from contracts.switch_registry import get_bool as _sw_bool
from .orchestrator__solve import subgoal_main_path  # noqa: F401
# 2026-10-07 阶段C4b：3.2_complete 段已抽到 agent/orchestrator__solve.py，
# 由 run 一行委托调用（阶段打点原位保留，契约不变）。
from .orchestrator__solve import complete_truncated  # noqa: F401
# 2026-10-07 阶段C4c：3.3_improve 段已抽到 agent/orchestrator__solve.py，
# 由 run 一行委托调用（阶段打点原位保留，契约不变）。
from .orchestrator__solve import self_improve_step  # noqa: F401
# 2026-10-08 阶段C4e：3_solve 段已抽到 agent/orchestrator__solve.py，
# 由 run 以「哨兵 + 解包」方式委托（阶段打点原位保留，契约不变）。
# ⚠ 非纯逐字搬：段体末尾新增哨兵 return，调用点由裸段改为解包透传。
from .orchestrator__solve import solve_stage, _SOLVE_CONTINUE  # noqa: F401
# ★ 2026-10-14：4.5_oracle / 4_verify / 4.6_adv 段已删（五道链定案：
#   oracle 删、投票并入 4.5_vote、对抗升级为 4.2_falsify）⇒ 不再 import。
# 2026-10-07 阶段C6：1.2_theorem_hint 段已抽到 agent/orchestrator__prep.py，
# 由 run 一行委托调用（阶段打点原位保留，契约不变）。
# ★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
#   `1.2_theorem_hint` 阶段**已整体撤销** ⇒ 不再 import `theorem_hint_stage`
#   （该函数已删）。检索改由 `sub_goal_solver.run` 体首调
#   `agent.theorem_hint.run_hint_for(agent, ctx)`；本模块的
#   `Orchestrator._retrieve_theorem_hints` 仍保留为薄委托（兼容既有调用/测试）。
from .orchestrator__prep import prep_tier_budget  # noqa: F401
# ★ 2026-10-14（任务列表 #1「题型判断要和题意理解合并」）：原独立阶段
#   `1.1_type_understanding` 已撤销 —— 其逻辑（`judge_question_type`）下沉为
#   `2.6_pre_audit`（题意理解）的第一个子步骤，由 `pre_audit_stage` 内部调用。
#   故此处**不再需要** import 判题函数（`pre_audit_stage` 同模块内直呼）。
# 2026-10-07 阶段C6：2.6_pre_audit 段已抽到 agent/orchestrator__prep.py，
# 由 run 一行委托调用（阶段打点原位保留，契约不变）。
from .orchestrator__prep import pre_audit_stage  # noqa: F401
# ★ 2026-10-14：4.6_adv 段已删 ⇒ 不再 import adversarial_stage。
from . import orchestrator__numeric as _orch_numeric
from . import orchestrator__select as _orch_select
from . import orchestrator__fallback as _orch_fallback
from . import orchestrator__budget as _orch_budget
from . import orchestrator__lean as _orch_lean
# 2026-10-09 批16：orchestrator 的模块级工具函数已搬到
# agent/orchestrator__helpers.py。此处**原位 re-export**，既有
# `from agent.orchestrator import <名>` 调用点零改动。
# （2026-10-15：`_emergency_direct_max_tokens` 已随直答兜底删除）
from .orchestrator__helpers import (  # noqa: F401
    _sw_bool_or, _sw_num_or,
    _load_lean_modules, _answer_fingerprint, _merge_review_into_feedback,
    _collect_tool_findings)
# ★ 2026-10-14：5_revise_or_fallback 段已删（用户决策②「全删」）⇒ 不再 import。

# ---- Lean 双通道（2026-09-06 晚恢复）----
# lean 系工具随 P2 去 Lean 化迁到 tools/lean_local/（归档），lean-toolchain
# 工具仓离线落地后重新接入检测链：Lean 环境可用 → 硬验证，不可用 → AuditGate 兜底。
# 惰性 import：模块缺失/import 异常一律视 lean 不可用，绝不拖垮主链路。
try:
    from tools.lean_local.lean_gate import LeanGate
    from tools.lean_local.lean_pre_verifier import LeanPreVerifier
    _LEAN_MODULES_OK = True
except Exception as _e_lean:  # noqa: BLE001
    LeanGate = None
    LeanPreVerifier = None
    _LEAN_MODULES_OK = False
    logger = logging.getLogger("MathPilot")
    # ★★ 2026-10-24（用户：「有问题却查不到本身就是 bug」）：
    #   原实现是裸 `except Exception:`——连异常对象都不接，只打一句"不可用"，
    #   于是 `lean_active=false` 连跑四轮都没人知道为什么。现在：
    #   ① 日志带 exc_info（完整 traceback）；
    #   ② 文本存进模块全局 `_LEAN_IMPORT_ERROR`，供诊断器/外部脚本直接读。
    import traceback as _tb
    _LEAN_IMPORT_ERROR = _tb.format_exc()
    logger.warning("tools.lean_local 不可用，Lean 双通道禁用（回落 AuditGate）: %s",
                   _e_lean, exc_info=True)


try:
    from utils.sympy_tools import (
        _HAS_SYMPY, eval_expression, compute_derivative,
        compute_integral, compute_determinant, solve_equation,
        compute_limit,
    )
except ImportError:
    _HAS_SYMPY = False

logger = logging.getLogger("MathPilot")


class Orchestrator(BaseAgent):
    name = "Orchestrator"

    def __init__(self, client, config):
        super().__init__(client, config)
        self.classifier = ClassifierAgent(client, config)
        self.solver = SolverAgent(client, config)
        self.sub_goal_solver = SubGoalSolverAgent(client, config)
        self.verifier = VerifierAgent(client, config)
        self.formatter = FormatterAgent(client, config)
        # 2026-09-29：DifficultyRouter（难度路由）与 PaperPacer（全卷时间池）
        # 已按用户决策删除 —— 统一单一档位，不再有按难度/配额裁剪方法的机制。
        # deep 档难题三Agent协作求解器（v2.6：解题→审查→整合→反复验证）
        self.collab = CollaborativeSolver(client, config)
        # AuditGate 答案审核闸门（2026-09-06 去 Lean 化：取代 lean_gate /
        # lean_pre_verifier 在检测链中的全部作用——平台无 Lean，硬核验/反例/
        # rubric 程序与 LLM 混合瀑布，只审不答）
        self.audit_gate = AuditGate(client, config)
        # Lean 双通道（2026-09-06 晚恢复）：lean 可用即 LeanGate 硬验证；
        # ★ 2026-10-14：LeanPreVerifier 在 2.6 的**前置形式化调用点已撤除**
        # （三条原始理由全失效 + 无 Lean 环境时从未跑过），其实例仍保留 ——
        # 供 2.6 题意理解 `run_understanding_chain` 与本地 `cli/leap_eval.py` 使用。
        # 两通道不可用时由 _lean_active() 回落 AuditGate。
        self.lean_gate = (LeanGate(client, config)
                          if _LEAN_MODULES_OK and LeanGate is not None else None)
        self.lean_pre_verifier = (
            LeanPreVerifier(client, config)
            if _LEAN_MODULES_OK and LeanPreVerifier is not None else None)
        self._lean_probe: bool | None = None   # lean 环境探测结果（进程内缓存）
        # 对抗式验证器（#16，2026-08-30）：正向验证**通过后**主动证伪，治漏检。
        # 与 Step 4（_review_bug_feedback，治误杀）互补：
        #   正向不过 → Step 4 复核是否误报
        #   正向通过 → 本模块去找漏掉的错误
        self.adv_verifier = AdversarialVerifier(client, config)

    # ------------------------------------------------------------------
    # Lean 双通道探测（2026-09-06 晚）
    # ------------------------------------------------------------------
    def _ensure_lean_modules(self) -> None:
        return _orch_lean._ensure_lean_modules(self)

    def _lean_active(self) -> bool:
        return _orch_lean._lean_active(self)

    def _lean_applicable(self, ctx) -> bool:
        return _orch_lean._lean_applicable(self, ctx)

    # ------------------------------------------------------------------
    # 领域 → 定理检索（2026-09-29 新增，截图 #3+#4）
    # ------------------------------------------------------------------
    def _retrieve_theorem_hints(self, ctx) -> None:
        return _orch_lean._retrieve_theorem_hints(self, ctx)

    # ------------------------------------------------------------------
    # 阶段耗时埋点（2026-09-03 老师：deep 档需要每个环节具体耗时做决定）
    # ------------------------------------------------------------------
    @staticmethod
    def _stage_start(ctx, name: str) -> None:
        # 2026-09-03 修正：进入新阶段前先结束所有未 stop 的旧阶段。
        # 否则 stop 只在流程末尾统一调用 → 每阶段耗时 = "从该阶段开始到
        # 流程结束"的剩余时间（实测 0_paper_pacer=1310s 全等于总时长，数据无效）。
        for _old in list((getattr(ctx, "_stage_starts", {}) or {}).keys()):
            if _old != name:
                Orchestrator._stage_stop(ctx, _old)
        ctx._stage_starts = getattr(ctx, "_stage_starts", {}) or {}
        ctx._stage_starts[name] = time.time()

    @staticmethod
    def _stage_stop(ctx, name: str) -> None:
        starts = getattr(ctx, "_stage_starts", {}) or {}
        s = starts.pop(name, None)
        if s is None:
            return
        ctx._stage_durs = getattr(ctx, "_stage_durs", {}) or {}
        ctx._stage_durs[name] = ctx._stage_durs.get(name, 0.0) + (time.time() - s)

    def _collect_stage_durs(self, ctx) -> dict:
        return dict(getattr(ctx, "_stage_durs", {}) or {})

    # ------------------------------------------------------------------
    # C-lite：数值攻击蓝图极值声称（2026-09-03 老师拍板）
    # ------------------------------------------------------------------
    def _final_answer_postprocess(self, ctx, ans: str = "") -> str:
        return _orch_numeric._final_answer_postprocess(self, ctx, ans)

    @staticmethod
    def _latex_const_value(core: str):
        return _orch_numeric._latex_const_value(core)

    def _maybe_numericize(self, ctx, ans: str) -> str:
        return _orch_numeric._maybe_numericize(self, ctx, ans)

    def _objective_selfcheck(self, ctx, ans: str) -> str:
        return _orch_numeric._objective_selfcheck(self, ctx, ans)

    # ---- B0 答案形态闸门（2026-09-13）----
    # 台账 B0 两个可救模式：
    #   ① 条件式代替具体值（015 答 `n ≥ 2` / 正解 `2`；064 答 `m is even` / 正解 `4`）
    #   ② 求所有却只给一支（003 漏 2030；074 漏取整解族）
    _ASKS_ALL_RE = _re.compile(
        r"find\s+all|determine\s+all|all\s+possible|"
        r"求.{0,12}所有|求.{0,12}全部|所有\s*可能|全部\s*可能",
        _re.IGNORECASE)
    _ASKS_RANGE_RE = _re.compile(
        r"范围|区间|取值范围|充要条件|必要条件|充分条件", _re.IGNORECASE)
    # 条件式答案：同时覆盖 **Unicode 符号**（≥≤）与 **LaTeX 命令**（\geq \le …）
    # —— 实测 015 的答案是 `n \geq 2`（LaTeX），只认 ≥ 会漏判。
    _COND_ANS_RE = _re.compile(
        r"(^|[\s（(])([a-zA-Z]\s*)?(≥|≤|>=|<=|>|<)"
        r"|\\geq|\\leq|\\ge\b|\\le\b|\\gt|\\lt"
        r"|is\s+(even|odd)|for\s+all|对任意|任意\s*[a-zA-Z]", _re.IGNORECASE)

    def _answer_form_gate(self, ctx, ans: str) -> str:
        return _orch_numeric._answer_form_gate(self, ctx, ans)

    # 元话语特征 —— 重问可能返回"角色说明/指令复述"而非答案。
    # 003 实测：模型返回 `The user wants me to act as a math answer formatting tool.`
    _META_ANS_RE = _re.compile(
        r"the user|user wants|i (will|should|am)\b|让我|需要我|要求我|"
        r"作为.{0,8}(工具|助手|规范|器)|assistant|instruction|prompt",
        _re.IGNORECASE)

    # 题面复述 / 疑问式措辞 —— 2026-09-14 加严（#005 实测）。
    # 实况：重问把 `\boxed{1234}` 改写成
    #   `The problem asks for all possible values of $C(1234)$ …`
    # 旧判据 `re.search(r"\d", txt)` 因为它含 `1234` 而**放行**，
    # 结果答案从"可判的数值"退化成"不可判的散文"（error_class=format_unresolved）。
    # ⚠ 刻意只收**题面/疑问式**措辞，不收一般英文散文 —— 因为本题库存在
    #   **合法英文答案**（如 011 的 gold `$f(x,y)= g(x+y, xy(x-y)^{2})$ for some
    #   polynomial $g$`），过宽会误杀正确解。
    _RESTATE_ANS_RE = _re.compile(
        r"the\s+(problem|question|task|answer\s+is\s+asked)\b|"
        r"\basks?\s+(for|us\s+to)\b|\bwe\s+(need|want|must|are\s+asked)\b|"
        r"\blet\s+us\b|\bfind\s+all\b|\ball\s+possible\s+values?\b|"
        r"\bwhat\s+is\b|\bwhich\s+of\s+the\s+following\b|\bnote\s+that\b|"
        r"\bthe\s+user\b",
        _re.IGNORECASE)

    @staticmethod
    def _looks_like_answer(txt: str) -> bool:
        return _orch_numeric._looks_like_answer(txt)

    def _ask_for_form(self, ctx, cur: str, asks_all: bool) -> str:
        return _orch_numeric._ask_for_form(self, ctx, cur, asks_all)

    def _lean_crosscheck_number(self, ctx, latex_expr: str, value: str) -> bool:
        return _orch_numeric._lean_crosscheck_number(self, ctx, latex_expr, value)

    def _self_improve_applicable(self, ctx) -> bool:
        return _orch_budget._self_improve_applicable(self, ctx)

    def _phase_budget(self, ctx, ratio: float = 0.0, floor: float = 180.0,
                      reserve: float = 0.0, stage: str = "") -> float:
        return _orch_budget._phase_budget(self, ctx, ratio, floor, reserve, stage)

    def _phase_deadline_guard(self, ctx, cap: float):
        return _orch_budget._phase_deadline_guard(self, ctx, cap)

    @staticmethod
    def _phase_deadline_restore(ctx, saved) -> None:
        return _orch_budget._phase_deadline_restore(ctx, saved)

    def _value_attack_blueprint(self, ctx, tier: str) -> None:
        return _orch_numeric._value_attack_blueprint(self, ctx, tier)

    # ----------------------------------------------------------
    # 主入口（简化版流水线）
    # ----------------------------------------------------------
    def run(self, problem: str, metadata: dict) -> dict:
        now = time.time()
        # ---- 全卷时钟锚点（2026-09-13 修复「全卷时钟是死的」）------------------
        # 原实现把 total_start_time / total_deadline 都基于 `now`，而平台是**逐题**
        # 调用 solve()→run() 的 ⇒ total_deadline 每题都被重置成"此刻 + 6.25h"
        # ⇒ elapsed_total 恒 ≈ 0、ratio 恒 0。
        # ★ 2026-10-01：原先依赖 ratio 的「应急模式 / 时间收紧」两个全卷保护分支
        #   已按研究期标尺**整体删除**（该阶段已于 2026-10-02 删除，见 CHANGES）。
        #   保留锚点仅用于**诊断字段**（`ctx.total_deadline` 等），不再驱动任何
        #   降级逻辑；全卷不做时间总量控制。
        # 锚点只在**首题**锁定一次、跨题保留（并发下"先到先写"，幂等）。
        if getattr(self, "_paper_anchor", None) is None:
            self._paper_anchor = now
        _anchor = float(self._paper_anchor)
        ctx = TaskContext(
            problem=problem,
            metadata=metadata or {},
            budget=Budget(max_calls=self.config.max_total_calls),
            start_time=now,
            deadline=now + getattr(self.config, 'max_time_per_question', 300),
            total_start_time=_anchor,
            total_deadline=_anchor + getattr(self.config, 'max_total_time_seconds', 21000),
            # 2026-10-02：题型惰性求值开关（`1_classify` 阶段删除后由属性消费）。
            _enable_question_type=getattr(self.config, "enable_question_type", True),
        )
        # ★ 2026-10-02 hook 0：中间结果存储层挂载（关闭/失败 → 静默 no-op，主链无感）。
        #   run_id 取进程级稳定值（一轮评测一个目录）；qid 由题面哈希生成（文件名不含中文）。
        try:
            _artifact_attach(ctx, config=self.config, problem=problem)
        except Exception:  # noqa: BLE001  存储层任何问题都不得阻断求解
            pass
        try:
            # ★ 2026-10-02：`0_paper_pacer` 阶段已删除（用户 2026-10-02 点名「全卷配速删了」）。
            #   该阶段原仅记录单题剩余时间的诊断字段、无任何实际逻辑（PaperPacer 已于
            #   2026-09-29 删除），其诊断字段随本阶段一并移除。历史报告仍含该阶段名
            #   （阶段名是稳定契约，此处仅留痕）。
            # 单题 20 分钟硬限：超时直接跳过（保留已有候选/兜底产出）
            if ctx.is_timed_out():
                self.record(ctx, "timeout", "单题超过硬限，跳过处理")
                # 超时分支先看已有候选（这是智能体已产出的答案，非直答）；
                # 无候选 ⇒ 收集错误信息 + 报错（2026-10-15 定案，不再直答）。
                answer = self._pick_best_from_candidates(ctx)
                if answer:
                    _artifact_put(ctx, "08_final", {
                        "final_response": answer, "timeout": True})
                    return safe_json_serialize({
                        "final_response": answer, "trace": ctx.trace,
                        "diag": self._collect_diag(ctx),
                    })
                return self._raise_solve_error(ctx, "timeout_no_candidate")
            # 2026-10-01：比赛期「全卷时间池按比例降级」已按研究期标尺删除
            # （审计 A 级第 1 条）。原 ratio>0.95 / >0.8 三档会静默置
            # `ctx.state.emergency`，级联关闭 P1 强制重解、2.6 前置验证、
            # 2.7 子目标主路径、3.2 续写、3.3 自改进等能力；且与上方
            # 「PaperPacer 已彻底删除、全卷不做时间总量控制」的注释直接矛盾。
            # ★ 注意：RunState.playoff_enabled 默认为 False，必须在正常态**显式打开**，
            #   否则会静默关掉 playoff（反而多裁一项能力）。
            # 单题唯一约束仍是 `max_time_per_question` 硬墙（见上方 is_timed_out）。
            ctx.state.emergency = False
            ctx.state.playoff_enabled = True

            # ★ 2026-10-02：`1_classify` 阶段已删除（老师建议「聚焦推导」）。
            #   题型不再预分类、不缓存进 ctx；`ctx.question_type` 改为**惰性属性**
            #   （首次读取时按需调 `classify_question_type`，见 agent/base.py）。
            #   下游 130+ 处 `getattr(ctx,"question_type","")` 读取点**零改动**。
            #   历史报告仍含 `1_classify` 阶段名（阶段名是稳定契约，此处仅留痕）。
            #   `enable_question_type=False` 时惰性属性直接返回 ""（消融路径不变）。

            # 1.1) Lean 适用性判定（2026-09-10 用户要求「所有题都要用 Lean，
            #      除非非常简单的题」）：答案不是数学对象的题（选项字母 /
            #      判断值 / 概念文字）Lean 结构上无从形式化核验，按题豁免，
            #      而不是用全局开关一刀切。official112 标定：豁免 18/112，
            #      误伤 0 道。结果写入 ctx.metadata 供 LeanGate 读取（tools/
            #      不反向依赖 agent/ 包），并把原因落进 diag 供事后追溯。
            try:
                from .question_type import lean_applicable
                _la_ok, _la_why = lean_applicable(
                    ctx.problem or "", getattr(ctx, "question_type", "") or "")
                if isinstance(getattr(ctx, "metadata", None), dict):
                    ctx.metadata["lean_applicable"] = bool(_la_ok)
                    if not _la_ok:
                        ctx.metadata["lean_skip_reason"] = _la_why
                    self.record(ctx, "lean_applicable",
                                f"Lean 适用性: {'适用' if _la_ok else '豁免(' + _la_why + ')'}",
                                lean_applicable=bool(_la_ok))
            except Exception as _e:  # noqa: BLE001
                # 判定失败 → 不豁免（保持 Lean 全跑，宁可多跑不漏跑）
                logger.warning("Lean 适用性判定失败，按适用处理: %s", _e)

            # 1.5) 领域（元数据已知时跳过 LLM）
            pre_known_domain = (metadata or {}).get("domain", "")
            if pre_known_domain and pre_known_domain in _KNOWN_DOMAINS:
                ctx.domain = pre_known_domain
                self.record(ctx, "classify",
                    f"题型分类（元数据已知）: {pre_known_domain}", domain=pre_known_domain)
            elif self.config.enable_domain_hint:
                self.classifier.run(ctx)

            # 1.6) 领域 → 定理检索（2026-09-29 新增，截图 #3+#4 老师重点关注）
            #      用户诉求：「我们要判断它是哪个领域的题目，会用到什么定理
            #      （这里就要使用 leansearch 去 mathlib 搜索对应的定理）……
            #      判断大模型或 leansearch 最后有没有找到正确的定理，以及
            #      定理对大模型的推理效果如何？」
            #      ★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
            #        **检索的执行点已不在此处** —— 改由 `sub_goal_solver.run`
            #        体首发起（2.7 内、蓝图构建之前）。本处只保留领域判定（1.5），
            #        检索所需的 `ctx.domain` / `ctx.question_type` 已就绪。
            #      **判定命中率**由离线脚本 `cli/theorem_probe.py` 对预标注表做
            #      （非运行时）；**效用**由 有/无定理 的 A/B（开关
            #      enable_theorem_hint）在 2.7 环节回答。
            _r = self._run_main_stages(ctx)
            if _r is not _SOLVE_CONTINUE:
                return _r

            # ★ 2026-10-02 hook 07：6.5 审核闸门后落盘（07_lean）—— Lean 门禁 / DAG 校验痕迹。
            _artifact_put(ctx, "07_lean", {
                "final_gate": getattr(ctx, "final_gate", None),
                "gate_rejected": getattr(ctx, "gate_rejected", None),
                "lean_reject_feedback": getattr(ctx, "lean_reject_feedback", None),
                "final_response_after_gate": getattr(ctx, "final_response", ""),
            })

            return self._run_finalize(ctx)
        except Exception as e:  # noqa: BLE001
            logger.error("Orchestrator run failed: %s", e)
            return self._fallback(ctx, problem, e)

    def _run_main_stages(self, ctx: TaskContext) -> dict:
        """主阶段序列 2.6_pre_audit → 6_format（含 3_solve 哨兵早退）。

        纯搬迁自 ``run``（2026-10-08 批8 原地拆分）；阶段名顺序在此后按任务列表
        调整过两次（见下方注释），**除这些显式删除外，其余阶段名与 record 步序逐字不变**。
        返回 ``_SOLVE_CONTINUE`` 表示「继续往下走」；其余值（含 3_solve 的早退产物）由调用方透传。
        """
        # ★ 2026-10-14（任务列表 #1「题型判断要和题意理解合并」）：
        #   **1.1_type_understanding 独立阶段已撤销** —— 题型判题（LLM + 规则
        #   交叉校验）下沉为 `2.6_pre_audit`（题意理解）的第一个子步骤，由
        #   `pre_audit_stage` 内部调用（见 agent/orchestrator__prep.py）。
        #   ⇒ 此处不再有 `_stage_start("1.1_type_understanding")`，阶段耗时表
        #     也随之少一行（题型判题的耗时并入 2.6_pre_audit）。
        #   产出字段（question_type / type_decision / type_path）与下游读取点不变。
        # ★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
        #   **1.2_theorem_hint 独立阶段亦已撤销** —— 检索改由
        #   `sub_goal_solver.run` 体首发起（2.7 内、蓝图构建之前），执行顺序变为
        #   「2.6 判题 → 2.7 检索 → 蓝图/子目标规划 → 求解/合并」。
        #   ⇒ 阶段耗时表再少一行（检索耗时并入 2.7_subgoal_main）；定理产物字段
        #     与 8 个消费点**零改动**（它们都在 2.7 及之后）。

        # 2) 2026-09-29：**快车道旁路已删除**（用户决策）。
        # 原 `_fast_path()` 用正则匹配题型（如 `\d+\s*[\+\-\*/×÷]\s*\d+`）后
        # 直接 SymPy 直解并 `return`，命中即**跳过全部 19 阶段**（含 Lean 验证、
        # 候选池、投票、审核闸门）——与"研究阶段跑全部方法"直接冲突，且正则过宽、
        # 无法关闭（审计报告 DEF-A3；云端实测 12/112 题命中）。
        # 现统一走下面的完整链路，不再有任何旁路。

        tier = prep_tier_budget(self, ctx)

        self._stage_start(ctx, "2.6_pre_audit")
        pre_audit_stage(self, ctx, tier)

        # 2026-09-29：**2.65_calc_prewarm 阶段已删除**（用户决策：
        # 「预计算没必要，且不合逻辑，删了。之后对于计算部分我们会再想办法」）。
        # 原设计在首个生成阶段前先让 LLM 预列算式并用 SymPy 算好、塞进 prompt，
        # 但用户判定其"不合逻辑"（把计算从推理链里剥离，掩盖了模型真实的
        # 计算能力）。计算部分待后续重新设计。
        # 阶段名 "2.65_calc_prewarm" 已从收尾 stage 列表中同步移除。

        self._stage_start(ctx, "2.7_subgoal_main")
        subgoal_main_path(self, ctx, tier)

        self._stage_start(ctx, "3_solve")
        # 3_solve 求解段已抽到 agent/orchestrator__solve.py::solve_stage，
        # 以「哨兵 + 解包」委托：段内唯一 run 级 return（_fallback_direct）
        # 经哨兵透传，其余路径返回 _SOLVE_CONTINUE。
        _r = solve_stage(self, ctx, tier)
        if _r is not _SOLVE_CONTINUE:
            return _r

        self._stage_start(ctx, "3.2_complete")
        complete_truncated(self, ctx, tier)

        # ★ 2026-10-17（用户定案「内化」）：3.3 的「无条件自改进」已内化为
        #   **蓝图 3 遍 + 子目标 3 遍**（blueprint_multi_gen_rounds /
        #   subgoal_collab_regen_rounds，默认均 3）⇒ 本阶段默认**跳过**
        #   （阶段名保留于 stage_audit 契约；开关 self_improve_stage 可恢复）。
        if _sw_bool("self_improve_stage"):
            self._stage_start(ctx, "3.3_improve")
            self_improve_step(self, ctx)

        self._stage_start(ctx, "3.4_collab")
        collaborative_deep_improve(self, ctx, tier)

        # 2026-09-29：**3.5_subgoal_sup 阶段已删除**（用户决策）。
        # 原逻辑「仅非 deep 档 + 候选不足时补跑子目标分解」在统一档位后
        # 条件恒假（`tier != 'deep'` 永假），且其语义与 2.7 子目标主路径
        # 完全重叠（2.7 已对所有题先跑一遍并把 `_subgoal_main_done` 置位）。
        # 该阶段已于 2026-09-29 删除；阶段名保留于审计清单（stage_audit）作历史记录。

        # ★ 2026-10-14（五道审核链第 0 道）：**4.0_normalize —— 答案规范化前置**。
        #   用户口径（原话）：「既然会用到 lean 和计算工具，就要提前把答案规范化」。
        #   纯增量：只挂 answer_norm / answer_kind（形态分流：numeric/symbolic/
        #   multi/proof/text/unparseable），**不改 c.answer** ⇒ 旧链行为零变化。
        #   五道全貌：4.0 规范化 → ①自审 → ②证伪 → ③Lean 代回 → ④计算 → ⑤投票。
        self._stage_start(ctx, "4.0_normalize")
        normalize_stage(self, ctx)

        # ★ 2026-10-14（五道审核链第 1 道）：**4.1_self_audit —— AI 自审逻辑错误**。
        #   用户口径：「先让 ai 自己审核一遍逻辑错误，然后排除错题」。
        #   只审逻辑不审数值（数值归第 4 道计算检测）；输出不可解析/LLM 故障
        #   一律保留候选（防误杀）；淘汰反馈入 audit_reject_feedback。
        self._stage_start(ctx, "4.1_self_audit")
        self_audit_stage(self, ctx)

        # ★ 2026-10-14（五道审核链第 2 道）：**4.2_falsify —— AI 对抗证伪**。
        #   用户口径：「让 ai 对还存活的答案生成对应的对抗性验证，来验证
        #   能不能证伪，证伪就判错」。机检证伪（SymPy 反例）+ LLM 对抗生成
        #   两层；跳过/故障一律保留候选（防误杀）。
        self._stage_start(ctx, "4.2_falsify")
        adversarial_gate_stage(self, ctx, tier)

        # ★ 2026-10-14（五道审核链第 3 道）：**4.3_lean_check —— Lean 逻辑判断**。
        #   用户口径：「用 lean 进行逻辑判断，就是让答案带回题目，看看逻辑对不对」。
        #   后端 = LeanGate.apply（候选级：题目+答案翻译为 Lean 命题编译判定）；
        #   部分淘汰落池、全拒标记不静默清池、不适用题豁免（沿用 3.6 口径）。
        self._stage_start(ctx, "4.3_lean_check")
        lean_backcheck_stage(self, ctx, tier)

        # ★ 2026-10-14（五道审核链第 4 道）：**4.4_math_check —— 数学工具计算检测**。
        #   用户口径：「利用数学工具进行计算检测，看看计算步骤有没有问题」。
        #   后端 = DeterministicChecker（SymPy 精确判定，可机检非自评）；
        #   fail ⇒ 淘汰、unknown ⇒ 放行、全拒 ⇒ 回退保留（宁 unknown 不误杀）。
        self._stage_start(ctx, "4.4_math_check")
        math_check_stage(self, ctx)

        # ★ 2026-10-14（五道审核链第 5 道）：**4.5_vote —— 投票**。
        #   用户口径：「大模型根据还存活的答案投票，先判断会不会是多个答案，
        #   再投票选择答案」。修完这五道后才进 3.6/4_verify 旧链（Step7 删）。
        self._stage_start(ctx, "4.5_vote")
        vote_stage(self, ctx, tier)

        # ★★ 2026-10-14（用户定案·重写通道）：**4.6_rewrite —— 池空重写**。
        #   用户原话：「不要这个东西（池空直答）！没有答案了就讲错误情况返回给
        #   大模型，让大模型找到问题出在哪里，重新构建蓝图和子目标，重写」。
        #   触发=池空；流程=错误汇总 → LLM 诊断根因 → 重建蓝图/子目标
        #   （rewrite_context 注入题目占位符内部）→ 重写 → 新候选重过五道链；
        #   轮数上限 rewrite_max_rounds（默认 2）。**取代「池空直接要答案」**。
        if getattr(ctx, "candidates", None):
            self.record(ctx, "rewrite", "4.6 重写：候选池非空，无需重写")
        else:
            self._stage_start(ctx, "4.6_rewrite")
            rewrite_stage(self, ctx, tier)

        # ★★ 2026-10-14（五道审核链定案 · 用户决策「旧链一步到位直接删」）：
        #   以下旧链阶段已**整段删除**，职责去向：
        #     · 3.6_audit_filter（Lean 双通道 + AuditGate 串行）
        #         → 拆分吸收进 **4.3_lean_check**（LeanGate.apply）与
        #           **4.4_math_check**（DeterministicChecker Level0）；
        #     · 4_verify（验证+投票一体，耗时单点 51.3%）→ **4.5_vote**（LLM 投票）；
        #     · 4.5_oracle（同源自评复核）→ 删（用户口径：其余都删）；
        #     · 4.6_adv（只探最佳簇一次）→ 升级为 **4.2_falsify**（逐候选证伪）；
        #     · 4.7_lean（投票前 Lean，曾为实验开关）→ 并入 **4.3_lean_check**；
        #     · 5_revise_or_fallback / 5.5_low_conf → 删（用户决策②：
        #       重解回环删除；★ 后续用户再定案：池空走 **4.6_rewrite**
        #       「错误诊断 → 重建蓝图/子目标 → 重写」，而非直答）；
        #     · 6.5_audit_gate（答案级 Lean/AuditGate 终审 + rework）→ 删
        #       （Lean 判断统一在投票前的 4.3）。
        #       其内 «calc_inconsistent 判据在当前配置下不可达» 的历史留痕
        #       （`<calc>` 板块删除后所留）随段体一并移除（机制已不在主链）。
        #   阶段名保留于 stage_audit 审计清单作历史记录（阶段名是稳定契约）。

        self._stage_start(ctx, "6_format")
        format_stage(self, ctx)
        return _SOLVE_CONTINUE

    def _run_finalize(self, ctx: TaskContext) -> dict:
        """收尾：统一 stage_stop + 输出装配（候选/裁决/簇）+ hook08 + 终返回字典。

        纯搬迁自 ``run`` 尾部（2026-10-08 批8 原地拆分），语句逐字不变。
        """
        # 阶段耗时收尾（2026-09-03）：统一 stop 全部阶段（start 在阶段开始）
        # ★ 2026-10-02 移除 `1_classify` / `2.5_difficulty` / `0_paper_pacer`（阶段均已删）；
        #   此前报告仍含这些阶段名（留痕，阶段名是契约）。
        # ★ 2026-10-14（任务列表 #1）移除 `1.1_type_understanding`：题型判题已并入
        #   `2.6_pre_audit`（题意理解），不再是独立阶段 ⇒ 无需单独收尾。
        # ★ 2026-10-14（任务列表 #2）移除 `1.2_theorem_hint`：检索已下移到
        #   `2.7_subgoal_main`（其耗时并入该阶段）⇒ 无需单独收尾。
        for _stg in ("2.6_pre_audit",
                     "2.7_subgoal_main","3_solve","3.2_complete","3.3_improve",
                     "3.4_collab",
                     "4.0_normalize",
                     "4.1_self_audit",
                     "4.2_falsify",
                     "4.3_lean_check",
                     "4.4_math_check",
                     "4.5_vote",
                     "4.6_rewrite",
                     "6_format"):
            self._stage_stop(ctx, _stg)

        # 构建返回
        candidates_out = [
            {"id": c.id, "answer": c.answer,
             "reasoning": c.reasoning, "revised": c.revised}
            for c in ctx.candidates
        ]
        verdicts_out = [
            {"id": v.id, "answer": v.answer,
             "confidence": v.confidence,
             "correct_votes": v.correct_votes,
             "total_votes": v.total_votes,
             "feedback": v.feedback}
            for v in (ctx.verdicts or [])
        ]
        cluster_out = None
        bc = getattr(ctx, '_best_cluster', None)
        if bc:
            cluster_out = {
                "answer_norm": bc.answer_norm,
                "size": bc.size,
                "confidence": bc.confidence,
                "candidate_ids": bc.candidate_ids,
            }
        # 最终答案后处理（P2 强制数值化 + P5 客观题自检）——统一出口挂载，
        # 每项带独立开关与埋点，便于单次评测逐项审查（2026-09-11 用户指示）
        # ★ 2026-10-04（D25）：**已上移到 6.5 闸门之前**（见 `6_format` 之后那段），
        #   此处刻意**不再调用** —— 否则会二次执行，并重新落回
        #   "闸门验 A、实际提交 B"。早退路径（超时兜底 / `_fallback_direct`）
        #   仍在各自出口自行调用，不受影响。

        # ★ 2026-10-02 hook 08：run() 正常返回前落盘终答（08_final）。
        _artifact_put(ctx, "08_final", {
            "final_response": ctx.final_response or "",
            "n_candidates": len(getattr(ctx, "candidates", None) or []),
            "n_verdicts": len(getattr(ctx, "verdicts", None) or []),
        })

        return safe_json_serialize({
            "final_response": ctx.final_response or "",
            "trace": ctx.trace,
            "candidates": candidates_out,
            "verdicts": verdicts_out,
            "cluster": cluster_out,
            # AI 实际检索/引用过的 Mathlib 定理（#1/#2 证据链）
            "used_theorems": list(ctx.used_theorems or []),
            # 检索/命中/编译通过统计（回应"调用频繁但定理不多"）
            "mathlib_usage_stats": {
                **(ctx.mathlib_usage_stats or {}),
                "distinct_theorems": len(ctx.used_theorems or []),
                # ★ 2026-09-23：「引用即采用」采用率（回答"定理有没有真用上"）
                "adoption": self._mathlib_adoption(ctx),
            },
            # ---- 逐步归因诊断（2026-09-02 用户要求：错题要能定位到环节）----
            # 统一由 _collect_diag 构建（全 getattr 兜底：提前 return / 异常路径也覆盖）
            "diag": self._collect_diag(ctx),
        })

    # ----------------------------------------------------------
    # 2026-09-29：快车道（_fast_path / _try_sympy_solve）已按用户要求**整体删除**。
    #
    # 原设计：正则匹配题型（如 `\d+\s*[\+\-\*/×÷]\s*\d+`）后直接 SymPy 直解并
    # `return`，命中即跳过全部后续阶段（Lean 验证 / 候选池 / 投票 / 审核闸门）。
    # 删除理由：
    #   ① 与"研究阶段用全部方法测试大模型"直接冲突 —— 它省掉的恰是待研究的环节；
    #   ② 首条正则过宽（任何含两个数字与运算符的题干都命中），且无开关可关
    #      （审计报告 DEF-A3）；云端实测 12/112 题命中；
    #   ③ 用户决策"统一一个档位、统一答题格式"，不允许任何旁路。
    # 保留说明：`utils.sympy_tools` 的 import 仍保留（其它模块可能引用），
    # 不再有未定义名风险。
    # ----------------------------------------------------------

    # ★★ 2026-10-14（五道审核链定案 · 用户决策「旧链一步到位直接删」）：
    #   revise 回环簇（_review_bug_feedback / _deep_revise_loop[_inner] /
    #   _build_revise_feedback / _revise_rounds / _update_accept_gate）与
    #   oracle/对抗/证伪簇（_enhance_window_ok / _adversarial_probe /
    #   _falsify_candidates / _falsify_rejects / _oracle_review_best）已随
    #   5_revise_or_fallback / 4.5_oracle / 5.5_low_conf 三阶段一并**整段删除**
    #   （用户决策②：重解循环全删，全灭走 emergency 直答兜底）。
    #   对应模块 agent/orchestrator__revise.py 与 agent/orchestrator__oracle.py
    #   已删除；机检证伪能力由 4.2_falsify（agent/adversarial_gate.py）承接。

    def _mathlib_adoption(self, ctx: TaskContext) -> dict:
        return _orch_select._mathlib_adoption(self, ctx)

    def _verdicts_from_ver_result(self, ver_result: dict, candidates: list = None) -> list:
        return _orch_select._verdicts_from_ver_result(self, ver_result, candidates)

    def _has_usable_candidate(self, ctx: TaskContext) -> bool:
        return _orch_select._has_usable_candidate(self, ctx)

    def _maybe_final_select(self, ctx: TaskContext) -> "str | None":
        return _orch_select._maybe_final_select(self, ctx)

    def _pick_best_from_candidates(self, ctx: TaskContext) -> str:
        return _orch_select._pick_best_from_candidates(self, ctx)

    @staticmethod
    def _set_final_response(ctx: TaskContext, ans: str, reason: str = "") -> bool:
        return _orch_select._set_final_response(ctx, ans, reason)

    def _collect_diag(self, ctx: TaskContext) -> dict:
        return _orch_select._collect_diag(self, ctx)

    # ★ 2026-10-15（用户定案「删直答兜底」）：
    #   `_DIRECT_SYS` / `_emergency_direct_solve` / `_fallback_direct` 已随直答链
    #   删除；新增「错误信息收集 + 报错」两个方法（无候选/超时/异常时用）。
    def _collect_solve_error(self, ctx: TaskContext, reason: str, exc=None) -> dict:
        return _orch_fallback._collect_solve_error(self, ctx, reason, exc)

    def _raise_solve_error(self, ctx: TaskContext, reason: str, exc=None):
        return _orch_fallback._raise_solve_error(self, ctx, reason, exc)

    def _fallback(self, ctx: TaskContext, problem: str, exc: Exception) -> dict:
        return _orch_fallback._fallback(self, ctx, problem, exc)
