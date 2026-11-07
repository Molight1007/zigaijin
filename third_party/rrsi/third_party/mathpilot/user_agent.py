from __future__ import annotations

"""
MathPilot — 数学推理智能体（多智能体版）
==========================================================

项目：多智能体协作的数学推理智能体，以官方 112 题为基准做本地/云端评测。
当前阶段：**赛后研究期** —— 不限时、不限 token、唯一目标是解题正确率上限；
主模型为 GLM-4.7-Flash / DeepSeek，仓库中不再保留 Intern-S（书生）特化逻辑。

架构（多智能体协作，简化版 v2）：
    题型识别 → 通用求解 → 过程校验 → 答案规范化
    由 Orchestrator 通过共享黑板（TaskContext）调度，借鉴 ss-main 的简洁流水线。
    不做复杂回环，每道题 LLM 调用控制在 7 次以内。

硬性接口规范（不可修改）：
    agent = ReasoningAgent(client=llm_client)
    result = agent.solve(problem, metadata)  # -> dict

注意事项：
    - 禁止硬编码 API Key，client 由调用方（评测引擎）注入
    - 禁止使用绝对路径，所有文件读取使用相对路径
    - solve 返回的字典必须支持 JSON 序列化
    - final_response 不可为空

调用契约防御（v2.3）：
    - sys.path 自举：无论以何种 cwd 运行，都能找到本包
    - solve(problem, metadata=None)：metadata 缺失时不崩溃
    - 核心模块初始化失败 → 抛异常（2026-10-15 用户定案：不做降级直答，
      失败即暴露，避免"agent 链坏了仍用直答兜底掩盖错误"）
    - _validate_output：返回前强制校验 final_response 非空 + 可 JSON 序列化

★ 2026-10-15：原「平台兼容层」（`__call__` / `run` 两个别名，用于平台以
  `agent(problem, metadata)` 或 `agent.run(...)` 形态调用）已删除 —— 已不打平台，
  统一走 `solve()`。本地评测（`run_eval__engine.py::ReasoningAgent`）不受影响。
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# sys.path 自举：保证从任何 cwd 都能 import 到本包
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger("MathPilot")

# ★ 2026-10-15（用户定案「不要降级兜底」）：
#   原 `_normalize_chat_response`（惰性委托）与 `_subgoal_answer_candidate`（子目标
#   结果形态判据）均**只为降级直答/抢救链服务**（`_fallback_solve` / `_rescue_answer`），
#   该链已整段删除 ⇒ 二者成为死代码，随之删除。agent/base 的响应归一化实现不受影响。

# ============================================================
# 配置（AgentConfig 已搬到 user_agent__config.py，2026-10-08 拆分；此处 re-export 保契约，同一对象）
# ============================================================
from user_agent__config import AgentConfig  # noqa: F401


# ============================================================
# 响应归一化工具（P0-1 契约防线核心）
# ============================================================


# ============================================================
# kwargs 覆盖白名单（2026-10-08 从 ReasoningAgent.__init__ 提取为模块级常量）
#   ★ 不在本白名单的键经 CLI/kwargs 传入会被 ReasoningAgent 静默丢弃；
#     新增配置项必须同时登记 AgentConfig 与本白名单（本项目已多次踩坑）。
# ============================================================
_CONFIG_OVERRIDE_KEYS = (
    "policy_sample_times", "policy_temperature", "policy_max_tokens",
    "verifier_voting_times", "verifier_temperature",
    # 2026-09-16 投票方差（候选分歧 → 提高票数 + 非零温度）
    "verifier_diversify_enabled", "verifier_disagreement_votes",
    "verifier_disagreement_temperature",
    # 2026-09-15：带推理的最终复核（验证器偏松的止血阀）
    "verifier_deep_final_enabled", "verifier_deep_final_min_remaining",
    "verifier_vote_reasoning",
    "verifier_deep_review_max_tokens",
    "verifier_deep_review_min_chars",
    "verifier_deep_review_temperature",
    "enable_domain_hint", "enable_question_type",
    # ★ 2026-09-16 审计修复：以下两个键此前**只有 getattr 兜底、未声明也未列白名单**
    #   ⇒ 注释承诺的"回退/A-B 开关"实际无效（典型假开关）。
    #   `enable_question_type_hint`：solver 注释称"仍可全题型强制开启做 A/B"；
    #   `verify_reserve_seconds`：orchestrator 注释称"可覆盖生成侧预留"。
    "enable_question_type_hint", "verify_reserve_seconds",
    "enable_web_search",
    "self_improve_stage", "cross_model_falsify", "cross_model_timeout", "subgoal_regen_self_review", "rewrite_on_low_conf", "rewrite_conf_threshold", "subgoal_theorem_search", "attention_focus_enabled", "smt_falsify", "lean_hammer", "answer_shape_strict", "enable_lean_tools",  # 2026-09-16 联网搜索工具（默认 False）
    "lean_dedup_by_answer",  # 2026-09-17 Lean 同答案去重缓存（默认 False）
    # ★★ 2026-09-16 审计修复（**关键断链**）：以下三键此前
    #   **有 argparse、有 AgentConfig 声明，却不在白名单** ⇒
    #   `--max_subgoals 13` 之类传进来会被 `__init__` **静默丢弃**。
    #   实测证据：日志反复出现「Blueprint 子目标数超上限，截断 13 个」
    #   —— 卡在默认值（standard 6 / deep 12），CLI 根本调不动。
    "max_subgoals", "max_subgoals_by_tier",
    # 2026-09-29：`enable_calc_prewarm` 已随预计算机制删除，从白名单移除。
    # 2026-10-01：`<calc>` 板块删除 ⇒ `tool_calc_enabled` / `enable_calc_tool`
    #   / `calc_mandatory` / `calc_hard_only` / `answer_selfcheck_enabled`
    #   / `subgoal_calc_router` 六键一并从白名单移除。
    "subgoal_reuse_done",  # 2026-09-12 已完成子目标跨 run() 复用
    "subgoal_collab_mode", "subgoal_collab_regen_rounds",
    "subgoal_adaptive_recover",  # 2026-09-12 子目标失败二选一（重做/重规划）
    # 2026-09-12 定型前审核修复：以下 5 个键此前**不在白名单** → 尽管
    # run_eval.py 有对应 argparse，kwargs 覆盖会在 __init__ 里被静默
    # 丢弃（即文档中写的 `--symbolic_solve_adopt false`、
    # `--answer_selfcheck_enabled false` 等回退路径**实际无效**）。
    "symbolic_crosscheck_enabled",
    "symbolic_solve_enabled", "symbolic_solve_feedback",
    "symbolic_solve_adopt",
    "max_total_calls", "max_time_per_question",
    "max_total_time_seconds", "max_tokens_cap",
    "use_scoring",
    "use_rubric", "use_challenge",  # 2026-09-06 sq 移植（A/B 开关）
    "max_revise_rounds", "max_workers",
    "require_answer_in_pool",
    # ★ 2026-10-07（问题 2.4 / 2.9）：入池门槛 v2 + `<check>` 协议守卫。
    #   必须在此白名单，否则 CLI/kwargs 传入会被静默丢弃（本项目多次踩坑）。
    "pool_gate_v2", "check_protocol_guard",
    "strict_all_rejected",
    "use_proof_channel", "use_lemma_accumulation",
    "lemma_domains",
    "max_answer_tokens", "revise_sample_times", "revise_min_candidates",
    # 2026-10-01 开关注册制：29 个已登记开关进入白名单（可经 CLI/kwargs 传入）
    #（2026-10-02 新增 artifact_store_enabled / minimal_mode / enable_subgoal_findings
    #  / enable_final_answer_selection）
    "artifact_store_enabled",  # 2026-10-02 中间结果存储层总开关
    "answer_form_gate",
    "deterministic_timeout_sec",
    "eval_alpha_equiv",
    "eval_split_cn",
    "expr_eval_grounding_guard",
    "enable_subgoal_findings",  # 2026-10-02 阶段一：主求解注入子目标结论
    "enable_final_answer_selection",  # 2026-10-02 阶段二-2：终答五层选择
    "final_answer_single_source",  # 2026-10-07：决策记录与最终产出单一来源
    "final_select_verify_aware",  # 2026-10-07：选择机制识别未验证候选
    "lean_gate_parallel",
    "lean_gate_strict_unknown",
    "lean_crosscheck_as_unknown",  # 2026-10-07 语义分离（cross_check→unknown）
    "lean_mcp_autolake",
    "lean_mcp_goal_loc",
    "lean_mcp_hover_check",
    "lean_mcp_multi_attempt",
    "lean_mcp_timeout_floor",
    "lean_mcp_verify_axioms",
    "lean_verify",
    "lean_xcheck_numeric",
    "llm_retry_on_timeout",
    "llm_timeout",
    "minimal_mode",  # 2026-10-02 最小基础模式（只留主链）
    "numericize_final",
    "objective_itemwise_priority",
    "objective_majority_vote",
    "objective_selfcheck",
    "self_improve_keep_original",
    "self_improve_objective_skip",
    "theorem_hint",
    "toolcall_text_fallback",
    "use_blueprint", "use_blueprint_dag",
    # 2026-09-15：DAG 依赖边开关（修复 depends_on 结构性恒空）
    "blueprint_deps_enabled",
    # 2026-09-20：OR 节点展开策略开关（默认 False = 保持历史行为）
    "blueprint_or_expand_all",
    # 2026-10-15：子目标承接 + 工具指派（用户「证明就用 lean」）
    "blueprint_node_contract", "blueprint_tool_exec", "blueprint_lean_max_per_q",
    # 2026-10-15：蓝图叶子级形式化下传（用户「把 lean 内化到蓝图构建里面」）
    "blueprint_lean_sig",
    # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：原
    #   `enable_dag_replan` / `dag_replan_gate` / `enable_skeleton_review` /
    #   `skeleton_review_max_rounds` / `dag_replan_lean_check` / `dag_lean_max_nodes`
    #   及孤儿键 `dag_replan_max_rounds` 已随三个默认关闭的门一并撤除。
    #   `dag_review_reject_count` 保留（DagReviewerAgent 仍在读）。
    "dag_review_reject_count",
    "enable_preverify_sketch_audit",
    # 2026-10-12 批次 1：题意理解（★ 2026-10-14 收敛：只留总开关）
    "enable_understanding_chain",
    # 2026-10-12 批次 2：蓝图多次生成 + 自审
    "blueprint_multi_gen_rounds", "blueprint_self_review",
    "blueprint_self_review_reject_ratio",
    "subgoal_inputs_validation",
    "pool_empty_clear",
    "enable_theorem_query_llm",
    "subgoal_self_improve_rounds",
    # 2026-10-15（用户「2 与 3 之间建一个环」）：求解期动态倒推 + 递归层数上限
    "subgoal_dynamic_backward", "subgoal_backward_max_depth",
    "enable_type_llm",
    "theorem_hint_block_max_chars",
    "theorem_hint_top_k_by_stage",
    # 2026-10-14 五道审核链（0 规范化/1 自审/2 对抗证伪/3 Lean 代回/4 计算检测/5 投票 + 重写通道）
    "five_gate_normalize", "five_gate_self_audit", "five_gate_falsify",
    "five_gate_subgoal_lean",
    "five_gate_lean_check", "five_gate_lean_check_tool", "five_gate_math_check", "five_gate_vote",
    "five_gate_grounding",
    "five_gate_audit_retry",
    "five_gate_self_audit_votes",
    "five_gate_falsify_votes",
    "subgoal_failed_retry_rounds",
    "blueprint_subtree_regen",
    "five_gate_verify_tiebreak",
    "lean_failure_kind_split",
    "five_gate_math_check_calcverify",
    "five_gate_rewrite",
    # 2026-10-20：审核侧义务账本总开关（4.1 自审 + 4.2 证伪共用）
    "audit_obligation_ledger",
    # 2026-09-29：DifficultyRouter 已删除 ⇒ 其开关（enable_difficulty_router /
    # enable_llm_difficulty / algebra_force_deep）不再有任何效果，一并从白名单移除。
    # tier_* 六键保留（仍被 solver / orchestrator 以 .get(tier) 读取，见
    # AgentConfig.__post_init__ 的"统一档位"说明）。
    "tier_sample_times", "tier_temperatures", "tier_voting_times",
    "tier_max_completions", "tier_max_calls", "tier_budget",
    # 2026-09-29：PaperPacer 已删除 ⇒ paper_* 三键（target_time / min_soft /
    # total_questions）与 deep_quota_ratio 不再有任何读取点，从白名单移除。
    "deep_use_sub_goal", "deep_revise_rounds", "deep_use_playoff",
    "enable_collaborative_deep", "collab_max_rounds",
    # 子目标阶段预算（P1 按档拆分）
    "subgoal_stage_budget_sec", "subgoal_stage_budget_sec_std",
    # 子目标上下文注入模式（老师 9/6：deps 最小依赖 | all 全量，A/B）
    "subgoal_ctx_mode",
    # 子目标级 0-LLM lean 代码片编译校验（S1-lite L1，deep 档+lean 可用）
    "enable_subgoal_lean_check",
    # 子目标交叉核对（老师 9/8 建议3：A 确定性冲突闸 0-LLM / B LLM 交叉核对轮）
    "subgoal_conflict_gate", "subgoal_crosscheck_llm",
    # L2 子目标数值/代数断言 Lean 验证（2026-09-08 去门后新钩子）
    "enable_numeric_lean_verify", "lean_numeric_max_per_q",
    # LeanSearch 引理检索（2026-09-15 重启；★ 必须在此白名单，否则 CLI 传参
    # 会被静默丢弃 —— 本项目已因漏加白名单踩过多次，见上方 720/733 行注释）
    "use_leansearch", "leansearch_top_k", "leansearch_max_calls_per_q",
    "leansearch_inject_verifier",
    # 时间预算（2026-08-28 新增；2026-09-29 删除 deep_quota_ratio 键）
    "critical_tail_seconds", "deep_critical_tail_seconds",
    # L1 验证优先（2026-08-31）
    # 结构化 bug report 反馈
    "use_bug_report_feedback",
    # Step 2 无条件自改进（IMO2025 论文）
    "enable_self_improve", "self_improve_max", "improve_min_remaining",
    # 2026-09-29：真无条件 + 遍数可配（截图 #7 A/B 抓手）
    "self_improve_rounds", "self_improve_conditional",
    # 2026-09-30：revise 后无条件自改进（截图 #9 A/B 抓手）
    #   ★ 必须在此白名单，否则 `--self_improve_after_revise` 会被静默
    #     丢弃（本项目已因漏加白名单踩过多次）。
    "self_improve_after_revise",
    # 2026-09-30：择优可信度门槛（截图 #9）
    "pick_confidence_floor",
    # 2026-09-30：子目标多 agent 求解（截图 #8 A/B 抓手）
    "enable_subgoal_multi_agent", "subgoal_agents_n",
    # 2026-09-29：领域→定理检索（截图 #3+#4）
    "enable_theorem_hint", "theorem_hint_max_queries",
    "theorem_hint_backend",
    # 易错点记忆注入（2026-09-06 A 档轻量经验）
    "enable_error_lessons",
    # Step 4 bug report 复核
    # 2026-10-06：Step 4 复核的修复（附加而非覆盖 / 产物长度上限 / 越权清洗）
    # 对抗式验证（#16）
    "enable_adversarial_verify", "adversarial_tiers",
    "adversarial_min_confidence", "adversarial_max_tokens",
    "adversarial_max_reasoning",
    # 检测链（2026-09-06 去 Lean 化；顶替原 Lean 硬验证开关）
    # Lean 双通道（2026-09-06 晚恢复：lean-toolchain 离线可用后接入）
    "enable_lean_verify", "enable_lean_preverify",     "preverify_max_rounds", "preverify_timeout",
    "lean_gate_all_proofs", "lean_gate_nonproof", "lean_gate_strict",
    "lean_gate_nonproof_deep_only",
    "lean_gate_unknown_stop",
    "lean_timeout", "lean_backend", "lean_executable", "lean_project_dir",
    "theorem_memory_enable",
    # lemma 记忆
    "lemma_storage_path",
    # ★★ 2026-10-04 开关注册制补齐：以下 4 键此前**不在白名单** ⇒
    #   CLI/kwargs 传入会被 `__init__` **静默丢弃**（本项目已因漏加白名单
    #   踩过多次，见上方注释）。补齐后可经 env / CLI / kwargs 生效。
    #   （另 5 键 `enable_subgoal_multi_agent` / `use_leansearch` /
    #     `dag_replan_gate` / `enable_skeleton_review` / `enable_question_type`
    #     原本已在白名单，未重复添加。）
    # 2026-10-06：子目标交付重写（结论契约 / 客观判定）—— 必须在此白名单，
    # 否则 CLI/kwargs 传入会被静默丢弃（本项目已因漏加白名单踩过多次）。
    "subgoal_conclusion_contract", "subgoal_objective_judge",
    "k11_mask_statement",     # K11 注入点守卫（消费方 sub_goal_solver）
    "candidate_pool_limit",   # 候选池上限（0 = 不限）
    "sub_goal_oracle_check",  # 子目标逐步 oracle 校验（默认关）
    "falsify_max_per_q",      # 单题最多证伪次数
    "phase_budget_enabled",   # 阶段时间预算总开关
    # ★ 2026-10-07 开关注册制（fix-audit-gate「解答题判别力」四件套）
    "deterministic_solution_check",   # 解答题确定性判别重写（默认 True）
    "audit_gate_rubric_nonproof",     # 解答题也跑 Level2 rubric（默认 False）
    # ★ 2026-10-07 开关注册制（fix-gate-timing-diag：AcceptGate 生效 + 计时归属）
    # ★ 2026-10-07 开关注册制补齐（fix-switch-debt）：
    #   ① 6 个账外 gate 型 env 的配置字段（默认值见 switch_registry 各条）；
    #   ★ 2026-10-15：emergency_direct_max_tokens / emergency_answer_max_tokens /
    #     final_answer_min_sec 已随直答兜底删除。
    "deep_max_rework", "objective_itemwise_plan",
    #   ② 9 条此前**漏加白名单**的键 ⇒ 相关开关的 CLI/kwargs **回退路径在
    #      代码层面不存在**（传参被 `__init__` 静默丢弃；本项目已因漏加
    #      白名单踩过多次，见上方多处注释）。补齐后可经 CLI/kwargs 生效。
    #      （另 `falsify_max_per_q` / `phase_budget_enabled` 两条原已在白名单，
    #        未重复添加。）
    "accept_confidence", "audit_gate_skip_non_proof",
    "enable_answer_falsifier", "enable_deterministic",
    "enable_subgoal_main_path", "inject_mathlib_to_solver",
    "objective_tactic_enabled",
    "symbolic_max_tokens", "symbolic_solve_max_tokens",
)


# ============================================================
# ReasoningAgent 装配门面（薄壳；本地评测引擎的入口）
#
# ⚠ 本类是**本地评测的唯一装配点**，不是"平台专用代码"：
#    · `run_eval__engine.py::ReasoningAgent(self.llm_client, **overrides)` 依赖它；
#    · `_finalize_backend` 里 `bind_config(self.config)` 是全仓**唯一**生产调用点，
#      删掉 ⇒ CLI/kwargs 传入的配置全部静默失效（2026-10-02 修过的 bug 会复活）；
#    · `_CONFIG_OVERRIDE_KEYS` 是四表之一的覆盖白名单宿主。
# ============================================================
class ReasoningAgent:
    """
    MathPilot 数学智能体主类（装配门面；唯一调用入口是 `solve()`）。

    solve() 的内部实现已委托给多智能体 Orchestrator，本类仅负责：
    - 接收调用方注入的 client；
    - 组装配置（kwargs 覆盖 + 绑定进开关注册表 + 最小模式）；
    - 透传 solve 调用并维持返回格式不变；
    - 契约防御：核心模块初始化失败即抛异常（2026-10-15 定案，不降级直答）。
    """

    def __init__(self, client, *args, **kwargs):
        self.client = client
        self.config = AgentConfig()

        # 允许通过 kwargs 覆盖配置（向后兼容 run_eval.py 的传参）
        for key in _CONFIG_OVERRIDE_KEYS:
            if key in kwargs:
                setattr(self.config, key, kwargs[key])
        # 覆盖 dict 型配置后需确保各档键完整
        self.config.__post_init__()

        self._finalize_backend(client)

    def _finalize_backend(self, client):
        """构造收尾：绑定配置 / 最小模式 / 初始化 orchestrator（核心模块不可用时降级）。"""
        # ★ 2026-10-02 修既有 bug（配置覆盖静默失效，team-lead 独立验收查出）：
        #   把 AgentConfig 的**当前取值**绑定进开关注册表。
        #   此前 `bind_config` **全仓零调用** ⇒ `_overrides` 恒空 ⇒ 注册表读取链
        #   退化成「env > 默认值」，中间那层「配置覆盖」是死的 —— 白名单允许
        #   CLI/kwargs 传入，但**传了不生效、也不报错**（静默失败）。
        #   必须放在 `__post_init__()` 之后（配置已定型）、任何业务逻辑之前。
        #   平台扁平导入下 `agent` 包不可用时优雅跳过（绝不影响构造）。
        _bind_cfg = None
        try:
            from agent.switch_registry import bind_config as _bind_cfg
        except ImportError:
            try:
                from switch_registry import bind_config as _bind_cfg
            except ImportError:  # pragma: no cover  两个路径都不可用 → 跳过
                _bind_cfg = None
        if _bind_cfg is not None:
            try:
                _bind_cfg(self.config)
            except Exception as _bc:  # noqa: BLE001  绑定失败不得影响构造
                logger.warning("bind_config 失败（忽略，按 env/默认跑）: %s", _bc)

        # ★ 2026-10-02 最小基础模式（MINIMAL_MODE，一键只留主链）：
        #   开启时关闭与「推导」无关的旁支口径（联网搜索 / 评分投票 / 证明旁路通道
        #   / rubric / 挑战分支）。读取走开关注册制（env MINIMAL_MODE > 本字段 > 默认 False）。
        #   ⚠ 此处置为 False 的字段，其 env（如 USE_SCORING）仍按注册表最高优先级生效。
        try:
            from agent.switch_registry import (
                bind_field as _sw_bind, get_bool as _sw_get_bool)
        except ImportError:
            from switch_registry import (
                bind_field as _sw_bind, get_bool as _sw_get_bool)
        try:
            _sw_bind("minimal_mode", getattr(self.config, "minimal_mode", None))
            if _sw_get_bool("minimal_mode"):
                _minimal_off = ("enable_web_search", "use_scoring",
                                "use_proof_channel", "use_challenge",
                                "use_rubric")
                for _f in _minimal_off:
                    setattr(self.config, _f, False)
                logger.warning(
                    "MINIMAL_MODE 开启：只留主链（题意理解→子目标→求解→形式化验证），"
                    "已关闭 %s", " / ".join(_minimal_off))
        except Exception as _me:  # noqa: BLE001
            logger.warning("MINIMAL_MODE 处理异常（忽略，按常规模式跑）: %s", _me)

        # ★ 2026-10-15（用户定案）：核心模块初始化失败即抛异常，不做降级直答。
        #   此前「失败 → 置 None → solve 走内置直答后端」会把 agent 链损坏
        #   （import 失败 / Orchestrator 构造失败）静默吞掉、用直答兜底掩盖错误。
        #   正确的启动方式 = 显式构造，失败即报错（调用方/评测层据此暴露问题）。
        from agent.orchestrator import Orchestrator
        self.orchestrator = Orchestrator(client, self.config)

        logger.info(
            "MathPilot ReasoningAgent (v2 simplified) initialized: "
            "samples=%d, votes=%d, domain_hint=%s, "
            "budget=%d, max_tokens_cap=%d, scoring=%s",
            self.config.policy_sample_times,
            self.config.verifier_voting_times,
            self.config.enable_domain_hint,
            self.config.max_total_calls,
            self.config.max_tokens_cap,
            self.config.use_scoring,
        )

    # ★ 2026-10-15（用户定案「不要降级兜底」）：以下整段已删除——
    #   `_fallback_solve`（内置直答后端）、`_DEGRADED_ANSWER_RE` / `_is_degraded_answer`
    #   （降级答案判据）、`_rescue_answer`（最终抢救直答）三件套。
    #   删除理由：答不出就绕开整条流水线直答、或从残料里"抢救"答案，等于用
    #   兜底掩盖模型/流水线的真实失败（「必须有答案」只隐藏错误）。现在的口径：
    #   流水线交什么就是什么，占位符/空答案如实暴露，由评测判 0，不再抢救。

    def _validate_output(self, result: dict) -> dict:
        """返回前强制校验：final_response 非空且可 JSON 序列化。"""
        fr = result.get("final_response", "")
        if not isinstance(fr, str) or not fr.strip():
            result["final_response"] = "未给出有效解答。"
        # 保证 JSON 可序列化
        if not isinstance(result.get("trace"), list):
            result["trace"] = []
        return result

    def solve(self, problem: str, metadata: dict = None, *args, **kwargs) -> dict:
        """
        求解单道数学题（唯一调用入口；原 `__call__` / `run` 平台别名已删）。

        参数:
            problem: 原始数学题目文本
            metadata: 题目元数据（可缺省，含 idx 字段）

        返回:
            {"final_response": str, "trace": list[dict]}
        """
        if metadata is None:
            metadata = {}
        if problem is None or not str(problem).strip():
            return self._validate_output({
                "final_response": "题目为空。",
                "trace": [{"stage": "input", "note": "empty problem"}],
            })

        # 核心流水线直接运行（2026-10-15 用户定案：异常向上抛，不做降级直答）。
        # 此前 `orchestrator.run` 抛异常会 catch 后走直答后端、final_response 为
        # 占位符会触发 `_rescue_answer` 抢救——这些"必须有答案"兜底只会隐藏
        # 流水线/模型的真实失败。现在流水线交什么就是什么，失败即暴露。
        result = self.orchestrator.run(problem, metadata)
        if not isinstance(result, dict):
            raise RuntimeError(
                "Orchestrator.run 返回异常类型（应返回 dict）: %s"
                % type(result).__name__)
        return self._validate_output(result)
