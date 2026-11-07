# -*- coding: utf-8 -*-
"""可调上限「配置 vs 实测」记录（2026-09-21）。

诉求（用户原话）：
    「对于 leansearch 到底要找多少 mathlib 的定理，检测打回要搞多少次，
      无条件重做要搞多少次，子目标要设立多少，像这种数据你都要找到并记录
      （不止我说的这几个）我们要测试出最合理的数据。」

问题：这些数字此前散落在 trace 与各 `ctx` 计数器里，**没有统一口径**的
「配置上限 / 实际用了多少 / 有没有撞顶」三元组 ⇒ 调参只能凭直觉
（`max_subgoals=6` / `leansearch_top_k=5` / `lean_gate_unknown_stop=2`
这些值都是当初拍的，没有一条来自实测分布）。

本模块把「上限」与「实测」放进同一张表，随每题的 diag 一起落盘::

    diag["param_usage"] = {
        "schema": "param_usage/1",
        "tier": "deep",
        "items": {
            "leansearch_max_calls_per_q": {
                "cap": 2, "used": 2, "capped": True,
                "unit": "次", "src": "ctx._leansearch_calls",
                "note": "单题 LeanSearch 检索调用次数上限"},
            ...,
            "max_answer_tokens": {
                "cap": 65536, "used": None, "capped": None,
                "unit": "token", "src": "",
                "note": "token 口径不在 ctx 上；由 LLM 截断台账统计（见 utils.llm_client）"},
        },
        "counters": {"subgoals": 12, "leansearch_hits": 14, "distinct_theorems": 9, ...},
        "gaps": ["max_answer_tokens", ...],   # 有上限但当前无实测口径
        "approx": ["policy_sample_times", ...],  # 只有代理指标，判不了是否撞顶
    }

设计原则
    1. **只读**：不写 ctx、不改任何行为，纯派生快照。
    2. **不抛异常**：全程 try/except，任一项取不到就保留 cap、used=None。
       埋点把主流程搞挂的代价，远大于少一个数字。
    3. **`used=None` 的含义是「当前没有埋点，无法判断」，不是 0**。
       这些键会同时进 ``gaps``，即"想调它就得先补埋点"。
    4. **声明式**：新增一个可调上限 = 在 :data:`_SPEC` 里加一行。

⚠ 本模块**不 import 任何 agent 子模块**（只有 getattr 鸭子取数），
  以免引入循环导入；这也是它能被单测直接 import 的原因。
"""
from __future__ import annotations

SCHEMA = "param_usage/1"


# 2026-10-08 批 P1：取数工具 + 实测值提取器已搬到 param_usage__extract.py。
# 原位 re-export 保外部导入与 `_SPEC` 内 lambda 裸引用契约。⚠ 勿按"未使用"清理。
from .param_usage__extract import (  # noqa: F401
    _int, _trace, _gate_events, _cfg_get, _cap, _cap_by_tier, _cap_gate_rework,
    _u_leansearch_calls, _u_leansearch_topk, _u_subgoals, _u_gate_tried,
    _u_unknown_streak, _u_revise_round, _u_preverify_rounds,
    _u_total_calls, _u_candidates, _u_verdicts, _u_deep_review)


# ======================================================================
# 规范表：一个可调上限 = 一行
#   (键名, cap 取值器, used 取值器, 单位, 说明, exact)
#
# ★ `exact` 决定 `capped` 能不能算：
#     True  = used 与 cap **是同一个量**（如「实际检索次数」对「检索次数上限」）
#             ⇒ 可以判是否撞顶。
#     False = 只是**近似/代理**指标（如拿「实际候选数」当「采样次数上限」的代理）
#             ⇒ `capped` 恒 None，并进 `approx` 清单。宁可标"判不了"，也绝不
#               让一个单位不符的数字冒充"撞顶了"——那会把调参带到错方向。
# ======================================================================
_SPEC = (
    # ==== 2026-10-01 开关注册制（审查 A 级第 2 条）======================
    # 以下 29 项来自 `agent/switch_registry.py::SWITCHES`。
    # 它们不是「用量上限（cap）」而是**开关当前取值**，故 used/capped 记 None，
    # 目的是让「启动统计」能把这些开关的生效值记录下来（注册制第③件）。
    # `src` 用 getattr 直读 AgentConfig —— 注意 env 优先级在 registry 内部处理，
    # 本表只反映配置字段值，故 note 里标注了对应的环境变量名。
    # ================================================================
    ("artifact_store_enabled",
     lambda cfg, tier, _f="artifact_store_enabled", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ARTIFACT_STORE_ENABLED — 中间结果落盘（results/<run_id>/<qid>/，默认开）", None),
    ("answer_form_gate",
     lambda cfg, tier, _f="answer_form_gate", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ANSWER_FORM_GATE — 答案形态闸门（非答案形态不放行）", None),
    ("deterministic_timeout_sec",
     lambda cfg, tier, _f="deterministic_timeout_sec", _d=5.0: getattr(cfg, _f, _d),
     None, "数",
     "DETERMINISTIC_TIMEOUT_SEC — 确定性验证通道的单次超时（秒）", None),
    ("enable_subgoal_findings",
     lambda cfg, tier, _f="enable_subgoal_findings", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_SUBGOAL_FINDINGS — 主求解注入子目标结论（阶段一信息流，默认开）", None),
    ("enable_final_answer_selection",
     lambda cfg, tier, _f="enable_final_answer_selection", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_FINAL_ANSWER_SELECTION — 终答五层选择（阶段二-2，默认开；=0 回退改动前）", None),
    ("final_answer_single_source",
     lambda cfg, tier, _f="final_answer_single_source", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FINAL_ANSWER_SINGLE_SOURCE — 决策记录与最终产出单一来源"
     "（picked=逐字提交值；默认开；=0 回退改动前）", None),
    ("final_select_verify_aware",
     lambda cfg, tier, _f="final_select_verify_aware", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FINAL_SELECT_VERIFY_AWARE — 选择机制识别未验证候选"
     "（未验证不参与胜出+投票并列不用推理长度；默认开；=0 回退改动前）", None),
    ("eval_alpha_equiv",
     lambda cfg, tier, _f="eval_alpha_equiv", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "EVAL_ALPHA_EQUIV — 字母等价（A/a）判定（默认关）", None),
    ("eval_split_cn",
     lambda cfg, tier, _f="eval_split_cn", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "EVAL_SPLIT_CN — 中文答案分隔符切分", None),
    ("expr_eval_grounding_guard",
     lambda cfg, tier, _f="expr_eval_grounding_guard", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "EXPR_EVAL_GROUNDING_GUARD — 表达式求值接地护栏", None),
    ("lean_gate_parallel",
     lambda cfg, tier, _f="lean_gate_parallel", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_GATE_PARALLEL — Lean 门禁并行预取（=0 串行）", None),
    ("lean_gate_strict_unknown",
     lambda cfg, tier, _f="lean_gate_strict_unknown", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_GATE_STRICT_UNKNOWN — Lean 门禁对 unknown 从严", None),
    ("llm_timeout",
     lambda cfg, tier, _f="llm_timeout", _d=120: getattr(cfg, _f, _d),
     None, "int",
     "LLM_TIMEOUT — LLM 读超时（秒，默认 120）。原为 utils/llm_client 模块级常量，"
     "现为调用时实时读", None),
    ("lean_crosscheck_as_unknown",
     lambda cfg, tier, _f="lean_crosscheck_as_unknown", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_CROSSCHECK_AS_UNKNOWN — cross_check/自证拦截落 unknown"
     "（形式化不充分=无法核验，非答案错）；=0 回退 proof_invalid", None),
    ("lean_mcp_autolake",
     lambda cfg, tier, _f="lean_mcp_autolake", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_MCP_AUTOLAKE — MCP 自动 lake 环境准备", None),
    ("lean_mcp_goal_loc",
     lambda cfg, tier, _f="lean_mcp_goal_loc", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_MCP_GOAL_LOC — MCP 回报 goal 位置（默认关）", None),
    ("lean_mcp_hover_check",
     lambda cfg, tier, _f="lean_mcp_hover_check", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_MCP_HOVER_CHECK — MCP hover 检查（默认关）", None),
    ("lean_mcp_multi_attempt",
     lambda cfg, tier, _f="lean_mcp_multi_attempt", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_MCP_MULTI_ATTEMPT — MCP 多次尝试（默认关）", None),
    ("lean_mcp_timeout_floor",
     lambda cfg, tier, _f="lean_mcp_timeout_floor", _d=300.0: getattr(cfg, _f, _d),
     None, "数",
     "LEAN_MCP_TIMEOUT_FLOOR — MCP 调用超时下限（秒）", None),
    ("lean_mcp_verify_axioms",
     lambda cfg, tier, _f="lean_mcp_verify_axioms", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_MCP_VERIFY_AXIOMS — MCP 校验公理使用", None),
    ("lean_verify",
     lambda cfg, tier, _f="lean_verify", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_VERIFY — Lean 通道总开关（=0 一键关闭）", None),
    ("lean_xcheck_numeric",
     lambda cfg, tier, _f="lean_xcheck_numeric", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_XCHECK_NUMERIC — 数值答案的 Lean 交叉校验", None),
    ("llm_retry_on_timeout",
     lambda cfg, tier, _f="llm_retry_on_timeout", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "LLM_RETRY_ON_TIMEOUT — LLM 超时是否重试（默认不重试）", None),
    ("minimal_mode",
     lambda cfg, tier, _f="minimal_mode", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "MINIMAL_MODE — 最小基础模式（只留主链，默认关）", None),
    ("numericize_final",
     lambda cfg, tier, _f="numericize_final", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "NUMERICIZE_FINAL — 终答数值化（把精确式转小数）", None),
    ("objective_itemwise_priority",
     lambda cfg, tier, _f="objective_itemwise_priority", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "OBJECTIVE_ITEMWISE_PRIORITY — 客观题逐项优先策略", None),
    ("objective_majority_vote",
     lambda cfg, tier, _f="objective_majority_vote", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "OBJECTIVE_MAJORITY_VOTE — 客观题多数投票（默认关）", None),
    ("objective_selfcheck",
     lambda cfg, tier, _f="objective_selfcheck", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "OBJECTIVE_SELFCHECK — 客观题自查", None),
    ("self_improve_keep_original",
     lambda cfg, tier, _f="self_improve_keep_original", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "SELF_IMPROVE_KEEP_ORIGINAL — 自改进保留原答案", None),
    ("self_improve_objective_skip",
     lambda cfg, tier, _f="self_improve_objective_skip", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "SELF_IMPROVE_OBJECTIVE_SKIP — 客观题跳过无条件自改进（=0 则也跑）", None),
    ("theorem_hint",
     lambda cfg, tier, _f="theorem_hint", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "THEOREM_HINT — 定理检索注入（=0 关闭）", None),
    ("subgoal_collab_mode",
     lambda cfg, tier, _f="subgoal_collab_mode", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "SUBGOAL_COLLAB_MODE — 子目标三角色协作求解（求解/审核/规划）", None),
    ("subgoal_collab_regen_rounds",
     lambda cfg, tier, _f="subgoal_collab_regen_rounds", _d=3: getattr(cfg, _f, _d),
     None, "num", "SUBGOAL_COLLAB_REGEN_ROUNDS — 协作求解生成遍数（默认 3）", None),
    ("enable_numeric_lean_verify",
     lambda cfg, tier, _f="enable_numeric_lean_verify", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_NUMERIC_LEAN_VERIFY — numbers 叶子 Lean 数值核验", None),
    ("self_improve_stage",
     lambda cfg, tier, _f="self_improve_stage", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "SELF_IMPROVE_STAGE — 3.3 候选级自改进（默认关，已内化为两处 3 遍）", None),
    ("cross_model_falsify",
     lambda cfg, tier, _f="cross_model_falsify", _d=True: getattr(cfg, _f, _d),
     None, "bool", "CROSS_MODEL_FALSIFY — 跨模型独立检错（第二模型 DeepSeek）", None),
    ("cross_model_timeout",
     lambda cfg, tier, _f="cross_model_timeout", _d=90.0: getattr(cfg, _f, _d),
     None, "num", "CROSS_MODEL_TIMEOUT — 跨模型检错超时（秒）", None),
    ("subgoal_regen_self_review",
     lambda cfg, tier, _f="subgoal_regen_self_review", _d=True: getattr(cfg, _f, _d),
     None, "bool", "SUBGOAL_REGEN_SELF_REVIEW — 子目标生成前自审", None),
    ("lean_hammer",
     lambda cfg, tier, _f="lean_hammer", _d=True: getattr(cfg, _f, _d),
     None, "bool", "LEAN_HAMMER — Lean 先试自动化策略链（失败再 sorry）", None),
    ("answer_shape_strict",
     lambda cfg, tier, _f="answer_shape_strict", _d=True: getattr(cfg, _f, _d),
     None, "bool", "ANSWER_SHAPE_STRICT — 4.0 条件式不算答案", None),
    ("smt_falsify",
     lambda cfg, tier, _f="smt_falsify", _d=True: getattr(cfg, _f, _d),
     None, "bool", "SMT_FALSIFY — 4.2 证伪的 Z3 通道（验蕴含式断言反例）", None),
    ("attention_focus_enabled",
     lambda cfg, tier, _f="attention_focus_enabled", _d=True: getattr(cfg, _f, _d),
     None, "bool", "ATTENTION_FOCUS_ENABLED — 2.6 关键注意点注入", None),
    ("subgoal_theorem_search",
     lambda cfg, tier, _f="subgoal_theorem_search", _d=True: getattr(cfg, _f, _d),
     None, "bool", "SUBGOAL_THEOREM_SEARCH — 每子目标单独检索定理", None),
    ("rewrite_on_low_conf",
     lambda cfg, tier, _f="rewrite_on_low_conf", _d=True: getattr(cfg, _f, _d),
     None, "bool", "REWRITE_ON_LOW_CONF — 置信度0+有淘汰证据也重写", None),
    ("rewrite_conf_threshold",
     lambda cfg, tier, _f="rewrite_conf_threshold", _d=0.0: getattr(cfg, _f, _d),
     None, "num", "REWRITE_CONF_THRESHOLD — 重写置信度阈值", None),
    ("enable_lean_tools",
     lambda cfg, tier, _f="enable_lean_tools", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_LEAN_TOOLS — Lean/Math 工具集（大模型直接调用通道）", None),
    ("toolcall_text_fallback",
     lambda cfg, tier, _f="toolcall_text_fallback", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "TOOLCALL_TEXT_FALLBACK — 文本工具调用兜底（2026-10-07 默认开：解析正文里的 "
     "<tool_call><function=web_search>… → 白名单校验 → 执行 → role=user 回填；"
     "=0 回退旧行为做 A/B）", None),

    # ==== 2026-10-04 开关注册制补齐（9 项：新增 2 + 补登记 7）============
    # 与 `agent/switch_registry.py::SWITCHES` 同名登记；格式照抄本块既有条目。
    # 说明列口径：「环境变量名 — 用途」。
    ("k11_mask_statement",
     lambda cfg, tier, _f="k11_mask_statement", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "K11_MASK_STATEMENT — K11 注入点守卫：子目标描述/蓝图结论断言值掩码（默认开）", None),
    ("sub_goal_oracle_check",
     lambda cfg, tier, _f="sub_goal_oracle_check", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "SUB_GOAL_ORACLE_CHECK — 子目标逐步 oracle 校验（默认关；"
     "此前字段全仓缺失 ⇒ 该门控结构性不可达，2026-10-04 补登记）", None),
    ("subgoal_conclusion_contract",
     lambda cfg, tier, _f="subgoal_conclusion_contract", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "SUBGOAL_CONCLUSION_CONTRACT — 子目标结论契约：强制每步产出"
     "`本步结论: 值=/等式=/判定=` + 依据 + 验算段（默认 True = 修好后行为；"
     "依据 smoke10：73 子目标 59 个 expected_output 为空 / 64 个 <120 字）", None),
    ("subgoal_objective_judge",
     lambda cfg, tier, _f="subgoal_objective_judge", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "SUBGOAL_OBJECTIVE_JUDGE — 子目标结论客观判定（SymPy 判等式 / 多候选多数），"
     "判不出时显式留痕原因（默认 True；依据：原实现 n_judged = 0/73）", None),
    ("revise_min_candidates",
     lambda cfg, tier, _f="revise_min_candidates", _d=3: getattr(cfg, _f, _d),
     None, "int",
     "REVISE_MIN_CANDIDATES — revise 每轮候选数下限（默认 3；防逐轮衰减 4→2→1）", None),
    ("require_answer_in_pool",
     lambda cfg, tier, _f="require_answer_in_pool", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "REQUIRE_ANSWER_IN_POOL — 候选入池前要求真含最终答案"
     "（默认 True；实测 59% 候选是推理句/元话语，2026-10-06 上游缺口修复）", None),
    # ---- agent/orchestrator.py（门全拒的处置策略，2026-10-06 新增）----
    # ★ 2026-10-08 补登：此前已登记 ①注册表 ②AgentConfig ③白名单，**漏本表** ⇒
    #   四处不同源。消费点 agent/orchestrator__audit_filter.py:66 / :110
    #   （Lean/Audit 门全拒时；置 True 进一步清空候选池触发下游兜底/重生成）。
    ("strict_all_rejected",
     lambda cfg, tier, _f="strict_all_rejected", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "STRICT_ALL_REJECTED — Lean/Audit 门把候选全部拒掉时的处置"
     "（默认 False = 只显式留痕，不静默保留；置 True = 清空候选池触发下游"
     "兜底/重生成链）；消费点 agent/orchestrator__audit_filter.py:66 / :110", None),
    # ★ 2026-10-07（问题 2.4 第二轮 / 2.9）开关注册制四件套（入池门槛 v2 + check 协议守卫）
    ("pool_gate_v2",
     lambda cfg, tier, _f="pool_gate_v2", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "POOL_GATE_V2 — 候选入池门槛 v2（默认 True）：判据升级为 "
     "`looks_like_pool_answer`（补题 000/002/007 规划/设定句漏网、修题 087/088 "
     "多字母选项与题 101/111 判断题误杀）+ 穷举 6 处写入入口（证明通道/生成失败"
     "抢救/自改进覆盖/截断续写/子目标候选/三 Agent 协作）。=0 回退上一轮口径做 A/B", None),
    ("check_protocol_guard",
     lambda cfg, tier, _f="check_protocol_guard", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "CHECK_PROTOCOL_GUARD — `<check>` 协议守卫（默认 True）：中文句子塞进 `<check>` "
     "⇒ 不可解析 ⇒ 子目标客观判定 0/73。打开后违规定性 + 留痕（trace/sg/diag）"
     "+ 一次定向重问；=0 回退旧行为（静默跳过）做 A/B", None),
    ("verifier_vote_reasoning",
     lambda cfg, tier, _f="verifier_vote_reasoning", _d="disagree": getattr(cfg, _f, _d),
     None, "str",
     "VERIFIER_VOTE_REASONING — 投票放开完整推理的档位"
     "（off / disagree=默认 / always；2026-10-04 B 折中）", None),
    ("blueprint_or_expand_all",
     lambda cfg, tier, _f="blueprint_or_expand_all", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "BLUEPRINT_OR_EXPAND_ALL — 蓝图 OR 节点是否展开全部分支（默认 False）", None),
    ("blueprint_node_contract",
     lambda cfg, tier, _f="blueprint_node_contract", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "BLUEPRINT_NODE_CONTRACT — 蓝图节点契约（每子目标 tool 指派 + inputs.use "
     "承接语义下传；默认 True）", None),
    ("blueprint_lean_sig",
     lambda cfg, tier, _f="blueprint_lean_sig", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "BLUEPRINT_LEAN_SIG — 蓝图叶子级形式化下传（tool=lean 节点给 formal_sig，"
     "求解侧只补证明体；默认 True）", None),
    ("blueprint_tool_exec",
     lambda cfg, tier, _f="blueprint_tool_exec", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "BLUEPRINT_TOOL_EXEC — 把声明的 tool 兑现成机器核验（lean 实编译 / sympy "
     "判等式 / numeric 跑 norm_num；默认 True）", None),
    ("use_lemma_accumulation",
     lambda cfg, tier, _f="use_lemma_accumulation", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "USE_LEMMA_ACCUMULATION — 引理积累/注入总开关（默认 True）", None),
    ("lemma_domains",
     lambda cfg, tier, _f="lemma_domains", _d=("Number theory", "数论"): getattr(cfg, _f, _d),
     None, "list",
     "LEMMA_DOMAINS — 引理注入的领域白名单（默认仅数论；arm4 放开到代数+数论）", None),
    ("candidate_pool_limit",
     lambda cfg, tier, _f="candidate_pool_limit", _d=0.0: getattr(cfg, _f, _d),
     None, "数",
     "CANDIDATE_POOL_LIMIT — 候选池上限（0 = 不限；研究期默认不限）", None),
    ("enable_subgoal_multi_agent",
     lambda cfg, tier, _f="enable_subgoal_multi_agent", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_SUBGOAL_MULTI_AGENT — 子目标内部多采样取多数（=0/未开为单次调用）", None),
    ("use_leansearch",
     lambda cfg, tier, _f="use_leansearch", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "USE_LEANSEARCH — LeanSearch 检索总开关（2026-10-12 起默认开）", None),
    # ---- 2026-10-12 补齐（门禁 `tests/test_switch_registry_four_way_guard.py`
    #      强制「注册表 A ⊆ 本表 D」—— 新增开关四处同登记，缺一即红）----
    ("enable_preverify_sketch_audit",
     lambda cfg, tier, _f="enable_preverify_sketch_audit", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_PREVERIFY_SKETCH_AUDIT — 预验证阶段骨架 Lean 审核（默认关）", None),
    # ---- 2026-10-12 批次 1：题意理解（★ 2026-10-14 收敛：只留总开关）----
    ("enable_understanding_chain",
     lambda cfg, tier, _f="enable_understanding_chain", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_UNDERSTANDING_CHAIN — 题意理解（终态 + 答案槽 + 缺口 A/B）", None),
    # ---- 2026-10-12 批次 2：蓝图多次生成 + 自审 ----
    ("blueprint_multi_gen_rounds",
     lambda cfg, tier, _f="blueprint_multi_gen_rounds", _d=3: getattr(cfg, _f, _d),
     None, "轮",
     "BLUEPRINT_MULTI_GEN_ROUNDS — 蓝图轮数（1 初始 + N-1 次自改进）", None),
    ("blueprint_self_review",
     lambda cfg, tier, _f="blueprint_self_review", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "BLUEPRINT_SELF_REVIEW — 第 4 次自审（DagReviewer）", None),
    ("blueprint_self_review_reject_ratio",
     lambda cfg, tier, _f="blueprint_self_review_reject_ratio", _d=0.4: getattr(cfg, _f, _d),
     None, "比例",
     "BLUEPRINT_SELF_REVIEW_REJECT_RATIO — 自审触发重写的拒收阈值", None),
    ("subgoal_inputs_validation",
     lambda cfg, tier, _f="subgoal_inputs_validation", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "SUBGOAL_INPUTS_VALIDATION — 子目标 inputs 机械校验（R1/R2，零 LLM）", None),
    ("enable_type_llm",
     lambda cfg, tier, _f="enable_type_llm", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_TYPE_LLM — 题型 LLM 判题（2.6 题意理解的第一步；2026-10-14 起并入）", None),
    ("subgoal_self_improve_rounds",
     lambda cfg, tier, _f="subgoal_self_improve_rounds", _d=3: getattr(cfg, _f, _d),
     None, "轮",
     "SUBGOAL_SELF_IMPROVE_ROUNDS — 子目标级无条件自改进轮数（E24）", None),
    ("subgoal_dynamic_backward",
     lambda cfg, tier, _f="subgoal_dynamic_backward", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "SUBGOAL_DYNAMIC_BACKWARD — 求解期动态倒推（缺前提→现场补一层→回填重解）", None),
    ("subgoal_backward_max_depth",
     lambda cfg, tier, _f="subgoal_backward_max_depth", _d=2: getattr(cfg, _f, _d),
     None, "层",
     "SUBGOAL_BACKWARD_MAX_DEPTH — 动态倒推递归层数上限（=0 等同关闭）", None),
    ("enable_theorem_query_llm",
     lambda cfg, tier, _f="enable_theorem_query_llm", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_THEOREM_QUERY_LLM — LLM 辅助 query（先出方向、再检索）", None),
    ("pool_empty_clear",
     lambda cfg, tier, _f="pool_empty_clear", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "POOL_EMPTY_CLEAR — 候选全滤掉时清空池并走兜底", None),
    ("five_gate_normalize",
     lambda cfg, tier, _f="five_gate_normalize", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_NORMALIZE — 4.0 答案规范化前置（五道审核链第 0 道）", None),
    ("five_gate_subgoal_lean",
     lambda cfg, tier, _f="five_gate_subgoal_lean", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_SUBGOAL_LEAN — 子目标级 Lean 断言验证（默认关）", None),
    ("five_gate_self_audit",
     lambda cfg, tier, _f="five_gate_self_audit", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_SELF_AUDIT — 4.1 AI 自审逻辑错误（五道审核链第 1 道）", None),
    ("five_gate_falsify",
     lambda cfg, tier, _f="five_gate_falsify", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_FALSIFY — 4.2 AI 对抗证伪（五道审核链第 2 道）", None),
    ("audit_obligation_ledger",
     lambda cfg, tier, _f="audit_obligation_ledger", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "AUDIT_OBLIGATION_LEDGER — 审核侧义务账本（蓝图倒推链 → 4.1/4.2 判定依据）",
     None),
    ("five_gate_lean_check_tool",
     lambda cfg, tier, _f="five_gate_lean_check_tool", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_LEAN_CHECK_TOOL — 4.3 工具①观测通道（lean_check）", None),
    ("five_gate_lean_check",
     lambda cfg, tier, _f="five_gate_lean_check", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_LEAN_CHECK — 4.3 Lean 代回题目（五道审核链第 3 道）", None),
    ("lean_failure_kind_split",
     lambda cfg, tier, _f="lean_failure_kind_split", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "LEAN_FAILURE_KIND_SPLIT — 4.3 只有 refuted 才淘汰（其余改判 unknown）", None),
    ("five_gate_verify_tiebreak",
     lambda cfg, tier, _f="five_gate_verify_tiebreak", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_VERIFY_TIEBREAK — 4.5 平票时优先「算术已核」的答案", None),
    ("blueprint_subtree_regen",
     lambda cfg, tier, _f="blueprint_subtree_regen", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "BLUEPRINT_SUBTREE_REGEN — 蓝图自审走局部子树重写（LCA 触根则整树）", None),
    ("subgoal_failed_retry_rounds",
     lambda cfg, tier, _f="subgoal_failed_retry_rounds", _d=2: getattr(cfg, _f, _d),
     None, "int",
     "SUBGOAL_FAILED_RETRY_ROUNDS — 子目标无有效结论时的重做轮数", None),
    ("five_gate_falsify_votes",
     lambda cfg, tier, _f="five_gate_falsify_votes", _d=3: getattr(cfg, _f, _d),
     None, "int",
     "FIVE_GATE_FALSIFY_VOTES — 4.2 对抗证伪采样次数（1=单次探测）", None),
    ("five_gate_self_audit_votes",
     lambda cfg, tier, _f="five_gate_self_audit_votes", _d=3: getattr(cfg, _f, _d),
     None, "int",
     "FIVE_GATE_SELF_AUDIT_VOTES — 4.1 自审采样次数（1=单次判定）", None),
    ("five_gate_audit_retry",
     lambda cfg, tier, _f="five_gate_audit_retry", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_AUDIT_RETRY — 4.1 解析失败按严格 JSON 重问一次", None),
    ("five_gate_grounding",
     lambda cfg, tier, _f="five_gate_grounding", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_GROUNDING — 五道链证据落地校验（4.1/4.2 淘汰须有原文引证）", None),
    ("five_gate_math_check_calcverify",
     lambda cfg, tier, _f="five_gate_math_check_calcverify", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_MATH_CHECK_CALCVERIFY — 4.4 工具③观测通道（calc_verify）", None),
    ("five_gate_math_check",
     lambda cfg, tier, _f="five_gate_math_check", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_MATH_CHECK — 4.4 数学工具计算检测（五道审核链第 4 道）", None),
    ("five_gate_vote",
     lambda cfg, tier, _f="five_gate_vote", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_VOTE — 4.5 投票·多答案判断+大模型投票（五道审核链第 5 道）", None),
    ("five_gate_rewrite",
     lambda cfg, tier, _f="five_gate_rewrite", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "FIVE_GATE_REWRITE — 4.6 重写通道（池空：错误诊断→重建蓝图/子目标→重写）", None),
    ("theorem_hint_backend",
     lambda cfg, tier, _f="theorem_hint_backend", _d="leanexplore": getattr(cfg, _f, _d),
     None, "str",
     "THEOREM_HINT_BACKEND — 定理检索后端（leanexplore=工具② / legacy）", None),
    ("theorem_hint_block_max_chars",
     lambda cfg, tier, _f="theorem_hint_block_max_chars", _d=2500: getattr(cfg, _f, _d),
     None, "字符",
     "THEOREM_HINT_BLOCK_MAX_CHARS — 定理注入块的字符上限", None),
    ("theorem_hint_top_k_by_stage",
     lambda cfg, tier, _f="theorem_hint_top_k_by_stage", _d=None: getattr(cfg, _f, _d),
     None, "dict",
     "THEOREM_HINT_TOP_K_BY_STAGE — 分环节定理注入条数", None),
    ("falsify_max_per_q",
     lambda cfg, tier, _f="falsify_max_per_q", _d=2: getattr(cfg, _f, _d),
     None, "数",
     "FALSIFY_MAX_PER_Q — 单题最多证伪次数（0 = 不限）", None),
    ("phase_budget_enabled",
     lambda cfg, tier, _f="phase_budget_enabled", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "PHASE_BUDGET_ENABLED — 阶段时间预算总开关（默认关）", None),
    ("enable_question_type",
     lambda cfg, tier, _f="enable_question_type", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "ENABLE_QUESTION_TYPE — 题型惰性分类总开关", None),
    # ---- agent/orchestrator.py（Step 4 反馈复核两处修复，2026-10-06 新增）----
    # ---- agent/orchestrator.py（AcceptGate 生效 + revise 计时归属，2026-10-07 新增）----

    # ---- ① LeanSearch（用户点名）--------------------------------------
    ("leansearch_max_calls_per_q",
     lambda cfg, tier: _cap(cfg, "leansearch_max_calls_per_q"),
     _u_leansearch_calls, "次",
     "单题 LeanSearch 检索调用次数上限", True),
    ("leansearch_top_k",
     lambda cfg, tier: _cap(cfg, "leansearch_top_k"),
     _u_leansearch_topk, "条",
     "单次检索返回定理条数上限（对照『到底要找多少 Mathlib 定理』）", True),

    # ---- ② 检测打回（用户点名）----------------------------------------
    ("lean_gate_unknown_stop",
     lambda cfg, tier: _cap(cfg, "lean_gate_unknown_stop"),
     _u_unknown_streak, "连击",
     "闸门连续 unknown 达 N 次即止损（0=关闭）。capped=True 表示止损真触发", True),
    ("gate_max_rework",
     _cap_gate_rework,
     _u_gate_tried, "次",
     "6.5 审核重做的候选数上限：standard=2 / deep=DEEP_MAX_REWORK"
     "（<0 = 无上限，默认 -1；与 orchestrator 消费口径一致）", True),

    # ---- ③ 无条件重做（用户点名）--------------------------------------
    ("max_revise_rounds",
     lambda cfg, tier: _cap(cfg, "max_revise_rounds"),
     _u_revise_round, "轮",
     "自纠错（revise）总轮数上限", True),
    ("deep_revise_rounds",
     lambda cfg, tier: _cap(cfg, "deep_revise_rounds"),
     _u_revise_round, "轮",
     "deep 档单次 0 票时的 revise 轮数上限（与上者同源计数 ctx.revise_round）", True),

    # ---- ④ 子目标（用户点名）------------------------------------------
    ("max_subgoals",
     lambda cfg, tier: _cap_by_tier(cfg, "max_subgoals",
                                   "max_subgoals_by_tier", tier),
     _u_subgoals, "个",
     "子目标个数上限（档位字典 max_subgoals_by_tier 优先，与代码消费逻辑一致）", True),

    # ---- ⑤ 其它可调上限（用户说的「不止这几个」）------------------------
    ("preverify_max_rounds",
     lambda cfg, tier: _cap(cfg, "preverify_max_rounds"),
     _u_preverify_rounds, "轮",
     "Lean 前置形式化验证的修正轮数上限", True),
    ("max_total_calls",
     lambda cfg, tier: _cap(cfg, "max_total_calls"),
     _u_total_calls, "次",
     "LLM 调用记账上限（⚠ 只记账不阻断，见 agent/base.py:327）", True),

    # ---- ⑥ 只有代理指标、判不了撞顶的（标"近似"，不冒充撞顶）------------
    ("collab_max_rounds",
     lambda cfg, tier: _cap(cfg, "collab_max_rounds"),
     _u_deep_review, "轮",
     "deep 档三 Agent 协作循环上限（无专有计数，代理=deep_review 事件数）", False),
    ("self_improve_max",
     lambda cfg, tier: _cap(cfg, "self_improve_max"),
     _u_candidates, "个",
     "3.3 自改进候选数上限（代理=实际候选数，二者非同一量）", False),
    ("policy_sample_times",
     lambda cfg, tier: _cap(cfg, "policy_sample_times"),
     _u_candidates, "个",
     "策略采样候选数上限（代理=实际候选数，二者非同一量）", False),
    ("verifier_voting_times",
     lambda cfg, tier: _cap(cfg, "verifier_voting_times"),
     _u_verdicts, "次",
     "验证器投票次数上限（代理=len(ctx.verdicts)，含加赛票，非同一量）", False),
    ("verifier_disagreement_votes",
     lambda cfg, tier: _cap(cfg, "verifier_disagreement_votes"),
     _u_verdicts, "次",
     "分歧时追加投票次数（代理=len(ctx.verdicts)，非同一量）", False),

    # ---- ⑦ token 类上限（实测口径不在 ctx 上）--------------------------
    ("max_answer_tokens",
     lambda cfg, tier: _cap(cfg, "max_answer_tokens"),
     lambda ctx: (None, ""), "token",
     "答案生成上限；实测见 LLM 截断台账（utils.llm_client._TRUNCATION_STATS）", False),
    ("verifier_deep_review_max_tokens",
     lambda cfg, tier: _cap(cfg, "verifier_deep_review_max_tokens"),
     lambda ctx: (None, ""), "token",
     "深复核 token 上限；实测见 LLM 截断台账", False),
    ("adversarial_max_tokens",
     lambda cfg, tier: _cap(cfg, "adversarial_max_tokens"),
     lambda ctx: (None, ""), "token",
     "对抗验证 token 上限；实测见 LLM 截断台账", False),
    ("max_subgoals_by_tier",
     lambda cfg, tier: (None if not isinstance(
         _cfg_get(cfg, "max_subgoals_by_tier"), dict)
         else len(_cfg_get(cfg, "max_subgoals_by_tier"))),
     lambda ctx: (None, ""), "档位",
     "档位化上限表（本身不是数值上限，仅登记其存在与档位数）", False),

    # ==== 2026-10-07 开关注册制补齐（fix-audit-gate「解答题判别力」）========
    # 与 `agent/switch_registry.py::SWITCHES` 同名登记；开关型 used/capped=None。
    ("deterministic_solution_check",
     lambda cfg, tier, _f="deterministic_solution_check", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "DETERMINISTIC_SOLUTION_CHECK — 解答题确定性判别重写总开关"
     "（默认 True：归一答案外壳 + 全量等式回代 + 函数方程回代；=0 回退旧行为）", None),
    ("audit_gate_rubric_nonproof",
     lambda cfg, tier, _f="audit_gate_rubric_nonproof", _d=False: getattr(cfg, _f, _d),
     None, "bool",
     "AUDIT_GATE_RUBRIC_NONPROOF — 解答题是否也跑 Level2 rubric"
     "（默认 False：LLM 同源自评、价值低耗时高；=1 开启）", None),

    # ==== 2026-10-07 开关注册制补齐（fix-switch-debt：6 个账外 gate 型 env）=======
    # 与 `agent/switch_registry.py::SWITCHES` 同名登记；开关型 used/capped=None，
    # 目的是让启动统计能把这些开关的生效值记录下来（注册制第③件）。
    # ★ 2026-10-15：emergency_direct_max_tokens / emergency_answer_max_tokens /
    #   final_answer_min_sec 已随直答兜底删除。
    ("deep_max_rework",
     lambda cfg, tier, _f="deep_max_rework", _d=-1: getattr(cfg, _f, _d),
     None, "次",
     "DEEP_MAX_REWORK — 6.5 重做候选上限（deep；-1=无上限）", None),
    ("objective_itemwise_plan",
     lambda cfg, tier, _f="objective_itemwise_plan", _d=True: getattr(cfg, _f, _d),
     None, "bool",
     "OBJECTIVE_ITEMWISE_PLAN — 子目标规划客观题逐项策略（=0 关闭）", None),
)


# ======================================================================
# 对外入口
# ======================================================================
def _counters(ctx) -> dict:
    """一题一次的关键计数快照（与上限无关，但报告里必须一起看）。"""
    mus = getattr(ctx, "mathlib_usage_stats", None)
    mus = mus if isinstance(mus, dict) else {}
    used_th = getattr(ctx, "used_theorems", None)
    return {
        "tier": str(getattr(ctx, "tier", "") or ""),
        "leansearch_search_calls": _int(mus.get("search_calls")),
        "leansearch_hits_raw": _int(mus.get("search_hits")),
        "distinct_theorems": (len(used_th)
                              if isinstance(used_th, list) and used_th else None),
        "lean_compile_valid": _int(mus.get("compile_valid")),
        "n_candidates": _int(getattr(ctx, "n_candidates", None)),
        "n_verdicts": _int(getattr(ctx, "n_verdicts", None)),
        "revise_round": _int(getattr(ctx, "revise_round", None)),
    }


_EMPTY = {"schema": SCHEMA, "tier": "", "items": {}, "counters": {},
          "gaps": [], "approx": []}


def collect_param_usage(ctx, cfg) -> dict:
    """采集本题「上限 vs 实测」表。**任何异常都被吞掉**，最坏返回空表。"""
    try:
        return _collect_inner(ctx, cfg)
    except Exception:  # noqa: BLE001
        return dict(_EMPTY)


def _collect_inner(ctx, cfg) -> dict:
    try:
        tier = str(getattr(ctx, "tier", "") or "")
    except Exception:  # noqa: BLE001
        # ⚠ 取不到 tier 不该放弃整张表：上限（来自 cfg）与其它实测都还在。
        #   0921 单测正是靠这条发现"提前 return 空表"会把有用信息一起丢掉。
        tier = ""
    try:
        counters = _counters(ctx)
    except Exception:  # noqa: BLE001
        counters = {}

    items: dict = {}
    gaps: list = []
    approx: list = []
    for name, cap_fn, used_fn, unit, note, exact in _SPEC:
        entry = {"cap": None, "used": None, "capped": None, "exact": bool(exact),
                 "unit": unit, "src": "", "note": note}
        try:
            cap = cap_fn(cfg, tier)
        except Exception:  # noqa: BLE001
            cap = None
        try:
            used, src = used_fn(ctx)
        except Exception:  # noqa: BLE001
            used, src = None, ""
        entry["cap"] = cap
        entry["used"] = used
        entry["src"] = src
        # capped：仅在「口径精确 + 有上限 + 有实测」时才判
        if cap is not None and used is not None and exact:
            if cap <= 0:
                # 多个上限用 0 表示"关闭该限制"（如 lean_gate_unknown_stop）
                entry["capped"] = None
                entry["note"] = note + "（cap=0 表示该限制已关闭）"
            else:
                entry["capped"] = bool(used >= cap)
        if used is None:
            gaps.append(name)
        if not exact:
            approx.append(name)
        items[name] = entry

    return {"schema": SCHEMA, "tier": tier, "items": items,
            "counters": counters,
            "gaps": sorted(gaps), "approx": sorted(approx)}


def aggregate(records) -> dict:
    """把多题的 `diag.param_usage` 汇总成「实测 vs 上限」对照。

    ``records`` = 可迭代的 ``param_usage`` 字典（各题）。
    返回::

        {"n_questions": N,
         "items": {name: {"cap": 上限众数, "cap_min", "cap_max", "n_used",
                          "used_mean", "used_max", "n_capped", "capped_rate"}},
         "counters": {key: {"mean","max","n"}}}

    只做纯算术，不做判断 —— 判断（该不该调）留给报告层，方便复算。
    """
    from collections import Counter

    rows = [r for r in (records or []) if isinstance(r, dict)]
    caps: dict = {}
    useds: dict = {}
    capped_cnt: dict = {}
    exact_seen: dict = {}
    ctr: dict = {}
    n = 0
    for r in rows:
        items = r.get("items")
        if not isinstance(items, dict):
            continue
        n += 1
        for name, e in items.items():
            if not isinstance(e, dict):
                continue
            if e.get("cap") is not None:
                caps.setdefault(name, []).append(e["cap"])
            if e.get("used") is not None:
                useds.setdefault(name, []).append(e["used"])
            if e.get("capped") is True:
                capped_cnt[name] = capped_cnt.get(name, 0) + 1
            # ★ exact 必须跟着聚合走：否则报告层会把代理口径当成可判撞顶，
            #   而 aggregate() 是报告/测试共用的入口，丢了它两处都会误判。
            if e.get("exact") is True:
                exact_seen[name] = exact_seen.get(name, 0) + 1
        c = r.get("counters")
        if isinstance(c, dict):
            for k, v in c.items():
                if isinstance(v, int):
                    ctr.setdefault(k, []).append(v)

    def _stat(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return None
        return {"n": len(vals), "mean": round(sum(vals) / len(vals), 2),
                "max": max(vals), "min": min(vals)}

    out_items = {}
    for name in sorted(set(caps) | set(useds) | set(capped_cnt) | set(exact_seen)):
        cap_vals = caps.get(name) or []
        mode = None
        if cap_vals:
            mode = Counter(cap_vals).most_common(1)[0][0]
        out_items[name] = {
            "cap": mode,
            "exact": bool(exact_seen.get(name)),
            "cap_min": min(cap_vals) if cap_vals else None,
            "cap_max": max(cap_vals) if cap_vals else None,
            "cap_values": sorted(set(cap_vals))[:6],
            "used": _stat(useds.get(name) or []),
            "n_capped": capped_cnt.get(name, 0),
            "capped_rate": (round(capped_cnt.get(name, 0) / n, 3) if n else None),
        }
    return {"n_questions": n, "items": out_items,
            "counters": {k: _stat(v) for k, v in sorted(ctr.items())}}
