# -*- coding: utf-8 -*-
"""开关注册表 —— 「配置开关唯一注册制」的唯一权威来源。

背景（2026-10-01 代码审查 A 级第 2 条）
======================================
`agent/` 核心逻辑里有 29 个 gate 型开关，直接用裸 `os.environ.get` 读取：
既不在 `AgentConfig` 声明、也不在白名单、也不在启动统计 ⇒
**改了不生效、生效查不到、报告还原不出来**。

本模块提供三件事：
  1. `SWITCHES`：**全部开关的集中登记**（字段名 / 环境变量 / 默认值 / 语义 / 说明）；
  2. `get_bool() / get_num() / get_str()`：**统一读取器**，调用点不再直接碰 `os.environ`；
  3. `snapshot()`：当前**生效值快照**，挂进 `diag` ⇒ 报告可还原。

读取优先级（★ 关键，保证既有行为不变）
=====================================
    环境变量  >  AgentConfig 覆盖（CLI / kwargs）  >  默认值

把**环境变量放在最前**是刻意的：本项目既有的跑法（`X=0 python run_eval.py`、
`deploy/run_112.py` 里的 env）全部依赖 env，若改成 config 优先，现有命令会静默失效。

取值语义统一
============
现状有四种写法混用：`== "1"` / `!= "0"` / `== "0"` 关 / `in ("0","false","no")` 关。
本模块统一为：**`"0"/"false"/"no"/"off"/""`（忽略大小写）= 关，其余 = 开**。
对默认值与常规 `"0"/"1"` 用法，行为与改前**完全一致**；只对 `"yes"/"true"` 之类的
非常规值有差异（改前视作关，现在视作开）—— 属修正而非回归。

★ 2026-10-07 架构解耦（阶段 B2）：本模块由 `agent/switch_registry.py` **下沉**至此。
它**零依赖**（只 import `os` / `typing`），却曾被 `utils/`（`llm_client.py`）与
`tools/lean_local/`（`lean_bridge.py` / `lean_gate.py`）导入 ⇒ 形成 `utils→agent`、
`tools→agent` 的跨层边。下沉到零依赖的 `contracts/` 后，这些边改走 `contracts/`（合法）。
`agent/switch_registry.py` 保留为**纯 re-export shim** ⇒ 全仓
`from agent.switch_registry import ...` **一行不改**（同一份状态、同一批函数对象，
见 shim 文件与 `tests/test_decouple_stage_a_1007.py`）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# field: (env_key, kind, default, doc)
#   kind: "bool" | "num" | "str"
SWITCHES: Dict[str, Tuple[str, str, Any, str]] = {
    # ---- agent/base.py ----
    # ★ 2026-10-07 默认值 False → **True**（本轮唯一一处默认翻转，理由如下）。
    #   依据（2026-10-06 smoke10 实测，`results/_cloud/.../run_2026-10-06_...jsonl`）：
    #     · 10 题里 `toolcall_text_detected` 共 **93 条**（题 000=18、007=16、003=15、
    #       004=11、001=13…），而 `tool_calls.web_search.calls` 多数题为 **0**
    #       ⇒ 模型把工具调用写在正文/推理里、平台不填原生 `tool_calls`
    #       ⇒ 这些调用**一次都没执行**，前面的散文被 `extract_final_answer` 当答案。
    #     · 题 000 最强证据：模型自己写道「上一轮解答中，我错误地假设水平移动只能单向
    #       …最终推出矛盾得到答案 0。但题目条…」并随即发出 `web_search` 去核对自纠错
    #       —— **这条自纠错/验证请求被整条丢弃**（该题最终答 0，标准答案 20460）。
    #   风险评估（"误解析模型的举例说明文本"）：**低**。
    #     · 93/93 抽样到的实例都是**真实调用意图**（`<function=web_search>` +
    #       `<parameter=query>…检索式…`），未见"讲解语法"式的举例；
    #     · 系统提示已明令禁止手写这些标签（`base.py::llm_with_tools` 的 `_tool_hint`），
    #       仍出现的即为真实调用意图；
    #     · 执行侧有**硬白名单**（仅 `web_search`）+ 必填参数非空校验 + 幂等去重 +
    #       `max_rounds` 限流，且只有 `enable_web_search=True`（默认已 True）时
    #       工具循环才会走到这里 ⇒ 不放宽任何权限。
    #   收益 ≥ 风险 ⇒ 翻转。`=0` 仍可一键回退旧行为做 A/B。
    "toolcall_text_fallback": (
        "TOOLCALL_TEXT_FALLBACK", "bool", True,
        "文本工具调用兜底（默认开，2026-10-07）：模型在正文/推理里手写 "
        "`<tool_call><function=web_search><parameter=query>…` 时，解析 → 白名单校验 "
        "→ 真正执行 → 结果以 role=user 回填后继续。依据 smoke10：10 题共 93 次"
        "文本调用被丢弃（含题 000 的自纠错检索请求）。=0 回退旧行为做 A/B"),
    "enable_lean_tools": (
        "ENABLE_LEAN_TOOLS", "bool", True,
        "Lean/Math 工具集注册（2026-10-15 工具化定稿 · 双通道）：大模型可直接调用 "
        "lean_search / calc_verify / to_lean / lean_check（实现在 tools/agent_toolkit）；"
        "=0 关闭做 A/B"),
    "subgoal_collab_mode": (
        "SUBGOAL_COLLAB_MODE", "bool", True,
        "子目标三角色协作求解（2026-10-15 用户定案）：求解/审核/规划三 agent 合作 "
        "+ 审核用工具①③ + 3 遍「在现有基础上再生成」+ 规划预取与蓝图回传；"
        "=1 启用；=0（默认）走旧逐步求解，可 A/B"),
    "subgoal_collab_regen_rounds": (
        "SUBGOAL_COLLAB_REGEN_ROUNDS", "num", 3,
        "子目标协作求解的生成遍数（默认 3；1=关闭迭代；上限 5）"),
    "enable_numeric_lean_verify": (
        "ENABLE_NUMERIC_LEAN_VERIFY", "bool", False,
        "numbers 叶子 Lean 数值核验开关（2026-10-15 审核 F5 补登记）：原字段只"
        "存在于 kwargs 白名单/评测覆盖 ⇒ bare 入口恒 False（空转）。默认 False "
        "与现行为一致；评测经覆盖传 True 时生效。"),
    "self_improve_stage": (
        "SELF_IMPROVE_STAGE", "bool", False,
        "3.3 候选级「无条件自改进」总开关（2026-10-17 用户定案内化：默认 False="
        "由蓝图 3 遍 + 子目标 3 遍覆盖；置 True 恢复旧阶段）。"),
    "cross_model_falsify": (
        "CROSS_MODEL_FALSIFY", "bool", True,
        "跨模型独立检错（2026-10-17 用户定案）：对机检+同模型证伪都放行的候选，"
        "调第二模型（DeepSeek）独立判断；wrong 即淘汰。失败/超时放行。"),
    "cross_model_timeout": (
        "CROSS_MODEL_TIMEOUT", "num", 90,
        "跨模型检错的单次 HTTP 超时（秒）"),
    "subgoal_regen_self_review": (
        "SUBGOAL_REGEN_SELF_REVIEW", "bool", True,
        "子目标 3 遍迭代的「生成前自审」（2026-10-17 用户定案：先自己审核再生成；"
        "3 遍本身保留，用于最大化推理）"),
    "lean_hammer": (
        "LEAN_HAMMER", "bool", True,
        "Lean 代回先试自动化策略链（omega/linarith/nlinarith/norm_num/ring/"
        "positivity/decide/tauto/simp/aesop/grind），失败再 sorry；关=旧行为"),
    "answer_shape_strict": (
        "ANSWER_SHAPE_STRICT", "bool", True,
        "4.0 答案形态严格校验：条件式/命题式（含「且/∧/∀/∃」）不算答案"),
    "smt_falsify": (
        "SMT_FALSIFY", "bool", True,
        "4.2 证伪增加 SMT(Z3) 通道：验证「蕴含式断言」是否有反例（覆盖逻辑命题类"
        "错误 008/015）"),
    "attention_focus_enabled": (
        "ATTENTION_FOCUS_ENABLED", "bool", True,
        "2.6 题意理解产出「关键注意点」，注入求解/子目标 prompt 顶部（注意力引导）"),
    "subgoal_theorem_search": (
        "SUBGOAL_THEOREM_SEARCH", "bool", True,
        "每个子目标按自身文本单独检索定理（per-subgoal，比全局一次更相关）"),
    "rewrite_on_low_conf": (
        "REWRITE_ON_LOW_CONF", "bool", True,
        "4.6 重写触发扩展：池非空但「有淘汰证据 且 投票置信度 0」也重写（不放行错答案）"),
    "rewrite_conf_threshold": (
        "REWRITE_CONF_THRESHOLD", "num", 0.0,
        "重写触发的置信度阈值（vote_correct/vote_total ≤ 此值 才考虑重写；默认 0）"),
    # ---- agent/deterministic.py ----
    "deterministic_timeout_sec": (
        "DETERMINISTIC_TIMEOUT_SEC", "num", 5.0,
        "确定性验证通道的单次超时（秒）"),
    # ---- agent/orchestrator.py ----
    "lean_verify": (
        "LEAN_VERIFY", "bool", True, "Lean 通道总开关（=0 一键关闭）"),
    "numericize_final": (
        "NUMERICIZE_FINAL", "bool", True, "终答数值化（把精确式转小数）"),
    "objective_selfcheck": (
        "OBJECTIVE_SELFCHECK", "bool", True, "客观题自查"),
    "answer_form_gate": (
        "ANSWER_FORM_GATE", "bool", True, "答案形态闸门（非答案形态不放行）"),
    "lean_xcheck_numeric": (
        "LEAN_XCHECK_NUMERIC", "bool", True, "数值答案的 Lean 交叉校验"),
    "self_improve_objective_skip": (
        "SELF_IMPROVE_OBJECTIVE_SKIP", "bool", True,
        "客观题跳过无条件自改进（=0 则也跑）"),
    "objective_majority_vote": (
        "OBJECTIVE_MAJORITY_VOTE", "bool", False, "客观题多数投票（默认关）"),
    # ---- agent/solver.py ----
    "expr_eval_grounding_guard": (
        "EXPR_EVAL_GROUNDING_GUARD", "bool", True, "表达式求值接地护栏"),
    "enable_subgoal_findings": (
        "ENABLE_SUBGOAL_FINDINGS", "bool", True,
        "主求解注入子目标结论（阶段一信息流：初始/证明/重解三路径；=0 关闭）"),
    # ★ 2026-10-01：`SELF_IMPROVE_CONDITIONAL` **不纳入注册表** ——
    #   `AgentConfig.self_improve_conditional` 已存在（默认 False），而 env 语义是
    #   `!= "0"` 的取反分支（默认 "1" ⇒ 该分支不执行），两者语义不一致。
    #   待厘清后再决定是「统一到配置字段」还是「保持 env」，此处不重复定义。
    "self_improve_keep_original": (
        "SELF_IMPROVE_KEEP_ORIGINAL", "bool", True, "自改进保留原答案"),
    # ---- agent/sub_goal_solver.py ----
    "objective_itemwise_priority": (
        "OBJECTIVE_ITEMWISE_PRIORITY", "bool", True, "客观题逐项优先策略"),
    # ---- agent/theorem_hint.py ----
    "theorem_hint": (
        "THEOREM_HINT", "bool", True, "定理检索注入（=0 关闭）"),
    # ---- utils/llm_client.py ----
    "llm_retry_on_timeout": (
        "LLM_RETRY_ON_TIMEOUT", "bool", False, "LLM 超时是否重试（默认不重试）"),
    # ---- tools/lean_local/lean_bridge.py ----
    "lean_mcp_autolake": (
        "LEAN_MCP_AUTOLAKE", "bool", True, "MCP 自动 lake 环境准备"),
    "lean_mcp_timeout_floor": (
        "LEAN_MCP_TIMEOUT_FLOOR", "num", 300.0, "MCP 调用超时下限（秒）"),
    "lean_mcp_goal_loc": (
        "LEAN_MCP_GOAL_LOC", "bool", False, "MCP 回报 goal 位置（默认关）"),
    "lean_mcp_multi_attempt": (
        "LEAN_MCP_MULTI_ATTEMPT", "bool", False, "MCP 多次尝试（默认关）"),
    "lean_mcp_hover_check": (
        "LEAN_MCP_HOVER_CHECK", "bool", False, "MCP hover 检查（默认关）"),
    "lean_mcp_verify_axioms": (
        "LEAN_MCP_VERIFY_AXIOMS", "bool", True, "MCP 校验公理使用"),
    # ---- tools/lean_local/lean_gate.py ----
    "lean_gate_parallel": (
        "LEAN_GATE_PARALLEL", "bool", True, "Lean 门禁并行预取（=0 串行）"),
    "lean_gate_strict_unknown": (
        "LEAN_GATE_STRICT_UNKNOWN", "bool", True, "Lean 门禁对 unknown 从严"),
    # ---- tools/lean_local/lean_bridge.py（verify_answer 的 cross_check/自证守卫）----
    # ★ 2026-10-07（语义分离）：「形式化不充分」≠「命题为假」。
    #   实测 smoke10 `lean_gate` 123 条 = proof_invalid 73 / unknown 41 /
    #   answer_valid 1，其中 51 条 cross_check_failed **全部 compiled=true**；
    #   铁证 official112-004：`example : (2025 : ℕ) / 2 = 1012 := by norm_num`
    #   （真命题、编译通过）被判 `proof_invalid` —— 把「无法核验」误当成「证明失败」。
    #   本开关=True 时 cross_check/自证拦截落 `unknown`（**无法核验，非答案错**）；
    #   =False 回退旧语义（一律 `proof_invalid`），用于 A/B 对照。
    # ---- utils/llm_client.py（2026-10-07 补齐：原为模块级常量且未登记）----
    "llm_timeout": (
        "LLM_TIMEOUT", "num", 120,
        "LLM 调用读超时（秒）。原为 utils/llm_client.py 的**模块级常量**"
        "`_DEFAULT_TIMEOUT_llm_client = int(os.getenv('LLM_TIMEOUT','120'))` —— "
        "① import 期求值，遇'先 import 后改 env'（cli/run_phase0_eval.py 等正是此写法）会静默失效；"
        "② 不入注册表 ⇒ diag 的 switch_snapshot 看不到、config/CLI 覆盖读不到。"
        "现改为调用时经注册表读（显式传入的 timeout 仍最高优先）"),
    "lean_crosscheck_as_unknown": (
        "LEAN_CROSSCHECK_AS_UNKNOWN", "bool", True,
        "cross_check/自证拦截命中时落 unknown（形式化不充分=无法核验，非答案错）；"
        "=0 回退旧语义（一律 proof_invalid）做 A/B"),
    # ---- run_eval.py ----
    "eval_split_cn": (
        "EVAL_SPLIT_CN", "bool", True, "中文答案分隔符切分"),
    "eval_alpha_equiv": (
        "EVAL_ALPHA_EQUIV", "bool", False, "字母等价（A/a）判定（默认关）"),
    # ---- deploy/run_112.py ----
    # ---- agent/artifact_store.py（2026-10-02 中间结果存储层）----
    "artifact_store_enabled": (
        "ARTIFACT_STORE_ENABLED", "bool", True,
        "中间结果落盘（results/<run_id>/<qid>/，append-only，默认开）"),
    # ---- user_agent.py（2026-10-02 最小基础模式）----
    "minimal_mode": (
        "MINIMAL_MODE", "bool", False,
        "最小基础模式：只留主链（题意理解→子目标→求解→形式化验证），"
        "关闭与推导无关的旁支（默认关）"),
    # ---- agent/final_selector.py（2026-10-02 阶段二-2 终答五层选择）----
    "enable_final_answer_selection": (
        "ENABLE_FINAL_ANSWER_SELECTION", "bool", True,
        "终答五层选择（②多答案判断→③客观验证→④模型对比→⑤投票兜底；"
        "ON 时终答不再走 _rank_key 票数优先 / objective_majority_vote，"
        "由 agent/final_selector.select_final_answer 统一选出；=0 全部回退改动前）"),
    # ---- 2026-10-04 开关注册制补齐（9 项：新增 2 + 补登记 7）----
    # ⚠ 下列默认值均与 `user_agent.py::AgentConfig` 同名字段的**当前默认值**逐条核对一致
    #   （否则 `bind_config` 会把两者绑成不一致的值）。本批**只登记，不翻转任何既有默认值**。
    # ---- agent/sub_goal_solver.py（K11 注入点守卫，消费方已存在）----
    "k11_mask_statement": (
        "K11_MASK_STATEMENT", "bool", True,
        "K11 注入点守卫：把子目标描述/蓝图结论里被断言的数值掩为占位符（=0 关闭）"),
    "sub_goal_oracle_check": (
        "SUB_GOAL_ORACLE_CHECK", "bool", False,
        "子目标逐步 oracle 校验（AnswerOracle sanity check，返回非空则重试该步；"
        "默认关 —— 2026-09-03 老师口径：oracle 是同源自评、价值低耗时高）"),
    # ---- agent/sub_goal_solver.py（子目标交付重写 2026-10-06）----
    "subgoal_conclusion_contract": (
        "SUBGOAL_CONCLUSION_CONTRACT", "bool", True,
        "子目标**结论契约**（默认 True = 修好后行为）：step 提示词强制每步产出"
        "`本步结论: 值=/等式=/判定=` + `本步依据:` + 验算段，"
        "并把结构化结论/推导长度/「仅断言」标记写进 trace。"
        "依据（smoke10 2026-10-06，两轮实测一致）：73 个子目标里 64 个交付 <120 字、"
        "59 个 expected_output 为空、客观判定 0/73、题 007 子目标只给结论无推导却被"
        "主求解采信 ⇒ 答错。=0 回退旧行为（只给自由文本结论）做 A/B"),
    "subgoal_objective_judge": (
        "SUBGOAL_OBJECTIVE_JUDGE", "bool", True,
        "子目标结论**客观判定**（默认 True = 修好后行为）：对每个子目标结论做 0-LLM "
        "客观判定（SymPy 判等式真伪 / 多候选符号多数），判不出时**显式留痕说明原因**"
        "（不再静默计 -1）。依据：原实现只在多 agent 路径且候选全可解析时才判，"
        "实测 `n_judged = 0/73`。=0 回退旧行为"),
    # ---- agent/final_selector.py 候选池（研究期口径）----
    "candidate_pool_limit": (
        "CANDIDATE_POOL_LIMIT", "num", 12.0,
        "候选池上限（**默认 12**；2026-10-09 用户拍板由 6 抬到 12：「先 12 试试」，"
        "原 6 = 与旧行为逐字一致。0 = 不限（受内部安全上界 12 约束）。"
        "⚠ 该值同时是本次生成数量，勿设天文数字 —— 设成天文数字会让 "
        "`_generate_initial` 尝试生成天量候选而跑飞；抬到 12 后 `0=不限` 与 `=12` "
        "行为等价，见 `agent/solver__pool._POOL_UNLIMITED_ROOM`）"),
    # ---- agent/final_selector.py（终答选择两条阻断级修复，2026-10-07 smoke10）----
    "final_answer_single_source": (
        "FINAL_ANSWER_SINGLE_SOURCE", "bool", True,
        "终答记录**单一来源**（默认 True = 修好后行为）：把 `_pick_diag.picked` 同步为"
        "**逐字提交值**（`ctx.final_response`）。依据：实测 official112-002 的 "
        "`pick_diag.picked`=计划句、而被评分的 `predicted`=`\\boxed{...}` —— 6.5 "
        "闸门在 formatter 之后经 `_set_final_response`→`adopt_single` 改写提交值却不"
        "动记录 ⇒ 决策与产出不同源、选择器结论无法验证。=0 回退改动前（picked 仍是"
        "决策值/截断值）做 A/B"),
    "final_select_verify_aware": (
        "FINAL_SELECT_VERIFY_AWARE", "bool", True,
        "终答选择**识别未验证候选**（默认 True = 修好后行为）：依 `ctx.verdicts[*]."
        "answer` 建答案级验证索引；有已验证候选时 ④模型对比/⑤投票只在已验证候选里选"
        "（跳过无验证记录者），全部未验证时显式留痕 all_unverified 且不得仅凭未验证"
        "胜出；⑤投票并列收尾不再用推理长度。依据：实测 official112-007 的 `\\boxed{0}`"
        "（gold=1）**无任何验证记录**（confidence/correct_votes/total_votes 全 None）"
        "却参与投票并在 1:1 平票后胜出。=0 回退改动前做 A/B"),
    # ---- user_agent.py / agent/sub_goal_solver.py（多 agent 求解）----
    "enable_subgoal_multi_agent": (
        "ENABLE_SUBGOAL_MULTI_AGENT", "bool", True,
        "子目标内部多采样取多数（2026-10-04 默认 True；=0 取回单次调用旧基线）"),
    # ---- user_agent.py / agent/solver.py（LeanSearch 引理检索）----
    "use_leansearch": (
        "USE_LEANSEARCH", "bool", True,
        "LeanSearch 检索总开关（★ 2026-10-12 用户决策 Q3：**纳入研究期必用清单，默认由 False 改为 True**）。"
        "注：实际评测入口（deploy/run_112.py:274 / run_research.py:69）此前已显式传 `--use_leansearch true`，"
        "故对云端评测**无行为变化**；本改动的作用是让「裸跑 run_eval」也默认开启，并消除「默认关」的误读。"),
    # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：原
    #   `dag_replan_gate` / `dag_replan_lean_check` / `dag_lean_max_nodes`
    #   （求解前 DAG 强制评审门 + 其 Lean 逻辑检查）三个开关已整块撤除。
    "enable_preverify_sketch_audit": (
        "ENABLE_PREVERIFY_SKETCH_AUDIT", "bool", False,
        "预验证阶段的骨架 Lean 审核（与 2.7 DAG 整树审核功能重叠，默认关）"),
    # ---- 2026-10-12 批次 1：题意理解（★ 2026-10-14 收敛）----
    "enable_understanding_chain": (
        "ENABLE_UNDERSTANDING_CHAIN", "bool", True,
        "题意理解：终态构建 + 答案槽（**纯符号**，代码机械剥离被写死的数值）+ "
        "缺口 A/B（用户 2026-10-12 决策启用；=0 回退）。"
        "★ 2026-10-14（用户口径「题意理解只做让模型注意到所求部分，整体理解归蓝图」）："
        "**删『逻辑链 + 一致性复推 + 链的 Lean 自检』** —— 链式分解与 2.7 蓝图 "
        "AND-OR DAG 同层同义，配套三个开关（understanding_refine_rounds / "
        "understanding_no_gain_stop / understanding_chain_lean_check）一并撤除"),
    # ---- 2026-10-12 批次 2：蓝图多次生成（迭代式自改进）+ 自审 ----
    "blueprint_multi_gen_rounds": (
        "BLUEPRINT_MULTI_GEN_ROUNDS", "num", 3,
        "蓝图生成轮数：1 次初始 + (N-1) 次在上一轮基础上的无条件自改进"
        "（=1 等价于单次生成；用户 Q11 决策默认 3）"),
    "blueprint_self_review": (
        "BLUEPRINT_SELF_REVIEW", "bool", True,
        "第 4 次自审：LLM 审一遍蓝图（DagReviewer），按建议重写 1 轮即停"),
    "blueprint_self_review_reject_ratio": (
        "BLUEPRINT_SELF_REVIEW_REJECT_RATIO", "num", 0.4,
        "自审触发重写的拒收比例阈值（与 dag_replan_gate 同口径 40%）"),
    # ---- 2026-10-12 批次 3：子目标 inputs 机械校验（R1/R2）----
    "subgoal_inputs_validation": (
        "SUBGOAL_INPUTS_VALIDATION", "bool", True,
        "子目标 inputs 机械校验（R1 叶子判据 / R2 断链检测；零 LLM）"),
    # ---- 2026-10-12 批次 4：验证链重排（用户 Q29–Q33）----
    "enable_type_llm": (
        "ENABLE_TYPE_LLM", "bool", True,
        "**题型 LLM 判题**（§3 方案 A）：在阶段 `2.6_pre_audit`（**题意理解**）的"
        "第一步用 LLM 判题型（2026-10-14 任务 #1 起与原 1.1 独立阶段合并），"
        "**与规则基线交叉校验**（规则强信号优先、低置信不覆盖）；失败完全回退规则"),
    "subgoal_self_improve_rounds": (
        "SUBGOAL_SELF_IMPROVE_ROUNDS", "num", 3,
        "子目标级**无条件**自改进轮数（用户 E24：「让大模型反复写一道题」）；"
        "同轮产出与上轮相同/为空即提前停；=0 关闭"),
    # ---- 2026-10-15（用户「2 与 3 之间建一个环」）：求解期动态倒推 ----
    "subgoal_dynamic_backward": (
        "SUBGOAL_DYNAMIC_BACKWARD", "bool", True,
        "**求解期动态倒推**：子目标判定「缺前提」时**现场再倒推一层**，把缺失的"
        "前提立为新子目标、就地求解（可递归）、把结论回填后重解原目标。"
        "生成期倒推（prompts/blueprint.py）只能按**题面**判断前提是否具备，"
        "求解期才暴露的缺口靠本机制补；=0 关闭（回到「失败只重做一次」旧行为）"),
    "subgoal_backward_max_depth": (
        "SUBGOAL_BACKWARD_MAX_DEPTH", "num", 2,
        "动态倒推的**递归层数上限**（防「缺前提→再倒推→又缺前提」无限套娃）；"
        "=0 等同关闭本机制"),
    "enable_theorem_query_llm": (
        "ENABLE_THEOREM_QUERY_LLM", "bool", True,
        "**LLM 辅助 query**（用户 A1）：先让大模型给出 2–3 条检索方向（只给关键词、"
        "不给定理名，防幻觉），再交检索器出事实；失败自动回退纯规则 query"),
    "pool_empty_clear": (
        "POOL_EMPTY_CLEAR", "bool", True,
        "候选全被滤掉时**清空池并走兜底**（用户 Q32）；=0 回退 2026-09-18 "
        "「不清空池」旧行为"),
    # ---- 2026-10-12 批次 5：定理注入块长上限 ----
    "theorem_hint_backend": (
        "THEOREM_HINT_BACKEND", "str", "leanexplore",
        "定理检索后端通道（2026-10-15 工具②接线）：leanexplore = 工具② "
        "tools/agent_toolkit.leansearch（LeanExplore 主 + 离线降级，实测 "
        "112 题命中 17.9%）；legacy = 原 MathlibTheoremSearcher（官方 API/源码扫描）"),
    "theorem_hint_block_max_chars": (
        "THEOREM_HINT_BLOCK_MAX_CHARS", "num", 2500,
        "定理注入块的字符上限（超限按相关度截断并标注未展开条数；"
        "用户 §5.3「注意力预算」）"),
    # ---- 2026-10-12 批次 5 剩余：「分环节条数」（用户 §5.2）----
    #   ⚠ 注册表**只支持 bool/int/num/str**（无 dict）⇒ 用 **JSON 字符串**承载；
    #     读取侧 `theorem_hint.run_hint_for` 解析为 dict 后写入
    #     `ctx._theorem_topk_by_stage`（2026-10-14 前该解析在 `orchestrator__lean`）。
    #   ★ 2026-10-14（任务列表 #2）：`subgoal` 8 → 25 —— 检索已下移到 2.7 子目标
    #     求解，该环节成为检索的主消费者（与 `solve` 齐平，对齐任务原文「这个
    #     环节的产出就是 25 个定理」）。`pre_audit` 键保留但已无调用者：2.6 内的
    #     两处注入（audit_gate / 理解链）已随检索下移撤除。
    "theorem_hint_top_k_by_stage": (
        "THEOREM_HINT_TOP_K_BY_STAGE", "str",
        '{"pre_audit": 4, "blueprint": 12, "subgoal": 25, "solve": 25, "revise": 25}',
        "分环节定理注入条数（JSON：前置理解 4 / 蓝图 12 / 子目标 25 / 主求解 25 / 修订 25）"),
    # ★ 2026-10-15：原 `enable_skeleton_review`（骨架编排层评审循环）与
    #   `enable_dag_replan`（求解后 DAG 复盘重规划）两个开关已整块撤除。
    # ---- agent/blueprint_planner.py（OR 节点展开策略，2026-10-04 补登记）----
    "blueprint_or_expand_all": (
        "BLUEPRINT_OR_EXPAND_ALL", "bool", True,
        "蓝图 OR 节点是否展开**全部**备选分支（默认 False = 只取首分支/保持历史行为；"
        "置 True 会显著放大子目标规模，需 A/B）"),
    # ---- 2026-10-15（用户要求「子目标承接 + 每个子目标设定工具」）----
    "blueprint_node_contract": (
        "BLUEPRINT_NODE_CONTRACT", "bool", True,
        "蓝图**节点契约**：① 每个节点声明 `tool`（lean/sympy/numeric/llm，证明类用 lean）；"
        "② `inputs[i].use` 声明「把上游哪个结论当成本步的什么条件」并下传为子目标"
        "`inputs_from`。开 ⇒ 求解侧按 tool 追加工具指令、并把承接前提注入 step 提示词；"
        "关 ⇒ 回退（tool/inputs 只落盘不兑现）"),
    # ---- 2026-10-15（用户：「把 lean 内化到蓝图构建里面」——叶子级形式化下传）----
    "blueprint_lean_sig": (
        "BLUEPRINT_LEAN_SIG", "bool", True,
        "蓝图**叶子级形式化下传**：`tool=lean` 的节点额外产出 Lean 4 形式化签名 "
        "`formal_sig`（`theorem/example … := by sorry`），下传给求解侧 —— "
        "分工变为「蓝图出**证什么**、求解出**怎么证**」；求解侧只补证明体，"
        "兑现层把签名与证明体拼接后交本地编译器。关 ⇒ `formal_sig` 不下传，"
        "求解侧自行写整段 Lean（历史行为；A/B 对照组）"),
    "blueprint_tool_exec": (
        "BLUEPRINT_TOOL_EXEC", "bool", True,
        "把蓝图声明的 `tool` **兑现成真正的机器核验**（lean 交本地编译器实编译 / "
        "sympy 判等式 / numeric 跑 norm_num）；未过带反馈重解**一次**，仍不过保留"
        "首解并留痕。关 ⇒ 只声明不执行（纯提示词增强，零机器开销）"),
    # ---- agent/sub_goal_solver.py 引理记忆（2026-10-04 补登记）----
    "use_lemma_accumulation": (
        "USE_LEMMA_ACCUMULATION", "bool", True,
        "引理积累/注入总开关（默认 True；实际注入还受 `lemma_domains` 领域路由约束）"),
    "lemma_domains": (
        "LEMMA_DOMAINS", "str", "Number theory,数论",
        "引理注入的**领域白名单**（逗号分隔）。默认仅数论 —— 依据本项目 A/B"
        "「数论 +23pp、代数/组合被拖累」；而 LEAP Table 6 认为代数也该受益 ⇒"
        "**两来源冲突，靠 `arm4` 裁决**。"
        "⚠ 消费方按**列表**读 `config.lemma_domains` ⇒ 走 `--arm`/AgentConfig 覆盖；"
        "env 路径需再补转换（当前 arm 驱动）"),
    # ---- agent/falsifier.py（单题证伪次数上限）----
    "falsify_max_per_q": (
        "FALSIFY_MAX_PER_Q", "num", 0.0,
        "单题最多证伪次数（**0 = 不限**；2026-10-04 默认 0。"
        "⚠ 语义已由旧的「0 = 禁用证伪」统一为「0 = 不限」）"),
    # ---- agent/orchestrator.py（阶段时间预算总开关）----
    "phase_budget_enabled": (
        "PHASE_BUDGET_ENABLED", "bool", False, "阶段时间预算总开关（默认关）"),
    # ---- user_agent.py / agent/orchestrator.py（题型惰性分类）----
    "enable_question_type": (
        "ENABLE_QUESTION_TYPE", "bool", True, "题型惰性分类总开关"),
    # ---- agent/orchestrator.py（零多样性补采，2026-10-06 新增）----
    # ---- agent/orchestrator.py（门全拒的处置策略，2026-10-06 新增）----
    "strict_all_rejected": (
        "STRICT_ALL_REJECTED", "bool", False,
        "Lean/Audit 门把候选**全部拒掉**时的处置：默认 False = 只显式留痕、"
        "不再静默保留（原实现 `if kept:` 在全拒时不更新候选池 ⇒ 门等于没作用）；"
        "置 True = 进一步清空候选池，触发下游兜底/重生成链"),
    # ---- agent/orchestrator.py（AcceptGate 真实生效，2026-10-07 新增）----
    # 背景（fix-gate-timing-diag 问题 2.3）：`base.RoundState` 契约阈值
    # ACCEPT_PASS=5 / REJECT_DEFECT=10 按"无上限连续回环"标定；但 `_update_accept_gate`
    # 唯一调用点在 `_deep_revise_loop_inner` 的每轮复验，受 `max_revise_rounds`(=5)
    # 约束 ⇒ 单题最多调用 ≤5 次（smoke10 实测 ≤2）⇒ 阈值**结构性不可达**、
    # decision 恒为 HOLD（10/10 题只出现 HOLD）⇒ AcceptGate 是空闸门；且 ACCEPT
    # 分支当时无消费者。本组开关把它变成可达且被消费的真闸门。
    # ---- agent/orchestrator.py（revise 耗时归属，2026-10-07 新增）----
    # ---- agent/solver.py（revise 候选数下限，2026-10-06 新增）----
    "revise_min_candidates": (
        "REVISE_MIN_CANDIDATES", "int", 3,
        "revise（纠错重解）每轮的**候选数下限**。原 count = 池剩余容量 ⇒ 随池增长单调递减"
        "（实测逐轮 4→2→1，最后一轮只试 1 个）⇒ 越到后面改对机会越少。设下限杜绝此退化。"
        "置 1 可回退旧行为做 A/B"),
    # ---- agent/solver.py（候选入池门槛，2026-10-06 新增）----
    "require_answer_in_pool": (
        "REQUIRE_ANSWER_IN_POOL", "bool", True,
        "候选入池前是否要求它**真的含一个最终答案**。默认 True：不含则不入池并留痕"
        "（实测 59% 的候选 `answer` 是推理句/元话语，入池后污染投票/检错/修正）。"
        "置 False 可回退旧行为（全部入池）做 A/B"),
    # ---- 候选入池门槛 v2（2026-10-07，问题 2.4 第二轮）----
    # 上一轮门槛只覆盖 `_generate_initial`/`_generate_revise` 主路径 → 实测仍有漏网：
    #   题 002 `'证明这个最小值就是所求的 $T$，并验证其最优性'` 入池并成唯一幸存候选
    #   （被 final_select③ 直接采用、predicted=该句）；题 000 `'子问题4：计算满足条件
    #   的路径总数'`（自改进覆盖）/`'{(x,y)|…}，共 20 × 3 = 60 个元素'`（revise 主路径
    #   判据漏网）；题 007 `'- 子问题4：求和并验证结果'`。同时**误杀**真答案：
    #   题 087 的 `BD`（gold `B, D`）、题 088 的 `AB` 各被拒 4/5 次（trace 实证）。
    # 本开关 = 「穷举全部 6 处写入入口 + 判据升级（补漏网/修误杀）」的**总闸**。
    "pool_gate_v2": (
        "POOL_GATE_V2", "bool", True,
        "候选入池门槛 v2（默认 True = 修好后行为）：① 判据换 `looks_like_pool_answer`"
        "（补 4 条规划/设定句漏网：题 000/002/007；修 9 次多字母选项误杀：题 087/088；"
        "修判断题 gold `正确`/`错误` 误杀：题 101/111）；② 穷举全部 6 处写入 "
        "`ctx.candidates` 的入口（证明通道 / 生成失败抢救 / 自改进覆盖 / 截断续写 / "
        "子目标候选 / 三 Agent 协作），不再只覆盖主路径。"
        "⚠ 仍有 5/112 条**自然语言 gold**（填空题/论述题：037/095/104/105/108）会被判 "
        "False，需题型感知改造（本次未做，见交付说明）。=0 完全回退上一轮口径做 A/B"),
    # ---- agent/sub_goal_solver.py（`<check>` 协议守卫，2026-10-07 问题 2.9）----
    "check_protocol_guard": (
        "CHECK_PROTOCOL_GUARD", "bool", True,
        "`<check>` **协议守卫**（默认 True = 修好后行为）：模型把**中文句子**塞进 "
        "`<check>`（实测题 000 大量 `<check>图 G 是弱连通的</check>`）⇒ 不可解析 "
        "⇒ 客观判定一条都出不来（`n_judged = 0/73` 的直接原因之一）。打开后："
        "① 判定 `<check>` 内容是否为可核验对象（纯数学等式）；② 违反则**留痕**"
        "（trace `check_protocol` + sg 字段进 diag，不再静默落 undecided）；"
        "③ **触发一次定向重问**（带正确写法反馈）。=0 回退旧行为（静默跳过）做 A/B"),
    # ---- agent/verifier.py（2026-10-04 B 折中：投票放开完整推理的档位）----
    "verifier_vote_reasoning": (
        "VERIFIER_VOTE_REASONING", "str", "disagree",
        "投票是否放开完整推理（不 prefill）：off=全部 prefill（旧行为，0.8s/票但"
        "「看一眼就点头」）/ **disagree=默认**（仅候选分歧时放开，把算力花在拿不准的题上）"
        " / always=每票都放开完整推理（最准，约 70s/票）。用户口径：先试 disagree，"
        "效果不理想即切 always（正确率第一位）"),
    # ---- agent/orchestrator.py（Step 4 反馈复核的两处修复，2026-10-06 新增）----
    # 背景（审计已确认）：① 一次实测灌入 157,550 字符反馈（`:2759`
    #   `_review_bug_feedback` 把模型裸输出不加限直接当反馈返回）；
    #   ② 该函数把 verifier 保留的深复核推理**整段重新摘要成短清单**再覆盖
    #   `ctx.revise_feedback` ⇒ 两处修复互相抵消、推理始终没到模型手上。
    # ---- agent/deterministic.py（2026-10-07「审计闸门对解答题结构性无输出」修复）----
    "deterministic_solution_check": (
        "DETERMINISTIC_SOLUTION_CHECK", "bool", True,
        "解答题确定性判别重写总开关。=1（默认，修好后的行为）：① 解析前归一答案"
        "（剥 `\\boxed{}`/`$`/LaTeX 外壳 —— 实测 smoke10 有 7/10 题因带壳在 method=\"parse\" "
        "即返回 unknown，Level0 根本没进入判据）；② 常量等式回代由「只取最后一条」"
        "改为「全量等式合取判定」；③ 新增**函数方程回代**判据（覆盖「求满足函数方程的"
        "函数」这一解答题子类，对候选函数做符号代入 + 恒等判定）。=0 完全回退改动前"
        "（raw parse、只取最后一条等式、无函数方程判据）做 A/B"),
    # ---- agent/audit_gate.py（2026-10-07 同上）----
    "audit_gate_rubric_nonproof": (
        "AUDIT_GATE_RUBRIC_NONPROOF", "bool", False,
        "是否允许 AuditGate 的 Level2 rubric 结构化判分对**解答题**（非证明题）运行。"
        "旧行为：rubric 仅对证明题运行 ⇒ 解答题 Level2 结构性不适用。经查 rubric 提示词"
        "（VERIFIER_RUBRIC_SYSTEM）为通用「数学解答评审」，并无证明专用假设 ⇒ 原限制缺少"
        "依据。**默认 False（保守）**：rubric 系 LLM 同源自评（价值低、耗时高），且候选级"
        "投票链本已可选跑 rubric；开启前应做 A/B。=1 则解答题也走 Level2（仍沿用高置信 B "
        "才否决的阈值）"),
    # ---- 2026-10-07 开关注册制补齐（fix-switch-debt：6 个账外 gate 型 env）----
    # 背景（问题总账 §2.6 / 原代码审查 A 级第 2 条）：下列 6 个 env 此前仍以裸
    # `os.environ.get` / `os.getenv` 读取（读取处见各条注释），既不在本注册表、也不在
    # `AgentConfig` / 白名单 ⇒ **改了不生效、生效查不到**。本批**只接线，不改默认值、
    # 不改行为** —— 每条的默认值**逐字照抄现存 env 读取处的字面默认**。
    # ⚠ 「读取侧」大多不在本次改动范围内（见交付清单）：读取处若仍是 `os.environ.get`
    #   或模块级常量，env 能生效（优先级不变）但 config 覆盖读不到，需改为实时读注册表。
    # ★ 2026-10-15（用户定案「删直答兜底」）：以下 3 个直答专用字段已随
    #   `_emergency_direct_solve` / `_emergency_answer` 删除，无消费者，四表同步移除。
    "deep_max_rework": (
        "DEEP_MAX_REWORK", "num", -1,
        "6.5 审核重做的候选数上限（deep 档）。**-1 = 无上限**（研究期默认；比赛期曾为 3）。"
        "standard 档恒为 2，不受本开关影响。"
        "读取处 agent/orchestrator.py:2508；param_usage.gate_max_rework 亦消费本开关"),
    # ---- 2026-10-14 五道审核链（用户定案：规范化前置→①自审→②证伪→③Lean
    #      代回→④计算检测→⑤投票；旧审核链同批删除）----
    #      每道独立子开关 ⇒ 评测后可逐道 A/B 归因（「一次只改一项」纪律）。
    "five_gate_normalize": (
        "FIVE_GATE_NORMALIZE", "bool", True,
        "4.0 答案规范化前置（五道审核链第 0 道）：候选进验证链前统一归一，"
        "挂 answer_norm/answer_kind 供 Lean/计算工具/投票消费；"
        "不改 c.answer（旧链行为零变化）；=0 关闭做 A/B"),
    "five_gate_self_audit": (
        "FIVE_GATE_SELF_AUDIT", "bool", True,
        "4.1 AI 自审逻辑错误（五道审核链第 1 道）：逐候选 LLM 逻辑自审"
        "（只审逻辑不审数值），能指出具体逻辑错误者淘汰并反馈；"
        "输出不可解析/LLM 故障一律保留候选（防误杀）；=0 关闭做 A/B"),
    "five_gate_falsify": (
        "FIVE_GATE_FALSIFY", "bool", True,
        "4.2 AI 对抗证伪（五道审核链第 2 道）：逐存活候选两层对抗验证——"
        "机检证伪（SymPy 反例）+ LLM 对抗生成；证伪即判错淘汰；"
        "跳过/故障一律保留候选（防误杀）；=0 关闭做 A/B"),
    "audit_obligation_ledger": (
        "AUDIT_OBLIGATION_LEDGER", "bool", True,
        "审核侧义务账本（2026-10-20）：把蓝图倒推链反演成「本题应建立哪些事实」"
        "的清单，注入 4.1 自审与 4.2 证伪 prompt；审查者指认的义务在清单里"
        "查不到 ⇒ 视为评论了蓝图未要求的内容，不淘汰候选。"
        "=0 退回「只喂题目/解答/答案」的老口径，用于 A/B"),
    "five_gate_lean_check_tool": (
        "FIVE_GATE_LEAN_CHECK_TOOL", "bool", False,
        "4.3 工具①观测通道（2026-10-15 工具化定稿）：开启后对存活候选调 "
        "tools/agent_toolkit.lean_check 做第二意见（错误点/缺少点写 diag，"
        "暂不改淘汰）；=1 开启（云端 A/B）"),
    "five_gate_lean_check": (
        "FIVE_GATE_LEAN_CHECK", "bool", True,
        "4.3 Lean 逻辑判断·答案代回题目（五道审核链第 3 道）：候选级 "
        "LeanGate.apply 硬验证（题目+答案翻译为 Lean 命题编译判定）；"
        "部分淘汰落池、全拒标记不静默清池、不适用题豁免；=0 关闭做 A/B"),
    "five_gate_audit_retry": (
        "FIVE_GATE_AUDIT_RETRY", "bool", True,
        "五道链·自审解析失败重问（2026-10-22 审计修复）：4.1 输出不可解析时"
        "按最严格式重问一次（不只输出 JSON 就再试），仍失败才记 unparsed 并"
        "标 verify_audit=unparsed；旧口径「一次不成即放行」让 4.1 长期空转"
        "（历史实测 84/88、12/12 候选 unparsed）；=0 回到旧口径"),
    "lean_failure_kind_split": (
        "LEAN_FAILURE_KIND_SPLIT", "bool", True,
        "4.3 Lean 编译失败按原因分级（2026-10-22 审计修复）：只有 "
        "failure_kind=refuted（Lean 明确输出 `⊢ False`）才判「答案错」并淘汰；"
        "unproved（代码合法但没证出）/ translation（生成的代码不合法）/ env"
        "（环境缺依赖）与答案对错无关，改判 unknown 并标 degraded=lean_<kind>；"
        "=0 回到旧行为（proof_invalid 一律淘汰），供 A/B 对照"),
    "blueprint_subtree_regen": (
        "BLUEPRINT_SUBTREE_REGEN", "bool", True,
        "蓝图自审用局部子树重写（2026-10-22 接线回补）：regenerate_subtree 只重写"
        "被拒节点的 LCA 子树、其余保留，且 LCA 触根时自动退化为整树；=0 回到纯整树重写"),
    "subgoal_failed_retry_rounds": (
        "SUBGOAL_FAILED_RETRY_ROUNDS", "num", 2,
        "子目标未产出有效结论时的重做轮数（2026-10-22）：实测 628 个子目标里 "
        "135 个失败（21.5%），其中 95 个是空返回——旧实现只在结果以 "
        "`[子目标` 开头时才重试一次，空串不匹配 ⇒ 从不重试，空结果直接进 merge。"
        "现改为空/占位一律重做 N 轮（带「上次为空」提示），仍空则落显式占位符"),
    "five_gate_falsify_votes": (
        "FIVE_GATE_FALSIFY_VOTES", "num", 3,
        "4.2 对抗证伪采样次数（2026-10-22 实测修复）：同一错误候选连跑 3 次探测"
        "得「证伪/放过/放过」⇒ 单次采样漏检约 2/3。现采 N 次，任一次 actionable "
        "且证据落地即淘汰，过程瑕疵/跳过只在全部未命中时计入；=1 回到旧行为"),
    "five_gate_self_audit_votes": (
        "FIVE_GATE_SELF_AUDIT_VOTES", "num", 3,
        "4.1 自审采样次数（2026-10-22 实测修复）：同一个错误候选连跑 3 次"
        "判定为「留/杀/留」⇒ 单次采样不稳定、漏杀率约 2/3。现采 N 次，"
        "任一次给出**带落地证据**的逻辑错即淘汰（编造位置被证据门挡住）；"
        "=1 回到旧行为（单次判定），供 A/B 对照"),
    "five_gate_verify_tiebreak": (
        "FIVE_GATE_VERIFY_TIEBREAK", "bool", True,
        "4.5 投票平票时以 4.4 机器核验为依据（2026-10-22）：4.4 把 "
        "verify_math 写在候选上却无消费方 ⇒ 检测结论到不了选择层。开启后票数"
        "并列时优先选「逐式复算全一致」的答案；=0 回到只按形态分/长度"),
    "five_gate_grounding": (
        "FIVE_GATE_GROUNDING", "bool", True,
        "五道链·证据落地校验（2026-10-22 审计修复）：4.1/4.2 的淘汰必须附"
        "**可机器复核**的证据——模型指认的出错位置/出错步骤必须是候选推理里"
        "逐字存在的原文片段，系统按归一化子串核对；编造的引文判为无效证据，"
        "不得据此淘汰（写 self_audit_ungrounded / adversarial_gate_ungrounded"
        " 留痕）；=0 回到旧口径（只认描述非空），仅用于 A/B 对照"),
    "five_gate_math_check_calcverify": (
        "FIVE_GATE_MATH_CHECK_CALCVERIFY", "bool", False,
        "4.4 工具③观测通道（2026-10-15）：开启后对存活候选推理提取数值等式、"
        "逐条 calc_verify 复算（算错条目写 diag，暂不改淘汰）；=1 开启（云端 A/B）"),
    "five_gate_math_check": (
        "FIVE_GATE_MATH_CHECK", "bool", True,
        "4.4 数学工具计算检测（五道审核链第 4 道）：候选级 SymPy 精确判定"
        "（DeterministicChecker：等式回带/函数方程回代/反例搜索）；"
        "fail ⇒ 淘汰、unknown ⇒ 放行、全拒 ⇒ 回退保留（宁 unknown 不误杀）；"
        "=0 关闭做 A/B"),
    "five_gate_vote": (
        "FIVE_GATE_VOTE", "bool", True,
        "4.5 投票（五道审核链第 5 道）：先多答案判断（规则+LLM，写 "
        "ctx.vote_multi_answer 单一来源）→ 大模型对存活候选逐候选投票"
        "（票数锁定 tier_votes 多票口径，绝不降为 1）；=0 关闭做 A/B"),
    "five_gate_subgoal_lean": (
        "FIVE_GATE_SUBGOAL_LEAN", "bool", False,
        "子目标级 Lean 断言验证（2026-10-23 调研 AlphaProof/Nexus 后新增）："
        "从最后几个子目标的结论里挑出**可判断言**（等式/不等式/极限），"
        "让 LLM 转成 Lean 定理并编译；编译不过 ⇒ 该断言不成立，作为上游可疑信号。"
        "默认关（成本与命中率待探针验证）；=1 开启"),
    "five_gate_rewrite": (
        "FIVE_GATE_REWRITE", "bool", True,
        "4.6 重写通道（五道审核链配套，2026-10-14 用户定案）：池空 ⇒ 汇总错误"
        "情况交大模型诊断根因 → 重建蓝图和子目标（rewrite_context 注入题目"
        "占位符内部）→ 重写 → 新候选重过五道链；轮数上限 rewrite_max_rounds；"
        "**取代「池空直接要答案」**；=0 关闭做 A/B"),
    "objective_itemwise_plan": (
        "OBJECTIVE_ITEMWISE_PLAN", "bool", True,
        "子目标**规划**阶段的客观题逐项策略（=0 关闭）。"
        "读取处 agent/sub_goal_solver.py:1376"),
}

_OFF = {"0", "false", "no", "off", ""}
_overrides: Dict[str, Any] = {}


def bind_config(cfg) -> None:
    """把 AgentConfig 的**当前取值**绑定进注册表（在 Agent 初始化时调一次）。

    ★ 必须**每次覆盖**（不再"等于默认值就跳过"）：
    同一进程内会先后构造多个 Agent（单测尤其常见）；若跳过默认值，前一实例
    绑定过的**非默认值会残留**在 `_overrides` 里、污染后一实例
    ⇒ 表现为「后一实例静默地用了前一实例的开关值」（2026-10-02 实测踩到）。

    每次覆盖后：单次绑定语义**不变**（默认值绑进去 ⇒ 取值仍等于默认值），
    只是把「多实例泄漏」修正为「正确」——属修复，非回归。

    读取优先级始终是 **环境变量 > 覆盖值 > 默认值**（env 最高优先，行为不变）。
    """
    if cfg is None:
        return
    for field in SWITCHES:
        val = getattr(cfg, field, None)
        if val is None:
            continue
        _overrides[field] = val


def bind_field(field: str, value: Any) -> None:
    """把**单个字段**的当前值绑定进注册表（供按字段读取的模块使用）。

    与 `bind_config` **同口径**：每次都覆盖 `_overrides[field]`（`value is None` 除外）。
    因为它是被**逐实例反复调用**的（同一进程内可能先后构造多个 Agent / 多次读取），
    必须支持「后一次把前一次的值重置回默认」，否则上一实例的值会**泄漏到下一实例**
    （实测踩到：`minimal_mode=True` 的实例会污染随后构造的默认实例）。

    读取优先级始终是 **环境变量 > 覆盖值 > 默认值**，本函数不会抬高 config 的优先级。
    """
    if field not in SWITCHES:
        return
    if value is None:
        return
    _overrides[field] = value


def apply_env_to_config(cfg) -> None:
    """把**环境变量里确实存在**的开关写回 config，兑现「env > config > 默认值」。

    为什么必须有这一层（2026-10-09 缺陷复现）
    =========================================
    本模块白纸黑字承诺读取优先级 **env > AgentConfig 覆盖 > 默认值**，但此前只做了
    两段接线：
      · `kwargs → config`   （`ReasoningAgent.__init__`）
      · `config → registry` （`bind_config`）
    **缺 env → config 这一段** ⇒ 对**只被 `getattr(self.config, X)` 读取的开关**，
    env 完全无效（注册表认了 env，消费方却读 config 的旧值）：
        env DIVERSITY_RESAMPLE=0 → 注册表 False、`config.diversity_resample_enabled` True
    连带的第二处缺陷：`agent/param_usage.py::_SPEC` 的 cap_fn 全是
    `getattr(cfg, _f, _d)` ⇒ 上报的是 **config 值而非生效值**（同一份 diag 内
    `switch_snapshot` 与 `param_usage` 自相矛盾）。

    本函数即补上缺失的那一层：env 存在时把值写回 config 字段，使
    `getattr(config, 字段)` 恒等于注册表的生效值（两层缺陷一并消解）。

    ★ 只写 env 里**真实存在**的项 ⇒ env 缺失时 config 的显式值原样保留，
      不破坏「config > 默认值」这一半语义。
    ★ 必须**直接读 os.environ**，不走 `get_bool/get_num/get_str` —— 后者会先看
      进程级 `_overrides`，而在 `__post_init__` 时 `bind_config` 还没跑，残留的
      `_overrides`（同一进程内多次构造 AgentConfig 时会串味）会污染结果。
    ★ 整段 try/except 包住，**绝不抛异常** —— 构造 AgentConfig 绝不能因 env 脏而失败。
    ★ 按 kind 做**类型正确**的转换；`str` 型字段若当前值是 list/tuple（如
      `lemma_domains`），按逗号切分为列表，否则 `list("A,B")` 会退化成逐字符列表。
    """
    try:
        for field, meta in SWITCHES.items():
            env_key, kind = meta[0], meta[1]
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if not hasattr(cfg, field):
                continue
            try:
                if kind == "bool":
                    # 与 get_bool 同口径：0/false/no/off/""（忽略大小写）= 关。
                    val: Any = str(raw).strip().lower() not in _OFF
                elif kind == "int":
                    # ★ 落在 config 上的值必须是 **int**（`getattr(config, 字段)`
                    #   的消费点按整数用：池上限/候选数下限等；且 `snapshot()` 对
                    #   kind=="int" 也返回 int）。失败（非数字）→ 跳过，保留 config 现值。
                    val = int(float(str(raw).strip()))
                elif kind == "num":
                    val = float(str(raw).strip())
                else:  # "str"
                    s = str(raw)
                    if not s:
                        # 与 get_str 一致：空串视作「未设置」，保留 config 现值。
                        continue
                    cur = getattr(cfg, field)
                    if isinstance(cur, (list, tuple)):
                        val = [p.strip() for p in s.split(",") if p.strip()]
                    else:
                        val = s
            except (TypeError, ValueError):
                continue
            try:
                setattr(cfg, field, val)
            except Exception:  # noqa: BLE001  单个字段写失败不影响其余
                continue
    except Exception:  # noqa: BLE001  桥整体失败绝不冒泡
        return


def _raw(field: str) -> Optional[str]:
    env_key = SWITCHES[field][0]
    v = os.environ.get(env_key)
    return None if v is None else str(v).strip()


def get_bool(field: str) -> bool:
    """读布尔开关（env > config 覆盖 > 默认值）。

    ★ 2026-10-04 修（D28）：`config 覆盖` 分支原为 `bool(_overrides[field])`，
    于是经 `bind_field` 传入**字符串** `"0"` 时 `bool("0") is True`
    ⇒ **"关"被读成"开"**（且 `get_str` 对空串的口径也不同）。
    现对**字符串**覆盖复用与 env 同一套 `_OFF` 语义；非字符串仍按 truthiness
    （`bind_config` 绑进来的是真 bool，行为不变）。
    """
    raw = _raw(field)
    if raw is not None:
        return raw.lower() not in _OFF
    if field in _overrides:
        ov = _overrides[field]
        if isinstance(ov, str):
            return ov.strip().lower() not in _OFF
        return bool(ov)
    return bool(SWITCHES[field][2])


def get_num(field: str) -> float:
    raw = _raw(field)
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    if field in _overrides:
        try:
            return float(_overrides[field])
        except (TypeError, ValueError):
            pass
    return float(SWITCHES[field][2])


def get_str(field: str) -> str:
    raw = _raw(field)
    if raw:
        return raw
    if field in _overrides and _overrides[field]:
        return str(_overrides[field])
    return str(SWITCHES[field][2])


def source_of(field: str) -> str:
    """该开关的生效值来自哪里：env / config / default。"""
    if _raw(field) is not None:
        return "env"
    if field in _overrides:
        return "config"
    return "default"


def snapshot() -> Dict[str, Dict[str, Any]]:
    """当前全部开关的生效值快照（挂进 diag ⇒ 报告可还原）。"""
    out = {}
    for field, (env_key, kind, default, doc) in SWITCHES.items():
        if kind == "bool":
            eff: Any = get_bool(field)
        elif kind == "num":
            eff = get_num(field)
        elif kind == "int":
            # ★ 2026-10-09 修（同源另一处类型错误）：`int` 型开关此前落进 `else`
            #   走 `get_str` ⇒ `switch_snapshot.value` 是**字符串** `"3"` 而非整数 `3`。
            #   危害：`switch_snapshot` 是运维/复盘判断"开关当时生效成什么"的唯一权威
            #   读数，类型错了会让 `value == 3` 恒为假、或被当字符串拼进报告。
            #   此处显式取数值并**保持 int 类型**（`get_num` 走 float 分支再 `int()`，
            #   不会把 `bool` 混进来 —— bool 虽为 int 子类，但不经此路径）。
            eff = int(get_num(field))
        else:
            eff = get_str(field)
        out[field] = {
            "env": env_key,
            "value": eff,
            "source": source_of(field),
            "default": default,
            "doc": doc,
        }
    return out


def summary_line() -> str:
    """一行摘要，便于写进 trace/日志。"""
    env_n = sum(1 for f in SWITCHES if source_of(f) == "env")
    cfg_n = sum(1 for f in SWITCHES if source_of(f) == "config")
    return (f"注册开关 {len(SWITCHES)} 个："
            f"来自环境变量 {env_n}、来自配置覆盖 {cfg_n}、"
            f"其余 {len(SWITCHES) - env_n - cfg_n} 用默认值")
