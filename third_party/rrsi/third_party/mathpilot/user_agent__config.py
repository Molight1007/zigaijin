# -*- coding: utf-8 -*-
"""平台入口配置类 `AgentConfig` —— 从 user_agent.py 逐字搬出（2026-10-08 拆分）。

对外契约不变（同一对象）：``from user_agent import AgentConfig`` 仍可用
（user_agent.py 原位 re-export）。
★ 无模块级 agent.* 依赖（只 import stdlib）⇒ 不破坏 user_agent 的降级导入契约。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from user_agent__config_base import _AgentConfigBase

# ============================================================
# 配置
# ============================================================
@dataclass
class AgentConfig(_AgentConfigBase):
    """智能体可调参数（选手可自由优化）
    
    适配竞赛规则 v3（2026-07-31）：
    - 并发=3，单题最长=20分钟，Agent最长=6小时
    - 反rollout：减少候选数与投票次数，依赖聚类共识而非暴力采样
    """
    # ---- 智能体补充部件配置 ----
    # v2.4.0：max_tokens/cap 同步 24576（ICMA reasoning 同款上限，模型实际用 3-7K token）
    # 2026-09-04：max_tokens_cap=0 关闭 base.llm 二次裁剪
    max_tokens_cap: int = 0            # 内部 token 裁剪上限：0=不裁剪（base.llm 语义）
    # ★ 2026-10-20：输出**保底**上限。推理模型先写 Thinking 再写正文，
    #   ≤2048 的"短输出"设计全部会被思考吃掉（实测 212 次截断的主因）。
    #   0 = 关闭保底；默认只抬不降（不影响投票这类短任务的速度）。
    # ★ 2026-10-22 4096 → 16384：端到端实测发现 4096 **完全不够**——
    #   official112-008 / 001 这类题上，推理模型的思考会吃光整个上限、
    #   正文输出 0 字（实测 max_tokens=16384 → reasoning=16384、content="";
    #   甚至 32768 也被吃光）。4096 在这种题上必然"无有效解答"。
    #   max_tokens 是**上限不是预留**，抬到 16384 不会让短调用变慢。
    llm_min_max_tokens: int = 16384
    max_workers: int = 3               # 并发验证线程数（匹配系统并发度=3）
    # ★ 2026-09-16 删除两个**死字段**（全仓 0 读取点，已实测确认）：
    #   · `max_tokens`  —— 实际生效的是 `max_answer_tokens`（solver 侧）
    #      与 `policy_max_tokens`；此处那份从未被读。
    #   · `temperature` —— 各调用点都显式传字面量（如 user_agent 直答
    #      `temperature=0.0`、verifier 走 `verifier_temperature`），此处那份从未被读。
    # 保留它们只会给人"可调"的假象（改了不起作用），属"老代码逻辑堆叠"。

    # ---- 自纠错参数 ----
    max_answer_tokens: int = 65536    # solver 单次调用最大 token 数（9/4 放开，防答案腰斩）
    revise_sample_times: int = 2       # 自纠错重解候选数
    # ★ 2026-10-06 新增：revise 的**候选数下限**。原 count = 池剩余容量，而每轮都往池里
    #   加候选 ⇒ 剩余容量单调递减 ⇒ 实测 `count` 逐轮为 4→2→1（最后一轮只试 1 个）。
    #   设下限保证"越到后面越没有改对机会"这个退化不再发生。
    revise_min_candidates: int = 3

    # ---- 新功能开关（简化）----
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    use_scoring: bool = True           # Verifier 用多维评分（原：不用/简化，减少误判）
    enable_deterministic: bool = True  # 确定性硬否决（v2.8）：SymPy 代入回验 fail 淘汰候选、unknown 放行
    # 2026-09-06（移植自 sq 分支，默认关，A/B 验证后开）：
    use_rubric: bool = False           # Verifier rubric 结构化判分（verdict+confidence+错因定位，JSON prefill）
    use_challenge: bool = False        # Verifier 反例挑战（LLM 命题 → SymPy 程序数值验证 → hard_fail 否决）
    # 2026-09-29：`by_enable_fast_path` 已删除 —— 快车道（_fast_path）整体移除，
    # 该字段本就是真死开关（全仓唯一读取点是 logger，不控制行为，审计报告 §四）。
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    use_proof_channel: bool = True     # 证明题专用通道（原：关闭/简化）
    use_lemma_accumulation: bool = True  # 引理积累（2026-08-29 起默认开，按领域路由）
    lemma_domains: list = field(default_factory=lambda: ["Number theory", "数论"])  # 领域路由：A/B 实测数论 +23pp、代数/组合被拖累
    # 2026-10-01 删除 `use_sub_goal`：档位时代遗留，声明+白名单齐全但全仓 **0 读取点**
    # （审查 A 级第 2 条 / B 级死配置）。真正生效的是 `deep_use_sub_goal`（见下方 :485）。
    # Step 2 无条件自改进（2026-08-29 新增，依据 IMO2025 验证-精炼论文）
    # 论文流水线六步中的 Step 2：初始解生成后**无条件**先 review+improve 一次
    # （注入第二段推理预算），再进入验证。论文实测：初始解质量低，此步显著改进。
    # 区别于 revise（验证失败才修正），自改进对每个候选都做一遍。
    enable_self_improve: bool = True
    self_improve_max: int = 2          # 每题最多自改进候选数（3→2 降本；见下）
    # ---- 2026-09-29 用户决策（截图 #7）：「无条件自改进是一遍还是两遍？哪种效果
    #      最好，需要尝试。」⇒ 恢复**真无条件**（默认跳过缺陷过滤），并把遍数做成
    #      可配置，供后续 A/B 对比。
    # · `self_improve_rounds`：自改进轮数。1 = 旧单遍；2 = 对同一批候选连做两遍
    #   （第二遍基于第一遍结果再注入一次推理预算）。默认 1（保守），实验设 2。
    # · `self_improve_conditional`：是否启用「只改有缺陷候选」的过滤。
    #   **默认 False = 真无条件**（用户要求）。设 True 恢复 2026-09-11 的条件化。
    #   ⚠ 历史证据：smoke6_v3 实测条件化前「无条件」正确定 1/6 不变、耗时 +34%，
    #   且 098 被从正确答案改错。用户已知悉，要求在新配置（统一 deep 档 +
    #   原版保留 + 下游投票择优）下重新验证 —— 本开关即为该验证的 A/B 抓手。
    # ★ 2026-10-13（用户 E24 明确：「让大模型反复写一道题」）：默认 1 → **3**。
    #   语义见上：3.3 环节对同一批候选**无条件**连做 3 遍自改进（不看有无增益）。
    #   ⚠ 成本随之上升（每遍都是一次全候选 LLM 调用）⇒ 由 `n_self_improved`
    #     与 `self_improve_rounds_used` 可观测；=1 可一键回退旧行为。
    self_improve_rounds: int = 3
    self_improve_conditional: bool = False
    # ---- 2026-09-30 用户决策（截图 #8）：子目标多 agent 求解 ----
    # 用户原话：「多 agent 拿子目标的求解能不能换成多 agent？那样是不是能提高
    #           子目标的正确率？」
    # 实现：对同一子目标独立采 N 个解，按**归一化关键值的一致性**选代表解
    #      （self-consistency 下沉到子目标粒度）。中间步骤错一个后面全错，
    #      故此处收益理论上高于只在最终答案层投票。
    #   · `enable_subgoal_multi_agent`：★ 2026-10-04 由 **False 改为 True**（D4）。
    #     依据：审核确认「机制已完整实现但默认不生效 = 空转」，且用户口径是
    #     「研究期用全部方法、正确率唯一目标」。子目标多采样正是为"提高子目标
    #     正确率"而建（用户 2026-09-30 点名）。A/B 仍可用
    #     `ENABLE_SUBGOAL_MULTI_AGENT=0` 取回旧基线（开关已注册，可还原）。
    #   · `subgoal_agents_n`：每题每个子目标采样次数（上限 8）。
    #     ⚠ 成本是 N 倍单步生成 —— 时间紧迫/预算不足时实现里会自动退回 1 次。
    enable_subgoal_multi_agent: bool = True
    subgoal_agents_n: int = 3
    # ---- 领域 → 定理检索（2026-09-29 新增，截图 #3+#4，老师重点关注）----
    # 用户诉求：判断题目属于哪个领域、该用什么定理，用 leansearch 去 Mathlib
    # 检索对应定理，并评估「定理对大模型的推理效果」。
    #   · 检索实现收敛在 agent/theorem_hint.py 单一入口（便于解耦与替换）；
    #   · enable_theorem_hint 是 A/B 抓手：设 False = 不检索、不注入
    #     ⇒ 与 True 对比即可回答「定理对推理有没有帮助」；
    #   · theorem_hint_max_queries 控制每题最多发几次 leansearch 检索。
    #   ★ 2026-10-14（任务列表 #2）：**执行点已下移到 2.7 子目标求解体首**
    #     （原 `1.2_theorem_hint` 独立阶段已撤），入口
    #     `agent.theorem_hint.run_hint_for(agent, ctx)`；A/B 开关语义不变。
    enable_theorem_hint: bool = True
    theorem_hint_max_queries: int = 2
    # ★ 2026-10-15（工具②接线）：定理检索后端通道。leanexplore（默认）=
    #   工具② tools/agent_toolkit.leansearch；legacy = 原 MathlibTheoremSearcher。
    theorem_hint_backend: str = "leanexplore"
    # ---- 子目标结论 → 主求解 信息流（2026-10-02 阶段一，用户拍板设计）----
    # 用户原话：「…主求解一定要在子目标的基础上。」
    # 把 2.7/3_solve(P&E) 已求得的子目标结论（ctx.subgoal_trace）作为**中间数据**
    # 注入主求解 prompt（初始 / 证明 / 重解三路径），让主答案建立在子目标之上。
    # 边界：只改信息流，不改投票/采样/终答/阈值。
    # 默认 True；设 False = 三路径均不注入（A/B 抓手）。已登记 switch_registry。
    enable_subgoal_findings: bool = True
    # ---- 终答五层选择（2026-10-02 阶段二-2，用户拍板：「投票应在最后，不要一开始就投票」）----
    # 用户原话：「把 n 个答案都给大模型让它来判断」「要按照正确率选取」。
    # ON 时终答由 agent/final_selector.select_final_answer 统一选出
    # （②多答案判断→③客观验证→④模型对比→⑤投票兜底），
    # 不再走 formatter._rank_key 票数优先 / objective_majority_vote「一开始就投票」。
    # 默认 True（用户明确要此行为）；设 False = 全部回退改动前（A/B 抓手）。已登记 switch_registry。
    enable_final_answer_selection: bool = True
    # ---- 终答选择两条阻断级修复（2026-10-07，smoke10 2026-10-06 实测驱动）----
    # ① 决策记录与最终产出单一来源：把 `_pick_diag.picked` 同步为逐字提交值。
    #    依据：official112-002 的 pick_diag.picked=计划句、predicted=`\boxed{...}`。
    # ② 选择机制识别"未验证候选"：无验证记录者不参与胜出（全部未验证时留痕放行）；
    #    ⑤投票并列收尾不再用推理长度。依据：official112-007 未验证的 `\boxed{0}` 胜出。
    # 两者默认 True（= 修好后行为）；设 False 回退做 A/B。均已登记 switch_registry。
    final_answer_single_source: bool = True
    final_select_verify_aware: bool = True
    # 依据：3.3 无条件改进实测只有成本没有收益（smoke6_v3：占单题耗时 29~45%、
    # 总耗时 +34%，正确率 1/6 不变）。配合 A2 条件化（只改有缺陷的候选）进一步压成本。
    improve_min_remaining: float = 300.0  # 3.3 改进停手预留（距生成软截止 < 此值不再开新候选；0=关，治 alg-060 3.3=640s 烧穿）
    # 易错点记忆注入（2026-09-06 A 档轻量经验，prompts/error_lessons.py）：
    # 命中题型/关键词时把历史易错自查清单拼进初始生成与 revise 提示，防重复踩坑。
    # 默认开（本地评测生效）；A/B 对照可 --override enable_error_lessons=False。
    enable_error_lessons: bool = True
    # Step 4 bug report 复核（2026-08-29 晚新增，论文流水线 Step 4）
    # 验证器给出缺陷反馈后，让模型先复核反馈是否属实、可驳回误报——
    # 论文：模型可驳回验证器的错误反馈，避免好答案被误报引导改坏。
    # 仅 deep 档 revise 回环触发（每次 +1 次 LLM 调用，预算可承受）。
    # ★ 2026-10-06 修两处已确认缺陷（见 agent/orchestrator.py:2759 `_review_bug_feedback`）：
    #   ① 复核产物**附加**而非覆盖：verifier 已把深复核推理（`【复核依据（判错报告
    #      原文摘录）】`，上限 8000）保留进 feedback，但复核会把它整段摘要成短清单
    #      再覆盖 `ctx.revise_feedback`（实测 q1–q6 最长仅 2.4–3.0k、且不含该段）
    #      ⇒ 推理始终没到模型手上。默认 True = 复核只做判定/标注、以 `【复核意见】`
    #      段并入；设 False 回退旧的「整段覆盖」做 A/B。已登记 switch_registry。
    #   ② 复核产物**长度上限**（保头截断）：实测 official112-007 单题返回了
    #      157,550 字符裸输出并被原样拼进提示词。默认 8000，与 verifier 的复核反馈
    #      同口径（verifier.py:1821 结构化分支 `[:8000]`）。设 0 = 不截断。
    #   ③ 复核产物**越权结论/重做正文清洗**：实测题 000 复核者输出「正确答案应为 0。
    #      下面给出严谨证明。…（2000+ 字符重做正文）」被模型整段照抄，最终答 0 而
    #      标准 20460。默认 True = 复用 `verifier._clip_review_prose` 同一道清洗
    #      （不复制正则，避免两处口径分叉）；设 False 回退旧行为做 A/B。

    # ---- 对抗式验证（#16，2026-08-30）----
    # 正向验证**通过**后主动证伪：假设答案错误，反向找反例 / 第一个错误步骤。
    # 基线依据：两层复核一致性仅 51%、反向案例 0 条（第二层只是加严不是独立），
    # 且正向验证存在漏检。正向验证问"这对吗"（确认偏误），
    # 对抗式验证问"假设它是错的，错在哪"（证伪 → 反例法）。
    # 与 enable_feedback_review 互补：那个治误杀，这个治漏检。
    enable_adversarial_verify: bool = True
    # 生效档位：fast 是简单题快速通道，跳过以省调用
    adversarial_tiers: list = field(default_factory=lambda: ["deep", "standard"])
    # 低于此置信度的"检出"不采信：宁可漏掉，不可误伤（治误杀优先于治漏检）
    adversarial_min_confidence: float = 0.5
    # 2026-10-02 DeepSeek 适配：原 640 ⇒ 8192。原值基于「对抗审查输出很短 + prefill
    # 抑制思维块」的 Intern-S 时代假设，对 reasoning 模型必然截断
    # （reasoning 先吃满预算、正文为空）。
    adversarial_max_tokens: int = 8192
    adversarial_max_reasoning: int = 2400
    # 2026-10-20：审核侧义务账本（蓝图倒推链反演），见 agent/obligation_ledger.py
    audit_obligation_ledger: bool = True
    # ---- 验证增强链时间护栏（2026-09-07：治 4.5_oracle / 4.6_adv 烧穿 6.5）----
    # 冒烟 v2 实证：3.3/3.6 止损省下的时间被 4.5 Oracle(365s)/4.6 对抗(372s)
    # 单次 300s+ 不可打断的复核吸收 → 6.5 Lean 终局仍 time_critical、0 绿点。
    # 4.5/4.6 是可弃增强：放行前要求剩余时间 ≥ 本值 + critical_tail + 30s 缓冲
    # （deep 需 ~450s / standard ~510s），否则跳过直进 6.5——宁少一层深查，
    # 不饿死最终闸门。设 0 = 关闭护栏（旧行为）。对齐 LLMClient 180s×2 重试上限。

    # ---- 2026-09-29：难度路由（DifficultyRouter）已删除 ----
    # 原 `enable_difficulty_router` / `enable_llm_difficulty` / `algebra_force_deep`
    # 三个开关随 DifficultyRouter 一并移除：统一单一档位后，"按难度选方法"整体废止。
    # 其对应行/键已从覆盖白名单同步删除（见下方 whitelist）。
    # 下表保留 `deep` 键作为**唯一配置项**，不再有任何档位分流语义：
    tier_sample_times: dict = None          # 候选数（统一档位：3）
    tier_temperatures: dict = None          # 温度分层（统一档位：4 层）
    tier_voting_times: dict = None          # 每候选投票数（统一档位：3）
    tier_max_completions: dict = None       # 截断续写数（统一档位：2）
    tier_max_calls: dict = None             # LLM 调用预算上限（统一档位：100）
    tier_budget: dict = None                # 设计预算帽秒（统一档位：86400，仅防挂死）
    # 2026-09-29：`paper_target_time` / `paper_min_soft` / `paper_total_questions` /
    # `deep_quota_ratio` / `pacer_normal_relax` / `paper_inflight` 随 PaperPacer 删除。
    # ⇒ **本仓库已不再保证"6h 全卷内完成"**：全卷无时间总量控制，单题只受
    #   `max_time_per_question` 硬墙约束（用户决策，研究阶段云端评测不限时）。
    # L1 验证优先（2026-08-31）：剩余时间不足该值时进入 verify_only，
    # 不再生成新候选（solver/续写/自改进/协作/子目标/Lean 门禁全跳过），
    # 把最后的时间留给验证投票 → 治 A_base 30 题里 117 次「验证 None 判错」。
    # ⚠ D 组对照实测（30 题）：7/30 = 23.3% vs A_base 8/30 = 26.7%，
    # 净 −1、McNemar p=1.0 → **噪声内，无收益** → 默认关闭（=0 不触发）。
    # 机制与测试保留（tests/test_verify_only.py）；若将来再试，
    # 先补「预算跳过/None 投票计数器」量化验证假设，再调阈值。
    deep_use_sub_goal: bool = True
    # ============================================================
    # 2026-10-01 开关注册制（审查 A 级第 2 条）
    # ------------------------------------------------------------
    # 以下字段与 `agent/switch_registry.py::SWITCHES` **一一对应**，
    # 默认值即该开关的历史默认值（从原裸 os.environ 读取处抄录）。
    # ★ 读取优先级：环境变量 > 本字段（显式设置）> 默认值
    #   —— 环境变量优先是刻意的，保证既有跑法（X=0 python ...）行为不变。
    # 改动任一开关前请先看 switch_registry 的注释。
    # ============================================================
    artifact_store_enabled: bool = True  # ARTIFACT_STORE_ENABLED — 中间结果落盘（results/<run_id>/<qid>/，默认开）
    answer_form_gate: bool = True  # ANSWER_FORM_GATE — 答案形态闸门（非答案形态不放行）
    deterministic_timeout_sec: float = 5.0  # DETERMINISTIC_TIMEOUT_SEC — 确定性验证通道的单次超时（秒）
    eval_alpha_equiv: bool = False  # EVAL_ALPHA_EQUIV — 字母等价（A/a）判定（默认关）
    eval_split_cn: bool = True  # EVAL_SPLIT_CN — 中文答案分隔符切分
    expr_eval_grounding_guard: bool = True  # EXPR_EVAL_GROUNDING_GUARD — 表达式求值接地护栏
    lean_gate_parallel: bool = True  # LEAN_GATE_PARALLEL — Lean 门禁并行预取（=0 串行）
    lean_gate_strict_unknown: bool = True  # LEAN_GATE_STRICT_UNKNOWN — Lean 门禁对 unknown 从严
    # ★ 2026-10-07（语义分离）：「形式化不充分」≠「命题为假」。见 agent/switch_registry.py
    #   与 tools/lean_local/lean_bridge.py `_crosscheck_failure_verdict()`。
    #   实测 smoke10：51 条 cross_check_failed **全部 compiled=true**（代码已编译通过）
    #   却因自证拦截被判 proof_invalid（铁证 official112-004 的真命题 (2025:ℕ)/2=1012）。
    #   True（默认）= 这类命中落 unknown（**无法核验，非答案错**）；False = 回退旧
    #   proof_invalid 语义做 A/B。读取链 = env > config > 默认（经 switch_registry）。
    lean_crosscheck_as_unknown: bool = True  # LEAN_CROSSCHECK_AS_UNKNOWN — cross_check/自证拦截落 unknown
    lean_mcp_autolake: bool = True  # LEAN_MCP_AUTOLAKE — MCP 自动 lake 环境准备
    lean_mcp_goal_loc: bool = False  # LEAN_MCP_GOAL_LOC — MCP 回报 goal 位置（默认关）
    lean_mcp_hover_check: bool = False  # LEAN_MCP_HOVER_CHECK — MCP hover 检查（默认关）
    lean_mcp_multi_attempt: bool = False  # LEAN_MCP_MULTI_ATTEMPT — MCP 多次尝试（默认关）
    lean_mcp_timeout_floor: float = 300.0  # LEAN_MCP_TIMEOUT_FLOOR — MCP 调用超时下限（秒）
    lean_mcp_verify_axioms: bool = True  # LEAN_MCP_VERIFY_AXIOMS — MCP 校验公理使用
    lean_verify: bool = True  # LEAN_VERIFY — Lean 通道总开关（=0 一键关闭）
    lean_xcheck_numeric: bool = True  # LEAN_XCHECK_NUMERIC — 数值答案的 Lean 交叉校验
    llm_retry_on_timeout: bool = False  # LLM_RETRY_ON_TIMEOUT — LLM 超时是否重试（默认不重试）
    # ★ 2026-10-07 补登：LLM 读超时（秒）。原为 utils/llm_client 的模块级常量，import 期求值 + 不入注册表 ⇒ config/CLI 覆盖读不到。
    llm_timeout: int = 120
    minimal_mode: bool = False  # MINIMAL_MODE — 最小基础模式（只留主链，默认关）
    numericize_final: bool = True  # NUMERICIZE_FINAL — 终答数值化（把精确式转小数）
    objective_itemwise_priority: bool = True  # OBJECTIVE_ITEMWISE_PRIORITY — 客观题逐项优先策略
    objective_majority_vote: bool = False  # OBJECTIVE_MAJORITY_VOTE — 客观题多数投票（默认关）
    objective_selfcheck: bool = True  # OBJECTIVE_SELFCHECK — 客观题自查
    self_improve_keep_original: bool = True  # SELF_IMPROVE_KEEP_ORIGINAL — 自改进保留原答案
    self_improve_objective_skip: bool = True  # SELF_IMPROVE_OBJECTIVE_SKIP — 客观题跳过无条件自改进（=0 则也跑）
    theorem_hint: bool = True  # THEOREM_HINT — 定理检索注入（=0 关闭）
    toolcall_text_fallback: bool = True  # TOOLCALL_TEXT_FALLBACK — 文本工具调用兜底（2026-10-07 默认翻为**开**：smoke10 实测 10 题共 93 次文本调用被丢弃，含题 000 的自纠错检索请求；执行侧有 web_search 白名单 + 必填参数 + 幂等 + max_rounds 限流。=0 回退旧行为做 A/B）
    # 2026-10-15 双通道：Lean/Math 工具集注册给大模型（默认开）
    enable_lean_tools: bool = True
    # ★ 2026-10-18：Lean 自动化策略链 + 4.0 答案形态严格校验
    lean_hammer: bool = True
    smt_falsify: bool = True
    attention_focus_enabled: bool = True
    subgoal_theorem_search: bool = True
    rewrite_on_low_conf: bool = True
    rewrite_conf_threshold: float = 0.0
    answer_shape_strict: bool = True
    # ★ 2026-10-17（用户定案）：跨模型独立检错 + 子目标生成前自审
    cross_model_falsify: bool = True
    cross_model_timeout: float = 90.0
    subgoal_regen_self_review: bool = True
    # ★ 2026-10-17（用户定案内化）：3.3 无条件自改进（默认 False）
    self_improve_stage: bool = False
    # ★ 2026-10-15（审核 F5 补登记）：numbers 叶子 Lean 数值核验（默认 False）
    enable_numeric_lean_verify: bool = False
    # ★ 2026-10-15（用户定案）：子目标三角色协作求解（默认关，可 A/B）
    subgoal_collab_mode: bool = True
    subgoal_collab_regen_rounds: int = 3
    # ---- 2026-10-07 开关注册制补齐（fix-switch-debt）：6 个账外 gate 型 env 的配置字段 ----
    # 这 6 个 env 此前只以裸 `os.environ.get` 读取 ⇒ 不在本类声明、不在白名单、
    # 不在注册表。默认值**逐字 = 现存 env 读取处的字面默认**（只接线，不改行为）。
    # 读取处仍需改为实时读注册表，见 agent/switch_registry.py 各条注释与交付清单。
    deep_max_rework: int = -1                 # DEEP_MAX_REWORK — 6.5 重做候选上限（deep；-1=无上限）
    objective_itemwise_plan: bool = True      # OBJECTIVE_ITEMWISE_PLAN — 子目标规划客观题逐项策略
          # 子目标分解主路径（统一档位下每题执行）
    deep_revise_rounds: int = 2             # deep 档 0 票时 revise 自纠错轮数（08-30：1→2，LeanSearch v2 反思循环）
    deep_use_playoff: bool = True           # deep 档 0 票且时间宽裕时 playoff 复算
    # 2026-09-14 **默认关闭**。依据不是"证明无影响"，而是**实测它已基本不触发**：
    #   5 题 × 2 臂 = 10 次观测，`3.4_collab` 耗时**全部 0.0s** ——
    #   ① 多数题被判 standard（按设计不跑，只有 deep 档跑）；
    #   ② 真判 deep 的题到达 3.4 时生成侧时间截止已过，被门控跳过
    #      （004 = 2.7 355s + 3_solve 243s ≈ 600s，之后还要留 3.6 121s + 4_verify 383s）。
    # ⇒ 关掉它既不损失正确率（本来就没跑），也**杜绝了历史最坏 691s 的复发**
    #   （004 旧配置下单题被 3.4 烧掉 691s，把 3.3 / 4_verify / 6.5 全挤成 0s）。
    # ⚠ 该 A/B **不是**"证明无影响"的有效对照（机制两臂都没触发），
    #   不能把这次结论外推成"3.4 无用"。若要真正评估它，必须在时间充裕的配置下单独测。
    # enable_collaborative_deep: 启用（用户指示）：deep 档三 Agent 协作。原 2026-09-14 关闭，且原注释明确要求「若要真正评估它，必须在时间充裕的配置下单独测」——当前正是时间无约束配置。回退：改回 False
    enable_collaborative_deep: bool = True  # 难题(deep 档)三Agent协作：解题→审查→整合→验证
    collab_max_rounds: int = 3              # 协作验证循环最大轮数（2026-09-06 P3 用户拍板 6→3：单轮含 3 次 LLM 不可中断、algebra-003 曾烧 535s，收紧省时；时间充裕时停滞检测照常兜底）
    # 子目标阶段预算（2026-09-06 P1 用户拍板按档拆分）：
    # deep 保留 750s（难题深度分解值）；standard/fast 用 450s——
    # 依据：2.7 子目标全档均 ~587s 为最大黑洞，standard 档性价比存疑，
    # 省下预算自然流向 3_solve/verify。原 subgoal_stage_budget_sec 未入
    # AgentConfig（sub_goal_solver getattr 兜底 750），现补全可配。
    subgoal_stage_budget_sec: float = 750.0     # deep 档子目标阶段预算
    subgoal_stage_budget_sec_std: float = 450.0  # standard/fast 档子目标阶段预算
    # ★★ 2026-09-15（用户指示「预算都删了，没意义，还卡正确率」）：
    # **阶段预算总开关，默认关闭**（关 = 不再有任何"阶段时间帽"）。
    # 关闭依据（用户判断 + 代码实测）：
    #   `_phase_budget` 给 2.7 的帽是 **600s**，而 2.7 实测 mean 513 / p50 537 /
    #   **max 1168s** ⇒ **一半以上的题会被这个帽砍断子目标链**。
    #   ⇒ 这不是"慢"，是**链没跑完 = 缺项 = 错**，直接损害正确率。
    # 关闭后的行为：`_phase_budget` 返回「无穷大」⇒ `_phase_deadline_guard` 做的
    #   `min(ctx.deadline, now + cap)` 退化为原 deadline ⇒ 阶段帽不再生效。
    # ⚠ **不影响"防卡死"类护栏**：LLM 超时重试、`lean_timeout`、符号求解线程超时、
    #   各类死循环硬上限（`max_revise_rounds` 等）**均保留** —— 那些不是配额，是"别挂住"。
    # 回退：置 True 即恢复赛期口径（12 个阶段帽全生效）。
    phase_budget_enabled: bool = False
    # ---- 2026-10-04 开关注册制补齐：新增 2 个字段（与 switch_registry 同名登记）----
    #   `k11_mask_statement`：消费方已存在（agent/sub_goal_solver.py::_k11_mask_on），
    #     此前字段/注册表皆缺 ⇒ 该函数走 try/except 兜底恒 True；现补全后可配。
    #   `candidate_pool_limit`：候选池上限（0 = 不限），默认 12（2026-10-09 由 6 抬高）。
    k11_mask_statement: bool = True          # K11 注入点守卫（子目标描述/蓝图结论断言值掩码）
    # ---- 2026-10-06 子目标交付重写（A/B/C）----
    # A: 让每个子目标产出**可独立校验的结论**（值/等式/判定），并把蓝图节点的
    #    `produces` 真正下传为 `expected_output`。依据（smoke10 实测）：73 个子目标
    #    里 59 个 expected_output 为空（81%）、64 个交付 <120 字（88%）——根因是
    #    蓝图路径 `to_subgoal_plan` 把该字段硬编码成 ""，且模型只给自然语言结论。
    # B: 让子目标结论**真的被客观判定**（SymPy 判等式真伪 / 多候选符号多数），
    #    判不出时显式留痕原因。依据：原实现 `n_judged = 0/73`（一个都没判定过）。
    # C: **无推导支撑的结论打标**并警示主求解/合并层。依据：题 007 的 7 个子目标
    #    只给结论无推导（其中一条臆造）却被主求解直接采用 ⇒ 答错。
    # 默认 True = 修好后的行为；=0 一键回退旧行为做 A/B（两开关均已登记注册表）。
    subgoal_conclusion_contract: bool = True   # A+C：结论契约（结构化结论 + 推导标记）
    subgoal_objective_judge: bool = True       # B：结论客观判定 + 判不出留痕
    #   ★ 2026-10-04（D7）：`candidate_pool_limit` = 候选池上限（原为三处硬编码 6，
    #     现收敛到 `agent/solver._pool_room` 单一来源）。
    #   ★ 2026-10-09（用户拍板「先 12 试试」）：**默认 6 → 12**（与注册表默认逐字一致）。
    #     ⚠ 本值**同时被当作本次生成数量**（`_generate_initial(count=remaining)`）
    #     ⇒ 不要设成天文数字；设 0/负 = "不限"（内部有安全上界 12；抬到 12 后
    #        「0=不限」与「=12」行为等价，安全上界仍生效，不会跑飞）。
    candidate_pool_limit: float = 12.0       # 候选池上限（0 = 不限，内部上界 12；默认 12）
    #   ★ 2026-10-04 修（D22）：`sub_goal_oracle_check` 此前**全仓无定义**，
    #     而唯一消费点 `agent/sub_goal_solver.py` 用 `getattr(config, ..., False)`
    #     读它 ⇒ `_oracle_check_step()` **结构性不可达**（代码写了永远跑不到）。
    #     现补上字段 + 注册表 + 白名单 + `_SPEC`，使其可配（默认关，行为不变）。
    sub_goal_oracle_check: bool = False      # 子目标逐步 oracle 校验（默认关，见注册表说明）
    # ★ 2026-09-15（用户：「子目标上限太少了；不同档位的最佳数量不同吗？」）：
    # **子目标上限按档位分档**。此前 `max_subgoals` 是**全局单一值 6**，
    # 而蓝图提示词已写明"简单 3~5 / 中等 5~12 / 难题 12~30" ⇒ **提示词与代码矛盾**。
    # 现按档位给不同上限（deep 是难题档，题更复杂 → 允许更多子目标）。
    # ⚠ 具体数值**无 A/B 依据**，取"比原 6 放宽但不失控"的保守值；
    #   验证方法：同一 commit 下对照 max_subgoals ∈ {6, 8/16, 0} 跑错题集。
    # 取值优先级：本字典 → `max_subgoals` → 全局兜底 6。
    max_subgoals_by_tier: dict = None  # {"standard": 8, "deep": 16}；None=用 max_subgoals
    # 2026-09-15（赛后无约束评测）：子目标规划数上限纳入 AgentConfig。
    # 此前该值**不在本类中**，全部靠 `getattr(self.config, "max_subgoals", 6)` 回退，
    # 而 ReasoningAgent.__init__ 的覆盖白名单只认本类已有字段
    # ⇒ CLI/overrides 传 max_subgoals **会被静默丢弃**（同 enable_calc_prewarm 那次的坑）。
    # 截断点 agent/sub_goal_solver.py:298 `subgoals = subgoals[:_max_sg]`——砍的是**拓扑序尾部**，
    # 可能正好丢掉 merge 所需的收尾子目标。6 是比赛期"少而精"口径；
    # 放开该值是验证「子目标冗余是否必要」的前置条件。
    # ★ 2026-10-12 用户决策 D22：「子目标上限太低了……先把上限直接修改为100，
    #   看看正常的子目标数量是多少再设」。留 100 而非 0（不截断）是刻意的：
    #   0 完全无界，若规划失控会放大成本；100 事实上不会撞到、但保留一个上界。
    max_subgoals: int = 100             # 子目标规划数上限（0/负 = 不截断）
    # 2026-09-06 老师建议（子目标独立性/最小上下文）：子目标上下文注入模式。
    # "deps"（默认）= 按 depends_on 只注入直接依赖结果，无依赖子目标零前序上下文
    # （可独立、可并行校验、不被无关中间量污染）；"all" = 旧行为全量前序注入。
    subgoal_ctx_mode: str = "deps"
    # P2（2026-09-09 老师：计算/推理子目标类型化）：DAG 子目标打标 calc_kind
    # （terminal=纯计算型 / inline=推理型，规则启发，字段总是进 trace 可观测）。
    # 2026-10-01：`subgoal_calc_router`（terminal 专用模板与轻校验）随 `<calc>`
    # 板块删除；`calc_kind` 打标保留（纯观测字段）。
    # 2026-09-06 老师建议（S1-lite 0-LLM 校验前移）：子目标结果自带 lean 代码片时
    # 做本地编译校验（L1，仅 deep 档 + lean 环境可用，0 LLM 5-21s，禁网纯本地）。
    # 异常/环境缺失一律放行，绝不阻断；False = 跳过 L1 只留 L0 截断检查（A/B 对照）。
    enable_subgoal_lean_check: bool = True
    # 子目标交叉核对（2026-09-08 老师建议3）——merge 前不让子目标间矛盾
    # 静默流入最终合并。两方向独立开关，可单开/同开做 A/B：
    #  A 确定性冲突闸（0 LLM）：汇总 <calc> 回填 + <check> + L2 断言，
    #    同名 LHS 出现不同常数（x=1 vs x=2）→ 矛盾注入 merge 提示词强制裁决。
    #    抓显式冲突；抓不到隐含矛盾（alg-060：xy=25 与 D=2 不共变量名，纯规则必漏）。
    #  B LLM 交叉核对轮（merge 前 +1 次小调用 1-2K token，60-110s 级）：
    #    各子目标结论行单独抽出集中裁决，能看隐含矛盾（不保证）。
    subgoal_conflict_gate: bool = False     # A 确定性冲突闸（0 LLM）
    subgoal_crosscheck_llm: bool = False    # B LLM 交叉核对轮（+1 调用）
    accept_confidence: float = 0.6          # AcceptGate 可接受置信度阈值（>=该值视为通过，v2.8）
    # ★ 2026-09-30（截图 #9）：**择优可信度门槛**。
    #   用户口径「投票…同时要有可信度要求（太低＝全军覆没）」。
    #   默认 0.0 = 关闭 ⇒ 行为与改动前逐字一致（不产生任何新记录）。
    #   开启后不改变"必须有答案"的地基：低于门槛的候选**仍提交**，
    #   但会在 trace / `_pick_diag` 里被显式标记，供 5.5 复核与数据闭环定向归因。
    pick_confidence_floor: float = 0.0
    # ★ 2026-09-30（截图 #9）：**revise 之后接一次无条件自改进**。
    #   用户口径「错误点返回大模型重新生成，**可否加无条件自改**」。
    #   revise = 告诉模型错哪了（有条件）；自改进 = 再给一段推理预算（无条件）。
    #   开启后：每轮 revise 生成的候选会**再**过一遍 `improve_candidates`。
    #   默认 False = 行为与改动前逐字一致（不产生额外 LLM 调用）。
    #   ⚠ 与 `self_improve_rounds`（3.3 独立环节的遍数）**正交**，不要合并。
    self_improve_after_revise: bool = False
    # 结构化 bug report 驱动的修正（论文依据：IMO 2025 验证-精炼流水线）
    # 验证器改为产出「分类 + 原文定位」的结构化错因，注入 revise 步骤。
    # 论文实测：best-of-32 仅 21.4%~38.1%，加验证-精炼后 85.7%，
    # 说明杠杆在错因质量而非候选数量。
    use_bug_report_feedback: bool = True

    # ---- 时间预算（2026-08-28 起；2026-09-29 随 PaperPacer 删除简化）----
    # 统一档位后实际生效的是 `deep_critical_tail_seconds`（orchestrator 恒取该值）。
    # `critical_tail_seconds` 仅作其它模块的通用兜底默认。
    critical_tail_seconds: float = 120.0      # 通用兜底：剩余不足该值则跳过可选步骤
    deep_critical_tail_seconds: float = 60.0  # 统一档位生效值：把时间用得更尽
    # 2026-09-29：`deep_quota_ratio` 已删除（PaperPacer 移除 ⇒ 全卷配额闸不存在）。

    # ---- 检测链（2026-09-06：AuditGate 为主；晚间恢复 Lean 双通道）----
    # AuditGate 多级检测链：Level0 程序硬核验 → Level1 反例搜索 → Level2
    # LLM rubric 判分 → Level3 playoff。硬否决优先、宁 unknown 不误杀、只审不答。
    # Lean 硬验证通道（2026-09-06 晚，lean-toolchain 工具仓离线可跑后恢复）：
    # Lean 环境可用 → 2.6 preverify / 3.6 候选 / 6.5 最终闸门走 LeanGate 硬验证
    # （证明题整题 verify、非证明题 verify_answer），Lean 不可用/异常自动回落
    # AuditGate（AI 判分链），本地评测与平台两种环境同一份代码都正确。
    # ---- Lean 双通道开关（2026-09-06 晚恢复；9/6 去 Lean 化前字段全量回归）----
    enable_lean_verify: bool = True         # Lean 通道总开关（配合环境探测，无 Lean 自动回落 AuditGate）
    # 2026-09-12 **按实测数据关闭**：2.6 前置形式化在冒烟中 **5/5 题全部 fail**
    # （且都用满 2 轮重试），每题白烧 100-150s（其中大部分是"LLM 把题目翻译成
    # Lean"的耗时，Lean 编译本身仅 3s）。实测结论：模型对题意的理解是正确的，
    # 失败全在 Lean 语言层面（API 不存在 / 值当类型用 / 语法）⇒ **该环节零产出**。
    # 注意：关闭它不影响"Lean 参与评测"—— Lean 的价值在 **3.6 候选级淘汰** 与
    # **6.5 最终答案闸门**（验证完整解答），而非 2.6 的"翻译题目"。
    # 回退：置 True 即恢复（debris 保留，便于赛后用改进后的提示词重试）。
    # enable_lean_preverify: 启用（用户指示）：2.6 题目前置形式化。原 2026-09-12 因实测 5/5 fail、零产出关闭（纯烧 100-150s/题）。平台无 Lean 时自动回落。回退：改回 False
    enable_lean_preverify: bool = True     # 2.6 题目前置形式化（题目转 Lean 声明校验理解，deep 档）
    # 2026-09-12 **按实测数据回退**：曾按用户"lean 一定要用"扩到
    # ("deep","standard")，但同日冒烟实测（051/010/000 三题）显示前置形式化
    # **0/3 通过，且三题全部用满 2 轮重试仍 fail**；报错均为 **Lean 代码层面**
    # （Application type mismatch / expected ';' / failed to synthesize instance），
    # 而非数学理解错误 ⇒ standard 扩展属**纯耗时无产出**（每题 100-150s），已回退。
    # 另：2.6 的耗时主要在"LLM 把题目翻译成 Lean"（20-40s/次），Lean 编译本身仅 3s
    # （日志实测）。deep 档是否保留 preverify，待全量数据再评估。
    preverify_max_rounds: int = 2           # preverify 编译失败重试上限（强制重新审题）
    preverify_timeout: float = 60.0         # preverify 单次编译/转化超时（秒）
    # 2026-10-12 补齐（原为裸 getattr，读取点 tools/lean_local/lean_pre_verifier.py:169）：
    #   预验证阶段的**骨架 Lean 审核**。与 2.7 的 DAG 整树审核功能重叠，故默认关；
    #   需要「骨架也过一遍 Lean 语法」时置 True。
    enable_preverify_sketch_audit: bool = False
    # ---- 2026-10-12 批次 1：题意理解（★ 2026-10-14 收敛）----
    #   默认 **True**（用户 2026-10-12 决策：本轮直接启用以便「全开测 10 题」）。
    #   缺省语义与既有 `lean_pre_verifier.run` **互不影响**（后者行为不变）。
    #   ★ 2026-10-14：删「逻辑链 + 一致性复推 + 链的 Lean 自检」⇒ 只保留
    #   终态 + 答案槽（纯符号）+ 缺口 A/B；配套三开关（understanding_refine_rounds /
    #   understanding_no_gain_stop / understanding_chain_lean_check）一并撤除。
    enable_understanding_chain: bool = True
    # ---- 2026-10-12 批次 2：蓝图多次生成（迭代式）+ 自审（用户 Q11–Q13/Q18）----
    #   1 次初始生成 + (N-1) 次在上一轮基础上无条件自改进；采用**最后一轮**；
    #   第 4 次交 LLM 自审（DagReviewer），拒收比例 ≥0.4 才重写 1 轮。
    blueprint_multi_gen_rounds: int = 3
    blueprint_self_review: bool = True
    blueprint_self_review_reject_ratio: float = 0.4
    # ---- 2026-10-12 批次 3：子目标 inputs 机械校验（用户 D19–D21）----
    subgoal_inputs_validation: bool = True
    # ---- 2026-10-12 批次 4：验证链重排（用户 Q29–Q33）----
    #   任一淘汰环节池空 ⇒ 汇总错误点交下游「先找错点 → 再生成」（该部分保留）。
    #   ★★ 2026-10-13 **语义更正**（根因：official112-008 由对变错）：
    #     本开关现在 = 「4_verify 的**候选判定强度**」——
    #       True（默认）= 多票（tier_votes=3，强度与基线一致）；
    #       False = 单票（**A/B 对照组**，已知会削弱兜底）。
    #     ⚠ 票数**不参与终选**（终选由 final_selector 独立完成，只看证伪结果与可解析性）
    #       ⇒ 「投票是最后的检测」由终选层保证，**无需降票数**；
    #       2026-10-12 曾把票数降为 1，真实后果是削弱验证强度：单票易出现
    #       `correct_votes>0` ⇒ `orchestrator__revise.py:466` 的「全 0 票 ⇒ 强制重解」
    #       不再触发（5_revise 5805s → 306s）⇒ 少了多轮修正与证伪机会
    #       ⇒ 终选由 `objective_single` 退化为 `vote` ⇒ 选错。
    # ★ 2026-10-13：**4.7_lean**（投票前 Lean 淘汰，用户 Q30）—— 默认关，先 A/B。
    # ★ 2026-10-13：**LLM 辅助 query**（用户 A1）—— 默认开（研究期用全部手段）。
    # ★ 2026-10-13：子目标级无条件自改进轮数（用户 E24）；=0 关闭。
    # ★ 2026-10-13（§3 方案 A）：题型 LLM 判题；False ⇒ 完全回退规则判定。
    enable_type_llm: bool = True
    subgoal_self_improve_rounds: int = 3
    # ★ 2026-10-15（用户「2 与 3 之间建一个环」）：**求解期动态倒推**。
    #   生成期倒推（prompts/blueprint.py 构造原则第 2/3 条）只能基于**题面**判断
    #   前提是否具备；求解期才发现的前提缺口由本机制现场补：倒推一层 → 解前提 →
    #   回填 → 重解原目标。此前这种子目标只能写 `ctx.revise_feedback` 交上游，
    #   而上游 replan 已于 2026-10-15 整块删除 ⇒ 反馈无人消费。
    #   默认值与 contracts/switch_registry.py 同名条目逐字一致（门禁强制比对）。
    subgoal_dynamic_backward: bool = True
    subgoal_backward_max_depth: int = 2
    enable_theorem_query_llm: bool = True
    pool_empty_clear: bool = True
    # ★ 2026-10-14 五道审核链（用户定案）：规范化前置 → ①AI 自审 → ②对抗证伪
    #   → ③Lean 代回 → ④计算检测 → ⑤投票（先判多答案再投票）。
    #   本字段 = 第 0 道（4.0_normalize）开关；默认值与
    #   contracts/switch_registry.py 同名条目逐字一致（门禁强制比对）。
    five_gate_normalize: bool = True
    #   ★ 2026-10-14 第 1 道（4.1_self_audit）：AI 自审逻辑错误 → 淘汰；
    #     输出不可解析/LLM 故障一律保留候选（防误杀）。
    # ★ 2026-10-23：子目标级 Lean 断言验证。默认关 —— 它是"真正用 Lean 判真伪"
    #   的第一步（原来的 answer_valid 只验表达式合法），但成本未测，先用探针验证。
    five_gate_subgoal_lean: bool = False
    five_gate_self_audit: bool = True
    #   ★ 2026-10-14 第 2 道（4.2_falsify）：对抗证伪逐候选化 ——
    #     机检证伪（SymPy 反例）+ LLM 对抗生成；证伪即判错淘汰。
    five_gate_falsify: bool = True
    #   ★ 2026-10-14 第 3 道（4.3_lean_check）：Lean 逻辑判断·答案代回题目；
    #     部分淘汰落池、全拒标记不静默清池、不适用题豁免（沿用 3.6 口径）。
    five_gate_lean_check: bool = True
    # ★ 2026-10-15（工具①观测通道）：4.3 调 lean_check 工具（默认关）
    five_gate_lean_check_tool: bool = False
    # ★ 2026-10-22（五道链审计修复）：证据落地校验 —— 4.1/4.2 的淘汰必须附
    #   可机器复核的原文引证（编造的出错位置/步骤不得据此淘汰）。
    five_gate_grounding: bool = True
    # ★ 2026-10-22：4.1 输出不可解析时按最严格式重问一次（旧口径一次放行）。
    five_gate_audit_retry: bool = True
    # ★ 2026-10-22：4.1 自审采样次数。实测同一错误候选 3 次判定为「留/杀/留」
    #   ⇒ 单次不可靠；多次采样、任一带落地证据的逻辑错即淘汰。1 = 旧行为。
    five_gate_self_audit_votes: int = 3
    # ★ 2026-10-22：4.2 对抗证伪采样次数（实测同样不稳定，3 次只证伪 1 次）。
    five_gate_falsify_votes: int = 3
    # ★ 2026-10-22：子目标未产出有效结论时的重做轮数（旧实现空返回从不重试）。
    subgoal_failed_retry_rounds: int = 2
    # ★ 2026-10-22：蓝图自审用局部子树重写（LCA 触根自动退化为整树）。
    blueprint_subtree_regen: bool = True
    # ================= 2026-10-22 全仓审计：补齐 6 个「环境变量改不动」的开关 =================
    # 这 6 个在 switch_registry 里注册了、也在白名单/param_usage 里，但 config 类**没有字段**
    # ⇒ 消费方用 `getattr(config, "x", 代码默认)` 读取时永远拿到代码默认值，
    #   设 `STRICT_ALL_REJECTED=1` 这类环境变量会被**静默忽略**（不是默认值错，是覆盖链断了）。
    # 取值一律与 registry 声明保持一致（getattr 兜底值必须等于 config 声明值）。
    falsify_max_per_q: float = 0.0            # 0 = 不限
    enable_question_type: bool = True
    strict_all_rejected: bool = False
    require_answer_in_pool: bool = True
    deterministic_solution_check: bool = True
    audit_gate_rubric_nonproof: bool = False
    # 余下 9 个只经 `get_bool()/get_num()/get_str()` 读取（行为本就正确），
    # 缺字段的唯一后果是 `param_usage` 报表显示代码默认值、与实际 env 覆盖值不符。
    # 一并补齐，使「注册表 / config / 白名单 / param_usage」四件套完整。
    pool_gate_v2: bool = True
    check_protocol_guard: bool = True
    verifier_vote_reasoning: str = "disagree"
    # ★ 2026-10-22：4.5 投票平票时以 4.4 的机器核验（verify_math）为依据。
    five_gate_verify_tiebreak: bool = True
    # ★ 2026-10-22（4.3）：Lean 编译失败按原因分级 —— 只有 refuted 才判答案错。
    lean_failure_kind_split: bool = True
    #   ★ 2026-10-14 第 4 道（4.4_math_check）：数学工具计算检测 ——
    #     SymPy 精确判定；fail 淘汰 / unknown 放行 / 全拒回退保留。
    five_gate_math_check: bool = True
    # ★ 2026-10-15（工具③观测通道）：4.4 等式逐条复算（默认关）
    five_gate_math_check_calcverify: bool = False
    #   ★ 2026-10-14 第 5 道（4.5_vote）：先多答案判断（ctx.vote_multi_answer
    #     单一来源）→ 大模型投票（票数锁定 tier_votes 多票口径）。
    five_gate_vote: bool = True
    #   ★ 2026-10-14（五道链配套·4.6_rewrite）：池空重写通道（错误情况→诊断→
    #     重建蓝图和子目标→重写→重过五道链）。轮数上限（每轮含 1 次诊断 +
    #     1 次子目标主链重跑 + 整链复核）。
    five_gate_rewrite: bool = True
    rewrite_max_rounds: int = 2
    lean_gate_all_proofs: bool = True       # 证明题全档 Lean 硬验证（False 回退仅 deep）
    lean_gate_nonproof: bool = True         # 非证明题候选级 Lean（2026-09-10 打开：用户要求全卷过 Lean，按题适用性豁免客观/概念题）
    lean_gate_nonproof_deep_only: bool = False  # True=非证明题 Lean 仅 deep 档（时间墙护栏；默认 False=全档）
    lean_gate_strict: bool = False          # Lean unknown 时严格拒绝（True 保守拒，默认放行保分）
    lean_gate_unknown_stop: int = 2         # 候选级连续 N 个 unknown 止损（0=关；治证明题整题 verify 空转 563s）
    lean_timeout: float = 60.0              # Lean 单次编译超时（秒）
    lean_backend: str = "mcp"               # 2026-09-11 修复后默认 mcp：根因=本机缺 `Mathlib.olean` 聚合入口，裸 import Mathlib 触发 fatalError；现已自动规范化为 `import Mathlib.Tactic`，mcp 诊断正常（精确 行:列 + goal，增量秒回）；env LEAN_BACKEND 可覆盖
    lean_executable: str = ""               # lean.exe 绝对路径（空=自动探测：LEAN_EXE env>elan>vendor/lean-toolchain）
    lean_project_dir: str = ""              # 带 Mathlib 的 lake 工程目录（空=自动探测）
    theorem_memory_enable: bool = False     # 跨题定理记忆（9/6 关闭维持；LeanGate 写入按此开关）
    # ==================================================================
    # LeanSearch 引理检索（2026-09-15 重启）
    # ------------------------------------------------------------------
    # 背景：老师建议「求解子目标关键在定理的 Mathlib 搜索」；且李平老师担心
    # 模型「硬套定理」。实测证据（2026-09-15）：历史 3 批共 202 题
    # `mathlib_usage_stats.search_calls` **恒为 0** —— 此前三重静默失效：
    #   ① 主仓 tools/lean_local/ 下**没有 lean_search.py**（只有 vendor 归档副本）；
    #   ② AgentConfig **没有 use_leansearch 字段** ⇒ getattr(...,False) 永远 False；
    #   ③ lean_gate/lean_refiner 的 import 被裸 except 吞掉（只打 debug）。
    # 2026-09-15 已补齐 ①（复制归档实现到主仓）与 ②（本字段）。
    # ⚠ 后端优先级：官方语义 API（leansearch.net，实测 1.7-2.8s，质量最高）
    #   → 本地 lsv2 语料（**HuggingFace 被本机代理挡死，暂时取不到**）
    #   → 本地 mathlib 源码扫描（D:/mathlib4-last_bump_for_v4.31.0，8.6 万条声明）。
    # ⚠ **默认关**：未做开/关 A/B 前不得默认启用（沿用本项目「未验证不上」纪律）。
    # ⚠ 接线点说明：原调用点 lean_refiner._search_mathlib 所在的 LeanRefinerAgent
    #   **不在主链上**（orchestrator 从不实例化它）⇒ 本开关实际生效的是
    #   `leansearch_inject_verifier`（方案 A：把定理原文注入验证器）。
    # ==================================================================
    # ★ 2026-10-04（D20）：**保持默认关**。曾短暂改为 True（按研究期"用全部方法"
    #   口径），但回归测试 `test_web_search_default_on_leansearch_still_off`
    #   明确锁定此默认值 —— 它是本项目「**未验证不上**」纪律 + 2026-10-01 的
    #   显式范围决策（"LeanSearch 未在本次清单内"），不是逻辑错误。
    #   要启用请显式 `USE_LEANSEARCH=1` 或走 A/B，**不要**改默认值。
    # ★ 2026-10-12（用户决策 Q3：「必须使用 leansearch，改为 true」）：
    #   默认值 False → True。与 `contracts/switch_registry.py` 的同名条目
    #   **必须一致**（两处默认值打架 = 项目最忌讳的"第二个事实来源"）。
    #   注：实际评测入口（deploy/run_112.py:274 / run_research.py:69）本就显式传
    #   `--use_leansearch true` ⇒ 对云端评测无行为变化；本改动让「裸跑 run_eval」
    #   也默认开启，并消除「默认关」的误读。守卫测试已同步更新（决策变更，非放宽判据）。
    use_leansearch: bool = True             # LeanSearch 总开关（2026-10-12 起默认开）
    # ★ 2026-10-12（批次 5，用户 A1：「定理数定为 25」）：
    #   **检索取回 25 条**（池子够大才能覆盖多方向），注入侧再由
    #   `theorem_hint_block_max_chars` 控块长（超限按相关度截断并标注剩余条数）。
    #   ⚠ 分环节条数（前置理解 3–5 / 蓝图 10–15 / 主求解 10–25）尚未实现 ——
    #     需把「渲染」从检索期后移到各注入点，属下一批（见 CHANGES 未完成项）。
    leansearch_top_k: int = 25              # 每次检索返回条数（2026-10-12: 5 → 25）
    theorem_hint_block_max_chars: int = 2500   # 注入块字符上限（2026-10-12 批次 5）
    # ★ 2026-10-12（用户 §5.2「不同环节的定理量可能不同」）：分环节注入条数。
    #   前置理解少（读题对标不该被定理淹没）、主求解/修订多（要拿全量线索）。
    #   ⚠ 注册表只支持 bool/int/num/str ⇒ 这里用 **JSON 字符串**承载；
    #     空串表示「用注册表默认表」（读取侧解析失败也回退默认表）。
    #   ⚠ 默认值必须与 `contracts/switch_registry.py::SWITCHES` 的**逐字一致**
    #     （门禁 `test_60_registry_default_matches_config_default` 强制比对 87 个字段）。
    # ★ 2026-10-14（任务列表 #2）：检索下移到 2.7 子目标求解后，`subgoal` 成为
    #   主消费者，按任务原文「这个环节的产出就是 25 个定理」把 8 → 25。
    theorem_hint_top_k_by_stage: str = (
        '{"pre_audit": 4, "blueprint": 12, "subgoal": 25, "solve": 25, "revise": 25}')
    leansearch_max_calls_per_q: int = 2     # 单题检索次数上限（护栏：防时间膨胀）
    # ⚠ 原 `leansearch_timeout` 已**删除**（2026-09-15 上线审计）：
    #   检索器把官方 API 超时**硬编码为 10 秒**（`lean_search.py` 的
    #   `requests.post(..., timeout=10)`），既无 `__init__` 参数也无 `search` 参数
    #   可以透传 ⇒ 该配置**全仓无任何读取点 = 假开关**。
    #   因为 `lean_search.py` 刻意保持与 vendor 归档**逐字节相同**（避免 diff 噪声），
    #   不值得为它改实现；且其硬编码值本就等于原声明默认值 10.0，删掉无行为影响。
    # 方案 A 接线：把检索到的定理原文注入验证器，让验证器"对着原文"核查前提是否成立。
    # 这是把「凭记忆想起定理」变成「读到定理原文」，直接服务「硬套定理」判定。
    leansearch_inject_verifier: bool = True  # 仅在 use_leansearch=True 时才有意义

    enable_subgoal_main_path: bool = True   # 子目标细化作为主路径
    # ---- Blueprint DAG 分解（LEAP Stage 1，#27）----
    use_blueprint_dag: bool = True          # 子目标规划先用 BlueprintPlanner 生成 AND-OR DAG 再求解（失败自动回退原规划）
    # ★ 2026-09-15 修复：DAG 的**依赖边**此前结构性恒空。
    # 旧实现 `deps = _ancestors(nid) & selected`：祖先全是内部节点、selected 只含
    # 叶子 ⇒ 交集恒空 ⇒ `depends_on` 全为 []，**整张 DAG 的结构没传到求解器**。
    # 实测佐证：2026-09-10 那批 112 题 `subgoal_stats.dep_edges` 全部为 0。
    # 讽刺的是既有测试 `test_depends_on_reflects_ancestors` 把"恒空"断言成了正确行为。
    # 现改为按**依赖锥**计算（见 BlueprintDAG._preceding_leaves）。
    # 本开关用于 A/B 对照：False = 复现旧行为（不产出依赖边）。
    blueprint_deps_enabled: bool = True
    # ★ 2026-09-20 新增：OR 节点是否展开**全部**备选分支。
    # 历史行为（False）= 只展开 children[0]（注释称"主策略分支"），使 OR 在语义上
    # 退化为 AND 的单分支 ⇒ 蓝图路径**结构性不存在"独立求解分支"**，这正是
    # "子目标链锁定错值"的上游结构原因（比提示词层规则更根本）。
    # True = 展开每个 children，子目标规模会明显变大 ⇒ 须 A/B 后再决定默认值。
    blueprint_or_expand_all: bool = True
    # ★ 2026-10-15（用户要求「子目标承接 + 每个子目标设定工具，证明就用 lean」）：
    #   ① `blueprint_node_contract`：**节点契约** —— 蓝图每个节点声明 `tool`
    #      （lean/sympy/numeric/llm），并用 `inputs[i].use` 声明「把上游的哪个
    #      结论当作本步的什么条件」；两者下传到子目标（`tool` / `inputs_from`），
    #      求解侧据此追加工具指令与承接前提。关 ⇒ 只落盘不兑现（A/B 对照组）。
    #   ② `blueprint_tool_exec`：把声明**兑现成机器核验**（lean 实编译 / sympy
    #      判等式 / numeric 跑 norm_num），未过带反馈重解一次。关 ⇒ 零机器开销。
    blueprint_node_contract: bool = True
    blueprint_tool_exec: bool = True
    # ★ 2026-10-15（用户：「把 lean 内化到蓝图构建里面」）：
    #   `blueprint_lean_sig` = **叶子级形式化下传**。`tool=lean` 的节点由蓝图
    #   直接给出 Lean 4 形式化签名（`formal_sig`），求解侧只补证明体 ——
    #   即「蓝图出证什么、求解出怎么证」，把 Lean 的能力用在**给定命题**这一层。
    #   关 ⇒ 签名不下传，求解侧自行形式化（历史行为；A/B 对照组）。
    #   ⚠ 形式化失败（签名为空）时下游**静默回落**，绝不阻断。
    blueprint_lean_sig: bool = True
    # 每题子目标级 Lean 实编译次数上限（防证明类子目标密集时把预算烧在编译器上；
    # 与既有 `lean_numeric_max_per_q` 同口径，但计的是**子目标级**编译）
    blueprint_lean_max_per_q: int = 2
    # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：
    #   以下 6 个开关**整组撤除** —— 均为默认关闭（False）的蓝图评审/重规划机制，
    #   在生产路径上从不执行（读取点的 getattr 兜底值亦为假），仅
    #   `deploy/run_112.py::arm2` 复测臂引用：
    #     · dag_replan_gate            （求解前 DAG 强制评审门）
    #     · dag_replan_lean_check      （求解前门 Lean 逻辑检查开关）
    #     · dag_lean_max_nodes         （同上，节点数上限）
    #     · enable_dag_replan          （候选入池后 DAG replan）
    #     · enable_skeleton_review     （骨架编排层评审循环）
    #     · skeleton_review_max_rounds （同上的轮数上限）
    #   宿主代码（`sub_goal_solver__review.py` / `agent/skeleton_reviewer.py` /
    #   `_lean_dag_logic_check`）已同步删除。`blueprint_self_review`（有信号自审）
    #   与 `blueprint_or_expand_all`（A/B 参数）、`blueprint_deps_enabled` 保留。
    # ---- lemma 记忆（#30，跨题持久化）----
    lemma_storage_path: str = ""            # LemmaMemory 跨题持久化路径（空=仅内存）

    def __post_init__(self):
        """初始化**统一档位**配置表默认值。

        ★★ 2026-09-29（用户决策）：**删除三档制（fast/standard/deep），统一为单一档位**。
        理由（用户原话）：「我们是研究，肯定要研究最好的办法，而不是省资源的办法」。

        · 「按难度选择答题方法」的全部机制（DifficultyRouter 难度路由、
          PaperPacer 全卷时间池与 deep 配额、快车道旁路、calc_prewarm）已删除；
        · 下表保留 `"deep"` 作为**唯一键**，取值一律取**原 deep 档的最强配置**；
        · 不保留任何按档位分流的语义 —— 键名仅为兼容下游 `.get(tier, 兜底)` 的
          读取姿势（`ctx.tier` 恒为 `"deep"`），不再有任何分支判据依赖它。
        """
        if self.max_subgoals_by_tier is None:
            # 原 {"standard": 8, "deep": 16} → 统一取 deep 值 16。
            # （2026-09-15 用户：「子目标上限太少了」⇒ 取更宽的 deep 值）
            self.max_subgoals_by_tier = {"deep": 16}
        if self.tier_sample_times is None:
            # 原 {"fast": 1, "standard": 2, "deep": 3} → 统一取 3。
            # （2026-09-04：deep 4→3，配每候选 3 票，验证成本 12→9 票 ≈ -25%）
            self.tier_sample_times = {"deep": 3}
        if self.tier_temperatures is None:
            # 原三档温度分层 → 统一取 deep 的 4 层（保证候选多样性最大）。
            self.tier_temperatures = {
                "deep": [0.1, 0.3, 0.5, 0.7],
            }
        if self.tier_voting_times is None:
            # 原 {"fast": 1, "standard": 1, "deep": 3} → 统一取 3。
            self.tier_voting_times = {"deep": 3}
        if self.tier_max_completions is None:
            # 原 {"fast": 0, "standard": 1, "deep": 2} → 统一取 2。
            self.tier_max_completions = {"deep": 2}
        if self.tier_max_calls is None:
            # 原 {"fast": 6, "standard": 30, "deep": 100} → 统一取 100。
            # 历史：v2.6.1 deep 30→60（三Agent协作反复验证需 24+ 次调用）；
            #      v2.8.1 deep 60→100（"协作 6 轮 + 子目标 + Lean + 投票 + revise +
            #      oracle" 链路下 60 仍耗尽，触发大量「跳过 LLM 调用」）。
            # 统一后每题都可能有协作链，故取 deep 的 100（研究阶段不省调用）。
            self.tier_max_calls = {"deep": 100}
        if self.tier_budget is None:
            # 2026-10-01 按用户决策：研究期不限时，此值仅作挂死兜底，不参与调度决策；
            #   调度侧已不再有任何按剩余时间降级的逻辑（soft_budget/gen_deadline/verify_only 触发路径已删）。
            # 查阅：CHANGES_2026-09-29.md §0.2
            self.tier_budget = {"deep": 86400.0}

        # ============================================================
        # env → config 桥（2026-10-09 修复「env 对 getattr 开关静默失效」）
        # ------------------------------------------------------------
        # `contracts/switch_registry.py` 承诺读取优先级「env > AgentConfig 覆盖 >
        # 默认值」，但此前**缺 env→config 这一段** ⇒ 只被
        # `getattr(self.config, X)` 读取的开关**完全无视 env**（注册表认了 env、
        # 消费方读的却是 config 的旧值）。本调用把**环境变量里确实存在**的开关值
        # 写回本对象的同名字段，使 `getattr(config, 字段)` 恒等于注册表生效值
        # （`param_usage` 因读 `getattr(cfg, ...)` 也随之自动纠正）。
        #
        # ★ 只写 env 里真实存在的项 ⇒ env 缺失时本对象的显式值原样保留，
        #   不破坏「config > 默认值」这一半语义。
        # ★ 必须放在 `__post_init__` **末尾**：前面是档位表初始化，不改动其逻辑。
        # ★ 懒导入（非模块级）：本模块对外契约是「只依赖 stdlib」，模块级 import
        #   contracts 会破坏 user_agent 的降级导入契约 ⇒ 仅在构造时按需导入。
        # ★ import 与调用**双重 try/except**：桥失败绝不使 AgentConfig 构造失败。
        #   import 方向：`contracts/` 是最底层（零依赖），根层向下 import 合法；
        #   已核 `contracts/switch_registry.py` 不反向 import user_agent。
        try:
            from contracts.switch_registry import apply_env_to_config
            apply_env_to_config(self)
        except Exception:  # noqa: BLE001  桥失败不得影响配置构造
            pass
