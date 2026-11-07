# -*- coding: utf-8 -*-
"""`AgentConfig` 前半字段（2026-10-08 拆分到本模块）。

`AgentConfig(_AgentConfigBase)` 继承本类；dataclass 继承语义下
`AgentConfig.__dataclass_fields__` = 基类字段(前) + 派生字段(后) = 原顺序。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _AgentConfigBase:
    """AgentConfig 前半字段（继承合并后字段集合/顺序与原类一致）。"""
    # 策略模型（解题）
    # P0-4 修复：候选 3→2（平台并发=3，且 3 候选×3 重试曾耗尽单题预算 → 45 error）
    # v2.4.0：恢复 24576 上限（ICMA 对齐）。ICMA 实测同模型首轮 24576 仅 143-231s，
    # 模型实际只用 3-7K token，24576 只是上限；降上限会牺牲贴上限的奥赛题成功区间。
    # 真正修超时靠：结构化四章节 prompt（抑制自由 CoT）+ 预算感知 + 压缩 prefill 兜底。
    # 2026-09-04：比赛不限制模型 token（单题边界=平台 1200s 时间墙）→ 上限放开 65536
    #   （2534334 平台实测 truncated 238 次答案被腰斩 → 64 invalid；截断比耗时更丢分）。
    policy_sample_times: int = 2       # 候选解答数量
    policy_temperature: float = 0.3    # 策略采样温度（提高以增加多样性）
    policy_max_tokens: int = 65536     # 策略最大 token（上限；9/4 放开，只受 1200s 时间墙）

    # 蓝图分解（2026-10-01 研究期按用户决策**默认开**；原为"简化版：关闭蓝图"）
    use_blueprint: bool = True         # 蓝图太长，Intern-S 思维流先被蓝图占满
    # 2026-10-01 按用户决策默认开（研究期不省资源）

    # 验证模型（评判）
    verifier_voting_times: int = 1     # 每个候选只投 1 票（避免无效重复投票）
    verifier_temperature: float = 0.0  # 验证温度（贪婪解码）
    # ★★ 2026-09-16 投票方差（针对实测的「零否决」）：
    #   实测 standard 档 `verifier_voting_times=1` 且 `_vote_one` 温度**硬编码 0.0**
    #   ⇒ 每题只有一次分类判断，多票也无方差（"多票"= 同一判断重复 N 次）；
    #   后果是 6 道错题中 5 道的**全部候选被全票判 A**。
    #   现：当**候选答案之间存在分歧**时，自动提高票数并启用非零温度；
    #   候选答案一致时保持原样（加票无意义且费时）。
    verifier_diversify_enabled: bool = True
    verifier_disagreement_votes: int = 3          # 有分歧时的每候选票数
    verifier_disagreement_temperature: float = 0.7  # 有分歧时的采样温度
    # ★ 2026-09-15 新增：带推理的最终复核（回应"验证器偏松"）。
    # 依据：常规投票走 prefill 强制单行输出（自述实测 0.8s vs 普通 70.2s），
    # **0.8 秒不可能完成"独立重算 + 逐条攻击"** ⇒ 投票退化成"看一眼点头"。
    # 实测后果：10 题里 5 道错题的**全部候选全票 A**，叠加 AuditGate 候选审核
    # 100% unknown、LeanGate 判 valid / 降级放行 ⇒ 三道闸门对错答**零否决**。
    # 本项对**最终选定答案**做一次不 prefill 的复核（1 次调用/题，可否决）。
    # ⚠ 默认关：未 A/B 前行为完全不变。
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    verifier_deep_final_enabled: bool = True
    # ★ 2026-10-04（B 折中）：投票是否放开**完整推理**（不 prefill）。
    #   `off` = 旧行为（0.8s/票，但代码自陈"看一眼就点头"）；
    #   **`disagree` = 默认**：仅当**候选答案分歧**时放开完整推理；
    #   `always` = 每票都放开（最准，约 70s/票 ⇒ 18 票 ≈ 21 分钟/题）。
    #   用户决策：「先试 B（折中），效果不理想就切 A；**正确率第一位**」。
    verifier_vote_reasoning: str = "disagree"
    verifier_deep_final_min_remaining: float = 150.0  # 剩余时间低于此值则跳过复核
    verifier_deep_review_max_tokens: int = 16384      # 复核输出上限（需容纳推理链）
    # ★ 2026-09-15 实测教训：用投票模板做复核时，模型只吐 **12 个字**
    # （一行 `VERDICT: A`）——因为投票模板写着"在心里完成即可，不输出"+
    # "请只输出 VERDICT"。该长度的输出**必须视为"未完成复核"**，
    # 否则"没做检查"会被当成"检查通过"。
    verifier_deep_review_min_chars: int = 200
    # ★ 2026-09-15 审计补漏：此键原先只被 `getattr(cfg, ..., 0.0)` 读取、
    # **没有声明、也不在白名单** ⇒ CLI/overrides 传它会被静默丢弃
    # （与 enable_dag_replan / symbolic_solve_adopt 同一类坑）。
    verifier_deep_review_temperature: float = 0.0

    # 题型分类（可选）
    enable_domain_hint: bool = True    # 是否启用领域提示增强
    enable_question_type: bool = True  # 是否启用题型识别（证明/选择/判断/填空/解答）+ 差异化策略
    # 2026-09-12 客观题特化解法（用户要求"保证检测到题型并采取特化技巧"）：
    # 选择/判断/填空三类客观题强制注入特化纪律（逐项判真+反例否证+清点选项 /
    # 绝对化措辞找反例+回定义核对 / 按空序逗号分隔），与"证明题不分流"的既有
    # 结论互不冲突（那一条针对 IMO 证明题）。回退：设为 False 即恢复旧行为。
    objective_tactic_enabled: bool = True
    # ★ 2026-09-16 审计修复（原为"假开关"）：solver.py 的注释承诺
    #   「`enable_question_type_hint`（默认 False）仍可**全题型强制开启**」用于 A/B，
    #   但该键此前**未在 AgentConfig 声明、不在白名单、无 CLI**
    #   ⇒ 只能走 `getattr(..., False)` 兜底，A/B 路径结构上不可用。
    #   现补全声明 + 白名单 + CLI（`--enable_question_type_hint true`）。
    enable_question_type_hint: bool = False
    # ★ 2026-09-16 审计修复（原为"假开关"）：orchestrator 注释称
    #   「config.verify_reserve_seconds 可覆盖」生成侧预留，但同样三处皆缺。
    #   实际语义：`_gen_deadline = deadline − verify_reserve`，
    #   生成侧软截止 = 该值；deep 档默认 540s、其余 480s。
    verify_reserve_seconds: float = 0.0  # 0 = 用按档位的默认（deep 540 / 其他 480）
    # ★ 2026-09-16 新增：联网搜索工具开关（`agent/base.py::llm_with_tools` 注册）。
    #   开启后模型可在原生 tool_calls 里自行调用 `web_search(query)`；
    #   逐题效果见 `tool_calls.web_search`（calls/ok/fail/results/verdict）。
    #   ⚠ 后端实测可用性（2026-09-16）：Math StackExchange ✓ / arXiv ✓ /
    #     Bing △（能连通但结果不可用）⇒ 已按此次序回退。
    #   （原理由：默认 False —— 工具已实现并接线，但不经 A/B 不改变主链行为。）
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    enable_web_search: bool = True
    # ★ 2026-09-17 新增：Lean 同答案候选去重缓存（**默认关**）。
    #   实测 003 的 7 候选里 5 个同答案，重复验证浪费约 120s；
    #   但复用报告会绕过"按各自 reasoning 判定"的守卫② ⇒ **改变验证语义**，
    #   收益（5.6%）不足以承担 ⇒ 默认关，供 A/B（`--lean_dedup_by_answer true`）。
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    lean_dedup_by_answer: bool = True

    # ---- 2026-10-01：`<calc>` 计算工具板块整体删除 ----
    # 用户决策：「这个板块是错误的，本身就有问题」⇒ 删除 `<calc>` 协议与工具化
    # 计算（原 `enable_calc_tool` / `calc_mandatory` / `calc_hard_only` /
    # `tool_calc_enabled` / `answer_selfcheck_enabled` / `subgoal_calc_router`
    # 六个开关一并移除）。纯数学求值内核迁至 `utils/math_eval.py`；符号核验通道
    # （`symbolic_crosscheck_enabled` / `symbolic_solve_enabled`）不受影响。
    # ★ 2026-09-29：`enable_calc_prewarm` 字段已随"生成前算式预计算"（方案 B /
    #   阶段 2.65_calc_prewarm）**整体删除**（用户决策：没必要、不合逻辑）。
    #   原字段的实测记录（保留备查）：12/12 题都产出预计算，但只有 1 题与答案相关；
    #   根因是"让模型在解题前凭题面猜该算什么"这一前提不成立；成本 30–111s/题。
    #   计算部分将另行设计。
    # 2026-09-12 用户要求「原本完成的子目标要记录，不能重头再来」：
    # True → 同一题的**已完成子目标结果**跨 run() 调用复用（零 LLM）。
    # 背景：orchestrator 会在 3_solve / 3.5 等阶段多次调用 sub_goal_solver.run()，
    # 原实现每次都重新规划 + 重解全部子目标，重复烧掉本就吃紧的单题预算。
    # 设 False 回到旧行为（每次全量重解）。
    subgoal_reuse_done: bool = True
    # 2026-09-12 用户要求：子目标级「失败二选一」处置。
    # True → 某子目标未产出有效结论时，优先**重做该子目标**（≤2 次）；重做仍失败
    # 或失败原因指向规划/依赖前提 → 记 subgoal_replan_needed 并写入 revise_feedback
    # 交上游重规划（不在子目标循环内新建重规划，避免破坏 depends_on 顺序）。
    # 设 False 回到旧行为（失败即占位放行）。
    subgoal_adaptive_recover: bool = True
    # 2026-09-10 L2（用户 9/10 思路 + 李平老师 9/9 建议）：**独立符号建模复核**——
    # 让模型当"数学问题拆解助手"，只把给定数值抽象成变量并输出目标量表达式
    # （禁止自算），由 utils/math_eval 精确代入求真值，与主链答案比对；不一致则打回
    # 一次，仅当新答案落回该真值才采纳。每题最多 1 次调用。默认关待 A/B。
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    symbolic_crosscheck_enabled: bool = True
    # 2026-10-02 DeepSeek 适配：原 512 ⇒ 8192。原值基于「建模输出很短（EQUATIONS+TARGET）
    # + prefill 抑制思维块」的 Intern-S 时代假设，对 reasoning 模型必然截断
    # （reasoning 先吃满预算、正文为空）。
    symbolic_max_tokens: int = 8192
    # 2026-09-12 符号化方程求解通道（用户 9/11 需求）：智能体把题面里「显式给定
    # 的具体数值」剥离存本地，给模型的是**未知数类型**题面；模型只输出方程（组）
    # + 求解目标，具体数值由本地 SymPy 回代算出 —— **模型不参与任何计算**，
    # 且"计算方程式交给工具、接收工具返回结果"。
    # 省时间设计（用户 9/12 硬要求）：剥离/校验/求解全部本地（毫秒~秒级）；
    # 只加 1 次**短**建模调用；工具值与主链答案一致时 **0 额外调用**，
    # 仅分歧才追加 1 次短回传；**不新增候选**（不触发下游验证/Lean 额外成本）。
    # 2026-10-01 按用户决策默认开（研究期不省资源）。
    symbolic_solve_enabled: bool = True
    # 2026-10-02 DeepSeek 适配：原 384 ⇒ 8192。原值基于「建模输出很短（EQUATIONS +
    # TARGET）+ prefill 抑制思维块」的 Intern-S 时代假设，对 reasoning 模型必然截断
    # （reasoning 先吃满预算、正文为空）。
    symbolic_solve_max_tokens: int = 8192
    symbolic_solve_feedback: bool = True     # 分歧时把工具结果回传模型定稿
    # 2026-09-12 方案④「把计算从模型手里拿走」：工具求解成功后**答案直接取
    # 工具值**（模型原先的数值被丢弃），而不是只做事后比对 —— 这样最终答案
    # 在架构上由本地计算器产生，模型无法心算。设 False 回到"仅比对/回传"旧行为。
    symbolic_solve_adopt: bool = True

    # ★ 2026-09-23 新增：候选答案**证伪器**（agent/answer_falsifier.py）。
    #   动机（0923 四维分析实证）：46 题里 25 题「全部候选 0 票」⇒ 触发零票兜底、
    #   整池被弃、改取 direct_solve 的另一条产线答案 ⇒ 18 题终答出池、16 题判错
    #   （占全部错题的 43.2%）；且 021 里错答与 gold **同时**拿到 3/3 票 ⇒ 投票无判别力。
    #   用户判断"错误答案一定是能证明错误的"，故补一条**只做证伪**的通道：
    #   数值回带（LLM 出闭式等式）→ SymPy 精确判定 → Lean 背书（编译通过则推翻证伪）。
    #   红线：宁可漏证，绝不误杀；任何不确定一律 unknown。
    enable_answer_falsifier: bool = True
    # ★ 2026-10-04 修（D6）：① 语义由「0 = 禁用证伪」统一为「**0 = 不限**」
    #   （全项目口径；见 `answer_falsifier._spend_cap`）；② 默认 2 → **0（不限）**。
    #   依据：证伪是当前**唯一能客观淘汰错误答案**的信号（LLM 证书 → SymPy 精确
    #   → Lean 背书），在「不限调用」口径下没有理由按"省一次短 LLM 调用"设 2 次/题。
    falsify_max_per_q: int = 0               # 单题最多证伪次数（0 = 不限）

    # ★ 2026-09-23 新增：把 LeanSearch 检索到的 Mathlib 定理注入**解题侧**引理记忆。
    #   实测（0923 效能审计）460 条检索结果里只有 4 条（1%）进了 Lean 代码；
    #   根因是检索结果此前**只注入验证器**（`leansearch_inject_verifier`），
    #   解题器从未收到 ⇒ 检索在最该用它的环节缺席。置 False 可回退。
    inject_mathlib_to_solver: bool = True

    # ★ 2026-09-23 新增：`6.5_audit_gate` 对**非证明题**早退（不再产出 unknown 噪音）。
    #   实测：37/46 题触发、235 条 verdict **全 unknown**（对解答题结构性无输出，
    #   见 audit_gate.py:255-268 的既有说明），却仍耗 0.37h 并污染 diag 统计。
    #   置 False 可回退到"照常跑但全 unknown"的旧行为。
    audit_gate_skip_non_proof: bool = True

    # ★ 2026-10-07 新增（fix-audit-gate）：**解答题确定性判别重写**总开关。
    #   诊断（smoke10 = run_2026-10-06_smoke10_1006_2347.jsonl，10 题实测）：
    #     · audit_gate 38 条 candidate_audit 全 verdict=unknown / applicable=false；
    #     · 根因两层：① **Level0 死在第一步** —— 答案仍带 `\boxed{...}` 外壳，
    #       `_safe_parse('\boxed{-4}')` → None ⇒ method="parse" 即返回 unknown，
    #       7/10 题命中（题面要求 within \boxed{}，模型照做反而解析失败）；
    #       ② 即使剥壳，原判据只认「最后一条一侧为常量的单变量等式」，而 8 道解答题
    #       全是竞赛题（计数 / 函数方程 / 极值 / 集合完备性）⇒ 无此结构 ⇒ 全 unknown。
    #   置 True（默认，修好后的行为）：解析前归一 + 全量等式回代 + 函数方程回代。
    #   置 False：完全回退改动前行为做 A/B。
    deterministic_solution_check: bool = True

    # ★ 2026-10-07 新增：是否允许 AuditGate 的 Level2 rubric 对**解答题**运行。
    #   旧行为 rubric 仅对证明题运行（解答题结构性不适用）；经查 rubric 提示词为通用
    #   「数学解答评审」、无证明专用假设 ⇒ 原限制缺少依据。默认 False（保守）：
    #   rubric 系 LLM 同源自评（价值低、耗时高），且候选级投票链本已可选跑 rubric。
    #   置 True 则解答题也走 Level2（仍沿用高置信 B 才否决的阈值）。
    audit_gate_rubric_nonproof: bool = False

    # ★ 2026-09-16 删除死字段 `extraction_mode`：AgentConfig 声明了、
    #   override 白名单里也列了，但**全仓 0 个读取点**（实测确认）⇒
    #   CLI/kwargs 传它会被静默丢弃，只给人"可调"的假象。

    # ---- 自主调控（大幅缩减）----
    # 2026-09-17（Audit-1）：本字段此前**声明 + 白名单 + CLI 全齐，但全仓零读取点**
    #   ⇒ `--revise_rounds N` 被 __init__ 写入 config 后无人消费，属**静默失效的 CLI**。
    #   现接到 `orchestrator._deep_revise_loop` 的**全局轮数上限**（该上限原先硬编码 5）。
    #   默认取 5 以保持既有行为不变；<=0 表示不设全局上限。
    #   注：单次调用的轮数由 `deep_revise_rounds` 控制（两回事）。
    # ★ 2026-10-06 新增（**上游缺口修复**）：候选入池前是否要求它真的含一个最终答案。
    #   实测 32 个候选里 **59% 的 `answer` 是推理句 / 元话语 / 英文片段**
    #   （如 `'因此，似乎没有满足条件的路径，答案为 0。'`、`'让我搜索这个问题的标准解法'`），
    #   却照样入池 ⇒ 被投票（题 002 的 6 个候选里 4 个逐字相同）、
    #   被检错（检错器在验垃圾）、被拿去修正 ⇒ 污染整条下游链。
    #   置 True：不含答案的候选**不入池**并留痕（宁可池子少一个，不放"假候选"）。
    #   置 False：回退旧行为（全部入池）做 A/B。
    require_answer_in_pool: bool = True
    # ★ 2026-10-07 新增（问题 2.4 第二轮）：候选**入池门槛 v2** —— 上一轮门槛只覆盖
    #   `_generate_initial`/`_generate_revise` 主路径，实测仍有漏网（题 002
    #   `'证明这个最小值就是所求的 $T$，并验证其最优性'` 入池且成唯一幸存候选、被
    #   final_select③ 直接采用；题 000/007 的 `子问题4：…` 类描述句入池）与误杀
    #   （题 087 `BD`/题 088 `AB` 各被拒 4/5 次）。置 True（默认，修好后行为）：
    #   ①判据升级为 `looks_like_pool_answer`（补漏网 / 修误杀）；②穷举全部 6 处写入
    #   `ctx.candidates` 的入口。置 False 完全回退上一轮口径做 A/B（详见注册表）。
    pool_gate_v2: bool = True
    # ★ 2026-10-07 新增（问题 2.9）：`<check>` **协议守卫** —— 实测题 000 大量把中文
    #   句子塞进 `<check>`（`<check>图 G 是弱连通的</check>`）⇒ 不可解析 ⇒ 子目标客观
    #   判定一条都出不来（`n_judged = 0/73` 直接原因之一）。置 True（默认，修好后行为）：
    #   违规定性 + 留痕（可见）+ 一次定向重问。置 False 回退旧行为（静默跳过）做 A/B。
    check_protocol_guard: bool = True
    # ★ 2026-10-06 新增：Lean/Audit 门把候选**全部拒掉**时的处置。
    #   默认 False = 只显式留痕（不静默保留，但也不清空）；
    #   置 True = 清空候选池，触发下游兜底/重生成链。
    strict_all_rejected: bool = False
    # ★ 2026-10-06 新增：候选**零多样性**（全题仅 1 种答案）时是否触发高温度补充采样。
    #   实测题 000 的 5 个候选答案全同 ⇒ 置信度被刷到 1.00 ⇒ 直接放行（答案是错的）。
    #   「答案一致」在采样无多样性时是**采样失败信号**，不是可靠性证据。
    # ★ 2026-10-07 新增（fix-gate-timing-diag 问题 2.3）：AcceptGate 真实生效。
    #   背景：`base.RoundState` 契约阈值 ACCEPT_PASS=5 / REJECT_DEFECT=10 按"无上限
    #   连续回环"标定，而 `_update_accept_gate` 唯一调用点受 `max_revise_rounds`(=5)
    #   约束 ⇒ 单题最多调用 ≤5 次（smoke10 实测 ≤2）⇒ 阈值结构性不可达、
    #   decision 恒为 HOLD（10/10 题）⇒ 空闸门。置 True（默认，修好后行为）：
    #   阈值改可达口径、ACCEPT/REJECT 均改变 revise 行为；置 False 回退旧行为。
    # ★ 2026-10-07 新增（问题 2.8）：revise 回环墙钟归属修正（纯可观测性）。
    #   实测题 001 revise_round=4 却记 `4_verify=5546s`、`5_revise_or_fallback=7.4e-06s`；
    #   题 000 同机制却记 3683s ⇒ 归属不一致。置 True（默认）统一归到 5 段；=0 回退。
    max_revise_rounds: int = 5
    max_total_calls: int = 150         # LLM 调用预算硬上限（v2.6.1：15→60；v2.7：60→150，
                                        # 覆盖 5 题 batch + 限流重试 + deep 档完整流程
                                        # = classifier 1 + 求解 3 + 投票 6 + self_audit 1
                                        # + revise 1 + lean 转换 1 + collab 6 轮
                                        # + sub_goal 规划 1 + N 个子目标 ≈ 25-30 次/题）

    # ---- 时间限制（2026-09-13 按实测重设；旧值 1200 / 22500 已废止）----
    # 【为什么重设】195 题实测（最近三代代码，`results/*.jsonl` 的 `stage_timers`）：
    #   · deep 档各阶段耗时 **p90 合计 = 2454s**，而旧单题硬顶只有 1200s
    #     ⇒ 绝大多数难题跑到一半被硬墙砍断（deep p50 实测 1230s，恰好压在墙上）；
    #   · standard 档 p50=981s / p90=1338s ⇒ 540s 档位帽严重不足；
    #   · fast 档 p50=681s（设计值仅 120s）⇒ 档位帽形同虚设。
    # 【为什么能放开】平台侧已确认**不设单题时限**：平台评测日志 `deadline_seconds: 0`，
    #   且实测整卷跑 8.89h 未被终止。旧注释里"平台单题硬限 20min / ICMA 同款 1200s"
    #   是跨赛事移植的臆测值 —— 全仓无任何平台下发字段支撑（无 deadline_seconds 解析）。
    # 【防卡死】放开的前提是"单题内部不卡死"，由三线兜底：
    #   (a) PaperPacer 全卷动态预算；(b) 每阶段 `_phase_deadline_guard` 上限；
    #   (c) 各生成循环的 `gen_time_up()` 检查（见 2026-09-13 审计修复）。
    # 关系仍须满足：paper_target_time < max_total_time_seconds。
    # ⚠⚠ 2026-09-13 更正（权威出处推翻此前判断）：
    # 单题 1200s 是**平台硬限**，不是我们能放开的。官方 baseline 仓库
    # github.com/InternLM/Challenge-Cup-2026 的 README「正式评测的并发、时限与
    # 计分」原文：
    #   「每道题的独立进程组有 **1200 秒硬时限**，包含选手模块加载、Agent 初始化
    #     和 solve 执行。超时后平台会终止整个进程组，不保证执行 finally、退出钩子
    #     或其它清理逻辑。」
    #   同节配套：Agent 阶段总硬时限 **6 小时**（不含 Judge）、最多同时运行
    #   **3 个**独立题目进程、超时/异常/缺失题**计 C 且分母不变**。
    # 此前"平台不设单题硬限、1200s 系自设"的判断**是错的** —— 错因是：我们代码
    # 里确实没有解析 deadline_seconds，但平台是**硬杀进程组**，无需下发字段；
    # "整卷 8.89h 未终止"那组数据来自**本地评测**（本地无平台限制），不能外推。
    #
    # 由此的硬结论：
    #   ① 单题**必须主动**在 1200s 内交卷 —— 超时 = 进程组被杀、finally 不执行、
    #      答案根本写不出来 ⇒ 该题计 C（0 分）且分母不变；
    #   ② 全卷 6h × 并发 3 = 64800 题·秒 ÷ 112 题 = **578s/题**。单题 1200s 只对
    #      少数难题可用；若每题都用满，只能做完 54 题，其余 58 题全部超时计 C。
    #      **平均 578s 才是真正的约束。**
    #   ③ 故本字段必须严格 < 1200，给"模块加载 + Agent 初始化 + 格式化输出"留余量。
    # 2026-09-14 上调：1000 → **1150**（用户要求："1000s 太少"）。
    # 演化史与依据（三段，务必连着看）：
    #   · 原值 1150。2026-09-13 二轮下调到 1000，触发点是实测 `official112-000`
    #     跑了 **1201s**，**越过平台 1200s 硬限**（平台会终止整个进程组、该题计 C=0）。
    #   · 当时跨墙的根因是 `agent/base.py` 的单次 `client.chat` **不可中断**，
    #     而 `LLMClient` 超时 180s × **重试 1 次** ⇒ 单次调用最坏 **360s**，
    #     阶段/单题截止点只能在"两次调用之间"生效，故必须为在途调用留足空间。
    #   · 2026-09-13 晚已拆掉那个放大器（`utils/llm_client.py`：**超时不再重试**，
    #     读超时 180→120s）⇒ 单次调用最坏从 **360s 降到 120s**，
    #     "上限必须压到 1000" 的前提不再成立，故回调到 1150
    #     （= 平台硬限 1200 − 50s 模块加载/收尾余量，与 `tier_budget.deep` 对齐）。
    # ✅ 2026-09-14 实测落实（不再是推算）：1150 → **1100**
    #   证据（两轮同批错题实测）：
    #     · 并发3 轮：000 = **1164.7s**（超 1150 上限 14.7s）⇒ 余量仅 35s
    #     · 并发1 轮：014 = **1211.0s**（超上限 **61s**）⇒ **1211 > 1200，越墙**
    #   ⇒ 在途调用的超限幅度可达 60s 量级，"1150 留 50s" 不够。
    #   取 1100 = 1200 − 60(实测最大超限) − 40(模块加载/收尾) ⇒ 留约 100s 实垫。
    # ⚠ 只动**这一处最后硬限**，`tier_budget.deep` **保持 1150 不变**：
    #   档位预算负责"资源分配"，压它会**提前掐断本可在 1200s 内跑完的题**；
    #   硬限负责"不许越墙"，只承担兜底。两者职责不同，故不联动调整。
    #   （生效值 = min(deadline, tier_cap, pacer) = 1100，档位侧仍按 1150 估资源。）
    # ⚠ 残留风险仍如实记录：上限在两次调用之间判定、无法中断在途调用，
    #   理论最坏 1100 + 120 = 1220s。缓解：`critical_tail_seconds`（默认 120s）
    #   让最后 120s 内不再启动可选步骤 —— 实测超限幅度（14.7s / 61s）远小于该值域。
    # 2026-10-01 按用户决策：研究期不限时，此值仅作挂死兜底，不参与调度决策；
    #   调度侧已不再有任何按剩余时间降级的逻辑（soft_budget/gen_deadline/verify_only 触发路径已删）。
    max_time_per_question: int = 86400  # 单题壁钟上限（秒，仅防挂死）
    # 2026-10-01 按用户决策：研究期不限时（原值 20700 = 赛期 6h 硬限 − 4% 余量）。
    # 此值仅作「进程永挂」兜底，不参与任何调度决策；单题上限见 max_time_per_question。
    max_total_time_seconds: int = 2592000  # 30 天（仅防永挂，非限制）

