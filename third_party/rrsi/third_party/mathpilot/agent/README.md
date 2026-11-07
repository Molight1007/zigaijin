# `agent/` — L4 编排 / L3 契约 / L2 业务

**放什么**：解题主链：流程编排、阶段串联、业务逻辑、数据结构契约。

**不放什么**：一次性脚本、部署脚本、纯通用工具。

**依赖规则**：L3 契约、L1 工具；**禁止**被 `tools/` 反向依赖（现存 6 处违规，见结构规范 §1）。

## 文件清单（27 个 py / 24,380 行，自动生成）

| 文件 | 一句话职责 | 行数 |
|---|---|---|
| `sub_goal_solver.py` | 子目标求解智能体（SubGoalSolverAgent） 将数学问题分解为有序子目标，逐步求解后合并结论。 | 3753 |
| `orchestrator.py` | 编排器（Orchestrator）—— 简化版 借鉴 ss-main 的简洁流水线， | 3728 |
| `solver.py` | 通用求解智能体（SolverAgent） 把原 ``ReasoningAgent._generate_can… | 2476 |
| `calc_tool.py` | agent/calc_tool.py —— 确定性精确计算工具（2026-08-31）。 | 1787 |
| `verifier.py` | 过程校验智能体（VerifierAgent） 功能演进： - v1: 二元投票（VERDICT A/B）+ … | 1587 |
| `formatter.py` | 答案规范化智能体（FormatterAgent） 把原 ``ReasoningAgent`` 末尾的答案抽取… | 1438 |
| `base.py` | 多智能体基础组件 提供： - ``Candidate`` / ``Verdict``：候选解答与验证结果的数… | 1330 |
| `blueprint_planner.py` | BlueprintPlanner —— | 1017 |
| `gap_analyzer.py` | 逻辑缺口识别（2026-09-29 新增，截图 #6）。 | 679 |
| `dag_reviewer.py` | DagReviewer —— | 602 |
| `question_type.py` | 题型识别与差异化策略（v2.6）。 | 576 |
| `theorem_hint.py` | 领域 → 定理检索（2026-09-29 新增，截图 #3+#4「老师重点关注」）。 | 550 |
| `deterministic.py` | 确定性验证通道（DeterministicChecker） 纯本地、零 LLM 预算的答案旁证/硬否决验证器… | 528 |
| `tool_gateway.py` | 统一工具调用门面（ToolGateway）—— | 508 |
| `param_usage.py` | 可调上限「配置 vs 实测」记录（2026-09-21）。 | 495 |
| `symbolic_solve.py` | 符号化方程求解通道：数值剥离 → 符号建模 → 硬校验 → 工具求解（2026-09-12）。 | 488 |
| `audit_gate.py` | AuditGate —— 答案审核闸门（取代 Lean 系在检测链中的全部作用）。 | 422 |
| `answer_falsifier.py` | 候选答案证伪层（AnswerFalsifier，2026-09-23 新建）。 | 404 |
| `adversarial_verifier.py` | 对抗式验证器（#16 改造）：**证伪**而非验证。 | 342 |
| `collaborative_solver.py` | 难题三Agent协作求解器（CollaborativeSolver，v2.6）。 | 339 |
| `classifier.py` | 题型识别智能体（ClassifierAgent） 关键词匹配优先（借鉴 math_agent-main 的成… | 297 |
| `answer_oracle.py` | 统一客观答案验证层（AnswerOracle，v2.7；2026-09-06 去 Lean 化）。 | 294 |
| `skeleton_reviewer.py` | SkeletonReviewer —— | 248 |
| `lemma_memory.py` | LemmaMemory —— | 162 |
| `symbolic_model.py` | L2 独立符号建模复核：解析与求值（2026-09-10）。 | 152 |
| `value_attack.py` | 数值攻击验证器（C-lite，2026-09-03 老师拍板）。 | 122 |
| `__init__.py` | MathPilot 多智能体包 对外暴露核心类， | 56 |

---

> 本文件由 `代码审查_2026-10-01/scripts/gen_dir_readmes.py` 自动生成，职责句取自源码 docstring 首行。
> 目录结构与分层规则见 `docs/项目结构规范_2026-10-01.md`；可视版见 `代码审查_2026-10-01/代码地图_MathPilot.html`。
