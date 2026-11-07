# `prompts/` — L2 业务（提示词）

**放什么**：提示词模板：纯字符串常量与构造函数。

**不放什么**：业务判定逻辑、IO、网络调用。

**依赖规则**：尽量无依赖；需要时仅依赖 L3。

## 文件清单（12 个 py / 2,157 行，自动生成）

| 文件 | 一句话职责 | 行数 |
|---|---|---|
| `policy.py` | 策略提示词模板 — 用于生成候选解答。 | 382 |
| `sub_goal.py` | 子目标求解提示词 -- 用于 SubGoalSolverAgent。 | 345 |
| `verifier.py` | 候选解答验证提示词 — 用于过程校验智能体（VerifierAgent）。 | 322 |
| `blueprint.py` | Blueprint DAG 生成提示词 —— | 318 |
| `error_lessons.py` | 易错点记忆库（error lessons，2026-09-06 A 档轻量经验注入）。 | 246 |
| `dag_review.py` | DAG Reviewer 提示词 —— | 138 |
| `symbolic_model.py` | L2 独立符号建模复核提示词（2026-09-10）。 | 129 |
| `skeleton_review.py` | Skeleton Reviewer 提示词 —— | 99 |
| `proof.py` | MathPilot 证明题专用提示词 借鉴 math_agent 的 proof_solver 和 Inte… | 87 |
| `revise.py` | 纠错重解提示词 — 用于 SolverAgent 的自纠错重解（revise）模式。 | 54 |
| `dag_lean_check.py` | DagLeanCheck —— | 31 |
| `__init__.py` | —（无 docstring，建议补一句话职责） | 6 |

---

> 本文件由 `代码审查_2026-10-01/scripts/gen_dir_readmes.py` 自动生成，职责句取自源码 docstring 首行。
> 目录结构与分层规则见 `docs/项目结构规范_2026-10-01.md`；可视版见 `代码审查_2026-10-01/代码地图_MathPilot.html`。
