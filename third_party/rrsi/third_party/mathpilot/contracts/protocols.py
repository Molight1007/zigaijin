"""被 ``tools/`` 反向依赖的**业务上下文协议**（零依赖）。

★ 2026-10-07 架构解耦（阶段 B1）
================================
``TaskContext`` 是 agent 的**共享黑板**（实体在 ``agent/base.py``），字段多、
带惰性求值属性与时间方法 ⇒ **不搬实体**（方案 §8：动它牵一发动全身）。

但 ``tools/lean_local/*.py`` 只在**类型注解**里用到它（不继承、不 isinstance、
不构造）⇒ 用 **Protocol** 声明 "``tools/`` 真正用到的那部分"即可：
``tools/`` 从此只依赖本协议（``contracts``），**运行时不再 import agent**。

结构化子类型：``agent.base.TaskContext`` 无需显式继承本协议即自动满足它。

⚠ 本协议**只声明 tools/ 实际访问的成员**（见下方"使用处"注释），不做全覆盖；
   漏掉的成员只会让类型检查器报错，**运行时无影响**（注解是字符串，见各模块
   ``from __future__ import annotations``）。
   若未来 tools/ 用到了新成员，在此补一行即可。
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from contracts.types import Budget


class TaskContext(Protocol):
    """``agent.base.TaskContext`` 被 ``tools/lean_local/`` 使用到的子集。"""

    # ---- 基础字段（lean_gate / lean_pre_verifier / lean_refiner / lean_translator）----
    problem: str
    metadata: dict
    trace: list
    budget: Optional[Budget]
    domain: Optional[str]

    # ---- 前置形式化验证/骨架（lean_pre_verifier）----
    formal_spec: str
    formal_gaps: list
    preverify_trace: dict
    # 2026-10-12 批次 1（★ 2026-10-14 收敛）：题意理解（终态 + 答案槽 + 缺口 A/B）
    understanding_chain: dict
    understanding_gaps: list
    understanding_rounds: int
    # 2026-10-14 任务 #3：单一「题目目标」（题意理解的唯一产物；
    # 蓝图/子目标/merge 渲染共用，取代原先三处 problem_analysis 副本）
    problem_goal: dict
    # 2026-10-12 批次 2：蓝图多次生成
    blueprint_diffs: list
    blueprint_gen_rounds: int
    # 2026-10-12 批次 3：子目标链的 inputs 校验缺口
    subgoal_chain_gaps: list
    # 2026-10-12 批次 4：池空时汇总的全部错误点（交下游先找错点再生成）
    pool_empty_errors: list
    # 2026-10-13：分环节定理注入条数（stage → 实际条数，观测用）
    theorem_hint_stage_usage: dict
    # 2026-10-13（§3 方案 A）：题型判题决策与路径分流
    type_decision: dict
    type_path: dict
    sketch: str
    sketch_audit: dict
    sketch_tree: dict

    # ---- 蓝图/精炼（lean_refiner / lean_translator）----
    blueprint: dict
    refine_result: dict

    # ---- Lean 门禁诊断（lean_gate）----
    lean_gate: list
    start_time: float

    # ---- 动态私有缓存（非 TaskContext 声明字段，由消费方 getattr/setattr）----
    # lean_gate: `getattr(ctx, "_lean_verify_cache", None)` / `ctx._lean_verify_cache = ...`
    _lean_verify_cache: Any
    # ★ 2026-10-14：`_preverify_unconfirmed` 声明**已删** —— 唯一写入点在
    #   `lean_pre_verifier.run()` 的 fail 分支，全仓无读取方（死字段）。

    # ---- 时间方法（lean_gate / lean_pre_verifier / lean_refiner）----
    def time_remaining(self) -> float: ...

    def is_time_critical(self) -> bool: ...
