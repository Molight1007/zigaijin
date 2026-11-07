# -*- coding: utf-8 -*-
"""子目标阶段前置：阶段预算时钟装配（2026-10-08 批1：从 agent/sub_goal_solver.py 的
SubGoalSolverAgent.run 抽出）。

抽出段（run 内顺序）：
  · 阶段预算装配 —— 预算闸门（仍留在 run）之后：按档位解析
    `subgoal_stage_budget_sec(_std)`，计算 `_fixed_end` 并写入
    `ctx._subgoal_stage_start / _subgoal_stage_budget / _subgoal_stage_end`。
    `_stage_left` 闭包（仅 5 行、只读 ctx）**仍留在 run**，未随本段搬出。

**纯搬迁**：段内源码逐字搬入（仅去 4 空格缩进），调用点由 run 改为一行委托。
无 return / 无循环；局部名 `_tier_now` / `_raw_budget` / `stage_budget` /
`stage_start` / `_fixed_end` 段后均不再读取（AST 实测）⇒ 局部名零外泄。

参数 `self` 即 `SubGoalSolverAgent` 实例（沿用方法体原文，故保留名 `self`，
与 agent/orchestrator__prep.py 同口径）；`ctx` 为 `TaskContext`。
返回 None；结果写回 `ctx`。
"""
from __future__ import annotations

import time


def wire_subgoal_stage_budget(self, ctx) -> None:
    """见模块 docstring。`self` = SubGoalSolverAgent 实例。返回 None。"""
    # 2026-09-04 子目标阶段预算（老师：砍环节内重复、不砍环节本身；
    # 且不止加限制，还要保证体系在规定时间内跑完、强制收尾产出）。
    # 背景：preverify 提速省下的 ~300s 全被子目标贪婪 re-plan/re-review
    # 吃掉（bridge sub 529-717s → mcp sub 780-1112s），merge 被挤进
    # is_time_critical 区 → 走 fallback 而非真合并 → 060 对变错
    # （答案停在 S(1)..S(12) 枚举、未合成 2617/2618）。
    # 依据（bridge 18 题实测）：正确题 sub 全在 661-717s，失控题 780-1112s。
    # 预算取 750s：保住已知正确深度、刹住失控；可配 subgoal_stage_budget_sec。
    # 时钟挂 ctx（⚠ 不能挂 self：EvalEngine 的 agent 是共享单例，
    # ThreadPoolExecutor 并发 2 时同一实例同时服务多题，self 属性会竞态；
    # ctx 每题独立）。orchestrator 可能多次调用 run()（2.7 / deep 3_solve /
    # 3.5），共享 ctx._subgoal_stage_start，避免"每次调用重置预算"把总时间再吃一遍。
    # 2026-09-06 P1（用户拍板）：子目标预算按档拆分——deep 保留 750s
    # （难题深度分解值），standard/fast 用 450s（省下预算流向 solve/verify）。
    # 依据：2.7 子目标全档均 ~587s 是最大黑洞，standard 档 geom-051 曾 1073s
    # 仍错（post-fix 靠 3_solve 做对）——standard 题子目标分解性价比存疑。
    # 兼容：config 未显式设 std 档字段（测试 SimpleNamespace）→ 回退主字段
    # 语义（主字段=0 表示停用固定预算，仅 tail_reserve 兜底）。
    _tier_now = str(getattr(ctx, "tier", "standard") or "standard")
    if _tier_now == "deep":
        _raw_budget = getattr(self.config, "subgoal_stage_budget_sec", 750.0)
    else:
        _raw_budget = getattr(self.config, "subgoal_stage_budget_sec_std", None)
        if _raw_budget is None:
            _raw_budget = getattr(self.config, "subgoal_stage_budget_sec", 750.0)
    stage_budget = float(_raw_budget or 0.0)
    stage_start = float(getattr(ctx, "_subgoal_stage_start", 0.0) or 0.0)
    if not stage_start:
        stage_start = time.time()
        ctx._subgoal_stage_start = stage_start
    ctx._subgoal_stage_budget = stage_budget
    # 2026-10-01 按用户决策：研究期不限时 —— **删除"全局 deadline − tail_reserve"
    # 的尾部预留**（原逻辑逼子目标在 `全局 deadline − 180s` 前结束，为下游
    # 3_solve/format/gate 预留时间）。研究期不省资源，不再为下游阶段预留时间。
    # ⚠ 固定阶段预算 `subgoal_stage_budget_sec` 仍生效（见下方 _fixed_end）；
    #   是否一并放开由主理人确认（不在本次明确清单内）。
    _fixed_end = (stage_start + stage_budget
                  if stage_budget > 0 else float("inf"))
    if stage_budget > 0:
        ctx._subgoal_stage_end = _fixed_end
    else:
        ctx._subgoal_stage_end = 0.0

