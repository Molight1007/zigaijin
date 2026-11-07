# -*- coding: utf-8 -*-
"""deep 档三 Agent 协作改进（2026-10-07 阶段C3c：从 agent/orchestrator.py 的 Orchestrator.run 抽出 3.4_collab 段）。

**纯搬迁**：段内源码逐字搬入（仅去 8 空格的缩进），调用点由 `run` 改为一行委托。
`run` 保留 `self._stage_start(ctx, "3.4_collab")` 打点**原位不动**（阶段名是稳定
契约，被 tools/stage_audit.py / tools/test_report.py 按名聚合）。
段后「3.5_subgoal_sup 已删除」的流程序言注释**留在 run 原位**（不属本段体）。

耦合实测（AST 自由变量分析）：本段对 `run` 的局部变量依赖**仅为 `ctx` 与 `tier`**；
段内 5 个局部名**零外泄**、无 `return`/`yield`、无循环。其余外部符号全部经 `self`
访问（`self._phase_budget` / `self._phase_deadline_guard` / `self._phase_deadline_restore` /
`self.collab` / `self.record` / `self.config`）与模块级 `time`，无其它依赖 ⇒ 本模块
仅需 `import time`。

返回 None；结果写回 `ctx`（candidates 等，由 `self.collab.run(ctx)` 完成）。
"""
from __future__ import annotations

import time


def collaborative_deep_improve(self, ctx, tier: str) -> None:
    """3.4 deep 档三 Agent 协作（解题→审查→整合→反复验证）。

    参数 `self` 即 `Orchestrator` 实例（沿用方法体原文，故保留名 `self`），
    `ctx` 为 `TaskContext`，`tier` 为当前档位（本项目恒为 "deep"）。
    """
    # 3.4) deep 档难题：三Agent协作（解题→审查→整合→反复验证）
    #      只要时间未到且未验证通过，CollaborativeSolver 内部反复循环，
    #      保证难题高正确率。
    if (tier == 'deep'
            and getattr(self.config, 'enable_collaborative_deep', True)
            and not ctx.state.emergency
            and not ctx.state.verify_only
            and not ctx.gen_time_up()):
        # 2026-09-13 **加阶段预算**（照搬 2.7 / 3_solve 的 P4 做法）。
        # 实测（004 复现）：本阶段**此前无任何上限**，单题烧掉 **691s**，
        # 把 `3.3`(自改进) / `4_verify`(投票) / `6.5`(闸门) 全部挤成 0s
        # ⇒ 错误答案未经任何后续防线直接提交（1012 ≠ 2024）。
        # 另：`3.3` 与 `3.4` 语义重叠（都是"生成后改进候选"），故二者
        # **共享"改进阶段总预算"**：`3.3` 先用、`3.4` 只用余额 ⇒
        # 既保住 3.3（论文实测 22.2%→31.1% 的强杠杆），又不让 3.4 重复吃时间。
        # 2026-09-13：改走**实测上限**口径（3.4 实测 max 691s × 余量）。
        # 3.3 与 3.4 语义重叠（都是"生成后改进候选"），继续共享同一额度：
        # 3.3 先用，3.4 只用余额。
        _improve_total = self._phase_budget(ctx, stage="3.4_collab")
        _imp_used = float(getattr(ctx, "_improve_phase_used", 0.0) or 0.0)
        _collab_cap = max(60.0, _improve_total - _imp_used)
        _collab_t0, _collab_saved = self._phase_deadline_guard(ctx, _collab_cap)
        try:
            self.record(ctx, "control", "deep 档启用三Agent协作验证机制")
            self.collab.run(ctx)
        finally:
            self._phase_deadline_restore(ctx, _collab_saved)
        self.record(ctx, "control",
                    f"3.4 阶段预算 {_collab_cap:.0f}s"
                    f"（改进共享预算 {_improve_total:.0f}s − 3.3 已用 "
                    f"{_imp_used:.0f}s；实耗 {time.time() - _collab_t0:.0f}s）")
