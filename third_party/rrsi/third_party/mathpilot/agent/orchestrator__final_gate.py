# -*- coding: utf-8 -*-
"""6_format 段体（2026-10-07 阶段C2 抽出；2026-10-14 五道链定案后仅保留本段）。

历史沿革
========
本模块原名「终答审核闸门」，承载 `6.5_audit_gate`（final_answer_gate：答案级
Lean/AuditGate 终审 + 换候选重做循环）。2026-10-14 用户定案「五道审核链一步
到位直接删」：**6.5 整段删除** —— Lean 判断统一收口在投票前的 `4.3_lean_check`
（候选级、淘汰制），审核后重做循环随 5_revise 一并删除（全灭走 formatter 的
emergency 直答兜底）。阶段名 `6.5_audit_gate` 保留于 `tools/stage_audit.py`
审计清单作历史记录。

保留：`format_stage`（= 6_format 段体）—— 格式化输出 + 终答后处理
（P2 数值化 / P5 客观自检 / B0 形态闸门）+ `_pick_diag` 单源同步。
"""
from __future__ import annotations


def format_stage(self, ctx) -> None:
    # 6) 格式化输出
    self.formatter.run(ctx)

    # ★ 2026-10-04 修（D25，主链顺序）：终答后处理（P2 数值化 / P5 客观自检 /
    #   B0 形态闸门）在 6.5 闸门之前执行（原为之后）。「闸门验 A、实际提交 B」
    #   缺陷随之消除。★ 2026-10-14：6.5 闸门已删（五道链定案），本段即主链
    #   最后一道出口；后处理仍保持在格式化之后、提交之前，不变量不变：
    #   **凡是提交的答案，都经过本后处理**。
    ctx.final_response = self._final_answer_postprocess(ctx)
    # ★ 2026-10-07（问题 2.1 第三条路径补漏，fix-select-consistency 提供入口）：
    #   后处理会**改写答案内容**且不经过 `adopt_single` ⇒ `_pick_diag.picked`
    #   会与真正提交的 `ctx.final_response` 不同源。此处补第三条路径。
    #   无条件调用：`sync_pick_diag_to_final` 内部按开关 `final_answer_single_source`
    #   自行决定是否生效（OFF ⇒ 空操作），且任何异常自吞 ⇒ 无条件调用最简且不破坏行为。
    try:
        from .final_selector import sync_pick_diag_to_final
        sync_pick_diag_to_final(ctx, "postprocess")
    except Exception:  # noqa: BLE001  纯埋点同步，绝不阻断主流程
        pass
