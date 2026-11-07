# -*- coding: utf-8 -*-
"""子目标「规划收尾」族（2026-10-08 批1第二增量：从 agent/sub_goal_solver.py 的
SubGoalSolverAgent.run 抽出）。

抽出段（run 内顺序）：
  · strip_and_mask_plan    —— P-20 剥「答案形态」+ K11 掩码 merge_strategy，
                              第二返回值改为 2.6 的唯一题目目标 ctx.problem_goal
                              （原 problem_analysis 已随任务 #3 删除）
  · finalize_plan_stats    —— 子目标统计（n/依赖边/入度）落 ctx + record 一条规划完成记录
  · build_plan_summary     —— 生成子目标规划摘要 + K11 摘要注入点掩码

**纯搬迁**：段内源码逐字搬入（仅去 4 空格缩进），调用点由 run 改为一行委托。
无 return / 无循环 / 无 break·continue。

引用 sub_goal_solver 的模块级符号（`_strip_answer_forms` / `_mask_asserted_values` /
`_k11_mask_on`）⇒ **函数内惰性 import**（单一来源不复制；避免顶层循环 import）。
参数 `self` 即 `SubGoalSolverAgent` 实例（沿用方法体原文，故保留名 `self`）。
"""
from __future__ import annotations


def strip_and_mask_plan(self, ctx, plan_data):
    """P-20 形态剥离 + K11 断言值掩码。返回 ``(merge_strategy, problem_goal)``。

    ★ 2026-10-14（任务列表 #3）：第二项**不再**是蓝图 / 子目标规划自产的
      `problem_analysis`（该字段已随题意理解收敛而删除），而是 2.6 产出的
      **唯一题目目标** `ctx.problem_goal`（`agent/problem_goal.py`）。
      本函数对它的职责只有"取出来"：目标里的终态/答案槽在渲染处
      （`problem_goal.render_problem_goal`）已显式标注为「**假设**，不得当作
      已证结论」，故此处不再做数值掩码。
    """
    from .sub_goal_solver import _strip_answer_forms, _mask_asserted_values
    # 2026-09-17（P-20）：剥掉规划结论里的「答案形态」，防其被当作指令
    merge_strategy = _strip_answer_forms(plan_data.get("merge_strategy", ""))
    # ★ 2026-09-22（用户口径）：**蓝图不得锁死答案**。
    #   P-20 只剥形态标记，自然语句里的断言数值仍会注入下游并锚死终答。
    #   此处再做一层：把「被断言的数值」换成占位符（不删句）。
    merge_strategy, _n_lock = _mask_asserted_values(merge_strategy)
    if _n_lock:
        self.record(ctx, "blueprint_lock_guard",
                    f"蓝图 merge_strategy 内嵌 {_n_lock} 处断言数值 → 已屏蔽为占位符"
                    "（蓝图只构建信息、不产出答案；2026-09-22 用户口径）")
    return merge_strategy, dict(getattr(ctx, "problem_goal", None) or {})


def finalize_plan_stats(self, ctx, subgoals, merge_strategy) -> None:
    """子目标统计落 ctx.subgoal_stats + record「子目标规划完成」。返回 None。"""
    _sg_stats = self._subgoal_stats(subgoals)
    ctx.subgoal_stats = _sg_stats   # S4-lite：落 ctx 供 _collect_diag 进结果文件（A/B 对照）
    self.record(ctx, "subgoal", f"子目标规划完成: {len(subgoals)} 个子目标",
                subgoal_titles=[sg["title"] for sg in subgoals],
                merge_strategy=merge_strategy,
                subgoal_stats=_sg_stats)


def build_plan_summary(self, ctx, subgoals, merge_strategy) -> str:
    """生成子目标规划摘要（K11 摘要注入点掩码）。返回 subgoal_plan_summary。"""
    from .sub_goal_solver import _k11_mask_on, _mask_asserted_values
    subgoal_plan_summary = self._format_plan_summary(subgoals, merge_strategy)
    # ★ 2026-10-04（K11 注入点守卫，额外加固）：`_format_plan_summary` 会把
    #   各子目标的 **title + description 原样** 写进摘要（约 :3480-3483），
    #   而该摘要随后被逐字注入 **step / merge / crosscheck 三个提示词**
    #   （`SUBGOAL_STEP_USER_TEMPLATE` 的 {subgoal_plan_summary}、merge 同名字段、
    #   crosscheck 摘要）。⇒ 仅掩码单条 `subgoal_description` 不足以封堵——
    #   同一断言值会经本摘要二次进入同一提示词。故在**摘要生成处**（其唯一
    #   用途就是提示词注入）统一掩码一次，覆盖三个消费者。只改本地变量，
    #   不动 subgoals / merge_strategy 本体。
    if _k11_mask_on():
        subgoal_plan_summary, _n_summ = _mask_asserted_values(subgoal_plan_summary)
        if _n_summ:
            self.record(ctx, "blueprint_lock_guard",
                        f"子目标规划摘要内嵌 {_n_summ} 处断言数值 → 已屏蔽为占位符"
                        "（K11 摘要注入点守卫）")
    return subgoal_plan_summary
