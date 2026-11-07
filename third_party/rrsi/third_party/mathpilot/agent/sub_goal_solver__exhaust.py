# -*- coding: utf-8 -*-
"""穷尽性搜索机制：题面问「所有/全部」时，代码强制追加『解族穷尽性检查』子目标
（2026-10-08 批1：从 agent/sub_goal_solver.py 的 SubGoalSolverAgent.run 抽出）。

职责：在规划**之后**追加一个「解族穷尽性检查」子目标（绕过 max_subgoals 截断、
覆盖全部规划路径），并把「穷尽性要求」并入 merge_strategy；同时落
`ctx._exhaust_diag` 埋点。整段包裹 try/except —— 追加失败绝不阻断主流程。

**纯搬迁**：段内源码逐字搬入（仅去 4 空格缩进），调用点由 run 改为一行委托。
`subgoals` 在段内原地 append（无需返回）；run 局部名 `merge_strategy` 在段内
被改写 ⇒ 由本函数**返回**。段外自由名实测为空。`logger` 取模块级同名
"MathPilot" logger。
"""
from __future__ import annotations

import logging

logger = logging.getLogger("MathPilot")


def append_exhaust_subgoal(self, ctx, subgoals, merge_strategy) -> str:
    """追加「解族穷尽性检查」子目标（原地改 `subgoals`），返回新 merge_strategy。

    `self` = SubGoalSolverAgent 实例；`ctx` = TaskContext；
    `subgoals` = 规划出的子目标列表（原地 append）；`merge_strategy` = 合并策略文本。
    """
    # ---- 2026-09-14 穷尽性搜索机制（B0 遗留课题）----
    # 实测 003（漏 `2030`）与 074（漏取整解族）的失败**不是形态问题**：
    # 模型自信地认为"只有一个解"，因此"必须枚举"的形态要求对它无效。
    # 现由代码**强制追加**一个「解族穷尽性检查」子目标，逼迫模型按解族
    # 分类穷举（常数/线性/周期/取整/分段…）并回答"是否还有其他解族"。
    # 追加发生在规划**之后** ⇒ 不会被 max_subgoals 截断，且覆盖全部规划路径
    # （选择题 / Blueprint DAG / LLM 规划）。
    try:
        from .question_type import asks_all_values as _aav
        if _aav(ctx.problem or "") and subgoals:
            _max_id = max(int(sg.get("id", 0) or 0) for sg in subgoals)
            subgoals.append({
                "id": _max_id + 1,
                "title": "解族穷尽性检查",
                "description": (
                    "本题问『所有 / 全部』取值。\n\n"
                    "⚠ **前序子目标的结论尚未经过完备性检验**，它们只是"
                    "**待检验的假设**，不是已确立的事实。你的任务是找出它们"
                    "**遗漏**了什么，**不是**确认它们正确。\n\n"
                    "必须按下列**固定格式**作答（缺任一段视为未完成检查）：\n\n"
                    "解族清单: 列出本题所有可能的解族类型"
                    "（如：常数解 / 线性解 / 多项式解 / 周期解 / 取整型解（⌈x⌉、⌊x⌋）/ "
                    "分段定义解 / 特殊函数解 / 指数对数型解 …）—— **尽量列全**；\n"
                    "逐族判定: 对上一步列出的**每一族**逐一给出『存在 / 不存在』"
                    "及其依据（**排除某类必须给出反证或构造性论证**，"
                    "不得只写“不可能”“显然无解”）；\n"
                    "前序结论复核: 对前序子目标已找到的解，明确写出"
                    "『该解属于上述哪一族』『是否还有其他解与它同族』；\n"
                    "结论: EXHAUSTIVE: yes 或 EXHAUSTIVE: no\n"
                    "遗漏解族: 若上一步为 no，写出遗漏解族的具体形式（含参数）；"
                    "若为 yes，写『无』。\n\n"
                    "⚠ 宁可多列一类再排除，也不得因“看起来不可能”而跳过；"
                    "给出结论时必须注明依据。"
                ),
                "type": "verify",
                # ⚠ **必须依赖全部已有子目标**：执行期 `_format_dep_results`
                # 只注入**直接依赖**的结果 —— 若只依赖最后一个，穷尽性检查就
                # 看不到前面找到的任何解，而它的任务恰恰是"检查是否遗漏解族"。
                # ★ 2026-09-17 改进：注入**保留**（否则它无从对照），但注入的
                #   语义已改为"待检验假设"（见上方 description 第一段）——
                #   实测 003 原实现下，该子目标只复述了前序结论 `a_n = n`，
                #   既没列解族也没做排除，属**被前序结论锚定**。
                #   同时要求**固定格式输出**，使"没做检查"从不可检测变为可检测
                #   （`_collect_diag` 会解析出 `exhaust_result`）。
                "depends_on": [sg["id"] for sg in subgoals],
                "expected_output": (
                    "必须包含四段：『解族清单:』『逐族判定:』『前序结论复核:』"
                    "『结论: EXHAUSTIVE: yes|no』；若为 no 还需『遗漏解族:』。"
                    "缺任一段视为未完成检查。"),
                "result": "",
            })
            # merge 必须把「解族穷尽性检查」的结论纳入最终答案（否则子目标白跑）
            _exh_note = (
                "\n【穷尽性要求（必须遵守）】最终答案**必须综合『解族穷尽性检查』"
                "子目标的结论**：把检查中确认存在的**所有解族**全部列出"
                "（含该检查新发现的遗漏解族的具体形式），"
                "**不得**只采用前面子目标找到的部分解；"
                "若检查结论与前面矛盾，以穷尽性检查为准并说明理由。"
            )
            merge_strategy = (merge_strategy or "直接给出最终答案") + _exh_note
            self.record(ctx, "subgoal",
                        "穷尽性搜索：追加『解族穷尽性检查』子目标"
                        "（题面要求『所有』，共 {} 个子目标）".format(len(subgoals)))
            # 埋点落 diag（2026-09-14）：trace 不进结果文件 ⇒ 此前无法确认
            # 机制是否真的生效，只能靠猜。写入 ctx 供 _collect_diag 落盘。
            try:
                ctx._exhaust_diag = {
                    "appended": True,
                    "n_before": len(subgoals) - 1,
                    "n_after": len(subgoals),
                    "trigger": "asks_all_values",
                }
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                ctx._exhaust_diag = {
                    "appended": False,
                    "reason": ("非『求所有』题" if not _aav(ctx.problem or "")
                               else "无子目标可追加"),
                }
            except Exception:  # noqa: BLE001
                pass
    except Exception as _e:  # noqa: BLE001
        logger.debug("[穷尽性] 子目标追加失败: %s", _e)
        try:
            ctx._exhaust_diag = {"appended": False,
                                 "reason": "异常: {}".format(type(_e).__name__)}
        except Exception:  # noqa: BLE001
            pass
    return merge_strategy
