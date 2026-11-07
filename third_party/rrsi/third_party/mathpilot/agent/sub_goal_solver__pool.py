# -*- coding: utf-8 -*-
"""候选入池门槛（池封顶 + v2 不合格拦截）（2026-10-08 批1第三增量：从
agent/sub_goal_solver.py 的 SubGoalSolverAgent.run 抽出）。

职责：① 按 `candidate_pool_limit` 封顶（池满 ⇒ 跳过该候选来源）；② v2 入池门槛
（`_pool_gate_v2_on`）——子目标候选不含最终答案 ⇒ 不入池并留痕。
返回 (final_answer, skip)：skip=True 表示**跳过入池**（对应原段两处裸 `return`）。

⚠ **哨兵式搬迁（非纯逐字搬）**：段体逐字搬入，**仅**把 2 处裸 `return`（返回
None）改写为 `return final_answer, True`，并新增末尾 `return final_answer, False`；
调用侧 `if _skip_pool: return` 透传（与原裸 return 等价，返回 None）。
与 `agent/orchestrator__solve.py` 的 3_solve 段同口径。

`_pool_gate_v2_on` 走函数内惰性 import（单一来源 = sub_goal_solver）。
参数 `self` 即 `SubGoalSolverAgent` 实例。
"""
from __future__ import annotations


def apply_pool_gate(self, ctx, final_answer) -> tuple:
    """池封顶 + v2 入池门槛。返回 (final_answer, skip)。"""
    from .sub_goal_solver import _pool_gate_v2_on
    # 2026-09-02 老师需求：候选池统一封顶（与 solver.run 同口径）。
    # 2026-09-04：cap 8→6（deep 候选 4→3 配套，验证成本 -25%）
    # ★ 2026-10-04 修（D7）：原硬编码 `>= 6` 与 `agent/orchestrator.py:1962-1969`
    #   自称「候选池封顶已按研究期标尺删除、不再截断候选」**直接矛盾**，
    #   且后果不只是少一个候选 —— 池满时**整条子目标候选来源被跳过**
    #   （`return` 直接出函数）。现统一读 `candidate_pool_limit`（0 = 不限）。
    _pool_lim = 0
    try:
        _pv = float(getattr(self.config, "candidate_pool_limit", 0.0) or 0.0)
        _pool_lim = int(_pv) if _pv > 0 else 0
    except (TypeError, ValueError):
        _pool_lim = 0
    if _pool_lim > 0 and len(getattr(ctx, 'candidates', None) or []) >= _pool_lim:
        self.record(ctx, "subgoal",
                    f"候选池已达上限 {_pool_lim}，跳过子目标候选入池")
        return final_answer, True
    # ★★ 2026-10-07（2.4 v2 入口穷举 · 入口⑤）：**子目标候选此前不过入池门槛**
    #   —— 它是独立于 solver 的一条 append（"只改主路径 = 没改"）。实测背景：
    #   题 002 的 `'证明这个最小值就是所求的 $T$，并验证其最优性'`、
    #   题 000 的 `'子问题4：计算满足条件的路径总数'` 这类"任务描述句"曾是
    #   池内候选（且 002 那条还成了唯一幸存候选被采用）。此处与 solver 同口径，
    #   不合格则**不入池并留痕**（`_merge_results` 的 `final_answer` 与
    #   `_fallback_from_last_subgoal` 的自然语言结论都可能命中）。
    if _pool_gate_v2_on():
        try:
            from utils.extract import normalize_pool_answer as _npa
            _fans, _fok = _npa(final_answer)
        except Exception:  # noqa: BLE001  校验不可用时放行，不阻断
            _fans, _fok = final_answer, True
        if not _fok:
            self.record(
                ctx, "candidate_invalid",
                "子目标候选 **不含最终答案 → 不入池**"
                "（final_answer 前 60 字：%r）" % (str(final_answer)[:60],))
            return final_answer, True
        final_answer = _fans
    return final_answer, False
