# -*- coding: utf-8 -*-
"""子目标结论注入（2026-10-08 拆分：从 agent/solver.py 抽出）。

职责：把 2.7_subgoal_main / 3_solve(P&E) 阶段写入 ``ctx.subgoal_trace`` 的
子目标结论，渲染成文本块并注入三条主求解生成路径的提示词
（``_generate_initial`` / ``_generate_proof`` / ``_generate_revise`` /
``_improve_candidates_once``）。

搬迁方式：**纯逐字搬** —— 两个函数的函数体与签名逐字搬入（仅去结构性缩进；
多行 docstring 续行保留原缩进），``agent/solver.py`` 原处改为一行委托。
类属性 ``_SUBGOAL_FINDING_ITEM_CHARS`` / ``_SUBGOAL_FINDINGS_MAX_CHARS``
仍留在 ``SolverAgent``（被 ``self.`` 访问，保持函数体逐字不变）。

外部符号依赖（原在 solver.py 模块级）：``_sw_bool`` —— 本模块顶部同口径导入。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用，避免运行期导入
    from .base import TaskContext

from agent.switch_registry import get_bool as _sw_bool

logger = logging.getLogger("MathPilot")


def _render_subgoal_findings(self, ctx: TaskContext) -> str:
    """把已求得的子目标结论渲染成文本块，供**主求解**在子目标基础上整合答案。

        数据源：`ctx.subgoal_trace`（SubGoalSolver 在 2.7 / 3_solve(P&E) 阶段写入；
        每项为 dict，含 id/title/description/type/expected_output/result）。

        空安全：无子目标 / 全部 result 为空 → 返回 ""（退回原主求解，零噪音）。
        开关：`enable_subgoal_findings`（已登记 switch_registry，默认 True）为
        False → 返回 ""。
        """
    if not _sw_bool("enable_subgoal_findings"):
        return ""
    trace = getattr(ctx, "subgoal_trace", None) or []
    if not isinstance(trace, list) or not trace:
        return ""
    items: list[str] = []
    for sg in trace:
        if not isinstance(sg, dict):
            continue
        _res = str(sg.get("result", "") or "").strip()
        if not _res:
            continue  # 空结论不注入（不占位、不制造"看起来有数据"的假象）
        _sid = sg.get("id", "")
        _title = str(sg.get("title", "") or "").strip()
        _exp = str(sg.get("expected_output", "") or "").strip()
        if len(_res) > self._SUBGOAL_FINDING_ITEM_CHARS:
            _res = _res[: self._SUBGOAL_FINDING_ITEM_CHARS].rstrip() + "…（截断）"
        _head = f"- 子目标 {_sid}「{_title}」"
        if _exp:
            _head += f"（应产出：{_exp}）"
        # ★★ 2026-10-06（子目标交付重写 C）：**无推导支撑的结论必须打标**。
        #   实测题 official112-007：7 个子目标**直接给结论、无任何推导**，
        #   主求解把其中被臆造的结论（f(3/71) 真值 0、子目标写 1）原样采信
        #   ⇒ 最终答错。此前子目标结论**不带"有没有推导"的信息**，主求解
        #   无从区分"算出来的"与"断言出来的"。现按 trace 的
        #   `has_derivation`（结论契约解析：结论行之后是否有 ≥20 字验算）
        #   与客观判定结论打标：未判定 + 无推导 = 「仅断言」。
        _verdict = str(sg.get("objective_verdict", "") or "")
        _has_der = bool(sg.get("has_derivation", False))
        _warn = ""
        if _verdict == "incorrect":
            _warn = ("\n  ⚠⚠【客观判定：**不成立**】"
                     + str(sg.get("objective_reason", "") or ""))
        elif not _has_der:
            _warn = ("\n  ⚠【**仅断言、无推导支撑** —— 本步未给出验算/论证，"
                     "不可无条件采信；请**独立核对**后再使用】")
        elif _verdict == "correct":
            _warn = "\n  ✔【客观判定：成立】"
        items.append(_head + f"\n  结论：{_res}" + _warn)
    if not items:
        return ""
    body = "\n".join(items)
    if len(body) > self._SUBGOAL_FINDINGS_MAX_CHARS:
        body = (body[: self._SUBGOAL_FINDINGS_MAX_CHARS].rstrip()
                + "\n…（其余子目标结论因过长省略）")
    # ★ 3.3：主求解指令 —— 在已确立的子目标结论基础上整合最终答案，且允许纠错。
    return (
        "【已确立的子目标结论】\n"
        "（以下为前面逐步求得的**关键中间数据 / 条件**，是本题求解的基础）\n"
        + body
        + "\n\n★ 请在**上面已确立的子目标结论**基础上，整合出本题的最终答案：\n"
        # ★ 2026-10-04 修（引导方向）：原文写「这些结论就是要用的数据，**直接使用、
        #   不要从头再推一遍**」—— 那是**无条件信任**上游结论，一旦某条子目标被
        #   蓝图层提前锁错值（K11，41.4% 唯一过半根因），主求解会把错值当事实沿用
        #   ⇒ 等于**放大错误**。用户口径是"检错的要能查出错误、推理要被引导向正确"，
        #   故改为**先核对再使用**：核对（量纲/取值范围/关键等式是否自洽）不是
        #   从头重推，仍遵守"主求解建立在子目标结论之上"的设计。
        "  1) 这些结论是**待核验的中间数据**——先做一次**快速自洽核对**"
        "（量纲 / 取值范围 / 关键等式 / 与题目已知条件是否矛盾），"
        "**确认无误后再使用**；核对不等于从头重推，仍应在此结论之上推进；\n"
        "  2) 若核对发现**某条子目标结论有误**（算错 / 前提误 /"
        " 与题意矛盾 / 明显不自洽），必须**指出并纠正**它，再据修正后的结论"
        "给出最终答案 —— **不要被错误的子目标结论绑死**；\n"
        "  3) 最终答案必须**建立在这些已确立结论之上**（一致时直接整合；"
        "不一致时说明以哪个为准及理由）。\n"
        # ★★ 2026-10-06（子目标交付重写 C）：把"仅断言"降级为待核项。
        "  4) 标注「⚠ 仅断言、无推导支撑」的子目标结论**没有给出验算**，"
        "可能是臆造：**不得**直接作为最终答案的依据 —— 必须自己独立复核"
        "（代入 / 用另一条路径重算 / 检查是否与题意矛盾），确认后才可使用；"
        "标注「⚠⚠ 客观判定：不成立」的结论**已被客观手段判错**，必须纠正。\n"
    )

def _apply_subgoal_findings(self, ctx: TaskContext, text: str) -> str:
    """把子目标结论块追加到 `text` 末尾，并落 trace / metadata（三路径共用）。

        三条主求解生成路径（初始 `_generate_initial` / 证明 `_generate_proof` /
        重解 `_generate_revise`）都调用本方法，**复用同一渲染与埋点逻辑**，
        避免"只改一处路径=没改"（本项目已多次踩到）。

        空块（无子目标结论 / 开关关）→ **原样返回 `text`**（零改动、零噪音）。
        """
    block = self._render_subgoal_findings(ctx)
    if not block:
        return text
    _n = sum(
        1 for s in (getattr(ctx, "subgoal_trace", None) or [])
        if isinstance(s, dict) and str(s.get("result", "") or "").strip())
    self.record(ctx, "subgoal_findings",
                "求解路径注入子目标结论 %d 字符（%d 条）"
                % (len(block), _n))
    # 3.4：diag 埋点（由 orchestrator._collect_diag 导出为
    # `subgoal_findings_injected: {count, chars}`）。
    try:
        if isinstance(getattr(ctx, "metadata", None), dict):
            ctx.metadata["subgoal_findings_injected"] = {
                "count": _n, "chars": len(block)}
    except Exception:  # noqa: BLE001
        pass
    return text + "\n\n" + block
