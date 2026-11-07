# -*- coding: utf-8 -*-
"""选择 / 装配职责簇（2026-10-08 批11：从 agent/orchestrator.py 逐字搬出）。

搬出方法：_pick_best_from_candidates / _maybe_final_select / _has_usable_candidate /
_set_final_response(@staticmethod) / _verdicts_from_ver_result / _collect_diag /
_mathlib_adoption

**纯搬迁**：方法体逐字搬入（tokenize-aware 去缩进，多行 docstring 字符串值不变），
签名原样保留（@staticmethod 在类内委托 stub 上保留装饰器）；Orchestrator 内保留
**同名一行委托** ⇒ 调用点零改。

耦合处理：
  · `TaskContext`/`Verdict`（.base）、`collect_diag`（.diag_export）、
    `_formatter_rank_key`/`_formatter_attach_support`（.formatter）、`_sw_bool`/`logger`
    ⇒ 新模块同名重建（同源）；
  · `_has_usable_candidate` 引用 orchestrator 模块级常量 `_DEGRADED_ANSWER_RE`
    （测试 `from agent.orchestrator import _DEGRADED_ANSWER_RE` 要求其留在 orchestrator）
    ⇒ 函数内**加 1 行惰性 import**（本仓既定模式，单一来源仍在 orchestrator，非复制）。
"""
from __future__ import annotations

import logging

from .base import TaskContext, Verdict
from .diag_export import collect_diag
from .formatter import (
    _rank_key as _formatter_rank_key,
    _attach_support as _formatter_attach_support,
)

from agent.switch_registry import get_bool as _sw_bool

logger = logging.getLogger("MathPilot")


def _mathlib_adoption(self, ctx: TaskContext) -> dict:
    """统计「检索到的定理」里有多少**真的进了 Lean 代码**（引用即采用）。

        ★ 2026-09-23 新增，回答用户"mathlib 的定理是不是都和本题有关、哪些帮上了忙"：
          实测 `used_theorems` 的条数恒等于 `search_hits` 恒等于 `top_k`
          ⇒ **该字段实为"检索结果"，不是"实际使用"**（字段名误导）。
          本轮实测 460 条里只有 4 条（1%）出现在 Lean 代码中。

        ⚠ 因此"提高定理选择数（top_k）"是**错的解**：`top_k_capped` 已 100% 饱和，
          调大只会拿回更多同样不被使用的条目。缺的是"**检索 → 采用 → 编译验证**"闭环。

        本方法把"采用率"变成可度量字段，使该闭环的成效可被逐轮跟踪。
        """
    try:
        names = [str(x) for x in (getattr(ctx, "used_theorems", None) or [])]
        if not names:
            return {"retrieved": 0, "adopted": 0, "adoption_rate": 0.0}
        code = []
        for t in (getattr(ctx, "trace", None) or []):
            if not isinstance(t, dict):
                continue
            if t.get("step") == "lean_gate" or t.get("step") == "subgoal_step":
                code.append(str(t.get("content") or ""))
        blob = "\n".join(code)
        adopted = sum(1 for n in names if n.split(".")[-1] and n.split(".")[-1] in blob)
        return {"retrieved": len(names), "adopted": adopted,
                "adoption_rate": round(100.0 * adopted / max(len(names), 1), 1)}
    except Exception:  # noqa: BLE001
        return {"retrieved": 0, "adopted": 0, "adoption_rate": 0.0}


def _verdicts_from_ver_result(self, ver_result: dict, candidates: list = None) -> list:
    """将验证器产出的多票结果汇总为 Verdict 数据类列表。

        每个候选可能有多张票（Verdict），此处聚合成一个汇总 Verdict：
        - confidence = 正确票 / 总票数
        - answer 取自候选（便于 Formatter 兜底直接使用）
        - feedback 取第一张有效票的反馈，score 取第一张非空票的评分
        """
    all_verdicts = ver_result.get("verdicts", [])
    result = []
    for idx, vds in enumerate(all_verdicts):
        # 2026-09-11 三态化：弃权票（LLM 调用失败 / 输出无法解析，属基础设施
        # 或格式故障）剔出分母——故障不是反证。此前 `len(vds)` 把 36 张故障票
        # 当"错票"计入（票池 ~120 → ~30% 错票）→ 置信度虚低 → 无谓 revise。
        effective = [v for v in vds if not getattr(v, "abstain", False)]
        correct_votes = sum(1 for v in effective if v.correct)
        total_votes = len(effective)
        candidate = candidates[idx] if candidates and idx < len(candidates) else None
        answer = candidate.answer if candidate else ""
        feedback = next((v.feedback for v in vds if v.feedback), "")
        score = next((v.score for v in vds if v.score is not None), None)
        result.append(Verdict(
            id=idx,
            answer=answer,
            confidence=correct_votes / total_votes if total_votes else 0.0,
            correct_votes=correct_votes,
            total_votes=total_votes,
            feedback=feedback,
            score=score,
            # ★ 2026-09-16 修复（误导性诊断字段）：`correct` 此前**从不赋值**，
            #   恒为 dataclass 默认 False ⇒ 导出的 `diag.verdicts[i].correct`
            #   永远是 False，即使 `correct_votes=2, total_votes=3`（多数票判对）。
            #   实测 003 因此显示 `{'correct_votes':2,'total_votes':3,
            #   'correct':False,'confidence':0.667}` —— 与置信度自相矛盾，
            #   读诊断的人会误判"验证器把候选全判错了"。
            #   注意：字段注释标明 `correct` 是**单票**语义（verifier 内部用），
            #   聚合对象此前没有定义其含义 ⇒ 这里补上"多数票是否判对"，与
            #   `confidence` 保持一致口径。**已核验无任何代码读聚合的 `correct`**
            #   （读取点都在单票上），故属纯诊断修复、不改行为。
            correct=bool(total_votes and correct_votes * 2 > total_votes),
        ))
    return result


def _has_usable_candidate(self, ctx: TaskContext) -> bool:
    """候选池里是否存在**可用**答案（2026-09-13 晚新增）。

        为什么不能只用 `if not ctx.candidates`：solver 原先在候选生成失败时会
        append 一个 reasoning=`[生成失败] 调用受限或模型拒绝回答` 的占位候选，
        于是池子"非空但不可用"，orchestrator 的兜底直接求解被永久绕开
        （4 题实测 010/016 的 `predicted` 就是这个占位符）。

        判据：候选必须有非空 answer 且不是拒绝/占位符；answer 为空时按 formatter
        的语义接受"reasoning 可作答案"的候选（formatter 会取 reasoning 尾部），
        避免把本可挽救的候选判死。
        """
    from .orchestrator import _DEGRADED_ANSWER_RE  # noqa: F401 （惰性；单一来源仍在 orchestrator）
    for c in (getattr(ctx, "candidates", None) or []):
        ans = (getattr(c, "answer", "") or "").strip()
        rea = (getattr(c, "reasoning", "") or "").strip()
        # ⚠ 用模块级常量：写成 `self._DEGRADED_ANSWER_RE` 不会回退到模块全局
        # （Python 的 self 查找只到实例/类/基类，不回落到模块命名空间）。
        if ans and not _DEGRADED_ANSWER_RE.search(ans):
            return True
        if (not ans) and rea and not _DEGRADED_ANSWER_RE.search(rea):
            return True
    return False


def _maybe_final_select(self, ctx: TaskContext) -> "str | None":
    """★ 2026-10-02 阶段二-2：终答五层选择（返回答案串；未接管返回 None）。

        与 `formatter._maybe_final_select` **共用** `final_selector.run_cached`
        ⇒ 两条路径同一决策、同一缓存、同一份 LLM 成本。
        开关 OFF / 无候选 / 异常 ⇒ 返回 None ⇒ 调用方**回退既有择优**。
        """
    try:
        from .final_selector import run_cached
        res = run_cached(ctx, self.llm, self.record)
        try:
            ctx._final_select_diag = dict(res.get("diag") or {})
            ctx._final_select_diag["branch"] = res.get("branch")
        except Exception:  # noqa: BLE001
            pass
        ans = str(res.get("answer") or "").strip()
        if not ans or res.get("skipped"):
            return None
        return ans
    except Exception as exc:  # noqa: BLE001  五层失败一律回退既有逻辑
        logger.debug("[orchestrator] 终答五层选择异常（回退）: %s: %s",
                     type(exc).__name__, exc)
        return None


def _pick_best_from_candidates(self, ctx: TaskContext) -> str:
    import re as _re
    # ★ 2026-10-02 阶段二-2：终答五层选择优先（开关 ON 时此处即返回，
    #   ⇒ 下方的 verdict 置信度 / `objective_majority_vote` 分支不执行）。
    _fs_ans = self._maybe_final_select(ctx)
    if _fs_ans is not None:
        return _fs_ans
    # 2026-09-13 P0 诊断埋点：记录"候选答案分布 + verdict 明细 + 命中分支"，
    # 用于确证 102 题（候选 4/6 正确却输出错答案）的根因。
    # 纯记录、不改选取行为；写入 ctx._pick_diag 供 _collect_diag 落盘。
    try:
        _pd: dict = {"cand_answers": [], "verdicts": [], "branch": ""}
        for _c in getattr(ctx, "candidates", None) or []:
            _a = getattr(_c, "answer", "") or ""
            _pd["cand_answers"].append(_a[:40])
        for _v in getattr(ctx, "verdicts", None) or []:
            _pd["verdicts"].append({
                "answer": (getattr(_v, "answer", "") or "")[:40],
                "conf": round(float(getattr(_v, "confidence", 0.0) or 0.0), 3),
                "cvotes": getattr(_v, "correct_votes", 0),
                "tvotes": getattr(_v, "total_votes", 0),
                "abstain": bool(getattr(_v, "abstain", False)),
            })
        ctx._pick_diag = _pd
    except Exception:  # noqa: BLE001
        ctx._pick_diag = {"error": "pick_diag_failed"}
    # 1) 从 verdicts 找有非拒绝答案的
    if ctx.verdicts:
        # ★ 2026-09-17（M5）：**票数优先于置信度**。`confidence = 正确票/总票`，
        # 单候选全对时 confidence=1.0 却只有 3 票；应先看"是否真的拿到正确票"。
        sorted_v = sorted(
            ctx.verdicts,
            key=lambda v: (int(getattr(v, "correct_votes", 0) or 0),
                           float(getattr(v, "confidence", 0.0) or 0.0)),
            reverse=True)
        for v in sorted_v:
            ans = getattr(v, "answer", "") or ""
            # ★ 2026-09-16 审计修复：去掉 `len(ans) > 3` 门槛（同 formatter.py:95-98
            #   的论证：单字符是合法完整答案；选择题多是 `A`/`AB`/`BCD`）。
            if ans.strip() and not _re.search(r"无法求解|无法解决|不能解决", ans):
                if isinstance(getattr(ctx, "_pick_diag", None), dict):
                    ctx._pick_diag["branch"] = "verdict_confidence"
                    ctx._pick_diag["picked"] = ans[:40]
                return ans
    # 2026-09-13 P1：选择题答案多数投票（用户确认"针对选择题尝试"）
    # 触发条件：① 环境变量 OBJECTIVE_MAJORITY_VOTE=1 ② 题面=选择题
    # ③ verdicts 路径未命中（即没人能给出高 confidence 的裁决）
    # 行为：对归一化后的候选答案做频次统计，取最高频。
    # 设计依据：self-consistency（Wang 2022），对"答案形态稳定"
    # （A–E 字母组合）的客观题有理论支撑。
    if (_sw_bool("objective_majority_vote")
            and getattr(ctx, "question_type", "") == "选择题"):
        _mv_count: dict = {}
        for _c in getattr(ctx, "candidates", None) or []:
            _a = (getattr(_c, "answer", "") or "").strip()
            # ★ 2026-09-16 审计修复：去掉 `len(_a) > 3`（否则 A/AB/BCD 类
            #   选择题答案进不了多数投票 ⇒ 该机制对目标题型失效）。
            if _a and not _re.search(
                    r"无法求解|无法解决|不能解决", _a):
                _mv_count[_a] = _mv_count.get(_a, 0) + 1
        if _mv_count:
            _top_ans, _top_n = max(
                _mv_count.items(), key=lambda kv: kv[1])
            if isinstance(getattr(ctx, "_pick_diag", None), dict):
                ctx._pick_diag["branch"] = "majority_vote_objective"
                ctx._pick_diag["picked"] = _top_ans[:40]
                ctx._pick_diag["majority_dist"] = {
                    k[:40]: v for k, v in _mv_count.items()}
                ctx._pick_diag["majority_top_n"] = _top_n
            return _top_ans
    # 2) 从 candidates 找有非拒绝答案的
    if ctx.candidates:
        # ★ 2026-10-07（问题总账 §2.6「择优逻辑口径分叉」）：原按 `len(reasoning)`
        #   降序（长推理优先），与 formatter 已统一的 `_rank_key`（票数→独立复现数
        #   →置信度→答案形态→答案更短）**同语义、两处判据不同** ⇒ 本兜底路径可能
        #   选出与主链不一致的答案。现统一复用 `_rank_key`（只读引用 formatter）。
        #   `_attach_support` 先把"独立复现数"挂到候选上（与 formatter 同口径）；
        #   失败不阻断（未挂时 `_rank_key` 兜底 `_support_n=1`，退化为票数/置信度口径）。
        try:
            _formatter_attach_support(ctx.candidates)
        except Exception:  # noqa: BLE001
            pass
        sorted_c = sorted(ctx.candidates, key=_formatter_rank_key, reverse=True)
        for c in sorted_c:
            if c.answer and c.answer.strip() and not _re.search(r"无法求解|无法解决|不能解决", c.answer):
                if isinstance(getattr(ctx, "_pick_diag", None), dict):
                    ctx._pick_diag["branch"] = "candidate_rank_key"
                    ctx._pick_diag["picked"] = c.answer[:40]
                return c.answer
    # 3) 最后防线：取最详细推理的尾部
    if ctx.candidates:
        # ★ 2026-10-07 同上：排序键也统一到 `_rank_key`（原为 len(reasoning)）。
        best = max(ctx.candidates, key=_formatter_rank_key)
        if best.reasoning and len(best.reasoning) > 50:
            if isinstance(getattr(ctx, "_pick_diag", None), dict):
                ctx._pick_diag["branch"] = "reasoning_tail"
            return best.reasoning.strip()[-500:]
    if isinstance(getattr(ctx, "_pick_diag", None), dict):
        ctx._pick_diag["branch"] = "empty"
    return ""


@staticmethod
def _set_final_response(ctx: TaskContext, ans: str, reason: str = "") -> bool:
    """统一写入 `ctx.final_response`，**先过非答案闸门**（2026-09-18 补）。

        背景：`ctx.final_response` 原有 **6 个赋值点**，其中三个**完全没有校验**：
          · `direct_answer`（零票兜底直答）
          · `_fresh.answer`（6.5 重做循环读审核反馈重生成的候选）
          · `_next.answer`（6.5 换下一个候选）
        ⇒ 6.5 重做循环在 **formatter 之后**运行，可以把一个"过程叙述"覆盖成最终答案。
        实测 official112-020/032/025 的最终答案正是这样变成
        `搜索已知结论：…` / `继续找规律，目前 type B：2, 8, 10。` / `步骤11：搜索已知结论`
        （而 `pick_diag.picked` 记录的是另一个值 ⇒ 说明确实被后写覆盖）。

        这与 09-17 已修的"三条切片路径绕过剥壳"是**同一族问题**：
        同一语义有多个写入点，只修了一部分。⇒ 收敛为**单一入口**。

        返回 True = 已写入；False = 被闸门拦下（调用方应换候选或继续重做）。
        """
    a = str(ans or "").strip()
    if not a:
        return False          # 空答案一律拒绝（不覆盖已有值）
    try:
        from .formatter import _looks_like_non_answer as _na
        if _na(a):
            return False
    except Exception:  # noqa: BLE001  判据不可用则放行（不阻断主流程）
        pass
    # ★ 2026-10-02 阶段二-2（team-lead Q4）：终答**唯一出口** —— 池外直答
    #   （零票兜底 / 6.5 重做循环）也过五层：输入仅一个答案 ⇒ 五层**直接采用、
    #   不调 LLM**（保护项：兜底路径不变慢/不变不稳）。开关 OFF ⇒ `skipped`
    #   ⇒ 原样写入（完全回退）。保持 `@staticmethod`（测试按静态签名调用）。
    try:
        from .final_selector import adopt_single
        _r = adopt_single(ctx, a, reason=str(reason or ""))
        if not _r.get("skipped"):
            _a2 = str(_r.get("answer") or "").strip()
            if _a2:
                a = _a2
    except Exception:  # noqa: BLE001  五层任何异常都不得阻断写答案
        pass
    ctx.final_response = a
    return True


def _collect_diag(self, ctx: TaskContext) -> dict:
    """逐步归因诊断（委托 `agent/diag_export.collect_diag`）。

        2026-10-07 阶段C1（拆上帝类）：实现已整体搬到 `agent/diag_export.py`
        （纯搬迁、键名零变更）。本方法保留为**一行委托** ⇒ 全部调用点
        （run_eval / 各 early-return 路径）零改动，diag 契约不变。
        """
    return collect_diag(
        ctx,
        config=self.config,
        lean_probe=getattr(self, "_lean_probe", None),
        lean_gate=self.lean_gate,
        stage_durs=self._collect_stage_durs,
    )
