"""终答选择封装与多答案并集（2026-10-08 从 agent/formatter.py 逐字搬出）。

簇⑥⑦：`_maybe_union_enumerated`（多答案题并集）· `_pick_best`（可信度门槛封装，
转调 `_pick_best_raw`）· `_maybe_final_select`（五层终答选择封装）。formatter.py 原位
保留**同名一行委托** ⇒ `FormatterAgent._*` 契约不变（`_maybe_union_enumerated` 被
`tests/test_p1b_to_p1f_fixes_0917.py` 按**实例法**直接调用，委托保证其可用）。

依赖（顶层）：`_looks_like_non_answer`(formatter__nonanswer) ·
`_passes_confidence_floor`(formatter__rank) · `logger`（同名 MathPilot 单例）。
⚠ 本模块**不** import formatter 主模块（无循环）；`final_selector` / `question_type`
仍为**函数内惰性 import**（与搬出前逐字一致）。
"""
from __future__ import annotations

import logging

from .base import TaskContext
from .formatter__nonanswer import _looks_like_non_answer
from .formatter__rank import _passes_confidence_floor

logger = logging.getLogger("MathPilot")


def _maybe_union_enumerated(self, ctx: TaskContext, current: str):
    """2026-09-17（P1-D）：多答案题的**候选并集**。

        用户要求（原话）：「如果大模型能确定多个正确的答案，那就把所有答案都输出。
        但要保证不是什么答案都输出，要保证输出的答案的正确率。」

        硬约束（**宁缺勿滥**，任一不满足即放弃并集、保持原逻辑）：
          ① 仅在题面明确要求「所有 / 全部」时启用（高精度，覆盖"求所有解"型）；
          ② 只并入**无反对票**的簇（`vote_total > 0` 且 `vote_correct == vote_total`）；
          ③ 每簇只取 1 个代表；并入项数上限 **3**（实测 gold 最多 3 项）；
          ④ **加法而非替换** —— 当前答案必须已属某个"全票簇"，否则放弃；
          ⑤ 逐项过 `_looks_like_non_answer` 闸门，并与已有项去重；
          ⑥ 用 **ASCII 逗号 + 空格**连接（对齐 `run_eval._split_multi` **只按 `,` 拆**
             的口径；**不可用顿号**，否则判分器拆不出多项）；
          ⑦ 结果项数 < 2 ⇒ 放弃。

        返回 `None` = 不改（保持原逻辑）。
        """
    try:
        from agent.question_type import asks_all_values as _aav
        if not _aav(getattr(ctx, "problem", "") or ""):
            return None
        _cd = getattr(ctx, "_cluster_data", None)
        if not isinstance(_cd, list) or len(_cd) < 2:
            return None
        _cur = (current or "").strip()
        if not _cur:
            return None

        def _vt(_cl):
            return (int(getattr(_cl, "vote_total", 0) or 0),
                    int(getattr(_cl, "vote_correct", 0) or 0))

        # ④ 当前答案必须已在某个全票簇内，否则放弃（防"什么都输出"）
        _cur_ok = False
        for _cl in _cd:
            _an = str(getattr(_cl, "answer_norm", "") or "").strip()
            _tv, _cv = _vt(_cl)
            if _an and _tv > 0 and _cv == _tv and (_an == _cur or _cur in _an):
                _cur_ok = True
                break
        if not _cur_ok:
            return None

        _items, _seen = [], set()
        for _cl in _cd:
            _an = str(getattr(_cl, "answer_norm", "") or "").strip()
            _tv, _cv = _vt(_cl)
            if not _an or _tv <= 0 or _cv != _tv:      # ②
                continue
            if _looks_like_non_answer(_an):            # ⑤
                continue
            _k = _an.lower()
            if _k in _seen:
                continue
            _seen.add(_k)
            _items.append(_an)
            if len(_items) >= 3:                       # ③
                break
        if len(_items) < 2:                            # ⑦
            return None
        return ", ".join(_items)                       # ⑥
    except Exception:  # noqa: BLE001
        return None


def _pick_best(self, ctx: TaskContext):
    """择优入口（★ 2026-09-30 截图 #9）：在原始择优之上加**可信度门槛**。

        用户口径：「投票要选出里面最像答案的一个，候选数量与逻辑通顺度都要是指标，
                  同时要**可信度要求**（太低＝全军覆没）」。

        本方法只做一件事：调用 `_pick_best_raw`（= 改动前的全部择优逻辑，逐字未动），
        然后检查被选中的候选**是否达到可信度底线**：
          · 达到 → 原样返回（**绝大多数情形走这里，行为与改动前完全一致**）；
          · 未达到 → 仍然返回它（不能把答案变成空，那是"全军覆没"本身），
            但**在 trace 里显式记一条 `low_confidence_pick`**，并把
            `ctx._pick_diag["below_confidence_floor"] = True` 落盘。

        为什么不"直接拒绝低可信度答案"：
          本项目的地基是"**必须有答案**"（评测判空 = 0 分）。若在此处拒绝，
          等于把单题从"可能答错"改成"必定 0 分"，与用户意图相反。
          「可信度要求」的正确落地是**让优化有据可依**——把"低于底线的题"标出来，
          供后续环节（5.5 强制复核 / 6.5 重做 / 数据闭环归因）定向处理。
          ⚠ `pick_confidence_floor` 默认 **0.0** ⇒ 本门槛默认**不产生任何新记录**，
            亦不改变任何行为；A/B 实验时再显式开启。
        """
    cand = self._pick_best_raw(ctx)
    try:
        floor = float(getattr(self.config, "pick_confidence_floor", 0.0) or 0.0)
    except (TypeError, ValueError):
        floor = 0.0
    if floor <= 0.0 or cand is None:
        return cand
    try:
        _ok, _why = _passes_confidence_floor(cand, floor)
        if not _ok:
            _ans = (getattr(cand, "answer", "") or "")[:60]
            if isinstance(getattr(ctx, "_pick_diag", None), dict):
                ctx._pick_diag["below_confidence_floor"] = True
                ctx._pick_diag["confidence_floor"] = floor
                ctx._pick_diag["picked_confidence"] = round(
                    float(getattr(cand, "confidence", 0.0) or 0.0), 4)
            self.record(
                ctx, "finalize",
                "★ 所选候选低于可信度门槛 %.2f：%s（conf=%.3f, 正确票 %s/%s）"
                "—— 仍提交（拒绝会使本题必得 0 分），但已标记待后续环节定向复核"
                % (floor, _ans,
                   float(getattr(cand, "confidence", 0.0) or 0.0),
                   getattr(cand, "correct_votes", 0),
                   getattr(cand, "total_votes", 0)))
        else:
            if isinstance(getattr(ctx, "_pick_diag", None), dict):
                ctx._pick_diag["below_confidence_floor"] = False
                ctx._pick_diag["confidence_gate_reason"] = _why
    except Exception:  # noqa: BLE001
        pass
    return cand


def _maybe_final_select(self, ctx: TaskContext):
    """★ 2026-10-02 阶段二-2：终答五层选择（开关 ON 时接管；未接管返回 None）。

        用户口径：「投票应在**最后**，不要一开始就投票」。开关
        `enable_final_answer_selection`（默认 True）ON 时，终答由
        `agent/final_selector.select_final_answer` 统一选出（②多答案判断→
        ③客观验证→④模型对比→⑤投票兜底）⇒ **不再走** `_rank_key` 票数优先 /
        `objective_majority_vote`。

        `skipped=True` / 无答案 ⇒ 返回 None ⇒ 调用方**回退既有择优**（不阻断主链）。
        结果经 `run_cached` 缓存 ⇒ 与 `orchestrator._pick_best_from_candidates`
        **共用同一决策**（同一 ctx 不会选出两个终答）。
        """
    try:
        from .final_selector import SynthCandidate, run_cached
        res = run_cached(ctx, self.llm, self.record)
        try:
            ctx._final_select_diag = dict(res.get("diag") or {})
            ctx._final_select_diag["branch"] = res.get("branch")
        except Exception:  # noqa: BLE001
            pass
        ans = str(res.get("answer") or "").strip()
        if not ans or res.get("skipped"):
            return None
        cand = res.get("cand")
        if cand is None:
            for c in (getattr(ctx, "candidates", None) or []):
                if (getattr(c, "answer", "") or "").strip() == ans:
                    cand = c
                    break
        if cand is None:
            cand = SynthCandidate(ans, "（终答五层选择产出）")
        # ★ 2026-10-02 阶段三（team-lead 复核）：此处原先只在 `_pick_diag`
        #   **已是 dict** 时才写 ⇒ 主路径上 `_pick_diag` 尚未创建、写入静默丢失，
        #   随后被 :522 兜底改标成 `formatter_pick_best` ⇒ **A/B 两 arm 看起来
        #   走同一条路**、五层分支不可归因。改为「不是 dict 就先创建」。
        try:
            if not isinstance(getattr(ctx, "_pick_diag", None), dict):
                ctx._pick_diag = {}
            ctx._pick_diag["branch"] = "final_select_" + str(res.get("branch"))
            ctx._pick_diag["picked"] = ans[:40]
        except Exception:  # noqa: BLE001
            pass
        return cand
    except Exception as exc:  # noqa: BLE001  五层失败一律回退既有逻辑
        logger.debug("[formatter] 终答五层选择异常（回退）: %s: %s",
                     type(exc).__name__, exc)
        return None
