# -*- coding: utf-8 -*-
"""4.5 投票 —— 五道审核链第 5 道（2026-10-14 用户定案）。

用户口径（原话）：「第五道就是投票，大模型根据还存活的答案投票，先判断
会不会是多个答案，再投票选择答案」。

本阶段的实现（复用既有投票与多答判定内核，不新写第二个）：
  ① **多答案判断在前**（用户顺序）：规则先判（`question_type.asks_all_values`）
     → 规则命中即多答案；否则 LLM 判定（复用 `prompts/final_select.py` 的
     `MULTI_ANSWER_JUDGE` 提示词 + `final_selector__parse._parse_multiple`），
     结果写 `ctx.vote_multi_answer`（单一来源，下游终选层直接消费不再重复问）；
  ② **大模型投票**：`verifier.run` 对**还存活的候选**逐候选 LLM 验证投票
     （票数 = `tier_votes` 多票口径，008 由对变错教训：绝不降为 1），
     产出 `ctx.verdicts` / `ctx._best_cluster` / `ctx._cluster_data`
     （既有契约，6_format 的选择层直接消费）。

选择动作的落位：`final_selector` 的 ②（多答判断）/⑤（投票兜底）保留并在
Step7 收编进本链出口（③客观验证/④模型对比按用户决策删除）；本阶段先把
「投票数据 + 多答判定」生产出来并完整留痕。

观测：trace step=``vote_gate``，diag 键 ``vote_gate`` / ``vote_gate_events``。

五道链位置：4.0 规范化 → 4.1 自审 → 4.2 对抗证伪 → 4.3 Lean 代回 →
4.4 计算检测 → **4.5 投票**。
"""
from __future__ import annotations

from .switch_registry import get_bool as _sw_get_bool

_PROBLEM_MAX = 4000
_ANSWERS_MAX = 20   # 判多答时列给模型的最大候选数（与 final_selector 同量级）


def _switch_enabled() -> bool:
    """读「五道链·第 5 道投票」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_vote")
    except Exception:  # noqa: BLE001
        return True


def judge_multi_answer(self, ctx) -> "bool | None":
    """多答案判断（规则优先、LLM 兜底）。返回 True/False/None（判不出）。"""
    problem = str(getattr(ctx, "problem", "") or "")
    try:
        from .question_type import asks_all_values as _aav
        if _aav(problem):
            self.record(ctx, "vote_gate",
                        "多答案判断：题面命中「求所有/全部」规则 ⇒ multiple=True")
            return True
    except Exception:  # noqa: BLE001  规则不可用 ⇒ 继续问 LLM
        pass
    try:
        from prompts.final_select import (
            MULTI_ANSWER_JUDGE_SYSTEM as _S,
            MULTI_ANSWER_JUDGE_USER as _U)
        from .final_selector__parse import _parse_multiple as _pm
        cands = list(getattr(ctx, "candidates", None) or [])
        answers_block = "\n".join(
            "- " + str(getattr(c, "answer_norm", None)
                       or getattr(c, "answer", "") or "")[:80]
            for c in cands[:_ANSWERS_MAX])
        raw = self.llm(ctx, [
            {"role": "system", "content": _S},
            {"role": "user", "content": _U.format(
                problem=problem[:_PROBLEM_MAX], answers=answers_block)},
        ], 0.1, 8192)
        m = _pm(raw)
        self.record(ctx, "vote_gate",
                    "多答案判断：LLM ⇒ multiple=%s" % m)
        return m
    except Exception as _e:  # noqa: BLE001  判不出 ⇒ None（按单答处理）
        self.record(ctx, "vote_gate",
                    "多答案判断：LLM 失败 ⇒ None（按单答处理）: %s"
                    % str(_e)[:120])
        return None


def vote_stage(self, ctx, tier: str) -> None:
    """4.5 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例）。"""
    try:
        if not _switch_enabled():
            self.record(ctx, "vote_gate",
                        "4.5 投票：开关关闭（five_gate_vote=0）")
            return
        cands = list(getattr(ctx, "candidates", None) or [])
        if not cands:
            self.record(ctx, "vote_gate", "4.5 投票：候选池为空，跳过")
            return

        # ① 多答案判断（用户顺序：先判多答，再投票）
        multi = judge_multi_answer(self, ctx)
        ctx.vote_multi_answer = multi

        # ② 大模型投票（票数锁定 tier_votes 多票口径）
        from .orchestrator__verify import (
            _verify_resolve_tier_votes as _tvr,
            _verify_artifact_verdicts as _wav,
        )
        tier_votes = _tvr(self, ctx, tier)
        is_proof = (str(getattr(ctx, "question_type", "") or "") == "证明题"
                    or str(getattr(ctx, "domain", "") or "") in ("证明", "证明题"))
        ver_result = self.verifier.run(
            ctx, problem=ctx.problem, candidates=ctx.candidates,
            use_clustering=True,
            use_scoring=getattr(self.config, "use_scoring", True),
            is_proof=is_proof,
            use_playoff=(
                (tier == "deep" and getattr(self.config, "deep_use_playoff", True))
                or (tier != "deep" and ctx.state.playoff_enabled)),
            use_deterministic=getattr(self.config, "enable_deterministic", True),
            use_rubric=getattr(self.config, "use_rubric", False),
            use_challenge=getattr(self.config, "use_challenge", False),
            voting_times=tier_votes,
        )
        ctx.verdicts = self._verdicts_from_ver_result(ver_result, ctx.candidates)
        ctx._best_cluster = ver_result.get("best_cluster")
        ctx._cluster_data = ver_result.get("cluster_data", [])
        try:
            _wav(self, ctx, ver_result)      # hook 06_verdicts 落盘（既有契约）
        except Exception as _e_av:  # noqa: BLE001  落盘失败不阻断投票结果
            self.record(ctx, "vote_gate",
                        "verdict 落盘失败（不影响结果）: %s" % str(_e_av)[:120])

        # ③ 汇总留痕（票型表 + best 簇）
        _votes = []
        for c in (getattr(ctx, "verdicts", None) or []):
            _votes.append({
                "answer": str(getattr(c, "answer", ""))[:80],
                "correct": int(getattr(c, "correct_votes", 0) or 0),
                "total": int(getattr(c, "total_votes", 0) or 0),
            })
        _bc = getattr(ctx, "_best_cluster", None)
        ctx.vote_gate = {
            "n": len(cands),
            "tier_votes": int(tier_votes),
            "multi_answer": multi,
            "best_cluster": (None if _bc is None else {
                "answer_norm": str(getattr(_bc, "answer_norm", "") or "")[:120],
                "size": getattr(_bc, "size", None),
                "vote_correct": getattr(_bc, "vote_correct", None),
                "vote_total": getattr(_bc, "vote_total", None),
            }),
            "votes": _votes,
        }
        self.record(
            ctx, "vote_gate",
            "4.5 投票：%d 候选 × %d 票 ⇒ best=%s（%s/%s 票）；多答案=%s"
            % (len(cands), tier_votes,
               (ctx.vote_gate["best_cluster"] or {}).get("answer_norm", "-")[:60],
               (ctx.vote_gate["best_cluster"] or {}).get("vote_correct", "-"),
               (ctx.vote_gate["best_cluster"] or {}).get("vote_total", "-"),
               multi))
    except Exception as _e:  # noqa: BLE001  增益环节绝不阻断主链
        self.record(ctx, "vote_gate",
                    f"4.5 投票异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
