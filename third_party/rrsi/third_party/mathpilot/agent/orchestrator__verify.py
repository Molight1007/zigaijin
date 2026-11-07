# -*- coding: utf-8 -*-
"""验证链共享工具（2026-10-14 五道审核链定案后仅保留被新链消费的三件）。

历史沿革
========
本模块原承载旧链三段：`4_verify`（verify_stage）/ `4.5_oracle`
（oracle_review_stage）/ `4.6_adv`（adversarial_stage）。2026-10-14 用户定案
「五道审核链一步到位直接删」，三段**整体删除**，职责去向：

  · 4_verify 的 LLM 投票        → `agent/vote_gate.py`（4.5_vote）
  · 4_verify 的 P1 硬信号重解   → 删（用户决策②：重解循环全删）
  · 4_verify 的非答案形态过滤   → 并入 `agent/answer_normalizer.py`（4.0）
  · 4.5_oracle 同源自评复核     → 删
  · 4.6_adv 单点对抗探测        → 升级为 `agent/adversarial_gate.py`（4.2）

保留的三件（均有新链消费方，逐字保留原实现）：
  · `_verify_resolve_tier_votes` —— 票数解析（vote_gate 调用；含"用户配置
    只抬不降"的既有口径）
  · `_pool_empty_fallback`       —— 池空错误点汇总 + 留痕（五道各环节调用）
  · `_verify_artifact_verdicts`  —— hook 06_verdicts 落盘（vote_gate 调用）

阶段名是稳定契约：旧阶段名（4_verify / 4.5_oracle / 4.6_adv）已保留于
`tools/stage_audit.py` 审计清单作历史记录。
"""
from __future__ import annotations


def _verify_resolve_tier_votes(self, ctx, tier):
    """档位票数解析：取「档位计划票数」与「用户配置」的较大者（后者默认不生效）。

    ★ 消费方：`agent/vote_gate.py`（4.5_vote）。票数**只抬不降**：
      档位计划票数是下限，`--voting_times` 调大才生效（调小不生效）。
    """
    tier_votes = self.config.tier_voting_times.get(tier, 1)
    # ★ 2026-10-07 修「假开关 --voting_times」（问题总账 §2.6）：
    #   主路径此前**只**读 `tier_voting_times`，并把结果**显式**传给
    #   `verifier.run(voting_times=tier_votes)`；而 `verifier.run` 仅在
    #   `voting_times is None` 时才回退 `config.verifier_voting_times`
    #   ⇒ `run_eval.py --voting_times N`（→ `verifier_voting_times`）
    #   对主路径**完全无效**（旧注释所称"可调"只对非主路径成立）。
    #   现让主路径取「档位票数」与「用户配置」的**较大者**：
    #     · 默认配置下 `verifier_voting_times`(2) ≤ `tier_voting_times[deep]`(3)
    #       ⇒ **默认行为零变化**；
    #     · 用户 CLI 调大（如 `--voting_times 5`）⇒ 真正生效。
    #   与 `verifier._vote_profile` 既有的「以下限方式合并」同口径。
    #   ⚠ 仅当该 tier 在 `tier_voting_times` 里有显式条目时才合并，避免影响
    #     无条目档位的默认 1 票；调**小**不生效（档位计划票数是下限）。
    _tier_vote_map = getattr(self.config, "tier_voting_times", None) or {}
    if tier in _tier_vote_map:
        try:
            _user_votes = int(getattr(
                self.config, "verifier_voting_times", 0) or 0)
        except (TypeError, ValueError):
            _user_votes = 0
        if _user_votes > int(tier_votes):
            self.record(ctx, "verifier",
                        f"投票票数提升：用户配置 verifier_voting_times="
                        f"{_user_votes} > 档位 {tier_votes} → 采用 {_user_votes}"
                        f"（--voting_times 对主路径生效）")
            tier_votes = _user_votes
    return tier_votes


def _pool_empty_fallback(self, ctx, stage: str) -> None:
    """池空兜底（2026-10-12 批次 4，用户 Q32/Q33）：
    任一淘汰环节把候选池清空 ⇒ 汇总该环节的**全部错误点**，供下游
    「先让模型找错点、再重新生成答案」。

    ⚠ 本函数**只做汇总 + 留痕**（不在此处直接重解）。2026-10-14 五道链定案后，
      重解循环（旧 5_revise）已删 —— 汇总的 `ctx.pool_empty_errors` 供
      emergency 直答兜底与逐题归因消费（用户决策②：全灭走 emergency 直答）。
    """
    try:
        if getattr(ctx, "candidates", None):
            return
        errs = []
        for v in (getattr(ctx, "verdicts", None) or []):
            fb = str(getattr(v, "feedback", "") or "").strip()
            if fb:
                errs.append({"stage": stage,
                             "id": getattr(v, "id", None),
                             "answer": str(getattr(v, "answer", ""))[:120],
                             "error": fb[:300]})
        for g in (getattr(ctx, "audit_gate", None) or []):
            if isinstance(g, dict) and (g.get("verdict") or "") == "reject":
                errs.append({"stage": stage, "kind": "audit_reject",
                             "error": str(g.get("reason") or "")[:300]})
        # ★ 2026-10-14：五道链各环节的结构化淘汰清单也并入汇总
        #   （自审/证伪/计算检测的淘汰原因，统一供水给下游归因）。
        for _key, _kind in (("self_audit_rejected", "self_audit_reject"),
                            ("adversarial_gate_rejected", "falsify_reject"),
                            ("math_check_rejected", "math_check_reject")):
            for _r in (getattr(ctx, _key, None) or []):
                if isinstance(_r, dict):
                    errs.append({
                        "stage": stage, "kind": _kind,
                        "id": _r.get("id"),
                        "answer": str(_r.get("answer") or "")[:120],
                        "error": str(_r.get("error_desc")
                                     or _r.get("reason")
                                     or _r.get("evidence") or "")[:300]})
        ctx.pool_empty_errors = list(
            getattr(ctx, "pool_empty_errors", None) or []) + errs
        self.record(ctx, "pool_empty",
                    "⚠ 候选池为空（%s 之后）：汇总 %d 条错误点交下游"
                    "「先找错点 → 再生成」" % (stage, len(errs)))
    except Exception as e:  # noqa: BLE001  观测逻辑绝不阻断主流程
        self.record(ctx, "pool_empty",
                    f"池空检查异常（已忽略）: {type(e).__name__}: {str(e)[:100]}")


def _verify_artifact_verdicts(self, ctx, ver_result) -> None:
    """投票后落盘 verdict（hook 06：06_verdicts）。消费方：vote_gate（4.5_vote）。"""
    from .orchestrator import _artifact_put
    _bc6 = getattr(ctx, "_best_cluster", None)
    _artifact_put(ctx, "06_verdicts", {
        "verdicts": [
            {"id": v.id, "answer": getattr(v, "answer", ""),
             "confidence": getattr(v, "confidence", None),
             "correct_votes": getattr(v, "correct_votes", None),
             "total_votes": getattr(v, "total_votes", None),
             "feedback": getattr(v, "feedback", "")}
            for v in (getattr(ctx, "verdicts", None) or [])
        ],
        "cluster": (None if _bc6 is None else {
            "answer_norm": getattr(_bc6, "answer_norm", ""),
            "size": getattr(_bc6, "size", None),
            "confidence": getattr(_bc6, "confidence", None),
            "candidate_ids": getattr(_bc6, "candidate_ids", None),
        }),
    })
