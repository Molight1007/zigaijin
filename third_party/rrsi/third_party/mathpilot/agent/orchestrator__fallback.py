# -*- coding: utf-8 -*-
"""求解失败处理（2026-10-15 用户定案：删「直答兜底」，改「错误信息收集 + 报错」）。

历史：本模块原承载 `_emergency_direct_solve` / `_fallback_direct` / `_fallback`
三个兜底方法 —— 无候选 / 异常 / 超时时**直接向 LLM 要一个答案**（绕过智能体
流程）。2026-10-15 用户定案：直答本身是「抛开智能体的设计」，整条删除；改为
**收集错误信息**（异常类型 / 候选数 / 裁决数 / 子目标状态）并**抛异常**，
失败即暴露、不再用兜底答案掩盖。

保留 `_fallback` 的签名（`orchestrator.run` 的 `except` 仍调用它），但其行为
已从「返回兜底答案」改为「收集上下文 → 落盘 09_solve_error → raise」。
"""
from __future__ import annotations

from .base import TaskContext

from agent.artifact_store import put_ctx as _artifact_put


def _collect_solve_error(self, ctx: TaskContext, reason: str, exc=None) -> dict:
    """收集求解失败的上下文（供归因），并落盘 09_solve_error。返回 err dict。"""
    err = {
        "reason": reason,
        "exception": ("%s: %s" % (type(exc).__name__, exc))
        if exc is not None else None,
        "n_candidates": len(getattr(ctx, "candidates", None) or []),
        "n_verdicts": len(getattr(ctx, "verdicts", None) or []),
    }
    try:
        _subs = []
        for _s in (getattr(ctx, "subgoal_trace", None) or []):
            if isinstance(_s, dict):
                _subs.append(str(_s.get("result", "") or "")[:80])
                if len(_subs) >= 5:
                    break
        err["subgoal_results"] = _subs
    except Exception:  # noqa: BLE001
        pass
    try:
        ctx.solve_error = err
    except Exception:  # noqa: BLE001
        pass
    try:
        _artifact_put(ctx, "09_solve_error", err)
    except Exception:  # noqa: BLE001
        pass
    return err


def _raise_solve_error(self, ctx: TaskContext, reason: str, exc=None):
    """收集错误信息后抛异常（取代原「直答兜底」）。

    失败即报错：不返回占位/直答答案，让评测层看到这道题**真实失败**了。
    """
    err = self._collect_solve_error(ctx, reason, exc)
    _detail = err.get("exception") or reason
    raise RuntimeError("MathPilot 求解失败[%s]: %s" % (reason, _detail))


def _fallback(self, ctx: TaskContext, problem: str, exc: Exception):
    """异常兜底（2026-10-15 重写）：收集错误信息并抛异常，不再直答。

    `orchestrator.run` 的 `except` 调用本方法。原实现会「用已有候选最佳答案
    或紧急直答」兜底，现在改为如实报错。
    """
    self.record(ctx, "solve_error",
                "求解异常：收集错误信息并报错（不再直答兜底）")
    self._raise_solve_error(ctx, "exception", exc)
