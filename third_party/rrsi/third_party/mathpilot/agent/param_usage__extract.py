# -*- coding: utf-8 -*-
"""param_usage 的「取数工具 + 实测值(used)提取器」
================================================

2026-10-08 批 P1：从 agent/param_usage.py **逐字搬出**（模块级函数，缩进不变）。
```
  · 取数工具：_int / _trace / _gate_events / _cfg_get / _cap / _cap_by_tier /
              _cap_gate_rework
  · 实测提取器（12）：_u_leansearch_calls … _u_deep_review
```
param_usage.py 原位 re-export 同名符号 ⇒ 外部 `from agent.param_usage import <name>`
与 `_SPEC` 内 lambda 的裸引用契约逐字不变。

★ 本模块**自洽**：只依赖内建函数与 `getattr` 鸭子取数（**无任何 import**），
  **不 import** param_usage.py（无循环）；这也是原文件「不 import 任何 agent 子模块」
  设计原则的延续。

⚠ 勿按"未使用"清理这里的任何名字 —— 全部是对外契约（见 param_usage.py 的 re-export）。
"""
from __future__ import annotations


# ======================================================================
# 取数工具
# ======================================================================
def _int(v):
    """宽松取整。取不到返回 ``None``（**不返回 0** —— 0 表示"确实用了 0 次"）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _trace(ctx, step):
    """trace 中指定 step 的事件列表。"""
    return [t for t in (getattr(ctx, "trace", None) or [])
            if isinstance(t, dict) and t.get("step") == step]


def _gate_events(ctx):
    """合并 ``audit_gate`` / ``lean_gate`` 的闸门事件（两键通常只有一个非空）。"""
    out = []
    for key in ("audit_gate", "lean_gate"):
        for e in (getattr(ctx, key, None) or []):
            if isinstance(e, dict):
                out.append(e)
    return out


def _cfg_get(cfg, name, default=None):
    try:
        return getattr(cfg, name, default)
    except Exception:  # noqa: BLE001
        return default


def _cap(cfg, name, default=None):
    return _int(_cfg_get(cfg, name, default))


def _cap_by_tier(cfg, flat, table, tier):
    """档位字典优先 → 回退同名扁平值。

    与 ``sub_goal_solver`` 里 `_max_subgoals_for()` 的消费逻辑**刻意保持一致**
    （字典命中档位就用字典，否则用扁平值），否则埋点记的上限会与真正生效的
    上限不是同一个数，调参就白调了。
    """
    d = _cfg_get(cfg, table, None)
    if isinstance(d, dict) and tier and tier in d:
        v = _int(d.get(tier))
        if v is not None:
            return v
    return _cap(cfg, flat)


def _cap_gate_rework(cfg, tier):
    """6.5 审核重做的候选数上限，口径与 `orchestrator.py:2513` 消费处**逐字对齐**：

      · standard 档恒为 **2**（该档无 env 开关，orchestrator 亦硬编码 2）；
      · deep 档读开关 `deep_max_rework`（env `DEEP_MAX_REWORK`），
        **<0 = 无上限 ⇒ None**（默认 -1；比赛期曾为 3）。

    此前本项**硬编码返回 3、从不读该开关** ⇒ 设 `DEEP_MAX_REWORK=N` 时报告仍显示 3、
    「是否撞顶」判断失真（问题总账 §2.6 / 原审查技术债）。
    """
    if tier == "standard":
        return 2
    cap = _int(_cfg_get(cfg, "deep_max_rework", -1))
    if cap is None or cap < 0:
        return None
    return cap


# ======================================================================
# 实测值（used）提取器 —— 每个都返回 (值, 取值来源字符串)
# ======================================================================
def _u_leansearch_calls(ctx):
    """① LeanSearch 实际调用次数（用户点名「到底要找多少定理」）。"""
    v = _int(getattr(ctx, "_leansearch_calls", None))
    if v is not None:
        return v, "ctx._leansearch_calls"
    # 回退：trace 里真正返回过 n_hits 的条目数（= 检索真的跑通的次数）
    n = sum(1 for t in _trace(ctx, "leansearch") if "n_hits" in t)
    return (n or None), "trace[step=leansearch 且含 n_hits]"


def _u_leansearch_topk(ctx):
    """② 单次检索实际拿到几条定理（对照 `leansearch_top_k`）。"""
    hits = [_int(t.get("n_hits")) for t in _trace(ctx, "leansearch")
            if "n_hits" in t]
    hits = [h for h in hits if h is not None]
    return (max(hits) if hits else None), "max(trace[step=leansearch].n_hits)"


def _u_subgoals(ctx):
    """③ 实际设立了几个子目标（对照 `max_subgoals(_by_tier)`）。"""
    st = getattr(ctx, "subgoal_stats", None)
    if isinstance(st, dict):
        v = _int(st.get("n_subgoals"))
        if v is not None:
            return v, "ctx.subgoal_stats.n_subgoals"
    tr = getattr(ctx, "subgoal_trace", None)
    if isinstance(tr, list) and tr:
        return len(tr), "len(ctx.subgoal_trace)"
    return None, ""


def _u_gate_tried(ctx):
    """④ 6.5 审核闸门实际试过几个候选（＝「检测打回」的次数口径）。

    `_gate_tried` 只记录"被换上重试过的候选 id"，**不含首个候选**，
    故它是打回次数的**下界**（首个候选被拒不计入）。这一点写进 note，
    避免报告把它当成精确值。
    """
    g = getattr(ctx, "_gate_tried", None)
    if isinstance(g, list) and g:
        return len(g), "len(ctx._gate_tried)（下界：不含首个候选）"
    n = sum(1 for e in _gate_events(ctx) if e.get("step") == "final_gate")
    return (n or None), "count(gate events step=final_gate)"


def _u_unknown_streak(ctx):
    """⑤ 闸门连续判 unknown 的最长连击（对照 `lean_gate_unknown_stop`）。

    该上限的语义是「**连续** N 个 unknown 后止损、剩余候选不再逐个整题 verify」
    ⇒ 真正的"撞顶"信号是**连续**长度，不是 unknown 总数。故这里算最长连击。
    """
    seq = []
    for e in _gate_events(ctx):
        if e.get("step") not in ("final_gate", "candidate_audit"):
            continue
        v = str(e.get("verdict", "") or "")
        if v:
            seq.append(v)
    if not seq:
        return None, ""
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v == "unknown" else 0
        best = max(best, cur)
    return best, "max 连续 unknown in gate events"


def _u_revise_round(ctx):
    """⑥ 无条件重做（自纠错）实际跑了几轮（对照 `max_revise_rounds`）。

    `ctx.revise_round` 全仓只有 `orchestrator._deep_revise_loop` 一处自增
    （已 grep 确认）⇒ 它就是"深档 revise 轮次"的权威计数。
    """
    return _int(getattr(ctx, "revise_round", None)), "ctx.revise_round"


def _u_preverify_rounds(ctx):
    """⑦ 前置形式化验证实际修正了几轮（对照 `preverify_max_rounds`）。"""
    pt = getattr(ctx, "preverify_trace", None)
    if isinstance(pt, dict):
        for k in ("rounds", "n_rounds", "attempts", "revisions"):
            v = _int(pt.get(k))
            if v is not None:
                return v, "ctx.preverify_trace.%s" % k
    n = len(_trace(ctx, "preverify"))
    return (n or None), "count(trace step=preverify)"


def _u_total_calls(ctx):
    """⑨ 本题 LLM 调用总次数（对照 `max_total_calls`；该上限**只记账不阻断**）。"""
    b = getattr(ctx, "budget", None)
    v = _int(getattr(b, "used_calls", None))
    return v, "ctx.budget.used_calls"


def _u_candidates(ctx):
    """⑩ 实际产出的候选数（对照 `policy_sample_times` / `tier_sample_times`）。"""
    cs = getattr(ctx, "candidates", None)
    if isinstance(cs, list) and cs:
        return len(cs), "len(ctx.candidates)"
    return _int(getattr(ctx, "n_candidates", None)), "ctx.n_candidates"


def _u_verdicts(ctx):
    """⑪ 实际投票次数（对照 `verifier_voting_times` / `tier_voting_times`）。"""
    vs = getattr(ctx, "verdicts", None)
    if isinstance(vs, list) and vs:
        return len(vs), "len(ctx.verdicts)"
    return None, ""


def _u_deep_review(ctx):
    """⑫ 带推理深复核实际跑了几次（对照 `verifier_deep_final` 链路）。"""
    n = len(_trace(ctx, "deep_review"))
    return (n or None), "count(trace step=deep_review)"
