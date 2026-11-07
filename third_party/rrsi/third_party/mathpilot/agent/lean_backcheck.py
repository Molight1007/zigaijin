# -*- coding: utf-8 -*-
"""4.3 Lean 逻辑判断（答案代回题目）—— 五道审核链第 3 道（2026-10-14 用户定案）。

用户口径（原话）：「第三道用 lean 进行逻辑判断，就是让答案带回题目，
看看逻辑对不对」。

实现口径（加模块优先，复用既有 Lean 通道）：
  · 后端 = `tools/lean_local/lean_gate.LeanGate.apply`（候选级 Lean 硬验证：
    把「题目 + 候选答案/推理」翻译为 Lean 命题并编译 —— 这正是「答案带回
    题目」的形式化判定），**不新写第二个 Lean 校验实现**；
  · 本模块只做：开关/适用性守卫 → 调用 → 淘汰落池 → 反馈接线 → 留痕。

判定纪律（沿用既有 Lean 门的实证口径，防误杀）：
  · 部分淘汰 ⇒ `ctx.candidates = kept`；
  · **全拒** ⇒ 记 `_gate_all_rejected` 显式留痕、**不静默清池**（Lean 在正确题
    上误报率实测 56%，仅凭 Lean 全拒清池会误杀整池；`strict_all_rejected=True`
    时才清空交给下游兜底）—— 与 3.6 既有处置逐条一致；
  · 环境不可用 / 不适用 / 时间硬顶 ⇒ 降级放行（LeanGate 内部自处理并留痕）。

观测：trace step=``lean_backcheck``，diag 键 ``lean_backcheck`` /
``lean_backcheck_events``；`ctx.lean_gate`（LeanGate.apply 既有契约）不变。

五道链位置：4.0 规范化 → 4.1 自审 → 4.2 对抗证伪 → **4.3 Lean 代回** →
④计算检测 → ⑤投票。
"""
from __future__ import annotations

from .switch_registry import get_bool as _sw_get_bool

def _switch_enabled() -> bool:
    """读「五道链·第 3 道 Lean 代回」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_lean_check")
    except Exception:  # noqa: BLE001
        return True


def _tool_channel_enabled() -> bool:
    """读「4.3 工具①观测通道」开关（默认 False，云端 A/B 时开）。

    ★ 2026-10-15（工具化定稿）：开启后对**存活候选**调工具①
    `tools/agent_toolkit.lean_check`（LLM 翻译 → Lean 检查 → 错误点/缺少点），
    结果写 diag（``ctx.lean_tool_check``）与 trace —— **观测模式：暂不改淘汰**，
    等云端验证通过后再切换为判定通道（届时把 verdict 并入淘汰条件）。
    """
    try:
        return _sw_get_bool("five_gate_lean_check_tool")
    except Exception:  # noqa: BLE001
        return False


def lean_backcheck_stage(self, ctx, tier: str) -> None:
    """4.3 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例）。

    注入点：``getattr(self, "lean_gate", None) or LeanGate(...)``（与旧链
    3.6/4.7 同款，便于单测注入桩）。
    """
    try:
        if not _switch_enabled():
            self.record(ctx, "lean_backcheck",
                        "4.3 Lean 代回：开关关闭（five_gate_lean_check=0）")
            return
        cands = list(getattr(ctx, "candidates", None) or [])
        if not cands:
            self.record(ctx, "lean_backcheck", "4.3 Lean 代回：候选池为空，跳过")
            return

        # 适用性守卫（与 3.6 同口径）：选项字母/判断值/概念文字答案结构上
        # 无从形式化 ⇒ 豁免 Lean（不豁免其余四道）
        _active = getattr(self, "_lean_active", None)
        _applicable = getattr(self, "_lean_applicable", None)
        if callable(_active) and not _active():
            self.record(ctx, "lean_backcheck",
                        "4.3 Lean 代回：Lean 通道未激活（降级放行）")
            return
        if callable(_applicable) and not _applicable(ctx):
            self.record(ctx, "lean_backcheck",
                        "4.3 Lean 代回：本题不适用（答案非数学对象）⇒ 豁免")
            ctx.lean_backcheck = {"enabled": False, "skipped": "not_applicable",
                                  "n": len(cands)}
            return

        from tools.lean_local.lean_gate import LeanGate as _LG
        _gate = getattr(self, "lean_gate", None) or _LG(self.client, self.config)

        _n_in = len(cands)
        kept, fbs = _gate.apply(ctx, tier, cands)

        # ★ 2026-10-15（工具①接线 · 观测模式）：对存活候选做第二意见检查。
        #   开关 five_gate_lean_check_tool（默认 False）；结果只写 diag/record，
        #   不影响淘汰（等云端验证后再升级为判定通道）。
        if _tool_channel_enabled() and kept:
            try:
                from tools.agent_toolkit import lean_check as _lc
                _res = []
                for _c in kept:
                    _r = _lc(
                        reasoning=((getattr(_c, "answer", "") or "") + "\n" +
                                   (getattr(_c, "reasoning", "") or "")[-1200:]),
                        problem=str(getattr(ctx, "problem", "") or ""),
                        client=getattr(self, "client", None),
                        work_dir=str(getattr(getattr(self, "config", None),
                                             "lean_project_dir", "") or ""))
                    _res.append({
                        "id": getattr(_c, "id", None),
                        "verdict": _r.get("verdict"),
                        "n_errors": len(_r.get("errors") or []),
                        "n_missing": len(_r.get("missing") or []),
                    })
                ctx.lean_tool_check = _res
                self.record(ctx, "lean_backcheck",
                            "4.3（工具①观测）: " + "; ".join(
                                "id=%s:%s" % (x["id"], x["verdict"])
                                for x in _res[:6]))
            except Exception as _e_tool:  # noqa: BLE001  观测通道绝不阻断主链
                self.record(ctx, "lean_backcheck",
                            "4.3（工具①观测）异常（跳过）: %s: %s"
                            % (type(_e_tool).__name__, str(_e_tool)[:100]))

        if kept:
            ctx.candidates = kept
            self.record(ctx, "lean_backcheck",
                        "4.3 Lean 代回：通过 %d/%d（淘汰 %d）"
                        % (len(kept), _n_in, _n_in - len(kept)))
        else:
            # ★★ 2026-10-22（4.3 语义分级）：全拒不再一律"标记放行"。
            #   先看**为什么全拒**（`lean_gate[i].failure_kind`）：
            #     · 全部 refuted（Lean 明确输出 `⊢ False`）⇒ 答案确实错，清池
            #       交 4.6 重写（这是正常判定，不是降级）；
            #     · 含 env / translation / unproved ⇒ Lean 这一环**没有判定能力**
            #       （缺依赖 / 生成的代码不合法 / 没证出来），与答案对错无关，
            #       不清池，但把分布写进 diag 并告警，避免"静默失效"。
            _kinds: dict = {}
            for _e in (getattr(ctx, "lean_gate", None) or []):
                if isinstance(_e, dict):
                    _k = str(_e.get("failure_kind") or "")
                    if _k:
                        _kinds[_k] = _kinds.get(_k, 0) + 1
            _refuted = int(_kinds.get("refuted", 0))
            _other = sum(int(_v) for _k, _v in _kinds.items() if _k != "refuted")
            ctx.lean_backcheck_reject_kinds = dict(_kinds)
            if _refuted and not _other:
                ctx.candidates = []
                self.record(ctx, "lean_backcheck",
                            "4.3 Lean 代回：%d 个候选**全部被证伪**"
                            "（failure_kind=refuted）⇒ 清空候选池交 4.6 重写"
                            % _n_in)
            else:
                ctx._gate_all_rejected = True
                _detail = ("、".join("%s %d" % (k, v)
                                    for k, v in sorted(_kinds.items())) or "无分类信息")
                self.record(ctx, "lean_backcheck",
                            "4.3 Lean 代回：⚠ %d 个候选全部未通过，但原因**不是证伪**"
                            "（%s）⇒ 无法据此判错，保留候选并标记"
                            % (_n_in, _detail))
                if bool(getattr(self.config, "strict_all_rejected", False)):
                    ctx.candidates = []
                    self.record(ctx, "lean_backcheck",
                                "strict_all_rejected=True ⇒ 清空候选池交下游兜底")

        if fbs:
            ctx.audit_reject_feedback = list(
                getattr(ctx, "audit_reject_feedback", None) or []) + list(fbs)
            # 既有契约：单独留一份 Lean 反馈（revise prompt 据此区分形式化缺陷）
            ctx.lean_reject_feedback = list(
                getattr(ctx, "lean_reject_feedback", None) or []) + list(fbs)

        ctx.lean_backcheck = {
            "enabled": True,
            "n": _n_in,
            "n_kept": len(kept),
            "n_rejected": _n_in - len(kept),
            "all_rejected": not kept,
            # ★ 2026-10-22：全拒原因分布（refuted/unproved/translation/env）
            "reject_kinds": dict(
                getattr(ctx, "lean_backcheck_reject_kinds", None) or {}),
        }
        self.record(ctx, "lean_backcheck",
                    "4.3 Lean 代回：%d 候选 ⇒ 通过 %d / 淘汰 %d%s"
                    % (_n_in, len(kept), _n_in - len(kept),
                       "（全拒已标记）" if not kept else ""))

        try:
            from .orchestrator__verify import _pool_empty_fallback as _pef
            _pef(self, ctx, "4.3_lean_check")
        except Exception as _e_pef:  # noqa: BLE001
            # ★ 2026-10-15（审核修复）：不再无痕 pass —— 记错误信息（可归因）
            self.record(ctx, "lean_backcheck",
                        "pool_empty_fallback 失败（不阻断）: %s: %s"
                        % (type(_e_pef).__name__, str(_e_pef)[:100]))
    except Exception as _e:  # noqa: BLE001  增益环节绝不阻断主链
        self.record(ctx, "lean_backcheck",
                    f"4.3 Lean 代回异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
