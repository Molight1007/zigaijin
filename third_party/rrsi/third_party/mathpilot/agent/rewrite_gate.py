# -*- coding: utf-8 -*-
"""4.6 重写通道 —— 五道审核链配套（2026-10-14 用户定案）。

用户口径（原话）：「不要这个东西（emergency 直答兜底）！没有答案了就讲错误
情况返回给大模型，让大模型找到问题出在哪里，重新构建蓝图和子目标，重写」。

职责：当五道审核把候选**全部淘汰**（池空）时，不走"直接要一个答案"，而是：
  ① 汇总错误情况（各道淘汰原因：pool_empty_errors + 各道 rejected 清单）；
  ② **大模型诊断**：找出问题出在哪里（root_cause）+ 给出重写调整指令（fix）；
  ③ **重新构建蓝图和子目标 + 重写**：把「错误情况 + 诊断」写入
     `ctx.rewrite_context`（经蓝图/子目标规划的"题目"占位符内部注入），
     重置规划状态后重跑 `sub_goal_solver.run`（蓝图 DAG → 子目标 → 逐步求解
     → 合并 ⇒ 追加新候选）；
  ④ 新候选**重过五道链**（规范化→自审→证伪→Lean→计算→投票）；
  ⑤ 仍全灭 ⇒ 带着上一轮错误再循环（上限 `rewrite_max_rounds`，默认 2）。

判定纪律：
  · 触发条件 = `ctx.candidates` 为空（五道链清空池）；池非空 ⇒ 本阶段不动作；
  · 轮数耗尽仍空 ⇒ 留痕交下游（formatter 的"必须有答案"兜底仍在地基，
    但不再是"池空直接要答案"的路径——先修复，修复尽再走）；
  · 任何异常 ⇒ 记录后不阻断主链。

观测：trace step=``rewrite``，diag 键 ``rewrite_gate`` / ``rewrite_gate_events``。

五道链位置：4.0 → 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → **4.6_rewrite（池空修复）** → 6_format。
"""
from __future__ import annotations

import re

from .switch_registry import get_bool as _sw_get_bool

_PROBLEM_MAX = 4000
_ERRORS_MAX = 3000
_REJECTED_MAX = 1600
_CTX_MAX = 2500        # 注入 rewrite_context 的总长度（蓝图/规划共用）


def _parse_diag_json(raw) -> dict:
    """容错解析诊断输出。{} = 无法解析（调用方用裸错误摘要继续重写）。"""
    if not raw:
        return {}
    txt = str(raw).strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
    try:
        import json as _json_rw
        _m = txt[txt.find("{"):txt.rfind("}") + 1] if "{" in txt else ""
        if not _m:
            return {}
        d = _json_rw.loads(_m)
        if not isinstance(d, dict):
            return {}
        if not (d.get("root_cause") or d.get("fix")):
            return {}
        return {"root_cause": str(d.get("root_cause") or "").strip()[:600],
                "fix": str(d.get("fix") or "").strip()[:1200]}
    except Exception:  # noqa: BLE001
        return {}


def _switch_enabled() -> bool:
    """读「五道链·重写通道」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_rewrite")
    except Exception:  # noqa: BLE001
        return True


def _collect_error_brief(ctx) -> str:
    """汇总各道淘汰原因（错误情况）。无留痕时给保守描述。"""
    lines = []
    for e in (getattr(ctx, "pool_empty_errors", None) or []):
        if isinstance(e, dict):
            lines.append("- [%s] %s" % (e.get("stage") or e.get("kind") or "?",
                                        (e.get("error") or "")[:220]))
    for _key, _tag in (("self_audit_rejected", "4.1自审"),
                       ("adversarial_gate_rejected", "4.2证伪"),
                       ("math_check_rejected", "4.4计算检测")):
        for r in (getattr(ctx, _key, None) or []):
            if isinstance(r, dict):
                lines.append("- [%s] %s" % (_tag, str(
                    r.get("error_desc") or r.get("reason")
                    or r.get("evidence") or "")[:220]))
    _lg = getattr(ctx, "lean_gate", None)
    if isinstance(_lg, list):
        for g in _lg:
            if not isinstance(g, dict):
                continue
            # ★ 2026-10-15 审核修复（S3）：lean_gate 的实际字段口径 ——
            #   verdict ∈ {proof_invalid, unknown}；strict 拒绝落在 `degraded`
            #   （如 "strict_reject"）；错误文本在 `feedback`。原实现只读
            #   verdict∈{reject,strict_reject}/reason ⇒ 两类淘汰原因全部丢失。
            _v = str(g.get("verdict") or "")
            _deg = str(g.get("degraded") or "")
            if (_v in ("reject", "strict_reject", "proof_invalid")
                    or _deg == "strict_reject"):
                _txt = str(g.get("feedback") or g.get("reason")
                           or g.get("error") or "").strip()
                lines.append("- [4.3Lean] %s%s"
                             % (_txt[:180] or "(无文本)",
                                ("（degraded=%s）" % _deg) if _deg else ""))
    if not lines:
        lines.append("（无具体错误点留痕：候选在各道被整体淘汰，或错误点未记录）")
    # ★★ 2026-10-23（用户：「把正确的判断为错误的，给大模型矛盾的答案」）：
    #   实测 val1_r1_1022 —— 4 条淘汰理由平铺，其中 cross_model 那条**已经写出了
    #   正确上界**「由 AM-HM 该和 ≥ (N-1)²/(N(N+1)-3) → 1，故 D ≤ M = 777² = 603729」，
    #   但它排在第 4 位、还和「构造证明有误 / 与义务 O5 矛盾」等理由混在一起，
    #   模型无法判断谁对 ⇒ 两轮重写仍给 777²/2。
    #   此外有一条理由正文为空（"- [4.2_falsify] "），纯噪声。
    #   新口径：
    #     ① 丢弃空壳理由（前缀之外没有实质内容）；
    #     ② **含具体数值/不等式/反例**的理由置顶，并显式标注「优先采信」——
    #        这类理由自带可核对依据，比"某某不成立"的结论性描述可信；
    #     ③ 其余理由单列，明确降级为"参考"。
    _clean = [l for l in lines if len(str(l).strip()) > 20]
    if not _clean:
        _clean = list(lines)
    # ★★ 2026-10-23（用户：「错误的地方在于把正确的判断为错误的」）：
    #   实测 cid=0 的淘汰理由 `error_at` 是——「但子目标描述要求的是 (m+n)D ≤ 777²，
    #   这与推导矛盾。」 这句话其实是**候选自己发现了上游子目标的矛盾**
    #   （sg4 假设 ∑(p_k+p_{k+1}) ≤ N，而正确量级是 ≈N²）—— 是**正确的洞察**。
    #   但 4.1 的视角是"解答是否自洽"，于是把这个洞察当成"解答逻辑错误"淘汰掉，
    #   **该回传上游的信号就此丢失**，4.6 重写仍然从错误的子目标链出发。
    #   ⇒ 这里把它单独拎成一段，明确告诉重写模型：这不是候选的错，是子目标要修。
    _up = []
    for r in (getattr(ctx, "self_audit_rejected", None) or []):
        if isinstance(r, dict):
            _ea = str(r.get("error_at") or "")
            if "子目标" in _ea:
                _up.append("- %s" % _ea[:200])
    _QUANT_RE = re.compile(r"\d{3,}|≤|≥|\\le|\\ge|反例")
    _quant = [l for l in _clean if _QUANT_RE.search(str(l))]
    _rest = [l for l in _clean if l not in _quant]
    _out = []
    if _up:
        _out.append("【候选指出的上游问题 —— 这不是候选算错，是**子目标链**要改的地方】")
        _out += _up
    if _quant:
        _out.append("【优先采信：以下理由给出了具体数值 / 不等式 / 反例，请以它们为基准】")
        _out += _quant
    if _rest:
        if _quant:
            _out.append("【其余（仅结论性描述、可信度较低，可参考但勿据此推翻上面的定量结论）】")
        _out += _rest
    return "\n".join(_out or _clean)[:_ERRORS_MAX]


def _collect_rejected_brief(ctx) -> str:
    """被淘汰候选答案摘要（含各道原因缩略）。"""
    lines = []
    for _key, _tag in (("self_audit_rejected", "4.1"),
                       ("adversarial_gate_rejected", "4.2"),
                       ("math_check_rejected", "4.4")):
        for r in (getattr(ctx, _key, None) or []):
            if isinstance(r, dict):
                lines.append("- [%s] %s" % (_tag, str(r.get("answer") or "")[:60]))
    return "\n".join(lines)[:_REJECTED_MAX] or "（无）"


def _reset_plan_state(ctx) -> None:
    """重置规划/求解状态，使「重建蓝图和子目标」真实重跑（而非复用旧产物）。"""
    # ★ 2026-10-17（p0b5_r3 审核）：原实现漏清 `_subgoal_done`（子目标结果缓存）
    #   ⇒ 重写时旧子目标指纹命中缓存、直接复用旧结果 ⇒ 产出候选与上一轮相同
    #   （题6/7 重写后仍是旧的 \boxed{4}/\boxed{76} 的根因）。一并清除。
    for _f in ("blueprint_plan", "blueprint_nodes", "_subgoal_done"):
        try:
            setattr(ctx, _f, None)
        except Exception:  # noqa: BLE001
            pass
    # 子目标主路径完成位（orchestrator 2.7 标记；重跑期间允许再次置位）
    try:
        ctx._subgoal_main_done = False
    except Exception:  # noqa: BLE001
        pass


def _best_confidence(ctx) -> "float | None":
    """投票最佳候选的置信度 = vote_correct / vote_total；无投票信息返回 None。"""
    try:
        _bc = (getattr(ctx, "vote_gate", None) or {}).get("best_cluster") or {}
        _c = _bc.get("vote_correct"); _t = _bc.get("vote_total")
        if _c is None or _t is None or int(_t) <= 0:
            return None
        return float(_c) / float(_t)
    except Exception:  # noqa: BLE001
        return None


def _should_rewrite(self, ctx) -> bool:
    """★ 2026-10-19（用户「检出错误 → 返回错误 → 重做，不放行错答案」）：
    重写触发条件由「池空」扩展到「池空 或 (有淘汰证据 且 最佳候选置信度为 0)」。

    · 池空 ⇒ 重写（原行为）；
    · 池非空但 4.2/4.4 检出过错、且投票最佳候选「0 正确票」⇒ 也重写（带回错误反馈）；
    · 无投票信息 / 置信度 > 阈值 / 无淘汰证据 ⇒ 不重写（避免每题都重写、成本失控）。
    """
    if not getattr(ctx, "candidates", None):
        return True
    try:
        from .switch_registry import get_bool as _g
        if not _g("rewrite_on_low_conf"):
            return False
    except Exception:  # noqa: BLE001
        pass
    _thr = float(getattr(self.config, "rewrite_conf_threshold", 0.0) or 0.0)
    _conf = _best_confidence(ctx)
    if _conf is None or _conf > _thr:
        return False
    _ad = getattr(ctx, "adversarial_gate", None) or {}
    _mc = getattr(ctx, "math_check", None) or {}
    return bool((_ad.get("n_rejected") or 0) > 0 or (_mc.get("n_fail") or 0) > 0)


def rewrite_stage(self, ctx, tier: str) -> None:
    """4.6 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例）。"""
    try:
        if not _switch_enabled():
            self.record(ctx, "rewrite",
                        "4.6 重写：开关关闭（five_gate_rewrite=0）")
            return
        if not _should_rewrite(self, ctx):
            self.record(ctx, "rewrite", "4.6 重写：候选池非空且无重写信号，无需重写")
            return
        if getattr(ctx, "candidates", None):
            self.record(ctx, "rewrite",
                        "4.6 重写：候选池非空但置信度 0 且存在淘汰证据 ⇒ 带反馈重写")

        # 延迟导入各阶段函数（重写成功后让新候选重过五道链；不触发本阶段自身）
        from .answer_normalizer import normalize_stage
        from .self_auditor import self_audit_stage
        from .adversarial_gate import adversarial_gate_stage
        from .lean_backcheck import lean_backcheck_stage
        from .math_check_gate import math_check_stage
        from .vote_gate import vote_stage
        from prompts.rewrite import REWRITE_DIAG_SYSTEM, REWRITE_DIAG_USER

        _max_rounds = int(getattr(self.config, "rewrite_max_rounds", 2) or 2)
        problem = str(getattr(ctx, "problem", "") or "")[:_PROBLEM_MAX]
        rounds, diag_ok, produced = 0, 0, 0

        for rnd in range(1, _max_rounds + 1):
            rounds = rnd
            errs = _collect_error_brief(ctx)
            rejs = _collect_rejected_brief(ctx)
            # ① 大模型诊断：问题出在哪里 + 重写调整指令
            diag_d = {}
            try:
                raw = self.llm(ctx, [
                    {"role": "system", "content": REWRITE_DIAG_SYSTEM},
                    {"role": "user", "content": REWRITE_DIAG_USER.format(
                        problem=problem, errors=errs, rejected=rejs)},
                ], 0.1, 8192)
                diag_d = _parse_diag_json(raw)
                if diag_d:
                    diag_ok += 1
            except Exception as _e_d:  # noqa: BLE001  诊断失败仍继续重写
                self.record(ctx, "rewrite",
                            "第%d轮诊断调用失败（用裸错误摘要重写）: %s"
                            % (rnd, str(_e_d)[:120]))
            _rc = diag_d.get("root_cause") or "（诊断未给出，依据错误情况直接重写）"
            _fx = diag_d.get("fix") or "请依据上述错误情况调整分解与求解方法后重写。"
            self.record(ctx, "rewrite",
                        "第%d轮诊断：根因=%s；调整=%s"
                        % (rnd, _rc[:160], _fx[:200]))

            # ② 组「错误情况 + 诊断」→ rewrite_context（注入规划/求解的题目占位符内部）
            ctx.rewrite_context = (
                "【上一轮全部候选被审核淘汰——错误情况汇总】\n" + errs +
                "\n【被淘汰的候选答案摘要】\n" + rejs +
                "\n【诊断与重写调整指令（必须遵循）】\n根因：" + _rc +
                "\n调整：" + _fx + "\n（重写时务必避免上述同类错误。）"
            )[:_CTX_MAX]
            ctx.rewrite_round = (getattr(ctx, "rewrite_round", 0) or 0) + 1

            # ③ 重建蓝图和子目标 + 重写（重跑子目标主链，结果追加到候选池）
            _reset_plan_state(ctx)
            try:
                self.sub_goal_solver.run(ctx)
            except Exception as _e_s:  # noqa: BLE001  重写异常 ⇒ 记录，按未产出处理
                self.record(ctx, "rewrite",
                            "第%d轮重写异常: %s: %s"
                            % (rnd, type(_e_s).__name__, str(_e_s)[:160]))
            if not getattr(ctx, "candidates", None):
                self.record(ctx, "rewrite", "第%d轮重写未产出候选" % rnd)
                continue
            produced += 1

            # ④ 新候选重过五道链（不触发 rewrite 自身，防递归）
            normalize_stage(self, ctx)
            self_audit_stage(self, ctx)
            adversarial_gate_stage(self, ctx, tier)
            lean_backcheck_stage(self, ctx, tier)
            math_check_stage(self, ctx)
            vote_stage(self, ctx, tier)
            if getattr(ctx, "candidates", None):
                self.record(ctx, "rewrite",
                            "第%d轮重写成功：新候选通过五道链（存活 %d 个）"
                            % (rnd, len(ctx.candidates)))
                break
            self.record(ctx, "rewrite",
                        "第%d轮重写候选再次被审核全灭，进入下一轮" % rnd)

        ctx.rewrite_gate = {
            "rounds": rounds, "max_rounds": _max_rounds,
            "n_diag_ok": diag_ok, "n_produced": produced,
            "survived": bool(getattr(ctx, "candidates", None)),
            "root_cause": (_rc if rounds else ""),
        }
        self.record(ctx, "rewrite",
                    "4.6 重写：%d/%d 轮完成（诊断成功 %d 次，产候选 %d 次，"
                    "最终%s）" % (rounds, _max_rounds, diag_ok, produced,
                                  "存活" if ctx.rewrite_gate["survived"] else "仍空"))
    except Exception as _e:  # noqa: BLE001  修复环节绝不阻断主链
        self.record(ctx, "rewrite",
                    f"4.6 重写异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
