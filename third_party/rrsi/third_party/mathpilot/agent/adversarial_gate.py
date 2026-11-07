# -*- coding: utf-8 -*-
"""4.2 AI 对抗证伪 —— 五道审核链第 2 道（2026-10-14 用户定案）。

用户口径（原话）：「第二道让 ai 对还存活的答案生成对应的对抗性验证，
来验证能不能证伪，证伪就判错」。

职责：对 4.1 自审后**存活**的每个候选执行两层对抗验证，**证伪即淘汰**：
  ① **机检证伪**（`AnswerFalsifier`）：LLM 构造证伪证书（数值回带等式）→
     SymPy 精确判定拿到反例 ⇒ 淘汰（最高精度，带可机检 witness）；
  ② **LLM 对抗生成**（`AdversarialVerifier.probe`）：假设答案错误、反向寻找
     反例或第一个错误步骤；`is_actionable`（**答案错**类，带具体位置/反例）⇒ 淘汰；
     `is_soft`（跳步 / 边界未讨论 / 题意形式偏差）⇒ **不淘汰**，只留痕（2026-10-20）。

判定纪律（防误杀，与既有对抗链同口径）：
  · 机检证伪以 `res.is_incorrect` 为准（证伪器内部已做"任何不确定 ⇒ 放行"）；
  · LLM 对抗以 `result.is_actionable` 为准（低置信"找到"在 prober 内部已过滤）；
  · 跳过 / 调用失败 / 异常 ⇒ **保留**候选（记为 skipped/failed，绝不因故障淘汰）；
  · 被证伪候选同时进 `ctx.audit_reject_feedback` 与 `ctx._falsified_answers`
    （后者被终选层消费，保持既有契约）。

观测：trace step=``adversarial_gate``，diag 键 ``adversarial_gate`` /
``adversarial_gate_rejected`` / ``adversarial_gate_events``。

五道链位置：4.0 规范化 → 4.1 自审 → **4.2 对抗证伪** → ③Lean 代回 → ④计算检测
→ ⑤投票。
"""
from __future__ import annotations

from .switch_registry import get_bool as _sw_get_bool
from .switch_registry import get_num as _sw_get_num

def _switch_enabled() -> bool:
    """读「五道链·第 2 道对抗证伪」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_falsify")
    except Exception:  # noqa: BLE001
        return True


def _probe_batch(prober, ctx, cand, tier: str, n: int) -> list:
    """并发跑 n 次对抗探测，返回长度 n 的列表（元素为探针结果或异常）。

    ★ 2026-10-22（不浪费并发额度）：多次探测互相独立，串行会让 4.2 的墙钟
      随票数线性增长；并发后墙钟 ≈ 单次。`BaseAgent.llm` 无共享锁、
      LLMClient 每次新建连接 ⇒ 安全；代价是"任一票命中即停"不再成立
      （固定发 n 次）。
    """
    from concurrent.futures import ThreadPoolExecutor
    if n <= 1:
        try:
            return [prober.probe(ctx, cand, tier=tier)]
        except Exception as exc:  # noqa: BLE001
            return [exc]
    out: list = [None] * n
    with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
        futs = {ex.submit(prober.probe, ctx, cand, tier=tier): i
                for i in range(n)}
        for fut, idx in futs.items():
            try:
                out[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                out[idx] = exc
    return out


def _llm_falsify_one(orch, ctx, cand, cid, tier, prober, grounding: bool,
                     fvotes: int) -> dict:
    """单个候选的 LLM 对抗判定（并发采 ``fvotes`` 票）。

    2026-10-22 从 `adversarial_gate_stage` 抽出（原函数 260 行超项目硬线 150）。
    判定纪律逐字保留：
      · **任一次** actionable 且证据落地 ⇒ 淘汰；
      · 证据无法落地的票只记 `ungrounded` 并继续看下一票；
      · 过程瑕疵（is_soft）与跳过只在**全部票都未命中**时才计入。

    返回 ``{"rej","soft","skip","failed","ungrounded"}``。
    """
    out = {"rej": None, "soft": None, "skip": "", "failed": False,
           "ungrounded": []}
    soft_hit = None
    for vi, pr in enumerate(_probe_batch(prober, ctx, cand, tier, fvotes)):
        if isinstance(pr, Exception):
            if vi == 0:
                out["failed"] = True
                orch.record(ctx, "adversarial_gate",
                            "cid=%s 对抗探测异常（保留候选）: %s: %s"
                            % (cid, type(pr).__name__, str(pr)[:120]))
            continue
        if getattr(pr, "is_actionable", False):
            estep = str(getattr(pr, "error_step", "") or "")
            gok, _gr, gmiss = (None, 0.0, [])
            if grounding:
                from .evidence_grounding import check_quote
                if estep:
                    gok, _gr, gmiss = check_quote(
                        estep, str(getattr(cand, "reasoning", "") or ""))
            if grounding and gok is not True:
                # 证据无法落地 ⇒ 本次不成立，但继续看下一票（可能给出可核对证据）
                out["ungrounded"].append({
                    "id": cid,
                    "answer": str(getattr(cand, "answer_norm", None)
                                  or getattr(cand, "answer", "") or "")[:120],
                    "error_step": estep[:120],
                    "error_type": str(getattr(pr, "error_type", "") or ""),
                    "note": ("missing" if not estep
                             else "not_found:%s" % "|".join(gmiss[:2])),
                })
                orch.record(ctx, "adversarial_gate",
                            "cid=%s 证伪证据无法落地（第 %d 票）⇒ 本次不计：%r"
                            % (cid, vi + 1, estep[:40]))
                continue
            out["rej"] = {
                "by": "llm",
                "error_type": str(getattr(pr, "error_type", "") or ""),
                "confidence": float(getattr(pr, "confidence", 0.0) or 0.0),
                "feedback": str(pr.to_feedback() or "")[:600],
            }
            return out
        if getattr(pr, "is_soft", False):
            if soft_hit is None:
                soft_hit = pr
            continue                  # 过程瑕疵不算结论，继续找硬错
        if getattr(pr, "skipped", ""):
            out["skip"] = out["skip"] or str(getattr(pr, "skipped", ""))
    if soft_hit is not None:
        st = str(getattr(soft_hit, "error_type", "") or "") or "?"
        out["soft"] = {
            "id": cid,
            "error_type": st,
            "confidence": float(getattr(soft_hit, "confidence", 0.0) or 0.0),
            "note": str(soft_hit.to_feedback() or "")[:300],
        }
        out["skip"] = ""      # 与原逻辑一致：soft 与 skip 互斥，soft 优先
        orch.record(ctx, "adversarial_gate",
                    "cid=%s 仅过程瑕疵（%s）⇒ 不淘汰，放行" % (cid, st))
    return out


def _grounding_enabled() -> bool:
    """读「证据落地校验」开关（默认 True，与 4.1 同一开关）。

    ★ 2026-10-22（五道链审计）：LLM 对抗给出的 ``error_step`` 必须能在候选
      推理里**逐字找到**，否则不构成淘汰依据。桩测试实测：随便编一句
      「故 n=2」作 error_step，正确答案即被判 `is_actionable` 淘汰。
    """
    try:
        return _sw_get_bool("five_gate_grounding")
    except Exception:  # noqa: BLE001
        return True


def _falsify_votes() -> int:
    """读「对抗证伪采样次数」（``five_gate_falsify_votes``，默认 3）。

    ★ 2026-10-22 实测（书生 Intern，真实调用）：同一个错误候选（推理正确、
      只把终答写错）连跑 3 次对抗探测，结果是 **「证伪 / 放过 / 放过」** ——
      与 4.1 自审同样的不稳定，单次采样漏检约 2/3。
      因此同样采 N 次，**任一次** actionable 且证据落地即淘汰。
      =1 回到旧行为（单次探测），供 A/B。
    """
    try:
        return max(1, int(_sw_get_num("five_gate_falsify_votes")))
    except Exception:  # noqa: BLE001
        return 3


def adversarial_gate_stage(self, ctx, tier: str) -> None:
    """4.2 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例）。

    注入点约定（与 4.7_lean 同款，便于单测注入桩）：
      · 证伪器：``getattr(self, "adv_gate_falsifier", None) or AnswerFalsifier(...)``
      · 对抗器：``getattr(self, "adv_verifier", None) or AdversarialVerifier(...)``
    """
    try:
        if not _switch_enabled():
            self.record(ctx, "adversarial_gate",
                        "4.2 对抗证伪：开关关闭（five_gate_falsify=0）")
            return
        cands = list(getattr(ctx, "candidates", None) or [])
        if not cands:
            self.record(ctx, "adversarial_gate", "4.2 对抗证伪：候选池为空，跳过")
            return

        from .answer_falsifier import AnswerFalsifier
        from .adversarial_verifier import AdversarialVerifier

        _falsifier = getattr(self, "adv_gate_falsifier", None) or \
            AnswerFalsifier(self.client, self.config)
        _prober = getattr(self, "adv_verifier", None) or \
            AdversarialVerifier(self.client, self.config)
        prob = str(getattr(ctx, "problem", "") or "")

        kept, rejected, soft = [], [], []
        n_skipped, n_failed = 0, 0
        _grounding = _grounding_enabled()
        _fvotes = _falsify_votes()
        ungrounded = []   # LLM 指认的出错步骤在原文里找不到 ⇒ 不得据此淘汰

        for c in cands:
            cid = getattr(c, "id", "?")
            # ① 机检证伪（LLM 证书 + SymPy 判定）
            _machine_rej = None
            try:
                res = _falsifier.falsify(ctx, c, problem=prob)
                if getattr(res, "is_incorrect", False):
                    _machine_rej = {
                        "by": "machine",
                        "reason": str(getattr(res, "reason", "") or "")[:200],
                        "witness": str(getattr(res, "witness", "") or "")[:200],
                        "oracle_type": str(getattr(res, "oracle_type", "") or ""),
                    }
            except Exception as _e_f:  # noqa: BLE001  故障绝不淘汰候选
                n_failed += 1
                self.record(ctx, "adversarial_gate",
                            "cid=%s 机检证伪异常（保留候选）: %s: %s"
                            % (cid, type(_e_f).__name__, str(_e_f)[:120]))

            # ② LLM 对抗生成（机检已判错则不再调用，省一次重证）
            #    2026-10-22：判定细节抽到 `_llm_falsify_one`（原函数 260 行超硬线）。
            _llm_rej = None
            if _machine_rej is None:
                _r2 = _llm_falsify_one(self, ctx, c, cid, tier, _prober,
                                       _grounding, _fvotes)
                _llm_rej = _r2["rej"]
                if _r2["soft"] is not None:
                    soft.append(_r2["soft"])
                if _r2["skip"]:
                    n_skipped += 1
                if _r2["failed"]:
                    n_failed += 1
                ungrounded.extend(_r2["ungrounded"])

            _rej = _machine_rej or _llm_rej
            # ★ 2026-10-17（用户定案「加一个其他大模型检错」）：生成/自审/证伪
            #   都是同一模型 ⇒ 盲区相同。对**前两道都放行**的候选，调第二模型
            #   （DeepSeek）独立检错（附带题面约束/歧义点）；wrong ⇒ 淘汰；
            #   失败/超时 ⇒ 放行并记 trace（不降级判错）。
            if _rej is None and _sw_get_bool("cross_model_falsify"):
                try:
                    from .cross_model_check import cross_check as _cross_check
                    _cons = list(getattr(ctx, "problem_ambiguities", None) or [])
                    _uc = getattr(ctx, "understanding_chain", None)
                    if isinstance(_uc, dict) and _uc.get("terminal_state"):
                        _cons.append(str(_uc["terminal_state"])[:300])
                    _ccr = _cross_check(
                        problem=prob,
                        answer=str(getattr(c, "answer_norm", None)
                                   or getattr(c, "answer", "") or ""),
                        reasoning=str(getattr(c, "reasoning", "") or ""),
                        constraints=_cons,
                        timeout=float(getattr(self.config,
                                              "cross_model_timeout", 90) or 90))
                    _st = getattr(ctx, "cross_model_stats", None)
                    if not isinstance(_st, dict):
                        _st = {}
                        ctx.cross_model_stats = _st
                    _vk = str(_ccr.get("verdict") or "unknown")
                    _st[_vk] = int(_st.get(_vk, 0)) + 1
                    if _vk == "wrong":
                        _rej = {"by": "cross_model",
                                "reason": str(_ccr.get("evidence") or "")[:300],
                                "assumption": str(_ccr.get("assumption") or "")[:200],
                                "model": str(_ccr.get("model") or "")}
                        self.record(ctx, "adversarial_gate",
                                    "cid=%s 跨模型独立检错：判错 ⇒ 淘汰；理由 %s"
                                    % (cid, str(_ccr.get("evidence") or "")[:160]))
                    elif _vk == "unknown":
                        self.record(ctx, "adversarial_gate",
                                    "cid=%s 跨模型检查未定（放行）: %s"
                                    % (cid, str(_ccr.get("error") or "")[:120]))
                except Exception as _e_c:  # noqa: BLE001
                    self.record(ctx, "adversarial_gate",
                                "cid=%s 跨模型检查异常（放行）: %s: %s"
                                % (cid, type(_e_c).__name__, str(_e_c)[:120]))
            if _rej is None:
                kept.append(c)
                self.record(ctx, "adversarial_gate",
                            "cid=%s 未证实伪（放行）" % cid)
                continue

            _ans = str(getattr(c, "answer_norm", None)
                       or getattr(c, "answer", "") or "")[:160]
            rejected.append({"id": cid, "answer": _ans, **_rej})
            self.record(ctx, "adversarial_gate",
                        "cid=%s **被证伪**（%s）⇒ 判错淘汰：%s"
                        % (cid, _rej["by"],
                           (_rej.get("reason") or _rej.get("error_type") or "")[:200]))

        if rejected:
            ctx.candidates = kept
            ctx.adversarial_gate_rejected = list(
                getattr(ctx, "adversarial_gate_rejected", None) or []) + rejected
            _fb = []
            for r in rejected:
                if r["by"] == "machine":
                    _fb.append("[对抗证伪·机检] cid=%s: %s | 反例：%s"
                               % (r["id"], r.get("reason") or "", r.get("witness") or ""))
                else:
                    _fb.append("[对抗证伪·LLM] cid=%s: %s"
                               % (r["id"], r.get("feedback") or r.get("error_type") or ""))
            ctx.audit_reject_feedback = list(
                getattr(ctx, "audit_reject_feedback", None) or []) + _fb
            # 既有契约：被证伪答案进 `_falsified_answers`（终选层消费）
            try:
                _fa = list(getattr(ctx, "_falsified_answers", None) or [])
                _fa += [r["answer"] for r in rejected if r["by"] == "machine"]
                ctx._falsified_answers = _fa
                _meta = ctx.metadata if isinstance(ctx.metadata, dict) else {}
                _meta["falsified_answers"] = _fa
                ctx.metadata = _meta
            except Exception:  # noqa: BLE001  元数据写入失败不影响淘汰结果
                pass

        if soft:
            ctx.adversarial_gate_soft = list(
                getattr(ctx, "adversarial_gate_soft", None) or []) + soft
        # ★ 按来源自动分组：`by` 目前有 machine / llm / cross_model 三种，
        #   旧写法只统计前两种 ⇒ 「淘汰 3（机检 0 / LLM 1）」分项和只有 1
        #   （2026-10-20 collab019_r1：实际 cross_model×2 + llm×1，跨模型这条主力被漏计）。
        #   此处不写死枚举：新增来源自动进 `n_by`，日志也自动带上，
        #   不会再出现「淘汰数对、分项加起来不对」的观测失真。
        n_by = {}
        for _r in rejected:
            _k = _r.get("by") or "unknown"
            n_by[_k] = n_by.get(_k, 0) + 1
        _BY_LABEL = {"machine": "机检", "llm": "LLM", "cross_model": "跨模型"}
        _by_txt = " / ".join(
            "%s %d" % (_BY_LABEL.get(k, k), v) for k, v in n_by.items()) or "无"
        ctx.adversarial_gate = {
            "n": len(cands),
            "n_soft": len(soft),
            "soft_ids": [s["id"] for s in soft],
            "n_rejected": len(rejected),
            # 兼容既有读者（stage_audit / diag_export 按这两个键取数）
            "n_machine": n_by.get("machine", 0),
            "n_llm": n_by.get("llm", 0),
            "n_cross_model": n_by.get("cross_model", 0),
            "n_by": n_by,
            "n_skipped": n_skipped,
            "n_failed": n_failed,
            "n_ungrounded": len(ungrounded),
            "ungrounded_ids": [r["id"] for r in ungrounded],
            "rejected_ids": [r["id"] for r in rejected],
        }
        if ungrounded:
            ctx.adversarial_gate_ungrounded = list(
                getattr(ctx, "adversarial_gate_ungrounded", None) or []) + ungrounded
        self.record(
            ctx, "adversarial_gate",
            "4.2 对抗证伪：%d 候选 ⇒ 证伪淘汰 %d（按来源：%s）"
            "（跳过 %d、故障 %d、过程瑕疵放行 %d、证据无法落地 %d）"
            % (len(cands), len(rejected), _by_txt,
               n_skipped, n_failed, len(soft), len(ungrounded)))

        try:
            from .orchestrator__verify import _pool_empty_fallback as _pef
            _pef(self, ctx, "4.2_falsify")
        except Exception:  # noqa: BLE001  兜底检查失败不影响主流程
            pass
    except Exception as _e:  # noqa: BLE001  增益环节绝不阻断主链
        self.record(ctx, "adversarial_gate",
                    f"4.2 对抗证伪异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
