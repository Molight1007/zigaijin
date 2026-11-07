# -*- coding: utf-8 -*-
"""AuditGate —— 答案审核闸门（取代 Lean 系在检测链中的全部作用）。

设计背景
========
平台无 Lean 可执行文件（历史归因实证），lean_gate / lean_pre_verifier /
answer_oracle.verify_proof 在平台上全部 unknown 空转：只付时间不审核，
最终答案闸门形同虚设。本模块把"客观审核"从 Lean 迁移到：

  Level 0  程序硬核验（0 LLM）  DeterministicChecker.check_answer
                                  （答案代回题干数值回验 / SymPy 判定）
  Level 1  程序反例证伪（0 LLM） search_counterexample
                                  （对非数值答案做数值采样反例挑战）
  Level 2  LLM rubric 判分       verifier._vote_rubric / 证明步骤审核
                                  （JSON prefill 秒级，错因定位驱动 revise）
  Level 3  独立重算一致性        verifier._playoff_recheck（低温重解对比）

职责边界（只审不答）
====================
- 审：对候选解答与最终答案产出裁决（verdict/feedback/evidence）；
- 不答：不生成答案、不改写题目、不做规划/子目标、不产出 Lean 代码。

硬否决优先、宁 unknown 不误杀：
- Level0 fail / Level1 hard_fail → 客观证据，硬否决（可触发换候选/重做）；
- Level2 rubric B（LLM 同源，可能误伤）→ 仅当置信度高才采纳，否则放行；
- 全部候选被否 → 回退保留原候选（保证有输出，不损失分数，镜像 LeanGate 降级）。

对外契约
========
- ``audit_candidates(ctx, tier, candidates) -> (kept, feedbacks)``
- ``gate_final_answer(ctx, tier, answer, reasoning) -> bool``

2026-09-06 依据《检测链重构_去Lean化_三方案_0905.md》方案二实现。
★ 2026-10-14（题意理解收敛）：原 ``confirm_understanding``（题前理解确认）**已删除**
  —— 它是**双重不可达的死分支**：读的开关 ``audit_confirm_understanding`` 在
  ``AgentConfig`` 里从未声明（``getattr(..., False)`` 恒 False ⇒ 恒 skipped），
  且调用点位于 ``lean_pre_verify`` 的 ``else`` 分支（deep 档 + Lean 可用时永远到不了）。
  其职责（复述题型 / 未知量 / 关键条件）已由 2.6 的 ``judge_question_type`` +
  ``problem_goal`` 覆盖。
"""
from __future__ import annotations

import logging

from .base import BaseAgent, TaskContext
from .deterministic import DeterministicChecker

logger = logging.getLogger("MathPilot.AuditGate")


class AuditGate(BaseAgent):
    """答案审核闸门：Level0-3 多级瀑布，只审不答。"""

    name = "AuditGate"

    def __init__(self, client, config, budget=None):
        super().__init__(client, config)
        self.budget = budget
        self._verifier = None

    # ------------------------------------------------------------------
    # 内部：懒加载复用 VerifierAgent（rubric/挑战/playoff/证明步骤验证）
    # ------------------------------------------------------------------
    @property
    def _verifier_inst(self):
        if self._verifier is None:
            from .verifier import VerifierAgent
            self._verifier = VerifierAgent(self.client, self.config)
        return self._verifier

    # ------------------------------------------------------------------
    # 公共：轨迹记录
    # ------------------------------------------------------------------
    def _record_ctx(self, ctx: TaskContext, data: dict) -> None:
        try:
            ctx.audit_gate.append(data)
        except Exception:  # noqa: BLE001
            try:
                ctx.audit_gate = [data]
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 主入口（满足 BaseAgent 抽象接口；按需分发到三个审核位）
    # ------------------------------------------------------------------
    def run(self, ctx: TaskContext, stage: str = "final",
            candidates: list | None = None) -> dict:
        """统一入口。``stage``: ``candidates``(3.6) / ``final``(6.5)。
        （原 ``understanding``(2.6) 位已随题意理解收敛删除，见模块头说明。）
        """
        if stage == "candidates":
            kept, fb = self.audit_candidates(
                ctx, getattr(ctx, "tier", "standard") or "standard",
                candidates or [])
            return {"kept": kept, "feedbacks": fb}
        # ★ 2026-09-23 新增：**非证明题早退**。
        #   依据（0923 环节效能审计实测）：本闸门 37/46 题触发、235 条 verdict
        #   **全 unknown**（本文件 255-268 行已自陈是"结构性无输出"——
        #   Level2 rubric 只对证明题运行，Level0 确定性检查对解答题极少判 fail），
        #   召回 3%、误杀 0% ⇒ **纯空转**，却仍耗 0.37h 并向 diag 注入 234 条噪音。
        #   早退同时达成"省时"与"去噪"两件事，且**不改变任何判定结果**（原本也全是 unknown）。
        # ★★ 2026-10-07 修正（fix-audit-gate）：早退的**前提（"解答题结构性无输出"）已不成立**。
        #   诊断发现"无输出"的真因是 **Level0 解析前未剥 `\boxed{}` 外壳**（7/10 题在
        #   method="parse" 即返回 unknown，根本没进判据）+ 判据过窄（只认"最后一条常量等式"）。
        #   两者已由 `DeterministicChecker` 重写（开关 `deterministic_solution_check`）修复，
        #   现对解答题确有输出（含函数方程回代等判据）⇒ 此时再早退就是**把刚修好的
        #   判别力重新关掉**。故：**仅当 solution_check 关闭时**才保留早退（=旧行为回退路径）。
        if (stage == "final"
                and getattr(self.config, "audit_gate_skip_non_proof", True)
                and not getattr(self.config, "deterministic_solution_check", True)):
            _qt = str(getattr(ctx, "question_type", "") or "")
            _dom = str(getattr(ctx, "domain", "") or "")
            if not ((_qt == "证明题") or ("证明" in _dom)):
                try:
                    self._record_ctx(ctx, {
                        "step": "final_gate", "skipped": "non_proof_early_exit",
                        "verdict": "unknown", "applicable": False,
                        "note": "解答题：确定性判别重写已关（deterministic_solution_check=0）"
                                "且 Level2 rubric 不适用 ⇒ 本闸门对该题型结构性无输出，早退"})
                except Exception as _e_rc:  # noqa: BLE001
                    # ★ 2026-10-15（审核修复）：不再无痕 pass —— 记错误信息
                    self.record(ctx, "audit_gate",
                                "记录 skipped 明细失败（不阻断）: %s: %s"
                                % (type(_e_rc).__name__, str(_e_rc)[:100]))
                self.record(ctx, "audit_gate",
                            "非证明题早退（solution_check 关闭 + 结构性无输出）")
                return {"ok": True, "skipped": "non_proof"}
        ok = self.gate_final_answer(
            ctx, getattr(ctx, "tier", "standard") or "standard",
            getattr(ctx, "final_response", "") or "")
        return {"ok": ok}

    # ------------------------------------------------------------------
    # 3.6 位：候选审核（顶替原 lean_filter）
    # ------------------------------------------------------------------
    def audit_candidates(self, ctx: TaskContext, tier: str,
                         candidates: list) -> tuple[list, list]:
        """对候选做客观审核，返回 ``(kept, feedbacks)``。

        Level0 程序硬核验：deterministic fail → 淘汰并收集反馈；
        全部候选被否 → 回退保留原候选（宁 unknown 不误杀）。
        证明题且开启 rubric 时叠加 Level2 步骤/结构化判分（B 且高置信才淘汰）。
        """
        kept = list(candidates)
        feedbacks: list[str] = []
        if not candidates:
            return kept, feedbacks
        ctx.audit_gate = getattr(ctx, "audit_gate", []) or []

        domain = getattr(ctx, "domain", "") or ""
        qtype = getattr(ctx, "question_type", "") or ""
        is_proof = (domain in ("证明", "证明题") or qtype == "证明题")
        # 2026-10-07：rubric 是否对本题型可用 —— 证明题恒可（旧行为）；解答题仅当
        #   `audit_gate_rubric_nonproof=1` 时可用（默认关，见 switch_registry 说明）。
        use_rubric = bool(getattr(self.config, "use_rubric", False)) and (
            is_proof or bool(getattr(self.config, "audit_gate_rubric_nonproof", False)))
        # 解答题确定性判别重写是否开（开则"结构性无输出"的旧结论不成立）
        sol_check = bool(getattr(self.config, "deterministic_solution_check", True))

        n_fail = 0
        rejected: list[tuple] = []
        # ★ 2026-09-16：空答案身份集（见循环内注释）。用 id() 而非值比较，
        #   因为候选可能是 dataclass 或 dict，且允许"同值不同对象"。
        _empty: set = set()
        for cand in candidates:
            # 时间保护：单题接近 deadline → 提前退出逐候选审核（Level0 数值代回不再
            # 触发），与下方 rubric 分支的 is_time_critical 口径一致；留痕便于区分
            # "因时间跳过" 与 "根本没触发"。
            if ctx.is_time_critical():
                self.record(ctx, "audit_gate_timeout",
                            f"时间临界，提前退出候选审核：已否决 {n_fail}/{len(candidates)}")
                break
            ans = (cand.get("answer", "") if isinstance(cand, dict)
                   else getattr(cand, "answer", ""))
            # ★★★ 2026-09-16 修复（实测驱动）：空答案此前只是 `continue`（**跳过不拒绝**），
            #   于是空答案留在候选池里进了投票 —— 实测 official112-003 的候选 3
            #   `\boxed{}`（**空**）**吃掉了 3 张票**、稀释了共识，还占用 LeanGate 时间。
            #   它还一路通过所有闸门（`audit_gate` 跳过、LeanGate 因 `unknown`
            #   走非 strict 的 `lenient_pass` 放行），只在最后 formatter 的
            #   `not answer.strip()` 才被丢弃。
            #   本函数跑在**投票之前**，是剔除它的正确位置。
            #   注意：不放进 `rejected`——那条路会触发"全部被否 → 回退保留"，
            #   反而把空答案放回来。这里单独记身份集，最后统一排除。
            # ⚠ 关键：`\boxed{}` **不是**空字符串（`.strip()` 非空），但它是**空答案**，
            #   而这正是模型实际吐出的形态（实测 003 候选 3 的 `answer` 就是 `\boxed{}`）。
            #   ⇒ 必须**剥壳后**再判空，否则本守卫形同虚设（首版就是这样漏掉的）。
            try:
                from .answer_oracle import AnswerOracle as _AO
                _core = _AO.strip_wrappers(str(ans))
            except Exception:  # noqa: BLE001
                _core = str(ans)
            if not _core or not _core.strip():
                _empty.add(id(cand))
                self._record_ctx(ctx, {
                    "step": "candidate_audit", "tier": tier,
                    "cand_id": (cand.get("id", "?") if isinstance(cand, dict)
                                else getattr(cand, "id", "?")),
                    "answer": "", "verdict": "empty_answer",
                    "note": "空/纯空白答案：不参与投票，剔出候选池",
                })
                continue
            entry = {"step": "candidate_audit", "tier": tier,
                     "cand_id": (cand.get("id", "?") if isinstance(cand, dict)
                                 else getattr(cand, "id", "?")),
                     "answer": str(ans)[:60], "verdict": "unknown"}
            # Level 0：程序硬核验
            try:
                det = DeterministicChecker().check_answer(
                    ctx, ctx.problem or "", str(ans), domain or "")
                entry["deterministic"] = det.get("verdict")
                # 2026-10-07 留痕：记录本候选走的确定性判据通道
                #   （parse / parse_only / substitution / functional_substitution …）
                entry["deterministic_method"] = det.get("method")
                entry["deterministic_evidence"] = str(
                    det.get("evidence", ""))[:160]
            except Exception as e:  # noqa: BLE001
                det = {"verdict": "unknown", "evidence": f"异常: {str(e)[:80]}"}
                entry["deterministic"] = "unknown"
            if det.get("verdict") == "fail":
                n_fail += 1
                entry["verdict"] = "incorrect"
                rejected.append((cand, det))
            # Level 2：rubric 结构化判分（LLM，高置信 B 才采纳）。2026-10-07：
            #   条件由 `is_proof and use_rubric` 放宽为 `use_rubric`（后者已在上面把
            #   题型门纳入：解答题需显式开 `audit_gate_rubric_nonproof`）。
            elif use_rubric and not ctx.is_time_critical():
                try:
                    rub = self._verifier_inst._vote_one_rubric(
                        ctx, ctx.problem or "", str(ans))
                    entry["rubric"] = (rub or {}).get("verdict")
                    if rub and str(rub.get("verdict", "B")).upper() == "B":
                        try:
                            conf = float(rub.get("confidence", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            conf = 0.0
                        if conf >= float(getattr(
                                self.config, "audit_rubric_reject_conf", 0.85)):
                            n_fail += 1
                            entry["verdict"] = "incorrect"
                            entry["rubric_conf"] = conf
                            rejected.append((cand, rub))
                except Exception as e:  # noqa: BLE001
                    entry["rubric"] = "exception"
                    logger.debug("AuditGate: rubric 审核异常（放行）: %s", e)
            # ★ 2026-09-17 语义澄清：区分「检查了但无法判定」与「该路径对本题型不适用」。
            #   此前两者都写 `verdict="unknown"` ⇒ 读诊断的人会以为"审核跑了但判不了"，
            #   实际原因是 **Level 2 rubric 只对证明题运行**（见上方 `is_proof and
            #   use_rubric` 分支），而 Level 0 确定性检查对解答题极少判 fail。
            #   实测 003/013（均为解答题）：6/6 候选全 unknown ⇒ **结构性无输出**。
            #   这里**不改变 verdict 取值**（避免破坏下游按 verdict 统计的消费方），
            #   只补 `applicable` 与 `note` 两个说明性字段。
            # ★★ 2026-10-07：`applicable` 语义更新。旧定义 = "本闸门对该题型是否有判据"
            #   （故解答题=False）。现解答题已有确定性判据（归一 + 全量等式 + 函数方程），
            #   故 `sol_check=1` 时解答题也标 applicable=True；并**如实记录本轮用了哪条
            #   判据、结论是什么**（`entry["deterministic_method"]`），便于区分
            #   "跑了但判不了" 与 "根本没跑"。
            if entry.get("verdict") == "unknown":
                entry["applicable"] = bool(is_proof or sol_check)
                entry["deterministic_method"] = str(
                    entry.get("deterministic_method") or "")
                if is_proof:
                    entry["note"] = "证明题：Level2 rubric 已运行但未达否决阈值"
                elif sol_check:
                    entry["note"] = (
                        "解答题：确定性判据已运行（归一/等式回代/函数方程回代）"
                        "但未达否决阈值 ⇒ 本候选无客观否决")
                else:
                    entry["note"] = (
                        "解答题：Level2 rubric 不适用（仅对证明题运行），"
                        "且确定性判别重写已关 ⇒ 本闸门对该题型结构性无输出")
            self._record_ctx(ctx, entry)

        return self._finalize_candidate_audit(ctx, candidates, n_fail, rejected, _empty)

    def _finalize_candidate_audit(self, ctx, candidates, n_fail, rejected, _empty):
        """聚合候选审核结果：回退保留 / 淘汰留痕 / 空答案剔除（2026-10-08 从 audit_candidates 逐字搬出）。"""
        kept = list(candidates)
        feedbacks: list[str] = []
        # 全部被否 → 回退保留（宁 unknown 不误杀，绝不整批清空）
        if rejected and n_fail < len(candidates):
            kept = [c for c in candidates
                    if not any(c is rc for rc, _ in rejected)
                    and id(c) not in _empty]
            for rc, evidence in rejected:
                feedbacks.append(
                    f"[AuditGate 客观审核] 候选被程序硬核验否决："
                    f"{str(evidence.get('evidence', ''))[:120]}")
            self.record(ctx, "audit_gate",
                        f"审核淘汰 {len(rejected)}/{len(candidates)} 候选，"
                        f"保留 {len(kept)}")
        elif rejected:
            self.record(ctx, "audit_gate",
                        "全部候选被客观否决 → 回退保留（宁 unknown 不误杀）")
            kept = [c for c in candidates if id(c) not in _empty]
            feedbacks = []
        else:
            kept = [c for c in candidates if id(c) not in _empty]
            self.record(ctx, "audit_gate",
                        f"候选审核通过 {len(candidates)}（无程序否决）")
        # ★ 2026-09-16：空答案剔除（投票前）——记录并守住"绝不整批清空"的不变量。
        if _empty:
            self.record(ctx, "audit_gate",
                        "剔除 %d 个空/空白答案候选（不参与投票）" % len(_empty))
            if not kept:
                # 2026-09-20 修复：原实现无条件 `kept = list(candidates)`，
                # 会把**刚刚被 Level0 硬否决的候选**连同空答案一起复活。
                # kept 为空有两种成因：① 所有候选都是空的；② 其余候选全被否决
                # 且存在空答案（走上方 `n_fail < len(candidates)` 分支）。
                # 现在只恢复空答案候选，绝不把被程序否决的候选放回；
                # 只有在"一个空答案候选都没有"的退化情形才回退原列表，
                # 避免在此处制造空池。
                _empties = [c for c in candidates if id(c) in _empty]
                if _empties:
                    kept = _empties
                    self.record(ctx, "audit_gate",
                                "保留候选为空 → 仅恢复 %d 个空答案候选，"
                                "被程序否决的候选不放回，交由下游兜底处理"
                                % len(_empties))
                else:
                    kept = list(candidates)
                    self.record(ctx, "audit_gate",
                                "所有候选答案均为空 → 恢复原列表交由下游兜底处理")
        return kept, feedbacks

    # ------------------------------------------------------------------
    # 6.5 位：最终答案闸门（顶替原 lean_gate.gate_final_answer）
    # ------------------------------------------------------------------
    def gate_final_answer(self, ctx: TaskContext, tier: str,
                          answer: str, reasoning: str = "") -> bool:
        """对最终答案做 Level0-2 审核，返回 ``True`` 放行 / ``False`` 打回。

        - 空答案 → True（formatter 已兜底，交由上层处理）
        - Level0 deterministic fail / Level1 反例命中 → False（客观硬否决，
          6.5 rework 循环据此换候选/重做）
        - Level2 rubric B（高置信）→ False；其余 → True（宁 unknown 不误杀）
        - 任何异常 → True（绝不让审核缺陷阻断输出）
        """
        entry = {"step": "final_gate", "tier": tier,
                 "answer": str(answer)[:80], "verdict": "pass"}
        if not answer or not str(answer).strip():
            entry["degraded"] = "empty_answer"
            self._record_ctx(ctx, entry)
            return True
        domain = getattr(ctx, "domain", "") or ""
        qtype = getattr(ctx, "question_type", "") or ""
        is_proof = (domain in ("证明", "证明题") or qtype == "证明题")
        use_challenge = bool(getattr(self.config, "use_challenge", False))
        # ⚠ 2026-10-07 注意：`gate_final_answer` 的 rubric **本就无题型门**
        #   （原实现即 `if use_rubric ...`，对解答题也跑）；"仅对证明题运行"的限制
        #   只存在于 `audit_candidates`（见该处）。此处**保持不加新限制**（避免回归），
        #   故 `audit_gate_rubric_nonproof` 不影响本方法。
        use_rubric = bool(getattr(self.config, "use_rubric", False))
        sol_check = bool(getattr(self.config, "deterministic_solution_check", True))

        # Level 0：程序硬核验（0 LLM、可复现、客观）。2026-10-07：check_answer 内部
        #   已按 `deterministic_solution_check` 归一答案外壳 + 追加解答题判据。
        try:
            det = DeterministicChecker().check_answer(
                ctx, ctx.problem or "", str(answer), domain or "")
            entry["deterministic"] = det.get("verdict")
            entry["deterministic_method"] = det.get("method")
            entry["deterministic_evidence"] = str(det.get("evidence", ""))[:200]
            if det.get("verdict") == "fail":
                entry["verdict"] = "reject"
                entry["reason"] = str(det.get("evidence", ""))[:200]
                self._record_ctx(ctx, entry)
                return False
        except Exception as e:  # noqa: BLE001
            logger.debug("AuditGate: 最终答案确定性核验异常（放行）: %s", e)

        # Level 1：反例挑战（仅非数值答案；数值答案已由 Level0 代入覆盖）
        if use_challenge:
            chal = self._verifier_inst._challenge_counterexample(
                ctx, ctx.problem or "", reasoning or answer, str(answer))
            entry["challenge"] = chal
            if chal.get("hard_fail"):
                entry["verdict"] = "reject"
                entry["reason"] = str(chal.get("evidence", ""))[:200]
                self._record_ctx(ctx, entry)
                return False

        # Level 2：rubric 结构化判分（LLM 同源，高置信 B 才打回）
        if use_rubric and not ctx.is_time_critical():
            try:
                rub = self._verifier_inst._vote_one_rubric(
                    ctx, ctx.problem or "", reasoning or answer)
                entry["rubric"] = (rub or {}).get("verdict")
                if rub and str(rub.get("verdict", "B")).upper() == "B":
                    try:
                        conf = float(rub.get("confidence", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        conf = 0.0
                    if conf >= float(getattr(
                            self.config, "audit_rubric_reject_conf", 0.85)):
                        entry["verdict"] = "reject"
                        entry["rubric_conf"] = conf
                        reason = str(rub.get("reason", ""))[:200]
                        entry["reason"] = reason
                        step = rub.get("step_index")
                        err = rub.get("error_type", "")
                        detail = "；".join(
                            x for x in
                            ((f"步骤{step}" if step is not None else ""),
                             str(err) if err and err != "无" else "",
                             reason) if x)
                        entry["feedback"] = (detail or "rubric 判 B")[:300]
                        self._record_ctx(ctx, entry)
                        return False
            except Exception as e:  # noqa: BLE001
                logger.debug("AuditGate: 最终答案 rubric 审核异常（放行）: %s", e)

        entry["verdict"] = "pass"
        if is_proof and use_rubric and not ctx.is_time_critical():
            # 证明题：无程序否决、rubric 未高置信拒绝 → 放行（投票共识已把关）
            entry["note"] = "证明题：无客观否决，交由投票共识"
        elif sol_check:
            # 2026-10-07：解答题——如实记录确定性判据是否真的跑过、结论是什么，
            #   区分"跑了但未否决"（有判别力）与旧的"结构性无输出"。
            entry["applicable"] = True
            dm = str(entry.get("deterministic_method") or "none")
            entry["note"] = (
                f"解答题：确定性判据已运行（method={dm}）但未客观否决 ⇒ 放行")
        self._record_ctx(ctx, entry)
        return True
