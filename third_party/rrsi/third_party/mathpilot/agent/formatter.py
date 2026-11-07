from __future__ import annotations

"""
答案规范化智能体（FormatterAgent）
==================================

把原 ``ReasoningAgent`` 末尾的答案抽取 / 兜底逻辑迁移为独立 Agent：
- 从 ``ctx.verdicts``（按置信度最高）或 ``ctx.candidates`` 选择最优答案；
- 通过 ``format_response`` 确保 ``final_response`` 非空且可序列化；
- 结果写入 ``ctx.final_response``，Orchestrator 负责封装返回字典。

BUG 修复：
  - 绝不输出"无法求解"等拒绝语（评测判 0）
  - _pick_best 启用共识加权（等价答案簇 → 更大簇更可信）
  - 增加 _is_valid_final_answer 终检
"""

import logging
import re

from .base import BaseAgent, TaskContext
from utils.extract import format_response

# 2026-10-08 拆分批：以下符号已整段迁至独立模块，此处**原位 re-export**
# 以保外部 import 契约（orchestrator / final_selector / 多处测试）。
from .formatter__nonanswer import _looks_like_non_answer  # noqa: F401
from .formatter__rank import (  # noqa: F401
    _answer_form_score, _answer_support, _attach_support,
    _passes_confidence_floor, _rank_key, _support_of, _is_real_enumeration,
)

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
# ★ 2026-10-07（fix-tail-gaps）：本文件两处选择题判据原为裸
#   `_os.environ.get("OBJECTIVE_*", ...) == "1"`，改走本注册制实时读
#   （默认值与注册表逐条核对一致，见 switch_registry.SWITCHES）。
from agent.switch_registry import get_bool as _sw_bool

logger = logging.getLogger("MathPilot")

# ★ 2026-10-15（用户定案「删直答兜底」）：`_final_answer_min_sec` /
#   `_emergency_answer_max_tokens`（紧急直答的参数配置）已随 `_emergency_answer`
#   删除 —— 无消费者。原为「调用时实时读 env/注册表」的开关，现整段移除。

# 拒绝回答/不完整答案的模式（2026-09-02 加"子目标求解失败"占位符：
# 占位符直接当最终答案 = 50% 错题（009/053/004/022 等），必须触发换候选兜底）
# 2026-09-13 晚补：新增 solver 的「生成失败」占位符族。实测 4 题里 010/016 的
# `predicted` 就是 `[生成失败] 调用受限或模型拒绝回答` —— 它从 solver 的占位
# 候选流到这里时**不被旧正则识别**，于是被当成合法答案直接提交（formatter:115）。
# `生成失败`/`调用受限`/`拒绝回答`/`未给出有效解答`/`无法作答` 一律视为无效答案，
# 触发 `_pick_fallback` → 换候选 → 最终 `_emergency_answer` 直答兜底。
_REFUSAL_RE = re.compile(
    r"无法求解|无法解决|不能解决|无法解答|我无法|暂无|无解|子目标求解失败|"
    r"生成失败|调用受限|拒绝回答|未给出有效解答|无法作答",
    re.IGNORECASE,
)

# 2026-09-13 晚：「答案缺失/占位」判据（**窄口径**，用于决定"要不要推翻现有答案
# 重新求一次"）。与上面的 `_REFUSAL_RE` 的区别是**刻意不含** `无解` / `暂无` ——
# 它们是合法答案（「该方程无解」就是正确答案），用它们触发 `_emergency_answer`
# 会让直答**覆盖正确解**（独立验证者实测：answer='无解' 被覆盖成 '7'）。
# `_REFUSAL_RE` 的宽松口径仅用于"换个候选试试"（第 89 行），语义不同故分开。
# ⚠ 本常量与 `agent/orchestrator.py::_DEGRADED_ANSWER_RE` **必须保持同一口径**
#   （2026-10-15：user_agent 那份已随降级兜底删除，现为两处）。
_MISSING_ANSWER_RE = re.compile(
    r"生成失败|调用受限|拒绝回答|未给出有效解答|无法作答|子目标求解失败|"
    r"我无法|无法求解|无法解决|不能解决|无法解答",
    re.IGNORECASE,
)

# 明显截断/不完整的 LaTeX 环境或元语句
_INCOMPLETE_RE = re.compile(
    r"\\begin\{[^}]*\}\s*$|\\begin\{[^}]*\}\s*\\end\{[^}]*\}\s*$|"
    r"通解为\s*：\s*$|最终答案\s*：\s*$|答案为\s*$|选\s*$|"
    r"不过.{0,30}$|然而.{0,30}$|但是.{0,30}$|这可能不是",
    re.IGNORECASE,
)


# ★ 2026-09-16 新增：**伪枚举**识别（枚举优先分支专用）。
# 背景：`enum_preferred` 原先只判"含逗号即枚举"，于是把 `0,1,2,3,\ldots`
# 这类**省略号糊弄式答案**当成合法枚举，还因为"推理最长"而胜出
# （实测 003：conf=0.667 的正确候选 `\boxed{2026}` 输给了 conf=0.000 的它）。
# 真正的枚举答案必须**逐项列出具体值**，出现省略号即说明未写全 ⇒ 排除。
_PSEUDO_ENUM_RE = re.compile(
    r"\\ldots|\\cdots|\\dots|\.\.\.|…|⋯|"
    r"等等|其余|以此类推|依此类推|以下省略|"
    r"such\s+that|and\s+so\s+on",
    re.IGNORECASE,
)


class FormatterAgent(BaseAgent):
    name = "Formatter"

    def run(self, ctx: TaskContext) -> TaskContext:
        answer, confidence = self._select_answer(ctx)

        # ★ 2026-09-17（P1-D）：**多答案题并集**（仅在题面明确要求「所有 / 全部」时启用）。
        _uni = self._maybe_union_enumerated(ctx, answer)
        if _uni:
            try:
                ctx._pick_diag = {
                    "branch": "enum_union",
                    "picked": _uni[:80],
                    "base": (answer or "")[:40],
                }
            except Exception:  # noqa: BLE001
                pass
            self.record(ctx, "format",
                        "多答案并集（P1-D）：%s → %s" % ((answer or "")[:30], _uni[:60]))
            answer = _uni

        self._record_divergent_clusters(ctx)

        answer, confidence = self._repair_and_fallback(ctx, answer, confidence)

        # 答案质量终检 + 自动修复
        answer = self._diagnose_and_repair(answer, ctx)

        # 2026-09-02 老师需求：最终答案截断必解决。
        # 003 题答案 g(x)=-2x^{ 就是生成截断直接提交 → expr_wrong。
        # 修复：检出截断 → 用 LLM 续写补全（最多 2 次），仍失败才原样返回。
        answer = self._repair_truncated(ctx, answer)

        # ★ 2026-10-15（用户定案「删直答兜底」）：原 `_try_emergency`（占位符→
        #   紧急直答）已删。答案为空/占位符时不再向 LLM 直答，如实暴露由评测判 0；
        #   仅记录一条错误信息供归因。
        _ans_txt = (answer or "").strip()
        if (not _ans_txt) or _looks_like_non_answer(_ans_txt) \
                or _MISSING_ANSWER_RE.search(_ans_txt):
            self.record(ctx, "solve_error",
                        "终答不可用（空/占位符/非答案形态）：如实返回，不再直答")

        self._write_final_response(ctx, answer)

        self._sync_pick_diag(ctx)

        self.record(
            ctx, "finalize",
            f"最终答案: {ctx.final_response[:200]} (置信度: {confidence:.2f})",
            confidence=round(confidence, 4),
        )
        return ctx

    def _select_answer(self, ctx: TaskContext) -> tuple:
        # ★ 2026-10-15（用户定案「删直答兜底」）：原「0 票直答兜底」分支已删 ——
        #   `ctx._zero_vote_fallback` 的写入点随 5_revise/直答链删除，此分支恒死；
        #   直答 preset（`ctx.final_response` 由直答预设）的来源也不复存在。
        #   现在直接 `_pick_best` 选答。
        best = self._pick_best(ctx)
        if best is None:
            # BUG-1 修复：绝不输出"无法求解"，也绝不把原题当答案。
            if ctx.candidates:
                best = max(ctx.candidates, key=_rank_key)
                # 2026-09-20 修复：删掉 `len(best.answer) > 2` 闸门。
                # 单字符是合法且完整的答案，仅空/纯空白才回退推理尾部。
                answer = best.answer if (best.answer or "").strip() else (
                    (best.reasoning or "")[-500:])
            else:
                answer = ""
            confidence = 0.0
        else:
            answer = getattr(best, "answer", "") or ""
            # 2026-09-12 定型前审核修复：单字符是**合法且完整**的答案，
            # 只在空（或纯空白）时才回退推理尾部。
            if not answer.strip():
                answer = (getattr(best, "reasoning", "") or "")[-500:]
            confidence = getattr(best, "confidence", 0.0)
            # 如果来自聚类路径，优先使用簇的置信度（更可靠：基于多票共识）
            best_cluster = getattr(ctx, '_best_cluster', None)
            if best_cluster and getattr(best_cluster, 'confidence', 0.0) > 0:
                confidence = best_cluster.confidence

        # 2026-09-17：兜底埋点 —— 保证 `_pick_diag` 有值，答案来源可追溯。
        if not getattr(ctx, "_pick_diag", None):
            try:
                ctx._pick_diag = {
                    "branch": "formatter_pick_best",
                    "picked": (answer or "")[:60],
                }
            except Exception:  # noqa: BLE001
                pass
        return answer, confidence

    def _record_divergent_clusters(self, ctx: TaskContext) -> None:
        # ★ 2026-09-17（P1-C）：把**分歧候选**（各不同答案簇的代表）记入 `_pick_diag`。
        # 用户决策「候选不必要强制不一致，但有不一致的思路一定要保留」——
        # 此处**只保留可见性**：不制造多样性、不改选答逻辑。
        # `ctx._cluster_data` 此前只有写入、**全项目零读取点** ⇒ 分歧信息一直在手边却
        # 从未被使用（025 的 diag 只能看到 1 个簇正是因此）。
        try:
            _cd_all = getattr(ctx, "_cluster_data", None)
            if isinstance(_cd_all, list) and _cd_all:
                _reps = []
                for _cl in _cd_all:
                    _an = str(getattr(_cl, "answer_norm", "") or "").strip()
                    if not _an:
                        continue
                    _reps.append({
                        "answer": _an[:60],
                        "size": int(getattr(_cl, "size", 0) or 0),
                        "confidence": round(
                            float(getattr(_cl, "confidence", 0.0) or 0.0), 3),
                        "correct_votes": int(getattr(_cl, "vote_correct", 0) or 0),
                        "total_votes": int(getattr(_cl, "vote_total", 0) or 0),
                    })
                if _reps and isinstance(getattr(ctx, "_pick_diag", None), dict):
                    ctx._pick_diag["divergent_clusters"] = _reps[:8]
        except Exception:  # noqa: BLE001
            pass

    def _repair_and_fallback(self, ctx: TaskContext, answer: str, confidence: float) -> tuple:
        # 答案过长检测：如果答案超过300字符，尝试从推理尾部重新提取
        if answer and len(answer) > 300:
            for c in (ctx.candidates or []):
                if c.reasoning and getattr(c, 'answer', '') == answer:
                    from utils.extract import extract_final_answer
                    retry = extract_final_answer(c.reasoning)
                    if retry and len(retry) < len(answer) and len(retry) > 1:
                        self.record(ctx, "finalize",
                                   f"长答案重提取: {len(answer)}→{len(retry)} 字符")
                        answer = retry
                        confidence = max(confidence, 0.5)  # 重提取成功，给默认置信度
                        break

        # 如果最佳答案是拒绝类 / 明显不完整 / 结构上不像答案，尝试从其他候选找更好答案
        # （2026-09-17：并入 `_looks_like_non_answer`，覆盖 `</tool_call>` 等脏答案）
        if (not answer or _looks_like_non_answer(answer)
                or _REFUSAL_RE.search(answer) or _INCOMPLETE_RE.search(answer)):
            fallback = self._pick_fallback(ctx, exclude_answer=answer)
            if fallback:
                answer = fallback
                self.record(ctx, "finalize",
                           "最佳候选答案为拒绝/不完整，改用候选兜底答案")
                confidence = 0.0
        return answer, confidence

    # ★ 2026-10-15（用户定案「删直答兜底」）：`_try_emergency`（占位符/空答案 →
    #   紧急直答 `_emergency_answer`）已删除 —— 答案不可用时如实暴露，不再直答。
    #   （原逻辑见 git 历史；调用点 run() 已改为记录错误信息。）

    def _write_final_response(self, ctx: TaskContext, answer: str) -> None:
        # ★ 2026-09-18（审核发现）：**这是 `ctx.final_response` 的第 4 个直写点**。
        # 另外 3 个（零票兜底直答 / 6.5 重做 / 6.5 换候选）已改用
        # `Orchestrator._set_final_response`（带非答案闸门），本行此前是漏网的。
        # 本模块自带 `_looks_like_non_answer`，直接复用，避免跨模块反向 import。
        # ★ 2026-10-02 阶段二-2（team-lead Q4-② 裁定）：**此处只做规范化（format_response
        #   + 非答案闸门），不选答** —— 终答由「五层」在更早的出口决定（`_pick_best` /
        #   `_set_final_response` / 紧急直答的 `adopt_single`）。**不是旁路，勿再改。**
        _fr_txt = format_response(answer)
        if str(_fr_txt or "").strip() and not _looks_like_non_answer(_fr_txt):
            ctx.final_response = _fr_txt
        else:
            self.record(ctx, "finalize",
                        "最终答案被判为非答案形态 → 不覆盖 ctx.final_response，"
                        "交上层最终兜底：%s" % str(_fr_txt)[:80])

    def _sync_pick_diag(self, ctx: TaskContext) -> None:
        # ★★ 2026-09-21 修复（云端 n=38 逐题复核）：把 `_pick_diag["picked"]`
        #   同步为**实际落盘值**。
        #
        #   缺陷：`_pick_diag` 在本函数**前半段**就已写入（见上方零票兜底分支
        #   与 `formatter_pick_best` 兜底分支），但其后
        #   `_pick_fallback` / `_diagnose_and_repair` / `_repair_truncated` /
        #   `_emergency_answer` **仍会改写 `answer`** ⇒ 埋点记下的是**被丢弃的
        #   中间值**。实测云端快照 20/38 题 `pick_diag.picked` ≠ `predicted`，
        #   其中 8 题终答是散文 —— 使「是选答错了还是终答被覆盖」这一最关键
        #   归因**无法回答**（极易把埋点错位误读成"答案被后写覆盖"）。
        #
        #   修法：此处以 `ctx.final_response`（真正要提交的字符串）为准，
        #   原值移入 `picked_pre_repair` 保留追溯性。此改动**只影响埋点**，
        #   不改变任何选答/修复行为（`_pick_diag` 全项目无功能性读取点，
        #   仅由 `Orchestrator._collect_diag` 落盘进 diag）。
        #   OFF ⇒ 逐字保持改动前（`picked = _picked_final[:60]`）做 A/B。
        _ss_on = False
        try:
            from .final_selector import _single_source_enabled
            _ss_on = bool(_single_source_enabled())
        except Exception:  # noqa: BLE001
            _ss_on = False
        if isinstance(getattr(ctx, "_pick_diag", None), dict):
            _picked_final = str(ctx.final_response or "")
            if _picked_final:
                if _picked_final != ctx._pick_diag.get("picked"):
                    ctx._pick_diag["picked_pre_repair"] = ctx._pick_diag.get("picked")
                if _ss_on:
                    ctx._pick_diag.setdefault(
                        "picked_decision", ctx._pick_diag.get("picked"))
                    ctx._pick_diag["picked"] = _picked_final
                    ctx._pick_diag["final_submit"] = _picked_final
                    ctx._pick_diag["final_submit_source"] = "formatter"
                    ctx._pick_diag["final_submit_seq"] = int(
                        ctx._pick_diag.get("final_submit_seq", 0) or 0) + 1
                elif _picked_final != ctx._pick_diag.get("picked"):
                    ctx._pick_diag["picked"] = _picked_final[:60]
        # ★ 2026-10-07（问题 2.2）：把 `picked` 归入哪个 cluster 写进记录
        #   （未归入则写明原因）—— 回应「picked 必须能追溯到某个 cluster」。
        try:
            from .final_selector import record_pick_cluster, _verify_aware_enabled
            if _verify_aware_enabled():
                record_pick_cluster(ctx)
        except Exception:  # noqa: BLE001
            pass

    # 2026-10-08 拆分批3-b：终答选择/多答案并集族 → agent/formatter__select.py（逐字搬，见该模块）。
    def _maybe_union_enumerated(self, ctx: TaskContext, current: str):
        from .formatter__select import _maybe_union_enumerated as _impl
        return _impl(self, ctx, current)

    def _pick_best(self, ctx: TaskContext):
        from .formatter__select import _pick_best as _impl
        return _impl(self, ctx)

    def _maybe_final_select(self, ctx: TaskContext):
        from .formatter__select import _maybe_final_select as _impl
        return _impl(self, ctx)

    def _pick_best_raw(self, ctx: TaskContext):
        """
        选择最优答案（BUG-13 修复：共识加权）。
        优先使用聚类结果中置信度最高且规模最大的簇；其次使用传统 verdict 置信度。
        """
        # ★ 2026-10-02 阶段二-2：终答五层选择优先（开关 ON 时此处即返回，
        #   ⇒ 下方的 `_rank_key` 票数优先 / `objective_majority_vote` 全程不执行）。
        _fs = self._maybe_final_select(ctx)
        if _fs is not None:
            return _fs

        self._pick_best_prepare(ctx)

        _r = self._pick_best_itemwise(ctx)
        if _r is not None:
            return _r

        _r = self._pick_best_objective(ctx)
        if _r is not None:
            return _r

        _r = self._pick_best_enum(ctx)
        if _r is not None:
            return _r

        _r = self._pick_best_cluster(ctx)
        if _r is not None:
            return _r

        # 1) 传统 verdict 置信度
        if ctx.verdicts:
            valid = [v for v in ctx.verdicts if v.total_votes > 0]
            if valid:
                return max(valid, key=lambda v: v.confidence)
            return max(ctx.verdicts, key=lambda v: v.confidence)

        # 2) 候选兜底
        if ctx.candidates:
            for c in ctx.candidates:
                if c.answer:
                    return c
            return ctx.candidates[0]
        return None

    def _pick_best_prepare(self, ctx: TaskContext) -> None:
        # ★★ 2026-09-30（截图 #9）：**择优前先算「独立复现数」**。
        #   用户口径：「候选数量与逻辑通顺度都要是指标」。
        #   本步骤把"池里有几个不同候选写出了同一结论"挂到每个候选的 `_support_n`，
        #   于是下面所有走 `_rank_key` 的排序都变成**五键**（票数→复现数→置信度→
        #   答案形态→紧凑度），"候选数量"由此真正成为择优依据。
        #   必须放在**所有择优分支之前**（含被证伪候选剔除之前会更好，但剔除只减
        #   候选、不产生虚假复现数，故放此处即可）。
        #   纯加性：异常/空池一律退化为 `_support_n=1`，与改动前逐字一致。
        try:
            _sup_map = _attach_support(getattr(ctx, "candidates", None))
            if _sup_map:
                # 埋点：让诊断报告能看出"本次终答背后有几个独立候选复现"
                setattr(ctx, "_support_map", _sup_map)
        except Exception:  # noqa: BLE001
            pass
        # ★ 2026-09-23 新增：**择优时跳过「已被客观证伪」的候选**。
        #   动机：零票兜底已改为"证伪优先"（orchestrator），被证伪的候选不得再参与择优，
        #   否则证伪等于白做。判据仍是"错误答案一定能被证明错误" —— 只剔除
        #   **拿到 SymPy 确认反例**的候选，任何不确定的候选照旧参与。
        #   实现纪律：仅在"过滤后仍有候选"且"确实剔除了东西"时才替换，
        #   否则原样走（零风险回退，不改变既有行为）。
        _fal = getattr(ctx, "_falsified_answers", None)
        if _fal and (getattr(ctx, "candidates", None) or []):
            try:
                from .answer_oracle import AnswerOracle as _AO
                _all = list(ctx.candidates)
                _keep = [c for c in _all
                         if not any(_AO.answers_equivalent(
                             getattr(c, "answer", "") or "", _x) for _x in _fal)]
                if _keep and len(_keep) < len(_all):
                    self.record(ctx, "finalize",
                                "择优前剔除 %d 个已被客观证伪的候选（剩 %d/%d）"
                                % (len(_all) - len(_keep), len(_keep), len(_all)))
                    ctx.candidates = _keep
            except Exception:  # noqa: BLE001
                pass

    def _pick_best_itemwise(self, ctx: TaskContext):
        # 2026-09-13：选择题答案多数投票（OBJECTIVE_MAJORITY_VOTE=1 时）。
        # 必须放在 best_cluster 分支**之前**：best_cluster 是 verifier 聚类，
        # 且簇内再按"推理长度"选（下见 #0），对客观题（答案形态稳定、候选
        # 易因裸字母 `A` 与 `\boxed{A}` 的格式差异被聚类拆散 / 簇内择优失真）
        # 直接做答案频次投票更可靠——这正是 102 候选 4B/2A 却被选 A 的根因。
        # 2026-09-14：选择题**优先采纳「逐项判定」候选**（origin="itemwise"）。
        # 依据：它是**逐个选项验证过**的结论（有共享基准、有逐项依据），可靠性
        # 高于"整体求解"候选。实测 #107 逐项判定正确得出 D（gold=D），却因候选池
        # 里有 5 个整体求解的 C，被 5:1 多数投票淹没 ⇒ 必须让**有依据的结论**
        # 优先于**数量多数**。置 OBJECTIVE_ITEMWISE_PRIORITY=0 可回退。
        # ★ 2026-10-07（fix-tail-gaps）：原裸读 env `== "1"` 已改走注册制
        #   `_sw_bool("objective_itemwise_priority")`（实时读，默认 True）。
        #   三态核对（未设/"1"/"0"）与原 `== "1"` 逐条一致；仅非标准值
        #   （如 "true"/"yes"）由 registry 口径判为「开」，属已文档化的统一修正。
        if (_sw_bool("objective_itemwise_priority")
                and getattr(ctx, "question_type", "") == "选择题"):
            _iw = [c for c in (getattr(ctx, "candidates", None) or [])
                   if getattr(c, "origin", "") == "itemwise"
                   and (getattr(c, "answer", "") or "").strip()]
            if _iw:
                # ★ 2026-09-22：原为 `key=len(reasoning)`（只看啰嗦程度）。改为统一
                #   `_rank_key`，保留"逐项判定优先"的设计意图，但簇内择优不再抖。
                _iw.sort(key=_rank_key, reverse=True)
                try:
                    ctx._pick_diag = {
                        "branch": "formatter_itemwise_priority",
                        "picked": (_iw[0].answer or "")[:40],
                        "n_itemwise": len(_iw),
                        "n_total": len(getattr(ctx, "candidates", None) or []),
                    }
                except Exception:  # noqa: BLE001
                    pass
                self.record(ctx, "finalize",
                            "选择题优先采纳逐项判定候选：{}"
                            "（逐项 {} 个 / 共 {} 个候选）".format(
                                (_iw[0].answer or "")[:40], len(_iw),
                                len(getattr(ctx, "candidates", None) or [])))
                return _iw[0]
        return None

    def _pick_best_objective(self, ctx: TaskContext):
        # ★ 2026-10-07（fix-tail-gaps）：原裸读 env `== "1"` 已改走注册制
        #   `_sw_bool("objective_majority_vote")`（实时读，默认 False）。
        #   与 orchestrator.py 同名开关判据（`_sw_bool("objective_majority_vote")`）保持同源。
        if (_sw_bool("objective_majority_vote")
                and getattr(ctx, "question_type", "") == "选择题"):
            _mv: dict = {}
            for _c in (ctx.candidates or []):
                _a = (getattr(_c, "answer", "") or "").strip()
                # ★ 2026-09-16 审计修复：去掉 `len(_a) > 3` 门槛。本文件 95-98 行
                #   已论证"单字符是**合法且完整**的答案"，且选择题多为 `A`/`AB`/`BCD`
                #   ⇒ 长度门槛会把多数投票的候选答案整条筛掉（客观题必判错）。
                if _a and not _REFUSAL_RE.search(_a):
                    _mv[_a] = _mv.get(_a, 0) + 1
            if _mv:
                # 2026-09-13 用户设计：**平票时做差分检测**——
                # 当多种选项组合并列最高频（如 AB×2 与 ABC×2），用集合差分
                # 定位真正的争议选项（此处 = C），只对争议项做定向验证；
                # 非平票则退回多数投票（取最高频组合）。
                _groups = self._group_option_sets(_mv)
                _fixed = self._objective_diff_probe(ctx, _groups)
                if _fixed:
                    from .base import Candidate as _Cand
                    return _Cand(id=-1, answer=_fixed,
                                 reasoning="[差分检测修正]")
                _top_ans, _top_n = max(_mv.items(), key=lambda kv: kv[1])
                # ⚠ P3 修复（2026-09-14 代码审查）：差分检测失败**且平票**时，
                # 上面的 `max()` 取的是 dict 插入序里第一个候选 ⇒ 等于**随机**。
                # 改为**取交集**：并列组合共同包含的选项 = 无争议的共识部分，
                # 是"争议项无法判定"时唯一有依据的保守选择。
                # （实测：AD×2 vs ACD×2 ⇒ **AD**，即剔除有争议的 C；
                #   A×3 vs B×3 完全对立、交集为空 ⇒ 只能退回最高票）
                _max_v = max(_mv.values())
                _tops_ans = [k for k, v in _mv.items() if v == _max_v]
                if len(_tops_ans) > 1:
                    try:
                        _sets = [set(re.findall(r"[A-E]", _t))
                                 for _t in _tops_ans]
                        _inter = set.intersection(*_sets) if _sets else set()
                        if _inter:
                            _ans_i = "".join(sorted(_inter))
                            try:
                                ctx._pick_diag = {
                                    "branch": "formatter_tie_intersection",
                                    "tops": ["".join(sorted(x)) for x in _sets],
                                    "intersection": _ans_i,
                                    "picked": _ans_i,
                                }
                            except Exception:  # noqa: BLE001
                                pass
                            self.record(ctx, "finalize",
                                        "选择题平票且差分失败 → 取交集 {} ⇒ {}".format(
                                            ["".join(sorted(x)) for x in _sets],
                                            _ans_i))
                            from .base import Candidate as _Cand2
                            return _Cand2(id=-1, answer=_ans_i,
                                          reasoning="[平票取交集]")
                    except Exception:  # noqa: BLE001
                        pass
                # 2026-09-13：埋点写入 ctx._pick_diag（_collect_diag 会落盘），
                # 与 orchestrator 兜底路径共用同一字段，保证"选取来源"可追溯。
                try:
                    _pd = {
                        "branch": "formatter_majority_vote",
                        "picked": _top_ans[:40],
                        "top_n": _top_n,
                        "dist": {k[:40]: v for k, v in _mv.items()},
                        "groups": {"".join(sorted(k)): g["votes"]
                                   for k, g in _groups.items()},
                    }
                    # 保留差分检测的失败原因（否则被本埋点覆盖，事后无法定位
                    # "为什么平票却没走差分检测"）
                    _prev = getattr(ctx, "_pick_diag", None)
                    if isinstance(_prev, dict) and str(
                            _prev.get("branch", "")).startswith("diff_probe"):
                        _pd["diff_probe_failed"] = _prev
                    ctx._pick_diag = _pd
                except Exception:  # noqa: BLE001
                    pass
                for _c in (ctx.candidates or []):
                    if (getattr(_c, "answer", "") or "").strip() == _top_ans:
                        return _c
        return None

    def _pick_best_enum(self, ctx: TaskContext):
        # ---- 2026-09-14 枚举优先（题面要求『所有』时）----
        # 实测 003：merge 已正确产出 `\boxed{0,2026}`，但 verdicts **三条全部失效**
        # （LLM 超时 —— 超时阈值被收到 120s 且 max_retries=1），答案选取随之退化，
        # **把正确的枚举丢掉了**，最终只剩 `\boxed{2026}`。
        # 依据：题面明确要求"所有/全部"时，**枚举形态的候选天然优于单值候选**
        # （单值必然不满足题意）。故在常规选答之前先做一次"枚举优先"。
        #
        # ★★★ 2026-09-16 修复（用户直接质问"为什么有正确答案却选了错的"）：
        #   原实现有**三重缺陷**，导致它反而成了错误来源。实测 003：
        #     候选1 `\boxed{2026}` 票 2/3 conf=0.667（**2026 是正确答案之一**）
        #     候选5 `\boxed{0,1,2,3,\ldots}` 票 0/3 conf=0.000（**省略号糊弄式伪答案**）
        #     本分支却选中了后者。
        #   ① "枚举"判据只是**含逗号** ⇒ 伪答案也算枚举；
        #   ② 排序**只看推理长度**，完全不看票数/置信度 ⇒ 0 票的长文本胜过 2 票；
        #   ③ 直接 `return` ⇒ **绕过后面全部正常选答逻辑**（聚类/置信度/多数票）。
        #   修法：加"伪枚举"过滤 + 排序改为 (票数, 置信度, 推理长度) 词典序。
        #   保留原设计意图：枚举形态确实优于单值（单值必然不满足"求所有"）。
        try:
            from .question_type import asks_all_values as _aav2
            if _aav2(ctx.problem or ""):
                _enum_c = []
                for _c in (ctx.candidates or []):
                    _a2 = (getattr(_c, "answer", "") or "").strip()
                    if not _a2 or _REFUSAL_RE.search(_a2):
                        continue
                    _core2 = _a2
                    _mb2 = re.search(r"\\boxed\{([^{}]*)\}", _a2)
                    if _mb2:
                        _core2 = _mb2.group(1)
                    if re.search(r"[,，;；、]", _core2):
                        # ① 伪枚举过滤：省略号 / "等等" / 未写全的形式（如
                        #    `0,1,2,3,\ldots`）**不是**合法枚举答案，必须排除。
                        if _PSEUDO_ENUM_RE.search(_core2):
                            continue
                        # ② ★ 2026-09-22：再加"真枚举"判据 —— 仅含逗号不等于枚举，
                        #    散文（014 的 0 票候选）曾被此分支优先返回。
                        if not _is_real_enumeration(_core2):
                            continue
                        _enum_c.append(_c)
                if _enum_c:
                    # ★ 2026-09-30（截图 #9）：统一改用 `_rank_key`。
                    #   原为**局部实现**的 (票数, 置信度, 推理长度) ——
                    #   ① 漏掉新增的「独立复现数」（候选数量指标在此分支失效）；
                    #   ② 末位仍是「推理长度」（本文件 222 行已论证它与正确性
                    #      无因果关系，且是同一条被修过三次的老毛病）。
                    #   改后与全项目择优口径**单一来源**，不再有第二套排序。
                    _enum_c.sort(key=_rank_key, reverse=True)
                    _top_enum = _enum_c[0]
                    try:
                        ctx._pick_diag = {
                            "branch": "enum_preferred",
                            "picked": (getattr(_top_enum, "answer", "") or "")[:60],
                            "n_enum_candidates": len(_enum_c),
                            "support_n": int(getattr(_top_enum, "_support_n", 1) or 1),
                            "enum_votes": int(getattr(_top_enum, "correct_votes", 0) or 0),
                        }
                    except Exception:  # noqa: BLE001
                        pass
                    self.record(ctx, "finalize",
                                "题面要求『所有』→ 枚举形态候选优先"
                                "（{} 个**合法**枚举候选，按统一 _rank_key：票数→"
                                "独立复现数→置信度→答案形态 排序）".format(
                                    len(_enum_c)))
                    return _top_enum
        except Exception:  # noqa: BLE001
            pass
        return None

    def _pick_best_cluster(self, ctx: TaskContext):
        # 0) 聚类数据（来自 verifier）→ 找最佳簇中第一个候选
        best_cluster = getattr(ctx, '_best_cluster', None)
        if best_cluster:
            cid_set = set(getattr(best_cluster, 'candidate_ids', []))
            # 2026-09-20 修复（口径错配）：candidate_ids 里存的是**候选在列表中的
            # 下标**，不是 Candidate.id。三处证据：建簇处（verifier.py）对
            # Candidate 对象取的是 idx；两个消费侧 verifier.py:1207 与
            # orchestrator.py:2729 也都写成 `cids[0] < len(candidates)` 后取
            # `candidates[idx]`。原实现用 `c.id in cid_set` 匹配 ⇒ 候选列表一旦
            # 发生位移（3.6 段过滤 / 候选池封顶 6 / 6.5 段换候选），matching 为空、
            # 整条簇分支被静默跳过，下方"按 (票数, 置信度) 择优"的修复失效。
            matching = [c for _i, c in enumerate(ctx.candidates or [])
                        if _i in cid_set]
            if matching:
                # ★ 2026-09-17（M5）：簇内改为 **(票数, 置信度) 优先，最后才比推理长度**。
                # 原实现只按"推理最长"选，与正确性脱钩 —— 本文件 418-431 已自述
                # 003 曾因此把已正确的 `\boxed{0,2026}` 丢掉、最终只剩 `\boxed{2026}`。
                # ★ 2026-09-30（截图 #9）：再统一到 `_rank_key`。原局部键的**末位是
                #   `len(reasoning)`**（M5 时保留了这一项），仍与正确性无因果；
                #   且它读的是候选对象上的 `correct_votes`/`confidence`，而本分支的
                #   票数来源是 `ctx.verdicts`（按 id 索引）—— 两者可能不同源。
                #   现改为：先把 verdict 的票数回填到候选对象（`_rank_key` 读的是
                #   对象属性），再统一排序 ⇒ 全项目择优口径单一来源。
                _vm = {}
                for _v in (getattr(ctx, "verdicts", None) or []):
                    try:
                        _vm[int(getattr(_v, "id", -1))] = (
                            int(getattr(_v, "correct_votes", 0) or 0),
                            float(getattr(_v, "confidence", 0.0) or 0.0))
                    except Exception:  # noqa: BLE001
                        continue
                for _c in matching:
                    _hit = _vm.get(int(getattr(_c, "id", -1)))
                    if _hit is not None:
                        try:
                            _c.correct_votes = int(_hit[0])
                            _c.confidence = float(_hit[1])
                        except Exception:  # noqa: BLE001
                            pass
                matching.sort(key=_rank_key, reverse=True)
                return matching[0]
        return None

    # 2026-10-08 拆分批3：簇④选择题差分族 → agent/formatter__diff.py（逐字搬，见该模块）。
    @staticmethod
    def _group_option_sets(mv: dict) -> dict:
        from .formatter__diff import _group_option_sets as _impl
        return _impl(mv)

    def _objective_diff_probe(self, ctx: TaskContext, groups: dict) -> str:
        from .formatter__diff import _objective_diff_probe as _impl
        return _impl(self, ctx, groups)

    # 2026-10-08 拆分批3：簇⑤兜底/续写/紧急直答族 → agent/formatter__fallback.py（逐字搬，见该模块）。
    def _pick_fallback(self, ctx: TaskContext, exclude_answer: str = "") -> str:
        from .formatter__fallback import _pick_fallback as _impl
        return _impl(self, ctx, exclude_answer)

    def _repair_truncated(self, ctx: TaskContext, answer: str) -> str:
        from .formatter__fallback import _repair_truncated as _impl
        return _impl(self, ctx, answer)

    # ★ 2026-10-15：`_emergency_answer`（紧急直答）已随直答兜底删除。

    @staticmethod
    def _diagnose_and_repair(answer: str, ctx: TaskContext) -> str:
        from .formatter__fallback import _diagnose_and_repair as _impl
        return _impl(answer, ctx)
