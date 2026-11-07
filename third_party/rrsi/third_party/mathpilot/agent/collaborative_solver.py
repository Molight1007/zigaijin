from __future__ import annotations
"""难题三Agent协作求解器（CollaborativeSolver，v2.6）。

针对 deep 档难题，三个角色分工协作（协作链路为串行依赖；平台并发度=3 由
Orchestrator/main 的信号量约束，本模块单题内不额外开线程）：

  1. 解题 Agent  —— 完整求解，给出初步解答；
  2. 审查 Agent  —— 审查解答，定位并指出错误；
  3. 整合 Agent  —— 综合解题输出与审查意见，给出最终答案；
  4. 验证 Agent  —— 对整合结果做正确性判定（VERDICT: A/B）。

反复验证循环：只要时间未到、预算未耗尽、且验证仍未通过，
就回到「审查 → 整合 → 验证」，用上一轮的审查/验证反馈驱动修正，
直到验证通过或资源耗尽，保证难题高正确率。

超时约束：每道题最长处理时间 20 分钟由 Orchestrator 的 ``ctx.deadline``
（config.max_time_per_question=1200s）硬限；所有 LLM 调用都经
``BaseAgent.llm`` 的时间预算感知（剩余<60s 跳过），保证超时题目有兜底产出。
"""

import logging
import re

# 2026-10-07：候选入池门槛 v2 开关改为自 base 引用（去重，单一来源）。
from .base import (
    BaseAgent, TaskContext, Candidate, next_candidate_id, _pool_gate_v2_on)
from utils.extract import (
    extract_final_answer,
    rescue_final_answer,
    smart_fallback_answer,
)
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot.Collaborative")


# ---------------------------------------------------------------------------
# 三角色提示词
# ---------------------------------------------------------------------------
_SOLVER_SYS = (
    "你是数学解题专家（协作链路中的【解题Agent】）。"
    "请对下面这道难题给出完整、严谨的求解过程，逐步推理并给出最终答案。"
    "不要省略关键步骤，最终以【最终答案】给出明确结论。"
)

_REVIEWER_SYS = (
    "你是数学证明审查专家（协作链路中的【审查Agent】）。"
    "请严格审查下面的解题过程，找出其中的错误、漏洞或跳步："
    "逐条列出「步骤定位 → 问题描述 → 修改建议」。"
    "若解答正确，也请明确说明其推理无误。不要重新解题，只做审查。"
)

_INTEGRATOR_SYS = (
    "你是数学解题总负责人（协作链路中的【整合Agent】）。"
    "请综合【解题Agent】的输出与【审查Agent】的意见，"
    "修正错误、补全漏洞，给出最终的正确解答，并以【最终答案】给出明确结论。"
)

_VERIFIER_SYS = (
    "你是一名极其严格的数学审稿人。你必须**首先假设给出的解答是【错误的】**，"
    "除非你能严格证明它正确。请带着这个假设去主动寻找：具体反例、计算或符号错误、"
    "逻辑漏洞、隐藏假设或跳步。只有经过严谨核查确实【无法找到任何错误】，"
    "才输出 VERDICT: A（确认正确）；一旦发现任何问题，输出 VERDICT: B（简述错误类型）。"
)


# VERDICT 之后（只允许标点/空白/markdown 装饰，最多 8 个）紧跟的单个 A 或 B。
# 负向先行断言 `(?![A-Z])` 排除 ABOVE / AB 这类词首；
# `[^\w\u4e00-\u9fff]` 明确排除中文与字母数字 ⇒ 中文解释里的 A/B 不会被误认成裁决。
_VERDICT_RE = re.compile(r"VERDICT[^\w\u4e00-\u9fff]{0,8}([AB])(?![A-Z])")


def _parse_verdict(text: str) -> bool:
    """解析验证结果：VERDICT: A → True（通过），B → False（未通过）。

    ★ 2026-09-21 修复（原判据过宽，实测可误判为「通过」）：
      原末条 `("VERDICT" in upper and "A" in upper)` 是**全文包含**测试，
      而 `stitch()` 保留 prefill 前缀 ⇒ 文本**恒含 VERDICT**
      ⇒ 等价于「只要有任意大写 A 就判通过」。
      实测反例：`"VERDICT: 该解答有 AB 两处问题"` 原判据返回 True（通过）。
      现改为「VERDICT 之后紧跟 A/B」，与提示词要求的输出格式一致。
      影响：判据变严 ⇒ 未按格式输出的答复按「未通过」处理（保守方向：继续协作迭代）。
    """
    if not text:
        return False
    upper = text.upper()
    if "INCORRECT" in upper or "WRONG" in upper or "FALSE" in upper:
        return False
    m = _VERDICT_RE.search(upper)
    if m:
        return m.group(1) == "A"
    # 无显式 VERDICT 标记时的保守回退：仅"明确出现 CORRECT"才判通过。
    return "CORRECT" in upper


# prefill 种子前缀（stitch 会**保留**它 ⇒ 各角色返回值必须剥前缀后再判有效性）
_PF_SOLVE = "## 解题过程\n"
_PF_REVIEW = "## 审查意见\n"
_PF_INTEGRATE = "## 最终解答\n"


def _has_content(text: str, prefix: str) -> bool:
    """stitch 结果里是否有**前缀之外的实际内容**。

    ★ 2026-09-21 修复（与 agent/classifier.py 的 prefill 前缀污染同族）：
      `stitch()` 的契约是**保留** prefill 前缀（见 utils/prefill.py 第 50 行）。
      模型只回一个空格时，`stitch(prefix, " ")` 得到 `prefix + " "` ——
      **非空但零内容**，会骗过调用方的 `if not solution` / `if not final` 守卫，
      下游 `extract_final_answer` 抽到空 ⇒ 产出空答案候选
      （甚至让前缀串本身经兜底函数变成候选答案）。
    """
    t = str(text or "")
    if not t:
        return False
    if t.startswith(prefix):
        t = t[len(prefix):]
    return bool(t.strip())


class CollaborativeSolver(BaseAgent):
    """难题三Agent协作：解题 → 审查 → 整合 → 验证（反复循环）。"""

    name = "CollaborativeSolver"

    def run(self, ctx: TaskContext) -> TaskContext:
        # 2026-09-06：时间判断升级 ctx.gen_time_up()（生成侧软截止，
        # 未设时回退 is_time_critical，行为不变）——collab 每轮含
        # 解题/审查/整合/验证多次 LLM，单轮可达数百秒，必须更早停手。
        if ctx.gen_time_up() or ctx.is_timed_out():
            self.record(ctx, "collab", "时间紧张/超时，跳过三Agent协作")
            return ctx

        # 1) 解题 Agent（一次性）
        solution = self._role_solve(ctx)
        if not solution:
            self.record(ctx, "collab", "解题Agent未产出有效结果，协作终止")
            return ctx

        review = ""
        final = solution
        max_rounds = getattr(self.config, 'collab_max_rounds', 4)
        # 停滞检测（2026-08-29）：连续 2 轮答案无变化 → 确认"不会了"，
        # 提前放弃协作，避免硬耗到时间结束（用户要求：确定不会才跳下一题）。
        last_answer = ""
        stagnant = 0

        for rnd in range(1, max_rounds + 1):
            # 时间/预算耗尽 → 用当前 best 兜底返回（2026-09-06 升级 gen_time_up）
            if ctx.gen_time_up() or ctx.is_timed_out():
                self.record(ctx, "collab", f"第{rnd}轮前时间紧张，停止协作循环")
                break

            # 2) 审查 Agent：首轮审 solution，后续审上一轮 final
            if rnd == 1:
                review = self._role_review(ctx, solution)
            else:
                review = self._role_review(ctx, final)

            # 3) 整合 Agent：综合解题输出 + 审查意见 +（后续轮）上一轮结果
            # ★ 2026-09-21 修复（先覆盖后校验 ⇒ 丢掉好答案）：
            #   原实现直接 `final = self._role_integrate(...)`，整合 Agent 返回空串时
            #   `final` **已被覆盖成空**，紧随的 `if not final: break` 把
            #   **解题 Agent 的有效解答一并丢弃**，末尾只能抽出空答案
            #   ⇒ 向候选池追加空候选（且损失一个可能正确的解答）。
            #   改为「先取临时变量、校验通过才赋值」，失败时保留既有 final。
            if rnd == 1:
                _cand = self._role_integrate(ctx, solution, review)
            else:
                _cand = self._role_integrate_round(ctx, solution, review, final)

            if not _cand:
                self.record(ctx, "collab",
                            f"第{rnd}轮整合Agent未产出有效结果（保留上一轮结果，不丢解答）")
                break
            final = _cand

            # 4) 验证 Agent：判定是否正确
            ok = self._role_verify(ctx, final)
            if ok:
                self.record(ctx, "collab", f"第{rnd}轮验证通过，协作成功")
                break
            self.record(ctx, "collab", f"第{rnd}轮验证未通过，继续审查修正")

            # 停滞检测：答案连续 2 轮不变 → 协作无进展，提前放弃
            cur_answer = extract_final_answer(final)
            if not cur_answer or len(cur_answer) > 300:
                cur_answer = smart_fallback_answer(final)
            if cur_answer and cur_answer == last_answer:
                stagnant += 1
                if stagnant >= 2:
                    self.record(ctx, "collab",
                                f"第{rnd}轮答案连续 {stagnant} 轮无变化，"
                                f"确认无进展，提前放弃协作")
                    break
            else:
                last_answer = cur_answer
                stagnant = 0

        answer = extract_final_answer(final)
        if not answer or len(answer) > 300:
            answer = rescue_final_answer(final)[0] or smart_fallback_answer(final)
        # ★★ 2026-10-07（2.4 v2 入口穷举 · 入口⑥）：**协作候选此前不过入池门槛**
        #   —— 三 Agent 协作是一条**独立写入 `ctx.candidates` 的路径**（与 solver /
        #   sub_goal 并列）。其答案同样由 extract/rescue/smart_fallback 三条抢救
        #   函数抽出，可能吐回推理句/元话语。此处与主路径同口径：不合格 ⇒
        #   **不入池并留痕**（不再向下游池注入需二次过滤的"假答案"）。
        if _pool_gate_v2_on():
            try:
                from utils.extract import normalize_pool_answer as _npa
                _ans2, _ok2 = _npa(answer)
            except Exception:  # noqa: BLE001  校验不可用时放行，不阻断
                _ans2, _ok2 = answer, True
            if not _ok2:
                self.record(
                    ctx, "candidate_invalid",
                    "协作候选 **不含最终答案 → 不入池**（answer 前 60 字：%r）"
                    % (str(answer or "")[:60],))
                return ctx
            answer = _ans2
        candidate = Candidate(
            id=next_candidate_id(ctx.candidates),
            answer=answer or "",
            reasoning=final,
            revised=False,
        )
        ctx.candidates.append(candidate)
        self.record(
            ctx, "collab",
            f"三Agent协作结束，生成候选 #{candidate.id}",
            answer=(answer or "")[:100],
        )
        return ctx

    # ------------------------------------------------------------------
    # 角色调用（均走 prefill 抑制 CoT + 预算/时间感知）
    # ------------------------------------------------------------------
    def _role_solve(self, ctx: TaskContext) -> str:
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _SOLVER_SYS},
                    {"role": "user", "content": ctx.problem},
                ],
                "## 解题过程\n",
            ),
            0.3,
            self.config.policy_max_tokens,
        )
        out = stitch(_PF_SOLVE, resp) if resp else ""
        return out if _has_content(out, _PF_SOLVE) else ""

    def _role_review(self, ctx: TaskContext, target: str) -> str:
        user = f"题目：\n{ctx.problem}\n\n待审查的解答：\n{target[-4000:]}"
        # v2.7：注入 oracle 客观错误信息（Lean Finding / 计算校验），让审查
        # Agent 有据可依地挑错，而非泛泛审查。
        oracle_ctx = self._collect_oracle_context(ctx, target)
        if oracle_ctx:
            user += f"\n\n[客观验证依据，请据此重点核查]\n{oracle_ctx}"
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _REVIEWER_SYS},
                    {"role": "user", "content": user},
                ],
                "## 审查意见\n",
            ),
            0.1,
            # ★ 2026-10-02：4096→8192。实测该处曾撞 finish_reason=length（DeepSeek 长审查意见
            #   被腰斩）；与其余 13 处统一抬到 8192（DeepSeek 适配方案 A）。
            8192,
        )
        out = stitch(_PF_REVIEW, resp) if resp else ""
        return out if _has_content(out, _PF_REVIEW) else ""

    @staticmethod
    def _collect_oracle_context(ctx: TaskContext, target: str) -> str:
        """收集客观验证依据（供审查 Agent 重点核查）。"""
        notes = []
        # 1) 上层已产生的客观反馈（AuditGate / Oracle 复核 / 对抗检出）
        audit_fb = getattr(ctx, 'audit_reject_feedback', None) or []
        for fb in audit_fb[:3]:
            notes.append(f"- {fb}")
        # 2) 对 target 答案的轻量客观 sanity check（SymPy 可解析性）
        try:
            from .answer_oracle import AnswerOracle
            ans = extract_final_answer(target)
            if ans and not AnswerOracle.is_parseable(ans):
                notes.append("- 该解答的最终答案无法解析为有效数学表达式，请重点核查")
        except Exception as exc:  # noqa: BLE001  解析检查异常吞掉 = 静默跳过
            # 2026-09-04 审核：与 lean_gate 吞 AttributeError 同型——
            # is_parseable 内部 bug 会让"解析检查"系统性失效。留证据。
            logger.debug("collab sanity 解析检查异常（跳过）: %s: %s",
                         type(exc).__name__, exc)
        return "\n".join(notes) if notes else ""

    def _role_integrate(self, ctx: TaskContext, solution: str, review: str) -> str:
        user = (
            f"题目：\n{ctx.problem}\n\n"
            f"【解题Agent】输出：\n{solution[-4000:]}\n\n"
            f"【审查Agent】意见：\n{review[-2000:]}"
        )
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _INTEGRATOR_SYS},
                    {"role": "user", "content": user},
                ],
                "## 最终解答\n",
            ),
            0.1,
            self.config.policy_max_tokens,
        )
        out = stitch(_PF_INTEGRATE, resp) if resp else ""
        return out if _has_content(out, _PF_INTEGRATE) else ""

    def _role_integrate_round(self, ctx: TaskContext, solution: str,
                              review: str, prev: str) -> str:
        """后续轮整合：综合原始解题 + 最新审查意见 + 上一轮结果。"""
        user = (
            f"题目：\n{ctx.problem}\n\n"
            f"【解题Agent】输出：\n{solution[-3000:]}\n\n"
            f"【上一轮解答】：\n{prev[-2000:]}\n\n"
            f"【本轮审查Agent】意见：\n{review[-2000:]}"
        )
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _INTEGRATOR_SYS},
                    {"role": "user", "content": user},
                ],
                "## 最终解答\n",
            ),
            0.1,
            self.config.policy_max_tokens,
        )
        out = stitch(_PF_INTEGRATE, resp) if resp else ""
        return out if _has_content(out, _PF_INTEGRATE) else ""

    def _role_verify(self, ctx: TaskContext, final: str) -> bool:
        answer = extract_final_answer(final)
        user = f"题目：\n{ctx.problem}\n\n待验证解答：\n{final[-3000:]}"
        if answer:
            user += f"\n\n候选最终答案：{answer}"
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _VERIFIER_SYS},
                    {"role": "user", "content": user},
                ],
                "VERDICT: ",
            ),
            0.0,
            # 2026-10-02 DeepSeek 适配：原 512 ⇒ 8192。原值基于「只输出一行 VERDICT +
            # prefill 抑制思维块」的 Intern-S 时代假设，对 reasoning 模型必然截断
            # （reasoning 先吃满预算、正文为空）。
            8192,
        )
        text = stitch("VERDICT: ", resp) if resp else ""
        return _parse_verdict(text)
