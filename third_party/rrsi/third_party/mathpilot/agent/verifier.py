from __future__ import annotations
"""
过程校验智能体（VerifierAgent）
================================

功能演进：
  - v1: 二元投票（VERDICT A/B）+ 等价答案分组（未启用）
  - v2: 二元投票 + 评分模式 + 跨候选共识聚类 + 证明步骤验证

借鉴 ss-main 的投票共识与 Intern-MO 的 step 级验证，
但保持 MathPilot 多智能体 + 共享黑板的架构主线。
"""

import json
import logging
import re

from .base import BaseAgent, TaskContext, Verdict
from prompts.verifier import (
    VERIFIER_RUBRIC_SYSTEM,
    VERIFIER_RUBRIC_TEMPLATE,
    VERIFIER_CHALLENGE_SYSTEM,
    VERIFIER_CHALLENGE_TEMPLATE,
)
from prompts.proof import PROOF_VERIFY_SYSTEM, PROOF_VERIFY_TEMPLATE
from utils.extract import smart_fallback_answer
from utils.prefill import prefill_messages, stitch
# ★ 2026-10-09 拆分（S3/S4）：VerifierAgent 的职责簇已搬到 agent/verifier__*.py
# （cluster / lemma / report / vote / deep），类上只留**一行委托跳板** +
# 编排面 + 客观探测 + 4 个 patch 靶点。此处可放模块顶部（S3 已消掉
# verifier ↔ verifier__cluster 的模块级循环）。
from . import verifier__cluster as _vc
from . import verifier__lemma as _vl
from . import verifier__report as _vr
from . import verifier__vote as _vv
from . import verifier__deep as _vd

logger = logging.getLogger("MathPilot.Verifier")

# ---------------------------------------------------------------------------
# 辅助数据类
# ---------------------------------------------------------------------------

# ★ 2026-10-09 拆分（S3）：`AnswerCluster` 已逐字搬到 agent/verifier__cluster.py，
# 此处**原位 re-export**，保持外部 `from agent.verifier import AnswerCluster` 契约
# （tests/test_answer_cluster_confidence_locked.py:30、
#   tests/test_verifier_deep_review.py:208 依赖），并让本模块顶部可安全
# 反向 import verifier__cluster。⚠ 勿按"未使用"清理。
from .verifier__cluster import AnswerCluster  # noqa: F401


# ===========================================================================
# VerifierAgent
# ===========================================================================

# ★ 2026-10-08 拆分（批6）：验证器原始产出（判定解析 / 错误标签 / 复核文本清洗 /
# 样本落盘）的**纯函数簇**已搬至 agent/verifier__review.py；此处 re-export 保持
# 外部 `from agent.verifier import <name>`（orchestrator 用 `_clip_review_prose`；
# 测试用 `_last_verdict_ab` / `ERROR_TYPE_TAGS` / `_extract_error_type`）与内部
# 引用契约逐字不变。⚠ 勿按"未使用"清理。
from .verifier__review import (  # noqa: F401
    ERROR_TYPE_TAGS, _ERR_TAG_RE, _extract_error_type,
    _VERDICT_ANY_RE, _VERDICT_HEAD_RE, _last_verdict_ab,
    _DEEP_VERDICT_CONDENSE_SYS, _DEEP_VERDICT_PREFIX,
    _dump_deep_review_sample, _clip_review_prose,
    _REVIEW_PROSE_CUT, _REVIEW_VERDICT_CLAIM,
)


class VerifierAgent(BaseAgent):
    """
    过程校验智能体

    核心职责：
    1. 对每份候选解答进行多票独立验证（A/B 投票）
    2. 将等价答案分组（文本 + SymPy 符号归一化）
    3. 计算簇级置信度（簇内总正确票 / 总票数）
    4. 对证明题启用逐步骤验证
    5. 提取失败原因供自纠错回环使用
    """

    def __init__(self, client, config):
        super().__init__(client, config)

    # ==================================================================
    # 投票与解析
    # ==================================================================

    def _is_correct_vote(self, text: str) -> bool | None:
        return _vv._is_correct_vote(self, text)

    # ==================================================================
    # 答案归一化与等价判定
    # ==================================================================

    def _normalize_answer_text(self, text: str) -> str:
        return _vc._normalize_answer_text(self, text)

    def _are_answers_equivalent(self, a: str, b: str) -> bool:
        return _vc._are_answers_equivalent(self, a, b)

    def _equiv_group(
        self, candidates: list, answers: list[str]
    ) -> list[list[int]]:
        return _vc._equiv_group(self, candidates, answers)

    # ==================================================================
    # 共识聚类与簇置信度
    # ==================================================================

    def _cluster_candidates(
        self, candidates: list, verdicts: list[list[Verdict]]
    ) -> list[AnswerCluster]:
        return _vc._cluster_candidates(self, candidates, verdicts)

    # ==================================================================
    # 带推理的最终复核（2026-09-15 新增，回应「验证器偏松」）
    # ==================================================================
    # 实测依据（2026-09-15 的 10 题评测 + 历史记录）：
    #   ① 常规投票走 `prefill_messages(..., "VERDICT: ")` 强制**单行输出**，
    #      等于抑制思维流。代码注释自述实测 prefill 0.8s vs 普通 70.2s（140×）——
    #      **0.8 秒不足以完成系统提示词要求的"独立重算 + 逐条攻击"**，
    #      投票因此退化成"看一眼就点头"，与 6 步验证流程直接矛盾。
    #   ② 后果：10 题中 5 道**错题**的全部候选都是**全票 A**
    #      （086/091/099/101/103），叠加 AuditGate 候选审核 100% `unknown`、
    #      LeanGate 判 `valid`（086）或 `lenient_pass` 降级放行
    #      ⇒ **三道闸门对错答零否决**。
    # 本方法只对**最终选定的答案**做一次**不 prefill**的复核（让模型跑完推理），
    # 成本 1 次调用/题（约占单题 1100s 预算的 6%），是目前最省的"真验证"。
    # ⚠ 默认**开**（`user_agent.py:148` 实测 `verifier_deep_final_enabled: bool = True`）。
    #   原注释写「默认关（False）」与代码相反，2026-10-04 修正注释（不改行为）。
    # ==================================================================
    def _deep_final_review(self, ctx, problem: str, best_cluster) -> str:
        return _vd._deep_final_review(self, ctx, problem, best_cluster)

    def _deep_review_finalize(self, ctx, raw: str, verdict, _conv_text: str) -> str:
        return _vd._deep_review_finalize(self, ctx, raw, verdict, _conv_text)

    def _deep_llm(self, ctx, messages: list, temperature: float,
                  max_tokens: int) -> str:
        """复核专用 LLM 调用：**优先流式**，失败再退回普通调用。

        为什么不能直接用 `BaseAgent.llm`（非流式）：
        - 全局读超时 `LLM_TIMEOUT` 默认 **120s**（`utils/llm_client._DEFAULT_TIMEOUT`）；
        - 2026-09-15 实测：**"允许推理（不 prefill）"的两例调用均在 120s 处超时、
          输出 0 字**（而同样内容的 prefill 单行版只要 1.3–3.6s）。
        - 流式逐步累积直到服务端自然结束（`finish_reason=stop`），
          只要服务端还在吐 token 就不会被判读超时。
        ⇒ 非流式路径会让"要跑完整推理"的复核**必然失败**，功能等于不可用。
        """
        client = getattr(self, "client", None)
        chat = getattr(client, "chat", None)
        if ctx.budget is not None:
            try:
                ctx.budget.spend(1)
            except Exception:  # noqa: BLE001
                pass
        if callable(chat):
            try:
                out = chat(messages=messages, temperature=temperature,
                           max_tokens=max_tokens, stream=True)
                if out is not None:
                    return str(out)
            except TypeError:
                # 客户端不支持 stream 关键字 → 退回非流式（老实现/桩）
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning("复核流式调用失败，退回非流式: %s", str(e)[:140])
        return self.llm(ctx, messages, temperature, max_tokens) or ""

    # ==================================================================
    # LeanSearch 引理检索（方案 A：定理原文注入，2026-09-15）
    # ==================================================================
    # 目的：把验证流程第 5 步「方法 / 定理适用性核查」从**凭记忆复述定理**
    # 变成**对着检索到的定理原文**判断，直接回应李平老师提出的"硬套定理"。
    #
    # 为什么接在验证器而不是求解侧：
    #   ① 成本最低（**每题只检索一次**，不是每个候选/每张票都检索）；
    #   ② 收益可用现有判分器直接度量（判 B 的准确率变化）；
    #   ③ 实测首题出现"6 候选 × 3 票全部投 A 放行错答"⇒ 验证器才是当前的
    #      薄弱环节，给它原文比给求解器加料更对症。
    #
    # ⚠ 全程默认关（`use_leansearch=False`）⇒ 未做 A/B 前行为与改动前**完全一致**。
    # ⚠ 检索失败/超时/时间紧张 → 返回空串，验证器行为不变（绝不因检索拖垮主流程）。
    # ==================================================================
    def _prepare_theorem_context(self, ctx, problem: str) -> str:
        return _vl._prepare_theorem_context(self, ctx, problem)

    # ==================================================================
    # 投票执行
    # ==================================================================

    def _vote_one(self, ctx, problem: str, candidate_text: str,
                  temperature: float = 0.0, full_reasoning: bool = False) -> str:
        return _vv._vote_one(self, ctx, problem, candidate_text, temperature, full_reasoning)

    def _vote_one_scoring(self, ctx, problem: str, candidate_text: str) -> dict | None:
        return _vv._vote_one_scoring(self, ctx, problem, candidate_text)

    # ==================================================================
    # v3 P2（2026-09-06 移植自 sq 分支）：rubric 结构化判分 + 反例挑战
    # ==================================================================
    # sq 分支（平台 ~24 分）的检测资产：rubric 让投票从 A/B 二元升级为
    # verdict+confidence+error_type+step_index+reason（错因质量 → revise
    # 定向性）；反例挑战让"非数值答案"也能被程序化数值验证证伪。

    def _vote_one_rubric(self, ctx, problem: str, candidate_text: str,
                         temperature: float = 0.0) -> dict | None:
        """一次 rubric 结构化判分（JSON prefill，秒级返回）。

        输出 {"verdict","confidence","error_type","step_index","reason"}，
        借鉴 Intern-MO judge：独立重算 → 对比 → 定位错因。
        """
        messages = [
            {"role": "system", "content": VERIFIER_RUBRIC_SYSTEM},
            {"role": "user", "content": VERIFIER_RUBRIC_TEMPLATE.format(
                problem=problem, candidate_answer=candidate_text
            )},
        ]
        try:
            raw = self.llm(ctx, prefill_messages(messages, '{"'), temperature, 32768)
            if raw:
                raw = stitch('{"', raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("Rubric 判分调用失败: %s", e)
            return None
        rub = self._parse_json_loose(raw)
        if rub is None:
            logger.warning("Rubric 判分 JSON 解析失败: %s", (raw or "")[:120])
        return rub

    def _vote_rubric(self, ctx, problem: str, candidate,
                     use_deterministic: bool = True,
                     temperature: float = 0.0) -> list[Verdict]:
        return _vv._vote_rubric(self, ctx, problem, candidate, use_deterministic, temperature)

    def _challenge_counterexample(self, ctx, problem: str,
                                  candidate_text: str, answer: str) -> dict:
        """反例挑战（P2）：LLM 生成候选命题 → 程序数值验证才生效。

        只对非纯数值答案触发（数值答案已走确定性代入验证）。
        返回 {"hard_fail": bool, "evidence": str}。
        """
        if not answer or not str(answer).strip():
            return {"hard_fail": False, "evidence": "无答案可挑战"}
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(answer).strip()):
            return {"hard_fail": False, "evidence": "数值答案，跳过命题反例搜索"}
        messages = [
            {"role": "system", "content": VERIFIER_CHALLENGE_SYSTEM},
            {"role": "user", "content": VERIFIER_CHALLENGE_TEMPLATE.format(
                problem=problem, candidate_answer=candidate_text
            )},
        ]
        try:
            raw = self.llm(ctx, prefill_messages(messages, '{"'), 0.0, 32768)
            if raw:
                raw = stitch('{"', raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("反例挑战调用失败: %s", e)
            return {"hard_fail": False, "evidence": "调用异常"}
        parsed = self._parse_json_loose(raw)
        if not parsed or not parsed.get("found"):
            return {"hard_fail": False, "evidence": "模型未提出反例命题"}
        statement = str(parsed.get("statement", "")).strip()
        if not statement:
            return {"hard_fail": False, "evidence": "反例命题为空"}
        try:
            from .deterministic import DeterministicChecker
            checker = DeterministicChecker(attempts=300)
            res = checker.search_counterexample(statement)
        except Exception as e:  # noqa: BLE001
            logger.warning("反例搜索异常: %s", e)
            return {"hard_fail": False, "evidence": f"搜索异常: {str(e)[:80]}"}
        if res.get("found"):
            return {"hard_fail": True,
                    "evidence": f"反例验证成功: {statement} 在 {res.get('counterexample')} 处不成立"}
        return {"hard_fail": False, "evidence": f"反例搜索未找到 ({statement[:80]})"}

    # ==================================================================
    # 投票力度自适应（2026-09-16 新增，针对「零否决」）
    # ==================================================================
    # 实测依据（2026-09-15 的 10 题 + 直接实验）：
    #   · standard 档 `verifier_voting_times=1` ⇒ 每题**只有一次**分类判断；
    #   · 且 `_vote_one` 把温度**硬编码 0.0**（`config.verifier_temperature`
    #     因此是死配置）⇒ 多张票在温度 0 下几乎完全相同，**投票没有方差**，
    #     "多票"只是同一判断重复 N 次；
    #   · 后果：6 道错题中 5 道的**全部候选被全票判 A**（零否决）。
    # 对策：**候选答案之间存在分歧时**（题目本身有争议/易错），自动提高票数
    # 并启用非零温度，让投票产生真实方差；分歧为 0 时保持原样（加票无意义且费时）。
    # ==================================================================
    @staticmethod
    def _vote_reasoning_mode() -> str:
        """投票是否放开**完整推理**（不 prefill）。三档：``off`` | ``disagree`` | ``always``。

        ★ 2026-10-04（**B 折中**，用户决策「正确率第一位」）：
        · ``off``      = 旧行为（全部 prefill，0.8s/票；本文件自陈"看一眼就点头"）
        · ``disagree`` = **B 折中（当前默认）**：只在**候选分歧**时放开完整推理
                         ⇒ 把昂贵算力花在真拿不准的题上
        · ``always``   = **A**：每票都跑完整推理（最准；约 70s/票 ⇒ 18 票 ≈ 21 分钟/题）
        用户口径：先试 B；效果不理想即切 A。
        """
        try:
            from .switch_registry import get_str as _sws
            return str(_sws("verifier_vote_reasoning") or "disagree").strip().lower()
        except Exception:  # noqa: BLE001
            return "disagree"

    def _vote_profile(self, ctx, candidates, base_votes=None) -> tuple:
        return _vv._vote_profile(self, ctx, candidates, base_votes)

    def _vote(
        self, ctx, problem: str, candidate, total_votes: int = 5,
        proportional: bool = True, use_scoring: bool = False,
        temperature: float = 0.0, full_reasoning: bool = False,
    ) -> list[Verdict]:
        return _vv._vote(self, ctx, problem, candidate, total_votes, proportional, use_scoring, temperature, full_reasoning)

    # ------------------------------------------------------------------
    # 结构化错误类型聚合（2026-09-15）
    # ------------------------------------------------------------------
    # 背景：用户要求"之后测试记录大模型的具体答题情况，把错误暴露得更加具体"。
    # 此前 trace 里投票只留 correct/abstain/raw，**错因全埋在自然语言里**，
    # 无法统计 ⇒ 归因只能靠人工读日志（实测"推理错 62 题"就是这么数出来的）。
    # 现在 VERIFIER_SYSTEM 判 B 时必须回带**封闭标签集**（ERROR_TYPE_TAGS），
    # 这里把它们聚合成 {标签: 票数}，`orchestrator._collect_diag()` 可直接取用。
    # 纯埋点，不参与任何判定逻辑。
    # ------------------------------------------------------------------
    @staticmethod
    def _error_type_dist(verdicts) -> dict:
        return _vv._error_type_dist(verdicts)

    def _record_error_types(self, ctx, verdicts,
                            step: str = "vote_error_types") -> dict:
        return _vv._record_error_types(self, ctx, verdicts, step)

    # ==================================================================
    # 反馈提取（自纠错用）
    # ==================================================================

    def _extract_feedback(self, ctx, problem: str, candidate) -> str:
        return _vr._extract_feedback(self, ctx, problem, candidate)

    # ------------------------------------------------------------------
    # 结构化 Bug Report（依据 IMO 2025 验证-精炼流水线论文）
    # ------------------------------------------------------------------
    def _extract_bug_report(self, ctx, problem: str, candidate,
                            tool_findings: str = "") -> dict:
        return _vr._extract_bug_report(self, ctx, problem, candidate, tool_findings)

    @staticmethod
    def _parse_json_loose(raw: str):
        """从 LLM 输出里尽力抠出 JSON 对象（去围栏、平衡括号）。"""
        if not raw:
            return None
        import json as _json
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
        try:
            return _json.loads(text)
        except (_json.JSONDecodeError, ValueError):
            pass
        # 平衡括号：取第一个能完整解析的对象
        depth = 0
        start = -1
        in_str = esc = False
        for i, ch in enumerate(text):
            if esc:
                esc = False
            elif in_str and ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            return _json.loads(text[start:i + 1])
                        except (_json.JSONDecodeError, ValueError):
                            start = -1
        return None

    @staticmethod
    def _format_bug_report(report: dict) -> str:
        return _vr._format_bug_report(report)

    def structured_report_for(self, ctx, problem: str, candidate,
                              tool_findings: str = "") -> str:
        return _vr.structured_report_for(self, ctx, problem, candidate, tool_findings)

    def _extract_revise_feedback(self, ctx, problem: str,
                                 candidates: list, best_cluster) -> str:
        return _vr._extract_revise_feedback(self, ctx, problem, candidates, best_cluster)

    @staticmethod
    def _oracle_sanity_feedback(candidate) -> str:
        return _vr._oracle_sanity_feedback(candidate)

    # ==================================================================
    # P1-1: Python/SymPy 独立验证通道 + 确定性复算（playoff）
    # ==================================================================

    _PLAYOFF_SYS = (
        "你是数学解题专家。重新独立地解答下面这道题，"
        "只输出最终答案（数值、表达式或选项字母），不要任何推理过程。"
    )

    def _deterministic_check(self, ctx, problem: str, candidate) -> dict:
        """对候选答案做确定性旁证/否决（0 LLM 预算）。

        返回 DeterministicChecker.check_answer 的结果 dict：
        {"verdict": "pass"|"fail"|"unknown", "confidence": float,
         "evidence": str, "method": str}。
        任何异常一律降级 unknown（宁可 unknown 绝不误杀）。
        """
        answer = (candidate.get("answer", "") if isinstance(candidate, dict)
                  else getattr(candidate, "answer", ""))
        if not answer:
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": "答案为空", "method": "none"}
        try:
            from .deterministic import DeterministicChecker
            checker = DeterministicChecker()
            return checker.check_answer(ctx, problem, answer,
                                        getattr(ctx, "domain", "") or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("Deterministic check failed: %s", e)
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": f"异常: {str(e)[:80]}", "method": "exception"}

    def _playoff_recheck(self, ctx, problem: str, top_answer: str) -> bool:
        """确定性复算（playoff）：用 temperature=0 重新解一遍，比对答案。

        解决"验证器与解题器同源一起错"的问题：低温重解是独立采样，
        若两次独立求解答案一致，则置信度大幅提升。
        """
        try:
            # v2.4.1：playoff 也走 prefill——「答案：」让答案前置，抑制 CoT 推理流
            resp = self.llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": self._PLAYOFF_SYS},
                        {"role": "user", "content": problem},
                    ],
                    "答案：",
                ),
                0.0, 32768,
            )
            if resp:
                resp = stitch("答案：", resp)
            if not resp or not resp.strip():
                return False
            recheck_ans = smart_fallback_answer(resp)
            if not recheck_ans:
                return False
            return self._are_answers_equivalent(
                self._normalize_answer_text(top_answer),
                self._normalize_answer_text(recheck_ans),
            )
        except Exception as e:
            logger.warning("playoff recheck failed: %s", e)
            return False

    # ==================================================================
    # 证明步骤验证
    # ==================================================================

    def _verify_proof_step(self, ctx, problem: str, solution: str) -> dict | None:
        messages = [
            {"role": "system", "content": PROOF_VERIFY_SYSTEM},
            {"role": "user", "content": PROOF_VERIFY_TEMPLATE.format(
                problem=problem, solution=solution
            )},
        ]
        try:
            # v2.4.1：prefill「{"」引导 JSON 输出，抑制 CoT 前置长推理
            raw = self.llm(ctx, prefill_messages(messages, '{"'), 0.0, 32768)
            if raw:
                raw = stitch('{"', raw)
            m = re.search(r'\{[^{}"]*(?:"[^"]*"[^{}]*)*\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            # fallback: larger match
            m2 = re.search(r'\{[\s\S]*\}', raw)
            if m2:
                return json.loads(m2.group())
            return {"overall": "unknown", "raw": raw[:500]}
        except Exception as e:
            logger.warning(f"Proof step verify failed: {e}")
            return None

    # ==================================================================
    # 主流程
    # ==================================================================

    def run(
        self, ctx: TaskContext, problem: str, candidates: list,
        use_clustering: bool = True,
        use_scoring: bool = False,
        is_proof: bool = False,
        use_playoff: bool = False,
        voting_times: int = None,
        use_deterministic: bool = False,
        use_rubric: bool = False,
        use_challenge: bool = False,
    ) -> dict:
        """
        验证主流程。

        参数:
            voting_times: 每候选投票数；None 时回退 config.verifier_voting_times。
                          （难题深度通道：deep 档传 3，fast/standard 传 1）
            use_rubric: 2026-09-06（移植自 sq）：rubric 结构化判分路径
                        （verdict+confidence+error_type+step_index+reason，
                        内嵌确定性旁证：fail 硬否决 / pass 独立正确票）。
            use_challenge: 2026-09-06（移植自 sq）：反例挑战——对全错票的
                        非数值答案，LLM 生成命题 → 程序数值验证 → hard_fail 硬否决。

        返回:
            {
                "cluster_data": list[AnswerCluster],  # 候选传给 orchestrator
                "feedback": str,                       # 自纠错用
                "verdicts": list[list[Verdict]],       # 原始裁决
                "best_cluster": AnswerCluster | None,
            }
        """
        # 特殊处理证明题
        if is_proof and len(candidates) == 1:
            return self._run_proof_single(ctx, problem, candidates)

        all_verdicts: list[list[Verdict]] = self._run_collect_votes(
            ctx, problem, candidates, voting_times,
            use_rubric=use_rubric, use_scoring=use_scoring,
            use_deterministic=use_deterministic, use_challenge=use_challenge)

        # v2.8 确定性硬否决：SymPy 代入回验/反例对候选做客观旁证，
        # fail → 该候选全部票判错（淘汰）；unknown/pass → 仅挂证据不改判决。
        # 全部 fail 时回退保留（宁可 unknown 绝不误杀，镜像 LeanGate 降级逻辑）。
        # （use_rubric=True 时确定性已在 _vote_rubric 内处理，此处跳过避免重复）
        if use_deterministic and candidates and not use_rubric:
            det_results = [self._deterministic_check(ctx, problem, c) for c in candidates]
            n_fail = sum(1 for r in det_results if r.get("verdict") == "fail")
            if 0 < n_fail < len(candidates):
                for i, r in enumerate(det_results):
                    if r.get("verdict") == "fail":
                        for v in all_verdicts[i]:
                            v.correct = False
                            v.deterministic = r
                        all_verdicts[i].append(Verdict(
                            correct=False, raw="deterministic_fail", deterministic=r))
                        self.record(ctx, "deterministic",
                                    f"确定性硬否决候选 #{i}: {r.get('evidence', '')[:120]}")
                    else:
                        for v in all_verdicts[i]:
                            if v.deterministic is None:
                                v.deterministic = r
            else:
                for i, r in enumerate(det_results):
                    for v in all_verdicts[i]:
                        if v.deterministic is None:
                            v.deterministic = r
                if n_fail == len(candidates) and n_fail > 0:
                    self.record(ctx, "deterministic",
                                "全部候选确定性否决，回退保留（宁 unknown 不误杀）")

        # 聚类 + 共识
        cluster_data = self._cluster_candidates(candidates, all_verdicts) if use_clustering else []
        best_cluster = cluster_data[0] if cluster_data else None

        # P1-1 确定性复算（playoff）：共识不强或验证全错时，低温独立重解
        # P0-4 修复：默认关闭（use_playoff=False），仅在时间宽裕时由 orchestrator 开启，
        # 避免叠加调用链耗尽单题预算 → 45 error
        if use_playoff and best_cluster is not None and ctx.budget is not None:
            confidence = (best_cluster.vote_correct / best_cluster.vote_total
                          if best_cluster.vote_total else 0.0)
            # 触发条件：置信度低（验证结果不可靠）或投票全否
            # 2026-09-12 定型前审核：原为 `if confidence < 0.5 and True:`
            # （`and True` 是残留死条件，删除后语义完全不变）
            if confidence < 0.5:
                top_ans = best_cluster.answer_norm or ""
                if self._playoff_recheck(ctx, problem, top_ans):
                    best_cluster.vote_correct += 1
                    best_cluster.vote_total += 1
                    self.record(ctx, "playoff", "确定性复算通过，答案可信")
                else:
                    # 复算不一致 → 置信度下调，标记供 orchestrator 走 revise
                    best_cluster.vote_correct = 0
                    best_cluster.vote_total = max(1, best_cluster.vote_total)
                    self.record(ctx, "playoff", "确定性复算不一致，转自纠错")

        # ★ 2026-09-15：带推理的最终复核（默认关）。
        # 判 B ⇒ 把该簇置信度压到 <0.5，**走既有的低置信度 → revise 通道**
        # （不新增任何流程分支），并直接复用它的推理文本当反馈，
        # 省掉一次额外的错因提取调用。
        _deep_reject = False
        _deep_raw = self._deep_final_review(ctx, problem, best_cluster)
        if _deep_raw and _last_verdict_ab(_deep_raw) is False:
            best_cluster.vote_correct = 0
            best_cluster.vote_total = max(1, best_cluster.vote_total)
            # ★★ 2026-10-05 **回退我自己引入的崩溃**：
            #   我一度在此加过 `best_cluster.confidence = 0.0`，依据是误读了
            #   `agent/base.py:317`（那是**另一个类**的字段）。`AnswerCluster`
            #   （`agent/verifier.py:48`，`__slots__` + `@property`）的 `confidence`
            #   是**只读属性**，计算自 `vote_correct / vote_total`
            #   ⇒ 赋值抛 `AttributeError: property 'confidence' has no setter`
            #   ⇒ `Orchestrator run failed`，**整轮评测在 3 小时后崩掉、0 条结果**
            #   （2026-10-05 19:23 实测）。
            #   ⇒ **完全不需要这行**：`vote_correct` 清零后 confidence 自动变 0.0。
            _deep_reject = True

        feedback, error_types = self._run_build_feedback(
            ctx, problem, candidates, best_cluster, all_verdicts,
            _deep_reject, _deep_raw)

        return {
            "cluster_data": cluster_data,
            "feedback": feedback,
            "verdicts": all_verdicts,
            "best_cluster": best_cluster,
            "error_types": error_types,
        }

    def _run_build_feedback(self, ctx, problem, candidates, best_cluster,
                            all_verdicts, _deep_reject: bool, _deep_raw: str) -> tuple:
        """拼装自纠错 feedback 与全题错因分布 error_types。

        纯搬迁自 ``run`` 的反馈段（2026-10-08 批7 原地拆分），语句逐字不变。
        """
        # Reflexion 修复：恢复失败反馈提取。仅在 best_cluster 低置信度/全错时
        # 提取（受预算约束），让 revise 回环拿到真实错误定位，而非空串导致的
        # "请重新审题"泛泛提示。
        # ★★ 2026-10-05 修（**结构化错因已解析、却没接进反馈** —— 三步论第 2 步）：
        #   原顺序是"**先拼 feedback、后统计 error_types**"，于是
        #   `漏项×4 / 前提不成立×4 / 计算错×2` 明明已从投票原文正确解析出来，
        #   却**只写进 diag**，模型收到的永远是"未标注" ⇒ 反馈不可用、重写只能瞎猜。
        #   现改为**先统计、再拼 feedback**，并把分布写进反馈首行。
        # （2026-09-15 原注释：跨候选汇总错误类型分布，与 `_vote` 内的同名埋点互补 ——
        #   那里是**单次投票**口径，这里是**整题**口径，供 diag 统计。）
        error_types = self._record_error_types(
            ctx, [v for vds in all_verdicts for v in vds],
            step="verify_error_types")
        _et_line = ""
        try:
            if isinstance(error_types, dict) and error_types:
                _ordered = sorted(error_types.items(), key=lambda kv: -kv[1])
                _et_line = "、".join(
                    "%s×%d" % (k, v) for k, v in _ordered[:4] if k)
        except Exception:  # noqa: BLE001
            _et_line = ""

        if _deep_reject:
            # ★★ 2026-10-05 修（**结构化 bug report 被深复核分支整个绕过** ——
            #   用户判定"方法有问题"，这是最直接的证据）：
            #   `_extract_revise_feedback` 内部已经实现了论文要求的结构化报告
            #   （`_extract_bug_report` → `_format_bug_report`：
            #    **位置（引用原句）+ 类型 + 问题**，按「关键错误 / 论证漏洞」分级，
            #    开关 `use_bug_report_feedback` **默认 True**，docstring 直接引用
            #    Huang & Yang 2025），但它**只在 else 分支执行**。
            #   一旦深复核判 B（`_deep_reject=True`），反馈就被**替换**成一行
            #   `【带推理复核判错｜错误类型：…】` —— 实测只剩 **44 字符**（且是乱码）
            #   ⇒ **模型拿不到任何定位**，重写只能瞎猜（6 轮不收敛的直接原因）。
            #   现改为：**先取结构化报告，再附复核原文作为证据**（两者互相补充，
            #   不再互相替代）。结构化报告为空（如预算不足）时，才退回原来的单行标签。
            _struct = self._extract_revise_feedback(
                ctx, problem, candidates, best_cluster)
            # ★★ 2026-10-06：拼入前先**裁掉复核者的重做正文与越权结论**
            #   （实测：`正确答案应为 0。下面给出严谨证明。**证明：**…` 被模型整段照抄
            #   ⇒ 题 000 最终答 0、而标准答案 20460）。详见 `_clip_review_prose` 注释。
            _deep_clip = _clip_review_prose(_deep_raw)
            if _struct:
                feedback = (_struct
                            + "\n\n【复核依据（判错报告原文摘录）】\n"
                            + _deep_clip)[:8000]
            else:
                _et = _extract_error_type(_deep_raw) or _et_line or "未标注"
                feedback = ("【带推理复核判错｜错误类型：%s】%s" % (
                    _et, _deep_clip))[:6000]
        else:
            feedback = self._extract_revise_feedback(
                ctx, problem, candidates, best_cluster)
        # 无论走哪条分支，都把**全题错因分布**放在最前面 ——
        # 它是模型判断"我这类错在这题犯了多少次"的最直接依据。
        if _et_line:
            feedback = ("【全题错因分布（各票判定）】%s\n" % _et_line) + feedback
        return feedback, error_types

    def _run_collect_votes(self, ctx, problem, candidates, voting_times,
                           use_rubric: bool, use_scoring: bool,
                           use_deterministic: bool, use_challenge: bool) -> list:
        """逐候选投票（rubric/scoring 路径 + 反例挑战），返回 all_verdicts。

        纯搬迁自 ``run`` 的投票段（2026-10-08 批7 原地拆分），语句逐字不变。
        """
        # 常规：每个候选投票（voting_times 参数化：难题深度通道 deep 档 3 票）
        # v2.8：ctx.state.voting_times 优先（RunState 运行时覆盖），config 兜底
        if voting_times is None:
            voting_times = (getattr(ctx.state, 'voting_times', None)
                            or getattr(self.config, 'verifier_voting_times', 1))
        all_verdicts: list[list[Verdict]] = []
        # ★ 2026-09-16：按候选分歧自适应票数与温度（见 _vote_profile）。
        #   **必须把档位票数传进去**——否则 deep 档的 3 票会被压回 1 票（自审发现的回归）。
        _d_votes, _d_temp, _d_reason = self._vote_profile(
            ctx, candidates, base_votes=voting_times)
        voting_times = _d_votes
        for i, cand in enumerate(candidates):
            if use_rubric:
                # rubric 结构化判分路径（含确定性旁证：fail 硬否决 / pass 独立票）
                vds = self._vote_rubric(ctx, problem, cand,
                                        use_deterministic=use_deterministic,
                                        temperature=_d_temp)
            else:
                vds = self._vote(ctx, problem, cand, total_votes=voting_times,
                                 use_scoring=use_scoring, temperature=_d_temp,
                                 full_reasoning=_d_reason)
            # 反例挑战（P2，sq 语义）：仅"该候选无任何正确票"且非数值答案时触发，
            # LLM 生成命题 → 程序数值验证 → hard_fail 硬否决（客观证伪）。
            # 2026-09-11：若该候选**全是弃权票**（判分链路故障、无任何判定信息），
            # 不再触发挑战——限流风暴中这会逐候选追加 LLM 调用、自激放大（详见
            # Bug清单 §5 的 176 次限流秒级连发）。有真实判定信息时才值得挑战。
            _has_real_vote = any(not getattr(v, "abstain", False) for v in vds)
            if (use_challenge and vds and _has_real_vote
                    and not any(v.correct for v in vds)):
                ans = (cand.get("answer", "") if isinstance(cand, dict)
                       else getattr(cand, "answer", ""))
                chal = self._challenge_counterexample(
                    ctx, problem, self._candidate_text(cand), str(ans))
                if chal.get("hard_fail"):
                    for v in vds:
                        v.correct = False
                        v.confidence = 0.0
                    vds.append(Verdict(correct=False, raw="counterexample_fail",
                                       feedback=f"反例挑战否决: {chal.get('evidence', '')[:120]}"))
                self.record(ctx, "challenge",
                            f"候选#{i} 反例挑战: {chal['evidence'][:100]}")
            all_verdicts.append(vds)
        return all_verdicts

    def _run_proof_single(self, ctx, problem, candidates) -> dict:
        """证明题单候选快速通道：逐步骤验证 → 单簇结果字典。

        纯搬迁自 ``run`` 的 ``if is_proof and len(candidates) == 1:`` 分支
        （2026-10-08 批7 原地拆分），语句逐字不变（缩进 12→8）。
        """
        # 逐步骤验证
        proof_text = self._candidate_text(candidates[0])
        step_result = self._verify_proof_step(ctx, problem, proof_text)
        overall_correct = (step_result.get("overall") == "proof_valid"
                           if step_result else False)
        v = Verdict(correct=overall_correct, raw=json.dumps(step_result or {}))
        cluster = AnswerCluster("proof")
        # 2026-09-12 定型前审核修复：候选是 `Candidate` dataclass（无 `.get`），
        # 原写法 `candidates[0].get("id", 0)` 在证明题单候选通道会抛
        # AttributeError（被外层吞掉 → 该通道静默失效）。
        cluster.candidate_ids = [getattr(candidates[0], "id", 0)]
        cluster.vote_correct = 1 if overall_correct else 0
        cluster.vote_total = 1
        # 2026-09-12 定型前审核修复：`step_verdicts` 键存在但为**空列表**时，
        # 原来的 `[0]` 会 IndexError（同样被外层吞掉 → 静默失效）。
        _sv = (step_result or {}).get("step_verdicts") or [{}]
        feedback = (_sv[0].get("note", "")
                    if step_result and not overall_correct else "")
        return {
            "cluster_data": [cluster],
            "feedback": feedback,
            "verdicts": [[v]],
            "best_cluster": cluster,
        }

    # ==================================================================
    # 辅助
    # ==================================================================

    def _candidate_text(self, candidate) -> str:
        if isinstance(candidate, dict):
            parts = []
            if candidate.get("reasoning"):
                parts.append(candidate["reasoning"])
            if candidate.get("answer"):
                parts.append(f"【最终答案】{candidate['answer']}")
            return "\n".join(parts)
        reasoning = getattr(candidate, "reasoning", "")
        answer = getattr(candidate, "answer", "")
        return f"{reasoning}\n【最终答案】{answer}" if reasoning and answer else str(candidate)
