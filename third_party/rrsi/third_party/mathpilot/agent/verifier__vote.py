# -*- coding: utf-8 -*-
"""投票执行（从 agent/verifier.py 逐字搬出）。

S2（2026-10-09）：``_vote_one`` / ``_vote_profile`` / ``_vote`` 搬为模块级
函数，原 ``self`` 作为第一个参数 ``owner`` 传入；所有需要走实例级
``mock.patch.object`` 的调用（如 ``owner._vote_one_rubric`` /
``owner._deterministic_check`` / ``owner._prepare_theorem_context`` /
``owner._is_correct_vote`` / ``owner._vote_one`` / ``owner._record_error_types``）
一律经 ``owner`` 解析 ⇒ patch 仍可穿透。
S3（2026-10-09）：``verifier.py`` 顶部已 `from . import verifier__vote as _vv`，
``_vote_one`` / ``_vote_profile`` / ``_vote`` 改为一行委托跳板。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re

from prompts.verifier import (
    VERIFIER_SCORING_SYSTEM,
    VERIFIER_SCORING_TEMPLATE,
    VERIFIER_SYSTEM,
    VERIFIER_USER_TEMPLATE,
)
from utils.prefill import prefill_messages, stitch

from .base import Verdict
from .verifier__review import _extract_error_type

logger = logging.getLogger("MathPilot.Verifier")


def _vote_one(owner, ctx, problem: str, candidate_text: str,
              temperature: float = 0.0, full_reasoning: bool = False) -> str:
    """单次投票（返回原始文本）。

        v2.4.1：prefill「VERDICT: 」抑制 CoT——投票输出只需 A/B，
        实测 prefill 仲裁 0.8s vs 普通 70.2s（140×），杜绝 Intern-S2 推理流占满预算。
        2026-09-15：注入检索到的 Mathlib 定理原文（方案 A，未启用时为空串）。
        2026-09-16：**温度改为参数化**。此前这里硬编码 0.0，导致
        `config.verifier_temperature` 是**死配置**（声明了/进白名单/有 CLI，
        但全仓无人读），且多张票在温度 0 下几乎完全相同 ⇒ **投票没有方差**，
        "多票" 只是同一判断重复 N 次（实测：6 候选 × 3 票对错答**全票投 A**）。

        ★ 2026-10-04（**B 折中**，用户决策「正确率第一位」）：新增 `full_reasoning`。
        为 True 时**不 prefill**，让验证器跑完完整推理再给判定 —— 依据是本文件自己的
        实测记录：「prefill 0.8s 不足以完成系统提示词要求的"**独立重算 + 逐条攻击**"
        ⇒ 投票退化成**"看一眼就点头"**」（10 题中 5 道错题的全部候选被全票判 A）。
        由 `_vote_profile` 在**候选分歧**时置位（折中：把昂贵算力花在真拿不准的题上）；
        开关 `verifier_vote_reasoning` 可切 `off`(C) / `disagree`(B) / `always`(A)。
        """
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": VERIFIER_USER_TEMPLATE.format(
            problem=problem,
            candidate_answer=candidate_text,
            theorem_context=(getattr(ctx, "_leansearch_ctx", "") or "(无)"),
        )},
    ]
    if full_reasoning:
        # 不 prefill ⇒ 原样返回完整原文（推理 + 判定行）。
        # ⚠ **不能**过 `stitch`：那会把 "VERDICT: " 种子里**拼到推理前面**，
        #   凭空造出一个空前缀行（虽然 `_last_verdict_ab` 取最后一个匹配仍正确，
        #   但文本被污染、且会给后续解析添乱）。直接返回最干净。
        return owner.llm(ctx, messages, temperature, 32768) or ""
    raw = owner.llm(ctx, prefill_messages(messages, "VERDICT: "),
                   temperature, 32768)
    return stitch("VERDICT: ", raw) if raw else raw


def _vote_profile(owner, ctx, candidates, base_votes=None) -> tuple:
    """返回本次投票的 ``(票数, 温度)``。

        ⚠ ``base_votes`` 必须传入**调用方决定的档位票数**（orchestrator 对 deep 档传 3）。
        2026-09-16 自审发现：本方法最初只读 `config.verifier_voting_times`(=1)，
        而 `run()` 里又无条件 `voting_times = _d_votes` ⇒ **把 deep 档的 3 票压回 1 票**
        （与 `tier_voting_times[deep]=3` 的设计直接矛盾）。现改为**以下限方式合并**：
        分歧时用 `max(档位票数, 分歧票数)`，一致时保留档位票数。
        """
    base_votes = int(
        base_votes if base_votes is not None
        else (getattr(owner.config, "verifier_voting_times", 1) or 1))
    base_temp = float(getattr(owner.config, "verifier_temperature", 0.0) or 0.0)
    _rmode = owner._vote_reasoning_mode()
    if not getattr(owner.config, "verifier_diversify_enabled", True):
        return base_votes, base_temp, (_rmode == "always")
    try:
        answers = set()
        # ★ 2026-09-16 审计修复：必须**剥壳后再比较**，否则 `\boxed{2026}` 与
        #   `2026` 会被当成两种答案 ⇒ 误判"候选分歧" ⇒ 无谓加票 + 调 0.7 温度
        #   （多花 LLM 调用）。与 `_are_answers_equivalent` 同口径。
        try:
            from .answer_oracle import AnswerOracle as _AO
        except Exception:  # noqa: BLE001
            _AO = None
        for c in (candidates or []):
            a = (c.get("answer", "") if isinstance(c, dict)
                 else getattr(c, "answer", "")) or ""
            a = str(a).strip()
            if not a:
                continue
            if _AO is not None:
                try:
                    a = _AO.strip_wrappers(a) or a
                except Exception:  # noqa: BLE001
                    pass
            answers.add(a)
    except Exception:  # noqa: BLE001
        return base_votes, base_temp, (_rmode == "always")
    if len(answers) < 2:
        # 候选答案一致 ⇒ 加票/加温都无意义（只会重复同一判断）
        return base_votes, base_temp, (_rmode == "always")
    n = max(base_votes, int(
        getattr(owner.config, "verifier_disagreement_votes", 3) or 1))
    t = float(getattr(
        owner.config, "verifier_disagreement_temperature", 0.7) or 0.0)
    owner.record(ctx, "vote_diversify",
                "候选答案存在 %d 种分歧 → 票数 %d、温度 %.2f"
                "（温度 0 下多票无方差，等于重复同一判断）"
                % (len(answers), n, t))
    return n, t, (_rmode != "off")


def _vote(
    owner, ctx, problem: str, candidate, total_votes: int = 5,
    proportional: bool = True, use_scoring: bool = False,
    temperature: float = 0.0, full_reasoning: bool = False,
) -> list[Verdict]:
    """
        批量投票并汇总（BUG-8 修复：用 valid_votes 替代 total_votes）。
        2026-09-02 修复：deadline 已过仍启动投票 → 每票 LLM 全被跳过返回 None，
        _is_correct_vote(None) 全判错 + 疯狂刷日志空转（027 实测超 deadline 137s）。
        """
    # 超时保护：deadline 已过 → 不投票（空 verdicts 由上层走兜底）
    if ctx.is_timed_out():
        logger.warning("Verifier: 单题 deadline 已过，跳过投票（%d 票）",
                       total_votes)
        return []
    text = owner._candidate_text(candidate)
    verdicts: list[Verdict] = []

    # ★ 2026-09-15 方案 A：**在启动投票线程之前**完成检索（单线程、每题一次），
    # 避免 6 候选 × 3 票 = 18 次并发重复检索。未启用/失败时注入为空串，
    # 验证器行为与改动前完全一致。
    owner._prepare_theorem_context(ctx, problem)

    # 二元投票
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(total_votes, owner.config.max_workers)
    ) as executor:
        futures = [executor.submit(owner._vote_one, ctx, problem, text,
                                   temperature, full_reasoning)
                   for _ in range(total_votes)]
        for f in concurrent.futures.as_completed(futures):
            try:
                raw = f.result()
                ok = owner._is_correct_vote(raw)
                verdicts.append(Verdict(
                    correct=bool(ok),
                    abstain=ok is None,   # 故障票 = 弃权，剔出 total_votes 分母
                    raw=raw,
                    # ★ 2026-09-15：判 B 时抽出结构化错误类型（A 票无此项）
                    error_type=(_extract_error_type(raw)
                                if ok is False else ""),
                ))
            except Exception as e:
                logger.warning(f"Vote failed: {e}")
                # 调用异常同样是基础设施故障 → 弃权，不得当反证
                verdicts.append(Verdict(correct=False, abstain=True, raw=str(e)))

    # 可选评分模式（补充/校准）
    if use_scoring and owner.config.use_scoring:
        try:
            scoring = owner._vote_one_scoring(ctx, problem, text)
            if scoring:
                verdicts.append(Verdict(
                    correct=scoring.get("overall", "B") == "A",
                    raw=json.dumps(scoring, ensure_ascii=False),
                    score=scoring,
                ))
        except Exception as e:
            logger.debug(f"Scoring vote failed: {e}")

    owner._record_error_types(ctx, verdicts)
    return verdicts


# VERDICT 行解析（2026-09-11）：放宽带换行 / \boxed{} 壳的写法。
# 此前拒绝词表里写死 `VERDICT\s*:\s*B`，模型输出 `VERDICT: \n\n\boxed{B}`
# 时被换行 + 壳打断 → 落入"无法解析" → 判错（实测 9 次）。
_VERDICT_LINE_RE = re.compile(
    r"VERDICT\s*[:：]\s*(?:\\boxed\s*\{\s*)?([AB])", re.IGNORECASE)


def _is_correct_vote(owner, text: str) -> bool | None:
    """解析 VERDICT 行 → True(判对) / False(判错) / None(弃权)。

        2026-09-11 三态化（依据 `Bug清单_团队提交de90cedc_0911.md` §3）：
        LLM 调用失败（text is None）与输出无法解析，语义上是**基础设施/格式
        故障 = 弃权**，不是"答案错误"。此前一律 `return False`，把 36 张故障票
        （27 次 None 输入 + 9 次解析失败，占票池 ~30%）当成反证计入：
        既压低簇置信度触发无谓 revise（单题白烧 96–158s），也把本可通过的
        候选直接投死。返回 None 的票在聚合时剔出 total_votes 分母。
        """
    if text is None:
        logger.warning("_is_correct_vote 收到 None 输入（LLM 调用失败）→ 弃权")
        return None
    text_upper = text.upper()

    # 0) VERDICT 行优先（权威信号，含换行/壳变体）。
    #    注意：在**原文**上做 IGNORECASE 匹配，不能在 text_upper 上匹配——
    #    upper() 会把 `\boxed` 变成 `\BOXED`，导致壳形态正则失配。
    m = _VERDICT_LINE_RE.search(text)
    if m:
        return m.group(1).upper() == "A"

    # 1) 拒绝词优先——规避"不正确"包含"正确"的误判
    reject_patterns = [
        r'\bINCORRECT\b', r'\bWRONG\b', r'\bFALSE\b',
        r'不\s*正\s*确', r'错\s*误', r'\bNO\b(?!\s*CHANGE|TE)',
    ]
    for pat in reject_patterns:
        if re.search(pat, text_upper):
            return False

    # 2) 接受词
    accept_patterns = [
        r'\bCORRECT\b', r'\bTRUE\b', r'正\s*确', r'\bYES\b',
    ]
    for pat in accept_patterns:
        if re.search(pat, text_upper):
            return True

    # 3) 无法判断 → 弃权（既不判对也不判错，不参与分母）
    logger.warning("无法从文本中解析 VERDICT，计为弃权票: %s", text[:100])
    return None


def _vote_one_scoring(owner, ctx, problem: str, candidate_text: str) -> dict | None:
    """评分模式投票（返回 JSON 或 None）。

        v2.4.1：prefill「{"」引导直接输出 JSON，抑制 CoT 前置长推理。
        """
    messages = [
        {"role": "system", "content": VERIFIER_SCORING_SYSTEM},
        {"role": "user", "content": VERIFIER_SCORING_TEMPLATE.format(
            problem=problem, candidate_answer=candidate_text
        )},
    ]
    raw = owner.llm(ctx, prefill_messages(messages, '{"'), 0.0, 32768)
    if raw:
        raw = stitch('{"', raw)
    # 2026-09-20 修复：self.llm 可能返回 None（_is_correct_vote 已自述该情形）。
    # 原实现直接 `json.loads(None)` 抛 TypeError，而这里只捕 JSONDecodeError，
    # 只能靠外层 `except Exception` 兜住 ⇒ 异常被静默吞掉、语义不清。
    # 同函数族其它 5 处均有 `if not raw` 守卫，此处补齐。
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None


def _vote_rubric(owner, ctx, problem: str, candidate,
                 use_deterministic: bool = True,
                 temperature: float = 0.0) -> list[Verdict]:
    """rubric 判分路径：结构化判分 + 确定性旁证（多证据汇审）。

        - rubric: verdict A/B + confidence + 错因定位（error_type/step_index/reason）
        - 确定性 fail → 硬否决（rubric 票全翻错，绕过 LLM 误判，0 LLM、可复现）
        - 确定性 pass → 追加一张独立正确票（deterministic_pass，非 LLM 客观票）
        - 确定性 unknown → 只挂证据，不改判
        """
    # 超时保护：deadline 已过 → 不判分（与 _vote 路径口径一致，空 verdicts 由上层走兜底）
    if ctx.is_timed_out():
        owner.record(ctx, "vote_timeout",
                    "rubric 判分路径：单题 deadline 已过，跳过判分（返回空票）")
        logger.warning("Verifier: 单题 deadline 已过，跳过 rubric 判分")
        return []
    text = owner._candidate_text(candidate)
    votes: list[Verdict] = []

    # 1) rubric 判分（1 次 JSON prefill）
    rub = owner._vote_one_rubric(ctx, problem, text)
    if rub is None:
        # 2026-09-11 三态化：判分调用/解析失败 = 基础设施故障 → 弃权。
        # 此前记 `correct=True`（"保守放行"）会凭空制造一张**正确票**、
        # 虚抬簇置信度——与"故障计成错票"是同一枚硬币的两面。
        # 改为弃权后既不判对也不判错，不参与 total_votes 分母；若该候选
        # 最终零有效票，orchestrator 的 5.5 低置信度通道仍会兜底复核。
        votes.append(Verdict(correct=False, abstain=True,
                             raw="rubric_parse_failed",
                             feedback="rubric 判分解析失败（弃权，不计票）"))
    else:
        correct = str(rub.get("verdict", "B")).upper() == "A"
        feedback = ""
        if not correct:
            parts = []
            step = rub.get("step_index")
            if step is not None:
                parts.append(f"步骤{step}")
            err = rub.get("error_type", "")
            if err and err != "无":
                parts.append(err)
            reason = rub.get("reason", "")
            if reason:
                parts.append(reason)
            feedback = "；".join(parts)
        try:
            conf = float(rub.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        votes.append(Verdict(
            correct=correct,
            raw=json.dumps(rub, ensure_ascii=False),
            feedback=feedback,
            score=rub,
            confidence=(conf if correct else 0.0),
            # ★ 2026-09-15：rubric 路径也回带结构化错误类型，口径与
            #   二元投票一致（同样走封闭标签集，认不出即空串）。
            error_type=(_extract_error_type(
                f"{rub.get('error_type', '')}；{feedback}") if not correct else ""),
        ))

    # 2) 确定性旁证（0 LLM 预算）：fail 硬否决 / pass 独立正确票 / unknown 挂证据
    if use_deterministic:
        det = owner._deterministic_check(ctx, problem, candidate)
        if det.get("verdict") == "fail":
            logger.warning("确定性验证失败 → 硬否决候选: %s",
                           det.get("evidence", "")[:80])
            for v in votes:
                v.correct = False
                v.confidence = 0.0
            votes.append(Verdict(correct=False, raw="deterministic_fail",
                                 feedback=f"确定性验证否决: {det.get('evidence', '')[:120]}",
                                 deterministic=det))
        elif det.get("verdict") == "pass":
            votes.append(Verdict(correct=True, raw="deterministic_pass",
                                 deterministic=det))
        else:
            # unknown：只挂证据，不追加票
            for v in votes:
                v.deterministic = det

    return votes


def _error_type_dist(verdicts) -> dict:
    """统计判 B 票的结构化错误标签分布。返回 {标签: 出现票数}。"""
    dist: dict = {}
    for v in (verdicts or []):
        et = getattr(v, "error_type", "") or ""
        if not et:
            continue
        for tag in et.split(","):
            tag = tag.strip()
            if tag:
                dist[tag] = dist.get(tag, 0) + 1
    return dist


def _record_error_types(owner, ctx, verdicts,
                        step: str = "vote_error_types") -> dict:
    """把结构化错误类型分布写进 trace，返回该分布字典。"""
    dist = owner._error_type_dist(verdicts)
    n_reject = sum(1 for v in (verdicts or [])
                   if v.correct is False and not getattr(v, "abstain", False))
    if n_reject:
        ordered = sorted(dist.items(), key=lambda kv: -kv[1])
        owner.record(
            ctx, step,
            "错误类型分布: " + (", ".join(f"{k}×{n}" for k, n in ordered)
                                or "(模型未回带标签)"),
            error_types=dist, n_reject=n_reject,
        )
    return dist
