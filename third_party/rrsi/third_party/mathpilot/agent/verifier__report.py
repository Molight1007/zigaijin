# -*- coding: utf-8 -*-
"""结构化 bug report + 反馈抽取（从 agent/verifier.py 逐字搬出）。

S2（2026-10-09）：``_extract_bug_report`` / ``_format_bug_report`` /
``structured_report_for`` 搬为模块级函数，原 ``self`` 作为第一个参数
``owner`` 传入（``_format_bug_report`` 原为 staticmethod，无接收者）。
S3（2026-10-09）：``verifier.py`` 顶部已 `from . import verifier__report as _vr`，
``_extract_bug_report`` / ``_format_bug_report`` / ``structured_report_for``
改为一行委托跳板（``_format_bug_report`` 保持 staticmethod）。
"""

from __future__ import annotations

import logging

from prompts.verifier import (
    VERIFIER_BUGREPORT_SYSTEM,
    VERIFIER_BUGREPORT_TEMPLATE,
    VERIFIER_FEEDBACK_SYSTEM,
    VERIFIER_FEEDBACK_TEMPLATE,
)
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot.Verifier")


def _extract_bug_report(owner, ctx, problem: str, candidate,
                        tool_findings: str = "") -> dict:
    """让验证器产出结构化 bug report（分类 + 精确定位），而非一句话错因。

        论文依据：Huang & Yang (2025) 用「验证 + 精炼」流水线把 IMO 2025
        从 best-of-32 的 21.4%~38.1% 提到 85.7%。关键不在于多采样，而在于
        验证器给出**可执行的错因**（哪一步、什么类型、为什么），
        修正步骤才能有的放矢。

        ★★ 2026-10-06 新增 `tool_findings`（**补上「工具判定 → 答案定位」这一环**）：
        工具的 finding 里 `location` 装的是**检查项名**（`answer_verify`/`lean_verify`），
        而 `agent/base.py:365` 对 `location` 的定义是「出错位置（步骤编号/行号/引用）」
        ⇒ **契约被违背**：模型收到 `location="answer_verify"` 等于没收到位置。
        工具看的是**形式化代码**，天然说不出"答案第 3 步错了"；但它最可靠的是
        「**这个等式/这条命题不成立**」（客观、可复现）。
        ⇒ 把工具判定作为**额外输入**喂进来，让检错器**只负责定位**
          （在答案里逐字引用写下该等式的原文），从而得到
          「**工具级的可靠性 + 答案级的定位**」。

        返回 dict：{verdict, findings:[{location, type, explanation}]}
        解析失败时返回 {"verdict": "unknown", "findings": []}。
        """
    text = owner._candidate_text(candidate)
    messages = [
        {"role": "system", "content": VERIFIER_BUGREPORT_SYSTEM},
        {"role": "user", "content": VERIFIER_BUGREPORT_TEMPLATE.format(
            problem=problem, candidate_answer=text
        )},
    ]
    if tool_findings and str(tool_findings).strip():
        # ★ 工具判定是**客观、可复现**的 ⇒ 不要求检错器复核它，只要求**定位**。
        #   ⚠ 明确约束"不要推翻"，否则模型可能去质疑工具结论、带回噪声。
        messages.append({"role": "user", "content": (
            "【工具已客观判定的错误（**不可质疑**，你只需定位）】\n"
            + str(tool_findings).strip()[:2000]
            + "\n\n请**只针对上面这些已被工具确认的错误**做定位：\n"
                  "  1. 在【待审查的解答】里**逐字引用**写下这些等式/命题的原文"
                  "（这是本步唯一的产出重点）；\n"
                  "  2. 说明它出现在第几步、为什么会写成这样；\n"
                  "  3. 给出可执行的修法。\n"
                  "工具看的是形式化代码、说不出答案文本里的位置 —— **这一步正是要你来补**。\n"
                  "若在解答里找不到对应原文（可能是形式化翻译偏差），"
                  "请如实写明，不要编造引用。")})
    empty = {"verdict": "unknown", "findings": []}
    try:
        # prefill 锚定到 JSON 开头：Intern 无短种子会先吐思维块吃满预算
        raw = owner.llm(ctx, prefill_messages(messages, "{"), 0.0, 32768)
        if not raw:
            return empty
        raw = stitch("{", raw)
    except Exception as e:  # noqa: BLE001
        logger.error(f"BugReport extraction failed: {e}")
        return empty

    data = owner._parse_json_loose(raw)
    if not isinstance(data, dict):
        return empty
    findings = []
    for f in (data.get("findings") or []):
        if not isinstance(f, dict):
            continue
        loc = str(f.get("location") or "").strip()
        expl = str(f.get("explanation") or "").strip()
        ftype = str(f.get("type") or "").strip().lower()
        if ftype not in ("critical_error", "justification_gap"):
            ftype = "justification_gap"
        if not (loc or expl):
            continue
        findings.append({"location": loc, "type": ftype,
                         "explanation": expl})
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in ("correct", "critical_error", "justification_gap"):
        # 以 findings 反推，比信任模型的自陈更可靠
        verdict = ("critical_error"
                   if any(f["type"] == "critical_error" for f in findings)
                   else "justification_gap" if findings else "unknown")
    return {"verdict": verdict, "findings": findings}


def _format_bug_report(report: dict) -> str:
    """把结构化 bug report 渲染成注入 solver 的反馈文本。

        按「关键错误优先」排序：修正步骤应当先修断链的错误，
        论证漏洞其次（否则会先去补一处无关紧要的严谨性，浪费修正预算）。
        """
    findings = report.get("findings") or []
    if not findings:
        return ""
    crit = [f for f in findings if f["type"] == "critical_error"]
    gaps = [f for f in findings if f["type"] == "justification_gap"]
    lines = []
    if crit:
        lines.append("【关键错误（必须修正，否则整条推理链作废）】")
        for i, f in enumerate(crit, 1):
            lines.append(f"{i}. 位置：“{f['location']}”")
            lines.append(f"   问题：{f['explanation']}")
    if gaps:
        lines.append("【论证漏洞（需补充论证，结论可能仍成立）】")
        for i, f in enumerate(gaps, 1):
            lines.append(f"{i}. 位置：“{f['location']}”")
            lines.append(f"   问题：{f['explanation']}")
    # ★★ 2026-10-05 新增（**"以模型能理解的方式反馈 + 让它修正"**）：
    #   ① **不生成 fix_hint**，而是把"怎么改"的责任交给解题方 —— 因为
    #      验证器的系统提示词明确写着「**绝不要试图纠正你发现的错误或填补漏洞**」
    #      （`prompts/verifier.py:224`），一旦让它给修法，它就会开始替解答辩护、
    #      漏掉真错。imo2025 §2.3 的做法正是「**让被批评方把表述改清楚**」。
    #   ② 加「逐条回应」纪律：把"报告"变成"必须逐条答复的清单"，
    #      避免模型忽略部分条目、或只做泛泛重写。
    #   ③ 加「**不要重写未被指出的步骤**」：防止被打回后整篇重写、
    #      把本来正确的部分也改掉（这是"越改越错"的来源之一）。
    lines.append("")
    lines.append("【修正纪律（必须逐条遵守）】")
    lines.append("1. 对上面**每一条**都要给出【同意 / 不同意】+ 理由，不许跳过；")
    lines.append("2. **同意**的必须改，并写出**改正后的那一步（或那段论证）**；")
    lines.append("   不同意的必须给出反驳依据（引用题目条件或已证结论）；")
    lines.append("3. ★ **不要重写未被指出的步骤** —— 上面没有点名的地方保持原样，")
    lines.append("   **正确的部分必须保留**（整篇重写会把对的也一起改掉）。")
    return "\n".join(lines)


def structured_report_for(owner, ctx, problem: str, candidate,
                          tool_findings: str = "") -> str:
    """公开入口：为**指定候选**取一份结构化 bug report（位置+类型+问题）。

        ★★ 2026-10-06 新增（"让结构化定位成为 revise 的**主导**反馈"）：
        实测题 000 在 `revise_round=5` 的情况下，反馈里结构化「位置：」出现 **0 次** ——
        因为 `_extract_revise_feedback` 的第一道门槛是
        `if best_cluster.confidence >= 0.5: return ""`，而该题恰好是
        **"投票高置信、但答案错误"** ⇒ **结构化报告从未生成**，
        模型只能靠 Lean「疑似自证」与对抗那几条噪声重写（12/15 被噪声占据）。

        本方法**不受 confidence 门槛约束**，供"**已经确定要重写**"的路径显式调用，
        确保模型手上一定有一份"错在第几步、哪句话、为什么"的定位。
        失败/未启用时返回空串，调用方按原逻辑继续（纯增量，不改变既有行为）。
        """
    try:
        if not getattr(owner.config, "use_bug_report_feedback", True):
            return ""
        if candidate is None:
            return ""
        rep = owner._extract_bug_report(ctx, problem, candidate,
                                       tool_findings=tool_findings)
        out = owner._format_bug_report(rep)
        if out:
            owner.record(ctx, "bug_report",
                        "为重写路径补取结构化 bug report（%d 字符；工具判定输入 %d 字符）"
                        % (len(out), len(str(tool_findings or ""))))
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("structured_report_for 失败（返回空）: %s", str(e)[:120])
        return ""


def _extract_feedback(owner, ctx, problem: str, candidate) -> str:
    text = owner._candidate_text(candidate)
    messages = [
        {"role": "system", "content": VERIFIER_FEEDBACK_SYSTEM},
        {"role": "user", "content": VERIFIER_FEEDBACK_TEMPLATE.format(
            problem=problem, candidate_answer=text
        )},
    ]
    try:
        # v2.4.1：prefill「错因：」抑制 CoT，直接输出错因定位
        raw = owner.llm(ctx, prefill_messages(messages, "错因："), 0.0, 32768)
        return stitch("错因：", raw) if raw else "无法提取失败原因。"
    except Exception as e:
        logger.error(f"Feedback extraction failed: {e}")
        return "无法提取失败原因。"


def _oracle_sanity_feedback(candidate) -> str:
    """用 AnswerOracle 做客观 sanity check（纯本地），返回客观反馈。"""
    answer = getattr(candidate, "answer", "") or ""
    if not answer:
        return "候选答案为空，需重新求解。"
    try:
        from .answer_oracle import AnswerOracle
        if not AnswerOracle.is_parseable(answer):
            return "候选答案无法解析为有效数学表达式，可能为幻觉或格式错误。"
    except Exception as exc:  # noqa: BLE001
        # 2026-09-04 审核：异常吞掉 = "可解析性检查"静默跳过 → 全部放行
        # （与 lean_gate 吞 AttributeError 历史同型）。留证据。
        logger.debug("verifier 可解析检查异常（跳过放行）: %s: %s",
                     type(exc).__name__, exc)
    return ""


def _extract_revise_feedback(owner, ctx, problem: str,
                             candidates: list, best_cluster) -> str:
    """恢复 Reflexion 反馈：仅在需要 revise 时提取错因（受预算约束）。

        触发条件：best_cluster 存在且置信度 < 0.5（正确票未过半，验证器自身
        都不确定）。此时提取 LLM 错因定位，并叠加 AnswerOracle 的客观
        sanity check（纯本地、不消耗预算）。

        返回空串表示无需 revise 或预算不足。
        """
    if best_cluster is None:
        return ""
    if getattr(best_cluster, "confidence", 0.0) >= 0.5:
        return ""
    if ctx.is_time_critical():
        return ""

    # 取 best_cluster 的代表候选（共识簇内第一个候选）
    rep_candidate = None
    cids = getattr(best_cluster, "candidate_ids", []) or []
    if cids and candidates:
        idx = cids[0] if cids[0] < len(candidates) else 0
        rep_candidate = candidates[idx]
    if rep_candidate is None and candidates:
        rep_candidate = candidates[0]
    if rep_candidate is None:
        return ""

    parts = []
    # 1) LLM 错因提取（消耗 1 次预算）
    #    优先用结构化 bug report（分类 + 原文定位，修正时有的放矢）；
    #    解析不出 findings 时回退到原来的一句话错因。二者都只花 1 次调用，
    #    所以这是替换不是叠加。
    llm_feedback = ""
    if getattr(owner.config, "use_bug_report_feedback", True):
        report = owner._extract_bug_report(ctx, problem, rep_candidate)
        llm_feedback = owner._format_bug_report(report)
    if not llm_feedback:
        llm_feedback = owner._extract_feedback(ctx, problem, rep_candidate)
        if llm_feedback == "无法提取失败原因。":
            llm_feedback = ""
    if llm_feedback:
        parts.append(llm_feedback)
    # 2) AnswerOracle 客观 sanity check（纯本地，不消耗预算）
    oracle_fb = owner._oracle_sanity_feedback(rep_candidate)
    if oracle_fb:
        parts.append(oracle_fb)

    if parts:
        return "\n".join(parts)
    return "所有候选均未获验证通过，请重新审题并纠正推理错误。"
