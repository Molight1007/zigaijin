# -*- coding: utf-8 -*-
"""深复核（带推理的最终复核）职责簇（S4，从 agent/verifier.py 逐字搬出）。

`_deep_final_review` / `_deep_review_finalize` 搬为模块级函数，原 ``self``
作为第一个参数 ``owner`` 传入。

⚠ patch 穿透契约：``_deep_final_review`` 内部的 ``owner._deep_llm(...)`` 必须
经**宿主实例**解析 —— ``_deep_llm`` 是实例级 ``mock.patch.object`` 靶点，仍是
``VerifierAgent`` 类上的真方法。任何写成 ``type(owner)._deep_llm(owner, ...)``
的绕过都会让实例 patch 静默失效（tests/test_verifier_decomposition_guard.py
的 B 类穿透断言会变红）。
"""

from __future__ import annotations

import logging
import time

from prompts.verifier import (
    VERIFIER_DEEP_REVIEW_SYSTEM,
    VERIFIER_DEEP_REVIEW_TEMPLATE,
)

from .verifier__review import (
    _DEEP_VERDICT_CONDENSE_SYS,
    _DEEP_VERDICT_PREFIX,
    _dump_deep_review_sample,
    _extract_error_type,
    _last_verdict_ab,
)

logger = logging.getLogger("MathPilot.Verifier")


def _deep_final_review(owner, ctx, problem: str, best_cluster) -> str:
    """对最终选定答案做一次带推理的复核；返回原始文本（未启用/失败返回空串）。"""
    cfg = owner.config
    if not getattr(cfg, "verifier_deep_final_enabled", False):
        return ""
    if best_cluster is None:
        return ""
    # 预算护栏：剩余时间不足则不做（复核本身要几十秒）
    # ⚠ TaskContext **没有** remaining_time()，只有 is_time_critical()/is_timed_out()；
    #   剩余秒数须由 deadline 自行推算（写 hasattr 兜底会静默失效，已踩）。
    need = float(getattr(cfg, "verifier_deep_final_min_remaining", 150.0) or 0)
    try:
        _dl = getattr(ctx, "deadline", None)
        remain = (float(_dl) - time.time()) if _dl else None
        if ctx.is_time_critical() or (need and remain is not None and remain < need):
            owner.record(ctx, "deep_review",
                        "剩余 %s < %.0fs，跳过带推理复核" % (
                            "?" if remain is None else "%.0fs" % remain, need))
            return ""
    except Exception:  # noqa: BLE001
        pass

    answer = getattr(best_cluster, "answer_norm", "") or ""
    if not answer:
        return ""
    messages = [
        {"role": "system", "content": VERIFIER_DEEP_REVIEW_SYSTEM},
        {"role": "user", "content": VERIFIER_DEEP_REVIEW_TEMPLATE.format(
            problem=problem,
            candidate_answer=f"【最终答案】{answer}",
            theorem_context=(getattr(ctx, "_leansearch_ctx", "") or "(无)"),
        )},
    ]
    try:
        # ★ 不 prefill：允许完整思维流，这才是"独立重算"。
        # ★ 必须走**流式**（见 `_deep_llm`）：实测两例"允许推理"的调用均在
        #   全局 120s 超时（`LLM_TIMEOUT`）处失败、输出 0 字 —— 非流式路径
        #   对"要跑完整推理"的调用必然失败，功能等于不可用。
        raw = owner._deep_llm(
            ctx, messages,
            float(getattr(cfg, "verifier_deep_review_temperature", 0.0) or 0.0),
            int(getattr(cfg, "verifier_deep_review_max_tokens", 16384) or 16384))
    except Exception as e:  # noqa: BLE001
        logger.warning("带推理复核调用失败（不影响其余流程）: %s", e)
        return ""
    if not raw:
        return ""
    # ★ 2026-09-15 实测教训：用投票模板时模型只吐 **12 个字**（就是一行
    #   `VERDICT: A`），一个字的检查过程都没有。**短于阈值一律视为未完成复核**，
    #   绝不能当成"检查通过"——否则"没做检查"与"检查通过"无法区分，
    #   正是本项目反复踩的"机制触发 ≠ 效果提升"。
    min_chars = int(getattr(cfg, "verifier_deep_review_min_chars", 200) or 0)
    if min_chars and len(raw) < min_chars:
        owner.record(ctx, "deep_review",
                    "复核未按要求输出检查过程（仅 %d 字 < %d），视为未完成"
                    % (len(raw), min_chars),
                    chars=len(raw))
        return ""
    verdict = _last_verdict_ab(raw)
    # ★★★ 2026-09-16 修复（实测驱动的第二段收敛调用）：
    #   诊断落盘（`results/deep_review_samples.jsonl`）8/8 样本显示——
    #     首行 = "We need to verify the answer to the problem: ..."
    #     **VERDICT 出现 0 次**、chars = 26,749 ~ 42,545
    #   ⇒ 模型**完全无视系统提示的格式要求**，直接用英文自由推理，
    #     不写 VERDICT、不按四步写。故"把 VERDICT 置顶"的提示词改动**无效**。
    #   且样本 [4]/[6] 的 chars **完全相同（26749）**、尾部是
    #   `a_{a_1} = a_{a_1} = ...` **无限重复** ⇒ 模型陷入**复读退化循环**。
    #   修法：**两段式**——
    #     ① 第一段保留原样（自由推理，"独立重算"的价值在这里，不 prefill）；
    #     ② 首段没给出可解析判定时，追加一次**廉价收敛调用**：把推理尾部喂回，
    #        要求只输出一行 `VERDICT: A/B（标签）`。成本极小（max_tokens=64），
    #        但把"4 万字符作废"变成"拿到判定"。
    # ★ 2026-10-04 修复（判定被丢弃，机制空转）：
    #   收敛段一旦补出判定，必须把**含 `VERDICT:` 的那段原文**带回给调用方。
    #   否则调用方对**第一段 raw** 重新解析（`_last_verdict_ab(raw)`）恒为 None
    #   （首段恰恰是「没有 VERDICT」才会走到收敛调用），于是
    #   「判 B ⇒ 压制置信度 ⇒ 触发 revise」整条机制**静默失效**，
    #   而每次仍白付 1 段 16384 自由推理 + 0~1 段 8192 收敛调用的成本。
    _conv_text = ""
    if verdict is None and (raw or "").strip():
        try:
            # ★ 2026-09-17 改进：收敛调用改用 **prefill** 强制以 `VERDICT: ` 开头。
            #   首版（昨天）只靠提示词要求"只输出一行"，实测**11 次只成功 2 次** ——
            #   模型仍可能用英文复述而不是写 `VERDICT:` 字面量。
            #   本仓库在其它"必须单行输出"的场合（如 value_attack 的 JSON 提取）
            #   一直用 `prefill_messages(..., prefix)` + `stitch(prefix, raw)`：
            #   服务端会把 prefix 作为 assistant 回复的开头，**结构性地**保证格式。
            #   这里同样处理：prefix=`VERDICT: `。
            _msgs = [{"role": "system",
                      "content": _DEEP_VERDICT_CONDENSE_SYS},
                     {"role": "user",
                      "content": "【你此前的分析（末尾节选）】\n"
                                 + raw[-6000:]
                                 + "\n\n请只输出一行判定，不要任何其它文字。"}]
            try:
                from utils.prefill import prefill_messages, stitch
                _msgs = prefill_messages(_msgs, _DEEP_VERDICT_PREFIX)
                # 2026-10-02 DeepSeek 适配：原 64 ⇒ 8192。原值基于「输出很短（只一行
                # VERDICT）+ prefill 抑制思维块」的 Intern-S 时代假设；对 reasoning
                # 模型（如 deepseek thinking）思维块先吃满预算 ⇒ 正文为空、必截断。
                _conv = owner._deep_llm(ctx, _msgs, 0.0, 8192)
                _conv = stitch(_DEEP_VERDICT_PREFIX, _conv) if _conv is not None else ""
            except Exception:  # noqa: BLE001  prefill 不可用则退回原提示词路径
                # 2026-10-02 DeepSeek 适配：原 64 ⇒ 8192（同上）。
                _conv = owner._deep_llm(ctx, _msgs, 0.0, 8192)
            _v2 = _last_verdict_ab(_conv or "")
            if _v2 is not None:
                verdict = _v2
                # 收敛段命中判定 → 记下含 VERDICT 的原文，函数末尾回传（见下方 return）。
                _conv_text = _conv or ""
                owner.record(ctx, "deep_review",
                            "收敛调用补出判定: %s（首段 %d 字符未给 VERDICT）"
                            % ("A" if _v2 else "B", len(raw)),
                            chars=len(_conv or ""))
        except Exception as _e:  # noqa: BLE001
            logger.debug("深复核收敛调用失败（不影响主流程）: %s", _e)
    return owner._deep_review_finalize(ctx, raw, verdict, _conv_text)


def _deep_review_finalize(owner, ctx, raw: str, verdict, _conv_text: str) -> str:
    """复核收尾：样本落盘 + diag 埋点 + 返回文本组装（收敛段命中时拼尾部）。

        纯搬迁自 ``_deep_final_review`` 尾部（2026-10-08 批7 原地拆分），语句逐字不变。
        """
    # ★★ 2026-09-16 诊断落盘（用户要求）：把每次复核输出的**头部样本**存成
    #   jsonl。为什么必须落盘：`diag.deep_review` 只存 `chars`，
    #   于是"输出 4 万字符却解析不到 VERDICT"这件事**只能靠猜**。
    #   实测 003 的 6 次复核里 5 次"无法解析"，而看不到模型到底写了什么，
    #   无法判断是"没写 VERDICT"还是"写法不在解析器覆盖范围内"。
    #   不落全文（会很大），只存前 1200 字符 + 尾部 400 字符 —— 判定通常在首尾。
    try:
        _dump_deep_review_sample(ctx, raw, verdict)
    except Exception:  # noqa: BLE001
        pass
    owner.record(
        ctx, "deep_review",
        "带推理复核: %s" % ("A(通过)" if verdict is True
                           else "B(否决)" if verdict is False else "无法解析"),
        verdict=("A" if verdict is True else "B" if verdict is False else "?"),
        error_type=(_extract_error_type(raw) if verdict is False else ""),
        chars=len(raw))
    # ★ 2026-10-04：收敛段命中判定时，回传**含 VERDICT 的原文**（供调用方
    #   `_last_verdict_ab` 正确解析，从而真正触发「判 B ⇒ revise」）；
    #   否则维持原行为返回首段 raw（无 VERDICT 时调用方解析为 None，语义不变）。
    # ★★ 2026-10-05 修（**推理被整段丢弃** —— 三步论第 2 步"反馈要准确"）：
    #   原写法 `return _conv_text if _conv_text else raw` 在收敛段命中时
    #   **只回一行 VERDICT**，把首段那 **5.5万–7.1万字符的完整推理整个丢掉**
    #   ⇒ 实测 feedback 最终只剩 **44 字符**，模型拿不到"为什么判错"。
    #   现改为 **「首段推理（尾部）+ 收敛判定」** 一起回传：
    #     · `_last_verdict_ab` 取**最后一个** VERDICT ⇒ 仍解析到收敛段的判定 ✓
    #     · 反馈里保留推理 ⇒ 模型能看到判错依据、才知道该怎么改 ✓
    #   ⚠ 顺序不可反：判定必须在**最后**，否则会被推理中的假设性 VERDICT 覆盖。
    if _conv_text:
        _tail = (raw or "").strip()
        if len(_tail) > 6000:
            # 首段过长时保留**尾部**（最贴近结论的部分）——
            # 否则被下游截断后只剩与结论无关的开头，等于没留。
            _tail = "…（推理前段省略）…\n" + _tail[-6000:]
        return (_tail + "\n\n" + _conv_text) if _tail else _conv_text
    return raw
