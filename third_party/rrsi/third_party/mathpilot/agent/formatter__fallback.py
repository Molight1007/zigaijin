"""兜底 / 续写 / 紧急直答 / 终检修复（2026-10-08 从 agent/formatter.py 逐字搬出）。

簇⑤：`_pick_fallback` · `_repair_truncated` · `_emergency_answer` ·
`_diagnose_and_repair`。formatter.py 原位保留**同名一行委托** ⇒
`FormatterAgent._*` 契约不变。

依赖（顶层）：`is_truncated_answer`(utils.extract) · `_rank_key`(formatter__rank) ·
`_REFUSAL_RE` / `_INCOMPLETE_RE` / `_final_answer_min_sec` /
`_emergency_answer_max_tokens`（仍留 formatter.py，经 .formatter 导入）·
`logger`（同名 MathPilot logger，`logging.getLogger` 单例 ⇒ 与 formatter.logger 同对象）。
⚠ formatter.py 对 `formatter__fallback` **只在委托方法内惰性 import** ⇒ 无循环。
`_normalize_chat_response` / `prefill_messages` / `stitch` / `extract_final_answer`
仍为**函数内惰性 import**（与搬出前逐字一致）。
"""
from __future__ import annotations

import logging
import re

from utils.extract import is_truncated_answer
from .base import TaskContext
from .formatter__rank import _rank_key
from .formatter import (
    _REFUSAL_RE, _INCOMPLETE_RE,
)

logger = logging.getLogger("MathPilot")


def _pick_fallback(self, ctx: TaskContext, exclude_answer: str = "") -> str:
    """当最佳答案是拒绝/不完整回答时，从候选中找到更可靠的答案"""
    exclude = exclude_answer or ""
    # 优先从有效 verdicts 中找非拒绝/非不完整答案（按置信度降序）
    valid_verdicts = [v for v in ctx.verdicts if v.total_votes > 0]
    if valid_verdicts:
        for v in sorted(valid_verdicts, key=lambda x: x.confidence, reverse=True):
            ans = getattr(v, "answer", "") or ""
            # ★ 2026-09-16 审计修复：去掉 `len(ans) > 3`（同 :205 的理由）。
            #   本函数是**兜底换候选**路径，门槛过严会把合法短答案（A/AB/BCD）
            #   全部跳过、退回推理尾部散文 ⇒ 客观题必错。
            if (ans.strip()
                    and ans != exclude
                    and not _REFUSAL_RE.search(ans)
                    and not _INCOMPLETE_RE.search(ans)):
                return ans
    # 从 candidates 中找非拒绝/非不完整答案
    if ctx.candidates:
        _cs = sorted(ctx.candidates, key=_rank_key, reverse=True)
        # ★ 2026-09-22：把"并列且无判别信号"这件事**显式暴露**出来。
        #   实测 011：两个候选的正确票数、置信度、答案形态**完全并列**
        #   （3/3 vs 3/3、conf 1.0 vs 1.0、都带 `\boxed`），此时**本地没有任何
        #   判别信号** —— 旧逻辑拿"推理更长者"当裁判，等于随机且系统性偏向啰嗦。
        #   本处只做两件事：
        #     ① 判据中性化（末键用答案紧凑度，不再用推理长度）；
        #     ② 把并列事实写进 diag 与 trace，使其**可统计、可追踪**。
        #   ⚠ 这不构成"解决了 011"：真正的解法是对并列候选做**外部验证**
        #   （符号等价 / 代回题设），已列入 CHANGES 待办，尚未实现。
        if len(_cs) >= 2 and _rank_key(_cs[0])[:3] == _rank_key(_cs[1])[:3]:
            self.record(ctx, "finalize",
                        "候选择优出现**无信号并列**（票数 / 置信度 / 答案形态全同）"
                        "→ 已按确定性规则择一，但该选择无证据支撑，标记 tie_unresolved")
            try:
                _pd = getattr(ctx, "_pick_diag", None)
                if isinstance(_pd, dict):
                    _pd["tie_unresolved"] = True
                    _pd["tie_pair"] = [(_cs[0].answer or "")[:40],
                                       (_cs[1].answer or "")[:40]]
            except Exception:  # noqa: BLE001
                pass
        for c in _cs:
            ans = c.answer or ""
            # ★ 2026-09-16 审计修复：去掉 `len(ans) > 3`（同 :493 的理由）。
            if (ans.strip()
                    and ans != exclude
                    and not _REFUSAL_RE.search(ans)
                    and not _INCOMPLETE_RE.search(ans)):
                return ans
            # 候选答案也是拒绝类，但推理足够长 → 提取尾部
            if c.reasoning and len(c.reasoning) > 200:
                from utils.extract import extract_final_answer
                fallback = extract_final_answer(c.reasoning)
                if (fallback and len(fallback) > 2
                        and fallback != exclude
                        and not _REFUSAL_RE.search(fallback)
                        and not _INCOMPLETE_RE.search(fallback)):
                    return fallback
    return ""


def _repair_truncated(self, ctx: TaskContext, answer: str) -> str:
    """2026-09-02 老师需求：截断答案续写补全（根治 -2x^{ 型截断）。

        截断信号：is_truncated_answer(answer) 为 True（LaTeX 半截/花括号不配对等）。
        用 LLM 从断点续写补全，最多 2 次；补全后仍截断则原样返回。
        """
    if not answer or not is_truncated_answer(answer):
        return answer
    try:
        from .base import _normalize_chat_response
        from utils.prefill import prefill_messages, stitch
        for attempt in range(2):
            if not is_truncated_answer(answer):
                break
            # 2026-09-13 超时护栏（既有失败语义 = 原样返回 answer）：
            # deadline / 生成侧软截止已到 → 不再续写，直接落到函数末尾
            # `return answer`，与原有"补全失败不阻断"语义一致。
            # ⚠ 2026-09-16 修复（老逻辑漏改）：本行原先用 `ctx.gen_time_up()`，
            #   而**同文件 414-418 行已明确论证不能这么用** ——
            #   formatter 是流程最后一个阶段，走到这里 `gen_time_up()` **必然为 True**
            #   ⇒ 截断答案续写**从未真正执行过**，只能原样交截断答案。
            #   414-421 与 589-593 两处当时都改了，**这处漏改**，属典型"同一 bug
            #   修了两处漏第三处"。现对齐 421 行的口径：只看硬限与真实剩余时间。
            if ctx.is_timed_out() or ctx.time_remaining() < 120:
                self.record(ctx, "finalize",
                            "截断答案续写：时间已到，跳过续写（原样返回）")
                break
            tail = answer[-200:]  # 断点前片段作锚
            msgs = [
                {"role": "system", "content":
                    "你是数学解答补全器。下面是一段被截断的解答结尾，"
                    "请从断点处继续，把最终答案补全为完整、规范的数学答案。"
                    "只输出续写内容（从断点开始），不要重复已给出的片段。"},
                {"role": "user", "content": tail},
            ]
            resp = self.client.chat(
                messages=prefill_messages(msgs, "【续写】: "),
                temperature=0.0, max_tokens=65536,
            )
            text = _normalize_chat_response(resp)
            if not text:
                break
            text = stitch("【续写】: ", text)
            m = re.search(r"【续写】[:：]?\s*([\s\S]+)", text)
            piece = m.group(1).strip() if m else text.strip()
            if not piece:
                break
            # 防重复拼接：piece 若以 answer 尾部开头则剪掉重复段
            for cut in (80, 50, 30):
                dup = answer[-cut:]
                if dup and piece.startswith(dup):
                    piece = piece[cut:]
                    break
            answer = (answer + piece).strip()
            self.record(ctx, "finalize",
                        f"截断答案续写补全（第{attempt + 1}次）→ {answer[:120]}")
    except Exception:  # noqa: BLE001  补全失败不阻断，原样返回
        pass
    return answer


# ★ 2026-10-15（用户定案「删直答兜底」）：`_emergency_answer`（占位符 → 最精简
#   直答）已删除 —— 答案占位/空时不再向 LLM 直答，如实暴露由评测判 0。
#   `_pick_fallback`（换候选，从智能体已产出候选里挑）与 `_repair_truncated`
#   （截断补全）、`_diagnose_and_repair`（纯规则终检）保留，性质不同。


def _diagnose_and_repair(answer: str, ctx: TaskContext) -> str:
    """
        答案质量终检 + 自动修复。

        检测项:
        - 42 幻觉兜底（孤立的 42）
        - 截断 LaTeX（未闭合的 $ / { / \\begin）
        - markdown 污染（**...** 残留）
        - 多余包装文字

        返回修复后的答案（或原答案）。
        """
    if not answer or answer == "无法求解":
        return answer

    fixed = answer

    # 1. markdown 污染检测与修复
    if "**" in fixed or "__" in fixed:
        fixed = fixed.replace("**", "").replace("__", "")
        logger.info("Formatter 终检: 移除 markdown 标记")

    # 2. 42 幻觉检测（孤立的 42 / 42.0）
    stripped = fixed.strip()
    if re.fullmatch(r"42(?:\.0+)?", stripped):
        logger.warning("Formatter 终检: 检测到 42 兜底幻觉 → 尝试回溯")
        # 从其他候选中找到非 42 的答案
        for v in (ctx.verdicts or []):
            ans = (getattr(v, "answer", "") or "").strip()
            if ans and not re.fullmatch(r"42(?:\.0+)?", ans) and len(ans) > 1:
                return ans
        for c in (ctx.candidates or []):
            ans = (c.answer or "").strip()
            if ans and not re.fullmatch(r"42(?:\.0+)?", ans) and len(ans) > 1:
                return ans

    # 3. 截断 LaTeX 检测
    if re.search(r"\\begin\{[^}]*\}\s*$", fixed):
        logger.warning("Formatter 终检: 答案以 \\begin 结尾（截断）→ 尝试补全")
        # 从 reasoning 找对应的完整表达式
        for c in (ctx.candidates or []):
            if not c.reasoning:
                continue
            env_match = re.search(
                r"\\begin\{([^}]+)\}.*?\\end\{\1\}",
                c.reasoning, re.DOTALL,
            )
            if env_match:
                return env_match.group().strip()
    # 未闭合的 $ 或 {
    if fixed.count("$") % 2 == 1:
        fixed = fixed.rstrip("$")  # 移除不配对的 $
    open_braces = fixed.count("{") - fixed.count("}")
    if open_braces > 0:
        fixed = fixed + "}" * open_braces  # 补全大括号

    # 4. 多余包装文字剥离（如「因此答案是 x」→「x」）
    wrapped = re.match(
        r"^(?:因此|所以|故|综上[所]?述|答案为?|最终答案[为是]?)[,，:：]?\s*(.+?)\s*(?:。|$)",
        fixed, re.DOTALL | re.IGNORECASE,
    )
    if wrapped and len(wrapped.group(1)) > 1:
        inner = wrapped.group(1).strip()
        if inner != fixed.strip():
            logger.info("Formatter 终检: 剥离包装文字")
            return inner

    return fixed
