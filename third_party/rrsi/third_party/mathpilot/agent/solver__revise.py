# -*- coding: utf-8 -*-
"""revise 重解执行链（2026-10-08 拆分：从 agent/solver.py 的
``_generate_revise`` 段 3–6 整段同迁）。

  · 段 3：装配「上一轮各候选答案清单 + 主推理尾部 1200 字」（``_prev_block``）；
  · 段 4：嵌套 ``_make_one`` —— prefill 压缩求解 + 幻觉/拒绝/截断重试，
    并注入定理线索、子目标结论；
  · 段 5：线程池分批提交（批间查生成侧软截止）+ 结果排序；
  · 段 6：逐条抽取 + 入池门槛 + append（失败候选写占位记录）。

搬运方式：**整段同迁**（段 3–6 逐字搬入，仅去 4 格缩进）。段 1–2（反馈去重、
易错点注入、候选数下限 / 池满腾位、``base_cid``）与段 7–8（本轮 record、
revise 后无条件自改进）仍留在 ``solver.py`` 的 ``_generate_revise`` 主体原位 ——
其中段 1 的 ``return``（无槽位 / 池满且无反馈时早退）保持原位，故「何时停止」
语义逐字不变。新增的只有：函数签名 / 段 3–6 所需输入改用形参传入 /
一处**惰性 import**（``_is_refusal`` 仍是 solver.py 模块级符号，惰性 import 避免循环导入）。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext

from .base import Candidate, detect_hallucination, detect_truncated
from prompts.revise import REVISE_SYSTEM, REVISE_USER_TEMPLATE
from utils.extract import (
    extract_final_answer, rescue_final_answer, smart_fallback_answer,
)

logger = logging.getLogger("MathPilot")


def _revise_run_sampling(self, ctx: TaskContext, count: int, base_cid: int,
                         feedback_text: str) -> None:
    """段 3–6：装配上一轮解答 → 并行重解 → 逐条入池。"""
    # 惰性 import：这是 solver.py 模块级符号，避免顶层循环导入。
    from .solver import _is_refusal

    # ★ 2026-09-17（M7）：把「上一轮解答」**真正**喂给 revise。
    # REVISE_SYSTEM 第 1/3 条明确要求“认真阅读【上一轮错误解答】”「保留正确的
    # 推理部分」，但 user 模板此前**只有题目 + 反馈** ⇒ 模型无从保留，只能整题
    # 重解。这正是 revise 越改越差的机制之一（013 蓝图 d=50 → revise 改成 19、
    # 025 蓝图 4 → 直答 509040-2*169680）。此处给出上一轮各候选的答案清单
    # 与主推理尾部，使“保留正确部分”成为可执行指令。
    _prev_lines = []
    for _pc in (getattr(ctx, "candidates", None) or [])[:6]:
        _pa = str(getattr(_pc, "answer", "") or "").strip()
        if _pa:
            _prev_lines.append("· 候选#%s 答案：%s"
                               % (getattr(_pc, "id", "?"), _pa))
    _cs_all = list(getattr(ctx, "candidates", None) or [])
    _main_reasoning = ""
    if _cs_all:
        _main_reasoning = max(
            (str(getattr(_c2, "reasoning", "") or "") for _c2 in _cs_all),
            key=len)
    _prev_block = "\n".join(_prev_lines) or "（无——本轮无既有候选答案）"
    if _main_reasoning:
        _prev_block += ("\n\n（上一轮主推理尾部 1200 字，供沿用正确结论）\n"
                        + _main_reasoning[-1200:])

    def _make_one(i: int):
        cid = base_cid + i
        user_content = REVISE_USER_TEMPLATE.format(
            problem=ctx.problem, feedback=feedback_text,
            prev_answer=_prev_block)
        # ★ 2026-09-29：revise 是**独立于 _generate_initial 的生成路径**，
        # 不注入则"重解一次"会把定理线索丢掉（本项目教训：只改一处路径=没改）。
        if getattr(self.config, 'enable_theorem_hint', True):
            _th = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
            if _th:
                user_content = user_content + "\n\n" + _th
        # ★ 2026-10-02（阶段一补）：重解同属"主求解"，不带子目标结论=又从头推
        # 一遍。复用同一注入逻辑（含 diag 埋点）。
        user_content = self._apply_subgoal_findings(ctx, user_content)
        for retry in range(3):
            # 2026-09-13 循环时间检查（防卡死）：同 _generate_initial 处说明。
            # 本函数在线程池内并行执行，到点即 break，避免"重试 × 并行"叠加穿预算。
            if retry > 0 and ctx.gen_time_up():
                self.record(ctx, "revise",
                            f"revise 候选 {cid} 重试因时间到点提前停止")
                break
            # v2.4.1：revise 也走 prefill（完整 CoT 在本环境必然超时）
            # 2026-09-04：prefill 种子与 REVISE_SYSTEM 五段格式对齐——【错误分析】开头
            resp = self._compressed_solve(
                ctx, REVISE_SYSTEM, user_content,
                temperature=self.config.policy_temperature,
                max_tokens=self._adaptive_max_tokens(
                    ctx, self.config.policy_max_tokens),
                prefill_seed="【错误分析】\n",
            )
            if resp is not None and resp.strip():
                # 幻觉/拒绝检测（与 _generate_initial 一致）
                hallu = detect_hallucination(resp)
                if any("42" in h[0] for h in hallu) and retry < 2:
                    logger.warning("Revise %d 42-dodge (retry %d/2)", cid, retry + 1)
                    time.sleep(1)
                    continue
                if _is_refusal(resp) and retry < 2:
                    logger.warning("Revise %d refused (retry %d/2)", cid, retry + 1)
                    time.sleep(1)
                    continue
                if detect_truncated(resp):
                    logger.info("Revise %d truncated; will attempt completion", cid)
                return cid, resp
            if retry < 2:
                logger.warning("Revise candidate %d empty response (retry %d/2)", cid, retry + 1)
                time.sleep(1)
        return cid, resp

    # 并行生成修正候选
    results = []
    max_workers = min(count, 6)
    # 2026-09-13：与 `_generate_initial` 同款修复 —— 改**分批提交 + 批间检查**。
    # 原实现一次性提交全部候选且 `as_completed` 内无任何时间检查、已提交任务无法取消
    # ⇒ 生成侧软截止 `gen_time_up()` 在这段失效（该病理已在 3_solve 实测造成
    # 单题超出软预算 421s，见 `_generate_initial` 处注释）。
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        _pending = list(range(count))
        while _pending:
            if results and ctx.gen_time_up():
                self.record(ctx, "revise",
                            f"生成侧时间到，跳过剩余 {len(_pending)} 个修正候选"
                            f"（已完成 {len(results)}/{count}）")
                break
            _batch, _pending = _pending[:max_workers], _pending[max_workers:]
            _futs = [pool.submit(_make_one, i) for i in _batch]
            for _f in as_completed(_futs):
                results.append(_f.result())
    results.sort(key=lambda x: x[0])

    for cid, resp in results:
        if resp is None:
            ctx.candidates.append(Candidate(
                id=cid, answer="", reasoning="[重解失败] 调用受限"))
            logger.warning("Revise candidate %d failed/skipped", cid)
            continue
        answer = extract_final_answer(resp)
        if not answer or len(answer) > 300:
            answer = rescue_final_answer(resp)[0]
        if not answer:
            answer = smart_fallback_answer(resp)
        # ★★ 2026-10-06：**重解路径同样做入池门槛**（与 _generate_initial 同口径）。
        #   ⚠ 注意：上面那条 `answer=""` 的"[重解失败]"占位候选**不加门槛**
        #   （它是显式标记的失败记录，不是"假答案"）。
        answer, _ans_ok = self._normalize_and_check_answer(ctx, cid, answer)
        if not _ans_ok:
            logger.warning("Revise candidate %d 不含最终答案 → 不入池", cid)
            continue
        ctx.candidates.append(Candidate(
            id=cid,
            answer=answer,
            reasoning=resp,
            revised=True,
        ))
