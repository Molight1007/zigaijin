# -*- coding: utf-8 -*-
"""初始候选并行采样与落盘（2026-10-08 拆分：从 agent/solver.py 的
``_generate_initial`` 段 13–16 整段同迁）。

  · ``_run_initial_sampling`` —— 段 13–14：嵌套 ``_make_one``（prefill 压缩求解 +
    空响应/模板泄露/幻觉/截断/拒绝检测与重试）+ 线程池分批提交（批间查生成侧软截止）
    + 结果排序；随后逐条委托 ``_store_candidate_result`` 落盘。
  · ``_store_candidate_result`` —— 段 15–16：单条结果的"失败抢救 / 正常抽取 +
    reask + 三通道 + 入池门槛 + append"。

搬运方式：**整段同迁**（段 13–14 逐字搬入）。段 15–16 原为 ``for`` 循环体，
搬入 help 时**唯一的 ``continue``（循环体末尾跳出本次迭代）等价改写为 ``return``**
（函数返回到 for 循环即进入下一次迭代，控制流语义不变）；其余逐字。
新增的只有：函数签名 / ``_make_one`` 所需闭包变量改由形参传入 /
``run_initial_sampling`` 内的 for 循环骨架 / 两处**惰性 import**
（``_is_refusal``/``_REINFORCED_SYSTEM`` 仍是 solver.py 模块级符号，惰性 import 避免循环导入）。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext

from .base import (
    Candidate, detect_hallucination, detect_truncated, detect_template_leak,
)
from utils.extract import (
    extract_final_answer, rescue_final_answer, smart_fallback_answer,
)

logger = logging.getLogger("MathPilot")


def _run_initial_sampling(self, ctx: TaskContext, count: int, base_cid: int,
                          _STRATIFIED_TEMPS: list, system_prompt: str,
                          user_content: str) -> None:
    """段 13–14：并行生成初始候选并排序，再逐条落盘（段 15–16）。"""
    # 惰性 import：这两个是 solver.py 模块级符号，避免顶层循环导入。
    from .solver import _is_refusal, _REINFORCED_SYSTEM
    def _make_one(i: int):
        cid = base_cid + i
        # 温度分层：按索引轮转取值（count>=3 时生效；deep 档 4 温度 0.1/0.3/0.5/0.7）
        base_temp = _STRATIFIED_TEMPS[i % len(_STRATIFIED_TEMPS)] if count >= 3 else self.config.policy_temperature
        # 候选 2+ 追加微扰动提示，引导不同解题思路
        _perturb_hints = [
            "",  # 候选 0: 无扰动（直接求解）
            "\n请特别注意计算过程中的每一步细节，确保数值精确。",  # 候选 1: 精度
            "\n如果可以，尝试用另一种方法重新审视这个问题。",  # 候选 2: 换方法
            "\n请先列出解题关键思路与可能用到的定理/公式，再逐步求解。",  # 候选 3: 计划先行
        ]

        # v2.4.1 主路径：prefill 压缩求解（诊断实测 37s 返回，答案前置不受截断影响）。
        # 本环境完整 CoT >200s 不返回（780s 仍读超时），prefill 是唯一保证
        # 300s 单题预算内出答案的路径。最多重试 2 次（原始 + 1 次重试）。
        resp = None
        template_leak_retry = False  # 标记是否为模板泄露后的重试
        for retry in range(2):
            # 2026-09-13 循环时间检查（防卡死）：单题上限已从 1200s 放开到
            # 3600s，必须在循环边界查时间，否则"单次 LLM 200-300s × 多次重试"
            # 会悄悄穿掉整题预算（实测 3_solve 曾超预算 1.7×）。
            # 只对"重试"生效（retry>0），保证首轮一定被尝试。
            if retry > 0 and ctx.gen_time_up():
                self.record(ctx, "solver",
                            f"候选 {cid} 生成重试因时间到点提前停止")
                break
            current_temp = base_temp
            current_system = system_prompt
            # 模板泄露重试时使用简化prompt（不覆盖）
            if template_leak_retry:
                current_user = f"请直接解答以下数学问题，只输出解答过程和最终答案：\n\n{ctx.problem}"
                current_system = _REINFORCED_SYSTEM
                current_temp = max(self.config.policy_temperature, 0.7) + 0.1 * retry
                template_leak_retry = False
            else:
                current_user = user_content + (_perturb_hints[i % len(_perturb_hints)] if retry == 0 else "")
                if retry > 0:
                    current_system = _REINFORCED_SYSTEM
                    current_temp = max(self.config.policy_temperature, 0.7) + 0.1 * retry

            # 主求解 = prefill（无完整 CoT 尝试；prefill 模式下模型输出克制，
            # max_tokens 上限 16384 已远超实测用量 ~2K token）
            resp = self._compressed_solve(
                ctx, current_system, current_user,
                temperature=current_temp,
                # 9/4：平台不限 token，去 16384 帽（截断=腰斩丢分），收敛到 policy_max_tokens
                max_tokens=self._adaptive_max_tokens(
                    ctx, self.config.policy_max_tokens),
            )
            # 空响应 -> 重试
            if resp is None or not resp.strip():
                if retry < 1:
                    logger.warning("Candidate %d empty response (retry %d/1)", cid, retry + 1)
                    time.sleep(1)
                continue
            # 模板泄露检测 → 重试
            if detect_template_leak(resp):
                if retry < 1:
                    logger.warning("Candidate %d template leak (retry %d/1)", cid, retry + 1)
                    template_leak_retry = True
                    time.sleep(0.5)
                    continue
            # 幻觉检测
            hallu = detect_hallucination(resp)
            if hallu:
                logger.warning("Candidate %d hallucination detected: %s", cid,
                               ", ".join(f"{h[0]}({h[1]:.0%})" for h in hallu))
                # 42 兜底 → 尝试重试
                if any("42" in h[0] for h in hallu) and retry < 1:
                    logger.warning("Candidate %d 42-dodge, retry", cid)
                    time.sleep(1)
                    continue
            # 截断检测 → 记录但不拒绝（后续由 orchestrator 续写）
            if detect_truncated(resp):
                logger.info("Candidate %d truncated; will attempt completion", cid)
            # 拒绝回答 -> 重试
            if _is_refusal(resp):
                if retry < 1:
                    logger.warning("Candidate %d refused to answer (retry %d/1)", cid, retry + 1)
                    time.sleep(1)
                continue
            # 有效回答
            return cid, resp, False
        # 全部重试失败，返回最后一次响应
        return cid, resp, True

    # 并行生成候选（用线程池提高吞吐，限制最大并发防止 API 过载）
    # P0-4 修复：并行度 6→2，降低并发 API 超时/限流风险
    results = []
    max_workers = min(count, 2)
    # 2026-09-13 修复（实测"单题预算被跨墙 421s"的根因）：
    # 原实现**一次性把 count(6) 个候选全部 submit**，而线程池只有 2 个 worker
    # ⇒ 6 / 2 × 单次最坏 360s（`LLMClient` 180s × 重试 1）= **最坏 1080s**；
    # 且 `as_completed` 循环内**没有任何时间检查**、已提交任务**无法取消**
    # ⇒ 生成侧软截止 `gen_time_up()` 在这一整段**完全失效**。
    # 实测后果：004 实耗 1171s 而其 `soft_budget` 只有 750s（**超 421s**）、
    # 010 超 182s、016 超 87s —— 每题都越过自己的软预算。
    # 改为**分批提交 + 批间检查**：每批 `max_workers` 个，批间查生成侧截止，
    # 到点即停止提交剩余候选；**已产出的候选照常进入后续验证**（既有语义不变）。
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        _pending = list(range(count))
        while _pending:
            if results and ctx.gen_time_up():
                self.record(ctx, "solver",
                            f"生成侧时间到，跳过剩余 {len(_pending)} 个候选"
                            f"（已完成 {len(results)}/{count}）")
                break
            _batch, _pending = _pending[:max_workers], _pending[max_workers:]
            _futs = [pool.submit(_make_one, i) for i in _batch]
            for _f in as_completed(_futs):
                results.append(_f.result())
    results.sort(key=lambda x: x[0])
    for cid, resp, is_fallback in results:
        _store_candidate_result(self, ctx, cid, resp, is_fallback)


def _store_candidate_result(self, ctx: TaskContext, cid, resp,
                            is_fallback: bool) -> None:
    """段 15–16：单条采样结果落盘（失败抢救 / 正常抽取+三通道+入池门槛）。"""
    # 惰性 import：_is_refusal 是 solver.py 模块级符号，避免顶层循环导入。
    from .solver import _is_refusal
    if resp is None or (is_fallback and _is_refusal(resp)):
        # 2026-09-13 晚（用户硬要求「无论超没超时都要把答案生成出来」）：
        # 原实现在此 append 一个
        #   Candidate(id=cid, answer="",
        #             reasoning="[生成失败] 调用受限或模型拒绝回答")
        # 的**占位候选**，造成两个后果（4 题实测里 010/016 全中）：
        #   ① `ctx.candidates` 因此**非空** ⇒ orchestrator 的
        #      「Solver 未产出候选 → 触发兜底直接求解」（orchestrator.py:1158）
        #      永不触发；`_emergency_direct_solve` 那一整条路被绕开；
        #   ② formatter 的 `_REFUSAL_RE` 旧版不含这几个词 ⇒ 占位文本被当作
        #      合法答案直接提交，`predicted` 就是这串占位符。
        # 现改为：先尽力从残缺响应里抢救一个真实答案；抢救不到就**不 append**，
        # 让 `ctx.candidates` 保持空，把控制权交给下游兜底链
        # （orchestrator `_fallback_direct` → formatter `_emergency_answer`
        #  → user_agent 最终兜底）。
        _salvaged = ""
        if resp is not None and str(resp).strip():
            _txt = str(resp)
            try:
                _salvaged = extract_final_answer(_txt) or ""
                if not _salvaged and rescue_final_answer is not None:
                    _salvaged = rescue_final_answer(_txt)[0] or ""
                if not _salvaged and smart_fallback_answer is not None:
                    _salvaged = smart_fallback_answer(_txt) or ""
            except Exception:  # noqa: BLE001
                _salvaged = ""
        _salvaged = (_salvaged or "").strip()
        if _salvaged and not _is_refusal(_salvaged) and len(_salvaged) <= 300:
            # ★★ 2026-10-07（2.4 v2 入口穷举 · 入口②）：**"生成失败抢救"
            #   这条 append 此前不过门槛** —— 抢救函数（extract / rescue /
            #   smart_fallback）本身就可能吐回推理句/元话语，于是"抢救"反倒
            #   成了非答案入池的后门。此处同口径复核：不合格则不入池，
            #   保持池干净、把控制权交给下游兜底链。
            _sans, _sok = self._normalize_and_check_answer(
                ctx, cid, _salvaged)
            if _sok:
                ctx.candidates.append(Candidate(
                    id=cid, answer=_sans, reasoning=str(resp)))
                logger.warning(
                    "Candidate %d 生成失败但抢救到答案: %s",
                    cid, _sans[:60])
            else:
                logger.warning(
                    "Candidate %d 生成失败，抢救结果仍非答案 → 不入池"
                    "（交由下游兜底）", cid)
        else:
            logger.warning(
                "Candidate %d generation failed/skipped（无可用答案，"
                "交由下游兜底）", cid)
        return
    answer = extract_final_answer(resp)
    # 如果提取不到答案 / 答案过长（>300字符大概率是推理文本），
    # 先试 rescue 兜底（嵌套 boxed / 中段强模式结论），再取尾部
    if not answer or len(answer) > 300 and resp.strip():
        rescued = rescue_final_answer(resp)[0]
        if rescued:
            answer = rescued
        else:
            fallback = smart_fallback_answer(resp)
            if fallback and (not answer or len(fallback) < len(answer)):
                answer = fallback
    # 2026-08-30（#51）：上述兜底仍拿到"疑似推理文本"时，做一次**定向重问**，
    # 而不是把长文本当答案交给判分器。
    # 依据：基线 45 题有 5 题抽出的答案 >60 字符（含整段计算步骤与结论句），
    # 本地宽松判分能看懂、平台判分看不懂。模型具备给出简洁答案的能力，
    # 缺的是一次明确要求——成本仅一次短调用，收益是消除平台侧的格式性丢分。
    if (getattr(self.config, 'enable_answer_reask', True)
            and not ctx.gen_time_up()):
        answer = self._reask_final_answer(ctx, resp, answer)
    # ★ 表达式范式（默认形态，9/12 用户要求）：模型只建模（设符号+给算式），
    # 数值代入与运算由本地完成 → 答案取工具值。
    resp, answer = self._maybe_expression_eval(ctx, resp, answer)
    # 2026-09-10 L2：独立符号建模复核（默认关；每题最多一次）
    resp, answer = self._maybe_symbolic_crosscheck(ctx, resp, answer)
    # 2026-09-12 符号化方程求解通道（默认关）：模型只交方程（组）+ 目标，
    # 数值一律由本地 SymPy 回代求出（模型不参与计算），异议时回传工具结果。
    resp, answer = self._maybe_symbolic_solve(ctx, resp, answer)
    # ★★ 2026-10-06：**入池门槛**（规范化 + 判定）——
    #   不含最终答案的候选不入池；句子形态但含明确最终值的**抢救出值再入池**
    #   （实测 59% 的候选是推理句/元话语；它们污染投票/检错/修正）。
    answer, _ans_ok = self._normalize_and_check_answer(ctx, cid, answer)
    if _ans_ok:
        ctx.candidates.append(Candidate(
            id=cid,
            answer=answer,
            reasoning=resp,
            revised=False,
        ))
    logger.debug("Candidate %d generated (len=%d)", cid, len(resp))
