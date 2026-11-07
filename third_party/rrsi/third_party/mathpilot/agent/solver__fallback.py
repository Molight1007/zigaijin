# -*- coding: utf-8 -*-
"""兜底直解 / 截断续写 / 无条件自改进 / 定向重问（2026-10-08 拆分：从 agent/solver.py 整段同迁）。

  · ``_answer_looks_suspicious`` —— 疑似推理文本判定（原 ``@classmethod``）；
  · ``_reask_final_answer`` —— 答案疑似推理文本时的定向重问；
  · ``_compressed_solve`` —— prefill 压缩求解原语（本模块与已抽模块共用）；
  · ``improve_candidates`` —— Step2 无条件自改进公开入口（多轮）；
  · ``direct_solve`` —— 兜底直接求解（last-resort）；
  · ``complete_truncated_candidates`` —— 截断候选批量续写；
  · ``complete_answer`` —— 单候选续写 + 答案前置重问。

搬运方式：**整段同迁**（``def`` 行 + 函数体逐字搬，仅去 4 格缩进；``@classmethod``
装饰器留在 solver.py 的**一行委托方法**上）。新增的只有：``self`` / ``cls`` 作首参 +
``.base`` / ``utils.extract`` / ``utils.prefill`` 的导入 + ``direct_solve`` 内一处**惰性
import**（``_REINFORCED_SYSTEM`` / ``_is_refusal`` 仍是 solver.py 模块级符号，惰性 import
避免循环导入）。``solver.py`` 保留同名**一行委托方法**，对外签名与返回值逐字不变
（``SolverAgent._answer_looks_suspicious`` 仍可按类方法直接调用）。类属性
``_SUSPICIOUS_ANSWER_LEN`` / ``_NARRATIVE_PAT`` / ``_COMPLETE_SYS`` / ``_ANSWER_PREFIX_SYS``
仍留在 ``SolverAgent``，本模块经 ``self.`` / ``cls.`` 访问。
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext

from .base import Candidate, detect_hallucination, detect_truncated
from utils.extract import (
    extract_final_answer, rescue_final_answer, smart_fallback_answer,
)
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot")


def _answer_looks_suspicious(cls, answer: str) -> bool:
    """判断抽取出的答案是否"疑似推理文本"而非答案本身。

        判定按「长度 → 句式 → 结构」三级，且刻意保守：
        **误判的代价是一次短调用**，但把合法答案判成可疑会导致重问，
        所以 `{1, 3, 5}`、`x = 2, y = 3` 这类列表/多值答案必须放过。
        """
    a = (answer or "").strip()
    if not a:
        return True
    # 1) 过长：答案是短语，不是段落
    if len(a) > cls._SUSPICIOUS_ANSWER_LEN:
        return True
    # 2) 成句：出现句号或推理连接词，说明抽到的是叙述而非结论
    if "。" in a or "．" in a:
        return True
    for pat in cls._NARRATIVE_PAT:
        if re.search(pat, a):
            return True
    # 3) 推导链：**链式等号** `a = b = c`（两个等号之间没有被逗号分隔）。
    #    用"中间无逗号"把推导链与并列赋值区分开：
    #      - "S = 1 + 2 + 3 = 6"  → 链式，是计算过程
    #      - "x = 2, y = 3"       → 并列，是合法的多值答案
    if re.search(r"=[^,，]*=", a):
        return True
    return False


def _reask_final_answer(self, ctx: TaskContext, reasoning: str,
                        answer: str) -> str:
    """答案疑似推理文本时，向模型定向重问一次"仅输出最终答案"。

        只在**抽取结果不可信**时触发（空 / 超长 / 含多步推导痕迹），
        正常答案直接原样返回，不增加任何开销。

        失败一律返回原答案——兜底动作不能让情况变得更糟。
        """
    if not self._answer_looks_suspicious(answer):
        return answer
    try:
        # 只喂推理尾部，避免长上下文拖慢这次短调用
        tail = reasoning[-1200:] if len(reasoning) > 1200 else reasoning
        system = (
            "你是数学答案格式化助手。只输出最终答案，不要解释、不要推导、"
            "不要任何多余文字。"
        )
        user = (
            "下面是某题的解答过程（可能不完整）。\n\n"
            f"{tail}\n\n"
            "请只输出这道题的最终答案，满足：\n"
            "1) 用 \\boxed{...} 包裹，例如 \\boxed{42}\n"
            "2) 下一行给出不含公式标记的最简形式，例如：最简形式：42\n"
            "3) 不要输出推导过程、单位说明或任何解释性文字\n"
            "4) 若答案是多个值，用逗号分隔放在同一个 \\boxed{} 内"
        )
        raw = self._compressed_solve(
            ctx, system, user,
            temperature=0.0,
            # 2026-10-02 DeepSeek 适配：原 256 ⇒ 8192。⚠ 该键全仓无 config 声明，
            # 故此处 getattr 兜底即为唯一默认值。原值假设「重问只需一行最简形式」，
            # 对 reasoning 模型必然截断（reasoning 先吃满预算、正文为空）。
            max_tokens=int(getattr(self.config, 'answer_reask_max_tokens', 8192)),
        )
        if not raw:
            return answer
        new_ans = extract_final_answer(raw)
        if not new_ans:
            return answer
        # 重问结果必须"比原来更像答案"才采纳
        if self._answer_looks_suspicious(new_ans):
            return answer
        self.record(ctx, "answer_reask",
                    f"答案疑似推理文本（{len(answer)} 字符），定向重问后收敛为 "
                    f"{len(new_ans)} 字符")
        return new_ans
    except Exception as exc:  # noqa: BLE001
        logger.debug("[solver] 答案定向重问失败，保留原答案: %s", str(exc)[:120])
        return answer


def _compressed_solve(self, ctx: TaskContext, system: str, user: str,
                      temperature: float = 0.1,
                      max_tokens: int = 8192,
                      prefill_seed: str = "## 问题分析\n") -> str | None:
    """ICMA 同款压缩求解：prefill 种子抑制 CoT，快速产出答案。

        v2.4.1 起为本环境**主求解路径**：诊断实测完整 CoT 单次调用 >200s 不返回
        （780s 仍读超时），而 prefill 压缩求解 36.7s 即返回 ~2000 tokens 结构化解答。
        prefill 答案前置：即使输出被截断，也只损失思考、不损失答案。

        2026-09-04：prefill_seed 改为可配参数（默认 "## 问题分析\n" 适配主求解/证明
        四章节格式；revise 传 "【错误分析】\n"、self-improve 传 "【第一步：诊断】\n"
        与各自 system 要求的输出开头对齐，避免格式错位）。
        """
    try:
        msgs = prefill_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            prefill_seed,
        )
        # 2026-09-09 试点起：满足工具循环条件（enable_web_search）走原生工具循环
        resp = self._maybe_tool_llm(ctx, msgs, temperature, max_tokens)
        if resp:
            return stitch(prefill_seed, resp)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Compressed solve failed: %s", e)
        return None


def improve_candidates(self, ctx: TaskContext) -> int:
    """对已有候选做 review+improve（论文流水线 Step 2），可多轮。

        论文（Huang & Yang 2025）观测：初始解质量普遍低，Step 2 给模型
        注入第二段推理预算后输出显著改进。与 revise 的关键区别：
        revise 是**验证失败才修正**（有条件），自改进是**无条件先做一遍**。

        2026-09-29 用户决策（截图 #7）：
          · **恢复真无条件** —— `self_improve_conditional` 默认 False，
            即默认不对候选做「看起来有问题才改」的过滤；
          · **遍数可配** —— `self_improve_rounds` 默认 1，可设 2 做 A/B。
            第 N 轮的输入是第 N-1 轮的产物（对同一批候选再注入一次推理预算）。

        返回本次累计改进成功的候选次数（跨轮累加）。
        """
    rounds = max(1, int(getattr(self.config, "self_improve_rounds", 1) or 1))
    total = 0
    for r in range(rounds):
        # 2026-09-29：每轮开始前重置 `self_improved` 标记 —— 否则第 2 轮
        # 会被第 1 轮设下的标记全部过滤掉（`not self_improved` 恒假 → 空转）。
        # 仅在第 2 轮及以后重置，避免影响其它调用方对标记的语义预期。
        if r > 0:
            for _c in ctx.candidates:
                try:
                    _c.self_improved = False
                except Exception:  # noqa: BLE001
                    pass
            self.record(ctx, "self_improve",
                        f"Step2 第 {r + 1}/{rounds} 轮开始（重置改进标记）")
        n = self._improve_candidates_once(ctx)
        total += n
        if n == 0:
            # 本轮无候选可改进 ⇒ 后续轮次必然同样空转，提前收敛省一次遍历。
            self.record(ctx, "self_improve",
                        f"Step2 第 {r + 1}/{rounds} 轮无改进产出 → 停止")
            break
    if total:
        self.record(ctx, "self_improve",
                    f"Step2 自改进累计 {total} 个候选（{rounds} 轮上限）")
    # ★ 2026-10-12 观测补齐（判读失真 #5 的**治本**修法）：
    #   `cli/test_report.py` 一直读 `diag.n_self_improved`，但该键
    #   **全仓无人写入** ⇒ 报告里 3.3_improve 恒显示 improved=0、
    #   被 `_pass_for` 判成「未触发」，把**正在工作的自改进报成没跑**。
    #   这里改为**落结构化字段**（累计，因为 3.3 与 revise 后可能各调一次）：
    #   · n_self_improved      —— 本题累计被改进成功的候选次数
    #   · self_improve_rounds_used —— 实际消耗的轮数（含 total=0 的情况，
    #     用于区分「没跑」与「跑了但没改进成功」）
    try:
        ctx.n_self_improved = int(
            getattr(ctx, "n_self_improved", 0) or 0) + total
        ctx.self_improve_rounds_used = int(
            getattr(ctx, "self_improve_rounds_used", 0) or 0) + rounds
    except Exception:  # noqa: BLE001  纯观测，绝不影响主流程
        pass
    return total


def direct_solve(self, ctx: TaskContext) -> str:
    """
        用最简提示词直接求解（跳过蓝图分解和领域提示），
        要求模型必须输出【最终答案】。适用于所有多智能体候选均失败时兜底。
        返回最终答案字符串。
        """
    # 惰性 import：这两个是 solver.py 模块级符号，避免顶层循环导入。
    from .solver import _REINFORCED_SYSTEM, _is_refusal
    direct_system = _REINFORCED_SYSTEM + "\n\n请直接在【最终答案】中给出答案，不要省略任何步骤。"
    user_content = f"请仔细求解以下数学问题，必须给出确定的答案。\n\n题目：\n{ctx.problem}"
    # P0-4 修复：兜底场景用 prefill 让答案前置——时间紧时优先保答案而非推理
    # 2026-09-13 修复（截断重试提示从未生效）：消息必须**每轮现建** —— 原先
    # 循环外先建好 base_msgs，下面的截断重试只改了 user_content，实际发出的
    # 仍是 base_msgs，续写提示从未进入模型调用。

    for attempt in range(3):
        # 2026-09-13 循环时间检查（防卡死）：兜底路径是"最后防线"，故用**最
        # 宽松**口径 —— 只在**单题硬超时**时才停（而非生成侧软截止），保证
        # "无论如何先出一个答案"的兜底价值不被削弱；同时防止 3 次重试
        # （单次可达 180s+）叠加穿掉单题上限。
        if attempt > 0 and ctx.is_timed_out():
            self.record(ctx, "direct_solve", "单题已超时，停止兜底重试")
            break
        try:
            _msgs = [
                {"role": "system", "content": direct_system},
                {"role": "user", "content": user_content},
            ]
            resp = self.llm(
                ctx,
                prefill_messages(_msgs, "最终答案："),
                0.1 if attempt == 0 else 0.4,   # 首次低温，重试时提高温度
                8192,
            )
            if resp:
                resp = stitch("最终答案：", resp)
        except Exception:  # noqa: BLE001
            resp = None

        if resp and resp.strip() and not _is_refusal(resp):
            # 幻觉检测：42 兜底 → 重试
            hallu = detect_hallucination(resp)
            if any("42" in h[0] for h in hallu) and attempt < 2:
                logger.warning("Direct solve %d 42-dodge (retry)", attempt + 1)
                time.sleep(1)
                continue
            # 截断检测 → 要求模型补充
            if detect_truncated(resp):
                logger.warning("Direct solve %d truncated → retry with continuation prompt", attempt + 1)
                if attempt < 2:
                    user_content = (
                        f"上一轮你的回答被截断在：{resp[-200:]}\n\n"
                        f"请从截断处续写，给出完整的最终答案。原题目：\n{ctx.problem}"
                    )
                    self.record(ctx, "direct_solve", f"截断重试 (attempt {attempt + 2})")
                    continue
            answer = extract_final_answer(resp)
            if answer:
                self.record(ctx, "direct_solve", f"兜底直接求解成功 (attempt {attempt + 1})")
                return answer
            # 常规提取失败 → rescue 兜底（嵌套 boxed / 中段强模式）
            rescued = rescue_final_answer(resp)[0]
            if rescued:
                self.record(ctx, "direct_solve",
                           f"兜底求解成功但常规提取失败，rescue 截取 (attempt {attempt + 1})")
                return rescued
            # 仍失败，取全文作为答案
            self.record(ctx, "direct_solve",
                       f"兜底求解成功但提取失败，使用全文 (attempt {attempt + 1})")
            return smart_fallback_answer(resp)
        logger.warning("Direct solve attempt %d/3 returned empty or refused", attempt + 1)
        time.sleep(1)

    self.record(ctx, "direct_solve", "兜底直接求解失败")
    return ""


def complete_truncated_candidates(self, ctx: TaskContext, max_count: int = 1) -> int:
    """对截断的候选发起续写，把完成的候选放回列表。返回续写成功数量。

        P0-4 修复：默认只续写"最有希望恢复"的 1 个截断候选（reasoning 最长的
        即信息最全、最接近完成的），避免 3 候选 × 多轮续写耗尽单题预算 → 45 error。
        修复：此前 orchestrator 只在日志里记录 truncated，从不真正续写，
        导致 65% 被截断的候选答案丢失 → invalid。
        """
    truncated = [c for c in ctx.candidates
                 if c.reasoning and c.reasoning.strip() and detect_truncated(c.reasoning)]
    if not truncated:
        return 0
    # 选信息最全的截断候选优先续写
    truncated.sort(key=lambda c: len(c.reasoning), reverse=True)
    chosen = truncated[:max_count]

    completed = 0
    new_list = []
    for c in ctx.candidates:
        if c not in chosen:
            new_list.append(c)
            continue
        # 预算允许才续写（2026-09-06 升级 gen_time_up）
        if not ctx.gen_time_up():
            new_c = self.complete_answer(ctx, c)
            if new_c is not c and new_c.answer and len(new_c.answer) > 1:
                # ★★ 2026-10-07（2.4 v2 入口穷举 · 入口⑤）：**截断续写产物
                #   此前不过门槛** —— 续写/答案前置重问走的是
                #   `extract/rescue/smart_fallback` 三条抢救函数，同样可能吐回
                #   推理句。不合格 ⇒ **保留原候选 c**（它入池时已过门槛），
                #   不用"续写出的非答案"替换。
                _cans, _cok = self._normalize_and_check_answer(ctx, c.id, new_c.answer)
                if _cok:
                    new_c.answer = _cans
                    new_list.append(new_c)
                    completed += 1
                    self.record(ctx, "complete", f"候选 #{c.id} 截断续写成功")
                    continue
                self.record(ctx, "complete",
                            f"候选 #{c.id} 续写结果不是最终答案 → 保留原候选")
        new_list.append(c)
    ctx.candidates = new_list
    return completed


def complete_answer(self, ctx: TaskContext, candidate: Candidate) -> Candidate:
    """
        对不完整的推理进行续写。

        P0-2/P0-3 强化：先尝试续写推理；若仍提取不到答案，
        再用【答案前置】紧急重问（prefill 精神：答案在前，截断不丢）。
        """
    if not candidate.reasoning or not candidate.reasoning.strip():
        return candidate

    reasoning = candidate.reasoning.strip()
    # 取尾部 1500 字符作为续写上下文
    context_tail = reasoning[-1500:]

    continue_prompt = (
        "你的推理在下面中断了，请直接从断点处继续完成推理，"
        "并在最后给出【最终答案】。不要重复之前的内容，直接接着写：\n\n"
        f"--- 断点 ---\n{context_tail}\n--- 请继续 ---"
    )

    # P0-4 修复：续写仅 1 次（2+1→1+1），压缩调用链防超时
    for attempt in range(1):
        try:
            # v2.4.1：续写也走 prefill（断点种子抑制 CoT，防 4096 token 超时）
            continuation = self.llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": self._COMPLETE_SYS},
                        {"role": "user", "content": continue_prompt},
                    ],
                    "--- 请继续 ---\n",
                ),
                0.2,
                # ★ 2026-10-02：4096→8192（DeepSeek 适配方案 A）。本处为"断点续写"，
                #   产出本身就是长推理，reasoning 模型下 4096 极易二次截断。
                8192,
            )
            if continuation:
                continuation = stitch("--- 请继续 ---\n", continuation)
        except Exception:
            continuation = None

        if continuation and continuation.strip():
            # 合并推理
            full_reasoning = reasoning + "\n\n[续写]\n" + continuation.strip()
            new_answer = extract_final_answer(full_reasoning)
            if not new_answer:
                new_answer = rescue_final_answer(full_reasoning)[0]
            if not new_answer:
                new_answer = smart_fallback_answer(continuation)
            if new_answer:
                self.record(ctx, "complete",
                           f"答案续写成功 (attempt {attempt + 1})")
                return Candidate(
                    id=candidate.id,
                    answer=new_answer,
                    reasoning=full_reasoning,
                    revised=candidate.revised,
                )
        logger.warning("Answer completion attempt %d returned empty", attempt + 1)
        time.sleep(0.5)

    # 答案前置紧急重问（P0-4：正式 prefill）：即使推理不全，也要把答案抢救出来。
    # 用 assistant 前缀"最终答案："抑制 CoT 开启，答案从开头生成——截断不丢答案。
    try:
        _prefill_msgs = prefill_messages(
            [
                {"role": "system", "content": self._ANSWER_PREFIX_SYS},
                {"role": "user",
                 "content": f"被截断的推理片段：\n{context_tail}"},
            ],
            "最终答案：",
        )
        direct = self.llm(ctx, _prefill_msgs, 0.0, 32768)
        if direct:
            direct = stitch("最终答案：", direct)
        if direct and direct.strip():
            direct_ans = smart_fallback_answer(direct)
            if direct_ans:
                self.record(ctx, "complete", "答案前置重问成功")
                return Candidate(
                    id=candidate.id,
                    answer=direct_ans,
                    reasoning=reasoning + "\n\n[答案重问]\n" + direct.strip(),
                    revised=candidate.revised,
                )
    except Exception as e:
        logger.warning("Answer prefill retry failed: %s", e)

    self.record(ctx, "complete", "答案续写失败，使用原始答案")
    return candidate
