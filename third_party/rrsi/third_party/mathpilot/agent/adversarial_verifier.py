# -*- coding: utf-8 -*-
"""对抗式验证器（#16 改造）：**证伪**而非验证。

为什么要单独做这个
--------------------
基线 45 题暴露了两个互相矛盾的问题：

1. **误杀**：两层复核一致性仅 51%，单层过、双层不过的有 22 题，
   其中 7 题最终判对（误杀率 31.8%），且**反向案例为 0 条** ——
   说明第二层不是"换个角度看"，只是在第一层基础上加严。
2. **漏检**：正向验证顺着作者思路走（确认偏误），错误没被审出来。

这两件事的**共同解**是换一种认知模式：正向验证问"这对吗"，
对抗式验证问"**假设它是错的，错在哪**"。后者在方法论上是证伪
（波普尔），在数学上对应反例法，比"确认正确"更能激发批判性思维。

与现有 Step 4（`Orchestrator._review_bug_feedback`）的分工
--------------------------------------------------------
- Step 4（已有）：验证器**给出负面反馈后**，复核是否误报 → 治误杀
- 本模块（新增）：验证器**判定通过后**，主动去找错     → 治漏检

两者互补，缺一则另一侧的漏洞仍在。

仲裁规则（由 orchestrator 组合使用）::

    正向通过 + 逆向无反例  → 高置信接受（不再被第二层无脑否掉）
    正向通过 + 逆向有反例  → 送 revise（抓漏检）
    正向不通过            → 直接 revise（无需证伪，省一次调用）

设计原则：本模块只做**探测**，不做决策。任何异常一律吞掉并返回
``found=False``——验证器的问题绝不能阻断主流程。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from .base import BaseAgent, TaskContext

logger = logging.getLogger("MathPilot")

# 6 类数学错误 checklist —— 从 45 题真实错误样本提炼，每条都有对应真题：
#   sign_error          algebra-064：答案 −3 写成 3
#   skipped_step        geometry-068："Solving yields s=187/228" 直接跳到结论
#   boundary_missed     number_theory-022：答"所有正奇数"，实际是 {1,3,5}
#   special_case_missed algebra-003：只分析线性解，错误排除三次多项式解
#   calculation_error   geometry-051：121/125 + 25 = 24（应为 25.968）
#   misread_requirement combinatorics-040：给了含 N 的表达式，题目要求数值
ERROR_CHECKLIST = (
    ("sign_error", "符号错误：正负号、不等号方向、分子分母颠倒"),
    ("skipped_step", "跳步：直接给出关键结论但缺少推导过程"),
    ("boundary_missed", "边界遗漏：定义域、退化情形、端点、特殊情况未讨论"),
    ("special_case_missed", "情形遗漏：只分析了一部分情形（如只考虑线性解）"),
    ("calculation_error", "计算错误：算术或代数运算出错"),
    ("misread_requirement", "题意偏差：答非所问，或答案形式不符合题目要求"),
)

_ERROR_TYPES = tuple(k for k, _ in ERROR_CHECKLIST)

# ★ 2026-10-20（用户定案）：「过程不完整」≠「答案错」。
#   以下三类只说明论证写得简略 / 覆盖不全 / 答案形式不符，**不构成答案错误**，
#   不得作为淘汰候选的依据 —— 降级为软信号（留痕，可作重写提示），不淘汰。
#   依据（p0b5_r5，2026-10-20）：official112-019 池中 8 个候选去重后答案只有 4
#   （= gold），却因 skipped_step / misread_requirement 被累计拒 9 次 ⇒ 池空 ⇒
#   输出「未给出有效解答」。同一模式在 015（候选 2）、008（候选 603729）上复现。
#   ⚠ prompt 侧的同类硬约束（2026-10-17）早已存在，但模型仍会回这些类型 ——
#     契约必须由代码兑现，不能靠提示词自觉。
SOFT_ERROR_TYPES = frozenset({
    "skipped_step", "boundary_missed", "misread_requirement",
})

# 模型常常用中文或自造词回类型（实测出现过"计算错误"），
# 归一化不了就会污染 #16 的错误类型统计，故给一份中英别名表兜底。
_ERROR_ALIASES = (
    ("符号", "sign_error"), ("正负", "sign_error"), ("不等号", "sign_error"),
    ("跳步", "skipped_step"), ("省略", "skipped_step"), ("缺.*推导", "skipped_step"),
    ("未推导", "skipped_step"),
    ("边界", "boundary_missed"), ("定义域", "boundary_missed"),
    ("退化", "boundary_missed"), ("端点", "boundary_missed"),
    ("范围", "boundary_missed"), ("遗漏", "boundary_missed"),
    ("情形", "special_case_missed"), ("特殊", "special_case_missed"),
    ("分类", "special_case_missed"), ("排除", "special_case_missed"),
    ("计算", "calculation_error"), ("算术", "calculation_error"),
    ("运算", "calculation_error"),
    ("题意", "misread_requirement"), ("答非所问", "misread_requirement"),
    ("形式不符", "misread_requirement"), ("不符合.*要求", "misread_requirement"),
    ("未理解", "misread_requirement"), ("理解.*题目", "misread_requirement"),
    ("答非", "misread_requirement"),
)

_ADVERSARY_SYSTEM = (
    "你是数学解答的**对抗式审查者**。\n"
    "你的任务不是确认答案正确，而是**尽力证明它是错的**。\n"
    "默认立场：给定的答案是错误的，你要找出它错在哪里。\n"
    "只有当你用尽方法仍找不到任何错误时，才承认它可能对。\n"
    "你是严谨的对手，不是附和者——但也不能凭空捏造错误。\n\n"
    "⚠ 硬约束（2026-10-17 用户定案：4.2 是「对抗证伪」，证伪=找到具体错误）：\n"
    "**「缺少推导 / 跳步 / 解答简略 / 未给出证明过程」不构成错误**，"
    "不得作为判 found=true 的理由；\n"
    "只有能指出**具体的数学错误**时才判 found=true，即至少满足其一：\n"
    "  · 某一步的推导/不等式/等式在数学上不成立（可指出错在哪）；\n"
    "  · 最终结论与题面条件矛盾；\n"
    "  · 能给出满足题面但使结论失败的具体反例；\n"
    "  · 计算确实算错（给出正确值）。\n"
    "若只是看不出它怎么推出来的、或觉得不够详细，一律判 found=false。\n"
    "★ 判断标准不是「一份完整证明通常长什么样」，而是**本题的义务清单**：\n"
    "  清单（若提供）列出本题应当建立的事实，以及哪些前提题面已给、"
    "哪些已由前序结论建立。清单之外的省略**不算缺陷**，"
    "已由前序建立的结论**不要求重复推导**。"
)

_ADVERSARY_TEMPLATE = (
    "【题目】\n{problem}\n\n"
    "{obligations}"
    "【待审查的解答】\n{reasoning}\n\n"
    "【该解答给出的最终答案】\n{answer}\n\n"
    "请按以下 {n} 类错误逐项排查，尽力找出答案的错误：\n{checklist}\n\n"
    "对每一类，判断是否存在该类型错误。若找到错误，必须给出：\n"
    "  - 出错的具体步骤（原文摘录，不要改写）\n"
    "  - 具体错在哪、正确应该是什么\n"
    "  - 若可能，给出一个反例或具体反证\n"
    "  - 这个错误发生在**哪一条义务**上（清单编号）\n"
    "⚠ 若你指不出它落在清单里的哪一条，说明你评论的可能是本题并不要求的内容，"
    "此时请把 obligation_id 填 none，不要硬套一个编号。\n\n"
    "只输出 JSON，不要任何多余文字：\n"
    '{{"found": <true 表示找到错误 / false 表示未找到>, '
    '"error_type": "<{types} 之一，未找到时为空字符串>", '
    '"error_step": "<出错步骤原文，未找到时为空字符串>", '
    '"counterexample": "<反例或反证，没有则空字符串>", '
    '"obligation_id": "<O1/O2…；不落在任何义务上填 none；无清单时填空字符串>", '
    '"confidence": <0到1之间的小数>, '
    '"reasoning": "<一句话说明你的判断依据>"}}'
)

# prefill 种子必须锚定顶层结构（Intern 系列铁律）：
# 只用 '{"' 会让模型以为已进入对象内部、丢掉外层键 → 结构非法
_PREFILL_SEED = '{"found": '

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AdversarialResult:
    """对抗式验证的探测结果。"""

    found: bool = False                 # 是否找到反例 / 错误
    error_type: str = ""                # 6 类之一（见 ERROR_CHECKLIST）
    error_step: str = ""                # 出错步骤原文
    counterexample: str = ""            # 反例或反证
    confidence: float = 0.0             # 模型自评置信度
    reasoning: str = ""                 # 判断依据
    raw: str = ""                       # 原始返回（排障用）
    parsed: bool = False                # JSON 是否被成功解析
    skipped: str = ""                   # 跳过原因（空 = 正常执行）
    # ★ 2026-10-20（用户：「把蓝图倒推过程反过来输出给 4 部分」）：
    #   `obligation_id` = 审查者指认的「该错误落在哪一条义务上」（O1/O2…，或 "none"）。
    #   `obligation_mapped` = 该引用能否在账本里找到（None=本次未用账本）。
    #   判据见 `is_actionable`：**明确宣称"不落在任何义务上"**的硬类型错误不予淘汰。
    obligation_id: str = ""
    obligation_mapped: bool = False
    ledger_used: bool = False

    @property
    def obligation_note(self) -> str:
        """排障用的账本判定说明（空 = 未启用账本）。"""
        if not self.ledger_used:
            return ""
        if not self.obligation_id:
            return "账本：审查者未指认义务（不据此放宽）"
        return ("账本：指认 %s（%s）"
                % (self.obligation_id,
                   "已对上" if self.obligation_mapped else "不在账本内"))

    @property
    def is_soft(self) -> bool:
        """找到的是「过程/形式问题」（跳步、边界遗漏、题意偏差），不是答案错。

        2026-10-20：这类结果**不淘汰候选**，只留痕（名单见 SOFT_ERROR_TYPES）。
        """
        return (self.found
                and bool(self.error_step or self.counterexample)
                and self.error_type in SOFT_ERROR_TYPES)

    @property
    def is_actionable(self) -> bool:
        """是否足以触发**淘汰** —— 只认「答案错」类证据。

        2026-10-20（用户定案）：「证明写得简略」不等于「答案错」。原先只要
        ``found and (error_step or counterexample)`` 就淘汰，等于让被要求
        「尽力找错」的审查者靠「跳步」把正确答案判死（p0b5_r5：15/49 条
        淘汰记录打中的是 gold 本身）。
        """
        if not (self.found and bool(self.error_step or self.counterexample)):
            return False
        if self.error_type in SOFT_ERROR_TYPES:
            return False
        # ★ 2026-10-20 义务账本门（仅硬类型、且本次真的注入了账本时生效）：
        #   审查者若**明确宣称**该错误不落在本题任何一条义务上（obligation_id
        #   填 none / 填了账本里不存在的编号），说明它评论的是蓝图并未要求的东西
        #   —— 这正是「证更强的命题」型误杀，不予淘汰。
        #   ⚠ 只惩罚「明确指认失败」，**不惩罚漏填**（漏填按原行为淘汰）：
        #     宁可漏杀，不可误伤（与 low-confidence 的处理口径一致）。
        if (self.ledger_used and self.obligation_id
                and not self.obligation_mapped):
            return False
        return True

    def to_feedback(self) -> str:
        """转成可注入 revise 通道的反馈文本（软信号也要有文本，供重写提示）。"""
        if not (self.is_actionable or self.is_soft):
            return ""
        label = dict(ERROR_CHECKLIST).get(self.error_type, self.error_type or "未知类型")
        parts = [f"[对抗式审查] 检出「{label}」"]
        if self.error_step:
            parts.append(f"出错步骤：{self.error_step[:300]}")
        if self.counterexample:
            parts.append(f"反例/反证：{self.counterexample[:300]}")
        if self.reasoning:
            parts.append(f"依据：{self.reasoning[:300]}")
        return "\n".join(parts)


class AdversarialVerifier(BaseAgent):
    """对抗式验证器：假设答案错误，反向寻找反例或第一个错误步骤。"""

    def __init__(self, client, config):
        # 注意：BaseAgent.__init__ 只接 (client, config)，没有 budget 形参
        super().__init__(client, config)
        self._enabled = bool(getattr(config, "enable_adversarial_verify", True))
        self._tiers = tuple(getattr(config, "adversarial_tiers", None) or
                            ("deep", "standard"))
        self._min_conf = float(getattr(config, "adversarial_min_confidence", 0.5))
        self._max_reasoning = int(getattr(config, "adversarial_max_reasoning", 2400))
        # ★ 2026-10-20：是否把蓝图倒推链反演成「义务账本」注入审查（见
        #   `agent/obligation_ledger.py`）。=0 退回「只喂题目/解答/答案」的老口径。
        self._use_ledger = bool(
            getattr(config, "audit_obligation_ledger", True))

    # ------------------------------------------------------------------
    # BaseAgent 接口（pipeline 形态入口）
    # ------------------------------------------------------------------
    def run(self, ctx: TaskContext, tier: str = "standard") -> TaskContext:
        """对最佳候选做证伪探测，把结果写入 ``ctx.adversarial_result``。

        命中错误时同时把反馈追加进 ``ctx.audit_reject_feedback``
        （2026-09-06 更名；原 lean_reject_feedback），由 orchestrator 的
        revise 通道消费（与 AuditGate/Oracle 客观反馈同一入口）。
        """
        cand = None
        bc = getattr(ctx, "_best_cluster", None)
        if bc is not None and getattr(bc, "rep_candidate", None) is not None:
            cand = bc.rep_candidate
        if cand is None and getattr(ctx, "candidates", None):
            cand = ctx.candidates[0]
        if cand is None:
            return ctx

        result = self.probe(ctx, cand, tier=tier)
        ctx.adversarial_result = result
        if result.is_actionable:
            fb = getattr(ctx, "audit_reject_feedback", None) or []
            fb.append(result.to_feedback())
            ctx.audit_reject_feedback = fb
        self.record(ctx, "adversarial",
                    ("检出 %s" % (result.error_type or "未知类型"))
                    if result.is_actionable else "未找到错误",
                    adv_found=result.is_actionable,
                    adv_error_type=result.error_type,
                    adv_confidence=result.confidence)
        return ctx

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def probe(self, ctx: TaskContext, candidate,
              tier: str = "standard") -> AdversarialResult:
        """对单个候选做证伪探测；返回 AdversarialResult。

        任何异常都被吞掉并返回 ``found=False`` —— 验证器出问题
        只意味着"这次没抓到错"，不应阻断主流程。
        """
        if not self._enabled:
            return AdversarialResult(skipped="disabled")
        if tier not in self._tiers:
            return AdversarialResult(skipped=f"tier_not_covered:{tier}")

        reasoning = str(getattr(candidate, "reasoning", "") or "")
        answer = str(getattr(candidate, "answer", "") or "")
        if not reasoning.strip():
            return AdversarialResult(skipped="empty_reasoning")

        ledger, ledger_text = None, ""
        if self._use_ledger:
            try:
                from .obligation_ledger import build_ledger, render_for_audit
                ledger = build_ledger(getattr(ctx, "blueprint", None))
                ledger_text = render_for_audit(ledger)
            except Exception as _e_l:  # noqa: BLE001  账本失败绝不阻断审查
                logger.debug("[adversarial] 义务账本构建失败（退回无账本口径）: %s",
                             str(_e_l)[:120])
                ledger, ledger_text = None, ""

        try:
            raw = self._call_probe(ctx, reasoning, answer, ledger_text=ledger_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[adversarial] 对抗式审查调用失败（降级为未找到）: %s",
                           str(exc)[:120])
            return AdversarialResult(skipped=f"call_failed:{type(exc).__name__}")

        if not raw:
            return AdversarialResult(skipped="empty_response")

        result = self._parse(raw)
        if ledger_text:
            from .obligation_ledger import has_obligation
            result.ledger_used = True
            result.obligation_mapped = has_obligation(ledger, result.obligation_id)
        # 低置信度的"找到"不采信：宁可漏掉，不可误伤
        if result.found and result.confidence < self._min_conf:
            logger.debug("[adversarial] 检出但置信度 %.2f < %.2f，不采信",
                         result.confidence, self._min_conf)
            result.found = False
        return result

    # ------------------------------------------------------------------
    # 内部：调用与解析
    # ------------------------------------------------------------------
    def _call_probe(self, ctx: TaskContext, reasoning: str, answer: str,
                    ledger_text: str = "") -> str:
        """调用模型做对抗式审查（走 prefill 压缩，避免 CoT 吃满 max_tokens）。

        ``ledger_text``：蓝图倒推链反演出的义务清单（空 = 本题无蓝图 / 开关关闭，
        此时 prompt 与历史口径完全一致）。
        """
        # 2026-09-04 修复：utils.llm 不存在（曾致 prefill 从未生效、审查静默降级）
        from utils.prefill import prefill_messages

        checklist = "\n".join(
            f"{i}. {name}：{desc}" for i, (name, desc) in enumerate(ERROR_CHECKLIST, 1))
        _lblock = (ledger_text.rstrip() + "\n\n") if ledger_text else ""
        user = _ADVERSARY_TEMPLATE.format(
            problem=str(getattr(ctx, "problem", ""))[:2000],
            obligations=_lblock,
            reasoning=reasoning[-self._max_reasoning:],
            answer=answer[:300],
            n=len(ERROR_CHECKLIST),
            checklist=checklist,
            types=" / ".join(_ERROR_TYPES),
        )
        msgs = prefill_messages(
            [{"role": "system", "content": _ADVERSARY_SYSTEM},
             {"role": "user", "content": user}],
            _PREFILL_SEED,
        )
        raw = self.llm(ctx, msgs, temperature=0.0,
                       # 2026-10-02 DeepSeek 适配：兜底值 640 ⇒ 8192，与 config 默认同步。
                       max_tokens=int(getattr(self.config, "adversarial_max_tokens", 8192)))
        if not raw:
            return ""
        # prefill 的种子不在返回里，解析前补回去
        return _PREFILL_SEED + raw if not raw.lstrip().startswith("{") else raw

    @staticmethod
    def _try_load(text: str) -> Optional[dict]:
        """尝试从文本里解析出 JSON 对象；失败返回 None。"""
        m = _JSON_RE.search(text or "")
        if not m:
            return None
        frag = m.group(0)
        # 先整体试；不行就逐步回退到最后一个 '}'（尾部有多余内容的情况）
        for cut in range(len(frag), 0, -1):
            if frag[cut - 1] != "}":
                continue
            try:
                obj = json.loads(frag[:cut])
            except Exception:  # noqa: BLE001
                continue
            return obj if isinstance(obj, dict) else None
        return None

    @staticmethod
    def _strip_truncated_tail(text: str) -> str:
        """剥掉末尾被截断的片段并补 '}'，抢救截断输出。

        ``{"found": true, "error_type": "sign_error", "error_step``
        → ``{"found": true, "error_type": "sign_error"}``

        实现：只从**合法的截断边界**（引号 / 逗号）处回退尝试，
        而不是逐字符——否则会把已完整的 `"sign_error"` 值也削掉半截引号。
        """
        frag = (text or "").strip()
        i = frag.find("{")
        if i < 0:
            return ""
        frag = frag[i:]
        for cut in range(len(frag), 0, -1):
            if frag[cut - 1] not in ('"', "'", ",", "，", " "):
                continue
            cand = frag[:cut].rstrip().rstrip(",，") + "}"
            if AdversarialVerifier._try_load(cand):
                return cand
        return ""

    @staticmethod
    def _parse(raw: str) -> AdversarialResult:
        """解析模型输出；解析失败返回 found=False（不可信就当没抓到）。"""
        text = (raw or "").strip()
        data = AdversarialVerifier._try_load(text)
        if data is None:
            # 截断/脏输出：剥掉末尾未闭合的键再补 '}' 重试一次
            data = AdversarialVerifier._try_load(
                AdversarialVerifier._strip_truncated_tail(text))
        if data is None:
            return AdversarialResult(raw=text, parsed=False, skipped="parse_failed")

        raw_type = str(data.get("error_type") or "").strip()
        etype = raw_type
        if etype and etype not in _ERROR_TYPES:
            # 模型自造类型（含中文）：先按英文键名子串匹配，再按中文别名表匹配；
            # 都匹配不上就置空——宁可丢掉类型标注，也不能让脏值污染 #16 的统计。
            # 注意：两轮匹配都必须基于原始 raw_type，用 etype 会被上一轮的空结果污染。
            low = raw_type.lower()
            etype = next((k for k in _ERROR_TYPES if k in low), "")
            if not etype:
                etype = next((v for pat, v in _ERROR_ALIASES
                              if re.search(pat, raw_type)), "")

        try:
            conf = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0

        return AdversarialResult(
            found=bool(data.get("found")),
            error_type=etype,
            error_step=str(data.get("error_step") or "").strip(),
            counterexample=str(data.get("counterexample") or "").strip(),
            confidence=max(0.0, min(1.0, conf)),
            reasoning=str(data.get("reasoning") or "").strip(),
            obligation_id=str(data.get("obligation_id") or "").strip(),
            raw=text,
            parsed=True,
        )
