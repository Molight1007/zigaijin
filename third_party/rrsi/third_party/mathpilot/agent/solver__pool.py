# -*- coding: utf-8 -*-
"""候选池容量 / 生成规模自适应 / 入场门槛（2026-10-08 拆分：从 agent/solver.py 整段同迁）。

  · ``_pool_room`` —— 候选池本次最多还能再补几个（``candidate_pool_limit`` 单一来源；
    原为模块级函数，整体搬迁）；
  · ``_adaptive_count`` —— 按领域 / 档位自适应候选数（原 ``@staticmethod``）；
  · ``_adaptive_max_tokens`` —— 各档统一返回 base_tokens（原 ``@staticmethod``）；
  · ``_normalize_and_check_answer`` —— 候选**入池前**答案规范化 + 有效性校验；
  · ``_answer_is_usable`` —— 旧签名的布尔包装（委托前者）。

搬运方式：**整段同迁**（``def`` 行 + 函数体逐字搬，仅去 4 格缩进；``@staticmethod``
装饰器留在 solver.py 的**一行委托方法**上）。新增的只有：``self`` 作首参（对方法而言）/
``_pool_gate_v2_on`` 从 ``.base`` 直接导入。``solver.py`` 中 ``_pool_room`` 以模块级
**re-export** 保留、其余方法保留同名委托，对外签名与返回值逐字不变。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext

from .base import _pool_gate_v2_on     # 候选入池门槛 v2 开关（单一来源，见 base.py）


# 「不限」时本次最多再补的候选数（安全上界，见 `_pool_room` docstring）
_POOL_UNLIMITED_ROOM = 12   # 「不限」时本次最多再补的候选数（安全上界，见下）


def _pool_room(cfg, ctx) -> int:
    """候选池**本次最多还能再补几个**候选（`candidate_pool_limit`，默认 12）。

    ★ 2026-10-04 修（D7）：原先本文件与 `sub_goal_solver` 共 **三处硬编码 `6`**，
    现统一到本函数（单一来源，可配）。
    同时修掉 `agent/orchestrator.py` 那处与事实不符的注释（它自称"封顶已删除、
    不再截断候选"，但三处仍在封顶）—— **口径以本函数为准**。
    ★ 2026-10-09：`candidate_pool_limit` 默认 **6 → 12**（用户拍板）⇒ 本函数
    `getattr` 兜底值同步为 12.0，与注册表 / `AgentConfig` 两处权威默认一致。

    ⚠⚠ 重要语义（2026-10-04 实测踩坑）：本返回值**同时被当作本次生成数量**
    （`run()` 里 `_generate_initial(ctx, count=remaining)`）—— 所以它**不能**是
    一个天文数字。我最初把"不限"实现成 `10**9`，结果 `count=1e9` 会让
    `_generate_initial` 尝试生成十亿个候选（`_adaptive_count` 对非证明题不收敛），
    **直接跑飞**。幸被回归测试 `test_M1_full_pool_non_revise_still_early_returns`
    抓住。故此处对"不限"设安全上界 `_POOL_UNLIMITED_ROOM`。
    """
    _lim = 0
    try:
        _v = float(getattr(cfg, "candidate_pool_limit", 12.0) or 0.0)
        _lim = int(_v) if _v > 0 else 0
    except (TypeError, ValueError):
        _lim = 0
    if _lim <= 0:
        _lim = _POOL_UNLIMITED_ROOM
    return _lim - len(getattr(ctx, "candidates", None) or [])


def _adaptive_count(ctx: TaskContext, default_count: int) -> int:
    """根据题目领域自适应调整候选数量"""
    domain = (ctx.domain or "").lower()
    # 难题深度通道：deep 档保持多候选（3 候选，不做缩减）
    if getattr(ctx, 'tier', 'standard') == 'deep':
        return default_count
    # 证明题 → 减少候选（精确推演比广度采样更重要）
    proof_keywords = ["proof", "prove", "证明", "证明题", "不等式证明", "几何证明"]
    if any(k in domain for k in proof_keywords):
        return max(1, default_count // 3)
    # P0-4 修复：不再为高难度题提高候选数——3 候选 × 3 重试曾耗尽单题
    # 300s 预算导致 45 error。保持 default_count（配置=2），省预算保产出。
    # 2026-09-12 定型前审核精简：此处原本有一段 hard_signals 判定
    # （题长 / 微分方程 / 级数 / 积分 / 客观题 → 意图给高难题更多候选），
    # 但上面的 P0-4 修复已取消"难题多给候选"策略，使该块的
    # **三个分支全部 `return default_count`** —— 计算了 12 行却零效果，
    # 属典型的"逻辑堆叠后残留"。整块删除（零行为变化）。
    return default_count


def _adaptive_max_tokens(ctx: TaskContext, base_tokens: int) -> int:
    """2026-09-04：比赛不限制模型 token → 各档统一返回 base_tokens（=policy_max_tokens=65536）。

        原分级（简单题 8192 / 难题 24576）在平台单题 1200s 时间墙下大量
        finish_reason=length 答案被腰斩（2534334 平台实测 truncated 238 次、
        64 题 invalid）——截断比多花时间更丢分。上限只作保险，prefill 压缩下
        模型实际输出仍克制；真正边界是 1200s 时间墙而非 token 帽。
        """
    return base_tokens


def _normalize_and_check_answer(self, ctx: TaskContext, cid,
                                answer) -> tuple:
    """候选**入池前**的答案规范化 + 有效性校验。

        ★ 2026-10-06 新增（由**端到端模拟**暴露的必要修正）：
        初版只做「接受 / 拒绝」，模拟里发现它会**误杀可抢救的候选** ——
        `'所以最小 d 是 48。'` 被判 False 直接丢，**但它能抢救出 `48`**。
        正确做法是**先抢救、再判定**：句子形态里含明确最终值时，
        **把 answer 换成那个值**（而不是丢候选，也不是留整句）。

        判据优先级：
          1. `_looks_like_answer(a)` 通过 ⇒ 原样可用；
          2. 否则尝试 `_salvage_value_from_line(a)` 抽出值 ⇒ **用抽出的值**；
          3. 都不行 ⇒ 判定不可用（调用方不入池）。

        返回 `(规范化后的 answer, 是否可用)`。
        开关 `require_answer_in_pool`（默认 True）；置 False 时恒返回 (原值, True) 做 A/B。
        """
    a = str(answer or "").strip()
    if not bool(getattr(self.config, "require_answer_in_pool", True)):
        return a, True
    try:
        from utils.extract import _looks_like_answer, _salvage_value_from_line
    except Exception:  # noqa: BLE001  校验不可用时放行，不阻断
        return a, True
    # ★★ 2026-10-07（问题 2.4 v2）：判据升级为 `looks_like_pool_answer`
    #   （补漏网 + 修误杀，见 `utils/extract.py` 顶部背景与实测数字）。
    #   开关 `pool_gate_v2`（默认 True）：置 0 完全回退上一轮口径
    #   （只用 `_looks_like_answer`）做 A/B。
    if _pool_gate_v2_on():
        try:
            from utils.extract import normalize_pool_answer
        except Exception:  # noqa: BLE001
            normalize_pool_answer = None
        if normalize_pool_answer is not None:
            _norm, _ok = normalize_pool_answer(a)
            if _ok:
                if _norm != a:
                    self.record(ctx, "answer_salvaged",
                                "候选#%s **从句子中抢救出答案** %r"
                                "（原句前 40 字：%r）" % (cid, _norm, a[:40]))
                return _norm, True
            self.record(ctx, "candidate_invalid",
                        "候选#%s **不含最终答案 → 不入池**（answer 前 40 字：%r）"
                        % (cid, a[:40]))
            return a, False
    if _looks_like_answer(a):
        return a, True
    _sal = _salvage_value_from_line(a)
    if _sal and _looks_like_answer(_sal):
        self.record(ctx, "answer_salvaged",
                    "候选#%s **从句子中抢救出答案** %r（原句前 40 字：%r）"
                    % (cid, _sal, a[:40]))
        return _sal, True
    self.record(ctx, "candidate_invalid",
                "候选#%s **不含最终答案 → 不入池**（answer 前 40 字：%r）"
                % (cid, a[:40]))
    return a, False


def _answer_is_usable(self, ctx: TaskContext, cid, answer) -> bool:
    """兼容旧签名的布尔包装（仅供既有调用方；新增代码请用 `_normalize_and_check_answer`）。"""
    _norm, ok = self._normalize_and_check_answer(ctx, cid, answer)
    return ok
