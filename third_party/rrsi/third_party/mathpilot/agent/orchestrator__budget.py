# -*- coding: utf-8 -*-
"""阶段预算/计时职责簇（2026-10-09 批14：从 agent/orchestrator.py 逐字搬出）。

搬出方法：_self_improve_applicable / _phase_budget / _phase_deadline_guard /
_phase_deadline_restore(@staticmethod)

**纯搬迁**：方法体逐字搬入（tokenize-aware 去缩进，多行 docstring 字符串值不变），
签名原样保留（@staticmethod 在类内委托 stub 上保留装饰器）；Orchestrator 内保留
**同名一行委托** ⇒ 调用点零改。

耦合处理：
  · 唯一外部符号 `_sw_bool`（switch_registry）⇒ 顶层同名导入；
  · `_phase_deadline_guard` 的 `import time as _t` 是**函数内局部 import** ⇒ 随体搬入；
  · C5 四方法不引用任何 orchestrator 模块级函数/常量 ⇒ 无需惰性 import。
"""
from __future__ import annotations

from agent.switch_registry import get_bool as _sw_bool

def _self_improve_applicable(self, ctx) -> bool:
    """3.3 自改进的**题型门控**（2026-09-11）。

        客观题（选择题 / 判断题 / 中文短语）**不做自改进**：
        它们没有"可改进的数学推导过程"，自改进只会白花一次 LLM 调用，
        且实测有害 —— smoke6_v3 实证：098 原答 B（正确），跑完 365s 自改进后
        变成 D（错误），而该题实为概念选择题（问"矩阵条件数定义"）。
        判据复用 Lean 适用性（客观题本就 lean_applicable=False）。
        开关：SELF_IMPROVE_OBJECTIVE_SKIP（默认 1，设 0 恢复旧行为）。
        """
    if not _sw_bool("self_improve_objective_skip"):
        return True
    try:
        m = ctx.metadata if isinstance(ctx.metadata, dict) else {}
        if m.get("lean_applicable") is False:
            self.record(ctx, "control",
                        "Step2 自改进跳过：题型为客观题（无可改进的推导过程）")
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _phase_budget(self, ctx, ratio: float = 0.0, floor: float = 180.0,
                  reserve: float = 0.0, stage: str = "") -> float:
    """阶段时间上限（秒）—— 2026-09-13 改为「实测口径」。

        ── 为什么废弃旧口径 ─────────────────────────────────────────────
        旧实现：`(soft_budget − reserve) × ratio`，ratio 取 0.30 / 0.35。
        三个致命问题（本次审计 + 195 题实测发现）：
          ① ratio 本身**没有任何实测依据**，是拍脑袋的固定切分；
          ② 它把"单题预算"当蛋糕切，而单题预算当时被 1200s 常量钉死
             ⇒ 切出 234s 这种碎块，把有用工作直接砍掉（与用户要求
             "确保时间开销有用、不做无用功"直接冲突）；
          ③ 各阶段 ratio 之和（0.30+0.30+0.35）已达 0.95 ⇒ 新增阶段无预算可分，
             所以 3.2 / 3.5 / 3.6 / 4_verify / 4.6 才会长期"无预算裸跑"。

        ── 新口径 ──────────────────────────────────────────────────────
        直接给该阶段**实测所需**（195 题 `stage_timers` 的 max × 约 1.2 余量），
        仅作"防单阶段无限膨胀"的上限；跨阶段的保护由 `_gen_deadline`
        （生成侧）与 `verify_reserve`（验证侧）统一负责，不再靠比例切分。

        传 `stage` 时走新口径；不传则回退旧公式（保证既有测试与外部调用零变化）。

        ── ★★ 2026-09-15（用户指示「预算都删了，没意义，还卡正确率」）──
        新增总开关 `phase_budget_enabled`（**默认 False = 不限**）。关闭时本函数
        直接返回「无穷大」⇒ `_phase_deadline_guard` 里 `min(ctx.deadline, now+cap)`
        退化为原 deadline ⇒ **阶段帽不再生效**。
        关闭依据：2.7 的帽是 600s，而实测 mean 513 / p50 537 / **max 1168s**
        ⇒ 一半以上的题被砍断子目标链（链没跑完 = 缺项 = 错，不只是慢）。
        ⚠ 本开关**只管"分配型"阶段帽**；LLM 超时重试 / lean_timeout /
          符号求解线程超时 / 各类死循环硬上限**全部保留**（那是"别挂住"，不是配额）。

        ── ★ 2026-10-07 实测结论（问题总账 §2.6「phase_budget_enabled 疑为结构性死开关」）──
        · **非结构性死开关**：本函数是**活路径**——`config.phase_budget_enabled=True`
          时 `_phase_budget(stage="2.7_subgoal_main")` 返回 600.0、阶段帽真实收紧
          （离线实测：False→inf/收紧 0s；True→600.0/收紧 85800s）。`_CAPS` 表在
          False 时确实整表 no-op（这正是"默认关=不限"的**设计意图**，非缺陷）。
        · **但当前评测入口无法打开它**：① 默认 False；② `run_eval.py` **无对应
          CLI flag**（旧判断"不在 CLI 覆盖白名单"已过期 —— 该键现已在
          `user_agent.py` 白名单，可经 kwargs/`agent_overrides` 设置）；
          ③ 读取走 `getattr(config, ...)`，而 env `PHASE_BUDGET_ENABLED` **不经
          config**（实测 env=1 时 `AgentConfig().phase_budget_enabled` 仍为 False）
          ⇒ 单靠 env 无效。
        ⇒ 结论：**代码层活、评测入口层死**；若要真正可调，需在 `run_eval.py` 加
          CLI flag 或把读取改走注册表（后者属行为改动，本次按"唯一行为改动仅 C/E"
          的约束**未改**，仅留此标注）。
        """
    if not getattr(self.config, "phase_budget_enabled", False):
        return float("inf")
    if stage:
        # 依据：`results/*.jsonl` 的 `stage_timers`，最近三代代码 195 题实测 max。
        _CAPS = {
            "2.6_pre_audit": 650.0,       # 实测 max 560
            # 2026-09-13 收紧：原 1200 取自"实测 max"，但它 **≥ 单题全部预算
            # （1150）** ⇒ `min(deadline, now+1200)` 恒等于 deadline，**等于没有
            # 上限**。而实测 2.7 占单题 45%（112 题 mean 513s / p50 537s），是
            # 单题最大开销。按"不超过单题预算一半"压到 600s，把余量让给
            # 3_solve / 3.6 / 4_verify。
            # ⚠ 注意 `base.py:619` 单次 `client.chat` **不可中断**、预算闸只在
            # 调用之间生效 ⇒ 本上限是"调用之间"的闸，不能硬切正在进行的一次调用。
            "2.7_subgoal_main": 600.0,    # 实测 mean 513 / p50 537 / max 1168
            # 2026-09-13 二轮收紧：**1200 → 600**。
            # 根因（同题 A/B 实测 + 代码判定）：`_phase_deadline_guard` 做的是
            # `min(ctx.deadline, now + cap)`，而 3_solve 开始时 `now + 1200` **恒大于**
            # 单题 deadline（1000s）⇒ min 恒等于原 deadline ⇒ 本上限**完全失效（no-op）**，
            # 于是 2.7 让出的时间被 3_solve 全部吸收。
            # 铁证（同一题，v7 旧 cap 234/180 → 新 cap 1200）：
            #   000: 2.7 425→337 而 3_solve 294→**429**；004: 258→135 而 365→**542**；
            #   010: 262→245 而 490→**585**；budget_skips 由 7/10/8/6 → 1/1/2/8。
            # 同一病理 2.7 已在上面修过（1200→600），**3_solve 当时被漏掉**。
            # 600 = 单题预算（1000）的六成，与 2.7 对称，保证后段 3.6/4_verify/6.5 不被饿死。
            # ⚠ 单次 LLM 200-300s 不可中断（`base.py:619`）⇒ 仍可能被跨过一次。
            "3_solve": 600.0,             # 实测 max 1148；**上限必须 < 单题预算才有效**
            # ★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
            #   原 `"1.2_theorem_hint": 60.0`（2026-09-29 新增；实测单题 8~11s、
            #   上限 60s 防后端退化）**已移除** —— 检索下移到 2.7 子目标求解，
            #   其耗时计入 `2.7_subgoal_main` 的阶段预算。
            #   若线上检索退化拖垮 2.7，先查 `theorem_hint_max_queries` /
            #   `leansearch_top_k` 与检索后端可用性，不要直接给 2.7 加额度。
            "3.2_complete": 220.0,        # 实测 max 175
            "3.3_improve": 500.0,         # 实测 max 453（与 3.4 共享改进额度）
            "3.4_collab": 800.0,          # 实测 max 691
            "3.5_subgoal_sup": 300.0,     # 实测 max 0（默认关闭）
            "3.6_audit_filter": 750.0,    # 实测 max 632
            "4_verify": 900.0,            # 实测 max 783
            "4.5_oracle": 150.0,          # 实测 max 116
            "4.6_adv": 900.0,             # 实测 max 856
            "6.5_audit_gate": 550.0,      # 实测 max 460
        }
        return max(floor, float(_CAPS.get(stage, 600.0)))
    _soft = float(getattr(ctx, "soft_budget", 0) or 0)
    _avail = max(0.0, _soft - max(0.0, reserve))
    return max(floor, _avail * ratio)


def _phase_deadline_guard(self, ctx, cap: float):
    """阶段预算包裹（同时收紧 `deadline` 与 `_gen_deadline`）。

        ⚠ 2026-09-13 关键修复（实测定位）：`BaseAgent.gen_time_up()` 用的是
        **`_gen_deadline`**（单题开始时按 `deadline − verify_reserve` 设一次），
        与 `ctx.deadline` 是**两个独立字段**。此前各阶段的预算包裹**只设
        `ctx.deadline`**，而循环体查的是 `gen_time_up()` ⇒ **阶段预算对循环完全
        无效**（形同虚设）。实测铁证：
          · `2.7_subgoal_main` 预算 660s → 实跑 **816s**
          · `3_solve` 预算 273s → 实跑 **617s**
          · `3.4_collab` 无包裹 → 实跑 **691s**
        三者叠加把 1200s 预算吃光 ⇒ 3.3/4_verify/4.6/6.5 全被饿死（0s）。

        本方法同时收紧两个字段，使 `gen_time_up()` 与 `is_time_critical()`
        **都**在阶段预算到点时返回 True ⇒ 循环真正停手。
        返回 (t0, saved) 供 `_phase_deadline_restore` 还原。
        """
    import time as _t
    _t0 = _t.time()
    _saved = (ctx.deadline, getattr(ctx, "_gen_deadline", 0.0))
    _end = _t0 + max(1.0, float(cap))
    try:
        _tightened = False
        if ctx.deadline and ctx.deadline > 10**8:
            ctx.deadline = min(ctx.deadline, _end)
            _tightened = True
        if getattr(ctx, "_gen_deadline", 0.0) and ctx._gen_deadline > 10**8:
            ctx._gen_deadline = min(ctx._gen_deadline, _end)
            _tightened = True
        # ★ 2026-10-07 分支静默跳过埋点（问题总账 §2.6「隐形门控」）：原两个 `if`
        #   在 deadline 未设（或为 0/非远期哨兵）时**静默不收紧**，事后只见
        #   "阶段帽没生效"却查不出原因（本项目反复吃的"未触发 vs 被吞"）。补一条。
        if not _tightened:
            self.record(ctx, "phase_budget",
                        f"阶段预算未收紧（cap={cap:.0f}s 跳过）：deadline 与 "
                        f"_gen_deadline 均未设或非远期哨兵")
    except Exception:  # noqa: BLE001
        pass
    return _t0, _saved


@staticmethod
def _phase_deadline_restore(ctx, saved) -> None:
    """还原阶段预算包裹前的两个时间字段。"""
    try:
        ctx.deadline, ctx._gen_deadline = saved[0], saved[1]
    except Exception:  # noqa: BLE001
        pass
