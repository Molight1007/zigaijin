# -*- coding: utf-8 -*-
"""单轮无条件自改进主循环（2026-10-08 拆分：从 agent/solver.py 的
``_improve_candidates_once`` 段 4–尾 整段同迁）。

``_improve_loop`` 覆盖：``n_ok`` / 单候选最坏成本预留读取 → 逐候选循环
（两道时间护栏：``is_time_critical`` 与「距硬墙余量 < improve_min_remaining」；
``SELF_IMPROVE_USER`` 提示 + 子目标结论注入；``_compressed_solve`` 改进；
空 / 拒绝 / 过短三出口丢弃；答案抽取 + ``_pool_gate_v2_on`` 入池门槛；
原版保留（可回退副本）；覆盖 reasoning / answer 并标记 ``self_improved``）
→ 循环后汇总 record → 返回 ``n_ok``。

搬运方式：**整段同迁**（段 4–尾逐字搬入，仅去 4 格缩进），**零改写等价搬运**
（``for`` / ``break`` / ``continue`` 随循环整体进入本函数，故无
``continue``→``return`` 改写）。段 1–3（docstring、嵌套 ``_needs_improve``
缺陷过滤、候选筛选与 ``if not targets: return 0``）留在
``_improve_candidates_once`` 主体原位 —— 函数级出口 ``return 0`` 与
``return n_ok`` 语义逐字不变（后者由本函数返回后透传）。新增的只有：函数签名 /
``targets`` 改用形参传入 / 一处**惰性 import**（``_is_refusal`` 仍是 solver.py
模块级符号，惰性 import 避免循环导入）。``_sw_bool`` 直接取自 ``switch_registry``
（与 solver.py 同一开关来源），故 solver.py 不再需要为其保留 re-export。
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext

from .base import _pool_gate_v2_on
from prompts.policy import SELF_IMPROVE_USER, get_policy_system
from utils.extract import (
    extract_final_answer, rescue_final_answer, smart_fallback_answer,
)

# 开关读取统一走 switch_registry（与 solver.py 同一来源；此处直接导入，
# 不再经由 solver.py re-export）。
from agent.switch_registry import get_bool as _sw_bool

def _improve_loop(self, ctx: TaskContext, targets: list) -> int:
    """段 4–尾：逐候选自改进主循环，返回本轮成功改进个数。"""
    # 惰性 import：这是 solver.py 模块级符号，避免顶层循环导入。
    from .solver import _is_refusal
    n_ok = 0
    _imp_min_remaining = float(
        getattr(self.config, "improve_min_remaining", 0.0) or 0.0)
    for cand in targets:
        # 2026-09-06：升级 gen_time_up——仅 is_time_critical 会在单候选
        # 300s 级 LLM 调用前放行最后 1-2 个候选，烧穿剩余预算（冒烟
        # nt-031 3.3=595s / algebra-003 3.3=325s 实证），须按生成侧软截止停。
        # N1''（2026-09-11 修正）：**改回硬墙口径**。原因：外层（orchestrator）
        # 已判定 3.3 该跑（用 time_remaining 门槛放行），但 3.3 排在 2.7/3_solve
        # 之后，而 gen_time_up() 用的是 _gen_deadline(=deadline−480s)=720s——
        # 走到这里必然已越过 → 内层立刻 break → 外层放行被作废（smoke6_v2
        # 六题 3.3 恒为 0s 的最终原因）。改为只防"逼近 1200s 硬墙"。
        if ctx.is_time_critical():
            break
        # 2026-09-07（A，冒烟 alg-060 3.3=640s 烧穿实证）：单候选改进
        # 是一次 200-300s 不可打断的 LLM 调用——仅靠 gen_time_up（=烧到
        # 生成软截止才停）会让"正在跑的最后一个候选"把验证预留吃光，
        # 6.5 终局 Lean/Audit 闸门 time_critical 饿死。
        # 预留 single-call 最坏成本：距生成软截止不足 improve_min_remaining
        # （默认 300s）即停手不再开新候选，把验证预算完整留给 4_verify+6.5。
        # N1''（2026-09-11 修正）：**基准改为硬墙 deadline，而非 _gen_deadline**。
        # 原因同 L37：_gen_deadline(720s) 是"生成类"的统一截止，3.3 走到这里
        # 必然已越过，用它做差会出现负值 → 恒 break（第三层拦截，实测默认
        # improve_min_remaining=300 时 100% 触发）。改用 deadline(1200s) 起算，
        # 语义仍是"给单候选最坏成本留够余量"，但不会在 3.3 处必然失败。
        _hd = float(getattr(ctx, "deadline", 0.0) or 0.0)
        if (_imp_min_remaining > 0 and _hd >= 10**8
                and _hd - time.time() < _imp_min_remaining):
            # ⚠ 2026-09-30 修正阶段名：原写 `"paper_pacer"` —— 但这里判的是
            #   **单题硬墙余量**，与已删除的 PaperPacer（全卷配速）毫无关系。
            #   错误阶段名的后果：诊断按 step 聚合时，这条记录会被算进
            #   `paper_pacer` 桶，而该桶在 PaperPacer 删除后本应只剩全卷配速
            #   那几行 ⇒ 归因时把"自改进因时间停手"误读成"全卷配速在收紧"。
            self.record(ctx, "self_improve",
                        f"Step2 自改进停手：距硬墙 {_hd - time.time():.0f}s < "
                        f"{_imp_min_remaining:.0f}s（单候选最坏成本预留），"
                        f"不再改进剩余候选")
            break
        user_content = SELF_IMPROVE_USER.format(
            problem=ctx.problem,
            candidate_solution=cand.reasoning,
        )
        # ★ 2026-10-04 修（**做题逻辑错误**，用户点名担心的那条）：
        #   3.3 自改进走 `_compressed_solve` **重写候选并覆盖 `cand.answer`**，
        #   但它此前**不注入子目标结论** ⇒ 改进版是"**脱离子目标结论的独立重解**"，
        #   等于把"**主求解必须建立在子目标结论之上**"的设计在自改进这一步悄悄作废
        #   （用户原话：「会不会出现之前子目标的结论没用上，再在总解题中自己再解一遍」
        #     —— 这里字面成立）。
        #   已知后果：项目自陈 098 原候选答 B（正确），3.3 后被改成 D（错误）。
        #   现与 `_generate_proof`(:954) / `_generate_initial`(:1076) /
        #   `_generate_revise`(:1498) **同口径注入**（开关 `enable_subgoal_findings`）。
        user_content = self._apply_subgoal_findings(ctx, user_content)
        resp = self._compressed_solve(
            ctx,
            get_policy_system(use_blueprint=getattr(self.config, "use_blueprint", True)),
            user_content,
            temperature=0.1,
            max_tokens=self._adaptive_max_tokens(
                ctx, self.config.policy_max_tokens),
            prefill_seed="【第一步：诊断】\n",
        )
        if not resp or not resp.strip():
            # 观测埋点（2026-09-11）：3.3 此前只报"完成 N 个"，失败出口
            # （空/拒绝/过短）**全部静默 continue** → 跑了几百秒却看到 n_imp=0
            # 却无从定位（smoke6_v3 实证：3.3=365s 但"自改进完成"0 条）。
            self.record(ctx, "self_improve",
                        f"Step2 丢弃候选#{getattr(cand, 'id', '?')}：空响应")
            continue
        if _is_refusal(resp):
            self.record(ctx, "self_improve",
                        f"Step2 丢弃候选#{getattr(cand, 'id', '?')}：模型拒绝作答")
            continue
        # 改进版比原版还差（明显更短/空壳）则丢弃
        _thr = max(40, len(cand.reasoning) // 3)
        if len(resp.strip()) < _thr:
            self.record(ctx, "self_improve",
                        f"Step2 丢弃候选#{getattr(cand, 'id', '?')}：输出过短"
                        f"（{len(resp.strip())} < 门槛 {_thr}）")
            continue
        answer = extract_final_answer(resp)
        if not answer or len(answer) > 300:
            answer = rescue_final_answer(resp)[0]
        if not answer:
            answer = smart_fallback_answer(resp)
        # ★★ 2026-10-07（2.4 v2 入口穷举 · 入口③）：**自改进是"改写既有候选
        #   answer 字段"的写入点，此前完全不过门槛** —— 实测题 003/007 的池内
        #   id=0（answer = `'实际上，这个问题是已知的竞赛题…'` /
        #   `'实际上，对于这种问题，通常有一个不变量…'`，reasoning 以
        #   「【第一步：诊断】」开头 = 自改进产物）正是**被覆盖进池**的非答案，
        #   且题 007 该候选还进了投票。此处：改进版不是答案 ⇒ **不覆盖原答案**
        #   （保留既有、通常更好的答案），只更新 reasoning 与改进标记。
        if _pool_gate_v2_on():
            _ians, _iok = self._normalize_and_check_answer(
                ctx, getattr(cand, "id", "?"), answer)
            if _iok:
                answer = _ians
            else:
                self.record(ctx, "self_improve",
                            f"Step2 改进版#{getattr(cand, 'id', '?')} 不含最终答案"
                            f" → **保留原答案不覆盖**（防非答案覆盖入池）")
                answer = ""
        # B 方案（2026-09-11，本地实验）：**覆盖前先把原版保留为独立候选**。
        # 依据：smoke6_v3 实证 098 原候选答 B（正确），3.3 自改进后被改成 D
        # （错误）——「无条件替换」会把好答案直接改坏，且没有任何回退机制。
        # 现在改为"原版 + 改进版并存"，交由下游验证/投票择优（不改变下游逻辑）。
        # 开关：SELF_IMPROVE_KEEP_ORIGINAL（默认 1）。
        if _sw_bool("self_improve_keep_original"):
            try:
                import copy as _copy
                _orig = _copy.copy(cand)
                _orig.id = 1 + max(
                    [int(getattr(c, "id", 0) or 0) for c in ctx.candidates] or [0])
                _orig.self_improved = True      # 防对同一份再改进
                _orig.source = (getattr(cand, "source", "") or "") + "+pre_improve"
                # （2.4 v2 入口清单）此处 append **刻意不再过门槛**：`_orig` 是
                #   "**已入池候选**的深拷贝"（其 answer 在入池时已过门槛），
                #   再判一次只会白丢这份"回退副本"，属冗余而非漏网。
                ctx.candidates.append(_orig)
                self.record(ctx, "self_improve",
                            f"Step2 保留原版候选#{_orig.id}"
                            f"（答案 {str(getattr(_orig, 'answer', ''))[:30]}），"
                            f"改进版写入#{getattr(cand, 'id', '?')}")
            except Exception as _ce:  # noqa: BLE001
                self.record(ctx, "self_improve",
                            f"Step2 保留原版失败（{type(_ce).__name__}）→ 走覆盖")
        cand.reasoning = resp
        if answer:
            cand.answer = answer
        # 2026-09-01（SU-01 优化 2）：标记 self_improved，下次再调 Step2 跳过此候选
        cand.self_improved = True
        n_ok += 1
        time.sleep(0.3)  # 速率限制

    if n_ok:
        self.record(ctx, "self_improve",
                    f"Step2 自改进 {n_ok}/{len(targets)} 个候选"
                    f"（无条件={not bool(getattr(self.config, 'self_improve_conditional', False))}）")
    elif targets:
        self.record(ctx, "self_improve",
                    f"Step2 本轮 {len(targets)} 个候选全部未产出改进")
    return n_ok
