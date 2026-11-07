# -*- coding: utf-8 -*-
"""求解族阶段（2026-10-07 阶段C4：从 agent/orchestrator.py 的 Orchestrator.run 抽出）。

已抽出段（按 run 内顺序）：
  · 2.7_subgoal_main —— 子目标细化主路径 + C-lite 蓝图数值攻击
  · 3_solve          —— 求解主采样（deep P&E / 蓝图·候选落盘 / Solver 多路采样 / 无可用候选兜底）

**搬迁方式分两类**：
  · 2.7_subgoal_main —— **纯搬迁**：段内源码逐字搬入（仅去 8 空格缩进），调用点改为一行委托。
  · 3_solve          —— **哨兵 + 解包**（⚠ 非纯逐字搬）：本段是 `run` 中唯一带 run 级
    `return self._fallback_direct(ctx)` 的段 ⇒ 段体逐字搬入后**新增**末尾
    `return _SOLVE_CONTINUE`，原 return 原样保留，调用侧以 `is not` 透传（详见
    `solve_stage` docstring）。

各段 `run` 保留 `self._stage_start(ctx, "<阶段名>")` 打点**原位不动**
（阶段名是稳定契约，被 tools/stage_audit.py / tools/test_report.py 按名聚合）。

耦合实测（AST 自由变量分析）：2.7 段对 `run` 的局部变量依赖**仅为 `ctx` 与 `tier`**；
段内 4 个局部名（`_p4_cap` / `_p4_saved_dl` / `_p4_t0` / `exc`）**零外泄**、
无 `return`/`yield`/`break`/`continue`。其余外部符号全部经 `self` 访问
（`self.record` / `self._phase_budget` / `self._phase_deadline_guard` /
`self._phase_deadline_restore` / `self.sub_goal_solver` / `self._value_attack_blueprint` /
`self.config`）与模块级 `time` / `logger`，无其它依赖。

返回 None；结果全部写回 `ctx`（`_subgoal_main_done` / 蓝图 / 候选等）。
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("MathPilot")

# 3_solve 段「哨兵 + 解包」哨兵：段内唯一 run 级 return 是 `_fallback_direct`，
# 其余路径一律返回本哨兵；调用侧以 `is not` 判定是否透传该 return。
_SOLVE_CONTINUE = object()


def subgoal_main_path(self, ctx, tier: str) -> None:
    """2.7 子目标细化主路径（含 P4 阶段预算）+ C-lite 蓝图数值攻击。

    参数 `self` 即 `Orchestrator` 实例（沿用方法体原文，故保留名 `self`），
    `ctx` 为 `TaskContext`，`tier` 为当前档位（本项目恒为 "deep"）。
    """
    # 2.7) 子目标细化主路径（v2.9）：统一档位下先跑一次子目标分解逐步求解
    # 2026-09-06：时间判断升级 gen_time_up（生成侧软截止，给验证留预算）
    if (getattr(self.config, 'enable_subgoal_main_path', True)
            and not ctx.state.emergency
            and not ctx.gen_time_up()):
        self.record(ctx, "control",
                    "子目标细化主路径先行（前置形式化已校准题意）")
        # P4（2026-09-11）：2.7 阶段预算上限——把超额时间让给验证与重解。
        # 数据依据（official112 冒烟实测）：2.7 单阶段耗 753–1078s（占 1200s 硬顶的
        #   63–90%），波动可达 base 均值(513s)的 2 倍；致 3.x / 4 验证 / 4.6 对抗 /
        #   P1 重解全部无预算（P1 三次前移均因时间耗尽而未触发）。
        # 口径：档位软预算 × ratio（默认 0.55 → standard≈297s / deep≈660s），
        #   且不低于 240s（保证子目标链有基本时间）；ratio 可经 config 覆盖
        #   （subgoal_phase_budget_ratio，设 1.0 等价于关闭本限制，便于 A/B 对照）。
        # 实现：临时收紧 ctx.deadline（子目标链内部的时间判断随之生效），
        #   finally 恢复原 deadline，不影响后续阶段。
        # 2026-09-13 预算修复（两处问题）：
        # ① 原用 `soft_budget × ratio`，**未扣除 reserve** ⇒ deep 档 660s
        #    直接吃掉本该留给后段（4_verify/4.6/6.5）的配额；
        # ② 默认 0.55 过大（实测 2.7 常跑到 816s，预算形同虚设）。
        # 现统一走 `_phase_budget`（(soft − reserve) × ratio）并下调到 0.30
        # ⇒ deep：(1200−420)×0.30 = **234s**，为 3_solve / 3.3 / 后段留空间。
        # 2026-09-13：改走**实测上限**口径（旧的比例切分 ratio 无实测依据，
        # 且把单题预算切成 234s 碎块、砍掉有用工作 —— 详见 `_phase_budget`
        # docstring）。本阶段上限 = 实测 max 1168s × 余量。
        _p4_cap = self._phase_budget(ctx, stage="2.7_subgoal_main")
        _p4_t0, _p4_saved_dl = self._phase_deadline_guard(ctx, _p4_cap)
        try:
            self.sub_goal_solver.run(ctx)
        finally:
            self._phase_deadline_restore(ctx, _p4_saved_dl)
        ctx._subgoal_main_done = True
        self.record(ctx, "control",
                    f"2.7 阶段预算 {_p4_cap:.0f}s"
                    f"（实耗 {time.time() - _p4_t0:.0f}s）")

    # C-lite（2026-09-03）：子目标蓝图 merge 产出后、候选生成前，
    # 数值攻击蓝图极值声称——009 型"蓝图心算错值"在求解前就证伪。
    # 审核修正（2026-09-03）：原读 `_blueprint_value_false`（带下划线），
    # 写入处是 `blueprint_value_false`（无下划线）→ 幂等守卫从未生效。
    if not getattr(ctx, "blueprint_value_false", None):
        try:
            self._value_attack_blueprint(ctx, tier)
        except Exception as exc:  # noqa: BLE001  数值攻击失败不阻断主链路
            # 2026-09-03 审核：原为 `pass` 无日志 → v17 全程 0 痕迹时
            # 无法区分"未触发"与"异常被吞"（与 lean_gate 吞 AttributeError
            # 同型）。改为 warning + 埋点，任何失败都留证据。
            logger.warning("[C-lite] 数值攻击异常（已跳过）: %s: %s",
                           type(exc).__name__, exc)
            self.record(ctx, "value_attack",
                        f"数值攻击异常跳过: {type(exc).__name__}: {exc}")


def complete_truncated(self, ctx, tier: str) -> None:
    """3.2 截断候选续写（统一档位 max_completions 个，emergency/verify_only 跳过）。

    参数 `self` 即 `Orchestrator` 实例（沿用方法体原文，故保留名 `self`），
    `ctx` 为 `TaskContext`，`tier` 为当前档位（本项目恒为 "deep"）。
    返回 None；结果写回 `ctx.candidates`。
    """
    # 3.2) 截断候选续写：统一档位 max_completions 个，应急模式跳过
    if (getattr(ctx, 'candidates', None)
            and not ctx.state.emergency
            and not ctx.state.verify_only):
        max_comp = self.config.tier_max_completions.get(tier, 1)
        if max_comp > 0:
            n_completed = self.solver.complete_truncated_candidates(
                ctx, max_count=max_comp)
            if n_completed > 0:
                self.record(ctx, "control",
                            f"截断续写完成 {n_completed} 个候选")
        else:
            # ★ 2026-10-07 分支静默跳过埋点（问题总账 §2.6「隐形门控」）：原
            #   `if max_comp > 0:` 无 else ⇒ max_completions 配 0 时**静默跳过
            #   整个 3.2**，阶段耗时为 0 却看不出"是配置 0 还是没触发"。
            self.record(ctx, "control",
                        f"3.2 截断续写跳过：档位 {tier} max_completions="
                        f"{max_comp}（<=0 ⇒ 本阶段不执行）")
    else:
        self.record(ctx, "control",
                    "3.2 截断续写跳过："
                    + ("无候选" if not getattr(ctx, "candidates", None) else
                       ("emergency 状态" if ctx.state.emergency else
                        "verify_only 状态")))


def self_improve_step(self, ctx) -> None:
    """3.3 Step 2 无条件自改进（生成后、验证前对候选 review+improve 一遍）。

    参数 `self` 即 `Orchestrator` 实例（沿用方法体原文，故保留名 `self`），
    `ctx` 为 `TaskContext`。返回 None；结果写回 `ctx`（候选改进 + `_improve_phase_used`）。
    """
    # 3.3) Step 2 无条件自改进（IMO2025 论文流水线）：
    #      生成后、验证前，对候选先 review+improve 一遍（注入第二段推理
    #      预算）。论文实测初始解质量低、此步显著改进。
    #      仅非应急模式执行（2026-09-14：fast 档已删除，原
    #      `tier != 'fast'` 条件恒真 ⇒ 移除，见 difficulty_router）。
    if (getattr(self.config, 'enable_self_improve', True)
            and not ctx.state.emergency
            and not ctx.state.verify_only
            and ctx.candidates
            and self._self_improve_applicable(ctx)):
        # N1''（2026-09-11 修正）：3.3 是「验证前的最后改进机会」，也是论文
        # 验证的最大杠杆（22.2%→31.1%），**不能用生成侧软截止拦它**。
        # 原因：3.3 排在 2.7/3_solve 之后，而 gen_time_up() 用的是
        # _gen_deadline = deadline − verify_reserve(480s) = 720s ——
        # 走到这里时必然已越过 720s → 恒被跳过（smoke6_v2 六题 3.3 全为 0s）。
        # 改用**硬墙口径**：只要距 1200s 墙还够一次改进调用，就执行。
        _imp_left = float(ctx.time_remaining())
        if _imp_left > float(getattr(
                self.config, "self_improve_min_left_sec", 150.0)):
            # 注意：本调用已由上方条件块门控
            # （`not ctx.state.verify_only` / `not ctx.state.emergency`）。
            _imp_t0 = time.time()
            n_imp = self.solver.improve_candidates(ctx)
            # 2026-09-13：记录本阶段实耗，供 3.4 计算"改进类共享预算"的余额
            # （两者语义重叠：3.3 是单 Agent 自审自改，3.4 是三 Agent 协作改进）。
            try:
                ctx._improve_phase_used = (float(
                    getattr(ctx, "_improve_phase_used", 0.0) or 0.0)
                    + (time.time() - _imp_t0))
            except Exception:  # noqa: BLE001
                pass
            if n_imp > 0:
                self.record(ctx, "control",
                            f"Step2 自改进完成 {n_imp} 个候选"
                            f"（剩余 {_imp_left:.0f}s）")
        else:
            self.record(ctx, "control",
                        f"Step2 自改进跳过（剩余 {_imp_left:.0f}s "
                        f"< 门槛 {getattr(self.config, 'self_improve_min_left_sec', 150.0):.0f}s）")


def solve_stage(self, ctx, tier: str) -> object:
    """3_solve 求解主采样段（deep P&E / 蓝图·候选落盘 / Solver 多路采样 / 兜底）。

    参数 `self` 即 `Orchestrator` 实例（沿用方法体原文，故保留名 `self`），
    `ctx` 为 `TaskContext`，`tier` 为当前档位（本项目恒为 "deep"）。

    ⚠ **非纯逐字搬**（与 2.7/3.2/3.3 段不同）：本段是 `run` 中**唯一带 run 级
    return** 的段（`return self._fallback_direct(ctx)`，在 `_has_usable_candidate`
    守卫内、且为该守卫与整段的末句）。为在不改 `run` 控制流语义的前提下搬出，
    采用**哨兵 + 解包**：段体逐字搬入（仅去 8 空格缩进），仅**新增**末尾
    `return _SOLVE_CONTINUE`；原 `return self._fallback_direct(ctx)` **原样保留**。
    调用侧以 `if _r is not _SOLVE_CONTINUE: return _r` 透传（严格 `is not`）。
    副作用顺序、异常传播（本段仍处原外层 try 内）均不变。

    耦合实测（AST 自由变量）：对 run 的局部变量依赖为空；段内 10 个局部名
    （`_dsp_*` / `_sp_*` / `_bp02` / `_pl03` / `_st03` 及推导式 `c`）零外泄、
    无 yield/break/continue。唯一模块级外部符号是 `_artifact_put` ⇒ 函数内惰性
    import（单一来源不复制，见 orchestrator__prep.py 同型做法）；`time` 已模块级 import。

    返回 `_SOLVE_CONTINUE`（本段正常走完）或 `self._fallback_direct(ctx)` 的结果
    （无可用候选兜底，调用方须直接 return）。
    """

    from .orchestrator import _artifact_put

    # 3) 求解
    # deep 档：Plan-and-Execute 主路径先行（子目标分解逐步求解 + 每步 oracle 校验），
    # 让结构化计划-执行候选先进入后续客观审核与投票。
    if (tier == 'deep'
            and getattr(self.config, 'deep_use_sub_goal', True)
            and not getattr(ctx, '_subgoal_main_done', False)
            and not ctx.state.emergency
            and not ctx.gen_time_up()):
        self.record(ctx, "control", "deep 档 Plan-and-Execute 主路径先行（子目标分解）")
        # P4-2（2026-09-11）：deep 档 P&E 同样纳入阶段预算（否则它可独占 3_solve）
        # 2026-09-13：改走实测上限口径（实测 max 1148s × 余量）。
        _dsp_cap = self._phase_budget(ctx, stage="3_solve")
        _dsp_t0, _dsp_saved = self._phase_deadline_guard(ctx, _dsp_cap)
        try:
            self.sub_goal_solver.run(ctx)
        finally:
            self._phase_deadline_restore(ctx, _dsp_saved)
        self.record(ctx, "control",
                    f"3_solve(deep P&E) 阶段预算 {_dsp_cap:.0f}s"
                    f"（实耗 {time.time() - _dsp_t0:.0f}s）")

    # ★ 2026-10-02 hook 02/03：蓝图(DAG) / 子目标落盘（只加不改）。
    #   位置刻意放在 **2.7 主路径 + deep 档 P&E 之后**：两条路径谁产出了蓝图
    #   都能覆盖；二者互斥（P&E 有 `not _subgoal_main_done` 守卫）故只写一次。
    #   02_blueprint = AND-OR DAG；03_subgoals = DAG 转出的子目标规划 + 逐步结果。
    _bp02 = getattr(ctx, "blueprint", None) or {}
    _pl03 = getattr(ctx, "blueprint_plan", None) or {}
    _st03 = getattr(ctx, "subgoal_trace", None) or []
    if _bp02:
        _artifact_put(ctx, "02_blueprint", _bp02)
    if _pl03 or _st03:
        _artifact_put(ctx, "03_subgoals", {"plan": _pl03, "trace": _st03})

    # Solver 多路采样（候选数/温度分层按档位，solver 内部读取 ctx.tier）
    # 2026-10-01 按用户决策：研究期不限时 —— **删除"L1 验证优先"触发路径**
    # （原：剩余 < verify_only_seconds 时停止生成新候选）。`ctx.state.verify_only`
    # 不再被置位（恒 False）⇒ 下游 `not ctx.state.verify_only` 门禁恒放行，
    # 不再有任何"为省时间而跳过生成/审核"的降级。
    if not ctx.state.verify_only:
        # 2026-09-06 超时修复：生成侧软截止已到且已有候选 → 不再追加
        # 生成（solver.run 单次可能 200-300s），直接带现有候选进验证。
        # 无候选时仍必须跑（兜底产出第一候选）。
        if ctx.gen_time_up() and ctx.candidates:
            # ⚠ `paper_pacer` 是遗留 event 标签（PaperPacer 已于 2026-09-29 删除），
            #   2026-10-02 起仅作标签保留，勿据此判断全卷配速存在。
            self.record(ctx, "paper_pacer",
                        "生成侧软截止已到且已有候选，跳过追加生成"
                        "直接进入验证/审核")
        else:
            # P4-2（2026-09-11）：3_solve 阶段预算上限。
            # v5 冒烟实证：`3_solve` 是本链路的**预算黑洞**（单阶段耗 303–1094s；
            # 014 至 3.6 累计 1348s，远超当时的生成截止 900s 达 448s）——原因是
            # `solver.run` 内部虽查 `gen_time_up()`，但**单次 LLM 调用可达 200-300s**，
            # 多次调用叠加即可穿越截止点。故对本次调用临时收紧 deadline，
            # 使循环边界的检查真正生效。
            # 2026-09-13：改走**实测上限**口径（实测 max 1148s × 余量）。
            # 旧比例切分（0.30 × (soft−420)）无实测依据，且把预算切成
            # 234s 碎块。跨阶段保护改由 `_gen_deadline` + `verify_reserve`
            # 统一负责，不再在此扣 reserve。
            _sp_cap = self._phase_budget(ctx, stage="3_solve")
            # 2026-09-13 修复（Explore 审计发现）：此处原为**手写**收紧，
            # 只改 `ctx.deadline`、**未改 `_gen_deadline`** ⇒ solver 内部
            # 循环查的 `gen_time_up()`（看 _gen_deadline）完全不受本阶段
            # 预算约束，单次 LLM 调用 200-300s 可反复叠加穿越预算
            # （实测 3_solve 超预算 1.7×）。统一改用 _phase_deadline_guard
            # （两个字段一起收紧），与 2.7 / 3.4 / 6.5 口径一致。
            _sp_t0, _sp_saved = self._phase_deadline_guard(ctx, _sp_cap)
            try:
                self.solver.run(ctx)
            finally:
                self._phase_deadline_restore(ctx, _sp_saved)
            self.record(ctx, "control",
                        f"3_solve 阶段预算 {_sp_cap:.0f}s"
                        f"（实耗 {time.time() - _sp_t0:.0f}s）")

    # ★ 2026-10-02 hook 05：3_solve 生成后落盘候选池（05_candidates）。
    _artifact_put(ctx, "05_candidates", [
        {"id": c.id, "answer": getattr(c, "answer", ""),
         "reasoning": getattr(c, "reasoning", ""),
         "revised": getattr(c, "revised", False)}
        for c in (getattr(ctx, "candidates", None) or [])
    ])

    if not self._has_usable_candidate(ctx):
        # ★ 2026-10-19（097 早退根因）：原「无候选 → 直接抛 no_usable_candidate」
        #   在 3_solve 后就中断流程，导致 4.6 池空重写根本没机会跑（097=292s 早退）。
        #   改为**继续**：4.0~4.5 对空池零成本跳过，4.6 触发「池空重写」；重写仍空
        #   时由 final_selector 的 no_usable_candidate 分支兜底（失败仍暴露，不直答）。
        self.record(ctx, "control",
                    "Solver 未产出可用候选（空），继续走五道链（4.6 池空重写）")
    return _SOLVE_CONTINUE
