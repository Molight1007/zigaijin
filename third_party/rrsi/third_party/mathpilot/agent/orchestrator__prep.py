# -*- coding: utf-8 -*-
"""求解前准备族（prep）（2026-10-07 阶段C6：从 agent/orchestrator.py 的 Orchestrator.run 抽出）。

已抽出段（按 run 内顺序）：
  · 2.6_pre_audit    —— 题意理解（**含题型判题 / 产出唯一题目目标** /
    题意理解：终态 + 答案槽 + 缺口）
    ★ 2026-10-14：原「Lean 前置形式化」调用点**已撤除**（三条原始理由全失效，
      且无 Lean 环境时从未跑过 ⇒ 零行为变化）；`LeanPreVerifier` 类本体保留给
      本地 `cli/leap_eval.py` 证据链用，不再是主流水线的阶段。

★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
  原独立阶段 `1.2_theorem_hint`（本模块的 `theorem_hint_stage`）**已撤销** ——
  定理检索改由 `sub_goal_solver.run` 体首调
  `agent.theorem_hint.run_hint_for(agent, ctx)` 发起（2.7 内、蓝图构建之前），
  于是顺序变为「2.6 判题 → 2.7 检索 → 蓝图/子目标规划 → 求解」。本模块因此
  只剩 `2.6_pre_audit` 一段。

★ 2026-10-14（任务列表 #1「题型判断要和题意理解合并」）：
  原独立阶段 `1.1_type_understanding`（LLM 判题 + 规则交叉校验）**已撤销**，
  其逻辑整体下沉为 `2.6_pre_audit` 的**第一个子步骤**（`judge_question_type`）——
  「先判是什么题（选择题/判断题/证明题/解答题…），再带着题型做题意理解」。
  题型的产出字段（`ctx.question_type` / `ctx.type_decision` / `ctx.type_path`）
  **一个都没变**，下游 130+ 读取点零改动；只是不再有独立的阶段打点与耗时行。

★ 2026-10-14（任务列表 #3「题意理解要和蓝图构建解耦合」）：
  本阶段末尾产出**唯一**题目目标 `ctx.problem_goal`（`agent/problem_goal.py`，
  零 LLM 调用、由既有产物派生）—— 题型 / 要做什么 / 待判选项 / 答案形态 /
  目标情形，即任务原文「给出大模型一个目标，让注意力集中到目标上面」。
  原先这份「理解题意、抽目标」在蓝图 LLM 的 `problem_analysis`、子目标 LLM 的
  `problem_analysis`、选择题两处代码构造的 `problem_analysis` 里各推了一遍
  （三处副本），**已随本任务删除** —— 蓝图从此只消费目标、只负责分解。

**纯搬迁**：段内源码逐字搬入（仅去 12 空格缩进），调用点由 `run` 改为一行委托。
`run` 保留各自的 `self._stage_start(ctx, "<阶段名>")` 打点**原位不动**
（阶段名是稳定契约，被 tools/stage_audit.py / tools/test_report.py 按名聚合）。

耦合实测（AST 作用域感知 —— Py3.14 下 symtable 会把推导式变量漏进外层作用域，
故不用 symtable）：两段对 run 局部变量依赖均为空（只用 self/ctx 参数）；
段内绑定仅异常名/推导式变量（内层作用域）；无 return / 无循环。
2.6 引用 orchestrator 模块级 `_artifact_put` ⇒ 函数内惰性 import（单一来源仍不复制）。
"""
from __future__ import annotations


def judge_question_type(self, ctx) -> None:
    """题型判题（§3 方案 A）：LLM 判题 + **交叉校验** + 路径分流。

    ★ 2026-10-14（任务列表 #1）：本函数**不再是独立阶段** `1.1_type_understanding`，
    而是 `pre_audit_stage`（2.6 题意理解）的**第一个子步骤** —— 由 `pre_audit_stage`
    在体首调用，与「题意理解」合并为同一阶段。

    设计（逐条对应方案）：
      · §3.2 结构化输出契约（`TYPE_LLM_SYSTEM`，禁自由文本）；
      · §3.3 **与规则基线交叉校验** —— 规则强信号优先、低置信不覆盖（防抖动误伤）；
      · §3.4 产出 `ctx.type_path`（路径分流表）。

    ⚠ 安全：LLM 失败/开关关 ⇒ **完全回退**规则判定（行为与接线前逐字一致）。

    ⚠ 顺序代价（合并后的已知影响）：`1.2_theorem_hint` 现**跑在本题之前**，
      它读到的 `ctx.question_type` 回落为**规则判定**（惰性属性），LLM 判题结论
      要到 2.6 才落定。这与 `enable_type_llm=False` 的消融路径行为一致
      （该路径下 theorem_hint 本来就只见规则判定），不影响任何下游正确性；
      若日后按任务 #2 把定理检索移到子目标求解（在 2.6 之后），该代价自动消失。
    """
    if not getattr(self.config, "enable_type_llm", True):
        return
    from agent.question_type import (classify_question_type as _cq,
                                    type_path_plan as _tpp,
                                    parse_type_json as _ptj,
                                    merge_type_decision as _mtd,
                                    TYPE_LLM_SYSTEM as _TS,
                                    TYPE_LLM_USER as _TU)
    _rule = ""
    try:
        _rule = str(_cq(getattr(ctx, "problem", "") or "") or "").strip().lower()
    except Exception:  # noqa: BLE001
        _rule = str(getattr(ctx, "question_type", "") or "").strip().lower()
    _llm_d = {}
    try:
        _raw = self.llm(ctx, [
            {"role": "system", "content": _TS},
            {"role": "user", "content": _TU.format(
                problem=(getattr(ctx, "problem", "") or "")[:4000],
                rule_guess=_rule or "（未知）")},
        ], 0.1, 4096)
        _llm_d = _ptj(_raw)
    except Exception as _e_t:  # noqa: BLE001  判题是增益，绝不阻断
        self.record(ctx, "type_understanding",
                    "LLM 判题失败（回退规则）: %s" % str(_e_t)[:100])
    _dec = _mtd(_rule, _llm_d.get("type", ""), _llm_d.get("confidence", 0.0))
    # ★ 2026-10-14：`_dec["final"]` 是**中文题型**（与 `ctx.question_type` 同一命名
    #   空间，既有契约）；查路径分流表必须用**类别键** `final_key`（英文）——
    #   两个命名空间各就各位，混用会让 `objective_injection` 等下游整段失效
    #   （详见 agent/question_type.py 的 QT_TO_KEY 注释）。
    if _dec.get("final"):
        ctx.question_type = _dec["final"]
    ctx.type_decision = _dec
    ctx.type_path = _tpp(_dec.get("final_key") or _rule)
    self.record(ctx, "type_understanding",
                "题型判定：规则=%s LLM=%s ⇒ 最终=%s（%s；分流=%s）"
                % (_dec.get("rule") or "-", _dec.get("llm") or "-",
                   _dec.get("final") or "-", _dec.get("reason") or "",
                   ctx.type_path))


def pre_audit_stage(self, ctx, tier: str) -> None:
    """2.6 题意理解（**含题型判题** / 题意理解：终态 + 答案槽 + 缺口 / 产出唯一目标）。

    参数 `self` 即 `Orchestrator` 实例（沿用方法体原文，故保留名 `self`）；
    `ctx` 为 `TaskContext`；`tier` 为当前档位（★ 2026-10-14：Lean 前置形式化
    调用点撤除后本参数已无消费点，保留仅为签名稳定）。
    返回 None；结果写回 `ctx`（trace / metadata）。

    ★ 2026-10-14（任务列表 #1「题型判断要和题意理解合并」）：
      本阶段体首先做**题型判题**（`judge_question_type`），再往下做 Lean 前置
      形式化 / AuditGate 复核 / 题意理解链 —— 即「先判是什么题，再理解题意」。
      撤销了原独立阶段 `1.1_type_understanding`（含其 `_stage_start` 打点与耗时行）。
    """

    from .orchestrator import _artifact_put

    # ★ 2026-10-14（任务列表 #1）：题型判题 = 题意理解的**第一步**。
    #   产出 `ctx.question_type` / `ctx.type_decision` / `ctx.type_path`；
    #   下方 2.7 子目标求解据此分流，因此必须排在它们**之前**。
    #   ⚠ 判题失败/开关关 ⇒ 只留规则判定（`ctx.question_type` 惰性属性），
    #     与接线前逐字一致，绝不阻断主链。
    try:
        judge_question_type(self, ctx)
    except Exception as _e_qt:  # noqa: BLE001  判题是增益，绝不阻断
        self.record(ctx, "type_understanding",
                    f"题型判题异常（沿用规则判定）: "
                    f"{type(_e_qt).__name__}: {str(_e_qt)[:120]}")

    # ★ 2026-10-14（第二次收敛，用户决策「直接删掉」）：**Lean 前置形式化
    #   `lean_pre_verifier.run(ctx)` 的调用点已从 2.6 主流水线撤除**。
    #
    #   为什么撤 —— 它当初被放在**最前**的三条理由现已全部失效（2026-10-14 逐条核查）：
    #     ① 【硬·已失效】产物 `formal_spec` / `formal_gaps` 原本是**下游规划的输入**
    #        （注入蓝图与子目标规划）⇒ 必须在 planning 之前。但这两处注入已随
    #        **2026-09-06 去 Lean 化**删除（`blueprint_planner.py` 与
    #        `sub_goal_solver__planner.py` 各留一句同文说明）—— 产物失去消费者。
    #     ② 【软·语义】「先证题意、再解题」已被 `judge_question_type` +
    #        `problem_goal` + 缩减版题意理解取代；且 Lean 形式化只能验「命题是否
    #        well-typed」，验不了「所求部分是什么」（K11 那种错值照样编译通过）。
    #     ③ 【实测反证】`user_agent__config.py` 自记「2026-09-12 实测 5/5 fail、
    #        零产出（纯烧 100-150s/题）」；本类自记「3 题 2 轮重试 0 成功，失败
    #        全在 Lean 语言层面，非题意层面」⇒ 它实测测的是「模型会不会写 Lean」，
    #        不是「有没有读懂题」。
    #   ⇒ 「前置检查的正当性 = 它的产物被下游消费」这条硬理由在 2026-09-06 消失后，
    #     「放在最前」就不再是必要性，只是位置习惯。
    #
    #   ⚠ 平台/本地均无 Lean 环境（`_lean_active()` 探测 `lean --version` 恒 False）
    #     ⇒ 此调用块**从来就没跑过** ⇒ 撤除是**零行为变化**，只消除「隐形门控」
    #     （图上像一步、实际永不执行）。
    #
    #   ⚠ **`LeanPreVerifier` 类与 `run()` 方法本体保留未删** —— 本地 LEAP 三阶段
    #     证据链脚本 `cli/leap_eval.py::run_single` 仍**独立构造并调用**它
    #     （`:283-289` 读 `ctx.preverify_trace` / `formal_spec` / `sketch_audit`）；
    #     且 `preverify_timeout` 被 `lean_refiner.py:142` / `lean_translator.py:232`
    #     共用。本类自此**只服务本地证据链，不再是主流水线的阶段**。
    #   （同理保留 `_lean_active` / `_lean_applicable`：仍被后置 `lean_backcheck.py` 消费。）

    # ★ 2026-10-12（批次 1，用户设计 D1–D4）；★ 2026-10-14 收敛：
    #   题意理解 = 只确定「题目要我们做什么」—— 建「题目完成后的终态」
    #   （答案用符号槽，代码机械剥离被写死的数值）+ 缺口 A/B（写入
    #   ctx.understanding_gaps，供子目标消费）。
    #   ★ 2026-10-14 删「逻辑链 + 一致性复推 + 链的 Lean 自检」——链式分解与
    #   2.7 蓝图 AND-OR DAG 同层同义（属「整体理解」），保留即重复劳动。
    #   独立于既有 `lean_pre_verifier.run`（后者保持原行为不变），异常一律降级。
    # ⚠ 兜底值必须与 `AgentConfig.enable_understanding_chain`(=True) **一致**：
    #   若写成 False，则任何"字段缺失"的旧 config/测试桩都会**静默关闭**本机制，
    #   与「配置默认开」矛盾 ⇒ 典型的"两套默认值"缺陷（code-audit 路线 C「隐形门控」）。
    if getattr(self.config, "enable_understanding_chain", True):
        try:
            self.lean_pre_verifier.run_understanding_chain(ctx)
        except Exception as _e_uc:  # noqa: BLE001
            self.record(ctx, "understanding_chain",
                        f"题意理解异常（跳过，不阻断）: "
                        f"{type(_e_uc).__name__}: {str(_e_uc)[:120]}")

    # ★ 2026-10-14（任务列表 #3「题意理解要和蓝图构建解耦合」）：
    #   题意理解收敛为**唯一产物** `ctx.problem_goal` —— 从既有产物派生（零 LLM 调用）：
    #     题型（已有）→ 要做什么（按题型分派）→ 选择题要判哪几个选项 / 答案口径
    #     → 目标情形（理解链的终态，符号槽）。
    #   原先这份「目标」散在三处各推一遍（蓝图 LLM 的 problem_analysis、
    #   子目标 LLM 规划的 problem_analysis、选择题两处代码构造的 problem_analysis），
    #   已随本任务删除 ⇒ 蓝图只**消费**目标、不再自己理解题意。
    try:
        from .problem_goal import build_problem_goal as _bpg
        ctx.problem_goal = _bpg(ctx)
        self.record(ctx, "problem_goal",
                    "题目目标：题型=%s、口径=%s、选项=%d 个 ⇒ %s"
                    % (ctx.problem_goal.get("question_type") or "-",
                       ctx.problem_goal.get("n_correct") or "-",
                       len(ctx.problem_goal.get("options") or []),
                       str(ctx.problem_goal.get("goal_statement") or "")[:120]))
    except Exception as _e_pg:  # noqa: BLE001  目标是增益，绝不阻断主链
        ctx.problem_goal = {}
        self.record(ctx, "problem_goal",
                    f"题目目标构建异常（留空，下游按原口径）: "
                    f"{type(_e_pg).__name__}: {str(_e_pg)[:120]}")

    # ★ 2026-10-02 hook 01：题意理解确认后落盘（2.6_pre_audit → 01_understanding）。
    _artifact_put(ctx, "01_understanding", {
        "question_type": getattr(ctx, "question_type", ""),
        "domain": getattr(ctx, "domain", ""),
        # 2026-10-14 任务 #3：题意理解的**唯一产物**随理解一起落盘
        "problem_goal": dict(ctx.problem_goal or {}),
        "lean_applicable": (ctx.metadata or {}).get("lean_applicable"),
        "lean_skip_reason": (ctx.metadata or {}).get("lean_skip_reason"),
        "trace": [t for t in (getattr(ctx, "trace", None) or [])
                  if isinstance(t, dict)
                  and t.get("step") in ("classify_type",
                                        "type_understanding",
                                        "understanding_chain",
                                        "problem_goal",
                                        "lean_preverify")][-20:],
    })


def prep_tier_budget(self, ctx) -> str:
    """2.5 档位/预算初始化（2026-09-29 起统一单一档位，恒为 "deep"）。

    **搬迁说明**：本块原为 `Orchestrator.run` 内**无阶段打点**的前置块
    （`2.5` 难度路由已删除，只余档位/预算赋值），故不属 15 个 `_stage_start` 段。
    搬出后由 `run` 以 `tier = prep_tier_budget(self, ctx)` 承接返回值。
    ⚠ **非纯逐字搬**：末尾**新增** `return tier`（原为隐式把 `tier` 留在 run 作用域）。
    """
    # 2.5) 2026-09-29：**难度路由已删除，统一为单一档位**（用户决策）。
    # 原 DifficultyRouter 静态预判 + LLM 自评 → fast/standard/deep 三档，
    # 档位再决定候选数/投票数/子目标数/预算等"方法集"——违背"用全部方法
    # 测试大模型能力"的初衷。现 `ctx.tier` 恒为 "deep"（= 原最强档配置），
    # 不再有任何按难度分流的分支。
    # ⚠ 同时删除的还有：应急降档（emergency → standard）、deep 全卷配额闸
    #   （allow_deep()，`deep_quota_ratio=0.25` × 112 = 28 恰等于实测 deep 题数，
    #   即"研究档下仍按配额裁剪方法"的头号来源）。
    ctx.tier = "deep"
    tier = "deep"
    # 2026-10-01 按用户决策：研究期不限时 —— **删除"单题按剩余时间收紧 deadline"
    # 的降级路径**（原 `min(硬顶, 档位软预算)` 会把单题拦腰截断）。
    # 现单题只受 `max_time_per_question`（86400，仅防挂死）约束，不再按预算降级。
    # soft_budget 仅保留为诊断字段（值=档位预算，已=86400）。
    ctx.soft_budget = float(
        (getattr(self.config, 'tier_budget', None) or {}).get("deep", 86400.0))
    # 尾部阈值：统一取 deep 档值 60s（把时间用得更尽）。
    # 2026-09-29：原 `if tier == 'deep'` 分支已因统一档位删除。
    ctx.critical_tail_seconds = float(
        getattr(self.config, 'deep_critical_tail_seconds', 60.0))
    # 2026-10-01 按用户决策：研究期不限时 —— **删除"生成侧软截止 (verify_reserve)"
    # 降级路径**。原逻辑预留 540s 给验证侧、逼生成在同一截止点前停手；
    # 研究期不省资源，生成/验证不再互相掐预算。
    # `_gen_deadline` 保持未设（0.0）⇒ `gen_time_up()` 回退 `is_time_critical()`，
    # 而单题 deadline 已达 86400 ⇒ 实际不再触发任何"按剩余时间降级"。
    ctx._gen_deadline = 0.0
    # 按档位调整 LLM 调用预算（deep 档需要更多调用次数）
    max_calls = self.config.tier_max_calls.get(
        tier, self.config.max_total_calls)
    if ctx.budget is not None:
        ctx.budget.set_max_calls(max_calls)
    else:
        # ★ 2026-10-07 分支静默跳过埋点（问题总账 §2.6「隐形门控」）：原
        #   `if ctx.budget is not None:` 无 else ⇒ 无 budget 时调用预算**静默
        #   不设**，而下方 record 仍会打印 max_calls，造成"看起来设了其实没设"。
        self.record(ctx, "paper_pacer",
                    f"调用预算未设置（ctx.budget 为空，跳过 set_max_calls，"
                    f"计划值 {max_calls}）")
    # ⚠ `paper_pacer` 是遗留 event 标签（PaperPacer 已于 2026-09-29 删除），
    #   内容（档位/调用预算）仍有效，2026-10-02 起仅作标签保留，勿据此判断全卷配速存在。
    self.record(ctx, "paper_pacer",
                f"档位 {tier} 软预算帽 {ctx.soft_budget:.0f}s "
                f"(调用预算 {max_calls})",
                tier=tier, soft_budget=round(ctx.soft_budget))
    return tier
