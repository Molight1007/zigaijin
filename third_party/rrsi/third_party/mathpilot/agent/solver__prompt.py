# -*- coding: utf-8 -*-
"""初始求解提示词/上下文装配（2026-10-08 拆分：从 agent/solver.py 的
``_generate_initial`` 段 4–11 逐字搬出）。

  · ``_build_initial_prompt``  —— 段 4–7：lemma 上下文 + 蓝图/领域 system/user 装配
    + 子目标结论注入 + 答案格式引导（``\boxed{}``）。返回 ``(system_prompt, user_content)``。
  · ``_inject_initial_context`` —— 段 8–11：定理清单 / 易错点记忆 / 客观题特化 /
    答案形态要求 依次追加到 ``user_content``。返回追加后的 ``user_content``。

搬运方式：**逐字搬**（段内源码逐字搬入，仅去结构性缩进；多行字符串无）。
新增的只有：函数签名 / 末尾 ``return`` / ``_inject_initial_context`` 顶部的
**惰性 import**（``error_lessons_block``/``error_lesson_ids`` 仍是 solver.py 的模块级
符号——按拆分规约「函数内惰性 import 或补 re-export」，此处取惰性 import 以避免
solver.py ↔ 本模块的顶层循环导入）。

``agent/solver.py::_generate_initial`` 原段改为两行委托调用。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext

from prompts.policy import (
    get_policy_system,
    get_domain_hint,
    build_blueprint_user_message,
)

logger = logging.getLogger("MathPilot")


def _build_initial_prompt(self, ctx: TaskContext) -> tuple[str, str]:
    """段 4–7：装配 system/user 提示词（lemma + 蓝图 + 子目标结论 + 答案格式）。"""
    # lemma 上下文注入
    lemma_ctx = self._collect_lemma_context(ctx)

    if self.config.use_blueprint:
        system_prompt = get_policy_system(use_blueprint=True)
        domain_hint = get_domain_hint(ctx.domain) if ctx.domain else ""
        user_content = build_blueprint_user_message(ctx.problem, domain_hint)
    else:
        system_prompt = get_policy_system(use_blueprint=False)
        user_content = ctx.problem
        if ctx.domain:
            user_content = get_domain_hint(ctx.domain) + "\n" + ctx.problem

    # 注入 lemma 上下文
    if lemma_ctx:
        system_prompt = lemma_ctx + system_prompt

    # ★ 2026-10-02（阶段一：子目标结论 → 主求解）：把 2.7 / 3_solve(P&E) 阶段
    # 已求得的子目标结论作为**中间数据**注入主求解提示词，让主答案建立在子目标
    # 之上（用户设计原话：「主求解一定要在子目标的基础上」）。
    # 位置：题目之后、答案格式引导之前 —— 紧贴题目，作为"已知数据"被读；不放
    #   提示词最末尾（避免模型进入续写模式，历史教训）也不早于题目（需贴题干语义）。
    # 空安全：无子目标结论时零噪音、零行为变化（helper 内部兜底）。
    # 只拼一次（与 error_lessons 同口径），retry / 多候选不重复注入。
    user_content = self._apply_subgoal_findings(ctx, user_content)

    # ICMA 对齐（v2.4.0）：末尾追加章节输出引导。系统 prompt 已要求四章节
    # 结构化输出并禁止思考过程，这里仅强调【最终答案】章节必须明确，不引导自由 CoT。
    #
    # 2026-08-30（#51 答案定型）：基线 45 题实测——**仅 7/45（15.6%）的推理里
    # 出现 \boxed{}**，5 题抽出的答案超过 60 字符（明显抽到了推理文本），
    # 且 `reference_matched` 精确匹配 45/45 全 False。本地宽松 LLM 判分能"看懂"，
    # 平台判分看不懂——这是「本地 46.7% vs 平台 20%」落差里可控性最高的一块。
    # 模型本身具备给出简洁答案的能力（#21 材料结论：模型可到 90 分），
    # 缺的是**强制定界**，故在提示词侧要求 \boxed{}，而非让抽取器去猜。
    _ANSWER_GUIDE = (
        "\n\n请严格按系统提示的四章节格式输出完整解答，"
        "确保【最终答案】章节给出明确、简洁的最终结论。"
    )
    if getattr(self.config, 'enable_answer_boxed', True):
        _ANSWER_GUIDE += (
            "\n【最终答案】章节中，最终结论必须且只能用 \\boxed{...} 包裹，"
            "并在其后另起一行给出不含任何公式标记的最简形式"
            "（例如：\\boxed{42}；最简形式：42）。"
            "\\boxed{} 内只放答案本身，不要放推导过程、单位说明或多余文字。"
        )
    user_content = user_content + _ANSWER_GUIDE
    return system_prompt, user_content


def _inject_initial_context(self, ctx: TaskContext, user_content: str) -> str:
    """段 8–11：向 user_content 追加定理/易错点/客观题特化/答案形态要求。"""
    # 惰性 import：这两个是 solver.py 模块级符号，避免顶层循环导入。
    from .solver import error_lessons_block, error_lesson_ids
    # ★ 2026-09-29（截图 #3+#4）：Mathlib 定理检索结果注入。
    #
    # 背景（用户重点关切）：1.2_theorem_hint 阶段用 leansearch 到 Mathlib 检索
    # "本题该用的定理"，但在此之前 `ctx.theorem_hint_block` 只是**算出来放着**，
    # 从未拼进任何提示词 ⇒ 检索对大模型的实际影响恒为 0，A/B 必然测出"无差异"，
    # 用户要回答的「定理对大模型的推理效果如何」在结构上就无法被回答。
    # 本注入是该功能**产生作用**的唯一通道，没有它整条链只是空转埋点。
    #
    # 位置：题目之后、答案格式之后，但**早于** error_lessons / objective。
    #   · 不放最末尾：避免模型把定理清单当成"待续写的前文"；
    #   · 不早于题目：定理必须紧贴题目语义才可读。
    # 长度：检索侧已限制 top_k(5) × max_queries(2)，实测块长约 200–400 字符，
    #   相对题干本身不构成噪音；无命中时 block 为空串 ⇒ **零成本零噪音**。
    # 开关：`enable_theorem_hint`（与检索侧同一开关，关掉即整条链关闭）。
    if getattr(self.config, 'enable_theorem_hint', True):
        _th_block = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
        if _th_block:
            user_content = user_content + "\n\n" + _th_block
            self.record(ctx, "theorem_hint",
                        "solver 主路径注入定理清单 %d 字符（%d 条）"
                        % (len(_th_block), len(getattr(ctx, 'theorem_hints', []) or [])))
            try:
                if isinstance(getattr(ctx, "metadata", None), dict):
                    ctx.metadata["theorem_hint_injected"] = {
                        "len": len(_th_block),
                        "n": len(getattr(ctx, 'theorem_hints', []) or []),
                    }
            except Exception:  # noqa: BLE001
                pass

    # 2026-09-06 易错点记忆注入（A 档轻量经验，prompts/error_lessons.py）：
    # 命中题型/关键词才注入自查清单（无命中返回空串=零噪音）。
    # 放 user 侧题目之后、_make_one 并行之前——只拼一次，retry 不加倍。
    if getattr(self.config, 'enable_error_lessons', True):
        lessons_block = error_lessons_block(ctx)
        if lessons_block:
            user_content = user_content + "\n\n" + lessons_block
            self.record(ctx, "error_lesson",
                        f"注入历史易错自查清单 {error_lesson_ids(ctx)}")


    # 题型差异化策略注入（v2.6）：
    #   选择题→选项逆推验证；判断题→不确定时合理猜测；
    #   证明题→逐步反复校验；解答题→附带答案结果检测；填空题→只输出结果。
    #
    # 2026-08-30（#45）：老师要求移除按题型分流、让 AI 按自身流程作答
    # （IMO 基本全为证明题，题型分支实测反而拉低证明题正确率）。
    # 此处改为 `enable_question_type_hint` 控制，**默认关闭**；需要 A/B
    # 对比或回归旧行为时置 True 即可，无需改代码。
    # 注：选择题的选项格式化属"输入信息补全"而非策略分流，故始终保留。
    # 2026-09-12 客观题特化（用户要求"保证能检测到题型并采取特化解题技巧"）：
    #   · 客观题（选择 / 判断 / 填空）与证明题的解题流程互斥，给它们注入
    #     特化纪律不触碰上面"证明题不分流"的既有结论；
    #   · 实测依据（official112 基线）：15 道客观题仅对 4 道——093/103 漏读
    #     E 项而缺项、101/111 判断方向反、102/106/110 概念题凭印象作答；
    #   · 开关 `objective_tactic_enabled`（默认 True）可一键回退旧行为，
    #     `enable_question_type_hint`（默认 False）仍可全题型强制开启。
    if getattr(ctx, 'question_type', ''):
        from .question_type import objective_injection
        _is_obj = ctx.question_type in ("选择题", "判断题", "填空题")
        _hint_on = bool(
            getattr(self.config, 'enable_question_type_hint', False)
            or (_is_obj and getattr(self.config, 'objective_tactic_enabled', True))
        )
        _inj = objective_injection(ctx.problem, ctx.question_type, _hint_on)
        if _inj:
            user_content = user_content + _inj
        # 埋点：客观题路由证据（题型 / 是否注入特化纪律 / 注入文本长度）。
        # 目的：让"是否真的走了特化策略"可被事后核验，而不是靠日志反推。
        try:
            if isinstance(getattr(ctx, "metadata", None), dict):
                ctx.metadata["objective_route"] = {
                    "type": ctx.question_type,
                    "hint_injected": bool(_inj),
                    "injection_len": len(_inj),
                }
            if _is_obj:
                logger.info("[客观题路由] 题型=%s 注入=%d字符",
                            ctx.question_type, len(_inj))
        except Exception:  # noqa: BLE001
            pass

    # ★ 2026-09-15 补：**答案形态要求此前只注入子目标 / 蓝图路径，solver 主路径缺失**。
    # 本项目教训「只改一处路径 = 没改」：solver 是一条**独立的生成路径**，
    # 而上面注入的 `objective_injection` 只覆盖客观题（选择/判断/填空），
    # 于是解答题的形态要求（求所有→必须枚举 / 具体值→禁条件式 / 极值→严格性 /
    # 2026-09-15 新增的「哪些→完备性」）在**纯 solver 路径下根本没注入**。
    # 实测对应错题：099（"…离散化方法**有哪些**"，正解为多个，只答了一个）。
    # ⚠ 位置刻意放在 objective 注入之后、_ANSWER_GUIDE 之前：既不在提示词最末尾
    #   （避免模型进入续写模式），也不与客观题特化冲突（该函数对选择题/证明题返回空串）。
    try:
        from .question_type import answer_form_requirement as _afr_sv
        _afr_txt = _afr_sv(ctx.problem, getattr(ctx, 'question_type', '') or '')
        if _afr_txt:
            user_content = user_content + _afr_txt
            if isinstance(getattr(ctx, "metadata", None), dict):
                ctx.metadata["answer_form_injected_solver"] = len(_afr_txt)
            self.record(ctx, "answer_form",
                        "solver 主路径注入答案形态要求 %d 字符" % len(_afr_txt))
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-15（审核 F1④ 修复）：**题目目标锚注入** —— 此前
    #   `render_problem_goal` 只进蓝图/子目标规划，主求解路径看不到
    #   「题目要什么」（审核结论：解答题注意力未聚焦"要求"）。此处把既有
    #   产出接进主生成端（修复断链，非新机制——该函数 docstring 声明
    #   "蓝图 / 子目标规划共用"，主求解属漏配）。
    #   K11 红线由 render_problem_goal 自身保证（答案槽=假设、显式标注不得
    #   当作已证结论）；位置置于形态要求之后、末尾之前（避免续写模式）。
    try:
        from .problem_goal import render_problem_goal as _rpg_sv
        _pg_txt = _rpg_sv(ctx)
        if _pg_txt:
            user_content = user_content + "\n\n" + _pg_txt
            self.record(ctx, "problem_goal",
                        "solver 主路径注入题目目标锚 %d 字符" % len(_pg_txt))
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-18（用户方向一「注意力引导」）：题意理解产出的**关键注意点**
    #   前置注入——放在题目/目标锚之后、其余注入之前，让注意力先锚定陷阱。
    try:
        from contracts.switch_registry import get_bool as _gb
        _on = _gb("attention_focus_enabled")
        _focus = list(getattr(ctx, "attention_focus", None) or [])
        if _on and _focus:
            user_content = user_content + (
                "\n\n【本题关键注意点（先读，求解全程对照）】\n"
                + "\n".join("- " + str(f)[:160] for f in _focus[:5]))
            self.record(ctx, "understanding_chain",
                        "solver 注入关键注意点 %d 条" % len(_focus))
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-17（用户要求「题意误读要用工具/前置理解抓」）：题意理解阶段
    #   发现的**题面歧义**必须显式进求解提示词——要求模型**先声明解读与
    #   题面依据**，不得默默假设（089 型误读的直接来源）。
    try:
        _amb = list(getattr(ctx, "problem_ambiguities", None) or [])
        if _amb:
            user_content = user_content + (
                "\n\n【题面歧义提醒（题意理解阶段发现，必须正面处理）】\n"
                + "\n".join("- " + str(a)[:220] for a in _amb[:4])
                + "\n⚠ 求解前必须**先明确说明你采用哪种解读、依据是题面哪句话**；"
                  "不得默默假设。若不同解读会给出不同答案，请明确说明你的选择。")
            self.record(ctx, "understanding_chain",
                        "solver 注入题面歧义提醒 %d 条" % len(_amb))
    except Exception:  # noqa: BLE001
        pass
    return user_content
