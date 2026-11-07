# -*- coding: utf-8 -*-
"""子目标求解器的 planner 职责簇
（2026-10-08 批3：从 agent/sub_goal_solver.py 逐字搬出 8 个方法）。

搬出后为模块级函数（实例方法保留 `self` 首参，方法体逐字不变；
跨模块引用原文件模块级符号者，在函数体内惰性 import —— 已显式列出）；
原文件类内保留 1 行跳板（`_x = _sgs_planner._x`）⇒ `self._x(...)` /
`SubGoalSolverAgent._x(...)` 契约逐字不变（同 verifier__review.py 批6）。
"""
from __future__ import annotations

import logging
import re

from .base import TaskContext

from prompts.sub_goal import (
    SUBGOAL_PLAN_SYSTEM,
    SUBGOAL_PLAN_USER_TEMPLATE,
)
from prompts.policy import get_domain_hint
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot")


def _is_value_choice(opts: list) -> bool:
    """判断是否为「求值选择型」（C 型）：选项都是**短数值/表达式候选**。

        这类题必须**先整体求解、再与选项比对** —— 逐项判定"选项 5 成立吗"
        是无意义的（选项不是命题，只是候选值）。official112 实测无此类题，
        但比赛题库可能有，故必须支持。

        判据（保守，宁可漏判不可误判）：每个选项 ①不含中文 ②长度 ≤15 ③含数字。
        空列表（提取失败）一律返回 False —— 绝不把"没提取到选项"当成求值型。
        """
    try:
        if not opts:
            return False
        for _l, _t in (opts or []):
            _s = (_t or "").strip()
            if not _s:
                return False
            if re.search(r"[\u4e00-\u9fff]", _s):   # 含中文 → 命题/文字选项
                return False
            if len(_s) > 15:                         # 过长 → 非候选取值
                return False
            if not re.search(r"\d", _s):             # 无数字 → 排除
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _plan_value_choice(self, ctx: TaskContext, opts: list) -> dict:
    """C 型（求值选择）规划：**先整体求解 → 再与选项比对匹配**。

        与 A/B 型的根本区别：A/B 型的选项是**待判定的命题**，逐项判定即可；
        C 型的选项是**候选数值**，必须先算出结果，再找出与之相等的选项。
        """
    self.record(ctx, "subgoal",
                "选择题求值型规划：先整体求解 → 再匹配选项（{} 个选项）"
                .format(len(opts)))
    return {
        "subgoals": [
            {
                "id": 0,
                "title": "整体求解",
                "description": (
                    "**不要**看选项，先独立把题目要求的量算出来。\n"
                    "要求：写出完整算式与推导，最后一行给出『结果：<值>』。"
                ),
                "type": "compute",
                "depends_on": [],
                "expected_output": "结果：<值>",
                "result": "",
            },
            {
                "id": 1,
                "title": "匹配选项",
                "description": (
                    "把【整体求解】得到的结果与下面每个选项逐一比对"
                    "（必要时做数值/符号等价判断）：\n"
                    + "\n".join("  {}. {}".format(l, t) for l, t in opts)
                    + "\n最后一行输出『匹配选项：<字母>』；"
                      "若结果与多个选项等价，则全部写出（如 `AC`）。"
                ),
                "type": "verify",
                "depends_on": [0],
                "expected_output": "匹配选项：<字母>",
                "result": "",
            },
        ],
        "merge_strategy": (
            "直接输出【匹配选项】给出的字母（连写，不要逗号/空格/解释）；"
            "若匹配失败，输出与结果最接近的选项字母。"
        ),
    }


def _plan_choice_by_options(self, ctx: TaskContext) -> dict | None:
    """选择题专用规划：**每个选项一个独立判定子目标**（2026-09-13 用户要求）。

        为什么不让 LLM 规划：096 实测——提示词里早已写了"逐项独立判真"，但模型
        仍然直接综合出 `ACD`（漏判 D 单独成立、误判 A/C）⇒ **提示词不具约束力**。
        故改为代码确定性构造：选项数 N → N 个判定子目标，一个都不能少。

        边界：① 非选择题 → None（走原流程）；② 选项 < 2 → None（判据不足）；
        ③ 选项数 > max_subgoals → None（交回 LLM 规划，避免被截断丢项）。
        """
    from .sub_goal_solver import _itemwise_plan_on
    try:
        # 独立开关（项目惯例：每项优化一个开关，便于 A/B 与逐项审查）。
        # 默认启用（用户 2026-09-13 明确要求：选择题每个选项都判断）；
        # 置 OBJECTIVE_ITEMWISE_PLAN=0 可一键回退到 LLM 规划。
        # ★ 2026-10-07（fix-tail-gaps）：原裸读 env `== "0"` 已改走注册制
        #   （`_itemwise_plan_on()` → `_sw_bool("objective_itemwise_plan")`，实时读）。
        if not _itemwise_plan_on():
            # ★ 2026-10-07（fix-tail-gaps · 静默跳过埋点）：原**静默** return None。
            self.record(ctx, "subgoal",
                        "选择题逐项判定规划已跳过：开关 OBJECTIVE_ITEMWISE_PLAN=0 "
                        "→ 回退 LLM 规划")
            return None
        if getattr(ctx, "question_type", "") != "选择题":
            self.record(ctx, "subgoal",
                        "选择题逐项判定规划已跳过：题型非『选择题』"
                        "（%r）→ 回退 LLM 规划"
                        % (getattr(ctx, "question_type", "") or "",))
            return None
        from .question_type import extract_options
        opts = extract_options(ctx.problem or "") or []
        if len(opts) < 2:
            self.record(ctx, "subgoal",
                        "选择题逐项判定规划已跳过：可用选项 < 2（%d 个），"
                        "不足以逐项判定 → 回退 LLM 规划" % len(opts))
            return None
        # ★ 2026-09-15：改走按档位分档的上限（见 _subgoal_cap）
        _max_sg = self._subgoal_cap(ctx)
        # 基准子目标(1) + N 个判定子目标(N)；超过上限则回退 LLM 规划
        # `_max_sg <= 0` = 不截断 ⇒ 逐项判定恒可用（2026-09-15 赛后口径）
        if _max_sg > 0 and len(opts) + 1 > _max_sg:
            self.record(ctx, "subgoal",
                        "选择题逐项判定规划已跳过：选项数 %d + 基准 1 > 子目标上限 %d，"
                        "为避免截断丢项 → 回退 LLM 规划"
                        % (len(opts), _max_sg))
            return None
        # ---- 题型子类识别（2026-09-13 用户要求：解题逻辑必须随题型而变）----
        #   A 命题判定型（选项是独立陈述，多选）→ 基准 + 逐项判定 → 全选为真
        #   B 定义/事实选择型（选项是互斥候选，单选）→ 基准 + 逐项判定 → 取唯一
        #   C 求值选择型（选项是短数值候选）→ **必须先整体求解**，再与选项比对
        _q = ctx.problem or ""
        if self._is_value_choice(opts):
            return self._plan_value_choice(ctx, opts)
        # ★ 2026-10-14（任务列表 #3）：「单选 / 多选」口径由 2.6 题意理解**一次判定**
        #   （`ctx.problem_goal["n_correct"]`），此处只读结论 —— 不再自己判题面，
        #   也不再把选项清单塞回 problem_analysis（那是题意，不是规划）。
        from .problem_goal import is_single_choice as _is_single_choice
        _is_single = _is_single_choice(ctx, _q)
        subgoals = []
        # ★ 子目标 0：判断基准 —— **只建一次、全员共享**。
        #   对治用户指出的"ab 项是 cd 项的基础"：若让每个选项各自重新推导基准，
        #   既浪费又易导致判定口径不一致（判 A 用一套理解、判 C 用另一套）。
        subgoals.append({
            "id": 0,
            "title": "建立判断基准",
            "description": (
                "先**只**建立本题的判断依据，**不要**在这里判定任何选项。\n"
                "⚠ **严禁**在本步输出 `\\boxed{...}`、『最终答案』『答案是』"
                "『综上，选』等**最终答案形态**的内容 —— 本步只提供依据，"
                "最终答案由后续汇总步骤给出。实测 #098 因本步违规吐 `\\boxed{D}`，"
                "导致该选项判定子目标照抄基准、整个判定链失效。\n"
                "按题目需要输出以下之一（或多项）：\n"
                "· 涉及定义/定理 → 写出相关定义或定理的**完整原文**；\n"
                "· 涉及对应关系（“X 变为 Y”“增大/减小”“系数/常数项”"
                "“约束/目标”等）→ 写出**完整映射表**，把原对象的每个元素"
                "与变换后的元素逐项对应；\n"
                "· 涉及计算 → 先算出**核心结果**（写出算式与结果）；\n"
                "· 涉及判定标准 → 写出教材/规范中的判定标准原文。\n"
                "这段基准是后续所有选项判定的**共同依据**，务必准确、可核验。"
            ),
            "type": "derive",
            "depends_on": [],
            "expected_output": "判断依据（定义原文 / 映射表 / 核心结果 / 判定标准）",
            "result": "",
        })
        for _i, (_lab, _txt) in enumerate(opts, 1):
            subgoals.append({
                "id": _i,
                "title": "判定选项 {}".format(_lab),
                "description": (
                    "基于【建立判断基准】得到的定义原文 / 映射表 / 核心结果，"
                    "**只**判定选项 {} 这一个选项的陈述是否成立，不要讨论其他选项。\n"
                    "选项 {}：{}\n\n"
                    "【判定规范 · 必须遵守】\n"
                    "A. 先判断该选项属于哪类陈述，再选对应方法：\n"
                    "   ① 定义/定理类 → 逐词比对定义或定理原文，不得凭印象；\n"
                    "   ② 计算/等式类 → 代入数值或符号**实际验算**，写出算式；\n"
                    "   ③ 存在/构造类 → 给出具体构造，或给出明确反例；\n"
                    "   ④ **对应关系类**（“X 变为 Y”“由 A 得到 B”“增大/减小”"
                    "“左端/右端”“系数/常数项”“约束/目标”“前件/后件”）→ "
                    "**必须先列出映射表**，把原对象的每个元素与变换后的元素逐项"
                    "对应，再核对该选项的表述是否与映射表一致；"
                    "严禁凭“大体相关 / 看起来对”判为正确。\n"
                    "B. 含绝对化措辞（一定/必然/都/仅/只能/唯一/所有）时，"
                    "必须给出**无反例的论证**或**具体反例**，不得只写“不一定”。\n"
                    "C. 依据必须可核验：写出定义/定理名称或原文，或给出算式/反例。\n\n"
                    "【输出格式】开头一行写“结论：正确”或“结论：错误”，"
                    "随后用 1–2 句给出依据。"
                ).format(_lab, _lab, _txt),
                "type": "verify",
                "depends_on": [0],
                "expected_output": "选项 {}：正确/错误 + 理由".format(_lab),
                "result": "",
            })
        self.record(ctx, "subgoal",
                    "选择题逐项判定规划：{} 个选项 → {} 个判定子目标"
                    "（代码强制，未走 LLM 规划）".format(len(opts), len(subgoals)))
        return {
            "subgoals": subgoals,
            "merge_strategy": (
                ("【单选型】" if _is_single else "【多选型】")
                + "逐项汇总（不得跳步）：\n"
                + (
                    "本题问法指向**单选**（“哪一项 / 最合适 / 定义是 / "
                    "通常采用”等）：在所有『结论：正确』的选项中，选择"
                    "**最准确、最直接回答题干**的一个；若只有一个判对则直接"
                    "输出它；若多个判对，输出最贴合题干的那一个，"
                    "并说明为何排除其余。\n"
                    if _is_single else
                    "本题问法指向**多选**（“哪些 / 正确的有”等）：把所有"
                    "『结论：正确』的选项字母**连写**输出（如 `ABD`），"
                    "不要逗号/空格。\n"
                )
                + "复核要求：① 对每个被判『错误』的选项，回看其依据是否真的成立；"
                  "② 对含绝对化措辞却判『正确』的选项，确认确实无反例；"
                  "③ **对应关系类选项**（“X 变为 Y”式）必须已完成映射表核对，"
                  "未核对的不计入正确项；④ **禁止**跳过逐项判定直接给答案。"
            ),
        }
    except Exception as _e:  # noqa: BLE001
        logger.warning("选择题逐项判定规划失败，回退 LLM 规划: %s", _e)
        return None


def _subgoal_cap(self, ctx) -> int:
    """当前题的子目标数上限（★ 2026-09-15 新增：按档位分档）。

        取值优先级：
          ① `config.max_subgoals_by_tier[ctx.tier]`（新，按档位）
          ② `config.max_subgoals`（旧，全局单一值，向后兼容）
          ③ 全局兜底 6

        语义：**返回 0 或负数 = 不截断**（与 `cap_subgoals_keep_last` 一致）。

        背景：此前 `max_subgoals` 是全局单一值 6，而蓝图提示词已写明
        "简单 3~5 / 中等 5~12 / **难题 12~30**" ⇒ **提示词与代码矛盾**
        （模型给难题画 12~30 个，代码一律切到 6）。
        现按档位给不同上限：deep 是难题档，题更复杂 → 允许更多子目标。
        """
    tier = str(getattr(ctx, "tier", "") or "")
    by_tier = getattr(self.config, "max_subgoals_by_tier", None)
    if isinstance(by_tier, dict) and tier in by_tier:
        try:
            return int(by_tier[tier])
        except (TypeError, ValueError):
            pass
    v = getattr(self.config, "max_subgoals", 6)
    return 6 if v is None else int(v)


def _plan_subgoals(self, ctx: TaskContext) -> dict | None:
    """调用 LLM 生成子目标规划 JSON"""
    # 2026-09-13 用户要求「选择题就每个选项都判断」：选择题**不走 LLM 规划**，
    # 由代码确定性构造 N 个判定子目标（每选项一个），一个都不漏。
    # 依据：096 实测——提示词已写"逐项独立判真"，模型仍直接综合出 ACD
    # （漏判 D、误判 A/C）⇒ 提示词不具约束力，必须**代码强制**。
    _choice_plan = self._plan_choice_by_options(ctx)
    if _choice_plan is not None:
        return _choice_plan
    # #27 Blueprint DAG（LEAP Stage 1）：use_blueprint_dag 开启时先由
    # BlueprintPlanner 生成 AND-OR DAG（依赖驱动分解），再转子目标序列；
    # 生成失败回退到原有 LLM 规划（不损失候选来源）。
    if getattr(self.config, "use_blueprint_dag", False) or \
            getattr(self.config, "use_blueprint", False):
        blueprint_plan = self._plan_from_blueprint(ctx)
        if blueprint_plan is not None:
            return blueprint_plan
        self.record(ctx, "blueprint", "Blueprint DAG 规划失败，回退到 LLM 子目标规划")
    domain = ctx.domain or ""
    domain_hint = get_domain_hint(domain) if domain else ""
    # v2.9 遗留：原 Lean 前置形式化注入（formal_spec/formal_gaps/leansearch
    # 定理检索）已随 2026-09-06 去 Lean 化移除——平台无 Lean，注入恒为空。
    problem_text = ctx.problem
    # ★ 2026-10-14（五道链·4.6_rewrite 重写通道）：仅当 ctx.rewrite_context
    #   存在（池空重写时写入「错误情况 + 诊断」）才追加进**"题目"占位符内部**；
    #   默认无该字段 ⇒ 行为逐字不变（注入位置遵循 objective_injection 纪律）。
    _rw_ctx = getattr(ctx, "rewrite_context", None)
    if _rw_ctx:
        problem_text = problem_text + "\n\n" + str(_rw_ctx)[:2500]
    # 2026-09-12 客观题特化（三处同源注入之①：规划阶段）：
    # 子目标主路径 2.7 对**所有档位默认先行**（enable_subgoal_main_path 默认
    # True），客观题的答案完全可能由本路径产出——只在 solver 里注入等于没注入。
    # 拼接位置固定在"题目"占位符内部，**不追加到提示词末尾**（历史教训
    # algebra-075：末尾追加破坏收尾结构 → 模型进入续写模式、泄漏占位符）。
    try:
        from .question_type import objective_injection as _obj_inj
        problem_text = problem_text + _obj_inj(
            ctx.problem or "", getattr(ctx, "question_type", "") or "",
            bool(getattr(self.config, "objective_tactic_enabled", True)))
    except Exception:  # noqa: BLE001
        pass
    # 2026-09-13 B0：答案形态要求**前置注入**（生成阶段就告知，而非事后重问）。
    # 依据：事后闸门在 formatter（流程末段）实测全部"时间不足跳过"——
    # deadline 被 tier_budget 收紧到 1800s，跑到末段必然剩余不足。
    try:
        from .question_type import answer_form_requirement as _afr
        problem_text = problem_text + _afr(
            ctx.problem or "", getattr(ctx, "question_type", "") or "")
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-14（任务列表 #2）：让**定理线索真正进入子目标规划**。
    #   检索已在 `SubGoalSolver.run` 体首完成（2.7 内），此处按环节取条数
    #   （默认 25，见 `theorem_hint.DEFAULT_TOPK_BY_STAGE["subgoal"]`）——
    #   用户原话：「用 leansearch 去搜索定理然后将定理返回给大模型，让大模型的
    #   推理方向更准确」。
    #   ⚠ 必须拼进 `problem_text`（即"题目"占位符**内部**），不得追加到提示词
    #     末尾 —— 与 objective_injection / answer_form_requirement 同一纪律
    #     （历史教训 algebra-075：末尾追加破坏收尾结构 → 模型进入续写模式）。
    #   ⚠ 无检索结果/开关关闭 ⇒ `render_for_stage` 返回空串，此处零成本、行为不变。
    try:
        from .theorem_hint import render_for_stage as _rfs
        _th_block = _rfs(ctx, "subgoal")
        if _th_block:
            problem_text = problem_text + "\n\n" + _th_block
    except Exception:  # noqa: BLE001  检索是增益，绝不阻断
        pass

    user_msg = SUBGOAL_PLAN_USER_TEMPLATE.format(
        domain_hint=domain_hint,
        problem=problem_text,
    )

    last_resp = None
    for attempt in range(2):  # v2.6.1：3→2 次（解析失败说明 LLM 输出结构异常，
        # 重试成功率低；省下时间给真正有意义的求解步骤）
        # v2.4.1：prefill「{"」引导直接输出规划 JSON，抑制 CoT
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": SUBGOAL_PLAN_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                '{"',
            ),
            0.2, 32768,
        )
        if resp:
            resp = stitch('{"', resp)
            last_resp = resp
        if resp is None:
            continue
        raw = self._extract_json(resp)
        if raw is None:
            logger.warning("SubGoal plan: JSON parse failed on attempt %d", attempt + 1)
            continue
        # ★ 2026-09-15：改走按档位分档的上限（见 _subgoal_cap）
        _sg_cap = self._subgoal_cap(ctx)
        subgoals = self._parse_subgoal_plan(
            raw, 6 if _sg_cap is None else int(_sg_cap))
        if subgoals is None:
            logger.warning("SubGoal plan: invalid subgoals on attempt %d", attempt + 1)
            continue
        return {
            "subgoals": subgoals,
            "merge_strategy": raw.get("merge_strategy", ""),
        }

    # v2.6.1 兜底：LLM 反复输出不合规 JSON 时，不再返回 None（损失整个候选来源），
    # 而是构造一个最小可用 plan（单步求解整道题），让 SubGoalSolver 至少产出 1 个
    # 候选；同时把最后一次响应记到 trace 便于排查 LLM 实际输出形态。
    self.record(ctx, "subgoal",
                f"子目标规划 2 次尝试均失败; 最后响应片段: {(last_resp or '<None>')[:200]};"
                " 回退到单步求解整道题")
    return {
        "subgoals": [{
            "id": 1,
            "title": "完整求解",
            "description": ctx.problem,
            "type": "compute",
            "depends_on": [],
            "expected_output": "",
            "result": "",
        }],
        "merge_strategy": "直接给出最终答案",
    }


def _plan_from_blueprint(self, ctx: TaskContext) -> dict | None:
    """用 BlueprintPlanner 生成 AND-OR DAG 并转为子目标规划。"""
    try:
        from .blueprint_planner import BlueprintPlannerAgent
    except Exception as e:  # noqa: BLE001
        logger.warning("BlueprintPlanner 导入失败: %s", e)
        return None
    planner = BlueprintPlannerAgent(self.client, self.config)
    # ★ 2026-10-12（批次 2，用户 Q11–Q13）：迭代式多次生成 + 第 4 次自审。
    #   轮数=1 时等价于原行为（零差异），故可由开关精确回退。
    if int(getattr(self.config, "blueprint_multi_gen_rounds", 3) or 3) > 1:
        dag = planner.generate_blueprint_iterative(ctx)
    else:
        dag = planner.generate_blueprint(ctx)
    if dag is not None and getattr(self.config, "blueprint_self_review", True):
        dag = planner.self_review_blueprint(ctx, dag)
    if dag is None:
        # ★ 2026-10-07（fix-tail-gaps · 静默跳过埋点）：原**静默** return None。
        self.record(ctx, "blueprint",
                    "Blueprint 路径已跳过：蓝图生成返回空（generate_blueprint=None）"
                    "→ 回退 LLM 子目标规划")
        return None
    # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：
    #   原「骨架编排层评审循环」（`enable_skeleton_review`）与「求解前 DAG 强制
    #   评审门」（`dag_replan_gate`）两块**整块撤除** —— 两者均默认关闭、从未在
    #   生产启用（仅 `deploy/run_112.py::arm2` 复测臂引用），其宿主模块
    #   `sub_goal_solver__review.py` / `agent/skeleton_reviewer.py` 已删除。
    #   蓝图生成后**直接**进入子目标求解（自审 `blueprint_self_review` 仍保留）。
    # 2026-09-15 同源修复：此前不传参 ⇒ 恒用默认 6，**config.max_subgoals
    # （含 0915 新增的 --max_subgoals CLI）对 blueprint 路径完全无效**。
    # 实测 003/074 都走 Blueprint DAG ⇒ 这些题的旋钮是死的。现与 LLM 规划
    # 路径（_parse_subgoal_plan）取同一个 config 字段。
    # ★ 2026-09-15：改走按档位分档的上限（见 _subgoal_cap）
    plan = dag.to_subgoal_plan(
        self._subgoal_cap(ctx),
        with_deps=getattr(self.config, "blueprint_deps_enabled", True),
        or_expand_all=getattr(self.config, "blueprint_or_expand_all", False))
    # ★ 2026-10-15（审核修复 F3）：生产路径补写 ctx.blueprint_plan（此前仅
    #   BlueprintPlannerAgent.run 写，而生产走本函数直接生成 ⇒ 03_subgoals.plan
    #   恒空、rewrite 复位亦空）。直接复用同一 plan 对象，零额外成本。
    try:
        ctx.blueprint_plan = plan
    except Exception as _e_bp:  # noqa: BLE001
        self.record(ctx, "blueprint",
                    "blueprint_plan 导出失败: %s: %s"
                    % (type(_e_bp).__name__, str(_e_bp)[:100]))
    if not plan.get("subgoals"):
        # 退化情形：整张 DAG 都是前瞻引理（无 required 叶子）。
        # 此时无子目标可解，仍视为规划失败并回退 LLM 规划（不静默产出空解）。
        logger.warning("Blueprint DAG 无可用叶子子目标")
        # ★ 2026-10-07（fix-tail-gaps · 静默跳过埋点）：trace 侧原无记录
        #   （仅 logger.warning，评测零痕迹）→ 补足。
        self.record(ctx, "blueprint",
                    "Blueprint 路径已跳过：DAG 无可用叶子子目标（整树均为前瞻引理）"
                    "→ 回退 LLM 子目标规划")
        return None
    # ★ 前瞻引理规划（LEAP §2.3）：anticipatory 节点不进求解链，
    #   其陈述写入引理记忆（ctx.lemma_repo），供后续子目标提示词的
    #   【已建立的结论】区块引用（见 _solve_subgoal 的 lemma_context）。
    #   ⚠ 用 if 而非无条件赋值：_use_lemma 按领域路由（A/B 实测全开净 0.0pp），
    #   要保持与 _accumulate_lemma 同一开关语义，不做绕过。
    # ★ 2026-09-23：把 LeanSearch 检索到的 Mathlib 定理注入**解题侧**引理记忆。
    #   此前检索结果只喂验证器（`leansearch_inject_verifier`）⇒ 解题器从未见过，
    #   实测 460 条里仅 1% 进入 Lean 代码。必须在**求解前**注入。
    # ★ 2026-10-12 批次 3（用户 D20/D21）：对子目标链做 **R1/R2 机械校验**
    #   （零 LLM）：R2 = inputs 是否引用了不存在/自引用的节点（断链）；
    #   R1 = 叶子节点却依赖前序结论（应继续分解 —— K11 空壳的形态学成因）。
    #   缺口按「题面是否已给」二分（D21）：
    #     · 缺的量**在题面出现过** ⇒ kind=B（题面给了但没读到）⇒ 就地补子目标；
    #     · 否则 ⇒ kind=A（题面确实没给）⇒ 回退蓝图作风险项。
    #   ⚠ 本轮只做**可观测 + 分类**（不自动改蓝图，遵循「先可观测→后改逻辑」）。
    if getattr(self.config, "subgoal_inputs_validation", True):
        try:
            _vres = dag.validate_inputs()
            _prob = str(getattr(ctx, "problem", "") or "")
            _gaps = []
            for _it in (_vres.get("broken") or []):
                _gaps.append({"kind": "A" if _it.get("ref") not in _prob else "B",
                              "detail": "%s: %s" % (_it.get("reason"), _it.get("ref")),
                              "node": _it.get("id")})
            for _it in (_vres.get("leaky_leaves") or []):
                _gaps.append({"kind": "B", "detail": _it.get("reason"),
                              "node": _it.get("id")})
            ctx.subgoal_chain_gaps = _gaps
            self.record(ctx, "subgoal_inputs",
                        "inputs 校验：%d 节点，断链 %d、疑似该分解的叶子 %d（缺口 A%d/B%d）"
                        % (_vres.get("n", 0), len(_vres.get("broken") or []),
                           len(_vres.get("leaky_leaves") or []),
                           sum(1 for g in _gaps if g["kind"] == "A"),
                           sum(1 for g in _gaps if g["kind"] == "B")))
        except Exception as _e_v:  # noqa: BLE001  校验是增益，绝不阻断
            self.record(ctx, "subgoal_inputs",
                        f"inputs 校验异常（跳过）: {type(_e_v).__name__}: {str(_e_v)[:120]}")
    self._inject_mathlib_theorems(ctx)
    self._inject_anticipatory_lemmas(ctx, plan.get("anticipatory_lemmas") or [])
    # 2026-09-06 去 Lean 化：原 LEAP Stage 2 整树 Lean 搭桥审核
    # （_audit_blueprint_tree / lean_translator / lean_refiner）已移除——
    # 平台无 Lean，整树翻译+编译只空转。骨架评审（上方 enable_skeleton_review）
    # 是 LLM 规划质量门，与 Lean 无关，保留。
    self.record(ctx, "blueprint",
                f"Blueprint DAG → {len(plan['subgoals'])} 个子目标 "
                f"(根={dag.root_id}, 节点={len(dag.nodes)}, "
                f"前瞻引理={len(plan.get('anticipatory_lemmas') or [])})")
    return plan


def _inject_mathlib_theorems(self, ctx: TaskContext) -> int:
    """把 LeanSearch 检索到的 Mathlib 定理写进引理记忆，**让解题器真正看得到**。

        ★ 2026-09-23 新增，回答用户"mathlib 的定理是不是和本题有关、哪些帮上了忙"。

        根因（0923 环节效能审计实测）：460 条检索结果里只有 **4 条（1%）** 出现在
        Lean 代码中。原因不是"定理无关"，而是——
        `verifier._prepare_theorem_context` 把检索结果注入的是**验证器**的提示词
        （`leansearch_inject_verifier`），而**解题器从未收到过这些定理**
        ⇒ 检索在"解题"这个最该用它的环节**完全缺席**。

        本方法在子目标求解**之前**检索一次，并把命中的定理写进 `ctx.lemma_repo`
        （与 `_accumulate_lemma` 同格式），由既有注入通路喂给每个子目标步。

        纪律：异常/超时/后端不可用一律静默返回 0，绝不阻断求解。
        """
    try:
        if not getattr(self.config, "inject_mathlib_to_solver", True):
            # ★ 2026-10-07（fix-tail-gaps · 静默跳过埋点）：原**静默** return 0。
            self.record(ctx, "leansearch",
                        "解题前 Mathlib 注入已跳过：开关 "
                        "inject_mathlib_to_solver=0")
            return 0
        if not getattr(self.config, "use_leansearch", False):
            self.record(ctx, "leansearch",
                        "解题前 Mathlib 注入已跳过：LeanSearch 总开关 "
                        "use_leansearch 未开启（默认关）")
            return 0
        if ctx.is_time_critical():
            self.record(ctx, "leansearch",
                        "解题前 Mathlib 注入已跳过：当前处于时间紧迫区")
            return 0
        if getattr(ctx, "_mathlib_injected", False):
            self.record(ctx, "leansearch",
                        "解题前 Mathlib 注入已跳过：本轮回合已注入过（去重短路）")
            return 0
        cap = int(getattr(self.config, "leansearch_max_calls_per_q", 2) or 0)
        used = int(getattr(ctx, "_leansearch_calls", 0) or 0)
        if cap and used >= cap:
            self.record(ctx, "leansearch",
                        "解题前注入跳过：单题检索次数已用 %d/%d" % (used, cap))
            return 0
        from tools.lean_local.lean_search import MathlibTheoremSearcher
        searcher = getattr(self, "_ls_searcher_solver", None)
        if searcher is None:
            searcher = MathlibTheoremSearcher()
            self._ls_searcher_solver = searcher
        k = max(1, min(10, int(getattr(self.config, "leansearch_top_k", 10) or 10)))
        ctx._leansearch_calls = used + 1
        self.note_mathlib_search(ctx)
        res = searcher.search((getattr(ctx, "problem", "") or "")[:800], limit=k) or {}
        items = res.get("results") or []
        added = 0
        repo = getattr(ctx, "lemma_repo", None)
        if not isinstance(repo, list):
            repo = []
            ctx.lemma_repo = repo
        for it in items:
            nm = ""
            if isinstance(it, dict):
                nm = str(it.get("name") or it.get("theorem") or it.get("id") or "").strip()
            else:
                nm = str(it).strip()
            if not nm:
                continue
            tail = nm.split(".")[-1]
            entry = "Mathlib 定理: %s（可尝试在 Lean 代码中直接引用）" % tail
            if entry not in repo:
                repo.append(entry)
                added += 1
        ctx._mathlib_injected = True
        self.record(ctx, "leansearch",
                    "解题前注入 Mathlib 定理 %d/%d 条到引理记忆（此前只喂验证器）"
                    % (added, len(items)))
        return added
    except Exception as e:  # noqa: BLE001
        logger.debug("解题前 Mathlib 注入异常（忽略）: %s", str(e)[:120])
        return 0


def _inject_anticipatory_lemmas(self, ctx: TaskContext,
                                lemmas: list) -> int:
    """把前瞻引理写入引理记忆（LEAP §2.3 落地，2026-09-15）。

        前瞻引理 = 蓝图生成时提议的"当前不需要、但后续可能用到"的辅助引理
        （论文 Figure 2 虚线边）。它们**不参与当前 AND 节点求解**，
        只作为可复用的已验证陈述挂到记忆里。

        写入形式与 `_accumulate_lemma` 保持一致（`"标题: 陈述"` 字符串），
        这样 `_solve_subgoal` 的现成注入通路无需任何改动即可消费。

        返回实际写入条数。
        """
    if not lemmas:
        return 0
    if not self._use_lemma(ctx):
        # 领域路由关闭时不注入（与 _accumulate_lemma 同规则，
        # 避免"引理注入只在数论开"这条既有实测结论被悄悄绕过）
        # ★ 2026-10-07（fix-tail-gaps · 静默跳过埋点）：原**静默** return 0。
        self.record(ctx, "blueprint",
                    "前瞻引理注入已跳过：领域路由（_use_lemma）判定本域不做引理注入")
        return 0
    added = 0
    for item in lemmas:
        if isinstance(item, dict):
            stmt = str(item.get("statement") or "").strip()
            nid = str(item.get("id") or "").strip()
        else:
            stmt, nid = str(item or "").strip(), ""
        if not stmt:
            continue
        entry = f"前瞻引理{('[' + nid + ']') if nid else ''}: {stmt}"
        if entry not in ctx.lemma_repo:
            ctx.lemma_repo.append(entry)
            added += 1
    if added:
        self.record(ctx, "blueprint",
                    f"前瞻引理入记忆 {added}/{len(lemmas)} 条（不阻塞求解）")
    return added
