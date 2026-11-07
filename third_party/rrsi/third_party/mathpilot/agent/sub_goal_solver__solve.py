# -*- coding: utf-8 -*-
"""子目标求解器的 solve 职责簇
（2026-10-08 批3：从 agent/sub_goal_solver.py 逐字搬出 9 个方法）。

搬出后为模块级函数（实例方法保留 `self` 首参，方法体逐字不变）；
跨模块引用原文件模块级符号者（含 `SubGoalSolverAgent` 与其常量/函数）
在**函数体内惰性 import**（避免循环导入）—— 下表已显式列出。
原文件类内保留 1 行跳板（`_x = _sgs_solve._x`）⇒ `self._x(...)` /
`SubGoalSolverAgent._x(...)` 契约逐字不变（同 verifier__review.py 批6）。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .base import TaskContext

from prompts.sub_goal import (
    SUBGOAL_CONCLUSION_CONTRACT,
    SUBGOAL_STEP_SYSTEM,
    SUBGOAL_STEP_USER_TEMPLATE,
    SUBGOAL_TOOL_INSTRUCTIONS,      # 2026-10-15：蓝图 tool 指派 → 工具指令
    SUBGOAL_LEAN_SIG_TEMPLATE,      # 2026-10-15：lean 节点已带形式化签名 → 只补证明体
    SUBGOAL_SUCCESSION_TEMPLATE,    # 2026-10-15：承接前提块
)
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot")


def _improve_subgoal_result(self, ctx, sg, prev_result):
    """子目标结果的**无条件**自改进一轮（2026-10-13，用户 E24）。

    与 `solver__fallback.improve_candidates`（主求解**候选列表**）不同：
    本函数作用于**单个子目标的文本结果**，用于「让大模型反复写一步」。
    返回新结果；失败/空 ⇒ 返回空串（调用方据此保留原结果）。
    """
    try:
        from prompts.sub_goal import (SUBGOAL_IMPROVE_SYSTEM as _IS,
                                     SUBGOAL_IMPROVE_USER_TEMPLATE as _IU)
    except ImportError:
        return ""
    _sg_title = str(sg.get("description") or sg.get("title") or "")
    try:
        raw = self.llm(ctx, [
            {"role": "system", "content": _IS},
            {"role": "user", "content": _IU.format(
                problem=getattr(ctx, "problem", "") or "",
                subgoal=_sg_title[:500],
                prev_result=str(prev_result)[:6000])},
        ], 0.2, 8192)
    except Exception as e:  # noqa: BLE001
        logger.warning("[subgoal_improve] LLM 异常: %s", str(e)[:120])
        return ""
    return str(raw or "").strip()


def _solve_subgoal(self, ctx: TaskContext, sg: dict,
                   plan_summary: str, prev_results: str,
                   extra_hint: str = "") -> str:
    """求解单个子目标，返回结果文本。

        v2.7：计算类子目标（compute/derive）求解后用 AnswerOracle 做客观
        sanity check，若结果明显非法（不可解析为数学表达式），带反馈重解一次，
        实现"每步 oracle 校验"（Plan-and-Execute + oracle-in-the-loop）。

        v2.10（2026-08-29）：use_lemma_accumulation 开启时，把已求得的
        引理列表注入子目标提示词（"已建立的结论"），让后续子目标直接复用，
        避免重复推导（D6 引理积累钥匙的真正落地点）。

        v2.11（2026-09-06 老师建议）：extra_hint 承载 0-LLM 校验（S1-lite）
        的失败反馈——截断/Lean 代码片编译错误，随重解请求带回，让模型
        带反馈修正后重新给出【本步结果】（验证-精炼下沉到子目标级）。
        """
    from .sub_goal_solver import _check_protocol_guard_on, _check_protocol_violations
    # 引理注入：已求得的子目标结论作为"前置引理"提供。
    # **必须放在提示词中部（最终指令之前），不能追加在末尾**——
    # 2026-08-29 A/B 实测：追加在末尾会改变提示词收尾结构，使模型
    # 进入"续写模式"，答案泄漏 `[续写]` 占位符（algebra-075 因此从对变错）。
    # ★ 2026-10-13 **接线 B**（方案 v4 §3.5.6）：把「题意缺口」并入 extra_hint。
    #   一次对**所有**子目标生效（缺口来自 ctx、与具体子目标无关）⇒ 无需改各调用方。
    #   为什么：`ctx.subgoal_chain_gaps`（批次 3）与 `ctx.understanding_gaps`（批次 1）
    #   此前**只落盘、无消费者**（改动效果审核报告点名）。
    _gap_block = _render_gap_block(ctx)
    if _gap_block:
        extra_hint = ((extra_hint + "\n\n" + _gap_block)
                      if extra_hint else _gap_block)
    # ★ 2026-10-13（§3.4 的**消费者**）：把题型分流转成「答题形态提示」注入求解。
    #   分流是**针对性增强**（如选择题必须给选项字母），**不裁剪**任何环节。
    try:
        from agent.question_type import render_type_path_hint as _rtp
    except ImportError:
        _rtp = None
    if _rtp is not None:
        try:
            _tph = _rtp(getattr(ctx, "type_path", None))
            if _tph:
                extra_hint = ((extra_hint + "\n\n" + _tph)
                              if extra_hint else _tph)
        except Exception:  # noqa: BLE001  提示是增益，绝不阻断
            pass

    user_msg = self._prepare_step_prompt(
        ctx, sg, plan_summary, prev_results, extra_hint)

    step_result = self._call_step(ctx, user_msg, sg=sg)

    # ★★ 2026-10-07（问题 2.9）：**`<check>` 协议校验 + 定向重问**。
    #   实测（smoke10 题 000）：大量子目标把**中文句子**塞进 `<check>`
    #   （`<check>图 G 是弱连通的</check>`）⇒ `is_parseable=False` ⇒
    #   客观判定**一条都出不来**（`n_judged = 0/73` 的直接原因之一）。
    #   处理（先读既有契约 `SUBGOAL_CONCLUSION_CONTRACT` 再强化，不另起一套）：
    #     ① 违规定性：`<check>` 内含非可核验对象（中文/文字/非单等式）；
    #     ② **留痕**：trace 记 `check_protocol`（违反内容 + 子目标号），
    #        并把违反项写进 sg，随 `_build_trace_item` 进 diag（可见，不静默）；
    #     ③ **一次定向重问**：把违规内容 + 正确写法（`本步结论: 判定 = 真`
    #        或纯数学等式）作为反馈重发；重问后合规即采用新结果。
    #     ⚠ 不做"重问 N 次"——单步成本高（本项目多次实测：生成侧吃光预算
    #       → 验证阶段饿死）；一次不成就保留并留痕，交由下游处置。
    if (_check_protocol_guard_on() and step_result
            and not step_result.startswith("[子目标")):
        _viol = _check_protocol_violations(step_result)
        if _viol:
            self.record(
                ctx, "check_protocol",
                "子目标#%s 违反 <check> 协议：%d 处 <check> 内含非可核验对象"
                "（应为纯数学等式；装句子 = 该步不可验证）。违反内容示例：%r"
                % (sg.get("id"), len(_viol), _viol[0][:60]))
            try:
                sg["check_protocol_violations"] = _viol[:3]
            except Exception:  # noqa: BLE001
                pass
            if not ctx.gen_time_up():
                _fix_msg = user_msg + (
                    "\n\n[协议违规：`<check>` 内容不是可核验对象]\n"
                    "你写的 `<check>%s</check>` 里是**文字描述**，系统无法核验。\n"
                    "`<check>` 只允许装**一个纯数学等式**（仅数字/变量/运算符 "
                    "+ - * / ^ ( )，恰含一个 `=`，不得含中文或文字）。\n"
                    "修正方式：\n"
                    "  · 自然语言命题（如「图 G 是弱连通的」）改写为"
                    "【本步结果】首行 `本步结论: 判定 = 真`；\n"
                    "  · `<check>` 只保留真正的数值/代数等式，"
                    "没有可核验等式就**不要写 `<check>`**。\n"
                    "请修正后重新给出【本步结果】。"
                    % (_viol[0][:80],))
                _retry = self._call_step(ctx, _fix_msg, sg=sg)
                if (_retry and not _retry.startswith("[子目标")
                        and not _check_protocol_violations(_retry)):
                    self.record(ctx, "check_protocol",
                                "子目标#%s 定向重问后 <check> 协议合规"
                                % sg.get("id"))
                    step_result = _retry
                else:
                    self.record(ctx, "check_protocol",
                                "子目标#%s 定向重问后仍违规/无改善 → 保留原结果"
                                "并留痕（不再重问）" % sg.get("id"))

    # 每步 oracle 校验：仅对计算类子目标（预期数值/表达式结果）做客观检查
    # 2026-09-03 老师：子目标是**简化求解**的，不是复杂化的。实测 deep 档
    # 每步 110s（主求解 60s + oracle 10-70s + 重试 60s）→ 8 步吃掉 882s，
    # 验证/Lean 闸门全没时间。oracle 是**同源自评**（价值低、耗时高），
    # 默认关闭（sub_goal_oracle_check=False 开启）。
    # 注意：这里**不是强制结束**——失败重试（_solve_subgoal 的上下文重试）
    # 仍保留，子目标一定会尽力做完（老师："强制结束就等于错误"）。
    sg_type = sg.get("type", "compute")
    if (getattr(self.config, "sub_goal_oracle_check", False)
            and sg_type in ("compute", "derive")
            and step_result
            and not step_result.startswith("[子目标")):
        oracle_fb = self._oracle_check_step(step_result)
        if oracle_fb and not ctx.gen_time_up():
            retry_msg = user_msg + (
                f"\n\n[上一步结果客观校验未通过] {oracle_fb}\n"
                f"请修正错误后重新给出【本步结果】。"
            )
            retry_result = self._call_step(ctx, retry_msg, sg=sg)
            if retry_result and not retry_result.startswith("[子目标"):
                return retry_result
    return step_result


def _node_contract_on() -> bool:
    """蓝图**节点契约**开关是否生效（2026-10-15）：tool 指派 + 承接前提。

    默认 True。走 `contracts.switch_registry`（环境变量 > 配置 > 默认），
    与全项目开关口径一致；注册表不可用时退回 True（研究期默认启用新契约）。
    """
    try:
        from contracts.switch_registry import get_bool
        return bool(get_bool("blueprint_node_contract"))
    except Exception:  # noqa: BLE001  注册表不可用 → 按默认启用
        return True


def _render_succession_block(sg: dict) -> str:
    """渲染「本步承接前提」块（2026-10-15，用户要求「前一个是后一个的条件」）。

    `sg["inputs_from"]` 由 `BlueprintDAG.to_subgoal_plan` 下传：
    `[{"id": 上游子目标序号, "use": 本步怎么把它当条件用,
       "produces": 上游产出物, "title": 上游标题}]`。

    为什么必须单独渲染一段：此前"承接"只活在蓝图的 `inputs[i].use` 里，
    **从没进过求解提示词** ⇒ 模型看到的是一串上游结果文本，却不知道本步该把
    其中哪一条当作**条件**来用 ⇒ 承接关系在提示词层是空的（有声明、无兑现）。
    """
    items = list((sg or {}).get("inputs_from") or [])
    if not items:
        return ""
    lines = []
    for it in items[:4]:
        if not isinstance(it, dict):
            continue
        _id = it.get("id", "?")
        _title = str(it.get("title") or "").strip()
        _use = str(it.get("use") or "").strip()
        _prod = str(it.get("produces") or "").strip()
        seg = "  · 承接子目标 #%s「%s」" % (_id, _title)
        if _prod:
            seg += " 产出的「%s」" % _prod
        if _use:
            seg += " —— 本步 %s" % _use
        lines.append(seg)
    if not lines:
        return ""
    return SUBGOAL_SUCCESSION_TEMPLATE.format(items="\n".join(lines))


def _prepare_step_prompt(self, ctx: TaskContext, sg: dict,
                          plan_summary: str, prev_results: str,
                          extra_hint: str) -> str:
    """单步提示词准备段（2026-10-08 批2：从 `_solve_subgoal` 原地拆出）。

        装配已建立结论（lemma_context）/ 客观题特化注入 / 剥离 boxed 指令的题面 /
        title·description 的 K11 断言数值掩码 / `SUBGOAL_STEP_USER_TEMPLATE`
        → 返回 `user_msg`。段体**逐字搬**，record 步序不变。
        """
    from .sub_goal_solver import _k11_mask_on, _mask_asserted_values
    lemma_context = ""
    if self._use_lemma(ctx):
        lemmas = list(getattr(ctx, "lemma_repo", []) or [])
        if lemmas:
            lemma_block = "\n".join(f"- {l}" for l in lemmas[-8:])
            lemma_context = (
                f"\n【已建立的结论（可直接引用，无需重新推导）】\n"
                f"{lemma_block}\n")

    # ② 原子目标级 leansearch 独立检索（LeanSearch v2 论文）已随
    # 2026-09-06 去 Lean 化移除——平台无外网/无 Lean，检索恒空转。

    # 2026-09-12 客观题特化（三处同源注入之②：子目标逐步求解）——这是子目标
    # 链里**真正产出内容**的一步，客观题答案可能直接来自本步结果。
    _obj_inj = ""
    try:
        from .question_type import objective_injection as _oi
        _obj_inj = _oi(
            ctx.problem or "", getattr(ctx, "question_type", "") or "",
            bool(getattr(self.config, "objective_tactic_enabled", True)))
    except Exception:  # noqa: BLE001
        _obj_inj = ""
    # ★ 2026-09-14（#103「丢 E」真根因修复）：剥离题面的
    #   「Remember to put your final answer within \boxed{}.」指令。
    #   实测该句**112/112 题都有**（题库统一附加），它会随题面进入**每个
    #   中间子目标**的 prompt ⇒ ① 选项判定子目标全都输出 `\boxed{A}`；
    #   ② **最后一个选项**直接吐**最终答案**（#103「判定选项 E」→
    #   `\boxed{ABCD}`）⇒ 聚合层认不出它是 E 的判定 ⇒ 静默丢项。
    #   该指令只对**最终汇总**步骤有效（merge 仍用原题、保留 boxed 要求），
    #   故此处对子目标 prompt 使用剥离版题面。
    try:
        from .question_type import strip_answer_format_directive as _safd
        _prob_for_step = _safd(ctx.problem or "")
    except Exception:  # noqa: BLE001
        _prob_for_step = ctx.problem or ""
    # ★ 2026-10-04（K11 注入点守卫）：本模板 **逐字** 注入的字段必须全部过掩码。
    #   背景：蓝图节点 `statement` 经 `blueprint_planner.to_subgoal_plan`（:302-303）
    #   被原样写成子目标 `title` / `description`，再由本模板送进每步求解提示词
    #   （见 prompts/sub_goal.py:229-231）⇒ 蓝图若写「证明最小值为 4030，故答案为
    #   4030」就等于**指令模型输出 4030**（K11，已数据确认的唯一过半根因 41.4%）。
    #   原守卫只覆盖 merge_strategy / problem_analysis（生产点；后者已于
    #   2026-10-14 随任务 #3 删除），此处为最后一道关口。
    #   只改局部变量，**不动 sg 本体**（其它消费者与 trace 不受影响）。
    _desc = str(sg.get("description") or "")
    _title = str(sg.get("title") or "")
    if _k11_mask_on():
        _desc, _n_desc = _mask_asserted_values(_desc)
        _title, _n_title = _mask_asserted_values(_title)
        _n_mask = _n_desc + _n_title
        if _n_mask:
            self.record(ctx, "blueprint_lock_guard",
                        f"子目标注入字段（title/description）内嵌 {_n_mask} 处断言数值"
                        " → 已屏蔽为占位符（K11 注入点守卫）")
    # ★ 2026-10-15（用户要求「前一个子目标可以作为后一个子目标的条件」）：
    #   把蓝图声明的承接关系渲染成**显式前提块**，注入【前面步骤的结果】区。
    #   位置刻意选在该区**顶部**（承接前提要先于上游结果原文被读到）；
    #   **不放 user 末尾** —— 2026-08-29 教训：改动提示词收尾结构会让模型
    #   进入续写模式、泄漏占位符。
    #   承接块里含上游 `produces`，同样可能是被写死的待求量 ⇒ 一并过 K11 掩码。
    if _node_contract_on():
        _succ = _render_succession_block(sg)
        if _succ:
            if _k11_mask_on():
                _succ, _n_succ = _mask_asserted_values(_succ)
                if _n_succ:
                    self.record(ctx, "blueprint_lock_guard",
                                f"子目标承接前提块内嵌 {_n_succ} 处断言数值"
                                " → 已屏蔽为占位符（K11 注入点守卫）")
            prev_results = _succ + "\n" + (prev_results or "")
    # ★ 2026-10-17（用户要求「子目标划分逻辑/计算 + 计算类强制返回算式」）：
    #   按子目标类型生成**验证纪律**——计算类强制逐行可复算式子（工具③逐行复算、
    #   算错打回）；逻辑类要求可形式化陈述（Lean 检查）；derive/未标注 ⇒ 两者都要。
    #   ⚠ 类型是"引导"，不是"排他"：审核侧（collab review）对两类都跑工具。
    _sg_type = str(sg.get("type") or "").strip().lower()
    _calc_disc = (
        "【计算纪律】本步属**计算类**：\n"
        "  · 【计算】节必须**逐行**写出可机器复算的式子：`LHS = RHS` 或 `LHS >= RHS`；\n"
        "  · 只用 ASCII 数学语法：sqrt(...) / abs(...) / ^ 或 ** / * /；"
        "微积分写 integrate(expr, x) / diff(expr, x)，概率写 binomial(n, k)；\n"
        "  · **不得含中文、不得只给结论数值**——系统会逐行用 SymPy 复算，算错立即打回重解。")
    _logic_disc = (
        "【逻辑纪律】本步属**逻辑类**：\n"
        "  · 结论必须是**可形式化的明确数学陈述**（量词清楚：对任意 / 存在 / 充要）；\n"
        "  · 推理链每一步给出依据（引用的定理/条件），不得跳步到结论；\n"
        "  · 系统会用 Lean 检查陈述与类型是否病态。")
    if _sg_type in ("compute", "numeric", "calculation"):
        _type_discipline = _calc_disc
    elif _sg_type in ("prove", "logical", "verify", "proof"):
        _type_discipline = _logic_disc
    elif _sg_type in ("derive", "推导", "derivation"):
        _type_discipline = _calc_disc + "\n" + _logic_disc
    else:
        _type_discipline = (
            "【类型纪律】本步类型未标注：请**同时**给出可复算算式（计算证据）与"
            "可形式化结论陈述（逻辑证据）——审核会分别用计算工具与 Lean 检查。")
    user_msg = SUBGOAL_STEP_USER_TEMPLATE.format(
        problem=_prob_for_step + _obj_inj,
        type_discipline=_type_discipline,
        subgoal_plan_summary=plan_summary,
        previous_results=prev_results,
        lemma_context=lemma_context,
        subgoal_id=sg["id"],
        subgoal_title=_title,
        subgoal_type=sg["type"],
        subgoal_description=_desc,
        subgoal_expected_output=sg["expected_output"],
    )
    # v2.11：0-LLM 校验反馈（S1-lite）追加在提示词末尾；只有校验真没过才带
    # （空串 = 通过）。不追加在开头/中部，避免改变模板收尾结构（2026-08-29
    # 教训：提示词收尾被破坏会让模型进入续写模式、泄漏占位符）。
    if extra_hint:
        user_msg = user_msg + (
            f"\n\n[本步结果校验反馈] {extra_hint}\n"
            f"请修正后重新给出【本步结果】。"
        )
    # ★ 2026-10-15（审核 F1④ 修复）：题目目标锚注入（子目标步）——与 solver
    #   主路径同款：让每个子目标求解都看得到「题目要什么」（此前只进蓝图）。
    try:
        from .problem_goal import render_problem_goal as _rpg_sg
        _pg_txt = _rpg_sg(ctx)
        if _pg_txt:
            user_msg = user_msg + "\n\n" + _pg_txt
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-18：关键注意点（子目标步同款，不 record 防噪）
    try:
        from contracts.switch_registry import get_bool as _gb
        _on = _gb("attention_focus_enabled")
        _focus = list(getattr(ctx, "attention_focus", None) or [])
        if _on and _focus:
            user_msg = user_msg + (
                "\n\n【本题关键注意点（先读，本步对照）】\n"
                + "\n".join("- " + str(f)[:160] for f in _focus[:5]))
    except Exception:  # noqa: BLE001
        pass
    # ★ 2026-10-17：题面歧义提醒（子目标步同款；不 record 防噪）
    try:
        _amb = list(getattr(ctx, "problem_ambiguities", None) or [])
        if _amb:
            user_msg = user_msg + (
                "\n\n【题面歧义提醒（题意理解阶段发现，必须正面处理）】\n"
                + "\n".join("- " + str(a)[:220] for a in _amb[:4])
                + "\n⚠ 本步求解前先明确你采用哪种解读、依据是题面哪句话。")
    except Exception:  # noqa: BLE001
        pass
    return user_msg


def _use_lemma(self, ctx: TaskContext) -> bool:
    """按领域路由 lemma 累积（与 SolverAgent 同规则，2026-08-29）。

        A/B 实测 lemma 全开净 0.0pp、数论 +23pp → 数论域开、其他关。
        """
    if not getattr(self.config, 'use_lemma_accumulation', False):
        return False
    domains = list(getattr(self.config, 'lemma_domains', []) or [])
    if not domains:
        return True
    d = str(getattr(ctx, 'domain', '') or '')
    return any(k in d for k in domains)


def _accumulate_lemma(self, ctx: TaskContext, sg: dict, result: str) -> None:
    """把已求得的子目标结论存入 ctx.lemma_repo（去重，单题内存）。

        只收有效结果（剔除占位符/失败标记），按「标题: 结论」存储，
        后续子目标与最终求解步骤通过提示词注入复用。
        """
    if not result or result.startswith("[子目标"):
        return
    title = (sg.get("title") or "").strip()
    text = str(result).strip()
    # ★ 2026-10-06（结论契约）：有结构化结论时用结论入引理记忆 ——
    #   引理是"可直接引用、无需重新推导"的前置结论，混入验算正文只会
    #   撑大后续每步提示词。
    _concl = str(sg.get("conclusion") or "").strip()
    if _concl:
        text = _concl
    if not title or not text:
        return
    entry = f"{title}: {text}"
    if entry not in ctx.lemma_repo:
        ctx.lemma_repo.append(entry)


def _strip_tool_residue(text: str) -> str:
    """剥除工具调用 XML 残留（2026-09-17 新增）。

        为什么需要：实测 official112-025 的 6 个候选中 **5 个** `answer` 是字面量
        `</tool_call>`（reasoning 却是完整推理）——根因是几条「把模型原文尾部切片
        当答案/结果」的路径不过剥壳，工具标签原样落盘 ⇒ 该题必然判错。
        `utils.extract` 已有 `_strip_toolcall_markers`，此处统一封装，避免各调用点
        口径漂移（`<calc>` 标记剥壳由 `Candidate.__post_init__` 统一出口处理）。

        惰性 import + try/except：剥壳失败一律退回原文，绝不阻断求解。
        """
    if not text:
        return text
    try:
        from utils.extract import _strip_toolcall_markers
        return _strip_toolcall_markers(str(text))
    except Exception:  # noqa: BLE001  剥壳失败不阻断
        return text


def _call_step(self, ctx: TaskContext, user_msg: str,
                 sg: dict | None = None) -> str:
    """单步子目标求解调用（prefill「【本步结果】」答案前置，抑制 CoT）。

        2026-10-01：原 2026-09-04 下沉的 `<calc>` 计算纪律（system 追加引导 +
        响应回填）已随计算工具板块整体删除；本方法现在只做提示词拼装与调用。
        """
    from .sub_goal_solver import _conclusion_contract_on
    _step_system = SUBGOAL_STEP_SYSTEM
    # ★ 2026-09-29（截图 #3+#4）：子目标求解是**用户点名的解题重点环节**，
    # 且它是独立于 solver 的生成路径 —— 定理线索必须在本路径单独注入，
    # 否则"检索到了定理"对子目标推理毫无作用（本项目教训：只改一处=没改）。
    # 放 system 而非 user：user_msg 由上层模板构造，改动面更小、风险更低。
    if getattr(self.config, 'enable_theorem_hint', True):
        _th = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
        # ★ 2026-10-18：按**本子目标**检索的定理块优先（比全局块更相关）；失败回退全局
        try:
            from .theorem_hint import search_for_subgoal as _sfs
            _ps = _sfs(ctx, sg)
            if _ps:
                _th = _ps
        except Exception:  # noqa: BLE001
            pass
        if _th:
            _step_system = _step_system + "\n\n" + _th
    # ★★ 2026-10-06（子目标交付重写 A）：结论契约注入（开关
    #   `subgoal_conclusion_contract`，默认开）。放 system 而非 user：
    #   user_msg 的收尾结构有历史教训（2026-08-29：末尾追加会让模型进入
    #   续写模式、泄漏占位符），system 追加无此风险。
    if _conclusion_contract_on():
        _step_system = _step_system + "\n" + SUBGOAL_CONCLUSION_CONTRACT
    # ★ 2026-10-15（用户要求「蓝图为每一个子目标设定工具，比如证明就用 lean」）：
    #   按蓝图在节点上声明的 `tool` 追加**工具专属指令** —— 让本步产出该工具
    #   **能消费的形态**（lean 代码块 / 可判真伪的等式 / 纯数值等式），
    #   随后由 `sg_tool_exec` 把声明兑现成真正的机器核验。
    #   放 system（与结论契约同处）：user 尾部追加有历史风险（2026-08-29 续写模式）。
    #   `tool` 为空 / "llm" ⇒ 不追加任何指令（保持历史行为，零回归面）。
    if _node_contract_on() and sg:
        _tool = str(sg.get("tool") or "").strip().lower()
        # ★★ 2026-10-15（用户：「把 lean 内化到蓝图构建里面」——叶子级形式化下传）：
        #   `tool=lean` 且蓝图**已给出** `formal_sig` 时，改用「补证明体」指令
        #   （分工：蓝图出"证什么"、求解出"怎么证"）；没有签名才退回原指令
        #   （求解侧自行写整段 Lean）。两条指令互斥，避免模型收到矛盾要求。
        _sig = str(sg.get("formal_sig") or "").strip() if _tool == "lean" else ""
        if _sig:
            _step_system = _step_system + "\n" + \
                SUBGOAL_LEAN_SIG_TEMPLATE.format(formal_sig=_sig)
        else:
            _tool_ins = SUBGOAL_TOOL_INSTRUCTIONS.get(_tool, "")
            if _tool_ins:
                _step_system = _step_system + "\n" + _tool_ins

    # ★ 2026-09-30（截图 #8）：**多 agent 求解子目标**。
    #   用户原话：「多 agent 拿子目标的求解能不能换成多 agent？
    #              那样是不是能提高子目标的正确率？」
    #
    #   实现：对**同一个子目标**独立采 N 个解（同 prompt、不同随机性），
    #   再按「归一化后的关键值一致性」选代表解 —— 即 self-consistency
    #   下沉到子目标粒度。与整体求解层的投票同源，但作用点在中间步骤：
    #   **中间步骤错一个，后面全错**，故此处收益理论上更高。
    #
    #   开关：`enable_subgoal_multi_agent`（默认 False = 与改动前逐字一致）
    #         `subgoal_agents_n`（默认 3）
    #   ⚠ 只在**非时间紧迫**时启用：N 次调用 = N 倍单步成本，
    #     预算不足时必须退回单次，否则会把后续验证阶段饿死
    #     （本项目多次实测的教训：生成侧吃光预算 → 4_verify 全跳）。
    _ma_n = self._multi_agent_n(ctx)
    if _ma_n > 1:
        # 阶段二-1：传入 sg，让「保留全部候选」落到该子目标的 trace 项上
        #（返回值仍是代表解字符串 ⇒ 下游消费零改动）。
        resp = self._sample_step_consensus(ctx, _step_system, user_msg, _ma_n,
                                           sg=sg)
    else:
        resp = self._maybe_tool_llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _step_system},
                    {"role": "user", "content": user_msg},
                ],
                "【本步结果】",
            ),
            0.2, 32768,
        )
        if not str(resp or "").strip():
            # ★ 2026-10-22：空/纯空白响应必须在 **stitch 之前** 判掉 ——
            #   否则 `stitch` 会把 "   " 包成「【本步结果】   」，变成"有内容的空壳"。
            return "[子目标求解失败]"
        if resp:
            resp = stitch("【本步结果】", resp)
    if resp is None:
        return "[子目标求解失败]"
    # ★★ 2026-10-22（产出为空的两个缺陷）：
    #   ① 原实现只拦 `None`，**空串 `""` 会一路落成空 result** —— 实测 628 个
    #      子目标里 103 个 result 为空，进 merge 当有效输入。空响应属于调用失败，
    #      与 None 同处理。
    if not str(resp).strip():
        return "[子目标求解失败]"

    # 提取「本步结果」部分
    # 2026-09-17：**切片/返回前先剥壳**。实测 official112-025 的 6 个候选中
    # 5 个 `answer` 是字面量 `</tool_call>`（reasoning 却是完整推理）——根因
    # 就是这里把模型原文尾部（工具调用 XML 残留）原样当成"本步结果"返回，
    # 再经 merge 变成最终答案。剥壳后为空则返回空串，交给上层兜底。
    result_match = re.search(r"【本步结果】\s*\n?(.*?)(?:$|【)", resp, re.DOTALL)
    if result_match and result_match.group(1).strip():
        step_result = self._strip_tool_residue(result_match.group(1)).strip()
        # ★★ 2026-10-22（实测：产出被截残）：模型 prefill 后**不会重复**
        #   「【本步结果】」，而是直接写「本步结论: …」「本步依据: …」再写
        #   「【推导过程】…」。上面截到下一个 `【` 为止 ⇒ **推导整段被丢弃**。
        #   实测 official112-008：原始 1753 字 → 只留 40 字（仅结论行），
        #   而 merge 需要推导来合成最终答案 ⇒ 这同样是"产出不完整"。
        #   判定：结果段**很短**（<80 字，通常只有结论两行）且存在推导段时，
        #   把推导一并带上；若结果段本身已含实质内容则保持原样。
        if len(step_result) < 80 and "【推导过程】" in resp:
            _dev2 = re.search(r"【推导过程】\s*\n?(.*?)(?:$|【)", resp,
                              re.DOTALL)
            if _dev2 and _dev2.group(1).strip():
                step_result = (step_result + "\n【推导过程】\n"
                               + self._strip_tool_residue(
                                   _dev2.group(1)).strip())
    else:
        # ★ 2026-10-22：`【本步结果】` 后**紧跟**另一个标记（如「【推导过程】」）时，
        #   上面的非贪婪正则 `(.*?)(?:$|【)` 会匹配到空内容 —— 模型把实质内容
        #   写在了推导段里。此时退到推导段取正文，而不是返回空串。
        _dev = re.search(r"【推导过程】\s*\n?(.*?)(?:$|【)", resp, re.DOTALL)
        if _dev and _dev.group(1).strip():
            step_result = self._strip_tool_residue(_dev.group(1)).strip()
        else:
            # 没有标记，取最后 500 字符（同样先剥壳再切片）
            step_result = self._strip_tool_residue(resp).strip()[-500:]
    if not str(step_result).strip():
        # 兜底：剥壳/切片后仍空 ⇒ 明确失败标记，绝不留空串给下游
        return "[子目标求解失败]"
    # ★★ 2026-10-06（子目标交付重写 A/B/C）：把结论契约解析 + 客观判定
    #   落到 sg（trace 由 run() 落盘）。传 resp 供兜底找【推导过程】块。
    if sg is not None and _conclusion_contract_on():
        self._apply_conclusion_contract(sg, step_result, raw_text=resp)
    return step_result


def _multi_agent_n(self, ctx: TaskContext) -> int:
    """本子目标该采几个解（1 = 关闭多 agent）。

        ★ 三道闸门，任一不满足即退回 1（**宁少不饿死**）：
          ① 开关 `enable_subgoal_multi_agent`（默认 False ⇒ 行为与改动前一致）；
          ② 时间：`is_time_critical()` 或生成软截止已到 ⇒ 退回 1；
          ③ 预算：剩余调用数不足 N 时退回 1。
        这么设计的原因：多 agent = N 倍单步成本，而单题 1150s 里验证侧
        （3.6+4_verify+4.5+4.6+5+5.5+6+6.5）要吃掉 ~45%。生成侧一旦吃光，
        验证全跳、错解裸奔 —— 本项目已有多次实测教训。
        """
    if not bool(getattr(self.config, "enable_subgoal_multi_agent", False)):
        return 1
    # ⚠ 2026-09-30：**只兜 None / 缺键**，绝不用 `or 3`。
    #   原因：0 与负数在 Python 里是 falsy，`0 or 3` 会**静默变成 3** ——
    #   用户明明把并发数配成 0（意图可能是"先关掉"），系统却按 3 倍成本
    #   跑起来，且日志里看不出任何异常。这是"配置写错反被兜底成危险值"
    #   的典型假安全。此处宁可把它夹到 1（= 退回单次），语义明确可解释。
    _raw = getattr(self.config, "subgoal_agents_n", None)
    if _raw is None:
        _raw = 3
    try:
        n = int(_raw)
    except (TypeError, ValueError):
        n = 3
    n = max(1, min(n, 8))            # 上限 8，防配置写错把预算打爆
    if n <= 1:
        return 1
    try:
        if ctx.is_time_critical() or ctx.gen_time_up():
            return 1
    except Exception:  # noqa: BLE001
        pass
    try:
        # ⚠ 2026-09-30：`Budget.can_spend()` 已于 2026-09-03 删除
        #   （比赛无次数上限，Budget 只记账不阻断）。此处改用
        #   `used_calls` / `max_calls` 直接比较 —— 语义是"**自留余量**"，
        #   不是"系统强制拦截"：多 agent 是可选加速手段，不该把总调用数
        #   花在自己身上，导致后续阶段（验证/自改进）无预算可用。
        _b = getattr(ctx, "budget", None)
        if _b is not None:
            _left = int(getattr(_b, "max_calls", 0) or 0) - int(
                getattr(_b, "used_calls", 0) or 0)
            if _left < n:
                return 1
    except Exception:  # noqa: BLE001
        pass
    return n


def _sample_step_consensus(self, ctx: TaskContext, step_system: str,
                           user_msg: str, n: int,
                           sg: dict | None = None) -> Optional[str]:
    """对同一子目标采 n 个解，返回**最可代表多数**的那一个原文。

        ★ 2026-10-02（阶段二-1：「淘汰」→「保留」）：
          用户原话「那怎么能靠长度选取呢？要按照正确率选取啊」——
          本函数**不再淘汰候选**：把**全部候选原文**写入 `sg["candidates"]`
          （由上层落到 `ctx.subgoal_trace[i]["candidates"]`），供**最终答案层**
          一次性对比判断 / 投票。**返回值语义不变**（仍返回一个代表解字符串）
          ⇒ 下游消费方式**零改动**。
          ⚠ 代表解**不是"唯一真理"**，只是"继续往下推的默认值"；其余候选仍保留。

        选择规则（先到先用，逐级降级 —— 每一级都可解释）：
          ① **关键值一致**：n 个解里，归一化后的「本步结果」出现次数最多者胜
             （self-consistency）。并列时取**更长的原文**（推理更完整）。
          ② 全都不同：取**最长**的一个 —— 短的那个多半是"没做出来"的敷衍。
          ③ 全部为空/失败占位：返回第一个非 None（让上层走既有的失败分支，
             **不在这里伪造成功**）。

        ⚠ 刻意**串行**采样，不用线程：
          · 项目铁律「测试任务级互斥、串行优先」；
          · LLM 客户端未必线程安全；
          · 温度 0.2 下并发采样反而可能拿回近似同解（无多样性收益）。

        埋点：每次采样数量、是否达成多数、代表了多久 —— 供 A/B 归因。
        """
    samples: list = []
    for _i in range(n):
        try:
            _r = self._maybe_tool_llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": step_system},
                        {"role": "user", "content": user_msg},
                    ],
                    "【本步结果】",
                ),
                0.2, 32768,
            )
        except Exception as exc:  # noqa: BLE001  单次失败不放弃整组
            logger.debug("多agent子目标采样 %d/%d 异常: %s: %s",
                         _i + 1, n, type(exc).__name__, exc)
            _r = None
        if _r:
            _r = stitch("【本步结果】", _r)
        samples.append(_r)
        # ★ 2026-10-02 阶段二-2：**移除**「采样中途超时就不凑满 N」
        #   （原 `if ctx.is_time_critical(): break`）——那是"为省时间降质量"
        #   的残留；研究期时间无限、一切为正确率让路（用户口径）。

    picked, key, n_distinct = self._pick_consensus_sample(samples)

    # ★ 阶段二-1：保留全部候选（不再淘汰）。只留**非空原文**，保持采样顺序。
    #   `candidates_picked_index` = 代表解在保留列表中的下标（-1 = 无有效候选）。
    _kept: list = [str(s) for s in samples if s and str(s).strip()]
    _picked_idx = _kept.index(picked) if (picked in _kept) else -1

    # ★ 阶段二-2：子目标层**客观验证**（本轮核心增量）。
    #   `candidate_answers` = 各候选**抽取后的干净答案串**（客观验证的输入——
    #   原文带推理没法算）；客观验证判出"对的那个" ⇒ **用它续推**（"按正确率选"，
    #   而非按长度/字符串多数）；判不出 ⇒ 保留多个、标未定（verified=-1）沿用代表解。
    # ★ 2026-10-06（结论契约）：候选答案取**结构化结论**（值/等式表达式），
    #   而非整段「本步结论+依据+验算」原文 —— 否则客观判定拿到的仍是
    #   不可解析的中文正文（实测 candidate_answers 大量是 <check> 包的中文句
    #   ⇒ `n_judged = 0/73`）。
    _answers: list = [self._answer_from_raw(s) for s in _kept]
    _verified_idx = self._verify_candidates_objective(_answers)
    if 0 <= _verified_idx < len(_kept):
        picked = _kept[_verified_idx]
    if sg is not None:
        try:
            sg["candidates"] = _kept
            sg["candidates_picked_index"] = _picked_idx
            sg["candidate_answers"] = _answers
            sg["candidates_verified_index"] = _verified_idx
        except Exception:  # noqa: BLE001  个别 sg 实现可能禁写
            pass

    try:
        self.record(
            ctx, "subgoal_multi_agent",
            f"多agent子目标求解：采样 {len(samples)}/{n} 个，"
            f"不同结论 {n_distinct} 种，保留候选 {len(_kept)} 条"
            f"（原代表解第 {_picked_idx} 条），"
            f"客观验证{'选中第 ' + str(_verified_idx) + ' 条'
                    if _verified_idx >= 0 else '判不出（沿用代表解）'}，"
            f"{'达成多数一致' if n_distinct < len(samples) else '无多数（取最长）'}",
            n_samples=len(samples), n_distinct=n_distinct,
            n_kept=len(_kept), picked_index=_picked_idx,
            n_answers=len(_answers), verified_index=_verified_idx,
            consensus=(n_distinct < len(samples)))
    except Exception:  # noqa: BLE001
        pass
    return picked


def _pick_consensus_sample(samples: list) -> tuple:
    """从多个采样里选代表解。返回 ``(样本原文, 归一化关键值, 不同值个数)``。

        纯函数（不碰 ctx / 网络），便于单测锁定选择规则 ——
        这是本机制**唯一需要正确的地方**：选错了，多 agent 只会放大错误。
        """
    valid = [s for s in (samples or []) if s and str(s).strip()]
    if not valid:
        # 全部失败：把第一个（可能是 None）原样交回，让上层走失败分支
        return (samples[0] if samples else None), "", 0

    def _key(text: str) -> str:
        """归一化关键值：优先取 \\boxed{}，否则取归一化后的末行。"""
        t = str(text)
        try:
            m = re.search(r"\\boxed\{([^{}]*)\}", t)
            if m:
                return m.group(1).strip().lower()
        except Exception:  # noqa: BLE001
            pass
        # 去空白/标点后取末尾 60 字符 —— 子目标结果的关键值通常在结尾
        tail = re.sub(r"\s+", "", t)[-60:]
        return tail.lower()

    groups: dict = {}
    for s in valid:
        groups.setdefault(_key(s), []).append(s)
    n_distinct = len(groups)
    # 多数票：组内样本数最多 → 并列时取组内最长原文
    best_key, best = max(
        groups.items(), key=lambda kv: (len(kv[1]), max(len(x) for x in kv[1])))
    # 组内选最长（推理最完整）；并列时取先出现的，保证确定性
    picked = max(best, key=lambda x: len(str(x)))
    return picked, best_key, n_distinct


def _render_gap_block(ctx) -> str:
    """渲染「题意缺口」块（2026-10-13 接线 B）。

    来源：`ctx.subgoal_chain_gaps`（批次 3 的 inputs 校验）+ `ctx.understanding_gaps`（批次 1）。
    两类缺口的处置语义不同（用户 Q33 的 A/B 分流）：
      · **B 类**（题面已给、前序环节没读到）⇒ 要求"求解时必须先检查并纳入"；
      · **A 类**（题面确实没给）⇒ 要求"若解法需要，必须显式写出「依赖假设：…」，
        **不得静默补入**"（反 K11：不得凭空补条件冒充已知）。

    无缺口 ⇒ 返回空串（下游零成本，行为与接线前一致）。
    """
    gaps = list(getattr(ctx, "subgoal_chain_gaps", None) or [])
    gaps += list(getattr(ctx, "understanding_gaps", None) or [])
    b_items, a_items = [], []
    for g in gaps:
        if not isinstance(g, dict):
            continue
        detail = str(g.get("detail") or "").strip()
        if not detail:
            continue
        if str(g.get("kind") or "").upper() == "B":
            b_items.append(detail)
        else:
            a_items.append(detail)
    if not b_items and not a_items:
        return ""
    lines = ["【题意缺口清单（来自前置理解与蓝图 inputs 校验）】"]
    if b_items:
        lines.append("以下条件**题面已经给出、但前序环节没有用上** —— "
                     "求解前必须逐一检查并纳入：")
        lines.extend("  · " + d[:120] for d in b_items[:6])
    if a_items:
        lines.append("以下内容**题面确实没有提供** —— 若你的解法需要它们，必须显式写出"
                     "「依赖假设：…」并说明依据，**不得静默补入**：")
        lines.extend("  · " + d[:120] for d in a_items[:6])
    return "\n".join(lines)
