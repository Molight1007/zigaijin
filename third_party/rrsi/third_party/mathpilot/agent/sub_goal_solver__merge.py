# -*- coding: utf-8 -*-
"""子目标求解器的 merge 职责簇
（2026-10-08 批3：从 agent/sub_goal_solver.py 逐字搬出 7 个方法）。

搬出后为模块级函数（实例方法保留 `self` 首参，方法体逐字不变）；
跨模块引用原文件模块级符号者（含 `SubGoalSolverAgent` 与其常量/函数）
在**函数体内惰性 import**（避免循环导入）—— 下表已显式列出。
原文件类内保留 1 行跳板（`_x = _sgs_merge._x`）⇒ `self._x(...)` /
`SubGoalSolverAgent._x(...)` 契约逐字不变（同 verifier__review.py 批6）。
"""
from __future__ import annotations

import logging
import os
import re
import time

from .base import TaskContext
from .subgoal_result import read_step


from prompts.sub_goal import (
    SUBGOAL_CROSSCHECK_SYSTEM,
    SUBGOAL_CROSSCHECK_USER_TEMPLATE,
    SUBGOAL_MERGE_SYSTEM,
    SUBGOAL_MERGE_USER_TEMPLATE,
)
from utils.extract import extract_final_answer, smart_fallback_answer
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot")


def _lookback_case_mark(text: str, pos: int) -> bool:
    """LHS 前 2 个非空白字符是否含情形/条件标记（当/若/情形…）。"""
    from .sub_goal_solver import SubGoalSolverAgent
    i = pos - 1
    n = 0
    while i >= 0 and n < 2:
        ch = text[i]
        if not ch.isspace():
            if ch in SubGoalSolverAgent._CASE_MARK:
                return True
            n += 1
        i -= 1
    return False


def _collect_claims(self, result: str) -> list:
    """从子目标结果汇总确定性等式断言 [(L, R)]（去重）。

        四来源并集：①<calc> 回填落地行/行内 `[计算] X = v`
        （允许自然语言前缀）；②<check>…=…</check>
        协议块（L3）；③L2 提取式等式断言（_extract_lean_assert_pairs，
        宁缺毋滥）；④松散 `X = 常数` 文本片段（"解得 x = 1"），带情形/
        条件标记（当/若/情形…）的前缀跳过，防分支讨论误报。合并去重。
        """
    pairs: list = []
    if not result:
        return pairs
    text = str(result)
    for m in self._CALC_INLINE_RE.finditer(text):
        L = m.group(1).strip().rstrip("= ")
        R = m.group(2).strip()
        if L and R:
            pairs.append((L, R))
    for L, R in self._extract_check_asserts(text):
        pairs.append((L, R))
    for L, R, *_rest in self._extract_lean_assert_pairs(text):
        pairs.append((L, R))
    # 松散源只在**不含** [计算]/<check> 标记的行上跑（避免与专用源重复，
    # 且函数调用形 RHS（sqrt(…)）会在左括号处被截断成假断言）。
    for ln in text.split("\n"):
        if "[计算]" in ln or "<check>" in ln:
            continue
        for m in self._LOOSE_EQ_RE.finditer(ln):
            if self._lookback_case_mark(ln, m.start()):
                continue
            L, R = m.group(1).strip(), m.group(2).strip()
            if L and R:
                pairs.append((L, R))
    seen, uniq = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _rhs_const(s: str):
    """RHS 解析为常数则返回 sympy 值对象，否则 None（变量式/不可解析）。

        符号化字符串（`^`/`**` 幂、分数、根式）交给 sympy；带自由符号或
        解析失败 → None（不做常量比较，宁漏勿误报）。
        """
    try:
        from utils.sympy_tools import _try_parse
    except Exception:  # noqa: BLE001
        return None
    try:
        a, _ = _try_parse(s.replace("^", "**"))
    except Exception:  # noqa: BLE001
        return None
    if a is None:
        return None
    try:
        if getattr(a, "free_symbols", None):
            return None
    except Exception:  # noqa: BLE001
        return None
    return a


def _cross_conflicts(self, subgoals: list[dict],
                     results_map: dict[int, str]) -> list:
    """确定性冲突闸（0 LLM）：同名单变量被给出不同常数 → 返回矛盾清单。

        判同值用 sympy 精确差化简（0.5 与 1/2、3*sqrt(5) 与 sqrt(45) 视为
        同值不报警）；同值不同写法不误报。返回人类可读行（空=无冲突）。
        """
    try:
        import sympy as _sp
    except Exception:  # noqa: BLE001
        return []
    claims: list = []
    for sg in subgoals:
        sid = sg.get("id")
        text = read_step(sg, results_map)   # 唯一读入口（2026-10-20）
        for L, R in self._collect_claims(text):
            claims.append((sid, L, R))
    if not claims:
        return []
    var_map: dict = {}          # LHS规范化名 -> [(sid, 值对象, R原文)]
    for sid, L, R in claims:
        Lc = re.sub(r"\s+", "", str(L))
        if not self._LHS_IDENT_RE.match(Lc):
            continue            # 复合 LHS（x^2 / x+1 / S(1)…）不做常量冲突判定
        v = self._rhs_const(str(R))
        if v is None:
            continue
        var_map.setdefault(Lc, []).append((sid, v, str(R).strip()))
    conflicts: list = []
    for Lc, lst in var_map.items():
        groups: list = []       # 值分组（sympy 差化简判同）
        for sid, v, Rs in lst:
            for g in groups:
                try:
                    if _sp.simplify(g[0][1] - v) == 0:
                        g.append((sid, v, Rs))
                        break
                except Exception:  # noqa: BLE001  化简失败按不同值处理（宁报）
                    pass
            else:
                groups.append([(sid, v, Rs)])
        if len(groups) < 2:
            continue
        # 只报**跨子目标**冲突：同一步内对同一量出现多值多为分支枚举
        # （"x=1 或 x=2"）或自我修正，不构成本机制目标（merge 自带矛盾
        # 检查兜底）；跨子目标对同一量的不同常数才是真正流毒信号。
        for g in groups:
            g.sort(key=lambda t: (str(t[0]), len(t[2])))
        base = min(groups, key=lambda g: (str(g[0][0]), len(g[0][2])))
        base_sid, _b_v, b_r0 = base[0]
        other = None
        for g in groups:
            if g is base:
                continue
            cand = [t for t in g if t[0] != base_sid]
            if cand:
                other = cand[0]
                break
        if other is None:
            continue            # 所有不同取值都出自同一子目标 → 不报
        o_sid, _o_v, o_r = other
        conflicts.append(
            f"「{Lc}」在子目标 #{base_sid} 中取 {b_r0}，"
            f"而在子目标 #{o_sid} 中取 {o_r}——同一变量不能同时成立，"
            "必有至少一方错误（含推导/抄写错误），需裁决修正")
    return conflicts[:6]          # 单次最多报 6 条，防提示词膨胀


def _llm_crosscheck(self, ctx: TaskContext, subgoals: list[dict],
                    results_map: dict[int, str],
                    plan_summary: str) -> str:
    """LLM 交叉核对轮：merge 前 +1 次小调用，返回矛盾裁决报告。

        只裁决矛盾（不求解/不合并/不给答案）；调用失败或预算不足返回空串
        （空串=未跑/无输出，由调用方与"未发现矛盾"区分）。~1-2K token。
        """
    if ctx.gen_time_up():
        return ""
    concl_lines = []
    for sg in subgoals:
        sid = sg.get("id")
        text = read_step(sg, results_map)   # 唯一读入口（2026-10-20）
        text = re.sub(r"\s+", " ", str(text)).strip()
        if not text:
            continue
        title = re.sub(r"\s+", " ", str(sg.get("title") or "")).strip()
        concl_lines.append(f"# 子目标 {sid}"
                           + (f"「{title}」" if title else "")
                           + f"：{text[:200]}")
    if not concl_lines:
        return ""
    user_msg = SUBGOAL_CROSSCHECK_USER_TEMPLATE.format(
        problem=(str(getattr(ctx, "problem", "") or "")[:800]),
        subgoal_plan_summary=(str(plan_summary or "")[:600]),
        subgoal_conclusions="\n".join(concl_lines),
    )
    try:
        t0 = time.time()
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system",
                     "content": SUBGOAL_CROSSCHECK_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "【矛盾列表】",
            ),
            # ★ 2026-10-02：2048→8192（DeepSeek 适配方案 A，统一上限）。
            0.0, 8192,
        )
        if resp:
            resp = stitch("【矛盾列表】", resp)
        self.record(ctx, "subgoal_crosscheck",
                    f"LLM 交叉核对轮耗时 {time.time() - t0:.0f}s"
                    + ("" if resp else "，返回空（不注入矛盾提示）"))
        return resp or ""
    except Exception as exc:  # noqa: BLE001  失败不阻断 merge
        logger.debug("LLM 交叉核对轮异常（跳过）: %s: %s",
                     type(exc).__name__, exc)
        return ""


def _merge_results(self, ctx: TaskContext, subgoals: list[dict],
                   plan_summary: str, results_map: dict[int, str],
                   merge_strategy: str) -> str:
    """调用 LLM 合并所有子目标结果。

        2026-10-01：原 2026-09-04 下沉的 `<calc>` 合并纪律（引导 + 回填 + 裸算
        打回）已随计算工具板块整体删除。
        2026-09-08 改进建议3+6：①蓝图最终结论注入 user 模板（merge 输出
        必须与蓝图结论对齐，防 alg-009/053/068 输出端数值漂移）；②merge 产出
        空转/占位/平凡 calc 回填（如 [计算] 0 = 0）时带反馈重试一次（alg-009
        merge 计划 = "[计算]0=0" 仍照常入候选的教训）。
        2026-09-08 老师建议3 交叉核对：merge 前先跑子目标交叉核对
        （A 确定性冲突闸 0-LLM + B LLM 交叉核对轮 +1 调用，各自独立开关），
        发现的矛盾强制注入 merge 提示词要求逐条裁决——不让子目标间矛盾
        静默流入最终合并（alg-060 型显式/隐含矛盾）。
        """
    # ★★ 2026-10-23（用户：「根据错误设置检测环节判断新版本有没有问题」）：
    #   子目标链在合并前用**第二模型复算**一遍。实测 official112-008：sg5 把
    #   `∑1/(…) ≥ (N-1)²/(N(N+1)-3) → 1` 推成 `D ≤ 777²/2`（应为 777²/1，凭空除 2），
    #   sg8 又高估构造下确界。这两处一旦进入 merge 就直接固化成最终答案，
    #   后面 4.1/4.2 只能"正确地全杀"，4.6 重写两轮（79 分钟/轮）也无济于事。
    #   自检结果既记 diag/错误库，也**注入合并提示词**（见下方）；
    #   但不强制照抄——要求合并方"以自己的重新核算为准"。
    _sus: list = []
    try:
        import json as _json
        from .subgoal_selfcheck import (check_subgoal_chain, chain_from_trace,
                                        summarize)
        _sus = check_subgoal_chain(
            chain_from_trace(getattr(ctx, "subgoal_trace", None) or []),
            str(getattr(ctx, "problem", "") or ""))
        if _sus:
            _md = getattr(ctx, "metadata", None)
            if isinstance(_md, dict):
                _md["subgoal_suspects"] = _sus
            self.record(ctx, "subgoal_selfcheck", summarize(_sus))
            # 注：不再写错误库 —— 2026-10-23 用户定，经验库改为「给人看的
            # 可迁移教训（现象/原因/应对）」，不承载单次运行的追踪记录。
    except Exception:  # noqa: BLE001  自检是增益，任何失败都不影响合并
        pass

    user_msg, _merge_system = self._prepare_merge_prompt(
        ctx, subgoals, plan_summary, results_map, merge_strategy)

    # ★★ 2026-10-23（用户 A 案：发现了就要用）：把复核结论**注入合并提示词**。
    #   只说"哪一步可疑 + 复核认为应为多少"，并要求**以自己重新核算为准**、
    #   不一致时说明理由 —— 不强制它照抄，避免把一个模型的错当权威。
    # ★★ 2026-10-23（调研 AlphaProof / Nexus 后新增）：子目标级 Lean 断言验证。
    #   与上面的「第二模型复算」互补：复算看"算得对不对"，Lean 看"这句断言是否成立"。
    #   默认关（five_gate_subgoal_lean=False）；失败/环境缺失一律跳过，绝不阻断合并。
    _lean_res = {}
    try:
        from contracts.switch_registry import get_bool as _swb
        if _swb("five_gate_subgoal_lean"):
            from .subgoal_lean import verify_chain, summarize as _sl_sum
            from .subgoal_selfcheck import chain_from_trace as _cft
            _cfg = getattr(self, "config", None)
            _wd = (os.environ.get("LEAN_PROJECT_PATH", "")
                   or getattr(_cfg, "lean_project_dir", "") or "")
            _le = (os.environ.get("LEAN_EXECUTABLE", "")
                   or getattr(_cfg, "lean_executable", "") or "")
            if _wd:
                _lean_res = verify_chain(
                    _cft(getattr(ctx, "subgoal_trace", None) or []),
                    str(getattr(ctx, "problem", "") or ""),
                    client=getattr(self, "client", None),
                    work_dir=_wd, lean_exe=_le)
                if _lean_res.get("n_picked"):
                    _md = getattr(ctx, "metadata", None)
                    if isinstance(_md, dict):
                        _md["subgoal_lean"] = _lean_res
                    self.record(ctx, "subgoal_lean", _sl_sum(_lean_res))
    except Exception as _e_sl:  # noqa: BLE001  增益机制，任何失败都不影响合并
        self.record(ctx, "subgoal_lean", "子目标 Lean 验证异常（跳过）: %s"
                    % type(_e_sl).__name__)

    if _sus:
        _hint = ["", "",
                 "【独立复核提示（第二模型重算，非最终结论）】",
                 "下列子目标结论经独立复算可疑，合并时**请重新核算，勿直接采用**："]
        for _s in _sus[:4]:
            _hint.append("- 子目标 %s（%s）：%s"
                         % (_s.get("id") or "?", _s.get("kind") or "?",
                            str(_s.get("step") or "")[:200]))
            if _s.get("expected"):
                _hint.append("  复核认为应为：%s" % str(_s["expected"])[:160])
            if _s.get("got"):
                _hint.append("  链里给出的：%s" % str(_s["got"])[:160])
        _hint.append("请以你**重新核算**的结果为准；若与复核意见不一致，"
                     "在最终答案的推导里写明理由。")
        user_msg = user_msg + "\n".join(_hint)

    # Lean 编译不通过 = 该断言不成立，比"复核认为"更硬（编译器判定）
    _lean_bad = [x for x in (_lean_res.get("items") or [])
                 if x.get("ok") is False]
    if _lean_bad:
        _lh = ["", "", "【Lean 编译结果（机器判定，优先级最高）】",
               "下列子目标断言**转成 Lean 后编译不通过** ⇒ 该断言不成立，"
               "合并时不得采用："]
        for x in _lean_bad[:4]:
            _lh.append("- sg%s：%s" % (x.get("id"), str(x.get("statement"))[:160]))
            _lh.append("  编译错误：%s" % str(x.get("error"))[:160])
        user_msg = user_msg + "\n".join(_lh)

    for attempt in range(2):
        # v2.4.1：prefill「【最终答案】」答案前置，抑制 CoT
        resp = self.llm(
            ctx,
            prefill_messages(
                [
                    {"role": "system", "content": _merge_system},
                    {"role": "user", "content": user_msg},
                ],
                "【最终答案】",
            ),
            0.2, 32768,
        )
        if resp:
            resp = stitch("【最终答案】", resp)
        if resp is None:
            if attempt == 0:
                self.record(ctx, "subgoal_merge",
                            "合并 LLM 调用失败，带反馈重试一次")
                user_msg = user_msg + (
                    "\n\n[上一轮合并未返回有效文本] 请完整输出"
                    "【矛盾检查】【结论合并】【最终答案】。")
                continue
            return self._fallback_from_last_subgoal(subgoals, results_map)

        # 优先提取「最终答案」
        answer = extract_final_answer(resp)
        if answer:
            # 2026-09-08：merge 空转（占位词 / [计算] X = X 平凡回填）→
            # 不允许带着空转结论入候选，带反馈重试一次
            if self._looks_placeholder(answer):
                if attempt == 0:
                    self.record(
                        ctx, "subgoal_merge",
                        f"合并输出为空转/占位（{answer[:60]}），带反馈重试")
                    user_msg = user_msg + (
                        f"\n\n[上一轮合并输出为空转：{answer[:120]}]\n"
                        "你没有真正合并各子目标结果。请：①逐条做矛盾检查；"
                        "②基于各子目标结果与蓝图结论真实推导；"
                        "③基于各子目标结果与蓝图结论真实推导，"
                        "在【最终答案】给出结论。")
                    continue
                return self._fallback_from_last_subgoal(subgoals, results_map)
            return answer
        if attempt == 0:
            self.record(ctx, "subgoal_merge",
                        "合并未提取到【最终答案】，带反馈重试一次")
            user_msg = user_msg + (
                "\n\n[上一轮未找到【最终答案】段] 请在末尾用【最终答案】"
                "明确给出结论（一个数值/表达式/集合），不要只给过程。")
            continue
        return smart_fallback_answer(resp) or self._fallback_from_last_subgoal(subgoals, results_map)
    return self._fallback_from_last_subgoal(subgoals, results_map)


def _prepare_merge_prompt(self, ctx: TaskContext, subgoals: list[dict],
                           plan_summary: str, results_map: dict[int, str],
                           merge_strategy: str) -> tuple:
    """合并提示词准备段（2026-10-08 批1：从 `_merge_results` 原地拆出）。
        装配 all_results / blueprint_conclusion(K11掩码) / 客观题注入 / user_msg /
        交叉核对 conflict_note / 可信度标记 / _merge_system；返回二元组。
        段体**逐字搬**，record 步序不变。
        """
    from .sub_goal_solver import _k11_mask_on, _mask_asserted_values, _strip_answer_forms
    all_results = self._format_all_results(results_map, subgoals)
    # 蓝图最终结论（root 节点 statement）：取 ctx.blueprint，兼容 dict/list 两种形态
    # 2026-09-17（P-20）：同上，蓝图根节点结论里若含 \boxed /「最终答案」先剥离
    blueprint_conclusion = _strip_answer_forms(
        self._blueprint_conclusion(ctx))
    # ★ 2026-10-04（K11 注入点守卫）：P-20 只剥「答案形态标记」，自然语句里的
    #   **断言数值**仍会原样注入 merge 提示词（见 prompts/sub_goal.py:286-287）
    #   ⇒ 再掩码一层（不删句，只换被断言的数）。这是 blueprint_conclusion 的
    #   唯一注入点，故在此收口。
    if _k11_mask_on():
        blueprint_conclusion, _n_bc = _mask_asserted_values(blueprint_conclusion)
        if _n_bc:
            self.record(ctx, "blueprint_lock_guard",
                        f"蓝图根结论内嵌 {_n_bc} 处断言数值 → 已屏蔽为占位符"
                        "（K11 merge 注入点守卫）")
    # 2026-09-12 客观题特化（三处同源注入之③：merge 合并）——merge 直接产出
    # 最终答案署名候选，特化纪律必须在这里也在场，否则"检测到了"却"没照做"。
    _obj_inj_m = ""
    try:
        from .question_type import objective_injection as _oi_m
        _obj_inj_m = _oi_m(
            ctx.problem or "", getattr(ctx, "question_type", "") or "",
            bool(getattr(self.config, "objective_tactic_enabled", True)))
    except Exception:  # noqa: BLE001
        _obj_inj_m = ""
    # 2026-09-13 B0：答案形态要求必须在 merge 阶段也在场（merge 直接产出最终答案）
    try:
        from .question_type import answer_form_requirement as _afr_m
        _obj_inj_m += _afr_m(
            ctx.problem or "", getattr(ctx, "question_type", "") or "")
    except Exception:  # noqa: BLE001
        pass
    user_msg = SUBGOAL_MERGE_USER_TEMPLATE.format(
        problem=(ctx.problem or "") + _obj_inj_m,
        subgoal_plan_summary=plan_summary,
        blueprint_conclusion=blueprint_conclusion,
        all_results=all_results,
        merge_strategy=merge_strategy or "将各子目标结果按逻辑顺序组合，得出原题的最终答案。",
    )
    # ③ 子目标交叉核对（2026-09-08 老师建议3）：合并前先核对，矛盾强制
    # 注入提示词要求逐条裁决（A 0-LLM 确定性冲突闸 + B LLM 交叉核对轮）。
    # 二者独立开关可单开/同开 A/B；任何一步失败/预算不足都不阻断 merge。
    conflict_note = ""
    try:
        enable_gate = bool(getattr(
            self.config, "subgoal_conflict_gate", False))
        enable_xllm = bool(getattr(
            self.config, "subgoal_crosscheck_llm", False))
        if enable_gate or enable_xllm:
            parts = []
            if enable_gate:
                cfl = self._cross_conflicts(subgoals, results_map)
                if cfl:
                    parts.append(
                        "【确定性冲突闸】系统自动核对各子目标对同一变量"
                        "的取值，发现以下显式矛盾（同名 LHS 出现不同常数"
                        "值，数学上不可能同时成立）：\n"
                        + "\n".join("  - " + c for c in cfl))
                    self.record(
                        ctx, "subgoal_crosscheck",
                        f"确定性冲突闸: 发现 {len(cfl)} 处同名不同值矛盾")
                else:
                    self.record(
                        ctx, "subgoal_crosscheck",
                        "确定性冲突闸: 未发现显式矛盾")
            if enable_xllm:
                rep = self._llm_crosscheck(
                    ctx, subgoals, results_map, plan_summary)
                if rep and rep.strip():
                    # 剥掉"未发现矛盾"类措辞后仍有实质内容 → 视为报了
                    # 矛盾（防"除上述外未发现矛盾"混合措辞漏注入）
                    _resid = re.sub(
                        r"未发现(?:明显|任何)?矛盾|无矛盾|没有矛盾",
                        "", rep, flags=re.IGNORECASE)
                    if len(_resid.strip()) <= 40:
                        self.record(
                            ctx, "subgoal_crosscheck",
                            "LLM 交叉核对轮: 未发现矛盾")
                    else:
                        parts.append(
                            "【LLM 交叉核对】独立评审轮对各子目标结论的"
                            "矛盾裁决报告：\n" + rep.strip()[:800])
                        self.record(
                            ctx, "subgoal_crosscheck",
                            f"LLM 交叉核对轮: 报告潜在矛盾"
                            f"（{rep.strip()[:80]!r}）")
            if parts:
                conflict_note = (
                    "\n\n【⚠ 系统交叉核对结果（最高优先级，必须先处理）】"
                    "合并前已发现子目标结论间存在潜在矛盾。你必须在"
                    "【矛盾检查】中**逐条裁决**：指出冲突双方各自的主张、"
                    "依据原题条件与推导复核判定哪一方错误、给出"
                    "修正后的取值；若实为不同对象/不同分支并不冲突，请"
                    "逐条说明理由。**禁止不处理矛盾就直接输出【最终答案】"
                    "。**\n" + "\n".join(parts))
    except Exception as exc:  # noqa: BLE001  交叉核对失败不阻断 merge
        logger.debug("子目标交叉核对异常（跳过）: %s: %s",
                     type(exc).__name__, exc)
    if conflict_note:
        user_msg = user_msg + conflict_note
    # ★★ 2026-10-06（子目标交付重写 C）：把"仅断言、无推导"与"客观判定不成立"
    #   的子目标**显式交给合并层裁决**。实测题 official112-007：子目标直接给
    #   结论、无推导，且其中一条是臆造（真值 0 写成 1），而合并层此前**看不到
    #   "哪条没有推导支撑"** —— 只能照抄。现按 trace 打标后注入，要求逐条复核。
    try:
        _weak = []
        for _t in (getattr(ctx, "subgoal_trace", None) or []):
            if not isinstance(_t, dict):
                continue
            _v = str(_t.get("objective_verdict", "") or "")
            _hd = bool(_t.get("has_derivation", False))
            if _v == "incorrect":
                _weak.append(
                    f"  · 子目标 #{_t.get('id')}「{_t.get('title', '')}」"
                    f"—— **客观判定不成立**（{_t.get('objective_reason', '')}），"
                    "必须纠正后再合并")
            elif _v != "correct" and not _hd:
                # 未被客观判定为成立、且无推导支撑 ⇒ 仅断言
                _weak.append(
                    f"  · 子目标 #{_t.get('id')}「{_t.get('title', '')}」"
                    "—— **仅断言、无推导支撑**（未给出验算），须独立复核")
        if _weak:
            user_msg = user_msg + (
                "\n\n【⚠ 结论可信度标记（合并前必须逐条处理）】\n"
                + "\n".join(_weak[:12])
                + "\n对上述子目标：**不得**直接采用其结论；须用代入 / 另一条路径"
                "重算 / 与题意对照等方式复核，判定其可信后再合并；"
                "若复核发现错误，以复核结果为准。")
            self.record(ctx, "subgoal_merge",
                        f"合并前注入可信度标记 {len(_weak)} 条"
                        "（无推导支撑/客观判定不成立）")
    except Exception:  # noqa: BLE001  标记注入失败不阻断 merge
        pass
    _merge_system = SUBGOAL_MERGE_SYSTEM
    # ★ 2026-09-29（截图 #8）：合并阶段是"把子目标答案总结成最终答案"的出口，
    # 是定理线索真正能落到**最终答案**上的最后机会，故同样注入。
    if getattr(self.config, 'enable_theorem_hint', True):
        _th = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
        if _th:
            _merge_system = _merge_system + "\n\n" + _th
    return user_msg, _merge_system
