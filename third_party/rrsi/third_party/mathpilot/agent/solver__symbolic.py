# -*- coding: utf-8 -*-
"""符号化"本地计算接管答案"通道（2026-10-08 拆分：从 agent/solver.py 抽出）。

含三条**独立于主链**的答案覆盖/旁证通道（均默认关，靠 config 开关控制）：
  · ``_maybe_expression_eval``     —— 表达式范式：模型只建模，本地精确求值取答案
  · ``_maybe_symbolic_crosscheck`` —— L2 独立符号建模复核（旁证 / 定向重问）
  · ``_maybe_symbolic_solve``      —— 符号化方程求解：模型只交方程，SymPy 回代求值

搬迁方式：
  · ``_maybe_expression_eval`` / ``_maybe_symbolic_crosscheck`` —— **纯逐字搬**
    （函数体与签名逐字搬入，仅去结构性缩进，多行 docstring 续行保留原缩进）。
  · ``_maybe_symbolic_solve`` —— **逐字搬 + 原地拆**：原 163 行 > 硬线 150，
    把尾部"工具求解成功后决定最终答案"整段（原 887–949，64 行）逐字搬到
    ``_symbolic_solve_apply``，主函数保留"守卫 → 建模 → 工具求解"，
    尾部以一行 ``return _symbolic_solve_apply(...)`` 委托（该段每个出口本就返回
    ``(resp, answer)`` 元组，故拆出后返回契约不变）。

``agent/solver.py`` 原处三个方法改为一行委托。
外部符号依赖（原在 solver.py 模块级）：开关 ``_sw_bool``、prompts ``SYMBOLIC_*``、
``parse_symbolic_payload``/``evaluate_payload``、``strip_given_numbers``/
``parse_symbolic_solve``/``validate_payload``/``solve_with_tool``、
``to_exact_number``、``extract_final_answer`` —— 本模块顶部同口径导入。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用，避免运行期导入
    from .base import TaskContext

from agent.switch_registry import get_bool as _sw_bool
from prompts.symbolic_model import (
    SYMBOLIC_MODEL_SYSTEM, SYMBOLIC_MODEL_USER, SYMBOLIC_JSON_SEED,
    SYMBOLIC_SOLVE_SYSTEM, SYMBOLIC_SOLVE_USER, SYMBOLIC_SOLVE_SEED,
    SYMBOLIC_SOLVE_RETRY_SYSTEM, SYMBOLIC_SOLVE_RETRY_USER,
    SYMBOLIC_FEEDBACK_SYSTEM, SYMBOLIC_FEEDBACK_USER,
)

try:
    from .symbolic_model import parse_symbolic_payload, evaluate_payload
except ImportError:                    # 模块不可用时降级（绝不阻断）
    parse_symbolic_payload = None
    evaluate_payload = None

try:
    from .symbolic_solve import (
        strip_given_numbers, parse_symbolic_solve, validate_payload,
        solve_with_tool)
except ImportError:                    # 模块不可用时降级（绝不阻断）
    strip_given_numbers = None
    parse_symbolic_solve = None
    validate_payload = None
    solve_with_tool = None

try:
    from utils.math_eval import to_exact_number
except ImportError:                    # 模块不可用时降级（绝不阻断）
    to_exact_number = None

from utils.extract import extract_final_answer

logger = logging.getLogger("MathPilot")


def _maybe_expression_eval(self, ctx: TaskContext, resp: str,
                           answer: str) -> tuple[str, str]:
    """表达式范式（2026-09-12，用户要求「默认生成方程式、不心算」）。

        解析模型输出的两行（【变量赋值】/【最终表达式】）：
            【变量赋值】x=5, y=3
            【最终表达式】2*x + y
        把赋值代入表达式 → **用本地精确计算内核（utils.math_eval）求值** → **答案取工具值**
        （模型原先写的数值被丢弃，仅在 reasoning 里留痕）。

        设计要点：模型**只负责建模**（设符号、给出算式），数值代入与运算
        全部由本地完成 —— 与"剥离"通道互补：剥离适用于题面有具体数据的题，
        本通道适用于参数是变量的题（B 类）。
        无这两行 / 求值失败 / 开关关闭 → 原样返回（零风险）。
        """
    try:
        if not getattr(self.config, "symbolic_solve_adopt", True):
            return resp, answer
        # 2026-09-12 逻辑堆叠治理③：客观题（选择/判断）的答案是选项字母或
        # 判断值，不是数值 —— 本关"工具代入求值"根本不适用。若模型按
        # 误写了【最终表达式】，会把选项答案改写成数值而丢分。
        _qt = getattr(ctx, "question_type", "") or ""
        # 2026-09-12 补漏：填空题同属客观题 —— 多空答案形如 `1, 2`，
        # 被本关覆盖成单个工具值会丢分（选择/判断上一轮已设防，填空漏了）。
        if _qt in ("选择题", "判断题", "填空题"):
            return resp, answer
        if not resp or "【最终表达式】" not in resp:
            return resp, answer
        import re as _re3
        m_e = _re3.search(r"【最终表达式】\s*(.+?)(?:\n|$)", resp)
        if not m_e:
            return resp, answer
        expr = m_e.group(1).strip().strip("$").strip().rstrip("。.")
        if not expr or len(expr) > 300:
            return resp, answer
        assigns: dict = {}
        m_v = _re3.search(r"【变量赋值】\s*(.+?)(?:\n|$)", resp)
        if m_v:
            for part in _re3.split(r"[,，;；、]", m_v.group(1)):
                if "=" in part:
                    _k, _v = part.split("=", 1)
                    _k = _k.strip().strip("$").strip()
                    _v = _v.strip().strip("$").strip().rstrip("。.")
                    if _k and _v:
                        assigns[_k] = _v
        # ============================================================
        # 2026-09-13 修复（q3_mcp 实测实锤）：符号答案不得被数值代入覆盖。
        # 实况（official112 #001）：gold `2-2m`，模型原答 `\boxed{-2(m-1)}`
        # —— 二者恒等，**答案是对的**；但响应里同时写了自造的
        # 【变量赋值】m=3 与【最终表达式】-2*(3-1)，本关遂以工具值 -4
        # **覆盖**模型答案 → 判 expr_wrong。根因：参数题的答案含自由变量，
        # 对变量代一个具体值只是"特例"，根本不是题目要的答案。
        # 判据（零后悔）：模型自己的答案若**不是纯数值**（即符号式/含自由变量）
        # → 代入求值不可能产出答案 → 弃权，保留模型原答案，交下游既有把关。
        # 开关 EXPR_EVAL_GROUNDING_GUARD=0 可恢复旧行为（A/B 对照用）。
        # ============================================================
        if (_sw_bool("expr_eval_grounding_guard")
                and answer and to_exact_number is not None
                and to_exact_number(str(answer)) is None):
            self.record(
                ctx, "expression_eval_skip",
                f"表达式范式：模型答案为符号式（{str(answer)[:24]}），"
                f"代入求值不适用，弃权保留模型原答案")
            return resp, answer
        # 符号 → 数值 代入（整词匹配，避免 x 误伤 exp）
        sub = expr
        for _k, _v in assigns.items():
            sub = _re3.sub(
                r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % _re3.escape(_k),
                "(%s)" % _v, sub)
        try:
            from utils.math_eval import safe_eval as _safe_eval_fn
        except ImportError:  # 模块不可用时降级（绝不阻断）
            return resp, answer
        raw = str(_safe_eval_fn(sub) or "")
        # 2026-09-12 定型前审核修复：**只采纳纯数值结果**。
        # 原实现用 findall 抓"最后一个数字"，当工具结果不是纯数值时会抓
        # 到碎片 —— `3*sqrt(5)`（精确根式）→ 取到 "5"、`2*x+3`（含变量）
        # → 取到 "3"、`WARN: …12…` → 取到 "12"；而本函数会把该值**直接
        # 写进【最终答案】**，等于把答案覆盖成一个错数。必须整串是数值才
        # 采纳，否则零后悔弃权（保留模型原答案）。
        _val_txt = raw.strip()
        for _pre in ("≈", "~", "="):
            _val_txt = _val_txt.lstrip(_pre).strip()
        if to_exact_number is None or to_exact_number(_val_txt) is None:
            self.record(ctx, "expression_eval_skip",
                        f"表达式范式：{expr[:60]} 工具结果非纯数值"
                        f"（{raw[:40]}），弃权保留模型原答案")
            return resp, answer
        val = _val_txt
        # 2026-09-12 逻辑堆叠治理（定型前审核）：本关已用本地计算器产出答案，
        # 登记标记让下游**不再重复处理**同一次计算：
        #   · _maybe_symbolic_solve 不会再重复建模/求解一遍并覆盖该答案。
        # 两者都是"答案取工具值"的同族机制，叠加只会白烧 LLM 时间与互相覆盖。
        if ctx is not None:
            try:
                setattr(ctx, "_expr_eval_adopted", True)
            except Exception:  # noqa: BLE001  标记失败不影响主流程
                pass
        self.record(ctx, "expression_eval",
                    f"表达式范式：{expr[:60]} 代入 {assigns} → {val}"
                    f"（答案取工具值，模型不参与计算；原答 {str(answer)[:24]}）")
        # 2026-09-12 逻辑堆叠治理②：resp 里若**已有**最终答案区块，必须
        # **替换**而不是在末尾追加 —— 否则会留下两个互相矛盾的【最终答案】
        # 标记，而下游 extract_final_answer 的正则取的是**第一个**（模型的
        # 心算旧值），本关刚采纳的工具值会被完全架空（已实测该正则行为：
        # `re.search(r"【最终答案】\s*\n?\s*([\s\S]+)")` 命中首个标记，
        # 且 first_line 分支直接返回其首行）；reasoning 注入下游时也会歧义。
        _new_resp = resp
        if "【最终答案】" in _new_resp:
            _new_resp = _re3.sub(r"【最终答案】[\s\S]*$",
                                 "【最终答案】" + str(val), _new_resp)
        else:
            _new_resp = _new_resp + "\n\n【最终答案】" + str(val)
        return _new_resp, val
    except Exception as exc:  # noqa: BLE001
        logger.debug("[solver] expression_eval 失败，保留原输出: %s",
                     str(exc)[:120])
        return resp, answer

def _maybe_symbolic_crosscheck(self, ctx: TaskContext, resp: str,
                               answer: str) -> tuple[str, str]:
    try:
        if not getattr(self.config, "symbolic_crosscheck_enabled", False):
            return resp, answer
        if (parse_symbolic_payload is None or evaluate_payload is None
                or to_exact_number is None):
            return resp, answer
        if getattr(ctx, "_sym_cross_checked", False):
            return resp, answer
        want = to_exact_number(answer)
        if want is None:                    # 非纯数值答案 → 本关卡不适用
            return resp, answer
        problem = (getattr(ctx, "problem", "") or "").strip()
        if not problem:
            return resp, answer
        from .question_type import classify_question_type
        if classify_question_type(problem) == "证明题":
            return resp, answer             # 证明题无数值答案，不适用
        if ctx is not None and ctx.gen_time_up():
            self.record(ctx, "symbolic_crosscheck", "跳过：生成侧时间到")
            return resp, answer
        setattr(ctx, "_sym_cross_checked", True)   # 每题只做一次
        raw = self._compressed_solve(
            ctx, SYMBOLIC_MODEL_SYSTEM,
            SYMBOLIC_MODEL_USER.format(problem=problem[:3000]),
            temperature=0.0,
            # 2026-10-02 DeepSeek 适配：兜底值 512 ⇒ 8192，与 config 默认同步。
            max_tokens=int(getattr(self.config, "symbolic_max_tokens", 8192)),
            prefill_seed=SYMBOLIC_JSON_SEED,
        )
        payload = parse_symbolic_payload(raw or "")
        value, why = (evaluate_payload(payload) if payload
                      else (None, "未解析出建模结果"))
        if value is None:
            self.record(ctx, "symbolic_crosscheck", f"跳过：{why}")
            return resp, answer
        got = to_exact_number(value)
        if got == want:
            self.record(ctx, "symbolic_crosscheck_pass",
                        f"独立建模复核一致：{why}；与答案 {str(answer)[:30]} 相符")
            return resp, answer
        self.record(ctx, "symbolic_crosscheck_mismatch",
                    f"独立建模得 {value}（{why}），与答案 "
                    f"{str(answer)[:30]} 不符，定向重问")
        if ctx is not None and ctx.gen_time_up():
            return resp, answer
        system = (
            "你是数学解题助手。有人**只依据题目条件**独立建模，把给定数值"
            f"抽象为变量后算出目标量应为 {value}，与你的最终答案不一致。"
            "请核对：若你的答案错了，请修正推导并给出新答案；"
            "若你认为独立建模有误，请指出其建模错在哪里，并维持你的答案。"
            "请核对后重写解答，并在末尾用【最终答案】给出结论。"
        )
        user = (f"题目：\n{problem[:2000]}\n\n"
                f"你的解答片段：\n{(resp[-1200:] if len(resp) > 1200 else resp)}\n\n"
                f"独立建模给出的目标量值为：{value}（依据：{why}）\n"
                "请核对后重写解答并给出【最终答案】。")
        new_raw = self._compressed_solve(
            ctx, system, user, temperature=0.0,
            max_tokens=int(getattr(self.config, 'max_answer_tokens', 4096)),
        )
        if not new_raw or len(new_raw.strip()) < 10:
            return resp, answer
        new_ans = extract_final_answer(new_raw)
        if to_exact_number(new_ans) != got:
            self.record(ctx, "symbolic_crosscheck_keep",
                        "重问结果未落到独立建模真值 → 保留原答案")
            return resp, answer
        self.record(ctx, "symbolic_crosscheck_fix",
                    f"答案经独立建模修正：{str(answer)[:30]} → {value}")
        return new_raw, new_ans
    except Exception as exc:  # noqa: BLE001  失败保留原输出
        logger.debug("[solver] symbolic crosscheck 失败，保留原输出: %s",
                     str(exc)[:120])
        return resp, answer

def _maybe_symbolic_solve(self, ctx: TaskContext, resp: str,
                          answer: str) -> tuple[str, str]:
    try:
        if not getattr(self.config, "symbolic_solve_enabled", False):
            return resp, answer
        if (strip_given_numbers is None or parse_symbolic_solve is None
                or validate_payload is None or solve_with_tool is None
                or to_exact_number is None):
            return resp, answer
        if getattr(ctx, "_sym_solve_done", False):
            return resp, answer
        # 2026-09-12 逻辑堆叠治理（定型前审核）：答案已由表达式范式（本地
        # 计算器）产出 —— 不再重复做一次符号建模 + 求解，避免同一答案被
        # 两条同族机制先后覆盖（谁生效取决于顺序，且白烧一次建模调用）。
        if getattr(ctx, "_expr_eval_adopted", False):
            self.record(ctx, "symbolic_solve_skip",
                        "答案已由表达式范式（本地计算器）产出，跳过重复建模求解")
            return resp, answer
        problem = (getattr(ctx, "problem", "") or "").strip()
        if not problem:
            return resp, answer
        # 只对"数值型答案"题目生效（证明题/文字结论不适用）
        want = to_exact_number(answer)
        if want is None:
            return resp, answer
        from .question_type import classify_question_type
        if classify_question_type(problem) == "证明题":
            return resp, answer
        if ctx.gen_time_up():
            self.record(ctx, "symbolic_solve", "跳过：生成侧时间到")
            return resp, answer
        setattr(ctx, "_sym_solve_done", True)   # 每题只做一次

        # ① 数值剥离（本地，零 LLM）：题面 → 符号题面 + 本地参数表
        strip = strip_given_numbers(problem)
        if strip is None:
            self.record(ctx, "symbolic_solve_skip",
                        "题面无「显式给定参数」可剥离，本通道不适用")
            return resp, answer

        # ② 符号建模（1 次短调用；不合格带**具体原因**重试 1 次）
        # 2026-10-02 DeepSeek 适配：兜底值 384 ⇒ 8192，与 config 默认同步。
        max_tok = int(getattr(self.config, "symbolic_solve_max_tokens", 8192))
        payload, reason, previous = None, "", ""
        for attempt in range(2):
            if attempt == 0:
                sys_p = SYMBOLIC_SOLVE_SYSTEM
                usr_p = SYMBOLIC_SOLVE_USER.format(
                    problem=strip.problem[:3000])
            else:
                sys_p = SYMBOLIC_SOLVE_RETRY_SYSTEM
                usr_p = SYMBOLIC_SOLVE_RETRY_USER.format(
                    problem=strip.problem[:3000],
                    previous=(previous or "")[:600], reason=reason)
            if ctx.gen_time_up():
                self.record(ctx, "symbolic_solve", "跳过：建模前生成侧时间到")
                return resp, answer
            raw = self._compressed_solve(
                ctx, sys_p, usr_p, temperature=0.0, max_tokens=max_tok,
                prefill_seed=SYMBOLIC_SOLVE_SEED,
            )
            previous = raw or ""
            payload = parse_symbolic_solve(previous)
            ok, reason = validate_payload(payload, strip.params)
            if ok:
                break
            self.record(ctx, "symbolic_solve_validate_fail",
                        f"第 {attempt + 1} 次建模不合格：{reason}")
            if payload is None:
                break
        if payload is None:
            return resp, answer
        ok, reason = validate_payload(payload, strip.params)
        if not ok:
            self.record(ctx, "symbolic_solve_reject",
                        f"建模两次不合格，弃权：{reason}")
            return resp, answer

        # ③/④ 工具求解（本地，零 LLM）：数值回代 → 解方程（组）→ 精确值
        value, why = solve_with_tool(payload, strip.params)
        if value is None:
            self.record(ctx, "symbolic_solve_reject",
                        f"工具求解未成功：{why}")
            return resp, answer
        got = to_exact_number(value)
        if got is None:
            self.record(ctx, "symbolic_solve_reject",
                        f"工具结果非精确数值，弃权：{value}")
            return resp, answer
        self.record(ctx, "symbolic_solve_solved",
                    f"符号建模+工具求解成功：{why}")

        return _symbolic_solve_apply(
            self, ctx, resp, answer, payload, value, got, want, problem)
    except Exception as exc:  # noqa: BLE001  失败保留原输出
        logger.debug("[solver] symbolic solve 失败，保留原输出: %s",
                     str(exc)[:120])
        return resp, answer

def _symbolic_solve_apply(self, ctx: TaskContext, resp: str, answer: str,
                          payload: dict, value, got, want,
                          problem: str) -> tuple[str, str]:
    # 2026-10-08 拆分：从 _maybe_symbolic_solve 尾部逐字搬出（保持 ≤150 行硬线）。
    # 语义：模型已交方程、本地工具已求出精确值 value；此处决定最终 (resp, answer)。
    # **工具求解成功 → 答案直接取工具值**，模型不参与计算。
    # 原设计只在"一致"时放行、"分歧"时才回传 → 最终答案仍可能是模型
    # 心算出来的值（工具只是事后核对，没有"接管"计算）。
    # 现改为：只要本地 SymPy 求出精确值，该值**就是**最终答案，
    # 模型原先给的数值被丢弃（保留在 reasoning 里供追溯）。
    # 回退：`--symbolic_solve_adopt false`（恢复"仅比对/回传"旧行为）。
    if getattr(self.config, "symbolic_solve_adopt", True):
        _tgt = str(payload.get("target") or "")
        _note = (f"\n\n【工具精确计算】{_tgt} = {value}"
                 "（该数值由本地计算器代入求解得出，非模型心算）")
        new_raw = ((resp or "") + _note) if (resp or "") else _note.strip()
        self.record(ctx, "symbolic_solve_adopt",
                    f"答案改用工具精确计算值 {value}"
                    f"（原答案 {str(answer)[:30]}；模型不参与计算）")
        return new_raw, value

    if got == want:
        self.record(ctx, "symbolic_solve_pass",
                    f"工具值与答案一致（{value}），模型未参与计算")
        return resp, answer

    # 分歧 → 1 次短回传（可关：symbolic_solve_feedback=0）
    if not getattr(self.config, "symbolic_solve_feedback", True):
        self.record(ctx, "symbolic_solve_mismatch",
                    f"工具值 {value} 与答案 {str(answer)[:30]} 不一致"
                    "（未回传）")
        return resp, answer
    if ctx.gen_time_up():
        self.record(ctx, "symbolic_solve_mismatch",
                    f"工具值 {value} 与答案不一致，但生成侧时间到，"
                    "保留原答案")
        return resp, answer
    eqs_txt = "\n".join(payload.get("equations") or []) or "（以 TARGET 为准）"
    new_raw = self._compressed_solve(
        ctx, SYMBOLIC_FEEDBACK_SYSTEM,
        SYMBOLIC_FEEDBACK_USER.format(
            problem=problem[:1500], equations=eqs_txt,
            target=payload.get("target") or "", value=value),
        temperature=0.0,
        max_tokens=int(getattr(self.config, "max_answer_tokens", 4096)),
    )
    if not new_raw or len(new_raw.strip()) < 10:
        self.record(ctx, "symbolic_solve_mismatch",
                    f"回传结果为空，保留原答案（工具值 {value}）")
        return resp, answer
    new_ans = extract_final_answer(new_raw)
    # 三重保险（防"盲从工具值"造成掉分）：
    #   ① 模型必须**显式确认建模正确**（ADOPT）；REJECT/无标记一律维持原答案
    #   ② 新答案的数值必须**正好落在工具值上**（模型不得改动数值）
    upper = new_raw.upper()
    if "ADOPT" not in upper:
        self.record(ctx, "symbolic_solve_keep",
                    f"模型未确认建模正确"
                    f"（{'REJECT' if 'REJECT' in upper else '无标记'}）"
                    f" → 保留原答案（工具值 {value}）")
        return resp, answer
    if to_exact_number(new_ans) != got:
        self.record(ctx, "symbolic_solve_keep",
                    f"回传后答案未落到工具值 {value} → 保留原答案")
        return resp, answer
    self.record(ctx, "symbolic_solve_fix",
                f"答案按工具结果修正：{str(answer)[:30]} → {value}")
    return new_raw, new_ans
