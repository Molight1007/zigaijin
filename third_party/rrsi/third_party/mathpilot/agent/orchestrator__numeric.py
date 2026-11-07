# -*- coding: utf-8 -*-
"""数值化 / 方案攻击职责簇（2026-10-08 批10：从 agent/orchestrator.py 逐字搬出）。

搬出方法：_final_answer_postprocess / _maybe_numericize / _latex_const_value /
_lean_crosscheck_number / _objective_selfcheck / _answer_form_gate /
_ask_for_form / _looks_like_answer / _value_attack_blueprint

**纯搬迁**：方法体逐字搬入（tokenize-aware 去缩进，多行 docstring 字符串值不变），
签名原样保留（@staticmethod 的两个方法在类内委托 stub 上保留装饰器）；Orchestrator
内保留**同名一行委托** ⇒ 调用点零改（含 orchestrator__solve.py / orchestrator__final_gate.py 的外部调用）。

耦合处理：
  · 类级常量 `_ASKS_ALL_RE` / `_ASKS_RANGE_RE` / `_COND_ANS_RE` **留在类里** ——
    委托传 `self` ⇒ 体内 `self.<常量>` 经实例解析到类属性，**方法体一行不改**；
  · 模块级 `_re` / `os` / `_sw_bool` / `logger`：新模块以同名（同源）重建，方法体不改；
  · `_looks_like_answer` 引用全局 `Orchestrator` ⇒ 函数内**加 1 行惰性 import**
    （本仓既定模式，单一来源仍在 orchestrator，非复制）。
"""
from __future__ import annotations

import logging
import os
import re as _re

from agent.switch_registry import get_bool as _sw_bool

logger = logging.getLogger("MathPilot")


def _final_answer_postprocess(self, ctx, ans: str = "") -> str:
    """最终答案后处理总入口（2026-09-11）：P2 强制数值化 + P5 客观题自检。

        设计原则（用户 9/11 指示：一次性实施全部优化 + 逐项埋点，避免"改一点测一点"）：
          - 每项优化独立开关（环境变量，便于 A/B 与逐项审查）
          - 每项优化独立埋点（record → diag，单次评测即可看出各项是否触发、影响哪些题）
          - 全部保守：任何异常或不确定情形一律原样返回，绝不降低既有正确率
        参数 ans 为空时回退读 ctx.final_response（供提前返回路径复用）。
        """
    ans = ans or ctx.final_response or ""
    if not ans.strip():
        return ans
    # 2026-09-12 审核补可观测（**不改行为**）：主出口调用点位于 **6.5 最终答案
    # 闸门之后**，而下面的 P2 数值化会**改写答案内容** ⇒ 闸门校验的字符串
    # 可能并非最终提交的字符串（"闸门验 A、实际提交 B"）。此处记录改动事实，
    # 便于用评测数据判断是否需要把后处理**前移到闸门之前**（那属主链顺序
    # 调整，定型前不动）。
    _before_pp = ans
    if _sw_bool("numericize_final"):
        try:
            ans = self._maybe_numericize(ctx, ans)
        except Exception as _e:  # noqa: BLE001
            logger.debug("[P2] 数值化异常跳过: %s", _e)
    if _sw_bool("objective_selfcheck"):
        try:
            ans = self._objective_selfcheck(ctx, ans)
        except Exception as _e:  # noqa: BLE001
            logger.debug("[P5] 客观题自检异常跳过: %s", _e)
    # B0（2026-09-13）：答案形态闸门——条件式→具体值 / 求所有→枚举
    if _sw_bool("answer_form_gate"):
        try:
            ans = self._answer_form_gate(ctx, ans)
        except Exception as _e:  # noqa: BLE001
            logger.debug("[B0] 形态闸门异常跳过: %s", _e)
    if ans != _before_pp:
        try:
            self.record(ctx, "final_postprocess_change",
                        "后处理改写了最终答案（闸门校验串≠提交串）："
                        f"{str(_before_pp)[:60]} → {str(ans)[:60]}")
        except Exception:  # noqa: BLE001
            pass
    return ans


@staticmethod
def _latex_const_value(core: str):
    """LaTeX 常数表达式 → Fraction（精确值）；不确定则 None。

        支持：数字、+ - * / **、括号、`\\frac{}{}`、`\\cdot`、`\\times`、
              `!`（阶乘，作用于整数）、`\\sqrt{}`。
        为什么不用 `utils.math_eval.to_exact_number`：实测它对本链路的典型失分格式
        （`2023^2`、`2^{19}\\cdot 19!`、`3!`）一律返回 None，等于 P2 无效。
        安全：仅接受白名单字符（数字/运算符/括号/空格/F），异常一律 None。
        """
    import re as _re
    from fractions import Fraction
    s = (core or "").strip()
    if not s or len(s) > 160:
        return None
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = _re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = _re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"((\1)**(1/2))", s)
    s = s.replace("{", "(").replace("}", ")")
    s = _re.sub(r"(\d+)\s*!", r"F(\1)", s)      # 阶乘：3! → F(3)
    s = s.replace("^", "**")
    s = s.replace("$", "").strip()
    if not _re.fullmatch(r"[0-9+\-*/().\sF]+", s) or "F" not in s and "*" not in s and "/" not in s and "(" not in s:
        # 纯数字/算式仍需含运算特征，否则交给原逻辑
        if not _re.search(r"[*/(^]|\*\*", s):
            return None
    try:
        from sympy import sympify, factorial as _fact, Rational
        v = sympify(s, locals={"F": _fact})
        if v is None or not getattr(v, "is_rational", False):
            return None
        r = Rational(v)
        return Fraction(int(r.p), int(r.q))
    except Exception:
        return None


def _maybe_numericize(self, ctx, ans: str) -> str:
    """P2：把"未求值的常数表达式"数值化（official112 有 4+ 道此类失分）。

        例：`2^{19}\\cdot 19!`、`2023^2`、`2 \\times 777^2`。
        严格保守：仅当 ① 答案含幂/阶乘/分式等表达式特征；② 可被
        `_latex_const_value` 精确求值；③ 结果与原写法不同 —— 才替换。
        """
    import re as _re
    m = _re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", ans)
    core = (m.group(1) if m else ans).strip()
    if not core or len(core) > 160:
        return ans
    # 仅处理"表达式型"答案（纯数字/纯文字不动）
    if not _re.search(r"[\^!]|\\cdot|\\times|\\frac|\\sqrt", core):
        return ans
    val = self._latex_const_value(core)
    if val is None:
        return ans
    try:
        new = (str(val.numerator) if val.denominator == 1
               else f"\\frac{{{val.numerator}}}{{{val.denominator}}}")
    except Exception:
        return ans
    if not new or new == core:
        return ans
    self.record(ctx, "numericize",
                f"P2 未求值表达式已数值化：{core[:60]} → {new[:40]}")
    # B1（2026-09-11）：Lean 侧独立交叉校验（与 SymPy 互为第二意见）
    self._lean_crosscheck_number(ctx, core, new)
    return ans.replace(core, new) if core in ans else f"\\boxed{{{new}}}"


def _objective_selfcheck(self, ctx, ans: str) -> str:
    """P5：客观题专项自检（official112 客观题段 2/11 且零验证兜底）。

        低风险动作：① 判断题写法归一（√/×/对/错 → 正确/错误）；
        ② 选项合法性**仅记录不改动**（题面未列出的选项字母 → 记录告警，避免误改）。
        """
    import re as _re
    q = ctx.problem or ""
    m = _re.search(r"\\boxed\{([^{}]*)\}", ans)
    core = (m.group(1) if m else ans).strip()
    # ① 判断题归一
    _jmap = {"√": "正确", "×": "错误", "对": "正确", "错": "错误",
             "T": "正确", "F": "错误", "true": "正确", "false": "错误"}
    _norm = _jmap.get(core)
    if _norm and _norm != core and (
            ("判断" in q) or _re.search(r"^\s*(正确|错误)\s*$", core)
            or core in ("√", "×", "T", "F", "true", "false")):
        self.record(ctx, "objective_check", f"P5 判断题归一：{core} → {_norm}")
        return f"\\boxed{{{_norm}}}"
    # ② 选项合法性（只诊断）
    # 2026-09-13 修复（实测 103 暴露）：原正则要求选项字母**前为空白/行首/
    # 括号**，而连排选项（`长期趋势B.季节变动C.循环变动`）中字母前是汉字
    # ⇒ 实际只匹配到 A，报出假告警「答案 ABCD 含题面未列选项 BCD（题面选项 A）」。
    # 改为复用 `extract_options`（与题型判定同源的「有序字母序列」定位），
    # 避免两套选项解析器长期漂移——本项目"同源注入"教训的又一次重现。
    # 当前该检测**只诊断不改动**，故此前无实际危害；但不修则 P5 统计失真，
    # 且一旦升级为实际校验会误改答案。
    _opts: set = set()
    try:
        from .question_type import extract_options as _extract_opts
        _opts = {lab for lab, _ in _extract_opts(q)}
    except Exception:  # noqa: BLE001
        _opts = set()
    if _opts and _re.fullmatch(r"[A-E]+", core or ""):
        _bad = [c for c in core if c not in _opts]
        if _bad:
            self.record(ctx, "objective_check",
                        f"P5 选项越界告警：答案 {core} 含题面未列选项 {''.join(_bad)}"
                        f"（题面选项 {''.join(sorted(_opts))}）")
    return ans


def _answer_form_gate(self, ctx, ans: str) -> str:
    """B0：答案形态闸门（条件式 → 具体值；求所有 → 枚举）。

        台账 B0 明确两个可救模式（本题库实测）：
          ① **条件式代替具体值**：015 `n ≥ 2`→`2`；064 `m is even`→`4`
          ② **求所有却只给一支**：003 只给 `2026`、漏 `2030`
        检测到形态不符 → **定向重问一次**（明确要求枚举/具体值）；
        时间不足或重问失败则原样返回（绝不降低既有答案质量）。

        设计要点：**只看硬限**（`is_timed_out` / 剩余 < 120s），
        不使用 `gen_time_up()` —— formatter 处于流程末段，该判据恒为 True
        （差分检测曾因此永久失效，见 formatter._objective_diff_probe 的教训）。
        """
    try:
        if not _sw_bool("answer_form_gate"):
            return ans
        _q = ctx.problem or ""
        _m = _re.search(r"\\boxed\{([^{}]*)\}", ans or "")
        _core = ((_m.group(1) if _m else ans) or "").strip()
        if not _core or len(_core) > 200:
            return ans
        _asks_all = bool(self._ASKS_ALL_RE.search(_q))
        _asks_range = bool(self._ASKS_RANGE_RE.search(_q))
        _cond = bool(self._COND_ANS_RE.search(_core))
        _is_enum = bool(_re.search(r"[,\uFF0C;；、]\s*\S", _core)) or \
            bool(_re.match(r"^\s*[\{\[]", _core))
        _trigger, _why = False, ""
        if _asks_all and not _is_enum:
            _trigger, _why = True, "题面要求『所有』但答案非枚举列表"
        elif _cond and not _asks_range:
            _trigger, _why = True, "答案为条件式，但题面未要求范围/条件"
        if not _trigger:
            return ans
        if ctx.is_timed_out() or ctx.time_remaining() < 120:
            self.record(ctx, "answer_form",
                        "B0 形态闸门：{}（时间不足一次重问，跳过）".format(_why))
            return ans
        self.record(ctx, "answer_form",
                    "B0 形态闸门触发：{} → 定向重问".format(_why))
        _fixed = self._ask_for_form(ctx, _core, _asks_all)
        if _fixed and _fixed != _core:
            # ⚠ 采纳前必须校验 —— 003 实测重问返回了**指令复述**
            # （`The user wants me to act as a math answer formatting tool.`），
            # 无校验直接采纳会把好答案改坏。不像答案 → 保留原答案。
            if not self._looks_like_answer(_fixed):
                self.record(ctx, "answer_form",
                            "B0 重问结果不像答案，弃用并保留原答案：{}"
                            .format(_fixed[:60]))
                return ans
            self.record(ctx, "answer_form",
                        "B0 重问修正：{} → {}".format(_core[:50], _fixed[:50]))
            return _fixed if _m is None else "\\boxed{{{}}}".format(_fixed)
        self.record(ctx, "answer_form", "B0 重问未获更优答案，原样返回")
        return ans
    except Exception as _e:  # noqa: BLE001
        logger.debug("[B0] 形态闸门异常跳过: %s", _e)
        return ans


@staticmethod
def _looks_like_answer(txt: str) -> bool:
    """重问结果是否"像答案"——防元话语 / 题面复述污染（003、005 实测教训）。

        判据：① 非空且不过长 ② 不含元话语 ③ **不含题面复述/疑问式措辞**
              ④ 含"答案核"（数字 / 数学宏 / 结构化集合）
        不满足则调用方**保留原答案**（绝不因重问而变差）。
        """
    from .orchestrator import Orchestrator  # noqa: F401 （惰性；单一来源仍在 orchestrator）
    if not txt or len(txt) > 120:
        return False
    if Orchestrator._META_ANS_RE.search(txt):
        return False
    # 2026-09-14（#005）：题面复述一律不是答案 —— 即便它里面含数字。
    if Orchestrator._RESTATE_ANS_RE.search(txt):
        return False
    if _re.search(r"\d", txt):
        return True
    return bool(_re.search(r"\\[a-zA-Z]+|π|√|∞|⌈|⌉|⌊|⌋|≤|≥", txt))


def _ask_for_form(self, ctx, cur: str, asks_all: bool) -> str:
    """B0 定向重问：按题目要求给出枚举列表 / 具体值（一次短调用）。"""
    try:
        from .base import _normalize_chat_response
        _rule = (
            "本题要求『求所有』——必须列出**全部**取值，用逗号分隔"
            "（如 `2026, 2030`），并确认已穷尽所有可能分支。"
            if asks_all else
            "本题要求**具体数值**——不要输出 `≥/≤/>/< / 任意 / for all / "
            "is even` 这类条件式或性质描述；若答案确为集合，请写成具体元素列表。"
        )
        _sys = ("你是数学答题规范器。根据题目要求给出最终答案，"
                "只输出答案本身（一行），不要任何解释或推导。")
        _user = "题目：\n{}\n\n当前答案：{}\n\n要求：{}".format(
            ctx.problem, cur, _rule)
        _resp = self.client.chat(
            messages=[{"role": "system", "content": _sys},
                      {"role": "user", "content": _user}],
            # 2026-10-02 DeepSeek 适配：原 512 ⇒ 8192。原值基于「答案规范化只需
            # 一行 + 提示词抑制输出」的 Intern-S 时代假设，对 reasoning 模型必然
            # 截断（reasoning 先吃满预算、正文为空）。
            temperature=0.0, max_tokens=8192,
        )
        _text = (_normalize_chat_response(_resp) or "").strip()
        if not _text:
            return ""
        return _text.split("\n")[0].strip()[:200]
    except Exception:  # noqa: BLE001
        return ""


def _lean_crosscheck_number(self, ctx, latex_expr: str, value: str) -> "bool | None":
    """B1（2026-09-11）：用 lean-lsp-mcp 独立复算，与 SymPy 结果交叉验证。

        仅验证"数值等式"：``example : ((<expr>) : ℚ) = <value> := by norm_num``。
        - ★ 2026-10-15（用户定案「删降级/保底」）：三态返回 ——
          True=校验通过 / False=Lean 认为不成立 / None=**未校验**（开关关、
          环境缺失、执行异常、输入不适用）。原实现「异常→return True」把
          "没校验"伪装成"通过"（掩盖），现全部改为 None + 记录未校验原因。
        - 返回 False 表示"Lean 认为该等式不成立" → 由调用方记录（可作硬信号）。
        价值：Lean 的 ℕ/ℤ/ℚ 语义精确，可捕捉 SymPy 在语义/精度上的偏差。
        """
    if not _sw_bool("lean_xcheck_numeric"):
        return None                      # 开关关 = 未校验（≠ 通过）
    try:
        from tools.lean_local.lean_bridge import mcp_run_code
    except Exception as _e_imp:  # noqa: BLE001
        self.record(ctx, "numericize",
                    "P2 Lean 交叉校验未执行（导入失败）: %s: %s"
                    % (type(_e_imp).__name__, str(_e_imp)[:120]))
        return None
    _cfg = getattr(self, "config", None)   # 测试以 __new__ 构造时无 config
    _wd = str(getattr(_cfg, "lean_project_dir", "") or "")
    if not _wd or not os.path.isdir(_wd):
        self.record(ctx, "numericize",
                    "P2 Lean 交叉校验未执行（lean_project_dir 无效）")
        return None
    le = (latex_expr or "").replace("\\cdot", "*").replace("\\times", "*")
    le = _re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", le)
    le = le.replace("{", "(").replace("}", ")").strip()
    if not le or len(le) > 120:
        return None                      # 输入不适用 = 未校验
    code = (f"import Mathlib.Tactic\n"
            f"example : (({le}) : ℚ) = ({value}) := by norm_num\n")
    try:
        res = mcp_run_code(code, _wd, timeout=90.0)
    except Exception as _e_run:  # noqa: BLE001
        self.record(ctx, "numericize",
                    "P2 Lean 交叉校验未执行（执行异常）: %s: %s"
                    % (type(_e_run).__name__, str(_e_run)[:120]))
        return None
    if res.get("ok"):
        self.record(ctx, "numericize",
                    f"P2 Lean 交叉校验通过：{le} = {value}")
        return True
    self.record(ctx, "numericize",
                f"P2 Lean 交叉校验不一致：{le} ≠ {value}"
                f"（{str(res.get('items'))[:140]}）")
    return False


def _value_attack_blueprint(self, ctx, tier: str) -> None:
    """蓝图 merge 声称"极值=数值"时，数值采样攻击验证。

        009 实况：蓝图声称最大 4∛(85/98)≈3.815，真值 2∛(196/13)≈4.94——
        LLM 心算错值后全链路自洽执行，Lean unknown 拦不住。这里用
        **确定性 Python 采样**（20k 点）找突破声称值的可行点 → 证伪。
        """
    # 审核补充（2026-09-03）：原实现所有早退分支都是裸 return，
    # v17 全程 0 条 value_attack 痕迹时无法判断"未触发"还是"异常被吞"。
    # 每个早退点都留 debug 日志，便于下一轮定位。
    try:
        if ctx.is_time_critical():
            logger.debug("[C-lite] 跳过数值攻击：时间紧张")
            self.record(ctx, "value_attack",
                        "跳过：时间紧张（is_time_critical）")   # Z2 2026-09-17
            return
        # 审核修正（2026-09-03）：原读 `ctx.blueprint_merge`——该属性**不存在**
        # （TaskContext 无此字段，diag 用的是 ctx.blueprint["merge_strategy"]），
        # 等于蓝图那一路 merge 文本永远拿不到，只剩子目标整合方案兜底。
        _bp = getattr(ctx, "blueprint", None) or {}
        _bp_merge = _bp.get("merge_strategy", "") if isinstance(_bp, dict) \
            else str(getattr(_bp, "merge_strategy", "") or "")
        merge_text = _bp_merge or getattr(ctx, "subgoal_merge_plan", "") or ""
        if not merge_text:
            logger.debug("[C-lite] 跳过数值攻击：无蓝图 merge 文本"
                         "（blueprint.merge_strategy / subgoal_merge_plan 均空）")
            self.record(ctx, "value_attack",
                        "跳过：无蓝图 merge 文本")            # Z2 2026-09-17
            return
        # 只对"极值声称"题下手（其余题型放行，零误伤）
        import re as _re2
        if not (_re2.search(r"max|min|最大|最小", merge_text, _re2.I)):
            logger.debug("[C-lite] 跳过数值攻击：merge 文本无极值关键词")
            self.record(ctx, "value_attack",
                        "跳过：merge 文本无极值关键词")        # Z2 2026-09-17
            return
        from .value_attack import attack_value_claim
        from utils.prefill import prefill_messages, stitch
        # 1) LLM 从题目 + 声称提取: 方向/声称值/目标函数 Python 源码
        # ★★★ 2026-09-16 修复（实测驱动）：本攻击是**连续优化**专用框架
        #   （要求 LLM 给出 `f(x)` 与 `sample_point()`），但触发条件
        #   **只查 merge 文本是否含 max/min/最大/最小** ⇒ **组合题**只要提到
        #   "最小"就会被拉进来。
        #   实测 official112-013（组合极值：求最小 d 使任意整数序列存在子集和
        #   落在 1810±d）：LLM 只能为**连续**框架瞎编一个 `f`，采样得
        #   `best=0.000000`，而 `claimed=54.0` ⇒ 判"证伪"并写入
        #   `audit_reject_feedback` ⇒ **用一个无意义的采样值驱动了修订**
        #   （该题 revise_round=4）。**在极值声称正确时同样会误杀**。
        #   修法：让提取器**同时判定题目是否属连续优化**，非连续则跳过。
        sys_p = (
            "你是数值提取器。判断题目**是否为连续优化/极值问题**"
            "（即目标可写成关于实数变量 x[0],x[1],… 的连续函数，且约束为"
            "连续可判定）。\n"
            "**组合/数论/离散/图论极值问题一律判 false**"
            "（例如『求最小 d 使任意整数序列存在子集和满足…』是组合极值，不是连续优化）。\n"
            "输出 JSON：\n"
            "{\"is_continuous_opt\": true|false, "
            "\"direction\": \"max\"|\"min\", \"claimed\": <声称的极值数值近似>, "
            "\"code\": \"def f(x): ... 用 x[0],x[1]... 计算目标函数返回 float; "
            "def sample_point(): 返回一个满足约束的可行点 list\"}\n"
            "注意：claimed 是把题目声称的极值（含根式分数）算出的十进制近似；"
            "code 必须自含约束检查（不可行点返回 None）；禁止 import 外部库之外的；"
            "只输出 JSON。"
        )
        user_p = f"题目：\n{ctx.problem[:1500]}\n\n声称的答案/蓝图结论：\n{merge_text[:800]}"
        # ★ 2026-09-17（Z1）：**前置**跳过离散/组合类题。
        # 原实现把 `is_continuous_opt` 判定放在下面那次 LLM 调用**之后**
        # （`max_tokens=32768`，最坏 2×300s 超时重试）⇒ 组合/数论/离散题
        # 也**先付一次超长调用**才被跳过（实测 002/003/013/025 四题最终都判为
        # 非连续优化，即那次调用纯浪费）。
        # 判据**只用 `ctx.domain`**（高精度、零误伤）：仅在明确属离散领域时前置
        # 跳过；其余仍由 LLM 的 `is_continuous_opt` 判定（下方原逻辑完全不动）。
        _dom_now = str(getattr(ctx, "domain", "") or "")
        if any(_k in _dom_now for _k in
               ("组合", "数论", "离散", "图论", "数理逻辑")):
            self.record(ctx, "value_attack",
                        "跳过数值攻击：领域=%s 属离散/组合类，连续采样框架不适用"
                        " —— **前置跳过，未付出 LLM 调用**（Z1 2026-09-17）"
                        % _dom_now[:20])
            return
        raw = self.llm(ctx, prefill_messages(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user_p}], '{"direction":'), 0.0, 32768)
        if not raw:
            logger.debug("[C-lite] 跳过数值攻击：LLM 提取返回空")
            self.record(ctx, "value_attack",
                        "跳过：LLM 提取返回空（**本次已付出一次超长调用**）")
            return
        raw = stitch('{"direction":', raw)
        m = _re2.search(r"\{[\s\S]*\}", raw)
        if not m:
            logger.debug("[C-lite] 跳过数值攻击：LLM 输出无 JSON 块")
            self.record(ctx, "value_attack",
                        "跳过：LLM 输出无 JSON 块（**已付出调用**）")
            return
        import json as _json
        try:
            parsed = _json.loads(m.group())
        except (_json.JSONDecodeError, ValueError) as exc:
            logger.debug("[C-lite] 跳过数值攻击：JSON 解析失败 %s", exc)
            self.record(ctx, "value_attack",
                        "跳过：JSON 解析失败（**已付出调用**）：%s" % str(exc)[:80])
            return
        direction = str(parsed.get("direction") or "")
        claimed = parsed.get("claimed")
        code = str(parsed.get("code") or "")
        # ★ 2026-09-16：非连续优化题**直接跳过**（见上方 sys_p 的注释）。
        #   缺省视为 True 以保持旧行为（只在新字段缺失时如此，避免误跳过）。
        if parsed.get("is_continuous_opt") is False:
            self.record(ctx, "value_attack",
                        "跳过数值攻击：提取器判定本题**不是**连续优化/极值问题"
                        "（组合/数论/离散类），连续采样框架不适用")
            return
        if direction not in ("max", "min") or claimed is None:
            logger.debug("[C-lite] 跳过数值攻击：direction=%r claimed=%r 不合法",
                         direction, claimed)
            self.record(ctx, "value_attack",
                        "跳过：direction/claimed 不合法"
                        "（**已付出调用**）direction=%r claimed=%r"
                        % (direction, claimed))               # Z2 2026-09-17
            return
        # 2) 数值攻击
        result = attack_value_claim(
            ctx.problem or "", float(claimed), direction, code)
        if not result.get("ok"):
            reason = result.get("reason", "数值攻击证伪")
            ctx.blueprint_value_false = reason
            self.record(ctx, "value_attack",
                        f"蓝图极值证伪（{direction}={claimed}）：{reason[:160]}")
            # 传给下游：候选生成/求解时提示蓝图方向不可信
            ctx.revise_feedback = list(getattr(ctx, "revise_feedback", []) or []) + [
                f"[数值攻击] 蓝图声称极值 {claimed} 已被数值采样证伪"
                f"（发现 {result.get('found', '?')}）。请重新推导极值，"
                f"不要沿用蓝图的候选值。{reason[:200]}"]
            # P1（2026-09-10）：同步写入**修订驱动通道**。
            # 背景：official112 全量实测——数值攻击已发现反例 47 题（其中错题 36 题），
            # 但仅 6 题触发修订；根因是本函数此前只写 revise_feedback（提示通道），
            # 而 _deep_revise_loop 只消费 audit_reject_feedback（修订驱动通道）。
            ctx.audit_reject_feedback = list(
                getattr(ctx, "audit_reject_feedback", []) or []) + [
                f"[数值攻击证伪] 蓝图声称极值 {claimed} 已被采样证伪"
                f"（发现 {result.get('found', '?')}）：{reason[:200]}"]
        else:
            self.record(ctx, "value_attack",
                        f"蓝图极值未被证伪（采样上界 "
                        f"{result.get('found', '?')}，声称 {claimed}）")
    except Exception as _e:  # noqa: BLE001  攻击失败不影响主流程
        self.record(ctx, "value_attack",
                    f"数值攻击异常跳过: {str(_e)[:120]}")
