from __future__ import annotations

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
from agent.switch_registry import (
    get_bool as _sw_bool, get_num as _sw_num)
"""
确定性验证通道（DeterministicChecker）
=====================================

纯本地、零 LLM 预算的答案旁证/硬否决验证器（v2.8，自 sq 分支增量移植）。设计来源：
- DeepSeek-Harness 的 code-runner 判分哲学：用确定性计算（而非模型直觉）验证答案
- Intern-MO 的自验证思想：多证据汇审，确定性证据权重最高

能力：
1. verify_by_substitution — 等式/恒等式代入采样验证（随机采样变量组，数值核对）
2. search_counterexample — 反例搜索（对一般性命题尝试数值反例；LLM 只生成候选、程序只判定）
3. numerical_backtrack — 数值回溯（把可解析的答案转为数值串，供一致性旁证）
4. check_answer — 高层入口：对单个候选答案做综合确定性旁证/硬否决

安全原则（宁可 unknown 绝不误杀）：
- 一切解析失败/计算失败都返回 unknown；
- 代入回验只作用于「等式一侧为纯常量」的方程（防止定义式/求值式的误杀）；
- 反例必须经程序数值验证才生效；
- 无 SymPy 时全部优雅降级为 unknown。
"""

import logging
import random
import re
import threading

logger = logging.getLogger("MathPilot.Deterministic")

try:
    import sympy as sp
    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - 环境缺依赖时的降级路径
    sp = None
    _HAS_SYMPY = False
    logger.warning("SymPy 未安装，确定性验证通道不可用")

# 2026-10-08 批 D1：「函数方程回代」簇 + 解析前置工具已搬到 deterministic__funceq.py。
# 原位 re-export 保外部 `from agent.deterministic import <name>` 契约；类内跳板保
# `self._x(...)` / `DeterministicChecker._x(...)` 契约。⚠ 勿按"未使用"清理。
from . import deterministic__funceq as _det_funceq
from .deterministic__funceq import (  # noqa: F401
    _normalize_answer, _math_trim,
    _latex_prep, _MATH_CHARS, _FUNC_SYMBOL_BLACKLIST)

# 代入回验的容差（浮点噪声）
_SUBSTITUTION_TOL = 1e-4

# 检查入口超时护栏（2026-09-13）：病态 SymPy 表达式会让 sp.simplify 卡死，实测单次
# 460s/116s；而单题硬时限仅 1200s，且 3.6 / 4_verify / 6.5 三处各跑一次 check_answer。
# 写法对齐 utils/math_eval._subst_with_timeout（join(_SYM_TIMEOUT_SEC)）与
# symbolic_solve.solve_with_tool（join(_SOLVE_TIMEOUT_SEC)）：daemon 线程 + join 超时。
# ★ 2026-10-07：超时秒数改为**调用时实时读取** `_sw_num("deterministic_timeout_sec")`
#   （见 `check_answer` 内 `det_timeout`）—— 原先在此处 import 期求值成模块常量，
#   会把 `ReasoningAgent.__init__` 之前的注册表值固化，导致 config/CLI 覆盖失效。
_DET_HUNG_MAX = 8      # 累计超时上限，超过后整体降级（防无界建线程）
_DET_HUNG = 0          # 累计超时次数；成功一次衰减 1，防跨题永久降级


def _safe_parse(text: str):
    """安全解析表达式 → SymPy 表达式；失败返回 None（永不抛异常）。

    数学文本常见隐式乘法（"2x"、"(x+1)(x-1)"），sympify 原生不支持，
    这里先做保守的补全（不做 letter( 规则，避免破坏 sin(x)/f(x) 函数调用）。
    """
    if not _HAS_SYMPY or not text or not text.strip():
        return None
    try:
        from utils.sympy_tools import _try_parse
        processed = _fix_implicit_multiplication(text.strip())
        parsed, _err = _try_parse(processed)
        return parsed
    except Exception:  # pragma: no cover - 防御性兜底
        return None


def _fix_implicit_multiplication(text: str) -> str:
    """为 sympify 补全隐式乘法：5x→5*x、2(→2*(、)x→)*x、)(→)*(。

    不做「字母+(」规则——"sin(x)" 会被误改成 "sin*(x)"。
    """
    t = text
    t = re.sub(r"(\d)([a-zA-Zα-ω])", r"\1*\2", t)  # 5x → 5*x
    t = re.sub(r"(\d)\(", r"\1*(", t)              # 2( → 2*(
    t = re.sub(r"\)\s*([a-zA-Zα-ω])", r")*\1", t)  # )x → )*x
    t = re.sub(r"\)\s*\(", r")*(", t)              # )( → )*(
    return t




# ======================================================================
# 2026-10-07 解答题判别力重写（fix-audit-gate）
# ----------------------------------------------------------------------
# 诊断（smoke10 = results/_cloud/results/results/run_2026-10-06_smoke10_1006_2347.jsonl，
# 10 题实测；8 道解答题 000-007 + 2 道选择题 087/088）：
#   · audit_gate 共 38 条 candidate_audit：verdict 全 = unknown、applicable 全 = false；
#   · **第一真因（可修的 bug）**：答案仍带 `\boxed{...}` 外壳
#     → `_safe_parse('\boxed{-4}')` 返回 None → `_check_answer_core` 在
#     method="parse" 就 return unknown。7/10 题命中（000/001/002/003/005/006/007
#     全是被 boxed 的答案）。即 **Level0 根本没进入任何判据**，不是"判不了"。
#   · **第二真因（判据过窄）**：剥壳后，原 `_check_answer_core` 只认
#     `_extract_constant_equation` 返回的"最后一条一侧为常量的单变量等式"；
#     而 smoke10 的解答题是竞赛题（计数 000/004、函数方程 006、极值 001/002、
#     集合完备性 003/005、F2 约束 007）⇒ 无此结构 ⇒ 全 unknown（method="parse_only"）。
# 重写（全部由开关 `deterministic_solution_check` 控制，默认 True = 修好后行为）：
#   ① 解析前归一答案外壳（`_normalize_answer`）；
#   ② 常量等式回代由"只取最后一条"改为"全量等式合取判定"
#      （任一条 fail ⇒ fail；至少一条 pass 且无 fail ⇒ pass）；
#   ③ 新增**函数方程回代**判据（`_verify_functional_equations`）：题目给函数方程、
#      答案给函数定义时，把定义符号代入方程做恒等判定 —— 能拦错误函数、放行正确函数。
# 仍无法覆盖（诚实记录，不伪造判别力）：计数（需按题建图 + DP/枚举）、
#   极值/完备性（需证明上下界）、F2 约束传播（需按题解析关系并传播）。这些属
#   "客观局部不可判定"，一律返回 unknown。
# ======================================================================








class DeterministicChecker:
    """确定性验证通道：纯本地计算，不消耗 LLM 预算、不触发限流。"""

    # 2026-10-08 批 D1：函数方程回代簇实现见 deterministic__funceq.py（跳板保契约）。
    _detect_function_symbols = staticmethod(_det_funceq._detect_function_symbols)
    _extract_functional_equation = staticmethod(_det_funceq._extract_functional_equation)
    _parse_with_function = _det_funceq._parse_with_function
    _extract_function_definitions = staticmethod(_det_funceq._extract_function_definitions)
    _check_one_function = _det_funceq._check_one_function
    _numerically_zero = _det_funceq._numerically_zero
    _verify_functional_equations = _det_funceq._verify_functional_equations

    def __init__(self, samples: int = 100, attempts: int = 200,
                 tol: float = 1e-6, seed: int = 42) -> None:
        """初始化采样/搜索参数与固定随机种子（可复现）。

        Args:
            samples: 代入采样验证的采样组数。
            attempts: 反例搜索的最大尝试次数。
            tol: 数值比较容差。
            seed: 随机种子（同题多次运行结果一致，便于 A/B 复现）。
        """
        self.samples = samples
        self.attempts = attempts
        self.tol = tol
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 数值化辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _num(expr) -> float | None:
        """把 SymPy 表达式转为 float；NaN/异常返回 None。"""
        if expr is None:
            return None
        try:
            val = float(sp.N(expr))
            if val != val:  # NaN
                return None
            return val
        except Exception:
            return None

    def _sample_value(self, symbol: str, ranges: dict | None = None) -> float:
        """按范围/默认区间采样一个数值。整型变量名（n/m/k/i/j 等）采整数。"""
        if ranges and symbol in ranges:
            lo, hi = ranges[symbol]
            if symbol in _INT_VARIABLES:
                return float(self._rng.randint(int(lo), int(hi)))
            return self._rng.uniform(float(lo), float(hi))
        if symbol in _INT_VARIABLES:
            return float(self._rng.randint(-20, 20))
        return self._rng.uniform(-10.0, 10.0)

    # ------------------------------------------------------------------
    # 1) 代入采样验证
    # ------------------------------------------------------------------
    def verify_by_substitution(self, expr: str, variables: list | None = None,
                               samples: int | None = None,
                               tol: float | None = None) -> dict:
        """对等式/恒等式做代入采样验证。

        支持两种形态：
        - 含 "="：如 "x^2+2x+1=(x+1)^2"——随机采样变量组，核对两侧数值差。
        - 不含 "="：仅验证表达式可解析且数值有限（弱验证）。

        Returns:
            {"verdict": "pass"|"fail"|"unknown", "evidence": str, "passed": int, "total": int}
        """
        samples = samples or self.samples
        tol = tol or self.tol
        if not _HAS_SYMPY or not expr:
            return {"verdict": "unknown", "evidence": "sympy 不可用或表达式为空",
                    "passed": 0, "total": 0}
        try:
            if "=" in expr:
                left_s, right_s = expr.split("=", 1)
                left, right = _safe_parse(left_s), _safe_parse(right_s)
                if left is None or right is None:
                    return {"verdict": "unknown", "evidence": "等式两侧解析失败",
                            "passed": 0, "total": 0}
                symbols = sorted(
                    {str(s) for s in set(left.free_symbols) | set(right.free_symbols)},
                    key=str,
                )
                if not symbols:
                    # 无变量：直接数值核对
                    diff = self._num(sp.simplify(left - right))
                    if diff is None:
                        return {"verdict": "unknown", "evidence": "常量等式计算失败",
                                "passed": 0, "total": 0}
                    ok = abs(diff) <= tol
                    return {"verdict": "pass" if ok else "fail",
                            "evidence": f"常量等式核对: 差={diff:.6g}",
                            "passed": 1 if ok else 0, "total": 1}
                passed, valid = 0, 0
                for _ in range(samples):
                    subs = {sp.Symbol(s): self._sample_value(s) for s in symbols}
                    val = self._num((left - right).subs(subs))
                    if val is None:
                        continue  # 奇点/无定义点跳过
                    valid += 1
                    if abs(val) <= tol:
                        passed += 1
                if valid == 0:
                    return {"verdict": "unknown", "evidence": "全部采样点无定义，无法判定",
                            "passed": 0, "total": 0}
                ratio = passed / valid
                if ratio >= 0.95:
                    verdict = "pass"
                elif ratio <= 0.5:
                    verdict = "fail"
                else:
                    verdict = "unknown"
                return {"verdict": verdict,
                        "evidence": f"代入采样 {passed}/{valid} 组一致 (比值={ratio:.0%})",
                        "passed": passed, "total": valid}
            # 不含 "="：弱验证（可解析 + 数值有限）
            parsed = _safe_parse(expr)
            if parsed is None:
                return {"verdict": "unknown", "evidence": "表达式解析失败",
                        "passed": 0, "total": 0}
            val = self._num(parsed)
            if val is None:
                return {"verdict": "unknown", "evidence": "表达式求值失败",
                        "passed": 0, "total": 0}
            return {"verdict": "pass",
                    "evidence": f"表达式可解析且有限 (≈{val:.6g})",
                    "passed": 1, "total": 1}
        except Exception as e:  # pragma: no cover - 防御性兜底
            return {"verdict": "unknown", "evidence": f"代入采样异常: {str(e)[:80]}",
                    "passed": 0, "total": 0}

    # ------------------------------------------------------------------
    # 2) 反例搜索
    # ------------------------------------------------------------------
    def search_counterexample(self, statement: str, variables: list | None = None,
                              ranges: dict | None = None,
                              attempts: int | None = None) -> dict:
        """对一般性命题（如 "n^2 >= n"）尝试数值反例。

        只对「可解析成 左式 关系符 右式」的命题有效；无法解析 → unknown/未找到。

        **注意：本方法只负责程序化判定，LLM 生成的候选反例必须经本方法验证后才算数。**

        Returns:
            {"found": bool, "counterexample": dict|None, "attempts": int, "statement": str}
        """
        attempts = attempts or self.attempts
        if not _HAS_SYMPY or not statement:
            return {"found": False, "counterexample": None, "attempts": 0,
                    "statement": statement}
        try:
            m = re.match(r"^\s*(.+?)\s*(>=|<=|>|<|=|≠|!=)\s*(.+?)\s*$", statement)
            if not m:
                return {"found": False, "counterexample": None, "attempts": 0,
                        "statement": statement}
            left_s, op, right_s = m.group(1), m.group(2), m.group(3)
            left, right = _safe_parse(left_s), _safe_parse(right_s)
            if left is None or right is None:
                return {"found": False, "counterexample": None, "attempts": 0,
                        "statement": statement}
            symbols = sorted(
                {str(s) for s in set(left.free_symbols) | set(right.free_symbols)},
                key=str,
            )
            if not symbols:
                return {"found": False, "counterexample": None, "attempts": 0,
                        "statement": statement}
            diff = sp.simplify(left - right)
            for _ in range(attempts):
                subs = {sp.Symbol(s): self._sample_value(s, ranges) for s in symbols}
                val = self._num(diff.subs(subs))
                if val is None:
                    continue
                if op in (">=", "≥"):
                    violated = val < -self.tol
                elif op in ("<=", "≤"):
                    violated = val > self.tol
                elif op == ">":
                    violated = val <= self.tol
                elif op == "<":
                    violated = val >= -self.tol
                elif op in ("≠", "!="):
                    violated = abs(val) <= self.tol
                else:  # "="
                    violated = abs(val) > self.tol
                if violated:
                    return {"found": True,
                            "counterexample": {k: float(v) for k, v in subs.items()},
                            "attempts": attempts, "statement": statement}
            return {"found": False, "counterexample": None, "attempts": attempts,
                    "statement": statement}
        except Exception as e:  # pragma: no cover - 防御性兜底
            return {"found": False, "counterexample": None, "attempts": 0,
                    "statement": statement, "error": str(e)[:80]}

    # ------------------------------------------------------------------
    # 3) 数值回溯
    # ------------------------------------------------------------------
    def numerical_backtrack(self, answer: str) -> str | None:
        """数值回溯：把可解析的答案转为数值串（用于一致性旁证）。失败返回 None。"""
        parsed = _safe_parse(answer)
        if parsed is None:
            return None
        val = self._num(parsed)
        if val is None:
            return None
        return f"{val:.10g}"

    # ------------------------------------------------------------------
    # 4) 高层入口：综合旁证 / 硬否决
    # ------------------------------------------------------------------
    def check_answer(self, ctx, problem: str, answer: str,
                     domain: str | None = None) -> dict:
        """对单个候选答案做确定性旁证/硬否决（**带内部超时护栏**）。

        Args:
            ctx: TaskContext（仅保留参数兼容，可传 None）。
            problem: 题目文本。
            answer: 候选最终答案。
            domain: 领域（预留，暂未使用）。

        Returns:
            {"verdict": "pass"|"fail"|"unknown", "confidence": float,
             "evidence": str, "method": str}

        超时后返回 verdict="unknown"（"不改判"语义，绝不返回 fail 误杀正确候选）。
        """
        global _DET_HUNG
        if _DET_HUNG >= _DET_HUNG_MAX:
            logger.warning(
                "确定性检查已降级（历史超时 %d 次 ≥ %d），返回 unknown",
                _DET_HUNG, _DET_HUNG_MAX)
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": "确定性检查已降级（历史超时过多）",
                    "method": "timeout_guard"}
        # 调用时实时读取（非 import 期常量）⇒ config/CLI 覆盖生效
        det_timeout = _sw_num("deterministic_timeout_sec")
        box: dict = {}

        def _run() -> None:
            try:
                box["r"] = self._check_answer_core(problem, answer, domain)
            except Exception as exc:  # noqa: BLE001  线程内异常回传，不逃逸
                box["e"] = exc

        th = threading.Thread(target=_run, name="det_check", daemon=True)
        th.start()
        th.join(det_timeout)
        if th.is_alive():                      # 卡死 → 放弃该线程（daemon，不阻塞退出）
            _DET_HUNG += 1
            logger.warning("确定性检查超时(>%.1fs)，返回 unknown（不改判）: answer=%s",
                           det_timeout, (answer or "")[:80])
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": f"确定性检查超时(>{det_timeout:.1f}s)",
                    "method": "timeout"}
        if _DET_HUNG:                          # 成功时让累计计数衰减（防跨题永久降级）
            _DET_HUNG -= 1
        if "e" in box:
            logger.debug("确定性检查线程异常 → unknown: %s: %s",
                         type(box["e"]).__name__, box["e"])
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": f"确定性检查异常: {type(box['e']).__name__}",
                    "method": "error"}
        return box.get("r") or {"verdict": "unknown", "confidence": 0.0,
                                "evidence": "确定性检查无结果", "method": "none"}

    def _check_answer_core(self, problem: str, answer: str,
                           domain: str | None = None) -> dict:
        """check_answer 的实际检查逻辑（在受超时护栏保护的 daemon 线程内执行）。

        2026-10-07 重写（见文件头诊断）：
        - 开关 ON（默认）：先归一答案外壳（`\\boxed{}` 等）再解析；常量等式回代改为
          **全量等式合取判定**；解析/回代皆无法判定时追加**函数方程回代**判据。
        - 开关 OFF：与改动前逐字一致（raw parse、只取最后一条等式、无函数方程判据）。
        """
        if not _HAS_SYMPY:
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": "sympy 不可用", "method": "none"}
        if not answer or not answer.strip():
            return {"verdict": "unknown", "confidence": 0.0,
                    "evidence": "答案为空", "method": "none"}
        raw_answer = answer.strip()
        on = _sw_bool("deterministic_solution_check")
        answer = _normalize_answer(raw_answer) if on else raw_answer
        evidence_parts = []
        # 1) 答案可解析性 + 数值回溯
        parsed = _safe_parse(answer)
        if parsed is None:
            # 开关 OFF：与改动前一致，就地返回 unknown。
            # 开关 ON：不早退 —— 答案可能是多解/函数定义等形式（非单一表达式），
            #   下面第 3 步的函数方程回代会自己从原文抽函数定义，必须给它机会。
            if not on:
                return {"verdict": "unknown", "confidence": 0.0,
                        "evidence": f"答案无法解析为表达式: {raw_answer[:80]}",
                        "method": "parse"}
            evidence_parts.append(
                f"答案非单一表达式（无法直接解析）: {raw_answer[:60]}")
        else:
            val = self._num(parsed)
            if val is not None:
                evidence_parts.append(f"数值≈{val:.6g}")
        # 2) 题目含「一侧为常量」的等式 → 代入回验
        if on:
            eqs = self._extract_all_constant_equations(problem)
        else:
            _one = self._extract_constant_equation(problem)
            eqs = [_one] if _one else []
        eq_pass, eq_fail = None, None
        for eq in eqs:
            check = self._verify_answer_in_equation(eq, answer)
            v = check.get("verdict")
            if v == "fail":
                eq_fail = check
                break
            if v == "pass" and eq_pass is None:
                eq_pass = check
        if eq_fail is not None:
            eq_fail["confidence"] = eq_fail.get("confidence", 1.0)
            eq_fail["method"] = "substitution"
            return eq_fail
        if eq_pass is not None:
            eq_pass["confidence"] = eq_pass.get("confidence", 1.0)
            eq_pass["method"] = "substitution"
            return eq_pass
        if eqs:
            evidence_parts.append("已提取 %d 条常量等式但均无法判定" % len(eqs))
        # 3) 函数方程回代（解答题子类：题目给函数方程、答案给函数定义）
        if on:
            fe = self._verify_functional_equations(problem, answer)
            if fe.get("verdict") != "unknown":
                return fe
            if fe.get("evidence"):
                evidence_parts.append(fe["evidence"])
        # 4) 无法进一步验证 → unknown（附可解析证据）
        evidence = "；".join(p for p in evidence_parts if p)
        return {"verdict": "unknown", "confidence": 0.0,
                "evidence": evidence or "答案可解析但无等式可代入",
                "method": "parse_only"}

    def _extract_all_constant_equations(self, problem: str) -> list:
        """收集题目中**全部**「一侧为纯常量、另一侧恰含一个自由符号」的等式。

        原 `_extract_constant_equation` 只返回**最后一条**；对"联立/多约束"类解答题，
        最后一条未必能核验答案。这里返回全量，由 `_check_answer_core` 做合取判定
        （任一条 fail ⇒ fail；至少一条 pass 且无 fail ⇒ pass）。
        """
        if not problem or not _HAS_SYMPY:
            return []
        out, seen = [], set()
        try:
            for seg in re.split(r"[，。；;\n]", problem):
                if seg.count("=") != 1:
                    continue
                left_s, right_s = seg.split("=", 1)
                left = _safe_parse(_math_trim(left_s))
                right = _safe_parse(_math_trim(right_s))
                if left is None or right is None:
                    continue
                lf, rf = list(left.free_symbols), list(right.free_symbols)
                key = None
                if not lf and len(rf) == 1:
                    key = _math_trim(seg)
                elif not rf and len(lf) == 1:
                    key = _math_trim(seg)
                if key and key not in seen:
                    seen.add(key)
                    out.append(key)
        except Exception:  # noqa: BLE001
            return out
        return out


    # ------------------------------------------------------------------
    # 内部：等式提取与代入回验
    # ------------------------------------------------------------------
    def _extract_constant_equation(self, problem: str) -> str | None:
        """从题目中提取最后一个「一侧为纯常量」的可解析等式（保守策略）。

        只返回一侧无自由符号的等式——例如 "x^2-5x+6=0"（右侧为常量 0）。
        定义式（如 "f(x)=x^2-5x+6"）两侧都含变量 → 跳过，防止把求值题误杀。
        """
        if not problem or not _HAS_SYMPY:
            return None
        try:
            segments = re.split(r"[，。；;\n]", problem)
            candidates = [s.strip() for s in segments if "=" in s]
            for seg in reversed(candidates):
                if seg.count("=") != 1:
                    continue
                left_s, right_s = seg.split("=", 1)
                left = _safe_parse(_math_trim(left_s))
                right = _safe_parse(_math_trim(right_s))
                if left is None or right is None:
                    continue
                left_free = list(left.free_symbols)
                right_free = list(right.free_symbols)
                if not left_free and len(right_free) == 1:
                    return seg
                if not right_free and len(left_free) == 1:
                    return seg
            return None
        except Exception:  # pragma: no cover - 防御性兜底
            return None

    def _verify_answer_in_equation(self, equation: str, answer: str) -> dict:
        """把答案代入等式验证：|LHS-RHS| ≈ 0 → pass；明显不为 0 → fail。"""
        try:
            left_s, right_s = equation.split("=", 1)
            left = _safe_parse(_math_trim(left_s))
            right = _safe_parse(_math_trim(right_s))
            if left is None or right is None:
                return {"verdict": "unknown", "evidence": "等式解析失败",
                        "method": "substitution"}
            free = list(set(left.free_symbols) | set(right.free_symbols))
            if len(free) != 1:
                return {"verdict": "unknown", "evidence": "等式含多变量，跳过代入",
                        "method": "substitution"}
            var = free[0]
            ans_expr = _safe_parse(answer)
            if ans_expr is None:
                return {"verdict": "unknown", "evidence": "答案不可解析",
                        "method": "substitution"}
            diff = sp.simplify(left - right)
            # 答案本身含变量（如 x=2 之外的表达式）→ 尝试符号验证
            if any(s in list(ans_expr.free_symbols) for s in free):
                try:
                    if sp.simplify(diff.subs(var, ans_expr)) == 0:
                        return {"verdict": "pass",
                                "evidence": f"符号代入验证通过: {var}={answer}",
                                "method": "substitution"}
                except Exception as exc:
                    # 2026-09-04 审核：符号代入异常 → unknown（合理降级），
                    # 但原来无日志，无法区分"真不可代入"与"代码 bug"。留证据。
                    logger.debug("符号代入验证异常 → unknown: %s: %s",
                                 type(exc).__name__, exc)
                return {"verdict": "unknown", "evidence": "答案含变量，无法数值代入",
                        "method": "substitution"}
            val = self._num(diff.subs(var, ans_expr))
            if val is None:
                return {"verdict": "unknown", "evidence": "代入数值计算失败",
                        "method": "substitution"}
            if abs(val) <= _SUBSTITUTION_TOL:
                return {"verdict": "pass",
                        "evidence": f"代入验证通过: {equation}，{var}={answer} → 差≈{val:.2e}",
                        "method": "substitution"}
            return {"verdict": "fail",
                    "evidence": f"代入验证失败: {equation}，{var}={answer} → 差={val:.4g}",
                    "method": "substitution"}
        except Exception as e:  # pragma: no cover - 防御性兜底
            return {"verdict": "unknown", "evidence": f"代入验证异常: {str(e)[:80]}",
                    "method": "substitution"}


# 常见整型变量名（采样时采整数，贴近数论/组合题语境）
_INT_VARIABLES: frozenset = frozenset({"n", "m", "k", "i", "j", "p", "q"})


# ============================================================
# 2026-09-10 L1：精确代入核验（与上面的浮点采样通道并列）
# ------------------------------------------------------------
# verify_by_substitution 走的是"随机采样 + 浮点容差"，适合核恒等式；
# 而"模型给关系式、题面给具体数值"这种场景要求**精确**比对（1e-4 容差会把
# 2.0001 判成 2）。这里补一条精确通道：代入 → Fraction 精确值 → 精确判等。
# 严格宁漏勿误：任一侧无法精确数值化 → unknown（绝不误杀）。
# ============================================================


