from __future__ import annotations
"""候选答案证伪层（AnswerFalsifier，2026-09-23 新建）。

要解决的问题
============
`AnswerOracle` 只能做「多候选自洽共识」，**无法证明某个候选是错的** ——
它没有 reference，判定不了绝对对错。后果（0923 四维分析实证）：

- 46 题里 25 题出现「全部候选 0 票」⇒ 触发 `_zero_vote_fallback`，
  系统**弃用整池**、改取 `direct_solve` 的另一条产线答案（终答与候选池无血缘）；
- 其中 **18 题终答出池、16 题判错（占全部错题的 43.2%）**；
- 更糟：`official112-015` 的候选 `\\boxed{2}` 与 gold 完全同形、仍被判 0/3；
  `official112-021` 里**错答 `\\boxed{39}` 拿 3/3、gold `\\boxed{21}` 也拿 3/3**
  ⇒ 投票没有判别力，只能靠簇规模选（选中错答）。

**核心信念（用户 2026-09-23 提出）：错误答案一定是能被证明错误的。**
所以不该问"哪个候选得票多"，而该问"**哪个候选能被证伪并踢掉**"。

设计
====
只做**证伪**，不做**证实**（证实交给 oracle / Lean gate 的既有通路）：

1. 让 LLM 产出**可机检的证伪证书**：把候选答案代回题目条件
   （"数值回带"），给出必须成立的闭式等式列表 `lhs == rhs`；
2. **本地用 SymPy 精确判定**每条等式（不信 LLM 的结论，只看它给的式子）；
3. **Lean 背书**（可选）：让 LLM 同时给出断言该候选合法的 Lean 代码；
   若 Lean 编译**失败**则加固证伪，若 Lean **通过**则**丢弃本次证伪**（防误杀）；
4. 只有"SymPy 判出确定性矛盾"且未被 Lean 推翻 ⇒ `verdict='incorrect'`；
   其余一律 `unknown`。

**红线：宁可漏证，绝不误杀。** 任何异常、任何不确定 ⇒ `unknown`。

隔离原则
========
与 `answer_oracle` 同：独立文件、不污染主流程、异常全吞、不阻断评测。
注意 `answer_oracle` 的 docstring 仍写着「2026-09-06 去 Lean 化：平台无 Lean 可执行
文件」—— **该前提已过时**：0923 云端实测 `lean_executable` 存在、
`lean_mcp` 332 次调用 0 失败。本模块据实使用 Lean。
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("MathPilot.Falsifier")


@dataclass
class FalsifyResult:
    """证伪结果（JSON 可序列化）。只有 incorrect / unknown 两种 verdict。"""
    verdict: str = "unknown"                  # 'incorrect' | 'unknown'
    reason: str = ""
    witness: str = ""                         # 人类可读的反例说明
    checks: list = field(default_factory=list)  # 机检明细 [{lhs,rhs,desc,verdict,detail}]
    oracle_type: str = "none"                 # 'sympy' | 'sympy+lean' | 'none'

    @property
    def is_incorrect(self) -> bool:
        return self.verdict == "incorrect"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason,
                "witness": self.witness, "checks": self.checks,
                "oracle_type": self.oracle_type}

    def to_feedback(self) -> str:
        """可注入 revise / 审核通道的反馈文本。"""
        if not self.is_incorrect:
            return ""
        lines = [f"【候选被客观证伪】{self.reason}"]
        for c in self.checks:
            if c.get("verdict") == "fail":
                lines.append(f"  - 代入后 {c.get('lhs')} = {c.get('rhs')} 不成立"
                             f"（{c.get('detail') or c.get('desc') or ''}）")
        if self.witness:
            lines.append(f"  反例：{self.witness}")
        return "\n".join(lines)


_CERT_PROMPT = """你是数学证明的「证伪员」。给你一道题和一个**待检验的候选答案**。

你的任务**不是**判断它对不对，而是尝试**把它证伪**：
把候选答案**代回题目条件**（数值回带），列出「若该答案正确则必须成立的等式」。

要求：
1. 只输出**闭式**（代入候选后不再含未知量）的等式，写成 `lhs` / `rhs` 两个字符串；
   lhs/rhs 必须是可直接计算的表达式，例如 "20**2 - 400" 与 "0"。
   只允许数字与 + - * / ** ( ) sqrt sin cos tan log exp 及常量 pi/E。
2. 每条等式必须在 `desc` 里**逐字摘录**它依据的题面原文（≤40 字，不得改写、
   不得概括、不得翻译成另一种语言）。摘不出原文，说明这条等式没有题面依据，
   必须删掉 —— **不要编造条件**；系统会逐字核对这段摘录是否真的在题面里。
3. 如果你能直接给出一个**具体反例**（代入后两侧不等的数值），把它写进 `witness`。
4. 如果该候选**无法**被数值回带检验（例如答案是集合/参数式/证明文本），
   `checks` 留空数组 —— **不要编造等式**。
5. 可选：给一段 Lean 4 代码 `lean_code`，声明该候选满足题目条件
   （若不满足，Lean 将编译失败，从而背书你的证伪）。无法给出就填空串。
6. 若候选的**错误是一条「蕴含式断言」**（形如"若 A 则 B"的推理错误，数值回带无法暴露），
   把它用 **SMT-LIB（Lisp 前缀）** 写成 `implication` 字段：用 declare-const/declare-fun
   声明变量，末尾一条 `(assert (not (=> A B)))`；系统用 Z3 判定，sat（有反例）即证伪。
   只用 Z3 标准算子（Int/Real/and/or/not/=>/<=/>=/=/*/+/-）；**变量的域必须与题面一致**（自然数/正整数用 Int 并显式约束 >=0 或 >=1，实数用 Real），否则会产生假反例导致误判；无法形式化就填空串。

只输出一个 JSON 对象，不要解释性前后缀、不要 Markdown 围栏：
{
  "checks": [
    {"desc": "来自条件：…", "lhs": "…", "rhs": "…"}
  ],
  "witness": "…",
  "lean_code": "…",
  "implication": "…"
}"""


class AnswerFalsifier:
    """候选答案证伪器（数值回带 + SymPy 精确判定 + Lean 背书）。"""

    name = "AnswerFalsifier"

    # 只允许这些记号进入 SymPy（防注入 / 防解析歧义）
    _SAFE_EXPR = re.compile(
        r"^[0-9a-zA-Z_+\-*/().,\s^]*$")
    _ALLOW_KW = ("sqrt", "sin", "cos", "tan", "log", "ln", "exp", "abs",
                 "pi", "E", "floor", "ceil", "factorial")

    def __init__(self, client=None, config=None, budget=None):
        self.client = client
        self.config = config
        self.budget = budget
        self._lean_bridge = None
        self._lean_tried = False

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def falsify(self, ctx, candidate, problem: str = "") -> FalsifyResult:
        """尝试证伪单个候选。返回 incorrect 或 unknown（**永不返回 correct**）。"""
        try:
            if not getattr(self.config, "enable_answer_falsifier", True):
                return FalsifyResult(reason="证伪器已关闭")
            prob = problem or getattr(ctx, "problem", "") or ""
            # ★ 2026-10-15（审核 S4）：优先取 4.0 规范化后的 answer_norm ——
            #   句子形答案（"所以最小 d 是 48。"）的净化收益此前只覆盖
            #   4.1/4.4/4.5，证伪处仍按原文处理。
            answer = str(getattr(candidate, "answer_norm", None)
                         or getattr(candidate, "answer", "") or "").strip()
            if not prob or not answer:
                return FalsifyResult(reason="缺少题面或答案")
            # 纯文本/超长答案不做数值回带（证明题、集合答案等）
            if len(answer) > 400:
                return FalsifyResult(reason="答案过长，非数值型")
            if ctx.gen_time_up():
                return FalsifyResult(reason="时间不足")
            if not self._spend_cap(ctx):
                return FalsifyResult(reason="本取证伪次数已达上限")

            cert = self._ask_certificate(ctx, prob, answer)
            if not cert:
                return FalsifyResult(reason="未取得证伪证书")

            checks = self._normalize_checks(cert.get("checks"))
            if not checks:
                return FalsifyResult(reason="证书无可机检等式")

            # ★ 2026-10-22（五道链审计·证据落地门）：等式必须带题面原文依据，
            #   否则机器锚点会给幻觉背书（编造 `1+1=3` ⇒ 正确答案被淘汰）。
            _dropped = []
            try:
                from .switch_registry import get_bool as _swb
                _grounding = _swb("five_gate_grounding")
            except Exception:  # noqa: BLE001
                _grounding = True
            if _grounding:
                checks, _dropped = self._drop_ungrounded(checks, prob)
                if not checks:
                    return FalsifyResult(
                        reason="证书等式均无题面依据（%d 条被丢弃）" % len(_dropped),
                        checks=[dict(c, verdict="ungrounded") for c in _dropped])

            fails, detail_lines = self._sympy_refute(checks)
            if not fails:
                # ★ 2026-10-18：SymPy 未证伪时，若证书带「蕴含式断言」，用 SMT 再验
                #   ——sat（有反例）即证伪（覆盖逻辑命题类错误 008/015）。
                _smt = self._smt_refute(ctx, cert)
                if _smt is not None:
                    return FalsifyResult(
                        verdict="incorrect",
                        reason="蕴含式断言被 SMT 反例证伪",
                        witness=str(_smt.get("model") or "")[:300],
                        checks=[dict(c, verdict="pass") for c in checks],
                        oracle_type="sympy+smt")
                return FalsifyResult(
                    reason="代入后未发现矛盾（不构成证伪）",
                    checks=[dict(c, verdict="pass") for c in checks],
                    oracle_type="sympy")

            lean_code = str(cert.get("lean_code") or "").strip()
            oracle_type = "sympy"
            if lean_code:
                backed = self._lean_corroborate(ctx, lean_code)
                if backed is False:
                    # Lean 成功编译 ⇒ 说明条件其实成立 ⇒ 丢弃本次证伪（防误杀）
                    return FalsifyResult(
                        reason="SymPy 判出矛盾但 Lean 编译通过 ⇒ 判为误报，放行",
                        checks=[dict(c, verdict="pass") for c in checks],
                        oracle_type="sympy+lean")
                if backed is True:
                    oracle_type = "sympy+lean"

            return FalsifyResult(
                verdict="incorrect",
                reason="候选答案代回题目条件后出现确定性矛盾",
                witness=str(cert.get("witness") or "").strip()[:300],
                checks=[dict(c, verdict=("fail" if c in fails else "pass"))
                        for c in checks],
                oracle_type=oracle_type)
        except Exception as e:  # noqa: BLE001  异常一律放行
            logger.warning("证伪器异常（放行）：%s", str(e)[:200])
            return FalsifyResult(reason="证伪器异常")

    def _smt_refute(self, ctx, cert) -> dict | None:
        """对证书的 `implication` 字段做 SMT 验证；sat（有反例）返回 dict，否则 None。

        ★ 2026-10-18（用户定案「检测机制缺失」）：覆盖 SymPy 无法验的**逻辑命题类**
        错误（008「有界⇒总变差有界」、015「非单调⇒等值点」）。未装 z3 / 失败 /
        无法形式化 ⇒ 一律 None（放行，不误杀）。
        """
        try:
            from .switch_registry import get_bool as _g
            if not _g("smt_falsify"):
                return None
        except Exception:  # noqa: BLE001
            return None
        _imp = str((cert or {}).get("implication") or "").strip()
        if not _imp:
            return None
        try:
            from tools.agent_toolkit.smt_check import verify_smt
            r = verify_smt(_imp)
            if r.get("verdict") == "sat":
                logger.info("SMT 证伪：蕴含式断言存在反例 %s", r.get("model"))
                return r
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("SMT 证伪异常（放行）：%s", str(e)[:160])
            return None

    # ------------------------------------------------------------------
    # 次数上限（与 numeric_lean 同口径，按题计数）
    # ------------------------------------------------------------------
    def _spend_cap(self, ctx) -> bool:
        """本次证伪是否**允许发起**。True = 允许；False = 已达上限。

        ★ 2026-10-04 修（D6，口径统一 + fail-open）：
        原实现是 `cap <= 0 → return False`，语义为「**0 = 禁用证伪**」——
        与**全项目口径相反**（`max_subgoals` / `candidate_pool_limit` /
        `max_tokens_cap` / `lean_numeric_max_per_q` 一律 **0 = 不限**）。
        这个反转极危险：把 `falsify_max_per_q` 设成 0（本意"放开上限"）
        实际会把**唯一的客观证伪器整个关掉**。现统一为 **0 或负 = 不限**。
        另：原 `except → return False` 会在任何异常下**静默永久禁用证伪**
        （审核 D6 指出），现改为 fail-open 并留痕。
        """
        try:
            meta = ctx.metadata if isinstance(ctx.metadata, dict) else {}
            cap = int(getattr(self.config, "falsify_max_per_q", 0) or 0)
            if cap <= 0:                       # 0/负 = 不限（对齐全项目口径）
                return True
            cnt = int(meta.get("falsify_count", 0) or 0)
            if cnt >= cap:
                return False
            meta["falsify_count"] = cnt + 1
            ctx.metadata = meta
            return True
        except Exception as exc:  # noqa: BLE001
            # ★ 2026-10-15（用户定案「删降级/保底」）：记账异常不再"放行本次"
            #   —— 异常本身是代码缺陷，报错暴露（原降级放行会掩盖）。
            raise RuntimeError(
                "[falsifier] 次数计数异常: %s: %s"
                % (type(exc).__name__, exc)) from exc

    # ------------------------------------------------------------------
    # LLM：取证伪证书
    # ------------------------------------------------------------------
    def _ask_certificate(self, ctx, problem: str, answer: str) -> Optional[dict]:
        msgs = [
            {"role": "system", "content": _CERT_PROMPT},
            {"role": "user",
             "content": "【题目】\n%s\n\n【待检验的候选答案】\n%s\n\n"
                        "请只输出一个 JSON 对象。" % (problem[:3000], answer)},
        ]
        # ★ 2026-10-02：2048→8192（DeepSeek 适配方案 A，统一上限）。
        #   实测本处曾撞 finish_reason=length（DeepSeek CoT 长，证伪 JSON 被腰斩）。
        #   ⚠ 该点经包装方法 self._chat(..., max_tokens=2048) 间接传值，
        #     原先 13 处清单 + 首版全仓扫（只扫 llm/chat 直接调用）均漏掉。
        raw = self._chat(ctx, msgs, max_tokens=16384)
        return self._parse_json(raw)

    def _chat(self, ctx, msgs, max_tokens: int) -> str:
        """优先流式（非流式在 120s 读超时下会必然失败，见 verifier._deep_llm 注释）。"""
        client = getattr(self, "client", None)
        chat = getattr(client, "chat", None)
        if ctx.budget is not None:
            try:
                ctx.budget.spend(1)
            except Exception:  # noqa: BLE001
                pass
        if not callable(chat):
            return ""
        try:
            out = chat(messages=msgs, temperature=0.0,
                       max_tokens=max_tokens, stream=True)
            if out is not None:
                return str(out)
        except TypeError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("证伪器流式调用失败，转非流式：%s", str(e)[:160])
        try:
            out = chat(messages=msgs, temperature=0.0, max_tokens=max_tokens)
            return "" if out is None else str(out)
        except Exception as e:  # noqa: BLE001
            logger.warning("证伪器调用失败：%s", str(e)[:160])
            return ""

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        if not raw:
            return None
        s = raw.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i:j + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    # ------------------------------------------------------------------
    # 证书清洗
    # ------------------------------------------------------------------
    @classmethod
    def _drop_ungrounded(cls, checks: list, problem: str) -> tuple:
        """★ 2026-10-22（五道链审计）：丢掉「题面里没有依据」的等式。

        为什么必须有这一步：证书是 LLM 写的，若它凭空编一条与本题无关的等式
        （桩测试：``1+1 = 3``），SymPy 会忠实地算出 2≠3 并判定「候选被证伪」——
        机器锚点反而给幻觉背书，正确答案被整条淘汰。判据是**可机器复核**的：
        ``desc`` 必须是题面原文的逐字摘录，系统归一化后做子串核对。

        Returns:
            ``(kept_checks, dropped)``；``dropped`` 供留痕（含丢弃原因）。
        """
        from .evidence_grounding import check_quote, norm_text
        import re as _re
        # 剥掉「来自条件：」「根据题意：」这类前缀，只留摘录正文
        _prefix = _re.compile(
            r"^(?:来自|出自|根据|依据|由)?(?:题目|题面|条件|题干|题意)?\s*[:：]?\s*")
        kept, dropped = [], []
        for c in checks:
            d = _prefix.sub("", str(c.get("desc") or "").strip())
            ok, ratio, miss = check_quote(d, problem, min_len=4, min_seg=3)
            if ok is True:
                kept.append(c)
            else:
                if not norm_text(d):
                    _note = "no_desc"
                elif ok is None:
                    _note = "unverifiable_too_short"
                else:
                    _note = "not_in_problem:%s" % "|".join(miss[:2])
                dropped.append({
                    "desc": str(c.get("desc") or "")[:120],
                    "expr": "%s = %s" % (c.get("lhs"), c.get("rhs")),
                    "note": _note,
                })
        return kept, dropped

    @classmethod
    def _normalize_checks(cls, checks) -> list:
        """只保留 lhs/rhs 都是**安全闭式**的条目；最多取 6 条。"""
        out = []
        if not isinstance(checks, list):
            return out
        for c in checks:
            if not isinstance(c, dict):
                continue
            lhs = str(c.get("lhs") or "").strip()
            rhs = str(c.get("rhs") or "").strip()
            if not lhs or not rhs:
                continue
            if not (cls._expr_safe(lhs) and cls._expr_safe(rhs)):
                continue
            out.append({"desc": str(c.get("desc") or "")[:160],
                        "lhs": lhs[:200], "rhs": rhs[:200]})
            if len(out) >= 6:
                break
        return out

    @classmethod
    def _expr_safe(cls, s: str) -> bool:
        """白名单：只允许数字/单字符变量/四则/幂/括号/逗号/空白 + 少量已知函数与常量。

        ⚠ 变量是**允许**的：数值回带时通常已代入为常数，但保留单字符变量不影响
        安全（SymPy 只把它当 Symbol）。真正的注入面是**多字母未知名**
        （`__import__` / `exec` / 属性穿透）—— `_SAFE_EXPR` 已挡下划线、引号、
        分号、方括号，这里再挡「两个及以上连续字母」即可。
        """
        if len(s) > 200:
            return False
        if not cls._SAFE_EXPR.match(s):
            return False
        t = s
        for kw in cls._ALLOW_KW:
            t = t.replace(kw, "")
        # 去掉白名单关键字后，剩余的字母不得连续出现 ≥2 个
        return not re.search(r"[A-Za-z]{2,}", t)

    # ------------------------------------------------------------------
    # SymPy 精确判定（红线：只在"确定性矛盾"时判 fail）
    # ------------------------------------------------------------------
    @staticmethod
    def _sympy_refute(checks: list) -> tuple:
        """逐条判 lhs == rhs。返回 (fail 列表, 说明列表)。

        判定纪律（与 sub_goal_solver._sympy_judge 同源）：
        - 两侧都必须能解析；任一失败 → 该条忽略（不构成证伪）
        - 差异含自由符号 → 忽略（不是闭式，无法确定）
        - 差异是**非零常数** ⇒ 确定性矛盾 ⇒ fail
        """
        fails = []
        details = []
        try:
            from utils.sympy_tools import _try_parse
        except Exception:  # noqa: BLE001
            return [], []
        try:
            import sympy as sp
        except Exception:  # noqa: BLE001
            return [], []
        for c in checks:
            try:
                a, _ = _try_parse(c["lhs"])
                b, _ = _try_parse(c["rhs"])
                if a is None or b is None:
                    details.append("跳过（无法解析）")
                    continue
                d = sp.simplify(a - b)
                if d == 0:
                    details.append("两侧相等（该条不构成证伪）")
                    continue
                if getattr(d, "free_symbols", None):
                    details.append("含自由符号，无法确定")
                    continue
                if d.is_number:
                    fails.append(c)
                    details.append("左式应等于 %s，与右侧相差 %s"
                                   % (sp.simplify(a).evalf(12), sp.simplify(d).evalf(12)))
                else:
                    details.append("差异非数值，无法确定")
            except Exception:  # noqa: BLE001
                details.append("判定异常（忽略）")
        return fails, details

    # ------------------------------------------------------------------
    # Lean 背书
    # ------------------------------------------------------------------
    def _get_bridge(self):
        if self._lean_tried:
            return self._lean_bridge
        self._lean_tried = True
        try:
            from tools.lean_local.lean_bridge import LeanBridge
            b = LeanBridge(self.client, self.config)
            self._lean_bridge = b if getattr(b, "lean_available", False) else None
        except Exception:  # noqa: BLE001
            self._lean_bridge = None
        return self._lean_bridge

    def _lean_corroborate(self, ctx, lean_code: str):
        """返回 True(编译失败⇒加固证伪) / False(编译通过⇒推翻本次证伪) / None(不可用⇒不加不减)。

        ⚠ `ok=False` 在 Lean 语义里是**正常结果**（编译报错），不是故障。
        """
        bridge = self._get_bridge()
        if bridge is None:
            return None
        try:
            work_dir = getattr(bridge, "_lean_project_dir", "") or ""
            if not work_dir:
                return None
            fname = "falsify_%d_%d.lean" % (os.getpid(), int(time.time() * 1000) % 1000000)
            r = None
            try:
                r = bridge._compile(lean_code, work_dir, lean_filename=fname,
                                    allow_sorry=False)
            finally:
                try:
                    from tools.lean_local.lean_bridge import _trash_lean_file
                    _trash_lean_file(work_dir, fname)
                except Exception:  # noqa: BLE001
                    pass
            if not isinstance(r, dict):
                return None
            if r.get("ok"):
                return False                      # 编译通过 ⇒ 推翻证伪
            err = str(r.get("error", ""))
            # 环境级错误（import 失败/找不到 Mathlib）不可信 ⇒ 不加不减
            if any(k in err for k in ("unknown module", "no such file",
                                      "failed to load", "cannot open")):
                return None
            return True                           # 编译失败 ⇒ 加固证伪
        except Exception:  # noqa: BLE001
            return None
