from __future__ import annotations
"""统一客观答案验证层（AnswerOracle，v2.7；2026-09-06 去 Lean 化）。

设计背景
========
当前"写不对难题"的根因之一是：验证器（Verifier）与解题器（Solver）同源，
A/B 投票本质是"让同一个模型再读一遍"，会一起错；而客观工具没有形成闭环。

本模块把"客观答案验证"从 verifier 中抽离为统一入口。
⚠ **2026-09-23 更正**：本节此前写"2026-09-06 去 Lean 化：平台无 Lean 可执行文件" ——
**该前提已过时**：0923 云端实测 `lean_executable` 存在、`lean_mcp` 332 次调用 0 失败。
但本轮 46/46 题均为**解答题**，所以"证明题走 Lean"的分支从未被触发，
故本模块当前的**实际缺口不在 Lean**，而在：
- 只有"多候选自洽共识"（self-consistency），**没有 reference ⇒ 判不出绝对对错**
  （0923 效能审计实测该环节召回仅 **3%**）。
- 补法见 `agent/answer_falsifier.py`（**只做证伪**：数值回带 → SymPy 精确判定 → Lean 背书），
  已由 `orchestrator._oracle_review_best` 与本模块**叠加调用**。

本模块保留：
- 计算/数值题 → SymPy 符号等价（多候选 self-consistency 聚类）+ 答案可解析性；
- 证明题 → 不编译不误判，直接返回 unknown（散文证明不可程序化等价判定），
  由 AuditGate（rubric 结构化判分）与对抗式验证承担客观把关。

客观验证不依赖"验证器与解题器同源"的 LLM 自评，是数学领域区别于通用
LLM 编排的最大增量（对应 LangGraph 的 oracle-in-the-loop 思想）。

隔离原则
========
- 独立文件，不污染 orchestrator 主流程；上层只调用
  ``AnswerOracle(client, config, budget).verify(ctx, candidate)`` 一行；
- 任何异常一律吞掉并降级 ``verdict='unknown'``，绝不因 oracle 导致评测崩溃；
- SymPy 不可用时仅打 warning 并整体降级 unknown，不阻断主流程。

对外契约
========
``verify()`` 返回 ``OracleResult``：
- verdict: 'correct' | 'incorrect' | 'unknown'
- feedback: 结构化错误定位（供 revise / 审查复用）
- evidence: 客观证据（sympy 重算值 / 共识统计）
- oracle_type: 'sympy' | 'none'
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .base import TaskContext

logger = logging.getLogger("MathPilot.Oracle")

# 证明题判定信号（与 lean_gate / solver 保持一致）
_PROOF_TYPE = "证明题"
_PROOF_DOMAINS = ("证明", "证明题")


@dataclass
class OracleResult:
    """客观验证结果（单一数据契约，JSON 可序列化）。"""
    verdict: str = "unknown"            # 'correct' | 'incorrect' | 'unknown'
    feedback: str = ""                  # 结构化错误定位（供 revise / 审查复用）
    evidence: dict = field(default_factory=dict)  # 客观证据
    oracle_type: str = "none"           # 'sympy' | 'none'（'lean' 已于 2026-09-06 移除）

    @property
    def is_correct(self) -> bool:
        return self.verdict == "correct"

    @property
    def is_incorrect(self) -> bool:
        return self.verdict == "incorrect"

    def to_dict(self) -> dict:
        """JSON 可序列化表示（供 trace / 诊断 / 评测报告复用）。"""
        return {
            "verdict": self.verdict,
            "feedback": self.feedback,
            "evidence": self.evidence,
            "oracle_type": self.oracle_type,
        }


class AnswerOracle:
    """统一客观答案验证层。按题型分流：证明题 unknown，计算题走 SymPy。"""

    name = "AnswerOracle"

    def __init__(self, client=None, config=None, budget=None):
        self.client = client
        self.config = config
        self.budget = budget

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def verify(self, ctx: TaskContext, candidate,
               question_type: Optional[str] = None,
               candidates: Optional[list] = None) -> OracleResult:
        """按题型分流验证单个候选解答。

        参数:
            ctx: 共享上下文（含 problem / domain / question_type）。
            candidate: 待验证候选（含 answer / reasoning）。
            question_type: 题型（证明题/选择题/判断题/填空题/解答题），缺省取 ctx。
            candidates: 全部候选（计算题做多候选符号等价 self-consistency 用）。

        返回:
            OracleResult（verdict / feedback / evidence）。

        2026-09-06 去 Lean 化：证明题不再走 LeanBridge 编译（平台无 Lean），
        散文证明无法程序化等价判定 → 直接 unknown 放行，避免误把证明文本
        当"无法解析的表达式"判 incorrect（会把每题证明都误触发 revise）。
        客观把关由 AuditGate（6.5 rubric）+ 对抗式验证（4.6）承担。
        """
        qt = question_type or getattr(ctx, "question_type", "") or ""
        domain = getattr(ctx, "domain", "") or ""
        if qt == _PROOF_TYPE or any(k in domain for k in _PROOF_DOMAINS):
            return OracleResult(verdict="unknown", oracle_type="none",
                                evidence={"reason": "proof_not_computational"})
        return self.verify_computational(candidate, candidates)

    # ------------------------------------------------------------------
    # 计算题：SymPy 客观验证
    # ------------------------------------------------------------------
    def verify_computational(self, candidate,
                             candidates: Optional[list] = None) -> OracleResult:
        """计算题 → SymPy 符号等价 + 答案可解析性。

        解题阶段无 reference_answer，无法对单候选做绝对对错判定，因此：
        - 多候选符号等价（self-consistency）给出"多数一致"客观信号；
        - 答案可解析性过滤明显非法答案；
        - 其余一律 unknown，交由上层投票/playoff 收敛，避免假阳性。
        """
        answer = getattr(candidate, "answer", "") or ""
        parseable = self.is_parseable(answer)
        group_size, total, usable = self._consensus(candidate, candidates)

        evidence = {
            "parseable": parseable,
            "group_size": group_size,
            "total": total if usable else None,
        }

        if not answer:
            return OracleResult(verdict="incorrect", feedback="候选答案为空",
                                evidence=evidence, oracle_type="sympy")
        if not parseable:
            return OracleResult(verdict="incorrect",
                                feedback="候选答案无法解析为有效数学表达式",
                                evidence=evidence, oracle_type="sympy")
        if usable and total >= 2 and group_size >= max(2, total // 2 + 1):
            # 多数候选符号一致 → self-consistency 强信号，视为可信
            return OracleResult(verdict="correct", oracle_type="sympy",
                                evidence=evidence)
        if usable and total >= 2 and group_size == 1:
            # 与其他候选均不等价 → 孤立答案，弱信号（供上层压低置信度）
            return OracleResult(verdict="unknown", oracle_type="sympy",
                                feedback="该候选答案与其他候选符号不等价",
                                evidence=evidence)
        return OracleResult(verdict="unknown", oracle_type="sympy",
                            evidence=evidence)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ★★ 2026-09-16 修复：`\boxed{...}` 外壳导致解析/等价**全面失效**
    # ------------------------------------------------------------------
    # 实测（0916 轮 003/013 两题同时命中）：
    #   is_parseable('\boxed{2025}')            -> False   （而 '2025' -> True）
    #   is_parseable('\boxed{50}')              -> False
    #   answers_equivalent('\boxed{2025}','2025')-> False
    # 底层原因：`utils/sympy_tools._try_parse` 的前处理把 `\boxed{2025}` 归一成
    #   `\boxed2025`（删了花括号、留下 `\boxed`）⇒ SymPy `could not parse`。
    # **而题目明确要求「put your final answer within \boxed{}」** —— 模型照做了，
    # 却被判"无法解析"。后果是**两个机制同时废掉**：
    #   ① 可解析性闸门 → 假判 incorrect → trace 记「客观复核判错，触发定向修正」
    #      ⇒ 触发长达 10 分钟的徒劳修正（003 实耗 **775.9s** / 013 **591.4s**）；
    #   ② `answers_equivalent` 归组失效 ⇒ 自洽共识（group_size）永远是 1 ⇒ 信号无用。
    # 修法：在 oracle 内部先**剥掉外表壳**再解析/比较（不动 `sympy_tools` 全局行为，
    # 避免影响 `run_eval.answers_match` 等其它调用方的既有语义）。
    # ------------------------------------------------------------------
    _BOXED_RE = re.compile(r"\\(?:boxed|fbox|overline)\s*\{")
    _TEXT_RE = re.compile(r"\\text\s*\{")

    @classmethod
    def strip_wrappers(cls, answer: str) -> str:
        """剥掉答案外部包装：`\\boxed{}`/`\\fbox{}`/`\\text{}`、`$`、`\\( \\)`、`\\[ \\]`。

        只剥**最外层**的包裹，不改变内部数学内容。用于解析与等价比较前的归一。
        """
        s = (answer or "").strip()
        # 数学定界符
        for a, b in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")):
            if s.startswith(a) and s.endswith(b) and len(s) > len(a) + len(b):
                s = s[len(a):-len(b)].strip()
        # \boxed{...} / \text{...}：用花括号配平取内层，可多层
        for _ in range(4):
            m = (cls._BOXED_RE.match(s) or cls._TEXT_RE.match(s))
            if not m:
                break
            depth, end = 0, -1
            for i, ch in enumerate(s[m.end() - 1:], start=m.end() - 1):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end < 0:
                break
            s = s[m.end():end].strip()
        return s.strip()

    @staticmethod
    def is_parseable(answer: str) -> bool:
        """答案能否被 SymPy 解析为有效表达式（纯本地，不消耗 LLM 预算）。

        ⚠ 解析前**必须** `strip_wrappers`：否则 `\\boxed{...}`（题面要求的书写格式）
        会被判为"无法解析"（2026-09-16 实测，详见类内注释）。
        """
        if not answer:
            return False
        try:
            from utils.sympy_tools import _try_parse
            for cand in (AnswerOracle.strip_wrappers(answer), answer):
                parsed, _ = _try_parse(cand)
                if parsed is not None:
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _consensus(candidate, candidates: Optional[list]) -> tuple[int, int, bool]:
        """计算候选答案在候选集中的符号等价组大小。

        返回 (组大小, 总候选数, 是否可用多候选共识)。
        """
        if not candidates:
            return 1, 1, False
        target = getattr(candidate, "answer", "") or ""
        total = len(candidates)
        group = 0
        for c in candidates:
            if AnswerOracle.answers_equivalent(target, getattr(c, "answer", "")):
                group += 1
        return group, total, True

    @staticmethod
    def answers_equivalent(a: str, b: str) -> bool:
        """两级答案等价：文本完全相同 → SymPy 符号等价。

        ⚠ 比较前**必须** `strip_wrappers`：否则 `\\boxed{2025}` 与 `2025` 会判为
        不等价 ⇒ **自洽共识永远归不了组**（group_size 恒为 1，信号作废）。
        2026-09-16 实测，详见 `is_parseable` 上方注释。
        """
        if not a or not b:
            return False
        if a.strip() == b.strip():
            return True
        ca, cb = AnswerOracle.strip_wrappers(a), AnswerOracle.strip_wrappers(b)
        if ca and cb and ca == cb:
            return True
        try:
            from utils.sympy_tools import are_expressions_equal
            for x, y in ((ca, cb), (a, b)):
                if x and y and are_expressions_equal(x, y):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def findings_to_feedback(findings) -> str:
        """把 Finding 列表结构化为可注入 revise 的错误定位文本。

        格式：`- {location} [{kind}](严重度{severity}): {desc}`
        2026-09-06 起主链路不再产生 Lean Finding（去 Lean 化）；本方法保留
        供本地 Lean 证据链工具（lean_gate 迁移 tools/lean_local 前）复用。
        """
        if not findings:
            return ""
        lines = []
        for f in findings:
            loc = getattr(f, "location", "") or "未定位"
            kind = getattr(f, "kind", "") or ""
            desc = getattr(f, "desc", "") or ""
            sev = getattr(f, "severity", 0) or 0
            tag = f"[{kind}]" if kind else ""
            sev_tag = f"(严重度{sev})" if sev else ""
            lines.append(f"- {loc} {tag}{sev_tag}: {desc}".strip())
        return "\n".join(lines)
