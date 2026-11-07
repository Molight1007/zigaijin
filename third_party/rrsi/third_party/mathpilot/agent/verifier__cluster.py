# -*- coding: utf-8 -*-
"""答案聚类簇：``AnswerCluster`` + 答案归一化 / 等价判定 / 共识聚类。

S2（2026-10-09，VerifierAgent 真语义重构）：把 VerifierAgent 的「答案聚类簇」
搬为**模块级函数**，原 ``self`` 作为第一个参数 ``owner`` 传入；函数体与原文
AST 完全一致（仅 ``self`` → ``owner`` 机械改写）。

S3（2026-10-09）：``AnswerCluster`` 亦从 ``agent/verifier.py:48`` **逐字搬**到本模块
（``__slots__`` / ``@property`` 只读语义与计算口径一字未改），verifier.py 原位 re-export。
此后本模块**不再 import ``agent.verifier``** ⇒ ``verifier ↔ verifier__cluster`` 模块级
循环被彻底消除（verifier.py 顶部可安全反向 import 本模块）。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Verdict


class AnswerCluster:
    """答案等价簇，用于跨候选多数投票。"""
    __slots__ = ("answer_norm", "candidate_ids", "vote_correct", "vote_total",
                 "n_distinct")

    def __init__(self, answer_norm: str):
        self.answer_norm = answer_norm
        self.candidate_ids: list[int] = []
        self.vote_correct: int = 0
        self.vote_total: int = 0
        # ★★ 2026-10-06 新增：**本题共有几种不同的答案**（证据独立性）。
        #   由投票方在统计时填；0 表示未知（保持旧行为）。
        self.n_distinct: int = 0

    @property
    def confidence(self) -> float:
        """置信度 = **独立证据支持度**（不是"票数一致率"）。

        ★★ 2026-10-06 修（**实测驱动：置信度被"同源复制"刷满**）：
        实测三题的最终置信度与候选分布：
            题 000：`\\boxed{0}` × **5（全部相同）** ⇒ 置信度 **1.00** ⇒ 直接放行（答案错）
            题 001：`-2(m-1)`×4 + 变体          ⇒ 置信度 0.00
            题 002：`(√5−1)/2`×4 + 变体        ⇒ 置信度 0.00
        ⇒ **候选越一致置信度越高** —— 但这在"采样没有多样性"时是**假证据**：
        5 个候选给出同一个答案，**不是 5 份独立同意，而是 1 份答案被复制了 5 遍**。
        ⇒ 原实现 `vote_correct / vote_total` 无法区分「独立同意」与「同源复制」。

        修法：引入**证据独立性折扣** —— 全题只有 1 种答案、**且有 ≥2 个候选**
        给出同一答案时（即"同源复制"），置信度**封顶 0.5**（低于 `accept_confidence`
        默认 0.6 ⇒ 不再直接放行，改由"零多样性补充采样"处置）。
        ⚠ **单候选时不折扣**：只有一个候选本来就没有多样性可言，此时票数反映的是
          "验证器自身对该候选的判断"，折扣它只会丢失信息
          （回归测试 `test_abstain_excluded_from_denominator` 锁定此语义）。
        `n_distinct <= 0` 表示未统计，**保持旧行为**（保守）。
        """
        if self.vote_total == 0:
            return 0.0
        base = self.vote_correct / self.vote_total
        if 0 < self.n_distinct <= 1 and len(self.candidate_ids) >= 2:
            return min(base, 0.5)
        return base

    @property
    def size(self) -> int:
        return len(self.candidate_ids)


def _normalize_answer_text(owner, text: str) -> str:
    """文本级归一化：去空白/去 $/浮点舍入/文本分数→数值/LaTeX 分数统一。"""
    if not text:
        return ""
    t = text.strip()
    t = t.replace("$", "").replace(" ", "")
    t = t.replace("\\displaystyle", "")
    t = t.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    # LaTeX 分数统一（与本地 _normalize_answer 一致，无括号便于数值解析）
    t = re.sub(r'\\frac\s*\{\s*([^}]*)\s*\}\s*\{\s*([^}]*)\s*\}', r'\1/\2', t)
    # 浮点舍入 6 位
    try:
        f = float(t)
        t = f"{f:.6g}"
    except (ValueError, TypeError):
        # 文本分数（如 1/2、(1)/(2)）→ 数值，统一 1/2 与 0.5、3 与 3.0
        stripped = re.sub(r'^\((-?\d+)\)/\((-?\d+)\)$', r'\1/\2', t)
        if re.fullmatch(r'-?\d+/\d+', stripped):
            try:
                num, den = stripped.split("/")
                f = int(num) / int(den)
                t = f"{f:.6g}"
            except (ValueError, ZeroDivisionError):
                pass
    return t


def _are_answers_equivalent(owner, a: str, b: str) -> bool:
    """三级等价判定：剥壳后文本相同 → 原文相同 → SymPy 符号等价。

        ★★ 2026-09-16 审计修复（与 `AnswerOracle` 那次同一病灶的第二份拷贝）：
        本方法与 `answer_oracle.answers_equivalent` 是**同一逻辑的两份实现**，
        而题目要求答案写成 `\\boxed{...}`。`_preprocess_latex` 会把 `\\boxed{2026}`
        归一成 `\\boxed2026` ⇒ 解析失败 ⇒ 判**不等价**。实测：
            verifier._are_answers_equivalent('\\boxed{2026}', '2026') -> **False**
            AnswerOracle.answers_equivalent ('\\boxed{2026}', '2026') -> True
        ⇒ verifier 自洽共识恒归不了组、playoff 复算对正确答案报"不一致" ⇒ 无谓 revise。
        修法：比较前先 `strip_wrappers`（与 answer_oracle 对齐口径），并保留原文兜底。
        """
    if not a or not b:
        return False
    try:
        from .answer_oracle import AnswerOracle
        ca, cb = AnswerOracle.strip_wrappers(a), AnswerOracle.strip_wrappers(b)
    except Exception:  # noqa: BLE001
        ca, cb = a.strip(), b.strip()
    # Level 1: 剥壳后字符串完全相同
    if ca and cb and ca == cb:
        return True
    # Level 2: 原文完全相同
    if a.strip() == b.strip():
        return True
    # Level 3: SymPy 符号等价（若可用）——剥壳版优先，原文兜底
    try:
        from utils.sympy_tools import are_expressions_equal
        for x, y in ((ca, cb), (a, b)):
            if x and y and are_expressions_equal(x, y):
                return True
    except ImportError:
        pass
    return False


def _equiv_group(
    owner, candidates: list, answers: list[str]
) -> list[list[int]]:
    """等价答案分组：返回候选 ID 列表的列表。（BUG-3 修复：实际使用返回值）"""
    n = len(answers)
    visited = [False] * n
    groups: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if visited[j]:
                continue
            if owner._are_answers_equivalent(answers[i], answers[j]):
                group.append(j)
                visited[j] = True
        groups.append(group)
    return groups


def _cluster_candidates(
    owner, candidates: list, verdicts: list[list[Verdict]]
) -> list[AnswerCluster]:
    """
        基于 Equivalent Answer 聚类 + 跨候选多数投票：
        1. 提取每个候选的答案（归一化）
        2. 等价分组
        3. 每个组统计"组内候选的总体正确票数 / 总票数"
        4. 返回簇列表，按置信度 × 规模排序
        """
    answers = []
    for cand in candidates:
        ans = (cand.get("answer", "") if isinstance(cand, dict)
               else getattr(cand, "answer", ""))
        answers.append(owner._normalize_answer_text(str(ans)))
    groups = owner._equiv_group(candidates, answers)

    clusters: list[AnswerCluster] = []
    # ★★ 2026-10-06 新增：统计**本题共有几种不同答案**（证据独立性），
    #   供 `AnswerCluster.confidence` 做折扣 —— 全题只有 1 种答案时，
    #   票数再多也只是"1 份证据被复制"，不能算高置信（详见 confidence 注释）。
    _n_distinct = len(groups)
    for g in groups:
        rep_idx = g[0]
        cluster = AnswerCluster(answers[rep_idx])
        cluster.n_distinct = _n_distinct
        for idx in g:
            cid = candidates[idx].get("id", idx) if isinstance(candidates[idx], dict) else idx
            cluster.candidate_ids.append(cid)
            # 统计该候选的所有票（2026-09-11：弃权票不计入分母 ——
            # 故障≠反证，见 _is_correct_vote 三态化说明）
            if idx < len(verdicts):
                for v in verdicts[idx]:
                    if getattr(v, "abstain", False):
                        continue
                    cluster.vote_total += 1
                    if v.correct:
                        cluster.vote_correct += 1
        clusters.append(cluster)

    # 排序：置信度 × 规模的加权（等价答案人越多且票越对 = 越可信）
    #
    # ★ 2026-09-22 修复（"删去原本不合理的逻辑"）：
    #   原键 `confidence * size + confidence` 在**所有簇置信度均为 0** 时
    #   对每簇都恰好等于 0 ⇒ `sort` 退化为**插入序**，即"随便挑一个"。
    #   实测 037：候选池里已有与 gold 一致的答案簇，但全簇置信度 0，
    #   最终按插入序落到了另一个簇上，正确答案被挤掉。
    #   改为**全序**键，保证任何情况下都有确定且合理的依据：
    #     置信度 → 得票率 → 正确票数 → 答案形态 → 簇规模
    #   ① 置信度非零时行为与原来完全一致（首键未变）⇒ 不扰动既有正确路径；
    #   ② 仅在"全零置信度"这一退化场景下，改用得票率/票数/形态裁决，
    #      簇规模**降到最后一键**，不再出现"0 票簇按规模压过有票簇"。
    def _ck(c):
        _vt = int(getattr(c, "vote_total", 0) or 0)
        _vc = int(getattr(c, "vote_correct", 0) or 0)
        _an = str(getattr(c, "answer_norm", "") or "")
        _form = 3 if "\\boxed" in _an else (2 if 0 < len(_an) <= 40 else (1 if _an else 0))
        return (float(getattr(c, "confidence", 0.0) or 0.0),
                (_vc / _vt) if _vt > 0 else 0.0,
                _vc, _form, int(getattr(c, "size", 0) or 0))

    clusters.sort(key=_ck, reverse=True)
    return clusters
