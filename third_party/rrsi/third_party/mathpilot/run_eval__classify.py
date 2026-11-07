# -*- coding: utf-8 -*-
"""run_eval 判分族之「错误分类」—— 2026-10-08 从 run_eval.py 逐字搬出。

`_classify_error` 及其专用常量（原 L604-626 / L933-986）。判决逻辑零变更；
复用 run_eval__scoring 的 `_norm_candidate` / `_strip_latex_cmds`（同一函数对象）。
"""
from __future__ import annotations

import re

from run_eval__scoring import _norm_candidate, _strip_latex_cmds


# ---------------------------------------------------------------------------
# 错误分类（老师要求 #6「先定位瓶颈」/#11「错误分推理性与非推理性」/#16「甄别错误来源」的落地）
# ---------------------------------------------------------------------------
# 判错样本分四类，用于决定优化资源投向：
#   empty_output       —— 空输出或只剩 LaTeX 定界符：解析/截断 bug，0 成本可修
#   format_unresolved  —— 答案未定型（含未求值符号或条件式），低成本可修
#   value_wrong        —— 两边都是裸数却不等：真算错，只能靠推理能力提升
#   expr_wrong         —— 表达式错：推理能力
#
# 注意：前两类不需要提升推理能力就能捡回来，是性价比最高的提分点。
# ---------------------------------------------------------------------------

# 只剩空白或 LaTeX/数学定界符（$$、\[、()、{}、标点）
_EMPTY_DELIM = re.compile(r'^[\s\$\\!\[\]\(\)\{\}\.,;：:、，。\*]*$')
# 纯数字四则式（允许千分位逗号、括号、除号）
_NUMERIC = re.compile(r'^[-+]?[\d\.\,/\(\)\s\+\-\*]+$')
# 未求值信号：条件式、不等式、量词
_UNRESOLVED_SYM = re.compile(
    r'[<>]|\\(?:geq|leq|ge\b|le\b)|存在|任意|所有|当且仅当|恒成立')
# 答案抽取失败信号：预测里混进了推理过程/步骤文本
_STEP_MARK = re.compile(r'步骤\s*\d|Step\s*\d|解\s*[：:]|综上|由此可知|由上述')
# Markdown 标题行（如 "## 最终答案"）——说明只输出了标题没输出答案
_MD_HEADER = re.compile(r'^#+\s*')


def _classify_error(pred: str, gold: str) -> str:
    """对判错的样本分类，定位瓶颈（#6 / #11 / #16）。

    仅在 is_correct 为 False 时调用，返回五类中的一类：
      empty_output      空输出 / 只剩定界符
      extract_failed    答案抽取失败（混进推理过程、LaTeX 截断、只吐标题）
      format_unresolved 答案未定型（含未求值符号或条件式）
      value_wrong       两边都是裸数却不等 → 真算错
      expr_wrong        表达式错 → 推理能力

    设计原则：判定必须保守——宁可归到 expr_wrong，也不要把真算错判成
    format_unresolved（后者会让人误以为"改改格式就能提分"）。
    """
    p = (pred or "").strip()
    g = (gold or "").strip()
    # 裸 `\boxed` / `\boxed{}`（无内容）应判为空输出（official112-011 实况：
    # pred='\boxed' 被误归 expr_wrong，掩盖了"答案未产出"这一事实）
    p_probe = re.sub(r"\\boxed\b", "", p).strip()

    # 1) 空输出 / 只剩定界符 → 解析或截断 bug
    if not p_probe or _EMPTY_DELIM.match(p_probe):
        return "empty_output"
    pf = _norm_candidate(p)
    gf = _norm_candidate(g)
    if not pf or _EMPTY_DELIM.match(pf):
        return "empty_output"

    # 2) 答案抽取失败：混入步骤文本、LaTeX 定界符不配对（截断）、只吐 Markdown 标题
    if _STEP_MARK.search(p) or p.count("$") % 2 == 1:
        return "extract_failed"
    if _MD_HEADER.match(p):
        rest = _MD_HEADER.sub("", p).strip()
        # 标题后没有内容，或只有一个"答案"字样（如 "## 最终答案"）→ 没吐出答案
        if not rest or re.fullmatch(r'(最终答案|答案|解答|解|Answer|ANSWER)[：:]?', rest):
            return "extract_failed"
    if len(p) > 120 and "\\boxed" not in p:
        return "extract_failed"

    # 3) 两边都是裸数（或纯数字四则式）却不等 → 数值真算错
    if _NUMERIC.match(pf) and _NUMERIC.match(gf):
        return "value_wrong"

    # 4) 预测含未求值符号 / 条件式 / 量词 → 答案未定型
    if _UNRESOLVED_SYM.search(pf):
        return "format_unresolved"
    # 去掉 LaTeX 命令后仍残留字母（自由变量），而 gold 是纯数值 → 未完全求值
    residual_p = _strip_latex_cmds(pf)
    residual_g = _strip_latex_cmds(gf)
    if (re.search(r'[A-Za-z]', residual_p)
            and not re.search(r'[A-Za-z]', residual_g)):
        return "format_unresolved"

    # 5) 其余 → 表达式/推理错
    return "expr_wrong"
