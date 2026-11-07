# -*- coding: utf-8 -*-
"""证据落地校验 —— 「LLM 指认的错误必须能在原文里找到」。

为什么需要这个模块（2026-10-22 五道链审计）：
4.1 自审与 4.2 对抗证伪的淘汰，全部建立在「模型说这里错了」之上，而模型给出的
「出错位置」从来不与候选原文核对。桩测试实测：让模型交出**原文并不存在**的一句话
作 ``error_at``，正确答案会被判 `logic_error=true` 直接淘汰；4.2 同理（编造的
``error_step`` 只要非空就触发 ``is_actionable``）。这是「把对的杀了」的主通道。

判据的性质（**不是降级通道**）：
本模块判定的是「这条证据是否落在原文里」，是一个**可机器复核的事实**，不是
「判不出来就放行」。落地失败 ⇒ 证据不成立 ⇒ 不得据此淘汰；同时必须留痕
（``ungrounded`` 计数 + 明细），并在选择层体现为「未验证」。

依赖：仅标准库 re（分层约束 contracts←utils←tools←agent←根）。
"""
from __future__ import annotations

import re

# 归一化：空白、常见全角标点、LaTeX 壳一律剥掉后再比对
_WS_RE = re.compile(r"[\s\u3000\xa0]+")
_PUNCT_RE = re.compile(r"[，。；：、！？（）【】「」『』《》〈〉“”‘’\"'`,;:!?()\[\]{}]")
_LATEX_WRAP = (("$$", ""), ("$", ""), ("\\(", ""), ("\\)", ""),
               ("\\[", ""), ("\\]", ""), ("\\text", ""), ("\\mathrm", ""))

# 纯「步骤序号」形态：不是原文片段，无法做落地校验（提示词已改为要求原文片段，
# 这里保留兼容旧输出，判为「不可校验」而不是「已落地」）
_STEP_REF_RE = re.compile(
    r"^(?:第?\s*[\d一二三四五六七八九十]+\s*步?|步骤\s*[\d一二三四五六七八九十]+"
    r"|step\s*\d+|\d+\s*[.、)）]?)$", re.I)

# 引文分段：省略号/分号把「多处摘录」拆开，逐段核对
# ★ 单个 `…` / `⋯` 也算省略（旧写法要求连续 2 个，实测漏拆导致整段核对失败）
_SEG_SPLIT_RE = re.compile(r"[…⋯]+|\.{2,}|；|;")


def norm_text(s) -> str:
    """归一化到「可比对」形态：去空白、去标点、剥 LaTeX 壳、统一小写。"""
    t = str(s or "")
    for _a, _b in _LATEX_WRAP:
        t = t.replace(_a, _b)
    t = _WS_RE.sub("", t)
    t = _PUNCT_RE.sub("", t)
    return t.lower()


def is_step_ref(s) -> bool:
    """是否只是「第二步 / step 3」这类位置编号（无原文可核对）。"""
    return bool(_STEP_REF_RE.match(str(s or "").strip()))


def check_quote(quote, source, min_ratio: float = 0.6,
                min_len: int = 6, min_seg: int = 4) -> tuple:
    """引文是否真的落在原文里。

    Returns:
        ``(verdict, ratio, missing)``
        verdict: ``True`` 已落地 / ``False`` 未落地 / ``None`` 不可判（太短、空）
        ratio:   分段命中率（0.0~1.0）
        missing: 未命中的分段（最多 3 条，供留痕）

    判定口径：
      1. 归一化后整段是原文子串 ⇒ 已落地（ratio=1.0）；
      2. 否则按省略号/分号切段，丢弃 < ``min_seg`` 的碎段，命中率 ≥ ``min_ratio``
         ⇒ 已落地（允许模型省略中间部分）；
      3. 归一化长度 < ``min_len`` 或切完没有有效段 ⇒ 不可判（``None``）。

    ★ 不设「模糊相似度」——引文核对必须是「摘录」而不是「大意」，否则幻觉照样通过。
    """
    q = norm_text(quote)
    src = norm_text(source)
    if not q or len(q) < min_len:
        return None, 0.0, []
    if not src:
        return False, 0.0, [str(quote)[:40]]
    if q in src:
        return True, 1.0, []
    segs = [norm_text(x) for x in _SEG_SPLIT_RE.split(str(quote or ""))]
    segs = [x for x in segs if len(x) >= min_seg]
    if not segs:
        return None, 0.0, []
    miss = [x for x in segs if x not in src]
    ratio = (len(segs) - len(miss)) / float(len(segs))
    return (ratio >= min_ratio), ratio, miss[:3]


def check_location(loc, source) -> tuple:
    """校验「错误位置」：返回 ``(verdict, ratio, note)``。

    比 :func:`check_quote` 多一层语义：
      · 空串            ⇒ ``(None, 0.0, "missing")``  —— 没给位置，证据不成立
      · 纯步骤序号      ⇒ ``(None, 0.0, "step_ref")`` —— 无法核对，证据不成立
      · 其余按引文核对  ⇒ ``(True/False, ratio, ...)``
    """
    s = str(loc or "").strip()
    if not s:
        return None, 0.0, "missing"
    if is_step_ref(s):
        return None, 0.0, "step_ref"
    ok, ratio, miss = check_quote(s, source)
    return ok, ratio, ("ok" if ok else "not_found:%s" % "|".join(miss[:2]))
