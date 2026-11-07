# -*- coding: utf-8 -*-
"""引理（lemma）累积上下文注入（2026-10-08 拆分：从 agent/solver.py 整段同迁）。

  · ``_use_lemma`` —— 按领域路由判断当前题是否启用 lemma 累积；
  · ``_collect_lemma_context`` —— 收集已验证的子结论（引理库），作为解题上下文注入。

搬运方式：**整段同迁**（``def`` 行 + 函数体逐字搬，仅去 4 格缩进）。``solver.py``
原地保留同名**一行委托**，对外签名与返回值逐字不变。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # 仅类型标注用
    from .base import TaskContext


def _use_lemma(self, ctx: TaskContext) -> bool:
    """判断当前题是否启用 lemma 累积（按领域路由，2026-08-29）。

        A/B 实测：lemma 全领域开 = 净 0.0pp（代数/组合被噪声拖累，数论 +23pp）。
        因此默认按领域路由：lemma_domains 命中才注入，把数论的收益变成
        确定收益，同时避免拖累其他领域。
        """
    if not getattr(self.config, 'use_lemma_accumulation', False):
        return False
    domains = list(getattr(self.config, 'lemma_domains', []) or [])
    if not domains:
        return True  # 空列表 = 全领域开启
    d = str(getattr(ctx, 'domain', '') or '')
    return any(k in d for k in domains)


def _collect_lemma_context(self, ctx: TaskContext) -> str:
    """收集已验证的子结论（引理库），作为解题上下文注入。"""
    if not self._use_lemma(ctx):
        return ""
    lemmas = getattr(ctx, 'lemma_repo', [])
    if not lemmas:
        return ""
    recent = lemmas[-5:]  # 最多注入 5 条
    return "【已验证的中间结论】\n" + "\n".join(f"- {l}" for l in recent) + "\n\n"
