# -*- coding: utf-8 -*-
"""LeanSearch 引理检索注入（从 agent/verifier.py 逐字搬出）。

S2（2026-10-09）：``_prepare_theorem_context`` 搬为模块级函数，原 ``self``
作为第一个参数 ``owner`` 传入。注意：``owner._ls_searcher`` /
``owner._ls_status_logged`` 仍是**宿主角上的懒建属性**（本模块不持有状态）。
S3（2026-10-09）：``verifier.py`` 顶部已 `from . import verifier__lemma as _vl`，
``VerifierAgent._prepare_theorem_context`` 改为一行委托跳板。
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("MathPilot.Verifier")


def _prepare_theorem_context(owner, ctx, problem: str) -> str:
    """检索与本题相关的 Mathlib 定理原文；结果缓存在 ctx 上（一题一次）。"""
    cfg = owner.config
    if not getattr(cfg, "use_leansearch", False):
        return ""
    if not getattr(cfg, "leansearch_inject_verifier", True):
        return ""
    cached = getattr(ctx, "_leansearch_ctx", None)
    if cached is not None:
        return cached

    out = ""
    try:
        if ctx.is_time_critical():
            owner.record(ctx, "leansearch", "时间紧张，跳过引理检索（注入为空）")
        else:
            # 延迟导入：模块缺失时只影响本功能，不拖垮验证器
            from tools.lean_local.lean_search import MathlibTheoremSearcher
            searcher = getattr(owner, "_ls_searcher", None)
            if searcher is None:
                searcher = MathlibTheoremSearcher()
                owner._ls_searcher = searcher
            limit = max(1, int(getattr(cfg, "leansearch_top_k", 5) or 5))
            max_calls = int(getattr(cfg, "leansearch_max_calls_per_q", 2) or 0)
            n_called = int(getattr(ctx, "_leansearch_calls", 0) or 0)
            if max_calls and n_called >= max_calls:
                owner.record(ctx, "leansearch",
                            "已达单题检索次数上限（%d），跳过" % max_calls)
            else:
                ctx._leansearch_calls = n_called + 1
                owner.note_mathlib_search(ctx)
                _t0 = time.perf_counter()
                res = searcher.search(problem[:800], limit=limit) or {}
                _ms = (time.perf_counter() - _t0) * 1000.0
                items = res.get("results") or []
                # 首次调用播报后端状态（启动门禁）：让"检索到底走没走通"
                # 一眼可见，而不是又一次静默（2026-09-15 教训）。
                if not getattr(owner, "_ls_status_logged", False):
                    owner._ls_status_logged = True
                    try:
                        st = searcher.status()
                        logger.info(
                            "[LeanSearch] 后端状态: available=%s root=%s "
                            "indexed=%s official=%s",
                            st.get("available"), str(st.get("root"))[:90],
                            st.get("indexed_declarations"), st.get("official"))
                    except Exception:  # noqa: BLE001
                        pass
                if items:
                    out = "\n".join(
                        "- %s (%s): %s" % (
                            r.get("name", ""), r.get("kind", "?"),
                            (r.get("snippet", "") or "").replace("\n", " ")[:200])
                        for r in items)
                    owner.add_used_theorems(
                        ctx, [r.get("name", "") for r in items])
                    owner.record(
                        ctx, "leansearch",
                        "检索到 %d 条相关定理（root=%s, %.0fms）" % (
                            len(items), str(res.get("root", ""))[:80], _ms),
                        n_hits=len(items), elapsed_ms=round(_ms, 1),
                        root=str(res.get("root", ""))[:120],
                        names=[r.get("name", "") for r in items])
                else:
                    owner.record(ctx, "leansearch",
                                "未检索到相关定理（注入为空）",
                                n_hits=0, elapsed_ms=round(_ms, 1))
    except ImportError as e:  # noqa: BLE001
        logger.warning(
            "LeanSearch 不可用（tools.lean_local.lean_search 导入失败）：%s "
            "—— 本次验证注入为空，行为与未启用时一致", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("LeanSearch 引理检索失败（注入为空）：%s", e)

    ctx._leansearch_ctx = out
    return out
