# -*- coding: utf-8 -*-
"""通用联网搜索工具（2026-09-16 新增，用户要求「补充联网搜索工具」）。

为什么需要
----------
此前仓库里**没有任何通用联网检索能力**：只有 `lean_search.py` 打 LeanSearch
（数学定理专用）。用户要求"联网搜索加使用全部工具"。

后端选择（2026-09-16 实测，见下方 `_BACKENDS` 注释）
--------------------------------------------------
本机走沙箱代理（`HTTP_PROXY/HTTPS_PROXY`），实测各后端连通性：
  · **Bing**            HTTP 200 ✓（本次采用为主后端）
  · **arXiv API**       HTTP 200 ✓（数学/CS 论文专用，作为数学场景补充）
  · DuckDuckGo HTML     ProxyError ✗
  · Wikipedia API       ProxyError ✗
  · Google              ProxyError ✗
⇒ 主后端 = Bing HTML 解析；补充后端 = arXiv API。

用法
----
    from tools.web_search import web_search, get_stats
    r = web_search("Cauchy Schwarz inequality proof", limit=5)
    # -> {"status":"ok","query":...,"backend":"bing","results":[{title,url,snippet}]}

设计要点
--------
- **不抛异常**：任何失败都返回 `{"status":"error","error":...}`，绝不打断主流程；
- **超时可控**：默认 20s（网络慢时不至于拖死单题）；
- **可观测**：每次调用记入 `WebSearchStats`（query/backend/条数/耗时/成败），
  供 `run_eval` 侧做逐题工具埋点（与 `lean_mcp` 埋点同一口径）；
- **只读**：仅做 GET 查询，不提交任何表单、不写远端状态。
"""
from __future__ import annotations

import html
import logging
import re
import threading
import time
import urllib.parse

logger = logging.getLogger("MathPilot.WebSearch")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# 实测可用性与**质量**（2026-09-16）：
#   ✓ **Math StackExchange API**（api.stackexchange.com）：数学题最佳——
#     "domino tiling recurrence" 命中「3×2n 矩形多米诺铺法」等强相关帖；
#   ✓ **arXiv API**：论文检索，"lean 4 tactic automation" 命中 Lean 4 形式化论文；
#   △ **Bing**：能连通**但结果不可用**——经沙箱代理时对
#     "domino tiling recurrence" 返回「多米诺披萨官网」、
#     对 ensearch 查询返回「Hotmail 登录 / Exchange 安全更新」等无关页。
#     ⇒ 降为**最后兜底**，并假定其质量不可靠。
#   ✗ DuckDuckGo / Wikipedia / Google / math.stackexchange.com 直连：ProxyError / 403
_BACKENDS = ("mathse", "arxiv", "bing")

_DEFAULT_TIMEOUT = 20.0
_MAX_SNIPPET = 320


# ---------------------------------------------------------------------------
# 调用统计（与 lean_bridge.mcp_stats 同口径，便于 run_eval 做逐题增量）
# ---------------------------------------------------------------------------
_STATS = {"calls": 0, "ok": 0, "fail": 0, "seconds": 0.0, "results": 0}
_STATS_LOCK = threading.Lock()


def web_search_stats() -> dict:
    """累计联网搜索统计（calls / ok / fail / seconds / results）。"""
    with _STATS_LOCK:
        return dict(_STATS)


def _note(seconds: float, ok: bool, n_results: int = 0) -> None:
    with _STATS_LOCK:
        _STATS["calls"] += 1
        _STATS["seconds"] += float(seconds)
        _STATS["ok" if ok else "fail"] += 1
        _STATS["results"] += int(n_results)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    """去标签、解实体、压空白。"""
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = html.unescape(t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _get(url: str, timeout: float = _DEFAULT_TIMEOUT):
    """统一的 GET 封装（延迟 import requests，缺依赖时优雅失败）。"""
    import requests  # noqa: PLC0415
    return requests.get(url, headers={"User-Agent": _UA,
                                      "Accept-Language": "en-US,en;q=0.9"},
                        timeout=timeout)


# ---------------------------------------------------------------------------
# 后端 1：Bing
# ---------------------------------------------------------------------------
_BING_ITEM_RE = re.compile(r'<li class="b_algo".*?</li>', re.S)
_BING_TITLE_RE = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_BING_SNIP_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S)


def _search_bing(query: str, limit: int, timeout: float) -> list:
    # `setmkt=en-US&setlang=en` 强制英文结果：数学题多为英文术语，
    # 不加的话中文市场会把 "Cauchy Schwarz" 返回成"柯西生平"（实测）。
    url = ("https://www.bing.com/search?q="
           + urllib.parse.quote(query)
           + "&setmkt=en-US&setlang=en&count=%d" % max(1, min(limit, 20)))
    resp = _get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError("bing HTTP %s" % resp.status_code)
    out = []
    for block in _BING_ITEM_RE.findall(resp.text):
        m = _BING_TITLE_RE.search(block)
        if not m:
            continue
        sn = _BING_SNIP_RE.search(block)
        out.append({
            "title": _clean(m.group(2))[:200],
            "url": m.group(1),
            "snippet": _clean(sn.group(1))[:_MAX_SNIPPET] if sn else "",
        })
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# 后端 1：Math StackExchange API（数学专用，免密钥，质量最佳）
# ---------------------------------------------------------------------------
def _search_mathse(query: str, limit: int, timeout: float) -> list:
    """math.stackexchange.com 官方 API。

    `filter=withbody` 才能拿到正文；否则只有标题/链接，模型无法据此推理。
    实践观察：过窄的查询（如 "splitting field x^4+5"）会 0 命中，属正常。
    """
    params = {
        "order": "desc", "sort": "relevance", "q": query,
        "site": "math", "pagesize": max(1, min(limit, 20)),
        "filter": "withbody",
    }
    resp = _get("https://api.stackexchange.com/2.3/search/advanced?"
                + urllib.parse.urlencode(params), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError("mathse HTTP %s" % resp.status_code)
    import json as _json
    data = _json.loads(resp.text)
    out = []
    for it in (data.get("items") or []):
        body = _clean(it.get("body") or "")
        out.append({
            "title": html.unescape(it.get("title") or "")[:200],
            "url": it.get("link") or "",
            "snippet": body[:_MAX_SNIPPET],
            "score": it.get("score"),
            "answered": bool(it.get("is_answered")),
        })
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# 后端 2：arXiv API（数学/CS 论文，免密钥）
# ---------------------------------------------------------------------------
_ARXIV_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.S)


def _search_arxiv(query: str, limit: int, timeout: float) -> list:
    url = ("http://export.arxiv.org/api/query?search_query=all:"
           + urllib.parse.quote(query)
           + "&start=0&max_results=%d" % max(1, min(limit, 20)))
    resp = _get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError("arxiv HTTP %s" % resp.status_code)
    out = []
    for ent in _ARXIV_ENTRY_RE.findall(resp.text):
        t = re.search(r"<title>(.*?)</title>", ent, re.S)
        l = re.search(r"<id>(.*?)</id>", ent, re.S)
        s = re.search(r"<summary>(.*?)</summary>", ent, re.S)
        if not (t and l):
            continue
        out.append({
            "title": _clean(t.group(1))[:200],
            "url": _clean(l.group(1)),
            "snippet": _clean(s.group(1))[:_MAX_SNIPPET] if s else "",
        })
        if len(out) >= limit:
            break
    return out


_BACKEND_FN = {"mathse": _search_mathse, "arxiv": _search_arxiv,
               "bing": _search_bing}


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def web_search(query: str, limit: int = 5, backend: str = "auto",
               timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """联网搜索。**永不抛异常**（失败返回 status=error）。

    backend: "auto"（默认，按 _BACKENDS 顺序回退）| "bing" | "arxiv"
    返回: {"status": "ok"|"error", "query", "backend", "results":[...], ...}
    """
    q = (query or "").strip()
    if not q:
        return {"status": "error", "query": q, "backend": "",
                "results": [], "error": "empty_query"}
    order = list(_BACKENDS) if backend in ("auto", "", None) else [backend]
    t0 = time.monotonic()
    errs = []
    for name in order:
        fn = _BACKEND_FN.get(name)
        if fn is None:
            errs.append("%s:unknown_backend" % name)
            continue
        try:
            hits = fn(q, limit, timeout)
            if hits:
                el = time.monotonic() - t0
                _note(el, True, len(hits))
                logger.info("[web_search] %s 命中 %d 条（%.1fs）",
                            name, len(hits), el)
                return {"status": "ok", "query": q, "backend": name,
                        "results": hits, "seconds": round(el, 2)}
            errs.append("%s:empty" % name)
        except Exception as e:  # noqa: BLE001
            errs.append("%s:%s" % (name, type(e).__name__))
            logger.info("[web_search] %s 失败：%s", name, str(e)[:120])
    el = time.monotonic() - t0
    _note(el, False, 0)
    return {"status": "error", "query": q, "backend": "",
            "results": [], "error": ";".join(errs) or "all_backends_failed",
            "seconds": round(el, 2)}


def web_search_block(query: str, limit: int = 4, max_chars: int = 900) -> str:
    """把搜索结果渲染成可注入提示词的文本块；无结果返回空串（零噪音）。

    与 `prompts/error_lessons.py::error_lessons_block` 同一约定：
    **无命中返回空串**，调用方直接拼接即可，不必判空。
    """
    r = web_search(query, limit=limit)
    if r.get("status") != "ok" or not r.get("results"):
        return ""
    lines = ["\n\n**【联网检索参考（仅供查证，须自行验证，不得直接照搬）】**"]
    for i, h in enumerate(r["results"][:limit], 1):
        lines.append("%d. %s\n   %s\n   %s" % (
            i, h.get("title", ""), h.get("url", ""), h.get("snippet", "")))
    txt = "\n".join(lines)
    return txt[:max_chars]


if __name__ == "__main__":  # 自测：python -m tools.web_search "query"
    import sys
    qq = sys.argv[1] if len(sys.argv) > 1 else "Cauchy Schwarz inequality"
    print(web_search_block(qq))
    print("stats:", web_search_stats())
