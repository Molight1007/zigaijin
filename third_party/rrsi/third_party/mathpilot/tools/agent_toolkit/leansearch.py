# -*- coding: utf-8 -*-
"""工具②：定理检索（LeanExplore 主通道 + 离线语料降级 · 2026-10-15 用户定稿）。

职责（用户原话）：「它的目的是接收大模型要处理的东西，根据传过来的东西去
mathlib 里面寻找可能相关的定理，给大模型提供思考方向建议，让大模型的注意力
能集中到正确的求解方法上，生成正确的思考过程。」

主通道：**LeanExplore 托管服务**（REST `https://www.leanexplore.com/api/v2/search`；
2026-10-15 实测：112 题官方预标注基准下逐定理命中 17.9% —— 显著优于本机可达的
其它通道；免安装、免密钥）。
降级：`tools/lean_local/lean_search.py`（离线语料 31 万条声明 —— 平台无外网兜底）。

注：LeanExplore 另有 MCP 端点（`https://www.leanexplore.com/mcp`）与本地 stdio
模式；本工具先用已验证的 REST 通道，MCP 接入留待整合层统一收口（增量即可）。
"""
from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_LEX_API = "https://www.leanexplore.com/api/v2/search"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 Chrome/126.0 Safari/537.36")


def _lex_search(query: str, top_k: int, timeout: float) -> list:
    """LeanExplore REST 检索 → 归一化列表；失败抛异常（由调用方降级）。"""
    url = _LEX_API + "?" + urlencode({"q": query, "limit": int(top_k)})
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for it in data.get("results", []):
        name = str(it.get("name") or "")
        mod = str(it.get("module") or "")
        full = (mod + "." + name) if (mod and name) else (name or mod)
        # 字段名容错（实测有 name/module/docstring；邻近版本可能带 informalization/source_code）
        informal = str(it.get("informalization")
                       or it.get("informal_statement")
                       or it.get("docstring") or "")
        out.append({
            "name": full,
            "short_name": name,
            "module": mod,
            "statement": str(it.get("source_code")
                             or it.get("signature") or "")[:400],
            "informal": informal[:300],
            "source": str(it.get("source_link")
                          or it.get("github_url") or ""),
        })
    return out


def _offline_search(query: str, top_k: int) -> list:
    """离线语料/源码扫描降级（惰性 import，避免拉重依赖）。"""
    from tools.lean_local.lean_search import MathlibTheoremSearcher
    searcher = MathlibTheoremSearcher()
    res = searcher.search(query, limit=top_k)
    out = []
    for r in (res.get("results") or []):
        out.append({
            "name": str(r.get("name") or ""),
            "short_name": str(r.get("name") or "").split(".")[-1],
            "module": str(r.get("file") or ""),
            "statement": str(r.get("snippet") or "")[:400],
            "informal": "",
            "source": "",
        })
    return out


def _make_direction(theorems: list) -> str:
    """从命中定理生成「思考方向建议」（规则拼接，不调 LLM —— 保持低耦合）。"""
    if not theorems:
        return "未检索到相关定理；可尝试换更具体的数学概念描述，或改用符号/类型模式查询。"
    lines = ["可参考以下定理方向（按相关度排序，先读前几条）："]
    for t in theorems[:5]:
        desc = t.get("informal") or t.get("statement") or ""
        desc = " ".join(str(desc).split())[:120]
        lines.append("- `%s`：%s" % (t.get("name", ""), desc or "（无摘要）"))
    lines.append("建议：优先围绕这些定理的名称与陈述调整解题思路，"
                 "把推理收敛到可用定理的适用范围上。")
    return "\n".join(lines)


def leansearch(query: str, top_k: int = 25, timeout: float = 20.0,
               offline_fallback: bool = True) -> dict:
    """定理检索（输入查询 → Mathlib 相关定理 + 思考方向建议）。

    参数
    ----
    query           : 大模型要处理的东西（题目/子目标/猜想，自然语言或含符号）
    top_k           : 取回条数（主链默认 25）
    timeout         : 在线通道超时（秒）
    offline_fallback: 在线失败时是否降级离线语料

    返回
    ----
    {"ok": bool, "backend": "leanexplore"|"offline"|"unavailable",
     "theorems": [{"name","short_name","module","statement","informal","source"}],
     "direction": str, "elapsed_s": float, "error": str}
    """
    q = str(query or "").strip()
    if not q:
        return {"ok": False, "backend": "unavailable", "theorems": [],
                "direction": "", "elapsed_s": 0.0, "error": "empty-query"}
    t0 = time.time()
    err = ""
    try:
        ths = _lex_search(q, top_k, timeout)
        return {"ok": True, "backend": "leanexplore", "theorems": ths,
                "direction": _make_direction(ths),
                "elapsed_s": round(time.time() - t0, 2), "error": ""}
    except Exception as exc:  # noqa: BLE001  在线失败 → 降级
        err = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    if offline_fallback:
        try:
            ths = _offline_search(q, top_k)
            if ths:
                return {"ok": True, "backend": "offline", "theorems": ths,
                        "direction": _make_direction(ths),
                        "elapsed_s": round(time.time() - t0, 2),
                        "error": "online-failed: " + err}
        except Exception as exc2:  # noqa: BLE001
            err += " | offline: %s: %s" % (type(exc2).__name__, str(exc2)[:120])
    return {"ok": False, "backend": "unavailable", "theorems": [],
            "direction": "", "elapsed_s": round(time.time() - t0, 2),
            "error": err or "unavailable"}
