# -*- coding: utf-8 -*-
"""文本通道工具调用解析 + 响应归一化 —— 从 agent/base.py 逐字搬出。

对外契约不变：``from agent.base import _normalize_chat_response`` 等仍可用（re-export）。
"""
from __future__ import annotations

import re
from typing import Optional

from agent.switch_registry import get_bool as _sw_bool

_TEXT_FN_RE = re.compile(r"<function\s*=\s*([A-Za-z_]\w*)\s*>", re.I)
_TEXT_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_]\w*)\s*>(.*?)"
    r"(?:</parameter>|</function>|</tool_call>|$)",
    re.I | re.S)


def _text_toolcall_fallback_enabled() -> bool:
    """文本通道兜底开关是否生效（**调用时实时读取**，env/config 覆盖均有效）。"""
    return _sw_bool("toolcall_text_fallback")


# ------------------------------------------------------------
# 文本通道**执行白名单**（2026-10-07）
# ------------------------------------------------------------
# ⚠ 安全约束：走文本通道**不放宽任何权限** —— 这里只允许执行「原生通道同样
#   注册给模型」的工具。
#   ★ 2026-10-17 修复（p0b5_r3 实测：12 条文本调用被吞）：白名单原只有
#     web_search，而 2026-10-15 新增的 4 个工具（lean_search / calc_verify /
#     to_lean / lean_check，`enable_lean_tools`）**已注册给原生通道**却未同步
#     进白名单 ⇒ 模型写文本 `<tool_call>` 时一律不执行、只留痕。
#     现按「原生注册集」同步（agent/base.py 的 _lean_tool_exec 注册表同源）。
_TEXT_TOOLCALL_WHITELIST = ("web_search", "lean_search", "calc_verify",
                           "to_lean", "lean_check")
# 白名单工具的**必填参数**（与 `WEB_SEARCH_TOOL_SCHEMA.parameters.required` 一致）。
# 缺失/空白 ⇒ 不执行（避免用空检索式打网络）。
_TEXT_TOOLCALL_REQUIRED = {"web_search": "query"}
# 检索式长度上限（防模型把整篇推导塞进 <parameter=query>）。
_TEXT_TOOLCALL_MAX_PARAM = 400


def _text_toolcall_acceptable(name, args):
    """文本通道调用的准入判定：白名单 + 必填参数非空。

    返回 `(True, "")` 或 `(False, 原因标签)`。原因标签落进 `toolcall_text_detected`
    诊断记录，使「解析到了但没执行」与「解析到了并执行」可区分、可统计。
    """
    _n = str(name or "").strip()
    if _n not in _TEXT_TOOLCALL_WHITELIST:
        return False, "unknown_function:%s" % (_n or "?")
    _req = _TEXT_TOOLCALL_REQUIRED.get(_n, "")
    if _req:
        _v = str((args or {}).get(_req, "") or "").strip()
        if not _v:
            return False, "missing_required_arg:%s" % _req
    return True, ""


def _parse_text_toolcall(text: str):
    """从文本通道解析工具调用，返回 `(函数名, 参数字典)`；解析不到返回 None。

    兼容三种实测形态（025/032/086 的真实输出）：
      · `<tool_call><function=web_search><parameter=query>Q</parameter></function></tool_call>`
      · 同上但 `</parameter>` 缺失（未闭合）
      · 标签内/外有空白或换行
    """
    if not text:
        return None
    _t = str(text)
    if "<function" not in _t.replace(" ", ""):
        return None
    m = _TEXT_FN_RE.search(_t)
    if not m:
        return None
    name = m.group(1)
    args = {}
    for pm in _TEXT_PARAM_RE.finditer(_t[m.end():]):
        args[pm.group(1)] = str(pm.group(2)).strip()
    if not args:
        return None
    return name, args


def _normalize_chat_response(resp) -> Optional[str]:
    """把 client.chat 的返回值统一成字符串。

    平台注入的 client 实现不定，常见返回形态：
      - str: 直接可用
      - dict: {"content": "...", "choices": [...], "message": {...}}
      - list: [{"content": "..."}, ...]
      - 对象: .content / .text / .message.content
      - bytes: 解码为 UTF-8
      - None / 异常: 返回 ""
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, bytes):
        try:
            return resp.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(resp, list):
        for item in resp:
            text = _normalize_chat_response(item)
            if text:
                return text
        return ""
    if isinstance(resp, dict):
        for key in ("content", "text", "output", "result"):
            if key in resp and resp[key] is not None:
                val = resp[key]
                if isinstance(val, str):
                    return val
                return _normalize_chat_response(val)
        if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
            choice = resp["choices"][0]
            if isinstance(choice, dict):
                if "message" in choice and isinstance(choice["message"], dict):
                    msg = choice["message"]
                    for key in ("content", "text"):
                        if key in msg and msg[key] is not None:
                            return str(msg[key])
                if "text" in choice and choice["text"] is not None:
                    return str(choice["text"])
            return _normalize_chat_response(choice)
        if "message" in resp and isinstance(resp["message"], dict):
            msg = resp["message"]
            for key in ("content", "text"):
                if key in msg and msg[key] is not None:
                    return str(msg[key])
        if "data" in resp:
            return _normalize_chat_response(resp["data"])
        return ""
    # 普通对象：尝试 .content / .text / .message
    for attr in ("content", "text", "response"):
        try:
            val = getattr(resp, attr, None)
            if val is not None:
                return _normalize_chat_response(val)
        except Exception:
            pass
    try:
        if hasattr(resp, "message") and resp.message is not None:
            return _normalize_chat_response(resp.message)
    except Exception:
        pass
    try:
        s = str(resp)
        if s and s != "None" and not s.startswith("<") and not s.startswith("{"):
            return s
    except Exception:
        pass
    return ""
