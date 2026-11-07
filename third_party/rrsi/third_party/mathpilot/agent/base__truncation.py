# -*- coding: utf-8 -*-
"""截断埋点（2026-08-31 L2.1 任务卡）—— 从 agent/base.py 逐字搬出。

对外契约不变：``from agent.base import get_truncation_log`` 等仍可用（re-export）。
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("MathPilot")

_TRUNCATION_LOG: list = []        # 全局追加列表（跨 question / 跨线程）
_TRUNCATION_LOCK = threading.Lock()
_TRUNCATION_ENABLED = True
# 阈值：响应长度达到 max_tokens 的 95% 即视为「疑似截断」
_TRUNCATION_RATIO = 0.95


def get_truncation_log() -> list:
    """返回当前所有疑似截断事件（按时间顺序）。"""
    with _TRUNCATION_LOCK:
        return list(_TRUNCATION_LOG)


def reset_truncation_log() -> None:
    """重置埋点。run 起点调用，避免上一次 run 残留。"""
    global _TRUNCATION_LOG
    with _TRUNCATION_LOCK:
        _TRUNCATION_LOG = []


def set_truncation_enabled(enabled: bool) -> None:
    """关闭埋点（默认开；性能/压力测试时可关）。"""
    global _TRUNCATION_ENABLED
    _TRUNCATION_ENABLED = bool(enabled)


def _record_truncation_suspect(agent_name: str, resp_len: int, max_tokens: int) -> None:
    """由 llm() 在响应长度接近 max_tokens 时调用，**零行为影响**。"""
    if not _TRUNCATION_ENABLED:
        return
    if not max_tokens:
        return
    if resp_len < max_tokens * _TRUNCATION_RATIO:
        return
    with _TRUNCATION_LOCK:
        _TRUNCATION_LOG.append({
            "ts": time.time(),
            "agent": agent_name,
            "resp_len": resp_len,
            "max_tokens": max_tokens,
        })


def _record_truncation_confirmed(model: str, max_tokens=None, finish_reason: str = "length") -> None:
    """真实 finish_reason=length 截断事件（llm_client 上报，2026-09-01 SU-01 优化 0）。

    与启发式「疑似截断」并存：confirmed=True 表示服务端明确截断。
    agent 名用 model 占位（llm_client 不知道调用点；调用点分布仍由启发式提供）。
    """
    if not _TRUNCATION_ENABLED:
        return
    with _TRUNCATION_LOCK:
        _TRUNCATION_LOG.append({
            "ts": time.time(),
            "agent": model,
            "resp_len": -1,                 # 真实信号无长度启发式
            "max_tokens": max_tokens or 0,
            "confirmed": True,
            "finish_reason": finish_reason,
        })


def _register_truncation_listener() -> None:
    """把 llm_client 的真实 finish_reason 信号接进截断日志（一次性注册，异常吞掉）。"""
    try:
        from utils.llm_client import set_truncation_listener
        set_truncation_listener(_record_truncation_confirmed)
        logger.debug("已注册 llm_client 真实截断信号 → base._TRUNCATION_LOG")
    except Exception:  # noqa: BLE001  平台环境无本地 llm_client 时静默跳过
        pass


_register_truncation_listener()
