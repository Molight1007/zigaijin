from __future__ import annotations

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
# ★ 2026-10-07 架构解耦（阶段 A）：注册表实现已**下沉到零依赖的 `contracts/`**
#   ⇒ 本模块直接 import `contracts.switch_registry`（**不再**是 `agent.switch_registry`），
#   `utils/` 由此不再依赖 `agent/`（分层规则禁止 utils→agent/tools；contracts 不在禁令内）。
#   读取点仍在本文件内（与改前一致）⇒ 行为零变更。
from contracts.switch_registry import (
    get_bool as _sw_bool, get_num as _sw_num)
"""
轻量级 OpenAI 兼容 LLM 客户端
==============================

用于本地测试时连接任意兼容 OpenAI Chat Completions API 的 LLM 服务。
支撑 ReasoningAgent 调用链：BaseAgent.llm() → client.chat()。

配置方式（优先级从高到低）：
1. 直接传参: LLMClient(api_key=..., base_url=..., model=...)
2. 环境变量: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
3. 默认: http://localhost:8000/v1, gpt-3.5-turbo

修改影响:
- 本模块**不依赖 agent/**（阶段 A 解耦）：开关改经 `contracts.switch_registry` 读。
- 修改 LLMClient.__init__ 签名时需同步检查: run_eval.py / tools/debug_one_trace.py / tests
- 修改 chat() 返回值格式时需同步检查: agent/base.py (BaseAgent.llm() 调用者)
- 被 agent/ 目录下所有 BaseAgent 子类通过 BaseAgent.llm() 间接引用
"""

import json
import logging
import os
import random
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger("MathPilot.LLMClient")

# ★ 2026-10-07（问题总账 §2.6 / §2.3）：此处原为**模块级常量**
#   `_DEFAULT_TIMEOUT_llm_client = int(os.getenv("LLM_TIMEOUT", "120"))`，有两重问题：
#   ① **import 期求值** ⇒ 若"先 import 后改 env"（仓库里 `cli/run_phase0_eval.py`
#      等正是这种写法）会**静默失效**；② 常量**不入注册表** ⇒ diag 的 switch_snapshot
#      看不到、报告还原不出来，且 config/CLI 覆盖读不到。
#   现改为**调用时**经注册表读（开关 `llm_timeout`，默认 120，读取优先级 env > config > 默认）。
#   2026-10-07 阶段 A/B2：注册表已下沉 `contracts/`，本处直接 import 它；
#   `_LLM_TIMEOUT_FALLBACK` 仍是「注册读取失败时的字面兜底」。
_LLM_TIMEOUT_FALLBACK = 120  # 秒；仅在注册表不可用时兜底
# 2026-09-13 晚（4 题实测复盘）：超时后**重试**才是真正的成本黑洞。
# 实测 4 题里 `failed after 2 attempts` 出现 5 次、5/5 全败，而单次故障 =
# 180s × 2 + 退避 ≈ **365s** —— 010/016 两题各被两轮 365s 烧光整题预算，
# 最终连答案都没生成（`predicted` 退化成 `[生成失败] 调用受限或模型拒绝回答`）。
# ⇒ 两处收紧：
#   ① 读超时 180 → 120（`LLM_TIMEOUT` 可覆盖；实测正常调用 1–3s，长推理留 120s 富余）
#   ② 超时类故障**默认不重试**（`LLM_RETRY_ON_TIMEOUT=1` 可恢复旧行为）——
#      "拿同样的请求再赌一次"，实测 5/5 全败，纯亏 timeout 秒。
# 单次故障成本：365s → 120s。
# ★ 2026-10-07：该开关为**调用时实时读取** `_sw_bool("llm_retry_on_timeout")`
#   （见 `chat` 内 `retry_on_timeout`）—— 若在 import 期求值成模块常量，
#   会把 `ReasoningAgent.__init__` 之前的注册表值固化，导致 config/CLI 覆盖失效。
_CONNECT_TIMEOUT = 30  # 秒（2026-09-03：45→30。connect/TLS 握手黑洞探测更快）
_MAX_RETRIES = 1      # 2026-09-03：2→1。持续网络问题时 3 次重试纯浪费；
                       # 瞬时抖动 1 次重试足够，配合调用方 is_time_critical 兜底
                       # （⚠ 仅对非超时故障生效，见 `llm_retry_on_timeout` 开关）
_RETRY_BACKOFF = 2.0
# 2026-09-11 限流专项退避（依据 Bug清单_团队提交de90cedc_0911.md §5）：
# 服务端 `-20048 请求过于频繁` 实测 176 次，且 5 个 revise 候选并发发起 →
# 固定 2s 退避（实测两次失败间隔仅 2.1s）等于没有退避，且无抖动 → 自激放大。
# 限流单独给更大基数 + 抖动 + 上限；并发线程因此错峰，不再同步撞墙。
_RETRY_BACKOFF_RATE = 8.0   # 限流场景退避基数（秒）
_RETRY_BACKOFF_MAX = 24.0   # 单次退避上限（秒）
_RETRY_JITTER = 0.5         # 抖动比例：wait += U(0, wait*JITTER)
_RATE_LIMIT_MARKERS = ("-20048", "429", "too many requests", "请求过于频繁", "rate limit",
                       # 2026-09-14 补：服务端过载也走大退避。
                       # 实测（今早 12 题回归）多次出现
                       # `HTTP 400 {"code":"-20014","message":"书生体验过于火爆，请稍后再试"}`
                       # —— 它是**服务端过载/限流**，但此前不在标记表里 ⇒ `_is_rate=False`
                       # ⇒ 只退避 2s（而 `max_retries=1` 意味着**只有一次重试机会**），
                       # 2s 后大概率仍撞墙、白烧一次调用。判定为大退避更合理。
                       "-20014", "过于火爆")

# ============================================================
# 真实截断信号（2026-09-01 SU-01 优化 0）
# ------------------------------------------------------------
# agent/base.py 已有启发式「疑似截断」埋点（resp_len >= max_tokens*0.95），
# 但服务端明确返回 finish_reason=length 才算**真实截断**。这里在
# 流式/非流式两条路径捕获真实 finish_reason，模块级统计 + listener 回调，
# 由调用方（agent/base.py）合并进截断日志，支撑截断率 <5% 健康阈值。
# 修改影响：chat() 返回值不变，纯增量。
# ============================================================
_TRUNCATION_STATS = {"calls": 0, "truncated": 0,
                     # ★ 2026-09-21 新增：截断的**分解**口径
                     #   by_max_tokens：哪一档上限在被撞（如 "32768": 4、"6144": 14）
                     #   by_site：哪段代码在撞（调用栈反查，如 "Solver@base.py:757 <- solver.py:1437"）
                     # 理由：本项目同时存在多个 token 上限（65536 / 6144 / 640 / 硬编码 32768），只报"截断了 N 次"无法决策该调哪个参数。
                     "by_max_tokens": {}, "by_site": {}}
_TRUNCATION_STATS_LOCK = threading.Lock()
_TRUNCATION_LISTENER = None

# ============================================================
# token 用量累计（2026-10-24 审计层 P0）
# ------------------------------------------------------------
# 为什么单独一个计数器：`_TRUNCATION_STATS` 只记「调用了几次 / 截断几次」，
# **不记 token**。成本分析（「涨 3 倍成本换 2 个点」）需要 token，而现成
# 的 `get_truncation_stats()` 是唯一被评测引擎按题差分的快照源
# （`run_eval__engine._llm_call_snapshot`）⇒ 把 token 挂在同一个快照上，
# 才能复用「本题 = 前后快照之差」这套已验证的口径，避免造第二套统计。
#
# 三条纪律：
#   1. 只在 HTTP 200 且解析成功时累加（重试路径不重复计费）；
#   2. 服务端未返回 usage 时累加 `usage_unavailable`，**不静默记 0**
#      （否则 token 看起来"有数"但只有一半）；
#   3. 全程 try/except，埋点绝不打断主流程。
# ============================================================
_USAGE_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls_with_usage": 0,
    "usage_unavailable": 0,
}
_USAGE_LOCK = threading.Lock()

# 流式 `stream_options.include_usage` 是 OpenAI 扩展。DeepSeek 兼容，
# 但别的网关可能直接 400 ⇒ 收到 400 就关掉并在本进程内永久降级，
# 不在每次请求上重复撞墙。
_STREAM_USAGE_SUPPORTED = True


def _record_usage(usage) -> None:
    """累加一次响应的 token 用量。``usage`` 可为 None / 非 dict（记 unavailable）。

    幂等于调用方：同一次响应只应调用一次（流式由 `_consume_stream` 内部去重）。
    """
    try:
        if not isinstance(usage, dict):
            with _USAGE_LOCK:
                _USAGE_STATS["usage_unavailable"] += 1
            return
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
        with _USAGE_LOCK:
            _USAGE_STATS["prompt_tokens"] += p
            _USAGE_STATS["completion_tokens"] += c
            _USAGE_STATS["total_tokens"] += t
            _USAGE_STATS["calls_with_usage"] += 1
    except Exception:  # noqa: BLE001
        pass


def get_usage_stats() -> dict:
    """token 累计快照（跨所有调用方）。与 `get_truncation_stats` 同源同口径。"""
    try:
        with _USAGE_LOCK:
            return dict(_USAGE_STATS)
    except Exception:  # noqa: BLE001
        return dict(_USAGE_STATS)


def set_truncation_listener(fn) -> None:
    """注册截断回调 fn(model: str, max_tokens: Optional[int], finish_reason: str)。

    在每次响应结束且 finish_reason=="length" 时被调用；异常被吞掉，零行为影响。
    """
    global _TRUNCATION_LISTENER
    _TRUNCATION_LISTENER = fn


def get_truncation_stats() -> dict:
    """返回 {"calls": N, "truncated": M}（真实 finish_reason 口径，跨所有调用方累计）。

    ★ 2026-10-24 增补（只加不改）：同时并入 token 累计键
    ``prompt_tokens / completion_tokens / total_tokens / calls_with_usage /
    usage_unavailable``。原有键与语义保持不变 ⇒ 老的调用方零影响；
    评测引擎因此能沿用同一套前后快照差分拿到「本题 token」。
    """
    with _TRUNCATION_STATS_LOCK:
        st = dict(_TRUNCATION_STATS)
    st.update(get_usage_stats())
    return st



def _capture_truncation_site(max_frames: int = 2) -> str:
    """取"真正发起这次 LLM 调用"的位置。

    形如 ``Solver@base.py:757 <- solver.py:1437``：
      · 跳过本文件自身的帧（chat / _mark_response 等）
        ⇒ 首个外部帧就是 `BaseAgent.llm` 包装层
        （顺便从它的 `self` 里取到 agent 名），
        再上一帧就是最终调用点。
      · 用 `sys._getframe` 逐级回退（比 inspect.stack 快得多，
        后者会读源码）。

    全程 try/except：取不到就返回空串，
    绝不因埋点本身影响主流程。
    """
    try:
        import sys as _sys
        _self_file = os.path.abspath(__file__)
        f = _sys._getframe(1)
        parts: list = []
        agent = ""
        hops = 0
        while f is not None and len(parts) < max_frames and hops < 60:
            hops += 1
            fn = os.path.abspath(f.f_code.co_filename)
            if fn != _self_file:
                if not agent:
                    try:
                        _s = f.f_locals.get("self")
                        agent = str(getattr(_s, "name", "") or "")
                    except Exception:  # noqa: BLE001
                        agent = ""
                parts.append("%s:%d" % (os.path.basename(fn), f.f_lineno))
            f = f.f_back
        if not parts:
            return ""
        head = parts[0]
        tail = (" <- " + parts[1]) if len(parts) > 1 else ""
        return (("%s@" % agent) if agent else "") + head + tail
    except Exception:  # noqa: BLE001
        return ""





def _mark_response(model: str, finish_reason: Optional[str],
                   max_tokens: Optional[int]) -> None:
    """每次 LLM 响应结束计数一次；finish_reason=='length' 时计截断并通知 listener。

    ★ 2026-09-21：截断时额外记录 **by_max_tokens**
      （哪一档上限在撞）与 **by_site**（哪段代码在撞）。
      理由：只报"截断了 N 次"无法决策该调哪个参数。
    """
    with _TRUNCATION_STATS_LOCK:
        _TRUNCATION_STATS["calls"] += 1
    if finish_reason != "length":
        return
    site = _capture_truncation_site()
    with _TRUNCATION_STATS_LOCK:
        _TRUNCATION_STATS["truncated"] += 1
        try:
            _k = str(int(max_tokens)) if max_tokens is not None else "?"
        except (TypeError, ValueError):
            _k = "?"
        _bmt = _TRUNCATION_STATS.setdefault("by_max_tokens", {})
        _bmt[_k] = _bmt.get(_k, 0) + 1
        _bs = _TRUNCATION_STATS.setdefault("by_site", {})
        _bs[site or "?"] = _bs.get(site or "?", 0) + 1
    logger.warning(
        "LLM response truncated (finish_reason=length, model=%s, max_tokens=%s, "
        "site=%s)",
        model, max_tokens, site or "?",
    )
    fn = _TRUNCATION_LISTENER
    if fn is not None:
        try:
            fn(model, max_tokens, finish_reason)
        except Exception:  # noqa: BLE001  listener 异常绝不外泄
            pass


class LLMError(Exception):
    """LLM 调用错误"""
    pass


class LLMClient:
    """OpenAI Chat Completions 兼容客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = _MAX_RETRIES,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
        # 规范化 URL：去尾部 /，确保以 /v1 结尾
        self.base_url = self.base_url.rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        # ★ 2026-10-07（阶段 A 解耦后仍成立）：超时**调用时**经注册表实时读（`llm_timeout`，
        #   默认 120，优先级 env > config > 默认）。显式传入的 timeout 仍最高优先。
        #   注册表来自 `contracts.switch_registry`（零依赖）⇒ 不再为读它而依赖 agent。
        if timeout is None:
            try:
                timeout = int(_sw_num("llm_timeout"))
            except Exception:  # noqa: BLE001  注册表不可用时字面兜底
                timeout = _LLM_TIMEOUT_FALLBACK
        self.timeout = timeout
        self.max_retries = max_retries

    # ── 主接口：与 BaseAgent.llm() 签名兼容 ──
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        # ★ 2026-10-02：4096→8192（DeepSeek 适配方案 A，统一上限）。此默认值仅在调用方
        #   完全省略 max_tokens 时生效（如 agent/base.py:824 的 TypeError 降级分支
        #   `client.chat(messages)`）；正常路径都显式传值，故只是抬高兜底、不改行为。
        max_tokens: int = 8192,
        stream: bool = False,
        tools: Optional[list] = None,
    ):
        """
        2026-09-09：新增 tools 支持（原生工具调用试点）——透传 tools 到 API；
        模型返回 tool_calls 时返回完整 assistant 消息 dict（供工具循环读取），
        否则保持原文本返回（兼容既有调用方）。
        """
        """
        发送 chat completion 请求，返回模型回复文本。

        参数:
            messages: [{"role": "system", "content": "..."}, ...]
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            stream: 是否使用流式输出。流式模式下逐步累积直到服务端
                自然结束（finish_reason=stop），可避免 max_tokens 截断
                导致推理不完整。默认 False 保持兼容。

        返回:
            模型回复的纯文本

        异常:
            LLMError: 所有重试耗尽后仍然失败
        """
        url = f"{self.base_url}/chat/completions"
        # 调用时实时读取 ⇒ config/CLI 覆盖生效（原先 import 期求值成模块常量）
        retry_on_timeout = _sw_bool("llm_retry_on_timeout")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
            # 2026-10-24：流式默认**不返回 usage**，必须显式要。否则审计层
            # 只能记到非流式的 token ⇒ 成本看起来"有数"但缺一半（静默失真）。
            if _STREAM_USAGE_SUPPORTED:
                payload["stream_options"] = {"include_usage": True}
        if tools is not None:
            payload["tools"] = tools

        return self._post_with_retries(
            url, headers, payload, stream, max_tokens, retry_on_timeout)

    def _post_with_retries(self, url, headers, payload, stream, max_tokens,
                           retry_on_timeout):
        """执行 POST 与重试/退避/异常分类（2026-10-08 从 chat 逐字搬出）。"""
        global _STREAM_USAGE_SUPPORTED
        last_error = None
        _tried = 0
        for attempt in range(self.max_retries + 1):
            _tried = attempt + 1
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(_CONNECT_TIMEOUT, self.timeout),
                    stream=stream,
                )
                if resp.status_code == 200:
                    if stream:
                        content = _consume_stream(resp, logger, self.model, max_tokens)
                        return content
                    data = resp.json()
                    # 2026-10-24 审计层：先记 token，再走任何 return 分支
                    # （tool_calls 分支会提前 return，漏记就会少算工具轮成本）
                    _record_usage(data.get("usage"))
                    # 2026-09-09：模型请求工具调用 → 返回完整 assistant 消息
                    # （含 tool_calls），由调用方工具循环读取；普通回答仍取文本
                    try:
                        _msg = (data.get("choices") or [{}])[0].get("message") or {}
                        if "tool_calls" in _msg and _msg["tool_calls"]:
                            # ★★ 2026-10-02 DeepSeek 适配（reasoning_content 必须回传）：
                            #   DeepSeek thinking 模式**要求把上一轮 assistant 消息里的
                            #   `reasoning_content` 原样回传**；此前这里重建 assistant
                            #   消息时**把它丢了** ⇒ 工具循环第二轮起被服务端拒绝：
                            #   `HTTP 400 The reasoning_content in the thinking mode
                            #   must be passed back to the API`。补回（存在才带，
                            #   非 reasoning 模型零影响）。
                            _assistant = {
                                "role": "assistant",
                                "content": _msg.get("content") or "",
                                "tool_calls": _msg["tool_calls"],
                            }
                            if _msg.get("reasoning_content"):
                                _assistant["reasoning_content"] = _msg["reasoning_content"]
                            return _assistant
                    except Exception:  # noqa: BLE001
                        pass
                    content = _extract_content(data)
                    # 真实截断信号：非流式响应的 finish_reason（部分代理可能缺失）
                    fr = None
                    try:
                        choices = data.get("choices") or []
                        if choices and isinstance(choices[0], dict):
                            fr = choices[0].get("finish_reason")
                    except Exception:  # noqa: BLE001
                        fr = None
                    _mark_response(self.model, fr, max_tokens)
                    logger.debug(
                        "LLM chat OK [model=%s, tokens=%s, len=%d]",
                        self.model,
                        data.get("usage", {}).get("total_tokens", "?"),
                        len(content),
                    )
                    return content

                # ★ 2026-10-24：网关不认 `stream_options`（OpenAI 扩展）时直接 400。
                #   去掉该字段立即重试一次，并在本进程内永久降级 —— 不能因为
                #   "想要 token" 就把整条推理链打挂。
                if resp.status_code == 400 and "stream_options" in payload:
                    _STREAM_USAGE_SUPPORTED = False
                    payload.pop("stream_options", None)
                    logger.warning(
                        "LLM stream_options 被服务端拒绝（HTTP 400）⇒ 本进程内"
                        "停止请求流式 usage；token 统计将只有非流式部分"
                        "（计入 usage_unavailable，不静默记 0）")
                    continue
                # 非 200：记录错误并重试
                last_error = (
                    f"HTTP {resp.status_code}: {resp.text[:300]}"
                )
                logger.warning(
                    "LLM chat attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries + 1, last_error,
                )

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout after {self.timeout}s"
                logger.warning("LLM chat timeout (attempt %d/%d)", attempt + 1, self.max_retries + 1)
                # 2026-09-13：**读超时**默认不重试。服务端挂住时重试是"同样的请求
                # 再赌一次"——实测 5/5 全败，每次白烧 self.timeout 秒，直接把单题
                # 预算烧穿（365s/次）。调用方（solver/orchestrator/formatter）已有
                # 各自的降级与兜底路径，交给它们比在这里重试更划算。
                # ⚠ 必须排除 `ConnectTimeout`：它同时继承 `ConnectionError` 与 `Timeout`
                # （`except Timeout` 在前会先命中它），而**连接**超时走的是 30s 的
                # `_CONNECT_TIMEOUT`、失败很快，重试仍有正收益（探测服务端抖动）。
                # 只有**读**超时（`self.timeout`，即 120s 那种）才值得放弃重试。
                if (not retry_on_timeout
                        and not isinstance(e, requests.exceptions.ConnectTimeout)):
                    break
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                logger.warning("LLM chat connection error: %s", e)
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning("LLM chat unexpected error: %s", e)

            if attempt < self.max_retries:
                # 2026-09-11 修正三处缺陷：
                #  ① 原判据 `attempt < _MAX_RETRIES` 用的是**模块常量**而非
                #     `self.max_retries` → 调用方传非默认 max_retries 时
                #     "重试循环次数"与"退避守卫"不一致（潜伏 bug）。
                #  ② 限流（-20048）未单独识别 → 固定 2s 退避对服务端节流无效。
                #  ③ 无抖动 → 并发候选同步重试，放大限流。
                _low = (last_error or "").lower()
                _is_rate = any(mk in _low for mk in _RATE_LIMIT_MARKERS)
                base = _RETRY_BACKOFF_RATE if _is_rate else _RETRY_BACKOFF
                wait = min(base * (2 ** attempt), _RETRY_BACKOFF_MAX)
                wait += random.uniform(0.0, wait * _RETRY_JITTER)
                logger.warning(
                    "LLM chat 重试 %d/%d（%s，退避 %.1fs）: %s",
                    attempt + 1, self.max_retries,
                    "限流" if _is_rate else "故障", wait, (last_error or "")[:160],
                )
                time.sleep(wait)

        # 实际尝试次数而非 max_retries+1：超时短路（break）时两者不等，
        # 报真实次数才能让日志/归因不被误导（旧写法会让"只试了 1 次"显示成 2 次）。
        raise LLMError(f"LLM call failed after {_tried} attempts: {last_error}")

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model}, base_url={self.base_url})"


def _consume_stream(resp, logger, model: str, max_tokens: Optional[int] = None) -> str:
    """消费 SSE 流式响应，累积 content + reasoning_content 直到 [DONE] 或结束。

    返回与 chat() 一致的纯文本：优先拼接 content；若 content 为空（纯推理模型
    如 DeepSeek-R1/Intern-S1 的推理全部在 reasoning_content），则回退拼接
    reasoning_content。这样调用方无需区分两种通道。

    结束时上报真实 finish_reason（截断信号）。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    chunk_count = 0
    _usage_seen = False

    # ★★ 2026-10-05 修（**流式中文乱码根因**，用户三步论的第 2 步"反馈要准确"）：
    #   原用 `iter_lines(decode_unicode=True)`，它按 `resp.encoding` 解码；
    #   而 SSE 的 Content-Type 是 `text/event-stream`（属 `text/*`）
    #   ⇒ requests 依 RFC 2616 **默认按 ISO-8859-1 解码** ⇒ UTF-8 中文被逐字节解坏。
    #   实测：`（计算错）` → `ï¼ˆè®¡ç®—é”™ï¼‰`，导致
    #   `_extract_error_type()` 的中文标签正则**全部匹配失败** ⇒ 一律回落"未标注"。
    #   非流式路径走 `resp.json()`（JSON 规范强制 utf-8）⇒ 所以同一题里
    #   "投票原文正常、深复核（流式）乱码"，这是同题内对照的强证据。
    #   修法：拿 **bytes** 后显式按 utf-8 解码。
    for _raw_line in resp.iter_lines(decode_unicode=False):
        if not _raw_line:
            continue
        line = (_raw_line.decode("utf-8", errors="replace")
                if isinstance(_raw_line, (bytes, bytearray)) else _raw_line)
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        choices = obj.get("choices", [])
        # ★ 2026-10-24（审计层 P0）：`stream_options.include_usage` 的用量帧
        #   形如 `{"choices": [], "usage": {...}}` —— 原逻辑在下一行就
        #   `continue`，导致**流式 token 恒为 0**（静默失真）。
        #   同一次响应只记一次：部分实现每个 chunk 都带累计 usage，重复加会翻倍。
        if not _usage_seen:
            _u = obj.get("usage")
            if isinstance(_u, dict) and _u:
                _record_usage(_u)
                _usage_seen = True
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        text = delta.get("content")
        if text:
            content_parts.append(str(text))
        reason = delta.get("reasoning_content")
        if reason:
            reasoning_parts.append(str(reason))
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        chunk_count += 1

    content = "".join(content_parts)
    if content:
        text_out = content
    else:
        text_out = "".join(reasoning_parts)
    # ★ 2026-10-04 修（D30，只加留痕、不改返回语义）：流式下若服务端返回 0 个有效
    #   chunk（content 与 reasoning 皆空），此处**原样返回 ""**，而上层（base/solver）
    #   可能把"空响应"当成一次正常但空答案 ⇒ **静默**（对比非流式路径
    #   `_extract_content` 会 raise LLMError，两条路径口径不一致）。
    #   这里补一条 warning 让"空流"可观测；返回语义保持不变（上层依赖 falsy 判定）。
    if not text_out:
        logger.warning(
            "LLM stream EMPTY [model=%s, chunks=%d, finish=%s] "
            "—— 流式返回空内容（content/reasoning 皆空），按空串返回",
            model, chunk_count, finish_reason,
        )
    # 真实截断信号：流式路径的 finish_reason
    _mark_response(model, finish_reason, max_tokens)
    # 整条流都没带 usage ⇒ 显式计一次「拿不到」，不静默当 0
    if not _usage_seen:
        _record_usage(None)
    logger.debug(
        "LLM stream OK [model=%s, chunks=%d, finish=%s, content=%d, reasoning=%d]",
        model, chunk_count, finish_reason, len(content), len("".join(reasoning_parts)),
    )
    return text_out


def _extract_content(data: dict) -> str:
    """从 OpenAI 格式响应中提取文本内容"""
    choices = data.get("choices", [])
    if not choices:
        # 部分代理返回格式不同
        if "response" in data:
            return str(data["response"])
        raise LLMError(f"No choices in response: {json.dumps(data, ensure_ascii=False)[:200]}")
    choice = choices[0]
    # 标准 OpenAI 格式
    message = choice.get("message", {})
    content = message.get("content", "")
    if content:
        return str(content)
    # DeepSeek-R1/V4 等推理模型: 优先 reasoning_content (含完整思考链)
    reasoning = message.get("reasoning_content", "")
    if reasoning:
        return str(reasoning)
    # 部分代理使用 text 字段
    text = choice.get("text", "")
    if text:
        return str(text)
    raise LLMError(f"Empty content in response: {json.dumps(choice, ensure_ascii=False)[:200]}")


# ── 模块自检 ──
