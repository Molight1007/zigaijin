from __future__ import annotations

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
"""
多智能体基础组件
================

提供：
- ``Candidate`` / ``Verdict``：候选解答与验证结果的数据结构
- ``Budget``：LLM 调用预算（研究期限时/限 token 已放开，此处仅作调用计数与诊断）
- ``TaskContext``：共享黑板（Blackboard），所有 Agent 读写同一上下文，全程可追溯
- ``BaseAgent``：抽象基类，统一封装 LLM 安全调用、预算扣减、trace 记录
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

# ★ 2026-10-07 架构解耦（阶段 B1）：纯数据结构下沉到零依赖的 `contracts/`，
#   在此 **re-export 保持对外契约不变** —— 全仓 `from agent.base import Budget/Finding/BugReport`
#   一行都不用改（且是**同一对象**）。目的：`tools/lean_local/*` 不再因这些类反向依赖 agent。
from contracts.types import Budget, Finding, BugReport  # noqa: F401

logger = logging.getLogger("MathPilot")


# ============================================================
# 截断埋点（2026-08-31 L2.1 任务卡）
# ------------------------------------------------------------
# 平台日志显示 14.2% 调用被截断（truncated_count=434 / 3061），
# 但本地的 llm_client 丢弃了 finish_reason → 无法定位集中在哪些调用点。
# 这里加一层「疑似截断」检测（resp 长度 ≥ max_tokens × 0.95），
# 按 self.name（agent 名）计数。零行为变化，run 结束后通过
# `base.get_truncation_log()` 拿到分布，决定下一步改哪里。
# ============================================================
# ↓ 截断埋点实现已搬到 base__truncation.py（2026-10-08 拆分），此处 re-export 保契约。
from .base__truncation import (  # noqa: F401
    get_truncation_log, reset_truncation_log, set_truncation_enabled,
    _record_truncation_suspect, _record_truncation_confirmed,
    _register_truncation_listener, _TRUNCATION_LOG, _TRUNCATION_ENABLED,
    _TRUNCATION_RATIO,
)

# 时间护栏哨兵：区别于 _done 可能返回的 None（工具循环用）
_NOT_TIME_UP = object()


# ============================================================
# 响应归一化（P0-1 契约防线）
# ============================================================
# ★★ 2026-09-18（F1）：**文本通道工具调用**的开关与解析器。
#   2026-10-07 修（本通道**默认翻转为开** —— 见 `switch_registry.SWITCHES` 注释）。
#   开启后：解析文本形态的调用 → **白名单校验** → 真正执行 → 把结果以
#   `role=user` 追加后 `continue`，让模型基于真实检索结果继续推导。
#
#   ⚠ 2026-10-07：原为模块级常量 `_TEXT_TOOLCALL_FALLBACK = _sw_bool(...)`，
#   在 **import 时**求值 ⇒ 之后 `bind_config(cfg)` 写进 `_overrides` 的
#   **CLI/AgentConfig 覆盖永远读不到**（env 因在进程启动前设定才碰巧有效）。
#   现改为**调用时实时读取**（`_text_toolcall_fallback_enabled()`），
#   env 与 config 两条路径都生效，口径与 `switch_registry.get_bool` 完全一致。
# ↓ 文本通道/响应归一化实现已搬到 base__toolcall.py（2026-10-08 拆分），此处 re-export 保契约。
from .base__toolcall import (  # noqa: F401
    _TEXT_FN_RE, _TEXT_PARAM_RE, _text_toolcall_fallback_enabled,
    _TEXT_TOOLCALL_WHITELIST, _TEXT_TOOLCALL_REQUIRED, _TEXT_TOOLCALL_MAX_PARAM,
    _text_toolcall_acceptable, _parse_text_toolcall, _normalize_chat_response,
)


# ↓ 候选入池门槛开关实现已搬到 base__candidates.py（2026-10-08 拆分），此处 re-export 保契约。
from .base__candidates import (  # noqa: F401
    _pool_gate_v2_on, next_candidate_id, pick_best_candidates,
)


# ============================================================
# 数据结构（已搬到 base__types.py，2026-10-08 拆分；此处 re-export 保契约，同一对象）
# ============================================================
from .base__types import (  # noqa: F401
    Candidate, Verdict, RunState, RoundState, TaskContext,
)


# ============================================================
# Agent 抽象基类
# ============================================================
class BaseAgent(ABC):
    """所有智能体的基类"""

    name: str = "base"

    @staticmethod
    def note_mathlib_search(ctx: "TaskContext") -> None:
        """记录一次 leansearch 调用（区分"检索频繁"与"定理少"）。"""
        stats = ctx.mathlib_usage_stats or {}
        stats["search_calls"] = stats.get("search_calls", 0) + 1
        ctx.mathlib_usage_stats = stats

    @staticmethod
    def note_compile_valid(ctx: "TaskContext") -> None:
        """记录一次 lean_gate 编译通过（真正的形式化验证成功）。"""
        stats = ctx.mathlib_usage_stats or {}
        stats["compile_valid"] = stats.get("compile_valid", 0) + 1
        ctx.mathlib_usage_stats = stats

    @staticmethod
    def add_used_theorems(ctx: "TaskContext", names: list) -> None:
        """记录 AI 实际检索/引用过的 Mathlib 定理名（去重追加）。

        老师 #1/#2 证据链：AI 解答 → Lean 形式化验证用了哪些定理，
        要有可回填的记录。leansearch 命中、lean_gate 编译、refiner
        补全都会调用本方法写入 ctx.used_theorems，最终随 solve() 返回。
        """
        if not names:
            return
        repo = ctx.used_theorems if isinstance(ctx.used_theorems, list) else []
        seen = set(repo)
        for n in names:
            n = str(n).strip()
            if n and n not in seen:
                repo.append(n)
                seen.add(n)
        ctx.used_theorems = repo
        # 命中计数（未去重）：回应"调用频繁但定理不多"——检索命中条数
        # 可能远大于去重后的定理种类数
        stats = ctx.mathlib_usage_stats or {}
        stats["search_hits"] = stats.get("search_hits", 0) + len(names)
        ctx.mathlib_usage_stats = stats

    def __init__(self, client, config):
        self.client = client
        self.config = config

    @abstractmethod
    def run(self, ctx: TaskContext) -> TaskContext:
        """处理上下文并返回（可能更新的）上下文"""
        ...

    # ----------------------------------------------------------
    # 通用能力：trace 记录 + 带预算管控的安全 LLM 调用
    # ----------------------------------------------------------
    def record(self, ctx: TaskContext, step: str, content: str, **extra) -> None:
        """向 trace 追加一条决策记录"""
        entry = {"agent": self.name, "step": step, "content": content}
        if extra:
            entry.update(extra)
        ctx.trace.append(entry)

    def llm(self, ctx: TaskContext, messages: list, temperature: float,
            max_tokens: int) -> Optional[str]:
        """
        带预算管控、Token 裁剪、自动重试的安全 LLM 调用（支持并发线程）。
        修复 BUG-6：追加 max_tokens 裁剪、上下文超长自动降级重试、异常写入 trace。
        P0-4 修复：新增时间预算感知——剩余不足时跳过/降级，杜绝单题超时（45 error 主因）。
        """
        # 时间预算感知：剩余 < 60s 直接跳过调用（保证有产出优于无产出）
        # 修复：区分「未设置 deadline / 测试 fixture 伪 deadline」与「真实 deadline 已过期」。
        #   - deadline < 1e8（epoch 伪值，如测试 fixture 的 999.0）→ 不启用预算（兼容测试）
        #   - 真实时间戳但已过期 → remaining <= 0 → 跳过，杜绝无界超时重试（45 error/2923s 根因）
        if ctx.deadline:
            if ctx.deadline < 10**8:
                remaining = float("inf")   # 测试 fixture 用 epoch 伪 deadline
            else:
                remaining = ctx.deadline - time.time()
        else:
            remaining = float("inf")
        if remaining < 10:
            # P1 修复：阈值从 60s 降到 10s。本地测试不再因"剩余时间不足"跳过验证；
            # 真要超时也只损失最后 < 10s 的 LLM 调用，单题 deadline 仍由 is_timed_out 把控。
            logger.warning("[%s] 剩余时间不足 (%.0fs)，跳过 LLM 调用", self.name, remaining)
            ctx.trace.append({"agent": self.name, "step": "budget_skip",
                              "content": f"剩余时间不足 {remaining:.0f}s，跳过 LLM 调用"})
            return None
        # 2026-10-01：比赛期「剩余 <30s ⇒ max_tokens 减半」已删除（审计 A 级第 5 条）。
        # 该分支与下方 2026-09-03「比赛无限 token，**不裁剪 max_tokens**——截断 = 白白
        # 丢分」的既定口径**直接矛盾**，且仅写 warning、不写 trace ⇒ 静默腰斩输出。
        # 研究期求正确率上限：不再按剩余时间压缩输出预算。

        # 2026-09-03 老师：比赛无限 token，**不裁剪 max_tokens**——截断 = 白白丢分
        # （v10 实测大量 finish_reason=length，答案被腰斩）。cap 配置保留但默认 0=不限；
        # 若调用方显式传 max_tokens 则尊重（小输出任务如投票只传 512 属正常设计）。
        # ★ 2026-10-20（212 次截断的根因收口）：该模型是**推理模型**，先吐长段
        #   Thinking 再给正文，512/1024/2048 这类"短输出"上限会被思考整段吃掉
        #   ⇒ finish_reason=length、正文腰斩（10-19 实测：1024×55、512×7、
        #   2048 若干，合计 100+ 次；答案因此不完整/无有效解答）。
        #   ⇒ 统一保底 `llm_min_max_tokens`（默认 4096，显式给更小值也会被抬）。
        #   max_tokens 只是**上限**不是预留，短输出任务（投票/一行 VERDICT）
        #   不会因此变慢，只是不再被"思考"挤死。
        _min_tok = int(getattr(self.config, 'llm_min_max_tokens', 0) or 0)
        if _min_tok and (not max_tokens or max_tokens < _min_tok):
            max_tokens = _min_tok

        cap = getattr(self.config, 'max_tokens_cap', 0) or 0
        if cap and cap < 65536:
            if max_tokens:
                max_tokens = min(max_tokens, cap)
            else:
                max_tokens = cap

        # 2026-09-03 老师：比赛无次数上限。删除 budget can_spend 强制（仅记账不阻断），
        # 极端预算耗尽场景不再 skip LLM（导致 Lean 闸门 4/10 题被假跳过），
        # spend 仍记账用于事后分析（v4 budget_skips 指标）。
        if ctx.budget is not None:
            ctx.budget.spend(1)
        try:
            resp = self.client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp = _normalize_chat_response(resp)
            # 截断埋点：响应长度接近 max_tokens 即视为疑似截断
            # 2026-08-31 L2.1 任务卡：定位 14.2% 截断集中在哪些 agent
            if resp is not None:
                _record_truncation_suspect(self.name, len(resp), max_tokens)
            # 诊断：返回空值时记录
            if not resp or not resp.strip():
                logger.warning("[%s] LLM 返回空响应 (len=%d, type=%s)",
                               self.name, len(resp) if resp else 0, type(resp).__name__)
            return resp
        except TypeError:
            # 平台 client 仅接受 positional args → 降级
            logger.warning("[%s] TypeError in chat() kwargs, fallback to positional", self.name)
            try:
                resp2 = self.client.chat(messages)
                resp2 = _normalize_chat_response(resp2)
                if not resp2 or not resp2.strip():
                    logger.warning("[%s] LLM fallback 返回空响应", self.name)
                return resp2
            except Exception as e2:
                logger.warning("[%s] LLM fallback 也失败: %s", self.name, e2)
        except Exception as e:  # noqa: BLE001
            err_str = str(e)[:200]
            if any(kw in err_str.lower() for kw in ('context length', 'too long', 'max token')):
                # ★ 2026-10-04 修（D26）：原为 `reduced = max_tokens // 2`（无下限）
                #   ⇒ 对 reasoning 模型（reasoning 吃 77–97% 输出预算）**腰斩输出预算
                #   = 思维块占满、正文为空**，等于把"上下文超长"这个可恢复错误
                #   变成"空答案"这个不可恢复错误；且全程**不留痕**（静默）。
                #   现加下限（保 reasoning 不被饿死）+ 写 trace 便于归因。
                #   语义仍是"重试一次"，不改调用契约。
                reduced = max(4096, max_tokens // 2) if max_tokens > 4096 else \
                    (max_tokens // 2 if max_tokens > 512 else 256)
                logger.warning("[%s] 上下文超长，降至 %s tokens 重试（原 %s）",
                               self.name, reduced, max_tokens)
                try:
                    ctx.trace.append({
                        "agent": self.name, "step": "llm_shrink_retry",
                        "content": f"上下文超长 → max_tokens {max_tokens}→{reduced} 重试"})
                except Exception:  # noqa: BLE001
                    pass
                try:
                    resp3 = self.client.chat(messages=messages, temperature=0.0,
                                             max_tokens=reduced)
                    return _normalize_chat_response(resp3)
                except Exception:
                    pass
            # 2026-09-03：budget 强制已删（reserved/refund 逻辑移除）
            ctx.trace.append({"agent": self.name, "step": "llm_error",
                              "content": f"LLM 调用失败: {err_str}"})
            logger.warning("[%s] LLM call failed: %s", self.name, e)
            return None


    # ============================================================
    # ★ 2026-09-16 新增：联网搜索工具（原生 tool_calls 协议）。
    #   此前 `tools/web_search.py` 已实现但**全仓无调用者** ⇒ `tool_calls.web_search`
    #   恒为"未调用"、能力形同虚设（代码审计发现）。这里把它接进工具循环。
    #   由 `config.enable_web_search`（2026-10-01 起默认 True）控制是否注册。
    # ★ 2026-10-15（工具化定稿 · 双通道 · 大模型直接调用）：
    #   Lean/Math 工具集 schema —— 实现在独立板块 tools/agent_toolkit（低耦合），
    #   本处仅注册给大模型（原生 tool_calls 协议）。
    LEAN_TOOLS_SCHEMAS = (
        {"type": "function", "function": {
            "name": "lean_search",
            "description": ("在 Mathlib 检索可能相关的定理（自然语言查询 → 定理列表 + "
                            "思考方向建议）。用于定位正确的求解方向。"),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "数学概念/子目标/猜想（自然语言）"},
                "top_k": {"type": "integer", "description": "返回条数（默认 10）"}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "calc_verify",
            "description": ("机械复算一个数值/符号等式：给出表达式与你的声称值，"
                            "工具算出正确值并判定是否一致。用于自查每一步计算。"),
            "parameters": {"type": "object", "properties": {
                "expr": {"type": "string", "description": "算式（如 19.9+(10-sqrt(9.9))**2）"},
                "claimed": {"type": "string", "description": "你声称的结果"}},
                "required": ["expr"]}}},
        {"type": "function", "function": {
            "name": "to_lean",
            "description": "把一段数学陈述翻译为 Lean 4 声明（供 lean_check 使用）。",
            "parameters": {"type": "object", "properties": {
                "statement": {"type": "string"}, "context": {"type": "string"}},
                "required": ["statement"]}}},
        {"type": "function", "function": {
            "name": "lean_check",
            "description": ("用 Lean 4 对一段推理/Lean 代码做逻辑检查，返回"
                            "逻辑错误点与逻辑缺少点（供修正）。"),
            "parameters": {"type": "object", "properties": {
                "lean_code": {"type": "string"},
                "reasoning": {"type": "string"},
                "problem": {"type": "string"}},
                "required": []}}},
    )

    WEB_SEARCH_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "联网检索：给定一条**检索式**，返回数学相关网页/问答的标题、"
                "链接与摘要（主后端为 Math StackExchange，次为 arXiv）。"
                "用于查证不熟悉的定义、术语、公式或已知结论。"
                "**返回内容仅供参考，必须自行验证，不得直接照搬。**"),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索式（建议用英文数学术语，效果更好）",
                    }
                },
                "required": ["query"],
            },
        },
    }

    @staticmethod
    def _web_search_tool_exec(query: str) -> str:
        """执行 web_search。**任何失败都返回可读文本**，绝不抛异常打断求解。"""
        try:
            from tools.web_search import web_search
        except Exception:  # noqa: BLE001  提交包路径兜底
            try:
                from web_search import web_search  # type: ignore[no-redef]
            except Exception:  # noqa: BLE001
                return "ERROR: 联网搜索工具不可用"
        try:
            r = web_search(query, limit=4)
            if not r or r.get("status") != "ok" or not r.get("results"):
                return "（无结果，可换关键词重试）"
            lines = []
            for i, h in enumerate(r["results"][:4], 1):
                lines.append("%d. %s\n%s\n%s" % (
                    i, h.get("title", ""), h.get("url", ""),
                    (h.get("snippet") or "")[:220]))
            return ("【联网检索结果（仅供参考，须自行验证）】\n"
                    + "\n".join(lines))[:1600]
        except Exception as exc:  # noqa: BLE001
            return "ERROR: %s: %s" % (type(exc).__name__, str(exc)[:120])

    def llm_with_tools(self, ctx, messages: list, temperature: float = 0.0,
                      max_tokens: int = 32768, max_rounds: int = 3
                      ) -> Optional[str]:
        """带原生工具（web_search）的对话循环（≤max_rounds 轮）。

        模型生成 tool_call → 执行工具 → 结果以 tool 消息回传 → 模型继续；
        无 tool_calls → 返回最终文本。任何异常/平台不支持 tools → 回落 self.llm
        （行为与现状一致，零风险）。tool 执行次数 record 供审计。
        """
        try:
            msgs = list(messages)
            msgs = self._inject_tool_hint(msgs)
            _tools = self._tool_schemas()
            if not _tools:
                # ★ 2026-10-07（fix-obs-tail · 静默跳过埋点）：原**静默**回落普通调用
                #   （该分支此前**一行 trace 都没有** ⇒ diag 里看不到"工具循环为何没跑"）。
                try:
                    if ctx is not None:
                        self.record(ctx, "toolcall_mode",
                                    "工具循环已跳过：无可用工具可注册"
                                    "（enable_web_search 关闭或注册异常）→ 回落普通 llm")
                except Exception:  # noqa: BLE001  埋点失败绝不阻断
                    pass
                # 无工具可注册 → 回落普通调用（与 _maybe_tool_llm 同口径）
                return self.llm(ctx, messages, temperature, max_tokens)
            return self._run_tool_loop(ctx, messages, temperature,
                                      max_tokens, max_rounds, msgs, _tools)
        except Exception as exc:  # noqa: BLE001  失败回落普通调用
            logger.warning("[%s] 工具循环异常（回落普通 llm）: %s",
                           self.name, str(exc)[:120])
            return self.llm(ctx, messages, temperature, max_tokens)


    def _inject_tool_hint(self, msgs):
        """为工具循环注入工具使用说明（不改调用方 user 结构）。"""
        # 模型不知道工具存在就不会调用 → 在 system 层自动注入工具使用说明
        # （不改调用方 user 结构）。
        _tool_hint = ""
        # ★ 2026-09-16：联网搜索工具说明
        #   （模型不知道工具存在就不会调用）。
        try:
            if getattr(self.config, "enable_web_search", False):
                _tool_hint += (
                    "\n\n【可用工具】你还有函数 web_search(query)：联网检索数学资料"
                    "（主后端 Math StackExchange、次 arXiv），返回标题/链接/摘要。"
                    "**仅在遇到不熟悉的定义、术语、公式或需要查证已知结论时调用**，"
                    "不要滥用；检索结果仅供参考，**必须自行验证，不得直接照搬**。"
                    "\n⚠ **调用方式**：**必须优先**使用**原生函数调用通道**"
                    "（框架提供的 function calling）；**严禁**把 "
                    "`<tool_call>` / `<function=...>` / `<parameter=...>` "
                    "当作**正文内容**书写 —— 标签会**残留在答案里、污染整段"
                    "推导**（2026-10-07 起系统会兜底解析**完整标签**并执行后回填，"
                    "但仍以原生通道为首选）。"
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                msgs = [dict(msgs[0], content=str(msgs[0].get("content") or "") + _tool_hint)] + msgs[1:]
            else:
                msgs = [{"role": "system", "content": _tool_hint.strip()}] + msgs
        except Exception:  # noqa: BLE001
            pass
        return msgs


    def _tool_schemas(self):
        """按开关构造要注册的工具 schema 列表。"""
        # ★ 2026-09-16：按开关决定注册哪些工具。
        _tools = []
        try:
            if getattr(self.config, "enable_web_search", False):
                _tools.append(self.WEB_SEARCH_TOOL_SCHEMA)
        except Exception:  # noqa: BLE001
            pass
        # ★ 2026-10-15（工具化定稿 · 双通道）：Lean/Math 工具集（默认注册；
        #   enable_lean_tools=False 可关做 A/B）
        try:
            if getattr(self.config, "enable_lean_tools", True):
                _tools.extend(self.LEAN_TOOLS_SCHEMAS)
        except Exception:  # noqa: BLE001
            pass
        return _tools


    def _run_tool_loop(self, ctx, messages, temperature, max_tokens,
                      max_rounds, msgs, _tools):
        """工具循环主体：计数器 + 每轮生成/工具执行（≤max_rounds 轮）。"""
        n_calls = 0
        last_text = None
        # ★ 2026-10-07：文本通道可观测计数（解析到几个 / 执行了几个 / 回填了几个）。
        #   全 `tools_call` 循环共用；出口处汇总成一条 `toolcall_text_summary`。
        _text_seen = 0            # 响应文本里出现 `<tool_call` 的次数
        _text_parsed = 0          # 成功解析出 (函数名, 参数) 的次数
        _text_executed = 0        # 真正执行了工具的文本通道调用次数
        _text_refilled = 0        # 把执行结果以 `role=user` 回填进对话的次数
        _seen_text_calls = set()  # 幂等：同一次 tool-loop 内已执行过的 (工具, 检索式)
        # 文本通道兜底开关：**每次工具循环开始时读一次**（实时读 → env/config 覆盖均有效）
        _text_fb_on = _text_toolcall_fallback_enabled()

        def _done(_val, _why=""):
            """统一出口：文本通道有活动时落一条汇总计数（便于事后统计）。"""
            if _text_seen or _text_parsed or _text_executed:
                try:
                    self.record(
                        ctx, "toolcall_text_summary",
                        "文本通道：出现 %d / 解析 %d / 执行 %d / 回填 %d（出口=%s）"
                        % (_text_seen, _text_parsed, _text_executed,
                           _text_refilled, _why or "final"))
                except Exception:  # noqa: BLE001
                    pass
            return _val

        for _round in range(max_rounds):
            # 2026-09-12 修复（时间护栏）：工具循环此前**完全绕过** `llm()` 的
            # 时间守卫与 max_tokens_cap，循环内不查 ctx 剩余时间 ——
            # `max_rounds=3` × 单次 client timeout(180s) 最坏 540s，异常再回落
            # `self.llm` 又 180s，足以烧穿单题预算（历史"工具循环撞 timeout
            # 再花 180s"的根因）。时间不足时直接停手并返回已有文本，
            # **不再回落**（回落等于再等一轮）。
            _r = self._time_guard_stop(ctx, last_text, _done)
            if _r is not _NOT_TIME_UP:
                return _r
            try:
                _send_msgs = self._build_send_msgs(msgs)
                resp = self.client.chat(
                    messages=_send_msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=_tools,
                )
            except TypeError:
                # 平台 client 不支持 tools 参数 → 永久回落普通调用
                logger.warning("[%s] client 不支持 tools，回落普通 llm",
                               self.name)
                return self.llm(ctx, messages, temperature, max_tokens)
            if isinstance(resp, dict) and resp.get("tool_calls"):
                n_calls += 1
                self._dispatch_tool_calls(ctx, msgs, resp)
                continue
            # 无 tool_calls：str（文本）或 dict（content）
            last_text = self._extract_last_text(resp)

            # ==================================================
            # ★★ 2026-09-18（F1）/ ★ 2026-10-07（默认翻转为开 + 白名单 + 幂等）
            #   **文本通道工具调用回填**。
            #   本模型把调用写在正文/推理里，平台不填原生 `tool_calls`
            #   ⇒ 调用不执行、不回填，本轮生成到此结束，前面的散文被
            #   `extract_final_answer` 当答案抽出。
            #   实测（2026-10-06 smoke10 10 题）：`toolcall_text_detected` 共 93 条，
            #   而 `tool_calls.web_search` 在多数题为 `calls:0` —— 单题：题 000 = 18、
            #   007 = 16、003 = 15、004 = 11。题 000 模型自述「上一轮解答中，我错误地
            #   假设水平移动只能单向…最终推出矛盾得到答案 0。但题目条…」并发出
            #   `web_search` 去核对自纠错 —— **这条自纠错请求被整条丢弃**。
            #   本节：解析文本形态 → 白名单校验 → 真正执行 → `role=user` 回填 → continue。
            #
            #   三条硬约束：
            #   ① 权限不放宽：只执行 `_TEXT_TOOLCALL_WHITELIST`（= 原生通道注册的工具集）；
            #   ② 幂等：原生 `tool_calls` 分支在上方且先 `continue`（同一响应不重复执行）；
            #      文本通道单轮最多执行 1 个调用，且同一 (工具, 检索式) 只执行一次；
            #   ③ 有界：`n_calls < max_rounds` 与循环体共同限流，不会无限执行。
            # ==================================================
            _skip_text_reason = None
            if last_text and "<tool_call" in last_text:
                _text_seen += 1
                if not _text_fb_on:
                    _skip_text_reason = "fallback_off"
                elif n_calls >= max_rounds:
                    _skip_text_reason = "round_limit"
                else:
                    _tc = _parse_text_toolcall(last_text)
                    if not _tc:
                        # 有标签但格式不完整（如 <function=...> 无参数）
                        _skip_text_reason = "unparsed"
                    else:
                        _text_parsed += 1
                        _fname, _args = _tc
                        _q = str(_args.get("query", "") or "").strip()[
                            :_TEXT_TOOLCALL_MAX_PARAM]
                        _ok, _why = _text_toolcall_acceptable(_fname, _args)
                        if _ok and (_fname, _q) in _seen_text_calls:
                            _ok, _why = False, "duplicate_call"
                        if not _ok:
                            _skip_text_reason = _why
                        else:
                            n_calls += 1
                            _res = self._web_search_tool_exec(_q)
                            # 回填：保留模型原文（含标签）+ 追加工具结果
                            msgs.append({"role": "assistant", "content": last_text})
                            msgs.append({
                                "role": "user",
                                "content": ("【系统工具执行结果】\n" + _res +
                                            "\n\n请基于以上真实结果继续推导，"
                                            "并在最后给出最终答案。"),
                            })
                            _text_executed += 1
                            _text_refilled += 1
                            _seen_text_calls.add((_fname, _q))
                            try:
                                self.record(
                                    ctx, "toolcall_text_exec",
                                    "<%s:text> %s -> %s" % (_fname, _q[:60], _res[:50]),
                                    tool=_fname, query=_q[:200],
                                    result_head=_res[:160])
                            except Exception:  # noqa: BLE001
                                pass
                            continue
            # ★★ 2026-09-18（可观测性，审核发现）保留：**本分支只认原生 `tool_calls`**
            #（见上方 `isinstance(resp, dict) and resp.get("tool_calls")`），模型一旦走
            # 文本通道，此前**一行 record 都没有** ⇒ "模型想调用却被协议挡住"在 diag 里
            # 完全不可见（实测代价：025 的 5/5、032 的 4/4 候选退化为散文
            # `步骤11：搜索已知结论`，并被选为最终答案）。
            # 现在：凡「出现 <tool_call 但未执行」一律留痕 + 落**具体不可执行原因**，
            # 使「解析到但没执行」与「解析到并执行」可分开统计。
            if _skip_text_reason:
                try:
                    self.record(ctx, "toolcall_text_detected",
                                "文本通道出现 <tool_call> 而原生 tool_calls 为空 ⇒ "
                                "未执行、未回填（%s）：%s"
                                % (_skip_text_reason,
                                   str(last_text)[:120].replace("\n", " ")))
                except Exception:  # noqa: BLE001
                    pass
            if last_text and last_text.strip():
                return _done(last_text)
        # 达 max_rounds（仍连续调工具）→ 取最后一次文本兜底
        if last_text and last_text.strip():
            return _done(last_text, _why="max_rounds")
        return _done(None, _why="empty")


    def _build_send_msgs(self, msgs):
        """构造发送用历史浅拷贝（assistant 消息补齐 reasoning_content）。"""
        # ★ 2026-10-02 DeepSeek(thinking) 适配：与原生 tools 同发时，
        #   历史里**每一条** assistant 消息都必须携带 reasoning_content
        #   （空串即可）—— **不是只补末条**！漏一条即被硬拒 400
        #   `HTTP 400: reasoning_content must be passed back`。
        #   实测（.pytmp_probe_step2/diag_tools4.py，真实 DeepSeek
        #   `deepseek-v4-flash`）：R1(种子带 rc)/R2a/R2e(全带 rc) → 200；
        #   R2b(工具轮缺 rc)/R2c(种子缺 rc)/R2d(都缺) → 400。
        #   ⇒ 首版"只补末条种子"仅够第 1 轮；第 2 轮起历史里的种子
        #     （在 msgs 本体里仍无 rc）+ 模型工具轮共同触发 400。
        #   修法：**每轮发送前对整段历史补齐**（原值优先，缺失补 ""）；
        #   只改发请求用的浅拷贝，不动 msgs 本体（后续 append 逻辑不变）。
        #   为何对非 reasoning 模型（Intern）无影响：补的是**空串**、
        #   语义即"无思考内容"，对不消费该字段的后端是无害注入
        #   （且 Intern token 已 401，此路径当前不启用）。
        _send_msgs = [
            (dict(m, reasoning_content=m.get("reasoning_content", ""))
             if isinstance(m, dict) and m.get("role") == "assistant" else m)
            for m in msgs
        ]
        return _send_msgs


    def _lean_tool_exec(self, ctx, _fname: str, args: dict) -> str:
        """执行 Lean/Math 工具集（独立板块 tools/agent_toolkit）。

        任何失败都返回可读文本（绝不抛异常打断求解）——与 web_search 同口径。
        """
        import json as _json
        try:
            if _fname == "lean_search":
                from tools.agent_toolkit import leansearch as _f
                r = _f(str(args.get("query", "") or ""),
                       top_k=int(args.get("top_k", 10) or 10))
                return _json.dumps({
                    "ok": r.get("ok"), "backend": r.get("backend"),
                    "direction": r.get("direction"),
                    "theorems": [{"name": t.get("name"),
                                  "informal": str(t.get("informal") or "")[:180]}
                                 for t in (r.get("theorems") or [])[:8]]},
                    ensure_ascii=False)
            if _fname == "calc_verify":
                from tools.agent_toolkit import calc_verify as _f
                r = _f([{"expr": str(args.get("expr", "") or ""),
                         "claimed": str(args.get("claimed", "") or "")}])
                return _json.dumps(r, ensure_ascii=False)
            if _fname == "to_lean":
                from tools.agent_toolkit import to_lean as _f
                r = _f(str(args.get("statement", "") or ""),
                       context=str(args.get("context", "") or ""),
                       client=getattr(self, "client", None))
                return _json.dumps(r, ensure_ascii=False)
            if _fname == "lean_check":
                from tools.agent_toolkit import lean_check as _f
                r = _f(lean_code=str(args.get("lean_code", "") or ""),
                       reasoning=str(args.get("reasoning", "") or ""),
                       problem=str(args.get("problem", "") or ""),
                       client=getattr(self, "client", None),
                       work_dir=str(getattr(self.config, "lean_project_dir", "") or ""))
                return _json.dumps(r, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            return "工具执行失败: %s: %s" % (type(e).__name__, str(e)[:160])
        return "未知工具: %s" % _fname

    def _dispatch_tool_calls(self, ctx, msgs, resp):
        """执行原生 tool_calls（web_search）并以 tool 消息回填 msgs。"""
        import json
        msgs.append(resp)            # assistant 消息原样（含 tool_calls）
        for tc in resp["tool_calls"]:
            fn = tc.get("function", {}) or {}
            try:
                args = json.loads(fn.get("arguments", "") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            # ★ 2026-09-16：按函数名分派；★ 2026-10-15 扩展 Lean/Math 工具集
            _fname = str(fn.get("name", "") or "")
            # ★ 2026-10-20 修复（服务器日志 3 次 `cannot access local variable '_q'`）：
            #   原 `_q` 只在 web_search 分支赋值，Lean/Math 工具走 elif 后，
            #   下方 `_q[:60]` 必然 UnboundLocalError ⇒ **整个工具循环崩溃**，
            #   外层捕获后回落成"无工具 LLM"（sympy/Lean 能力整段失效）。
            #   现两分支都赋值，并把埋点标签改成真实工具名
            #   （原先把所有 lean 结果一律标成 <web_search>，日志失真）。
            _q = ""
            if _fname == "web_search":
                _q = str(args.get("query", "") or "")
                result = self._web_search_tool_exec(_q)
            elif _fname in ("lean_search", "calc_verify", "to_lean",
                            "lean_check"):
                _q = str(args.get("query", "") or args.get("statement", "")
                         or args.get("expr", "") or args.get("lean_code", "") or "")
                result = self._lean_tool_exec(ctx, _fname, args)
            else:
                continue
            _ev = "<%s> %s -> %s" % (_fname, _q[:60], result[:50])
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })
            try:
                self.record(ctx, "toolcall_exec", _ev)
            except Exception:  # noqa: BLE001
                pass


    def _extract_last_text(self, resp) -> str:
        """从模型响应提取最终文本（str / dict / 其他归一化）。"""
        if isinstance(resp, str):
            last_text = resp
        elif isinstance(resp, dict):
            last_text = str(resp.get("content", "") or "")
        else:
            last_text = _normalize_chat_response(resp)
        return last_text


    def _time_guard_stop(self, ctx, last_text, _done):
        """生成侧时间护栏：到时停手并返回已有文本；未到返回哨兵。"""
        if (ctx is not None and hasattr(ctx, "gen_time_up")
                and ctx.gen_time_up()):
            # ★ 2026-10-07（fix-obs-tail · 静默跳过埋点）：原**静默**停手返回
            #   （`_done(_why="time_up")` 只在文本通道有活动时才落一条汇总 ⇒
            #   纯原生 tool_calls 循环到点停手时 diag 无任何痕迹）。此处显式留痕。
            try:
                self.record(ctx, "toolcall_mode",
                            "工具循环已跳过：生成侧时间护栏触发（gen_time_up）"
                            "→ 取已有文本返回，不再回落普通 llm")
            except Exception:  # noqa: BLE001  埋点失败绝不阻断
                pass
            if last_text and last_text.strip():
                return _done(last_text, _why="time_up")
            return _done(None, _why="time_up")
        return _NOT_TIME_UP

    def _maybe_tool_llm(self, ctx, messages: list, temperature: float,
                        max_tokens: int) -> Optional[str]:
        """开关分派：`enable_web_search` 打开时走工具循环，否则走原 llm。

        ★★ 2026-09-16 审计发现过的**关键断链**：工具循环的开关判据曾与注册点不
        一致，导致 `enable_web_search=True` 完全无效。后改为按注册点判据统一。
        2026-10-01：`<calc>` 计算工具板块已整体删除，工具循环只剩联网搜索一个工具。
        """
        if bool(getattr(self.config, "enable_web_search", False)):
            try:
                if ctx is not None:
                    self.record(ctx, "toolcall_mode",
                                "工具循环开启（%s；web_search）" % self.name)
            except Exception:  # noqa: BLE001
                pass
            return self.llm_with_tools(ctx, messages, temperature, max_tokens)
        # ★ 2026-10-07（fix-obs-tail · 静默跳过埋点）：开关关闭时原**静默**走普通 llm
        #   （上游 ON 分支已有 `toolcall_mode` 记录，OFF 分支此前无任何留痕 ⇒
        #   A/B 时无法从 diag 判定"工具循环是否被开关关掉"）。
        try:
            if ctx is not None:
                self.record(ctx, "toolcall_mode",
                            "工具循环已跳过：enable_web_search 关闭 → 走原 llm")
        except Exception:  # noqa: BLE001  埋点失败绝不阻断
            pass
        return self.llm(ctx, messages, temperature, max_tokens)

# ↓ 超时/检测实现已搬到 base__detect.py（2026-10-08 拆分），此处 re-export 保契约。
from .base__detect import (  # noqa: F401
    TimeoutError, wall_clock_timeout,
    detect_hallucination, detect_truncated, detect_template_leak,
)
