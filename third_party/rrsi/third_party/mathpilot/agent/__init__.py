from __future__ import annotations
"""
MathPilot 多智能体包
====================

对外暴露核心类，便于 ``user_agent.py`` 以薄壳方式调用：

    from agent.orchestrator import Orchestrator
    from agent.base import TaskContext, Budget, Candidate, Verdict
"""

from .base import BaseAgent, TaskContext, Budget, Candidate, Verdict
from .classifier import ClassifierAgent
from .solver import SolverAgent
from .verifier import VerifierAgent
from .formatter import FormatterAgent

# 2026-09-10 断环（重要）：**原第 17 行 `from .orchestrator import Orchestrator`
# 改为此处的惰性导出**。
#
# 原来的模块级 eager 导入制造了一个循环导入：
#   tools/lean_local/lean_bridge.py / lean_gate.py
#       模块级 `from agent.base import ...` → 触发执行 agent/__init__.py
#       → eager 导入 .orchestrator → orchestrator 顶部
#         `from tools.lean_local.lean_gate import LeanGate`
#       → 此刻 lean_gate 仍**处于部分初始化**（尚未执行到类定义）
#       → ImportError 被 orchestrator 的裸 `except` 吞掉
#       → `_LEAN_MODULES_OK=False`、`LeanGate=None`：**整个 Lean 通道永久
#         静默关闭**（不报错、不打日志、输出看起来完全正常，只是 Lean 一次没跑）。
#
# 后果是「谁先 import tools.lean_local.* 谁中招」的顺序依赖：评测入口
# （user_agent → agent.orchestrator）恰好安全，而独立核验脚本 / 平台 main /
# 部分单测会静默失去 Lean。9/10 实测：同一进程先 import lean_gate 则
# `_LEAN_MODULES_OK=False`，先 import agent.orchestrator 则 True。
#
# 断环方式：不在包初始化时拉起 orchestrator（agent/orchestrator 自身的
# `from agent.base import ...` 与 orchestrator 的 Lean 导入都在其之后执行即可）。
# 直接路径 `from agent.orchestrator import Orchestrator`（user_agent.py 及各
# 单测的实际用法）不受影响；包级 `from agent import Orchestrator` 由下方
# PEP 562 惰性提供，语义不变。
_LAZY_EXPORTS = ("Orchestrator",)


def __getattr__(name: str):
    """包级惰性导出（PEP 562）：仅在真正取用 ``agent.Orchestrator`` 时才导入。"""
    if name in _LAZY_EXPORTS:
        from .orchestrator import Orchestrator
        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseAgent", "TaskContext", "Budget", "Candidate", "Verdict",
    "ClassifierAgent", "SolverAgent", "VerifierAgent",
    "FormatterAgent", "Orchestrator",
]
