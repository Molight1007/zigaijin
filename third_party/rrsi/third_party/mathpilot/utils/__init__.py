from __future__ import annotations
"""
utils 工具包 — 提供 LLM 客户端、答案提取、数学计算等通用能力。

修改影响:
- 修改本目录任何文件时需同步检查: agent/solver.py, agent/verifier.py, agent/orchestrator.py
- utils/llm_client.py → 被 agent/ 所有模块依赖
- utils/extract.py → 被 agent/solver.py 依赖
- utils/sympy_tools.py → 被 agent/verifier.py, agent/orchestrator.py 依赖
"""
from .llm_client import LLMClient, LLMError  # noqa: F401 — 供本地测试使用
