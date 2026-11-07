from __future__ import annotations
"""契约层（零依赖）。

本包只允许依赖 Python 标准库（``dataclasses`` / ``typing`` / ``threading`` ...），
**不得** import ``agent`` / ``tools`` / ``utils`` 中任何模块 —— 它是分层的最底层：

    contracts/  ←  utils/  ←  tools/  ←  agent/  ←  root

目的：把被 ``tools/`` 反向依赖的**纯数据结构**与**协议**下沉到此处，
使 ``tools/ → agent`` 的这些依赖消失（``tools/`` 只依赖 ``contracts/``），
从而打断 ``agent ↔ tools`` 的包级环。

内容：
- ``contracts.types``：纯数据类（``Finding`` / ``BugReport`` / ``Budget``）
- ``contracts.protocols``：``TaskContext`` 的 Protocol（只声明 tools/ 真正用到的接口）
- ``contracts.switch_registry``：开关注册表（零依赖的核心语义 + 读取器）。
  由 ``agent/switch_registry.py``（**纯 re-export shim**）与全仓 ``from agent.switch_registry import ...``
  透明转发 ⇒ 下游零改动、状态只有一份。
"""
