# -*- coding: utf-8 -*-
"""开关注册表 —— **兼容 shim**（实现已下沉到零依赖的 ``contracts/switch_registry.py``）。

★ 2026-10-07 架构解耦（阶段 B2）
===============================
本文件**不再包含实现**，只把 `contracts.switch_registry` 的公开符号**原样再导出**，
使全仓既有 `from agent.switch_registry import ...`（约 48 行、覆盖 20+ 文件）
**一行不改**即可继续工作。

为什么用**显式再导出**而不是 `from contracts.switch_registry import *`：
  ① `import *` 会把该模块的**标准库导入**（`os` / `Any` / `Dict` / `Optional` / `Tuple`）
     一并泄漏进 `agent.switch_registry` 命名空间，污染面难控；
  ② 显式列出即**成文的公开 API 契约**，谁被下游用了一目了然；
  ③ 再导出的是**同一批函数对象**（非拷贝），其闭包指向 `contracts` 里**同一个
     `_overrides`** ⇒ 「经 `agent.switch_registry.bind_field` 设值，`contracts.switch_registry.get_bool`
     能读到」⇒ **状态只有一份**（见 `tests/test_decouple_stage_a_1007.py` 的实测断言）。
"""
from __future__ import annotations

from contracts.switch_registry import (  # noqa: F401  —— re-export（同一对象）
    SWITCHES,
    bind_config,
    bind_field,
    get_bool,
    get_num,
    get_str,
    source_of,
    snapshot,
    summary_line,
)
