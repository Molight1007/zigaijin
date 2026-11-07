# -*- coding: utf-8 -*-
"""Lean 环境 / 定理检索职责簇（2026-10-09 批15：从 agent/orchestrator.py 逐字搬出）。

搬出方法：_ensure_lean_modules / _lean_active / _lean_applicable /
_retrieve_theorem_hints

**搬迁说明**：
  · _lean_active / _lean_applicable / _retrieve_theorem_hints **逐字搬**；
  · _ensure_lean_modules 有一处**显式 shim**（非逐字，team-lead 批准）：它读主模块全局
    `_LEAN_MODULES_OK`/`LeanGate`/`LeanPreVerifier`/`_load_lean_modules`，且在
    `_load_lean_modules()` **之后会再次读取**（自愈语义）⇒ 用顶层 import 快照会让第二次
    读拿到旧值、自愈失效。故函数内**惰性 `from . import orchestrator as _orch`** + 把这 4
    个符号改为 `_orch.<名>`（**每次调用实时读取**，兼容 tests/test_lean_channel.py:37 对
    `agent.orchestrator._LEAN_MODULES_OK` 的 monkeypatch）。
"""
from __future__ import annotations

import logging

from agent.switch_registry import get_bool as _sw_bool

logger = logging.getLogger("MathPilot")


def _ensure_lean_modules(self) -> None:
    """首用时自愈 Lean 模块（见 :func:`_load_lean_modules`）。

        顶部 import 若被循环导入阻断（``_LEAN_MODULES_OK=False``），此处重试
        导入并补建 ``lean_gate`` / ``lean_pre_verifier`` 实例——把「静默永久
        禁用」变成「首用自愈」。已就绪时零成本直接返回。
        """
    from . import orchestrator as _orch  # noqa: F401 （惰性；实时读取主模块全局，兼容 monkeypatch）
    if _orch._LEAN_MODULES_OK and _orch.LeanGate is not None:
        if self.lean_gate is None:
            try:
                self.lean_gate = _orch.LeanGate(self.client, self.config)
            except Exception as _e:  # noqa: BLE001
                logger.warning("LeanGate 重建失败: %s", _e)
        if self.lean_pre_verifier is None and _orch.LeanPreVerifier is not None:
            try:
                self.lean_pre_verifier = _orch.LeanPreVerifier(self.client, self.config)
            except Exception as _e:  # noqa: BLE001
                logger.warning("LeanPreVerifier 重建失败: %s", _e)
        return
    if not _orch._load_lean_modules():
        return
    try:
        if self.lean_gate is None and _orch.LeanGate is not None:
            self.lean_gate = _orch.LeanGate(self.client, self.config)
        if self.lean_pre_verifier is None and _orch.LeanPreVerifier is not None:
            self.lean_pre_verifier = _orch.LeanPreVerifier(self.client, self.config)
        logger.info("Lean 双通道已自愈启用（此前被循环导入静默禁用）")
    except Exception as _e:  # noqa: BLE001
        logger.warning("Lean 双通道自愈后实例化失败，保持禁用: %s", _e)


def _lean_active(self) -> bool:
    """Lean 通道是否启用：总开关开 + lean 模块在 + Lean 环境可用（结果缓存）。

        探测 = LeanGate 懒加载的 LeanBridge.lean_available（跑一次
        `lean --version`，进程内只探一次）。任何异常都视为不可用 → 回落
        AuditGate，绝不让 lean 探测本身拖垮/阻断评测。
        环境变量 LEAN_VERIFY=0（或 false/no/off）可一键关闭 lean 通道
        （单测经 tests/conftest.py 统一置 0；评测 A/B 对照也可用它）。
        """
    self._ensure_lean_modules()          # 2026-09-10：治循环导入静默禁用
    # ★★ 2026-10-24（第 4 次 lean_active=false 后）：**每个 return 点都必须留原因**。
    #   前四轮都是"结果 false、原因不明"，只能靠猜。记进 self._lean_why，
    #   由 diag_export 写进 diag.lean_skip_reason，跑完直接看是哪一步卡住。
    _env_off = "0" if not _sw_bool("lean_verify") else "1"
    if _env_off in ("0", "false", "no", "off"):
        self._lean_why = "switch_off: lean_verify=%r" % _env_off
        return False
    if not getattr(self.config, "enable_lean_verify", True):
        self._lean_why = "config_off: enable_lean_verify=%r" % getattr(
            self.config, "enable_lean_verify", None)
        return False
    if self.lean_gate is None:
        from . import orchestrator as _o
        self._lean_why = ("gate_none: _LEAN_MODULES_OK=%r LeanGate=%r"
                          % (getattr(_o, "_LEAN_MODULES_OK", None),
                             getattr(_o, "LeanGate", None)))
        return False
    if self._lean_probe is None:
        # ★★ 2026-10-24（第 4 次 lean_active=false 后定位到的两个真 bug）：
        #   ① **失败被缓存**：`self._lean_probe = False` 一旦写下就永不再探。
        #      而 `LeanBridge.lean_available` 在早期可能是 False（尚未探到），
        #      于是"第一次探测时机不好"⇒ 整题永久禁用 Lean。
        #      实测对照：诊断器新建编排器后立刻调 `_lean_active()` 返回 True，
        #      而 r4 里是 False ⇒ 差异只在**探测时机**，不在环境。
        #   ② **`_b` 可能未定义**：原 `except` 里没给 `_b` 赋值，紧接着的
        #      logger/why 又用 `_b` ⇒ 异常路径会抛 NameError 掩盖真实原因。
        #   现改为：**成功才缓存；失败只记原因，下次重探**（读属性零成本）。
        _b = None
        try:
            _b = self.lean_gate._bridge_inst
            _ok = bool(_b is not None and _b.lean_available)
        except Exception as _e:  # noqa: BLE001
            _ok = False
            self._lean_why = "probe_exc: %s: %s" % (
                type(_e).__name__, str(_e)[:120])
        if _ok:
            self._lean_probe = True
            self._lean_why = ""
        else:
            # 不写 self._lean_probe ⇒ 保持 None ⇒ 下次调用重新探测
            if not getattr(self, "_lean_why", ""):
                self._lean_why = "probe_false: bridge=%r available=%r" % (
                    type(_b).__name__ if _b is not None else None,
                    getattr(_b, "lean_available", None) if _b is not None else None)
            logger.warning("Lean 环境当前不可用（本次放行，下次重探）: %s",
                           self._lean_why)
    else:
        self._lean_why = ""
    return bool(self._lean_probe)


def _lean_applicable(self, ctx) -> bool:
    """按题 Lean 适用性（2026-09-10 用户要求「所有题都要用 Lean，除非非常
        简单的题」）。

        判定值由 1.1) 步骤落盘到 ``ctx.metadata["lean_applicable"]``（见
        :func:`agent.question_type.lean_applicable`）。此处只读，用于把
        **整题豁免**的题（答案不是数学对象：选项字母 / 判断值 / 概念文字）
        从 Lean 通道改路由到 AuditGate —— 即「豁免 Lean，但不豁免检测」，
        而不是让它们裸奔输出。

        缺省语义：metadata 无该键（旧结果、手搓 ctx、单测 fixture）→ 视为
        适用，保持既有行为完全不变。
        """
    md = getattr(ctx, "metadata", None)
    if isinstance(md, dict) and md.get("lean_applicable") is False:
        return False
    return True


def _retrieve_theorem_hints(self, ctx) -> None:
    """1.6) 定理检索（**已下移至 2.7 子目标求解**，2026-10-14，任务列表 #2）。

    ★ 逻辑整体搬到 :func:`agent.theorem_hint.run_hint_for` —— 它是**鸭子类型**
      （只要求 `agent.llm` / `agent.record` / `agent.config`），因此子目标求解器
      可原地调用，无需反向 import 编排层。

    本函数保留为**薄委托**（行为与搬移前逐字一致）：
      · `Orchestrator._retrieve_theorem_hints` 仍指向它（兼容既有调用/测试）；
      · 主链**不再**经此触发 —— `_run_main_stages` 里的 `1.2_theorem_hint` 阶段
        已撤销，检索改由 `sub_goal_solver.run` 体首发起。
    """
    from .theorem_hint import run_hint_for
    run_hint_for(self, ctx)
