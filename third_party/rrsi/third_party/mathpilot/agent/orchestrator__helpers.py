# -*- coding: utf-8 -*-
"""orchestrator 通用模块级工具函数簇（2026-10-09 批16：从 agent/orchestrator.py 搬出）。

搬出函数：_sw_bool_or / _sw_num_or / _emergency_direct_max_tokens /
_load_lean_modules / _answer_fingerprint / _merge_review_into_feedback /
_collect_tool_findings

搬迁说明：
  · 除 _load_lean_modules 外**逐字搬**（AST 逐节点一致，见 check_h16_equiv.py）；
  · _load_lean_modules 有一处**显式 shim**（非逐字，team-lead 已知悉同类做法）：
    原体首行 `global LeanGate, LeanPreVerifier, _LEAN_MODULES_OK` 把重载成功的
    类回填**主模块 orchestrator** 的全局。搬出后 `global` 只会写本模块命名空间
    ⇒ 主模块的 `_LEAN_MODULES_OK/LeanGate/LeanPreVerifier` 不再被回填 ⇒
    ① `Orchestrator.__init__`（读主模块全局建 Lean 实例）②
    `orchestrator__lean._ensure_lean_modules`（在 `_load_lean_modules()` **之后
    再次读取**主模块全局，自愈语义）③ tests/test_lean_channel.py:37 对
    `agent.orchestrator._LEAN_MODULES_OK` 的 monkeypatch —— 三者全部失联。
    故改为函数内**惰性 `from . import orchestrator as _orch`** + 把那 3 个全局的
    读写改成 `_orch.<名>`（**实时读写主模块命名空间**，等价于原 `global`）。
  · 主模块以 `from .orchestrator__helpers import ...` **原位 re-export**；所有既有的
    `from agent.orchestrator import <名>` 调用点（tests/test_feedback_review.py 与
    orchestrator__revise/verify/final_gate/fallback 的惰性 import）零改动。

依赖：switch_registry 的容错读取（_sw_bool/_sw_num）。
本模块**不**在顶层 import orchestrator（仅 `_load_lean_modules` 内惰性 import），
故可被 orchestrator 顶层安全 import，无循环导入。
"""
from __future__ import annotations

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
from agent.switch_registry import (
    get_bool as _sw_bool, get_num as _sw_num)
import logging

logger = logging.getLogger("MathPilot")


def _sw_bool_or(field: str, default: bool) -> bool:
    """开关读取的**容错**版：注册表缺该字段时回退 `default`（不抛）。

    为什么需要（2026-10-07 fix-gate-timing-diag）：新开关的登记落在
    `switch_registry.py` / `user_agent.py` / `param_usage.py` 三个**共享文件**，
    可能与本施工员的 `orchestrator.py` 改动**分批落盘**。若裸调 `_sw_bool(field)`
    而此时注册未落，`SWITCHES[field]` 会 KeyError → 被调用点 except 吞掉 →
    功能**静默失效**（且日志无迹）。本函数保证"注册未落也用修好后的默认值"，
    注册落盘后自动获得 env/config 覆盖能力。
    """
    try:
        return bool(_sw_bool(field))
    except Exception:  # noqa: BLE001
        return bool(default)


def _sw_num_or(field: str, default: float) -> float:
    """`get_num` 的容错版（口径同上）。"""
    try:
        return float(_sw_num(field))
    except Exception:  # noqa: BLE001
        return float(default)


# 2026-09-13 晚：紧急直答的输出上限（token）。旧值直接用 `max_answer_tokens`
# （65536）= 允许"只输出一行答案"的调用写一整篇论文，与 prefill 语义矛盾，
# 且在读超时下更容易被截断/挂住。紧急直答只需要一行 ⇒ 1024 足够且更快更稳。
# 2026-10-02 DeepSeek 适配：env 默认 1024 ⇒ 8192。原值基于「直答只需一行 + prefill
# 抑制思维块」的 Intern-S 时代假设，对 reasoning 模型必然截断（reasoning 先吃满
# 预算、正文为空）。仍可用 EMERGENCY_DIRECT_MAX_TOKENS 覆盖。
# ★ 2026-10-15（用户定案「删直答兜底」）：`_emergency_direct_max_tokens`（紧急直答
#   参数）已随 `_emergency_direct_solve` / `_fallback_direct` 删除 —— 无消费者。


def _load_lean_modules() -> bool:
    """（重）加载 Lean 双通道模块 —— 治「循环导入导致的静默永久禁用」。

    2026-09-10 定位到的真实环：``tools/lean_local/lean_gate.py`` 模块级
    ``from agent.base import ...`` → 触发 ``agent/__init__.py`` → 它 eager
    导入 ``.orchestrator`` → 本文件顶部的 ``from tools.lean_local.lean_gate
    import LeanGate`` 此时命中**部分初始化模块**（lean_gate 还没执行到类定义）
    → ImportError 被上面的裸 ``except`` 吞掉 → ``_LEAN_MODULES_OK=False``、
    ``LeanGate=None``，**整个 Lean 通道永久静默关闭**（不报错、不打日志、
    输出完全正常，只是 Lean 一次没跑）。

    该错序与入口的 import 顺序绑定：谁先 import ``tools.lean_local.*``（如
    独立核验脚本、平台 main、部分单测），谁就会中招；先 import
    ``agent.orchestrator`` 则正常。**顺序依赖 = 不可靠**。

    修法：保留顶部快速路径，另设本函数在**首次真正需要 Lean 时**重试导入
    （那时环已解开，必然成功），成功即回填全局并重建实例。一次性开关变成
    可自愈的懒加载，任何入口顺序都不再能永久禁用 Lean。
    """
    from . import orchestrator as _orch  # noqa: F401 （惰性；实时读写主模块全局，兼容 monkeypatch）
    if _orch._LEAN_MODULES_OK and _orch.LeanGate is not None:
        return True
    try:
        from tools.lean_local.lean_gate import LeanGate as _LG
        from tools.lean_local.lean_pre_verifier import LeanPreVerifier as _LPV
        _orch.LeanGate, _orch.LeanPreVerifier = _LG, _LPV
        _orch._LEAN_MODULES_OK = True
        logger.info("Lean 双通道模块延迟加载成功（顶部导入曾被循环导入阻断，已自愈）")
        return True
    except Exception as _e:  # noqa: BLE001
        # ★ 2026-10-24：带 traceback，并留下可供诊断器读取的原因。
        import traceback as _tb
        _orch._LEAN_IMPORT_ERROR = _tb.format_exc()
        logger.warning("Lean 双通道模块重新加载仍失败，Lean 保持禁用: %s",
                       _e, exc_info=True)
        return False


def _answer_fingerprint(ctx) -> str:
    """当前「答案形态」的指纹 —— 用于跨 revise 轮判断**有没有真进展**。

    ★★ 2026-10-05 修（**两道"无进展保护"是死代码** —— 三步论第 3 步"修正要有效"）：
    两处保护原本都拿 `ctx.final_response` 比较，但它**直到
    `orchestrator.py:2191`（6_format 之后）才第一次被赋值**，而
    `_deep_revise_loop` 的**全部 5 个调用点（P1 / 4.5 oracle / 4.6 对抗 /
    5 全 0 票 / 5.5 低置信）都在 `:2191` 之前** ⇒ 比较对象**恒为 `""`**
    ⇒ `if _npa is not None and _cur_ans and ...` 与
    `if _ans_before and _ans_after and ...` **恒短路边** ⇒ 保护从未生效。
    日志实证：`P1 重解成效：答案未变化（0→0 字符）`。

    改用**候选答案集合**（排序去重后以 `|` 连接）作指纹 ——
    它在 revise 阶段一直有值，且只有"答案集合真的变了"才算有进展。
    """
    try:
        ans = sorted({str(getattr(c, "answer", "") or "").strip()
                      for c in (getattr(ctx, "candidates", None) or [])
                      if str(getattr(c, "answer", "") or "").strip()})
        return "|".join(ans)
    except Exception:  # noqa: BLE001
        return ""


def _merge_review_into_feedback(original: str, review: str,
                                append: bool) -> str:
    """把 Step 4 复核产物并入 revise 反馈 —— **附加**而非覆盖（默认）。

    ★★ 2026-10-06 修（**两处修复互相抵消**）：
      上游 `verifier.py:1819-1821` 已把「深复核推理」（尾段 6000 字符 + 判定段，
      标题 `【复核依据（判错报告原文摘录）】`，整体上限 8000）保留进 feedback；
      但 `_deep_revise_loop` 随后把复核产物**整段覆盖**回 `feedback`
      ⇒ 实测 q1–q6 最终 `diag.revise_feedback` 最长仅 2.4–3.0k、且**不含该段**
      ⇒ 那 2.5–6 万字符的推理仍没到模型手上。

      语义：复核只做「某条缺陷是否属实」的判定与标注，其产物以
      `【复核意见（逐条是否属实，误报已驳回）】` 段**附加**在原文之后；
      `append=False` 回退旧的「整段覆盖」行为（A/B 抓手）。
    """
    if not review:
        return original
    if not append:
        return review
    return (original
            + "\n\n【复核意见（逐条是否属实，误报已驳回）】\n"
            + review)


def _collect_tool_findings(ctx) -> str:
    """收集**纯工具**产出的判定（Lean 编译 / 系统命题验算 / SymPy），供结构化检错器做定位。

    ★★ 2026-10-06 新增（补上「**工具判定 → 答案级定位**」这一环）：
    工具的 finding 里 `location` 装的是**检查项名**（`answer_verify` / `lean_verify`），
    而 `agent/base.py:365` 对 `location` 的定义是「出错位置（步骤编号/行号/引用）」
    ⇒ **契约被违背**：模型收到 `location="answer_verify"` 等于没收到位置。
    工具看的是**形式化代码**、天然说不出"答案第 3 步错了"，但它
    「**这个等式/命题不成立**」的判定是**客观、可复现**的 ⇒ 是定位的最佳锚点。

    本函数把这些锚点抽出来，交给 `verifier.structured_report_for(tool_findings=...)`，
    让它**只负责在答案里逐字引用写下该等式的原文** ⇒
    产出「工具级的可靠性 + 答案级的定位」。

    去重（工具对多个候选常给出同一判定）+ 最多 3 条，避免污染提示词。
    """
    out = []
    try:
        for _k in ("lean_gate", "audit_gate"):
            for e in (getattr(ctx, _k, None) or []):
                if not isinstance(e, dict):
                    continue
                v = str(e.get("verdict") or "")
                if v in ("proof_invalid", "unknown", "reject"):
                    fb = str(e.get("feedback") or e.get("reason") or "").strip()
                    if fb:
                        out.append("[%s|%s] %s" % (_k, v, fb[:400]))
    except Exception:  # noqa: BLE001  收集失败不影响主流程
        pass
    seen, uniq = set(), []
    import re as _re_tf
    for x in reversed(out):
        k = " ".join(x.split())
        k = _re_tf.sub(r"\d+", "#", k)[:120]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
        if len(uniq) >= 3:
            break
    return "\n".join(reversed(uniq))
