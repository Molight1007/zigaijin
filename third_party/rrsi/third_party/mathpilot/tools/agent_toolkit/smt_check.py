# -*- coding: utf-8 -*-
"""工具⑤（SMT 命题验证）—— Z3 验证「逻辑命题/蕴含式断言」。2026-10-18。

**为什么需要**
工具③（SymPy）只能验「可代值的等式/不等式」，对**逻辑命题**——「若非严格
单调 ⇒ 存在 a1≠a2 使 P(a1)=P(a2)」（015）、「有界 ⇒ 总变差有界」（008）——
无能为力。Z3（SMT，微软开源、免费）能做：

  · `unsat`   ⇒ 断言**不可满足**（即命题恒真 / 无反例）；
  · `sat`     ⇒ 存在**反例**（返回具体模型 = 证伪）；
  · `unknown` ⇒ 无法判定（放行）。

**分层**：本模块属 `tools/agent_toolkit`，只做**底层验证**；「自然语言 → SMT-LIB」
转写由调用方（4.2 证伪的 LLM）完成。输入是**完整 SMT-LIB 文本**（含变量声明
与 `(assert …)`）。

**健壮性**：z3 未安装 / 解析失败 / 超时 ⇒ `unavailable/error/unknown`（调用方
放行，绝不因工具故障误杀）。
"""
from __future__ import annotations


def verify_smt(smt_text: str, timeout_ms: int = 30000) -> dict:
    """运行一段 SMT-LIB 文本并判定可满足性。

    参数：smt_text —— 完整 SMT-LIB（含 `(declare-fun …)` / `(declare-const …)`
          与 `(assert …)`），通常末尾是一条 `(assert (not <目标命题>))`，
          含义为「目标命题恒真 ⇔ 该断言不可满足(unsat)」。

    返回：{"verdict": unsat | sat | unknown | unavailable | error,
           "model": <sat 时的反例模型>, "reason": ...}
      · unsat ⇒ 命题恒真；sat ⇒ 有反例（证伪）；其余放行。
    """
    if not (smt_text or "").strip():
        return {"verdict": "unknown", "reason": "empty"}
    try:
        import z3  # 延迟导入；未安装 → unavailable
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "unavailable",
                "reason": "%s: %s" % (type(exc).__name__, str(exc)[:80])}
    try:
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(z3.parse_smt2_string(smt_text.strip()))
        r = s.check()
        if r == z3.unsat:
            return {"verdict": "unsat"}
        if r == z3.sat:
            return {"verdict": "sat", "model": str(s.model())[:240]}
        return {"verdict": "unknown"}
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "error",
                "reason": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
