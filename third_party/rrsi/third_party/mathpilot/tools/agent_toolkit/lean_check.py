# -*- coding: utf-8 -*-
"""工具①：Lean4 逻辑审核（逻辑错误点 + 逻辑缺少点 · 2026-10-15 用户定稿）。

职责（用户原话）：「接收大模型产出的东西进行逻辑审核，它的产出要是逻辑错误点
和逻辑缺少点……主要目的是让他纠出大模型的错误，并返回具体情况让大模型修正。」

能力最大化（依据 Lean/lean-lsp-mcp 实际能力设计）：
  ① 编译级检查（mcp_run_code → lean-lsp-mcp 本地执行）：
       · 类型/语法错误 → 「逻辑错误点」（带行号 + 诊断原文）
       · 未解决的证明目标（unsolved goals / tactic 失败）→ 「逻辑错误点（证明层）」
  ② 缺口定位：逐行扫描 `sorry` 占位 → 「逻辑缺少点」（哪一步没证）
  ③ 公理审计（lean_verify）与反例判定为后续增强（增量接口已预留 verdict 位）。

边界（诚实声明）：Lean 只认形式化语言——审核上限由「翻译质量」决定（工具④）；
「证不出」≠「命题假」，本工具不臆断（未找到反例时不下"假"结论）。
"""
from __future__ import annotations

import re
import time

_DIAG_RE = re.compile(
    r"^(.*?):(\d+):(\d+):\s*(error|warning)\b[:：]?\s*(.*)$", re.MULTILINE)

# 「证明层」错误关键词（区分"陈述写错"与"证明证不出"）
_PROOF_LAYER_HINTS = ("unsolved goals", "tactic", "tactic '", "no goals",
                      "failed to", "invalid '", "linarith", "omega",
                      "not a definitional equality", "type mismatch")

# 环境/工具执行失败标记（❗实测教训：proxy 的 `ok=True` 只表示"MCP 调用完成"，
#   LSP 起不来时仍 ok=True、错误藏在 `raw`；不识别会把"环境坏"误报成"代码 ok"）
_ENV_FAIL_MARKERS = ("error executing tool", "failed to start",
                     "closed the connection", "mcp-env-missing",
                     "no such file", "not found")


def _detect_env_failure(res: dict) -> str:
    """从 raw/error 提取环境失败痕迹；无则返回 ""。"""
    blob = (str(res.get("raw") or "") + " " + str(res.get("error") or ""))
    low = blob.lower()
    for m in _ENV_FAIL_MARKERS:
        if m in low:
            return blob.strip()[:220]
    return ""


def _parse_items(items) -> list:
    """proxy `items`（LSP 诊断列表）→ 错误点（字段名容错）。"""
    out = []
    if isinstance(items, list):
        for it in items[:12]:
            if isinstance(it, dict):
                sev = str(it.get("severity") or it.get("level") or "").lower()
                msg = str(it.get("message") or it.get("text")
                          or it.get("content") or it)
                pos = it.get("line") or it.get("start_line") or ""
                out.append({
                    "step": ("L%s" % pos) if pos else "",
                    "type": ("error" if ("err" in sev or sev in ("1", "2"))
                             else (sev or "info")),
                    "detail": msg[:300],
                })
            elif isinstance(it, str) and it.strip():
                out.append({"step": "", "type": "error",
                            "detail": it.strip()[:300]})
    return out


def _parse_diagnostics(text: str) -> list:
    """从编译输出提取诊断（行号 + 消息）→ 逻辑错误点列表。

    2026-10-15 增强：MCP 的 error 字段可能是**一段 JSON**
    （``{"success": false, "timed_out": false, "diagnostics": [...]}``）——
    优先按 JSON 解析取结构化诊断；失败则退回行级正则。
    """
    txt = str(text or "")
    if txt.strip().startswith("{"):
        try:
            import json as _json
            d = _json.loads(txt)
            diags = d.get("diagnostics") if isinstance(d, dict) else None
            out = []
            if isinstance(diags, list):
                for it in diags[:12]:
                    if isinstance(it, dict):
                        msg = str(it.get("message") or "")
                        sev = str(it.get("severity") or "error").lower()
                        ln = it.get("line") or it.get("start_line") or ""
                        out.append({
                            "step": ("L%s" % ln) if ln else "",
                            "type": ("error" if "err" in sev else sev),
                            "detail": msg[:300],
                        })
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    errs = []
    for m in _DIAG_RE.finditer(txt):
        errs.append({
            "step": "L%s" % m.group(2),
            "type": m.group(4),                       # error / warning
            "detail": (m.group(5) or "").strip()[:300],
        })
    # ★ 2026-10-15 审核修复（S5）：删「整段当 error」兜底 —— 非诊断格式的文本
    #   不是"逻辑错误点"，塞进来会把工具/环境类问题误判为 statement_error；
    #   主流程在"无任何诊断"时按 unknown 处理（不臆断）。
    return errs[:12]


def _is_proof_layer(errs: list) -> bool:
    """错误是否属「证明层」（证不出）而非「陈述层」（写错）。"""
    blob = " ".join(e.get("detail", "") for e in errs).lower()
    return any(h in blob for h in _PROOF_LAYER_HINTS)


def _find_sorry_gaps(code: str) -> list:
    """逐行扫描 sorry → 逻辑缺少点（附带上下文行，便于大模型定位补证处）。"""
    gaps = []
    for i, ln in enumerate((code or "").splitlines(), 1):
        if re.search(r"\bsorry\b", ln):
            gaps.append({
                "step": "L%d" % i,
                "goal": "",
                "hint": "该处为 `sorry` 占位（未证明）：%s" % ln.strip()[:120],
            })
    return gaps[:20]


def lean_check(lean_code: str = "", reasoning: str = "", problem: str = "",
               work_dir: str = "", client=None, runner=None,
               timeout: float = 120.0) -> dict:
    """Lean4 逻辑审核（输入 Lean 代码 或 自然语言推理 → 错误点 + 缺少点）。

    参数
    ----
    lean_code : 已形式化的 Lean 代码（优先；有它就直接审核）
    reasoning : 自然语言推理（无 lean_code 时经工具④ to_lean 翻译，需注入 client）
    problem   : 题目上下文（翻译/审核参考）
    work_dir  : Lean 工程目录（传给 mcp_run_code；缺省由 runner 决定）
    client    : LLM 客户端（仅翻译路径需要；由调用方注入）
    runner    : 编译执行器（默认惰性用 lean_bridge.mcp_run_code；测试可注入桩）

    返回
    ----
    {"ok": bool,
     "verdict": "ok"|"statement_error"|"proof_failed"|"proof_incomplete"
                |"needs_translation"|"unknown",
     "errors":  [{"step","type","detail"}],      # 逻辑错误点
     "missing": [{"step","goal","hint"}],        # 逻辑缺少点
     "lean_code": str, "notes": "", "elapsed_s": float, "error": str}
    """
    t0 = time.time()
    out = {"ok": False, "verdict": "unknown", "errors": [], "missing": [],
           "lean_code": "", "notes": "", "elapsed_s": 0.0, "error": ""}
    code = (lean_code or "").strip()

    # 无代码 → 尝试翻译（需 client；没有 ⇒ 如实告知"需要翻译"）
    if not code:
        if not (reasoning or "").strip():
            out["error"] = "empty-input（既无 lean_code 也无 reasoning）"
            out["elapsed_s"] = round(time.time() - t0, 2)
            return out
        from .to_lean import to_lean
        tr = to_lean(reasoning, context=problem, client=client)
        if not tr.get("ok"):
            out["verdict"] = "needs_translation"
            out["error"] = "翻译失败/缺 client：%s" % tr.get("error", "")
            out["elapsed_s"] = round(time.time() - t0, 2)
            return out
        code = tr["lean_code"]
        out["lean_code"] = code
        out["notes"] = tr.get("notes", "")
    else:
        out["lean_code"] = code

    # 惰性取默认执行器（可注入桩：runner(code, work_dir, timeout) -> {"ok","error"}）
    if runner is None:
        try:
            from tools.lean_local.lean_bridge import mcp_run_code as runner  # noqa: F811
        except Exception as exc:  # noqa: BLE001
            out["error"] = "lean-runner-unavailable: %s: %s" % (
                type(exc).__name__, str(exc)[:120])
            out["elapsed_s"] = round(time.time() - t0, 2)
            return out

    # ★ 2026-10-15：work_dir 缺省 → 复用 lean_bridge 的**自动探测**
    #   （与主链同口径；本机探测 = D:/mathlib4-last_bump_for_v4.31.0 ——
    #   这正是「之前能跑通」的配置。此前不探测直接用空目录 ⇒ mcp-env-missing）
    if not work_dir:
        try:
            from tools.lean_local.lean_bridge import (
                _detect_lean_project_dir as _detect_wd)
            work_dir = str(_detect_wd() or "")
        except Exception:  # noqa: BLE001
            work_dir = ""

    try:
        res = runner(code, work_dir, timeout)
    except Exception as exc:  # noqa: BLE001  执行异常 ⇒ unknown（不误判）
        out["error"] = "runner-exception: %s: %s" % (
            type(exc).__name__, str(exc)[:160])
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out
    res = res if isinstance(res, dict) else {"ok": bool(res)}

    err_text = str(res.get("error") or "")
    env_fail = _detect_env_failure(res)

    # ★ 环境失败优先判定（含"假 ok"：ok=True 但 raw 写明 LSP 起不来/工具执行失败）
    #   环境问题不是"逻辑错误"，绝不能误报 —— 一律 unknown。
    if env_fail and ("mcp-env-missing" in err_text
                     or "error executing tool" in env_fail.lower()
                     or "failed to start" in env_fail.lower()
                     or "closed the connection" in env_fail.lower()
                     or not res.get("ok")):
        out["verdict"] = "unknown"
        out["error"] = "lean-env-unavailable: %s" % env_fail
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    if res.get("ok"):
        # LSP 正常时：诊断在 items；先取错误项（若有 → 错误点）
        items_diag = _parse_items(res.get("items"))
        items_err = [d for d in items_diag if d.get("type") == "error"]
        if items_err:
            out["errors"] = items_err
            out["verdict"] = ("proof_failed" if _is_proof_layer(items_err)
                              else "statement_error")
            out["elapsed_s"] = round(time.time() - t0, 2)
            return out
        gaps = _find_sorry_gaps(code)
        out["ok"] = True
        if gaps:
            out["verdict"] = "proof_incomplete"
            out["missing"] = gaps
        else:
            out["verdict"] = "ok"
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    # 编译失败 → 分类错误点（★ 优先 `items` —— MCP 已结构化的诊断列表；
    #   `error`/`raw` 的 JSON 包装作为次选；无任何诊断 ⇒ unknown 不臆断）
    errs = _parse_items(res.get("items"))
    if not errs:
        errs = _parse_diagnostics(err_text)
    if not errs:
        errs = _parse_diagnostics(str(res.get("raw") or ""))
    if not errs:
        # ★ 2026-10-15 审核修复（S5）：无诊断文本 ⇏ statement_error（可能是
        #   工具/环境/超时类问题）——按 unknown 处理，绝不误判成逻辑错误。
        out["verdict"] = "unknown"
        out["error"] = "no-diagnostics（失败但无诊断文本）: %s" % (
            (err_text or str(res.get("raw") or ""))[:160] or "(空)")
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out
    out["errors"] = errs
    out["verdict"] = "proof_failed" if _is_proof_layer(errs) else "statement_error"
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out
