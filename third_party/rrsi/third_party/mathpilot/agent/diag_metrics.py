# -*- coding: utf-8 -*-
"""诊断导出专用汇总函数（2026-10-07 阶段C1：从 agent/orchestrator.py 抽出）。

本模块只做一件事：把 `ctx` 里的埋点收拢成 diag 需要的汇总结构。
**纯搬迁**：函数体逐字来自 orchestrator.py，行为与取值口径完全不变。
本模块**不依赖 Orchestrator**（只用 ctx / 标准库 / 同包纯函数）
⇒ 无反向依赖、不参与任何 import 环。
"""
from __future__ import annotations

import logging
import re as _re

logger = logging.getLogger("MathPilot")


# ---------------------------------------------------------------------------
# 结构化错误类型聚合（2026-09-15）
# ---------------------------------------------------------------------------
# 背景：用户要求「之后测试记录大模型的具体答题情况，把错误暴露得更加具体」。
# 验证器（VerifierAgent）判 B 时会从 VERIFIER_SYSTEM 约定的**封闭标签集**
# 里回带错误类型（见 `agent/verifier.py: ERROR_TYPE_TAGS`），并以
# `error_types={标签: 票数}` 的形式写进 trace。此处把它们从 trace 收拢成
# 一个整题口径的字典，供 diag / 归因脚本直接消费。
# ⚠ 纯埋点：不参与任何判定，缺 trace 时安静返回 {}。
# ---------------------------------------------------------------------------
def _merge_error_types(ctx) -> dict:
    """从 trace 收拢本题的结构化错误类型分布（整题口径优先）。"""
    trace = getattr(ctx, "trace", None) or []
    merged: dict = {}
    fallback: dict = {}
    for t in trace:
        if not isinstance(t, dict):
            continue
        step = t.get("step")
        dist = t.get("error_types")
        if not isinstance(dist, dict):
            continue
        if step == "verify_error_types":
            merged = dist          # 整题终态，直接采用
        elif step == "vote_error_types" and not merged:
            for k, v in dist.items():   # 单次投票口径，仅作回退累加
                fallback[k] = fallback.get(k, 0) + v
    return merged or fallback


def _sum_reject_votes(ctx) -> int:
    """本题被验证器判 B 的票数（整题口径，无记录则 0）。"""
    trace = getattr(ctx, "trace", None) or []
    for t in reversed(trace):
        if (isinstance(t, dict) and t.get("step") == "verify_error_types"
                and isinstance(t.get("n_reject"), int)):
            return t["n_reject"]
    return 0


# ---------------------------------------------------------------------------
# LeanSearch 检索埋点汇总（2026-09-15，老师 #44）
# ---------------------------------------------------------------------------
# 老师 #44 要求埋点必须能回答四个维度：调用次数 / 命中数 / 去重后条数 / 被采用条数。
# 数据源 = trace 里 step=="leansearch" 的条目（**每个 TaskContext 一题**，
# 所以天然是按题隔离的，不会把全局累计值错当成本题数据）。
# ⚠ 「被采用条数」由 LeanGate 在 Lean 编译通过时回记（`note_adopted`），
#   属**跨题累计**语义，因此这里只报本题的 calls/hits/unique/elapsed，
#   不把累计值混进来（否则同一份 diag 里两个口径打架）。
# ⚠ 纯埋点：无记录时安静返回零值字典。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _compute_subgoal_gaps(ctx) -> dict:
    """子目标逻辑缺口分析（2026-09-29，截图 #6）—— 纯埋点，不改任何分支。

    用户原话：
    > 子目标的设立有没有帮助大模型简化题目……可不可以用 lean 来检测有没有
    > sorry 的地方，**像这种缺少的逻辑点是不是就是大模型需要推理出来的点**？

    本函数把「子目标求解结果」翻译成**可归因的缺口清单**，回答三件事：
      ① 有多少子目标没解出来；
      ② 这些缺口**是什么性质**（纯逻辑跳跃 / 缺引理 / 形式化 / 计算 / 未归类）；
      ③ 其中**哪些才是"必须由大模型推理出来"的点**
         （只有「纯逻辑跳跃」算；形式化与检索问题不算 —— 那两类是工具链的事）。

    ★ 为什么必须做这个区分：若把全部失败笼统算作"模型推理不行"，
      优化方向会被引到调推理提示词；而实际上可能大半是译题错误 —— 那该修的是
      `lean_translator`。没有这个区分，后续所有优化决策都建立在错误归因上。

    ★ 数据源：只读 `ctx.subgoal_trace`（sub_goal_solver 已写入），
      **零新增开销、零 Lean 依赖、零 LLM 调用**。
      缺陷数据（`ctx.refine_result`，LEAP Stage3）当前链路未接入，故不参与；
      一旦接入，`gap_analyzer.merge_gap_sources()` 可直接做双源交集。
    """
    try:
        from .gap_analyzer import extract_subgoal_gaps
    except Exception as e:  # noqa: BLE001
        logger.debug("[gap_analyzer] 导入失败，跳过缺口分析: %s", e)
        return {"ok": False, "error": "模块导入失败"}
    try:
        r = extract_subgoal_gaps(getattr(ctx, "subgoal_trace", None) or [])
        # 明细只留前若干条，避免 diag 膨胀（完整清单离线分析时另取）
        if r.get("gaps"):
            r["gaps"] = r["gaps"][:20]
        return r
    except Exception as e:  # noqa: BLE001  埋点失败绝不阻断求解
        logger.debug("[gap_analyzer] 缺口分析异常: %s", e)
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


# ---------------------------------------------------------------------------
def _switch_snapshot() -> dict:
    """开关注册表生效值快照（2026-10-01 注册制）。

    永不抛异常：注册表不可用时返回空 dict，不影响 diag 生成。
    """
    try:
        from agent.switch_registry import snapshot as _snap
        return _snap()
    except Exception:  # noqa: BLE001
        return {}


def _tool_gateway_summary(ctx) -> dict:
    """统一工具调用遥测（2026-09-30，截图 #10）—— 纯埋点，不改分支。

    数据源 = `agent/tool_gateway.ToolGateway.summarize(ctx)`，即本 ctx 上
    那一个门面实例的调用统计。

    ★ 为什么必须落盘：
      用户诉求是「调用工具的方法有没有写成规范性的类函数，需要 Lean 检测时
      直接调用，适配各阶段」。改造后"工具到底被调了几次、为什么没成功"
      必须有**单一数据源** —— 否则又要回到"遍历 trace 猜"的老路（本项目
      已因此把"没跑"误读成"没问题"多次）。

    ⚠ 口径红线：本字段为全零**只说明"没有代码走门面"**，
      绝不能读成"本轮没调用工具" —— 既有调用点（orchestrator 自建 LeanGate、
      answer_falsifier 自建 LeanBridge 等）**尚未迁移到门面**，
      它们的调用量仍在各自的既有埋点里（`leansearch` / `toolcall_exec` 等）。
    """
    # ★ 2026-10-04（冗余清理，用户指示"不要留下代码地雷"）：
    #   `agent/tool_gateway.py` 整个模块（**508 行 / 17 函数**）**已删除**。
    #   判定依据：其唯一入口 `ToolGateway.for_ctx` 全仓生产零调用 ⇒
    #   `ctx._tool_gateway` 从不被设置 ⇒ `summarize` 恒返回全零
    #   ⇒ 本表是一张**永远没有内容的表**（典型"不改行为的死代码 + 误导性数据"，
    #   见 `docs/解题逻辑审核_函数级_2026-10-04.md` D16）。
    #   本函数**保留 key**（而非删 key）是为了不破坏下游对 diag 结构的既有假设。
    return {"total": 0, "by_tool": {}, "by_reason": {}, "elapsed_total": 0.0,
            "_note": "ToolGateway 模块已于 2026-10-04 删除（从未接线 ⇒ 遥测恒空）。"
                     "实际工具调用量见 leansearch / toolcall_exec 等既有埋点"}


# ---------------------------------------------------------------------------
def _parse_exhaust_result(subgoal_trace) -> dict:
    """解析「解族穷尽性检查」子目标的输出，判断它**是否真的做了检查**。

    ★ 2026-09-17 新增（针对"不可检测"问题）：
    此前该子目标的 `result` 是**自由文本** ⇒ 无法区分"列了所有解族后确认只有一解"
    与"压根没列、直接把前序结论抄了一遍"。
    实测 003 的 result 就是一句 `a_n = n \\text{ for all } n \\ge 0` —— 无从判断。

    现按 `sub_goal_solver` 中强制的固定格式解析四段：
      `解族清单:` / `逐族判定:` / `前序结论复核:` / `结论: EXHAUSTIVE: yes|no`
    并给出 `complete`（四段是否齐全）与 `n_families`。
    **`complete=False` 即视为"未完成检查"** —— 这是可机检的硬信号。
    """
    out = {"found": False, "complete": False, "n_families": 0,
           "verdict": "", "missed": "", "reason": "无该子目标"}
    try:
        for sg in (subgoal_trace or []):
            if str(sg.get("title", "")).strip() != "解族穷尽性检查":
                continue
            out["found"] = True
            txt = str(sg.get("result", "") or "")
            if not txt.strip():
                out["reason"] = "result 为空"
                return out
            has_list = "解族清单" in txt
            has_judge = "逐族判定" in txt
            has_review = "前序结论复核" in txt
            # ⚠ 本模块用 `import re as _re`（非 `re`）—— 首版写 `re.search` 直接
            #   NameError 被兜底吞成"解析异常"，测试才发现。别再用裸 `re`。
            m_verdict = _re.search(r"EXHAUSTIVE\s*[:：]\s*(yes|no)", txt, _re.I)
            out["complete"] = bool(has_list and has_judge and has_review
                                   and m_verdict)
            out["verdict"] = (m_verdict.group(1).lower() if m_verdict else "")
            # 解族数量：取「解族清单」段里出现的分隔符数量 + 1（保守估计）
            if has_list:
                seg = txt.split("解族清单", 1)[-1]
                seg = _re.split(r"逐族判定", seg, maxsplit=1)[0]
                n = len(_re.findall(r"[/、,，;；]|\n\s*[-*\d]", seg))
                out["n_families"] = min(max(n, 0), 40)
            if "遗漏解族" in txt:
                out["missed"] = txt.split("遗漏解族", 1)[-1].strip()[:300]
            out["reason"] = ("格式完整" if out["complete"]
                             else "缺段：%s" % ",".join(
                                 [n for n, ok in (("解族清单", has_list),
                                                  ("逐族判定", has_judge),
                                                  ("前序结论复核", has_review),
                                                  ("EXHAUSTIVE", bool(m_verdict)))
                                  if not ok]))
            return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = "解析异常: %s" % type(exc).__name__
    return out


# ★ 2026-09-17（P1-F）：受监控的诊断键（trace step 名）。
_DIAG_MONITORED_KEYS = (
    "value_attack", "formal_spec", "formal_gaps", "pick_diag", "lemma_repo",
    "preverify_trace", "symbolic_solve_events", "objective_check_events",
    "numericize_events", "expression_eval_events",
    # 2026-09-18：文本通道工具调用（模型想调用但被协议挡住，此前不可见）
    "toolcall_text_detected",
)


# 属性型埋点键 → ctx 属性名（不经 `self.record` 写入，须单独查）
_DIAG_ATTR_KEYS = {
    "pick_diag": "_pick_diag",
    "formal_spec": "formal_spec",
    "formal_gaps": "formal_gaps",
    "lemma_repo": "lemma_repo",
    "preverify_trace": "preverify_trace",
}


def _diag_completeness(ctx) -> dict:
    """2026-09-17（P1-F）：**区分「未执行」与「执行但无产出」**。

    背景：实测 13/14 个诊断键在多数题为「键存在但值为空」，读 `diag` 时**无法判断**
    是"该环节根本没跑"还是"跑了但没产出" ⇒ 归因只能靠猜（此前已多次因此误判）。

    三态：
      · `not_ran`        —— trace 里没有该 step 的任何记录（环节未执行）
      · `ran_no_output`  —— 有记录，但内容与附加字段均为空（执行过、无产出）
      · `ran_with_data`  —— 有记录且有内容/附加字段（正常）
    """
    out = {}
    try:
        _tr = [t for t in (getattr(ctx, "trace", None) or [])
               if isinstance(t, dict)]
    except Exception:  # noqa: BLE001
        _tr = []
    for _k in _DIAG_MONITORED_KEYS:
        _hits = [t for t in _tr if str(t.get("step", "")) == _k]
        # ★ 2026-09-18 修（审计发现）：受监控键里有**属性型**的（不经 trace 写入），
        #   如 `pick_diag` 实为 `ctx._pick_diag`。原实现只查 trace step ⇒ 对它们
        #   **恒报 `not_ran`**，与事实矛盾（实测 10/10 题 `pick_diag` 非空却报 not_ran）。
        #   属性存在且非空 ⇒ 视为有产出。
        if not _hits:
            _attr = _DIAG_ATTR_KEYS.get(_k)
            if _attr:
                try:
                    if getattr(ctx, _attr, None):
                        out[_k] = "ran_with_data"
                        continue
                except Exception:  # noqa: BLE001
                    pass
            out[_k] = "not_ran"
            continue
        _has = False
        for _t in _hits:
            if str(_t.get("content", "") or "").strip():
                _has = True
                break
            if any(_kk not in ("step", "content") for _kk in _t.keys()):
                _has = True
                break
        out[_k] = "ran_with_data" if _has else "ran_no_output"
    # ★ 2026-10-07（fix-gate-timing-diag 问题 2.7 ①）：`formal_spec` 的"跑过没有"
    #   不能只看 `ctx.formal_spec` —— 它**只在 preverify verdict=ok 时写入**
    #   （`tools/lean_local/lean_pre_verifier.py:127`），verdict=fail 会被刻意清空
    #   （同文件 :174 `ctx.formal_spec = ""`）。
    #   实测 smoke10：8/10 题有 `preverify_trace`（verdict **全部 = fail**）
    #   ⇒ `ctx.formal_spec` 恒空，旧口径把"跑了但未通过"误报成 `not_ran`，
    #   与同一份 diag 里 `preverify_trace.formal_spec` 有值**自相矛盾**。
    #   现以「preverify_trace 是否留下痕迹」作为"跑过没有"的权威依据。
    if out.get("formal_spec") == "not_ran":
        _pv = getattr(ctx, "preverify_trace", None)
        if isinstance(_pv, dict) and _pv:
            out["formal_spec"] = (
                "ran_with_data"
                if str(getattr(ctx, "formal_spec", "") or "").strip()
                else "ran_no_output")
    return out


def _summarize_deep_review(ctx) -> dict:
    """按题汇总「带推理的最终复核」的判定（2026-09-15）。

    ★ 为什么必须单独导出：复核的否决是通过**压低簇置信度**表达的（复用既有
      revise 通道），**不产生新的 Verdict** ⇒ `n_reject_votes` / `error_types`
      只看投票列表，**看不到复核说了什么**。第一轮实测就吃了这个亏：
      086 的 `4_verify` 从十几秒涨到 249 秒（说明复核跑了），但导出里
      `n_reject_votes=0`、`error_types={}`，**无法判断它是判了 A 还是判了 B**。
      故此处把复核的判定作为**独立信号**导出（不伪造投票票数）。
    """
    entries = [t for t in (getattr(ctx, "trace", None) or [])
               if isinstance(t, dict) and t.get("step") == "deep_review"]
    if not entries:
        return {"ran": False, "verdict": "", "error_type": "", "skipped": ""}
    last = entries[-1]
    verdict = str(last.get("verdict") or "")
    ran = verdict in ("A", "B")
    return {
        "ran": ran,
        "verdict": verdict,
        "error_type": str(last.get("error_type") or ""),
        "chars": int(last.get("chars") or 0),
        # 未运行时的原因（时间不足 / 已达上限 / 未启用）
        "skipped": "" if ran else str(last.get("content") or "")[:120],
    }


def _summarize_leansearch(ctx, cfg=None) -> dict:
    """按题汇总 LeanSearch 检索埋点。

    ★ 2026-09-21 扩写（用户要求「记录 leansearch 到底要找多少 Mathlib
      定理」）：此前只有**产出侧**口径（calls / hits / unique / elapsed_ms /
      root），缺**需求侧与触顶侧**口径 ⇒ 事后无法回答"到底是检索次数不够、
      还是命中条数被 `leansearch_top_k` 截了、还是压根没时间检索"。
    新增字段：
      · n_queries        —— 真正发起检索的次数（= 有 n_hits 的条目数）
      · per_query        —— 每次检索的命中条数序列（看分布，不只看总和）
      · top_k / top_k_capped          —— 单次命中是否顶到 leansearch_top_k
      · max_calls_per_q / calls_capped —— 单题调用次数是否顶到上限
      · skipped_time_critical / skipped_call_cap —— 两类跳过各几次
    """
    entries = [t for t in (getattr(ctx, "trace", None) or [])
               if isinstance(t, dict) and t.get("step") == "leansearch"]
    hits_seq = [int(e.get("n_hits", 0) or 0) for e in entries
                if "n_hits" in e]
    calls = len(hits_seq)
    hits = sum(hits_seq)
    ms = sum(float(e.get("elapsed_ms", 0.0) or 0.0) for e in entries)
    uniq = set()
    for e in entries:
        for n in (e.get("names") or []):
            if n:
                uniq.add(n)
    roots = [str(e.get("root", "")) for e in entries if e.get("root")]
    # 两类跳过的 record 文案（verifier._prepare_theorem_context 里各一处）
    skipped_time = sum(1 for e in entries
                       if "时间紧张" in str(e.get("content", "")))
    skipped_cap = sum(1 for e in entries
                      if "上限" in str(e.get("content", "")))
    cap_calls = top_k = None
    try:
        if cfg is not None:
            cap_calls = int(getattr(cfg, "leansearch_max_calls_per_q", 0) or 0)
            top_k = int(getattr(cfg, "leansearch_top_k", 0) or 0)
    except Exception:  # noqa: BLE001
        cap_calls = top_k = None
    return {
        "calls": calls,                 # 兼容旧字段名（= n_queries）
        "n_queries": calls,
        "hits": hits,
        "unique": len(uniq),
        "per_query": hits_seq,          # 每次检索命中条数（分布口径）
        "top_k": top_k or None,
        "top_k_capped": bool(top_k and hits_seq and max(hits_seq) >= top_k),
        "max_calls_per_q": cap_calls or None,
        "calls_capped": bool(cap_calls and calls >= cap_calls),
        "skipped_time_critical": skipped_time,
        "skipped_call_cap": skipped_cap,
        "elapsed_ms": round(ms, 1),
        "root": roots[-1][:120] if roots else "",
    }
