# -*- coding: utf-8 -*-
"""诊断导出（2026-10-07 阶段C1：从 agent/orchestrator.py 抽出 _collect_diag）。

逐步归因诊断的**唯一实现**：打包各阶段中间状态（理解→蓝图→骨架评审→子目标→
Lean→验证→预算），由 run_eval.solve_one 落盘为结果行 `diag` 字段。
**纯搬迁**：键名与取值口径与迁移前逐字一致（下游 tools/*.py 按键名消费，
改名/漏键 = 静默断链）。
依赖由调用方 `Orchestrator` 注入：config / lean_probe / lean_gate / stage_durs。
**2026-10-07 二次拆分**：collect_diag 已按语义拆为 `_diag_*` 模块级
helper（每个 ≤150 行），本函数只留编排 + 组装 dict（键名/取值口径一字不变）。
"""
from __future__ import annotations

import re as _re

from utils.extract import is_truncated_answer as _is_truncated_answer
from .param_usage import collect_param_usage
from .diag_metrics import (
    _merge_error_types, _sum_reject_votes, _compute_subgoal_gaps,
    _switch_snapshot, _tool_gateway_summary,
    _parse_exhaust_result, _diag_completeness, _summarize_deep_review,
    _summarize_leansearch,
)


def collect_diag(ctx, *,
                 config=None,
                 lean_probe=None,
                 lean_gate=None,
                 stage_durs=None) -> dict:
    """逐步归因诊断（2026-09-02 用户要求：错题要能定位到环节）。

    打包各阶段中间状态：理解→蓝图→骨架评审→子目标→Lean→验证→预算。
    全部 getattr + 默认值兜底：任意提前 return / 异常路径下调用都安全
    （TaskContext 各字段均有默认值，直接访问也不会崩）。
    由 run_eval.solve_one 落盘为结果行 diag 字段；tools/analyze_errors.py 消费。
    """
    _md = getattr(ctx, "metadata", None)
    _md = _md if isinstance(_md, dict) else {}
    # ⑤''' Lean 链路自证（2026-09-10 审核教训：结果文件没有 lean_active /
    #     lean_executable 就绝不能声称 Lean 跑过 —— 闸门身份会被静默降级
    #     顶换成 audit_gate 而输出看起来完全正常）。只读已算好的 _lean_probe，
    #     不在此触发新的环境探测。
    _lean_on, _lean_exe = _diag_lean_probe_state(lean_probe, lean_gate)
    _pv = getattr(ctx, "preverify_trace", None)
    _pv = _pv if isinstance(_pv, dict) else {}
    _counts_recon = _diag_counts_reconciliation(ctx)
    return {
        **_diag_identity_and_problem(ctx, _pv),
        **_diag_blueprint_review(ctx, config),
        **_diag_subgoal_selection(ctx),
        **_diag_pipeline_lean(ctx, _md, _lean_on, _lean_exe),
        **_diag_verdicts_metrics(ctx, config, _md, _counts_recon),
        **_diag_budget_health(ctx, stage_durs),
    }


def _diag_lean_probe_state(lean_probe, lean_gate):
    """Lean 链路自证状态（lean_probe 是否启用 + 桥接到的 lean 可执行文件）。"""
    _lean_on, _lean_exe = False, ""
    try:
        _lean_on = bool(lean_probe)
        if lean_gate is not None:
            _b = lean_gate._bridge_inst
            if _b is not None:
                _lean_exe = str(_b._lean_executable or "")
    except Exception:  # noqa: BLE001
        pass
    return _lean_on, _lean_exe


def _diag_counts_reconciliation(ctx) -> dict:
    """diag 计数的口径对账（n_candidates / n_verdicts / 选择池 的分母标注）。"""
    # ★ 2026-10-07（fix-gate-timing-diag 问题 2.6）：diag 计数**口径对账**。
    #   实测三处"对不上"：① 6/10 题 `n_candidates ≠ n_verdicts`
    #   （001: 3 vs 5；004: 3 vs 6；087: 7 vs 3）；② `pick_diag.divergent_clusters`
    #   各簇 size 之和 > `n_candidates`（001: 5>3；004: 6>3）；③ `pick_diag.picked`
    #   与任一簇代表不精确相等。根因是这三组数**属于不同时点/不同集合**而字段名
    #   未标注分母：
    #     · `n_candidates`   = 终态 `ctx.candidates`（6.5 重做/清洗后可能已裁剪）；
    #     · `n_verdicts`     = **最后一次复验**的快照（早于 6.5 裁剪）⇒ 两者天然可不等；
    #     · `divergent_clusters[].size` 是**终答五层输入池**（含子目标候选等）上的簇，
    #       正确分母 = `final_selection.n_in`，**不是** `n_candidates`
    #       （实测 001: Σsize=5=n_in；087: Σsize=3≤n_in=7 ⇒ 分母错了才"看起来矛盾"）；
    #     · `picked` 是**后处理/修复后**的终答（可能被 B0/多答案拼接改写），
    #       与簇代表的**原始**串不可精确比较（006 另有 `picked_pre_repair` 佐证）。
    #   以下 `counts_reconciliation` 把三个分母与"picked 是否落在池中"一并导出，
    #   使"据此复算选择过程"成为可能。纯读、不改任何选取行为。
    _pd = getattr(ctx, "_pick_diag", None)
    _pd = _pd if isinstance(_pd, dict) else {}
    _fsd = getattr(ctx, "_final_select_diag", None)
    _fsd = _fsd if isinstance(_fsd, dict) else {}

    def _nm_ans(s) -> str:  # 归一化：去空白/LaTeX 外壳，供"是否同答案"比较
        return _re.sub(r"[\s\\$]|boxed|\{|\}", "", str(s or ""))

    _clusters = _pd.get("divergent_clusters")
    _clusters = _clusters if isinstance(_clusters, list) else []
    _cluster_sizes = []
    for _c in _clusters:
        if not isinstance(_c, dict):
            continue
        try:
            _cluster_sizes.append(int(_c.get("size") or 0))
        except Exception:  # noqa: BLE001  纯埋点，脏数据不崩
            continue
    _picked_nm = _nm_ans(_pd.get("picked"))
    _picked_in_pool = bool(_picked_nm) and any(
        _nm_ans(c.get("answer")) and _nm_ans(c.get("answer")) == _picked_nm
        for c in _clusters if isinstance(c, dict))
    _sel_pool = _fsd.get("n_in")
    try:
        _sel_pool = int(_sel_pool) if _sel_pool is not None else None
    except Exception:  # noqa: BLE001
        _sel_pool = None
    _counts_recon = {
        "n_candidates_final": len(getattr(ctx, "candidates", None) or []),
        "n_verdicts_last_verify": len(getattr(ctx, "verdicts", None) or []),
        "selection_pool_size": _sel_pool,
        "divergent_clusters_count": len(_clusters),
        "divergent_clusters_size_sum": sum(_cluster_sizes),
        "pick_diag_picked_in_selection_pool": _picked_in_pool,
        "note": "n_verdicts 为最后一次复验快照（早于 6.5 裁剪）⇒ 可 ≠ n_candidates；"
                "divergent_clusters 的分母是 selection_pool_size（final_selection.n_in），"
                "不是 n_candidates；picked 为后处理终答，与簇原始串不必然精确相等。",
    }
    return _counts_recon


def _diag_identity_and_problem(ctx, _pv) -> dict:
    """① 题型与档位 + ② 题目理解（formula 语义订正 + preverify 轨迹）。"""
    return {
        # ① 题型与档位
        "question_type": getattr(ctx, "question_type", "") or "",
        "domain": getattr(ctx, "domain", "") or "",
        "tier": getattr(ctx, "tier", "") or "",
        "tier_evidence": getattr(ctx, "tier_evidence", None) or {},
        "soft_budget": round(float(getattr(ctx, "soft_budget", 0) or 0), 1),
        # ② 题目理解（formula 语义订正 2026-10-07，问题 2.7 ①）
        #   ⚠ 旧注释「preverify 已移除、字段恒空」**与事实不符**：preverify 仍在跑
        #     （`preverify_trace` 8/10 题有值）。真实语义是：
        #       · `formal_spec` = **被接受**的形式化（仅 verdict=ok 时写入，
        #         见 lean_pre_verifier.py:127/174）⇒ smoke10 全 = fail ⇒ 恒空；
        #       · `preverify_trace.formal_spec` = **最后一次尝试**（不管通过与否）。
        #     两者同名但语义不同，故同时导出 `formal_spec_status` 以杜绝误读。
        "formal_spec": (getattr(ctx, "formal_spec", "") or "")[:600],
        "formal_spec_status": {
            "accepted": bool(str(getattr(ctx, "formal_spec", "") or "").strip()),
            "preverify_verdict": str(_pv.get("verdict", "") or ""),
            "attempted": bool(_pv),
            "attempted_chars": len(str(_pv.get("formal_spec", "") or "")),
            "note": "accepted=False 且 attempted=True ⇒ 形式化运行过但未被接受"
                    "（verdict≠ok，属正常降级、非写入失败）",
        },
        "formal_gaps": list(getattr(ctx, "formal_gaps", None) or [])[:10],
        "preverify_trace": _pv or {},
    }


def _diag_blueprint_review(ctx, config) -> dict:
    """③ 蓝图 DAG 评审 + 被客观证伪答案。"""
    return {
        # ③ 蓝图 DAG 评审（DagReviewerAgent 写入，供 `blueprint_self_review` 消费）
        #   ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：
        #     原「骨架评审」`enable_skeleton_review`、「求解前 DAG 强制门」
        #     `dag_replan_gate`、「求解后 DAG replan」`enable_dag_replan` 三块
        #     已**整块撤除** ⇒ 其 diag 字段（skeleton_review / skeleton_review_events /
        #     dag_replan_events）与 `disabled_mechanisms` 中对应项一并移除。
        "blueprint_nodes": len((getattr(ctx, "blueprint", None) or {}).get("nodes", []) or []),
        "blueprint_merge": (getattr(ctx, "blueprint", None) or {}).get("merge_strategy", ""),
        "dag_review": getattr(ctx, "dag_review_report", None) or {},
        "sketch_audit": getattr(ctx, "sketch_audit", None) or {},
        "disabled_mechanisms": [
            _n for _n, _f, _d in (
                ("preverify_sketch_audit", "enable_preverify_sketch_audit", False),
            ) if not bool(getattr(config, _f, _d))
        ],
        # ★ 2026-10-12（判读失真 #5 治本）：自改进的结构化观测。
        #   此前报告读 `n_self_improved` 而无人写 ⇒ 恒 0 ⇒ 假「未触发」。
        # ★ 2026-10-12 批次 1（★ 2026-10-14 收敛）：题意理解（终态 + 答案槽 + 缺口 A/B）
        "understanding_chain": dict(
            getattr(ctx, "understanding_chain", None) or {}),
        "understanding_gaps": list(
            getattr(ctx, "understanding_gaps", None) or []),
        "understanding_rounds": int(
            getattr(ctx, "understanding_rounds", 0) or 0),
        # ★ 2026-10-12 批次 2：蓝图的多次生成轮数与逐轮差异（结构层 + 语义层）
        "blueprint_diffs": list(getattr(ctx, "blueprint_diffs", None) or []),
        "blueprint_gen_rounds": int(
            getattr(ctx, "blueprint_gen_rounds", 0) or 0),
        # ★ 2026-10-12 批次 3：子目标链 inputs 校验缺口（A=题面没给 / B=题面给了没读到）
        "subgoal_chain_gaps": list(
            getattr(ctx, "subgoal_chain_gaps", None) or []),
        # ★ 2026-10-12 批次 4：池空兜底汇总的错误点（用户 Q33：字段全含）
        "pool_empty_errors": list(
            getattr(ctx, "pool_empty_errors", None) or []),
        # ★ 2026-10-13 观测补齐（审核时这两项被判「无法判定」）：
        #   ① 蓝图自审事件（此前只写 trace、未导出 diag）
        "blueprint_self_review_events": [str(t.get("content", ""))[:200]
                                         for t in (getattr(ctx, "trace", None) or [])
                                         if isinstance(t, dict)
                                         and t.get("step") == "blueprint_self_review"],
        #   ② 分环节定理注入条数（stage → 实际条数）
        # ★ 2026-10-18：跨模型检错统计（wrong/ok/unknown 计数；未触发则为 None）
        "cross_model_stats": (dict(getattr(ctx, "cross_model_stats", None))
                              if isinstance(getattr(ctx, "cross_model_stats", None), dict)
                              else None),
        "theorem_hint_stage_usage": dict(
            getattr(ctx, "theorem_hint_stage_usage", None) or {}),
        # ★ 2026-10-13（§3 方案 A）：题型判题决策（rule/llm/overrides）+ 路径分流
        "type_decision": dict(getattr(ctx, "type_decision", None) or {}),
        "type_path": dict(getattr(ctx, "type_path", None) or {}),
        # ★ 2026-10-14（五道审核链第 0 道）：答案规范化前置的汇总 + 逐候选事件。
        #   汇总含形态分布（kinds）/ 抢救改写数（n_salvaged）/ 是否证明题。
        "answer_normalize": dict(
            getattr(ctx, "answer_normalize", None) or {}),
        "answer_normalize_events": [str(t.get("content", ""))[:200]
                                    for t in (getattr(ctx, "trace", None) or [])
                                    if isinstance(t, dict)
                                    and t.get("step") == "answer_normalize"],
        # ★ 2026-10-14（五道审核链第 1 道）：AI 自审逻辑错误（汇总 + 逐候选事件 +
        #   被淘汰清单，供「自审是否误杀」逐题复核）
        "self_audit": dict(getattr(ctx, "self_audit", None) or {}),
        "self_audit_rejected": list(
            getattr(ctx, "self_audit_rejected", None) or []),
        "self_audit_events": [str(t.get("content", ""))[:200]
                              for t in (getattr(ctx, "trace", None) or [])
                              if isinstance(t, dict)
                              and t.get("step") == "self_audit"],
        # ★ 2026-10-14（五道审核链第 2 道）：对抗证伪（汇总 + 淘汰清单 + 逐候选事件）
        "adversarial_gate": dict(getattr(ctx, "adversarial_gate", None) or {}),
        "adversarial_gate_rejected": list(
            getattr(ctx, "adversarial_gate_rejected", None) or []),
        "adversarial_gate_events": [str(t.get("content", ""))[:200]
                                    for t in (getattr(ctx, "trace", None) or [])
                                    if isinstance(t, dict)
                                    and t.get("step") == "adversarial_gate"],
        # ★ 2026-10-14（五道审核链第 3 道）：Lean 代回题目（汇总 + 逐候选事件）
        "lean_backcheck": dict(getattr(ctx, "lean_backcheck", None) or {}),
        "lean_backcheck_events": [str(t.get("content", ""))[:200]
                                  for t in (getattr(ctx, "trace", None) or [])
                                  if isinstance(t, dict)
                                  and t.get("step") == "lean_backcheck"],
        # ★ 2026-10-14（五道审核链第 4 道）：数学工具计算检测
        "math_check": dict(getattr(ctx, "math_check", None) or {}),
        "math_check_rejected": list(
            getattr(ctx, "math_check_rejected", None) or []),
        "math_check_events": [str(t.get("content", ""))[:200]
                              for t in (getattr(ctx, "trace", None) or [])
                              if isinstance(t, dict)
                              and t.get("step") == "math_check"],
        # ★ 2026-10-21：工具③证据快照（提取到的每条式子 + 逐条复算结论）。
        #   补的是「4.4 整池清空但无法归因」这个缺口 —— 候选 reasoning 不落盘，
        #   只留判词等于让人猜是真算错还是提取误伤。
        "math_calcverify_gate": dict(
            getattr(ctx, "math_calcverify_gate", None) or {}),
        # ★ 2026-10-14（五道审核链第 5 道）：投票（多答判断 + 票型表 + best 簇）
        "vote_gate": dict(getattr(ctx, "vote_gate", None) or {}),
        "vote_multi_answer": getattr(ctx, "vote_multi_answer", None),
        "vote_gate_events": [str(t.get("content", ""))[:200]
                             for t in (getattr(ctx, "trace", None) or [])
                             if isinstance(t, dict)
                             and t.get("step") == "vote_gate"],
        # ★ 2026-10-14（用户定案·重写通道）：4.6 池空重写（汇总 + 逐轮事件 + 诊断）
        # ★ 2026-10-15（工具化定稿）：工具①②③的观测留痕
        "lean_tool_check": list(getattr(ctx, "lean_tool_check", None) or []),
        "math_calcverify": dict(getattr(ctx, "math_calcverify", None) or {}),
        # ★ 2026-10-15（用户定案 · 三角色协作）：子目标协作求解全景
        "subgoal_collab": dict(getattr(ctx, "subgoal_collab_diag", None) or {}),
        "subgoal_blueprint_updates": list(
            getattr(ctx, "subgoal_blueprint_updates", None) or [])[:8],
        # ★ 2026-10-15（审核修复）：4.3 全拒标记此前"只写不读"（死状态）——
        #   接入 diag（全拒是关键事件，报告/归因需要看到）。
        "gate_all_rejected": bool(
            getattr(ctx, "_gate_all_rejected", False)),
        "rewrite_gate": dict(getattr(ctx, "rewrite_gate", None) or {}),
        "rewrite_context": str(getattr(ctx, "rewrite_context", None) or "")[:800],
        "rewrite_gate_events": [str(t.get("content", ""))[:200]
                                for t in (getattr(ctx, "trace", None) or [])
                                if isinstance(t, dict)
                                and t.get("step") == "rewrite"],
        "n_self_improved": int(getattr(ctx, "n_self_improved", 0) or 0),
        "self_improve_rounds_used": int(
            getattr(ctx, "self_improve_rounds_used", 0) or 0),
        "self_improve_events": [str(t.get("content", ""))[:200]
                                for t in (getattr(ctx, "trace", None) or [])
                                if isinstance(t, dict)
                                and t.get("step") == "self_improve"],
        # ★ 本轮优化埋点（2026-09-11，用户要求"一次评测即可逐项审查"）
        #   P2 数值化 / P5 客观题自检 / P4 阶段预算（control 中的预算记录）
        "numericize_events": [str(t.get("content", ""))[:200]
                              for t in (getattr(ctx, "trace", None) or [])
                              if isinstance(t, dict)
                              and t.get("step") == "numericize"],
        # ⚠ 2026-10-07（问题 2.7 ③）：`objective_check_events` 与相邻的
        #   `final_selection.objective` **不是同一机制**（同前缀造成误读）：
        #     · `objective_check_events`  ← P5 **客观题自检**（判断题写法归一 /
        #        选项越界告警，见 `_objective_selfcheck`）。它只在"真发生归一或
        #        越界"时才 record ⇒ smoke10 全空是**正确**的（无触发），非写入失败。
        #     · `final_selection.objective` ← 终答五层里的**客观验证层**
        #        （SymPy/回代淘汰），由 final_selector 产出，**不写 trace**。
        #   为消除"空数组却有汇总数"的表象，另导出等价的顶层 `objective_filter`
        #   （= final_selection.objective）并标注各自**触发口径**。
        "objective_check_events": [str(t.get("content", ""))[:200]
                                   for t in (getattr(ctx, "trace", None) or [])
                                   if isinstance(t, dict)
                                   and t.get("step") == "objective_check"],
        "objective_check_scope": "P5 客观题自检（判断题归一/选项越界告警；"
                                 "空 = 本题未触发，非失败）",
        "objective_filter": (
            dict(getattr(ctx, "_final_select_diag", {}).get("objective") or {})
            if isinstance(getattr(ctx, "_final_select_diag", None), dict) else None),
        "phase_budget_events": [str(t.get("content", ""))[:200]
                                for t in (getattr(ctx, "trace", None) or [])
                                if isinstance(t, dict)
                                and t.get("step") == "control"
                                and "阶段预算" in str(t.get("content", ""))],
        # 数值攻击证伪信号（此前未导出，导致 v1–v3 诊断时误判"无信号"）
        "blueprint_value_false": (str(getattr(ctx, "blueprint_value_false", "") or "")[:200]),
        # P1 未触发原因（2026-09-12 补：只导出 revise 事件时，"0 触发"在日志里
        # 表现为"什么都没记"，无法区分「没信号」与「有时间但被档位/emergency
        # /门槛拦」。此处单独导出，与 revise_events 并列、互不污染口径。
        "p1_check_events": [str(t.get("content", ""))[:220]
                            for t in (getattr(ctx, "trace", None) or [])
                            if isinstance(t, dict)
                            and t.get("step") == "p1_check"],
        # ★ 2026-09-23 新增：候选证伪器的**命中**记录（answer_falsifier）。
        #   必须导出 —— 否则"证伪器没生效"与"埋点缺失"无法区分
        #   （教训见 skill mathpilot-deepseek-error-audit §10.1）。
        #   ⚠ `falsify_events` = **命中**键（`step == "falsify"`，过滤条件与截断
        #   长度 220 **保持不变**）；**未命中**另有 `falsify_miss_events`（见下，
        #   `step == "falsify_miss"`）。
        #   2026-10-09 P0：此前只导出命中键 ⇒ `falsify_miss` 一条不落 ⇒
        #   "没调用 / 调了没命中 / 被前置条件挡住"三者不可区分（观测缺口）。
        "falsify_events": [str(t.get("content", ""))[:220]
                           for t in (getattr(ctx, "trace", None) or [])
                           if isinstance(t, dict)
                           and t.get("step") == "falsify"],
        # ★ 2026-10-09 P0 新增：`falsify_events` 的**显式命名**命中键。
        #   与 `falsify_events` **同源同长**（过滤条件与 220 截断逐字一致），
        #   仅提供语义化键名供下游按"命中/未命中"对称消费，口径零漂移。
        "falsify_hit_events": [str(t.get("content", ""))[:220]
                               for t in (getattr(ctx, "trace", None) or [])
                               if isinstance(t, dict)
                               and t.get("step") == "falsify"],
        # ★ 2026-10-09 P0 新增：证伪器**未命中**记录（新键）。
        #   证伪器每个失败出口都经 `orchestrator__oracle._falsify_rejects` 的
        #   `falsify_miss` 埋点留痕（含具体 reason：未取得证书 / 无可机检等式 /
        #   答案过长 / 时间不足 / 已达上限 / 异常 / 被前置条件挡住）。
        #   截断长度与 `falsify_events` 一致（220）。
        "falsify_miss_events": [str(t.get("content", ""))[:220]
                                for t in (getattr(ctx, "trace", None) or [])
                                if isinstance(t, dict)
                                and t.get("step") == "falsify_miss"],
        # 被客观证伪的候选答案原文（供审计"证伪器到底剔除了什么"）
        # ★ 2026-10-17 修复（p0b5_r3 实测崩溃）：本列表可能混入**字符串**
        #   （证伪淘汰链在不同路径 append 的形态不一）——原实现一律 `x.get`
        #   ⇒ AttributeError 在 diag 导出阶段抛出、整题结果被 fallback 吞成
        #   ERROR（2/10 题因此丢答案）。现按元素类型容错。
        "falsified_answers": [
            (str(x.get("answer"))[:160] if isinstance(x, dict)
             else str(x)[:160])
            for x in ((ctx.metadata or {}).get("falsified_answers") or [])
        ] if isinstance(getattr(ctx, "metadata", None), dict) else [],
    }


def _diag_subgoal_selection(ctx) -> dict:
    """子目标交付度/候选保留 + 终答五层选择观测 + 事件白名单 + verdicts 明细。"""
    return {
        # ★ 2026-09-23 新增：子目标「做成了没」的度量（用户判据：
        #   目标不是"跑通/有数据"，而是"有没有起到作用"）。
        #   0923 实测：279 个子目标里 88% 的 result < 120 字符（中位仅 30）、
        #   96% 没有 expected_output ⇒ 名义上"拆了"，实质上"没算"。
        #   该字段把"空壳率"逐题导出，使"阻断重做"有判据可依。
        "subgoal_delivery": {
            "n": len(getattr(ctx, "subgoal_trace", None) or []),
            "n_underdeliver_120": sum(
                1 for s in (getattr(ctx, "subgoal_trace", None) or [])
                if isinstance(s, dict) and len(str(s.get("result") or "")) < 120),
            "n_missing_expected_output": sum(
                1 for s in (getattr(ctx, "subgoal_trace", None) or [])
                if isinstance(s, dict)
                and not str(s.get("expected_output") or "").strip()),
        },
        # ★ 2026-10-02（阶段二-1：「淘汰」→「保留」）：**每个子目标保留了几条候选、
        #   代表解是第几条**（-1 = 无有效候选）。数据源 = `subgoal_trace[i]["candidates"]`
        #   （阶段二-1 新增；下游消费方式零改动）。用途：验证"候选真的被保留下"，
        #   并支撑后续「最终答案层一次性对比判断」的归因。
        "subgoal_candidates": [
            {"id": s.get("id"),
             "n_kept": len(s.get("candidates") or []),
             "n_answers": len(s.get("candidate_answers") or []),
             "picked_index": s.get("candidates_picked_index", -1),
             # ★ 2026-10-02（阶段二-2）：子目标层**客观验证**选中的候选下标
             #   （-1 = 客观验证判不出，沿用原代表解）。
             "verified_index": s.get("candidates_verified_index", -1)}
            for s in (getattr(ctx, "subgoal_trace", None) or [])
            if isinstance(s, dict)
        ],
        # ★ 2026-10-02（阶段二-2）：终答五层选择观测（②多答案/③客观验证/
        #   ④模型对比/⑤投票 各自是否触发、命中分支、执行到的层顺序）。
        #   `exit` = 最终落地的写出口（single_adopt = 池外直答单答案快速采纳，
        #   Q4 纳入后"终答唯一出口"可归因）。
        "final_selection": {
            **dict(getattr(ctx, "_final_select_diag", {}) or {}),
            "exit": getattr(ctx, "_final_select_exit", "") or "",
        },
        # ★ 2026-10-02（阶段二-2）：**子目标层客观验证**汇总（team-lead Q1 要求）——
        #   阶段三 A/B 用它归因「客观验证到底有没有挑出对的那个」。
        "subgoal_verify_summary": {
            "n_with_multi_candidates": sum(
                1 for s in (getattr(ctx, "subgoal_trace", None) or [])
                if isinstance(s, dict) and len(s.get("candidates") or []) >= 2),
            "n_judged": sum(
                1 for s in (getattr(ctx, "subgoal_trace", None) or [])
                if isinstance(s, dict)
                and s.get("candidates_verified_index", -1) >= 0),
            "n_undecided": sum(
                1 for s in (getattr(ctx, "subgoal_trace", None) or [])
                if isinstance(s, dict)
                and len(s.get("candidates") or []) >= 2
                and s.get("candidates_verified_index", -1) < 0),
        },
        # ★ 2026-09-12 补导出：符号化求解 / 独立符号复核 / 答案工具自检 的埋点。
        # 此前这些只写进 ctx.trace，而 diag 导出按固定字段挑 → jsonl 里看不到，
        # 导致"机制到底跑没跑"无法从结果文件判定（只能靠日志旁证）。
        "symbolic_solve_events": [str(t.get("content", ""))[:220]
                                  for t in (getattr(ctx, "trace", None) or [])
                                  if isinstance(t, dict)
                                  and str(t.get("step", "")).startswith(
                                      "symbolic_solve")],
        "symbolic_crosscheck_events": [str(t.get("content", ""))[:220]
                                       for t in (getattr(ctx, "trace", None) or [])
                                       if isinstance(t, dict)
                                       and str(t.get("step", "")).startswith(
                                           "symbolic_crosscheck")],
        "answer_selfcheck_events": [str(t.get("content", ""))[:220]
                                    for t in (getattr(ctx, "trace", None) or [])
                                    if isinstance(t, dict)
                                    and str(t.get("step", "")).startswith(
                                        "answer_selfcheck")],
        # ★ 表达式范式（9/12）：模型只建模、本地代入求值 → 答案取工具值。
        # 不加这行则 jsonl 里看不到"默认生成方程"是否真的被触发。
        "expression_eval_events": [str(t.get("content", ""))[:220]
                                   for t in (getattr(ctx, "trace", None) or [])
                                   if isinstance(t, dict)
                                   and str(t.get("step", "")).startswith(
                                       "expression_eval")],
        # 2026-09-12 补导出：一批「record 了但 diag 看不到」的事件统一补齐。
        # 本项目多次出现"埋点写了、jsonl 里找不到" ⇒ A/B 归因失效；本次新加的
        # 几处又踩了同一个坑（run_eval 只落盘 diag，trace 被丢弃），故一次补齐。
        **{f"{_k}_events": [str(t.get("content", ""))[:220]
                            for t in (getattr(ctx, "trace", None) or [])
                            if isinstance(t, dict) and t.get("step") == _k]
           for _k in ("final_postprocess_change", "subgoal_recover",
                      "subgoal_replan_needed", "lean_feedback_revise",
                      # ★ 2026-09-21（可观测性）：又是同一个坑 ——
                      #   `oracle_review`（4.5_oracle 的 AnswerOracle 复核结果，
                      #   orchestrator L2763-2774 共 3 处 record）与
                      #   `adversarial`（4.6_adv 的触发/跳过原因，L2694-2728 共 5 处）
                      #   都只写进 trace，而**不在本白名单** ⇒ jsonl/diag 里
                      #   一条都看不到。后果：无法从结果文件判定"4.5_oracle 跑了
                      #   几次、判了什么；4.6_adv 是跑了还是被窗口检查跳过"，
                      #   只能靠读日志猜（而日志里这两处恰好 0 条）。
                      #   ⇒ 补入白名单，导出 `oracle_review_events` /
                      #   `adversarial_events`。
                      "oracle_review", "adversarial",
                      # ★ 2026-09-30（截图 #8）：子目标多 agent 采样埋点。
                      #   不加这条 ⇒ 无法从结果文件区分「多 agent 没开启」
                      #   与「开启了但每次都无多数一致」—— A/B 归因失效。
                      "subgoal_multi_agent")},
        # revise 事件（含"P1 硬信号触发强制重解"/自纠错轮次/预算不足终止）
        # ——2026-09-11 补：此前仅 revise_round 可观测，看不到触发原因
        "revise_events": [str(t.get("content", ""))[:200]
                          for t in (getattr(ctx, "trace", None) or [])
                          if isinstance(t, dict) and t.get("step") == "revise"],
        # 2026-09-13 补导出（**今天多次被"看不到"卡住后的根治**）：
        # `control` 事件是"阶段为何跳过/触发"的唯一记录（如 3.3 自改进的
        # "跳过（剩余 X < 门槛）"），`verdicts` 是 4_verify 投票明细、
        # `audit_gate` 是客观审核明细 —— 三者此前都不在导出白名单里，
        # 导致归因时只能靠猜（本轮即因 `control_events` 缺失，无法判定
        # 004 的 3.3 为何从 363s 变成 0s）。
        "control_events": [str(t.get("content", ""))[:200]
                           for t in (getattr(ctx, "trace", None) or [])
                           if isinstance(t, dict) and t.get("step") == "control"],
        "verdicts": [
            {"correct_votes": getattr(v, "correct_votes", None),
             "total_votes": getattr(v, "total_votes", None),
             "correct": getattr(v, "correct", None),
             "confidence": getattr(v, "confidence", None)}
            for v in (getattr(ctx, "verdicts", None) or [])],
    }


def _diag_pipeline_lean(ctx, _md, _lean_on, _lean_exe) -> dict:
    """④ 子目标求解/值攻击/工具通道 + ⑤ 检测链 AuditGate / Lean 双通道自证。"""
    return {
        # ⚠ 2026-09-17：此处原先还有一条 `"audit_gate": [...][:6]`，与本文件
        #   下方（`_collect_diag` 的 ⑤ 段）**同名重复**。字典字面量里后键覆盖
        #   前键 ⇒ 那条"只留 dict、限 6 条"的实现**被静默丢弃**，一直是死代码。
        #   已删除，避免误读与 pyflakes `dictionary key repeated` 告警。
        # ④ 子目标求解（结构化轨迹）
        "subgoal_trace": getattr(ctx, "subgoal_trace", None) or [],
        "subgoal_merge_plan": (getattr(ctx, "subgoal_merge_plan", "") or "")[:300],
        # ④'' S4-lite 独立性指标（老师 9/6：子目标数/依赖边/平均入度/上下文注入量）
        "subgoal_stats": getattr(ctx, "subgoal_stats", None) or {},
        "subgoal_ctx_inject_chars": int(
            getattr(ctx, "subgoal_ctx_inject_chars", 0) or 0),
        "lemma_repo": list(getattr(ctx, "lemma_repo", None) or [])[:20],
        # ④' C-lite 数值攻击（2026-09-03 审核补：原本只 record 进 trace，
        # 而 trace 不落盘 → v17 事后完全查不到是否执行。现在进 diag。）
        "value_attack": [str(t.get("content", ""))[:200]
                         for t in (getattr(ctx, "trace", None) or [])
                         if isinstance(t, dict)
                         and t.get("step") == "value_attack"],
        # ★ 2026-09-18（可观测性）：**文本通道工具调用** —— 模型在正文/推理里
        #   手写 tool_call / function= / parameter= 这类**文本标记**，而平台未填
        #   原生 `tool_calls` ⇒ 调用**不执行、不回填**，答案退化为散文
        #   （实测 official112-025 的 5/5、032 的 4/4 候选）。
        #   该现象此前在 diag 中**完全不可见**（本键为新增）。
        "toolcall_text_detected": [
            str(t.get("content", ""))[:160]
            for t in (getattr(ctx, "trace", None) or [])
            if isinstance(t, dict)
            and t.get("step") == "toolcall_text_detected"][:20],
        # ⑦''' 工具循环审计（2026-10-01：原 calc 埋点随 `<calc>` 板块删除，
        # 改记 web_search 工具循环的模式与执行次数）
        "toolcall_mode": [str(t.get("content", ""))[:120]
                          for t in (getattr(ctx, "trace", None) or [])
                          if isinstance(t, dict)
                          and t.get("step") == "toolcall_mode"][:20],
        "toolcall_exec": [str(t.get("content", ""))[:120]
                          for t in (getattr(ctx, "trace", None) or [])
                          if isinstance(t, dict)
                          and t.get("step") == "toolcall_exec"][:20],
        # ⑤ 检测链 AuditGate（2026-09-06 去 Lean 化顶替 lean_gate；
        #    Level0-3 多级瀑布记录：候选过滤 / 最终答案门禁 / 理解确认；
        #    2026-09-06 晚 Lean 双通道恢复后，lean 不可用时的兜底记录）
        "audit_gate": [e for e in (getattr(ctx, "audit_gate", None) or [])][:60],
        # ⑤' Lean 双通道记录（2026-09-06 晚恢复：lean 环境可用时
        #     LeanGate/LeanPreVerifier 的候选过滤与最终闸门写这里，材料证据链）
        "lean_gate": [e for e in (getattr(ctx, "lean_gate", None) or [])][:60],
        # ⑤'' Lean 按题适用性（2026-09-10）：False = 本题答案非数学对象
        #     （选项字母/判断值/概念文字）→ 整题豁免 Lean、改路由 AuditGate。
        #     字段缺失（旧结果）视为 True，与 _lean_applicable 缺省口径一致。
        "lean_applicable": bool(_md.get("lean_applicable", True)),
        # ★ 2026-10-24：`_lean_active()` 每一步的失败原因（第 4 次 lean_active=false
        #   之后加的）—— 前四轮都只看到 false、看不到为什么，只能靠猜。
        "lean_skip_reason": (str(_md.get("lean_skip_reason", "") or "")
                             or str(getattr(ctx, "_lean_why", "") or "")),
        # ⑤''' Lean 链路自证：lean_active=True 且 lean_executable 非空，
        #      才可声称本题走过 Lean 通道（闸门身份 = diag.lean_gate 非空）。
        "lean_active": _lean_on,
        "lean_executable": _lean_exe,
    }


def _diag_verdicts_metrics(ctx, config, _md, _counts_recon) -> dict:
    """⑥ 候选/验证/自纠错/闸门终态 + 错误类型/检索/工具门面/参数/预算等汇总指标。"""
    return {
        # ⑥ 候选/验证/自纠错
        "n_candidates": len(getattr(ctx, "candidates", None) or []),
        "n_verdicts": len(getattr(ctx, "verdicts", None) or []),
        # ★ 2026-10-07（问题 2.6）：三组计数的**口径对账**（见函数内注释）。
        "counts_reconciliation": _counts_recon,
        # ★ 2026-10-07（问题 2.3 ①）：AcceptGate 终态。让 `HOLD` 不再是"无声通过"
        #   —— 读诊断即可区分「ACCEPT 采纳」与「HOLD 带疑提交（门未能裁定）」，
        #   并给出当时的阈值与 enforce 开关，使判定可复算。
        "accept_gate": dict(getattr(ctx, "_accept_gate_final", None) or {}) or None,
        "revise_round": getattr(ctx, "revise_round", 0),
        "revise_feedback": list(getattr(ctx, "revise_feedback", None) or [])[:20],
        # ★ 2026-09-21（可观测性）：6.5 审核闸门的**整题终态**标量。
        #   此前只有 `lean_gate` / `audit_gate` 里的逐条 `final_gate` 事件，
        #   没有"本题最终是否被拒 / 换过哪些候选 / 是否回滚"的汇总字段 ⇒
        #   事后要重建结论必须逐事件扫（0921 巡检即因此多绕了几步）。
        #   `gate_rollback` 与 0921 的回滚修复配对，用于 A/B 计数。
        "gate_rejected": bool(getattr(ctx, "gate_rejected", False)),
        "gate_tried": list(getattr(ctx, "_gate_tried", None) or []),
        "gate_rollback": bool(getattr(ctx, "_gate_rollback", False)),
        # 2026-09-13 P0：答案选取埋点（候选分布 + verdict 明细 + 命中分支）
        "pick_diag": getattr(ctx, "_pick_diag", None) or {},
        # B0：答案形态闸门事件（从 trace 中挑出，条数少，便于事后判定触发情况）
        "answer_form_events": [
            str(t.get("content")) for t in (getattr(ctx, "trace", None) or [])
            if isinstance(t, dict) and t.get("step") == "answer_form"
        ][:10],
        # ★ 2026-09-15（用户要求"把错误暴露得更具体"）：验证器判 B 时的
        # **结构化错误类型分布** {标签: 票数}，标签集 = verifier.ERROR_TYPE_TAGS
        # （符号错/计算错/边界遗漏/定义域错/跳步/循环论证/方法不适用/
        #  前提不成立/方向反了/漏分支/其它），回应李平老师"是否硬套定理"。
        # 口径优先取 `verify_error_types`（整题终态），无则回退
        # `vote_error_types`（单次投票）。两者都是纯埋点，不影响判定。
        "error_types": _merge_error_types(ctx),
        "n_reject_votes": _sum_reject_votes(ctx),
        # ★ 2026-09-15：LeanSearch 检索埋点（老师 #44）——本题的
        # 调用次数 / 命中数 / 去重后条数 / 耗时 / 后端 root。
        # `used_theorems`（run_eval 已导出）是命中定理名清单，两者互为交叉验证。
        "leansearch": _summarize_leansearch(ctx, config),
        # ★ 2026-09-29（截图 #3+#4）：领域→定理检索埋点。
        #   与上面的 `leansearch` 分开：后者是**验证期**的引理检索，
        #   本项是**理解期**的定理预取，两者用途不同、不该混算。
        #   含 query / 命中全名 / 耗时 / 后端 / （若注入标注）命中率。
        "theorem_hint": dict(getattr(ctx, "theorem_hint_trace", None) or {}),
        # ★ 2026-10-02（阶段一：子目标结论 → 主求解 信息流）：主求解是否
        #   真的用上了子目标结论 —— 注入的**条数 / 字符数**。
        #   口径：{count = 有非空 result 的子目标条数；chars = 注入块长度}。
        #   缺键 / count=0 ⇒ 未注入（无子目标结论 / 开关关闭）；便于 A/B 归因，
        #   使"主求解建立在子目标之上"这件事可被结果文件核验（而非靠日志猜）。
        "subgoal_findings_injected": dict(
            _md.get("subgoal_findings_injected", {}) or {}),
        # ★ 2026-09-29（截图 #6）：**逻辑缺口分析** —— 回答用户
        #   「缺的逻辑点是不是大模型需要推理出来的点」。
        #
        # 口径说明（重要，勿混算）：
        #   · `subgoal_gaps` 消费 `ctx.subgoal_trace`（自然语言求解视角）——
        #     每步子目标有没有产出有效结论，失败时按报错性质分五类；
        #   · 其中 `reasoning_gaps` 才是**真·推理瓶颈**（前提齐全却推不出）；
        #     `non_reasoning_gaps` 是形式化/检索/计算问题，**不该计入推理能力评估**。
        #   这个区分是本埋点的全部价值：不区分的话「大量子目标失败」会被
        #   笼统读成"推理能力差"，而实际可能大半是译题错误 —— 那会把优化
        #   方向引到完全错误的地方。
        "subgoal_gaps": _compute_subgoal_gaps(ctx),
        # ★ 2026-10-15：`lean_gaps` 已随「求解前 Lean 逻辑检查」撤除 ——
        #   `_lean_dag_logic_check` 的唯一调用者是默认关闭的 `dag_replan_gate`，
        #   该字段在生产中恒为空。`subgoal_gaps` 不受影响，仍为现役。
        # ★ 2026-09-30（截图 #10）：**统一工具调用遥测**。
        #   用户诉求「调用工具的方法有没有写成规范性的类函数，需要 Lean 检测时
        #   直接调用，适配各阶段」—— `agent/tool_gateway.py` 是那个统一入口，
        #   这里把它的调用统计落盘，使诊断报告的「工具使用」维度
        #   有**单一数据源**（此前只能靠遍历 trace 猜）。
        #   口径说明：
        #     · `total`      = 本门面被调用次数（**不等于**流水线全部工具调用，
        #                      因为只有走门面的调用才计数）；
        #     · `by_reason`  = **闭合原因码**分布，可直接聚合归因；
        #     · 关键区分：`unavailable`（环境没装）≠ `reject`（跑了判否）
        #                      ≠ `error`（调用炸了）—— 三者修法完全不同。
        #   ⚠ 若本字段为全零，**只说明"没有代码走门面"**，
        #     绝不可读成"本轮没调用工具"（既有调用点尚未迁移）。
        "tool_gateway": (_tool_gateway_summary(ctx)),
        # 2026-10-01 开关注册制：26 个开关的生效值快照（含来源 env/config/default）
        # ⇒「生效查不到」问题根治：任何一轮评测都能从 diag 还原开关状态。
        "switch_snapshot": _switch_snapshot(),
        # ★ 2026-09-21（可调参数实测记录）：把「配置上限 / 实际用了多少 / 有没有撞顶」放进同一张表，随 diag 落盘。
        #   用户诉求：「cleansearch 到底要找多少定理、检测打回要搞多少次、无条件重做多少次、子目标要设多少……我们要测出最合理的数据」（不止这几个）。
        #   明细与拓展方法见 agent/param_usage.py 的模块文档。
        "param_usage": collect_param_usage(ctx, config),
        # ★ 2026-09-15：带推理复核的判定（独立信号，不并入投票统计）。
        # 第一轮实测因缺此字段，无法判断复核"跑了没、判了什么"。
        "deep_review": _summarize_deep_review(ctx),
        # 穷尽性搜索机制：是否追加了「解族穷尽性检查」子目标
        "exhaust_diag": getattr(ctx, "_exhaust_diag", None) or {},
        # ★ 2026-09-17 新增：解析该子目标的输出，判断它**是否真的做了检查**。
        #   此前 result 是自由文本 ⇒ 无法区分"确认只有一解"与"没做检查"。
        #   `complete=False` 即可机检的"未完成检查"信号。
        "exhaust_result": _parse_exhaust_result(
            getattr(ctx, "subgoal_trace", None) or []),
    }


def _diag_budget_health(ctx, stage_durs) -> dict:
    """⑦ 预算健康（budget_skip/degraded/占位符）+ 阶段计时 + 埋点三态自查。"""
    return {
        # ⑦ 预算健康（trace 中 budget_skip / degraded / 占位符计数）
        "budget_skips": sum(1 for t in (getattr(ctx, "trace", None) or [])
                            if isinstance(t, dict) and t.get("step") == "budget_skip"),
        "degraded_flags": sum(1 for t in (getattr(ctx, "trace", None) or [])
                              if isinstance(t, dict)
                              and ("degrad" in str(t.get("step", "")).lower()
                                   or "degrad" in str(t.get("content", "")).lower())),
        "placeholder": "[子目标求解失败]" in (getattr(ctx, "final_response", "") or ""),
        # 2026-09-02 老师需求：答案截断可观测（003 题 g(x)=-2x^{ 截断被识别）
        "answer_complete": not _is_truncated_answer(
            getattr(ctx, "final_response", "") or ""),
        # 阶段耗时（2026-09-03 老师要看 deep 档每环节具体耗时）
        "stage_timers": stage_durs(ctx),
        # ★ 2026-09-17（P1-F）：埋点三态自查 —— 未执行 / 执行无产出 / 有产出。
        # 解决「键存在但值为空 ⇒ 无法判断该环节跑没跑」的归因盲区。
        "diag_completeness": _diag_completeness(ctx),
    }
