#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage_audit.py — 环节审计：每个阶段/机制的「原定目的 → 完成判据 → 实测结论」

为什么需要它
    `cloud_health_report.py` 的 C 节只报「触发次数」，但**次数为 0 有两种含义**：
      · 设计上就该跑却没跑 ⇒ **真 bug**
      · 设计上关闭 / 前置条件不满足 ⇒ **正常**
    只看次数无法区分，会把「正常」误报成「失效」（反之亦然）。
    本脚本对每个环节给出**原定目的**与**可达判据**，再判 ✅/⚪/❌ 三态。

结论三态
    ✅ 完成     设计上应跑，实测有产出
    ⚪ 设计如此  开关默认关 / 前置条件结构性不满足（非缺陷）
    ❌ 未完成   设计上应跑，实测无产出（需查因）

用法
    python tools/stage_audit.py <run.jsonl> [--log <run.log>] [--out <x.md>]
"""
# 2026-10-01 去同名：load -> load_jsonl_rows（读 JSONL 成行列表；原名 load 太泛，无法自解释）
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict

# ---------------------------------------------------------------- 阶段意图表
# (阶段名, 中文名, 原定目的, 完成判据, 预期态)
#   预期态：'run' = 设计上应跑；'cond' = 条件触发（凭条件判）；'off' = 设计上关闭
#
# ★★ 判据必须用「产出物」而不是「耗时」——
#    `_stage_start` 的计时是「本阶段开始 → 下一阶段开始」，本身极快的阶段
#    （如 6_format 纯字符串处理）会记成 0.0s，**不能据此判定「没跑」**。
#    2026-09-21 首版误把 6_format / 0_paper_pacer 判成「未完成」，已修正。
STAGES_stage_audit = [
    # ★ 2026-10-02 移除 `0_paper_pacer`（阶段已删，用户 2026-10-02 点名「全卷配速删了」）；
    #   阶段名是稳定契约，下方按名分派的 handler 保留，供**历史 jsonl** 兼容渲染。
    # ★ 2026-10-02 移除 `1_classify` / `2.5_difficulty`（前者阶段已删、后者统一档位后
    #   空转）；此前报告仍含这两个阶段名（留痕）。下方按名分派的 handler 保留，
    #   供**历史 jsonl** 兼容渲染。
    # ★ 2026-10-02 补登 `1.2_theorem_hint`（当时为现行阶段，此前审计清单漏登
    #   ⇒ 其耗时从未进审计/报告）；★ 2026-10-14（任务列表 #2「定理检索移到后面
    #   子目标求解过程中」）**撤下** —— 检索已下移为 `2.7_subgoal_main` 体首动作，
    #   不再有独立阶段行，其耗时并入 2.7。历史 jsonl 的 1.2 行仍可渲染
    #   （阶段名是稳定契约，按名分派 handler 一律保留）。
    ("2.6_pre_audit", "题意理解（含题型判题）",
     "① 题型判题：LLM 判题型 + 规则交叉校验，产出 ctx.type_decision / ctx.type_path；"
     "② 题意理解：Lean 前置形式化（题目转 Lean 声明编译校验，失败带错强制重审）",
     "type_decision 非空（题型判题恒跑）；Lean 前置形式化另见日志侧证据", "run"),
    # ★ 2026-10-02 移除 `2.65_calc_prewarm`（calc 板块 2026-10-01 整体删除时的漏网残留）；
    #   其 D 类兜底 handler 保留，供**历史 jsonl** 兼容渲染。
    ("2.7_subgoal_main", "子目标主循环（含定理检索）",
     "体首先做 Mathlib 定理检索（原 1.2 阶段，2026-10-14 下移）+ 子目标分解"
     " + 逐步求解 + 结论合并（v2.9 主路径）",
     "subgoal_trace 非空", "run"),
    ("3_solve", "主体求解",
     "生成候选解答（deep 档先走 Plan-and-Execute）",
     "candidates 非空", "run"),
    ("3.2_complete", "截断候选续写",
     "对被 max_tokens 截断的候选补写完整（每档 max_completions 个）",
     "候选数 > 初始候选数", "cond"),
    ("3.3_improve", "无条件自改进",
     "生成后、验证前先 review+improve 一遍（IMO2025 论文：初始解质量低，此步显著改进；本地实测 22.2%→31.1%）",
     "control_events 含 3.3/improve 记录", "run"),
    ("3.4_collab", "三 Agent 协作",
     "deep 档难题：协作（解题→审查→整合→反复验证）",
     "control_events 含「三Agent协作」记录", "cond"),
    # ★★★ 2026-10-14 五道审核链（用户定案）：规范化前置 → ①自审 → ②对抗证伪
    #   → ③Lean 代回 → ④计算检测 → ⑤投票。以下 6 行为现行验证链。
    ("4.0_normalize", "答案规范化前置（五道链第 0 道）",
     "候选进验证链前统一归一（answer_norm/answer_kind），供 Lean/计算工具/投票消费",
     "answer_normalize 非空", "run"),
    ("4.1_self_audit", "AI 自审逻辑错误（第 1 道）",
     "逐候选 LLM 逻辑自审（只审逻辑不审数值），能指出具体逻辑错误者淘汰",
     "self_audit 非空", "run"),
    ("4.2_falsify", "AI 对抗证伪（第 2 道）",
     "逐存活候选两层对抗验证：机检证伪（SymPy 反例）+ LLM 对抗生成；证伪即判错",
     "adversarial_gate 非空", "run"),
    ("4.3_lean_check", "Lean 代回题目（第 3 道）",
     "候选级 LeanGate.apply：题目+答案翻译为 Lean 命题编译判定（答案带回题目）",
     "lean_backcheck 非空", "run"),
    ("4.4_math_check", "数学工具计算检测（第 4 道）",
     "候选级 SymPy 精确判定（等式回带/函数方程回代/反例搜索）；fail 淘汰",
     "math_check 非空", "run"),
    ("4.5_vote", "投票（第 5 道）",
     "先多答案判断（ctx.vote_multi_answer）→ 大模型对存活候选逐候选投票",
     "vote_gate 非空 或 verdicts 非空", "run"),
    # ★ 2026-10-14 二次定案（用户）：「不要池空直答！没有答案了就讲错误情况返回
    #   给大模型，让它找到问题出在哪里，重新构建蓝图和子目标，重写」。
    ("4.6_rewrite", "池空重写（用户定案）",
     "池空 ⇒ 错误情况汇总 → LLM 诊断根因 → 重建蓝图和子目标（rewrite_context "
     "注入题目占位符内部）→ 重写 → 新候选重过五道链（轮数上限 rewrite_max_rounds）",
     "rewrite_gate 非空（池空才触发）", "off"),
    ("3.5_subgoal_sup", "子目标补充候选",
     "非 deep 档候选不足时补一轮子目标分解（**已被 2.7 取代**）",
     "control_events 含「子目标分解补充候选」", "off"),
    # ★★ 2026-10-14（五道链定案）：以下 7 行为**已删阶段的历史行**（预期态 off）
    #   —— 阶段名是稳定契约，保留行供历史 jsonl 兼容渲染；新 run 中渲染为「设计如此」。
    ("3.6_audit_filter", "候选客观审核（已删 2026-10-14）",
     "Lean 双通道 + AuditGate 串行审核候选（职责拆入 4.3_lean_check / 4.4_math_check）",
     "lean_gate 非空", "off"),
    ("4_verify", "验证器投票（已删 2026-10-14）",
     "多候选投票定对错（投票并入 4.5_vote；P1 硬信号重解已删）",
     "verdicts 非空", "off"),
    ("4.5_oracle", "Oracle 客观复核（已删 2026-10-14）",
     "deep 档 AnswerOracle 客观复核 best_cluster（用户口径：其余都删）",
     "control/oracle 记录 或 stage 耗时 > 0", "off"),
    ("4.6_adv", "对抗式证伪（升级为 4.2_falsify）",
     "正向通过后主动证伪（原只探最佳簇一次；现逐候选化）",
     "stage 耗时 > 0", "off"),
    ("5_revise_or_fallback", "修订/兜底（已删 2026-10-14）",
     "全 0 票时 revise 回环（用户决策②：重解循环全删，全灭走 emergency 直答）",
     "revise_round > 0 或兜底记录", "off"),
    ("5.5_low_conf", "低置信强制复核（已删 2026-10-14）",
     "best_cluster 置信度 < 0.5 时强制复核（随 5_revise 一并删除）",
     "stage 耗时 > 0", "off"),
    ("6_format", "格式化输出",
     "最终响应格式化（纯字符串处理，本身极快 ⇒ 耗时必然≈0）",
     "response_full 非空", "run"),
    ("6.5_audit_gate", "最终答案闸门（已删 2026-10-14）",
     "最终答案过 Lean/AuditGate 双后端（Lean 判断统一在投票前的 4.3；"
     "emergency 直答兜底保留在 formatter）",
     "audit_gate 含 final 记录", "off"),
]

# ---------------------------------------------------------------- 机制意图表
# (diag 字段, 中文名, 原定目的, 预期态)
MECHS = [
    # ★ 2026-10-14 五道审核链新增机制键（与 diag_export 导出键逐字一致）
    ("answer_normalize", "答案规范化前置", "4.0：候选归一与形态分流（含不可判形态剔除）", "run"),
    ("self_audit", "AI 自审", "4.1：逐候选逻辑自审（淘汰+反馈）", "run"),
    ("adversarial_gate", "对抗证伪", "4.2：机检证伪 + LLM 对抗生成（逐候选）", "run"),
    ("lean_backcheck", "Lean 代回", "4.3：候选级 Lean 命题编译判定", "run"),
    ("math_check", "计算检测", "4.4：SymPy 精确判定（fail 淘汰）", "run"),
    ("vote_gate", "五道链投票", "4.5：多答案判断 + LLM 投票（票数 tier_votes）", "run"),
    ("rewrite_gate", "池空重写", "4.6：错误诊断 → 重建蓝图/子目标 → 重写（池空触发）", "cond"),
    ("lean_gate", "Lean 硬验证闸门", "编译验证候选/答案，淘汰 proof_invalid 并收 revise 反馈", "cond"),
    ("audit_gate", "审计闸门", "客观审核（随旧链删除后仅剩 2.6 复核路径；候选级 Level0 已并入 4.4）", "cond"),
    ("control_events", "流程控制事件", "记录兜底/协作/触发等控制流决策", "run"),
    ("revise_events", "修订事件（已删）", "验证不过时的定向修正记录（随 5_revise 删除）", "off"),
    ("skeleton_review_events", "骨架评审", "对蓝图骨架做多轮评审（arm2 专用）", "off"),
    ("dag_replan_events", "DAG 重规划", "依赖图变化时重规划（arm2 专用）", "off"),
    ("subgoal_trace", "子目标轨迹", "子目标分解的执行轨迹", "run"),
    ("answer_form_events", "答案形式检查", "答案格式/形式合规检查", "run"),
    ("p1_check_events", "P1 检查（已删）", "候选质量一级检查（随 4_verify 删除）", "off"),
    ("phase_budget_events", "阶段预算事件", "各阶段预算的收紧/恢复记录", "run"),
    ("toolcall_text_detected", "伪工具调用检测", "模型在正文里「假装」调用工具的检测", "run"),
    ("revise_feedback", "修订反馈（已删）", "喂给修订的反馈文本（随 5_revise 删除）", "off"),
    ("value_attack", "数值攻击", "对候选做数值反例攻击（C-lite）", "cond"),
    ("exhaust_diag", "穷尽性诊断", "枚举/穷尽性搜索的可行性诊断", "cond"),
    ("deep_review", "深度评审", "验证器的深度复核（deep_final）", "cond"),
    ("formal_gaps", "形式化缺口", "Lean 形式化中未覆盖的缺口记录", "cond"),
    ("lemma_repo", "引理库", "累积的引理仓库（use_lemma_accumulation 关闭时为空）", "off"),
    ("toolcall_mode", "工具调用模式", "记录工具循环的启用模式", "run"),
    ("toolcall_exec", "工具调用执行", "工具循环里实际执行的调用", "cond"),
    ("numericize_events", "数值化", "把符号问题数值化以便验证", "cond"),
    ("objective_check_events", "客观检查", "客观数值回代检查", "cond"),
    ("expression_eval_events", "表达式求值", "表达式求值验证", "cond"),
    ("final_postprocess_change_events", "终稿后处理改写", "格式化阶段对答案的改写记录", "cond"),
    ("lean_feedback_revise_events", "Lean 反馈修订", "把 Lean 错误反馈给修订回环", "cond"),
    ("subgoal_l0c", "子目标 L0c", "子目标一级检查", "cond"),
    ("subgoal_recover_events", "子目标恢复", "子目标失败后的恢复", "cond"),
    ("subgoal_replan_needed_events", "子目标重规划", "子目标需重规划的判定", "cond"),
    ("symbolic_crosscheck_events", "符号交叉核对", "符号解与数值解交叉核对", "cond"),
    ("symbolic_solve_events", "符号求解", "符号求解通道（symbolic_solve_enabled）", "off"),
    ("dag_review", "DAG 评审", "蓝图依赖图评审", "off"),
    ("preverify_trace", "预验证轨迹", "前置形式化的细粒度轨迹", "cond"),
    ("sketch_audit", "骨架审计", "骨架/草图的审计（LEAP Stage2/3，未接入）", "off"),
]


def load_jsonl_rows(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return rows


def _log_evidence(key: str, logtxt: str) -> tuple[int, str]:
    """从**日志**取该阶段的执行痕迹（jsonl 埋点缺失时的兜底判据）。

    ★ 2026-09-21 教训：不能只靠 jsonl。`2.6_pre_audit` 的
      `LeanPreVerifier.run()` 成功路径**不写 record**，jsonl 侧看起来「没跑」，
      但日志里明明有 `走 mcp 后端：preverify_*.lean`。
      只信 jsonl 会把「在跑」误判成「未跑」。
    返回 (命中次数, 命中的模式描述)。
    """
    if not logtxt:
        return 0, ""
    PATS = {
        "2.6_pre_audit": [r"走 mcp 后端：preverify_",
                          r"前置形式化", r"lean_preverify"],
        "2.65_calc_prewarm": [r"prewarm", r"预计算"],
        "3.4_collab": [r"三Agent协作", r"协作", r"collab"],
        "3.5_subgoal_sup": [r"子目标分解补充候选"],
        "4.5_oracle": [r"oracle", r"Oracle"],
        "5.5_low_conf": [r"低置信"],
        "0_paper_pacer": [r"软预算帽", r"paper_pacer"],
    }.get(key, [])
    for p in PATS:
        n = len(re.findall(p, logtxt))
        if n:
            return n, p
    return 0, ""


def judge_stage(key, rows, ts, logtxt):
    """按**产出物**判定阶段是否完成，返回 (实测描述, 结论)。

    ★ 不用耗时做判据：`_stage_start` 计的是「本阶段开始→下一阶段开始」，
      本身极快的阶段（6_format 纯字符串处理）必然记成 0.0s。
    ★ jsonl 判不出时用**日志侧痕迹**兜底（见 `_log_evidence`）。
    """
    n = len(rows)
    # ★ 2026-10-14：4.6_rewrite 为**条件阶段**（仅池空才启动 ⇒ 多数题无 stage_timer）。
    #   无计时不代表未完成 ⇒ 按 diag.rewrite_gate 判：有产出=完成，无产出=设计如此。
    if key == "4.6_rewrite" and not ts:
        _wg = sum(1 for r in rows
                  if (r.get("diag") or {}).get("rewrite_gate"))
        return ("未触发（池空才跑）；diag.rewrite_gate 命中 %d/%d 题"
                % (_wg, n),
                "完成（diag 有产出）" if _wg else "设计如此")
    if not ts:
        return "无该阶段埋点", "未完成"
    med, mx, nz = st.median(ts), max(ts), sum(1 for x in ts if x > 0.05)

    def cnt(pred):
        return sum(1 for r in rows if pred(r))

    d = lambda r: (r.get("diag") or {})  # noqa: E731
    ctrl = lambda r, s: any(  # noqa: E731
        s in str(e) for e in (d(r).get("control_events") or []))

    ok = None
    extra = ""
    if key == "0_paper_pacer":
        # ★ soft_budget 在 diag 里，不在顶层（首版从顶层取 ⇒ 误判为未跑）
        ok = cnt(lambda r: ctrl(r, "paper_pacer") or d(r).get("soft_budget") is not None)
        extra = "(soft_budget 已设置)"
    elif key == "1_classify":
        ok = cnt(lambda r: bool(d(r).get("question_type")))
    elif key == "2.5_difficulty":
        ok = cnt(lambda r: bool(d(r).get("tier")))
    elif key == "2.6_pre_audit":
        # ★ 2026-10-14（任务列表 #1：题型判题并入题意理解）：本阶段现以
        #    `type_decision` 为**必产**证据（判题是题意理解的第一步，恒跑）。
        # ⚠ 原判据缺口仍在：Lean 前置形式化成功时**不写 record**、`preverify_trace`
        #    实测恒空 ⇒ 该子步骤的 jsonl 侧无判据，真实证据在服务器
        #    `deploy/mathlib-olean/_lean_trash/preverify_*.lean` 与日志
        #    「走 mcp 后端：preverify_*」（由 `_log_evidence` 兜底）。
        _td = cnt(lambda r: bool(d(r).get("type_decision")))
        ok = _td
        extra = ("(type_decision 命中 %d/%d；Lean 前置形式化见日志侧证据)"
                 % (_td, n))
    elif key == "2.65_calc_prewarm":
        ok = cnt(lambda r: bool(d(r).get("calc_prewarm_events")))
    elif key == "2.7_subgoal_main":
        ok = cnt(lambda r: bool(d(r).get("subgoal_trace")))
    elif key == "3_solve":
        ok = cnt(lambda r: bool(r.get("candidates")))
    elif key == "3.2_complete":
        ok = cnt(lambda r: len(r.get("candidates") or []) > 1)
        extra = "(>1 候选)"
    elif key == "3.3_improve":
        ok = cnt(lambda r: ctrl(r, "3.3") or ctrl(r, "improve") or nz > 0)
    elif key == "3.4_collab":
        ok = cnt(lambda r: ctrl(r, "三Agent协作") or ctrl(r, "3.4"))
    elif key == "3.5_subgoal_sup":
        ok = cnt(lambda r: ctrl(r, "子目标分解补充候选"))
    elif key == "3.6_audit_filter":
        ok = cnt(lambda r: bool(d(r).get("lean_gate")))
    elif key == "4_verify":
        ok = cnt(lambda r: bool(r.get("verdicts")))
    elif key == "4.5_oracle":
        ok = nz
    elif key == "4.6_adv":
        ok = nz
    elif key == "5_revise_or_fallback":
        ok = cnt(lambda r: (d(r).get("revise_round") or 0) > 0)
    elif key == "5.5_low_conf":
        ok = nz
    elif key == "6_format":
        ok = cnt(lambda r: bool(r.get("response_full") or r.get("response")))
        extra = "(最终响应非空)"
    elif key == "6.5_audit_gate":
        ok = cnt(lambda r: bool(d(r).get("audit_gate")))
    # ★ 2026-10-14 五道审核链（新增阶段判据；1.1 为 v4 补登）
    elif key == "1.1_type_understanding":
        ok = cnt(lambda r: bool(d(r).get("type_decision")))
    elif key == "4.0_normalize":
        ok = cnt(lambda r: bool(d(r).get("answer_normalize")))
    elif key == "4.1_self_audit":
        ok = cnt(lambda r: bool(d(r).get("self_audit")))
    elif key == "4.2_falsify":
        ok = cnt(lambda r: bool(d(r).get("adversarial_gate")))
    elif key == "4.3_lean_check":
        ok = cnt(lambda r: bool(d(r).get("lean_backcheck")))
    elif key == "4.4_math_check":
        ok = cnt(lambda r: bool(d(r).get("math_check")))
    elif key == "4.5_vote":
        ok = cnt(lambda r: bool(d(r).get("vote_gate")) or bool(r.get("verdicts")))
    else:
        ok = nz

    # 日志侧兜底
    loghit = ""
    if logtxt and ok == 0:
        for pat in ("paper_pacer", "calc_prewarm", "三Agent协作"):
            if key.split("_")[0] in pat and pat in logtxt:
                loghit = "（日志有痕迹）"
    logn, logpat = _log_evidence(key, logtxt)
    if ok is None:
        # jsonl 侧无判据 ⇒ 用**日志侧痕迹**兜底
        if logn:
            return ("耗时中位 %.1fs；**jsonl 无判据，但日志命中 `%s` %d 次** ⇒ 实际在跑；"
                    "max %.1fs" % (med, logpat, logn, mx),
                    "完成（日志侧证据）")
        return ("耗时中位 %.1fs；max %.1fs；**jsonl 与日志均无判据**" % (med, mx),
                "无法判定")
    meas = "耗时中位 %.1fs；**产出 %d/%d 题**%s；max %.1fs" % (med, ok, n, extra, mx)
    if logn:
        meas += "；日志命中 `%s` %d 次" % (logpat, logn)
    if ok >= n * 0.5:
        v = "完成"
    elif ok > 0:
        v = "部分完成"
    else:
        # ★ jsonl 说 0，但日志有痕迹 ⇒ 埋点缺口，不是功能失效
        v = ("完成（jsonl 无埋点，日志侧 %d 次）" % logn) if logn else "未完成"
    return meas, v


def main() -> int:
    ap = argparse.ArgumentParser(description="环节审计：目的 → 判据 → 结论")
    ap.add_argument("jsonl")
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_jsonl_rows(args.jsonl)
    if not rows:
        print("解析出 0 条记录")
        return 2

    n = len(rows)
    logtxt = ""
    if args.log and os.path.isfile(args.log):
        logtxt = open(args.log, encoding="utf-8", errors="replace").read()

    # ---- 阶段：耗时 + 覆盖题数 ----
    stage_t = defaultdict(list)
    for r in rows:
        for k, v in ((r.get("diag") or {}).get("stage_timers") or {}).items():
            try:
                stage_t[k].append(float(v))
            except (TypeError, ValueError):
                pass

    # ---- 机制：触发次数分布 ----
    mech_c = defaultdict(list)
    for r in rows:
        d = r.get("diag") or {}
        for f, *_ in MECHS:
            v = d.get(f)
            if v is None:
                mech_c[f].append(None)
            elif isinstance(v, (list, dict)):
                mech_c[f].append(len(v))
            else:
                try:
                    mech_c[f].append(float(v))
                except (TypeError, ValueError):
                    mech_c[f].append(None)

    out = []
    W = out.append

    W("# 环节审计：原定目的 → 完成判据 → 实测结论")
    W("")
    W("数据源：%s（%d 题）" % (os.path.basename(args.jsonl), n))
    if logtxt:
        W("日志：%s" % os.path.basename(args.log))
    W("")
    W("结论三态：**完成** = 应跑且跑出东西；**设计如此** = 开关默认关或前置条件")
    W("结构性不满足；**未完成** = 应跑却无产出（需查因）。")
    W("")

    # ---------------- 阶段表 ----------------
    W("## 一、%d 个阶段（判据基于**产出物**，非耗时；含已删阶段的历史行）"
      % len(STAGES_stage_audit))
    W("")
    W("| 阶段 | 原定目的 | 完成判据 | 实测 | 结论 |")
    W("|---|---|---|---|---|")
    stats = Counter()
    for key, cn, purpose, crit, expect in STAGES_stage_audit:
        ts = stage_t.get(key, [])
        meas, verdict = judge_stage(key, rows, ts, logtxt)
        if expect == "off" and verdict == "未完成":
            verdict = "设计如此"
        stats[verdict] += 1
        W("| %s（%s） | %s | %s | %s | **%s** |" % (
            key, cn, purpose, crit, meas, verdict))
    W("")
    W("统计：%s" % "，".join("%s %d 项" % (k, v) for k, v in stats.items()))
    W("")

    # ---------------- 机制表 ----------------
    W("## 二、%d 个机制字段" % len(MECHS))
    W("")
    W("| 机制 | 原定目的 | 实测 | 结论 |")
    W("|---|---|---|---|")
    stats2 = Counter()
    for f, cn, purpose, expect in MECHS:
        vals = [v for v in mech_c.get(f, []) if v is not None]
        if not vals:
            meas = "该轮无此字段"
            verdict = "无数据"
        else:
            zero = sum(1 for v in vals if not v)
            mx = max(vals)
            meas = "触发 %d/%d 题（中位 %.4g，max %.4g）" % (
                len(vals) - zero, len(vals), st.median(vals), mx)
            if zero == len(vals):
                verdict = "设计如此" if expect == "off" else (
                    "未完成" if expect == "run" else "未触发（条件未满足）")
            else:
                verdict = "完成"
        stats2[verdict] += 1
        W("| %s | %s | %s | **%s** |" % (f, purpose, meas, verdict))
    W("")
    W("统计：%s" % "，".join("%s %d 项" % (k, v) for k, v in stats2.items()))
    W("")

    # ---------------- 日志侧补充 ----------------
    if logtxt:
        W("## 三、日志侧补充判据")
        W("")
        W("| 判据 | 计数 | 说明 |")
        W("|---|---|---|")
        for pat, label, note in (
            (r"mcp 返回成功：ok=True", "Lean 编译成功", "应 > 0；恒 0 = 环境或代码质量全线失败"),
            (r"mcp 返回成功：ok=False", "Lean 编译失败", "与上一行对照看成功率"),
            (r"走 mcp 后端", "走 MCP 后端", "MCP 通道被实际使用"),
            (r"回落 bridge", "回落 bridge", "应尽量为 0"),
            (r"axiom 检查通过", "axiom 检查通过", "修 example 盲区后此数应上升"),
            (r"axiom 检查发现 sorryAx", "★ 检出 sorryAx", "该数 > 0 说明防线真的在拦"),
            (r"axiom 检查改用 example 改写副本", "★ example 改写副本", "新修复的触发证据"),
            (r"multi_attempt 找到可用策略", "multi_attempt 找到策略", "MCP 增强 B4"),
            (r"hover 查证", "hover 查证", "MCP 增强 B6"),
        ):
            W("| %s | %d | %s |" % (label, len(re.findall(pat, logtxt)), note))
        W("")

    text = "\n".join(out).replace("|", "\u2223") if False else "\n".join(out)
    # 只把**正文里**的竖线替换掉会破坏表格，故此处保留表格竖线；
    # 交给 md→docx 生成器时再按需处理。
    print(text)
    if args.out:
        # 写 md：表格竖线保留（生成器已能处理表格），正文另行约定
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
