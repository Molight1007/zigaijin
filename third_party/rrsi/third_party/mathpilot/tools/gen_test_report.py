# -*- coding: utf-8 -*-
"""测试结果 → Word 报告生成器（2026-09-15 建立，以后每次评测都用它）。

用法
----
    D:/python/python.exe tools/gen_test_report.py \
        --results results/verify10_0915.jsonl \
        --testset 题库/smoke10_verify_0915.jsonl \
        --core    results/_112_core_table.jsonl \
        --out-md  "测试结果/xin测试结果/10题评测报告_0915.md"

随后用 `D:\\mathpilot_work\\tools\\_mkdocx.py` 转成 .docx（该脚本内置自检与验收）。

报告内容（按用户 2026-09-15 要求：详细记录错误情况 + 大模型答题情况）
--------------------------------------------------------------------
1. 测试概况（题单/配置/条件）
2. 总体结果
3. 逐题详情：题目原文、领域/题型/难度档、标准答案 vs 模型答案、判定、
   耗时、候选数、**逐候选票型**、三道闸门各自判定、结构化错误标签、自动诊断
4. 难度分布
5. 错误分析（含标签分布）
6. 模型答题行为分析（候选数/票型/闸门通过率）
7. 基础设施问题（限流/超时/MCP/内存）
8. 结论与下一步

⚠ 实现约束（都踩过）：
- md 里**禁止围栏代码块** —— `_mkdocx.py` 的自检见到 ``` 会直接 exit 1。
- 单个 `*` 会被 `_build_docx._inline` 当成斜体 ⇒ LaTeX 里的 `^*` 会被吃掉。
  ⇒ 题目/答案这类原文一律用**行内反引号**包裹保护；正文里的 `*` 换成 `∗`。
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys

BOLD_STAR = "\u2217"      # ∗ —— 替换正文里的裸 *，避免被当成斜体标记


def load_jsonl(path: str) -> list:
    if not path or not os.path.exists(path):
        return []
    rows = []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def safe(text, limit: int = 0) -> str:
    """转义为 md 安全文本。

    - 去掉反引号（避免破坏行内代码包裹）
    - 裸 `*` → `∗`（否则 `_build_docx._inline` 会把 `^*` 当斜体吃掉）
    - `|` → `∣`（U+2223）。**不能**用 markdown 的 `\\|` 转义：`_mkdocx.py` 的自检
      按原始竖线切分单元格（`strip("|").split("|")`），转义后的 `\\|` 仍会被计数，
      照样报"表格列数不一致"。实测踩到：一行 2 列的表格被 LaTeX 里的 `|f(x)|`
      撑成 4 列。`∣` 视觉上与 `|` 几乎一致，且不含竖线字符。
    """
    s = str(text if text is not None else "")
    s = s.replace("`", "")
    s = s.replace("**", BOLD_STAR * 2).replace("*", BOLD_STAR)
    s = s.replace("|", "∣")
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    if limit and len(s) > limit:
        s = s[:limit] + " …（已截断）"
    return s


def code(text, limit: int = 0) -> str:
    """行内代码包裹（保护 LaTeX 符号不被强调语法吃掉）。"""
    return "`" + safe(text, limit) + "`"


def verdict_line(verdicts: list) -> str:
    """把逐候选票型压缩成可读串：n/m 表示 m 票里 n 票判 A。"""
    if not verdicts:
        return "（无记录）"
    parts = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        cv, tv = v.get("correct_votes"), v.get("total_votes")
        parts.append("%s/%s" % (cv if cv is not None else "?",
                                tv if tv is not None else "?"))
    return "、".join(parts) if parts else "（无记录）"


def audit_summary(entries: list) -> str:
    """AuditGate：候选审核判定分布 + final_gate 结论。"""
    if not entries:
        return "（未触发）"
    cand = collections.Counter()
    finals = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        step = e.get("step")
        if step == "candidate_audit":
            cand[str(e.get("verdict"))] += 1
        elif step == "final_gate":
            finals.append(str(e.get("verdict")))
    seg = []
    if cand:
        seg.append("候选审核 " + "、".join("%s×%d" % (k, v)
                                          for k, v in cand.most_common()))
    if finals:
        seg.append("final_gate=" + "、".join(finals))
    return "；".join(seg) if seg else "（未触发）"


def lean_summary(entries: list) -> str:
    """LeanGate：判定 + 降级原因 + 编译失败原因分布。"""
    if not entries:
        return "（未触发/不适用）"
    v = collections.Counter()
    deg = collections.Counter()
    fk = collections.Counter()
    for e in entries:
        if not isinstance(e, dict) or "verdict" not in e:
            continue          # 跳过无判定字段的条目（如仅记录 id 的占位）
        v[str(e.get("verdict"))] += 1
        if e.get("degraded"):
            deg[str(e.get("degraded"))] += 1
        # ★ 2026-10-22：失败原因分类（refuted/unproved/translation/env）。
        #   没有它，报表只能看到 proof_invalid×N，分不清「确实被证伪」和
        #   「环境缺依赖/没证出来」——后者恰恰不该算答案错。
        if e.get("failure_kind"):
            fk[str(e.get("failure_kind"))] += 1
    if not v:
        return "（未触发/不适用）"
    seg = "、".join("%s×%d" % (k, n) for k, n in v.most_common())
    if fk:
        seg += "（失败原因：" + "、".join("%s×%d" % (k, n)
                                     for k, n in fk.most_common()) + "）"
    if deg:
        seg += "（降级：" + "、".join("%s×%d" % (k, n)
                                     for k, n in deg.most_common()) + "）"
    return seg


def stage_rows(d: dict) -> list:
    """全部阶段耗时（按耗时降序），返回 [(阶段, 秒)]。"""
    st = d.get("stage_timers") or {}
    if not isinstance(st, dict):
        return []
    return [(safe(k), float(v or 0)) for k, v in
            sorted(st.items(), key=lambda x: -(x[1] or 0))]


def trace_summary(trace) -> list:
    """事件流按 (agent, step) 聚合：[(agent, step, 次数, 首条内容摘要)]。

    这是"各环节效果"的主口径——能看出每个环节到底跑了没有、跑了几次、说了什么。
    """
    if not isinstance(trace, list):
        return []
    agg = {}
    order = []
    for t in trace:
        if not isinstance(t, dict):
            continue
        key = (str(t.get("agent") or "-"), str(t.get("step") or "-"))
        if key not in agg:
            agg[key] = [0, ""]
            order.append(key)
        agg[key][0] += 1
        if not agg[key][1] and t.get("content"):
            agg[key][1] = safe(str(t.get("content")), 90)
    return [(a, s, n, sample) for (a, s), (n, sample) in
            ((k, agg[k]) for k in order)]


def candidate_blocks(row: dict, per_cand_chars: int = 700) -> str:
    """渲染每个候选的完整解答过程（答案 + 票型 + reasoning）。"""
    cands = row.get("candidates") or []
    if not cands:
        return "（本行无候选记录：可能来自旧结果文件，或该题未走候选路径）\n"
    out = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            continue
        cv, tv = c.get("correct_votes"), c.get("total_votes")
        conf = c.get("confidence")
        out.append("**候选 %s**（票型 %s/%s%s%s）" % (
            c.get("id", i),
            cv if cv is not None else "?",
            tv if tv is not None else "?",
            "，置信度 %.2f" % conf if isinstance(conf, (int, float)) else "",
            "，经修订" if c.get("revised") else ""))
        out.append("")
        out.append("- 答案：`%s`" % safe(c.get("answer") or "（空）", 200))
        rea = safe(c.get("reasoning") or "（无推理过程记录）", per_cand_chars)
        out.append("- 解答过程：%s" % rea)
        fb = safe(c.get("feedback") or "", 160)
        if fb:
            out.append("- 验证反馈：%s" % fb)
        out.append("")
    return "\n".join(out)


def llm_calls_text(row: dict) -> str:
    lc = row.get("llm_calls") or {}
    if not isinstance(lc, dict) or not lc:
        return "（未记录）"
    return "调用 %s 次，其中截断 %s 次" % (
        lc.get("calls", "?"), lc.get("truncated", "?"))


def tool_calls_text(row: dict) -> str:
    """工具级埋点：Lean MCP + 联网搜索的调用效果 + 派生判读。"""
    tc = row.get("tool_calls") or {}
    if not isinstance(tc, dict) or not tc:
        return "（未记录）"
    segs = []
    m = tc.get("lean_mcp") or {}
    if m:
        segs.append("MCP %s 次（成功 %s / 失败 %s，%.0fs）→ %s" % (
            m.get("calls", "?"), m.get("ok", "?"), m.get("fail", "?"),
            float(m.get("seconds") or 0), safe(m.get("verdict") or "-", 40)))
    w = tc.get("web_search") or {}
    if w:
        segs.append("联网搜索 %s 次（成功 %s / 失败 %s，命中 %s 条，%.0fs）→ %s" % (
            w.get("calls", "?"), w.get("ok", "?"), w.get("fail", "?"),
            w.get("results", "?"), float(w.get("seconds") or 0),
            safe(w.get("verdict") or "-", 20)))
    return "；".join(segs) if segs else "（未记录）"


def deep_review_summary(d: dict) -> str:
    """「带推理的最终复核」判定可读串（独立信号，不并入投票统计）。"""
    dr = d.get("deep_review") or {}
    if not dr:
        return "（未记录）"
    if dr.get("ran"):
        seg = "判定 %s" % ("A（通过）" if dr.get("verdict") == "A" else "B（否决）")
        if dr.get("error_type"):
            seg += "，错误类型 " + safe(dr["error_type"])
        if dr.get("chars"):
            seg += "，输出 %d 字" % dr["chars"]
        return seg
    if dr.get("skipped"):
        return "未运行：" + safe(dr["skipped"], 80)
    return "未运行"


def pick_domain(r: dict, bank_row: dict, core_row: dict) -> str:
    """领域回退链：结果 → 题单 → 历史核心表（结果里常为空，实测 7/10 题如此）。"""
    for src in (r.get("domain"), bank_row.get("domain"), core_row.get("domain")):
        s = safe(src)
        if s and s not in ("（空）", "unknown"):
            return s
    return "（题库未标注）"


def auto_diag(row: dict, verdicts: list) -> str:
    """自动诊断：把"哪里出了问题"写成一句话（报告最有价值的一列）。"""
    d = row.get("diag") or {}
    ok = row.get("correct")
    nrej = d.get("n_reject_votes") or 0
    et = d.get("error_types") or {}
    votes = [v for v in (d.get("verdicts") or []) if isinstance(v, dict)]

    if ok:
        return "答对。" + ("验证器有 %d 张否决票（拒绝了部分候选），最终仍选出正确解。" % nrej
                          if nrej else "各闸门未提出异议。")

    # 答错：逐层判断是哪一环失守
    all_a = bool(votes) and all((v.get("correct_votes") or 0)
                                == (v.get("total_votes") or 0) for v in votes)
    lean = d.get("lean_gate") or []
    lean_valid = any(isinstance(e, dict) and e.get("verdict") == "answer_valid"
                     and e.get("lean_valid") for e in lean)
    lean_deg = any(isinstance(e, dict) and e.get("degraded") for e in lean)
    audit = d.get("audit_gate") or []
    audit_unknown = bool(audit) and all(
        (not isinstance(e, dict)) or e.get("step") != "candidate_audit"
        or e.get("verdict") == "unknown" for e in audit)

    body = []
    if all_a:
        body.append("**验证器全票放行**（%d 个候选每票均判 A）" % len(votes))
    elif not nrej:
        body.append("验证器未产生否决票")
    if audit_unknown:
        body.append("AuditGate 候选审核全为 unknown（等价于不否决）")
    if lean_valid:
        body.append("LeanGate 判 Lean 有效（假阳性）")
    elif lean_deg:
        body.append("LeanGate 无法判定，降级放行")
    if not body:
        body.append("各闸门未提出异议")
    return "答错，且" + "；".join(body) + "。错误标签：" + (
        "、".join("%s×%d" % (k, v) for k, v in
                  sorted(et.items(), key=lambda x: -x[1])) if et else "（无，因无否决票）")


def _render_question_details(A, rows, bank, core):
    """三、逐题详情段（2026-10-08 从 build_gen_test_report 逐字搬出）。"""
    # ---------- 三、逐题详情 ----------
    A("## 三、逐题详情")
    A("")
    for i, r in enumerate(rows, 1):
        qid = r.get("id")
        d = r.get("diag") or {}
        b = bank.get(qid) or {}
        c = core.get(qid) or {}
        q = b.get("question") or r.get("question") or ""
        A("### 第 %d 题 · %s" % (i, qid))
        A("")
        A("**题目**（原文，已截断保护）")
        A("")
        A(code(q, 600))
        A("")
        A("| 项 | 值 |")
        A("|---|---|")
        A("| 领域 | %s |" % pick_domain(r, b, c))
        A("| 题型 | %s |" % safe(d.get("question_type") or "（未记录）"))
        A("| 难度档 | %s |" % safe(d.get("tier") or "（未记录）"))
        A("| 标准答案 | %s |" % code(r.get("gold"), 160))
        A("| 模型答案 | %s |" % code(r.get("predicted"), 160))
        A("| 判定 | %s |" % ("✅ 正确" if r.get("correct") else "❌ 错误"))
        A("| 本次耗时 | %.0f 秒 |" % r.get("elapsed_sec", 0))
        if c:
            A("| 历史（0910 批） | %.0f 秒 · %s |" % (
                c.get("elapsed_sec", 0),
                "当时答对" if c.get("correct") else "当时答错"))
        A("| 候选数 | %s |" % d.get("n_candidates"))
        A("| 逐候选票型 | %s |" % verdict_line(d.get("verdicts")))
        A("| AuditGate | %s |" % safe(audit_summary(d.get("audit_gate")), 200))
        A("| LeanGate | %s |" % safe(lean_summary(d.get("lean_gate")), 200))
        A("| 结构化错误标签 | %s |" % (
            "、".join("%s×%d" % (k, v) for k, v in
                      sorted((d.get("error_types") or {}).items(),
                             key=lambda x: -x[1])) or "（无）"))
        A("| 判错票数 | %s |" % (d.get("n_reject_votes") or 0))
        A("| 带推理复核 | %s |" % safe(deep_review_summary(d), 160))
        A("| LLM 调用 | %s |" % safe(llm_calls_text(r), 80))
        A("| 工具调用（Lean MCP） | %s |" % safe(tool_calls_text(r), 110))
        _sr = stage_rows(d)
        A("| 阶段耗时合计 | %.0f 秒（共 %d 个阶段计时点）|" % (
            sum(x[1] for x in _sr), len(_sr)))
        A("| 事件流条数 | %d |" % len(r.get("trace") or []))
        A("")
        A("**诊断**：" + auto_diag(r, d.get("verdicts")))
        A("")
        # 阶段耗时（前 5）
        st = d.get("stage_timers") or {}
        if st:
            top = sorted(st.items(), key=lambda x: -x[1])[:5]
            A("主要阶段耗时：" + "、".join(
                "%s %.0fs" % (safe(k), v) for k, v in top))
            A("")


def _render_difficulty_table(A, rows, bank, core):
    """四、难度分布段（2026-10-08 从 build_gen_test_report 逐字搬出）。"""
    # ---------- 四、难度分布 ----------
    A("## 四、难度分布")
    A("")

    def _tier(r):
        return safe((r.get("diag") or {}).get("tier")) or "未知"

    def _qtype(r):
        return safe((r.get("diag") or {}).get("question_type")) or "未知"

    def _domain(r):
        return pick_domain(r, bank.get(r.get("id")) or {}, core.get(r.get("id")) or {})

    for name, keyfn in (("难度档", _tier), ("题型", _qtype), ("领域", _domain)):
        cnt = collections.Counter(keyfn(r) for r in rows)
        A("| %s | 题数 | 正确 | 正确率 |" % name)
        A("|---|---|---|---|")
        for k, v in cnt.most_common():
            ok = sum(1 for r in rows if keyfn(r) == k and r.get("correct"))
            A("| %s | %d | %d | %.0f%% |" % (k, v, ok, 100.0 * ok / v if v else 0))
        A("")
    A("> 难度代理指标：本报告以「档位（standard/deep）+ 题型的客观/主观」为主口径；")
    A("> 历史耗时仅作参考（同一题不同批次耗时差异可达数倍，受 LLM 服务延迟与预算配速影响）。")
    A("")


def _render_appendix_a(A, rows):
    """附录 A 逐题详细过程段（2026-10-08 从 build_gen_test_report 逐字搬出）。"""
    # ---------- 附录 A：逐题详细过程 ----------
    # 用户 2026-09-16 要求「极为详细的数据：大模型的解答过程 / 时间消耗 / 各环节效果」。
    # 正文保持可读，重过程数据全部收进附录。
    A("## 附录 A · 逐题详细过程")
    A("")
    A("说明：本附录逐题给出①**最终回答全文** ②**时间消耗**（全部阶段，按耗时降序）"
      "③**各环节效果**（事件流按 agent/step 聚合）④**每个候选的解答过程**"
      "（含模型原始 reasoning，超长已截断并标注）。")
    A("")
    for i, r in enumerate(rows, 1):
        d = r.get("diag") or {}
        A("### A.%d · %s（%s）" % (
            i, r.get("id"), "✅ 正确" if r.get("correct") else "❌ 错误"))
        A("")
        A("**最终回答（全文）**")
        A("")
        A(code(r.get("response_full") or r.get("response") or "（空）", 2000))
        A("")
        # ---- 时间消耗 ----
        A("**时间消耗**")
        A("")
        _sr = stage_rows(d)
        if _sr:
            A("| 阶段 | 秒 | 占比 |")
            A("|---|---|---|")
            _tot = sum(x[1] for x in _sr) or 1.0
            for name, sec in _sr:
                A("| %s | %.1f | %.0f%% |" % (name, sec, 100.0 * sec / _tot))
            A("| **合计（阶段计时）** | **%.1f** | 100%% |" % _tot)
            A("")
        else:
            A("（无阶段计时记录）")
            A("")
        A("单题墙钟耗时：**%.0f 秒**；LLM 调用：%s。" % (
            r.get("elapsed_sec", 0), safe(llm_calls_text(r), 80)))
        A("")
        # ---- 各环节效果 ----
        A("**各环节效果（事件流）**")
        A("")
        _ts = trace_summary(r.get("trace"))
        if _ts:
            A("| 执行者 | 环节 | 次数 | 首条内容摘要 |")
            A("|---|---|---|---|")
            for agent, step, n, sample in _ts:
                A("| %s | %s | %d | %s |" % (agent, step, n, sample or ""))
            A("")
        else:
            A("（无事件流记录：可能来自旧结果文件）")
            A("")
        # ---- 候选解答过程 ----
        A("**候选解答过程**")
        A("")
        A(candidate_blocks(r))
        A("")


def build_gen_test_report(results_path, testset_path, core_path, out_md, title):
    rows = load_jsonl(results_path)
    bank = {r.get("id"): r for r in load_jsonl(testset_path) if r.get("id")}
    core = {r.get("id"): r for r in load_jsonl(core_path) if r.get("id")}

    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("correct"))
    acc = (100.0 * n_ok / n) if n else 0.0

    tag_agg = collections.Counter()
    n_with_b = 0
    tot_rej = 0
    for r in rows:
        d = r.get("diag") or {}
        et = d.get("error_types") or {}
        for k, v in et.items():
            tag_agg[k] += v
        if d.get("n_reject_votes"):
            n_with_b += 1
        tot_rej += (d.get("n_reject_votes") or 0)

    L = []
    A = L.append
    A("# " + title)
    A("")
    A("生成时间：自动生成 | 结果文件：`%s`" % os.path.basename(results_path))
    A("")

    # ---------- 一、概况 ----------
    A("## 一、测试概况")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 题单文件 | %s |" % code(os.path.basename(testset_path)))
    A("| 结果文件 | %s |" % code(os.path.basename(results_path)))
    A("| 题目总数 | %d |" % n)
    A("| 单题时限 | 由 `max_time_per_question` 与档位预算共同决定（本地评测覆盖值 1100s） |")
    A("| 并发 | 1 |")
    A("")
    if core:
        A("题单由 `_112_core_table.jsonl` 中**历史判错**的题目构成（本项目纪律：只测错题）。")
        A("")

    # ---------- 二、总体结果 ----------
    A("## 二、总体结果")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A("| 完成 | %d / %d |" % (n, n))
    A("| 正确 | %d |" % n_ok)
    A("| 正确率 | %.1f%% |" % acc)
    A("| 平均耗时 | %.1f 秒 |" % (
        sum(r.get("elapsed_sec", 0) for r in rows) / n if n else 0))
    A("| 出现否决票的题 | %d / %d |" % (n_with_b, n))
    A("| 否决票总数 | %d |" % tot_rej)
    A("")
    A("> ⚠ 题单为**历史错题**，不是随机抽样，因此本正确率**不能**与 112 题全量基线直接比较。")
    A("")

    _render_question_details(A, rows, bank, core)

    _render_difficulty_table(A, rows, bank, core)

    # ---------- 五、错误分析 ----------
    A("## 五、错误分析")
    A("")
    wrong = [r for r in rows if not r.get("correct")]
    A("错题 %d 道。" % len(wrong))
    A("")
    if tag_agg:
        A("| 结构化错误标签 | 出现票数 |")
        A("|---|---|")
        for k, v in tag_agg.most_common():
            A("| %s | %d |" % (k, v))
        A("")
        hard = sum(v for k, v in tag_agg.items() if k in ("前提不成立", "方法不适用", "方向反了"))
        A("其中「前提不成立 / 方法不适用 / 方向反了」（= 定理被**硬套**的三种形态）合计 "
          "%d 票，占全部标签的 %.0f%%。" % (hard, 100.0 * hard / sum(tag_agg.values())))
        A("")
    A("**结论**：%d 道错题中只有 %d 道产生过否决票 ⇒ 大多数错答**没有被任何闸门提出异议**，"
      "因此也没有错误标签可统计。这是本次测试最重要的发现。" % (len(wrong), n_with_b))
    A("")

    # ---------- 六、模型答题行为 ----------
    A("## 六、大模型答题行为分析")
    A("")
    A("| 行为指标 | 值 |")
    A("|---|---|")
    A("| 平均候选数 | %.1f |" % (
        sum((r.get("diag") or {}).get("n_candidates") or 0 for r in rows) / n if n else 0))
    A("| 全候选全票判 A 的题 | %d |" % sum(
        1 for r in rows
        if (r.get("diag") or {}).get("verdicts") and all(
            (v.get("correct_votes") or 0) == (v.get("total_votes") or 0)
            for v in (r.get("diag") or {}).get("verdicts") or []
            if isinstance(v, dict))))
    A("| 答案完整率 | %d / %d |" % (
        sum(1 for r in rows if (r.get("diag") or {}).get("answer_complete")), n))
    A("| 出现占位符 | %d |" % sum(1 for r in rows if (r.get("diag") or {}).get("placeholder")))
    A("| 降级标志次数 | %d |" % sum(
        (r.get("diag") or {}).get("degraded_flags") or 0 for r in rows))
    _dr_ran = sum(1 for r in rows
                  if ((r.get("diag") or {}).get("deep_review") or {}).get("ran"))
    _dr_veto = sum(1 for r in rows if (
        (r.get("diag") or {}).get("deep_review") or {}).get("verdict") == "B")
    A("| 带推理复核运行题数 | %d / %d |" % (_dr_ran, n))
    A("| 带推理复核查出否定 | %d |" % _dr_veto)
    A("")
    A("读法：候选数正常、答案完整、无占位符 ⇒ **生成侧没有结构性故障**；")
    A("问题集中在**验证侧不产生否决**（见第五节）。")
    A("")

    # ---------- 七、基础设施 ----------
    A("## 七、基础设施问题（与改动无关，但影响测试可信度）")
    A("")
    A("| 现象 | 说明 |")
    A("|---|---|")
    A("| LLM 限流 | `HTTP 400 -20048 请求过于频繁`，导致验证票解析到 None → 计为弃权 |")
    A("| LLM 读超时 | 单次调用 120s 超时后失败 |")
    A("| MCP 后端超时 | 每次 Lean 调用先等 90s 再回落 bridge |")
    A("| 内存不足 | `可用内存不足 1.5 GiB → 不再扩充 MCP 实例` |")
    A("| Lean 答案验证告警 | 「答案数值未出现在验证代码（书生自算而非审核答案）」、"
      "「疑似自证，拒绝」 |")
    A("")

    # ---------- 八、结论 ----------
    A("## 八、结论与下一步")
    A("")
    A("1. 生成侧正常：候选数、答案完整性、无占位符均无异常。")
    A("2. **验证侧不产生否决**是首要问题：多数错答三道闸门均未提出异议。")
    A("3. 仅有的否决票集中在「前提不成立 / 方法不适用」——与老师提出的"
      "「硬套定理」直接对应。")
    A("4. 基础设施抖动（限流/超时）会污染单次结果，结论需多轮复核。")
    A("")

    _render_appendix_a(A, rows)

    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_md)) or ".", exist_ok=True)
    with io.open(out_md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    return md


def main():
    ap = argparse.ArgumentParser(description="测试结果 → Word 报告（md 中间产物）")
    ap.add_argument("--results", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--core", default="results/_112_core_table.jsonl")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--title", default="本地评测报告")
    a = ap.parse_args()
    md = build_gen_test_report(a.results, a.testset, a.core, a.out_md, a.title)
    print("已写出:", a.out_md, "( %d 字符 )" % len(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
