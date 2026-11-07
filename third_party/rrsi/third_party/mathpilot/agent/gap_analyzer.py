# -*- coding: utf-8 -*-
"""逻辑缺口识别（2026-09-29 新增，截图 #6）。

用户原话
--------
> 子目标的设立有没有帮助大模型简化题目，将一个难题简化为许多简单题。
> 蓝图的设计有没有问题，**可不可以用 lean 来检测有没有 sorry 的地方，
> 像这种缺少的逻辑点是不是就是大模型需要推理出来的点呢**？这个要思考一下。

思路拆解
--------
「用 Lean 检测 sorry」这件事项目里**已经有底座**（`lean_refiner`）：
  · `extract_sorry_blocks()` —— 找出每个含 sorry 的定理
  · `refine_one()`         —— 让 LLM 尝试补全该 sorry，编译通过才算补上
  · `refine_tree()`        —— 整树补全并产出 `per_node` 状态

★ 但**有底座 ≠ 回答了用户的问题**。用户要的不是「有几个 sorry」，
  而是「**哪些 sorry 是大模型的推理瓶颈**」—— 这是一个**归因**问题，
  而 `refine_result` 只给 `done/failed` 计数，没有任何可用于归因的结构化信息。

本模块补的就是这一层：把 `refine_result` 的失败节点**提炼成有意义的缺口清单**，
回答三个可判定的子问题：

  ① 缺口在**哪**：失败节点的命题陈述（statement）是什么？
  ② 缺口**是什么性质**：定义缺失 / 引理缺失 / 类型不匹配 / 纯逻辑跳跃 /
     算不动 —— 由 Lean 报错文本分类（`classify_gap`）。
  ③ 缺口**是不是"必须由大模型推理出来"的点**：
     - 若缺口是「引理缺失但 Mathlib 里有」→ 是**检索问题**，不是推理瓶颈，
       被 leansearch / 定理注入解决；
     - 若缺口是「纯逻辑跳跃」（有全部前提却推不出）→ **才是真正的推理瓶颈**，
       是子目标分解该起作用的地方；
     - 若缺口是「定义/类型缺失」→ 是**形式化问题**（译错题），
       不该算进推理能力评估。

★ 这个三分是**本模块的核心价值**：不区分的话，「60% 的 sorry 没补上」会被
  笼统读成「大模型推理能力差」，而实际上可能大半是形式化译题错误 —— 那会
  把优化方向引到完全错误的地方（去调推理提示词，而该修的是翻译器）。

设计约束（与项目既有纪律一致）
------------------------------
  · **纯函数、零 LLM、零 Lean 调用** —— 只消费已有的 `refine_result` / 报错文本；
  · 永不抛异常，拿不到数据就返回空结论（`verdict="unknown"`）；
  · 不改任何既有分支，只新增只读分析。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 缺口分类规则
# ---------------------------------------------------------------------------
# ⚠ 顺序敏感：先匹配到的先归类。刻意把「形式化问题」放最前 ——
#   这类缺口**不该算作推理瓶颈**，若被误归为"逻辑跳跃"，会误导优化方向。
_GAP_RULES = [
    # ---- ① 形式化 / 翻译问题：题被译错了，不是模型推理不出来 ----
    ("formalization", [
        r"unknown identifier",
        r"unknown constant",
        r"unknown namespace",
        r"invalid field notation",
        r"unexpected token",
        r"expected token",
        r"function expected at",
        r"type mismatch",              # 类型不匹配，通常是译错
        r"application type mismatch",
        r"failed to synthesize",
        r"declaration uses 'sorry'",   # 只有这一条也要看上下文，见下
    ]),
    # ---- ② 引理缺失：Mathlib 里其实有，属**检索问题** ----
    ("missing_lemma", [
        r"unknown theorem",
        r"try this",                   # Lean 的 `Try this:` 建议，通常是给出 lemma 名
        r"did you mean",
        r"no goals to be solved",      # 常出现在套用了不存在的引理之后
    ]),
    # ---- ③ 算不动：简化/规范化失败，属**计算工具**范畴 ----
    ("computation", [
        r"norm_num",
        r"failed to prove",
        r"linarith failed",
        r"ring_nf",
        r"positivity",
    ]),
    # ---- ④ 纯逻辑跳跃：前提齐全但推不出 ⇒ **真正的推理瓶颈** ----
    ("logical_jump", [
        r"unsolved goals",
        r"tactic .* failed",
        r"no progress",
        r"goals remaining",
    ]),
]

_COMPILED = [(kind, [re.compile(p, re.I) for p in pats])
             for kind, pats in _GAP_RULES]

# 缺口性质 → 中文释义 + 「该往哪个方向优化」的建议
_GAP_MEANING = {
    "logical_jump": (
        "纯逻辑跳跃（前提齐全却推不出）",
        "**这是真正的推理瓶颈**：子目标分解 / 定理注入应在这一步起作用。"
        "统计它的占比，才能判断「子目标有没有帮助大模型简化题目」。"),
    "missing_lemma": (
        "缺引理（Mathlib 里可能有）",
        "属**检索问题**而非推理问题：应由 leansearch / 定理注入解决。"
        "这类缺口多 ⇒ 优化方向是检索，不是推理提示词。"),
    "formalization": (
        "形式化/译题问题（定义、类型、语法）",
        "**不该算作推理能力不足**：是题目→Lean 的翻译有误。"
        "这类缺口多 ⇒ 优化方向是 lean_translator，不是解题链。"),
    "computation": (
        "计算/化简未完成",
        "属**计算工具**范畴：应由数值代入 / 符号求解解决，"
        "不算推理瓶颈。"),
    "unknown": (
        "未能归类",
        "报错文本没有可识别特征 —— 需人工看原始错误。"),
}

# 缺口性质 → 「是否算作大模型必须推理出来的点」
_COUNTS_AS_REASONING = {
    "logical_jump": True,
    "missing_lemma": False,
    "formalization": False,
    "computation": False,
    "unknown": None,   # 未知
}


def classify_gap(error_text: str) -> str:
    """把 Lean 报错文本归类成缺口性质（永不抛异常，未命中返回 'unknown'）。"""
    txt = str(error_text or "")
    if not txt.strip():
        return "unknown"
    for kind, pats in _COMPILED:
        for p in pats:
            if p.search(txt):
                return kind
    return "unknown"


def gap_meaning(kind: str) -> tuple:
    """缺口性质 → (中文释义, 优化方向建议)。"""
    return _GAP_MEANING.get(kind, _GAP_MEANING["unknown"])


def counts_as_reasoning(kind: str) -> bool | None:
    """该缺口是否算「大模型必须推理出来的点」。None = 无法判定。"""
    return _COUNTS_AS_REASONING.get(kind)


# ---------------------------------------------------------------------------
# 从 refine_result 提取缺口清单
# ---------------------------------------------------------------------------
def extract_gaps(refine_result: dict,
                 blueprint: dict | None = None) -> dict:
    """把 `ctx.refine_result` 提炼成**结构化逻辑缺口清单**。

    参数
    ----
    refine_result : `ctx.refine_result`（lean_refiner 的产物）
    blueprint     : `ctx.blueprint`（可选，用于把节点 id 还原成命题陈述）

    返回
    ----
    {
      "ok": bool,                  # 是否拿到可分析的数据
      "verdict": str,              # 透传 refine_result 的 verdict
      "n_nodes": int,              # 参与精炼的节点总数
      "n_done": int, "n_failed": int,
      "gaps": [                    # 每个失败节点一条
         {"node_id", "theorem", "statement", "kind", "meaning",
          "is_reasoning", "advice", "error_excerpt"}
      ],
      "by_kind": {kind: count},    # 缺口性质分布
      "reasoning_gaps": int,       # 其中"真·推理瓶颈"的条数
      "non_reasoning_gaps": int,   # 形式化/检索/计算问题（不算推理）
      "unclassified": int,         # 未能归类
      "reasoning_share": float,    # reasoning_gaps / n_failed（0 表示无失败）
      "diagnosis": str,            # 一句话结论（可直接进报告）
    }
    """
    out = {
        "ok": False, "verdict": "unknown",
        "n_nodes": 0, "n_done": 0, "n_failed": 0,
        "gaps": [], "by_kind": {},
        "reasoning_gaps": 0, "non_reasoning_gaps": 0, "unclassified": 0,
        "reasoning_share": 0.0, "diagnosis": "",
    }
    if not isinstance(refine_result, dict) or not refine_result:
        out["diagnosis"] = "无 refine_result（未执行 Stage3 精炼）"
        return out

    per_node = refine_result.get("per_node")
    if not isinstance(per_node, dict):
        out["verdict"] = str(refine_result.get("verdict") or "unknown")
        out["diagnosis"] = "refine_result 无 per_node 明细，无法定位缺口"
        return out

    # 节点 id → 命题陈述（从 blueprint 的 nodes 还原）
    stmt_map = _stmt_map_from_blueprint(blueprint)

    out["ok"] = True
    out["verdict"] = str(refine_result.get("verdict") or "unknown")
    out["n_nodes"] = len(per_node)
    out["n_done"] = int(refine_result.get("done") or 0)
    out["n_failed"] = int(refine_result.get("failed") or 0)

    by_kind: dict = {}
    for node_id, status in per_node.items():
        if not isinstance(status, dict):
            continue
        if status.get("ok"):
            continue                      # 补上了 ⇒ 不是缺口
        err = str(status.get("error") or "")
        kind = classify_gap(err)
        meaning, advice = gap_meaning(kind)
        is_reasoning = counts_as_reasoning(kind)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        out["gaps"].append({
            "node_id": str(node_id),
            "theorem": str(status.get("theorem") or ""),
            "statement": stmt_map.get(str(node_id), ""),
            "kind": kind,
            "meaning": meaning,
            "is_reasoning": is_reasoning,
            "advice": advice,
            "error_excerpt": err[:400],
            "attempts": status.get("attempts"),
        })
        if is_reasoning is True:
            out["reasoning_gaps"] += 1
        elif is_reasoning is False:
            out["non_reasoning_gaps"] += 1
        else:
            out["unclassified"] += 1

    out["by_kind"] = by_kind
    n_failed = max(out["n_failed"], len(out["gaps"])) or 0
    if n_failed:
        out["reasoning_share"] = round(out["reasoning_gaps"] / n_failed, 4)

    # ---- 一句话诊断（可直接进报告）----
    if not out["gaps"]:
        out["diagnosis"] = ("无遗漏子目标：所有 sorry 均被补全 ⇒ "
                            "蓝图分解的子目标均在大模型能力范围内。")
    else:
        top = sorted(by_kind.items(), key=lambda kv: -kv[1])
        top_desc = "、".join(f"{k}×{v}" for k, v in top[:3])
        out["diagnosis"] = (
            f"{n_failed} 个子目标未能补全（未补全率 "
            f"{n_failed / max(out['n_nodes'], 1):.0%}），"
            f"缺口性质分布：{top_desc}；"
            f"其中**真·推理瓶颈** {out['reasoning_gaps']} 个"
            f"（占失败节点 {out['reasoning_share']:.0%}）。"
            + (f" 注：{out['non_reasoning_gaps']} 个属形式化/检索/计算问题，"
               "不应计入推理能力评估。" if out["non_reasoning_gaps"] else "")
            + (f" 另有 {out['unclassified']} 个未归类，需人工看原始报错。"
               if out["unclassified"] else "")
        )
    return out


# ---------------------------------------------------------------------------
# 汇总多题（离线分析用）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ② 子目标运行期缺口（消费 ctx.subgoal_trace，**不需要 Lean**）
# ---------------------------------------------------------------------------
# 动机：`extract_gaps()` 消费的是 LEAP Stage3（`refine_result`）的产物，
#   而该链在**当前 orchestrator 中并未接入**（见文件末 ORPHAN_NOTE）。
#   但用户 #6 的问题「哪些逻辑点是大模型必须推理出来的」**现在就要能回答**。
#   好在 `sub_goal_solver` 已经把每步子目标的结果写进 `ctx.subgoal_trace`，
#   且求解失败时已有 `_plan_err` 归因（规划错 vs 算不出）——
#   直接消费它即可，零新增开销、零 Lean 依赖。
#
# ★ 判据（与 sub_goal_solver 的既有语义严格对齐，不另立一套）：
#   `sub_goal_solver.py:832-839` 用这组特征判定"该子目标未产出有效结论"：
#     _looks_placeholder(result) or result.startswith("[子目标")
#     or "未产出有效结论" in result or "仍为空转占位" in result
#   以及 `_plan_err` = 结果文本含 依赖/前提/缺少/不成立/无法确定 之一。
#   本模块**复用同一组判据**，保证两侧口径一致（口径漂移会让统计失去意义）。
_PLAN_ERR_MARKERS = ("依赖", "前提", "缺少", "不成立", "无法确定")


# 占位/失败文本特征
# ★ 2026-09-29 修正（实测暴露）：除占位符外，**模型自陈"没做出来"的措辞**
#   也必须算失败，否则「计算未能完成」「无法化简」这类会被当作**成功结论**
#   记入 trace，导致缺口统计系统性偏低（把"没解出来"读成"解出来了"）。
_FAIL_MARKERS = (
    # 上游 sub_goal_solver 已有的占位标记（口径对齐，勿改）
    "[子目标", "未产出有效结论", "仍为空转占位", "[子目标求解失败]",
    # 模型自陈失败（新增，实测补充）
    "未能完成", "无法完成", "无法求解", "求解失败", "无法确定结论",
    "未能求出", "无法求出", "推不出", "未能推出", "无法推出",
    "化简失败", "计算失败", "无法化简", "无法计算", "未能计算",
)

# ★ 2026-10-22（判据误报修复）：结论契约行——子目标给出可判定结论即视为成功。
_CONCLUSION_LINE_RE = re.compile(
    r"本步结论\s*[:：]\s*(?:值|等式|判定)\s*=\s*\S")
# 失败声明必须出现在**开头**这么多个字符内才算（占位符/自陈失败都在开头）。
_HEAD_CHARS = 80


def is_subgoal_failed(result_text: str) -> bool:
    """该子目标是否**未产出有效结论**。

    ★ 2026-09-29 修正：`_plan_err` 标记（依赖/前提/缺少/不成立/无法确定）
      本身是失败信号，但当时用的是**全文包含**判据。
    ★★ 2026-10-22 修正（判据误报，实测证据）：
      22 条被判 ``plan_err`` 的子目标里，**含合规「本步结论」行的 8 条、
      0 条是"开头即失败"**，其余 14 条是正常分析文本 —— 例如
      「图 G 满足 BEST 定理的**前提条件**」、「解族清单…逐族判定：常数解**不存在**」，
      都只是正文里出现了关键词。全文包含判据把这类**正常产出**判成失败，
      直接后果是报表里的「子目标失败率」虚高，且会把不存在的病当成规划缺陷去改。

      新判据（三条，按优先级）：
        ① 空 / 纯空白 / 极短 ⇒ 失败；
        ② **有合规「本步结论: 值|等式|判定 = …」行 ⇒ 成功**（结论契约优先）；
        ③ 失败标记只在前 ``_HEAD_CHARS`` 字符内命中才算失败
           （占位符 `[子目标N 求解失败]` 与自陈"无法推出…"都出现在开头）。
    """
    t = str(result_text or "")
    if not t.strip():
        return True
    if len(t.strip()) < 4:
        return True
    # ② 契约优先：已给出可判定结论 ⇒ 不算失败（哪怕正文里讨论了"不存在/矛盾"）
    if _CONCLUSION_LINE_RE.search(t):
        return False
    head = t.lstrip()[: _HEAD_CHARS]
    if any(m in head for m in _FAIL_MARKERS):
        return True
    if any(m in head for m in _PLAN_ERR_MARKERS):
        return True
    return False


def classify_subgoal_gap(result_text: str) -> str:
    """把子目标求解结果归类成缺口性质。

    ★ 与 `classify_gap()`（消费 Lean 报错）**共用同一套 kind 命名**，
      这样两条数据源可以汇总到同一张表里，不会出现两套口径。
    """
    t = str(result_text or "")
    if any(m in t for m in _PLAN_ERR_MARKERS):
        # 「依赖/前提/缺少」= 规划粒度或前置条件有误 ⇒ 属**规划问题**，
        # 不是模型推理能力不足。映射到 formalization（"不该算作推理瓶颈"）。
        return "formalization"
    # 含数值/算式但没算出来 ⇒ 计算问题
    if re.search(r"计算|化简|求值|数值", t):
        return "computation"
    # 其余（含 unsolved/推不出 语义、或纯占位）⇒ 真正需要推理的点
    return "logical_jump"


def extract_subgoal_gaps(subgoal_trace: list,
                         meta_by_id: dict | None = None) -> dict:
    """从 `ctx.subgoal_trace` 提炼子目标级缺口清单（**不依赖 Lean**）。

    参数
    ----
    subgoal_trace : `ctx.subgoal_trace`（每项含 id/title/description/result）
    meta_by_id    : 可选 {id: {type, calc_kind, depends_on}}（trace 里已有则不用传）

    返回结构与 `extract_gaps()` **完全一致**（便于统一汇总），额外带
    `"source": "subgoal_trace"` 以区分数据源。
    """
    out = {
        "ok": False, "source": "subgoal_trace", "verdict": "unknown",
        "n_nodes": 0, "n_done": 0, "n_failed": 0,
        "gaps": [], "by_kind": {},
        "reasoning_gaps": 0, "non_reasoning_gaps": 0, "unclassified": 0,
        "reasoning_share": 0.0, "diagnosis": "",
    }
    if not isinstance(subgoal_trace, list) or not subgoal_trace:
        out["diagnosis"] = "无 subgoal_trace（未走子目标路径）"
        return out

    meta = meta_by_id or {}
    by_kind: dict = {}
    for item in subgoal_trace:
        if not isinstance(item, dict):
            continue
        out["n_nodes"] += 1
        res = item.get("result")
        if not is_subgoal_failed(res):
            out["n_done"] += 1
            continue
        out["n_failed"] += 1
        kind = classify_subgoal_gap(res)
        meaning, advice = gap_meaning(kind)
        is_reasoning = counts_as_reasoning(kind)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        sid = item.get("id")
        extra = meta.get(sid, {}) if isinstance(meta, dict) else {}
        out["gaps"].append({
            "node_id": str(sid),
            "theorem": "",
            "statement": str(item.get("description") or item.get("title") or ""),
            "title": str(item.get("title") or ""),
            "sg_type": str(item.get("type") or extra.get("type") or ""),
            "calc_kind": str(item.get("calc_kind") or ""),
            "depends_on": item.get("depends_on") or extra.get("depends_on") or [],
            "kind": kind,
            "meaning": meaning,
            "is_reasoning": is_reasoning,
            "advice": advice,
            "error_excerpt": str(res)[:400],
            "attempts": None,
        })
        if is_reasoning is True:
            out["reasoning_gaps"] += 1
        elif is_reasoning is False:
            out["non_reasoning_gaps"] += 1
        else:
            out["unclassified"] += 1

    out["ok"] = True
    out["by_kind"] = by_kind
    if out["n_failed"]:
        out["reasoning_share"] = round(
            out["reasoning_gaps"] / out["n_failed"], 4)

    if not out["gaps"]:
        out["diagnosis"] = (
            f"全部 {out['n_nodes']} 个子目标均产出有效结论 ⇒ "
            "**子目标分解有效**（难题被成功简化为可独立求解的简单题）。")
    else:
        top = sorted(by_kind.items(), key=lambda kv: -kv[1])
        top_desc = "、".join(f"{k}×{v}" for k, v in top[:3])
        out["diagnosis"] = (
            f"{out['n_failed']}/{out['n_nodes']} 个子目标未产出有效结论"
            f"（{out['n_failed'] / max(out['n_nodes'], 1):.0%}），"
            f"缺口性质：{top_desc}；"
            f"其中**真·推理瓶颈** {out['reasoning_gaps']} 个"
            f"（占失败子目标 {out['reasoning_share']:.0%}）。")
    return out
def _stmt_map_from_blueprint(blueprint: dict | None) -> dict:
    """从 blueprint 还原 {节点id: statement}（兼容 list / dict 两种 nodes 形态）。"""
    stmt_map: dict = {}
    try:
        nodes = (blueprint or {}).get("nodes")
        if isinstance(nodes, list):
            for n in nodes:
                if isinstance(n, dict) and n.get("id") is not None:
                    stmt_map[str(n["id"])] = str(n.get("statement") or "")
        elif isinstance(nodes, dict):
            for k, v in nodes.items():
                if isinstance(v, dict):
                    stmt_map[str(k)] = str(v.get("statement") or "")
    except Exception:  # noqa: BLE001
        pass
    return stmt_map


def merge_gap_sources(*results: dict) -> dict:
    """把多个数据源（subgoal_trace + refine_result）的缺口清单**合并**成一份。

    用途：两条链都在跑时，得到的是**同一批子目标的两个独立视角**——
      ① subgoal_trace 视角：自然语言求解有没有得出结论；
      ② refine_result 视角：Lean 形式化后 sorry 补不补得上。
    两者同时失败 ⇒ 该子目标是**高置信度的推理瓶颈**（两个证据源一致）；
    只有一个失败 ⇒ 可能是表示/工具问题。这个交集信息比任一单源都强。
    """
    merged = {
        "ok": False, "sources": [], "n_nodes": 0, "n_done": 0, "n_failed": 0,
        "by_kind": {}, "gaps": [],
        "reasoning_gaps": 0, "non_reasoning_gaps": 0, "unclassified": 0,
        "reasoning_share": 0.0, "diagnosis": "",
        "in_both_failed": [],       # ★ 两个源都失败 ⇒ 高置信推理瓶颈
    }
    for r in results:
        if not isinstance(r, dict) or not r.get("ok"):
            continue
        merged["sources"].append(str(r.get("source") or "?"))
        merged["n_nodes"] += int(r.get("n_nodes") or 0)
        merged["n_done"] += int(r.get("n_done") or 0)
        merged["n_failed"] += int(r.get("n_failed") or 0)
        for k, v in (r.get("by_kind") or {}).items():
            merged["by_kind"][k] = merged["by_kind"].get(k, 0) + int(v)
        merged["gaps"].extend(r.get("gaps") or [])
        merged["reasoning_gaps"] += int(r.get("reasoning_gaps") or 0)
        merged["non_reasoning_gaps"] += int(r.get("non_reasoning_gaps") or 0)
        merged["unclassified"] += int(r.get("unclassified") or 0)

    if not merged["sources"]:
        merged["diagnosis"] = "无任何可用缺口数据源"
        return merged
    merged["ok"] = True

    # ★ 交集：跨源按 node_id 找"都失败"的节点
    try:
        from collections import defaultdict
        failed_by_src = defaultdict(set)
        for r in results:
            if not isinstance(r, dict) or not r.get("ok"):
                continue
            src = str(r.get("source") or "?")
            for g in (r.get("gaps") or []):
                failed_by_src[src].add(str(g.get("node_id")))
        if len(failed_by_src) >= 2:
            sets = list(failed_by_src.values())
            common = set.intersection(*sets)
            merged["in_both_failed"] = sorted(common)
    except Exception:  # noqa: BLE001
        pass

    if merged["n_failed"]:
        merged["reasoning_share"] = round(
            merged["reasoning_gaps"] / merged["n_failed"], 4)
    merged["diagnosis"] = (
        f"数据源 {merged['sources']}；共 {merged['n_failed']} 个缺口，"
        f"其中真·推理瓶颈 {merged['reasoning_gaps']} 个"
        f"（{merged['reasoning_share']:.0%}）；"
        f"**双源一致的推理瓶颈** {len(merged['in_both_failed'])} 个"
        f"（自然语言求解与 Lean 形式化都失败 ⇒ 高置信度）。")
    return merged


# ---------------------------------------------------------------------------
# ORPHAN_NOTE —— 哪些函数在什么条件下才有效（**判读前必读**，2026-09-30）
# ---------------------------------------------------------------------------
# 本模块有三个入口，**数据源不同、可用性不同**，绝不可混为一谈：
#
#   ① `extract_subgoal_gaps(subgoal_trace)`  ← **现役，每轮都可用**
#        消费 `ctx.subgoal_trace`（sub_goal_solver 已写入）。
#        回答：自然语言求解时，"每步子目标有没有得出结论"。
#        接入点：`orchestrator._compute_subgoal_gaps()` → diag["subgoal_gaps"]。
#
#   ② `extract_gaps(refine_result)`  ← **孤儿（ORPHAN），恒空**
#        消费 `ctx.refine_result` —— 该字段由 LeanRefinerAgent 写入，
#        而 **LeanRefinerAgent 已于 2026-09-06 从链路移除**
#        （`sub_goal_solver.py:1602-1605`：平台无 Lean，整树翻译+编译只空转）。
#        ⇒ 现链路下它**恒返回 `{"ok": False, "diagnosis": "无 refine_result"}`**。
#        **保留原因**：①历史结果文件里可能有该字段，离线复算用得上；
#                    ②将来若恢复 Stage3 精炼链，`merge_gap_sources()` 可直接
#                      做"双源一致"的高置信推理瓶颈判定，无需重写。
#        **判读禁区**：不得因它返回空就宣称"没有 sorry 缺口"。
#
# ★ 一句话：**要看推理瓶颈，用 ①；要看 Lean 抓的命题缺陷，用 ②；
#   ③ 现在没数据，别拿它下任何结论。**
