# -*- coding: utf-8 -*-
"""环节卡生成器 —— 把每个阶段的可编辑面与契约从源码里抽出来（只读）。

为什么需要
==========
14 个环节签名统一为 ``(self, ctx[, tier])``，可插拔已经成立；但 ``ctx`` 是**隐式契约**：
读哪些字段、写哪些字段、失败怎么表达，全埋在函数体里 ⇒ 无法单独测、无法只回滚一处。

本工具把每个环节的契约自动抽出来，生成「环节卡」：

    输入（读 ctx 的哪些字段） / 输出（写哪些字段） / diag 键 /
    五维各自的实证（提示词、控制流、工具、记忆、上下文切片）

⇒ 手抄会过期，AST 不会。

用法
====
    python tools/stage_card.py --list            # 14 个环节一览
    python tools/stage_card.py --stage 4.1       # 单张卡
    python tools/stage_card.py --all             # 全部卡片（markdown）
    python tools/stage_card.py --all --json out.json

⚠ 只读：不 import agent（避免副作用），全部按源码解析。
⚠ `role` / `fail_policy` 两栏是**人工维护**的知识，写在 STAGES 表里；
   其余全部由 AST 生成，两者在输出中明确分开标注。
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# stage_id, 名称, 文件, 函数, 职责（人工）, 失败语义（人工）, 真实实现模块（人工，薄壳用）
STAGES = [
    ("2.6_pre_audit", "题意理解 + 题型判定", "agent/orchestrator__prep.py",
     "pre_audit_stage", "判断题型/领域，产出 problem_goal 与 lean_applicable",
     "判定失败按保守分支继续，不阻断",
     ["agent/classifier.py", "agent/question_type.py", "agent/problem_goal.py"]),    ("2.7_subgoal_main", "定理检索 → 蓝图 → 子目标 → 求解 → 合并", "agent/orchestrator__solve.py",
     "subgoal_main_path", "整题分解与求解的主路径（K11 根因所在）",
     "子目标失败可重试；全失败则交下游",
     ["agent/sub_goal_solver*", "agent/blueprint_planner*", "agent/theorem_hint.py"]),
    ("3_solve", "求解（候选池生成）", "agent/orchestrator__solve.py",
     "solve_stage", "生成候选解答池", "失败走 _fallback_direct 哨兵",
     ["agent/solver*"]),
    ("3.2_complete", "截断候选补全", "agent/orchestrator__solve.py",
     "complete_truncated", "对 finish_reason=length 的候选补写",
     "补写失败保留原候选", []),
    ("3.3_improve", "候选级自改进（默认关闭）", "agent/orchestrator__solve.py",
     "self_improve_step", "对候选做无条件自改进（已内化为 3 遍生成）",
     "失败保留原候选", []),
    ("3.4_collab", "三角色协作深化", "agent/orchestrator__collab.py",
     "collaborative_deep_improve", "求解/审核/规划三角色协作改进候选",
     "协作失败保留原候选", []),
    ("4.0_normalize", "答案规范化", "agent/answer_normalizer.py",
     "normalize_stage", "把候选答案规范化并分流形态（numeric/symbolic/proof/…）",
     "不可解析标 unparseable，仍放行", []),
    ("4.1_self_audit", "AI 自审逻辑错误", "agent/self_auditor.py",
     "self_audit_stage", "只审逻辑不审数值，能指出具体逻辑错误才淘汰",
     "解析失败/LLM 故障 ⇒ 一律保留（防误杀）", []),
    ("4.2_falsify", "对抗证伪", "agent/adversarial_gate.py",
     "adversarial_gate_stage", "机检反例 + LLM 对抗验证，证伪即淘汰",
     "跳过/故障 ⇒ 保留候选", []),
    ("4.3_lean_check", "Lean 代回题目判断", "agent/lean_backcheck.py",
     "lean_backcheck_stage", "题目+答案翻译为 Lean 命题，编译判定",
     "Lean 不可用/不适用 ⇒ 豁免，不淘汰", ["tools/lean_local/lean_gate.py"]),
    ("4.4_math_check", "数学工具计算检测", "agent/math_check_gate.py",
     "math_check_stage", "SymPy 精确判定计算是否正确",
     "unknown 放行；全拒回退保留原池", []),
    ("4.5_vote", "多答案判断 + 投票", "agent/vote_gate.py",
     "vote_stage", "先判断是否多答案，再投票选答案",
     "零票走兜底分支", []),
    ("4.6_rewrite", "池空重写", "agent/rewrite_gate.py",
     "rewrite_stage", "诊断错误 → 重建蓝图/子目标 → 重写并重过五道链",
     "达上限仍失败则返回错误状态", []),
    ("6_format", "终答格式化", "agent/orchestrator__final_gate.py",
     "format_stage", "把选中答案格式化为最终输出", "格式化失败回退原文",
     ["agent/formatter*"]),
]

# config 名误匹配黑名单（`config.py` 这类）
_CFG_DENY = {"py", "json", "yaml", "toml", "path", "name", "get", "items", "key", "value"}

_CTX_RE = re.compile(r"\bctx\.([a-zA-Z_][a-zA-Z0-9_]*)")
_GETATTR_CTX_RE = re.compile(r"getattr\(\s*ctx\s*,\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']")
_CFG_RE = re.compile(r"(?:self\.)?config\.([a-zA-Z_][a-zA-Z0-9_]*)")
_GETATTR_RE = re.compile(r"getattr\(\s*(?:self\.)?(?:config|ctx\.config)\s*,\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']")
_SW_RE = re.compile(r"_sw_(?:get_)?(?:bool|num|str)\(\s*[\"']([a-zA-Z0-9_]+)[\"']")
_SLICE_RE = re.compile(r"\[:\s*(\d{2,6})\s*\]")
_DIAG_SUB_RE = re.compile(r"(?:ctx\.)?diag\[\s*[\"']([a-zA-Z0-9_]+)[\"']\s*\]")
_TOK_RE = re.compile(r"max_tokens\s*=\s*(\d+)")
_TEMP_RE = re.compile(r"temperature\s*=\s*([0-9.]+)")
# 模块级数值常量：_PROBLEM_MAX = 4000
_MOD_CONST_RE = re.compile(r"^_([A-Z][A-Z0-9_]*)\s*=\s*(\d+(?:\.\d+)?)\s*$", re.M)


def _func_source(path: str, name: str):
    """返回 (源码切片, 起始行, 行数)；找不到返回 (None, 0, 0)。"""
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return None, 0, 0
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
            return seg, node.lineno, node.end_lineno - node.lineno + 1
    return None, 0, 0


def _assign_targets(seg: str) -> set:
    """函数体里被赋值的 ctx 字段（写）。"""
    writes = set()
    try:
        tree = ast.parse(seg)
    except SyntaxError:
        return writes
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                            and sub.value.id == "ctx":
                        writes.add(sub.attr)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute) \
                and isinstance(node.target.value, ast.Name) and node.target.value.id == "ctx":
            writes.add(node.target.attr)
    return writes


def _imports_of(path: str) -> dict:
    """文件级的 prompts / tools 导入常量。"""
    out = {"prompts": set(), "tools": set()}
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return out
    src = re.sub(r"import\s*\(([^)]*)\)", lambda m: "import " + m.group(1).replace("\n", " "), src)
    for pkg in ("prompts", "tools"):
        for names in re.findall(r"from\s+(?:\.\.)?%s[\w\.]*\s+import\s+([^\n]+)" % pkg, src):
            for n in names.split(","):
                n = n.strip().split(" as ")[0].strip()
                if re.match(r"^[a-zA-Z_][\w]*$", n):
                    out[pkg].add(n)
    return out


def _helper_sources(path: str) -> str:
    """文件内**非环节函数**的源码（模块级 helper：开关读取、常量包装等）。"""
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return ""
    lines = src.splitlines()
    stage_funcs = {r[3] for r in STAGES}
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name not in stage_funcs:
            out.append("\n".join(lines[node.lineno - 1: node.end_lineno]))
    return "\n".join(out)


def _mod_consts(path: str) -> dict:
    """模块级数值常量 {名字: 值}，用于把 [:N] 还原成有名字的截断。"""
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    return {m.group(1): m.group(2) for m in _MOD_CONST_RE.finditer(src)}


def _scan_file(rel: str) -> dict:
    """整文件级五维扫描（用于薄壳环节的真实实现模块）。"""
    path = os.path.join(_ROOT, rel)
    try:
        src = io.open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {"error": "文件不存在"}
    imps = _imports_of(path)
    return {
        "prompts": sorted(c for c in imps["prompts"] if re.search(r"\b%s\b" % c, src)),
        "tools": sorted(c for c in imps["tools"] if re.search(r"\b%s\b" % c, src)),
        "config": sorted((set(_CFG_RE.findall(src)) | set(_GETATTR_RE.findall(src)) | set(_SW_RE.findall(src)))
                         - _CFG_DENY),
        "caps_named": {k: v for k, v in _mod_consts(path).items()
                       if re.search(r"\b_%s\b" % k, src)},
        "lines": len(src.splitlines()),
    }


def analyze(stage_row) -> dict:
    sid, name, path, func, role, fail, delegates = stage_row
    full = os.path.join(_ROOT, path)
    seg, lineno, nlines = _func_source(full, func)
    if seg is None:
        return {"stage": sid, "name": name, "impl": "%s::%s" % (path, func),
                "error": "函数未找到"}
    helper = _helper_sources(full)
    scope = seg + "\n" + helper          # 函数体 + 同文件 helper
    writes = _assign_targets(seg)
    reads = set(_CTX_RE.findall(seg)) | set(_GETATTR_CTX_RE.findall(seg))
    imps = _imports_of(full)
    prompts = sorted(c for c in imps["prompts"] if re.search(r"\b%s\b" % c, seg))
    tools = sorted(c for c in imps["tools"] if re.search(r"\b%s\b" % c, scope))
    cfg = (set(_CFG_RE.findall(scope)) | set(_GETATTR_RE.findall(scope)) | set(_SW_RE.findall(scope))) - _CFG_DENY
    # 截断：优先还原成模块常量名（_PROBLEM_MAX=4000），裸数字另列
    consts = _mod_consts(full)
    used_consts = {n: v for n, v in consts.items() if re.search(r"\b_%s\b" % n, seg)}
    bare = sorted({int(x) for x in _SLICE_RE.findall(seg)})
    return {
        "stage": sid, "name": name,
        "impl": "%s::%s" % (path, func),
        "lineno": lineno, "nlines": nlines,
        "role": role, "fail_policy": fail,
        "reads": sorted(reads),
        "writes": sorted(writes),
        "diag_keys": sorted(set(_DIAG_SUB_RE.findall(scope))),
        "prompts": prompts,
        "tools": tools,
        "config": sorted(cfg),
        "caps_named": used_consts,
        "caps_bare": bare,
        "max_tokens": sorted(set(_TOK_RE.findall(scope))),
        "temperature": sorted(set(_TEMP_RE.findall(scope))),
        "delegates": _scan_delegates(delegates or []),
    }


def _scan_delegates(pats: list) -> list:
    """展开 delegate 通配（`agent/sub_goal_solver*`），逐文件扫描并汇总。"""
    import glob as _glob
    out, seen = [], set()
    for pat in pats:
        for p in sorted(_glob.glob(os.path.join(_ROOT, pat))):
            rel = os.path.relpath(p, _ROOT).replace("\\", "/")
            if "__pycache__" in rel or rel in seen or not rel.endswith(".py"):
                continue
            seen.add(rel)
            out.append(dict(_scan_file(rel), rel=rel))
    return out


def render(card: dict) -> str:
    if card.get("error"):
        return "### %s · %s\n- ❌ %s\n" % (card["stage"], card["name"], card["error"])
    L = []
    L.append("### %s · %s" % (card["stage"], card["name"]))
    L.append("")
    L.append("- **实现**：`%s`（第 %d 行起，%d 行）" % (card["impl"], card["lineno"], card["nlines"]))
    L.append("- **职责**（人工）：%s" % card["role"])
    # 输入：只读字段单列；既读又写的归"就地更新"（多为累加/改写）
    w = set(card["writes"])
    r = set(card["reads"])
    pure_in, inplace = sorted(r - w), sorted(r & w)
    L.append("- **输入**（只读 ctx）：%s" % (", ".join("`%s`" % x for x in pure_in) or "—"))
    if inplace:
        L.append("- **就地更新**（读+写）：%s" % ", ".join("`%s`" % x for x in inplace))
    L.append("- **输出**（写 ctx）：%s" % (", ".join("`%s`" % x for x in card["writes"]) or "—"))
    L.append("- **① 提示词**：%s" % (", ".join("`%s`" % x for x in card["prompts"]) or "—"))
    L.append("- **② 控制流**：%s" % (", ".join("`%s`" % x for x in card["config"]) or "—"))
    L.append("- **③ 工具**：%s" % (", ".join("`%s`" % x for x in card["tools"]) or "—"))
    L.append("- **④ 记忆**：%s" % ("见 `prompts/error_lessons.py`（全链共享注入点）"
                                   if "match_lessons" in card["tools"] else "—"))
    ctx = []
    if card["caps_named"]:
        ctx.append("命名常量 " + " / ".join("%s≤%s" % (k, v) for k, v in card["caps_named"].items()))
    if card["caps_bare"]:
        ctx.append("裸切片 " + "/".join(str(c) for c in card["caps_bare"]))
    if card["max_tokens"]:
        ctx.append("max_tokens=" + "/".join(card["max_tokens"]))
    if card["temperature"]:
        ctx.append("temperature=" + "/".join(card["temperature"]))
    L.append("- **⑤ 上下文**：%s" % ("；".join(ctx) or "—"))
    L.append("- **diag 键**：%s" % (", ".join("`%s`" % x for x in card["diag_keys"]) or "—（诊断字段见「输出」）"))
    L.append("- **失败语义**（人工）：%s" % card["fail_policy"])
    dels = [d for d in card.get("delegates", []) if not d.get("error")]
    if dels:
        agg = {"prompts": set(), "tools": set(), "config": set(), "caps": {}}
        for d in dels:
            agg["prompts"] |= set(d["prompts"])
            agg["tools"] |= set(d["tools"])
            agg["config"] |= set(d["config"])
            agg["caps"].update(d["caps_named"])
        L.append("")
        L.append("- **真实实现**（薄壳，共 %d 个模块 / %d 行）："
                 % (len(dels), sum(d["lines"] for d in dels)))
        if agg["prompts"]:
            L.append("  - 提示词（%d）：%s" % (len(agg["prompts"]),
                     ", ".join("`%s`" % x for x in sorted(agg["prompts"])[:8])
                     + ("…" if len(agg["prompts"]) > 8 else "")))
        if agg["tools"]:
            L.append("  - 工具（%d）：%s" % (len(agg["tools"]),
                     ", ".join("`%s`" % x for x in sorted(agg["tools"])[:6])))
        if agg["config"]:
            L.append("  - 开关/配置（%d）：%s" % (len(agg["config"]),
                     ", ".join("`%s`" % x for x in sorted(agg["config"])[:8])
                     + ("…" if len(agg["config"]) > 8 else "")))
        if agg["caps"]:
            L.append("  - 数值常量（%d）：%s" % (len(agg["caps"]),
                     " / ".join("%s≤%s" % (k, v) for k, v in list(agg["caps"].items())[:6])))
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="环节卡生成器")
    ap.add_argument("--list", action="store_true", help="列出全部环节")
    ap.add_argument("--stage", help="只输出某环节（如 4.1）")
    ap.add_argument("--all", action="store_true", help="输出全部卡片")
    ap.add_argument("--json", metavar="PATH", help="导出 JSON")
    args = ap.parse_args(argv)

    rows = STAGES
    if args.stage:
        rows = [r for r in STAGES if r[0].startswith(args.stage)]
        if not rows:
            print("未找到环节：%s" % args.stage, file=sys.stderr)
            return 2
    cards = [analyze(r) for r in rows]

    if args.list:
        print("| 环节 | 职责 | 实现 | 行数 |")
        print("|---|---|---|---:|")
        for c in cards:
            print("| `%s` | %s | `%s` | %s |" % (
                c["stage"], c.get("role", ""), c.get("impl", ""), c.get("nlines", "-")))
    elif args.all or args.stage:
        print("# 环节卡\n")
        for c in cards:
            print(render(c))
    else:
        print("用 --list / --stage / --all 之一", file=sys.stderr)
        return 1

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as fh:
            json.dump(cards, fh, ensure_ascii=False, indent=1)
        print("\n[json] %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
