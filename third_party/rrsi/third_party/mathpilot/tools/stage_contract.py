# -*- coding: utf-8 -*-
"""阶段契约盘点：把 14 个阶段的 **harness 五维**现状扫出来（只读，不改行为）。

为什么要扫而不是人工记
----------------------
用户要求「每个环节的这 5 条都能修改且检测」。开工前必须有一张**准确的现状矩阵**，
否则后面全是盲改。人工梳理容易漏（尤其 agent/ 有 99 个 py），所以用 AST。

五维（harness 定义）
--------------------
① 提示词：`prompts.*` 的引用
② 控制流：读 config 的哪些字段（轮数 / 停止条件 / 开关）
③ 工具：`tools.*` 的引用与工具调用
④ 记忆：`memory` / `theorem_memory` 之类
⑤ 上下文管理：切片截断（`[:N]`）/ budget / truncate

输出
----
一张表：阶段 × 五维，标 ✓(引用数) 或 ✗，并列出具体标识符，便于定位改哪里。

用法
----
    python tools/stage_contract.py            # 打表
    python tools/stage_contract.py --detail   # 附带每维的具体标识符
    python tools/stage_contract.py --md out.md
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

AGENT_DIR = os.path.join(_ROOT, "agent")

# 14 个阶段（与 orchestrator 的 _stage_start 一致；顺序即执行顺序）
STAGES = [
    "2.6_pre_audit", "2.7_subgoal_main", "3_solve", "3.2_complete",
    "3.3_improve", "3.4_collab", "4.0_normalize", "4.1_self_audit",
    "4.2_falsify", "4.3_lean_check", "4.4_math_check", "4.5_vote",
    "4.6_rewrite", "6_format",
]


def _iter_py(root: str):
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def build_index() -> dict:
    """函数名 → (文件路径, AST 节点)。同名取第一个（agent 内基本不重名）。"""
    idx = {}
    for path in _iter_py(AGENT_DIR):
        try:
            tree = ast.parse(io.open(path, encoding="utf-8", errors="ignore").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                idx.setdefault(node.name, (path, node))
    return idx


def map_stage_to_funcs() -> dict:
    """解析 orchestrator*.py：`_stage_start(ctx, "X")` 之后紧跟的调用就是该阶段的实现。"""
    out: dict[str, list[str]] = {}
    for name in os.listdir(AGENT_DIR):
        if not (name.startswith("orchestrator") and name.endswith(".py")):
            continue
        path = os.path.join(AGENT_DIR, name)
        try:
            src = io.open(path, encoding="utf-8", errors="ignore").read()
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # 按行号排序所有语句，便于取"下一个调用"
        stmts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and node.lineno:
                stmts.append(node)
        stmts.sort(key=lambda n: n.lineno)
        for i, node in enumerate(stmts):
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "_stage_start"):
                continue
            if len(call.args) < 2:
                continue
            arg = call.args[1]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            stage = arg.value
            # 往后找第一个 Call，取被调函数名
            for nxt in stmts[i + 1:]:
                c = nxt.value
                if isinstance(c, ast.Call):
                    f2 = c.func
                    name2 = (f2.attr if isinstance(f2, ast.Attribute)
                             else f2.id if isinstance(f2, ast.Name) else "")
                    if name2 and name2 != "record":
                        out.setdefault(stage, []).append(name2)
                        break
                # 下一个 _stage_start 到了就停
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                        and c.func.attr == "_stage_start"):
                    break
    return out


def scan_func(node: ast.AST) -> dict:
    """扫一个函数体，返回五维各自命中的标识符集合。"""
    dims = {"prompt": set(), "control": set(), "tool": set(),
            "memory": set(), "context": set()}
    for n in ast.walk(node):
        # ① 提示词
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("prompts"):
            dims["prompt"].add("from %s" % n.module)
        if isinstance(n, ast.Name) and n.id == "prompts":
            dims["prompt"].add("prompts.*")
        if isinstance(n, ast.Attribute) and "PROMPT" in n.attr.upper():
            dims["prompt"].add(n.attr)

        # ② 控制流：读 config 字段
        if isinstance(n, ast.Attribute) and n.attr == "config":
            pass
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == "getattr" and len(n.args) >= 2:
            a1 = n.args[1]
            if isinstance(a1, ast.Constant) and isinstance(a1.value, str):
                dims["control"].add(a1.value)

        # ③ 工具
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("tools"):
            dims["tool"].add(n.module.split(".")[-1])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in ("mcp_run_code", "leansearch", "verify_answer",
                               "cross_check", "search"):
                dims["tool"].add(n.func.attr)

        # ④ 记忆
        if isinstance(n, ast.Name) and "memory" in n.id.lower():
            dims["memory"].add(n.id)
        if isinstance(n, ast.Attribute) and "memory" in n.attr.lower():
            dims["memory"].add(n.attr)

        # ⑤ 上下文管理：切片截断 / budget
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            up = n.slice.upper
            if isinstance(up, ast.Constant) and isinstance(up.value, int):
                dims["context"].add("[:%d]" % up.value)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in ("truncate", "budget", "compress"):
            dims["context"].add(n.func.attr)
        if isinstance(n, ast.Name) and "budget" in n.id.lower():
            dims["context"].add(n.id)
    return dims


def render(detail: bool = False) -> str:
    idx = build_index()
    mapping = map_stage_to_funcs()
    lines = []
    lines.append("# 阶段契约盘点（harness 五维）\n")
    lines.append("| 阶段 | 实现函数 | 提示词 | 控制流 | 工具 | 记忆 | 上下文 |")
    lines.append("|---|---|:--:|:--:|:--:|:--:|:--:|")
    details = []
    covered = {"prompt": 0, "control": 0, "tool": 0, "memory": 0, "context": 0}
    for stage in STAGES:
        funcs = mapping.get(stage) or []
        cells = {}
        found = {k: set() for k in covered}
        for fn in funcs:
            hit = idx.get(fn)
            if not hit:
                continue
            dims = scan_func(hit[1])
            for k, v in dims.items():
                found[k] |= v
        for k in covered:
            n = len(found[k])
            cells[k] = ("✓(%d)" % n) if n else "✗"
            if n:
                covered[k] += 1
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            stage, ", ".join("`%s`" % f for f in funcs) or "**未解析**",
            cells["prompt"], cells["control"], cells["tool"],
            cells["memory"], cells["context"]))
        if detail and any(found.values()):
            details.append("### %s" % stage)
            for k, label in (("prompt", "提示词"), ("control", "控制流"),
                             ("tool", "工具"), ("memory", "记忆"),
                             ("context", "上下文")):
                if found[k]:
                    details.append("- %s：%s" % (label, ", ".join(
                        "`%s`" % x for x in sorted(found[k])[:8])))
            details.append("")

    lines.append("")
    lines.append("## 覆盖统计（有该维证据的阶段数 / 14）\n")
    lines.append("| 维度 | 阶段数 | 覆盖率 |")
    lines.append("|---|---:|---:|")
    for k, label in (("prompt", "提示词"), ("control", "控制流"),
                     ("tool", "工具"), ("memory", "记忆"),
                     ("context", "上下文")):
        lines.append("| %s | %d | %.0f%% |" % (label, covered[k],
                                              100.0 * covered[k] / len(STAGES)))
    lines.append("")
    if details:
        lines.append("## 明细\n")
        lines.extend(details)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阶段契约盘点（只读）")
    ap.add_argument("--detail", action="store_true", help="附带具体标识符")
    ap.add_argument("--md", default="", help="写入 Markdown 文件")
    args = ap.parse_args(argv)
    text = render(args.detail)
    print(text)
    if args.md:
        io.open(args.md, "w", encoding="utf-8").write(text)
        print("\n已写入：%s" % args.md, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
