# -*- coding: utf-8 -*-
"""生成全仓「代码架构文档」（详细到函数）。

设计原则：
  · 从 AST 生成，不手工摘抄（手写必漏、必过期）；
  · 按「分层」组织，而非目录字母序（人理解的是职责）；
  · 每个文件给：层 / 一句话职责 / 规模 / 类与方法清单 / 顶层函数清单；
  · 可重跑：代码变了就重跑。

用法：D:/python/python.exe tools/gen_architecture_doc.py
输出：docs/代码架构_<日期>.md
"""
from __future__ import annotations

import ast
import datetime
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"D:/对标leap的数学智能体项目"
OUT = os.path.join(ROOT, "docs", f"代码架构_{datetime.date.today().isoformat()}.md")

SKIP_DIRS = {"vendor", "lean下载版", "lean-lsp-mcp", ".git", "__pycache__",
             ".workbuddy", "node_modules", ".lake", "output", "论文实践",
             "测试工具", "imo_steps", "ripgrep-14.1.1-x86_64-pc-windows-msvc",
             "ripgrep-14.1.1-x86_64-unknown-linux-musl", "与lean相关的插件",
             "代码审查_2026-10-01"}

# 分层规则（顺序敏感：具体的在前）
LAYER_RULES = [
    ("L5 入口", ["cli/"]),
    ("L5 入口", ["run_eval.py", "run_research.py", "main.py"]),
    ("L5 入口", ["deploy/"]),
    ("L4 编排", ["user_agent.py", "agent/orchestrator.py"]),
    # ★ 2026-10-07 架构解耦（方案 架构解耦方案_2026-10-07.md）：
    #   · 新增 `contracts/` 零依赖契约层（Budget/Finding/BugReport + TaskContext Protocol
    #     + switch_registry 已下沉于此）——必须排在 `agent/` 之前，否则会被归到「其它」层。
    #   · `agent/base.py` 与 `agent/switch_registry.py` 现为**re-export shim**（指向 contracts/），
    #     仍判 L3 契约；顺序必须在 `agent/`（L2 业务）之前 —— 匹配是前缀匹配，靠顺序区分。
    #   · `utils/` 现位于 `tools/` **之下**（目标分层 contracts ← utils ← tools ← agent），
    #     故由 L1 工具降为 L0 基础。
    ("L3 契约", ["contracts/"]),
    ("L3 契约", ["agent/base.py"]),
    ("L3 契约", ["agent/switch_registry.py"]),
    ("L2 业务", ["agent/"]),
    ("L2 业务", ["prompts/"]),
    ("L1 工具", ["tools/"]),
    ("L0 基础", ["utils/"]),
    ("测试", ["tests/"]),
    ("一次性脚本", ["scripts/"]),
]
LAYER_ORDER = ["L5 入口", "L4 编排", "L3 契约", "L2 业务", "L1 工具",
               "测试", "一次性脚本", "其它"]
LAYER_DESC = {
    "L5 入口": "参数解析、启动、部署。仅依赖 L4",
    "L4 编排": "流程调度、阶段串联、异常处理。可依赖 L3/L2",
    "L3 契约": "数据结构、接口、常量、开关注册。**最稳定层**，无业务依赖",
    "L2 业务": "具体业务逻辑（求解/验证/格式化/子目标/提示词）",
    "L1 工具": "通用能力与外部系统对接（报告/云端/Lean 桥接/LLM 客户端）",
    "测试": "单元与契约测试",
    "一次性脚本": "带日期的一次性分析脚本，跑完留档",
    "其它": "未归入以上各层",
}


def classify(path: str) -> str:
    for layer, prefixes in LAYER_RULES:
        for p in prefixes:
            if path == p or path.startswith(p):
                return layer
    return "其它"


def first_sentence(text, n=110):
    if not text:
        return ""
    s = re.sub(r"\*\*", "", str(text))   # 2026-10-01：剥 markdown 强调，防截断在 ** 中间
    s = re.sub(r"[=\-*_]{3,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    for sep in ("。", "；", "，", "—— ", ". "):
        i = s.find(sep)
        if 0 < i <= n:
            return s[: i + len(sep)].strip()
    return s[:n] + ("…" if len(s) > n else "")


def module_doc(rel):
    """模块级 docstring（★ 本项目大量文件把它写在 from __future__ 之后）。"""
    try:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore").read())
    except Exception:
        return ""
    for st in tree.body[:8]:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                and isinstance(st.value.value, str):
            t = st.value.value.strip()
            if len(t) >= 8 and not t.startswith("coding"):
                return t
    return ""


def sig(fn) -> str:
    a = fn.args
    parts = [x.arg for x in getattr(a, "posonlyargs", [])]
    parts += [x.arg for x in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    parts += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = ""
    if fn.returns is not None:
        try:
            ret = " -> " + ast.unparse(fn.returns)
        except Exception:
            ret = ""
    pre = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
    return f"{pre}{fn.name}({', '.join(parts)}){ret}"


def collect(rel):
    """返回 (classes, functions)。classes: [(name, line, doc, [(sig, line, doc)])]"""
    try:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return [], []
    classes, funcs = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if m.name.startswith("__") and m.name not in ("__init__", "__call__"):
                        continue
                    methods.append((sig(m), m.lineno,
                                    first_sentence(ast.get_docstring(m), 80)))
            classes.append((node.name, node.lineno,
                            first_sentence(ast.get_docstring(node), 100), methods))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append((sig(node), node.lineno,
                          first_sentence(ast.get_docstring(node), 90)))
    return classes, funcs


# ---------------- 收集
files = []
for root, dirs, fs in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in fs:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        files.append(rel)
files.sort()

by_layer = {}
for rel in files:
    src = open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore").read()
    nlines = len(src.splitlines())
    classes, funcs = collect(rel)
    layer = classify(rel)
    by_layer.setdefault(layer, []).append({
        "rel": rel, "lines": nlines, "duty": first_sentence(module_doc(rel), 130),
        "classes": classes, "funcs": funcs,
    })

# ---------------- 出 Markdown
L = []
L.append(f"# MathPilot 代码架构（自动生成）\n")
L.append(f"**生成时间**：{datetime.datetime.now():%Y-%m-%d %H:%M}　"
         f"**来源**：`tools/gen_architecture_doc.py`（AST 提取，非手工摘抄）\n")
tot_f = sum(len(v) for v in by_layer.values())
tot_lines = sum(x["lines"] for v in by_layer.values() for x in v)
tot_cls = sum(len(x["classes"]) for v in by_layer.values() for x in v)
tot_fn = sum(len(x["funcs"]) + sum(len(c[3]) for c in x["classes"])
             for v in by_layer.values() for x in v)
L.append(f"**规模**：{tot_f} 个 Python 文件 / {tot_lines:,} 行 / "
         f"{tot_cls} 个类 / **{tot_fn} 个函数与方法**\n")
L.append("> ⚠ 本文件由脚本生成：代码改动后请重跑 "
         "`D:/python/python.exe tools/gen_architecture_doc.py` 刷新。\n")
L.append("---\n")

for layer in LAYER_ORDER:
    items = by_layer.get(layer)
    if not items:
        continue
    n_l = sum(x["lines"] for x in items)
    L.append(f"\n## {layer}　—　{LAYER_DESC.get(layer,'')}\n")
    L.append(f"*{len(items)} 个文件 / {n_l:,} 行*\n")
    for it in sorted(items, key=lambda x: -x["lines"]):
        L.append(f"\n### `{it['rel']}`　（{it['lines']} 行）\n")
        L.append(f"**职责**：{it['duty'] or '—（无模块 docstring，建议补一句话职责）'}\n")
        if it["classes"]:
            L.append("\n**类**：\n")
            for cname, cline, cdoc, methods in it["classes"]:
                L.append(f"- `class {cname}`（L{cline}）：{cdoc or '—'}")
                for msig, mline, mdoc in methods:
                    L.append(f"    - `{msig}`（L{mline}）{('— ' + mdoc) if mdoc else ''}")
        if it["funcs"]:
            L.append("\n**顶层函数**：\n")
            for fsig, fline, fdoc in it["funcs"]:
                L.append(f"- `{fsig}`（L{fline}）{('— ' + fdoc) if fdoc else ''}")
        L.append("")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ★ 自动拼接「架构总览」（人工撰写，独立源文件，重跑不丢）
#   2026-10-02 修正：此前总览是脚本外手工拼的，一重跑就被覆盖 ⇒ 改为从源文件读取。
OVERVIEW_SRC = os.path.join(ROOT, "docs", "架构总览.src.md")
head = ""
if os.path.isfile(OVERVIEW_SRC):
    head = open(OVERVIEW_SRC, encoding="utf-8").read().rstrip() + "\n\n---\n---\n\n"
    print(f"  已拼接架构总览: docs/架构总览.src.md（{len(head.splitlines())} 行）")
else:
    print("  !! 未找到 docs/架构总览.src.md —— 输出将不含人工总览部分")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(head + "\n".join(L) + "\n")

# ★ 保留策略：只保留最新一份（用户 2026-10-02 要求）
#   先确认新文件确实写成功（存在且非空），再删旧的 —— 否则会出现「旧的删了新的没生成」空窗。
if os.path.isfile(OUT) and os.path.getsize(OUT) > 0:
    import glob as _glob
    keep = os.path.basename(OUT)
    for _pat in ("代码架构_*.md", "代码架构_*.docx", "代码架构_*.pdf"):
        for _p in _glob.glob(os.path.join(os.path.dirname(OUT), _pat)):
            if os.path.basename(_p) != keep:
                try:
                    os.remove(_p)
                    print(f"  已删除旧版: {os.path.basename(_p)}")
                except Exception as _e:  # noqa: BLE001
                    print(f"  !! 旧版删除失败（可忽略）: {os.path.basename(_p)} {type(_e).__name__}")
else:
    print("  !! 新文件写入异常，跳过旧版清理")

print(f"已生成: {OUT}")
print(f"  文件 {tot_f} / 行 {tot_lines:,} / 类 {tot_cls} / 函数与方法 {tot_fn}")
print(f"  产出大小 {os.path.getsize(OUT):,} bytes")
for layer in LAYER_ORDER:
    if by_layer.get(layer):
        print(f"    {layer}: {len(by_layer[layer])} 文件")
