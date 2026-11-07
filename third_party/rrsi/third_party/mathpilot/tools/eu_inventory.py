# -*- coding: utf-8 -*-
"""可编辑单元（EU）清单扫描器 —— 五维 harness 的可编辑面盘点（只读）。

为什么需要
==========
「变量表」不能只是一张手抄的 Excel：手抄一次就过期，而且没人知道代码里到底
还有哪些面没被覆盖。本工具把 5 个维度的**可编辑面**从源码里扫出来，
让「代码里有 / 表里没有」这件事随时可算。

五维（与 RRSI 的 harness 定义一致）
==================================
  ① prompt   系统和任务提示词      —— prompts/*.py 的模块级文本常量 + 占位符
  ② flow     控制流（何时计划/行动/反思/停止）—— AgentConfig 字段 + SWITCHES
  ③ tool     工具接口及其描述      —— tools/agent_toolkit 的导出函数
  ④ memory   智能体可查阅的记忆与技能文件 —— prompts/error_lessons.py::LESSONS 条目
  ⑤ context  上下文管理（每一步看到什么）—— agent/*.py 里的硬编码切片 [:N]

输出
====
每个 EU 一条记录，字段固定（便于与编辑账本 L_t 对齐）：:

    {"dim","id","form","source","atomic","coupling","observable","extra"}

用法
====
    python tools/eu_inventory.py                # 五维汇总表
    python tools/eu_inventory.py --dim context  # 某一维明细
    python tools/eu_inventory.py --json out.json
    python tools/eu_inventory.py --missing      # 只报"代码里有、表里没有"

⚠ 只读：不 import agent / prompts（避免循环与副作用），全部按源码文件解析。
⚠ 依据：AST 优先，正则兜底；解析不出的条目宁可漏也不要编（漏的会在 --missing 里显形）。
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

DIMS = ("prompt", "flow", "tool", "memory", "context")

# 维度 → 一次原子编辑的默认定义（可被单条覆盖）
ATOMIC = {
    "prompt": "改一句 / 增删 1 个占位符",
    "flow": "换一档 / ±1 / 开关",
    "tool": "增删 1 个工具 / 改 1 段描述",
    "memory": "增删 1 条 / 改 1 条文本",
    "context": "改 1 个截断值 / 换 1 种选择策略",
}

# 老师变量表的 6 个域（用于把 AgentConfig 的 232 个字段归位）
DOMAINS6 = ("提示策略", "工具集", "记忆/上下文", "控制流", "输出与格式", "预算与资源", "未归类")

# 按参数名启发式归到 6 域；只做粗分，细分类须人工确认
_DOMAIN_RULES = (
    ("预算与资源", r"(token|timeout|budget|workers|parallel|rate_limit|max_calls|max_time|cap)"),
    ("记忆/上下文", r"(context|memory|retention|top_k|search|retrieval|truncat|feedback_review|chars)"),
    ("输出与格式", r"(format|numericize|answer_|final_|output)"),
    ("工具集", r"(tool|lean|smt|sympy|calc)"),
    ("控制流", r"(round|retry|times|limit|stop|threshold|window|rework|regen|max_subgoal|improve|vote|gate|depth)"),
    ("提示策略", r"(prompt|instruction|template|lesson|hint|policy|attention|domain)"),
)


def _domain_hint(name: str) -> str:
    for dom, pat in _DOMAIN_RULES:
        if re.search(pat, name, re.I):
            return dom
    return "未归类"


# ---------------------------------------------------------------- 工具函数
def _read(path: str) -> str:
    try:
        return io.open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def _iter_py(root: str):
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _consumers(name: str, corpus: dict) -> list:
    """在 agent/utils/tools/cli/deploy/contracts 里找该名字的出现文件（影响半径证据）。"""
    hit = []
    for path, text in corpus.items():
        if path.endswith("eu_inventory.py"):
            continue
        if re.search(r"\b%s\b" % re.escape(name), text):
            hit.append(os.path.relpath(path, _ROOT).replace("\\", "/"))
    return sorted(hit)


# ---------------------------------------------------------------- ① prompt
_PROMPT_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*(?:f?r?[\"'])", re.M)
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def scan_prompts() -> list:
    items = []
    pdir = os.path.join(_ROOT, "prompts")
    if not os.path.isdir(pdir):
        return items
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        text = _read(os.path.join(pdir, f))
        for m in _PROMPT_RE.finditer(text):
            const = m.group(1)
            # 取该常量到下一个模块级常量之间的文本，用于提占位符
            seg_end = _PROMPT_RE.search(text, m.end() + 1)
            seg = text[m.end(): seg_end.start() if seg_end else len(text)]
            ph = sorted(set(_PLACEHOLDER_RE.findall(seg)))
            items.append({
                "dim": "prompt",
                "id": "prompt.%s.%s" % (f[:-3], const.lower()),
                "form": "template" if ph else "text",
                "source": "prompts/%s::%s" % (f, const),
                "atomic": ATOMIC["prompt"],
                "coupling": "占位符需同步注册填充点" if ph else "",
                "observable": "",
                "extra": {"chars": len(seg), "placeholders": ph,
                          "consumers": _consumers(const, _CORPUS)},
            })
    return items


# ---------------------------------------------------------------- ② flow
_NUM_TYPES = {"bool", "int", "float", "str"}


def _scan_dataclass_fields(path: str) -> list:
    """AST 取 dataclass 类字段（name, type, default 源码）。"""
    out = []
    src = _read(path)
    if not src:
        return out
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                t = ast.unparse(stmt.annotation) if hasattr(ast, "unparse") else ""
                d = ast.unparse(stmt.value) if (stmt.value is not None and hasattr(ast, "unparse")) else ""
                out.append((node.name, stmt.target.id, t, d))
    return out


def scan_flow() -> list:
    items = []
    seen = set()
    for fname in ("user_agent__config.py", "user_agent__config_base.py"):
        path = os.path.join(_ROOT, fname)
        if not os.path.isfile(path):
            continue
        for cls, name, typ, default in _scan_dataclass_fields(path):
            if name in seen:
                continue
            seen.add(name)
            base = typ.replace("Optional[", "").replace("]", "").strip()
            items.append({
                "dim": "flow",
                "id": "flow.%s" % name,
                "form": base or "auto",
                "source": "%s::%s.%s" % (fname, cls, name),
                "atomic": ATOMIC["flow"],
                "coupling": "",
                "observable": "",
                "extra": {"default": default, "class": cls, "hint": _domain_hint(name),
                          "consumers": _consumers(name, _CORPUS)},
            })
    # SWITCHES（env 型开关，单独来源）
    sw_path = os.path.join(_ROOT, "contracts", "switch_registry.py")
    sw_src = _read(sw_path)
    for m in re.finditer(r'^\s{4}"([a-z0-9_]+)":\s*\(\s*\n?\s*"([A-Z0-9_]+)",\s*"(\w+)"',
                         sw_src, re.M):
        name, env, kind = m.group(1), m.group(2), m.group(3)
        if name in seen:
            continue
        seen.add(name)
        items.append({
            "dim": "flow",
            "id": "flow.%s" % name,
            "form": kind,
            "source": "contracts/switch_registry.py::SWITCHES[%s]" % name,
            "atomic": ATOMIC["flow"],
            "coupling": "",
            "observable": "",
            "extra": {"env": env, "consumers": _consumers(name, _CORPUS)},
        })
    return items


# ---------------------------------------------------------------- ③ tool
def scan_tools() -> list:
    items = []
    tk = os.path.join(_ROOT, "tools", "agent_toolkit")
    if not os.path.isdir(tk):
        return items
    init = _read(os.path.join(tk, "__init__.py"))
    exported = re.findall(r'"(tool_[a-z_]+|[a-z_]+)"', init.split("__all__")[-1]) if "__all__" in init else []
    exported = [e for e in exported if not e.endswith(".py")]
    for f in sorted(os.listdir(tk)):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        name = f[:-3]
        items.append({
            "dim": "tool",
            "id": "tool.%s" % name,
            "form": "function",
            "source": "tools/agent_toolkit/%s.py" % name,
            "atomic": ATOMIC["tool"],
            "coupling": "描述与其它工具同处一段 prompt ⇒ 改描述会牵连全部" if name in exported else "",
            "observable": "",
            "extra": {"exported": name in exported,
                      "consumers": _consumers(name, _CORPUS)},
        })
    return items


# ---------------------------------------------------------------- ④ memory
def scan_memory() -> list:
    items = []
    path = os.path.join(_ROOT, "prompts", "error_lessons.py")
    src = _read(path)
    if not src:
        return items
    for m in re.finditer(r'"id"\s*:\s*"([A-Za-z0-9_\-]+)"', src):
        lid = m.group(1)
        start = m.start()
        seg = src[start: start + 900]
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', seg)
        qt = re.search(r'"qtypes"\s*:\s*\(([^)]*)\)', seg)
        items.append({
            "dim": "memory",
            "id": "mem.lesson.%s" % lid,
            "form": "entry",
            "source": "prompts/error_lessons.py::LESSONS[%s]" % lid,
            "atomic": ATOMIC["memory"],
            "coupling": "",
            "observable": "",
            "extra": {"name": nm.group(1) if nm else "",
                      "qtypes": qt.group(1).replace('"', "").strip() if qt else "",
                      "consumers": _consumers("match_lessons", _CORPUS)},
        })
    # 经验库（结构化，但未接入运行时）
    kb = os.path.join(_ROOT, "docs", "经验库.jsonl")
    if os.path.isfile(kb):
        n = sum(1 for line in io.open(kb, encoding="utf-8", errors="ignore") if line.strip())
        items.append({
            "dim": "memory",
            "id": "mem.kb.experience_library",
            "form": "collection",
            "source": "docs/经验库.jsonl",
            "atomic": "增删 1 条",
            "coupling": "",
            "observable": "",
            "extra": {"entries": n, "wired": False,
                      "note": "已结构化但未接入运行时注入路径"},
        })
    return items


# ---------------------------------------------------------------- ⑤ context
_SLICE_RE = re.compile(r"\[:\s*(\d{2,6})\s*\]")


def scan_context() -> list:
    items = []
    adir = os.path.join(_ROOT, "agent")
    if not os.path.isdir(adir):
        return items
    for path in _iter_py(adir):
        rel = os.path.relpath(path, _ROOT).replace("\\", "/")
        lines = _read(path).splitlines()
        for i, line in enumerate(lines, 1):
            for m in _SLICE_RE.finditer(line):
                n = int(m.group(1))
                items.append({
                    "dim": "context",
                    # 同一行可能有多个切片 ⇒ ID 必须带列位置，否则重复
                    "id": "ctx.%s:%d#c%d" % (os.path.basename(path)[:-3], i, m.start()),
                    "form": "truncation",
                    "source": "%s:%d" % (rel, i),
                    "atomic": ATOMIC["context"],
                    "coupling": "同一函数内的其它切片共享同一输入长度" ,
                    "observable": "",
                    "extra": {"limit": n, "code": line.strip()[:110]},
                })
    return items


# ---------------------------------------------------------------- 汇总
_CORPUS: dict = {}


def _load_corpus() -> None:
    for sub in ("agent", "utils", "tools", "cli", "deploy", "contracts", "prompts"):
        d = os.path.join(_ROOT, sub)
        if os.path.isdir(d):
            for p in _iter_py(d):
                _CORPUS[p] = _read(p)


def summarize(items: list) -> dict:
    out = {}
    for d in DIMS:
        sub = [x for x in items if x["dim"] == d]
        out[d] = len(sub)
    return out


def render_md(items: list) -> str:
    s = summarize(items)
    lines = ["# 可编辑单元盘点（eu_inventory）", "",
             "| 维 | EU 数 | 一次原子编辑 |", "|---|---:|---|"]
    for d in DIMS:
        lines.append("| %s | %d | %s |" % (d, s[d], ATOMIC[d]))
    lines.append("| **合计** | **%d** | |" % len(items))
    # AgentConfig 按老师 6 域归类（启发式）
    from collections import Counter
    c = Counter(x["extra"].get("hint", "") for x in items if x["dim"] == "flow")
    lines += ["", "## flow 维按老师 6 域归类（启发式，须人工复核）", "",
              "| 域 | 字段数 |", "|---|---:|"]
    for dom in DOMAINS6:
        if c.get(dom):
            lines.append("| %s | %d |" % (dom, c[dom]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="可编辑单元（EU）清单扫描器")
    ap.add_argument("--dim", choices=DIMS, help="只输出某一维明细")
    ap.add_argument("--json", metavar="PATH", help="导出 JSON")
    ap.add_argument("--missing", action="store_true", help="只报没有 consumers 的条目（疑似游离/未接线）")
    args = ap.parse_args(argv)

    _load_corpus()
    items = []
    for fn in (scan_prompts, scan_flow, scan_tools, scan_memory, scan_context):
        try:
            items.extend(fn())
        except Exception as exc:  # noqa: BLE001  单维失败不影响其它维
            print("[warn] %s 扫描失败: %s" % (fn.__name__, exc), file=sys.stderr)

    if args.dim:
        sub = [x for x in items if x["dim"] == args.dim]
        print("## %s （%d 条）" % (args.dim, len(sub)))
        for x in sub:
            ex = x["extra"]
            tail = ex.get("placeholders") or ex.get("default") or ex.get("limit") or ex.get("name") or ""
            print("- `%s`  [%s]  %s" % (x["id"], x["form"], tail))
    elif args.missing:
        lonely = [x for x in items
                  if x["dim"] in ("prompt", "flow", "tool", "memory")
                  and not x["extra"].get("consumers")]
        print("## 无 consumers 的条目（%d / %d）" % (len(lonely), len(items)))
        for x in lonely:
            print("- `%s`  %s" % (x["id"], x["source"]))
        print("\n> 本清单只按「名字是否出现」判定，偏粗。"
              "更严格的剪枝清单（区分「有入口无读取」）见 `python tools/prune_scan.py`。")
    else:
        print(render_md(items))

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summarize(items), "items": items}, fh,
                      ensure_ascii=False, indent=1)
        print("\n[json] %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
