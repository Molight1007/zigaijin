# -*- coding: utf-8 -*-
"""剪枝扫描 —— 找出「用不到的」与「已经不起效的」单元（只读）。

用户口径（2026-10-25）：
    「剪枝我觉得是要将用不到的或者已经不起效的都减掉，因为我们开始没有做，
      所以现在要做的就要全方面的做」

所以本工具不只扫"零正贡献"，还要扫**从没接过人的**：死模块、死配置、死提示词、
未接线资产、未导出工具。

⚠ 与《智能体要求》R14 的关系
==========================
R14：剪枝用**关闭**不用删除（删掉会同时失去回滚与对照）。
本工具因此只出**两档清单**，都不自动执行：
    A 档 = 建议物理删除（无消费者、无动态导入迹象）—— 需单独授权
    B 档 = 建议默认关闭（有代码，但入口不成立或从未接线）—— 可回滚

用法
====
    python tools/prune_scan.py                 # 全部清单
    python tools/prune_scan.py --only modules  # 只看某一类
    python tools/prune_scan.py --json out.json

⚠ 只读：不改文件、不改配置、不关开关。
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

SCAN_DIRS = ("agent", "utils", "tools", "cli", "deploy", "contracts", "prompts")
EXCLUDE_DIRS = ("__pycache__", "archive", "tests", "backups", "node_modules")
ENTRY_FILES = {"run_eval.py", "user_agent.py", "run_research.py", "run_eval__engine.py",
               "run_eval__scoring.py", "run_eval__classify.py"}


def _read(p):
    try:
        return io.open(p, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def load_corpus() -> dict:
    corpus = {}
    for sub in SCAN_DIRS:
        d = os.path.join(_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for dp, _dirs, files in os.walk(d):
            if any(x in dp for x in EXCLUDE_DIRS):
                continue
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    corpus[p] = _read(p)
    # 根目录入口
    for f in os.listdir(_ROOT):
        if f.endswith(".py"):
            p = os.path.join(_ROOT, f)
            corpus[p] = _read(p)
    return corpus


# ------------------------------------------------------------------ 1 死模块
_IMPORT_PATTERNS = (
    r"from\s+(?:\.{{0,2}}|agent\.){mod}\s+import",
    r"from\s+agent\s+import[^\n]*\b{mod}\b",
    r"import\s+(?:agent\.)?{mod}\b",
)


def scan_dead_modules(corpus: dict) -> list:
    out = []
    agent_dir = os.path.join(_ROOT, "agent")
    if not os.path.isdir(agent_dir):
        return out
    names = {}
    for f in os.listdir(agent_dir):
        if f.endswith(".py") and f != "__init__.py":
            names[f[:-3]] = os.path.join(agent_dir, f)
    for mod, path in sorted(names.items()):
        pats = [p.format(mod=re.escape(mod)) for p in _IMPORT_PATTERNS]
        hits = []
        for p, text in corpus.items():
            if os.path.abspath(p) == os.path.abspath(path):
                continue
            if any(re.search(pt, text) for pt in pats):
                hits.append(os.path.relpath(p, _ROOT).replace("\\", "/"))
        if not hits:
            src = corpus.get(path, "")
            dynamic = bool(re.search(r"importlib|__import__|spec_from_file_location", src))
            standalone = "__main__" in src
            if standalone:
                # 独立命令行工具：按设计不被 import，不算死模块
                out.append({
                    "kind": "standalone_script", "unit": "agent/%s.py" % mod,
                    "lines": len(src.splitlines()), "consumers": 0, "grade": "C",
                    "note": "带 __main__ 入口，按设计不被 import（人用 CLI）",
                })
                continue
            out.append({
                "kind": "dead_module",
                "unit": "agent/%s.py" % mod,
                "lines": len(src.splitlines()),
                "consumers": 0,
                "dynamic_hint": dynamic,
                "grade": "B" if dynamic else "A",
                "note": "存在动态导入迹象，需人工确认" if dynamic else "全仓无任何 import 引用",
            })
    return out


# ------------------------------------------------------------------ 2 死配置
def _config_fields() -> list:
    fields = []
    for fn in ("user_agent__config.py", "user_agent__config_base.py"):
        path = os.path.join(_ROOT, fn)
        src = _read(path)
        if not src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append((stmt.target.id, fn))
    return fields


def _consumed_names(corpus: dict) -> dict:
    """全仓「作为位置实参传给函数」的字符串常量 -> 首次出现的文件。

    为什么不用正则：项目里有 _g("x") / _swb("x") / _cfg_get(cfg,"x") / getattr(cfg,"x")
    等多种自定义读取器，枚举不完（实测 3 轮仍漏）。而"字符串作为函数实参"这一形态
    能**一次覆盖全部读取器**，同时天然排除白名单列表（List/Tuple 元素不是 Call 实参）
    与 CLI 名（以 -- 开头）。
    """
    out = {}
    for p, text in corpus.items():
        if os.path.basename(p).startswith("user_agent__config"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rel = os.path.relpath(p, _ROOT).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        v = a.value
                        if v.startswith("--") or not re.match(r"^[a-z][a-z0-9_]{3,}$", v):
                            continue
                        out.setdefault(v, rel)
            elif isinstance(node, ast.Attribute):
                # 属性访问 config.X / cfg.X / self.config.X —— 也是读取
                # 保守起见任何 .attr 都算（误判成"活着"比"死了"安全，见 R14）
                out.setdefault(node.attr, rel)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                # SWITCH_FIELD = "artifact_store_enabled" + getattr(cfg, SWITCH_FIELD)
                # —— 字段名先存进变量再间接读取（实测踩到，必须算消费）
                v = node.value.value
                if re.match(r"^[a-z][a-z0-9_]{3,}$", v):
                    out.setdefault(v, rel)
    return out


def scan_dead_config(corpus: dict) -> list:
    """字段名从未作为任何函数调用的实参出现 => 无人读取（有入口也只说明能设）。"""
    consumed = _consumed_names(corpus)
    out, seen = [], set()
    for name, fn in _config_fields():
        if name in seen:
            continue
        seen.add(name)
        if name in consumed:
            continue
        # 有入口（CLI / 白名单字符串）但无读取
        has_entry = any(re.search(r"[\"']%s[\"']" % re.escape(name), t)
                        for p, t in corpus.items()
                        if not os.path.basename(p).startswith("user_agent__config"))
        out.append({"kind": "dead_config", "unit": name, "source": fn, "consumers": 0,
                    "grade": "B" if has_entry else "A",
                    "entry_only": has_entry,
                    "note": "有入口无读取（可设但无人消费）" if has_entry else "既无入口也无读取"})
    return out


# ------------------------------------------------------------------ 3 死提示词
def scan_dead_prompts(corpus: dict) -> list:
    out = []
    pdir = os.path.join(_ROOT, "prompts")
    if not os.path.isdir(pdir):
        return out
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        src = _read(os.path.join(pdir, f))
        for m in re.finditer(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*(?:f?r?[\"'])", src, re.M):
            const = m.group(1)
            hits = []
            own = os.path.abspath(os.path.join(pdir, f))
            for p, text in corpus.items():
                if os.path.abspath(p) == own:
                    # 同文件：出现次数 > 1 才算被引用（1 次是定义本身）
                    if len(re.findall(r"\b%s\b" % re.escape(const), text)) > 1:
                        hits.append(os.path.relpath(p, _ROOT).replace("\\", "/"))
                    continue
                if re.search(r"\b%s\b" % re.escape(const), text):
                    hits.append(os.path.relpath(p, _ROOT).replace("\\", "/"))
            if not hits:
                out.append({"kind": "dead_prompt", "unit": "prompts/%s.py::%s" % (f[:-3], const),
                            "consumers": 0, "grade": "B",
                            "note": "无 import、无同文件引用；可能是纯文档性常量"})
    return out


# ------------------------------------------------------------------ 4 未接线资产
def scan_unwired(corpus: dict) -> list:
    out = []
    kb = os.path.join(_ROOT, "docs", "经验库.jsonl")
    if os.path.isfile(kb):
        refs = [p for p, t in corpus.items() if "经验库" in t]
        n = sum(1 for line in io.open(kb, encoding="utf-8", errors="ignore") if line.strip())
        out.append({"kind": "unwired_asset", "unit": "docs/经验库.jsonl",
                    "entries": n, "consumers": len(refs), "grade": "B",
                    "note": "结构化 %d 条，代码仅 %d 处提及" % (n, len(refs))})
    # agent_toolkit 未导出
    tk = os.path.join(_ROOT, "tools", "agent_toolkit")
    if os.path.isdir(tk):
        init = _read(os.path.join(tk, "__init__.py"))
        exported = set(re.findall(r'"([a-z_]+)"', init.split("__all__")[-1])) if "__all__" in init else set()
        for f in sorted(os.listdir(tk)):
            if f.endswith(".py") and f != "__init__.py":
                mod = f[:-3]
                if mod not in exported:
                    out.append({"kind": "unexported_tool", "unit": "tools/agent_toolkit/%s.py" % mod,
                                "consumers": 0, "grade": "B",
                                "note": "不在 __all__，智能体无法调用"})
    return out


# ------------------------------------------------------------------ 输出
KIND_NAMES = {
    "dead_module": "死模块（无人 import）",
    "dead_config": "死配置（零引用）",
    "dead_prompt": "死提示词（零引用）",
    "unwired_asset": "未接线资产",
    "unexported_tool": "未导出工具",
    "standalone_script": "独立脚本（非剪枝对象）",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="剪枝扫描（只读）")
    ap.add_argument("--only", choices=list(KIND_NAMES), help="只看某一类")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)

    corpus = load_corpus()
    items = []
    items += scan_dead_modules(corpus)
    items += scan_dead_config(corpus)
    items += scan_dead_prompts(corpus)
    items += scan_unwired(corpus)
    if args.only:
        items = [x for x in items if x["kind"] == args.only]

    print("# 剪枝扫描（只读，未改动任何文件）\n")
    print("扫描范围：%d 个 py 文件（%s）\n" % (len(corpus), " / ".join(SCAN_DIRS)))
    by_kind = {}
    for x in items:
        by_kind.setdefault(x["kind"], []).append(x)
    print("| 类别 | 条数 | A 档（建议删除） | B 档（建议关闭/确认） |")
    print("|---|---:|---:|---:|")
    for k, label in KIND_NAMES.items():
        sub = by_kind.get(k, [])
        a = sum(1 for x in sub if x["grade"] == "A")
        print("| %s | %d | %d | %d |" % (label, len(sub), a, len(sub) - a))
    print("| **合计** | **%d** | **%d** | **%d** |"
          % (len(items), sum(1 for x in items if x["grade"] == "A"),
             sum(1 for x in items if x["grade"] == "B")))
    print("\n> A 档=无消费者且无动态导入迹象（物理删除需单独授权）；B 档=建议默认关闭或人工确认。")
    print("> 依据 R14：剪枝用关闭不用删除。\n")

    for k, label in KIND_NAMES.items():
        sub = by_kind.get(k, [])
        if not sub:
            continue
        print("## %s（%d）\n" % (label, len(sub)))
        print("| 档 | 单元 | 消费者 | 说明 |")
        print("|---|---|---:|---|")
        for x in sub[:40]:
            extra = ""
            if x["kind"] == "dead_module":
                extra = "%d 行" % x["lines"]
            if x["kind"] == "unwired_asset":
                extra = "%d 条" % x.get("entries", 0)
            print("| %s | `%s` | %s | %s %s |" % (
                x["grade"], x["unit"], x["consumers"], x["note"], extra))
        if len(sub) > 40:
            print("| … | 另有 %d 条 | | |" % (len(sub) - 40))
        print()

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"corpus_files": len(corpus), "items": items}, fh,
                      ensure_ascii=False, indent=1)
        print("[json] %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
