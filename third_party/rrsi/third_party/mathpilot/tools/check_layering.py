#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分层门禁（AST 实测，非文本匹配）—— codegen-workflow §2.4 的可观测落地。

目标分层（自下而上，单向）::

    contracts/  ←  utils/  ←  tools/  ←  agent/  ←  cli/  ←  root(run_eval.py / user_agent.py)

本脚本的判据（全部基于 `ast`，不靠文本匹配 —— skill §5#9）：

1. **contracts/** 不得 import 任何一方包（agent/tools/utils/cli）—— 纯零依赖层。
2. **utils/**  不得 import agent / tools。
3. **tools/**  不得 import agent —— **除显式豁免清单**（每条附理由）。
   豁免 = 本仓库已被证实"不可归零"的债务（真继承 BaseAgent / 集成测试），
   §8 明确"不硬拆、如实登记"。门禁的作用是**防新增**，不是自欺欺人地清零。
4. **contracts/ utils/ tools/ agent/** 均不得 import **cli/**（cli 是最上层入口，
   可 import 任何下层，反之不然）。
5. **无包级环**（在 {contracts, utils, tools, agent, cli} 五层之间）。

附：两段**报告**（不参与退出码，供交叉核对）
- 模块级 `os.environ` 直读计数（registered-switch 纪律的健康度）
- 顶层 `class`/`def` 重名组数（§8「重复实现」的输入）

退出码：0 = 通过；1 = 有硬违规。

可测性：所有检查逻辑都接受 `root` 参数（默认仓库根），`main(root)` 可对任意
目录树运行 —— 供 `tests/test_layering_gate.py` 用 tmp_path 合成用例验证判别力。
不带参数调用 `main()` 的行为与本文件早期版本**逐字一致**（仅新增第 4 条硬检查）。
"""
from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 一方包名（本仓库自有的五层 + 顶层入口模块）
# ★ 2026-10-07：加入 `cli/`（消费 agent/root 的上层 CLI 入口，由 tools/ 归位而来）。
#   它是**最上层**，可以 import 任何一方包；反过来，下层四包**不得** import cli。
FIRST_PARTY = {"contracts", "utils", "tools", "agent", "cli"}

# cli/ 是最上层 ⇒ 这四包不得 import 它（防新增：当前仓库 0 处）。
CLI_FORBIDDEN_FROM = ("contracts", "utils", "tools", "agent")

# ---------------------------------------------------------------------------
# tools/ → agent 的**显式豁免清单**（每条必须给出可核对理由）。
# 新增违规不在清单里 ⇒ 门禁失败。移除了某个文件 ⇒ 从清单删掉（保持收紧）。
# ---------------------------------------------------------------------------
TOOLS_AGENT_ALLOW = {
    # —— tools/lean_local/ 生产模块：真继承 BaseAgent，无法在本轮解耦 ——
    # 方案 §8：「BaseAgent 若被 tools/ 真继承 ⇒ 耦合过深 ⇒ 写明不可拆，不硬来」
    "tools/lean_local/lean_pre_verifier.py":
        "真继承 BaseAgent（LeanPreVerifier(BaseAgent)）+ 用继承来的 self.llm/record/config",
    "tools/lean_local/lean_refiner.py":
        "真继承 BaseAgent + 依赖 agent.blueprint_planner/lemma_memory 业务类",
    "tools/lean_local/lean_translator.py":
        "真继承 BaseAgent + 依赖 agent.blueprint_planner 业务类",
    "tools/lean_local/lean_gate.py":
        "运行期静态调用 BaseAgent.add_used_theorems / note_compile_valid（功能性依赖；"
        "开关已改走 contracts.switch_registry）",
    # —— tools/lean_local/tests/：集成测试本就跨层（方案 §2.5 显式豁免）——
    "tools/lean_local/tests/test_lean_bridge.py": "集成测试（跨层验证 LeanBridge）",
    "tools/lean_local/tests/test_lean_gate.py": "集成测试",
    "tools/lean_local/tests/test_lean_pre_verify.py": "集成测试",
    "tools/lean_local/tests/test_lean_refiner.py": "集成测试",
    "tools/lean_local/tests/test_lean_translator.py": "集成测试",
    "tools/lean_local/tests/test_sketch_audit.py": "集成测试",
    # —— tools/stage_probe.py：分阶段探针（调试工具，非生产路径）——
    # 它必须构造 TaskContext（agent.base）才能从 run 目录重建真实现场、只跑单阶段；
    # 这是"离线复现"的唯一办法——否则排查任一环节都要跑完整 112 题（实测单题 5h40m）。
    # 它不被主链 import，只由人从命令行调用，故不构成运行时耦合。
    "tools/stage_probe.py":
        "调试工具：需 TaskContext 重建现场以单跑某阶段；不被主链 import",
}


def _iter_py(rel_dir: str, root: str = ROOT):
    base = os.path.join(root, rel_dir)
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def _rel(path: str, root: str = ROOT) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def _resolve_relative(path: str, root: str, level: int, module: str | None) -> str | None:
    """把**相对 import** 解析成绝对一方包路径的首段（缺陷 1 修复）。

    语义（`level==1` 停在本包，`level==2` 上溯一层 …）::

        utils/a.py  level=2  module="agent.x"  →  "agent"
        agent/b.py  level=1  module="x"        →  "agent"（同包；调用方 `-{own}` 后为空）
        agent/b.py  level=1  module="x.y"      →  "agent"

    返回解析到的**首段包名**；若越界、结果为空、或首段不在 `FIRST_PARTY`
    ⇒ 返回 `None`（调用方退回 `own`，保守，避免造出幻影边）。
    """
    parts = _rel(path, root).split("/")[:-1]  # 文件所在包路径（去掉文件名）
    idx = len(parts) - (level - 1)
    if idx < 0:  # 相对层级越过仓库根 ⇒ 无法解析
        return None
    full = parts[:idx] + (module.split(".") if module else [])
    if not full:
        return None
    seg = full[0]
    return seg if seg in FIRST_PARTY else None


def _imported_roots(path: str, root: str = ROOT) -> set:
    """该文件引用到的**其它**一方一级包名（含函数内 import）。

    相对 import 按**绝对包路径**解析（缺陷 1：修复前一律记为 own、随后被减掉，
    导致跨层违规与包内环全部隐形）。同包自引用（如 `contracts/protocols.py`
    里的 `from contracts.types import ...`）不计入 —— 它不是跨层依赖。
    """
    own = _rel(path, root).split("/")[0]
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level and n.level > 0:
                seg = _resolve_relative(path, root, n.level, n.module)
                out.add(seg if seg else own)
            elif n.module:
                out.add(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[0])
    return (out & FIRST_PARTY) - {own}


def _module_level_env_reads(path: str) -> int:
    """统计**模块级**（函数/类体之外）的 os.environ / os.getenv 直读次数。

    ⚠ 不进入 FunctionDef / ClassDef 体内 —— 那些属"调用时读"，不是"import 期读"。
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    n = 0

    def scan(node):
        nonlocal n
        for sub in ast.iter_child_nodes(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # 不进函数/类体
            if isinstance(sub, ast.Attribute) and sub.attr in ("environ", "getenv"):
                n += 1
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr == "getenv":
                n += 1
            scan(sub)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            n += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "getenv":
            n += 1
        scan(node)
    return n


def _top_level_defs(root: str = ROOT):
    """跨文件统计顶层 class/def 重名组数（诊断用）。"""
    seen = defaultdict(list)
    for rel in ("agent", "tools", "utils", "contracts"):
        for p in _iter_py(rel, root):
            try:
                tree = ast.parse(open(p, encoding="utf-8").read(), p)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    seen[node.name].append(_rel(p, root))
    return {k: v for k, v in seen.items() if len(v) > 1}


def main(root: str | None = None) -> int:
    root = root or ROOT
    fails = []

    # 1) contracts/ 零依赖
    c_bad = []
    for p in _iter_py("contracts", root):
        hits = _imported_roots(p, root)
        if hits:
            c_bad.append((_rel(p, root), sorted(hits)))
    if c_bad:
        fails.append(("contracts 不得 import 一方包", c_bad))

    # 2) utils/ 不得 import agent|tools
    u_bad = []
    for p in _iter_py("utils", root):
        hits = _imported_roots(p, root) & {"agent", "tools"}
        if hits:
            u_bad.append((_rel(p, root), sorted(hits)))
    if u_bad:
        fails.append(("utils 不得 import agent/tools", u_bad))

    # 3) tools/ 不得 import agent（除豁免）
    t_bad = []
    for p in _iter_py("tools", root):
        if "agent" in _imported_roots(p, root):
            r = _rel(p, root)
            if r not in TOOLS_AGENT_ALLOW:
                t_bad.append(r)
    if t_bad:
        fails.append(("tools 不得 import agent（非豁免）", t_bad))

    # 3b) 下层四包不得 import cli（cli 是最上层入口；防新增）
    cli_bad = []
    for rel in CLI_FORBIDDEN_FROM:
        for p in _iter_py(rel, root):
            if "cli" in _imported_roots(p, root):
                cli_bad.append(_rel(p, root))
    if cli_bad:
        fails.append(("contracts/utils/tools/agent 不得 import cli", cli_bad))

    # 4) 包级环检测（五层：contracts/utils/tools/agent/cli）
    #    · 全量环（含已豁免的 tools→agent）：作为**残余债务**报告；
    #    · 生效环（扣掉豁免清单后的边）：这才是**硬判据** —— 防新增。
    edges_full = defaultdict(set)
    edges_eff = defaultdict(set)
    for rel in ("contracts", "utils", "tools", "agent", "cli"):
        for p in _iter_py(rel, root):
            imp = _imported_roots(p, root)
            edges_full[rel] |= imp
            if rel == "tools" and _rel(p, root) in TOOLS_AGENT_ALLOW:
                edges_eff[rel] |= (imp - {"agent"})  # 豁免的 tools→agent 边不计
            else:
                edges_eff[rel] |= imp
    cycles_full = _find_cycles(edges_full)
    cycles = _find_cycles(edges_eff)
    if cycles:
        fails.append(("包级环（扣豁免后仍有）", cycles))

    # ---- 报告 ----
    print("=" * 72)
    print("分层门禁（AST）")
    print("=" * 72)
    print("目标：contracts/ ← utils/ ← tools/ ← agent/ ← cli/ ← root")
    print()
    print("[硬检查]")
    print("  contracts/ → 一方包        :",
          "PASS" if not c_bad else "FAIL %s" % c_bad)
    print("  utils/ → agent|tools       :",
          "PASS（0 处）" if not u_bad else "FAIL %s" % u_bad)
    exempt_hits = sorted(r for r in _tools_agent_hits(root) if r in TOOLS_AGENT_ALLOW)
    print("  tools/ → agent（非豁免）   :",
          "PASS（0 处）" if not t_bad else "FAIL %s" % t_bad)
    print("  contracts/utils/tools/agent → cli :",
          "PASS（0 处）" if not cli_bad else "FAIL %s" % cli_bad)
    print("  包级环（扣豁免后）         :",
          "PASS（无环）" if not cycles else "FAIL %s" % cycles)
    print("  包级环（全量，含豁免）     :",
          "PASS（无环）" if not cycles_full
          else "残余债务 %s（源自已豁免的 tools→agent）" % cycles_full)
    print()
    print("[残余 tools/ → agent（已豁免，逐条附理由）: %d 处]" % len(exempt_hits))
    for r in exempt_hits:
        print("  · %s\n      → %s" % (r, TOOLS_AGENT_ALLOW[r]))
    print()

    # 报告 A：模块级 env 直读
    env_reads = []
    for rel in ("agent", "tools", "utils", "contracts"):
        for p in _iter_py(rel, root):
            n = _module_level_env_reads(p)
            if n:
                env_reads.append((_rel(p, root), n))
    print("[报告] 模块级 os.environ/os.getenv 直读：%d 个文件" % len(env_reads))
    for r, n in sorted(env_reads):
        print("      %-55s x%d" % (r, n))
    print()

    # 报告 B：顶层重名
    dup = _top_level_defs(root)
    print("[报告] 顶层 class/def 重名组：%d 组" % len(dup))
    print()

    print("=" * 72)
    if fails:
        print("结果：FAIL（%d 类硬违规）" % len(fails))
        for name, detail in fails:
            print("  ✗", name, detail)
        return 1
    print("结果：PASS")
    return 0


def _tools_agent_hits(root: str = ROOT):
    for p in _iter_py("tools", root):
        if "agent" in _imported_roots(p, root):
            yield _rel(p, root)


def _find_cycles(edges):
    """返回各方包之间出现的环（用 DFS）。"""
    cycles = []

    def dfs(node, stack):
        for nb in sorted(edges.get(node, ())):
            if nb not in edges:
                continue
            if nb in stack:
                cyc = stack[stack.index(nb):] + [nb]
                key = " → ".join(cyc)
                if key not in cycles and key[::-1] not in cycles:
                    cycles.append(key)
            else:
                dfs(nb, stack + [nb])

    for n in edges:
        dfs(n, [n])
    return sorted(set(cycles))


if __name__ == "__main__":
    sys.exit(main())
