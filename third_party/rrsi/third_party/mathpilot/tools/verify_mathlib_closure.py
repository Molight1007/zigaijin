#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_mathlib_closure.py — 闭包完整性验收（防「残缺闭包」复发）

背景（2026-09-21 事故复盘）
    云端 `deploy/mathlib-olean` 曾只有 920 个 olean、`Mathlib/Data/Real` **0 个**，
    导致任何含 ℝ 的 Lean 代码在云端**必然编译失败**（数值题占 57%）。
    成因三重叠：
      ① 旧测量脚本用 `lake env lean --deps`（自身注释承认会漏 2000+ 模块）低估闭包；
      ② 验收探针 `PROBE_SOURCE` **6 条用例全是 ℕ/ℚ，一条 ℝ 都没有** ⇒ "全过"；
      ③ 决策只看体积不看能力，把"完整"理解成 `Mathlib.Tactic` 闭包 ——
         实测它**仍缺 10/14 常见定理模块**，根本不是完整定理库。
    ⇒ 本脚本把验收标准从「体积达标」改为「**能力达标**」。

设计要点（2026-09-21 二次修正）
    · 验收主力是 **import 探针**而非文件路径清单 —— 文件路径随 Mathlib 版本漂移
      （v4.31 把 `Data/Nat/Factorial` 挪到了 `Data/Nat/Prime/Factorial`，
       `Data/Nat/Prime.lean` 变成了目录）会制造**假警报**。import 是稳定接口。
    · 文件路径检查只保留跨版本稳定的少数项。
    · **依赖包必须一起带**：`Mathlib/Tactic.olean` 依赖 `Batteries`；只打主 lib 会报
      `unknown module prefix 'Batteries'`（2026-09-21 实测踩到）。故单列依赖包检查项。

用法
    python tools/verify_mathlib_closure.py --closure <闭包根>
    python tools/verify_mathlib_closure.py --closure deploy/mathlib-olean \
        --lean deploy/lean-4.31.0-linux/bin/lean --skip-compile

退出码：0 = 全部通过；1 = 有检查未过；2 = 用法错误
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

# ---------------------------------------------------------------- 文件级检查
# 只放**跨版本稳定**的路径（v4.31 实测存在）。易漂移的一律交给 import 探针。
REQUIRED_MODULES = [
    # 实数域（2026-09-21 事故核心：这批曾全部缺失）
    "Mathlib/Data/Real/Basic",
    "Mathlib/Data/Real/Sqrt",
    "Mathlib/Data/Real/Archimedean",
    "Mathlib/Analysis/SpecialFunctions/Sqrt",
    "Mathlib/NumberTheory/Real/Irrational",
    # 代数 / 分析 / 组合 / 数论
    "Mathlib/Algebra/Polynomial/Basic",
    "Mathlib/Analysis/Calculus/Deriv/Basic",
    "Mathlib/Combinatorics/SimpleGraph/Basic",
    "Mathlib/NumberTheory/Modular",
    # tactic 核心
    "Mathlib/Tactic/NormNum",
    "Mathlib/Tactic/Ring",
    "Mathlib/Tactic/Linarith",
    "Mathlib/Tactic/Positivity",
    # 聚合入口
    "Mathlib/Tactic",
]

# ★ 依赖包（lake packages）：缺任一都会让 `import Mathlib.Tactic` 直接断链
REQUIRED_PACKAGES = [
    "Batteries",
    "Aesop",
    "Qq",
]

# ---------------------------------------------------------------- 编译探针
# 探针 1：★ 四域 + 各域模块 import —— 这是本次事故的核心防线
PROBE_DOMAIN_IMPORTS = """import Mathlib.Tactic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.NumberTheory.Real.Irrational
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.NumberTheory.Modular

example : (1:\u2115) + 1 = 2 := by norm_num
example : (1:\u2124) + 1 = 2 := by norm_num
example (x : \u211a) : x + x = 2*x := by ring
example : (1:\u211d) + 1 = 2 := by norm_num
example (x : \u211d) (h : x > 0) : x^2 > 0 := by positivity
example (p : Polynomial \u211d) : p + 0 = p := by ring
"""

# 探针 2：常见定理「名字」在 `import Mathlib.Tactic` 下是否可达
PROBE_COMMON_THEOREMS = """import Mathlib.Tactic

example (n : \u2115) (h : Nat.Prime n) : n \u2265 2 := by sorry
example (x : \u211d) : Real.sqrt (x^2) = |x| := by sorry
example : Nat.factorial 5 = 120 := by norm_num
example (m n k : \u2115) : m * (n + k) = m*n + m*k := by ring
example (s : Finset \u2115) : s.sum id = s.sum id := by rfl
"""


def check_modules(closure: str) -> tuple[int, int, list[str]]:
    """文件级核对必需模块。返回 (命中数, 总数, 缺失清单)。"""
    miss = []
    for mod in REQUIRED_MODULES:
        p = os.path.join(closure, mod.replace("/", os.sep) + ".olean")
        if not os.path.isfile(p):
            miss.append(mod)
    return len(REQUIRED_MODULES) - len(miss), len(REQUIRED_MODULES), miss


def find_lean(explicit: str) -> str:
    """定位 lean 可执行文件，**一律返回绝对路径**。

    ⚠️ 必须绝对化：`compile_probe` 会 `cwd=tmpdir` 运行，相对路径在那里失效
    （2026-09-21 实测：传 `--lean deploy/.../lean` 报 FileNotFoundError）。
    """
    if explicit and os.path.isfile(explicit):
        return os.path.abspath(explicit)
    from shutil import which
    for c in ("lean", "lean.exe"):
        p = which(c)
        if p:
            return os.path.abspath(p)
    for c in (r"deploy/lean-4.31.0-linux/bin/lean",
              r"deploy\lean-4.31.0-linux\bin\lean.exe"):
        if os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def compile_probe(lean: str, closure: str, source: str, label: str) -> bool:
    """用 lean + LEAN_PATH 直编一段探针源码（闭包模式，非 lake 工程）。"""
    if not lean:
        print("  [%s] 跳过：未找到 lean 可执行文件（用 --lean 指定）" % label)
        return False
    tmpdir = tempfile.mkdtemp(prefix="closure_probe_")
    path = os.path.join(tmpdir, "probe.lean")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    env = dict(os.environ)
    prev = (env.get("LEAN_PATH", "") or "")
    env["LEAN_PATH"] = os.path.abspath(closure) + os.pathsep + prev
    try:
        r = subprocess.run([lean, path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=900, env=env, cwd=tmpdir)
    except subprocess.TimeoutExpired:
        print("  [%s] 编译超时（>900s）" % label)
        return False
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    ok = r.returncode == 0
    # 「declaration uses sorry」只是 warning，不算失败
    print("  [%s] rc=%d %s" % (label, r.returncode, "通过" if ok else "失败"))
    if not ok:
        for ln in out.splitlines()[:14]:
            print("      " + ln)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Mathlib 闭包完整性验收")
    ap.add_argument("--closure", required=True, help="闭包根目录（含 Mathlib/）")
    ap.add_argument("--lean", default="", help="lean 可执行文件路径")
    ap.add_argument("--skip-compile", action="store_true",
                    help="只做文件级核对，不跑编译探针")
    args = ap.parse_args()

    closure = args.closure
    if not os.path.isdir(closure):
        print("错误：闭包目录不存在：%s" % closure)
        return 2

    print("闭包根 %s" % os.path.abspath(closure))
    n_olean = 0
    for root, _d, files in os.walk(closure):
        n_olean += sum(1 for f in files if f.endswith(".olean"))
    print("olean 总数 %d" % n_olean)

    fails = []

    print("\n== 1) 必需模块核对（%d 项，稳定路径） ==" % len(REQUIRED_MODULES))
    hit, tot, miss = check_modules(closure)
    print("  命中 %d / %d" % (hit, tot))
    for m in miss:
        print("   缺  " + m)
    if miss:
        fails.append("必需模块缺 %d 项" % len(miss))

    print("\n== 2) 依赖包核对（缺则 import Mathlib.Tactic 直接断链） ==")
    for pkg in REQUIRED_PACKAGES:
        p = os.path.join(closure, pkg + ".olean")
        print("  %s %s.olean" % ("有" if os.path.isfile(p) else "缺", pkg))
        if not os.path.isfile(p):
            fails.append("依赖包缺 %s" % pkg)

    print("\n== 3) 聚合入口 ==")
    for h in ("Mathlib/Tactic", "Mathlib"):
        p = os.path.join(closure, h.replace("/", os.sep) + ".olean")
        print("  %s %s.olean" % ("有" if os.path.isfile(p) else "无", h))

    if not args.skip_compile:
        print("\n== 4) 编译探针 A：★ 四域 + 各域模块 import ==")
        lean = find_lean(args.lean)
        if not lean:
            print("  ⚠️ 未找到 lean，用 --lean 指定（如 "
                  "deploy/lean-4.31.0-linux/bin/lean）")
            fails.append("编译探针未执行（缺 lean）")
        else:
            if not compile_probe(lean, closure, PROBE_DOMAIN_IMPORTS,
                                 "域+模块 import"):
                fails.append("域/模块 import 探针失败")
            print("\n== 5) 编译探针 B：常见定理名可达性 ==")
            if not compile_probe(lean, closure, PROBE_COMMON_THEOREMS,
                                 "常见定理"):
                fails.append("常见定理探针失败")

    print("\n== 结论 ==")
    if fails:
        print("未通过：")
        for f in fails:
            print("   - " + f)
        print("⇒ 该闭包**不可部署**（能力不达标）。")
        return 1
    print("全部通过：闭包能力达标，可部署。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
