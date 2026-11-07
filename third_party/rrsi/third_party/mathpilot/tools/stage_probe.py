# -*- coding: utf-8 -*-
"""分阶段探针：从一次真实 run 的现场重建 ctx，**只跑指定阶段**。

为什么需要它
------------
全量跑一道 IMO 题要 5.7 小时（实测 val1_r1_1022），其中 46.6% 花在 4.6 重写。
但排查"4.1 为什么把这条判错"这类问题，只需要 4.1 的输入（候选 + 题面 + 子目标）。
本工具用 run 目录里已落盘的分阶段产物重建现场，把单阶段测试压缩到分钟级。

产物来源（`results/<run_id>/<qid>/`）
--------------------------------------
    01_understanding.json   题型 / 领域 / problem_goal / lean_applicable
    02_blueprint.json       蓝图
    03_subgoals.json        子目标计划
    04_subgoal_results.jsonl 子目标结果（含多轮重写）
    05_candidates.json      候选（id / answer / reasoning）
    07_lean.json            Lean 结果
    08_final.json           最终输出

用法
----
    python tools/stage_probe.py --run-dir <dir> --list
    python tools/stage_probe.py --run-dir <dir> --stage 4.1
    python tools/stage_probe.py --run-dir <dir> --stage 4.1 --cand 0      # 只测第 0 个候选
    python tools/stage_probe.py --run-dir <dir> --stage 4.2 --dry-run     # 只回放，不调模型
    python tools/stage_probe.py --run-dir <dir> --stage 4.1 --no-llm      # 用桩模型

⚠ 会真实调用 LLM（除 --dry-run / --no-llm），产生费用与耗时。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

STAGES = {
    "4.1": "self_audit（AI 自审：逻辑错误淘汰）",
    "4.2": "adversarial_gate（对抗证伪 + 跨模型）",
    "4.3": "lean_check（Lean 编译验证）",
    "4.4": "math_check（数学工具计算检测）",
    "4.5": "vote（多答案判断 + 投票）",
    "merge": "子目标合并（含第二模型复核实检 / 提示词注入）",
}




def _load_prod_env() -> None:
    """按 run_112 的方式加载环境（.env → lean-env.sh → cloud_run.env）。

    ★★ 2026-10-23 教训：**探针必须与生产同环境**，否则测出来的是假象。
      第一次实测 4.3 一直报 `failure_kind=env`，查了半天发现是探针自己
      没加载环境（LEAN_PATH 为空，Lean 找不到 Mathlib）。
      这与「--check-env 第一版误报 lake ✗」是同一个错。
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        _envf = os.path.join(_root, ".env")
        if os.path.isfile(_envf):
            for _ln in open(_envf, encoding="utf-8", errors="ignore"):
                _ln = _ln.strip()
                if "=" in _ln and not _ln.startswith("#"):
                    _k, _v = _ln.split("=", 1)
                    os.environ.setdefault(_k.strip(),
                                          _v.strip().strip('"').strip("'"))
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess as _sp
        for _f in ("deploy/lean-env.sh", "deploy/cloud_run.env"):
            _p = os.path.join(_root, _f)
            if not os.path.isfile(_p):
                continue
            _r = _sp.run(["bash", "-c", 'set -a; . "%s" 2>/dev/null; env' % _p],
                         cwd=_root, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=60)
            for _ln in (_r.stdout or "").splitlines():
                if "=" in _ln:
                    _k = _ln.partition("=")[0]
                    _v = _ln.partition("=")[2]
                    if _k and not _k.startswith("_"):
                        os.environ[_k] = _v
    except Exception:  # noqa: BLE001
        pass


# ================================================================== 环境自检
def check_env() -> int:
    """逐项体检**环境依赖**，每项独立、失败不影响其余，5~30 秒出结论。

    为什么单独做成一个入口（2026-10-23 教训）
    -----------------------------------------
    Lean 通道连着两轮在真实 run 里 `lean_active=false`，而我都是从
    "跑完 4 小时 → 看 diag → 发现没跑"这个方向发现的 —— 太晚、太贵。
    环境类问题根本不需要 ctx，一个 20 秒的体检就够。

    检查项：Lean 可执行 / lake / MCP run_code / Mathlib olean / 主模型 / 第二模型。
    """
    import shutil
    ok_all = True

    def _line(name, good, detail=""):
        nonlocal ok_all
        if not good:
            ok_all = False
        print("  %-14s %s  %s" % (name, "✓" if good else "✗", detail))

    _load_prod_env()
    print("=== 环境自检 ===")

    # ★ 必须**按 run_112 的方式**加载环境，否则体检结果与 run 的真实环境不符
    #   （实测：不 source 时云端 lease/lake 全 ✗，而 run 里其实是好的）。
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        _envf = os.path.join(_root, ".env")
        if os.path.isfile(_envf):
            for _ln in open(_envf, encoding="utf-8", errors="ignore"):
                _ln = _ln.strip()
                if "=" in _ln and not _ln.startswith("#"):
                    _k, _v = _ln.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:  # noqa: BLE001
        pass
    # 用 bash source lean-env.sh + cloud_run.env（与 run_112.py 的 main 一致），
    # 拿回它们设的 PATH / LEAN_PATH / LEAN_PROJECT_PATH。
    try:
        import subprocess as _sp
        for _f in ("deploy/lean-env.sh", "deploy/cloud_run.env"):
            _p = os.path.join(_root, _f)
            if not os.path.isfile(_p):
                continue
            _r = _sp.run(["bash", "-c", 'set -a; . "%s" 2>/dev/null; env' % _p],
                         cwd=_root, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=60)
            for _ln in (_r.stdout or "").splitlines():
                if "=" in _ln:
                    _k, _v = _ln.partition("=")[0], _ln.partition("=")[2]
                    if _k and not _k.startswith("_"):
                        os.environ[_k] = _v
    except Exception:  # noqa: BLE001
        pass

    # 1) lean / lake 是否可解析
    exe = os.environ.get("LEAN_EXECUTABLE", "") or shutil.which("lean") or ""
    _line("lean", bool(exe), exe or "未找到（设 LEAN_EXECUTABLE 或加 PATH）")
    lake = shutil.which("lake") or ""
    if not lake and exe:
        cand = os.path.join(os.path.dirname(exe), "lake")
        lake = cand if os.path.isfile(cand) else ""
    _line("lake", bool(lake), lake or "未找到（MCP 用它启动 LSP，缺则 4.3 全空）")

    # 2) MCP run_code —— 真正决定 4.3/4.4 能不能用
    proj = os.environ.get("LEAN_PROJECT_PATH", "")
    if not proj:
        _line("LEAN_PROJECT_PATH", False, "未设置（MCP 需要合法 Lean 工程根）")
    else:
        try:
            from tools.lean_local.lean_bridge import mcp_run_code
            r = mcp_run_code("#eval (1+1 : Nat)", proj, timeout=180)
            good = bool(r.get("ok")) and bool(r.get("items"))
            _line("MCP run_code", good,
                  "ok=%s items=%d%s" % (r.get("ok"), len(r.get("items") or []),
                                        ("  err=" + str(r.get("error"))[:90])
                                        if not good else ""))
        except Exception as exc:  # noqa: BLE001
            _line("MCP run_code", False, "%s: %s" % (type(exc).__name__, str(exc)[:90]))

    # 3) Mathlib olean 数量（只看 LEAN_PROJECT_PATH，别扫整棵树）
    if proj:
        try:
            cnt = 0
            for _r, _d, files in os.walk(proj):
                cnt += sum(1 for f in files if f.endswith(".olean"))
            _line("Mathlib olean", cnt > 100, "%d 个（%s）" % (cnt, proj[-30:]))
        except Exception:  # noqa: BLE001
            _line("Mathlib olean", False, "统计失败")

    # 4) 主模型 / 第二模型：各发一次极小请求
    for name, groups in (
            ("主模型", (("INTERN_API_KEY", "INTERN_API_BASE"),
                        ("OPENAI_API_KEY", "OPENAI_BASE_URL"))),
            ("第二模型", (("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE"),))):
        hit = next((g for g in groups if all(os.environ.get(k) for k in g)), None)
        _line(name + "凭据", bool(hit),
              " ".join(hit) if hit else "缺 %s" % groups[0][0])

    print()
    print("结论：%s" % ("环境就绪" if ok_all else "有项目未通过（见 ✗ 行）"))
    return 0 if ok_all else 1


# ------------------------------------------------------------------ 现场重建
def load_env() -> dict:
    env = {}
    path = os.path.join(_ROOT, ".env")
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def find_q_dir(run_dir: str) -> str:
    """run_dir 可以是 run 根目录，也可以直接是 <qid> 目录。"""
    run_dir = os.path.abspath(run_dir)
    if os.path.isfile(os.path.join(run_dir, "05_candidates.json")):
        return run_dir
    for name in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, name)
        if os.path.isdir(p) and os.path.isfile(
                os.path.join(p, "05_candidates.json")):
            return p
    raise SystemExit("在 %s 下找不到含 05_candidates.json 的题目目录" % run_dir)


def load_json(qdir: str, name: str, default=None):
    p = os.path.join(qdir, name)
    if not os.path.isfile(p):
        return default
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def load_jsonl(qdir: str, name: str) -> list:
    p = os.path.join(qdir, name)
    if not os.path.isfile(p):
        return []
    out = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def build_ctx(qdir: str, problem: str, u: dict, cands: list, sg_trace: list):
    """构造 TaskContext（只填本工具需要的字段）。"""
    from agent.base import TaskContext
    ctx = TaskContext(problem=problem, metadata={})
    try:
        ctx.blueprint = load_json(qdir, "02_blueprint.json", {}) or {}
        ctx.subgoal_trace = sg_trace
        ctx.candidates = cands
        md = getattr(ctx, "metadata", None) or {}
        md.update({
            "question_type": (u or {}).get("question_type", ""),
            "domain": (u or {}).get("domain", ""),
            "problem_goal": (u or {}).get("problem_goal") or {},
            "lean_applicable": (u or {}).get("lean_applicable", True),
        })
        ctx.metadata = md
        # 让 is_time_critical() 恒为 False：探针不应被超时逻辑干扰
        ctx.deadline = 0
        ctx.start_time = time.time()
        ctx.max_time_per_question = 0
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("构造 TaskContext 失败: %s" % exc)
    return ctx


def build_candidates(raw: list, only=None):
    from agent.base import Candidate
    out = []
    for i, c in enumerate(raw):
        if only is not None and i != only:
            continue
        if not isinstance(c, dict):
            continue
        out.append(Candidate(id=c.get("id", i),
                             answer=c.get("answer", "") or "",
                             reasoning=c.get("reasoning", "") or ""))
    return out


# ------------------------------------------------------------------ 轻量 orch
class ProbeOrch:
    """只提供阶段函数用到的成员：config / client / record / llm。"""

    def __init__(self, use_llm: bool):
        from user_agent__config import AgentConfig
        self.config = AgentConfig()
        self.records: list = []
        self.client = None
        self.n_llm = 0
        if use_llm:
            from utils.llm_client import LLMClient
            env = load_env()
            self.client = LLMClient(
                api_key=env.get("INTERN_API_KEY", ""),
                base_url=env.get("INTERN_API_BASE", ""),
                model=env.get("INTERN_MODEL", ""), timeout=900)

    def record(self, ctx, step, msg, **kw):
        self.records.append((step, str(msg)[:300]))

    def llm(self, ctx, messages, temperature=0.0, max_tokens=4096):
        if self.client is None:
            raise RuntimeError("stub 模式下不应调用 llm")
        self.n_llm += 1
        return self.client.chat(messages=messages, temperature=temperature,
                                max_tokens=max(16384, max_tokens))


def _stub_llm(orch):
    """把 llm 替换成"什么都不判"的桩（只验证管线不验证判断）。"""
    orch.llm = lambda *a, **k: "{}"


# ------------------------------------------------------------------ 各阶段
def run_stage(stage: str, orch: ProbeOrch, ctx):
    if stage == "4.1":
        from agent.self_auditor import self_audit_stage
        self_audit_stage(orch, ctx)
    elif stage == "4.2":
        from agent.adversarial_gate import adversarial_gate_stage
        adversarial_gate_stage(orch, ctx, "deep")
    elif stage == "4.3":
        from agent.lean_backcheck import lean_backcheck_stage
        lean_backcheck_stage(orch, ctx, "deep")
    elif stage == "merge":
        # merge 本体需要完整 Orchestrator（_prepare_merge_prompt 等），
        # 这里退一步：只单测「第二模型复核 + 提示词注入」这段**新增逻辑**，
        # 正好覆盖最容易出问题的地方（自检命中率 / 注入内容）。
        from agent.subgoal_selfcheck import (check_subgoal_chain,
                                            chain_from_trace, summarize)
        _sus = check_subgoal_chain(
            chain_from_trace(getattr(ctx, "subgoal_trace", None) or []),
            str(getattr(ctx, "problem", "") or ""))
        print("\n=== 复核结果 ===")
        print(summarize(_sus) or "(未发现可疑)")
        for _s in _sus:
            print("  - [%s] %s" % (_s.get("id"), str(_s.get("step"))[:140]))
            print("      应为: %s" % str(_s.get("expected"))[:120])
    elif stage == "4.4":
        from agent.math_check_gate import math_check_stage
        math_check_stage(orch, ctx)
    elif stage == "4.5":
        from agent.final_selector import vote_stage
        vote_stage(orch, ctx, "deep")
    else:
        raise SystemExit("未支持的阶段: %s（支持 %s）" % (stage, ", ".join(STAGES)))


def summarize(ctx) -> dict:
    cands = getattr(ctx, "candidates", None) or []
    out = {
        "候选数": len(cands),
        "候选答案": [str(getattr(c, "answer", ""))[:40] for c in cands],
    }
    # ★ 2026-10-23：Lean 阶段必须看得见**每候选的 verdict 与 failure_kind**，
    #   否则"通过 6/6"到底是真通过、还是"非证伪⇒保留"根本分不清。
    lg = getattr(ctx, "lean_gate", None)
    if isinstance(lg, list) and lg:
        out["lean_gate"] = [
            {"id": e.get("id"), "verdict": e.get("verdict"),
             "failure_kind": e.get("failure_kind"),
             "degraded": e.get("degraded")}
            for e in lg[:8] if isinstance(e, dict)]
    return out


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="分阶段探针：只跑一个环节")
    ap.add_argument("--run-dir", default="", help="run 目录或 <qid> 目录")
    ap.add_argument("--stage", default="", help="4.1 / 4.2 / 4.4 / 4.5")
    ap.add_argument("--problem-file", default="", help="题面文本文件（不给则用占位）")
    ap.add_argument("--cand", type=int, default=None, help="只测第 N 个候选")
    ap.add_argument("--dry-run", action="store_true", help="只回放现场，不调模型")
    ap.add_argument("--no-llm", action="store_true", help="用桩模型（验证管线）")
    ap.add_argument("--list", action="store_true", help="列出可用阶段")
    ap.add_argument("--check-env", action="store_true",
                    help="只做环境体检（Lean/lake/MCP/Mathlib/模型凭据），不需要 --run-dir")
    args = ap.parse_args(argv)

    if args.list:
        for k, v in STAGES.items():
            print("  %-8s %s" % (k, v))
        print("  %-8s %s" % ("--check-env", "环境体检（Lean/lake/MCP/Mathlib/凭据）"))
        return 0
    if args.check_env:
        return check_env()
    if not args.run_dir:
        ap.error("必须给 --run-dir")
    if not args.stage:
        ap.error("必须给 --stage（或 --list 查看）")

    _load_prod_env()          # ★ 先与生产同环境，再取现场
    qdir = find_q_dir(args.run_dir)
    u = load_json(qdir, "01_understanding.json", {}) or {}
    raw_c = load_json(qdir, "05_candidates.json", []) or []
    sg = load_jsonl(qdir, "04_subgoal_results.jsonl")

    problem = ""
    if args.problem_file and os.path.isfile(args.problem_file):
        problem = open(args.problem_file, encoding="utf-8").read()

    cands = build_candidates(raw_c, args.cand)
    print("现场: %s" % qdir)
    print("  子目标结果 %d 条 | 候选 %d 个%s"
          % (len(sg), len(cands),
             "（只测第 %d 个）" % args.cand if args.cand is not None else ""))
    print("  题型=%s 领域=%s lean_applicable=%s"
          % (u.get("question_type"), u.get("domain"), u.get("lean_applicable")))

    if args.dry_run:
        print("\n[dry-run] 候选清单：")
        for c in cands:
            print("  id=%s answer=%r" % (c.id, str(c.answer)[:60]))
        return 0

    ctx = build_ctx(qdir, problem, u, cands, sg)
    orch = ProbeOrch(use_llm=not args.no_llm)
    if args.no_llm:
        _stub_llm(orch)

    before = summarize(ctx)
    t0 = time.time()
    try:
        run_stage(args.stage, orch, ctx)
    except Exception:  # noqa: BLE001
        print("\n[阶段异常]")
        traceback.print_exc()
        return 2
    dt = time.time() - t0
    after = summarize(ctx)

    print("\n=== 阶段 %s 执行完毕（%.1fs，LLM 调用 %d 次）==="
          % (args.stage, dt, orch.n_llm))
    print("  候选：%d → %d" % (before["候选数"], after["候选数"]))
    if after["候选答案"]:
        print("  存活答案：")
        for a in after["候选答案"]:
            print("    - %s" % a)
    else:
        print("  存活答案：（无，全部被淘汰）")
    if after.get("lean_gate"):
        print("\n  Lean 逐候选：")
        for _e in after["lean_gate"]:
            print("    id=%-4s verdict=%-14s failure_kind=%-14s degraded=%s"
                  % (_e.get("id"), _e.get("verdict"),
                     _e.get("failure_kind") or "-", _e.get("degraded") or "-"))
    print("\n  阶段留痕（%d 条）：" % len(orch.records))
    for step, msg in orch.records[:40]:
        print("    [%s] %s" % (step, msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
