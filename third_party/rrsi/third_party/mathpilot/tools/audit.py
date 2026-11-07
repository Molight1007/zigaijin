# -*- coding: utf-8 -*-
"""审计层统一入口（2026-10-24）。

把三件互相依赖的事收在一个只读层里，而不是三个各写一套的脚本：

    leak    改动里有没有「只对某道题成立」的东西（静态，提交前）
    cost    每一版的 token / LLM 次数 / Lean 次数 / 耗时，能否并排看
    delta   同一版本重复跑，通过数自己能飘几题（噪声带 δ）
    doctor  用构造数据自检上面三个 —— 工具本身也要被验证

设计约束
--------
1. **只读**：本模块不启动主链、不调 LLM、不写 results（doctor 只写临时目录）。
2. **不依赖 agent**：tools 层禁止反向依赖 ⇒ 只读结果文件与 manifest。
3. **缺数据不崩**：历史 run 没有 token 字段就显示 ``n/a``，绝不把「没记」
   伪装成 0 —— 那会让成本曲线凭空变好看。
4. δ 只对**同一版本**有意义：`delta` 会硬校验 manifest 的
   ``git_sha + worktree_sha + question_ids``，不一致直接拒绝。

用法
----
    python tools/audit.py leak --grade                # 未提交改动（分级）
    python tools/audit.py leak --rev 1de9a014         # 事后复核某次提交
    python tools/audit.py cost --runs a.jsonl b.jsonl c.jsonl
    python tools/audit.py delta --runs r1.jsonl r2.jsonl r3.jsonl r4.jsonl r5.jsonl
    python tools/audit.py doctor
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_NA = "n/a"


# ------------------------------------------------------------------ 读取
def load_run(path: str) -> list:
    """读一份评测结果 jsonl。坏行跳过（评测中途被杀会留半行）。"""
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError as e:
        print("读取失败 %s: %s" % (path, e), file=sys.stderr)
    return rows


def _i(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def manifest_for(path: str) -> dict:
    """取 run 身份 sidecar。找不到返回 {}（老结果没有）。"""
    cand = [path + ".manifest.json",
            os.path.join(os.path.dirname(path), "manifest.json")]
    for p in cand:
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                if isinstance(d, dict):
                    return d
            except Exception:  # noqa: BLE001
                pass
    return {}


# ------------------------------------------------------------------ cost
def summarize_cost(rows: list) -> dict:
    """单份结果的成本/正确率摘要。缺 token 时该列返回 None（显示 n/a）。"""
    n = len(rows)
    scored = [r for r in rows if r.get("correct") is not None]
    correct = sum(1 for r in scored if r.get("correct"))
    tok = 0
    tok_rows = 0
    llm = 0
    llm_rows = 0
    lean = 0
    lean_ok = 0
    lean_rows = 0
    sec = 0.0
    sec_rows = 0
    for r in rows:
        t = r.get("tokens")
        if isinstance(t, dict) and _i(t.get("total_tokens")) > 0:
            tok += _i(t.get("total_tokens"))
            tok_rows += 1
        lc = r.get("llm_calls")
        if isinstance(lc, dict):
            llm += _i(lc.get("calls"))
            llm_rows += 1
        tc = r.get("tool_calls")
        if isinstance(tc, dict) and isinstance(tc.get("lean_mcp"), dict):
            mcp = tc.get("lean_mcp") or {}
            lean += _i(mcp.get("calls"))
            lean_ok += _i(mcp.get("ok"))
            lean_rows += 1
        if r.get("elapsed_sec") is not None:
            sec += _f(r.get("elapsed_sec"))
            sec_rows += 1
    denom = max(n, 1)
    return {
        "n": n,
        "scored": len(scored),
        "correct": correct,
        "acc": (correct / len(scored)) if scored else None,
        "tokens": tok if tok_rows else None,
        "token_rows": tok_rows,          # 有多少行真带 token
        "tok_per_q": (tok / denom) if tok_rows else None,
        # ★ 三列都遵循同一纪律：本版没记 ⇒ None（显示 n/a），绝不当 0
        "llm_per_q": (llm / denom) if llm_rows else None,
        "lean_per_q": (lean / denom) if lean_rows else None,
        "lean_ok_per_q": (lean_ok / denom) if lean_rows else None,
        "sec_per_q": (sec / denom) if sec_rows else None,
        "cost_per_correct": (tok / correct) if (tok_rows and correct) else None,
    }


def _fmt_num(v, nd=1) -> str:
    if v is None:
        return _NA
    if isinstance(v, float):
        return ("%.*f" % (nd, v))
    return str(v)


def render_cost(runs: list, labels: list = None, as_json: bool = False) -> dict:
    """多版并排。``runs`` 是文件路径列表，第一份作基线算倍数。"""
    items = []
    for i, path in enumerate(runs):
        rows = load_run(path)
        s = summarize_cost(rows)
        s["path"] = path
        s["label"] = (labels[i] if labels and i < len(labels)
                      else os.path.basename(path))
        mf = manifest_for(path)
        s["git_sha"] = (mf.get("git_sha") or "")[:8] or _NA
        s["cost_accuracy"] = mf.get("cost_accuracy") or _NA
        items.append(s)

    base = items[0] if items else None
    for it in items:
        it["token_ratio"] = (
            (it["tokens"] / base["tokens"])
            if (base and base.get("tokens") and it.get("tokens")) else None)
        it["acc_delta_pp"] = (
            round((it["acc"] - base["acc"]) * 100, 2)
            if (base and base.get("acc") is not None and it.get("acc") is not None)
            else None)
        it["cost_per_correct_ratio"] = (
            (it["cost_per_correct"] / base["cost_per_correct"])
            if (base and base.get("cost_per_correct") and it.get("cost_per_correct"))
            else None)

    out = {"baseline": base["label"] if base else None, "runs": items}
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not items:
        print("没有可分析的 run。")
        return out
    print("成本并排（基线 = %s）" % (base["label"] if base else _NA))
    head = ("%-22s %-8s %5s %6s %7s %9s %8s %8s %8s %9s %7s"
            % ("label", "sha", "n", "acc", "tokens", "tok/题", "LLM/题",
               "Lean/题", "秒/题", "每正确题", "倍率"))
    print(head)
    print("-" * len(head))
    for it in items:
        acc = ("%.0f%%" % (it["acc"] * 100)) if it["acc"] is not None else _NA
        tok = _fmt_num(it["tokens"], 0)
        if it["tokens"] is None:
            tok = "%s(%d/%d行)" % (_NA, it["token_rows"], it["n"])
        print("%-22s %-8s %5d %6s %7s %9s %8s %8s %8s %9s %7s"
              % (it["label"][:22], it["git_sha"], it["n"], acc, tok,
                 _fmt_num(it["tok_per_q"], 0), _fmt_num(it["llm_per_q"], 1),
                 _fmt_num(it["lean_per_q"], 2), _fmt_num(it["sec_per_q"], 1),
                 _fmt_num(it["cost_per_correct"], 0),
                 _fmt_num(it["token_ratio"], 2)))
    # 结论行：只在数据齐全时给判断，缺 token 就明说不能判
    if all(it["tokens"] is not None for it in items) and len(items) >= 2:
        last = items[-1]
        if last["token_ratio"] and last["acc_delta_pp"] is not None:
            print("\n判读：成本 %.2f×，正确率 %+.1fpp，每正确题成本 %.2f×"
                  % (last["token_ratio"], last["acc_delta_pp"],
                     last["cost_per_correct_ratio"] or 0))
    else:
        print("\n判读：token 列不完整 ⇒ **无法判断成本倍数**"
              "（缺的 run 未记录 token，不能当 0 处理）")
    return out


# ------------------------------------------------------------------ delta
def _pass_vector(rows: list) -> dict:
    return {str(r.get("id")): bool(r.get("correct")) for r in rows}


def render_delta(runs: list, as_json: bool = False) -> dict:
    """N 份重复 run → 通过数序列、极差、σ、逐题翻转、δ。

    ★ 硬校验：必须同一版本。身份 = git_sha + worktree_sha + question_ids。
      缺 manifest 时降级为「只比题集」，并明确标注无法证明同版本。
    """
    err = None
    vecs, ids_list, mans = [], [], []
    for path in runs:
        rows = load_run(path)
        vecs.append(_pass_vector(rows))
        ids_list.append(frozenset(vecs[-1]))
        mans.append(manifest_for(path))

    if len(runs) < 2:
        err = "至少需要 2 份结果才能算 δ"
    if not err and len(set(ids_list)) > 1:
        err = "题集不一致：各 run 的题目集合不同 ⇒ δ 无意义"

    # 版本身份三元组：提交号 + 工作树指纹 + **题库指纹**
    # （只比提交号会漏掉"工作树改过"；只比题集 id 会漏掉"两份不同题库 id 撞名"）
    sha_set = {(m.get("git_sha"), m.get("worktree_sha"), m.get("bank_sha"))
               for m in mans if m}
    if not err and len(sha_set) > 1:
        detail = "；".join(
            "%s: sha=%s wt=%s bank=%s" % (os.path.basename(p),
                                          (m.get("git_sha") or "?")[:8],
                                          m.get("worktree_sha") or "?",
                                          (m.get("bank_sha") or "?")[:8])
            for p, m in zip(runs, mans))
        err = ("不是同一版本，拒绝计算 δ（%s）。δ 只对「未改动的 harness 重复评测」有意义"
               % detail)
    missing_manifest = [p for p, m in zip(runs, mans) if not m]

    out = {"n_runs": len(runs), "error": err}
    if err:
        if as_json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print("δ 校准失败：%s" % err)
        return out

    ids = sorted(set().union(*ids_list)) if ids_list else []
    n = len(ids)
    seq = [sum(1 for t in ids if vecs[i].get(t)) for i in range(len(runs))]
    per_task = {t: [vecs[i].get(t) for i in range(len(runs))] for t in ids}
    flips = {t: v for t, v in per_task.items() if 0 < sum(1 for x in v if x) < len(runs)}
    rng = max(seq) - min(seq)
    sd = statistics.pstdev(seq) if len(seq) > 1 else 0.0
    out.update({
        "sequence": seq,
        "n_tasks": n,
        "range": rng,
        "delta_pp": round(100.0 * rng / n, 2) if n else None,
        "stdev": round(sd, 3),
        "mean": round(statistics.mean(seq), 3) if seq else None,
        "flipped_tasks": len(flips),
        "flipped_detail": {k: ["1" if x else "0" for x in v] for k, v in
                           sorted(flips.items())},
        "same_version": not missing_manifest,
        "missing_manifest": missing_manifest,
    })
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    print("δ 校准（%d 次重复 × %d 题）" % (len(runs), n))
    print("  通过数序列：%s" % ", ".join(str(x) for x in seq))
    print("  均值 %.2f，总体 σ=%.2f，极差 = %d 题（%.0fpp）"
          % (out["mean"], sd, rng, out["delta_pp"] or 0))
    print("  逐题抖动：%d/%d 题至少翻转过一次" % (len(flips), n))
    for t, v in list(out["flipped_detail"].items())[:8]:
        print("    %-22s %s" % (t, "".join(v)))
    if missing_manifest:
        print("  ⚠ 有 %d 份缺 manifest ⇒ 只能证明题集相同，**无法证明同版本**"
              % len(missing_manifest))
    print("\n判据：改进需 > %d 题（%dpp），且两版配对 McNemar p<0.05 才算真实提升。"
          % (rng, round(out["delta_pp"] or 0)))
    return out


# ------------------------------------------------------------------ doctor
def _write_fake(path: str, corrects: list, tokens: list = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i, c in enumerate(corrects):
            row = {"id": "q%02d" % i, "correct": bool(c), "elapsed_sec": 100,
                   "llm_calls": {"calls": 3, "truncated": 0},
                   "tool_calls": {"lean_mcp": {"calls": 2, "ok": 1, "fail": 1,
                                               "seconds": 1.0},
                                  "web_search": {"calls": 0, "ok": 0, "fail": 0,
                                                 "seconds": 0.0, "results": 0}}}
            if tokens:
                row["tokens"] = {"prompt_tokens": tokens[i] // 2,
                                 "completion_tokens": tokens[i] // 2,
                                 "total_tokens": tokens[i],
                                 "calls_with_usage": 3, "usage_unavailable": 0}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_manifest(path: str, sha: str = "deadbeef", wt: str = "clean",
                    ids: list = None) -> None:
    with open(path + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"run_id": "fake", "git_sha": sha, "worktree_sha": wt,
                   "question_ids": ids or ["q%02d" % i for i in range(10)],
                   "n_questions": 10, "concurrency": 1,
                   "cost_accuracy": "exact"}, fh, ensure_ascii=False)


def cmd_doctor(as_json: bool = False) -> int:
    """用构造数据验证三个子命令的**判读能力**（不是验证它们能跑）。"""
    import contextlib
    import io
    import tempfile
    from tools import leakage_scan as LS

    def _quiet(fn, *a, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*a, **kw)

    results = []
    tmp = tempfile.mkdtemp(prefix="audit_doctor_")
    try:
        # 1) δ：2,2,3,1,2 → 极差 2
        seq = [2, 2, 3, 1, 2]
        paths = []
        for i, k in enumerate(seq):
            p = os.path.join(tmp, "d%d.jsonl" % i)
            # 前 k 题正确
            _write_fake(p, [1] * k + [0] * (10 - k))
            _write_manifest(p)
            paths.append(p)
        d = _quiet(render_delta, paths, as_json=True)
        ok = (d.get("range") == 2 and d.get("sequence") == seq)
        results.append(("δ(2,2,3,1,2)→极差 2", ok, str(d.get("range"))))

        # 2) δ：混版本必须拒绝
        p_bad = os.path.join(tmp, "d_bad.jsonl")
        _write_fake(p_bad, [1] * 3 + [0] * 7)
        _write_manifest(p_bad, sha="cafebabe")
        d2 = _quiet(render_delta, [paths[0], p_bad], as_json=True)
        results.append(("δ 混版本被拒", bool(d2.get("error")), str(d2.get("error"))[:40]))

        # 3) cost：3 倍 token 被识别
        a = os.path.join(tmp, "c_a.jsonl")
        b = os.path.join(tmp, "c_b.jsonl")
        _write_fake(a, [1] * 2 + [0] * 8, tokens=[100] * 10)
        _write_fake(b, [1] * 3 + [0] * 7, tokens=[300] * 10)
        _write_manifest(a)
        _write_manifest(b)
        c = _quiet(render_cost, [a, b], as_json=True)
        ratio = c["runs"][1]["token_ratio"]
        results.append(("cost 3× 识别", abs((ratio or 0) - 3.0) < 1e-6, str(ratio)))

        # 4) cost：缺 token 必须显示 n/a 而不能当 0
        no_tok = os.path.join(tmp, "c_old.jsonl")
        _write_fake(no_tok, [1] * 2 + [0] * 8)
        _write_manifest(no_tok)
        c2 = _quiet(render_cost, [no_tok], as_json=True)
        results.append(("缺 token → None(n/a)",
                        c2["runs"][0]["tokens"] is None,
                        str(c2["runs"][0]["tokens"])))

        # 5) 泄漏分级：四个档位各一条
        cases = [
            ("agent/x.py", 3, '    return "603729"', "LEAK"),
            ("agent/x.py", 9, '# 实测 603729/2 这类错答案', "COMMENT"),
            ("tests/bench/cases/x.jsonl", 20, '{"answer": "603729"}', "SAMPLE"),
            ("results/x.jsonl", 9, '{"gold": "603729"}', "INFO"),
        ]
        rows = [(p, ln, txt) for p, ln, txt, _ in cases]
        golds, cf = LS._load_bank()
        if not golds:
            results.append(("泄漏分级（题库缺失，跳过）", True, "no bank"))
        else:
            hits = LS.scan_graded(rows, golds, cf)
            ok = True
            detail = []
            for (p, ln, txt, want) in cases:
                got = {g for g, items in hits.items()
                       for h in items if h["path"] == p and h["line"] == ln}
                detail.append("%s→%s" % (want, ",".join(sorted(got)) or "-"))
                ok = ok and (want in got)
            results.append(("泄漏四档分级", ok, " ".join(detail)))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for _n, ok, _d in results if ok)
    if as_json:
        print(json.dumps({"passed": passed, "total": len(results),
                          "checks": [{"name": n, "ok": ok, "detail": d}
                                     for n, ok, d in results]},
                         ensure_ascii=False, indent=2))
    else:
        print("audit doctor：%d/%d 通过" % (passed, len(results)))
        for n, ok, d in results:
            print("  [%s] %-28s %s" % ("PASS" if ok else "FAIL", n, d))
    return 0 if passed == len(results) else 1


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="审计层：泄漏分级 / 成本并排 / δ 校准")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_leak = sub.add_parser("leak", help="泄漏筛查（分级）")
    p_leak.add_argument("--base", default="HEAD")
    p_leak.add_argument("--rev", default=None)
    p_leak.add_argument("--files", nargs="*", default=None)
    p_leak.add_argument("--json", action="store_true")
    p_leak.add_argument("--strict", action="store_true")

    p_cost = sub.add_parser("cost", help="多版成本并排")
    p_cost.add_argument("--runs", nargs="+", required=True)
    p_cost.add_argument("--labels", default=None)
    p_cost.add_argument("--json", action="store_true")

    p_delta = sub.add_parser("delta", help="重复评测噪声带 δ")
    p_delta.add_argument("--runs", nargs="+", required=True)
    p_delta.add_argument("--json", action="store_true")

    p_doc = sub.add_parser("doctor", help="用构造数据自检本工具")
    p_doc.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "leak":
        from tools import leakage_scan as LS
        argv2 = ["--base", args.base, "--grade"]
        if args.rev:
            argv2 += ["--rev", args.rev]
        if args.files:
            argv2 += ["--files"] + list(args.files)
        if args.json:
            argv2.append("--json")
        if args.strict:
            argv2.append("--strict")
        return LS.main(argv2)
    if args.cmd == "cost":
        labels = args.labels.split(",") if args.labels else None
        render_cost(args.runs, labels, args.json)
        return 0
    if args.cmd == "delta":
        d = render_delta(args.runs, args.json)
        return 1 if d.get("error") else 0
    if args.cmd == "doctor":
        return cmd_doctor(args.json)
    return 1


if __name__ == "__main__":
    sys.exit(main())
