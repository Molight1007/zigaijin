# -*- coding: utf-8 -*-
"""云端评测「修改效果捕捉」脚本 —— 为 12 处修复各定义一个可测量指标。

用法：
    python tools/effect_capture.py <结果.jsonl> [--baseline <对照.jsonl>]

设计目的（用户要求"要能捕捉修改带来的效果"）：
    每一处代码修复都必须有**能从评测产物里读出来的观测指标**，
    而不是只看正确率。正确率受模型能力限制（实测候选池含正确答案 0/6），
    因此本脚本用于回答一个更精确的问题：**我改的这 12 处，是否真的生效了。**

判据分三档：
    ✅ 生效   —— 指标达到预期
    ⚪ 未触发 —— 该场景本轮没出现（如 Lean 没生成恒等式），不算失败
    ❌ 未生效 —— 场景出现了，但指标仍不符合预期（这才是回归）
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

# 乱码模式：UTF-8 字节被按 ISO-8859-1 解出来的典型形态
MOJIBAKE = re.compile(r"(?:[ÃÂÐÑ][\u0080-\u00bf][\u0080-\u00ff]?|ï¼|â€|è®|é”|ï¼ˆ)")

CHECK_SPECS = [
    ("1", "流式中文乱码", "revise_feedback 中含乱码的条目数"),
    ("2", "深复核推理被丢弃", "revise_feedback 最长条目的字符数"),
    ("3", "结构化错因未入反馈", "revise_feedback 中『全题错因分布』出现次数"),
    ("4", "6.5 不读对抗定位", "revise_feedback 中『已检出的具体错误』出现次数"),
    ("5", "Lean 反馈角色错配", "revise_feedback 中『本条是无法核验』出现次数"),
    ("6", "证伪失败不留痕", "diag.falsify_miss 条目数"),
    ("7", "6.5 绕过轮数上限", "diag.revise_round 最大值"),
    ("8", "无进展保护（死代码）", "trace 中『无进展』出现次数"),
    ("9", "反馈重复灌入", "revise_feedback 条数 / 去重后条数"),
    ("10", "恒等式骗过验证", "trace 中『恒等式』出现次数"),
    ("11", "门全拒等于全留", "trace 中『全部拒掉』出现次数"),
    ("12", "候选含非答案", "candidate_invalid / answer_salvaged 条目数"),
    ("13", "零多样性不补采", "trace 中『零多样性』补采触发次数"),
    ("14", "反馈含重做正文", "反馈中『证明：』/『正确答案应为』出现次数"),
]


def _norm(s: str) -> str:
    return re.sub(r"[\s]+", " ", str(s or "")).strip()


def _dedup_key(s: str) -> str:
    k = _norm(s)
    k = re.sub(r"候选\s*\d+", "候选", k)
    k = re.sub(r"#\d+", "#", k)
    return k[:160]


def collect(path: pathlib.Path) -> dict:
    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    out = {
        "题数": len(rows),
        "正确数": sum(1 for r in rows if r.get("correct")),
        "乱码条目": 0, "反馈最长": 0, "错因分布": 0, "对抗定位": 0, "无法核验": 0,
        "falsify_miss": 0, "revise_max": 0, "无进展": 0,
        "反馈条数": 0, "去重后条数": 0, "恒等式": 0, "全拒": 0,
        "candidate_invalid": 0, "answer_salvaged": 0,
        "去重事件": 0, "去重前": 0, "去重后": 0, "入池拦截": 0, "抢救": 0,
        "零多样性补采": 0, "反馈含重做": 0,
        "池含gold": 0, "候选总数": 0, "真答案候选": 0,
    }

    def _norm_ans(s):
        s = str(s or "")
        m = re.search(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", s)
        if m:
            s = m.group(1)
        s = re.sub(r"[\s$\\{}\[\]()]", "", s)
        return s.strip().lower()

    for r in rows:
        d = r.get("diag") or {}
        tr = r.get("trace") or []
        fb = [_norm(x) for x in (d.get("revise_feedback") or [])]
        out["反馈条数"] += len(fb)
        out["去重后条数"] += len({_dedup_key(x) for x in fb})
        out["乱码条目"] += sum(1 for x in fb if MOJIBAKE.search(x))
        out["反馈最长"] = max(out["反馈最长"], max((len(x) for x in fb), default=0))
        out["错因分布"] += sum(1 for x in fb if "全题错因分布" in x)
        out["对抗定位"] += sum(1 for x in fb if "已检出的具体错误" in x)
        out["无法核验"] += sum(1 for x in fb if "本条是无法核验" in x)
        out["falsify_miss"] += len(d.get("falsify_miss") or [])
        out["revise_max"] = max(out["revise_max"], int(d.get("revise_round") or 0))
        out["candidate_invalid"] += len(d.get("candidate_invalid") or [])
        out["answer_salvaged"] += len(d.get("answer_salvaged") or [])
        for t in tr:
            if not isinstance(t, dict):
                continue
            c = str(t.get("content", ""))
            if "无进展" in c:
                out["无进展"] += 1
            if "恒等式" in c:
                out["恒等式"] += 1
            if "全部拒掉" in c:
                out["全拒"] += 1
            # ★ 去重与入池门槛的留痕在 trace 里（不在 diag），必须从这里读
            m = re.search(r"反馈去重：(\d+) 条 → (\d+) 条", c)
            if m:
                out["去重事件"] += 1
                out["去重前"] += int(m.group(1))
                out["去重后"] += int(m.group(2))
            if "不含最终答案 → 不入池" in c:
                out["入池拦截"] += 1
            if "从句子中抢救出答案" in c:
                out["抢救"] += 1
            if "零多样性" in c:
                out["零多样性补采"] += 1
        # ★ 反馈里不应再出现"重做正文"（`_clip_review_prose` 的判据）
        for x in fb:
            if "证明：" in x or "正确答案应为" in x:
                out["反馈含重做"] += 1
            if t.get("step") == "falsify_miss":
                out["falsify_miss"] += 1
        gold = _norm_ans(r.get("gold"))
        hit = False
        for c in (r.get("candidates") or []):
            a = str(c.get("answer") if isinstance(c, dict) else getattr(c, "answer", "") or "")
            out["候选总数"] += 1
            na = _norm_ans(a)
            if na and re.search(r"[\d$\\=]|boxed", a, re.I):
                out["真答案候选"] += 1
            if na and gold and (na == gold or (not gold.isdigit() and not na.isdigit()
                                               and (gold in na or na in gold))):
                hit = True
        out["池含gold"] += 1 if hit else 0
    return out


def verdict(idx: str, m: dict) -> tuple:
    n, fmax = m["乱码条目"], m["反馈最长"]
    if idx == "1":
        return ("✅" if n == 0 else "❌"), f"{n} 条（应 0）"
    if idx == "2":
        return ("✅" if fmax > 1000 else ("⚪" if m["反馈条数"] == 0 else "❌")), f"最长 {fmax} 字符（应 >1000）"
    if idx == "3":
        return ("✅" if m["错因分布"] > 0 else "❌"), f"出现 {m['错因分布']} 次"
    if idx == "4":
        return ("✅" if m["对抗定位"] > 0 else "⚪"), f"出现 {m['对抗定位']} 次"
    if idx == "5":
        return ("✅" if m["无法核验"] > 0 else "⚪"), f"出现 {m['无法核验']} 次"
    if idx == "6":
        return ("✅" if m["falsify_miss"] > 0 else "⚪"), f"{m['falsify_miss']} 条"
    if idx == "7":
        return ("✅" if m["revise_max"] <= 5 else "❌"), f"max={m['revise_max']}（应 ≤5）"
    if idx == "8":
        return ("✅" if m["无进展"] > 0 else "⚪"), f"出现 {m['无进展']} 次"
    if idx == "9":
        # ★ 判据以 trace 的去重事件为准（diag.revise_feedback 是**渲染前**的原始列表，
        #   去重发生在 solver 渲染处，因此 diag 里必然仍有重复 —— 原判据测错了位置）。
        if m["去重事件"] > 0:
            return "✅", (f"去重执行 {m['去重事件']} 次："
                          f"{m['去重前']} 条 → {m['去重后']} 条")
        return "⚪", "本轮未触发去重（反馈本无重复或未发生重写）"
    if idx == "10":
        return ("✅" if m["恒等式"] > 0 else "⚪"), f"拦截 {m['恒等式']} 次"
    if idx == "11":
        return ("✅" if m["全拒"] > 0 else "⚪"), f"出现 {m['全拒']} 次"
    if idx == "12":
        # ★ 同上：门槛留痕在 trace
        if m["入池拦截"] or m["抢救"]:
            return "✅", f"入池拦截 {m['入池拦截']} 个 / 抢救出答案 {m['抢救']} 个"
        return "⚪", "本轮无候选被拦或被抢救（模型给出的都已是规范答案）"
    if idx == "13":
        if m["零多样性补采"]:
            return "✅", f"零多样性 → 触发补采 {m['零多样性补采']} 次"
        return "⚪", "本轮无零多样性场景（候选本就有分歧）"
    if idx == "14":
        return ("✅" if m["反馈含重做"] == 0 else "❌"), \
               f"{m['反馈含重做']} 条含『证明：/正确答案应为』（应 0）"
    return "?", ""


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = pathlib.Path(sys.argv[1])
    m = collect(path)
    base = None
    if "--baseline" in sys.argv:
        bp = pathlib.Path(sys.argv[sys.argv.index("--baseline") + 1])
        if bp.exists():
            base = collect(bp)

    print("=" * 78)
    print("修改效果捕捉 · %s" % path.name)
    print("=" * 78)
    print()
    print("总体：%d 题 / 正确 %d / 池含 gold %d" % (m["题数"], m["正确数"], m["池含gold"]))
    print("候选：共 %d 个，其中含真答案 %d 个（%.0f%%）" % (
        m["候选总数"], m["真答案候选"],
        100.0 * m["真答案候选"] / max(1, m["候选总数"])))
    print()
    print("%-3s %-22s %-5s %s" % ("#", "修复项", "判定", "实测"))
    print("-" * 78)
    ok = trig = bad = 0
    for idx, name, _ in CHECK_SPECS:
        v, txt = verdict(idx, m)
        ok += (v == "✅"); trig += (v == "⚪"); bad += (v == "❌")
        print("%-3s %-22s %-5s %s" % (idx, name, v, txt))
    print("-" * 78)
    print("✅ 生效 %d ／ ⚪ 未触发 %d ／ ❌ 未生效 %d" % (ok, trig, bad))
    if base:
        print()
        print("=== 与对照轮对比 ===")
        print("%-26s %-14s %-14s" % ("指标", "对照", "本轮"))
        for k in ("正确数", "池含gold", "乱码条目", "反馈最长", "错因分布",
                  "对抗定位", "revise_max", "入池拦截", "抢救", "真答案候选"):
            print("%-26s %-14s %-14s" % (k, base.get(k), m.get(k)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
