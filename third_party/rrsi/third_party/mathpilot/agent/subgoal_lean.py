# -*- coding: utf-8 -*-
"""子目标级 Lean 断言验证（2026-10-23，调研 AlphaProof / Nexus 后新增）。

为什么需要（问题定位）
----------------------
我们原来的 Lean 只做一件事：检查候选答案是不是一个**合法表达式**
（`answer_valid`）。它从不检查「这个值是否满足题目条件」，所以

    603729/2、777²/2、1207458    ← 全是错的
    Lean 判定：answer_valid（6/6 全过）

而 AlphaProof 的做法是把**候选答案注入命题再要求证明** —— 答案错就证不出来。
我们没做整题形式化（成本极高、准确率仅 60%），所以退一步：

    **只形式化子目标里的一句「可判断言」。**

例如本题 sg5 的断言：

    ∑_{k=1}^{N-1} 1/(p_k+p_{k+1}) ≥ (N-1)²/(N(N+1)-3) → 1

它形式化只要三行：

    theorem sg5 : Tendsto (fun N : ℕ => ((N:ℝ)-1)^2 / ((N:ℝ)*(N+1)-3))
                     atTop (nhds 1) := by ...

**证得出来 ⇒ 极限确实是 1 ⇒ 候选的 777²/2 被证伪。**

为什么只做"子目标级"而不是"整题级"
------------------------------------
· 整题形式化要表达「对所有序列、所有 m<n…」这类高阶结构，LLM 一次成型准确率极低
  （AlphaProof 用 Gemini 微调 + 2500 道专家种子，才做到 60%）。
· 子目标断言多是等式/不等式/极限，形状固定、依赖少，一次成型率高得多。
· 而且它是**中间信号**：证出来能立刻回灌搜索，不必等整题结束。

设计取舍
--------
· 默认 **关闭**（开关 `five_gate_subgoal_lean`）：未知成本，先用探针验证命中率。
· 只验**最后 limit 个子目标**：答案由最后几步决定，前面多是铺垫。
· 断言"可判性"用**轻量启发式**（含 = / ≤ / ≥ / → / Tendsto），不做语义判断。
· LLM 转 Lean 失败 / 编译超时 / 环境不可用 ⇒ 一律跳过（增益机制，绝不阻断主链）。
"""
from __future__ import annotations

import json
import os
import re

# 只挑"形状明确"的断言：等式、不等式、极限/收敛、整除
_ASSERT_RE = re.compile(
    r"(=|≤|≥|<|>|→|⟶|\\to|\\le|\\ge|Tendsto|收敛|趋于|整除|∣|\|)")

_SYSTEM = """你是 Lean 4 + Mathlib 的形式化工程师。任务：把一句数学断言写成
**能被 Lean 4 编译通过的定理**，并给出完整证明（**严禁使用 sorry / axiom**）。

硬性要求：
1. 只输出一个 ```lean 代码块，不要解释。
2. 结构固定：`import Mathlib.Tactic` + `theorem <名字> : <命题> := by <证明>`。
3. 证明必须真实成立。若你判断该断言**为假**，请改为证明它的否定，并把定理名
   写成 `not_<名字>`；这是合法且有价值的输出。
4. 不要引入题面之外的假设（不要偷偷加强条件）。
5. 若该断言无法用 Lean 表达（涉及无法形式化的对象），输出 `CANNOT_FORMALIZE`。
"""

_USER = """【原题（仅供理解符号含义，不要形式化整题）】
{problem}

【需要形式化的子目标断言】
{statement}

【上下文（前一步的结论，可直接引用为假设）】
{context}

请输出一个 ```lean 代码块。"""


def pick_assertions(chain: list[dict], limit: int = 6) -> list[dict]:
    """从子目标链里挑出「形如断言」的条目（只挑形状明确的）。

    判据是**轻量启发式**：结论行里出现等号/不等号/箭头/极限词。
    不做语义判断 —— 宁可漏，不可误挑，因为误挑会让 LLM 去形式化一段散文。
    """
    out = []
    for it in (chain or [])[-limit:]:
        concl = str(it.get("conclusion") or "").strip()
        # 结论行形如「本步结论: 等式 = ...」/「本步结论: 判定 = 真」/「本步结论: 值 = ...」
        body = concl
        for sep in ("本步结论:", "本步结论："):
            if sep in body:
                body = body.split(sep, 1)[1]
        body = body.strip()
        # 去掉「等式 = 」「判定 = 」这类前缀标签，留下真正的数学内容
        body = re.sub(r"^(等式|判定|值|不等式|结果)\s*[=＝:：]\s*", "", body).strip()
        # ★ 2026-10-23 实测：sg4/sg5 的结论只有「判定 = 真」「值 = 777²/2」，
        #   真正的断言在 `basis`（依据）里。所以结论行不可判时，从依据里
        #   **取最后一句含等号/不等号的话**（结论性最强的那句）。
        if not body or not _ASSERT_RE.search(body) or body in ("真", "假"):
            _basis = str(it.get("basis") or "")
            _cands = [x.strip() for x in re.split(r"[。；;\n]", _basis)
                      if _ASSERT_RE.search(x)]
            body = _cands[-1] if _cands else ""
        # 元话语（"本步依据: …" "由子目标 X 的 …"）不是断言，跳过
        if body.startswith(("本步依据", "本步推导", "由子目标", "子目标", "【")):
            continue
        if not body or len(body) < 6:
            continue
        out.append({"id": str(it.get("id")), "statement": body[:400],
                    "title": str(it.get("title") or "")[:120]})
    return out


def _extract_lean(raw: str) -> str:
    """从模型输出里抽出 Lean 代码。

    ★ 2026-10-23 实测（第一版在这里翻车）：推理模型会把"该怎么写"的思考过程
      也输出出来，里面同样出现 ```lean 字样 ⇒ 按第一个代码块取，取到的是自言自语
      （`code block". "Structure fixed: ...`）。必须：
        ① 只取**最后一个**含 theorem/example 的代码块（最终答案在最后）；
        ② 没有代码块时，从 "import " 开始截断。
    """
    t = str(raw or "")
    blocks = re.findall(r"```(?:lean|Lean)?\s*([\s\S]*?)```", t)
    cands = [b.strip() for b in blocks
             if ("theorem" in b or "example" in b or "lemma" in b)]
    if cands:
        return cands[-1]
    if blocks:
        return blocks[-1].strip()
    i = t.find("import ")
    return t[i:].strip() if i != -1 else ""


def to_lean(assertion: str, problem: str, context: str, client,
            temperature: float = 0.0, max_tokens: int = 32768) -> str:
    """调 LLM 把断言转成 Lean 定理（含证明）。失败返回空串。

    ⚠ `max_tokens` 必须给足：推理模型（Intern / DeepSeek）的思考会吃光预算，
      实测 8192 时 finish_reason=length ⇒ 正文没输出 ⇒ 看起来像"模型不肯写"。
    """
    try:
        raw = client.chat(messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(
                problem=str(problem or "")[:2000],
                statement=assertion[:600],
                context=str(context or "")[:800])},
        ], temperature=temperature, max_tokens=max_tokens)
    except Exception:  # noqa: BLE001  增益机制，任何失败都跳过
        return ""
    code = _extract_lean(raw)
    if "CANNOT_FORMALIZE" in code or "theorem" not in code:
        return ""
    return code


_FIX_SYSTEM = """你是 Lean 4 + Mathlib 工程师。下面这段 Lean 代码**编译失败**，请修正。

修正纪律（按此顺序判断）：
1. `unknown constant/identifier X`：X 不存在。换成真实存在的引理，或改用
   `simp` / `omega` / `norm_num` / `linarith` / `ring` 这类通用 tactic。
2. `consider marking it as 'noncomputable'`：给定义加上 `noncomputable` 前缀。
3. `unexpected token / invalid syntax`：语法错，检查括号与 `:= by` 结构。
4. `unsolved goals`：证明没做完，补 tactic。
5. **严禁**用 `sorry` / `axiom` / `admit` 绕过。
6. 不要改变命题本身的意思（左端的 statement 必须保持一致）。

只输出一个 ```lean 代码块，不要解释。"""

_FIX_USER = """【原命题（必须保持一致）】
{statement}

【当前代码】
```lean
{code}
```

【Lean 编译错误】
{error}

请输出修正后的完整代码。"""


def _fix_once(code: str, error: str, statement: str, client,
              max_tokens: int = 32768) -> str:
    """把编译错误喂回模型，返回修正后的代码；失败返回空串。

    ⚠ max_tokens 必须给足（实测 8192 时证明写到 `:= by` 就断了，编译报
      `expected token` —— 看起来像"改不对"，其实是输出被截断）。
    """
    try:
        raw = client.chat(messages=[
            {"role": "system", "content": _FIX_SYSTEM},
            {"role": "user", "content": _FIX_USER.format(
                statement=str(statement or "")[:600],
                code=str(code or "")[:4000],
                error=str(error or "")[:1200])},
        ], temperature=0.0, max_tokens=max_tokens)
    except Exception:  # noqa: BLE001
        return ""
    new = _extract_lean(raw)
    if not new or "theorem" not in new:
        return ""
    # 防退化：修正版不许引入 sorry/axiom（源码层先拦一道，编译层还会再拦）
    low = new.lower()
    if "sorry" in low or re.search(r"\baxiom\b", low):
        return ""
    return new


def to_lean_iter(assertion: str, problem: str, context: str, client, *,
                 work_dir: str = "", lean_exe: str = "",
                 max_rounds: int = 3, timeout: float = 300) -> dict:
    """生成 Lean 代码，**并用编译错误迭代修正**（AlphaProof 的 truncate-and-resume 简化版）。

    为什么需要（2026-10-24 用户点出）：
    模型不是"不知道要用 Lean"，而是**记不住 300 万行 Mathlib 里的确切名字**——
    实测两个错：`Unknown constant Finset.card_Ico`（引理名不存在）、
    `Real.sqrt 需要 noncomputable`（漏标记）。这类问题**不该靠模型记忆解决**，
    而应由智能体用编译器反馈去修。

    流程：生成 → 编译 → 失败则把错误原文喂回模型 → 再编译 …… 最多 max_rounds 轮。

    返回 ``{"code","ok","error","rounds"}``；`work_dir` 为空时只生成不迭代。
    """
    code = to_lean(assertion, problem, context, client)
    if not code:
        return {"code": "", "ok": None, "error": "cannot_formalize", "rounds": 0}
    if not work_dir:
        return {"code": code, "ok": None, "error": "no_work_dir", "rounds": 0}
    try:
        from tools.lean_local.lean_bridge import _compile_lean
    except Exception as exc:  # noqa: BLE001
        return {"code": code, "ok": None, "error": "lean_bridge: %s" % exc,
                "rounds": 0}

    last_err = ""
    for rnd in range(1, max_rounds + 1):
        try:
            res = _compile_lean(code, work_dir, lean_executable=lean_exe,
                                timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"code": code, "ok": None,
                    "error": "%s: %s" % (type(exc).__name__, str(exc)[:160]),
                    "rounds": rnd}
        if res.get("ok"):
            return {"code": code, "ok": True, "error": "", "rounds": rnd}
        last_err = str(res.get("error") or "")[:800]
        if rnd >= max_rounds:
            break
        fixed = _fix_once(code, last_err, assertion, client)
        if not fixed:
            break
        code = fixed
    return {"code": code, "ok": False, "error": last_err, "rounds": max_rounds}


def verify_chain(chain: list[dict], problem: str, *, client, work_dir: str,
                 lean_exe: str, limit: int = 4, timeout: float = 300) -> dict:
    """主入口：挑断言 → 转 Lean → 编译 → 返回逐条结果。

    返回 ``{"n_picked": n, "items": [{"id","statement","ok","error","lean"}],
             "n_ok": k, "n_refuted": r}``。
    """
    picks = pick_assertions(chain, limit=limit)
    items = []
    if not picks:
        return {"n_picked": 0, "items": [], "n_ok": 0, "n_refuted": 0}
    try:
        from tools.lean_local.lean_bridge import _compile_lean
    except Exception:  # noqa: BLE001  环境缺失 ⇒ 整体跳过
        return {"n_picked": len(picks), "items": [], "n_ok": 0, "n_refuted": 0,
                "error": "lean_bridge_unavailable"}

    ctx_text = ""
    for p in picks:
        # ★ 2026-10-24：改用「生成 + 编译反馈迭代」——模型记不住 Mathlib 的确切
        #   引理名/标记要求，靠编译器反馈纠正，而不是指望它一次写对。
        r = to_lean_iter(p["statement"], problem, ctx_text, client,
                         work_dir=work_dir, lean_exe=lean_exe,
                         max_rounds=3, timeout=timeout)
        code = r.get("code") or ""
        ok = bool(r.get("ok"))
        items.append({"id": p["id"], "statement": p["statement"], "ok": ok,
                      "error": "" if ok else str(r.get("error") or "")[:300],
                      "lean": code, "rounds": r.get("rounds")})
        if ok:
            ctx_text = (ctx_text + "\n" + p["statement"])[:800]
        # 把已证出的结论作为下一步的上下文（减少重复推导）
        if ok:
            ctx_text = (ctx_text + "\n" + p["statement"])[:800]

    n_ok = sum(1 for x in items if x.get("ok") is True)
    n_ref = sum(1 for x in items if x.get("ok") is False)
    return {"n_picked": len(picks), "items": items,
            "n_ok": n_ok, "n_refuted": n_ref}


def summarize(res: dict) -> str:
    """一行摘要，写进 record / diag。"""
    if not res or not res.get("n_picked"):
        return ""
    bad = [x for x in res.get("items") or [] if x.get("ok") is False]
    if not bad:
        return "子目标 Lean 验证：%d 条断言，全部编译通过" % res["n_picked"]
    return "子目标 Lean 验证：%d 条断言，%d 条**未通过**（%s）" % (
        res["n_picked"], len(bad),
        "，".join("sg%s" % b.get("id") for b in bad[:5]))
