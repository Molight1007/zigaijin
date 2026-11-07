# -*- coding: utf-8 -*-
"""子目标链**复算式**自检：在求解层收口前，找"最后一步算错"。

为什么需要（2026-10-23 实测）
------------------------------
official112-008 的子目标链本身几乎都对：

    sg2  D·∑1/(p_k+p_{k+1}) ≤ 777²
    sg3  ∑1/(p_k+p_{k+1}) ≥ (N-1)² / ∑(p_k+p_{k+1})        （AM-HM）
    sg4  ∑(p_k+p_{k+1}) = N(N+1) - (p_1+p_N) ≤ N(N+1) - 3    （**正确**）
    sg5  ∴ 下界 → 1  ⇒  D ≤ 777² / 1 = 777²                  ← 应为这个
    sg5' 实际输出   D = 777²/2                               ← **凭空除了 2**

错在**最后一步的代换**，而且它会一路传下去：merge 直接写 `\boxed{777²/2}`，
3_solve 的 6 个候选全围绕它，4.1/4.2 全杀（正确），4.6 重写两轮（79 分钟/轮）还是它。

结论：**结论行（"判定=真"）看不出错，必须看依据与结论之间的代换**。
机械正则做不了这件事（公式是自然语言混合的），所以用**第二模型复算**
——它与生成模型不同源，能独立算一遍（与 `cross_model_check` 同一思路，
但作用点从"答案"上移到"子目标链"）。

设计约束
--------
· **只查可复算的量**：数值、量级、代数代换；不做"讲得够不够"的语义判断。
· **必须给出可核对的一步**：指不出具体哪一步，就不算命中（防瞎猜）。
· 失败/超时/无 key ⇒ 返回空列表（增益机制，绝不阻断主链）。
"""
from __future__ import annotations

import json
import os
import re

_DEFAULT_BASE = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"

_SYSTEM = """你是数学推导的**复核员**，只做一件事：把别人给出的子目标链**自己重算一遍**，
找出**数值 / 量级 / 代数代换**层面的错误。

严格纪律：
1. 只报"可复算"的错：算错的数、错的量级、把 A 代成 B、极限取错、凭空多/少一个因子。
2. **不评论**推理是否详细、是否跳步、是否严谨 —— 那不是你的职责。
3. 每一条报错必须写清：哪个子目标、依据里的哪一步、正确结果应该是多少。
4. 找不到可复算的错，就返回空数组 —— 不许为了"有输出"而硬凑。

只输出 JSON，不要任何解释文字：
{"suspects": [{"id": "<子目标id>", "kind": "numeric|algebra|limit|substitution",
               "step": "<出错的依据/代换，一句话>", "expected": "<正确结果>",
               "got": "<链里给出的结果>"}]}"""

_USER = """【题目】
{problem}

【子目标链（按依赖顺序，含结论与依据）】
{chain}

请复算：从最早的子目标一路推到最后一条，检查每一处的数值与代换是否成立，
特别是**最后一条把前面结论合起来求最终值时**有没有算错。
只报可复算的错，输出 JSON。"""


def _env(name: str) -> str:
    v = os.environ.get(name, "")
    if v:
        return v.strip()
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for line in open(os.path.join(root, ".env"), encoding="utf-8",
                         errors="ignore"):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _fmt_chain(chain: list[dict], max_each: int = 600,
               limit: int = 4) -> str:
    # ★ 2026-10-23 实测：8 个子目标 × 600 字 ⇒ DeepSeek 的推理把 16384 token
    #   全吃光（finish_reason=length ⇒ 空响应 ⇒ 静默失效）。只喂**最后 limit 个**
    #   （最终答案来自最后几步，前面已通过的结论不必复述），并抬高 max_tokens。
    lines = []
    for it in chain[-limit:]:
        cid = it.get("id")
        title = str(it.get("title") or "")[:200]
        concl = str(it.get("conclusion") or it.get("concl") or "")[:200]
        basis = str(it.get("basis") or "")[:max_each]
        lines.append("--- 子目标 %s ---\n结论: %s\n依据: %s"
                     % (cid, concl or "(未给出)", basis or title))
    return "\n\n".join(lines)


def _check_batch(chain: list[dict], problem: str, *,
                 timeout: int = 900, max_tokens: int = 32768) -> list[dict]:
    """对一批子目标做一次复算（内部函数，见 check_subgoal_chain 的分批策略）。"""
    """返回可疑子目标列表（空列表 = 未发现可复算的错 / 机制不可用）。

    ⚠ `max_tokens` 必须给够：DeepSeek 是推理模型，2048 会被思考吃光
      （实测 finish_reason=length ⇒ 拿到空响应 ⇒ 静默失效）。

    每项形如 ``{"id","kind","step","expected","got"}``。
    """
    if not chain:
        return []
    key = _env("DEEPSEEK_API_KEY")
    if not key:
        return []
    base = (_env("DEEPSEEK_API_BASE") or _DEFAULT_BASE).rstrip("/")
    model = _env("DEEPSEEK_MODEL") or _DEFAULT_MODEL
    try:
        from utils.llm_client import LLMClient
        cli = LLMClient(api_key=key, base_url=base, model=model, timeout=timeout)
        raw = cli.chat(messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(
                problem=str(problem or "")[:2500],
                chain=_fmt_chain(chain))},
        ], temperature=0.0, max_tokens=max_tokens)
    except Exception:  # noqa: BLE001  增益机制，任何失败都静默
        return []
    return _parse(raw)


def check_subgoal_chain(chain: list[dict], problem: str, *,
                        timeout: int = 900, max_tokens: int = 32768,
                        batch: int = 4, char_budget: int = 3500) -> list[dict]:
    """复算子目标链，**分批覆盖全链**（避免早期子目标被截断漏检）。

    分批理由（2026-10-23 实测）：一次喂 8 个子目标时，DeepSeek 的推理会把
    16384 token 吃光（finish_reason=length ⇒ 空响应 ⇒ 静默失效）；只喂最后 4 个
    能跑通，但会**漏掉更早的错误**（实测 sg4 的极限代换就因此没被检出）。
    ⇒ 每批 `batch` 个、相邻批重叠 1 个（保证跨批的依赖能对上），结果按 id 去重。
    """
    if not chain:
        return []
    parts = _batches_by_chars(chain, budget=char_budget, max_items=batch)
    if len(parts) == 1:
        return _check_batch(parts[0], problem, timeout=timeout,
                            max_tokens=max_tokens)
    out: dict[str, dict] = {}
    for part in parts:
        for s in _check_batch(part, problem, timeout=timeout,
                              max_tokens=max_tokens):
            out.setdefault(s.get("id") or "", s)
    return list(out.values())


def _batches_by_chars(chain: list[dict], budget: int = 3500,
                      max_items: int = 4) -> list[list[dict]]:
    """按**字符预算**分批（而不是按个数），相邻批重叠 1 个。

    为什么改成按字符（2026-10-24 实测）：按个数分批时，8 个子目标切成 3 批，
    其中 2 批仍 `finish_reason=length`（单条 basis 最长近 600 字，4 条就 2400 字，
    加上题目与提示，推理预算仍会吃满）。按字符攒批后，长条目自然落在小批、
    短条目能多带几条 —— 既减少批数（调用次数），又降低截断率。

    重叠 1 个是为了让跨批的依赖关系（"由子目标 X 得…"）能在下一批里对上。
    """
    parts: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for it in chain:
        ln = (len(str(it.get("basis") or ""))
              + len(str(it.get("conclusion") or ""))
              + len(str(it.get("title") or "")))
        if cur and (cur_len + ln > budget or len(cur) >= max_items):
            parts.append(cur)
            cur, cur_len = [cur[-1]], len(str(cur[-1].get("basis") or ""))  # 重叠 1 个
        cur.append(it)
        cur_len += ln
    if cur:
        parts.append(cur)
    return parts


def _parse(raw) -> list[dict]:
    t = str(raw or "")
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for s in (d.get("suspects") or []):
        if not isinstance(s, dict):
            continue
        step = str(s.get("step") or "").strip()
        if not step:            # 指不出具体一步 ⇒ 不算命中
            continue
        out.append({
            "id": str(s.get("id") or ""),
            "kind": str(s.get("kind") or ""),
            "step": step[:240],
            "expected": str(s.get("expected") or "")[:160],
            "got": str(s.get("got") or "")[:160],
        })
    return out


def chain_from_trace(trace: list) -> list[dict]:
    """把 `ctx.subgoal_trace`（或 04_subgoal_results.jsonl 的行）转成检查用链。

    只取最后一轮（同一 id 保留最后出现的），因为最终答案来自最后一轮。
    """
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for it in trace or []:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("id"))
        res = str(it.get("result") or "")
        concl = ""
        basis = ""
        for line in res.split("\n"):
            if "本步结论" in line:
                concl = line.strip()[:200]
            elif "本步依据" in line:
                basis = line.strip()[:400]
        if not concl:
            # 无结论行 ⇒ 取前 300 字当"依据"（有些子目标直接写推导）
            concl = ""
            basis = res[:300]
        item = {"id": cid, "title": str(it.get("title") or ""),
                "conclusion": concl, "basis": basis}
        if cid not in by_id:
            order.append(cid)
        by_id[cid] = item
    return [by_id[c] for c in order]


def summarize(suspects: list[dict]) -> str:
    if not suspects:
        return ""
    ids = ",".join(s["id"] for s in suspects[:6])
    return "子目标链自检：%d 处可疑（%s）%s" % (
        len(suspects), ids,
        "；" + suspects[0]["step"][:80] if suspects else "")
