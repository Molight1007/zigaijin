# -*- coding: utf-8 -*-
"""跨模型独立检错（第二模型在线证伪）—— 2026-10-17 用户定案。

**为什么需要**
生成、自审、证伪**都由同一个模型（Intern-S2）完成** ⇒ 盲区相同
（它不觉得自己错的地方，换提示也审不出来）。本模块用**另一个模型**
（DeepSeek，读 `.env` 的 `DEEPSEEK_*` 凭据）做**独立**检错。

**判据（与 4.2 一致，用户定案）**
  · 只认**具体数学错误**：某步推导/不等式/等式不成立、结论与题面矛盾、
    能给出反例、算错（给正确值）；
  · 「缺推导 / 跳步 / 解答简略 / 未给证明过程」**不构成错误**；
  · 附带**题面约束与歧义点**（2.6 `gaps_a`）——检查答案是否**只在某个
    题面未明说的假设下**成立（089 型题意误读）。

**失败语义**
无 key / 网络失败 / 超时 / 输出不可解析 ⇒ `verdict="unknown"`（调用方放行、
只记 trace）。**不降级为"判错"**，也不掩盖：`error` 字段带原因。

对外：`cross_check(problem, answer, reasoning="", constraints=None, ...) -> dict`
"""
from __future__ import annotations

import json
import os
import urllib.request

_DEFAULT_BASE = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "deepseek-v4-flash"

_SYSTEM = """你是数学解答的**独立审查者**（第二个模型，与生成方无关）。
你的唯一任务：判断给定答案**是否正确**。

【只有以下情形才判 wrong=true】
1. 某一步的推导/等式/不等式在数学上**不成立**（能指出错在哪一步）；
2. 最终答案与题面条件**矛盾**；
3. 能给出**满足题面的具体反例**使答案失败；
4. 计算确实算错（给出正确值）。

【明确不算错误】
· 解答简略、缺少推导细节、跳过中间步骤、没有写证明过程；
· 你"觉得"答案不对但说不出具体理由；
· 答案的写法/形式与你的习惯不同。

【题面约束检查】
若给了【题面约束/歧义点】，必须额外判断：答案是否**只在某个题面未明说的
假设下**才成立？若是，在 evidence 中写明"依赖假设：…"（这属于需要在报告
中标出的风险，不一定是 wrong）。

只输出一个 JSON 对象（第一个字符必须是 `{`，不要任何解释或围栏）：
{"wrong": true/false, "confidence": 0.0-1.0, "evidence": "具体错误位置与理由（≤200字；无错误则空串）", "assumption": "答案依赖的未明说假设（无则空串）"}"""


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name) or default).strip()


def _parse_json(raw: str) -> dict:
    """容错解析（取最后一个含 wrong 的 JSON 对象）。"""
    if not raw:
        return {}
    txt = str(raw).strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
    dec = json.JSONDecoder()
    d = {}
    pos = 0
    while True:
        i = txt.find("{", pos)
        if i < 0:
            break
        try:
            obj, _ = dec.raw_decode(txt[i:])
        except Exception:  # noqa: BLE001
            pos = i + 1
            continue
        if isinstance(obj, dict) and "wrong" in obj:
            d = obj
        pos = i + 1
    return d


def cross_check(problem: str, answer: str, reasoning: str = "",
                constraints=None, model: str = "", timeout: float = 90.0,
                base_url: str = "", api_key: str = "") -> dict:
    """第二模型独立检错。返回 {"verdict": wrong|ok|unknown, "evidence", ...}。"""
    key = api_key or _env("DEEPSEEK_API_KEY")
    base = (base_url or _env("DEEPSEEK_API_BASE") or _DEFAULT_BASE).rstrip("/")
    mdl = model or _env("DEEPSEEK_MODEL") or _DEFAULT_MODEL
    if not key:
        return {"verdict": "unknown", "evidence": "", "error": "no DEEPSEEK_API_KEY",
                "model": mdl}
    if not str(answer or "").strip():
        return {"verdict": "unknown", "evidence": "", "error": "empty answer",
                "model": mdl}
    cons = [str(x)[:300] for x in (constraints or []) if str(x).strip()][:6]
    user = ("【题目】\n%s\n\n【待审答案】\n%s\n" % (str(problem or "")[:3000],
                                              str(answer or "")[:2000]))
    if reasoning:
        user += "\n【解答的推理过程（节选）】\n%s\n" % str(reasoning)[:6000]
    if cons:
        user += ("\n【题面约束/歧义点（题意理解阶段发现）】\n"
                 + "\n".join("- " + c for c in cons) + "\n")
    user += "\n请按系统提示判断，只输出 JSON。"
    body = json.dumps({
        "model": mdl,
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        # ★ 2026-10-18（p0b5_r4 实测：8/8 次 unparseable）：DeepSeek 是推理模型，
        #   2048 会被推理过程吃光、`content` 返回空 ⇒ 解析恒失败。实测给 8192。
        "max_tokens": 8192,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", "replace")
        data = json.loads(raw)
        _ch = (data.get("choices") or [{}])[0]
        _msg = _ch.get("message", {}) or {}
        txt = _msg.get("content", "") or ""
        _rc = _msg.get("reasoning_content", "") or ""
        _fr = _ch.get("finish_reason", "")
        # content 为空/无 JSON 时，退回在推理内容里找（推理模型常见：内容全在
        # reasoning_content，content 只剩空串或被截断）
        d = _parse_json(txt)
        if not d and _rc:
            d = _parse_json(_rc)
        if not d:
            return {"verdict": "unknown", "evidence": "",
                    "error": "unparseable(content_len=%d, reasoning_len=%d, finish=%s)"
                             % (len(txt), len(_rc), _fr),
                    "model": mdl}
    except Exception as exc:  # noqa: BLE001  失败不阻断、不判错
        return {"verdict": "unknown", "evidence": "",
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:160]),
                "model": mdl}
    if not d:
        return {"verdict": "unknown", "evidence": "", "error": "unparseable",
                "model": mdl}
    wrong = d.get("wrong")
    if isinstance(wrong, str):
        wrong = wrong.strip().lower() in ("true", "1", "yes", "是")
    return {"verdict": "wrong" if wrong else "ok",
            "evidence": str(d.get("evidence") or "")[:400],
            "assumption": str(d.get("assumption") or "")[:300],
            "confidence": float(d.get("confidence") or 0.0),
            "model": mdl}
