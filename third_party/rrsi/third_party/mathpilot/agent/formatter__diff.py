"""选择题选项聚类与差分检测（2026-10-08 从 agent/formatter.py 逐字搬出）。

簇④：`_group_option_sets`（把 {答案: 频次} 归并成 {选项字母集合: {...}}）+
`_objective_diff_probe`（平票时的定向 LLM 差分验证）。formatter.py 原位保留
**同名一行委托** ⇒ `FormatterAgent._group_option_sets` / `._objective_diff_probe`
契约不变。

⚠ 本模块**不** import formatter 主模块（仅依赖 `re` + 函数内惰性 import
`question_type.extract_options` / `base._normalize_chat_response`）。
"""
from __future__ import annotations

import re

from .base import TaskContext


def _group_option_sets(mv: dict) -> dict:
    """把 {答案字符串: 频次} 归并成 {选项字母集合: {"answers": [...], "votes": N}}。

        2026-09-13：供选择题「平票检测 + 差分定位」使用。只处理选项字母类答案。

        ⚠ 必须累加**票数**（`votes`），不能用 `len(answers)`——后者是"该集合有
        几种不同写法"（通常就是 1），会把任何分组都算成 1 票 ⇒ 误判平票。
        （实测教训：102 候选 B×4/AB×1/A×1，B 本应绝对多数，却因该口径错误
        触发差分检测，把答案从 B 改成 A，把本可答对的题改错。）
        """
    groups: dict = {}
    for _ans, _n in (mv or {}).items():
        _letters = re.findall(r"[A-E]", _ans)
        if not _letters:
            continue
        _k = frozenset(_letters)
        _g = groups.setdefault(_k, {"answers": [], "votes": 0})
        _g["answers"].append(_ans)
        try:
            _g["votes"] += int(_n)
        except (TypeError, ValueError):
            _g["votes"] += 1
    return groups


def _objective_diff_probe(self, ctx: TaskContext, groups: dict) -> str:
    """选择题候选平票时的**差分检测**（2026-09-13 用户设计）。

        设计意图：当多种选项组合并列最高频（如 `AB×2` 与 `ABC×2`），
        不要盲目取其一 —— A、B 是共识，**争议只在于 C**。于是：
            争议项 = 并列组合的并集 − 交集（此处 = {C}）
        只对争议项做一次定向 LLM 验证，其余沿用共识，把预算花在真正的分歧点。

        例：
            AB×2  vs ABC×2   → 争议 {C}
            ABD×2 vs ABC×2   → 争议 {C, D}
            A×2   vs B×2     → 争议 {A, B}（交集为空，两个都验）

        返回修正后的答案字母串（共识 ∪ 判为正确的争议项）；
        无平票 / 无争议项 / 调用失败 → ""（调用方回退多数投票）。
        """
    try:
        if len(groups) < 2:
            return ""
        _max_n = max(g["votes"] for g in groups.values())
        _tops = [k for k, g in groups.items() if g["votes"] == _max_n]
        if len(_tops) < 2:
            return ""                       # 无平票（存在唯一最高票）→ 交给多数投票
        _inter = set.intersection(*[set(k) for k in _tops])
        _union = set.union(*[set(k) for k in _tops])
        _diff = sorted(_union - _inter)
        if not _diff:
            return ""
        _opt_text = {}
        try:
            from .question_type import extract_options
            _opt_text = dict(extract_options(ctx.problem or "") or [])
        except Exception:  # noqa: BLE001
            _opt_text = {}
        _diff_lines = [
            "- 选项 {}：{}".format(
                L, _opt_text.get(L, "(题干未提取到该选项文本)"))
            for L in _diff]
        _sys = (
            "你是选择题审题专家。用户给你一道选择题，以及若干**存在分歧的选项**。"
            "请**只**针对这些分歧选项逐一判定其陈述是否正确，每个选项给出"
            "『正确』或『错误』并附一句理由；最后一行必须输出"
            "『【结论】: <这些选项中所有正确的字母>』（只写字母如 C 或 CD；"
            "若都不正确则写 无）。不要重述题目。"
        )
        _user = ("题目：\n{}\n\n以下选项在候选解答中存在分歧，请逐一判定：\n{}"
                 .format(ctx.problem, "\n".join(_diff_lines)))
        # 2026-09-13 超时护栏（口径照抄 verifier.py:334）：本阶段唯一 LLM 调用
        # 原先 max_tokens=65536 且无任何时间护栏 → 诱导长思考撞 LLMClient
        # 默认 180s 超时 + 1 次重试 ≈363s（实测 352/363/364s，占单题 1200s
        # 硬时限 30%）。deadline / 生成侧软截止已到 → 放弃差分检测，返回 ""
        # 由 _pick_best 回退到现成的多数投票结果。
        # 2026-09-13 修复（关键）：**不能**用 `gen_time_up()` 判断！
        # 它是"生成侧软截止"（= 单题 deadline 前 verify_reserve 秒），而
        # formatter 是流程的**最后一个阶段** ⇒ 走到这里时 `gen_time_up()`
        # **必然为 True** ⇒ 差分检测在 096/102 实测中**从未真正执行过**，
        # 全部静默回退多数投票（平票时 = 随机取第一个，正是用户指出的问题）。
        # 改为：只看"离单题硬限是否还够一次短调用"（max_tokens=512，
        # 约 20–40s；留 120s 余量以容纳一次重试）。
        if ctx.is_timed_out() or ctx.time_remaining() < 120:
            self.record(ctx, "finalize",
                        "选择题差分检测：剩余时间不足一次短调用，"
                        "跳过（回退多数投票）")
            return ""
        from .base import _normalize_chat_response
        # 2026-09-13：单次调用 → **最多 2 次**。原实现只调一次，且只认
        # `【结论】:` 一种写法，任一环节出问题就静默回退多数投票。
        # 现在：空响应/异常/**有输出但无结论**都会重试；失败写埋点。
        _text, _errs, _m = "", [], None
        for _attempt in range(2):
            try:
                _resp = self.client.chat(
                    messages=[{"role": "system", "content": _sys},
                              {"role": "user", "content": _user}],
                    # 2026-10-02 DeepSeek 适配：原 512 ⇒ 8192。原值基于「短直答 +
                    # prefill 抑制思维块」的 Intern-S 时代假设，对 reasoning 模型
                    # 必然截断（reasoning 先吃满预算、正文为空）。
                    temperature=0.0, max_tokens=8192,
                )
                _text = _normalize_chat_response(_resp) or ""
                if not _text.strip():
                    _errs.append("attempt{}:empty".format(_attempt + 1))
                    continue
                _m = (re.search(r"【结论】[:：]?\s*([A-E]+|无)", _text)
                      or re.search(r"(?:结论|答案|正确(?:的)?选项)\s*[:：]?\s*"
                                   r"([A-E]{1,5}|无)", _text))
                if _m:
                    break
                _errs.append("attempt{}:no_verdict".format(_attempt + 1))
            except Exception as _e:  # noqa: BLE001
                _errs.append("attempt{}:{}".format(
                    _attempt + 1, type(_e).__name__))
        if not _m:
            try:
                ctx._pick_diag = {
                    "branch": "diff_probe_no_verdict",
                    "errs": _errs,
                    "raw_tail": (_text or "")[-200:],
                }
            except Exception:  # noqa: BLE001
                pass
            return ""
        _v = _m.group(1)
        _ok = (set(_v) if _v != "无" else set()) & set(_diff)
        _ans = "".join(sorted(_inter | _ok))
        try:
            ctx._pick_diag = {
                "branch": "formatter_diff_probe",
                "tops": ["".join(sorted(t)) for t in _tops],
                "diff": "".join(_diff),
                "ok": "".join(sorted(_ok)),
                "picked": _ans,
            }
        except Exception:  # noqa: BLE001
            pass
        self.record(
            ctx, "finalize",
            "选择题差分检测：并列 {} → 争议项 {} → 判对 {} → 修正答案 {}".format(
                [''.join(sorted(t)) for t in _tops],
                ''.join(_diff),
                ''.join(sorted(_ok)) or '无',
                _ans or '(空)'))
        return _ans
    except Exception:  # noqa: BLE001
        return ""
