# -*- coding: utf-8 -*-
"""4.1 AI 自审逻辑错误 —— 五道审核链第 1 道（2026-10-14 用户定案）。

用户口径（原话）：「第一道先让 ai 自己审核一遍逻辑错误，然后排除错题」。

职责：对每个候选解答做一次 LLM **逻辑自审**（只审逻辑、不审数值 —— 数值归
第四道计算检测）；能明确指出具体逻辑错误的候选 → **淘汰** + 反馈入
``ctx.audit_reject_feedback``。

判定纪律（宁缺勿滥，防误杀）：
  · 只有 ``logic_error=true`` **且** ``error_desc`` 非空才淘汰；
  · 输出不可解析 / LLM 调用失败 ⇒ **保留**候选（记为 unparsed，绝不因基础设施
    故障淘汰候选）；
  · 拿不准由提示词纪律约束（拿不准 ⇒ false）。

★ 2026-10-20 义务账本（用户：「把蓝图倒推链反过来输出到第 4 部分」）：
  蓝图是倒推（目标 → 缺什么前提），自审是正推看解答文本 —— 审查者原先不知道
  「本题应当建立哪些事实」，只能拿完整证明模板去套，于是解答写得简略就被判
  「逻辑断裂」（p0b5_r5：`official112-019` 正确答案 `4` 因此被淘汰）。现注入
  `agent/obligation_ledger.py` 渲染的义务清单，并加第三条门：
  **账本可用 + 模型指认的义务在账本里查不到 ⇒ 放行**（漏填不惩罚，照原行为淘汰）。
  ★ 同时收敛提示词的审查范围：**省略中间步骤不算逻辑错误**，只认「断裂」。

观测：trace step=``self_audit``（逐候选 + 汇总），diag 键
``self_audit`` / ``self_audit_events``。

五道链位置：4.0 规范化 → **4.1 自审** → ②对抗证伪 → ③Lean 代回 → ④计算检测
→ ⑤投票。
"""
from __future__ import annotations

from .switch_registry import get_bool as _sw_get_bool
from .switch_registry import get_num as _sw_get_num
# 单候选自审的输入截断（防超长上下文；题目 4000 / 推理 6000 与既有阶段同量级）
_PROBLEM_MAX = 4000
_REASONING_MAX = 6000
_ANSWER_MAX = 500
# 自审输出是短 JSON（error_desc ≤300 字）。★ 2026-10-20：1024 → 8192。
#   推理模型先写 Thinking 再写 JSON，1024 被思考吃光 ⇒ finish_reason=length
#   （服务器实测 55 次），自审 JSON 解析失败 ⇒ 全部按"不可解析"保留。
_MAX_TOKENS = 8192
_TEMPERATURE = 0.1

_TRUE_TOKENS = {"true", "1", "yes", "y", "是", "对", "有", "有错误"}


def parse_self_audit_json(raw) -> dict:
    """容错解析自审输出。返回 {} 表示无法解析（调用方**保留**候选）。

    ★ 2026-10-17 修复（p0b5_r3 实测：84/88 候选 unparsed = 95%）：
      原实现取「第一个 `{` 到最后一个 `}`」再 json.loads —— 模型输出
      「推理文本 + JSON」或「多个 JSON 片段」时，该切片跨段**必然非法**
      ⇒ 解析恒失败。现改用 `JSONDecoder.raw_decode` 从每个 `{` 位置逐个尝试，
      取第一个能独立解析成 dict 且含 `logic_error` 的对象（可跳过前导推理）。
    """
    if not raw:
        return {}
    txt = str(raw).strip()
    # 剥 markdown 围栏（```json ... ```）
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
    import json as _json_sa
    _dec = _json_sa.JSONDecoder()
    d = None
    _pos = 0
    while True:
        _i = txt.find("{", _pos)
        if _i < 0:
            break
        try:
            _obj, _end = _dec.raw_decode(txt[_i:])
        except Exception:  # noqa: BLE001  该位置不是合法 JSON，试下一个 `{`
            _pos = _i + 1
            continue
        if isinstance(_obj, dict) and "logic_error" in _obj:
            # ★ 取**最后一个**含 logic_error 的合法对象：模型常见形态是
            #   「先举例格式 JSON / 再给实际结论 JSON」，结论在后；
            #   纯「推理 + JSON」时 JSON 也在末尾 —— 两者都覆盖。
            d = _obj
        _pos = _i + 1
    if d is None:
        # ★ 2026-10-18（p0b5_r4：自审 12/12 unparsed——模型不按 JSON 输出）：
        #   保守兜底——只认**明确表达「无逻辑错误」**的回答；说「有错但不给
        #   JSON」仍记为 unparsed（保留候选，绝不因格式问题淘汰）。
        _low = txt.lower()
        _no = any(k in _low for k in ("无逻辑错误", "没有逻辑错误", "未发现逻辑错误",
                                      "no logic error", "no error found"))
        _yes = any(k in _low for k in ("存在逻辑错误", "逻辑错误：", "有逻辑错误",
                                       "logic error:"))
        if _no and not _yes:
            return {"logic_error": False, "error_desc": "", "error_at": ""}
        return {}
    _v = d.get("logic_error")
    if isinstance(_v, str):
        d["logic_error"] = _v.strip().lower() in _TRUE_TOKENS
    else:
        d["logic_error"] = bool(_v)
    d["error_desc"] = str(d.get("error_desc") or "").strip()
    d["error_at"] = str(d.get("error_at") or "").strip()
    # ★ 2026-10-20：义务账本引用（"O2" / "none" / 空）。归一与是否对上都交给
    #   `obligation_ledger`，这里只做去空白，避免把模型的 `o2`、`第2条` 当非法值。
    d["obligation_id"] = str(d.get("obligation_id") or "").strip()
    return d


def _switch_enabled() -> bool:
    """读「五道链·第 1 道自审」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_self_audit")
    except Exception:  # noqa: BLE001
        return True


def _grounding_enabled() -> bool:
    """读「证据落地校验」开关（默认 True）。

    ★ 2026-10-22（五道链审计）：淘汰必须附「可机器复核的证据」——
      模型指认的错误位置必须是候选推理里**真实存在的原文片段**。
      桩测试实测：随便编一句「原文并不存在的一句话」作 ``error_at``，
      正确答案会被判逻辑错直接淘汰（幻觉型误杀）。
      =0 时回到旧口径（只认 desc 非空），仅用于 A/B 对照。
    """
    try:
        return _sw_get_bool("five_gate_grounding")
    except Exception:  # noqa: BLE001
        return True


def _retry_enabled() -> bool:
    """读「自审解析失败重问」开关（默认 True）。

    ★ 2026-10-22（五道链审计）：``unparsed`` 一律放行曾让这道长期空转
      （历史实测 84/88、12/12）。格式失败是可修复的失败，按最严格式重问一次；
      仍失败才记 unparsed 并标 ``verify_audit="unparsed"``（选择层可据此降权）。
      =0 回到旧口径（一次不成即放行）。
    """
    try:
        return _sw_get_bool("five_gate_audit_retry")
    except Exception:  # noqa: BLE001
        return True


def _vote_count() -> int:
    """读「自审采样次数」（默认 3，1 = 旧行为）。

    ★ 2026-10-22 端到端实测（DeepSeek 真实调用）：同一个**错误**候选
      （推理正确、只把终答写错）连跑 3 次，判定是「留 / 杀 / 留」——
      单次判定**不稳定**，漏杀率约 2/3。这正是「错的没杀」的直接原因：
      不是判据太松，是一次采样太偶然。
      同一实验里**正确**候选 3/3 稳定通过，所以提高采样不会增加误杀。
    """
    try:
        return max(1, int(_sw_get_num("five_gate_self_audit_votes")))
    except Exception:  # noqa: BLE001
        return 3


def _audit_once(orch, ctx, msgs) -> tuple:
    """调一次自审并解析；解析失败且开关允许时用最严格式重问一次。

    返回 ``(d, used_retry)``。``d`` 为空 dict = 两次都不可解析。
    异常不吞（调用方据此计 ``llm_fail``）。
    """
    raw = orch.llm(ctx, msgs, _TEMPERATURE, _MAX_TOKENS)
    d = parse_self_audit_json(raw)
    if d or not _retry_enabled():
        return d, False
    try:
        raw2 = orch.llm(ctx, msgs + [
            {"role": "assistant", "content": str(raw or "")[:2000]},
            {"role": "user", "content":
             "上一次回答无法解析。请**只**输出一个 JSON 对象，"
             "第一个字符必须是 `{`，最后一个字符必须是 `}`；"
             "不要任何解释、不要 Markdown 围栏。"},
        ], 0.0, _MAX_TOKENS)
    except Exception:  # noqa: BLE001
        # 重问失败只说明"格式救不回来"，**不是模型不可用** —— 记为 unparsed
        # 而不是 llm_fail，否则会把格式问题误算成基础设施故障。
        return {}, True
    return parse_self_audit_json(raw2), True

def _judge_once(d, full_reasoning: str, grounding: bool) -> str:
    """单次判定 → ``error`` / ``clean`` / ``ungrounded`` / ``""``（不可解析）。

    ``error`` 只在这三件事同时成立时给出：模型判了逻辑错、给了描述、
    且**指认的出错位置能在原文里逐字找到**（证据落地门，见 evidence_grounding）。
    """
    if not d:
        return ""
    if not (d.get("logic_error") and d.get("error_desc")):
        return "clean"
    if grounding:
        from .evidence_grounding import check_location
        ok, _r, _note = check_location(str(d.get("error_at") or ""),
                                       full_reasoning)
        if ok is not True:
            return "ungrounded"
    return "error"


def _audit_batch(orch, ctx, msgs, n: int) -> list:
    """并发跑 n 次自审，返回长度 n 的列表：元素是 ``(d, used_retry)`` 或异常。

    ★ 2026-10-22（不浪费并发额度）：多次采样是**互相独立**的调用，
      串行执行会让 4.1 的耗时随票数线性增长（3 票 = 3 倍时间）。
      而云端实测的主要瓶颈正是"单题墙钟太长"——用线程池并发后
      墙钟 ≈ 单次耗时，代价只是"错误候选不再首票短路"（最多多 2 次调用）。
      `BaseAgent.llm` 无共享锁、LLMClient 每次新建连接 ⇒ 并发安全；
      唯一副作用是 `ctx.budget.spend` 计数可能少记，不影响判定。
    """
    from concurrent.futures import ThreadPoolExecutor
    if n <= 1:
        try:
            return [_audit_once(orch, ctx, msgs)]
        except Exception as exc:  # noqa: BLE001  调用方按"故障"处理
            return [exc]
    out: list = [None] * n
    with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
        futs = {ex.submit(_audit_once, orch, ctx, msgs): i for i in range(n)}
        for fut, idx in futs.items():
            try:
                out[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                out[idx] = exc
    return out


def _obligation_block(ctx, config) -> tuple:

    """取「蓝图倒推链反演出的义务清单」（2026-10-20，与 4.2 同一开关同一实现）。

    返回 ``(ledger, text)``；无蓝图 / 开关关闭 / 构建异常 ⇒ ``(None, "")``，
    调用方必须据此回落到**原口径**（不加这一段、不加义务门）。
    """
    try:
        if not bool(getattr(config, "audit_obligation_ledger", True)):
            return None, ""
        from .obligation_ledger import build_ledger, render_for_audit
        lg = build_ledger(getattr(ctx, "blueprint", None))
        _text = render_for_audit(lg)
        # ★★ 2026-10-23（用户：「把正确的判断为错误的」）：
        #   实测 official112-008 的义务 O5 断言 `(m+n)D ≤ 777²`，代入 gold D=777²
        #   得 m+n ≤ 1，与 m<n 正整数矛盾 —— **账本自己错了**；而 4.1 按「必须满足
        #   义务」判，把正确指出该矛盾的候选淘汰掉。账本错 ⇒ 下游越审越"自洽"。
        #   这里做一次机械自检，**把可疑义务显式标给审查者**（不剔除、不改判据），
        #   让"候选反驳该义务"成为可接受的输出，而不是被判成逻辑错误。
        try:
            from .obligation_selfcheck import check_ledger
            _sus = check_ledger(lg)
            if _sus:
                _md = getattr(ctx, "metadata", None)
                if isinstance(_md, dict):
                    _md["obligation_suspects"] = _sus
                _warn = ["【自检提示：下列义务可能**不成立**（待求量被变量因子压死 / "
                         "重复 / 空断言）—— 若候选明确反驳它们，不得据此判错，"
                         "应视为「上游义务可疑」并记录】"]
                for _s in _sus[:5]:
                    _warn.append("- %s（%s）：%s" % (
                        _s.get("oid", "?"), _s.get("kind", "?"),
                        str(_s.get("why", ""))[:140]))
                _text = (_text.rstrip() + "\n\n" + "\n".join(_warn))
        except Exception:  # noqa: BLE001  自检是增益，失败不影响账本本身
            pass
        return lg, _text
    except Exception:  # noqa: BLE001  账本是增益，构建失败绝不影响自审
        return None, ""


def self_audit_stage(self, ctx) -> None:
    """4.1 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例）。"""
    try:
        if not _switch_enabled():
            self.record(ctx, "self_audit",
                        "4.1 自审：开关关闭（five_gate_self_audit=0）")
            return
        cands = list(getattr(ctx, "candidates", None) or [])
        if not cands:
            self.record(ctx, "self_audit", "4.1 自审：候选池为空，跳过")
            return

        from prompts.self_audit import SELF_AUDIT_SYSTEM, SELF_AUDIT_USER

        problem = str(getattr(ctx, "problem", "") or "")[:_PROBLEM_MAX]
        kept, rejected, unparsed, llm_fail = [], [], [], 0
        n_retry = 0
        n_weak = 0                    # 只靠"部分票"淘汰的数量（弱证据，供观察）
        _votes = _vote_count()
        _grounding = _grounding_enabled()
        ungrounded = []   # 证据无法落地（位置缺失 / 原文里找不到）⇒ 不得据此淘汰
        # ★ 2026-10-20（用户：「把蓝图倒推链反过来输出到第 4 部分」）：
        #   账本只建一次（全候选共用），注入 prompt 并作为淘汰的第三条门。
        ledger, ledger_text = _obligation_block(ctx, getattr(self, "config", None))
        _lblock = (ledger_text.rstrip() + "\n\n") if ledger_text else ""
        unmapped = []   # 指认不到义务 ⇒ 视为评论了本题不要求的内容，放行

        for c in cands:
            cid = getattr(c, "id", "?")
            # 规范化形态优先（4.0 已挂）；否则回退原 answer
            ans = str(getattr(c, "answer_norm", None)
                      or getattr(c, "answer", "") or "")[:_ANSWER_MAX]
            reasoning = str(getattr(c, "reasoning", "") or "")[:_REASONING_MAX]
            # ★ 2026-10-22（多次采样）：单次判定不稳定 —— 实测同一错误候选
            #   连跑 3 次得「留/杀/留」；正确候选 3/3 稳定通过。
            #   因此采 `_votes` 次，**任一次给出带落地证据的逻辑错即淘汰**
            #   （漏杀代价 > 误杀代价；编造的证据已被 `_judge_once` 的落地门挡住）。
            full_reasoning = str(getattr(c, "reasoning", "") or "")
            _msgs = [
                {"role": "system", "content": SELF_AUDIT_SYSTEM},
                {"role": "user", "content": SELF_AUDIT_USER.format(
                    problem=problem, obligations=_lblock,
                    answer=ans, reasoning=reasoning)},
            ]
            d = None
            _hit = None
            _seen: list = []
            _fail_first = False
            # ★ 2026-10-22：并发采样（见 `_audit_batch`）—— 墙钟 ≈ 单次耗时。
            _outs = _audit_batch(self, ctx, _msgs, _votes)
            for _vi, _r in enumerate(_outs):
                if _r is None or isinstance(_r, Exception):
                    if _vi == 0:
                        _fail_first = True
                        llm_fail += 1
                        self.record(ctx, "self_audit",
                                    "cid=%s LLM 调用失败（保留候选）: %s"
                                    % (cid, str(_r)[:120]))
                    continue
                _dd, _used = _r
                if _used:
                    n_retry += 1
                _v = _judge_once(_dd, full_reasoning, _grounding)
                _seen.append(_v)
                if _v == "error" and _hit is None:
                    _hit = _dd
            if _fail_first:
                kept.append(c)
                continue

            if _hit is None:
                # 没有任何一次采样判定成立
                if not any(_seen):
                    unparsed.append(cid)
                    kept.append(c)
                    try:                # 未审 = 未验证，交给选择层降权
                        c.verify_audit = "unparsed"
                    except Exception:  # noqa: BLE001
                        pass
                    self.record(ctx, "self_audit",
                                "cid=%s %d 次采样均不可解析 ⇒ 保留并标记 unverified"
                                % (cid, len(_seen)))
                    continue
                if "ungrounded" in _seen:
                    ungrounded.append({
                        "id": cid, "answer": ans[:120], "at": "",
                        "desc": "", "note": "ungrounded_in_votes"})
                    self.record(ctx, "self_audit",
                                "cid=%s 指认证据无法落地（%d 次中 %d 次）⇒ 不淘汰"
                                % (cid, len(_seen), _seen.count("ungrounded")))
                kept.append(c)
                self.record(ctx, "self_audit",
                            "cid=%s 通过（采样 %d 次：%s）"
                            % (cid, len(_seen), "/".join(_seen)))
                continue

            d = _hit          # 命中：走义务账本门与淘汰
            # ★ 2026-10-22：并发采样后无法再按"第几票命中"判弱证据，
            #   改为按**票数比例**：只有部分票判错 ⇒ 弱证据（供观察/调阈值）。
            if sum(1 for _v2 in _seen if _v2 == "error") < len(_seen):
                n_weak += 1
            if d.get("logic_error") and d.get("error_desc"):
                desc = d["error_desc"][:300]
                at = (d.get("error_at") or "")[:120]
                oid_raw = str(d.get("obligation_id") or "")
                # 证据落地门已在 `_judge_once` 内逐票执行，此处不重复。
                # ★ 2026-10-20 义务账本门：账本已注入、模型也给了引用，但这条引用
                #   在账本里查不到（填了 none / 编号不存在）⇒ 它评论的是蓝图并未
                #   要求的东西，属「证更强命题」型误判，放行。
                #   ⚠ 与 4.2 同口径：**只惩罚明确指认失败，不惩罚漏填**。
                if ledger_text and oid_raw:
                    from .obligation_ledger import normalize_oid
                    oid = normalize_oid(oid_raw, ledger)
                    if not oid:
                        unmapped.append({"id": cid, "answer": ans[:120],
                                         "ref": oid_raw[:40], "desc": desc})
                        kept.append(c)
                        self.record(ctx, "self_audit",
                                    "cid=%s 指认义务 %r 不在账本内 ⇒ 不淘汰，放行（%s）"
                                    % (cid, oid_raw[:20], desc[:120]))
                        continue
                else:
                    oid = ""
                rejected.append({"id": cid, "answer": ans[:120],
                                 "error_desc": desc, "error_at": at,
                                 "obligation_id": oid,
                                 "ledger_used": bool(ledger_text)})
                self.record(ctx, "self_audit",
                            "cid=%s 逻辑错误 ⇒ 淘汰：%s（位置：%s%s）"
                            % (cid, desc[:200], at or "-",
                               "；义务 %s" % oid if oid else ""))
            else:
                # 含「logic_error=true 但 desc 为空」⇒ 按纪律保留（无具体依据不淘汰）
                kept.append(c)
                self.record(ctx, "self_audit",
                            "cid=%s 通过（logic_error=%s）"
                            % (cid, d.get("logic_error")))

        if rejected:
            ctx.candidates = kept
            ctx.self_audit_rejected = list(
                getattr(ctx, "self_audit_rejected", None) or []) + rejected
            ctx.audit_reject_feedback = list(
                getattr(ctx, "audit_reject_feedback", None) or []) + [
                    "[自审逻辑错误] cid=%s: %s"
                    % (r["id"], r["error_desc"]) for r in rejected]

        ctx.self_audit = {
            "n": len(cands),
            "n_rejected": len(rejected),
            "n_unparsed": len(unparsed),
            "n_llm_fail": llm_fail,
            "n_retry": n_retry,
            "n_weak": n_weak,
            "votes": _votes,
            "n_unmapped": len(unmapped),
            "unmapped_ids": [r["id"] for r in unmapped],
            "n_ungrounded": len(ungrounded),
            "ungrounded_ids": [r["id"] for r in ungrounded],
            "ledger_used": bool(ledger_text),
            "rejected_ids": [r["id"] for r in rejected],
        }
        if ungrounded:
            ctx.self_audit_ungrounded = list(
                getattr(ctx, "self_audit_ungrounded", None) or []) + ungrounded
        if unmapped:
            ctx.self_audit_unmapped = list(
                getattr(ctx, "self_audit_unmapped", None) or []) + unmapped
        self.record(
            ctx, "self_audit",
            "4.1 自审：%d 候选 ⇒ 淘汰 %d / 保留 %d"
            "（不可解析 %d、重问 %d、LLM 故障 %d、义务指认失败放行 %d、"
            "证据无法落地 %d）"
            % (len(cands), len(rejected), len(kept), len(unparsed), n_retry,
               llm_fail, len(unmapped), len(ungrounded)))

        # 淘汰后池空 ⇒ 汇总错误点交下游（与 4.0/旧链同一兜底，纯留痕）
        try:
            from .orchestrator__verify import _pool_empty_fallback as _pef
            _pef(self, ctx, "4.1_self_audit")
        except Exception as _e_pef:  # noqa: BLE001
            # ★ 2026-10-15（审核修复）：不再无痕 pass —— 记错误信息（可归因）
            self.record(ctx, "self_audit",
                        "pool_empty_fallback 失败（不阻断）: %s: %s"
                        % (type(_e_pef).__name__, str(_e_pef)[:100]))
    except Exception as _e:  # noqa: BLE001  增益环节绝不阻断主链
        self.record(ctx, "self_audit",
                    f"4.1 自审异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
