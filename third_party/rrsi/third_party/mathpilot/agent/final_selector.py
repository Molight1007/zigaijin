# -*- coding: utf-8 -*-
"""终答五层选择（阶段二-2，2026-10-02）
========================================

背景（用户 2026-10-02 意见 + docs/主链重构方案_2026-10-02.md §三）
------------------------------------------------------------------
用户口径：
  · 「投票应在**最后**，不要一开始就投票」；
  · 「子目标**先按正确率选**，不能靠长度选」；
  · 「把 n 个答案**都给大模型**让它来判断」。

本模块把"终答从 `ctx.candidates` 里选出来"这件事，从「**一开始就投票**」
（`formatter._rank_key` 票数优先 / `objective_majority_vote`）改为**五层顺序**：

    ② **多答案判断**：先问模型"这题是否本来就可能有多个答案"？
       判"可能多个" ⇒ **直接输出多个终答，③④⑤ 全跳过**
       （否则会把"本该输出集合"的题压成单值 ⇒ 直接判错）。
    ③ **客观验证**：只可能一个 ⇒ 用确定性校验淘汰
       （`ctx._falsified_answers` 已证伪答案 + `AnswerOracle` 可解析性）；
       幸存者恰为 1 个 ⇒ 直接采用，**不进 ④⑤**。
    ④ **模型对比**：还剩多个 ⇒ 把所有幸存候选的**推理原文**一次性给模型对比裁决
       （**子目标保留候选作为旁证一起给**）。
    ⑤ **投票兜底**：模型也判不出 ⇒ 归一化后做频次投票。

设计纪律
--------
· 全部逻辑集中在 `select_final_answer()`：`formatter` 与 `orchestrator` 的
  「候选池 → 终答」入口**共用同一个函数**（本项目教训：只改一处 = 没改）。
· **任何异常/判不出 ⇒ `skipped=True`**，调用方**回退既有逻辑**，绝不阻断主链
  （评测判空 = 0 分）。
· 不修改 `formatter._rank_key`（其键序被单测锁定）；本模块的"停用票数优先"
  由**入口短路**实现（开关 ON 时根本不走 `_rank_key` 排序）。

2026-10-07 修复（两条阻断级，smoke10 2026-10-06 实测驱动）
--------------------------------------------------------
2.1 决策记录与最终产出「不同源」
    实测题 official112-002：`pick_diag.picked` = `证明这个最小值就是所求的 $T$，
    并验证其最优性`（一个**计划句**），而落盘被评分的 `predicted` =
    `\\boxed{\\dfrac{\\sqrt{5}-1}{2}}`。分叉点**不在本模块**，而在 orchestrator
    6.5 审核闸门：`formatter.run` 写 `_pick_diag` 之后，闸门调
    `_set_final_response(ctx, _next.answer, "gate_next")`（→ 本模块 `adopt_single`）
    **改写 `ctx.final_response`**，却不动 `_pick_diag`。
    真值 = 提交串（`ctx.final_response`），故令记录侧向它对齐：
    · `adopt_single()` 每次采纳都把 `_pick_diag.picked` 同步为**逐字提交值**
      （开关 `final_answer_single_source`，默认 True）；
    · `formatter` 末尾的同步也改为**不截断**（原 `[:60]` 使 006 的 picked 与
      predicted 不逐字一致）；
    · 原"决策值"移入 `picked_decision`，来源写入 `final_submit_source`。
    ⚠ `_final_answer_postprocess`（orchestrator:2355）也会在 formatter 之后改写
      答案且**不经过本模块** ⇒ 该路径需 orchestrator 侧一行同步（已在回报中说明；
      smoke10 10 题该路径触发 0 次，故未在本批落地）。

2.2 未验证候选仍参与投票并胜出
    实测题 official112-007：3 个候选，`verdicts` 只有 1 条（对应 id=0 的计划句）；
    候选 id=2/3 **无任何验证记录**（导出里 `confidence/correct_votes/total_votes`
    全为 `None`），却进入 ⑤ 投票；1:1 平票后由**推理长度**裁决，选中 `\\boxed{0}`
    （gold=1）；且 `picked` 不在任何 cluster 内。
    修法（开关 `final_select_verify_aware`，默认 True）：
    · 依 `ctx.verdicts[*].answer` 建**答案级验证索引**（Verdict.answer 是聚合时
      从候选复制的）⇒ 判每个候选"有/无验证记录"；
    · 有已验证候选时，④模型对比 / ⑤投票**只在已验证候选里选**（跳过未验证者）；
      全部未验证时保留全体，但**显式留痕** `all_unverified` 且不得仅凭未验证胜出；
    · ⑤ 投票并列收尾**不再用推理长度**，改为 (已验证, 票数, 答案形态, 更短答案)；
    · `record_pick_cluster()` 把 picked 归入哪个 cluster 写入 `_pick_diag`
      （未归入则写明原因）。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from prompts.final_select import (
    MULTI_ANSWER_JUDGE_SYSTEM, MULTI_ANSWER_JUDGE_USER,
)
logger = logging.getLogger("MathPilot")

# 2026-10-08 批 F1：「答案归一/等价/脏答案 + LLM 输出解析」纯函数簇已搬到
# final_selector__parse.py。原位 re-export 保外部导入与内部裸调用契约。
# ⚠ 勿按"未使用"清理。
from .final_selector__parse import (  # noqa: F401
    _REFUSAL_RE, _norm, _items_equiv, _is_bad_answer,
    _parse_multiple, _parse_compare_pick)

# ④ 注入口径：单候选推理原文上限 / 最多注入候选数 / 子目标旁证总上限
_MAX_REASONING_CHARS = 4000
_MAX_CANDIDATES_IN_PROMPT = 6


class _Cand:
    """轻量候选适配器（供 `AnswerOracle` 消费；仅需 answer/reasoning）。"""

    __slots__ = ("answer", "reasoning")

    def __init__(self, answer: str, reasoning: str = ""):
        self.answer = answer
        self.reasoning = reasoning


class SynthCandidate:
    """合成候选（五层选出的答案若**匹配不到原候选对象**时用它返回）。

    供 `formatter._pick_best` / `orchestrator._pick_best_from_candidates` 的
    调用方读取 `.answer` / `.reasoning` / `.confidence` —— 字段与 `Candidate` 对齐
    （但刻意不依赖 `agent.base.Candidate`，避免循环 import）。
    """

    def __init__(self, answer: str, reasoning: str = "",
                 origin: str = "final_select"):
        self.id = -1
        self.answer = answer
        self.reasoning = reasoning
        self.revised = False
        self.confidence = 0.0
        self.correct_votes = 0
        self.total_votes = 0
        self.origin = origin


# ----------------------------------------------------------------------
# 2026-10-07（问题 2.2）：候选「验证状态」索引
# ----------------------------------------------------------------------
def _verification_index(ctx) -> dict:
    """建 **答案级** 验证索引：``{归一化答案: {"correct_votes","total_votes","n"}}``。

    为什么用"答案"而不是"候选 id"做键：
      · `Verdict.id` 是 `_verdicts_from_ver_result` 写入的**列表下标**（idx），
        不是 `Candidate.id`；候选池一旦增删（腾位 / 6.5 换候选），二者**不再对齐**
        （实测 000：candidate.id=[7,11,12,13,14,15] 而 verdict.id=[0..5]）
        ⇒ 按 id 匹配会**全部落空**。
      · 而 `Verdict.answer` 是聚合时从**当时的候选**复制的答案串
        （`orchestrator._verdicts_from_ver_result`：`answer=candidate.answer`），
        与候选池是否重排无关 ⇒ 用答案做键稳健。
    未出现在本索引里的答案 = **没有任何验证记录**（正是 007 的 `\boxed{0}`）。
    任何异常返回空 dict（绝不阻断选答）。
    """
    out: dict = {}
    try:
        for v in (getattr(ctx, "verdicts", None) or []):
            a = str(getattr(v, "answer", "") or "")
            k = _norm(a)
            if not k:
                continue
            cv = int(getattr(v, "correct_votes", 0) or 0)
            tv = int(getattr(v, "total_votes", 0) or 0)
            e = out.setdefault(k, {"correct_votes": 0, "total_votes": 0, "n": 0})
            e["correct_votes"] += cv
            e["total_votes"] += tv
            e["n"] += 1
    except Exception:  # noqa: BLE001
        return {}
    return out


def _verify_status(answer: str, vindex: dict) -> Tuple[str, dict]:
    """返回 ``(status, info)``；status ∈ {"verified","unverified"}。

    · `unverified` 判据两类（都属"无**有效**验证记录"）：
        ① 答案完全不在验证索引里（该候选未进入 verifier 投票）；
        ② 有记录但 `total_votes == 0`（全弃权/基础设施故障 —— 与
           `formatter._passes_confidence_floor` 的"故障不是反证"口径一致，
           但**同样不构成可信证据**）。
    · `verified` ⇒ info 带正确的票/总票/记录条数。
    """
    k = _norm(answer)
    e = vindex.get(k) if isinstance(vindex, dict) else None
    if e is None:
        return "unverified", {"reason": "无验证记录（该答案未进入 verifier 投票）"}
    if int(e.get("total_votes", 0) or 0) <= 0:
        return "unverified", {
            "reason": "验证记录 total_votes=0（全弃权/故障，无有效验证）",
            "n_verdicts": int(e.get("n", 0) or 0)}
    return "verified", {"correct_votes": int(e.get("correct_votes", 0) or 0),
                        "total_votes": int(e.get("total_votes", 0) or 0),
                        "n_verdicts": int(e.get("n", 0) or 0)}


# ----------------------------------------------------------------------
# 五层
# ----------------------------------------------------------------------
def _llm_text(llm: Callable, ctx, system: str, user: str,
              max_tokens: int) -> str:
    """一次 temperature=0 的短调用；失败返回空串（不抛）。"""
    try:
        resp = llm(ctx, [{"role": "system", "content": system},
                         {"role": "user", "content": user}], 0.0, max_tokens)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[final_selector] LLM 调用异常: %s: %s",
                     type(exc).__name__, exc)
        return ""
    if resp is None:
        return ""
    return str(resp)


def _vote(survivors: List[Tuple[str, str, Any]],
          verified=None, verify_aware: bool = False,
          machine_ok=None
          ) -> Tuple[str, Any, dict]:
    """步⑤：归一化频次投票。

    ★ 2026-10-07 修（问题 2.2）：`verify_aware=True` 时并列收尾**不再用推理长度**。
      原实现 `top = max(best, key=lambda t: (len(t[1]), -len(t[0])))` 在票数并列时
      取"推理更长者" —— 与正确性无因果，且正是实测 007「1:1 平票、**无任何验证
      记录**的 `\\boxed{0}` 仅凭更长推理胜出（gold=1）」的直接机制。
      新收尾 = (是否已验证, 该结论的独立支撑数, 答案形态分, 更短答案)，**全序确定**。
    `verify_aware=False` ⇒ 逐字保持改动前行为（A/B 对照）。

    ★★ 2026-10-22（检测效果传导 · 用户要求「让检测环节的结论影响选择」）：
      `machine_ok` = 「4.4 逐式复算全一致」的**答案集合**（`verify_math == "pass"`）。
      它插在**支持数之后**，只影响"票数并列"的取舍 —— 因为
      「算术全对」只说明推理没写错等式，**不等于答案正确**，不该压过票数。
      排在形态分/长度之前，是为了替代"更短/更像答案"这类与正确性无因果的兜底。
    """
    groups: dict = {}
    for a, r, c in survivors:
        groups.setdefault(_norm(a), []).append((a, r, c))
    diag = {"n_distinct": len(groups),
            "dist": {k[:40]: len(v) for k, v in groups.items()}}
    if not verify_aware:
        best = max(groups.values(),
                   key=lambda g: (len(g), max(len(r) for _, r, _ in g)))
        top = max(best, key=lambda t: (len(t[1]), -len(t[0])))
        diag["tie_break"] = "reasoning_length(legacy)"
        return top[0], top[2], diag
    _ver = verified if isinstance(verified, (set, frozenset, dict)) else set()
    _mok = machine_ok if isinstance(machine_ok, (set, frozenset, dict)) else set()

    def _key(t):
        a = t[0]
        n = _norm(a)
        return (1 if n in _ver else 0,
                len(groups.get(n, ())),
                1 if n in _mok else 0,
                _answer_form_score(a),
                -len(a))

    _flat = [t for g in groups.values() for t in g]
    top = max(_flat, key=_key)
    diag["tie_break"] = ("verified_then_support_then_machine_ok_then_form_then_shorter"
                         if _mok else "verified_then_support_then_form_then_shorter")
    diag["n_machine_ok"] = len(_mok)
    return top[0], top[2], diag


# ★ 2026-10-15（审核 + Q1 无用功清理）：原 `_build_sg_evidence`
#   （"子目标保留候选 → 步④旁证块"）已删除 —— 其消费场景「步④模型对比」
#   已按 2026-10-14 五道链定案整块删除（见 _select_from_survivors docstring），
#   函数属历史决策残留、全仓无调用。
def select_final_answer(ctx, candidates, llm: Callable,
                        record: Optional[Callable] = None) -> dict:
    """五层选择终答。返回::

        {"answer": str,           # 选出的终答（空串 = 未能选出 ⇒ 调用方回退）
         "branch": str,           # multi_answer / objective_single / model_compare / vote / <未走完>
         "steps": [str],          # 实际执行到的层（顺序证据）
         "diag": dict,            # 观测数据
         "cand": object | None}   # 命中的原始候选对象（保持 confidence 等字段）

    ★ `skipped=True` ⇒ 调用方**必须回退既有逻辑**（不阻断主链）。
    """
    diag: dict = {"branch": "", "steps": []}
    steps: List[str] = diag["steps"]

    def _rec(content: str, **extra):
        if record is None:
            return
        try:
            record(ctx, "final_select", content, **extra)
        except Exception:  # noqa: BLE001
            pass

    try:
        # ---- 收集可用候选（归一化去重，保留推理更长者）----
        seen: dict = {}
        for c in (candidates or []):
            a = str(getattr(c, "answer", "") or "").strip()
            r = str(getattr(c, "reasoning", "") or "")
            if _is_bad_answer(a):
                continue
            k = _norm(a)
            if k not in seen or len(r) > len(seen[k][1]):
                seen[k] = (a, r, c)
        uniq = list(seen.values())
        diag["n_in"] = len(list(candidates or []))
        diag["n_distinct"] = len(uniq)
        if not uniq:
            diag["branch"] = "no_usable_candidate"
            _rec("五层选择：无可用候选（空/拒绝词），回退既有逻辑")
            return {"answer": "", "branch": "no_usable_candidate",
                    "skipped": True, "steps": steps, "diag": diag, "cand": None}

        # ★ 2026-10-07（问题 2.2）：**验证状态**索引。
        #   依 `ctx.verdicts[*].answer` 建"答案 → 票数"表；不在表里的答案=无验证记录。
        #   实测 007 的 `\boxed{0}`（gold=1）正是"无任何验证记录却参与投票并胜出"。
        verify_aware = _verify_aware_enabled()
        vindex = _verification_index(ctx) if verify_aware else {}
        _vstat: dict = {}
        for a, _r, _c in uniq:
            _vstat[_norm(a)] = _verify_status(a, vindex) if verify_aware \
                else ("verified", {})
        _n_unv = sum(1 for s, _i in _vstat.values() if s == "unverified")
        diag["verification"] = {
            "enabled": verify_aware,
            "n_in": len(uniq),
            "n_verified": len(uniq) - _n_unv,
            "n_unverified": _n_unv,
            "unverified": [{"answer": a[:60],
                            "reason": _vstat[_norm(a)][1].get("reason", "")}
                           for a, _r, _c in uniq
                           if _vstat[_norm(a)][0] == "unverified"][:8],
        }
        if verify_aware and _n_unv:
            _rec(f"五层选择：{_n_unv}/{len(uniq)} 个候选**无验证记录**"
                 f"（有已验证候选时不参与胜出）")

        problem = str(getattr(ctx, "problem", "") or "")

        # ================= ② 多答案判断 =================
        # ★ 2026-10-14（五道链定案·单一来源）：优先复用 4.5_vote 的结果
        #   （`ctx.vote_multi_answer`）——「先判断会不会是多个答案，再投票」
        #   的判定在投票阶段已完成，此处不再重复问一次 LLM。
        steps.append("2_multi_answer")
        multi = getattr(ctx, "vote_multi_answer", None)
        if multi is None:
            answers_block = "\n".join("- " + a for a, _, _ in
                                      uniq[:_MAX_CANDIDATES_IN_PROMPT])
            # 2026-10-02 DeepSeek 适配：原 768 ⇒ 8192。原值基于「判一条 JSON + 思维块被
            # 抑制」的假设，对 reasoning 模型必然截断（reasoning 先吃满预算、正文为空）。
            raw2 = _llm_text(llm, ctx, MULTI_ANSWER_JUDGE_SYSTEM,
                             MULTI_ANSWER_JUDGE_USER.format(
                                 problem=problem, answers=answers_block), 8192)
            multi = _parse_multiple(raw2)
            diag["multi_answer_source"] = "select_llm"
        else:
            diag["multi_answer_source"] = "vote_gate"
            _rec(f"五层选择②：复用 4.5_vote 多答案判定 ⇒ multiple={multi}")
        diag["multi_answer"] = multi
        if multi is True:
            # 2026-10-04 修（fix-a1）：原实现直接 `", ".join(a for a, _, _ in uniq)`，
            # 用的是**原始答案串**——而 uniq 已按 `_norm(a)` 去重，导致
            # 「同一答案的不同写法」「集合相同但顺序不同」被重复输出。
            # 实测候选 ['0,2026','2026','0, 2026','2026,0'] ⇒ 旧输出
            # '0,2026, 2026, 2026,0'（5 项）；判分器 _split_multi 只按 ',' 拆且
            # 不去重 ⇒ 拆出 5 项 vs gold 2 项 ⇒ 判 False。
            # 修法：拆成单项、按 `_norm(单项)` 去重（保留首次出现的原始写法）、
            # 用 ASCII ", " 连接（对齐判分器口径，**不可用顿号**）。不设条数上限。
            _items: List[str] = []
            for a, _, _ in uniq:
                for piece in re.split(r"[,，]", a):
                    p = piece.strip()
                    if not p:
                        continue
                    # ★ 2026-10-04（D9-a）：去重从 `_norm` 字符串比较升级为
                    #   `_items_equiv`（SymPy 等价优先）⇒ `1/2` 与 `0.5`、
                    #   `\frac12` 与 `0.5` 这类**同一答案的不同写法**不再被当成
                    #   两项输出（判分器按 `,` 拆项与 gold 比对，多输出即必错）。
                    if any(_items_equiv(p, q) for q in _items):
                        continue
                    _items.append(p)
            diag["multi_items"] = _items
            ans = ", ".join(_items) if _items else ", ".join(a for a, _, _ in uniq)
            diag["branch"] = "multi_answer"
            _rec(f"五层选择②：判为多答案题 ⇒ 展平去重后输出 {len(_items)} 项终答，"
                 f"跳过 ⑤：{ans[:80]}")
            return {"answer": ans, "branch": "multi_answer", "skipped": False,
                    "steps": steps, "diag": diag, "cand": None}
        _rec("五层选择②：判为单一答案 ⇒ 进入投票（⑤）"
             + ("" if multi is False else "（判不出，按单一处理）"))

        return _select_from_survivors(
            ctx, uniq, problem, llm, steps, diag, _rec,
            verify_aware, _vstat)
    except Exception as exc:  # noqa: BLE001  五层任何异常都不得阻断主链
        logger.warning("[final_selector] 五层选择异常（回退既有逻辑）: %s: %s",
                       type(exc).__name__, exc)
        diag["branch"] = "exception"
        diag["error"] = f"{type(exc).__name__}: {exc}"
        return {"answer": "", "branch": "exception", "skipped": True,
                "steps": steps, "diag": diag, "cand": None}



def _select_from_survivors(ctx, uniq, problem, llm, steps, diag, _rec,
                           verify_aware, _vstat) -> dict:
    """五层 ⑤ 投票（2026-10-14 五道链定案：③客观验证 / ④模型对比已删）。

    用户口径：「先判断会不会是多个答案，再投票选择答案」——
      · ②多答案判断在调用方完成（优先复用 4.5_vote 的 ctx.vote_multi_answer）；
      · 客观验证（原③）职责已由五道链前四道承担（自审/证伪/Lean/计算）；
      · 模型对比（原④）按用户决策删除；
      · 本段 = **归一化频次投票**（last-resort 兜底）。
    verify-aware 过滤（2026-10-07 问题 2.2 修复）保留：有已验证候选时不选
    未验证者（实测 007 未验证 `\boxed{0}` 曾靠更长推理胜出）。
    """
    # ★ 2026-10-07（问题 2.2）：投票池 = **已验证候选**（"跳过"未验证者）。
    #   · 有 ≥1 个已验证候选 ⇒ 池 = 已验证者（未验证者记入 diag 但不参与胜出）；
    #   · 全部未验证 ⇒ 保留全体（否则"必得 0 分"），但显式留痕 all_unverified。
    pool = uniq
    _pool_diag = {"n_surv": len(uniq), "restricted": False, "all_unverified": False,
                  "excluded_unverified": []}
    if verify_aware:
        _ver_pool = [t for t in uniq if _vstat.get(_norm(t[0]), ("verified",))[0]
                     == "verified"]
        if _ver_pool and len(_ver_pool) < len(uniq):
            _pool_diag["restricted"] = True
            _pool_diag["excluded_unverified"] = [
                {"answer": t[0][:60], "reason": _vstat.get(
                    _norm(t[0]), ("unverified", {}))[1].get("reason", "")}
                for t in uniq if t not in _ver_pool][:8]
            pool = _ver_pool
            _rec(f"五层选择：⑤ 仅在 {len(_ver_pool)} 个**已验证**候选里选，"
                 f"跳过 {len(uniq) - len(_ver_pool)} 个无验证记录者")
        elif not _ver_pool:
            _pool_diag["all_unverified"] = True
            _rec("五层选择：⚠ 全部候选**均无有效验证记录** ⇒ 保留全体但不得"
                 "仅凭未验证胜出（已留痕 all_unverified）")
    diag["pool"] = _pool_diag
    if len(pool) == 1:
        a, r, c = pool[0]
        diag["branch"] = "verified_only_single"
        _rec(f"五层选择：验证过滤后仅剩 1 个候选 ⇒ 直接采用：{a[:80]}")
        return {"answer": a, "branch": "verified_only_single", "skipped": False,
                "steps": steps, "diag": diag, "cand": c}

    # ================= ⑤ 投票（末层兜底） =================
    steps.append("5_vote")
    _verified_norms = {_norm(t[0]) for t in pool
                       if _vstat.get(_norm(t[0]), ("verified",))[0] == "verified"}
    # ★ 2026-10-22：把 4.4 的机器核验结论传进投票 —— 只作**票数并列**时的次级依据。
    #   之前 `verify_math` 写在候选上却没人读，等于检测环节的结论到不了选择层。
    _machine_ok: set = set()
    if _machine_tiebreak_enabled():
        for _a, _r, _c in pool:
            if str(getattr(_c, "verify_math", "") or "") == "pass":
                _machine_ok.add(_norm(_a))
    a, c, v_diag = _vote(pool, verified=_verified_norms,
                         verify_aware=verify_aware,
                         machine_ok=_machine_ok)
    diag["vote"] = v_diag
    diag["branch"] = "vote"
    _rec(f"五层选择⑤：投票兜底选中：{a[:80]}"
         + ("（⚠ 全体候选无验证记录）"
            if _pool_diag.get("all_unverified") else ""))
    return {"answer": a, "branch": "vote", "skipped": False,
            "steps": steps, "diag": diag, "cand": c}

def _switch_enabled() -> bool:
    from agent.switch_registry import get_bool as _swb
    try:
        return bool(_swb("enable_final_answer_selection"))
    except Exception:  # noqa: BLE001
        return False


def _machine_tiebreak_enabled() -> bool:
    """读「4.4 机器核验作投票平票依据」开关（``five_gate_verify_tiebreak``，默认 True）。

    ★ 2026-10-22：4.4 一直把 ``verify_math``（逐式复算是否全一致）写在候选上，
      但**没有任何消费方**阅读它 ⇒ 检测环节的结论到不了选择层。
      开启后：票数并列时优先选「算术已核」的答案；OFF ⇒ 回到只按形态分/长度。
    """
    from agent.switch_registry import get_bool as _swb
    try:
        return bool(_swb("five_gate_verify_tiebreak"))
    except Exception:  # noqa: BLE001  注册表不可用 ⇒ 新语义
        return True


def _single_source_enabled() -> bool:
    """问题 2.1 开关：把 `_pick_diag` 同步为**逐字提交值**（默认 True）。

    OFF ⇒ 完全回退改动前（`picked` 仍是决策值/截断值），便于 A/B。
    """
    from agent.switch_registry import get_bool as _swb
    try:
        return bool(_swb("final_answer_single_source"))
    except Exception:  # noqa: BLE001
        return True


def _verify_aware_enabled() -> bool:
    """问题 2.2 开关：选择机制识别"未验证候选"（默认 True）。

    OFF ⇒ 回退改动前（未验证候选照旧参与 ④⑤ 且投票并列按推理长度裁决）。
    """
    from agent.switch_registry import get_bool as _swb
    try:
        return bool(_swb("final_select_verify_aware"))
    except Exception:  # noqa: BLE001
        return True


def _answer_form_score(ans) -> int:
    """答案形态分（复用 formatter 的**唯一**实现，避免第二套口径）。

    惰性 import；不可用时退化为 0（不影响确定性收尾的其余键）。
    """
    try:
        from agent.formatter import _answer_form_score as _afs
    except ImportError:
        return 0
    try:
        return int(_afs(ans))
    except Exception:  # noqa: BLE001
        return 0


def _sync_pick_diag_final(ctx, value: str, source: str = "") -> None:
    """:把 `_pick_diag` 对齐到**最终提交值**（问题 2.1 的修法本体）。

    真值 = 提交串 `ctx.final_response`（`run_eval` 把它当作 `predicted` 评分）。
    本函数让记录侧向它对齐：
      · `picked`        = **逐字提交值**（不再截断）
      · `picked_decision`= 首次同步前的值（= 五层/formatter 的原始**决策值**，保追溯）
      · `final_submit`  = 逐字提交值（冗余键，语义自明）
      · `final_submit_source` / `final_submit_seq` = 谁在什么顺序上定的
    ⚠ **只在 `_pick_diag` 已是 dict 时更新**（不新建）：`adopt_single` 也会在
      formatter **之前**被零票兜底路径调用，此时新建 `_pick_diag` 会干扰 formatter
      「零票兜底 / formatter_pick_best」的分支判定；只在 dict 已存在时同步则无副作用。
    任何异常一律忽略（埋点绝不能阻断主链）。
    """
    try:
        v = str(value or "")
        if not v:
            return
        pd = getattr(ctx, "_pick_diag", None)
        if not isinstance(pd, dict):
            return
        if "picked_decision" not in pd:
            pd["picked_decision"] = pd.get("picked")
        pd["picked"] = v
        pd["final_submit"] = v
        pd["final_submit_source"] = str(source or "")
        pd["final_submit_seq"] = int(pd.get("final_submit_seq", 0) or 0) + 1
        # 同步后顺带刷新 cluster 归属（问题 2.2 留痕：picked 能追溯到簇）。
        try:
            if _verify_aware_enabled():
                record_pick_cluster(ctx)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def sync_pick_diag_to_final(ctx, source: str = "final_sink") -> None:
    """**公开入口**（问题 2.1）：把 `_pick_diag` 同步为 `ctx.final_response` 的逐字值。

    供 **orchestrator 的后处理路径**接入（`_final_answer_postprocess` 在 formatter
    之后改写答案、且不经过 `adopt_single` ⇒ 需显式同步一行）。开关
    `final_answer_single_source` OFF ⇒ 什么都不做（完全回退）。任何异常一律忽略。
    """
    if not _single_source_enabled():
        return
    try:
        _sync_pick_diag_final(
            ctx, str(getattr(ctx, "final_response", "") or ""), str(source))
    except Exception:  # noqa: BLE001
        pass


def record_pick_cluster(ctx) -> None:
    """问题 2.2（留痕）：把 `_pick_diag["picked"]` 归入的 cluster 写进记录。

    · 命中 ⇒ `picked_cluster = {answer,size,confidence,correct_votes,total_votes}`
    · 未命中 / 无 cluster 数据 ⇒ `picked_cluster = None` 且写明 `picked_cluster_missing`
      （原因）。这正是"picked 必须能追溯到某个 cluster，或明确记录未归入的原因"。
    cluster 来源 = `ctx._cluster_data`（verifier 聚类，formatter 已导出 divergent_clusters）。
    任何异常一律忽略。
    """
    try:
        pd = getattr(ctx, "_pick_diag", None)
        if not isinstance(pd, dict):
            return
        ans = str(pd.get("picked") or "")
        if not ans:
            return
        cd = getattr(ctx, "_cluster_data", None)
        if not isinstance(cd, list) or not cd:
            pd["picked_cluster"] = None
            pd["picked_cluster_missing"] = "无 cluster 数据（verifier 未聚类或未产出）"
            return
        for cl in cd:
            an = str(getattr(cl, "answer_norm", "") or "").strip()
            if not an:
                continue
            if _norm(an) == _norm(ans) or _items_equiv(an, ans):
                pd["picked_cluster"] = {
                    "answer": an[:60],
                    "size": int(getattr(cl, "size", 0) or 0),
                    "confidence": round(
                        float(getattr(cl, "confidence", 0.0) or 0.0), 3),
                    "correct_votes": int(getattr(cl, "vote_correct", 0) or 0),
                    "total_votes": int(getattr(cl, "vote_total", 0) or 0),
                }
                pd.pop("picked_cluster_missing", None)
                return
        pd["picked_cluster"] = None
        pd["picked_cluster_missing"] = ("该答案未归入任何 cluster"
                                        "（无等价簇；通常=该候选无验证记录）")
    except Exception:  # noqa: BLE001
        pass


def adopt_single(ctx, answer: str, reason: str = "",
                 record: Optional[Callable] = None) -> dict:
    """**单答案快速采纳**（并入五层唯一出口，**不调 LLM**）。

    team-lead 2026-10-02 Q4 裁定：「让 `_set_final_response(direct_answer)` 也经过五层；
    但若传入的只有一个答案 ⇒ 五层直接采用、**不额外调 LLM**（避免兜底路径变慢/变不稳）」。

    用于 `orchestrator._set_final_response` 的**池外直答**路径（零票兜底 / 6.5 重做循环）：
    那里输入只有一个答案 ⇒ 无需 ②多答案判断 / ③客观验证 / ④模型对比 / ⑤投票，
    直接落到"唯一候选 ⇒ 采用"。仍写 `ctx._final_select_exit` / trace ⇒
    「这个终答是谁定的」**可归因**（这也是它存在的意义——消灭旁路）。

    开关 OFF ⇒ `skipped=True` ⇒ 调用方按原逻辑写入（**完全回退**）。
    """
    a = str(answer or "").strip()
    # ★ 2026-10-07（问题 2.1）：本函数是**池外直答 / 6.5 闸门换候选**的唯一入口
    #   （`Orchestrator._set_final_response` → 本函数）。它在 `formatter` **之后**
    #   改写最终提交值 ⇒ 必须把 `_pick_diag` 同步为**逐字提交值**，否则决策记录
    #   与被评分的 `predicted` 不同源（实测 002：记录是计划句、提交是 `\boxed{...}`）。
    #   `_sync_pick_diag_final` 只在 `_pick_diag` 已是 dict 时更新（本调用可能在
    #   formatter 之前发生 ⇒ 不新建、不干扰 formatter 的分支判定）。
    _ss = _single_source_enabled()
    if _ss:
        _sync_pick_diag_final(ctx, a, "single_adopt:" + str(reason or ""))
    if not _switch_enabled():
        return {"answer": a, "branch": "disabled", "skipped": True,
                "disabled": True, "steps": [], "diag": {}, "cand": None}
    try:
        setattr(ctx, "_final_select_exit", "single_adopt:" + str(reason or ""))
    except Exception:  # noqa: BLE001
        pass
    # ⚠ 只在五层**没跑过**（无 _final_select_diag）时补一个，避免覆盖全量五层的归因。
    try:
        if not isinstance(getattr(ctx, "_final_select_diag", None), dict):
            ctx._final_select_diag = {
                "branch": "single_adopt", "steps": ["single_adopt"],
                "reason": "池外直答：单答案 ⇒ 五层直接采用（不调 LLM）"}
    except Exception:  # noqa: BLE001
        pass
    try:
        _rec_trace = {
            "agent": "FinalSelector", "step": "final_select",
            "content": "五层选择(单答案快速采纳)：唯一答案直接采用，不调 LLM",
            "branch": "single_adopt", "reason": str(reason or "")}
        if _ss:
            _rec_trace["content"] += f"（最终提交的就是: {a[:120]}）"
            _rec_trace["final_submit"] = a
        ctx.trace.append(_rec_trace)
    except Exception:  # noqa: BLE001
        pass
    return {"answer": a, "branch": "single_adopt", "skipped": False,
            "steps": ["single_adopt"],
            "diag": {"branch": "single_adopt", "steps": ["single_adopt"]},
            "cand": None}


def run_cached(ctx, llm: Callable, record: Optional[Callable] = None) -> dict:
    """带 `ctx` 缓存的五层入口（**formatter 与 orchestrator 共用同一决策**）。

    为什么必须缓存：终答选取在两条路径上都会被触发（`formatter._pick_best` 主路径、
    `orchestrator._pick_best_from_candidates` 超时/零票/异常兜底）。若各跑一遍，
    既有重复 LLM 成本，更糟的是**两次决策可能不一致**（同一 ctx 选出两个终答）。
    故第一次跑完把结果挂 `ctx._final_select_cache`，后续直接复用。

    开关 `enable_final_answer_selection` OFF ⇒ 返回 `disabled`（**不缓存**，便于 A/B）。
    """
    if not _switch_enabled():
        return {"answer": "", "branch": "disabled", "skipped": True,
                "disabled": True, "steps": [], "diag": {}, "cand": None}
    cached = getattr(ctx, "_final_select_cache", None)
    if isinstance(cached, dict):
        return cached
    res = select_final_answer(ctx, getattr(ctx, "candidates", None), llm, record)
    try:
        setattr(ctx, "_final_select_cache", res)
    except Exception:  # noqa: BLE001
        pass
    return res
