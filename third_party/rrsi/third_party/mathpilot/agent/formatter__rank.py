"""候选择优指标（2026-10-08 从 agent/formatter.py 逐字搬出）。

簇②：答案「形态分」/ 统一排序键 / 独立复现数（support）/ 可信度门槛 /
真枚举判定。均为纯函数（无副作用、异常一律兜底），供 `_pick_best*` 等调用。

formatter.py 原位 re-export 全部 7 个符号 ⇒ 外部 import 契约不变：
  · `_rank_key` / `_attach_support` 被 `agent/orchestrator.py` 导入；
  · `_answer_form_score` 被 `agent/final_selector.py` 导入；
  · 其余被 `tests/test_answer_pick_support_0930.py` 导入。

依赖：`_answer_support` 用到 `_looks_like_non_answer` —— 经**顶层**从
`formatter__nonanswer` 导入（非循环，且**不新增函数内语句**，保证搬出函数体逐字不变）。
"""
from __future__ import annotations

import re

from .formatter__nonanswer import _looks_like_non_answer


def _answer_form_score(ans) -> int:
    """答案「形态分」—— 与正确性相关；**推理长度与正确性无因果关系**。

    ★ 2026-09-22（用户口径：「制定正确的逻辑，删去原本不合理的逻辑」）：
    本项目已三次修复「按推理长度选答案」（09-16 枚举分支、09-17 簇内分支、
    09-20 长答案重提取），但**仍有三处遗漏**（本文件 259 / 510 / 876 行）。
    实测 011：票数与置信度**完全并列**的两个候选，交给「推理更长者」裁决
    （18015 vs 6938 字符），选中错答 —— 正确候选就在池里。
    ⇒ 判据改为「答案形态」：像答案的（boxed / 短数值表达式）优于散文。
    """
    s = str(ans or "").strip()
    if not s:
        return 0
    if "\\boxed" in s:
        return 3
    if len(s) <= 40 and not re.search(r"[\u4e00-\u9fff]{6,}", s):
        return 2
    if len(s) <= 40:
        return 1
    return 0


def _rank_key(c):
    """候选择优的**统一**排序键。

    ★ 2026-09-30（截图 #9）扩为五项：
        ① `correct_votes`   —— 验证器给该候选投了几张对票；
        ② **`_support_n`**  —— ★新增：池里有几个**独立候选**写出了同一结论；
        ③ `confidence`      —— 正确票 / 有效票；
        ④ `_answer_form_score` —— 「最像答案」（boxed > 短数学式 > 短文本 > 散文）；
        ⑤ 答案更短（仅确定性收尾）。

    ② 的插入位置有意在 `confidence` **之前**：用户口径「候选数量与逻辑通顺度
    都要是指标」，而 `support` 正是"候选数量"在单候选身上的投影。
    为什么它该压过 `confidence`：
      · 一条结论被 3 个独立候选复现 ⇒ 是 self-consistency 意义上的**共识**；
      · 单个候选自评 3/3 票 ⇒ 只是"验证器没看出它错"（本项目实测验证器误报率 56%）。
    历史上本项目已三次因"只看单一指标"丢过正确答案（003/011/014），故此处
    刻意**新增指标而非替换指标** —— 票数仍是第 1 键，`support` 只在票数相同时起作用，
    属零风险加性改动。

    ⚠ `_support_n` 由调用方（`_pick_best` 等）按**当前候选池**预先算好并挂到候选对象上；
    未挂时 `getattr` 兜底为 1（= "它自己就是一票"，不罚也不奖），
    这样所有既有调用点不传 support 时行为与改动前**完全一致**。

    全项目凡需在候选间择优处**一律复用本函数**，禁止再出现「只看推理长度」的
    局部实现（历史上已因此丢掉过正确答案：003 的 `\\boxed{0,2026}` 曾输给 0 票的
    `\\boxed{0,1,2,\\ldots}`；011 的正确候选曾输给更啰嗦的错答）。
    末项取「答案更短者优先」仅为**确定性收尾**（避免退化成插入序/随机），
    不代表长答案更差或更好。
    """
    ans = str(getattr(c, "answer", "") or "")
    return (
        int(getattr(c, "correct_votes", 0) or 0),
        int(getattr(c, "_support_n", 1) or 1),
        float(getattr(c, "confidence", 0.0) or 0.0),
        _answer_form_score(ans),
        -len(ans),
    )


def _attach_support(candidates) -> dict:
    """把「独立复现数」挂到每个候选的 `_support_n` 上，返回 support 表。

    ★ 2026-09-30（截图 #9）：「候选数量」作为指标的**落地方式**。
    调用点在 `_pick_best` 开头（择优之前），因此后续所有走 `_rank_key` 的
    排序（逐项判定优先、簇内择优、verdict 兜底、兜底池）都自动受益。

    刻意**不**改 `_rank_key` 的签名 —— 它是模块级纯函数、无池上下文，
    且全项目多处直接调用（含单元测试）。把"池"的知识留在调用方，
    是本次实现的最小侵入路径。

    ⚠ 幂等：重复调用只覆盖同值，无副作用。任何异常一律忽略（择优绝不能被埋点搞挂）。
    """
    try:
        _m = _answer_support(candidates)
        for _c in (candidates or []):
            try:
                setattr(_c, "_support_n", _support_of(_c, _m))
            except Exception:  # noqa: BLE001
                pass
        return _m
    except Exception:  # noqa: BLE001
        return {}


# ----------------------------------------------------------------------
# 2026-09-30（截图 #9）：**「最像答案」+ 证据量 + 可信度门槛**
# ----------------------------------------------------------------------
# 用户原话：「投票要选出里面最像答案的一个，候选数量与逻辑通顺度都要是指标，
#           同时要有可信度要求（太低＝全军覆没）」。
#
# 现状拆解（已核验）：
#   · 「最像答案」⇒ `_answer_form_score`（boxed 3 分 / 短数学式 2 分 / 短文本 1 分
#     / 散文 0 分）已在 `_rank_key` 第 3 位生效 —— **该项已有基础**；
#   · 「候选数量」⇒ **完全没有**被当作指标。原 `_rank_key` 只看「该候选拿到几票」，
#     不看「这个结论在池里被**几个不同候选独立复现**」。
#     二者语义不同：一条结论若只被 1 个候选写出（哪怕它自评 3/3 票），
#     与「3 个独立候选都写出同一答案」相比，后者证据强得多
#     （self-consistency 的原始口径就是"独立采样间的一致性"，不是"单候选自评"）。
#   · 「可信度要求」⇒ 散落在 `accept_confidence=0.6`（AcceptGate）与硬编码 `< 0.5`
#     （5.5 低置信度复核），**没有**在"选答"这个动作上做统一闸门。
#
# 设计纪律（遵守"一次只改一项"与"宁缺勿滥"）：
#   ① 只做**加性**指标：`_answer_support` 是新增的第 2 排序键，不删除任何既有键；
#   ② 所有新闸门默认**不改变行为**（`pick_confidence_floor` 默认 0.0 = 关闭）；
#   ③ 每个新函数都是纯函数，可被单元测试直接覆盖，绝不抛异常。
def _answer_support(candidates, key_of=None) -> dict:
    """统计**答案归一化后**每个结论被多少个**独立候选**复现。

    返回 ``{归一化答案: 候选个数}``。用途 = 给"候选数量"这个指标提供口径。

    ⚠ 与本文件既有的"票数"区别：
      · 票数（`correct_votes`）= **验证器对同一个候选投了几张对票**；
      · 本函数 = **池里有几个不同候选写出了这个结论**（独立复现）。
    两者都是指标，不是替代关系 —— 故本函数只新增排序键，不覆盖票数。

    归一化走 `AnswerOracle.strip_wrappers`（剥 `\\boxed{}` 等外壳）后 `strip().lower()`
    —— 与 `_are_answers_equivalent` 同源口径，避免 `\\boxed{5}` 与 `5` 被算成两个结论。
    任何异常一律返回空 dict（绝不阻断选答）。
    """
    out: dict = {}
    try:
        if not candidates:
            return out
        try:
            from .answer_oracle import AnswerOracle as _AO
        except Exception:  # noqa: BLE001
            _AO = None
        for _c in candidates:
            _a = ((key_of(_c) if key_of else getattr(_c, "answer", "")) or "")
            _a = str(_a).strip()
            if not _a or _looks_like_non_answer(_a):
                continue
            if _AO is not None:
                try:
                    _a = _AO.strip_wrappers(_a) or _a
                except Exception:  # noqa: BLE001
                    pass
            _k = _a.strip().lower()
            if not _k:
                continue
            out[_k] = out.get(_k, 0) + 1
    except Exception:  # noqa: BLE001
        return {}
    return out


def _support_of(candidate, support_map) -> int:
    """取某候选的「独立复现数」；查不到记 1（它自己就是一票，不该被罚成 0）。"""
    try:
        if not support_map:
            return 1
        ans = str(getattr(candidate, "answer", "") or "").strip()
        if not ans:
            return 1
        try:
            from .answer_oracle import AnswerOracle as _AO
            ans = _AO.strip_wrappers(ans) or ans
        except Exception:  # noqa: BLE001
            pass
        return int(support_map.get(ans.strip().lower(), 1) or 1)
    except Exception:  # noqa: BLE001
        return 1


def _passes_confidence_floor(candidate, floor: float) -> tuple:
    """★ 可信度门槛：低于底线的候选**不得**被选为终答（返回 ``(是否放行, 原因)``）。

    用户口径：「可信度要求（太低＝全军覆没）」——
    即"低可信度答案提交出去几乎必错，等于整题白做"，所以宁可**显式记录**
    "本次无候选达线"也不要静默地挑一个最弱的交上去。

    判定用**双指标**（两条都看，任一条成立即放行）：
      ① `confidence`（= 正确票/有效票）≥ floor；
      ② `correct_votes` ≥ 1 且 `total_votes` ≥ 1 —— 即"**至少有一张真实的正确票**"。
         为什么要加 ②：`confidence` 在 total_votes=0（全弃权票）时被定义为 0.0，
         但那是**基础设施故障**（LLM 超时/输出不可解析），不是"被判错"
         —— 与 `Verdict.abstain` 的既有三态口径一致（故障不是反证）。
         若只按 ① 判，故障轮次的候选会被全部打成"不可信"，触发无谓的回退。

    `floor <= 0` ⇒ 恒放行（= 默认关闭，行为与改动前逐字一致）。
    """
    try:
        if float(floor or 0.0) <= 0.0:
            return True, "floor_disabled"
        _cv = int(getattr(candidate, "correct_votes", 0) or 0)
        _tv = int(getattr(candidate, "total_votes", 0) or 0)
        if _tv <= 0:
            # 全弃权票 = 故障，不是反证 ⇒ 放行并如实说明（不伪装成"可信"）
            return True, "no_effective_votes"
        _conf = float(getattr(candidate, "confidence", 0.0) or 0.0)
        if _conf >= float(floor):
            return True, "confidence_ok"
        if _cv >= 1:
            return True, "has_correct_vote"
        return False, "below_floor"
    except Exception:  # noqa: BLE001
        return True, "judge_error"


_CJK_PROSE_RE = re.compile(r"[\u4e00-\u9fff]{4,}")
_ENUM_ITEM_MAX = 40


def _is_real_enumeration(core: str) -> bool:
    """core 是否为**真枚举答案**（而非"含逗号的散文 / 推导片段"）。

    ★ 2026-09-22 修复（"删去原本不合理的逻辑"）：原判据仅是
    `re.search(r"[,，;；、]")` ⇒ **任何含逗号的中文散文**都会被当成"枚举答案"，
    并在 `enum_preferred` 分支里**优先返回**（该分支直接 `return`，绕过后面全部
    选答逻辑）。实测 014：正确答案 995018（票数最高 2/3、置信度 0.667）
    被一条 **0 票的散文候选** 顶掉，只因后者含逗号 ⇒ 终答以散文输出，判分必错。

    现在的判据：按逗号/分号/顿号拆分后
      ① 至少 2 项；
      ② **每一项都必须是短数学对象** —— 长度 ≤ 40 且不含 4 个以上连续汉字
         （长中文短语 ⇒ 是句子而非数学对象）；
      ③ 项末不得是句末标点。
    宁缺勿滥：判否只是放弃"枚举优先"这条捷径，仍会走正常选答，不会更差。
    """
    s = str(core or "").strip()
    if len(s) < 3:
        return False
    parts = [p.strip() for p in re.split(r"[,，;；、]", s) if p.strip()]
    if len(parts) < 2:
        return False
    for p in parts:
        if len(p) > _ENUM_ITEM_MAX:
            return False
        if _CJK_PROSE_RE.search(p):
            return False
        if re.search(r"[。！？：]$", p):
            return False
    return True
