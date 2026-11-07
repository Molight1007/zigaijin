# -*- coding: utf-8 -*-
"""单一「题目目标」（任务列表 #3，2026-10-14）。

================================================================================
职责
================================================================================
把「题目要我们做什么」压成**唯一一份**目标，供蓝图构建、子目标求解、merge
渲染共用。对齐任务列表第 3 条原文：

    题意理解要和蓝图构建解耦合，它应该是单纯确定题目需要我们做什么 ——
    选择题要选哪几个选项、有几个正确选项？解答题要求的是什么？
    证明题要证明的是什么？**给出大模型一个目标**。

================================================================================
本模块取代的「题意理解副本」（这是 #3 的关键：不是新增一层，而是收敛）
================================================================================
原先「理解题意 / 抽目标」散在**四处**、互不知情：

  ① 2.6 `run_understanding_chain` —— 终态 + 逻辑链 + 缺口（**保留，成为唯一来源**）
  ② 蓝图 LLM 的 `problem_analysis` —— domain / core_objective / key_constraints
  ③ 子目标 LLM 规划的 `problem_analysis` —— 同上
  ④ 选择题两处**代码构造**的 `problem_analysis` —— 选项清单 / 单选多选口径

其中 ④ 最刺眼：选择题「要判哪几个选项」这条**目标信息**，本属题意理解，
却由 2.7 子目标规划现场重造一遍。本模块把它上移到 2.6，②③④ 的产出被删除。

================================================================================
零新增 LLM 调用
================================================================================
全部由**既有产物**派生：`ctx.question_type` / `ctx.type_path` /
`ctx.understanding_chain` / `extract_options` / `answer_form_requirement`。
⇒ 接线后不增加任何一次 LLM 往返，也不改变既有行为口径（未识别时退化为空字段）。
"""
from __future__ import annotations

import re

from .question_type import (
    QT_CHOICE, QT_FILL, QT_JUDGE, QT_PROOF, QT_SOLUTION,
    asks_all_values, extract_options, type_path_plan,
)

# ---------------------------------------------------------------------------
# 单选 / 多选口径（原 `sub_goal_solver._SINGLE_CHOICE_RE` 上移至此，成为唯一来源）
# ---------------------------------------------------------------------------
# 为什么上移：该正则原本只服务于 2.7 的选择题规划（决定 merge 话术「单选型/多选型」），
# 但「本问要选一个还是选全部」是**题意**，不是**规划**。上移后由 2.6 一次判定，
# 2.7 直接读 `ctx.problem_goal["n_correct"]`，不再各判一次。
SINGLE_CHOICE_RE = re.compile(
    r"哪一项|哪一种|哪一类|哪一个|最为?合适|最恰当|最符合|最适合"
    r"|分别是?|定义是|通常采用|应采用|应选择|最好采用")
_MULTI_CHOICE_RE = re.compile(r"哪些|正确的有|全部|所有|哪几项|哪几个")

# 题型 → 「要做什么」的一句话（目标陈述的骨架）
_TASK_BY_QT = {
    QT_PROOF: "证明题面所给命题成立（给出完整证明过程）",
    QT_SOLUTION: "求出题目所要求的量，给出具体的值 / 对象",
    QT_CHOICE: "判定下列各选项中哪些成立，并按选项标签给出答案",
    QT_JUDGE: "判断题面命题是否成立（回答 成立 / 不成立）",
    QT_FILL: "填出题面空缺处的值",
}
_TASK_DEFAULT = "完成题目所要求的目标"


def judge_n_correct(problem: str, qtype: str) -> str:
    """选择题 / 判断题的**答案口径**：``"single"`` / ``"all"`` / ``"unknown"``（零 LLM）。

    非客观题返回空串。判定顺序：单选措辞优先（"哪一项/最合适/定义是"），
    其次多选措辞（"哪些/正确的有/全部"），都不命中则 ``"unknown"``
    —— **不猜**，下游按原有默认话术处理（与接线前一致）。
    """
    if qtype not in (QT_CHOICE, QT_JUDGE, QT_FILL):
        return ""
    t = problem or ""
    if SINGLE_CHOICE_RE.search(t):
        return "single"
    if _MULTI_CHOICE_RE.search(t) or asks_all_values(t):
        return "all"
    return "unknown"


def is_single_choice(ctx, problem: str = "") -> bool:
    """选择题是否「单选」口径。**单一来源**：先读 2.6 判定的 ``n_correct``，
    缺失（未接线 / 老 ctx）才回退题面正则 —— 与接线前行为逐字一致。

    为什么上移到此处：该判定原本只存在于 2.7 的选择题规划里
    （`sub_goal_solver._SINGLE_CHOICE_RE`，仅用于选 merge 话术的「单选型/多选型」），
    但「本问选一个还是选全部」属**题意**，应由 2.6 一次定下、2.7 只读结论。
    """
    g = getattr(ctx, "problem_goal", None)
    n = str((g or {}).get("n_correct") or "") if isinstance(g, dict) else ""
    if n:
        return n == "single"
    return bool(SINGLE_CHOICE_RE.search(
        problem or str(getattr(ctx, "problem", "") or "")))


def build_problem_goal(ctx) -> dict:
    """从既有产物派生**唯一**题目目标（不调用 LLM）。返回 dict；失败不抛异常。

    字段：
      · ``question_type``  中文题型（与 ``ctx.question_type`` 同源）
      · ``task``           一句话：题目要我们做什么（按题型分派）
      · ``options``        ``[{"label","text"}]``；非选择题为 ``[]``
      · ``n_correct``      ``single`` / ``all`` / ``unknown``（客观题口径）
      · ``answer_form``    答案形态（``proof`` / ``value`` / ``option`` / ``yesno``）
      · ``terminal_state`` 题目完成后的情形（理解链，符号槽；**假设，非结论**）
      · ``answer_slot``    待求量的符号形式
      · ``goal_statement`` 供提示词注入的单句目标
    """
    qtype = str(getattr(ctx, "question_type", "") or "").strip()
    problem = str(getattr(ctx, "problem", "") or "")

    options: list = []
    n_correct = ""
    if qtype == QT_CHOICE:
        try:
            options = [{"label": _l, "text": _t}
                       for _l, _t in (extract_options(problem) or [])]
        except Exception:  # noqa: BLE001  选项提取失败不影响目标本身
            options = []
        n_correct = judge_n_correct(problem, qtype)
    elif qtype == QT_JUDGE:
        n_correct = "single"

    # 理解链（②的替代来源）：终态 + 答案槽
    _uc = getattr(ctx, "understanding_chain", None)
    _uc = _uc if isinstance(_uc, dict) else {}
    terminal = str(_uc.get("terminal_state") or "").strip()
    answer_slot = str(_uc.get("answer_slot") or "").strip()

    # 答案形态：优先取已算好的 type_path（题型分流表），缺失时现算
    _tp = getattr(ctx, "type_path", None)
    if not isinstance(_tp, dict) or not _tp:
        try:
            _tp = type_path_plan(qtype)
        except Exception:  # noqa: BLE001
            _tp = {}
    answer_form = str((_tp or {}).get("answer_form") or "")

    task = _TASK_BY_QT.get(qtype, _TASK_DEFAULT)
    # 目标句：通用任务句；理解链给出终态时用终态把它具体化
    goal = task
    if terminal:
        if qtype == QT_PROOF:
            goal = "证明：" + terminal[:300]
        elif qtype in (QT_SOLUTION, QT_FILL):
            goal = task + "；目标情形：" + terminal[:300]

    return {
        "question_type": qtype,
        "task": task,
        "options": options,
        "n_correct": n_correct,
        "answer_form": answer_form,
        "terminal_state": terminal,
        "answer_slot": answer_slot,
        "goal_statement": goal,
        # 来源标记：派生（零 LLM）。历史 jsonl 无此字段 ⇒ .get 容错。
        "source": "derived",
    }


def render_problem_goal(ctx) -> str:
    """把题目目标渲染成提示词块（蓝图 / 子目标规划共用）。无目标返回空串。

    ⚠ **红线（K11）**：``terminal_state`` 里的答案槽是**假设**，本块显式标注
      "不得当作已证结论"，绝不把假设值下传成结论（答案槽本身已在 2.6
      `lean_pre_verifier` 侧被**机械剥离**数值 —— 见 `_mask_answer_values`）。
    """
    g = getattr(ctx, "problem_goal", None)
    if not isinstance(g, dict) or not g.get("question_type"):
        return ""
    lines = ["【题目目标（**唯一口径**：蓝图与子目标分解一律以此为准）】",
             "· 题型：" + g["question_type"]
             + ("（答案口径：" + str(g["n_correct"]) + "）"
                if g.get("n_correct") in ("single", "all") else ""),
             "· 要做什么：" + str(g.get("task") or "")]
    _opts = g.get("options") or []
    if _opts:
        lines.append("· 待判选项（%d 个，一个都不能漏）：%s"
                     % (len(_opts),
                        " / ".join("%s. %s" % (o.get("label"), str(o.get("text"))[:60])
                                   for o in _opts)))
    if g.get("answer_form"):
        lines.append("· 答案形态：" + str(g["answer_form"]))
    if g.get("terminal_state"):
        lines.append("· 目标情形（题目完成后的情形；其中答案槽是**假设**，"
                     "**不得当作已证结论**）：" + str(g["terminal_state"])[:400])
    return "\n".join(lines)
