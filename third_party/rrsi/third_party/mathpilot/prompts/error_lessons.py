# -*- coding: utf-8 -*-
"""易错点记忆库（error lessons，2026-09-06 A 档轻量经验注入）。

把项目历轮评测/归因沉淀的「同类题常错点」做成检查清单片段，按题型与
题干关键词匹配后注入 solver 生成 / revise 提示，让 LLM 交付前自查。
与 B 档（自动入库+命中统计）共用同一数据源，未来可平滑升级。

防误伤三原则（历史教训：theorem_memory 无写入方变死代码、正则判据静默
失效——宁缺毋滥）：
1. 只匹配**命中关键词/题型**才注入，无命中返回空串 = 零噪音；
2. 表述为「自查清单」，绝不暗示答案内容（防把模型带偏）；
3. 输出规范类易错（截断/占位符/拒答/超时）已由代码层防护，**不注入**，
   避免 prompt 膨胀干扰正常输出。

来源：
- E-check  <- comb-032 51vs43 / geom-051 25vs29（数值代回错）
- E-form   <- algebra-003 2x³ 漏 +c（求函数题解族不完整）
- E-eq     <- nt-031 与 5(l-1)² 不等价（形变后不自检等价）
- E-extreme<- 009 心算极值 3.815 真值 4.94（全链自洽错值）
"""

from __future__ import annotations

import os
import re as _re

# domain 判定用子串（题库 domain 有中英两套：Algebra/代数/NumberTheory/数论/…）
_ANSWER_QRY_DOMAINS = (
    "algebra", "代数", "number", "数论", "combin", "组合",
    "geometry", "几何", "equation", "方程", "calculus", "微积分",
    "analysis", "分析", "函数",
)

LESSONS = [
    {
        "id": "E-check",
        "name": "数值答案代回自检",
        # 解答题统一注入；证明题走整题推理，不适用"数值代回"检查清单
        "qtypes": ("解答题", "计算题"),
        "domains": _ANSWER_QRY_DOMAINS,
        "text": (
            "若最终答案包含具体数值/点/集合，交付前将其代回原题条件独立"
            "复核一次（必要时换一种算法重算），确认没有算错或代错。"
        ),
    },
    {
        "id": "E-form",
        "name": "解族完整性（常数项/参数）",
        "qtypes": ("解答题",),
        "domains": _ANSWER_QRY_DOMAINS,
        # 求函数/通解/一般形式/Find all → 检查是否漏任意常数项/参数及其范围
        "re_problem": r"(find all|求所有|通解|一般解|一般形式|general\s*solution|"
                      r"all\s*functions|函数\s*[:：]|f\s*\([^)]*\)\s*[:：])",
        "text": (
            "若题目求函数/解族/一般形式，交付前确认解族完整：是否遗漏任意"
            "常数项或参数（如 +c、n∈Z）、是否声明参数取值范围、边界情形"
            "（如零函数/退化情形）是否单独说明。"
        ),
    },
    {
        "id": "E-eq",
        "name": "符号形变等价自检",
        "qtypes": ("解答题", "证明题"),
        "domains": ("algebra", "代数", "number", "数论", "combin", "组合",
                    "equation", "方程", "calculus", "微积分", "analysis", "分析"),
        "text": (
            "若对表达式做了配方/换元/化简等形变，交付前将变形前后做符号展开"
            "对比（或取 2-3 个具体参数值代入两边核对），确保真正等价——"
            "仅『看起来像』不算数。"
        ),
    },
    {
        "id": "E-extreme",
        "name": "极值/最值声称先数值验证",
        "qtypes": ("解答题", "证明题"),
        "domains": ("algebra", "代数", "geometry", "几何", "calculus", "微积分",
                    "analysis", "分析", "equation", "方程"),
        # ⚠ 2026-09-15 修复：原正则含**裸** `max|min`，会误匹配 examine / determine /
        # administration 等含 "min" 的普通词（实测 44/112 命中，明显偏高）。
        # 收紧为词边界 `\bmax\b|\bmin\b` + 完整词形。
        "re_problem": r"(最大|最小|极大|极小|极值|最值|最大值|最小值|"
                      r"\bmax\b|\bmin\b|maximum|minimum|maximal|minimal|"
                      r"largest|smallest|greatest|least|extreme|"
                      r"at\s+most|at\s+least)",
        "text": (
            "若题目涉及最大值/最小值/极值，先对声称的极值点做数值采样或代入"
            "边界验证，再下结论——心算错值后全链路自洽是高风险错误。"
        ),
    },
    # ==================================================================
    # 2026-09-15 新增三条（依据《错题错误类型归因分析报告 v1.1》§7.2
    # 「优先攻克技巧类」）。对应错题：E-count ← B 类 14 题；
    # E-bound ← C 类 3 题（002/004/066）；E-magnitude ← A 类 21 题。
    # ⚠ 消融开关：环境变量 ERROR_LESSONS_EXTRA（见 _extra_enabled），
    #   默认 "all" 全开；A/B 对照时设 "none"，或 "count"/"bound"/"magnitude"
    #   逐条验证。三条均只改提示词，不新增 LLM 调用、不改控制流。
    # ==================================================================
    {
        "id": "E-count",
        "name": "计数/枚举完备性自检",
        # 对应 B 类失败模式：漏算、重复计入、边界情形未覆盖（013/016/017/031/073）
        "qtypes": ("解答题", "填空题"),
        "domains": _ANSWER_QRY_DOMAINS,
        "re_problem": r"(多少种|多少个|多少条|共有|几种|个数|方案数|计数|"
                      r"排列|组合|ways|count|how\s+many|number\s+of|"
                      r"find\s+all|determine\s+all|求所有|所有可能|全部|枚举)",
        "text": (
            "若本题要求计数或枚举（“有多少种”“共有多少个”“求方案数”“所有可能”），"
            "交付前做一次穷尽性自检，逐条回答："
            "① 你枚举的范围是什么（从哪个值到哪个值、边界条件是什么）；"
            "② 列出已计入的情形清单（可简写，但要能数得清条数）；"
            "③ 显式声明被排除的情形及排除理由；"
            "④ 追问一次：是否存在第二个解族——被漏掉的对称情形、退化情形、边界取值？"
            "若第 ④ 问无法明确排除，重新枚举后再交付，不要带着“应该没有别的了”的猜测结束。"
        ),
    },
    {
        "id": "E-bound",
        "name": "极值必须构造+上界双向论证",
        # 对应 C 类：极值题只给一半论证（002 构造+上界、004 上界缺失、066 面数上界差一）
        # 与已有 E-extreme 互补不重叠：E-extreme 管“数值算得对不对”，本条管“论证全不全”
        "qtypes": ("解答题", "证明题"),
        "domains": _ANSWER_QRY_DOMAINS,
        "re_problem": r"(最大|最小|极大|极小|极值|最值|最多|最少|至多|至少|"
                      r"\bmax\b|\bmin\b|maximum|minimum|maximal|minimal|"
                      r"largest|smallest|greatest|least|fewest|"
                      r"at\s+most|at\s+least)",
        "text": (
            "若本题求最大值/最小值/极值，结论必须由两半论证共同支撑，缺一不可："
            "① 构造（可达性）：给出一个具体实例，并验证它确实取到该值；"
            "② 上界（不可能性）：证明不存在更优的值——用反证、放缩、单调性或不变量，"
            "说明再大（小）就会矛盾。"
            "只有上界没有构造 ⇒ 不知道能否达到；只有构造没有上界 ⇒ 不知道是否最优。"
            "若发现缺一半，补齐后再交付。"
        ),
    },
    {
        "id": "E-magnitude",
        "name": "数值答案的量级合理性核对",
        # ★ 唯一能低成本触及 A 类（思路/建模错，43.8%）的机制：
        #   A 类客观指纹 = 量级级偏差 10–1000 倍（报告 §4.3），量级可无参考答案自检。
        # ⚠ 已知失效边界：若模型对量级的估计也沿同一错误方向，本自检会“自洽通过”
        #   （报告 §6 证据指向该可能）⇒ 必须 A/B 验证，不能凭设计直觉认为有效。
        "qtypes": ("解答题", "填空题"),
        "domains": _ANSWER_QRY_DOMAINS,
        # 不做题干匹配：凡求数值解的题都注入
        "text": (
            "若最终答案是具体数值，交付前做一次量级核对（这与“检查算术”是两件事）："
            "① 只用题目条件做粗放缩，独立估出这个量应该落在什么数量级"
            "（例如“应为 10³ 量级”“介于 100 与 5000 之间”）——"
            "估算时不要参考你已经算出的结果；"
            "② 把你的答案与这个估计相比：若相差 10 倍以上，说明大概率是选错了数学模型，"
            "请回到“把问题转化成什么数学结构”这一步重新审视，而不是去检查算术；"
            "③ 若数量级一致，再检查边界情形与计算细节。"
        ),
    },
]

# ---------------------------------------------------------------------
# 消融控制（2026-09-15）：仅作用于上面三条 2026-09-15 新增的 lesson。
# 原有 4 条（E-check / E-form / E-eq / E-extreme）始终生效，不受影响，
# 保证任何对照实验都与“新增三条之前”的行为可比。
# ---------------------------------------------------------------------
_EXTRA_IDS = ("E-count", "E-bound", "E-magnitude")
_EXTRA_ENV = "ERROR_LESSONS_EXTRA"
_EXTRA_ALIAS = {"count": "E-count", "bound": "E-bound", "magnitude": "E-magnitude"}


def _extra_enabled() -> set:
    """本次启用哪些新增 lesson（消融控制）。

    ERROR_LESSONS_EXTRA 取值（大小写不敏感）：
      "all" / "1" / "on"（默认）  → 三条全开
      "none" / "0" / "off"        → 三条全关（回到 2026-09-15 之前的行为）
      "count,bound" 等逗号分隔     → 只开指定子集（支持短名 count/bound/magnitude）
    取值无法识别时按“全开”处理（宁可多注入，也不静默失效）。
    """
    raw = (os.environ.get(_EXTRA_ENV, "all") or "all").strip().lower()
    if raw in ("none", "0", "off", "false"):
        return set()
    if raw in ("all", "1", "on", "true"):
        return set(_EXTRA_IDS)
    out = set()
    for part in raw.split(","):
        p = part.strip()
        if p in _EXTRA_ALIAS:
            out.add(_EXTRA_ALIAS[p])
        elif p in _EXTRA_IDS:
            out.add(p)
    return out if out else set(_EXTRA_IDS)


def _active_lessons():
    """当前生效的 lesson 列表（新增三条受 ERROR_LESSONS_EXTRA 消融控制）。"""
    on = _extra_enabled()
    for lesson in LESSONS:
        if lesson["id"] in _EXTRA_IDS and lesson["id"] not in on:
            continue
        yield lesson


def match_lessons(domain: str = "", question_type: str = "",
                  problem: str = "") -> str:
    """按题型/领域/题干关键词返回命中的易错自查清单（无命中返回空串）。

    返回的文本为提示片段，调用方自行决定拼接位置与开关（enable_error_lessons）。
    """
    hits: list[dict] = []
    for lesson in _active_lessons():
        qts = lesson.get("qtypes") or ()
        if qts and question_type not in qts:
            continue
        # ⚠ 2026-09-15 框架级修复：原有的「domain 白名单硬过滤」已移除。
        # 原实现 `if doms and not any(k in dom_l for k in doms): continue` 是**纯白名单**，
        # 缺项即静默跳过 —— 实测题库 19 种 domain 取值中白名单只覆盖 16.1%（18/112），
        # 其中最大域「离散数学」（51 题）与 25 道空 domain 题**整域失效**，
        # 89 道错题里仅 14.6% 能命中任何 lesson（机制事实上空转）。
        # 这些清单均为**通用数学自查**，按域裁剪的收益远小于静默失效的代价；
        # 适用性交由 `qtypes`（题型）与 `re_problem`（题干关键词）决定。
        rx = lesson.get("re_problem")
        if rx and not _re.search(rx, problem or "", _re.IGNORECASE):
            continue
        hits.append(lesson)
    if not hits:
        return ""
    lines = "\n".join(f"- {lesson['text']}" for lesson in hits)
    return (
        "【历史易错自查清单】（同类题曾反复出错，交付前逐条对照，"
        "不要盲从、以实际推理为准）：\n" + lines
    )


def lesson_ids(domain: str = "", question_type: str = "",
               problem: str = "") -> list[str]:
    """返回命中的 lesson id 列表（诊断/日志用）。"""
    out = []
    for lesson in _active_lessons():
        qts = lesson.get("qtypes") or ()
        if qts and question_type not in qts:
            continue
        # 与 match_lessons 严格同源：domain 白名单硬过滤已移除（2026-09-15）
        rx = lesson.get("re_problem")
        if rx and not _re.search(rx, problem or "", _re.IGNORECASE):
            continue
        out.append(lesson["id"])
    return out
