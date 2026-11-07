# -*- coding: utf-8 -*-
"""子目标求解产出的**唯一读写入口**与分节投影。

--------------------------------------------------------------------------
为什么要单独一层（2026-10-20）
--------------------------------------------------------------------------
此前同一份求解产出被写进两个容器（`results_map[id]` 与 `sg["result"]`），
而协作路径只写了前者 ⇒ 所有读 `sg["result"]` 的消费者拿到空值，候选的
`reasoning` 退化成蓝图骨架（soft019_r1 实测：5 个子目标结果全空）。
当时的补救是在三个模块各加一句 `a or b` 兜底——那是**把耦合摊到了每个读点**：
读者必须知道"有哪几个容器、按什么顺序退"。本模块把这件事收回来：

    写 —— 只经 `write_step()`：两个容器同值写入，调用点无从漏写。
    读 —— 只经 `read_step()` / `read_conclusion()`：容器数量与顺序是本模块
           的知识，读点不再各自拼 fallback。
    分节 —— `parse_sections()` 通用解析：**不写死三节**。模型多写一节
           （「验证」「反例」「引理」…）原样保留；未来新增节无需改代码。

设计取舍（与「降级」的界限）：
  · 存储一律**完整文本**，不做"只留结论节"的压缩——压缩放到投影层显式做。
  · 投影（`clip`）**优先保留结论节**：直接 `[:n]` 会把末尾结论切掉，而结论
    恰恰是下游最需要的东西。
  · 结构化视图 `sg["step_sections"]` 随写入落盘，未来按节取用不必重新解析。
"""

from __future__ import annotations

import re

__all__ = [
    "section", "section_of", "parse_sections", "compose_text", "split_blocks",
    "clip", "demote_headings", "write_step", "read_step", "read_conclusion",
    "conclusion_of", "cached_text", "cached_contract",
]

#: `## 标题` —— 允许无空格（`##标题`），但**排除** `###` 及更深层级。
#: 排除方式用 `(?!#)`：`###` 在 `##` 之后紧跟的是 `#`。
_SECTION_RE = re.compile(r"^[ \t]*##[ \t]*(?!#)([^\n#]+?)[ \t]*$", re.M)

#: 旧三节键 → 可接受的节名（含常见别名）。仅用于 `split_blocks()` 的
#: 向后兼容投影，**不是** `parse_sections()` 的白名单。
_LEGACY_BLOCKS = {
    "logic": ("逻辑推导", "推导", "推理", "论证"),
    "calc": ("计算", "验算", "计算过程"),
    "result": ("结论", "本步结论", "最终结论", "结果"),
}


# ------------------------------------------------------------------ 分节解析
def section(text: str, name: str) -> str:
    """取 `## name` 一节的正文（不存在 → ""）。

    保留此签名是为了兼容 `planning_agent` 等既有调用点；新代码优先用
    `parse_sections()` 一次解析全部节，避免对同一段文本反复正则扫描。
    """
    return str(parse_sections(text).get(str(name), ""))


def parse_sections(text: str) -> dict:
    """`## 标题` 切块 → {标题: 正文}，**保持原文顺序**。

    · 无任何 `##` 分节 ⇒ 返回 `{}`（由调用方决定整段怎么用）。
    · 空正文的节被丢弃（只有标题没有内容不算一节）。
    · 同名节多次出现 ⇒ 按顺序拼接（模型偶尔续写同一节标题）。
    """
    t = str(text or "")
    marks = list(_SECTION_RE.finditer(t))
    if not marks:
        return {}
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        name = m.group(1).strip()
        if not name:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        body = t[m.end():end].strip()
        if not body:
            continue
        out[name] = (out[name] + "\n" + body).strip() if name in out else body
    return out


def compose_text(sections: dict, raw: str = "") -> str:
    """分节 → 单段文本（节名加粗，顺序即原文顺序）。

    ⚠ **只用于"从结构重建"**（例如把投影后的三节拼回去）。**不要**用它
      生产存储形态：它产出 `**节名**`，而 `parse_sections()` 认 `## 节名`，
      存成这样下次就解析不回来了。`write_step()` 因此优先保留原文。
    """
    if not sections:
        return str(raw or "").strip()
    parts = ["**%s**\n%s" % (k, v) for k, v in sections.items() if str(v).strip()]
    return "\n\n".join(parts).strip() or str(raw or "").strip()


def split_blocks(text: str) -> dict:
    """旧三节视图 `{"logic","calc","result"}`（向后兼容投影）。

    ⚠ 这是**投影**，不是权威存储：`parse_sections()` 才是。新代码若只关心
      某一节，直接用 `section()` / `parse_sections()`，别再增加写死的键。
    """
    t = str(text or "")
    secs = parse_sections(t)
    if not secs:
        return {"logic": "", "calc": "", "result": t.strip()}
    out = {}
    for key, names in _LEGACY_BLOCKS.items():
        out[key] = _pick(secs, names)
    return out


def _pick(secs: dict, names: tuple) -> str:
    """按别名取节：先精确匹配，再退到包含匹配（「结论（汇总）」这类）。"""
    for n in names:
        if n in secs:
            return secs[n]
    for n in names:
        for k, v in secs.items():
            if n in k:
                return v
    return ""


def section_of(sections: dict, kind: str) -> str:
    """按**语义类别**取节，`kind ∈ {"logic", "calc", "result"}`。

    调用方只声明"我要逻辑/计算/结论"，具体接受哪些节名（「推导」「推理」
    「验算」…）是**本模块的知识**。这样模型换了说法、或未来新增别名，
    调用点一行都不用改 —— 别在业务代码里再写
    `secs.get("逻辑推导") or secs.get("推导")` 这类枚举。
    """
    return _pick(sections or {}, _LEGACY_BLOCKS.get(str(kind), ()))


def conclusion_of(text: str) -> str:
    """从一段产出里取「结论」节正文（取不到 → ""）。"""
    return _pick(parse_sections(text), _LEGACY_BLOCKS["result"])


# ------------------------------------------------------------------ 预算裁剪
def demote_headings(text: str) -> str:
    """把 `## 标题` 降为 `**标题**`（嵌入更深层文档时用）。

    场景：`_build_full_reasoning` 用 `### 子目标 #1` 作为小节标题，若把某步的
    完整求解文本原样嵌进去，其中的 `## 结论` 会被 Markdown 解析成**文档级**标题，
    整体结构就乱了（审核模型读到的层级也不再表示"这是第 1 步的内部"）。
    降级是无损的：文本一字不改，只换标记 —— 而 `parse_sections()` 仍认原始
    存储形态，两者互不影响。
    """
    return _SECTION_RE.sub(lambda m: "**%s**" % m.group(1).strip(),
                           str(text or ""))


def clip(text: str, budget: int, note: str = "…（已按预算截断）") -> str:
    """按字符预算裁剪，**优先保留结论节**。

    为什么不能直接用 `text[:budget]`：结论节在末尾，硬切最先被切掉的正是
    下游最需要的结论 ⇒ 合并层会以为该步没有产出。本实现在超预算时把结论节
    整段留下，剩余预算按原文顺序分给其余节。

    `budget <= 0` 表示不裁（默认口径：宁可长，不静默丢信息）。
    """
    t = str(text or "")
    if budget is None or budget <= 0 or len(t) <= budget:
        return t
    secs = parse_sections(t)
    tail_key = _conclusion_name(secs)
    if not tail_key:
        return t[:budget] + "\n" + note
    tail = "**%s**\n%s" % (tail_key, secs[tail_key])
    head_budget = budget - len(tail) - len(note) - 8
    if head_budget <= 64:
        return tail          # 预算太紧：只保留结论，别切成半句
    head, used = [], 0
    for k, v in secs.items():
        if k == tail_key:
            continue
        blk = "**%s**\n%s" % (k, v)
        if used + len(blk) > head_budget:
            left = head_budget - used
            if left > 64:
                head.append(blk[:left].rstrip() + "…")
            break
        head.append(blk)
        used += len(blk) + 2
    return "\n\n".join(head + [tail]) + "\n" + note


def _conclusion_name(secs: dict) -> str:
    for n in _LEGACY_BLOCKS["result"]:
        if n in secs:
            return n
    for n in _LEGACY_BLOCKS["result"]:
        for k in secs:
            if n in k:
                return k
    return ""


# ------------------------------------------------------------------ 唯一读写
def write_step(sg: dict, results_map: dict | None, raw: str,
               sections: dict | None = None) -> str:
    """**唯一写入口**：把一次求解产出写进全部容器（同值），返回完整文本。

    设计要点：
      · `sg["result"]` 与 `results_map[id]` **写同一份完整文本** —— 不存在
        "一个存摘要、一个存全文"的口径分裂（那正是空壳事故的土壤）。
      · `sg["step_sections"]` 存结构化视图，未来按节消费免重解析。
      · 返回值即完整文本，调用方需要摘要时用 `conclusion_of()` / `split_blocks()`
        显式投影，而不是拿存储层的形态凑合。
    """
    secs = sections if sections is not None else parse_sections(raw)
    # ★ 存储保留**原文**：原文天然可被 `parse_sections()` 再解析，而
    #   `compose_text()` 产出的 `**节名**` 不可 —— 用合成文本当存储会让
    #   下游再也取不到节（这是个循环依赖陷阱，2026-10-20 踩到）。
    text = str(raw or "").strip() or compose_text(secs)
    sg["result"] = text
    if secs:
        sg["step_sections"] = secs
    if results_map is not None and sg.get("id") is not None:
        results_map[sg["id"]] = text
    return text


def cached_text(cached) -> str:
    """`ctx._subgoal_done` 条目 → 求解文本（**缓存形态的唯一规范化点**）。

    缓存历史上出现过两种形态（`{fp: str}` 与 `{fp: {"result":…, "contract":…}}`）。
    形态判断只允许出现在这里：`subgoal_collab` 与 `sub_goal_solver__loop`
    两条求解路径共用同一个读法，避免"每条路径各写一段 isinstance"。
    """
    if isinstance(cached, dict):
        return str(cached.get("result") or "").strip()
    return str(cached or "").strip()


def cached_contract(cached) -> dict:
    """`ctx._subgoal_done` 条目 → 结论契约快照（无则 `{}`）。"""
    if isinstance(cached, dict):
        snap = cached.get("contract")
        return snap if isinstance(snap, dict) else {}
    return {}


def read_step(sg: dict, results_map: dict | None = None) -> str:
    """**唯一读入口**：取该子目标的完整求解文本（取不到 → ""）。

    两个容器在正常写入下同值，这里保留顺序只是为了兼容旧数据/外部注入的
    `results_map`；调用方**不需要**知道这件事。
    """
    text = str(sg.get("result") or "").strip()
    if text:
        return text
    if results_map:
        return str(results_map.get(sg.get("id")) or "").strip()
    return ""


def read_conclusion(sg: dict, results_map: dict | None = None) -> str:
    """取该子目标的**答案级**结论（供最终答案兜底/投票）。

    优先级：结构化结论契约 → 文本里的「结论」节 → 整段。
    契约优先是因为它已是 `value/equation/verdict` 归一形态，可直接当答案；
    整段兜底只用于完全没有分节的产出，避免"把整篇推导当终答"。
    """
    concl = str(sg.get("conclusion") or "").strip()
    if concl:
        return concl
    text = read_step(sg, results_map)
    if not text:
        return ""
    return conclusion_of(text) or text
