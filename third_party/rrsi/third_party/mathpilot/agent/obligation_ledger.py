"""蓝图倒推链 → 审核侧「义务账本」（Obligation Ledger）。

============================================================
为什么需要这个模块（方向不对称）
============================================================

蓝图构建是**倒推**（backward chaining）：
    目标 G → 「要有 G 必须先有什么」→ 前提 P1/P2 → 对 P 递归 → 停在题面已给处。

审核链（4.1 自审 / 4.2 证伪）是**正推**：拿到 problem + reasoning + answer，
判断「这段推导讲得够不够、对不对」。两者信息严重不对称 ——

    审查者**不知道这道题应当建立哪些事实**，只能拿"一份完整证明的默认模板"
    去套。于是蓝图**有意省略**的环节（题面已给、或由上游节点建立）被读成
    「跳步」，答案解析得简略就被判死。

实测依据（p0b5_r5，2026-10-20）：49 条淘汰记录里有 15 条打中的是 gold 本身；
`skipped_step` 9 次淘汰打中 gold 5 次、`misread_requirement` 6 次打中 4 次。
而 `adversarial_verifier._ADVERSARY_TEMPLATE` 当时只喂了题目、解答、答案三样，
**一个字都没告诉审查者"这题的义务是什么"**。

============================================================
本模块做什么
============================================================

把倒推链**反着映射一遍**：从根目标出发，沿 `children`（= "为建立它所需的前提"）
做层序遍历，把每个必需节点翻成一条带编号的**审核义务** O1/O2/…，并记录

  · 它是为建立哪一条义务而被倒推出来的（`for_oid`）—— 这就是"必要性方向"；
  · 它应当产出什么（`produces`，蓝图已声明，可机械核对）；
  · 它的前提里哪些已由别的义务建立、哪些由题面直给（`deps` / `from_problem`）。

审核侧据此可做三件事：
  1. 清单外的内容**不构成本题缺陷**（蓝图没要求的，不许当"跳步"来算）；
  2. 已由上游义务建立的结论**不要求在本步重复推导**；
  3. 报错时必须指认落在哪一条义务上（`obligation_id`），指认不出 ⇒ 不成立。

⚠ 本模块**只读** `ctx.blueprint`，不改蓝图、不改判决口径的阈值，
  也不产出任何答案（蓝图本身禁止书写待求量，账本原样沿用其字段）。
"""

from typing import Optional

import re

# 台账字符上限：账本会拼进审核 prompt，过长会挤压待审查解答的窗口。
MAX_LEDGER_CHARS = 2000

# 单条字段截断（statement 才是判断依据，produces 只作核对锚点）
_MAX_STMT = 130
_MAX_PRODUCES = 80


def _relabel(text: str, mapping: dict) -> str:
    """把 statement 里引用的节点 id（`n1`/`n2`…）重标成义务编号（`O1`…）。

    蓝图的 `statement` 是按节点 id 写的（「由 n1 的构造与 n2 的不等式夹逼」），
    而账本重编号后清单里只有 O1/O2 —— 不重标的话模型会对不上，反而更容易误判。
    长 id 优先替换，避免 `n1` 命中 `n10` 的前缀；前后加边界，避免命中 `n` 变量。
    """
    if not text or not mapping:
        return text
    for nid in sorted(mapping, key=len, reverse=True):
        text = re.sub(
            r"(?<![A-Za-z0-9_])%s(?![0-9])" % re.escape(str(nid)),
            mapping[nid], text)
    return text


def _clip(text: str, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _node_map(bp: dict) -> dict:
    """{node_id: node_dict}；容忍 `nodes` 缺失 / 元素非 dict 的脏数据。"""
    out = {}
    for nd in (bp or {}).get("nodes") or []:
        if isinstance(nd, dict) and nd.get("id"):
            out[str(nd["id"])] = nd
    return out


def build_ledger(bp: Optional[dict]) -> dict:
    """把 `ctx.blueprint`（`BlueprintDAG.to_dict()` 的结果）翻成义务账本。

    返回空 dict 表示**无可用蓝图**（未走蓝图路径 / 解析失败 / 节点为空）——
    调用方必须据此回落到原有行为（不得因为"没有账本"就放宽任何判据）。
    """
    if not isinstance(bp, dict):
        return {}
    nodes = _node_map(bp)
    if not nodes:
        return {}

    root_id = str(bp.get("root_id") or "")
    if root_id not in nodes:
        # 兜底：无人认领为 child 的节点即根（多根时取节点表首个）
        claimed = set()
        for nd in nodes.values():
            for ch in nd.get("children") or []:
                claimed.add(str(ch))
        roots = [nid for nid in nodes if nid not in claimed]
        root_id = roots[0] if roots else next(iter(nodes))

    # 层序遍历 = 倒推顺序：先目标，再"为它所需的前提"，逐层向外。
    order, seen = [], set()
    queue = [root_id]
    parent = {}
    while queue:
        nid = queue.pop(0)
        if nid in seen or nid not in nodes:
            continue
        seen.add(nid)
        order.append(nid)
        for ch in nodes[nid].get("children") or []:
            ch = str(ch)
            if ch in nodes and ch not in seen:
                parent.setdefault(ch, nid)
                queue.append(ch)

    # 前瞻引理（anticipatory）不进求解链，也不是本题必须建立的义务。
    required = [nid for nid in order
                if str(nodes[nid].get("node_role") or "").strip().lower()
                != "anticipatory"]
    # 根 = 目标本身，单独以「目标 G」呈现，不占义务编号：
    #   否则渲染成「O1 建立：<目标>」，审查者会以为目标还要再被证明一遍。
    goal_nodes = [nid for nid in required if nid == root_id]
    required = [nid for nid in required if nid != root_id]
    if not required and not goal_nodes:
        return {}

    oid_of = {nid: "O%d" % (i + 1) for i, nid in enumerate(required)}
    _raw_merge = bp.get("merge_strategy") or ""
    _raw_term = nodes[root_id].get("produces") or ""
    # 节点 id → 呈现用编号（根节点记作目标 G），用于把 statement 内部的
    # 「由 n1 的…」重标成「由 O1 的…」，否则清单与正文的引用对不上。
    _relabel_map = dict(oid_of)
    _relabel_map[root_id] = "G"

    obligations = []
    for nid in required:
        nd = nodes[nid]
        deps = []
        from_problem = 0
        for dep in nd.get("inputs") or []:
            dep = str(dep)
            if dep in oid_of:
                deps.append(oid_of[dep])
            elif dep and dep not in nodes:
                # inputs 里出现非节点 id ⇒ 该前提源自题面（蓝图按约定不建节点）
                from_problem += 1
        _par = parent.get(nid)
        obligations.append({
            "oid": oid_of[nid],
            "node": nid,
            "statement": _clip(_relabel(nd.get("statement") or "", _relabel_map),
                               _MAX_STMT),
            "produces": _clip(_relabel(nd.get("produces") or "", _relabel_map),
                              _MAX_PRODUCES),
            # 指向根节点时记 "G"（目标），其余记义务编号
            "for_oid": ("G" if _par == root_id else oid_of.get(_par, "")),
            "deps": deps,
            "from_problem": from_problem,
            "tool": str(nd.get("tool") or ""),
        })

    return {
        "goal": _clip(_relabel(nodes[root_id].get("statement") or "", _relabel_map),
                      _MAX_STMT),
        "root_id": root_id,
        "obligations": obligations,
        "merge": _clip(_relabel(_raw_merge, _relabel_map), 200),
        "terminal": _clip(_relabel(_raw_term, _relabel_map), _MAX_PRODUCES),
        "node_to_oid": dict(oid_of),
        "n_required": len(required),
        # 前瞻引理全局计数（可能游离在 root 之外，故不能用 order 的差来算）
        "n_skipped": sum(
            1 for nd in nodes.values()
            if str(nd.get("node_role") or "").strip().lower() == "anticipatory"),
    }


def normalize_oid(ref: str, ledger: Optional[dict] = None) -> str:
    """把模型给的义务引用归一成 `O<n>`；无法归一返回 ""。

    容忍 `o3` / `O3` / `第3条` / `义务3` / `3` / `O-3` 等写法（实测模型会自造格式）。

    ★ 复合引用（`O5/O7`、`O1 和 O2`、`O1,O2`）取**第一个落在账本内的编号**。
      旧实现把所有数字拼成一个数（`O5/O7` → `"57"` → `O57`），
      而 `O57` 必然不在账本里 ⇒ 被判成「指认失败」放行。
      实测 2026-10-20 collab019_r1：4.1 三次放行全部源于此，
      其中 cid=0 随后被 4.2 跨模型以同一理由（下界假设 Natasha 不干扰 2×2 块）
      独立证伪 ⇒ 是真错被误放行，不是模型乱指认。

    语义边界（与调用点「只惩罚明确指认失败，不惩罚漏填」一致）：
      只要模型点到了账本里**任一条真实义务**，就算指认成立 ——
      复合里混进不存在的编号不推翻它；一条都对不上才返回 ""。
    """
    s = str(ref or "").strip()
    if not s:
        return ""
    if s.lower().strip() in ("none", "null", "n/a", "无", "不适用", "空"):
        return ""
    groups = re.findall(r"\d+", s)
    if not groups:
        return ""
    valid = set((ledger.get("node_to_oid") or {}).values()) if ledger else None
    for g in groups:
        oid = "O%d" % int(g)
        if valid is None or oid in valid:
            return oid
    return ""


def has_obligation(ledger: Optional[dict], ref: str) -> bool:
    """`ref` 是否指向账本内真实存在的义务（空账本恒 False）。"""
    if not ledger:
        return False
    return bool(normalize_oid(ref, ledger))


def render_for_audit(ledger: Optional[dict]) -> str:
    """渲染成可注入审核 prompt 的文本块；无账本返回 ""（调用方据此不加这一段）。"""
    if not ledger or not ledger.get("obligations"):
        return ""

    lines = ["【本题应当建立的义务（由蓝图倒推得出，编号供你引用）】"]
    if ledger.get("goal"):
        lines.append("目标 G：%s" % ledger["goal"])

    for ob in ledger["obligations"]:
        head = ob["oid"]
        if ob.get("for_oid"):
            head += "（为建立 %s 所需）" % ob["for_oid"]
        seg = ["%s 建立：%s" % (head, ob["statement"] or "（未描述）")]
        if ob.get("produces"):
            seg.append("应产出：%s" % ob["produces"])
        if ob.get("deps"):
            seg.append("前提已由 %s 建立" % "/".join(ob["deps"]))
        if ob.get("from_problem"):
            seg.append("另有 %d 项前提题面已给" % ob["from_problem"])
        lines.append("· " + "；".join(seg))

    if ledger.get("merge"):
        lines.append("归并方式：%s" % ledger["merge"])
    if ledger.get("terminal"):
        lines.append("终答形态：%s" % ledger["terminal"])

    lines.append(
        "判定纪律：① 上表未要求的环节，不得作为「跳步」或「题意偏差」计入；"
        "② 某条结论若已由上表某条义务建立，不得要求在本步重复推导；"
        "③ 你报任何错误都必须给出 `obligation_id`，指向它发生在哪条义务上；"
        "指认不出对应义务的错误一律视为对本题论证链的误读。")

    text = "\n".join(lines)
    if len(text) > MAX_LEDGER_CHARS:
        text = text[: MAX_LEDGER_CHARS - 20] + "\n…（账本已截断）"
    return text
