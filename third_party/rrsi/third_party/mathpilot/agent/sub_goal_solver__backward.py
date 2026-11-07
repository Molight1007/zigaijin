# -*- coding: utf-8 -*-
"""子目标求解期的「动态倒推」——把蓝图从**静态图**变成**可增量生长**的图。

用户提议（2026-10-15）：「2 与 3 之间建一个环——蓝图把子蓝图生成完之后，3 从小
到大证明，每证明一次就填回作为条件。」核实后（任务 #12）：「从小到大 + 回填」
**已是现状**（`to_subgoal_plan` 的展开序即拓扑序；`solve_subgoals` 逐步回填
`results_map` 并按 `depends_on` 注入），真正缺的是**环**——蓝图在
`to_subgoal_plan` 之后被冻结，求解期只读。本模块补的就是这个环。

★ 与生成期倒推的分工（两者同源、缺一不可）：
  · **生成期**（`prompts/blueprint.py` 构造原则第 2/3 条）：一次性倒推，判据只能
    来自**题面**——「这个前提题面给没给」。它看不到求解过程中才暴露的缺口。
  · **求解期**（本模块）：某子目标**实际解不动**且判定为缺前提 ⇒ 现场再倒推一层，
    把前提立为新子目标、就地求解（前提自己缺前提时**递归同法**）、回填、重解原目标。

此前这种子目标只有两条出路：`sg_adaptive_recover` 重做一次 → 仍失败 →
写 `ctx.revise_feedback` 交上游；而**上游 replan 已于 2026-10-15 整块删除**
（`enable_dag_replan` / `dag_replan_gate`）⇒ 该反馈**无人消费** ⇒ 子目标以占位符
放行、merge 拿残缺链合成 ⇒ 必错。本模块是那条断裂链的替代品。

★★ 设计取舍：**递归内联，绝不改动全局 `subgoals` 列表**
  为什么不能"插进 subgoals"：`subgoals` 的 `id` 是 1-based 序号，`results_map`
  以 id 为键、`depends_on` / `inputs` / `inputs_use` 全按 id 引用（`to_subgoal_plan`
  已把位置索引 `_pos` 烤进这些字段）。中途插入会打乱编号 ⇒ **所有后续子目标的
  依赖边集体错位**（且错得静默）。
  故动态子目标用**独立 id 空间**（`<父id>B<n>`，如 `3B1`）：只在 `results_map` 里
  **新增键**、不改任何既有键 ⇒ 零错位。语义上也本该如此——动态前提是**父目标的
  实现细节**（为解 T 而临时补的料），兄弟节点依赖的是 T 的**结论**、不是 T 的**前提**。

★ 深度上限（`subgoal_backward_max_depth`）：防「缺前提 → 再倒推 → 又缺前提」
  无限套娃。超限即停、保留原结果（绝不阻断 merge）。

★ 与 `sg_adaptive_recover` 的职责边界：
  · recover = **不拆图**的恢复（重做该子目标 / 交上游重规划），便宜，先跑；
  · 本模块 = **拆图**的恢复（现场补前提），贵，放在 recover **之后**兜底。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("MathPilot")

# `<need_premise>…</need_premise>`：求解侧**显式**声明"缺前提"的协议标记。
# 与 `SUBGOAL_STEP_USER_TEMPLATE` 末尾的引导一一对应。
_NEED_PREMISE_RE = re.compile(r"<need_premise>(.*?)</need_premise>", re.S | re.I)

# 失败关键词兜底（与 `sg_adaptive_recover` 的 `_plan_err` **同源同口径**）。
# ⚠ 两处必须一起改：口径漂移会让"缺前提"在两条路径上判定不一致。
_PLAN_ERR_KEYS = ("依赖", "前提", "缺少", "不成立", "无法确定", "无法判定", "条件不足")

# 允许的 tool 白名单（与蓝图 `tool` 字段、`sg_tool_exec` 的消费口径一致）
_TOOLS = ("llm", "sympy", "numeric", "lean")


def _first_plan_err_sentence(text: str) -> str:
    """从失败文本里摘出**含缺前提关键词的那一句**（供倒推提示词当输入）。

    不整段回传的理由：失败文本常含 Thinking/推理草稿，整段进提示词既费 token、
    又容易把无关噪声当成"缺失的前提"。
    """
    for _seg in re.split(r"[。\n；;]", str(text or "")):
        _s = _seg.strip()
        if _s and any(_k in _s for _k in _PLAN_ERR_KEYS):
            return _s[:200]
    return "（模型未显式标注；按其失败描述推定：前提条件不足）"


def sg_missing_premise(step_result: str, failed: bool) -> str:
    """判定「缺前提」并返回缺的前提描述；**空串 = 不触发**动态倒推。

    两条判据：
      ① **显式协议**：结果里出现 `<need_premise>…</need_premise>`。
         模型主动声明 ⇒ **不要求**结果看起来失败（它可能把话说得很完整，
         但明确说了卡在缺前提上）。
      ② **关键词兜底**：结果**被判失败**（占位 / `[子目标` 标记 / 未产出）
         **且**含缺前提类关键词。
         ⚠ `failed` 这道门槛是必须的：正常结果里偶然出现「前提」「不成立」
         二字（如「该假设不成立，故取另一支」）极常见，不设门槛会大面积误触发。

    两道闸门之后，倒推结论还要再经**一次 LLM 判定**（`_ask_premise` 的 `ok=false`
    即放弃）⇒ 判据可以放宽，不会直接造成无谓拆解。
    """
    text = str(step_result or "")
    if not text:
        return ""
    m = _NEED_PREMISE_RE.search(text)
    if m:
        _p = m.group(1).strip()
        if _p:
            return _p
    if failed and any(_k in text for _k in _PLAN_ERR_KEYS):
        return _first_plan_err_sentence(text)
    return ""


def _backward_messages(ctx, sg: dict, missing: str, prev_results: str):
    """构造倒推分析的 (system, user) 消息。"""
    from prompts.sub_goal import (
        SUBGOAL_BACKWARD_PREMISE_SYSTEM,
        SUBGOAL_BACKWARD_PREMISE_USER_TEMPLATE,
    )
    user = SUBGOAL_BACKWARD_PREMISE_USER_TEMPLATE.format(
        problem=(ctx.problem or "")[:4000],
        subgoal_id=sg.get("id"),
        subgoal_title=str(sg.get("title") or ""),
        subgoal_description=str(sg.get("description") or ""),
        subgoal_expected_output=str(sg.get("expected_output") or ""),
        failed_result=str(missing)[:1200],
        prev_results=(str(prev_results) if prev_results else "（无）")[:2000],
    )
    return SUBGOAL_BACKWARD_PREMISE_SYSTEM, user


def _ask_premise(self, ctx, sg: dict, missing: str, prev_results: str) -> dict:
    """调 LLM 现场倒推一层。

    返回 `{"ok": True, "premise": …, "why": …, "tool": …}`
    或 `{"ok": False, "reason": …}`（**永不抛异常**，失败即放弃倒推）。
    """
    try:
        system, user = _backward_messages(ctx, sg, missing, prev_results)
        resp = self.llm(
            ctx,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            0.2, 8192,
        )
    except Exception as e:  # noqa: BLE001  LLM 故障不该拖垮求解
        return {"ok": False, "reason": f"倒推调用异常: {str(e)[:100]}"}
    if not resp:
        return {"ok": False, "reason": "LLM 无响应"}
    try:
        from .blueprint_planner import extract_json
    except Exception:  # noqa: BLE001
        from agent.blueprint_planner import extract_json  # type: ignore
    raw = None
    try:
        raw = extract_json(resp)
    except Exception:  # noqa: BLE001
        raw = None
    if not isinstance(raw, dict):
        return {"ok": False, "reason": "倒推输出非 JSON"}
    if not raw.get("ok"):
        return {"ok": False,
                "reason": str(raw.get("reason") or "模型判定：并不缺前提")}
    _p = str(raw.get("premise") or "").strip()
    if not _p:
        return {"ok": False, "reason": "模型未给出前提内容"}
    _tool = str(raw.get("tool") or "llm").strip().lower()
    if _tool not in _TOOLS:
        _tool = "llm"
    return {"ok": True, "premise": _p,
            "why": str(raw.get("why") or "").strip(), "tool": _tool}


def _new_backward_id(ctx, parent_id) -> str:
    """动态子目标 id：`<父id>B<n>`。

    ★ **独立 id 空间**——不动任何既有 id ⇒ `results_map` / `depends_on` 零错位
    （见模块文档的设计取舍）。计数器挂 `ctx`，跨父目标唯一；递归时 id 会叠加
    （`3B1` 再倒推得 `3B1B2`），仍天然唯一。
    """
    _seq = int(getattr(ctx, "_sg_backward_seq", 0) or 0) + 1
    try:
        setattr(ctx, "_sg_backward_seq", _seq)
    except Exception:  # noqa: BLE001  个别 ctx 实现可能禁写
        pass
    return f"{parent_id}B{_seq}"


def _make_premise_sg(ctx, sg: dict, premise: str, tool: str, why: str = "") -> dict:
    """把倒推出的前提包成一个**与 `to_subgoal_plan` 产物同构**的子目标 dict。

    字段必须齐 `id/title/description/type/depends_on/expected_output`——
    `_solve_subgoal` → `_prepare_step_prompt` 与 `_build_trace_item` 都是**直接索引**
    这些键（不是 `.get`），缺一个就 KeyError。
    """
    _pid = _new_backward_id(ctx, sg.get("id"))
    return {
        "id": _pid,
        "title": f"子目标{_pid}: {str(premise)[:60]}",
        "description": str(premise),
        "type": "derive",
        # 动态前提的输入就是"已建立的上下文"，没有跨子目标的结构依赖 ⇒ 空。
        "depends_on": [],
        "expected_output": "",
        "inputs": [],
        "inputs_raw": [],
        "method": "",
        "tool": tool,
        "inputs_use": {},
        "inputs_from": [],
        "tool_verdict": "",
        "result": "",
        # ★ 归因标记：进 trace / diag 后可一眼看出哪些子目标是求解期补入的。
        "dynamic": True,
        "backward_from": sg.get("id"),
        "backward_why": why,
    }


def _render_backward_block(items: list) -> str:
    """把动态前提的结论渲染成「新增已知条件」块（注入重解提示词）。"""
    if not items:
        return ""
    from prompts.sub_goal import SUBGOAL_BACKWARD_BLOCK_TEMPLATE
    lines = []
    for it in items:
        lines.append("  · 前提 #%s「%s」：%s"
                     % (it.get("id"), str(it.get("title") or "")[:40],
                        str(it.get("result") or "")[:400]))
    return SUBGOAL_BACKWARD_BLOCK_TEMPLATE.format(items="\n".join(lines))


def sg_dynamic_backward(self, ctx, sg: dict, step_result: str,
                        plan_summary: str, prev_results: str,
                        results_map: dict = None, depth: int = 1) -> str:
    """求解期动态倒推主入口。返回（可能被重解替换的）`step_result`。

    流程：判定缺前提 → LLM 倒推一层 → 递归求解该前提（自己缺前提时同法）→
    回填 `results_map` → 把前提结论作为「新增已知条件」注入 → **重解原目标**。

    调用位置：`solve_subgoals` 里 `sg_adaptive_recover` **之后**（先用便宜的不拆图
    恢复；仍失败且指向缺前提，才现场拆一层）。

    任何一步失败都**保留原结果**（绝不阻断 merge）——本项目铁律：兜底出答案优先
    于流程完美。
    """
    # ① 开关与上限
    if not getattr(self.config, "subgoal_dynamic_backward", True):
        return step_result
    _max = int(getattr(self.config, "subgoal_backward_max_depth", 2) or 0)
    if _max <= 0 or depth > _max:
        return step_result
    if ctx.is_time_critical():
        return step_result
    if not step_result:
        return step_result

    # ② 缺前提判定（口径与 sg_adaptive_recover 的失败判据一致，见模块文档）
    _failed = bool(
        self._looks_placeholder(step_result)
        or step_result.startswith("[子目标")
        or "未产出有效结论" in step_result
        or "仍为空转占位" in step_result)
    missing = sg_missing_premise(step_result, _failed)
    if not missing:
        return step_result

    self.record(ctx, "subgoal_backward",
                f"子目标 #{sg['id']}「{sg['title']}」判定缺前提（depth={depth}/{_max}）："
                f"{missing[:100]} → 现场倒推一层")

    # ③ 倒推
    got = _ask_premise(self, ctx, sg, missing, prev_results)
    if not got.get("ok"):
        self.record(ctx, "subgoal_backward",
                    f"子目标 #{sg['id']} 倒推未产出有效前提"
                    f"（{str(got.get('reason') or '')[:80]}）⇒ 维持原结果")
        return step_result

    p_sg = _make_premise_sg(ctx, sg, got["premise"], got.get("tool", "llm"),
                            got.get("why", ""))
    self.record(ctx, "subgoal_backward",
                f"子目标 #{sg['id']} 倒推出前提 #{p_sg['id']}「{p_sg['title']}」"
                f"（tool={p_sg['tool']}）：{str(got.get('why') or '')[:100]}")

    # ④ 求前提（★ 递归：前提自己也可能缺前提 ⇒ 同法再倒推一层，受 depth 封顶）
    _prev_for_p = prev_results or ""
    try:
        _p_res = self._solve_subgoal(ctx, p_sg, plan_summary, _prev_for_p)
        _p_res = sg_dynamic_backward(self, ctx, p_sg, _p_res, plan_summary,
                                     _prev_for_p, results_map, depth + 1)
        _p_res = self._strip_check_tags(_p_res)
    except Exception as e:  # noqa: BLE001  前提求解失败 ⇒ 放弃倒推、保留原结果
        self.record(ctx, "subgoal_backward",
                    f"子目标 #{sg['id']} 前提 #{p_sg['id']} 求解异常"
                    f"（{str(e)[:80]}）⇒ 维持原结果")
        return step_result
    # ★ 2026-10-20：唯一写入口（两容器同值）。手写两行就意味着"漏写一处"
    #   的可能 —— 动态子目标同样要进 reasoning 文本，否则又造一个空壳。
    from .subgoal_result import write_step
    write_step(p_sg, results_map, _p_res)
    try:
        ctx.subgoal_trace.append(self._build_trace_item(p_sg, _p_res))
    except Exception:  # noqa: BLE001  trace 写入失败不该影响求解
        pass

    # ⑤ 回填并重解原目标：前提结论以「本步新增前提」块注入（放在前序结果**之前**，
    #    与 `_render_succession_block` 的位置口径一致——承接前提要先被读到）。
    _block = _render_backward_block([p_sg])
    _prev2 = (_block + "\n" + (prev_results or "")) if prev_results else _block
    try:
        _again = self._solve_subgoal(
            ctx, sg, plan_summary, _prev2,
            extra_hint=("该子目标上一次因缺少前提而失败；现已通过**倒推**把缺失的"
                        "前提 #%s 求解出来，并作为【本步新增前提】给出。"
                        "请据此重新完成本步。" % p_sg["id"]))
    except Exception as e:  # noqa: BLE001
        self.record(ctx, "subgoal_backward",
                    f"子目标 #{sg['id']} 回填后重解异常（{str(e)[:80]}）⇒ 维持原结果")
        return step_result

    if _again and not _again.startswith("[子目标"):
        self.record(ctx, "subgoal_backward",
                    f"子目标 #{sg['id']} 回填前提 #{p_sg['id']} 后重解完成")
        return _again
    self.record(ctx, "subgoal_backward",
                f"子目标 #{sg['id']} 回填后重解仍未产出 ⇒ 维持原结果")
    return step_result
