# -*- coding: utf-8 -*-
"""子目标求解器的 parse 职责簇
（2026-10-08 批3：从 agent/sub_goal_solver.py 逐字搬出 3 个方法）。

搬出后为模块级函数（实例方法保留 `self` 首参，方法体逐字不变）；
跨模块引用原文件模块级符号者（含 `SubGoalSolverAgent` 与其常量/函数）
在**函数体内惰性 import**（避免循环导入）—— 下表已显式列出。
原文件类内保留 1 行跳板（`_x = _sgs_parse._x`）⇒ `self._x(...)` /
`SubGoalSolverAgent._x(...)` 契约逐字不变（同 verifier__review.py 批6）。
"""
from __future__ import annotations

import json
import logging
import re


logger = logging.getLogger("MathPilot")


def _extract_json(text: str) -> dict | None:
    r"""从 LLM 输出中提取 JSON 对象。

        v2.6 修复：原实现对最外层一对花括号到另一对花括号做贪婪匹配，会被 LaTeX /
        中文解释 / 嵌套 JSON 干扰（LaTeX 里的左花括号、右花括号、解释文字里的成对
        花括号都会被吃进 JSON 段导致解析失败）。改为**平衡括号匹配**：从每个左
        花括号出发，用栈找匹配的右花括号，并正确处理字符串内花括号与反斜杠转义。
        """
    if not text:
        return None
    # 1) 优先 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass  # 代码块不合法，回退到平衡括号匹配

    # 2) 平衡括号匹配：扫描每个 {，用深度栈找到对应的 }。
    #    关键：正确处理字符串字面量（跳过字符串内的 { 和 }）
    def _try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 尝试修复常见错误：尾随逗号
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None

    for i, c in enumerate(text):
        if c != '{':
            continue
        depth, j, in_str, esc = 1, i + 1, False, False
        while j < len(text):
            ch = text[j]
            if esc:
                esc = False
            elif in_str and ch == '\\':
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        parsed = _try_parse(text[i:j + 1])
                        if parsed is not None:
                            return parsed
                        break  # 该 { 不是合法 JSON 起点，继续找下一个
            j += 1
    return None


def _calc_kind_of(sg_type: str, title: str, expected_output: str) -> str:
    from .sub_goal_solver import SubGoalSolverAgent
    blob = f"{title} {expected_output}".strip()
    if sg_type == "compute":
        if any(k in blob for k in SubGoalSolverAgent._TERM_ACTION_KW):
            return "terminal"
        eo = (expected_output or "").strip()
        # expected_output 为短数值/表达式形态（数字/负号/等号/左括号开头）
        if eo and len(eo) <= 60 and (eo[0].isdigit() or eo[0] in "-=("):
            return "terminal"
    return "inline"


def _parse_subgoal_plan(raw: dict, max_subgoals: int = 6) -> list[dict] | None:
    """验证并规范化子目标规划，返回按拓扑序排列的子目标列表。

        v2.6 宽容性增强：LLM 实际输出常省略字段（如用 step/name/task 代替
        id/title/description）、或把列表字段命名为 plan/steps/subproblems/tasks
        等。本方法兼容多种字段名，单个条目缺失字段时也能构造出可用条目，
        而不再直接放弃。
        """
    from .sub_goal_solver import SubGoalSolverAgent
    # 1) 宽容多种列表字段名
    subgoals = None
    for key in ("subgoals", "subproblems", "sub_problems",
                "plan", "steps", "tasks", "items"):
        v = raw.get(key)
        if isinstance(v, list) and v:
            subgoals = v
            break
    if subgoals is None:
        # 兜底：找任何非空 list[dict] 字段
        for v in raw.values():
            if (isinstance(v, list) and v
                    and all(isinstance(x, dict) for x in v)):
                subgoals = v
                break
    if subgoals is None:
        return None
    # 2026-09-03 老师：子目标是**简化求解**的，不是把题目拆得更复杂。
    # 原上限 10（deep 实测 8 步 × 110s = 882s 吃掉 73% 时间）。
    # 改为可配置 max_subgoals（默认 6）：少而精，每步都是真正必要的
    # 简化步骤；省下的时间留给验证器与 Lean 闸门（老师核心诉求）。
    # 注意：这是"规划时少拆几步"，**不是执行到一半强制中断**——
    # 已规划的子目标一定会跑完（老师："强制结束就等于错误"）。
    # 2026-09-03 审核修复：原实现在 @staticmethod 里写 `self.config` /
    # `self.record(ctx, ...)` → 每次调用必抛 NameError（本方法无 self/ctx），
    # 且该分支无条件执行 = 子目标规划 100% 崩溃。改为**上限走参数**，
    # 调用方从 config 读取后传入，静态方法保持无状态（测试可直接调用）。
    # 2026-09-15：改用与 blueprint 路径**共用**的 cap_subgoals_keep_last。
    # 语义统一（None→默认 / <=0→不截断），且**保留收尾子目标**——
    # 原 `subgoals[:_max_sg]` 盲截拓扑序尾部，会丢掉"合并结论/得出最终
    # 答案"那一步（prompts/sub_goal.py 明写最后一个是合并结论），
    # 导致前面所有子目标白解。两条路径必须同源，故抽成公共函数。
    from .blueprint_planner import cap_subgoals_keep_last
    _n_before = len(subgoals)
    subgoals, _dropped_sg = cap_subgoals_keep_last(subgoals, max_subgoals)
    if _dropped_sg:
        logger.info("SubGoal plan: 子目标 %d 个，截断 %d 个"
                    "（保留收尾子目标；简化求解，非中断）",
                    _n_before, len(_dropped_sg))

    valid_types = {"compute", "prove", "derive", "verify"}
    seen_ids = set()
    parsed = []
    for sg in subgoals:
        # 宽容 id 字段名：id / step / index / number
        raw_id = sg.get("id", sg.get("step", sg.get("index",
                      sg.get("number", len(parsed) + 1))))
        try:
            sg_id = int(raw_id)
        except (TypeError, ValueError):
            sg_id = len(parsed) + 1
        if sg_id in seen_ids:
            continue
        seen_ids.add(sg_id)

        # 宽容 title/description/expected_output/depends_on/type 字段名
        title = str(sg.get("title",
                   sg.get("name",
                   sg.get("step_name",
                   sg.get("task_name", f"子目标{sg_id}")))))
        description = str(sg.get("description",
                        sg.get("task",
                        sg.get("content",
                        sg.get("detail", "")))))
        sg_type = sg.get("type", sg.get("kind", "compute"))
        if sg_type not in valid_types:
            sg_type = "compute"
        deps_raw = sg.get("depends_on",
                   sg.get("deps",
                   sg.get("dependencies", [])))
        if not isinstance(deps_raw, list):
            deps_raw = []
        expected_output = str(sg.get("expected_output",
                               sg.get("output",
                               sg.get("expected",
                               sg.get("result", ""))))).strip()
        # ★ 2026-09-23：把"填了但等于没填"的空话也判为缺失。
        #   实测 96% 为空；提示词虽已要求该字段，但模型常写"一个结果"这类空话
        #   ⇒ 只判 `== ""` 会低估缺口。此标记随 subgoal_trace 一起导出，
        #   供 orchestrator 的 `subgoal_delivery` 统计（"做成了没"的判据）。
        _eo_filler = ("", "结果", "一个结果", "期望的输出形式", "数值/表达式/证明",
                      "数值", "表达式", "无", "n/a", "na", "待定", "答案")
        eo_missing = (expected_output in _eo_filler) or (len(expected_output) < 6)
        parsed.append({
            "id": sg_id,
            "title": title,
            "description": description,
            "type": sg_type,
            "depends_on": [d for d in deps_raw
                           if isinstance(d, int) and d in seen_ids],
            "expected_output": expected_output,
            "expected_output_missing": bool(eo_missing),
            # P2（2026-09-09）：计算/推理类型化标签（宁 inline 勿误 terminal）
            "calc_kind": SubGoalSolverAgent._calc_kind_of(
                sg_type, title, expected_output),
            "result": "",
        })
    return parsed if parsed else None
