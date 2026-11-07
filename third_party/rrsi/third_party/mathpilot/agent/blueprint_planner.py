"""
BlueprintPlanner —— LEAP Stage 1：Blueprint DAG 分解（#27）
==============================================================

将数学问题分解为 AND-OR 有向无环图（DAG），作为子目标求解的依据：

- **AND 节点**：所有子节点都必须证明/求解（分解关系，取与）
- **OR 节点**：任一子节点证明成功即可（策略分支，取或）
- **叶子节点**：无子节点的原子子目标（"可直接证明"粒度）

与 SubGoalSolver 的关系：
- SubGoalSolver 原有流程：LLM 一步生成"有序子目标列表"（线性切块）
- 本模块：先由 LLM 生成 AND-OR DAG（依赖驱动分解），再转成
  SubGoalSolver 兼容的子目标序列（拓扑序 + 依赖关系）

设计参考：
- LEAP 论文（Po-Nien Kung et al. 2026）：Blueprint DAG 分解阶段
- 老师 8/26 要求 #27：依赖驱动的分解，而非简单切块

对外接口：
- BlueprintNode / BlueprintDAG：数据结构 + 校验 + 转换（纯逻辑，可单测）
- BlueprintPlannerAgent.generate_blueprint(ctx)：LLM 生成 DAG 并校验
"""

import json
import logging
import re
from typing import Optional

from .base import BaseAgent, TaskContext

# 2026-10-08 批 D-bp1：DAG 数据结构/图算法/子目标截断簇已逐字搬到
# agent/blueprint_planner__dag.py。此处**跳板式 re-export**，保
# `from agent.blueprint_planner import <X>` 零改动 + **同一对象 `is`**。
# ⚠ 勿按「未使用」清理（契约面；`_flatten_produces` 供本文件 parse_blueprint 用）。
from .blueprint_planner__dag import (  # noqa: F401
    MAX_DAG_NODES,
    MAX_SUBGOALS,
    cap_subgoals_keep_last,
    _flatten_produces,
    _as_id_list,
    _normalize_tool,      # 2026-10-15：工具指派白名单归一（parse_blueprint 用）
    _flatten_formal_sig,  # 2026-10-15：Lean 形式化陈述归一（tool=lean 叶子用）
    _input_use_map,       # 2026-10-15：inputs 的 use（承接语义）抽取
    BlueprintNode,
    BlueprintDAG,
)

from prompts.blueprint import (
    BLUEPRINT_DAG_SYSTEM,
    BLUEPRINT_DAG_USER_TEMPLATE,
)
from utils.prefill import prefill_messages, stitch

logger = logging.getLogger("MathPilot")


# ============================================================
# JSON 提取（与 SubGoalSolverAgent 同款平衡括号算法）
# ============================================================

def extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出中提取 JSON 对象（平衡括号匹配，防 LaTeX 花括号干扰）。

    增强容错：
    - 优先 ```json 代码块
    - 逐 { 平衡括号匹配
    - 失败后兜底：取全文中最长的一段可解析 JSON（处理夹杂说明文字的场景）
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    def _try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
            except json.JSONDecodeError:
                return None

    best, best_len = None, 0
    # 关键改动（2026-08-28）：不再「命中第一个平衡片段就返回」。
    # 实测 Intern 会输出形如
    #   {"domain": "...", "key_constraints": [...]}, "root_id": "g", "nodes": [...]
    # 这种把嵌套字段摊平、并提前闭合花括号的内容——此时第一个平衡片段
    # 只是前面那段元数据碎片，root_id / nodes 全被丢掉，
    # 表现为「Blueprint: DAG 结构非法」。改为收集全部候选并取最长者：
    # 完整 DAG 的长度远大于任何前言碎片，这样能把正确主体捞回来。
    best_parsed, best_parsed_len = None, 0
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
                        if parsed is not None and (j - i) > best_parsed_len:
                            best_parsed, best_parsed_len = parsed, j - i
                        elif parsed is None and (j - i) > best_len:
                            # 解析失败也记录最长候选（兜底用）
                            best, best_len = text[i:j + 1], j - i
                        break
            j += 1
    if best_parsed is not None:
        return best_parsed
    # 兜底：最长候选 + 容错（去注释/去多余逗号后重试）
    if best:
        cleaned = re.sub(r"//[^\n]*", "", best)
        parsed = _try_parse(cleaned)
        if parsed is not None:
            return parsed
    return None


def _subtree_regen_enabled() -> bool:
    """读「蓝图自审用局部子树重写」开关（``blueprint_subtree_regen``，默认 True）。

    ★ 2026-10-22：`regenerate_subtree`（只重写被拒节点所在的 LCA 子树、其余保留）
      此前实现完整但**全仓无调用**——历史备份里 `sub_goal_solver.py` 有三处，
      重构时丢了接线。接回后：拒绝率高时由它自己判断 LCA 是否触根，触根则退化为
      整树重写，因此与旧行为兼容。=0 回到纯整树重写（A/B 对照）。
    """
    try:
        from .switch_registry import get_bool as _g
        return bool(_g("blueprint_subtree_regen"))
    except Exception:  # noqa: BLE001  注册表不可用 ⇒ 新语义
        return True


def _effective_problem(ctx) -> str:
    """题目文本（含「重写通道」注入）。★ 2026-10-14（五道链定案）：

    仅当 ``ctx.rewrite_context`` 存在（4.6_rewrite 池空重写时写入「错误情况 +
    诊断」）才在**"题目"占位符内部**追加；默认无该字段 ⇒ 逐字返回
    ``ctx.problem``（行为零变化）。注入位置遵循 `objective_injection` 纪律：
    拼在题目内部，不追加到提示词末尾。
    """
    p = getattr(ctx, "problem", "") or ""
    _rw = getattr(ctx, "rewrite_context", None)
    if _rw:
        p = p + "\n\n" + str(_rw)[:2500]
    return p


def parse_blueprint(raw: dict) -> Optional[BlueprintDAG]:
    """从 LLM 输出 dict 构造 BlueprintDAG（宽容字段名），非法返回 None。

    兼容的字段名：
    - root: root / root_id / root_node
    - nodes: nodes / blueprint / dag / subgoals / children_map
    - 节点 type: type / kind / node_type；children: children / deps / subgoals / child_ids
    """
    if not isinstance(raw, dict):
        return None
    root_id = str(raw.get("root_id", raw.get("root",
                  raw.get("root_node", ""))) or "")
    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list):
        for k in ("blueprint", "dag", "subgoals", "children_map"):
            v = raw.get(k)
            if isinstance(v, list) and v:
                nodes_raw = v
                break
    if not isinstance(nodes_raw, list) or not nodes_raw:
        return None

    nodes = {}
    for nd in nodes_raw:
        if not isinstance(nd, dict):
            continue
        nid = str(nd.get("id", nd.get("node_id", nd.get("name", ""))) or "")
        if not nid:
            continue
        ntype = str(nd.get("type", nd.get("kind",
                    nd.get("node_type", "and")))).lower()
        if ntype not in ("and", "or"):
            ntype = "and"
        statement = str(nd.get("statement",
                       nd.get("description",
                       nd.get("task", nd.get("content", "")))))
        children = nd.get("children", nd.get("deps",
                    nd.get("subgoals", nd.get("child_ids", []))))
        if not isinstance(children, list):
            children = []
        children = [str(c) for c in children]
        nodes[nid] = BlueprintNode(
            id=nid,
            node_type=ntype,
            statement=statement,
            children=children,
            rationale=str(nd.get("rationale", nd.get("reason", ""))),
            # ★ 2026-10-06（子目标交付重写 A）：读取模型声明的**产出物**
            #   （prompts/blueprint.py:89-101/:179 一直要求 `produces`，但此前
            #   解析器不读 ⇒ 静默丢弃 ⇒ to_subgoal_plan 的 expected_output 恒空）。
            #   兼容别名 `output` / `expected_output`，且把 `produces` 里模型
            #   偶尔写成 dict（{object, use}）的形态摊平为文本。
            inputs=_as_id_list(nd.get("inputs", nd.get("needs", []))),
            method=str(nd.get("method", nd.get("how", "")) or ""),
            # ★ 2026-10-15（用户要求「为每一个子目标设定工具，证明就用 lean」）：
            #   读取模型声明的 `tool`（白名单归一，见 `_normalize_tool`；
            #   无法识别 ⇒ "" = 未指派，下游不加工具指令）。
            #   此前 `prompts/blueprint.py` 还没要求该字段，本次同步补上解析。
            tool=_normalize_tool(nd.get("tool", nd.get("use_tool", ""))),
            # ★ 2026-10-15（用户：「把 lean 内化到蓝图构建里面」）：
            #   读取 `tool == "lean"` 节点由蓝图给出的 **Lean 形式化陈述**
            #   （签名部分，见 `BlueprintNode.formal_sig`）。非 lean 节点忽略该键
            #   （模型可能在其它节点上乱写，一律丢弃，避免下游误注入）。
            #   兼容别名 `lean_sig` / `signature`。空值 = 形式化失败 ⇒ 下游回落。
            formal_sig=(_flatten_formal_sig(
                nd.get("formal_sig", nd.get("lean_sig",
                                           nd.get("signature", ""))))
                if _normalize_tool(nd.get("tool", nd.get("use_tool", ""))) == "lean"
                else ""),
            # ★ 2026-10-15（用户要求「前一个子目标作为后一个的条件」）：
            #   `use` 是承接语义的唯一载体（`_as_id_list` 会把它丢掉），单独抽成
            #   {上游 id: 用途} 一并落到节点上。
            #   `_input_use_map` 同时吃两种形态：提示词要求的 `inputs[].use` 列表，
            #   以及偶尔出现的 `inputs_use` 字典（`to_dict` 回灌形态）。
            inputs_use=_input_use_map(
                nd.get("inputs_use")
                if nd.get("inputs_use")
                else nd.get("inputs", nd.get("needs", []))),
            produces=_flatten_produces(
                nd.get("produces", nd.get("output",
                       nd.get("expected_output", "")))),
            # 2026-10-04 修（fix-a1）：原实现**不读 node_role/role** ⇒ 所有节点都取
            # 默认 "required" ⇒ 下游 to_subgoal_plan 收集的 anticipatory 恒为空 ⇒
            # sub_goal_solver._inject_anticipatory_lemmas 永远拿到 []（机制空转）。
            # prompts/blueprint.py:130/187/222 明确要求前瞻引理节点写 "role": "anticipatory"。
            # 与同文件 BlueprintDAG.from_dict()（:526）写法完全一致。
            node_role=nd.get("node_role", nd.get("role", "required")) or "required",
        )
    if not nodes:
        return None
    if not root_id:
        # 兜底：取"根"特征——没有被任何节点引用的节点；多个则取第一个
        referenced = {c for n in nodes.values() for c in n.children}
        candidates = [nid for nid in nodes if nid not in referenced]
        root_id = candidates[0] if candidates else next(iter(nodes))

    return BlueprintDAG(
        nodes=nodes,
        root_id=root_id,
        merge_strategy=str(raw.get("merge_strategy", "")),
    )


# ============================================================
# BlueprintPlannerAgent
# ============================================================

class BlueprintPlannerAgent(BaseAgent):
    """LEAP Stage 1：生成并校验 Blueprint DAG（#27）。"""

    name = "BlueprintPlanner"

    def run(self, ctx: TaskContext) -> TaskContext:
        """生成 Blueprint DAG 并转为子目标规划（写入 ctx.blueprint_plan）。

        作为独立 Agent 被调用时使用；SubGoalSolver 内部直接调
        generate_blueprint() 更高效（避免重复实例化）。

        ⚠ 前瞻引理（node_role="anticipatory"）**不在此处注入引理记忆**：
        写入记忆需要走 SubGoalSolver._inject_anticipatory_lemmas 的领域路由
        （_use_lemma），本 Agent 不持有该规则。前瞻引理随
        ctx.blueprint_plan["anticipatory_lemmas"] 一并下传，由 SubGoalSolver
        消费；此处只记录条数，便于 trace 核对是否漏传。
        """
        dag = self.generate_blueprint(ctx)
        if dag is not None:
            ctx.blueprint_plan = dag.to_subgoal_plan(
                getattr(self.config, "max_subgoals", None),
                # 2026-09-15：依赖边开关（默认开＝修复后的正确行为）
                with_deps=getattr(self.config, "blueprint_deps_enabled", True),
                # 2026-09-20：OR 节点展开策略（默认 False = 只取 children[0]）
                or_expand_all=getattr(self.config, "blueprint_or_expand_all", False))
            _anti = len(ctx.blueprint_plan.get("anticipatory_lemmas") or [])
            self.record(ctx, "blueprint",
                        f"Blueprint DAG → {len(ctx.blueprint_plan['subgoals'])} 个子目标"
                        + (f"（含前瞻引理 {_anti} 条，待 SubGoalSolver 注入）"
                           if _anti else ""))
        return ctx

    def generate_blueprint(self, ctx: TaskContext,
                           max_attempts: int = 3,
                           prior_dag_json: str = "") -> Optional[BlueprintDAG]:
        """调用 LLM 生成 AND-OR DAG 并校验；成功返回 DAG，失败返回 None。

        结果同时写入 ctx.blueprint（dict 序列化），供后续阶段消费。

        ★ 2026-10-12（批次 2，用户 Q11）：新增可选参数 `prior_dag_json` ——
          非空时走**「在上一轮蓝图基础上改进」**模板（BLUEPRINT_IMPROVE_USER_TEMPLATE），
          而不是从头构建。**其余逻辑（prefill/解析重试/校验/留痕）一字未改**，
          故本参数不改变原有调用方的任何行为（默认空串）。
        """
        # 预算闸门
        if ctx.is_time_critical():
            self.record(ctx, "blueprint", "预算耗尽，跳过蓝图生成")
            return None

        problem_text = _effective_problem(ctx)
        # v2.9 遗留：原 Lean 前置形式化/leansearch/跨题定理记忆注入
        # （formal_spec/formal_gaps/theorem_memory）已随 2026-09-06 去 Lean 化
        # 全部移除（theorem_memory 为 lean_gate"编译验证通过"定理的复用缓存，
        # 无 Lean 后无写入方，2026-09-06 用户确认一并关闭），此处直接使用题干。
        # 2026-09-13 B0：答案形态要求必须**也在 Blueprint 路径在场** ——
        # 实测 003/074 都走 Blueprint DAG（blueprint_nodes=5/6），而此前只注入
        # LLM 规划路径的 problem_text ⇒ 形态要求完全没生效（两条路径必须同源）。
        try:
            from .question_type import answer_form_requirement as _afr_bp
            problem_text = problem_text + _afr_bp(
                ctx.problem or "", getattr(ctx, "question_type", "") or "")
        except Exception:  # noqa: BLE001
            pass
        # ★ 2026-10-14（题意理解收敛 item ④）：目标锚改由 2.6 的**唯一目标**
        #   `ctx.problem_goal` 注入，且**位置在「题目」占位符内部**（al-075 纪律：
        #   注入块绝不追加到提示词末尾，否则破坏收尾结构、诱使模型进入续写模式）。
        #   原 `_render_understanding_anchor` 把「题意理解链」注入蓝图并要求
        #   "蓝图必须覆盖链上每个中间结论" —— 该函数已删除：链本身随题意理解
        #   收敛撤除（链式分解与蓝图 DAG 同层同义），保留即让蓝图照抄一份错链。
        try:
            from .problem_goal import render_problem_goal as _rpg
            _anchor = _rpg(ctx)
            if _anchor:
                problem_text = problem_text + "\n\n" + _anchor
        except Exception:  # noqa: BLE001  目标是增益，取不到就退回纯题干
            pass

        if prior_dag_json:
            # 迭代式自改进轮（2026-10-12 批次 2）：把上一轮 DAG 原样喂回，要求改进。
            from prompts.blueprint import BLUEPRINT_IMPROVE_USER_TEMPLATE as _BUIT
            user_msg = _BUIT.format(problem=problem_text,
                                    prior_dag=str(prior_dag_json)[:8000])
        else:
            user_msg = BLUEPRINT_DAG_USER_TEMPLATE.format(problem=problem_text)
        # ★ 2026-09-29（截图 #6）：蓝图规划决定"需要推理出哪些中间结论"，
        # 正是 Mathlib 定理线索最该起作用的地方（用户原话："子目标的设立有没有
        # 帮助大模型简化题目"）。注入到 user 侧（不污染 JSON 输出的 system 契约）。
        # 无命中时为空串，零影响。
        if getattr(self.config, 'enable_theorem_hint', True):
            _th = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
            if _th:
                user_msg = user_msg + "\n\n" + _th
        last_resp = None
        # prefill 种子前缀必须**锚定到顶层包装**，不能只用 '{"'。
        # 实测两种失败形态：
        #   1) 无 prefill：Intern 先吐长思维块吃满 token，JSON 被腰斩
        #      → 「Blueprint: JSON 解析失败」（eval_A 0/3 的原始根因）
        #   2) 仅用 '{"'：模型认为已进入对象内部，直接吐 nodes 数组的元素，
        #      丢掉 root_id / nodes 外层包装 → 「Blueprint: DAG 结构非法」
        # 锚定到 `{"root_id": "g", "nodes": [` 后，模型会接着写节点数组，
        # 结构完整（实测 13 节点、validate 零错误）。
        _PREFILL = '{"root_id": "g", "nodes": ['
        for attempt in range(max_attempts):
            # 必须用 prefill：Intern 系列无短种子时会先输出长思维块，
            # 把 token 预算吃满后 JSON 被腰斩（finish_reason=length），
            # 表现为「Blueprint: JSON 解析失败」重试 3 次全败——这是 eval_A 0/3 的根因。
            resp = self.llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": BLUEPRINT_DAG_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    _PREFILL,
                ),
                # ★ 2026-10-02：6144→8192。实测该处曾撞 finish_reason=length（DAG JSON 被腰斩）；
                #   与其余 13 处统一抬到 8192（DeepSeek 适配方案 A）。本文件共 3 处 6144 同改。
                0.2, 8192,
            )
            if resp:
                resp = stitch(_PREFILL, resp)
                last_resp = resp
            if not resp:
                continue
            raw = extract_json(resp)
            if raw is None:
                logger.warning("Blueprint: JSON 解析失败 (attempt %d)", attempt + 1)
                continue
            dag = parse_blueprint(raw)
            if dag is None:
                logger.warning("Blueprint: DAG 结构非法 (attempt %d)", attempt + 1)
                continue
            ok, errors = dag.validate()
            if not ok:
                logger.warning("Blueprint: 校验失败 (attempt %d): %s",
                               attempt + 1, "; ".join(errors[:5]))
                # 记录错误供 trace，然后重试
                self.record(ctx, "blueprint",
                            f"DAG 校验失败: {'; '.join(errors[:5])}")
                continue
            # 成功：写入 ctx.blueprint
            ctx.blueprint = dag.to_dict()
            self.record(ctx, "blueprint",
                        f"Blueprint DAG 生成成功: {len(dag.nodes)} 节点, "
                        f"根={dag.root_id}")
            return dag

        self.record(ctx, "blueprint",
                    f"蓝图生成 {max_attempts} 次尝试均失败; "
                    f"最后响应片段: {(last_resp or '<None>')[:200]}")
        return None

    # ----------------------------------------------------------
    # 整树重生成（#34，老师要求："dag 框架错了要重新生成")
    #
    # 触发场景：
    #   - DagReviewer.review() 报告 should_replan=True
    #   - SubGoalSolver.run 末尾 if replan → 调用本方法
    #
    # 输入：
    #   - prior_dag: 上一轮的 BlueprintDAG（记录到 trace 便于诊断）
    #   - feedback_lines: list[str]，DagReviewer 聚合的 reject hints
    #   - extra_context: 可选，每子目标求解/验证结果（用于让 LLM 看"答错了"信号）
    #
    # 输出：新 BlueprintDAG；写入 ctx.blueprint。
    # ----------------------------------------------------------
    def regenerate_with_feedback(self, ctx: TaskContext,
                                  prior_dag: "BlueprintDAG",
                                  feedback_lines: list,
                                  extra_context: str = "",
                                  max_attempts: int = 3) -> Optional["BlueprintDAG"]:
        """基于失败反馈重建 DAG（重写提示词 = 原题 + 反馈，避免循环/粒度问题）。"""
        if ctx.is_time_critical():
            self.record(ctx, "blueprint_replan", "预算不足，跳过整树重生成")
            return None
        from prompts.blueprint import BLUEPRINT_REPLAN_USER_TEMPLATE
        # 反馈块 = 拒收节点 + hint，逐条列出
        feedback_block_lines = list(feedback_lines or [])
        if extra_context:
            feedback_block_lines.append("【补充上下文（求解/验证反馈）】")
            feedback_block_lines.append(extra_context)
        feedback_block = "\n".join(
            f"  - {line}" for line in feedback_block_lines) or "  （无明确反馈，请自行审视）"

        user_msg = BLUEPRINT_REPLAN_USER_TEMPLATE.format(
            problem=_effective_problem(ctx),
            feedback_block=feedback_block,
        )
        # ★ 2026-09-29：重规划路径同样注入定理线索（独立于首次生成的调用点）
        if getattr(self.config, 'enable_theorem_hint', True):
            _th = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
            if _th:
                user_msg = user_msg + "\n\n" + _th
        # prefill 锚定顶层包装（同首次生成）
        _PREFILL = '{"root_id": "g", "nodes": ['
        for attempt in range(max_attempts):
            if ctx.is_time_critical():
                self.record(ctx, "blueprint_replan", "重生成预算耗尽，终止")
                return None
            resp = self.llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": BLUEPRINT_DAG_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    _PREFILL,
                ),
                # ★ 2026-10-02：6144→8192。实测该处曾撞 finish_reason=length（DAG JSON 被腰斩）；
                #   与其余 13 处统一抬到 8192（DeepSeek 适配方案 A）。本文件共 3 处 6144 同改。
                0.2, 8192,
            )
            if resp:
                resp = stitch(_PREFILL, resp)
            if not resp:
                continue
            raw = extract_json(resp)
            if raw is None:
                logger.warning("BlueprintReplan: JSON 解析失败 (attempt %d)", attempt + 1)
                continue
            dag = parse_blueprint(raw)
            if dag is None:
                logger.warning("BlueprintReplan: DAG 结构非法 (attempt %d)", attempt + 1)
                continue
            ok, errors = dag.validate()
            if not ok:
                logger.warning("BlueprintReplan: 校验失败 (attempt %d): %s",
                               attempt + 1, "; ".join(errors[:5]))
                continue
            ctx.blueprint = dag.to_dict()
            self.record(ctx, "blueprint_replan",
                        f"DAG 重生成成功: {len(dag.nodes)} 节点, "
                        f"根={dag.root_id}, "
                        f"上一轮 {len(prior_dag.nodes)} 节点")
            return dag

        self.record(ctx, "blueprint_replan",
                    f"DAG 重生成 {max_attempts} 次尝试均失败")
        return None

    # ----------------------------------------------------------
    # 子树级局部重写（#34 第二级，LEAP 2.5 核心："子树缺陷 → DFS 回溯"）
    #
    # 与整树重生成的区别：
    #   - 整树：reject 多 / 根级结构错误 → 全推倒重来（改动大，误伤好的部分）
    #   - 子树：只定位"错在哪"（LCA 最低公共祖先），只重写该子树，
    #     其余节点原样保留（改动精准，不误伤）
    #
    # 输入：
    #   - prior_dag: 当前 BlueprintDAG
    #   - rejected_ids: DagReviewer 判 reject 的节点 id 列表（"错误的地方"）
    #   - feedback_lines: 聚合的 reconstruction_hint（"错误的原因"，注入提示词）
    #
    # 策略：
    #   1. 计算 rejected_ids 的 LCA。若 LCA 是根 → 波及全图，退化为整树重生成。
    #   2. 否则 _prune_subtree(LCA) 摘除该子树（保留 LCA 锚点），
    #      让 LLM 只针对"LCA 这一层如何分解"重新生成子树。
    #   3. _graft_subtree 接回，validate 通过才返回。
    # ----------------------------------------------------------
    def regenerate_subtree(self, ctx: TaskContext,
                           prior_dag: "BlueprintDAG",
                           rejected_ids: list,
                           feedback_lines: list,
                           max_attempts: int = 3) -> Optional["BlueprintDAG"]:
        """基于评审反馈**局部重写**被拒节点所在子树（其余节点保留）。"""
        if ctx.is_time_critical():
            self.record(ctx, "blueprint_subtree", "预算不足，跳过子树重写")
            return None
        if not rejected_ids:
            return None
        from prompts.blueprint import BLUEPRINT_SUBTREE_USER_TEMPLATE
        lca = prior_dag._lca(rejected_ids)
        if lca is None:
            return None
        # 波及根 → 直接整树（子树重写无意义）
        if lca == prior_dag.root_id:
            self.record(ctx, "blueprint_subtree",
                        "LCA=根，退化整树重生成")
            return self.regenerate_with_feedback(
                ctx, prior_dag, feedback_lines, max_attempts=max_attempts)
        lca_node = prior_dag.nodes[lca]

        # 反馈块：只保留"位于该子树内"的 reject 节点 hint（减少噪声）
        subtree_ids = prior_dag._subtree_ids(lca)
        feedback_block_lines = []
        for line in (feedback_lines or []):
            for nid in subtree_ids:
                if line.startswith(f"[{nid}]"):
                    feedback_block_lines.append(line)
                    break
        if not feedback_block_lines:
            feedback_block_lines = list(feedback_lines or [])[:6]
        feedback_block = "\n".join(
            f"  - {line}" for line in feedback_block_lines) or "  （无明确反馈，请自行审视）"

        user_msg = BLUEPRINT_SUBTREE_USER_TEMPLATE.format(
            problem=_effective_problem(ctx)[:2000],
            root_id=lca,
            target_statement=lca_node.statement,
            feedback_block=feedback_block,
        )
        # ★ 2026-09-29：子树重写路径同样注入定理线索（第三条独立调用点）
        if getattr(self.config, 'enable_theorem_hint', True):
            _th = (getattr(ctx, 'theorem_hint_block', '') or '').strip()
            if _th:
                user_msg = user_msg + "\n\n" + _th
        # prefill 锚定（子树也是标准 DAG JSON 结构）
        _PREFILL = '{"root_id": "r", "nodes": ['
        for attempt in range(max_attempts):
            if ctx.is_time_critical():
                self.record(ctx, "blueprint_subtree", "重写预算耗尽，终止")
                return None
            resp = self.llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": BLUEPRINT_DAG_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    _PREFILL,
                ),
                # ★ 2026-10-02：6144→8192。实测该处曾撞 finish_reason=length（DAG JSON 被腰斩）；
                #   与其余 13 处统一抬到 8192（DeepSeek 适配方案 A）。本文件共 3 处 6144 同改。
                0.2, 8192,
            )
            if resp:
                resp = stitch(_PREFILL, resp)
            if not resp:
                continue
            raw = extract_json(resp)
            if raw is None:
                logger.warning("BlueprintSubtree: JSON 解析失败 (attempt %d)", attempt + 1)
                continue
            new_dag = parse_blueprint(raw)
            if new_dag is None:
                logger.warning("BlueprintSubtree: DAG 结构非法 (attempt %d)", attempt + 1)
                continue
            # 新子树根 statement 应 ≈ 被重写的目标节点（防止模型跑题）
            new_root_stmt = (new_dag.nodes.get(new_dag.root_id)
                             or new_dag.nodes[next(iter(new_dag.nodes))]).statement.strip()
            tgt_stmt = lca_node.statement.strip()
            if new_root_stmt and tgt_stmt and new_root_stmt[:30] != tgt_stmt[:30]:
                logger.warning(
                    "BlueprintSubtree: 新根 statement 与目标不符 (%s != %s), attempt %d",
                    new_root_stmt[:30], tgt_stmt[:30], attempt + 1)
                continue
            # 接回原图
            merged = prior_dag._graft_subtree(lca, new_dag, id_prefix=f"r{attempt}")
            if merged is None:
                logger.warning("BlueprintSubtree: 拼接校验失败 (attempt %d)", attempt + 1)
                continue
            ctx.blueprint = merged.to_dict()
            self.record(ctx, "blueprint_subtree",
                        f"子树重写成功: LCA={lca}, "
                        f"原 {len(prior_dag.nodes)} 节点 → {len(merged.nodes)} 节点, "
                        f"reject_ids={rejected_ids[:5]}")
            return merged

        self.record(ctx, "blueprint_subtree",
                    f"子树重写 {max_attempts} 次尝试均失败, LCA={lca}")
        return None

# ---------------------------------------------------------------------------
# 批次 2（2026-10-12 用户 Q11–Q18）：蓝图的多次生成 / 自审 / 差异观测
# ---------------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 批次 2（2026-10-12，用户 Q11–Q18）：多次生成 + 自审
    # ------------------------------------------------------------------
    def generate_blueprint_iterative(self, ctx: TaskContext,
                                     max_attempts: int = 3) -> Optional[BlueprintDAG]:
        """**迭代式**多次生成：1 次初始生成 + (N-1) 次在上一轮基础上无条件自改进。

        ★ 用户 Q11 原话：「反复三次就是让大模型生成三次，生成完成后再重新生成，
          就是无条件自改进，让大模型自己再改进两次」⇒ 语义是**迭代精化**，
          **不是**独立重采样（后者会把差异当噪声）。
        ★ 采用策略（Q13）：**以最后一轮为准**（"最后一份应该是最好的一份"）。
        ★ 每轮差异写入 `ctx.blueprint_diffs`（结构层 + 语义层），只记录不裁决（Q15）。

        轮数由 `blueprint_multi_gen_rounds` 控制（默认 3，=1 时等价于单次生成）。
        任一改进轮未产出 ⇒ **保留上一轮**（不因一次失败丢掉已得蓝图）。
        """
        rounds = max(1, int(getattr(self.config, "blueprint_multi_gen_rounds", 3) or 3))
        dag = self.generate_blueprint(ctx, max_attempts=max_attempts)
        if dag is None:
            return None
        diffs = []
        for r in range(1, rounds):
            if ctx.is_time_critical():
                self.record(ctx, "blueprint",
                            f"多次生成在第 {r + 1}/{rounds} 轮前因时间紧迫停止"
                            f"（保留第 {r} 轮蓝图）")
                break
            import json as _json
            new_dag = self.generate_blueprint(
                ctx, max_attempts=max_attempts,
                prior_dag_json=_json.dumps(dag.to_dict(), ensure_ascii=False))
            if new_dag is None:
                self.record(ctx, "blueprint",
                            f"第 {r + 1}/{rounds} 轮自改进未产出 ⇒ 保留第 {r} 轮蓝图")
                break
            diffs.append(_dag_diff(dag, new_dag))
            self.record(
                ctx, "blueprint",
                f"第 {r + 1}/{rounds} 轮自改进：节点 "
                f"{diffs[-1]['struct_prev']['nodes']}→{diffs[-1]['struct_new']['nodes']}、"
                f"陈述变化 {diffs[-1]['stmt_changed']}、新增 {len(diffs[-1]['stmt_added'])}、"
                f"删除 {len(diffs[-1]['stmt_removed'])}")
            dag = new_dag
        ctx.blueprint = dag.to_dict()
        ctx.blueprint_diffs = diffs
        ctx.blueprint_gen_rounds = 1 + len(diffs)
        return dag

    def self_review_blueprint(self, ctx: TaskContext,
                              dag: "BlueprintDAG") -> "BlueprintDAG":
        """第 4 次自审（用户 Q12）：LLM 审一遍蓝图，**按建议重写 1 轮即停**。

        复用既有 `DagReviewerAgent`（判「子目标是否真的简化父问题」—— 正是 LEAP 论文
        「LLM reviewer 语义过滤」的角色，见方案 v4 §3.6.7）。
        失败/无 reject/重写未产出 ⇒ 原样返回（**绝不阻断**）。
        """
        try:
            from .dag_reviewer import DagReviewerAgent
        except ImportError:
            from agent.dag_reviewer import DagReviewerAgent
        try:
            if ctx.is_time_critical():
                return dag
            reviewer = DagReviewerAgent(self.client, self.config)
            report = reviewer.review(ctx, dag, results_map={})
            rejected = [nid for nid in report.rejected_nodes()
                        if nid in dag.nodes] if hasattr(report, "rejected_nodes") else []
            total = max(1, len(getattr(dag, "nodes", {}) or {}))
            ratio = len(rejected) / float(total)
            self.record(ctx, "blueprint_self_review",
                        f"蓝图自审：{len(rejected)}/{total} 节点被拒（{ratio:.0%}）")
            threshold = float(getattr(self.config,
                                      "blueprint_self_review_reject_ratio", 0.4) or 0.4)
            if not rejected or ratio < threshold:
                return dag
            lines = []
            for nid in rejected[:10]:
                node = dag.nodes.get(nid)
                # ★ 2026-10-22：加 `[nid]` 前缀 —— `regenerate_subtree` 靠它筛出
                #   "位于被重写子树内"的反馈（减少无关噪声）；整树路径不解析格式。
                lines.append("[%s] %s" % (
                    nid, (getattr(node, "statement", "") or "")[:120]))
            # ★★ 2026-10-22（接线回补）：`regenerate_subtree` 的实现一直存在但
            #   全仓无调用（历史备份里 `sub_goal_solver.py` 有三处调用，重构时丢了）。
            #   它是**局部重写**：算出被拒节点的 LCA，只重写该子树、其余节点保留；
            #   且内部**已内建**「LCA 是根 ⇒ 退化为整树」的分支 ⇒ 接回是安全的。
            #   开关 `blueprint_subtree_regen`（默认 True）关闭则回到纯整树重写。
            if _subtree_regen_enabled():
                new_dag = self.regenerate_subtree(
                    ctx, dag, rejected, lines, max_attempts=1)
                self.record(ctx, "blueprint_self_review",
                            "自审重写：局部子树（被拒 %d 个，LCA 自动判定）"
                            % len(rejected))
            else:
                new_dag = self.regenerate_with_feedback(
                    ctx, dag, lines, extra_context="【自审（第 4 次）】",
                    max_attempts=1)
                self.record(ctx, "blueprint_self_review",
                            "自审重写：整树（开关关闭局部重写）")
            if new_dag is None:
                self.record(ctx, "blueprint_self_review",
                            "自审建议重写但未产出 ⇒ 采用自审前的蓝图")
                return dag
            ctx.blueprint = new_dag.to_dict()
            self.record(ctx, "blueprint_self_review", "自审后重写 1 轮完成（采用新蓝图）")
            return new_dag
        except Exception as e:  # noqa: BLE001  自审是增益，不是门禁
            self.record(ctx, "blueprint_self_review",
                        f"蓝图自审异常（跳过，不阻断）: {type(e).__name__}: {str(e)[:120]}")
            return dag

def _dag_diff(prev, new) -> dict:
    """结构层 + 语义层的蓝图差异（**只记录、不裁决** —— 用户 Q15）。

    结构层：节点数 / AND 数 / OR 数 / 叶子数；
    语义层：statement 文本集合的增删 + 同 id 下 statement 变化数。
    """
    def _stat(d):
        nodes = getattr(d, "nodes", None) or {}
        return {
            "nodes": len(nodes),
            "and": sum(1 for n in nodes.values()
                       if getattr(n, "node_type", "") == "and"),
            "or": sum(1 for n in nodes.values()
                      if getattr(n, "node_type", "") == "or"),
            "leaves": sum(1 for n in nodes.values()
                          if not getattr(n, "children", None)),
        }
    st1 = {k: (getattr(n, "statement", "") or "").strip()
           for k, n in (getattr(prev, "nodes", None) or {}).items()}
    st2 = {k: (getattr(n, "statement", "") or "").strip()
           for k, n in (getattr(new, "nodes", None) or {}).items()}
    s1, s2 = set(st1.values()), set(st2.values())
    return {
        "struct_prev": _stat(prev), "struct_new": _stat(new),
        "stmt_added": sorted(s2 - s1)[:10],
        "stmt_removed": sorted(s1 - s2)[:10],
        "stmt_changed": sum(1 for k in (set(st1) & set(st2))
                            if st1[k] != st2[k]),
    }
