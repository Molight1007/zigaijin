"""
DagReviewer —— LEAP 5.3 核心：评估 DAG 子目标分解质量（#34）
====================================================================
对应老师要求："dag 板块重点搞好"——让 LLM reviewer 评估每个子目标
是否"使父目标更简单 / 提供可行路径 / 无循环"，拒绝弱分解、
驱动局部/整树重生成。

设计依据：
- LEAP 2.5（Verification-Guided Proof Search）—LLM reviewer 是过滤无意义
  子目标的关键；去 reviewer 后 agent 反复扩展弱蓝图（直到预算耗尽）。
- LEAP 5.3（Toward LLM-Guided Proof Search）—ablation 实验：去掉 LLM
  reviewer 后 Putnam 2025 A5 跑 8 rollout 仍失败（变体 § 5.3）。
- LEAP Figure 2—DAG 节点状态可视化（green=proved, brown=定义，dashed=
  anticipatory lemma）；本模块不画图但产出与之一致的状态判定。

工作流：
1. 启发式预筛（不调 LLM）：
   a. 与祖辈词袋相似 > 阈值 → 候选评审（循环风险）
   b. statement 字数 < 阈值 → 也进候选评审（粒度过粗）
2. LLM 深度评估（对启发式筛出的"可疑节点"调 1 次 LLM）：
   a. 评估"简单化/可行/无循环/依赖正确"四维度
   b. reject 时产出 reconstruction_hint（局部重写建议）
3. 产出 DagReviewReport：每节点 verdict + reject 时回写 hint

安全护栏：
- 单次评审节点数 ≤ MAX_REVIEW_NODES（防深度 DAG 烧光预算）
- 所有评审 LLM 调用都过 budget.can_spend
- 启发式已筛出风险的不跳过 LLM（信任但不放过）

对外接口：
- DagReviewResult：单节点评审结果（含 verdict/issues/hint）
- DagReviewReport：整图报告（聚合 + replan 决策依据）
- DagReviewerAgent.review(ctx, dag, **kwargs)：主入口
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .base import BaseAgent, TaskContext
from .blueprint_planner import BlueprintDAG, extract_json

logger = logging.getLogger("MathPilot")

# ============================================================
# 评审参数（可被 config 覆盖——见 DagReviewerAgent.__init__）
# ============================================================

# 安全上限：单题评审节点数（防深度 DAG 烧光预算——LEAP 实践通常 < 30）
MAX_REVIEW_NODES = 30
# 启发式：词袋相似度 > 该阈值 → 视为循环风险
CIRCULARITY_TOKEN_OVERLAP = 0.70
# 启发式：statement 字数 < 该阈值 → 视为粒度过粗
MIN_STATEMENT_CHARS = 8
# 评审触发比：reject >= 该比例 → 提示整树重生成（Stage 4 调度层使用）
REJECT_REPLAN_THRESHOLD = 0.40
# 评审触发数：reject 节点数 >= 该绝对值 → 提示整树重生成
# 9/1 用户拍板 3→5：006/017/027 冒烟实锤——24-31% 比例的 DAG 被绝对数 3 误卡
# （大图 17 节点 4 个 reject=24% 仍触发重构，浪费预算）。比例阈值 40% 继续兜底。
REJECT_REPLAN_COUNT = 5

# ============================================================
# issues 标签归一化（2026-09-30，用户 #6「蓝图设计有没有问题」）
# ============================================================
# 背景：提示词原先**没有给出 issues 的固定标签表**（只写"category:具体描述"），
# 于是 LLM 自造标签，同一含义出现多种写法。official112_local_0910 实测：
#   circular_risk=190 / circularity=190      ← 同一含义被算成两类
#   invalid_dependencies=38 / invalid_dependency=30
#   feasible_path=25 / missing_decomposition=19 /
#   missing_dependency=17 / missing_dependencies=16
# ⇒ 「蓝图哪一维度最常出问题」的归因统计被系统性打散。
# 两道修：① prompts/dag_review.py 固化合法标签表（治源头）；
#         ② 此处兜底归一化（治存量 + 防模型偶发不听）。
# 只做**同义词映射**，不猜语义：认不出的标签归 `other`，绝不硬塞。
ISSUE_CANON = {
    # 循环
    "circularity": "circular_risk",
    "circular": "circular_risk",
    "circular_risk": "circular_risk",
    # 依赖
    "invalid_dependency": "invalid_dependencies",
    "invalid_dependencies": "invalid_dependencies",
    "missing_dependency": "missing_dependency",
    "missing_dependencies": "missing_dependency",
    "dependency": "invalid_dependencies",
    # 简单化
    "decomposition": "no_simplification",
    "no_simplification": "no_simplification",
    "missing_decomposition": "no_simplification",
    "under_specified": "under_specified",
    "underspecified": "under_specified",
    # ⚠ 以下 7 种 `simplif*` 变体来自**存量数据实测**（official112_local_0910
    #   共 55 条全落 other）：模型把"未真正简化"写成了十几种形式。
    #   语义明确 = 与 no_simplification 同义，故合并（不是猜）。
    "simplification": "no_simplification",
    "simplification_failed": "no_simplification",
    "simplification_failure": "no_simplification",
    "simplification_fail": "no_simplification",
    "simplifies": "no_simplification",
    "simplifies_fail": "no_simplification",
    "simplifies_failure": "no_simplification",
    "simplifies_parent": "no_simplification",
    "simplifies_child": "no_simplification",
    # 可行路径
    "feasibility": "feasible_path",
    "feasible_path": "feasible_path",
    "feasible_route": "feasible_path",
    "no_feasible_path": "feasible_path",
    "no_plausible_route": "feasible_path",
    "no_valid_path": "feasible_path",
    "unfeasible": "feasible_path",
    "infeasible": "feasible_path",
    # 依赖（dependency_error / dependency_issue / dependency_circular 实测各 3-5 条）
    "dependency_error": "invalid_dependencies",
    "dependency_issue": "invalid_dependencies",
    "dependency_circular": "circular_risk",
    # 数学合理性
    "soundness": "math_soundness",
    "math_soundness": "math_soundness",
    "math_soundness_issue": "math_soundness",
    "math_unsoundness": "math_soundness",
    # 分解依据 / 缺引理支撑
    "missing_rationale": "missing_rationale",
    "rationale": "missing_rationale",
    # missing_justification 实测 12 条，描述均为"无前置引理支撑就直接断言"
    # ⇒ 语义 = 缺依据，归 missing_rationale（不是 missing_lemma 那种检索问题）
    "missing_justification": "missing_rationale",
    "missing_lemma": "missing_rationale",
    # 衔接
    "coherence": "coherence",
    "cohesion": "coherence",
    # ★ Lean 逻辑层检测（sub_goal_solver._lean_dag_logic_check 升级时写入，
    #   格式 `lean_logic_error:首条诊断`）。它是**由 Lean 编译失败**产生的硬证据，
    #   与 LLM 主观评审是两类信号 ⇒ 单列一类便于归因（回答用户 #6
    #   「用 Lean 检测有没有 sorry 的地方」到底抓到了什么）。
    "lean_logic_error": "lean_logic_error",
}

# 归因报告用的中文含义（诊断报告「蓝图哪一维度出问题」表直接取用）
ISSUE_CN = {
    "under_specified": "粒度过粗/目标不可验证",
    "no_simplification": "未真正简化父目标",
    "circular_risk": "与祖辈重复（循环分解）",
    "feasible_path": "找不到可信解题路径",
    "invalid_dependencies": "依赖与图结构不匹配",
    "missing_dependency": "缺失关键前置",
    "math_soundness": "断言与题目条件矛盾",
    "missing_rationale": "缺分解依据",
    "coherence": "与父节点衔接不上",
    "lean_logic_error": "Lean 编译不过（命题形式不成立）",
    "other": "其他/未归类",
}


def normalize_issue_tag(issue: str) -> str:
    """把一条 issue 字符串的标签归一化。

    输入形如 ``"circularity:与祖辈 X 相似"`` → 返回 ``"circular_risk"``。
    无冒号 / 空标签 → ``"other"``。**绝不抛异常**（诊断路径不能因脏数据崩）。
    """
    s = str(issue or "")
    tag = s.split(":", 1)[0].strip().lower() if ":" in s else ""
    if not tag:
        return "other"
    return ISSUE_CANON.get(tag, "other")

# ============================================================
# 数据结构
# ============================================================

@dataclass
class DagReviewResult:
    """单个节点的评审结果。"""
    node_id: str
    verdict: str                         # "accept" | "reject"
    quality_score: float = 1.0           # 0.0-1.0
    issues: list = field(default_factory=list)   # reject 时填充：["circular_risk:...", ...]
    reconstruction_hint: str = ""        # reject 时填充：给 LLM 重写时一句话指引
    heuristic_only: bool = False         # 仅启发式、未走 LLM
    # 2026-10-04 增（fix-a1）：LLM 调用异常 / 空响应 / JSON 非 dict ⇒ 默认 accept(0.5)。
    # 这是**解析失败导致的降级**，与「LLM 真的判 accept」不可区分，单列此信号，
    # 供 review() 汇总后把整份报告标 degraded（宁可重规划，不要假放行）。
    parse_failed: bool = False

    @property
    def is_reject(self) -> bool:
        return self.verdict == "reject"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "verdict": self.verdict,
            "quality_score": self.quality_score,
            "issues": list(self.issues),
            "reconstruction_hint": self.reconstruction_hint,
            "heuristic_only": self.heuristic_only,
            "parse_failed": self.parse_failed,
        }


@dataclass
class DagReviewReport:
    """整张 DAG 的评审报告（聚合 + 触发重生成决策依据）。"""
    results: dict = field(default_factory=dict)   # node_id -> DagReviewResult
    # 9/1 增：评审是否降级（预算耗尽跳过部分/全部 LLM 节点）。
    # 冒烟 007 实锤：超时风暴下 14 个节点 LLM 全跳过 → reject=0 → 假 pass=True。
    # degraded=True 时 should_replan 强制返回 True（宁重构不假放行）。
    degraded: bool = False

    @property
    def reject_count(self) -> int:
        return sum(1 for r in self.results.values() if r.is_reject)

    @property
    def accept_count(self) -> int:
        return sum(1 for r in self.results.values() if not r.is_reject)

    @property
    def reject_ratio(self) -> float:
        return self.reject_count / len(self.results) if self.results else 0.0

    def get(self, node_id: str) -> Optional[DagReviewResult]:
        return self.results.get(node_id)

    def rejected_nodes(self) -> list:
        return [r.node_id for r in self.results.values() if r.is_reject]

    def accepted_nodes(self) -> list:
        return [r.node_id for r in self.results.values() if not r.is_reject]

    def should_replan(self,
                      reject_ratio_thr: float = REJECT_REPLAN_THRESHOLD,
                      reject_count_thr: int = REJECT_REPLAN_COUNT) -> bool:
        """是否建议整树重生成？规则：
        - reject 数 >= 绝对阈值（9/1 3→5） → 强烈建议（局部修不动）
        - reject 比例 >= 40%（相对）→ 也建议（系统性失败）
        - 评审降级（预算耗尽跳过 LLM）→ 强制 True，宁重构不假放行
        """
        if self.degraded:
            return True
        return (self.reject_count >= reject_count_thr
                or self.reject_ratio >= reject_ratio_thr)

    def merge_from_hints(self) -> str:
        """聚合所有 reject 的 reconstruction_hint，供 LLM 整树重生成时参考。"""
        hints = []
        for r in self.results.values():
            if r.is_reject and r.reconstruction_hint:
                hints.append(f"[{r.node_id}] {r.reconstruction_hint}")
        return "\n".join(hints)

    def issue_histogram(self) -> dict:
        """归一化后的 issues 标签直方图 → ``{规范标签: 条数}``。

        回答用户 #6「蓝图的设计有没有问题」——**问题出在哪一维度**。
        标签经 `normalize_issue_tag` 归一，故 `circularity` 与 `circular_risk`
        会被合并计数（历史实测同义异写导致统计被打散，详见模块顶部注释）。
        """
        c: dict = {}
        for r in self.results.values():
            if not r.is_reject:
                continue
            for it in (r.issues or []):
                k = normalize_issue_tag(it)
                c[k] = c.get(k, 0) + 1
        return c

    def issue_fix_rate(self, prev: "Optional[DagReviewReport]" = None) -> dict:
        """**每条 issue 在重写后有没有被修掉**（对应"设计有没有问题"的下半问）。

        做法：拿本轮 report.results 与 `prev`（重写前那轮的 report.results）
        比对 —— 若节点 nid 上一轮 reject 且本轮 accept ⇒ 该 issue 已修复。
        返回 ``{规范标签: {"seen": n, "fixed": m, "fixed_ratio": p}}``。

        prev 为空 / None ⇒ 返回空 dict（首轮无对照，不应伪造修复率）。
        ⚠ 只统计**两轮都出现过的节点 id**：重写后 id 变了（整树重生成很常见）
           则本轮 accept 无法归因给旧 issue，**不计入分母也不计入分子**，
           避免"节点没了就当修好了"这种自欺。
        """
        if not prev or not getattr(prev, "results", None):
            return {}
        out: dict = {}
        for nid, old in prev.results.items():
            if not old.is_reject:
                continue
            new = self.results.get(nid)
            if new is None:
                continue          # 节点已消失/换 id：不作为修复证据
            tags = {normalize_issue_tag(it) for it in (old.issues or [])} or {"other"}
            for t in tags:
                rec = out.setdefault(t, {"seen": 0, "fixed": 0, "fixed_ratio": 0.0})
                rec["seen"] += 1
                if not new.is_reject:
                    rec["fixed"] += 1
        for rec in out.values():
            rec["fixed_ratio"] = round(rec["fixed"] / rec["seen"], 3) if rec["seen"] else 0.0
        return out

    def to_dict(self) -> dict:
        return {
            "results": {nid: r.to_dict() for nid, r in self.results.items()},
            "reject_count": self.reject_count,
            "reject_ratio": self.reject_ratio,
            "degraded": self.degraded,
            "should_replan": self.should_replan(),
            "issue_histogram": self.issue_histogram(),
        }


# ============================================================
# 启发式工具
# ============================================================

# 用 Unicode 词集 + 拉丁/数字词组，过滤 LaTeX 花括号碎片
_TOKEN_RE = re.compile(r"[一-鿿\w]+", re.UNICODE)


def _token_overlap(a: str, b: str) -> float:
    """词袋 Jaccard 相似度（启发式循环检测）。

    中文按单字即可（数学题中连续词已是术语），英文按 \\w+。
    """
    ta = set(_TOKEN_RE.findall((a or "").lower()))
    tb = set(_TOKEN_RE.findall((b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ============================================================
# Agent
# ============================================================

class DagReviewerAgent(BaseAgent):
    """AND-OR DAG 子目标分解质量评审（LEAP 5.3 复现）。"""

    name = "DagReviewer"

    def __init__(self, client, config):
        super().__init__(client, config)
        # 复用蓝图的 JSON 解析器（已通过平衡括号算法优化）
        self._max_review = int(getattr(config, "dag_review_max_nodes",
                                        MAX_REVIEW_NODES) or MAX_REVIEW_NODES)
        self._reject_thr = float(getattr(config, "dag_review_reject_thr",
                                          REJECT_REPLAN_THRESHOLD)
                                  or REJECT_REPLAN_THRESHOLD)
        # 绝对 reject 阈值也可配置（冒烟 006：3/13=23% 仍触发重生成 → 大图偏严，
        # 默认保持 3 不动，等真实 A/B 数据再决定是否收紧）
        self._reject_count_thr = int(getattr(config, "dag_review_reject_count",
                                              REJECT_REPLAN_COUNT)
                                      or REJECT_REPLAN_COUNT)

    # ----------------------------------------------------------
    # BaseAgent 抽象方法实现：包装 review() 入口（与 orchestrator 集成时用）
    # ----------------------------------------------------------
    def run(self, ctx: TaskContext) -> TaskContext:
        """标准 Agent run 入口：从 ctx.blueprint 读取 DAG → 评审 → 报告回写 ctx。"""
        if not ctx.blueprint:
            self.record(ctx, "dag_review", "无 ctx.blueprint，跳过")
            return ctx
        try:
            dag = BlueprintDAG.from_dict(ctx.blueprint)
        except Exception as exc:  # noqa: BLE001
            self.record(ctx, "dag_review", f"DAG 解析失败：{exc}")
            return ctx
        self.review(ctx, dag, results_map={})
        return ctx

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def review(self, ctx: TaskContext, dag: BlueprintDAG,
               results_map: Optional[dict] = None) -> DagReviewReport:
        """评审整张 DAG，返回 DagReviewReport。

        results_map: 子目标 id → 求解结果，用于二次确认"可行路径"。
                     为 None 时仅按 DAG 结构本身评审。
        """
        # 预算闸门
        if ctx.is_time_critical():
            # 2026-10-04 修（fix-a1）：原实现返回 **degraded 缺省 False** 的空报告，
            # 而 should_replan() 的声明是「降级（预算耗尽跳过 LLM）→ 强制 True，
            # 宁重构不假放行」（见 :227-239）。实际 degraded=False ⇒ 落到
            # 「空报告 reject_count=0」⇒ should_replan 返回 False = **假放行**，
            # 声明与实现相反。此处与下方 `report.degraded = bool(llm_skipped)`
            # 同一写法置位，让早退分支也强制重规划。
            self.record(ctx, "dag_review", "预算不足，跳过 DAG 评审（降级）")
            return DagReviewReport(degraded=True)
        if not dag or not dag.nodes:
            return DagReviewReport()

        # 1) 启发式预筛（不调 LLM，立刻产出可用的 reject 候选）
        heuristic_results = self._heuristic_screen(dag)
        # 2) LLM 深度评估（启发式筛出 + 所有叶子），产出更精细的 verdict
        results: dict = {}
        llm_skipped = []
        # 启发式已 reject 的先入报告（避免重复 LLM）
        for nid, hres in heuristic_results.items():
            if hres.heuristic_only:
                results[nid] = hres
        # 找出还需要 LLM 的节点（启发式放过 + 所有叶子）
        candidates = self._llm_candidates(dag, heuristic_results)
        candidates = candidates[: self._max_review]
        for node in candidates:
            if ctx.is_time_critical():
                self.record(ctx, "dag_review", "LLM 预算耗尽，评审提前终止")
                llm_skipped.append(node.id)
                break
            res = self._llm_review_node(ctx, dag, node, results_map or {})
            results[node.id] = res

        report = DagReviewReport(results=results)
        # 9/1 增：预算耗尽导致 LLM 节点被跳过 → 标记评审降级，
        # should_replan 强制 True（宁重构不假放行，冒烟 007 教训）
        # 2026-10-04 修（fix-a1）：同一次评审里若有节点走「解析失败默认 accept」
        # 路径（LLM 异常/空响应/JSON 非 dict），其 verdict 不可信 ⇒ 同样标降级。
        _n_parse_failed = sum(1 for r in results.values()
                              if getattr(r, "parse_failed", False))
        report.degraded = bool(llm_skipped) or _n_parse_failed > 0
        ctx.dag_review_report = report.to_dict()
        replan_flag = report.should_replan(self._reject_thr,
                                           self._reject_count_thr)
        # 2026-09-15：Purpose 覆盖率（LEAP §2.1）。
        # 用途：核实"rationale 必填"这条提示词约束是否**真的生效**——
        # 否则无法区分"模型没写"与"评审器没看"（后者已由第 6 维修复）。
        # 只做观测计数，不改变 verdict（避免与既有启发式语义冲突）。
        _missing_rationale = sum(
            1 for n in dag.nodes.values()
            if len((getattr(n, "rationale", "") or "").strip()) < 4)

        self.record(
            ctx, "dag_review",
            f"DAG 评审: total={len(results)} "
            f"reject={report.reject_count}/{len(results)} "
            f"({report.reject_ratio:.0%}), "
            f"llm_skipped={len(llm_skipped)}, "
            f"parse_failed={_n_parse_failed}, "
            f"degraded={report.degraded}, "
            f"missing_rationale={_missing_rationale}/{len(dag.nodes)}, "
            f"should_replan={replan_flag}",
            dag_review_replan=replan_flag,
            dag_review_reject_ratio=round(report.reject_ratio, 3),
            dag_review_reject_count=report.reject_count,
            dag_review_degraded=report.degraded,
            dag_review_parse_failed=_n_parse_failed,
            dag_review_missing_rationale=_missing_rationale,
        )
        return report

    # ----------------------------------------------------------
    # 启发式
    # ----------------------------------------------------------
    def _heuristic_screen(self, dag: BlueprintDAG) -> dict:
        """启发式筛出循环风险 / 粒度过粗的节点，直接产出 reject。

        与 LLM 评审互补：
        - 启发式抓的（高相似 + 字数过短）是高置信度 reject
        - LLM 评审抓的是更细的语义问题（粒度不当、依赖错）
        """
        out: dict = {}
        for node in dag.nodes.values():
            issues: list = []

            # 启发式 1：statement 字数过短 → 粒度过粗
            if len((node.statement or "").strip()) < MIN_STATEMENT_CHARS:
                issues.append(f"under_specified:statement 不足 {MIN_STATEMENT_CHARS} 字")

            # 启发式 2：与祖辈词袋高度相似 → 循环风险（LEAP 5.3 主案例）
            anc_statements = []
            for anc_id in dag._ancestors(node.id):
                anc = dag.nodes.get(anc_id)
                if anc:
                    anc_statements.append((anc_id, anc.statement))
            for anc_id, anc_stmt in anc_statements:
                sim = _token_overlap(node.statement, anc_stmt)
                if sim > CIRCULARITY_TOKEN_OVERLAP:
                    issues.append(
                        f"circular_risk:与祖辈 {anc_id} 词袋相似 {sim:.2f}")
                    break  # 只记最近的一个祖辈

            if issues:
                out[node.id] = DagReviewResult(
                    node_id=node.id,
                    verdict="reject",
                    quality_score=0.2,
                    issues=issues,
                    reconstruction_hint=(
                        "改用更细的子目标；与祖辈陈述不要重复；"
                        "明确指出本子目标与父目标的可验证差异"),
                    heuristic_only=True,   # 启发式直出，未走 LLM
                )
        return out

    def _llm_candidates(self, dag: BlueprintDAG,
                         heuristic_results: dict) -> list:
        """LLM 评审候选 = 所有叶子 + 启发式放过的非叶。

        启发式已 reject 的不再 LLM（避免烧预算 + 启发式判断已够用）。

        ★★ 2026-10-04 修（关键设计错误）：**排除根节点**。
        根节点的 `statement` 就是**原题本身**，而本评审第 1 维判据是
        「是否比**父目标**更简单 / 粒度是否细化 / 有无'证明原题'当子目标」
        —— 根节点按定义**永远无法满足**（它没有父目标，且它就是题目）。
        叠加「6 维任一命中即 reject」+ `should_replan` 的
        `reject_count>=5 或 reject_ratio>=40%`，一个小 DAG 只要 5 个 reject
        就触发；**根节点必然贡献至少 1 个**，其余节点随手就能凑够
        ⇒ 门**结构性不收敛**（实测：112 题中 **80 题（71%）走满 2 轮仍判 replan**）。
        这就是它"拦得住、修不好"的真实原因 —— **不是重写救不了，是判据不可达**。
        根节点需要的只是"等于原题陈述"，不需要"比原题更简单"。
        """
        already_rejected = set(heuristic_results.keys())
        _root = str(getattr(dag, "root_id", "") or "")
        # 叶子：最关键的评审目标（决定求解能否展开）
        leaves = [n for nid, n in dag.nodes.items()
                  if not n.children and nid not in already_rejected
                  and nid != _root]
        # 启发式放过但非叶节点（仍可能有"分解不当"问题）
        non_leaves = [n for nid, n in dag.nodes.items()
                      if n.children and nid not in already_rejected
                      and nid != _root]
        return list(leaves) + non_leaves

    # ----------------------------------------------------------
    # LLM 评估单节点
    # ----------------------------------------------------------
    def _llm_review_node(self, ctx: TaskContext, dag: BlueprintDAG,
                         node, results_map: dict) -> DagReviewResult:
        """LLM 深度评估单个节点的分解质量。"""
        from prompts.dag_review import (
            DAG_REVIEW_SYSTEM, DAG_REVIEW_USER_TEMPLATE)
        from utils.prefill import prefill_messages, stitch
        # 上下文：父节点、兄弟节点、依赖节点 statement
        parent_stmts, child_stmts, dep_stmts = [], [], []
        for pid, pnode in dag.nodes.items():
            if node.id in pnode.children:
                parent_stmts.append(f"  - {pnode.statement}")
            if pid in node.children:
                child_stmts.append(f"  - {dag.nodes[pid].statement}")
        dep_stmts_set = set()
        for anc_id in (dag._ancestors(node.id) | {node.id}):
            anc = dag.nodes.get(anc_id)
            if anc:
                dep_stmts_set.add(anc.statement)
        dep_stmts = [f"  - {s}" for s in list(dep_stmts_set)][:6]

        # 已建立的引理（来自当前 DAG 求解过程）
        lemma_lines = []
        for l in (getattr(ctx, "lemma_repo", []) or [])[-8:]:
            lemma_lines.append(f"  - {str(l)[:140]}")
        lemma_block = "\n".join(lemma_lines) or "  （尚无）"

        # 求解结果（若有）
        result_block = ""
        if results_map.get(node.id):
            r = str(results_map[node.id])[:300]
            result_block = f"\n（该子目标求解结果：{r}）"

        user_msg = DAG_REVIEW_USER_TEMPLATE.format(
            problem=(ctx.problem or "")[:1200],
            node_id=node.id,
            node_type=node.node_type,
            statement=node.statement[:400],
            # LEAP §2.1：Purpose 是评审器判断"子目标是否真的简化了问题"的唯一依据。
            # 2026-09-15 前该字段**根本没有传给评审器** ⇒ 第 6 维"分解依据"无从判断。
            rationale=(getattr(node, "rationale", "") or "（未给出）")[:400],
            parent_statement="\n".join(parent_stmts) or "  （根节点 / 无父）",
            children_statements="\n".join(child_stmts) or "  （叶子节点 / 无子）",
            deps_statements="\n".join(dep_stmts) or "  （无前置依赖）",
            lemma_block=lemma_block,
        ) + result_block

        try:
            # prefill 锚定 JSON 顶层：Intern 系列无短种子时会先输出长思维块，
            # 把 token 预算吃满后 JSON 被腰斩 → extract_json 返回 None → 全部
            # 默认 accept（score=0.50 恒等，评审失去意义）。与 BlueprintPlanner
            # 生成 DAG 时的 prefill 修复同根因（eval_A 0/3）。
            _PREFILL = '{"verdict": '
            resp = self.llm(
                ctx,
                prefill_messages(
                    [
                        {"role": "system", "content": DAG_REVIEW_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    _PREFILL,
                ),
                0.2, 32768,
            )
            if resp:
                resp = stitch(_PREFILL, resp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DagReviewer LLM 调用异常 %s: %s", node.id, exc)
            # 2026-10-04 修（fix-a1）：解析失败 ⇒ 标 parse_failed，由 review() 汇总降级
            return DagReviewResult(node.id, "accept", 0.5, [], "", False,
                                   parse_failed=True)

        if not resp:
            return DagReviewResult(node.id, "accept", 0.5, [], "", False,
                                   parse_failed=True)

        # 复用 blueprint_planner.extract_json（容错）
        raw = extract_json(resp)
        if not isinstance(raw, dict):
            return DagReviewResult(node.id, "accept", 0.5, [], "", False,
                                   parse_failed=True)

        verdict = str(raw.get("verdict", "accept")).lower().strip()
        if verdict not in ("accept", "reject"):
            verdict = "accept"
        try:
            score = float(raw.get("quality_score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        score = max(0.0, min(1.0, score))

        issues_field = raw.get("issues", [])
        if not isinstance(issues_field, list):
            issues_field = []
        issues = [str(x)[:120] for x in issues_field]

        hint = str(raw.get("reconstruction_hint", "") or "")[:400]

        return DagReviewResult(
            node_id=node.id,
            verdict=verdict,
            quality_score=score,
            issues=issues,
            reconstruction_hint=hint,
            heuristic_only=False,
        )
