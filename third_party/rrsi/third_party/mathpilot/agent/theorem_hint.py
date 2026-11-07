# -*- coding: utf-8 -*-
"""领域 → 定理检索（2026-09-29 新增，截图 #3+#4「老师重点关注」）。

设计目标（用户原话）
--------------------
> 我们要判断它是哪个领域的题目，会用到什么定理（这里面就要使用 leansearch
> 去 mathlib 搜索对应的定理了，这也是老师重点关注的地方，就是判断定理对
> 大模型的帮助效果。我的思路是我们先把题库每一题对应的会用到的定理总结下来，
> 然后判断大模型或 leansearch 最后有没有找到正确的定理。以及定理对大模型的
> 推理效果如何？

拆成三件事，**本模块只负责第 ① 件**：
  ① 【检索】给定题目 + 领域 → 用 leansearch 去 Mathlib 找候选定理；
  ② 【判定】命中的定理是否 == 题库预标注的「正确定理」→ ``cli/theorem_probe.py``；
  ③ 【效用】把定理注入解题提示词后，正确率是否提升 → ``theorem_hint_*`` 埋点 + A/B。

★ 关键的架构约束（用户要求「修改 lean 相关调用不会波及大模型部分」）
-------------------------------------------------------------------
本模块是 **agent/ 侧唯一的定理检索入口**。它对外只暴露
``retrieve_theorems_for_question()`` 一个函数，返回**纯数据结构**，
不抛异常、不返回 Lean 对象、不依赖 tools.lean_local 的内部类型。
因此：
  · 换检索后端（官方 API / 离线语料 / 源码扫描）→ 只改本文件；
  · 换解题提示词 / 大模型 → 完全不碰本文件；
  · tools.lean_local 不可用 → 本模块降级返回空结果，主流程照常。
"""

from __future__ import annotations

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
from agent.switch_registry import (
    get_bool as _sw_bool)
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("MathPilot")


# ---------------------------------------------------------------------------
# 数据结构（纯数据，无外部依赖）
# ---------------------------------------------------------------------------

@dataclass
class TheoremHit:
    """一条检索命中的定理。"""
    name: str = ""            # Mathlib 全名，如 Mathlib.Data.Nat.Choose.Basic.Nat.choose
    short: str = ""           # 末段短名，如 Nat.choose
    source: str = ""          # 命中的 query（便于归因「哪个 query 找到的」）
    rank: int = 0             # 在该次检索中的位次（1-based）
    # ★ 2026-10-04 新增：定理**陈述片段**。检索层 `lean_search` 早就返回了它
    #   （`{"name","kind","file","line","snippet"}`），但这里构造时**丢掉了** ⇒
    #   注入给模型的只有"短名"，模型无从判断该定理讲什么 ⇒ 历史实测**采用率仅 1%**。
    snippet: str = ""         # 定理陈述片段（来自检索层，截断后入提示词）

    def to_dict(self) -> dict:
        return {"name": self.name, "short": self.short,
                "source": self.source, "rank": self.rank,
                "snippet": self.snippet}


@dataclass
class TheoremRetrieval:
    """一次「题目 → 定理」检索的完整结果（可直接进 diag / JSON）。"""
    ok: bool = False                     # 检索本身是否跑通（False=后端不可用/异常）
    domain: str = ""
    queries: list = field(default_factory=list)      # 实际发出的 query 列表
    hits: list = field(default_factory=list)         # list[TheoremHit]
    backend: str = ""                    # 实际生效的后端（供归因）
    elapsed: float = 0.0
    reason: str = ""                     # ok=False 时的原因

    @property
    def names(self) -> list:
        return [h.name for h in self.hits]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "domain": self.domain,
            "queries": list(self.queries),
            "hits": [h.to_dict() for h in self.hits],
            "n_hits": len(self.hits),
            "distinct_short": sorted({h.short for h in self.hits}),
            "backend": self.backend,
            "elapsed": round(self.elapsed, 2),
            "reason": self.reason,
        }

    # ------------------------------------------------------------------
    # 提示词渲染（★ 2026-09-29 实测修正）
    # ------------------------------------------------------------------
    # 问题：最初的 block 用「短名 + 全名」并列，实测 5 条命中即 **798 字符**，
    #   而其中 `Mathlib.Geometry.Euclidean.Angle.Oriented.Affine.EuclideanGeometry.`
    #   这类命名空间前缀占了 60%+。对以中文推理的大模型，这串路径**信息量近似为零**，
    #   纯属 token 噪音，还挤占推理预算。注入的本意是"帮模型"，不是"堆 token"。
    # 做法：提示词只给**短名**（模型靠名字+领域语境已能理解，如
    #   `angle_add_angle_add_angle_eq_pi` 一眼就是三角形内角和），
    #   全名**完整保留在 `ctx.theorem_hint_trace` 埋点里**供 Lean 侧/A-B 归因使用。
    #   实测压缩后 5 条从 798 → 约 250 字符（−69%），语义不减。
    def render_hint_block(self, max_items: int = 5,
                          max_chars: int = 0,
                          include_premises: bool = True) -> str:
        """渲染成注入提示词的文本块（无命中返回空串 ⇒ 下游零成本）。

        ★ 2026-10-04 修（**采用率 1% 的直接病因**）：
        原实现「只输出短名 + 一句『不必强行使用』免责句」，模型**无从判断该定理
        讲的是什么、适不适用** ⇒ 实测 460 条检索结果里只有 **1%** 进入实际推理。
        现改为输出「**短名 + 一行定理陈述**」（陈述来自检索层一直就有、却被丢弃的
        `snippet`），并把免责句改为**正向引导**（相关就优先采用并标注）。
        ⚠ 仍保留"以你的判断为准"这一层 —— 检索是**线索**不是裁决，不得覆盖题目事实。
        """
        hits = [h for h in self.hits if (h.short or h.name)][:max(0, max_items)]
        if not hits:
            return ""
        # ★ 2026-10-12（批次 5，用户 §5/§6）两处升级：
        #   ① **前提行**：定理的「前提」正是用户要的**警戒线** ——
        #      引用前必须逐条核对是否被本题满足（此前只给名字+陈述，模型无从核对）；
        #   ② 尾注由单向「优先采用」改为**双作用说明**（警戒线 + 路标）。
        #   ⚠ 仍保留「以你的判断为准」——检索是**线索不是裁决**（有测试锚定，不得删）。
        lines = ["【本题可能在 Mathlib 中用到的定理（leansearch 自动检索，供参考）】"]
        for i, h in enumerate(hits, 1):
            _nm = h.short or h.name
            _sn = " ".join(str(getattr(h, "snippet", "") or "").split())
            lines.append(f"{i}. {_nm} —— {_sn[:200]}" if _sn else f"{i}. {_nm}")
            if include_premises and _sn and "→" in _sn:
                _pre = _sn.split("→", 1)[0].strip()
                if _pre and len(_pre) > 3:
                    lines.append(f"     前提：{_pre[:160]}")
        # ⚠ 尾注**必须压成单行**：`tests/test_theorem_hint_injection.py` 的守卫测试
        #   锁定「块形状 = 标题 + N 项 + 尾注」且「块要显著短于全名版」——
        #   多行尾注会同时破坏这两条（= 真实的 token 噪音回归，不是测试过时）。
        lines.append(
            # ⚠ 措辞受两份守卫共同约束（不得删任一侧）：
            #   ① `请优先采用` —— 2026-10-04 的正向引导修复（提高采用率，历史仅 1%）；
            #   ② 双作用说明 —— 2026-10-12 批次 5（警戒线 + 路标）。
            "（以上为检索到的候选定理及陈述：**请优先采用**与本题相关者、并标注它用在哪一步。"
            "两种作用：【警戒线】引用前先核对前提是否被本题满足，结论若与之矛盾须先排查"
            "自己哪一步错了、禁止无视；【路标】可由其前提→结论反推证明框架或中间量。"
            "若确实不适用，以题意与你自己的判断为准。）")
        _txt = "\n".join(lines)
        if max_chars and len(_txt) > max_chars:
            _keep, _n = [], 0
            for _ln in lines:
                if _n + len(_ln) > max_chars - 60 and _keep and not _ln.startswith("（"):
                    break
                _keep.append(_ln)
                _n += len(_ln) + 1
            _omit = len(hits) - sum(1 for x in _keep if x[:2].rstrip(".").isdigit())
            _keep.append(f"（另有 {max(0, _omit)} 条候选，因篇幅未展开）")
            _txt = "\n".join(_keep)
        return _txt


# ---------------------------------------------------------------------------
# 领域 → 检索关键词（零 LLM，纯映射；避免为检索再花一次大模型调用）
# ---------------------------------------------------------------------------
# 设计说明：leansearch 的源码扫描后端是**英文标识符匹配**（实测见 CHANGES
# 文档），中文领域名直接传进去命中率极差。故这里把题库实际出现的领域名
# 映射成英文数学关键词组。未收录的领域走「题干英文关键词」兜底。
_DOMAIN_KEYWORDS: dict = {
    "离散数学": ["Finset.card", "permutations", "combinatorics", "choose"],
    "组合数学": ["Finset.card", "permutations", "choose", "combinatorial"],
    "数论": ["Nat.Prime", "Nat.ModEq", "Int.gcd", "number theory"],
    "代数": ["Polynomial", "RingHom", "Group", "Algebra"],
    "线性代数": ["Matrix", "LinearMap", "VectorSpace", "determinant"],
    "几何": ["EuclideanGeometry", "angle", "distance", "triangle"],
    "概率统计": ["Probability", "MeasureTheory", "variance", "expectation"],
    "微积分": ["deriv", "integral", "Continuous", "Differentiable"],
    "分析": ["Continuous", "Converges", "limit", "series"],
    "数学分析": ["Continuous", "Deriv", "Integral", "Sequence"],
    "不等式": ["inequality", "le_trans", "pow_le_pow", "Real.sqrt"],
    "数列": ["Nat.rec", "Fibonacci", "geom_sum", "sum_range"],
    "计数": ["Finset.card", "Fintype.card", "card_eq", "count"],
    "图论": ["Graph", "SimpleGraph", "Adj", "Walk"],
    "集合论": ["Set.", "Finset.", "Subset", "card"],
}

# 题干英文停用词（生成兜底 query 时剔除）
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "be", "for",
    "that", "with", "as", "by", "on", "at", "it", "its", "this", "these",
    "let", "show", "prove", "find", "compute", "determine", "calculate",
    "all", "any", "each", "such", "then", "where", "which", "what", "how",
    "there", "we", "you", "if", "no", "not", "every", "some", "given",
}

# 噪声词：在数学题里高频但无区分度（抽出来只会污染 query）
_NOISE_TOKENS = {
    "said", "satisfies", "satisfying", "matter", "five", "three", "four",
    "two", "one", "following", "condition", "conditions", "number",
    "numbers", "value", "values", "positive", "negative", "nonnegative",
    "integer", "integers", "real", "reals", "distinct", "different",
    "respectively", "denote", "denotes", "defined", "definition",
    "suppose", "consider", "example", "problem", "solution", "answer",
    "minimum", "maximum", "greatest", "least", "larger", "smaller",
    # ★ 2026-09-29 实测修正（关键）：**答案格式样板词**必须当噪声剔除。
    #   实测 official112 的 096/101/103~111 共 12 道中文题，题干里唯一的英文
    #   就是样板句「Remember to put your final answer within \boxed{}」——
    #   不剔除时 12 题抽出**完全相同**的关键词，检索结果雷同、标注表报废
    #   （实测 12x 重复簇 ('upper_mem','TheoremForm','pi_le_four','form')）。
    "remember", "put", "your", "final", "within", "boxed", "write",
    "give", "provide", "express", "note", "must", "should", "please",
    "sure", "carefully", "read", "question", "statement", "true", "false",
}

# LaTeX / MathJax 残渣（正则抽出的字母序列里混进来的）
_TEX_NOISE = {
    "ldots", "cdots", "dots", "leq", "geq", "neq", "frac", "sqrt", "cdot",
    "times", "quad", "qquad", "left", "right", "text", "math", "mathrm",
    "begin", "end", "array", "matrix", "cases", "pmod", "bmod", "equiv",
    "sum", "prod", "int", "lim", "infty", "alpha", "beta", "gamma", "delta",
    "epsilon", "theta", "lambda", "sigma", "omega", "phi", "psi", "prime",
    "circ", "star", "ast", "oplus", "otimes", "subset", "subseteq", "in",
    "notin", "forall", "exists", "implies", "iff", "mapsto", "to", "gets",
}

# 数学术语词表（题干里出现即视为高区分度信号）
# 选取原则：这些词在 Mathlib 声明名里**有直接对应**（如 permutation→Perm、
# polynomial→Polynomial、prime→Nat.Prime），因此作为 query 能精准命中。
_MATH_TERMS = [
    "permutation", "combination", "factorial", "binomial",
    "polynomial", "root", "coefficient", "degree", "factor",
    "prime", "divisible", "divisor", "modulo", "congruent", "remainder",
    "gcd", "lcm", "coprime",
    "matrix", "determinant", "eigenvalue", "eigenvector", "linear",
    "vector", "span", "basis", "dimension", "rank",
    "derivative", "differentiable", "integral", "continuous", "limit",
    "convergent", "converges", "diverges", "series", "sequence", "summable",
    "monotone", "bounded", "supremum", "infimum",
    "probability", "expectation", "variance", "random", "distribution",
    "triangle", "circle", "angle", "perpendicular", "parallel", "midpoint",
    "area", "perimeter", "radius", "diameter",
    "graph", "edge", "vertex", "tree", "path", "cycle", "colorable",
    "set", "subset", "finset", "cardinality", "bijection", "injection",
    "surjection", "function", "inverse", "composition",
    "inequality", "equality", "absolute", "floor", "ceiling",
    "group", "ring", "field", "ideal", "homomorphism", "isomorphism",
    "recurrence", "induction", "summation",
]


def domain_keywords(domain: str, question_type: str = "") -> list:
    """领域名 → 英文检索关键词组（未收录则回退到通用词）。"""
    d = (domain or "").strip()
    for key, kws in _DOMAIN_KEYWORDS.items():
        if key and key in d:
            return list(kws)
    return []


# 中文数学术语 → 英文检索词（2026-09-29 新增）。
# ★ 必要性：official112 含大量**中文题**（统计推断 / 线性回归 / 线性规划 等），
#   而 leansearch 的源码扫描后端只认英文标识符 ⇒ 不做这层映射时，中文题抽不出
#   任何有效关键词，会退化到"题干里的英文样板词"（实测 12 题撞成同一串）。
# 选取原则：只收**在 Mathlib 声明名里有对应**的词，避免映射到搜不到的词。
_CN_TERM_MAP = [
    ("概率", "Probability"), ("期望", "expectation"), ("方差", "variance"),
    ("正态分布", "normal distribution"), ("随机变量", "random variable"),
    ("统计", "statistics"), ("回归", "regression"), ("相关", "correlation"),
    ("时间序列", "time series"), ("抽样", "sampling"), ("假设检验", "hypothesis"),
    ("线性规划", "linear programming"), ("对偶", "duality"),
    ("最优化", "optimization"), ("约束", "constraint"), ("目标函数", "objective"),
    ("矩阵", "Matrix"), ("行列式", "determinant"), ("特征值", "eigenvalue"),
    ("向量", "Vector"), ("线性", "linear"), ("秩", "rank"), ("逆", "inverse"),
    ("多项式", "polynomial"), ("方程", "equation"), ("根", "root"),
    ("因式", "factor"), ("次数", "degree"), ("系数", "coefficient"),
    ("质数", "prime"), ("素数", "prime"), ("整除", "divisible"),
    ("余数", "remainder"), ("同余", "congruent"), ("最大公约数", "gcd"),
    ("最小公倍数", "lcm"), ("互质", "coprime"), ("阶乘", "factorial"),
    ("组合", "combination"), ("排列", "permutation"), ("二项式", "binomial"),
    ("计数", "count"), ("集合", "Set"), ("子集", "Subset"),
    ("映射", "function"), ("双射", "bijection"), ("函数", "function"),
    ("导数", "derivative"), ("微分", "derivative"), ("积分", "integral"),
    ("连续", "continuous"), ("极限", "limit"), ("收敛", "convergent"),
    ("级数", "series"), ("数列", "sequence"), ("单调", "monotone"),
    ("有界", "bounded"), ("三角", "triangle"), ("角", "angle"),
    ("圆", "circle"), ("面积", "area"), ("周长", "perimeter"),
    ("中点", "midpoint"), ("垂直", "perpendicular"), ("平行", "parallel"),
    ("图", "Graph"), ("顶点", "vertex"), ("边", "edge"), ("路径", "path"),
    ("树", "tree"), ("着色", "colorable"), ("不等式", "inequality"),
    ("绝对值", "absolute value"), ("取整", "floor"), ("群", "Group"),
    ("环", "Ring"), ("域", "Field"), ("同态", "homomorphism"),
    ("递推", "recurrence"), ("归纳", "induction"), ("求和", "summation"),
]


def cn_term_keywords(problem: str, limit: int = 5) -> list:
    """中文题干 → 英文数学术语（命中即映射；未命中返回空）。"""
    text = problem or ""
    out: list = []
    for cn, en in _CN_TERM_MAP:
        if cn in text:
            for w in en.split():
                if w not in out:
                    out.append(w)
            if len(out) >= limit:
                break
    return out[:limit]


def question_keywords(problem: str, limit: int = 6) -> list:
    """从题干抽**数学实词**，拼成兜底 query。

    ★ 2026-09-29 实测修正：原实现只做「英文字母序列 - 停用词」，抽出来的是
      ``set / ordered / pairs / integer / real / numbers`` 这类**通用词**，
      在 Mathlib 源码扫描后端上区分度极差 —— 实测同领域多道题的 rank-1
      命中恒为同一条（``card_perms_of_finset``），检索退化。
      现按三级优先抽取：
        ① 英文数学术语（``_MATH_TERMS``）
        ② **中文数学术语**（``_CN_TERM_MAP``）—— 中文题唯一的有效信号
        ③ 通用英文实词（兜底，已剔除答案格式样板词）
    """
    text = problem or ""
    low = text.lower()

    # ① 英文数学术语
    math_hits: list = []
    for term in _MATH_TERMS:
        if term in low and term not in math_hits:
            math_hits.append(term)
    # ② 中文术语（英中混合也要）
    cn_hits = cn_term_keywords(text, limit=limit)

    primary = math_hits + [w for w in cn_hits if w not in math_hits]
    if primary:
        out = primary[:limit]
        for g in _generic_tokens(text):
            if len(out) >= limit:
                break
            if g not in out:
                out.append(g)
        return out

    # ③ 退回：通用英文实词（对无术语抽象的题目仍有价值）
    return _generic_tokens(text, limit)


def _generic_tokens(text: str, limit: int = 6) -> list:
    """题干里的英文实词（去停用词、去纯数字/公式碎片）。"""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_']{2,}", text or "")
    out: list = []
    for t in toks:
        tl = t.lower()
        if tl in _STOPWORDS or tl in _NOISE_TOKENS:
            continue
        if len(tl) < 3:
            continue
        # 排除 LaTeX 命令残渣（ldots / leq / frac / sqrt 等）
        if tl.startswith("\\") or tl in _TEX_NOISE:
            continue
        if tl not in out:
            out.append(tl)
        if len(out) >= limit:
            break
    return out


def build_queries(problem: str, domain: str = "",
                  question_type: str = "",
                  question_first: bool = False) -> list:
    """构造检索 query 列表。

    ★ 2026-09-29 修正：**题干词置于领域词之前**。
      原顺序（领域优先）会导致同领域不同题的首条 query 完全相同 ⇒ 首条命中
      列表雷同、淹没题干信息（实测 000/003/004 三题命中列表一致）。
      现默认题干优先，领域词作为**补充**（而非主导）。

    参数 question_first=False 时保持旧顺序（领域优先），仅供对照实验。
    """
    qs: list = []
    dk = domain_keywords(domain, question_type)
    qk = question_keywords(problem)
    d_q = " ".join(dk) if dk else ""
    q_q = " ".join(qk) if qk else ""
    mixed = " ".join((qk[:3] if qk else []) + (dk[:2] if dk else []))

    # ★ 2026-09-29 二次修正：**领域 query 降级为最后一条**。
    #   实测（8 题）：领域 query 排第 1 时，其宽泛命中（如 离散数学→Finset.card
    #   →card_perms_of_finset）会占据 rank-1，8 题里 6 题的 rank-1 是同一条，
    #   检索无法反映"这道题"。改为领域 query 兜底（仅在题干 query 无命中时才生效）
    #   后，rank-1 去重数由 1 → 4，几何题正确浮出 angle_ne_zero_of_not_collinear。
    for q in (q_q, mixed, d_q):
        if q:
            qs.append(q)
    if question_first:
        qs.reverse()
    # 去重保序
    seen: set = set()
    out: list = []
    for q in qs:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# 主入口：agent/ 侧唯一的定理检索接口
# ---------------------------------------------------------------------------

# 分环节注入条数的**默认表**（2026-10-12，用户 §5.2）：前置理解少、主求解/修订多。
# 为什么前置理解只给 4：读题对标阶段给多了反而干扰审题；主求解/修订要全量线索。
# ★ 2026-10-14（任务列表 #2）：检索已下移到 2.7 子目标求解，`subgoal` 由此成为
#   **检索的主消费者**（蓝图 12 条在其后、merge 用同一批），按任务原文「这个环节
#   的产出就是 25 个定理」把 8 → 25，与 `solve` 齐平。
# ★ 2026-10-14 同日：`pre_audit` 一项**保留但已无调用者** —— 2.6 的两处定理注入
#   （audit_gate.confirm_understanding / lean_pre_verifier 理解链）已随检索下移撤除；
#   保留键位是为了 `render_for_stage(ctx, "pre_audit")` 在外部脚本/历史数据里
#   仍能取到确定性条数（不返回 1/2 全量的兜底值）。
DEFAULT_TOPK_BY_STAGE = {
    "pre_audit": 4, "blueprint": 12, "subgoal": 25,
    "solve": 25, "revise": 25,
}


def render_for_stage(ctx, stage: str = "solve", max_items: int = 0) -> str:
    """按环节渲染定理线索块（**分环节条数**，用户 §5.2/§5.3）。

    为什么需要：`ctx.theorem_hint_block` 是**检索期渲染一次**的字符串，被 8 个
    注入点共用 ⇒ 无法做到「前置理解只给 3–5 条、蓝图 10–15 条、主求解 10–25 条」。
    本函数从 `ctx.theorem_hint_result`（检索**全量**，含陈述与前提）按环节重新渲染。

    · 无检索结果 / 数量 ≤0 ⇒ 返回空串（下游零成本）；
    · 条数优先级：显式 `max_items` > `theorem_hint_top_k_by_stage[stage]` > 1/2 全量。
    """
    res = getattr(ctx, "theorem_hint_result", None)
    if not isinstance(res, dict) or not res.get("ok"):
        return ""
    raw_hits = res.get("hits") or []
    if not raw_hits:
        return ""
    n = int(max_items or 0)
    if n <= 0:
        by_stage = getattr(ctx, "_theorem_topk_by_stage", None) or {}
        try:
            n = int(by_stage.get(stage) or 0)
        except (TypeError, ValueError):
            n = 0
    if n <= 0:
        n = max(1, len(raw_hits) // 2)
    hits = []
    for h in raw_hits[:max(1, n)]:
        if not isinstance(h, dict):
            continue
        hits.append(TheoremHit(
            name=str(h.get("name") or ""), short=str(h.get("short") or ""),
            source=str(h.get("source") or ""),
            rank=int(h.get("rank") or 0),
            snippet=str(h.get("snippet") or "")))
    if not hits:
        return ""
    # ★ 2026-10-13 观测补齐：把「本环节实际注入几条」写进 ctx
    #   （此前无法判定分环节裁剪是否生效 —— 审核时被列为「无法判定」）。
    try:
        _usage = getattr(ctx, "theorem_hint_stage_usage", None)
        if not isinstance(_usage, dict):
            _usage = {}
            ctx.theorem_hint_stage_usage = _usage
        _usage[str(stage)] = len(hits)
    except Exception:  # noqa: BLE001  纯观测，绝不影响主流程
        pass
    return TheoremRetrieval(ok=True, hits=hits).render_hint_block(
        max_items=max(1, n))


THEOREM_QUERY_LLM_SYSTEM = """你是 Mathlib 检索助手。给你一道数学题，请给出 **2–3 条**
用于检索 Mathlib 定理的**英文关键词串**。

要求：
1. 只写**数学概念 / 结构的英文词**（如 "prime number infinite"、"Euclid theorem"），
   **不要写定理名**（写定理名会诱发幻觉，检索器只用它做关键词匹配）；
2. 每条 3–6 个词，覆盖**不同方向**（不要三条同义）；
3. 只输出一个 JSON 数组，例如 ["prime number infinitely many", "Euclid proof"]；
   不要解释、不要 markdown 围栏。"""


def _looks_like_query(s: str) -> bool:
    """判断一个候选串是否**像**一条 Mathlib 关键词 query。

    ★ 2026-10-14（任务 #2 A/B 小样实测暴露）：本函数是「垃圾 query 注入」的闸门。

    事故链（小样 trace 原文）：
      1. 本次调用 `max_tokens=512`，而 `intern-s2-preview` 是**推理模型**，先吐
         "Thinking Process" 块 ⇒ 512 token 全被思考吃掉，JSON 数组从未产出
         （日志：`finish_reason=length, max_tokens=512, <- theorem_hint.py:675`）；
      2. `utils/llm_client.py` 对推理模型**优先返回 reasoning_content**
         ⇒ 调用方拿到的是**被截断的思考文本**，而不是答案；
      3. 旧 `parse_llm_queries` 的②号降级路径把这段思考文本按行切分，产出
         `'Problem Analysis:**'` / `'y) \\mid 0 \\leq x \\leq 19'` 这类**碎片**；
      4. 检索层把 LLM query **前置**并占用配额 ⇒ 碎片把正确的规则 query 挤掉，
         `source_scan` 在碎片上匹配出 99 条噪声命中全量注入提示词
         （实测 `adopted_ratio=0.0`，一条也没被采用）。

    ⇒ 结论：**垃圾 query 比没有 query 更糟**（没有 query 会走规则兜底拿回正确命中）。
      故降级路径必须**宁缺毋滥**：像不像「几个纯英文概念词」是硬门槛。
    """
    _s = (s or "").strip()
    if not (4 <= len(_s) <= 60):
        return False
    # 禁字符：markdown / LaTeX / JSON / 命题符号 —— 均为思考文本或题干的特征
    for _bad in ("**", "\\", "$", "{", "}", "`", "<", ">", "=", "#", "|",
                 "(", ")", "[", "]", "_", "////", "分析", "题目", "要求"):
        if _bad in _s:
            return False
    # ★ 叙述性开头黑名单：思考文本被截断后最典型的残留形态是「一句英文叙述」。
    #   注意不能只靠"词数/字符"筛（'The problem asks for the number' 是 6 个纯英文词，
    #   字符与词数检查全过），必须识别**叙述语气**。
    _low = _s.lower()
    for _lead in ("the problem", "this problem", "the question", "this question",
                  "the task", "we need", "we must", "we should", "we are asked",
                  "i need", "i will", "i should", "let us", "let's", "first,",
                  "next,", "then,", "finally,", "step ", "analysis",
                  "requirement", "therefore", "note that", "given that",
                  "to solve", "so the", "in this", "here ", "think",
                  "thinking", "reasoning", "the answer", "we can"):
        if _low.startswith(_lead) or (" " + _lead.strip(" ,") + " ") in (" " + _low + " "):
            return False
    # 词数 2–8，且**至少 2 个纯英文词**（发音词/概念词），排除单点长句
    _words = [w for w in _s.split() if w]
    if not (2 <= len(_words) <= 8):
        return False
    _alpha = [w for w in _words
              if len(w) >= 2 and all(c.isascii() and (c.isalpha() or c == "-")
                                     for c in w)]
    return len(_alpha) >= 2


def parse_llm_queries(raw) -> list:
    """把 LLM 输出的 query 解析成字符串列表（**容错**：JSON 数组 / 逗号分隔 / 逐行）。

    为什么要容错：本项目实测「要求 JSON 但模型给了别的东西」很常见
    （题意理解链曾因四合一输出被截断而整体不可解析）⇒ 解析失败必须**降级**而不是丢弃。
    返回去重后的列表；无法解析 ⇒ 空列表（下游用规则 query 兜底）。

    ⚠ 2026-10-14（任务 #2 A/B 小样）：降级路径加**质量闸门** `_looks_like_query`。
      推理模型的思考文本被截断后回传时，旧的裸行切分会产生垃圾 query，
      而垃圾 query 会被**前置**并挤掉规则 query ⇒ 注入整片噪声。
      现在垃圾一律丢弃，宁可返回 `[]` 让规则 query 兜底。详见 `_looks_like_query`。
    """
    if not raw:
        return []
    txt = str(raw).strip()
    out = []
    # ① 优先按 JSON 数组解析（模型按要求产出时走这条，可信度高）
    try:
        import json as _json_q
        _m = txt[txt.find("["):txt.rfind("]") + 1] if "[" in txt else ""
        if _m:
            _arr = _json_q.loads(_m)
            if isinstance(_arr, list):
                out = [str(x).strip() for x in _arr if str(x).strip()]
    except Exception:  # noqa: BLE001
        out = []
    # ★ JSON 路径也要过闸门：思考文本里也可能出现 `[...]`，解析成功≠内容可信
    out = [q for q in out if _looks_like_query(q)]
    # ② 退化为按行 / 逗号切分（**必须过闸门**，见 docstring 事故链）
    if not out:
        import re as _re_q
        for _ln in _re_q.split(r"[\n,;；]+", txt):
            _s = _ln.strip().strip("").strip('').strip("[]").strip()
            # ⚠ 必须剥掉**列表符号与序号**（实测模型常输出 ' - prime' / '1. Euclid' 这种形态）
            _s = _s.lstrip("-•*·").strip()
            _s = _s.lstrip("0123456789").lstrip(".)、 ").strip()
            if _looks_like_query(_s):
                out.append(_s)
    # ③ 去重保序 + 上限 3 条
    seen, uniq = set(), []
    for q in out:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq[:3]


def retrieve_theorems_for_question(
    problem: str,
    domain: str = "",
    question_type: str = "",
    config=None,
    max_queries: int = 2,
    top_k: int = 5,
    extra_queries=None,
) -> TheoremRetrieval:
    """题目 + 领域 → Mathlib 定理候选（**永不抛异常**）。

    参数
    ----
    problem        : 题干原文
    domain         : 领域名（来自 1_classify，可为空）
    question_type  : 题型（来自 classify_question_type，可为空）
    config         : AgentConfig（读开关；可传 None 用默认）
    max_queries    : 最多发几次检索（默认 2，与 leansearch_max_calls_per_q 对齐）
    top_k          : 每次检索取回条数（与 leansearch_top_k 对齐）

    返回
    ----
    TheoremRetrieval —— ok=False 表示后端不可用/异常（**调用方据此跳过即可**）。
    """
    t0 = time.time()
    out = TheoremRetrieval(domain=domain)

    # 开关（默认开；设 0 可关，便于 A/B「有定理 vs 无定理」）
    if config is not None and not bool(
            getattr(config, "enable_theorem_hint", True)):
        out.reason = "开关关闭（enable_theorem_hint=False）"
        return out
    if not _sw_bool("theorem_hint"):
        out.reason = "环境变量 THEOREM_HINT=0"
        return out

    # 客观题/判断题：答案是选项字母或真值，定理检索无意义
    if question_type and question_type in ("选择题", "判断题"):
        out.reason = f"题型为{question_type}，跳过定理检索"
        return out
    if not (problem or "").strip():
        out.reason = "题干为空"
        return out

    # ★ 2026-10-13（用户 A1「借大模型的力量找定理」）：**LLM 出的 query 前置**，
    #   规则 query 兜底；合并后**放宽上限**（LLM 2–3 条 + 规则 max_queries 条）。
    #   为什么放宽：原实现 `[:max_queries]` 会把 LLM 刚给出的方向直接截掉（静默无效）。
    _rule_q = build_queries(problem, domain, question_type)
    _llm_q = [str(x).strip() for x in (extra_queries or []) if str(x).strip()]
    _seen_q, queries = set(), []
    for _q in list(_llm_q) + list(_rule_q):
        _k = _q.lower()
        if _k not in _seen_q:
            _seen_q.add(_k)
            queries.append(_q)
    queries = queries[:max(0, max_queries) + len(_llm_q)]
    if not queries:
        out.reason = "未能构造检索 query（题干无英文实词且领域未收录）"
        return out
    out.queries = list(queries)

    # ---- 检索执行（2026-10-15 工具②接线）----
    # 后端开关 theorem_hint_backend：
    #   "leanexplore"（默认）= 工具② tools/agent_toolkit.leansearch
    #        （LeanExplore 主通道 + 离线语料降级；实测官方预标注 112 题命中 17.9%）
    #   "legacy"             = 原 MathlibTheoremSearcher（官方 API / 源码扫描）
    _backend = "leanexplore"
    try:
        from contracts.switch_registry import get_str as _sw_get_str
        _backend = (str(_sw_get_str("theorem_hint_backend") or "").strip()
                    or "leanexplore")
    except Exception:  # noqa: BLE001  读不到开关 ⇒ 默认通道
        pass

    _legacy_searcher = None
    _tool_search = None
    if _backend == "legacy":
        try:
            from tools.lean_local.lean_search import MathlibTheoremSearcher
        except Exception as e:  # noqa: BLE001
            out.reason = f"lean_search 导入失败: {type(e).__name__}: {e}"
            out.elapsed = time.time() - t0
            logger.warning("[theorem_hint] %s", out.reason)
            return out
        try:
            _legacy_searcher = MathlibTheoremSearcher()
            st = _legacy_searcher.status() or {}
            if not st.get("available"):
                out.reason = f"检索后端不可用: {st}"
                out.elapsed = time.time() - t0
                return out
        except Exception as e:  # noqa: BLE001
            out.reason = f"检索异常: {type(e).__name__}: {e}"
            out.elapsed = time.time() - t0
            return out
        out.backend = "legacy_mathlib_search"
    else:
        try:
            from tools.agent_toolkit import leansearch as _ls_tool
        except Exception as e:  # noqa: BLE001
            out.reason = f"工具②（leansearch）导入失败: {type(e).__name__}: {e}"
            out.elapsed = time.time() - t0
            logger.warning("[theorem_hint] %s", out.reason)
            return out
        _tool_search = _ls_tool
        out.backend = "leanexplore_tool"

    def _search_one(q: str):
        """一次检索 → (归一化 results, backend_tag)；两条通道同一结构。"""
        if _tool_search is not None:
            r = _tool_search(q, top_k=top_k) or {}
            items = [{"name": t.get("name", ""),
                      "snippet": (t.get("statement")
                                  or t.get("informal") or "")}
                     for t in (r.get("theorems") or [])]
            return items, str(r.get("backend") or "")
        res = _legacy_searcher.search(q, limit=top_k) or {}
        return list(res.get("results") or []), "legacy"

    try:
        seen_names: set = set()
        hits: list = []
        _be_seen = ""
        for q in queries:
            try:
                _results, _be = _search_one(q)
            except Exception as e:  # noqa: BLE001
                logger.warning("[theorem_hint] query=%r 检索异常: %s", q, e)
                continue
            if _be:
                _be_seen = _be
            for i, item in enumerate(_results, start=1):
                name = str(item.get("name") or "").strip()
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                hits.append(TheoremHit(
                    name=name,
                    short=name.rsplit(".", 1)[-1],
                    source=q,
                    rank=i,
                    # ★ 2026-10-04：把检索层已有的陈述片段带上来（此前被丢弃）
                    snippet=" ".join(str(item.get("snippet") or "").split())[:240],
                ))
        if _be_seen and _tool_search is not None:
            # 记录实际生效的通道（含降级：leanexplore / offline）
            out.backend = "leanexplore_tool:%s" % _be_seen
        # ★ 2026-09-29 修复（实测暴露）：多 query 结果必须**按位次重排**后再截断。
        #   原实现是「先到先得、跑满 top_k 就停」—— 但领域 query（第 1 条）本身就
        #   会返回 top_k 条，于是题干 query 的命中全被挤在后面。
        #   实测后果：同领域的三道不同题（000/003/004）拿回**完全相同**的命中列表，
        #   检索退化成了"领域查询" ⇒ 无法反映"这道题"该用什么定理。
        #   现在改为：并集后按 (rank 升序) 排序，位次越靠前越优先保留。
        hits.sort(key=lambda h: (h.rank, h.name))
        out.hits = hits[:max(top_k, 1) * max(len(queries), 1)]
        out.ok = True
        if not out.hits:
            out.reason = "检索跑通但零命中"
    except Exception as e:  # noqa: BLE001  —— 任何异常都不许冒泡到主流程
        out.reason = f"检索异常: {type(e).__name__}: {e}"
        logger.warning("[theorem_hint] %s", out.reason)

    out.elapsed = time.time() - t0
    return out


def search_for_subgoal(ctx, sg, top_k: int = 6) -> str:
    """★ 2026-10-18（用户「每个子目标都用 leansearch」）：按**子目标文本**检索
    定理（而非全局一次），返回渲染好的定理块；失败/无结果返回空串（调用方回退
    全局块）。结果按子目标 id 缓存（3 遍迭代同一子目标只检索一次）。
    """
    try:
        from contracts.switch_registry import get_bool as _g
        if not _g("subgoal_theorem_search"):
            return ""
        from tools.agent_toolkit import leansearch as _ls
    except Exception:  # noqa: BLE001
        return ""
    _txt = " ".join(str(sg.get(k) or "") for k in ("title", "description")).strip()
    if not _txt:
        return ""
    _cache = getattr(ctx, "_subgoal_hints", None)
    if not isinstance(_cache, dict):
        _cache = {}
        try:
            ctx._subgoal_hints = _cache
        except Exception:  # noqa: BLE001
            pass
    _key = str(sg.get("id") or _txt[:40])
    if _key in _cache:
        return _cache[_key]
    block = ""
    try:
        r = _ls(_txt, top_k=int(top_k)) or {}
        items = [TheoremHit(
            name=str(t.get("name") or ""),
            short=str(t.get("name") or "").rsplit(".", 1)[-1],
            source=_txt, rank=i + 1,
            snippet=str(t.get("statement") or t.get("informal") or ""))
            for i, t in enumerate((r.get("theorems") or [])[:top_k])
            if str(t.get("name") or "")]
        block = (TheoremRetrieval(ok=True, hits=items)
                 .render_hint_block(max_items=top_k)) if items else ""
    except Exception as e:  # noqa: BLE001  检索是增益，绝不阻断
        logger.warning("[theorem_hint] per-subgoal 检索异常: %s", str(e)[:120])
    _cache[_key] = block
    return block


def run_hint_for(agent, ctx) -> None:
    """在**任意 Agent 上下文**里执行一次定理检索，并把结果写回 ``ctx``。

    ★ 2026-10-14（任务列表 #2「定理检索移到后面子目标求解过程中」）：
      本函数承载原 ``orchestrator__lean._retrieve_theorem_hints`` 的**全部逻辑**，
      但**不绑定编排层** —— 参数只需鸭子类型（``agent.llm`` / ``agent.record``
      / ``agent.config``），故子目标求解器可原地调用，无需反向 import
      ``orchestrator__*``（保持 agent 包内依赖方向自洽）。

    调用时机：``sub_goal_solver.run`` 体首（**2.7 内、蓝图构建之前**）。
      顺序由此变为「2.6 判题 → 2.7 检索 → 蓝图/子目标规划 → 求解 → merge」：
        · 蓝图 3 处、子目标 merge/solve 2 处、3_solve 3 处都仍在检索**之后**
          ⇒ 消费点零改动；
        · 顺带修掉 #1 遗留的顺序代价（检索此前的题型的来源是规则判定）。

    写出的字段（与搬移前逐字一致，下游零改动）：
      · ``ctx.theorem_hints``        : list[str] 命中的 Mathlib 全名
      · ``ctx.theorem_hint_block``   : str 默认注入块（主求解口径）
      · ``ctx.theorem_hint_trace``   : dict 完整埋点（query/命中/耗时/后端）
      · ``ctx.theorem_hint_result``  : dict 全量结果，供 `render_for_stage` 按环节重渲染
      · ``ctx._theorem_topk_by_stage``: dict 分环节条数表

    ⚠ 失败与超时一律降级（不阻断主流程）：检索只是**增益**，不是门禁。
    """
    # ★ 2026-10-13（用户 A1）：**LLM 辅助 query** —— 让大模型先给出检索方向，
    #   再交检索器出事实（LLM 不直接吐定理名，防幻觉）。失败 ⇒ 完全回退规则 query。
    _extra_q: list = []
    if getattr(agent.config, "enable_theorem_query_llm", True):
        try:
            _raw_q = agent.llm(ctx, [
                {"role": "system", "content": THEOREM_QUERY_LLM_SYSTEM},
                {"role": "user", "content": "题目：%s\n领域：%s" % (
                    ctx.problem or "", getattr(ctx, "domain", "") or "未知")},
            # ★ 2026-10-14（任务 #2 A/B 小样实测）：512 → 2048。
            #   `intern-s2-preview` 是**推理模型**，先输出一长段 "Thinking Process"
            #   再给 JSON 数组。512 token 连思考都写不完 ⇒ `finish_reason=length`
            #   ⇒ content 空 ⇒ `llm_client` 回退返回**截断的 reasoning_content**
            #   ⇒ 解析器拿到思考碎片当 query（实测产 99 条噪声命中）。
            #   2048 给思考留出空间，使真正的 JSON 数组能产出。
            #   代价极小：这只是一次短输出调用，且失败仍回退规则 query。
            ], 0.2, 8192)
            _extra_q = parse_llm_queries(_raw_q)
            agent.record(ctx, "theorem_hint",
                         "LLM 辅助 query：%s" % (_extra_q or "（空，回退规则 query）"))
        except Exception as _e_q:  # noqa: BLE001  检索是增益，绝不阻断
            agent.record(ctx, "theorem_hint",
                         "LLM 辅助 query 失败（回退规则 query）: %s" % str(_e_q)[:100])

    try:
        res = retrieve_theorems_for_question(
            ctx.problem or "",
            domain=getattr(ctx, "domain", "") or "",
            question_type=getattr(ctx, "question_type", "") or "",
            config=agent.config,
            max_queries=int(getattr(
                agent.config, "theorem_hint_max_queries", 2) or 2),
            top_k=int(getattr(agent.config, "leansearch_top_k", 5) or 5),
            extra_queries=_extra_q,
        )
    except Exception as e:  # noqa: BLE001  # 兜底：接口内已 try，此处再保险
        agent.record(ctx, "theorem_hint",
                     f"定理检索异常（跳过，不阻断）: {type(e).__name__}: {e}")
        return

    ctx.theorem_hints = list(res.names)
    md = res.to_dict()
    # 与题库预标注比对（若 metadata 里带了该题标注；云端/离线注入时才有）
    gt = None
    try:
        _md = getattr(ctx, "metadata", None)
        if isinstance(_md, dict):
            gt = _md.get("theorem_gt")
    except Exception:  # noqa: BLE001
        gt = None
    if gt:
        try:
            md["gt_match"] = match_against_ground_truth(res, gt)
        except Exception as e:  # noqa: BLE001
            md["gt_match"] = {"error": f"{type(e).__name__}: {e}"}
    ctx.theorem_hint_trace = md
    # ★ 2026-10-12（批次 5 剩余，用户 §5.2）：保留**全量**检索结果，
    #   供 `render_for_stage()` 按环节重新渲染（分环节条数）。
    ctx.theorem_hint_result = md
    # 分环节条数：config 里是 **JSON 字符串**（注册表无 dict 类型）⇒ 在此解析。
    # 解析失败/为空 ⇒ 用模块内默认表（避免「配置写错就静默失效」）。
    try:
        import json as _json_th
        _raw_topk = str(getattr(
            agent.config, "theorem_hint_top_k_by_stage", "") or "").strip()
        _parsed = _json_th.loads(_raw_topk) if _raw_topk else {}
        ctx._theorem_topk_by_stage = (dict(_parsed) if isinstance(_parsed, dict)
                                      else dict(DEFAULT_TOPK_BY_STAGE))
        if not ctx._theorem_topk_by_stage:
            ctx._theorem_topk_by_stage = dict(DEFAULT_TOPK_BY_STAGE)
    except Exception:  # noqa: BLE001  解析失败 ⇒ 回退默认表，绝不阻断
        ctx._theorem_topk_by_stage = dict(DEFAULT_TOPK_BY_STAGE)

    # 默认注入块（主求解口径）。分环节取用请调 `render_for_stage(ctx, "<stage>")`；
    # 本块口径与搬移前逐字一致，**不改变**既有 8 个直接读 `theorem_hint_block`
    # 的消费点行为。
    try:
        ctx.theorem_hint_block = res.render_hint_block(
            max_items=int(getattr(agent.config, "leansearch_top_k", 5) or 5),
            max_chars=int(getattr(
                agent.config, "theorem_hint_block_max_chars", 2500) or 0))
    except Exception as e:  # noqa: BLE001  渲染失败不阻断
        logger.warning("[theorem_hint] 渲染注入块失败，跳过注入: %s", e)
        ctx.theorem_hint_block = ""

    agent.record(
        ctx, "theorem_hint",
        f"定理检索 {'命中 ' + str(len(res.hits)) + ' 条' if res.ok else '未跑通'}"
        f"（{res.elapsed:.1f}s，后端 {res.backend or '-'}）"
        + (f"；{res.reason}" if res.reason else ""),
        n_hits=len(res.hits), elapsed=round(res.elapsed, 2),
        backend=res.backend)


# ---------------------------------------------------------------------------
# 评审：命中的定理是否 == 预标注的正确定理
# ---------------------------------------------------------------------------

def match_against_ground_truth(
    retrieval: TheoremRetrieval,
    gt: dict,
) -> dict:
    """把一次检索结果与题库预标注比对，产出**可上报的三指标**。

    gt 形如::

        {"required": ["Nat.choose", "Finset.card_perms_of_finset"],
         "optional": ["Nat.factorial"]}

    匹配规则（**宽松后缀匹配**）：
      预标注可写末段短名（``choose``）、限定短名（``Nat.choose``）或全名
      （``Mathlib.Data.Nat.Choose.Basic.Nat.choose``）。命中项同样可写成
      全名或任意后缀。判定为「命中」当且仅当**存在一对（标注, 命中）使得
      两者互为后缀或相等**。这样：
        · Mathlib 重构命名空间前缀 → 比对仍稳定；
        · 标注写 ``Nat.choose`` 而命中是 ``...Basic.Nat.choose`` → 命中；
        · 标注写 ``choose`` 而命中是 ``...Basic.Nat.choose`` → 命中。

    返回::

        {"required_n": 2, "required_hit": 1, "required_miss": ["..."],
         "hit_rate": 0.5, "all_required_hit": False,
         "best_rank": 2, "optional_hit": 0}
    """
    req = [str(x).strip() for x in (gt or {}).get("required") or [] if str(x).strip()]
    opt = [str(x).strip() for x in (gt or {}).get("optional") or [] if str(x).strip()]

    # 命中侧：收集全名与**所有点分段后缀**（"A.B.C.d" → A.B.C.d / B.C.d / C.d / d）
    got_segs: set = set()
    rank_of: dict = {}
    for h in retrieval.hits:
        full = h.name
        got_segs.add(full)
        parts = full.split(".")
        for i in range(len(parts)):
            seg = ".".join(parts[i:])
            got_segs.add(seg)
            rank_of.setdefault(seg, h.rank)

    def _hit(name: str) -> bool:
        """标注名是否命中：本身在命中集，或它与某命中项互为后缀。"""
        if name in got_segs:
            return True
        # 反向：标注写全名而命中只记了短名的情况
        parts = name.split(".")
        for i in range(len(parts)):
            if ".".join(parts[i:]) in got_segs:
                return True
        return False

    req_hit = [n for n in req if _hit(n)]
    req_miss = [n for n in req if not _hit(n)]
    opt_hit = [n for n in opt if _hit(n)]

    best_rank = None
    for n in req_hit:
        # 取该标注在各命中项里的最优位次
        for h in retrieval.hits:
            if h.name == n or h.name.endswith("." + n) or n.endswith("." + h.short) \
                    or h.name == n.split(".")[-1] or h.short == n.split(".")[-1]:
                if best_rank is None or h.rank < best_rank:
                    best_rank = h.rank
                break

    return {
        "required_n": len(req),
        "required_hit": len(req_hit),
        "required_miss": req_miss,
        "hit_rate": (len(req_hit) / len(req)) if req else None,
        "all_required_hit": bool(req) and not req_miss,
        "best_rank": best_rank,
        "optional_n": len(opt),
        "optional_hit": len(opt_hit),
    }


__all__ = [
    "TheoremHit", "TheoremRetrieval",
    "domain_keywords", "question_keywords", "build_queries",
    "retrieve_theorems_for_question", "match_against_ground_truth",
    "render_for_stage", "DEFAULT_TOPK_BY_STAGE",
    "run_hint_for",
]
