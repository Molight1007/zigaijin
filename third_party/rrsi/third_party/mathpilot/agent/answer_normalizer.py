# -*- coding: utf-8 -*-
"""4.0 答案规范化前置 —— 五道审核链第 0 道（2026-10-14 用户定案）。

用户口径（原话）：「整体我认为既然会用到 lean 和计算工具，就要提前把答案规范化！」

职责边界（单一来源纪律）：
  · 归一**算法**全部复用 ``utils/extract.normalize_pool_answer`` 的
    「先抢救、再判定」纯函数内核（句子形态含明确最终值 ⇒ 抽出该值），
    本模块**不新写第二个归一实现**；
  · 本模块只做四件事：**分类**（answer_kind）→ **挂载**（answer_norm /
    answer_kind 写到候选对象上，**不改 ``c.answer``**）→ **剔除不可判形态**
    （``unparseable``：Markdown 标记/过程叙述/空答案 —— 承接旧链 4_verify
    ``_verify_filter_candidates`` 的过滤职责，2026-10-14 五道链定案收编）
    → **留痕**（trace step=``answer_normalize``，diag 键
    ``answer_normalize`` / ``answer_normalize_events``）。

answer_kind 六类（下游五道的分流依据）：
  · numeric      纯数值（整数/小数/分数/科学计数）→ 第四道计算检测主对象
  · symbolic     符号表达式（含未知数/运算符）→ 第三/四道
  · multi        多答案（顶层逗号分隔 ≥2 项且每项都像单个答案）→ 第五道多答判断
  · proof        证明题候选（无数值答案）→ 第三道走整题 verify、第四道跳过
  · text         自然语言文本答案（论述/概念）→ 只参与一、二、五道
  · unparseable  非答案形态（Markdown 标记/过程叙述/拒绝语）→ 不参与三、四道

五道审核链总设计：规范化前置 → ①AI 自审 → ②对抗证伪 → ③Lean 代回 →
④计算检测 → ⑤投票（先判多答案再投票）。旧审核链按用户 2026-10-14 决策
「一步到位直接删」（删除在 Step7，本模块先落地为纯增量）。
"""
from __future__ import annotations

import re

# ---- 开关读取（注册制：env > config > 默认；与 AgentConfig.five_gate_normalize
#      的默认值 True 逐字一致，详见 contracts/switch_registry.py）----
from .switch_registry import get_bool as _sw_get_bool
# 纯数值：整数 / 小数 / 分数 / 百分数 / 科学计数（LaTeX 壳已由归一化剥掉）
_NUMERIC_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:/[+-]?(?:\d+(?:\.\d+)?|\.\d+))?"
    r"%?(?:[eE][+-]?\d+)?$")

# 符号式信号：字母（未知数/常量/函数名，归一化后 \pi→pi、\sqrt→sqrt()）
# 或比较/幂运算符。纯中文论述不含 ASCII 字母 ⇒ 落 text。
_SYMBOLIC_RE = re.compile(r"[A-Za-z]|[=<>≤≥^]")

# 多答案判定：顶层 ASCII 逗号分隔 ≥2 项、每项非空 ≤40 字符、
# 至少一半的项「像单个答案」（数值或短符号式）。宁缺勿滥：
# 长论述里的逗号（如「步骤1，先…」）不得误判为多答案。
_MULTI_MAX_ITEMS = 10

# Markdown 行首结构标记（标题/列表/引用/表格）⇒ 非答案形态。
# ★ 与 3.6 `_verify_filter_candidates._keep_cand` 的行首正则**同口径**
#   （该判据在 3.6 是独立于 `_looks_like_non_answer` 的第一道检查；
#   Step7 删旧链时该过滤职责整体并入本模块，届时此正则为唯一来源）。
_MD_LINE_RE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|\|\s)")


def _looks_single_item(item: str) -> bool:
    """单答案项判据：纯数值，或 ≤40 字符且含字母/数字的短符号式/选项字母。"""
    s = item.strip()
    if not s or len(s) > 40:
        return False
    if _NUMERIC_RE.match(s):
        return True
    return bool(re.search(r"[0-9A-Za-z]", s)) and not re.search(r"[，。；：\n]", s)


# ★ 2026-10-18（p0b5_r4-015 实测：答案被写成 `\boxed{n ≥ 2 且 n 为正整数}`，
#   条件式被当成答案进入投票）。判据：含「且 / 并且 / ∧ / ∀ / ∃ / 对所有」
#   ⇒ 非答案形态（unparseable，不参与三/四道）。
#   ⚠ 不把「或」列入：多答案常写作 `x=1 或 x=2`，拒掉会误杀。
_COND_SHAPE_RE = re.compile(r"(且|并且|∧|∀|∃|对于所有|对所有)")


def classify_answer_kind(norm: str, is_proof: bool = False) -> str:
    """对**已归一化**的答案文本做形态分类（纯函数，便于单测覆盖）。

    判定顺序（先窄后宽）：
      空 → unparseable；证明题 → proof；非答案形态 → unparseable；
      纯数值 → numeric；多答案 → multi；符号式 → symbolic；其余 → text。
    """
    s = str(norm or "").strip()
    if not s:
        return "unparseable"
    if is_proof:
        return "proof"
    if _MD_LINE_RE.match(s):
        return "unparseable"
    try:
        from .formatter__nonanswer import _looks_like_non_answer as _nma
        if _nma(s):
            return "unparseable"
    except Exception:  # noqa: BLE001  形态判据失败 ⇒ 按 text 处理，不阻断
        pass
    if _NUMERIC_RE.match(s):
        return "numeric"
    # 条件式/命题式（如 `n >= 2 且 n 为正整数`）不是答案 ⇒ 不参与三/四道
    # 开关 answer_shape_strict（默认 True；关=恢复旧行为）
    try:
        _strict = _sw_get_bool("answer_shape_strict")
    except Exception:  # noqa: BLE001
        _strict = True
    if _strict and _COND_SHAPE_RE.search(s):
        return "unparseable"
    items = [p for p in s.split(",")]
    if 2 <= len(items) <= _MULTI_MAX_ITEMS:
        ok = sum(1 for it in items if _looks_single_item(it))
        if ok == len(items):
            return "multi"
    if len(s) <= 80 and _SYMBOLIC_RE.search(s):
        return "symbolic"
    return "text"


def _switch_enabled() -> bool:
    """读「五道链·第 0 道规范化」开关（默认 True，=0 关闭做 A/B）。"""
    try:
        return _sw_get_bool("five_gate_normalize")
    except Exception:  # noqa: BLE001  注册表异常 ⇒ 走默认（与注册表默认一致）
        return True


def normalize_stage(self, ctx) -> None:
    """4.0 阶段入口（orchestrator 一行委托；``self`` 即 Orchestrator 实例，
    参数模式与 orchestrator__*.py 各阶段函数一致）。

    幂等性：对已挂 ``answer_norm`` 的候选**重算覆盖**（候选可能在
    3.2/3.3/3.4 阶段被改写；重入时以最新 answer 为准）。
    绝不抛异常 —— 规范化失败只留痕，不阻断主链（下游各道各自兜底）。
    """
    try:
        if not _switch_enabled():
            self.record(ctx, "answer_normalize",
                        "4.0 答案规范化：开关关闭（five_gate_normalize=0）")
            return
        cands = list(getattr(ctx, "candidates", None) or [])
        if not cands:
            self.record(ctx, "answer_normalize",
                        "4.0 答案规范化：候选池为空，跳过")
            return

        from utils.extract import normalize_pool_answer

        qtype = str(getattr(ctx, "question_type", "") or "")
        is_proof = (qtype == "证明题"
                    or str(getattr(ctx, "domain", "") or "") in ("证明", "证明题"))

        kinds: dict = {}
        n_salvaged = 0
        for c in cands:
            raw = str(getattr(c, "answer", "") or "")
            try:
                norm, usable = normalize_pool_answer(raw, qtype or None)
            except Exception:  # noqa: BLE001  归一失败 ⇒ 原样保留、标 unparseable
                norm, usable = raw, False
            if usable and norm.strip() != raw.strip():
                n_salvaged += 1
            kind = classify_answer_kind(norm, is_proof)
            try:
                c.answer_norm = norm
                c.answer_kind = kind
            except Exception as _e_mount:  # noqa: BLE001
                self.record(ctx, "answer_normalize",
                            f"挂载失败 cid={getattr(c, 'id', '?')}: "
                            f"{type(_e_mount).__name__}")
                continue
            kinds[kind] = kinds.get(kind, 0) + 1
            self.record(ctx, "answer_normalize",
                        "cid=%s kind=%s norm=%s"
                        % (getattr(c, "id", "?"), kind, str(norm)[:120]))

        # ---- 剔除「不可判形态」（承接旧链 4_verify `_verify_filter_candidates`
        #      的过滤职责；2026-10-14 五道链定案收编到 4.0）----
        # 背景（旧判据实证）：096 候选 answer 混入 Markdown 标题污染投票；
        #   032 的过程叙述候选（「步骤11：搜索已知结论」）同样污染票型。
        # 语义（与旧链逐字一致）：
        #   · 部分剔除 ⇒ 落池；
        #   · **全部**不可判 ⇒ 由 `pool_empty_clear` 决定：True（默认）清空池
        #     交下游兜底（用户 Q32）；False 保留脏候选（2026-09-18 旧行为）。
        try:
            _kept_c = [c for c in cands
                       if str(getattr(c, "answer_kind", "") or "") != "unparseable"]
            if _kept_c and len(_kept_c) < len(cands):
                ctx.candidates = _kept_c
                self.record(ctx, "answer_normalize",
                            "剔除不可判形态候选 %d → %d（Markdown/过程叙述/空答案）"
                            % (len(cands), len(_kept_c)))
            elif not _kept_c:
                if bool(getattr(self.config, "pool_empty_clear", True)):
                    ctx.candidates = []
                    self.record(ctx, "answer_normalize",
                                "候选池 %d 个全部呈不可判形态 ⇒ **已清空**"
                                "（pool_empty_clear=True，走下游兜底）" % len(cands))
                    # ★ 2026-10-15（审核 S2）：清池后把「池为何被清」喂给下游
                    #   （4.6 重写诊断读 ctx.pool_empty_errors）——此前不调用
                    #   ⇒ 4.6 拿到空错误清单、根因丢失。
                    try:
                        from .orchestrator__verify import (
                            _pool_empty_fallback as _pef)
                        _pef(self, ctx, "4.0_normalize")
                    except Exception as _e_pef:  # noqa: BLE001
                        self.record(ctx, "answer_normalize",
                                    "pool_empty_fallback 失败（不阻断）: %s: %s"
                                    % (type(_e_pef).__name__,
                                       str(_e_pef)[:100]))
                else:
                    self.record(ctx, "answer_normalize",
                                "候选池 %d 个全部呈不可判形态（pool_empty_clear=False，"
                                "保留原池）" % len(cands))
        except Exception as _e_f:  # noqa: BLE001  过滤失败不影响已挂载的归一结果
            self.record(ctx, "answer_normalize",
                        f"不可判形态剔除失败（{type(_e_f).__name__}: "
                        f"{str(_e_f)[:120]}）→ 脏候选可能进入后续环节")

        # 汇总挂 ctx（diag_export 导出为 diag["answer_normalize"]）
        ctx.answer_normalize = {
            "n": len(cands),
            "kinds": kinds,
            "n_salvaged": n_salvaged,
            "is_proof": is_proof,
        }
        self.record(
            ctx, "answer_normalize",
            "4.0 答案规范化：%d 候选 → 形态分布 %s（抢救改写 %d%s）"
            % (len(cands),
               ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
               n_salvaged,
               "；证明题（三四道走整题 verify）" if is_proof else ""))
    except Exception as _e:  # noqa: BLE001  增益环节绝不阻断主链
        self.record(ctx, "answer_normalize",
                    f"4.0 答案规范化异常（跳过，不阻断）: "
                    f"{type(_e).__name__}: {str(_e)[:120]}")
