# -*- coding: utf-8 -*-
"""run_eval 评测引擎 + 调用埋点快照 —— 2026-10-08 从 run_eval.py 逐字搬出。

含 `DEFAULT_AGENT_OVERRIDES`（原 L989-1070）、三个调用快照（原 L877-930）、
`EvalEngine`（原 L1073-1377）。判决/行为零变更；run_eval.py 原位 re-export ⇒
`run_eval.EvalEngine` / `run_eval.DEFAULT_AGENT_OVERRIDES` / `run_eval._mcp_call_snapshot`
均**同一对象**。日志 logger 名保持 MathPilot.Eval（同一 logging 单例）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from user_agent import ReasoningAgent
from utils.llm_client import LLMClient

from run_eval__classify import _classify_error
from run_eval__scoring import answers_match

logger = logging.getLogger("MathPilot.Eval")


# ===========================================================================
# 评测引擎
# ===========================================================================

# 本地评测默认参数。
#
# 重要（2026-08-28 修正）：此前这份配置是"全开"版本，与 user_agent.py 的平台默认
# 不一致（samples 3 vs 2、calls 40 vs 150、time 1100 vs 1200、scoring/lemma 开关相反），
# 导致本地测出来的数字不能代表平台表现。当时已把大部分键对齐平台默认值。
#
# ★★ 2026-09-16 更正（用户：「现在的测试和比赛没关系，比赛已经结束了」）：
#   上面那句"**现已全部对齐平台默认值，保证本地基线 == 平台基线**"**已不成立**，
#   实测仍有 **3 个键与 AgentConfig 默认不同**（本条改前没有任何注释说明）：
#     · `tier_budget`              本地 {fast 120, standard 540, deep 1150}
#                                 vs 配置 {300, 750, 1150}
#     · `use_lemma_accumulation`   本地 False  vs 配置 True
#     · `verifier_voting_times`    本地 2      vs 配置 1
#   ⇒ **本地基线 ≠ 平台基线**。比赛已结束，**不再以"对齐平台"为目标**：
#   本地这份就是**研究阶段基线**，有意与平台版解耦（平台版已冻结不再维护）。
#   ⚠ 引用历史 A/B 结论时，须注意它们分别是在哪一套基线下取得的。
# 需要临时偏离本地基线时，请用 CLI 显式指定（如 `--tier_budget`、`--voting_times`）。
DEFAULT_AGENT_OVERRIDES: Dict[str, Any] = {
    # ---- 对齐 user_agent.py:64 / :72 ----
    "policy_sample_times": 2,
    "verifier_voting_times": 2,
    # ---- 对齐 user_agent.py:84 / :94 ----
    "max_total_calls": 150,
    # 2026-09-03 老师："先让它完成题目，看对不对；先改对再节约时间"。
    # 本地评测单题时限放宽到 3600s（1h 封顶防意外挂死），让每题自然跑完
    # 全部环节（子目标/求解/验证/Lean 闸门），不被 1200s 截断。比赛平台
    # 时限是平台侧约束（官方 Client 托管），与本文件无关。
    # 2026-09-04 平台教训（14.29% 归因）：放开单题时限 → 时间爆炸 → 64 题被
    # 时间墙切掉 invalid。改回比赛档 1200s（deep 档上限）——超时截断宁可 invalid
    # 也不拖垮整卷。
    # 2026-09-14 实测落实：1200 → 1150 → **1100**，与提交配置
    # `user_agent.py::AgentConfig.max_time_per_question` 对齐。
    # 依据：同批错题实测两次超限（并发3 轮 1164.7s / 并发1 轮 **1211s > 1200 越墙**）。
    # ⚠ **只改这一处硬限**；`tier_budget.deep` 仍为 1150（档位预算管资源分配，
    #   压它会提前掐断本可在 1200s 内跑完的题）。
    # 2026-10-01 按用户决策：研究期不限时，此值仅作挂死兜底，不参与调度决策；
    #   调度侧已不再有任何按剩余时间降级的逻辑（soft_budget/gen_deadline/verify_only 触发路径已删）。
    "max_time_per_question": 86400,
    # ---- 对齐 user_agent.py:101 / :105 / :106 ----
    "max_workers": 3,
    # 9/4：平台不限 token → 本地 override 同步放开（防截断腰斩；上探 65536 对齐 AgentConfig）
    "max_answer_tokens": 65536,
    "revise_sample_times": 2,
    # 2026-10-01：原值 1 会把自纠错回环整体压到 1 轮（审计 A 级第 7 条）。
    # 与 `user_agent.py:293` 声明的 5 分叉 ⇒ 口径不一致且属比赛期省时裁剪。
    # 研究期求正确率上限，对齐配置声明值 5。
    "max_revise_rounds": 5,
    # ---- 对齐 user_agent.py（2026-10-01 按用户决策默认开，研究期不省资源）----
    "use_scoring": True,
    "use_proof_channel": True,
    # 2026-10-01：原先这里把 `use_lemma_accumulation` 压回 False，与
    # `AgentConfig.use_lemma_accumulation = True` 自相矛盾（跑 eval 时被静默关闭）。
    # 已移除该 override，改为继承配置默认值（仍可经 CLI 显式传参覆盖）。
    # 2026-09-29：`by_enable_fast_path` 已随快车道删除（死开关，无行为作用）。
    # ---- 统一档位预算（2026-09-29）----
    # 三档（fast/standard/deep）已按用户决策收敛为**单一档位**，取值 = 原 deep 最强配置。
    # 原 `{fast:120, standard:540, deep:1150}` 与 `paper_*` 全卷调度参数随
    # PaperPacer 删除；全卷不再有时间总量控制（研究阶段云端评测不限时）。
    # 单题唯一约束是下方 `max_time_per_question` 硬墙。
    "tier_budget": {"deep": 86400.0},
    # 统一档位其余参数（与 user_agent.py __post_init__ 保持一致）
    "tier_sample_times": {"deep": 3},
    "tier_voting_times": {"deep": 3},
    "tier_max_completions": {"deep": 2},
    "tier_max_calls": {"deep": 100},
    # 前置验证最多 2 次尝试（原默认 2 轮 = 3 次，每次 21s 编译 + LLM 调用，
    # 单题可烧掉 3-5 分钟；preverify 是「检查理解」不是「写论文」，1 轮足够）
    # 2026-09-11：#13 前置验证修复——轮数 1→2（即最多 3 轮尝试）。
    # 依据：112 题实测 13 题 preverify fail，全部是「形式化代码编译错误」
    # （臆造 API/类型不匹配/语法错/引用未定义谓词），而原配置仅 2 轮机会；
    # 配合 prompt 增补的「常见错误规避」清单，给修正留出足够轮次。
    "preverify_max_rounds": 2,
    # 2026-09-11（B1 独立化）：启用"子目标数值断言的 Lean 核验"。
    # 此前该开关默认 False → B1 只能靠 P2 数值化间接触发，覆盖面过窄（实测 0/10）。
    # 打开后由 _numeric_lean_verify 对子目标里的数值断言做 Lean 复核
    # （走当前后端 mcp；每题次数受 lean_numeric_max_per_q 限制，默认 2 → 成本可控）。
    "enable_numeric_lean_verify": True,
}


def _llm_call_snapshot() -> dict:
    """LLM 调用计数的全局快照（用于算单题增量）。失败时返回空统计，绝不打断评测。

    来源：`utils.llm_client.get_truncation_stats()` —— 每次响应结束计数一次，
    流式/非流式两条路径都覆盖（含真实 finish_reason 口径的截断数）。
    """
    try:
        from utils.llm_client import get_truncation_stats
        st = get_truncation_stats() or {}
        return {"calls": int(st.get("calls", 0) or 0),
                "truncated": int(st.get("truncated", 0) or 0)}
    except Exception:  # noqa: BLE001
        return {"calls": 0, "truncated": 0}


def _token_snapshot() -> dict:
    """token 用量的全局快照（用于算单题增量）。

    ★ 2026-10-24 审计层：与 `_llm_call_snapshot` **数据源同源**
    （`utils.llm_client.get_usage_stats()`），但**不并入 `llm_calls`** ——
    那个字段的语义是"调用次数"，混进 token 会污染既有下游
    （`test_result_detail_export` 断言 `llm_calls["calls"]` 为 int）。
    缺字段一律补 0，绝不打断评测。
    """
    try:
        from utils.llm_client import get_usage_stats
        st = get_usage_stats() or {}
        return {"prompt_tokens": int(st.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(st.get("completion_tokens", 0) or 0),
                "total_tokens": int(st.get("total_tokens", 0) or 0),
                "calls_with_usage": int(st.get("calls_with_usage", 0) or 0),
                "usage_unavailable": int(st.get("usage_unavailable", 0) or 0)}
    except Exception:  # noqa: BLE001
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "calls_with_usage": 0, "usage_unavailable": 0}


def _mcp_call_snapshot() -> dict:
    """Lean MCP 诊断调用的全局快照（用于算单题增量）。

    ★ 2026-09-16 用户要求「这些工具的调用的效果也都记录」。
    来源：`tools.lean_local.lean_bridge.mcp_stats()`（calls / ok / fail / seconds）。
    **判读口径**（这是回答"到底用了 MCP 还是 bridge"的关键证据）：
      · `calls>0, ok>0`      → MCP 真正生效；
      · `calls>0, fail==calls` → MCP **每次都失败** ⇒ 实际由 bridge 完成（0916 轮即此）；
      · `calls==0`           → 根本没走 MCP。
    失败时返回零值，绝不打断评测。
    """
    try:
        from tools.lean_local.lean_bridge import mcp_stats
        st = mcp_stats() or {}
        return {"calls": int(st.get("calls", 0) or 0),
                "ok": int(st.get("ok", 0) or 0),
                "fail": int(st.get("fail", 0) or 0),
                "seconds": float(st.get("seconds", 0.0) or 0.0)}
    except Exception:  # noqa: BLE001
        return {"calls": 0, "ok": 0, "fail": 0, "seconds": 0.0}


def _web_call_snapshot() -> dict:
    """通用联网搜索的全局快照（用于算单题增量）。

    来源：`tools.web_search.web_search_stats()`（calls/ok/fail/seconds/results）。
    与 `_mcp_call_snapshot` 同口径 ⇒ 逐题 `tool_calls` 可并列比较
    "Lean MCP 调用" 与 "联网搜索调用" 的实际效果。
    """
    try:
        from tools.web_search import web_search_stats
        st = web_search_stats() or {}
        return {"calls": int(st.get("calls", 0) or 0),
                "ok": int(st.get("ok", 0) or 0),
                "fail": int(st.get("fail", 0) or 0),
                "seconds": float(st.get("seconds", 0.0) or 0.0),
                "results": int(st.get("results", 0) or 0)}
    except Exception:  # noqa: BLE001
        return {"calls": 0, "ok": 0, "fail": 0, "seconds": 0.0, "results": 0}


class EvalEngine:
    def __init__(self, concurrency: int = 1, resume: bool = False,
                 api_key: str = "", base_url: str = "", model: str = "",
                 verbose: bool = False, agent_overrides: Optional[Dict[str, Any]] = None):
        self.concurrency = concurrency
        self.resume = resume
        self.verbose = verbose
        # 创建 LLM 客户端（通过环境变量或参数配置）
        self.llm_client = LLMClient(
            api_key=api_key or None,
            base_url=base_url or None,
            model=model or None,
        )
        # 配置覆盖：先取本地评测默认值，再叠加 CLI 传入的 A/B 开关
        overrides = dict(DEFAULT_AGENT_OVERRIDES)
        if agent_overrides:
            overrides.update(agent_overrides)
        # 保存生效覆盖（小样本自适应要用），避免二次重建时丢失
        self._effective_overrides = overrides
        self.agent = ReasoningAgent(self.llm_client, **overrides)
        # ★ 2026-10-02 中间结果存储层接线：本轮评测 run_id（一轮一个目录：
        #   results/<run_id>/<qid>/）。显式设定后由 artifact_store 统一取用；
        #   平台直调（不经本脚本）时回退进程级稳定默认值。失败不阻断评测。
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        try:
            from agent.artifact_store import set_run_id as _set_run_id
            _set_run_id(self.run_id)
        except Exception:  # noqa: BLE001
            try:
                from artifact_store import set_run_id as _set_run_id
                _set_run_id(self.run_id)
            except Exception:  # noqa: BLE001
                pass
        logger.info("EvalEngine init: %s, overrides=%s", self.llm_client, overrides)
        self.domain_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )

    def load_tests(self, filepath: str) -> List[Dict[str, Any]]:
        tests = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"第 {line_no} 行 JSON 解析失败")
                    continue
                if "question" not in item and "problem" not in item:
                    logger.warning(f"第 {line_no} 行缺少 question/problem")
                    continue
                # 统一规范化字段名（本地题库兼容：problem/question、subject/domain、idx/id）
                if "question" not in item:
                    item["question"] = item["problem"]
                if "domain" not in item and "subject" in item:
                    item["domain"] = item["subject"]
                if "id" not in item and "idx" in item:
                    item["id"] = item["idx"]
                item["_line_no"] = line_no
                tests.append(item)
        logger.info(f"加载 {len(tests)} 道测试题")
        return tests

    def solve_one(self, test: Dict[str, Any]) -> Dict[str, Any]:
        question = test["question"]
        gold = test.get("answer", "")
        domain = test.get("domain", "unknown")
        pid = test.get("id", str(test.get("_line_no", "?")))
        start = time.time()
        # ★ 2026-09-16 用户要求「极为详细的数据」：记录本题的 LLM 调用增量。
        # 全局计数器是进程累计的，必须做前后快照求差才等于"本题用了多少次"。
        _llm_before = _llm_call_snapshot()
        _mcp_before = _mcp_call_snapshot()
        _web_before = _web_call_snapshot()
        # ★ 2026-10-24 审计层：token 也做前后差分 ⇒ 「本题用了多少 token」
        _tk_before = _token_snapshot()
        # ★★ 2026-09-16 修既有 bug（由 tests/test_result_detail_export.py 抓到）：
        # `result` 原先**只在 try 内赋值**，一旦 `agent.solve()` 抛异常，
        # 紧随其后的 `isinstance(result, dict)` 会抛 `UnboundLocalError`
        # ⇒ 整个 `solve_one` 崩溃、**该题在结果里彻底消失**（不是记为错，是丢失），
        # 批量评测时可能直接打断整轮。此处先给默认值。
        result: Dict[str, Any] = {}
        try:
            result = self.agent.solve(question, {})
            elapsed = time.time() - start
            pred_answer = result.get("final_response", "") if isinstance(result, dict) else ""
            response = result.get("final_response", "") if isinstance(result, dict) else str(result) if result else ""
        except Exception as e:
            logger.error(f"题目 {pid} 求解异常: {e}", exc_info=True)
            elapsed = time.time() - start
            pred_answer = ""
            response = f"ERROR: {e}"
            result = {}
        _llm_after = _llm_call_snapshot()
        llm_calls = {k: (_llm_after.get(k, 0) - _llm_before.get(k, 0))
                     for k in set(_llm_before) | set(_llm_after)}
        # ★ 2026-10-24 审计层：本题 token 增量（并发>1 时是进程级串味值，
        #   run manifest 会把 cost_accuracy 标成 "coarse"）
        _tk_after = _token_snapshot()
        tokens = {k: (_tk_after.get(k, 0) - _tk_before.get(k, 0))
                  for k in _tk_after}
        # ★ 工具级埋点（用户 2026-09-16 要求）：本题 Lean MCP 的调用次数/成功/失败/耗时。
        _mcp_after = _mcp_call_snapshot()
        mcp_calls = {
            "calls": _mcp_after["calls"] - _mcp_before["calls"],
            "ok": _mcp_after["ok"] - _mcp_before["ok"],
            "fail": _mcp_after["fail"] - _mcp_before["fail"],
            "seconds": round(_mcp_after["seconds"] - _mcp_before["seconds"], 2),
        }
        # 派生判读：MCP 是否真正生效（否则实际由 bridge 完成）
        if mcp_calls["calls"] == 0:
            mcp_calls["verdict"] = "未走 MCP"
        elif mcp_calls["ok"] > 0:
            mcp_calls["verdict"] = "MCP 生效"
        else:
            mcp_calls["verdict"] = "MCP 全部失败 → 实际由 bridge 完成"
        # ★ 联网搜索埋点（用户 2026-09-16 要求「这些工具的调用的效果也都记录」）
        _web_after = _web_call_snapshot()
        web_calls = {
            "calls": _web_after["calls"] - _web_before["calls"],
            "ok": _web_after["ok"] - _web_before["ok"],
            "fail": _web_after["fail"] - _web_before["fail"],
            "results": _web_after["results"] - _web_before["results"],
            "seconds": round(_web_after["seconds"] - _web_before["seconds"], 2),
        }
        web_calls["verdict"] = ("未调用" if web_calls["calls"] == 0
                                else "成功" if web_calls["ok"] > 0
                                else "全部失败")
        is_correct = answers_match(pred_answer, gold) if pred_answer and gold else None
        if not gold:
            is_correct = None
        # 错误分类：仅对判错样本分类（#6/#11/#16），用于定位瓶颈与决定优化投向
        error_class = ""
        if is_correct is False:
            error_class = _classify_error(pred_answer, gold)
        # Mathlib 使用证据（2026-08-29）：AI 检索/验证用到的定理 + 使用统计
        used_theorems = []
        if isinstance(result, dict):
            used_theorems = list(result.get("used_theorems") or [])
        usage_stats = {}
        if isinstance(result, dict):
            usage_stats = result.get("mathlib_usage_stats") or {}
        # 逐步归因诊断（2026-09-02）：orchestrator 已打包各阶段中间状态，
        # 这里落盘为结果行 diag 字段（错题可定位到理解/蓝图/子目标/验证等环节）
        diag = {}
        if isinstance(result, dict):
            diag = result.get("diag") or {}
        # ★ 2026-09-16 用户要求「极为详细的数据：大模型的解答过程 / 时间消耗 / 各环节效果」。
        # 此前只落盘 response[:2000] 与有限 diag，**候选的解答过程、全量事件流都没导出**
        # ⇒ 事后无法复盘"模型到底怎么想的、在哪一步跑偏"。以下四项把过程完整留痕：
        #   · response_full  最终回答**全文**（不再截断）
        #   · candidates     每个候选的 answer + **reasoning（解答过程）** + 得票
        #   · trace          全量事件流（各 agent/step 的顺序与内容）
        #   · verdicts/cluster 验证裁决与共识簇
        #   · llm_calls      本题 LLM 调用次数（全局计数前后快照之差）
        # ⚠ 体积可控：单题约几十 KB，10 题 ~1MB（此前每行只存 2KB 摘要）。
        cand_out: list = []
        verd_out: list = []
        cluster_out = None
        trace_out: list = []
        if isinstance(result, dict):
            _verds = result.get("verdicts") or []
            for i, c in enumerate(result.get("candidates") or []):
                if not isinstance(c, dict):
                    continue
                v = _verds[i] if i < len(_verds) and isinstance(_verds[i], dict) else {}
                cand_out.append({
                    "id": c.get("id", i),
                    "answer": c.get("answer", ""),
                    "reasoning": c.get("reasoning", ""),     # ★ 解答过程
                    "revised": bool(c.get("revised")),
                    "confidence": v.get("confidence"),
                    "correct_votes": v.get("correct_votes"),
                    "total_votes": v.get("total_votes"),
                    "feedback": v.get("feedback", ""),
                })
            for v in _verds:
                if isinstance(v, dict):
                    verd_out.append({
                        "id": v.get("id"),
                        "answer": v.get("answer", ""),
                        "confidence": v.get("confidence"),
                        "correct_votes": v.get("correct_votes"),
                        "total_votes": v.get("total_votes"),
                    })
            cluster_out = result.get("cluster")
            trace_out = result.get("trace") or []
        return {
            "id": pid, "domain": domain,
            "question": question, "gold": gold,
            "predicted": pred_answer, "response": response[:2000],
            # ★ 最终回答全文（不截断）
            "response_full": response,
            "correct": is_correct, "elapsed_sec": round(elapsed, 2),
            "error_class": error_class,
            # #1/#2 证据链：实际用到的 Mathlib 定理（leansearch 命中/编译通过）
            "used_theorems": used_theorems,
            "mathlib_usage_stats": usage_stats,
            # 逐步归因诊断（可空：老结果文件无此字段）
            "diag": diag,
            # ★ 2026-09-16 新增：过程留痕（候选解答过程 / 事件流 / 裁决 / 调用次数）
            "candidates": cand_out,
            "verdicts": verd_out,
            "cluster": cluster_out,
            "trace": trace_out,
            "llm_calls": llm_calls,
            # ★ 2026-10-24 审计层：本题 token（prompt/completion/total）
            "tokens": tokens,
            # ★ 工具级埋点：Lean MCP + 通用联网搜索（calls/ok/fail/seconds + 派生判读）
            "tool_calls": {"lean_mcp": mcp_calls, "web_search": web_calls},
        }

    def _write_manifest(self, output_file: str, test_file: str,
                        tests: List[Dict]) -> None:
        """写 run 身份 sidecar：``<output>.manifest.json``（2026-10-24 审计层）。

        为什么必须：δ（噪声带）的语义是「**同一版本**重复评测」。仅凭文件名
        无法证明 5 份结果是同一版本；一旦中间改过一行代码，δ 就变成跨版本
        差异，测出来的噪声带毫无意义。因此记录三元身份：

          · ``git_sha``      —— 提交号；
          · ``worktree_sha`` —— `git diff HEAD` 的指纹（未提交改动也算版本的一部分）；
          · ``bank_sha``     —— 题集文件指纹（防止题集被换过）。

        `delta` 子命令据此硬校验，不一致直接拒绝。失败只告警，绝不阻断评测。
        """
        try:
            import hashlib
            import subprocess as _sp

            _root = os.path.dirname(os.path.abspath(__file__))

            def _run_git(args, timeout=15):
                try:
                    r = _sp.run(["git"] + args, cwd=_root, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=timeout)
                    return (r.stdout or "").strip()
                except Exception:  # noqa: BLE001
                    return ""

            _sha = _run_git(["rev-parse", "HEAD"]) or "unknown"
            _diff_text = _run_git(["diff", "HEAD"], timeout=30)
            _worktree_sha = (hashlib.sha256(_diff_text.encode("utf-8")).hexdigest()[:16]
                             if _diff_text else "clean")
            _bank_sha = ""
            try:
                with open(test_file, "rb") as fh:
                    _bank_sha = hashlib.sha256(fh.read()).hexdigest()[:16]
            except OSError:
                pass
            mf = {
                "run_id": self.run_id,
                "git_sha": _sha,
                "worktree_sha": _worktree_sha,
                "git_dirty": _worktree_sha != "clean",
                "model": getattr(self.llm_client, "model", ""),
                "base_url": getattr(self.llm_client, "base_url", ""),
                "test_file": os.path.abspath(test_file),
                "bank_sha": _bank_sha,
                "question_ids": [str(t.get("id", t.get("_line_no"))) for t in tests],
                "n_questions": len(tests),
                "concurrency": self.concurrency,
                # 诚实字段：进程级计数器在多线程下差分必然串味
                "cost_accuracy": "exact" if int(self.concurrency) <= 1 else "coarse",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "resume": bool(self.resume),
                # 开关快照：没有它，两版结果无法解释"差在哪"
                "engine_overrides": {
                    k: v for k, v in (self._effective_overrides or {}).items()
                    if isinstance(v, (bool, int, float, str)) or v is None},
            }
            with open(output_file + ".manifest.json", "w", encoding="utf-8") as fh:
                json.dump(mf, fh, ensure_ascii=False, indent=2)
            logger.info("run manifest: %s.manifest.json (sha=%s wt=%s)",
                        output_file, _sha[:8], _worktree_sha)
        except Exception as e:  # noqa: BLE001
            logger.warning("run manifest 写入失败（不阻断评测）：%s", e)

    def run(self, test_file: str, output_file: str) -> Dict[str, Any]:
        tests = self.load_tests(test_file)
        self._write_manifest(output_file, test_file, tests)
        done_ids = set()
        results = []
        if self.resume and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            done_ids.add(r.get("id"))
                            results.append(r)
                            domain = r.get("domain", "unknown")
                            self.domain_stats[domain]["total"] += 1
                            if r.get("correct"):
                                self.domain_stats[domain]["correct"] += 1
                        except json.JSONDecodeError:
                            pass
            logger.info(f"断点续跑：跳过 {len(done_ids)} 道已完成")
        pending = [t for t in tests if str(t.get("id", t.get("_line_no"))) not in done_ids]
        logger.info(f"待评测: {len(pending)} / 总计: {len(tests)}")
        # ---- 2026-09-29：小样本自适应块已删除 ----
        # 原逻辑在"待评测题数 < 全卷题数"时重建 agent，调整 PaperPacer 的
        # `paper_total_questions` / `paper_target_time`，以免小卷被误判"卷面落后"
        # 而收紧单题预算。PaperPacer 已整体删除（全卷无时间总量控制）⇒ 该
        # workaround 失去存在前提，整块移除。
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as executor:
            future_map = {executor.submit(self.solve_one, t): t for t in pending}
            for future in as_completed(future_map):
                row = future.result()
                results.append(row)
                domain = row.get("domain", "unknown")
                self.domain_stats[domain]["total"] += 1
                if row.get("correct"):
                    self.domain_stats[domain]["correct"] += 1
                # 增量落盘（2026-08-28 新增）：每题完成立即追加写盘。
                # 此前全部跑完才写一次文件，中途 Ctrl-C / 杀进程 / 超时
                # 会丢掉所有已完成题目的结果。增量写 + --resume 断点续跑，
                # 保证任何时刻中断都能保留进度。
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.sort(key=lambda r: str(r.get("id", "")))
        with open(output_file, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"结果已写入 {output_file}")
        return self._build_summary(results)

    def _build_summary(self, results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        scored = [r for r in results if r["correct"] is not None]
        correct = sum(1 for r in scored if r["correct"])
        accuracy = correct / len(scored) if scored else 0.0
        avg_elapsed = sum(r.get("elapsed_sec", 0) for r in results) / max(total, 1)
        summary = {
            "total": total, "scored": len(scored),
            "correct": correct, "accuracy": round(accuracy, 4),
            "avg_elapsed_sec": round(avg_elapsed, 2),
            "per_domain": {},
        }
        for domain, stats in sorted(self.domain_stats.items()):
            tot = stats["total"]
            cor = stats["correct"]
            summary["per_domain"][domain] = {
                "total": tot, "correct": cor,
                "accuracy": round(cor / tot, 4) if tot else 0.0,
            }

        # 错误分类分布（#6/#11/#16）——决定下一轮资源投向的核心依据
        wrong = [r for r in results if r.get("correct") is False]
        err_dist: Dict[str, int] = {}
        for r in wrong:
            k = r.get("error_class") or "unclassified"
            err_dist[k] = err_dist.get(k, 0) + 1
        n_wrong = len(wrong)
        summary["error_distribution"] = {
            "wrong_total": n_wrong,
            "counts": err_dist,
            "ratios": {k: round(v / n_wrong, 4) for k, v in err_dist.items()} if n_wrong else {},
        }
        # 决策提示：按计划 §2 P0.5 的决策规则给出建议，避免每次人工判读
        ratio = summary["error_distribution"]["ratios"]
        # 非推理类（不需要提升推理能力就能修）：空输出 + 抽取失败 + 答案未定型
        cheap = (ratio.get("empty_output", 0.0)
                 + ratio.get("extract_failed", 0.0)
                 + ratio.get("format_unresolved", 0.0))
        if n_wrong:
            if cheap >= 0.25:
                summary["recommendation"] = (
                    f"先做答案定型（Phase 1-A）：非推理类错误占比 {cheap:.0%}，"
                    "不动推理即可提分，风险最低")
            elif ratio.get("value_wrong", 0.0) >= 0.55:
                summary["recommendation"] = (
                    f"主攻答案题推理深度：value_wrong 占比 {ratio['value_wrong']:.0%}，"
                    "瓶颈在推理能力，投多候选/投票/revise")
            else:
                summary["recommendation"] = (
                    "天花板在推理深度，投 deep 通道扩容（多候选 + 预算生效）")
        return summary
