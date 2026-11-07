#!/usr/bin/env python
from __future__ import annotations

# 2026-10-01 开关注册制（审查 A 级第 2 条）：开关统一走 switch_registry，
# 不再裸读 os.environ —— 既保持 env 优先级（行为不变），又能被 diag/报告还原。
# 2026-10-08 拆分：判分族搬到 run_eval__scoring.py 后，本文件已不再直接读开关
#   （判分族的 `_sw_bool` 在 scoring 模块就地 import）；此行保留作**契约面**
#   （`run_eval._sw_bool` 仍可解析），故标 noqa。
try:
    from agent.switch_registry import (  # noqa: F401
        get_bool as _sw_bool, get_num as _sw_num, get_str as _sw_str)
except ImportError:
    from switch_registry import (  # noqa: F401
        get_bool as _sw_bool, get_num as _sw_num, get_str as _sw_str)
# -*- coding: utf-8 -*-
"""
MathPilot 本地评测脚本 —— 仅用于本地开发调试，非平台正式评测调用入口。
平台只调用 user_agent.py 的 ReasoningAgent.solve()，不会执行此文件。
支持 JSONL 题库批量评测、答案规范化匹配、领域细分统计、断点续跑。

本文件为本地版（题库注册表）与赛事版（答案提取增强 / A/B 能力开关）的合并版：
- 保留本地题库注册表: --bank 新高数 / 1000题高数 / 高数a / IMO-AnswerBench / IMO-ProofBench / all
- 引入赛事版答案匹配增强: _extract_equals_candidates 结论提取、LaTeX 符号归一化
- 引入赛事版 A/B 能力开关: --voting_times / --use_scoring / --revise_rounds / --use_proof / ...

用法:
    python run_eval.py --test_file tests.jsonl --output results.jsonl
    python run_eval.py --test_file tests.jsonl --concurrency 4 --resume results.jsonl
    python run_eval.py --bank 新高数 --output results.jsonl
    python run_eval.py --bank 1000题高数 --concurrency 4
    python run_eval.py --bank all --concurrency 4
"""

import argparse
import json
import logging
import os
import re  # noqa: F401  （拆分后本文件不再直接用 re；保留作契约面）
import sys
import time  # noqa: F401  （评估引擎搬到 run_eval__engine.py 后不再直接用）
from collections import defaultdict  # noqa: F401  （同上，保留作契约面）
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from typing import Any, Dict, List, Optional  # noqa: F401  （Optional 仅引擎用）

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("MathPilot.Eval")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from user_agent import ReasoningAgent  # noqa: F401  （引擎用；此处保留作契约面）
from utils.llm_client import LLMClient  # noqa: F401  （引擎用；此处保留作契约面）

# ===========================================================================
# 题库注册表
# ===========================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA = os.path.join(PROJECT_ROOT, "sample_data")

# 题库名称 → 文件路径映射
BANK_REGISTRY: Dict[str, str] = {
    # ---- 自建题库 ----
    "新高数": os.path.join(SAMPLE_DATA, "新高数.jsonl"),
    "1000题高数": os.path.join(SAMPLE_DATA, "1000题高数.jsonl"),
    "高数a": os.path.join(SAMPLE_DATA, "高数a.jsonl"),
    # ---- IMO-Bench (Google DeepMind 公开基准) ----
    "IMO-AnswerBench": os.path.join(SAMPLE_DATA, "IMO-AnswerBench.jsonl"),
    "IMO-ProofBench": os.path.join(SAMPLE_DATA, "IMO-ProofBench.jsonl"),
}

def resolve_bank(bank_name: str):
    """解析题库名称，返回 JSONL 文件路径。支持 'all' 返回所有题库路径。"""
    if bank_name == "all":
        return [p for _, p in BANK_REGISTRY.items() if os.path.exists(p)]
    if bank_name in BANK_REGISTRY:
        path = BANK_REGISTRY[bank_name]
        if os.path.exists(path):
            return path
        else:
            logger.error(f"题库 '{bank_name}' 的文件不存在: {path}")
            return None
    return None

def list_banks():
    """列出所有已注册的题库。"""
    print("\n已注册的题库:")
    print("-" * 60)
    for name, path in BANK_REGISTRY.items():
        if os.path.exists(path):
            count = sum(1 for _ in open(path, "r", encoding="utf-8"))
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name:<15} {count:>5} 题  {size_kb:>8.1f} KB  ({path})")
        else:
            print(f"  {name:<15} [文件缺失] ({path})")
    print("-" * 60)
    print("用法: python run_eval.py --bank <题库名> [其他参数]")
    print("       python run_eval.py --bank all [其他参数]  # 评测所有题库")
    print()


# ===========================================================================
# 2026-10-08 拆分 re-export（保对外契约 · 同一对象）
# ---------------------------------------------------------------------------
# 判分族 → run_eval__scoring.py / run_eval__classify.py；
# 评测引擎 + 埋点快照 → run_eval__engine.py。以下**原位 re-export** ⇒ 全仓
# `from run_eval import answers_match / EvalEngine / ...` 一行都不用改。
# ===========================================================================
from run_eval__scoring import (  # noqa: F401
    _LATEX_SYMBOL_MAP, _FUNC_CMDS, _FRAC_ALIASES, _LAYOUT_CMDS,
    _FRAC_ONLY_RE, _FRAC_LATEX_RE, _DIVISIBLE_RE, _FUNC_PREFIX_RE,
    _NO_SOLUTION_WORDS, _SPLIT_SEP_RE,
    _clean_answer, _norm_candidate, _extract_equals_candidates, _extract_boxed,
    _laTeX_to_py_frac, _try_float_compare, _frac_pair, _try_fraction_compare,
    _try_sympy_reduce_match, _matches_one, _letter_combo_match,
    _option_letter_match, answers_match, _match_gold_process_tail,
    _match_stripped_func_prefix, _match_divisibility, _match_no_solution,
    _normalize_ellipsis, _strip_set_braces, _split_multi, _alpha_equivalent,
    _poly_equal, _item_match, _multi_value_match, _strip_latex_cmds,
)
from run_eval__classify import (  # noqa: F401
    _EMPTY_DELIM, _NUMERIC, _UNRESOLVED_SYM, _STEP_MARK, _MD_HEADER,
    _classify_error,
)
from run_eval__engine import (  # noqa: F401
    DEFAULT_AGENT_OVERRIDES, EvalEngine,
    _llm_call_snapshot, _mcp_call_snapshot, _web_call_snapshot,
)

# ===========================================================================
# CLI
# ===========================================================================

def _add_core_and_ab_args(parser: argparse.ArgumentParser) -> None:
    """注册核心参数与 A/B 能力开关（2026-10-08 从 main 逐字搬出；argparse 语义不变）。"""
    parser.add_argument("--test_file", default="", help="JSONL 测试文件路径")
    parser.add_argument("--bank", default="", help="题库名称（如 新高数、1000题高数、高数a、IMO-AnswerBench、all）")
    parser.add_argument("--list_banks", action="store_true", help="列出所有已注册题库")
    parser.add_argument("--output", default="eval_results.jsonl", help="输出结果文件")
    parser.add_argument("--concurrency", type=int, default=2, help="并发数")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    parser.add_argument("--api_key", default="", help="LLM API Key（或设置 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--base_url", default="", help="LLM Base URL（或设置 OPENAI_BASE_URL 环境变量）")
    parser.add_argument("--model", default="", help="模型名（或设置 LLM_MODEL 环境变量）")
    # ---- A/B 能力开关（None 表示使用本地评测默认值）----
    parser.add_argument("--voting_times", type=int, default=None, help="verifier_voting_times（每个候选验证票数）")
    parser.add_argument("--verifier_deep_final_enabled", type=str, default=None,
                        choices=["true", "false"],
                        help="verifier_deep_final_enabled（带推理的最终复核，默认 false）")
    # ★ 2026-09-16 审计修复：以下两个键此前**只有 getattr 兜底**，
    #   AgentConfig 未声明、白名单未列、CLI 也没有 ⇒ 注释承诺的
    #   "回退/AB 开关"实际不可用（假开关）。现补全三处。
    parser.add_argument("--enable_question_type_hint", type=str, default=None,
                        choices=["true", "false"],
                        help="全题型强制注入题型特化纪律（默认 false；"
                             "仅客观题默认由 objective_tactic_enabled 控制）")
    parser.add_argument("--verify_reserve_seconds", type=float, default=None,
                        help="生成侧预留秒数（_gen_deadline = deadline − 此值）；"
                             "0/缺省 = 按档位默认（deep 540 / 其他 480）")
    parser.add_argument("--enable_web_search", type=str, default=None,
                        choices=["true", "false"],
                        help="启用联网搜索工具（web_search，默认 false；"
                             "经原生 tool_calls 调用，效果见 tool_calls.web_search）")
    parser.add_argument("--verifier_deep_final_min_remaining", type=float, default=None,
                        help="verifier_deep_final_min_remaining（剩余秒数低于此值则跳过复核）")
    # ★ 2026-09-15 审计补漏：以下两键原先**只有配置与白名单、没有 CLI 参数**
    # ⇒ 命令行根本传不进去（与"有 CLI 没白名单"是同一类坑的镜像）。
    parser.add_argument("--verifier_deep_review_temperature", type=float, default=None,
                        help="verifier_deep_review_temperature（复核采样温度，默认 0.0）")
    parser.add_argument("--verifier_deep_review_min_chars", type=int, default=None,
                        help="verifier_deep_review_min_chars（复核最短输出字数，低于此值视为未完成；0=关闭该护栏）")
    # ★ 2026-09-17 补：该键此前**有 AgentConfig 声明、有白名单、有读取点，却唯独没有 CLI**
    #   （同批的 temperature/min_chars 都有）⇒ 属审核列的"半假开关"，
    #   命令行无法调。实测需要它：默认 16384 过大，深复核产出 26k–42k 字符、
    #   VERDICT 出现 0 次且出现复读退化（两样本 chars 完全相同=26749）。
    parser.add_argument("--verifier_deep_review_max_tokens", type=int, default=None,
                        help="verifier_deep_review_max_tokens（复核单次最大 token，默认 16384；"
                             "建议 ≤6144 以缩短退化窗口）")
    # 2026-09-16 投票方差（候选分歧 → 提高票数 + 非零温度）
    parser.add_argument("--verifier_diversify_enabled", type=str, default=None,
                        choices=["true", "false"],
                        help="verifier_diversify_enabled（候选有分歧时提高票数+温度，默认 true）")
    parser.add_argument("--verifier_disagreement_votes", type=int, default=None,
                        help="verifier_disagreement_votes（有分歧时的每候选票数，默认 3）")
    parser.add_argument("--verifier_disagreement_temperature", type=float, default=None,
                        help="verifier_disagreement_temperature（有分歧时的采样温度，默认 0.7）")
    parser.add_argument("--use_scoring", type=str, default=None, choices=["true", "false"], help="use_scoring（验证器多维评分）")
    parser.add_argument("--revise_rounds", type=int, default=None, help="max_revise_rounds（自纠错回环轮数）")
    parser.add_argument("--use_proof", type=str, default=None, choices=["true", "false"], help="use_proof_channel（证明题专用通道）")
    parser.add_argument("--use_blueprint", type=str, default=None, choices=["true", "false"], help="use_blueprint（蓝图分解）")
    parser.add_argument("--blueprint_deps_enabled", type=str, default=None,
                        choices=["true", "false"],
                        help="blueprint_deps_enabled（DAG 依赖边；false=复现'依赖恒空'旧行为做 A/B）")
    parser.add_argument("--blueprint_or_expand_all", type=str, default=None,
                        choices=["true", "false"],
                        help="blueprint_or_expand_all（OR 节点是否展开全部备选分支；"
                             "默认 false=只取 children[0]，true 会放大子目标规模）")


def _add_extended_switch_args(parser: argparse.ArgumentParser) -> None:
    """注册其余扩展开关（自改进/协作/符号/诊断/LeanSearch/开关注册制 9 键）。"""
    # 2026-09-13：为 A/B 对比新增两个开关（此前只能改代码才能切换）。
    # 背景：`3.3_improve`(单Agent自审自改) 与 `3.4_collab`(三Agent协作改进)
    # 语义重叠，需要实测"哪个更高效"才能决定舍去哪一个。
    parser.add_argument("--enable_self_improve", type=str, default=None,
                        choices=["true", "false"],
                        help="enable_self_improve（3.3 Step2 无条件自改进）")
    parser.add_argument("--enable_collaborative_deep", type=str, default=None,
                        choices=["true", "false"],
                        help="enable_collaborative_deep（3.4 deep档三Agent协作）")
    # ★ 2026-10-15（用户决策「把没有用和关掉开关的部分全删了」）：
    #   `--enable_dag_replan` / `--dag_replan_gate` 两个 CLI 参数已随三个默认
    #   关闭的门整块撤除（见 user_agent__config.py 同名说明）。
    # 2026-09-08：L2 子目标数值/代数断言 Lean 验证（lean-lsp-mcp norm_num/ring）
    parser.add_argument("--enable_numeric_lean_verify", type=str, default=None, choices=["true", "false"], help="enable_numeric_lean_verify（子目标数值断言 Lean 验证）")
    parser.add_argument("--lean_numeric_max_per_q", type=int, default=None, help="lean_numeric_max_per_q（每题数值 Lean 验证限额，默认 2）")
    # 2026-09-10 L2：独立符号建模核验关卡（A/B 用）
    # 2026-10-01：`--calc_mandatory` / `--calc_hard_only` / `--subgoal_calc_router` /
    #   `--tool_calc_enabled` / `--answer_selfcheck_enabled` 随 `<calc>` 板块删除。
    parser.add_argument("--symbolic_crosscheck_enabled", type=str, default=None, choices=["true", "false"], help="symbolic_crosscheck_enabled（L2：独立符号建模求真值 → 与答案比对，不符则打回）")
    # 2026-09-12 符号化方程求解通道：模型只交方程（组）+ 目标，数值由本地工具算
    parser.add_argument("--symbolic_solve_enabled", type=str, default=None, choices=["true", "false"], help="symbolic_solve_enabled（数值剥离→模型符号建模→工具求解，模型不参与计算）")
    parser.add_argument("--symbolic_solve_feedback", type=str, default=None, choices=["true", "false"], help="symbolic_solve_feedback（工具值与答案分歧时回传工具结果给模型定稿）")
    parser.add_argument("--symbolic_solve_adopt", type=str, default=None, choices=["true", "false"], help="symbolic_solve_adopt（方案④：工具求解成功后答案直接取工具值，模型不参与计算）")
    # 2026-09-29：`--use_fast_path` 已删除（快车道机制整体移除）
    parser.add_argument("--max_total_calls", type=int, default=None, help="max_total_calls（单题 LLM 调用预算）")
    # ---- 2026-09-13 诊断模式：时间限制放开（默认不传 = 保持比赛口径，行为不变）----
    # 用途：服务端高延迟时（实测单次 LLM 60–180s），比赛口径会让每题被"预算不足"
    # 中途截断（budget_skips 飙升），**看不到完整的失败路径**。诊断跑分时放开，
    # 只为定位"错在哪一步"；⚠ 放宽口径的成绩**不得对外报数**。
    parser.add_argument("--max_time_per_question", type=int, default=None,
                        help="单题壁钟上限秒（诊断用；不传=1200 比赛口径）")
    parser.add_argument("--tier_budget", type=str, default=None,
                        help="统一档位预算 'f,s,d'（2026-09-29 后只取最后一段作 deep 值；"
                             "不传=1150）")
    # 2026-09-29：`--paper_target_time` / `--deep_quota_ratio` / `--paper_total_questions`
    # 三个参数随 PaperPacer 删除（已无任何消费者）。
    # ---- 2026-09-15 赛后无约束评测：补齐此前**没有 CLI 入口**的旋钮 ----
    # 背景：上述 4 个诊断参数只覆盖了一部分约束，`max_total_time_seconds` /
    # 子目标阶段预算 / `max_subgoals` / `improve_min_remaining` / `deep_quota_ratio`
    # 此前只能改代码，导致"解除比赛限制"做不彻底（改了单题上限，全卷仍被 20700 卡）。
    # 全部默认 None = 不传即保持比赛口径，行为与改动前逐字节一致。
    parser.add_argument("--max_total_time_seconds", type=int, default=None,
                        help="Agent 总运行上限秒（诊断用；不传=20700 比赛口径）")
    parser.add_argument("--subgoal_stage_budget_sec", type=float, default=None,
                        help="deep 档子目标阶段预算秒（诊断用；不传=750）")
    parser.add_argument("--subgoal_stage_budget_sec_std", type=float, default=None,
                        help="standard/fast 档子目标阶段预算秒（诊断用；不传=450）")
    parser.add_argument("--max_subgoals", type=int, default=None,
                        help="子目标规划数上限（诊断用；不传=6 比赛口径；<=0 表示不截断）")
    # ---- LeanSearch 引理检索（2026-09-15 重启；A/B 用）----
    parser.add_argument("--use_leansearch", type=str, default=None,
                        choices=["true", "false"],
                        help="use_leansearch（LeanSearch 引理检索总开关，默认关）")
    parser.add_argument("--leansearch_top_k", type=int, default=None,
                        help="leansearch_top_k（每次检索返回条数，默认 5；老师 #46 要求扫 3/5/10）")
    parser.add_argument("--leansearch_max_calls_per_q", type=int, default=None,
                        help="leansearch_max_calls_per_q（单题检索次数上限，默认 2）")
    parser.add_argument("--leansearch_inject_verifier", type=str, default=None,
                        choices=["true", "false"],
                        help="leansearch_inject_verifier（方案 A：把定理原文注入验证器，默认 true）")
    parser.add_argument("--improve_min_remaining", type=float, default=None,
                        help="3.3 改进停手预留秒（诊断用；不传=300；0=关闭该护栏）")
    # 2026-09-29（截图 #7）：无条件自改进 A/B 抓手 —— 遍数（1/2）与缺陷过滤开关。
    parser.add_argument("--self_improve_rounds", type=int, default=None,
                        help="3.3 Step2 自改进遍数（默认 1；设 2 做「一遍 vs 两遍」A/B）")
    parser.add_argument("--self_improve_conditional", type=str, default=None,
                        choices=["true", "false"],
                        help="3.3 是否只改有缺陷候选（默认 false = 真无条件）")
    # 2026-09-29（截图 #3+#4）：领域→定理检索 A/B 抓手（有定理 vs 无定理）。
    parser.add_argument("--enable_theorem_hint", type=str, default=None,
                        choices=["true", "false"],
                        help="1.6 领域→Mathlib 定理检索并注入（默认 true）")
    parser.add_argument("--theorem_hint_max_queries", type=int, default=None,
                        help="1.6 单题最多发几次 leansearch 检索（默认 2）")
    # 2026-09-29：--deep_quota_ratio / --paper_total_questions 随 PaperPacer 删除。
    # ---- 2026-10-07 开关注册制补齐（fix-switch-debt）：9 条此前**漏加白名单**的
    #      A/B 开关补 CLI 入口（原只有 AgentConfig 字段 + 读取点，命令行传不进 ⇒
    #      「回退路径在代码层面不存在」）。全部 default=None ⇒ 不传即保持类默认，
    #      行为与改动前逐字一致。9 键均经 grep 确认有**真实消费点**（非死配置），
    #      故全部面向命令行；无「不该有 CLI」的排除项。
    parser.add_argument("--accept_confidence", type=float, default=None,
                        help="AcceptGate 可接受置信度阈值（默认 0.6；"
                             "orchestrator._update_accept_gate）")
    parser.add_argument("--enable_deterministic", type=str, default=None,
                        choices=["true", "false"],
                        help="确定性硬否决：SymPy 回验 fail 淘汰候选、unknown 放行"
                             "（默认 true；orchestrator）")
    parser.add_argument("--enable_answer_falsifier", type=str, default=None,
                        choices=["true", "false"],
                        help="候选答案证伪器（默认 true；answer_falsifier/orchestrator）")
    parser.add_argument("--audit_gate_skip_non_proof", type=str, default=None,
                        choices=["true", "false"],
                        help="6.5 AuditGate 对非证明题早退、不再产 unknown 噪音"
                             "（默认 true；audit_gate）")
    parser.add_argument("--enable_subgoal_main_path", type=str, default=None,
                        choices=["true", "false"],
                        help="子目标细化作为主路径（默认 true；orchestrator）")
    parser.add_argument("--inject_mathlib_to_solver", type=str, default=None,
                        choices=["true", "false"],
                        help="把 LeanSearch 检索到的 Mathlib 定理注入解题侧引理记忆"
                             "（默认 true；sub_goal_solver）")
    parser.add_argument("--objective_tactic_enabled", type=str, default=None,
                        choices=["true", "false"],
                        help="客观题特化解法纪律（默认 true；solver/sub_goal_solver）")
    parser.add_argument("--symbolic_max_tokens", type=int, default=None,
                        help="独立符号建模复核单次 max_tokens（默认 8192；solver）")
    parser.add_argument("--symbolic_solve_max_tokens", type=int, default=None,
                        help="符号化方程求解建模单次 max_tokens（默认 8192；solver）")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MathPilot 本地评测工具")
    _add_core_and_ab_args(parser)
    _add_extended_switch_args(parser)
    return parser


def _collect_arg_overrides(args) -> Dict[str, Any]:
    """收集 A/B 能力开关（2026-10-08 从 main 逐字搬出；每键 default=None ⇒ 不传即不改行为）。"""
    # 收集 A/B 能力开关
    overrides: Dict[str, Any] = {}
    if args.voting_times is not None:
        overrides["verifier_voting_times"] = args.voting_times
    if args.verifier_deep_final_enabled is not None:
        overrides["verifier_deep_final_enabled"] = (
            args.verifier_deep_final_enabled == "true")
    # ★ 2026-09-16 审计补全：让这两个键的 CLI 真正生效
    #   （此前只有 getattr 兜底，注释承诺的回退/AB 开关是假的）
    if getattr(args, "enable_question_type_hint", None) is not None:
        overrides["enable_question_type_hint"] = (
            args.enable_question_type_hint == "true")
    if getattr(args, "verify_reserve_seconds", None) is not None:
        overrides["verify_reserve_seconds"] = args.verify_reserve_seconds
    if getattr(args, "enable_web_search", None) is not None:
        overrides["enable_web_search"] = (args.enable_web_search == "true")
    if args.verifier_deep_final_min_remaining is not None:
        overrides["verifier_deep_final_min_remaining"] = (
            args.verifier_deep_final_min_remaining)
    if args.verifier_deep_review_temperature is not None:
        overrides["verifier_deep_review_temperature"] = (
            args.verifier_deep_review_temperature)
    if args.verifier_deep_review_min_chars is not None:
        overrides["verifier_deep_review_min_chars"] = (
            args.verifier_deep_review_min_chars)
    if getattr(args, "verifier_deep_review_max_tokens", None) is not None:
        overrides["verifier_deep_review_max_tokens"] = (
            args.verifier_deep_review_max_tokens)
    if args.verifier_diversify_enabled is not None:
        overrides["verifier_diversify_enabled"] = (
            args.verifier_diversify_enabled == "true")
    if args.verifier_disagreement_votes is not None:
        overrides["verifier_disagreement_votes"] = args.verifier_disagreement_votes
    if args.verifier_disagreement_temperature is not None:
        overrides["verifier_disagreement_temperature"] = (
            args.verifier_disagreement_temperature)
    if args.use_scoring is not None:
        overrides["use_scoring"] = args.use_scoring == "true"
    if args.revise_rounds is not None:
        overrides["max_revise_rounds"] = args.revise_rounds
    if args.use_proof is not None:
        overrides["use_proof_channel"] = args.use_proof == "true"
    if args.use_blueprint is not None:
        overrides["use_blueprint"] = args.use_blueprint == "true"
    if args.blueprint_deps_enabled is not None:
        overrides["blueprint_deps_enabled"] = (
            args.blueprint_deps_enabled == "true")
    if args.blueprint_or_expand_all is not None:
        overrides["blueprint_or_expand_all"] = (
            args.blueprint_or_expand_all == "true")
    if args.enable_self_improve is not None:
        overrides["enable_self_improve"] = args.enable_self_improve == "true"
    if args.enable_collaborative_deep is not None:
        overrides["enable_collaborative_deep"] = \
            args.enable_collaborative_deep == "true"
    if args.enable_numeric_lean_verify is not None:
        overrides["enable_numeric_lean_verify"] = \
            args.enable_numeric_lean_verify == "true"
    if args.lean_numeric_max_per_q is not None:
        overrides["lean_numeric_max_per_q"] = args.lean_numeric_max_per_q
    # 2026-09-10 L2 独立符号建模核验关卡
    if args.symbolic_crosscheck_enabled is not None:
        overrides["symbolic_crosscheck_enabled"] = args.symbolic_crosscheck_enabled == "true"
    # 2026-09-12 符号化方程求解通道
    if args.symbolic_solve_enabled is not None:
        overrides["symbolic_solve_enabled"] = args.symbolic_solve_enabled == "true"
    if args.symbolic_solve_feedback is not None:
        overrides["symbolic_solve_feedback"] = args.symbolic_solve_feedback == "true"
    if args.symbolic_solve_adopt is not None:
        overrides["symbolic_solve_adopt"] = args.symbolic_solve_adopt == "true"
    if args.max_total_calls is not None:
        overrides["max_total_calls"] = args.max_total_calls
    # 2026-09-13 诊断模式：时间限制放开
    if args.max_time_per_question is not None:
        overrides["max_time_per_question"] = args.max_time_per_question
    if args.tier_budget:
        # 2026-09-29：统一档位 —— CLI 仍兼容 `f,s,d` 三段写法（取 deep 段，
        # 保持 `run_research.py` 的 "86400,86400,86400" 等既有调用可用），
        # 也支持单段 "1150" 直写。
        _tb = [float(x) for x in args.tier_budget.split(",")]
        if len(_tb) == 3:
            overrides["tier_budget"] = {"deep": _tb[2]}
        elif len(_tb) == 1:
            overrides["tier_budget"] = {"deep": _tb[0]}
    # 2026-09-15 赛后无约束评测：补齐 7 个旋钮（默认 None ⇒ 不改变比赛口径）
    if args.max_total_time_seconds is not None:
        overrides["max_total_time_seconds"] = args.max_total_time_seconds
    if args.subgoal_stage_budget_sec is not None:
        overrides["subgoal_stage_budget_sec"] = args.subgoal_stage_budget_sec
    if args.subgoal_stage_budget_sec_std is not None:
        overrides["subgoal_stage_budget_sec_std"] = args.subgoal_stage_budget_sec_std
    if args.max_subgoals is not None:
        overrides["max_subgoals"] = args.max_subgoals
    # ---- LeanSearch 引理检索（2026-09-15 重启）----
    if args.use_leansearch is not None:
        overrides["use_leansearch"] = args.use_leansearch == "true"
    if args.leansearch_top_k is not None:
        overrides["leansearch_top_k"] = args.leansearch_top_k
    if args.leansearch_max_calls_per_q is not None:
        overrides["leansearch_max_calls_per_q"] = args.leansearch_max_calls_per_q
    if args.leansearch_inject_verifier is not None:
        overrides["leansearch_inject_verifier"] = (
            args.leansearch_inject_verifier == "true")
    if args.improve_min_remaining is not None:
        overrides["improve_min_remaining"] = args.improve_min_remaining
    if args.self_improve_rounds is not None:
        overrides["self_improve_rounds"] = args.self_improve_rounds
    if args.self_improve_conditional is not None:
        overrides["self_improve_conditional"] = (
            args.self_improve_conditional == "true")
    if args.enable_theorem_hint is not None:
        overrides["enable_theorem_hint"] = args.enable_theorem_hint == "true"
    if args.theorem_hint_max_queries is not None:
        overrides["theorem_hint_max_queries"] = args.theorem_hint_max_queries
    # 2026-09-29：--deep_quota_ratio / --paper_total_questions 随 PaperPacer 删除。
    # ---- 2026-10-07 开关注册制补齐（fix-switch-debt）：9 条 A/B 开关的 CLI → 覆盖 ----
    if args.accept_confidence is not None:
        overrides["accept_confidence"] = args.accept_confidence
    if args.enable_deterministic is not None:
        overrides["enable_deterministic"] = args.enable_deterministic == "true"
    if args.enable_answer_falsifier is not None:
        overrides["enable_answer_falsifier"] = args.enable_answer_falsifier == "true"
    if args.audit_gate_skip_non_proof is not None:
        overrides["audit_gate_skip_non_proof"] = (
            args.audit_gate_skip_non_proof == "true")
    if args.enable_subgoal_main_path is not None:
        overrides["enable_subgoal_main_path"] = (
            args.enable_subgoal_main_path == "true")
    if args.inject_mathlib_to_solver is not None:
        overrides["inject_mathlib_to_solver"] = (
            args.inject_mathlib_to_solver == "true")
    if args.objective_tactic_enabled is not None:
        overrides["objective_tactic_enabled"] = (
            args.objective_tactic_enabled == "true")
    if args.symbolic_max_tokens is not None:
        overrides["symbolic_max_tokens"] = args.symbolic_max_tokens
    if args.symbolic_solve_max_tokens is not None:
        overrides["symbolic_solve_max_tokens"] = args.symbolic_solve_max_tokens
    return overrides


def _print_eval_reports(all_summaries: List) -> None:
    """打印汇总报告（2026-10-08 从 main 逐字搬出；纯输出，无副作用）。"""
    # 打印汇总报告
    for test_file, summary in all_summaries:
        print("\n" + "=" * 60)
        print(f"MathPilot 评测报告 - {os.path.basename(test_file)}")
        print("=" * 60)
        print(f"题目总数:   {summary['total']}")
        print(f"可判分题:   {summary['scored']}")
        print(f"正确数:     {summary['correct']}")
        print(f"准确率:     {summary['accuracy']:.2%}")
        print(f"平均耗时:   {summary['avg_elapsed_sec']} 秒")
        print("-" * 60)
        print(f"{'领域':<25} {'总数':<6} {'正确':<6} {'准确率':<8}")
        print("-" * 60)
        for domain, stats in summary.get("per_domain", {}).items():
            print(f"{domain:<25} {stats['total']:<6} {stats['correct']:<6} {stats['accuracy']:<8.2%}")
        print("=" * 60)

    # 多题库时打印总汇总
    if len(all_summaries) > 1:
        total_q = sum(s['total'] for _, s in all_summaries)
        total_correct = sum(s['correct'] for _, s in all_summaries)
        total_scored = sum(s['scored'] for _, s in all_summaries)
        print("\n" + "=" * 60)
        print("全部题库汇总")
        print("=" * 60)
        print(f"题库数:     {len(all_summaries)}")
        print(f"题目总数:   {total_q}")
        print(f"可判分题:   {total_scored}")
        print(f"正确数:     {total_correct}")
        print(f"总准确率:   {total_correct/total_scored:.2%}" if total_scored else "总准确率:   N/A")
        print("=" * 60)


def _flush_leansearch_stats() -> None:
    """LeanSearch 检索埋点落盘（2026-10-08 从 main 逐字搬出；异常不阻断评测）。"""
    # ---- LeanSearch 检索埋点落盘（2026-09-15，老师 #44）----
    # 只在**真正发生过检索**时写文件，避免生成空文件误导后续聚合。
    # 输出：每行一个事件（call / adopted），末尾一行 summary，便于离线按题统计漏斗。
    try:
        from tools.lean_local.lean_search import get_stats as _ls_stats
        _ls = _ls_stats()
        _ls_sum = _ls.summary()
        if _ls_sum.get("calls"):
            _ls_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "results", "leansearch_calls.jsonl")
            if _ls.flush(_ls_path):
                print("LeanSearch 埋点已落盘:", _ls_path)
                print("  汇总:", json.dumps(_ls_sum, ensure_ascii=False))
            else:
                print("LeanSearch 埋点落盘失败（不影响评测结果）")
    except Exception as _e:  # noqa: BLE001
        print("LeanSearch 埋点落盘跳过:", str(_e)[:160])


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_banks:
        list_banks()
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    # 解析 test_file：--bank 优先
    test_files: List[str] = []
    if args.bank:
        resolved = resolve_bank(args.bank)
        if resolved is None:
            print(f"错误: 未知题库 '{args.bank}'。使用 --list_banks 查看可用题库。")
            sys.exit(1)
        if isinstance(resolved, list):
            test_files = resolved
        else:
            test_files = [resolved]
    elif args.test_file:
        test_files = [args.test_file]
    else:
        print("错误: 请指定 --test_file 或 --bank。使用 --list_banks 查看可用题库。")
        sys.exit(1)

    overrides = _collect_arg_overrides(args)

    engine = EvalEngine(
        concurrency=args.concurrency, resume=args.resume,
        api_key=args.api_key, base_url=args.base_url, model=args.model,
        verbose=args.verbose, agent_overrides=overrides,
    )

    # 支持多题库评测
    all_summaries = []
    base_output = args.output
    for i, test_file in enumerate(test_files):
        # 多题库时自动命名输出文件
        if len(test_files) > 1:
            bank_name = os.path.splitext(os.path.basename(test_file))[0]
            stem, ext = os.path.splitext(base_output)
            output_file = f"{stem}_{bank_name}{ext}"
        else:
            output_file = base_output

        print(f"\n{'='*60}")
        print(f"题库 [{i+1}/{len(test_files)}]: {os.path.basename(test_file)}")
        print(f"输出文件: {output_file}")
        print(f"{'='*60}")

        summary = engine.run(test_file, output_file)
        all_summaries.append((test_file, summary))

    _print_eval_reports(all_summaries)
    _flush_leansearch_stats()


if __name__ == "__main__":
    main()
