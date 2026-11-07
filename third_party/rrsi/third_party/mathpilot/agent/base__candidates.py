# -*- coding: utf-8 -*-
"""候选入池门槛开关 + 候选 id/腾位助手（跨模块单一来源）—— 从 agent/base.py 逐字搬出。

对外契约不变：``from agent.base import _pool_gate_v2_on, next_candidate_id,
pick_best_candidates`` 仍可用（re-export）。
"""
from __future__ import annotations

from agent.switch_registry import get_bool as _sw_bool
# ============================================================
# 候选入池门槛 v2 开关（跨模块**单一来源**）
# ============================================================
# 2026-10-07（架构解耦 阶段D 去重）：原 `agent/solver.py` /
#   `agent/sub_goal_solver.py` / `agent/collaborative_solver.py` 各写了一份
#   **函数体逐字相同**的读取器（仅 docstring/注释有别）；合并到 base —— 三者
#   本就从 `base` 导入，零新增 import 边、消除口径分叉风险。
def _pool_gate_v2_on() -> bool:
    """候选**入池门槛 v2** 开关（`pool_gate_v2`，默认 True）。

    ★ 2026-10-07 新增（问题 2.4 第二轮修复）。True = 修好后行为：
      · 判据换成 `utils.extract.looks_like_pool_answer`（补 4 条规划/设定句漏网、
        修 087/088 多字母选项误杀、修 101/111 判断题误杀）；
      · **穷举全部写入 `ctx.candidates` 的入口**（不再只覆盖 `_generate_initial`
        一处）。
    =0 → 完全回退上一轮口径（仅 `_generate_initial`、`_generate_revise` 主路径过
      旧判据 `_looks_like_answer`）做 A/B。

    ── 全仓"候选写入 `ctx.candidates`"入口清单（共 11 处，逐一核对）──────────
    【新增门槛（本次）】
      ① solver.py  `_generate_initial` 证明通道 append        （原缺 → 已补）
      ② solver.py  `_generate_initial` 生成失败抢救 append    （原缺 → 已补）
      ③ solver.py  `_improve_candidates_once` **改写 cand.answer**
                    （自改进覆盖，原缺 → 已补；不合格则保留原答案）
      ④ solver.py  `complete_truncated_candidates` 替换为续写候选
                    （原缺 → 已补；不合格则保留原候选）
      ⑤ sub_goal_solver.py `run()` 子目标候选 append          （原缺 → 已补）
      ⑥ collaborative_solver.py `run()` 协作候选 append       （原缺 → 已补）
    【本已覆盖（上一轮已有，判据升级为 v2）】
      ⑦ solver.py  `_generate_initial` 主成功 append
      ⑧ solver.py  `_generate_revise`   主成功 append
      （`add_candidates` 委托 `_generate_initial`，同一入口，无需单列）
    【刻意不过门槛（有理由，非遗漏）】
      ⑨ solver.py  `_generate_revise` 失败占位 `Candidate(answer="", reasoning=
                    "[重解失败]…")` —— 显式失败标记，非"假答案"
      ⑩ solver.py  `_improve_candidates_once` 保留原版 `_orig` —— 已入池候选的
                    深拷贝，其 answer 入池时已过门槛
      ⑪ orchestrator.py / formatter.py 的 `ctx.candidates = <子集>`
                    （lean_kept / audit_kept / _clean / _keep / 腾位）—— **过滤/
                    裁剪既有候选**，不引入新答案，天然不含"新非答案"
      （`collaborative_solver.py` 因不在其他施工员占用范围内，一并纳入本次修复。）

    单一来源说明：本函数被 solver / sub_goal_solver / collaborative_solver 三处
    引用；任何口径调整只改这里，不允许再在调用方另起一份。
    """
    try:
        return bool(_sw_bool("pool_gate_v2"))
    except Exception:  # noqa: BLE001  注册表不可用时取默认（=开）
        return True


def next_candidate_id(candidates) -> int:
    """返回**不会与既有候选冲突**的下一个候选 id（2026-09-17 新增）。

    为什么需要：全仓多处用 `len(candidates)` 生成 Candidate.id，而**腾位/删除会让
    len 变小** ⇒ 新 id 与旧候选 id 重叠（实测 official112-013 出现**两个 id=3**、
    025 出现 id=7）。下游凡按 id 匹配的地方都会错位 ——
    `orchestrator` 的 `_gate_tried`、`formatter` 的簇内候选匹配、
    `verdicts[i].id` ↔ `candidates[i].id` 的对齐关系。

    统一收敛到本函数，避免各写入点口径再次分叉（本仓库已有"口径不一"的历史教训）。
    """
    _mx = -1
    for _c in (candidates or []):
        try:
            _mx = max(_mx, int(getattr(_c, "id", -1)))
        except (TypeError, ValueError):
            continue
    return _mx + 1


def pick_best_candidates(candidates, verdicts, k: int = 3) -> list:
    """按 **(正确票数, 推理长度)** 降序取前 k 个候选（2026-09-17 新增）。

    用途 = 候选池腾位。原实现是 `list(ctx.candidates)[-3:]`，即按 id 尾部保留
    "最新"的 3 个，**与正确性无关** —— 若新追加的候选恰好最差，就会留下最差 3 个、
    丢掉初始解（通常较好）。抽出此函数使口径单一、且可被单元测试直接覆盖。
    """
    _votes = {}
    for _v in (verdicts or []):
        try:
            _votes[int(getattr(_v, "id", -1))] = int(
                getattr(_v, "correct_votes", 0) or 0)
        except (TypeError, ValueError):
            continue
    _pool = list(candidates or [])
    _pool.sort(key=lambda _c: (
        -_votes.get(int(getattr(_c, "id", -1)), 0),
        -len(getattr(_c, "reasoning", "") or "")))
    return _pool[:max(0, int(k))]
