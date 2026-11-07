# -*- coding: utf-8 -*-
"""中间结果存储层 —— 唯一落盘出口（2026-10-02，李平老师架构建议 #5）。

背景（老师原话）
================
「**如果有中间结果，应进行存储。现在的感觉代码文件间耦合过紧。以代码围绕存储开展处理**」

现状：题意理解 / 蓝图(DAG) / 子目标列表与每步推导 / 候选 / verdict / Lean 结果
**全部只在内存 `TaskContext` 上**，模块间靠 `ctx.xxx` 互相读写 ⇒ 这是"耦合过紧"的根因；
`Orchestrator.run()` 结束时只导出 `candidates_out`/`verdicts_out`，中间产物全程无落盘。

本模块
======
把中间产物**多存一份**到磁盘，供离线复现/归因（**不改任何现有行为**：不删分支、
不改判定、不动返回值）。落点：

    results/<run_id>/<qid>/
        01_understanding.json
        02_blueprint.json
        03_subgoals.json
        04_subgoal_results.jsonl      # 每个子目标一行，追加写
        05_candidates.json
        06_verdicts.json
        07_lean.json
        08_final.json

设计纪律
========
· **append-only**：`put()` 同名**不覆盖**（已存在则写 `<stage>.<n>.json`）；
  `append()` 逐行追加。
· **永不抛异常**：落盘失败只记一条 `ctx.trace`，绝不阻断主链（研究期产物不该拖垮求解）。
· **文件名不含中文**：run_id / qid / stage 一律经 `_safe_name` 净化。
· 写盘统一 `ensure_ascii=False` + `newline="\\n"`。
· 总开关 `artifact_store_enabled`（默认 True）走注册制（env `ARTIFACT_STORE_ENABLED`）。

读取优先级（沿用 `agent/switch_registry.py` 口径）
================================================
    环境变量  >  AgentConfig 覆盖（CLI / kwargs）  >  默认值
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from agent.switch_registry import bind_field as _bind_field
from agent.switch_registry import get_bool as _get_bool

logger = logging.getLogger("MathPilot")

DEFAULT_BASE_DIR = "results"
SWITCH_FIELD = "artifact_store_enabled"

# 净化用：只允许 ASCII 字母数字与 . _ -（⇒ 天然不含中文、无路径分隔符）
_UNSAFE_RE = re.compile(r"[^0-9A-Za-z._-]")


# ============================================================ 开关
def enabled(config=None) -> bool:
    """存储层总开关（注册制读取，禁止裸 os.environ）。默认 **True**。"""
    try:
        if config is not None:
            _bind_field(SWITCH_FIELD, getattr(config, SWITCH_FIELD, None))
        return bool(_get_bool(SWITCH_FIELD))
    except Exception:  # noqa: BLE001
        return True


# ============================================================ 工具
def _safe_name(s: Any) -> str:
    """把任意 run_id/qid/stage 净化成安全、非空、**不含中文**的文件名片段。"""
    t = str(s if s is not None else "").strip()
    t = _UNSAFE_RE.sub("_", t)
    t = t.strip("._-")
    return (t or "x")[:80]


_RUN_ID_CACHE: Optional[str] = None


def _default_run_id() -> str:
    """进程内**稳定**的默认 run id（同一进程所有题共用一个目录）。

    ★ 关键：若每题各取一次 `strftime`，则一题一个目录、无法按"轮次"归集；
    故此处缓存首次取值，保证 `results/<run_id>/` 是「一轮评测 = 一个目录」。
    """
    global _RUN_ID_CACHE
    if _RUN_ID_CACHE is None:
        _RUN_ID_CACHE = time.strftime("%Y%m%d-%H%M%S")
    return _RUN_ID_CACHE


_RUN_ID: Optional[str] = None


def set_run_id(run_id: Any) -> None:
    """由评测入口（`EvalEngine`）显式指定本轮 run id（空 ⇒ 回退进程默认）。

    这是"run_id 接线"的落点：评测轮次 id 由入口决定，而非散落各处各自取时间。
    """
    global _RUN_ID
    _RUN_ID = _safe_name(run_id) if run_id else None


def current_run_id() -> str:
    """当前生效的 run id（显式设定优先，否则进程默认）。"""
    return _RUN_ID or _default_run_id()


def _qid_from_problem(problem: str) -> str:
    """题面 → 稳定 qid（同一道题跨进程/跨轮次必须落到同一目录）。

    ★ 2026-10-04 修（D29）：原用内置 `hash()`，受 `PYTHONHASHSEED` 随机化影响
    ⇒ **跨进程不稳定**（实测 seed=1/2/3 得到 `q60992660`/`q31868867`/`q14196416`
    三个不同目录名），破坏了本模块「按题归集中间结果、可离线复现」的核心目的。
    改用 md5 摘要（确定性）。
    """
    try:
        import hashlib
        return "q" + hashlib.md5(str(problem).encode("utf-8")).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        return "q0"


# ============================================================ 存储
class ArtifactStore:
    """一题一目录的中间结果存储（append-only，永不抛异常）。"""

    def __init__(self, run_id: Any = None, qid: Any = None,
                 base_dir: Any = None) -> None:
        self.run_id = _safe_name(run_id if run_id else current_run_id())
        self.qid = _safe_name(qid)
        self.base_dir = str(base_dir) if base_dir else DEFAULT_BASE_DIR
        self._dir = os.path.join(self.base_dir, self.run_id, self.qid)
        self.errors: List[Dict[str, str]] = []

    # ---- 路径 ----
    def dir(self) -> str:
        """该题目录（绝对/相对路径，取决于 base_dir）。"""
        return self._dir

    def _json_path(self, stage: str, n: int = 1) -> str:
        name = _safe_name(stage)
        if n <= 1:
            return os.path.join(self._dir, name + ".json")
        return os.path.join(self._dir, "%s.%d.json" % (name, n))

    def _jsonl_path(self, stage: str) -> str:
        return os.path.join(self._dir, _safe_name(stage) + ".jsonl")

    def _ensure_dir(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    def _fail(self, stage: str, exc: Exception) -> None:
        self.errors.append({"stage": str(stage),
                            "error": "%s: %s" % (type(exc).__name__, exc)})
        logger.warning("[ArtifactStore] %s 落盘失败（不阻断）: %s", stage, exc)

    # ---- 写 ----
    def put(self, stage: str, obj: Any) -> Optional[str]:
        """落盘一步产物。**同名不覆盖**：已存在则改写到 `<stage>.<n>.json`。

        返回写入路径；失败返回 None（并记入 self.errors）。
        """
        try:
            self._ensure_dir()
            n = 1
            target = self._json_path(stage, n)
            while os.path.exists(target):
                n += 1
                target = self._json_path(stage, n)
            with open(target, "w", encoding="utf-8", newline="\n") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
                f.write("\n")
            return target
        except Exception as exc:  # noqa: BLE001
            self._fail(stage, exc)
            return None

    def append(self, stage: str, row: Dict[str, Any]) -> Optional[str]:
        """逐行追加到 `<stage>.jsonl`（永不覆盖）。返回路径；失败返回 None。"""
        try:
            self._ensure_dir()
            target = self._jsonl_path(stage)
            with open(target, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            return target
        except Exception as exc:  # noqa: BLE001
            self._fail(stage, exc)
            return None

    # ---- 读（调试 / 复现）----
    def get(self, stage: str, latest: bool = True):
        """读回某阶段产物。

        · jsonl 阶段 → 返回 list[dict]（按行序）
        · json 阶段  → latest=True 取最后一版，False 取第一版
        · 不存在 / 读取失败 → None
        """
        try:
            name = _safe_name(stage)
            lp = self._jsonl_path(stage)
            if os.path.exists(lp):
                rows: List[Any] = []
                with open(lp, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except Exception:  # noqa: BLE001  坏行跳过，不影响其余
                            continue
                return rows
            versions = []
            n = 1
            while True:
                p = self._json_path(name, n)
                if not os.path.exists(p):
                    break
                versions.append(p)
                n += 1
            if not versions:
                return None
            pick = versions[-1] if latest else versions[0]
            with open(pick, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None


# ============================================================ ctx 便捷层
def store_for(ctx) -> Optional[ArtifactStore]:
    """取出挂在 ctx 上的存储实例（未挂载 / 关闭时返回 None）。"""
    s = getattr(ctx, "_artifact_store", None) if ctx is not None else None
    return s if isinstance(s, ArtifactStore) else None


def _trace(ctx, stage: str, msg: str) -> None:
    """落盘失败写一条 trace（走 ctx.trace，与 BaseAgent.record 同结构）。"""
    try:
        if ctx is not None and hasattr(ctx, "trace"):
            ctx.trace.append({"agent": "ArtifactStore", "step": "artifact_error",
                              "content": "%s: %s" % (stage, str(msg)[:200])})
    except Exception:  # noqa: BLE001
        pass


def attach(ctx, config=None, run_id=None, qid=None,
           base_dir=None, problem=None) -> Optional[ArtifactStore]:
    """在 `run()` 起点把存储挂到 `ctx._artifact_store`（受开关控制）。

    永不抛异常：任何失败都返回 None，主链继续。
    """
    try:
        if not enabled(config):
            return None
        if qid is None:
            qid = _qid_from_problem(problem if problem is not None
                                    else getattr(ctx, "problem", ""))
        s = ArtifactStore(run_id=run_id, qid=qid, base_dir=base_dir)
        try:
            ctx._artifact_store = s
        except Exception:  # noqa: BLE001
            return None
        return s
    except Exception as exc:  # noqa: BLE001
        _trace(ctx, "attach", str(exc))
        return None


def put_ctx(ctx, stage: str, obj: Any) -> bool:
    """落盘一步（无存储时静默 no-op）。返回是否写入成功。"""
    s = store_for(ctx)
    if s is None:
        return False
    ok = s.put(stage, obj) is not None
    if not ok and s.errors:
        _trace(ctx, stage, s.errors[-1].get("error", ""))
    return ok


def append_ctx(ctx, stage: str, row: Dict[str, Any]) -> bool:
    """逐行追加一步（无存储时静默 no-op）。返回是否写入成功。"""
    s = store_for(ctx)
    if s is None:
        return False
    ok = s.append(stage, row) is not None
    if not ok and s.errors:
        _trace(ctx, stage, s.errors[-1].get("error", ""))
    return ok
