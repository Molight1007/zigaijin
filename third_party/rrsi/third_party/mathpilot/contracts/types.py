"""纯数据结构（零依赖）。

★ 2026-10-07 架构解耦（阶段 B1）：这些类是**纯数据**（`dataclass` + 标准库），
原先定义在 `agent/base.py`，被 `tools/lean_local/{lean_bridge,lean_gate}.py`
**反向依赖** ⇒ 构成 `agent ↔ tools` 包级环的一环。
现下沉到本层（零依赖），由 `agent/base.py` re-export 保持对外契约不变：
    `from agent.base import BugReport, Finding, Budget`  仍然可用（同一对象）。

⚠ 只搬**纯数据**。`Candidate.__post_init__` 依赖 `utils.extract`（非纯），
   `TaskContext` 是共享黑板（有惰性求值/时间方法），`BaseAgent` 是业务基类
   —— 均**不下沉**（见 `contracts/protocols.py` 与交付说明）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


@dataclass
class Budget:
    """LLM 调用**计数器**（线程安全）。

    2026-09-03 老师定调"比赛无调用次数上限"后，预算不再阻断任何调用，
    本类只保留记账用途（事后分析调用量）。原 ``can_spend`` / ``refund`` /
    ``remaining`` 三个方法在全项目已无调用点（审核确认），一并删除，
    避免后人误以为预算仍会拦截。
    """
    max_calls: int
    used_calls: int = 0
    _lock: RLock = field(default_factory=RLock, repr=False)

    def spend(self, n: int = 1) -> None:
        with self._lock:
            self.used_calls += n

    def set_max_calls(self, new_max: int) -> None:
        """动态调整总调用预算（deep 档位需要更多预算时调用）。"""
        with self._lock:
            self.max_calls = max(self.used_calls, new_max)


@dataclass
class Finding:
    """单个步骤级缺陷（P3）"""
    location: str = ""     # 出错位置（步骤编号/行号/引用）
    kind: str = "Gap"      # Critical（致命）| Gap（缺口/瑕疵）
    severity: int = 0       # 严重度 1-5（5 最严重）
    desc: str = ""         # 缺陷描述


@dataclass
class BugReport:
    """步骤级验证报告（P3 一等公民）

    契约：
      - ``findings``：list[Finding]，含 location / kind(Critical|Gap) / severity / desc；
      - ``verdict``：'proof_valid' | 'proof_invalid' | 'unknown'；
      - ``is_valid()`` / ``has_critical()``；
      - ``to_dict()`` / ``to_json()`` / ``from_dict()``：JSON 可序列化（P6 要求）。
    """
    findings: list = field(default_factory=list)  # list[Finding]
    verdict: str = "unknown"
    # 改造2：可修复性判定（yes/no/partial）+ 修正建议（可选字段，向后兼容）
    repairable: str = ""       # 'yes' | 'no' | 'partial'（空=未判定）
    suggestion: str = ""       # 修正建议文本（空=无）

    def is_valid(self) -> bool:
        return self.verdict == "proof_valid"

    def has_critical(self) -> bool:
        return any(f.kind == "Critical" for f in self.findings)

    def to_dict(self) -> dict:
        d = {
            "findings": [
                {"location": f.location, "kind": f.kind,
                 "severity": f.severity, "desc": f.desc}
                for f in self.findings
            ],
            "verdict": self.verdict,
        }
        # 可选字段仅在非空时序列化，保持旧 JSON 契约兼容
        if self.repairable:
            d["repairable"] = self.repairable
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "BugReport":
        d = d or {}
        findings = [
            Finding(location=f.get("location", ""), kind=f.get("kind", "Gap"),
                    severity=int(f.get("severity", 0)), desc=f.get("desc", ""))
            for f in (d.get("findings", []) or [])
        ]
        return cls(findings=findings, verdict=d.get("verdict", "unknown"),
                   repairable=d.get("repairable", "") or "",
                   suggestion=d.get("suggestion", "") or "")
