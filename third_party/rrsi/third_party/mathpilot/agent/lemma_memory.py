"""
LemmaMemory —— 引理记忆模块（老师 #30）
=========================================

在 LEAP 三阶段流程中累积/复用已证引理：

- **题目内记忆**：当前题目求解过程中已证明的引理（叶子子目标成功后即入册），
  后续子目标 / 兄弟分支可直接引用，避免重复证明
- **跨题记忆**：跨题目/跨会话持久化（可选，写入磁盘 JSON），
  支撑"子智能体间传递已知条件"与长期复用

与现有黑板（TaskContext.lemma_repo）的关系：
- TaskContext.lemma_repo 是单题内存列表（list[str]）
- LemmaMemory 提供结构化存取（name + statement + proof + 状态）、
  查重、序列化（含跨题持久化），是 #30 的正式落地

对外接口：
- LemmaMemory()：内存版（题目内）
- LemmaMemory(storage_path=...)：带磁盘持久化的跨题版
- add(name, statement, proof="", source="") -> bool（查重后添加）
- lookup(keyword) -> list[dict]（按名称/陈述关键词检索）
- get_all() / to_json() / load_from_json()
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger("MathPilot")


def _norm_key(name: str) -> str:
    """规范化引理名（去空白/小写），用于查重。"""
    return re.sub(r"\s+", "", str(name or "")).lower()


class LemmaMemory:
    """引理记忆：累积 + 检索 + 序列化。

    线程安全说明：单线程智能体循环内使用为主；若并发子智能体写入，
    调用方负责加锁（本模块保持无锁简单实现）。
    """

    def __init__(self, storage_path: str = ""):
        self._lemmas: dict = {}          # key -> {name, statement, proof, source, ts}
        self.storage_path = storage_path
        if storage_path and os.path.exists(storage_path):
            self.load_from_json(storage_path)

    # ---------------- 写入 ----------------
    def add(self, name: str, statement: str, proof: str = "",
            source: str = "") -> bool:
        """添加引理。同名（规范化）已存在则不重复添加，返回 False。"""
        name = (name or "").strip()
        if not name or not statement:
            return False
        key = _norm_key(name)
        if key in self._lemmas:
            return False
        self._lemmas[key] = {
            "name": name,
            "statement": statement.strip(),
            "proof": proof.strip(),
            "source": source,
            "ts": time.time(),
        }
        # ★ 2026-10-15（审核 Q1：修正无用功）：`save` 此前全仓无调用者 ⇒
        #   「跨题持久化」名存实亡。现改为**添加即惰性落盘**（配置了
        #   storage_path 才写；失败只警告不阻断）——自包含接线。
        if getattr(self, "storage_path", ""):
            try:
                self.save()
            except Exception:  # noqa: BLE001
                pass
        return True


    # ---------------- 检索 ----------------
    def lookup(self, keyword: str, limit: int = 5) -> list:
        """按关键字在名称/陈述中检索引理（不区分大小写）。"""
        kw = (keyword or "").strip().lower()
        if not kw:
            return list(self._lemmas.values())[:limit]
        hits = []
        for lemma in self._lemmas.values():
            if kw in lemma["name"].lower() or kw in lemma["statement"].lower():
                hits.append(lemma)
                if len(hits) >= limit:
                    break
        return hits

    def get(self, name: str):
        return self._lemmas.get(_norm_key(name))

    def get_all(self) -> list:
        return list(self._lemmas.values())

    def __len__(self) -> int:
        return len(self._lemmas)

    # ---------------- 序列化 ----------------
    def to_json(self) -> str:
        return json.dumps(list(self._lemmas.values()), ensure_ascii=False, indent=1)

    def save(self, path: str = "") -> bool:
        """保存到磁盘（跨题持久化）。"""
        p = path or self.storage_path
        if not p:
            return False
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.to_json())
            return True
        except OSError as e:
            logger.warning("LemmaMemory 保存失败: %s", e)
            return False

    def load_from_json(self, path: str) -> int:
        """从磁盘 JSON 载入；返回加载条数。"""
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("LemmaMemory 载入失败: %s", e)
            return 0
        n = 0
        for it in items or []:
            if isinstance(it, dict) and self.add(
                    it.get("name", ""), it.get("statement", ""),
                    it.get("proof", ""), it.get("source", "")):
                n += 1
        return n

    # ---------------- 与黑板互通 ----------------
    def import_from_ctx(self, ctx) -> int:
        """把 TaskContext.lemma_repo（旧式 list[str]）并入记忆，返回新增数。"""
        added = 0
        repo = getattr(ctx, "lemma_repo", None) or []
        for i, text in enumerate(repo):
            if text and self.add(f"ctx_lemma_{i}", str(text),
                                 source="ctx.lemma_repo"):
                added += 1
        return added

    def export_to_ctx(self, ctx) -> None:
        """把记忆里所有引理写回 TaskContext.lemma_repo（供下游提示词注入）。"""
        ctx.lemma_repo = [f"{l['name']}: {l['statement']}" for l in self.get_all()]

    def format_for_prompt(self, limit: int = 10) -> str:
        """生成供提示词注入的引理清单文本。"""
        items = self.get_all()
        if not items:
            return ""
        lines = []
        for l in items[-limit:]:
            proof_tag = "[已证]" if l.get("proof") else "[声明]"
            lines.append("  - %s %s: %s" % (proof_tag, l["name"], l["statement"]))
        return "\n".join(lines)
