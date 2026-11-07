# -*- coding: utf-8 -*-
"""子目标求解器的 contract 职责簇
（2026-10-08 批3：从 agent/sub_goal_solver.py 逐字搬出 16 个方法）。

搬出后为模块级函数（实例方法保留 `self` 首参，方法体逐字不变）；
跨模块引用原文件模块级符号者（含 `SubGoalSolverAgent` 与其常量/函数）
在**函数体内惰性 import**（避免循环导入）—— 下表已显式列出。
原文件类内保留 1 行跳板（`_x = _sgs_contract._x`）⇒ `self._x(...)` /
`SubGoalSolverAgent._x(...)` 契约逐字不变（同 verifier__review.py 批6）。
"""
from __future__ import annotations

import logging
import re
import time

from .base import TaskContext

logger = logging.getLogger("MathPilot")


def _parse_step_contract(cls, section: str) -> dict:
    """解析【本步结果】段内的结论契约。返回结构化字典（永不抛异常）。

        返回字段：
          ok          是否给出合法 `本步结论:` 行（契约是否达成）
          kind        "value" | "equation" | "verdict" | ""
          expr        value→表达式；equation→"L = R"；verdict→真/假/不确定
          lhs / rhs   equation 形态的左右两侧（客观判定输入）
          verdict     verdict 形态的判定词
          support     `本步依据:` 行（空 = 模型没写）
          derivation  头两行之后的验算/论证正文
          has_derivation 验算篇幅 >= `_MIN_DERIVATION_CHARS`
          reason      未达成契约时的**原因**（供 B 留痕，不返回空串）
        """
    out = {"ok": False, "kind": "", "expr": "", "lhs": "", "rhs": "",
           "verdict": "", "support": "", "derivation": "",
           "has_derivation": False, "reason": ""}
    text = str(section or "")
    if not text.strip():
        out["reason"] = "本步结果为空"
        return out
    m = cls._CONCL_LINE_RE.search(text)
    if not m:
        out["reason"] = "未给出结构化结论行（本步结论: 值/等式/判定 = …）"
        # 未达成契约也要把验算正文抽出来（供 C 判"是否有推导"）
        out["derivation"] = cls._tail_after_conclusion(text)
        out["has_derivation"] = (
            len(out["derivation"].strip()) >= cls._MIN_DERIVATION_CHARS)
        return out
    kind_cn, body = m.group(1), m.group(2).strip()
    ms = cls._SUPPORT_LINE_RE.search(text)
    out["support"] = ms.group(1).strip() if ms else ""
    if kind_cn == "值":
        out["kind"], out["expr"] = "value", body
    elif kind_cn == "等式":
        out["kind"] = "equation"
        out["expr"] = body
        parts = body.split("=")
        if len(parts) >= 2:
            out["lhs"], out["rhs"] = parts[0].strip(), "=".join(parts[1:]).strip()
    elif kind_cn == "判定":
        out["kind"] = "verdict"
        _v = body.strip()
        for token in ("不确定", "真", "假"):
            if token in _v:
                out["verdict"] = token
                break
        out["expr"] = _v
    # 验算正文：结论行 + 依据行之后的内容
    out["derivation"] = cls._tail_after_conclusion(text)
    out["has_derivation"] = (
        len(out["derivation"].strip()) >= cls._MIN_DERIVATION_CHARS)
    out["ok"] = bool(out["kind"])
    if not out["kind"]:
        out["reason"] = "结论形态无法识别"
    return out


def _extract_derivation_block(raw: str) -> str:
    """从模型原文里取「【推导过程】…（到下一个 【 段头或结尾）」块（无则空串）。"""
    if not raw:
        return ""
    m = re.search(r"【推导过程】\s*\n?(.*?)(?=\n?【|$)", str(raw), re.DOTALL)
    return m.group(1).strip() if m else ""


def _tail_after_conclusion(cls, text: str) -> str:
    """去掉 `本步结论:` / `本步依据:` 两行后剩下的正文（= 验算/论证段）。"""
    lines = str(text or "").split("\n")
    kept = []
    for ln in lines:
        if cls._CONCL_LINE_RE.search(ln) or cls._SUPPORT_LINE_RE.search(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _judge_subgoal_objective(self, sg: dict, contract: dict,
                             result_text: str,
                             answers: list,
                             existing_idx: int = -1) -> tuple:
    """对一个子目标结论做 **0-LLM 客观判定**。

        返回 ``(verified_index, verdict, reason)``：
          · verdict: "correct" | "incorrect" | "undecided"
          · verified_index: >=0 = 客观判定的候选下标；-1 = 判不出
          · reason: **总是**非空说明（判不出时即为"为什么判不出"的留痕）

        客观信号（按可靠性排序，全部本地 SymPy / 0-LLM）：
          ① 契约等式断言：`本步结论: 等式 = L = R` → `_sympy_judge(L, R)`；
          ② 多候选符号多数：同一子目标多采样结果按 SymPy 等价归组取严格多数
             （复用 `_verify_candidates_objective`）；
          ③ `<check>` 协议断言：结果里显式声明的等式（复用 `_extract_check_asserts`）；
          ④ 以上皆不适用 → undecided + **明确原因**（不再静默计 -1）。
        """
    from .sub_goal_solver import _check_protocol_violations
    # 已由多 agent 路径判出 → 直接沿用（不重复判、不覆盖）
    if existing_idx is not None and existing_idx >= 0:
        return existing_idx, "correct", "多候选符号多数（多 agent 路径已判定）"
    # ① 契约等式断言
    _chk = []          # [(L, R, info)] 通过的 / 失败的
    if contract.get("kind") == "equation" and contract.get("lhs") \
            and contract.get("rhs"):
        _v, _info = self._sympy_judge(contract["lhs"], contract["rhs"])
        if _v == "fail":
            return (-1, "incorrect",
                    "本步结论等式经 SymPy 客观判定**不成立**"
                    + (f"（{_info}）" if _info else ""))
        if _v == "pass":
            _chk.append(("pass", ""))
    # ③ <check> 协议断言（结论行未给等式时兜底）
    if not _chk:
        for L, R in self._extract_check_asserts(result_text):
            _v, _info = self._sympy_judge(L, R)
            if _v == "fail":
                return (-1, "incorrect",
                        f"结果中 <check> 断言「{L} = {R}」客观判定不成立"
                        + (f"（{_info}）" if _info else ""))
            if _v == "pass":
                _chk.append(("pass", ""))
    if _chk:
        return 0, "correct", "等式断言经 SymPy 判定成立（恒等/数值为真）"
    # ② 多候选符号多数
    if answers and len(answers) >= 2:
        _idx = self._verify_candidates_objective(answers)
        if _idx >= 0:
            return _idx, "correct", "多候选符号等价取得严格多数"
        return (-1, "undecided",
                "多候选可解析但**无严格多数**（意见分歧，无客观依据选边）")
    # ④ 判不出 → 留痕说明原因
    kind = contract.get("kind") or ""
    # ★ 2026-10-07（问题 2.9）：**违反 `<check>` 协议要显式可见**，不能静默
    #   落成与"未给断言"同款的 undecided —— 二者根因与处置完全不同
    #   （前者=格式违约，后者=模型确实没给可核验对象）。
    _viol = _check_protocol_violations(result_text)
    if _viol:
        return (-1, "undecided",
                "违反 <check> 协议：<check> 内含非可核验对象（%r…），"
                "无法客观判定（应改写为「本步结论: 判定 = 真」或纯数学等式）"
                % (_viol[0][:40],))
    if not contract.get("ok"):
        return (-1, "undecided",
                "未达成结论契约：" + (contract.get("reason") or "无结构化结论"))
    if kind == "verdict":
        return (-1, "undecided",
                f"结论为『判定 = {contract.get('verdict') or '?'}』型自然语言判定，"
                "无外部真值可比对，客观手段判不出真伪")
    if kind == "value":
        return (-1, "undecided",
                f"结论为『值 = {contract.get('expr')}』型单值断言，"
                "单次求解无可比对对象（多采样时方可按符号多数判定）")
    return (-1, "undecided", "无可用客观判据")


def _apply_conclusion_contract(self, sg: dict, result_text: str,
                               raw_text: str = "") -> None:
    """把结论契约结果落到 `sg`（供 `run()` 写进 trace）。异常一律吞掉。

        ★ 只写 `sg`，不改 `result_text` —— 返回给上层的字符串语义保持不变。
        `raw_text`（模型原始响应）仅用于**兜底找推导段**：prefill 会把
        「【本步结果】」钉在开头，模型若仍另外输出「【推导过程】」块，
        该块不在【本步结果】切片内，需从原文补取（否则 C 的「是否有推导」
        会误判为「无推导」）。
        """
    from .sub_goal_solver import _objective_judge_on
    try:
        contract = self._parse_step_contract(result_text)
        if not contract.get("has_derivation") and raw_text:
            _db = self._extract_derivation_block(raw_text)
            if len(_db.strip()) > len(contract.get("derivation") or ""):
                contract["derivation"] = _db
                contract["has_derivation"] = (
                    len(_db.strip()) >= self._MIN_DERIVATION_CHARS)
        _ans = list(sg.get("candidate_answers") or [])
        if not _ans and contract.get("kind") == "value" and contract.get("expr"):
            _ans = [contract["expr"]]
        _existing = int(sg.get("candidates_verified_index", -1) or -1)
        if _objective_judge_on():
            idx, verdict, reason = self._judge_subgoal_objective(
                sg, contract, result_text, _ans, existing_idx=_existing)
        else:
            idx, verdict, reason = _existing, "undecided", ""
        sg["conclusion"] = (
            contract.get("expr") if contract.get("ok") else "")
        sg["conclusion_kind"] = contract.get("kind", "")
        sg["conclusion_support"] = contract.get("support", "")
        sg["conclusion_contract_ok"] = bool(contract.get("ok"))
        sg["conclusion_reason"] = contract.get("reason", "")
        sg["has_derivation"] = bool(contract.get("has_derivation"))
        sg["derivation_chars"] = len(contract.get("derivation") or "")
        sg["objective_verdict"] = verdict
        sg["objective_reason"] = reason
        if _objective_judge_on():
            if idx >= 0:
                # 单次求解（无多 agent 候选）也要让客观判定**可归因**：
                # 候选集 = {本步自己的结论}，verified_index 指向它。
                if not list(sg.get("candidates") or []):
                    sg["candidates"] = [result_text]
                    sg["candidate_answers"] = _ans or [result_text]
                    sg["candidates_picked_index"] = 0
                sg["candidates_verified_index"] = idx
            else:
                sg["candidates_verified_index"] = -1
    except Exception as _e:  # noqa: BLE001  契约解析失败绝不影响求解
        logger.debug("结论契约落盘异常（忽略）: %s: %s", type(_e).__name__, _e)


def _contract_snapshot(cls, sg: dict) -> dict:
    """快照与结论契约/客观判定相关的 sg 字段（供决策缓存跨 run() 复用）。"""
    return {k: sg.get(k) for k in cls._CONTRACT_KEYS if k in sg}


def _restore_contract(cls, sg: dict, snap: dict) -> None:
    """把快照写回 sg（只写存在的键，异常吞掉）。"""
    if not isinstance(snap, dict):
        return
    for k in cls._CONTRACT_KEYS:
        if k in snap:
            sg[k] = snap[k]


def _answer_from_raw(self, raw: str) -> str:
    """从子目标采样原文取**可判定的答案串**（2026-10-06 结论契约）。

        结论契约达成 → 返回结构化结论（`值 =`/`等式 =` 的右半，或 `判定 =` 词），
        这是客观判定（SymPy 等价 / 多数）能直接消费的形态；
        未达成 → 退回既有的「本步结果」切片（保持旧行为）。
        """
    sec = self._extract_step_result(raw)
    c = self._parse_step_contract(sec)
    if c.get("ok") and c.get("expr"):
        return str(c["expr"]).strip()
    return sec


def _extract_step_result(text: str) -> str:
    """从子目标采样原文里抽取「本步结果」**干净答案串**（阶段二-2）。

        与 `_call_step` 的抽取口径**一字不差**（同一条正则 + 同一条剥壳），
        抽不出标记则取尾部 500 字符。空/异常一律返回空串。
        用途：`candidate_answers` —— 客观验证的输入（原文带推理没法算）。
        """
    from .sub_goal_solver import SubGoalSolverAgent
    try:
        if not text:
            return ""
        m = re.search(r"【本步结果】\s*\n?(.*?)(?:$|【)",
                      str(text), re.DOTALL)
        if m:
            return SubGoalSolverAgent._strip_tool_residue(
                m.group(1)).strip()
        return SubGoalSolverAgent._strip_tool_residue(
            str(text)).strip()[-500:]
    except Exception:  # noqa: BLE001
        return ""


def _verify_candidates_objective(answers: list) -> int:
    """子目标层**客观验证**：返回"已验证为对"的候选下标（判不出 = -1）。

        客观信号（**0-LLM，纯本地 SymPy**；只做裁决、不确定一律放行）：
        `AnswerOracle.verify_computational` 的**自洽共识** —— n 个候选答案按
        SymPy **符号等价**归组，只有落在**严格多数簇**里的候选才拿到 `correct`。
        · 恰有一个严格多数簇 ⇒ 该簇即"客观验证存活的那个"（按正确率选，
          **不是按长度/字符串多数**——这才是用户要的"按照正确率选取"）；
        · 判不出（无多数 / 全 unknown / 不可解析 / 异常）⇒ -1 ⇒ 上层沿用原代表解。

        ⚠ 与 `_pick_consensus_sample` 的区别：后者用**字符串化末 60 字**当键
          （`\boxed{5}` 与 `5` 会算成两族）；本方法用 SymPy 等价（同源口径见
          `AnswerOracle.answers_equivalent`），是**更可靠的"按正确率"信号**。
        """
    if not answers or len(answers) < 2:
        return -1
    try:
        from .answer_oracle import AnswerOracle
        # ⚠ `verify_computational` 是**实例方法**（内部 `self.is_parseable`）——
        #   必须用实例调用，不能 `AnswerOracle.verify_computational(cand, ...)`
        #   （那会把 cand 当成 self ⇒ AttributeError ⇒ 静默判不出）。
        _oracle = AnswerOracle()
        cands = [type("_C", (), {"answer": a, "reasoning": ""})()
                 for a in answers]
        good: list = []
        for i, a in enumerate(answers):
            if not str(a or "").strip():
                continue
            try:
                res = _oracle.verify_computational(cands[i], cands)
            except Exception:  # noqa: BLE001  单候选判定异常不放弃整组
                continue
            if str(getattr(res, "verdict", "")) == "correct":
                good.append(i)
        if not good:
            return -1
        # 同簇多个（等价）⇒ 取**原文更长**者（推理更完整；确定性收尾）
        return max(good, key=lambda i: len(str(answers[i])))
    except Exception:  # noqa: BLE001
        return -1


def _oracle_check_step(step_result: str) -> str:
    """用 AnswerOracle 对子目标结果做客观 sanity check，返回反馈（空=通过）。"""
    try:
        from .answer_oracle import AnswerOracle
        # 结果可解析为数学表达式 → 通过；否则视为非法（可能为幻觉/格式错误）
        if not AnswerOracle.is_parseable(step_result):
            return "该步结果无法解析为有效数学表达式"
    except Exception as exc:  # noqa: BLE001
        # 2026-09-04 审核：异常吞掉 = 子目标 sanity check 静默放行。留证据。
        logger.debug("子目标 sanity check 异常（放行）: %s: %s",
                     type(exc).__name__, exc)
    return ""


def _looks_placeholder(cls, result: str) -> bool:
    """短结果占位/空转检测：占位词命中 或 平凡 calc 回填（无运算发生）。"""
    if not result:
        return False
    text = result.strip()
    # 平凡 calc 回填：整体就是一行 `[计算] X = X`（结果≤120 字符内判定）
    if len(text) <= 120:
        m = cls._TRIVIAL_CALC_RE.match(text)
        if m and m.group("left").strip():
            left = re.sub(r"\s+", "", m.group("left")).replace("^", "**")
            right = re.sub(r"\s+", "", m.group("right")).replace("^", "**")
            if left == right:          # 去空白后两侧相同 = 无运算发生
                return True
    if len(text) > 50:            # 长结果视为有内容，不做占位词判定
        return False
    return bool(cls._PLACEHOLDER_RE.search(text))


def _extract_lean_code(cls, result: str) -> str:
    """从子目标结果提取 lean 代码片（fence 或裸 import 块）；无则空串。"""
    if not result:
        return ""
    m = cls._LEAN_FENCE.search(result)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 无 fence：整段以 import 开头（疑似裸 lean 代码）时收整段
    head = result.strip()[:80]
    if head.startswith("import ") or head.startswith("import\n"):
        return result.strip()
    return ""


def _subgoal_light_check(self, ctx: TaskContext, sg: dict,
                         result: str) -> str:
    """子目标结果 0-LLM 校验，返回反馈文本（空=通过）。异常全部放行。"""
    from .sub_goal_solver import _exhaust_missing_sections
    try:
        if not result or not result.strip():
            return "【本步结果】为空，请给出完整结果。"
        # L0P 占位/空转检测（全档，0 LLM）——2026-09-07 冒烟实证拦截项
        if self._looks_placeholder(result):
            return ("【本步结果】疑似占位/空转：仅给出“（无）”“已求解”等占位词，"
                    "没有真正给出基于题目条件的推导与结论。"
                    "请真正求解：给出基于题目条件的推导与数值/表达式结论。")
        # L0 截断检查（全档，0 LLM）
        from utils.extract import is_truncated_answer as _trunc
        if _trunc(result):
            return ("【本步结果】被截断（超出输出上限），"
                    "请压缩推理、只保留结论与关键计算，完整给出【本步结果】。")
        # ★★ 2026-09-17（P-9）：穷尽性检查的**格式校验提前到循环内**。
        # 原实现只在 `orchestrator._parse_exhaust_result`（事后 `_collect_diag`）
        # 解析 ⇒ 「缺段」这一**可机检**的硬信号来得太晚。实测 003（唯一的
        # 「求所有」题）：`exhaust_diag.appended=true`（子目标 6→7）、
        # `exhaust_result.complete=false`（四段全缺），而答案照旧提交 ——
        # 只给 `2026`、漏了 `2030`。
        # 现缺段 ⇒ 返回反馈 ⇒ 触发既有的「带反馈重解一次」分支。
        if str(sg.get("title", "") or "").strip() == "解族穷尽性检查":
            _miss = _exhaust_missing_sections(result)
            if _miss:
                self.record(ctx, "subgoal_exhaust_check",
                            "穷尽性检查缺段，打回重解: " + "、".join(_miss))
                return ("【本步结果】未按要求完成穷尽性检查 —— 缺少段落："
                        + "、".join(_miss)
                        + "。请**严格按固定格式**重写，四段必须具备：\n"
                        "解族清单: …（列出本题所有可能的解族类型）\n"
                        "逐族判定: …（对每一族给『存在 / 不存在』及其依据）\n"
                        "前序结论复核: …（前序找到的解属于哪一族、同族是否还有别的）\n"
                        "结论: EXHAUSTIVE: yes 或 EXHAUSTIVE: no\n"
                        "（若结论为 no，另加一段『遗漏解族: …』）")
    except Exception:  # noqa: BLE001
        pass
    # L1 lean 代码片编译（仅 deep 档 + lean 可用；禁网纯本地）
    try:
        if str(getattr(ctx, "tier", "") or "") != "deep":
            return ""
        if not getattr(self.config, "enable_subgoal_lean_check", True):
            return ""
        code = self._extract_lean_code(result)
        if not code:
            return ""
        from tools.lean_local.lean_bridge import LeanBridge
        from tools.lean_local.lean_bridge import _trash_lean_file
        bridge = LeanBridge(self.client, self.config)
        if not bridge.lean_available:
            return ""
        work_dir = bridge._lean_project_dir or ""
        if not work_dir:
            return ""          # 无 lake 工程 → 不尝试（避免直编找不到 Mathlib）
        fname = f"sg_check_{int(time.time() * 1000) % 1000000}.lean"
        r = bridge._compile(code, work_dir, lean_filename=fname)
        try:
            _trash_lean_file(work_dir, fname)
        except Exception:  # noqa: BLE001
            pass
        if not r.get("ok"):
            err = str(r.get("error", ""))[:200]
            return (f"本步给出的 Lean 代码编译失败（{err}）。"
                    "请修正代码（或改用数学表述，不必强制给 Lean）。")
    except Exception as exc:  # noqa: BLE001  校验失败放行，绝不阻断
        logger.debug("子目标 0-LLM 校验异常（放行）: %s: %s",
                     type(exc).__name__, exc)
    # L2/L3 数值断言验证（2026-09-08 去门后新钩子）：
    # L3 协议版优先——模型 <check> 显式声明的等式（sympy 判 + Lean 背书）；
    # L2 提取式兜底——无 <check> 时 0-LLM 提取纯数值/整式断言。
    # 失败 = 计算未通过验证 → 反馈带真值让 LLM 带错重解。0-LLM 构造；
    # 无数值断言零成本；异常/语法不支持一律放行不误报；每题限额防烧预算。
    try:
        _nv_fb = self._check_assert_verify(ctx, result)
        if not _nv_fb:
            _nv_fb = self._numeric_lean_verify(ctx, result)
        if _nv_fb:
            return _nv_fb
    except Exception:  # noqa: BLE001
        pass
    return ""


def _strip_check_tags(text: str) -> str:
    """剥 <check>...</check> 验证标签：去标签保留内部内容（数学等式文本无害）。

        仅当标签完整闭合时剥除；不闭合的残留 <check> 一并删除防泄漏。
        """
    if not text or "<check>" not in text:
        return text
    t = re.sub(r"<check>(.*?)</check>", r"\1", str(text), flags=re.DOTALL)
    # 兜底：未闭合/孤立标签删除
    t = t.replace("<check>", "").replace("</check>", "")
    return t


def _pow_to_python(s: str) -> str:
    """裸 ^ → **（先保护已有 **）。"""
    s = s.replace("**", "@@P@@")
    s = re.sub(r"\^", "**", s)
    return s.replace("@@P@@", "**")
