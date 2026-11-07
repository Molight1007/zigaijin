# -*- coding: utf-8 -*-
"""验证器原始产出的解析与清洗（从 agent/verifier.py 逐字搬出）
============================================================

2026-10-08 批6：把 ``agent/verifier.py`` 里**纯函数/常量**职责簇整体搬到这里，
原文件 re-export 同名符号 ⇒ 外部 ``from agent.verifier import <name>`` 与内部
引用契约**逐字不变**。本模块**不依赖** verifier.py 的任何符号（自洽）。

  - ``ERROR_TYPE_TAGS`` / ``_extract_error_type``：判 B 票的错误类型标签抽取
  - ``_VERDICT_*_RE`` / ``_last_verdict_ab``：带推理复核的判定解析
  - ``_DEEP_VERDICT_*`` / ``_dump_deep_review_sample``：深复核收敛提示词 + 样本落盘
  - ``_clip_review_prose`` / ``_REVIEW_*``：复核反馈正文裁剪（orchestrator 复用）

⚠ 勿按"未使用"清理这里的任何名字 —— 全部是对外契约（见 verifier.py 的 re-export）。
"""

import re


# ★ 2026-09-15（用户要求"测试时把错误暴露得更具体"）：
# 与 `prompts/verifier.py` 的 VERIFIER_SYSTEM 约定的**封闭标签集**。
# 解析时只认这些标签 ⇒ 既能结构化统计，又避免模型自造新词污染归因。
ERROR_TYPE_TAGS = (
    "符号错", "计算错", "边界遗漏", "定义域错", "跳步", "循环论证",
    "方法不适用", "前提不成立", "方向反了", "漏分支", "其它",
    # ★ 2026-09-15 补：10 题实测里**真实出现但标签集里没有**的两类错法。
    # 099「只答一个方法、正解多个」与 103「答 ABCD 漏 E」都无处归类，
    # 只能塞进"其它"⇒ 归因失真。（题型提示词里早就写了"按问法定项数"，
    # 但没有任何标签能记录它是否被遵守。）
    "漏项", "多项",
)
# 标签出现形态：VERDICT: B（A, B）或 VERDICT: B(A)
_ERR_TAG_RE = re.compile("|".join(ERROR_TYPE_TAGS))


def _extract_error_type(text: str) -> str:
    """从投票原文里抽出结构化错误类型标签，返回逗号分隔串（无则空串）。

    只在**判 B 的票**上调用（A 票没有错误类型）。命中多个标签时按标签集
    在文中的**出现顺序**去重拼接，便于统计"最常见的错误组合"。
    """
    if not text:
        return ""
    hits, seen = [], set()
    for m in _ERR_TAG_RE.finditer(text):
        tag = m.group(0)
        if tag not in seen:
            seen.add(tag)
            hits.append(tag)
    return ",".join(hits)


# 带推理复核专用：优先取**开头**的 VERDICT 行（新格式），否则取**最后**一个（旧格式）。
# ⚠ 背景（2026-09-16 实测，两题同时命中）：
#   旧模板要求「四步逐条书写检查过程，**最后**用单独一行给出 VERDICT」，
#   而 `verifier_deep_review_max_tokens` 默认 **16384** ⇒ 模型把预算全用在推理上，
#   **还没写到 VERDICT 就被截断**（日志里大量 `finish_reason=length, max_tokens=16384`）
#   ⇒ 判定为"无法解析" ⇒ 复核被跳过。实测两题输出 **43,960 / 37,633 字符**却都无 VERDICT。
#   现改为「**VERDICT 写在最前面**」⇒ 无论是否截断，判定一定拿得到。
# 保留取末尾的分支是为了向后兼容旧格式与意外顺序。
_VERDICT_ANY_RE = re.compile(
    r"VERDICT\s*[:：]\s*(?:\\boxed\s*\{\s*)?([AB])", re.IGNORECASE)
# 判定行是否**位于输出最前**（允许前导空白/加粗标记）。
# ⚠ 不能用"前 N 字符内"这种宽松窗口：实测文本
#   「先看看：如果前提不成立则 VERDICT: B。\n但独立重算后一致，最终 VERDICT: A」
#   的首个 VERDICT 就落在前 20 字符内，宽松窗口会误取**假设句**。
#   锚定行首才能真正对上提示词「第 0 步 · 必须是输出的第一行」的语义。
_VERDICT_HEAD_RE = re.compile(
    r"^\s*(?:\*\*)?VERDICT\s*[:：]\s*(?:\\boxed\s*\{\s*)?([AB])",
    re.IGNORECASE)


def _last_verdict_ab(text: str):
    """解析带推理复核的判定。返回 True(A)/False(B)/None(无法解析)。

    策略：
      · VERDICT **出现在输出最前**（新格式：判定置顶，防截断）⇒ 取**首个**；
      · 否则取**最后一个**（旧格式：判定在末尾；同时避免误采推理中的假设句）。
    """
    if not text:
        return None
    m = _VERDICT_HEAD_RE.match(text)
    if m:
        return m.group(1).upper() == "A"
    hits = _VERDICT_ANY_RE.findall(text)
    if not hits:
        return None
    return hits[-1].upper() == "A"


# ★★ 2026-09-16：深复核的**第二段收敛调用**专用系统提示词。
# 背景：第一段自由推理**实测 8/8 不给 VERDICT**（模型无视格式要求，写英文 CoT），
# 且会陷入复读退化（同长度 26749 字符、尾部 `a_{a_1} = a_{a_1} = ...` 无限重复）。
# 故追加一次极小成本的收敛调用，只让它把已有分析**压成一行判定**。
_DEEP_VERDICT_CONDENSE_SYS = (
    "你刚刚完成了一次数学解答复核（上文是你的分析）。"
    "现在只需把结论压缩成**一行**，严禁输出任何其它文字、解释或标点修饰。\n"
    "格式（二选一，必须严格照抄形状）：\n"
    "VERDICT: A\n"
    "VERDICT: B（错误类型标签）\n"
    "其中 A = 候选答案与你的独立重算完全一致且未发现问题；"
    "B = 发现任何错误、不一致或无法确认之处。\n"
    "错误类型标签只能从这个集合里选（可多选、逗号分隔、不得自造）："
    "符号错 / 计算错 / 边界遗漏 / 定义域错 / 跳步 / 循环论证 / "
    "方法不适用 / 前提不成立 / 方向反了 / 漏分支 / 漏项 / 多项 / 其它\n"
    "**必须以上述 `VERDICT:` 开头，这是唯一允许的输出形式。**"
)


# 收敛调用用的 prefill 前缀（结构性保证输出以 `VERDICT: ` 开头，见调用处注释）
_DEEP_VERDICT_PREFIX = "VERDICT: "


def _dump_deep_review_sample(ctx, raw: str, verdict) -> None:
    """把一次带推理复核的输出样本追加到 ``results/deep_review_samples.jsonl``。

    背景：`diag.deep_review` 只存 `chars`，导致"4 万字符却解析不到 VERDICT"
    **无从诊断**。本函数记录**头部/尾部**样本（不存全文，避免文件膨胀），
    用于回答"模型到底写没写 VERDICT、以什么格式写"。

    统计口径：同时记 `verdict_first_line`（首行是否直接就是 VERDICT），
    便于区分"模型没遵守置顶"与"解析器覆盖不足"两种根因。
    """
    import json as _json
    import os as _os
    import time as _time

    # ⚠ 2026-09-16 实测：跑 pytest 时会走 `fake_output()` 桩，把**测试数据**
    #   写进生产诊断文件（`results/deep_review_samples.jsonl`），污染真实样本
    #   （实测一次回归留下 3 条 273/15/280 字的假样本）。pytest 会设
    #   `PYTEST_CURRENT_TEST` 环境变量，据此跳过测试期落盘。
    if _os.environ.get("PYTEST_CURRENT_TEST"):
        return

    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    path = _os.path.join(root, "results", "deep_review_samples.jsonl")
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    head = raw[:1200]
    tail = raw[-400:] if len(raw) > 1600 else ""
    first_line = (raw.split("\n", 1)[0] if raw else "")[:200]
    rec = {
        "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "qid": str(getattr(ctx, "task_id", "") or getattr(ctx, "qid", "") or ""),
        "chars": len(raw),
        "verdict_parsed": ("A" if verdict is True else "B" if verdict is False
                           else "?"),
        "verdict_first_line": first_line,
        # 首行是否就是 VERDICT（模型是否遵守"置顶"要求）
        "first_line_is_verdict": bool(_VERDICT_HEAD_RE.match(raw or "")),
        "n_verdict_tokens": len(_VERDICT_ANY_RE.findall(raw or "")),
        "head": head,
        "tail": tail,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")


def _clip_review_prose(text: str) -> str:
    """裁掉复核输出里的**重做正文**与**越权结论断言**，只留诊断。

    ★★ 2026-10-06 新增（**实测驱动：反馈把答案喂给了模型**）：
    题 000 的反馈第 0 条 2185 字符，开头是：
        「保留的缺陷：
          ① 原答案 40 错误，**正确答案应为 0。下面给出严谨证明。**
          **证明：** 设 G 为题中描述的图…（其后 2000+ 字符是复核者**自己重做了一遍**的解答）」
    trace 完整记录了后果链：
        [review] 反馈复核完成：…正确答案应为 0。下面给出严谨证明…
        [revise] 重解启动 count=4、反馈 1 条、revise_round=1   ← 模型拿到这段
        最终 = \\boxed{0}     而标准 = 20460
    ⇒ **复核者自己判错（说"正确答案应为 0"），模型照抄了这个结论。**

    两重问题：
      ① 「**正确答案应为 X**」是**结论性断言** —— 验证器越权替解答下结论；
      ② 后面附了**完整重做正文** —— 模型可整段照抄（远比自己去改省力）。
    这违反验证器自身纪律：其系统提示词明写「**绝不要试图纠正你发现的错误或填补漏洞**
    —— 一旦你开始替它修正，就会不自觉地**为它辩护**」。

    判据：只保留「判定 + 诊断」，在首个"重做/mark"标记处**截断**；
    并逐行剔除带结论断言的句子。**无标记时原样返回**（保守，不误删诊断）。
    """
    t = str(text or "")
    if not t.strip():
        return ""
    cut = _REVIEW_PROSE_CUT.search(t)
    if cut and cut.start() > 0:
        t = t[:cut.start()]
    # 再逐行剔除含"正确答案应为/是 X"这类越权断言的句子（但保留整段为空则不剔）
    lines = []
    for ln in t.split("\n"):
        if _REVIEW_VERDICT_CLAIM.search(ln):
            continue
        lines.append(ln)
    out = "\n".join(lines).strip()
    return out if out else t.strip()


_REVIEW_PROSE_CUT = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?\s*(?:"
    r"下面(?:给出|是)?(?:严谨|完整)?(?:证明|推导|解法|解答)"
    r"|证明\s*[:：]|证明如下|正确的?(?:答案|解法|结论)(?:应(?:该|当)?为|是|：)"
    r"|因此[，,]?\s*正确(?:答案|解法)"
    r"|故正确(?:答案|解法)"
    r")")

# "正确答案应为 X" 这类**结论性断言**（验证器不应下这种结论）
_REVIEW_VERDICT_CLAIM = re.compile(
    r"正确的?(?:答案|结果|值)(?:应(?:该|当)?为|是)\s*\S+"
    r"|答案(?:应(?:该|当)?为|是)\s*\\?boxed")
