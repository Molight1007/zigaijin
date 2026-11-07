# -*- coding: utf-8 -*-
"""超时上下文 + 幻觉/截断/模板泄露检测 —— 从 agent/base.py 逐字搬出。

对外契约不变：``from agent.base import wall_clock_timeout, detect_hallucination,
detect_truncated, detect_template_leak, TimeoutError`` 仍可用（re-export）。
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Iterator

# ============================================================
# 安全防护工具
# ============================================================

class TimeoutError(Exception):
    """wall-clock 超时异常"""


@contextmanager
def wall_clock_timeout(seconds: float, label: str = "") -> Iterator[None]:
    """
    wall-clock 超时上下文管理器。

    在 Windows 上使用 threading.Timer 实现（signal.alarm 不可用），
    超时时会终止当前线程内的阻塞调用。

    用法::

        try:
            with wall_clock_timeout(30, "Solver"):
                resp = client.chat(...)
        except TimeoutError:
            # 超时处理
    """
    timer = [None]  # mutable 闭包

    def _raise():
        import _thread
        _thread.interrupt_main()

    if seconds <= 0:
        yield  # 不限制
        return

    try:
        timer[0] = threading.Timer(seconds, _raise)
        timer[0].daemon = True
        timer[0].start()
        yield
    except KeyboardInterrupt:
        raise TimeoutError(f"{label} timed out after {seconds}s")
    finally:
        if timer[0] is not None:
            timer[0].cancel()


# 模型常见幻觉模式
_HALLUCINATION_PATTERNS = [
    # 孤立的 42/42.0 作为答案（避免误匹配 142、42x 等）
    (re.compile(r"(?:^|(?<=\s))42(?:\.0+)?\s*(?:[，,。.\n]|$)", re.MULTILINE),
     "42 魔法数字兜底"),
    (re.compile(r"I am (?:sorry|unable).*?(?:\n|$)", re.IGNORECASE),
     "英文拒绝回答"),
    (re.compile(r"as an AI.*?(?:\n|$)", re.IGNORECASE),
     "AI 身份声明"),
    (re.compile(r"我(?:无法|不能|没办法|不擅长).{0,20}(?:解|答|计算)", re.IGNORECASE),
     "中文拒绝回答"),
    (re.compile(r"根据.{0,10}安全.{0,10}政策", re.IGNORECASE),
     "安全策略拒绝"),
]


def detect_hallucination(text: str) -> list[tuple[str, float]]:
    """
    检测 LLM 输出中的已知幻觉模式。

    返回: [(模式名, 置信度), ...]，空列表表示无检测。
    """
    if not text:
        return [("空响应", 1.0)]
    findings = []
    for pattern, label in _HALLUCINATION_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # 置信度基于匹配次数（但不超过 0.9）
            confidence = min(0.5 + 0.15 * len(matches), 0.9)
            findings.append((label, confidence))
    return findings


def detect_truncated(text: str) -> bool:
    """
    检测 LLM 输出是否被 max_tokens 截断（代码/LaTeX 不完整）。

    检测规则:
    - 以不闭合的 LaTeX 环境结尾（\\begin 无 \\end）
    - 以不闭合的 markdown 代码块结尾
    - 以不闭合的大括号/中括号/括号结尾
    - 以行内 $$ 无闭合结尾
    """
    if not text:
        return False
    stripped = text.rstrip()
    # 未闭合 LaTeX 环境
    begins = len(re.findall(r"\\begin\{([^}]+)\}", stripped))
    ends = len(re.findall(r"\\end\{([^}]+)\}", stripped))
    if begins > ends:
        return True
    # 未闭合的代码块
    code_fences = len(re.findall(r"^```", stripped, re.MULTILINE))
    if code_fences % 2 == 1:
        return True
    # 未闭合的括号（排除 LaTeX 转义的 \{ \}）
    bracket_pairs = {"{": "}", "[": "]", "(": ")"}
    stack: list[str] = []
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        # 跳过转义字符（如 \{, \}）
        if ch == "\\" and i + 1 < len(stripped) and stripped[i + 1] in ("{", "}", "[", "]", "(", ")"):
            i += 2
            continue
        if ch in bracket_pairs:
            stack.append(bracket_pairs[ch])
        elif ch in bracket_pairs.values():
            if stack and stack[-1] == ch:
                stack.pop()
        i += 1
    if stack:
        return True
    # 未闭合的 $ 标记（$$ 算作双美元符号，检查偶数对）
    # 移除转义后的 $ 计数
    dollar_count = len(re.findall(r"(?<!\\)\$", stripped))
    if dollar_count % 2 == 1:
        return True
    return False


# ============================================================
# Intern-S1 思考链污染检测（英文 thinking chain 污染）
# ============================================================
# 当使用 Intern-S1 等模型的 thinking_mode 时，英文思考链可能污染中文答案。
# 参考 math_agent-main 的 ENGLISH_THINK_PATTERNS。

_ENGLISH_THINK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"<\|im_start\|>assistant", re.IGNORECASE), "特殊 token 泄露"),
    (re.compile(r"Please respond (in|with)\s", re.IGNORECASE), "英文指令残留"),
    (re.compile(r"Sure[,!]?\s+I.*?(?:\n|$)", re.IGNORECASE), "英文肯定前缀"),
    (re.compile(r"(I think|Let me|First,? let me|Now I will)", re.IGNORECASE), "英文自言自语"),
    (re.compile(r"Here[\'']s the solution:", re.IGNORECASE), "英文解题标记"),
    (re.compile(r"(User:|Assistant:|System:)\s", re.IGNORECASE), "角色标记泄露"),
]



# ============================================================
# 模板泄露检测（借鉴 math_agent-main 的成功经验）
# Intern-S 模型常会输出 prompt 模板描述而非实际解答，这是致命问题。
# ============================================================
_TEMPLATE_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(r'(?i)Strategy Planning\)?.*specific to this problem'),
    re.compile(r'(?i)Key Insight\)?.*specific to this problem'),
    re.compile(r'(?i)Heuristic Summary\)?.*specific to this problem'),
    re.compile(r'(?i)<Specific mathematical conclusion'),
    re.compile(r'(?i)<Final Answer only'),
    re.compile(r'(?i)1-2 sentences analyzing core'),
    re.compile(r'(?i)1 sentence highlighting the key'),
    re.compile(r'(?i)1-2 sentences explaining broader'),
    re.compile(r'(?i)必须针对本题具体内容.*不要泛泛而谈'),
    re.compile(r'(?i)用1-2句话分析'),
    re.compile(r'(?i)用1句话点出'),
    re.compile(r'(?i)用1-2句话说明'),
    re.compile(r'(?i)解题步骤（摘要）.*Make sure'),
    re.compile(r'你是一位.{1,30}(?:专家|助手|模型)', re.IGNORECASE),
    re.compile(r'请按照.{1,30}格式', re.IGNORECASE),
    re.compile(r'SYSTEM_PROMPT|TEMPLATE_CONTENT'),
]


def detect_template_leak(text: str) -> bool:
    """检测模型是否输出了 prompt 模板描述而非真正解答。"""
    if not text or len(text.strip()) < 20:
        return False
    leak_count = sum(1 for p in _TEMPLATE_LEAK_PATTERNS if p.search(text))
    if leak_count >= 2:
        return True
    # 检测 ANSWER: 后面跟的是模板占位符
    m = re.search(r'(?:ANSWER|答案|最终答案)\s*[：:]\s*(.+)', text, re.IGNORECASE)
    if m:
        ans_text = m.group(1).strip()
        if re.search(r'(?i)<[^>]*(?:Specific|Final Answer|具体数学结论|最终答案)[^>]*>', ans_text):
            return True
        if re.search(r'(?i)(?:must include|no more than|only.*no reasoning)', ans_text):
            return True
    return False
