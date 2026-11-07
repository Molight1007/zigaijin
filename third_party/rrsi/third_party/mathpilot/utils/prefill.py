"""Assistant-prefill 解码：抑制 CoT 开启、让答案前置生成（借鉴 ICMA 高分样例）。

实测数据（intern-s2-preview 系列，同 prompt）：
  - 分类  普通 max_tokens=8192 → 40.4s,  1969 completion tokens
  - 分类  prefill max_tokens=64  →  0.9s,    12 completion tokens   (58×)
  - 仲裁  普通 max_tokens=8192 → 70.2s, ~2000 completion tokens
  - 仲裁  prefill max_tokens=64  →  0.8s,     4 completion tokens  (140×)

机制：在消息末尾追加 assistant 种子前缀，让模型进入"续写模式"，从而不会
先开启 `reasoning_content` 思维块。与"减小 max_tokens"不同：减 max_tokens
会在思维中途截断（finish_reason=length），答案永远不出现；prefill 是移除
CoT 而非腰斩它，因此可以与较小的 token 预算安全搭配——即使输出被截断，
也只损失思考、不损失答案（答案在开头）。

兼容性：仅追加一条 assistant 消息，通过平台注入的 client.chat(messages=...)
即可工作，无需改动 client。后端对"末尾 assistant 轮"的处理有 续写/回显/忽略
三种形态，`stitch()` 全部兼容；调用方保留无 prefill 的回退分支。
"""

from __future__ import annotations


def prefill_messages(messages: list[dict], prefix: str) -> list[dict]:
    """在消息末尾追加 assistant 种子前缀，抑制推理、让答案前置。"""
    return list(messages) + [{"role": "assistant", "content": prefix}]


def stitch(prefix: str, completion: str) -> str:
    """把 prefill 种子前缀与后端返回拼接回完整 assistant 文本。

    处理后端三种行为：
      * continuation —— completion 紧跟前缀继续 → 拼接
      * echo         —— completion 已复述前缀     → 原样使用
      * ignored      —— completion 是独立完整回答  → 原样使用

    仅靠前缀匹配区分 echo/continuation 不可靠（续写可能恰好以相同字符开头），
    因此同时接受"开头附近包含前缀"的 completion。
    """
    body = completion if isinstance(completion, str) else str(completion or "")
    seed = prefix if isinstance(prefix, str) else str(prefix or "")
    if not seed:
        return body
    stripped = body.lstrip()
    if stripped.startswith(seed):
        return stripped
    # 回显 + 前导包装文本（多余换行/代码围栏等）
    head = body[: len(seed) + 40]
    if seed in head:
        return body[body.index(seed):]
    return seed + body
