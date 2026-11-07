# -*- coding: utf-8 -*-
"""智能体工具集（4 个常用工具 · 2026-10-15 用户定稿）。

设计原则（用户原话）：
  「这几个要做成工具，不要和其他代码过耦合。智能体代码使用它的逻辑应该是
   用函数什么的调用代码板块，然后将要输入的发给工具，工具将输出的返回，
   智能体接收。」

⇒ 本板块**独立自包含**：对外只暴露 4 个函数（输入 dict/标量 → 输出 dict），
  不反向依赖 agent 业务层；智能体环节以「准备输入 → 调用 → 接收输出」使用。

工具清单：
  ① lean_check    Lean4 逻辑审核 —— 逻辑错误点 + 逻辑缺少点（纠错并带回修正线索）
  ② leansearch    定理检索     —— LeanExplore 主通道 + 离线语料降级（思考方向建议）
  ③ calc_verify   数学计算验证 —— 逐条机械复算（算错处 + 正确答案）
  ④ to_lean       Lean 语言转化 —— 自然语言陈述 → Lean 4 代码（LLM 翻译，client 注入）
"""
from .calc_verify import calc_verify
from .lean_check import lean_check
from .leansearch import leansearch
from .to_lean import to_lean

__all__ = ["lean_check", "leansearch", "calc_verify", "to_lean"]
