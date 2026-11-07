# official112 本地测试题库（复现自样例仓库的"预存题目+答案"）

> 生成日期：2026-09-09 ｜ 依据：`其他优秀样例/math_agent-main` + `ICMAnew-main`（两仓库确认同源、存同一套 112 题）
> 用途：**本地评测基准**（与现有 `su01_ab45.jsonl`/`smoke10_*.jsonl` 同类的 bank），**不用于赛事平台提交**。

## 文件清单

| 文件 | 内容 | 用途 |
|---|---|---|
| `official112_full.jsonl` | 112 行：`id / question / domain / answer / meta` | 主题库（推荐评测用） |
| `official112_noanswer.jsonl` | 112 行：`id / question / domain`（无 answer） | 只测题、防误泄答案 |
| `official112_full.csv` | 同全量（utf-8-sig，Excel 直接开） | 人工浏览/筛选 |

字段与 `测试工具/loader.py` 别名映射兼容：`id`、`question`、`domain`（← category/type）、`answer`（← reference_answer/solution）。判分器只在打分阶段读 `answer`，不注入模型推理提示词。

## 数据来源与可信度
- **题目 + 答案（权威）**：`math_agent/knowledge/official_golds.jsonl`，112 条。其代码注释自述："官方 112 题核定表……表来自早期评测包 outputs/{idx}.json 的 problem + reward_model.ground_truth"。→ 本库 `question`=题面原文，`answer`=gold 原文，**逐条 1:1 复制**，无改写。
- **学科与解题口径（增强，仅供参考）**：把 18 册 skill 手册里的"解法直达/判分口径"卡按 `\boxed{}` 判定值与 gold 相等的规则关联到对应 idx，写入 `meta.card_titles`（卡标题，自带题目摘要+答案）与 `meta.card_hints`（错值警示/适用条件摘录，≤160 字）。
- 注意：卡关联用的是"答案值相等"，个别通用答案（如 idx 48/52/66/109 的 gold=8/2/4/D）可能与多张不同卡重影或归属错位——`subject` 仅供参考，**判分永远以 `answer` 字段为准**。

## 卡覆盖情况
- 有卡可关联：**87/112** 个 idx（`meta.card_count>0`，卡标题/错值警示可当"标准解法提示"检索）。
- **完全没有卡（25 个）**，这些题两个样例仓库也只有题面+答案、无预写口径：
  `15, 21, 27, 48, 50, 52, 62, 63, 64, 65, 66, 78, 86, 88, 91, 92, 93, 94, 95, 96, 98, 99, 100, 109, 110`
- 注：早期按单卡单命中口径计算时缺口略有出入（21/48/50/52/63/66/78/109/110 等通用答案值存在归属歧义），以上表为准。

## 题面概览
- 主语言：英文 + LaTeX（多数题尾带 `Remember to put your final answer within \boxed{}.` 提示句），少量中文客观题/判断题（idx 101–111，答案多为 A/B/D/正确/错误）。
- idx 0–100 主要为组合/离散/代数/分析类竞赛题（很多来自公开竞赛题改编，如 2002 IMO Shortlist A2 型——见卡内推导链）；idx 101–111 为统计/计量/运筹等客观题。
- 分布参考（按卡 domain）：离散数学最多（约 50+），其次高等代数/数学分析/概率论/非基础及进阶课程/统计推断/线性回归/运筹学/抽象代数/偏微分方程/数值分析/随机过程。

## 快速使用
```python
# 用测试工具的 loader 直接读
import sys; sys.path.insert(0, r"D:\挑战杯\测试工具")
from loader import load_problems
ps = load_problems(r"D:\挑战杯\题库\official112_本地测试题库\official112_full.jsonl")
print(len(ps), ps[0].id, ps[0].domain, ps[0].reference_answer)
```
或从命令行/评测器传入 jsonl 路径作为 bank（字段名与现有 bank 一致，无侵入）。

## 合规边界（重要）
- 本库的 `answer` 源自样例仓库内嵌的"评测包/判分口径"回灌数据。**仅限本地对照、离线评测**；
- 平台规则敏感：不要把这个库的答案或题面以任何形式带入平台提交链路/提示词（防止判分口径污染与合规风险）；
- 用它测出的正确率代表"查表覆盖率 + 模型能力"的混合，不代表模型真实推理水位——如需衡量推理能力请用无答案库或自行出题。

---

## 题面完整性核查（2026-09-09 补）
- 对 official_golds.jsonl 全部 112 条 `problem` 逐条体检：**无一截断**。
- 长度 72–1312 字符；112/112 条均以正常句式收尾，其中 112/112 条末尾带评测统一提示句 `Remember to put your final answer within \boxed{}.`（AI-MO 类评测题的固定格式，非拼接痕迹；样例仓库 `official_golds.py` 的 stems() 也专门兼容"带/不带该提示句"两种形态）。
- 本库 `question` 字段 = 上述 `problem` **逐字 1:1 复制，无任何补全/改写**；idx 101–111 为中文客观题，同样保留原文+统一英文提示句。
