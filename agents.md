# AGENTS.md

把「用户反馈 → 可复现评测 → 可验证实验 → 沉淀」压到天级。不训练权重，改的是 harness 本身。
当前目标是**产出论文**：被测对象用别人论文里的 agent（官方 Terminus-2 为对标锚点；
OpenHands / mini-swe-agent 作为强/弱基线）+ 别人的数据集（terminal-bench 2.1），
我们的工具是**外层** harness，经适配层接入，不是改用别人的系统。

本文件是项目纪律的**唯一权威出处**。原先的 `资料文档/01_课题依据/目标与验收底线_20261031.md`
与自检脚本 `docs/_goal_audit.py` 已随工作区重组消失，不要再去引用那两个路径。

## 跑起来（命令均已实测）

系统 Python **没装 radon**，直接 `python -m evo` 会 ModuleNotFoundError。一律用工作区 venv：

```
.venv\Scripts\python.exe -m pytest                 # 50 文件 / 808 用例
.venv\Scripts\python.exe -m evo <子命令> --help    # 34 个子命令
```

子命令：`version measure scan prestudy ablate ablate-run census traces-merge baseline
harbor-results harbor-run run simplify workforce simplify-effect incident ai-cleanup ledger state freeze
compare power version-compare round evolve-round propose diagnose diagnose-merge faults fault-cases
fault-run report verify-evidence apply`

依赖只有 `radon / PyYAML / filelock / pytest`。**不要引入 numpy、pandas、scipy、matplotlib**——
统计口径是自己手写的（`evo/domain/stats.py`、`evo/application/power.py`、`evo/domain/gate.py`），
语义特殊（dS 不对称、噪声带、δ 地板），通用统计库会把它冲掉。

## 目录

| 路径 | 内容 |
|---|---|
| `evo/adapters/` | 外部接入：`deepseek` `llm` `git_worktree` `harbor_tb21` `store_fs` `coverage_py` `traced_terminus2` |
| `evo/application/` | 用例编排：`ablate` `apply` `armrun` `baseline` `compare` `diagnose` `evolve` `faultcase` `faultrun` `freeze` `power` `prestudy` `propose` `report` `round` `runstate` `simplify` `simplify_arm` `simplify_effect` `traces` `version_compare` `workforce` |
| `evo/domain/` | 判据与度量：`gate` `kmetrics` `stats` `ablation` `ai_cleanup` `arms` `census` `components` `coreset` `critic` `evolve` `faults` `generalization` `incidents` `ledger` `prestudy` `protection` `protocol` `scan` `state` `verify` `workforce` |
| `evo/contracts/` | 数据契约 |
| `tests/` | 50 文件 808 用例 |
| `specs/terminus2` `vendor/terminus2` | 被测基准规格与源码 |
| `third_party/rrsi` | 对标论文 RRSI 源码 |
| `external/` | 数据集、论文 PDF、对标仓库（**gitignore，不要提交**） |
| `runs/` | 实验产物（**gitignore**） |
| `论文准备/` | 29 份论文用文档（**gitignore**） |
| `.tmp/` | 临时文件**只准落这里**（gitignore） |

`D:\智能体经验集\` 三份 Word（要求 R / 错误 E / 标准 S）是**跨项目**经验，不是本课题交付物。
拒绝某条建议时注明编号，便于统计哪条出现最多。

## 九条硬线

| # | 要求 | 判据 |
|---|---|---|
| R1 | 可复现错误 | 同条件跑 10 次 = **10/10 错**；n<10 的样本不得当优化证据 |
| R2 | 失败样本池 | 模型/提示词/检索/工具返回冻结 + `expect` 判据，可回放 |
| R3 | 一次只改一处 | 每次 diff 只落一个单元；改前预注册标准（主指标/不可退项/条数） |
| R4 | 只写三样 | 适配层 / 实验流程 / 归因算法；**别重造** Langfuse、Snorkel、Docker |
| T1 | 地板 `S*−δ` | **不是** `S_t−δ`——相对上一轮每轮掉 δ，T 轮累积掉 T·δ |
| T2 | 纯压分支 | 非反馈轮每轮额外一次纯压，与 RRSI 共用**同一块**地板 |
| T3 | 抗泄漏靠 critic | ΔK 只是字节数，不区分内容来源；必须评测前语义级预筛 |
| T4 | ν 与删除冲突 | 因 ν 准入的组件，`n_prune` 轮内不得成为删除目标 |
| A1 | 降本/无效实验/采用率 | 口径须写清基线来源；采用率需 ≥3 位真人 |

## 开工前 5 问

1. 这轮做的是哪条编号？
2. 会不会让已做到的退回去？
3. 证据是否依赖我先知道答案？
4. n 够不够？
5. 措辞超过证据了吗？

## 代码与证据纪律（长期有效）

**① docstring 短，且必须同笔更新。** 一段 ≤4 行：1 行做什么 + ≤2 行关键约束 + 1 行锚点。
改签名 / 行为 / 返回 / 异常，必须在同一笔改动里改 docstring。
理由：过期的 docstring 比过期的注释危险一个量级——注释被当随口备注，docstring 会被 `help()` 和 IDE 悬浮当权威说明。

**② K / 代码量的分母只用「净代码」。** `K_lines_nb`（tokenize 排除空行、注释、独占语句的 STRING）是唯一口径。
空行 / 注释 / docstring 的增减**一律不计入成果**。同一份文件物理行 202 / `K_lines` 166 / `K_lines_nb` 84，差 2.4 倍。
话术分界：净代码降 → 可写「简化 X 行」；物理行降而 K 不动 → 只能写「搬家 X 行」。

**③ 静态引用计数必须包含同文件内的调用。** 已在此栽过三次：grep「零引用」时排除了本文件，
把 `_sha`（本文件内 3 次）、`run_pristine_tests`（2 次）、`_jaccard`（2 次）误判成死代码。
**grep 会误判、AST 会误判，只有「关掉跑一遍」（消融）是硬证据。**

## 三条判定纪律

1. **顺序不可换**：泄漏筛查在评测**之前**。测完再筛，虚高分已污染历史。
2. **δ 是测出来的，不是选的**：同一版本重复评测评散布，只升不降。调低 δ = 用噪声换分数。
3. **非补偿**：分数 / 成本 / 泄漏是「与」关系，不能用高分补成本超支。

## 论文侧纪律

- **主指标用 strict pass@1**：分母 = 任务数 × k；`AgentTimeoutError` / `VerifierTimeoutError` / infra / 缺 trial 全部计 0。valid-only 只作敏感性分析（依据：AHE §4.1、RRSI coding README、Terminal-Bench 官方口径）。
- **先定表格再采数据**。表格里没有的列，采了也是废数据。
- **措辞不超证据**。「没测出影响」≠「没用」；「我们更低」必须带 n 和区间。
- **任何数字都要能追到一条命令**。
- 「只有 / 全都 / 唯一 / 没有一篇」开头的句子，必须附穷举清单。
- 说「某方法缺少 X」之前，必须搜过正文 + 全部附录。

## Skill 使用约定（2026-11-04 复核后）

用户级 Skill 只留 4 个：`code-symbols`（ast-grep 已装 0.51.1，做结构化查找/改名/删函数）、
`bug-review`、`deep-research`、`self-improving-agent`。
插件 7 个：`superpowers` `research-copilot` `architecture-visualization` `Presentations`
`grillme-matt` `office-docx`（Word 交付）`qoder-context`。

- 写/改代码：`superpowers:test-driven-development` → `superpowers:systematic-debugging` → `superpowers:verification-before-completion`
- 大规模删函数/改符号：`code-symbols`（**必须含同文件内调用**，见纪律 ③）
- 论文/文献/引用核查：`research-copilot`；泛技术调研：`deep-research`
- 架构图/流程图：`architecture-visualization`；组会答辩 PPT：`Presentations`
- 给老师的 Word 交付：`office-docx`
- 需求不清：`grillme-matt:grill-with-docs` / `to-spec`；实现阶段切回 `superpowers`
- `self-improving-agent` 的记录**写到 auto-memory，不要在项目根新建 `.learnings/`**——
  本项目已有 auto-memory + `D:\智能体经验集\` 两套记录，不要长第三套。

已删除且**不要再装回来**：`fix-code`（流程委托给不存在的 `.claude/rules/`，且主张 Grep-first，
与纪律 ③ 相反）、`python-programmer`（要求 no for-loops / 79 字符 / ruff+uv / 详尽文档，
与 `evo/` 现状 212 个 for 循环、537 行超 79 字符、以及纪律 ① 全面冲突）。

## 不要做的事

- 不要 `git add -f` 任何被 gitignore 的目录（`external/` `runs/` `论文准备/` `.tmp/` `.venv/`）。
- 不要把临时文件写到项目根。已出现过的违规残留：`tmp1t4xfo7k/`、`pytest-of-35174/`。
- 不要为了「代码质量」做风格重写。目标是**净代码行下降且行为对等**，不是符合某个通用风格规范。
- 不要在 n<10 的样本上宣布优化有效。
