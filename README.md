# evo

面向 LLM-agent harness 的外层研发实验工具：**冻结模型权重，只改 harness**（提示、工具、控制流、记忆、上下文管理），并用可复现的评测流程验证改动是否真的更好。

> 当前工具**不依赖真实用户反馈**；文中的“负反馈/失败信号”指评测失败轨迹或失败样本。用户反馈闭环是远期研究目标，不是现有输入。

## 它做什么

- 评测 baseline / 候选 harness 的严格 pass@1、token、K（净代码行）与 holdout；
- 按“监测 ∥ 架构 → 总结 → 执行 → 审核”的五岗位流程跑一轮或多轮演化；
- 在评测前做泄漏筛查；在评测后用 paired bootstrap、噪声地板、K 门、holdout 与成本规则决定是否采纳；
- 把每轮的 plan、diff、verify、review、evaluation、effect 与 ledger 落盘，支持回放和归因；
- 删除/简化也走同一套验证与回退流程，不只比较分数。

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m evo --help
```

Linux/macOS 把 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`。

## 外部依赖与下载入口

本仓库不包含 `external/` 下的论文、数据集和第三方仓库；按下面入口自行拉取。

### 评测底座与数据

| 资源 | 用途 | 入口 |
|---|---|---|
| Harbor | Terminal-Bench 执行底座、数据集/agent 适配 | https://github.com/laude-institute/harbor |
| Harbor datasets | 任务包 | https://github.com/laude-institute/harbor-datasets |
| Terminal-Bench | Terminal-Bench 2.0/2.1 任务与评测 | https://github.com/laude-institute/terminal-bench |
| Terminal-Bench 2.1 | 89 题评测集（经 Harbor 拉取） | https://github.com/harbor-framework/terminal-bench-2-1 |
| SWE-bench / Verified | 真实仓库修复评测 | https://github.com/SWE-bench/SWE-bench |
| Harvey LAB | 文档/知识工作 agent 评测 | https://github.com/harveyai/harvey-labs |
| JobBench | 跨职业任务评测 | https://github.com/Job-Bench/job-bench-eval |
| GDPval | 知识工作评测 | https://openai.com/index/gdpval/ |
| APEX-Agents | 专业人士任务评测 | https://www.mercor.com/apex/apex-agents-leaderboard/ |
| EngDesign | 工程设计评测 | https://github.com/AGI4Engineering/EngDesign |
| Frontier-Eng | OOD 工程设计评测 | https://github.com/Einsia/Frontier-Engineering |
| MLE-bench | Kaggle/ML 工程评测 | https://github.com/openai/mle-bench |
| RE-Bench | ML 研究工程评测 | https://github.com/METR/RE-Bench |
| ScienceAgentBench | 科学 agent 评测 | https://github.com/OSU-NLP-Group/ScienceAgentBench |

### 被测 agent / 运行环境

| 资源 | 用途 | 入口 |
|---|---|---|
| Terminus-2 | 官方 Terminal-Bench 基准 agent（经 Harbor） | https://github.com/laude-institute/harbor |
| OpenHands | 强基线 | https://github.com/All-Hands-AI/OpenHands |
| mini-swe-agent | 弱基线 / 消融对照 | https://github.com/SWE-agent/mini-swe-agent |
| Codex CLI | 部分参考实现的执行壳 | https://github.com/openai/codex |
| DeepSeek API | 被测 policy / 编辑模型 | https://api-docs.deepseek.com/ |
| Docker | 沙箱与 Harbor 运行 | https://docs.docker.com/get-docker/ |

中国网络下可直接 GitHub 可能很慢；本机实测可用 `gh-proxy.com` 拉归档包：

```bash
curl -L --max-time 1800 \
  "https://gh-proxy.com/https://github.com/<owner>/<repo>/archive/refs/heads/main.zip" \
  -o repo.zip
```

## 直接对标论文与代码

| 论文 | arXiv | 代码 | 许可证 | 我们借鉴/对照的点 |
|---|---|---|---|---|
| RRSI: Regularized Recursive Self-Improvement of Agent Harnesses | https://arxiv.org/abs/2609.24972 | https://github.com/google-research/rrsi | Apache-2.0 | 退火编辑预算、历史条件化、评前 critic、噪声地板、成本规则、剪枝 |
| ModularRSI | https://arxiv.org/abs/2609.14857 | https://github.com/IQuestLab/ModularRSI | CC BY-NC 4.0（原创）+ Apache-2.0（Harbor 派生） | 5 模块独立演化、模块级信用分配、合并复测 |
| HarnessPro | https://arxiv.org/abs/2610.26770 | https://github.com/pzs19/HarnessPro | MIT | lite/medium/heavy 三档、性能—成本 Pareto；仓库只含产物+评测 |
| Meta-Harness | https://arxiv.org/abs/2603.28052 | https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact | 无 LICENSE | 文件系统即记忆；只读设计，不复制代码 |
| RHO: Retrospective Harness Optimization | https://arxiv.org/abs/2606.05922 | https://github.com/wbopan/retro-harness | MIT | 无标签轨迹、DPP 核心集、自校验/自一致、成对自偏好 |
| AHE: Agentic Harness Engineering | https://arxiv.org/abs/2604.25850 | https://github.com/china-qijizhifeng/agentic-harness-engineering | MIT | 组件/经验/决策三层可观测、预测—证伪 |
| TTHE: Test-Time Harness Evolution | https://arxiv.org/abs/2607.08124 | https://github.com/junnie00/TTHE | MIT | 测试流无标签 batch、G 分支 × R 轮；测试时自适应有泄漏风险 |
| HarnessX | https://arxiv.org/abs/2606.14249 | https://github.com/Darwin-Agent/HarnessX | MIT（疑似官方） | 类型化原语；其轨迹转训练信号路径不适用本课题 |
| HarnessDev | https://arxiv.org/abs/2609.01437 | 未找到公开代码；项目页 https://self-developing-agents.github.io/ | — | 执行普查、held-out 方向一致率 |
| DGM: Darwin Gödel Machine | https://arxiv.org/abs/2505.22954 | https://github.com/jennyzzt/dgm | Apache-2.0 | archive/谱系/垫脚石；必须加 K 压力防膨胀 |
| SICA: Self-Improving Coding Agent | https://arxiv.org/abs/2504.15228 | https://github.com/MaximeRobeyns/self_improving_coding_agent | MIT | 最小自编辑闭环、负结果披露 |

## 相邻参考论文与代码

| 工作 | 主题 | 入口 | 许可证/备注 |
|---|---|---|---|
| ADAS | 自动设计 agent 代码 | https://github.com/ShengranHu/ADAS | Apache-2.0 |
| STOP | 自改进优化器 | https://github.com/microsoft/stop | MIT |
| GEPA | 反思式 prompt/代码进化、逐实例 Pareto | https://github.com/gepa-ai/gepa | MIT |
| Trace | 工作流 DAG 归因与优化 | https://github.com/microsoft/Trace | MIT |
| TextGrad | 文本梯度、critic/optimizer 分角色 | https://github.com/zou-group/textgrad | MIT |
| Gödel Agent | 自指 agent、改坏回退 | https://github.com/Arvid-pku/Godel_Agent | MIT |
| AlphaEvolve | 级联评估、进化数据库 | https://github.com/google-deepmind/alphaevolve | Apache-2.0（最小实现） |
| AI Scientist v1 | 自动科研四阶段、语义查重 | https://github.com/SakanaAI/AI-Scientist | 自定义 Responsible-AI 许可证，非 OSI |
| AI Scientist v2 | 树搜索实验节点 | https://github.com/SakanaAI/AI-Scientist-v2 | 自定义 Responsible-AI 许可证，非 OSI |
| Agent Laboratory | 角色分工、checkpoint、人在环 | https://github.com/SamuelSchmidgall/AgentLaboratory | MIT |
| MLAgentBench | 13 个 ML 任务、预注册阈值 | https://github.com/snap-stanford/MLAgentBench | MIT |
| MLE-bench | 75 个 Kaggle 竞赛、奖牌率 | https://github.com/openai/mle-bench | MIT |
| RE-Bench | 8 小时同预算人机对照 | https://github.com/METR/RE-Bench | MIT；仓库已 archived |
| ScienceAgentBench | 成功率+成本+rubric | https://github.com/OSU-NLP-Group/ScienceAgentBench | MIT |
| Reflexion | 文本反思记忆 | https://github.com/noahshinn/reflexion | MIT |
| ExpeL | 跨任务 insight 抽取 | https://github.com/LeapLabTHU/ExpeL | Apache-2.0 |
| AWM | 抽象 workflow 记忆 | https://github.com/zorazrw/agent-workflow-memory | Apache-2.0 |
| ACE | playbook 条目增量策展 | https://github.com/ace-agent/ace | 许可证以仓库为准 |

## 我们实际搬进本仓库的零件

- **RRSI**：critic 的确定性 precheck + LLM 语义审查；编辑预算、历史条件化、噪声地板、成本规则。
- **RHO**：纯 Python DPP 核心集；无标签选题与自偏好（只做选题/诊断，不替代官方 verifier）。
- **AHE**：预测—证伪字段进入 history，提案者下一轮可读。
- **HarnessDev**：feedback vs held-out 方向一致率，只写诊断，不进 `gate()`。
- **暂缓**：GEPA 的 Pareto 档案（当前只演化 incumbent，没有父代采样消费者）、AlphaEvolve 的级联评估、ACE 的增量 playbook；等有真实调用点再上。

## 许可证与复用边界

- Apache-2.0 / MIT 的代码可以读、可以搬，但保留版权与许可；
- **Meta-Harness 没有 LICENSE，默认保留所有权利，只能读设计，不能复制代码**；
- AI Scientist v1/v2 使用自定义 Responsible-AI 许可证，不是 OSI 开源；只读设计；
- ModularRSI 原创部分 CC BY-NC 4.0，不可商业复用；
- DS-Agent 无 LICENSE、InternAgent 为 Other；不要直接复制；
- 本仓库不包含 `external/` 下的第三方代码与数据；引用前请回各自 LICENSE 核对。

## 目录

```text
evo/
  adapters/     外部接入：DeepSeek、Harbor / Terminal-Bench、traces、coverage、git worktree
  application/  用例编排：baseline、workforce、evolve、simplify、round、compare、report
  domain/       判据与度量：gate、K、critic、coreset、prediction、stats、ledger
  contracts/    数据契约
tests/          单元与回归测试
specs/          Terminus-2 评测规格
docs/           架构、协议、修复计划
论文准备/        调研报告与组会材料
```

`external/`、`.venv/`、`.tmp/` 等本地资产不入库。

## 当前状态

- 891 个测试通过（49 个测试文件）。
- 支持 H0 / SIM / EVO / COMBO / PLACEBO / SHARP 臂的评测编排。
- 评测底座通过适配层接入，不把 Harbor、Terminal-Bench 或被测 agent 的代码复制进本仓库。
- 尚未完成真实全量 pilot；Pareto 档案、用户反馈到 failcase 的冻结回放仍在路线图中。

## 设计约束

- 不训练权重；不引入 numpy/pandas 等通用统计库，统计口径由仓库内纯 Python 实现。
- 评测缺失、token 缺失、agent error 与成本缺失均 fail-closed，不按 0 冒充。
- 一次只改一处；默认关闭新机制；安全类检查优先于评测。
