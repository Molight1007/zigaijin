# 架构 v2：代码简化为第一贡献

> **已被 `architecture_v3.md` 取代（2026-11-06）。本文保留为历史设计稿。**

2026-11-04 ｜ 状态：设计稿，未写代码 ｜ 依据：`论文准备/09/10/11/12/15/22/23/24/27`

---

## 0. 主张与边界

### 主张（论文要证明的一句话）

> **自演化只会长；把删除做成一等演化算子，用统计准入 + 行为回放证明：
> 删掉 X% 的 harness 代码，S 不劣、holdout 不劣、可归因性上升，
> 且随机删等量代码做不到。**

### 不是主张

| 不主张 | 原因 |
|---|---|
| 「我们加了 K 轴」 | K 是自证尺子（09 号文档）；K 只作剂量 |
| 「我们第一个压缩 agent 代码」 | SHarP/Cortex 已证明冗余存在 |
| 「删得越多越好」 | SHarP 反例：删终止判定后 token 反升 |
| 「行为等价」 | 领域共识：只能说「未观测到退化」 |
| 「省成本」 | ACON：成本 −13.9%，延迟 +19.7% |

### 角色分工

```
代码简化 = 干预（treatment）
K        = 剂量（dose），不是疗效
S / holdout / 可归因性 = 疗效（outcome）
统计准入 + 行为回放   = 安全机制（safety）
演化     = 底座（substrate），不是主贡献
```

---

## 1. 五问 → 五条能力

| RQ | 问题 | 能力 | 主要产物 |
|---|---|---|---|
| **RQ1** | 这个 harness 到底有多少可删质量？ | `prestudy`：静态 + 触达 + 抽样消融 | `prestudy.json`；换不换对象的决策 |
| **RQ2** | 删了会不会掉分？ | `simplify` + `gate`：非劣检验 + K 互锁 + token 门 | 删测证据表；主结果表 |
| **RQ3** | 删了除了短，还有什么用？ | `diagnose`：故障定位 top-1/top-3 | 可归因性表 |
| **RQ4** | 是不是「更保守」带来的？ | `compare`：placebo + SHarP 模块消融 | 基线/安慰剂表 |
| **RQ5**（可选） | 简化让演化更省/更准吗？ | `evolve` + `simplify` 组合臂 | 每轮 token / 提案命中率 |

**架构纪律**：每个模块必须服务一个 RQ 或一张论文表；服务不了的，砍掉。

---

## 2. v1 → v2 变更

| # | v1 | v2 | 原因 |
|---|---|---|---|
| 1 | 演化是主干，简化是一轮里的一个候选 | 简化是独立 arm；演化是底座 | 创新点必须有主实验 |
| 2 | A10 只做静态扫描（L1） | 补 L2 触达、L3 消融；L1 只作地基 | L1 预期效应为 0（23 号文档） |
| 3 | 没有预研 | `prestudy` + 决策门 | 先确认对象有可删质量 |
| 4 | 没有可归因性端点 | `diagnose` 基准 | 给简化一个独立于 K 的收益 |
| 5 | 没有 placebo / SHarP 基线 | `compare` 两个对照臂 | 回答「是不是更保守」 |
| 6 | K 门用点估计 | CI 下界 + 预注册非劣界 | n=180 时点估计不可判 |
| 7 | `K_lines_nb` 可被压成一行套利 | `K_ast` 必降 + `K_lines_nb` 必降 + `K_chars` 不升 | AST 节点数对格式免疫 |
| 8 | 代码 K 与文本 K 混算 | 代码 K / 文本 K 分开报、分开判 | 一行代码 ≠ 一行中文 |
| 9 | 组件由适配器自报 | 边界由目录/文件约定，适配器只声明边界 | K 分解必须可复算 |
| 10 | 没有 AST 白名单 | `protection`：assert/raise/异常/装饰器/公开签名 | REFINE：assert 只保 57.1% |
| 11 | 没有差分/重放 | `verify`：编译 + 冒烟 + 失败池回放 + 差分重放 | 判断不能替代测量 |
| 12 | holdout 只做诊断 | 简化通道必须过 holdout 非劣 | 防在演化池上过拟合 |
| 13 | DPP coreset / 模块级演化 / RRSI 全量复现 | 砍掉，留阶段二 | 不服务简化主张，避免膨胀 |
| 14 | 缺失记 0 | 写死口径：infra 缺失判无效；token 覆盖不足判无效 | 不能拿 0 冒充实测 |

---

## 3. 分层与模块

```
contracts/   C01 types（Task/Trial/Score/KVector/Component/Candidate/Decision/Arm）
             C02 ports（Subject/Evaluator/LLM/Store/Worktree/Coverage/Clock）
domain/      D01 kmetrics   D02 components  D03 census    D04 candidates
             D05 protection D06 verify      D07 stats     D08 gate
             D09 diagnosis  D10 ledger      D11 protocol
application/ A01 prestudy   A02 baseline    A03 arm       A04 simplify
             A05 evolve     A06 diagnose    A07 compare   A08 report
             A09 verify_evidence
adapters/    AD01 harbor_tb21  AD02 terminus2  AD03 deepseek  AD04 git_worktree
             AD05 store_fs      AD06 proc      AD07 coverage_py
             AD08 openhands（阶段二，stub）
infra/       I01 config  I02 runtime（logging/redact/ids/hash）  I03 preflight
cli.py       `python -m evo ...`
```

### 模块职责（只列会写代码的）

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `D01 kmetrics` | K_lines_nb / K_ast / K_chars / K_cc / K_bytes / K_gzip；按组件、按函数 | 只量被测 harness；工具版本冻结 |
| `D02 components` | 从 `BoundarySpec` 派生组件；稳定 ID | 适配器只声明边界，不提供组件列表 |
| `D03 census` | 静态引用 + coverage 触达；`unit → touches/tasks` | 文件级 trace 不算行级触达 |
| `D04 candidates` | L1 静态、L2 零触达、L3 消融、L4 安全改写；带证据与置信度 | detector 只说「可能」，gate 才说「可信」 |
| `D05 protection` | AST 白名单 + 冻结补丁 + 公开签名 + 入口 | 硬编码，不靠提示 |
| `D06 verify` | 编译/导入、冒烟、失败池回放、差分重放；三态 | `unknown` 不能当 `pass` |
| `D07 stats` | 配对 bootstrap、TOST、MDE、每任务最差 | 先算 MDE，再定 n |
| `D08 gate` | 有效性 → 非劣 → K → token → 保护 → T4 → Pareto | 简化门与演化门分开 |
| `D09 diagnosis` | 故障定位：top-1/top-3；二分类归因 | 控制组件列表长度 |
| `D10 ledger` | 只追加 JSONL；四态 + 理由 | 没测过 ≠ 被拒过 |
| `D11 protocol` | 分池、冻结、预注册、种子、哈希 | 先写表头，再跑实验 |
| `A01 prestudy` | RQ1；输出可删质量与决策 | 决策门不过，停 |
| `A02 baseline` | H0：K + 评测 + 冻结 | 基线无效即阻塞 |
| `A03 arm` | 一条流水线跑所有 arm | 同评测协议，禁止 per-arm 特判 |
| `A04 simplify` | 候选 → 验证 → 评测 → 门 → 账本 | 一次只删一处 |
| `A05 evolve` | RRSI 式提案/评测/选择（底座） | 只作对照与组合，不抢主贡献 |
| `A06 diagnose` | RQ3：故障集 → 预测 → 打分 | 固定模型、提示、温度、组件描述 |
| `A07 compare` | placebo + SHarP 模块消融 | 同题池、同 k、同门 |
| `A08 report` | 六张论文表 + S-K 散点 | 每个数字追到命令 |

---

## 4. 统一实验臂

**一条流水线，arm 只是配置**，保证同题池、同 k、同门、同统计：

```python
Arm = {
  "evolve":   "off | on",
  "simplify": "off | static | census | ablate | full | random | sharp",
  "order":    "evolve_then_simplify",   # 预注册，不事后改
  "diagnose": "off | on",
}
```

| Arm | evolve | simplify | 回答什么 |
|---|---|---|---|
| **H0** | off | off | 基线 |
| **EVO** | on | off | 自演化会不会长（复现 HarnessDev 的膨胀） |
| **SIM** | off | full（L1+L2+L3） | 简化本身是否保分、降 K |
| **COMBO** | on | full | 组合是否落在 S-K 左上象限 |
| **PLACEBO** | off | random（等量 K） | 收益是不是「更保守」 |
| **SHARP** | off | sharp（模块消融） | 与 SHarP 同层对手比 |

**主表**：H0 / EVO / SIM / COMBO。**对照表**：PLACEBO / SHARP。
**子消融**（小规模，不单独跑全量）：SIM-static、SIM-census。

---

## 5. 简化通道

### 5.1 四层，顺序不能换

| 层 | 判据 | 动作 | 预期效应 | 成本 |
|---|---|---|---|---|
| **L1 静态卫生** | 零引用 / 未用导入 / 不可达 | 删 | **0**（本来就不执行） | 极低 |
| **L2 零触达** | 跑 N 次从未执行 | 删 | 小~中 | 中 |
| **L3 零贡献** | 执行了，关掉后 S 不掉、token 不升 | **先关开关**，再决定是否删调用点 | **大** | 高（1+m 次） |
| **L4 安全改写** | 具名算子 + 等价句 + 差分 100% | 改写 | 可读性 | 中高 |

**L1 只作地基，不作主结果。** L3 是主战场；L4 只在 L1–L3 做干净后开启，且只收两个低风险算子。

### 5.2 候选 schema

```json
{
  "id": "c001",
  "layer": "L2",
  "component": "terminus2:agent.py::_unwind_messages_to_free_tokens",
  "evidence": {"static_refs": 0, "touches": 0, "tasks_touched": 0},
  "expected": {"dK_lines": -12, "dK_ast": -20, "dK_chars": -340},
  "protection": {"whitelist_hits": [], "frozen_patch": false},
  "diff": "candidates/c001/diff.patch",
  "verify": {"compile": "pass", "smoke": "pass", "replay": "pass", "differential": "unknown"},
  "eval": {"S": 0.71, "S_ci": [0.64, 0.78], "tokens": 12345, "missing": 0.0},
  "decision": {"verdict": "accept", "reason": "all gates pass; dK_ast=-3.1%"}
}
```

### 5.3 保护（硬编码，不靠提示）

| 保护类 | 规则 | 依据 |
|---|---|---|
| `assert` / `raise` / `fail` | AST 遍历直接跳过 | REFINE：只保 57.1% |
| `try/except/finally` | 禁止删除整个结构 | REFINE：异常结构 92–96% |
| 装饰器 / 框架钩子 | 禁止删除 | REFINE：90.4–94.4% |
| 公开签名 / `__all__` | 禁止改 | REFINE：86.2–94.2% |
| CLI 命令 / 入口 | 禁止删 | 框架契约 |
| 冻结兼容补丁 | 排除出候选集 | Terminus-2 的 MODEL_CONNECTION 补丁 |
| T4 ν 组件 | `n_prune` 轮内不得删 | 硬线 T4 |

### 5.4 验证

```
编译/导入 → 冒烟 2 题 → 失败池回放 10/10 → 差分重放（有记录输入才做）
```

- 三态：`pass / fail / unknown`；`unknown` 不能支撑 L3/L4 的接受。
- 差分重放：从轨迹里取被改函数的真实输入，旧/新实现逐条比对；没有输入记录就标 `unknown`，不标 `pass`。
- 删除后必须能 `import`；删函数导致调用点残留 → `undeletable`，不进分母。

### 5.5 演化闭环（EVO / COMBO）

`evo loop` 是演化臂的完整闭环，一条命令跑完搜索、判定与 holdout 复核：

1. **单 incumbent，多候选**：每轮从 incumbent 复制出 N 个候选目录，候选之间互不影响。
2. **先筛后评**（省预算）：`propose → verify（编译/导入）→ K 门（仅 simplify != off）→ 评估`。任一步失败记 `gate_failure`，不进测量、不占评估预算。
3. **历史条件提案**（RRSI / Meta-Harness 做法）：提案端拿到轮次、退火后的编辑预算、历史（accepted / delta_S / component）、探索配额、剪枝集、S* 与 incumbent 分数；不是每轮从零重写提示。
4. **判定复用 `evolve_round`**：gate_failure → invalid；无分数 → floor S*−δ；否则走成本规则（增益换 token 预算 / 成形分数）。胜者按 `(-score, tokens, k_lines, variant)` 选出。
5. **目录换位**：胜者整目录替换 incumbent，下一轮从新 incumbent 出发。
6. **holdout 只在最后跑一次**：搜索期间绝不触碰 holdout；结束后在最终 incumbent 上跑 `--holdout-pool`，写进 `loop.json.final.holdout`。
7. **产物可判定**：`final-outcomes.jsonl` 是最终 incumbent 在演化池 + holdout 上的原始 trials，直接喂给 `evo compare` 出非劣 / holdout / K / token 四块；简化臂的逐候选判定仍走 `evo round` 的九门。

`evo loop` 只接受 `evolve=on` 的臂（**EVO / COMBO**）；SIM / PLACEBO / SHARP 预注册为 `evolve=off`，走 `evo scan` / `evo apply` + `evo round`，不许混用。`requires_k` 只在 `simplify != off` 的臂为真：COMBO 的候选必须先过 K 门，EVO 不因 K 被拦（它的主张不是降 K）。

---

## 6. 判定

### 6.1 有效性（先于一切）

```
missing_frac ≤ 0.10
infra_frac   ≤ 0.10
token_coverage ≥ 0.90
每个任务 k 次齐全（或按预注册的缺失口径）
```

任一不满足 → `invalid_measurement`，不产生判定。

**缺失口径写死**：infra 缺失 = 无效测量，不是 0 分；token 缺失 = 该臂 token 不可比，token 门关闭（不能用部分和）。

### 6.2 非劣（简化门）

设参考臂 `ref`：

- SIM：`ref = H0`
- COMBO：`ref = 当前演化 incumbent`
- EVO：`ref = S*`（RRSI 规则）

```
ΔS = S_cand − S_ref
CI = paired_bootstrap(ΔS, 10_000)
接受条件：CI_low ≥ −m
```

`m` 是**预注册的非劣界**，不是 δ。默认 `m = 0.10`（10pp），理由：n=180 时 CI 半宽约 7.3pp，1.7pp 的界不可判。
若审稿要求 5pp，则 n 必须增加到 ~800 obs/arm；若坚持 1.7pp，n 需要 ~7,000 obs/arm——不现实。

**holdout 同门**：`CI_low_holdout ≥ −m_h`，`m_h` 预注册。

### 6.3 K 门（代码）

一个代码候选算「简化」，必须同时满足：

```
K_ast_after     ≤ K_ast_before     × (1 − 0.005)   # 必降，防压行/改名
K_lines_nb_after ≤ K_lines_nb_before × (1 − 0.005)  # 必降，主口径
K_lines_nb_before − K_lines_nb_after ≥ 10           # 至少 10 行
K_chars_after   ≤ K_chars_before                    # 不升
K_cc_after      ≤ K_cc_before                       # 不升
```

`K_ast` 是**必降项**，不是可选交叉校验：它对换行、分号压行、改名、删注释都免疫。
文本候选走另一套：`K_tokens` 降 ≥0.5% 且 `K_gzip` 不升。

### 6.4 token 门

```
tokens_total_after ≤ tokens_total_before
每个任务 tokens_after ≤ tokens_before × 1.10   # 单题允许 10% 抖动，报最差
```

总量严格不升；单题允许 10% 抖动但必须报 worst-task。SHarP 的反例说明：删了控制代码，token 可能反升。

### 6.5 其他门

| 门 | 规则 |
|---|---|
| 保护 | 白名单命中 → 拒；冻结补丁 → 拒；公开签名改动 → 拒 |
| 行为 | `verify` 任一 `fail` → 拒；L3/L4 遇 `unknown` → 拒 |
| T4 | `protection_until > t` 的组件 → 拒 |
| Pareto | `ΔS ≤ 0` 且 `ΔK ≥ 0` → 拒 |
| 每任务最差 | `worst_task_ΔS < −m_task` → 拒（`m_task` 预注册） |

### 6.6 选择

- 简化臂：过门者取 `ΔK_ast` 最大；平手取 token 低、CC 低。
- 演化臂：过门者取 `S` 最大；平手取 K 低、C 低、CC 低。
- 一次只接受一个候选；其余进候选池，不重复生成。

### 6.7 统计族与功效

- **族错误率**：`evo compare --family-size N` 对 α 做 Bonferroni（`α/N`），CI 与判定都用校正后的 α；`aggregate.family` 记录 `size/alpha/method`。多臂同表比较不再各自用 0.05 裸跑。
- **p 值表**：`evo.domain.stats.holm` 提供 Holm step-down（族内单调校正 + reject 掩码），用于需要报 p 值的表。
- **功效**：`evo power --reference ... --candidate ... --tasks ...` 输出 `n_pairs / mean_delta / sd / mde / required_pairs / enough`；`enough = mde <= margin`。不达标先加题或加 k，不得把「没显著下降」当非劣证据。

---

## 7. K 口径

| 指标 | 定义 | 角色 | 防什么 |
|---|---|---|---|
| **K_lines_nb** | tokenize 排除空行/注释/docstring 后的净可执行行 | **主口径（报告）** | 注释/空行刷分 |
| **K_ast** | AST 节点数（排除 docstring 常量） | **必降项（门）** | 压成一行、改名、删注释 |
| **K_chars** | UTF-8 字符数 | sanity（不升） | 体量悄悄变大 |
| **K_cc** | 圈复杂度总和（McCabe） | 交叉校验（不升） | 行少但分支多 |
| **K_bytes** | UTF-8 字节数 | sanity | 编码/体量异常 |
| **K_gzip** | 排序拼接后 gzip 字节（mtime=0） | sanity，不作主判据 | 冗余密度参考 |
| **K_tokens** | 文本组件 token 数 | 文本主口径 | 中文/英文提示 |
| **K_gzip_text** | 文本 gzip 字节 | 文本 sanity | 重复表述 |

**代码 K 与文本 K 分开报、分开判**：一行 Python 和一行中文提示不是同一个量；不合成一个标量。

**组件边界**：`BoundarySpec` 由目录/文件约定给出；适配器只声明「哪些目录属于哪个组件」，不提供组件列表。
换边界 = 换口径 = 换一次实验，写进 `run.json`。

**K 的可复算性**：冻结 `radon` 版本、`tokenize` 版本、gzip 级别、注释识别规则、组件切分规则；`K` 只量静态交付物，运行时生成的 prompt/schema 按其生成代码计入，不按运行时实例计入。

---

## 8. 可归因性基准（RQ3）

### 8.1 为什么需要

K 降了、S 没降，是**非劣**，不是收益。审稿人会问「so what」。
HarnessMinimizer 给了唯一现成的独立收益支点：精简后 LLM 归因准确率 **28% → 44–51%**。
我们在 agent harness 上复现它。

### 8.2 设计

| 项 | 设计 |
|---|---|
| **故障集 A（主）** | 注入故障：选行为相关组件，注入 1 处变异（翻条件/改常量/删重试/换分支），跑到失败；注入点即 ground truth |
| **故障集 B（辅）** | 自然失败：失败池里 10/10 可复现的失败；二人标注 + 仲裁 |
| **规模** | A：100 个故障（20 组件 × 5 变异）；B：30 个自然失败 |
| **提问** | 给失败轨迹 + 组件清单（名称 + 一行描述 + K），问 top-3 责任组件 |
| **打分** | top-1 / top-3 准确率；二分类「harness vs 环境」 |
| **控制** | ① 组件清单长度控制：随机抽取等长清单；② 顺序打乱；③ 固定模型/提示/温度；④ 同一故障集 before/after |
| **重复** | 每故障 3 次，取多数 |

### 8.3 必须报告的两个混杂

1. **清单变短本身会让定位更容易**——必须报「等长随机清单」的对照。
2. **注入故障可能比自然故障简单**——A/B 两集分开报，不合并。

---

## 9. 预研决策门（RQ1）

### 9.1 三步

| 步 | 做什么 | 成本 | 产物 |
|---|---|---|---|
| **P1 静态** | L1 扫描 + 组件切分 + K 分解 | 零模型，分钟级 | `prestudy.static.json` |
| **P2 触达** | 10 题 × 1 rep + coverage；统计 L2 | 小 | `prestudy.census.json` |
| **P3 抽样消融** | 5–10 个大组件 × 10 题 × k=3；估 L3 yield | 150–300 trials | `prestudy.ablate.json` |

### 9.2 决策门（预注册，不事后改）

```
若 L3_yield < 20% 且 (L1_K% + L2_K%) < 1%：
    → 换对象（OpenHands/worktree）或把 prompt/skill 文本纳入
否则：
    → 进入主实验
```

同时估 `trials_per_1pct_K`：如果每降 1% K 要 > 500 trials，缩减候选集或改题集。

**这一步不做，后面所有实验都可能是在一个没有油水的对象上空转。**

---

## 10. 产物与论文表

```
runs/<id>/
  run.json                 # 冻结配置、subject hash、K spec、池、种子、版本
  state.json               # 状态机
  events.jsonl             # 事件
  ledger.jsonl             # 只追加判定
  prestudy.json            # RQ1 + 决策
  rounds/<nn>/
    prereg.json            # arm / 假设 / 主端点 / 非劣界 / n / k / 停止规则
    candidates/<cid>/      # candidate.json / diff.patch / k_*.json / verify.json / eval.json / decision.json
  diagnosis/
    failures.jsonl         # 故障集
    ground_truth.jsonl     # 注入/标注
    predictions.jsonl      # 模型 top-3
    score.json             # top-1/top-3
  loop/                    # 演化臂（--out runs/<id>/loop）
    round-<nn>.json        # 每轮 evolve_round 判定 + 候选筛选元数据
    history.jsonl          # 提案端使用的历史（accepted / delta_S / component）
    final-outcomes.jsonl   # 最终 incumbent 的演化池 + holdout 原始 trials
    loop.json              # spec / reference / rounds / final(score, holdout, subject_hash, K)
  tables/
    t1_subject.csv
    t2_arms.csv
    t3_deletions.csv
    t4_baselines.csv
    t5_diagnosis.csv
    t6_cost.csv
```

| 表 | 内容 | 命令 |
|---|---|---|
| T1 | 对象、组件、K 分解、L1/L2/L3 可删质量 | `evo prestudy` |
| T2 | 主 arms：S/CI、K、token、诊断 | `evo run --arm ...` / `evo loop` + `evo compare` |
| T3 | 逐候选删测证据（四态 + 理由） | `evo report` |
| T4 | placebo / SHarP / static 对照 | `evo compare` |
| T5 | 可归因性 before/after | `evo diagnose` |
| T6 | 每降 1% K 的 trials/tokens/wall-clock | `evo report --cost` |

**头图**：S–K 散点。EVO 右上（S↑K↑），SIM 左（S 平 K↓），COMBO 左上（S↑K↓）。

---

## 11. 状态机

```
created
  → prestudy_pending → prestudy_running → prestudy_done
  → baseline_pending → baseline_running → baseline_done
  → arm_pending → arm_running → arm_done
  → diagnosis_pending → diagnosis_running → diagnosis_done
  → report_pending → report_done

阻塞态：blocked_subject / blocked_docker / invalid_measurement / crashed
```

- 每步先写 artifact，再改 state；`state.json` + 文件锁保证幂等恢复。
- `blocked_subject` 只能由人解除（换对象/加文本），代码不自动降级。
- `invalid_measurement` 必须保留原始日志，不能静默重跑覆盖。

---

## 12. 明确不做

| 不做 | 原因 |
|---|---|
| DPP coreset（RHO） | 不服务简化主张；阶段二 |
| 模块级独立演化/合并（ModularRSI） | 只保留 SHarP 式模块消融作基线 |
| RRSI 全量复现 | 只做同协议薄底座；数字不跨底座比 |
| OpenHands / Harvey / EngDesign 适配 | 阶段二；由 prestudy 决策门触发 |
| L4 自由重构 | REFINE 证明等价性保证不了；只收两个低风险算子 |
| 训练/微调、造数据集、GUI、分布式 | 课题前提与论文无关 |
| 把 K 合成一个标量 | 代码与文本不可比；加权会被套利 |
| 用「没显著下降」代替「非劣」 | 缺功效的零结果不是证据 |

**机制必要性判定（2026-11-04）**：

| 机制 | 判定 | 理由 |
|---|---|---|
| 演化闭环（提案→筛选→测量→判定→换位） | **必要，已实现** | 没有闭环就只是「扫描 + 删除」，EVO/COMBO 无从比较 |
| L3 运行时消融（`evo ablate-run`） | **必要，已实现** | RQ2 的零贡献判定只需关开关，不改代码；缺它只能靠静态猜测 |
| holdout 末次复核 | **必要，已实现** | 在演化池上搜索必须有过拟合对照，否则主张不成立 |
| 族错误率 + 功效 | **必要，已实现** | 六臂同表比较，不校正 α 就是多重比较；无功效就不许把零结果当证据 |
| L4 自由重构 | 推迟 | 等价性无法保证；只保留最低风险算子 |
| 模块级独立演化/合并 | 推迟 | 只保留 SHarP 式模块消融作基线 |
| DPP coreset / 方向一致性 / 开放式档案 / weak seed / meta-agent | 不做 | 不服务「简化不劣」的主主张，属阶段二 |

---

## 13. 构建顺序

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | contracts + D01 K + D02 components + D11 protocol + store | 同一 harness 两次测量一致；组件数可复算 |
| **P1** | A01 prestudy（P1 静态 + P2 触达） | 出 `prestudy.json`；决策门可执行 |
| **P2** | A02 baseline + AD01/AD02/AD03 + D06 verify | H0 跑通；基线无效能阻塞 |
| **P3** | A04 simplify（L1/L2）+ D05 protection + D08 gate | 删一个零触达组件，四态表正确 |
| **P4** | L3 消融 + D07 stats | 关一个模块，出有/无贡献结论 |
| **P5** | A06 diagnose | 注入 20 个故障，出 top-1/top-3 |
| **P6** | A05 evolve + COMBO + A07 compare | 六臂可跑；S-K 散点出图 |
| **P7** | A08 report + A09 verify_evidence | 六张表由命令生成 |

**P1 是决策门**：它不过，P2 以后不要开工。

---

## 14. 论文映射

| 论文位置 | 用什么 | 命令 |
|---|---|---|
| 方法：删除算子 | L1–L4 + 四态 + 保护 | `evo scan` + `evo apply`；L3 用 `evo ablate` / `evo ablate-run` |
| 方法：准入控制 | 非劣 + K 互锁 + token + 行为 | `evo round`（九门） |
| 实验：对象可删质量 | T1 | `evo prestudy` |
| 实验：主结果 | T2 + S-K 散点 | `evo run --arm ...` / `evo loop` + `evo compare` |
| 实验：删测证据 | T3 | `evo report` |
| 实验：不是更保守 | T4（placebo/SHarP） | `evo compare` |
| 实验：可归因性 | T5 | `evo diagnose` |
| 实验：成本 | T6 | `evo report --cost` |
| 方法：统计校准 | 族错误率 + 功效 | `evo compare --family-size` / `evo power` |
| 附录：可复现 | run.json + 哈希 + 种子 | `evo verify-evidence` |

---

## 只记一件事

**K 降了不算成果；「K 降了、删测证明没掉分、holdout 没掉、可归因性上升、随机删做不到」才算。**
