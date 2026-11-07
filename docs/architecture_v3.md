# 架构 v3：五员工 + 代码简化

2026-11-06 ｜ 状态：当前实现 ｜ 取代 `architecture_v2.md`

---

## 0. 一句话

> **以 RRSI 已证明的“分析 → 历史条件化提案 → 语义预筛 → 评测 → 选择”为底座，
> 把“删除代码”做成一等计划，并用 K / token / holdout / 可归因性证明：
> 删掉一部分 harness 代码后，S 不劣、holdout 不劣、token 不升，且随机删做不到。**

RRSI 是底座，不是主贡献；主贡献是**代码简化通道**和它的**效果证据**。

---

## 1. 五员工（不是 RRSI 的四员工）

```
[监测员] ─┐
          ├─→ [总结员] → [执行员] → [审核员]
[架构师] ─┘
```

| 员工 | 回答的问题 | 输入 | 产物 | 来源 |
|---|---|---|---|---|
| **监测员** monitor | 测试数据里发生了什么？哪些题弱？ | reference outcomes、traces、coverage | `monitor.json` | RRSI analyst/digester 的“观察”位置；当前为确定性汇总，LLM digest 仅可选 |
| **架构师** architect | 代码怎么切、K 多少、哪里可删？ | subject root、boundary、census/ablation | `architecture.json` | 本项目新增；RRSI 只有静态组件词表，没有架构员工 |
| **总结员** summarizer | 要解决什么问题、改什么？ | monitor + architecture + history/budget/exploration/prune | `plan.json` | RRSI proposer 的 planning 半 |
| **执行员** executor | 计划能否落地并改对代码？ | plan + monitor/architecture | `execution.json`、`diff.patch` | RRSI proposer 的 editing 半；DeepSeek 改代码，工具物化/审计 |
| **审核员** reviewer | 产出安全吗？值得采纳吗？ | execution、verify、evaluation、gate | `review.json`、`ledger.jsonl` | RRSI critic + selection + 本项目的简化门 |

监测员和架构师**并行**，互不依赖；两者都只向总结员供数。实现见
`evo/application/workforce.py::run_workforce_round`（`ThreadPoolExecutor`）。

---

## 2. RRSI 的哪些机制照抄了

RRSI 实际是 4 个 LLM 角色（analyst / digester / proposer / critic）+ 1 个算法选择器；
没有架构师，proposer 同时计划和执行，critic 在评测前。

| RRSI 机制 | 本实现的落点 | 状态 |
|---|---|---|
| 退火编辑预算 `b_t` | `evo.domain.evolve.edit_budget` → `summarize_edit_plan` | 已接 |
| 停滞标志 `σ_t` | `stall_flag` | 已接 |
| 未探索组件 / 保留探索位 | `exploration`、`--reserved` | 已接 |
| 剪枝集 `B_t` | `prune_set`（最近 `ΔS ≤ 0`） | 已接 |
| 逐 edit 历史 `L_t` | `--history` JSONL → proposer 上下文 | 已接 |
| proposer 直接改源码 | `evo.application.propose`，在 staged copy 上跑 | 已接 |
| critic 评测前语义审查 | `review_semantics`，六类拒绝 | 已接，`--semantic-review` / `--deepseek-assist` |
| `S*−δ` + 成本规则 + novelty | `evo.domain.evolve.judge` | 已接 |
| analyst + digester 逐轨迹摘要 | `analyze_evidence` + `digest_trace`；`--analysis --analysis-digests` | 已接（单层 digest，不是完整工具子代理） |
| critic 驳回后的有界 repair | `repair_edit_plan` + runner repair loop；`--repair-rounds` | 已接 |
| `m` 个 variant 并行 | `run_workforce_loop` + `--variants`；删除批用 static/random/sharp，编辑用 variant brief | 已接 |
| scoreboard / per-task digests | `_scoreboard` 从可核对 edit 历史回灌 proposer | 已接（不是 per-task 预测 scoreboard） |

---

## 3. 我们的优化：删除是一等计划

### 3.1 总结员的三条路径

`evo.application.workforce.summarize_*`：

- `delete`：从架构师的 L1/L2/L3 候选里选非重叠批；这是主路径。
- `edit`：DeepSeek-native 时总结员只出 brief（`summarize_direction`），执行员用 `propose` 改源码；非 native 时沿用 staged copy。
- `auto`：有可删候选优先删除，否则编辑。

### 3.2 执行员

- `execute_deletion_plan`：复制 subject → 应用删除批 → 计算 changed / diff / K before-after。
- `execute_edit_plan`：把 staged copy 物化为候选副本 → 计算 changed / diff / K。
- `execute_deepseek_plan`：DeepSeek-native 执行员，读 brief + 架构建议 + 失败证据，用 `propose` 直接改代码/配置。

执行员不做判定；判定全部留给审核员。

### 3.3 审核员的判定栈

```
1) RRSI selection（所有臂）
   S ≥ S* − δ
   cost rule: ΔS > δ ? ΔC ≤ β0+β1ΔS : w_sΔS − w_cΔC + w_n·novelty > 0

2) 简化门（SIM / COMBO / PLACEBO / SHARP）
   有效测量 → 分数非劣 → holdout 非劣 → K 互锁 → token → 保护 → T4 → Pareto → worst-task

3) 最终 verdict
   delete（接受）/ keep（拒绝）/ undecidable（缺证据）/ undeletable（删了但调用点残留）
```

- **EVO**：只走 RRSI selection，不被 K 门拦（它的主张不是降 K）。
- **SIM / COMBO**：RRSI selection 和简化门必须同时过。
- 判定结果写 `ledger.jsonl`，证据写 `review.json`。

---

## 3.4 DeepSeek 员工分配（参考 RRSI）

| 员工 | DeepSeek 负责 | 确定性工具负责 |
|---|---|---|
| 监测员 | 失败归因、逐轨迹 digest（`--analysis --analysis-digests`） | outcomes/traces/coverage 采集、打分 |
| 架构师 | 读确定性代码图，产出 hotspots / opportunities / risks（`--deepseek-assist`） | AST 代码图、组件、K、候选、保护、AI cleanup |
| 总结员 | 问题总结 + 修改方向 + executor brief；删除候选选择（`--llm-summarizer` / `--deepseek-assist`） | 候选枚举、ID/重叠校验、非法回退 |
| 执行员 | DeepSeek-native 执行员直接改代码/配置（`--deepseek-native`） | 复制、diff、K、compile/import、审计 |
| 审核员 | 语义 critic（accept/revise/reject + suggestions）、repair 回灌 | verify、RRSI selection、K/token/holdout 门 |

原则：**LLM 做判断、方向、改代码和审查；工具做可复现操作；算法做最终判定。** DeepSeek 的删除选择只能从
架构师给出的候选 ID 里选；未知 ID、重叠选择、空选择都会回退到确定性规则。

## 4. 代码简化效果（主贡献的证据）

### 4.1 每轮自动写

删除计划通过审核后，`run_workforce_round` 自动写 `effect.json`：

| 块 | 内容 |
|---|---|
| `k` | 6 个 K 指标的 before / after / delta / reduction_pct |
| `k_gate` | K_ast 降 ≥0.5%、K_lines_nb 降 ≥0.5% 且 ≥10 行、chars/cc 不升 |
| `score` | mean delta、paired bootstrap CI、non_inferior、holdout、worst-task |
| `tokens` / `cost` | 总量与比值；token 升直接不判 effective |
| `layers` | L1 / L2 / L3 各删多少、各贡献多少 K |
| `diagnosis` | before/after top1、top3、delta、是否改善 |
| `controls` | PLACEBO / SHARP 的 K 降幅与非劣结果 |
| `verdict` | `simplification_effective` / `not_supported` |
| `claim` | 可直接进论文的一句话结果 |

判定不是“K 降了就赢”：必须 **K 门 + 分数非劣 + holdout 非劣 + token 不升 +
worst-task 不越界**，才写 `simplification_effective`。

### 4.2 独立命令

```
evo simplify-effect \
  --reference H0.outcomes.jsonl \
  --candidate SIM.outcomes.jsonl \
  --tasks ... --holdout ... \
  --applied execution.json \
  --plan plan.json \
  --diagnosis-before before.json --diagnosis-after after.json \
  --placebo placebo.json --sharp sharp.json \
  --out simplification.json
```

实现：`evo/application/simplify_effect.py`；入口：`evo/cli.py::_cmd_simplify_effect`。

---

## 5. 运行入口与产物

```
evo workforce \
  --run runs/<id>/run.json \
  --subject-root vendor/terminus2 \
  --reference-outcomes <incumbent outcomes.jsonl> \
  --pool evolution --holdout-pool holdout --smoke-tasks ... \
  --mode delete|edit|auto \
  --candidates <census.l2.json> --ablation <ablation.json> --traces <trace dir> \
  --history <history.jsonl> --trajectory <trajectory.json> \
  --semantic-review \
  --jobs-dir <harbor jobs> --work <work> --out <out>
```

产物：

```
out/
  monitor.json          监测员
  architecture.json     架构师
  plan.json             总结员
  execution.json        执行员
  diff.patch
  verify.json           审核员-确定性验证
  evaluation.jsonl      评测
  review.json           审核员-最终判定
  effect.json           简化效果（删除计划）
  state.json            阶段状态
  events.jsonl          阶段事件
  ledger.jsonl          只追加判定
  round.json            本轮汇总
```

多轮闭环：`evo workforce --loop --total-rounds N ...`。搜索期只评演化池，
`require_holdout=false`；每轮接受后 incumbent 换位、更新 `S*` 与 history；
最后对最终 incumbent 只跑一次 holdout，写 `loop.json` / `holdout.json`。

---

## 6. 模块地图

### 必须保留

- 判据/度量：`domain/gate` `kmetrics` `stats` `protection` `ledger` `evolve`
  `components` `scan` `census` `ablation` `arms` `protocol` `state` `faults` `verify`
- 用例：`baseline` `compare` `power` `freeze` `propose` `apply` `simplify`
  `simplify_arm` `version_compare` `round` `evolve` `report` `diagnose` `faultcase` `faultrun`
  `ablate` `prestudy` `traces` `armrun` `workforce` `simplify_effect`
- 适配：`harbor_tb21` `deepseek` `llm` `git_worktree` `store_fs` `coverage_py`
  `traced_terminus2`（动态加载）
- 契约：`contracts/types`

### 已删除（2026-11-06）

| 模块 | 原因 |
|---|---|
| `domain/diagnosis.py` | 只有测试引用；`application/diagnose.py` 已有生产实现，功能重复 |
| `application/loop.py` + `tests/test_loop.py` | 旧多轮 RRSI/EVO 闭环，已被 `workforce --loop` 取代；删除 559 行实现 + 389 行测试 |

AI 代码删减经验与检测器说明见 `docs/ai_code_cleanup.md`；`map_architecture` 会把
`ai_cleanup` findings 写进 `architecture.json`，但它们是候选，不是自动删除项。
| `infra/runtime.py` | 只有测试引用；redaction / run-id 工具未接进任何生产路径 |

### 仍然重叠、暂不删除

| 模块 | 原因 |
|---|---|
| `application/round.py` `simplify.py` `simplify_arm.py` | 旧 `round` / `simplify` 命令仍在用；论文实验要复现 |

---

## 6.5 第一版臂语义

| 臂 | 入口 | 语义 |
|---|---|---|
| H0 | `baseline` / `run` | 基线，不跑 workforce |
| EVO | `workforce --loop --mode edit` | RRSI selection，不被 K 门拦 |
| SIM | `workforce --loop --mode delete`；或 `simplify --mode full --l3` | 删除 + K/token/holdout 门；full 含 L1+L2+L3 |
| COMBO | `workforce --loop --arm COMBO --mode auto` | 优先删除，无候选则编辑；仍需 K 门 |
| PLACEBO | `workforce --arm PLACEBO --batch-mode random --target-k-ast N` | 等量随机删 |
| SHARP | `workforce --arm SHARP --batch-mode sharp --target-k-ast N` | 按组件整组删；模块开关消融另走 `ablate-run` |

`simplify --mode full --l3 ablation.json` 会把 `zero_contribution` 的 lever 作为
agent kwargs 传给评测；K 门只看代码删除，S/token 看整体效果。

## 7. 当前状态与缺口

- 测试：50 文件 / **808 用例**，`pytest` 退出码 0。
- 预研：`pending_l2`（census 仅 2 题，< 10）；L3 未跑。
- H0：冻结协议 60×3=180 trials，实际约 19 个 `result.json`。
- 主臂 SIM / COMBO / PLACEBO / SHARP：尚未全量运行。
- 诊断 / 报告：尚未产出真实 `fault-run`、`diagnose`、`report` 产物。

下一步优先级：

1. L4 安全改写与文本 K。
2. SHARP 的真实模块开关消融仍由 `ablate-run` 承担；workforce SHARP 是分组删除近似。
3. 删除批的 variant 差异目前只靠 static/random/sharp，后续可加 target-K 分层与多目标 K。
4. 预研 P2/P3 与 H0 全量。

---

## 8. 明确不做

- 不训练权重、不造数据集、不做 GUI/分布式。
- 不把 K 合成一个标量；代码 K 与文本 K 分开。
- 不把“没显著下降”当“非劣”。
- 不把 RRSI 的四员工说成我们的五员工；五员工是重新分解，不是复现。
