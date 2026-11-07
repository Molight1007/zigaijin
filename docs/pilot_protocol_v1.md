# Pilot v1：第一轮改进对比的题目分配

2026-11-06 ｜ 对应 manifest：`specs/pilot_v1.json`

---

## 1. 目的与边界

这轮只回答三件事：

1. 改进后的 agent 有没有明显跑坏；
2. 有没有出现“改对”的正向信号；
3. 值不值得扩到正式 SIM/COMBO 实验。

**它不用于论文非劣结论。** 6–11 题 × k=3 只能发现大效应，不能证明 5pp 小效应。

---

## 2. 题目分配（全部来自冻结题库）

| 集合 | 数量 | 用途 | 是否可用于调参 |
|---|---|---|---|
| `smoke` | 2 | 基础跑通 | 是（只做冒烟） |
| `pilot_dev` | 8 | 第一轮改进对比、问题定位 | 是 |
| `pilot_validation` | 3 | dev 出现信号后做防过拟合检查 | **否** |
| `sensitivity_smoke` | 1 | cancel-async-tasks 的混合结果敏感性 | 仅看信号 |
| `holdout` | 27 | 最终确认 | **绝对不碰** |

### pilot_dev（8）

`largest-eigenval`、`model-extraction-relu-logits`、`vulnerable-secret`、
`sqlite-db-truncate`、`password-recovery`、`configure-git-webserver`、
`torch-tensor-parallelism`、`count-dataset-tokens`

选择依据：H0 里这些题便宜，且至少有 pass / agent_error / 可修复失败中的一种信号；不是单纯按最低成本排序。

### pilot_validation（3）

`mcmc-sampling-stan`、`merge-diff-arc-agi-task`、`query-optimize`

只在 dev 出现正向信号后跑；不参与改进决策。

### sensitivity_smoke（1）

`cancel-async-tasks`：H0 smoke 里 1 过 / 2 挂，是现有数据里唯一有方差的题。
它是 smoke，不是 holdout，不能用于最终主张。

### holdout（27）

完全使用 `runs/frozen/tb21-h0/run.json` 的 holdout 池；不在 pilot、不在改进回路、
不在 monitor/architect/summarizer/executor 的输入里出现。

---

## 3. OOD / 防过拟合

- **当前没有真正的 OOD。** 题库只有 TB2.1；pilot_validation 和 holdout 都是
  同分布 disjoint，不是 OOD。不能把“同分布留出”写成 OOD。
- 真正的 OOD 需要第二套 benchmark（TB2.0 / SWE-bench 等），当前没有下载；
  这轮不做，也不假装。
- 防过拟合靠三层：
  1. `pilot_dev` 与 `pilot_validation` 预注册、互斥；
  2. `holdout` 只在最终确认时用一次；
  3. 任务选择在候选结果之前固定，看到结果后不改题。

---

## 4. 规模、成本、停止规则

- 设计：`pilot_dev` 8 + `pilot_validation` 3，H0 与 candidate 各 k=3。
- 预计 trials：candidate 33 + H0 补 22 + sensitivity 3 = **58**。
- 预计 API 成本：约 **$1.72 ≈ ¥12.2**；预算上限 **¥20**。
- 时间：约 8–14 小时（并发 2，便宜题通常更快）。

停止规则（预注册）：

1. **地板**：H0 和 candidate 在 ≥7/8 dev 题上都为 0 → 测不出，停止；
2. **dev**：candidate 让任何 H0 已过的 dev 题倒退，或净胜题 <2 → 停止；
3. **validation**：只在 dev 有信号后跑；validation 净胜 <0 或 >1 题倒退 → 停止；
4. **holdout**：只在最终确认跑，不参与调参；
5. **预算**：直接 API 花费超过 ¥20 → 中止，不重试。

---

## 5. 现在准备到哪一步

- 题目分配已固定：`specs/pilot_v1.json`；
- 已用冻结池校验：dev/validation ⊂ evolution，sensitivity ⊂ smoke，
  与 holdout 零交集（`tests/test_pilot_manifest.py`）；
- 等服务器原始 H0 数据积累完，补齐 H0 在 pilot 题上的 k=3，再跑 candidate。

**在看到服务器数据前不改题目、不改 stop rules。**
