# 版本对比数据协议

2026-11-06 ｜ 对应命令：`evo version-compare`

---

## 1. 目标

回答三个问题，且每个数字都能追到原始 trial：

1. 当前版相对 **H0** 有没有提升/退步；
2. 当前版相对 **上一个改动版本** 有没有提升/退步；
3. 提升是不是用更多 token/更大 K 换来的。

不能只报总体均值；必须保留 **逐题、逐 trial、配对** 的原始数据。

---

## 2. 每个版本必须存的东西

```
runs/versions/<version>/
  outcomes.jsonl        # 原始 trial，一行一个
  applied.json          # 该版本相对上一版的 K before/after（可选但强烈建议）
  execution.json        # workforce 执行产物（可选）
  version.json          # 版本元数据
```

### outcomes.jsonl 必填字段

| 字段 | 含义 |
|---|---|
| `task_id` | 题库 id，必须和 H0/上一版完全一致 |
| `trial_id` | 全局唯一；用于配对和去重 |
| `status` | `completed / agent_error / infra_error` |
| `reward` | 0/1 或任务分；`completed` 必须有 |
| `tokens_in` / `tokens_out` | 缺失就留 null，不能填 0 冒充 |
| `cost_usd` | 缺失留 null |
| `failure_cause` | 可选，失败原因 |

### version.json 必填字段

```json
{
  "version_id": "v1",
  "parent": "h0",
  "subject_hash": "...",
  "tasks": ["terminal-bench/..."],
  "k": 3,
  "outcomes": "runs/versions/v1/outcomes.jsonl",
  "applied": "runs/versions/v1/applied.json",
  "created_at": "2026-11-06"
}
```

---

## 3. 可比性硬约束

`version-compare` 会检查：

- H0 / previous / candidate 的 `tasks` 必须完全一致；
- 每个任务都按同一个 `k` 记录；
- holdout 必须是 tasks 的子集；
- 缺 trial / 多 trial / 未知 task 会让对应 arm 的 validity 失败，不能静默忽略。

**不允许**：上一版跑了 10 题、当前版跑了 12 题就直接比均值；也不允许换题后比。

---

## 4. 命令

```bash
.venv/Scripts/python.exe -m evo version-compare \
  --h0     runs/versions/h0/outcomes.jsonl \
  --previous runs/versions/v0/outcomes.jsonl \
  --candidate runs/versions/v1/outcomes.jsonl \
  --tasks  terminal-bench/a,terminal-bench/b,terminal-bench/c \
  --holdout terminal-bench/c \
  --k 3 \
  --previous-applied runs/versions/v0/applied.json \
  --candidate-applied runs/versions/v1/applied.json \
  --csv runs/comparisons/v1.csv \
  --out runs/comparisons/v1.json
```

输出：

| 块 | 内容 |
|---|---|
| `per_task` | 每个 task 在 H0 / previous / candidate 下的 mean_reward、trials、tokens、cost |
| `comparisons` | `H0->previous`、`previous->candidate`、`H0->candidate` 三组配对 bootstrap CI、non_inferior、worst_task、wins/losses/ties |
| `validity` | 每个 arm 的 missing/infra/token coverage |
| `k` | previous vs candidate 的 K before/after/delta/reduction_pct |
| `verdict` | `candidate_improved / candidate_non_inferior / candidate_unchanged / candidate_regressed / candidate_holdout_regressed` |

CSV 只放逐题三臂均值，方便人工核对；JSON 放完整统计。

---

## 5. 怎么判定“改动有效”

先说清楚：**pilot 只能看方向，不能当论文非劣证据。**

- `candidate_improved`：相对上一版 non-inferior，且赢题数 > 输题数，均值上升；
- `candidate_non_inferior`：没掉，但没有净胜；
- `candidate_unchanged`：逐题完全平；
- `candidate_regressed`：配对 CI 跌破 `-margin`；
- `candidate_holdout_regressed`：holdout 掉出界，直接否决。

论文口径还要加：

1. 有效测量（validity pass）；
2. K 不升且达到 K 门；
3. token 总量不升；
4. holdout 非劣；
5. n 足够（用 `evo power` 看 MDE，而不是“没显著下降”）。

---

## 6. 和 pilot 的关系

- `specs/pilot_v1.json` 固定 pilot 的 dev/validation 题；
- 每轮改动后存一个 `runs/versions/<version>/`；
- 下一轮用 `version-compare` 对 H0、上一版、当前版；
- dev 有信号才跑 validation；holdout 只在最终确认跑一次；
- 不改题、不改 stop rules，避免事后挑题。
