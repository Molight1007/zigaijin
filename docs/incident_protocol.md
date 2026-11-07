# Bug / Incident 定位与记录协议

2026-11-06 ｜ 对应命令：`evo incident open/list/show`

---

## 1. 先说结论

能做到的“精准”分两层：

| 层级 | 能否做到 | 依据 |
|---|---|---|
| 组件 / 函数级定位 | 能 | coverage、AST span、组件图、fault ground truth |
| 精确 root cause | 需要复现 + 回放/消融；否则只是假设 | replay、differential、ablate、最小复现 |

没有轨迹、没有复现、没有回放时，**不能把 LLM 的猜测写成“根因”**，只能写 `cause` 假设，`status=open`。

---

## 2. 问题位置从哪里来

按可靠性从高到低：

1. **注入故障 ground truth**：`evo faults` 生成的 mutation 自带 `path/line/operator`，`fault-run` 记录 truth；这是最硬的定位。
2. **异常 / traceback**：失败 trial 的 trajectory / result 里的最后一个异常、文件和行号。
3. **coverage / census**：`traced_terminus2` 的行级轨迹 + `census` 的函数 span，能定位到实际执行的函数。
4. **组件图**：`architecture.json` 的 components 与 graph，把文件/函数映射到组件。
5. **消融**：`ablate-run` 关掉某个 lever 后现象复现/消失，确认因果。
6. **版本 diff**：`execution.json` / `diff.patch` 定位本次改动引入的位置。

记录时 `location` 写 `path:line` 或 `component`，二者能同时写就同时写。

---

## 3. 后果从哪里来

必须同时记，不能只写“失败了”：

- `reward`：该 task/trial 的分数变化；
- `status`：`completed / agent_error / infra_error`；
- `tokens_in/out` 与 `cost_usd`：是否用更多 token 换来的；
- `wall time`：是否超时；
- `K`：是否改了代码量；
- `validity`：是不是缺 trial / 缺 token 导致的假结论；
- `version`：H0 / previous / candidate。

---

## 4. 记录格式

`evo incident open` 写入 append-only JSONL：

```bash
.venv/Scripts/python.exe -m evo incident open \
  --id inc-20261106-001 \
  --version v1 \
  --phase evaluation \
  --location vendor/terminus2/terminus_2.py:837 \
  --component terminus2:summarize \
  --task terminal-bench/example \
  --trial trial-1 \
  --symptom "timeout after fixed wait" \
  --consequence "reward 0; tokens +40%" \
  --cause "fixed duration instead of polling" \
  --fix "poll with bounded retries" \
  --evidence runs/versions/v1/outcomes.jsonl \
  --path runs/incidents.jsonl
```

查询：

```bash
evo incident list --path runs/incidents.jsonl --json
evo incident show --path runs/incidents.jsonl --id inc-20261106-001
```

字段：

| 字段 | 说明 |
|---|---|
| `id` | 唯一 id |
| `version` | 出问题的版本 |
| `phase` | monitor / architect / summarize / execute / review / evaluation |
| `location` | `path:line` 或组件 |
| `component` | 组件 id（可选） |
| `task_id` / `trial_id` | 复现位置 |
| `symptom` | 看到的现象 |
| `consequence` | reward/token/cost/K/validity 的后果 |
| `cause` | 根因或**假设**；没证据就标假设 |
| `fix` | 修法 |
| `status` | open / fixed / verified / wontfix |
| `evidence` | outcomes、trace、diff、ledger 路径 |

---

## 5. 定位流程

1. 发现：`version-compare` / `simplify-effect` / validity 报异常；
2. 复现：同 task、同 trial、同模型、同 k；
3. 定位：traceback → coverage/census → 组件图；
4. 确认：注入/消融/回放，看现象是否随该位置变化；
5. 修复：改一处，记录 `fix`；
6. 回归：新版本重跑同一 task/trial，`status=verified`；
7. 入账：`ledger` + `incident`，下一轮 `version-compare` 检查是否复发。

**没有第 4 步，就只能写“疑似原因”，不能写成“原因”。**
