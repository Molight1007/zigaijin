# math — MathPilot 域适配

把 MathPilot 接进 RRSI 的受控演化循环。按 `domains/coding/` 的契约实现，不自创接口。

## 目录

| 文件 | 作用 |
|---|---|
| `adapter.py` | `MathDomain(Domain)`：`run` / `score` / `load_trial` / `smoke` 等 12 个方法 |
| `briefs.py` | 四套角色提示（analyst / digester / proposer / critic） |
| `render.py` | 把一条 trial 渲染成 analyst 可读的文本 |
| `rrsi.json` | 实例超参（T=2 / k=1 / m=2 / δ=null / β₁=5.0） |
| `SKILL.md` | proposer 宪法：判分方式、三道门、越界、litmus test |
| `PATTERNS.md` | 五类已知失败模式 + 未试方向 |
| `make_split.py` | 生成 evolve / held-out 分层划分（**跑一次即冻结**） |
| `sync_from_source.py` | 把主项目的可编辑子集同步进 `third_party/mathpilot` |
| `data/` | `evolve_ids.txt` / `heldout_ids.txt` / `smoke_ids.txt` / `split_meta.json` |

## 前置

```bash
# 1) 镜像必须与主项目一致，否则演的是上周的代码
python domains/math/sync_from_source.py --dry-run
python domains/math/sync_from_source.py

# 2) 题库划分（只需一次；换了种子，此前所有结论作废）
python domains/math/make_split.py
```

环境变量（都可省，省了走默认）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MATHPILOT_SOURCE` | `D:\对标leap的数学智能体项目` | 同步源 |
| `MATHPILOT_ROOT` | `third_party/mathpilot` | 运行时 harness 位置，覆盖镜像 |
| `MATHPILOT_BANK` | `<root>/题库/official112_本地测试题库/official112_full.jsonl` | 题库 |
| `MATHPILOT_PYTHON` | 当前解释器 | 跑评测用的 Python |
| `MATHPILOT_BASE_URL` / `MATHPILOT_MODEL` | 读 `DEEPSEEK_*` | 显式传给 `run_eval.py`，避免回落到 `localhost:8000` |

`adapter.run()` 会**主动剔除代理变量**（`HTTP_PROXY` 等）。本机曾因一个未启动的
`127.0.0.1:56398` 代理导致整批 502，这一步不要去掉。

## 跑

```bash
python rrsi.py --domain math smoke      # 存活：编译 + 跑 2 道 smoke 题
python rrsi.py --domain math baseline   # Evaluate(H₀)，落 frontier
python rrsi.py --domain math run        # 演化（T=2）
python rrsi.py --domain math status
```

## 当前状态（诚实标注）

| 项 | 状态 |
|---|---|
| 域加载 / 题库读取 / 打分反验 | ✅ 已实测（见下） |
| `smoke` | ⚠️ 未跑（需一次真实评测，单题 15–35 分钟） |
| `baseline` | ❌ 未跑，δ 未标定 |
| `run` | ❌ 未跑 |
| 成本口径 | ⚠️ 用 `elapsed_sec` 顶替 token（项目结果 jsonl 无 token 字段） |
| held-out | ✅ 27 题，按 domain 分层冻结（seed 20261027） |

**已实测的三条**：

1. `load_domain('math')` → briefs 4 套 / critic 7 条 / signals 7 类。
2. 题库读到 112 题；划分 evolve 85 / held-out 27，无交集、并集完整。
3. `score()` 反验 `official112_local_0910.jsonl`：答对 19/112、缺失 0、pass_rate
   16.96%、成本中位 1088 秒 —— 与独立统计的 1088.7 秒一致。

## 两个必须记住的约束

1. **成本与题长无关**：单题 15–35 分钟取决于阶段数与重写轮次。一个删掉重写的编辑，
   价值高于所有微优化之和。
2. **缺失 ≠ 0 分**：RRSI 把「没有记录」算 missing 且占分母，缺失率 >15% 整轮判
   invalid。所以 `run_eval.py` 必须对每题落记录，否则整轮白跑。
