# 代码架构：`exp` — agent 自改进实验外壳

2026-11-01 ｜ 目标态架构。配套 `13_工具实现计划.md`（施工顺序）与 `12_工具目标与验收.md`（判据）。

---

## 〇、一句话

> 整个工具就是一台**三段机**：**变异 → 评测 → 判定**。
> 所有模块都挂在这三段的某一格上，挂不上的功能一律不做。

```
     ┌─────────┐      ┌──────────┐      ┌─────────┐
     │  变异    │ ───▶ │   评测    │ ───▶ │  判定    │
     │ mutate  │      │ evaluate │      │  judge  │
     └─────────┘      └──────────┘      └─────────┘
      删哪个组件         跑变体得 S/C       三态 + K 否决
      ↓                ↓                 ↓
   变体目录          记录（十类字段）      结论 + 证据
```

**为什么是这个形状**：`14` 的三条硬失败里，两条直接对应它的两端——
"删测结论靠人判断"是**判定**没做成代码，"数字要人工抄"是**评测**没留痕。
中间那条"K 算不出分解"是变异的前提（不知道删什么）。

---

## 一、五条设计原则

| # | 原则 | 具体含义 | 违反会长什么样 |
|---|---|---|---|
| 1 | **单一契约源** | 所有跨模块数据结构只在 `contracts.py` 定义一次 | 两个模块各自定义 `Task`，字段悄悄错位 |
| 2 | **适配层只加文件** | 新增 agent / bench = 新增一个文件 + 注册一行 | 换 agent 要改 runner |
| 3 | **K 离线算** | K 是源码的静态属性，与运行无关，永不进入 rollout 循环 | 把 token 计费混进 K，导致 K 随机器变 |
| 4 | **删除必须物化** | 删 = 真的从源码里移除并重新跑；运行时禁用**不算**删除 | 用 patch 短路冒充删除，K 没降却报"删了" |
| 5 | **不可判是一等公民** | `undecidable` 是独立枚举值，任何路径都不许把它折叠成 `delete` | 三态变两态，核心创新归零 |

---

## 二、分层

```
┌──────────────────────────────────────────────────────────────┐
│  cli.py            命令面：doctor / k / dead / run / calib    │
│                    delete / report / pool                     │
├──────────────────────────────────────────────────────────────┤
│  run/              编排：一次 run = agent × bench × config × seeds │
│                    runner.py · record.py · store.py            │
├───────────────┬───────────────┬──────────────┬───────────────┤
│  adapt/       │  bench/       │  measure/    │  remove/      │
│  被测 agent   │  场地与判分    │  K 度量      │  变异         │
│  （可插拔）    │  （可插拔）    │  （离线）     │  （物化）      │
├───────────────┴───────────────┴──────────────┴───────────────┤
│  judge/            判定：gate（K 硬否决）· delta（δ 实测）· pools（双池）│
├──────────────────────────────────────────────────────────────┤
│  report/           出表：四张 markdown + bootstrap CI          │
├──────────────────────────────────────────────────────────────┤
│  contracts.py      唯一契约源（Task/Component/KVector/Record/Verdict）│
│  reuse.py          对旧 src.domain 的唯一引用点                 │
└──────────────────────────────────────────────────────────────┘
```

**依赖方向严格单向**：`cli → run → {adapt, bench, measure, remove, judge} → contracts`。
`report` 只依赖 `run.store` 与 `judge`。**任何下层不许 import 上层。**

---

## 三、目录树（目标态）

```
exp/
  __init__.py
  __main__.py                # python -m exp  →  cli.main()
  cli.py                     # argparse 子命令；只做参数解析与打印，不含业务
  contracts.py               # ★ 唯一契约源（见 §4）
  reuse.py                   # ★ 对 src.domain 的集中引用（见 §7）
  errors.py                  # PoolViolation / MutateError / BenchUnavailable

  adapt/                     # ── G1 适配层：一个 agent = 一个文件 ──
    __init__.py              #   AGENTS 注册表：name -> factory
    base.py                  #   AgentAdapter 协议 + 通用 run_loop()
    terminus2.py             #   TB 的 Terminus-2（容器内，tmux）
    miniswe.py               #   mini-swe-agent（进程内）
    react_toolbelt.py        #   RRSI 形态的 ReAct（自建薄壳）
    classify.py              #   分类 harness（Label-Primed，最便宜的删测场地）
    retrieval.py             #   检索 harness（BM25 / dense）

  bench/                     # ── 场地与判分：一个基准 = 一个文件 ──
    __init__.py              #   BENCHES 注册表：name -> factory（支持 name:version）
    base.py                  #   Bench 协议 + Session 协议 + Grader
    local.py                 #   tasks/v1.jsonl（30 题，本机目录，无容器）
    tb2.py                   #   Terminal-Bench 2.x（Docker，pytest 判分）
    classify.py              #   LawBench / Symptom2Disease / USPTO-50k

  measure/                   # ── G2 K 度量（纯离线） ──
    k.py                     #   KVector 计算 + agent 级汇总
    tools.py                 #   radon / tiktoken / gzip 的版本钉死与调用
    split.py                 #   六类组件切分（切分规则即口径，版本化）

  remove/                    # ── G3 删除 ──
    static.py                #   路线 A：AST 可达性 ∩ sys.monitoring 覆盖
    mutate.py                #   变体物化：块级删除 + 语法/引用/冒烟三道门
    counterfactual.py        #   路线 B：逐组件删 → 真跑 → 三态
    whitelist.py             #   防御式代码白名单（assert/try/参数校验）

  judge/                     # ── G4 + G5 ──
    gate.py                  #   判定：K 硬否决 + δ 双地板 + 三维支配
    delta.py                 #   δ 实测：重复评测评散布，只升不降
    pools.py                 #   双池 + ContextVar 运行时拦截

  report/                    # ── G6 ──
    tables.py                #   四张 markdown 表（列名逐字对齐模板）
    stats.py                 #   bootstrap（默认 100,000）+ 配对 CI

  run/                       # ── 编排 ──
    runner.py                #   一次 run 的主循环（串行、可中断、可续跑）
    record.py                #   十类字段 + 追溯三元组
    store.py                 #   runs/<run_id>/ 的读写与索引

exp/tests/                   # 9 条自证测试（见 15_工具实现计划 §6）
  test_contracts.py  test_k_reproducible.py  test_dead_static.py
  test_mutate_gates.py  test_three_state.py  test_judge_k_veto.py
  test_pool_guard.py  test_tables.py  test_record_lineage.py
```

**规模估算**（按现有 `src/` 的代码密度反推，非承诺值）：实现约 5,000–5,500 行，测试约 800 行。

---

## 四、四个核心契约（`contracts.py`）

这是全文最重要的一节。**契约定错，后面全是返工。**

### 4.1 任务与场地

```python
@dataclass(frozen=True)
class Task:
    id: str                       # 全局唯一，落地为记录主键的一部分
    pool: str                     # evolve | ih | ood | quarantine
    instruction: str              # 给 agent 的自然语言
    assets: Mapping[str, bytes]   # 种子文件（TB2 从 task.yaml + 镜像来）
    checks: tuple[Check, ...]     # 判分规则（TB2 是 pytest 文件清单）
    bench: str
    version: str                  # 2.0 / 2.1 —— 版本必须进记录，否则两个版本混报

class Session(Protocol):
    """agent 与场地之间的唯一通道。容器与本地目录都是它的实现。"""
    def exec(self, cmd: str, timeout: float | None = None) -> ExecResult: ...
    def write(self, rel: str, data: bytes) -> None: ...
    def read(self, rel: str) -> bytes: ...
    def close(self) -> None: ...

class Bench(Protocol):
    name: str
    version: str
    def tasks(self, pool: str, limit: int | None = None) -> list[Task]: ...
    def open(self, task: Task) -> Session: ...           # 建环境（容器/目录）
    def grade(self, task: Task, session: Session) -> Grade: ...   # 判分
```

**为什么把 `Session` 抽出来**：Terminus-2 的 `TmuxSession`、本地的 `subprocess`、Docker 的 `exec`，
三者的差异全被 `exec/read/write` 三件事消化。agent 适配器因此**完全不知道**自己在容器里还是在本机——
这正是"换 agent 不改主流程"的另一半。

### 4.2 agent 与组件

```python
@dataclass(frozen=True)
class Component:
    id: str                 # "prompt::system" / "tool::bash" / "code::terminus_2.execute_commands"
    kind: str               # prompt | tools | middleware | memory | subagent | skills
    locator: Locator        # 怎么找到它、怎么删它
    meta: Mapping[str, Any] # 行数、符号名、是否白名单

@dataclass(frozen=True)
class Locator:
    mode: str               # text-span | py-node | py-file | config-entry
    file: str               # 相对 harness root
    start: int              # 1-based
    end: int
    symbol: str = ""        # py-node 时的限定名，用于删除后做引用闭合检查

class AgentAdapter(Protocol):
    name: str
    def components(self) -> list[Component]: ...          # G1 的四件事之一
    def prepare(self, workdir: Path) -> None: ...         # 把 harness 物化到变异目录
    def reset(self, task: Task, session: Session) -> None: ...
    def step(self) -> StepResult: ...                     # 单步推进；done=True 结束
    def observe(self) -> Observation: ...                 # 观察规范化（喂失败签名）
    def close(self) -> None: ...
```

**为什么 `step()` 是单步而不是"跑完整题"**：
① 超时、中断、续跑都需要在步粒度上留钩子；
② 失败签名要用规范化后的观察，而不是最后一条原始文本；
③ 记录 schema 的第 10 类字段（失败签名 + 同因率）只能从逐步观察里归一化出来。
**同时提供 `base.run_loop(adapter)`** 做便捷封装，避免每个适配器重复写 while。

### 4.3 K 向量

```python
@dataclass(frozen=True)
class KVector:
    k1_bytes: int          # 字节数
    k2_lines: int          # 非空行
    k3_tokens: int         # ★ 主口径：去注释 token（tiktoken，编码器钉死）
    k4_ast: int            # AST 节点数（代码类）
    k5_halstead: float     # Halstead Volume（代码类）
    cc_mean: float         # 平均圈复杂度（代码类）
    gzip_tau: float        # 压缩率 τ = 压缩后/原始（文本类交叉校验）

@dataclass(frozen=True)
class MetricSpec:          # 钉死清单，进 manifest，缺一项即"不可复算"
    radon: str             # "6.0.1"
    tiktoken: str          # "0.14.0"
    encoding: str          # "o200k_base"
    gzip_level: int        # 9
    splitter: str          # "v1"
```

**判定用哪个数**：`K_total = Σ k3_tokens`（跨组件求和，不加权）。
**报告给哪些数**：分类型小计 + 五指标分解。
**为什么不合成加权标量**：加权系数没有出处，合成一次就要在论文里辩护一次；
分项报，判定只用"和"，辩护成本最低。

### 4.4 记录与判定

```python
@dataclass(frozen=True)
class Record:                     # 一条 = 一个 (task, seed, variant) 的一次 rollout
    run_id: str
    task_id: str
    pool: str
    variant: str                  # "H0" / "del::tool::bash"
    score: Grade                  # passed / score / pass_at_k / pass_hat_k
    usage: Usage                  # tokens(分 cached/output) / steps / seconds
    k: KSnapshot                  # total + by_kind + per_component
    diff: DiffStat | None         # 变更分解：added/removed/rewritten（改写不算删）
    deletion: DeletionEvidence | None
    leak: LeakCheck | None
    floor: FloorInfo              # s_star / delta / delta_source
    lineage: Lineage | None
    failure: FailureSig | None
    trace: str                    # 逐步观察的落盘路径

@dataclass(frozen=True)
class Verdict:
    accepted: bool
    branch: str      # k_veto | floor | cost | delete | compress | pareto | band | undecidable
    reason: str
    numbers: Mapping[str, float]
```

**`branch` 是枚举而不是自由文本**——G6 的消融表要按 branch 统计"哪条门拒了多少次"，
自由文本统计不了。

---

## 五、两条数据流

### 5.1 一次 `run`（基线或候选评测）

```
cli.run(args)
  └─ runner.Runner(agent, bench, config, seeds)
       ├─ bench.tasks(pool="evolve", limit)     ← PoolGuard 在这里拦
       ├─ measure.snapshot(agent, variant="H0") ← 离线算 K，只算一次
       └─ for seed in seeds:                    ← 串行！不并行
            for task in tasks:
              session = bench.open(task)
              adapter.prepare(workdir); adapter.reset(task, session)
              while not adapter.step().done: ...
              grade = bench.grade(task, session)
              records.append(Record(...))
              session.close()
       └─ store.write(run_id, manifest, records, components)
```

产物：

```
runs/2026-11-01_m4/
  manifest.json     # argv / cmd_hash / commit / dirty / MetricSpec / bench+agent 版本 / δ
  records.jsonl     # 一条记录一行（十类字段）
  components.jsonl  # 组件清单 + K 分解（每个变体一份）
  traces/           # 逐步观察（失败签名与回放的原料）
  summary.json      # 聚合：S / C / K + bootstrap CI
```

### 5.2 一次 `delete`（反事实删测）

```
cli.delete(args)
  └─ counterfactual.run(agent, bench, tasks, n, whitelist)
       1. comps  = agent.components()
       2. dead   = static.dead(comps, coverage)        # 路线 A 候选
       3. pool   = whitelist.filter(dead)              # 防御式剔除
       4. base   = evaluate(agent, n)                  # 删前 S/C
       5. for c in pool:                               # 一次只删一个（R3）
            mut = mutate.remove(agent, c)               # 物化变体
            if mut.failed_gate:                         # 语法/引用/冒烟门
                verdict = "undecidable"; cause = mut.failed_gate
            else:
                cand = evaluate(mut, n)                 # 真跑
                verdict = three_state(base, cand, delta, n)
            table.append(Row(component=c, k_delta=mut.k_delta, ...))
```

**三态的判据写死**（这是 `14` 反复强调的"不可判不能被默认成无用"）：

| 判定 | 条件 |
|---|---|
| `keep` | `cand.S < base.S - δ` —— 删了掉分，这组件是活的 |
| `delete` | `cand.S ≥ base.S - δ` **且** `ΔK ≤ -阈值` |
| `undecidable` | ① 变体不可运行；② `n < required_n(σ, 0.1)`；③ `ΔS` 的 95% CI 覆盖 0 |

第 ②③ 条存在，就意味着**n=3 的删测表里必然大量显示 `undecidable`——这是正确行为，不是失败。**

---

## 六、八条设计决策（ADR）

| # | 决策 | 理由 | 被否掉的替代方案 |
|---|---|---|---|
| **D1** | 另起 `exp/` 包，不改造 `src/cli.py` | `src` 面向"改 skill 文本"的旧场景，614 个测试锁住了它的形状；在它上面改会同时破测试与本项目新口径 | 原地扩展 `src`：改动面 10,158 行，风险不可控 |
| **D2** | `exp` 不复制 `src` 代码，用 `reuse.py` 单点引用 | 复制必然分叉（`prototype/simplify` 已经踩过，靠 `sync_check.py` 兜底）；收口在一个文件，迁移只改一处 | 全量重写：`stats.py`/`detector.py` 已被真阳性验证过，重写是浪费 |
| **D3** | K 主口径 = 去注释 token（K3） | 跨语言、可复算、与计费口径同源；行数可被"两行合一"刷分（`kmetric.py` 的注释里已写明这个局限） | 用 SLOC 当主口径：中文提示词那段没法算 |
| **D4** | `δ_K = 0`，只有 `δ_S` 需要实测 | K 是静态源码的确定性函数，同一份代码算两次必然相同；给它设噪声带等于给"K 涨"开后门 | 照搬 δ_S 给 K：会允许 2% 的 K 增长被噪声解释掉 |
| **D5** | K 硬否决**默认开启**，可显式关闭 | `14` 的假候选（S+0.5/K+30%）在现有 `gate` 里会被接受（`κ1=0.5` 时预算 0.35 > 0.30）；硬否决是最小改动就能堵住的口子 | 继续调 `κ1`：调到 0 就是硬否决，不如直接写清楚 |
| **D6** | 删除必须物化，运行时禁用不算删 | "禁用"不改源码，K 不降；若允许它冒充删除，删测表会出现"删了 15 个组件、K 没变"的自相矛盾 | 用 patch/monkeypatch 做必要性探测：**那是路线 A 的运行时覆盖证据，不是删除** |
| **D7** | 双池用 `ContextVar` + 默认 `evolve`，不用函数参数 | 参数模式依赖调用方记得传；默认最严时，忘了设置也不会误读量外池 | `bench.tasks(pool=...)` 显式传参：漏传即无保护 |
| **D8** | 报告只出 markdown，不做 GUI | `14` 的非目标写死了"唯一消费者是论文"；markdown 能直接进 LaTeX 流程 | 做看板：多一个要维护的界面 |

---

## 七、与现有 `src/` 的关系

`reuse.py` 是唯一耦合点，全部 import 集中于此：

```python
# exp/reuse.py —— 只允许这一个文件 import src.*
from src.domain.stats import bootstrap_ci, paired_diff_ci, wilson_ci      # G6 统计
from src.domain.detector import analyze as dead_code_scan                 # G3 路线 A
from src.domain.complexity import measure as local_code_metrics           # G2 交叉校验
from src.domain.kmetric import lines_nb_code, lines_nb_text               # G2 辅助指标
```

| `src` 模块 | 用途 | 状态 |
|---|---|---|
| `domain/stats.py`（166 行） | bootstrap / 配对 CI / Wilson | **直接复用** |
| `domain/detector.py`（728 行） | 9 类死代码检测器 | **直接复用**，作为路线 A 的静态半边 |
| `domain/complexity.py`（218 行） | 自实现 Halstead/CC/MI | **交叉校验**：与 radon 结果对比，差异 > 5% 时报警 |
| `domain/kmetric.py`（87 行） | 非空行口径 | 复用（K2 与 K3 的对照） |
| `domain/gate.py`（281 行） | 旧判定器 | **不复用**，保留为 A1 消融的对照臂 |
| `domain/sample_pool.py`（219 行） | 失败样本池 | 与"数据集双池"是两件事，不动 |
| `cli.py` / `usecases/` | 旧 M0 入口 | 冻结，不再加功能 |

**冻结规则**：`src/` 从现在起只接受 bug 修复，不接受新功能。新功能一律进 `exp/`。
理由：两套入口并存不致命，两套**实现**并存才会互相污染。

---

## 八、记录 schema ←→ 十类字段

`00_总纲` §3.1 定的十类字段，逐一对到 `Record` 上（缺一格，论文对应小节就写不出来）：

| # | 字段组 | Record 字段 | 论文位置 |
|---|---|---|---|
| 1 | 能力 S | `score.passed / score / pass_at_k / pass_hat_k` | 主结果表 |
| 2 | 成本 C | `usage.{input,cached,output}_tokens / steps / seconds` | 副轴、帕累托图 |
| 3 | 复杂度 K | `k.total / by_kind / per_component` | K 定义表 |
| 4 | 变更分解 ΔK | `diff.{added, removed, rewritten}`（**改写不算删**） | "降在哪" |
| 5 | 删测证据 | `deletion.{component, s_before, s_after, c_after, verdict, evidence}` | **核心表** |
| 6 | 泄漏筛查 | `leak.{verdict, at, hits}`（`at` 必须在 `before_eval`） | 抗泄漏章节 |
| 7 | 地板与噪声 | `floor.{s_star, delta, delta_source}` | 判据章节 |
| 8 | 池标记 | `pool` | 双池图 |
| 9 | 谱系 | `lineage.{parent, accepted, branch, diff_ref}` | 演化轨迹 |
| 10 | 失败签名 | `failure.{signature, same_cause_rate}` | P1 可复现性 |

**外加追溯三元组**（不属于十类，但 `14` 的硬要求）：`run_id / commit / cmd_hash`，写在 `manifest.json`，
每条记录只存 `run_id` 外键。

---

## 九、命令面

```bash
# 环境自检：Docker / key / 工具版本 / 被装 agent，缺什么直接说
python -m exp doctor

# S0：K 分解（离线，不烧模型）
python -m exp k --agent terminus2 --root external/repos/terminal-bench --json

# S0：静态死组件（离线）
python -m exp dead --agent terminus2 --root external/repos/terminal-bench

# S1：一次 run（对齐 14 的目标命令）
python -m exp run --agent miniswe --bench local:v1 --backbone deepseek-v4.1-flash \
                  --seeds 3 --k 3 --gate k_veto --guard on --pool evolve --limit 5

# S1：δ 实测
python -m exp calib --agent miniswe --bench local:v1 --n 5

# S2：反事实删测
python -m exp delete --agent terminus2 --bench tb2:2.0 --n 10 --max-components 8 \
                     --whitelist on

# S3：四张表
python -m exp report runs/2026-11-01_m4 --out tables/ --bootstrap 100000

# S3：双池检查（会故意触发拦截，用于自证）
python -m exp pool --bench tb2:2.0 --list
python -m exp pool --bench tb2:2.0 --read quarantine   # 预期：抛 PoolViolation
```

**`doctor` 为什么是第一个命令**：`14` 的四种硬失败里没有一条是"环境不对"，
但实际施工第一个卡住的一定是 Docker 没起、key 没配、包没装。把它做成命令，
比写进 README 有效。

---

## 十、三个扩展点（判据就藏在这里）

| 要加什么 | 加在哪 | 需要改的现有代码 | 这就是 G1 的判据 |
|---|---|---|---|
| 新 agent | `adapt/<name>.py` + 注册表一行 | **0 行** | 连续加两个 agent 不用碰 `run/judge` → 达标 |
| 新基准 | `bench/<name>.py` + 注册表一行 | **0 行** | 换数据集不改主流程 |
| 新消融（A1–A4） | `config` 里加一个开关（`gate=k_record_only` / `guard=off` / `pool=single` / `delete=heuristic`） | 只在 `judge` 加一个分支 | 四组消融共用同一条 run 命令 |

**A1–A4 不是四套代码，是四个开关。** 如果为了做消融要复制流程，说明架构错了。

---

## 十一、测试与自证

9 条测试（清单见 `13_工具实现计划.md` §6）与能力的对应：

| 测试 | 背书的能力 | `14` 的原话 |
|---|---|---|
| `test_adapter_zero_change.py` | G1 | "新增一个 agent 只允许改一个新文件" |
| `test_k_reproducible.py` | G2 | "同一份 harness 连算两次，K 完全一致" |
| `test_dead_static.py` | G3-A | "死组件清单每条带证据" |
| `test_mutate_gates.py` | G3 | "删测无法复跑同一段"→ 必失败 |
| `test_three_state.py` | G3 | "'不可判'必须是独立取值" |
| `test_judge_k_veto.py` | G4 | "S+0.5、K+30% 的假候选必须拒" |
| `test_pool_guard.py` | G5 | "试图读量外池的测试用例，必须抛异常" |
| `test_tables.py` | G6 | "列名与论文模板一致" |
| `test_record_lineage.py` | 记录 | "每条记录带 run_id + commit + 命令哈希" |

---

## 十二、这套架构自己怎么被检验

一个架构如果不能用一句话否证，就是空话。这里给三个否证条件：

1. **如果新增第三个 agent 需要改 `run/`** —— 契约错了，`AgentAdapter` 少给了东西（大概率是 `Session` 的粒度）。
2. **如果删测表里 `undecidable` 恒为 0** —— 三态判据写错了，n 的约束没进代码。
3. **如果 A1 消融（K 只记账）跑不出与主臂不同的结论** —— 说明 K 在判定里根本没起作用，`14` 的整个主张塌掉。

第三条是真正的生死线：**A1 是"K 是不是装饰"的唯一证伪实验。**

---

## 出处

- 判据与硬失败：`12_工具目标与验收.md`
- 阶段与门槛：`13_工具实现计划.md`
- 十类字段：`00_总纲.md` §3.1
- 四张表列名：`08_测试计划.md` §7
- K 的口径与工具版本：`10_K的度量标准.md`
- 现有代码：`src/`（10,158 行 / 614 测试，本轮实测）
