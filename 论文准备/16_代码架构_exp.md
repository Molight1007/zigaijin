# 代码架构 v2：自改进 agent（`exp`）

2026-11-01 ｜ v2。相对 v1 的改动见 `18_反思_从平台到自改进agent.md`；v1 的批判见 `17_架构批判_自审.md`。

> **`exp` 现在读作 experimenter（实验者），不是 experiment（实验）。**
> CLI 名沿用 `python -m exp`，是为了让 `14` 的验收命令原样可用——**改的是身份，不是命令。**

---

## 〇、一句话

> **一个以自己为改造对象的 agent：它观察自己的身体（harness），选一个算子，提出候选，
> 真跑一遍，过判定闸门，采纳或回滚，把结果写进记忆。**
> 删代码（`simplify`）是它的算子之一，也是唯一**自带因果证据**的那个。

```
        ┌──────────────────────────────────────────────────┐
        │                 agent.py  主循环                  │
        │  observe → propose → trial → evidence → gate      │
        │      ↑                          ↓                 │
        │   archive（记忆）  ←──── commit / rollback        │
        └───────┬──────────────────────────┬───────────────┘
                │                          │
        ┌───────▼────────┐        ┌────────▼────────┐
        │  ops/  算子     │        │  body/  自己的身体 │
        │ simplify ★     │        │  (harness 目录)   │
        │ rewrite/tune/  │        └────────┬────────┘
        │ augment/route  │                 │
        └───────┬────────┘        ┌────────▼────────┐
                │                 │  env/  场地与判分 │
                └────────────────▶│ local / tb2      │
                                  └─────────────────┘
```

**与 v1 的根本差别**：v1 里 agent 在 `adapt/`（被测对象，只读），工具在 `exp/`（观察者）；
v2 里 agent **就是** `exp/`，它要改的东西在 `body/`（它自己的身体），**读写都在自己身上**。

---

## 一、五条设计原则

| # | 原则 | 含义 | 违反会长什么样 |
|---|---|---|---|
| 1 | **主体是 agent** | 它自己决定下一步改什么；平台设施（记录/判定/报表）是它的观测层 | 做成一堆命令，人决定跑什么 → 那就是平台 |
| 2 | **功能 = 算子** | 每个改进能力实现同一个 `Op` 协议，注册在 `ops/` | 删代码硬编码成流程 → 加第二个功能无处安放 |
| 3 | **能改自己** | 算子改 `body/`，采纳后 `body/` 前进 | 只能改别人仓库的副本 → 是"改进器"，不是"自改进 agent" |
| 4 | **K 离线算** | K 是源码的静态函数，永不进 rollout 循环 | K 随运行漂移 → 不可比，"可复算"塌掉 |
| 5 | **不可判是一等公民** | `undecidable` / `undeletable` 是独立取值，任何路径不许折叠成 `delete` | 三态变两态 → 核心创新归零 |

---

## 二、目录

```
exp/
  __init__.py
  __main__.py                # python -m exp
  cli.py                     # run / step / k / dead / delete / report / replay / doctor
  contracts.py               # ★ 唯一契约源（Component / State / Candidate / Evidence / Record）
  reuse.py                   # ★ 对 src.domain 的唯一引用点
  errors.py                  # PoolViolation / MutateError / BenchUnavailable / BudgetExceeded

  agent.py                   # ★ 主循环（observe→propose→trial→evidence→gate→commit）
  state.py                   # HarnessState：身体 + 组件 + K + S/C + 谱系
  archive.py                 # ★ 记忆：candidates / incumbent / rejected
  gate.py                    # 全局闸门（K 硬否决 · δ 地板 · 三维支配）
  delta.py                   # δ 实测（只升不降）

  ops/                       # ── 算子（功能）：一个功能一个条目 ──
    __init__.py              #   OPS 注册表：name -> Op
    base.py                  #   Op 协议 + Candidate 构造工具
    simplify/                #   ★ 创新点：删代码（唯一自带因果证据）
      __init__.py            #     对外只出 propose() / evidence()
      k.py                   #     K 度量：六类 × 五指标
      split.py               #     客观切分（不依赖谁自报）
      dead.py                #     静态死组件：AST 可达性 ∩ 运行时覆盖
      mutate.py              #     物化删除 + 三道门（语法/引用/冒烟）
      counterfactual.py      #     逐组件删 → 真跑
      whitelist.py           #     防御式豁免（assert/try/参数校验）
      evidence.py            #     四态判定 + 证据表行
    rewrite.py               #   stub：改 prompt 措辞
    tune.py                  #   stub：调参数
    augment.py               #   stub：加 middleware / 工具
    route.py                 #   stub：换底座档位

  body/                      # ★ 它自己的身体（从上游 fork 的 harness 目录）
                             #   算子改这里；采纳后前进；旧状态进 archive

  env/                       # ── 场地与判分 ──
    base.py                  #   Bench + Session（能力协商）
    local.py                 #   tasks/v1.jsonl（30 题，本机目录）
    tb2.py                   #   Terminal-Bench 2.x（Docker + pytest）

  feedback/                  # ── 用户反馈入口（课题名里的"闭环"） ──
    ingest.py                #   失败样本 → Task → evolve 池

  io/
    record.py                # 轨迹记录（十类字段 + 追溯三元组）
    replay.py                # 轨迹重放（中强度复现）
    store.py                 # runs/<id>/ 读写

  report/
    tables.py                # 四张表
    stats.py                 # bootstrap（默认 100,000）
```

**规模**：实现约 4,500–5,000 行，测试约 900 行。比 v1 少——因为 `simplify` 内聚后少了一层胶水。

---

## 三、四个核心契约

### 3.1 身体与状态

```python
@dataclass(frozen=True)
class Component:
    id: str                  # "prompt::system" / "tool::bash" / "code::loop.execute"
    kind: str                # prompt | tools | middleware | memory | subagent | skills
    locator: Locator         # 怎么找到它、怎么删它
    k: KVector
    deletability: str       # block | statement | atom（决定能不能单独删）

@dataclass(frozen=True)
class HarnessState:
    rev: str                        # body/ 的目录哈希（= 这一版的身份证）
    root: Path
    components: tuple[Component, ...]
    k_total: int                    # Σ k3_tokens
    score: Score | None             # 空 = 还没测
    parent: str | None              # 谱系
```

**`rev` 为什么是目录哈希而不是 git commit**：`body/` 可能不是 git 仓库（自建 harness）。
用目录内容哈希，天然支持"任意来源的身体"，且**可验证**（两个 rev 相同 = 身体相同）。

### 3.2 算子

```python
@dataclass(frozen=True)
class Prediction:
    """提案时自己说的预期 —— 下一轮要被证伪。
    没有它，archive 只能记"改了什么"，记不了"它当时怎么想的"。"""
    dk: float                # 预期 K 变化（相对）
    ds: str                  # "no-change" | "up" | "down"
    why: str                 # 一句话理由（进论文的编辑历史表）

@dataclass(frozen=True)
class Candidate:
    id: str
    op: str
    parent_rev: str
    mutation: Mutation              # 一次只改一处（R3）
    prediction: Prediction
    evidence: Evidence | None       # 算子自带的因果证据；simplify 有，其他为 None

class Op(Protocol):
    name: str
    def propose(self, state: HarnessState, budget: Budget) -> list[Candidate]: ...
```

**`Mutation` 只允许一条**：这是 R3（一次只改一处）的代码化。
想改两处 → 两个候选 → 两次判定。**没有"批量改"的入口。**

### 3.3 证据（`simplify` 的专利）

```python
@dataclass(frozen=True)
class Evidence:
    kind: str            # "counterfactual-delete" | "static-dead" | "none"
    verdict: str         # keep | delete | undecidable | undeletable
    s_before: float
    s_after: float | None
    ci: tuple[float, float] | None
    n: int
    reason: str          # undeletable 时必须写原因（dep-broken / syntax-broken / smoke-failed）
    trace: str           # 证据落盘路径（删测表"证据"列指到这里）
```

**四态定义写死在这里**，任何调用方不许自己判：

| 判定 | 条件 |
|---|---|
| `keep` | `s_after < s_before − δ` |
| `delete` | `s_after ≥ s_before − δ` 且 `ΔK ≤ −阈值` 且 `n ≥ required_n(σ, 0.1)` |
| `undecidable` | ① `n < required_n`；② `ΔS` 的 95% CI 覆盖 0 |
| `undeletable` | 变体过不了语法/引用/冒烟三关门 —— **不是"没用"，是"删不掉"** |

`undeletable` 不进分母。论文里要单独报"可删组件 / 总组件"这个比例——它本身就是 agent 模块化程度的度量。

### 3.4 记录

```python
@dataclass(frozen=True)
class Record:
    run_id: str; task_id: str; pool: str
    rev: str                        # 哪一版身体
    candidate_id: str | None        # 哪一次改动（基线为 None）
    score: Grade
    usage: Usage                    # tokens(分 cached/output) / steps / seconds
    k: KSnapshot                    # total + by_kind + per_component
    evidence: Evidence | None
    verdict: Verdict | None
    lineage: Lineage | None
    failure: FailureSig | None
    trace: str
```

---

## 四、主循环（`agent.py`）

```python
def run(agent: Agent, budget: Budget, ops: list[Op]) -> Trajectory:
    state = agent.observe()                      # 读 body/ → 切分 → 算 K → 可选测 S
    while not budget.exhausted():
        op = agent.pick(ops)                     # 选算子（v1：按优先级轮转）
        for cand in op.propose(state, budget):   # 候选由算子生成
            if archive.seen(cand.fingerprint()):  # 记忆：试过就别再试
                continue
            trial = agent.trial(cand, state)      # ★ 真跑（env/ 完成）
            ev    = cand.evidence                 # 算子自带（simplify 有）
            v     = gate.judge(trial, state, ev, delta, pool)
            archive.add(cand, trial, ev, v)
            if v.accepted:
                state = agent.commit(cand, trial) # body/ 前进
                break
        budget.charge(...)
    return archive.trajectory()
```

**三个关键点**：

1. **`gate.judge` 是所有算子的闸门**，不是删测专用。
   - `simplify` 的候选：K 必然降 → 闸门自动放行，但它在**提案阶段**已经过了自己的证据关。
   - `rewrite` / `augment`：可能让 K 涨 → **这里是 K 硬否决唯一真正生效的地方**。
   - 这是 v1 那个"K 否决是死代码"的批判的正确解法：**不是删掉否决，是给它正确的消费者**。
2. **`archive.seen()`** 防重复试错。没有它，"自改进"退化成随机搜索。
3. **串行**。测试本身要求串行（`14` 非目标：单机串行即可）。

---

## 五、五个算子：谁是创新点，为什么

| 算子 | 做什么 | ΔK 方向 | 自带因果证据 | v1 实现 | 别人做过 |
|---|---|---|---|---|---|
| **`simplify`** | 删组件 | **↓ 必然** | **✅ 反事实删测** | **完整** | RRSI 的 pruner 只报不删 |
| `rewrite` | 改 prompt 措辞/结构 | ~ 或 ↑ | ❌ 只有分数 | stub | RRSI 主路 |
| `tune` | 调参数（步数、上下文阈值、压缩比） | ~ | ⚠️ 可 A/B | stub | HarnessPro |
| `augment` | 加 middleware / 工具 | **↑** | ❌ | stub | AHE（增益主要来源） |
| `route` | 换底座档位 | — | ❌ | stub | 工程常规 |

**这张表就是论文"创新点"的准确位置**：

> 五个算子里，**只有 `simplify` 的 K 方向是确定的**，也只有它能说"我删掉的东西真的没用"——
> 其余四个只能拿出"分数变了"，拿不出"为什么变"。

所以论文的贡献句应该是：

> 我们给出一个多算子自改进 agent，其中**删除算子**用反事实删测为 K 的下降提供因果证据，
> 并用 K 硬否决约束其余算子的膨胀倾向。

—— 而不是"我们做了一个能自改进的 agent"（那是 RRSI/DGM 已占的位置）。

---

## 六、`ops/simplify/` 的内部（创新点的完整落点）

七个子模块，一条闭合的证据链：

```
split.py      客观切分：从 body/ 的目录结构 + AST 自动切六类组件
   ↓
k.py          每个组件算 KVector（五指标），汇总 K_total
   ↓
dead.py       路线 A：AST 可达性 ∩ sys.monitoring 运行时覆盖 → "从未执行"清单
   ↓
whitelist.py  剔除防御式代码（assert / try / 参数校验 / 边界检查）
   ↓
mutate.py     路线 B 前置：物化删除 + 三道门（语法 / 引用闭合 / 冒烟）
   ↓           过不了 → Evidence(undeletable, reason=...)
counterfactual.py   真跑 n 次，得 s_after / c_after
   ↓
evidence.py   四态判定 → Evidence（含 CI、n、trace 路径）
```

**为什么必须内聚在一个包**：这条链的每一环都只为下一环服务，
拆到三层（measure / remove / judge）就会变成"三个模块的产物"，而不是"一个功能的证据"。
`17` 的第 4 条批判说的就是这个。

**对外只出两件事**：`propose(state, budget) -> list[Candidate]`（带 `Evidence(kind="static-dead")` 的前置证据）
和 `evidence(cand) -> Evidence`（删测后的判定）。其余都是内部实现。

---

## 七、两层判定

```
候选 ──┬─ 算子是 simplify ──▶ ① 证据关（四态）──┐
       │                                        ├──▶ ② 全局闸门 ──▶ 采纳 / 拒绝
       └─ 其他算子 ─────────────────────────────┘
```

| 层 | 在哪 | 判什么 | 失败模式 |
|---|---|---|---|
| ① 证据关 | `ops/simplify/evidence.py` | 这次删除可信吗（四态） | n 不够 → `undecidable`；删不动 → `undeletable` |
| ② 全局闸门 | `gate.py` | 这个候选该不该被采纳 | K 涨 → 拒；掉分超 δ → 拒；被存档支配 → 拒 |

**①不通过，候选根本不该被提出来**——但为了留痕，仍然记进 archive（标 `rejected-by-evidence`）。
**②对所有算子一视同仁**，这是"K 是第三目标"的代码化。

---

## 八、与 v1 的差异（七处）

| # | v1 | v2 | 为什么 |
|---|---|---|---|
| 1 | `adapt/` 装别人的 agent，只读 | `body/` 是自己的身体，可读写 | "自"字 |
| 2 | 删除是唯一流程 | `ops/` 注册表，5 个算子 | "众多功能之一" |
| 3 | `harness_code.apply()` 抛 PermissionError | 算子改 `body/`，采纳后前进 | 同 1 |
| 4 | K/删测/判定散在三层 | `ops/simplify/` 内聚 | 创新点要有一个目录 |
| 5 | `gate` 唯一判定 | 两层：证据关 + 全局闸门 | K 否决的消费者是其他算子 |
| 6 | 无演化循环 | `agent.py` + `archive.py` | 不会决定下一步的不叫 agent |
| 7 | 反馈只有 benchmark | `feedback/ingest.py` | 课题名有"闭环" |

**原样保留的四条**（v1 里已经对的）：
`Session` 能力协商 · 切分客观化 · `KVector` + `MetricSpec` 版本钉死 · 四态（原三态扩一条）。

---

## 九、复现：两种强度，不要混着写

| 类型 | 复现方式 | 强度 | 论文里怎么说 |
|---|---|---|---|
| **评测**（给定身体跑基准） | 命令 + rev + seed + 配置哈希 + MetricSpec | **强**（字节级可重算） | 主结果表每格 |
| **轨迹**（agent 自己选路） | 记录 rev 链 + 每步候选 + 判定 + 证据 | **中**（同种子下决策可重演，外部服务非确定） | 演化轨迹图 |

**这是 v2 必须付的代价**：agent 自己决定下一步，就没有"一条命令复现整条轨迹"这回事。
`18` 第 9 遍反思记的就是这个。**如实写，别混。**

---

## 十、八条 ADR（对 v1 的修订）

| # | 决策 | 理由 |
|---|---|---|
| D1 | 包名沿用 `exp`，身份改称 experimenter | 让 `14` 的验收命令原样可用；改名收益为零 |
| D2 | `simplify` 内聚成一个包，不拆三层 | 创新点要有一个可指认的落点 |
| D3 | 其余四个算子只留 stub | `14` 非目标"不自研 agent 框架"；做全五个 = 自研演化框架 |
| D4 | `Mutation` 只允许一条 | R3 的代码化；批量改会让判定失去归因能力 |
| D5 | K 硬否决放在全局闸门 | 它的消费者是 `rewrite`/`augment`，不是 `simplify` |
| D6 | 切分只由 `split.py` 定，算子不自报组件 | 否则跨 agent 的 K 分布量的是"切得细不细" |
| D7 | `Session` 用能力协商（`shell`/`fs`/`tmux`） | Terminus-2 要的是有状态终端，无状态 `exec` 装不下 |
| D8 | `rev` 用目录哈希，不用 git commit | `body/` 可能不是 git 仓库；内容哈希更普适且可验证 |

---

## 十一、测试（10 条）

| 测试 | 背书什么 |
|---|---|
| `test_op_registry.py` | 注册第三个算子不用改 `agent.py`（"众多功能之一"的判据） |
| `test_agent_loop.py` | 主循环能在一轮内完成 propose→trial→gate→commit，且 archive 有 4 条记录 |
| `test_self_modify.py` | 采纳后 `body/` 真的变了（rev 变），回滚后变回来 |
| `test_k_reproducible.py` | 同输入两算，`KVector` 全字段相等 |
| `test_split_objective.py` | 两个人按同一目录约定写边界，切出的组件数相同 |
| `test_dead_static.py` | 死组件清单每条带证据 |
| `test_mutate_gates.py` | 删出语法错误 → `undeletable`，且不记成 `delete` |
| `test_four_state.py` | n=3 时必出 `undecidable`；`undeletable` 不进分母 |
| `test_gate_k_veto.py` | `rewrite` 候选 S+0.5/K+30% → 拒；`simplify` 候选 S 不变/K−20% → 收 |
| `test_end_to_end.py` | 真删一个组件、真跑 3 题、真出表、真写 run 目录 |

**第 10 条是 v1 缺的那条**——只测形状的测试全绿，也可能一张表都出不来。

---

## 十二、这套架构怎么被否证

1. **加第三个算子要改 `agent.py`** → `Op` 协议少给了东西。
2. **删测表里 `undecidable` 恒为 0** → n 的约束没进代码。
3. **A1 消融（K 只记账）与主臂结论一致** → K 在闸门里没起作用，论文主张塌。
4. **`simplify` 的可删组件比例 < 10%** → 身体的模块化程度不足以支撑删测，得换身体（选 mini-swe-agent 这类极简的）。

---

## 出处

- 定位（v2 的依据）：`18_反思_从平台到自改进agent.md`
- 判据与硬失败：`14_工具目标与验收.md`
- v1 的批判与处置：`17_架构批判_自审.md`
- 十类字段：`00_总纲_对标_测试_数据.md` §3.1
- 四张表列名：`10_论文测试计划与功能优化.md` §7
- K 的口径：`12_K的度量标准调研.md`
