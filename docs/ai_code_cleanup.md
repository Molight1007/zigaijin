# AI 代码的删减与简化经验

2026-11-06 ｜ 对应模块：`evo/domain/ai_cleanup.py`、`evo ai-cleanup`、`map_architecture.ai_cleanup`

---

## 0. 为什么单列这一页

本工具的大部分代码是 AI 写的。AI 代码的典型问题不是“写不出来”，而是：

- 为了防御不存在的调用方，加一层层校验和 try/except；
- 一个逻辑只调用一次，却抽成 private helper；
- 不同 CLI/命令复制同一段 JSON 读取、路径检查、错误包装；
- 为了“兼容”保留旧命令、旧 dataclass、旧状态机；
- 为了“可读性”写只转发一层调用的 wrapper；
- 测试只覆盖 helper，不覆盖生产路径，删了也没人发现。

这些不是靠“代码风格”能解决的，必须靠**引用证据 + 运行证据 + 净代码口径**。

---

## 1. 检测器能自动找的四类

`evo ai-cleanup --root <dir>` 输出 `findings`，每条含 `kind/path/span/symbol/reason/action/confidence`：

| kind | 判据 | 建议动作 | 置信度 |
|---|---|---|---|
| `trivial_wrapper` | private 函数体只有 `return helper(...)` | inline | medium |
| `single_call_helper` | private 函数在全仓只有 1 个 `Name` 调用点 | inline/delete（先内联再删） | low |
| `duplicate_body` | 两个函数参数名相同、函数体 AST 相同且 ≥3 条语句 | merge | medium |
| `unread_cli_option` | `add_argument("--x")` 声明了，但没有任何 `args.x` / `getattr(args, "x")` 读取 | delete option | medium |

检测器只给**候选**，不自动删。原因是：

- `single_call_helper` 可能是公开 API、动态调用、测试或跨模块入口；
- `duplicate_body` 可能是两个领域概念碰巧同形，合并会耦合；
- `unread_cli_option` 可能被 `**vars(args)` 或外部脚本使用。

---

## 2. 我们自己的扫描结果（2026-11-06）

扫描 `evo/` 得到 111 条候选（DeepSeek-native 接线后重扫）：

- `single_call_helper`：101
- `trivial_wrapper`：10
- `duplicate_body`：0
- `unread_cli_option`：0

这说明当前 AI 代码的主要冗余不是“重复函数”，而是**单次调用的小 helper**。其中很多是有意保留的命名边界（例如 `_read_source`、`_k_block`），不能一刀切内联；真正的删除要结合调用方、测试和可读性判断。

### 已经实际删除的

| 删除项 | 规模 | 依据 |
|---|---|---|
| `evo/domain/diagnosis.py` + `tests/test_diagnosis.py` | 1 模块 + 测试 | 生产代码零引用；`application/diagnose.py` 已有实现 |
| `evo/infra/runtime.py` + `tests/test_runtime.py` + `evo/infra/` | 1 模块 + 测试 | 生产代码零引用；工具未接任何路径 |
| `evo/application/loop.py` + `tests/test_loop.py` | 559 行实现 + 389 行测试 | 已被 `evo workforce --loop` 取代；CLI `loop` 删除 |

净效果：**少了一套重复的多轮闭环、两个测试专用模块**；`workforce --loop` 成为唯一多轮入口。

---

## 3. 删减规则（按优先级）

1. **先证据，后删除**：`grep` 必须包含同文件内调用；动态导入（`COVERAGE_AGENT` 字符串）不能按零引用删。
2. **先内联，再删 helper**：单调用 helper 先内联到调用点，跑测试，再删函数；不要直接删导致调用点残留。
3. **只删生产零引用**：只有测试引用的模块是候选；删模块时同步删测试，避免“测试在保护死代码”。
4. **旧命令先有替代**：`loop` 是因为 `workforce --loop` 已覆盖才删；没有替代就不删。
5. **净代码口径**：只看 `K_lines_nb` / `K_ast`，不把删注释、删空行算成果。
6. **不删公开契约**：CLI 参数、dataclass 字段、`__all__`、入口函数先当公开 API；要删先确认没有外部脚本/论文复现依赖。
7. **一次只删一类**：先删测试专用模块，再删旧命令，再内联 helper；每步跑全量测试。

---

## 4. AI 代码里最常见的“可简化点”

| 模式 | 例子 | 处理 |
|---|---|---|
| 复制 JSON 读取 | 多个 `try: json.loads(path.read_text())` | 合并成一个 loader，统一报错 |
| 复制路径检查 | 每个 CLI 命令重复 `if not path.is_file()` | 用共享 `_load_json_object` |
| 单调用 helper | `_normalize_newlines`、`_batch_rows`、`_now` | 内联或保留（如果命名确实解释意图） |
| 兼容 wrapper | 只为旧参数保留的 `_cmd_loop` | 有替代就删 |
| 过度校验 | private 函数对内部调用再查类型 | 删到系统边界（CLI/外部输入）即可 |
| 测试专用模块 | `domain/diagnosis`、`infra/runtime` | 删模块 + 测试 |
| 重复状态机 | 每轮自己写 state/events | 统一用 `RunStore`/journal |
| 猜测性抽象 | 只有一个实现的 Protocol/Factory | 删到需要第二个实现时再加 |

---

## 5. 和简化通道的关系

- `map_architecture` 会把 `ai_cleanup` findings 放进 `architecture.json`，总结员可以看到这些“可删/可简化”候选。
- 这些 findings **不是自动删除候选**；删除仍走 `scan`/`apply`/`workforce` 的保护与审核门。
- 对“删除代码”的度量仍用 `K_lines_nb`/`K_ast`；对“简化代码”的度量仍用 `simplify-effect`。

**一句话**：AI 代码最容易多出来的是“防御层”和“单次 helper”；删它们要靠引用证据、运行证据和净代码口径，不靠感觉。
