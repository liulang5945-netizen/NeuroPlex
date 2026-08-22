# Seed 架构方向决策：Taiji 替代 Transformer 底层

> 决策日期：2026-08-21（Seed 命名迁移 2026-08-22）
>
> 决策：项目和模型是 **Seed**；**Taiji 是 Seed 的底层基底，全面替代 Transformer 的计算职责**，不作为 Legacy NeuroPlex 的成员插件。

## 0. 规范词表（唯一口径）

“Taiji / 太极 / 态极”在历史文档里被用于五种不同含义。此后只允许下表左列的写法：

| 规范名 | 指代 | 代码/文件事实 |
|---|---|---|
| **Seed** | 项目与模型级主体 | 顶层 `seed/`；分发名 `seed`；拥有模型组合与 `seed-native-v1` checkpoint envelope |
| **Taiji / Taiji Predictive Fabric（TPF）** | Seed 的**底层基底**，替代 Transformer | 顶层 `taiji/` 9 个模块；当前 checkpoint line Native v6；不导入 `seed`、`neuroplex` 或 `transformers` |
| **Legacy NeuroPlex** | 冻结的 Transformer 基线（9 个成员） | `neuroplex/` 包（113 文件 / 36420 行）；底层 Transformer 是 `neuroplex/layers.py::TransformerBlock`，live 消费点 3 处（见下） |
| **`taiji.*`（历史 import 别名）** | `neuroplex/` 的旧包名 | 只在历史 pickle 与 `scripts/archive/` 中出现；由 `neuroplex/legacy_checkpoint.py` 在受控作用域内临时映射 |
| **`taiji` / `taiji_model`（历史 HTTP 路径与指标名）** | Legacy 应用兼容契约，**不定义 Seed/Taiji 新边界** | 在 Seed 原生服务路径完成前保持兼容；新增 API 必须使用 Seed 命名 |
| ~~态极~~ | Legacy NeuroPlex 的旧中文称呼，**不指新基底** | 冻结代码内仍有 202 处 / 55 文件（日志与用户文案），不改名；**新文档与新代码禁止使用**，需要指代时写 “Legacy NeuroPlex” |

被替代的边界是明确的：Taiji 顶掉 `neuroplex/layers.py::TransformerBlock` 承担的计算职责，而不是顶掉 `api/`、`neuroplex/life/` 等外围工程层。

该 Transformer 底层当前的 **live 消费点恰好 3 处**（`scripts/archive/` 冻结层不计）：

| 消费点 | 性质 |
|---|---|
| `neuroplex/resonance/neuron.py:25` | Legacy 基线自身的构成部分 |
| `scripts/training/train_tinystories.py:26` | **有意保留**的纯 Transformer 对照实验（验证 training pipeline 正确性） |
| `scripts/training/train_tinystories_field.py:32` | 同上，field 变体 |

这份名单由 `tests/taiji_native/test_naming_boundary_contract.py` 按 import 语句（AST，非文本匹配）强制封闭：新增任何消费点都会让 CI 失败，必须先在本文件记录“为什么还要在被替代的底层上继续投入”。

## 1. 不可回退边界

1. `Seed` 指模型主体；`Taiji` 指完整原生底层基底，不指 cell、adapter、router 或 memory plugin。
2. 模型代码位于顶层 `seed/`，基底代码位于顶层 `taiji/`；`neuroplex/` 是冻结 Legacy 基线。
3. Taiji 自己定义输入表示、时间状态、上下文计算、学习、输出、生成和 substrate checkpoint；Seed 只组合公开合同。
4. Taiji forward 不调用 tokenizer、Transformer、attention、KV cache、Cortex、ResonanceEnsemble 或 Legacy LM head。
5. 旧 1.5B 蒸馏、7.58M/10M 小 Transformer、5/9 成员装配都不能成为 Taiji 的身份。
6. Legacy 可做离线同预算对照，但不能向 Taiji 提供 hidden state、teacher logits 或运行时决策。

## 2. 正式算法名称与组成

基础算法称为 **Taiji Predictive Fabric（TPF）**。Native v5 是其当前可执行参考实现：

- raw event receptor population；
- hierarchical reciprocal prediction error；
- local recurrent transition；
- inhibitory/homeostatic state dynamics；
- fast activity + slow trace；
- balanced sparse cortical receptor bank；
- shared motor evidence and one action organ；
- compressed existing-edge local plasticity；
- closed autoregressive action feedback；
- atomic cognition checkpoint。
- compressed fixed-fan-in edge execution。
- pending action eligibility + reward-modulated local policy learning。
- fixed-population distributed episodic field + recurrent pattern completion。
- novelty/reward-gated cue/action/outcome/time/episode/provenance binding。
- resonance-gated motor evidence + one-tick cortical memory feedback。

公式、张量形状、精确 tick 顺序、局部更新和代码映射见 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md)。这些内容构成实现合同，变更状态顺序或张量语义必须升级 state/checkpoint 版本并重新通过 N0–N11/M0–M5。

## 3. 本轮结构决策：公共运动感受器

动作单元不能各自读取随机且不同的皮层子空间，否则 softmax 比较的是不同证据；也不能共同只取一个 48/224 坐标子集，否则有效上下文会被结构性丢弃。Native v5 固定采用平衡单 fan-out receptor map：全部皮层 activity/trace 坐标各进入一个公共运动通道，257 个动作共享全部 48 个通道。场 readout 同样先进入共享 `K_m` 通道再比较动作证据。

## 4. 包和兼容边界

`neuroplex/__init__.py` 不再把 `taiji` 全局映射为 `neuroplex`。历史 pickle 由 `neuroplex.legacy_checkpoint` 在受控作用域内加载，结束后恢复原生 Taiji 命名空间。

旧 `neuroplex/taiji/` 已删除。历史代码可从 Git 提交恢复，不在当前包中暴露。

`scripts/archive/` 里 98 个文件的 301 处 `from taiji.<legacy>` 属于历史别名（含义＝`neuroplex`），在当前包布局下会误解析到新基底 `taiji/`。处置口径：**不重写、不改名**，因为其依赖的 Legacy 符号与数据路径本身已不存在（`scripts/archive/architecture_verification.py:8-10` 已自证），重写只会产出可导入但不可运行的假活代码。风险已被界定：`scripts/archive/` 无 `test_*.py`，pytest 不收集；CI 只跑 `tests/taiji_native` 与 `tests/`；无任何在用代码引用该目录。判定依据写在 `scripts/archive/README.md`。

### 4.1 是否删除 Legacy NeuroPlex（`neuroplex/`）

判定：**现在不删**。不是出于工作量，而是因为删除会同时摧毁两样东西：对外服务层，以及本项目核心主张的举证能力。实测依赖事实：

| 方向 | 实测 | 含义 |
|---|---|---|
| `taiji/` → `neuroplex` | **0** | 新基底自足，删除不影响基底本身 |
| `taiji/` → `transformers` | **0** | 基底与 Transformer 生态无关 |
| `neuroplex/` → `taiji` | **0** | 替代关系单向，基线是干净参照系 |
| `neuroplex/` → `transformers` | **0** | HF transformers 只是 `legacy` extra，未实际 import |
| `api/` → `neuroplex` | **40+ import 行**（5 个路由文件，多为函数内懒加载） | 删除即整个服务层失效 |
| `tests/` → `neuroplex` | **13 import 行**（10+ 文件） | 删除即丢失这批回归 |
| `scripts/training/` → `neuroplex` | **100+ import 行**（约 50 个诊断/对照脚本） | 删除即丢失全部对照实验能力 |

两条不可忽视的理由：

1. **举证依赖对照。** 本项目的核心主张是“Transformer 不能达到目标、Taiji 能”。这是一个**比较性命题**，它的证据形式必然是同预算对照。删掉 Transformer 基线，等于删掉唯一的对照臂——此后 Taiji 的任何指标都变成无参照的绝对数字，无法反驳“换个 Transformer 配置也能做到”。`scripts/training/train_tinystories.py` 正是为此有意保留的纯 Transformer 对照（其 docstring 自陈“目标：验证训练 pipeline 是否正确”）。
2. **`api/` 的 `taiji` 是对外契约。** `api/` 内约 60 处 `taiji` 是 HTTP 路径与 Prometheus 指标名（`/api/taiji/*`、`taiji_requests_total`），已被外部消费者依赖。它们既不指新基底也不指 Legacy，删除或改名属于破坏性变更。

**允许删除的前置条件**（全部满足才重新评估，缺一即维持不删）：

1. Taiji 通过语言能力反证门槛，并在**同预算**下给出优于 Legacy 基线的实测结果（对照结论一旦落定并归档，对照臂才失去价值）；
2. `api/` 已有一条不经 `neuroplex/` 的 Taiji 原生服务路径，且旧路由的对外契约有迁移或兼容方案；
3. `tests/` 中依赖 `neuroplex` 的 13 处已迁移或明确废弃；
4. `scripts/training/` 的对照结论已归档到 plans，脚本不再是唯一证据载体。

**在此之前的正确做法是"冻结"而非"删除"**：`neuroplex/` 不接受新功能，只做不改变行为的修复；边界由 `tests/taiji_native/test_naming_boundary_contract.py` 强制——新增 Transformer 底层消费点会让 CI 失败。冻结保留了对照能力和回滚余地，删除则不可逆。

## 5. 能力声明边界

Native v5 是完整可运行的非 Transformer 感知—状态—情景—行动参考架构，已通过在线学习、128 步自由回灌、二阶上下文、固定延迟 trace、真实按边执行、主动环境和八条 one-shot 跨 episode 情景反证。它尚未证明大容量记忆、巩固、语言能力、组合推理或 AGI。后续仍由可反证门槛决定，不由“类脑”命名、参数规模或单个 demo 决定。

## 6. 当前唯一下一步

M6 内生 replay/巩固和覆盖修复已落地；signed-opponent + replay-winner resource 的 12-seed 离线反证也已通过。当前唯一下一步见 [SEED_ARCHITECTURE.md](SEED_ARCHITECTURE.md) §6：把 K64 shared-support、waking signed baseline 与 bout-local winner resource retention `0.9` 实现为新的 Taiji 原生 checkpoint line。本文件只维护决策与命名边界。
