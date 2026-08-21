# NeuroPlex 架构方向决策：Taiji 替代 Transformer 底层

> 决策日期：2026-08-21（命名收敛 2026-08-21）
>
> 决策：项目是 **NeuroPlex**；**Taiji 是 NeuroPlex 的新底层基底，全面替代 Transformer 的计算职责**，不作为 Legacy NeuroPlex 的成员插件。

## 0. 规范词表（唯一口径）

“Taiji / 太极 / 态极”在历史文档里被用于五种不同含义。此后只允许下表左列的写法：

| 规范名 | 指代 | 代码/文件事实 |
|---|---|---|
| **NeuroPlex** | 整个项目 | 仓库本身；`pyproject.toml` 的分发名仍是历史遗留 `taiji-neuron`，不改（会破坏已装环境与 CI） |
| **Taiji / Taiji Predictive Fabric（TPF）** | NeuroPlex 的**新底层基底**，替代 Transformer | 顶层 `taiji/` 9 个模块；`Native v5` 是当前参考实现；不导入 `neuroplex` 或 `transformers` |
| **Legacy NeuroPlex** | 冻结的 Transformer 基线（9 个成员） | `neuroplex/` 包；底层 Transformer 就是 `neuroplex/layers.py::TransformerBlock`，唯一消费点 `neuroplex/resonance/neuron.py:25` |
| **`taiji.*`（历史 import 别名）** | `neuroplex/` 的旧包名 | 只在历史 pickle 与 `scripts/archive/` 中出现；由 `neuroplex/legacy_checkpoint.py` 在受控作用域内临时映射 |
| ~~态极~~ | Legacy NeuroPlex 的旧中文称呼，**不指新基底** | 冻结代码内仍有 196 处（日志与用户文案），不改名；**新文档与新代码禁止使用**，需要指代时写 “Legacy NeuroPlex” |

被替代的边界是明确的单点：Taiji 顶掉 `neuroplex/layers.py::TransformerBlock` 承担的计算职责，而不是顶掉 `api/`、`neuroplex/life/` 等外围工程层。

## 1. 不可回退边界

1. `Taiji` 指完整原生底层基底，不指 cell、adapter、router 或 memory plugin。
2. 正式代码位于顶层 `taiji/`；`neuroplex/` 是冻结 Legacy 基线。
3. Taiji 自己定义输入表示、时间状态、上下文计算、学习、输出、生成和 checkpoint。
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

## 5. 能力声明边界

Native v5 是完整可运行的非 Transformer 感知—状态—情景—行动参考架构，已通过在线学习、128 步自由回灌、二阶上下文、固定延迟 trace、真实按边执行、主动环境和八条 one-shot 跨 episode 情景反证。它尚未证明大容量记忆、巩固、语言能力、组合推理或 AGI。后续仍由可反证门槛决定，不由“类脑”命名、参数规模或单个 demo 决定。

## 6. 当前唯一下一步

M6 内生 replay/巩固已落地并 5/5 seed 通过（提交 `52162c0`）。当前唯一下一步见 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](BIO_INSPIRED_ARCHITECTURE_PLAN.md) §6：修复 replay 选择覆盖不均。本文件只维护决策与命名边界，不再复制下一步内容。
