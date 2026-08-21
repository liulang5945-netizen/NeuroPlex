# Taiji 架构方向决策

> 决策日期：2026-08-21
>
> 决策：Taiji 全面替代 Transformer 的计算职责，不作为 NeuroPlex 的成员插件。

## 1. 不可回退边界

1. `Taiji` 指完整原生架构，不指一个 cell、adapter、router 或 memory plugin。
2. 正式代码位于顶层 `taiji/`；`neuroplex/` 是冻结 Legacy 基线。
3. Taiji 自己定义输入表示、时间状态、上下文计算、学习、输出、生成和 checkpoint。
4. Taiji forward 不调用 tokenizer、Transformer、attention、KV cache、Cortex、ResonanceEnsemble 或 Legacy LM head。
5. 旧 1.5B 蒸馏、7.58M/10M 小 Transformer、5/9 成员装配都不能成为 Taiji 的身份。
6. Legacy 可做离线同预算对照，但不能向 Taiji 提供 hidden state、teacher logits 或运行时决策。

## 2. 正式算法名称

当前基础算法称为 **Taiji Predictive Fabric（TPF）**：

- raw event receptors；
- hierarchical reciprocal prediction；
- local recurrent transition；
- inhibitory/homeostatic dynamics；
- masked local plasticity；
- one motor organ；
- closed autoregressive action feedback。

公式与实现逐项对应见 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md)。

## 3. 包和兼容边界

`neuroplex/__init__.py` 不再把 `taiji` 全局映射为 `neuroplex`。历史 pickle 由 `neuroplex.legacy_checkpoint` 在受控作用域内加载，结束后恢复原生 Taiji 命名空间。

旧 `neuroplex/taiji/` 已删除，因为保留两套同名实现会继续让正式架构看起来像补丁。历史代码可从 Git 提交恢复，不在当前包中暴露。

## 4. 能力声明边界

Native v1 是完整可运行的非 Transformer 序列学习架构，但尚未证明语言能力、长程稳定、组合推理或 AGI。架构是否值得扩展由 N0–N10 反证门槛决定，不由“类脑”命名、参数规模或单个 demo 决定。

## 5. 当前唯一下一步

运行 N7 二阶上下文任务与 trace lesion，验证 TPF 是否真的用持久状态解决一阶转移无法解决的问题。
