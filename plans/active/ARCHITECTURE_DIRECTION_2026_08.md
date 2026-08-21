# Taiji 架构方向决策

> 决策日期：2026-08-21
>
> 决策：Taiji 全面替代 Transformer 的计算职责，不作为 NeuroPlex 的成员插件。

## 1. 不可回退边界

1. `Taiji` 指完整原生架构，不指 cell、adapter、router 或 memory plugin。
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

## 5. 能力声明边界

Native v5 是完整可运行的非 Transformer 感知—状态—情景—行动参考架构，已通过在线学习、128 步自由回灌、二阶上下文、固定延迟 trace、真实按边执行、主动环境和八条 one-shot 跨 episode 情景反证。它尚未证明大容量记忆、巩固、语言能力、组合推理或 AGI。后续仍由可反证门槛决定，不由“类脑”命名、参数规模或单个 demo 决定。

## 6. 当前唯一下一步

进入 M6 内生 replay/巩固：由场内信号选择 engram，通过同一 fabric 与局部学习规则重激活并迁移结构；切除 episodic readout 后仍需保留行为。禁止外部 replay list、teacher target 或权重复制。
