# Taiji、当前 NeuroPlex 与生物神经系统的边界比较

> 本文只说明设计差距与可测映射，不宣称工程模块等价于脑区，也不把“类脑”当成 AGI 证明。目标底座规范见 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md)。

## 1. 最关键的底层差异

| 维度 | 当前 NeuroPlex | Taiji 目标 | 生物系统启发 |
|---|---|---|---|
| 基本计算 | 每个成员对 token 窗口运行 Transformer | 细胞对事件持续更新内部状态 | 神经活动是连续、历史依赖的动力学 |
| 时间 | 固定轮次或围绕整次 forward 的连续积分 | 全局 tick、事件 delay、异步稀疏活动 | 多时间尺度和传播延迟 |
| 状态 | 主要 hidden 在 forward 后丢弃；task field 每次 reset | cell/field/topology/memory 都跨事件持续 | 状态本身承载短期认知 |
| 学习 | AdamW、CE/NLL、LoRA 和睡眠训练为主 | eligibility × 调质的局部在线更新，睡眠快到慢巩固 | 局部可塑性受全局调质门控 |
| 记忆 | 外部 field 向量库、DialogueState、文本记忆分裂 | 工作、联想、情景、语义、程序和稳态记忆统一所有权 | 多记忆系统协同 |
| 活动 | 活动成员通常重复完整前向 | 有事件才发放，受误差、目标、相位、能量调度 | 稀疏、竞争、稳态调节 |
| 输出 | 每个成员有 LM/judge 头，再融合或选 leader | 独立 motor population 竞争并提交动作 | 感觉、联络、运动回路分工 |
| 环境 | 对话输入到文本输出，反馈链分散 | 动作改变环境，结果成为下一感觉和学习信号 | 闭环感知—行动 |

## 2. 哪些现有机制值得保留思想

| 现有机制 | 可保留的思想 | 为什么需要重写 |
|---|---|---|
| Resonance field | 群体共享状态、兴奋/抑制和贡献可观测 | 当前 field 生命周期绑定一次调用，非原生持续记忆 |
| Phasor/Oscillator | 相位绑定、节奏和时间门控 | 当前主要调制 Transformer 参与度，非 cell 原生状态 |
| STDP/coactivation | 局部时序信用和拓扑生长 | 当前“时间”多为共振轮次，更新 side-channel 缩放 |
| Neuromodulator | 广播奖励/警觉/稳态信号 | 当前主要缩放 attention、FFN、LR 和写场 |
| Sleep/replay | 经验重放、巩固、修剪和生长 | 当前学习对象仍是 LM head/LoRA/Transformer read path |
| Lifecycle | 成熟、隔离、复活、凋亡 | 指标要换成 Taiji 的因果贡献、能量、误差和 lesion 效果 |

## 3. 不应做的类比

- 一个 `TaijiCell` 不是一枚生物神经元；它更接近可学习的工程微回路。
- `TaijiField` 不是“意识”本身，只是可测的群体共享状态。
- phase lock 不等于注意或绑定已经形成；必须通过干预实验验证因果作用。
- replay 不等于理解；必须证明它改善未来预测/行动而不是只降低训练损失。
- 自主修改参数不等于自主目标，更不等于 AGI。

## 4. AGI 仍缺什么

即使 Taiji 的全部底座门槛通过，仍需独立解决：

- 可持续的真实环境与身体反馈；
- 目标形成、冲突解决和长期信用分配；
- 世界模型与真实/想象边界；
- 社会学习、语言发展和课程；
- 可控的自我修改、安全边界与价值稳定；
- 跨规模后仍成立的效率和稳定性。

因此 Taiji 的近期目标不是“宣布 AGI”，而是用干预实验回答：持续状态、局部学习、内生记忆和群体动力学是否比同预算序列模型形成更强的适应与协作。

## 5. 当前可验证指标

- 状态保留/重置 lesion 的因果差值；
- 单次经验后的局部适应速度；
- 顺序学习后的遗忘率；
- 活动稀疏度、能量和数值稳定性；
- 1-cell 与 3-cell 的组合任务差值；
- field/memory/cell lesion 后的能力下降；
- 真实事件与 imagined/replay 事件的来源正确率；
- save/load 后下一 tick 的因果连续性；
- 动作改变环境并反向改善预测的闭环增益。
