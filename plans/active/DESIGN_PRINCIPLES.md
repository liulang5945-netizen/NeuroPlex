# 设计文档：设计原则与 Phase 1-8 历史记录

> **拆分文档**（2026-08-10）：从 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](BIO_INSPIRED_ARCHITECTURE_PLAN.md) 按内容拆分。
> 态极三大核心原则、与人脑机制的对应关系、以及 Phase 1-8 实施历史（大部分已完成或废弃）。
> 当前项目状态、路线图与接口梳理以主 plan 为准。
>
> **2026-08-21 解释边界**：本文是 Legacy NeuroPlex 的设计历史，不是 Taiji 新底座实现合同。文中“态极”一律指 **Legacy NeuroPlex**，与替代 Transformer 的新基底 Taiji 无关（词表见 [ARCHITECTURE_DIRECTION_2026_08.md](ARCHITECTURE_DIRECTION_2026_08.md) §0）。部分早期描述（例如抑制向量直接取反、外部触发新生、Transformer 树突扩展）已被后续代码或新决策取代。正式 Taiji 是顶层 TPF；新代码服从 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md) 的确定状态方程、局部学习和 N0–N11/M0–M6 反证门槛。

**内容**：
- 设计原则（三大核心原则 + 人脑对应）
- Phase 1: 基础生物学化（已实施）

---

> 以下为历史设计文档（Phase 1-8 实施记录），大部分已完成或废弃，保留供架构追溯。

### 设计原则

### 1.1 三大核心原则

1. **神经元差异性第一**：架构设计必须保留 per-neuron 个性，不以"减少参数"为首要目标
2. **自我进化能力**：架构不能限制态极进化到上千甚至上万神经元
3. **借鉴人脑神经科学**：兴奋/抑制分化、不应期、Hebbian 学习、突触可塑性、神经元凋亡/新生等

### 1.2 与人脑机制的对应关系

| 人脑机制 | 态极对应实现 |
|----------|-------------|
| 兴奋性/抑制性神经元分化 | neuron_type 字段，抑制性神经元 field_vector 取反 |
| 不应期 (Refractory Period) | refractory_counter，写入场后冷却 N 轮 |
| 突触可塑性 (STDP) | side_channels 权重的局部学习规则 |
| Hebbian 共激活组装 | CoactivationTracker EMA 矩阵 |
| 神经元凋亡与新生 | ApoptosisTracker 淘汰弱神经元 + NeurogenesisTrigger 检测信号（P7：创建由外部 train_neuron.py 执行） |
| 神经调质系统 | Neuromodulator 全局信号调节学习率 |
| 睡眠/记忆巩固 | consolidation_cycle() 离线重放 |
| 功能柱 (Cortical Column) | CorticalColumn 作为一等公民 |
| 注意力增益 | attention_beam 向量 boost 相关神经元 |
| 阈值可塑性 | per-neuron firing_threshold 自适应 |
| 域专用词汇输出 | domain-specific tokenizer + per-neuron 独立 lm_head（P7） |
| 从零独立训练 | 每 neuron 独立 embedding + body + lm_head，零教师依赖（P7-P8） |

---
### Phase 1: 基础生物学化 (已实施)

### 2.1 兴奋/抑制神经元分化

**人脑参考**：约 80% 谷氨酸能兴奋性神经元 + 20% GABA 能抑制性神经元。抑制性对防止癫痫、sharpening 信号、专注关键目标至关重要。

**实现**：
- `NeuronConfig.neuron_type: Literal["excitatory", "inhibitory"]`
- 抑制性神经元的 `field_vector` 在 `ResonanceNeuron.forward` 中取反 (`v = -v`)
- 场的 `write`/`update` 无需感知 neuron_type，向量已携带正负号

**文件**：`neuroplex/resonance/config.py`, `neuroplex/resonance/neuron.py`

### 2.2 不应期 (Refractory Period)

**人脑参考**：神经元发放后 1-2ms 绝对不应期 + 10ms 相对不应期，防止持续发放，强制信息分流。

**实现**：
- `ResonanceNeuron.refractory_counter` buffer
- `enter_refractory()`: 写入场后调用，设置冷却计数
- `tick_refractory()`: 每轮结束递减
- `in_refractory` 属性：round 2+ 中不应期神经元只读场不写场
- 旧贡献从场 state 中移除，避免"幽灵"影响

**文件**：`neuroplex/resonance/neuron.py`, `neuroplex/resonance/ensemble.py`

### 2.3 兴奋/抑制双通道

**人脑参考**：每个神经元接收兴奋性和抑制性突触输入，E/I 平衡是核心机制。

**实现**：
- `excite_channels`: 正向残差调制 (`h += 0.1 * proj(signal)`)
- `inhibit_channels`: 负向残差调制 (`h -= 0.1 * proj(signal)`)
- `establish_side_channel(peer_id, channel_type)` 支持双类型
- 同一 peer 可同时拥有两种通道，由 STDP 学习决定主导

**文件**：`neuroplex/resonance/neuron.py`

---

### Phase 2: 自我进化闭环

### 3.1 神经元凋亡 (Apoptosis)

**人脑参考**：弱连接神经元通过凋亡被清除，保持系统健康。

**实现**：
- `ApoptosisTracker` 追踪连续 K 次评估 PPL > threshold 的神经元
- 标记为"凋亡"后：
  - 从 ensemble 移除
  - 从磁盘删除 ckpt
  - 释放 neuron_id
  - 清理相关 side_channels
- 防止"僵尸神经元"占用资源

**触发条件**：
- 连续 3 次评估 PPL > 200（远超阈值）
- 或 compression_loss 持续 > 0.8（部落内无法代表）

### 3.2 神经元新生 (Neurogenesis)

**人脑参考**：海马齿状回成年后仍有新生神经元，填补知识盲区。

**实现**：
- `NeurogenesisTrigger` 检测"知识盲区"：
