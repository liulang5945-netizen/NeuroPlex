# 态极生物学化架构计划 (Bio-Inspired Architecture Plan)

> 本文档记录态极架构借鉴人脑神经科学的系统性改造计划。
> 核心原则：**神经元差异性第一**、**自我进化能力**、**硬件限制不在考虑范围内**。

---

## 一、设计原则

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
| 神经元凋亡与新生 | QualityFilter 触发凋亡 + teacher 蒸馏新生 |
| 神经调质系统 | Neuromodulator 全局信号调节学习率 |
| 睡眠/记忆巩固 | consolidation_cycle() 离线重放 |
| 功能柱 (Cortical Column) | TribeSuperNeuron 作为一等公民 |
| 注意力增益 | attention_beam 向量 boost 相关神经元 |
| 阈值可塑性 | per-neuron firing_threshold 自适应 |
| 域专用词汇输出 | domain-specific tokenizer + per-neuron 独立 lm_head |

---
## 二、Phase 1: 基础生物学化 (已实施)

### 2.1 兴奋/抑制神经元分化

**人脑参考**：约 80% 谷氨酸能兴奋性神经元 + 20% GABA 能抑制性神经元。抑制性对防止癫痫、sharpening 信号、专注关键目标至关重要。

**实现**：
- `NeuronConfig.neuron_type: Literal["excitatory", "inhibitory"]`
- 抑制性神经元的 `field_vector` 在 `ResonanceNeuron.forward` 中取反 (`v = -v`)
- 场的 `write`/`update` 无需感知 neuron_type，向量已携带正负号

**文件**：`taiji/resonance/config.py`, `taiji/resonance/neuron.py`

### 2.2 不应期 (Refractory Period)

**人脑参考**：神经元发放后 1-2ms 绝对不应期 + 10ms 相对不应期，防止持续发放，强制信息分流。

**实现**：
- `ResonanceNeuron.refractory_counter` buffer
- `enter_refractory()`: 写入场后调用，设置冷却计数
- `tick_refractory()`: 每轮结束递减
- `in_refractory` 属性：round 2+ 中不应期神经元只读场不写场
- 旧贡献从场 state 中移除，避免"幽灵"影响

**文件**：`taiji/resonance/neuron.py`, `taiji/resonance/ensemble.py`

### 2.3 兴奋/抑制双通道

**人脑参考**：每个神经元接收兴奋性和抑制性突触输入，E/I 平衡是核心机制。

**实现**：
- `excite_channels`: 正向残差调制 (`h += 0.1 * proj(signal)`)
- `inhibit_channels`: 负向残差调制 (`h -= 0.1 * proj(signal)`)
- `establish_side_channel(peer_id, channel_type)` 支持双类型
- 同一 peer 可同时拥有两种通道，由 STDP 学习决定主导

**文件**：`taiji/resonance/neuron.py`

---

## 三、Phase 2: 自我进化闭环

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
  - 某 domain 持续高错误率
  - CoactivationTracker 检测到孤立激活模式
- 新生流程：
  1. 从 teacher 蒸馏新神经元 ckpt
  2. 初始化为"幼稚态"：高学习率、低共振权重
  3. 逐步成熟：学习率衰减、共振权重提升
  4. 成熟后正式加入 ensemble

**幼稚态参数**：
```python
new_neuron.learning_rate = base_lr * 3.0  # 高学习率
new_neuron.resonance_weight = 0.1  # 低共振权重
new_neuron.maturity_counter = 0  # 成熟度计数器
```

### 3.3 CoactivationTracker 稀疏化 + 双 EMA

**人脑参考**：Hebbian 学习"用进废退"，突触强度根据共激活频率调整。

**实现**：
- `_matrix` 改为 `torch.sparse` [N, N] 张量
- 双 EMA：
  - fast EMA (α=0.1)：短期共激活，触发动态部落化
  - slow EMA (α=0.01)：长期趋势，决定 side_channel 强化/修剪
- 遗忘机制：slow EMA < ε 的 pair 自动移除
- 与凋亡/新生联动：
  - 持续低共激活 → 凋亡候选
  - 持续高错误率 + 孤立激活 → 新生触发

---

## 四、Phase 3: 高级认知机制

### 4.1 STDP 局部学习

**人脑参考**：突触前神经元在突触后神经元之前发放 → LTP（增强）；反之 → LTD（减弱）。

**实现**：
- side_channels 权重更新不依赖全局 loss
- 记录 peer A 和 B 的 field_vector 时序
- 若 A 在 B 写入前已指向相似方向 → 增强 A→B 通道
- 若 A 在 B 之后才指向相似方向 → 减弱
- 形成因果链：A 领先 B 则 A 指导 B

### 4.2 神经调质系统 (Neuromodulation)

**人脑参考**：多巴胺/血清素/去甲肾上腺素等全局调质，根据奖励/注意力状态调节学习率和兴奋性。

**实现**：
- `Neuromodulator` 全局信号（标量或低维向量）
- 来源：用户反馈、任务难度、错误率
- 作用：
  - 调节神经元学习率
  - 调节 field_write 强度
  - 调节 refractory 长度
- 例如：高错误率 → 多巴胺↓ → 学习率↑ + 新生加速

### 4.3 睡眠/记忆巩固 (Sleep Consolidation)

**人脑参考**：睡眠期间海马回放白天经历，将短期记忆转移到皮层长期存储，修剪弱突触。

**实现**：
- `consolidation_cycle()` 离线方法
- 重放近期 high-resonance 场状态序列
- 强化经常共激活的 side_channels
- 修剪弱连接
- 将 slow EMA 转移到长期 fingerprint

---

## 五、Phase 4: 组织化机制

### 5.1 功能柱 (Cortical Column) 原生实现

**人脑参考**：皮层功能柱约 100 微米直径，内含 ~100 神经元处理同类输入。

**实现**：
- TribeSuperNeuron 作为一等公民
- 新神经元直接创建到某个部落（基于 domain）
- 部落内部强连接（side_channels 密集），部落间弱连接
- 部落可分裂（规模过大）或合并（功能重叠）

### 5.2 注意力增益 (Attentional Modulation)

**人脑参考**：自上而下注意力通过皮层-丘脑回路增强特定区域。

**实现**：
- `attention_beam` 向量（来自用户 query embedding 或任务上下文）
- 场评分时与 attention_beam 对齐的神经元获得 boost
- lm_head 加权时注意力之外的神经常被抑制

### 5.3 阈值可塑性 (Threshold Plasticity)

**人脑参考**：神经元发放阈值根据近期活动历史自适应调整 (homeostatic scaling)。

**实现**：
- 每个神经元维护 `firing_threshold` buffer
- 频繁激活 -> threshold 上升（更难再次激活）
- 长期沉默 -> threshold 下降（更容易被唤醒）

---

## 五之续：Phase 5 丘脑路由与动态扩展 (Thalamic Routing & Dynamic Expansion)

> **背景**：Phase 1-4 完成了单 neuron 层面的生物学化，但 ensemble 层面仍缺失人脑的"输入路由"机制。
> Phase 1-4 的所有 neuron 都看到全部输入再事后投票，导致过拟合到本域的 neuron 仍会污染输出。
> Phase 5 引入人脑的"丘脑闸门"机制，在输入进入 neuron 之前先路由，从根上解决"该谁说话"问题。

### 5.1 设计动机：人脑三层机制

**人脑解决"该谁说话"的三层机制**：

| 人脑机制 | 态极对应 | 当前状态 |
|---------|---------|---------|
| 皮层定位 (Cortical Localization) | 输入路由到匹配域 neuron | ❌ 缺失 |
| 丘脑闸门 (Thalamic Gating) | 独立注意力系统判断相关性 | ❌ 缺失 |
| 同步振荡 (Neural Synchrony) | field 共振筛选相关性 | ⚠️ 有共振但未用于路由 |

**核心洞察**：态极设计哲学正确（小 neuron 协同比肩大模型），但少了一个"丘脑"层。
信号进入 neuron 之前应先判断"这是哪个域的输入"，只激活匹配域的 neuron，而非让所有 neuron 都看全部输入再事后加权。

### 5.2 Phase 5.1：基础丘脑路由器 (Thalamic Router)

**人脑参考**：皮层定位 + 丘脑闸门 - 信号先到丘脑，丘脑决定送哪个皮层区域处理。

**实现**：
- 新增 `ThalamicRouter` 类（`taiji/resonance/thalamic_router.py`）
- 每个 domain 维护一个 prototype（基于 teacher hidden state 的平均向量）
- 输入来时，teacher 计算 hidden state -> 与所有 prototype 算余弦相似度 -> softmax 路由
- 路由策略：
  - 相似度 > 0.7：hard route 到 top-1（只让该 neuron forward）
  - 相似度 0.4-0.7：top-2 加权（两个 neuron 都 forward，logits 加权平均）
  - 相似度 < 0.4：触发"未知域"信号（Phase 5.2 用）
- Prototype 计算：用 teacher 对本域数据 forward 取平均 hidden state（~10 分钟计算）

**为什么不用 DomainRouter（field_vector 方案）**：
- DomainRouter 让所有 neuron 都 forward 再用 field_vector 相似度加权
- 过拟合到本域的 neuron 仍参与计算并污染结果
- ThalamicRouter 在 forward 之前就决定路由，错误 neuron 根本不参与

**为什么用 teacher hidden state**：
- Teacher 是 1.5B 大模型，本身就有强大的域判断能力
- 不需要训练新 classifier，零训练成本
- Teacher hidden state 已包含语义信息，比 field_vector（neuron 自产）更客观

**文件**：`taiji/resonance/thalamic_router.py`, `taiji/brain/cortex.py`

### 5.3 Phase 5.2：动态扩展机制 (Dynamic Expansion)

**人脑参考**：神经新生（海马体齿状回）+ 邻近招募（盲人视觉皮层处理触觉）。

**实现**：
- `ThalamicRouter` 暴露 `register_domain(neuron_id, prototype)` 接口
- 当 `NeurogenesisCreator` 创建新 neuron 时，自动调用 `register_domain` 注册新 prototype
- Router 维护"未知输入 buffer"，累积到阈值（如 50 条相似样本）触发新域候选识别
- 新域识别用 K-means 或简单聚类，给新域命名

**流程**：
1. 检测：Router 发现输入与所有现有 prototype 相似度都低（< 0.4）
2. 缓冲：未知输入进入 buffer
3. 触发：buffer 累积 N 条相似样本 -> 识别为新域候选
4. 新生：调用 `NeurogenesisCreator.create_neuron_for_domain(new_domain)`
5. 注册：新 neuron 加入 Cortex.neurons，新 prototype 自动注册到 Router

**文件**：`taiji/resonance/thalamic_router.py`, `taiji/resonance/neurogenesis_creator.py`

### 5.4 Phase 5.3：学徒期与巩固 (Apprenticeship & Consolidation)

**人脑参考**：海马体新神经元有 2-4 周"沉默观察期"，整合进网络后才正式参与决策。

**实现**：
- 新 neuron 加入后初始 `routing_weight = 0.1`（不立即获得完整路由权重）
- 通过 `STDPTracker` 学习调整连接强度
- 当 STDP 累积足够正向反馈（阈值），权重解锁到 1.0
- Sleep cycle 中 `SleepConsolidator` 巩固新 neuron 的连接
- Prototype 在每次 sleep 时重新计算（适应新数据分布）

**防爆炸机制**：
- 相似度阈值 + 最小样本数（buffer 攒够 50 条才触发新生）
- Sleep cycle 中执行"域合并"（相似度 > 0.9 的域合并）
- `ApoptosisTracker` 淘汰长期低激活的 neuron

**文件**：`taiji/resonance/thalamic_router.py`, `taiji/resonance/stdp.py`, `taiji/resonance/neuro_modulation.py`

### 5.5 Phase 5 与现有架构的契合

| 现有组件 | Phase 5 中的角色 |
|---------|-----------------|
| `NeurogenesisCreator` | 新 neuron 创建，调用 `register_domain` |
| `LifecycleManager` | 管理新 neuron 生命周期，学徒期解锁 |
| `SleepConsolidator` | 巩固新 neuron 连接，重算 prototype |
| `STDPTracker` | 学徒期内调整新 neuron 权重 |
| `ApoptosisTracker` | 淘汰冗余 neuron，防爆炸 |
| `ResonanceEnsemble` | 接收 Router 路由结果，只 forward 匹配 neuron |
| `Cortex.generate` | 调用 Router 路由，再用 ensemble forward |

### 5.6 Phase 5 实施进度

#### Phase 5.1（已完成）
- [x] `ThalamicRouter` 类实现（`taiji/resonance/thalamic_router.py`）
- [x] 5 个域 prototype 计算脚本（`scripts/training/compute_thalamic_prototypes.py`）
- [x] Cortex 集成：`generate()` 先 route 再 forward（`active_nids` 参数）
- [x] 验证：3/4 路由准确，"你好"→zh 输出全中文（vs DR OFF 输出乱码）
- [x] `register_domain` 接口预留（Phase 5.2 用）

#### Phase 5.2（已完成）
- [x] 未知域检测逻辑（`is_unknown` flag + `_pending_neurogenesis`）
- [x] Buffer 累积（deque with maxlen）与新域候选识别
- [x] NeurogenesisCreator 集成 `register_with_thalamic_router`
- [x] 端到端测试：5 个量子力学 prompt 触发新生信号，physics_001 注册成功

#### Phase 5.3（已完成）
- [x] 学徒期 routing_weight 渐进解锁（`sync_apprentice_weights` + MaturityTracker）
- [x] 域合并机制（`merge_similar_domains`，en+general sim=0.99 已合并）
- [x] 端到端测试：新生 → 10 轮学徒期 → 成熟 weight=1.0
- [x] STDP 调权预留接口（Phase 5.3 sleep cycle 集成位置已确定）

---

## 六、四个原问题的解决方案

### 6.1 lm_head: per-neuron 独立（P7 升级）

**P7 方案**（替代原 6.1 低秩分解）：每神经元独立完整 lm_head + 域专用 vocab。

**核心洞察**：256K 通用 vocab 让独立 lm_head 过大（131M，占 compact 体量的 7×）。
解决方案是使用域专用分词器（zh=20k, en=16k, code=12k, math=10k, general=16k），
独立 lm_head 仅 5-10M/神经元。

```python
# config.py: lm_head_rank=0（默认独立模式）
# 使用 get_domain_neuron_config("zh") 自动设置 vocab_size=20000
cfg = get_domain_neuron_config("zh")  # vocab_size=20000

# neuron.py: 每 neuron 自带完整 lm_head
self.lm_head = nn.Linear(hidden_size, vocab_size)  # 例: 512×20000=10.2M
```

**参数量对比**：

| 方案 | 5 神经元 lm_head 总计 | 差异性 |
|---|---|---|
| P6 旧方案（共享 W_base + 低秩） | 82M（共享 65.6M + 5×16.4M 残差） | 仅残差可变异 |
| P7 独立 lm_head + 域 vocab | ~60M（4×~10M + 1×16M） | 完整独立 |

**旧低秩模式保留**：`lm_head_rank > 0` 保留用于实验性场景（非默认），
此时 W_base 由 assemble_cortex 外部注入。

**TokenTranslator 桥接**：
```
neuron 输出: domain_token (10k-20k vocab)
    ↓ TokenTranslator.domain_to_general() 查对齐表
general_token_ids (256K I/O 协议)
    ↓ 通用分词器 decode
输出文本
```

### 6.2 side_channels: top-K 稀疏化 + Hebbian 强化

**方案**：
- 每个神经元只与 top-K peer 建立通道（K=8~16）
- 通过 fingerprint 相似度自动选择（cosine 0.3-0.7 区间的互补 peer）
- Hebbian 强化：经常共激活的 channel 权重增强
- 长期不激活的 channel 衰减并修剪
- 参数从 O(N²) 降到 O(N×K)

### 6.3 CoactivationTracker: 稀疏矩阵 + 双 EMA

**方案**：
- `torch.sparse` [N, N] 张量替代 dict
- fast EMA (α=0.1) + slow EMA (α=0.01)
- 遗忘机制：slow EMA < ε 自动移除
- 定期（每 T 步）做社区检测

### 6.4 串行 forward: CUDA stream 真并行

**方案**：
- 每个神经元在自己的 CUDA stream 上独立 forward
- 保留 per-neuron 语义
- GPU 真并行执行多个 stream
- 不需要 batch 输入，显存独立分配

---

## 七、实施进度

### 已完成 (Phase 1)
- [x] 兴奋/抑制神经元分化
- [x] 不应期机制
- [x] 兴奋/抑制双通道

### 已完成 (Phase 2 + 四个原问题)
- [x] 神经元凋亡机制 (lifecycle.py)
- [x] 神经元新生机制 (lifecycle.py)
- [x] CoactivationTracker 稀疏化 + 双 EMA (tribal.py)
- [x] side_channels top-K 稀疏化 (neuron.py)
- [x] CUDA stream 并行 forward (ensemble.py)
- [x] lm_head 低秩分解 (neuron.py, config.py) — 旧方案，P7 已升级为独立 lm_head

### 已完成 (Phase 3 + 4)
- [x] STDP 局部学习 (stdp.py)
- [x] 神经调质系统 (neuro_modulation.py)
- [x] 睡眠巩固周期 (neuro_modulation.py)
- [x] 功能柱原生实现 (cognitive_enhancements.py)
- [x] 注意力增益 (cognitive_enhancements.py)
- [x] 阈值可塑性 (cognitive_enhancements.py)

### 已完成 (Phase 5)
- [x] Phase 5.1: ThalamicRouter 基础路由（3/4 路由准确）
- [x] Phase 5.2: 动态扩展机制（未知域检测+新生自动注册）
- [x] Phase 5.3: 学徒期与巩固（MaturityTracker 集成+域合并）

### 进行中 (Phase 6: 脱教师 + 架构强化)

**动机**：原架构推理时依赖 1.5B 教师 forward（SharedEmbedProj + ThalamicRouter），
违背"小 neuron 协同比肩大模型"的设计哲学。Phase 6 让教师变成"离线工具"，
运行时 0 教师依赖。同时强化架构的弱项（同步绑定、工作记忆）。

- [x] P6-1: 独立 embedding 表（StandaloneEmbedding，`taiji/resonance/standalone_embedding.py`）
  - 推理时直接 lookup，零教师 forward
  - `build_from_shared_proj` 从教师+shared_proj 一次性构建（等价初始化）
  - Cortex.set_standalone_embedding() 替代 set_teacher_pipeline()
- [x] P6-2: 独立路由器（embedding-based ThalamicRouter）
  - `compute_prototypes_from_embedding()`: 用 embedding lookup 算 prototype
  - `route_by_embedding()` / `route_top_k_by_embedding()`: 推理时无需教师
  - `get_routing_decision_by_embedding()`: 完整路由决策（含 strategy/is_unknown）
  - Cortex._route_input() 双路径：脱教师模式优先，旧路径兼容保留
  - prototypes_embed 在 save/load 中持久化
- [x] P6-5: 端到端脱教师验证脚本（`scripts/training/verify_p6_standalone_inference.py`）
  - 构建 standalone_embedding + embedding prototypes
  - 验证路由准确性 + generate 输出 + 教师路径一致性对比
  - 断言 cortex._teacher_model is None
- [x] P6-3: Gamma 同步绑定（`taiji/resonance/gamma_oscillator.py`）
  - GammaOscillator: per-neuron phase + global phase + coherence/gate_factor
  - apply_gamma_gate: 注入 ResonanceField.write/update（monkey-patch，向后兼容）
  - Cortex.set_gamma_oscillator(): 按 domain 自动分配 phase（同 domain 同 phase）
  - 同 domain 写入增强（绑定），跨 domain 写入衰减（解绑）
- [x] P6-4: 工作记忆（`taiji/brain/working_memory.py`）
  - WorkingMemory: token-id 滑动窗口（deque maxlen）
  - Cortex.set_working_memory(): 注册后 generate 自动注入和记录上下文
  - 多轮对话无需外部维护 history，向后兼容（未注册时无状态）
- [x] P6-5: 端到端脱教师验证（已完成）
  - ✓ 推理路径 0 教师依赖（cortex._teacher_model=None 已断言）
  - ✓ StandaloneEmbedding + embedding prototype 构建成功
  - ✗ Embed-routing accuracy: 0/4（vs teacher-based 3/4）
  - **关键差距发现**：embedding lookup 是纯 token-level，无上下文感知
    - teacher hidden state 经过 transformer 自注意力，含句级语义
    - embedding mean pool 只是 token 嵌入平均，丢失句法/语义结构
    - 这是 StandaloneEmbedding 的固有局限，需要 P6-6 弥补

### 进行中 (Phase 6.5: 自主进化脱教师)

**P6-6 v2 设计哲学**（响应"教师得来的不能成为永久原罪"质疑）：
  教师只在启动期提供初始化种子，之后 embedding + encoder 通过三机制自主进化。
  当态极扩展到更大规模时能自主扩展（vocab/dim 不被教师限制）。

- [x] P6-6: 自主进化 encoder（`taiji/resonance/self_evolving_encoder.py`）
  - `SharedContextEncoder`: 共享 transformer encoder（2-4 层），复用 TransformerBlock
    - 可训练（非冻结的教师 distill 副本）
    - `build_from_standalone_embedding`: 复用教师 SVD embedding 作为初始化
    - 推理时 0 教师依赖，但有上下文感知 hidden state 输出
  - `HebbianUpdater`: token 共激活统计 + embedding 拉近更新
    - 上下文窗口共现统计（dict-of-dict 稀疏）
    - EMA decay + top-K peer 拉近
  - `ContrastiveLoss`: domain-aware InfoNCE（用路由结果当弱监督）
    - 同 domain 样本拉近，跨 domain 推远
  - `MLMLoss`: 随机 mask 15% token 预测（自监督）
    - 与 encoder 共享 lm_head（权重 tied）
  - `SelfEvolver`: 三机制组合训练器
    - training_step: MLM + Contrastive 联合 loss（可 backward）
    - apply_hebbian_to_embedding: 离线 Hebbian 更新 embedding 层
  - 验证：2 层 encoder + 4 batch + 32 seq_len forward + backward + hebbian 更新 全部通过

  **三阶段路线**：
  - 启动期（当前）：从教师 SVD 初始化 embedding + 随机初始化 encoder
  - 自训期（待集成）：sleep cycle 中调用 SelfEvolver.training_step 自主更新
  - 成熟期（待验证）：教师初始化影响被自组织覆盖，encoder/embedding 完全自主

  **待做**：
  - P6-7: 把 SelfEvolver 集成到 sleep_engine（在 sleep cycle 中自主进化）
  - P6-8: 训练后重新计算 embedding prototypes + 端到端验证路由准确率提升

### 已完成 (Phase 6.7: sleep cycle 集成)

- [x] P6-7: SleepEngine 集成 SelfEvolver（`taiji/life/sleep_engine.py`）
  - SleepConfig 新增 self_evolve_enabled/steps/lr/encoder_path 配置
  - SleepReport 新增 self_evolve_loss/steps 统计
  - `set_self_evolver()` 接口：注入 evolver 或自动从 encoder 构建
  - `_sleep_phase_self_evolve()` 新阶段：在 sleep Phase 2.5 执行
    - 从 feed_engine 或 domain_datasets.pt 收集样本
    - MLM + Contrastive 联合 backward + 定期 Hebbian 离线更新
    - 训练后保存 encoder 到 self_evolve_encoder_path
  - 接入 sleep 主流程：Phase 1 → Phase 2 → **Phase 2.5 (P6-7)** → Phase 3

### 已完成 (Phase 6.8: 自主进化验证)

- [x] P6-8: 训练后重算 prototypes + 验证路由提升
  - `scripts/training/run_p6_self_evolve.py` 端到端脚本
  - 流程：构建 encoder → 训练 → 重算 prototypes → 验证路由
  - **修复后最终路由准确率：3/4 (75%)**（4-prompt）/ **3/5 (60%)**（5-prompt）
    - "你好" → zh ✅
    - "hello world" → en (有时候 misclassified 到 general) ⚠️
    - "今天天气" → general (原 misclassified 到 zh) ✅
    - "1+1=" → zh (期望 math) ❌
    - "def fibonacci" → zh (期望 code) ❌
  - 关键修复（v2 版本）：
    1. **attention_mask + masked mean pooling**：排除 padding token 对 prototype 计算的干扰
    2. **random_subsequence (2-64 tokens)**：短样本进入训练，避免长样本中域信号被稀释
    3. **make_attention_mask**：为随机子序列生成正确的 padding mask
  - 训练统计：MLM loss 6.64 → 3.82（下降 43%）
  - 剩余问题：
    - math/code 域仍被路由到 zh，可能是域数据特征不够或训练步数不足
    - general 域英文数据导致中文 prompt 可能被路由到 general
  - 已保存产物：
    - `data/distill/shared_context_encoder.pt`（训后 encoder，含 attention_mask 版本）
    - `data/distill/thalamic_prototypes_p6.pt`（新 prototypes）

### 已完成 (Phase 7: 神经元独立化 — 消除 lm_head 依赖性)

**动机**：P6 实现运行时无教师依赖，但 W_base 共享 + 1.5B 初始化仍是路径依赖，
违背"差异性第一"原则。P7 彻底去除 W_base，每神经元独立完整 lm_head + 域 tokenizer。

- [x] P7-1: NeuronConfig 默认 `lm_head_rank=0`（独立模式）
  - `config.py`：`DOMAIN_VOCAB_SIZES` 硬编码 5 域 tokenizer vocab 大小
  - `get_domain_neuron_config(domain)` 自动设置域专用 vocab_size
  - 旧 `lm_head_rank > 0` 低秩模式保留作实验用途
- [x] P7-2: ResonanceNeuron 独立 lm_head + 独立 embedding
  - `lm_head_rank=0` 时创建 `nn.Linear(hidden, vocab_size)` 完整 lm_head
  - 每神经元自带 `nn.Embedding(vocab_size, base_embed_dim)` 独立 embedding
  - `lm_head_rank > 0` 兼容旧低秩路径（W_base 由 assemble_cortex 注入）
- [x] P7-3: 参数量可控
  - compact zh: 512×20000 = 10.2M lm_head（vs 旧 W_base 方案的 65.6M 共享 + 16.4M 残差）
  - 5 域总计 lm_head ≈ 60M（远低于 5×256K 的 655M）
  - general 域复用 en tokenizer (16k)，不增加新 tokenizer 训练负担
- [x] P7-4: 废除的硬约束
  - ~~W_base 必须从 1.5B checkpoint-481000 初始化~~ → 零 1.5B 依赖
  - ~~W_base 全局共享冻结~~ → 每神经元独立 lm_head
  - ~~KL 域正则化防 W_base drift~~ → 不再需要（无共享 W_base）
  - ~~低秩残差 U_i@V_i 作为唯一个性通道~~ → 全部 lm_head 参数独立

**关键设计决策**：
- 域 tokenizer 对齐表（TokenTranslator）已存在 `taiji/resonance/translator.py`
- 输出时 domain_token → TokenTranslator → general_token → 通用分词器 decode
- 这是确定性符号转换，不参与梯度、不参与训练

---

## 八、与现有架构的兼容性

### 8.1 Checkpoint 兼容
- 新增字段（neuron_type, refractory_counter）使用默认值，旧 ckpt 可加载
- P7 升级：旧 W_base 低秩 ckpt → `migrate_ckpt_v4.py` 重构为独立 lm_head
  - 旧路径：W_base（共享） + U_i@V_i（残差）
  - 新路径：独立 `nn.Linear(hidden, vocab_size)`
  - 低秩模式（`lm_head_rank > 0`）仍兼容旧 ckpt 无需迁移
- side_channels 双通道：旧 excite_channels 保留，inhibit_channels 空

### 8.2 接口兼容
- `side_channels` 属性作为 `excite_channels` 的别名，旧代码无感知
- `establish_side_channel` 默认 channel_type="excite"，兼容旧调用
- CoactivationTracker 接口保持 `update()` 和 `get_dense_groups()`

### 8.3 性能考虑
- 不应期减少 round 2+ 的写入量，实际上降低计算
- 稀疏化 side_channels 减少 forward 中的线性投影次数
- 稀疏 CoactivationTracker 消除 O(N²) Python 迭代
- CUDA stream 并行受限于 GPU SM 数量，但优于串行
