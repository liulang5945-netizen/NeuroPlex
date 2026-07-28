# 态极生物学化架构计划 (Bio-Inspired Architecture Plan)

> 本文档记录态极架构借鉴人脑神经科学的系统性改造计划。
> 核心原则：**神经元差异性第一**、**自我进化能力**、**硬件限制不在考虑范围内**。

> **📋 项目主 plan（活跃维护）**
> 本文档是项目的状态总览和路线图。其他 plans/ 文件：
> - `TRAINING_REFERENCE.md` — 训练准则参考（非 plan）
> - `BODY_LIFE_BRAIN_INTEGRATION_PLAN.md` — body/life/brain 子系统设计参考
> - `archive/` — 历史归档（COMPREHENSIVE v1.0、H1-H8 修复记录、side-channels 实现细节）

---

## 🚨 紧急更新（2026-07-26）：架构级 bug 修复 — 之前所有训练无效

### bug 发现

**ResonanceNeuron.forward 的因果掩码缺失**：调用 `block.attention(h_normed)` 时没传
`mask` 参数，导致 GQA 的 `is_causal=(mask is not None) and (seqlen>1)` 为 False →
**双向注意力**（位置 K 能看到所有未来 token）。

### 决定性诊断证据

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| Shift teacher-forcing（完整序列 forward） | 100% | 0% |
| 逐步预测（只给前 K 个 token） | 0% | 0% |
| 完整 forward vs 逐步 forward 一致性 | 12/12 不一致 | ✅ 一致 |

- 修复前 100% 是**假象**（模型偷看未来 token）
- 修复后 0% 是**真实性能**（现有权重依赖偷看，学坏了）

### 影响（之前所有结果无效）

- ❌ PPL 1.79~3.85：虚低（偷看未来使预测虚高准确）
- ❌ argmax 73~74%：虚高（同样原因）
- ❌ Teacher-forcing 100%：假象
- ❌ 所有神经元 ckpt（zh_j0~j9, zh_leader0, zh_simple0, zh_par1, zh_par2）：权重学坏
- ❌ "数据不足是根因"的结论：错误（数据是次要因素，架构 bug 才是主因）
- ❌ "容量瓶颈"假说：错误（之前 standard 族长 argmax 73.8% 是虚高）

### 修复

`taiji/resonance/neuron.py` forward Step 2：创建标准下三角 causal mask 并传给
`block.attention(h_normed, mask=causal_mask)`。commit 2921b43。

### 新方向（修复后路线）

1. ✅ bug 已修复（commit 2921b43）
2. ✅ 架构底层审计完成（5 组件逐一验证，因果掩码是唯一关键 bug）
3. ✅ 数据质量重新评估（数据清洗有效，但"数据量不足"仍是真约束——见 0.4 最新状态）
4. ✅ 5×compact 联合训练验证（simple_zh 50K 数据，3000 步，完成）
   - 协作 PPL=173.8 vs 最强个体 653.2（涌现-73.4%，确认因果掩码修复后协作机制仍有效）
   - 但生成仍乱码 → PPL=173.8 远高于 TinyStories baseline 16.6，数据/参数比 0.008:1 太低
5. 🔄 4×compact 并行独立训练（差异化数据，全量 simple_zh，8000 步/人，进行中）
   - 策略转变：联合训练（1/5 梯度每人）→ 独立训练（100% 梯度每人），先验证单神经元能否生成连贯
   - zh_full0（全量 787K + train embed）+ zh_full1/2/3（差异化 class 文件 + frozen embed）并行
   - 当前：zh_full0 step 4400/8000, val PPL=84.93, 训练 PPL=4.2
   - 修复：GBK 编码崩溃 → commit 8a6f341（emoji → ASCII）
6. ⏳ 训练完成后评估：单神经元生成质量 + 四神经元协作效果
7. ⏳ 如果仍然乱码 → RoPE theta=500000 是下一嫌疑（审计中标记的唯一非 bug 问题）

### 架构底层审计结果（2026-07-26）

对 5 个基础组件逐一验证，确认因果掩码是**唯一**关键 bug：

| 组件 | 状态 | 依据 |
|------|------|------|
| ① 因果掩码 | ✅ 已修复正确 | neuron.py 创建下三角掩码传给 block.attention；layers.py is_causal=True 启用 SDPA 内置因果掩码 |
| ② RoPE 旋转位置编码 | ✅ 正确 | 频率公式 1/(theta^(2i/dim)) 正确；apply_rotary_emb 交错复数旋转数学正确；Q/K 同变换；kv_cache start_pos 逻辑正确 |
| ③ Tokenization 对齐 | ✅ 正确 | build_position_alignment 字符跨度重叠映射；训练 shift CE / 推理"预测 domain→解码→编码 general 追加"一致 |
| ④ compute_logits | ✅ 正确 | 简单 lm_head(h) 投影，无问题 |
| ⑤ 前向流程 | ✅ 正确（修复后）| embed_adapter → 因果掩码 Transformer → field conditioning(round 2+) → norm → field write → logits |

**非 bug 问题（不影响正确性）**：
- ⚠️ RoPE theta=500000（LLaMA 3 长上下文值）对 compact(36M, block_size=128) 偏大
  - theta=10000（GPT-2/nanoGPT 标准）对短序列位置分辨率更好
  - 不影响正确性，但可能影响小模型学习效率
  - 等 zh_fix0 联合训练结果出来后再决定是否调整

### 数据质量重新评估（2026-07-26）

| 维度 | 之前结论 | 重新评估 |
|------|----------|----------|
| 数据清洗 | 0% 精确重复，3.0% 前缀重复 | ✅ 审计有效，数据本身干净 |
| "数据不足是根因" | 成立 | ❌ 错误，建立在 bug 失真指标上 |
| 数据/参数比 | compact 0.18:1（维基） | simple_zh 8.0:1，已改善 |
| 数据复杂度 | 维基太复杂 | simple_zh 小学水平，匹配 36M |

**核心洞察**：bug 存在时所有指标失真（PPL 虚低、argmax 虚高），无法判断数据质量真实影响。
现在 bug 修复后，联合训练才能给出第一份真实评估。

### 教训

- **诊断顺序错误**：之前先怀疑数据/容量/采样器，最后才检查因果掩码
- **teacher-forcing 高准确率不等于模型好**：必须验证自回归生成
- **架构修改必须消融验证**：因果掩码是基础组件，但从没验证过是否生效
- **PPL 虚低是危险信号**：PPL 1.79（远低于 TinyStories baseline 16.6）应该是警钟
- **修复后第一件事是验证修复生效**：初始 PPL=8506（而非 1.79）确认因果掩码生效

---

## 零、项目状态总览（导航仪表盘）— 2026-07-26 更新（修复后真实数据）

> **因果掩码 bug 修复后，所有指标基于真实因果训练重新评估。**
> 四个问题有严格依赖顺序：因果掩码 → 数据量 → 架构优化 → 协作涌现。

### 0.1 关键路径（修复后，2026-07-26）

```
TinyStories 验证实验（✅ 全部完成，2026-07-25）→ 验证基础 pipeline 无 bug
  实验 A：纯 transformer baseline + TinyStories ✅ PPL=16.6，生成连贯故事
  实验 B：field-augmented + TinyStories ✅ PPL=14.3，field 组件有用（-13.9%）
  实验 C：多神经元协作 + TinyStories ❌ PPL=16.7，logits 平均是坏的协作方式
     ↓
因果掩码 bug 发现 + 修复（✅ 2026-07-26，commit 2921b43）
  ResonanceNeuron.forward 未传 causal mask → 双向注意力 → 所有之前结果无效
  修复后同一权重 argmax=0%（真实性能），PPL=8506（真实随机初始化）
  架构审计：5 组件逐一验证，因果掩码是唯一关键 bug
     ↓
5×compact 联合训练（✅ 完成，50K 数据，3000 步）
  协作 PPL=173.8 vs 最强个体 653.2 → 涌现确认（修复后协作机制仍有效）
  但生成仍乱码（全标点重复）→ PPL 仍太高（173.8 >> 16.6）
  → 根因：数据/参数比 0.008:1 极低（每人 10K 条），联合训练每人获 1/5 梯度
     ↓
4×compact 并行独立训练（🔄 进行中）
  策略：独立训练 100% 梯度 + 全量 simple_zh 数据 + 差异化数据分配
  zh_full0: 787K 全量（train embed）| zh_full1: 248K 中文 | zh_full2: 340K 百科 | zh_full3: 670K 故事
  目标：先验证单神经元能否生成连贯，再验证协作
     ↓
（若生成连贯）验证多神经元协作（forward_train + 残差融合）
     ↓
（若仍乱码）RoPE theta=500000 调整 → 再评估
```

### 0.2 当前瓶颈分析（2026-07-26）

| 瓶颈 | 状态 | 证据 |
|------|------|------|
| 因果掩码 | ✅ 已修复 | 唯一架构 bug，commit 2921b43 |
| 数据量 | 🔄 验证中 | TinyStories 12M+PPL=16.6（33:1），zh_full0 36M+4.4:1 → 差距尚待量化 |
| 架构优化 | ⏳ 待数据验证后 | field 组件 TinyStories 消融确认有用（-13.9%），但 RoPE theta 待调 |
| 协作涌现 | ⏳ 待单神经元验证后 | 5×compact 联合训练涌现确认（协作 PPL < 最强个体 -73%），但需先有连贯单神经元 |

### 0.3 当前运行的实验（2026-07-26）

**上一轮 4 路并行训练已停止**（进程中断，未保存 ckpt）。

**问题诊断**：
- 8000 步只看 256K 样本，数据利用率 <33%（zh_full0 仅 32.5%）
- val PPL=84.93 > train PPL=66.7（exp(4.2)），已过拟合
- shared_embedding.pt 是 7/25 旧版（因果掩码修复前，学坏的），zh_full1/2/3 frozen 模式加载了学坏的 embedding

**新方向：数据多样性增强（方向 B，保持简单数据）**（2026-07-26）：

策略：不增加数据复杂度（仍用 simple_zh 小学水平），通过数据增强 + 正则化提升多样性。

| 改动 | 旧值 | 新值 | 目的 |
|------|------|------|------|
| 数据增强 | 无 | 随机截断(50-100%) + 片段拼接(30%概率) | 防 epoch 间逐字记忆 |
| dropout | 0.1 | 0.2 | 加大正则化 |
| 训练步数 | 8000 | 16000 | 让数据真正看完一遍（zh_full0 需 ~24600 步看完整 epoch） |
| shared_embedding | 旧版(学坏) | 重新训练 | 因果掩码修复后干净起点 |

**串行依赖**：zh_full0（train 模式）→ 完成后自动保存 shared_embedding → 启动 3 路并行 zh_full1/2/3（frozen 模式）。

**当前状态**：
- ✅ 4 路并行训练完成（2026-07-27）
  - zh_aug0: 787K 全量, best_val_ppl=39.6（过拟合，单字重复）
  - zh_aug1: 249K 中文, best_val_ppl=146.6（生成乱码）
  - zh_aug2: 341K 百科, best_val_ppl=22.5（最连贯，知识类数据质量优势）
  - zh_aug3: 670K 故事, best_val_ppl=71.8（半连贯）
  - 关键发现：数据质量 > 数据量（百科 341K 战胜全量 787K）
  - 关键修复：训练脚本保存 per-neuron shared_embedding，避免覆盖导致 embedding 空间不一致

### 0.4 side_channels 联合微调（2026-07-28 进行中）

**目标**：协作 PPL < 114（zh_aug1 的最强个体 PPL），验证 EMERGE 现象。

| 配置 | 值 |
|------|------|
| 训练数据 | simple_zh 10000 条 |
| epochs | 6 |
| batch_size | 4 |
| lr | 1e-3 |
| 可训练参数 | 12.58M（仅 side_channels，神经元核心冻结） |
| 协作机制 | 12 条 excite 通道（4×3 双向），Ensemble max_rounds=2 |
| 工程 | 日志 tee + 每 epoch checkpoint + 断点续训 |

**当前进度**：Epoch 1/6 step 100, PPL=132.4（目标 <114，下降中，ETA 14.5h 总计）
**日志**：`logs/finetune_side_channels_20260728_093654.log`
**脚本**：`scripts/training/finetune_side_channels.py --resume`（支持断点续训）

### 0.5 Playbook 合规状态（side_channels 微调）

| # | 条目 | 要求 | 当前值 | 判定 |
|---|------|------|--------|------|
| 1 | 数据/参数比 | >= 20:1 | 10000*~200tokens/12.58M = 159:1 | ✅ |
| 2 | 数据复杂度匹配 | simple_zh 级别 | simple_zh 小学水平 | ✅ |
| 3 | batch_size | >= 32 | 4（CPU 限制） | ⚠️ 偏小 |
| 4 | warmup | 100-2000 步 | 无（lr 恒定） | ❌ 待加 |
| 5 | decay | 最后 10-20% | 无（lr 恒定） | ❌ 待加 |
| 6 | 工程保障 | 日志+checkpoint+续训 | 已实现 | ✅ |
| 7 | 评估用 PPL + 生成质量 | 双指标 | eval_aug_joint.py | ✅ |
| 8 | 保存 best checkpoint | best loss | 每 epoch 保存 | ✅ |

**合规判断**：数据/参数比充足（159:1），但 lr 调度缺失。如 PPL 在 114 上方停滞，需加 warmup+decay。

---

### 0.6 闭环修复批次（2026-07-28）

系统性审查发现 23 项未闭环点，已处理 21 项，剩余 2 项待训练完成后处理。

**已修复（21 项）：**

| 类别 | 修复内容 | commit |
|------|---------|--------|
| 缺失模块 | 创建 agent_ext/ (9模块) + services/ (5模块) + memory_watchdog + output_engine stub | 2d7f2e7 |
| 接线缺失 | assemble_cortex Step 9.2-9.6 批量接线 play/evolution/limbs/Agent/ContextManager | 4affa2f |
| 死代码 | life_scheduler research 分支遮蔽修复 | 27d4853 |
| 进化闭环 | 凋亡→新生反馈（Phase 4 凋亡后补偿创建） | 560e47b |
| 数据闭环 | explore 结果接入 feed_engine 训练数据 | 2c1a66d |
| 睡眠闭环 | Phase 3.5 knowledge_distillation 实现 | 3bec93f |
| 方法接线 | #20-23: should_trigger_neurogenesis + record_sleep_training + 注释 deprecated | 32c8080 |
| 状态保护 | #19: cortex_state.pt vs neuron_*.pt 时间戳检查 | ea20376 |

**待处理（2 项）：**
- #6 训练-推理路径对齐（forward_train vs forward 机制不一致）— 等 side_channels 训练完成后处理
- #16-18 训练脚本改进（val/早停/决策）— 等当前训练完成后处理

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
| 神经元凋亡与新生 | ApoptosisTracker 淘汰弱神经元 + NeurogenesisTrigger 检测信号（P7：创建由外部 train_neuron.py 执行） |
| 神经调质系统 | Neuromodulator 全局信号调节学习率 |
| 睡眠/记忆巩固 | consolidation_cycle() 离线重放 |
| 功能柱 (Cortical Column) | CorticalColumn 作为一等公民 |
| 注意力增益 | attention_beam 向量 boost 相关神经元 |
| 阈值可塑性 | per-neuron firing_threshold 自适应 |
| 域专用词汇输出 | domain-specific tokenizer + per-neuron 独立 lm_head（P7） |
| 从零独立训练 | 每 neuron 独立 embedding + body + lm_head，零教师依赖（P7-P8） |

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
- `NeuromodulatorState` 全局信号（3 个标量：dopamine/serotonin/norepinephrine）
- **来源（2026-07-22 升级：双信号驱动，自主进化）**：
  - 快速信号：loss 变化率 → 多巴胺目标值（每轮更新）
    - Loss Δ < -20% → DA=0.85（强奖励，lr×2.0）
    - Loss Δ < -5% → DA=0.6（适度奖励，lr×1.4）
    - Loss Δ < 5% → DA=0.3（停滞，lr×0.95）
    - Loss Δ ≥ 5% → DA=0.15（惩罚，lr×0.72）
  - 慢速信号：next-token 准确率 → 血清素目标值（每 5 轮更新）
    - Acc Δ > 2% → 5HT=0.7（满足）
    - Acc Δ ±2% → 5HT=0.5（中性）
    - Acc Δ < -2% → 5HT=0.3（不满足）
- **EMA 趋近**：alpha=0.1，调质缓慢调整不突变
- 作用：
  - 调节神经元学习率：`lr = base_lr × get_lr_multiplier()`（多巴胺驱动）
  - 调节 field_write 强度：`get_field_write_scale()`（去甲肾上腺素驱动）
  - 调节 refractory 长度：`get_refractory_multiplier()`（血清素驱动）
- **持久化**：NeuromodulatorState 纳入 cortex_state.pt，跨会话调质状态连续
- **三调质全接线（2026-07-22 完成）**：
  - DA 由 sleep_engine 的 loss 趋势驱动（每轮）
  - 5HT 由准确率驱动（每5轮）
  - NE 由 metabolism 的 CPU 负载驱动（每次训练前）：CPU 高→NE↓（节能），CPU 低→NE↑（专注）
  - 覆盖优先级：metabolism 仅在内存>90%或资源不健康时覆盖 DA/5HT，否则不干扰 sleep_engine
- **验证**：DA 0.50→0.57, lr_mult 1.25→1.36 over 5 cycles；CPU 10%→NE=0.83, CPU 100%→NE=0.20

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

## 五之续：Phase 5 丘脑路由与动态扩展 — **[已废弃，P7 替代]**

> **废弃原因**：Phase 5 ThalamicRouter 依赖 1.5B 教师 hidden state 做路由判断（prototype 计算、实时路由决策），
> 与 P7 从零训练、零教师依赖的架构方向冲突。P7 每 neuron 自带独立 embedding + 域 tokenizer，
> 域路由由 `Cortex._infer_domain()` 启发式完成（CJK 字符集检测），训练完成后按 `neuron.{domain}` 匹配。
> 
> Phase 5 代码（`thalamic_router.py`, `neurogenesis_creator.py`, `domain_router.py`）已全部删除。
> 路由系统留待 P8-3 基于从零训练的 neuron prototype 重建。

### 5.1 ~ 5.6（已删除）

Phase 5 所有子任务（5.1 丘脑路由、5.2 动态扩展、5.3 学徒期、5.4 域合并）的代码实现已删除。
相关文件：`thalamic_router.py`, `neurogenesis_creator.py`, `domain_router.py`。

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

### 进行中 (Phase 6: 脱教师 + 架构强化) — **[已废弃，P7 替代]**

> **废弃原因**：Phase 6 的目标是"脱教师"——通过 StandaloneEmbedding + SelfEvolver 减少推理时的 1.5B 教师依赖。
> 但 P7 方案更进一步：从零训练，每 neuron 自带独立 embedding + 独立 lm_head + 域 tokenizer，
> 从根本上零教师依赖。Phase 6 的中间方案（teacher SVD 初始化 → SelfEvolver 渐进自主）变得多余。
>
> Phase 6 代码（`standalone_embedding.py`, `self_evolving_encoder.py`, `shared_embed.py`, `init_from_teacher.py`）已全部删除。
> GammaOscillator 和 WorkingMemory（P6-3/P6-4）是独立模块，已保留。

- [x] ~~P6-1: 独立 embedding 表~~ — 已删除，P7 每 neuron 自带 embedding
- [x] ~~P6-2: 独立路由器~~ — 已删除，P8-3 重建
- [x] ~~P6-5: 端到端脱教师验证~~ — 已删除
- [x] **P6-3: Gamma 同步绑定** — 保留，`gamma_oscillator.py` 活跃
- [x] **P6-4: 工作记忆** — 保留，`working_memory.py` 活跃
- [x] ~~P6-6: 自主进化 encoder~~ — 已删除
- [x] ~~P6-7: SleepEngine SelfEvolver 集成~~ — 已删除
- [x] ~~P6-8: 训练后重算 prototypes~~ — 已删除

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

- [x] P7-5: 推理链路改造（2026-07-21）
  - TokenizerHub 增强（encode_tensor/decode/vocab_size/eos_token_id/load_default_domains）
  - ResonanceEnsemble.forward 支持 input_ids（per-neuron encode_input_ids 路径）
  - Cortex P7 模式（set_tokenizer_hub/_infer_domain/_generate_p7）
  - assemble_cortex 移除 W_base，注册 TokenizerHub

- [x] P7-6: 代码清理（2026-07-21）
  - 删除 `_shared_lm_head_base*.pt` 数据文件
  - 删除 `init_w_base_from_teacher.py` / `diagnose_w_base_init.py`
  - neuron.py 移除 set_shared_lm_head() + lm_head_base
  - cortex.py 清理 W_base 注释 + DeprecationWarning on set_teacher_pipeline()

- [x] P7-7: 架构一致性验证（2026-07-21）
  - 6 个文件语法检查通过
  - P7 数据流验证一致
  - __init__.py 导出完整

- [x] P7-8: 计划文档更新（2026-07-21）
  - 新增第九节（P7 架构落地）+ 第十节（Phase 8 从零独立训练）+ 第十一节（项目健康度）

- [x] P7-9: 全项目错误方向清理（2026-07-21）
  - 删除 19 个错误方向文件（resonance 死模块 12 个 + 死脚本 7 个）
  - 清理 8 个活跃文件的残留引用（cortex.py, loader.py, senses.py, sleep_engine.py, play_engine.py, config.py, ensemble.py, evolution_engine.py）
  - 删除 dead code blocks（evolution_engine.py: 120 行蒸馏过渡代码, sleep_engine.py: _sleep_phase_self_evolve, cortex.py: 路由/门控/教师 pipeline 方法）
  - 全局搜索验证：零残留引用

- [x] P7-10: 功能补足与数据清理（2026-07-21）
  - 删除 3 个废弃神经元备份目录（~30 个 .pt 文件）
  - `_infer_domain()` 增强：code 关键字检测 + math 符号密度检测
  - `_train_single_neuron()` 实现：P7/旧双模式，支持 tokenizer_hub 域 tokenizer 训练
  - `play_engine.py` P7 兼容：tokenizer_hub encode + per-neuron encode_input_ids

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

---

## 九、P7 架构落地（已完成）

> 2026-07-21：P7-1 ~ P7-9 全部完成。项目从"蒸馏为主"彻底转向"从零训练"。

### 9.1 P7 推理链路改造（P7-5）

- [x] **TokenizerHub 增强**（`taiji/resonance/translator.py`）
  - `encode_tensor()/decode()/vocab_size()/eos_token_id()/list_domains()` 方法
  - `load_default_domains()` classmethod 自动加载 `taiji/domains/` 下域 tokenizer
  - "general" 域复用 en tokenizer (16k vocab)

- [x] **ResonanceEnsemble 支持 input_ids**（`taiji/resonance/ensemble.py`）
  - `forward()` 新增 `input_ids: Optional[Tensor]` 参数
  - `_parallel_forward()` 支持 P7 路径：每 neuron 调 `encode_input_ids(input_ids)`
  - 向后兼容 `shared_embeddings` 路径
  - 移除 ConfidenceGate/EarlyStopResonance/QualityFilter/DivisionPath/DomainRouter

- [x] **Cortex P7 模式**（`taiji/brain/cortex.py`）
  - `set_tokenizer_hub()`：注册域 tokenizer，自动进入 P7 模式
  - `_infer_domain()`：CJK 字符集启发式推断域
  - `_generate_p7()`：域 tokenizer encode → think → 域 tokenizer decode
  - `think()`：两路径——shared_embedding 或 P7 input_ids 直传
  - `generate()`：自动检测 P7 模式，选域 tokenizer
  - 删除：路由系统、门控系统、教师 pipeline、context_encoder

- [x] **assemble_cortex 更新**（`taiji/loader.py`）
  - W_base 注入、confidence_threshold、enable_gating 参数全部移除
  - 自动注册 `TokenizerHub.load_default_domains()`
  - GammaOscillator 相位按 cortex.neurons 的 domain 分配
  - 删除 SharedContextEncoder/ThalamicRouter 接线

### 9.2 代码清理（P7-6 + P7-9）

- [x] 删除废弃 resonance 模块 12 个：`gating.py`, `quality.py`, `division.py`, `domain_router.py`, `domain_detector.py`, `thalamic_router.py`, `neurogenesis_creator.py`, `standalone_embedding.py`, `self_evolving_encoder.py`, `shared_embed.py`, `init_from_teacher.py`, `channel_broker.py`
- [x] 删除废弃 training 模块：`distill.py`, `checkpoint_bridge.py`, `contrastive.py`, `joint.py`, `single.py`
- [x] 删除废弃脚本 7 个：蒸馏相关 `distill_neurons.py` 等、fix_token_offsets.py
- [x] 删除废弃数据目录：`data/distill/`, `data/neurons_backup*/`, `data/real/`
- [x] `neuron.py` 移除 `set_shared_lm_head()` + `lm_head_base` 属性
- [x] `compute_logits()` 简化为纯 per-neuron 路径
- [x] `sleep_engine.py` 移除 `_sleep_phase_self_evolve()`（678 行）、SelfEvolver 集成、NeurogenesisCreator 调用
- [x] `evolution_engine.py` 移除 120 行 dead code（蒸馏过渡代码）
- [x] 清理 8 个活跃文件的残留引用，全局搜索确认零残留

### 9.3 P7 架构验证

- [x] 所有修改文件语法检查通过
- [x] P7 数据流验证：`assemble_cortex → think → ensemble.forward → neuron.encode_input_ids` 链路一致
- [x] `__init__.py` 导出完整，仅保留活跃模块
- [x] 域 tokenizer 跨 vocab 兼容性处理（token ID clamping、logits padding）

### 9.4 当前状态（2026-07-21 P7-10 功能补足后）

| 功能 | 状态 |
|------|------|
| 域路由（code/math/zh/en/general） | ✅ 启发式检测已实现（code 关键字、math 符号密度、CJK 中文） |
| Sleep 训练（P7/旧双模式） | ✅ `_train_single_neuron` 已实现，支持 tokenizer_hub + per-neuron lm_head |
| PlayEngine P7 兼容 | ✅ 支持 tokenizer_hub 编码 topic，per-neuron encode_input_ids |
| 废弃神经元备份 | ✅ 全部 3 个备份目录已删除（~30 个 .pt 文件） |

### 9.5 L2 指纹共振路由实验（2026-07-23）

**实验**：`verify_routing_accuracy.py` 训练 24 轮（CYCLES 12→24），观察 loss 收敛与 L2 路由改善。

**结果**：
- Loss: 0.2915 → 0.2634（-9.6%），C7 后趋平台（0.26-0.27 震荡，C16 出现 0.29 尖峰）
- L1 域路由: 100% → 100%（启发式检测已完美）
- **L2 共振路由: 8% → 8%（零改善，与 12 轮完全相同）**
- 对比损失全程近零：inter ≈ 0.0000-0.0002，intra ≈ 0.0000

**根因分析**：L2 路由停滞是**结构性问题**，非训练量问题。
- `_fingerprint_route` 依赖神经元指纹的 cosine 相似度选路
- 对比损失 inter≈0.0001 说明指纹间几乎无梯度信号推动分化
- 神经元指纹未被区分开 → 路由无法辨别哪个神经元应处理给定输入
- **增加训练轮次无法解决**（已证伪"12 轮不足"假设）

**已清理错误方向**：
- ~~增加训练轮次 12→24 以改善 L2 路由~~（已证伪：24 轮 L2 仍 8%）

**结论**：当前阶段使用 L1 域路由（100% 准确）。L2 共振路由需修复指纹/对比学习机制本身。

### 9.6 L2 路由方案选型讨论（2026-07-23）

**用户取向**：效果与上限优先，不考虑下限/成本。

**两个原方案的上限分析**：

| 维度 | 方向 A（权重指纹路由） | 方向 B（共振分数路由） |
|------|---------------------|---------------------|
| 路由信号本质 | 静态权重统计量（mean of rows） | 动态输入响应（完整前向的 final_scores） |
| 信号表达力 | 弱——降维统计丢失信息 | 强——真实推理能力 |
| 分化空间 | 768 维（hidden_size），易分化 | 4096 维（field_dim），高维诅咒 |
| 训练信号路径 | 长——fingerprint 是 buffer，需 proxy loss | 中——field_vector 可直接 contrastive |
| 上限天花板 | **低**——权重相似≠能力相似 | **中**——受 4096 维稀疏性限制 |

**核心洞察**：两个原方案上限都不够高。
- A 被"权重是能力的间接代理"限制
- B 被"4096 维高维诅咒"限制

**改造方案与融合可能性**（详见下方讨论）：
- 改造 A 为"动态 prototype fingerprint"——数据驱动，上限中高
- 改造 B 为"端到端监督路由"——直接监督路由准确性，上限最高但需标签
- **融合点**：用共振分数作为 prototype 训练的辅助监督信号

### 9.7 融合方案实施：死代码暴露与三信号修复（2026-07-23）

**用户指令**：按融合方案推进，同时暴露之前借鉴社区项目机械塞入的死代码。

**死代码审计结果**：

| 借鉴机制 | 状态 | 判定 |
|---------|------|------|
| NeuronSpark `lateral_inhibition_norm` | ensemble.forward() 每轮调用 | ✅ 已融入 |
| Deviance WTA `apply_inhibitory_wta` | ensemble.forward() 抑制路径 | ✅ 已融入 |
| Hi-MoE `W_cond` + `prediction_complementarity` | field.score() + ensemble 加权 | ✅ 已融入 |
| RSGN `NeuronGeometry` | coaction 距离门控 | ✅ 已融入 |
| LuminaNet splitting | neurogenesis 路径 | ✅ 已融入 |
| **对比损失三信号（route/proto/align）** | sleep 循环调用但信号趋零 | ❌ **机械塞入死代码** |

**三信号结构性缺陷与修复**：

1. **route_loss 自相矛盾**：原版遍历全序对 (i,j)+(j,i)，要求 sim_i>sim_j 且 sim_j>sim_i，
   梯度互相抵消，净效果推向均匀化（与分化目标相反）。注释说"让正确 neuron 最高"但无域标签。
   → 修复：注入域标签，每个域样本喂给所有 neuron，正确域 adapter(prompt) 与 domain_prototype
   cosine 应最高（与推理路径 _fingerprint_route 一致）。margin=0.2。

2. **proto_loss 地板问题**：原版 `relu(sim-0.1)²`，高维空间 sim≈0（正交），relu(-0.1)=0 无梯度。
   → 修复：`(sim+0.1)²`，sim=0 时 loss=0.01>0，梯度=0.2，持续推向负相关。

3. **align_loss 均匀问题**：前两信号失效时 softmax 均匀导致 KL≈0。
   → 修复：前两信号有效后自然生效，保持 KL 蒸馏。

**根因**：对比损失从 MoCo 借鉴了 margin ranking 形式，但丢了最关键的**正负样本标识**
（哪个 neuron 是正确域），变成自相矛盾的死代码。24 轮实验"对比损失近零"正是此 bug 的表现。

**关键洞察**：`_fingerprint_route`（L2 推理路径）本身不是死代码——它是 contrastive phase 的训练
目标。死的是训练信号，导致推理路径从未被有效训练。修复训练信号即可复活两者。

**修复验证（8 轮）**：

| 指标 | 修复前（24轮） | 修复后（8轮） |
|------|--------------|-------------|
| route_loss | ≈0（自相矛盾） | 0.168→0.073（↓56.6%） |
| proto_loss | ≈0（地板问题） | 0.010→0.004（↓59%） |
| domain_prototype | 全零（未更新） | 全部激活（norm=1.0） |
| L2 路由准确率 | 8% | 3/4 域正确（zh/en/code） |

**端到端验证（8轮完整训练管线，lm_head + contrastive 协同）**：

| 指标 | 修复前（24轮） | 修复后（8轮） |
|------|--------------|-------------|
| L2 指纹路由准确率 | 8% | 21%→29%（3.6×） |
| route_loss | ≈0 | 0.169→0.121（↓28%） |
| proto_loss | ≈0 | 0.008→0.007（非零稳定） |
| align_loss | ≈0 | 0.008→0.002（↓75%） |
| lm_head 训练 loss | — | 4.21→1.36（↓68%） |

**遗留**：L2（29%）仍低于 L1（86%），主因 math 域误判为 zh（符号密集型重叠）+ contrastive
信号尚未完全收敛。更多训练轮次或更强域特异性样本可进一步提升。

### 9.8 多样本实验：域特异性 > 数量（2026-07-23）

**假设**：从 feed_engine 实际训练数据每域采样 3 条（vs 硬编码 1 条），增加 route_loss
的 margin ranking 对数（20→60 对），加速 prototype 分化。

**结果**：L2 路由 **回退** 21%→14%（vs 硬编码 21%→29%）。

**根因**：硬编码样本是最大域特异性的（"神经元共振场架构设计"极中文、"def resonance(field)"
极代码），而训练数据是短句（"今天天气真好"），域区分度更低。更多但更模糊的样本使 prototype
漂移向泛化方向，丢失尖锐域信号。

**初步结论**：对比学习样本的**域特异性质量**远比**数量**重要。

### 9.9 策划高域特异性多样本：质量+数量双优（2026-07-23）

**改进方向**：9.8 证明"训练数据随机采样"失败，但"硬编码单样本"非最优——样本太少导致
prototype 过拟合单一模式。正确路径：**手工策划多条高域标志性样本**，兼顾域内多样性与
域间最大区分度。

**策划样本**（每域 3 条，CURATED_SAMPLES）：

| 域 | 样本 | 域标志性特征 |
|----|------|------------|
| zh | "神经元共振场架构设计原理" / "深度学习模型训练优化方法" / "中文自然语言处理技术应用" | 纯中文长句 |
| en | "neural resonance field architecture design" / ... | 纯英文长句 |
| code | "def resonance(field): return field.sync()" / "class Neuron(nn.Module): ..." / "import torch; ..." | 语法关键字+符号 |
| math | "integral of sin(x) over [0, pi]" / "gradient descent: theta -= lr * dL/dtheta" / "P(A\|B) = P(B\|A) * P(A) / P(B)" | 公式符号密集 |
| general | "system design pattern overview methodology" / ... | 中性长句无强域信号 |

**三方案对比（8 轮完整训练管线）**：

| 方案 | 样本来源 | L2 路由（pre→post） | vs 原版 |
|------|---------|-------------------|--------|
| 原版死代码 | 硬编码 1 条 | 21%→8% | 1×（基线） |
| 单样本硬编码 | 硬编码 1 条 | 21%→29% | 3.6× |
| feed_engine 多样本 | 训练数据随机 3 条 | 21%→14% | 1.75×（回退） |
| **策划高域特异性多样本** | **手工策划 3 条/域** | **21%→36%** | **4.5×** |

**route_loss 收敛状态**：8 轮后 0.130 仍下降（未收敛），更多轮次可进一步提升。

**关键洞察**：
1. **域特异性质量 + 数量双优** >> 单一维度。策划样本同时满足"高域标志性"（质量）和
   "域内多模式覆盖"（数量，防过拟合）。
2. feed_engine 随机采样失败不是"数量"的错，而是"低域特异性数量"的错——短句天然域模糊。
3. **策划样本是 contrastive phase 的最高杠杆点**：样本质量直接决定 prototype 分化方向，
   优于调 margin/lr/轮次等超参。

**遗留**：L2（36%）仍低于 L1（86%）。route_loss 未收敛（0.130 仍降），下一步验证 16 轮
能否继续提升。math 域仍为误判热点（符号密集型与 code/zh 重叠）。

### 9.10 16轮 benchmark + tokenization 不一致根因修复（2026-07-23）

**16 轮 benchmark 结果**：

| 指标 | 8 轮 | 16 轮 |
|------|------|-------|
| route_loss | 0.139→0.130 | 0.139→0.085（↓更多） |
| L2 路由准确率 | 21%→36% | 21%→**29%**（反降！） |

**关键发现**：route_loss 持续下降但 L2 准确率反降。更多训练不仅没帮助，反而过拟合
导致退化。这证明 route_loss 与 L2 准确率之间存在**结构性错配**，非超参问题。

**根因定位：训练/推理 tokenization 路径不一致**

对比两条路径的 prompt 编码：

| 路径 | 编码方式 | 代码位置 |
|------|---------|---------|
| **训练**（route_loss） | 域分词器→逐token映射到general ids | sleep_engine.py:1168-1186（原版） |
| **推理**（_fingerprint_route） | 直接 general 分词器编码 | cortex.py:978-980 |

SentencePiece 对**子串**和**全文**的切分粒度不同：
- 训练：`domain_sp.encode("神经元共振场")` → ["神经","元","共振","场"] → 逐个 piece 用
  `general_sp.EncodeAsIds(piece)` 映射 → 4 个 general ids
- 推理：`general_sp.EncodeAsIds("神经元共振场")` → 可能切为 ["神经元","共振","场"] → 3 个
  general ids（不同切分！）

`embed_adapter` 在训练分布（域分词粒度）上优化，推理时面对不同分布（通用分词粒度），
cosine 相似度计算失真。更多训练→更过拟合训练分布→推理退化。

**修复**：contrastive phase 改用直接 general 分词，与推理路径完全一致：

```python
# 修复前（不一致）：
domain_ids = tokenizer_hub.encode(sample_text, domain=sample_domain)
for did in domain_ids:
    piece = domain_sp.id_to_piece(did)
    general_ids.append(general_sp.EncodeAsIds(piece)[0])

# 修复后（一致）：
general_ids = general_sp.EncodeAsIds(sample_text)  # 与 _fingerprint_route 相同
```

**验证**：8 轮 benchmark 修复前后对比（运行中，结果待填）。

**同时启动的并行方向**：用真实 HF 数据（alpaca-zh + wikipedia-zh，3MB 已缓存）重训 zh
神经元，验证生成质量从胡言乱语→连贯句子的跃升。L2 路由质量依赖于神经元本身的质量
（domain_prototype 的区分度），两条路径不独立。

**tokenization 修复验证结果**：

| 指标 | 修复前（8轮） | 修复后（8轮） |
|------|-------------|-------------|
| L2 路由 | 21%→36% | 21%→**43%** |
| route_loss | 0.139→0.130（未收敛） | 0.189→**0.032**（收敛！） |

route_loss 从 0.130 卡住变为 0.032 真正收敛。tokenization 不一致确实是结构性缺陷。

### 9.11 生成质量实测 + 架构方向重定位（2026-07-23）

**zh 神经元训练**（3MB alpaca-zh + wikipedia-zh，2000 步，compact spec）：
- PPL: 62.4（训练前 PPL=1.3 但那是 100 句玩具数据过拟合）
- 生成输出：从纯乱码（"许多耐耐耐耐"）→ 词组拼贴（"副作用可以大大降低成功的关键...从伦敦到莫斯科...计算机系统"）
- 判断：有进步但仍不可读。学会了词汇，没学会语法。PPL 需降至 ~25-30 才能连贯。

**与 GPT-2 对比**：参数量差 3 倍（36M vs 117M），数据量差 13,000 倍（3MB vs 40GB）。
瓶颈在数据规模，不在架构。

**路由设计讨论：用户原始思路 vs 当前实现**

用户原始思路（竞争式反馈路由）：
1. 族长先看输入 → 给出反馈（实际模型响应）
2. 按反馈质量排序 → 好的族长带领聚落
3. 聚落执行 → 按反馈好坏赋权
4. 好的聚落带领全部神经元完成生成

当前实现（分类式域路由）：启发式域检测 → 直接划定 → 无竞争无反馈。

**根因**：P7 域专用分词器迫使先知道域才能分词，L1 启发式域检测是阻力最小路径，
竞争式路由从未实现。

**社区参考验证**：
- MoA（Together AI, ICLR 2025）：Proposer+Aggregator 多层协作，65.1% 超过 GPT-4o。
  但本质是 API 编排（50 行），用冻结大模型，文本级浅协作。**不是态极**——态极用小神经元
  + 共振场向量级深协作 + 睡眠训练闭环。
- Conf-SMoE（ICML 2026）：confidence-guided gating，"confidence > embedding similarity"
  直接验证用户思路。
- RouteMoA：两阶段路由（轻量预筛 + 置信度评估），对应用户"族长先看反馈"。

**路由方案决策：Plan C — 激活能量路由（生物学思路）**

不预测谁最好，而是看谁被激活（like 大脑）：
- 输入 → 每个神经元 forward → hidden state 激活强度 = "被点亮"
- 最高激活者胜出

用户认可 Plan C。但依赖神经元训练充分（欠训练的神经元激活信号不可靠）。

**核心实验：多同域神经元协作（用户核心假设验证）**

用户洞察："单个神经元肯定不能对话，但由神经元组成的大脑可以。应该尝试用多个同领域
神经元来测试能否在单个小神经元智能不够的情况下，多个协同就可以。"

实验设计：
- 将 zh_texts.jsonl（23443 条）均分 3 份
- 训练 3 个 zh 神经元（zh_1/zh_2/zh_3，各 ~7814 条，1MB）
- 对比：1×3MB 单干（PPL=62.4）vs 3×1MB 协作
- 如果协作 > 单干 → 验证"小神经元协作涌现智能"核心论点

技术实现：
- `cortex.generate()` 新增 `active_nids` 参数（显式指定激活神经元列表）
- MoCo 动态 logit 融合：多神经元 logits 按共振分数加权融合（已有机制）
- 共振场累积多神经元 field vectors（已有机制）

---

## 十、Phase 8: 从零独立训练（下一步）

> **核心命题**：神经元规模小（24M-118M），从零训练完全可行。人脑不蒸馏，婴儿直接暴露于语言环境。
>
> **目标**：去掉所有 1.5B 教师依赖，每 neuron 独立训练 → 共振场协作 → 超越单体模型。

### 10.1 训练可行性

| 神经元 | 参数量 | 10B tokens（单 4090） | 10B tokens（单 A100） |
|--------|--------|----------------------|----------------------|
| compact | ~25M | ~12 小时 | ~3 小时 |
| standard | ~60M | ~1 天 | ~5 小时 |
| expert | ~120M | ~2 天 | ~8 小时 |

5 域各训 10B tokens = 50B total ≈ 一周 A100。GPT-2 124M 在 10B tokens 上即可产生连贯文本。

### 10.2 任务分解

- [ ] **P8-1: 从零训练脚本** `train_neurons_from_scratch.py`
  - 替换 `sft_train_neurons_v2.py`（旧 W_base + 低秩残差）
  - 每 neuron 独立数据加载 + forward + backward
  - 域 tokenizer encode → neuron.forward → CE loss → backward
  - 支持 resume 和 checkpoint 保存
  - 关键：neuron lm_head 从随机初始化开始训练（无 1.5B 教师）

- [ ] **P8-2: 域数据 tokenize**
  - 用 `TokenizerHub` 的域 tokenizer 重新 tokenize SFT 数据
  - 替换 `data/sft/` 中旧共享 tokenizer 的数据
  - 确保每个域有足够数据（≥2000 samples）

- [ ] **P8-3: 纯 P7 路由**
  - 让 ThalamicRouter 支持纯 tokenizer 模式（无需 context_encoder）
  - 或：从零训练期间关闭路由，全部 neuron 参与
  - 训练完后再计算 prototypes 启路由

- [ ] **P8-4: sleep_engine P7 升级**
  - `_train_cortex_neurons` 移除 W_base 冻结逻辑
  - 改为独立 lm_head 训练
  - 支持 per-neuron 独立 lr 和 optimizer

- [ ] **P8-5: 端到端验证**
  - 训练 5 域 neuron（各 10B tokens）
  - 验证 generate 质量（中文输出中文、英文输出英文）
  - 验证路由准确率
  - 验证共振场协作效果

### 10.3 废弃项清单（P7-9 已完成）

| 文件/目录 | 原因 | 状态 |
|-----------|------|------|
| `taiji/training/distill.py` | 蒸馏不再需要 | ✅ 已删除 |
| `taiji/training/checkpoint_bridge.py` | 1.5B 桥接不再需要 | ✅ 已删除 |
| `taiji/resonance/standalone_embedding.py` | P7 每 neuron 自带 embedding | ✅ 已删除 |
| `taiji/resonance/shared_embed.py` | SharedEmbedProj 已废弃 | ✅ 已删除 |
| `taiji/resonance/init_from_teacher.py` | 从教师初始化不再需要 | ✅ 已删除 |
| `taiji/brain/cortex.py::set_teacher_pipeline()` | 教师 pipeline | ✅ 已删除 |
| `taiji/resonance/thalamic_router.py` | 教师依赖路由 | ✅ 已删除 |
| `taiji/resonance/neurogenesis_creator.py` | 蒸馏依赖神经新生 | ✅ 已删除 |
| `taiji/resonance/self_evolving_encoder.py` | 教师 SVD 初始化 | ✅ 已删除 |
| `taiji/resonance/gating.py` | ConfidenceGate 等 | ✅ 已删除 |
| `taiji/resonance/quality.py` | QualityFilter | ✅ 已删除 |
| `taiji/resonance/division.py` | DivisionPath | ✅ 已删除 |
| `scripts/training/distill_neurons.py` | 蒸馏脚本 | ✅ 已删除 |
| `data/distill/` 目录 | 蒸馏中间产物 | ✅ 已删除 |
| `data/neurons_backup*/` 目录 | 旧备份 | ✅ 已删除 |

---

## 十一、项目健康度

### 11.1 代码统计

| 分类 | 数量 | 状态 |
|------|------|------|
| 核心库文件 (`taiji/`) | ~65 个 .py | 架构一致，P7-9 清理后无冗余 |
| resonance 模块 | 16 个 .py | 全部活跃，无死模块 |
| training 模块 | 2 个 .py | scheduler.py + __init__.py，旧蒸馏模块已删 |
| 训练/验证脚本 (`scripts/`) | ~45 个 .py | P7-9 清理后无蒸馏/教师依赖 |
| 计划文档 (`plans/`) | 4 个 .md | 本文档已更新到 P8 路线图 |
| 数据文件 (`data/`) | ~40 个 | P7 废弃文件/目录已删除 |

### 11.2 架构原则对齐

| 原则 | 状态 |
|------|------|
| 差异性第一 | ✅ P7 每 neuron 独立完整参数（embedding + body + lm_head） |
| 自我进化 | ✅ P6-6 SelfEvolver + P8 从零训练 |
| 人脑启发 | ✅ 皮层柱统一结构 + 独立经验学习 + 无教师依赖 |
| 结构性约束 | ✅ 域 tokenizer 控制 vocab，独立 lm_head ~5-10M/neuron |
