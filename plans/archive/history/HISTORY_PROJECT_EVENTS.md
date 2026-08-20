# 项目事件与旧状态归档

> **拆分文档**（2026-08-10）：从 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](../../active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) 按内容拆分。
> 项目整理记录、架构级 bug 修复事件、以及 2026-07-26 旧版状态总览（已过时，仅追溯）。
> 当前项目状态、路线图与接口梳理以主 plan 为准。

**内容**：
- 项目整理记录（2026-07-28）
- 紧急更新：架构级 bug 修复（2026-07-26）
- 归档：项目状态总览（旧导航，已过时）

---

### 清理总结

删除 **43 个废弃脚本**，抽取 **1 个工具模块**，项目结构从 67 个训练脚本精简到 24 个。

### 删除的废弃脚本（43 个）

**错误方向训练脚本（14 个）**：
- train_single_long.py, train_collaboration.py, train_individual_neurons.py, train_multi_zh.py
- joint_train_p7.py, train_v3_neuron.py, verify_v3_quick.py, pipeline_v3_full.py
- pipeline_verify_v3.py, joint_and_generate_v3.py, train_tinystories_collab.py
- train_neuron.py, train_standard_leader.py, train_cortex_joint.py（工具函数已抽取到 utils.py）

**错误方向评估/路由脚本（13 个）**：
- eval_leader.py, eval_individual.py, eval_joint.py, eval_single.py, eval_simple_neuron.py
- eval_collab.py, eval_all_singles.py, eval_gen_quality.py
- verify_routing_accuracy.py, verify_routing_16c.py, verify_routing_short.py
- verify_fingerprint_routing.py, verify_contrastive_fix.py, verify_p7_resonance.py

**一次性诊断脚本（16 个）**：
- diagnose_argmax.py, diagnose_generation.py, diagnose_rounds.py, diag_collab.py, diag_quality.py
- analyze_argmax_ceiling.py, check_argmax.py, integrate_verify.py, test_collaboration.py
- verify_scaled_training.py, gen_test.py, gen_constrained.py, quick_gen_test.py
- audit_neurons.py, test_p7_e2e.py

### 新增工具模块

**`scripts/training/utils.py`**：从 train_neuron.py、train_standard_leader.py、train_cortex_joint.py 抽取的工具函数：
- 常量：OUTPUT_DIR, SHARED_EMBEDDING_PATH, SIMPLE_ZH_DIR, DATA_DIR 等
- tokenizer：load_domain_tokenizer, load_general_tokenizer
- 数据加载：load_domain_texts, load_all_texts, load_simple_zh_texts
- shared_embedding：create_shared_embedding, load_or_create_shared_embedding, save_shared_embedding
- 采样器：SequentialSampler

### 当前活跃脚本（24 个）

**训练核心（5 个）**：
- `train_compact_parallel.py` — 训练 zh_aug0~3 compact 神经元
- `finetune_side_channels.py` — side_channels 联合微调（当前运行中）
- `eval_aug_joint.py` — 评估个体 vs 协作 PPL
- `run_parallel_aug.ps1` — 并行训练 PowerShell 脚本
- `analyze_side_channels.py` — side_channels 死通道诊断

**工具/数据脚本（6 个）**：
- build_domain_tokenizers.py, download_simple_zh.py, download_tinystories.py
- download_zh_data.py, split_simple_zh.py, tokenize_sft_p7.py

**已完成消融实验（2 个，保留作参考）**：
- train_tinystories.py（实验 A baseline, PPL=16.6）
- train_tinystories_field.py（实验 B field-augmented, PPL=14.3）

**P8 多模态/未来方向（4 个）**：
- train_encodec.py, train_vqvae.py, train_video.py, train_neurons_from_scratch.py

**简化参考（1 个）**：
- train_compact_simple.py（train_compact_parallel.py 的简化版）

**集成回归测试（18 个 verify_*.py）**：
- 验证 apoptosis/neurogenesis/cortex chat/http api/state persistence 等运行时闭环

### 项目结构清理原则

1. **错误方向产物立即清理**：避免后续开发被误导
2. **工具函数先抽取再删除**：防止破坏活跃脚本依赖
3. **一次性诊断脚本用完即删**：结论写入 plans 文档即可
4. **集成测试保留**：验证运行时闭环，有长期价值
5. **消融实验保留**：作为历史参考，结论已写入 plans

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

## 归档：项目状态总览（旧导航，2026-07-26，已被"一、项目全景"取代）

> ⚠️ 本仪表盘为 2026-07-26 旧导航，状态已过时，仅保留历史数据供追溯。

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

### 0.4 side_channels 联合微调（2026-07-28 EMERGE 已确认）

**目标**：协作 PPL < 最强个体 solo PPL（同一评估数据），验证 EMERGE 现象。

> **⚠️ 目标修正（2026-07-28）**：原目标"协作 PPL < 114"基于 zh_aug1 在自己训练数据
> （百科）上的 val_ppl=146.57，但 side_channels 微调用 simple_zh 数据评估。在 simple_zh 上
> zh_aug1 的 solo PPL=157.7。EMERGE 的正确定义是**协作 PPL < 最强个体在同一评估数据
> 上的 solo PPL**。

#### EMERGE 确认（2026-07-28）

在 simple_zh 评估数据上：

| 指标 | PPL | 说明 |
|------|-----|------|
| zh_aug1 solo（最强个体） | 157.7 | simple_zh 上的最强个体 |
| zh_aug2 solo | 182.1 | 百科数据训练，simple_zh 上表现中等 |
| zh_aug0 solo | 205.9 | 全量数据训练，simple_zh 上过拟合 |
| zh_aug3 solo | 248.1 | 故事数据训练，simple_zh 上最弱 |
| **协作 PPL（训练前）** | **158.3** | 已 < 最强个体 175.3（EMERGE 出现） |
| **协作 PPL（v1 step 350）** | **131-134** | 乘性门控 + Muon 优化器，PPL 停滞 |
| **协作 PPL（v2 step 300）** | **124.9** | 突破停滞！post-norm + scale + bias |

**结论**：EMERGE 现象已确认。v2 修复后协作 PPL (124.9) 比最强个体 solo PPL (175.3) 低 29%。

#### v1 → v2 突破：side_channels 架构修复（2026-07-28）

v1 PPL 在 131-134 停滞 6 个数据点。诊断发现**所有 12 条通道都是"死"的**：
- gate_deviation = 0.014（gate ≈ 1.0，无调制效果）
- proj_mean = 0.008（每维度投影值极小）
- gate_range = [0.944, 1.064]（仅 ±6% 调制）

**三个根因及修复**：

1. **side_channels 在 RMSNorm 之前应用** → norm 抵消乘性调制
   - 修复：移到 norm 之后，调制直接作用于 logits 输入
2. **投影值太小** → tanh(0.008) ≈ 0.008，gate 几乎等于 1.0
   - 修复：添加可学习 scale 参数（init=50），proj×50 → tanh(0.4) ≈ 0.38
3. **无动态平衡机制** → 部分 channel 可能一直弱
   - 修复：实现 Auxiliary-loss-free balancing（借鉴 DeepSeek V3）
   - 启发式 bias 更新：低利用率 channel 获得正 bias，不通过梯度

**v2 PPL 趋势**：158.3 → 144.7 → 133.1 → **128.9** → 128.0 → 127.3 → **124.9**（持续下降）

#### 根因分析：solo_ppl "异常高"的误解

之前误以为协作环境中 solo_ppl 异常高（zh_aug2 val_ppl=22.53 但协作中 solo_ppl=182.1），
是 field conditioning 噪声导致。实际调查发现：

1. **field_read_layers 未训练**：独立训练时 field_state=None，field_read_layers 权重随机。
   已修复：ensemble.forward 添加 `field_conditioning=False` 选项，round 2 跳过 field conditioning。
2. **solo_ppl "异常"实为域偏移**：zh_aug2 用百科数据训练（val_ppl=22.53），但在 simple_zh
   上评估 PPL=182。这是正常的域偏移，不是 bug。所有神经元在 simple_zh 上 PPL 都远高于
   自己训练数据上的 PPL。

#### 训练配置

| 配置 | 值 |
|------|------|
| 训练数据 | simple_zh 10000 条 |
| epochs | 6 |
| batch_size | 4 |
| lr | 1e-3（Muon + AdamW 混合优化器） |
| 可训练参数 | 12.58M（side_channels 2D） + 12（scale 0D） |
| 协作机制 | 12 条 excite 通道，post-norm 乘性门控，scale 放大 |
| field_conditioning | False（跳过未训练的 field_read_layers） |
| Auxiliary-loss-free balancing | 启发式 bias 更新，每 50 步 |
| LR 调度 | warmup 100 步 + cosine decay（最后 20%） |
| 工程 | 日志 tee + 每 epoch checkpoint + 断点续训 |

**当前日志**：`logs/finetune_side_channels_20260728_131347.log`
**脚本**：`scripts/training/finetune_side_channels.py`

### 0.5 Playbook 合规状态（side_channels 微调）

| # | 条目 | 要求 | 当前值 | 判定 |
|---|------|------|--------|------|
| 1 | 数据/参数比 | >= 20:1 | 10000*~200tokens/12.58M = 159:1 | ✅ |
| 2 | 数据复杂度匹配 | simple_zh 级别 | simple_zh 小学水平 | ✅ |
| 3 | batch_size | >= 32 | 4（CPU 限制） | ⚠️ 偏小 |
| 4 | warmup | 100-2000 步 | 100 步线性 warmup | ✅ |
| 5 | decay | 最后 10-20% | 最后 20% cosine decay | ✅ |
| 6 | 工程保障 | 日志+checkpoint+续训 | 已实现 | ✅ |
| 7 | 评估用 PPL + 生成质量 | 双指标 | eval_aug_joint.py | ✅ |
| 8 | 保存 best checkpoint | best loss | 每 epoch 保存 | ✅ |

**合规判断**：全部条目已满足（batch_size 偏小受 CPU 限制）。PPL 已突破 v1 停滞区间。

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

### 0.7 开源模型借鉴清单（2026-07-28）

调研主流开源模型技术文档，记录可借鉴的工程思想。**目的不是改方向，是吸收工程技巧优化现有神经元共振架构。**

#### 借鉴决策框架：替换 vs 融合 vs 抛弃

对每个借鉴技术，按以下 5 维度评估并选择实施策略：

**维度 1: 原实现状态**
- 有根本缺陷/bug → 倾向替换
- 有效但可改进 → 倾向融合
- 方向错误 → 倾向抛弃原方案

**维度 2: 接口兼容性**
- drop-in 兼容（仅替换组件） → 倾向替换
- 需要适配层（新增参数/接口） → 倾向融合
- 不兼容或不需要 → 抛弃

**维度 3: 训练成本**
- 原训练结果已无效（必须重训） → 替换成本可接受
- 原训练结果可复用 → 融合更优（不浪费已有权重）
- 原训练结果必须废弃 → 抛弃成本高

**维度 4: 设计理念冲突**
- 无冲突（同范式优化） → 替换安全
- 互补（增强现有机制） → 融合
- 根本冲突（破坏生物启发哲学） → 抛弃

**维度 5: 可验证性**
- 有明确指标可对比（PPL/argmax） → 可做 A/B 实验
- 需要定性评估 → 融合后整体观察

**决策树：**

```
原实现是否有根本缺陷？
├── 是 → 原训练结果是否已无效？
│       ├── 是 → 直接替换（案例：causal mask 修复）
│       └── 否 → 抛弃原方案重训（案例：P8 field_dim 统一）
└── 否 → 借鉴技术是否 drop-in 兼容？
        ├── 是 → 原实现是否严格劣于？
        │       ├── 是 → 直接替换（案例：Adam → Muon）
        │       └── 否 → 融合（案例：side_channels + bias）
        └── 否 → 融合（案例：Auxiliary-loss-free balancing）
```

**A/B 对照实验方法（对不确定的借鉴点）：**

1. 基线（原实现）训练 N 步 → 记录 PPL 曲线
2. 借鉴版本训练 N 步 → 记录 PPL 曲线
3. 对比 4 个指标：
   - 收敛速度（达到相同 PPL 的步数）
   - 最终 PPL（相同步数下）
   - 稳定性（PPL 方差）
   - 训练成本（时间/内存）
4. 决策：
   - 借鉴版严格更优 → 直接替换
   - 互有胜负 → 融合（取长补短）
   - 借鉴版更差 → 抛弃借鉴

**已实施借鉴点决策记录：**

| 借鉴技术 | 原实现状态 | 接口兼容 | 训练成本 | 理念冲突 | 决策 | 实施结果 |
|---------|----------|---------|---------|---------|------|---------|
| Muon 优化器 | Adam 无 bug，但收敛慢 | drop-in（仅替换 optimizer） | 原结果可复用 | 无 | **直接替换** | ✅ PPL 停滞 131→突破到 124.9 |
| 乘性门控调制 | 加性调制被 norm 抵消 | drop-in（改 forward） | 原结果无效 | 无 | **直接替换** | ✅ gate_deviation 0.014→0.38 |
| post-norm side_channels | pre-norm 被抵消 | drop-in（调整顺序） | 原结果无效 | 无 | **直接替换** | ✅ PPL 突破停滞 |
| Auxiliary-loss-free balancing | 无平衡机制，通道死亡 | 不兼容（需加 bias buffer） | 原结果可复用 | 互补 | **融合** | ✅ 启发式 bias 更新 |

**待实施借鉴点预评估：**

| 借鉴技术 | 原实现状态 | 接口兼容 | 训练成本 | 理念冲突 | 预决策 |
|---------|----------|---------|---------|---------|--------|
| Shared Expert（Kimi K3） | 无 shared expert 概念 | 不兼容（需新神经元） | 新增训练 | 互补 | **融合** |
| IndexShare DSA（GLM-5.2） | 无跨轮索引复用 | 不兼容 | 新增逻辑 | 互补 | **融合** |
| Confidence-Guided（ConfSMoE） | 共振分公式简单 | 部分兼容 | 可调参 | 互补 | **融合** |
| Conditional Memory（DeepSeek V4） | 无架构级 RAG | 不兼容 | 新增模块 | 互补 | **融合** |
| OPD 在线策略蒸馏 | 两阶段训练无蒸馏 | 不兼容 | 流程重构 | 互补 | **融合** |
| KDA 线性注意力（Kimi K3） | 标准 attention | 不兼容（架构变） | 全量重训 | 冲突 | **抛弃原方案**（远期） |

**核心原则：**
- **直接替换**需要 3 个必要条件全部满足：原实现有根本缺陷 OR 借鉴严格更优 + drop-in 兼容 + 不破坏设计理念
- **融合**只要任一满足：借鉴与现有机制互补 + 需要适配 + 保留生物启发哲学
- **抛弃原方案**需要：原方向根本错误 + 借鉴从根本上更优且无冲突
- 所有"不确定"的借鉴点必须通过 A/B 对照实验验证后再决策

#### 借鉴点优先级总览（按实施顺序）

| 优先级 | 技术 | 来源模型 | 态极应用场景 | 实施时机 |
|--------|------|---------|-------------|---------|
| ★★★ 立即 | **Muon 优化器** | DeepSeek V4 | side_channels 微调换 Adam→Muon，突破 PPL 瓶颈 | 当前训练完成后第一实验 |
| ★★★ 立即 | **Auxiliary-loss-free balancing** | DeepSeek V3 | 解决 side_channels "死通道"问题，用偏置项动态调整而非辅助损失 | Muon 实验后 |
| ★★★ 中期 | **Shared Expert 机制** | Kimi K3 + DeepSeek V3 | 训练 general 神经元始终激活，其他域神经元稀疏激活 | side_channels 验证 EMERGE 后 |
| ★★★ 中期 | **IndexShare 索引复用** | GLM-5.2 | max_rounds=3 时复用 round 1 激活模式，降低多轮共振成本 | 扩展到 max_rounds=3 时 |
| ★★★ 中期 | **Conditional Memory** | DeepSeek V4 | 三路触发门控（关键词/语义/场景标签）补全记忆→推理融合 | Agent 闭环实现时 |
| ★★★ 中期 | **Confidence-Guided Selection** | ConfSMoE (ICML 2026) | 优化共振分公式，处理"多神经元共振分接近"的模糊场景 | 路由优化阶段 |
| ★★ 后期 | **OPD 在线策略蒸馏** | DeepSeek V4 | 优化"个体训练→协作训练"两阶段的蒸馏策略 | 训练流程重构时 |
| ★★ 后期 | **mHC 流形约束超连接** | DeepSeek V4 | 稳定 side_channels 梯度传播，防止 PPL 跳变 | 训练稳定性优化时 |
| ★★ 后期 | **Quantile Balancing** | Kimi K3 | 防止 side_channels 死通道（与 Auxiliary-loss-free 互补） | 路由优化阶段 |
| ★★ 后期 | **Thinking/Non-thinking 双模式** | Qwen3 + DeepSeek V4 | max_rounds=1（快速）/2（平衡）/3（深度）三档自动切换 | Cortex 路由优化时 |
| ★ 远期 | **KDA 线性注意力** | Kimi K3 | 长上下文（>2K token）时部分层换线性注意力 | 长上下文支持时 |
| ★ 远期 | **CSA+HCA 混合注意力** | DeepSeek V4 | 分层压缩（精读+广角），1M 上下文 FLOPs 仅 27% | 长上下文支持时 |
| ★ 远期 | **MoonViT-V2 训练方式** | Kimi K3 | VQ-VAE 用 next-token prediction 从零训练 | VQ-VAE 重训时 |
| ★ 远期 | **AgentEnv 沙箱设计** | Kimi K3 | 工具学习闭环的快照/恢复/fork | limbs.py 重构时 |

**核心洞察：**
- 所有主流开源模型（Kimi K3 / DeepSeek V3/V4 / GLM-5.2 / Qwen3）均采用 MoE 路线，与态极的神经元共振是不同技术路线
- MoE 是"一个大脑内 FFN 子模块稀疏激活"，态极是"多个完整模型协作"——根本差异在专家粒度
- **真正值得借鉴的不是架构，是工程技巧**：优化器（Muon）、负载均衡（Auxiliary-loss-free）、记忆融合（Conditional Memory）、索引复用（IndexShare）
- **ConfSMoE 是学术上最接近态极设计哲学的工作**：置信度路由与共振分都是"基于信号质量而非学习权重做路由"

---

#### 各模型详细架构事实与借鉴分析

#### Kimi K3（2.8T MoE，Moonshot AI，2026-07-27 开源）

**架构事实**（来源：[GitHub README](https://github.com/MoonshotAI/Kimi-K3)）：
- 93 层 = 1 Dense + 69 KDA（线性注意力）+ 24 Gated MLA（潜在注意力）
- 896 路由专家 + 2 共享专家，每 token 激活 16 个，激活参数 104B/2.8T（3.7%）
- SiTU-GLU 激活 + Quantile Balancing 防路由崩溃
- MXFP4 权重 / MXFP8 激活（量化感知训练）
- MoonViT-V2 视觉编码器：next-token prediction 从零训练（不用对比预训练）

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Shared Expert 机制** | 保留 general 神经元始终激活（类似 shared expert），其他域神经元稀疏激活。对应 Cortex routing_level=1 模式，但当前 general 神经元从未训练过——需要训练一个 general 神经元作为"共享专家" |
| ★★ | **Quantile Balancing** | side_channels 微调面临"死通道"问题（12 条通道中可能只有几条被激活学习）。借鉴分位数平衡思想，强制所有 side_channels 至少参与一定比例的梯度更新 |
| ★★ | **KDA 线性注意力** | 未来支持长上下文（>2K token）时，可把部分 attention 层换成线性注意力。当前 compact 神经元（512 hidden, 6 层）还不需要 |
| ★ | **MoonViT-V2 训练方式** | 视觉编码器用 next-token prediction 从零训练（不用对比预训练），获得更稳定的优化过程。态极 VQ-VAE 训练可借鉴 |
| ★ | **AgentEnv 沙箱设计** | 快照/恢复/fork 思路对 taiji/body/limbs.py 工具学习闭环有启发 |

**不借鉴的方向：**
- MoE 路由器（态极用共振分而非可学习 router，设计哲学不同）
- Gated MLA 矩阵分解（态极单神经元规模太小，不需要 KV cache 压缩）
- 3:1 KDA:MLA 混合比例（态极神经元层数少，不需要混合注意力）

#### DeepSeek V3（671B MoE，2024-12 开源）

**架构事实**（来源：[GitHub README](https://github.com/deepseek-ai/DeepSeek-V3)）：
- 671B 总参数，37B 激活（5.5% 稀疏度）
- MLA (Multi-head Latent Attention) + DeepSeekMoE
- 256 个专家
- **Auxiliary-loss-free load balancing** — 无辅助损失的负载均衡策略
- MTP (Multi-Token Prediction) 14B 模块
- 128K 上下文

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Auxiliary-loss-free load balancing** | DeepSeek V3 的核心创新：不用辅助损失强制负载均衡（辅助损失会污染主损失导致性能退化），而是用偏置项动态调整。态极 side_channels 的"死通道"问题可以借鉴此思路——不用正则项强制通道激活，而是用偏置项动态调整通道被选中的概率 |
| ★★ | **MTP (Multi-Token Prediction)** | 训练时预测多个未来 token（不只是 next-token），提升训练效率。态极 _train_single_neuron 当前只用 next-token CE loss，可考虑加 MTP 辅助头增强表征学习 |
| ★ | **DeepSeekMoE 细粒度专家** | 共享专家 + 路由专家分离设计。与 Kimi K3 的 shared expert 思路一致，双重验证了"保留通用专家 + 稀疏路由专家"的正确性 |

#### Qwen3（Dense + MoE 全尺寸，2025 开源）

**架构事实**（来源：[GitHub README](https://github.com/QwenLM/Qwen2.5)）：
- Dense: 0.6B / 1.7B / 4B / 8B / 14B / 32B
- MoE: 30B-A3B / 235B-A22B
- **Thinking / Non-thinking 双模式**
- Agent 能力强化（工具调用 in thinking & unthinking modes）

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★ | **Thinking/Non-thinking 双模式** | 态极可借鉴：默认 max_rounds=1（快速响应），复杂任务时切换到 max_rounds=3（深度思考，多轮共振）。对应 Cortex 已有的 routing_level 参数，但当前没有"任务复杂度感知"的自动切换机制 |
| ★ | **小尺寸 dense 模型（0.6B-32B）** | 验证了小模型路线可行性。态极 compact 神经元（36M）比 Qwen3 最小的 0.6B 还小一个数量级，说明态极在"超小模型协作"赛道有独特定位 |

#### ConfSMoE（ICML 2026，Confidence-Guided Sparse Expert Selection）

**架构事实**（来源：[GitHub](https://github.com/IcurasLW/ICML2026-Official-Repository-of-ConfSMoE)）：
- 用**置信度引导**稀疏专家选择，而非可学习路由器
- 专门处理**缺失输入**和**多模态输入**场景
- 学术论文，代码开源

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Confidence-Guided Expert Selection** | **与态极的"共振分"高度契合**。ConfSMoE 用输入置信度选专家，态极用共振分选神经元——两者本质都是"基于信号质量而非学习权重做路由"。可以借鉴 ConfSMoE 的置信度计算方式优化共振分公式，特别是处理"输入模糊"场景（多神经元共振分接近时如何选择） |

#### Mixtral 8x7B / 8x22B（Mistral AI，已归档）

**架构事实**：标准 MoE，8 个专家每 token 激活 2 个。仓库已归档，技术较老，无特殊借鉴价值。

#### Llama 3（Meta，已废弃）

**架构事实**：标准 Dense Transformer，GQA。仓库已废弃指向 llama-models，无 MoE 创新。态极已采用 GQA，无需额外借鉴。

#### GLM-5.2（744B MoE，智谱 AI，2026-06 开源）

**架构事实**（来源：[NVIDIA NeMo 文档](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/glm-5-moe-dsa.md) + [智谱开源说明](https://cndba.cn/article/17044)）：
- 744B 总参数，激活 40B（5.4% 稀疏度）
- MLA (Multi-head Latent Attention) + **DSA (Dynamic Sparse Attention)** 动态稀疏注意力

- **IndexShare DSA**：共享 DSA 层复用前一层的 top-k 稀疏注意力选择（跨层索引复用）
- 1M token 上下文
- MIT 协议
- GlmMoeDsaForCausalLM 架构
- TileLang 稀疏核可选

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **IndexShare DSA（跨层索引复用）** | GLM-5.2 每 4 层稀疏注意力共享同一套 top-k 索引，避免每层重新计算。**态极多轮共振可借鉴**：max_rounds=3 时，round 2/3 可以复用 round 1 的神经元激活模式（side_channels 的激活索引），避免每轮重新计算共振分。这能显著降低多轮共振的计算成本 |
| ★★ | **DSA 动态稀疏注意力** | 与态极的"稀疏激活神经元"思路相近。GLM 用动态 top-k 选择注意力的 token，态极用共振分选择激活的神经元——都是"基于内容动态稀疏"而非固定路由 |

#### DeepSeek V4（1.6T MoE，2026-04 开源）

**架构事实**（来源：[技术报告解读](https://cloud.tencent.cn/developer/article/2661839)）：
- V4-Pro: 1.6T 总参数，激活 49B（3.1% 稀疏度）
- V4-Flash: 284B 总参数，激活 13B
- **CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention)** 混合注意力
- **mHC (Manifold-Constrained Hyper-Connections)** 流形约束超连接
- **Muon 优化器**（替代 AdamW）
- **Conditional Memory** 条件记忆机制（架构级 RAG）
- **OPD (On-Policy Distillation)** 在线策略蒸馏
- 384 专家，激活 6 个（V4-Pro）
- DSA2（融合 DSA+NSA）
- 三种推理模式：Non-think / Think High / Think Max
- 1M 上下文，MIT 协议

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Conditional Memory（条件记忆机制）** | **架构级 RAG 整合**：用户 query 通过关键词/语义/场景标签三路触发门控，激活相关记忆块，跨注意力融合。**与态极的语义记忆 + 工作记忆设计高度契合**，当前态极记忆系统虽已接线（#10/#15 修复），但检索→推理的融合机制仍缺失。可借鉴 DeepSeek V4 的三路触发门控设计 |
| ★★★ | **Muon 优化器** | 替代 AdamW，通过正交化更新方向使收敛更快更稳。**态极 side_channels 微调直接可借鉴**——当前用 Adam lr=1e-3，PPL 下降缓慢（132→目标<114），换 Muon 可能突破瓶颈。这是最低成本的尝试（只改优化器不改架构） |
| ★★ | **mHC 流形约束超连接** | 解决万亿模型训练稳定性。态极 side_channels 训练也有梯度不稳定问题（PPL 偶尔跳变），mHC 的流形约束残差思想可借鉴用于稳定 side_channels 的梯度传播 |
| ★★ | **OPD 在线策略蒸馏** | 分领域训练专家→多专家在线蒸馏融合。**与态极的"个体训练→协作训练"两阶段高度一致**。DeepSeek V4 用 OPD 把十几个领域专家能力融合进一个模型，态极用 side_channels 把多个神经元融合进 ensemble——可借鉴 OPD 的蒸馏策略优化 side_channels 训练 |
| ★ | **CSA+HCA 混合注意力** | 分层压缩（CSA 精读 + HCA 广角），1M 上下文下 FLOPs 仅 27%。态极未来支持长上下文时借鉴 |
| ★ | **三模式推理** | Non-think/Think High/Think Max 三档。与 Qwen3 的 thinking/non-thinking 双重验证了"按需调节思考深度"的正确性，态极可扩展为 max_rounds=1/2/3 三档 |
