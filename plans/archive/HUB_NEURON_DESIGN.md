# Hub Neuron 设计草案（联合皮层）

> 解决跨域语义对齐问题（缺口 L），参考人脑联合皮层机制，
> 让 zh/code 等不同域 neuron 能通过 hub neuron 中转实现跨域协作。
>
> 设计原则：**上限优先**——选择能力上限更高的方案，而非技术难度更低的方案。
>
> 状态：草案（待讨论决策点 1-4 后实施）
> 关联缺口：L（跨域语义对齐）、M（forward_train 跨 vocab 崩溃）

---

## 一、设计背景

### 1.1 当前 EMERGE 的真相

5 个 zh neuron 协作 PPL=24.0（最强个体 34.5，提升 30.5%），靠的是 **logits 层融合**，**不是 field 语义对齐**：

- [ensemble.py:1034-1039](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1034-L1039) 明确：`field.score()` 跨 neuron 不可比（最差 PPL neuron 拿最高分），已禁用 resonance score boost
- EMERGE 主因：同 vocab(20K) logits 加权融合 + 同域数据训练让 hidden space 自然相近
- field vector 方向相近是**训练数据驱动的副产物**，不是架构保证

### 1.2 加入 code neuron 的瓶颈

| 层面 | 问题 |
|------|------|
| logits 融合 | zh(20K) vs code(12K) 不同 vocab，[ensemble.py:736](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L736) `same_vocab=False` 直接跳过融合 |
| side_channels | 传递 code 的 field_vector 给 zh，但 zh 无法"理解"代码结构，训练信号稀薄 |
| field state 共享 | code 的 field_vector 对 zh 是语义噪声，互相污染 |

**核心缺失**：无显式跨域对齐约束，无联合皮层中转。

---

## 二、生物学参考

人脑跨模态对齐的关键机制：

| 人脑机制 | 作用 | 态极对应 | 状态 |
|---------|------|---------|------|
| **联合皮层**（association cortex） | 接收多模态输入，学习跨模态对应关系，**能产生内部表征** | hub neuron 带有 lm_head，可生成内部思维 | 本设计 |
| **海马 hub 神经元** | 跨域信息中转，绑定到同一事件 | hub 的 field_vector 作为跨域共享锚点 | 本设计 |
| **多模态神经元** | 单 neuron 同时响应多种刺激 | hub 用 general vocab 覆盖所有域 | 本设计 |
| Hebbian 跨突触可塑性 | 同时激活的脑区突触增强 | CoactivationTracker | 已有，需扩展 |

**核心借鉴**：
1. 联合皮层不仅中转，**还能产生内部表征**（思维、想象）——hub 有 lm_head 对应此能力
2. 联合皮层是**高容量区域**（人脑皮层占脑体积最大）——hub 用 expert 规格对应
3. 跨模态对应关系是**显式学习**的（如婴儿通过多模态输入学习"苹果"这个词↔苹果的形象）——跨域对比 loss 对应

---

## 三、架构设计（上限优先版）

### 3.1 hub neuron 角色

hub neuron 是一个**高容量、全词表、可生成的联合皮层 neuron**：

```
        zh neuron ◄──── side_channel ────► hub neuron ◄──── side_channel ────► code neuron
        (zh vocab 20K)                     (general vocab 256K)               (code vocab 12K)
            │                                    │                                  │
            │                                    │ ← 可生成（lm_head 256K）         │
            └──────── field_write ────► ◄ field_write ◄ ──── field_write ◄──────┘
                                       (共振场，hub 锚点，field_dim=4096)
```

- **输入**：与普通 neuron 一样，接收 shared_embedding（general tokenizer 编码）
- **生成**：拥有 lm_head（general vocab 256K），可独立生成文本，可作为 general 域 fallback
- **中转**：通过 side_channels 与所有域 neuron 双向连接
- **锚点**：field_vector（field_dim=4096）作为跨域共享语义锚点写入共振场
- **容量**：expert 规格（~300M），hidden=1024，是所有 neuron 中容量最大者

### 3.2 side_channels 拓扑：hub-and-spoke + 同域全连接 + hub 参与域内融合

当前 side_channels 是全连接（N×(N-1) 条）。引入 hub 后采用**混合拓扑 + hub 域内增强**：

| 连接类型 | 说明 | 设计意图 |
|---------|------|---------|
| 同域 spoke 间全连接 | zh_aug0 ↔ zh_aug1 ↔ ... | 保留 zh EMERGE（已验证 30.5% 提升） |
| hub ↔ 所有 spoke（双向） | hub ↔ zh_aug0, hub ↔ code, ... | 跨域中转 + 域内增强 |
| 跨域 spoke 间 | 无 | 避免跨域语义噪声直接污染 |

**hub 参与域内融合**：hub 的 lm_head 输出（general vocab 256K）与同域 neuron 的 logits 不直接融合（vocab 不同），但通过 side_channels 调制同域 neuron 的 hidden，间接提升域内 EMERGE。

参数量对比（5 zh + 1 code + 1 hub = 7 neuron）：
- 全连接：7×6 = 42 条
- 混合拓扑：同域(zh 5×4=20) + hub-spoke(6×2=12) = 32 条
- 减少 10 条跨域 spoke-spoke 连接，避免噪声

### 3.3 hub neuron 的 field_write 语义

hub neuron 用**跨域混合数据**训练（zh + code + en + math 对话数据），它的 field_write 层学到的是"跨域共享子空间"——不是某个域的特有结构，而是所有域共有的语义原语（如"实体"、"动作"、"关系"）。

**跨域对比 loss** 显式约束对齐：
- 同义跨域输入对（zh"函数" ↔ code"function"）的 field_vector cosine 最大化
- 不同义跨域输入对的 field_vector cosine 最小化（对比学习）
- 这让 hub 的 field_vector 成为真正的跨域语义锚点，而非 CE loss 的副产物

### 3.4 hub 的 lm_head 角色

hub 的 lm_head（general vocab 256K）有 3 个作用：
1. **内部思维**：hub 能产生内部表征（logits），通过 side_channels 调制其他 neuron
2. **general 域 fallback**：当输入不属于任何已知域时，hub 作为 general neuron 直接生成
3. **对齐质量评估**：hub 自身的 PPL 可作为跨域对齐质量的间接指标

---

## 四、关键决策（上限优先版）

### 决策 1：hub neuron 是否需要 lm_head？

| 选项 | 说明 | 上限 | 技术难度 |
|------|------|------|---------|
| A. 无 lm_head（纯 field neuron） | hub 只参与 field 写入和中转 | 低 | 低 |
| **B. 有 lm_head（general vocab 256K）** | hub 可生成、可评估、可作 general fallback | **高** | 中 |

**选择 B**：上限优先。理由：
1. 符合人脑联合皮层"能产生内部表征"的特征（非纯中转）
2. hub 可作为 general 域 fallback，综合体能力上限更高
3. hub 自身 PPL 可评估跨域对齐质量
4. 256K lm_head 参数量大（256K×1024≈262M），但 expert 规格本就是最大容量
5. general vocab 融合问题通过 side_channels 调制间接解决（不走 logits 直接融合）

### 决策 2：hub 的规格

| 选项 | 说明 | 上限 | 技术难度 |
|------|------|------|---------|
| A. standard（116M，hidden=768, field_dim=3072） | 与现有 zh_std0 同规格 | 中 | 低 |
| **B. expert（~300M，hidden=1024, field_dim=4096）** | 最大容量 | **高** | 中 |

**选择 B**：上限优先。理由：
1. 联合皮层是人脑容量最大的区域，hub 应有最大容量
2. field_dim=4096 容纳最丰富的跨域语义原语
3. hidden=1024 提供更强的多域信息整合能力
4. 跨规格投影层已实现，与现有 compact/standard neuron 兼容
5. 风险（expert 未训练验证）通过阶段 1 单独训练 hub 验证来缓解

### 决策 3：side_channels 拓扑

| 选项 | 说明 | 上限 | 技术难度 |
|------|------|------|---------|
| A. 全连接 | 所有 neuron 两两连接 | 低（跨域噪声） | 低 |
| **B. hub-and-spoke + 同域全连接 + hub 域内增强** | 跨域只通过 hub 中转；hub 参与域内调制 | **高** | 中 |
| C. 纯 hub-and-spoke | 所有 spoke 间不直接连接 | 低（破坏 EMERGE） | 低 |

**选择 B**：上限优先。理由：
1. 同域全连接保留已验证的 zh EMERGE
2. 跨域只通过 hub 中转避免语义噪声
3. hub 参与域内增强提升 EMERGE 上限（通过 side_channels 调制，不是 logits 融合）
4. 符合人脑"联合皮层与各感觉皮层双向连接"的机制

### 决策 4：训练 loss 设计

| 选项 | 说明 | 上限 | 技术难度 |
|------|------|------|---------|
| A. 纯 CE loss（跨域混合数据） | 对齐是 CE 副产物 | 低 | 低 |
| B. CE + hub 锚定 loss | 约束域 neuron field_vector 与 hub field_vector cosine 最大化 | 中 | 中 |
| **C. CE + 跨域对比 loss** | 同义跨域输入对 cosine 最大化，不同义对最小化 | **高** | 高 |

**选择 C**：上限优先。理由：
1. 显式跨域语义对齐，最强保证
2. 跨域对比 loss 直接约束 zh"函数" ↔ code"function" 的 field_vector 对齐
3. 对比学习是已验证的强对齐方法（SimCLR/CLIP 范式）
4. 需构建跨域平行语料，成本高但上限最高

**跨域对比 loss 实现**：
```python
# 正样本对：同义跨域输入（zh"函数定义" ↔ code"function definition"）
# 负样本对：不同义跨域输入
# field_vec_zh = neuron_zh.forward(zh_input)["field_vector"]
# field_vec_code = neuron_code.forward(code_input)["field_vector"]
# loss_contrastive = -log(sim(zh, code_pos) / sum(sim(zh, code_neg)))
```

**跨域平行语料构建**：
- 代码注释翻译对（code 注释 ↔ zh 解释）
- 同义概念对（zh"循环" ↔ code"for/while"）
- 可半自动构建：用现有 code/zh 对话数据，用关键词匹配 + 人工校验

---

## 五、与现有架构的兼容性

| 组件 | 改动 | 影响 |
|------|------|------|
| ResonanceNeuron | 无改动（hub 是普通 neuron，只是规格=expert + vocab=256K） | ✅ 零影响 |
| ResonanceEnsemble | side_channels 建立逻辑改为混合拓扑（决策 3B） | 需改 `establish_side_channel` 调用逻辑 |
| ResonanceField | 无改动 | ✅ 零影响 |
| Cortex | 域路由：hub neuron always-active（与 general 同级）；hub 可作 general fallback | 需改 `_infer_domain` / 路由逻辑 |
| 训练脚本 | 新增 `train_hub_neuron.py`（跨域混合数据 + expert 规格） | 新文件 |
| 对比 loss | 新增 `contrastive_cross_domain.py`（跨域对比 loss + 平行语料加载） | 新文件 |
| 协作层训练 | `finetune_cross_spec.py` 扩展支持 hub-spoke 拓扑 + 对比 loss | 需改 side_channels 建立逻辑 + 加 loss 项 |
| eval 脚本 | 评估跨域协作效果 + hub 锚点效应 | 新增评估场景 |

**关键保证**：现有 5 zh neuron 的 EMERGE（30.5% 提升）不受影响，因为同域全连接保留、ensemble 融合逻辑不改。

---

## 六、实施计划（上限优先版）

### 阶段 1：跨域平行语料构建
1. 从现有 zh/code 对话数据中提取跨域概念对
2. 半自动构建平行语料（关键词匹配 + 人工校验）
3. 产物：`data/cross_domain_pairs.jsonl`（zh↔code 同义对）

### 阶段 2：hub neuron 训练（expert 规格）
1. 准备跨域混合训练数据（zh + code + en + math 对话数据混合）
2. 新增 `train_hub_neuron.py`，用 expert 规格 + general tokenizer(256K) + lm_head 训练
3. 验证 hub neuron 单独的生成能力（用 zh/code/general 数据评估 PPL）
4. 产物：`neuron_hub.pt`

### 阶段 3：hub-and-spoke 协作层训练（含对比 loss）
1. 改 `finetune_cross_spec.py` 支持混合拓扑（决策 3B）
2. 新增 `contrastive_cross_domain.py`，实现跨域对比 loss
3. 冻结 neuron 核心 + hub 核心，训练 side_channels + 对比 loss
4. 产物：`cross_spec_hub.pt`

### 阶段 4：跨域协作评估
1. 装配 5 zh + 1 code + 1 hub 综合体
2. 评估跨域对话能力（含代码问题的中文解释）
3. 对照实验：
   - 对照组 1：无 hub 的 5 zh + 1 code（跨域噪声组）
   - 对照组 2：5 zh + 1 code + 1 hub（纯 CE，无对比 loss）
   - 实验组：5 zh + 1 code + 1 hub（CE + 对比 loss）

### 阶段 5：forward_train 修复（缺口 M）
1. 改 `ensemble.py:828-833` 按 vocab 分组处理 logits
2. 跨域联合训练路径可用

---

## 七、风险评估（上限优先版）

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| expert 规格 hub 训练不稳定 | 中 | hub 能力不足 | 阶段 2 单独验证；回退到 standard 规格 |
| 跨域平行语料质量不足 | 中 | 对比 loss 效果差 | 半自动构建 + 人工校验；回退到决策 4B（hub 锚定 loss） |
| 256K lm_head 参数量大训练慢 | 高 | 训练时间长 | 冻结 shared_embedding；用 LoRA 低秩 lm_head |
| side_channels 混合拓扑引入新 bug | 低 | 协作层训练失败 | 保留全连接作为 fallback |
| forward_train 跨 vocab 修复复杂 | 中 | 联合训练不可用 | 先用 finetune_cross_spec（不需要 forward_train） |
| 对比 loss 与 CE loss 权重难调 | 中 | 训练不收敛 | 用 warmup 策略（先 CE 后加对比） |

---

## 八、成功标准（上限优先版）

1. **跨域协作不退化**：加入 code neuron 后，zh 域对话 PPL 不显著上升（< 30.0，当前 24.0）
2. **跨域能力涌现**：综合体能处理"用中文解释代码"类跨域问题（无 hub 时无法处理）
3. **hub 锚点效应**：hub 的 field_vector 与各域 neuron 的 field_vector cosine 相似度 > 0.5（随机基线 ~0.0，对比 loss 驱动）
4. **hub 自身生成能力**：hub neuron 单独 PPL < 50.0（general vocab 256K，跨域混合数据）
5. **对比 loss 增益**：CE + 对比 loss 组合的跨域协作 PPL 优于纯 CE（验证对比 loss 价值）

---

## 九、待确认决策清单（上限优先版）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| 1 | hub 是否有 lm_head | **B. 有 lm_head（general 256K）** | 可生成、可评估、可作 fallback；符合联合皮层能产生内部表征 |
| 2 | hub 规格 | **B. expert（~300M）** | 最大容量；field_dim=4096 容纳最丰富跨域语义 |
| 3 | side_channels 拓扑 | **B. hub-and-spoke + 同域全连接 + hub 域内增强** | 保护 EMERGE + 跨域中转 + 域内增强 |
| 4 | 训练 loss | **C. CE + 跨域对比 loss** | 显式跨域语义对齐，最强保证 |

**上限提升对比**：
| 维度 | 易实现版 | 上限优先版 | 提升 |
|------|---------|-----------|------|
| hub 生成能力 | 无 | 256K vocab lm_head | 可独立生成 + fallback |
| hub 容量 | 116M (standard) | 300M (expert) | 2.6× 参数 |
| 对齐强度 | CE 副产物 | CE + 显式对比 loss | 强对齐保证 |
| 评估深度 | 无 hub 评估 | PPL + cosine + 对照实验 | 可量化对齐质量 |

确认后进入阶段 1（跨域平行语料构建）。
