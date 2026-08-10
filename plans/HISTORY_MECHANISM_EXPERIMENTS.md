# 机制实验与里程碑记录

> **拆分文档**（2026-08-10）：从 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](BIO_INSPIRED_ARCHITECTURE_PLAN.md) 按内容拆分。
> 记录机制实验的里程碑与结论（EMERGE 确认、aux-free balancing、shared expert 负向）。
> 当前项目状态、路线图与接口梳理以主 plan 为准。

**内容**：
- 重大里程碑：EMERGE 现象确认（2026-07-29）
- Auxiliary-loss-free balancing 实施（2026-07-29）
- Shared Expert 机制实施（2026-07-29，负向结论）

---

### 实验结果

**4 神经元协作 side_channels 微调（6 epochs，14950 步，~14 小时）**

| 神经元 | solo PPL | 融合权重 |
|--------|---------|---------|
| zh_aug0 | 211.6 | 0.250 |
| zh_aug1 | 114.6（最强个体） | 0.555（主导） |
| zh_aug2 | 225.3 | 0.129 |
| zh_aug3 | 246.9 | 0.066 |
| **协作** | **62.6** | - |

### 关键指标

- **协作 PPL: 62.6**
- **最强个体 PPL: 114.6**
- **EMERGE 幅度: 协作比最强个体好 45.3%**

### 技术配置

- 神经元规格：4× compact（36M 参数/个，共 144M）
- 可训练参数：side_channels 12.58M + scale 12 个
- 优化器：Muon（ns_steps=5, momentum=0.95, nesterov=True）
- 学习率调度：warmup(100步) + constant + cosine decay(最后 20%)
- side_channels 调制：乘性门控 `h = h * (1 + tanh(proj))`
- field_conditioning：False（跳过未训练的 field_read_layers）
- max_rounds：2（round 1 独立，round 2 带 side_signals）
- 数据：simple_zh 10000 条，6 epochs

### 意义

这是态极架构的**关键验证点**：多个小型神经元通过 side_channels 协作，涌现出超越最强个体的能力。验证了"小神经元协同工作匹配大模型"的核心设计理念。

### 生成质量

PPL 指标确认协作有效，但生成文本仍有重复（所有神经元共性问题）。原因是 compact 神经元训练不充分（CPU 限制，数据/参数比不足）。后续需要：
1. 更充分的神经元训练（更多数据，更多步数）
2. 改进生成策略（sampling, repetition penalty）
3. 扩大规模（更多神经元，更大规格）

### 生成质量评估（2026-07-29，top-k sampling 改进后）

**PPL 结果**（v2 baseline，已恢复）：
- 个体 [zh_aug0]: PPL=211.6
- 个体 [zh_aug1]: PPL=114.6（最强个体）
- 个体 [zh_aug2]: PPL=225.3
- 个体 [zh_aug3]: PPL=246.9
- **协作 PPL=62.6**（EMERGE 协作比最强个体好 45.3%）

**生成质量观察**：
- top-k sampling (k=40) + repetition penalty (1.2) + temperature (0.8) **消除了机械重复**
  （之前是"天气天气天气..."纯重复，现在是有变化的生成）
- 但生成文本仍**语义不连贯**，有乱码、断裂、混合多种风格
- 这是典型的"PPL 好但生成差"问题，根因是 compact 神经元训练不充分

**根本瓶颈确认**：神经元训练不充分是当前所有生成质量问题的根因。架构改进（side_channels、
Auxiliary-loss-free balancing、sampling 策略）已到位，但无法弥补神经元本身能力不足。

### 产物

- 微调权重：`data/neurons/side_channels_finetuned.pt`（v2 baseline, PPL=62.6）
- v2 baseline 备份：`data/neurons/side_channels_finetuned_v2_baseline.pt`
- 训练历史：`logs/finetune_side_channels_history.json`（299 条记录）
- 评估日志：`logs/eval_aug_joint_sampling_20260729_132638.log`

---

## 🔧 Auxiliary-loss-free balancing 实施（2026-07-29）

### 背景

EMERGE 已确认（协作 PPL 62.6 << 最强个体 114.6），但 plans 中标记的"Auxiliary-loss-free balancing"
此前只是框架（scale + bias buffer 已注册，但**启发式 bias 更新逻辑未实现**）。本次完成完整实施。

### 借鉴来源

DeepSeek V3 的 Auxiliary-loss-free Load Balancing：不通过辅助损失，而是用非梯度启发式更新
bias 项，动态平衡各专家（channel）利用率，解决"死通道"问题。

### 实施细节

**`taiji/resonance/neuron.py`**：
1. `__init__`：添加 `_channel_usage: Dict[str, float]` 运行时统计字段（不持久化）
2. `forward` Step 4：side_signal 处理时记录 `proj.detach().abs().mean().item()` 到 `_channel_usage`
3. 新增 `update_channel_bias(update_rate=0.1)`：根据 usage 偏离平均的程度更新 bias
   - `delta = update_rate * (avg_usage - channel_usage)`
   - 低 usage → 正 bias（鼓励激活）
   - 高 usage → 负 bias（抑制过度激活）
4. 新增 `get_channel_usage_stats()`：返回当前 usage 统计（用于日志/诊断）

**`scripts/training/finetune_side_channels.py`**：
1. 每 50 步（`BIAS_UPDATE_EVERY=50`）调用 `update_channel_bias(update_rate=0.1)`
2. 每 50 步（`LOG_EVERY`）输出 channel usage 诊断：avg/min/max/dead count
   - 死通道判定：`usage < avg * 0.1`

### 验证

端到端测试通过：
- 单 channel：avg=usage → delta=0（预期，无竞争）
- 双 channel（强/弱信号）：强 channel 获得负 bias (-8.65)，弱 channel 获得正 bias (+8.65)

### 短训练验证（2026-07-29）

运行 300 条文本 × 1 epoch（74 步）验证 bias 更新机制：

```
[bias update] step 50: 12 channels, total_delta=0.0104
Epoch 1/1 step 50: loss=5.0741 PPL=159.8 [50/74 ETA 1.9min]
  [channels] usage avg=0.3926 min=0.3683 max=0.4160 dead=0/12
```

**关键发现**：
1. ✅ bias 更新机制工作正常（step 50 触发，delta 计算 correct）
2. ✅ channel usage 诊断输出正常（avg/min/max/dead count）
3. ✅ **当前无死通道**（dead=0/12）—— v2 修复（scale=50 + post-norm）已解决死通道问题
4. ✅ bias 更新在 usage 均匀时幅度很小（total_delta=0.0104），不会破坏已训练的平衡

**结论**：死通道问题已被 v2 修复解决，Auxiliary-loss-free balancing 作为"保险"机制存在，
在当前 usage 分布均匀的情况下对 PPL 影响可忽略。无需为此重新完整训练（保留 v2 baseline PPL=62.6）。

---

## 🧠 Shared Expert 机制实施（2026-07-29，已完成，结论：负向）

### 背景

生成质量评估确认根本瓶颈：compact 神经元训练不充分导致生成不连贯（PPL 好但生成差）。
借鉴 Kimi K3 / DeepSeek V3 的 Shared Expert 机制，添加 always-active 的 general 神经元
提供基础语言能力，与域特定神经元互补。

### 借鉴来源

Kimi K3 / DeepSeek V3 的 Shared Expert：一个 always-active 的通用专家，与稀疏激活的
域特定专家协同，提供基础能力保障。

### 实施进度

**1. general 神经元训练**（✅ 已完成）
- 数据：`shared_core.jsonl`（236K 条通用核心数据）
- 规格：36M compact，train 模式（保存自己的 shared_embedding）
- 训练参数：4000 步，batch 8×grad_accum 4=32，lr=1e-3，dropout=0.2
- 结果：best_val_PPL=148.80@step4000，耗时 57.8min
- 训练日志：`logs/train_zh_general_20260729.log`

**2. ensemble Shared Expert 架构**（✅ 已完成）
- `ResonanceEnsemble.__init__` 添加 `shared_expert_id` 和 `shared_expert_weight` 参数
- `forward()` 中 shared expert 始终加入 `active_ids`（不受路由/精简模式影响）
- 最终融合后重新加权：`final = sw * shared_logits + (1-sw) * original_fused`
- 默认 `shared_expert_weight=0.3`（general 神经元获得 30% 固定权重）

**3. 评估脚本支持**（✅ 已完成）
- `eval_aug_joint.py` 添加 `--shared_expert` 和 `--shared_expert_weight` 参数
- `load_aug_neurons()` 支持 `include_shared_expert` 加载 general 神经元
- `eval_ppl()` 和 `eval_generation()` 传递 shared_expert 配置到 ensemble

### 评估结果（2026-07-29，负向）

运行命令：`python -u scripts/training/eval_aug_joint.py --shared_expert --shared_expert_weight 0.3`
评估日志：`logs/eval_shared_expert_20260729_145858.log`

**PPL 对比**：

| 模式 | 协作 PPL | 对比 baseline |
|------|---------|--------------|
| 无 Shared Expert（v2 baseline） | 62.6 | - |
| **Shared Expert (w=0.3)** | **108.6** | **恶化 +73.6%** |

**个体 PPL**：
- zh_aug0: 211.6 | zh_aug1: 114.6（最强个体）| zh_aug2: 225.3 | zh_aug3: 246.9
- **zh_general: 257.5（最弱）** ← 关键问题

**融合权重**：zh_aug1:0.320, zh_general:0.300, zh_aug0:0.122, zh_aug2:0.062, zh_aug3:0.032

### 结论与教训

**Shared Expert 机制在当前实施下负向**，反而降低了协作质量。

**根本原因**：Shared Expert 机制的前提是 general 神经元必须足够强（提供基础能力保障），
但实际 zh_general 神经元训练不充分（best_val_PPL=148.80，评估 PPL=257.5），反而比所有
aug 神经元都差。强制给它 30% 固定权重稀释了 zh_aug1（最强个体）的主导作用。

**关键教训**：
1. **借鉴机制不能盲目照搬**：Shared Expert 在 Kimi K3/DeepSeek V3 中有效，是因为它们的
   general 专家训练充分（数万亿 token）。在小规模 compact 神经元（36M, 4000 步）上，
   general 神经元反而成为最弱环节
2. **机制有效性依赖前置条件**：Shared Expert 要求 general ≥ 域特定神经元的能力，
   否则会拖累整体
3. **固定权重的风险**：30% 固定权重缺乏自适应，无论 general 神经元质量如何都会强制分配，
   应该改为基于神经元实际能力的动态权重

### 后续方向

Shared Expert 机制暂不启用（保持 v2 baseline PPL=62.6 作为最佳协作结果）。
若要重新启用，需要：
1. 大幅增加 general 神经元训练数据量和步数（达到或超过 aug 神经元水平）
2. 改为动态权重（基于 general 神经元实际 PPL 自适应调整）
3. 或放弃固定权重，让 general 神经元参与正常的共振评分竞争

---
