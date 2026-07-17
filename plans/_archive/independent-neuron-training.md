# 独立神经元训练方案 — 去 Teacher 依赖

## 当前问题

三阶段管线的 Phase 1 依赖 1.5B teacher：

```
Phase 1: 从 1.5B teacher 蒸馏 ← 每个新神经元都需要 teacher
Phase 2: 联合 field conditioning
Phase 3: 共振验证
```

## 目标：零 teacher 依赖

新神经元应该从**领域数据**直接训练，不依赖任何外部模型：

```
新领域数据 → 独立 LM 训练 → field_write 自监督 → field conditioning → 接入共振场
```

## 改造方案

### Phase 1 改造：从蒸馏 → 从零训练

| 当前（蒸馏） | 改后（从零训） |
|-------------|--------------|
| Teacher embedding → 投影到 512 | 随机初始化共享嵌入 → 512 |
| Teacher hidden → MSE loss | ❌ 去掉 |
| Teacher direction → 对比 loss | 领域内聚类方向 → 对比 loss |
| lm_loss + distill_loss + field_loss | **lm_loss + field_loss** |

### field_contrastive loss 的自监督替代

没有 teacher direction 时：

```python
# 方式1: 用 field_write 自身的 PCA 主方向
# 同一领域的不同 batch → field_vector 应该聚在一起
# 不同领域的 field_vector → 应该分散

# 方式2: 用领域标签做对比学习
# batch_A (领域 zh) → field_vectors_A
# batch_B (领域 code) → field_vectors_B
# loss: cos(fv_A[i], fv_A[j]) 大, cos(fv_A[i], fv_B[k]) 小

# 方式3: 离线预计算领域代表方向
# 训练前，用已有数据跑一次前向 → 收集 field_vectors
# 取平均值作为该领域的"锚点方向"
# 训练时 field_vector → pull toward 锚点
```

最简单实现：**方式3**，与当前代码改动最小。

### 代码改动范围

| 文件 | 改动 |
|------|------|
| [`distill_neurons.py`](scripts/training/distill_neurons.py) | Phase 1 新增 `--no_teacher` 模式 |
| 新增约 30 行 | 跳过 teacher 加载 + 调整 loss 权重 |

### 使用方式

```bash
# 从 Teacher 蒸馏（现有）
python distill_neurons.py --checkpoint ... --steps 2000

# 从零训练新神经元（新增）
python distill_neurons.py --no_teacher \
    --data_dir data/real \
    --steps 5000 \
    --field_contrastive_weight 0.3 \
    --field_cond_steps 500
```

区别只是加了一个 flag。
