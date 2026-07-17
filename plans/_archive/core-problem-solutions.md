# 共振场四大核心问题 — 解决方案

> 基于各问题根因分析的具体修复路径

---

## 问题 1：field_vector 含义模糊 → field_write 未经训练

**根因**：[`distill.py`](taiji/training/distill.py) 的蒸馏 loss 只有 `lm_loss + distill_loss`，`field_write` 投影层收不到有意义的梯度。

```
当前 loss = 0.7 * LM_loss + 0.3 * distill_loss
                              ↑ 只对齐 hidden state
                              field_write 投影处于随机初始化状态
```

### 方案 A：蒸馏时加入 field_write 对比学习（推荐先做）

修改 [`distill.py`](taiji/training/distill.py) 中的 `distill_neuron()`：

```python
# 新增: 对比学习 loss，训练 field_write
def field_contrastive_loss(student_field_vecs, domain_label, 
                           other_domain_samples):
    """
    student_field_vecs: [B, D] — 本领域样本的 field_vector
    other_domain_samples: {domain: [B, D]} — 其他领域样本的 field_vector
    
    目标: 同领域 field_vector 彼此靠近，不同领域推远
    """
    # 同领域: 与教师方向对齐
    pos_loss = (1 - cosine(student_field_vecs, teacher_direction)).mean()
    
    # 异领域: 推远
    neg_loss = 0
    for other_domain, other_vecs in other_domain_samples.items():
        # hinge loss: cosine < 0.3 就不惩罚（已足够远）
        cos = cosine(student_field_vecs, other_vecs)
        neg_loss += torch.clamp(cos - 0.3, min=0).mean()
    
    return pos_loss + 0.1 * neg_loss

# 在 distill_one_neuron() 中:
loss = lm_weight * lm_loss + distill_weight * distill_loss + 0.2 * contrastive_loss
```

**关键**：跨领域的负样本从哪里来？
- 蒸馏 neuron_zh 时，用 neuron_en、neuron_code 的 field_vector 作为负样本
- 蒸馏 neuron_code 时，用 neuron_zh、neuron_math 的 field_vector 作为负样本

### 方案 B：用教师方向初始化 field_write

在创建神经元时，不随机初始化 `field_write.weight`，而是用教师模型在该领域的隐藏态方向来初始化：

```python
# 伪代码
teacher_direction = extract_teacher_directions(teacher, domain_data)  # [2048]

# SVD 降维: 2048 → D (如 3072 或 4096)
U, S, V = torch.svd(teacher_direction.unsqueeze(0))  
init_weight = V[:, :D]  # 取前 D 个主方向

neuron.field_write.weight.data = init_weight
```

这样做的好处：field_vector 一开始就指向有意义的领域方向，不需要对比学习也能有一定的领域区分度。

---

## 问题 2：分工 = 标量加权 → 需要 per-position 路由

**现状**（[`ensemble.py:290`](taiji/resonance/ensemble.py:290)）：

```python
# 所有位置用同一个权重
weighted_logits = Σ w_i × logits_i  # w_i 是标量
```

**理想**：不同位置用不同神经元的 logits。

### 方案：per-position 软路由

修改 [`ensemble.py`](taiji/resonance/ensemble.py) 的最终输出部分：

```python
# 每个位置，每个神经元计算"契合度"
# field_vector 表达了"这个输入整体属于我的领域"
# 但还需要 per-token 的"这个具体 token 我擅长吗"

def compute_position_weights(field_vectors, logits_dict, hidden_states):
    """
    field_vectors: {nid: [B, D]} — 每个神经元的领域信号
    logits_dict:   {nid: [B, L, vocab]} — 每个神经元的预测
    hidden_states: {nid: [B, L, hidden]} — 每个神经元的隐藏态
    
    Returns: [B, L, N] — 每个位置/每个神经元的权重
    """
    B, L, _ = next(iter(logits_dict.values())).shape
    N = len(logits_dict)
    weights = torch.zeros(B, L, N)
    
    # 策略1: logits 熵越低 → 越确定 → 权重越高
    for i, (nid, logits) in enumerate(logits_dict.items()):
        probs = F.softmax(logits, dim=-1)  # [B, L, vocab]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, L]
        weights[:, :, i] = 1.0 / (entropy + 1e-8)  # 低熵 → 高权重
    
    # 策略2: 全局 field_vector 得分作为偏置
    for i, nid in enumerate(logits_dict):
        field_score = cosine(field_vectors[nid], field_state)  # 标量
        weights[:, :, i] *= field_score
    
    # Softmax 归一化
    weights = F.softmax(weights, dim=-1)  # [B, L, N]
    
    # 加权组合
    weighted = sum(weights[:,:,:,i:i+1] * logits_dict[nid] 
                   for i, nid in enumerate(logits_dict))
    return weighted
```

**实现优先级**：
1. 先方案 A（对比学习训练 field_write）— 这是前提
2. field_vector 有意义之后，per-position 路由才有用
3. 当前可先用标量加权，等 field_vector 训练好后再升级

---

## 问题 3：集群 = 手动 → 需要涌现

**当前**：硬编码 `{"language": ["zh","en"], "code": ["code"]}`

### 方案：三阶段涌现场景

#### 阶段 1：field_vector 聚类（被动发现）

当 field_write 训练好后（解决 Problem 1），field_vector 自然形成领域聚类：

```python
# 对每个神经元的 field_write.weight 做 PCA/聚类
# 不需要手动标记，field_vector 的方向自动表达了领域
# 例如：
#   zh neuron: field_vector 方向 [0.8, 0.1, 0.05, ...]
#   en neuron: field_vector 方向 [0.75, 0.15, 0.03, ...]  ← 相似！
#   code neuron: field_vector 方向 [0.02, 0.9, 0.01, ...]  ← 完全不同！

# 用 cosine 相似度自动分组:
clusters = auto_cluster_by_fingerprint(neurons, threshold=0.7)
# → {"language": ["zh", "en"], "code": ["code"], "math": ["math"]}
```

#### 阶段 2：共激活追踪（主动发现）

[`tribal.py`](taiji/resonance/tribal.py) 中 `CoactivationTracker` 已就位，当有真实推理数据后：

```python
tracker = CoactivationTracker(ema_alpha=0.1, threshold=0.6)

for each inference:
    active_neurons = [nid for nid, score in resonance_scores.items() if score > 0.3]
    tracker.update(active_neurons)

# 定期检查: 哪些神经元经常一起激活？
groups = tracker.get_dense_groups(min_size=2)
# → [["zh", "en"], ["code", "math"]]  
#    ↑ 涌现结果，不是手动标记
```

#### 阶段 3：部落化（主动重组）

当共激活密度超过阈值，自动触发 `TribeSuperNeuron` 创建：

```python
if len(group) >= MIN_TRIBE_SIZE and coactivation_density > 0.7:
    tribe = TribeSuperNeuron(tribe_id=next_id, members=group, sub_field=...)
    # 部落对外表现为一个超级神经元
    # 上级场不需要知道内部有 N 个成员
```

---

## 问题 4：1+1>2 未观测到

**根因因果链**：

```
Problem 1（field_write 未训练）
    ↓
field_vector 随机 → 共振路由失效
    ↓
Problem 2（标量加权）放大了路由失效的影响
    ↓
Problem 3（手动集群）无法利用神经元间的自然协同
    ↓
1+1 ≈ 0.8（多个弱信号平均后比最好的单个还差）
```

**验证 1+1>2 的最小可行实验**：

1. **训练 field_write**（方案 A + B）
2. **准备两个互补的神经元**：
   - neuron_A：训练的很好（PPL < 50）
   - neuron_B：在 A 不擅长的子领域训练好
3. **准备混合测试数据**：50% A 擅长 + 50% B 擅长
4. **对比**：
   - 单独 A 的 PPL
   - 单独 B 的 PPL  
   - A + B 共振的 PPL
   - **预期**：共振 PPL < min(单独A, 单独B)

---

## 执行优先级

```
P0（必须）: 方案 A+B — 训练 field_write
    │  修改 distill.py 加入对比学习
    │  用教师方向初始化 field_write
    ↓
P1（重要）: 方案 阶段1 — field_vector 自动聚类
    │  验证 field_vector 是否形成领域聚类
    │  替换手动 clusters 为自动发现
    ↓
P2（提升）: per-position 路由
    │  修改 ensemble.py 的加权逻辑
    │  从标量权重 → per-token 注意力权重
    ↓
P3（验证）: 1+1>2 最小实验
        准备互补神经元对
        对比单独 vs 共振 PPL
```

**当前最紧急**：修改 [`distill.py`](taiji/training/distill.py) 加入 field_write 对比学习 loss，这是整个架构的"最后一公里"。
