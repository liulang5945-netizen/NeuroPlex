import pathlib

content = r'''# Taiji Neuron 架构机制详细解析

> **版本**: v2.1
> **日期**: 2026-07-16
> **状态**: 核心机制已实现，v2 架构改进已合入

---

## 一、架构全景

### 1.1 三层抽象

```
Level 0: 通用分词器 (256K vocab)
    |  token_id -> embedding lookup
    v
Level 1: 领域神经元 (N 个独立 Transformer)
    |  embed_adapter -> Transformer blocks -> field_write / field_read
    v
Level 2: 共振场 (4096-dim 共享向量空间)
    |  L2-normalized writes -> cosine similarity scoring -> complementarity
    v
输出: per-position 加权 logits -> 下一 token
```

核心思想：用一个 4096 维的共享向量空间取代"单体大模型内部隐状态"
作为知识协作的介质。每个神经元独立处理输入、向场写入自己的理解，
再从场中读取其他神经元的理解来修正自身输出。

"1+1>2" 的含义：两个神经元的共振 PPL 低于任一单独神经元的 PPL。
这要求场通信承载的信息足够丰富，且加权机制能正确地**按位置**选择
最合适的神经元。

### 1.2 信息流详图

```
输入 token_ids [B, L]
    |
    v
共享嵌入表 Embedding(vocab=256000, dim=512)
    |  shared_embeddings: [B, L, 512]
    |
    +-----> neuron_A.forward(emb)
    |           |
    |           v
    |       embed_adapter: 512 -> hidden_A
    |           |
    |           v
    |       TransformerBlocks x N (RoPE + GQA + SwiGLU)
    |           |
    |           +<--- field_read (round 2+): gated per-position
    |           |
    |           v
    |       attention-pooled field_write -> [B, 4096] (L2-normalized)
    |           |
    |           v
    |       lm_head -> [B, L, 256000]
    |
    +-----> neuron_B.forward(emb, field_state=...)
    |           ... 同上，但读取 neuron_A 写入的场状态 ...
    |
    ... 其他神经元 ...
    |
    v
共振场 (累积所有写入)
    |
    v
per-position 路由: logit-entropy 置信度 x 互补性加成
    |
    v
weighted_logits [B, L, 256000] -> 下一 token 概率分布
```

### 1.3 与单体模型的区别

| 维度 | 单体 1.5B ModelSelf | 共振场多神经元 |
|---|---|---|
| 大脑 | 一个 1.5B Transformer | N 个 24M-118M 独立 Transformer |
| 内部通信 | 隐状态层间直连，一切在一个 forward 内 | 通过场向量间接通信，多轮迭代 |
| 推理 | 单次 forward | 多轮：Round 1 独立 -> Round 2+ 条件化 |
| 扩展 | 重新训练整个模型 | 热插拔新神经元 |
| 知识定位 | 全部分散在所有参数中 | 领域专用、各有指纹方向 |
| 路由 | 不需要 | **核心挑战：如何在每个位置选对神经元** |

---

## 二、单个神经元的内部机制

### 2.1 结构概览

每个 ResonanceNeuron ([neuron.py](../taiji/resonance/neuron.py)) 是一个完整的
Transformer，加上三个与场交互的投影层：

```
输入: shared_embeddings [B, L, 512]
  |
  v
embed_adapter (Linear 512 -> hidden)     <-- 把共享嵌入投影到自己的概念空间
  |
  v
+-- Transformer Block i (i=0..N-1) -----+
|  attention_norm -> GroupedQueryAttention -> residual                      |
|  ffn_norm -> SwiGLU -> residual                                         |
|  + field_read_layers[i] (round 2+): gated per-position conditioning     |
+------------------------------------------------------------------------+
  |
  v
norm (RMSNorm)
  |
  +--> lm_head (Linear hidden -> 256000)     <-- 语言建模输出
  |
  +--> attention pooling -> field_write (Linear hidden -> 4096)  <-- 场写入
       (L2-normalized)
```

### 2.2 嵌入适配器 (embed_adapter)

```python
embed_adapter = nn.Linear(base_embed_dim=512, hidden_size, bias=False)
```

所有神经元共享一个 512 维的嵌入表，但每个神经元把自己的概念空间
投影到不同的隐藏维度（compact=512, standard=768, expert=1024）。
`embed_adapter` 学习从通用语义空间到该神经元专属理解的映射。

为什么不直接共享隐藏维度？因为不同规模的神经元需要不同的表达力。
expert 1024 维能表达更细腻的代码语义，compact 512 维已足够辅助任务。
统一嵌入维度的代价是信息损失；用可训练的线性投影来弥合比强制统一更好。

### 2.3 Transformer 主体

直接复用 [layers.py](../taiji/layers.py) 的零改动组件：

- **RMSNorm**: 比 LayerNorm 快且稳定，去掉 mean centering
- **RoPE**: 旋转位置编码，外推能力好，用 LRU 缓存避免内存泄漏
- **GQA**: 分组查询注意力，KV heads < Q heads，省显存且质量不降
- **SwiGLU**: 门控激活，比 GELU 效果好
- **Flash Attention**: PyTorch 2.0+ 原生 `scaled_dot_product_attention`
- **Pre-Norm**: 在 attention/ffn 之前归一化，训练更稳定

每个神经元用自己的层数和维度（compact 6L / standard 10L / expert 14L），
但核心算子完全相同。这保证了蒸馏和训练是一致的。

### 2.4 场写入：注意力池化 (v2 改进)

#### 旧版（v1）：最后 token 写入

```python
hidden_last = h[:, -1, :]              # 只取最后一个位置
v_raw = self.field_write(hidden_last)  # [B, D]
```

问题：256 个 token 序列的理解被压缩成最后一步的隐状态。就像一个人读完
一整篇文章，但对外只说一个词，丢失了前面 255 个位置的信息。

#### 新版（v2）：注意力池化写入

```python
attn_scores = torch.matmul(h, self.field_pool_query) * scale  # [B, L]
attn_weights = torch.softmax(attn_scores, dim=-1)              # [B, L]
pooled = (attn_weights.unsqueeze(-1) * h).sum(dim=1)          # [B, hidden]
v_raw = self.field_write(pooled)                               # [B, D]
```

`field_pool_query` 是一个可学习的 query 向量，对序列所有位置做注意力，
自动决定哪些位置最值得代表这个序列的"概念内容"。
对于代码序列，它可能关注函数签名行；对于中文文本，它可能关注主题句。

**为什么用单 query 而不是多头注意力？** 场通信需要的是**单一概括向量**，
不是 per-head 的多维表示。单 query 参数量极小（hidden 个参数），
却能学到"该从哪个角度概括这段输入"。

### 2.5 场读取：位置门控 (v2 改进)

#### 旧版（v1）：全局广播

```python
conditioning = self.field_read_layers[i](field_state)  # [hidden]
h = h + conditioning.unsqueeze(0).unsqueeze(0)         # 广播到所有位置
```

同一个 conditioning 向量加到序列上所有 256 个位置。无法做到"在前半段
听 neuron A，后半段听 neuron B"。对于中文里嵌代码的混合文本，这种粒度太粗。

#### 新版（v2）：per-position 门控

```python
projection = conditioning.unsqueeze(0).unsqueeze(0)          # [1, 1, hidden]
gate = torch.sigmoid(self.field_read_gate(h))                 # [B, L, 1]
h = h + gate * projection                                     # 逐位置选择性吸收
```

`field_read_gate` 是一个 `Linear(hidden, 1)` 层。每个位置根据自己的隐状态
计算一个 0-1 的门值，决定这个位置多大程度吸收场的条件信息。

直觉：模型在某段代码处，gate 可能接近 0（不需要中文神经元的帮助）；
在下一段中文说明处，gate 自动打开。这种机制让一个神经元可以在"需要时"
才听场的话，不被无关信息干扰。

### 2.6 规格差异

| 规格 | hidden | layers | heads | KV heads | intermediate | 参数量 | 角色 |
|------|--------|--------|-------|----------|-------------|--------|------|
| compact | 512 | 6 | 8 | 2 | 1536 | ~24M | 辅助执行 |
| standard | 768 | 10 | 12 | 4 | 2304 | ~59M | 主要执行 |
| expert | 1024 | 14 | 16 | 4 | 3072 | ~118M | 决策+把关 |

这种三层角色体系来自大脑的启发：前额叶（决策）-> 运动皮层（执行）->
辅助系统（协作）。通过 ScaleLayering 机制继承到分工路由中。

### 2.7 语言建模头 (lm_head)

```python
lm_head = nn.Linear(hidden_size, vocab_size=256000, bias=False)
```

每个神经元独立预测整个 256K 词表。这是 PPL 评估的基础：
单神经元的 forward 直接输出 logits，可以直接计算 cross-entropy loss。

vocab=256000 看起来很大，但这是 256K 通用分词器的全局词表。即使某个神经元
是 code 专用，它仍然需要输出在这个词表上的概率分布，因为下游的 per-position
路由需要所有神经元在**同一个输出空间**里比较置信度。

### 2.8 指纹方向 (fingerprint)

```python
@torch.no_grad()
def freeze_fingerprint(self):
    fp = self.field_write.weight.mean(dim=0)  # [hidden]
    self.fingerprint.copy_(fp / (fp.norm() + 1e-8))
```

指纹是 `field_write` 权重行的归一化均值，代表这个神经元在概念空间中的
"主方向"。它用于快速预筛选：如果两个神经元的指纹几乎正交（|cos| < 0.3），
它们大概率覆盖不同领域，值得考虑共振；如果几乎平行，可能冗余。

这是未来部落级 optimization 的基础，当前实现中主要做诊断用途。

---

## 三、共振场机制

### 3.1 场的数据结构

ResonanceField ([field.py](../taiji/resonance/field.py)) 本质是一个 4096 维向量
加上一些元数据：

```python
class ResonanceField(nn.Module):
    state: torch.Tensor          # [4096] 累积写入向量
    W_cond: nn.Parameter         # [4096, 4096] 条件化矩阵（可学习）
    scores: Dict[str, float]     # 最新一轮的各神经元共振分
    n_active: int                # 当前活跃写入数
```

`W_cond` 是一个可学习的变换矩阵，设计用于未来"场条件化"（哪些神经元
组的组合模式产生好输出）。当前未训练，保留接口。

### 3.2 写入：L2 归一化

```python
def write(self, neuron_id, vector):
    v_norm = vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)
    self.state = self.state + v_norm.squeeze(0)
    self.n_active += 1
```

关键设计：所有写入都被 L2 归一化。这保证**神经元的"音量"不取决于其参数量**。
一个 24M 的 compact 神经元和一个 118M 的 expert 神经元写入场的向量长度相同，
在场状态中贡献一样多。这是公平协作的前提。

场状态是所有写入向量的简单累加。没有任何平滑或衰减——每次推理时重置为零，
累积一轮，使用，然后重置。

### 3.3 共振打分：对齐 vs 互补 (v2 改进)

#### 旧版（v1）：纯对齐打分

```python
def score(self, vector):
    return cosine(vector, self.state)  # 范围 [-1, 1]
```

问题：分高只说明"和集体一致"，不等于"预测正确"。两个都错但方向一致的
神经元会互相强化，得到高权重。更糟糕的是，它**惩罚互补性**——如果 neuron B
写了与场状态正交的方向（带来了全新的信息），它的对齐分接近 0，
反而被滤波滤掉。

#### 新版（v2）：互补性打分

```python
def complementarity_score(self, vector):
    v_norm = vector / (vector.norm() + 1e-8)
    if self.state.norm() < 1e-8:
        return 1.0  # 空场：一切都是新的
    f_norm = self.state / (self.state.norm() + 1e-8)
    alignment = dot(v_norm, f_norm)
    orthogonal = v_norm - alignment * f_norm  # 去掉投影，留正交分量
    return orthogonal.norm()  # [0, 1]
```

互补性测量的是 v 相对于当前场状态的**正交分量**——也就是 v 带来的
当前场中还不存在的信息。高互补分说明这个神经元说的东西和已有的不一样，
可能填了空白。

#### 混合打分

```python
def combined_score(self, vector, alpha=0.5):
    align_01 = (self.score(vector) + 1.0) / 2.0  # 对齐映射到 [0,1]
    comp = self.complementarity_score(vector)
    return (1 - alpha) * align_01 + alpha * comp
```

`alpha=0.5` 是默认的平衡点。太偏对齐回到"从众"问题；太偏互补可能
引入无关噪声。这个超参数需要实验调优。

### 3.4 拥堵度与动态阈值

```python
def directional_congestion(self, vector, active_vectors):
    # 与所有活跃向量的平均 cosine 相似度
    ...

def compute_threshold(self, directional_congestion):
    return 0.30 + directional_congestion * 3.0
```

拥堵度衡量"有多少其他神经元在相似方向上写"。高拥堵意味着这个方向
已经很拥挤，后面进入的同类神经元需要更高的共振分才能留下。

阈值公式：低拥堵 (0.1) -> T=0.60（容易进）；高拥堵 (0.85) -> T=2.85（几乎不可能）。
这是一种隐式的**去冗余**机制——如果场里已经有三个 code 神经元，
第四个方向相似的 code 神经元不太可能被选中。

### 3.5 分工路径 (DivisionPath)

[division.py](../taiji/resonance/division.py) 实现了两层策略：

**ScaleLayering（策略 A）**：不同规格的神经元权重不同。
```python
SPEC_WEIGHTS = {"expert": 3.0, "standard": 2.0, "compact": 1.0}
```
expert 的权重是 compact 的 3 倍——大模型在决策上更有话语权，
小模型主要提供辅助信号。

**ClusterDominance（策略 B）**：最匹配输入的"集群"主导。
```python
cluster_fit = internal_coherence * external_relevance
# dominant cluster 权重 0.7，其余分 0.3
```
一个集群的内部一致性（成员间 cosine 均值）乘以它与输入向量的相关度，
决定哪个集群主导。

分工路径**不是**共识路径（加权平均）的替代——它是当 `division_path`
被传入 ensemble 时使用的权重模式。如果未传入，ensemble 用默认的
per-position 路由（v2 新增）。

### 3.6 部落压缩质量 (tribal.py)

[tribal.py](../taiji/resonance/tribal.py) 是三级架构的第三层：

```
Level 1: 领域神经元（个体）
Level 2: 部落 = N 个经常共激活的神经元压缩成一个超神经元
Level 3: 共振场（部落和个体混在同一个场里）
```

`TribalMetrics` 量化一个部落的压缩质量：

- **alpha（coherence）**: 成员间 pairwise cosine 均值。高 -> 输出方向可信。
- **beta（stability）**: 子场状态在轮次间的指数衰减变化率。高 -> 快速收敛。
- **gamma（spread）**: 成员到质心的平均欧氏距离的倒数。高 -> 紧密聚集。

**Q = alpha * beta * gamma** 是信号质量因子。Q >= 0.8 时上级场可以重度
依赖这个部落的输出；Q <= 0.05 时应该忽略它。

`compression_loss` 测量部落对外向量 (v_tribe) 能否代表所有成员的写入。
残差均值超过 0.5 触发解散条件——成员差异太大，强行压缩降低信息效率。

---

## 四、共振循环机制

### 4.1 整体流程

ResonanceEnsemble ([ensemble.py](../taiji/resonance/ensemble.py)) 编排
多轮共振：

```
Round 1: 所有神经元独立前向
    -> 各自写入场
    -> 计算共振分 + 互补分
    -> 门控检查：要不要继续？

Round 2-N: 条件化共振
    -> 读取归一化场状态
    -> 门控式条件吸收(逐位置)
    -> 重新写入场
    -> 计算新分数
    -> 过滤低分神经元
    -> 早停检查

最终输出: per-position 加权 logits
```

### 4.2 门控机制

三个门控来自 Experiment 12，都在 [gating.py](../taiji/resonance/gating.py)：

**ConfidenceGate**：如果最匹配的神经元已经很自信（max_prob > 0.9），
跳过共振。这避免对确定预测的过度思考。

**EarlyStopResonance**：连续两轮加权 logits 的相对 L2 差 < 1e-3 时停止。
logits 收敛意味着更多轮不会改变结果，省时间并避免噪声累积。

**ResonanceTrigger**：三个条件全部满足才共振：
1. 预测不确定（max_prob < 0.9）
2. 多个神经元有不同知识（field_vector 多样性 > 0.3 cosine distance）
3. 有改善空间（不近完美预测）

### 4.3 质量过滤

[quality.py](../taiji/resonance/quality.py)：PPL >= 100 的神经元不参与共振。
这是防止"最弱一环"稀释强神经元的机制。自适应阈值用最佳 PPL * 2，
在困难领域（math PPL 天然高）更宽松。

### 4.4 逐位置路由 (v2 改进)

#### 旧版（v1）：全局标量加权

```python
weights = softmax(resonance_scores * 2.0)  # [N] 每个神经元一个标量
weighted_logits = sum(weights[i] * logits_i)
```

一个问题：一个标量权重施加在所有 256 个位置上。如果 neuron_A 在前 100 个
token 上最好，neuron_B 在后 156 个 token 上最好，你无法分别选它们。

#### 新版（v2）：per-position 置信路由 + 互补加成

```python
# 1. 每个位置上，计算每个神经元的 logit 熵
for nid in neuron_ids:
    log_probs = F.log_softmax(all_logits[nid], dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]
    entropies.append(entropy)

# 2. 置信度 = 1/entropy，softmax over neurons
confidence = 1.0 / (ent_stack + 1e-8)       # [N, B, L]
position_weights = F.softmax(confidence * 2.0, dim=0)  # [N, B, L]

# 3. 互补分加成：带来新信息的神经元权重放大
comp_boost = (1.0 + complementarity_scores)  # [N, 1, 1]
position_weights *= comp_boost
position_weights /= position_weights.sum(dim=0, keepdim=True)

# 4. 逐位置加权 logits
weighted_logits = sum(position_weights[i].unsqueeze(-1) * logits_i)
```

直觉：在每个位置上，**最自信的神经元说话最大声**，但互补分让你在平局
时倾向带来新信息的神经元。对于混合领域文本（中英夹杂、代码嵌数学），
这让每个 token 区域都能选到最适合的那个神经元。

内存优化：不 stack 整个 [N, B, L, 256000] 的 logits（太大了），
而是每次只处理一个神经元的 [B, L, 256000] 来算熵，最后用 [N, B, L] 的
权重矩阵来加权。

---

## 五、训练管线

### 5.1 单神经元训练

[single.py](../taiji/training/single.py)：标准的 LM loss 训练，
每个神经元独立优化自己的参数。目标是让每个神经元在自己的领域文本上
达到 PPL < 50。

### 5.2 蒸馏管线

[distill.py](../taiji/training/distill.py)：从 1.5B teacher 蒸馏到小神经元。

```
Teacher (1.5B) --forward--> hidden_states [2048]
                            |
                            v
Student neuron --forward--> hidden_before_write [hidden]
                            |
                            v
loss = 0.7 * LM_loss + 0.3 * distill_loss(MSE)
```

蒸馏 loss 是 student 池化位置的隐状态与 teacher 最后 token 隐状态之间的 MSE。
这让 student 直接继承 teacher 的语义理解能力，只需学会用自己的方式表达。

### 5.3 对比学习 (contrastive.py)

让不同领域的 field_write 方向分离：同领域 -> 拉近对齐 teacher 方向，
跨领域 -> 推开（hinge loss on cosine > 0.3）。这确保场向量天然形成
领域聚类，不需要手动标 cluster。

### 5.4 检查点桥接

[checkpoint_bridge.py](../taiji/training/checkpoint_bridge.py)：加载一代
teacher checkpoint，做 key remapping（`embed` -> `embedding`，
`attn` vs `attention`，`w1/wg/w2` vs `w1/w_gate/w2`），
用 importlib 隔离一代 taiji 包的导入，避免与二代 taiji-neuron 冲突。

### 5.5 联合训练

[joint.py](../taiji/training/joint.py)：所有神经元共享一个 embedding table，
联合训练让 embed_adapter 学习到一致的从 512 维到各自 hidden 的投影，
同时让 field_write/field_read 协同优化。

---

## 六、分词器协议

### 6.1 全局词表 (256K)

[tokenizer_native_v2.py](../taiji/tokenizer_native_v2.py)：256K SentencePiece
词表作为通用的 I/O 协议。所有输入和输出都在这个空间里编码。

### 6.2 领域分词器 (32K-48K)

[translator.py](../taiji/resonance/translator.py)：每个神经元可以有
自己的领域优化分词器。`TokenTranslator` 在全局词表和领域词表之间双向翻译。
`TokenizerHub` 管理热插拔——新增领域分词器不影响已有神经元。

三层解耦：全局分词器（I/O）-> 领域分词器（内部）-> 共振场（完全独立）。
场的语义空间不依赖任何词表设计，这是架构可扩展的根基。

---

## 七、v2 架构改进总结

### 7.1 变更对照

| 模块 | v1 (旧) | v2 (新) | 问题 | 解决 |
|------|---------|---------|------|------|
| neuron field_write | 取最后 token | 注意力池化 | 单 token 信息瓶颈 | 学一个 query 概括全序列 |
| neuron field_read | 全局广播 | 逐位置门控 | 无法分区域听 | sigmoid gate 控制吸收量 |
| field scoring | 纯 cosine 对齐 | 对齐 + 互补 | 互补被惩罚 | 正交分量作为额外信号 |
| ensemble 路由 | 标量 softmax | per-position 置信路由 | 无法分位置选神经元 | logit-entropy 逐位置加权 |

### 7.2 为什么这些改动让 1+1>2 更可能

v1 的四条失败因果链：

```
field_write 只用最后 token
    -> field_vector 信噪比低
    -> 共振路由失去了对前文的区分能力
    -> ensemble 输出接近随机平均

field_read 全局广播
    -> 条件信号在无关位置造成干扰
    -> 与本领域信号冲突
    -> 共振比舍入更差

纯对齐打分
    -> 互补神经元被惩罚
    -> 只有相似神经元留下
    -> 1+1 = 1（自言自语）

标量加权
    -> 所有位置同一个权重
    -> 混合文本中弱神经元拖累强区间
    -> ensemble PPL > best single
```

v2 打断每条链：池化让向量富含信息，门控让吸收有选择，互补分让新知识
被奖励，per-position 路由让每个 token 选最合适的神经元。

### 7.3 开放问题

- **互补性的正确性**：高互补不等于正确的互补。一个完全错误的预测也可能
  在正交方向上，得到高互补分。需要实际 PPL 验证来确认互补是有用的。
- **门控的冷启动**：`field_read_gate` 没有预训练时门接近 0（bias 初始化），
  共振在早期不起作用。需要蒸馏时联合训练 gate。
- **W_cond 未训练**：现在 `score()` 用 raw cosine，
  可学习的 `W_cond` 已有但未接入循环中。这是下一步该做的事。
  `W_cond` 学到的本质是"哪些神经元组合模式对应好输出"，
  它应该替代 raw cosine 作为共振信号。

---

## 八、模块依赖图

```
ensemble.py
  |-- field.py       (场状态、共振分、互补分)
  |-- neuron.py      (单神经元 forward)
  |-- gating.py      (ConfidenceGate, EarlyStop, Trigger)
  |-- quality.py     (QualityFilter)
  |-- division.py    (ScaleLayering, ClusterDominance)

neuron.py
  |-- layers.py      (TransformerBlock, RMSNorm)
  |-- config.py      (NeuronConfig)

cortex.py
  |-- ensemble.py
  |-- field.py
  |-- config.py

distill.py
  |-- neuron.py (student)
  |-- checkpoint_bridge.py (teacher)

cortex.py is the high-level wrapper used by:
  |-- api/routes_chat.py
  |-- agent/planner.py
  |-- agent/reflector.py
```
'''

pathlib.Path('docs/NEURON_MECHANISM_ANALYSIS.md').write_text(content, encoding='utf-8')
print(f'Written {len(content)} chars to docs/NEURON_MECHANISM_ANALYSIS.md')
