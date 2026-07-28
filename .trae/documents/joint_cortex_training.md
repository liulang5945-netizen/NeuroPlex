# 整体训练态极——多神经元联合训练方案

## Context（为什么做这个）

**问题**：当前范式是"单独训练每个神经元 → 拼装 → 测试协作"。摸底审计发现 0 个神经元能产出连贯句子，且协作测试（L3≈S）不可信——因为神经元从没学过如何协作，协作机制（共振场、族长选择）是手工设计的、不是学出来的。

**用户洞察**："我能不能直接做10个神经元然后一起训练呢，相当于是一个10神经元体积的态极整体进行训练"——直接整体训练态极，让协作在训练中学习，而非事后拼装。

**核心改变**：从 `train_single_neuron × N → assemble → test` 变为 `train_cortex(N neurons together)`。前向传播时所有神经元参与，共振场聚合，反向传播流经协作机制 → 神经元学习如何写入场、如何协同输出。专精化在训练中自然涌现（像 MoE）。

## 关键技术发现

现有代码有两个 **梯度断裂点**阻止联合训练：

1. **`field.score()`** (field.py:385): `return float(sims.mean().item())` — 返回 Python float，断开梯度
2. **`_dynamic_logit_fusion`** (ensemble.py:662): `w = weights[i].item()` — 同样断开梯度

底层数学全部可微（cosine similarity、softmax、weighted sum），只需绕过 `.item()` 保持 tensor 链。

## 实现方案

### Step 1: 在 ResonanceEnsemble 添加 `forward_train()` 方法

**文件**: `taiji/resonance/ensemble.py`

新增单轮、全可微的前向方法，**绕过 field 对象的状态管理**（refractory/WTA/top-K 是推理用的），直接内联计算共振数学：

```python
def forward_train(self, shared_embeddings, temperature=1.0):
    """单轮全可微前向，用于联合训练。
    
    与 forward()（推理）的区别：
    - 单轮（无多轮共振）
    - 全可微（无 .item()、无 argmax、无 hard top-K）
    - 内联计算共振分（绕过 field.score() 的 detach）
    - 返回 fused_logits + 负载均衡 loss
    """
```

核心计算（全可微）：

```python
# 1. 所有神经元前向
all_vecs = torch.stack([neuron.forward(emb, return_logits=True)["field_vector"] 
                        for neuron in neurons])  # [N, B, D]
all_logits = torch.stack([...])  # [N, B, L, V]

# 2. 场状态 = 所有 field_vector 归一化后求和
all_vecs_norm = F.normalize(all_vecs, dim=-1)  # [N, B, D]
field_state = all_vecs_norm.sum(dim=0)  # [B, D]

# 3. Leave-one-out 共振分（可微）
loo_state = field_state.unsqueeze(0) - all_vecs_norm  # [N, B, D]
loo_norm = F.normalize(loo_state, dim=-1)
scores = (all_vecs_norm * loo_norm).sum(dim=-1).mean(dim=1)  # [N]

# 4. 软加权聚合（可微，无 .item()）
weights = F.softmax(scores / temperature, dim=0)  # [N]
fused_logits = torch.einsum('n,nblv->blv', weights, all_logits)  # [B, L, V]

# 5. 负载均衡 loss（防止一个神经元垄断）
balance_loss = -(weights * torch.log(weights + 1e-8)).sum()  # 负熵
```

### Step 2: 创建联合训练脚本 `train_cortex_joint.py`

**文件**: `scripts/training/train_cortex_joint.py`（新建）

```python
def train_cortex_joint(
    n_neurons=5,      # 先 5 个验证，架构支持 10+
    domain="zh",
    num_steps=3000,
    batch_size=4,
    lr=5e-4,
    balance_lambda=0.1,   # 负载均衡系数
    temperature=1.0,      # 软路由温度（可后期退火）
):
    # 1. 创建 N 个神经元（同域，随机初始化）
    # 2. 创建/加载 shared_embedding（可训练）
    # 3. 加载 domain_sp + general_sp
    # 4. 创建 ResonanceEnsemble(neurons, field)
    # 5. 优化器 = AdamW(all_neurons_params + shared_embedding_params)
    
    for step in range(num_steps):
        # 数据：batch_align_and_embed → shared_emb, targets, mask
        # 前向：ensemble.forward_train(shared_emb) → fused_logits, balance_loss
        # 损失：CE(fused_logits, targets) + λ * balance_loss
        # 反向：loss.backward() → 更新所有神经元 + shared_embedding
        # 追踪 best（滑动 avg loss）
    
    # 保存：每个 neuron 单独存 neuron_zh_joint_{i}.pt + shared_embedding.pt
```

**复用现有组件**：

* `batch_align_and_embed()` (translator.py:523) — 数据对齐管线，直接复用

* `load_domain_texts()` (train\_neuron.py:107) — 数据加载

* `load_domain_tokenizer()` / `load_general_tokenizer()` (train\_neuron.py) — tokenizer 加载

* `get_domain_neuron_config("zh")` (config.py:161) — 神经元配置

* `ResonanceNeuron(cfg)` — 神经元创建

* `ResonanceEnsemble(neurons, field)` — ensemble 创建

* 滑动窗口 best 模型保存逻辑 (train\_neuron.py:413-422) — 复用

### Step 3: 创建评估脚本 `eval_joint.py`

**文件**: `scripts/training/eval_joint.py`（新建）

对比个体 vs 协作：

```python
# A. 个体 PPL：每个 neuron 单独 forward → CE loss
# B. 协作 PPL：ensemble.forward_train → fused_logits → CE loss
# C. 生成对比：individual vs collaborative 生成质量
# 
# 关键判据：协作 PPL < min(个体 PPL) → 涌现确认
```

## 关键设计决策

| 决策                | 选择                  | 理由                            |
| ----------------- | ------------------- | ----------------------------- |
| 聚合方式              | 软加权（softmax scores） | 可微，让系统学习谁该贡献；argmax 不可微       |
| 场的角色              | 内联计算（绕过 field 对象）   | field 对象有 detach 和状态管理，训练只需数学 |
| 数据路由              | 所有神经元看所有数据          | 专精化自然涌现，最符合"整体训练"理念           |
| 神经元数              | 先 5 个验证             | 5 个 \~30min，确认有效后扩到 10+       |
| shared\_embedding | 可训练                 | 它是"感官层"，需适应所有神经元需求            |
| 负载均衡              | 负熵 loss (λ=0.1)     | 防止一个神经元垄断，但允许专精化              |
| 训练轮次              | 单轮                  | 多轮共振是推理用，训练只需前向→聚合→loss       |

## 验证方案

1. **训练收敛**：loss 持续下降，PPL < 30（连贯基线）
2. **个体 vs 协作 PPL**：协作 PPL < min(个体 PPL) → 涌现确认
3. **生成质量**：协作输出比任何个体更连贯
4. **负载分布**：weights 不过度集中（无 collapse）

```bash
# 训练
python -u scripts/training/train_cortex_joint.py --n_neurons 5 --steps 3000

# 评估
python -u scripts/training/eval_joint.py
```

## 后续扩展（确认有效后）

* 扩到 10 神经元 + 5000 步

* 温度退火（高→低，先学后专精）

* 混合域神经元（zh + en + code 联合训练）

* 多轮共振训练（在 forward\_train 中加 round 2）

