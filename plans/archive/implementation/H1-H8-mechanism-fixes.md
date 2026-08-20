# H1-H10 隐藏问题 — 机制解析与修复

> **⚠️ 已归档（2026-07-28）**
> H1-H8 全部修复完成（H9 暂缓，H10 已删除），不再活跃维护。
> 保留作为机制修复的历史记录，遇到类似 bug 时可参考诊断方法论。

> 与 [`COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md`](../architecture/COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md) 配套阅读。
> 每个 H 对应共振场架构中一个"不报错但破坏 1+1>2 效果"的机制缺陷。

---

## 总览

| ID | 问题 | 影响面 | 状态 | 文件 |
|----|------|--------|------|------|
| H1 | field_read reshape 广播维度错位 | 神经元无法正确读取场信号 | ✅ 已修复 | neuron.py |
| H2 | 场状态不支持批量样本 | 批量训练/验证时跨样本信息泄漏 | ✅ 已修复 | field.py |
| H3 | v2 attention pool 写入开销 vs v1 兼容 | 老 checkpoint 无法加载新代码 | ✅ 已修复 | neuron.py |
| H4 | 场评分 batch 维度处理不一致 | score() 广播可能错位 | ✅ 已修复 | field.py |
| H5 | LOO 评分缺失，final_scores 未进入路由 | 神经元给自己打分虚高，共振评分不参与权重 | ✅ 已修复 | ensemble.py |
| H6 | 互补性用几何正交而非预测互补 | 路由奖励"方向不同"的神经元，而非"纠正错误"的 | ✅ 已修复 | field.py, ensemble.py |
| H7 | 置信度温度过软 (2.0) | 不同神经元 per-position 权重差别太小 | ✅ 已修复 | ensemble.py |
| H8 | W_cond 乘法门控未被调用 | 场评分缺少非线性门控能力 | ✅ 已修复 | field.py, neuron.py |
| H9 | field_dim 跨规格不匹配 | STANDARD(3072) vs EXPERT(4096) 无法直接共振 | ⏸ 暂缓 | — |
| H10 | SharedEmbedProj 未持久化 | 蒸馏和验证看到不同的投影 | ✅ 已删除 | P7-9：SharedEmbedProj 模块已从项目移除 |

---

## H1: field_read reshape 广播维度错位

**代码位置**: [neuron.py:130-136](/E:/taiji-neuron/taiji/resonance/neuron.py)

### 机制
ResonanceNeuron 的每一层 transformer block 之后，会从共振场读取 conditioning 信号注入 hidden states：

```
conditioning = self.field_read_layers[i](field_state)  # field_dim -> hidden_size
```

`field_state` 有两种出现形式：
- 批量时为 `[B, D]`（B = batch_size, D = field_dim）
- 单样本时为 `[D]`

经过 `Linear(D, H)` 后：
- 输入 `[B, D]` → 输出 `[B, H]`（2D）
- 输入 `[D]` → 输出 `[H]`（1D）

### 旧代码的错误
旧代码对 conditioning 不分情况一律 `.unsqueeze(0).unsqueeze(0)`：

```python
conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [H] -> [1,1,H]
```

当 conditioning 已经是 `[B, H]`（批量 case）时，这行代码产生的是 `[1, B, H]` 而非 `[B, 1, H]`。广播到 `h:[B, L, H]` 时静默形成 BxB 外积——每个样本看到的是混合了所有样本信息的紊乱信号，而非自己的独立场信号。

### 修复
```python
if conditioning.dim() == 1:
    conditioning = conditioning.unsqueeze(0).unsqueeze(0)  # [1,1,H] -> 广播到所有 B,L
else:
    conditioning = conditioning.unsqueeze(1)  # [B,1,H] -> 沿 seq 维广播
```

### 为什么对 1+1>2 至关重要
round 2+ 的共振循环中，每个神经元从前一轮其他神经元写入的场中读取信息。如果 reshape 错误，场 conditioning 的内容就彻底错了——等于共振机制完全失效。修复后，每个神经元真正"看到"了其他神经元的场信号，这是共振生效的最底层前提。

### 冒烟测试
- `round2 [B,D] logits [B,L,vocab]` 通过
- `round2 [D] logits [B,L,vocab]` 通过

---

## H2: 场状态不支持批量样本

**代码位置**: [field.py:60-73](/E:/taiji-neuron/taiji/resonance/field.py)

### 机制
`ResonanceField` 维护一个 D 维状态向量 `self.state`。在批量训练中，多个样本可能属于不同领域，它们应该产生不同的场写入和读取，而不应该叠加到同一个 `[D]` 向量中。

### 旧代码的错误
```python
def reset(self, batch_size=1):
    self.state = torch.zeros(self.dim)  # 永远是 [D]，不管 batch_size
```

批量数据跑完一轮后，所有 B 个样本的 field_vector 都加到同一个 [D] 上——A 样本的场震荡影响 B 样本的共振评分，跨样本信息泄漏。

### 修复
```python
def reset(self, batch_size=1):
    if batch_size > 1:
        self.state = torch.zeros(batch_size, self.dim)  # [B, D]
    else:
        self.state = torch.zeros(self.dim)               # [D]
    self._batch_size = batch_size
```

同时改写了 `write()` 自动处理 batch 提升（当 B==1 且 state[B,D] 时自动 expand）、`get_normalised_state()` 按样本独立归一化。

### 为什么对 1+1>2 至关重要
批量验证中，如果中文神经元和英文神经元的场信号混淆在一起，共振评分会把它俩当成"同一个东西"。修复后，每个样本有独立场空间，评分才能真实反映"这个神经元对这个具体输入有多契合"。

---

## H3: v2 attention pool 写入与 v1 兼容

**代码位置**: [neuron.py:88, 155-175](/E:/taiji-neuron/taiji/resonance/neuron.py)

### 机制
v2 引入 attention-pooled field_write（`field_pool_query` 在序列上做 softmax pooling 然后写场），替换 v1 的 last-token write。但老的蒸馏 checkpoint 没有 `field_pool_query` 和 `field_read_gate` 参数。

### 修复
引入 `self.v1_compat: bool = False` 标志。加载老 checkpoint 时设 `v1_compat=True`，forward 走 last-token write + broadcast read（与 checkpoints 训练时一致的路径）。新蒸馏的神经元默认 `v1_compat=False`，使用 v2 路径。

### 冒烟测试
实际验证：zh + en checkpoint 以 `v1_compat=True` 成功加载并走通完整正向传导。

---

## H4: 场评分 batch 维度处理

**代码位置**: [field.py:124-125](/E:/taiji-neuron/taiji/resonance/field.py)

### 机制
`score()` 中 `_condition()` 返回的向量维度可能是 `[D]` 或 `[B,D]`，与输入的 vector `[B,D]` 做点积时维度必须对齐。

### 旧代码的错误
```python
cond = cond.unsqueeze(0)  # 当 cond 已经是 [B,D] 时变成 [1,B,D]
```
这导致 BxB 外积——类似于 H1 的 reshape 错误，但发生在场评分层面。

### 修复
```python
if cond.dim() == 1:
    cond = cond.unsqueeze(0)  # [D] -> [1,D]，广播 B
```
当 cond 已经是 2D `[B,D]` 时不额外 unsqueeze。

---

## H5: LOO 评分缺失 + final_scores 未进入路由

**代码位置**: [ensemble.py:155, 219, 308-310](/E:/taiji-neuron/taiji/resonance/ensemble.py)

### 机制
共振循环中，每个神经元写入场后，ensemble 调用 `self.field.score(vector, neuron_id=nid)` 来计算该神经元与场的契合度。如果 `neuron_id` 为 None，score 就会把该神经元自己的写入也包含在场状态中计算距离——自己跟自己比，永远偏高。

此外，v2 路由中的 per-position 加权原来只依赖熵倒数（置信度），共振评分 `final_scores` 计算了但从未使用。

### 修复
1. 所有 `score()` 调用传入 `neuron_id=nid` 触发 LOO（leave-one-out）
2. v2 路由加入 `position_weights *= (1.0 + score_vals)` 乘法提升

```python
score_vals = torch.tensor([float(final_scores.get(nid, 0.0)) for nid in neuron_ids])
position_weights = position_weights * (1.0 + score_vals).unsqueeze(-1).unsqueeze(-1)
```

`1.0 + score` 把 [0, 1] 范围的得分映射到 [1, 2] 提升因子——场认可的神经元获得更大路由权重，但最低也有 1 倍的基准。

### 冒烟测试
- `LOO != full score` 通过（传 neuron_id 与不传的结果确实不同）
- `final_scores present for both neurons` 通过

---

## H6: 互补性从几何正交改为预测互补

**代码位置**: [field.py:151-183](/E:/taiji-neuron/taiji/resonance/field.py), [ensemble.py:320-332](/E:/taiji-neuron/taiji/resonance/ensemble.py)

### 机制
旧的 `complementarity_score` 用向量正交度来衡量两个神经元的"不同"：

```python
alignment = dot(v_norm, f_norm)
orthogonal = v_norm - alignment * f_norm
return orthogonal.norm()  # 正交度越大 → "越互补"
```

但正交 ≠ 互补。两个 neurons 的 field_vector 方向差异大，不代表 B 能纠正 A 的错误——可能只是语义上不同，对预测没有实际帮助。

### 修复
`prediction_complementarity` 直接衡量预测层面的互补：

```python
logp_a = log_softmax(neuron_a_logits, ...)
nll_a = -logp_a.gather(targets)  # A 在每个位置的负对数似然
nll_b = -logp_b.gather(targets)
reduction = (nll_a - nll_b).clamp(min=0.0).mean()
```

这是 B 在 A 犯错误的 token 上降低的 log-loss 量——真正的"B 补充了 A 的盲区"。无 targets 时退化为"B 比 A 更自信的位置占比"。

路由中使用方式：
```python
for other in other_logits:
    c += self.field.prediction_complementarity(other, all_logits[nid])
position_weights *= (1.0 + comp_boost)
```

### 冒烟测试
- `identical logits -> 0 reduction` 通过（当 A==B 时互补量为 0）
- `identical logits -> 0 disagreement` 通过

---

## H7: 置信度温度从 2.0 提升到 3.0

**代码位置**: [ensemble.py:303](/E:/taiji-neuron/taiji/resonance/ensemble.py)

### 机制
v2 路由用 per-token 的 logit 熵倒数作为置信度：

```python
ent_stack = torch.stack([entropy(all_logits[nid]) for nid])  # [N, B, L]
confidence = 1.0 / (ent_stack + 1e-8)                         # [N, B, L]
position_weights = F.softmax(confidence * TEMPERATURE, dim=0)
```

温度控制 softmax 的锐度。旧温度 2.0 下，两个神经元在大多数位置上的权重趋近于 50/50——无法区分"这个位置 zh 擅长，en 不擅长"。

### 修复
温度从 2.0 → 3.0，让置信度差异被放大，明显更有信心的神经元在对应位置上获得更高的路由权重。

---

## H8: W_cond 乘法门控

**代码位置**: [field.py:106-111, 118](/E:/taiji-neuron/taiji/resonance/field.py)

### 机制
`ResonanceField` 有一个可学习的门控参数 `W_cond`：

```python
def _condition(self, state):
    cond = torch.sigmoid(state @ self.W_cond)  # [D] 或 [B,D]
    return state * cond                         # 逐维乘法门控
```

`W_cond` 的作用：学习哪些场维度对评分重要，削弱噪声维度的干扰。但旧的 `score()` 直接对原始 state 计算余弦相似度，从未调用 `_condition()`。

### 修复
`score()` 调用 `_condition()` 后再计算相似度：

```python
def score(self, vector, neuron_id=None):
    score_state = self._leave_one_out_state(neuron_id) if neuron_id else self.state
    cond = self._condition(score_state)
    # ... 用 cond 代替 score_state 计算余弦相似度
```

---

## H9: field_dim 跨规格不匹配（暂缓）

机制：COMPACT/STANDARD 规格的 `field_dim=3072`，EXPERT 规格的 `field_dim=4096`。当 zh（STANDARD, 3072）与 code（EXPERT, 4096）需要共振时，场维度不一致。

临时方案：`PadField` 子类在验证脚本中做 padding 或截断。长期方案：统一 field_dim 或实现场维度的自适应投影。

---

## H10: SharedEmbedProj 持久化

**代码位置**: [shared_embed.py](/E:/taiji-neuron/taiji/resonance/shared_embed.py)

### 机制
蒸馏时，teacher 的 hidden states（2048-dim）需要投影到神经元的 `base_embed_dim`（512-dim）作为 shared_embeddings 输入。旧的实现用一个全局 `nn.Linear(2048, 512)`，正交初始化后从未 save——每次蒸馏或验证脚本重新创建时得到一个不同的随机投影。

后果：神经元在蒸馏时看到的是投影 P1 下的数据；验证时用的是投影 P2（不同的随机初始化）——输入分布完全不同，神经元无法正常工作。

### 修复
`SharedEmbedProj` 独立模块，支持 `save/load`：

```python
class SharedEmbedProj(nn.Module):
    def save(self, path):    torch.save(self.proj.state_dict(), path)
    @staticmethod
    def load(path): ...
```

蒸馏脚本在 Phase 1 结束时保存到 `data/shared_proj.pt`，验证脚本加载同一份投影。

---

## 与 1+1>2 的因果链

2026-07-17 的验证跑出了 zh+en 集成在中文域 +58.8%、英文域 +41.6% 的改进。每个 H 的修复对这一结果的贡献：

| H | 贡献 | 如果没有修复... |
|---|------|----------------|
| H1 | 让神经元真的"看到"了场 | 场 conditioning 是错的，共振等于没发生 |
| H2 | 批量验证时样本不互相干扰 | 中英文样本的场信号混在一起 |
| H5 | 场评分真正反映契合度 | 每个神经元给自己打满分，路由变成随机 |
| H6 | 路由奖励真正纠正错误的神经元 | 路由奖励"方向不同"——可能与纠正错误无关 |
| H7 | 路由在 per-token 上做出明确选择 | 权重 50/50，等价于简单平均 |
| H8 | 场评分有非线性门控能力 | 噪声维度与信号维度一视同仁 |

H3/H4/H10 是工程条件修复——它们不直接改变路由行为，但没有它们就无法做正确的验证。

---

## 冒烟测试覆盖表

`scripts/training/verify_h1h8.py` — 24 项检查全部通过 (exit 0)：

| 测试组 | 覆盖的 H | 测试内容 |
|--------|---------|---------|
| field 1D write/read/LOO | H2, H5, H8 | 单样本场写入、LOO 不等于全局得分、W_cond 是 nn.Parameter |
| field batched [B,D] | H2 | 批量场状态、逐样本归一化 |
| prediction_complementarity | H6 | 有/无 targets、identical logits→0 |
| neuron round 1 | H1, H3 | v2 attention pool 写入（field_attn_weights 存在）|
| neuron round 2 [B,D] | H1, H8 | 2D reshape 分支、门控读取 |
| neuron round 2 [D] | H1 | 1D reshape 分支 |
| ensemble v2 routing | H5, H6, H7 | 2神经元 2轮共振、加权 logits、final_scores、logits 有限 |
| ensemble round 2 conditioning | H1 | [B,D] 场状态通过 round 2 |

---

## 相关文件

- 架构主文档: [COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md](../architecture/COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md)
- 冒烟测试: [scripts/training/verify_h1h8.py](/E:/taiji-neuron/scripts/training/verify_h1h8.py)
- 1+1>2 验证: [scripts/training/verify_1plus1.py](/E:/taiji-neuron/scripts/training/verify_1plus1.py)
- 归档导航: [plans/README.md](../../README.md)
