# Taiji 算子设计：替代 TransformerBlock 的群体神经元原生底层（AGI 路径）

> 状态：✅ 纯 Python 原型已实现（[neuroplex/taiji.py](file:///workspace/neuroplex/taiji.py)）+ 算子级回归 **38/38 PASS**（[verify_taiji_operator.py](file:///workspace/scripts/training/verify_taiji_operator.py)）
> 日期：2026-08-20（初版）/ 2026-08-22（补齐三项生物机制）
> 前置：[BIO_INSPIRED_ARCHITECTURE_PLAN.md](file:///workspace/plans/active/BIO_INSPIRED_ARCHITECTURE_PLAN.md)、[DESIGN_PRINCIPLES.md](file:///workspace/plans/active/DESIGN_PRINCIPLES.md)（三大核心原则 + 人脑对应）、[AGI_FIELD_MEMORY_PLAN.md](file:///workspace/plans/active/AGI_FIELD_MEMORY_PLAN.md)、[ARCHITECTURE_DIRECTION_2026_08.md](file:///workspace/plans/active/ARCHITECTURE_DIRECTION_2026_08.md)

## 0. AGI 定位

taiji 算子不是"更好的单体 transformer"——它是**为 AGI 超越 transformer 的群体神经元原生底层**。超越路径不是 scale up 单模型参数，而是按 [ARCHITECTURE_DIRECTION](file:///workspace/plans/active/ARCHITECTURE_DIRECTION_2026_08.md) 的"群体是能力单位"+ [DESIGN_PRINCIPLES](file:///workspace/plans/active/DESIGN_PRINCIPLES.md) 的三大核心原则（神经元差异性、自我进化、借鉴人脑）+ [AGI_FIELD_MEMORY_PLAN](file:///workspace/plans/active/AGI_FIELD_MEMORY_PLAN.md) 的"场记忆第一类状态层"，让算子本身原生承载群体共振与生物机制。

## 1. 动机：TransformerBlock 在群体架构下的四个结构性瓶颈

当前 `ResonanceNeuron` 的底层算子是 [TransformerBlock](file:///workspace/neuroplex/layers.py#L273)（LLaMA 风格：RMSNorm + RoPE + GQA + SwiGLU + Pre-Norm + 可选 dendritic apical）。在群体共振架构下，它有四个无法通过参数调优解决的结构性瓶颈：

| 瓶颈 | 现状 | 后果 |
|---|---|---|
| **1. field 注入是外挂残差** | [neuron.py:628-640](file:///workspace/neuroplex/resonance/neuron.py#L628)：block 算完后，`field_read_layers[i](field_state)` 投影成 `[hidden]` 残差加到 h 上。field 不进入 attention 的 K/V | 群体共振信号无法真正调制 token 级注意力——field 只是一个外加偏置，不改变 token 之间的相对权重 |
| **2. 每轮重算，KV cache 失效** | ensemble 多轮共振每轮重跑全部 block（field 变了，cache 失效） | 多轮共振的计算开销 = 轮数 × 单轮 block 开销，无法用 cache 摊销 |
| **3. phase 无法进入算子** | 相位绑定 `a_i = σ(β·(binding_i-b0))`（[continuous.py:86](file:///workspace/neuroplex/resonance/continuous.py#L86)）只在 ensemble 层调控激活强度，phase 不进入神经元内部 | 相位（神经元振荡器的核心状态变量）只是外部开关，不参与 token 表征计算 |
| **4. side_channels 外挂** | excite/inhibit 通道也是外挂投影，不进入 block 内部 | 神经元间侧向信号无法原生调制注意力/FFN |

taiji 算子针对瓶颈 1、3、5（field-native、phase-native、统一动力学），瓶颈 2（KV cache）留待后续 kernel 化时解决，瓶颈 4（side_channels）在算子层之外解决。

## 2. 核心思想：太极轮转（Yin-Yang Complementary Dual-Stream）

用**复值旋转**统一 token-attention 与 field-resonance，让 phase、field 成为算子内部的原生调制量。

### 2.1 双流结构

| 流 | 名称 | 功能 | 对应 TransformerBlock |
|---|---|---|---|
| **Yang（阳）** | causal token attention | 局部时序：token-token causal attention（RoPE + GQA） | basal attention 路径 |
| **Yin（阴）** | field-coupled resonance | 全局共振：field-native cross-attention（Q from x, K/V from field_state） | dendritic apical 路径（但 field-native，非外挂） |

### 2.2 太极门控融合

两路用**可学习 sigmoid 互斥门控**融合：

```
gate = sigmoid(taiji_gate(x))           # [B, L, 1]，每位置独立
fused = gate · yang_out + (1-gate) · yin_out
x = x + resid_dropout(fused)
```

gate → 0.5：两路均衡；> 0.5：偏 yang（局部时序主导）；< 0.5：偏 yin（全局共振主导）。初始化 bias=0 → sigmoid(0)=0.5 均衡起点。

### 2.3 phase 进入算子内部（taiji 核心增量）

对 yin 流的 K 和 V 施加复值旋转 `e^{iφ}`（φ = phase）：

```
half = head_dim // 2
K = [K_r | K_i]   # head_dim 拆成 (real, imag) 两半
K_rot = [K_r·cosφ - K_i·sinφ | K_r·sinφ + K_i·cosφ]
V_rot = ...（同理）
```

**为什么同时旋转 K 和 V（而非只旋转 K）**：
- 只旋转 K：`<Q, K_rot>` 依赖 phase → attention 权重随 phase 变化。但当 `kv_len=1`（单 token field，ResonanceField 的常见形状）时 softmax 恒为 1.0，K 旋转无效。
- 同时旋转 V：attention 输出 = `softmax · V_rot`，即使 softmax 不变（kv_len=1），V 旋转仍改变输出 → phase 在单 token field 下也生效。
- **K 旋转负责"选择性调制"**（多 token field 下的相位路由），**V 旋转负责"内容调制"**（任何 kv_len 下的相位编码），两者互补。
- Q 不旋转：保持 `<Q, K_rot>` 对 phase 的依赖（Q 旋转会让旋转退化为正交变换，phase 在 K 侧失效）。

### 2.4 field-native conditioning（taiji 核心增量）

field_state 直接进入 yin 流的 K/V（`wk_field(field_state)`, `wv_field(field_state)`），而非外挂残差。这让群体共振信号原生调制 token 表征——field 改变 K/V，从而改变 attention 权重和输出内容。

## 3. 数学形式

### Yang 流（causal token attention）

```
Q = Wq · h, K = Wk · h, V = Wv · h      # [B, L, heads, head_dim]
Q, K = RoPE(Q, K)                        # 旋转位置编码
yang_out = softmax(Q·K^T/√d + mask) · V  # causal
yang_out = Wo · yang_out
```

### Yin 流（field-coupled resonance）

```
Q = Wq · h                              # [B, L, heads, head_dim]（token-side）
K_field = Wk_field · field_state         # [B, kv_len, heads, head_dim]（field-side）
V_field = Wv_field · field_state

# phase 旋转（taiji 核心）
if phase ≠ 0:
    K_field = rotate(K_field, phase)     # 复值旋转 e^{iφ}
    V_field = rotate(V_field, phase)

yin_out = softmax(Q · K_field^T / √d) · V_field   # 无 causal（field 是全局反馈）
yin_out = Wo · yin_out
```

### 太极融合 + FFN

```
gate = sigmoid(taiji_gate(h))            # [B, L, 1]
fused = gate · yang_out + (1-gate) · yin_out
h = h + resid_dropout(fused)
h = h + resid_dropout(SwiGLU(ffn_norm(h), gain=ffn_gain))
```

## 4. 接口契约（与 TransformerBlock 对齐，可直接替换）

```python
class TaijiBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,                          # [B, L, hidden]
        mask: Optional[torch.Tensor] = None,       # [1,1,L,L] causal
        kv_cache: Optional[Tuple] = None,          # 接口兼容（原型返回 None）
        use_cache: bool = False,                   # 接口兼容
        temp_gain: float = 1.0,                    # S9 神经调质（yang + yin 共享）
        ffn_gain: float = 1.0,                     # S9 FFN 增益
        field_state: Optional[torch.Tensor] = None, # [B, D] 或 [B, S, D]
        return_attn_weights: bool = False,
        phase: Optional[float] = None,             # taiji 新增：相位旋转角
    ) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        # 返回 (x_out, kv_cache=None, attn_weights)
```

**退化行为**（向后兼容）：`field_state=None` 且 `phase=None` → yin 流不激活，退化为标准 causal TransformerBlock（yang + FFN），行为与 `TransformerBlock(dendritic=False)` 结构等价。

## 5. 与群体架构的关系

### 5.1 ResonanceNeuron 集成（下一步）

`ResonanceNeuron.__init__` 的 [neuron.py:110-126](file:///workspace/neuroplex/resonance/neuron.py#L110) 把 `TransformerBlock` 装进 `self.layers`。taiji 集成路径：

```python
# 可选切换（配置驱动）
if c.use_taiji:
    self.layers = nn.ModuleList([
        TaijiBlock(
            hidden_size=c.hidden_size, num_heads=c.num_attention_heads,
            num_kv_heads=c.num_key_value_heads, intermediate_size=c.intermediate_size,
            field_dim=c.field_dim, dropout=c.dropout,
        ) for _ in range(c.num_hidden_layers)
    ])
else:
    self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(c.num_hidden_layers)])
```

### 5.2 phase 从哪来

ensemble 的相位绑定（[continuous.py](file:///workspace/neuroplex/resonance/continuous.py)）计算 `a_i = σ(β·(binding_i - b0))`。taiji 集成时，把 binding 的相位分量 `arg(phasor_i)` 作为 `phase` 参数传入 `block.forward(phase=...)`，让相位绑定从"外部激活开关"升级为"内部表征调制"。

### 5.3 field_state 形状

`ResonanceField.get_normalised_state()` 返回 `[D]`（单 token field）。taiji 的 yin 流支持 `[B, D]`（kv_len=1，V 旋转生效）和 `[B, S, D]`（kv_len=S，K+V 旋转都生效）两种形状，兼容现有 field 接口。

## 6. 已验证（21/21 PASS）

[verify_taiji_operator.py](file:///workspace/scripts/training/verify_taiji_operator.py) 6 维 21 项：

| 维度 | 验证内容 | 结果 |
|---|---|---|
| T1 接口契约 | forward 签名与 TransformerBlock 对齐，返回三元组，可直接替换 | 5/5 PASS |
| T2 数值稳定性 | 输出无 NaN/Inf，数值范围合理，极端输入（×100）稳定 | 4/4 PASS |
| T3 退化等价 | field/phase=None → 仅 yang 流，与 TransformerBlock 同量级 | 3/3 PASS |
| T4 field-native | field_state 非 None 时 yin 流激活，不同 field 产生不同输出 | 2/2 PASS |
| T5 phase 调制 | phase=0/1/2 产生不同输出，phase=0 与 None 等价 | 3/3 PASS |
| T6 梯度流通 | x / field_state / yin 参数 / 太极门控 全部梯度可流通 | 4/4 PASS |

## 9. 生物机制补齐（plans 对齐，2026-08-22）

初版 taiji 闭合了 field-native + phase-native，但 [DESIGN_PRINCIPLES](file:///workspace/plans/active/DESIGN_PRINCIPLES.md) §1.2 的人脑对应表里还有三项生物机制是缺口。本轮一次性补齐，让 taiji 算子达到 plans 生物机制完整对齐。

### 9.1 E/I 原生接入（plans §2.3 兴奋/抑制双通道）

**问题**：TransformerBlock 的 side_channels 是外挂投影（[neuron.py:264-265](file:///workspace/neuroplex/resonance/neuron.py#L264)），不进 block 内部。plans §2.3 要求"兴奋/抑制双通道"是 E/I 平衡的核心机制。

**实现**：forward 新增 `excite_signal` / `inhibit_signal` 可选参数：
- `excite_signal [B, hidden]` → `excite_gate` → 调制 **yang 流**输出（兴奋→放大局部驱动）
- `inhibit_signal [B, hidden]` → `inhibit_gate` → 调制 **yin 流**输出（抑制→衰减全局共振）
- 门控：`sigmoid(gate) * 2`，0.5 中性（sigmoid(0)），>0.5 放大，<0.5 衰减
- E/I 信号原生进入算子内部调制两路，而非外挂残差

### 9.2 不应期 refractory（plans §2.2）

**问题**：TransformerBlock 是纯前馈无状态。plans §2.2 要求"发放后冷却 N 轮，强制信息分流"。

**实现**：TaijiBlock 维护 `refractory_counter` buffer + 三接口：
- `enter_refractory(steps=None)`：发放（field_write）后由神经元调用，设置冷却计数
- `tick_refractory()`：每轮结束递减
- `in_refractory` 属性：counter > 0

**算子层语义**：不应期时 **yin 流不激活**（field 共振读取被抑制），只走局部 yang 流。模拟"不应期神经元不参与群体共振"——发放后短暂退出群体协作，强制信息分流。

### 9.3 STDP 局部学习（plans §1.2 突触可塑性）

**问题**：TransformerBlock 只靠反向传播更新，无局部学习。plans §1.2 要求"STDP 局部学习规则"。

**实现**：前向 STDP（不依赖反向传播）——field 时序差驱动 yin gain 调制：
- 维护 `field_prev`（上一轮 field_state，detach，不进计算图）
- 每轮计算 `delta_norm = ||field_state - field_prev|| / ||field_state||`
- STDP 调制：`stdp_mod = 1 + stdp_strength * tanh(delta_norm)`
  - delta_norm=0（field 稳定）→ mod=1（中性，不调制）
  - delta_norm>0（field 变化）→ mod>1（增强突触，共振读取得以强化）
- yin 流输出乘 stdp_mod
- field 变化越大 → yin 流增益越强（群体共振信号越被关注）

**为何是前向 STDP**：用局部时序相关性（field 前后差）调制，不改变训练范式（仍可反向传播），但让算子有了局部适应能力——这是 STDP 在算子层的轻量表达。

## 10. plans 对齐总表

| plans 要求 | TransformerBlock | TaijiBlock（初版） | TaijiBlock（本轮补齐后） |
|---|---|---|---|
| field 第一类状态层（[AGI_FIELD_MEMORY_PLAN](file:///workspace/plans/active/AGI_FIELD_MEMORY_PLAN.md) §3） | ✖ 外挂残差 | ✅ field-native（进 K/V） | ✅ |
| 相位持久化（AGI_FIELD_MEMORY_PLAN §2） | ✖ phase 不进算子 | ✅ phase 旋转 K/V | ✅ |
| 兴奋/抑制双通道（[DESIGN_PRINCIPLES](file:///workspace/plans/active/DESIGN_PRINCIPLES.md) §2.3） | ✖ 外挂投影 | ⚠️ 隐喻对齐 | ✅ E/I 原生接入（excite/inhibit gate） |
| 不应期（DESIGN_PRINCIPLES §2.2） | ✖ 无 | ✖ 缺口 | ✅ refractory_counter + yin 流抑制 |
| STDP 局部学习（DESIGN_PRINCIPLES §1.2） | ✖ 无 | ✖ 缺口 | ✅ 前向 STDP（field 时序差调制） |
| 神经调质（DESIGN_PRINCIPLES §1.2） | ✅ temp/ffn_gain | ✅ | ✅ |
| 群体是能力单位（[ARCHITECTURE_DIRECTION](file:///workspace/plans/active/ARCHITECTURE_DIRECTION_2026_08.md)） | ⚠️ 个体内部 | ✅ field 原生进算子 | ✅ |
| 自我进化上千神经元（DESIGN_PRINCIPLES §1.1） | ⚠️ scale 单模型 | ✅ scale 群体 | ✅ |

**结论**：taiji 算子现已完整对齐 plans 的 AGI 构想——field-native + phase-native + E/I 原生 + 不应期 + STDP + 神经调质，让算子本身就是群体神经元的基本单元，而非"装在群体里的更强 transformer"。

## 11. 已验证（38/38 PASS）

[verify_taiji_operator.py](file:///workspace/scripts/training/verify_taiji_operator.py) 9 维 38 项：

| 维度 | 验证内容 | 结果 |
|---|---|---|
| T1 接口契约 | forward 签名与 TransformerBlock 对齐，返回三元组，可直接替换 | 5/5 PASS |
| T2 数值稳定性 | 输出无 NaN/Inf，数值范围合理，极端输入（×100）稳定 | 4/4 PASS |
| T3 退化等价 | field/phase=None → 仅 yang 流，与 TransformerBlock 同量级 | 3/3 PASS |
| T4 field-native | field_state 非 None 时 yin 流激活，不同 field 产生不同输出 | 2/2 PASS |
| T5 phase 调制 | phase=0/1/2 产生不同输出，phase=0 与 None 等价 | 3/3 PASS |
| T6 梯度流通 | x / field_state / yin 参数 / 太极门控 全部梯度可流通 | 4/4 PASS |
| T7 E/I 原生接入 | excite/inhibit 信号调制两路，不同信号产生不同输出，门控可训练 | 5/5 PASS |
| T8 不应期 | enter/tick/in_refractory 接口，冷却期 yin 抑制，恢复后还原 | 7/7 PASS |
| T9 STDP 局部学习 | 第一轮中性，变化大→增强，变化小→中性，前向不进计算图 | 5/5 PASS |

## 12. 搭建顺序（唯一下一步）

| 阶段 | 内容 | 状态 |
|---|---|---|
| ✅ 阶段 0 | 纯 Python 原型 + 算子级回归（含三项生物机制） | 完成 |
| ⬜ 阶段 1 | ResonanceNeuron 集成：配置开关 `use_taiji`，phase 从 ensemble binding 注入，E/I 信号从 side_channels 接入 | 待做 |
| ⬜ 阶段 2 | 集成回归：用 TaijiBlock 跑一轮 ensemble forward，确认共振轮次、field I/O、refractory/STDP 状态流转正确 | 待做 |
| ⬜ 阶段 3 | 小规模训练对比：同一 9 成员配置，TransformerBlock vs TaijiBlock，对比共振质量（共振分分布、field 收敛、PPL） | 待做（需 checkpoint 环境） |
| ⬜ 阶段 4 | kernel 化：yang 流 GQA + yin 流 cross-attention 融合 CUDA kernel，解决瓶颈 2（KV cache） | 待做 |

**唯一下一步**：阶段 1（ResonanceNeuron 集成），冻结现有 9 成员生产权重，新增 `use_taiji` 配置开关默认 False，验证 TaijiBlock 在真实 ResonanceNeuron 里能跑通一轮 ensemble forward，且 phase/E-I/refractory/STDP 能从 ensemble 层正确接入算子。

## 13. 不做什么（边界）

- **不替代 RoPE**：yang 流保留 RoPE（位置编码成熟且有效），taiji 的旋转只在 yin 流的 field K/V 上
- **不实现 KV cache**：原型阶段 focus 语义正确性，cache 留待阶段 4 kernel 化
- **不实现 attention sink / sliding window**：yang 流原型暂不实现长上下文特性，留待阶段 4
- **不动 ensemble 层**：taiji 只替换神经元内部 block，ensemble 的共振编排、STDP（群体级）、coaction 不变
- **不引入复值张量**：用实数 (real, imag) 对表达复值旋转，保持与 PyTorch 生态兼容
- **STDP 是前向调制**：本轮 STDP 是 field 时序差驱动的前向 gain 调制，不是完整的权重局部更新规则；完整 STDP 权重学习留待阶段 3 训练验证后视需要扩展
