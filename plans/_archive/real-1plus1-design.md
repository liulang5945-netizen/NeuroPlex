# 真正 1+1>2 的实验设计

## 当前实验的诚实评估

| 你说的"1+1>2" | 实际是什么 | 可靠性 |
|---------------|-----------|--------|
| 10/10 组合有效 | 正确 ✅ | 数据是真的 |
| 提升来自"共振协同" | ❌ 不是 | 来自领域路由 |
| 能证明共振场架构有效 | ❌ 不能 | 任何两个独立模型的 logit 平均都能做到 |

**本质**：你验证的是"两个领域专家投票比一个全科医生好"，不是"两个专家通过共振变得更聪明"。

---

## 真正的 1+1>2 需要什么

### 概念对比

| | 你的实验（领域路由） | 真正的共振协同 |
|---|---|---|
| 输入 | 中文样本 + 代码样本 | **同一个**跨领域样本 |
| 每个神经元 | 各自在自己领域 sample 上推理 | 都在**同一个** sample 上推理 |
| 协同方式 | logit 取平均（哪家强用哪家） | 读其他神经元的 field_vector → 调整自己的推理 |
| 验证指标 | PPL 降低 | **Round 2 PPL < Round 1 PPL** |

### 具体场景

```
跨领域输入: "请分析归并排序的时间复杂度，用中文解释"
         ↑                    ↑
    code neuron 擅长       zh neuron 擅长

Round 1（独立推理）:
  code neuron → logits_code（懂算法但"表达"生硬）
  zh neuron   → logits_zh（懂中文但算法部分可能出错）

Round 2（field conditioning）:
  code neuron 收到 zh 的 field_vector 后 → 调整输出，加入中文表达
  zh neuron 收到 code 的 field_vector 后 → 调整输出，修正算法细节
  
  → 组合后的输出同时具备"中文流畅"和"算法正确"
  → PPL(Round2) < PPL(Round1)
```

---

## 缺失的环节：field_read_layers 从未被训练

当前 neuron.forward 中：

```python
if field_state is not None and round_num > 1:
    conditioning = self.field_read_layers[i](field_state)  # 随机投影！
    h = h + conditioning
```

**field_read_layers 是随机初始化的 nn.Linear**，从未在蒸馏中收到过梯度。因为蒸馏训练时 `field_state=None`。

---

## 实验设计：联合训练 field_read_layers

### 目标

让 neuron 学会"听"其他 neuron 的 field_vector 并利用它改善自己的输出。

### 训练流程

```
对每个 batch:
  1. 两个 neuron 独立 forward（Round 1，field_state=None）
     记录: logits_r1_zh, logits_r1_code
  
  2. 将 field_vectors 写入场
     field_state = zh_field_vec + code_field_vec
  
  3. 两个 neuron 条件化 forward（Round 2，field_state != None）
     记录: logits_r2_zh, logits_r2_code
  
  4. 损失函数:
     L = L_lm_r2 + α * max(0, PPL_r2 - PPL_r1)
         ↑                ↑
     基本的语言建模   确保 Round 2 不比 Round 1 差
  
  5. 只训练 field_read_layers（冻结其他所有参数）
```

### 实现位置

修改 `distill_neurons.py` 中的 `distill_one_neuron`，新增 `train_field_conditioning` 模式。

### 预期结果

训练后：
- 中文 sample: zh neuron 的 Round 2 PPL ≈ Round 1 PPL（field 帮不上忙，因为 code neuron 也不懂中文）
- 纯代码 sample: code neuron 的 Round 2 PPL ≈ Round 1 PPL
- 跨领域 sample: Round 2 PPL < Round 1 PPL（zh neuron 利用 code neuron 的 field_vector，code 利用 zh 的）
