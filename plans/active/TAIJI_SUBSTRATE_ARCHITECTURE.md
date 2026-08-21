# Taiji 原生计算架构与代码规范

> 状态：Native v1 已有可执行代码，不是概念规划。
>
> 权威实现：仓库顶层 `taiji/`。
>
> 边界：`neuroplex/` 是冻结的 Transformer 基线，不是 Taiji 的宿主、成员容器或运行时。

## 1. 定义

Taiji 是一个**持续状态、分层预测、稀疏局部连接、在线局部学习**的计算架构。它以连续到来的事件推进状态，不读取完整 token 窗口，也不在旧 Transformer 外围添加“神经元”适配器。

Native v1 已经闭合以下完整算法链：

```text
raw bytes
  ↓ ByteSensor（257 个固定感受器，无 tokenizer/learned embedding）
Taiji predictive fabric（多个递归预测区域）
  ↓ 单一 corticostriatal context
ByteMotor（257 个动作单元）
  ↓
next byte / free-running byte stream

观察下一真实字节时：
motor outcome error + region prediction error
  ↓
只更新相邻、已有的局部突触
```

这使 Taiji Native v1 在算法完整性上与一个最小自回归序列模型处于同一比较层级：有输入表示、时序状态转移、上下文形成、输出分布、学习规则、生成循环和 checkpoint。它不是 AGI 完成证明；当前代码只证明这套非 Transformer 计算链可以独立运行和学习。

## 2. 为什么旧 Taiji-0 被废止

旧实现位于 `neuroplex/taiji/`，虽然没有直接导入 Transformer，但仍然是补丁式内核：

| 旧机制 | 实际问题 | Native v1 处理 |
|---|---|---|
| 固定维度 `TaijiEvent` 向量由外部提供 | 没有自己的输入表示 | 原始 byte 直接变为感受器活动 |
| 全局 priority + top-k 选择 cell | 中央调度决定群体活动 | 所有区域并行更新，由区域内抑制和阈值形成稀疏性 |
| 活动 cell 保存精确 cue/value slot | 本质是复制式查表 | 时序经验沉入递归/预测突触，不在每 cell 复制 K/V 表 |
| 活动 cell 输出向量取平均 | 没有原生动作语义 | 唯一运动器官产生可执行 byte 动作 |
| `neuroplex.taiji` | Taiji 仍是旧产品内部组件 | 顶层 `taiji` 独立拥有命名空间和 checkpoint |
| 最终 event gateway 接回 Cortex | 目标仍是兼容旧 Transformer 产品 | Legacy 只作为离线同预算基线，不进入 Taiji forward |

旧原型及 T4/T5 报告已从当前源码树移除；证据可从 Git 提交 `52fcb5c`、`9671ab7`、`57e3fba` 恢复。

## 3. 输入与时间

### 3.1 原始字节感受器

默认动作/感觉字母表大小：

```text
A = 257
0..255  原始 byte
256     episode boundary
```

观察符号 `b_t` 时，`ByteSensor` 产生固定 one-hot 感觉活动：

```math
x_t = onehot(b_t) \in \mathbb{R}^{A}
```

它不是 tokenizer ID，也没有可学习 embedding。UTF-8 文本、二进制协议、工具返回值都可以作为同一 byte 流进入。图像和声音以后需要各自的感受器，但必须输出同样的“当前活动”，不能调用 Transformer 编码器替代感觉器官。

### 3.2 时间合同

一次 `observe()` 就是一个因果 tick。历史不作为 `L × d` 矩阵重新输入；它只通过下列持久状态影响未来：

- 区域膜状态；
- 当前活动；
- 多时间尺度 trace；
- 自适应阈值与抑制状态；
- 局部预测误差；
- 已学习的预测、递归和运动突触。

只有显式 `reset_dynamics()` 清除活动状态，学习到的突触不会被清除。

## 4. 参数与拓扑

设区域数为 `R`，区域 `r` 的单元数为 `n_r`，并定义 `n_-1 = A`。

每个区域只拥有两类慢参数：

```math
D^r \in \mathbb{R}^{n_{r-1} \times n_r}
```

`D^r` 是 reciprocal predictive synapses：正向从区域 `r` 预测下一层，转置方向把该层的局部预测误差送回区域 `r`。

```math
T^r \in \mathbb{R}^{n_r \times n_r}
```

`T^r` 预测区域自身的下一时刻活动。自连接默认禁止。

运动器官参数：

```math
M \in \mathbb{R}^{A \times \sum_r n_r}, \qquad b \in \mathbb{R}^{A}
```

每个矩阵都有不可学习的二值结构 mask。每个 postsynaptic unit 只有固定 fan-in；不存在任意两单元默认全连接，也不存在运行时构造的注意力矩阵。

实际存储为了 PyTorch 向量化仍使用二维 tensor，但 mask 外权重恒为零，局部更新也永远不能写入 mask 外。

## 5. 持久状态

区域 `r` 在 tick `t` 的状态为：

```text
u_t^r      membrane               n_r
a_t^r      current activity        n_r
q_t^r      temporal trace          n_r
yhat_t^r   lower-level prediction  n_{r-1}
e_t^r      lower prediction error  n_{r-1}
theta_t^r  adaptive threshold      n_r
i_t^r      inhibitory pool         scalar
```

完整 `TaijiState` 还保存 `tick/episode_id`、全部区域状态、motor context、motor probabilities 和最后观察符号。checkpoint 另行保存所有稀疏权重、结构 mask 和 RNG 状态。因此 save/load 后的下一 tick，包括在线学习产生的参数更新，必须逐 tensor 一致。

## 6. 一个 tick 的精确前向算法

以下顺序与 `taiji/fabric.py`、`taiji/model.py` 一致，不允许实现自行交换。

### 6.1 用真实结果结算上一个动作预测

若存在上一个 motor context `c_{t-1}` 和预测分布 `p_{t-1}`，当前真实符号首先形成运动误差：

```math
\delta_t^m = onehot(b_t) - p_{t-1}
```

这个误差只用于运动突触，不反向穿过全部历史。

### 6.2 区域自底向上推进

令最低层真实活动 `y_t^{-1}=x_t`。对每个区域 `r=0..R-1`：

1. 用上一个局部 trace 预测当前下层活动：

```math
\hat{y}_t^{r-1}=D^r q_{t-1}^r
```

2. 计算该突触末端可直接获得的预测误差：

```math
e_t^{r-1}=y_t^{r-1}-\hat{y}_t^{r-1}
```

3. 同一 reciprocal synapse 把误差投回本区域：

```math
g_t^r=(D^r)^T e_t^{r-1}
```

4. 递归突触产生局部下一状态预测：

```math
\hat{a}_t^r=T^r q_{t-1}^r
```

5. 上一区域通过其 decoder 提供延迟一个 tick 的 top-down context：

```math
c_t^r = D^{r+1}q_{t-1}^{r+1} \quad (r<R-1), \qquad c_t^{R-1}=0
```

6. 膜状态积分：

```math
u_t^r=Bound(\lambda_u u_{t-1}^r+\alpha_g g_t^r+\alpha_T \hat{a}_t^r+\alpha_c c_t^r)
```

7. 区域内抑制池由正驱动均值更新，不使用全局 top-k：

```math
v_t^r=ReLU(u_t^r-\theta_{t-1}^r)
```

```math
i_t^r=\lambda_i i_{t-1}^r+(1-\lambda_i)\gamma_i mean(v_t^r)
```

8. 当前活动：

```math
a_t^r=tanh(ReLU(u_t^r-\theta_{t-1}^r-i_t^r))
```

9. 每个单元只根据自己的活动率调整阈值：

```math
\theta_t^r=clip(\theta_{t-1}^r+\eta_h(I[a_t^r>0]-\rho_*))
```

10. 形成跨时间 eligibility/context trace：

```math
q_t^r=Bound(\lambda_q q_{t-1}^r+(1-\lambda_q)a_t^r)
```

然后令 `y_t^r=a_t^r`，继续推进上一区域。

### 6.3 形成唯一运动上下文

运动器官读取所有区域的 trace，但只有一个输出器官，不给每个区域复制完整输出头：

```math
q_t=[q_t^0;q_t^1;...;q_t^{R-1}]
```

```math
c_t=\gamma_c\frac{q_t}{\lVert q_t\rVert_2+\epsilon}
```

固定范数是必要的接口合同：它防止内部 trace 振幅过小，使运动证据永远被 257 路 softmax 和 bias 淹没。

### 6.4 动作概率

```math
p_t=softmax((Mc_t+b)/\tau_m)
```

默认执行 `argmax(p_t)`；探索时可从 `p_t` 采样。softmax 是运动竞争算子，不是 attention，也不访问历史序列。

## 7. 局部学习算法

所有更新发生在 `torch.no_grad()` 中；Taiji 参数不是 autograd Parameter，没有 optimizer 或 `loss.backward()`。

### 7.1 下层预测突触

```math
\Delta D^r = \eta_D\,e_t^{r-1}(q_{t-1}^r)^T
```

只有 `D^r` 的结构 mask 内连接更新。一个突触需要的信息只有其 presynaptic trace 和 postsynaptic prediction error。

### 7.2 区域转移突触

```math
\delta_t^r=a_t^r-T^r q_{t-1}^r
```

```math
\Delta T^r=\eta_T\,\delta_t^r(q_{t-1}^r)^T
```

### 7.3 运动突触

```math
\Delta M=\eta_M\,\delta_t^m c_{t-1}^T, \qquad \Delta b=\eta_b\,\delta_t^m
```

三类更新都执行 mask、微小衰减和逐 postsynaptic row 范数约束。Native v1 没有梯度跨区域传播，也没有 BPTT。

## 8. 训练、评估和生成

### 8.1 在线训练

`Taiji.learn_bytes(data, epochs)` 每轮显式清空动态状态，输入 boundary、原始 bytes 和结束 boundary。每到一个真实 byte，先结算上一步 motor error，再推进当前 fabric；所有局部突触即时更新。

不存在 batch token matrix、teacher Transformer、蒸馏目标或 1.5B/7.58M/10M 身份。

### 8.2 无副作用评估

`score_bytes()` 保存完整 checkpoint，以 `learn=False` 运行流，再恢复 checkpoint。它报告 teacher-forced next-byte accuracy 和平均 surprise，不改变参数、状态或 RNG。

### 8.3 自由生成

`generate(prompt, length)` 感知 boundary 和 prompt，选择 motor action，再把自己产生的 byte 重新送入 ByteSensor，循环直到长度用尽或产生 boundary。因此生成和训练使用同一条感觉—认知—动作路径，不存在单独的 Transformer decode 路径。

## 9. 复杂度

设所有有效 decoder、transition 和 motor 边总数为 `E`。

| 架构 | 单步主要计算 | 运行状态随历史长度增长 |
|---|---:|---:|
| causal Transformer | 长度为 `L` 时 attention 为 `O(Ld)`；完整序列训练为 `O(L²d)` | KV cache `O(Ld)` |
| Taiji Native v1 | `O(E)` sparse edge operations | `O(sum n_r)`，与已经经历的长度无关 |

运行 `L` 个事件的总计算为 `O(LE)`。代价是历史被压缩进有限状态，不能像 attention 一样无损回看任意旧位置；长期记忆必须由慢突触、情景系统和受控复习解决，而不是隐藏在无限 context window 中。

当前 PyTorch 原型用 masked dense tensor 执行，理论边数是稀疏的，但物理计算尚未使用 sparse kernel。报告必须同时给出 active edge count 与 dense tensor storage，不能把 mask 后的参数节省冒充实际 FLOPs 节省。

## 10. 代码结构

```text
taiji/
├── config.py    所有形状、动力学、学习率和稳定上界
├── sparse.py    固定 fan-in、mask、前向/反投影、局部 delta
├── state.py     RegionState、TaijiState、TaijiStep
├── organs.py    ByteSensor、ByteMotor
├── fabric.py    第 6 节的分层 tick 与第 7 节的区域更新
├── model.py     observe/learn/score/generate/checkpoint
└── __init__.py  原生公共 API

tests/taiji_native/
├── test_architecture_contract.py
└── test_sequence_learning.py

scripts/training/verify_taiji_native_v1.py
reports/taiji_native_v1_20260821.json
```

顶层 `taiji` 不导入 `neuroplex`、`transformers` 或旧序列层。PyTorch 只承担 tensor 运算。

## 11. 与 Transformer 的逐功能替代

| Transformer 功能 | Taiji 原生算子 |
|---|---|
| tokenizer + embedding | raw-byte receptor population |
| positional encoding | 真实 tick + 持久递归状态 |
| self-attention | reciprocal prediction error + sparse recurrent edges |
| FFN block | 区域膜积分、阈值、抑制和非线性活动 |
| residual stream | membrane 与 multi-timescale trace |
| KV cache/context window | 有界持久状态与慢突触 |
| 每层全局反传 | 区域局部 prediction delta |
| LM head | 单一 motor organ |
| autoregressive decoder | motor action 回灌 ByteSensor 的闭环 |
| model checkpoint | 参数 + masks + 全部认知状态 + RNG |

这张表表示算法职责已覆盖，不表示当前小规模 Taiji 已达到 Transformer 的语言质量。

## 12. Native v1 已通过的反证门槛

| ID | 合同 | 当前结果 |
|---|---|---|
| N0 | 顶层包不依赖 NeuroPlex/Transformer/attention/BPTT | PASS |
| N1 | raw byte 输入，无 tokenizer | PASS |
| N2 | 经历状态因果影响未来，显式 reset 才消失 | PASS |
| N3 | 学习只写结构 mask 内局部突触，全部 tensor `requires_grad=False` | PASS |
| N4 | checkpoint 后下一步输出和局部更新逐 tensor 一致 | PASS |
| N5 | 19,521 active parameters 在线学习 byte cycle | PASS：accuracy `0 → 76.47%`，surprise 下降 `81.15%` |
| N6 | 自由生成真正回灌自身动作 | PASS（当前只验证前四步 `a → bcda`） |
| N7 | 相同当前 byte、不同历史能稳定预测不同后继 | 未验收 |
| N8 | 长程自由生成不塌缩、不漂移 | 未验收；当前第 5 步后会出现错误 |
| N9 | masked dense 改为真实 sparse/event kernel 后仍保持结果 | 未实现 |
| N10 | 在动作会改变后续感觉的环境中在线学习 | 未实现 |

## 13. 当前唯一下一步

实现并执行 **N7 二阶上下文反证任务**：构造当前 byte 相同但前一历史不同、目标后继相反的确定性流，要求 Taiji 明显超过只看当前 byte 的一阶转移基线，同时做 `trace lesion`。这是判断当前递归预测状态是否真的承担“上下文计算”的最小实验；若失败，先修区域状态方程，不扩大数据、区域或训练轮数。
