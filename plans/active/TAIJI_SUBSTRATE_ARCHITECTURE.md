# Taiji 原生计算架构与代码规范

> 状态：Native v7 已有可执行代码、方程、状态协议、双时间尺度 cortical path、真实按边内核、主动环境学习、原生分布式情景场与内生 replay，不是概念规划。
>
> 权威实现：仓库顶层 `taiji/`。
>
> 边界：`neuroplex/` 是冻结的 Transformer 基线，不是 Taiji 的宿主、成员容器或运行时。

## 1. 定义

Taiji 是一个**持续状态、分层预测、稀疏局部连接、在线局部学习**的计算架构。它以连续到来的事件推进状态，不读取完整 token 窗口，也不在旧 Transformer 外围添加“神经元”适配器。

Native v7 已经闭合以下完整算法链：

```text
raw bytes
  ↓ ByteSensor（257 个固定感受器，无 tokenizer/learned embedding）
Taiji predictive fabric（多个递归预测区域）
  ↕ fast activity + slow trace + one-tick episodic feedback
EpisodicField（固定群体、局部循环边、无 event K/V slot）
  ↓ resonance-gated action/outcome/value evidence
稀疏运动感受器组（覆盖全部皮层坐标的 48 个公共证据通道）
  ↓ 单一 corticostriatal context
ByteMotor（257 个动作单元）
  ↓
action → environment transition → reward + next sensation

观察下一真实字节时：
motor outcome error + region prediction error + episodic cue/event error
  ↓
只更新相邻、已有的局部突触
```

这使 Taiji Native v7 覆盖最小自回归序列模型的输入、状态、输出、学习、生成、checkpoint、动作改变感觉的环境闭环、跨 episode 的分布式情景检索与内生巩固。它不是 AGI 完成证明；当前代码证明的是这套非 Transformer 计算链可独立执行并通过 N0–N11/M0–M6 的反证门槛。

## 2. 为什么旧 Taiji-0 被废止

旧实现位于 `neuroplex/taiji/`，虽然没有直接导入 Transformer，但仍然是补丁式内核：

| 旧机制 | 实际问题 | Native v5 处理 |
|---|---|---|
| 固定维度 `TaijiEvent` 向量由外部提供 | 没有自己的输入表示 | 原始 byte 直接变为感受器活动 |
| 全局 priority + top-k 选择 cell | 中央调度决定群体活动 | 所有区域并行更新，由区域内抑制和阈值形成稀疏性 |
| 活动 cell 保存精确 cue/value slot | 本质是复制式查表 | 事件叠加进固定场群体与局部循环/readout 突触，事件数不改变拓扑 |
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
- 场活动、场 trace 与上一 tick 恢复的 cortical feedback；
- 自适应阈值与抑制状态；
- 局部预测误差；
- 已学习的预测、递归、运动和情景关联/readout 突触。

只有显式 `reset_dynamics()` 清除区域/场活动状态，学习到的突触不会被清除。存在未结算 `PendingAction` 或尚未看到结果感觉的 `PendingExperience` 时禁止 reset，避免丢失因果信用。

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

定义完整皮层输出维数与运动证据通道数：

```math
C=2\sum_r n_r, \qquad K=\text{motor\_fan\_in}, \qquad K\le C
```

运动感受器结构：

```math
H\in\mathbb{R}^{K\times C}
```

`H` 是固定、平衡、带极性的稀疏映射。每个皮层坐标恰好连接一个运动感受器，即每列恰有一个非零值；每行接收 `floor(C/K)` 或 `ceil(C/K)` 个坐标。它不学习，作用是让全部皮层状态进入公共动作证据空间，同时保持 `O(C)` 条感受器边。

运动器官慢参数：

```math
M \in \mathbb{R}^{A \times K}, \qquad b \in \mathbb{R}^{A}
```

设情景场单元数为 `N_m`、公共读出通道为 `K_m`。固定稀疏投影 `Q/A_o/O/Tau/E/P` 分别把皮层状态、动作、结果、时间码、episode 签名和 provenance 投进同一个群体；`rho∈R^{N_m}` 是固定 reward polarity。可塑慢参数为：

```math
W^{mem}\in\mathbb{R}^{N_m\times N_m}
```

以及从共享场上下文 `z=H_m h` 到 action、outcome、reward、familiarity、cortical state、time、episode 和 provenance 的局部 readout。`W^{mem}` 禁止自连接并使用固定 fan-in；所有 readout 共享同一 `K_m` 证据空间。固定编码器和可塑图的维度都由 `TaijiConfig` 决定，事件到来时不会新增 tensor、row 或 slot。

每个 postsynaptic unit 只有固定 fan-in `F`。实现不保存 `out × in` 矩阵或二值 mask，而保存压缩行：

```math
P\in\mathbb{N}^{out\times F},\qquad W_e\in\mathbb{R}^{out\times F}
```

`P[i,l]` 是 postsynaptic 单元 `i` 的第 `l` 条真实边所连接的 presynaptic 坐标，`W_e[i,l]` 是该边唯一权重。前向只计算：

```math
y_i=\sum_{l=1}^{F}W_{e,i,l}x_{P[i,l]}
```

反投影按边 scatter-add：

```math
g_j=\sum_{i,l:P[i,l]=j}W_{e,i,l}\delta_i
```

所有 `A` 个动作单元读取相同的 `K` 个公共证据通道，因而不同动作的 evidence 可直接比较；不存在按动作随机丢弃不同皮层坐标的非对称读出，也不存在运行时构造的注意力矩阵。

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

场快状态为 `MemoryState(activity, trace, cortical_feedback, threshold, inhibition, last_confidence)`；`cortical_feedback∈R^C` 只在下一 tick 进入 fabric。

完整 `TaijiState` 还保存 `tick/episode_id`、全部区域状态、场状态、motor context/probabilities、最后观察符号、可选 `PendingAction` 和可选 `PendingExperience`。pending action 原子保存所选动作、affordance、当时 context 与受限 policy；未结算前禁止再次 act 或 observe。pending experience 保存 tick、episode、provenance、动作时 cortical context、动作、reward 与 memory-learning gate；未观察 outcome sensation 前禁止再次 act 或 reset。

Native v7 checkpoint 另行保存每组突触的 int32 `pre_index`、edge weights、`consolidation_decoders`、waking `trace_baselines`、运动/场感受器 channel/polarity、固定事件编码器、memory write count、motor reward baseline/update count 和行为 RNG 状态。因此在动作已选择或 reward 已返回但 outcome sensation 尚未到达时保存/恢复，后续更新也必须逐 tensor 一致。场、lateral 与 consolidation 结构各用独立 RNG 子流；新增器官不得消耗主流并改变既有 topology。

## 6. 一个 tick 的精确前向算法

以下顺序与 `taiji/fabric.py`、`taiji/model.py` 一致，不允许实现自行交换。

### 6.1 完成 pending experience

若上一步已经收到 reward 并建立 `PendingExperience`，当前 `symbol` 就是该动作造成的 outcome sensation。`EpisodicField.write()` 在推进当前 fabric 之前，用冻结的动作时 cortical context 与当前 outcome 完成一次原子 cue/action/reward/outcome/time/episode/provenance 写入；然后本 tick 新状态清除 pending experience。若其 `learn_memory=False`，事务仍被消费但不改场突触。

### 6.2 用真实结果结算上一个动作预测

若存在上一个 motor context `c_{t-1}` 和预测分布 `p_{t-1}`，当前真实符号首先形成运动误差：

```math
\delta_t^m = onehot(b_t) - p_{t-1}
```

这个误差只用于运动突触，不反向穿过全部历史。

### 6.3 区域自底向上推进

令最低层真实活动 `y_t^{-1}=x_t`。对每个区域 `r=0..R-1`：

1. 用上一个局部 trace 预测当前下层活动：

```math
\bar q_t^r\leftarrow(1-\eta_b)\bar q_{t-1}^r+\eta_b q_t^r
```

`bar q` 只在 waking learning 更新；evaluation、generation probe 和 replay 冻结。令 signed residual `z=q-bar q`。快速稀疏 decoder 为 `D`，零初始化、全共享支撑的慢速 consolidation decoder 为 `C`：

```math
\hat{y}_t^{r-1}=D^r q_{t-1}^r+C^r z_{t-1}^r
```

普通清醒读取两条路径；sleep burst 暂时关闭 `C` 的前向贡献，避免本 bout 刚写入的证据改变后续写 basis。`C` 的 topology 与 waking baseline 进入 checkpoint，但用独立 RNG 子流初始化，不能移动 `D/T/motor/memory` 的既有随机拓扑。

2. 计算该突触末端可直接获得的预测误差：

```math
e_t^{r-1}=y_t^{r-1}-\hat{y}_t^{r-1}
```

3. 同一 reciprocal synapse 把误差投回本区域：

```math
g_t^r=(D^r)^T e_t^{r-1}+(C^r)^T e_t^{r-1}
```

4. 递归突触产生局部下一状态预测：

```math
\hat{a}_t^r=T^r q_{t-1}^r
```

5. 上一区域通过其 decoder 提供延迟一个 tick 的 top-down context：

```math
c_t^r = D^{r+1}q_{t-1}^{r+1}+C^{r+1}z_{t-1}^{r+1}
\quad (r<R-1), \qquad c_t^{R-1}=0
```

6. 将上一 tick 场回忆按区域切分为 fast/trace feedback `f_{a,t-1}^r,f_{q,t-1}^r`，再做膜状态积分：

```math
u_t^r=Bound(\lambda_u u_{t-1}^r+\alpha_g g_t^r+\alpha_T \hat{a}_t^r+\alpha_c c_t^r
+\alpha_m(f_{a,t-1}^r+f_{q,t-1}^r))
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

### 6.4 形成完整皮层状态与运动上下文

先把每个区域的快活动和慢 trace 拼接为完整皮层状态；两种时间尺度都必须显式存在：

```math
s_t=[a_t^0;\ldots;a_t^{R-1};q_t^0;\ldots;q_t^{R-1}]\in\mathbb{R}^{C}
```

初始化时，把每个坐标 `j` 均衡分配给唯一通道 `h(j)`，并固定极性 `\sigma_j\in\{-1,+1\}`。令 `G_k=\{j\mid h(j)=k\}`，运动感受器计算：

```math
\tilde c_{t,k}=\frac{1}{\sqrt{|G_k|}}\sum_{j\in G_k}\sigma_j s_{t,j}
```

```math
c_t=\gamma_c\frac{\tilde c_t}{\lVert\tilde c_t\rVert_2+\epsilon}\in\mathbb{R}^{K}
```

这相当于单 fan-out 的稀疏 feature hashing，但在架构中具有明确器官语义：每个皮层信号都到达一个运动感受器，所有动作读取同一组感受器。固定范数防止内部状态振幅过小，使运动证据被 257 路 softmax 和 bias 淹没。`H` 不保存历史，也不执行内容寻址。

### 6.5 情景场检索与 readback

场先用固定 cue encoder 和除法归一化形成：

```math
h_0=\phi(Norm(Qs_t)+(1-\lambda_{mem})q^{mem}_{t-1})
```

然后执行固定 `J=memory_iterations` 次局部循环补全：

```math
h_{j+1}=\phi(Norm(Qs_t)+\gamma_{rec}W^{mem}h_j
+(1-\lambda_{mem})q^{mem}_{t-1})
```

`phi` 使用每单元 threshold、population mean inhibition、ReLU/tanh 与 norm bound，不调用 `topk`。共享 readout context 恢复 action/outcome/reward/cortical/time/episode/provenance。熟悉度和循环支持共同给出 `c_mem`；所有读出效果都由它门控。恢复的 cortical state 保存到 `MemoryState.cortical_feedback`，只在下一 tick 的 6.3 步骤进入 fabric，避免同 tick 代数环。

### 6.6 动作概率

```math
p_t=softmax((Mc_t+b+\gamma_{read}c_{mem}v_a^{mem})/\tau_m)
```

默认执行 `argmax(p_t)`；探索时可从 `p_t` 采样。softmax 是运动竞争算子，不是 attention，也不访问历史序列。

### 6.7 环境 affordance 与 pending action/experience

外部环境给出当前允许动作集合 `A_t`，Taiji 只在该集合内归一化：

```math
\pi_t(a)=\frac{p_t(a)}{\sum_{j\in A_t}p_t(j)},\qquad a\in A_t
```

`act(A_t)` 从 `pi_t` 采样或取 argmax，并冻结 `(c_t,pi_t,a_t)` 为 pending eligibility。环境执行动作后返回 `(sensation,reward,terminal)`；它不返回正确动作标签。`settle_action(reward, provenance)` 结算 motor 后把动作时 cortical context、action、reward、tick/episode/provenance 转成 `PendingExperience`。下一次 `observe(sensation, learn_motor=False)` 才完成场写入并推进感觉 fabric，防止把环境结果误当作 teacher action，也防止尚未发生的结果提前进入记忆。

## 7. 局部学习算法

所有更新发生在 `torch.no_grad()` 中；Taiji 参数不是 autograd Parameter，没有 optimizer 或 `loss.backward()`。

实现共用同一个精确按边局部更新算子。对 postsynaptic error `delta`、presynaptic eligibility `z` 和真实边 `(i,l)`：

```math
S(z)=\max(1,\sqrt{\lVert z\rVert_0})
```

```math
W_{e,i,l}\leftarrow(1-\lambda_w)W_{e,i,l}
+\eta\frac{\delta_i z_{P[i,l]}}{S(z)}
```

随后 `RowBound` 独立限制每个 postsynaptic row 的 L2 范数不超过 `L_w`。不存在 mask 外权重，也不计算 dense outer product。

### 7.1 下层预测突触

```math
D_e^r\leftarrow EdgeLocal(D_e^r,P_D^r,e_t^{r-1},q_{t-1}^r,\eta_D)
```

只有 `P_D^r` 中真实存在的边更新。一个突触需要的信息只有其 presynaptic trace 和 postsynaptic prediction error。

慢速巩固突触只在 accepted replay 的 outcome 写入 tick 更新。设场自身读出的 action winner 为 `a`，一次 sleep bout 开始时每个 winner 神经元的局部资源 `R_a=1`：

```math
\Delta C^r\propto \eta_D\,g_{replay}\,R_a
(y^{r-1}-C^r z^r)(z^r)^T,
\qquad R_a\leftarrow0.9R_a
```

同一 replay 的重复写共享当前 `R_a`；资源只存在于 bout 内，醒来释放，不进入 state/checkpoint。它是 winner 神经元的短期可塑性资源，不是 event ID、engram counter 或外部配额。`C` 每行接触全部 `z` 坐标，所以不存在固定 16-contact 的支撑彩票；运行仍只计算这些真实存在的 edge，不使用 attention 或序列矩阵。

### 7.2 区域转移突触

```math
\delta_t^r=a_t^r-T^r q_{t-1}^r
```

```math
T_e^r\leftarrow EdgeLocal(T_e^r,P_T^r,\delta_t^r,q_{t-1}^r,\eta_T)
```

### 7.3 运动突触

```math
M_e\leftarrow EdgeLocal(M_e,P_M,\delta_t^m,c_{t-1},\eta_M)
```

```math
b\leftarrow clip\left(b+\eta_b\delta_t^m-mean(b+\eta_b\delta_t^m),-L_w,L_w\right)
```

### 7.4 奖励调制动作突触

motor 保存指数 reward baseline `v`。环境结算时：

```math
m_t=r_t-v_t,\qquad v_{t+1}=v_t+\eta_v m_t
```

```math
\delta_t^{reward}=m_t\left(onehot(a_t)-\pi_t\right)
```

```math
M_e\leftarrow EdgeLocal(M_e,P_M,\delta_t^{reward},c_t,\eta_M)
```

这是 action eligibility × local policy error × global reward prediction error 的三因子规则。正奖励强化已执行动作，负奖励压低它并提升同 affordance 集内的替代动作；未提供 teacher action。

### 7.5 情景场写入与读出学习

固定投影把动作、结果、reward polarity、sin/cos tick、稳定 bipolar episode 签名和 `experienced/imagined/replayed/external` provenance 与 cortical cue 叠加为 `h_event`。令：

```math
e^{mem}=h_{event}-W^{mem}h_{cue},\qquad
n=clip(\lVert e^{mem}\rVert/(\lVert h_{event}\rVert+\epsilon),0,1)
```

```math
g=clip(\alpha_n n+\alpha_r tanh(|r|),0,1)
```

循环边先执行 cue→event，再以一半学习率执行 event→event autoassociation：

```math
W^{mem}\leftarrow EdgeLocal(W^{mem},P_{mem},e^{mem},h_{cue},\eta_{mem}g)
```

动作 readout 使用 `r(onehot(a)-softmax(v_a))`，因此失败经历抑制重复动作；其余 readout 用 outcome/provenance 分类误差或 reward/cortical/time/episode 局部预测误差。familiarity 学习目标为 1，但只有循环 resonance 非零时才可产生有效 recall confidence。

全部更新执行微小衰减和逐 postsynaptic row 范数约束。Native v5 没有梯度跨区域传播，也没有 BPTT；固定感觉、运动和情景编码映射不更新。

## 8. 训练、评估和生成

### 8.1 在线训练

`Taiji.learn_bytes(data, epochs)` 每轮显式清空动态状态，输入 boundary、原始 bytes 和结束 boundary。每到一个真实 byte，先结算上一步 motor error，再推进当前 fabric；所有局部突触即时更新。

不存在 batch token matrix、teacher Transformer、蒸馏目标或 1.5B/7.58M/10M 身份。

### 8.2 无副作用评估

`score_bytes()` 保存完整 checkpoint，以 `learn=False` 运行流，再恢复 checkpoint。它报告 teacher-forced next-byte accuracy 和平均 surprise，不改变参数、状态或 RNG。

### 8.3 自由生成

`generate(prompt, length)` 感知 boundary 和 prompt，选择 motor action，再把自己产生的 byte 重新送入 ByteSensor，循环直到长度用尽或产生 boundary。因此生成和训练使用同一条感觉—认知—动作路径，不存在单独的 Transformer decode 路径。

### 8.4 主动环境交互

标准顺序为：`observe(cue, learn_motor=False) → act(affordances) → environment.step(action) → settle_action(reward, provenance) → observe(outcome.sensation, learn_motor=False)`。fabric 仍可在线学习感觉预测；motor 只由 reward-modulated pending eligibility 更新；field 在最后一步才获得完整真实事件。`EnvironmentOutcome` 与 `TaijiEnvironment` protocol 位于 `taiji/environment.py`。

## 9. 复杂度

设区域 decoder/transition 的有效边总数为 `E_f`，运动路径边数为 `E_a`，情景固定编码/循环/readout 总边数为 `E_m`，补全迭代为常数 `J`。

| 架构 | 单步主要计算 | 运行状态随历史长度增长 |
|---|---:|---:|
| causal Transformer | 长度为 `L` 时 attention 为 `O(Ld)`；完整序列训练为 `O(L²d)` | KV cache `O(Ld)` |
| Taiji Native v5 | `O(E_f+E_a+J E_m)` sparse/local edge operations | `O(sum n_r + N_m + C + K)`，与经历长度无关 |

运行 `L` 个事件的总计算仍与 `L` 线性；单 tick 不随已经历长度增长。代价是历史被叠加压缩进有限状态/突触，不能像 attention 一样无损回看任意旧位置；容量饱和时会发生干扰，必须由巩固、遗忘与结构生长解决。

Native v5 的 forward/local update 只访问 `[out,F]` edge weights 和 int32 pre-indices；backproject 只对这些边 scatter-add。小张量下 gather/scatter 未必比 BLAS dense matmul 更快，索引也占内存。加入情景场后，小基准 learned edge 权重+索引为 dense 权重字节的 `111.22%`，默认配置投影为 `98.59%`；这是因为当前小场/readout 密度仍高。报告必须同时给出 edge density、权重、索引字节和实测耗时，不能把按边语义冒充普遍加速。

## 10. 代码结构

```text
taiji/
├── config.py    所有形状、动力学、学习率和稳定上界
├── sparse.py    压缩固定 fan-in、gather/scatter、按边局部 delta
├── state.py     Region/MemoryState、两个 pending 事务、公开结果、TaijiState
├── environment.py  action-dependent sensation/reward 协议
├── organs.py    ByteSensor、SparseReceptorBank、ByteMotor
├── memory.py    分布式事件编码、pattern completion、局部写入与 readback
├── fabric.py    第 6 节的分层 tick、场 feedback 与第 7 节区域更新
├── model.py     observe/learn/score/generate/checkpoint
└── __init__.py  原生公共 API

tests/taiji_native/
├── test_architecture_contract.py
├── test_naming_boundary_contract.py
├── test_sequence_learning.py
├── test_context_memory.py
├── test_delayed_memory.py
├── test_long_free_run.py
├── test_sparse_kernel.py
├── test_active_environment.py
├── test_episodic_field.py
└── test_endogenous_replay.py

scripts/training/verify_taiji_native_v7.py
scripts/training/verify_taiji_n7_context.py
scripts/training/verify_taiji_n8_delayed_trace.py
scripts/training/verify_taiji_n9_long_free_run.py
scripts/training/verify_taiji_n10_sparse_migration.py
scripts/training/verify_taiji_n11_active_environment.py
scripts/training/verify_taiji_m5_episodic_field.py
scripts/training/verify_taiji_m6_endogenous_replay.py
reports/taiji_native_v5_20260821.json
reports/taiji_n7_context_20260821.json
reports/taiji_n8_delayed_trace_20260821.json
reports/taiji_n9_long_free_run_20260821.json
reports/taiji_n10_sparse_migration_20260821.json
reports/taiji_n11_active_environment_20260821.json
reports/taiji_m5_episodic_field_20260821.json
reports/taiji_m6_endogenous_replay_20260821.json
reports/taiji_m6_seed_panel_20260821.json
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
| KV cache/context window | 有界区域/场状态与分布式慢突触 |
| external vector/KV memory | 固定群体上的 cue→event completion，无 per-event slot |
| 每层全局反传 | 区域与场的 existing-edge 局部 prediction delta |
| LM head | 稀疏公共感受器组 + 单一 motor organ |
| autoregressive decoder | motor action 回灌 ByteSensor 的闭环 |
| model checkpoint | 压缩拓扑 + 参数 + 全部认知/事务状态 + RNG |

这张表表示算法职责已覆盖，不表示当前小规模 Taiji 已达到 Transformer 的语言质量。

## 12. Native v7 反证门槛

| ID | 合同 | 当前结果 |
|---|---|---|
| N0 | 顶层包不依赖 NeuroPlex/Transformer/attention/BPTT | PASS |
| N1 | raw byte 输入，无 tokenizer | PASS |
| N2 | 经历状态因果影响未来，显式 reset 才消失 | PASS |
| N3 | 学习只写已存储的真实边，全部 tensor `requires_grad=False` | PASS |
| N4 | checkpoint 后下一步输出和局部更新逐 tensor 一致 | PASS |
| N5 | 83,841 active parameters 的完整 v7 在线学习 byte cycle | PASS：accuracy `0 → 94.12%`，surprise 下降 `98.02%` |
| N6 | 自由生成真正回灌自身动作 | PASS：`a → bcdabcda`，8 步全部正确 |
| N7 | 相同当前 byte、不同历史能稳定预测不同后继 | PASS：完整状态 `100%`，一阶基线/全状态切除均 `50%` |
| N8 | 跨干扰延迟后，慢 trace 对正确动作具有独立因果贡献 | PASS：完整/trace-only `100%`，no-trace/全状态切除/一阶基线 `50%` |
| N9 | 长程自由生成不塌缩、不漂移 | PASS：无终点循环 128/128 正确、无非法动作、全部状态有界 |
| N10 | masked dense 区域改为真实 sparse/event kernel 后仍保持结果 | PASS：算子误差 ≤ `2.98e-8`，dense 算子参考一致且 N5–N9 全部回归通过 |
| N11 | 在动作会改变后续感觉的环境中在线学习 | PASS：末 40 次 `100%`，随机 `50%`，action-lesion `57.5%` |
| M5 | 跨 episode 分布式情景回忆优于同宽 trace，并通过循环/读取切除 | PASS：action `87.5%` vs trace/recurrent lesion `25%`；outcome/provenance `100%` |
| M6 | 内生 replay 后切除 episodic readout 仍保留 contingency | PASS：12/12 seed 均 4/4；control 均 25%；mean gain `+0.75` |

N7 的结论必须精确：该任务的即时上下文主要保存在 membrane/activity；单独清零 slow trace 不会破坏结果。N8 在线索与 probe 间加入共同干扰 `1234` 后，在 probe 前清零 trace 会使准确率从 `100%` 降至 `50%`；反向只保留 trace、清空 membrane/activity/threshold/inhibition 仍为 `100%`。M5 才证明跨 reset 的可检索情景场；它仍只覆盖八条微型经历，不代表大容量自传记忆。

N9 的训练流显式设置 `include_boundary=False`，因为它检验无限循环吸引子。若同时把第四轮 `d → boundary` 当作真实监督，又要求同一状态 `d → a` 无限继续，目标本身矛盾。N9 没有增加训练字节或 epoch，只移除与非终止任务冲突的结束标签。

## 13. 当前唯一下一步

M7 基准已建立并 FAIL：现有 burst 保留 action→outcome `100%`，但 cue 慢通路为 `50%`/零 margin，行为不优于 no-replay。下一步扩展 accepted replay 的真实顺序：先让内生 `cortical_projection` 经 fabric 重建 cue basis并写 action，再运行现有 action basis 写 outcome。必须加入 control、内容 lesion、顺序 lesion；仍禁止外部 event list、teacher target、dense attention 或 memory-weight 复制。
