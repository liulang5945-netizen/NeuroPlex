# Taiji 非 Transformer 计算底座架构规范

> **状态**：架构决策与实现合同 · 2026-08-21
>
> **名称边界**：`Taiji` 是新的认知计算底座；`NeuroPlex` 在迁移期继续表示现有产品、群体装配和兼容运行时。代码实现使用 `neuroplex.taiji`，因为顶层模块名 `taiji` 已被 `neuroplex/__init__.py` 用作旧 checkpoint 的 pickle 兼容别名。
>
> **结论边界**：本规范把“Transformer 不适合作为群体神经元的长期底座”作为项目的可检验架构假设。Taiji 的目标是补上持续状态、局部在线学习、内生记忆、稀疏事件协作和感知—行动闭环；它不能预先证明或保证 AGI，是否成立只能由下面的反证实验决定。

## 1. 决策

从本文件生效起：

1. `Taiji` 是目标计算底座，不是 Transformer 的插件、LoRA 变体或新名字。
2. `ResonanceNeuron` 和现有 9 个成员仍保留为可运行基线；其中 5 个 dialogue 成员不会删除，也不会被文档隐去。
3. Transformer、self-attention、KV cache、每成员完整 LM head 不进入 Taiji 核心。
4. 现有场、相位、兴奋/抑制、STDP、调质、睡眠和生命周期只复用经过验证的机制意图，不直接继承当前围绕 Transformer 的实现。
5. Taiji 不走“1.5B 蒸馏”或固定 `7.58M/10M` 尺寸叙事；不以教师 logits 或旧模型权重为成立条件。参数规模是实验变量，不是架构定义。
6. 第一阶段不接管生产生成，不混写现有 `ResonanceField`，先在隔离运行时完成最小可证伪验证。

目标关系是：

```text
NeuroPlex（当前产品/群体运行时）
├── LegacyResonancePopulation：现有 9 个 Transformer 成员，作为基线
└── TaijiPopulation：目标认知主体
    ├── 感觉器官（事件编码）
    ├── 同构 TaijiCell 群体（状态、预测、局部可塑性）
    ├── 持续 TaijiField（多时间尺度共享状态）
    ├── 情景/联想/语义/程序记忆
    ├── 稀疏异步调度和可塑拓扑
    └── 运动器官（文本、工具、环境动作事件）
```

## 2. 为什么不能在现有实现上继续叠加模块

本结论来自函数体和状态读写，不来自计划标题。

| 线路 | 当前源码事实 | 对 Taiji 的约束 |
|---|---|---|
| 单成员计算 | `neuroplex/resonance/neuron.py` 明确构造多层 `TransformerBlock`；树突路径仍是 field cross-attention | 细胞更新必须由持久递归动力学定义，不得把 attention 改名为树突 |
| 序列机制 | `neuroplex/layers.py` 使用 GQA、RoPE、causal attention、KV cache 和 SwiGLU | 事件和时间必须成为一等公民，不能以 token 窗口作为全部认知状态 |
| 群体场 | `ResonanceEnsemble.forward/continuous_forward` 每次调用先 `_get_task_field().reset()` | TaijiField 必须跨事件持续并显式衰减，只能在会话重置或受控遗忘时清空 |
| 对话状态 | `DialogueState.start_round()` 恢复默认 field；真实推理使用线程 task field 并再次 reset；结束时又保存默认 field | 状态所有权必须统一，禁止“恢复了一个对象、计算用了另一个对象” |
| 生成 | `Cortex._generate_p7()` 每个 token 都重新运行群体，再由某个成员的 LM head 采样 | 输出应由运动器官读取持续群体状态产生，而不是每个细胞都复制语言头 |
| 长期记忆 | `FieldMemoryBank` 保存归一化场向量、标签和文本，正常 `generate()` 不自动写入；召回主要是 top-1 向量回注 | 情景记忆必须由运行时在显著转移时原生记录，并保存原因、动作、结果和状态变化 |
| 学习 | `SleepEngine` 的主更新仍是 AdamW、CE/NLL、LoRA、域 tokenizer 和 LM head | Taiji 唤醒期必须有无需全局反传的局部可塑性；睡眠负责巩固而非唯一学习入口 |
| STDP | 当前 STDP 记录“共振轮次”并按场向量 cosine 缩放 side-channel 权重 | 可保留三因子局部学习思想，但时间必须是真实 tick/delay，更新对象是原生突触 |
| 相位 | `PhasorDynamics`、`OscillatorNode` 已真实参与连续路径，但主要调制 Transformer 成员的参与和写场 | 相位成为每个 TaijiCell 的原生状态和事件调度条件，不再是外围门控器 |
| 调质 | `NeuromodulatorState` 主要缩放 attention 温度、FFN、学习率、写场和不应期 | 调质改为直接门控局部 eligibility、探索、能量和记忆写入 |
| 持久化 | neuron、cortex、collab、field memory、agent memory、DialogueState 分散保存 | `TaijiState` 必须能一次保存/恢复完整认知状态和下一 tick 的因果连续性 |

因此，继续给 Transformer 增加“树突、相位、睡眠、场记忆”只能增加外围机制，不能改变其核心仍是“读取一个窗口、完成一次前向、丢弃主要隐状态”。Taiji 要替换的是这个最底层的状态转移函数。

## 3. 不可妥协的底座公理

### 3.1 状态先于参数

智能运行的基本对象是随时间变化的状态，而不只是固定权重。每个细胞、共享场、拓扑、短期记忆、目标和能量都有明确所有者、生命周期与持久化语义。

### 3.2 事件先于 token

底座只接收带时间、来源、可靠度和现实性标记的事件。文本 token、图像 patch、声音帧、工具反馈和内部预测都只是不同感觉器官产生的事件；token 序列不是底座 API。

### 3.3 局部因果先于全局反传

在线更新只能使用突触两端的局部状态、该连接的 eligibility trace 和广播调质信号。离线启动阶段允许对单个细胞或短时间片使用 surrogate gradient，但 Taiji 的运行不依赖跨全群体、跨完整生命周期的 BPTT。

### 3.4 稀疏活动先于全量前向

没有“每轮所有成员都重新前向”的默认动作。事件只唤醒满足新奇度、误差、目标相关性、相位和能量条件的细胞；预算调度器限制资源，但不替代细胞作认知判断。

### 3.5 记忆是动力学的一部分

工作记忆、联想快记忆和情景记忆在唤醒期直接读写；语义/程序记忆在睡眠中巩固。记忆不能只是 prompt 前缀或外部向量库。

### 3.6 感觉、认知、行动分离

细胞不各自拥有完整词表输出头。感觉器官把环境变为事件，运动器官把群体决策变为字节、工具或身体动作；认知细胞学习可跨模态复用的状态变化。

### 3.7 同构起点、涌现分化

Taiji-0 的认知细胞使用相同结构，不预先硬编码“数学、哲学、对话”等角色。差异由输入历史、拓扑、局部学习和资源竞争形成。感觉/运动器官可以有接口分工，但不等于预设认知人格。

## 4. Taiji 的七个原生计算对象

### 4.1 `TaijiEvent`

最小事件合同：

```text
tick            单调逻辑时间
episode_id      因果片段标识
source          sensor / cell / memory / goal / motor / environment
target          可选目标；None 表示广播到场
kind            sensory / peer / prediction / reward / goal / motor / control
value           固定维度稀疏或稠密向量
salience        当前显著度
reliability     来源可信度
mode            real / imagined / replay
```

`mode` 是关键边界：想象事件可以参与内部推演和学习，但不能直接提交真实动作或冒充环境反馈。

### 4.2 `TaijiCellState`

每个细胞至少持有：

| 状态 | 含义 | 时间尺度 |
|---|---|---|
| `dendrites[K,D]` | K 个基底树突分支的局部证据 | 快 |
| `apical[D]` | 场、目标和上下文形成的自上而下预测 | 快/中 |
| `soma[D]` | 细胞当前信念/控制状态 | 中 |
| `prediction[E]` | 对下一事件或场变化的预测 | 快 |
| `error[E]` | 观察与预测的局部差 | 快 |
| `phase[2]` | 单位圆相位向量 | 持续 |
| `energy` | 可用计算/发放预算 | 中 |
| `threshold` | 自适应发放阈值 | 中 |
| `refractory` | 剩余不应期 | 快 |
| `eligibility` | 最近因果贡献的低秩迹 | 中 |
| `fast_memory` | 本细胞的键值联想槽 | 中/会话 |

这些状态不是诊断缓存，而是下一 tick 的必要输入。

### 4.3 `TaijiFieldState`

共享场不是单向量，而是同一语义空间中的多时间尺度状态：

```text
fast       瞬时同步、竞争和感觉突变
working    当前任务、实体绑定和行动准备
context    跨片段背景、目标和自我状态
inhibit    维度级分流/抑制门
```

每层有独立衰减率。正常 tick 只衰减和更新，不 reset；新会话也只根据策略清理 `fast/working`，不能无条件删除 `context`、情景记忆和已学习拓扑。

### 4.4 `TaijiSynapse`

每条稀疏有向连接拥有：

```text
pre_id / post_id / branch_id
sign                 excitatory / inhibitory
weight               慢权重
fast_weight          唤醒期快速可塑增量
delay                事件传播延迟
eligibility          因果信用痕迹
usage / stability    生长、修剪和保护依据
```

兴奋/抑制是连接和细胞输出的真实符号语义，不靠把归一化向量简单取负来模拟。

### 4.5 `TaijiEpisode`

情景记忆条目必须能够回答“发生了什么、谁做了什么、结果怎样”：

```text
episode_id / tick range
real events and imagined events（分开）
field before / field after
active cells and emitted events
prediction / action / observed outcome
reward / surprise / confidence
goal and homeostatic state
causal parent ids
```

仅保存一个 field vector 和文本 label 不足以支持因果回放。

### 4.6 `TaijiScheduler`

调度器维护延迟事件队列、每 tick 激活上限和能量预算。它读取每个细胞自己报告的优先级：

\[
p_i = w_n\,novelty_i + w_e\,\lVert error_i\rVert + w_g\,goal_i
      + w_\phi\,phase_i - w_r\,refractory_i - w_c\,cost_i
\]

预算内高优先级细胞被执行；其余状态自然衰减。这个 top-k 只是资源约束，不是一个替群体决定语义的中心路由模型。

### 4.7 `TaijiState`

一个版本化状态包统一持久化：

- 时钟、episode 和随机数生成器状态；
- 全部细胞的快状态、快记忆和可塑性状态；
- 全部场层、事件队列和延迟事件；
- 稀疏拓扑及连接统计；
- 情景记忆索引与调质/稳态变量；
- 感觉/运动器官的流状态。

要求保存后恢复的下一 tick 与未中断运行数值一致，而不只是模型参数可以加载。

## 5. 单个 TaijiCell 的状态方程

以下是实现合同，不宣称与生物神经元逐项等价。

### 5.1 分支输入

对细胞 `i` 的第 `k` 个树突分支：

\[
u_{ik}^{t} = S_{ik}x_t
 + \sum_j C_{ijk}(W_{ji}+A_{ji}^{t})y_j^{t-d_{ji}}
 + R_{ik}(q_i^t)
\]

- `x_t`：感觉/目标事件聚合；
- `y_j`：经过真实 delay 的同伴事件；
- `A^t`：快速可塑增量；
- `q_i`：从本地联想记忆召回的值。

分支是有泄漏的持续状态：

\[
d_{ik}^{t+1}=(1-\alpha_k)d_{ik}^{t}+\alpha_k\,\phi(u_{ik}^{t})
\]

### 5.2 顶树突预测与局部误差

\[
a_i^{t+1}=(1-\alpha_a)a_i^t+\alpha_a\,\phi(A_i[F_t^{working},F_t^{context},g_t])
\]

\[
\hat b_i^t=P_s s_i^t+P_a a_i^{t+1},\qquad
\epsilon_i^t=\bar d_i^{t+1}-\hat b_i^t
\]

这里的 apical 路径真正预测 basal 证据；误差不再退化成 `x - (x + h_apical) = -h_apical`。

### 5.3 胞体更新

\[
s_i^{t+1}=Norm((1-\lambda_i)s_i^t
 + G_b\bar d_i^{t+1}+G_a a_i^{t+1}
 + G_e\epsilon_i^t+G_m q_i^t)
\]

门控系数由细胞状态、调质和能量产生，并被限制在稳定范围。`Norm` 可以是 RMS/向量范数稳定器，但不是 Transformer block。

### 5.4 发放与输出

\[
r_i^t = novelty_i + \lVert\epsilon_i^t\rVert + goal_i - threshold_i - energyCost_i
\]

\[
z_i^t=\mathbb{1}[r_i^t>0\land refractory_i=0],\qquad
y_i^t=z_i^t\,O_i s_i^{t+1}
\]

训练时可用有界连续门近似 `z`；运行时使用稀疏事件。发放后消耗能量、进入不应期，并提高短期阈值；静默时能量恢复、阈值缓慢回落。

### 5.5 多时间尺度场更新

所有细胞先基于 `F_t` 计算 proposal，再一次性提交，避免 Python 迭代顺序改变因果结果：

\[
F_{t+1}^{\tau}=Clip(\lambda_{\tau}F_t^{\tau}+X_t^{\tau}+M_t^{\tau}
 + \sum_i z_i g_i E_i y_i^t)
\]

抑制通过独立的 shunting gate 作用：

\[
F_{t+1}^{effective}=F_{t+1}\odot\sigma(-I_{t+1})
\]

`fast/working/context` 使用不同 `λ`；`M_t` 是召回记忆，不与新感觉混为同一种来源。

## 6. 每个 tick 的唯一合法顺序

1. 推进逻辑时钟，投递到期的外部与延迟事件。
2. 衰减 field、树突、eligibility、阈值和快记忆；恢复细胞能量。
3. 以不可变的 `state_t` 快照为所有候选细胞计算局部输入、预测误差和激活优先级。
4. 调度器按预算和能量选择活动细胞；未选择细胞只推进衰减状态。
5. 活动细胞计算 proposal：新状态、发放事件、field 写入、记忆候选和 motor proposal。
6. 原子提交全部 proposal；同一 tick 的细胞不能读到另一个细胞刚写入的半成品状态。
7. 运动器官竞争并提交最多一个互斥动作；`imagined` proposal 永不提交到真实环境。
8. 接收即时环境结果/奖励，更新调质信号和 eligibility。
9. 执行局部快可塑性，记录完整 episode transition。
10. 达到睡眠条件时进入隔离 replay；否则开始下一 tick。

这套 two-phase tick 是 Taiji 的因果底线，也是并行化边界。

## 7. 学习规则

### 7.1 唤醒期：三因子局部学习

每条活动连接维护 eligibility：

\[
e_{ji}^{t+1}=\gamma e_{ji}^{t}+pre_j^t\otimes\epsilon_i^t
\]

广播调质 `m_t` 只表示结果好坏、惊奇度或稳态压力，不传递完整梯度：

\[
\Delta A_{ji}^{t}=clip(\eta_{fast}m_t e_{ji}^{t}-\lambda_A A_{ji}^{t})
\]

快速增量进入 `fast_weight/fast_memory`，可在一次经验后立即改变行为。与本 transition 无关的细胞和连接不得变化。

### 7.2 局部预测学习

每个细胞预测下一感觉事件、下一 field 变化或动作结果。局部目标只训练该细胞的预测器和相关入边：

\[
L_i^{pred}=\rho(x_{t+1}-\hat x_i)+\rho(F_{t+1}-\hat F_i)
\]

启动训练可以用短窗口 autograd 优化这个局部损失；验收时必须证明在线适应不调用全局 `optimizer.step()`。

### 7.3 睡眠：快到慢巩固

睡眠不再围绕 LM head/LoRA 运行，而是：

1. 按惊奇、奖励、未解决误差和遗忘风险选择 episode；
2. 重放真实事件，并显式标记反事实/想象分支；
3. 把反复有用的 `fast_weight` 合并进慢权重；
4. 以旧 episode 交错重放防止灾难性遗忘；
5. 下调无用突触，生长高 eligibility 且反复共激活的连接；
6. 用 shadow state 验证稳定性，未通过则回滚本次巩固。

## 8. 原生记忆分层

| 记忆层 | 所有者 | 写入 | 读取 | 持久化 |
|---|---|---|---|---|
| 感觉缓冲 | sensory organ | 每个输入事件 | 相邻 tick | 流状态 |
| 工作记忆 | TaijiField `fast/working` | 每个有效 field proposal | 所有活动细胞 | 是 |
| 联想快记忆 | 每个 TaijiCell | 高误差/高奖励局部键值对 | 分支 query | 是 |
| 情景记忆 | TaijiRuntime | 显著 transition 自动记录 | 目标/状态/因果检索 | 是 |
| 语义记忆 | 慢权重与稳定原型 | 睡眠巩固 | 细胞动力学 | 是 |
| 程序记忆 | motor 连接与动作结果模型 | 行动反馈/睡眠 | 行动竞争 | 是 |
| 自我/稳态 | runtime + field context | 能量、目标、能力估计 | 调度和调质 | 是 |

召回不是固定 top-1：候选先按状态相似、目标相关、时间和因果链接检索，再由当前预测误差验证。召回事件带来源标记，系统可区分“我记得”与“环境刚刚发生”。

## 9. 感觉和运动器官

### 9.1 文本不是特殊本体

最初的文本感觉器官使用 UTF-8 byte 事件，避免 256K shared embedding 和每成员完整词表头成为底座成本。它可以学习字节片段、词和概念的时间层级，但 TaijiCell 只看事件向量。

### 9.2 输出属于 motor population

文本 motor 最小输出空间是 256 个字节加少量控制事件（如 EOS、THINK、ACTION）。同一 motor 接口以后可扩展到工具和身体动作。认知细胞提交意图/预测，motor 细胞竞争后才输出；不允许每个认知细胞各自生成一份完整文本再做字符串融合。

### 9.3 内部想象

Taiji 使用同一事件格式运行短程内部 rollout，但所有事件标记为 `imagined`。世界模型预测的结果可更新计划置信度，只有真实环境回执能产生 `real outcome` 和外部 reward。这是后续主动探索和规划的必要接口，不在 Taiji-0 首个内核中假装已经实现。

## 10. 从现有代码迁移什么、重写什么

| 当前部件 | 处理 | Taiji 落点 |
|---|---|---|
| `ResonanceNeuron.layers/lm_head` | 替换 | `TaijiCell` 持续状态转移 + 独立 motor organ |
| `ResonanceField` 的贡献/抑制思想 | 重写 | 多时间尺度、跨 tick 持续的 `TaijiField` |
| `continuous_forward` 的时间思想 | 重写 | 事件队列和 two-phase tick；不重复整网 forward |
| `PhasorDynamics/OscillatorNode` | 选择性复用数学 | cell 原生 phase 与调度条件 |
| `STDPTracker` | 重写 | 真实 tick/delay eligibility + 三因子更新 |
| `NeuromodulatorState` | 重写接线 | reward/surprise/homeostasis 门控局部可塑性和探索 |
| `FieldMemoryBank/DialogueState` | 替换 | 统一 episode store、cell fast memory、持久 field |
| `SleepConsolidator` 的 replay/prune/grow 意图 | 重写 | Taiji episode replay 和快到慢巩固 |
| `LifecycleManager` | 后期适配 | 使用 Taiji 的能量、贡献、误差和 lesion 指标 |
| `Cortex/API/body/tools` | 兼容适配 | 只通过 event gateway 接入，不让它们定义底座 |

现有 9 个成员以及 5 个 dialogue 成员保持冻结基线。Taiji 首阶段不与它们共享 field；不同底座的向量语义尚未校准，直接混写会让实验无法归因。通过独立基准后，才能用显式 `LegacyEventGateway` 做输入/输出级互操作。

## 11. Taiji-0：最小可证伪内核

Taiji-0 不是语言模型，也不用于证明“已经有智能”。它只验证底座是否真的拥有旧架构缺少的因果性质。

建议初始配置：

```text
cells                 3 个同构认知细胞
event_dim             32
state_dim             64
field_dim             128
dendritic_branches    4
fast_memory_slots     16 / cell
active_budget         每 tick 最多 2 个 cell
topology              稀疏有向 E/I，固定起点后允许局部变化
runtime               CPU、确定性 two-phase tick
```

这些数值只是快速实验点，不是新一轮固定尺寸身份。Taiji-0 的参数量应远低于一个现有 micro Transformer；若状态合同不成立，扩大参数没有意义。

### 11.1 必须通过的门槛

| ID | 可证伪命题 | 通过线 |
|---|---|---|
| T0 | 核心无 Transformer | `neuroplex/taiji` 不导入 attention、TransformerBlock、KV cache 或 transformers |
| T1 | 状态有因果作用 | 相同 probe 在“保留经历状态”和“reset 状态”下产生稳定可重复的不同结果；lesion 后差异消失 |
| T2 | field 跨事件持续 | 空白 tick 后按配置衰减但不归零；只有显式 reset 才清空 |
| T3 | two-phase 确定性 | 改变 cell 容器迭代顺序不改变同一 seed 的提交结果 |
| T4 | 局部在线学习 | 一次新关联后误差下降至少 30%，且不调用全局 optimizer；无关 cell 慢权重 bitwise 不变 |
| T5 | 持续学习 | 顺序学习至少 20 个关联后，首四分位关联保留率 ≥ 70% |
| T6 | 群体增益 | 3-cell 在组合/延迟任务上优于最佳单 cell；lesion 任一关键 cell 有可测下降，而非三份重复副本 |
| T7 | 稀疏和能量 | 每 tick 活动数不超过预算，长期无“所有 cell 永久高活性” |
| T8 | 数值稳定 | 10,000 tick 无 NaN/Inf，state/field/fast-weight 均在显式上界内 |
| T9 | 完整恢复 | save/load 后下一 tick 的事件、活动 cell、field 和输出数值一致 |
| T10 | 行动闭环 | 至少一个环境任务中，动作改变后续感觉，且该结果反向改变预测/策略 |

任一门槛失败都先修底座，不用扩大训练数据或参数掩盖。

## 12. 评估顺序

1. **动力学合同**：T0/T1/T2/T3/T7/T8/T9。
2. **局部学习合同**：T4/T5。
3. **群体涌现合同**：T6；同时做 1-cell、3-cell、field lesion、memory lesion 四组消融。
4. **感知—行动合同**：T10，在可复现小环境中验证。
5. **字节流能力**：只在前四层通过后训练文本 sensor/motor；比较同数据、同参数、同算力下的 Taiji 与 micro Transformer。
6. **产品互操作**：通过 event gateway A/B，不直接替换现有 9 成员。

评价必须同时报告任务分、在线适应速度、遗忘、活动稀疏度、能耗代理、状态 lesion 效果和参数/计算量。单独报告 next-byte loss 不能证明 Taiji 方向成立。

## 13. 迁移阶段和停止条件

### Phase A：隔离内核

建立 `neuroplex/taiji`，只实现 event/state/field/cell/runtime 和合同测试。禁止 loader、Cortex、生产 checkpoint 接线。

### Phase B：局部学习与记忆

加入 fast memory、eligibility、三因子更新和 episode store，通过 T4/T5。

### Phase C：群体与环境

加入可塑 topology、sensor/motor、内部想象隔离，在小环境完成 T6/T10。

### Phase D：同预算基准

用相同数据、参数、训练 FLOPs 和推理预算比较 Taiji 与 micro Transformer。若 Taiji 只在参数更多或外部记忆更大时获胜，结论必须按真实资源重算。

### Phase E：兼容接入

仅当核心门槛通过，才让 NeuroPlex 通过 event gateway 装配 TaijiPopulation。旧 9 成员继续作为回归对照，直到 Taiji 在目标任务上稳定超过基线。

停止条件：若 T1/T4/T6 在三轮结构性修改后仍不能优于无状态 RNN/单细胞对照，应暂停“Taiji 可形成新底座”的结论，回到状态方程和学习规则，而不是继续堆外围仿生模块。

## 14. 当前唯一下一步

实现 **Phase A 的 Taiji-0 动力学合同**：

```text
neuroplex/taiji/__init__.py
neuroplex/taiji/config.py
neuroplex/taiji/events.py
neuroplex/taiji/state.py
neuroplex/taiji/field.py
neuroplex/taiji/cell.py
neuroplex/taiji/runtime.py
tests/taiji/test_state_contract.py
```

本切片只做 T0/T1/T2/T3/T7/T9 的最小实现与测试，不训练、不接生产、不修改现有 9 个成员或 D1 工作区。通过后唯一下一步才是 T4 的一次性局部关联学习。
