# Taiji Native v5 原生场记忆算法

> 权威实现：`taiji/memory.py`、`taiji/state.py`、`taiji/model.py`、`taiji/fabric.py`。
>
> 状态：M0–M5 已有代码和因果反证；M6 replay/巩固尚未实现。

## 1. 记忆层级与所有权

| 层级 | 运行实体 | 时间尺度 | reset 行为 |
|---|---|---|---|
| 快活动 | `RegionState.activity` / `MemoryState.activity` | 当前 tick | 清除 |
| 膜与局部抑制 | `membrane/threshold/inhibition` | 数 tick | 清除 |
| 工作 trace | 区域与场的 `trace` | 有限延迟 | 清除 |
| 预测/转移记忆 | `fabric.decoders/transitions` | 慢突触 | 保留 |
| 动作策略 | `motor.synapses/bias` | 慢突触 | 保留 |
| 情景场 | `EpisodicField.association/readouts` | 跨 episode 慢突触 | 保留 |
| 单步动作信用 | `PendingAction` | action→reward | 结算后清除 |
| 完整事件事务 | `PendingExperience` | reward→next sensation | 写入后清除 |

场记忆不是外部文本库，也不保存 `events[]`、`keys[]`、`values[]` 或每条经历的 slot。事件数增加时，单元数和拓扑不增加；经历只叠加到固定的局部边权。

## 2. 原子事件合同

一条可写经历必须依次经过：

```text
observe(cue/state)
  → act(affordances)
  → environment.step(action)
  → settle_action(reward, provenance)
  → observe(outcome sensation)
```

`act()` 冻结动作时的运动 eligibility。`settle_action()` 完成 reward 三因子更新，并建立 `PendingExperience`：

```text
(tick, episode_id, provenance, cortical_context, action, reward)
```

下一次 `observe()` 提供真实 outcome sensation 后才允许写场。未完成时禁止再次 `act()` 或 `reset_dynamics()`，因此动作、奖励与结果不会错位。pending action 和 pending experience 都进入 checkpoint，恢复后的下一次写入必须逐 tensor 一致。

## 3. 分布式事件编码

设完整皮层状态为：

```math
s=[a^0;\ldots;a^{R-1};q^0;\ldots;q^{R-1}]\in\mathbb{R}^{C}
```

场有 `M` 个单元。固定稀疏投影分别编码皮层 cue、动作、结果、时间、episode 和来源：

```math
d_s=Norm(Qs),\quad d_a=Norm(A\,onehot(a)),\quad d_o=Norm(O\,onehot(o))
```

```math
d_t=Norm(T\tau(t)),\quad d_e=Norm(E\epsilon(id)),\quad d_p=Norm(P\,onehot(p))
```

- `tau(t)` 是多周期 sin/cos 因果时间码；
- `epsilon(id)` 是固定长度的稳定 bipolar episode 签名；
- `p ∈ {experienced, imagined, replayed, external}`；
- `rho` 是固定 reward polarity population。

所有 drive 先做场内 RMS 除法归一化，不做 global top-k。事件群体为：

```math
h^{event}=\phi\left(d_s+\frac{\gamma_e}{\sqrt{6}}
(d_a+d_o+r\rho+d_t+d_e+d_p)\right)
```

`phi` 由自适应阈值、均值抑制、ReLU/tanh 和范数上界组成。不同经历激活重叠群体；不存在一条经历对应一个神经元或一行表。

## 4. Pattern completion

只有当前皮层 cue 时：

```math
h_0=\phi(d_s+(1-\lambda_m)q^{mem}_{t-1})
```

固定迭代 `J` 次：

```math
h_{j+1}=\phi(d_s+\gamma_r W^{mem}h_j+(1-\lambda_m)q^{mem}_{t-1})
```

场 readout 共享同一个压缩上下文 `z=H_m h_J`，恢复：

```text
action evidence, outcome distribution, expected reward,
cortical state, time code, episode code, provenance distribution
```

熟悉度 readout 与循环支持共同形成置信度：

```math
c_{fam}=1-e^{-ReLU(f(z))},\qquad
c_{res}=1-e^{-\lVert W^{mem}h_J\rVert_2},\qquad
c=c_{fam}c_{res}
```

所有可执行回忆都由 `c` 门控。清零循环关联边时，直接 cue 即使碰巧激活读出，也因 `c_res=0` 不能改变行为。

动作证据在当前 tick 进入同一个 ByteMotor competition：

```math
p_t=softmax((M c_t+b+\gamma_{read}\,c\,v^{mem}_a)/\tau_m)
```

恢复的 cortical state 写入 `MemoryState.cortical_feedback`，在下一 tick 分别加到各区域的 activity/trace 坐标：

```math
u_{t+1}^r\;{+}{=}\;\gamma_{fb}
(f^{mem}_{a,r}+f^{mem}_{q,r})
```

这个一 tick 延迟避免 fabric↔memory 形成同 tick 代数环。

## 5. 局部写入规则

写入前的 cue→event 误差与 novelty 为：

```math
e^{mem}=h^{event}-W^{mem}h^{cue}
```

```math
n=clip\left(\frac{\lVert e^{mem}\rVert_2}
{\lVert h^{event}\rVert_2+\epsilon},0,1\right),\qquad
s_r=tanh(|r|)
```

写门：

```math
g=clip(\alpha_n n+\alpha_r s_r,0,1)
```

循环场在已有边上执行 cue→event 与 event→event 两次局部 delta：

```math
\Delta W^{mem}_{ij}=\eta_{mem}g\,e^{mem}_i h^{cue}_j
```

```math
\Delta W^{mem}_{ij}\;{+}{=}\;\frac{1}{2}\eta_{mem}g
(h^{event}_i-(W^{mem}h^{event})_i)h^{event}_j
```

动作 readout 使用 reward 三因子，而不是把失败动作也正向记住：

```math
\delta^{episode}_a=r(onehot(a)-softmax(v_a))
```

outcome/provenance 使用局部分类误差；reward、cortical state、time code 和 episode code 使用局部预测误差。全部调用压缩 `SparseSynapses.local_update()`，只更新 `pre_index` 中真实存在的边，无 autograd、optimizer、BPTT 或 dense outer product。

## 6. 状态、容量与 checkpoint

默认 Native v5 使用：

```text
memory_units M = 192
memory_fan_in = 32
memory_context_dim = 48
completion iterations J = 3
```

M5 反证使用 `M=128/K=32`。它的动态状态为 608 个标量：`activity 128 + trace 128 + threshold 128 + cortical feedback 224`。full-field 与 trace-only 对照由同一 checkpoint、同一 608 标量状态和同一参数容量运行；差别只有 `use_memory` 因果开关。循环 lesion 保留结构和所有读出，只把 `association.edge_weight` 置零。

Native v5 checkpoint 保存固定编码器、循环/读出边、场动态、write count、两个 pending 事务、fabric/motor 与行为 RNG。记忆结构初始化使用行为 RNG 的克隆流，因此增加记忆器官不会改变既有探索采样序列。

## 7. M5 实测

八条经历各只呈现一次，写入时环境只开放该经历实际动作，reward 为 `+1`，并关闭 fabric/motor 学习；查询时开放两个动作。这是对“看过一次后能否跨 episode 恢复 action/outcome”的隔离实验，不是自主试错发现动作的证明。

| 指标 | Full field | 同宽 trace-only | recurrent lesion |
|---|---:|---:|---:|
| action recall | **87.5%** | 25.0% | 25.0% |
| outcome recall | **100%** | 0% | 0% |
| provenance recall | **100%** | 25% | 25% |
| episode identity | **75%** | 12.5% | 12.5% |
| mean time-code cosine | **0.519** | 0 | 0 |
| mean cortical-feedback norm | **0.256** | 0 | 0 |

action 相对两个对照均提升 **62.5 个百分点**。另有独立反证确认 recall 后的同一 probe 会产生不同 region membrane，证明 cortical feedback 真实进入下一 fabric tick。8 次写入前后 association 始终为 4,096 条边，event slot 为 0。

证据：`tests/taiji_native/test_episodic_field.py`、`scripts/training/verify_taiji_m5_episodic_field.py`、`reports/taiji_m5_episodic_field_20260821.json`。

## 8. 已证明与未证明

| ID | 命题 | 状态 |
|---|---|---|
| M0 | 历史状态改变未来输出，reset 后差异消失 | PASS |
| M1 | 慢突触在线局部变化，无 optimizer | PASS |
| M2 | checkpoint 保持下一 tick/事务写入一致 | PASS |
| M3 | 相同当前输入由不同动态历史产生不同正确后继 | PASS（N7） |
| M4 | 共同干扰后 slow trace 对延迟任务必要且足够 | PASS（N8） |
| M5 | 分布式情景场优于同宽 trace-only，且 recurrent/read lesion 消除收益 | **PASS** |
| M6 | 内生 replay 巩固后，切除情景读出仍保留能力 | 未实现 |

M5 只证明一个 8-event one-shot 微型场，不证明大容量无干扰记忆、语言情景理解、自传连续性或人脑等价。当前 write_count 是诊断计数，不是事件索引；代码还没有内生 replay、睡眠相位、巩固选择或结构生长。

## 9. 当前唯一下一步

实现 M6 内生 replay/巩固：场必须从自身 novelty、reward salience、familiarity 和时间信号中选择重激活内容，把回忆产生的感觉/状态通过同一 `TaijiFabric.step(..., learn=True)` 局部规则沉入 decoder/transition；随后关闭 episodic action/readback，在跨 episode 任务上仍显著高于未 replay 对照。外部 Python event list、teacher target 和直接复制 memory weights 到 fabric 均禁止。
