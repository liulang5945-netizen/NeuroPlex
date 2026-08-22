# NeuroPlex 机制运行地图（源码审计版）

> 审计日期：2026-08-20
>
> 本文以源码函数体、调用者、状态读写和可复现实验为唯一依据。计划文件中的“已接入”只有在这里找到真实入口后才成立。行号按本次审计时的工作区记录，代码改动后应重新核对。
>
> **2026-08-21 架构边界**：本文现在是 Legacy NeuroPlex（现有 9 个 Transformer 成员）的事实基线，不是 Taiji 新底座规范。Taiji 的替换设计见 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md)；本审计保留用于解释底层替换原因和后续同预算/消融对照。

## 0. 审计结论先行

当前项目不是“没有架构”，而是存在多条并行线路，接入深度不同：

1. 生产文本主链已经明确：
   `assemble_cortex → Cortex.generate → _generate_p7 → Cortex.think → ResonanceEnsemble.continuous_forward → ResonanceNeuron.forward → field write/read → logits decode`。
2. 共振场、神经元间 side channel、相位绑定、调质、STDP 记录和睡眠阶段都有真实代码；但它们并不都在默认生成路径中以相同方式生效。
3. 默认文本生成会读取场记忆，但不会自动把正常交互写回场记忆；当前长期场记忆主要依赖外部调用 `SleepEngine.record_field_memory()`。
4. `Cortex.working_memory` 是注册接口，不是生产生成所读取的工作记忆；实际文本上下文链在 `ContextManager`，但 `Cortex.generate()` 也不自动调用它。
5. `PlayEngine` 的自由共振路径绕过 `Ensemble`，并读取 `ResonanceNeuron.forward()` 没有产生的 `resonance_score` / `_last_field_state`，所以这条“游戏→高共振 replay”链当前不能当作正常生产闭环。
6. 睡眠的主模型训练是单神经元直训；协作 `forward_train()` 是另一条训练线路，不能把“睡眠执行过”理解为“协作层已经被训练”。

## 1. 状态标记与审计方法

| 标记 | 含义 |
|---|---|
| ✅ | 默认生产路径中有真实调用，且输入/输出状态能在代码中闭合 |
| 🟡 | 有真实接线，但默认关闭、依赖外部调用、依赖 checkpoint 或只有部分状态闭合 |
| 🔬 | 训练、睡眠、诊断或实验脚本专用，不等于生产推理会使用 |
| ⚠️ | 代码存在调用意图，但调用契约不成立、状态丢失或路径绕过核心机制 |
| DEAD | 有实现但本次 `rg` 未发现生产调用者，仅被 archive/smoke 使用 |

本次核对至少追踪了：

- loader 装配与权重恢复；
- `Cortex.generate/think` 到 ensemble 的完整调用链；
- neuron.forward 的输入、field read/write、side signal 和输出字典；
- 离散共振、连续共振和可微训练共振的差异；
- field、DialogueState、FieldMemoryBank、ContextManager、WorkingMemory 的状态所有权；
- Feed/Sleep/Play/Lifecycle 的真实调用者；
- 训练脚本到底调用 `neuron.forward` 还是 `ensemble.forward_train`；
- 测试和验证脚本是否真正走生产链，还是手工注入状态。

## 2. 生产装配链：谁创建、谁注入、谁恢复

入口是 `neuroplex/loader.py:232-881` 的 `assemble_cortex()`。

| 阶段 | 源码实际动作 | 状态结论 |
|---|---|---|
| 神经元集合 | `loader.py:294-305` 先加载 `extra_neurons_dir`，之后由 Cortex 加载指定/扫描 neuron checkpoint | ✅ 生产集合可为 5 dialogue + 4 general；不是单一中心模型 |
| tokenizer | `loader.py:278-292` 加载 `TokenizerHub`；通用 tokenizer、各域 tokenizer 和 modal codec 分开 | ✅ 输入 general 空间，输出按域 tokenizer 解码 |
| 共享输入 | `loader.py:307-443` 加载 general SentencePiece、共享 embedding、per-neuron shared embedding | ✅ 训练/推理都可能使用 per-neuron embedding；需看装配结果 |
| 协作产物 | `loader.py:444-451`、`1057-1310` 加载 side channel、跨规格 projectors、body/quality/LoRA、sparse router、`field_w_cond` | 🟡 没有协作 checkpoint 时不会自动构建训练好的协作关系 |
| STDP | `loader.py:456-464` 创建并注入 `STDPTracker` | ✅ 运行时记录；🔬 结构更新只在睡眠调用 |
| 调质 | `loader.py:466-475` 创建 `NeuromodulatorState`，注入 Cortex 和 Ensemble | ✅ 影响写入强度、注意力温度、不应期和训练学习率 |
| 相位 | `loader.py:477-582` 优先装配 `PhasorDynamics`，恢复 phasor state；失败才回退标量 Gamma | ✅ 默认有相位模块；推理冻结参数但状态会演化 |
| BioOSS | `loader.py:543-552` 用 `make_default_oscillators()` 注入 theta/gamma 两个 o 型振荡节点 | ✅ 连续推理/连续训练路径使用；不承担内容生成 |
| brain WorkingMemory | `loader.py:584-595` 创建并调用 `cortex.set_working_memory()` | ⚠️ Cortex 保存了引用，但 `generate/think` 不读取它 |
| 生命周期/睡眠 | `loader.py:597-693` 创建并接到全局 Sleep/Life/Play/Evolution engine | 🟡 接线存在，但各 engine 内部路径仍需单独判定 |
| agent 记忆 | `loader.py:695-740` 创建 Perception/Planner/Reflector/MemorySystem，接入全局 ContextManager | 🟡 子系统已装配；不代表 Cortex 生成自动使用 |
| Cortex 状态 | `loader.py:839-867` 加载 `cortex_state.pt` | 🟡 只恢复该文件契约包含的状态，不恢复 FieldMemoryBank/DialogueState |

### 2.1 装配后的核心对象关系

```text
loader.assemble_cortex
 ├─ Cortex.neurons
 ├─ Cortex.field ─────────────── 默认 ResonanceField
 ├─ Cortex.ensemble ─────────── ResonanceEnsemble
 │    ├─ thread-local task field（每次推理实际使用）
 │    ├─ CoactivationTracker
 │    ├─ PhasorDynamics / BioOSS oscillators
 │    └─ optional router / spatial diffuser / projectors
 ├─ SleepEngine ─────────────── FieldMemoryBank 的实际所有者
 ├─ ContextManager ──────────── 文本记忆系统的实际所有者
 └─ PlayEngine ───────────────── 独立自由活动路径
```

关键所有权问题：`Cortex.field` 是默认/诊断场；生产并行推理使用 `Ensemble._get_task_field()` 创建的线程本地任务场，见 `ensemble.py:617-637`。因此记忆捕获必须读取 `Cortex.get_last_field_state()`（`cortex.py:3098-3112`），不能只读 `Cortex.get_field_state()`。

## 3. 文本生产推理的真实执行链

### 3.1 主链

```text
用户文本
  ↓
Cortex.generate()                         cortex.py:1129+
  ↓
_generate_p7()                             cortex.py:2283+
  ├─ general tokenizer 编码输入
  ├─ 可选 _executive_route：用 round-1 judge/quality 选域
  ├─ 可选 FieldMemoryBank 检索（当前 top-1）
  ├─ 可选 DialogueState.start_round()
  ├─ 生成每个 token 的 general/shared embedding
  └─ Cortex.think()
       ↓
       collab_mode="continuous"（generate 默认）
       ↓
       ResonanceEnsemble.continuous_forward()  ensemble.py:2008+
       ↓
       t=0：所有 active neuron 独立 neuron.forward(round_num=1)
       ↓
       field.write / write_inhibit + t=0 judge/score
       ↓
       memory seed 写入场（如有）
       ↓
       t=1..T：相位演化 → 软激活 → field-conditioned neuron.forward(round_num>1)
       ↓
       field 积分、GABA 时间门控、连续权重融合
       ↓
       Cortex 选 leader / 解码域 token / 整体文本重编码
  ↓
 结束时可选 DialogueState.end_round()
  ↓
  不会自动 record_field_memory()
```

### 3.2 输入对齐和生成 token 反馈

- general tokenizer 的 token id 先经过 shared embedding；per-neuron shared embedding 存在时，`_generate_p7` 按 neuron 使用对应 embedding，见 `cortex.py:2582-2597`。
- 域神经元输出域词表 token。生成后不能逐 piece 追加 general id；`cortex.py:2807-2821` 对 `prefix + generated_text` 整体重新编码，以避免 SentencePiece 边界错误。此修复已在 `5c7e2a5` 提交。
- `Translator`/`field_alignment` 提供域 token 与 general token 的近似映射，训练睡眠直训和跨域训练会使用；它不是自然语言语义等价证明。

### 3.3 `ResonanceNeuron.forward()` 的实际内部顺序

入口 `neuroplex/resonance/neuron.py:528-850`：

1. `shared_embeddings [B,L,base_embed_dim]` 经过 `embed_adapter` 到本 neuron hidden；
2. Transformer block 做标准 causal attention/FFN；若 `dendritic_enabled` 且有场状态，走 basal+apical cross-attention；
3. `round_num>1` 且有 `field_state` 时，用 `field_read_layers` + `field_read_gate` 做 additive/multiplicative/predictive 场条件化；
4. `side_signals` 经 `excite_channels` / `inhibit_channels` 投影，调制 hidden；
5. `field_write` 或多头 field pooling 生成场向量；
6. 可选 `score_proj` 生成共享评分向量；
7. `lm_head` 生成域 logits，`judge_lm_head` 生成共享判定 logits；
8. round 1 可运行 `quality_head`。

实际返回字典含 logits、field vector、score/judge/quality 等已实现输出，但**不含 `resonance_score`，也不写入 `_last_field_state`**。因此任何读取这两个字段的调用者都不能被当作已闭合的神经元共振接口。`quick_probe()`（`neuron.py:896-910`）也只在 archive smoke 中出现，属于 DEAD。

## 4. 共振场和群体编排

### 4.1 `ResonanceField`：运行时工作场

入口 `neuroplex/resonance/field.py:40-553`。

| 机制 | 实际动作 | 梯度/持久化 | 状态 |
|---|---|---|---|
| reset | `field.py:77-94` 清空 state、mask、贡献、分数、history | 运行时瞬态 | ✅ 每个 task forward 开始清空任务场 |
| excitatory write | `field.py:100-142` L2 归一后累加到 state，贡献 detached 保存 | 通过 `forward_train` 的另一套张量路径可微；运行时 write 本身 detach | ✅ |
| round 2 update | `field.py:144-193` 先减旧贡献再加新贡献 | 运行时状态；训练路径明确不用它 | ✅/🔬 |
| inhibitory write | `field.py:195-236` 生成 multiplicative GABA-like mask | detached contribution | ✅ |
| inhibitory WTA | `field.py:238-294` 保留最强抑制贡献 | 推理离散路径使用 | ✅ |
| lateral norm | `field.py:296-313` 约束场振幅 | 运行时使用；连续路径每步使用 | ✅ |
| leave-one-out score | `field.py:329-401` 去除自身贡献后评分 | 运行时 score 含 detach；训练 score 使用可微近似/部分 detached 口径 | 🟡 训练推理口径不同 |
| round snapshot | `field.py:509-553` 保存/恢复场、mask、贡献 | 只被 DialogueState 使用；不是全局持久化 | 🟡 |

### 4.2 离散 `forward()` 与连续 `continuous_forward()`

| 路径 | 真实调用 | 生效机制 | 重要差异 |
|---|---|---|---|
| 离散推理 | `ensemble.py:1208-1938` | round 1 写场；refractory；round 2+ side signal、field update、score、WTA、gamma binding、Kuramoto | 默认不是文本 `generate` 的主路径，但可由 `think(collab_mode!=continuous)` 使用 |
| 连续推理 | `ensemble.py:2008-2327` | t=0 独立前向；相位演化；soft activation；场积分；BioOSS GABA；时间平均融合 | `generate()` 默认使用；没有离散 refractory；当前连续代码只记录 STDP，不调用 `coaction.update()` |
| 可微训练 | `ensemble.py:2329+` | 不用 hard top-k/refractory/`field.update`/runtime score detach；round 1 + round 2+ field/side signals；可选 continuous | 这是训练专用梯度路径，不等于推理 forward 的数值路径 |

连续路径的真实状态链在 `ensemble.py:2086-2258`：t=0 写入场，记忆向量在 `2138-2149` 注入，t 循环在 `2169+` 进行相位/激活/场积分；最终只把时间平均 activation 作为 `final_scores`。因此“相位参与了推理”成立，但“连续路径已经形成共激活统计”不成立，除非外部脚本手工调用 `coaction.update()`。

### 4.3 Side channel、拓扑和跨规格

- `ResonanceNeuron` 的 side channel 实际消费点在 `neuron.py:667-714` 的 `side_signals` 分支。
- `ResonanceEnsemble` 构造 geometry 和跨规格投影，但不会在生产装配时自动调用 `build_topology()` / `establish_topology_channels()`。
- 训练/分析脚本通常先在 `scripts/training/analyze_side_channels.py:64-89` 等位置重建 topology，再加载 checkpoint。
- 生产 loader 主要在 `loader.py:1111-1173` 从协作产物恢复已有 channel/projector；没有产物就没有训练好的 peer 关系。
- `Cortex.add_neuron()` 会把新 neuron 加入 ensemble；`Cortex.revive_neuron()` 明确提示 side channel topology 需由调用方重新建立，见 `cortex.py:959-964`。

结论：side channel 机制本身 ✅；生产“动态拓扑自动建立”🟡；新生/复活后的拓扑自动闭合 ⚠️。

### 4.4 稀疏路由和空间扩散

- Sparse Router：`ensemble.py:462-474` 默认 `use_sparse_router=False`；如果训练产物有 `sparse_router_state`，loader 在 `loader.py:1262-1286` 动态创建并恢复。状态恢复已经补齐，但“默认启用”不成立。
- SpatialDiffuser：`spatial_diffusion.py:40-168` 是可微图 Laplacian 扩散；Ensemble 只有 `spatial_diffusion_enabled=True` 且 alpha>0 才创建，默认关闭，见 `ensemble.py:293-377`。因此它是实验/训练候选，不是当前生产默认机制。
- Geometry：`ensemble.py:868-884` 创建并注册给 CoactivationTracker；距离门控能影响共激活/扩散先验，但它不自动改变 side channel 拓扑。

## 5. 相位、振荡器和调质线路

### 5.1 PhasorDynamics / Gamma

`neuroplex/resonance/phasor.py` 的真实职责：

- `register_neurons/assign_phase_by_domain`：构建 `[N,2]` phasors 和 `[N]` omega 参数；
- `binding_tensor`：可微相位相似度，供 `forward_train`；
- `evolve`：可微 Kuramoto 计算；
- `kuramoto_step`：推理状态推进时 `no_grad` 写回 phasors；
- `pairwise_binding`：推理评分/写入调制；
- `phases` 属性：兼容旧 Gamma 接口。

生产 loader 在推理模式把 phasor 参数冻结（`loader.py:559-565`），但 `continuous_forward` 仍通过 `kuramoto_step` 推进运行时相位。训练时 `forward_train` 通过 `evolve/binding_tensor` 保留梯度。相位“存在且参与”✅；相位参数“会被默认生产推理在线学习”❌。

旧 `GammaOscillator` 的 monkey-patch 版本在 `gamma_oscillator.py:234-269`，仅在 fallback 或显式调用时使用。它把 `field.write/update` 包装为 gate；默认 Phasor 路径不应把旧标量 Gamma 的所有行为都假设为同时存在。

### 5.2 BioOSS o 型振荡节点

`oscillator.py:36-130` 定义两个无内容输出头的节奏节点：

- `omega/coupling/gaba_amp` 是可学习参数；
- `step/unit/gaba_gate` 是推理 float 状态路径；
- `theta_tensor/phase_unit_tensor/gaba_gate_tensor` 是训练可微路径；
- `make_default_oscillators()` 默认创建 theta + gamma 两个节点。

实际消费点：

- 连续推理 `ensemble.py:2169-2185` 用 o 型相位牵引 p 型 phasor；
- 连续推理 `ensemble.py:2244-2258` 用 GABA gate 写入抑制场；
- `forward_train` 的 continuous 分支在 `ensemble.py:2543-2565`、`2751-2772` 使用训练侧相位/损失；
- 睡眠 Phase 1.8 `sleep_engine.py:1144-1256` 只优化 oscillator omega/coupling/gaba_amp 和 rhythm/phase loss。

结论：BioOSS 在连续路径 ✅，离散默认路径不是主要消费方 🟡，不是内容神经元/语言头 🔬。

### 5.3 NeuromodulatorState

`neuro_modulation.py:29-173` 的真实影响：

- dopamine → 学习率倍数/FFN gain；
- serotonin → refractory multiplier；
- norepinephrine → field write scale / attention temperature；
- low dopamine 可触发 neurogenesis 信号。

推理 Ensemble 在 `ensemble.py:1324-1337` 读取温度/FFN，离散 round 在 `1435-1444` 读取不应期调制，写场处读取 NE scale；睡眠训练在 `_train_single_neuron` 计算学习率，见 `sleep_engine.py:1877+`。状态由 `Cortex.save_state/load_state` 持久化。它是实接入 ✅，但“调质自动从真实奖励/环境反馈学习”尚未闭合；目标值来自引擎内部更新或外部设置。

## 6. 记忆线路：四种记忆不是一件事

### 6.1 瞬时共振场

`ResonanceField` 只保存当前 task 的向量、抑制 mask 和贡献；每次 Ensemble forward 开始 reset。它是工作态通信，不是跨会话记忆。

### 6.2 DialogueState

`dialogue_state.py:23-180` 保存最近若干轮的 `field.save_round_state()`，`start_round()` 恢复上一轮，`end_round()` 写入当前轮。Cortex 只有在外部调用 `set_dialogue_state()` 后才会使用，`assemble_cortex()` 没有默认装配它。其 `get_state_dict()` 存在，但 `Cortex.save_state()` 没有保存 `_dialogue_state`。

结论：进程内可用 🟡；默认生产未装配；跨重启 ❌。

### 6.3 FieldMemoryBank

`field_memory.py:91-366` 的实际链：

```text
外部 record_field_memory()
  → SleepEngine.pending_field_memories
  → sleep Phase 1.5 bank.consolidate()
  → field_memory.pt
  → Cortex.set_field_memory(bank)
  → generate 内 extra think 查询场
  → retrieve_with_phase() 当前 top-1
  → seed_memories 写入 round 2+/continuous 场
```

具体事实：

- `WriteGate` 输入只有 field vector 与最近相似度，近似“新颖/冗余”，没有未来效用、奖励或结果；
- `Cortex._generate_p7()` 在 `cortex.py:2374-2400` 只在 bank 非空时额外前向查询，并取 top-1；
- 正常 `generate()` 没有 `record_field_memory()` 调用；
- 只有 `generate_task_chain(record_memory=True)` 在 `cortex.py:1264-1385` 明确记录，或外部验证脚本手工记录；
- bank 的 `save()` 在 `field_memory.py:319-337` 没保存 `phase`，`load()` 在 `339-366` 也不恢复 `phase`。因此相位记忆跨重启会丢失，这是已确认的具体 bug。

### 6.4 Agent 文本记忆

`ContextManager` 的实现位于 `neuroplex/agent/context_manager.py:42-864`：

- `remember()` 写 context cache，并可同步到 WorkingMemory、MemorySystem、语义记忆和持久 JSON；
- `build_context()` 做关键词/语义检索、历史和长期记忆拼接；
- `consolidate_for_sleep()` 在睡眠时提升短期记忆、保存持久记忆；
- `LifeScheduler` 每 5 个 heartbeat 调 `decay_memories()`，见 `life_scheduler.py:553-561`。

但本次全库调用检索没有发现 `Cortex.generate()` 自动调用 `get_context_manager().build_context()` 或 `add_message()`。所以它是 agent 层可用的文本记忆系统，不是当前 Cortex 神经元场的自动输入/输出闭环。

`neuroplex/agent/working_memory.py` 是文件/工具结果 LRU 文本缓存；`neuroplex/brain/working_memory.py` 是另一个接口。loader 注入的是 agent 侧实例，Cortex 的 brain 侧字段仍然只是兼容引用。这两个工作记忆不能混称。

## 7. 睡眠、回放和训练的真实关系

`SleepEngine.sleep()` 的阶段顺序在 `sleep_engine.py:314-444`：

```text
1 memory_consolidation
1.5 field_consolidation
1.6 synaptic_consolidation
1.7 forward_replay
1.8 osc_train
2 model_training（配置开启时）
3 knowledge_integration
3.5 experience_consolidation
4 evaluation
5 recursive_improvement
```

| 阶段 | 实际代码 | 真正训练/更新的对象 | 状态 |
|---|---|---|---|
| 1 | `sleep_engine.py:1258-1293` | ContextManager + agent WorkingMemory 清理/巩固 | 🟡 不自动把场记忆转为文本记忆 |
| 1.5 | `586-615` | FieldMemoryBank 去重、写 gate、持久化 | ✅ 但写入依赖 pending 队列 |
| 1.6 | `617-778` | 高频 bank 条目；shadow neuron 的 LoRA/直前向；近似 token 对齐 | 🔬 只训练指定 dialogue 域，非 ensemble 协作训练 |
| 1.7 | `838-1135` | 场向量 replay；shadow `field_read_layers/gate + LoRA`；judge-driven decay | 🔬 依赖已固化条目或 replay buffer |
| 1.8 | `1144-1256` | oscillator omega/coupling/gaba_amp；`forward_train(continuous=True)` | 🔬 节奏参数训练，不是语言能力训练 |
| 2 | `1295-1645` | FeedEngine 样本；shadow neuron；每个域 `_train_single_neuron` | ✅ 但路线是 direct neuron.forward |
| contrastive | `2066-2398` | shared embedding、adapter、field_write、prototype EMA、route/proto/align loss | 🔬 仍是单 neuron + 对比信号，不是 full ensemble round training |
| multimodal | `2399-2525` | ensemble route，但主要训练 modality projection | 🔬 多模态专线 |
| 3 | `2534-2565` | SleepConsolidator：side channel 强化/下调、STDP structure/prune/fingerprint | ✅ 结构巩固；非反向梯度 |
| 3.5 | `2567-2610` | agent 长期记忆转 FeedEngine 样本 | 🟡 文本训练数据闭环，不等于 field memory |
| 4 | `2612+` | PPL/activity/connectivity/maturity/inhibitory 评价、lifecycle.step | ✅ 生命周期决策 |

### 7.1 训练脚本的三条互不等价线路

1. 单神经元能力训练：
   `train_compact_simple.py`、`train_compact_parallel.py`、`train_domain_target_sft.py`、`train_neurons_from_scratch.py`、`finetune_neuron_dialogue.py` 都直接调用 `neuron.forward()`。这训练的是单 neuron 的语言头/adapter/embedding，不训练 peer channel 的端到端协同。
2. 群体协作训练：
   `train_cross_domain_collab.py:1030-1067`、`finetune_cross_spec.py:738-799`、`finetune_side_channels.py:594-606`、`train_round_level_quality.py:400-423` 调用 `ensemble.forward_train()`。这条路径才把 field conditioning、side channel、融合、diversity/quality/contrastive 等纳入梯度。
3. 睡眠训练：
   `SleepEngine._train_single_neuron()` 在 `sleep_engine.py:1877-2064` 又回到 direct `neuron.forward(field_state=None, round_num=1)`；睡眠的 field replay 在 1.7 只训练 shadow 的读路径和 LoRA；因此“睡眠已运行”不能推出“群体协作权重已更新”。

`forward_train()` 自己明确写出差异：`ensemble.py:2329+` 声明它不使用 hard top-k、refractory、runtime `field.update/score detach`，而推理 `forward/continuous_forward` 正好使用这些运行时动力学。这是设计上的训练/推理分离，不应在评估时混为同一行为。

## 8. 经验、身体和生命周期线路

### 8.1 FeedEngine

`feed_engine.py:83+` 的 `feed()` 依次收集 conversation、knowledge store、data collector，生成按域样本并写入 `feed_data/pending_samples.jsonl`；`get_pending_samples_by_domain()` 在 `410-427` 提供给 SleepEngine。`feed_text()`/`feed_file()` 是外部显式入口，`body.limbs` 的执行结果通过 loader `loader.py:614-693` 接到 FeedEngine。

这条链是“数据→睡眠训练”，不是“交互结果→场记忆”。交互要进入 FieldMemoryBank 仍需显式 `record_field_memory()`。

### 8.2 PlayEngine：当前不可作为高共振 replay 证据

`play_engine.py:158-278` 的 `_free_resonance_session()`：

- 读取 topic 后直接对每个 `neuron` 调 `neuron.forward(shared_emb, return_logits=False)`；
- 没有经过 `Cortex.think()`、`ResonanceEnsemble.forward/continuous_forward`，所以没有共享 task field、round 评分、refractory 或连续相位链；
- 接着读取 `output.get('resonance_score', 0.0)`，但 `ResonanceNeuron.forward()` 没有该 key；
- replay 记录又依赖 `neuron._last_field_state`，但 neuron.forward 没有写这个属性。

因此 loader 的 PlayEngine 接线 ✅，PlayEngine 的“自由共振→高共振状态→SleepConsolidator”功能闭合 ⚠️。历史 A/B/C/D 验证脚本中大量手工调用 `sleep_engine.record_field_memory()` 和 `record_high_resonance_state()`，只能证明手工编排后的机制，不证明普通 PlayEngine 自动产生 replay。

### 8.3 生命周期

`lifecycle.py:25-752` 提供：

- apoptosis survival score、突触修剪和 dead cleanup；
- maturity register/tick/resonance weight；
- neurogenesis error history、domain diagnosis、规格选择、孤立模式检测；
- `LifecycleManager.step()` 调整状态机。

真实调用在 SleepEngine：

- `sleep_engine.py:1554-1604` 读取 FeedEngine 错误率，触发 neurogenesis 诊断和孤立模式检测；
- `sleep_engine.py:2728-2770` 调 `lifecycle.step()`，然后由 Cortex revive/dead/neurogenesis 处理。

结论：生命周期不是死代码 ✅；但新生/复活后的 side-channel topology 和 phasor/geometry 注册要单独验证，不能由“Cortex.add_neuron 成功”推断完整接入。

## 9. 状态持久化分布和已确认缺口

| 状态 | 保存位置 | 恢复位置 | 实际结论 |
|---|---|---|---|
| neuron backbone/域 head | 各 `neuron_*.pt` | loader/Cortex load | ✅ |
| shared embedding、lm_head、adapter | `cortex_state.pt` | `Cortex.save/load_state` `cortex.py:375-539` | ✅ |
| neuromodulator/coaction | `cortex_state.pt` | Cortex save/load | ✅ |
| SleepConsolidator replay | `cortex_state.pt` | Cortex save/load | ✅，前提是 replay 真有生产写入 |
| oscillator 参数 | `cortex_state.pt` | Cortex save/load | ✅；相位 float 运行轨迹不是完整 checkpoint 语义 |
| side channels/projectors/router/W_cond | 协作 checkpoint | loader `_load_collab_weights...` | 🟡 依赖协作产物，非 Cortex state 统一保存 |
| transient task field | 不保存 | 每次 forward 新建/reset | ✅ 按设计；不是长期记忆 |
| DialogueState | 只有对象 `get_state_dict()` | 外部手工保存/恢复 | 🟡 Cortex save_state 不承载 |
| FieldMemoryBank entries | `sleep_data/field_memory.pt` | SleepEngine lazy load | ⚠️ entry `phase` 在 save/load 丢失 |
| ContextManager persistent memory | `data/agent_memory.json` | ContextManager.set_persistent_path | 🟡 agent 层独立于 Cortex field |
| brain WorkingMemory | 进程内对象 | 无 Cortex 持久化 | ⚠️ 生产生成不读 |

这说明当前项目不是一个单一 checkpoint 能恢复全部认知状态的系统。至少存在 `neuron ckpt / cortex_state / collab ckpt / field_memory.pt / agent_memory.json / optional DialogueState` 六类状态边界。

## 10. 验证脚本能证明什么，不能证明什么

| 验证 | 实际覆盖 | 不能据此宣称 |
|---|---|---|
| `tests/resonance/test_forward_train_smoke.py` | `forward_train` 可微路径基本契约 | 默认生产 generate 的全链路正确 |
| `tests/test_population_baseline.py` | 最小 population ensemble/forward_train | 9 成员 checkpoint 的协作产物已加载 |
| `tests/resonance/test_side_channels.py`、`test_ensemble_side_channels.py` | channel 建立/消费/状态接口 | loader 无 checkpoint 时会自动建拓扑 |
| `tests/resonance/test_ensemble_state_dict.py` | ensemble 聚合状态保存恢复 | FieldMemoryBank/DialogueState 与 Cortex state 一起恢复 |
| `verify_c26_auto_memory.py`、`verify_c26_memory_read_gen.py` | FieldMemoryBank 读入和 seed 注入 | 普通 generate 会自动写 memory |
| `verify_c26_sleep_e2e.py`、`verify_c27_forward_replay.py` | 手工/编排的 field memory → sleep 固化/replay | 用户交互自然产生 pending memory |
| `verify_c27_osc_sleep_e2e.py` | oscillator sleep training | 语言 neuron 协作能力获得训练 |
| `verify_c27_self_organize.py` | 生命周期/新生/自组织实验 | 新 neuron 复活后 topology/phasor 全自动闭合 |
| `verify_play_engine_*` | 大量场景统计、探索、sleep、D1 | PlayEngine 自身已正确捕获 resonance；脚本中多处手工注入 memory/replay |
| `train_cross_domain_collab.py` / `finetune_*` | `forward_train` 协作训练 | SleepEngine Phase 2 会走同一协作梯度路径 |

当前最重要的证据边界是：测试通过说明某条函数路径可以工作，不说明生产装配会调用这条路径。

## 11. 机制状态总表

| 机制 | 生产默认推理 | 生产默认睡眠/训练 | 结论 |
|---|---|---|---|
| 独立 ResonanceNeuron | 是 | 是 | ✅ 基础能力单位 |
| shared embedding + domain tokenizer | 是 | 是 | ✅ |
| field write/read | 是（continuous/离散） | 是（部分直训/回放） | ✅ 但训练口径分叉 |
| side channels | 依赖协作产物 | 协作脚本/Phase 3 | 🟡 |
| refractory/WTA | 离散推理 | 不进 forward_train 梯度 | ✅/🔬 |
| continuous phase activation | generate 默认使用 | 可微 continuous 可训练 | ✅ |
| coactivation update | 离散 forward；continuous 本身没有 | sleep 消费统计 | ⚠️ 默认连续文本链统计不完整 |
| STDP firing record | 离散/连续/训练都记录 | sleep 才 apply | ✅/🔬 |
| sparse router | 默认关闭，checkpoint 可恢复 | 训练脚本可启用 | 🟡 |
| spatial diffusion | 默认关闭 | 显式启用才训练/评估 | 🔬 |
| neuromodulation | 是 | 是 | ✅ |
| FieldMemory read | bank 非空时 | Phase 1.5 写入 | 🟡 top-1、依赖显式写入 |
| normal interaction FieldMemory capture | 否 | 无 pending 就无操作 | ⚠️ |
| DialogueState | 外部注入才有 | 不由 Cortex state 保存 | 🟡 |
| agent ContextManager | 独立 agent API 可用 | sleep 可巩固 | 🟡 不自动进入 Cortex.generate |
| PlayEngine replay | 路径存在但契约断裂 | 依赖手工/其他调用者 | ⚠️ |
| Lifecycle | Sleep evaluation 调用 | 新增/隔离/复活/凋亡 | ✅ 但动态拓扑需补证 |

## 12. 真实运行时 trace 证据（2026-08-20）

诊断命令：

```powershell
python scripts/training/diag_runtime_mechanism_trace.py
```

原始结果：`reports/runtime_mechanism_trace_20260820.json`。该脚本使用真实
`assemble_cortex()` 装配，不训练、不保存 checkpoint，并在 `Cortex.generate()` 和
`PlayEngine._free_resonance_session()` 外围记录调用事件。

### 12.1 `assemble_cortex → Cortex.generate` 实测

| 观测项 | 实测结果 | 代码结论 |
|---|---:|---|
| 最终装配成员 | 9 | `code/en/math/zh` + 5 个 dialogue neuron，装配不是只有 5 个 |
| 中文 prompt 实际 active 集合 | 6 | 5 个 dialogue neuron + `zh` general；`en/math/code` 没进入该次 active 路径 |
| `Cortex.think()` | 1 次 | 生成默认确实进入 think |
| continuous ensemble | 1 次完整进入/退出 | `collab_mode="continuous"` 确实走到 `Ensemble.continuous_forward()` |
| neuron forward | 18 次 | 6 个 active neuron × round 1/2/3 |
| field reset/write | 发生 | round 1 写场，round 2/3 读取并继续写场 |
| BioOSS inhibit | 发生 | theta/gamma 两个振荡器向场写入抑制项 |
| phase | `mean≈0.324486`, `lock≈0.784527` | 相位状态实际参与 continuous 路径 |
| side signal | 0 | 本次生产装配没有产生可消费的侧通道信号 |
| coaction update | 0 次 | continuous 路径本次没有更新 coaction |
| pending field memory | `0 → 0` | 普通 `generate()` 没有自动捕获场记忆 |
| sleep replay buffer | `0 → 0` | 本次交互没有产生 replay 样本 |

每次 `ResonanceNeuron.forward()` 的实际返回键包含
`field_vector/field_confidence/logits`，但不包含 `resonance_score`；对象也没有被设置
`_last_field_state`。这直接验证了代码审计中关于 PlayEngine 字段契约断裂的判断。

### 12.2 PlayEngine 实测

PlayEngine 的前置条件全部满足：模块存在、绑定 Cortex、9 个 neuron、TokenizerHub 和
shared embedding 均存在；tokenizer 和 shared embedding probe 也成功。但真实行级 trace
在 `neuroplex/life/play_engine.py:211-212` 记录到：

```text
TypeError: 'dict_values' object is not an iterator
```

异常随后被 `play_engine.py:276-278` 的外层兜底吞掉，返回 `None`，因此本次没有任何
`neuron.forward()`、共振判定、`record_high_resonance_state()` 或 replay 写入。

这意味着历史 PlayEngine 验证脚本即使完成，也不能证明 PlayEngine 自身的自由共振→replay
生产链闭合；还需要区分“容器迭代器错误”和“直接 neuron probe 与真实 Ensemble 输出契约不一致”
这两个问题。

## 13. Legacy 修复建议（已被原生 Taiji 主线暂停）

基于源码和运行时证据，Legacy NeuroPlex 不能直接进入大规模训练，也不能先假设场记忆控制器已有可靠捕获点。当时建议是：

> **修复 PlayEngine 的实际运行契约：先修复首个 neuron 参数的迭代器错误，再让 PlayEngine
> 通过真实 `Cortex.think()/Ensemble` 协作结果获取场状态和 resonance 分数，并为这条路径增加
> 行级 trace 回归；完成后重新验证是否产生高共振 replay。**

本步骤只修复运行线路和验证，不启动训练、不改变 9 成员生产权重；场记忆自动捕获和 coaction
连续路径补全要等这条真实 replay 边界重新跑通后再定。

## 14. PlayEngine 运行契约修复落地（2026-08-20）

§13 唯一下一步已实现并验证。三处源码修复（`neuroplex/life/play_engine.py::_free_resonance_session`）：

| 编号 | 修复点 | 旧实现（断裂） | 新实现（闭合） |
|---|---|---|---|
| **R-PE-1** | line 211 迭代器 bug | `next(self._cortex.neurons.values())` 抛 `TypeError: 'dict_values' object is not an iterator`，被外层 except 吞掉 → PlayEngine 永远返回 None | `next(iter(self._cortex.neurons.values()))` |
| **R-PE-2** | 共振信号源 | 直调 `neuron.forward(shared_emb)` 读 `output['resonance_score']`（§3.3 确认 ResonanceNeuron.forward 不产出此字段）→ activated_neurons 永远为空、coaction / replay 永不触发 | 走 `cortex.think(shared_embeddings=shared_emb, collab_mode="continuous")`，共振分来自 `result["final_scores"]`（per-neuron 时间平均激活），按群体内 max 归一化到 [0,1] 与原 0.3/0.5 阈值语义对齐 |
| **R-PE-3** | field_state 来源 | 读 `neuron._last_field_state`（§3.3 确认 forward 不写该属性）→ record_high_resonance_state 即使触发也写入 None | `cortex.get_last_field_state()`（取 `ensemble._get_task_field()`，即真实线程本地任务场，§2.1 所有权结论） |

### 14.1 回归验证

| 脚本 | 依赖 | 结果 |
|---|---|---|
| `scripts/training/verify_play_engine_contract_mock.py` | 无需 checkpoint（mock cortex） | **13/13 PASS** — 4 场景全过：① final_scores 全正 → record 调用 + field_state 非空张量；② final_scores 空 → 低质量活动不抛；③ field_state=None → record 跳过不抛；④ coaction 接线 → update 收 ≥2 active_ids |
| `scripts/training/verify_play_engine_runtime_contract.py` | 9 成员 production checkpoint（`data/neurons` + `data/foundation_v1_dual` + `collab_v3_c24v2.ckpt.pt`） | **待用户在含 checkpoint 的环境运行** — 5 维判据：C1 行级 trace 无 TypeError；C2 cortex.think 被调用；C3 final_scores 非空；C4 返回非 None PlayActivity；C5 record_high_resonance_state 调用且 field_state 非空 |
| 源码级 inspect 检查 | 无依赖 | **PASS** — `next(iter(` / `cortex.think(` / `get_last_field_state()` 三处修复点均在代码中；`neuron._last_field_state` / `output.get('resonance_score'` / `next(self._cortex.neurons.values())` 三处旧断裂契约已从代码中移除（仅余注释中的解释性引用） |

### 14.2 行级 trace 回归守卫

`diag_runtime_mechanism_trace.py` 用 `sys.settrace()` 记录 `_free_resonance_session` 的所有行号 + 异常事件；新版 `verify_play_engine_runtime_contract.py` 在此基础上新增 5 维判据，使 PlayEngine 不能再"静默回归"——任何把 `cortex.think()` 改回 `neuron.forward()`、或把 `get_last_field_state()` 改回 `neuron._last_field_state`、或重新引入 `next(dict.values())` 的改动都会被立即判 FAIL。

### 14.3 行为变化边界

- **不影响其他 play engine 验证脚本**：B1/B1-bis/B2/C1/C2/D1/A4/A5 全部直接调 `sleep_engine.record_field_memory()` 和 `sc.record_high_resonance_state()` 手工注入 replay（§8.2 已记录），不依赖 `_free_resonance_session` 内部记录。本修复只让"自动生产路径"开始具备同样的能力。
- **不写 checkpoint、不训练**：仅推理路径修复，9 成员 production weights 完全不动。
- **不改变 `play()` 公开接口**：`_free_resonance_session` 仍是 `play()` 候选活动之一，返回 `PlayActivity | None` 的契约不变；只是从"永远 None"变成"按共振强度返回"。

### 14.4 新的唯一下一步

§13 的"修复 PlayEngine 运行契约"已完成。新的下一步收敛为：

> **用户在含 9 成员 checkpoint 的环境运行 `verify_play_engine_runtime_contract.py`，确认生产路径 5/5 PASS 后，再决定场记忆自动捕获（普通 `Cortex.generate()` 自动 record_field_memory）和 coaction 连续路径补全（`continuous_forward` 内调 `coaction.update()`）的优先级与实现顺序。**

场记忆自动捕获与 coaction 连续路径补全仍是 §11 表中两项 ⚠️ 缺口；它们的实现边界取决于本节修复在生产路径上的实测结果。

## 15. C28 Gap 1 + Gap 2 落地（2026-08-20）

§14.4 的两项 ⚠️ 缺口已实现并验证。两处源码修复均冻结 9 成员生产权重、不写 checkpoint、不训练。

### 15.1 Gap 1 — 场记忆自动捕获（`neuroplex/brain/cortex.py`）

| 修复点 | 旧实现（断裂） | 新实现（闭合） |
|---|---|---|
| `generate()` / `_generate_p7()` 签名 | 无自动沉淀参数 | 新增 `auto_capture: bool = True`，向后兼容（隔离验证脚本传 False） |
| 普通 `generate()` 调用 | 不调 `record_field_memory`（§6.3 ⚠️ 缺口）→ 普通对话经验无法进入 sleep 固化→replay 训练闭环 | `_generate_p7()` 返回前调 `_capture_field_memory(prompt, result_text)`，把 (field_state, prompt[:40], generated_text, phase) 喂给全局 `SleepEngine.record_field_memory()` |
| SMCS 多候选路径 | N/A | 候选生成阶段 `auto_capture=False`（避免重复沉淀），仅对最终选中候选沉淀一次 |
| 退化重试路径 | N/A | 两次 `_generate_p7` 各自沉淀（field_state 不同，记忆库按相似度去重，可接受） |

**捕获模式**与 `generate_task_chain` [cortex.py:1392](file:///workspace/neuroplex/brain/cortex.py#L1392) 现成模式一致：`get_sleep_engine().record_field_memory(fs, label, text=text, phase=self.get_last_phase())`，区别仅在 label 来源（task_chain 用 stage template，generate 用 prompt[:40]）。field_state 取 `self.get_last_field_state()`（真实任务场，§2.1 所有权结论），phase 取 `self.get_last_phase()`（C27 KoPE 相位归属记忆）。

**行为变化边界**：
- 生产 `generate()` 现在自动喂全局 SleepEngine（loader 用 `get_sleep_engine()` 装配，即全局单例）的 `pending_field_memories` 队列 → 睡眠 Phase 1.5 固化进持久场记忆库 → replay 训练可消费
- 28 个调 `cortex.generate()` 的 verify/diag 脚本默认开启 auto_capture，会向全局 SleepEngine 单例的 pending 队列追加（瞬时，非致命；这些脚本多用本地 SleepEngine 实例跑固化，全局单例的 pending 不会被消费，属良性副作用）
- 隔离需求时传 `auto_capture=False`

### 15.2 Gap 2 — coaction 连续路径补全（`neuroplex/resonance/ensemble.py`）

| 修复点 | 旧实现（断裂） | 新实现（闭合） |
|---|---|---|
| `continuous_forward()` t-loop | 仅记 STDP，不调 `coaction.update`（§4.2/§11 ⚠️ 缺口）→ 连续动力学下的共激活结构无法生长 | 每积分步 STDP 记录后、`lateral_inhibition_norm` 前插入 `self.coaction.update(active_this, round_num=t+1)`，带 try/except 非致命包装 |

**模式**与离散 `forward()` [ensemble.py:2788](file:///workspace/neuroplex/resonance/ensemble.py#L2788) 一致：`if self.coaction is not None and len(active_this) >= 2: try: self.coaction.update(...) except ...`。门控 `len(active_this) >= 2` 与 `CoactivationTracker.update` [tribal.py:83](file:///workspace/neuroplex/resonance/tribal.py#L83) 单 neuron 短路一致。`CoactivationTracker.update` 用 EMA 累积 slow 矩阵，多次调用安全（不会重置）。

### 15.3 回归验证

`scripts/training/verify_c28_gap1_gap2_contract.py`（无需 checkpoint，mock + 源码级 inspect）**21/21 PASS**：

- **Gap 1（11 维）**：G1.1 `_capture_field_memory` 直接调用 → record_field_memory 收到正确 (field_state, label, text, phase)；G1.2/G1.3/G1.4 auto_capture 透传门控；G1.5/G1.6/G1.7 源码级 inspect（参数存在 / 门控存在 / 方法定义存在）
- **Gap 2（7 维）**：G2.1 continuous_forward 含 coaction.update；G2.2 带 round_num=t+1 + 非致命包装；G2.3 len>=2 门控；G2.4/G2.5 离散 forward 对照基准

### 15.4 §11 缺口表状态更新

| 缺口 | 修复前状态 | 修复后状态 |
|---|---|---|
| §6.3 普通 `Cortex.generate()` 不自动 `record_field_memory` | ⚠️ 缺口 | ✅ 闭合（Gap 1，auto_capture 默认开） |
| §4.2/§11 `continuous_forward` 不调 `coaction.update` | ⚠️ 缺口 | ✅ 闭合（Gap 2，每积分步补全） |
| PlayEngine `_free_resonance_session` 契约断裂 | ⚠️ 缺口 | ✅ 闭合（§14，R-PE-1/2/3） |

### 15.5 新的唯一下一步

三项机制缺口（PlayEngine 契约 / 场记忆自动捕获 / coaction 连续路径）已全部闭合。下一步收敛为**用户在含 9 成员 checkpoint 的环境运行两条生产路径验证**：

1. `python -u scripts/training/verify_play_engine_runtime_contract.py` — PlayEngine 自由共振→高共振 replay 链 5/5 PASS
2. `python -u scripts/training/diag_runtime_mechanism_trace.py` — 重跑机制 trace，确认：
   - `pending_field_memories: {before: 0, after: >0}`（Gap 1 闭合证据，普通 generate 现在自动沉淀）
   - `coaction_pairs: >0`（Gap 2 闭合证据，continuous 路径现累积共激活）
   - `play_result: 非 None`（PlayEngine 修复证据）

两条生产验证全过后，自举门槛 A（"眼睛驱动手"闭环：对话→经验→sleep 固化→replay 训练→能力增长）的运行时契约在源码层面完整闭合，可进入下一阶段决策：是否启动小规模 replay 训练验证能力增长信号，或继续推进门槛 B/C 的其他缺口。

## 16. 当前归属

§14–§15 的修复已并入代码，但它们属于冻结的 Legacy NeuroPlex 运行事实，不进入原生 Taiji forward。项目当前名称、Seed 上层模型与 Taiji 基底的边界，以 `ARCHITECTURE_DIRECTION_2026_08.md` 和 `TAIJI_SUBSTRATE_ARCHITECTURE.md` 为准；本文件不再发布项目主线的“下一步”。
