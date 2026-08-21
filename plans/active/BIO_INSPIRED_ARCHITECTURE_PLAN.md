# Taiji 当前实施计划

> 本文件只记录当前源码状态、已复现实证与唯一下一步。历史 NeuroPlex/D1/PlayEngine 结论不混入当前执行线。

## 1. 当前架构

```text
ByteSensor
  → reciprocal predictive region 0..R
  → [all current activities; all slow traces]
  ↔ distributed EpisodicField + one-tick cortical feedback
  → balanced SparseReceptorBank
  → shared K-channel motor evidence
  → ByteMotor
  → emitted byte loops back to ByteSensor
```

区域 decoder 学习下层预测，transition 学习局部下一状态，motor 学习动作结果，field 在 outcome 到达后学习 cue→event completion 与 causal readout。所有更新均在 `torch.no_grad()` 内执行，不使用 autograd、optimizer、BPTT、attention、tokenizer、教师模型、蒸馏或 event K/V slot。

## 2. 实现地图

| 代码 | 已实现职责 | 状态 |
|---|---|---|
| `taiji/config.py` | 形状、动力学、学习率、稳定上界与上下文维数 | ✅ |
| `taiji/sparse.py` | 压缩固定 fan-in、gather/scatter reciprocal 投影、按边 delta | ✅ |
| `taiji/state.py` | 区域/场/整机状态、pending action/experience 原子事务 | ✅ |
| `taiji/organs.py` | raw-byte 感觉器官、全坐标覆盖的稀疏感受器组、唯一 byte 运动器官 | ✅ |
| `taiji/memory.py` | 分布式事件编码、循环补全、novelty/reward 写门、因果 readout | ✅ |
| `taiji/fabric.py` | 分层预测误差、递归状态、抑制、稳态、场 feedback 与区域局部学习 | ✅ |
| `taiji/model.py` | observe、act、settle_action、learn/score/generate、Native v5 checkpoint | ✅ |
| `taiji/environment.py` | action-dependent sensation/reward 环境协议 | ✅ |
| `tests/taiji_native/` | 独立性、局部性、状态、感受器覆盖、命名/边界守护、N5–N11/M5–M6 | ✅ 27 passed |
| `verify_taiji_native_v5.py` | 独立端到端、主动/情景状态与压缩存储基准 | ✅ PASS |
| `verify_taiji_n7_context.py` | 二阶歧义与因果切除基准 | ✅ PASS |
| `verify_taiji_n8_delayed_trace.py` | 共同干扰后的 slow-trace 必要性/充分性 | ✅ PASS |
| `verify_taiji_n9_long_free_run.py` | 128 步纯动作回灌与逐 tick 状态上界 | ✅ PASS |
| `verify_taiji_n10_sparse_migration.py` | dense 算子参考、v2 行为参考与 N5–N9 回归 | ✅ PASS |
| `verify_taiji_n11_active_environment.py` | reward action、随机与 action-lesion 因果对照 | ✅ PASS |
| `verify_taiji_m5_episodic_field.py` | one-shot field、同宽 trace、循环 lesion、metadata/readback | ✅ PASS |

## 3. Native v5 实测

固定 seed `7`、区域 `[64,48]`、区域 fan-in `16`、运动感受器 `48`：

| 指标 | 结果 |
|---|---:|
| active learned parameters | 62,529 |
| fixed receptor edges | 224（每个皮层坐标恰好一条） |
| actual learned scalar storage | 62,529 |
| dense-equivalent learned scalars | 112,241 |
| learned synapse edges / int32 indices | 62,272 / 62,272 |
| byte-cycle accuracy | 0% → 94.12% |
| mean surprise | 5.4041 → 0.1090 |
| surprise reduction | 97.98% |
| free generation | `a → bcdabcda`，8 步全部正确 |
| checkpoint exact next step | PASS |
| Transformer/NeuroPlex runtime dependency | 0 |

N7 流 `axbcxd × 4` 中，当前符号同为 `x`，历史分别要求后继 `b`/`d`：

| 对照 | 歧义位置 accuracy |
|---|---:|
| 完整 Taiji 状态 | 100% |
| 只看当前 byte 的一阶基线 | 50% |
| 每 tick 清空全部动态状态 | 50% |
| 只清空 slow trace | 100% |

N7 单独能成立的结论是：持久动态状态已具有二阶上下文能力，但短间隔主要由 membrane/activity 承担，N7 本身没有证明 slow trace。

N8 在线索与 probe 之间加入共同干扰 `1234`：完整状态与 trace-only 均为 100%，在 probe 前清零 trace 或清零全部动态状态均为 50%。这证明 slow trace 对该固定延迟任务既必要又足够；N8 本身仍不是可检索情景记忆。

N9 在明确无终点的 `abcd × 4` 循环合同下，只给 prompt `a`，随后 128 个动作全部自反馈：128/128 正确、无非法/boundary 动作，membrane/trace/threshold 每 tick 有界。若训练含结束 boundary，则第四轮后停止是正确监督，不能拿来要求无限循环。

N10 把全部区域突触从 masked dense 改为 `[post, local_edge]` 压缩行。dense reference 最大误差：forward `2.98e-8`、backproject `0`、local update `0`；N5–N9 与 v2 报告一致。包含场以后，小 v5 基准的权重+int32 索引为 dense learned-weight 字节的 `111.22%`，默认配置投影为 `98.59%`。因此当前结论仍是“真实按边执行并在足够稀疏时节省存储”，不是“小张量必然更快”。

N11 的两 cue/两 action 环境中，action 同时改变 reward 与下一 `+/-` sensation。200 次在线交互后：学习组末 40 次 `100%`，随机基线 `50%`，禁用 action learning `57.5%`；deterministic policy 两 cue 全对。Taiji 只收到 reward 与 outcome sensation，未收到正确动作标签。

M5 在同一个 128-unit 场里各写一次八条 action/outcome/time/episode/provenance 经历；写入用 singleton demonstrated affordance 并关闭 fabric/motor 学习，因此只声明 associative recall。跨 episode action recall `87.5%`，同宽 trace-only 与 recurrent-association lesion 都是 `25%`；outcome/provenance `100%`，episode identity `75%`，time cosine `0.519`，cortical feedback 会改变下一 tick。拓扑始终 4,096 条 association edge，event slot 为 0。

M6 在关闭 episodic action/readback 的前提下，只靠场自己的 novelty/value/familiarity/time 信号选 engram 并重激活同一 fabric。384 cycle、5 seed 全部 `pass`（修复前 2/5）：

| seed | 状态 | gain | pre | full | ctrl | engram lesion | recurrent lesion |
|---|---|---:|---:|---:|---:|---:|---:|
| 11 | pass | +0.25 | 0.50 | 0.75 | 0.50 | 0.50 | 0.50 |
| 17 | pass | +0.75 | 0.25 | 1.00 | 0.25 | 0.25 | 0.25 |
| 29 | pass | +0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| 43 | pass | +0.50 | 0.00 | 0.50 | 0.00 | 0.25 | 0.00 |
| 61 | pass | +0.25 | 0.25 | 0.50 | 0.25 | 0.25 | 0.25 |

10 项 check 全部成立，包含三条禁止项：评测期无 episodic readback、sleep 只改 cortex（11 个非 fabric 张量 `|dw| = 0`）、拓扑固定且 event slot 为 0。两个 lesion 组都不高于 control，说明增益确实来自 engram 内容与循环补全，而不是 replay 这个动作本身。

**关键修复（homeostasis 棘轮）**：此前 10 个假设全部被反证后，真正原因是恒常性设定点的路径不对称——probe 走 `reset_dynamics` 把 threshold 重置到 `threshold_base`，而 replay 走 `clear_dynamics` 保留设定点。replay 的输入是退化的：单一符号连驱 16 tick，没有醒时流量平衡它，于是被 engram 驱动的单元每 tick 增 `rate*(1-target)`、沉默单元只减 `rate*target`，正好在承载记忆的单元上形成 7:1 棘轮。实测设定点冲到 `0.4280`（21× base），而 `activity` 直接减掉 threshold，写入基底塌到 probe 的 1/22；`local_update` 对 `|trace|` 是线性的，写入几乎归零，`captured` 在近零 trace 上变成任意值，某个 decoder row churn 了 118 次 rewire 也不收敛。

修复是 `fabric.step(..., adapt_homeostasis=True)`，`consolidate` 的两个 replay 循环传 `False`：睡眠期**读**设定点但绝不**写**它。这既保留醒时学到的设定点，又不让退化 burst 有权改写它——生物的恒常性可塑性是小时级、群体驱动的，同理。选型不是靠基底保真度（reset 与 freeze 都能把基底救回来、rewire 都从 311 降到 16），而是靠 probe 真正读到的证据仲裁：freeze 在 4 对里 3 对的 true-cell 位移更大，mean `|delta|` `0.0088` vs `0.0073`，logit spread `0.05539` vs `0.04907`。

修复后 rewire 会**饱和**：24/48/96/192 cycle 都停在 12 个 contact；缺陷版本则是 8/23/43/81，随 cycle 近似线性增长、永不终止。

报告：`reports/taiji_native_v5_20260821.json`、`reports/taiji_m5_episodic_field_20260821.json`、`reports/taiji_m6_endogenous_replay_20260821.json`、`reports/taiji_n11_active_environment_20260821.json`、`reports/taiji_n10_sparse_migration_20260821.json`，以及 N7/N8/N9 独立报告。M6 的 5 seed 明细在 `reports/_sweep_{11,17,29,43,61}.json`，修复前基线保留在 `reports/_prefix/`。v2–v4 只保留为迁移参考。

## 4. 本轮删除的错误机制

Native v5 不再让 257 个动作各自随机抽取不同皮层坐标，也不让所有动作共同只看 224 维中的同一随机 48 维。正式 motor 使用平衡、固定极性的稀疏感受器组；正式 memory 也不恢复旧 cue/value cell，而用固定群体上的重叠 engram、循环 resonance 和共享 readout。

旧 `neuroplex.taiji` K/V cell、全局 top-k、输出平均、event gateway 回接 Cortex、蒸馏底座和小 Transformer 身份继续保持废止。

## 5. 当前限制

- 当前只证明小型 byte 流、短程二阶上下文和八条 one-shot 情景，不代表语言理解；
- 场已能跨 reset 检索 action/outcome/metadata，但尚未证明大容量、长期抗干扰或自传连续性；
- PyTorch 已真实按边执行，但通用 gather/scatter 尚非定制 event kernel，小张量加速不作保证；
- M6 的绝对水平仍低：5 seed 的 `full` 只有 0.50–1.00（chance 0.25），gain 只有 +0.25–+0.75。已定量归因于 **replay 选择覆盖不均**（见第 6 节表），是**选择覆盖**问题，不是可塑性问题；
- 写入基底仍比 probe 基底大 1.13–1.43×，且在一次 bout 内轻微下漂（`0.2159→0.2065`）。cos ≥ 0.996 说明方向对、幅度不对，尚未解释；
- 已有 reward action、provenance 与内生 replay/巩固，但尚无内生想象生成、多感官器官；
- 现有 5 个 dialogue + 4 个 general Transformer 成员只作为冻结离线基线，不进入 Taiji forward。

## 6. M6 replay 选择覆盖失衡（已修复，2026-08-21）

### 6.1 症状（修复前基线）

已用 `_diag_m6_coverage.py` 在真实 `consolidate` 路径上（同一 RNG 流、同一 6-epoch 预训练、同一被验证器打分的 `full` arm）把每次 accepted replay 实际排练的 pair 与同一次运行的 per-pair margin 对齐，5 seed × 384 cycle：

| seed | 排练份额 0/1/2/3 | 最低份额 | accuracy | 读不出的 pair |
|---|---|---:|---:|---|
| 11 | 48.3 / 27.3 / **1.8** / 22.6 | 1.8% | 0.75 | `2` |
| 17 | 11.0 / 47.6 / 28.8 / 12.6 | **11.0%** | **1.00** | 无 |
| 29 | 3.5 / 26.9 / **67.2** / 2.4 | 2.4% | 0.50 | `0` `3` |
| 43 | 20.9 / 14.5 / **63.1** / 1.5 | 1.5% | 0.50 | `1` `3` |
| 61 | **0.3** / 63.9 / 12.7 / 23.2 | 0.3% | 0.50 | `0` `3` |

结论是定量的、不是轶事：**份额低于 4% 的 5 个 pair 全部读不出；高于 4% 的 15 个 pair 里 13 个读得出**；唯一拿到 4/4 的 seed 17 也正是唯一最低份额超过 10% 的 seed。`mis-rehearsed = 0`（381/382/375/325/332 次全部排练的是真 pair），所以问题不是重激活错、而是重激活的**分配**错。

### 6.2 被证伪的原方向：`priority` 不是杠杆

原计划设想把已下降的局部预测误差反馈进 `priority`，让 error 低的 engram 让位。落地前用两个已有测量否掉了：

- `reports/taiji_m6_endogenous_replay_20260821.json`：`accepted 95 / cycles 96`、`mean_priority 0.148` vs `replay_priority_threshold 0.05`。**接受门槛几乎从不生效**，垄断不是"门槛把饥饿的 engram 拒了"；
- `model.py:355-365`：`endorsement = min(1.0, priority / threshold)`，在 `0.148/0.05 ≈ 3.0` 处**已饱和于 1.0**。任何加在 `priority` 上的抑制项要削掉 3× 才开始影响接受，而这 3× 的削减会先全部落到 `learn_scale` 上——即先削弱可塑性，再谈覆盖。

### 6.3 真实成因与修复

垄断来自 §6 原诊断漏掉的**第二条正反馈**：`replay()` 的 `seed_drive` 里含 `+ (1 - memory_trace_decay) * previous.trace`，也就是把刚排练完那条 engram 的残留痕迹**正向**喂回种子。场被吸引到它刚被吸引过的地方，这正好作用在诊断所测量的补全盆地上。

修复是一次由机制含义决定的符号翻转，落在 `taiji/memory.py:625-627`：

```python
adapted = previous.threshold + replay_fatigue_gain * (previous.trace - previous.trace.mean())
```

- **为什么清醒时该加、睡眠时该减**：清醒时是外部 cue 决定回忆什么，痕迹只负责把相继的 cue 绑起来，所以它属于 drive（`recall()` 至今仍这样做）；睡眠时没有 cue，痕迹成了唯一决定"重生成什么"的量，此时它必须表达"这条刚排练过"——即真实皮层的 spike-frequency adaptation，实现为一个抬高刚放电单元阈值的瞬态偏置；
- **为什么必须零均值**：`resonance`/`familiarity` 都从重生成 pattern 的**幅度**读出。单向压低会把整场活动一起压暗，使 `priority` 因"与哪条 engram 胜出无关的理由"跌破门槛（实测 accepted 325→98）。皮层稳态守恒的是群体总活动，适应只重分配由哪些单元承担这份活动；零均值化后疲劳只动选择、不动表达；
- **同时抬高内生噪声** `replay_noise_scale 0.25 → 0.75`：疲劳只覆盖约 2–3 个 bout（`memory_trace_decay=0.72`），它打断"连续重复"，但决定默认落入哪个盆地的是 `seed_drive` 里每个 bout 完全相同的 `reward_code` 项（权重 0.60）。真实重放由内生随机性（sharp-wave ripple 的随机内容）点燃，而不是恒定驱动；
- **走过的弯路（勿重走）**：曾尝试在疲劳竞争后用未疲劳阈值再 settle 一次，以"把选择与表达解耦"。接受数恢复但覆盖增益全丢（最低份额回落到 1.7/11.5/3.9/2.1/1.1）——未疲劳的最后一步会直接吸回主导 engram。**疲劳必须在 pattern 被表达时仍在场，而不只在它竞争时在场。**

新增 `replay_fatigue_gain: float = 1.20`（非负校验在 `config.py:157`）。无新增持久状态、无 `STATE_VERSION` 变更、无 checkpoint 格式变更——`adapted` 是瞬态局部量，写回 `MemoryState` 的仍是 `previous.threshold` 原值。gain 试过 0.60/1.20/2.00，2.00 反而更差（最低份额 4.2%）。

### 6.4 验收结果

`python scripts/training/_diag_m6_coverage.py 384 11 17 29 43 61`：

| seed | 排练份额 0/1/2/3 | 最低份额 | accuracy | accepted |
|---|---|---:|---:|---:|
| 11 | 27.4 / 29.3 / 10.0 / 33.3 | **10.0%** | 0.75 | 351 |
| 17 | 30.5 / 28.4 / 23.3 / 17.8 | **17.8%** | **1.00** | 331 |
| 29 | 13.0 / 39.0 / 30.8 / 17.2 | **13.0%** | 0.75 | 354 |
| 43 | 24.1 / 34.1 / 27.8 / 14.1 | **14.1%** | **1.00** | 320 |
| 61 | 8.2 / 56.6 / 15.1 / 20.1 | **8.2%** | 0.75 | 279 |

- 5 seed 最低排练份额全部 ≥ 8%（判据达成），最差 0.3% → 8.2%；
- `covered 4/4` 在全部 5 个 seed 成立（修复前仅 seed 17）；
- accuracy 均值 0.65 → **0.85**，两个 seed 达到 1.00（修复前仅一个）；
- `mis-rehearsed = 0` 保持；accepted 回到 279–354（基线 325–382 量级），未牺牲写入 burst 数量；
- 未使用外部 replay 列表、外部配额/轮询或 per-engram 计数器，覆盖均衡完全出自场自身动力学。

回归：`verify_taiji_m6_endogenous_replay.py` `status: pass`（10/10 check），`verify_taiji_m5_episodic_field.py` `pass`，`pytest tests/taiji_native -q` 27 passed，`pytest tests/ -q` 74 passed。

### 6.5 残留 3 对 margin 为负的定量归因（已完成，2026-08-21）

`full` arm 的 4 对 true-cell 未全部战胜竞争者（seed 11 的 `2`、29 的 `0`、61 的 `3`，margin -0.0021 / -0.0009 / -0.0012）。份额已不是瓶颈——这三条分别占 10.0% / 13.0% / 20.1%。用 `scripts/training/_diag_m6_margin.py` 做了三层测量，把 §6.5 原本的二选一（写入剂量 vs 竞争者被顺带抬高）替换成一条闭合的定量律。

**归因得以成立的前提**：`sparse.local_update` 对突触前痕迹**线性**（`edge_weight += lr * error ⊗ trace[pre_index] / scale`），所以只要把 4 个探测 basis 从睡前 checkpoint 冻结下来（探测口径与 `_evaluate_contingency` 完全一致），就能在每次 accepted replay 前后对 `decoders[0]` 在全部 4 个 basis 上各求一次值，把差分全额记到中间那一次 replay 名下——即每个 burst 对**每个** basis 干了什么，而不只是对自己那个。

**测量一：每次排练的 margin 增量矩阵**（`384 cycle`，×1e4，行=burst pair，列=探测 basis）

| seed | burst | n | →basis 0 | →1 | →2 | →3 | 读回 |
|---|---|---:|---:|---:|---:|---:|---|
| 11 | `3->?` | 117 | -0.55 | -0.01 | -0.00 | **5.88** | ok |
| 11 | `1->-` | 103 | -0.00 | **4.98** | *-2.22* | -0.01 | ok |
| 11 | `0->+` | 96 | **4.03** | -0.02 | 0.01 | *-1.36* | ok |
| 11 | `2->!` | 35 | 0.00 | *-2.22* | **7.04** | -0.01 | **WRONG** |
| 29 | `1->-` | 138 | *-1.69* | **—** | — | — | ok |
| 29 | `0->+` | 46 | **4.24** | — | — | — | **WRONG** |
| 61 | `1->-` | 158 | — | **14.24** | — | *-1.69* | ok |
| 61 | `3->?` | 56 | — | — | — | **4.54** | **WRONG** |

对角线是自我教学，非对角线是附带损伤，且**只在 basis 相关时出现**：cosine 0.37 → -2.22（对称）、0.31 → -1.69、0.28 → -1.69/-1.36/-0.55，而 cosine ≤ 0.02 的对全部是 ~0.00。合并成一条律：

```
margin_i  ≈  Σ_j  n_j · d_ij         d_ii > 0，d_ij < 0 且随 cos(basis_i, basis_j) 增长
```

**剂量假说被明确排除**：seed 11 失败的 `2` 拥有四对中**最高**的单次自我增益（+7.04e-4），仍然输，因为 35 × 7.04 抵不过 103 × 2.22 的反向充电；seed 61 的赢家 `1->-` 自我增益是全场最高（+14.24e-4），正是这份垄断饿死了与它相关的 `3`。三个残留失败全部是"最相关那一对里排练较少的一方"。对照 seed 17 从反面确认：最大 cosine 仅 0.14、crosstalk ≤ -1.15、份额最均（101/94/77/59），4/4 全对。

**测量二：相关性来自递归扩散吗——否**（`sweep` 模式，burst 长度 1→12 重探 basis）

| seed | tick 1 max cos | tick 8 | tick 12 | 走向 |
|---|---:|---:|---:|---|
| 11 | 0.321 | 0.369 | 0.332 | 基本持平 |
| 29 | 0.270 | 0.314 | 0.308 | 基本持平 |
| 61 | 0.258 | 0.275 | 0.224 | 后段下降 |
| 17 | 0.200 | 0.139 | 0.123 | **单调下降** |

原假设是"一个字节只驱动 fan-in 命中它的 ~4/64 个单元、四动作近正交，之后每一 tick 都靠**所有动作共享**的 transition 矩阵扩散，于是 burst 越长四个 basis 越趋同"。**证伪**：相关性在第 1 个 tick 就已满额，对照 seed 17 甚至随 burst 变长而下降。缩短 burst 不是解，成因是静态的。

**测量三：相关性是单一共模**（`origin` 模式，把 basis 拆成四动作均值与残差）

| seed | 共模能量 | 原始 cos max / mean | 残差 cos max / mean | 全 4 动作都驱动的单元 | 其阈值 |
|---|---:|---|---|---:|---:|
| 11 | **35.0%** | 0.369 / 0.138 | **-0.014** / -0.330 | 3 / 45 触及 | 5.10× base |
| 29 | 30.4% | 0.314 / 0.084 | -0.109 / -0.318 | 2 / 56 | 4.60× |
| 61 | 28.5% | 0.275 / 0.050 | -0.019 / -0.299 | 1 / 52 | 3.10× |
| 17 | 29.1% | **0.139** / 0.053 | -0.220 / -0.328 | 1 / 56 | **1.60×** |

- **4 个向量的共模能量下界是 1/k = 25%**（恰好正交时取到）。实测 28.5–35.0%，即超出下界 3.5–10.0 个百分点，seed 11 超得最多、也正是 cosine 最高的那个；
- 均值 cosine 由这一个标量近似决定：`mean_cos ≈ (4f-1)/3` 给出 0.133 / 0.072 / 0.047 / 0.055，实测 0.138 / 0.084 / 0.050 / 0.053（seed 29 偏差最大，因其 basis 范数最不齐：活跃 21/29/28/34）；
- **关键**：剥掉共模后，6 个残差配对的 cosine **无一为正**（max 分别是 -0.014 / -0.109 / -0.019 / -0.220）。残差均值 ≈ -1/3 是零和约束的自动结果、不构成证据，但"max ≤ 0"是：若哪两个 basis 还共享动作特异的子结构，必有一对残差正对齐。没有。**全部正相关都住在一个 rank-1 方向里**，不是逐对的几何问题；
- 那几个"滥交单元"只有 1–3 个，但阈值被 homeostasis 抬到 3.1–5.1× base（对照 seed 17 只有 1.60×）。高阈值是它们长期对一切输入放电的**记录**、不是成因：稳态一直在压它们，只是压不过来；而巩固期 `adapt_homeostasis=False`，这份压制在写入时完全缺席。

**结论**：残留失败既不是剂量不足、也不是"同一 burst 顺带抬高竞争者的行"，而是**每次写入都有约 30% 的剂量落在一个四动作共享的 rank-1 基底方向上，而 4 个探测都要透过它读数**。于是教一对必然按 cosine 比例损伤与它相关的另一对，胜负由份额加权的 `Σ n_j·d_ij` 决定。因为串扰是单一方向，**消掉这一个方向即可一次性消掉全部串扰**。

### 6.6 当前唯一下一步

在 `fabric.step` 中把全局标量抑制换成逐单元竞争性抑制（侧抑制 / 去相关）。当前 `inhibition` 是一个 Python `float`（`inhibition_gain * positive_drive.mean()`），对所有单元**等量**相减：它能整体锐化阈值，却在原理上无法"因为某个单元对一切输入都响应"而专门压低它——而这正是共模的载体。经典的去相关机制（Földiák 式反 Hebb 侧抑制）恰好做这件事：让区域内单元互相竞争，把共同响应的分量压掉、保留动作特异的残差。判据是共模能量 f 向 25% 下界回落、`_diag_m6_margin.py` 的非对角项趋零，最终 4 对 true-cell 全部转正。


## 7. 附录：已废止的 D1 长程稳定性档案（NeuroPlex/PlayEngine）

> 完整判定标准与所有方案讨论见 [BOOTSTRAP_CRITERIA.md](BOOTSTRAP_CRITERIA.md) 第 4 节。本节只记录 v9 修复结论与对 Taiji 主线的隐含信号。

- **D1 系列目标**：1000 步压力测试下，3 组（dialogue/knowledge/unfamiliar）std ratio ≥ pre × 0.90
- **v3/v4/v5/v6/v7/v8 演化**：见 BOOTSTRAP_CRITERIA.md
- **v9（2026-08-21，方案 N 落地）**：修复 `pre_lora_l2_baseline=0.0` 的理论缺陷（`LoRA/0` 无意义使 ceiling 永远不触发），改为前 50 步 LoRA L2 均值
  - 结果 2/5：dialogue 1.0854 ✅（首次完整超过 v5 0.9127 维度）；knowledge 0.8177 ❌；unfamiliar 0.8190 ❌
  - 0 崩溃，26.3 min
  - **关键确认**：post_lora_l2 = **10.96**（v5/v7/v8 都是 11.84，**v9 < v8 -0.88**），ceiling 机制真的开始工作
  - **但 k/u 与 v5/v7/v8 字面相近**：DECAY 0.85 仍是决定性因素，ceiling 仅在 LoRA 终值上显出差异
  - 报告：`reports/play_engine_d1_fix_v9_baseline_fix_20260821.json`
- **对 Taiji 主线的隐含信号**：D1 修复无法靠调整 PlayEngine sleep 参数闭环——`std ratio 0.82-0.91` 已成天花板；要让 D1 完整通过，要么换架构路线（去 LoRA ceiling 转别机制），要么承认 D 系列不是当前瓶颈，转向 D2 长程记忆检索
- **Taiji 不复用任何 PlayEngine 代码**：Taiji 是顶层原生 TPF，无 LoRA、无 sleep、无 play engine；D1 修复经验仅作为"持续学习系统需要衰减 + 抑制上限 + 软起点"的设计直觉
