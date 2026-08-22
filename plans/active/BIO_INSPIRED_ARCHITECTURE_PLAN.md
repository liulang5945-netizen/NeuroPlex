# Seed 当前实施计划（Taiji 基底）

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
| `verify_taiji_n10_sparse_migration.py` | dense 算子参考与 N5–N9 回归 | ✅ PASS |
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

M6 在关闭 episodic action/readback 的前提下，只靠场自己的 novelty/value/familiarity/time 信号选 engram 并重激活同一 fabric。机制级判断一律读 12 seed 面板（`verify_taiji_m6_endogenous_replay.py --panel`），**不读单 seed**：

| seed | 状态 | gain | full | ctrl | margin gain |
|---|---|---:|---:|---:|---:|
| 11 | fail | +0.00 | 0.50 | 0.50 | +0.0021 |
| 17 | pass | +0.75 | 1.00 | 0.25 | +0.0081 |
| 23 | fail | +0.00 | 1.00 | 1.00 | +0.0064 |
| 29 | pass | +0.50 | 0.50 | 0.00 | +0.0043 |
| 37 | pass | +0.50 | 1.00 | 0.50 | +0.0069 |
| 43 | pass | +0.75 | 0.75 | 0.00 | +0.0048 |
| 53 | pass | +0.25 | 0.75 | 0.50 | +0.0052 |
| 61 | pass | +0.25 | 0.50 | 0.25 | +0.0040 |
| 71 | pass | +0.50 | 0.75 | 0.25 | +0.0052 |
| 79 | pass | +0.50 | 1.00 | 0.50 | +0.0062 |
| 89 | pass | +0.50 | 1.00 | 0.50 | +0.0086 |
| 97 | pass | +1.00 | 1.00 | 0.00 | +0.0071 |

面板 `status: pass`：`passing_seeds 10 / 12`、`mean_accuracy_gain_over_control +0.4583`、`no_seed_is_harmed_by_replay` 成立（12 个 gain 全部 ≥ 0）。seed 11 与 23 的失败是**这两个 seed 本身**的性质（见 §6.6 的 HEAD 对照）：它们的 control 已经等于 full（0.50/0.50 与 1.00/1.00），replay 没有可争取的余量，于是三条因果 check 同时空转，而 margin gain 仍为正。

默认 seed（29）的完整报告 10 项 check 全部成立，包含三条禁止项：评测期无 episodic readback、sleep 只改 cortex（11 个非 fabric 张量 `|dw| = 0`）、拓扑固定且 event slot 为 0。两个 lesion 组都不高于 control，说明增益确实来自 engram 内容与循环补全，而不是 replay 这个动作本身。

**关键修复（homeostasis 棘轮）**：此前 10 个假设全部被反证后，真正原因是恒常性设定点的路径不对称——probe 走 `reset_dynamics` 把 threshold 重置到 `threshold_base`，而 replay 走 `clear_dynamics` 保留设定点。replay 的输入是退化的：单一符号连驱 16 tick，没有醒时流量平衡它，于是被 engram 驱动的单元每 tick 增 `rate*(1-target)`、沉默单元只减 `rate*target`，正好在承载记忆的单元上形成 7:1 棘轮。实测设定点冲到 `0.4280`（21× base），而 `activity` 直接减掉 threshold，写入基底塌到 probe 的 1/22；`local_update` 对 `|trace|` 是线性的，写入几乎归零，`captured` 在近零 trace 上变成任意值，某个 decoder row churn 了 118 次 rewire 也不收敛。

修复是 `fabric.step(..., adapt_homeostasis=True)`，`consolidate` 的两个 replay 循环传 `False`：睡眠期**读**设定点但绝不**写**它。这既保留醒时学到的设定点，又不让退化 burst 有权改写它——生物的恒常性可塑性是小时级、群体驱动的，同理。选型不是靠基底保真度（reset 与 freeze 都能把基底救回来、rewire 都从 311 降到 16），而是靠 probe 真正读到的证据仲裁：freeze 在 4 对里 3 对的 true-cell 位移更大，mean `|delta|` `0.0088` vs `0.0073`，logit spread `0.05539` vs `0.04907`。

修复后 rewire 会**饱和**：24/48/96/192 cycle 都停在 12 个 contact；缺陷版本则是 8/23/43/81，随 cycle 近似线性增长、永不终止。

报告：`reports/taiji_native_v5_20260821.json`、`reports/taiji_m5_episodic_field_20260821.json`、`reports/taiji_m6_endogenous_replay_20260821.json`、`reports/taiji_m6_seed_panel_20260821.json`、`reports/taiji_n11_active_environment_20260821.json`、`reports/taiji_n10_sparse_migration_20260821.json`，以及 N7/N8/N9 独立报告。M6 的多 seed 明细由 `--panel` 现算并落在 seed panel 报告里，不再手工维护逐 seed 快照。v2–v4 只保留为迁移参考。

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
  - **⚠️ 2026-08-21 §6.6 实测订正**：这条对 f 的解读是错的。对**非负**向量，支撑集完全不相交时 f **恰好等于** 1/k，不是"趋近下界"；且该恒等式与向量是否零均值无关（随机对照：不相交支撑 + 全正 → f=25.0%，不相交支撑 + 零均值 → 同样 25.0%）。因为 `activity = tanh(relu(...)) ≥ 0`，四个 basis 全在非负象限内，其均值不可能为零向量。**f 因此不是"干扰强度"的度量，而是"支撑集重叠度"的一个已经饱和的代理量**，且 25% 是代数恒等式而非可逼近的物理下界。以"f→25%"为判据是无效的，详见 §6.6。
- 均值 cosine 由这一个标量近似决定：`mean_cos ≈ (4f-1)/3` 给出 0.133 / 0.072 / 0.047 / 0.055，实测 0.138 / 0.084 / 0.050 / 0.053（seed 29 偏差最大，因其 basis 范数最不齐：活跃 21/29/28/34）；
- **关键**：剥掉共模后，6 个残差配对的 cosine **无一为正**（max 分别是 -0.014 / -0.109 / -0.019 / -0.220）。残差均值 ≈ -1/3 是零和约束的自动结果、不构成证据，但"max ≤ 0"是：若哪两个 basis 还共享动作特异的子结构，必有一对残差正对齐。没有。**全部正相关都住在一个 rank-1 方向里**，不是逐对的几何问题；
- 那几个"滥交单元"只有 1–3 个，但阈值被 homeostasis 抬到 3.1–5.1× base（对照 seed 17 只有 1.60×）。高阈值是它们长期对一切输入放电的**记录**、不是成因：稳态一直在压它们，只是压不过来；而巩固期 `adapt_homeostasis=False`，这份压制在写入时完全缺席。

**结论**：残留失败既不是剂量不足、也不是"同一 burst 顺带抬高竞争者的行"，而是**每次写入都有约 30% 的剂量落在一个四动作共享的 rank-1 基底方向上，而 4 个探测都要透过它读数**。于是教一对必然按 cosine 比例损伤与它相关的另一对，胜负由份额加权的 `Σ n_j·d_ij` 决定。因为串扰是单一方向，**消掉这一个方向即可一次性消掉全部串扰**。

### 6.6 逐单元竞争性抑制（已实现并保留；原判据已证伪，2026-08-21）

**做了什么**：在 `fabric.step` 中把全局标量抑制换成逐单元竞争性抑制。原 `inhibition` 是一个 Python `float`（`inhibition_gain * positive_drive.mean()`），对所有单元**等量**相减；现在每个区域配一组侧抑制银行 `fabric.laterals[i]`（`SparseSynapses(n, n, lateral_fan_in, allow_self=False)`），抑制律为

```
inh_i ← λ·inh_i + (1−λ)·g·(1/k)·Σ_{j∈N(i)} W_ij·relu(membrane_j − θ_j)
```

学习律是 Földiák 式反 Hebb 去相关 `ΔW_ij = η(a_i·a_j − ā²)`（`SparseSynapses.anti_hebbian_update`），权重 clamp 非负（负的抑制接触会变成兴奋、把竞争反号）。

**为什么保留（12 seed 面板实测，非单 seed）**：银行初始化为 `W ≡ 1`。此时均匀邻域均值是全局均值的无偏估计，**旧的标量律恰好是新银行的退化解**，不是与之并存的第二条通路；n5 teacher-forced 精度回到 `0.9411764705882353`，与改动前逐位相同。但这只是"没退化"，不足以支撑保留。真正的依据是在 `git worktree` 上跑干净 HEAD（`0aa64f1`）与工作树的同一 12 seed 面板：

| | pass | mean gain |
|---|---:|---:|
| HEAD `0aa64f1`（全局标量抑制） | 10/12 | **+0.4167** |
| §6.6（逐单元竞争抑制） | 10/12 | **+0.4583** |

逐 seed：71 从 `+0.25 → +0.50`、97 从 `+0.75 → +1.00`，**没有任何 seed 变差**；seed 11 与 23 在**两侧都 fail**（HEAD 的 seed 11 也是 `fail gain=+0.00 full=0.50 ctrl=0.50 margin=+0.0020`），因此是 seed 自身性质，与本机制无关。结论：§6.6 是净收益，机制保留；退役的只是它原本那套（用共模 f 衡量的）验收判据。

**⚠️ 方法论规则（这一条是踩过坑换来的）**：**已提交的 report 快照不能当基线**。此前我拿 `reports/_sweep_11.json` 与新代码对比，得出"seed 11 从 pass 掉到 fail、§6.6 造成回归"的结论，并据此推演了一整轮归因（还做了 `lateral_learning_rate=1e-9` 与 `lateral_fan_in=64` 两组对照，四臂输出完全相同，两个候选机制都被排除）——因为那个快照本身是更早的代码产生的，早已过期。基线必须从干净 worktree **重新执行**。同理，机制级结论必须读 seed 面板的聚合量，单 seed 无法把机制效应与 seed 特异性分开。据此已退役并删除 `reports/_sweep_*.json` 与 `reports/_prefix/` 这一类手工维护的快照，改由验收脚本内建 `--panel` 现算，产物为 `reports/taiji_m6_seed_panel_20260821.json`。

**判据全部未达成，且判据本身被证伪**：

| lateral_learning_rate | W mean | W max | W std | 共模 f (seed 11) |
|---|---:|---:|---:|---:|
| 1e-9 | 1.00000 | 1.0000 | 0.0 | 35.1% |
| 0.02（默认） | 0.99955 | 1.0376 | 2.6e-3 | 35.1% |
| 0.5 | 0.98866 | 1.9259 | 6.4e-2 | 35.1% |
| 5.0 | 0.88593 | 9.4765 | 6.2e-1 | 35.2% |
| 50.0 | 0.44792 | 9.9944 | 1.7 | 35.6% |
| 500.0 | 0.52316 | 9.9495 | 1.8 | 35.5% |

学习率跨 **11 个数量级**、W 从纹丝不动到撞上范数上限饱和，f 始终锁在 35.1–35.6%（还微升）。5 seed 的 `origin` 复测：35.1 / 29.1 / 30.5 / 38.4 / 28.3%，与改动前的 28.5–35.0% 同一区间。seed 11 的 pair `2` margin 仍为 `-0.00214`。**"剂量不足"被排除，这是结构性无效。**

**根因（比 §6.5 的归因深一层）**：`activity = tanh(relu(membrane − θ − inh))`，故 basis **逐元素非负**（实测 `basis min = +0.0000`，负值占比 `0.000`）。于是：

1. 侧抑制只能把分量往 0 压，`relu` 之后**永远无法产生负分量**去抵消共模。压得越狠向量越稀疏，但仍全在非负象限，均值仍非零 —— 所以 f 对 g 和 η 都不敏感；
2. 共模是**非负性的代数后果**，不是促杂单元的行为后果。决定性反例：seed 61 的 4-动作促杂单元数为 **0**，f 仍有 28.3%，其中 27.5% 的共模能量落在**只被 1 个动作驱动**的单元上。§6.5 认定的"rank-1 载体 = 少数滥交单元"因此不成立；
3. 随机对照证明支撑集完全不相交时 f **恰好** = 1/k = 25.0%。真实测得 28.3–38.4% 与 25% 的差距只有 3–13 个百分点，全部来自支撑重叠，而这部分正是被非负性封死的。

**结论**：`f → 25%`、"非对角项趋零"这两条判据无效，予以退役——f 是饱和的重叠度代理量，25% 是恒等式而非可逼近下界。唯一仍然有效的判据是端到端的 **4 对 true-cell 全部转正**。下一步不应再在"压低共模能量"这个方向上投入。

**遗留的真问题**：读出通路是"非负 basis × 单层稀疏解码器"，四个探测共享同一非负锥，任何一对的写入都会沿锥内夹角泄漏到其它对。要打破这一点必须让读出侧拿到**带符号**的对比量，而不是继续在前端压制活动。

**⚠️ 2026-08-22 订正：上面这句原本举的例子（"解码器行之间的差分/归一化竞争"）是错的，且落点必须是 fabric、不是 motor。** 两条都在实现过程中被否掉，代码已整体回退（HEAD 仍为 `3fdd34a`，74 passed）。留档是因为这两个坑都不便宜：

1. **行间竞争在数学上是空操作。** 令证据为 $e_i$、总和 $S=\sum_j e_j$，"每行减去竞争行的加权均值"展开是
   $$\tilde e_i = e_i - \frac{\lambda}{n-1}\sum_{j\neq i} e_j = \Big(1+\frac{\lambda}{n-1}\Big) e_i - \frac{\lambda S}{n-1}$$
   减掉的那一项**与 $i$ 无关、对每行相同**，而 softmax 对全体 logit 平移不变，故 $\mathrm{softmax}(\tilde e/T) = \mathrm{softmax}(e/T')$，$T' = T/(1+\lambda/(n-1))$ —— **只等价于换温度**，排序与相对证据分毫不动。任何"能写成行间线性组合、且以行共享方式相减"的竞争都塌缩成这个恒等式。**在 `evidence` 上做行间竞争是死路。**
2. **对比量必须作用在 basis 上、在解码之前，且那个 basis 是 `regions[0].trace`，不是 `motor_context`。** 因为非负性，每个探测的 basis ≈ $\alpha_p u + \delta_p$（共模 $u$ 与探测无关的标量 $\alpha_p$，只有 $\delta_p$ 携带身份）；解码后得 $\alpha_p(Wu) + W\delta_p$，其中 $Wu$ 是一个**行间不均匀**的固定模式，正是 §6.5 测到的 rank-1 串扰，softmax 消不掉。曾误把机制装进 `ByteMotor.encode_context`，实测 `gain=0.0` 与 `0.75` 的四个 margin **逐位相同**（`-0.00089 / +0.00644 / +0.00170 / -0.00256`）——因为 `_evaluate_contingency` 读的是 `fabric.decoders[0].forward(regions[0].trace)`，而 motor 是另一个器官，其输出只流向 `motor.probabilities` / `motor.learn`，从不回流 fabric。该脚本自己的注释早已写明这一点：巩固只触碰 `fabric.decoders` 与 `fabric.transitions`，**motor 出睡眠时逐位不变**，所以 motor 读出面在设计上就与巩固无关。

### 6.7 公共基线离线前置验证（FAIL，2026-08-22）

已按 §6.6 的前置条件实现 `_diag_m6_margin.py locus`，冻结睡前四个 probe basis，在不改 state/checkpoint 的前提下比较 raw、oracle 共模、256-tick 在线流均值和纯残差。结果：

| seed | raw | pure residual | gain≤1 最好 |
|---:|---:|---:|---:|
| 11 | 2/4 | 2/4 | 2/4 |
| 29 | 2/4 | 4/4 | 4/4 |
| 61 | 2/4 | 2/4 | 3/4 |

所以“在 `fabric.step` 给 trace 扣公共基线”被否决，不升级 `RegionState` 或 checkpoint。seed 11 的 `0/2` 与 seed 61 的 `2` 在纯残差上仍为负；true row 已接触峰值残差单元，但固定 16-contact 支撑让 rival row 读到更大正量。构建上限从“rank-1 共模”进一步收敛为“**带符号残差 × 固定随机稀疏支撑的可分性没有结构保证**”。

**当前唯一下一步**：先做 signed-opponent basis 离线反证，不改运行态。每个 cortical 坐标变成严格零和的 ON/OFF 或正/负误差对，并为 decoder row 提供 opponent-pair 覆盖约束；完整 12-seed 面板每个 seed 都达到 4/4 正 margin 后，才允许进入 `fabric.step`。具体边界见 [SEED_ARCHITECTURE.md](SEED_ARCHITECTURE.md) §5–§6。

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
