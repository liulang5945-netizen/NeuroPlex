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
| `tests/taiji_native/` | 独立性、局部性、状态、感受器覆盖、N5–N11/M5–M6 | ✅ 23 passed |
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

## 6. 当前唯一下一步

修复 **replay 选择覆盖不均**。

已用 `_diag_m6_coverage.py` 在真实 `consolidate` 路径上（同一 RNG 流、同一 6-epoch 预训练、同一被验证器打分的 `full` arm）把每次 accepted replay 实际排练的 pair 与同一次运行的 per-pair margin 对齐，5 seed × 384 cycle：

| seed | 排练份额 0/1/2/3 | 最低份额 | accuracy | 读不出的 pair |
|---|---|---:|---:|---|
| 11 | 48.3 / 27.3 / **1.8** / 22.6 | 1.8% | 0.75 | `2` |
| 17 | 11.0 / 47.6 / 28.8 / 12.6 | **11.0%** | **1.00** | 无 |
| 29 | 3.5 / 26.9 / **67.2** / 2.4 | 2.4% | 0.50 | `0` `3` |
| 43 | 20.9 / 14.5 / **63.1** / 1.5 | 1.5% | 0.50 | `1` `3` |
| 61 | **0.3** / 63.9 / 12.7 / 23.2 | 0.3% | 0.50 | `0` `3` |

结论是定量的、不是轶事：**份额低于 4% 的 5 个 pair 全部读不出；高于 4% 的 15 个 pair 里 13 个读得出**；唯一拿到 4/4 的 seed 17 也正是唯一最低份额超过 10% 的 seed。`mis-rehearsed = 0`（381/382/375/325/332 次全部排练的是真 pair），所以问题不是重激活错、而是重激活的**分配**错：`priority = familiarity_confidence * resonance * selection * recency` 里每一项都随熟悉度单调上升，形成正反馈——越巩固越被抽中，越被抽中越巩固，被冷落的那条永远追不上。生物睡眠不是这样：已巩固的痕迹会退出重放竞争。

需要让场自己的信号包含"这条已经巩固够了"的抑制项（例如把已下降的局部预测误差反馈进 `priority`，使 error 低的 engram 自然让位），让覆盖自发均衡。禁止外部 replay 列表、外部配额/轮询、per-engram 计数器等不属于场自身状态的簿记。判据：5 seed 的最低排练份额都进入 ≥ 8%，`full` 的 4 对 true-cell 全部战胜竞争者，且 5 seed 的 gain 中位数高于当前 +0.50。

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
