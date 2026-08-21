# Taiji 原生场记忆计划

> **状态**：活跃子计划 · 2026-08-21
>
> 现有 NeuroPlex 记忆事实以 [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) 为准；目标状态合同以 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md) 为准。

## 1. 决策

场记忆不再作为 Transformer 群体外部的可选向量库继续扩建。Taiji 把记忆拆成持久工作场、细胞联想快记忆、运行时情景记忆、慢语义权重和程序记忆，并由同一个 `TaijiState` 持有。

```text
真实事件
  → 持续 cell/field 状态变化
  → 显著 transition 自动形成 episode
  → 目标/误差驱动召回
  → 召回以 memory event 进入细胞（保留来源）
  → 行为与环境结果
  → eligibility + 调质局部更新
  → 睡眠把重复有用的快记忆巩固到慢权重/拓扑
```

## 2. 当前 NeuroPlex 的真实边界

| 机制 | 已有能力 | 不能据此宣称 |
|---|---|---|
| `ResonanceField` | 一次 ensemble 调用内可读写、评分和抑制 | 跨调用持续；它在 `forward/continuous_forward` 开始时 reset |
| `DialogueState` | 可保存/恢复某个 field 对象的轮次快照 | 真实 task field 已形成多轮记忆；当前对象所有权不一致 |
| `FieldMemoryBank` | 场向量去重、top-k 检索、持久化、回注 | 正常交互自动形成完整情景记忆 |
| `Cortex.generate` | bank 非空时可先做 query forward，再 top-1 回注 | 记忆有因果结构，或召回经过冲突/结果验证 |
| `SleepEngine` | pending 向量固化、LoRA/field-read replay | 唤醒期原生在线学习；其更新仍围绕 CE/NLL/Transformer |
| agent 文本记忆 | 可保存和拼接文本上下文 | 已自动进入 Cortex/Taiji 的认知状态 |

额外已确认缺口：`FieldMemoryBank.save()` 当前没有把 entry 的 `phase` 写入 payload；Cortex state、field memory、DialogueState 和 agent memory 也不属于同一个原子 checkpoint。

## 3. Taiji 的记忆所有权

| 层 | 状态所有者 | 写入触发 | 遗忘/巩固 |
|---|---|---|---|
| 感觉缓冲 | sensor | 每个外部事件 | 短衰减 |
| `field.fast` | TaijiField | 同 tick 同步/竞争 | 快衰减 |
| `field.working` | TaijiField | 任务相关 field proposal | 中衰减、显式会话策略 |
| `field.context` | TaijiField | 目标、自我和长期背景 | 慢衰减、受保护 |
| fast associative memory | TaijiCell | 高误差/高奖励局部关联 | 使用率与干扰驱动替换 |
| episodic store | TaijiRuntime | 显著真实 transition 自动记录 | replay 优先级、合并、归档 |
| semantic memory | cell 慢权重/原型 | 睡眠巩固 | 稳定性保护与下调 |
| procedural memory | motor/topology | 动作—结果反馈 | 失败抑制、成功强化 |

## 4. 情景条目合同

一个 episode 不能只保存向量和文本标签，至少包含：

```text
episode_id, tick_start, tick_end
real_events, imagined_events
field_before, field_after
cell_state_refs, active_cells, emitted_events
prediction, chosen_action, observed_outcome
reward, surprise, confidence, goal_state
causal_parent_ids, replay_count, consolidation_state
```

`real` 与 `imagined` 必须分开；只有环境回执能写 `observed_outcome`。这条边界防止内部预测在反复 replay 后变成“伪经验”。

## 5. 读写规则

### 写入

- 工作场每 tick 更新，不经过外部 `record_field_memory()` 才生效。
- episode 在显著度、预测误差、动作结果、目标变化或奖励超过阈值时自动封装。
- fast memory 只写活动细胞的局部键值和 eligibility，不广播改写全群体。
- 重复事件可以提高稳定度，但不能把同一 transition 重复计为多次独立证据。

### 召回

1. 用当前 cell/field/goal 生成查询；
2. 先按相似、目标、时间、因果邻接产生候选；
3. 由当前预测误差和冲突门重排，不固定 top-1；
4. 以 `kind=memory` 的事件读入，并保留 episode 来源和真实/想象标记；
5. 只有召回后确实改善预测或行动结果，才增加使用价值。

### 睡眠

- 优先 replay 高惊奇、高价值、未解决误差和有遗忘风险的 episode；
- 新旧 episode 交错，避免只巩固最近数据；
- 快权重写回慢权重前在 shadow state 做稳定性/旧记忆回归；
- replay 失败必须可回滚，不能靠固定全局衰减修饰所有记忆。

## 6. 可证伪门槛

| ID | 记忆命题 | 通过线 |
|---|---|---|
| M0 | field 具有持续因果性 | 经过空白 tick 仍按衰减保留；reset 消融后行为变化消失 |
| M1 | 一次性局部关联 | 单次关联后同类 probe 误差下降 ≥ 30%，无全局 optimizer |
| M2 | 来源可区分 | 系统状态可区分 sensory/memory/imagined，不把想象当环境结果 |
| M3 | 情景召回有效 | 召回组优于 memory lesion 组，且错误召回可被冲突门拒绝 |
| M4 | 持续学习 | 20 个顺序关联后首四分位保留率 ≥ 70% |
| M5 | 睡眠巩固 | 清空 fast memory 后，巩固组保留显著高于未巩固组；旧任务不越过遗忘阈值 |
| M6 | 完整持久化 | save/load 后下一 tick、召回候选和输出一致 |

## 7. 迁移边界

- 现有 `FieldMemoryBank` 和 DialogueState 保留用于 Legacy NeuroPlex 回归，不删除数据。
- Taiji-0 Phase A 只实现持久 field 和完整 state round-trip，不提前伪造 episode/睡眠闭环。
- Phase B 才加入 fast memory、episode 和局部 eligibility。
- 在 Taiji 独立通过 M0–M6 前，不把现有场向量直接灌入 TaijiField；需要时通过带来源的 event gateway 转译。

## 8. 当前唯一下一步

Taiji-0 的 M1/T4 已通过：一次真实关联只写 cue-active cells，精确 cue 的 MSE 下降 100%，慢权重和无关细胞不变。下一步执行 M4/T5：顺序学习 20 个关联，验证最早 5 个保留率 ≥ 70%，同时报告槽占用、键干扰和 memory lesion；先区分规则遗忘与容量淘汰。旧 PlayEngine replay 修复继续暂停。
