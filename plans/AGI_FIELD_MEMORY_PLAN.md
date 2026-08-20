# AGI 场记忆架构计划

> 2026-08-20 重新基线：场记忆是 AGI 主线能力，不再作为睡眠模块的附属验证项。
>
> 源码审计总图：[`plans/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md`](NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md)。本计划中的“已有能力”必须以该运行地图的代码证据为准。

## 1. 结论

当前项目已经有共振场、对话轮次状态、`FieldMemoryBank`、睡眠固化和记忆向量条件化生成，
但它们还没有形成完整的长期智能闭环。源码审计后，当前实现更准确的定义是：

```text
瞬时共振场 + 可选外部向量记忆库 + 睡眠重放
```

其中“可选外部向量记忆库”只有在外部显式记录场状态后才会增长；正常 `Cortex.generate()`
目前只做读取，不自动写回。它还不是：

```text
经验 → 情景记忆 → 主动召回 → 场状态回注 → 行为改变 → 睡眠抽象巩固
```

AGI 目标下，场记忆必须是群体神经元的第一类状态层，不能只靠文本前缀、单轮 field state
或外置标签检索来替代。

## 2. 当前已有能力与真实边界

| 能力 | 当前实现 | 结论 |
|---|---|---|
| 瞬时场 | `ResonanceField.write/update/read`，供同一轮神经元协作 | 已有，但主要是工作态，不是长期记忆 |
| 多轮场状态 | `DialogueState` 保存最近轮次快照 | 有短期状态，默认进程内，缺跨会话语义索引 |
| 场记忆库 | `FieldMemoryBank` 向量、标签、文本、访问计数、去重 | 有情景记忆容器，但依赖显式写入 |
| 记忆召回 | 当前 query 场状态 top-1 检索，向量注入 round 2+ | 已能影响 logits/生成，但召回控制过于单一 |
| 睡眠固化 | Phase 1.5 场固化、1.6 LoRA 沉淀、1.7 条件化重放 | 验证闭环存在，但样本主要来自显式测试/编排 |
| 正常交互自动写入 | 普通 `generate()` 不自动记录场记忆 | **核心缺口** |
| 记忆效用学习 | `WriteGate` 主要学习新颖/冗余近似，不看未来任务收益 | **核心缺口** |
| 记忆结构 | 主要是向量 + 文本标签 | 缺时间、主体、因果、来源、置信度、奖励/惊奇度 |
| 跨重启完整恢复 | 记忆库单独由 `SleepEngine` 管理；`Cortex.save_state` 不统一承载 | 状态所有权分裂 |
| 相位持久化 | entry 内有 `phase`，但 `FieldMemoryBank.save/load` 未完整保存/恢复 | **已发现具体缺陷** |
| 生产写入来源 | `record_field_memory()` 只有显式调用者；普通 `generate()` 不自动调用 | **核心缺口** |
| PlayEngine replay | PlayEngine 直接调用 neuron.forward，却读取不存在的 resonance 字段 | **当前线路断裂** |

## 3. 面向人脑智能的目标分层

```text
当前输入 / 行动结果 / 内部误差
             ↓
      Episodic Capture Layer
  prompt、output、场轨迹、激活群体、时间、惊奇度、奖励、置信度
             ↓
      Hippocampal Episodic Buffer
  快速写入、可修改、按任务/时间/主体/相位/语义检索
             ↓
      Recall Controller
  当前场状态 + 任务需求 → top-k 记忆、门控、冲突检测、重排
             ↓
      Resonance Field Re-entry
  记忆向量/相位/关系写回场，影响神经元读路径与路由
             ↓
      行为与反馈
             ↓
      Sleep Consolidation
  高频/高价值情景 → 语义/程序性权重，低价值记忆衰减与修剪
```

## 4. 必须补齐的记忆条目契约

长期记忆条目不能只保存 `(vector, label)`，至少需要：

```text
memory_id
content / text
field_vector
phase
timestamp / session_id / turn_id
active_neurons
source_prompt / action / outcome
confidence / surprise / reward
access_count / last_access
memory_type: episodic | semantic | procedural
links: related memories / contradiction / cause-effect
consolidation_state
```

## 5. 当前架构决策

1. 保留 `ResonanceField` 作为工作态和长期记忆回注的统一通信介质。
2. 保留 `FieldMemoryBank` 作为底层存储，但在它之上增加记忆控制层，不让 bank 直接承担
   全部认知职责。
3. 正常交互必须产生结构化 episodic trace；不能要求每个调用方手工调用
   `record_field_memory()`。
4. 记忆召回必须从 top-1 升级为带效用/新颖性/冲突门控的 top-k，并记录召回对行为的影响。
5. 睡眠巩固要区分情景、语义、程序性记忆；LoRA 只是其中一个可选沉淀通道。
6. 先完成“写入—召回—行为改变—跨重启恢复”的最小闭环，再扩大训练和神经元规模。

## 6. 唯一下一步

源码审计和真实 runtime trace 已完成：普通 `Cortex.generate()` 能产生连续场和相位事件，
但 `pending memory` 仍为 `0 → 0`；PlayEngine 则在进入 neuron 前于
`play_engine.py:211-212` 因 `dict_values` 迭代器错误退出，且其后续字段契约也与
`ResonanceNeuron.forward()` 不一致。详见
[`plans/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md`](NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md)
和 `reports/runtime_mechanism_trace_20260820.json`。

因此唯一下一步是：先修复 PlayEngine 的迭代器和结果契约，让它通过真实
`Cortex.think()/Ensemble` 获取场状态与 resonance 分数，并增加回归 trace，重新确认是否能
自动产生高共振 replay。本阶段不训练生产 checkpoint、不改变 9 成员默认激活集合；场记忆
自动捕获控制器要等真实 replay 边界跑通后再实现。
