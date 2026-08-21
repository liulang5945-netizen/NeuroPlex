# Taiji 原生记忆状态规范

> 本文只描述顶层 `taiji/` 当前真实记忆所有权。旧 `ResonanceField`、`FieldMemoryBank` 和已删除的 `neuroplex.taiji` K/V slots 都不属于正式 Taiji。

## 1. Native v2 已实现的记忆

| 类型 | 代码状态 | 时间尺度 | 清除方式 |
|---|---|---|---|
| 当前活动 | `RegionState.activity` | 当前 tick/快上下文 | 下一 tick 重算 |
| 膜状态 | `RegionState.membrane` | 短时递归 | 衰减或显式 reset |
| temporal trace | `RegionState.trace` | 较慢工作上下文 | 衰减或显式 reset |
| 自适应阈值/抑制 | `threshold/inhibition` | 活动稳态历史 | 动力学变化或 reset |
| 预测记忆 | `decoder D` | 慢感觉/层间模型 | 局部 prediction delta |
| 转移记忆 | `transition T` | 慢时序模型 | 局部 state delta |
| 程序/动作记忆 | `motor M,b` | 慢动作策略 | 真实后继 motor delta |

`SparseReceptorBank H` 只把全部 activity/trace 均衡折叠进公共运动证据通道；它是固定器官拓扑，不保存经历，不得被称为记忆。

`TaijiState` 原子保存所有快状态；Native v2 checkpoint 同时保存慢突触、结构 mask、感受器 channel/polarity、motor context、概率和 RNG。当前没有单独的“场记忆控制器”：区域膜状态、activity 与 trace 组成分布式工作场。

## 2. N7 已证明和未证明的边界

二阶流 `axbcxd × 4` 中，相同 `x` 必须按历史产生 `b` 或 `d`。完整系统为 100%，全动态状态逐 tick 切除后为 50%，一阶基线也是 50%。这证明**有限持久状态参与了上下文条件化**。

但只清空 `trace` 后仍为 100%。因此 N7 没有证明慢 trace 是因果载体；短间隔线索仍可停留在 membrane/activity。不能把 N7 结果扩大解释为长期场记忆、情景记忆或人脑式工作空间。

## 3. 明确未实现

- 跨干扰延迟仍可读出的慢状态；
- 可检索情景记忆与 autobiographical timeline；
- imagined/replay provenance；
- 睡眠巩固；
- 受奖励门控的长期信用；
- 跨感官共同情景。

未来情景记忆必须记录真实事件、动作、环境结果、状态摘要和因果来源，并通过容量匹配与 lesion 证明改善行为；不能由外部文本库或精确 K/V slot 冒充。

## 4. 记忆反证顺序

| ID | 命题 | 状态 |
|---|---|---|
| M0 | 历史状态改变未来输出，reset 后差异消失 | PASS |
| M1 | 慢突触在线局部变化，无 optimizer | PASS |
| M2 | checkpoint 保持下一 tick 与下一次学习完全一致 | PASS |
| M3 | 相同当前输入可因不同完整动态状态产生不同正确后继 | PASS（N7：100% vs 50%） |
| M4 | 快状态被共同干扰后，trace lesion 显著破坏延迟任务 | 未验收（N7 trace lesion 无影响） |
| M5 | 情景记忆优于等容量 trace-only 对照 | 未实现 |
| M6 | replay 巩固后清除情景缓存仍保留能力 | 未实现 |

## 5. 当前唯一下一步

执行 N8/M4 延迟上下文任务，明确分离 activity/membrane 与 trace 的因果贡献。只有 M4 通过后，才设计可检索情景记忆；否则先修当前状态时间尺度或局部信用分配。
