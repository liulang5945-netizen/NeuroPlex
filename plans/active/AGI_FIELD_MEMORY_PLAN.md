# Taiji 原生记忆状态规范

> 本文只描述顶层 `taiji/` 当前真实记忆所有权。旧 `ResonanceField`、`FieldMemoryBank` 和已删除的 `neuroplex.taiji` K/V slots 都不属于正式 Taiji。

## 1. Native v3 已实现的记忆

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

`TaijiState` 原子保存所有快状态；Native v3 checkpoint 同时保存慢突触的压缩 pre-index/edge weights、感受器 channel/polarity、motor context、概率和 RNG。当前没有单独的“场记忆控制器”：区域膜状态、activity 与 trace 组成分布式工作场。

## 2. N7/N8 已证明和未证明的边界

二阶流 `axbcxd × 4` 中，相同 `x` 必须按历史产生 `b` 或 `d`。完整系统为 100%，全动态状态逐 tick 切除后为 50%，一阶基线也是 50%。这证明**有限持久状态参与了上下文条件化**。

N7 只清空 `trace` 后仍为 100%，说明短间隔线索可停留在 membrane/activity。N8 将线索与 probe 隔开共同干扰 `1234`：完整状态 100%，清零 trace 50%，只保留 trace 100%，全状态切除 50%。因此 slow trace 对这一固定延迟行为既必要又足够。

这个结论仍不能扩大为长期场记忆、情景记忆或人脑式工作空间：N8 没有跨 episode 检索、容量竞争、来源标记或巩固。

## 3. 明确未实现

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
| M4 | 快状态被共同干扰后，trace lesion 显著破坏延迟任务 | PASS（N8：100% vs 50%；trace-only 100%） |
| M5 | 情景记忆优于等容量 trace-only 对照 | 未实现 |
| M6 | replay 巩固后清除情景缓存仍保留能力 | 未实现 |

## 5. 当前唯一下一步

M4 已闭合当前工作场的最小因果链，N9 已证明无终点循环可稳定自反馈 128 步，N10 已让底座按真实边执行。项目当前先完成 N11 环境行动学习，使事件真正包含 agent action 与 outcome；随后 M5 情景记忆才有可记录、可反事实检验的因果事件，而不是又退回文本 K/V。
