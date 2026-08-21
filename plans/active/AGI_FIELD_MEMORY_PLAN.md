# Taiji 原生记忆状态规范

> 本文只描述顶层 `taiji/` 当前真实记忆所有权。旧 `ResonanceField`、`FieldMemoryBank` 和已删除的 `neuroplex.taiji` K/V slots 都不属于正式 Taiji。

## 1. Native v1 已实现的记忆

| 类型 | 代码状态 | 时间尺度 | 清除方式 |
|---|---|---|---|
| 当前活动 | `RegionState.activity` | 当前 tick | 下一 tick 重算 |
| 膜状态 | `RegionState.membrane` | 短时递归 | 衰减或显式 reset |
| temporal trace | `RegionState.trace` | 工作上下文 | 衰减或显式 reset |
| 自适应阈值/抑制 | `threshold/inhibition` | 活动稳态历史 | 动力学变化或 reset |
| 预测记忆 | `decoder D` | 慢语义/感觉预测 | 局部 prediction delta |
| 转移记忆 | `transition T` | 慢时序模型 | 局部 state delta |
| 程序/动作记忆 | `motor M,b` | 慢动作策略 | 真实后继 motor delta |

`TaijiState` 原子保存所有快状态；checkpoint 同时保存慢突触、结构 mask 和 RNG。当前没有单独的“场记忆控制器”：区域膜状态与 trace 就是 Native v1 的分布式工作场。

## 2. 明确未实现

- 可检索情景记忆；
- autobiographical episode timeline；
- imagined/replay provenance；
- 睡眠巩固；
- 受奖励门控的长期信用；
- 跨感官共同情景。

这些能力不能继续由外部文本库或精确 K/V slot 冒充。未来情景记忆必须记录真实事件、动作、环境结果、状态摘要和因果来源，并通过 lesion 证明它改善行为。

## 3. 当前记忆反证顺序

| ID | 命题 | 状态 |
|---|---|---|
| M0 | 历史状态改变未来输出，reset 后差异消失 | PASS |
| M1 | 慢突触在线局部变化，无 optimizer | PASS |
| M2 | checkpoint 保持下一 tick 与下一次学习完全一致 | PASS |
| M3 | 相同当前输入可因不同 trace 产生不同正确后继 | 当前 N7，未验收 |
| M4 | trace lesion 显著破坏二阶任务 | 当前 N7，未验收 |
| M5 | 情景记忆优于等容量 trace-only 对照 | 未实现 |
| M6 | replay 巩固后清除情景缓存仍保留能力 | 未实现 |

## 4. 当前唯一下一步

先完成 N7/M3/M4 二阶上下文与 trace lesion。若当前 trace 连最小历史分歧都不能表达，增加情景数据库只会再次变成外挂补丁。
