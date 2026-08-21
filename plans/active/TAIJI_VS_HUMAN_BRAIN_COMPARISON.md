# Taiji、Transformer 与生物神经系统的边界比较

> 本文依据当前顶层 `taiji/` 代码，不把工程名称当成生物等价或 AGI 证明。

| 维度 | Transformer | Taiji Native v1 | 生物启发边界 |
|---|---|---|---|
| 输入 | token/patch embedding | raw-byte receptor population | 感受器有固定物理来源；Taiji 当前仅实现 byte |
| 时间 | 位置编码和上下文窗口 | 每次观察推进持久状态 | 生物时间连续且多尺度；Taiji 当前是离散 tick |
| 上下文 | 对历史位置做 attention | membrane + recurrent trace 压缩历史 | 有界状态更接近持续动力学，但会遗忘 |
| 通信 | 动态全局加权 | 固定 fan-in reciprocal/recurrent edges | 当前 mask 只是工程稀疏图，不等于真实突触 |
| 稀疏 | 通常 dense block | 区域抑制 + 结构 mask | PyTorch 当前仍用 masked dense tensor，尚未节省真实 FLOPs |
| 学习 | 全局反向传播 | local prediction/state/motor delta | 局部性更强，但生物可塑性远比当前规则复杂 |
| 工作记忆 | KV cache/context | membrane + trace | Taiji 状态有界，需独立长期记忆系统 |
| 输出 | LM head | 单一 byte motor organ | 工程动作群体，不等于生物运动皮层 |
| 行动闭环 | 通常外部 agent 包装 | 生成 byte 会回灌 ByteSensor | 当前只闭合符号动作，尚无真实环境 |

## 当前能够成立的结论

- Taiji 已经不是 Transformer 外围插件；其输入、状态转移、学习和生成都由独立算法完成。
- 历史状态、局部更新和 checkpoint 已有因果测试。
- 小型 byte-cycle 实验表明算法可以在线降低预测惊奇并自由生成短前缀。

## 当前不能成立的结论

- 不能称为人脑仿真；
- 不能称为已经具有神经元分工、情景记忆、世界模型或自我；
- 不能从四步生成推出语言理解；
- 不能从非 Transformer 推出 AGI；
- 不能把 masked parameter count 当成实际稀疏算力。

当前最关键的生物/计算共同问题是：有限持续状态是否真的承担历史条件化。N7 二阶上下文与 trace lesion 是下一项最小因果实验。
