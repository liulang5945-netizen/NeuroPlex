# Taiji、Transformer 与生物神经系统的边界比较

> 本文依据顶层 `taiji/` Native v2 源码与 N5/N7 实验，不把工程名称当成生物等价或 AGI 证明。

| 维度 | Transformer | Taiji Native v2 | 生物启发边界 |
|---|---|---|---|
| 输入 | token/patch embedding | raw-byte receptor population | 感受器有固定物理来源；Taiji 当前仅实现 byte |
| 时间 | 位置编码和上下文窗口 | 每次观察推进持久状态 | 生物时间连续且多尺度；Taiji 当前是离散 tick |
| 上下文 | 对历史位置做 attention | membrane/activity/trace 压缩历史 | 有界状态更接近持续动力学，但会遗忘 |
| 通信 | 动态全局加权 | 固定 fan-in reciprocal/recurrent edges | 当前 mask 是工程稀疏图，不等于真实突触 |
| 动作读出 | dense LM head | 全坐标稀疏折叠后的公共 48 通道 + 单一 motor | 解决证据可比性，不等于基底核/运动皮层 |
| 稀疏 | 通常 dense block | 区域抑制 + 结构 mask + 单 fan-out 感受器 | 区域矩阵仍以 masked dense tensor 运行 |
| 学习 | 全局反向传播/BPTT | local prediction/state/motor delta | 局部性更强，但真实可塑性和调制更复杂 |
| 工作记忆 | KV cache/context | membrane + activity + trace | 状态有界；slow trace 的独立作用尚未通过延迟 lesion |
| 行动闭环 | 通常外部 agent 包装 | 生成 byte 回灌同一 ByteSensor | 当前只闭合符号动作，尚无真实环境 |

## 当前能够成立的结论

- Taiji 已经不是 Transformer 外围插件；输入、状态转移、学习、运动和生成都由独立算法完成。
- Native v2 checkpoint 原子保存固定器官拓扑、慢参数、全部认知状态与 RNG。
- N5 以 19,521 个 active learned parameters 达到 94.12% byte-cycle accuracy，并自由生成 8 个正确后继。
- N7 在一阶上限 50% 的歧义流达到 100%；全状态切除回落到 50%，说明有限动态状态确实参与历史条件化。

## 当前不能成立的结论

- 不能称为人脑仿真、语言智能或 AGI；
- N7 的 trace-only lesion 仍为 100%，不能宣称 slow trace 已是场记忆；
- 尚无情景记忆、世界模型、自我模型、价值系统、神经调制或器官级分工；
- masked parameter count 不是实际稀疏 FLOPs；
- 非 Transformer 本身不保证通用智能。

当前最关键的问题已经从“状态是否有上下文”推进为“慢状态能否在共同干扰后独立保留并支配行为”。N8 延迟上下文/trace lesion 是下一项最小因果实验。
