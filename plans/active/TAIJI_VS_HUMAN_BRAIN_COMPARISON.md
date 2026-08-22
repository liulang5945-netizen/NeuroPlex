# Taiji、Transformer 与生物神经系统的边界比较

> 本文依据顶层 `taiji/` Native v7 源码与 N5–N11/M5–M6 实验，不把工程名称当成生物等价或 AGI 证明。

| 维度 | Transformer | Taiji Native v7 | 生物启发边界 |
|---|---|---|---|
| 输入 | token/patch embedding | raw-byte receptor population | 感受器有固定物理来源；Taiji 当前仅实现 byte |
| 时间 | 位置编码和上下文窗口 | 每次观察推进持久状态 | 生物时间连续且多尺度；Taiji 当前是离散 tick |
| 上下文 | 对历史位置做 attention | membrane/activity/trace 压缩历史 | 有界状态更接近持续动力学，但会遗忘 |
| 通信 | 动态全局加权 | 压缩固定 fan-in reciprocal/recurrent edges | 工程稀疏图不等于真实突触 |
| 动作读出 | dense LM head | 全坐标稀疏折叠后的公共 48 通道 + 单一 motor | 解决证据可比性，不等于基底核/运动皮层 |
| 稀疏 | 通常 dense block | 压缩固定 fan-in 边 + 单 fan-out 感受器 | 已按边执行；仍是通用 gather/scatter，不是真实脉冲硬件 |
| 学习 | 全局反向传播/BPTT | local prediction/state/motor/memory delta | 局部性更强，但真实可塑性和调制更复杂 |
| 工作记忆 | KV cache/context | membrane + activity + trace | N8 已证明固定延迟 trace 因果性 |
| 情景记忆 | 通常外接 vector/KV store | 固定群体上的重叠 engram + recurrent completion | M5 只有 8 条经历，远非海马—皮层系统 |
| 行动闭环 | 通常外部 agent 包装 | pending action → reward/outcome sensation → 三因子更新 | N11 是最小符号环境，尚非真实具身世界 |

## 当前能够成立的结论

- Taiji 已经不是 Transformer 外围插件；输入、状态转移、学习、运动和生成都由独立算法完成。
- Native v5 checkpoint 原子保存固定器官拓扑、edge weights、pending action/experience、reward baseline、场状态、全部认知状态与 RNG。
- N5 以完整 v7 的 83,841 个 active learned parameters 达到 94.12% byte-cycle accuracy，并自由生成 8 个正确后继；其中情景/慢巩固参数在被动序列基准中不写入。
- N7 在一阶上限 50% 的歧义流达到 100%；全状态切除回落到 50%，说明有限动态状态确实参与历史条件化。
- N8 在四字符共同干扰后保持 100%；清零 trace 降至 50%，只保留 trace 仍为 100%。
- N9 在无终点循环中连续自反馈 128/128 正确，状态上界全程成立。
- N10 的按边 forward/backproject/update 与 dense reference 等价，并保持 N5–N9 行为。
- N11 的动作真实改变 outcome sensation；奖励局部学习达到 100%，显著超过随机/action-lesion。
- M5 的八条 one-shot 经历 action recall 为 87.5%，同宽 trace-only/循环 lesion 为 25%；能恢复 outcome、time、episode、provenance 并回注下一 fabric tick。
- M6 的分布式场能用自身 novelty/value/familiarity/time 信号选择 replay，并经慢速 signed cortical path 沉淀 contingency；12/12 seed 均 4/4，control 均为 25%。

## 当前不能成立的结论

- 不能称为人脑仿真、语言智能或 AGI；
- 八条微型情景和四对 contingency 不能推出大容量抗干扰、自传连续性或开放域巩固；
- 尚无内生想象、完整世界模型、自我模型、内在价值系统或多感官器官；
- 通用 PyTorch sparse gather/scatter 不能等同生物事件计算或硬件能效；
- 非 Transformer 本身不保证通用智能。

当前最关键边界从“能否巩固无条件 contingency”转向“能否巩固情境因果链”：Native v7 的 replay 只驱动 action→outcome，没有重建 cue，因此尚不能在 episodic lesion 后形成 cue-conditioned policy。M7 必须补上 cue→action→outcome 顺序，并用顺序 lesion 证明不是词频联结。
