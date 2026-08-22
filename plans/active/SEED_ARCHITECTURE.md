# Seed 模型总架构

> 决策日期：2026-08-22
>
> 当前代码：`seed/` 是模型边界，`taiji/` 是唯一原生计算基底，`neuroplex/` 是冻结的 Transformer 对照。

## 1. 三层所有权

```text
Seed model / organism                         seed/
  ├─ identity, lifecycle and future organs
  ├─ environment-facing model API
  ├─ Seed checkpoint envelope
  └─ Taiji substrate                         taiji/
       ├─ raw-event sensation
       ├─ persistent predictive fabric
       ├─ distributed episodic field
       ├─ local plasticity and replay
       └─ action organ

Frozen comparison runtime                    neuroplex/
  └─ nine-member Transformer population; never imported by Seed/Taiji
```

Seed 与 Taiji 不能再互换使用：Seed 是会继续增加器官、目标、发展阶段和群体协作的模型主体；Taiji 是它执行感觉—状态—记忆—动作循环的底层算法。当前 `Seed` 只组合一个 Taiji 实例，这是诚实的最小边界，不虚构尚未实现的多器官能力。

## 2. 当前可执行合同

`seed.model.Seed` 明确委托 `observe/act/settle_action/consolidate/learn_bytes/score_bytes/generate` 给 `substrate: Taiji`。Seed checkpoint 使用 `format=seed-native-v1`，内部嵌套完整 Taiji checkpoint；裸 Taiji checkpoint 不能冒充 Seed checkpoint。

依赖方向由测试强制：

```text
seed  ──public API──>  taiji
  X                       X
  └──── neuroplex <───────┘
```

- `seed/` 可以导入 `taiji`，不得导入 `neuroplex` 或 `transformers`；
- `taiji/` 不得导入 `seed`、`neuroplex` 或 `transformers`；
- `neuroplex/` 不得反向导入 `seed`/`taiji`。

## 3. 云端架构吸收判定

云端 `origin/trae/agent-FCnvzE` 已合并进主线。PlayEngine 真实任务场、普通生成场记忆捕获和 continuous coaction 三项运行修复保留在 Legacy 对照中。新增的 `neuroplex/taiji.py` / `taiji_arch.py` 已清退，因为实际路径仍是 Q/K/V attention + RoPE + SwiGLU，并存在 field/lifecycle 每 forward 清零、STDP step 恒零、睡眠只统计不改权重、future-token 泄漏等结构问题。完整逐文件证据见归档 `TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md`。

## 4. Seed 面向 AGI 的增长接口

新增能力必须属于以下两类之一：

1. **Taiji 基底能力**：改变持久状态、局部学习、场记忆、事件计算、可塑拓扑或感觉—动作闭环；落在 `taiji/`，必须通过因果 lesion。
2. **Seed 模型能力**：增加器官、目标/价值系统、发展调度、群体协作、多模态身体或自我模型；落在 `seed/`，只能调用 Taiji 的公开合同。

禁止把 tokenizer、attention、TransformerBlock、teacher logits、外部 event K/V 表或 Python replay list 放进 Seed，再声称是 Taiji 能力。

## 5. 当前构建上限

M6 已证明场可内生选择 engram 并把 action→outcome 结构巩固进 fabric，但残留错误不是 replay 覆盖不足。`_diag_m6_margin.py locus 11 29 61` 的离线前置验证结果：

| seed | 原始正 margin | 去共模后的正 margin | gain≤1 最好结果 |
|---:|---:|---:|---:|
| 11 | 2/4 | 2/4 | 2/4 |
| 29 | 2/4 | 4/4 | 4/4 |
| 61 | 2/4 | 2/4 | 3/4 |

因此“给 trace 加自适应公共基线”被否决：它只修复一部分随机拓扑。seed 11/61 的错误已存在于去共模残差本身；失败 true row 虽读到峰值残差单元，但 16-contact 固定随机 fan-in 同时给 rival row 更强支持。当前上限是：

```text
非负 cortical trace
  × 固定随机稀疏解码支撑
  × 单层局部 delta
  → 部分 seed 的动作残差在线性读出前已不可分
```

这会限制 Seed 的可扩展记忆与组合学习：扩大单元数只能降低概率，不能给出结构保证。

机器可读结果：`reports/taiji_m6_locus_20260822.json`。

## 6. 当前唯一下一步

先实现一个**离线 signed-opponent basis 反证器**，不改 Taiji 状态和 checkpoint：把每个区域的活动表示成匹配的 ON/OFF（或正/负误差）双通道，令每个原坐标对读出贡献严格零和；同时让每个目标 decoder row 对每个 opponent pair 至少有一个可学习接触。必须在完整 12-seed M6 面板上达到每 seed 4/4 正 margin，且 N5–N11/M5–M6 不退化，才允许把该表示写入 `fabric.step`。若离线仍失败，继续修改支撑生成规则，不升级运行态。
