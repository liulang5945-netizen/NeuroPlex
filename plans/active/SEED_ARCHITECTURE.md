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

M6 已证明场可内生选择 engram 并把 action→outcome 结构巩固进 fabric，但残留错误不是 replay 覆盖不足。`scripts/archive/native_v6/_diag_m6_margin.py locus 11 29 61` 的历史离线验证结果：

| seed | 原始正 margin | 去共模后的正 margin | gain≤1 最好结果 |
|---:|---:|---:|---:|
| 11 | 2/4 | 2/4 | 2/4 |
| 29 | 2/4 | 4/4 | 4/4 |
| 61 | 2/4 | 2/4 | 3/4 |

因此“给 trace 加自适应公共基线”被否决：它只修复一部分随机拓扑。进一步的完整 12-seed 离线镜像已经把上限拆成两个独立因素：

- K64 signed shared-support 单独达到 11/12；它消除了固定 16-contact 的随机盲区，但 seed 11 仍因 `2→!` 只得到 6 次 replay、其它模式得到 23–32 次而停在 3/4；
- 保留所有内生 replay target/rate/count、只把写 basis 替换成清醒 probe，seed 11 仍是 3/4，排除“睡眠写入与清醒读取错位”为首因；
- 把四个模式的自然写入量截到相同下限后，K32/K64 在 12/12 seed 全部 4/4；
- 给 replay winner 神经元配置睡眠 bout 内的局部可塑性资源，winner 每次获胜后资源乘 retention，K64 在 retention `0.5/0.7/0.8/0.9` 下均为 12/12 seed × 4/4，旋转内容 lesion 全部 0/4。retention `0.9` 的跨面板最小/平均 margin 最大：`0.0004383 / 0.0043703`。

当前上限已经从单一“不可分”订正为：

```text
非负 cortical trace
  × 固定随机稀疏解码支撑        → 几何盲区
  × 无 bout 级资源的 replay winner → 多数模式覆盖少数模式
  → 扩大单元数不能给出巩固可分性的结构保证
```

这曾限制 Seed 的可扩展记忆与组合学习：扩大单元数只能降低概率，不能给出结构保证。

逐 cortical channel 的熟悉度资源、现有 fast fatigue 的 decay/gain 扫描、睡眠期慢速 memory-unit threshold offset 都未救回 seed 11，不能进入运行态。有效机制的粒度必须落在**内生 replay winner 神经元**，而不是外部 event ID、Python replay list 或每 engram 配额。

Native v7 已把反证结果落实为双时间尺度 cortical path：清醒快速 decoder 仍为固定稀疏预测通路；零初始化的慢速 consolidation decoder 读取 waking baseline 周围的 signed residual，每个 row 共享完整支撑；winner resource 为 bout-local，retention `0.9`；慢通路使用独立 RNG 子流，新增器官不会改变 transitions/motor/memory 的既有拓扑。运行态 M6 达到 12/12 seed × 4/4，所有 control 为 25%，mean gain `+0.75`。N5–N11/M5 与 83 项全仓测试均通过。

机器可读结果：`reports/taiji_m6_locus_20260822.json`、`reports/taiji_signed_opponent_20260822.json`、`reports/taiji_m6_seed_panel_v7_20260822.json`。

## 6. 当前唯一下一步

定义并实现 M7 cue-conditioned consolidation 反证：内生场必须重建完整 `cue → action → outcome` 顺序，慢速 cortical path 在切除 episodic action/cortical readout 后仍能依据 cue 选择对应 action，并在 action 后预测 outcome。禁止把 cue/action/outcome 列表交给 `consolidate`，禁止 teacher action、event slot 或 memory-weight 复制；control、内容 lesion 和顺序 lesion 必须回落。先固定可执行基准与状态流，再改 replay burst。
