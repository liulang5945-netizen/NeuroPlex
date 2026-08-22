# Taiji Transformer 壳原型源码审计（2026-08-22）

> 来源：云端分支 `origin/trae/agent-FCnvzE`，提交 `596541c`、`13ee6ae`、`87d1067`、`dce4cca`。
>
> 判定：保留 Git 历史，不进入 Seed/Taiji 当前运行路径。有效的 PlayEngine、Cortex 场记忆捕获与 continuous coaction 修复已单独并入主线。

## 1. 实际执行核心

该分支新增的 `neuroplex/taiji.py` 并不是 Transformer 的完整替代。其 Yang 路径仍执行：

```text
Linear Q/K/V
  → RoPE
  → scaled_dot_product_attention / softmax(QKᵀ)
  → residual
  → SwiGLU FFN
```

Yin 路径是 field 作为单个或多个 K/V 的 cross-attention，并用复数对旋转 K/V。`TaijiBlock` 的目标被明确写成“与 TransformerBlock.forward 完全对齐、可直接替换”，无 field 时则退化成标准 causal TransformerBlock。它因此是一个 Transformer 兼容补丁，不是 Taiji 原生基底。

## 2. 已确认的结构上限

| 源码事实 | 直接后果 |
|---|---|
| token 路径仍是全序列 QKᵀ attention | 时间/显存仍随序列长度二次增长，仍受上下文窗口约束 |
| 每层为每个“神经元”复制一整套 attention + FFN | 参数量和计算量近似随 `layers × population × TransformerBlock` 增长 |
| `kv_cache` 仅保留签名，始终返回 `None` | 自回归推理不能获得声明的缓存收益 |
| learned absolute position embedding | 超过 `max_seq_len` 无定义，不是持续时间状态 |
| 第一层 field 从零开始；每次 layer forward 都 `reset_field()` | field 不是跨交互持续状态，单层模型的 Yin 输入恒为零 |
| 模型每次 forward 调 `reset_lifecycle()` | 不应期在跨交互前被清零，无法承担持续轮替 |
| 所有 firing 都以 `step=0` 记录 | STDP 的 `delta_t` 恒为零，实际 pair update 恒为零 |
| `sleep_consolidate(recent_fields)` 不使用 `recent_fields`，只累计数值 | 睡眠不修改任何连接；注释也承认“只统计” |
| `phase` 由 `.item()` 转成 Python float | 相位路径与可学习振荡器的梯度在接口处断开 |
| `inhibit_signal` 在群体层从未传入，`W_cond`/`field_vectors` 未使用 | E/I 与场条件有声明但没有完整运行线路 |
| 对整段序列 `mean(dim=1)` 后写 field，再反馈所有位置 | 后半 token 可影响前半 logits；验证脚本把未来泄漏标成“全局反馈特性” |

这些不是扩大参数或追加训练能够消除的问题；它们由计算图和状态生命周期决定。

## 3. 验证脚本为何没有发现

云端 `verify_taiji_architecture.py` 的 23 项检查以“不报错、形状正确、有梯度、输出不同”为主，没有验证机制的因果贡献：

- STDP 测试只要求 `pairs_updated >= 0`，所以零更新也通过；
- 睡眠测试不比较任何权重前后差异；
- 群体测试只要求随机初始化的单体/多体输出不同；
- 相位测试只要求梯度对象存在，不要求非零且能改变任务结果；
- causal 测试主动要求未来 token 改变过去 logits；
- 没有学习任务、自由运行、跨 episode、主动环境或 lesion 对照。

合并后，现有命名边界测试立即失败，因为 `verify_taiji_operator.py` 成为新的 `TransformerBlock` live 消费者。这是正确的方向守卫。

## 4. 吸收与淘汰

可吸收的是机制意图：调质、持续相位、稀疏选择、不应期、局部可塑性、场传导和睡眠巩固。它们必须落在顶层 `taiji/` 的持久状态、稀疏事件计算和局部更新上，不能继续以 attention/FFN 外壳表达。

被淘汰的是这五个 live 产物：

```text
neuroplex/taiji.py
neuroplex/taiji_arch.py
plans/active/TAIJI_OPERATOR_DESIGN.md
scripts/training/verify_taiji_operator.py
scripts/training/verify_taiji_architecture.py
```

当前边界是：**Seed 是模型/项目；Taiji 是 Seed 的原生计算基底；`neuroplex/` 只保留旧九成员 Transformer 基线。**
