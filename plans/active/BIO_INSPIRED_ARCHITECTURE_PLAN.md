# Taiji 当前实施计划

> 本文件只记录当前源码状态与唯一下一步。历史 NeuroPlex/D1/PlayEngine 结论不再混入当前执行线。

## 1. 当前架构

```text
ByteSensor
  → reciprocal predictive region 0
  → recurrent predictive region 1..R
  → normalized multi-region motor context
  → ByteMotor
  → emitted byte loops back to ByteSensor
```

学习与 forward 同时发生：decoder 学习下层预测，transition 学习局部下一状态，motor 学习真实后继。所有权重都受固定 fan-in mask 约束，不使用 autograd 或全局 optimizer。

## 2. 实现地图

| 代码 | 已实现职责 | 状态 |
|---|---|---|
| `taiji/config.py` | 完整形状、动力学、学习率、边界 | ✅ |
| `taiji/sparse.py` | 稀疏结构、前向/反投影、局部更新、权重约束 | ✅ |
| `taiji/state.py` | 区域/整机持久状态与结果合同 | ✅ |
| `taiji/organs.py` | 原始 byte 感受器与唯一 byte 运动器官 | ✅ |
| `taiji/fabric.py` | 分层预测误差、递归状态、抑制、稳态与局部学习 | ✅ |
| `taiji/model.py` | observe、learn、score、generate、checkpoint | ✅ |
| `neuroplex/legacy_checkpoint.py` | 旧 pickle 的作用域兼容加载，不污染原生命名空间 | ✅ |
| `tests/taiji_native/` | 原生架构与学习合同 | ✅ 7 passed |
| `verify_taiji_native_v1.py` | 独立端到端可复现基准 | ✅ PASS |

## 3. Native v1 实测

固定 seed `7`、区域 `[64,48]`、fan-in `16`、motor fan-in `48`：

| 指标 | 结果 |
|---|---:|
| active parameters | 19,521 |
| dense tensor storage | 54,961 |
| structural sparsity | 64.48% |
| byte-cycle accuracy | 0% → 76.47% |
| mean surprise | 5.5622 → 1.0484 |
| surprise reduction | 81.15% |
| free generation | `a → bcdaccbd`，前四步正确，之后漂移 |
| changed parameter tensors | 6/6 |
| checkpoint exact next step | PASS |
| Transformer/NeuroPlex dependency | 0 |

报告：`reports/taiji_native_v1_20260821.json`。

## 4. 已废止方向

- `neuroplex.taiji` 作为旧运行时内部 cell；
- 全局 priority/top-k 调度三个 cell；
- 每个活动 cell 复制精确 cue/value memory；
- 活动输出向量平均；
- 通过 event gateway 接回 Cortex 后再称为 Taiji；
- 继续调 T5-bis 活动稳态来修补上述原型；
- 用蒸馏或小 Transformer 为 Taiji 提供核心能力。

旧代码和报告已经从工作树删除，Git 历史可恢复。

## 5. 当前限制

Native v1 还不能宣称语言智能：

- byte-cycle teacher-forced accuracy 只有 76.47%；
- 自由生成四步后会漂移；
- 尚未证明相同当前输入在不同历史下可产生不同正确动作；
- masked dense tensor 尚未带来真实 sparse FLOPs；
- 没有情景记忆、内部想象、环境行动学习和睡眠巩固。

这些缺口不能通过改名、扩大参数或下载更多文本掩盖。

## 6. Legacy 隔离

现有 5 个 dialogue + 4 个 general Transformer 成员不删除、不修改，继续作为离线成本/质量基线。当前工作区中的 D1 修改由另一条 Legacy 工作保留，不能暂存到 Taiji 提交。

## 7. 当前唯一下一步

实现 **N7 二阶上下文反证基准**：设计两个共享当前 byte、但因前序历史不同而要求相反后继的循环；同时比较完整 Taiji、清零 temporal trace 的 lesion、只看当前 byte 的一阶统计基线。

通过线：完整 Taiji accuracy ≥ 75%，且至少比两个 lesion/baseline 高 20 个百分点。若失败，下一次修改只能针对区域状态方程或局部转移学习，不增加区域尺寸、epoch 或外部记忆。
