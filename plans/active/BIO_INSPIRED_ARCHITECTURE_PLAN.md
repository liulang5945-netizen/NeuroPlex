# Taiji 当前实施计划

> 本文件只记录当前源码状态、已复现实证与唯一下一步。历史 NeuroPlex/D1/PlayEngine 结论不混入当前执行线。

## 1. 当前架构

```text
ByteSensor
  → reciprocal predictive region 0..R
  → [all current activities; all slow traces]
  → balanced SparseReceptorBank
  → shared K-channel motor evidence
  → ByteMotor
  → emitted byte loops back to ByteSensor
```

区域 decoder 学习下层预测，transition 学习局部下一状态，motor 在下一真实事件到达时学习动作结果。所有更新均在 `torch.no_grad()` 内执行，不使用 autograd、optimizer、BPTT、attention、tokenizer、教师模型或蒸馏。

## 2. 实现地图

| 代码 | 已实现职责 | 状态 |
|---|---|---|
| `taiji/config.py` | 形状、动力学、学习率、稳定上界与上下文维数 | ✅ |
| `taiji/sparse.py` | 固定 fan-in、reciprocal 投影、局部 delta 与权重约束 | ✅ |
| `taiji/state.py` | 区域/整机持久状态与原子 checkpoint 状态 | ✅ |
| `taiji/organs.py` | raw-byte 感觉器官、全坐标覆盖的稀疏感受器组、唯一 byte 运动器官 | ✅ |
| `taiji/fabric.py` | 分层预测误差、递归状态、抑制、稳态与区域局部学习 | ✅ |
| `taiji/model.py` | observe、learn、score、generate、Native v2 checkpoint | ✅ |
| `tests/taiji_native/` | 独立性、局部性、状态、感受器覆盖、N5/N6/N7 | ✅ 9 passed |
| `verify_taiji_native_v2.py` | 独立端到端基准 | ✅ PASS |
| `verify_taiji_n7_context.py` | 二阶歧义与因果切除基准 | ✅ PASS |

## 3. Native v2 实测

固定 seed `7`、区域 `[64,48]`、区域 fan-in `16`、运动感受器 `48`：

| 指标 | 结果 |
|---|---:|
| active learned parameters | 19,521 |
| fixed receptor edges | 224（每个皮层坐标恰好一条） |
| dense learned tensor storage | 38,513 |
| learned structural sparsity | 49.31% |
| byte-cycle accuracy | 0% → 94.12% |
| mean surprise | 5.4041 → 0.1090 |
| surprise reduction | 97.98% |
| free generation | `a → bcdabcda`，8 步全部正确 |
| checkpoint exact next step | PASS |
| Transformer/NeuroPlex runtime dependency | 0 |

N7 流 `axbcxd × 4` 中，当前符号同为 `x`，历史分别要求后继 `b`/`d`：

| 对照 | 歧义位置 accuracy |
|---|---:|
| 完整 Taiji 状态 | 100% |
| 只看当前 byte 的一阶基线 | 50% |
| 每 tick 清空全部动态状态 | 50% |
| 只清空 slow trace | 100% |

结论只到这里：持久动态状态已具有二阶上下文能力，但 N7 的短间隔主要由 membrane/activity 承担，尚未证明 slow trace 或长期场记忆的独立因果作用。

报告：`reports/taiji_native_v2_20260821.json`、`reports/taiji_n7_context_20260821.json`。

## 4. 本轮删除的错误机制

Native v2 不再让 257 个动作各自随机抽取不同皮层坐标，也不让所有动作共同只看 224 维中的同一随机 48 维。前者让动作 evidence 不可比较，后者会永久丢弃 176 个坐标。正式实现使用平衡、固定极性的稀疏感受器组，把全部 224 维折叠为 48 个公共证据通道；动作参数数目不增加。

旧 `neuroplex.taiji` K/V cell、全局 top-k、输出平均、event gateway 回接 Cortex、蒸馏底座和小 Transformer 身份继续保持废止。

## 5. 当前限制

- 当前只证明小型 byte 流学习和短程二阶上下文，不代表语言理解；
- slow trace 尚无独立因果实证，更没有可检索情景/自传记忆；
- PyTorch 区域矩阵仍以 masked dense tensor 执行，active edge 数不等于真实 FLOPs；
- 尚无内部想象、奖励调制、睡眠巩固、多感官器官和真实环境行动学习；
- 现有 5 个 dialogue + 4 个 general Transformer 成员只作为冻结离线基线，不进入 Taiji forward。

## 6. 当前唯一下一步

执行 **N8 延迟上下文/trace 因果反证**：在线索与共同 probe 之间插入相同干扰序列，预先固定 delay 与通过线；比较完整状态、trace-only lesion、全状态 lesion 和一阶基线。目标不是再提高 N5，而是判断 slow trace 是否在快 activity 被干扰后仍承担可用历史。如果失败，只根据状态轨迹修正时间尺度或局部信用分配，不扩大区域、epoch 或数据。

## 7. 附录：已废止的 D1 长程稳定性档案（NeuroPlex/PlayEngine）

> 完整判定标准与所有方案讨论见 [BOOTSTRAP_CRITERIA.md](BOOTSTRAP_CRITERIA.md) 第 4 节。本节只记录 v9 修复结论与对 Taiji 主线的隐含信号。

- **D1 系列目标**：1000 步压力测试下，3 组（dialogue/knowledge/unfamiliar）std ratio ≥ pre × 0.90
- **v3/v4/v5/v6/v7/v8 演化**：见 BOOTSTRAP_CRITERIA.md
- **v9（2026-08-21，方案 N 落地）**：修复 `pre_lora_l2_baseline=0.0` 的理论缺陷（`LoRA/0` 无意义使 ceiling 永远不触发），改为前 50 步 LoRA L2 均值
  - 结果 2/5：dialogue 1.0854 ✅（首次完整超过 v5 0.9127 维度）；knowledge 0.8177 ❌；unfamiliar 0.8190 ❌
  - 0 崩溃，26.3 min
  - **关键确认**：post_lora_l2 = **10.96**（v5/v7/v8 都是 11.84，**v9 < v8 -0.88**），ceiling 机制真的开始工作
  - **但 k/u 与 v5/v7/v8 字面相近**：DECAY 0.85 仍是决定性因素，ceiling 仅在 LoRA 终值上显出差异
  - 报告：`reports/play_engine_d1_fix_v9_baseline_fix_20260821.json`
- **对 Taiji 主线的隐含信号**：D1 修复无法靠调整 PlayEngine sleep 参数闭环——`std ratio 0.82-0.91` 已成天花板；要让 D1 完整通过，要么换架构路线（去 LoRA ceiling 转别机制），要么承认 D 系列不是当前瓶颈，转向 D2 长程记忆检索
- **Taiji 不复用任何 PlayEngine 代码**：Taiji 是顶层原生 TPF，无 LoRA、无 sleep、无 play engine；D1 修复经验仅作为"持续学习系统需要衰减 + 抑制上限 + 软起点"的设计直觉
