# Taiji 计划与架构入口

本项目的目标系统是独立的 **Taiji 原生计算架构**。`neuroplex/` 只保留为历史 Transformer 基线，不再承载 Taiji、不再决定新架构接口，也不会在 Taiji forward 中被调用。

## 当前权威文档

| 文档 | 权威范围 |
|---|---|
| [TAIJI_SUBSTRATE_ARCHITECTURE.md](active/TAIJI_SUBSTRATE_ARCHITECTURE.md) | 完整算法：张量、状态方程、tick、局部学习、训练、生成、复杂度、代码映射和反证门槛 |
| [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) | 当前实现状态、实测结果和唯一下一步 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | “全面替代而非补丁”的不可回退决策与命名边界 |
| [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) | 冻结 Legacy NeuroPlex 的源码事实基线 |

其余 active 文档是记忆、自举、生物类比等专项参考；若与上述三份文件冲突，以 Taiji 算法规格和当前实现计划为准。

## 当前代码事实

- 正式包：顶层 `taiji/`；不导入 `neuroplex` 或 `transformers`。
- 原生链：raw-byte sensor → hierarchical predictive fabric → byte motor → action feedback。
- 原生学习：区域预测误差、递归状态误差、运动结果误差的局部 masked delta；无 optimizer/BPTT。
- 当前可复现实验：19,521 active parameters，byte-cycle accuracy `0 → 76.47%`，surprise 下降 `81.15%`，自由生成前四步正确。
- 旧 `neuroplex.taiji` K/V 原型及 T4/T5 活动文件已删除；Git 历史仍可恢复。
- 现有 9 个 Transformer 成员（含 5 个对话成员）未被改写，只作为离线对照。

## 当前唯一下一步

执行 **N7 二阶上下文反证**：相同当前 byte 在不同历史下必须预测不同后继，并用 trace lesion 证明差异来自 Taiji 持久状态，而不是一阶字节频率。失败时只修改状态方程，不扩大规模或训练数据。

## 归档

`archive/` 保存旧架构、审计、实施历史和参考资料。旧 Taiji-0 补丁原型的废止说明见 [TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md](archive/architecture/TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md)。归档中的下一步不再有效。
