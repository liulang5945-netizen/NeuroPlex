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
- 原生链：raw-byte sensor → hierarchical predictive fabric ↔ distributed episodic field → 全皮层覆盖稀疏感受器组 → byte motor → action feedback。
- 原生学习：区域预测误差、递归状态误差、运动结果误差和情景 cue→event/readout 的真实边局部 delta；无 optimizer/BPTT。
- 当前可复现实验：62,529 active learned parameters，byte-cycle accuracy `0 → 94.12%`；N7/N8 上下文与 trace 因果门槛通过；N9 自反馈 128/128；N10 真实按边等价；N11 主动环境末 40 次成功率 `100%`，随机 `50%`，action-lesion `57.5%`；M5 八条 one-shot 情景 action recall `87.5%`，同宽 trace-only 与 recurrent lesion 均 `25%`。
- 旧 `neuroplex.taiji` K/V 原型及 T4/T5 活动文件已删除；Git 历史仍可恢复。
- 现有 9 个 Transformer 成员（含 5 个对话成员）未被改写，只作为离线对照。

## 当前唯一下一步

进入 **M6 内生 replay 与巩固**：由 Taiji 场内 novelty/value 信号选择真实 engram，重激活同一 predictive fabric，通过现有局部误差规则把可迁移结构写入皮层突触；随后切除情景 readout，能力仍须保留。禁止外部 replay 清单或离线 teacher。

## 归档

`archive/` 保存旧架构、审计、实施历史和参考资料。旧 Taiji-0 补丁原型的废止说明见 [TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md](archive/architecture/TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md)。归档中的下一步不再有效。
