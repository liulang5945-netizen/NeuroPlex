# NeuroPlex 计划与架构入口

本项目是 **NeuroPlex**。当前目标是用 **Taiji 原生计算基底** 替代底层 Transformer：Taiji 承担全部计算职责（输入表示、时间状态、上下文、学习、输出、生成、checkpoint），而 `neuroplex/` 作为冻结的 Transformer 基线保留，其底层 Transformer（`neuroplex/layers.py::TransformerBlock`）正是被替代的对象。Taiji forward 不调用 `neuroplex/` 的任何代码。

命名口径（NeuroPlex / Taiji / Legacy NeuroPlex / 历史 `taiji.*` 别名）见 [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) §0 规范词表。

## 当前权威文档

| 文档 | 权威范围 |
|---|---|
| [TAIJI_SUBSTRATE_ARCHITECTURE.md](active/TAIJI_SUBSTRATE_ARCHITECTURE.md) | 完整算法：张量、状态方程、tick、局部学习、训练、生成、复杂度、代码映射和反证门槛 |
| [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) | 当前实现状态、实测结果和唯一下一步 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 规范词表、“全面替代而非补丁”的不可回退决策与命名边界 |
| [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) | 冻结 Legacy NeuroPlex 的源码事实基线 |

其余 active 文档是记忆、自举、生物类比等专项参考；若与上述四份文件冲突，以 Taiji 算法规格和当前实现计划为准。

## 当前代码事实

- 正式基底包：顶层 `taiji/`；不导入 `neuroplex` 或 `transformers`。
- 被替代的 Transformer 底层：`neuroplex/layers.py::TransformerBlock`，live 消费点 3 处（`neuroplex/resonance/neuron.py:25`、`scripts/training/train_tinystories.py:26`、`scripts/training/train_tinystories_field.py:32`），由 `tests/taiji_native/test_naming_boundary_contract.py` 强制封闭。
- 原生链：raw-byte sensor → hierarchical predictive fabric ↔ distributed episodic field → 全皮层覆盖稀疏感受器组 → byte motor → action feedback。
- 原生学习：区域预测误差、递归状态误差、运动结果误差和情景 cue→event/readout 的真实边局部 delta；无 optimizer/BPTT。
- 当前可复现实验：62,529 active learned parameters，byte-cycle accuracy `0 → 94.12%`；N7/N8 上下文与 trace 因果门槛通过；N9 自反馈 128/128；N10 真实按边等价；N11 主动环境末 40 次成功率 `100%`，随机 `50%`，action-lesion `57.5%`；M5 八条 one-shot 情景 action recall `87.5%`，同宽 trace-only 与 recurrent lesion 均 `25%`；M6 内生 replay/巩固 5/5 seed 通过。
- 旧 `neuroplex.taiji` K/V 原型及 T4/T5 活动文件已删除；Git 历史仍可恢复。
- 现有 9 个 Transformer 成员（含 5 个对话成员）未被改写，只作为离线对照。
- `scripts/archive/` 内 `from taiji.<legacy>` 是历史别名（含义＝`neuroplex`），已确认不重写；判定见 `scripts/archive/README.md`。

## 当前唯一下一步

修复 M6 replay 选择覆盖不均：`priority` 随 familiarity 单调上升形成正反馈，导致低占比 engram 被结构性饿死。验收与禁止项见 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) §6。

## 归档

`archive/` 保存旧架构、审计、实施历史和参考资料。旧 Taiji-0 补丁原型的废止说明见 [TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md](archive/architecture/TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md)。归档中的下一步不再有效。
