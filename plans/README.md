# Seed 计划与架构入口

本项目和模型是 **Seed**。**Taiji** 是 Seed 的原生计算基底，承担输入表示、时间状态、上下文、学习、输出、生成和 substrate checkpoint；`seed/` 拥有模型级组合与身份，`neuroplex/` 只作为冻结的 Transformer 基线保留。Seed 只通过 Taiji 公共 API 组合基底，Taiji forward 不调用 `seed/`、`neuroplex/` 或 `transformers`。

命名口径（Seed / Taiji / Legacy NeuroPlex / 历史 `taiji.*` 别名）见 [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) §0 规范词表。

## 当前权威文档

| 文档 | 权威范围 |
|---|---|
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed 模型与 Taiji 基底的所有权、云端实现吸收判定和当前构建上限 |
| [TAIJI_SUBSTRATE_ARCHITECTURE.md](active/TAIJI_SUBSTRATE_ARCHITECTURE.md) | 完整算法：张量、状态方程、tick、局部学习、训练、生成、复杂度、代码映射和反证门槛 |
| [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) | 当前实现状态、实测结果和唯一下一步 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 规范词表、“全面替代而非补丁”的不可回退决策与命名边界 |
| [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) | 冻结 Legacy NeuroPlex 的源码事实基线 |

其余 active 文档是记忆、自举、生物类比等专项参考；若冲突，以 Seed 总架构、Taiji 算法规格和当前实现计划为准。

## 当前代码事实

- 正式模型包：顶层 `seed/`；当前明确组合一个 `Taiji` substrate，并拥有 `seed-native-v1` checkpoint envelope。
- 正式基底包：顶层 `taiji/`；不导入 `seed`、`neuroplex` 或 `transformers`。
- 被替代的 Transformer 底层：`neuroplex/layers.py::TransformerBlock`，live 消费点 3 处（`neuroplex/resonance/neuron.py:25`、`scripts/training/train_tinystories.py:26`、`scripts/training/train_tinystories_field.py:32`），由 `tests/taiji_native/test_naming_boundary_contract.py` 强制封闭。
- 原生链：raw-byte sensor → hierarchical predictive fabric ↔ distributed episodic field → 全皮层覆盖稀疏感受器组 → byte motor → action feedback。
- 原生学习：区域预测误差、递归状态误差、运动结果误差和情景 cue→event/readout 的真实边局部 delta；无 optimizer/BPTT。
- 当前可复现实验：Native v7 为 83,841 active learned parameters，byte-cycle accuracy `0 → 94.12%`；N7/N8 上下文与 trace 因果门槛通过；N9 自反馈 128/128；N10 真实按边等价；N11 主动环境末 40 次成功率 `100%`，随机 `50%`，action-lesion `57.5%`；M5 one-shot action recall `87.5%`；M6 12/12 seed 均为 4/4，control 均为 25%，mean gain `+0.75`。
- 旧 `neuroplex.taiji` K/V 原型及 T4/T5 活动文件已删除；Git 历史仍可恢复。
- 现有 9 个 Transformer 成员（含 5 个对话成员）未被改写，只作为离线对照。
- `scripts/archive/` 内 `from taiji.<legacy>` 是历史别名（含义＝`neuroplex`），已确认不重写；判定见 `scripts/archive/README.md`。

## 当前唯一下一步

M7 FAIL 基线已建立：action→outcome `100%`，cue 慢通路 `50%`/零 margin，行为 `62.5%` 且不优于 control。当前唯一下一步是让 accepted replay 先以内生 `cortical_projection` 重建 cue 并局部写入 cue→action，再执行现有 action→outcome burst。判据见 [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) §6。

## 归档

`archive/` 保存旧架构、审计、实施历史和参考资料。旧 Taiji-0 补丁原型见 [TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md](archive/architecture/TAIJI0_PATCH_PROTOTYPE_RETIRED_20260821.md)，本次云端 Transformer 壳审计见 [TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md](archive/architecture/TAIJI_TRANSFORMER_SHELL_AUDIT_20260822.md)。归档中的下一步不再有效。
