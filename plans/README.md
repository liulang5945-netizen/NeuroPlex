# Taiji / NeuroPlex 计划与架构文档

本目录只做文档导航，不再把不同生命周期的计划平铺在一起。`Taiji` 是目标非
Transformer 计算底座；`NeuroPlex` 是当前产品、群体装配和迁移期兼容运行时。

## 当前有效文档

当前底层架构以 `active/TAIJI_SUBSTRATE_ARCHITECTURE.md` 为准；执行状态以
`active/BIO_INSPIRED_ARCHITECTURE_PLAN.md` 为准。现有 NeuroPlex 机制是否真正接入，
仍以 `active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md` 的源码审计和运行时证据为准。

| 文档 | 用途 | 权威范围 |
|---|---|---|
| [TAIJI_SUBSTRATE_ARCHITECTURE.md](active/TAIJI_SUBSTRATE_ARCHITECTURE.md) | 非 Transformer 底座的状态方程、运行时、学习与反证门槛 | **目标底层架构** |
| [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) | 当前总架构、状态、唯一下一步 | 当前执行主计划 |
| [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) | 按源码调用链核对现有生产、训练、睡眠、记忆和生命周期 | Legacy NeuroPlex 事实基线 |
| [AGI_FIELD_MEMORY_PLAN.md](active/AGI_FIELD_MEMORY_PLAN.md) | 现有场记忆缺口与 Taiji 原生记忆迁移 | 记忆子计划 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 群体神经元网络的架构决策和术语 | 架构身份与公共叙事 |
| [BOOTSTRAP_CRITERIA.md](active/BOOTSTRAP_CRITERIA.md) | 旧底座自举实验及 Taiji 后续迁移判据 | 历史证据；不能自动外推到 Taiji |
| [DESIGN_PRINCIPLES.md](active/DESIGN_PRINCIPLES.md) | 生物启发设计原则及其实施历史 | 历史参考；新设计服从 Taiji 规范 |
| [TAIJI_VS_HUMAN_BRAIN_COMPARISON.md](active/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md) | 当前 NeuroPlex、Taiji 目标与生物系统的边界类比 | 概念解释，不作实现证明 |

## 当前唯一执行入口

当前不要从归档文档、旧 D1 实验或单独的验收判据推导下一步。执行入口是：

1. 阅读 [TAIJI_SUBSTRATE_ARCHITECTURE.md](active/TAIJI_SUBSTRATE_ARCHITECTURE.md) 的实现合同。
2. 阅读 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) 的“唯一下一步”。
3. 用 [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) 核对迁移前代码事实。
4. 完成一个步骤后，回写底座规范与主计划，再进入下一步。

当前唯一下一步：在已通过状态合同的 Taiji-0 上实现 T4 一次性局部关联学习；只更新活动细胞 fast memory，不使用全局 optimizer、不改慢权重、不接生产。

## 历史归档

归档文件是证据和上下文，不是当前执行计划。按内容分为：

- [archive/architecture](archive/architecture/)：旧架构总案、身体—生命—大脑集成案、Hub 草案。
- [archive/audits](archive/audits/)：阶段性审计、妥协点审查和旧缺口判断。
- [archive/history](archive/history/)：项目事件、机制实验、对话训练时间线。
- [archive/implementation](archive/implementation/)：已完成或被替代的机制实施/修复方案。
- [archive/reference](archive/reference/)：训练方法和外部经验参考。

归档文档可以追溯“当时为什么这样设计”，但其中的状态、下一步和路径不自动代表当前项目状态。
