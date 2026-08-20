# NeuroPlex 计划与架构文档

本目录只做文档导航，不再把不同生命周期的计划平铺在一起。

## 当前有效文档

当前执行优先级以 `active/BIO_INSPIRED_ARCHITECTURE_PLAN.md` 为准；机制是否真正接入，
以 `active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md` 的源码审计和运行时证据为准。

| 文档 | 用途 | 权威范围 |
|---|---|---|
| [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) | 当前总架构、状态、唯一下一步 | 当前执行主计划 |
| [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) | 按源码调用链核对生产、训练、睡眠、记忆和生命周期 | 机制接入事实 |
| [AGI_FIELD_MEMORY_PLAN.md](active/AGI_FIELD_MEMORY_PLAN.md) | 场记忆目标、条目契约和闭环路线 | 场记忆子计划 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 群体神经元网络的架构决策和术语 | 架构身份与公共叙事 |
| [BOOTSTRAP_CRITERIA.md](active/BOOTSTRAP_CRITERIA.md) | 自举门槛、判据和实验证据 | 验收判据；不覆盖当前执行顺序 |
| [DESIGN_PRINCIPLES.md](active/DESIGN_PRINCIPLES.md) | 生物启发设计原则及其实施历史 | 设计约束与历史参考 |
| [TAIJI_VS_HUMAN_BRAIN_COMPARISON.md](active/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md) | 工程机制与生物系统的边界类比 | 概念解释，不作实现证明 |

## 当前唯一执行入口

当前不要从归档文档或单独的验收判据推导下一步。执行入口是：

1. 阅读 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) 的“唯一下一步”。
2. 用 [NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md](active/NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md) 核对代码证据。
3. 完成一个步骤后，回写这两个文档，再决定是否进入下一步。

当前唯一下一步：修复 PlayEngine 的实际运行契约，并用回归 trace 确认高共振 replay 是否真实产生。

## 历史归档

归档文件是证据和上下文，不是当前执行计划。按内容分为：

- [archive/architecture](archive/architecture/)：旧架构总案、身体—生命—大脑集成案、Hub 草案。
- [archive/audits](archive/audits/)：阶段性审计、妥协点审查和旧缺口判断。
- [archive/history](archive/history/)：项目事件、机制实验、对话训练时间线。
- [archive/implementation](archive/implementation/)：已完成或被替代的机制实施/修复方案。
- [archive/reference](archive/reference/)：训练方法和外部经验参考。

归档文档可以追溯“当时为什么这样设计”，但其中的状态、下一步和路径不自动代表当前项目状态。
