# NeuroPlex Active Architecture Plan

> **状态**：当前活跃计划 · 2026-08-18
>
> 本文件只描述当前项目状态和下一步，不承载旧实验的叙事。机制历史、项目事件、训练参考和历史审计统一见 `archive/`。

## 1. 架构决策

NeuroPlex 采用**稀疏路由群体共振网络**。系统的能力单位是神经元群体，不是单一中心模型。

```text
输入
  ↓
共享感知对齐 + 域 tokenizer
  ↓
神经元群体：独立 Transformer / 领域 / 角色 / 亚型
  ↕  field_write / field_read + peer channels
共振场：共享通信状态
  ↓
稀疏路由、质量门控、记忆、睡眠、突触可塑性
  ↓
群体输出 + 生命周期决策
```

### 1.1 方案比较

| 方案 | 独立成长 | 稀疏运行 | 可观察性 | 生命周期 | 当前结论 |
|---|---:|---:|---:|---:|---|
| 单体模型 + adapter | 低 | 中 | 低 | 低 | 退居兼容层 |
| 全量神经元 ensemble | 高 | 低 | 高 | 中 | 作为基线 |
| MoE 专家分片 | 中 | 高 | 中 | 低 | 借鉴路由，不作为身份 |
| **稀疏群体共振** | **高** | **高** | **高** | **高** | **当前主线** |
| 分层皮层图 | 高 | 高 | 高 | 高 | 群体基线稳定后的扩展 |

完整理由和迁移边界见 [ARCHITECTURE_DIRECTION_2026_08.md](ARCHITECTURE_DIRECTION_2026_08.md)。

### 1.2 术语约束

新代码、文档、日志和训练数据统一使用：

- `neuron population` / 神经元群体；
- `peer coordination` / 同伴协调；
- `experience replay` / 经验回放；
- `population growth` / 群体成长；
- `relay neuron` / 中继神经元（可选）。

教师—学生迁移、整体模型升级等词只允许出现在兼容层说明或历史记录中，不得出现在产品身份、快速开始和当前架构主叙事中。

## 2. 当前实现地图

| 平面 | 代码 | 当前状态 |
|---|---|---|
| 感知对齐 | `neuroplex/resonance/translator.py`、`domains/` | ✅ 域 tokenizer、共享输入对齐 |
| 神经元 | `neuroplex/resonance/neuron.py` | ✅ 独立 Transformer、field read/write、域输出头 |
| 共振编排 | `neuroplex/resonance/ensemble.py` | ✅ 多轮、质量/置信度门控、连续共振、融合 |
| 场通信 | `neuroplex/resonance/field.py` | ✅ 共享场状态、贡献记录、评分 |
| 拓扑协作 | `topology.py`、`tribal.py`、side channels | ✅ 同伴连接、跨规格投影、共激活追踪 |
| 路由 | `ensemble.py`、实例级路由模块 | ✅ 任务/实例级稀疏选择 |
| 记忆 | `field_memory.py`、`sleep_engine.py` | ✅ 写门控、锚点检索、条件化生成、跨重启保存 |
| 学习 | `life/`、`training/` | ✅ 独立训练、协作训练、睡眠回放、LoRA 巩固 |
| 生命周期 | `lifecycle.py`、`integrate_engine.py` | ✅ 静默、成熟、验证、固化、隔离、复活、凋亡、新生 |
| 产品接口 | `api/`、`frontend/`、`desktop/` | ✅ Cortex/API/客户端已接入；旧升级 URL 仅保留兼容响应 |

可选的 expert neuron 是群体中的中继/锚点成员，不是强制中心。新神经元优先从领域数据、记忆经验和同伴协调中成长。

## 3. 训练与运行闭环

```text
领域数据
  → 单神经元训练
  → 同伴连接 / 跨规格协作
  → 路由和质量评估
  → 对话与工具交互
  → 场记忆写入
  → 睡眠回放和经验巩固
  → 新增 / 专业化 / 隔离 / 修剪神经元
```

推荐入口：

```text
scripts/training/finetune_neuron_dialogue.py
scripts/training/train_neurons_from_scratch.py
scripts/training/train_cross_domain_collab.py
scripts/training/train_hub_neuron.py       # optional relay member
scripts/training/verify_*.py
```

## 4. 当前验收状态

- `python -m compileall -q api neuroplex scripts/data_prep scripts/training`：通过。
- `tests/`：25 项核心回归测试通过，覆盖对话格式契约和共振 side-channel。
- `verify_c26_*` / `verify_c27_*`：覆盖场记忆、睡眠巩固、跨频耦合、实例路由和自组织新生。
- 生产路径默认加载 Cortex 群体，并由 API/客户端使用群体状态。
- 默认 tokenizer 已切换到 `neuroplex/domains/general/sp_general.model`；旧 checkpoint 路径不再是主加载路径。
- 旧教师对齐模块未从仓库删除，以避免历史 checkpoint 和实验脚本失效；它不再从 `neuroplex.resonance` 顶层导出，也不在 README/quick start 中出现。

## 5. 本轮已完成

1. 新增架构决策记录，明确采用稀疏路由群体共振网络。
2. README、CONTRIBUTING、CODE_WIKI 和本文件统一为群体神经元口径。
3. 身份/开发者训练数据移除旧基座模型、整体变大和递归迁移故事，改为群体成长叙事。
4. 清理代码 Wiki 中的旧包路径、绝对本地文件 URI 和过期模块索引。
5. 旧模型升级 API 改为兼容状态接口；整体模型升级动作明确返回退役状态。
6. 将 dense 转换器、旧域语料补全器、旧教师对齐训练器和损坏的生命体数据生成器移入 `scripts/archive/`，不再出现在主训练入口。
7. 删除 `ModelConfig` 中旧的整体模型尺寸/Qwen 预设，保留 token 合约和 checkpoint 兼容字段。
8. 将审计、实验历史和旧训练参考移入 `plans/archive/`，活跃 plans 只保留当前架构与验收口径。

## 6. 唯一下一步

完成提交后，对发布面执行一次秘密信息与绝对路径扫描；只保留法律、贡献、第三方数据来源和明确的历史/兼容记录。
