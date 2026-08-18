# NeuroPlex Active Architecture Plan

> **状态**：当前活跃计划 · 2026-08-19
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
- `python -m pytest tests -q`：30 项核心回归测试通过，覆盖对话格式契约、共振 side-channel、API 健康检查、最小群体基线和跨域评估词表契约。
- `python -m pip install -e ".[dev]" --no-deps`：可完成 editable 安装。
- 干净启动烟测通过：空神经元目录可以启动 Cortex，并明确进入 fallback；域 tokenizer 由 TokenizerHub 注册。
- API 烟测通过：健康检查、架构能力接口返回 200；旧整体升级入口返回 410 退役响应。
- `verify_c26_*` / `verify_c27_*`：已有机制级验证，覆盖场记忆、睡眠巩固、跨频耦合、实例路由和自组织新生。
- 生产路径默认加载 Cortex 群体，并由 API/客户端使用群体状态。
- 默认 tokenizer 已切换到 `neuroplex/domains/general/sp_general.model`；旧 checkpoint 路径不再是主加载路径。
- 旧教师对齐模块未从仓库删除，以避免历史 checkpoint 和实验脚本失效；它不再从 `neuroplex.resonance` 顶层导出，也不在 README/quick start 中出现。

### 4.1 真实 checkpoint 的 P1 短周期验收（2026-08-19）

本轮只做小样本、可中断的匹配评估，不把旧长训练重新包装成产品能力。

| checkpoint | code | math | zh | 均值 |
|---|---:|---:|---:|---:|
| smoke（无协作权重） | +0.003 | -0.005 | +0.006 | +0.001 |
| w3.0（单域锚定） | +0.236 | +0.015 | +0.117 | +0.123 |
| global（全域锚定） | +0.175 | +0.237 | +0.423 | +0.279 |
| full（全量协作训练） | -0.154 | +0.498 | -0.212 | +0.044 |

上表是 general 同款阵容的 hub 锚点 cosine；`full` 相对 `global` 在 code / zh 分别下降
0.329 / 0.635，因此不具备主线晋级资格。保留 `global` 作为实验参考，不继续投入同配方的长时训练。

逐域 PPL 只取每域 2 条、共振 1 轮，用于检查评估链路而非产品质量：无协作权重平均 EMERGE
为 +13.1%，`full` 仍为 +13.1%，四域数值差异低于 0.1%。这说明当前增益主要来自固定的
多域融合/词表投影路径，不能归因于 `full` checkpoint 的训练成果。

本轮同时修复两个评估契约问题：无 targets 的生成探针不再引用 fusion 局部变量；共享 256K
输出头由通用词表解码，旧域专用 head 仍按目标域词表兼容解码。生成探针已不再触发
`OUT_OF_RANGE`，但当前随机生成文本仍不可作为语言能力证据。

当前尚未验收的不是启动能力，而是真实训练后的语言能力：公开仓库没有随代码交付可用于质量展示的训练后神经元群体；空目录 fallback 和本次 synthetic probe 只能证明工程链路可启动、路由可观测，不能证明生成质量。跨域协作的历史正式训练还出现过 hub 锚点退化，因此不能直接把旧实验结果当作主线结论。

## 5. 本轮已完成

1. 新增架构决策记录，明确采用稀疏路由群体共振网络。
2. README、CONTRIBUTING、CODE_WIKI 和本文件统一为群体神经元口径。
3. 身份/开发者训练数据移除旧基座模型、整体变大和递归迁移故事，改为群体成长叙事。
4. 清理代码 Wiki 中的旧包路径、绝对本地文件 URI 和过期模块索引。
5. 旧模型升级 API 改为兼容状态接口；整体模型升级动作明确返回退役状态。
6. 将 dense 转换器、旧域语料补全器、旧教师对齐训练器和损坏的生命体数据生成器移入 `scripts/archive/`，不再出现在主训练入口。
7. 删除 `ModelConfig` 中旧的整体模型尺寸/Qwen 预设，保留 token 合约和 checkpoint 兼容字段。
8. 将审计、实验历史和旧训练参考移入 `plans/archive/`，活跃 plans 只保留当前架构与验收口径。
9. 完成开源发布面的安装、README、Cortex 快速开始、API 健康检查和兼容入口验收。
10. 修正启动进度接口的调用契约，以及空群体加载时错误选择 general 域 tokenizer 的问题。
11. 新增 `scripts/verify_population_baseline.py`：固定种子下完成单神经元、稠密协作、稀疏协作、场贡献和路由观测。
12. 完成小型 checkpoint 的内存序列化 round-trip、Cortex.think 和 API health 联合烟测，并加入第 27 项回归测试。
13. 完成真实 checkpoint 的 general 口径 hub 锚点 A/B；拒绝 `full` 晋级，保留 `global` 为实验参考。
14. 修复无 targets 探针和共享 general 词表解码契约，新增回归测试并将总数提升到 30 项。

## 6. 后续工作顺序

### P0：建立最小可复现群体基线（已完成）

已完成 `scripts/verify_population_baseline.py`。它不加载本机大 checkpoint，而是固定随机种子和小型 Transformer 群体，至少输出：

- 单神经元、稠密协作、稀疏协作三组质量指标；
- 每个样本的路由命中、平均激活数、场贡献和共振分；
- Cortex.think 的端到端共振结果和 API health；
- checkpoint 序列化 round-trip、配置、随机种子和可选指标文件，能够让社区复跑同一结论。

当前结果：稀疏 Router 实际介入，round-2 平均激活从 3 降到 2；Cortex 序列化恢复、共振分、场贡献和 API health 均通过。报告明确标记为 `synthetic_probe_only`，因此只验收运行时结构和可观测性，不把随机小模型的 PPL 写成语言能力结论。

真实 checkpoint 的协作质量仍需在 P1 中验收；在此之前，不继续扩展新的生物机制，也不把长时训练结果写成产品能力。

### P1：修复跨域协作训练闭环（下一阶段）

以 `global` 参考和无协作权重基线为参照，先做可归因的短跑：固定 holdout、逐域指标，
仅允许一组协作参数变化，并把 hub/anchor 非退化作为停止条件。先隔离 cross-spec 投影与
side/scale/body 的贡献，再决定是否保留训练更新；不重复当前 `full` 的长时配方。

### P1：完成稀疏路由的真实性验证

用同一训练后 checkpoint 做 dense/sparse A/B，确认 Router 不是随机初始化，并记录质量、激活数量、吞吐和路由熵。只有在质量接近 dense 且激活明显减少时，稀疏路由才算主线能力，而不是代码开关。

### P2：打通生命周期闭环

把“交互 → 场记忆 → 睡眠回放 → 重启恢复 → 再生成”固化成一个端到端验收；再验证新神经元加入、成熟、隔离和恢复不会破坏既有群体。已有 `verify_c26_*` / `verify_c27_*` 机制脚本要收敛成可回归的产品级入口。

### P2：完善社区可用的发布形态

在能力基线稳定后，再决定训练后 demo 群体的交付方式（小型可下载 artifact 或明确的 bootstrap 流程），补齐版本化指标、示例输出和 API/前端的公开命名。秘密信息与绝对路径扫描属于发布收尾，不应替代能力验收。

### 暂不进入主线

- 不再恢复 1.5B 蒸馏、单体大模型升级或模型尺寸阶梯叙事；
- 不继续增加新的“类脑”模块，除非它能进入 P0/P1 的可测闭环；
- 不重复执行未有验收门槛的 40 小时级全量训练；
- 不把历史实验、兼容代码和本地 checkpoint 当作公开产品能力。

## 7. 唯一下一步

执行一次 100 步以内、仅更新 cross-spec 投影的可归因短跑，并用固定 holdout 同时验收逐域 PPL 与 hub 锚点非退化；若任一域退化，立即停止该配方。
