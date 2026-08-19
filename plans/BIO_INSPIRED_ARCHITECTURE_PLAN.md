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

集中式迁移、整体模型升级等词只允许出现在兼容层说明或历史记录中，不得出现在产品身份、快速开始和当前架构主叙事中。

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
- `python -m pytest tests -q`：31 项核心回归测试通过，覆盖对话格式契约、共振 side-channel、API 健康检查、最小群体基线、跨域评估词表契约和固定 anchor 参考加载。
- `python -m pip install -e ".[dev]" --no-deps`：可完成 editable 安装。
- `python -m pip install -e . --no-deps --no-build-isolation`：在当前公开安装入口下完成 editable 安装，包版本为 `neuroplex 1.6.0`。
- 干净启动烟测通过：空神经元目录可以启动 Cortex，并明确进入 fallback；域 tokenizer 由 TokenizerHub 注册。
- API 烟测通过：健康检查、架构能力接口返回 200；旧整体升级入口返回 410 退役响应。
- `verify_c26_*` / `verify_c27_*`：已有机制级验证，覆盖场记忆、睡眠巩固、跨频耦合、实例路由和自组织新生。
- `python -X utf8 -u scripts/training/verify_c27_osc_sleep_e2e.py`：17/17 通过（约 190.5 秒），
  固定 checkpoint 完成“交互→场记忆→两轮睡眠→前向重放→状态保存→重新装配→场记忆/睡眠历史/振荡器恢复→再生成”；
  验证脚本使用项目可控临时目录并在结束后清理，避免系统 Temp 权限差异污染结果。
- `python -X utf8 -u scripts/training/verify_c27_self_organize.py`：20/20 通过（约 87.8 秒），
  固定群体完成“新生→场记忆整合→成熟→低生存分隔离→摘除路由→复活→最新读路径/LoRA 恢复→再生成”；
  同时修复热插拔后的 per-neuron embedding 注册，以及隔离前最新权重未写回 checkpoint 两个真实缺口。
- 社区发布面审计通过：README、贡献指南、代码 Wiki、API 公开文案和活跃计划不再出现旧的
  固定规模或集中式迁移叙事；旧对齐实现仅作为未导出的 checkpoint 兼容模块保留，合法的
  系统安全路径和合成训练样例路径不作为项目痕迹清理。
- `python -X utf8 -u scripts/bootstrap_population_demo.py`：通过（约 2 秒），无需私有
  checkpoint 或外部下载即可展示 3 个 tiny neuron、共振场、稀疏路由和 Cortex 状态 round-trip；
  输出继续明确标记为 `synthetic_probe_only`。
- `python -X utf8 -u scripts/bootstrap_population_demo.py --json-out reports/bootstrap_population_demo.json`：
  通过并生成版本化社区验收输出；固定种子下 dense/sparse PPL 分别为 `33.4602077`/
  `33.4601917`，round-2 平均激活由 `3` 降至 `2`，Cortex round-trip 为真。
- `.github/workflows/ci.yml`：已加入 editable 安装、bootstrap smoke 和最小群体/API smoke；
  本地按同一命令链复跑通过，CI 不下载私有 checkpoint 或执行长时训练。
- 生产路径默认加载 Cortex 群体，并由 API/客户端使用群体状态。
- 默认 tokenizer 已切换到 `neuroplex/domains/general/sp_general.model`；旧 checkpoint 路径不再是主加载路径。
- 旧教师对齐模块未从仓库删除，以避免历史 checkpoint 和实验脚本失效；它不再从 `neuroplex.resonance` 顶层导出，也不在 README/quick start 中出现。

### 4.1 真实 checkpoint 的 P1 短周期验收（2026-08-19）

本轮只做小样本、可中断的匹配评估，不把旧长训练重新包装成产品能力。anchor 结果统一以
`code_sft[16:24]` 固定 holdout 为准；此前前 8 条样本的输出只保留为诊断记录，不作为验收依据。

| checkpoint | code | math | zh | 均值 |
|---|---:|---:|---:|---:|
| smoke（无协作权重） | +0.003 | -0.013 | +0.002 | -0.003 |
| w3.0（单域锚定） | +0.205 | +0.037 | +0.088 | +0.110 |
| global（全域锚定） | +0.142 | +0.286 | +0.427 | +0.285 |
| full（全量协作训练） | -0.442 | +0.545 | -0.353 | -0.083 |

上表是 general 同款阵容的 hub 锚点 cosine；`full` 相对 `global` 在 code / zh 分别下降
0.584 / 0.780，因此不具备主线晋级资格。保留 `global` 作为实验参考，不继续投入同配方的长时训练。

逐域 PPL 只取每域 2 条、共振 1 轮，用于检查评估链路而非产品质量：无协作权重平均 EMERGE
为 +13.1%，`full` 仍为 +13.1%，四域数值差异低于 0.1%。这说明当前增益主要来自固定的
多域融合/词表投影路径，不能归因于 `full` checkpoint 的训练成果；这些样本也不是独立
holdout，因此不计入语言能力结论。

本轮同时修复三个评估/训练契约问题：无 targets 的生成探针不再引用 fusion 局部变量；共享
256K 输出头由通用词表解码，旧域专用 head 仍按目标域词表兼容解码；anchor 评估器固定
holdout 且按实际传入 checkpoint 动态汇总，避免旧投影残留污染。生成探针已不再触发
`OUT_OF_RANGE`，但当前随机生成文本仍不可作为语言能力证据。

补充的 cross-spec-only 诊断（8 步、code+zh）只证明了参数隔离可执行：移动 hub 投影时
训练域提升但未见 math 下降；冻结 hub 投影时 code 也下降。两者均未通过非退化门槛，临时
checkpoint 和日志已清理，不进入主线。

固定 anchor 契约短验收（global 参考、hub 冻结、域投影独立、三域各 2 条、3 步、lr=1e-4）
在 `code_sft[16:24]` holdout 上与 global 完全一致：均值 `+0.285`，code/math/zh 为
`+0.142/+0.286/+0.427`。它证明参考加载与冻结边界正确，但不构成语言能力增益；临时
checkpoint 和日志已清理。

稀疏 Router 机制级 A/B（global 固定参考、Router top-k=2、无 warmup、12 步）通过门槛：
同一 general holdout 上 dense PPL `387.47`，trained-sparse `385.33`（-0.55%）；平均激活
`4→3`（减少 25%），吞吐 `18.72→19.91 tok/s`。随机 Router 对照 PPL `385.75`，选择
分布距离 `0.111`。Router 熵仍约 `1.0`，且样本/步数很小，因此这只是机制级通过，不是
语言能力或生产性能结论。

当前尚未验收的不是启动能力，而是真实训练后的语言能力：公开仓库没有随代码交付可用于质量展示的训练后神经元群体；空目录 fallback 和本次 synthetic probe 只能证明工程链路可启动、路由可观测，不能证明生成质量。跨域协作的历史正式训练还出现过 hub 锚点退化，因此不能直接把旧实验结果当作主线结论。

## 5. 本轮已完成

1. 新增架构决策记录，明确采用稀疏路由群体共振网络。
2. README、CONTRIBUTING、CODE_WIKI 和本文件统一为群体神经元口径。
3. 身份/开发者训练数据移除旧中心化模型、整体升级和递归迁移故事，改为群体成长叙事。
4. 清理代码 Wiki 中的旧包路径、绝对本地文件 URI 和过期模块索引。
5. 旧模型升级 API 改为兼容状态接口；整体模型升级动作明确返回退役状态。
6. 将 dense 转换器、旧域语料补全器、旧迁移对齐训练器和损坏的生命体数据生成器移入 `scripts/archive/`，不再出现在主训练入口。
7. 删除 `ModelConfig` 中旧的整体模型尺寸和品牌预设，保留 token 合约和 checkpoint 兼容字段。
8. 将审计、实验历史和旧训练参考移入 `plans/archive/`，活跃 plans 只保留当前架构与验收口径。
9. 完成开源发布面的安装、README、Cortex 快速开始、API 健康检查和兼容入口验收。
10. 修正启动进度接口的调用契约，以及空群体加载时错误选择 general 域 tokenizer 的问题。
11. 新增 `scripts/verify_population_baseline.py`：固定种子下完成单神经元、稠密协作、稀疏协作、场贡献和路由观测。
12. 完成小型 checkpoint 的内存序列化 round-trip、Cortex.think 和 API health 联合烟测，并加入第 27 项回归测试。
13. 完成真实 checkpoint 的 general 口径 hub 锚点 A/B；拒绝 `full` 晋级，保留 `global` 为实验参考。
14. 修复无 targets 探针和共享 general 词表解码契约，新增回归测试并将总数提升到 30 项。
15. 将 hub anchor 评估固定到独立 holdout，并验证 cross-spec-only 的两种短跑均不能晋级。
16. 落实固定 anchor 目标契约：只加载参考 checkpoint 的 cross-spec 投影，强制冻结 hub，拒绝混用 resume/optimizer 状态。
17. 完成固定 anchor 契约的三域短验收，holdout 非退化通过但不宣称新增能力。
18. 新增 general+hub 稠密/稀疏 A/B 评估器，完成真实 Router 状态的机制级验收。
19. 将 C27 睡眠/振荡器脚本收敛为跨重启端到端回归：场记忆、睡眠历史、前向重放、振荡器参数和重启后生成均通过 17/17。
20. 将 C27 自组织新生脚本收敛为群体成长生命周期回归：新生、成熟、隔离、复活、路由/生成保护和最新权重恢复均通过 20/20。
21. 完成社区发布面审计，移除当前入口中的旧规模/集中式迁移叙事，保留兼容模块但不纳入产品主路径。
22. 新增 `scripts/bootstrap_population_demo.py`，把无 checkpoint 的确定性群体基线收敛为社区首运行入口。
23. 完成 editable 安装、bootstrap 版本化输出、API health、31 项核心测试和全量 Python 编译验收。
24. 将 editable 安装、bootstrap smoke、最小群体/API smoke 纳入 `.github/workflows/ci.yml`，并在本地复跑同一命令链。

## 6. 后续工作顺序

### P0：建立最小可复现群体基线（已完成）

已完成 `scripts/verify_population_baseline.py`。它不加载本机大 checkpoint，而是固定随机种子和小型 Transformer 群体，至少输出：

- 单神经元、稠密协作、稀疏协作三组质量指标；
- 每个样本的路由命中、平均激活数、场贡献和共振分；
- Cortex.think 的端到端共振结果和 API health；
- checkpoint 序列化 round-trip、配置、随机种子和可选指标文件，能够让社区复跑同一结论。

当前结果：稀疏 Router 实际介入，round-2 平均激活从 3 降到 2；Cortex 序列化恢复、共振分、场贡献和 API health 均通过。报告明确标记为 `synthetic_probe_only`，因此只验收运行时结构和可观测性，不把随机小模型的 PPL 写成语言能力结论。

真实 checkpoint 的协作质量仍需在 P1 中验收；在此之前，不继续扩展新的生物机制，也不把长时训练结果写成产品能力。

### P1：修复跨域协作训练闭环（anchor 契约已完成）

anchor 目标契约已经落地并通过短验收：参考 checkpoint 只提供 cross-spec 投影，hub 投影
必须冻结，optimizer/body/side/field 不会从参考继承。该路径只作为安全实验基础，不作为
语言能力来源。

### P1：完成稀疏路由的真实性验证（机制级完成）

已用同一训练后 checkpoint 做 dense/sparse A/B，并增加随机 Router 对照；质量接近 dense、激活
明显减少且选择分布与随机对照有差异。由于 Router 仍是 12 步小样本训练，暂不将它写入
产品性能，后续随正式群体能力评估再复核。

### P2：打通生命周期闭环（已完成，2026-08-19）

已用固定 checkpoint 和固定输入把“交互 → 场记忆 → 睡眠回放 → 重启恢复 → 再生成”固化为
`scripts/training/verify_c27_osc_sleep_e2e.py`，17/17 通过；群体成长阶段已由 P2.1 回归覆盖。

### P2.1：验证群体成长生命周期（已完成，2026-08-19）

已用 `scripts/training/verify_c27_self_organize.py` 固定群体验证新神经元从场记忆生长，
幼稚态权重逐步成熟，低生存分进入隔离并摘除路由，复活后恢复最新读路径和 LoRA；隔离前后
既有生成、路由和场记忆保持可用。过程中修复了新生 neuron 缺少 per-neuron shared
embedding、以及隔离 checkpoint 落后于运行时权重两个生命周期缺口。

### P3：完善社区可用的发布形态（bootstrap、发布收口与 CI smoke 已完成）

已选择可复现 bootstrap 流程作为社区首运行入口：它不下载或加载私有 checkpoint，展示群体运行
契约但不伪装成训练后语言能力。版本化指标、示例输出、editable 安装和 API/测试验收已完成；
秘密信息与绝对路径扫描仍属于发布收尾，不应替代能力验收。

轻量 CI 现在覆盖 Python 3.10/3.12 的 editable 安装、bootstrap、最小群体/API smoke 和核心
回归测试；它只验证公开运行时契约，不把私有模型或长训练带入开源流水线。

### 暂不进入主线

- 不回退集中式迁移、单体大模型升级或模型尺寸阶梯叙事；
- 不继续增加新的“类脑”模块，除非它能进入 P0/P1 的可测闭环；
- 不重复执行未有验收门槛的 40 小时级全量训练；
- 不把历史实验、兼容代码和本地 checkpoint 当作公开产品能力。

## 7. 唯一下一步

进入 P3.3：执行一次公开发布残留扫描，聚焦入口文档、CI、示例和可导入 API 中的秘密、绝对路径、
旧整体模型/蒸馏措辞；归档、兼容、安全和合成探针路径只做边界确认，不做误删。
