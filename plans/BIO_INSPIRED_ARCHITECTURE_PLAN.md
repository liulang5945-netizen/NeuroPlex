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

### 1.3 小规模神经元路线（重新打开，实验候选）

本轮重新启用“小规模神经元”路线，但不把 `10M` 当作硬规格。历史方案中同时存在两种
参考：早期 compact（约 18M/36M，随词表和实现版本变化）以及 TinyStories 的独立约 10M
模型。后者使用 tied token embedding、独立 tokenizer 和简化的 field 消融路径，不能直接
冒充当前生产 `ResonanceNeuron` 的参数契约。

当前路线的统一口径是：

- 继续使用生产 `ResonanceNeuron`，保留共享输入、field read/write、域输出头和跨规格投影；
- `10M` 只作为目标量级上限，优先寻找更小但仍能参与群体通信的候选；
- 神经元本地参数与群体级共享 embedding 分开计量。共享 `256K×512` 感知表只计一次，不能
  因为每个 checkpoint 保存副本就误判为每个神经元都需要承担完整成本；
- 新候选先作为研究 canary 与现有 **5 个 dialogue + 4 个 general** 混合前向，不能替换五个
  dialogue 神经元，也不能改变默认生产阵容；
- 只有同时通过参数预算、单体前向、跨规格场投影和混合群体回归后，才进入小规模训练。

这条路线用于回答两个独立问题：小神经元能否以更低成本形成有效成员，以及它加入当前群体后
是否带来可测的协作增益。现有五个 dialogue 的 quality gate 仍保持阻断，不能用随机或未训练
的 canary 掩盖现有语言能力问题。

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
- `python -m pytest tests -q`：37 项核心回归测试通过，覆盖对话格式契约、生产 5-dialogue
  默认阵容契约、共振 side-channel、API 健康检查、最小群体基线、跨域评估词表契约和固定
  anchor 参考加载。
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
- 公开发布残留扫描通过：README、贡献指南、代码 Wiki、CI、API 文案和当前数据生成样例
  不再突出整体模型、1.5B 或蒸馏；计划中的边界说明、标准 teacher-forcing 术语和未导出的
  兼容实现按保留规则处理。
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

### 4.2 P1.1 真实资产与能力探针复核（2026-08-19）

本地忽略目录中仍有可复用的 `foundation_v1_general`、`code/math/zh/en + hub` 辅助研究权重、
三域 SFT 数据和 `cross_domain_collab_verify_global.ckpt.pt`；它们不属于公开仓库交付物。
固定 `code_sft[16:24]` anchor 评估复现原记录：smoke `-0.003`、w3 `+0.110`、global
`+0.285`、full `-0.083`，资产和评估口径没有漂移。

使用 global 参考做每域 2 条的短 PPL 探针，协作相对各域个体改善约 `+8.1%～+21.3%`，但
绝对 PPL 仍约为 code `8.8e5`、math `7.9e5`、zh `1.5e5`、en `6.2e4`，生成结果不可读。
因此这次只证明真实 checkpoint 可装配、协作链路有改善信号，不证明语言能力；当前机器也没有
CUDA，不能在未确认预算前启动长时训练。该探针是 `code/math/zh + hub` 辅助路线，不能替代
生产阵容验收。

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
23. 完成 editable 安装、bootstrap 版本化输出、API health、35 项核心测试和全量 Python 编译验收。
24. 将 editable 安装、bootstrap smoke、最小群体/API smoke 纳入 `.github/workflows/ci.yml`，并在本地复跑同一命令链。
25. 完成公开发布残留扫描并收敛当前入口措辞；保留归档、兼容、安全和合成探针边界。
26. 完成辅助研究路线的真实资产盘点、固定 anchor 复核和短 PPL 能力探针；确认协作信号存在但当前生成质量未达可用门槛。
27. 发现并纠正验收阵容遗漏：生产主线是 5 个对话神经元 + 4 个 general 神经元的 9 成员 Cortex，`code/math/zh + hub` 只作辅助研究路径。
28. 按生产默认装配完成 9 成员加载与单问题生成 smoke，并将该阵容列为后续验收的唯一主口径。
29. 将 5-dialogue + 4-general 的生产阵容写入公开文档，并加入无 checkpoint 依赖的默认 ID 回归测试。
30. 完成 9 成员 API 等价对话 smoke；确认运行时可用但生成质量未过门槛，转入 dialogue 路由/融合根因诊断。
31. 受控对比确认五个 dialogue 神经元实际参与生产路由：单 dialogue、显式 5-dialogue
    continuous、5-dialogue fusion 和完整 9 成员 continuous 均完成；遗漏来自验收口径误把辅助
    路径当主路径，不是运行时没有加载五个 dialogue。
32. 发现并修正 C25-E 质量代理的 tokenizer 位置错位：生成输入使用 general tokenizer，旧实现却
    直接使用 zh tokenizer 的 token 位置；统一改用 `build_position_alignment()`，并加入回归测试，
    同步修正 leader 诊断脚本。
33. 用正确对齐重跑 12 条 dialogue 样本：原始共振 leader 命中 NLL 最优仅 1/12（8%），现有
    50/50 质量融合为 2/12（17%）；短生成仍未过质量门槛，说明责任边界尚未收敛，不能开始长训练。
34. 修复 singleton continuous 路由的无偏标准差 NaN 边界，补回归测试；五个 dialogue 单体基线
    在同一生产 Cortex 下重跑完成，均出现不同程度的重复、答非所问或格式退化。
35. 完成五个 dialogue 的来源核对：均为 `alpaca_zh_sft_finetune`、8000 steps、含独立 shared
    embedding；对齐后平均 prompt NLL 以 `zh_aug1_dialogue` 最低（2.3595），但单体短生成仍
    未过质量门槛，说明不能只归因于群体路由，需继续检查训练-推理解码链。
36. 完成 direct/leader/continuous 的训练-推理解码 A/B，并修正 direct 旁路的判定口径：旧旁路
     未清零 refractory 状态，不能作为公平基线；清零后确认生产 ensemble 的 round1 会额外注入
     `ffn_gain=1.25`。
37. 完成五 dialogue 的中性 gain 对照：将 `ffn_gain` 显式设为 1.0 后，单 token round1
    `max_abs_diff` 由约 1.12 降至约 0.011，argmax 对齐，但短生成的重复问题仍在；因此这是
    训练-推理契约修复候选，不是完整语言质量修复。
38. 按决策将默认 DA=0.5 的 `ffn_gain` 收敛为 1.0，映射范围调整为 [0.5, 1.5]，补充中性
    调质契约测试，并同步修正归档 smoke 中的旧断言。
39. 用中性调质重跑完整 9 成员 API 等价对话 smoke：加载与运行均正常，总耗时 228.4 秒；
    但仍出现答非所问、重复、截断和 `1.<0x0A>` 退化，确认调质 parity 修复没有解决语言能力。
40. 完成生成质量责任边界审计：在去重后的固定 hash holdout 上抽取 32 条跨文件 dialogue 样本，
    五个 dialogue 神经元均实现 32/32 的 `答：` answer mask / general→zh 前缀对齐 / 域 token→general
    回填；固定 8 条样本的 full teacher-forcing 与 prompt-only 首 token logits 最大差为 0～7e-6。
    但五个神经元首 token Top-1 仅 28.1%～34.4%，中位目标排名为第 4～9 位，故责任边界收敛到
    对话 checkpoint 的首 token 分布与训练数据/目标，不再修改推理映射链，也不启动长训练。
41. 完成训练目标短审计并修复评估契约：按真实入口的 `max_texts=100000` 复现后，去重得到
    55,661 条，hash 切分为 train=52,900、eval pool=2,761，但训练只使用前 100 条验证样本；
    128 token 截断覆盖 eval pool 的 1,211/2,761 条，answer mask 对齐覆盖率为 97.8125%。
    发现验证 PPL 分母错误包含未对齐 target，已改为只计有效 aligned answer token，新增训练口径
    元数据和回归测试；不回溯修改现有 checkpoint，不启动重训。
42. 完成五个现有 dialogue checkpoint 的修正 holdout PPL 质量门：同一 100 条样本上，corrected
    PPL 为 67.34～72.57，旧分母将 180 个未对齐位置计入 6,884 个 token，系统性低估约 7.0～7.8
    PPL；该评估偏差不足以解释首 token Top-1 仅 28.1%～34.4%，且五个 checkpoint 没有形成可用
    质量梯度，因此当前训练配方不直接进入长训晋级。
43. 完成 100 条 holdout 的答案首 token 数据审计：78% 首 token 为汉字、14% 为数字、5% 为拉丁
    字母、1% 为特殊/byte；来源文件为 `alpaca_zh_sft_clean` 84 条、`dialogue_extended_clean`
    16 条，截断 42%。汉字首 token 在五个 neuron 上 Top-1 仅 10.3%～12.8%，而数字首 token 为
    64.3%～85.7%，故当前主因不是英文/代码污染，而是中文答案起始分布未学好。
44. 完成单 neuron 不落盘微型过拟合：固定 8 条未截断汉字首答案，32 步后 answer loss 从 4.6889
    降至 0.2568，首 token NLL 从 6.7835 降至 0.0569，中位 rank 从 244 降至 0，Top-1 从 12.5%
    升至 100%。因此域输出头、token 对齐和有效 mask 均可学习，长训练退化责任收敛到优化配方/损失
    对中文答案起始的权重不足，不修改域映射链，也不直接启动长训。
45. 完成首答案 loss 权重短 A/B：固定 8 条训练/8 条评估样本、32 步、同一 `zh_aug0_dialogue`
    初始权重；原始 token-mean 的评估首 token NLL 从 5.680 升至 6.514、Top-1 从 12.5% 降至 0%，
    `0.8×token_mean+0.2×first_token` 则升至 7.262、Top-1 仍为 0%。两种配方训练集目标均下降
    但泛化变差，因此不把首 token 权重直接纳入正式训练入口。
46. 完成生产 9 成员解码敏感性审计：固定 4 个代表问题、seed、soft fusion、temperature=0.55、
    repetition penalty=1.4 和 8 token，比较 `top_k=15/40/100/1`；`top_k=40` 仅改善问候，greedy
    仅改善天气，身份和诗歌在所有候选集都不可靠，故 Top-K 不是根因，不改变默认解码参数。
47. 完成五个 dialogue 的 500 步短答案 curriculum pilot（不落盘）：短答案子集 64,476 条，固定
    100 条原始 holdout；五个 corrected PPL 全部从 67.34～72.57 恶化到 84.26～89.99，首 token
    Top-1 没有改善。训练 loss 下降但泛化一致退化，故否决当前 curriculum，不覆盖现有权重，也不
    启动正式重训。
48. 完成 `zh_aug2_dialogue` 的 200 步冻结 shared embedding pilot（不落盘）：原始混合数据下
    corrected PPL 仍从 67.7344 恶化到 72.0445，首 token Top-1 仅 23%→24%，rank 轻微改善但不
    足以通过质量门；排除“只因 shared embedding 更新”的解释，停止继续试训练配方。
49. 完成五个 dialogue checkpoint provenance 审计：全部为 8000 步，日志最后一次评估即历史 best；
    optimizer 当前 lr=1e-5、initial lr=1e-4，scheduler `last_epoch=8000`，五个 per-neuron shared
    embedding 均为 256000×512，未发现 latest/best、optimizer 恢复或结构错位。当前训练侧低成本
    修复路径全部否决，不再盲目续训；旧 checkpoint 的 `data_source` 标签仍是历史值，新训练入口已
    改为记录完整 dialogue 数据口径。
50. 建立并提交 `reports/dialogue_quality_baseline_20260819.json` 发布阻断报告：固定 9 成员阵容、
    corrected PPL、首 token、答案数据、解码敏感性和两种短 pilot 结果全部版本化；quality gate 明确
    为不通过，当前权重冻结为 baseline，不再包装为已达标语言能力。
51. 完成生产 `ResonanceNeuron` 小规模候选扫描并生成 `reports/micro_spec_sweep_20260819.json`：
    6.97M～11.81M 的 8 个候选全部通过单体前向；与 compact neuron 的两轮混合共振均通过，
    跨规格正向/反向投影均实际建立且输出有限。结果确认 `10M` 不是硬值，当前最适合首轮试验
    的 4 层候选为 `hidden=128 / intermediate=384 / field_dim=512`，实际本地参数约 7.58M；
    共享 `256K×512` 感知表约 131M，只按群体级共享成本计一次。此次只验证结构，不写入权重。
52. 完成 `micro` 规格的不落盘对话 pilot：4 层/128 hidden/512 field，实际本地参数约 7.58M，
    冻结共享感知表，160 步、128 条训练样本；固定 100 条 holdout 的 corrected PPL 从随机态
    82,416.47 降至 6,139.25，首 token Top-1 从 0 升至 11%。这只证明小成员可以学习目标，
    远未达到可用语言能力，也没有覆盖或写入现有五个 dialogue 权重。
53. 完成真实群体 canary：将上述 `micro` 成员通过 `ResonanceEnsemble.add_neuron` 临时追加到真实
    5 个 dialogue + 4 个 general 阵容。10 成员、3 轮混合前向有限，场维度为 3072；固定 4 个
    问题的 9/10 成员生成中，3 个输出完全不变，1 个仅末尾 token 有差异，重复率/可读性没有
    可测改善，也未见结构性退化。结论是“混规格接入成立”，但 160 步未成熟 micro 尚不能证明
    能力增益；不进入默认装配、不写入 checkpoint。
54. 完成 1,000 步 `micro` 延长 pilot 和第二次真实群体 canary：训练集 answer loss 降至 0.33，
    但固定 holdout corrected PPL 为 15,851.42（差于 160 步的 6,139.25），首 token Top-1 为
    5%（低于 160 步的 11%）。10 成员生成虽有表面变化，仍是中英碎片与重复，不能计为能力
    提升。重要口径修正：该 pilot 只取了 128 条训练样本，不能据此断言当前完整对话数据不足；
    它只证明在小样本上延长步数会过拟合。完整数据已知为去重 55,661 条、train=52,900、
    eval pool=2,761，但尚未用同一 micro 规格完成全量训练验证。
55. 完成 Hugging Face 标准候选数据的来源审计与下载：选用
    [`fnlp/moss-003-sft-data`](https://huggingface.co/datasets/fnlp/moss-003-sft-data)
    的 no-plugin 原始包，数据卡许可证为 CC BY 4.0，固定 revision 为
    `42e216d3e3fb331c18d5fa6e7cb4f1c53eef24a4`，原始包 SHA-256 已写入候选 manifest。
    新增 `scripts/data_prep/download_hf_dialogue_candidates.py`，将 `chat.turn_N.Human/MOSS`
    转为当前 `问：...\n答：...` 契约，剔除中文占比低于 20%、空/模板答案、重复项和与现有
    `data/simple_zh` 的精确重叠；结果为每类 6,000 条、共 48,000 条候选回合，其中 train=45,600、
    eval=2,400，8 类均衡，train/eval 无交集。候选数据写入独立的
    `data/hf_candidates/moss_003_dialogue`，不覆盖现有数据、不写入任何五个 dialogue checkpoint，
    当前仍是 candidate-only；`alpaca-zh`（许可证口径冲突/README 限研究）、Firefly（无明确许可证）
    和 Belle（GPL-3.0）未纳入主来源。
56. 完成工作区存储审计并建立显式清理边界：总量约 129.1GB，其中 `data/neurons` 单独约
    97.4GB，主要是已否决的 C13/C14/C16/cross-domain 历史 collab checkpoint、smoke 产物和
    `pre_t12_backup`；另有已被 `foundation_v1_dual` 替代的 `foundation_v1_general`/
    `foundation_v1`，以及废弃的 `distill`、`neurons_joint`。核对确认 7 个 `sft_*_clean` 文件
    与 `alpaca_zh_sft_clean.jsonl` 逐条重复，原始 `sft_*` 是中间拆分产物。新增显式 allowlist
    清理脚本 `scripts/maintenance/cleanup_redundant_artifacts.py`，干运行精确识别 60 个目标、
    约 99.2GB；保留 5 个 dialogue、C24v2、slow-test hub fixture、`foundation_v1_dual`、
    canonical 对话数据和 HF candidate。`DIALOGUE_DATA_FILES` 已收敛为
    `alpaca_zh_sft_clean.jsonl` + `dialogue_extended_clean.jsonl`，运行时不再静默下载旧的
    Belle/COIG 来源。
57. 完成 7.58M micro 的完整数据 A/B pilot：同一随机初始化、冻结 shared embedding、160 步、
    训练池为 current=95,059 与 current+HF=140,658，评估池为 current=4,941、HF=2,400。
    `current-only` 在 current eval 上 corrected PPL=3,369.54、首 token Top-1=5.07%，
    在 HF eval 上 PPL=6,109.23；`current-plus-hf` 在 current eval 上 PPL=3,549.24、
    Top-1=0.67%，但在 HF eval 上 PPL=3,402.01、median rank=479，明显优于 current-only。
    结论：HF 数据提供互补分布能力，直接无权重合并会损伤 current 分布，不能直接替换
    canonical 数据或写入五个 dialogue checkpoint。增广 micro 临时加入真实 5 dialogue +
    4 general 后，10 成员/3 轮 forward 有限、field shape=3072，但生成仍有高重复，暂不计为
    群体能力提升；完整报告为 `reports/micro_data_ab_20260819.json`。

    随后执行的固定 75% current / 25% HF pilot 使用 95,059 条 current 训练样本与确定性抽取的
    31,686 条 HF 训练样本，实际比例为 75.0002% / 24.9998%，仍保持 7.581313M local params、
    shared embedding 冻结、160 步且不写 checkpoint。current eval corrected PPL=3,566.36，
    相比 current-only 的 3,369.54 恶化 5.8%，首 token Top-1 由 5.07% 降至 0.67%；HF eval
    PPL=3,537.45，相比 6,109.23 明显改善，median rank 由 1,410 降至 764。真实 5 dialogue
    + 4 general 装配仍为 9 成员，临时加入 micro 后 10 成员/3 轮 forward 有限通过，但生成
    重复率仍为 0.57/0.61，不能计为群体能力提升。因此 25% HF 仍然过重，不能写入生产或替换
    canonical 数据；完整报告为 `reports/micro_data_mix_7525_20260819.json`。

    进一步执行的固定 90% current / 10% HF pilot 使用 95,059 条 current 与 10,562 条 HF
    训练样本，实际比例为 90.0001% / 9.9999%。current eval corrected PPL=3,440.38，较
    current-only 的 3,369.54 上升 2.1%，但 first-token NLL=8.933、median rank=1,046，
    均优于 current-only 的 9.048 与 1,316；首 token Top-1 仍由 5.07% 降至 1.85%。HF eval
    PPL=4,784.22，较 current-only 的 6,109.23 改善，median rank=973（对照 1,410）。
    9 成员装配与临时 10 成员 forward 均有限通过，但生成仍高重复，尚未证明群体能力提升；
    完整报告为 `reports/micro_data_mix_9010_20260819.json`。因此 10% 是当前唯一保留的
    实验比例候选，但仍不可写入生产。

    在固定 90% current / 10% HF 下继续执行 800 步后，current-only 的 current eval PPL=1,737.48、
    median rank=296、首 token Top-1=5.31%；90/10 的 current eval PPL=1,713.50、median
    rank=386、Top-1=5.42%。90/10 的 HF eval PPL=2,020.61、median rank=153，优于
    current-only 的 PPL=3,906.83、median rank=306。说明 7.581313M local params 在现有
    数据上可继续学习，且 10% HF 对跨分布泛化有明确收益；但同 prompt/同 seed 的 9 成员
    vs 10 成员生成中，重复 bigram 从 0.5714/0.6105 变为 0.5859/0.6667，micro 尚未
    形成正向群体增益。完整报告为 `reports/micro_long_9010_800_20260819.json`。

    路由 canary 进一步确认，单独对 256 条 current 样本做 prototype warm-up 不足以让 micro
    进入 auto-top-k；随后只优化 micro 的 `embed_adapter` 与自身 hidden response 的 cosine
    对齐（2 epochs，loss 0.953012→0.188699），再更新 prototype。此时 auto-top1 已选中
    `zh_micro_dialogue_ab`，auto-top2 已选中 `zh_micro_dialogue_ab + zh_aug0_dialogue`，
    auto-top2 重复 bigram 降至 0.2727/0.3256；但输出仍有明显语义碎片，不能只凭表面重复率
    认定能力提升。完整报告为 `reports/micro_route_adapter_calibration_9010_800.json`。

    路由 adapter 回归筛查证明，直接改写 micro 原有 `embed_adapter` 会破坏语言能力：在
    512 条 current eval 上 PPL 从 1,681.64 升至 2,351.87，在 512 条 HF eval 上从
    2,006.51 升至 2,482.57；尽管 micro 成功进入 auto-top-k，代价不可接受。因此该
    “复用语言 adapter 做路由校准”的方向已标记为失败，不得写入 checkpoint 或生产配置；
    完整报告为 `reports/micro_route_regression_9010_800.json`。

    外部独立 route projection（65,536 参数）在冻结 micro 语言主体和原有 `embed_adapter`
    的条件下完成：路由拟合 loss=1.018911→0.087409，current/HF 两套 512 条回归筛查的
    PPL/NLL 前后完全不变；它能把 micro 排到 external top1，并在 top2 中与 `zh_aug0_dialogue`
    共存。但生成仍是碎片化文本，top2 重复 bigram=0.4096/0.6765，没有形成可接受的群体
    语义增益。因此外部 route projection 只证明“可无损接入路由实验”，不证明 micro 可以
    进入生产；完整报告为 `reports/micro_external_route_9010_800_final.json`。

    真实 9 成员单体归因进一步显示，5 个 dialogue 单体的 general→domain 对齐 prompt
    NLL 均值为 2.3601–2.6909，单体短生成大多保持可读片段；因此当前严重碎片化主要由
    群体融合/路由引入，而不是五个 dialogue checkpoint 全部失效。完整报告为
    `reports/production_dialogue_single_baseline_20260819.json`。

    9 成员 fusion A/B（soft、per_position、residual、division）在 4 个固定问题中有 3 个
    输出完全一致，只有问候问题从“你没有什么？”变为“你怎么样？”；身份、天气和诗歌结果
    没有实质改善。因此不存在一个现成 fusion mode 可以修复生产语义退化，完整报告为
    `reports/production_dialogue_fusion_ab_20260819.json`。

    5 dialogue-only 与完整 9 成员的同 prompt/seed 对照显示，general 成员会改变输出，但
    没有一方在问候、身份、天气、诗歌四类问题上全面占优：dialogue-only 的天气输出为
    “今天天气是晴朗的天气模式”，完整 9 成员为“今天天气很好，我很高兴”；身份和诗歌也
    各有碎片化。结论是不能删除 `data/foundation_v1_dual` 的 4 个 general，当前责任边界
    仍需回到生成契约/对齐链路。完整报告为 `reports/production_dialogue_population_subset_ab_20260819.json`。

    完成 2,000 步全量 micro 数据 A/B：同一随机初始化、冻结 shared embedding、current
    train=95,059、HF train=10,562（实际 90.0001% / 9.9999%），current eval=4,941、HF
    eval=2,400。current-only 的 current eval PPL=1,033.12、median rank=161、首 token
    Top-1=7.11%，HF eval PPL=2,484.42；90/10 的 current eval PPL=1,051.17、median
    rank=162、Top-1=6.84%，HF eval PPL=1,148.10、median rank=66。相较 800 步，90/10
    current PPL 继续下降 38.65%，HF PPL 继续下降 43.19%；相较同步 current-only，HF
    PPL 下降 53.79%，而 current 几乎不退化。结论是 7.581313M local params 已证明有
    可测的单体学习和跨分布泛化，但 9+micro 临时群体仅 finite forward 通过，固定生成中
    一条 prompt 发生轻微变化且重复率变差，未形成群体净增益；不写入生产。完整报告为
    `reports/micro_long_9010_2000_20260819.json`。

    完成三档小规格同预算筛选：6.974593M（2 层/field=256）、7.196033M（3 层/field=256）和
    7.581313M（4 层/field=512）均使用相同 90% current / 10% HF 数据、冻结同一份 shared
    embedding、800 步和完整 current/HF holdout。最小的 6.97M 单体结果最佳：current PPL=
    1,697.77、HF PPL=1,989.95、current median rank=323；7.20M 为 1,701.95/2,005.29、
    rank=350；7.58M 为 1,713.50/2,020.61、rank=386。6.97M 不是因为参数更多而获胜，且
    local cost 最低；三档 9+1 canary 的 forward 均有限通过，但固定生成没有可重复净增益，
    7.58M 还出现重复率上升。因此当前最值得延长训练的是 6.97M，而不是默认回到 7.58M；
    完整报告为 `reports/micro_spec_data_ab_800_20260819.json`。

    继续完成 6.974593M（2 层/hidden=128/field=256）的 2,000 步延长训练：current eval
    PPL=1,028.05、median rank=134、首 token Top-1=7.68%；HF eval PPL=1,136.01、median
    rank=77、Top-1=0.50%。相较同口径 7.581313M 的 2,000 步结果，current/HF PPL 分别再
    下降约 2.20%/1.05%，local 参数减少约 8.00%；但群体级 shared embedding 成本仍只
    能通过一次加载共享，不能误报为总成本同步下降。9+1 canary forward 通过，固定生成一
    条 prompt 发生变化且重复率由 0.5714 升至 0.5859，另一条不变，仍无群体净增益；不
    写入生产。完整报告为 `reports/micro_2x128_long_9010_2000_20260819.json`。

    完成 3 个 6.974593M micro 专长成员的同进程实验（各 800 步、shared embedding 只加载
    一次）：current-only 在 current/HF eval 上为 PPL=1,702.13/3,882.57，HF-only 为
    5,959.87/1,031.60，90/10 为 1,697.77/1,989.95，证明小成员可以因数据分工形成可测
    专长。三者临时加入后成为 12 成员，mixed forward 有限且 3 个成员均实际装配，但两条
    固定生成与原 9 成员完全一致，未出现群体净增益。结论是继续复制或延长小成员训练前，
    必须先审计路由可见性和场贡献；完整报告为 `reports/micro_specialist_group_697m_800_20260819.json`。

    完成 3 个 6.97M 专长成员的只读 route/contribution audit：为每个成员单独拟合 65,536
    参数的 external route projection（语言主体和原 embed_adapter 冻结），两条固定 prompt
    上三个 specialist 均进入 external top1～top3，route score 约 0.632～0.699；其投影后
    field norm 约 0.987～1.020，与现有成员同量级，12 成员 field state norm 约 1.0，说明
    成员不是不可见或场贡献为零。all-12 生成仍只在一条 prompt 上变为更高重复，另一条不变；
    external top1/top3 能改变表面重复但仍是语义碎片，不能计为能力提升。结论是小成员容量、
    路由可见性和跨规格投影均已打通，当前瓶颈收敛到 route/fusion 的贡献利用与信用分配；
    external projection 仍不接入生产。完整报告为 `reports/micro_specialist_route_audit_697m_800_20260819.json`。

## 6. 后续工作顺序

### P0：建立最小可复现群体基线（已完成）

已完成 `scripts/verify_population_baseline.py`。它不加载本机大 checkpoint，而是固定随机种子和小型 Transformer 群体，至少输出：

- 单神经元、稠密协作、稀疏协作三组质量指标；
- 每个样本的路由命中、平均激活数、场贡献和共振分；
- Cortex.think 的端到端共振结果和 API health；
- checkpoint 序列化 round-trip、配置、随机种子和可选指标文件，能够让社区复跑同一结论。

当前结果：稀疏 Router 实际介入，round-2 平均激活从 3 降到 2；Cortex 序列化恢复、共振分、场贡献和 API health 均通过。报告明确标记为 `synthetic_probe_only`，因此只验收运行时结构和可观测性，不把随机小模型的 PPL 写成语言能力结论。

真实 checkpoint 的协作质量仍需在 P1 中验收；在此之前，不继续扩展新的生物机制，也不把长时训练结果写成产品能力。

### P1：修复跨域协作训练闭环（anchor 契约已完成，生产阵容待验收）

anchor 目标契约已经落地并通过短验收：参考 checkpoint 只提供 cross-spec 投影，hub 投影
必须冻结，optimizer/body/side/field 不会从参考继承。该路径只作为安全实验基础，不作为
语言能力来源。

辅助路线的 P1.1 资产复核已完成，但真实能力尚未达标；后续训练不能沿用 `full` 失败配方，也不能把
当前高 PPL 或不可读生成包装成产品结论。

此前执行的 P1.2 安全短跑固定 `code/math/zh + hub`，每域 2 条样本、batch 1、1 epoch（共 3 步）、
`lr=1e-4`；只训练 cross-spec 投影，hub 投影冻结，仅从 `global` 读取 anchor 投影，不加载旧
side/body/optimizer 状态。它只作辅助实验诊断，不能代表生产群体；临时 checkpoint 与日志已清理。

生产阵容的真相源是 `neuroplex/core/model_loader.py`：默认加载
`zh_aug0_dialogue`、`zh_aug1_dialogue`、`zh_aug2_dialogue`、`zh_aug3_dialogue`、
`zh_std0_dialogue` 五个对话神经元，再通过 `data/foundation_v1_dual` 加载 code/math/zh/en
四个 general 神经元，合计 9 个成员。后续生产能力验收必须以这条装配链为准。

生产主线不能遗漏 5 个对话神经元：它们是当前默认 Cortex 的主要对话能力来源，必须和
`foundation_v1_dual` 的 4 个 general 成员一起做装配、路由、生成和回归验收。

生产装配 smoke 已按 `model_loader.py` 的默认参数通过：实际加载 9 个成员
（5 个 dialogue + code/en/math/zh 4 个 general），并完成一条 zh 对话生成；这只证明装配
和路由入口可用，不等于完整语言质量验收。

公开架构契约已补齐：README、CODE_WIKI、CONTRIBUTING 明确 5 个 dialogue + 4 个 general
的默认 9 成员阵容，`tests/test_population_assembly_contract.py` 锁定 5 个 dialogue ID。

9 成员 API 等价对话 smoke 已完成：8 个问题均能返回结果，总耗时约 209.6 秒，无加载或运行时
异常；但出现重复、答非所问和截断输出，语言质量门槛未通过。生产阵容已确认，下一步应诊断
dialogue 路由/融合与基座能力的责任边界，不应直接开始长训练。

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

### P3：完善社区可用的发布形态（bootstrap、发布收口、CI smoke 与残留扫描已完成）

已选择可复现 bootstrap 流程作为社区首运行入口：它不下载或加载私有 checkpoint，展示群体运行
契约但不伪装成训练后语言能力。版本化指标、示例输出、editable 安装和 API/测试验收已完成；
秘密信息与绝对路径扫描仍属于发布收尾，不应替代能力验收。

轻量 CI 现在覆盖 Python 3.10/3.12 的 editable 安装、bootstrap、最小群体/API smoke 和核心
回归测试；它只验证公开运行时契约，不把私有模型或长训练带入开源流水线。

公开入口扫描已经完成：1.5B/蒸馏不再出现在产品身份和首运行路径；兼容实现仍保留在非顶层
模块中，以避免历史 checkpoint 失效。

### 暂不进入主线

- 不回退集中式迁移、单体大模型升级或模型尺寸阶梯叙事；
- 不继续增加新的“类脑”模块，除非它能进入 P0/P1 的可测闭环；
- 不重复执行未有验收门槛的 40 小时级全量训练；
- 不把历史实验、兼容代码和本地 checkpoint 当作公开产品能力。

## 7. 唯一下一步

生成契约审计已完成：固定 32 条 current holdout 全部 prefix 对齐，域 token→general 回填
32/32 非空；8 条 teacher-forcing 与 prompt-only 首 token 对照的最大绝对差为 0～7e-6、
cosine 约 1。五个 dialogue 首 token Top-1 为 31.25%～40.62%，中位排名为第 2～4 位。
实现链路通过，当前问题不再归因于 tokenizer 位置、answer mask 或 token 回填；报告为
`reports/production_dialogue_generation_contract_20260819.json`。

基于 7.58M micro 的本地参数、低训练成本和 2,000 步结果，项目继续推进小神经元路线；
但当前证据只支持“单体可学习、HF 跨分布泛化可改善”，不支持“已经产生群体智能”。生产
9 成员和五个 dialogue checkpoint 仍保持冻结，不被实验权重覆盖。

路由/融合信用分配试验已完成（2026-08-19）：在真实 9 成员上加入 3 个 6.974593M
临时专家，三个专家各训练 800 步；随后冻结 12 个语言主体、field、跨规格投影和 shared
embedding，只训练 quality_head 40 步。三个小专家本体的 held-out answer PPL 均显著下降：
current-only 为 83,475→1,754，HF-only 为 83,475→5,781，90/10 混合为 83,475→1,743
（同一固定 8 条 current holdout；HF holdout 分别为 91,222→2,961、91,222→780、
91,222→1,219），再次证明小神经元可学习且能形成数据专长。

但路由训练没有形成群体增益：shadow NLL 在第 10/20 步约 22.33/20.04 后升至
174.85/961.92，显示 quality_head 尺度与当前 soft shadow 目标不稳定；真实生产硬路由
训练前后都 100% 选择 `zh`，9→12 群体的固定留出 teacher-forcing NLL、PPL 和两条生成
均无改善（报告 delta 为 0，第二条生成完全不变，第一条重复率 0.3253→0.5859）。
这排除了“7M 小神经元不可见/容量不足”作为当前主因，当前瓶颈明确为 quality_head 的
初始尺度、温度和质量信号校准；本次试验没有写入生产 checkpoint。完整报告为
`reports/micro_route_fusion_pilot_697m_20260819.json`。

本轮不再延长路由训练，也不把失败的 quality_head 产物写回生产。只读 route calibration
已完成：12 个成员的 quality-logit 偏置确实巨大（`zh` 均值约 18,096，dialogue 约
-9,128～-17,459，三个 micro 约 0），member-wise zero-center/unit-scale + bounded
trust 能把 `zh` 的平均路由权重从 1.0 降到 0.531，并让多个 dialogue/en 成员被选中；
但固定 8 条留出集的 hard-route teacher-forcing NLL 从 116.231 恶化到 933.673。结论是
尺度偏置是真问题，但简单校准不是能力信号，不能接入生产。完整报告为
`reports/micro_route_calibration_697m_20260819.json`。

只读 oracle projected-NLL 上界审计已完成：固定 8 条留出集上，raw hard route 的
teacher-forcing NLL 为 116.231，单一最佳成员为 14.881，逐 token oracle 为 12.330，
oracle 相对 raw 的理论改善为 89.4%；三个 micro 专家合计赢得约 29.0% 的答案 token，
说明 9+3 群体确实存在互补能力。当前 raw 路由选择 `zh`，但 `zh` 的 projected NLL
均值约 98.22，明显劣于 code/en/math 的约 14.85/15.78/16.23；因此当前主因确定为
route credit assignment 与输出尺度，而不是 micro 容量。完整报告为
`reports/micro_oracle_route_audit_697m_20260819.json`。

当前停在实现决策节点。唯一推荐下一步：构造一个临时、有界、输入归一化的 route head，
只用 projected per-member NLL 生成的理想路由分布做监督，先在同一固定留出集验证它能否
逼近 oracle；仍冻结 12 个语言主体、原 embed_adapter、field 和 shared embedding，
只允许内存中的 route head 变化，验证前不得写入生产。当前环境未检测到 CUDA，继续采用
单进程复用 shared embedding。

临时有界 route head 试验已完成（2026-08-20）：LayerNorm + MLP + `2*tanh` 只训练
route head 80 步，语言主体、embed_adapter、field、跨规格投影和 shared embedding 全部
冻结。固定 current holdout 上 hard-route teacher-forcing NLL 从生产 raw route 的
116.231 降到 17.184，接近单一最佳成员 14.881，但仍高于 oracle 12.330；这证明输入
归一化和有界输出确实修复了 `zh` 独占造成的主要损失。与此同时 quality logits 很快
触及 ±2 边界，最终路由主要落在 en/code（0.766/0.234），三个 micro 仍未获得有效
路由权重，因此不能宣称群体能力已被完全利用。完整报告为
`reports/micro_bounded_route_head_697m_20260819.json`。

唯一推荐下一步：在完全相同的临时 route head 配方下加入独立 HF eval holdout，验证
17.184 的收益是否跨数据分布保持；仍只改内存 route head，不写生产 checkpoint。若 HF
收益消失，再调整监督温度/边界；在独立泛化验收前不接入默认路由。当前环境未检测到 CUDA，
继续采用单进程复用 shared embedding。

独立 HF holdout 验收已完成（2026-08-20）：同一 LayerNorm + MLP + `2*tanh` route head
在 current holdout 上将 hard-route NLL 从 116.231 降到 17.184，在独立 HF holdout 上从
129.804 降到 55.725，说明 sample-level route head 的收益可以跨数据分布泛化，但幅度
明显减弱。训练后路由仍主要选择 en/code（current 约 0.766/0.234，HF 约 0.670/0.330），
三个 micro 基本没有被选中；quality logits 触及边界，说明当前监督用“整回合平均 NLL”
会偏向稳定的 general 成员，无法利用 micro 在不同 token 位置上的互补优势。完整报告为
`reports/micro_bounded_route_head_hf_697m_20260820.json`。

当前停在下一项架构决策节点。唯一推荐下一步：保持有界、归一化和全冻结边界不变，把
route supervision 从 sample-level 平均 NLL 改为 per-position projected-NLL 目标，验证
route head 能否逼近 oracle 并让 micro 赢得其已证明具备的 token 级互补份额；验证前仍不
接入默认路由或写生产 checkpoint。当前环境未检测到 CUDA，继续采用单进程复用 shared embedding。

per-position projected-NLL 监督实验已进入实现阶段（2026-08-20）：临时 route head 仍为
LayerNorm + MLP + `2*tanh`，但训练目标改为每个答案 token 上各成员 projected NLL 的
softmax，再聚合为样本级信任分布；同时报告 current/HF 的逐 token target 权重和 oracle
位置胜率。该改动仍只作用于内存 route head，不改生产 forward、默认 loader 或 checkpoint；
下一步先完成烟测，再运行正式 current + HF 验收。

per-position projected-NLL 正式验收已完成（2026-08-20）：在同一 9+3 临时群体、三组
micro 各 800 步、bounded route head 80 步的条件下，current holdout hard-route NLL 为
`116.231 → 17.876`，独立 HF holdout 为 `129.804 → 59.675`，均略差于 sample-level
目标的 `17.184/55.725`。逐 token target 本身确认了 micro 的互补能力仍存在：current
三组 micro 获得约 14.9% 的平均 target 权重、赢得 29.0% 的 oracle 位置；HF 获得约
13.7% target 权重、赢得 21.3% oracle 位置。但训练后的实际 hard route 中三组 micro
仍为 0%，而 code/en/math logits 触及 `2*tanh` 上界，说明瓶颈已从监督目标进一步收敛
为“标量样本级 quality head 无法表达逐位置信任”。本实验没有写入生产 checkpoint；
正式报告为 `reports/micro_per_position_route_head_697m_20260820.json`。

当前停在架构决策节点。唯一推荐下一步：保持全冻结和生产边界不变，把临时 route head
升级为能读取 token-level hidden state、输出 `[成员, 位置]` trust logits 的 per-position
route head，再用 projected token NLL 监督；在该 head 通过 current/HF 验收前，不接入默认
路由，也不保存生产权重。当前环境未检测到 CUDA，继续采用单进程复用 shared embedding。

token-level route head 实现已开始（2026-08-20）：新增显式实验开关，让 Neuron 在默认
关闭时完全保持原有回合级 quality_head；实验打开时返回 token hidden 上的逐位置 logits，
并让临时 route fusion 使用 `[成员, batch, 位置]` trust。生产 loader、默认 forward 行为、
语言主体、field 和 shared embedding 均保持冻结；下一步先跑 smoke，确认默认路径和 token
路径的张量形状都稳定，再运行正式 current + HF 验收。

token-level route head 烟测已通过（2026-08-20）：默认全量测试保持 `45 passed`，token
route 的最小 current/HF 样本均完成 forward、逐位置 projected-NLL 反传和 hard-route 评估；
烟测中 micro 已获得非零实际位置份额，但样本量不足以作结论。正式验收现已开始，仍不写入
生产 checkpoint。

token-level route head 正式验收已完成（2026-08-20）：current hard-route NLL 为
`116.231 → 82.869`，HF 为 `129.804 → 86.481`，证明 token-level head 能改善 raw route，
但明显不如先前的 sample-level bounded head（`17.184/55.725`）；current/HF 的 micro
实际 hard-route 份额仍接近 0%，尽管 oracle 位置胜率仍存在。结论是逐位置输出形式正确，
但不同 hidden-size 成员的 token hidden 空间仍不可直接比较，per-member 独立 head 学到
了 general 成员偏置。本实验没有写入生产 checkpoint，正式报告为
`reports/micro_token_route_head_697m_20260820.json`。

下一步已确定并开始：在完全相同的冻结边界下增加共享 route feature 对齐层，把 hidden=512
和 hidden=128 的 token hidden 投影到共同 route 空间，再由一个共享 token route head 输出
逐成员/逐位置 trust；这样只训练临时 route adapter + shared route head，直接检验问题是否
 来自成员表示空间不可比。通过 current/HF 验收前仍不接入默认路由、不写生产 checkpoint。

共享 route feature 对齐实验开始（2026-08-20）：保留 token-level 路由和 projected-NLL
目标，新增共同 128 维 route 空间；每个成员仅训练一个 hidden→128 的临时 route adapter，
所有成员共享同一个 LayerNorm + MLP + `2*tanh` scorer。上一轮 token-level 独立 head
保留为默认实验选项，避免历史结果失去可复现性；本轮正式报告单独保存。

共享 route feature 烟测已通过（2026-08-20）：默认 token route 选项仍可用，共享 128D
adapter + scorer 的 current/HF 单条 smoke 均完成逐位置反传和 hard route，micro 出现
非零位置份额；正式验收现已开始，仍不写入生产 checkpoint。

共享 route feature 正式验收已完成（2026-08-20）：共同 128D route adapter + shared
scorer 没有解决表示偏置，current hard-route NLL 从 `116.231` 恶化到 `432.393`，HF
从 `129.804` 恶化到 `213.569`，三组 micro 的正式 hard-route 份额仍为 0%。因此三种
临时 route 方向的证据已闭合：sample-level bounded head 能大幅改善 raw route 但偏向
general；per-member token head 能表达逐位置 trust 但跨 hidden 空间不可比；shared
feature alignment 又引入更强的统一偏置。所有实验均未写生产 checkpoint，正式报告为
`reports/micro_shared_route_feature_697m_20260820.json`。

当前停在项目路线决策节点。唯一推荐下一步：停止继续堆叠临时 route head 变体，保持生产
9 成员路由不变，转回 7.58M micro 专家本体训练与更大独立 holdout 评估；先用主体能力和
数据专长证据决定 micro 是否值得进入后续架构，而不是继续用未经验证的路由层消耗训练预算。

7.58M micro 专长组的正式验收已完成（2026-08-20）：将训练脚本参数化为可复用的小规格，
在 `micro_4x128_field512`（实际 local params=7.581313M）上固定同一随机初始化、冻结
shared embedding、每个成员 800 步，并使用完整独立 holdout（current=4,941、HF=2,400）。
current-only 的 current/HF PPL 为 `1,737.48/3,906.83`，HF-only 为 `6,117.97/1,023.99`，
90/10 mixed 为 `1,713.50/2,020.61`；三种数据角色均形成清晰专长，且 12 成员临时装配的
forward 全部 finite。与同预算 6.97M 专长组的 `1,702.13/3,882.57`、`5,959.87/1,031.60`、
`1,697.77/1,989.95` 相比，7.58M 没有能力优势，反而略高成本且指标略弱。固定生成中加入
三个 specialist 后重复 bigram 从 `0.5714/0.6105` 升至 `0.5859/0.6667`，仍没有群体净增益；
不写入生产。完整报告为 `reports/micro_specialist_group_758m_800_20260820.json`。

当前进入 micro 规格选择决策节点。唯一推荐下一步：冻结 7.58M 为已验证但不优先的对照，
转向 6.97M 90/10 mixed 单体候选的正式 checkpoint 产出与加载验收；在该候选通过加载、
单体 holdout 和不改变 9 成员生产配置的临时装配前，不再延长 7.58M，也不重启 route head 变体。
