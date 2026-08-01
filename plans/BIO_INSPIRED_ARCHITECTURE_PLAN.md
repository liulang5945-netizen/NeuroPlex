# 态极生物学化架构计划 (Bio-Inspired Architecture Plan)

> 本文档记录态极架构借鉴人脑神经科学的系统性改造计划。
> 核心原则：**神经元差异性第一**、**自我进化能力**、**硬件限制不在考虑范围内**。

> **📋 项目主 plan（活跃维护）**
> 本文档是项目的状态总览和路线图。其他 plans/ 文件：
> - `TRAINING_REFERENCE.md` — 训练准则参考（非 plan）
> - `BODY_LIFE_BRAIN_INTEGRATION_PLAN.md` — body/life/brain 子系统设计参考
> - `archive/` — 历史归档（COMPREHENSIVE v1.0、H1-H8 修复记录、side-channels 实现细节）

---

## 🗺️ 一、项目全景（2026-08-01 更新）

> 本章节是文档导航入口，按主题组织项目现状。第三章起为训练流水线详情，第四章起为历史实验记录归档。

### 1.1 系统架构总览

```
taiji/                    核心包
├── resonance/            共振场核心：Neuron/Field/Ensemble + lifecycle(进化) + STDP + 神经调质 + gamma + 几何
├── brain/                Cortex 认知主体：封装 ResonanceEnsemble，generate() 高层接口
├── life/                 生命系统：life_scheduler / feed / sleep / play / evolution 引擎（在线学习闭环）
├── agent/ + agent_ext/   认知五元系统 + ReAct 推理 / MCP / 工具 / 自修改
├── body/                 身体系统：core / limbs(工具) / metabolism(硬件) / senses
├── core/                 app_state / model_loader / websocket / security
├── domains/              域专用 tokenizer（zh/en/code/math/general）
├── multimodal/           多模态（VQVAE / EnCodec / Video）
└── loader.py             统一装配入口 assemble_cortex（接线所有 bio 模块）

api/                      FastAPI 服务层：app.py(lifespan) / chat_strategies / routes_life / routes_taiji / ...
scripts/training/         离线训练流水线（手动脚本）：base → dialogue → cross_spec → eval
data/                     数据 + 产物：simple_zh/ distill/ sft/ neurons/（checkpoint）
```

**核心数据流**：神经元规格分 compact(36M)/standard(116M)/expert(~300M)，各有独立 domain lm_head 和 shared_embedding 副本；协作通过 per-pair side_channels（excite/inhibit）+ 跨规格投影层（field_dim→unified）实现。

### 1.2 训练→推理链路与闭环状态

```
离线训练（手动脚本，已闭环）                    运行时（Cortex/API，未闭环）
base 预训练 ──► dialogue fine-tune ──► cross_spec 协作层 ──╳──► Cortex.forward
(train_compact)  (finetune_neuron)    (finetune_cross_spec)      ↑ 断裂 A-D
       └──► checkpoint: neuron_*.pt / *_dialogue.pt / cross_spec_dialogue.pt
```

**训练侧**：base → dialogue → cross_spec → eval 配置一致（`ENSEMBLE_DIALOGUE_IDS`）、产物齐全，**闭环成立**。
**推理侧**：Cortex 无法装配训练好的综合体（详见 1.4），**未闭环**。

### 1.3 当前状态（2026-08-01）

| 环节 | 状态 | 关键产物 / 结果 |
|------|------|----------------|
| base 训练（5 神经元） | ✅ 完成 | zh_aug0~3 (compact) + zh_std0 (standard, val PPL 34.07) |
| dialogue fine-tune（5 神经元） | ✅ 完成 | 4×compact_dialogue (val PPL 88.85~102.01) + zh_std0_dialogue (95.27) |
| cross_spec 协作层（对话版） | 🔄 训练中 | `finetune_cross_spec.py --data dialogue`，产物 `cross_spec_dialogue.pt` |
| 综合体对话评估 | ⏳ 待协作层完成 | `eval_dialogue.py` |
| 运行时（Cortex/API） | ❌ 断裂 | 混合规格崩溃 + 协作权重未加载 + embedding 随机初始化 + 域路由 key 不匹配 |
| 进化机制（在线） | ⚠️ 部分接线 | 斜率判别器已接入 sleep_engine；`select_spec`/`diagnose_domain` 零调用 |
| 硬编码治理 | ✅ P0-P3 完成 | `experiment_config.py` + `utils.py` 集中管理 |

### 1.4 闭环缺口清单（未闭环设计）

| # | 类型 | 位置 | 说明 |
|---|------|------|------|
| A | 断裂 | [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py#L66-L88) | 混合规格（compact+standard）导致 `Cortex.__init__` 抛 ValueError，运行时启动即崩溃 |
| B | 断裂 | `taiji/`、`api/` 零引用 | `cross_spec_dialogue.pt` 协作层权重（side_channels+投影层）从不加载进运行时 |
| C | 断裂 | [loader.py](file:///e:/taiji-neuron/taiji/loader.py#L278-L283) | 运行时 shared_embedding 随机初始化，不读 `data/shared_embedding.pt` |
| D | 断裂 | [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) `_load_neurons`/`_infer_domain` | 神经元 key 用文件名（zh_aug0...），与域路由期望（zh/en/code/math/general）不匹配 |
| E | 未接线 | [lifecycle.py](file:///e:/taiji-neuron/taiji/resonance/lifecycle.py#L399) `select_spec` | SpecSelector 零调用，`cortex.add_neuron` 硬编码 COMPACT——进化永远生成最小规格 |
| F | 未接线 | [lifecycle.py](file:///e:/taiji-neuron/taiji/resonance/lifecycle.py) `diagnose_domain` | 斜率判别器诊断 API 零调用（判别逻辑已内嵌生效，但对外诊断入口未用） |
| G | 断裂 | loader.py Step 9.2 + play_engine.py | Play→CoactivationTracker 强化链路双重断裂（注入独立实例 + 调用不存在的方法） |
| H | 缺失 | `scripts/training/` | 无真正的多轮对话评测脚本（eval_dialogue 仅 5 个独立单轮 prompt） |
| I | 缺失 | `api/` | 综合体未接入聊天接口，无发布/导出脚本 |
| J | 死代码 | cognitive_enhancements.py | CorticalColumn/ColumnRegistry/AttentionBeam/ThresholdPlasticity 全库零调用 |
| K | 死代码 | translator.py TokenTranslator | P7 后旧对齐机制无调用方 |

---

## 🧭 二、路线图

### 2.1 当前执行中
- 🔄 cross_spec 对话协作层训练（`--data dialogue`，预计 ~13h）
- 完成后：`eval_dialogue.py` 评估综合体对话 PPL + 生成质量（对照最强单神经元）

### 2.2 下一步：闭环修复（按依赖排序）
1. **运行时装配**（断裂 A/B/C/D）：让 Cortex 支持混合规格或统一装配，加载协作层权重 + shared_embedding，修正域路由 key —— 打通"训练产物→实际对话"
2. **进化规格选择**（断裂 E）：`cortex.add_neuron` 接入 `select_spec`，让进化按错误率自动选 compact/standard/expert
3. **多轮对话评测**（缺口 H）：实现带上下文的真多轮评测脚本

### 2.3 中期
- 综合体接入 API/聊天（缺口 I）
- play→coaction 链路修复（缺口 G）
- 清理死代码（J/K）

### 2.4 远期
- 多 standard 神经元协作验证
- 更长训练（8000→16000 步）
- expert 规格

---

## 🎯 全神经元对话训练（2026-07-31，standard 已成功）

### 背景

协作层训练实验结论：side_channels 协作层只能让神经元协作，**不能让神经元产生新能力**。用对话数据训练协作层（3 epoch，PPL 203），但生成是词汇碎片拼接，完全无法交流。

根本原因：神经元本身（百科/作文训练）没有对话能力，协作层无法创造它们不具备的能力。

### 从头训练对话神经元失败（zh_sft_std0）

用 alpaca-zh SFT 21458 条对话数据从头训练 standard 神经元（8000 步，4 小时）：
- val PPL=57.14（train PPL 3.6，严重过拟合）
- 生成是词汇碎片拼接，无法交流
- **根本原因**：数据量不足（21K 条 vs 116M 参数，数据/参数比 0.18:1，远低于 Chinchilla 20:1）

### 🐛 关键突破：生成路径 bug（2026-07-31）

**所有生成脚本（eval_single_dialogue / eval_dialogue / finetune generate_sample）存在严重 bug**：
- neuron 的 lm_head 输出是 **domain token ID**（zh=20K），不是 general token ID（256K）
- 旧代码把 domain token ID 当 general token ID 追加到输入（查到错误 embedding）→ 用 general_sp 解码 → 中英混杂碎片
- **此 bug 导致所有"生成质量差"评估结果失真**（PPL 评估正确，但生成评估完全错误）

修复方案（domain ID → 文本 → general ID 转换）：
```python
# neuron 输出 domain token ID
piece_text = domain_sp.decode([next_domain_token])
new_general_ids = general_sp.encode(piece_text)
ids = torch.cat([ids, torch.tensor([new_general_ids])], dim=1)
# 解码用 domain_sp，不是 general_sp
text = domain_sp.DecodeIds(generated_domain_ids)
```
- 修复前 zh_std0 生成："重塑day有害物质检查一个 |一样 program激活函数..."（碎片）
- 修复后 zh_std0 生成："当然知道啦！我今天要学习题..."（连贯中文）

### ✅ fine-tune 成功（zh_std0_dialogue）

**关键配置**：lr=5e-4 + 冻结 shared_embedding + 生成 bug 已修复

| 版本 | 配置 | 结果 |
|------|------|------|
| v1 | lr=5e-4, embedding 可训练 | val PPL=149（生成 bug 掩盖了真实进展）|
| v2 | lr=1e-4, 冻结 embedding | val PPL=192（lr 太低，学习慢）|
| **v3** | **lr=5e-4, 冻结 embedding** | **val PPL=95.27 ✅** |

v3 训练曲线：166(step1000) → 130(2000) → 119(3000) → **95.27(4000)**，持续下降

**v3 生成效果**（对话能力已成型）：
```
问：你好，请介绍一下自己
答：那是指您的，所以我无法为您提供帮助。能否提供更多详细信息。请问您想的内容，以便我能为您提供一些帮助、一些建议？
问：什么是人工智能？
答：人工智能是一种人工智能技术，具有计算机能够模拟人类执行、决策和AI...
问：今天天气怎么样？
答：作为一个人工智能助手你的，我无法选择天气晴朗...
```

**教训**：
1. fine-tune 必须冻结 shared_embedding（token 映射不可破坏）
2. lr=5e-4 对 fine-tune 有效（v2 的 1e-4 学习太慢）
3. 所有生成评估必须确认 token ID 空间（domain vs general）

### ✅ Compact 对话训练完成（2026-08-01）

4 个 compact 神经元用同一份 48K alpaca-zh SFT fine-tune（lr=5e-4, 冻结 embedding, 4000 步）：

| 神经元 | 基础 PPL | 最终 val PPL | 状态 |
|--------|---------|-------------|------|
| zh_aug0_dialogue | 39.6 | 88.85 | ✅ 完成 |
| zh_aug1_dialogue | 146.6 | 99.39 | ✅ 完成 |
| zh_aug2_dialogue | 22.5 | 90.43 | ✅ 完成（最强） |
| zh_aug3_dialogue | 71.8 | 102.01 | ✅ 完成 |

> 注：checkpoint 已支持每次 eval 保存 latest（commit 6b94214），resume 从最新 step 继续。

### 🔄 对话协作层训练（2026-08-01，进行中）

**目标**：用对话数据训练 side_channels + 跨规格投影层，让 5 个对话神经元（4×compact_dialogue + 1×std_dialogue）协作交流。

**配置**：`finetune_cross_spec.py --data dialogue --epochs 3 --max_texts 10000`
- 神经元：`ENSEMBLE_DIALOGUE_IDS`（zh_aug0~3_dialogue + zh_std0_dialogue）
- 训练数据：alpaca-zh SFT 对话数据
- 产物：`cross_spec_dialogue.pt`（独立于 simple_zh 训练的 `cross_spec_finetuned.pt`）
- 预计耗时：~13 小时（参考 simple_zh 3 epochs × 264 min）

### ✅ 错误率斜率判别器落地（2026-08-01）

**问题**：NeurogenesisTrigger 原逻辑只看"错误率超阈值"就触发新生，无法区分"数据不足"还是"容量不足"——两者都表现为高错误率，但需要不同的进化动作。

**落地**：[lifecycle.py](file:///e:/taiji-neuron/taiji/resonance/lifecycle.py#L265-L374) NeurogenesisTrigger 增加斜率判别：

| 错误率曲线斜率 | 诊断 | 系统动作 |
|---------------|------|---------|
| < -0.02（持续下降）| `data_insufficient` | 不触发新生，继续喂数据 |
| ≥ -0.02（平台/上升）| `capacity_limited` | 触发 neurogenesis（加神经元）|
| ≤ 阈值 | `healthy` | 无需进化 |
| 历史 < 5 次 | `unknown` | 沿用原计数逻辑（保守触发）|

**生物学对应**：Hebbian 学习饱和（突触塑性到极限）vs 结构性增长（新生神经元）。

**验证**：6 个单元测试全部通过（下降/平台/上升/低错误率/历史不足/平台触发新生）。

**使用方式**：
```python
# 系统自诊断（不触发动作）
state = lifecycle.neurogenesis.diagnose_domain("dialogue")
# → "data_insufficient" / "capacity_limited" / "healthy" / "unknown"

# 记录错误率（斜率判别自动启用）
should_create = lifecycle.neurogenesis.record_domain_error("dialogue", 0.7)
```

### ✅ P1-P3 硬编码修复（2026-08-01，commit 55521f4）

**问题**：硬编码审计发现 P1-P3 级漏洞——WSD 调度公式在 5 个文件重复、采样参数/PROMPTS 散落 4 个 eval 脚本、Muon 配置在 2 个 finetune 脚本重复、核心模块阈值缺依据注释、eval 脚本硬编码 `DEVICE="cpu"`。

| 级别 | 修复项 | 产物 |
|------|--------|------|
| P1 | WSD 调度抽取 | [utils.make_wsd_scheduler](file:///e:/taiji-neuron/scripts/training/utils.py#L322)，替换 5 文件 |
| P1 | 采样参数集中 | `experiment_config.SAMPLING_*`，4 脚本 generate 默认参数替换 |
| P1 | 评估 prompt 集中 | `DIALOGUE_PROMPTS`/`BASE_PROMPTS`，3 eval 脚本替换 |
| P1 | Muon 配置抽取 | [utils.build_muon_adamw_optimizers](file:///e:/taiji-neuron/scripts/training/utils.py#L361)，2 finetune 脚本替换 |
| P2 | 阈值依据注释 | lifecycle.py（ApoptosisTracker/MaturityTracker/NeurogenesisTrigger）+ ensemble.py（构造函数/bias 更新）|
| P3 | eval --device 参数 | 3 eval 脚本支持 `--device` 覆盖 `DEVICE` |

**验证**：py_compile 12 文件通过 + 8 脚本 import 链验证通过。

**设计决策**：
- PROMPTS 分两组：`DIALOGUE_PROMPTS`（"问：答："格式，匹配对话训练数据）vs `BASE_PROMPTS`（纯问题，base 神经元评估）
- `SAMPLING_MAX_TOKENS=100`（折中 single=100/aug_joint=80/dialogue=120）
- 阈值注释标注生物学/工程依据（如 `activation_ratio=0.05` 对应 4 神经元均匀激活 25% 的 1/5）

### ✅ P1-P3 硬编码修复补充（2026-08-01）

**问题**：P1-P3 首轮修复存在 4 处遗漏，覆盖训练流程的 dialogue/cross_spec/评估阶段。

| 漏洞 | 级别 | 修复项 | 产物 |
|------|------|--------|------|
| A | P1 | `eval_aug_joint.py:365` `generate_collab` 函数签名硬编码采样参数 | 替换为 `SAMPLING_*` 常量 |
| B | P1 | `eval_std_neuron.py` 整脚本未替换 | PROMPTS→`BASE_PROMPTS`，采样参数→`SAMPLING_*` |
| C | P3 | 5 脚本仍硬编码 `DEVICE="cpu"` 无 `--device` 覆盖 | `finetune_cross_spec`/`finetune_side_channels`/`finetune_neuron_dialogue`/`eval_std_neuron`/`analyze_side_channels` 添加 `--device` 参数 |
| D | P1 | `finetune_neuron_dialogue.py:106,352` 训练监控 prompt 散落 | 替换为 `DIALOGUE_PROMPTS[0]` |

**验证**：py_compile 6 文件通过 + import 链验证通过。

**设计决策**：
- 漏洞 D 用 `DIALOGUE_PROMPTS[0]`（"问：你好，请介绍一下自己\n答："）替换原简化版 "问：你好\n答："——监控 prompt 与评估 prompt 统一，避免新增冗余常量
- `analyze_side_channels.py` 原无 argparse，本次新增 `--device` 参数支持

---

## 📚 历史实验记录（归档）

> 以下章节按时间线记录历史实验与结论，供追溯参考。当前项目状态以"一、项目全景"和"二、路线图"为准。
> 关键结论已提炼到全景章节，此处保留细节。

### 协作层训练实验（2026-07-30，负向结论）

### 实验

用 alpaca-zh SFT 10000 条对话训练协作层（side_channels + 跨规格投影层），3 epoch，PPL 203。

### 结果

生成完全是词汇碎片拼接，无法交流：
```
问：你好，请介绍一下自己
答：易于 C来帮助你有害物质lo算法通过有更多的ctions自然有关...
```

### 结论

**协作层只能让神经元协作，不能让神经元产生新能力。** 神经元本身没有对话能力时，协作层无法创造对话能力。必须先让神经元具备对话能力，再用协作层融合。

---

## 🔄 Standard 神经元训练（2026-07-29，已完成，生成质量突破）

### 背景

Shared Expert 评估负向结论确认：机制改进无法弥补神经元本身能力不足。
所有评估都指向同一个根本瓶颈：compact 神经元（36M, 4000 步）训练不充分，
导致生成质量差（PPL 好但生成不连贯）。

### 实验目标

训练 standard 规格（116M）单神经元，验证更大容量 + 充分训练能否生成连贯文本。
这是解决生成质量问题的最直接验证路径。

### 训练配置

- **规格**：standard（hidden=768, layers=10, heads=12, kv_heads=4, ~110.5M 参数）
- **数据**：shared_core.jsonl（236K）+ class_b_encyclopedia.jsonl（105K）= 340K 条
- **训练参数**：8000 步，batch 8×grad_accum 4=32，lr=1e-3，dropout=0.2，WSD 调度
- **shared_emb_mode**：train（训练自己的 shared_embedding）
- **side_channels**：4 条（指向 zh_aug0~3 compact 神经元，peer_cfg=compact）
- **耗时**：194.6min（约 3.2 小时）

### 训练结果

| step | val PPL | 趋势 |
|------|---------|------|
| 1000 | 104.37 | 起始 |
| 2000 | 65.57 | 快速下降 |
| 3000 | 57.97 | 持续下降 |
| 4000 | 47.73 | 持续下降 |
| 5000 | 41.64 | 持续下降 |
| 6000 | 37.46 | 持续下降 |
| 7000 | 36.63 | 接近收敛 |
| **8000** | **34.07** | **最终** |

**对比 compact 神经元**：
- compact zh_aug1 best_val_ppl=146.6（最强 compact）
- standard zh_std0 best_val_ppl=34.07
- **standard 比 compact 好 76.8%**

### 生成质量评估（2026-07-29，关键突破）

**评估 PPL**：294.9（评估集分布与训练 val 不同，但生成质量是关键指标）

**生成质量对比**（top-k sampling + repetition penalty + temperature）：

| 神经元 | 生成样本 | 质量评估 |
|--------|---------|---------|
| compact | "天气天气天气..." | 纯重复，无语义 |
| **standard** | "喜欢的小明信！你没有注意到吗？如果你的我们还是找不到，我问记得一次我的朋友..." | **有语义连贯性** |
| **standard** | "树叶洒落在面上，一只老鹰正在低声。鹰们纷纷奔去..." | **有画面感** |
| **standard** | "老师，是什么？它是什么：古代的几何星座" | **有问答结构** |

**关键结论**：
1. ✅ **生成质量突破**：standard 神经元首次生成有语义连贯性的中文文本
2. ✅ **容量是关键**：110.5M standard >> 36M compact，验证"扩大规模"方向正确
3. ✅ **训练充分性重要**：8000 步 + 340K 数据，数据/参数比 3.1:1
4. ⚠️ 仍有不连贯处（如数学题部分），但相比 compact 是质的飞跃

### 技术改进

训练过程中修复了两个 bug：
1. `args.spec` 在 train_parallel 函数中不可用 → 添加 spec 参数
2. checkpoint 只在训练结束时保存 → 改为每次 best val PPL 刷新时立即保存

### 产物

- checkpoint：`data/neurons/neuron_zh_std0.pt`（991MB, standard 规格）
- 训练日志：`logs/train_zh_std0_20260729_184015.log`
- 评估日志：`logs/eval_std_single_*.log`
- 评估脚本：`scripts/training/eval_std_neuron.py`

### 下一步方向

1. ✅ **混合协作评估**：zh_std0 (standard) + zh_aug0~3 (compact) 协作效果 — 已完成，跨规格微调后 PPL=66.3，EMERGE 42.2%（详见下方"跨规格协作最终评估结果"章节）
2. ⏳ **多 standard 神经元训练**：训练 3-4 个 standard 神经元，验证多 standard 协作能否进一步降低 PPL
3. ⏳ **更长训练**：8000→16000 步，看生成质量能否进一步提升
4. ⏳ **更大规格**：如果 standard 效果好，尝试 expert 规格（~300M）

### 混合协作评估结果（2026-07-29，NO_EMERGE）

运行命令：`python -u scripts/training/eval_std_neuron.py --mode mixed --n_eval 50`
评估日志：`logs/eval_std_mixed_*.log`

**个体 PPL**（simple_zh 评估集）：

| 神经元 | 规格 | 评估 PPL | 训练 val PPL |
|--------|------|---------|-------------|
| zh_aug0 | compact | 275.6 | 39.63 |
| zh_aug1 | compact | 149.0（最强） | 146.57 |
| zh_aug2 | compact | 286.2 | 22.53 |
| zh_aug3 | compact | 311.6 | 71.84 |
| zh_std0 | standard | 294.9 | 34.07 |

**协作 PPL**（std_w=0.5 logits 融合）：213.8
**结果**：NO_EMERGE（213.8 >= 149.0）

### 关键结论

1. **简单 logits 融合无效**：50% standard + 50% compact 平均稀释了 zh_aug1 的表现
2. **side_channels 是有效协作机制**：之前 compact 协作 PPL=62.6 << 114.6 靠的是 side_channels，
   不是 logits 融合。简单 logits 融合无法复现 EMERGE
3. **side_channels 无法跨规格**：field_dim 不同（standard=3072, compact=2048），
   per-pair side_channels 投影矩阵维度不匹配
4. **评估集分布很重要**：zh_std0 训练数据（shared_core+百科）与 simple_zh 分布差异大，
   导致评估 PPL=294.9 远高于训练 val PPL=34.07
5. **训练 val PPL ≠ 评估 PPL**：zh_aug2 训练 val PPL=22.53 但评估 PPL=286.2，
   说明训练 val 集与 simple_zh 分布也不同

### 后续方向

要验证多规格协作 EMERGE，有两条路径：

**路径 A：多 standard 神经元 + side_channels**（推荐）
- 训练 3 个 additional standard 神经元（差异化数据）
- 同规格（field_dim=3072）可通过 side_channels 协作
- side_channels 微调后验证 EMERGE
- 预计耗时：3×3.2 小时训练 + 14 小时微调 = ~24 小时

**路径 B：Field Projector 跨规格协作**
- 实现 Field Projector: Linear(field_dim -> unified_field_dim)
- 让 standard 和 compact 通过投影到统一 field_dim 协作
- 需要额外架构改动 + 微调
- 优势：能利用现有 compact 神经元，不需要全部重新训练

### 跨规格 side_channels 协作突破（2026-07-29，EMERGE 确认）

**关键发现**：side_channels 本身已支持跨规格（`establish_side_channel` 用 `src_dim=peer.field_dim, dst_dim=self.hidden_size`），但 ResonanceField 是单一维度，无法容纳不同 field_dim 的向量。

**解决方案**：在 ensemble.py 添加跨规格投影层：
1. **正向投影**（field_dim → unified_dim）：neuron 写入 field 前投影到统一维度
2. **反向投影**（unified_dim → field_dim）：round 2+ conditioning 时将 field.state 投影回 neuron.field_dim

**评估结果**（5 神经元：4×compact + 1×standard）：

| 模式 | 协作 PPL | 对比最强个体 |
|------|---------|-------------|
| 4×compact（v2 baseline） | 62.6 | -45.3% EMERGE |
| **4×compact + 1×standard（side_channels）** | **96.5** | **-35.3% EMERGE** |
| 4×compact + 1×standard（logits 融合） | 213.8 | NO_EMERGE |

**融合权重**：zh_aug1:0.471（最强 compact 主导）, zh_std0:0.194（standard 第二）, zh_aug0:0.180, zh_aug2:0.092, zh_aug3:0.063

**关键结论**：
1. ✅ **跨规格 side_channels 协作成功**：大神经元 + 小神经元联合验证通过
2. ✅ **EMERGE 确认**：协作 PPL=96.5 < 最强个体 149.0，涌现 35.3%
3. ⚠️ **投影层未训练引入噪声**：PPL 96.5 > 纯 compact 62.6，因为跨规格投影层是随机初始化
4. ✅ **生成质量改善**："树叶洒落下来，鸟儿悠闲地翻。一只小兔子在树枝间，毛茸" — 有画面感

**技术产物**：
- `taiji/resonance/ensemble.py`：添加 `_cross_spec_projectors` 和 `_cross_spec_back_projectors`
- `_project_vec()` 方法：统一处理 write/score 时的维度投影
- `_parallel_forward` 中的 `_forward_neuron`：round 2+ conditioning 时反向投影 field.state

**下一步**：微调跨规格投影层 + side_channels，消除投影噪声，使 PPL 从 96.5 → 接近或超越 62.6

### 跨规格协作最终评估结果（2026-07-30，EMERGE 42.2%）

**目标**：通过联合微调 side_channels + 跨规格投影层，消除随机投影噪声，验证混合规格协作能否超越纯 compact 协作。

**训练配置**：

| 配置 | 值 |
|------|-----|
| 神经元 | 4×compact（zh_aug0~3）+ 1×standard（zh_std0） |
| 训练数据 | simple_zh 10000 条 |
| epochs | 3 |
| batch_size | 4 |
| lr | 1e-3（Muon + AdamW 混合优化器） |
| 可训练参数 | side_channels 25.17M（2D） + 跨规格投影层 50.33M（2D） + scale 20（0D） |
| 跨规格投影层 | 4 正向（field_dim→3072）+ 4 反向（3072→field_dim） |
| LR 调度 | warmup 100 步 + cosine decay（最后 20%） |
| 总耗时 | ~13.2 小时（3 epochs × 264 min） |

**训练 PPL 趋势**：

| Epoch | avg PPL | 趋势 |
|-------|---------|------|
| 1 | 116.9 | 144.2 → 116.9（快速下降） |
| 2 | 103.2 | 95.1 起步 → 103.0 收敛 |
| 3 | 97.5 | 95.1 起步 → 97.5 收敛（趋稳） |

**最终评估结果**（simple_zh 100 条，eval_aug_joint.py --include_std）：

| 神经元 | 规格 | solo PPL | 融合权重 |
|--------|------|---------|---------|
| zh_aug0 | compact | 211.6 | 0.233 |
| **zh_aug1** | **compact** | **114.6（最强个体）** | **0.494（主导）** |
| zh_aug2 | compact | 225.3 | 0.120 |
| zh_aug3 | compact | 246.9 | 0.061 |
| zh_std0 | standard | 229.1 | 0.091 |
| **协作** | **混合** | **66.3** | - |

**共振分**：zh_aug1:-0.362（最强）, zh_std0:-0.370, zh_aug3:-0.399, zh_aug2:-0.434, zh_aug0:-0.517

**EMERGE 对比**：

| 模式 | 协作 PPL | 对比最强个体 114.6 |
|------|---------|---------------------|
| 4×compact + 1×standard（随机投影） | 96.5 | -15.8% EMERGE |
| 4×compact + 1×standard（logits 融合） | 213.8 | NO_EMERGE |
| **4×compact + 1×standard（跨规格微调）** | **66.3** | **-42.2% EMERGE** |
| 4×compact（纯 compact v2 baseline） | 62.6 | -45.3% EMERGE |

**关键结论**：
1. ✅ **跨规格微调成功**：微调后 PPL 从 96.5（随机投影）降至 66.3，降幅 31.3%
2. ✅ **EMERGE 42.2%**：协作 PPL 66.3 << 最强个体 114.6，强 EMERGE 现象确认
3. ✅ **接近纯 compact 协作**：66.3 vs 62.6（纯 compact），差距仅 5.9%，跨规格投影噪声基本消除
4. ✅ **standard 神经元有效参与**：zh_std0 获得 9.1% 融合权重，作为辅助信号贡献协作
5. ⚠️ **standard 未成为主导**：zh_std0 融合权重仅 0.091（vs zh_aug1 的 0.494），原因是评估集 simple_zh 与 zh_std0 训练数据（百科+shared_core）分布差异大，solo PPL=229.1 远高于训练 val PPL=34.07

**生成质量对比**（prompt: "在公园里，阳光透过"）：

| 神经元 | 生成样本 |
|--------|---------|
| 协作 | "树叶洒落下来，几阳光透得。小兔子兴奋地好看书。主人看到小兔子说：'我来你愿意分享！我们可以一起吃果味！'" |
| zh_aug1（最强个体） | "树叶，洒落下的地面，是一个着，。小明和朋友们一起玩耍，阳光轻轻洒落在地上..." |
| zh_std0 | "树叶洒在一个阳光明媚下午，几只小猫们在公园里，突然被一只小兔子..." |

**关键洞察**：
- 协作生成具备画面感（"树叶洒落"、"小兔子兴奋"）和叙事结构（对话+动作），优于多数个体
- zh_aug1（compact）仍是主导，因为 simple_zh 评估集与 zh_aug1 训练数据分布更接近
- standard 神经元在评估集上表现不佳是**域偏移**问题（训练数据 vs 评估数据分布不同），不是能力问题（训练 val PPL=34.07 是所有神经元中最好的）

**技术产物**：
- 微调权重：`data/neurons/cross_spec_finetuned.pt`（含 side_channels + 跨规格投影层）
- 训练 checkpoint：`data/neurons/cross_spec_finetuned.ckpt.pt`
- 训练脚本：`scripts/training/finetune_cross_spec.py`
- 评估脚本：`scripts/training/eval_aug_joint.py`（添加 `_load_cross_spec_weights`）
- 训练日志：`logs/finetune_cross_spec_20260729_233044.log`
- 训练历史：`logs/finetune_cross_spec_history.json`（149 条记录）
- 评估日志：`logs/eval_cross_spec_finetuned_20260730_124105.log`

---

## 🎉 重大里程碑（2026-07-29）：EMERGE 现象确认

### 实验结果

**4 神经元协作 side_channels 微调（6 epochs，14950 步，~14 小时）**

| 神经元 | solo PPL | 融合权重 |
|--------|---------|---------|
| zh_aug0 | 211.6 | 0.250 |
| zh_aug1 | 114.6（最强个体） | 0.555（主导） |
| zh_aug2 | 225.3 | 0.129 |
| zh_aug3 | 246.9 | 0.066 |
| **协作** | **62.6** | - |

### 关键指标

- **协作 PPL: 62.6**
- **最强个体 PPL: 114.6**
- **EMERGE 幅度: 协作比最强个体好 45.3%**

### 技术配置

- 神经元规格：4× compact（36M 参数/个，共 144M）
- 可训练参数：side_channels 12.58M + scale 12 个
- 优化器：Muon（ns_steps=5, momentum=0.95, nesterov=True）
- 学习率调度：warmup(100步) + constant + cosine decay(最后 20%)
- side_channels 调制：乘性门控 `h = h * (1 + tanh(proj))`
- field_conditioning：False（跳过未训练的 field_read_layers）
- max_rounds：2（round 1 独立，round 2 带 side_signals）
- 数据：simple_zh 10000 条，6 epochs

### 意义

这是态极架构的**关键验证点**：多个小型神经元通过 side_channels 协作，涌现出超越最强个体的能力。验证了"小神经元协同工作匹配大模型"的核心设计理念。

### 生成质量

PPL 指标确认协作有效，但生成文本仍有重复（所有神经元共性问题）。原因是 compact 神经元训练不充分（CPU 限制，数据/参数比不足）。后续需要：
1. 更充分的神经元训练（更多数据，更多步数）
2. 改进生成策略（sampling, repetition penalty）
3. 扩大规模（更多神经元，更大规格）

### 生成质量评估（2026-07-29，top-k sampling 改进后）

**PPL 结果**（v2 baseline，已恢复）：
- 个体 [zh_aug0]: PPL=211.6
- 个体 [zh_aug1]: PPL=114.6（最强个体）
- 个体 [zh_aug2]: PPL=225.3
- 个体 [zh_aug3]: PPL=246.9
- **协作 PPL=62.6**（EMERGE 协作比最强个体好 45.3%）

**生成质量观察**：
- top-k sampling (k=40) + repetition penalty (1.2) + temperature (0.8) **消除了机械重复**
  （之前是"天气天气天气..."纯重复，现在是有变化的生成）
- 但生成文本仍**语义不连贯**，有乱码、断裂、混合多种风格
- 这是典型的"PPL 好但生成差"问题，根因是 compact 神经元训练不充分

**根本瓶颈确认**：神经元训练不充分是当前所有生成质量问题的根因。架构改进（side_channels、
Auxiliary-loss-free balancing、sampling 策略）已到位，但无法弥补神经元本身能力不足。

### 产物

- 微调权重：`data/neurons/side_channels_finetuned.pt`（v2 baseline, PPL=62.6）
- v2 baseline 备份：`data/neurons/side_channels_finetuned_v2_baseline.pt`
- 训练历史：`logs/finetune_side_channels_history.json`（299 条记录）
- 评估日志：`logs/eval_aug_joint_sampling_20260729_132638.log`

---

## 🔧 Auxiliary-loss-free balancing 实施（2026-07-29）

### 背景

EMERGE 已确认（协作 PPL 62.6 << 最强个体 114.6），但 plans 中标记的"Auxiliary-loss-free balancing"
此前只是框架（scale + bias buffer 已注册，但**启发式 bias 更新逻辑未实现**）。本次完成完整实施。

### 借鉴来源

DeepSeek V3 的 Auxiliary-loss-free Load Balancing：不通过辅助损失，而是用非梯度启发式更新
bias 项，动态平衡各专家（channel）利用率，解决"死通道"问题。

### 实施细节

**`taiji/resonance/neuron.py`**：
1. `__init__`：添加 `_channel_usage: Dict[str, float]` 运行时统计字段（不持久化）
2. `forward` Step 4：side_signal 处理时记录 `proj.detach().abs().mean().item()` 到 `_channel_usage`
3. 新增 `update_channel_bias(update_rate=0.1)`：根据 usage 偏离平均的程度更新 bias
   - `delta = update_rate * (avg_usage - channel_usage)`
   - 低 usage → 正 bias（鼓励激活）
   - 高 usage → 负 bias（抑制过度激活）
4. 新增 `get_channel_usage_stats()`：返回当前 usage 统计（用于日志/诊断）

**`scripts/training/finetune_side_channels.py`**：
1. 每 50 步（`BIAS_UPDATE_EVERY=50`）调用 `update_channel_bias(update_rate=0.1)`
2. 每 50 步（`LOG_EVERY`）输出 channel usage 诊断：avg/min/max/dead count
   - 死通道判定：`usage < avg * 0.1`

### 验证

端到端测试通过：
- 单 channel：avg=usage → delta=0（预期，无竞争）
- 双 channel（强/弱信号）：强 channel 获得负 bias (-8.65)，弱 channel 获得正 bias (+8.65)

### 短训练验证（2026-07-29）

运行 300 条文本 × 1 epoch（74 步）验证 bias 更新机制：

```
[bias update] step 50: 12 channels, total_delta=0.0104
Epoch 1/1 step 50: loss=5.0741 PPL=159.8 [50/74 ETA 1.9min]
  [channels] usage avg=0.3926 min=0.3683 max=0.4160 dead=0/12
```

**关键发现**：
1. ✅ bias 更新机制工作正常（step 50 触发，delta 计算 correct）
2. ✅ channel usage 诊断输出正常（avg/min/max/dead count）
3. ✅ **当前无死通道**（dead=0/12）—— v2 修复（scale=50 + post-norm）已解决死通道问题
4. ✅ bias 更新在 usage 均匀时幅度很小（total_delta=0.0104），不会破坏已训练的平衡

**结论**：死通道问题已被 v2 修复解决，Auxiliary-loss-free balancing 作为"保险"机制存在，
在当前 usage 分布均匀的情况下对 PPL 影响可忽略。无需为此重新完整训练（保留 v2 baseline PPL=62.6）。

---

## 🧠 Shared Expert 机制实施（2026-07-29，已完成，结论：负向）

### 背景

生成质量评估确认根本瓶颈：compact 神经元训练不充分导致生成不连贯（PPL 好但生成差）。
借鉴 Kimi K3 / DeepSeek V3 的 Shared Expert 机制，添加 always-active 的 general 神经元
提供基础语言能力，与域特定神经元互补。

### 借鉴来源

Kimi K3 / DeepSeek V3 的 Shared Expert：一个 always-active 的通用专家，与稀疏激活的
域特定专家协同，提供基础能力保障。

### 实施进度

**1. general 神经元训练**（✅ 已完成）
- 数据：`shared_core.jsonl`（236K 条通用核心数据）
- 规格：36M compact，train 模式（保存自己的 shared_embedding）
- 训练参数：4000 步，batch 8×grad_accum 4=32，lr=1e-3，dropout=0.2
- 结果：best_val_PPL=148.80@step4000，耗时 57.8min
- 训练日志：`logs/train_zh_general_20260729.log`

**2. ensemble Shared Expert 架构**（✅ 已完成）
- `ResonanceEnsemble.__init__` 添加 `shared_expert_id` 和 `shared_expert_weight` 参数
- `forward()` 中 shared expert 始终加入 `active_ids`（不受路由/精简模式影响）
- 最终融合后重新加权：`final = sw * shared_logits + (1-sw) * original_fused`
- 默认 `shared_expert_weight=0.3`（general 神经元获得 30% 固定权重）

**3. 评估脚本支持**（✅ 已完成）
- `eval_aug_joint.py` 添加 `--shared_expert` 和 `--shared_expert_weight` 参数
- `load_aug_neurons()` 支持 `include_shared_expert` 加载 general 神经元
- `eval_ppl()` 和 `eval_generation()` 传递 shared_expert 配置到 ensemble

### 评估结果（2026-07-29，负向）

运行命令：`python -u scripts/training/eval_aug_joint.py --shared_expert --shared_expert_weight 0.3`
评估日志：`logs/eval_shared_expert_20260729_145858.log`

**PPL 对比**：

| 模式 | 协作 PPL | 对比 baseline |
|------|---------|--------------|
| 无 Shared Expert（v2 baseline） | 62.6 | - |
| **Shared Expert (w=0.3)** | **108.6** | **恶化 +73.6%** |

**个体 PPL**：
- zh_aug0: 211.6 | zh_aug1: 114.6（最强个体）| zh_aug2: 225.3 | zh_aug3: 246.9
- **zh_general: 257.5（最弱）** ← 关键问题

**融合权重**：zh_aug1:0.320, zh_general:0.300, zh_aug0:0.122, zh_aug2:0.062, zh_aug3:0.032

### 结论与教训

**Shared Expert 机制在当前实施下负向**，反而降低了协作质量。

**根本原因**：Shared Expert 机制的前提是 general 神经元必须足够强（提供基础能力保障），
但实际 zh_general 神经元训练不充分（best_val_PPL=148.80，评估 PPL=257.5），反而比所有
aug 神经元都差。强制给它 30% 固定权重稀释了 zh_aug1（最强个体）的主导作用。

**关键教训**：
1. **借鉴机制不能盲目照搬**：Shared Expert 在 Kimi K3/DeepSeek V3 中有效，是因为它们的
   general 专家训练充分（数万亿 token）。在小规模 compact 神经元（36M, 4000 步）上，
   general 神经元反而成为最弱环节
2. **机制有效性依赖前置条件**：Shared Expert 要求 general ≥ 域特定神经元的能力，
   否则会拖累整体
3. **固定权重的风险**：30% 固定权重缺乏自适应，无论 general 神经元质量如何都会强制分配，
   应该改为基于神经元实际能力的动态权重

### 后续方向

Shared Expert 机制暂不启用（保持 v2 baseline PPL=62.6 作为最佳协作结果）。
若要重新启用，需要：
1. 大幅增加 general 神经元训练数据量和步数（达到或超过 aug 神经元水平）
2. 改为动态权重（基于 general 神经元实际 PPL 自适应调整）
3. 或放弃固定权重，让 general 神经元参与正常的共振评分竞争

---

## 🧹 项目整理记录（2026-07-28）

### 清理总结

删除 **43 个废弃脚本**，抽取 **1 个工具模块**，项目结构从 67 个训练脚本精简到 24 个。

### 删除的废弃脚本（43 个）

**错误方向训练脚本（14 个）**：
- train_single_long.py, train_collaboration.py, train_individual_neurons.py, train_multi_zh.py
- joint_train_p7.py, train_v3_neuron.py, verify_v3_quick.py, pipeline_v3_full.py
- pipeline_verify_v3.py, joint_and_generate_v3.py, train_tinystories_collab.py
- train_neuron.py, train_standard_leader.py, train_cortex_joint.py（工具函数已抽取到 utils.py）

**错误方向评估/路由脚本（13 个）**：
- eval_leader.py, eval_individual.py, eval_joint.py, eval_single.py, eval_simple_neuron.py
- eval_collab.py, eval_all_singles.py, eval_gen_quality.py
- verify_routing_accuracy.py, verify_routing_16c.py, verify_routing_short.py
- verify_fingerprint_routing.py, verify_contrastive_fix.py, verify_p7_resonance.py

**一次性诊断脚本（16 个）**：
- diagnose_argmax.py, diagnose_generation.py, diagnose_rounds.py, diag_collab.py, diag_quality.py
- analyze_argmax_ceiling.py, check_argmax.py, integrate_verify.py, test_collaboration.py
- verify_scaled_training.py, gen_test.py, gen_constrained.py, quick_gen_test.py
- audit_neurons.py, test_p7_e2e.py

### 新增工具模块

**`scripts/training/utils.py`**：从 train_neuron.py、train_standard_leader.py、train_cortex_joint.py 抽取的工具函数：
- 常量：OUTPUT_DIR, SHARED_EMBEDDING_PATH, SIMPLE_ZH_DIR, DATA_DIR 等
- tokenizer：load_domain_tokenizer, load_general_tokenizer
- 数据加载：load_domain_texts, load_all_texts, load_simple_zh_texts
- shared_embedding：create_shared_embedding, load_or_create_shared_embedding, save_shared_embedding
- 采样器：SequentialSampler

### 当前活跃脚本（24 个）

**训练核心（5 个）**：
- `train_compact_parallel.py` — 训练 zh_aug0~3 compact 神经元
- `finetune_side_channels.py` — side_channels 联合微调（当前运行中）
- `eval_aug_joint.py` — 评估个体 vs 协作 PPL
- `run_parallel_aug.ps1` — 并行训练 PowerShell 脚本
- `analyze_side_channels.py` — side_channels 死通道诊断

**工具/数据脚本（6 个）**：
- build_domain_tokenizers.py, download_simple_zh.py, download_tinystories.py
- download_zh_data.py, split_simple_zh.py, tokenize_sft_p7.py

**已完成消融实验（2 个，保留作参考）**：
- train_tinystories.py（实验 A baseline, PPL=16.6）
- train_tinystories_field.py（实验 B field-augmented, PPL=14.3）

**P8 多模态/未来方向（4 个）**：
- train_encodec.py, train_vqvae.py, train_video.py, train_neurons_from_scratch.py

**简化参考（1 个）**：
- train_compact_simple.py（train_compact_parallel.py 的简化版）

**集成回归测试（18 个 verify_*.py）**：
- 验证 apoptosis/neurogenesis/cortex chat/http api/state persistence 等运行时闭环

### 项目结构清理原则

1. **错误方向产物立即清理**：避免后续开发被误导
2. **工具函数先抽取再删除**：防止破坏活跃脚本依赖
3. **一次性诊断脚本用完即删**：结论写入 plans 文档即可
4. **集成测试保留**：验证运行时闭环，有长期价值
5. **消融实验保留**：作为历史参考，结论已写入 plans

---

## 🚨 紧急更新（2026-07-26）：架构级 bug 修复 — 之前所有训练无效

### bug 发现

**ResonanceNeuron.forward 的因果掩码缺失**：调用 `block.attention(h_normed)` 时没传
`mask` 参数，导致 GQA 的 `is_causal=(mask is not None) and (seqlen>1)` 为 False →
**双向注意力**（位置 K 能看到所有未来 token）。

### 决定性诊断证据

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| Shift teacher-forcing（完整序列 forward） | 100% | 0% |
| 逐步预测（只给前 K 个 token） | 0% | 0% |
| 完整 forward vs 逐步 forward 一致性 | 12/12 不一致 | ✅ 一致 |

- 修复前 100% 是**假象**（模型偷看未来 token）
- 修复后 0% 是**真实性能**（现有权重依赖偷看，学坏了）

### 影响（之前所有结果无效）

- ❌ PPL 1.79~3.85：虚低（偷看未来使预测虚高准确）
- ❌ argmax 73~74%：虚高（同样原因）
- ❌ Teacher-forcing 100%：假象
- ❌ 所有神经元 ckpt（zh_j0~j9, zh_leader0, zh_simple0, zh_par1, zh_par2）：权重学坏
- ❌ "数据不足是根因"的结论：错误（数据是次要因素，架构 bug 才是主因）
- ❌ "容量瓶颈"假说：错误（之前 standard 族长 argmax 73.8% 是虚高）

### 修复

`taiji/resonance/neuron.py` forward Step 2：创建标准下三角 causal mask 并传给
`block.attention(h_normed, mask=causal_mask)`。commit 2921b43。

### 新方向（修复后路线）

1. ✅ bug 已修复（commit 2921b43）
2. ✅ 架构底层审计完成（5 组件逐一验证，因果掩码是唯一关键 bug）
3. ✅ 数据质量重新评估（数据清洗有效，但"数据量不足"仍是真约束——见 0.4 最新状态）
4. ✅ 5×compact 联合训练验证（simple_zh 50K 数据，3000 步，完成）
   - 协作 PPL=173.8 vs 最强个体 653.2（涌现-73.4%，确认因果掩码修复后协作机制仍有效）
   - 但生成仍乱码 → PPL=173.8 远高于 TinyStories baseline 16.6，数据/参数比 0.008:1 太低
5. 🔄 4×compact 并行独立训练（差异化数据，全量 simple_zh，8000 步/人，进行中）
   - 策略转变：联合训练（1/5 梯度每人）→ 独立训练（100% 梯度每人），先验证单神经元能否生成连贯
   - zh_full0（全量 787K + train embed）+ zh_full1/2/3（差异化 class 文件 + frozen embed）并行
   - 当前：zh_full0 step 4400/8000, val PPL=84.93, 训练 PPL=4.2
   - 修复：GBK 编码崩溃 → commit 8a6f341（emoji → ASCII）
6. ⏳ 训练完成后评估：单神经元生成质量 + 四神经元协作效果
7. ⏳ 如果仍然乱码 → RoPE theta=500000 是下一嫌疑（审计中标记的唯一非 bug 问题）

### 架构底层审计结果（2026-07-26）

对 5 个基础组件逐一验证，确认因果掩码是**唯一**关键 bug：

| 组件 | 状态 | 依据 |
|------|------|------|
| ① 因果掩码 | ✅ 已修复正确 | neuron.py 创建下三角掩码传给 block.attention；layers.py is_causal=True 启用 SDPA 内置因果掩码 |
| ② RoPE 旋转位置编码 | ✅ 正确 | 频率公式 1/(theta^(2i/dim)) 正确；apply_rotary_emb 交错复数旋转数学正确；Q/K 同变换；kv_cache start_pos 逻辑正确 |
| ③ Tokenization 对齐 | ✅ 正确 | build_position_alignment 字符跨度重叠映射；训练 shift CE / 推理"预测 domain→解码→编码 general 追加"一致 |
| ④ compute_logits | ✅ 正确 | 简单 lm_head(h) 投影，无问题 |
| ⑤ 前向流程 | ✅ 正确（修复后）| embed_adapter → 因果掩码 Transformer → field conditioning(round 2+) → norm → field write → logits |

**非 bug 问题（不影响正确性）**：
- ⚠️ RoPE theta=500000（LLaMA 3 长上下文值）对 compact(36M, block_size=128) 偏大
  - theta=10000（GPT-2/nanoGPT 标准）对短序列位置分辨率更好
  - 不影响正确性，但可能影响小模型学习效率
  - 等 zh_fix0 联合训练结果出来后再决定是否调整

### 数据质量重新评估（2026-07-26）

| 维度 | 之前结论 | 重新评估 |
|------|----------|----------|
| 数据清洗 | 0% 精确重复，3.0% 前缀重复 | ✅ 审计有效，数据本身干净 |
| "数据不足是根因" | 成立 | ❌ 错误，建立在 bug 失真指标上 |
| 数据/参数比 | compact 0.18:1（维基） | simple_zh 8.0:1，已改善 |
| 数据复杂度 | 维基太复杂 | simple_zh 小学水平，匹配 36M |

**核心洞察**：bug 存在时所有指标失真（PPL 虚低、argmax 虚高），无法判断数据质量真实影响。
现在 bug 修复后，联合训练才能给出第一份真实评估。

### 教训

- **诊断顺序错误**：之前先怀疑数据/容量/采样器，最后才检查因果掩码
- **teacher-forcing 高准确率不等于模型好**：必须验证自回归生成
- **架构修改必须消融验证**：因果掩码是基础组件，但从没验证过是否生效
- **PPL 虚低是危险信号**：PPL 1.79（远低于 TinyStories baseline 16.6）应该是警钟
- **修复后第一件事是验证修复生效**：初始 PPL=8506（而非 1.79）确认因果掩码生效

---

## 归档：项目状态总览（旧导航，2026-07-26，已被"一、项目全景"取代）

> ⚠️ 本仪表盘为 2026-07-26 旧导航，状态已过时，仅保留历史数据供追溯。

> **因果掩码 bug 修复后，所有指标基于真实因果训练重新评估。**
> 四个问题有严格依赖顺序：因果掩码 → 数据量 → 架构优化 → 协作涌现。

### 0.1 关键路径（修复后，2026-07-26）

```
TinyStories 验证实验（✅ 全部完成，2026-07-25）→ 验证基础 pipeline 无 bug
  实验 A：纯 transformer baseline + TinyStories ✅ PPL=16.6，生成连贯故事
  实验 B：field-augmented + TinyStories ✅ PPL=14.3，field 组件有用（-13.9%）
  实验 C：多神经元协作 + TinyStories ❌ PPL=16.7，logits 平均是坏的协作方式
     ↓
因果掩码 bug 发现 + 修复（✅ 2026-07-26，commit 2921b43）
  ResonanceNeuron.forward 未传 causal mask → 双向注意力 → 所有之前结果无效
  修复后同一权重 argmax=0%（真实性能），PPL=8506（真实随机初始化）
  架构审计：5 组件逐一验证，因果掩码是唯一关键 bug
     ↓
5×compact 联合训练（✅ 完成，50K 数据，3000 步）
  协作 PPL=173.8 vs 最强个体 653.2 → 涌现确认（修复后协作机制仍有效）
  但生成仍乱码（全标点重复）→ PPL 仍太高（173.8 >> 16.6）
  → 根因：数据/参数比 0.008:1 极低（每人 10K 条），联合训练每人获 1/5 梯度
     ↓
4×compact 并行独立训练（🔄 进行中）
  策略：独立训练 100% 梯度 + 全量 simple_zh 数据 + 差异化数据分配
  zh_full0: 787K 全量（train embed）| zh_full1: 248K 中文 | zh_full2: 340K 百科 | zh_full3: 670K 故事
  目标：先验证单神经元能否生成连贯，再验证协作
     ↓
（若生成连贯）验证多神经元协作（forward_train + 残差融合）
     ↓
（若仍乱码）RoPE theta=500000 调整 → 再评估
```

### 0.2 当前瓶颈分析（2026-07-26）

| 瓶颈 | 状态 | 证据 |
|------|------|------|
| 因果掩码 | ✅ 已修复 | 唯一架构 bug，commit 2921b43 |
| 数据量 | 🔄 验证中 | TinyStories 12M+PPL=16.6（33:1），zh_full0 36M+4.4:1 → 差距尚待量化 |
| 架构优化 | ⏳ 待数据验证后 | field 组件 TinyStories 消融确认有用（-13.9%），但 RoPE theta 待调 |
| 协作涌现 | ⏳ 待单神经元验证后 | 5×compact 联合训练涌现确认（协作 PPL < 最强个体 -73%），但需先有连贯单神经元 |

### 0.3 当前运行的实验（2026-07-26）

**上一轮 4 路并行训练已停止**（进程中断，未保存 ckpt）。

**问题诊断**：
- 8000 步只看 256K 样本，数据利用率 <33%（zh_full0 仅 32.5%）
- val PPL=84.93 > train PPL=66.7（exp(4.2)），已过拟合
- shared_embedding.pt 是 7/25 旧版（因果掩码修复前，学坏的），zh_full1/2/3 frozen 模式加载了学坏的 embedding

**新方向：数据多样性增强（方向 B，保持简单数据）**（2026-07-26）：

策略：不增加数据复杂度（仍用 simple_zh 小学水平），通过数据增强 + 正则化提升多样性。

| 改动 | 旧值 | 新值 | 目的 |
|------|------|------|------|
| 数据增强 | 无 | 随机截断(50-100%) + 片段拼接(30%概率) | 防 epoch 间逐字记忆 |
| dropout | 0.1 | 0.2 | 加大正则化 |
| 训练步数 | 8000 | 16000 | 让数据真正看完一遍（zh_full0 需 ~24600 步看完整 epoch） |
| shared_embedding | 旧版(学坏) | 重新训练 | 因果掩码修复后干净起点 |

**串行依赖**：zh_full0（train 模式）→ 完成后自动保存 shared_embedding → 启动 3 路并行 zh_full1/2/3（frozen 模式）。

**当前状态**：
- ✅ 4 路并行训练完成（2026-07-27）
  - zh_aug0: 787K 全量, best_val_ppl=39.6（过拟合，单字重复）
  - zh_aug1: 249K 中文, best_val_ppl=146.6（生成乱码）
  - zh_aug2: 341K 百科, best_val_ppl=22.5（最连贯，知识类数据质量优势）
  - zh_aug3: 670K 故事, best_val_ppl=71.8（半连贯）
  - 关键发现：数据质量 > 数据量（百科 341K 战胜全量 787K）
  - 关键修复：训练脚本保存 per-neuron shared_embedding，避免覆盖导致 embedding 空间不一致

### 0.4 side_channels 联合微调（2026-07-28 EMERGE 已确认）

**目标**：协作 PPL < 最强个体 solo PPL（同一评估数据），验证 EMERGE 现象。

> **⚠️ 目标修正（2026-07-28）**：原目标"协作 PPL < 114"基于 zh_aug1 在自己训练数据
> （百科）上的 val_ppl=146.57，但 side_channels 微调用 simple_zh 数据评估。在 simple_zh 上
> zh_aug1 的 solo PPL=157.7。EMERGE 的正确定义是**协作 PPL < 最强个体在同一评估数据
> 上的 solo PPL**。

#### EMERGE 确认（2026-07-28）

在 simple_zh 评估数据上：

| 指标 | PPL | 说明 |
|------|-----|------|
| zh_aug1 solo（最强个体） | 157.7 | simple_zh 上的最强个体 |
| zh_aug2 solo | 182.1 | 百科数据训练，simple_zh 上表现中等 |
| zh_aug0 solo | 205.9 | 全量数据训练，simple_zh 上过拟合 |
| zh_aug3 solo | 248.1 | 故事数据训练，simple_zh 上最弱 |
| **协作 PPL（训练前）** | **158.3** | 已 < 最强个体 175.3（EMERGE 出现） |
| **协作 PPL（v1 step 350）** | **131-134** | 乘性门控 + Muon 优化器，PPL 停滞 |
| **协作 PPL（v2 step 300）** | **124.9** | 突破停滞！post-norm + scale + bias |

**结论**：EMERGE 现象已确认。v2 修复后协作 PPL (124.9) 比最强个体 solo PPL (175.3) 低 29%。

#### v1 → v2 突破：side_channels 架构修复（2026-07-28）

v1 PPL 在 131-134 停滞 6 个数据点。诊断发现**所有 12 条通道都是"死"的**：
- gate_deviation = 0.014（gate ≈ 1.0，无调制效果）
- proj_mean = 0.008（每维度投影值极小）
- gate_range = [0.944, 1.064]（仅 ±6% 调制）

**三个根因及修复**：

1. **side_channels 在 RMSNorm 之前应用** → norm 抵消乘性调制
   - 修复：移到 norm 之后，调制直接作用于 logits 输入
2. **投影值太小** → tanh(0.008) ≈ 0.008，gate 几乎等于 1.0
   - 修复：添加可学习 scale 参数（init=50），proj×50 → tanh(0.4) ≈ 0.38
3. **无动态平衡机制** → 部分 channel 可能一直弱
   - 修复：实现 Auxiliary-loss-free balancing（借鉴 DeepSeek V3）
   - 启发式 bias 更新：低利用率 channel 获得正 bias，不通过梯度

**v2 PPL 趋势**：158.3 → 144.7 → 133.1 → **128.9** → 128.0 → 127.3 → **124.9**（持续下降）

#### 根因分析：solo_ppl "异常高"的误解

之前误以为协作环境中 solo_ppl 异常高（zh_aug2 val_ppl=22.53 但协作中 solo_ppl=182.1），
是 field conditioning 噪声导致。实际调查发现：

1. **field_read_layers 未训练**：独立训练时 field_state=None，field_read_layers 权重随机。
   已修复：ensemble.forward 添加 `field_conditioning=False` 选项，round 2 跳过 field conditioning。
2. **solo_ppl "异常"实为域偏移**：zh_aug2 用百科数据训练（val_ppl=22.53），但在 simple_zh
   上评估 PPL=182。这是正常的域偏移，不是 bug。所有神经元在 simple_zh 上 PPL 都远高于
   自己训练数据上的 PPL。

#### 训练配置

| 配置 | 值 |
|------|------|
| 训练数据 | simple_zh 10000 条 |
| epochs | 6 |
| batch_size | 4 |
| lr | 1e-3（Muon + AdamW 混合优化器） |
| 可训练参数 | 12.58M（side_channels 2D） + 12（scale 0D） |
| 协作机制 | 12 条 excite 通道，post-norm 乘性门控，scale 放大 |
| field_conditioning | False（跳过未训练的 field_read_layers） |
| Auxiliary-loss-free balancing | 启发式 bias 更新，每 50 步 |
| LR 调度 | warmup 100 步 + cosine decay（最后 20%） |
| 工程 | 日志 tee + 每 epoch checkpoint + 断点续训 |

**当前日志**：`logs/finetune_side_channels_20260728_131347.log`
**脚本**：`scripts/training/finetune_side_channels.py`

### 0.5 Playbook 合规状态（side_channels 微调）

| # | 条目 | 要求 | 当前值 | 判定 |
|---|------|------|--------|------|
| 1 | 数据/参数比 | >= 20:1 | 10000*~200tokens/12.58M = 159:1 | ✅ |
| 2 | 数据复杂度匹配 | simple_zh 级别 | simple_zh 小学水平 | ✅ |
| 3 | batch_size | >= 32 | 4（CPU 限制） | ⚠️ 偏小 |
| 4 | warmup | 100-2000 步 | 100 步线性 warmup | ✅ |
| 5 | decay | 最后 10-20% | 最后 20% cosine decay | ✅ |
| 6 | 工程保障 | 日志+checkpoint+续训 | 已实现 | ✅ |
| 7 | 评估用 PPL + 生成质量 | 双指标 | eval_aug_joint.py | ✅ |
| 8 | 保存 best checkpoint | best loss | 每 epoch 保存 | ✅ |

**合规判断**：全部条目已满足（batch_size 偏小受 CPU 限制）。PPL 已突破 v1 停滞区间。

---

### 0.6 闭环修复批次（2026-07-28）

系统性审查发现 23 项未闭环点，已处理 21 项，剩余 2 项待训练完成后处理。

**已修复（21 项）：**

| 类别 | 修复内容 | commit |
|------|---------|--------|
| 缺失模块 | 创建 agent_ext/ (9模块) + services/ (5模块) + memory_watchdog + output_engine stub | 2d7f2e7 |
| 接线缺失 | assemble_cortex Step 9.2-9.6 批量接线 play/evolution/limbs/Agent/ContextManager | 4affa2f |
| 死代码 | life_scheduler research 分支遮蔽修复 | 27d4853 |
| 进化闭环 | 凋亡→新生反馈（Phase 4 凋亡后补偿创建） | 560e47b |
| 数据闭环 | explore 结果接入 feed_engine 训练数据 | 2c1a66d |
| 睡眠闭环 | Phase 3.5 knowledge_distillation 实现 | 3bec93f |
| 方法接线 | #20-23: should_trigger_neurogenesis + record_sleep_training + 注释 deprecated | 32c8080 |
| 状态保护 | #19: cortex_state.pt vs neuron_*.pt 时间戳检查 | ea20376 |

**待处理（2 项）：**
- #6 训练-推理路径对齐（forward_train vs forward 机制不一致）— 等 side_channels 训练完成后处理
- #16-18 训练脚本改进（val/早停/决策）— 等当前训练完成后处理

### 0.7 开源模型借鉴清单（2026-07-28）

调研主流开源模型技术文档，记录可借鉴的工程思想。**目的不是改方向，是吸收工程技巧优化现有神经元共振架构。**

#### 借鉴决策框架：替换 vs 融合 vs 抛弃

对每个借鉴技术，按以下 5 维度评估并选择实施策略：

**维度 1: 原实现状态**
- 有根本缺陷/bug → 倾向替换
- 有效但可改进 → 倾向融合
- 方向错误 → 倾向抛弃原方案

**维度 2: 接口兼容性**
- drop-in 兼容（仅替换组件） → 倾向替换
- 需要适配层（新增参数/接口） → 倾向融合
- 不兼容或不需要 → 抛弃

**维度 3: 训练成本**
- 原训练结果已无效（必须重训） → 替换成本可接受
- 原训练结果可复用 → 融合更优（不浪费已有权重）
- 原训练结果必须废弃 → 抛弃成本高

**维度 4: 设计理念冲突**
- 无冲突（同范式优化） → 替换安全
- 互补（增强现有机制） → 融合
- 根本冲突（破坏生物启发哲学） → 抛弃

**维度 5: 可验证性**
- 有明确指标可对比（PPL/argmax） → 可做 A/B 实验
- 需要定性评估 → 融合后整体观察

**决策树：**

```
原实现是否有根本缺陷？
├── 是 → 原训练结果是否已无效？
│       ├── 是 → 直接替换（案例：causal mask 修复）
│       └── 否 → 抛弃原方案重训（案例：P8 field_dim 统一）
└── 否 → 借鉴技术是否 drop-in 兼容？
        ├── 是 → 原实现是否严格劣于？
        │       ├── 是 → 直接替换（案例：Adam → Muon）
        │       └── 否 → 融合（案例：side_channels + bias）
        └── 否 → 融合（案例：Auxiliary-loss-free balancing）
```

**A/B 对照实验方法（对不确定的借鉴点）：**

1. 基线（原实现）训练 N 步 → 记录 PPL 曲线
2. 借鉴版本训练 N 步 → 记录 PPL 曲线
3. 对比 4 个指标：
   - 收敛速度（达到相同 PPL 的步数）
   - 最终 PPL（相同步数下）
   - 稳定性（PPL 方差）
   - 训练成本（时间/内存）
4. 决策：
   - 借鉴版严格更优 → 直接替换
   - 互有胜负 → 融合（取长补短）
   - 借鉴版更差 → 抛弃借鉴

**已实施借鉴点决策记录：**

| 借鉴技术 | 原实现状态 | 接口兼容 | 训练成本 | 理念冲突 | 决策 | 实施结果 |
|---------|----------|---------|---------|---------|------|---------|
| Muon 优化器 | Adam 无 bug，但收敛慢 | drop-in（仅替换 optimizer） | 原结果可复用 | 无 | **直接替换** | ✅ PPL 停滞 131→突破到 124.9 |
| 乘性门控调制 | 加性调制被 norm 抵消 | drop-in（改 forward） | 原结果无效 | 无 | **直接替换** | ✅ gate_deviation 0.014→0.38 |
| post-norm side_channels | pre-norm 被抵消 | drop-in（调整顺序） | 原结果无效 | 无 | **直接替换** | ✅ PPL 突破停滞 |
| Auxiliary-loss-free balancing | 无平衡机制，通道死亡 | 不兼容（需加 bias buffer） | 原结果可复用 | 互补 | **融合** | ✅ 启发式 bias 更新 |

**待实施借鉴点预评估：**

| 借鉴技术 | 原实现状态 | 接口兼容 | 训练成本 | 理念冲突 | 预决策 |
|---------|----------|---------|---------|---------|--------|
| Shared Expert（Kimi K3） | 无 shared expert 概念 | 不兼容（需新神经元） | 新增训练 | 互补 | **融合** |
| IndexShare DSA（GLM-5.2） | 无跨轮索引复用 | 不兼容 | 新增逻辑 | 互补 | **融合** |
| Confidence-Guided（ConfSMoE） | 共振分公式简单 | 部分兼容 | 可调参 | 互补 | **融合** |
| Conditional Memory（DeepSeek V4） | 无架构级 RAG | 不兼容 | 新增模块 | 互补 | **融合** |
| OPD 在线策略蒸馏 | 两阶段训练无蒸馏 | 不兼容 | 流程重构 | 互补 | **融合** |
| KDA 线性注意力（Kimi K3） | 标准 attention | 不兼容（架构变） | 全量重训 | 冲突 | **抛弃原方案**（远期） |

**核心原则：**
- **直接替换**需要 3 个必要条件全部满足：原实现有根本缺陷 OR 借鉴严格更优 + drop-in 兼容 + 不破坏设计理念
- **融合**只要任一满足：借鉴与现有机制互补 + 需要适配 + 保留生物启发哲学
- **抛弃原方案**需要：原方向根本错误 + 借鉴从根本上更优且无冲突
- 所有"不确定"的借鉴点必须通过 A/B 对照实验验证后再决策

#### 借鉴点优先级总览（按实施顺序）

| 优先级 | 技术 | 来源模型 | 态极应用场景 | 实施时机 |
|--------|------|---------|-------------|---------|
| ★★★ 立即 | **Muon 优化器** | DeepSeek V4 | side_channels 微调换 Adam→Muon，突破 PPL 瓶颈 | 当前训练完成后第一实验 |
| ★★★ 立即 | **Auxiliary-loss-free balancing** | DeepSeek V3 | 解决 side_channels "死通道"问题，用偏置项动态调整而非辅助损失 | Muon 实验后 |
| ★★★ 中期 | **Shared Expert 机制** | Kimi K3 + DeepSeek V3 | 训练 general 神经元始终激活，其他域神经元稀疏激活 | side_channels 验证 EMERGE 后 |
| ★★★ 中期 | **IndexShare 索引复用** | GLM-5.2 | max_rounds=3 时复用 round 1 激活模式，降低多轮共振成本 | 扩展到 max_rounds=3 时 |
| ★★★ 中期 | **Conditional Memory** | DeepSeek V4 | 三路触发门控（关键词/语义/场景标签）补全记忆→推理融合 | Agent 闭环实现时 |
| ★★★ 中期 | **Confidence-Guided Selection** | ConfSMoE (ICML 2026) | 优化共振分公式，处理"多神经元共振分接近"的模糊场景 | 路由优化阶段 |
| ★★ 后期 | **OPD 在线策略蒸馏** | DeepSeek V4 | 优化"个体训练→协作训练"两阶段的蒸馏策略 | 训练流程重构时 |
| ★★ 后期 | **mHC 流形约束超连接** | DeepSeek V4 | 稳定 side_channels 梯度传播，防止 PPL 跳变 | 训练稳定性优化时 |
| ★★ 后期 | **Quantile Balancing** | Kimi K3 | 防止 side_channels 死通道（与 Auxiliary-loss-free 互补） | 路由优化阶段 |
| ★★ 后期 | **Thinking/Non-thinking 双模式** | Qwen3 + DeepSeek V4 | max_rounds=1（快速）/2（平衡）/3（深度）三档自动切换 | Cortex 路由优化时 |
| ★ 远期 | **KDA 线性注意力** | Kimi K3 | 长上下文（>2K token）时部分层换线性注意力 | 长上下文支持时 |
| ★ 远期 | **CSA+HCA 混合注意力** | DeepSeek V4 | 分层压缩（精读+广角），1M 上下文 FLOPs 仅 27% | 长上下文支持时 |
| ★ 远期 | **MoonViT-V2 训练方式** | Kimi K3 | VQ-VAE 用 next-token prediction 从零训练 | VQ-VAE 重训时 |
| ★ 远期 | **AgentEnv 沙箱设计** | Kimi K3 | 工具学习闭环的快照/恢复/fork | limbs.py 重构时 |

**核心洞察：**
- 所有主流开源模型（Kimi K3 / DeepSeek V3/V4 / GLM-5.2 / Qwen3）均采用 MoE 路线，与态极的神经元共振是不同技术路线
- MoE 是"一个大脑内 FFN 子模块稀疏激活"，态极是"多个完整模型协作"——根本差异在专家粒度
- **真正值得借鉴的不是架构，是工程技巧**：优化器（Muon）、负载均衡（Auxiliary-loss-free）、记忆融合（Conditional Memory）、索引复用（IndexShare）
- **ConfSMoE 是学术上最接近态极设计哲学的工作**：置信度路由与共振分都是"基于信号质量而非学习权重做路由"

---

#### 各模型详细架构事实与借鉴分析

#### Kimi K3（2.8T MoE，Moonshot AI，2026-07-27 开源）

**架构事实**（来源：[GitHub README](https://github.com/MoonshotAI/Kimi-K3)）：
- 93 层 = 1 Dense + 69 KDA（线性注意力）+ 24 Gated MLA（潜在注意力）
- 896 路由专家 + 2 共享专家，每 token 激活 16 个，激活参数 104B/2.8T（3.7%）
- SiTU-GLU 激活 + Quantile Balancing 防路由崩溃
- MXFP4 权重 / MXFP8 激活（量化感知训练）
- MoonViT-V2 视觉编码器：next-token prediction 从零训练（不用对比预训练）

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Shared Expert 机制** | 保留 general 神经元始终激活（类似 shared expert），其他域神经元稀疏激活。对应 Cortex routing_level=1 模式，但当前 general 神经元从未训练过——需要训练一个 general 神经元作为"共享专家" |
| ★★ | **Quantile Balancing** | side_channels 微调面临"死通道"问题（12 条通道中可能只有几条被激活学习）。借鉴分位数平衡思想，强制所有 side_channels 至少参与一定比例的梯度更新 |
| ★★ | **KDA 线性注意力** | 未来支持长上下文（>2K token）时，可把部分 attention 层换成线性注意力。当前 compact 神经元（512 hidden, 6 层）还不需要 |
| ★ | **MoonViT-V2 训练方式** | 视觉编码器用 next-token prediction 从零训练（不用对比预训练），获得更稳定的优化过程。态极 VQ-VAE 训练可借鉴 |
| ★ | **AgentEnv 沙箱设计** | 快照/恢复/fork 思路对 taiji/body/limbs.py 工具学习闭环有启发 |

**不借鉴的方向：**
- MoE 路由器（态极用共振分而非可学习 router，设计哲学不同）
- Gated MLA 矩阵分解（态极单神经元规模太小，不需要 KV cache 压缩）
- 3:1 KDA:MLA 混合比例（态极神经元层数少，不需要混合注意力）

#### DeepSeek V3（671B MoE，2024-12 开源）

**架构事实**（来源：[GitHub README](https://github.com/deepseek-ai/DeepSeek-V3)）：
- 671B 总参数，37B 激活（5.5% 稀疏度）
- MLA (Multi-head Latent Attention) + DeepSeekMoE
- 256 个专家
- **Auxiliary-loss-free load balancing** — 无辅助损失的负载均衡策略
- MTP (Multi-Token Prediction) 14B 模块
- 128K 上下文

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Auxiliary-loss-free load balancing** | DeepSeek V3 的核心创新：不用辅助损失强制负载均衡（辅助损失会污染主损失导致性能退化），而是用偏置项动态调整。态极 side_channels 的"死通道"问题可以借鉴此思路——不用正则项强制通道激活，而是用偏置项动态调整通道被选中的概率 |
| ★★ | **MTP (Multi-Token Prediction)** | 训练时预测多个未来 token（不只是 next-token），提升训练效率。态极 _train_single_neuron 当前只用 next-token CE loss，可考虑加 MTP 辅助头增强表征学习 |
| ★ | **DeepSeekMoE 细粒度专家** | 共享专家 + 路由专家分离设计。与 Kimi K3 的 shared expert 思路一致，双重验证了"保留通用专家 + 稀疏路由专家"的正确性 |

#### Qwen3（Dense + MoE 全尺寸，2025 开源）

**架构事实**（来源：[GitHub README](https://github.com/QwenLM/Qwen2.5)）：
- Dense: 0.6B / 1.7B / 4B / 8B / 14B / 32B
- MoE: 30B-A3B / 235B-A22B
- **Thinking / Non-thinking 双模式**
- Agent 能力强化（工具调用 in thinking & unthinking modes）

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★ | **Thinking/Non-thinking 双模式** | 态极可借鉴：默认 max_rounds=1（快速响应），复杂任务时切换到 max_rounds=3（深度思考，多轮共振）。对应 Cortex 已有的 routing_level 参数，但当前没有"任务复杂度感知"的自动切换机制 |
| ★ | **小尺寸 dense 模型（0.6B-32B）** | 验证了小模型路线可行性。态极 compact 神经元（36M）比 Qwen3 最小的 0.6B 还小一个数量级，说明态极在"超小模型协作"赛道有独特定位 |

#### ConfSMoE（ICML 2026，Confidence-Guided Sparse Expert Selection）

**架构事实**（来源：[GitHub](https://github.com/IcurasLW/ICML2026-Official-Repository-of-ConfSMoE)）：
- 用**置信度引导**稀疏专家选择，而非可学习路由器
- 专门处理**缺失输入**和**多模态输入**场景
- 学术论文，代码开源

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Confidence-Guided Expert Selection** | **与态极的"共振分"高度契合**。ConfSMoE 用输入置信度选专家，态极用共振分选神经元——两者本质都是"基于信号质量而非学习权重做路由"。可以借鉴 ConfSMoE 的置信度计算方式优化共振分公式，特别是处理"输入模糊"场景（多神经元共振分接近时如何选择） |

#### Mixtral 8x7B / 8x22B（Mistral AI，已归档）

**架构事实**：标准 MoE，8 个专家每 token 激活 2 个。仓库已归档，技术较老，无特殊借鉴价值。

#### Llama 3（Meta，已废弃）

**架构事实**：标准 Dense Transformer，GQA。仓库已废弃指向 llama-models，无 MoE 创新。态极已采用 GQA，无需额外借鉴。

#### GLM-5.2（744B MoE，智谱 AI，2026-06 开源）

**架构事实**（来源：[NVIDIA NeMo 文档](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/glm-5-moe-dsa.md) + [智谱开源说明](https://cndba.cn/article/17044)）：
- 744B 总参数，激活 40B（5.4% 稀疏度）
- MLA (Multi-head Latent Attention) + **DSA (Dynamic Sparse Attention)** 动态稀疏注意力
- **IndexShare DSA**：共享 DSA 层复用前一层的 top-k 稀疏注意力选择（跨层索引复用）
- 1M token 上下文
- MIT 协议
- GlmMoeDsaForCausalLM 架构
- TileLang 稀疏核可选

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **IndexShare DSA（跨层索引复用）** | GLM-5.2 每 4 层稀疏注意力共享同一套 top-k 索引，避免每层重新计算。**态极多轮共振可借鉴**：max_rounds=3 时，round 2/3 可以复用 round 1 的神经元激活模式（side_channels 的激活索引），避免每轮重新计算共振分。这能显著降低多轮共振的计算成本 |
| ★★ | **DSA 动态稀疏注意力** | 与态极的"稀疏激活神经元"思路相近。GLM 用动态 top-k 选择注意力的 token，态极用共振分选择激活的神经元——都是"基于内容动态稀疏"而非固定路由 |

#### DeepSeek V4（1.6T MoE，2026-04 开源）

**架构事实**（来源：[技术报告解读](https://cloud.tencent.cn/developer/article/2661839)）：
- V4-Pro: 1.6T 总参数，激活 49B（3.1% 稀疏度）
- V4-Flash: 284B 总参数，激活 13B
- **CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention)** 混合注意力
- **mHC (Manifold-Constrained Hyper-Connections)** 流形约束超连接
- **Muon 优化器**（替代 AdamW）
- **Conditional Memory** 条件记忆机制（架构级 RAG）
- **OPD (On-Policy Distillation)** 在线策略蒸馏
- 384 专家，激活 6 个（V4-Pro）
- DSA2（融合 DSA+NSA）
- 三种推理模式：Non-think / Think High / Think Max
- 1M 上下文，MIT 协议

**借鉴点：**

| 优先级 | 技术 | 态极应用场景 |
|--------|------|-------------|
| ★★★ | **Conditional Memory（条件记忆机制）** | **架构级 RAG 整合**：用户 query 通过关键词/语义/场景标签三路触发门控，激活相关记忆块，跨注意力融合。**与态极的语义记忆 + 工作记忆设计高度契合**，当前态极记忆系统虽已接线（#10/#15 修复），但检索→推理的融合机制仍缺失。可借鉴 DeepSeek V4 的三路触发门控设计 |
| ★★★ | **Muon 优化器** | 替代 AdamW，通过正交化更新方向使收敛更快更稳。**态极 side_channels 微调直接可借鉴**——当前用 Adam lr=1e-3，PPL 下降缓慢（132→目标<114），换 Muon 可能突破瓶颈。这是最低成本的尝试（只改优化器不改架构） |
| ★★ | **mHC 流形约束超连接** | 解决万亿模型训练稳定性。态极 side_channels 训练也有梯度不稳定问题（PPL 偶尔跳变），mHC 的流形约束残差思想可借鉴用于稳定 side_channels 的梯度传播 |
| ★★ | **OPD 在线策略蒸馏** | 分领域训练专家→多专家在线蒸馏融合。**与态极的"个体训练→协作训练"两阶段高度一致**。DeepSeek V4 用 OPD 把十几个领域专家能力融合进一个模型，态极用 side_channels 把多个神经元融合进 ensemble——可借鉴 OPD 的蒸馏策略优化 side_channels 训练 |
| ★ | **CSA+HCA 混合注意力** | 分层压缩（CSA 精读 + HCA 广角），1M 上下文下 FLOPs 仅 27%。态极未来支持长上下文时借鉴 |
| ★ | **三模式推理** | Non-think/Think High/Think Max 三档。与 Qwen3 的 thinking/non-thinking 双重验证了"按需调节思考深度"的正确性，态极可扩展为 max_rounds=1/2/3 三档 |

## 📐 设计文档：设计原则与 Phase 1-8 历史记录

> 以下为历史设计文档（Phase 1-8 实施记录），大部分已完成或废弃，保留供架构追溯。

### 设计原则

### 1.1 三大核心原则

1. **神经元差异性第一**：架构设计必须保留 per-neuron 个性，不以"减少参数"为首要目标
2. **自我进化能力**：架构不能限制态极进化到上千甚至上万神经元
3. **借鉴人脑神经科学**：兴奋/抑制分化、不应期、Hebbian 学习、突触可塑性、神经元凋亡/新生等

### 1.2 与人脑机制的对应关系

| 人脑机制 | 态极对应实现 |
|----------|-------------|
| 兴奋性/抑制性神经元分化 | neuron_type 字段，抑制性神经元 field_vector 取反 |
| 不应期 (Refractory Period) | refractory_counter，写入场后冷却 N 轮 |
| 突触可塑性 (STDP) | side_channels 权重的局部学习规则 |
| Hebbian 共激活组装 | CoactivationTracker EMA 矩阵 |
| 神经元凋亡与新生 | ApoptosisTracker 淘汰弱神经元 + NeurogenesisTrigger 检测信号（P7：创建由外部 train_neuron.py 执行） |
| 神经调质系统 | Neuromodulator 全局信号调节学习率 |
| 睡眠/记忆巩固 | consolidation_cycle() 离线重放 |
| 功能柱 (Cortical Column) | CorticalColumn 作为一等公民 |
| 注意力增益 | attention_beam 向量 boost 相关神经元 |
| 阈值可塑性 | per-neuron firing_threshold 自适应 |
| 域专用词汇输出 | domain-specific tokenizer + per-neuron 独立 lm_head（P7） |
| 从零独立训练 | 每 neuron 独立 embedding + body + lm_head，零教师依赖（P7-P8） |

---
### Phase 1: 基础生物学化 (已实施)

### 2.1 兴奋/抑制神经元分化

**人脑参考**：约 80% 谷氨酸能兴奋性神经元 + 20% GABA 能抑制性神经元。抑制性对防止癫痫、sharpening 信号、专注关键目标至关重要。

**实现**：
- `NeuronConfig.neuron_type: Literal["excitatory", "inhibitory"]`
- 抑制性神经元的 `field_vector` 在 `ResonanceNeuron.forward` 中取反 (`v = -v`)
- 场的 `write`/`update` 无需感知 neuron_type，向量已携带正负号

**文件**：`taiji/resonance/config.py`, `taiji/resonance/neuron.py`

### 2.2 不应期 (Refractory Period)

**人脑参考**：神经元发放后 1-2ms 绝对不应期 + 10ms 相对不应期，防止持续发放，强制信息分流。

**实现**：
- `ResonanceNeuron.refractory_counter` buffer
- `enter_refractory()`: 写入场后调用，设置冷却计数
- `tick_refractory()`: 每轮结束递减
- `in_refractory` 属性：round 2+ 中不应期神经元只读场不写场
- 旧贡献从场 state 中移除，避免"幽灵"影响

**文件**：`taiji/resonance/neuron.py`, `taiji/resonance/ensemble.py`

### 2.3 兴奋/抑制双通道

**人脑参考**：每个神经元接收兴奋性和抑制性突触输入，E/I 平衡是核心机制。

**实现**：
- `excite_channels`: 正向残差调制 (`h += 0.1 * proj(signal)`)
- `inhibit_channels`: 负向残差调制 (`h -= 0.1 * proj(signal)`)
- `establish_side_channel(peer_id, channel_type)` 支持双类型
- 同一 peer 可同时拥有两种通道，由 STDP 学习决定主导

**文件**：`taiji/resonance/neuron.py`

---

### Phase 2: 自我进化闭环

### 3.1 神经元凋亡 (Apoptosis)

**人脑参考**：弱连接神经元通过凋亡被清除，保持系统健康。

**实现**：
- `ApoptosisTracker` 追踪连续 K 次评估 PPL > threshold 的神经元
- 标记为"凋亡"后：
  - 从 ensemble 移除
  - 从磁盘删除 ckpt
  - 释放 neuron_id
  - 清理相关 side_channels
- 防止"僵尸神经元"占用资源

**触发条件**：
- 连续 3 次评估 PPL > 200（远超阈值）
- 或 compression_loss 持续 > 0.8（部落内无法代表）

### 3.2 神经元新生 (Neurogenesis)

**人脑参考**：海马齿状回成年后仍有新生神经元，填补知识盲区。

**实现**：
- `NeurogenesisTrigger` 检测"知识盲区"：
  - 某 domain 持续高错误率
  - CoactivationTracker 检测到孤立激活模式
- 新生流程：
  1. 从 teacher 蒸馏新神经元 ckpt
  2. 初始化为"幼稚态"：高学习率、低共振权重
  3. 逐步成熟：学习率衰减、共振权重提升
  4. 成熟后正式加入 ensemble

**幼稚态参数**：
```python
new_neuron.learning_rate = base_lr * 3.0  # 高学习率
new_neuron.resonance_weight = 0.1  # 低共振权重
new_neuron.maturity_counter = 0  # 成熟度计数器
```

### 3.3 CoactivationTracker 稀疏化 + 双 EMA

**人脑参考**：Hebbian 学习"用进废退"，突触强度根据共激活频率调整。

**实现**：
- `_matrix` 改为 `torch.sparse` [N, N] 张量
- 双 EMA：
  - fast EMA (α=0.1)：短期共激活，触发动态部落化
  - slow EMA (α=0.01)：长期趋势，决定 side_channel 强化/修剪
- 遗忘机制：slow EMA < ε 的 pair 自动移除
- 与凋亡/新生联动：
  - 持续低共激活 → 凋亡候选
  - 持续高错误率 + 孤立激活 → 新生触发

---

### Phase 3: 高级认知机制

### 4.1 STDP 局部学习

**人脑参考**：突触前神经元在突触后神经元之前发放 → LTP（增强）；反之 → LTD（减弱）。

**实现**：
- side_channels 权重更新不依赖全局 loss
- 记录 peer A 和 B 的 field_vector 时序
- 若 A 在 B 写入前已指向相似方向 → 增强 A→B 通道
- 若 A 在 B 之后才指向相似方向 → 减弱
- 形成因果链：A 领先 B 则 A 指导 B

### 4.2 神经调质系统 (Neuromodulation)

**人脑参考**：多巴胺/血清素/去甲肾上腺素等全局调质，根据奖励/注意力状态调节学习率和兴奋性。

**实现**：
- `NeuromodulatorState` 全局信号（3 个标量：dopamine/serotonin/norepinephrine）
- **来源（2026-07-22 升级：双信号驱动，自主进化）**：
  - 快速信号：loss 变化率 → 多巴胺目标值（每轮更新）
    - Loss Δ < -20% → DA=0.85（强奖励，lr×2.0）
    - Loss Δ < -5% → DA=0.6（适度奖励，lr×1.4）
    - Loss Δ < 5% → DA=0.3（停滞，lr×0.95）
    - Loss Δ ≥ 5% → DA=0.15（惩罚，lr×0.72）
  - 慢速信号：next-token 准确率 → 血清素目标值（每 5 轮更新）
    - Acc Δ > 2% → 5HT=0.7（满足）
    - Acc Δ ±2% → 5HT=0.5（中性）
    - Acc Δ < -2% → 5HT=0.3（不满足）
- **EMA 趋近**：alpha=0.1，调质缓慢调整不突变
- 作用：
  - 调节神经元学习率：`lr = base_lr × get_lr_multiplier()`（多巴胺驱动）
  - 调节 field_write 强度：`get_field_write_scale()`（去甲肾上腺素驱动）
  - 调节 refractory 长度：`get_refractory_multiplier()`（血清素驱动）
- **持久化**：NeuromodulatorState 纳入 cortex_state.pt，跨会话调质状态连续
- **三调质全接线（2026-07-22 完成）**：
  - DA 由 sleep_engine 的 loss 趋势驱动（每轮）
  - 5HT 由准确率驱动（每5轮）
  - NE 由 metabolism 的 CPU 负载驱动（每次训练前）：CPU 高→NE↓（节能），CPU 低→NE↑（专注）
  - 覆盖优先级：metabolism 仅在内存>90%或资源不健康时覆盖 DA/5HT，否则不干扰 sleep_engine
- **验证**：DA 0.50→0.57, lr_mult 1.25→1.36 over 5 cycles；CPU 10%→NE=0.83, CPU 100%→NE=0.20

### 4.3 睡眠/记忆巩固 (Sleep Consolidation)

**人脑参考**：睡眠期间海马回放白天经历，将短期记忆转移到皮层长期存储，修剪弱突触。

**实现**：
- `consolidation_cycle()` 离线方法
- 重放近期 high-resonance 场状态序列
- 强化经常共激活的 side_channels
- 修剪弱连接
- 将 slow EMA 转移到长期 fingerprint

---

### Phase 4: 组织化机制

### 5.1 功能柱 (Cortical Column) 原生实现

**人脑参考**：皮层功能柱约 100 微米直径，内含 ~100 神经元处理同类输入。

**实现**：
- TribeSuperNeuron 作为一等公民
- 新神经元直接创建到某个部落（基于 domain）
- 部落内部强连接（side_channels 密集），部落间弱连接
- 部落可分裂（规模过大）或合并（功能重叠）

### 5.2 注意力增益 (Attentional Modulation)

**人脑参考**：自上而下注意力通过皮层-丘脑回路增强特定区域。

**实现**：
- `attention_beam` 向量（来自用户 query embedding 或任务上下文）
- 场评分时与 attention_beam 对齐的神经元获得 boost
- lm_head 加权时注意力之外的神经常被抑制

### 5.3 阈值可塑性 (Threshold Plasticity)

**人脑参考**：神经元发放阈值根据近期活动历史自适应调整 (homeostatic scaling)。

**实现**：
- 每个神经元维护 `firing_threshold` buffer
- 频繁激活 -> threshold 上升（更难再次激活）
- 长期沉默 -> threshold 下降（更容易被唤醒）

---

### 废弃：Phase 5 丘脑路由与动态扩展 — **[已废弃，P7 替代]**

> **废弃原因**：Phase 5 ThalamicRouter 依赖 1.5B 教师 hidden state 做路由判断（prototype 计算、实时路由决策），
> 与 P7 从零训练、零教师依赖的架构方向冲突。P7 每 neuron 自带独立 embedding + 域 tokenizer，
> 域路由由 `Cortex._infer_domain()` 启发式完成（CJK 字符集检测），训练完成后按 `neuron.{domain}` 匹配。
> 
> Phase 5 代码（`thalamic_router.py`, `neurogenesis_creator.py`, `domain_router.py`）已全部删除。
> 路由系统留待 P8-3 基于从零训练的 neuron prototype 重建。

### 5.1 ~ 5.6（已删除）

Phase 5 所有子任务（5.1 丘脑路由、5.2 动态扩展、5.3 学徒期、5.4 域合并）的代码实现已删除。
相关文件：`thalamic_router.py`, `neurogenesis_creator.py`, `domain_router.py`。

---

### 四个原问题的解决方案

### 6.1 lm_head: per-neuron 独立（P7 升级）

**P7 方案**（替代原 6.1 低秩分解）：每神经元独立完整 lm_head + 域专用 vocab。

**核心洞察**：256K 通用 vocab 让独立 lm_head 过大（131M，占 compact 体量的 7×）。
解决方案是使用域专用分词器（zh=20k, en=16k, code=12k, math=10k, general=16k），
独立 lm_head 仅 5-10M/神经元。

```python
# config.py: lm_head_rank=0（默认独立模式）
# 使用 get_domain_neuron_config("zh") 自动设置 vocab_size=20000
cfg = get_domain_neuron_config("zh")  # vocab_size=20000

# neuron.py: 每 neuron 自带完整 lm_head
self.lm_head = nn.Linear(hidden_size, vocab_size)  # 例: 512×20000=10.2M
```

**参数量对比**：

| 方案 | 5 神经元 lm_head 总计 | 差异性 |
|---|---|---|
| P6 旧方案（共享 W_base + 低秩） | 82M（共享 65.6M + 5×16.4M 残差） | 仅残差可变异 |
| P7 独立 lm_head + 域 vocab | ~60M（4×~10M + 1×16M） | 完整独立 |

**旧低秩模式保留**：`lm_head_rank > 0` 保留用于实验性场景（非默认），
此时 W_base 由 assemble_cortex 外部注入。

**TokenTranslator 桥接**：
```
neuron 输出: domain_token (10k-20k vocab)
    ↓ TokenTranslator.domain_to_general() 查对齐表
general_token_ids (256K I/O 协议)
    ↓ 通用分词器 decode
输出文本
```

### 6.2 side_channels: top-K 稀疏化 + Hebbian 强化

**方案**：
- 每个神经元只与 top-K peer 建立通道（K=8~16）
- 通过 fingerprint 相似度自动选择（cosine 0.3-0.7 区间的互补 peer）
- Hebbian 强化：经常共激活的 channel 权重增强
- 长期不激活的 channel 衰减并修剪
- 参数从 O(N²) 降到 O(N×K)

### 6.3 CoactivationTracker: 稀疏矩阵 + 双 EMA

**方案**：
- `torch.sparse` [N, N] 张量替代 dict
- fast EMA (α=0.1) + slow EMA (α=0.01)
- 遗忘机制：slow EMA < ε 自动移除
- 定期（每 T 步）做社区检测

### 6.4 串行 forward: CUDA stream 真并行

**方案**：
- 每个神经元在自己的 CUDA stream 上独立 forward
- 保留 per-neuron 语义
- GPU 真并行执行多个 stream
- 不需要 batch 输入，显存独立分配

---

### 实施进度

### 已完成 (Phase 1)
- [x] 兴奋/抑制神经元分化
- [x] 不应期机制
- [x] 兴奋/抑制双通道

### 已完成 (Phase 2 + 四个原问题)
- [x] 神经元凋亡机制 (lifecycle.py)
- [x] 神经元新生机制 (lifecycle.py)
- [x] CoactivationTracker 稀疏化 + 双 EMA (tribal.py)
- [x] side_channels top-K 稀疏化 (neuron.py)
- [x] CUDA stream 并行 forward (ensemble.py)
- [x] lm_head 低秩分解 (neuron.py, config.py) — 旧方案，P7 已升级为独立 lm_head

### 已完成 (Phase 3 + 4)
- [x] STDP 局部学习 (stdp.py)
- [x] 神经调质系统 (neuro_modulation.py)
- [x] 睡眠巩固周期 (neuro_modulation.py)
- [x] 功能柱原生实现 (cognitive_enhancements.py)
- [x] 注意力增益 (cognitive_enhancements.py)
- [x] 阈值可塑性 (cognitive_enhancements.py)

### 已完成 (Phase 5)
- [x] Phase 5.1: ThalamicRouter 基础路由（3/4 路由准确）
- [x] Phase 5.2: 动态扩展机制（未知域检测+新生自动注册）
- [x] Phase 5.3: 学徒期与巩固（MaturityTracker 集成+域合并）

### 进行中 (Phase 6: 脱教师 + 架构强化) — **[已废弃，P7 替代]**

> **废弃原因**：Phase 6 的目标是"脱教师"——通过 StandaloneEmbedding + SelfEvolver 减少推理时的 1.5B 教师依赖。
> 但 P7 方案更进一步：从零训练，每 neuron 自带独立 embedding + 独立 lm_head + 域 tokenizer，
> 从根本上零教师依赖。Phase 6 的中间方案（teacher SVD 初始化 → SelfEvolver 渐进自主）变得多余。
>
> Phase 6 代码（`standalone_embedding.py`, `self_evolving_encoder.py`, `shared_embed.py`, `init_from_teacher.py`）已全部删除。
> GammaOscillator 和 WorkingMemory（P6-3/P6-4）是独立模块，已保留。

- [x] ~~P6-1: 独立 embedding 表~~ — 已删除，P7 每 neuron 自带 embedding
- [x] ~~P6-2: 独立路由器~~ — 已删除，P8-3 重建
- [x] ~~P6-5: 端到端脱教师验证~~ — 已删除
- [x] **P6-3: Gamma 同步绑定** — 保留，`gamma_oscillator.py` 活跃
- [x] **P6-4: 工作记忆** — 保留，`working_memory.py` 活跃
- [x] ~~P6-6: 自主进化 encoder~~ — 已删除
- [x] ~~P6-7: SleepEngine SelfEvolver 集成~~ — 已删除
- [x] ~~P6-8: 训练后重算 prototypes~~ — 已删除

### 已完成 (Phase 7: 神经元独立化 — 消除 lm_head 依赖性)

**动机**：P6 实现运行时无教师依赖，但 W_base 共享 + 1.5B 初始化仍是路径依赖，
违背"差异性第一"原则。P7 彻底去除 W_base，每神经元独立完整 lm_head + 域 tokenizer。

- [x] P7-1: NeuronConfig 默认 `lm_head_rank=0`（独立模式）
  - `config.py`：`DOMAIN_VOCAB_SIZES` 硬编码 5 域 tokenizer vocab 大小
  - `get_domain_neuron_config(domain)` 自动设置域专用 vocab_size
  - 旧 `lm_head_rank > 0` 低秩模式保留作实验用途
- [x] P7-2: ResonanceNeuron 独立 lm_head + 独立 embedding
  - `lm_head_rank=0` 时创建 `nn.Linear(hidden, vocab_size)` 完整 lm_head
  - 每神经元自带 `nn.Embedding(vocab_size, base_embed_dim)` 独立 embedding
  - `lm_head_rank > 0` 兼容旧低秩路径（W_base 由 assemble_cortex 注入）
- [x] P7-3: 参数量可控
  - compact zh: 512×20000 = 10.2M lm_head（vs 旧 W_base 方案的 65.6M 共享 + 16.4M 残差）
  - 5 域总计 lm_head ≈ 60M（远低于 5×256K 的 655M）
  - general 域复用 en tokenizer (16k)，不增加新 tokenizer 训练负担
- [x] P7-4: 废除的硬约束
  - ~~W_base 必须从 1.5B checkpoint-481000 初始化~~ → 零 1.5B 依赖
  - ~~W_base 全局共享冻结~~ → 每神经元独立 lm_head
  - ~~KL 域正则化防 W_base drift~~ → 不再需要（无共享 W_base）
  - ~~低秩残差 U_i@V_i 作为唯一个性通道~~ → 全部 lm_head 参数独立

- [x] P7-5: 推理链路改造（2026-07-21）
  - TokenizerHub 增强（encode_tensor/decode/vocab_size/eos_token_id/load_default_domains）
  - ResonanceEnsemble.forward 支持 input_ids（per-neuron encode_input_ids 路径）
  - Cortex P7 模式（set_tokenizer_hub/_infer_domain/_generate_p7）
  - assemble_cortex 移除 W_base，注册 TokenizerHub

- [x] P7-6: 代码清理（2026-07-21）
  - 删除 `_shared_lm_head_base*.pt` 数据文件
  - 删除 `init_w_base_from_teacher.py` / `diagnose_w_base_init.py`
  - neuron.py 移除 set_shared_lm_head() + lm_head_base
  - cortex.py 清理 W_base 注释 + DeprecationWarning on set_teacher_pipeline()

- [x] P7-7: 架构一致性验证（2026-07-21）
  - 6 个文件语法检查通过
  - P7 数据流验证一致
  - __init__.py 导出完整

- [x] P7-8: 计划文档更新（2026-07-21）
  - 新增第九节（P7 架构落地）+ 第十节（Phase 8 从零独立训练）+ 第十一节（项目健康度）

- [x] P7-9: 全项目错误方向清理（2026-07-21）
  - 删除 19 个错误方向文件（resonance 死模块 12 个 + 死脚本 7 个）
  - 清理 8 个活跃文件的残留引用（cortex.py, loader.py, senses.py, sleep_engine.py, play_engine.py, config.py, ensemble.py, evolution_engine.py）
  - 删除 dead code blocks（evolution_engine.py: 120 行蒸馏过渡代码, sleep_engine.py: _sleep_phase_self_evolve, cortex.py: 路由/门控/教师 pipeline 方法）
  - 全局搜索验证：零残留引用

- [x] P7-10: 功能补足与数据清理（2026-07-21）
  - 删除 3 个废弃神经元备份目录（~30 个 .pt 文件）
  - `_infer_domain()` 增强：code 关键字检测 + math 符号密度检测
  - `_train_single_neuron()` 实现：P7/旧双模式，支持 tokenizer_hub 域 tokenizer 训练
  - `play_engine.py` P7 兼容：tokenizer_hub encode + per-neuron encode_input_ids

**关键设计决策**：
- 域 tokenizer 对齐表（TokenTranslator）已存在 `taiji/resonance/translator.py`
- 输出时 domain_token → TokenTranslator → general_token → 通用分词器 decode
- 这是确定性符号转换，不参与梯度、不参与训练

---

### 与现有架构的兼容性

### 8.1 Checkpoint 兼容
- 新增字段（neuron_type, refractory_counter）使用默认值，旧 ckpt 可加载
- P7 升级：旧 W_base 低秩 ckpt → `migrate_ckpt_v4.py` 重构为独立 lm_head
  - 旧路径：W_base（共享） + U_i@V_i（残差）
  - 新路径：独立 `nn.Linear(hidden, vocab_size)`
  - 低秩模式（`lm_head_rank > 0`）仍兼容旧 ckpt 无需迁移
- side_channels 双通道：旧 excite_channels 保留，inhibit_channels 空

### 8.2 接口兼容
- `side_channels` 属性作为 `excite_channels` 的别名，旧代码无感知
- `establish_side_channel` 默认 channel_type="excite"，兼容旧调用
- CoactivationTracker 接口保持 `update()` 和 `get_dense_groups()`

### 8.3 性能考虑
- 不应期减少 round 2+ 的写入量，实际上降低计算
- 稀疏化 side_channels 减少 forward 中的线性投影次数
- 稀疏 CoactivationTracker 消除 O(N²) Python 迭代
- CUDA stream 并行受限于 GPU SM 数量，但优于串行

---

### P7 架构落地（已完成）

> 2026-07-21：P7-1 ~ P7-9 全部完成。项目从"蒸馏为主"彻底转向"从零训练"。

### 9.1 P7 推理链路改造（P7-5）

- [x] **TokenizerHub 增强**（`taiji/resonance/translator.py`）
  - `encode_tensor()/decode()/vocab_size()/eos_token_id()/list_domains()` 方法
  - `load_default_domains()` classmethod 自动加载 `taiji/domains/` 下域 tokenizer
  - "general" 域复用 en tokenizer (16k vocab)

- [x] **ResonanceEnsemble 支持 input_ids**（`taiji/resonance/ensemble.py`）
  - `forward()` 新增 `input_ids: Optional[Tensor]` 参数
  - `_parallel_forward()` 支持 P7 路径：每 neuron 调 `encode_input_ids(input_ids)`
  - 向后兼容 `shared_embeddings` 路径
  - 移除 ConfidenceGate/EarlyStopResonance/QualityFilter/DivisionPath/DomainRouter

- [x] **Cortex P7 模式**（`taiji/brain/cortex.py`）
  - `set_tokenizer_hub()`：注册域 tokenizer，自动进入 P7 模式
  - `_infer_domain()`：CJK 字符集启发式推断域
  - `_generate_p7()`：域 tokenizer encode → think → 域 tokenizer decode
  - `think()`：两路径——shared_embedding 或 P7 input_ids 直传
  - `generate()`：自动检测 P7 模式，选域 tokenizer
  - 删除：路由系统、门控系统、教师 pipeline、context_encoder

- [x] **assemble_cortex 更新**（`taiji/loader.py`）
  - W_base 注入、confidence_threshold、enable_gating 参数全部移除
  - 自动注册 `TokenizerHub.load_default_domains()`
  - GammaOscillator 相位按 cortex.neurons 的 domain 分配
  - 删除 SharedContextEncoder/ThalamicRouter 接线

### 9.2 代码清理（P7-6 + P7-9）

- [x] 删除废弃 resonance 模块 12 个：`gating.py`, `quality.py`, `division.py`, `domain_router.py`, `domain_detector.py`, `thalamic_router.py`, `neurogenesis_creator.py`, `standalone_embedding.py`, `self_evolving_encoder.py`, `shared_embed.py`, `init_from_teacher.py`, `channel_broker.py`
- [x] 删除废弃 training 模块：`distill.py`, `checkpoint_bridge.py`, `contrastive.py`, `joint.py`, `single.py`
- [x] 删除废弃脚本 7 个：蒸馏相关 `distill_neurons.py` 等、fix_token_offsets.py
- [x] 删除废弃数据目录：`data/distill/`, `data/neurons_backup*/`, `data/real/`
- [x] `neuron.py` 移除 `set_shared_lm_head()` + `lm_head_base` 属性
- [x] `compute_logits()` 简化为纯 per-neuron 路径
- [x] `sleep_engine.py` 移除 `_sleep_phase_self_evolve()`（678 行）、SelfEvolver 集成、NeurogenesisCreator 调用
- [x] `evolution_engine.py` 移除 120 行 dead code（蒸馏过渡代码）
- [x] 清理 8 个活跃文件的残留引用，全局搜索确认零残留

### 9.3 P7 架构验证

- [x] 所有修改文件语法检查通过
- [x] P7 数据流验证：`assemble_cortex → think → ensemble.forward → neuron.encode_input_ids` 链路一致
- [x] `__init__.py` 导出完整，仅保留活跃模块
- [x] 域 tokenizer 跨 vocab 兼容性处理（token ID clamping、logits padding）

### 9.4 当前状态（2026-07-21 P7-10 功能补足后）

| 功能 | 状态 |
|------|------|
| 域路由（code/math/zh/en/general） | ✅ 启发式检测已实现（code 关键字、math 符号密度、CJK 中文） |
| Sleep 训练（P7/旧双模式） | ✅ `_train_single_neuron` 已实现，支持 tokenizer_hub + per-neuron lm_head |
| PlayEngine P7 兼容 | ✅ 支持 tokenizer_hub 编码 topic，per-neuron encode_input_ids |
| 废弃神经元备份 | ✅ 全部 3 个备份目录已删除（~30 个 .pt 文件） |

### 9.5 L2 指纹共振路由实验（2026-07-23）

**实验**：`verify_routing_accuracy.py` 训练 24 轮（CYCLES 12→24），观察 loss 收敛与 L2 路由改善。

**结果**：
- Loss: 0.2915 → 0.2634（-9.6%），C7 后趋平台（0.26-0.27 震荡，C16 出现 0.29 尖峰）
- L1 域路由: 100% → 100%（启发式检测已完美）
- **L2 共振路由: 8% → 8%（零改善，与 12 轮完全相同）**
- 对比损失全程近零：inter ≈ 0.0000-0.0002，intra ≈ 0.0000

**根因分析**：L2 路由停滞是**结构性问题**，非训练量问题。
- `_fingerprint_route` 依赖神经元指纹的 cosine 相似度选路
- 对比损失 inter≈0.0001 说明指纹间几乎无梯度信号推动分化
- 神经元指纹未被区分开 → 路由无法辨别哪个神经元应处理给定输入
- **增加训练轮次无法解决**（已证伪"12 轮不足"假设）

**已清理错误方向**：
- ~~增加训练轮次 12→24 以改善 L2 路由~~（已证伪：24 轮 L2 仍 8%）

**结论**：当前阶段使用 L1 域路由（100% 准确）。L2 共振路由需修复指纹/对比学习机制本身。

### 9.6 L2 路由方案选型讨论（2026-07-23）

**用户取向**：效果与上限优先，不考虑下限/成本。

**两个原方案的上限分析**：

| 维度 | 方向 A（权重指纹路由） | 方向 B（共振分数路由） |
|------|---------------------|---------------------|
| 路由信号本质 | 静态权重统计量（mean of rows） | 动态输入响应（完整前向的 final_scores） |
| 信号表达力 | 弱——降维统计丢失信息 | 强——真实推理能力 |
| 分化空间 | 768 维（hidden_size），易分化 | 4096 维（field_dim），高维诅咒 |
| 训练信号路径 | 长——fingerprint 是 buffer，需 proxy loss | 中——field_vector 可直接 contrastive |
| 上限天花板 | **低**——权重相似≠能力相似 | **中**——受 4096 维稀疏性限制 |

**核心洞察**：两个原方案上限都不够高。
- A 被"权重是能力的间接代理"限制
- B 被"4096 维高维诅咒"限制

**改造方案与融合可能性**（详见下方讨论）：
- 改造 A 为"动态 prototype fingerprint"——数据驱动，上限中高
- 改造 B 为"端到端监督路由"——直接监督路由准确性，上限最高但需标签
- **融合点**：用共振分数作为 prototype 训练的辅助监督信号

### 9.7 融合方案实施：死代码暴露与三信号修复（2026-07-23）

**用户指令**：按融合方案推进，同时暴露之前借鉴社区项目机械塞入的死代码。

**死代码审计结果**：

| 借鉴机制 | 状态 | 判定 |
|---------|------|------|
| NeuronSpark `lateral_inhibition_norm` | ensemble.forward() 每轮调用 | ✅ 已融入 |
| Deviance WTA `apply_inhibitory_wta` | ensemble.forward() 抑制路径 | ✅ 已融入 |
| Hi-MoE `W_cond` + `prediction_complementarity` | field.score() + ensemble 加权 | ✅ 已融入 |
| RSGN `NeuronGeometry` | coaction 距离门控 | ✅ 已融入 |
| LuminaNet splitting | neurogenesis 路径 | ✅ 已融入 |
| **对比损失三信号（route/proto/align）** | sleep 循环调用但信号趋零 | ❌ **机械塞入死代码** |

**三信号结构性缺陷与修复**：

1. **route_loss 自相矛盾**：原版遍历全序对 (i,j)+(j,i)，要求 sim_i>sim_j 且 sim_j>sim_i，
   梯度互相抵消，净效果推向均匀化（与分化目标相反）。注释说"让正确 neuron 最高"但无域标签。
   → 修复：注入域标签，每个域样本喂给所有 neuron，正确域 adapter(prompt) 与 domain_prototype
   cosine 应最高（与推理路径 _fingerprint_route 一致）。margin=0.2。

2. **proto_loss 地板问题**：原版 `relu(sim-0.1)²`，高维空间 sim≈0（正交），relu(-0.1)=0 无梯度。
   → 修复：`(sim+0.1)²`，sim=0 时 loss=0.01>0，梯度=0.2，持续推向负相关。

3. **align_loss 均匀问题**：前两信号失效时 softmax 均匀导致 KL≈0。
   → 修复：前两信号有效后自然生效，保持 KL 蒸馏。

**根因**：对比损失从 MoCo 借鉴了 margin ranking 形式，但丢了最关键的**正负样本标识**
（哪个 neuron 是正确域），变成自相矛盾的死代码。24 轮实验"对比损失近零"正是此 bug 的表现。

**关键洞察**：`_fingerprint_route`（L2 推理路径）本身不是死代码——它是 contrastive phase 的训练
目标。死的是训练信号，导致推理路径从未被有效训练。修复训练信号即可复活两者。

**修复验证（8 轮）**：

| 指标 | 修复前（24轮） | 修复后（8轮） |
|------|--------------|-------------|
| route_loss | ≈0（自相矛盾） | 0.168→0.073（↓56.6%） |
| proto_loss | ≈0（地板问题） | 0.010→0.004（↓59%） |
| domain_prototype | 全零（未更新） | 全部激活（norm=1.0） |
| L2 路由准确率 | 8% | 3/4 域正确（zh/en/code） |

**端到端验证（8轮完整训练管线，lm_head + contrastive 协同）**：

| 指标 | 修复前（24轮） | 修复后（8轮） |
|------|--------------|-------------|
| L2 指纹路由准确率 | 8% | 21%→29%（3.6×） |
| route_loss | ≈0 | 0.169→0.121（↓28%） |
| proto_loss | ≈0 | 0.008→0.007（非零稳定） |
| align_loss | ≈0 | 0.008→0.002（↓75%） |
| lm_head 训练 loss | — | 4.21→1.36（↓68%） |

**遗留**：L2（29%）仍低于 L1（86%），主因 math 域误判为 zh（符号密集型重叠）+ contrastive
信号尚未完全收敛。更多训练轮次或更强域特异性样本可进一步提升。

### 9.8 多样本实验：域特异性 > 数量（2026-07-23）

**假设**：从 feed_engine 实际训练数据每域采样 3 条（vs 硬编码 1 条），增加 route_loss
的 margin ranking 对数（20→60 对），加速 prototype 分化。

**结果**：L2 路由 **回退** 21%→14%（vs 硬编码 21%→29%）。

**根因**：硬编码样本是最大域特异性的（"神经元共振场架构设计"极中文、"def resonance(field)"
极代码），而训练数据是短句（"今天天气真好"），域区分度更低。更多但更模糊的样本使 prototype
漂移向泛化方向，丢失尖锐域信号。

**初步结论**：对比学习样本的**域特异性质量**远比**数量**重要。

### 9.9 策划高域特异性多样本：质量+数量双优（2026-07-23）

**改进方向**：9.8 证明"训练数据随机采样"失败，但"硬编码单样本"非最优——样本太少导致
prototype 过拟合单一模式。正确路径：**手工策划多条高域标志性样本**，兼顾域内多样性与
域间最大区分度。

**策划样本**（每域 3 条，CURATED_SAMPLES）：

| 域 | 样本 | 域标志性特征 |
|----|------|------------|
| zh | "神经元共振场架构设计原理" / "深度学习模型训练优化方法" / "中文自然语言处理技术应用" | 纯中文长句 |
| en | "neural resonance field architecture design" / ... | 纯英文长句 |
| code | "def resonance(field): return field.sync()" / "class Neuron(nn.Module): ..." / "import torch; ..." | 语法关键字+符号 |
| math | "integral of sin(x) over [0, pi]" / "gradient descent: theta -= lr * dL/dtheta" / "P(A\|B) = P(B\|A) * P(A) / P(B)" | 公式符号密集 |
| general | "system design pattern overview methodology" / ... | 中性长句无强域信号 |

**三方案对比（8 轮完整训练管线）**：

| 方案 | 样本来源 | L2 路由（pre→post） | vs 原版 |
|------|---------|-------------------|--------|
| 原版死代码 | 硬编码 1 条 | 21%→8% | 1×（基线） |
| 单样本硬编码 | 硬编码 1 条 | 21%→29% | 3.6× |
| feed_engine 多样本 | 训练数据随机 3 条 | 21%→14% | 1.75×（回退） |
| **策划高域特异性多样本** | **手工策划 3 条/域** | **21%→36%** | **4.5×** |

**route_loss 收敛状态**：8 轮后 0.130 仍下降（未收敛），更多轮次可进一步提升。

**关键洞察**：
1. **域特异性质量 + 数量双优** >> 单一维度。策划样本同时满足"高域标志性"（质量）和
   "域内多模式覆盖"（数量，防过拟合）。
2. feed_engine 随机采样失败不是"数量"的错，而是"低域特异性数量"的错——短句天然域模糊。
3. **策划样本是 contrastive phase 的最高杠杆点**：样本质量直接决定 prototype 分化方向，
   优于调 margin/lr/轮次等超参。

**遗留**：L2（36%）仍低于 L1（86%）。route_loss 未收敛（0.130 仍降），下一步验证 16 轮
能否继续提升。math 域仍为误判热点（符号密集型与 code/zh 重叠）。

### 9.10 16轮 benchmark + tokenization 不一致根因修复（2026-07-23）

**16 轮 benchmark 结果**：

| 指标 | 8 轮 | 16 轮 |
|------|------|-------|
| route_loss | 0.139→0.130 | 0.139→0.085（↓更多） |
| L2 路由准确率 | 21%→36% | 21%→**29%**（反降！） |

**关键发现**：route_loss 持续下降但 L2 准确率反降。更多训练不仅没帮助，反而过拟合
导致退化。这证明 route_loss 与 L2 准确率之间存在**结构性错配**，非超参问题。

**根因定位：训练/推理 tokenization 路径不一致**

对比两条路径的 prompt 编码：

| 路径 | 编码方式 | 代码位置 |
|------|---------|---------|
| **训练**（route_loss） | 域分词器→逐token映射到general ids | sleep_engine.py:1168-1186（原版） |
| **推理**（_fingerprint_route） | 直接 general 分词器编码 | cortex.py:978-980 |

SentencePiece 对**子串**和**全文**的切分粒度不同：
- 训练：`domain_sp.encode("神经元共振场")` → ["神经","元","共振","场"] → 逐个 piece 用
  `general_sp.EncodeAsIds(piece)` 映射 → 4 个 general ids
- 推理：`general_sp.EncodeAsIds("神经元共振场")` → 可能切为 ["神经元","共振","场"] → 3 个
  general ids（不同切分！）

`embed_adapter` 在训练分布（域分词粒度）上优化，推理时面对不同分布（通用分词粒度），
cosine 相似度计算失真。更多训练→更过拟合训练分布→推理退化。

**修复**：contrastive phase 改用直接 general 分词，与推理路径完全一致：

```python
# 修复前（不一致）：
domain_ids = tokenizer_hub.encode(sample_text, domain=sample_domain)
for did in domain_ids:
    piece = domain_sp.id_to_piece(did)
    general_ids.append(general_sp.EncodeAsIds(piece)[0])

# 修复后（一致）：
general_ids = general_sp.EncodeAsIds(sample_text)  # 与 _fingerprint_route 相同
```

**验证**：8 轮 benchmark 修复前后对比（运行中，结果待填）。

**同时启动的并行方向**：用真实 HF 数据（alpaca-zh + wikipedia-zh，3MB 已缓存）重训 zh
神经元，验证生成质量从胡言乱语→连贯句子的跃升。L2 路由质量依赖于神经元本身的质量
（domain_prototype 的区分度），两条路径不独立。

**tokenization 修复验证结果**：

| 指标 | 修复前（8轮） | 修复后（8轮） |
|------|-------------|-------------|
| L2 路由 | 21%→36% | 21%→**43%** |
| route_loss | 0.139→0.130（未收敛） | 0.189→**0.032**（收敛！） |

route_loss 从 0.130 卡住变为 0.032 真正收敛。tokenization 不一致确实是结构性缺陷。

### 9.11 生成质量实测 + 架构方向重定位（2026-07-23）

**zh 神经元训练**（3MB alpaca-zh + wikipedia-zh，2000 步，compact spec）：
- PPL: 62.4（训练前 PPL=1.3 但那是 100 句玩具数据过拟合）
- 生成输出：从纯乱码（"许多耐耐耐耐"）→ 词组拼贴（"副作用可以大大降低成功的关键...从伦敦到莫斯科...计算机系统"）
- 判断：有进步但仍不可读。学会了词汇，没学会语法。PPL 需降至 ~25-30 才能连贯。

**与 GPT-2 对比**：参数量差 3 倍（36M vs 117M），数据量差 13,000 倍（3MB vs 40GB）。
瓶颈在数据规模，不在架构。

**路由设计讨论：用户原始思路 vs 当前实现**

用户原始思路（竞争式反馈路由）：
1. 族长先看输入 → 给出反馈（实际模型响应）
2. 按反馈质量排序 → 好的族长带领聚落
3. 聚落执行 → 按反馈好坏赋权
4. 好的聚落带领全部神经元完成生成

当前实现（分类式域路由）：启发式域检测 → 直接划定 → 无竞争无反馈。

**根因**：P7 域专用分词器迫使先知道域才能分词，L1 启发式域检测是阻力最小路径，
竞争式路由从未实现。

**社区参考验证**：
- MoA（Together AI, ICLR 2025）：Proposer+Aggregator 多层协作，65.1% 超过 GPT-4o。
  但本质是 API 编排（50 行），用冻结大模型，文本级浅协作。**不是态极**——态极用小神经元
  + 共振场向量级深协作 + 睡眠训练闭环。
- Conf-SMoE（ICML 2026）：confidence-guided gating，"confidence > embedding similarity"
  直接验证用户思路。
- RouteMoA：两阶段路由（轻量预筛 + 置信度评估），对应用户"族长先看反馈"。

**路由方案决策：Plan C — 激活能量路由（生物学思路）**

不预测谁最好，而是看谁被激活（like 大脑）：
- 输入 → 每个神经元 forward → hidden state 激活强度 = "被点亮"
- 最高激活者胜出

用户认可 Plan C。但依赖神经元训练充分（欠训练的神经元激活信号不可靠）。

**核心实验：多同域神经元协作（用户核心假设验证）**

用户洞察："单个神经元肯定不能对话，但由神经元组成的大脑可以。应该尝试用多个同领域
神经元来测试能否在单个小神经元智能不够的情况下，多个协同就可以。"

实验设计：
- 将 zh_texts.jsonl（23443 条）均分 3 份
- 训练 3 个 zh 神经元（zh_1/zh_2/zh_3，各 ~7814 条，1MB）
- 对比：1×3MB 单干（PPL=62.4）vs 3×1MB 协作
- 如果协作 > 单干 → 验证"小神经元协作涌现智能"核心论点

技术实现：
- `cortex.generate()` 新增 `active_nids` 参数（显式指定激活神经元列表）
- MoCo 动态 logit 融合：多神经元 logits 按共振分数加权融合（已有机制）
- 共振场累积多神经元 field vectors（已有机制）

---

### Phase 8: 从零独立训练（已执行 → standard 训练）

> **核心命题**：神经元规模小（24M-118M），从零训练完全可行。人脑不蒸馏，婴儿直接暴露于语言环境。
>
> **目标**：去掉所有 1.5B 教师依赖，每 neuron 独立训练 → 共振场协作 → 超越单体模型。

### 10.1 训练可行性

| 神经元 | 参数量 | 10B tokens（单 4090） | 10B tokens（单 A100） |
|--------|--------|----------------------|----------------------|
| compact | ~25M | ~12 小时 | ~3 小时 |
| standard | ~60M | ~1 天 | ~5 小时 |
| expert | ~120M | ~2 天 | ~8 小时 |

5 域各训 10B tokens = 50B total ≈ 一周 A100。GPT-2 124M 在 10B tokens 上即可产生连贯文本。

### 10.2 任务分解

- [ ] **P8-1: 从零训练脚本** `train_neurons_from_scratch.py`
  - 替换 `sft_train_neurons_v2.py`（旧 W_base + 低秩残差）
  - 每 neuron 独立数据加载 + forward + backward
  - 域 tokenizer encode → neuron.forward → CE loss → backward
  - 支持 resume 和 checkpoint 保存
  - 关键：neuron lm_head 从随机初始化开始训练（无 1.5B 教师）

- [ ] **P8-2: 域数据 tokenize**
  - 用 `TokenizerHub` 的域 tokenizer 重新 tokenize SFT 数据
  - 替换 `data/sft/` 中旧共享 tokenizer 的数据
  - 确保每个域有足够数据（≥2000 samples）

- [ ] **P8-3: 纯 P7 路由**
  - 让 ThalamicRouter 支持纯 tokenizer 模式（无需 context_encoder）
  - 或：从零训练期间关闭路由，全部 neuron 参与
  - 训练完后再计算 prototypes 启路由

- [ ] **P8-4: sleep_engine P7 升级**
  - `_train_cortex_neurons` 移除 W_base 冻结逻辑
  - 改为独立 lm_head 训练
  - 支持 per-neuron 独立 lr 和 optimizer

- [ ] **P8-5: 端到端验证**
  - 训练 5 域 neuron（各 10B tokens）
  - 验证 generate 质量（中文输出中文、英文输出英文）
  - 验证路由准确率
  - 验证共振场协作效果

### 10.3 废弃项清单（P7-9 已完成）

| 文件/目录 | 原因 | 状态 |
|-----------|------|------|
| `taiji/training/distill.py` | 蒸馏不再需要 | ✅ 已删除 |
| `taiji/training/checkpoint_bridge.py` | 1.5B 桥接不再需要 | ✅ 已删除 |
| `taiji/resonance/standalone_embedding.py` | P7 每 neuron 自带 embedding | ✅ 已删除 |
| `taiji/resonance/shared_embed.py` | SharedEmbedProj 已废弃 | ✅ 已删除 |
| `taiji/resonance/init_from_teacher.py` | 从教师初始化不再需要 | ✅ 已删除 |
| `taiji/brain/cortex.py::set_teacher_pipeline()` | 教师 pipeline | ✅ 已删除 |
| `taiji/resonance/thalamic_router.py` | 教师依赖路由 | ✅ 已删除 |
| `taiji/resonance/neurogenesis_creator.py` | 蒸馏依赖神经新生 | ✅ 已删除 |
| `taiji/resonance/self_evolving_encoder.py` | 教师 SVD 初始化 | ✅ 已删除 |
| `taiji/resonance/gating.py` | ConfidenceGate 等 | ✅ 已删除 |
| `taiji/resonance/quality.py` | QualityFilter | ✅ 已删除 |
| `taiji/resonance/division.py` | DivisionPath | ✅ 已删除 |
| `scripts/training/distill_neurons.py` | 蒸馏脚本 | ✅ 已删除 |
| `data/distill/` 目录 | 蒸馏中间产物 | ✅ 已删除 |
| `data/neurons_backup*/` 目录 | 旧备份 | ✅ 已删除 |

---

### 项目健康度

### 11.1 代码统计

| 分类 | 数量 | 状态 |
|------|------|------|
| 核心库文件 (`taiji/`) | ~65 个 .py | 架构一致，P7-9 清理后无冗余 |
| resonance 模块 | 16 个 .py | 全部活跃，无死模块 |
| training 模块 | 2 个 .py | scheduler.py + __init__.py，旧蒸馏模块已删 |
| 训练/验证脚本 (`scripts/`) | ~45 个 .py | P7-9 清理后无蒸馏/教师依赖 |
| 计划文档 (`plans/`) | 4 个 .md | 本文档已更新到 P8 路线图 |
| 数据文件 (`data/`) | ~40 个 | P7 废弃文件/目录已删除 |

### 11.2 架构原则对齐

| 原则 | 状态 |
|------|------|
| 差异性第一 | ✅ P7 每 neuron 独立完整参数（embedding + body + lm_head） |
| 自我进化 | ✅ P6-6 SelfEvolver + P8 从零训练 |
| 人脑启发 | ✅ 皮层柱统一结构 + 独立经验学习 + 无教师依赖 |
| 结构性约束 | ✅ 域 tokenizer 控制 vocab，独立 lm_head ~5-10M/neuron |
