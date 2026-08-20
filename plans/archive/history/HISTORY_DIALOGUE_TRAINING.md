# 对话/Standard 神经元训练历史

> **拆分文档**（2026-08-10）：从 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](../../active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) 按内容拆分。
> 记录对话神经元与 Standard 神经元的训练历程、突破与负向结论。
> 当前项目状态、路线图与接口梳理以主 plan 为准。

**内容**：
- 全神经元对话训练（2026-07-31，standard 已成功）
- 历史实验记录（归档，协作层训练负向结论）
- Standard 神经元训练（2026-07-29，生成质量突破）

---

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

> 以下章节按时间线记录历史实验与结论，供追溯参考。当前项目状态以 [BIO_INSPIRED_ARCHITECTURE_PLAN.md](../../active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) 为准。
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
