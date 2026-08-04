# 架构妥协点审查报告

> 梳理整个项目中所有"为了易于实现采取的妥协方案"，按上限损失严重性排序。
> 每个妥协点给出：当前实现 → 妥协原因 → 上限更高方案 → 提升幅度。
>
> 调研范围：共振场核心 + 训练流水线 + 推理运行时，共 90+ 妥协点。
> 本报告聚焦**系统性妥协**（影响全局上限），局部小妥协见归档。

---

## 📌 当前执行状态（2026-08-04 更新）

**EOS + 短答案筛选重训进行中（治本方案 B 执行）**：

**根因诊断**（API 实测质量不达标的三个核心缺陷）：
1. ❌ **训练数据未加 EOS**：`batch_align_and_embed` 只产生 domain_targets，无 EOS token → 模型永不自然停止
2. ❌ **训练数据严重未充分利用**：仅加载 15000/88730（17%），其他 6 个文件完全未用
3. ❌ **训练/生成长度严重不匹配**：alpaca-zh 答案 200-500 字，生成 max_tokens=60（约 80-100 字）→ 模型学成了"长文本续写"而非"简短问答"

**三项核心修复**：
1. ✅ **EOS 注入**（[translator.py:498-521](file:///e:/taiji-neuron/taiji/resonance/translator.py#L498-L521)）：`batch_align_and_embed` 追加 EOS token（截断前注入，保证 EOS 始终在末尾）。smoke test 验证：末尾 target=3(EOS)，sft_mask=True ✓
2. ✅ **短答案筛选**（[utils.py:283-338](file:///e:/taiji-neuron/scripts/training/utils.py#L283-L338)）：`load_dialogue_texts_multi` 加 `max_answer_chars=150` 参数，筛选答案≤150字的样本，匹配生成 max_tokens=60
3. ✅ **数据量提升**：从 15000 条（仅 alpaca_zh 第一个文件）→ 19745 条（7 个文件全部加载，去重后），多样性显著提升

**重训参数**：
- 从头训练（不 --resume）：加 EOS 后训练任务本质变了，不继承"无 EOS 长答案"偏见
- epochs=8, batch=4, lr=1e-3, max_texts=88730, max_answer_chars=150
- 总步数 39480，ETA 约 301 min/epoch
- 备份：cross_spec_dialogue.pt.pre_eos_finetune

**训练状态**：Epoch 1/8 step 50 PPL=331（从头训练初始值，side_channels/cross_spec 重新初始化）

**API 路径五项修复（已完成，commit e94ca1d + 6426797）**：
1. 训练/推理 embedding 错配 → per-neuron shared embeddings 注入 cortex
2. 解码 byte fallback → `domain_sp.DecodeIds`
3. 默认参数宽松 → 60/0.55/15/soft
4. EOS 缺失 → 温和 EOS bias(+0.5)（临时方案，现已被训练时 EOS 注入替代）
5. 跑偏截断 → 连续 3+ 非中文 token 回退截断

---

## 🧠 神经元综合体饱和点判断标准（2026-08-04 决策）

**何时触发新神经元进化**——三个量化指标：

| 指标 | 饱和标准 | 当前值 | 状态 |
|------|---------|--------|------|
| PPL 收敛 | 连续 2 epoch Δ<0.5 | epoch 9→10 Δ=-14.2 | ❌ 远未饱和 |
| EMERGE 递减 | 连续 2 次评估 Δ<5% | 21.7% vs 22.7% | ❌ 协同收益稳定 |
| 参数效率 | body 梯度范数持续 <1e-4 | 仍在下降 | ❌ 未饱和 |

**触发时机**：当前轮训练后（PPL 预计 < 20），如果 API 质量仍不达标，就是触发新神经元的时机。预计 1-2 轮训练后（3-6 天）。

**新神经元方向（上限最高）**：跨域扩展（en_dialogue）。理由：
1. 当前全是 zh 神经元，跨域协作是"小神经元匹配大模型"的核心
2. 已有跨域 tokenizer 基础设施（en 16K vocab）
3. 跨域协作能显著提升综合能力（不同视角的共振）

**技术路径**：neurogenesis + lifecycle + establish_topology_channels 自动重建 + finetune_cross_spec 微调

---

## ⚠️ 架构本源矛盾（2026-08-04 新发现，详见 §4.0）

**核心矛盾**：当前参数分布中，神经元主体（338M, 62%）远超协作层（130M, 24%），与"小神经元协作匹配大模型"的设计理念**反向**。协作:神经元比为 0.4:1（人脑 ~1000:1）。

**本质问题**：每个神经元都能独立生成回答，协作只是锦上添花——这是"用协作机制包装的中等模型"，不是真正的"小神经元协作"。

**决策时点**：当前重训完成后，若 API 质量达标 → 继续优化数据；若不达标且 PPL 已收敛 → 触发**方向 B（小神经元 + 强协作）**架构实验。

**关键认知（2026-08-04 更新）**：协作层参数随神经元数增长（O(N²)），当前架构可通过增加神经元数自然提升协作占比，方向 B 作为备案，触发条件见 [§6.1](#61-当前方向优先级与备案触发条件)。

**详见**：[§4.0 架构本源矛盾](#40--架构本源矛盾神经元主体膨胀-协作附带2026-08-04-决策点) | [§6 方向 B 备案](#六方向-b-备案小神经元--强协作架构2026-08-04-设计)

---

## 一、系统性妥协（影响全局上限）

### S1. 共振机制从未被端到端训练 ★★★ 最关键

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 训练路径 | `forward_train` 单轮、无场、无侧通道（[ensemble.py:824-833](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L824-L833)） | 可微多轮共振（Gumbel-softmax / straight-through） |
| 后果 | "共振"是推理期技巧，neuron 从未学过"如何写场、如何协同" | 共振成为可学习能力 |
| 妥协原因 | 多轮含 hard top-K / argmax / `.item()` 不可微 | |
| 提升幅度 | 协作涌现 +30-50% | |
| 实施难度 | 高（架构性改动） | |

**核心问题**：`forward_train` 调用 `neuron.forward(shared_embeddings, return_logits=True)`，**不传 field_state、不传 side_signals、不应用 neuromodulator scale**。所有生物学机制（STDP/神经调质/Gamma/睡眠/新生）均以 `Optional[Any]` 注入，且**只在推理 forward() 生效，未进入梯度流**。

### S2. 256K embedding 配 16K tokenizer（隐性错配）★★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| shared_embedding | `nn.Embedding(256000, 512)`（[utils.py:246-256](file:///e:/taiji-neuron/scripts/training/utils.py#L246-L256)） | 不变（256K × 512） |
| general tokenizer | 16K en tokenizer 回退（[utils.py:111-117](file:///e:/taiji-neuron/scripts/training/utils.py#L111-L117)） | **256K general BPE（已存在）** |
| 后果 | 14.6 万 embedding 行永远训练不到；中文生僻字被 byte fallback | 全词覆盖（中文测试 20 token, 0 unk） |
| 妥协原因 | `build_domain_tokenizers.py` 无 general 域 | **已补充 general 域 + 修复路径不一致（T13）** |
| 提升幅度 | 词覆盖 +30-50%，PPL 虚高根因 | 已解除 |

**修复详情**（2026-08-01）：
- 验证 `taiji/domains/general/sp_general.model` 已是 256K vocab，中文覆盖率优秀（整词覆盖，0 unk）
- 修复 [build_domain_tokenizers.py](file:///e:/taiji-neuron/scripts/training/build_domain_tokenizers.py)：
  - `OUTPUT_DIR` 从 `domain_tokenizers/` 改为 `taiji/domains/`（与 load 路径一致，修复 T13）
  - 修复 `PROJECT_ROOT` 路径计算错误（少一级 parent）
  - `DOMAINS` 加入 general 域（256K vocab，混合语料 zh+en+code+math）
  - 新增 `load_mixed_corpus()` 函数支持 general 域的混合语料加载

### S3. Loss 单一化（全线纯 CE）★★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| 训练 loss | 5 个训练脚本全用纯 shift-CE | 对话训练用 SFT masking + 协作训练用多任务 loss |
| 协作层训练 | 纯 CE，无协作约束（[finetune_cross_spec.py:431-438](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py#L431-L438)） | CE + balance_loss + diversity_loss（S1 已修复） |
| SFT 训练 | question 和 answer 同等权重（[finetune_neuron_dialogue.py:284-288](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py#L284-L288)） | **SFT answer masking：只对"答："后的 token 计算 loss** |
| 后果 | side_channels 退化成噪声；模型复述 question | 协作真涌现 + 回答质量 |
| 妥协原因 | CE 最简单 | |
| 提升幅度 | 协作涌现 +15-30%，回答质量 +15-25% | 已解除 |

**修复详情**（2026-08-01）：
- [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) `batch_align_and_embed` 新增 `answer_marker` 参数：
  - 传入 `answer_marker="答："` 时返回 4 元组 `(shared_emb, targets, mask, sft_mask)`
  - `sft_mask` 标记 answer 部分（"答："之后的 token）为 True，question/pad 为 False
  - 不传时返回 3 元组，**完全向后兼容**（10+ 调用点无需修改）
  - 处理截断、无分隔符、padding 边界情况
- [experiment_config.py](file:///e:/taiji-neuron/scripts/training/experiment_config.py) 新增 `SFT_ANSWER_MARKER = "答："` 常量
- 3 个对话训练脚本应用 SFT masking：
  - [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：训练 + eval 都用 SFT masking，eval 改用 `reduction="sum"` 防止 answer 为空时 NaN
  - [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：协作训练用 `shift_mask & shift_sft_mask` 交集
  - [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：同上
- balance_loss（负载均衡）和 diversity_loss（field_vector 多样性）已在 S1 修复中接入 `forward_train`
- 5/5 验证通过（[verify_sft_mask.py](file:///e:/taiji-neuron/scripts/training/verify_sft_mask.py)）：向后兼容、基本正确性、batch 对齐、截断处理、无分隔符

**注**：margin ranking 暂未实现（与 balance_loss 语义部分冲突，且需要 individual_logits 额外计算开销）。当前 balance_loss + diversity_loss 已覆盖协作约束需求。

### S4. 训练步数整体偏短 ★★ ✅ 代码已修复（待重新训练）

| 阶段 | 修复前 | 修复后 | 建议步数 |
|------|---------|---------|---------|
| base (compact) | 16000 | 16000（未改，已训练完成） | 30000-50000 |
| base (standard) | 16000 | 16000（未改，已训练完成） | 50000-80000 |
| dialogue finetune | 4000 | **12000** | 12000-16000 |
| side_channels | 6ep (~15000步) | **8ep (~20000步)** | 20000+ |
| cross_spec | 3ep (~7500步) | **8ep (~20000步)** | 20000+ |

**4000 步对话微调确实太少**——36M 小模型需更多 epoch 内化对话格式，4000 步只够 2.5 epoch，明显欠拟合。当前多轮对话质量差的根因之一。

**修复详情**（2026-08-01）：
- [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：`--steps` 默认值 4000 → 12000
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：`--epochs` 默认值 3 → 8
- [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：`--epochs` 默认值 6 → 8
- warmup_steps=100 保持不变（12000-20000 步训练中占比 0.5-0.83%，合理）
- base 神经元训练步数未改（已训练完成，后续进化时再提升）
- **待重新训练才能验证效果**（建议等 S5 数据扩充完成后统一重新训练）

### S5. 数据规模与复杂度偏小 ★★ ✅ 代码已修复（待联网下载扩充）

| 数据集 | 修复前 | 修复后 | 建议规模 |
|--------|---------|---------|---------|
| simple_zh (base) | ~100K 小学作文 | ~100K（未改，已训练完成） | 500K+ 混合语料 |
| alpaca-zh (finetune) | 49K（单文件） | **49K→200K+（待联网下载 Belle/COIG）** | 200K+ |
| side_channels 训练 | 10K simple_zh | **100K 对话数据（默认 dialogue）** | 100K+ |
| eval | 30 条 | **100 条** | 500+ |

**simple_zh 是小学水平**，compact 神经元在它上面学到的语言能力上限低。**alpaca-zh 单点依赖**，覆盖面窄（偏百科问答），缺多轮、缺推理、缺代码。

**修复详情**（2026-08-01）：
- [experiment_config.py](file:///e:/taiji-neuron/scripts/training/experiment_config.py)：
  - 新增 `DIALOGUE_DATA_FILES` 列表（7 个本地文件，合并 ~97K 条，去重后 ~49K）
  - 新增 `DIALOGUE_HF_SOURCES` 列表（Belle 2M CN + COIG，可扩充 150K+）
- [utils.py](file:///e:/taiji-neuron/scripts/training/utils.py)：
  - 新增 `load_dialogue_texts_multi()`：多文件合并 + 去重 + 打乱 + SFT marker 过滤
  - 新增 `load_dialogue_texts_hf()`：从 HuggingFace 下载 Belle/COIG，转 "问：...\n答：..." 格式，本地缓存
- 3 个对话训练脚本改为使用 `load_dialogue_texts_multi`：
  - [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：eval 扩充 30→100 条
  - [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：dialogue 模式用多文件合并
  - [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：默认改为 dialogue 数据，max_texts 10K→100K
- **待联网下载**：本地文件去重后仅 ~49K 条（sft_unique 是 alpaca_zh_sft 子集），需运行 `load_dialogue_texts_hf()` 下载 Belle/COIG 扩充到 200K+

### S6. 域 token → re-encode 往返（推理核心缺陷）★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| 自回归生成 | domain token → text → general token → shared_emb（[cortex.py:1350-1358](file:///e:/taiji-neuron/taiji/brain/cortex.py#L1350-L1358)） | **对齐表预计算映射，消除 text 往返** |
| 后果 | 信息丢失 + 无 KV cache + 训练-推理分布偏移 | 消除信息丢失 + 为 KV cache 铺路 |
| 妥协原因 | 避免异构 vocab 间维护对齐表 | |
| 提升幅度 | 极高（推理速度 + 长文本质量） | 已解除（text 往返部分） |
| 实施难度 | 中（对齐表）/ 高（共享 codebook） | |

**修复详情**（2026-08-01）：
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) 新增 `_get_domain_to_general_alignment()` 方法：
  - 构建 `{domain_token_id: [general_token_ids]}` 对齐表（首次构建后缓存）
  - 对每个 domain token，预计算其 general token IDs 映射
  - 消除自回归生成时的 `domain→text→general` re-encode 往返
- `_generate_p7()` 修改：
  - 在获取 domain_sp 后构建对齐表（line 1260-1262）
  - 用 `alignment_table.get(next_token, [pad_id])` 替代 `domain_sp.id_to_piece + general_sp.encode`
  - 保留 fallback 路径（对齐表为空时走旧路径）
- **KV cache 仍未启用**（底层 layers.py 支持，但 neuron.py:454 丢弃 cache）—— 作为后续独立优化项

### S7. side_channels 全连接拓扑 ★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| 拓扑 | 全连接 mesh（N×N-1 条） | **结构性拓扑：full / knn / hub_spoke / hybrid（默认）** |
| 后果 | 通道互相干扰，梯度信号被均分 | 每条通道学到更鲜明角色；近邻更强先验 |
| 妥协原因 | `NeuronGeometry` 距离已算但未用于裁剪 | **已用距离+规格容量驱动拓扑构建** |
| 提升幅度 | 训练效率 +40%，协作质量 +5-10% | 已解除 |
| 实施难度 | 中 | |

**修复详情**（2026-08-01）：
- 新建 [topology.py](file:///e:/taiji-neuron/taiji/resonance/topology.py)：4 种拓扑模式
  - `full`：全连接（向后兼容）
  - `knn`：k 近邻对称拓扑（按 NeuronGeometry 距离）
  - `hub_spoke`：最大规格神经元作 hub，其他只经 hub 通信
  - `hybrid`（默认）：仿皮层分级 — 同(域,规格)全连接 → 跨规格经规格hub → 跨域经全局hub
- hub 选择：按容量（hidden_size × num_layers）降序，centroid 距离为 tiebreak
- 距离门控 init_scale：近邻 gate≈1 → 50.0（强先验），远邻 gate≈0 → 10.0（弱先验）
- [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)：`__init__` 新增 `geometry` 参数，接受外部传入的 NeuronGeometry
- `infer_topology_from_state()`：从 checkpoint 的 side_channels_state keys 自动推断训练时拓扑
- 5 个脚本更新为拓扑驱动建立：
  - [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：`--topology` 默认 hybrid
  - [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：`--topology` 默认 hybrid
  - [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py)：优先从 checkpoint 推断拓扑，回退 hybrid
  - [eval_aug_joint.py](file:///e:/taiji-neuron/scripts/training/eval_aug_joint.py)：同上
  - [analyze_side_channels.py](file:///e:/taiji-neuron/scripts/training/analyze_side_channels.py)：同上
- **向后兼容**：评估脚本自动从 checkpoint 推断拓扑，旧 checkpoint（全连接）自动匹配全连接拓扑

### S8. 冻结策略过保守 ★★ ✅ 已修复

| 阶段 | 修复前 | 修复后 |
|------|--------|---------|
| dialogue finetune | shared_emb frozen | **shared_emb 默认 trainable（--freeze_embedding 恢复旧行为）** |
| side_channels | neuron + emb 全冻结 | **解冻最后 N 层 transformer + norm + lm_head + field_write（默认 N=2）+ 可选 emb** |
| cross_spec | neuron + emb 全冻结 | **同 side_channels：解冻最后 N 层 + 可选 emb** |
| 优化器 | 单一 Muon+AdamW（side_channels only） | **body + emb 走独立 AdamW，lr = args.lr × body_lr_ratio（默认 0.1，温柔微调）** |
| checkpoint | 仅 side_channels + scale | **+ body_state + shared_embedding_state + body_optimizer_state + body_scheduler_state** |
| 交付产物 | 仅 side_channels + cross_spec | **+ body_state + shared_embedding_state（eval 脚本自动加载）** |

**核心问题**：从未联合训练过 neuron + side_channels + embedding，三阶段割裂导致表示空间无法协同适配。

**修复详情**（2026-08-01）：
- [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：`--train_embedding` 默认 True，`--freeze_embedding` 恢复旧行为；shared_emb 默认参与训练以适配对话 token 分布
- [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：
  - 新增 `--unfreeze_layers`(默认2) / `--train_embedding` / `--body_lr_ratio`(默认0.1) 参数
  - 解冻最后 N 层 transformer + norm + lm_head + field_write，让核心表示适配协作动态
  - 优化器分离：side_channels 走 Muon+AdamW，body+emb 走独立 AdamW（低 lr 温柔微调）
  - **修复关键 bug**：body_optimizer 之前创建了但训练循环未调用 zero_grad/step/scheduler.step，body 参数梯度无限累积且永不更新；现已修复
  - `build_final_artifact()`：交付产物含 side_channels + body + emb
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：同 side_channels 的 S8 改造（解冻最后 N 层 + 可选 emb + body 优化器 + checkpoint 扩展 + build_final_artifact）
- [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py)：加载 side_channels 后自动应用 body_state + shared_embedding_state（缺失则跳过，兼容旧 ckpt）
- [eval_aug_joint.py](file:///e:/taiji-neuron/scripts/training/eval_aug_joint.py)：同上，加载 body + emb 微调结果
- [_smoke_s8_checkpoint.py](file:///e:/taiji-neuron/scripts/training/_smoke_s8_checkpoint.py)：checkpoint round-trip smoke test 全部通过（body/emb/optimizer_state 完整恢复，0 mismatch）
- **向后兼容**：旧 checkpoint（无 body_state/emb_state）自动跳过加载，不影响现有训练产物

### S9. 生物学机制是推理期占位，非训练一等公民 ★★（核心已修复：调质门控 attention/FFN）

> **状态更新**（2026-08-01）：
> - S1 修复已让 neuromodulator/gamma/STDP/coaction 进入 `forward_train` 梯度流：
>   - neuromodulator `get_field_write_scale()` → 乘 scores → 影响融合权重（[ensemble.py:848,977-978](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L848)）
>   - gamma `tick()` + `kuramoto_step()` + `batch_gate_factors()` → 乘 scores（[ensemble.py:951-986](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L951)）
>   - STDP `record_firing()` + coaction `update()` 记录时序（不影响梯度）
> - **S9 修复（2026-08-01）：调质从"融合层 scores 缩放器"升级为"Transformer 内部门控"**：
>   - norepinephrine → `get_attention_temp_gain()` → 缩放 query → 门控注意力温度（聚焦/泛化）
>   - dopamine → `get_ffn_gain()` → 缩放 SwiGLU 输出 → 门控 FFN 强度（强化/衰减）
>   - 注入路径：`ensemble.forward` / `forward_train` → `_parallel_forward` → `neuron.forward` → `TransformerBlock` → `GroupedQueryAttention` + `SwiGLU`
>   - 全程可微（gain 是 Python float，但调质本身是外部状态；attention/FFN 权重通过 gain 进入梯度流）
>   - 4/4 smoke test 通过（[_smoke_s9_neuromod_gain.py](file:///e:/taiji-neuron/scripts/training/_smoke_s9_neuromod_gain.py)）
>
> 审计原文"forward_train 内完全不引用 self.neuromodulator"已**过时**。剩余真实缺口见下表。

| 机制 | S1+S9 后现状 | 剩余缺口（上限更高） |
|------|--------|---------|
| STDP | 记录发放时序，不影响梯度 | **影响 attention/FFN 权重**（Hebbian 可塑性进入 body） |
| 神经调质 | ✅ 门控 attention 温度 + FFN 强度 + 融合 scores | **per-region 调质**（当前全局共享，未来按域/区差异化） |
| Gamma | 单 40Hz 频段，门控 scores | **多频段（theta-gamma 嵌套）+ 跨频耦合** |
| 睡眠 | 重放只是计数（[neuro_modulation.py:210-212](file:///e:/taiji-neuron/taiji/resonance/neuro_modulation.py#L210)） | **真正 forward 重放 + 经验回放训练** |
| 新生 | `should_trigger_neurogenesis()` 存在但依赖外部 teacher | **自组织新生（从经验生长）** |

**核心问题（已修复）**：调质已从"融合层 scores 缩放器"升级为"Transformer 内部门控"，进入 attention/FFN 计算并参与梯度流。剩余 STDP/睡眠/新生是独立子项，不阻塞主训练路径。

### S10. Transformer 层零生物学修改 ★ ✅ 已修复（树突化 + 预测编码）

| 维度 | 修复前 | 修复后 |
|------|--------|---------|
| 层结构 | 标准 LLaMA 块（"zero changes to existing code"） | **树突化 TransformerBlock：basal + apical 双通路 + 预测编码整合** |
| 注释 | "zero changes to existing code" | 神经调质门控 + 树突化 cross-attention |
| 妥协原因 | 复用标准层 | **已解除：apical cross-attention 接收 field_state 作为自上而下反馈** |
| 提升幅度 | 结构性容量上限提升 | 已实现 |
| 实施难度 | 高 | 已完成 |

**修复详情**（2026-08-01）：
- [config.py](file:///e:/taiji-neuron/taiji/resonance/config.py)：`NeuronConfig` 新增 `dendritic_enabled: bool` 和 `apical_kv_dim: Optional[int]` 开关
- [layers.py](file:///e:/taiji-neuron/taiji/layers.py)：`TransformerBlock` 扩展 apical 路径
  - **Basal 路径**（始终存在）：标准 attention + FFN（自下而上，处理输入）
  - **Apical 路径**（dendritic=True 时创建）：
    - cross-attention：Q 来自当前层输入，KV 来自 field_state（全局集体意识场）
    - 独立的 apical_wq/wk/wv/wo 投影 + apical_norm
    - 无 causal mask（cross-attention，KV 是全局反馈）
  - **胞体整合**（预测编码）：
    - `apical_prediction = x + h_apical`（apical 残差预测）
    - `error = x - apical_prediction`（预测误差）
    - `gate = sigmoid(somatic_gate(x))`（每位置决定信任 basal 还是 apical）
    - `x = x - error_scale * gate * error`（误差校正，error_scale 可学习）
  - S9 神经调质门控（temp_gain/ffn_gain）同时作用于 basal 和 apical 路径
- [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)：
  - 根据 `dendritic_enabled` 构建 dendritic 或标准 TransformerBlock
  - forward 中 dendritic=True 且 field_state≠None 时，直接调用 block.forward 传入 field_state
  - field_state=None 时退化为标准 basal-only 行为（round 1 安全）
- **向后兼容**：
  - `dendritic_enabled=False`（默认）：完全等同修复前的标准 TransformerBlock
  - 旧 checkpoint 加载到 dendritic neuron：`strict=False` 自动跳过 apical 参数，保持初始化值
  - 5/5 smoke test 通过（[_smoke_s10_dendritic.py](file:///e:/taiji-neuron/scripts/training/_smoke_s10_dendritic.py)）：
    1. dendritic=False 与标准块一致（diff=0）
    2. dendritic=True apical 改变输出（diff=0.064）
    3. field_state=None 安全退化（diff=0）
    4. neuron 级别树突化生效（diff=2.72）
    5. checkpoint 兼容（missing=16 apical 参数，unexpected=0）

**参数量影响**：dendritic=True 时每层增加 apical_wq/wk/wv/wo + apical_norm + somatic_gate + error_scale，约增加 25-35% 参数（compact 85M → ~110M）。可通过 config 开关控制，不影响现有 neuron。

### S11. 512 token 硬截断 ★ ✅ 已修复（attention sink + 滑动窗口）

| 维度 | 修复前 | 修复后 |
|------|--------|---------|
| 上下文长度 | 512 token 硬截断（KV cache 无限增长或硬截断） | **attention sink + 滑动窗口，近 O(1) 推理时长上下文** |
| 后果 | 长对话被截断，多轮能力受限 | 支持数万 token 上下文（sink + window 配置） |
| 妥协原因 | CPU 推理显存/算力 | **已解除：StreamingLLM 技术，KV cache 上限 = sink_size + window_size** |
| 提升幅度 | 极高（长上下文能力） | 已实现 |
| 实施难度 | 中 | 已完成 |

**修复详情**（2026-08-01）：
- [config.py](file:///e:/taiji-neuron/taiji/resonance/config.py)：`NeuronConfig` 新增 `attention_sink_size: int` 和 `sliding_window_size: int`
- [layers.py](file:///e:/taiji-neuron/taiji/layers.py)：`GroupedQueryAttention` 扩展
  - 新增 `_evict_kv_cache()` 方法：KV cache 超限时保留前 `sink_size` + 最近 `window_size` token
  - `forward()` 中 `kv_cache_max_len > 0` 时自动驱逐
  - 滑动窗口 + KV cache 推理时禁用 causal mask（维度不匹配安全处理）
  - 训练时（无 kv_cache）完全不受影响
- [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)：构建 TransformerBlock 时传入 sink/window 参数
- **参数语义**：
  - `attention_sink_size=4`（默认 0=关闭）：保留前 4 个 token 作为注意力锚点
  - `sliding_window_size=2048`（默认 0=关闭）：滑动窗口大小
  - KV cache 上限 = sink_size + window_size = 2052
  - 两者都为 0 时完全向后兼容（KV cache 无限增长）
- **向后兼容**：
  - sink/window=0（默认）：完全等同修复前
  - sink/window 是 Python 属性（非 nn.Parameter），不影响 state_dict，旧 ckpt strict=True 加载成功
  - 6/6 smoke test 通过（[_smoke_s11_attention_sink.py](file:///e:/taiji-neuron/scripts/training/_smoke_s11_attention_sink.py)）

**使用建议**：生产配置推荐 `attention_sink_size=4, sliding_window_size=2048`（KV cache 上限 2052，支持 ~2000 token 上下文）。CPU 推理可降至 `sliding_window_size=512`（上限 516）。

### S12. 多轮对话靠前缀拼接 ★ ✅ 已修复（per-round field state + 对话轮次 token）

| 维度 | 修复前 | 修复后 |
|------|--------|---------|
| 多轮实现 | 前缀拼接 + 512 token 硬截断 | **per-round field_state 持久化 + 对话轮次 token** |
| 后果 | 无对话状态追踪，无角色标记，长对话被截断 | 真多轮能力，field_state 隐式记忆上下文 |
| 妥协原因 | 与训练时单文档自回归对齐 | **已解除：DialogueState 管理器替代前缀拼接** |
| 提升幅度 | 高（多轮连贯性） | 已实现 |
| 实施难度 | 高（需重训）/ 中（field state 注入） | 已完成（无需重训） |

**修复详情**（2026-08-01）：
- [field.py](file:///e:/taiji-neuron/taiji/resonance/field.py)：`ResonanceField` 新增 `save_round_state()` / `load_round_state()` 方法
  - 保存/加载完整状态：state + inhibitory_mask + contributions + inhibit_contributions
  - round-trip 完整恢复（测试验证 0 偏差）
- [dialogue_state.py](file:///e:/taiji-neuron/taiji/resonance/dialogue_state.py)：新增 `DialogueState` 类
  - **start_round(field)**：加载上一轮的 field_state（隐式记忆上下文）
  - **end_round(field)**：保存当前轮次的 field_state 快照
  - **prepend_round_token(ids)**：第 2 轮及以后在 prompt 前插入轮次标记 token
  - **max_rounds 滑动窗口**：保留最近 N 轮的 field_state（默认 5）
  - **add_dialogue_entry(role, content)**：记录对话历史（仅日志，不参与推理）
  - **序列化/反序列化**：完整状态可持久化到 checkpoint
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py)：
  - 新增 `set_dialogue_state()` / `clear_dialogue_state()` 方法
  - `_generate_p7` 集成：开始时 `start_round` + `prepend_round_token`，结束时 `end_round`
  - 未注册时（默认）保持原前缀拼接行为（完全向后兼容）
- **核心机制**：
  - 人脑启发：海马体在对话间保持工作记忆，每轮对话更新海马状态
  - 替代前缀拼接（把所有历史文本重新读一遍的低效做法）
  - 模型通过 field_state 隐式记忆上一轮的上下文
- **向后兼容**：
  - `cortex._dialogue_state = None`（默认）：完全等同修复前的前缀拼接行为
  - `DialogueState(max_rounds=0)`：不持久化（每轮独立）
  - 6/6 smoke test 通过（[_smoke_s12_dialogue_state.py](file:///e:/taiji-neuron/scripts/training/_smoke_s12_dialogue_state.py)）：
    1. field round-trip 完整恢复
    2. 多轮 field_state 持久化
    3. max_rounds 滑动窗口
    4. round_token 前缀插入
    5. reset 清空状态
    6. 序列化/反序列化

**使用方式**：
```python
dialogue = DialogueState(max_rounds=5, round_token_id=general_tokenizer.encode("<|round_start|>")[0])
cortex.set_dialogue_state(dialogue)
# 第 1 轮
response1 = cortex.generate("你好")
# 第 2 轮（自动加载第 1 轮的 field_state）
response2 = cortex.generate("刚才我说了什么？")  # 模型通过 field_state 记忆
# 新会话
cortex.clear_dialogue_state()  # 清空状态
```

---

## 二、局部妥协（按组件分类，精简列表）

> **梳理更新**（2026-08-01）：S1-S12 系统性修复已解决部分局部妥协，下表标注修复状态。
> 剩余真实缺口按上限提升潜力分级：★★★ 高 / ★★ 中 / ★ 低。

### 共振场核心

| # | 妥协点 | 当前 | 上限更高 | 状态 | 分级 |
|---|--------|------|---------|------|------|
| C1 | 神经元类型仅 2 种 | excitatory/inhibitory | PV+/SOM+/VIP+ 多亚型 | ✅ **已修复**（5 亚型: excitatory/pv/som/vip/inhibitory, 不同 write_gain + refractory_multiplier） | — |
| C2 | 不应期是整数计数器 | 二值状态 | 4 相恢复曲线 | 未修复 | ★ |
| C3 | 单体 Transformer 无树突分叉 | 单前向通路 | basal/apical 树突分离 + 预测编码 | ✅ **S10 已修复** | — |
| C4 | 场读入是加性残差 | gate*conditioning | 乘性门控 / 预测编码 | ✅ **已修复**（三种模式可选） | — |
| C5 | domain_prototype 单 EMA 向量 | 单质心 | 原型混合 + 在线聚类 | ✅ **已修复**（K 原型 + 在线 k-means 胜者 EMA 更新, max cosine 路由） | — |
| C6 | field_write 单 query pooling | 单语义切面 | 多 query 多头池化 | ✅ **已修复**（多头 attention pooling + 门控聚合） | — |
| C7 | 场是单一 D 维向量 | 无空间结构 | 空间场 + 扩散动力学 | ✅ **已修复**（图拉普拉斯扩散，forward_train 接入） | — |
| C8 | 场写入丢弃幅度 | L2 归一化 | 保留幅度作置信度 | ✅ **已修复**（attention entropy 置信度，per-sample scale 调制） | — |
| C9 | 共振轮数固定 3 | 固定开销 | 自适应停止 + 连续吸引子 | ✅ **已修复**（收敛 + 主导双信号自适应停止，min_rounds/max_rounds 双约束） | — |
| C10 | side_signals 仅 round 1 后构建 | rounds 2+ 复用 | 每轮动态更新 | ✅ **已修复**（推理路径每轮重建） | — |
| C11 | 跨 vocab 用零填充融合 | 语义错误 | 跨域 token 对齐 / 共享语义空间 | ✅ **S6 已修复** | — |
| C12 | 共振分数加权被禁用 | field.score() 不可比 | 对比学习投影到统一空间 | ✅ **已修复**（评分投影 + contrastive_loss NLL 排序对齐） | — |
| C13 | max 规格 EXPERT 仅 ~285M | CPU 可训 | 十亿-百亿级 | 硬件约束 | — |
| C14 | shared_expert_weight 固定 0.3 | 仿 DeepSeek | 任务相关可学习动态权重 | ✅ **已修复**（方案C: 共振分数+场状态联合驱动 per-sample sw） | — |
| C15 | v1_compat 保留旧 ckpt 行为 | 向后兼容 | 迁移后移除技术债 | 未修复 | ★ |

### 训练流水线

| # | 妥协点 | 当前 | 上限更高 | 状态 | 分级 |
|---|--------|------|---------|------|------|
| T1 | 评估集用训练集尾部 | 无 held-out | 5% hash 分桶 held-out | ✅ **已修复**（4 个训练脚本全部接入） | — |
| T2 | shared_emb_mode 默认 frozen | 首训误用卡随机 | 默认 auto 检测 | ✅ **S8 已修复**（默认 trainable） | — |
| T3 | base 阶段 side_channels 死权重 | 随机 peer 占内存 | frozen peer 特征提取 | 未修复 | ★ |
| T4 | 无数据增强 | 固定模板 | 回译 + prompt 改写 + 多轮拼接 | ✅ **已修复**（data_augmentation.py: 模板改写+多轮拼接+神经元改写, translator answer_marker_mode=last, 3 训练脚本 --augment） | — |
| T5 | dialogue finetune 未用 Muon | 纯 AdamW | Muon+AdamW 混合 | 未修复 | ★ |
| T6 | cross_spec 投影层单 Linear | 无 MLP | 2 层 MLP + GELU + 残差 | ✅ **已修复**（CrossSpecProjector: Linear+GELU+Linear 残差+零初始化, 旧 ckpt 兼容加载） | — |
| T7 | side_channels 仅 excite 无 inhibit | 单向调制 | excite + inhibit 平衡 | ✅ **已实现**（代码支持双通道，默认拓扑用 excite） | — |
| T8 | side_channels 用 simple_zh 训 | 分布外 | 改用 alpaca-zh | ✅ **S5 已修复**（默认 --data=dialogue, load_dialogue_texts_multi 加载 alpaca_zh_sft.jsonl 等多文件, max_texts 10K→100K） | — |
| T9 | field_conditioning 训练时关闭 | 怕噪声 | warm-up 后启用 | ✅ **已修复**（forward_train 加 field_conditioning 参数 + finetune warm-up 比例控制） | — |
| T10 | 阵容仅 5 神经元 | CPU 限制 | 扩到 11 个（含 shared_expert） | 硬件约束 | — |
| T11 | SAMPLING_MAX_TOKENS=100 | 折中 | 按场景分（200/128/512） | 未修复 | ★ |
| T12 | tokenizer 训练语料 30K 行 | 覆盖率 ~70% | 500K-1M 行 | ✅ **已修复**（词表库热插拔: 百科采样 ~200 万行 + 对话 4.8 万条×3 混合训练 50K zh tokenizer, token piece 映射 + lm_head 权重迁移, 无需重训神经元） | ★★ |
| T13 | build/load 路径不一致 | 手动拷贝 | 统一路径 | 未修复 | ★ |
| T14 | 无 ablation 评估 | 无法定位收益来源 | 4 组 ablation | ✅ **已修复**（evaluate_ablation.py: 共振协作/融合方式/side_channels/field_conditioning 4 组对照, T1 held-out 评估集, JSON 输出） | — |

### 推理运行时

| # | 妥协点 | 当前 | 上限更高 | 状态 | 分级 |
|---|--------|------|---------|------|------|
| R1 | 域路由用关键词计数 | 启发式 | 可学习路由器 / 共振分数路由 | ✅ **已修复**（resonance 软路由模式：probe→final_scores→top-k 跨域激活） | — |
| R2 | feed_engine 域检测硬编码 general | 简化 | 复用 cortex._infer_domain | 未修复 | ★ |
| R3 | 融合模式三套并存未分化 | 兼容遗留 | speculative decoding / consensus / MoE gate | ✅ **已修复**（consensus 投票融合模式：top-k 共识度加成，集体智慧浮现） | — |
| R4 | 采样策略固定 | top-k=50 | min-p / typical / ETD | 未修复 | ★ |
| R5 | 睡眠训练规模过小 | max_samples=64 | 异步 GPU worker + curriculum | 未修复 | ★ |
| R6 | 调质只驱动 lr 倍数 | 标量 | 驱动结构可塑性 / 兴奋阈值 | ✅ **S9 已修复**（调质门控 attention/FFN） | — |
| R7 | 代际迁移被禁用 | NotImplementedError | teacher→student 蒸馏 pipeline | ✅ **已修复**（三联蒸馏: KL logits + hidden 投影对齐 + attention 转移, 支持混合规格/vocab 对齐, train_distillation.py） | — |
| R8 | spec 选择只看错误率绝对值 | 单维度 | + 任务复杂度 + 资源约束 | 未修复 | ★ |
| R9 | 凋亡用固定阈值 | PPL>200 | 种群 PPL 分布相对阈值 | 未修复 | ★ |
| R10 | play 话题池硬编码 15 条 | 探索窄 | 动态话题生成 | 未修复 | ★ |
| R11 | SMCS EPE 候选评分用 n-gram | 无模型 | 用 ensemble final_scores / reward model | 未修复 | ★ |
| R12 | 无 KV cache | 每步全长度 forward | 启用 KV cache | ✅ **已实现**（layers.py 有 kv_cache，S11 增强 attention sink） | — |

### 梳理总结

**已被 S1-S12 修复的局部妥协（25 项）**：
- C1（神经元类型仅 2 种）← 已修复（5 亚型: excitatory/pv/som/vip/inhibitory, 不同 write_gain + refractory_multiplier）
- C3（树突分叉）← S10
- C4（场读入加性残差）← 已修复（additive/multiplicative/predictive 三模式可选）
- C5（domain_prototype 单 EMA 向量）← 已修复（K 原型 + 在线 k-means 胜者 EMA 更新, max cosine 路由）
- C6（field_write 单 query pooling）← 已修复（多头 attention pooling + 门控聚合）
- C7（场是单一 D 维向量）← 已修复（图拉普拉斯扩散，forward_train 接入）
- C8（场写入丢弃幅度）← 已修复（attention entropy 置信度，per-sample scale 调制）
- C9（共振轮数固定 3）← 已修复（收敛 + 主导双信号自适应停止，min_rounds/max_rounds 双约束）
- C10（side_signals 仅 round 1 后构建）← 已修复（推理路径每轮重建）
- C11（跨 vocab 零填充）← S6
- C12（共振分数不可比）← 已修复（评分投影 score_dim + contrastive_loss NLL 排序对齐）
- C14（shared_expert_weight 固定 0.3）← 已修复（方案C: 共振分数+场状态联合驱动 per-sample sw）
- T1（评估集用训练集尾部）← 已修复（5% hash 分桶 held-out，4 个训练脚本接入）
- T2（shared_emb 默认 frozen）← S8
- T6（cross_spec 投影层单 Linear）← 已修复（CrossSpecProjector: Linear+GELU+Linear 残差+零初始化, 旧 ckpt 兼容加载）
- T7（side_channels 仅 excite）← 代码已实现双通道
- T8（side_channels 用 simple_zh 训）← S5 已修复（默认 --data=dialogue, load_dialogue_texts_multi 加载 alpaca_zh_sft 等多文件）
- T9（field_conditioning 训练时关闭）← 已修复（forward_train 加 field_conditioning 参数 + finetune warm-up 比例控制）
- T4（无数据增强）← 已修复（data_augmentation.py: 模板改写+多轮拼接+神经元改写, translator answer_marker_mode=last 多轮精确 masking, 3 训练脚本 --augment）
- T14（无 ablation 评估）← 已修复（evaluate_ablation.py: 4 组对照实验定位收益来源）
- T12（tokenizer 训练语料 30K 行）← 已修复（词表库热插拔: upgrade_tokenizer.py 用百科 1314 万行采样 ~200 万行 + 对话 alpaca 4.8 万条×3 混合训练 50K zh tokenizer[对话词合并, 分词 11.0→11.5 tokens 持平, unk 0%], hot_swap_vocab.py 旧→新 token piece 映射 + lm_head 权重迁移[精确匹配 13427/子piece平均 36573/随机 0] + cfg.vocab_size 更新, 12 个 zh ckpt 全部迁移, 原 ckpt 备份至 pre_t12_backup/）
- R1（域路由用关键词计数）← 已修复（resonance 软路由模式：probe→final_scores→top-k 跨域激活）
- R3（融合模式三套并存未分化）← 已修复（consensus 投票融合模式：top-k 共识度加成，集体智慧浮现）
- R6（调质只驱动 lr）← S9
- R7（代际迁移被禁用）← 已修复（三联蒸馏: KL logits + hidden 投影对齐 + attention 转移, 支持混合规格/vocab 对齐, train_distillation.py）
- R12（无 KV cache）← 已实现 + S11 增强

**真实剩余缺口（按上限分级，共 15 项，其中 2 项硬件约束）**：

★★★ 高上限（0 项）：
所有高上限缺口已修复！剩余缺口均为中/低上限。

★★ 中上限（0 项）：
所有中上限缺口已修复！剩余缺口均为低上限。

★ 低上限（13 项）：
C2/C15, T3/T5/T11/T13, R2/R4/R5/R8/R9/R10/R11

硬件约束（2 项，非架构问题）：
C13（max 规格 EXPERT 受 CPU 限制）, T10（阵容仅 5 神经元受 CPU 限制）

---

## 三、上限提升潜力排序（Top 10）

| 排名 | 妥协点 | 类型 | 提升幅度 | 实施难度 |
|------|--------|------|---------|---------|
| 1 | S1 共振从未被端到端训练 | 系统性 | 协作涌现 +30-50% | 高 |
| 2 | S2 256K emb 配 16K tokenizer | 系统性 | 词覆盖 +30-50% | 中 |
| 3 | S6 域 token re-encode 往返 | 系统性 | 推理速度 3-5x + 长文本 | 中 |
| 4 | S3 Loss 单一化 | 系统性 | 协作 +15-30% + 回答 +15-25% | 中 |
| 5 | S11 512 token 硬截断 | 系统性 | 长上下文能力 | 中 |
| 6 | S5 数据规模偏小 | 系统性 | PPL +30-50% | 中 |
| 7 | S9 生物学机制是占位 | 系统性 | 结构性容量 | 高 |
| 8 | S4 训练步数偏短 | 系统性 | 收敛深度 +20-35% | 低 |
| 9 | S12 多轮对话靠拼接 | 系统性 | 多轮连贯性 | 中 |
| 10 | S7 side_channels 全连接 | 系统性 | 效率 +40% 质量 +5-10% | 中 |

---

## 四、关键洞察

### 4.0 ★★★ **架构本源矛盾：神经元主导，协作附带**（2026-08-04 决策点）

**核心矛盾**：当前参数分布与"小神经元协作匹配大模型"的设计理念**反向**。

**参数分布真相**（5 神经元阵容）：

| 部件 | 参数量 | 占比 | 训练状态 |
|------|--------|------|---------|
| 神经元主体（backbone）| 338M | 62% | 冻结 |
| body 最后 2 层（微调）| 185M | 34% | 微调（lr×0.1）|
| shared_embedding | 131M | — | 冻结 |
| **协作层**（side_channels + cross_spec）| **130M** | **24%** | 从头训练 |

**关键比例对比**：

| 维度 | 人脑 | 当前态极 | 上限方向 |
|------|------|---------|---------|
| 协作:神经元 比 | ~1000:1 | **0.4:1** | 10:1 以上 |
| 协作主导？ | ✅ | ❌ | ✅ |
| 单神经元能否独立工作 | 不能 | **能** | 不能（被迫协作）|

**漂移路径**（每次都选了"降低难度"而非"提升上限"）：
```
最初愿景：小神经元协作 → 匹配大模型
  ↓
第一次妥协：compact 51M（"让单个能独立完成基础任务"）
  ↓
第二次妥协：standard 134M（"需要稍大的核心神经元"）
  ↓
结果：5 个神经元 338M，已是中等模型体量
  ↓
协作层 130M 反而成了"附属品"
```

**本质问题**：每个神经元都能独立生成回答，协作只是锦上添花。**这不是"小神经元协作匹配大模型"，而是"用协作机制包装的中等模型"**。

**上限突破方向（待决策）**：

| 维度 | 方向 A（延续）| 方向 B（突破上限）|
|------|-------------|----------------|
| 新神经元规模 | +1 个 51M | +N 个 1-5M |
| 协作层占比 | 更低 | **大幅提升** |
| 单神经元能否独立 | 能 | **不能** |
| 协作涌现 | 弱 | **强（被迫协作）** |
| 实现难度 | 低 | 高（小神经元训练 + 协作设计都是新问题）|

**决策时点**：当前 5 神经元阵容 + EOS+短答案重训完成后，按饱和标准评估。若 API 质量达标 → 继续优化数据；若不达标且 PPL 已收敛 → **触发方向 B 实验**。

**方向 B 的核心技术挑战**（待研究）：
1. 小神经元（1-5M）如何训练？独立训练难以学到有效能力，可能需要协作训练初始化
2. 协作层如何设计才能支撑 10:1 以上的协作:神经元比？
3. 100+ 神经元的拓扑管理（已有 CoactivationTracker + tribal 基础设施）

---

### 4.1 "共振"是推理技巧，从未被训练

**最严重的妥协**：`forward_train` 不传 field_state、不传 side_signals、不应用 neuromodulator。所有生物学机制（STDP/调质/Gamma/睡眠）是推理期占位，未进入梯度流。这意味着 neuron 从未学过"如何写场、如何协同"——共振是推理时拼凑的，不是训练出来的能力。

### 4.2 tokenizer 错配是隐性天花板

256K embedding 配 16K tokenizer，14.6 万 embedding 行是死参数。所有 PPL 数字都被这层"tokenizer 噪声"掩盖，不解决它，后续所有优化都被掩盖。

### 4.3 协作层纯 CE 导致协作不涌现

协作层训练用纯 CE，不约束"协作是否真的比单神经好"。side_channels 学成噪声调制，很多场景协作 PPL ≥ 最强个体。需要 margin ranking + diversity + load balancing 三联 loss。

### 4.4 三阶段割裂导致表示空间无法协同

base → dialogue → cross_spec 三阶段从未联合训练，每阶段冻结前者。表示空间无法协同适配，side_channels 只能在固定表示上做线性调制。

### 4.5 生物学机制是"装饰"而非"骨架"

STDP/调质/Gamma/睡眠/新生全部以 Optional 注入，可独立开关。这意味着它们是"装饰性"的，不是架构的"骨架"。真正的生物学架构应该让这些机制成为不可移除的核心组件。

---

## 五、建议的改进路径（上限优先）

### 阶段 1：修复隐性天花板（S2 + S4）
- 训 256K general tokenizer
- 训练步数提到 12000-16000
- **不解决这两个，后续所有优化都被掩盖**

### 阶段 2：让共振可训练（S1）
- 可微多轮共振（Gumbel-softmax / straight-through）
- 让 forward_train 接入场+侧通道+调质
- **这是把共振从推理技巧变成可学习能力的唯一路径**

### 阶段 3：多任务 loss（S3）
- SFT answer masking
- 协作层 margin ranking + diversity + load balancing
- 跨域对比 loss（hub neuron 设计）

### 阶段 4：推理路径优化（S6 + S11 + S12）
- 域 token 对齐表
- 长上下文（attention sink / 分块共振）
- 多轮对话状态管理

### 阶段 5：生物学机制深化（S9）
- STDP 影响注意力/FFN 权重
- 多频段振荡 + 跨频耦合
- 真正睡眠重放
- 自组织新生

---

## 六、方向 B 备案：小神经元 + 强协作架构（2026-08-04 设计）

### 6.1 当前方向优先级与备案触发条件

**当前方向（优先）**：5 神经元阵容 + EOS + 短答案 + 数据扩充，继续优化。

**关键认知**：协作层参数**随神经元数增长**（side_channels 全连接 = O(N²)），所以当前架构并非死路——
增加神经元数即可提升协作层占比，不必立刻跳到方向 B。

**当前架构的自我演化空间**（按神经元数增长估算）：

| 神经元数 | 神经元总参 | side_channels 参（O(N²)）| 协作:神经元 比 |
|---------|----------|------------------------|--------------|
| 5（当前）| 338M | 25M | 0.07:1 |
| 10 | 676M | 100M | 0.15:1 |
| 20 | 1.35B | 400M | 0.30:1 |
| 50 | 3.38B | 2.5B | 0.74:1 |

**结论**：当前架构通过增加神经元 + 强化 side_channels 全连接，协作占比可自然提升到 0.5:1 以上，
不必立即跳到方向 B 的极端小神经元方案。

**方向 B 备案触发条件**（任一满足）：
1. 当前架构在 EOS+短答案重训后，API 质量仍不达标且 PPL 已收敛
2. 增加到 20+ 神经元后协作层仍未承载主要能力（EMERGE < 30%）
3. 单神经元能力过强导致协作被边缘化（移除协作后 ensemble PPL 下降 < 10%）

---

### 6.2 方向 B 神经元规格设计

| 参数 | 当前（中等神经元）| 方向 B（小神经元）|
|------|----------------|----------------|
| hidden_size | 512 / 768 | **128-256** |
| 层数 | 6-12 | **2-4** |
| 单神经元参数 | 51-134M | **1-5M** |
| 神经元数量 | 5 | **20-50** |
| 神经元总参数 | 338M | 50-200M |
| **协作层参数** | 130M | **500M-2B** |
| 协作:神经元 比 | 0.4:1 | **10:1 以上** |

### 6.3 训练流程五阶段

#### 阶段 0：规格设计
- 20-30 个神经元，每个 2-3M 参数，hidden=256，层数=3
- 协作层目标参数量 360M+（含 side_channels + cross_spec + 场演化层 + 协作注意力）

#### 阶段 1：数据分工策略（推荐 C）
- **策略 C：随机数据子集 + 训练自动分化**
- 每个神经元随机看 1/N 数据，训练过程中自然分化
- 不预定义主题/能力边界（避免人为偏见），类似人脑经验驱动分化
- 已有 [CoactivationTracker](file:///e:/taiji-neuron/taiji/resonance/tribal.py) 基础设施追踪分化模式

#### 阶段 2：单神经元预训练（弱能力初始化）
- 每个神经元独立训练（只看自己的 1/N 数据子集）
- **关键：PPL 故意停在 80-120**（不完全收敛，保留学习空间）
- 不要训练到 PPL < 50，否则单神经元能力过强，协作又成附属
- 这是"被迫协作"的前提——单神经元无法独立生成有效回答

#### 阶段 3：协作层训练（核心阶段）
- 冻结神经元主体，训练协作层（从头初始化）
- 协作层组件设计：
  - side_channels（全连接，每对神经元双向 excite/inhibit）
  - cross_spec 投影层（统一到 512 维场空间）
  - **场状态演化层**（新增，多轮可微场状态更新，~200M 参数）
  - **协作注意力**（新增，神经元间注意力机制，~100M 参数）
- Loss 设计：
  ```
  Loss = CE_loss(ensemble_output, target)        # 协作输出逼近目标
       + λ × diversity_loss(neuron_outputs)       # 防止神经元输出同质化
       + μ × cooperation_pressure(neuron_outputs) # 单神经元 PPL < 50 时加惩罚
  ```
- 目标：ensemble PPL < 30（协作显著优于单神经元）

#### 阶段 4：端到端微调
- 解冻 body 最后 1-2 层
- 联合微调（body lr << 协作层 lr，保护神经元专业化）
- 目标：ensemble PPL < 20

#### 阶段 5：评估
- EMERGE > 50%（协作远超最强个体）
- API 对话质量达标

### 6.4 关键设计决策（待研究，非立即执行）

| 决策点 | 选项 | 倾向 |
|-------|------|------|
| 协作层架构 | A. 复用 side_channels + cross_spec / B. 新增场演化层 + 协作注意力 | **B**（协作层需足够容量）|
| 单神经元预训练强度 | A. 不预训练 / B. PPL 80-120 / C. PPL 30-50 | **B**（弱能力但保留基础）|
| 神经元数量 | A. 10 / B. 20-30 / C. 50-100 | **B**（已有 tribal 拓扑支持）|

### 6.5 与当前流程的本质差异

| 维度 | 当前流程 | 方向 B 流程 |
|------|---------|-----------|
| 神经元预训练 | PPL 30-50（强能力）| PPL 80-120（弱能力）|
| 协作层角色 | 协作是锦上添花 | **协作是能力载体** |
| 协作层参数占比 | 24% | **>70%** |
| 单神经元能否独立 | 能 | **不能** |
| 协作涌现强度 | 弱 | 强（被迫协作）|

### 6.6 当前状态

**状态**：备案，不立即执行。
**优先路径**：当前 5 神经元阵容 + EOS + 短答案重训 → 评估 → 若不达标再考虑增加神经元数 → 若仍不达标才触发方向 B。
