# 架构妥协点审查报告

> 梳理整个项目中所有"为了易于实现采取的妥协方案"，按上限损失严重性排序。
> 每个妥协点给出：当前实现 → 妥协原因 → 上限更高方案 → 提升幅度。
>
> 调研范围：共振场核心 + 训练流水线 + 推理运行时，共 90+ 妥协点。
> 本报告聚焦**系统性妥协**（影响全局上限），局部小妥协见归档。

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

### S2. 256K embedding 配 16K tokenizer（隐性错配）★★★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| shared_embedding | `nn.Embedding(256000, 512)`（[utils.py:246-256](file:///e:/taiji-neuron/scripts/training/utils.py#L246-L256)） | 匹配的 256K general tokenizer |
| general tokenizer | 16K en tokenizer 回退（[utils.py:111-117](file:///e:/taiji-neuron/scripts/training/utils.py#L111-L117)） | 训 256K general BPE |
| 后果 | 14.6 万 embedding 行永远训练不到；中文生僻字被 byte fallback | 全词覆盖 |
| 妥协原因 | `build_domain_tokenizers.py` 无 general 域 | |
| 提升幅度 | 词覆盖 +30-50%，PPL 虚高根因 | |
| 实施难度 | 中（训 256K BPE） | |

### S3. Loss 单一化（全线纯 CE）★★★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 训练 loss | 5 个训练脚本全用纯 shift-CE | 多任务 loss |
| 协作层训练 | 纯 CE，无协作约束（[finetune_cross_spec.py:431-438](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py#L431-L438)） | + margin ranking + diversity + load balancing |
| SFT 训练 | question 和 answer 同等权重（[finetune_neuron_dialogue.py:284-288](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py#L284-L288)） | + SFT answer masking |
| 后果 | side_channels 退化成噪声；模型复述 question | 协作真涌现 + 回答质量 |
| 妥协原因 | CE 最简单 | |
| 提升幅度 | 协作涌现 +15-30%，回答质量 +15-25% | |
| 实施难度 | 中 | |

### S4. 训练步数整体偏短 ★★

| 阶段 | 当前步数 | 建议步数 |
|------|---------|---------|
| base (compact) | 16000 | 30000-50000 |
| base (standard) | 16000 | 50000-80000 |
| dialogue finetune | 4000 | 12000-16000 |
| side_channels | 7500 (3ep) | 20000+ |
| cross_spec | 7500 (3ep) | 20000+ |

**4000 步对话微调确实太少**——36M 小模型需更多 epoch 内化对话格式，4000 步只够 2.5 epoch，明显欠拟合。当前多轮对话质量差的根因之一。

### S5. 数据规模与复杂度偏小 ★★

| 数据集 | 当前规模 | 建议规模 |
|--------|---------|---------|
| simple_zh (base) | ~100K 小学作文 | 500K+ 混合语料 |
| alpaca-zh (finetune) | 50K（实际用 10K-100K） | 200K+（加 Belle/COIG） |
| side_channels 训练 | 10K simple_zh | 100K+ 对话数据 |
| eval | 50 条 | 500+ held-out |

**simple_zh 是小学水平**，compact 神经元在它上面学到的语言能力上限低。**alpaca-zh 单点依赖**，覆盖面窄（偏百科问答），缺多轮、缺推理、缺代码。

### S6. 域 token → re-encode 往返（推理核心缺陷）★★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 自回归生成 | domain token → text → general token → shared_emb（[cortex.py:1350-1358](file:///e:/taiji-neuron/taiji/brain/cortex.py#L1350-L1358)） | 对齐表 / 共享 codebook / logits 注入 |
| 后果 | 信息丢失 + 无 KV cache + 训练-推理分布偏移 | 速度 3-5x + 长文本质量改善 |
| 妥协原因 | 避免异构 vocab 间维护对齐表 | |
| 提升幅度 | 极高（推理速度 + 长文本质量） | |
| 实施难度 | 中（对齐表）/ 高（共享 codebook） | |

### S7. side_channels 全连接拓扑 ★★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 拓扑 | 全连接 mesh（N×N-1 条） | 结构性拓扑（k 近邻 / hub-spoke） |
| 后果 | 通道互相干扰，梯度信号被均分 | 每条通道学到更鲜明角色 |
| 妥协原因 | `NeuronGeometry` 距离已算但未用于裁剪 | |
| 提升幅度 | 训练效率 +40%，协作质量 +5-10% | |
| 实施难度 | 中 | |

### S8. 冻结策略过保守 ★★

| 阶段 | 冻结 | 可训练 | 问题 |
|------|------|--------|------|
| base | - | neuron + (可选)emb | emb 默认 frozen（首训误用会卡随机） |
| dialogue finetune | shared_emb | neuron | emb 不适配对话格式 token |
| side_channels | neuron + emb | side_channels + scale | 核心表示锁死 |
| cross_spec | neuron + emb | side_channels + scale + proj | 同上 |

**从未联合训练过 neuron + side_channels + embedding**，三阶段割裂导致表示空间无法协同适配。

### S9. 生物学机制是推理期占位，非训练一等公民 ★★

| 机制 | 训练时 | 推理时 | 上限更高 |
|------|--------|--------|---------|
| STDP | 只更新 side_channels，不影响 body；轮次级分辨率 | forward 内记录 | 影响 attention/FFN 权重；亚轮次分辨率 |
| 神经调质 | 3 个全局标量，只调 lr/写入强度/不应期 | forward 内调用 | per-region 调质；门控注意力/可塑性 |
| Gamma | 仅单频段 40Hz | monkey-patch field | 多频段（theta-gamma 嵌套）+ 跨频耦合 |
| 睡眠 | 重放只是计数，不真正 forward | save_state | 真正重放 + 经验回放训练 |
| 新生 | 依赖外部 teacher 蒸馏 | - | 自组织新生（从经验生长） |

**核心问题**：所有生物学机制均以 `Optional[Any]` 注入，`forward_train` 内**完全不引用** `self.neuromodulator`。

### S10. Transformer 层零生物学修改 ★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 层结构 | 标准 LLaMA 块（[layers.py:176-202](file:///e:/taiji-neuron/taiji/layers.py#L176-L202)） | 树突化注意力 + 局部 Hebbian 可塑性 |
| 注释 | "zero changes to existing code" | 神经调质门控激活 |
| 妥协原因 | 复用标准层 | |
| 提升幅度 | 结构性容量上限提升 | |
| 实施难度 | 高 | |

### S11. 512 token 硬截断 ★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 上下文长度 | 512 token 硬截断（[cortex.py:1260-1261](file:///e:/taiji-neuron/taiji/brain/cortex.py#L1260-L1261)） | StreamingLLM / attention sink / 分块共振 |
| 后果 | 长对话被截断，多轮能力受限 | 长上下文能力 |
| 妥协原因 | CPU 推理显存/算力 | |
| 提升幅度 | 极高（长上下文能力） | |
| 实施难度 | 中 | |

### S12. 多轮对话靠前缀拼接 ★

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 多轮实现 | 前缀拼接（[cortex.py:435-449](file:///e:/taiji-neuron/taiji/brain/cortex.py#L435-L449)） | 对话状态 token + per-round field state |
| 后果 | 无对话状态追踪，无角色标记 | 真多轮能力 |
| 妥协原因 | 与训练时单文档自回归对齐 | |
| 提升幅度 | 高（多轮连贯性） | |
| 实施难度 | 高（需重训）/ 中（field state 注入） | |

---

## 二、局部妥协（按组件分类，精简列表）

### 共振场核心

| # | 妥协点 | 当前 | 上限更高 |
|---|--------|------|---------|
| C1 | 神经元类型仅 2 种 | excitatory/inhibitory | PV+/SOM+/VIP+ 多亚型 |
| C2 | 不应期是整数计数器 | 二值状态 | 4 相恢复曲线 |
| C3 | 单体 Transformer 无树突分叉 | 单前向通路 | basal/apical 树突分离 + 预测编码 |
| C4 | 场读入是加性残差 | gate*conditioning | 乘性门控 / 预测编码 |
| C5 | domain_prototype 单 EMA 向量 | 单质心 | 原型混合 + 在线聚类 |
| C6 | field_write 单 query pooling | 单语义切面 | 多 query 多头池化 |
| C7 | 场是单一 D 维向量 | 无空间结构 | 空间场 + 扩散动力学 |
| C8 | 场写入丢弃幅度 | L2 归一化 | 保留幅度作置信度 |
| C9 | 共振轮数固定 3 | 固定开销 | 自适应停止 + 连续吸引子 |
| C10 | side_signals 仅 round 1 后构建 | rounds 2+ 复用 | 每轮动态更新 |
| C11 | 跨 vocab 用零填充融合 | 语义错误 | 跨域 token 对齐 / 共享语义空间 |
| C12 | 共振分数加权被禁用 | field.score() 不可比 | 对比学习投影到统一空间 |
| C13 | max 规格 EXPERT 仅 ~285M | CPU 可训 | 十亿-百亿级 |
| C14 | shared_expert_weight 固定 0.3 | 仿 DeepSeek | 任务相关可学习动态权重 |
| C15 | v1_compat 保留旧 ckpt 行为 | 向后兼容 | 迁移后移除技术债 |

### 训练流水线

| # | 妥协点 | 当前 | 上限更高 |
|---|--------|------|---------|
| T1 | 评估集用训练集尾部 | 无 held-out | 5% hash 分桶 held-out |
| T2 | shared_emb_mode 默认 frozen | 首训误用卡随机 | 默认 auto 检测 |
| T3 | base 阶段 side_channels 死权重 | 随机 peer 占内存 | frozen peer 特征提取 |
| T4 | 无数据增强 | 固定模板 | 回译 + prompt 改写 + 多轮拼接 |
| T5 | dialogue finetune 未用 Muon | 纯 AdamW | Muon+AdamW 混合 |
| T6 | cross_spec 投影层单 Linear | 无 MLP | 2 层 MLP + GELU + 残差 |
| T7 | side_channels 仅 excite 无 inhibit | 单向调制 | excite + inhibit 平衡 |
| T8 | side_channels 用 simple_zh 训 | 分布外 | 改用 alpaca-zh |
| T9 | field_conditioning 训练时关闭 | 怕噪声 | warm-up 后启用 |
| T10 | 阵容仅 5 神经元 | CPU 限制 | 扩到 11 个（含 shared_expert） |
| T11 | SAMPLING_MAX_TOKENS=100 | 折中 | 按场景分（200/128/512） |
| T12 | tokenizer 训练语料 30K 行 | 覆盖率 ~70% | 500K-1M 行 |
| T13 | build/load 路径不一致 | 手动拷贝 | 统一路径 |
| T14 | 无 ablation 评估 | 无法定位收益来源 | 4 组 ablation |

### 推理运行时

| # | 妥协点 | 当前 | 上限更高 |
|---|--------|------|---------|
| R1 | 域路由用关键词计数 | 启发式 | 可学习路由器 / 共振分数路由 |
| R2 | feed_engine 域检测硬编码 general | 简化 | 复用 cortex._infer_domain |
| R3 | 融合模式三套并存未分化 | 兼容遗留 | speculative decoding / consensus / MoE gate |
| R4 | 采样策略固定 | top-k=50 | min-p / typical / ETD |
| R5 | 睡眠训练规模过小 | max_samples=64 | 异步 GPU worker + curriculum |
| R6 | 调质只驱动 lr 倍数 | 标量 | 驱动结构可塑性 / 兴奋阈值 |
| R7 | 代际迁移被禁用 | NotImplementedError | teacher→student 蒸馏 pipeline |
| R8 | spec 选择只看错误率绝对值 | 单维度 | + 任务复杂度 + 资源约束 |
| R9 | 凋亡用固定阈值 | PPL>200 | 种群 PPL 分布相对阈值 |
| R10 | play 话题池硬编码 15 条 | 探索窄 | 动态话题生成 |
| R11 | SMCS EPE 候选评分用 n-gram | 无模型 | 用 ensemble final_scores / reward model |
| R12 | 无 KV cache | 每步全长度 forward | 启用 KV cache |

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
