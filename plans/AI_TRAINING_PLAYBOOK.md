# AI 训练准则 (AI Training Playbook)

> 本文档基于社区主流实践编写，是态极项目所有训练工作的**强制性参考标准**。
> 来源：SmolLM3 Training Playbook (HuggingFace 2025)、nanoGPT (Karpathy)、TinyStories (Microsoft 2023)、Chinchilla Scaling Laws (DeepMind 2022)。
> **核心原则：数据质量 > 架构创新。不要在违反基础训练定律的前提下验证架构创新。**

---

## 零、核心原则（优先级从高到低）

### 0.1 不要从零训练除非必须

SmolLM3 Playbook 的第一条建议：**先尝试用现有开源模型解决问题**。从零训练只在以下场景合理：
- **研究领域**：验证特定假设（如"新架构能否扩展"）
- **领域专属**：现有模型无法满足（如 DNA 模型、法律模型）
- **部署约束**：无人机/本地 FPGA 部署需要特定规格
- **战略开源**：填补生态空白

态极的场景是**研究领域**（验证多神经元协作架构），从零训练合理，但必须遵循以下准则。

### 0.2 数据质量 > 架构创新

SmolLM3 Playbook 强调："**最大的性能提升始终来自数据质量和混合的改进，而非追求新颖架构**"。

- 优秀团队更关注数据而非架构
- 每个架构修改都必须通过**消融实验**验证
- "直觉是廉价的，但 GPU 是昂贵的"——不要凭直觉添加组件

### 0.3 消融一切（Ablate Everything）

SmolLM3 的核心方法论：**对每一个修改运行数百个小规模实验来"去风险化"**。
- 注意力机制、嵌入共享、位置编码、数据混合——全部消融
- 在 3B 模型上用 100B tokens 做消融（约总训练量的 1%）
- **态极对应**：在 compact(36M) 上用 ~10M tokens 做消融

### 0.4 迭代速度优先

Qwen/DeepSeek 团队每季度训练 1 个模型，快速积累经验。
- 不要一次性投入全部资源做大规模训练
- 先小规模验证，再放大
- **态极对应**：CPU 训练必须先做小规模（<1M tokens）验证 pipeline

---

## 一、数据准则

### 1.1 Chinchilla 定律（硬约束）

**DeepMind 2022 年 Chinchilla 论文的核心发现**：大多数模型都 undertrained。

| 模型规模 | Chinchilla 最优 tokens | 实际主流做法 |
|---------|----------------------|------------|
| 10M | 200M | TinyStories: 3.28B (328:1) |
| 36M (compact) | 720M | LLaMA 风格: 5B+ (139:1) |
| 131M (standard) | 2.6B | LLaMA-7B: 1T (143:1) |
| 3B (SmolLM3) | 60B | SmolLM3: 11T (3667:1) |
| 70B | 1.4T | LLaMA-2-70B: 1.4T+ |

**硬规则**：
- **预训练阶段**：数据/参数比 ≥ 20:1（Chinchilla 最优）
- **推理优化阶段**：可超过 20:1（LLaMA 用 143:1，因为推理更便宜）
- **Beyond Chinchilla**：loss 在 10,000:1 时仍在下降，不 plateauau
- **态极当前违规**：compact 0.18:1（差 111 倍），standard 0.05:1（差 400 倍）——**这是生成乱码的根因**

### 1.2 数据复杂度匹配模型能力（TinyStories 启示）

**Microsoft 2023 年 TinyStories 论文的关键发现**：
- **<10M 参数** + TinyStories 简化数据 = 生成**连贯多段落故事**
- 单 transformer block 就能产出近完美语法的文本
- 关键不是模型大小，而是**数据复杂度匹配模型能力**

**数据复杂度层次**（从简到繁）：
1. **TinyStories**（3-4岁儿童词汇，简单叙事）→ 适合 <10M 参数
2. **Shakespeare**（字符级，文学语言）→ 适合 10M 参数
3. **FineWeb-Edu**（教育类网页，质量筛选）→ 适合 100M-1B
4. **维基百科全文**（成人级，专有名词、数字、多语言）→ 需要 1B+ 参数

**态极的教训**：36M 参数 + 维基百科 = 数据复杂度严重不匹配。应该用 TinyStories 级别的数据验证基础能力。

### 1.3 数据质量筛选

SmolLM3 三阶段预训练策略：
- **Stage 1 (8T tokens)**：基础数据（FineWeb, DCLM）
- **Stage 2 (2T tokens)**：高质量过滤数据（Stack-Edu 代码, FineMath4+ 数学）
- **Stage 3 (1T tokens)**：指令和推理数据（学习率衰减阶段）
- **关键原则**：**最高质量数据留到最后**（模型最终行为受后期数据影响最大）

**数据质量原则**：
- 多样性 > 数量（n-gram diversity 是可学习性的更强预测器）
- 上采样高质量数据
- 过滤低质量内容（垃圾文本、重复内容）
- **态极对应**：当前维基百科全文无质量筛选，需要改进

### 1.4 Token 化策略

**主流做法**：
- GPT-2 BPE (tiktoken)：vocab=50,257，通用、成熟
- SentencePiece BPE：vocab=32,000-128,000，多语言友好
- 字符级：vocab=65-200，适合教学/超小模型

**态极的教训**：
- 域专用 vocab=20,000 太小 → 很多中文 token 变成 byte_fallback (<0xXX>)
- byte_fallback token 准确率 88.5% 但占比 24%，拉低整体 argmax
- **建议**：用 GPT-2 BPE (50,257) 或更大 vocab，避免 byte_fallback

---

## 二、训练超参数准则

### 2.1 Batch Size

| 场景 | batch_size | 每步 tokens | 说明 |
|------|-----------|------------|------|
| nanoGPT CPU | 12-64 | 768-16384 | 最小可接受范围 |
| nanoGPT GPU | 64-512 | 16K-130K | 标准范围 |
| SmolLM3 | - | 2.36M | 大规模训练 |
| **态极当前** | **4** | **800** | **严重不足** |

**硬规则**：
- **最小 batch_size = 32**（或用梯度累积达到等效）
- 每步 tokens ≥ 8,192（batch_size × seq_len）
- 小 batch 导致梯度噪声大，训练不稳定
- **态极修正**：batch_size=32 或 grad_accum=8（当前 4×8=32）

### 2.2 学习率

**nanoGPT 的经验法则**：
- 小模型（<100M）：lr=1e-3（"baby networks can afford to go a bit higher"）
- 中模型（100M-1B）：lr=6e-4（GPT-2 标准）
- 大模型（1B+）：lr=2e-4 到 5e-4（SmolLM3 用 4e-4 peak）

**态极对应**：
- compact(36M)：lr=1e-3（当前 3e-4 偏低）
- standard(131M)：lr=6e-4（当前 1e-4 严重偏低）

### 2.3 学习率调度

**WSD 调度（SmolLM3 标准）**：
```
Warmup → Stable → Decay
  2000步    主体    最后10%线性衰减到0
```

**Cosine 调度（nanoGPT 标准）**：
```
warmup_iters=100 → cosine decay 到 min_lr=lr/10
```

**硬规则**：
- 必须有 warmup（100-2000 步，视规模）
- 必须有 decay（最后 10-20% 线性或 cosine 衰减）
- **态极当前**：WSD 已实现，正确

### 2.4 优化器

- **AdamW** 是主流（SmolLM3, nanoGPT, LLaMA 全用）
- beta1=0.9, beta2=0.95-0.99
  - 小 batch 时 beta2=0.99（nanoGPT："tokens per iter 少时稍大"）
  - 大 batch 时 beta2=0.95
- weight_decay=0.1（SmolLM3, nanoGPT 标准）
- **embedding 层不加 weight_decay**（SmolLM3 发现影响稳定性）

---

## 三、架构准则

### 3.1 纯 Transformer Decoder 是基线

**主流架构**（SmolLM3, nanoGPT, LLaMA, GPT-2）：
```
Token Embedding (+ Positional Embedding)
  → N × TransformerBlock(
      LayerNorm → CausalSelfAttention → LayerNorm → MLP
    )
  → LayerNorm → LM Head (tied with embedding)
```

**关键组件**：
- **GQA** (Grouped Query Attention)：减少 KV cache，性能无损
- **Tied Embeddings**：输入输出嵌入共享，节省 17% 参数
- **RoPE/NoPE**：位置编码
- **RMSNorm**：比 LayerNorm 更稳定

### 3.2 额外组件必须消融验证

**SmolLM3 的做法**：每个架构修改都用 100B tokens 消融验证。

**态极的教训**：
- field_write, field_read_layers, field_projector, domain_prototype 等组件**从未做消融**
- 这些组件增加参数量但是否贡献语言建模能力？未知
- **硬规则**：任何额外组件必须与"去掉该组件的 baseline"对比验证

### 3.3 架构规模选择

**nanoGPT 的经验**：
- Shakespeare demo：6层, 6头, 384维, ~10M 参数（CPU 3-5分钟）
- GPT-2 复现：12层, 12头, 768维, 124M 参数（8×A100, 4天）

**TinyStories 的发现**：
- 8层, 8头, 512维, 76.8M 参数 → 连贯短叙事
- 单 transformer block + 10M 参数也能生成连贯文本

**态极对应**：
- compact(6层, 8头, 512维, 36M) 规模合理，但需匹配简单数据
- standard(10层, 12头, 768维, 131M) 规模合理，但需更多数据

---

## 四、评估准则

### 4.1 不要用 teacher-forcing argmax 评估模型质量

**当前态极的问题**：追求 argmax 85%，这是**非主流指标**。

**主流评估方式**：
1. **Perplexity (PPL)**：标准语言建模指标
   - <30：连贯生成基线
   - <10：良好
   - <6：优秀（StoryGPT 在 TinyStories 上达到 6.23）
2. **生成质量人工评估**：直接看生成文本是否连贯
3. **GPT-4 评估**（TinyStories 做法）：用大模型评估小模型输出
4. **下游任务 benchmark**（lighteval, MMLU 等）

**argmax 的问题**：
- 很多 token 本质不可预测（日期、数字、专有名词）
- argmax 天花板 ~75% 可能是维基数据的特性，不是模型问题
- **主流从不追求 argmax 85%**

### 4.2 评估流程

**nanoGPT 的做法**：
- 每 250 步评估 val loss
- 仅在 val loss 下降时保存 checkpoint
- 训练结束后用 sample.py 生成文本看效果

**SmolLM3 的做法**：
- 用 lighteval 框架定期评估
- 训练中监控 loss curve 异常

**态极对应**：
- 评估 PPL（不是 argmax）
- 定期生成样本文本人工检查
- 保存 best val loss 的 checkpoint

---

## 五、CPU 训练的特殊限制

### 5.1 CPU 训练的现实

**nanoGPT 的 CPU 配置**：
- Shakespeare demo：10M 参数, 1M tokens, batch=12, 2000步, ~3分钟
- 配置：`--device=cpu --compile=False --block_size=64 --batch_size=12`

**CPU 训练的硬限制**：
- 无法训练到 Chinchilla 最优（36M 需要 720M tokens，CPU 太慢）
- 适合：教学验证、小规模实验、pipeline 调试
- 不适合：生产级训练、大规模数据

### 5.2 CPU 训练的应对策略

1. **用更小的模型**（<10M）+ 简单数据（TinyStories）
2. **用梯度累积**模拟大 batch（batch=4 × grad_accum=8 = effective 32）
3. **用更短的序列**（block_size=128-256，不是 512+）
4. **关闭 torch.compile**（CPU 上可能更慢）
5. **优先验证 pipeline 正确性**，再考虑规模

---

## 六、常见陷阱（态极已踩过的）

### 6.1 数据量不足（Chinchilla 违规）
- **症状**：loss 下降但生成乱码；argmax 卡在天花板
- **诊断**：计算实际训练 tokens / 参数比，应 ≥ 20:1
- **修复**：增加数据量或减小模型

### 6.2 数据复杂度不匹配
- **症状**：模型对"见过的"高置信度正确，对"没见过的"完全不知道
- **诊断**：Top-5 几乎不高于 Top-1（正常应 +15-25%）
- **修复**：用更简单的数据（TinyStories 级别）

### 6.3 batch_size 太小
- **症状**：训练不稳定，loss 震荡
- **诊断**：batch < 32
- **修复**：增加 batch_size 或用梯度累积

### 6.4 额外组件未消融
- **症状**：架构有额外组件但性能不如纯 transformer
- **诊断**：与去掉额外组件的 baseline 对比
- **修复**：消融验证，去掉无用的组件

### 6.5 评估方式错误
- **症状**：追求 argmax 85% 但生成仍乱码
- **诊断**：argmax 是非主流指标
- **修复**：改用 PPL + 生成质量评估

### 6.6 学习率不匹配
- **症状**：loss 下降慢或不收敛
- **诊断**：对照同规模模型的主流 lr
- **修复**：小模型用 1e-3，中模型用 6e-4

### 6.7 保存末步而非 best
- **症状**：末步性能远差于训练中最佳
- **诊断**：末步 loss vs best loss 差距大
- **修复**：保存 best val loss 的 checkpoint（已修复）

---

## 七、态极项目的训练检查清单

每次启动训练前，必须确认：

- [ ] **数据/参数比 ≥ 20:1**（或明确说明为何违反）
- [ ] **数据复杂度匹配模型规模**（小模型不用复杂数据）
- [ ] **batch_size ≥ 32**（或梯度累积等效）
- [ ] **学习率匹配规模**（小模型 1e-3，中模型 6e-4）
- [ ] **有 warmup + decay 调度**
- [ ] **评估用 PPL + 生成质量**，不用 argmax
- [ ] **保存 best val loss checkpoint**
- [ ] **额外组件有消融 baseline 对比**（或明确标注"未验证"）
- [ ] **先小规模验证 pipeline**，再放大

---

## 八、参考资源

- [SmolLM3 Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) - HuggingFace 2025 年最权威的小模型训练指南
- [SmolLM3 博客](https://github.com/huggingface-cn/hf-blog-translation/blob/main/smollm3.md) - 3B 模型训练细节
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy 的极简 GPT 实现，CPU 友好
- [TinyStories 论文](https://arxiv.org/abs/2305.07759) - 小模型生成连贯文本的里程碑
- [Chinchilla 论文](https://arxiv.org/abs/2203.15556) - 数据/参数比 20:1 的来源
- [Beyond Chinchilla-Optimal](https://arxiv.org/pdf/2401.00448v3) - 推理考虑下的扩展定律
- [StoryGPT 实践](https://app.readytensor.ai/publications/storygpt-pretraining-a-small-language-model-from-scratch-on-tinystories-ZzOynh7puXuD) - TinyStories 训练实例

---

## 九、版本历史

- **v1.0 (2026-07-25)**：基于社区学习创建。核心发现：当前训练方法违反 Chinchilla 定律（差 111 倍）、数据复杂度不匹配、batch_size 太小、评估方式非主流。
