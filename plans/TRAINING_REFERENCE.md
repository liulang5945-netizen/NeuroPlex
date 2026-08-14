# AI 训练准则 (Training Reference)

> **📖 参考文档（非 plan）**
> 本文档是训练工作的强制性参考标准，不是项目计划文档。
> 项目当前进度见 `BIO_INSPIRED_ARCHITECTURE_PLAN.md`。

> 本文档基于社区主流实践编写，是态极项目所有训练工作的**强制性参考标准**。
> **v2.0 (2026-07-25)**：整合 2025-2026 最新开源成果（DeepSeek-R1 Nature 封面、SmolLM3、OLMo 3、OpenThoughts3、gpt-oss）。
> 来源：
> - **DeepSeek-R1** ([Nature 2025-09-17](https://www.nature.com/articles/s41586-025-09422-z)) — 首个同行评审主流大模型，29.4万美元训练成本
> - **DeepSeek-V3** 技术报告 — 14.8T tokens，三层质量过滤
> - **SmolLM3** (HuggingFace 2025-07) — 3B 最强小模型，11.2T tokens，完全开源
> - **SmolLM2** (COLM 2025) — 1.7B，11T overtrain，FineMath/Stack-Edu/SmolTalk 数据集
> - **Smol Training Playbook** (HuggingFace 2025-10-30) — 214 页训练实战指南
> - **OLMo 3** (AI2 2025) — 7B/32B 全流程开源，三阶段基础训练
> - **OpenThoughts3** (2025-06) — 1.2M 推理数据，纯 SFT 达到 SOTA
> - **gpt-oss** (OpenAI 2025-08) — Apache 2.0 开源推理模型
> - **TinyStories** (Microsoft 2023) — 小模型生成连贯文本的里程碑
> - **Chinchilla** (DeepMind 2022) — 数据/参数比 20:1 的来源
> - **nanoGPT** (Karpathy) — 极简 GPT 实现，CPU 友好
> **核心原则：数据质量 > 架构创新。不要在违反基础训练定律的前提下验证架构创新。**

---

## 零、核心原则（优先级从高到低）

### 0.1 不要从零训练除非必须

SmolLM3 Playbook 的第一条建议：**先尝试用现有开源模型解决问题**。从零训练只在以下场景合理：
- **研究领域**：验证特定假设（如"新架构能否扩展"、"纯 RL 能否激发推理"）
- **领域专属**：现有模型无法满足（如 DNA 模型、法律模型）
- **部署约束**：无人机/本地 FPGA 部署需要特定规格
- **战略开源**：填补生态空白

态极的场景是**研究领域**（验证多神经元协作架构），从零训练合理，但必须遵循以下准则。

### 0.2 数据质量 > 架构创新

SmolLM3 Playbook 强调："**最大的性能提升始终来自数据质量和混合的改进，而非追求新颖架构**"。

- 优秀团队更关注数据而非架构
- 每个架构修改都必须通过**消融实验**验证
- "直觉是廉价的，但 GPU 是昂贵的"——不要凭直觉添加组件
- **OLMo 3 实践**：用 `olmOCR` 把学术 PDF 转高质量 Markdown，用 `Duplodocus` 全局去重，用"质量感知上采样"高倍率训练高质量数据

### 0.3 消融一切（Ablate Everything）

SmolLM3 的核心方法论：**对每一个修改运行数百个小规模实验来"去风险化"**。
- 注意力机制、嵌入共享、位置编码、数据混合——全部消融
- 在 3B 模型上用 100B tokens 做消融（约总训练量的 1%）
- **态极对应**：在 compact(36M) 上用 ~10M tokens 做消融
- **OLMo 3 微退火（Microannealing）**：先在小规模数据上快速验证数据源是否能提升特定能力，再整合进大混合
- **OpenThoughts3 实践**：1000+ 严格控制实验，逐步固定每一步最佳选择

### 0.4 迭代速度优先

Qwen/DeepSeek 团队每季度训练 1 个模型，快速积累经验。
- 不要一次性投入全部资源做大规模训练
- 先小规模验证，再放大
- **态极对应**：CPU 训练必须先做小规模（<1M tokens）验证 pipeline
- **SmolLM3 教训**：11T 训练到 1T 时才发现张量并行种子 bug，全量重启——系统化测试每个组件才能快速隔离 bug

### 0.5 训练后期数据决定最终行为

SmolLM3 关键洞察：**模型最终行为深受训练末期数据影响**。
- **训练早期**：用丰富、多样化但质量稍低的数据（如网页文本）
- **训练末期（学习率衰减阶段）**：引入稀缺、高质量数据（专业数学/代码）
- **OLMo 3 中期训练**：预训练后用 100B tokens "退火"，针对性提升数学/代码/推理

---

## 一、数据准则

### 1.1 Chinchilla 定律（硬约束）

**DeepMind 2022 年 Chinchilla 论文的核心发现**：大多数模型都 undertrained。

| 模型规模 | Chinchilla 最优 tokens | 实际主流做法 |
|---------|----------------------|------------|
| 10M | 200M | TinyStories: 3.28B (328:1) |
| 36M (compact) | 720M | LLaMA 风格: 5B+ (139:1) |
| 131M (standard) | 2.6B | LLaMA-7B: 1T (143:1) |
| 1.7B (SmolLM2) | 34B | SmolLM2: 11T (6471:1) **overtrain** |
| 3B (SmolLM3) | 60B | SmolLM3: 11.2T (3733:1) **overtrain** |
| 7B (OLMo 3) | 140B | OLMo 3: 6T+100B+50B (428:1) |
| 660B (DeepSeek-R1) | - | R1 训练成本仅 29.4万美元 |

**硬规则**：
- **预训练阶段**：数据/参数比 ≥ 20:1（Chinchilla 最优）
- **推理优化阶段**：可超过 20:1（LLaMA 用 143:1，因为推理更便宜）
- **SmolLM2/3 overtrain**：1.7B/3B 用 11T tokens（6000+ 倍），证明小模型 overtrain 收益巨大
- **Beyond Chinchilla**：loss 在 10,000:1 时仍在下降，不 plateau
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

### 1.3 数据质量筛选（2025 最新实践）

#### 1.3.1 DeepSeek-V3 三层质量过滤体系（14.8T tokens，2025 Nature 封面）

1. **基础过滤（自动化）**：
   - SimHash 去重（重复率 >80% 的文本段去除）
   - Perplexity 评分筛选（阈值 <50）
   - UTF-8 编码 + 字符长度 >512
2. **领域专家审核**：
   - 学术数据：DOI 验证 + 引用量筛选（>10 次引用）
   - 代码数据：语法检查模块验证可执行性
   - 数学数据：公式解析器验证逻辑正确性
3. **对抗性过滤**：
   - 注入 5% 错误数据做鲁棒性训练
   - RLHF 阶段的奖励模型过滤有害内容
   - 多轮人工标注修正（每个样本 ≥3 名标注员审核）

#### 1.3.2 OLMo 3 三级去重策略（极致 token 效率）

OLMo 3 删除了原始池约 **84.6%** 的文档，采用三级去重：
1. **精确去重**（Exact Dedup）：基于文档哈希值，删除完全相同的副本（删除 66% 剩余数据）
2. **近似去重**（MinHash + LSH）：相似度 >80% 的文本段去除
3. **语义去重**（2025 推荐）：基于嵌入模型（如 `BAAI/bge-m3`），threshold=0.92，含义相近的文本去重

#### 1.3.3 质量感知上采样（OLMo 3 创新）

**关键转变**：不是过滤低质量，而是**高倍率上采样高质量数据**。
- 数学、代码等高质量数据高倍率重复训练
- 比简单过滤更有效，能让模型专注学习高价值模式

#### 1.3.4 PreSelect 数据筛选（香港科大 2025）

**核心洞察**："压缩即智能"——能区分模型能力的文本 = 高价值训练数据。
- 通过分析不同模型在文本上的"理解难度"预测训练价值
- 仅用 30B 样本达到传统方法 300B 样本效果，**10 倍效率提升**
- 论文：arXiv:2503.00808v3

#### 1.3.5 DeepSeek-V3 数据配比（精心设计）

| 数据类型 | 占比 | 目标 |
|---------|------|------|
| 学术文献 | 35% | MMLU-Pro 知识覆盖 |
| 代码库 | 25% | HumanEval 代码能力 |
| 多语言文本 | 20% | C-Eval 中文能力 |
| 数学问题 | 12% | MATH 推理能力 |
| 对话数据 | 5% | IF-Eval 对话质量 |
| 其他专业内容 | 3% | DROP 阅读理解 |

**数据质量原则**：
- 多样性 > 数量（n-gram diversity 是可学习性的更强预测器）
- 上采样高质量数据
- 过滤低质量内容（垃圾文本、重复内容）
- **基准去污染**：DeepSeek 仅数学领域预训练就删除 600 万条潜在污染样本

### 1.4 Token 化策略

**主流做法**：
- GPT-2 BPE (tiktoken)：vocab=50,257，通用、成熟
- SentencePiece BPE：vocab=32,000-128,000，多语言友好
- 字符级：vocab=65-200，适合教学/超小模型

**态极的教训**：
- 域专用 vocab=20,000 太小 → 很多中文 token 变成 byte_fallback (<0xXX>)
- byte_fallback token 准确率 88.5% 但占比 24%，拉低整体 argmax
- **✅ T12 已修复**：词表库热插拔（百科 200 万行 + 对话 4.8 万条×3 混合训练，
  50K zh tokenizer，unk 率 0%），token piece 映射 + lm_head 权重迁移无需重训神经元。
  命令：`python scripts/training/upgrade_tokenizer.py` → `python scripts/training/hot_swap_vocab.py`

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

**OLMo 3 退火策略**：
- 预训练后用 100B tokens "中期训练"退火
- 针对性提升数学/代码/推理能力
- 微退火实验：先小规模验证数据源有效性

**硬规则**：
- 必须有 warmup（100-2000 步，视规模）
- 必须有 decay（最后 10-20% 线性或 cosine 衰减）
- **高质量数据放在 decay 阶段**（SmolLM3 + OLMo 3 共同验证）
- **态极当前**：WSD 已实现，正确

### 2.4 优化器

- **AdamW** 是主流（SmolLM3, nanoGPT, LLaMA 全用）
- beta1=0.9, beta2=0.95-0.99
  - 小 batch 时 beta2=0.99（nanoGPT："tokens per iter 少时稍大"）
  - 大 batch 时 beta2=0.95
- weight_decay=0.1（SmolLM3, nanoGPT 标准）
- **embedding 层不加 weight_decay**（SmolLM3 + OLMo 2 发现影响稳定性）

---

## 三、架构准则

### 3.1 纯 Transformer Decoder 是基线

**主流架构**（SmolLM3, nanoGPT, LLaMA, GPT-2, DeepSeek-R1）：
```
Token Embedding (+ Positional Embedding)
  → N × TransformerBlock(
      LayerNorm → CausalSelfAttention → LayerNorm → MLP
    )
  → LayerNorm → LM Head (tied with embedding)
```

**DeepSeek-R1 的关键澄清**：
- R1 主干仍是 **Decoder-only Transformer**（MLA + MoE 骨干）
- 推理模型在**架构上和普通模型没有本质区别**
- 差异在**训练目标与数据管线**（RL + 可验证奖励）
- "推理时更深" = 生成更多中间 token（显式思维链），不是网络层不同

### 3.2 SmolLM3 架构改进（2025 最新）

**关键架构优化**：
- **GQA 机制**：4 组 Grouped Query Attention，性能无损但 KV 缓存大幅降低
- **NoPE 编码**：每隔 4 层移除 RoPE，显著提升长文本处理能力，短文本不受影响
- **文档内注意力屏蔽**：同一序列不同文档的 token 彼此隔离，提升训练稳定性和长文本学习能力
- **嵌入层不衰减权重**（借鉴 OLMo 2）：参数收敛更稳，对性能无负面影响
- **Tied Embeddings**：输入输出嵌入共享，节省 17% 参数

### 3.3 额外组件必须消融验证

**SmolLM3 的做法**：每个架构修改都用 100B tokens 消融验证。

**态极的教训**：
- field_write, field_read_layers, field_projector, domain_prototype 等组件**从未做消融**
- 这些组件增加参数量但是否贡献语言建模能力？未知
- **硬规则**：任何额外组件必须与"去掉该组件的 baseline"对比验证

### 3.4 架构规模选择

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

### 4.2 SmolLM3 可靠评估的四个标准

一个可靠的评估任务应具备：
1. **单调性**：模型越大/数据越好，分数越高
2. **低噪声**：同样配置多次运行，结果稳定
3. **超随机性能**：必须显著优于随机基线
4. **排名一致性**：在不同评估集上排名一致

### 4.3 评估流程

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

## 六、后训练准则（2025 最新）

### 6.1 DeepSeek-R1 训练流程（Nature 2025）

**R1 训练成本**：仅 **29.4 万美元**（512×H800，R1 训练 80 小时，R1-Zero 训练 198 小时）。

**训练流程（4 阶段）**：
1. **冷启动 SFT**：数千条长 CoT 数据微调 DeepSeek-V3-Base
2. **推理 RL**：GRPO + 可验证奖励（数学/代码），激发推理能力
3. **拒绝采样 SFT**：从 RL 检查点采样 + 通用 SFT 数据（写作/事实问答/自我认知），80万混合数据集两轮 SFT
4. **全场景 RL**：兼顾推理 + 通用对话 + 安全

**R1 数据集（15.4 万条）**：
- 数学：2.6 万道定量推理题（考试+竞赛）
- 代码：1.7 万算法竞赛 + 8 千代码修复
- STEM：2.2 万选择题（物理/化学/生物）
- 逻辑：1.5 万真实+合成题
- 通用：6.6 万题（创意写作/事实问答/角色扮演/无害性）

### 6.2 GRPO 算法（Group Relative Policy Optimization）

**核心思想**：每个 prompt 采样 G 个 completions，每个得奖励 r_i，**组内归一化**（减均值除标准差）作为优势。
- **不需要 value network（critic）**，大幅节省训练成本
- G 越大，组均值方差越低，但 rollout 成本越高
- **可验证奖励**：数学对错可判定、代码单元测试 pass/fail，比人类偏好标签更稳定

**"Aha Moment"**：训练中模型自发学会使用 "wait" 进行反思，推理能力发生相变式跳跃。
- AIME 2024 pass@1：15.6% → 71.0%（自洽解码 86.7%）

### 6.3 OLMo 3 后训练三路径

- **Think（推理）**：SFT → DPO（Delta Learning）→ RLVR（改进 GRPO）
- **Instruct（对话）**：SFT + 偏好优化
- **RL-Zero**：纯 RL 路径

**Delta Learning（OLMo 3 创新）**：
- 用强模型（如 Qwen-32B）生成"胜出"回答
- 用弱模型（如 Qwen-0.6B）生成"拒绝"回答
- 巨大的质量差异提供更强学习信号

### 6.4 OpenThoughts3 发现（2025-06）

**纯 SFT（无 RL）就能达到 SOTA**：
- OpenThinker3-7B：AIME 2025 53%，LiveCodeBench 51%，GPQA 54%
- 比 DeepSeek-R1-Distill-Qwen-7B 提升 15-20 个百分点
- 关键是数据集 OpenThoughts3-1.2M（85万数学+25万代码+10万科学）
- 用 QwQ-32B 标注推理轨迹

**5 个关键洞察**（1000+ 实验得出）：
1. 每个 question 从 teacher 采样多个答案，能让数据源规模至少扩大 16 倍
2. **模型性能好 ≠ 当老师好**：QwQ-32B 是更好的老师（尽管自身分数低于 DeepSeek-R1）
3. 各种验证和答案过滤方法都没显著提升
4. 从少量高质量源（top 1-2）选题 > 多样性优化（top 8-16）
5. 用 LLM 标注的难度/响应长度过滤 > 传统嵌入/fastText 过滤

### 6.5 态极后训练启示

- **可验证奖励优先**：态极若有可判分任务（如路由准确率），用 GRPO 而非人类偏好
- **数据 > 算法**：OpenThoughts3 证明纯 SFT 配好数据就能 SOTA
- **不一定要大模型当老师**：选 teacher 要做实验对比，不是选最强模型
- **多答案采样扩大数据**：一个 question 多个 rollout 比多个 question 更高效

---

## 七、常见陷阱（态极已踩过的）

### 7.1 数据量不足（Chinchilla 违规）
- **症状**：loss 下降但生成乱码；argmax 卡在天花板
- **诊断**：计算实际训练 tokens / 参数比，应 ≥ 20:1
- **修复**：增加数据量或减小模型

### 7.2 数据复杂度不匹配
- **症状**：模型对"见过的"高置信度正确，对"没见过的"完全不知道
- **诊断**：Top-5 几乎不高于 Top-1（正常应 +15-25%）
- **修复**：用更简单的数据（TinyStories 级别）

### 7.3 batch_size 太小
- **症状**：训练不稳定，loss 震荡
- **诊断**：batch < 32
- **修复**：增加 batch_size 或用梯度累积

### 7.4 额外组件未消融
- **症状**：架构有额外组件但性能不如纯 transformer
- **诊断**：与去掉额外组件的 baseline 对比
- **修复**：消融验证，去掉无用的组件

### 7.5 评估方式错误
- **症状**：追求 argmax 85% 但生成仍乱码
- **诊断**：argmax 是非主流指标
- **修复**：改用 PPL + 生成质量评估

### 7.6 学习率不匹配
- **症状**：loss 下降慢或不收敛
- **诊断**：对照同规模模型的主流 lr
- **修复**：小模型用 1e-3，中模型用 6e-4

### 7.7 保存末步而非 best
- **症状**：末步性能远差于训练中最佳
- **诊断**：末步 loss vs best loss 差距大
- **修复**：保存 best val loss 的 checkpoint（已修复）

### 7.8 大规模训练隐藏 bug（SmolLM3 教训）
- **症状**：小规模测试正常，大规模训练性能异常
- **SmolLM3 案例**：11T 训练到 1T 时发现张量并行相同种子 bug，全量重启
- **诊断**：系统化测试每个组件，能快速隔离 bug
- **修复**：保留小规模验证集，定期对照预期性能

### 7.9 数据污染（DeepSeek 教训）
- **症状**：基准测试分数异常偏高，但实际能力弱
- **DeepSeek 做法**：仅数学领域预训练就删除 600 万条潜在污染样本
- **修复**：用 Duplodocus 等工具做基准去污染

---

## 八、态极项目的训练检查清单

每次启动训练前，必须确认：

- [ ] **数据/参数比 ≥ 20:1**（或明确说明为何违反）
- [ ] **数据复杂度匹配模型规模**（小模型不用复杂数据）
- [ ] **数据三级去重**：精确 Hash + MinHash LSH + 嵌入语义去重
- [ ] **基准去污染**：移除与评估集重叠的数据
- [ ] **batch_size ≥ 32**（或梯度累积等效）
- [ ] **学习率匹配规模**（小模型 1e-3，中模型 6e-4）
- [ ] **有 warmup + decay 调度**
- [ ] **高质量数据放在 decay 阶段**
- [ ] **embedding 层不加 weight_decay**
- [ ] **评估用 PPL + 生成质量**，不用 argmax
- [ ] **评估满足四标准**：单调性 + 低噪声 + 超随机 + 排名一致
- [ ] **保存 best val loss checkpoint**
- [ ] **额外组件有消融 baseline 对比**（或明确标注"未验证"）
- [ ] **先小规模验证 pipeline**，再放大

---

### 8.1 Hub 神经元正式训练操作指南（2026-08）

hub = EXPERT 规格（hidden 1024 / 14 层 / field 4096）+ general 256K lm_head，495M 参数，
数据 = 域 SFT 混合 31000 + 跨域平行语料 1629 对（zh 指令→code 响应）。smoke 链路已通
（2 步 loss 11.46，10s/步 CPU）；正式训练待 GPU 执行。

```powershell
# GPU（推荐）——495M × 32629 条，CPU 预估 40+ 小时，GPU 约 1-3 小时
python -u scripts/training/train_hub_neuron.py --epochs 2 --max-steps 16000 `
    --device cuda --out-name neuron_hub_formal --save-every 500

# 日志落盘（N3 规范）
python -u scripts/training/train_hub_neuron.py --epochs 2 --max-steps 16000 `
    --device cuda --out-name neuron_hub_formal 2>&1 | Tee-Object -FilePath ("logs\train_hub_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".log")

# 中断续训（8-12 中断事件教训：--resume 恢复权重+loss_history，预算重计）
python -u scripts/training/train_hub_neuron.py --resume --epochs 1 `
    --max-steps 8000 --device cuda --out-name neuron_hub_formal
```

**完成后依次执行**（协作层正式训练 + 评估）：
1. `train_cross_domain_collab.py --hub-path <hub 产物> --hub-anchor-weight 0.5 --hub-contrastive-weight 1.0 --unified-field-dim 3072`（正式协作层，替换 hub_collab_v2 smoke 产物）
2. 重跑 `verify_wcond_ab.py --collab-ckpt <正式协作层产物>`——R1 收益判定（当前 smoke 产物 Δ=5.6e-5，无收益证据）
3. 阶段 4 跨域评估（锚点 cos > 0.5 目标）

**检查清单对照**：数据/参数比 32629×~500 token / 495M ≈ 33:1（达标）；batch=4 偏小（CPU 限制，GPU 可调大）；
lr 5e-4 无 warmup（域 SFT 成功配方沿用）；每 500 步保存 best+回读验证（防坏产物，8-12 教训已内建）。

---

## 九、参考资源

### 9.1 2025-2026 最新（必读）

- [DeepSeek-R1 Nature 论文](https://www.nature.com/articles/s41586-025-09422-z) — 首个同行评审主流大模型，2025-09-17
- [DeepSeek-R1 同行评审报告](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM2_ESM.pdf) — 64 页评审文件
- [Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/) — HuggingFace 2025-10-30，214 页训练实战指南
- [SmolLM3 模型](https://hf.co/HuggingFaceTB/SmolLM3-3B) — 3B 最强小模型，100% 开源
- [SmolLM2 论文](https://openreview.net/forum?id=3JiCl2A14H) — COLM 2025，1.7B 11T overtrain
- [OLMo 3](https://huggingface.co/allenai) — AI2 全流程开源（数据+代码+检查点）
- [OpenThoughts3](https://www.open-thoughts.ai/blog/ot3) — 1.2M 推理数据，纯 SFT 达 SOTA
- [gpt-oss](https://openai.com/index/introducing-gpt-oss/) — OpenAI 2025-08，Apache 2.0 开源
- [PreSelect 论文](https://arxiv.org/abs/2503.00808) — 香港科大 2025，10 倍数据效率

### 9.2 经典基础

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT 实现，CPU 友好
- [TinyStories 论文](https://arxiv.org/abs/2305.07759) — 小模型生成连贯文本的里程碑
- [Chinchilla 论文](https://arxiv.org/abs/2203.15556) — 数据/参数比 20:1 的来源
- [Beyond Chinchilla-Optimal](https://arxiv.org/pdf/2401.00448v3) — 推理考虑下的扩展定律
- [StoryGPT 实践](https://app.readytensor.ai/publications/storygpt-pretraining-a-small-language-model-from-scratch-on-tinystories-ZzOynh7puXuD) — TinyStories 训练实例

---

## 十、版本历史

- **v1.0 (2026-07-25)**：基于社区学习创建。核心发现：当前训练方法违反 Chinchilla 定律（差 111 倍）、数据复杂度不匹配、batch_size 太小、评估方式非主流。
- **v2.0 (2026-07-25)**：整合 2025-2026 最新开源成果：
  - **DeepSeek-R1 Nature 论文**（2025-09-17）：29.4 万美元训练成本、GRPO 算法、4 阶段训练流程、"Aha Moment"
  - **SmolLM3**（2025-07）：NoPE + 文档内注意力屏蔽 + 嵌入层不衰减；11T 训练 1T 后发现种子 bug 全量重启教训
  - **OLMo 3**（2025）：三阶段基础训练（预训练+中期训练退火+长上下文）+ 微退火实验 + Delta Learning + 质量感知上采样
  - **OpenThoughts3**（2025-06）：纯 SFT 达 SOTA；模型性能好 ≠ 当老师好
  - **gpt-oss**（2025-08）：OpenAI Apache 2.0 开源
  - **PreSelect**（香港科大 2025）：用模型预测能力筛选数据，10 倍效率
  - 新增第六章"后训练准则"，第八章检查清单增加 5 项
