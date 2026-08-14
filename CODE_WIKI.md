# 态极神经元 (Taiji Neuron) Code Wiki

> **版本**: v2.0
> **日期**: 2026-07-17
> **项目状态**: 🏗️ Architecture Ready — 架构就绪，待更多训练验证

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [核心模块详解](#3-核心模块详解)
   - [共振场引擎 (Resonance Field)](#31-共振场引擎-resonance-field)
   - [大脑模块 (Brain)](#32-大脑模块-brain)
   - [训练管线 (Training)](#33-训练管线-training)
   - [基础组件 (Layers)](#34-基础组件-layers)
   - [Agent系统](#35-agent系统)
   - [生命系统 (Life)](#36-生命系统-life)
   - [核心基础设施 (Core)](#37-核心基础设施-core)
4. [API层](#4-api层)
5. [前端模块](#5-前端模块)
6. [依赖关系](#6-依赖关系)
7. [项目运行方式](#7-项目运行方式)
8. [关键设计决策](#8-关键设计决策)
9. [实验结论](#9-实验结论)
10. [技术债务与后续工作](#10-技术债务与后续工作)

---

## 1. 项目概述

态极神经元是从一代态极（1.5B 单体模型）进化而来的第二代架构。核心思想是**用多个领域专用的小神经元替代一个大模型**，通过共振场实现神经元间的知识协作。

### 与一代的关键区别

| | 一代 (Taiji v1) | 二代 (Taiji Neuron) |
|---|---|---|
| 大脑 | 1.5B 单体 ModelSelf | 5+ 个领域神经元 (24M-118M) |
| 推理 | 单体 forward | 共振场多轮协作 |
| 训练 | 端到端预训练 | 蒸馏 + 对比学习 |
| 扩展 | 重新训练 | 热插拔新神经元 |
| 硬件 | GPU 必需 | CPU 可训练+推理 |

---

## 2. 整体架构

### 三层设计

```
Level 0: 通用分词器 (256K)      ← I/O 协议层，可替换
    ↓
Level 1: 领域神经元 (5+个)       ← 独立 Transformer，领域专用
    ↓  field_write / field_read
Level 2: 共振场 (4096-dim)      ← 共享意识，独立于分词器
```

### 推理流程

```
输入文本
    ↓ tokenizer (256K)
共享嵌入 (512-dim)
    ↓
┌─────────────────────────────────────────┐
│  Round 1: 所有神经元独立前向              │
│    zh ─→ field_vector ─→ 写入场          │
│    en ─→ field_vector ─→ 写入场          │
│    code ─→ field_vector ─→ 写入场        │
│    math ─→ field_vector ─→ 写入场        │
│    general ─→ field_vector ─→ 写入场     │
│                                          │
│  ┌─ ConfidenceGate: 是否需要共振? ──┐    │
│  │  如果 max_prob > 0.9 → 跳过共振  │    │
│  └──────────────────────────────────┘    │
│                                          │
│  Round 2-N: 条件化共振                    │
│    读场状态 → 条件化前向 → 重新写入       │
│    ┌─ QualityFilter: 过滤弱神经元 ──┐    │
│    └─ EarlyStop: logits 收敛即停 ──┘    │
└─────────────────────────────────────────┘
    ↓
┌─ 分工路径: 集群主导 × 规模分层 ─┐
│  主导集群权重 0.7，辅助集群 0.3  │
│  集群内部: expert×3 > standard×2 │
│           > compact×1            │
└──────────────────────────────────┘
    ↓
加权 logits → 下一个 token
```

---

## 3. 核心模块详解

### 3.1 共振场引擎 (Resonance Field)

**模块位置**: [taiji/resonance/](file:///e:/taiji-neuron/taiji/resonance)

#### 3.1.1 ResonanceField

**文件**: [field.py](file:///e:/taiji-neuron/taiji/resonance/field.py)

共享共振场 - 架构的"神经语言"。所有神经元在此写入 L2 归一化向量并读取累积状态。

**关键属性**:
- `dim`: 场维度，默认 4096
- `state`: 当前场状态向量
- `W_cond`: 条件化投影矩阵（可学习的门控参数）
- `_contributions`: 各神经元的贡献记录
- `scores`: 各神经元的共振分数

**核心方法**:

| 方法 | 功能 | 参数 | 返回值 |
|---|---|---|---|
| `reset(batch_size)` | 重置场状态 | batch_size: 批次大小 | None |
| `write(neuron_id, vector)` | 写入场向量 | neuron_id: 神经元ID, vector: 场向量 | 归一化后的向量 |
| `score(vector, neuron_id)` | 计算共振分数 | vector: 向量, neuron_id: 排除的神经元 | 余弦相似度分数 |
| `prediction_complementarity(a_logits, b_logits, targets)` | 计算预测互补性 | a/b_logits: 两个神经元的logits | 互补性分数 |
| `get_normalised_state()` | 获取归一化场状态 | 无 | 归一化后的状态向量 |

**关键修复 (H系列)**:
- **H2**: 支持 per-sample 场状态，避免跨样本污染
- **H5**: 使用 leave-one-out 计算分数，排除自身贡献
- **H6**: 基于预测的互补性计算，而非几何正交性
- **H8**: W_cond 作为乘法门控激活使用

---

#### 3.1.2 ResonanceNeuron

**文件**: [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)

单个共振神经元 — 独立 Transformer + 场接口。

**架构组成**:

```
共享嵌入 (512-dim)
    ↓
embed_adapter → [B, L, hidden]
    ↓
TransformerBlock × N
    ↓
norm
    ↓
field_write → [B, D] L2-normalised
```

**关键属性**:
- `embed_adapter`: 共享嵌入 → 神经元内部维度的投影
- `layers`: Transformer层列表
- `field_write`: 场写入投影
- `field_read_layers`: 每层一个的场读取投影
- `lm_head`: 语言建模头（用于PPL评估）
- `field_pool_query`: 注意力池化查询（v2新特性）
- `field_read_gate`: 位置门控读取（v2新特性）

**核心方法**:

| 方法 | 功能 |
|---|---|
| `forward(shared_embeddings, field_state, round_num, return_logits)` | 前向传播，支持条件化共振 |
| `freeze_fingerprint()` | 冻结方向指纹（用于预筛选） |
| `quick_probe(shared_embeddings)` | 轻量级前向（跳过完整Transformer） |

**v1 vs v2**:
- v1: 最后token写入 + 广播读取
- v2: 注意力池化写入 + 位置门控读取

---

#### 3.1.3 ResonanceEnsemble

**文件**: [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)

编排多轮共振推理的核心类。

**初始化参数**:

| 参数 | 类型 | 说明 |
|---|---|---|
| `neurons` | Dict[str, ResonanceNeuron] | 神经元字典 |
| `field` | ResonanceField | 共振场实例 |
| `max_rounds` | int | 最大共振轮数 |
| `confidence_gate` | ConfidenceGate | 置信度门控 |
| `early_stop` | EarlyStopResonance | 早停机制 |
| `quality_filter` | QualityFilter | 质量过滤器 |
| `division_path` | DivisionPath | 分工路径 |

**核心方法**:

| 方法 | 功能 |
|---|---|
| `forward(shared_embeddings, return_logits, enable_gating)` | 完整共振循环 |
| `evaluate_ppl(dataloader, shared_embedding)` | 评估困惑度 |
| `evaluate_single_neuron(neuron, dataloader, shared_embedding)` | 单神经元PPL评估 |

**共振流程**:
1. Round 1: 所有神经元独立前向，写入场
2. 门控检查：是否需要共振？
3. Round 2-N: 条件化共振（读取场状态）
4. 动态阈值过滤低共振神经元
5. 早停检查：logits是否收敛？
6. 最终输出：加权logits

---

#### 3.1.4 门控机制 (Gating)

**文件**: [gating.py](file:///e:/taiji-neuron/taiji/resonance/gating.py)

实验12发现的三个关键机制：

##### ConfidenceGate

当预测已经足够自信（max_prob > threshold）时跳过共振，避免场噪声。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `threshold` | 0.9 | 置信度阈值 |

##### EarlyStopResonance

当logits收敛时停止迭代。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `threshold` | 1e-3 | L2差阈值 |
| `min_rounds` | 2 | 最小轮数 |

##### ResonanceTrigger

综合触发条件：
1. 预测不确定性高（置信度门控）
2. 多个神经元有互补知识（多样性检查）
3. 有足够的改进空间

---

#### 3.1.5 质量过滤 (QualityFilter)

**文件**: [quality.py](file:///e:/taiji-neuron/taiji/resonance/quality.py)

实验9结论：弱神经元会稀释强神经元。质量过滤确保只有PPL < 100的神经元参与共振。

**方法**:
- `filter(neuron_ids)`: 静态阈值过滤
- `filter_adaptive(neuron_ids)`: 自适应阈值过滤（best_ppl × 2）

---

#### 3.1.6 分工路径 (DivisionPath)

**文件**: [division.py](file:///e:/taiji-neuron/taiji/resonance/division.py)

两种互补策略的组合：

##### ScaleLayering (策略A)

不同规模的神经元承担不同角色：

| 规格 | 参数量 | 角色 | 权重 |
|---|---|---|---|
| expert | ~118M | 决策+把关 | 3.0 |
| standard | ~59M | 主要执行 | 2.0 |
| compact | ~24M | 辅助执行 | 1.0 |

##### ClusterDominance (策略B)

最佳匹配集群主导，其他集群辅助：
- 主导集群权重：0.7
- 辅助集群权重：0.3（按拟合度分配）

##### DivisionPath

组合策略：集群主导 × 内部规模分层

---

#### 3.1.7 神经元配置 (NeuronConfig)

**文件**: [config.py](file:///e:/taiji-neuron/taiji/resonance/config.py)

**标准规格**:

| 规格 | hidden_size | num_layers | num_heads | field_dim | 参数量 |
|---|---|---|---|---|---|
| COMPACT | 512 | 6 | 8 | 2048 | ~24M |
| STANDARD | 768 | 10 | 12 | 3072 | ~59M |
| FOUNDATION | 384 | 6 | 6 | 4096 | ~18M |
| EXPERT | 1024 | 14 | 16 | 4096 | ~118M |

> R21（2026-08-14）：field_dim 按 config.py 实际值修正（原表 COMPACT=3072 / FOUNDATION=3072 为过时声明；FOUNDATION/EXPERT 同为 4096，但 COMPACT=2048 / STANDARD=3072 并非"统一 4096"）。

**参数计算属性**:
- `approx_params_m`: 估算参数量（百万）

---

#### 3.1.8 共享嵌入投影 (SharedEmbedProj)

**文件**: [shared_embed.py](file:///e:/taiji-neuron/taiji/resonance/shared_embed.py)

解决问题 **H10**：蒸馏训练中使用的随机正交投影从未保存，导致验证脚本使用不同的投影，神经元从未看到训练时的嵌入分布。

**核心类**: `SharedEmbedProj`

| 方法 | 功能 |
|---|---|
| `__init__(src_dim=2048, target_dim=512)` | 创建投影层，正交初始化 |
| `forward(emb)` | 将教师嵌入投影到神经元基础维度 |
| `save(path)` | 保存投影权重 |
| `load(path)` | 加载并冻结投影权重 |

**使用示例**:
```python
from taiji.resonance.shared_embed import SharedEmbedProj
proj = SharedEmbedProj.load("data/shared_proj.pt")
emb_512 = proj(teacher_emb_2048)  # [B, L, 2048] -> [B, L, 512]
```

---

#### 3.1.9 分词翻译器 (TokenTranslator & TokenizerHub)

**文件**: [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py)

每个神经元可以有自己的领域专用分词器（32K-48K tokens），通用256K分词器作为公共I/O协议。

**核心类**:

##### TokenTranslator

双向翻译领域分词器和通用分词器。

| 方法 | 功能 |
|---|---|
| `build_alignment(domain_tokenizer, general_tokenizer)` | 构建对齐表 |
| `domain_to_general(domain_ids)` | 领域token → 通用token |

##### TokenizerHub

管理可热插拔的领域分词器。

| 方法 | 功能 |
|---|---|
| `register_domain(domain, tokenizer)` | 注册新领域分词器 |
| `get_tokenizer(domain)` | 获取领域分词器 |
| `encode(text, domain)` | 使用指定领域分词器编码 |
| `build_translator(domain)` | 构建翻译器 |

---

#### 3.1.10 部落压缩 (Tribal)

**文件**: [tribal.py](file:///e:/taiji-neuron/taiji/resonance/tribal.py)

部落压缩质量量化模块，将N个成员的内部子场动态压缩为一个4096维单位向量写入上级场。

**核心类**:

##### TribalMetrics

量化两件事：
1. **信号质量因子 Q = α·β·γ** — 压缩后的输出有多可信
2. **压缩损失** — 单向量能否代表所有成员的输出

**三个内部指标**:

| 指标 | 符号 | 含义 |
|---|---|---|
| 内部相干度 | α | 成员写入向量的pairwise cosine均值 |
| 收敛速度 | β | 子场状态在轮次间的指数衰减加权稳定度 |
| 方向散布度 | γ | 成员方向在质心周围的空间集中度 |

**核心方法**:

| 方法 | 功能 |
|---|---|
| `record_round(member_writes, sub_field_state)` | 记录一轮共振后的部落状态 |
| `quality_factor()` | 计算 Q = α·β·γ |
| `compression_loss()` | 计算压缩损失 |
| `summary()` | 返回所有指标汇总 |

##### TribeSuperNeuron

部落超级神经元——在上级场中表现为一个普通神经元。

| 方法 | 功能 |
|---|---|
| `forward_tribe(input_ids, max_rounds)` | 执行部落内部完整共振循环 |
| `compute_resonance(parent_field_state)` | 计算质量因子调权后的共振度 |
| `freeze_fingerprint()` | 固化部落指纹 |
| `get_status()` | 获取部落状态摘要 |

**解散条件**: 压缩损失 > 0.5 时建议解散部落。

##### CoactivationTracker

跨神经元共激活矩阵追踪，用于检测应该主动部落化的神经元组。

---

### 3.2 大脑模块 (Brain)

**模块位置**: [taiji/brain/](file:///e:/taiji-neuron/taiji/brain)

#### Cortex

**文件**: [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py)

共振场意识中心，封装ResonanceEnsemble的高层接口。

**核心方法**:

| 方法 | 功能 |
|---|---|
| `__init__(neurons_dir, device, max_rounds)` | 加载神经元并创建共振场 |
| `set_tokenizer(tokenizer)` | 设置分词器 |
| `set_shared_embedding(embedding)` | 设置共享嵌入 |
| `think(input_ids)` | 运行一轮共振思考 |
| `generate(prompt, max_tokens, temperature, top_k)` | 生成文本 |
| `get_field_state()` | 获取当前场状态 |
| `get_dominant_domain()` | 获取主导领域 |

**使用示例**:
```python
from taiji.brain.cortex import Cortex

cortex = Cortex(neurons_dir="data/neurons")
cortex.set_tokenizer(tokenizer)
result = cortex.generate("今天天气怎么样？", max_tokens=256)
```

---

### 3.3 训练管线 (Training)

**模块位置**: [taiji/training/](file:///e:/taiji-neuron/taiji/training)

#### 3.3.1 蒸馏训练 (Distill)

**文件**: [distill.py](file:///e:/taiji-neuron/taiji/training/distill.py)

从1.5B教师模型蒸馏到单个神经元。

**核心函数**:

| 函数 | 功能 |
|---|---|
| `distill_neuron(teacher, student, dataloader, shared_embedding, num_steps)` | 蒸馏训练 |
| `extract_teacher_directions(teacher, dataloader)` | 提取教师方向向量 |

**损失组合**:
- LM loss (0.7): 语言建模损失
- Distill loss (0.3): 隐藏状态对齐损失

---

#### 3.3.2 单神经元训练 (Single)

**文件**: [single.py](file:///e:/taiji-neuron/taiji/training/single.py)

**函数**: `train_single_neuron(neuron, dataloader, shared_embedding, num_steps)`

用于从头训练新领域的神经元。

---

#### 3.3.3 联合训练 (Joint)

**文件**: [joint.py](file:///e:/taiji-neuron/taiji/training/joint.py)

**类**: `JointTrainingLoop`

多神经元同步训练，共振场作为知识传递的共享内存。

**损失组成**:
1. LM loss: 语言建模
2. Contrastive loss: 场写入差异化（LUCKY v4关键）
3. Niche-seeking loss: 防止同质化（训练30%后启用）

---

#### 3.3.4 对比学习 (Contrastive)

**文件**: [contrastive.py](file:///e:/taiji-neuron/taiji/training/contrastive.py)

**函数**: `train_field_write_contrastive(neuron, dataloader, domain_label, other_domain_vectors)`

专门训练field_write投影，使同一领域的向量更接近，不同领域的向量更远。

---

#### 3.3.5 Checkpoint桥接 (CheckpointBridge)

**文件**: [checkpoint_bridge.py](file:///e:/taiji-neuron/taiji/training/checkpoint_bridge.py)

加载一代1.5B模型用于蒸馏，从一代项目路径直接导入，避免与taiji-neuron包冲突。

**核心函数**:

| 函数 | 功能 |
|---|---|
| `load_teacher_model(checkpoint_dir, device, dtype)` | 加载一代1.5B教师模型 |
| `extract_hidden_states(teacher_model, input_ids)` | 提取教师隐藏状态用于蒸馏 |

**关键技术**:
- 使用 `importlib` 从特定路径加载一代模块
- 处理一代不同版本的模型结构差异
- 支持 `backbone.layers` 和直接 `.layers` 两种结构

---

### 3.4 基础组件 (Layers)

**文件**: [layers.py](file:///e:/taiji-neuron/taiji/layers.py)

LLaMA 3风格的Transformer组件：

| 类 | 功能 |
|---|---|
| `RMSNorm` | RMS归一化（比LayerNorm更快） |
| `RotaryEmbedding` | 旋转位置编码（带LRU缓存） |
| `GroupedQueryAttention` | GQA分组查询注意力 |
| `SwiGLU` | 门控激活函数 |
| `TransformerBlock` | Pre-Norm Transformer块 |

**关键特性**:
- 自动调度Flash Attention（PyTorch 2.0+）
- 兼容旧版PyTorch的回退实现
- RoPE缓存限制（最多4个条目）

---

### 3.5 Agent系统

**模块位置**: [taiji/agent/](file:///e:/taiji-neuron/taiji/agent)

| 模块 | 功能 |
|---|---|
| `reflector.py` | 反思系统 |
| `planner.py` | 规划系统 |
| `perception.py` | 感知系统 |
| `memory.py` | 记忆系统 |
| `working_memory.py` | 工作记忆 |
| `context_manager.py` | 上下文管理 |
| `semantic_memory.py` | 语义记忆 |

---

### 3.6 生命系统 (Life)

**模块位置**: [taiji/life/](file:///e:/taiji-neuron/taiji/life)

| 模块 | 功能 |
|---|---|
| `life_scheduler.py` | 生命调度器（心跳循环） |
| `feed_engine.py` | 喂养引擎 |
| `sleep_engine.py` | 睡眠引擎 |
| `play_engine.py` | 玩耍引擎 |
| `evolution_engine.py` | 进化引擎 |
| `explore_engine.py` | 探索引擎 |
| `science_engine.py` | 科学引擎 |
| `recursive_improver.py` | 递归改进器 |
| `life_interface.py` | 生命接口 |

---

### 3.7 核心基础设施 (Core)

**模块位置**: [taiji/core/](file:///e:/taiji-neuron/taiji/core)

| 模块 | 功能 |
|---|---|
| `utils.py` | 通用工具（日志、路径、JSON等） |
| `plugin_manager.py` | 插件管理器 |
| `security.py` | 安全模块（认证管理） |
| `app_state.py` | 应用状态管理 |
| `hardware.py` | 硬件检测 |
| `pii_sanitizer.py` | PII脱敏 |
| `websocket_server.py` | WebSocket服务器 |

---

## 4. API层

**模块位置**: [api/](file:///e:/taiji-neuron/api)

### 路由模块

| 路由文件 | 功能 |
|---|---|
| `routes_agent.py` | Agent管理 |
| `routes_agent_memory.py` | Agent记忆 |
| `routes_agent_workspace.py` | Agent工作空间 |
| `routes_agent_mcp.py` | Agent MCP工具调用 |
| `routes_auth.py` | 认证 |
| `routes_chat.py` | 聊天 |
| `routes_life.py` | 生命系统 |
| `routes_models.py` | 模型管理 |
| `routes_model_switch.py` | 模型切换 |
| `routes_multimodal.py` | 多模态处理 |
| `routes_plugins.py` | 插件管理 |
| `routes_rag.py` | RAG检索 |
| `routes_runtime.py` | 运行时 |
| `routes_settings.py` | 设置 |
| `routes_system.py` | 系统 |
| `routes_taiji.py` | 态极核心接口 |
| `routes_taiji_model.py` | 态极模型管理 |
| `routes_terminal.py` | 终端管理 |
| `routes_training.py` | 训练 |
| `routes_update.py` | 更新 |
| `routes_workflows.py` | 工作流管理 |

### 中间件

**目录**: [api/middleware/](file:///e:/taiji-neuron/api/middleware)

| 中间件 | 功能 |
|---|---|
| `metrics.py` | 指标监控 |
| `security.py` | 安全中间件 |

### API核心模块

| 文件 | 功能 |
|---|---|
| `api_server.py` | API服务器配置 |
| `chat_strategies.py` | 聊天策略管理 |
| `cli.py` | 命令行接口 |
| `models.py` | API数据模型定义 |
| `models_runtime.py` | 运行时模型管理 |
| `run_app.py` | 应用运行入口 |

### 训练API

**目录**: [api/training/](file:///e:/taiji-neuron/api/training)

| 文件 | 功能 |
|---|---|
| `common.py` | 训练通用工具 |
| `control.py` | 训练控制 |
| `datasets.py` | 数据集管理 |
| `publish.py` | 模型发布 |
| `recommend.py` | 训练推荐 |
| `resume.py` | 训练恢复 |
| `stream.py` | 训练流处理 |
| `taiji_train.py` | 态极训练专用 |

### 应用工厂

**文件**: [app.py](file:///e:/taiji-neuron/api/app.py)

`create_app(startup_tasks=True)` 创建FastAPI应用，包含：
- 中间件配置
- 路由注册
- 静态资源挂载
- 生命周期管理

---

## 5. 前端模块

**模块位置**: [frontend/](file:///e:/taiji-neuron/frontend)

### 技术栈

| 技术 | 版本 | 用途 |
|---|---|---|
| Vue | 3.5+ | 前端框架 |
| Vue Router | 4.6+ | 路由 |
| Pinia | 3.0+ | 状态管理 |
| Naive UI | 2.40+ | UI组件库 |
| Monaco Editor | 0.55+ | 代码编辑器 |
| xterm.js | 6.0+ | 终端 |
| Vite | 8.0+ | 构建工具 |

### 核心组件

| 组件 | 功能 |
|---|---|
| `AppSidebar.vue` | 侧边栏导航 |
| `ChatView.vue` | 聊天视图 |
| `MonacoEditor.vue` | 代码编辑器 |
| `WebTerminal.vue` | Web终端 |
| `ToastManager.vue` | 通知管理 |
| `RuntimeExceptionCenter.vue` | 运行时异常中心 |

### 视图页面

| 视图 | 功能 |
|---|---|
| `WorkspaceView.vue` | 工作空间 |
| `AgentConfigView.vue` | Agent配置 |
| `KBView.vue` | 知识库 |
| `TrainingView.vue` | 训练管理 |
| `SettingsView.vue` | 设置 |
| `LifeStatusView.vue` | 生命状态 |

### Composables

| Composable | 功能 |
|---|---|
| `useApi.js` | API调用 |
| `useAuth.js` | 认证管理 |
| `useChatUpload.js` | 聊天上传 |
| `useMarkdown.js` | Markdown处理 |
| `useSettings.js` | 设置管理 |
| `useWebSocket.js` | WebSocket连接 |

---

## 6. 依赖关系

### 核心依赖

| 类别 | 依赖 | 版本 |
|---|---|---|
| 深度学习 | torch | 2.0.0+ |
| | transformers | 4.40.0+ |
| | sentencepiece | 0.2.0+ |
| | peft | 0.8.0+ |
| | accelerate | 0.28.0+ |
| API框架 | fastapi | 0.110.0+ |
| | uvicorn | 0.27.0+ |
| | pydantic | 2.0.0+ |
| Agent | langchain | 0.1.0+ |
| | beautifulsoup4 | 4.12.0+ |
| RAG | sentence-transformers | 2.2.0+ |
| | jieba | 0.42.0+ |
| 数据处理 | pandas | 2.0.0+ |
| | numpy | 1.24.0+ |
| 训练 | datasets | 2.18.0+ |
| | tensorboard | 2.14.0+ |

### 可选依赖

| 类别 | 依赖 | 用途 |
|---|---|---|
| GPU | bitsandbytes | 量化 |
| | scipy | 科学计算 |
| 桌面 | PyQt6 | 桌面端 |
| 语音 | edge-tts, pyttsx3 | 语音合成 |
| 构建 | pyinstaller | 打包 |

---

## 7. 项目运行方式

### 7.1 快速验证

```bash
# 口径契约 + 共振 side_channels 回归（16 用例）
python -m pytest tests/ -q
# 预期: 16 passed
```

> 注：原 `test_distill_bridge.py` / `test_division_path.py` 已随 2026-08 训练管线重构退役；当前机制级验证脚本位于 `scripts/training/verify_*.py`（运行日志落盘 `logs/`，N3 规范）。

### 7.2 训练管线（当前链路）

```bash
# ① 领域 SFT 微调（对话神经元）
python scripts/training/finetune_neuron_dialogue.py

# ② 协作层训练（side_channels + 跨规格投影）
python scripts/training/finetune_cross_spec.py
python scripts/training/finetune_side_channels.py

# ③ 跨域协作层联合训练（含 hub，可选 --hub-path）
python scripts/training/train_cross_domain_collab.py

# ④ hub 神经元训练（EXPERT 规格 + general 256K，从零）
python scripts/training/train_hub_neuron.py

# ⑤ 回合级质量判定头训练
python scripts/training/train_round_level_quality.py
```

> 原蒸馏管线（`prepare_distill_data.py` / `distill_neurons.py` / `verify_distilled_neurons.py`）为一代→二代迁移期的临时产物，已归档退役；当前 neurons 均为独立 SFT 训练（详见 `plans/HISTORY_DIALOGUE_TRAINING.md`）。

### 7.3 启动API服务

```bash
# 启动完整服务（模型+前端）
python api/main.py

# 仅加载模型（无UI）
python api/main.py --no-ui

# 训练模式
python api/main.py --train
```

### 7.4 使用Cortex

```python
from taiji.brain.cortex import Cortex

cortex = Cortex(neurons_dir="data/neurons")

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(os.path.join(os.environ.get("TAICHI_TEACHER_PATH", "checkpoint-481000"), "sentencepiece.model"))
cortex.set_tokenizer(sp)

result = cortex.generate("今天天气怎么样？", max_tokens=256)
```

### 7.5 前端开发

```bash
cd frontend
npm install
npm run dev    # 开发模式
npm run build  # 生产构建
```

---

## 8. 关键设计决策

### 8.1 为什么用共振场？

- **知识协作**: 多个领域专家通过共享场进行知识交换
- **去中心化**: 没有单一控制点，神经元平等协作
- **动态路由**: 根据场状态动态选择最佳神经元组合
- **硬件友好**: 小神经元可在CPU上运行

### 8.2 门控机制的必要性

实验12结论：1+1>2不是默认行为
- **ConfidenceGate**: 确定预测应跳过共振（避免场噪声）
- **EarlyStop**: logits收敛时停止迭代（防止过度共振）
- **QualityFilter**: 弱神经元会稀释强神经元

### 8.3 分工路径的优势

对比等权共识（consensus）策略：
- **code领域**: scale_layering PPL比consensus好2.6×
- **组合策略**: 集群主导 × 内部规模分层

---

## 9. 实验结论

### 实验 12: 门控机制
- 1+1>2不是默认行为，共振只在不确定时才有帮助
- ConfidenceGate: 确定预测应跳过共振（避免场噪声）
- EarlyStop: logits收敛时停止迭代

### 实验 9: 质量过滤
- 弱神经元稀释强神经元
- QualityFilter: 仅PPL < 100的神经元参与共振

### Phase 3: 分工路径
- 当专家神经元匹配领域时，规模分层优于等权共识
- 组合策略: 集群主导 × 内部规模分层

---

## 10. 技术债务与后续工作

> 更新于 2026-08（修复审计 R 系列后）；完整状态见 `plans/BIO_INSPIRED_ARCHITECTURE_PLAN.md` 与 `plans/REMEDIATION_PLAN.md`。

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 🔴 P0 | hub正式训练 | hub neuron（495M）smoke链路已通，正式GPU训练待执行；随后正式协作层训练（`--hub-anchor-weight --hub-contrastive-weight`）+ 阶段4跨域评估 |
| 🔴 P0 | 共振机制A/B证据 | W_cond / field_read_layers 已训练闭环（R1/R2），但收益A/B报告尚未落盘（N2规范） |
| 🟡 P1 | 验证硬化 | 关键verify脚本转真实ckpt加载的pytest（slow标记）+ 最小CI |
| 🟡 P1 | 共享嵌入初始化 | 从teacher embedding用SVD初始化512-dim共享嵌入（低优先，现用正交随机） |
| 🟢 P2 | Agent适配 | planner/reflector改用cortex.think() |
| 🟢 P2 | 工程加固 | ensemble.py拆分 / 裸except加日志 / state_dict聚合接口（R14相关） |
| 🟢 P3 | GPU加速 | 支持CUDA推理（loader已有device传播，待实测） |

---

## 文件索引

### 核心模块

| 文件 | 说明 |
|---|---|
| [taiji/resonance/field.py](file:///e:/taiji-neuron/taiji/resonance/field.py) | 共振场核心 |
| [taiji/resonance/neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py) | 共振神经元 |
| [taiji/resonance/ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) | 共振循环编排（forward/forward_train/continuous_forward） |
| [taiji/resonance/continuous.py](file:///e:/taiji-neuron/taiji/resonance/continuous.py) | 连续时间共振动力学（theta-gamma） |
| [taiji/resonance/config.py](file:///e:/taiji-neuron/taiji/resonance/config.py) | 神经元配置 |
| [taiji/resonance/translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) | 分词翻译器 |
| [taiji/resonance/tribal.py](file:///e:/taiji-neuron/taiji/resonance/tribal.py) | 部落压缩 + 共激活追踪 |
| [taiji/resonance/lifecycle.py](file:///e:/taiji-neuron/taiji/resonance/lifecycle.py) | 生命周期（凋亡/成熟/新生） |
| [taiji/resonance/stdp.py](file:///e:/taiji-neuron/taiji/resonance/stdp.py) | STDP 突触可塑性 |
| [taiji/resonance/neuro_modulation.py](file:///e:/taiji-neuron/taiji/resonance/neuro_modulation.py) | 神经调质 + 睡眠固化 |
| [taiji/resonance/phasor.py](file:///e:/taiji-neuron/taiji/resonance/phasor.py) | 相位动力学（Kuramoto） |
| [taiji/resonance/oscillator.py](file:///e:/taiji-neuron/taiji/resonance/oscillator.py) | o 型振荡神经元（可学习节奏） |
| [taiji/resonance/topology.py](file:///e:/taiji-neuron/taiji/resonance/topology.py) | 拓扑构建（生产接线见 loader） |
| [taiji/resonance/field_memory.py](file:///e:/taiji-neuron/taiji/resonance/field_memory.py) | 场记忆库（写门控+锚点检索） |
| [taiji/brain/cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) | 意识中心 |
| [taiji/layers.py](file:///e:/taiji-neuron/taiji/layers.py) | Transformer基础组件 |
| [taiji/loader.py](file:///e:/taiji-neuron/taiji/loader.py) | 模型加载器（assemble_cortex） |
| [taiji/config.py](file:///e:/taiji-neuron/taiji/config.py) | 配置和Token合约 |

### 训练脚本

| 文件 | 说明 |
|---|---|
| [scripts/training/finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py) | 对话神经元 SFT 微调 |
| [scripts/training/finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py) | 跨规格协作层微调 |
| [scripts/training/finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py) | side_channels 微调 |
| [scripts/training/train_cross_domain_collab.py](file:///e:/taiji-neuron/scripts/training/train_cross_domain_collab.py) | 跨域协作层联合训练（含 hub） |
| [scripts/training/train_hub_neuron.py](file:///e:/taiji-neuron/scripts/training/train_hub_neuron.py) | hub 神经元从零训练 |
| [scripts/training/train_round_level_quality.py](file:///e:/taiji-neuron/scripts/training/train_round_level_quality.py) | 回合级质量判定头训练 |

### API入口

| 文件 | 说明 |
|---|---|
| [api/main.py](file:///e:/taiji-neuron/api/main.py) | 主入口 |
| [api/app.py](file:///e:/taiji-neuron/api/app.py) | 应用工厂 |

---

*本文档基于项目源码自动生成，最后更新于 2026-07-17*