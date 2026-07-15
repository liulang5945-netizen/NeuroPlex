# 态极下一代（共振场神经元架构）— 完整规划

> **版本**: v1.1
> **日期**: 2026-07-15
> **状态**: 已评审，待执行

---

## 〇、总体原则

1. **两个完全分立的项目**：当前 `taiji`（一代单体）封存开源 → 新项目（二代神经元）独立开发
2. **"换个脑子"**：保留所有可复用的"身体"（工具、API、前端、数据处理），只替换"大脑"（单体 ModelSelf → 共振场神经元集群）
3. **硬件普适**：CPU 可训练+推理，GPU 加速但不强制
4. **利于开发**：模块化、可测试、渐进式

---

## 一、当前项目资产审计与迁移决策

### 1.1 taiji/ 核心模块

| 模块 | 文件 | 一代作用 | 二代决策 | 理由 |
|------|------|---------|---------|------|
| `layers.py` | RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock | 一代替换器基础组件 | ✅ **直接复用** | 零改动，神经元 Transformer 体完全依赖此文件 |
| `config.py` | ModelConfig | 一代模型配置 | ✅ **复用+扩展** | 保留基础配置，新增 NeuronConfig 引用 |
| `loader.py` | save_model/load_model | 模型持久化 | ✅ **复用** | 神经元也需要 save/load，逻辑通用 |
| `tokenizer_native_v2.py` | 256K 词表 | 一代文本编码 | ✅ **复用** | 作为通用 I/O 协议层 |
| `tokenizer_contract.json` | ID 空间契约 | 全局 token ID 分配 | ✅ **复用** | 不动 |
| `tokenizer.py` | 旧版 tokenizer | 兼容层 | ❌ **舍弃** | 二代只用 native_v2 |
| `architecture.py` | ModelSelf backbone（1B） | 一代大脑 | ❌ **替换** | 被 ResonanceNeuron 替代 |
| `README.md` | 模块说明 | — | 📝 **重写** | — |

### 1.2 taiji/resonance/（关键）

| 模块 | 文件 | 当前状态 | 二代决策 | 待改进 |
|------|------|---------|---------|--------|
| `field.py` | ResonanceField（150 行） | ✅ 可用 | ✅ **复用+增强** | +W_cond 训练 + 部落压缩 Q 值写入 |
| `neuron.py` | ResonanceNeuron（172 行） | ✅ 可用 | ✅ **复用+增强** | +领域专用 tokenizer 接口 + fingerprint 固化 |
| `ensemble.py` | ResonanceEnsemble（319 行） | ✅ 可用 | ⚠️ **重点改造** | +ConfidenceGate + EarlyStop + 触发条件 + 分工路径 |
| `config.py` | NeuronConfig（107 行） | ✅ 可用 | ✅ **复用** | 基本不变 |

### 1.3 taiji/core/（推理核心，大部替换）

| 模块 | 二代决策 | 理由 |
|------|---------|------|
| `app_state.py` | ✅ 复用 | 全局状态管理通用 |
| `hardware.py` | ✅ 复用 | 硬件探测通用 |
| `model_loader.py` | ⚠️ 改造 | 改为加载神经元而非单体模型 |
| `inference.py` | ❌ 替换 | 单体推理 → ResonanceEnsemble |
| `cuda_inference.py` | ❌ 替换 | 同上 |
| `hybrid_engine.py` | ❌ 替换 | 同上 |
| `native_agent.py` | ⚠️ 改造 | Agent 引擎需适配共振推理 |
| `memory_watchdog.py` | ✅ 复用 | 通用 |
| `pii_sanitizer.py` | ✅ 复用 | 通用 |
| `plugin_manager.py` | ✅ 复用 | 通用 |
| `quantization.py` | ❌ 舍弃 | 二代神经元太小，不需要量化 |
| `security.py` | ✅ 复用 | 通用 |
| `taiji_bridge.py` | ❌ 舍弃 | 一代特定 |
| `taiji_builder.py` | ❌ 舍弃 | 一代特定 |
| `taiji_context.py` | ⚠️ 改造 | 上下文管理需适配 |
| `tokenizer_compat.py` | ❌ 舍弃 | 一代兼容层 |
| `websocket_server.py` | ✅ 复用 | 通用 |
| `utils.py` | ✅ 复用 | 通用工具函数 |
| `config.py` | ⚠️ 改造 | 训练配置通用，早停等已有但需调整 |
| `api.py` | ❌ 舍弃 | 一代特定 |

### 1.4 taiji/agent/（Agent 系统）

| 模块 | 二代决策 | 理由 |
|------|---------|------|
| `perception.py` | ✅ 复用 | 工作区编码通用 |
| `memory.py` | ✅ 复用 | 情景/语义记忆通用 |
| `planner.py` | ⚠️ 改造 | 规划逻辑不变，但推理调用改为共振 |
| `reflector.py` | ⚠️ 改造 | 反思触发共振而非单体 forward |
| `semantic_memory.py` | ✅ 复用 | 通用 |
| `working_memory.py` | ✅ 复用 | 通用 |
| `context_manager.py` | ✅ 复用 | 通用 |

### 1.5 taiji/life/（生命系统）

| 模块 | 二代决策 | 理由 |
|------|---------|------|
| `life_scheduler.py` | ⚠️ 改造 | 调度逻辑不变，训练目标从单体→神经元 |
| `feed_engine.py` | ⚠️ 改造 | 数据投喂→触发神经元训练 |
| `sleep_engine.py` | ⚠️ 改造 | 睡眠→神经元内部参数整合+抱合生长 |
| `explore_engine.py` | ⚠️ 改造 | 探索→激活未使用神经元/加神经元 |
| `play_engine.py` | ⚠️ 改造 | 玩耍→神经元自由共振 |
| `evolution_engine.py` | ⚠️ 改造 | 进化→神经元培育+淘汰+重组 |
| `recursive_improver.py` | ⚠️ 改造 | 策略改进适配共振场 |
| `science_engine.py` | ❌ 舍弃 | 一代实验性模块 |
| `life_interface.py` | ✅ 复用 | 接口定义通用 |

### 1.6 taiji/brain/ & taiji/body/

| 模块 | 二代决策 | 理由 |
|------|---------|------|
| `brain/cortex.py` | ⚠️ 重写 | 意识中心从单体状态→共振场状态 |
| `body/core.py` | ✅ 复用 | BodyCore 通用 |
| `body/senses.py` | ✅ 复用 | 感官模块通用 |
| `body/limbs.py` | ✅ 复用 | 肢体模块通用 |
| `body/metabolism.py` | ✅ 复用 | 代谢模块通用 |

### 1.7 taiji/tools/（工具系统）

| 模块 | 二代决策 | 理由 |
|------|---------|------|
| **全部** | ✅ **全部复用** | 搜索、网页、文件解析、代码执行、RAG、MCP 桥接——与大脑无关，全部保留 |

### 1.8 应用层

| 模块 | 二代决策 | 理由 |
|------|---------|------|
| `api/` 全部路由 | ⚠️ **保留+适配** | FastAPI 全部保留，推理路由改为调用 ResonanceEnsemble |
| `frontend/` | ✅ **全部复用** | Vue 3 UI 与大 ba 无关 |
| `desktop/` | ✅ **全部复用** | PyQt6 桌面端无关 |
| `scripts/` 数据处理 | ✅ **复用** | 数据下载、清洗、tokenizer 训练脚本全部保留 |
| `scripts/` 训练 | ⚠️ **改造** | 单体训练脚本替换为神经元训练脚本 |
| `tests/` | 📝 **重写** | 测试针对新架构重写 |

### 1.9 taiji_portable/

| 内容 | 二代决策 |
|------|---------|
| `taiji/resonance/tribal.py` | ✅ 迁移到二代 resonance/ |
| `taiji/resonance/field.py`, `neuron.py` | ❌ 已被主仓库版本取代 |
| 领域 tokenizer（sp_zh/en/code/math） | ✅ 迁移到二代 |
| 训练脚本 | 📝 参考设计，重写 |

---

## 二、新项目目录结构

```
taiji-neuron/                         # 新项目根目录
├── taiji/
│   ├── __init__.py
│   ├── layers.py                     # ← 复用（TransformerBlock 等，零改动）
│   ├── config.py                     # ← 复用+扩展
│   ├── loader.py                     # ← 复用（模型持久化）
│   ├── tokenizer_native_v2.py        # ← 复用（256K 通用词表）
│   ├── tokenizer_contract.json       # ← 复用（ID 契约）
│   │
│   ├── resonance/                    # ★ 核心：共振场引擎
│   │   ├── __init__.py               # 公共接口导出
│   │   ├── field.py                  # ResonanceField（复用+增强）
│   │   ├── neuron.py                 # ResonanceNeuron（复用+增强）
│   │   ├── ensemble.py              # ResonanceEnsemble（重点改造）
│   │   ├── config.py                 # NeuronConfig（复用）
│   │   ├── gating.py                 # ★ 新增：ConfidenceGate + EarlyStopResonance
│   │   ├── quality.py                # ★ 新增：QualityFilter + 自适应阈值
│   │   ├── tribal.py                 # ← 迁移（部落压缩指标）
│   │   ├── translator.py            # ★ 新增：TokenTranslator + TokenizerHub
│   │   └── division.py              # ★ 新增：分工路径（规模分层+集群主导）
│   │
│   ├── domains/                      # ★ 新增：领域专用 tokenizer
│   │   ├── __init__.py
│   │   ├── zh/  (sp_zh.model/vocab)
│   │   ├── en/  (sp_en.model/vocab)
│   │   ├── code/ (sp_code.model/vocab)
│   │   └── math/ (sp_math.model/vocab)
│   │
│   ├── training/                     # ★ 新增：神经元训练管线
│   │   ├── __init__.py
│   │   ├── scheduler.py              # TrainingScheduler（训练 vs 进化统一决策）
│   │   ├── distill.py                # 蒸馏管线（1.5B → 神经元）
│   │   ├── joint.py                  # 联合训练循环
│   │   ├── contrastive.py            # field_write 对比学习
│   │   └── single.py                 # 单神经元训练
│   │
│   ├── tools/                        # ← 全部复用
│   │   ├── search/                   # 搜索引擎
│   │   ├── web.py, browser.py        # 网页工具
│   │   ├── file_parser.py            # 文件解析
│   │   ├── builtin_tools.py          # 内置工具
│   │   ├── rag.py                    # RAG 检索
│   │   └── ...
│   │
│   ├── agent/                        # ← 复用+适配
│   │   ├── perception.py             # 感知（不变）
│   │   ├── memory.py                 # 记忆（不变）
│   │   ├── planner.py                # 规划（推理调用改为共振）
│   │   ├── reflector.py              # 反思（推理调用改为共振）
│   │   └── ...
│   │
│   ├── life/                         # ← 复用+适配
│   │   ├── life_scheduler.py         # 调度器（训练目标→神经元）
│   │   ├── sleep_engine.py           # 睡眠→神经元整合+抱合
│   │   ├── evolution_engine.py       # 进化→培育+淘汰+重组
│   │   └── ...
│   │
│   ├── brain/
│   │   └── cortex.py                 # ⚠️ 重写（场状态作为意识）
│   │
│   ├── body/                         # ← 全部复用
│   │   ├── core.py, senses.py, limbs.py, metabolism.py
│   │
│   ├── core/                         # ← 精简复用
│   │   ├── app_state.py              # 全局状态
│   │   ├── hardware.py               # 硬件探测
│   │   ├── security.py               # 安全
│   │   └── utils.py                  # 工具函数
│   │
│   └── safety/                       # ← 全部复用
│       ├── constitutional_ai.py
│       └── ...
│
├── api/                              # ← 全部复用+适配
│   ├── app.py                        # FastAPI lifespan（加载神经元替代单体）
│   ├── routes_chat.py                # 对话路由→共振推理
│   ├── routes_agent.py               # Agent 路由→共振推理
│   └── ...
│
├── frontend/                         # ← 全部复用
├── desktop/                          # ← 全部复用
│
├── scripts/
│   ├── data_prep/                    # ← 全部复用
│   ├── training/                     # ★ 新增：神经元训练入口
│   │   ├── train_neuron.py           # 单神经元训练
│   │   ├── distill_neurons.py        # 蒸馏脚本
│   │   ├── train_field_write.py      # 场写入训练
│   │   └── evaluate_resonance.py     # 共振评估
│   └── utils/                        # ← 复用
│
├── tests/
│   ├── test_resonance_field.py       # 场核心测试
│   ├── test_resonance_neuron.py      # 神经元测试
│   ├── test_resonance_ensemble.py    # 共振循环测试
│   ├── test_gating.py                # 门控机制测试
│   ├── test_quality_filter.py        # 质量过滤测试
│   ├── test_division.py              # 分工路径测试
│   ├── test_distillation.py          # 蒸馏测试
│   └── test_1plus1gt2.py             # 1+1>2 验证测试
│
├── docs/
│   ├── RESONANCE_ARCHITECTURE.md      # 新架构文档
│   ├── TRAINING_GUIDE.md             # 训练指南
│   └── ...
│
├── configs/                          # 训练/推理配置
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 三、三个关键机制的设计（实验 12 产物）

### 3.1 在架构中的位置

这三个机制全部位于 [`taiji/resonance/gating.py`](taiji/resonance/gating.py)（新文件），在 `ResonanceEnsemble.forward()` 中调用：

```
输入 shared_embeddings
    │
    ▼
┌─────────────────────────────┐
│ 1. ConfidenceGate           │  ← 新增：判断是否启动共振
│    should_resonate(logits)  │
│    若 max_prob > 0.9        │
│    → skip_resonance=True    │
│    → 直接返回单神经元输出    │
└─────────────┬───────────────┘
              │ 需要共振
              ▼
┌─────────────────────────────┐
│ 2. 共振循环 (现有逻辑)       │
│    for round in 1..max:     │
│      神经元前向 → 写入场     │
│      → 计算共振度 → 过滤     │
│                             │
│    ┌────────────────────┐   │
│    │ 3. EarlyStop        │   │  ← 新增：每轮后检查收敛
│    │    should_stop(      │   │
│    │      logits_history) │   │
│    │    diff < threshold  │   │
│    │    → break           │   │
│    └────────────────────┘   │
└─────────────────────────────┘
              │
              ▼
         输出 weighted_logits
```

### 3.2 ConfidenceGate 接口设计

```python
class ConfidenceGate:
    """置信度门控：低置信度才启动共振，避免对确定预测的过思考。

    位置：在 ResonanceEnsemble.forward() 第一轮前调用。
    输入：单个神经元（或默认神经元）的 logits
    输出：bool（是否需要共振）
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold  # top-1 概率阈值

    def should_resonate(self, logits: torch.Tensor) -> bool:
        """检查预测是否足够确定。

        Args:
            logits: [B, L, vocab] 或 [L, vocab]

        Returns:
            True = 需要共振（不确定）
            False = 跳过共振（已足够确定）
        """
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values.mean()  # 平均 top-1 概率
        return float(max_prob) < self.threshold
```

### 3.3 EarlyStopResonance 接口设计

```python
class EarlyStopResonance:
    """早停共振：logits 收敛时就停，避免过度迭代。

    位置：在 ResonanceEnsemble.forward() 每轮循环后调用。
    输入：logits 历史列表
    输出：bool（是否应该停止）
    """

    def __init__(self, threshold: float = 1e-3, min_rounds: int = 2):
        self.threshold = threshold  # 连续两轮 logits 差的阈值
        self.min_rounds = min_rounds  # 最少共振轮数

    def should_stop(self, logits_history: list[torch.Tensor]) -> bool:
        """检查 logits 是否已收敛。

        Args:
            logits_history: 最近几轮的 weighted_logits 列表

        Returns:
            True = 已收敛，可以停止
        """
        if len(logits_history) < self.min_rounds:
            return False
        # 比较最近两轮的差异
        diff = torch.norm(
            logits_history[-1] - logits_history[-2]
        ) / torch.norm(logits_history[-1] + 1e-8)
        return float(diff) < self.threshold
```

### 3.4 ResonanceEnsemble 改造后的 forward 流程

```python
def forward(self, shared_embeddings, return_logits=False, active_filter=True):
    self.field.reset()

    # ── Step 0: 先跑一轮独立前向，检查是否需要共振 ──
    round1_logits = {}
    for nid, neuron in self.neurons.items():
        result = neuron.forward(shared_embeddings, return_logits=True)
        round1_logits[nid] = result["logits"]
        # 写入场
        self.field.write(nid, result["field_vector"])

    # ── 置信度门控：如果最相关的神经元已经很确定，跳过共振 ──
    if self.confidence_gate is not None and not active_filter:
        best_nid = max(self.field.scores, key=self.field.scores.get)
        if not self.confidence_gate.should_resonate(round1_logits[best_nid]):
            return {
                "field_state": self.field.get_state(),
                "weighted_logits": round1_logits[best_nid],
                "n_rounds": 1,
                "skipped_resonance": True,
            }

    # ── 后续共振轮（现有逻辑 + 早停）──
    logits_history = [self._average_logits(round1_logits)]
    
    for round_num in range(2, self.max_rounds + 1):
        # ... 现有共振逻辑 ...

        # 早停检查
        if self.early_stop is not None:
            current = self._average_logits(all_logits)
            logits_history.append(current)
            if self.early_stop.should_stop(logits_history):
                break

    # ... 返回结果 ...
```

---

## 四、蒸馏路线方案

### 4.1 目标

从一代已训练的 1.5B checkpoint（31B tokens 训练）蒸馏出 5 个高质量神经元（zh/en/code/math/general），每个神经元 PPL < 50，质量均衡。

### 4.2 为什么是蒸馏而非从零训

- 一代 1.5B checkpoint 已经跑了 ~480K optimizer steps，31B tokens
- 紧凑型从零训在 20-35K 序列上 PPL 天花板为两位数到三位数（实验 6-9 证实）
- 蒸馏让神经元直接继承 1.5B 的语义理解能力，只需学会"用自己的方式表达"

### 4.3 蒸馏流程

```
Step 1: 加载 1.5B checkpoint
    ↓
Step 2: 提取各领域的"教师方向"
    - 用 1.5B 在各领域数据上跑前向
    - 收集隐藏态作为教师信号
    ↓
Step 3: 初始化 5 个紧凑型/标准型神经元
    - 神经元 Transformer 体随机初始化
    - 共享嵌入从 1.5B 嵌入 SVD 初始化（LUCKY v4 已验证有效）
    ↓
Step 4: 蒸馏训练（每神经元独立）
    - LM loss（学习语言建模）
    - 蒸馏 loss（field_write 对齐教师方向）
    - 对比 loss（field_write 领域分化）
    ↓
Step 5: 质量闸门
    - 同域 PPL < 50
    - 跨域 Gap > 100
    - 指纹 |cos| < 0.7（各神经元方向不重叠）
    ↓
Step 6: 5 个神经元接入共振场，联合微调
```

### 4.4 蒸馏 loss 设计

```python
def distillation_loss(student_output, teacher_hidden, domain_label, other_field_vectors):
    """三项 loss 组合。

    Args:
        student_output: 学生神经元的 forward 结果
        teacher_hidden: 1.5B 在相同输入上的隐藏态
        domain_label: 领域标签（zh/en/code/math/general）
        other_field_vectors: 其他神经元的 field_vector
    """
    # 1. 语言建模 loss（学语言本身）
    lm_loss = F.cross_entropy(student_output["logits"], targets)

    # 2. 蒸馏 loss（对齐教师方向）
    student_direction = student_output["hidden_before_write"]
    distill_loss = F.mse_loss(student_direction, teacher_hidden)

    # 3. 对比学习 loss（field_write 领域分化）
    contrastive_loss = compute_contrastive_loss(
        student_output["field_vector"],
        domain_label,
        other_field_vectors,
    )

    return lm_loss + 0.3 * distill_loss + 0.1 * contrastive_loss
```

---

## 五、分工路径实验设计

### 5.1 策略 A：规模分层

**假设**：不同规模的神经元担任不同角色，专家型决策、标准型执行、紧凑型辅助，比"所有神经元平等加权平均"更好。

**实验设置**：
```
配置：
  紧凑型（1个）+ 标准型（1个）+ 专家型（1个）
  同一跨领域任务

对比 3 种输出方式：
  A. 加权平均（所有神经元等权）
  B. 规模分层（专家型权重×3，标准型×2，紧凑型×1）
  C. 单独专家型输出

评估指标：PPL、生成质量人工评估
```

### 5.2 策略 B：集群主导

**假设**：问题来了，让最契合的集群主导，其他集群辅助，比所有集群等权更好。

**实验设置**：
```
配置：
  集群 A（3 个中文神经元）
  集群 B（2 个代码神经元）

任务：
  T1: 中文写作任务
  T2: 代码生成任务
  T3: 跨领域任务（用中文解释代码）

对比：
  A. 所有 5 个神经元加权平均
  B. 集群契合度加权（契合度高的集群权重更大）
  C. 集群主导（最契合集群权重 0.7，其余 0.3）
```

### 5.3 组合实验（最终目标）

```
第一层：找到最契合的集群（策略 B）
  ↓
第二层：集群内部按规模分工（策略 A）
  ├── 专家型：决定分工 + 把关质量
  ├── 标准型：执行主要任务
  └── 紧凑型：执行辅助任务
  ↓
第三层：集群间协同（其他集群辅助）
```

---

## 六、实施阶段

### Phase 0：项目初始化（当前 taiji 封存）

- [ ] 当前 `taiji` 仓库做最终整理（README 更新、CHANGELOG、开源协议确认）
- [ ] 创建新仓库 `taiji-neuron`
- [ ] 从当前仓库复制可复用资产（见第一章迁移清单）
- [ ] 搭建新项目基础结构（pyproject.toml、requirements.txt）
- [ ] 确认所有复用文件在新项目中可独立运行（导入路径调整）

### Phase 1：共振场核心增强

按优先级排序：

1. **gating.py** — 实现 `ConfidenceGate` + `EarlyStopResonance`（最高优先级，实验 12 直接产物）
2. **ensemble.py 改造** — 在 `ResonanceEnsemble.forward()` 中集成门控机制
3. **translator.py** — 实现 `TokenTranslator` + `TokenizerHub`（领域专用 tokenizer 热插拔）
4. **quality.py** — 实现 `QualityFilter` + 自适应阈值（实验 9 已验证有效）
5. **field.py 增强** — `W_cond` 训练、部落压缩 Q 值写入
6. **neuron.py 增强** — 领域专用 tokenizer 接口、fingerprint 固化流程

### Phase 2：蒸馏管线

- [ ] 加载一代 1.5B checkpoint 的适配脚本
- [ ] 教师方向提取脚本
- [ ] 单神经元蒸馏训练脚本
- [ ] 蒸馏后质量评估（PPL、领域 Gap、指纹分散度）
- [ ] 5 个神经元联合微调

### Phase 3：分工路径实验

- [ ] 实验 A：规模分层 vs 加权平均
- [ ] 实验 B：集群主导 vs 等权
- [ ] 实验 C：组合（集群×规模分层）
- [ ] 1+1>2 跨领域验证

### Phase 4：上层适配

- [ ] `api/` 路由改为调用 `ResonanceEnsemble`
- [ ] `life/` 生命系统适配（睡眠→神经元整合、饥饿→加神经元）
- [ ] `agent/` Agent 系统适配
- [ ] `brain/cortex.py` 重写（场状态作为意识中心）
- [ ] `frontend/` 确认兼容

### Phase 5：测试与文档

- [ ] 完整测试套件
- [ ] 新架构文档
- [ ] 训练指南

---

## 七、关键设计决策

| 决策 | 结论 | 依据 |
|------|------|------|
| 是否保留 256K 通用词表 | 保留作为 I/O 协议 | 一代已验证，作为领域 tokenizer 的通信桥 |
| 领域 tokenizer 大小 | 32K-48K | 实验证明了领域专用 tokenizer 的必要性 |
| 共享嵌入维度 | 512（不变） | 人脑方案：512 是感官分辨率，认知在第二层 |
| 场维度 | 4096（不变） | 当前够用，预留软扩张接口 |
| 蒸馏 vs 从零训 | 优先蒸馏 | 一代 1.5B checkpoint 是宝贵资产，从零训质量不足 |
| 分工路径 vs 共识路径 | 分工路径优先 | 共识路径（加权平均）已验证有"弱者稀释"问题 |
| 三机制实现位置 | 集中在 `gating.py` | 高内聚、低耦合，ensemble.forward 中清晰调用 |

---

## 八、已确认决策

| # | 问题 | 决策 |
|---|------|------|
| 1 | 新项目名称 | **`taiji-neuron`** |
| 2 | 1.5B checkpoint 使用权 | **允许**，可用于蒸馏 |
| 3 | 领域 tokenizer 来源 | **均可**，从 256K 提取或独立训练都行 |
| 4 | 前端改动 | **不需要**，Vue 3 UI 保持不变 |

---

## 九、Phase 0 执行清单：文件复制

> 以下是从当前 `taiji` 项目复制到新 `taiji-neuron/` 项目的完整文件清单。
> 目标目录：`e:/taiji-neuron/`

### 9.1 直接复制文件清单

**核心基础层（零改动）：**
```
taiji/layers.py              → taiji-neuron/taiji/layers.py
taiji/loader.py              → taiji-neuron/taiji/loader.py
taiji/tokenizer_native_v2.py → taiji-neuron/taiji/tokenizer_native_v2.py
taiji/tokenizer_contract.json→ taiji-neuron/taiji/tokenizer_contract.json
taiji/config.py              → taiji-neuron/taiji/config.py
```

**共振场核心（复用+后续增强）：**
```
taiji/resonance/__init__.py  → taiji-neuron/taiji/resonance/__init__.py
taiji/resonance/field.py     → taiji-neuron/taiji/resonance/field.py
taiji/resonance/neuron.py    → taiji-neuron/taiji/resonance/neuron.py
taiji/resonance/ensemble.py  → taiji-neuron/taiji/resonance/ensemble.py
taiji/resonance/config.py    → taiji-neuron/taiji/resonance/config.py
```

**部落指标（从 taiji_portable 迁移）：**
```
taiji_portable/taiji/resonance/tribal.py → taiji-neuron/taiji/resonance/tribal.py
```

**领域 tokenizer（从 taiji_portable 迁移）：**
```
taiji_portable/domain_tokenizers/sp_zh.model   → taiji-neuron/taiji/domains/zh/sp_zh.model
taiji_portable/domain_tokenizers/sp_zh.vocab   → taiji-neuron/taiji/domains/zh/sp_zh.vocab
taiji_portable/domain_tokenizers/sp_en.model   → taiji-neuron/taiji/domains/en/sp_en.model
taiji_portable/domain_tokenizers/sp_en.vocab   → taiji-neuron/taiji/domains/en/sp_en.vocab
taiji_portable/domain_tokenizers/sp_code.model → taiji-neuron/taiji/domains/code/sp_code.model
taiji_portable/domain_tokenizers/sp_code.vocab → taiji-neuron/taiji/domains/code/sp_code.vocab
taiji_portable/domain_tokenizers/sp_math.model → taiji-neuron/taiji/domains/math/sp_math.model
taiji_portable/domain_tokenizers/sp_math.vocab → taiji-neuron/taiji/domains/math/sp_math.vocab
```

**工具系统（全部复用）：**
```
taiji/tools/__init__.py       → taiji-neuron/taiji/tools/
taiji/tools/*.py               → taiji-neuron/taiji/tools/
taiji/tools/search/            → taiji-neuron/taiji/tools/search/  (整个目录)
```

**Agent 系统（复用+后续适配）：**
```
taiji/agent/                    → taiji-neuron/taiji/agent/  (整个目录)
```

**生命系统（复用+后续适配）：**
```
taiji/life/__init__.py          → taiji-neuron/taiji/life/
taiji/life/life_scheduler.py    → taiji-neuron/taiji/life/
taiji/life/feed_engine.py       → taiji-neuron/taiji/life/
taiji/life/sleep_engine.py      → taiji-neuron/taiji/life/
taiji/life/explore_engine.py    → taiji-neuron/taiji/life/
taiji/life/play_engine.py       → taiji-neuron/taiji/life/
taiji/life/evolution_engine.py  → taiji-neuron/taiji/life/
taiji/life/recursive_improver.py→ taiji-neuron/taiji/life/
taiji/life/life_interface.py    → taiji-neuron/taiji/life/
```

**身体系统、安全系统（全部复用）：**
```
taiji/body/                      → taiji-neuron/taiji/body/  (整个目录)
taiji/safety/                    → taiji-neuron/taiji/safety/  (整个目录)
```

**精简的 core（只保留通用模块）：**
```
taiji/core/__init__.py           → taiji-neuron/taiji/core/
taiji/core/app_state.py          → taiji-neuron/taiji/core/
taiji/core/hardware.py           → taiji-neuron/taiji/core/
taiji/core/security.py           → taiji-neuron/taiji/core/
taiji/core/utils.py              → taiji-neuron/taiji/core/
taiji/core/pii_sanitizer.py      → taiji-neuron/taiji/core/
taiji/core/plugin_manager.py     → taiji-neuron/taiji/core/
taiji/core/websocket_server.py   → taiji-neuron/taiji/core/
```

**应用层（全部复用+后续适配）：**
```
api/                              → taiji-neuron/api/  (整个目录)
frontend/                         → taiji-neuron/frontend/  (整个目录)
desktop/                          → taiji-neuron/desktop/  (整个目录)
```

**脚本（数据处理+工具）：**
```
scripts/data_prep/                → taiji-neuron/scripts/data_prep/  (整个目录)
scripts/utils/                    → taiji-neuron/scripts/utils/  (整个目录)
taiji_portable/scripts/build_domain_tokenizers.py → taiji-neuron/scripts/training/
taiji_portable/scripts/architecture_verification.py → taiji-neuron/tests/
```

**项目配置文件（后续调整）：**
```
.gitignore                        → taiji-neuron/.gitignore
requirements.txt                  → taiji-neuron/requirements.txt (后续调整)
pyproject.toml                    → taiji-neuron/pyproject.toml (后续调整)
```

### 9.2 不复制清单（一代独有）

```
taiji/architecture.py     — 单体 ModelSelf，被 ResonanceNeuron 替代
taiji/core/inference.py   — 单体推理引擎，被 ResonanceEnsemble 替代
taiji/core/cuda_inference.py — CUDA 推理，被 ResonanceEnsemble 替代
taiji/core/hybrid_engine.py  — 混合引擎，被 ResonanceEnsemble 替代
taiji/core/native_agent.py   — 旧 Agent 引擎
taiji/core/model_loader.py   — 单体模型加载器
taiji/core/quantization.py   — 量化（神经元太小不需要）
taiji/core/taiji_bridge.py   — 一代桥接层
taiji/core/taiji_builder.py  — 一代构建器
taiji/core/taiji_context.py  — 一代上下文
taiji/core/config.py         — 一代训练配置
taiji/core/api.py            — 一代 API 通用层
taiji/core/tokenizer_compat.py — 旧 tokenizer 兼容
taiji/life/science_engine.py — 实验性模块
taiji/tokenizer.py           — 旧 tokenizer 兼容层
taiji/agent_ext/             — 旧的 Agent 扩展
taiji/multimodal/             — 多模态（后置）
taiji/plugins/                — 旧插件系统
taiji/services/               — 旧服务层
taiji/infra/                  — 旧基础设施
taiji/data/                   — 一代数据层
taiji/train/                  — 一代训练系统
taiji/mcp_servers_config.json — MCP 配置（一代特定）
taiji/play_data/              — 一代玩耍数据
taiji/test_user/              — 一代测试用户
taiji/training_data/          — 一代训练数据索引
scripts/training/             — 一代训练脚本
```
