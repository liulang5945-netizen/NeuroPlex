# 态极神经元网络 vs 人脑神经元网络：机制详细对比

> **日期**: 2026-08-08
> **状态**: 梳理文档（基于当前代码 + C16-C21 演进事实，非设计文档）
> **关联**: [BIO_INSPIRED_ARCHITECTURE_PLAN.md](file:///e:/taiji-neuron/plans/BIO_INSPIRED_ARCHITECTURE_PLAN.md) / [ARCHITECTURE_COMPROMISE_AUDIT.md](file:///e:/taiji-neuron/plans/ARCHITECTURE_COMPROMISE_AUDIT.md)

---

## Part 1：态极项目机理与架构流程梳理

### 1.1 核心理念

态极（二代）的核心命题：**用多个领域专用的小神经元替代一个大模型**，通过共振场实现神经元间的知识协作，并获得单体模型不具备的能力——**词库可插拔、神经元可拓展、硬件可降级（CPU 可训可推）**。

与一代（1.5B 单体）的关键区别：

| 维度 | 一代 (v1) | 二代 (态极神经元) |
|---|---|---|
| 大脑 | 1.5B 单体 ModelSelf | 5+ 个领域神经元 (24M-118M) |
| 推理 | 单体 forward | 任务级路由 + 共振场协作 |
| 训练 | 端到端预训练 | 蒸馏 + 联合 + 回合级监督 |
| 扩展 | 重新训练 | 热插拔新神经元 + 神经发生 |
| 词表 | 固定 256K | 多独立词表集合（容量不限，可插拔） |

### 1.2 分层架构

```
Level 0: 词库（TokenizerHub）——多独立词表的可扩展集合
           general 256K / zh 50K / code / en / math / 多模态 encoder
           ↑ 每个 neuron 绑定自己的词表；跨词表靠"词库转译"协作
Level 1: 领域神经元（ResonanceNeuron）——独立 Transformer + 域 lm_head
           + quality_head（回合级质量判定）+ LoRA + side_channels
           ↓ field_write / field_read（树突式场接口）
Level 2: 共振场（ResonanceField 4096-dim）——共享"意识"，独立于词表
           + 侧抑制 WTA / 伽马门控 / STDP / 神经调质 / 空间扩散
```

### 1.3 推理数据流（C21 词库多词表架构正式化后）

```
用户回合文本
  │
  ├─ ① 回合级任务判定 _executive_route（任务模式，非 token 级竞争）
  │    信号1: _infer_domain 启发式（code>math>zh>en>general，快）
  │    信号2: quality_head 回合级聚合（learned：各 neuron round1 质量 logit）
  │    融合: per-neuron EMA z-score + 成熟度门(count<20 回退启发式)
  │          + z 绝对差门(≥0.7σ 防全能型错误覆盖)
  │    → 主导任务模式 leader（如 zh / code / math）
  │
  ├─ ② leader 激活（+ general 辅助），回合内稳定生成，不做 token 级切换
  │    生成空间 = leader 词表空间（词库转译）：
  │      general 256K 空间 → identity 回填
  │      zh 50K 空间      → zh decode + domain→general 回填（v3 口径）
  │    leader logits 用 round1 独立 logits（无场条件化——协作只用于判定，
  │    不污染 leader 的域词表能力；C21 修复 round2 场污染导致的中英混合）
  │
  └─ ③ 协作/共振（仅用于判定与弱任务补强，不直接掺入 leader 生成）
        Round 1: 所有 neuron 独立前向 → 写入场（field_write，L2 归一）
        ConfidenceGate → Round 2-N 条件化共振（读场 → 前向 → 重写）
        QualityFilter / EarlyStop / 伽马门控 / 调质×scores
```

### 1.4 训练数据流

```
离线训练（脚本闭环）：
  base 预训练（train_compact_simple）→ dialogue fine-tune（finetune_neuron_dialogue，
  v3 口径：general 输入表 + zh 头 + zh decode + domain→general 回填）
  → cross_spec 协作层（finetune_cross_spec，side_channels + 跨规格投影）
  → C20 回合级质量监督（train_round_level_quality：只训 quality_head，
     body/LoRA/side 冻结；同域 batch——batch 内同域回合 NLL 才可比；
     answer_mask 只对回复部分算回合级 NLL，prompt 无区分度）

在线学习（生命周期闭环）：
  Feed（喂养样本累积）→ Sleep（睡眠巩固：记忆整合 + 模型训练 + 神经调质更新
  + 诊断：healthy / data_insufficient / capacity_limited）
  → 瓶颈 → Neurogenesis（新生：IntegrateEngine 4 阶段）→ 固化 / Apoptosis
```

### 1.5 生命周期闭环（自演化）

| 引擎 | 机制 | 人脑对应 |
|---|---|---|
| FeedEngine | 喂入文本/多模态，累积 pending samples | 感觉输入 |
| SleepEngine | 睡眠期：记忆重放式整合 + 模型微训 + 调质更新 + 域诊断 | 睡眠巩固 + 突触稳态 |
| ExploreEngine | 搜索/读网页搜集新知识 | 主动探索/好奇 |
| PlayEngine | 玩耍式交互，记录 Coactivation | 发育期游戏/共激活 |
| EvolutionEngine | 错误率斜率判别 + select_spec + diagnose_domain | 环境适应/进化 |
| IntegrateEngine | 新 neuron 整合：静默期→蒸馏期→验证期→固化/凋亡 | 海马齿状回神经发生 |
| ScienceEngine | 科学式实验验证 | 假设检验 |

**IntegrateEngine 4 阶段**（maturity_ratio 驱动）：

| 阶段 | maturity | 融合权重 | 可训练 | 机制 |
|---|---|---|---|---|
| ① 静默期 | 0–0.3 | 0.1 起步 | side+head+LoRA | 只连输入侧，fusion 权重近 0 |
| ② 蒸馏期 | 0.3–0.8 | ramp 0.3→0.8 | 高 lr (3×) | 邻居蒸馏 KL 对齐 |
| ③ 验证期 | 0.8–1.0 | 0.8→1.0 | 冻结 | ablation 贡献评估 |
| ④ 固化/凋亡 | 1.0 | 1.0 | — | 正贡献 commit / 负贡献 apoptosis |

### 1.6 关键演进脉络（C16 → C21）

```
C16  LoRA 保护 body + quality_head + 对比学习（个体能力零破坏原则确立）
C16d per-neuron EMA z-score + 绝对质量 gate（quality 信号归一化）
C19  范式转变：token 级竞争 → 回合级任务路由（ExecutiveControl）
      全局 256K 空间收敛 + identity 回填（C16b 前身）
C20  回合级监督训练：answer_mask + 同域 batch + z 绝对差门（验证 5/5）
C21  词库多词表架构正式化：decode 按 leader 词表空间、LoRA 按 lm_head 过滤、
      leader 用 round1 无场 logits（dialogue 流畅中文恢复）
      → 用户核心需求落地：词库=多独立词表集合（可插拔），neuron 绑定自己词表
C22  路径收敛（2026-08-08）：executive 设为默认主路径（此前 API 仍走 C19 前
      旧 fusion token 级路径）；executive 跳过 hybrid 共振校验消除双路径打架；
      残留实验路径（fusion/leader/routing_mode/fusion_mode 多分支）仅显式实验用
      → 用户"反复推翻留下多条路径"梳理落点
      → 用户确认：振荡相位同步为设计本意，当前"场向量累加+相位门控"为实现偏移
```

---

## Part 2：态极神经元网络 vs 人脑神经元网络机制对比

### 2.1 总体范式对比

| 维度 | 态极神经元网络 | 人脑神经网络 |
|---|---|---|
| 基本单元 | ResonanceNeuron（独立小 Transformer） | 生物神经元（~860 亿个，皮层） |
| 单元差异化 | 领域专家分工（zh/en/code/math/general） | 功能柱/脑区分工（梭状回、布罗卡区、视皮层） |
| 信息载体 | logits（token 概率分布）+ field_vector（4096-dim 场状态） | 动作电位（脉冲）+ 突触权重 + 局部场电位 |
| 协作方式 | 共振场写入/读取 + side_channels + 词库转译 | 突触连接 + 振荡同步 + 重入回路 |
| 竞争机制 | 任务级路由（回合级 WTA）+ 场侧抑制 | 皮层柱内 WTA（侧抑制），跨脑区是信息传递 |
| 记忆 | WorkingMemory + DialogueState + SleepConsolidator | 工作记忆（持续放电）+ 海马-皮层系统巩固 |
| 学习 | 蒸馏/联合/回合级监督/睡眠微训 | 突触可塑性（LTP/LTD/STDP）+ 睡眠重放 |
| 扩展 | 热插拔新 neuron + Neurogenesis + 词库可插拔 | 神经发生（海马）+ 突触修剪/生长 |

**范式定位差异**：人脑是"同质单元的异质连接+异质发放"；态极是"异质单元的工程协作+场通信"。态极把生物脑的**连接可塑性**（突触成长/修剪）在工程上替换为**单元/词表可插拔**，这是"神经元即模块"的刻意简化——换来可操作性，失去的是生物脑连接层面的精粒度自组织。

### 2.2 单神经元层面（ResonanceNeuron vs 生物神经元）

| 态极机制 | 人脑对应 | 对应关系说明 |
|---|---|---|
| `field_write`（单头/多头注意力池化 → 场向量） | 树突整合 + 胞体发放 | 输入处理后把状态"写出去"；多头=多树突分支汇聚 |
| `field_read_layers` + `field_read_gate`（位置门控读取） | 树突接收突触前输入 + 门控调制 | 每层读场 = 每层都有输入侧调制通道 |
| `side_channels`（per-pair excite/inhibit 权重） | 神经元间突触（兴奋/抑制性） | 显式的双向突触权重，学习得到 |
| `enter_refractory` / `tick_refractory` | 绝对/相对不应期 | 发放后冷却，防止过度激活 |
| `quality_head`（回合级质量 logit） | 神经元对任务的相关性编码 | 近似"该神经元是否适合当前任务的内部读出" |
| `lm_head`（域词表头 / shared+delta 低秩） | 神经元的输出词汇/动作偏好 | 域空间输出；delta 低秩=词表层轻量适应 |
| `LoRA`（body 冻结 + 低秩增量） | 突触微调（细粒度可塑性）而不重排网络 | 保护既有能力，只改小增量 |
| STDP tracker（时序可塑性） | 时序依赖可塑性（Hebbian） | 脉冲时间差 → 权重变化 |
| `write_gain` / `refractory_multiplier` | 增益调制（intrinsic excitability） | 神经元发放增益的动态调节 |
| `quick_probe`（轻量前向） | 快速通路的粗加工（dorsal stream） | 跳过完整计算做预筛选 |
| `freeze_fingerprint`（方向指纹） | 神经元的方向选择性（tuning） | 用指纹向量做预筛选路由 |

**关键差异**：生物神经元是**脉冲事件 + 时间维度编码**（rate/temporal coding），态极 neuron 是**稠密向量 + 层间连续前向**。态极丢失了时间维度的计算（精确脉冲时刻携带的信息），换来了可微性和工程可行性。refractory/STDP 在态极是"机制占位"（详见 2.10 借鉴边界）。

### 2.3 群体/场层面（ResonanceField vs 皮层网络）

| 态极机制 | 人脑对应 | 对应关系说明 |
|---|---|---|
| `ResonanceField`（4096-dim 共享状态） | 皮层局部场电位（LFP）/ 共享神经池 | 群体级状态的共享介质 |
| 多轮共振（Round 2-N 读场→前向→重写） | 皮层重入回路（reentry）/ 迭代精化 | 前馈+反馈循环直到收敛 |
| `write_inhibit` + `apply_inhibitory_wta` + `lateral_inhibition_norm` | 侧抑制 + winner-take-all（皮层柱内） | 同类内部竞争选优 |
| `W_cond`（乘法门控条件化） | 增益调制（gain modulation） | 场状态作为乘法门控影响神经处理 |
| `scores`（余弦共振分） | 群体编码的相似度匹配 | 场状态与神经元方向的对齐度 |
| `CrossSpecProjector`（field_dim→unified 投影） | 跨脑区纤维束传递（胼胝体/上纵束） | 不同规格 neuron 的场维度对齐 |
| `SparseRouter`（round1 后选 top-K） | 稀疏群体编码（sparse coding） | 每轮只激活少数最相关单元 |
| GammaOscillator（伽马门控） | 伽马振荡（30-100Hz 同步绑结 binding） | 用振荡相位门控信息传递 |
| SpatialDiffusion（空间扩散） | 神经递质空间扩散 / 容积传递 | 邻近区域的慢速弥散调制 |
| CoactivationTracker（共激活矩阵） | 细胞集群（cell assembly）形成 | Hebbian 共激活 → 聚群 |
| Tribal（部落压缩：Q=αβγ） | 皮层微柱/功能柱的汇聚输出 | 子群体压缩为上级单位向量 |
| NeuromodulatorState | 神经调质（多巴胺/5-HT/去甲肾上腺素/乙酰胆碱） | 全局行为状态调制 |

**关键差异（⚠️ 设计本意 vs 实现偏移，2026-08-08 用户确认）**：振荡相位同步是态极的**设计本意**——共振的本体应是相位同步驱动（谁同相谁绑结成知觉单元，feature binding 本义）。但实现过程偏移为**共振场静态向量累加为主、gamma 相位仅作门控调制**：`GammaOscillator` 已完整实现（Kuramoto 相位耦合 L[gamma_oscillator.py](file:///e:/taiji-neuron/taiji/resonance/gamma_oscillator.py#L98-L144) + 写入门控 `apply_gamma_gate` + scores 门控 [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L2002-L2008)），loader 装配时也真实注入（[loader.py](file:///e:/taiji-neuron/taiji/loader.py#L447-L463)），但相位只做**幅度调制**（gate_factor∈[0.2,1.0]），没有让"相位关系"本身成为共振的信息传递载体——主次颠倒：场向量累加是主调，相位是配角。共振"轮次"对应生物的多轮重入，但生物是连续时间动力学，态极是离散轮次。**相位同步本体化 = 当前态极 vs 人脑对照中最深的差距（缺口 R 核心）。**

### 2.4 路由与执行控制（_executive_route vs 前额叶执行控制）

**这是态极最"人脑化"的设计，也是 C19 范式转变的核心。**

| 态极机制 | 人脑对应 | 对应关系说明 |
|---|---|---|
| 回合级任务判定（不逐字切换） | 前额叶执行控制（dlPFC 的 task set） | 任务模式决定后整条通路激活到任务结束 |
| `_infer_domain` 启发式判定 | 背侧通路的快速自动加工 | 快、可解释、无需学习 |
| quality_head 回合级聚合（learned） | 前额叶对任务相关性的习得评估 | 慢、可学习、跨 neuron 归一化 |
| 混合信号 + EMA z-score + 成熟度门 | 前额叶-皮层下回路的多信号整合 | 融合快慢双通道 |
| z 绝对差门（≥0.7σ） | 显著门控（避免噪声切换） | 防止"全能型"错误覆盖 |
| leader 回合内稳定生成 | 任务模式保持（task maintenance） | 风格/一致性优先 |
| 同域竞争（同域 batch 监督） | 局部竞争只在同功能内部可比 | 跨域不可比 → 不同域不竞争 |

**C19 的关键结论（与神经科学一致）**：token 级全局 softmax 竞争（C12-C16 范式）类似"每一刻全脑所有神经元竞争解码"，这在生物上不存在——跨脑区不是竞争而是信息传递；竞争只发生在**同功能内部**（同皮层柱、同候选解释）。态极据此把竞争粒度从 token 级收敛到**回合级**，把跨域竞争降级为"判定 → 传递"。

### 2.5 词库/语言（TokenizerHub 多词表 vs 人脑词汇加工）

| 态极机制 | 人脑对应 | 对应关系说明 |
|---|---|---|
| 多独立词表集合（容量不限） | 功能脑区的词汇分工 | 每个脑区有自己的加工单元 |
| 词库转译（domain→general 对齐回填 / v3 口径） | 跨脑区词汇/语义映射 | 韦尼克↔布罗卡的信息转换 |
| neuron 绑定自己词表 | 母语区 vs 二语区的分工 | 可插拔 = 新语言=新 neuron+新词表 |
| 256K general 词表作为公共 I/O | 共享的概念层/共同语义 | 跨域公共协议 |
| alignment table 预计算 | 长时记忆中的映射表 | 免文本往返的信息丢失 |

**用户核心需求与此对应**：词库=多独立词表的可扩展集合（C21 正式化）。人脑中不存在"唯一词汇表"——每个感觉/运动系统有自己的表征空间，跨系统靠转换。态极 C21 反转 C19 的"全 general decode"，正是恢复这种"每 neuron 自带词表 + 词库转译"的生物结构。

### 2.6 记忆与巩固（WorkingMemory / SleepConsolidator vs 海马-皮层系统）

| 态极机制 | 人脑对应 | 对应关系说明 |
|---|---|---|
| WorkingMemory（上下文窗口） | 前额叶工作记忆（持续放电保持） | 在线维护当前任务信息 |
| DialogueState（对话状态） | 情境记忆（episodic context） | 多轮对话的状态维持 |
| SleepEngine._sleep_phase_memory_consolidation | 睡眠期记忆巩固（慢波+锐波涟漪） | 离线重放式整合 |
| SleepEngine._sleep_phase_model_training | 睡眠期突触强化/弱化 | 离线学习（NREM 重放） |
| SleepEngine._update_neuromodulators | 睡眠-觉醒调质循环 | 状态循环调节学习 |
| FeedEngine 累积样本 → 睡眠消费 | 日常经验 → 睡眠巩固 | 采集→离线加工 |
| MaturityTracker（0.1→1.0 ramp） | 发育关键期 | 幼稚高可塑 → 成熟低可塑 |

**关键差异**：人脑睡眠巩固的核心机制是**海马重放**（锐波涟漪把海马记忆回放给皮层）和**突触稳态下调**（downscaling：睡眠期整体按比例缩小突触强度，突出强信号）。态极 sleep 是"拿累积样本离线训练"，重放/下调是方向性借鉴，未实现生物意义上的"逐条回放 + 全局缩放"。

### 2.7 发育与可塑性（Lifecycle vs 神经发生/修剪）

| 态极机制 | 人脑对应 | 对应关系说明 |
|---|---|---|
| NeurogenesisTrigger + select_spec | 海马齿状回神经发生 | 按错误率/域诊断触发新生 |
| IntegrateEngine 静默期（只输入不输出） | 沉默突触（silent synapse） | 树突先成熟、轴突后建立 |
| IntegrateEngine 蒸馏期（高 lr 3×） | 关键期高可塑性 | LTP 阈值低、可塑性远超成熟 |
| IntegrateEngine 验证期（ablation） | 突触竞争/功能验证 | use-it-or-lose-it |
| IntegrateEngine 固化/凋亡 | 存活/凋亡选择 | 正贡献保留、负贡献修剪 |
| ApoptosisTracker（record_ppl） | 细胞凋亡/突触修剪 | 低质神经元退出 |
| MaturityTracker（maturity ramp） | 关键期关闭（可塑性下降） | 成熟后固化 |
| 邻居蒸馏 KL（向拓扑最近邻学习） | 引导性突触形成（依赖现有回路） | 新生 unit 参考成熟邻居 |

**这是态极最完整的生物映射段**：IntegrateEngine 的四阶段几乎逐条对应海马神经发生的已知机制。且工程上验证过（verify_integrate.py / verify_neurogenesis.py / verify_apoptosis.py）。

### 2.8 多模态与感知

| 态极机制 | 人脑对应 |
|---|---|
| TokenizerHub.register_modality（EnCodec/VQVAE/Video encoder） | 各感觉通道的初级皮层 |
| neuron.auto_register_modalities + encode_multimodal_input | 多感觉整合（跨通道绑定） |
| MultimodalOutputEngine | 运动输出/效应器 |

人脑多模态的核心是**跨模态绑定**（伽马同步实现"视听同一物体"）和**预测编码**（每层预测下层输入、反馈误差）。态极目前是多模态 encoder 接入词库转译，属于"通道接入"而非"跨模态绑结"——伽马绑结仍是 Optional。

### 2.9 涌现与协作（EMERGE 验证）

| 指标 | 数值 | 人脑对应 |
|---|---|---|
| 协作 PPL 24.0 vs 最强个体 34.5 | 提升 30.5% | 群体编码优于单神经元 |
| 融合权重分布（zh_aug0:0.312 > ... > aug3:0.122） | 非等权 | 群体加权决策 |
| 多轮上下文维持（跨轮引用） | 有效 | 工作记忆+情境维持 |
| 回合级判定 5/5（C20） | 通过 | 任务模式识别 |

人脑"1+1>2"来自**连接冗余 + 群体编码 + 振荡同步**；态极"1+1>2"来自**side_channels 协作层 + 场通信 + 任务级分工**。两者都验证了"多样性单元 + 协作机制"能超越单一大单元，但机制底座完全不同。

### 2.10 逐项机制对照总表

| # | 机制 | 态极实现 | 人脑对应 | 忠实度 |
|---|---|---|---|---|
| 1 | 单元分工 | 领域 neuron | 脑区功能分工 | ★★★★ 高 |
| 2 | 输入整合 | field_write 池化 | 树突整合 | ★★★☆ 中 |
| 3 | 突触连接 | side_channels | 突触权重 | ★★★☆ 中 |
| 4 | 不应期 | refractory_counter | 动作电位不应期 | ★★★☆ 中（语义占位） |
| 5 | 时序可塑性 | STDP tracker | STDP | ★★☆☆ 低（未驱动权重） |
| 6 | 振荡绑结 | GammaOscillator（Kuramoto + 门控） | 伽马振荡 | ★★★☆ 中（已接入但为门控角色；本意=相位驱动共振本体） |
| 7 | 侧抑制 | field WTA / inhibitory mask | 皮层柱侧抑制 | ★★★☆ 中 |
| 8 | 任务路由 | _executive_route | 前额叶 task set | ★★★★ 高 |
| 9 | 竞争粒度 | 回合级 | 任务模式级 | ★★★★ 高 |
| 10 | 词汇分工 | 多词表 + 转译 | 脑区词汇分工 | ★★★★ 高 |
| 11 | 记忆巩固 | SleepConsolidator | 海马重放+下调 | ★★☆☆ 低 |
| 12 | 工作记忆 | WorkingMemory | 持续放电 | ★★★☆ 中 |
| 13 | 神经发生 | IntegrateEngine 4 阶段 | 齿状回神经发生 | ★★★★★ 极高 |
| 14 | 凋亡/修剪 | ApoptosisTracker | 凋亡/修剪 | ★★★★ 高 |
| 15 | 发育关键期 | MaturityTracker | 关键期可塑性 | ★★★★ 高 |
| 16 | 调质状态 | NeuromodulatorState | 神经调质 | ★★★☆ 中 |
| 17 | 群体编码 | 共振场 | LFP/群体编码 | ★★★☆ 中 |
| 18 | 稀疏激活 | SparseRouter | 稀疏编码 | ★★★★ 高 |

### 2.11 借鉴边界（哪些是生物启发，哪些是工程简化）

**生物启发且已生效**：
- 领域分工 → 任务级路由（C19 范式转变，验证 5/5）
- 词库多词表（C21，用户核心需求）
- 神经发生四阶段（IntegrateEngine）
- 关键期/凋亡（MaturityTracker/ApoptosisTracker）
- 侧抑制 WTA（field 层）

**生物启发但当前是"装饰"或"角色偏移"（审查 S9 缺口 R）**：
- STDP（只追踪不驱动权重）
- Gamma 振荡（⚠️ 已接入训练/推理主路径，但角色偏移：作门控调制器而非共振本体——设计本意为相位同步驱动共振）
- Sleep 重放（近似为离线训练，非逐条回放）
- 神经调质（状态记录，未深度耦合训练）

**刻意工程简化（与生物不同的设计选择）**：
- 无时间维度脉冲编码（稠密向量替代）
- 连接可塑性替换为单元可插拔（"神经元即模块"）
- 离散共振轮次替代连续动力学
- 回合级路由替代连续任务切换（多阶段任务留 v2）

### 2.12 局限与下一步（基于当前事实）

**已验证的能力**：
- 回合级任务判定 5/5（C20）
- dialogue 流畅中文问答（C21 多词表修复后）
- EMERGE：协作 30.5% 优于最强个体
- 神经发生/整合/凋亡链路（verify 脚本通过）

**当前短板**：
- 4 个 general neuron 生成能力弱（zh 回显/math/en 碎片/code 简短）——foundation 600 步训练不足，非架构问题；修复路径：域目标空间训练（同 dialogue 修复路径）
- 睡眠"重放"语义未实现生物粒度
- 多阶段任务（zh 理解→code 生成→zh 表达）留作 v2

**架构层面的最高上限方向**（若延续"上限更高优先"原则）：
1. **振荡相位同步本体化**（用户设计本意恢复，缺口 R 核心）：让相位关系成为共振的信息传递载体——场写入/读取按相位相干调制（同相增强/异相解绑），Kuramoto 相位耦合驱动的激活选择替代静态 top-k，伽马相位作为轮次间信息传递的通道。这是态极 vs 人脑对照中最深的差距
2. 把 STDP/调质从 Optional 装饰推进为 forward_train 骨架（缺口 R）
3. 睡眠重放：实现真正的 forward 重放 + 经验回放训练（缺口 R 子项）
4. 多阶段任务模式链（task-set 序列）——人脑"任务集切换"的完整版

---

*本对比基于当前代码实现（field/neuron/ensemble/translator/cortex/life/lifecycle）+ C16-C21 演进记录，忠于事实。*
