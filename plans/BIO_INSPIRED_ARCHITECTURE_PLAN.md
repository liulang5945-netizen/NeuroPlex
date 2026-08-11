# 态极生物学化架构计划 (Bio-Inspired Architecture Plan)

> 本文档记录态极架构借鉴人脑神经科学的系统性改造计划。
> 核心原则：**神经元差异性第一**、**自我进化能力**、**硬件限制不在考虑范围内**。

> **📋 项目主 plan（活跃维护）**
> 本文档是项目的状态总览、路线图与接口梳理（按内容拆分后的核心导航，2026-08-10）。其他 plans/ 文件：
> - `HISTORY_DIALOGUE_TRAINING.md` — 对话/Standard 神经元训练历史（训练流水线详情）
> - `HISTORY_MECHANISM_EXPERIMENTS.md` — 机制实验与里程碑（EMERGE、aux-free balancing、shared expert）
> - `HISTORY_PROJECT_EVENTS.md` — 项目事件与旧状态归档（整理记录、紧急 bug 修复、旧总览）
> - `DESIGN_PRINCIPLES.md` — 设计原则与 Phase 1-8 历史记录
> - `TRAINING_REFERENCE.md` — 训练准则参考（非 plan）
> - `TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` — 项目机理梳理 + 态极 vs 人脑机制详细对比（2026-08-08）
> - `archive/` — 历史归档（COMPREHENSIVE v1.0、H1-H8 修复、side-channels 实现、架构妥协审计、body/life/brain 设计参考、hub neuron 草案）

---

## 🗺️ 一、项目全景（2026-08-01 更新）

> 本章节是文档导航入口，按主题组织项目现状。训练流水线详情见 `HISTORY_DIALOGUE_TRAINING.md`，历史实验与事件见 `HISTORY_MECHANISM_EXPERIMENTS.md` / `HISTORY_PROJECT_EVENTS.md`。

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
| cross_spec 协作层（对话版） | ✅ 完成 | `cross_spec_dialogue.pt`，train PPL 54.2→43.4（3 epochs） |
| 综合体对话评估 | ✅ 完成（EMERGE） | **协作 PPL=24.0 vs 最强个体 34.5，提升 30.5%**；多轮上下文维持有效；生成质量有限（语法错误较多，需更多数据） |
| 运行时（Cortex/API） | ✅ 断裂 A-D 已修复 | 混合规格装配通过、协作权重加载、embedding 加载、域路由前缀修正 |
| 进化机制（在线） | ✅ 全接线 | 斜率判别器 + `select_spec` + `diagnose_domain` 均已接入 sleep_engine/cortex |
| 硬编码治理 | ✅ P0-P3 完成 | `experiment_config.py` + `utils.py` 集中管理 |

### 1.4 闭环缺口清单（未闭环设计）

| # | 类型 | 位置 | 说明 |
|---|------|------|------|
| A | ✅ 已修复 | [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py#L66-L88) | 混合规格装配：删 hidden_size 校验，field_dim 取 max，ensemble 自动建跨规格投影层 |
| B | ✅ 已修复 | [loader.py](file:///e:/taiji-neuron/taiji/loader.py) `_load_collab_weights_into_cortex` | 运行时加载 `cross_spec_dialogue.pt` 的 side_channels + 跨规格投影层（ID 不匹配时警告） |
| C | ✅ 已修复 | [loader.py](file:///e:/taiji-neuron/taiji/loader.py#L270-L323) | shared_embedding 用 base_embed_dim(512) + 优先加载 `data/shared_embedding.pt` 训练权重 |
| D | ✅ 已修复 | [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) `_infer_domain`/路由 | key 前缀提取纯域（zh_aug0_dialogue→zh）+ Level 1 路由激活同域全部神经元 |
| D+ | ✅ 已修复 | [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) `generate` | 预存 bug：`fusion_mode` 未传给 `_generate_p7` 导致 NameError |
| E | ✅ 已修复 | [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py#L538-L553) `add_neuron` | SpecSelector 已接入：`lifecycle.neurogenesis.select_spec(domain)` 按错误率选 compact/standard/expert；split 模式继承父规格 |
| F | ✅ 已修复 | [sleep_engine.py](file:///e:/taiji-neuron/taiji/life/sleep_engine.py#L646-L649) | `diagnose_domain` 已接入：record_domain_error 后记录域诊断状态（healthy/data_insufficient/capacity_limited） |
| G | ✅ 已修复 | [loader.py](file:///e:/taiji-neuron/taiji/loader.py#L457-L474) Step 9.2 + [play_engine.py](file:///e:/taiji-neuron/taiji/life/play_engine.py#L237-L243) | Play→CoactivationTracker 链路修复：使用 cortex.coaction 实例（非新建）+ 调用 update(ids)（非不存在的 record_coactivation） |
| H | ✅ 已修复 | [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py) `eval_multi_turn_conversation` | 多轮对话评测：3 场景 × 3 轮追问，维护对话历史测试上下文连贯性（`--multi_turn` 启用） |
| I | 缺失 | `api/` | 综合体未接入聊天接口，无发布/导出脚本 |
| J | ✅ 已清理 | ~~cognitive_enhancements.py~~ | 已删除：CorticalColumn/ColumnRegistry/AttentionBeam/ThresholdPlasticity 全库零调用，从 __init__.py 移除导入 |
| K | ✅ 已修复 | translator.py TokenTranslator | 删除死代码类 + build_translator + translators 字段；同时修复 `_get_token_spans` 空格对齐 bug（独立 `▁` token 零长度 span 导致缩进丢失，代码文本对齐率 38%→100%） |
| L | ⚠️ 架构缺失 | ensemble.py 跨域语义对齐 | 当前 EMERGE 靠同 vocab logits 融合（非 field 对齐）；跨域（zh/code）field_vector 语义不对齐，加入 code neuron 会互相噪声污染。详见 [HUB_NEURON_DESIGN.md](file:///e:/taiji-neuron/plans/HUB_NEURON_DESIGN.md) |
| M | ⚠️ 代码缺口 | ensemble.py `forward_train` | `torch.stack(all_logits)` 要求同形状，跨 vocab 联合训练直接崩溃；跨域联合训练路径未实现 |
| N | ✅ 已修复 | ensemble.py `forward_train` 共振从未训练 | forward_train 重写为全可微多轮共振：round 2+ 注入 side_signals+field_state，调质×scores，Gamma 门控，diversity_loss。6/6 验证通过。详见审查报告 S1 |
| O | ✅ 已修复 | utils.py:111-117 256K emb 配 16K tokenizer | general tokenizer 已是 256K（中文 0 unk）；修复 build_domain_tokenizers.py 路径不一致 + 补充 general 域。详见审查报告 S2 |
| P | ✅ 已修复 | 全训练脚本 Loss 单一化 | SFT answer masking 已接入 3 个对话训练脚本（finetune_neuron_dialogue/cross_spec/side_channels）；balance_loss + diversity_loss 已在 S1 修复中接入 forward_train。5/5 验证通过。详见审查报告 S3 |
| Q | ✅ 已修复 | cortex.py:1350-1358 域 token re-encode 往返 | 对齐表预计算 domain→general 映射，消除 text 往返信息丢失。KV cache 仍未启用（后续独立优化）。详见审查报告 S6 |
| R | ⚠️ 系统性妥协 | 生物学机制是推理期占位 | STDP/调质/Gamma/睡眠/新生均 Optional 注入，forward_train 不引用，是"装饰"非"骨架"。详见审查报告 S9 |

---

## 🧭 二、路线图

### 2.1 当前执行中
- ✅ cross_spec 对话协作层训练完成（train PPL 54.2→43.4）
- ✅ eval_dialogue.py 评估完成：**EMERGE 现象确认**（协作 PPL=24.0 vs 最强个体 34.5，提升 30.5%）

### 2.2 EMERGE 评估结果（2026-08-01）

| 指标 | 值 |
|------|-----|
| 最强个体 PPL | 34.5（zh_std0_dialogue） |
| 协作 PPL | **24.0** |
| 提升幅度 | **30.5%** |
| 融合权重 | zh_aug0:0.312 > zh_aug2:0.243 > zh_std0:0.198 > zh_aug1:0.124 ≈ zh_aug3:0.122 |

**多轮对话**：能维持上下文（场景2轮2引用轮1的"机器学习/深度学习"概念；场景3轮2引用轮1的"Python"）。
**生成质量**：语法错误较多，有无关内容——训练数据仅 4000 步，需扩充。

### 2.3 下一步建议
1. **修复隐性天花板**（缺口 O + 训练步数）：训 256K general tokenizer + 步数提到 12000-16000。不解决这两个，后续所有优化都被掩盖。详见 [ARCHITECTURE_COMPROMISE_AUDIT.md](file:///e:/taiji-neuron/plans/ARCHITECTURE_COMPROMISE_AUDIT.md)
2. **让共振可训练**（缺口 N）：可微多轮共振，让 forward_train 接入场+侧通道+调质

### 2.4 中期：跨域协作（hub neuron，上限优先版）
- **hub neuron 设计与实现**（缺口 L）：参考人脑联合皮层，引入 hub neuron 作为跨域语义对齐枢纽。**上限优先设计**：expert 规格 + 256K lm_head + hub-and-spoke 拓扑 + 跨域对比 loss。详见 [HUB_NEURON_DESIGN.md](file:///e:/taiji-neuron/plans/HUB_NEURON_DESIGN.md)
- 修复 `forward_train` 跨 vocab 崩溃（缺口 M）
- 多任务 loss（缺口 P）：SFT masking + margin ranking + diversity + 对比 loss
- 推理路径优化（缺口 Q）：域 token 对齐表 + 长上下文 + 多轮对话状态

### 2.5 远期：生物学机制深化（缺口 R）
- STDP 影响注意力/FFN 权重（非仅 side_channels）
- 多频段振荡（theta-gamma 嵌套）+ 跨频耦合
- 真正睡眠重放（forward 重放 + 经验回放训练）
- 自组织新生（从经验生长，非 teacher 蒸馏）
- 多 standard 神经元协作验证
- 更长训练（8000→16000 步）

### 2.6 ✅ 神经发生无缝衔接设计（IntegrateEngine，2026-08-08）

**背景与动机**：真正的神经元新生应是"态极自主演化"的一部分——喂入数据 / 自我搜集学习达到瓶颈后触发新生，而非手动粗暴加入。当前 `cortex.add_neuron` 后仅做权重继承 + 投影补建 + `maturity.register_new`，**无整合训练**：side_channels 需调用方重建、quality_head 随机、加入即全权参与融合。

**人脑参照**（海马体齿状回神经发生）：
1. **沉默突触（silent synapse）**：新生神经元只接收输入、不参与输出（树突先成熟，轴突后建立）
2. **关键期可塑性（critical period）**：新生神经元 LTP 阈值低、可塑性远超成熟神经元，依赖现有回路引导整合
3. **use-it-or-lose-it**：只有成功整合进回路的存活，未整合的凋亡
4. **关键期关闭**：成熟后可塑性下降、突触稳定固化

**现有基础（可复用）**：
- `MaturityTracker`（taiji/resonance/lifecycle.py#L366）：`get_resonance_weight`（幼稚 0.1 → 成熟 1.0 线性 ramp）、`get_lr_multiplier`（幼稚 3× → 成熟 1×）、`tick_all`、`is_mature`
- `apoptosis.record_ppl`：凋零记录；`gen_test_collab.py`：subset 对照 = ablation 工具
- C16 LoRA（body 冻结，个体能力零破坏）+ C15 quality_head/contrastive（路由监督）
- `FeedEngine.get_pending_samples_by_domain()`：喂养数据流（睡眠时累积样本）
- 挂载点：`SleepEngine._train_cortex_neurons` 中 neurogenesis 创建新 neuron 之后

**设计：`taiji/life/integrate_engine.py`** — 类 `IntegrateEngine`，由 sleep_engine 在 neurogenesis 分支后调用 `integrate(new_nid)`。按 `maturity_ratio` 分 4 阶段：

| 阶段 | maturity | 融合权重 | 可训练参数 | 机制 |
|------|----------|---------|-----------|------|
| ① 静默期 | 0–0.3 | `get_resonance_weight`（0.1 起步） | side_channels + quality_head + LoRA | 只连输入侧，用 feed 样本 forward_train 学习；fusion 权重近 0（对比当前"加入即全权"） |
| ② 可塑+蒸馏期 | 0.3–0.8 | ramp 0.3→0.8 | 同上 + 高 lr（`get_lr_multiplier` 3×） | 加**邻居蒸馏 loss**：KL(新 neuron logits ‖ 拓扑最近邻输出)；LoRA 快速适配 |
| ③ 验证期 | 0.8–1.0 | 0.8→1.0 | 冻结 | **ablation 贡献评估**：有 vs 无该 neuron 的 ensemble 生成/路由指标（复用 gen_test subset 对照） |
| ④ 固化/凋亡 | 1.0 | 1.0 | — | 贡献正 → commit（tick 满，成为导师）；负 → apoptosis 信号 + 移出 ensemble |

**关键技术决策**：
1. **蒸馏源 = 拓扑最近邻**：side_channels 已按 geometry 建立，新 neuron 的 indegree 邻居即"导师"——自然衔接共振场（人脑"被邻居回路引导"）
2. **融合权重**：复用 `get_resonance_weight`（0.1→1.0）。可选增强：`maturity_min_resonance_weight` 降到 0（完全静默起步，更贴"沉默突触"），配置项
3. **训练数据**：全部来自 `FeedEngine` 累积样本 → "喂养训练"闭环数据源，无需手动脚本
4. **ablation 指标**：优先生成质量（gen_test 采样），辅以 PPL/EMERGE
5. **C16 保护延续**：新 neuron body 冻结，只训 LoRA/side/head → 整合不破坏个体能力

**喂养闭环（态极自主训练方式）**：

```
探索/玩耍/喂数据（FeedEngine.feed_*）
    ↓ 累积 pending 样本
睡眠（SleepEngine.run）
    ├─ 训练既有 neuron（_train_cortex_neurons，P7 模式）
    ├─ neurogenesis 触发 → add_neuron → IntegrateEngine.integrate（静默→蒸馏→验证→固化）
    └─ apoptosis 筛选（未整合 / 冗余 neuron 凋零）
    ↓ maturity.tick_all
醒来应用新能力 → 循环
```

这是态极自主演化的完整闭环：**喂（数据）→ 睡（整合+新生+凋零）→ 醒（应用）**，无需手动训练脚本。IntegrateEngine 是最后一块拼图（当前缺口：新生 neuron 无整合、粗暴加入）。

### 2.7 🔄 任务级路由设计（Executive Control Routing，C19，2026-08-08）

**背景与动机**：C12-C16 四次路由迭代（LOO cosine / 域判别 head / quality_head NLL / gate+z-score）全部失败的共同根因 = **"统一空间 + 全局 token 级竞争"范式本身与生物机制相悖**：
- NLL/cosine/logit 跨 neuron 天然不可比（native general vs 转译、英文 vs 中文匹配度、分布锐度）→ 每次修复（温度/gate/z-score）都是在不可比信号上打补丁
- token 级 softmax 竞争导致回复频繁切换 neuron、风格与一致性断裂（C16 生成空洞乱码的直接表现）

**人脑参照**（执行控制 + 结构分工）：
1. **解剖结构分工**：面孔→梭状回（FFA）、语言→布罗卡/韦尼克区。分工是硬连线（发育期突触竞争+修剪固化），不是推理时动态学出来的
2. **任务级执行切换**：前额叶执行控制网络（dorsolateral PFC）决定"当前任务模式"（task set），模式确定后整条通路激活直到任务结束——不在句子内逐字切换脑区
3. **局部竞争**：winner-take-all（侧抑制）只发生在同功能内部（同皮层柱、同输入候选解释之间），天然可比；跨脑区不是竞争，是信息传递
4. **层级流水线**：跨脑区协作 = 前馈预测 + 反馈误差（预测编码），串行/层级，非同一时刻 logits 融合

**范式转变**：
```
token 级（C12-C16，失败）：每 token 位置 softmax 竞争选 winner，统一 256K 空间投影
  ↓
任务级（C19）：回合（用户消息）级判定任务模式 → 主导 neuron 回合内稳定生成，
             不做 token 级竞争；quality_head 升级为回合级信号
```

**现有基础（可复用，无需重造）**：
- `cortex._infer_domain(text)`：启发式回合级域判定（code>math>zh>en>general，关键词+结构），✅ 已实现
- `cortex.generate(domain=...)` / `_generate_p7`：`domain` 参数直接指定回合级主导域，✅ 已实现
- hybrid 共振校验（routing_mode）：domain 判定后 probe forward 校验/切换，✅ 已实现
- `_fingerprint_route` / `_auto_topk_route`：prototype cosine 回合级 top-k，✅ 已实现
- quality_head（C16 产物）：MLP 结构保留，监督目标升级为**回合级**
- LoRA 保护 body（C16 原则）：body 冻结 + 低秩增量，零崩坏已验证，不变
- C18 客户端链路：assemble_cortex + chat/feed/sleep，✅ 已实现

**核心设计**：

**组件 1：ExecutiveController（执行控制器，新）**
```
回合输入（用户消息 + 对话历史）
  ├─ 信号 1: 启发式域判定 _infer_domain（快、可解释、回合级稳定）
  ├─ 信号 2: quality_head 回合级聚合（learned：各 neuron round1 对回合文本
  │          质量 logit → 回合级 [N]，融合置信度）
  ├─ 信号 3: prototype cosine（_fingerprint_route，可选）
  └─ 融合 → 主导任务模式 dominant_domain + 置信度
```

**组件 2：任务模式生成（回合内稳定）**
- 主导 neuron 激活（+ general 辅助），其他 neuron 不参与
- 回合内不做 token 级切换（风格/一致性；人脑任务模式激活直到任务结束）
- 生成空间：保留统一 general 空间（C18 客户端链路不动，dominant 在 general 空间自回归）

**组件 3：回合级监督训练（quality_head 升级，核心）**
- 监督粒度：token 级 NLL 排序 → **回合级真实生成质量**
- 机制：候选 neuron 轮流主导生成完整回复 → 对比回复质量 → 质量最优者获该回合标签 → 训练 quality_head（回合级 softmax 对齐）
- **为什么能避免 C16 的不可比**：比较对象是"完整回复的优劣"（回合粒度、同一评估器、天然可比），而非 token 级 NLL（跨空间不可比）
- 评估器候选：① 全量融合 NLL（dominant 回复为目标的 CE）；② 外部评估器（LLM 评分）；③ 启发式指标（长度/重复率/风格一致性）

**关键决策点（2026-08-08 已确认）**：
- A. ✅ 判定信号：**混合**（启发式 _infer_domain + quality_head 回合级聚合，置信度融合）
- B. ✅ 回合级监督评估器：**融合 NLL 自监督**（候选轮流主导生成 → 全量融合 NLL 评估 → 标签训练 quality_head，免外部依赖）
- C. 多阶段任务（zh 理解→code 生成→zh 表达）留作 v2，v1 先回合级稳定

**实施进度（2026-08-08 冒烟验证通过，verify_c19_executive.py）**：

| 验证项 | 结果 |
|---|---|
| round1 quality_logits 全收集 | ✅ 9/9（修复：final 聚合用 round1 快照，非共振过滤后 active_ids） |
| 回合级判定（warmup 内启发式主导） | ✅ code→code / zh→zh / dialogue→zh / en→en；math→en 为启发式误判（C20 quality 修正场景） |
| quality 混合信号 | ✅ per-neuron EMA z-score + 成熟度门（count<20 回退启发式，C20 训练后自动生效） |
| executive 生成（40 token） | ✅ 无 OUT_OF_RANGE；leader 限定 dominant 域 |
| fusion 生成 | ✅ 无 OUT_OF_RANGE |

**关键架构收敛（本次实施发现的遗留问题）**：
- 2026-08-07 所有 neuron 共享 general lm_head（logits 统一 256K 空间）后，`_generate_p7` 仍用 **domain tokenizer decode**（2026-07-31 per-neuron lm_head 时代的正确做法）→ general 空间 token id 被当 domain vocab 解析 → **OUT_OF_RANGE**。已收敛：生成/decode 全程 general 256K 空间（identity 回填），domain 只负责激活 neuron 选择。
- quality_logit 跨 neuron 不可比（C16b 教训在 quality 域重演：未校准 code head 恒高 16.9 vs 其他 -2~-5）→ `_executive_route` 复用 C16b EMA z-score + warmup 成熟度门，未训练时回退启发式（回退安全）。

**已知限制**：executive 生成质量仍为碎片（C16 基座训练限制，zh 域 neuron 在 general 空间中文能力弱），非 C19 机制问题——C20 回合级训练提升 quality 判定的同时，基座能力提升靠后续训练。

**C20 实施（组件 3：回合级监督训练，2026-08-08 ✅ 验证通过）**：
- **回合级监督粒度**（[ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) `forward_train` 新增 `answer_mask`）：per_neuron_nll 只对 answer（回复）部分计算回合级 NLL——prompt 部分所有 neuron 都能续写（无区分度），answer 才是"谁能生成好这个回复"的回合粒度真实质量信号。C16d 的全序列 NLL 被 prompt 稀释。
- **训练目标聚焦**（train_round_level_quality.py）：只训 quality_head（body/LoRA/side 冻结，C16 保护原则延续）；监督 = C16d 复用（per-neuron EMA z-score + 绝对质量 gate），但作用于回合级 NLL。
- **关键工程决策：同域 batch**——batch 内同域回合（NLL 可比），否则混合域 batch 被低 NLL 域（code）拉低 batch 最优 → dialogue neuron（转译，NLL 基线巨大）被 gate 全排除，监督失效。冒烟实测：混合域 dialogue NLL 2000+（被 gate 排除）；同域 batch 后 dialogue 对中文回合 NLL 15-16（参与监督）。
- **warm start**：quality_head 从 collab_v3_c16.ckpt.pt 加载（保留已学信号）继续回合级训练。正式训练 1100 steps（200 条/域 + 300 对话 × 2 epochs，~2h CPU）。
- **✅ 验证结果**（verify_c20_round_quality.py，C16 vs C20 head 对比 + _executive_route 混合信号）：
  - C16 head：code 恒高 16.91 → 所有文本 best_q=code（未校准，C19 已发现）
  - C20 head：code 不再独占；回合级判定 **5/5**：code→code / math→math / zh→zh / dialogue→zh / en→en（修正 C19 的 math→en 启发式误判）
  - **关键调优**：切换条件加 z 绝对差阈值（≥0.7σ）——纯比例（1.5×）对接近 0 的 z 太宽松（en 回合 zh 0.49 vs en 0.04 也满足 1.5× → 错误覆盖启发式正确的 en）。0.7σ 实测区分：math 1.08 vs en 0.13（差 0.95 正确切）、en 回合 zh 0.49 vs en 0.04（差 0.45 不切）
  - **已知限制**：zh general neuron 是"全能型"（对多数文本 NLL 低）→ quality z 系统性偏高，靠 0.7σ 显著门防错误覆盖；生成质量仍碎片（C16 基座限制，非 C20 范围）
- **产出**：collab_v3_c20.ckpt.pt（head_state 分量，C18 注入格式兼容）；验证脚本 verify_c20_round_quality.py

**C21 实施（词库多词表架构正式化，2026-08-08 ✅ 核心目标达成）**：
- **架构定位（用户核心需求）**：词库 = 多独立词表的可扩展集合（容量不限），neuron 绑定自己的词表，跨词表靠词库转译协作；新 neuron 自带词表可插拔接入。**反转 C19 的"全 general decode"**（把 5 个 dialogue neuron 的 zh 头当"第二空间"是历史残留，但统一到 256K 违背可插拔需求）。
- **关键验证发现**：
  - dialogue neuron 能力**未退化**：general 输入 + zh 头 + zh decode（v3 口径）能生成中文——碎片主因是 C19 用 general 词表解析 zh 空间 id（错位）
  - **C16 LoRA 是 dialogue 负资产**：C16 在 general 目标空间 + 转译投影下训 LoRA，注入后 dialogue zh 生成退化；跳过（loader 按 lm_head 空间判断，256K 头才注入）后恢复流畅中文
  - **round2 场污染**：装配后 dialogue 的 zh 输出被英文 neuron 的混合场污染 → 中英混合；leader 改用 **round1 独立 logits**（无场条件化，协作只用于判定）后生成干净
- **代码落地**：
  - [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) `_generate_p7`：decode 按 leader 词表空间（general 256K → identity 回填；zh 50K → zh decode + domain→general 回填 v3 口径）；leader 用 `round1_logits`（无场）
  - [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)：forward 暴露 `round1_logits`
  - [loader.py](file:///e:/taiji-neuron/taiji/loader.py)：lora_state 按 lm_head 空间过滤（≠256K 跳过）
- **✅ 验证**（verify_c21_generate.py）：回合级判定 5/5；dialogue executive 生成**流畅中文问答**（"，我会尽力给您。这些时间可以帮助您给出一个问题："）；code/math/zh/en 生成弱 = general 基座能力问题（foundation 600 步训练不足，非架构问题）
- **遗留**：4 个 general neuron 的生成能力弱（zh 回显/math/en 碎片/code 简短）——后续用域目标空间训练增强（同 dialogue 修复路径）

**C22 实施（路径收敛 + 设计本意确认，2026-08-08 ✅）**：
- **背景（用户"梳理"的真实诉求）**：C12-C21 反复推翻与修改留下多条并存路径，默认入口指向旧范式 → 后续调用混乱。
- **审计结论**：`generate()` 默认 `collab_mode="fusion"`（token 级，C19 已否定的范式），API（routes_taiji/senses/context_manager/test_api_dialogue）全部走默认 → **线上实际是旧路径**；executive（C19-C21 验证 5/5）需显式传参才启用；`routing_mode`（hybrid/resonance/keyword）+ `fusion_mode`（soft/residual/consensus/per_position/division/division_norm）+ 4 套路由方法（_executive_route/_fingerprint_route/_auto_topk_route/_infer_domain）并存；且 executive 模式下 hybrid 共振校验块仍会执行 → 可能覆盖 executive 判定（双路径打架）。
- **收敛动作**：
  1. `generate()` 默认 `collab_mode` → `"executive"`（C19-C21 验证过的正确范式默认化，旧 fusion/leader 保留为显式实验参数）
  2. `_generate_p7` 中 `collab_mode=="executive"` 时跳过 hybrid 共振校验块（消除双路径打架）
  3. **废弃 `verify_c21_generate.py --no-dialogue-lora`**：loader 已按 lm_head 空间过滤 C16 负资产 lora_state（zh 50K 头不注入），该参数清零的是 dialogue neuron **v3 微调自带的 LoRA**（域能力一部分）——实测清零后 zh/dialogue 生成退化（"我的眼�"乱码），移除参数后恢复流畅中文（"其中，当时间序列…"）。残留参数干扰调用的实例，已清理
- **✅ 验证**：py_compile 通过；verify_c21_generate.py 重跑确认 executive 生成不受影响
- **设计本意确认（用户）**：**振荡相位同步是态极设计本意**——共振本体应为相位同步驱动（谁同相谁绑结，feature binding 本义）；当前"共振场静态向量累加 + gamma 相位作门控调制"是实现偏移（GammaOscillator 已实现 Kuramoto 耦合并真实注入训练/推理，但相位只做幅度调制 ∈[0.2,1.0]，未成为信息传递载体）。→ **相位同步本体化 = 缺口 R 核心方向**（记录于 [TAIJI_VS_HUMAN_BRAIN_COMPARISON.md](file:///e:/taiji-neuron/plans/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md) 2.3/2.10/2.12）
- **残留实验路径（暂不删除）**：fusion/leader collab_mode、routing_mode 三态、fusion_mode 六态——均有验证脚本引用，保留为显式实验入口；后续若新范式稳定可归档。

**C23 实施（相位同步本体化·增量 A，2026-08-08 ✅ 冒烟 6/6）**：
- **背景（设计本意恢复）**：振荡相位同步是态极最初的设计本意——共振本体应为相位同步驱动（谁同相谁绑结成知觉单元）。但实现偏移为"场向量累加 + 相位仅作标量门控"：推理路径 Kuramoto 相位演化无消费端（纯装饰）；训练路径仅对全局相位做幅度门控（gate_factor∈[0.2,1.0]），相位从未成为 neuron 之间的**关系度量**。
- **核心机制**：`GammaOscillator.pairwise_binding` —— binding_i = mean_{j≠i}[cos(θ_i-θ_j)]（可选共激活调制，与 Kuramoto 一致）。同相群体 binding→+1（绑结），异相→-1（解绑）。
- **接入**（推理 forward + 训练 forward_train 一致）：`scores = scores × (1 + binding_scale·binding)`，binding_scale 默认 0.3。相位从"对全局相位的标量门控"升级为"驱动共振强度的关系度量"——**相位同步直接决定谁参与共振、权重多少**。
- **动态闭环**（冒烟验证）：共激活强 → Kuramoto 相位牵引（耦合 k∝coactivation）→ 相位差缩小 → binding 上升（b0=-0.375 → b1=0.900）→ 共振分增强。即"共激活 → 相位同步 → 绑结 → 共振增强"闭环。
- **✅ 冒烟**（_smoke_c23_phase_binding.py）：6/6 PASS（同相 +1 / 异相 -1 / 混合 / Kuramoto 闭环 / 调制效果 / 双路径接入）
- **安全性**：`_executive_route` 只消费 quality_head 聚合（不消费 final_scores）→ binding 不影响 C19-C21 已验证的回合级判定与 leader 生成；影响面 = fusion 融合权重 + 共振演化过滤（实验/协作路径）
- **下一步（增量 B，未开始）**：场写入按相位绑定加权（同相 neuron 写入场上叠加增强），让相位同步直接塑造场状态形成；增量 C：相位可微化（2D 相位向量 + Kuramoto ODE 离散化），让 binding 参与训练梯度流

**C23-B 实施（场写入按相位绑定加权，2026-08-08 ✅ 冒烟 8/8）**：
- **目标**：让相位同步直接塑造场状态形成——"谁与谁同步"决定共享场的结构（增量 A 已让相位决定共振权重，本增量让相位决定场本身）
- **实现**：
  1. ensemble 新增辅助 `_phase_binding_map` / `_phase_binding_scale`（复用 C23 pairwise_binding）
  2. 推理 `forward`：round1 写入 scale ×(1+β·binding)；round2+ 每轮重算 binding（相位随 Kuramoto 演化逐步绑结 → 写入逐轮增强）；抑制性 neuron 同相 → 抑制增强（生物：PV+ 相位锁定 gamma）
  3. 训练 `forward_train`：场构造 `all_vecs_weighted ×(1+β·binding)`（同相群体写入分量增强），让相位绑结成为可学习信号
- **✅ 冒烟 8/8 PASS**：同相写入 scale×1.100 / 异相×0.700；训练场同相分量 norm×1.100 / 异相×0.700
- **语义闭环（C23 全）**：共激活 → Kuramoto 相位牵引 → binding 上升 → 共振分增强（增量 A）+ 场写入增强（增量 B）→ 绑结单元在场上主导、被解绑 neuron 退场——**共振本体（场）+ 共振权重（scores）都由相位同步驱动**
- **下一步（增量 C，未开始）**：相位可微化——2D 相位向量（cosθ,sinθ）+ Kuramoto ODE 离散化，让 binding 参与训练梯度流（当前 binding 是离散标量，不直接可微；可微化后相位绑结成为端到端可学机制）

**C23-C 实施（相位可微化：PhasorDynamics，2026-08-08 ✅ 冒烟 13/13）**：
- **目标**：相位从"启发式调制"升级为"端到端可学机制"——谁同相/异相由任务学出，而非先验同域同相
- **新模块** `taiji/resonance/phasor.py`：`PhasorDynamics(nn.Module)`
  - 相位 = 2D 单位向量 p_i=(cosθ_i,sinθ_i)；binding = p_i·p_j（**可微点积**）；Kuramoto 牵引 = det([p_i,p_j])（**可微叉积**）
  - 可学习：phasors（Parameter，任务梯度直接调相位）+ ω（自然频率）+ coupling_k（耦合强度）
  - **双驱动相位动力学**：前向 Kuramoto 物理牵引（in-place 状态推进）+ 反向任务梯度（`task_gradient_step` 黎曼切向更新）
- **关键工程发现**：相位是单位向量（流形约束），普通 SGD 径向梯度被归一化抹掉（完全对齐/反相是 binding 驻点，sin(Δθ)=0）；正确更新 = 切向投影 `tangent = g−(g·p)·p`（黎曼梯度下降）
- **接入**：ensemble.forward_train 的 scores 段 + 场构造段加 `differentiable` 分支（binding_tensor 替代标量 dict 版）；loader 推理仍用标量 GammaOscillator（互不干扰）
- **✅ 冒烟 13/13 PASS**：binding 可微（梯度流经 phasors）；ω/K 驱动演化改变 binding（Δ=2.87）；演化保持单位范数；任务梯度驱动相位演化（Δ=1.18）+ 同相绑定提升（0.276→0.385）；接口兼容（assign_phase_by_domain/batch_gate_factors）；forward_train 可微分支接入
- **遗留（下一阶段）**：ω/K 梯度路径（当前在 no_grad 演化中，需 evolve 可微化——演化输出直接参与 loss）；PhasorDynamics 接入训练脚本（train_round_level_quality 显式启用）

**C23-C2 实施（ω/K 梯度路径打通 + PhasorDynamics 接入训练，2026-08-08 ✅ 冒烟 15/15）**：
- **ω/K 梯度路径**：新增 `PhasorDynamics.evolve()`（可微 Kuramoto 演化，返回归一化新相位 [N,2]，不 in-place）；`kuramoto_step` 重构为 evolve + detach 状态推进（接口不变）；`binding_tensor` 加 `phasors` 外部相位参数
- **forward_train 打通**：演化段对 PhasorDynamics 保存 `_last_evolved_phasors`（最后一轮可微演化输出）；scores 段 + 场构造段用该可微相位算绑定 → **任务 loss 梯度经 binding → new_p → dtheta → ω/K**（梯度路径打通）；phasors 亦收到梯度（task_gradient_step 切向更新）
- **训练接入**（train_round_level_quality `--enable-phasor` 显式启用，默认关闭不破坏 C20 流程）：
  - PhasorDynamics 创建 + `assign_phase_by_domain`（同域同相先验）→ 传 ensemble `gamma_oscillator`
  - optimizer 含 ω/K（可学）；phasors 排除（用 `task_gradient_step` 黎曼切向更新）
  - 每 step backward 后 `task_gradient_step`；checkpoint 含 `phasor_state` 分量
- **✅ 冒烟 15/15 PASS**：ω 收到梯度 |∇ω|=0.21、K 收到梯度 |∇K|=0.49；任务驱动 ω 演化 Δω=0.07
- **C23 全部完成**：共振权重（A）+ 场本体（B）+ 相位动力学端到端可学（C）——相位同步本体化闭环，从"装饰"变为"骨架"（缺口 R 核心项落地）

**C23-C3 实施（训练验证发现 + phase-binding loss 修复，2026-08-08 ✅ 训练实测 ω/K 学习）**：
- **验证发现（重要）**：带 --enable-phasor 跑 C20 训练（60 条/域 × 116 steps），checkpoint 显示 **ω/K 恒初始值**（0.7854/0.05）——**contrastive_loss 只依赖 quality_logits/NLL，完全不经过 binding 路径** → 相位绑定的梯度在真实训练中断（冒烟梯度通 ≠ 训练生效）。phases 全收敛 0.314 是 Kuramoto 物理牵引（同频+全激活共牵引），非任务信号。
- **修复**：新增 **phase-binding loss**——`binding ∥ normalize(scores_pre)`（调制前共振分），语义"谁共振贡献大谁同相"（绑结与协作贡献对齐）。`forward_train` 计算并返回 `phase_loss`；train 脚本 `--phasor-weight`（默认 1.0）加入 total_loss。
- **✅ 训练实测（30 条/域 × 56 steps）**：ω 分化 0.7854→[0.738,0.731,0.833,0.837]；K 学习 0.05→−0.0014；相位自组织分化（不再 Kuramoto 同化的全 0.314，而为 [−0.39,−0.27,0.76,1.11]）——**任务驱动的相位自组织验证成立**
- **✅ phase_loss 记录进 loss_history**（2026-08-08 验证）：train 脚本 loss_history 增补 `phase_loss` 字段（result.get 容错），验证训练 7/7 条记录含 phase_loss，值 0.99→0.13 收敛；checkpoint 含 `phasor_state` 分量确认
- **C23-C4 修复（完整配方训练暴露监督污染，2026-08-08）**：
  - **现象**：collab_v3_c23_full（200 条/域 + 300 对话，2 epoch，--enable-phasor）训练完成后端到端判定退化——C20 判定 5/5 全对（math→math），C23 full 4/5（**math→zh**）。quality_logits 膨胀（zh_aug2_dialogue ql −4.2→50），head 权重差异 <0.08 却产生 54 级 ql 差。
  - **根因（监督污染链）**：forward_train 场构造段 binding 调制 `all_vecs_weighted×(1+β·bvec)` → field_state → round2 logits → `per_neuron_nll` → contrastive 监督目标（ideal）被相位自组织驱动漂移。phase_loss 的目标是 binding∥共振分（与 NLL 质量语义无关）→ **两个监督信号打架** → quality_head 学乱 → E2 段 contrastive 饱和 18.42（KL 上界，C20 零饱和）。冒烟训练太短未暴露，完整配方暴露。
  - **修复（分工修正）**：训练 forward_train 场构造**不再**按 binding 调制——监督测"谁能预测好"（纯净 NLL，与 C20 一致）；相位只经 scores 段调制共振分 + phase_loss 可微（ω/K 梯度路径保留）；**推理 forward 场写入 binding 本体化（C23-B）保留不动**。
  - **✅ 冒烟 15/15 PASS**（test_6/test_8 更新为新语义：训练场无调制断言 + scores 段可微保留）
  - **✅ 同 seed 复现验证（60 条/域 1 epoch）**：C20 无 phasor 与 C23 有 phasor 的 ql 分布**完全一致**——C23-C4 修复后 phasor 对 quality_head 零干扰。此前 C20(5/5) vs C23_c4(3/5) 判定差异归因于 **seed bug**（`random.seed(42)` 在 shuffle 之后，两次完整训练数据顺序不同，对比不公）→ 已修复 seed 位置。
  - **✅ c23_final_seeded 完整验证（同 seed 200 条/域 + 300 对话 2 epoch + phasor，2026-08-08）**：饱和 0/109（C23 full 曾 54/109）；phase_loss 收敛 0.77→0.105；端到端判定 **5/5 与 C20 基线一致**（code→code/math→math/zh→zh/dialogue→zh/en→en，无回归）；ω 全分化 [0.68~0.93]、K 学习 −0.083、相位自组织分化（角度覆盖 −164°~110°）——**C23 相位同步本体化闭环最终验证成立**

**C23-C5 实施（PhasorDynamics 提升为 loader 默认装配，2026-08-08 ✅）**：
- **目标**：推理与训练相位路径统一——训练用 --enable-phasor 学到的 ω/K/相位自组织，推理直接复用（此前推理用标量 GammaOscillator，相位动力学不一致，训练成果在生成路径未落地）。
- **改动**：
  - train 脚本 checkpoint `phasor_state` 附 `id_order`（训练 _id_to_idx 顺序）
  - loader Step 6 默认装配 **PhasorDynamics**（替代标量）：① 协作层含 phasor_state → 按训练顺序（id_order，旧 ckpt 回退 head_state keys 顺序）**重排 phasors/omega 行到当前装配顺序**（推理 dialogue 在前、general 在后，防错位）→ load_state_dict 注入；② 无 phasor_state → assign_phase_by_domain 域先验（0/π/3/2π/3/π，与旧标量等价）；③ 装配失败 → 回退标量 GammaOscillator（非致命）
  - 推理态冻结：gamma.eval() + requires_grad_(False)（推理 forward 只走 dict binding/gate 标量路径，Kuramoto 状态推进仍生效）
  - cortex.set_gamma_oscillator：已有相位则跳过 assign（防止覆盖 loader 注入的训练相位）
  - PhasorDynamics 补 `phases` property + `get_phase`（apply_gamma_gate/cortex 日志兼容标量接口）
- **✅ 验证**：c23_final_seeded 装配 → 9 neuron 相位/ω/K 与训练一致（code=−55.3°/math=−90.3°/zh_aug0=74.7°/K=−0.0834）；collab_v3_c16（无 phasor_state）→ 域先验 0°/60°/120°/180°；端到端判定 5/5 无回归；冒烟 15/15 PASS
- **✅ 全部完成**：C23 相位同步本体化（A 共振分 / B 场本体 / C 可微化 / C2 ω·K 梯度 / C3 phase-binding loss / C4 训练监督纯净化 / C5 默认装配）——训练-推理统一，闭环落地

**C24 实施（4 个 general neuron 域目标空间 SFT 增强，落地 C21 遗留，2026-08-09 ✅ 完成）**：
- **目标**：C21 遗留——4 个 general neuron 生成能力弱（zh 回显/math/en 碎片/code 简短），根因 = foundation_v1_general 在 general 256K 空间（英文主导）续写训练 + 无 SFT QA 能力。修复路径（同 dialogue 修复，C21 已验证）：**域目标空间**——general 输入 + 域词表目标 + answer masking，让 neuron 在自己的词表空间表达域内容。
- **v1 产物**：data/foundation_v1_sft/（从 foundation_v1 域头基座 SFT，6 epochs → train PPL code 2.1/math 2.1/zh 8.6/en 10.6）——**生成能力验证成立**（code 生成 `def Fib(n):` 结构），但 **端到端判定退化**（collab_v3_c24 1/5：code→en/math→en/zh→math/en→zh）。
- **判定退化根因（2026-08-09 诊断闭环）**：
  1. foundation_v1（C24 域头基座）body 在 general 256K 空间 **NLL 无对角**（code 回合 zh=14.85 反而最低）→ C20 head（general 空间校准）失配；
  2. native NLL 监督（各 neuron 用自己的域词表算回合 NLL）**跨 neuron 不可比**——en 16K 英文专精词表对英文回合（code/math/en 占训练 55%）NLL 恒定低 → en z-score 恒负 → en quality_logit 膨胀常数头（17-38 vs 其他 -1~-7）→ 判定全错。
  3. 对照：foundation_v1_general body + general 256K 头 NLL **完美对角 4/4**（code=1.16/math=3.13/zh=11.10/en=2.50）→ C20 当年判定 5/5 的信号链依赖 general 空间可比性。
- **C24v2 双头架构（上限最高方案）**：neuron 同时保留 **judge_lm_head（general 256K 判定头，冻结，C20 信号链）+ 域头（生成，C24 目标）**。基座从 foundation_v1_general 出发（body 保留 general 判定能力），训练双 loss：域 SFT（answer PPL 收敛）+ general 空间保留（gen_loss，防 body 漂移破坏判定空间）。
  - `ResonanceNeuron`：新增 `judge_lm_head` 属性 + `return_judge_logits`（forward 输出 general 256K logits）
  - `train_domain_target_sft.py`：基座改 foundation_v1_general、双头加载、双 loss 训练、save/verify 含 judge_lm_head_state
  - `ensemble.py forward_train`：per_neuron_nll 回退 C20 general 空间投影 NLL（judge_logits 直接在 256K 空间对齐 targets，无转译噪声）
  - `train_round_level_quality.py`：移除 native NLL（build_per_neuron_targets 删除），dialogue neuron 注入共享 general 256K 判定头
  - `loader.py`：识别 `judge_lm_head_state` 注入判定头
  - 冒烟验证（60 步）：域头 answer PPL 收敛 + general 判定对角保留（code=3.65 最低）——**双头方案成立**
- **✅ 完整重训完成（foundation_v1_dual，4 域 × 6 epochs）**：
  - code ✅（best eval PPL 6.8）→ 训练后 general 判定对角保留（code=1.2/math=7.7/zh=15.8/en=5.1）
  - math ✅（best eval PPL 5.5）→ 判定对角保留（code=9.7/math=2.7/zh=16.0/en=6.5）
  - zh ✅（best eval PPL 319.1）/ en ✅（best eval PPL 167.3）
- **C20 判定重训完成 + 判定 5/5 达成（collab_v3_c24v2，2026-08-10）**：
  - C20 重训（train_round_level_quality.py，judge_logits general 空间监督）完成：per_neuron_nll 正常（dialogue 回合 zh=6.90 最低）
  - **但端到端判定 1/5 失败** → 诊断（diag_c20v2_route）：
    - judge NLL 完美对角 4/4（code=1.15/math=2.72/zh=11.36/en=1.72）——C24 双头信号链可靠
    - **quality_head 学成常数偏移**（zh_aug2_dialogue ql 68-102 对任何回合内容无关；code≈-2/zh≈0 也内容无关）→ 膨胀根因：C23 时代已膨胀（−4.2→50），logit 大 → softmax 饱和 → KL(actual||ideal) 梯度消失 → 自增强压不住；C24v2 绝对 NLL 监督（nll_z=per_neuron_nll.clone()）也没救回
    - dialogue neuron 推理时无 judge_lm_head（ckpt 无 judge_lm_head_state，loader 不注入）——训练有 fallback 头、推理没有，信号链断一环
  - **修复（C20v2，上限最高）**：executive 判定改用 **judge NLL 主信号**（C20 当年 5/5 的原始信号链，general 空间可比，无训练依赖、无膨胀）——`_parallel_forward`/`forward`/`think` 增 return_judge_logits 收集 round1 judge logits；`_executive_route` 算各 neuron judge NLL → 域聚合取最低 → 与启发式融合（NLL 差 ≥1.0 显著占优才切换，回退安全）；quality z-score 保留为 judge 不可用时回退
  - **✅ 端到端判定 5/5**（code→code/math→math/zh→zh/dialogue→zh/en→en，无回归）
  - **遗留（已修 2026-08-10）**：~~quality_head 膨胀根因未修~~（C25-G std 标准化修复，见下）；C24 域生成能力仍碎片（code "def __[3,b]"/zh "。"——域 SFT 数据少，非架构问题，v2 数据扩充重训中）
  - **域生成碎片根因闭环（2026-08-10，diag_c24_domain_generate）**：单独验证 4 个域头 neuron 生成能力（不经 ensemble/head，`--dir data/foundation_v1_dual`）——code→". a given range where a function is traination..."（乱码）、math→"for every triangular..."（乱码）、zh→"数列 函数已知的发现"（碎片）、en→空。**独立域头生成同样碎片 → 确认是训练数据不足（每域仅 3000 条短 QA），非推理/装配 bug**。数据源：`data/sft/{domain}_sft.pt`（如 code 第 1 条 instruction='Create an array of length 5...' response='arr = [2, 4, 6, 8, 10]'，短指令-响应对，过拟合片段、泛化差）。
  - **待办（下一步候选）**：扩充域 SFT 数据（如从 pretrain 语料构造续写样本 + 多样化 QA，目标每域 1-3 万条）→ 重跑 `train_domain_target_sft.py --domains code,math,zh,en --epochs 6`。数据规模/预算需与用户确认后启动。
  - **数据扩充完成（2026-08-10，build_domain_sft_v2.py）**：用户确认"本地缓存组合 + 2-3 万条/域"。盘点发现 data/cache 有未利用的 HF 缓存标准 SFT 数据集 + 本地大语料：
    - code: code_alpaca-20k 全量（原 3000 条正是其子集）→ **17599 条**（MAX_FULL_CHARS=512 过滤长样本）
    - math: gsm8k main train+test 完整 QA + math_texts 行级续写样本 → **22264 条**
    - zh: alpaca-zh → **30000 条**；en: alpaca → **30000 条**
    - 输出格式与 train_domain_target_sft.py 完全兼容（{instruction,input,response,prompt,full}，prompt 前缀匹配 answer 定位验证 OK）
  - **Smoke 链路验证通过（CPU）**：4 域新数据加载/训练/保存/回读一致；judge 判定对角保留（math: code=9.7/math=3.1/zh=16.2/en=5.9）——双头装配未被破坏。**全量重训在本机 CPU 执行**（无 4090D；此前 C24 完整重训亦为本机 CPU 3.4s/步）。
  - **✅ 全量重训完成（2026-08-11 02:15，`train_domain_target_sft.py --domains code,math,zh,en --epochs 2`，数据量 ×10）**：4 域 × 2 epochs 全部跑完（总 ~23700 步 CPU）。各域 best answer PPL：code=3.4（step 3600）/ math=3.7（step 3400）/ zh=70.2（step 7000）/ en=69.9（step 4800）——code/math 拟合良好（<4），**zh/en 仍高（~70，zh 训练 loss PPL 4000+ 收敛差）**。
  - **✅ 生成验证（2026-08-11，diag_c24_domain_generate --dir data/foundation_v1_dual）**：
    - code（"Write a Python function to compute the Fibonacci sequence"）→ "…series of integers in the range…def __PString of_thrices fib; fib = (1) +" —— 仍碎片（含 def/fib 代码痕迹）
    - math（"60 mph for 3 hours…"）→ "First find out how many hours faster the trips time…" —— 英文片段可读、语义错乱
    - zh（"写一个 Python 函数计算斐波那契数列"）→ "这是一个简单的我的斐波那契数列斐波那契数列斐波那契数列…" —— **较 v2 前（"."）改善为可读中文片段**，但重复无进展
    - en（"What is the capital of France?"）→ "by the United of the world." —— 较 v2 前（空输出）改善为短句，仍碎片
    - **结论：数据扩充 ×10 后生成从碎片/空输出→部分可读短片段（zh/en 明显改善），但未达流畅完整文本；code/math answer PPL 虽低（3.4/3.7）生成仍碎片（SFT 数据短 QA 过拟合片段）。**
  - **复验（2026-08-11 06:01 定时任务第 2 次运行，随机采样）**：
    - code → "to calculate the squres10thLocal a = . def __PSelect the fib1 and n is in range(2, ): = int(i"（仍碎片，含 def/fib 痕迹）
    - math → "First find the number of miles theyatching to work: the speed: …dri to"（英文碎片可读、语义错乱）
    - zh → "和，，然后然后打开一个一个直角三角形，，，然后将 将 将将 将将将 将 将 将将将…"（中文碎片重复无进展）
    - en → "that are equal to the number of countries and China. .ZZQQ…"（短碎片）
    - 结论：**第 2 次抽样同为碎片/短片段，结论稳健（生成未达流畅文本，zh/en 仅片段级改善）**。
  - **复验 2（2026-08-11 08:01 定时任务第 3 次运行）**：code → "in the range. by using a list…def _____(n):"（含 def/list 痕迹）；math → "First let the to run the pool…60 miles: hour.3 hours"（英文碎片）；zh → "中中包含…下面是一个简单的斐波那契数列 斐波那契数列斐波那契数列…"（可读中文片段+重复循环）；en → "in terms of the country, and the country. and the city is an average.-�.�&&>^"（英文碎片+乱码尾）。**结论与第 1/2 次一致：生成从碎片/空→片段级改善（zh 可读），未达流畅文本。**
  - **复验 3（2026-08-11 09:40，人工诊断：diag 输入格式 bug 修复）**：发现 diag_c24_domain_generate.py 生成输入未按训练格式补 "\n"——训练样本 = prompt+"\n"+response（answer 起点在 prompt+'\n' 之后，ckpt 标记 c24_domain_sft=True 已注明"生成时输入需补 \n"）。修复（prompt+"\n"）后生成显著改善：
    - code → "def Fib (n):  a = (n - -1)) Fib i fib i 0, (n - - n - -"（def 结构+函数体痕迹，较修复前 "fib, fib. fibub." 结构化）
    - zh → "下面是一个 Python code\n```python的函数\n```\ndefport = 1\nfout = ="（**markdown 代码块结构**，较修复前碎片重复大幅改善）
    - en → "The author of America is that India... is is the capital of ..."（英文句子结构，较修复前空输出改善）
    - math → 仍有碎片
    - **结论：定时任务 3 次复验的"碎片"结论部分受 diag 输入格式错误影响（未补 \n）；修复后生成从"碎片/空"→"有结构代码/中文/英文片段"，长程连贯仍有限（模型规模 + SFT 短 QA 数据限制，非架构问题）**
  - **✅ judge 对角验证（训练前后 general 判定 NLL 对比）**：**3/4 保留，zh 例外**——
    - code neuron：前 code=1.1/math=8.0/zh=15.8/en=5.3 → 后 code=1.0/math=6.5/zh=14.0/en=4.2 ✅（code 最低）
    - math neuron：前 code=9.7/math=3.1/zh=16.2/en=5.9 → 后 code=8.5/math=2.6/zh=15.4/en=6.3 ✅（math 最低）
    - zh neuron：前 code=10.0/math=8.9/zh=11.2/en=7.1 → 后 code=7.8/math=6.9/zh=7.4/en=6.2 ❌（**en=6.2 < zh=7.4，训练前后均非对角**——zh 基座 general 空间中文 NLL 天然偏高（C24v1 已记录），本次重训 gen_loss 未保护住）
    - en neuron：前 code=6.5/math=5.3/zh=17.6/en=2.2 → 后 code=2.5/math=4.2/zh=15.0/en=2.1 ✅（en 最低）
    - **风险（已重验解除，2026-08-11）**：judge NLL 是 executive 判定主信号（C20v2），zh neuron 非对角 → 中文回合存在被误判为 en 的风险。**verify_c21_generate 重验（9 神经元装配 collab_v3_c24v2 + foundation_v1_dual）：端到端判定 5/5 无回归**（code→code/math→math/zh→zh/dialogue→zh/en→en）——zh 回合时 5 个 dialogue neuron（zh 空间）参与域聚合，zh 域仍最低，判定安全。
  - **✅ C24 验证使命完成（2026-08-11 08:01，定时任务共运行 3 次，结论一致收敛）**：生成碎片→片段级改善、judge 对角 3/4 保留（zh 例外）均已在计划记录。**定时任务建议暂停/删除（使命已完成）**；若需复验生成，手动运行 `diag_c24_domain_generate.py --dir data/foundation_v1_dual` 即可（已修复输入格式，prompt 自动补 "\n"）。
  - **⏳ 待办（下一步候选）**：① ~~修复 zh 域判定对角~~（2026-08-11 端到端判定 5/5 重验保持——zh 单 neuron 非对角被 dialogue 域聚合覆盖，非阻断，长期可调 zh SFT/gen_loss 权重）；② 提升 zh/en 生成质量（answer PPL ~70 远高于 code/math，SFT 数据/训练配置待调优；挂载后培养期喂养数据渐进改善）。
  - **zh_general 残留收敛（2026-08-10，用户确认 9 阵容）**：9 = 5 对话（zh_aug0-3_dialogue + zh_std0_dialogue）+ 4 域（code/math/zh/en）。查证：zh_general 设计为 SHARED_EXPERT_ID（experiment_config），但 **assemble_cortex/cortex 从未传 shared_expert_id → shared_expert 机制从未启用**；实际被 cortex 全量扫描误加载为普通 neuron（中文任务竞争者、训练最弱 PPL 257，verify_c19 注释"排除 zh_general 旧产物干扰"）。C24 双头后每 neuron 自带 judge_lm_head，single always-active 底座机制冗余 → **删除 data/neurons/neuron_zh_general.pt**，verify_hotswap_integration 改用 zh_std0_dialogue，experiment_config SHARED_EXPERT_ID 废弃注释。装配收敛为 9 阵容。

  - **✅ 9 神经元挂载就绪验证（2026-08-11）**：
    - **test_api_dialogue 装配升级为 9 阵容**（5 对话 + 4 域 + collab_v3_c24v2 + judge EMA 预热）实测：Q1 "你好" → **"你好！今天天气很好。有什么情况吗？"**（流畅完整）；Q3 → "我是一个人工智能助手，无法正文"（半流畅）；Q6 → "当然。这本书"（自然开头）；Q2/Q4/Q7/Q8 短碎（模型规模限制，培养期喂养渐进改善）；**符号乱码/混字消失**（原 5 neuron + 旧 collab 装配 Q8 出乱码 "漫步a 江莜れ赌博…"）
    - **根因修复：API 装配路径用旧协作层**——`load_model_on_startup` 默认 collab_name=`cross_spec_dialogue.pt`（2026-08-06 旧产物，非 C16-C24 验证链产物）→ 对话乱码。已修复：collab 显式用 `collab_v3_c24v2.ckpt.pt`（C20v2 判定重训，judge NLL 主信号）+ `extra_neurons_dir=data/foundation_v1_dual`（C24v2 双头域 neuron，9 阵容）；环境变量 `TAIJI_COLLAB_NAME`/`TAIJI_EXTRA_NEURONS_DIR` 可覆盖。`load_model_on_startup` 验证装配 9 神经元（5 对话 + code/en/math/zh）
    - **挂载就绪结论**：判定 5/5 + 对话链路（API 等价参数 temperature 0.55/top_k 15/rep 1.4）工作正常，对话质量达"培养起点"——可挂载客户端进入培养期（喂养数据渐进改善）；域生成能力（C24）留待培养期验证/喂养
  - **✅ C20 判定重训 v2 完成 + 验证闭环（2026-08-11）**：8/10 的 c24v2 quality_head 基于 **v1 域 neuron**（3000 条/域）重训；C24 v2 全量重训（2-3 万条/域）覆盖域 neuron 后 quality_head 与新域 neuron **失配**。判定 5/5 不受影响（C20v2 判定主信号 = judge NLL），但 C25-G 膨胀修复后的 **quality proxy 恢复闭环**需在新域 neuron 上重跑 C20 重训。已备份 `collab_v3_c24v2_v1.ckpt.pt`，重训命令 = `train_round_level_quality.py`（neuron-dir foundation_v1_dual + warm start 自 v1 + save-name c24v2 覆盖，1090 步 ≈ 74min CPU）。定时任务 e1ec4a91（C24 完成后自动 C20 重训，描述 v1 场景）已删除（手动接管）。
    - **v2 产物验证（全部通过）**：
      1. `collab_v3_c24v2.ckpt.pt` 已覆盖为 v2（12:37，quality_head 9 neuron 全更新 + phasor 演化：phasors max|d|=1.82 近乎相位翻转、omega 差 0.245）
      2. `verify_c21_generate.py`：**判定 5/5 无回归**（code→code/math→math/zh→zh/dialogue→zh/en→en）
      3. `verify_c25_f_e2e.py`：**端到端 10/10 PASS**（判定链路 5/5 + 三阶段 {prev} 传递 + 异常隔离 + continuous 阶段可用）
      4. `verify_c20_quality_fallback.py`：**3/3 PASS**（judge 失效时 quality z-score 回退不崩溃、判定合法域）
    - **挂载生成无退化（diag_c20_v1_vs_v2.py 对比）**：v2 全 4 问产出完整句子（"你好，很高兴。"等）；v1 碎片更多（Q2"抱歉（"/Q3"1/ ("/Q4"我抱歉，"）——**v2 略优于 v1**。test_api_dialogue 单次抽样碎片为随机波动（temperature=0.55），非 phasor 大差异导致。
    - **已知限制（记录不阻塞）**：quality proxy 回退判定与 judge NLL 一致率 **2/5**（math→en/zh→code 错判）——z-score 是"相对自身历史水平"的弱信号，跨 neuron 可比性架构性弱于 judge NLL（general 空间天然可比）；兜底场景（judge 完全不可用）可接受，挂载主路径 judge NLL 5/5 不受影响。C25-F 端到端阶段 2（code 域）在 prev 碎片污染时不出代码——模型能力上限（C24 zh PPL 70.2 高），diag_c25_f_stage2 确认无 prev 时中文/英文指令均出代码，编排机制正常。
    - **C20 v2 完成 → C25-F 端到端验证闭环关闭；9 神经元挂载完全就绪（判定 5/5 + 生成无退化 + 回退路径存活）**

---

**C25 对比问题解决（2026-08-09 用户指令：态极 vs 人脑对比中的问题开始解决；词库容量不限 + 实时编辑 → 不需要热插拔）**：
- **背景**：[TAIJI_VS_HUMAN_BRAIN_COMPARISON.md](file:///e:/taiji-neuron/plans/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md) 2.11 借鉴边界（装饰/角色偏移项）+ 2.12 最高上限方向。相位同步本体化（缺口 R 核心）已由 C23 闭环；剩余对比问题清单如下。
- **C25-A ✅ 词库实时编辑（2026-08-09，用户决策落地）**：热插拔（态极工程简化）→ **词库不做限制 + 实时编辑**（人脑词汇加工：脑区词汇分工 + 词汇生长，无"拔插"）。
  - `EditableVocabulary`（translator.py）：包装 SentencePiece，运行时 `add_tokens` 追加 token（id ≥ base vocab，base 已含自动复用 base id）；encode 扩展区前缀树最长匹配 + 剩余走 SP；decode/vocab_size/id_to_piece/piece_to_id 合并扩展区；扩展区持久化 JSON 可热加载。
  - `TokenizerHub`：`to_editable`（幂等升级）/ `add_tokens`（实时追加 + 持久化）/ `unregister_domain`（集合级编辑）。
  - `resize_linear_for_vocab` / `resize_lm_head_for_vocab`：neuron lm_head 随词表扩展，旧行权重保留、新行均值+噪声初始化（judge_lm_head 不受影响）。
  - **下游自动重建**：tokenizer_fingerprint 的 vocab_size 变化 → 对齐/转译表缓存自动失效重建（端到端验证：12000×50000 → 12000×50002）。
  - **256K 去硬编码（2026-08-09，用户质疑"256k 怎么好像是个硬设计"）**：256K 是当前 general 词表（sp_general.model）的**实例值**，判定可比性的本质是"所有 neuron 共享同一投影空间"，不依赖 256K 数字。修复：判定头/共享表维度一律从权重 shape 推断（`judge_lm_head_state.shape[0]` / `shared_lm_head["weight"].shape[0]`），LoRA 过滤改用 `loader.general_vocab_size()`（从 sp_general.model 动态获取，失败回退 256000）——general 词表可重训/实时扩展（C25-A EditableVocabulary + resize 工具）而不破坏装配。验证：动态 256000 == ckpt 256000。
  - 冒烟：verify_c25_vocab_edit.py **27/27 PASS**。
- **剩余对比问题清单（待办）**：
  - C25-B ✅ STDP 突触生长/修剪本体化（2026-08-10，C20 重训完成后实施，缺口 R 修复）
    - **现状盘点（2026-08-09）**：STDPTracker 已注入 ensemble（loader Step 2）+ 推理 record_firing + sleep 期 apply_all_updates（权重缩放 [0.5,2.0]）+ SleepConsolidator 弱连接修剪（weight<0.01）——**非纯装饰，但缺通道级结构可塑性（excite/inhibit_channels 条目修剪/生长）且不参与 forward_train**
    - **设计（上限最高，突触生长/修剪本体化）**：① STDPTracker 增共激活统计累积（(pre,post)→count/sim/dt 持久化，跨会话）；② `apply_structure_updates`：长期低共激活通道条目修剪 + 高共激活缺失通道生长（邻居相似初始化）——连接层"突触可塑性"从权重缩放升级为结构演化；③ 时机：C20 重训完成后接入 sleep（离线路径，不碰 forward_train 监督，规避 C23-C4 式监督打架）
    - **实施（2026-08-10，C20 判定 5/5 后启动）**：
      - `STDPTracker.accumulate_coactivation()`：firing_history 按 round 排序，pre 先于 post 发放 → (pre,post) 有向对累积 count+total_sim（与 STDP 语义一致，幂等）
      - `STDPTracker.apply_structure_updates(neurons)`：**修剪**——通道存在但 (post,pre) count < prune_count_threshold=2 **且** 权重 L1 均值 < 0.01 → 删除条目 + 关联 scale param/bias buffer 一并清理（防孤儿参数）；**生长**——count ≥ grow_count_threshold=5 且 avg_sim ≥ grow_sim_threshold=0.3 但通道缺失 → 建立新通道（邻居相似初始化：与目标 peer 共激活最相似的已有通道权重 + 0.005 噪声；无邻居则 init_std 标准初始化）；保守规则：强权重通道即使无共激活也保留（防误删已学习连接）
      - `get_state_dict/load_state_dict`：共激活统计持久化跨会话（与 C25-D replay buffer 同一模式，firing_history 短期不持久化）
      - `SleepConsolidator.consolidate` 增 `stdp_tracker` 参数（向后兼容）：步骤 2.6 结构演化（accumulate → apply_structure_updates），返回 channels_struct_pruned/channels_grown；sleep_engine REM 阶段调用传入 self._stdp_tracker
      - 验证：verify_c25_b_stdp.py **21/21 PASS**（有向统计累积/修剪+param 清理/强权重保留/生长+邻居初始化/持久化/sleep 接入）；C25-D 无回归 17/17 PASS
  - C25-C ✅ 神经调质深度耦合训练（2026-08-10，对比文档 2.11"调质状态记录未深度耦合训练"+ 171 行乙酰胆碱未实现修复）
    - **现状盘点**：DA/5-HT/NE 已耦合 lr/refractory/field_write/attention 温度/FFN 增益（S9 冒烟过），但调质目标由 sleep 手工阈值规则更新（未形成可验证的训练闭环）；对比文档 171 行乙酰胆碱（attention 调制）缺失
    - **实施（上限最高，DA=奖励 / ACh=新颖性互补）**：
      - `NeuromodulatorState` 增 **acetylcholine**（0-1，默认 0.5=中性）+ `set_targets(acetylcholine=...)` + `step` EMA + 持久化（旧 ckpt 无 ACh → 默认中性兼容）
      - `get_attention_focus_gain()`：ACh → attention 聚焦增益（映射 0.6+ACh×0.8 ∈ [0.6,1.4]，0.5→1.0 与 DA/NE/5-HT 中性约定一致）
      - **ensemble 注入**（两处 temp_gain 路径）：`temp_gain = NE_temp × ACh_focus`——NE=警觉主调制、ACh=新颖性精细调节，互补不覆盖（hasattr 兼容旧实例）
      - **训练闭环**（sleep_engine._update_neuromodulators）：loss 变化率同时驱动 DA 与 ACh——loss 上升（新颖/困难）→ ACh↑0.85（聚焦新输入）；停滞→中性；快速下降（熟悉）→ ACh↓0.35（习惯化）。DA=奖励预测误差、ACh=新颖性，同一 delta 双信号
      - 验证：verify_c25_c_neuromod.py **23/23 PASS**（EMA/映射/组合调制/持久化+旧 ckpt 兼容/训练闭环 DA-ACh 联动/既有接口无回归）；C25-B 21/21、C25-D 17/17 无回归
  - C25-D ✅ 睡眠重放真重放 + 突触稳态下调（2026-08-09，对比文档 2.6/2.11 弱项"态极 sleep 是'拿累积样本离线训练'，重放/下调是方向性借鉴，未实现生物意义上的'逐条回放 + 全局缩放'"修复）：
    - **真重放**：`record_high_resonance_state` 增 `active_nids`（PlayEngine 记录共振时传激活 neuron 集）；`consolidate` 重放时用 active_nids 再激活共激活统计（人脑海马回放 → 皮层再激活 → 突触巩固），取代"纯统计占位"假重放；旧格式记录（无 active_nids）兼容仅计数
    - **突触稳态下调（downscaling）**：consolidate 新增全局 side_channels ×0.98（NREM 慢波全局缩放）——强通道净保留（强化×1.1 后 ×0.98 ≈ ×1.08）、弱信号整体下压，与弱通道修剪互补（连续调节 vs 离散清除）
    - 冒烟：verify_c25_d_replay.py **17/17 PASS**
  - C25-E ✅ 连续时间动力学替代离散共振轮次（2026-08-11，对比文档 2.11"刻意简化：离散共振轮次替代连续动力学"修复）
    - **核心**：`taiji/resonance/continuous.py`（ContinuousResonance）+ `ensemble.continuous_forward`（可选路径，不改变 forward/executive）——相位绑定驱动的连续激活替代不应期硬门轮替：
      - 时间步进 T（默认 8）微步积分（dt=1/8）；每步相位 Kuramoto 演化（复用 PhasorDynamics.evolve）
      - 激活强度 a_i(t) = σ(β·(binding_i(t)−b0)) 连续驱动"谁参与、权重多少"（同相强参与、异相退场）——替代 round1 全量 + 不应期硬门 + max_rounds 的离散轮替
      - 场随时间积分 F(t+dt) = F(t) + dt·Σ a_i·project(v_i)·conf_i（confidence 只调制场写入）
      - 融合权重 w_i = Σ_t dt·a_i（时间平均激活=参与度，与离散"共振分"对齐）
      - 收敛 = 绑定分布 std 稳定（相位锁定，min_steps 后检查防单步假收敛）——连续版自适应停止
      - **安全性边界（C23 同款）**：t=0 独立前向采集判定信号（judge NLL 主信号链），连续激活不进入判定路径
    - **验证**：verify_c25_e_continuous.py **20/20 PASS**（激活单调/中性/连续无硬跳变/权重=Σdt·a·conf/收敛判据/Kuramoto→绑定→激活闭环/输出结构/同相权重 0.233>异相 0.068/场积分/判定信号保留/forward 无回归/确定性）；verify_c21_generate **判定 5/5 无回归**
    - **✅ 增量一：cortex 生成路径接入（2026-08-11）**：`collab_mode="continuous"` 显式启用——cortex.think 转发 ensemble.continuous_forward；_generate_p7 复用 executive 判定（judge NLL 主信号）+ domain 内 leader 选择（continuous_weights=时间平均激活）。verify_c25_e_collab_ab.py **20/20 PASS**：
      - 判定 5/5 两种模式一致（code→code/math→math/zh→zh/dialogue→zh/en→en）
      - A/B 生成（max_tokens=20）：**continuous 在 dialogue/zh 质量优于 executive**（dialogue "对不起。" vs executive "。我非常开心…"；zh "下面是一个简单的 Python 代码斐波那契数列" vs executive "。"）——连续激活选择让 leader 更稳定；code/en 相当
    - **✅ 增量二：训练路径 forward_train 连续化（2026-08-11，C25-E 最后增量）**：`forward_train` 新增 `continuous: bool = False` 参数（默认 False = 原离散路径，全部既有调用点零影响）——round 1（t=0 独立前向）后进入连续积分主循环替代 round 2+ 离散轮次：
      - 相位连续演化（可微 Kuramoto）→ 激活 σ(β·binding) → 软过滤（低激活退场，保留≥1）→ 场条件化 forward（只 forward 激活 neuron）→ 场积分 F(t+dt) = F(t) + dt·Σ a_i·project(v_i)·conf_i → 权重累积 Σdt·a
      - **融合权重 = 时间平均激活归一化**（替代 softmax(scores/temp)），与推理 continuous_forward 同口径
      - **C23-C4 监督纯净化**：final_judge_logits/final_logits 在 round 1 采集（`if round_num == n_rounds or continuous`），连续积分不更新 final_logits——监督测"谁能预测好"（纯净 NLL），相位不被自组织驱动漂移；phase_loss/scores 段保留（ω/K 梯度路径）
      - 输出新增 `continuous_weights`（未归一化时间平均激活）
      - 顺手修复基线缺陷：`quality_logits_t` 提前统一初始化（原 router/residual 融合分支未定义 → UnboundLocalError，verify_forward_train_diff 4/6→5/6）
      - 验证：verify_c25_e_forward_train_continuous.py **25/25 PASS**（输出结构/权重=时间平均激活/融合权重归一化/监督纯净（per_neuron_nll round 1）/连续可微（phasors+omega+coupling_k 梯度全通）/离散无回归/相位演化推进）；verify_c25_e_continuous **20/20 无回归**；注意 [0,0,0,π] 是绑定驻点（det=0 无牵引）+ 同 omega 整体旋转不改变相对绑定——演化/梯度测试需非驻点相位 + 异质 omega
    - **遗留（下一步增量）**：训练路径 forward_train 连续化（可微积分，C23-C4 监督纯净化模式）✅ 已完成（见上）；loader 默认装配（continuous 替换 executive 需 A/B 规模化验证后决策）
    - **✅ 增量三：loader 默认装配决策（2026-08-11，A/B 规模化 + 装配实测后**回退**）**：
      - **A/B 规模化（verify_c25_e_ab_scale.py，22 混合域 prompt）**：continuous 全面不劣且质量占优——非空率 22/22 持平；平均重复率 continuous 0.012 < executive 0.027；逐 prompt 质量 continuous 10 胜 > executive 6 胜（6 平）
      - **装配实测反转（关键）**：默认切 continuous 后 test_api_dialogue 8 问空输出 5/8（Q2-Q6 全空）→ 回退默认 executive。根因诊断（diag_c25_e_default_empty.py）：**连续模式多 neuron 协作不稳定**——zh 对话激活 5 个 dialogue neuron（zh_aug0-3 + zh_std0），连续激活时间平均权重均分（同相群体绑定→权重近均等）→ leader 选到弱响应 neuron → 空输出/短碎；executive 用 LOO 共振分能区分 neuron 强弱 → 稳定。单 neuron 域（code/math/en 各 1 neuron）无协作问题 → continuous 稳定（A/B 多数 prompt 落此场景，掩盖了 zh 协作缺陷）
      - **决策：loader 默认保持 executive**；continuous 留作显式可选（collab_mode="continuous"）。**遗留修复方向**：continuous leader 选择需融合质量信号（如连续权重 × round1 共振分/NLL 质量）防止弱 neuron 独占——待后续增量
      - 已知限制（与模式无关，同源判定）：判定正确率 18/22 (81.8%)——中文 code/math 指令判到 zh（zh 域 5 neuron 聚合优势），挂载培养期可数据喂养改善
    - **✅ 增量四：continuous leader 质量信号修复（2026-08-11，增量三遗留落地）**：
      - **根因**：continuous leader 用时间平均激活（final_scores=continuous_weights）选，同相群体权重均分（验证实测 5 个 dialogue neuron 全 ~0.29）→ leader 选到弱响应 neuron → zh 对话空输出 5/8
      - **修复**：`continuous_forward` 新增 `round1_scores`（t=0 场共振分，field.score 口径与离散 forward 的 round_scores 一致）——质量信号有区分度（验证实测 max-min=0.70：zh_aug2=0.73 最高 / zh_std0=0.027 最低）；cortex `_generate_p7` continuous 分支 leader 改用 round1_scores 优先（fallback 时间平均激活）
      - **验证**：verify_c25_e_leader_quality.py **3/3 PASS**——continuous 挂载 8/8 非空（此前 5/8 空输出消除）+ round1_scores 有区分度 + executive 8/8 无回归；verify_c25_e_ab_scale.py 修复后 **continuous 22/22 非空 + 质量 17/22 ≥ executive**（三质量断言全过）
      - **注意（数据波动）**：单次 A/B 重复率方向不稳定（增量三 0.012<0.027，增量四后 0.039>0.021）——temperature 0.55 采样波动大，重复抑制优势需多次采样取均值确认，**默认装配保持 executive 的决策不变**（连续模式空输出已消除，质量不劣，可作显式可选；默认切换需更大样本统计支撑）
      - **✅ 增量五：默认装配切换 continuous（2026-08-11，多次采样统计确认后落地）**：verify_c25_e_ab_stats.py（12 prompt × 3 次采样取均值）**4/4 PASS**——非空率 1.00 持平、重复率 **continuous 0.011 < executive 0.022**（3 次采样均值稳定，确认增量三/四单次反转是采样噪声）、逐 prompt 质量 **9 胜 2 负 1 平**。`cortex.generate` 默认 `collab_mode` 切换为 `"continuous"`：
        - 判定 5/5 无回归（continuous 复用 executive 判定，judge NLL 主信号不受影响）
        - 挂载实测（test_api_dialogue 默认参数）8/8 全非空，Q1 "你好" → "你好，很高兴！今天天气真美好的一天。"（流畅完整，空输出 5/8 问题彻底消除）
        - **C25-E 全部增量闭环：核心机制 → cortex 接入 → 训练路径连续化 → leader 质量修复 → 默认装配切换**
  - **✅ 培养期端到端闭环验证（2026-08-11，C25-E 后下一阶段：喂养数据渐进改善落地）**
    - **验证**：verify_feed_sleep_e2e.py **14/14 PASS**——"feed → sleep Phase 2 训练 → 影子写回 live → ckpt 保存 → 训练后推理"完整闭环（真实 9 神经元装配 + FeedEngine/SleepEngine 接线）：
      - feed：zh 样本 8/8 消化（质量评估通过）→ `get_pending_samples_by_domain` 按域分类（{'zh': 8}）
      - sleep 训练：样本被消费（training_samples_used=8）、loss 有限（5.39）、训练-训练互斥锁释放（可进入下一睡眠周期）
      - 影子 COW 写回：zh lm_head + shared_embedding 训练前后权重变化（经验积累生效，live 推理稳定）
      - ckpt：训练后自动保存 `cortex_state.pt`（968.5 MB，fp16 shared_embedding + per-neuron lm_head/embed_adapter）→ `load_state` 恢复成功（隔离临时目录验证，不污染生产）
      - 训练后推理：code/zh generate 非空不崩（continuous 默认装配）
    - **顺带修复（机制演化收敛）**：contrastive phase 混合规格维度崩溃——compact（field/hidden 512）与 standard（768）直接 stack/EMA → "size mismatch 512 vs 768" 每次睡眠训练都失败（被 try/except 压制）。修复（sleep_engine `_train_contrastive_phase`）：hidden 无统一投影层 → pad 到公共 max dim（pad 部分贡献 0，L2 归一化后 cosine 语义不变）；field → 优先用 ensemble 跨规格投影层（`_project_vec`，与推理路径一致）否则 pad；domain_prototype 更新与 route_loss 冷启动 → 用**原始维度** hidden（prototype 在 neuron 自身空间 512/768 各自）。修复后 contrastive **route=0.2048/proto=0.0214/align=0.0323，9 神经元全参与**，不再打 warning
    - **结论**：培养期闭环可用——客户端可挂载，喂养数据在每次睡眠周期渐进改善 zh/en 生成（answer PPL ~70 → 喂养逐步下降）
    - **✅ 渐进改善验证 + 破坏性更新修复（2026-08-11，verify_feed_sleep_progressive.py 24/24 PASS）**
      - **验证**：5 轮"feed 8 条新 zh 样本 → sleep Phase 2 训练"循环，held-out 评估集（10 条独立提问式句子，从未参与训练，口径与 `_train_single_neuron` 一致）逐轮追踪 PPL
      - **根因实证（首跑 FAIL）**：原配置（lr 1e-3 × maturity 幼稚态 3.0 = 3e-3，shared_embedding 直接训练，3 epoch）下训练 loss 单调降（5.04→2.44）而 held-out zh PPL 单调爆炸 **10761 → 26392 → 24952 → 59809 → 154179 → 342100**——灾难性遗忘/破坏性更新（小样本 × 高 lr × 共享大嵌入表）
      - **修复（sleep_engine `_train_single_neuron`）**：分层学习率——shared_embedding（256K vocab 共享感官层）lr 降 100 倍至 **1e-5**（经验驱动本质是长期缓变积累），lm_head/embed_adapter 用温和 lr **3e-4**（min(adaptive_lr, 3e-4)）；epoch **3 → 1**（小样本重复学习加深过拟合）
      - **修复后趋势（24/24 PASS）**：held-out PPL **10761 → 7924 → 6607 → 5633（最低）→ 5887 → 7181**——末轮比 baseline 降 33%，前 3 轮单调降；生成非空率 4/4 持平、重复率 0 不升；每轮样本消费/训练锁/ckpt 保存全过；verify_feed_sleep_e2e 回归 14/14 无破坏
      - **遗留（已知限制）**：① zh general 基座 baseline PPL 10761 极高（≈随机 50K vocab，C24 遗留弱基座）——短期喂养改善有限，长期需基座级 SFT；② 轮 4-5（编程/数学主题批次）PPL 轻微回升——训练批次主题与评估集分布偏移所致，后续可调每轮样本量/主题混合/多轮验证集
  - C25-C：神经调质深度耦合训练（✅ 已完成 2026-08-10，见上，此条目残留清理）
  - C25-F ✅ 多阶段任务模式链（task-set 序列，2026-08-11，对比文档 2.11"回合级路由替代连续任务切换（多阶段任务留 v2）"修复）
    - **实现**：`cortex.generate_staged(stages)`——每阶段 = task-set（"prompt" 指令 + "mode" 任务模式 executive/continuous + 可选 "domain" 域约束 + "max_tokens" 覆盖）；阶段间显式传递中间输出（"{prev}" 模板填充或自动拼接），如"zh 理解 → code 生成 → zh 表达"；异常阶段隔离（输出空串，后续继续）
    - **验证**：verify_c25_f_staged.py **18/18 PASS**（{prev} 模板/自动拼接/首阶段无拼接/task-set 参数透传（domain+mode+max_tokens）/空 prompt 跳过/异常隔离/zh→code→zh 三阶段编排）；py_compile OK
    - **✅ 端到端完成（2026-08-11，C20 重训 v2 完成后）**：verify_c25_f_e2e.py **10/10 PASS**（真实装配 9 神经元：判定链路 5/5 + 三阶段 zh→code→zh {prev} 传递 + 异常隔离 + continuous 阶段可用）。内容质量降级为信息性报告——diag_c25_f_stage2 定位：无 prev 时中文指令/英文 prompt 均出代码（domain/mode/判定正常），阶段 1 zh 碎片污染 prev 是模型能力上限（C24 zh PPL 70.2 高），非编排机制问题
  - C25-G ✅ quality_head 膨胀根因修复（2026-08-10，C24 遗留闭环）
    - **膨胀根因**（C23 时代诊断）：quality_head 学成常数偏移（zh_aug2 ql 68-102 内容无关）——logit 大 → actual=softmax(logit/1.0) 完全饱和（0/1 独热）→ KL(actual||ideal) 梯度消失 → 自增强压不住；C24v2 绝对 NLL 监督（nll_z）也没救回
    - **修复（上限最高，std 标准化）**：C15 contrastive loss 的 actual 改为 **std 标准化**（减 detach 均值 ÷ detach 标准差）再 ÷ 温度 1.0——softmax 输入恒 ~±2，永不饱和、梯度恒非零；语义：actual 只反映 neuron 间相对质量差异（与 ideal z-score 同构）。尺度完全不变：logit 68 与 1000 训练行为一致（Adam 归一化 ÷std 因子无影响）
    - **验证**：verify_c25_quality_fix.py **11/11 PASS**（原逻辑梯度 1e-2 显著小于修复后 0.12（饱和）/修复后一步梯度下降 KL 减小/actual 有熵不独热/×10+500、÷10-3、+1000 三尺度 KL 值一致+梯度下降行为一致）；C25-B/C/D 无回归
    - **意义**：learned quality proxy 恢复可用（judge 不可用时回退），C24 重训完成后 C20 判定重训不再单点依赖 judge NLL

---

## 📖 接口梳理（2026-08-10，用户痛点：经常用错接口）

- **产物**：[INTERFACE_REFERENCE.md](file:///e:/taiji-neuron/INTERFACE_REFERENCE.md)（根目录接口速查与易错点手册，基于 16 模块源码实读）
- **核心发现（8 大陷阱）**：① `side_channels` 是 `excite_channels` 别名不含 inhibit；② `forward`（weighted_logits）vs `forward_train`（fused_logits）key 完全不同；③ `_parallel_forward` 返回 6 元组但 docstring 写 5；④ judge_lm_head（判定）vs lm_head（生成）双头；⑤ `get_ffn_gain==get_lr_multiplier`、`get_attention_temp_gain==get_field_write_scale` 同公式双语义；⑥ `batch_align_and_embed` 返回元数随 answer_marker 变化；⑦ `consolidate` 位置参数顺序（current_step vs stdp_tracker）易写反；⑧ `resize_embedding_for_vocab` 文档提到但不存在
- **✅ 文档级修复完成（2026-08-10，4 项）**：① `_parallel_forward` docstring 修正 5→6 元组（补 round_judge_logits）；② `forward` fusion_mode 默认值 `"per_position"`→`"soft"` 统一（与训练/cortex 对齐，无调用点依赖旧默认；per_position 降为诊断选项）；③ translator.py:383 修正不存在的 `resize_embedding_for_vocab` 引用（改 `resize_linear_for_vocab`）；④ `consolidate` 强 strong_pairs 改**精确双向强化**（pair (i,j) → 只强化 i→j 与 j→i，消除 sorted 字典序隐式约定 + 原"所有含 post_key 的 neuron"过宽误强化）。验证：C25-D 17/17、C25-B 21/21、C25-C 23/23 无回归；手册同步更新

---
