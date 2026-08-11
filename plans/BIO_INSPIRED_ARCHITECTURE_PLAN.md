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
  - **✅ judge 对角验证（训练前后 general 判定 NLL 对比）**：**3/4 保留，zh 例外**——
    - code neuron：前 code=1.1/math=8.0/zh=15.8/en=5.3 → 后 code=1.0/math=6.5/zh=14.0/en=4.2 ✅（code 最低）
    - math neuron：前 code=9.7/math=3.1/zh=16.2/en=5.9 → 后 code=8.5/math=2.6/zh=15.4/en=6.3 ✅（math 最低）
    - zh neuron：前 code=10.0/math=8.9/zh=11.2/en=7.1 → 后 code=7.8/math=6.9/zh=7.4/en=6.2 ❌（**en=6.2 < zh=7.4，训练前后均非对角**——zh 基座 general 空间中文 NLL 天然偏高（C24v1 已记录），本次重训 gen_loss 未保护住）
    - en neuron：前 code=6.5/math=5.3/zh=17.6/en=2.2 → 后 code=2.5/math=4.2/zh=15.0/en=2.1 ✅（en 最低）
    - **风险**：judge NLL 是 executive 判定主信号（C20v2），zh neuron 非对角 → 中文回合存在被误判为 en 的风险（端到端 5/5 需重验）
  - **✅ C24 验证使命完成（2026-08-11 08:01，定时任务共运行 3 次，结论一致收敛）**：生成碎片→片段级改善、judge 对角 3/4 保留（zh 例外）均已在计划记录。**定时任务建议暂停/删除（使命已完成）**；若需复验生成，手动运行 `diag_c24_domain_generate.py --dir data/foundation_v1_dual` 即可。
  - **⏳ 待办（下一步候选）**：① 修复 zh 域判定对角（zh SFT 训练扰动 general 空间，可调低 zh SFT loss 权重/提高 gen_loss 权重，或先清洗 alpaca-zh 噪声数据）+ 重验端到端判定；② 提升 zh/en 生成质量（answer PPL ~70 远高于 code/math，SFT 数据/训练配置待调优）。
  - **zh_general 残留收敛（2026-08-10，用户确认 9 阵容）**：9 = 5 对话（zh_aug0-3_dialogue + zh_std0_dialogue）+ 4 域（code/math/zh/en）。查证：zh_general 设计为 SHARED_EXPERT_ID（experiment_config），但 **assemble_cortex/cortex 从未传 shared_expert_id → shared_expert 机制从未启用**；实际被 cortex 全量扫描误加载为普通 neuron（中文任务竞争者、训练最弱 PPL 257，verify_c19 注释"排除 zh_general 旧产物干扰"）。C24 双头后每 neuron 自带 judge_lm_head，single always-active 底座机制冗余 → **删除 data/neurons/neuron_zh_general.pt**，verify_hotswap_integration 改用 zh_std0_dialogue，experiment_config SHARED_EXPERT_ID 废弃注释。装配收敛为 9 阵容。

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
  - C25-C：神经调质深度耦合训练（当前仅状态记录，缺口 R）
  - C25-E：连续时间动力学替代离散共振轮次（相位同步本体化剩余）
  - C25-F：多阶段任务模式链（task-set 序列，对比文档 v2 项）
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
