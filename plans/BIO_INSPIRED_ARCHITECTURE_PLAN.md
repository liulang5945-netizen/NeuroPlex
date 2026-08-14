# 态极生物学化架构计划 (Bio-Inspired Architecture Plan)

> 本文档记录态极架构借鉴人脑神经科学的系统性改造计划。
> 核心原则：**神经元差异性第一**、**自我进化能力**、**硬件限制不在考虑范围内**。

> **📋 项目主 plan（活跃维护）**
> 本文档是项目的**当前状态导航**（实时匹配项目），不记录实施历史。编号规范：
> - `x.y`（如 1.1/2.1）= 本文档章节号
> - `C16`-`C26` = 机制演进迭代代号（工程版本号）——**实施记录与验证见 [HISTORY_MECHANISM_EXPERIMENTS.md](file:///e:/taiji-neuron/plans/HISTORY_MECHANISM_EXPERIMENTS.md)（唯一边集）**
> - `日期` = 时间戳，仅出现于 HISTORY 系列
>
> 其他 plans/ 文件：
> - `HISTORY_MECHANISM_EXPERIMENTS.md` — **机制迭代记录（C 编号唯一边集）**，早期里程碑（EMERGE/aux-free/shared expert）
> - `HISTORY_DIALOGUE_TRAINING.md` — 对话/Standard 神经元训练历史
> - `HISTORY_PROJECT_EVENTS.md` — 项目事件与旧状态归档
> - `DESIGN_PRINCIPLES.md` — 设计原则与 Phase 1-8 历史记录
> - `TRAINING_REFERENCE.md` — 训练准则参考（非 plan）
> - `TAIJI_VS_HUMAN_BRAIN_COMPARISON.md` — 态极 vs 人脑机制详细对比（2026-08-08）
> - `archive/` — 历史归档

---

## 🗺️ 一、项目全景（2026-08-11 更新）

### 1.1 系统架构总览

```
taiji/                    核心包
├── resonance/            共振场核心：Neuron/Field/Ensemble + lifecycle(进化) + STDP + 神经调质 + gamma/phasor + 场记忆
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

**核心数据流**：神经元规格分 compact(51M)/standard(134M)，各有独立 domain lm_head + judge_lm_head（general 256K 判定头）；协作通过 per-pair side_channels（excite/inhibit）+ 跨规格投影层（field_dim→unified）+ 共振场（3072-dim）。

### 1.2 训练→推理链路与闭环状态

```
离线训练（手动脚本，已闭环）                    运行时（Cortex/API，已挂载）
base → dialogue fine-tune → cross_spec 协作层 ──► Cortex.generate（continuous 默认）
（5 对话 + 4 域，C24 双头 SFT）                 （9 阵容装配 + judge NLL 判定 5/5）

在线学习（培养期，已闭环）：Feed（喂养）→ Sleep（巩固+训练+调质+场固化）→ Wake（应用）
```

### 1.3 当前状态（2026-08-12）

| 环节 | 状态 | 关键产物 / 结果 |
|------|------|----------------|
| 装配 | ✅ 9 阵容 | 5 dialogue（zh_aug0-3 + zh_std0，51M/134M）+ 4 域（code/math/zh/en）+ collab_v3_c24v2 + PhasorDynamics 默认 |
| 生成默认路径 | ✅ continuous | C25-E 增量五：连续时间共振为默认 collab_mode（generate 默认，相位绑定驱动激活，leader 用 round1_scores × NLL 质量融合）|
| 回合级判定 | ✅ 5/5 | judge NLL 主信号（general 空间可比）+ 启发式融合；quality z-score 回退存活 |
| 记忆（C26） | ✅ 全生命周期闭环 | 写（WriteGate）→ 读（向量条件化生成，阈值已对齐精确 4/4，见 N4）→ 沉淀（LoRA）→ 跨重启保留 → sleep() 编排 → **自动检索注入对话**（增量四，产品默认）→ 生产沉淀接线（R10，2026-08-14：cortex_chat 对话后记录场快照 → 睡眠固化；此前 record_field_memory 零生产调用者）→ **跨频耦合**（增量五：记忆 entrain theta 相位对齐峰值→gamma 绑定增强"记忆注意窗"，verify_c26_cross_freq 12/12）→ **真正睡眠重放**（增量六：记忆向量场条件化 forward 重放固化"注意窗下生成"，读路径+LoRA 双训，verify_c27 12/12）→ **自组织新生**（增量七：新生 neuron 从记忆经验生长，三源样本+注意窗预训练+蒸馏降权辅助，verify_c27 10/10）|
| 培养期闭环 | ✅ 可用 | feed → sleep 训练（分层 lr 防破坏性更新）→ 影子写回 → ckpt；渐进改善已验证（held-out PPL 降 79%，口径 = 同分布列表式评估集；提问式口径仅 33%，见 N1） |
| 续训（完成） | ✅ 4000→8000 | 5 dialogue 全部完成（best PPL：compact 101-107 / std0 95.27@4000）；口径机制化后回归通过 |
| 回归测试 | ✅ 16/16 | tests/ 下 pytest 统一入口（3 文件 16 用例）：口径契约 10 + 共振 side_channel 6；requirements.txt 补 pytest |
| 遗留瓶颈 | ⚠️ zh/en 生成 | answer PPL ~70（词表大 + 响应长 + 51M 容量），域生成片段级；对话质量受欠训练限制（续训中） |

### 1.4 闭环缺口清单

| # | 类型 | 说明 |
|---|------|------|
| A-H | ✅ 已修复 | 混合规格装配 / 协作权重加载 / shared_embedding / 域路由 / fusion_mode bug / SpecSelector / diagnose_domain / Play→Coactivation 链路（详见 HISTORY C17-C24） |
| I | ✅ 已挂载 | 综合体接入聊天接口：API 装配 9 阵容（collab_v3_c24v2 + foundation_v1_dual），test_api_dialogue 对话流畅；环境变量 TAIJI_COLLAB_NAME / TAIJI_EXTRA_NEURONS_DIR 可覆盖 |
| J | ✅ 已完成 | dialogue neuron 欠训练（4000 步预算截断非收敛平台）→ 续训 4000→8000 步全部完成（compact 68.74-73.60 / std0 67.42，2026-08-14） |
| K | ✅ 已接入产品路径 | **可学习写策略（WriteGate）**：verify_c26_write_gate.py **8/8 PASS**——门控（场向量+最近邻 sim → P(值得写入)）可学习；**门控优于硬阈值**（拒绝 sim=0.9 模糊重复，硬阈值 0.92 漏判）；**已接入 sleep 场固化**（sleep_engine 自动装配 data_dir/write_gate.pt，存在即用无则回退） |
| L | ✅ 已接入产品路径 | **跨域语义对齐（锚点投影）**：verify_c26_field_alignment.py **8/8 PASS**——AnchorProjector 产品化 + cortex.set_anchor_projector 挂载；30 对双语术语训练后同义 0.932 vs 错配 0.681（+0.251）；**已接入记忆检索**（FieldMemoryBank.projector：检索在跨域语义锚点空间进行，睡眠固化自动装配 data_dir/anchor_projector.pt） |
| M | ✅ 闭环 | **多频段振荡 theta-gamma 嵌套 + 跨频耦合**：verify_c26_theta_gamma.py **9/9 PASS**（嵌套：theta_omega/theta_amp 默认 0 零回归、相位单调推进、包络周期复原、调制在 [base×(1-A), base×(1+A)]）；**verify_c26_cross_freq.py 12/12 PASS**（跨频耦合：记忆 entrain theta 相位对齐峰值→gamma 绑定增强"记忆注意窗"，接入 continuous_forward 主循环，无记忆零回归，集成 final_scores 1.686→2.023 与 theta_amp=0.2 数值吻合）|
| N | ✅ 产品闭环 | **场记忆组件接入验证**：verify_c26_field_memory_product.py **11/11 PASS**——训练产物保存 → SleepEngine 自动装配 gate/projector → 门控固化（4 写入+重复拒）→ 锚点检索 4/4 → 重启恢复（组件+记忆+检索）→ 无产物回退兼容 |

---

## 🧭 二、当前装配态与路线图

### 2.1 机制迭代索引（C16-C26，详见 HISTORY_MECHANISM_EXPERIMENTS.md）

| C | 迭代 | 状态 |
|---|------|------|
| C15-C18 | 回合级质量信号 / LoRA 保护（零破坏原则）/ 神经发生 / 客户端链路 | ✅ |
| C19 | **任务级路由**（范式转变：token 级竞争 → 回合级任务模式） | ✅ |
| C20 | 回合级监督训练（answer_mask + 同域 batch，判定 5/5） | ✅ |
| C21 | **词库多词表架构**（用户核心需求，neuron 绑定自己词表 + 词库转译） | ✅ |
| C22 | 路径收敛（默认 executive；消除双路径打架） | ✅ |
| C23 | **相位同步本体化**（缺口 R 核心：共振分/场本体/可微化/ω·K 梯度/监督纯净化/默认装配） | ✅ |
| C24 | 域目标空间 SFT + judge/域双头 + 9 阵容挂载 | ✅ |
| C25 | 对比问题解决 A-G（词库编辑/STDP 生长修剪/调质深度耦合/睡眠重放+稳态下调/连续时间共振默认）+ 培养期闭环 + zh 诊断 | ✅ |
| C26 | **场固化**（可写记忆第 0 格：睡眠沉淀 + 跨会话检索 + 注入 + 真正睡眠重放） | ✅ |
| C27 | **实例级路由**（SMCS）+ **场向量相位编码**（KoPE）+ **BioOSS p/o 双神经元**（o 型节奏源） | ✅ |

### 2.2 下一步建议（当前活跃，按优先级）

1. **✅ dialogue 续训回归（2026-08-14 完成）**：5 个 dialogue 全部完成新数据重训（compact best 68.74-73.60，std0=67.42，旧 best 95-102 → 降 27-30%）+ 三项回归全过（A/B 边际、API 无死循环、C26 11/11）；std0 补验：A/B 均长 30.0/39.2、重复 0.186/0.230、命中 0.281/0.275——**强制 134M 长度优但重复差，收益边际，leader 改进维持不采纳**
   - **T12 词表迁移**（2026-08-12）：dialogue neuron lm_head 仍 20K（08-01 训练），与当前 50K tokenizer 不匹配 → resume 崩溃 `Target 38070 out of bounds`。已用 hot_swap_vocab.py 迁移 5 个 dialogue + 5 个 base 共 10 个 ckpt 到 50K（精确 13427 + 子 piece 36573），backup 在 pre_t12_backup/
   - **C26 lr 修复（补充）**：resume 后仅改 optimizer.lr 不够——LambdaLR.step() 用 scheduler.base_lrs（旧 5e-4）覆盖 → 首日志实测 lr=5e-4（大 lr 冲击已收敛权重）。已补充 `scheduler.base_lrs = [args.lr]*n`，验证通过（lr 保持 1e-4）
   - **训练结果**（compact 4 个已完成 4000→8000）：val PPL 160→101-107（降 33%），每 1000 步稳定降 ~15，无平台；EOS 学习正常（自然停止）；生成质量验证见下
   - **⚠️ 评估口径发现（重要）**：verify_zh_leader_ab.py 用**裸 prompt**（"请介绍什么是神经网络"）评估，但 dialogue neuron 训练/产品路径（test_api_dialogue.py）都用 `问：xxx\n答：` 格式 → 裸 prompt 下 50K 模型陷入 `。`→换行→空格死循环（top1 恒为 `▁`），触发跑偏截断 → 假退化（均长 9.6）。**test_api_dialogue.py（正确格式）生成完整正常**（Q5 诗/两行、Q7 幸福/完整句）。verify_zh_leader_ab 需改为训练格式 prompt 才有效
   - **✅ 口径机制化（2026-08-12，提交 04d1ee8）**：根治（历史同类：07-31 domain/general token ID 错位、07-29 评估集分布失真——均靠人工发现）。① `experiment_config.build_dialogue_prompt()` 统一构造（"问：{q}\n答："唯一入口）；② `cortex.generate/_generate_p7` 新增**硬失败守卫**——domain=zh 且激活 dialogue neuron 时裸 prompt 直接抛 ValueError（active_nids 归一化后判断，避免 str 误伤；`_allow_plain_prompt=True` 显式放行 base/域 neuron 评估）；③ 13 个验证/诊断脚本统一改口径（verify_zh_leader_ab/diag_zh_leader/diag_zh_capacity/c25_e×3/c19/c20/c21/c26_field_memory/feed_sleep×2/sleep_learning）。验证：py_compile 全过、test_api_dialogue 8 问正常（不误伤）、裸 prompt 拦截/对话格式放行/例外放行三项全过
   - **待完成**：std0 续训完成（~15:00）+ **用修复后口径跑 verify_zh_leader_ab.py 确认假退化消失** + test_api_dialogue.py 重新评估质量基线 + C26 记忆复述回归
   - **✅ 续训全部完成（2026-08-12 18:47）**：std0 8000/8000 完成，best_val_PPL=95.27@step4000（最终 105.98，WSD 衰减后正常）。5 个 ckpt 全部保存（std0 2.97GB / compact 各 2.13GB，50K 词表规模）
   - **✅ 回归结果（2026-08-12）**：① verify_zh_leader_ab（修复后口径）：均长 9.6→30.9（假退化消失），非空 36/36，重复率 0.133/0.084，强制 134M leader 收益边际（不采纳）；② test_api_dialogue：8 问全部正常无死循环，质量仍受欠训练限制（遗留瓶颈不变）；③ C26 记忆复述：11/11 PASS（检索 4/4 命中、注入生效 4/4、跨重启恢复）
   - **✅ 口径契约可回归化（2026-08-12，提交 c0cec3b）**：根因是 87 个一次性 verify/_smoke 脚本各自为政、无回归保障。落地：① `taiji/resonance/dialogue_format.py` 作口径单一真相源（`build_dialogue_prompt` + `dialogue_prompt_requires_guard`），experiment_config 改 re-export（35 引用点零破坏），cortex 守卫改用纯函数消除漂移；② `tests/test_dialogue_format.py` 首个可回归契约（10 用例 + 核心不变量"构造产物必过守卫"）；③ pytest 入 requirements + tests/ 统一入口（16/16 通过）
2. **✅ C26 增量一：可学习写策略（2026-08-14 完成）**：轻量门控 MLP WriteGate（输入 = 场向量 + 最近邻相似度 → P(值得写入)），替代硬阈值 0.92 去重。**训练产物落盘**（train_field_memory_components.py → taiji_data/sleep_data/{write_gate,anchor_projector}.pt，回读校验通过）；**修复装配 bug**：WriteGate/AnchorProjector.load() 原不重建网络——构造维度与产物不一致（sleep_engine 无 cortex 默认 4096 vs 产物 3072）时 load_state_dict 静默失败 → 已改为按产物维度重建后加载；**验证全绿**：verify_c26_write_gate 8/8（门控判别 12/12、拒绝 0.9 模糊重复而硬阈值漏判、固化 4 新增/0 重复、检索 4/4、持久化恢复）、产品装配确认 gate+projector 均加载、C26 记忆复述 11/11、tests 16/16
3. **✅ C26 增量二：记忆可读进生成（2026-08-14 完成）**：检索到的记忆向量（统一场空间快照）**直接写入共振场**做生成条件化——写入点在 round1 判定信号之后（判定保持"无记忆的天然反应"，C23 安全边界），round2+ 的场条件化 forward 让记忆通过**已训练的 field_state 注入路径**直接参与 token 生成（"读"免训练——神经元 forward_train 即用 field_conditioning 训练过该路径）。写入权重 = 检索相似度（近记忆强条件化）。实现：`FieldMemoryBank.retrieve_vectors()`（返回 (label, sim, vector)）+ ensemble `forward/continuous_forward(seed_memories)` + cortex `generate/_generate_p7/think(memory_vectors)` 透传。**验证 verify_c26_memory_read_gen.py 10/10**：检索升级 4/4（label+sim+vec 同 dim）、安全边界（round1_scores 容差内不被记忆污染）、场拉拽 4/4（cos 0.39-0.57→0.65-0.81）、leader 场条件化 logits 因记忆改变 4/4（硬）、向量通道单独注入改变生成输出 4/4（区别于文本通道）、文本通道回归 4/4、跨重启恢复；tests 16/16 全过
4. **✅ C26 增量三：记忆→突触沉淀（2026-08-14 完成）**：海马→皮层两层记忆——高频场记忆（access_count≥2）经睡眠重放**沉淀进神经元权重**（LoRA 增量，enable_lora 冻结 body 只训尾层低秩，B 初始 0 零破坏起点，避免培养期"直接微调 lm_head/embedding 灾难性遗忘"同款教训）。实现：`FieldMemoryBank` 加访问计数 + 已沉淀标记 + 内容文本（frequent_entries/mark_consolidated）；`SleepEngine` 新增 **Phase 1.6 突触沉淀**（影子权重 COW clone → shadow 重建 lora_adapters → 训后只写回 lora 参数 → 标记条目防重复重放）；样本 = 问答对 + 原文混合（用户决策），域内全部 dialogue neuron 协作（rank=16 尾层 2 层）。**验证 verify_c26_synaptic_consolidation.py 10/10**：高频 3 条沉淀/低频 1 条不沉淀、沉淀后记忆文本 NLL 降 0.15-0.19（LoRA 记住）、LoRA B 非零写回 5/5、零破坏（未沉淀文本 NLL 不暴涨）、二次沉淀跳过、重启恢复标记；tests 16/16 + 增量二回归 10/10
   - **✅ 跨重启保留补（2026-08-14）**：增量三原验证只覆盖 session 内写回 live——enable_lora 是运行时方法（不写 config），装配重建的 neuron 无 lora_adapters，strict=False 加载静默丢弃 lora keys → 皮层记忆重启即失。**修复**：`neuron.load_lora(sd)`（rank 从 a.weight 推断 + 恢复），三处装配路径接入（Cortex 主加载 / extra neurons / revive）。**验证 verify_c26_lora_persist.py 8/8**：沉淀→保存 ckpt（含 8 lora keys）→ 重启装配恢复（adapters 非空 + B 非零）→ **重启后记忆文本 NLL 完全保留（rebound=0.0）**；无 lora 普通 ckpt 不误触发
   - **✅ sleep() 端到端收尾（2026-08-14）**：三格此前均单 phase 验证，本验证走**完整 sleep() 主流程**（record → sleep → 场固化 → 会话检索 → sleep → 突触沉淀 → LoRA 写回 → 记忆条件化生成回归；SleepConfig(training_enabled=False) 跳过无关 Phase 2）。**verify_c26_sleep_e2e.py 14/14**：编排完整（memory/field/synaptic/knowledge_integration/knowledge_distillation/evaluation/recursive_improvement 7 phase 全挂载）、#1 固化 4 条+沉淀 0（无高频候选）、#2 沉淀 3 条（低频 1 条不沉淀）、LoRA B 非零 5/5、consolidated 标记持久化、向量通道生成非空 3/3；tests 16/16
   - **✅ 增量四：记忆自动检索注入对话（2026-08-14 完成）**：此前记忆是显式 API（调用方手动传 memory_vectors）。增量四让 generate **自动检索**——未显式传向量且注入过记忆库时，用 prompt 场状态自动检索 top-1 记忆注入生成（Titans 式内部记忆的产品化落点）。实现：`cortex.set_field_memory(bank)` + `generate/_generate_p7(auto_memory=True 默认)`（检索一次额外共振前向，失败静默跳过）；`sleep_engine.set_brain_interfaces` 装配时自动注入记忆库（**产品默认接入**，assemble_cortex Step 9 即生效）。**验证 verify_c26_auto_memory.py 7/7**：装配即注入、自动检索触发（access_count 证据，total 0→4）、auto 开/关生成不同 4/4、显式传向量时自动检索跳过、无记忆库静默跳过；tests 16/16 + 增量二回归 10/10
   - **✅ 增量五：跨频耦合（记忆驱动的 theta-gamma，2026-08-14 完成）**：嵌套机制（缺口 M）此前仅单元验证——theta_modulate 是死代码未接入 continuous_forward 主循环，theta 相位无驱动源。增量五把 theta-gamma 从"可验证机制"变成"记忆生命周期的一环"：记忆注入（seed_memories）时 `entrain_memory()` 将 theta 相位对齐峰值 → gamma 激活经 `theta_modulate` 增强（**"记忆注意窗"**：检索到的记忆带动相关回路同步激活，跨频耦合 = 慢 theta 相位 entrain 慢变量驱动快 gamma 绑定）；返回前 `reset_entrain()` 防跨 token 泄漏。**实现**：continuous.py `entrain_memory/reset_entrain` + theta_phase_at/envelope 记忆语义 + 无嵌套默认恒等零回归；ensemble.py continuous_forward 三处接入（entrain / t=0 与每步 theta_modulate / reset）。**验证 verify_c26_cross_freq.py 12/12**：单元 8（默认恒等零回归、entrain 峰值 1+amp 相位恒 0、reset 恢复、显式嵌套振荡/半周期谷值/entrain 锁定）+ 集成 3（记忆窗口放大 final_scores 3/3：**1.686→2.023 ≈ ×1.2 与 theta_amp=0.2 数值精准吻合**、无记忆激活正常）+ 行为 1（记忆条件化生成非空 3/3）；tests 16/16 + 增量二回归 10/10（R6 阈值 4/4 对齐后仍全过）
   - **✅ 增量六：真正睡眠重放（记忆向量场条件化 forward 重放，2026-08-14 完成）**：增量三（Phase 1.6）只把记忆文本做**无场条件化**的纯文本 SFT 进 LoRA（round1, field_state=None）——神经元"记住内容"，但记忆向量从未参与条件化；推理的记忆注意窗（round2+ 场条件化 + 增量五 theta entrain）依赖**随机初始化的 field_read_layers**（R2 审计发现：全仓库无训练路径）。增量六让睡眠重放真正驱动 forward：以记忆向量/白天场状态作 field_state（round2+ 读路径），重放记忆文本与触发文本——**把"记忆注意窗下如何生成"固化为可学习权重**（读路径 field_read_layers/gate + LoRA 双训，用户决策；样本源 = 已沉淀记忆 + 场状态混合，用户决策）。**实现**：`record_high_resonance_state` 增 text 参数（play_engine 记录触发文本）；SleepEngine 新增 **Phase 1.7** `_sleep_phase_forward_replay`（影子 COW + back-projector 投影记忆向量到 neuron.field_dim + 只写回读路径/LoRA，body 不进 optimizer 零破坏）。**验证 verify_c27_forward_replay.py 12/12**：A 重放 5 neuron 全完成、E 场状态样本被消费（loss 记录）、C 读路径 delta=0.018 已学习、**B 条件化 NLL 下降 0.46-0.64（硬，记忆注意窗下生成被固化）**、D 零破坏（round1 对照不暴涨）、F 持久化（读路径+LoRA 8 keys 随 state_dict 保存，重建恢复一致）；tests 16/16 + 增量五回归 12/12
   - **✅ 增量七：自组织新生（从经验生长，非 teacher 蒸馏，2026-08-14 完成）**：审计 S9 缺口"新生依赖外部 teacher → 自组织新生"。现有 IntegrateEngine（C17）新生整合只依赖 FeedEngine 样本——**feed 为空时新生直接 skipped**（no_feed_samples），未利用 C26 记忆库积累的经验。增量七让新生 neuron **从记忆经验生长**：① `_memory_pretrain` 记忆注意窗预训练（记忆向量作 field_state 的 round2+ 场条件化 forward，读路径+LoRA 双训——从经验出生而非 teacher）；② 样本源三路混合（用户决策：feed + 记忆问答对 + 记忆原文）；③ 邻居蒸馏保留降权 0.3 辅助（用户决策：记忆生长为主，蒸馏辅助融入共振场）。**修复两处运行时缺口**：cortex.add_neuron 不注册相位 → PhasorDynamics.add_neuron 动态追加（continuous_forward binding_tensor 维度错配 9 vs 10 崩溃）；cortex.set_tokenizer_hub 不转发 ensemble（integrate 的 forward_train 跨 vocab 训练缺 hub 报错）。**验证 verify_c27_self_organize.py 10/10**：A feed 空 + 记忆经验 → 新生不 skipped（从经验生长）、B 读路径 delta=0.0235 已学习、C 条件化 NLL 9.383 ≤ 无条件化 9.415、D DISTILL_WEIGHT==0.3、E 零破坏（父本 delta=0）、F 读路径+LoRA 持久化；tests 16/16 + 增量六回归 12/12
   - **✅ 增量八：多阶段任务模式链 v2（TaskSet 序列，2026-08-14 完成）**：C25-F 首步 generate_staged（dict 阶段 + {prev} 文本传递）被 R17 标注死代码（生产 generate 路径不消费）。增量八升级为 v2：① **TaskSet dataclass**（prompt/mode/domain/active_nids/max_tokens/temperature/quality_gate/record_memory/memory_label，用户决策"TaskSet 类+调度器"）——显式激活子集 = 任务集切换；② **三重阶段间传递**（用户决策）：文本 {prev} 模板填充 + 场状态（prev_fs → 下阶段 seed_memories 记忆注意窗，权重 0.8）+ 记忆写入（record_memory → sleep_engine.record_field_memory 睡眠固化候选）；③ **阶段质量门**：_is_degenerate_text 检测 → 高温重试 min(temp+0.15, 1.2) → 仍退化隔离（阶段互不污染，gate 记 ok/retried/degenerate）；④ **生产接入**（用户决策）：cortex.generate_task_chain 调度器 + /api/taiji/cortex/task_chain API 端点；generate_staged 转 C25-F 兼容层（dict → TaskSet 转发）；_select_best_candidate R17 死代码标记更正（核实为 generate(n_candidates>1) SMCS EPE 活跃生产代码）。**验证 verify_c27_task_chain.py 13/13**：A TaskSet 对象化 2/2、B 三重传递 5/5（三阶段输出全非空、场状态截获 (3072,)×3、record_memory 写 gate=recorded、记忆库 pending 收到"任务链阶段3"、seed_memories 改变生成）、C 质量门 2/2、D 兼容层 1/1、E 生产接入 2/2；tests 16/16 + 增量六回归 12/12 + 增量七回归 10/10
   - **✅ C27 增量一：实例级路由 + 混合后验（SMCS 借鉴，2026-08-14 完成）**：现有路由 = 回合级任务判定（C19/C22：dominant 域激活集回合内静态）+ C25-E leader 质量融合（共振分 + **prompt 一次性** NLL）。SMCS 的 contextual selection 在实例内重新选 expert 子集——增量一让 continuous 生成中激活子集按 **chunk 级混合后验**（共振分 + **已生成文本滚动 NLL**）双向域内演化（用户决策：chunk 级每 8-16 token / 双向域内演化+迟滞 / continuous 默认路径开关默认开）。**实现**：`_rolling_nll_quality`（round1_logits 尾部窗口续写 NLL，零额外前向）；`_probe_inactive_fused`（未激活同域 neuron 的 chunk 边界轻量 probe，ensemble.forward 独立共振场不污染 cortex 场）；`_instance_route_evolve`（剔除：后验 < evict_ratio×leader 且连续 evict_streak chunk，迟滞防抖；加入：后验 > 激活集最小×add_ratio；保护：同域激活 ≥ min_active、general 恒激活、域外不动）；`_generate_p7` chunk 边界评估点 + `generate(instance_routing=True)` 透传。**验证 verify_c27_instance_routing.py 14/14**：A 滚动后验合法 2/2（真实 round1_logits 6 neuron 有限值、空文本安全）、B 演化单元 6/6（B1 迟滞单次不剔/连续剔、B2 双向加入、B3 域内约束跨域不动、B4 min_active 保护 ≥2、B5 稳定不变）、C 集成长生成 3/3（48 token 触发 3 次 evolve、非空、不退化）、D 开关回归 2/2（关闭回退 C25-E 不调用）；tests 16/16 + 增量六回归 12/12 + 增量七回归 10/10 + 增量八回归 13/13
   - **✅ C27 增量二：场向量相位编码（KoPE，2026-08-14 完成）**：相位此前只驱动"激活强度/融合权重"（纯动力学，C23/C25-E），记忆/路由读到的 field_state 无相位语义。增量二把相位编码为**显式表征**（用户决策：result 附加字段 + 记忆条目扩展，不改 field_state 维度契约），并让记忆注入**按记忆相位对齐 theta**（用户决策：相位归属记忆，不同记忆不同相位唤醒，θ 相位序列编码）。**实现**：`continuous.py` entrain_memory(target_phase) + `_entrain_phase` + theta_phase_at 目标相位（默认 0 = 峰值，增量五零回归）；`ensemble._encode_phase_code`（phase_code [2N] 全量相位分布 + phase_mean 加权均值相角 + phase_lock 锁相度）接入 forward/continuous_forward 输出，seed_memories 支持 3 元组 (vec, w, phase)；`field_memory.py` entry["phase"] + consolidate(phases) + retrieve_with_phase（retrieve_vectors 改薄封装，旧调用方零破坏）；`cortex.py` think 3 元组/dict phase + auto_memory 相位检索注入 + get_last_phase 截获 + task_chain record 带相位；`sleep_engine.py` record_field_memory(phase) + pending 4 元组（兼容旧 3 元组）。**验证 verify_c27_kope.py 13/13**：A 相位编码 2/2（continuous code_dim=12 [2×6] / 离散 6、phase_mean=1.48、lock∈[0,1]）、B 记忆带相位 3/3（pending 4 元组、entry phase=0.7、retrieve_with_phase/vectors 兼容）、C 相位注入 4/4（entrain 目标相位 0.7、reset 恢复、spy 捕获 seed_memories 3 元组→0.7、2 元组回退 0 零回归）、D 生产 3/3（generate 3 元组非空、get_last_phase=2.18、task_chain 记忆带相位 -0.67）；回归 tests 16/16 + 增量一 14/14 + 增量六 12/12 + 增量七 10/10 + 增量八 13/13 + C26 自动记忆 7/7（retrieve_vectors 兼容）
   - **✅ C27 增量三：BioOSS p/o 双神经元模型（2026-08-14 完成）**：态极 neuron 已有 excitatory/inhibitory 亚型（单维标记），增量三把角色分工正式化——**p（投射型，内容生成，现有全部 neuron）+ o（振荡型，节奏源，轻量合成 OscillatorNode 无需训练 ckpt，用户决策）**。o 型三职责：① 相位推进（theta 慢 ω=0.5 + gamma 快 ω=π/4 双层，用户决策）；② p 型 Kuramoto 相位牵引（dtheta += K·sin(θ_osc−θ_i)，"o 型驱动 p 型锁相"）；③ **GABA 式节奏门控**（write_inhibit 半周期窗口 max(0,cosθ)，时间门控而非内容污染，用户决策）。**实现**：`oscillator.py` OscillatorNode（step/unit/gaba_gate/gaba_vec 门控方向）+ make_default_oscillators；`phasor.py` evolve/kuramoto_step 支持 external_phases/external_weights 外部牵引；`ensemble.py` set_oscillators + continuous_forward 每步 osc.step(ct.dt) → 相位牵引 + GABA write_inhibit min(gaba_amp·gate,1.0) + `_encode_phase_code` 追加振荡段 [2M]（节奏中心 1:1 融合 phase_mean/lock，phase_code=2N+2M）；`loader.py` 装配注入双层振荡节点（失败非致命 warning）。**验证 verify_c27_biooss.py 14/14**：A 单元 3/3（相位推进/unit/GABA π 关闭）、B 装配 2/2（双层 + gaba_vec=3072）、C 牵引 2/2（diff=0.364 方向正确）、D 门控 2/2（mask 0.988 温和）、E KoPE 节奏中心 2/2（16=2×6+2×2）、F 生产 2/2（生成非空 + last_phase）；回归 tests 16/16 + 增量一 14/14 + 增量二 13/13 + 增量六 12/12 + 增量七 10/10 + 增量八 13/13 + C26 自动记忆 7/7
   - **✅ C27 三增量联合端到端验证（2026-08-14 完成）**：三机制在**真实长生成中同时开启**（SMCS instance_routing 默认开 × KoPE 相位编码 × BioOSS 振荡节点），验证协同不退化。**验证 verify_c27_joint.py 16/16**：A 联合装配 3/3（振荡双层 + gamma + instance_routing 默认 True）、B KoPE 联合 3/3（phase_code 16=2N+2M 维度不退化、phase_mean 有限、lock=0.918∈[0,1]）、C 记忆×振荡 4/4（带相位记忆 3 元组 → entrain 0.7 按记忆相位对齐、`__memory_0__` 写入共振场、场状态非零、GABA mask 0.989 温和**不抑制记忆写路径**）、D 路由×振荡 4/4（48 token 长生成非空不退化、chunk 边界触发 evolve=2、振荡相位生成期间推进 0.375→4.5、last_phase 正常）、E 三轮生成稳定性 3/3（非空不退化 + 相位有限）；回归 verify_c27_biooss 14/14 + kope 13/13 + instance_routing 14/14
   - **✅ C27 增量四：o 型振荡节点 → 可学习节奏控制器（2026-08-14 完成）**：增量三的 OscillatorNode 是固定常量节奏源（纯 float），forward_train continuous 训练路径未消费振荡器——ω/coupling/gaba_amp 无梯度。增量四让 o 型从固定节奏源走向**可学习节奏控制器**（用户决策：三参数全部进梯度流 / 牵引+门控全链路 / **节奏对齐自监督 osc_rhythm_loss 作 gaba_amp 梯度源**——C23-C4 监督纯净下主 NLL 不触达门控，锁相强→弱抑制、发散→强抑制）。**实现**：`oscillator.py` OscillatorNode 升级 nn.Module（omega/coupling/gaba_amp 三 Parameter + gaba_vec buffer + 可微相位 API theta_tensor/phase_unit_tensor/gaba_gate_tensor，推理 float 路径零回归）；`phasor.py` evolve external_weights 支持张量（不再 float() 截断梯度，coupling 可微）；`ensemble.py` forward_train continuous 分支每步可微牵引（external_phases 张量 + external_weights=coupling Parameter）+ GABA 门控衰减 field_state（与推理 write_inhibit 同公式 mask*=1-w·|v_abs|）+ 融合段 osc_rhythm_loss（w=gaba_amp·gate 对齐 1-sigmoid(bvec·4)）+ result 新键；`cortex.py` save_state/load_state 持久化振荡器参数（跨会话节奏连续）。**验证 verify_c27_osc_train.py 20/20**：A 单元 6/6（3 Parameter+buffer、float 兼容 step/unit/gate、theta_tensor dθ/dω=t、gate_tensor 可微）、B 牵引可微 2/2（coupling/ω 经 evolve 梯度非空）、C 训练接入 6/6（phase_unit_tensor 16 次 [8步×2osc]、osc_rhythm_loss=0.203、omega/coupling/gaba_amp 梯度全非空、coupling 以可微张量传入 evolve）、D 持久化 3/3（state_dict 往返 + cortex 集成 + 双层兼容）、E 生产 2/2（生成非空 + last_phase）；回归 verify_c27_biooss 14/14 + joint 16/16（推理路径零回归）
   - **✅ C27 增量五：振荡器节奏训练接入睡眠重放（Phase 1.8，2026-08-14 完成）**：增量四打通振荡器梯度路径（三参数可微 + osc_rhythm_loss），但尚无训练脚本实际更新参数。增量五在 sleep_engine 新增 **Phase 1.8**：样本源与 Phase 1.7 同口径（已沉淀记忆 + 场状态重放文本混合），continuous 模式 forward_train，loss = osc_rhythm_loss + phase_loss（C23-C4 监督纯净：节奏梯度源独立，主 NLL 不触达门控），optimizer 只含振荡器参数（内容层由 1.6/1.7 学习，节奏独立分层），关收敛提前 break（min_steps 拉大）保证牵引梯度稳定，训练后参数随 cortex.save_state 持久化（增量四已接入）。**实现**：`sleep_engine.py` `_sleep_phase_osc_train` + sleep() 主流程 Phase 1.8 插入（1.7 后）+ SleepReport 加 osc_trained/osc_train_loss 字段。**验证 verify_c27_osc_sleep.py 10/10**：A 端到端训练 4/4（osc_trained=2、osc_train_loss=0.307 有限、三参数实际更新 omega 0.5→0.498 / gaba_amp 0.08→0.082、内容层零破坏 neuron 参数不变）、B 无振荡器静默跳过 1/1（osc_trained=0 零回归）、C 生产零回归 2/2（训练后生成非空不退化 + 相位推进兼容）；回归 verify_c27_biooss 14/14（推理路径零回归）
   - **✅ C27 增量五 sleep() 端到端（Phase 1.8 主流程共存，2026-08-14 完成）**：单 phase 验证通过后补完整 sleep() 主流程回归，确认增量五插入的 Phase 1.8 与既有睡眠链路（1.5 场固化 / 1.6 突触沉淀 / 1.7 前向重放 / 2 训练(跳过) / 3 知识整合 / 3.5 蒸馏 / 4 评估 / 5 递归改进）真实共存。**验证 verify_c27_osc_sleep_e2e.py 12/12**：A sleep #1 完整编排（phases_completed 9 项含 osc_train）、B #1 振荡器训练生效（osc_trained=2、omega 0.5→0.497）、C #1 内容层零破坏、D #2 沉淀 3 条 + 振荡器再训练（loss=0.294）+ 参数连续演化（0.497→0.494，连续学习跨睡眠）、E 生产零回归（两轮睡眠后生成非空不退化 + 相位兼容）
   - **hub neuron（缺口 L）设计启动（2026-08-14）**：参考人脑联合皮层（草案 [HUB_NEURON_DESIGN.md](file:///e:/taiji-neuron/plans/archive/HUB_NEURON_DESIGN.md) 已定上限优先 4 决策）：① hub 有 lm_head（general 256K——可生成/可评估/general fallback，符合联合皮层"能产生内部表征"）；② expert 规格 ~300M（hidden=1024，联合皮层容量最大，上限对比 2.6× 参数）；③ hub-and-spoke + 同域全连接 + hub 域内增强（保护 zh EMERGE 30.5% + 跨域只经 hub 中转避免语义噪声）；④ CE + 跨域对比 loss（同义跨域对 cosine 最大化/不同义最小化，**分阶段落地：先 hub 锚定 loss 后叠加对比 loss**——最终上限相同、渐进降低风险）。**当前架构适配确认**：general vocab 实际 256000 ✓（sp_general.model）、shared_embedding 256K×512 ✓、field_dim=3072（草案 4096，expert 规格需适配跨规格投影）、C27 机制已接入（hub always-active 需与实例路由/相位机制接线）。**✅ 阶段 1 跨域平行语料构建完成（2026-08-14）**：新增 [build_cross_domain_corpus.py](file:///e:/taiji-neuron/scripts/data_prep/build_cross_domain_corpus.py) 从 alpaca-zh（48818 条）自动提取含代码块样本 → 过滤（代码块 ≥15 字符 / 中文指令 ≥4 字 / 去重）→ **1629 对 zh↔code 同义对**（中文指令↔代码实现天然同义，质量抽样确认：factorial/链表反转/LIS 等）→ 产物 [data/cross_domain_pairs.jsonl](file:///e:/taiji-neuron/data/cross_domain_pairs.jsonl)（对比 loss 地基）
5. **zh 对话数据主线**：C24 dialogue 数据扩充重训（zh_aug*/zh_std0 共用瓶颈，对话级数据直接提升生成）
   - **✅ 数据扩充（2026-08-12，提交 84c2e9a）**：发现 sft_shared_core/unique 与 alpaca **100% 重复**（实际唯一仅 44.4K 条，数据/参数比远低于预期 → 直接解释质量瓶颈）。新增 build_dialogue_extended.py 从 BelleGroup/train_2M_CN 下载 150K → 去重/清洗后 +123K 唯一 → 总 167K 条（3.8×）
   - **⚠️ 中断事件（2026-08-12 20:15）**：软件更新终止全部训练进程（aug0 step200、其余加载中，eval_every=1000 → 零 ckpt 保存、白跑）。**改进（提交 15e4509）**：eval_every 默认 1000→500（中断最多丢 500 步 ≈1h）。20:21 已重启 5 进程
   - **✅ 重训完成（2026-08-14 03:19）**：compact 4 个全部 8000/8000（best val PPL：aug2=68.74 / aug3=71.81 / aug0=72.03 / aug1=73.60）；std0 8000/8000 完成 best_val_PPL=**67.42**（134M 上限收益显现）。5 个 ckpt 全部落盘 data/neurons/neuron_zh_*_dialogue.pt，旧数据 best 95-102 → 降 27-30%
   - **✅ 新数据回归（2026-08-13/14）**：① verify_zh_leader_ab（修复后口径）：8-13 均长 35.2/23.4（A 胜 10/12）；8-14 std0 升级后复测均长 30.0/39.2（B 胜 7）、重复 0.186/0.230（A 优）、命中 0.281/0.275（A 微优）——**两轮均判定收益边际，leader 信号改进维持不采纳**；② test_api_dialogue：8 问无死循环、假退化彻底消失，闲聊级正常，知识答问仍受 51M/134M 参数规模限制；③ C26 记忆复述：11/11 PASS
   - **✅ 连续默认化回归（2026-08-14）**：generate 默认 continuous（line 936）跑产品路径 test_api_dialogue 8 问全非空无死循环（Q1/Q3/Q7 正常）；A/B（verify_c25_e_collab_ab 修复口径后）**20/20 PASS**——executive/continuous 判定一致、5 域生成全非空、leader 域合理。**连续默认路径稳定，无需回退**
   - **✅ 融合后质量基线（2026-08-14）**：verify_zh_leader_ab（融合状态）：A 均长 31.6/重复 0.163/命中 0.270，非空 36/36；A vs B 从 7:2 收敛到 3:4:5——**leader 融合无回归且轻微改善**（重复率 0.186→0.163，当前机制不再系统性落后 134M）。test_api_dialogue 8 问全非空无死循环
6. **✅ C25-E 遗留：continuous leader 融合质量信号（2026-08-14 完成）**：诊断（diag_c25e_leader_quality_gap.py）证实**弱 neuron 独占真实存在**——aug2 场共振分系统性碾压（0.7-0.93 vs 0.01-0.17）当选 leader 5/7 次，但其生成质量（zh lm_head NLL）常是 5 个 dialogue 中最差（leader 恰为 NLL 最优仅 1/7，Spearman=-0.171）。**修复**：continuous leader 融合 = 域内归一化共振分 × 质量信号（-NLL，round1_logits 零额外前向）等权求和（`_fuse_leader_quality` + `_nll_quality_from_round1_logits`，质量信号首算缓存避免每 token 重复 softmax）。**回归 verify_c25_e_leader_fusion 3/3**：leader NLL 质量位次均值 2.12→1.12（「推荐一本好书」4→0、「怎么学好英语」4→1，aug2 不再独霸），连续生成非空率 8/8 不降，tests 16/16 全过
7. ~~缺口 L 落地：场级锚点投影正式化~~ ✅ 已完成（AnchorProjector + WriteGate + theta-gamma 三件套 + 产品闭环，见缺口清单 K/L/M/N）
8. ~~锚点投影/写门控进装配~~ ✅ 已完成（train_field_memory_components.py 训练产物 → sleep_engine 场固化自动装配）
9. **对话数据扩充重训**（zh_aug*/zh_std0 主线，续训完成后）：C24 dialogue 数据扩充 → 重跑 finetune_neuron_dialogue
10. **✅ 项目整理（2026-08-12，提交 a4064f9）**：① 数据层清理 13.71GB 废弃产物（foundation_v1_general_smoke 7.4G + foundation_v1_sft 3G + verify_v3 2.3G + neurons_backup_3000step 1.2G + verify_v3_full + 空目录×4；distill 因 experiment_config DATA_DIR 活跃引用保留）；② 脚本层归档 103 个一次性 verify/_smoke/diag 到 scripts/archive/（git mv 保留历史），scripts/training/ 收敛 149→46 主训练脚本；③ 测试层 pytest 入 requirements + tests/ 统一入口 16/16。回归：tests 16/16 通过、核心 import 正常

### 2.3 中期：跨域协作（上限优先版）

- **hub neuron 设计与实现**（缺口 L，进行中：阶段 1 语料已构建 1629 对）：参考人脑联合皮层，expert 规格 + 256K lm_head + hub-and-spoke 拓扑 + 跨域对比 loss。详见 [HUB_NEURON_DESIGN.md](file:///e:/taiji-neuron/plans/archive/HUB_NEURON_DESIGN.md)
- 多任务 loss（缺口 P）：SFT masking + margin ranking + diversity + 对比 loss
- 推理路径优化（缺口 Q）：域 token 对齐表 + 长上下文 + 多轮对话状态

### 2.4 远期：生物学机制深化（缺口 R 剩余）

- ~~记忆向量直接条件化 leader 生成（打通"场影响生成"这条遗留路径）~~ ✅ 已落地（C26 增量二 2026-08-14：检索向量经共振场写入，round2+ 场条件化 forward 直接参与 token 生成，verify 10/10）
- ~~记忆 → 突触沉淀：高频场模式写入神经元权重（LoRA 增量），海马→皮层两层记忆~~ ✅ 已落地（C26 增量三 2026-08-14：高频记忆睡眠重放沉淀为 LoRA，verify 10/10）
- ~~多频段振荡（theta-gamma 嵌套）+ 跨频耦合~~ ✅ 已闭环（2026-08-14 增量五：记忆 entrain theta 相位→gamma 注意窗，verify_c26_cross_freq 12/12，接入 continuous_forward 主循环）
- ~~真正睡眠重放（forward 重放 + 经验回放训练）~~ ✅ 已落地（C26 增量六 2026-08-14：记忆向量场条件化 forward 重放，读路径+LoRA 双训，verify_c27_forward_replay 12/12）
- ~~自组织新生（从经验生长，非 teacher 蒸馏）~~ ✅ 已落地（C26 增量七 2026-08-14：记忆注意窗预训练 + feed/问答对/原文三源混合 + 邻居蒸馏降权 0.3，verify_c27_self_organize 10/10；修复 add_neuron 相位注册 + ensemble hub 转发两处运行时缺口）
- ~~多阶段任务模式链 v2（task-set 序列完整版，C25-F 已落地首步）~~ ✅ 已闭环（C26 增量八 2026-08-14：TaskSet 类 + 调度器 generate_task_chain，三重传递 + 阶段质量门 + 生产接入，verify_c27_task_chain 13/13）
- ~~实例级路由（SMCS contextual selection：激活子集随实例内容演化）~~ ✅ 已落地（C27 增量一 2026-08-14：chunk 级混合后验双向域内演化+迟滞，continuous 默认路径，verify_c27_instance_routing 14/14）
- ~~场向量相位编码（KoPE：相位编码进表征，相位归属记忆）~~ ✅ 已落地（C27 增量二 2026-08-14：phase_code/phase_mean/phase_lock + 记忆条目相位 + 按记忆相位对齐 theta，verify_c27_kope 13/13）
- ~~BioOSS p/o 双神经元模型（投射型内容 + 振荡型节奏源分工）~~ ✅ 已落地（C27 增量三 2026-08-14：OscillatorNode theta/gamma 双层 + Kuramoto 牵引 + GABA 节奏门控 + KoPE 节奏中心，verify_c27_biooss 14/14）

### 2.5 架构方向记录（2026-08-11 讨论，未实施）

- **态极定位**：不是"更好的 Transformer"，而是"Transformer 组件化 + 生命周期闭环"的工程先行者
- **可写记忆路径**：共振场已踩在"可写记忆"门槛上（推理时可写共享态），C26 补齐固化/检索/注入；完善后**在记忆生命周期维度可超越 Titans**（睡眠巩固/记忆→突触沉淀/群体协作是 Titans 缺失维度），最大差距 = 可学习写策略
- 详见 [TAIJI_VS_HUMAN_BRAIN_COMPARISON.md](file:///e:/taiji-neuron/plans/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md)

---

## 📖 三、接口梳理（2026-08-10）

- **产物**：[INTERFACE_REFERENCE.md](file:///e:/taiji-neuron/INTERFACE_REFERENCE.md)（根目录接口速查与易错点手册）
- **8 大陷阱**：① `side_channels` 是 `excite_channels` 别名不含 inhibit；② `forward`（weighted_logits）vs `forward_train`（fused_logits）key 完全不同；③ `_parallel_forward` 返回 6 元组但 docstring 写 5；④ judge_lm_head（判定）vs lm_head（生成）双头；⑤ `get_ffn_gain==get_lr_multiplier`、`get_attention_temp_gain==get_field_write_scale` 同公式双语义；⑥ `batch_align_and_embed` 返回元数随 answer_marker 变化；⑦ `consolidate` 位置参数顺序易写反；⑧ `resize_embedding_for_vocab` 文档提到但不存在
- **✅ 文档级修复完成（4 项）**：`_parallel_forward` docstring 5→6；`forward` fusion_mode 默认统一 "soft"；translator 修正不存在引用；`consolidate` 强 pair 精确双向强化。验证 C25-B/C/D 无回归
