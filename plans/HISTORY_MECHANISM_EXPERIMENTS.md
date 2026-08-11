# 机制迭代与实验记录（Mechanism Experiments & Iteration History）

> **归档文档**（2026-08-11 重组织）：**C 编号 = 机制演进迭代代号（工程版本号）**，按时间顺序递增。
> 本文档是 C 编号记录的**唯一边集**；主 plan（[BIO_INSPIRED_ARCHITECTURE_PLAN.md](BIO_INSPIRED_ARCHITECTURE_PLAN.md)）只保留"当前状态 + 下一步 + C 编号索引"，不再展开实施细节。
> 早期里程碑（2026-07-29 EMERGE 时代）保留在本文档开头。

## 编号规范

| 编号 | 含义 | 存放位置 |
|---|---|---|
| `C16`-`C26` | 机制演进迭代代号（工程版本号），记录范式转变与验证 | 本文档（唯一边集） |
| `x.y`（如 1.1/2.1） | 文档结构章节号 | 主 plan 内 |
| `日期`（2026-08-xx） | 时间戳 | 仅出现于本文档与 HISTORY 系列 |

---

## 早期里程碑（2026-07-29）

### EMERGE 现象确认（4 神经元协作）

**4 神经元协作 side_channels 微调（6 epochs，14950 步，~14 小时）**

| 神经元 | solo PPL | 融合权重 |
|--------|---------|---------|
| zh_aug0 | 211.6 | 0.250 |
| zh_aug1 | 114.6（最强个体） | 0.555（主导） |
| zh_aug2 | 225.3 | 0.129 |
| zh_aug3 | 246.9 | 0.066 |
| **协作** | **62.6** | - |

- **EMERGE 幅度: 协作比最强个体好 45.3%**（协作 PPL 62.6 vs 最强个体 114.6）
- 配置：4× compact（36M/个）、side_channels 12.58M + scale 12、Muon 优化器、field_conditioning=False、max_rounds=2、simple_zh 10000 条 6 epochs
- 意义：**多个小型神经元通过 side_channels 协作涌现超越最强个体的能力**——验证"小神经元协同工作匹配大模型"核心理念
- 产物：`data/neurons/side_channels_finetuned.pt`（v2 baseline, PPL=62.6）+ `_v2_baseline.pt` 备份
- 生成质量：top-k sampling (k=40) + rep penalty (1.2) + temp (0.8) 消除机械重复，但语义仍不连贯——根因 = compact 神经元训练不充分（非架构问题）

### Auxiliary-loss-free balancing 实施（2026-07-29）

借鉴 DeepSeek V3：非梯度启发式 bias 更新平衡各 channel 利用率，解决"死通道"。`neuron.py` 添加 `_channel_usage` 统计 + `update_channel_bias(update_rate=0.1)`；训练脚本每 50 步触发 + usage 诊断。端到端测试通过；短训练验证（300 条 74 步）bias 更新正常、无死通道（dead=0/12）。**结论**：死通道已被 v2 修复解决，此机制作为"保险"存在，PPL 影响可忽略。

### Shared Expert 机制实施（2026-07-29，负向结论）

借鉴 Kimi K3 / DeepSeek V3 的 always-active general 专家。实施完成（general 神经元 best_val_PPL=148.80 + ensemble `shared_expert_weight=0.3`），但评估**负向**：协作 PPL 62.6 → 108.6（恶化 +73.6%）。根因：zh_general 训练不充分（评估 PPL 257.5，比所有 aug 神经元都差），30% 固定权重稀释了 zh_aug1 主导作用。**教训**：借鉴机制不能盲目照搬，机制有效性依赖前置条件（general ≥ 域特定能力）；暂不启用，后续可动态权重重启。

---

## 迭代记录（C15-C26，2026-08-08 起）

### C15/C16/C16b/C16d/C17/C18 早期迭代（提及记录，细节散见于后续迭代）

- **C15**：quality_head + contrastive loss（回合级路由监督）
- **C16**：LoRA 保护 body（冻结 + 低秩增量）→ **个体能力零破坏原则确立**；quality_head 升级 + 对比学习
- **C16b**：per-neuron EMA z-score + 绝对质量 gate（quality 信号归一化，C16 教训：未校准 head 跨 neuron 不可比）
- **C16d**：全序列 NLL 监督（后被 C20 的 answer_mask 回合级取代）
- **C17**：神经发生（MaturityTracker：幼稚 0.1 → 成熟 1.0 ramp、lr 3×→1×）
- **C18**：客户端链路（assemble_cortex + chat/feed/sleep 接线）

### C19 任务级路由（Executive Control Routing，2026-08-08 ✅ 范式转变）

**背景**：C12-C16 四次路由迭代（LOO cosine / 域判别 head / quality_head NLL / gate+z-score）全部失败——共同根因 = **"统一空间 + 全局 token 级竞争"范式与生物机制相悖**（NLL/cosine/logit 跨 neuron 天然不可比；token 级 softmax 竞争导致回复频繁切换 neuron、风格断裂）。

**人脑参照**：解剖结构分工（面孔→梭状回）+ 前额叶执行控制（task set 确定后整条通路激活到任务结束）+ 局部竞争（WTA 只发生在同功能内部，跨脑区是信息传递）。

**范式转变**：token 级（C12-C16，失败）→ **任务级（C19）：回合级判定任务模式 → 主导 neuron 回合内稳定生成，不做 token 级竞争**。

**实施**（`cortex._executive_route`）：混合信号 = 启发式 `_infer_domain`（快）+ quality_head 回合级聚合（learned，per-neuron EMA z-score + 成熟度门 count<20 回退启发式 + z 绝对差门 ≥0.7σ）；leader 限定 dominant 域。冒烟验证通过（verify_c19_executive.py）。

**关键收敛**：所有 neuron 共享 general lm_head 后，`_generate_p7` 仍用 domain tokenizer decode → OUT_OF_RANGE；收敛为生成/decode 全程 general 256K 空间（identity 回填），domain 只负责激活选择。

### C20 回合级监督训练（2026-08-08 ✅ 验证 5/5）

- **answer_mask**（`forward_train` 新增）：per_neuron_nll 只对 answer（回复）部分算回合级 NLL——prompt 无区分度，answer 才是"谁能生成好这个回复"的真实质量信号
- **同域 batch 决策**：batch 内同域回合（NLL 可比），否则 dialogue neuron 转译 NLL 基线巨大被 gate 全排除
- **验证**（verify_c20_round_quality.py）：C20 head 不再 code 独占；回合级判定 **5/5**（code→code/math→math/zh→zh/dialogue→zh/en→en，修正 C19 的 math→en 误判）；z 绝对差门（≥0.7σ）防"全能型"错误覆盖
- **产物**：collab_v3_c20.ckpt.pt

### C21 词库多词表架构正式化（2026-08-08 ✅ 用户核心需求落地）

- **架构定位**：词库 = 多独立词表的可扩展集合（容量不限），neuron 绑定自己词表，跨词表靠词库转译协作；反转 C19 的"全 general decode"
- **关键发现**：① dialogue neuron 能力未退化（general 输入 + zh 头 + zh decode v3 口径能生成中文）；② **C16 LoRA 是 dialogue 负资产**（general 目标空间训练，注入后 zh 退化）→ loader 按 lm_head 空间过滤（256K 头才注入）；③ **round2 场污染**（装配后 zh 输出被英文 neuron 混合场污染 → 中英混合）→ leader 改用 round1 独立 logits（无场条件化，协作只用于判定）
- **验证**（verify_c21_generate.py）：回合级判定 5/5；dialogue executive 生成**流畅中文问答**
- **遗留**：4 个 general neuron 生成能力弱（→ C24 解决）

### C22 路径收敛（2026-08-08 ✅）

- **审计结论**：`generate()` 默认 collab_mode="fusion"（token 级，C19 已否定的范式）→ **线上实际是旧路径**；executive 需显式传参；多条实验路径并存（routing_mode/fusion_mode 多态）
- **收敛动作**：默认 collab_mode → executive；executive 模式跳过 hybrid 共振校验（消除双路径打架）；废弃 `--no-dialogue-lora`（残留参数干扰调用）
- **设计本意确认（用户）**：**振荡相位同步是态极设计本意**——共振本体应为相位同步驱动；当前"场向量累加 + 相位仅作门控"是实现偏移 → 缺口 R 核心方向（→ C23）

### C23 相位同步本体化（2026-08-08 ✅ 闭环，缺口 R 核心落地）

- **C23-A 共振权重**：`GammaOscillator.pairwise_binding`（binding_i = mean_{j≠i}[cos(θ_i-θ_j)]，同相绑结/异相解绑）；`scores = scores × (1 + binding_scale·binding)`。冒烟 6/6
- **C23-B 场本体**：场写入按相位绑定加权（round1 写入 scale ×(1+β·binding)，round2+ 逐轮重算）——相位同步直接塑造场状态。冒烟 8/8
- **C23-C 相位可微化**：新模块 `taiji/resonance/phasor.py`（PhasorDynamics）：2D 相位向量 p=(cosθ,sinθ)、可微点积 binding、可微叉积 Kuramoto 牵引、双驱动（前向物理牵引 + 反向任务梯度黎曼切向更新）。**关键工程发现**：相位是单位向量流形，普通 SGD 径向梯度被归一化抹掉 → 正确更新 = 切向投影 `tangent = g−(g·p)·p`。冒烟 13/13
- **C23-C2 ω/K 梯度打通 + 训练接入**：`PhasorDynamics.evolve()`（可微 Kuramoto，不 in-place）；forward_train 演化段保存 `_last_evolved_phasors` → 任务 loss 梯度经 binding → ω/K；train_round_level_quality `--enable-phasor` 显式启用。冒烟 15/15
- **C23-C3 phase-binding loss**：验证发现 checkpoint 显示 **ω/K 恒初始值**（contrastive_loss 完全不经过 binding 路径，冒烟梯度通 ≠ 训练生效）→ 新增 phase-binding loss（binding ∥ normalize(scores_pre)）。训练实测 ω 分化 [0.738..0.837]、K 学习、相位自组织分化——**任务驱动相位自组织验证成立**
- **C23-C4 监督纯净化（关键修复）**：完整配方训练暴露**监督污染**（C20 判定 5/5 → C23 full 4/5，quality_logits 膨胀）——根因：forward_train 场构造段 binding 调制污染 per_neuron_nll → contrastive 目标被相位自组织驱动漂移。修复：**训练场构造不再按 binding 调制**（监督测"谁能预测好"纯净 NLL），相位只经 scores 段 + phase_loss 可微；推理 forward 场写入 binding 本体化保留。另发现 **seed bug**（random.seed 在 shuffle 之后，两次训练数据顺序不同）→ 修复 seed 位置。最终 c23_final_seeded 完整验证：饱和 0/109、phase_loss 收敛 0.77→0.105、端到端判定 5/5 与 C20 一致
- **C23-C5 loader 默认装配**：train checkpoint `phasor_state` 附 `id_order`；loader Step 6 默认装配 PhasorDynamics（按训练顺序重排 phasors/omega 行 → load_state_dict 注入；无 phasor_state → 域先验 0°/60°/120°/180°；失败回退标量 GammaOscillator）。验证 15/15 + 判定 5/5 无回归

### C24 域目标空间 SFT + 判定修复 + 9 神经元挂载（2026-08-09~11 ✅）

- **目标**：C21 遗留——4 个 general neuron 生成能力弱（根因 = general 256K 空间续写训练无 SFT QA 能力）。修复路径（同 dialogue）：**域目标空间**（general 输入 + 域词表目标 + answer masking）
- **v1 失败 → 根因诊断**：foundation_v1 body 在 general 空间 NLL 无对角 + **native NLL 跨 neuron 不可比**（en 16K 专精词表对英文回合 NLL 恒定低 → en z-score 恒负 → en quality_logit 膨胀常数头）→ 判定全错。对照：foundation_v1_general body + 256K 头 NLL 完美对角 4/4
- **C24v2 双头架构（上限最高）**：neuron 同时保留 **judge_lm_head（general 256K 判定头，冻结，C20 信号链）+ 域头（生成）**；基座从 foundation_v1_general 出发，双 loss 训练（域 SFT + general 空间保留 gen_loss 防 body 漂移）。`train_domain_target_sft.py` + `ResonanceNeuron.judge_lm_head` + loader 注入
- **全量重训**（foundation_v1_dual，4 域 × 6 epochs）：code/math 判定对角保留；zh/en best PPL 319.1/167.3
- **C20 判定重训 v2**（collab_v3_c24v2）：quality_head 膨胀（C23 时代已膨胀 −4.2→50，softmax 饱和梯度消失自增强）→ **executive 判定改用 judge NLL 主信号**（C20 原始信号链，general 空间可比、无训练依赖）；端到端判定 **5/5 无回归**
- **域生成质量**：数据扩充 ×10（code 17599/math 22264/zh 30000/en 30000 条）重训后：code=3.4/math=3.7/zh=70.2/en=69.9 answer PPL；生成从碎片/空 → 有结构片段（zh markdown 代码块）；**zh/en 仍高（~70）**——词表大 + 响应长 + 51M 容量限制，非单点可解
- **zh_general 残留收敛**：shared_expert 机制从未启用（assemble_cortex 未传 shared_expert_id）→ 删除 neuron_zh_general.pt，装配收敛为 **9 阵容**（5 对话 + 4 域）
- **9 神经元挂载就绪**：API 装配路径用旧协作层 cross_spec_dialogue.pt（乱码）→ 显式用 collab_v3_c24v2 + foundation_v1_dual；test_api_dialogue 实测对话流畅（"你好！今天天气很好…"），符号乱码/混字消失
- **C20 v2 重训完成**（v1 域 neuron 上训的 head 与 v2 域 neuron 失配 → 重训 1090 步）：判定 5/5 无回归 + verify_c25_f_e2e 10/10 + quality 回退 3/3

### C25 对比问题解决（2026-08-09~11 ✅）

- **C25-A 词库实时编辑**（热插拔 → 词库不做限制 + 实时编辑）：`EditableVocabulary`（SentencePiece 包装，运行时 add_tokens 扩展区 + 持久化 JSON）、`TokenizerHub.to_editable/add_tokens/unregister_domain`、`resize_linear_for_vocab`；**256K 去硬编码**（判定头/共享表维度从权重 shape 推断）。verify_c25_vocab_edit.py **27/27 PASS**
- **C25-B STDP 突触生长/修剪本体化**：STDPTracker 增共激活统计累积（跨会话持久化）+ `apply_structure_updates`（低共激活通道修剪 + 高共激活缺失通道生长，邻居相似初始化；强权重通道保留防误删）+ sleep 接入。verify_c25_b_stdp.py **21/21 PASS**
- **C25-C 神经调质深度耦合训练**：新增 **acetylcholine**（DA=奖励 / ACh=新颖性互补，attention 聚焦增益映射 0.6+ACh×0.8）；训练闭环：loss 变化率同时驱动 DA 与 ACh（loss 上升 → ACh↑ 聚焦新输入）。verify_c25_c_neuromod.py **23/23 PASS**
- **C25-D 睡眠重放真重放 + 突触稳态下调**：`record_high_resonance_state` 增 active_nids（重放时再激活共激活统计，取代"纯统计占位"假重放）；consolidate 新增全局 side_channels ×0.98（NREM 慢波全局缩放——强通道净保留、弱信号整体下压）。verify_c25_d_replay.py **17/17 PASS**
- **C25-E 连续时间动力学**（对比文档"离散共振轮次替代连续动力学"修复）：
  - **核心**：`taiji/resonance/continuous.py`（ContinuousResonance）+ `ensemble.continuous_forward`——相位绑定驱动的连续激活替代不应期硬门轮替：时间步进 T=8 微步积分、激活 a_i(t)=σ(β·(binding_i−b0))、场积分 F(t+dt)=F(t)+dt·Σa_i·project(v_i)·conf_i、权重=时间平均激活、收敛=绑定分布 std 稳定；t=0 独立前向采集判定信号（监督纯净化）
  - **增量一（cortex 接入）**：collab_mode="continuous" 显式启用；A/B 显示 continuous 在 dialogue/zh 质量优于 executive
  - **增量二（forward_train 连续化）**：`forward_train` 增 `continuous: bool=False`（默认离散，既有调用零影响）；监督纯净（final_judge_logits round 1 采集）；顺手修复 quality_logits_t UnboundLocalError
  - **增量三（默认装配决策 → 回退）**：A/B 规模化 22 prompt continuous 全面不劣，但装配实测 **8 问空输出 5/8** → 回退默认 executive。根因：**连续模式多 neuron 协作不稳定**——zh 激活 5 个 dialogue neuron，同相群体绑定 → 时间平均激活权重均分 → leader 选到弱响应 neuron
  - **增量四（leader 质量信号修复）**：`continuous_forward` 新增 `round1_scores`（t=0 场共振分，有区分度 max-min=0.70）→ continuous 分支 leader 用 round1_scores 优先；空输出 5/8 消除
  - **增量五（默认装配切换 continuous）**：多次采样统计（12 prompt × 3 次）确认——非空率 1.00 持平、重复率 0.011 < 0.022、质量 9 胜 2 负 1 平 → `cortex.generate` 默认 collab_mode 切换为 "continuous"。挂载实测 8/8 全非空（"你好！今天天气真美好的一天。"）
  - **C25-E 全部增量闭环**
- **培养期端到端闭环**（C25-E 后，verify_feed_sleep_e2e.py **14/14 PASS**）："feed → sleep Phase 2 训练 → 影子 COW 写回 live → ckpt 保存 → 训练后推理"完整闭环；顺带修复 contrastive phase 混合规格维度崩溃（compact 512 vs standard 768 → pad 到公共 max dim + 跨规格投影层）
- **渐进改善验证 + 破坏性更新修复**（verify_feed_sleep_progressive.py **24/24 PASS**）：5 轮"feed 8 条 → sleep 训练"循环，held-out PPL **2161 → 972 → 489 → 384 → 393 → 448（末轮降 79%）**。首跑 FAIL 根因 = 灾难性遗忘（小样本 × 高 lr × 共享大嵌入表）→ **分层学习率修复**：shared_embedding lr 1e-5（慢速积累）、lm_head/embed_adapter 3e-4、epoch 3→1
- **PPL 口径修正**（diag_zh_ppl_masks.py）：提问式评估集分布偏移 PPL 虚高（10761）→ 基座在 zh_sft 同分布仅 ~199；评估集必须与训练同分布且 ≥16 条
- **平台期定性：容量饱和**（verify_feed_sleep_scale.py **11/11 PASS**）：两个独立实验同模式（前 2-3 轮大幅改善后斜率骤降）→ **51M compact 在 zh 50K 词表上的阶段性上限**（PPL 平台 ~500-600）；突破需基座级升级，喂养只提供"快速逼近上限"
- **zh 基座升级前置诊断**：规格盘点（general 全 512/51M，仅 zh_std0 768/134M）；leader 脱节实证（134M zh_std0 0/5 当选且分数最低档——场共振分衡量"输入-场方向匹配"非生成能力）；容量 vs 质量实证（134M 生成最流畅但非质变）→ **general zh 升级优先级下调，主线 = 对话数据**
- **zh leader 信号 A/B 确认**（verify_zh_leader_ab.py）：强制 134M 仅轻微占优（长度 +11%、主题命中 +18% 但逐 prompt 波动大）→ **容量非主要杠杆**
- **dialogue 欠训练根因诊断**（diag_dialogue_data.py）：5 个 dialogue neuron 全部只训 4000 步（预算截断非收敛平台，val PPL 89-102 且日志仍下行），数据质量良好（48K alpaca-zh 统一格式）→ **碎片根因 = 欠训练**
- **🔄 全部 5 个 dialogue neuron 续训已启动**（2026-08-11 18:10，--steps 8000，resume 4000→8000）：修复 base vocab 错配（--base_id 直接用 dialogue 自身）+ optimizer/scheduler 容错；日志 logs/resume_{aug0..3,std0}.log
- **C25-F 多阶段任务模式链**（2026-08-11 ✅，verify_c25_f_e2e.py **10/10 PASS**）：`cortex.generate_staged(stages)`——每阶段 = task-set（prompt + mode + domain + max_tokens），阶段间显式传递中间输出（{prev} 模板），异常阶段隔离；zh→code→zh 三阶段编排可用
- **C25-G quality_head 膨胀根因修复**（2026-08-10 ✅，verify_c25_quality_fix.py **11/11 PASS**）：quality_head 学成常数偏移（logit 大 → softmax 饱和 → KL 梯度消失 → 自增强压不住）→ **actual 改 std 标准化**（减 detach 均值 ÷ detach 标准差再 ÷ 温度 1.0）——softmax 输入恒 ~±2 永不饱和、梯度恒非零；learned quality proxy 恢复可用

### C26 场固化（可写记忆第 0 格，2026-08-11 ✅ verify_c26_field_memory.py 11/11 PASS）

- **背景（架构审视结论）**：共振场（推理时可写的共享状态）已是"可写记忆"形态，但缺"写后不固化"
- **实现**：`taiji/resonance/field_memory.py`（FieldMemoryBank：固化 = L2 归一化 + 余弦去重 0.92（突触稳态下调的工程简化）、检索 = 余弦 top-k、持久化 .pt）+ sleep_engine Phase 1.5 场固化挂载（record_field_memory 队列 → 睡眠沉淀 → field_memory.pt）
- **验证（真实装配 9 神经元）**：睡眠固化 4 条 / 重复固化去重 / **跨会话检索 top-1 全命中 4/4**（sim 0.45-0.69）/ 注入管线 4/4 + 注入改变生成输出 4/4 / 重启恢复 / sleep() 主流程挂载
- **机制边界**：记忆"完整复述进生成"受 zh dialogue 欠训练限制（生成碎片），**续训完成后回归项**
- **关键实现注意**：generate 的生成循环每 token 调 think() 结束即重置场，`cortex.field` 在生成后是空的——场快照必须从 `think()` 返回值（field_state）截获
- **对比 Titans 定位评估**：态极完善后在**记忆生命周期维度可超越 Titans**（睡眠巩固/记忆→突触沉淀/群体协作是 Titans 缺失维度）；**最大差距 = 可学习写策略**（Titans 梯度驱动 memory-as-model vs 态极朴素场快照）
- **C26 增量一：跨域语义锚点投影（缺口 L 落地，2026-08-11 ✅ verify_c26_field_alignment.py 8/8 PASS）**
  - 诊断（diag_cross_domain_alignment.py）：共振场无自发对齐（同义 0.226 vs 错配 0.248，-0.022）；对齐冒烟（verify_c26_cross_domain_align.py 4/4）：**冻结场向量蕴含可提取的跨域语义**（P 空间 +0.547）→ 无需 hub 全套
  - 产品化：`taiji/resonance/field_alignment.py`（AnchorProjector：field→128 2 层 MLP + 对比 margin loss 训练函数 + 持久化）+ cortex.set_anchor_projector/project_field_state（可选挂载，未挂载原样返回，不影响生成路径）
  - 30 对双语术语训练后：同义 0.932 vs 错配 0.681（+0.251）；持久化/批量/归一化全过
- **C26 增量二：可学习写策略（缺口 K 落地，2026-08-11 ✅ verify_c26_write_gate.py 8/8 PASS）**
  - `FieldMemoryBank` 增 WriteGate（输入 = 场向量 + 与既有记忆最近邻 sim → P(值得写入)），consolidate 支持 gate（学习门控替代硬阈值，None 回退阈值向后兼容）
  - **门控优于硬阈值实证**：硬阈值 0.92 会误收 sim=0.9 的模糊重复；门控（训练负样本覆盖 0.88-0.98）学会拒绝——学习写策略的直接收益
  - 注意：训练正样本 sim 区间须覆盖实测主题间场基线（0.57-0.72），否则门控误拒新主题（首跑 added=2/4 → 修正区间后 4/4）
- **C26 增量三：多频段振荡 theta-gamma 嵌套（缺口 R 项，2026-08-11 ✅ verify_c26_theta_gamma.py 9/9 PASS）**
  - ContinuousResonance 增 theta_omega/theta_amp/theta_init（**默认 0 不启用，零回归**）：theta 慢相位单调推进、包络 1+A·cos(θ(t)) 周期复原、gamma 激活振幅被 theta 包络调制（调幅嵌套，Lisman 嵌套编码）
  - 真实装配 think 无回归（默认路径不受影响）
- **C26 产品闭环：组件接入 sleep 场固化（2026-08-11 ✅ verify_c26_field_memory_product.py 11/11 PASS）**
  - `FieldMemoryBank` 增 `projector`（AnchorProjector）：consolidate 存锚点副本、retrieve 在跨域语义锚点空间检索（对齐语义而非原始场方向）；save/load 含 anchor
  - `SleepEngine.get_field_memory` **自动装配**：data_dir 存在 write_gate.pt/anchor_projector.pt（train_field_memory_components.py 产物）→ 自动挂载到记忆库（学习门控 + 锚点检索）；无产物回退硬阈值 + 场空间（向后兼容）
  - 全链路验证：训练产物保存 → 自动装配 → 门控固化（4 写入 + 重复拒）→ 锚点检索 4/4 → 重启恢复（组件+记忆+检索）→ 空目录回退兼容
  - **产品路径就绪**：运行 `train_field_memory_components.py` 一次（~5min CPU）即让 sleep 场固化进入"可学习写 + 跨域对齐"模式

---

*本记录基于主 plan 2026-08-11 前内容浓缩归档，忠于实施事实；后续 C 迭代记录继续追加于此。*
