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
| 记忆（C26） | ✅ 场固化 | FieldMemoryBank + sleep Phase 1.5：跨会话检索 4/4 命中、跨重启恢复 |
| 培养期闭环 | ✅ 可用 | feed → sleep 训练（分层 lr 防破坏性更新）→ 影子写回 → ckpt；渐进改善已验证（held-out PPL 降 79%） |
| 续训（完成） | ✅ 4000→8000 | 5 dialogue 全部完成（best PPL：compact 101-107 / std0 95.27@4000）；口径机制化后回归通过 |
| 回归测试 | ✅ 16/16 | tests/ 下 pytest 统一入口（3 文件 16 用例）：口径契约 10 + 共振 side_channel 6；requirements.txt 补 pytest |
| 遗留瓶颈 | ⚠️ zh/en 生成 | answer PPL ~70（词表大 + 响应长 + 51M 容量），域生成片段级；对话质量受欠训练限制（续训中） |

### 1.4 闭环缺口清单

| # | 类型 | 说明 |
|---|------|------|
| A-H | ✅ 已修复 | 混合规格装配 / 协作权重加载 / shared_embedding / 域路由 / fusion_mode bug / SpecSelector / diagnose_domain / Play→Coactivation 链路（详见 HISTORY C17-C24） |
| I | ✅ 已挂载 | 综合体接入聊天接口：API 装配 9 阵容（collab_v3_c24v2 + foundation_v1_dual），test_api_dialogue 对话流畅；环境变量 TAIJI_COLLAB_NAME / TAIJI_EXTRA_NEURONS_DIR 可覆盖 |
| J | ⏳ 训练中 | dialogue neuron 欠训练（4000 步预算截断非收敛平台）→ 续训 4000→8000 步 |
| K | ✅ 已接入产品路径 | **可学习写策略（WriteGate）**：verify_c26_write_gate.py **8/8 PASS**——门控（场向量+最近邻 sim → P(值得写入)）可学习；**门控优于硬阈值**（拒绝 sim=0.9 模糊重复，硬阈值 0.92 漏判）；**已接入 sleep 场固化**（sleep_engine 自动装配 data_dir/write_gate.pt，存在即用无则回退） |
| L | ✅ 已接入产品路径 | **跨域语义对齐（锚点投影）**：verify_c26_field_alignment.py **8/8 PASS**——AnchorProjector 产品化 + cortex.set_anchor_projector 挂载；30 对双语术语训练后同义 0.932 vs 错配 0.681（+0.251）；**已接入记忆检索**（FieldMemoryBank.projector：检索在跨域语义锚点空间进行，睡眠固化自动装配 data_dir/anchor_projector.pt） |
| M | ✅ 已验证 | **多频段振荡 theta-gamma 嵌套（缺口 R 项）**：verify_c26_theta_gamma.py **9/9 PASS**——ContinuousResonance 增 theta_omega/theta_amp（默认 0 不启用，零回归）；theta 相位单调推进、包络周期复原、嵌套调制在 [base×(1-A), base×(1+A)] 内、真实装配 think 无回归 |
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
| C26 | **场固化**（可写记忆第 0 格：睡眠沉淀 + 跨会话检索 + 注入） | ✅ |

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
3. **zh 对话数据主线**：C24 dialogue 数据扩充重训（zh_aug*/zh_std0 共用瓶颈，对话级数据直接提升生成）
   - **✅ 数据扩充（2026-08-12，提交 84c2e9a）**：发现 sft_shared_core/unique 与 alpaca **100% 重复**（实际唯一仅 44.4K 条，数据/参数比远低于预期 → 直接解释质量瓶颈）。新增 build_dialogue_extended.py 从 BelleGroup/train_2M_CN 下载 150K → 去重/清洗后 +123K 唯一 → 总 167K 条（3.8×）
   - **⚠️ 中断事件（2026-08-12 20:15）**：软件更新终止全部训练进程（aug0 step200、其余加载中，eval_every=1000 → 零 ckpt 保存、白跑）。**改进（提交 15e4509）**：eval_every 默认 1000→500（中断最多丢 500 步 ≈1h）。20:21 已重启 5 进程
   - **✅ 重训完成（2026-08-14 03:19）**：compact 4 个全部 8000/8000（best val PPL：aug2=68.74 / aug3=71.81 / aug0=72.03 / aug1=73.60）；std0 8000/8000 完成 best_val_PPL=**67.42**（134M 上限收益显现）。5 个 ckpt 全部落盘 data/neurons/neuron_zh_*_dialogue.pt，旧数据 best 95-102 → 降 27-30%
   - **✅ 新数据回归（2026-08-13/14）**：① verify_zh_leader_ab（修复后口径）：8-13 均长 35.2/23.4（A 胜 10/12）；8-14 std0 升级后复测均长 30.0/39.2（B 胜 7）、重复 0.186/0.230（A 优）、命中 0.281/0.275（A 微优）——**两轮均判定收益边际，leader 信号改进维持不采纳**；② test_api_dialogue：8 问无死循环、假退化彻底消失，闲聊级正常，知识答问仍受 51M/134M 参数规模限制；③ C26 记忆复述：11/11 PASS
   - **✅ 连续默认化回归（2026-08-14）**：generate 默认 continuous（line 936）跑产品路径 test_api_dialogue 8 问全非空无死循环（Q1/Q3/Q7 正常）；A/B（verify_c25_e_collab_ab 修复口径后）**20/20 PASS**——executive/continuous 判定一致、5 域生成全非空、leader 域合理。**连续默认路径稳定，无需回退**
   - **✅ 融合后质量基线（2026-08-14）**：verify_zh_leader_ab（融合状态）：A 均长 31.6/重复 0.163/命中 0.270，非空 36/36；A vs B 从 7:2 收敛到 3:4:5——**leader 融合无回归且轻微改善**（重复率 0.186→0.163，当前机制不再系统性落后 134M）。test_api_dialogue 8 问全非空无死循环
4. **✅ C25-E 遗留：continuous leader 融合质量信号（2026-08-14 完成）**：诊断（diag_c25e_leader_quality_gap.py）证实**弱 neuron 独占真实存在**——aug2 场共振分系统性碾压（0.7-0.93 vs 0.01-0.17）当选 leader 5/7 次，但其生成质量（zh lm_head NLL）常是 5 个 dialogue 中最差（leader 恰为 NLL 最优仅 1/7，Spearman=-0.171）。**修复**：continuous leader 融合 = 域内归一化共振分 × 质量信号（-NLL，round1_logits 零额外前向）等权求和（`_fuse_leader_quality` + `_nll_quality_from_round1_logits`，质量信号首算缓存避免每 token 重复 softmax）。**回归 verify_c25_e_leader_fusion 3/3**：leader NLL 质量位次均值 2.12→1.12（「推荐一本好书」4→0、「怎么学好英语」4→1，aug2 不再独霸），连续生成非空率 8/8 不降，tests 16/16 全过
5. ~~缺口 L 落地：场级锚点投影正式化~~ ✅ 已完成（AnchorProjector + WriteGate + theta-gamma 三件套 + 产品闭环，见缺口清单 K/L/M/N）
6. ~~锚点投影/写门控进装配~~ ✅ 已完成（train_field_memory_components.py 训练产物 → sleep_engine 场固化自动装配）
7. **对话数据扩充重训**（zh_aug*/zh_std0 主线，续训完成后）：C24 dialogue 数据扩充 → 重跑 finetune_neuron_dialogue
8. **✅ 项目整理（2026-08-12，提交 a4064f9）**：① 数据层清理 13.71GB 废弃产物（foundation_v1_general_smoke 7.4G + foundation_v1_sft 3G + verify_v3 2.3G + neurons_backup_3000step 1.2G + verify_v3_full + 空目录×4；distill 因 experiment_config DATA_DIR 活跃引用保留）；② 脚本层归档 103 个一次性 verify/_smoke/diag 到 scripts/archive/（git mv 保留历史），scripts/training/ 收敛 149→46 主训练脚本；③ 测试层 pytest 入 requirements + tests/ 统一入口 16/16。回归：tests 16/16 通过、核心 import 正常

### 2.3 中期：跨域协作（上限优先版）

- **hub neuron 设计与实现**（缺口 L）：参考人脑联合皮层，expert 规格 + 256K lm_head + hub-and-spoke 拓扑 + 跨域对比 loss。详见 [HUB_NEURON_DESIGN.md](file:///e:/taiji-neuron/plans/archive/HUB_NEURON_DESIGN.md)
- 多任务 loss（缺口 P）：SFT masking + margin ranking + diversity + 对比 loss
- 推理路径优化（缺口 Q）：域 token 对齐表 + 长上下文 + 多轮对话状态

### 2.4 远期：生物学机制深化（缺口 R 剩余）

- 记忆向量直接条件化 leader 生成（打通"场影响生成"这条遗留路径——协作目前只在判定层）
- 记忆 → 突触沉淀：高频场模式写入神经元权重（LoRA 增量），海马→皮层两层记忆
- ~~多频段振荡（theta-gamma 嵌套）+ 跨频耦合~~ ✅ 嵌套机制已验证（缺口 M），跨频耦合待续
- 真正睡眠重放（forward 重放 + 经验回放训练，C25-D 已落地首步）
- 自组织新生（从经验生长，非 teacher 蒸馏）
- 多阶段任务模式链 v2（task-set 序列完整版，C25-F 已落地首步）

### 2.5 架构方向记录（2026-08-11 讨论，未实施）

- **态极定位**：不是"更好的 Transformer"，而是"Transformer 组件化 + 生命周期闭环"的工程先行者
- **可写记忆路径**：共振场已踩在"可写记忆"门槛上（推理时可写共享态），C26 补齐固化/检索/注入；完善后**在记忆生命周期维度可超越 Titans**（睡眠巩固/记忆→突触沉淀/群体协作是 Titans 缺失维度），最大差距 = 可学习写策略
- 详见 [TAIJI_VS_HUMAN_BRAIN_COMPARISON.md](file:///e:/taiji-neuron/plans/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md)

---

## 📖 三、接口梳理（2026-08-10）

- **产物**：[INTERFACE_REFERENCE.md](file:///e:/taiji-neuron/INTERFACE_REFERENCE.md)（根目录接口速查与易错点手册）
- **8 大陷阱**：① `side_channels` 是 `excite_channels` 别名不含 inhibit；② `forward`（weighted_logits）vs `forward_train`（fused_logits）key 完全不同；③ `_parallel_forward` 返回 6 元组但 docstring 写 5；④ judge_lm_head（判定）vs lm_head（生成）双头；⑤ `get_ffn_gain==get_lr_multiplier`、`get_attention_temp_gain==get_field_write_scale` 同公式双语义；⑥ `batch_align_and_embed` 返回元数随 answer_marker 变化；⑦ `consolidate` 位置参数顺序易写反；⑧ `resize_embedding_for_vocab` 文档提到但不存在
- **✅ 文档级修复完成（4 项）**：`_parallel_forward` docstring 5→6；`forward` fusion_mode 默认统一 "soft"；translator 修正不存在引用；`consolidate` 强 pair 精确双向强化。验证 C25-B/C/D 无回归
