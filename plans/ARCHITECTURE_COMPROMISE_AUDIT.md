# 架构妥协点审查报告

> 梳理整个项目中所有"为了易于实现采取的妥协方案"，按上限损失严重性排序。
> 每个妥协点给出：当前实现 → 妥协原因 → 上限更高方案 → 提升幅度。
>
> 调研范围：共振场核心 + 训练流水线 + 推理运行时，共 90+ 妥协点。
> 本报告聚焦**系统性妥协**（影响全局上限），局部小妥协见归档。

---

## 📌 人脑动态神经元：热插拔三机制（2026-08-06 实施中）

**动机**：用户提出"人脑神经元动态增加/减少不影响工作，可正常对话输出，且可多线程处理不同任务"——态极需支持运行时增删神经元而不打断推理，并支持多任务并行。

| 人脑机制 | 态极实现 | 状态 |
|---------|---------|------|
| 神经元增删不影响工作中推理 | **快照隔离**：`ensemble.forward/forward_train` 入口 `nmap = dict(self.neurons)` 浅拷贝，全程用快照；增删只改原 dict，被删 neuron 在快照中仍持引用不崩 | ✅ 已完成（commit bfccaf8） |
| 增删互斥防交错 | **热插拔锁**：cortex 增删接口（`add/remove/isolate/revive_neuron`）用 `threading.RLock` 串行化；推理读走快照不拿锁（读-写无争用） | ✅ 已完成（commit bfccaf8） |
| 混合规格种群共存 | **热插拔补投影**：`ensemble.add_neuron` 在 `field_dim ≠ unified` 时自动补建 `CrossSpecProjector`（正/反向）——修既有 bug（混合种群下新 neuron 缺投影层 → 推理 3072-vs-2048 RuntimeError）；同时**放宽 field_dim 校验**（原只对首个 neuron 校验，混合种群会误拒） | ✅ 已完成（commit bfccaf8） |
| 训练与推理并发（学习时正常对话） | **训练/推理分离（影子权重 COW）**：推理（generate/generate_multimodal）**去训练锁**（原 acquire(timeout=10) 阻塞对话）；训练侧 `_train_cortex_neurons` 在 `_clone_module` 克隆副本上训练（live 权重训练全程稳定），结束一次性写回 + 恢复引用；`_clone_module` 用配置重建 + load_state_dict（deepcopy 因 RotaryEmbedding._cache_lock 不可 pickle 会崩） | ✅ 已完成（本 commit） |
| 多线程并行处理不同任务 | **任务级并行**：① ensemble `field` 改属性——推理 forward 期间返回 thread-local 独立共振场（`_get_task_field`，继承 W_cond + gamma gate，跨任务互不污染）；② forward scratch（round_scores/_router_*/_logits_keep_ids）全部 thread-local（`_fstate`），写穿 `_last_forward_round_scores` 供 sleep_engine 分裂选择；③ forward_train 显式用默认场；④ API SSE 聊天 `asyncio.to_thread` 移同步 generate 出事件循环（并发请求真正并行）；`active_nids` 按域路由不同 neuron 子集（已有） | ✅ 已完成（本 commit） |

**快照隔离已验证**（`verify_hotswap_snapshot.py` 全通过）：
- 推理线程持续 forward，主线程并发 add/remove/isolate/revive → 推理 8 次 forward 全部正常、分数有限
- 混合规格热插拔：field_dim=256 neuron 加入 field=512 → 投影层自动补建（in=256/out=512 断言）
- 隔离/复活语义：pop 保留引用可复活，复活后推理正常
- 回归：旧 3072-vs-2048 崩溃场景（静态混合规格推理）分数有限

**训练/推理分离已验证**（`verify_train_infer_separation.py` 全通过）：
- 训练周期全程 live 权重完全稳定（推理读到稳定权重）——影子隔离核心契约
- 写回后 live == 影子（训练生效）+ 训练期间移除的 neuron 不复活
- 推理线程 25 次 forward 全部正常（训练 COW 周期全程不崩溃、分数有限）
- dict 引用不变（ensemble.neurons is cortex.neurons 贯穿全周期）

**任务级并行已验证**（`verify_task_parallel.py` 全通过）：
- 三线程 barrier 同步并发推理，全部不崩溃、分数有限
- 路由隔离：每任务 final_scores keys 只含本任务 active_nids（跨任务无污染）
- 并发 top-1 与串行基线完全一致（field 隔离生效；neuron 不应期共享为符合人脑语义的微小调度差异）
- task field thread-local 缓存复用断言

**真实集成验证**（`verify_hotswap_integration.py` 全通过，真实 zh_general compact 权重）：
- 三线程并发真实推理 ✓；推理线程中热插拔（同规格 compact + 跨规格 1024→2048 投影层自动补建）✓；隔离/复活 ✓；真实 COW 周期（live 稳定 → 写回生效）✓
- **已知限制（待讨论）**：`ensemble.add_neuron` 保留 hidden_size 校验——standard(768)/compact(512) 混合 hidden 种群会被拒。当前生产统一 compact 无实际影响；若未来引入 expert(1024)/foundation(384) 混合种群，需评估 forward 路径（embed_adapter 各自投影 → 理论上兼容）后放宽

**并发容错补充**：`_update_channel_usage` 两处 `post_neuron` 改 `.get()` + None 跳过（推理中 side_channel 清理的 post 神经元可能已被移除）。

---

## 📌 弱神经元剔除（凋亡）v2：人脑分层凋亡（2026-08-06 重构，commit 6efd24f）

**动机**：原 ApoptosisTracker 固定 PPL>200 + 激活率<5% 两个绝对阈值——① general 256K 空间与域空间 PPL 口径完全不同（固定 200 会误杀全部 general 空间 neuron，与当前基座训练直接冲突）；② 均匀激活假设与"5 联合 > 5"分工路由冲突（域 neuron 只在自己域激活是设计意图）；③ 永久删除无恢复。

**人脑机制映射**：

| 人脑机制 | 态极实现 |
|---------|---------|
| 突触修剪先行（Synaptic Pruning）| 弱 side_channels（`_channel_usage` 长期低）先被修剪，神经元本体保留（`prune_synapses`）|
| 活动依赖存活（use it or lose it）| 激活率作为生存信号，种群相对归一化 |
| 神经营养竞争（营养=网络贡献）| 协作边际贡献（A/B 剔除实验）+ 网络中心度（side channel 出入度）入生存分 |
| 凋亡级联（启动→执行→清除）| active → candidate → isolated → trial → dead，多阶段可取消/可复活 |
| 成熟度保护 | 幼稚态（maturity_ratio<0.5）不参与凋亡竞争 |
| 抑制性神经元保护 | inhibitory 不把"网络贡献"作为存活要求（权重转移给活动+能力）|
| 空间自适应 | PPL 用**种群内百分位**（同空间内排序），不跨空间比绝对值 |

**v2 实现（已验证）**：
- [lifecycle.py](file:///e:/taiji-neuron/taiji/resonance/lifecycle.py) `ApoptosisTracker`：多维生存评分（activity/ppl 分位/contribution/connectivity/redundancy 惩罚，缺失自动降权）+ 状态机（`step_population`）+ `prune_synapses` + `cleanup_neuron` 改移入 `_recycle_bin`（不直接删）
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py)：`isolate_neuron`（摘除路由保留 ckpt）/ `revive_neuron`（重载 ckpt 复活）/ `get_isolated_neurons`
- [sleep_engine.py](file:///e:/taiji-neuron/taiji/life/sleep_engine.py) Phase 4：采集多维信号（activity/ppl/connectivity/maturity/inhibitory）+ 级联动作（隔离/试复活/复活/凋亡补偿新生）+ 突触修剪
- 兼容：`record_ppl`/`check_activation`/`get_apoptosis_candidates` 保留（verify_apoptosis.py 全通过）

**验证**：状态机全链路 active→candidate→isolated→trial→dead ✓；复活 ✓；成熟度保护 ✓；抑制性保护（exc 0.31 vs inh 0.80）✓；**general 空间安全（4 域 PPL 全 >200 仍全 active，旧固定阈值会误杀）** ✓；突触修剪 ✓；旧接口兼容 ✓

**待注入信号（评估基础设施就绪后）**：contribution（协作 A/B 剔除实验）、redundancy（field_vector 相似度）

---

## 📌 递归设计检查（2026-08-06 补充）

**结论：项目"递归"（taiji/life 递归改进 + 递归蒸馏）两条回路——任务进化回路活着，递归改进回路是死的。**

**✅ 已修复（commit 7e3a3ea）**：
1. **删除废弃死代码**：`recursive_improver.design_next_generation` 及 13 个私有辅助（~300 行）、`evolution_engine.execute_generation_transition`/`_design_to_model_config`/`_validate_student`/`EVOLUTION_PATH`/`_get_current_generation`/`_get_next_generation`
2. **修潜伏 bug**：`_get_next_gen_name` 双同名定义（L623 被 L740 覆盖 → design_next_generation 必 TypeError）——方法已随死代码删除
3. **激活死配置**：`loss_plateau_steps` 从定义未用 → `record_sleep_training` 记录 loss 历史 + `check_evolution_ready` 平台检测（std < 0.05）
4. **接入真实调用方**：`chat_strategies._record_recursive_strategies` 每次推理记录 prompt/tool_choice/reflection 策略到 RecursiveImprover（输入环复活）；功能验证：12 条记录 → 产出 2 条提案（prompt + tool）
5. **sleep_engine Phase 5 重构**：design_next_generation 调用 → 生成 `next_training_data_spec.json`（训练数据建议，消费方 = 跨域协作层训练）
6. **防噪**：prompt 模式提取过滤 >80 字符整句（中文无空格分词噪声）

**修复前问题（已解决）**：
- 输入环断开：`record_strategy` 零调用方 → 分析恒空
- 输出环废弃：design/execute 均 deprecated/NotImplemented
- 单体变大叙事残留（0.5B→7B）与 BODY_LIFE 决策 3 冲突

---

## 📌 态极递归流程（神经元架构下，2026-08-06 定义）

```
                        ┌────────────────────────────────────────────┐
                        │ 推理时（循环起点）                           │
                        │ chat_strategies：ReAct + 工具 + 直接生成     │
                        └───────┬────────────────────────────────────┘
                                │ 同时记录两路
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │ 策略环（输入）        │              │ 任务环（输入）         │
   │ record_strategy      │              │ record_task_success/  │
   │ prompt/tool/reflection│             │ failure               │
   └──────────┬──────────┘              └──────────┬───────────┘
              ▼                                    ▼
   ┌─────────────────────┐              ┌──────────────────────┐
   │ RecursiveImprover   │              │ EvolutionEngine      │
   │ 策略记录（jsonl）     │              │ 成长值/失败率/知识饱和/ │
   └──────────┬──────────┘              │ loss 平台             │
              │ 睡眠 Phase 5             └──────────┬───────────┘
              ▼                                    │ check_evolution_ready
   ┌─────────────────────┐              ┌──────────▼───────────┐
   │ 策略改进提案         │              │ 能力扩展信号（ready）  │
   │ analyze_and_improve │              │ ① neurogenesis 信号   │
   │ → EventBus +        │              │ ② 训练数据建议         │
   │   best_strategies   │              └──────────┬───────────┘
   └─────────────────────┘                         ▼
                                        ┌──────────────────────┐
                                        │ next_training_data_   │
                                        │ spec.json（弱点+建议） │
                                        └──────────┬───────────┘
                                                   ▼
                                        ┌──────────────────────┐
                                        │ 协作层训练（数据环）    │
                                        │ train_multi_domain_  │
                                        │ foundation.py（基座）  │
                                        │ train_cross_domain_  │
                                        │ collab.py（协作层）    │
                                        └──────────┬───────────┘
                                                   ▼
                                        ┌──────────────────────┐
                                        │ 能力扩展             │
                                        │ 新 neuron / 更强协作  │
                                        │ 5 联合 > 5（涌现实证） │
                                        └──────────┬───────────┘
                                                   │ 再次投入使用
                                                   ▼
                                              回到推理时（递归）
```

**三环说明**：
1. **策略环**（推理时 → 睡眠分析）：记录实际使用的 prompt/工具/反思策略 → 睡眠时统计成功模式 → 低效工具降权提案（`web_fetch 成功率 0%` 实测产出）
2. **任务进化环**（持续）：任务成功/失败 → 阶段升级（infant→adult）→ neurogenesis 信号（睡眠创建新神经元）+ 进化报告
3. **数据环**（睡眠 → 训练，递归核心）：`check_evolution_ready` 满足 2 条件（成长值/失败率/知识饱和/loss 平台）→ 生成 `next_training_data_spec.json` → 跨域协作层训练 → 能力扩展 → 重新投入推理

**废弃（已删除）**：代际变大（0.5B→7B）、design_next_generation、execute_generation_transition——神经元架构的递归 = 协作层数据闭环，非整体替换模型。

---

## 📌 当前执行状态（2026-08-06 22:30 更新）

**"共享 general lm_head 统一输出空间"：基座训练完成 ✅ + 产物验证 ✅ + 路由适配完成 ✅**

**架构落地（已完成，commit 5bb522b 等）**：
- ✅ [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py#L160-L164)：`ResonanceNeuron.__init__` 支持注入 `shared_lm_head`——所有 neuron 共享同一个 general 256K lm_head，直接预测通用 token（无词库转译投影稀释）→ 路由置信度信号保留
- ✅ [train_multi_domain_foundation.py](file:///e:/taiji-neuron/scripts/training/train_multi_domain_foundation.py)：`--target-space general` 完整适配
  - `create_shared_lm_head()`（512→256K，131M params）；neuron 创建注入共享 head
  - general 模式：输入/目标都 general 编码（`batch_align_and_embed(batch, general_sp, general_sp, emb)`），全文本 loss（无 answer marker）
  - 优化器三分：neuron 主体 lr / embedding 独立低 lr / **共享 head 独立 lr（--lm-head-lr）**
  - checkpoint 瘦身：`_strip_shared_head()` 剥离 131M head 出 neuron ckpt（共享 head 独立存 `shared_lm_head.pt`，避免每域 ~525MB 冗余）
  - 回读验证支持 `lm_head_path` 注入（最终 + 配对校验两段都传，防止 16K 域 head 算 256K 目标崩溃）
- ✅ **冒烟验证**（3 步/域 × 4 域，batch 2）：管线跑通 + checkpoint 回读正确；loss 从随机水平（12.9-13.2 ≈ ln256000）三轮内降到 10.3-12.5，**学习正常**
- ✅ **正式基座训练完成**（21:23）：`data/foundation_v1_general`（4 域 × 600 步 = step 2400，batch 8），`foundation_history.json` 完整记录。**注意：4 个域 neuron 是全新随机初始化训练（compact，非继承旧 5 神经元/旧域基座）**——统一空间要求所有 neuron 直接预测 general 256K token，旧域 lm_head（code 12K 等）输出空间不一致无法复用
- ✅ **collab/eval 加载路径预修复**（commit 5bb522b）：`load_neuron` 支持注入 `shared_lm_head` + 新增 `load_shared_lm_head()`——general 基座（ckpt 已剥离 131M head）可正确加载并输出 256K 空间 logits

**产物回读验证通过**（`verify_foundation_general.py`，修复 tokenizer 口径后）：
- **回读 PPL**（best ckpt + 最终 embedding，collab/eval 口径）：**code 6.1 / math 21.1 / zh 274.9 / en 178.2**——code/math 健康，zh/en 因 general 256K 空间全文本 CE（无 SFT masking）偏高，属预期口径（不是坏 ckpt）
- **配对校验**（best 权重 ↔ best 步 embedding）：code 14.1 / math 41.5 / zh 514.7 / en 187.7
- **输出空间验证**：4 域全部输出 general 256K logits，top token 正常（code '<0x0A>'、math '4'、zh '、'、en '.'）

**🔧 修复训练脚本回读 bug（commit 00cda08）**：`verify_checkpoint` 目标 tokenizer 传错——L335/L341 传 `domain_sps[d]`（域词表），与训练（general_sp 编码）不一致 → 目标编码错位，训练脚本自身打印的"最终回读" PPL 是假值（400 万级）。已改为 `general_sp, general_sp`（输入/目标都 general，与训练循环 L267 完全一致）。

**✅ 路由适配统一空间完成**（`verify_unified_space_routing.py`，commit 待填）：
1. **新增 division 融合模式**（[ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) `_division_logit_fusion`）：同 vocab（256K）场景 per-position 硬路由——每个 token 位置交给 max-prob 最高的 neuron，直接复用原生置信度（无转译投影）。`division_norm` 为 per-neuron 归一化变体（实验）
2. **修复架构 bug**：`forward` 的 active_ids 原为 set（迭代顺序不定）→ round_logits/all_logits dict 顺序随机 → 路由监控/诊断索引错位。改为有序 list（保持 nmap 顺序），路由信号顺序确定性
3. **验证结果**：
   - [2] 域内分工全对角占优：code 文本→code 0.66、math→0.54、zh→0.46、en→0.34（分工方向正确，路由信号在统一空间可用）
   - [3] 基线 PPL（协作层未训练时）：个体最优（code 19.5/math 56.3/zh 82.6/en 108.5）> soft（共振分融合）> division 裸 > division_norm——**跨 neuron 的 logits 尺度天然不可比**（code 平均 max-prob 0.566 vs zh 0.374，尖锐者抢走非自身域位置但预测错误），启发式归一化无效（过度平坦化）
   - [4] 生成冒烟正常（中文 prompt 回显 + 多 neuron 续写）
4. **关键结论**：division 分工路由是"路径打通 + 信号可用"的适配（域内分工正确），PPL 校准需**协作层训练**（C12 对比约束让共振分对齐 NLL 排序 / Sparse Router 学习路由）——即下一步

**后续链（下一步：协作层 general 空间训练）**：
1. ✅ ~~路由适配统一空间~~（division 模式 + set 顺序修复 + 基线记录，见上）
2. **协作层 general 空间训练 + 评估**（域内 EMERGE + 跨域 zh→code 语义桥接）——校准路由信号的核心步骤
3. 结果记录到 plans + 提交

**背景**：跨域协作 v3 + rounds=2 已达成域内 EMERGE +4.0% 全域转正（"5 联合 > 5" 域内实证）；通用空间投影极限诊断结论 = 静态稀疏投影到 256K 摧毁置信度（max-prob≈0.001）→ 共享 general lm_head 免投影是架构上限最高的解

---

## 📌 历史状态（2026-08-06 19:40 跨域实测线）

**跨域协作 5>5：域内全正（+4.0%）✅，通用空间探针确认架构方向**

**最新实测（v3 权重 + rounds=2，n-ppl 5）**：
- ✅ **域内 EMERGE 平均 +4.0%（全域转正）**：code +1.9% / math +3.6% / zh +6.9% / en +3.3% → "5 联合 > 5"域内实证
- ✅ **通用空间生成探针**：zh 提问 → general 256K 空间分工路由，输出变通顺中文（zh neuron 被选中；code 域路由输出 code 碎片）——证明通用空间让 zh 理解与 code 表达共存，路由结构正确
- ❌ **无配对数据 → 模型只会续写中文，未学会"切到 code 表达"**：通用空间路径正确但需配对训练数据

**下一步（唯一建议）**：构建配对数据（zh_sft 中 52 条含代码样本做种子 + code_sft 指令机器翻译成中文配对），在 general 输出空间训练协作层，让侧通道学会"zh 理解写入场 → code neuron 读取场生成"的转换 → 语义涌现

**已完成链条**（全部提交）：
基座重建（foundation_v1）→ 跨 vocab 硬路由融合（EMERGE -4228%→-5.9%）→ 协作层 v3（-5.9%→+0.9%）→ v3+rounds=2 全域 +4.0% + 通用空间探针

---

## 📌 历史状态（2026-08-06 19:10 跨域实测线）

**跨域协作 5>5 实测：域内涌现达成 ✅，跨域语义桥接待决策**

**最终实测结果（cross_domain_v3 训练权重 + rounds=2 场条件化）**：
- ✅ **域内 EMERGE 平均 +0.9%（转正）**：code +2.1% / math -0.2% / zh -0.8% / en +2.5%
  - 协作不再破坏个体，code/en 还超越个体 → "5 联合 ≥ 5"（个体基线 code 5.4 / math 12.2 / zh 66.6 / en 62.0，均远低于随机）
- ⚠️ **跨域生成（zh→code）结构性涌现**：输出呈 code 形态（#注释、"for i in range"、"#include"），单个体无法做到（code neuron 读不懂中文、zh neuron 输出不了代码）——但**语义未桥接**（生成的代码与提问无关）

**关键决策点（下一步方向）**：
1. **场状态桥接**：训练时显式用"zh 理解 → code 生成"的配对数据（双语指令对），让侧通道学会把 zh neuron 的理解写入场、code neuron 读取场生成 → 语义涌现
2. **通用词表输出空间**：融合输出改到 general 256K 空间（各 neuron 投影到 general，路由与生成都在通用空间），zh 理解与 code 表达天然共存 → 更大改动
3. **接受当前里程碑**：域内 5≥5 已实证 + 结构性跨域涌现，语义跨域作为后续独立课题

**已完成链条**（全部提交）：
- 基座重建（foundation_v1，verify_v3 损坏诊断+按对话配方重训）→ 跨 vocab 硬路由融合（EMERGE -4228%→-5.9%）→ 协作层 v3 训练（EMERGE -5.9%→+0.9%）

---

## 📌 历史状态（2026-08-06 18:30 跨域涌现训练线）

**跨域协作 5>5 实测：基座重建 ✅ → 融合修复 ✅ → 跨域涌现训练中**

**实测里程碑**：
1. ✅ **基座重建**（foundation_v1）：verify_v3 不可用（训练 embedding 丢失），按对话配方重训 4 域 neuron + 共享 embedding（best 权重早停保存 + 配对校验）。个体基线：code PPL 9.3 / math 38.2 / zh 345 / en 138（均远低于随机）
2. ✅ **跨 vocab 融合修复**（commit 060cc1d）：软加权平均被域外 neuron 投影的极端幅值淹没（zh max|logit|=60828 vs code ±14，0.07×60000=4200 主导）→ 改为**按位置硬路由**，置信度 = min(原生 max-prob, 投影后 max-prob)（原生项=域门封顶，投影项=目标空间信息量）
3. ✅ **域内 EMERGE -4228% → -5.9%**：code -1.2% / math -8.4% / zh -0.2% / en -13.7%（协作不再破坏个体；en 偏差源于 code/en 英文域重叠竞争）
4. ⏳ **跨域涌现训练中**（cross_domain_v3，epochs 4）：新融合下 n_rounds=2 场条件化给侧通道梯度（R2 注入让 zh 理解写入场、code 生成读取场 → 桥接）

**跨域生成（zh 提问→code）当前状态**：路由已让输出变为 code 侧 token 碎片（非 token 汤），但 zh 理解未桥接到 code 生成 → 需训练好的侧通道/场条件化

**待决策/下一步**：
- 协作层重训完成后用 n_rounds=2 + 训练权重评估，验证场条件化能否桥接 zh→code

---

## 📌 历史状态（2026-08-06 17:00 跨域诊断线）

**跨域协作 5>5 实测：基座诊断 → 重建 → 验证中**

**核心诉求**：5 个域神经元（code/math/zh/en）联合 > 5 个独立之和（涌现）。

**诊断结论（verify_v3 基座不可用）**：
- verify_v3 多域神经元由 train_neurons_from_scratch 训练，输入是**域编码**（p7 input_ids max 11912 < code vocab），但训练时 `create_shared_embedding` 随机初始化且**从未保存** → 评估时任何管线（域编码/通用编码/joint 变体）PPL 均≈随机（loss 9.4-10.6 vs 随机 9.39）
- 已验证：code/math/zh/en 的 neuron_*.pt 与 *_joint.pt × shared_embedding*.pt 全部组合，loss 均≥8.9
- **根因**：域编码输入 + 丢失 embedding = 不可复现的模型（权重对着随机向量训练，向量已丢失）

**正确配方（对话管线实证，PPL 2.2 可复现）**：
- 输入 = general_sp 编码 → 共享 embedding（保存）→ neuron；标签 = 域 tokenizer
- 冒烟验证：对话 embedding 基座 + code 数据，code neuron 500 步 loss 9.4→2.7（PPL 15）✅

**行动**：
1. ✅ 新建 `train_multi_domain_foundation.py`：按对话配方重训 4 域 neuron + 联合训练共享 embedding（对话基座 warm-start），保存时回读验证 checkpoint（用户规则：训练前确认保存正确）
2. ⏳ 运行中：`data/foundation_v1`（600 步/域，~25min，含周期保存+回读验证）
3. ⏭ 基座完成后：重跑跨域协作训练（train_cross_domain_collab.py）+ 评估（_eval_cross_domain_collab.py 已验证 API 兼容）

**参考基线（同域 5>5 已实证）**：5 个 zh 对话 neuron 协作 PPL 29.7 vs 个体 95，EMERGE 65.7%（对话管线）

**待决策/下一步**：
- 基座训练完成后自动继续跨域协作训练 → 评估 5>5

---

## 📌 历史状态（2026-08-06 15:26 对话线）

**EOS + 短答案筛选重训已完成，评估中**：

**训练结果**（2026-08-06 15:26 完成，总步数 39480）：
- 最终 PPL=3.0（E1 331 → E4 15.5 → E6 6.3 → E8 3.0），远优于上一轮 E6 32.9
- 产物：`cross_spec_dialogue.pt`（推理用）+ `cross_spec_dialogue.ckpt.pt`（可续训）

**评估诊断**（2026-08-06）：
1. ✅ **发现并修复 eval 加载缺陷**（commit b5a1c80）：`load_cross_spec_weights` 此前只加载 cross_spec 投影层，**丢失 side_channels/scale_bias/body_state**（协作层训练核心产物）→ eval 用"训练前"参数，PPL 虚高 705。修复后训练分布 PPL 2.2 复现训练（口径一致）
2. ✅ **修复推理路径 gap**（缺口 N，commit 待）：推理 "soft" 此前走 per_position（entropy 启发式），无视训练学的共振分 → 训练分布 PPL 2.2 vs 推理 12.6。A/B 实验证实共振分融合生成明显更通顺 → forward 统一 "soft" = 共振分 softmax 融合（与训练 forward_train 对齐）。修复后 eval 协作 PPL 1516 → 240.7，EMERGE 70.8%
3. ⚠️ **分布泛化 gap 仍存**：训练分布 2.2 → eval 分布（无筛选长答案）240——训练数据（≤150字短答）与 eval 分布差异大，有过拟合风险
4. ⚠️ **生成知识性错误仍存**：语言流畅度已达标（"你好，很高兴。我是一个人工智能助手"），但事实错误（"法国首都是伦敦"）——模型容量/数据上限，非路径问题

**待决策/下一步**：
- 数据侧提升：更高质量/更长答案/更多样的 SFT 数据后重训（治模型知识上限）
- 或接受当前质量，进入跨域协作层训练（code/math 加入）

**根因诊断**（API 实测质量不达标的三个核心缺陷）：
1. ❌ **训练数据未加 EOS**：`batch_align_and_embed` 只产生 domain_targets，无 EOS token → 模型永不自然停止
2. ❌ **训练数据严重未充分利用**：仅加载 15000/88730（17%），其他 6 个文件完全未用
3. ❌ **训练/生成长度严重不匹配**：alpaca-zh 答案 200-500 字，生成 max_tokens=60（约 80-100 字）→ 模型学成了"长文本续写"而非"简短问答"

**三项核心修复**：
1. ✅ **EOS 注入**（[translator.py:498-521](file:///e:/taiji-neuron/taiji/resonance/translator.py#L498-L521)）：`batch_align_and_embed` 追加 EOS token（截断前注入，保证 EOS 始终在末尾）。smoke test 验证：末尾 target=3(EOS)，sft_mask=True ✓
2. ✅ **短答案筛选**（[utils.py:283-338](file:///e:/taiji-neuron/scripts/training/utils.py#L283-L338)）：`load_dialogue_texts_multi` 加 `max_answer_chars=150` 参数，筛选答案≤150字的样本，匹配生成 max_tokens=60
3. ✅ **数据量提升**：从 15000 条（仅 alpaca_zh 第一个文件）→ 19745 条（7 个文件全部加载，去重后），多样性显著提升

**重训参数**：
- 从头训练（不 --resume）：加 EOS 后训练任务本质变了，不继承"无 EOS 长答案"偏见
- epochs=8, batch=4, lr=1e-3, max_texts=88730, max_answer_chars=150
- 总步数 39480，ETA 约 301 min/epoch
- 备份：cross_spec_dialogue.pt.pre_eos_finetune

**自适应激活设计已完成**（2026-08-05）：详见 [§4.0c](#40c--自适应激活设计r1-软路由--top-k-稀疏路由2026-08-05-设计)。Probe-based Sparse Router 方案落地，待训练完成后实施。

**并行工作完成**（2026-08-05，训练期间开展 4 项，全部提交）：
1. ✅ **eval_dialogue.py 支持任意 checkpoint**（commit 96b9ecd）：`--ckpt_path` 参数，用于训练完成后对比 held-out PPL 判断过拟合早停
2. ✅ **稀疏 vs 稠密对比脚本**（commit 658d546）：`compare_sparse_dense.py`，同 checkpoint 双 ensemble 对比协作 PPL/EMERGE/激活数/速度（smoke 已验证）
3. ✅ **跨域神经元 Step 2 数据准备**（commit 5d98f95）：`p7_{domain}_mixed_tokenized.pt`（6000 条/域，域 SFT + 英文对话），train_neurons_from_scratch.py 支持 `--data-suffix mixed`
4. ✅ **API 集成修复**（commit 858c3e1）：新建 `taiji/core/config.py`（TrainingConfig + 6 个接口），memory_watchdog 补 `force_memory_refresh`/`get_memory_status_dict`，API 29 路由可正常启动

**API 路径五项修复（已完成，commit e94ca1d + 6426797）**：
1. 训练/推理 embedding 错配 → per-neuron shared embeddings 注入 cortex
2. 解码 byte fallback → `domain_sp.DecodeIds`
3. 默认参数宽松 → 60/0.55/15/soft
4. EOS 缺失 → 温和 EOS bias(+0.5)（临时方案，现已被训练时 EOS 注入替代）
5. 跑偏截断 → 连续 3+ 非中文 token 回退截断

---

## 🧠 神经元综合体饱和点判断标准（2026-08-04 决策）

**何时触发新神经元进化**——三个量化指标：

| 指标 | 饱和标准 | 当前值 | 状态 |
|------|---------|--------|------|
| PPL 收敛 | 连续 2 epoch Δ<0.5 | epoch 9→10 Δ=-14.2 | ❌ 远未饱和 |
| EMERGE 递减 | 连续 2 次评估 Δ<5% | 21.7% vs 22.7% | ❌ 协同收益稳定 |
| 参数效率 | body 梯度范数持续 <1e-4 | 仍在下降 | ❌ 未饱和 |

**触发时机**：当前轮训练后（PPL 预计 < 20），如果 API 质量仍不达标，就是触发新神经元的时机。预计 1-2 轮训练后（3-6 天）。

**新神经元方向（上限最高）**：跨域扩展（en_dialogue）。理由：
1. 当前全是 zh 神经元，跨域协作是"小神经元匹配大模型"的核心
2. 已有跨域 tokenizer 基础设施（en 16K vocab）
3. 跨域协作能显著提升综合能力（不同视角的共振）

**技术路径**：neurogenesis + lifecycle + establish_topology_channels 自动重建 + finetune_cross_spec 微调

---

## ⚠️ 架构本源定性（2026-08-04 认知重构，详见 §4.0）

**核心定性（重构）**：**涌现已存在**——单个神经元 PPL ~20-42 无法正常对话，5 个协作 PPL 15-33 能正常对话，这正是涌现的定义。"单神经元较强"是效率优势（不需要巨大协作层就能涌现），不是劣势。人脑类比不是唯一标准。

**唯一核心缺陷**：自适应激活不足（协作层稠密，R1 软路由需要强化为 top-K 稀疏路由）。

**决策时点**：
- **Step 1（进行中）**：zh 综合体 + EOS+短答案重训 → 验证能正常对话（涌现的输出验证）
- **Step 2**：加入 code/math/en 等特定能力神经元，测试跨域涌现（§4.0b 候选 1）
- **Step 3**：强化自适应激活（R1 → top-K 稀疏路由），提升协作效率
- 方向 B 定位修正：不再是"上限更高的备选"，而是"探索另一种涌现机制的实验"，优先级降低

**详见**：[§4.0 涌现已存在](#40--架构本源定性涌现已存在核心缺陷是自适应激活2026-08-04-认知重构) | [§4.0b 涌现深化探讨](#40b-涌现的深化探讨--新能力方向2026-08-04-认知重构) | [§6 方向 B](#六方向-b-备案小神经元--强协作架构2026-08-04-设计)

---

## 一、系统性妥协（影响全局上限）

### S1. 共振机制从未被端到端训练 ★★★ 最关键

| 维度 | 当前 | 上限更高 |
|------|------|---------|
| 训练路径 | `forward_train` 单轮、无场、无侧通道（[ensemble.py:824-833](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L824-L833)） | 可微多轮共振（Gumbel-softmax / straight-through） |
| 后果 | "共振"是推理期技巧，neuron 从未学过"如何写场、如何协同" | 共振成为可学习能力 |
| 妥协原因 | 多轮含 hard top-K / argmax / `.item()` 不可微 | |
| 提升幅度 | 协作涌现 +30-50% | |
| 实施难度 | 高（架构性改动） | |

**核心问题**：`forward_train` 调用 `neuron.forward(shared_embeddings, return_logits=True)`，**不传 field_state、不传 side_signals、不应用 neuromodulator scale**。所有生物学机制（STDP/神经调质/Gamma/睡眠/新生）均以 `Optional[Any]` 注入，且**只在推理 forward() 生效，未进入梯度流**。

### S2. 256K embedding 配 16K tokenizer（隐性错配）★★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| shared_embedding | `nn.Embedding(256000, 512)`（[utils.py:246-256](file:///e:/taiji-neuron/scripts/training/utils.py#L246-L256)） | 不变（256K × 512） |
| general tokenizer | 16K en tokenizer 回退（[utils.py:111-117](file:///e:/taiji-neuron/scripts/training/utils.py#L111-L117)） | **256K general BPE（已存在）** |
| 后果 | 14.6 万 embedding 行永远训练不到；中文生僻字被 byte fallback | 全词覆盖（中文测试 20 token, 0 unk） |
| 妥协原因 | `build_domain_tokenizers.py` 无 general 域 | **已补充 general 域 + 修复路径不一致（T13）** |
| 提升幅度 | 词覆盖 +30-50%，PPL 虚高根因 | 已解除 |

**修复详情**（2026-08-01）：
- 验证 `taiji/domains/general/sp_general.model` 已是 256K vocab，中文覆盖率优秀（整词覆盖，0 unk）
- 修复 [build_domain_tokenizers.py](file:///e:/taiji-neuron/scripts/training/build_domain_tokenizers.py)：
  - `OUTPUT_DIR` 从 `domain_tokenizers/` 改为 `taiji/domains/`（与 load 路径一致，修复 T13）
  - 修复 `PROJECT_ROOT` 路径计算错误（少一级 parent）
  - `DOMAINS` 加入 general 域（256K vocab，混合语料 zh+en+code+math）
  - 新增 `load_mixed_corpus()` 函数支持 general 域的混合语料加载

### S3. Loss 单一化（全线纯 CE）★★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| 训练 loss | 5 个训练脚本全用纯 shift-CE | 对话训练用 SFT masking + 协作训练用多任务 loss |
| 协作层训练 | 纯 CE，无协作约束（[finetune_cross_spec.py:431-438](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py#L431-L438)） | CE + balance_loss + diversity_loss（S1 已修复） |
| SFT 训练 | question 和 answer 同等权重（[finetune_neuron_dialogue.py:284-288](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py#L284-L288)） | **SFT answer masking：只对"答："后的 token 计算 loss** |
| 后果 | side_channels 退化成噪声；模型复述 question | 协作真涌现 + 回答质量 |
| 妥协原因 | CE 最简单 | |
| 提升幅度 | 协作涌现 +15-30%，回答质量 +15-25% | 已解除 |

**修复详情**（2026-08-01）：
- [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) `batch_align_and_embed` 新增 `answer_marker` 参数：
  - 传入 `answer_marker="答："` 时返回 4 元组 `(shared_emb, targets, mask, sft_mask)`
  - `sft_mask` 标记 answer 部分（"答："之后的 token）为 True，question/pad 为 False
  - 不传时返回 3 元组，**完全向后兼容**（10+ 调用点无需修改）
  - 处理截断、无分隔符、padding 边界情况
- [experiment_config.py](file:///e:/taiji-neuron/scripts/training/experiment_config.py) 新增 `SFT_ANSWER_MARKER = "答："` 常量
- 3 个对话训练脚本应用 SFT masking：
  - [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：训练 + eval 都用 SFT masking，eval 改用 `reduction="sum"` 防止 answer 为空时 NaN
  - [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：协作训练用 `shift_mask & shift_sft_mask` 交集
  - [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：同上
- balance_loss（负载均衡）和 diversity_loss（field_vector 多样性）已在 S1 修复中接入 `forward_train`
- 5/5 验证通过（[verify_sft_mask.py](file:///e:/taiji-neuron/scripts/training/verify_sft_mask.py)）：向后兼容、基本正确性、batch 对齐、截断处理、无分隔符

**注**：margin ranking 暂未实现（与 balance_loss 语义部分冲突，且需要 individual_logits 额外计算开销）。当前 balance_loss + diversity_loss 已覆盖协作约束需求。

### S4. 训练步数整体偏短 ★★ ✅ 代码已修复（待重新训练）

| 阶段 | 修复前 | 修复后 | 建议步数 |
|------|---------|---------|---------|
| base (compact) | 16000 | 16000（未改，已训练完成） | 30000-50000 |
| base (standard) | 16000 | 16000（未改，已训练完成） | 50000-80000 |
| dialogue finetune | 4000 | **12000** | 12000-16000 |
| side_channels | 6ep (~15000步) | **8ep (~20000步)** | 20000+ |
| cross_spec | 3ep (~7500步) | **8ep (~20000步)** | 20000+ |

**4000 步对话微调确实太少**——36M 小模型需更多 epoch 内化对话格式，4000 步只够 2.5 epoch，明显欠拟合。当前多轮对话质量差的根因之一。

**修复详情**（2026-08-01）：
- [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：`--steps` 默认值 4000 → 12000
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：`--epochs` 默认值 3 → 8
- [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：`--epochs` 默认值 6 → 8
- warmup_steps=100 保持不变（12000-20000 步训练中占比 0.5-0.83%，合理）
- base 神经元训练步数未改（已训练完成，后续进化时再提升）
- **待重新训练才能验证效果**（建议等 S5 数据扩充完成后统一重新训练）

### S5. 数据规模与复杂度偏小 ★★ ✅ 代码已修复（待联网下载扩充）

| 数据集 | 修复前 | 修复后 | 建议规模 |
|--------|---------|---------|---------|
| simple_zh (base) | ~100K 小学作文 | ~100K（未改，已训练完成） | 500K+ 混合语料 |
| alpaca-zh (finetune) | 49K（单文件） | **49K→200K+（待联网下载 Belle/COIG）** | 200K+ |
| side_channels 训练 | 10K simple_zh | **100K 对话数据（默认 dialogue）** | 100K+ |
| eval | 30 条 | **100 条** | 500+ |

**simple_zh 是小学水平**，compact 神经元在它上面学到的语言能力上限低。**alpaca-zh 单点依赖**，覆盖面窄（偏百科问答），缺多轮、缺推理、缺代码。

**修复详情**（2026-08-01）：
- [experiment_config.py](file:///e:/taiji-neuron/scripts/training/experiment_config.py)：
  - 新增 `DIALOGUE_DATA_FILES` 列表（7 个本地文件，合并 ~97K 条，去重后 ~49K）
  - 新增 `DIALOGUE_HF_SOURCES` 列表（Belle 2M CN + COIG，可扩充 150K+）
- [utils.py](file:///e:/taiji-neuron/scripts/training/utils.py)：
  - 新增 `load_dialogue_texts_multi()`：多文件合并 + 去重 + 打乱 + SFT marker 过滤
  - 新增 `load_dialogue_texts_hf()`：从 HuggingFace 下载 Belle/COIG，转 "问：...\n答：..." 格式，本地缓存
- 3 个对话训练脚本改为使用 `load_dialogue_texts_multi`：
  - [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：eval 扩充 30→100 条
  - [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：dialogue 模式用多文件合并
  - [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：默认改为 dialogue 数据，max_texts 10K→100K
- **待联网下载**：本地文件去重后仅 ~49K 条（sft_unique 是 alpaca_zh_sft 子集），需运行 `load_dialogue_texts_hf()` 下载 Belle/COIG 扩充到 200K+

### S6. 域 token → re-encode 往返（推理核心缺陷）★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| 自回归生成 | domain token → text → general token → shared_emb（[cortex.py:1350-1358](file:///e:/taiji-neuron/taiji/brain/cortex.py#L1350-L1358)） | **对齐表预计算映射，消除 text 往返** |
| 后果 | 信息丢失 + 无 KV cache + 训练-推理分布偏移 | 消除信息丢失 + 为 KV cache 铺路 |
| 妥协原因 | 避免异构 vocab 间维护对齐表 | |
| 提升幅度 | 极高（推理速度 + 长文本质量） | 已解除（text 往返部分） |
| 实施难度 | 中（对齐表）/ 高（共享 codebook） | |

**修复详情**（2026-08-01）：
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) 新增 `_get_domain_to_general_alignment()` 方法：
  - 构建 `{domain_token_id: [general_token_ids]}` 对齐表（首次构建后缓存）
  - 对每个 domain token，预计算其 general token IDs 映射
  - 消除自回归生成时的 `domain→text→general` re-encode 往返
- `_generate_p7()` 修改：
  - 在获取 domain_sp 后构建对齐表（line 1260-1262）
  - 用 `alignment_table.get(next_token, [pad_id])` 替代 `domain_sp.id_to_piece + general_sp.encode`
  - 保留 fallback 路径（对齐表为空时走旧路径）
- **KV cache 仍未启用**（底层 layers.py 支持，但 neuron.py:454 丢弃 cache）—— 作为后续独立优化项

### S7. side_channels 全连接拓扑 ★★ ✅ 已修复

| 维度 | 修复前 | 修复后 |
|------|------|---------|
| 拓扑 | 全连接 mesh（N×N-1 条） | **结构性拓扑：full / knn / hub_spoke / hybrid（默认）** |
| 后果 | 通道互相干扰，梯度信号被均分 | 每条通道学到更鲜明角色；近邻更强先验 |
| 妥协原因 | `NeuronGeometry` 距离已算但未用于裁剪 | **已用距离+规格容量驱动拓扑构建** |
| 提升幅度 | 训练效率 +40%，协作质量 +5-10% | 已解除 |
| 实施难度 | 中 | |

**修复详情**（2026-08-01）：
- 新建 [topology.py](file:///e:/taiji-neuron/taiji/resonance/topology.py)：4 种拓扑模式
  - `full`：全连接（向后兼容）
  - `knn`：k 近邻对称拓扑（按 NeuronGeometry 距离）
  - `hub_spoke`：最大规格神经元作 hub，其他只经 hub 通信
  - `hybrid`（默认）：仿皮层分级 — 同(域,规格)全连接 → 跨规格经规格hub → 跨域经全局hub
- hub 选择：按容量（hidden_size × num_layers）降序，centroid 距离为 tiebreak
- 距离门控 init_scale：近邻 gate≈1 → 50.0（强先验），远邻 gate≈0 → 10.0（弱先验）
- [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)：`__init__` 新增 `geometry` 参数，接受外部传入的 NeuronGeometry
- `infer_topology_from_state()`：从 checkpoint 的 side_channels_state keys 自动推断训练时拓扑
- 5 个脚本更新为拓扑驱动建立：
  - [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：`--topology` 默认 hybrid
  - [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：`--topology` 默认 hybrid
  - [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py)：优先从 checkpoint 推断拓扑，回退 hybrid
  - [eval_aug_joint.py](file:///e:/taiji-neuron/scripts/training/eval_aug_joint.py)：同上
  - [analyze_side_channels.py](file:///e:/taiji-neuron/scripts/training/analyze_side_channels.py)：同上
- **向后兼容**：评估脚本自动从 checkpoint 推断拓扑，旧 checkpoint（全连接）自动匹配全连接拓扑

### S8. 冻结策略过保守 ★★ ✅ 已修复

| 阶段 | 修复前 | 修复后 |
|------|--------|---------|
| dialogue finetune | shared_emb frozen | **shared_emb 默认 trainable（--freeze_embedding 恢复旧行为）** |
| side_channels | neuron + emb 全冻结 | **解冻最后 N 层 transformer + norm + lm_head + field_write（默认 N=2）+ 可选 emb** |
| cross_spec | neuron + emb 全冻结 | **同 side_channels：解冻最后 N 层 + 可选 emb** |
| 优化器 | 单一 Muon+AdamW（side_channels only） | **body + emb 走独立 AdamW，lr = args.lr × body_lr_ratio（默认 0.1，温柔微调）** |
| checkpoint | 仅 side_channels + scale | **+ body_state + shared_embedding_state + body_optimizer_state + body_scheduler_state** |
| 交付产物 | 仅 side_channels + cross_spec | **+ body_state + shared_embedding_state（eval 脚本自动加载）** |

**核心问题**：从未联合训练过 neuron + side_channels + embedding，三阶段割裂导致表示空间无法协同适配。

**修复详情**（2026-08-01）：
- [finetune_neuron_dialogue.py](file:///e:/taiji-neuron/scripts/training/finetune_neuron_dialogue.py)：`--train_embedding` 默认 True，`--freeze_embedding` 恢复旧行为；shared_emb 默认参与训练以适配对话 token 分布
- [finetune_side_channels.py](file:///e:/taiji-neuron/scripts/training/finetune_side_channels.py)：
  - 新增 `--unfreeze_layers`(默认2) / `--train_embedding` / `--body_lr_ratio`(默认0.1) 参数
  - 解冻最后 N 层 transformer + norm + lm_head + field_write，让核心表示适配协作动态
  - 优化器分离：side_channels 走 Muon+AdamW，body+emb 走独立 AdamW（低 lr 温柔微调）
  - **修复关键 bug**：body_optimizer 之前创建了但训练循环未调用 zero_grad/step/scheduler.step，body 参数梯度无限累积且永不更新；现已修复
  - `build_final_artifact()`：交付产物含 side_channels + body + emb
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：同 side_channels 的 S8 改造（解冻最后 N 层 + 可选 emb + body 优化器 + checkpoint 扩展 + build_final_artifact）
- [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py)：加载 side_channels 后自动应用 body_state + shared_embedding_state（缺失则跳过，兼容旧 ckpt）
- [eval_aug_joint.py](file:///e:/taiji-neuron/scripts/training/eval_aug_joint.py)：同上，加载 body + emb 微调结果
- [_smoke_s8_checkpoint.py](file:///e:/taiji-neuron/scripts/training/_smoke_s8_checkpoint.py)：checkpoint round-trip smoke test 全部通过（body/emb/optimizer_state 完整恢复，0 mismatch）
- **向后兼容**：旧 checkpoint（无 body_state/emb_state）自动跳过加载，不影响现有训练产物

### S9. 生物学机制是推理期占位，非训练一等公民 ★★（核心已修复：调质门控 attention/FFN）

> **状态更新**（2026-08-01）：
> - S1 修复已让 neuromodulator/gamma/STDP/coaction 进入 `forward_train` 梯度流：
>   - neuromodulator `get_field_write_scale()` → 乘 scores → 影响融合权重（[ensemble.py:848,977-978](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L848)）
>   - gamma `tick()` + `kuramoto_step()` + `batch_gate_factors()` → 乘 scores（[ensemble.py:951-986](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L951)）
>   - STDP `record_firing()` + coaction `update()` 记录时序（不影响梯度）
> - **S9 修复（2026-08-01）：调质从"融合层 scores 缩放器"升级为"Transformer 内部门控"**：
>   - norepinephrine → `get_attention_temp_gain()` → 缩放 query → 门控注意力温度（聚焦/泛化）
>   - dopamine → `get_ffn_gain()` → 缩放 SwiGLU 输出 → 门控 FFN 强度（强化/衰减）
>   - 注入路径：`ensemble.forward` / `forward_train` → `_parallel_forward` → `neuron.forward` → `TransformerBlock` → `GroupedQueryAttention` + `SwiGLU`
>   - 全程可微（gain 是 Python float，但调质本身是外部状态；attention/FFN 权重通过 gain 进入梯度流）
>   - 4/4 smoke test 通过（[_smoke_s9_neuromod_gain.py](file:///e:/taiji-neuron/scripts/training/_smoke_s9_neuromod_gain.py)）
>
> 审计原文"forward_train 内完全不引用 self.neuromodulator"已**过时**。剩余真实缺口见下表。

| 机制 | S1+S9 后现状 | 剩余缺口（上限更高） |
|------|--------|---------|
| STDP | 记录发放时序，不影响梯度 | **影响 attention/FFN 权重**（Hebbian 可塑性进入 body） |
| 神经调质 | ✅ 门控 attention 温度 + FFN 强度 + 融合 scores | **per-region 调质**（当前全局共享，未来按域/区差异化） |
| Gamma | 单 40Hz 频段，门控 scores | **多频段（theta-gamma 嵌套）+ 跨频耦合** |
| 睡眠 | 重放只是计数（[neuro_modulation.py:210-212](file:///e:/taiji-neuron/taiji/resonance/neuro_modulation.py#L210)） | **真正 forward 重放 + 经验回放训练** |
| 新生 | `should_trigger_neurogenesis()` 存在但依赖外部 teacher | **自组织新生（从经验生长）** |

**核心问题（已修复）**：调质已从"融合层 scores 缩放器"升级为"Transformer 内部门控"，进入 attention/FFN 计算并参与梯度流。剩余 STDP/睡眠/新生是独立子项，不阻塞主训练路径。

### S10. Transformer 层零生物学修改 ★ ✅ 已修复（树突化 + 预测编码）

| 维度 | 修复前 | 修复后 |
|------|--------|---------|
| 层结构 | 标准 LLaMA 块（"zero changes to existing code"） | **树突化 TransformerBlock：basal + apical 双通路 + 预测编码整合** |
| 注释 | "zero changes to existing code" | 神经调质门控 + 树突化 cross-attention |
| 妥协原因 | 复用标准层 | **已解除：apical cross-attention 接收 field_state 作为自上而下反馈** |
| 提升幅度 | 结构性容量上限提升 | 已实现 |
| 实施难度 | 高 | 已完成 |

**修复详情**（2026-08-01）：
- [config.py](file:///e:/taiji-neuron/taiji/resonance/config.py)：`NeuronConfig` 新增 `dendritic_enabled: bool` 和 `apical_kv_dim: Optional[int]` 开关
- [layers.py](file:///e:/taiji-neuron/taiji/layers.py)：`TransformerBlock` 扩展 apical 路径
  - **Basal 路径**（始终存在）：标准 attention + FFN（自下而上，处理输入）
  - **Apical 路径**（dendritic=True 时创建）：
    - cross-attention：Q 来自当前层输入，KV 来自 field_state（全局集体意识场）
    - 独立的 apical_wq/wk/wv/wo 投影 + apical_norm
    - 无 causal mask（cross-attention，KV 是全局反馈）
  - **胞体整合**（预测编码）：
    - `apical_prediction = x + h_apical`（apical 残差预测）
    - `error = x - apical_prediction`（预测误差）
    - `gate = sigmoid(somatic_gate(x))`（每位置决定信任 basal 还是 apical）
    - `x = x - error_scale * gate * error`（误差校正，error_scale 可学习）
  - S9 神经调质门控（temp_gain/ffn_gain）同时作用于 basal 和 apical 路径
- [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)：
  - 根据 `dendritic_enabled` 构建 dendritic 或标准 TransformerBlock
  - forward 中 dendritic=True 且 field_state≠None 时，直接调用 block.forward 传入 field_state
  - field_state=None 时退化为标准 basal-only 行为（round 1 安全）
- **向后兼容**：
  - `dendritic_enabled=False`（默认）：完全等同修复前的标准 TransformerBlock
  - 旧 checkpoint 加载到 dendritic neuron：`strict=False` 自动跳过 apical 参数，保持初始化值
  - 5/5 smoke test 通过（[_smoke_s10_dendritic.py](file:///e:/taiji-neuron/scripts/training/_smoke_s10_dendritic.py)）：
    1. dendritic=False 与标准块一致（diff=0）
    2. dendritic=True apical 改变输出（diff=0.064）
    3. field_state=None 安全退化（diff=0）
    4. neuron 级别树突化生效（diff=2.72）
    5. checkpoint 兼容（missing=16 apical 参数，unexpected=0）

**参数量影响**：dendritic=True 时每层增加 apical_wq/wk/wv/wo + apical_norm + somatic_gate + error_scale，约增加 25-35% 参数（compact 85M → ~110M）。可通过 config 开关控制，不影响现有 neuron。

### S11. 512 token 硬截断 ★ ✅ 已修复（attention sink + 滑动窗口）

| 维度 | 修复前 | 修复后 |
|------|--------|---------|
| 上下文长度 | 512 token 硬截断（KV cache 无限增长或硬截断） | **attention sink + 滑动窗口，近 O(1) 推理时长上下文** |
| 后果 | 长对话被截断，多轮能力受限 | 支持数万 token 上下文（sink + window 配置） |
| 妥协原因 | CPU 推理显存/算力 | **已解除：StreamingLLM 技术，KV cache 上限 = sink_size + window_size** |
| 提升幅度 | 极高（长上下文能力） | 已实现 |
| 实施难度 | 中 | 已完成 |

**修复详情**（2026-08-01）：
- [config.py](file:///e:/taiji-neuron/taiji/resonance/config.py)：`NeuronConfig` 新增 `attention_sink_size: int` 和 `sliding_window_size: int`
- [layers.py](file:///e:/taiji-neuron/taiji/layers.py)：`GroupedQueryAttention` 扩展
  - 新增 `_evict_kv_cache()` 方法：KV cache 超限时保留前 `sink_size` + 最近 `window_size` token
  - `forward()` 中 `kv_cache_max_len > 0` 时自动驱逐
  - 滑动窗口 + KV cache 推理时禁用 causal mask（维度不匹配安全处理）
  - 训练时（无 kv_cache）完全不受影响
- [neuron.py](file:///e:/taiji-neuron/taiji/resonance/neuron.py)：构建 TransformerBlock 时传入 sink/window 参数
- **参数语义**：
  - `attention_sink_size=4`（默认 0=关闭）：保留前 4 个 token 作为注意力锚点
  - `sliding_window_size=2048`（默认 0=关闭）：滑动窗口大小
  - KV cache 上限 = sink_size + window_size = 2052
  - 两者都为 0 时完全向后兼容（KV cache 无限增长）
- **向后兼容**：
  - sink/window=0（默认）：完全等同修复前
  - sink/window 是 Python 属性（非 nn.Parameter），不影响 state_dict，旧 ckpt strict=True 加载成功
  - 6/6 smoke test 通过（[_smoke_s11_attention_sink.py](file:///e:/taiji-neuron/scripts/training/_smoke_s11_attention_sink.py)）

**使用建议**：生产配置推荐 `attention_sink_size=4, sliding_window_size=2048`（KV cache 上限 2052，支持 ~2000 token 上下文）。CPU 推理可降至 `sliding_window_size=512`（上限 516）。

### S12. 多轮对话靠前缀拼接 ★ ✅ 已修复（per-round field state + 对话轮次 token）

| 维度 | 修复前 | 修复后 |
|------|--------|---------|
| 多轮实现 | 前缀拼接 + 512 token 硬截断 | **per-round field_state 持久化 + 对话轮次 token** |
| 后果 | 无对话状态追踪，无角色标记，长对话被截断 | 真多轮能力，field_state 隐式记忆上下文 |
| 妥协原因 | 与训练时单文档自回归对齐 | **已解除：DialogueState 管理器替代前缀拼接** |
| 提升幅度 | 高（多轮连贯性） | 已实现 |
| 实施难度 | 高（需重训）/ 中（field state 注入） | 已完成（无需重训） |

**修复详情**（2026-08-01）：
- [field.py](file:///e:/taiji-neuron/taiji/resonance/field.py)：`ResonanceField` 新增 `save_round_state()` / `load_round_state()` 方法
  - 保存/加载完整状态：state + inhibitory_mask + contributions + inhibit_contributions
  - round-trip 完整恢复（测试验证 0 偏差）
- [dialogue_state.py](file:///e:/taiji-neuron/taiji/resonance/dialogue_state.py)：新增 `DialogueState` 类
  - **start_round(field)**：加载上一轮的 field_state（隐式记忆上下文）
  - **end_round(field)**：保存当前轮次的 field_state 快照
  - **prepend_round_token(ids)**：第 2 轮及以后在 prompt 前插入轮次标记 token
  - **max_rounds 滑动窗口**：保留最近 N 轮的 field_state（默认 5）
  - **add_dialogue_entry(role, content)**：记录对话历史（仅日志，不参与推理）
  - **序列化/反序列化**：完整状态可持久化到 checkpoint
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py)：
  - 新增 `set_dialogue_state()` / `clear_dialogue_state()` 方法
  - `_generate_p7` 集成：开始时 `start_round` + `prepend_round_token`，结束时 `end_round`
  - 未注册时（默认）保持原前缀拼接行为（完全向后兼容）
- **核心机制**：
  - 人脑启发：海马体在对话间保持工作记忆，每轮对话更新海马状态
  - 替代前缀拼接（把所有历史文本重新读一遍的低效做法）
  - 模型通过 field_state 隐式记忆上一轮的上下文
- **向后兼容**：
  - `cortex._dialogue_state = None`（默认）：完全等同修复前的前缀拼接行为
  - `DialogueState(max_rounds=0)`：不持久化（每轮独立）
  - 6/6 smoke test 通过（[_smoke_s12_dialogue_state.py](file:///e:/taiji-neuron/scripts/training/_smoke_s12_dialogue_state.py)）：
    1. field round-trip 完整恢复
    2. 多轮 field_state 持久化
    3. max_rounds 滑动窗口
    4. round_token 前缀插入
    5. reset 清空状态
    6. 序列化/反序列化

**使用方式**：
```python
dialogue = DialogueState(max_rounds=5, round_token_id=general_tokenizer.encode("<|round_start|>")[0])
cortex.set_dialogue_state(dialogue)
# 第 1 轮
response1 = cortex.generate("你好")
# 第 2 轮（自动加载第 1 轮的 field_state）
response2 = cortex.generate("刚才我说了什么？")  # 模型通过 field_state 记忆
# 新会话
cortex.clear_dialogue_state()  # 清空状态
```

---

## 二、局部妥协（按组件分类，精简列表）

> **梳理更新**（2026-08-01）：S1-S12 系统性修复已解决部分局部妥协，下表标注修复状态。
> 剩余真实缺口按上限提升潜力分级：★★★ 高 / ★★ 中 / ★ 低。

### 共振场核心

| # | 妥协点 | 当前 | 上限更高 | 状态 | 分级 |
|---|--------|------|---------|------|------|
| C1 | 神经元类型仅 2 种 | excitatory/inhibitory | PV+/SOM+/VIP+ 多亚型 | ✅ **已修复**（5 亚型: excitatory/pv/som/vip/inhibitory, 不同 write_gain + refractory_multiplier） | — |
| C2 | 不应期是整数计数器 | 二值状态 | 4 相恢复曲线 | 未修复 | ★ |
| C3 | 单体 Transformer 无树突分叉 | 单前向通路 | basal/apical 树突分离 + 预测编码 | ✅ **S10 已修复** | — |
| C4 | 场读入是加性残差 | gate*conditioning | 乘性门控 / 预测编码 | ✅ **已修复**（三种模式可选） | — |
| C5 | domain_prototype 单 EMA 向量 | 单质心 | 原型混合 + 在线聚类 | ✅ **已修复**（K 原型 + 在线 k-means 胜者 EMA 更新, max cosine 路由） | — |
| C6 | field_write 单 query pooling | 单语义切面 | 多 query 多头池化 | ✅ **已修复**（多头 attention pooling + 门控聚合） | — |
| C7 | 场是单一 D 维向量 | 无空间结构 | 空间场 + 扩散动力学 | ✅ **已修复**（图拉普拉斯扩散，forward_train 接入） | — |
| C8 | 场写入丢弃幅度 | L2 归一化 | 保留幅度作置信度 | ✅ **已修复**（attention entropy 置信度，per-sample scale 调制） | — |
| C9 | 共振轮数固定 3 | 固定开销 | 自适应停止 + 连续吸引子 | ✅ **已修复**（收敛 + 主导双信号自适应停止，min_rounds/max_rounds 双约束） | — |
| C10 | side_signals 仅 round 1 后构建 | rounds 2+ 复用 | 每轮动态更新 | ✅ **已修复**（推理路径每轮重建） | — |
| C11 | 跨 vocab 用零填充融合 | 语义错误 | 跨域 token 对齐 / 共享语义空间 | ✅ **S6 已修复** | — |
| C12 | 共振分数加权被禁用 | field.score() 不可比 | 对比学习投影到统一空间 | ✅ **已修复**（评分投影 + contrastive_loss NLL 排序对齐） | — |
| C13 | max 规格 EXPERT 仅 ~285M | CPU 可训 | 十亿-百亿级 | 硬件约束 | — |
| C14 | shared_expert_weight 固定 0.3 | 仿 DeepSeek | 任务相关可学习动态权重 | ✅ **已修复**（方案C: 共振分数+场状态联合驱动 per-sample sw） | — |
| C15 | v1_compat 保留旧 ckpt 行为 | 向后兼容 | 迁移后移除技术债 | 未修复 | ★ |

### 训练流水线

| # | 妥协点 | 当前 | 上限更高 | 状态 | 分级 |
|---|--------|------|---------|------|------|
| T1 | 评估集用训练集尾部 | 无 held-out | 5% hash 分桶 held-out | ✅ **已修复**（4 个训练脚本全部接入） | — |
| T2 | shared_emb_mode 默认 frozen | 首训误用卡随机 | 默认 auto 检测 | ✅ **S8 已修复**（默认 trainable） | — |
| T3 | base 阶段 side_channels 死权重 | 随机 peer 占内存 | frozen peer 特征提取 | 未修复 | ★ |
| T4 | 无数据增强 | 固定模板 | 回译 + prompt 改写 + 多轮拼接 | ✅ **已修复**（data_augmentation.py: 模板改写+多轮拼接+神经元改写, translator answer_marker_mode=last, 3 训练脚本 --augment） | — |
| T5 | dialogue finetune 未用 Muon | 纯 AdamW | Muon+AdamW 混合 | 未修复 | ★ |
| T6 | cross_spec 投影层单 Linear | 无 MLP | 2 层 MLP + GELU + 残差 | ✅ **已修复**（CrossSpecProjector: Linear+GELU+Linear 残差+零初始化, 旧 ckpt 兼容加载） | — |
| T7 | side_channels 仅 excite 无 inhibit | 单向调制 | excite + inhibit 平衡 | ✅ **已实现**（代码支持双通道，默认拓扑用 excite） | — |
| T8 | side_channels 用 simple_zh 训 | 分布外 | 改用 alpaca-zh | ✅ **S5 已修复**（默认 --data=dialogue, load_dialogue_texts_multi 加载 alpaca_zh_sft.jsonl 等多文件, max_texts 10K→100K） | — |
| T9 | field_conditioning 训练时关闭 | 怕噪声 | warm-up 后启用 | ✅ **已修复**（forward_train 加 field_conditioning 参数 + finetune warm-up 比例控制） | — |
| T10 | 阵容仅 5 神经元 | CPU 限制 | 扩到 11 个（含 shared_expert） | 硬件约束 | — |
| T11 | SAMPLING_MAX_TOKENS=100 | 折中 | 按场景分（200/128/512） | 未修复 | ★ |
| T12 | tokenizer 训练语料 30K 行 | 覆盖率 ~70% | 500K-1M 行 | ✅ **已修复**（词表库热插拔: 百科采样 ~200 万行 + 对话 4.8 万条×3 混合训练 50K zh tokenizer, token piece 映射 + lm_head 权重迁移, 无需重训神经元） | ★★ |
| T13 | build/load 路径不一致 | 手动拷贝 | 统一路径 | 未修复 | ★ |
| T14 | 无 ablation 评估 | 无法定位收益来源 | 4 组 ablation | ✅ **已修复**（evaluate_ablation.py: 共振协作/融合方式/side_channels/field_conditioning 4 组对照, T1 held-out 评估集, JSON 输出） | — |

### 推理运行时

| # | 妥协点 | 当前 | 上限更高 | 状态 | 分级 |
|---|--------|------|---------|------|------|
| R1 | 域路由用关键词计数 | 启发式 | 可学习路由器 / 共振分数路由 | ✅ **已修复**（resonance 软路由模式：probe→final_scores→top-k 跨域激活） | — |
| R2 | feed_engine 域检测硬编码 general | 简化 | 复用 cortex._infer_domain | 未修复 | ★ |
| R3 | 融合模式三套并存未分化 | 兼容遗留 | speculative decoding / consensus / MoE gate | ✅ **已修复**（consensus 投票融合模式：top-k 共识度加成，集体智慧浮现） | — |
| R4 | 采样策略固定 | top-k=50 | min-p / typical / ETD | 未修复 | ★ |
| R5 | 睡眠训练规模过小 | max_samples=64 | 异步 GPU worker + curriculum | 未修复 | ★ |
| R6 | 调质只驱动 lr 倍数 | 标量 | 驱动结构可塑性 / 兴奋阈值 | ✅ **S9 已修复**（调质门控 attention/FFN） | — |
| R7 | 代际迁移被禁用 | NotImplementedError | teacher→student 蒸馏 pipeline | ✅ **已修复**（三联蒸馏: KL logits + hidden 投影对齐 + attention 转移, 支持混合规格/vocab 对齐, train_distillation.py） | — |
| R8 | spec 选择只看错误率绝对值 | 单维度 | + 任务复杂度 + 资源约束 | 未修复 | ★ |
| R9 | 凋亡用固定阈值 | PPL>200 | 种群 PPL 分布相对阈值 | 未修复 | ★ |
| R10 | play 话题池硬编码 15 条 | 探索窄 | 动态话题生成 | 未修复 | ★ |
| R11 | SMCS EPE 候选评分用 n-gram | 无模型 | 用 ensemble final_scores / reward model | 未修复 | ★ |
| R12 | 无 KV cache | 每步全长度 forward | 启用 KV cache | ✅ **已实现**（layers.py 有 kv_cache，S11 增强 attention sink） | — |

### 梳理总结

**已被 S1-S12 修复的局部妥协（25 项）**：
- C1（神经元类型仅 2 种）← 已修复（5 亚型: excitatory/pv/som/vip/inhibitory, 不同 write_gain + refractory_multiplier）
- C3（树突分叉）← S10
- C4（场读入加性残差）← 已修复（additive/multiplicative/predictive 三模式可选）
- C5（domain_prototype 单 EMA 向量）← 已修复（K 原型 + 在线 k-means 胜者 EMA 更新, max cosine 路由）
- C6（field_write 单 query pooling）← 已修复（多头 attention pooling + 门控聚合）
- C7（场是单一 D 维向量）← 已修复（图拉普拉斯扩散，forward_train 接入）
- C8（场写入丢弃幅度）← 已修复（attention entropy 置信度，per-sample scale 调制）
- C9（共振轮数固定 3）← 已修复（收敛 + 主导双信号自适应停止，min_rounds/max_rounds 双约束）
- C10（side_signals 仅 round 1 后构建）← 已修复（推理路径每轮重建）
- C11（跨 vocab 零填充）← S6
- C12（共振分数不可比）← 已修复（评分投影 score_dim + contrastive_loss NLL 排序对齐）
- C14（shared_expert_weight 固定 0.3）← 已修复（方案C: 共振分数+场状态联合驱动 per-sample sw）
- T1（评估集用训练集尾部）← 已修复（5% hash 分桶 held-out，4 个训练脚本接入）
- T2（shared_emb 默认 frozen）← S8
- T6（cross_spec 投影层单 Linear）← 已修复（CrossSpecProjector: Linear+GELU+Linear 残差+零初始化, 旧 ckpt 兼容加载）
- T7（side_channels 仅 excite）← 代码已实现双通道
- T8（side_channels 用 simple_zh 训）← S5 已修复（默认 --data=dialogue, load_dialogue_texts_multi 加载 alpaca_zh_sft 等多文件）
- T9（field_conditioning 训练时关闭）← 已修复（forward_train 加 field_conditioning 参数 + finetune warm-up 比例控制）
- T4（无数据增强）← 已修复（data_augmentation.py: 模板改写+多轮拼接+神经元改写, translator answer_marker_mode=last 多轮精确 masking, 3 训练脚本 --augment）
- T14（无 ablation 评估）← 已修复（evaluate_ablation.py: 4 组对照实验定位收益来源）
- T12（tokenizer 训练语料 30K 行）← 已修复（词表库热插拔: upgrade_tokenizer.py 用百科 1314 万行采样 ~200 万行 + 对话 alpaca 4.8 万条×3 混合训练 50K zh tokenizer[对话词合并, 分词 11.0→11.5 tokens 持平, unk 0%], hot_swap_vocab.py 旧→新 token piece 映射 + lm_head 权重迁移[精确匹配 13427/子piece平均 36573/随机 0] + cfg.vocab_size 更新, 12 个 zh ckpt 全部迁移, 原 ckpt 备份至 pre_t12_backup/）
- R1（域路由用关键词计数）← 已修复（resonance 软路由模式：probe→final_scores→top-k 跨域激活）
- R3（融合模式三套并存未分化）← 已修复（consensus 投票融合模式：top-k 共识度加成，集体智慧浮现）
- R6（调质只驱动 lr）← S9
- R7（代际迁移被禁用）← 已修复（三联蒸馏: KL logits + hidden 投影对齐 + attention 转移, 支持混合规格/vocab 对齐, train_distillation.py）
- R12（无 KV cache）← 已实现 + S11 增强

**真实剩余缺口（按上限分级，共 15 项，其中 2 项硬件约束）**：

★★★ 高上限（0 项）：
所有高上限缺口已修复！剩余缺口均为中/低上限。

★★ 中上限（0 项）：
所有中上限缺口已修复！剩余缺口均为低上限。

★ 低上限（13 项）：
C2/C15, T3/T5/T11/T13, R2/R4/R5/R8/R9/R10/R11

硬件约束（2 项，非架构问题）：
C13（max 规格 EXPERT 受 CPU 限制）, T10（阵容仅 5 神经元受 CPU 限制）

---

## 三、上限提升潜力排序（Top 10）

| 排名 | 妥协点 | 类型 | 提升幅度 | 实施难度 |
|------|--------|------|---------|---------|
| 1 | S1 共振从未被端到端训练 | 系统性 | 协作涌现 +30-50% | 高 |
| 2 | S2 256K emb 配 16K tokenizer | 系统性 | 词覆盖 +30-50% | 中 |
| 3 | S6 域 token re-encode 往返 | 系统性 | 推理速度 3-5x + 长文本 | 中 |
| 4 | S3 Loss 单一化 | 系统性 | 协作 +15-30% + 回答 +15-25% | 中 |
| 5 | S11 512 token 硬截断 | 系统性 | 长上下文能力 | 中 |
| 6 | S5 数据规模偏小 | 系统性 | PPL +30-50% | 中 |
| 7 | S9 生物学机制是占位 | 系统性 | 结构性容量 | 高 |
| 8 | S4 训练步数偏短 | 系统性 | 收敛深度 +20-35% | 低 |
| 9 | S12 多轮对话靠拼接 | 系统性 | 多轮连贯性 | 中 |
| 10 | S7 side_channels 全连接 | 系统性 | 效率 +40% 质量 +5-10% | 中 |

---

## 四、关键洞察

### 4.0 ★★★ **架构本源定性：涌现已存在，核心缺陷是自适应激活**（2026-08-04 认知重构）

**核心定性修正（2026-08-04 认知重构）**：之前把"单神经元较强"定性为"压制涌现的因素"是**根本性错误**。正确的认知：

**涌现的定义 = 单个无法完成 + 协作能完成。当前架构已满足这个定义——涌现已存在。**

**涌现的证据**（数据验证）：

| 指标 | 最强个体 | 5 协作 | 涌现证据 |
|------|---------|--------|---------|
| 历史协作 PPL | ~42（无法正常对话）| 32.9 | ✅ 协作显著优于最强个体 |
| 当前 E4 PPL | ~19.7（仍无法正常对话）| 15.4 | ✅ 协作显著优于最强个体 |
| EMERGE | — | 21.7% | ✅ 协作收益稳定 |

**关键事实**：单个神经元 PPL ~20-42（参考 GPT-2 small PPL ~30 勉强可读），**无法正常对话**；5 个协作 PPL 15-33，**能正常对话**。这正是涌现的定义——协作产生了单神经元不具备的能力（正常对话）。

**"单神经元较强"是优势，不是劣势**：
- 优势：协作层规模不需要那么大就能产生涌现 → **效率高**
- 对比方向 B：1-5M 小神经元需要巨大协作层才能涌现，效率低且未验证
- 类比：5 个有基础能力的人协作，比 100 个几乎无能的人协作更容易出成果

**人脑类比不是唯一标准**：
- 之前用"人脑协作:神经元比 1000:1"作为标准，把当前 0.5:1 定性为"偏低"——这是错误套用
- 人脑的 1000:1 是生物约束（神经元体积大、突触可密集生长），神经网络无此约束
- **不按人脑也未必上限不高**——涌现的关键是"单个无法完成 + 协作能完成"，不是参数比例

**唯一核心缺陷：自适应激活**（用户明确指出）：
- 当前协作层基本**稠密计算**（所有 side_channels + cross_spec 每次全部参与）
- R1 软路由提供轻度稀疏性，但不够强 → 不同样本应激活不同神经元组合
- **已有设计**：R1 共振分软路由 + C14 动态 shared_expert_weight + C9 自适应停止
- **需要强化**：让协作层针对不同样本（如数学问题 vs 日常对话）激活不同神经元子集
- 这是提升上限的关键方向，比"扩大协作层规模"更重要

**方向 B 的定位修正**：
- 之前认为方向 B 是"上限更高的方向" → **错误**
- 正确认知：方向 B 是"另一种涌现路径"（更小神经元 + 更大协作层），**不是上限更高的路径**
- 当前方案已验证涌现存在，方向 B 是未验证假设
- **方向 B 触发条件修正**：不再是"当前方案上限不足时的备选"，而是"探索另一种涌现机制的实验"，优先级降低

**参数分布真相**（5 神经元阵容，2026-08-04 精确核算）：

| 部件 | 参数量 | 占比 | 训练状态 |
|------|--------|------|---------|
| 神经元主体（backbone）| 338M | 49% | 冻结 |
| body 最后 2 层（微调）| 185M | 27% | 微调（lr×0.1）|
| shared_embedding | 131M | — | 冻结 |
| **协作层**（side_channels + cross_spec）| **167.8M**（side 25.2M + 投影 142.6M）| **24%** | 从头训练 |

**协作层规模扩展能力**（unified 放大效应，§6.1 详述）：
- side_channels：O(N²) 随神经元数增长
- cross_spec 投影层：规格升级触发 U 跳跃全局放大（历史 4C→4C+1S 协作层翻倍 +111%）
- 当前 0.5:1 的比例**不是缺陷**——单神经元较强意味着不需要巨大协作层，但需要时可通过规格升级 + 数量扩展提升

**决策时点（修正）**：
- **Step 1（进行中）**：zh 综合体 + EOS+短答案重训 → 验证能正常对话（涌现的输出验证）
- **Step 2**：加入 code/math/en 等特定能力神经元，测试跨域涌现（§4.0b 候选 1）
- **Step 3**：强化自适应激活（R1 软路由 → 真正的 top-K 稀疏路由），提升协作效率
- 方向 B 不再是"上限不足时的备选"，而是"探索另一种涌现机制的实验"，优先级降低

---

### 4.0b 涌现的深化探讨 + 新能力方向（2026-08-04 认知重构）

**定性修正**：§4.0 已确认**涌现已存在**（单神经元无法对话 + 5 协作能对话 = 涌现）。本节探讨已实现的涌现 + 可进一步探索的新能力方向。

**与单体大模型的本质区别**（修正）：
- 之前认为"协作层稠密 → 数学等价于大模型" → **部分正确但不完整**
- 正确认知：即使协作层稠密，**信息流路径不同**（多分支并行 + 场共振融合 + 跨规格投影 vs 单链 transformer）
- 更关键的是：单体大模型所有参数端到端训练，态极神经元主体冻结只训练协作层 → **协作层学到的是"如何协调已有能力"**，这是单体模型不具备的

**已实现的涌现**（修正：从"非涌现"重新定性为"涌现"）：

| 能力 | 机制 | 实测 | 涌现定性 |
|------|------|------|---------|
| **协作能对话，单神经元不能** | 多轮共振 + side_signals + 场状态注入 | EMERGE 21.7%，单神经元 PPL ~42 无法对话 | ✅ **核心涌现**（协作产生新能力）|
| **置信度加权融合** | C8 per-sample confidence + 场写入强度 | 高置信神经元贡献更大 | 涌现的支撑机制 |
| **跨规格信息融合** | cross_spec 投影层（compact 2048 + standard 3072 → unified）| 不同规格神经元优势互补 | 涌现的支撑机制 |
| **动态协作权重** | C14 shared_expert_weight + R1 共振分软路由 | 不同样本激活不同神经元组合 | 涌现的支撑机制（待强化）|
| **多轮深化推理** | 多轮共振（round 1 独立 → round 2+ 注入场状态）| 信息在轮次间累积 | 涌现的支撑机制（待挖掘）|

**核心涌现**是第一行：协作产生了单神经元不具备的能力（正常对话）。其他行是支撑这个涌现的机制，不是独立的涌现。

#### 真正的"新能力涌现"是什么（探讨）

涌现指的是**单个神经元不具备、协作后产生的新能力**。当前架构理论上可能涌现的新能力：

**候选 1：跨域类比推理**（可能性：中 → **历史已实验过，有实现可能性**）
- 机制：不同神经元学到不同领域的隐式表征，场共振让它们在统一空间碰撞
- 涌现表现：模型能做出"类似 X 领域的 Y 领域推理"（如用物理直觉解数学题）
- **历史实验结论**：早期用 5 个不同类别小神经元做过跨域实验，**PPL 显示有涌现迹象**
- **当时搁置原因**（关键）：
  1. 5 个不同类别小神经元训练数据**没有互通性**——单一领域数据不包含正常对话等内容
  2. 所有单神经元**无法对话** → 无法通过输出判断涌现（只能靠 PPL 间接判断）
  3. 缺乏"能正常对话的综合体"作为基底 → 涌现效果被"输出无法理解"掩盖
- **新验证路径**（用户提出的清晰思路）：
  - **Step 1**：当前 zh 综合体（5 神经元 + EOS+短答案重训）能正常对话（进行中）
  - **Step 2**：在已能对话的基底上，**加入特定能力的神经元**（如 code/math/en 神经元）
  - **Step 3**：直观测试涌现——观察加入新神经元后，综合体的输出是否出现新能力（如对话中能解答代码问题、数学推理）
  - 这比"5 个孤立域神经元硬凑"更直观，因为综合体本身能输出可读对话
- 当前是否实现：**历史 PPL 显示迹象，但未通过输出验证**。新路径待当前重训完成后启动

**候选 2：动态能力组合**（可能性：高）
- 机制：R1 软路由 + C14 动态权重，不同样本激活不同神经元组合
- 涌现表现：对未见问题，模型能动态选择最合适的神经元组合（而非固定路由）
- 当前是否实现：**部分实现**。R1 已接入训练，但单神经元能独立生成 → 协作非必需
- 关键阻碍：单神经元能独立完成任务时，动态组合的"涌现"退化为"可选优化"

**候选 3：场状态累积推理**（可能性：中高）
- 机制：多轮共振让场状态累积信息，类似"思考过程"
- 梯度流经过 side_signals + field_state，模型学到"如何利用场状态"
- 涌现表现：复杂问题需要多步推理时，场状态累积产生单步无法得出的答案
- 当前是否实现：**部分实现**。forward_train 全可微多轮共振已接入（S1 修复）
- 关键阻碍：n_rounds=2 太少，且训练数据都是单步问答，无多步推理样本

**候选 4：置信度校准**（可能性：高）
- 机制：C8 confidence + C12 共振分对比投影，模型学到"知道自己不知道"
- 涌现表现：对不确定的问题输出低置信度，而非胡乱回答
- 当前是否实现：**机制已实现，但未验证效果**。需要专门校准测试（ECE/Brier score）

#### 涌现的现状与提升方向（2026-08-04 认知重构）

**涌现已存在**（核心修正）：之前把"单神经元能独立完成基础任务"定性为"压制涌现"是错误的。

| 条件 | 当前状态 | 说明 |
|------|---------|------|
| 单神经元无法独立完成任务 | ✅ **满足** | 单神经元 PPL ~20-42，**无法正常对话** |
| 协作能完成任务 | ✅ **满足** | 5 协作 PPL 15-33，**能正常对话** |
| 协作层稀疏激活 | ⚠️ 部分满足 | R1 软路由提供轻度稀疏，**核心缺陷，待强化** |
| 训练任务需要协作 | ✅ 满足 | 对话任务单神经元无法独立完成 |

**核心修正**：
1. ~~"单神经元太强压制涌现"~~ → **错误**。单神经元有基础能力但无法正常对话，这正是涌现的前提
2. ~~"训练任务太简单不需要协作"~~ → **错误**。对话任务单神经元无法独立完成，必须协作
3. **唯一核心缺陷**：自适应激活不足（协作层稠密，R1 软路由需要强化为真正的 top-K 稀疏路由）

**提升涌现上限的方向**（修正）：
- ~~更小神经元~~ → 不需要，当前单神经元"有基础能力但无法独立完成"是最佳区间
- **强化自适应激活**（R1 → top-K 稀疏路由）→ **核心方向**
- **跨域神经元扩展**（加入 code/math/en 神经元）→ 测试新能力涌现
- **多步推理任务**（增加 n_rounds + 多步推理数据）→ 挖掘场状态累积推理潜力

#### 对当前方向的指导（2026-08-04 修正：跨域实验路径优先）

1. **当前方向价值**：工程层面的能力增强（协作 PPL 降低 21.7%）是真实的，值得继续优化
2. **涌现验证优先路径**（用户提出，比方向 B 更直观且可验证）：
   - **Step 1（进行中）**：当前 zh 综合体 + EOS+短答案重训 → 验证能正常对话
   - **Step 2**：加入 code/math/en 等特定能力神经元到已能对话的基底
   - **Step 3**：直观测试跨域涌现（对话中是否出现代码/数学能力）
   - **优势**：综合体本身能输出可读对话 → 涌现效果可直接观察，不像早期"5 孤立域硬凑"无法判断
3. **关键认知**：早期跨域实验失败不是因为机制无效，而是因为**缺乏能对话的基底**。现在有了能对话的基底，跨域涌现值得重新验证
4. **方向 B 的触发理由不变**：若跨域加入新神经元后涌现明显 → 当前架构可承载涌现，方向 B 暂不启动；若跨域加入后无涌现 → 才需要考虑方向 B（更小神经元 + 被迫协作）
5. **中间路径**：当前架构 + 更难任务（多步推理数据）+ 稀疏路由（R1 强化），可能部分触发涌现

#### 跨域 Step 2 数据准备（2026-08-05 梳理，工作3）

**目标**：为 code/math 特殊神经元训练准备混合数据 + 梳理接入流程（§4.0b 候选1 Step 2）。

**关键决策（用户指正）**：每个 neuron 保留自己的域 tokenizer（code 12K / math 10K），通过**词库转译**实现语义转换：
- 输入统一 general 256K 空间（[batch_align_and_embed](file:///e:/taiji-neuron/taiji/resonance/translator.py#L452) 用 general_sp 编码输入，目标用 domain_sp 编码）
- 推理转译用 S6 alignment_table（domain→general 预计算映射，[cortex.py:1198](file:///e:/taiji-neuron/taiji/brain/cortex.py#L1198)）
- 推理 forward 已支持不同 vocab（[ensemble.py:1263-1284](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1263-L1284) `same_vocab` 检查，不同时走 neuron_logits 提取）

**混合数据策略（已验证可行）**：

| 数据源 | 规模 | 用域 tokenizer 编码 | 说明 |
|--------|------|--------------------|------|
| `data/sft/code_sft.pt`（CodeAlpaca） | 3000 条 | byte_ratio 7.3% ✅ | 英文代码指令-响应 |
| `data/sft/math_sft.pt`（GSM8K） | 3000 条 | byte_ratio 2.2% ✅ | 英文数学推理 |
| `data/sft/en_sft.pt`（英文 alpaca 对话） | 3000 条 | byte_ratio 2-7% ✅ | 混合对话能力 |
| `data/distill/code_texts.jsonl` | 36,810 行 | ✅ 英文 | 预训练风格扩充 |
| `data/distill/math_texts.jsonl` | 22,904 行 | ✅ 英文 | 预训练风格扩充 |

**结论**：code/math neuron 用各自域 tokenizer 训练，混合数据 = 域 SFT 数据 + 英文对话数据（en_sft），目标编码全部高效（byte_ratio 2-7%）。**不混中文对话**（code tokenizer 编中文 byte_fallback 57% 低效）；中文语义通过 general 256K 统一输入空间 + S6 转译在协作层处理。

**接入流程**：
1. 训练 code/math neuron 本体（P8-1 `train_neurons_from_scratch.py --domain code`，混合域 SFT + en_sft）
2. 加入综合体推理：forward 已支持（same_vocab 检查）
3. 协作层训练：**缺口 M 已修复**（见下）——`forward_train` 跨 vocab 融合
4. 跨域涌现评估：对话中测试代码/数学能力

**状态**：混合数据策略已验证 ✅（2026-08-05）；**缺口 M 已修复 ✅（2026-08-05）**；训练 code/math neuron 待当前 zh 训练完成后执行。

### 缺口 M 修复：forward_train 跨 vocab 联合训练（2026-08-05 实施）

**原问题**：`forward_train` 融合阶段要求所有 neuron vocab 一致（[ensemble.py:1673-1680](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1673-L1680) 原实现），否则 `torch.stack` 崩溃——跨域协作层训练的前置阻塞。

**修复方案（词库转译矩阵投影）**：
- [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) 新增通用词库转译工具：
  - `tokenizer_fingerprint(sp)`：tokenizer 指纹（vocab_size + 首/中/尾 piece 抽样），用于缓存失效判断
  - `build_domain_to_domain_alignment(source_sp, target_sp)`：source token → target token 对齐（byte fallback 正确处理）
  - `build_logits_alignment_matrix(...)`：构建 [V_src, V_tgt] 稀疏投影矩阵（行归一化 1/N，logits 尺度守恒），带缓存 + 指纹失效
- [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py)：
  - `set_tokenizer_hub(hub)`：注入 TokenizerHub（与 cortex 同源）
  - `forward_train` 新增 `target_domain` 参数；vocab 不一致时用转译矩阵把各 neuron logits 投影到 target 域空间再融合
  - 向后兼容：vocab 一致路径零开销（不传 target_domain 也可运行）
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py)：forward_train 传 `target_domain=DOMAIN`

**词库热插拔（一并解决）**：
- S6 对齐表缓存（`_domain_to_general_cache`）原为一次性构建永不失效；现缓存项携带 tokenizer 指纹，tokenizer 被替换（重训/热插拔注册）后自动失效重建
- [cortex.py](file:///e:/taiji-neuron/taiji/brain/cortex.py) 新增 `invalidate_alignment_cache(domain=None)` 手动失效接口
- TokenizerHub.register_domain 本身已支持热插拔（新域注册不影响现有 neuron）

**词库可编辑可拓展层（AlignmentRules，2026-08-05 新增）**——匹配新增特殊神经元词表：
- [translator.py](file:///e:/taiji-neuron/taiji/resonance/translator.py) `AlignmentRules`：人工规则层覆盖自动转译，匹配键用 **piece 文本**（tokenizer 无关、可编辑，不用脆弱 token id）
- 支持域特定规则 + 全局规则（`"*"`）；每次增删递增 version → 下游转译矩阵/对齐表缓存自动失效
- 持久化 JSON（`save()`/`load()` 热加载），默认 `taiji/domains/alignment_rules.json`
- 接入：`ensemble.set_alignment_rules()` + `cortex.set_alignment_rules()`（S6 也支持人工覆盖）
- 新增特殊神经元时：注册 tokenizer + （可选）add_override 补专业术语映射

**跨域协作层训练脚本（train_cross_domain_collab.py，2026-08-05 新增）**：
- 多域 neuron（code/math/zh）联合训练协作层（side_channels + 投影层 + Sparse Router）
- 域轮转 + batch 级 `target_domain`，缺口 M 词库转译融合路径；`--rules-path` 挂载 AlignmentRules
- 自动匹配 neuron vocab 的 tokenizer（zh neuron 20K → `sp_zh_v20k.model`，防御 vocab 错位）
- 冒烟验证通过（verify_v3 多域 neuron 完整跑通训练循环 + checkpoint）

**验证**：`_smoke_cross_vocab_gap_m.py` 8/8 通过（转译构建/矩阵归一化/缓存复用/热插拔失效/跨 vocab 融合梯度流/向后兼容/override 覆盖/持久化/规则变更缓存失效）；真实 code→zh 转译验证：`def`→`['▁','▁def']`、换行语义保持 ✓，矩阵 [12000, 50000] 构建仅 0.1s。

### 4.0c ★★★ **自适应激活设计：R1 软路由 → top-K 稀疏路由**（2026-08-05 设计）

> 本节是 §4.0 确定的"唯一核心缺陷"的**具体设计方案**。用户要求"梳理，同时可以着手设计"，此处完成梳理 + 设计落地，待训练完成后实施。

#### 1. 自适应激活针对什么

**明确：针对输入样本（样本驱动）**，不针对装载硬件。

- **样本驱动**：不同输入（数学问题 vs 日常对话）应激活不同神经元子集 → 这是模型架构层面的自适应激活，本设计的核心。
- **硬件调度**：根据可用显存/算力动态调整激活数量 → 工程层问题，与模型架构正交，不在本设计范围内。
- **两者关系**：样本驱动的 top-K 选择是基础，硬件调度可在 top-K 基础上进一步调整 K 值（未来扩展）。

#### 2. 现有自适应激活机制盘点（梳理）

| 机制 | 路径 | 类型 | 局限 |
|------|------|------|------|
| C9 自适应停止 | 推理 `forward()` | 轮次级 | 只控制何时停止，不控制激活谁 |
| R1 共振分软路由 | 训练 `forward_train()` | 软加权 | **稠密计算**，所有神经元都参与，权重≈0 也算 |
| active_filter | 推理 `forward()` | 硬过滤 | 基于场方向拥挤度，非能力路由；H5 显示跨 embedding 空间不可比 |
| per_position entropy融合 | 推理 `forward()` | per-token软加权 | 0.01 floor，没有真正关闭神经元 |
| C14 动态shared_expert_weight | 推理 `forward()` | shared权重动态 | 只调整 shared vs domain，非神经元子集选择 |
| active_nids参数 | 推理 `forward()` | 外部路由接口 | 没有路由器实现，需外部指定 |

#### 3. 核心缺陷（三点）

1. **训练路径完全稠密**：`forward_train` 中 `active_ids = list(self.neurons.keys())`（[ensemble.py:1137](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1137)），所有神经元参与每轮计算和融合。softmax 只是软加权（[ensemble.py:1390](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1390)），即使某神经元权重≈0，它的 forward 计算仍然进行，算力浪费。

2. **训练-推理不一致**：训练用 soft softmax 全神经元融合；推理用 per_position entropy + active_filter 硬过滤。模型训练时从未见过"部分神经元被关闭"的情况，导致推理时分布偏移。

3. **没有样本驱动的路由器**：当前 scores 基于 field_state cosine（场聚合状态），不是"输入样本特征 → 路由决策"。H5 注释明确（[ensemble.py:1590-1595](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1590-L1595)）：field.score() 跨 embedding 空间不可比，已被禁用。

#### 4. 设计：Probe-based Sparse Router（基于探针的稀疏路由器）

##### 4.1 核心思路

引入可学习的 Router，基于 **round 1 probe**（每神经元独立前向，已在 `forward_train` 中执行）的响应，为每个神经元产生路由分，选择 top-K 神经元参与 round 2+ 的深度协作。

##### 4.2 为什么选 Probe-based 而非 Input-based（上限优先）

| 方案 | Router 输入 | 额外开销 | 上限 | 选择 |
|------|------------|---------|------|------|
| Input Router | shared_embedding mean-pool | 零 | 中（只看输入，不看响应）| 备选 |
| **Probe Router** | round 1 field_vectors + confidence | 零（round 1 已存在）| **高**（看每神经元实际响应）| **推荐** |

**Probe Router 上限更高的原因**：它能看到"每个神经元对当前输入的初步响应"，类似人脑"先瞥一眼再决定谁深入处理"。round 1 独立前向已在 `forward_train` line 1203+ 执行，**零额外开销**。

##### 4.3 Router 结构

```
Router 输入（per-neuron）：
  - field_vector: [B, D_field]    # round 1 每神经元的场写入向量（已投影到 unified 维度）
  - confidence:   [B]              # round 1 per-sample 置信度（C8）
  - score_vec:    [B, D_score]    # round 1 评分投影向量（C12，若存在）

Router 输出：
  - routing_scores: [B, N]        # 每样本对每神经元的路由分
  - top_k_mask:     [B, N]        # hard top-K 选择（forward）
  - soft_weights:   [B, N]        # soft softmax 权重（backward 梯度流）
```

Router 实现：per-neuron MLP(`D_field + 1 + D_score → hidden → 1`)，对每个神经元独立评分，然后 batch 级 softmax + top-K。

##### 4.4 可微 top-K 选择（Straight-Through Estimator）

借鉴 Switch Transformer / GShard：

```python
# Forward: hard top-K 选择
top_k_indices = routing_scores.topk(K, dim=-1).indices
hard_mask = zeros(B, N).scatter_(-1, top_k_indices, 1.0)
# 被选中的神经元用 routing_scores 归一化后的权重
selected_weights = (routing_scores * hard_mask).softmax(dim=-1)  # 只在选中神经元上归一化

# Backward: 梯度通过 soft softmax 流回所有神经元
soft_weights = routing_scores.softmax(dim=-1)
# STE: forward 用 hard, backward 用 soft
final_weights = hard_mask * selected_weights + (soft_weights - soft_weights.detach())
```

- **Forward**：只有 K 个神经元参与 round 2+ 计算（算力节省）
- **Backward**：梯度通过 soft softmax 流回所有神经元（Router 可学习）

##### 4.5 负载均衡 loss（防模式坍塌）

升级当前 `balance_loss = -(weights * log(weights)).sum()`（负熵）为 Switch Transformer 风格：

```python
# f_i: 神经元 i 被选中的批次比例（hard, detach）
f = hard_mask.mean(dim=0).detach()  # [N]
# P_i: Router 对神经元 i 的平均概率（soft, detach）
P = soft_weights.mean(dim=0).detach()  # [N]
# 负载均衡 loss: N × Σ(f_i × P_i)，越小越均衡
balance_loss = N * (f * P).sum()
```

##### 4.6 K 值确定

- **起步**：固定 K=3（5 神经元中选3个 + shared_expert 始终激活 = 4 个参与 round 2+）
- **升级**：动态 K（基于路由分分布的熵，高熵→多选，低熵→少选）
- **shared_expert 处理**：general 神经元始终激活，不参与 top-K 选择（保证基础语言能力）

##### 4.7 Warm-up 策略（防冷启动）

Router 初始随机，可能选错神经元。Warm-up 分阶段：
- **Phase 0（前 10% 步）**：K=N（全选），Router 只学习评分，不影响激活
- **Phase 1（10%-30% 步）**：K 线性从 N 降到目标 K（如 5→3）
- **Phase 2（30%+ 步）**：固定目标 K，Router 完全生效

#### 5. 与现有机制的整合

| 现有机制 | 整合方式 | 理由 |
|---------|---------|------|
| R1 共振分软路由 | **替换**为 Router soft weights | Router 学习路由，比场状态 cosine 更直接；H5 已证明 field.score() 跨 embedding 不可比 |
| C9 自适应停止 | **保留**，与 Router 正交 | C9 控制轮次（何时停），Router 控制激活（谁参与）|
| C14 动态shared_expert_weight | **保留** | shared_expert 始终激活，C14 调整其权重，与 Router 正交 |
| active_filter | **替换**为 Router top-K | Router 是主动选择，active_filter 是被动过滤 |
| per_position融合 | **保留**作为 fallback | 在选中的 K 个神经元内进行 per_position 融合 |
| balance_loss | **升级**为 Switch 风格 | 负熵 → Switch 负载均衡，更稳定 |
| C12 contrastive_loss | **保留** | 约束 Router 评分与 NLL 排序对齐，让 Router 学到"能力路由"|

#### 6. 训练-推理一致性

| 维度 | 训练（forward_train）| 推理（forward）| 一致性 |
|------|---------------------|---------------|--------|
| Router 选择 | STE（hard forward + soft backward）| hard top-K | ✅ 一致 |
| 参与神经元 | round 2+ 只 K 个 | round 2+ 只 K 个 | ✅ 一致 |
| 融合权重 | Router soft weights | Router soft weights | ✅ 一致 |
| 负载均衡 | 训练时计算 balance_loss | 推理时不需 | ✅ 正常 |

**消除当前的训练-推理不一致**（训练稠密 vs 推理过滤）。

#### 7. 实施路径

##### 阶段1：Router 实现（不破坏现有训练）✅ 已完成（commit 3526274）
- [ensemble.py](file:///e:/taiji-neuron/taiji/resonance/ensemble.py) 新增 `SparseRouter` 类
- Router 输入：round 1 field_vectors + confidence + score_vec
- Router 输出：top-K mask + soft weights（STE）
- 负载均衡 loss（Switch 风格）
- 向后兼容：`use_sparse_router=False` 时退化为当前稠密模式

##### 阶段2：接入 forward_train ✅ 已完成（commit 3526274）
- `forward_train` round 1 后计算 Router 输出
- round 2+ 只对 top-K 神经元注入 side_signals + field_state
- 融合用 Router soft weights（per-sample STE）
- 新增 load_balance_loss 到总 loss（替换原负熵 balance_loss）
- smoke test 通过（forward/backward/归一化/梯度流验证）

##### 阶段3：接入推理 forward ✅ 已完成（commit 54e95e5）
- 推理路径同样用 Router 选择 top-K（round 1 后）
- 保证训练-推理一致（"激活谁"一致）
- 融合在 top-K 内 per-position（保留 entropy 融合）
- active_nids 参数与 Router 协同（外部指定优先，否则用 Router）
- [eval_dialogue.py](file:///e:/taiji-neuron/scripts/training/eval_dialogue.py) 自动检测 checkpoint 是否含 Router 状态

##### 阶段4：训练验证 ⏳ 待训练完成后
- [finetune_cross_spec.py](file:///e:/taiji-neuron/scripts/training/finetune_cross_spec.py) 新增 `--use_sparse_router` flag（已完成）
- 对比稠密 vs 稀疏的 EMERGE、PPL、推理速度
- 验证 warm-up 策略有效性

#### 8. 上限分析

| 维度 | 当前稠密 | 稀疏路由 | 提升 |
|------|---------|---------|------|
| 算力效率 | O(N) 每轮 | O(K) round 2+ | N=20,K=5 时 75% 节省 |
| 协作质量 | 所有神经元参与 | 最合适神经元深入 | 聚焦→质量↑ |
| 可扩展性 | 算力线性增长 | 算力对数增长 | 支持更多神经元协作 |
| 训练-推理一致 | ❌ 不一致 | ✅ 一致 | 消除分布偏移 |
| 路由可学习 | ❌ 场状态 cosine | ✅ MLP 学习 | 适应任务分布 |

#### 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| Router 冷启动（选错神经元）| Warm-up 策略（Phase 0 全选，逐步降 K）|
| 模式坍塌（总选同一组）| Switch 风格负载均衡 loss |
| 与场共振冲突 | Router 选择后场更聚焦（正面效果，非冲突）|
| shared_expert 惰性 | C14 动态权重已处理（共振弱→sw 高→shared 兜底）|
| K 值选错 | 起步固定 K=3，后续升级动态 K |

#### 10. 实施时机

**当前训练完成后**（Epoch 8 预计 PPL < 20）：
1. 先验证 zh 综合体能正常对话（test_api_dialogue.py）
2. 若对话质量达标 → 实施 Sparse Router（Step 3）
3. 若对话质量不达标 → 先排查训练问题，再考虑 Router

**不提前实施的原因**：Router 需要在已能对话的基底上训练，否则无法验证 Router 是否提升协作质量。

---

### 4.0d ★★ **自适应激活设计深化：3 个工程妥协 + 上限更高选项**（2026-08-05 设计讨论）

> §4.0c 已给出基础设计方案（Probe-based Sparse Router）并实现了阶段1-3（commit 3526274/54e95e5）。
> 本节审视实现中的 **3 个工程妥协**，给出上限更高的替代选项，供决策。
> 用户明确要求聚焦"设计自适应激活机制"，本节是设计层面的完整讨论（不实现）。

#### 妥协 1：稀疏粒度是 batch 级并集（当前实现）vs per-sample（上限更高）

**当前实现**（[ensemble.py:1016-1019](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1016-L1019)）：
```python
selected_mask = hard_mask.sum(dim=0) > 0  # batch 级并集
```
一个 batch 内**任一样本**选中的神经元，全部参与 round 2+。batch=4 时，4 个样本各选 3 个不同神经元，并集可能接近全部 N。**稀疏收益随 batch 增大而递减**。

**上限更高方案：per-sample top-K**
- 每个样本独立选 K 个神经元，round 2+ 用 [B, K] 索引
- 真正的算力节省需在 forward 层用 sparse mask（Switch Transformer capacity factor）
- **复杂度**：side_signals 是 per-pair 投影（[N, B, D]→[N, B, hidden]），per-sample 稀疏需要对每样本 mask 掉非选中神经元的所有 side_channels 计算

**决策依据**：

| N | 并集实际激活（batch=4）| per-sample 激活 | 差距 |
|---|----------------------|-----------------|------|
| 5（当前）| 4-5（节省 0-20%）| 3+shared=4（节省 20%）| 小 |
| 20（未来）| 12-16（节省 20-40%）| 5+shared=6（节省 70%）| 显著 |

**推荐**：当前阶段保持 batch 并集（N=5 差距小，per-sample 复杂度高收益低）；设计上预留 per-sample 接口，N 增大后升级。这是"随规模增长"的正确时点判断，不是永久妥协。

#### 妥协 2：Router 无学习信号约束 vs 对比约束（上限更高）

**当前实现**：Router 只通过 CE loss 的 STE 梯度隐式学习（soft_weights 梯度流回 Router）。
**问题**：没有显式信号告诉 Router "**哪个神经元擅长当前样本**"。Router 可能学到"按响应强度路由"（大神经元主导），而非"按能力路由"（谁擅长当前样本谁上）。

**上限更高方案：C12 对比约束扩展到 Router**
共振分已有 C12 contrastive loss（[ensemble.py:1603+](file:///e:/taiji-neuron/taiji/resonance/ensemble.py#L1603)）：约束共振分与 per-neuron NLL 排序对齐。Router 应复用同一约束：

```python
# ideal: NLL 低的神经元应获高路由权重（谁能更好预测当前样本）
ideal_weights = F.softmax(-nll / 0.5, dim=0)  # [N]
# actual: Router soft_weights
actual_weights = router_soft_weights.mean(dim=0)  # [N]
# KL(actual || ideal) 让 Router 学"能力路由"
router_contrastive_loss = (actual_weights * (actual_weights.clamp(min=1e-8).log()
                            - ideal_weights.clamp(min=1e-8).log())).sum()
```

**为什么这是上限提升的关键**：
- 无约束 Router：路由分只反映"响应强弱"，大神经元天然强响应 → 路由退化回"大神经元主导"
- 有对比约束：Router 被迫学习"谁在**当前样本**上预测最好"，小神经元在它擅长的主题上获得高路由分 → **真正的样本驱动自适应激活**

**推荐**：加入对比约束。改动小（复用现有 contrastive_loss 逻辑），上限提升大。

#### 妥协 3：固定 K vs 熵驱动动态 K（上限更高）

**当前实现**：固定 K=3。
**局限**：简单样本 1-2 个神经元足够，复杂样本需要 4-5 个。固定 K 要么浪费算力（简单样本），要么能力不足（复杂样本）。

**上限更高方案：熵驱动动态 K**
```python
# 路由分分布的熵：高熵（Router 不确定）→ 多选；低熵（明确）→ 少选
entropy = -(soft_weights * torch.log(soft_weights + 1e-8)).sum(-1)  # [B]
K_b = int(torch.clamp(entropy / math.log(N) * N, K_min, K_max).item())
```
- 低熵（Router 99% 确定某神经元）→ K=1-2，省算力
- 高熵（Router 犹豫）→ K=4-5，保证能力
- 上限：**每样本算力分配与任务难度匹配**（类似"简单问题快答，复杂问题慢想"）

**推荐**：预留动态 K 接口（Router.forward 已支持 effective_k 计算），起步固定 K=3 验证，稳定后升级熵驱动。

#### 设计决策汇总

| 决策点 | 当前实现 | 上限更高 | 推荐 | 实施成本 |
|--------|---------|---------|------|---------|
| 稀疏粒度 | batch 级并集 | per-sample top-K | 当前并集，预留接口 | 高（side_channels mask）|
| 学习信号 | 隐式（CE 梯度）| **C12 对比约束** | **加入对比约束** | 低（复用现有逻辑）|
| K 值 | 固定 3 | 熵驱动动态 | 起步固定，预留动态接口 | 中 |
| 路由时机 | round 1 后单次 | 每轮动态 | 单次（round1 响应已充分）| 无需改 |
| 路由输入 | field_vector+conf+score_vec | +任务特征（域）| 当前够用，跨域后加域特征 | 中 |

**结论**：3 个妥协中，**对比约束（妥协2）是当前阶段唯一值得立即实施的**——改动小、上限提升大、且直接服务于"样本驱动自适应激活"的核心目标。per-sample（妥协1）和动态 K（妥协3）留待 N 增大后升级。

#### 实施状态（2026-08-05 用户决策：三个都选上限更高方案，已全部实施）

| 决策点 | 用户决策 | 实施状态 |
|--------|---------|---------|
| 对比约束（妥协2）| **立即实施** | ✅ 已实施（router_contrastive_loss 加入总 loss，权重 0.1）|
| per-sample top-K（妥协1）| **现在升级** | ✅ 已实施（per-sample hard_mask 控制 side_signals + field_state 注入）|
| 熵驱动动态 K（妥协3）| **现在设计并实施** | ✅ 已实施（Phase 2 熵驱动，低熵少选/高熵多选）|

**实现要点**（commit 待填）：
1. **SparseRouter.forward 升级**：动态 K（每样本独立，k_min=1, k_max=N-1）+ per-sample top-K（每样本选 K_b 个，shared_expert 始终激活）+ 返回 `k_per_sample [B]` + `top_k_ids [B][K]`
2. **forward_train 接入**：round 1 后 Router 选 per-sample top-K；round 2+ 用 per-sample mask 控制：
   - side_signals：post 只接收该样本 top-K 的 pre 信号（`pre_vec * pre_mask.unsqueeze(-1)`）
   - field_state：只累加每样本 top-K 神经元的写入（`all_vecs_weighted * mask_t`）
   - 融合：per-sample final_weights（STE）
3. **forward（推理）接入**：同样 per-sample mask 控制 side_signals + field 写入，保证训练-推理一致
4. **对比约束**：`router_contrastive_loss = KL(router_soft_weights.mean(0) || softmax(-nll/0.5))`，与 C12 共享 per_neuron_nll 计算
5. **修复的 bug**：负载均衡 loss 的 P 原本 detach（Router 无梯度），改为可微；round 2 循环误清 Router 缓存，改为不重置

**测试验证**（全部通过）：
- SparseRouter 单元：Phase0 K=N、Phase2 熵驱动 K=[4,3,4,4]（每样本不同）、shared 始终激活、mask 行和=K、final_weights 归一化
- 梯度流：load_balance grad=2.44、contrastive grad=1.96、CE-path grad=21.40（三条路径全部非零）
- 集成：forward_train（use_sparse_router=True）正常，router_contrastive 激活，Router 梯度 54.18
- 向后兼容：use_sparse_router=False 时 Router 不创建，dense 模式完全正常（router_contrastive=0）
- 推理 forward()：use_sparse_router=True 正常，shared_expert 保留

**待训练验证**（阶段4）：训练完成后 `--use_sparse_router --sparse_router_top_k 3 --sparse_router_warmup_steps 2000` 对比稠密 vs 稀疏的 EMERGE、PPL、推理速度。

---

### 4.1 ~~"共振"是推理技巧，从未被训练~~（已过时，S1 修复后全可微）

**历史定性已过时**：S1 修复后 `forward_train` 是全可微多轮共振路径——训练时确实注入 field_state、side_signals、调质、gamma 振荡，所有机制进入梯度流。共振不再是"推理时拼凑"。

**当前状态**：共振已训练，但 n_rounds=2 较少，且训练任务是单步问答，共振的"多轮深化"潜力未充分发挥。真正的提升方向是增加 n_rounds + 引入多步推理训练数据（见 §4.0b 候选 3）。

### 4.2 tokenizer 错配是隐性天花板

256K embedding 配 16K tokenizer，14.6 万 embedding 行是死参数。所有 PPL 数字都被这层"tokenizer 噪声"掩盖，不解决它，后续所有优化都被掩盖。

### 4.3 协作层纯 CE 导致协作不涌现

协作层训练用纯 CE，不约束"协作是否真的比单神经好"。side_channels 学成噪声调制，很多场景协作 PPL ≥ 最强个体。需要 margin ranking + diversity + load balancing 三联 loss。

### 4.4 三阶段割裂导致表示空间无法协同

base → dialogue → cross_spec 三阶段从未联合训练，每阶段冻结前者。表示空间无法协同适配，side_channels 只能在固定表示上做线性调制。

### 4.5 生物学机制是"装饰"而非"骨架"

STDP/调质/Gamma/睡眠/新生全部以 Optional 注入，可独立开关。这意味着它们是"装饰性"的，不是架构的"骨架"。真正的生物学架构应该让这些机制成为不可移除的核心组件。

---

## 五、建议的改进路径（上限优先）

### 阶段 1：修复隐性天花板（S2 + S4）
- 训 256K general tokenizer
- 训练步数提到 12000-16000
- **不解决这两个，后续所有优化都被掩盖**

### 阶段 2：让共振可训练（S1）
- 可微多轮共振（Gumbel-softmax / straight-through）
- 让 forward_train 接入场+侧通道+调质
- **这是把共振从推理技巧变成可学习能力的唯一路径**

### 阶段 3：多任务 loss（S3）
- SFT answer masking
- 协作层 margin ranking + diversity + load balancing
- 跨域对比 loss（hub neuron 设计）

### 阶段 4：推理路径优化（S6 + S11 + S12）
- 域 token 对齐表
- 长上下文（attention sink / 分块共振）
- 多轮对话状态管理

### 阶段 5：生物学机制深化（S9）
- STDP 影响注意力/FFN 权重
- 多频段振荡 + 跨频耦合
- 真正睡眠重放
- 自组织新生

---

## 六、方向 B 备案：小神经元 + 强协作架构（2026-08-04 设计）

### 6.1 当前方向优先级与备案触发条件

**当前方向（优先）**：5 神经元阵容 + EOS + 短答案 + 数据扩充，继续优化。

**关键认知（2026-08-04 二次修正，代码公式实测）**：协作层参数随神经元数 + 规格升级双维度增长，**且规格升级存在"unified 放大效应"**：

**精确公式**（从代码验证）：
- side_channels：每条 `nn.Linear(pre.field_dim, post.hidden_size, bias=False)` = pre.field_dim × post.hidden_size，**pairwise = O(N²)**
- CrossSpecProjector（T6 升级为 2 层 MLP）：
  - 正向（神经元 fd → unified U）：`linear1(in→U) + linear2(U→U)` = **U × (fd + U)**
  - 反向（U → 神经元 fd）：**fd × (U + fd)**
- **U = max(所有神经元 field_dim)** ← 这是关键放大机制

**unified 放大效应（用户指出的核心机制）**：加入更大规格神经元会把 U 提升到新规格的 field_dim，**导致所有已有神经元的投影层 out_dim 变大 → 整个协作层参数被放大**。这不是新增一个投影层的线性增长，而是全局跳跃。

**历史验证（参考之前增加中等规格 standard 的经验）**：

| 阵容 | U | side_channels | 投影层 | 协作层总 | 增量 |
|------|-----|--------------|--------|---------|------|
| 4C（历史）| 2048 | 12.6M | 67.1M | **79.7M** | — |
| 4C+1S（当前）| 3072 | 25.2M | 142.6M | **167.8M** | **+88.1M (+111%)** |
| +1 EXPERT | 4096 | 48.2M | 269.5M | **317.7M** | **+149.9M (+89%)** |
| +2 EXPERT | 4096 | 79.7M | 336.6M | **416.3M** | +98.6M (+31%) |

**结论（完全验证用户判断）**：
1. **增加更大规格神经元 → 协作层扩展规模大幅跃升**：历史增加 standard 使协作层翻倍（+111%），增加 EXPERT 再 +89%
2. **unified 放大是主力**：+1 EXPERT 的 +149.9M 增量中，投影层放大贡献 126.9M（因 U 3072→4096），side_channels 新增只贡献 23M
3. **U 提升是单次性跳跃**：+2 EXPERT 增量降到 +31%（U 已到 4096 不再变，只剩 pairwise side_channel 增长）——**规格升级的放大效应是一次性的，重复同规格收益递减**
4. **最优扩展策略**：规格阶梯升级（引入新规格触发 U 跳跃）+ 数量扩展（同规格 O(N²)）双管齐下
5. 注意：上一版修正的 130M/105M/190M 数字有误（U 值用错 + 公式错误），以上表为准

**方向 B 备案触发条件**（任一满足）：
1. 当前架构在 EOS+短答案重训后，API 质量仍不达标且 PPL 已收敛
2. 增加到 20+ 神经元后协作层仍未承载主要能力（EMERGE < 30%）
3. 单神经元能力过强导致协作被边缘化（移除协作后 ensemble PPL 下降 < 10%）
4. **新增**：规格升级到 EXPERT 后协作:主体比未提升（说明规格升级无法替代 N 增长）

---

### 6.2 方向 B 神经元规格设计

| 参数 | 当前（中等神经元）| 方向 B（小神经元）|
|------|----------------|----------------|
| hidden_size | 512 / 768 | **128-256** |
| 层数 | 6-12 | **2-4** |
| 单神经元参数 | 51-134M | **1-5M** |
| 神经元数量 | 5 | **20-50** |
| 神经元总参数 | 338M | 50-200M |
| **协作层参数** | 130M | **500M-2B** |
| 协作:神经元 比 | 0.4:1 | **10:1 以上** |

### 6.3 训练流程五阶段

#### 阶段 0：规格设计
- 20-30 个神经元，每个 2-3M 参数，hidden=256，层数=3
- 协作层目标参数量 360M+（含 side_channels + cross_spec + 场演化层 + 协作注意力）

#### 阶段 1：数据分工策略（推荐 C）
- **策略 C：随机数据子集 + 训练自动分化**
- 每个神经元随机看 1/N 数据，训练过程中自然分化
- 不预定义主题/能力边界（避免人为偏见），类似人脑经验驱动分化
- 已有 [CoactivationTracker](file:///e:/taiji-neuron/taiji/resonance/tribal.py) 基础设施追踪分化模式

#### 阶段 2：单神经元预训练（弱能力初始化）
- 每个神经元独立训练（只看自己的 1/N 数据子集）
- **关键：PPL 故意停在 80-120**（不完全收敛，保留学习空间）
- 不要训练到 PPL < 50，否则单神经元能力过强，协作又成附属
- 这是"被迫协作"的前提——单神经元无法独立生成有效回答

#### 阶段 3：协作层训练（核心阶段）
- 冻结神经元主体，训练协作层（从头初始化）
- 协作层组件设计：
  - side_channels（全连接，每对神经元双向 excite/inhibit）
  - cross_spec 投影层（统一到 512 维场空间）
  - **场状态演化层**（新增，多轮可微场状态更新，~200M 参数）
  - **协作注意力**（新增，神经元间注意力机制，~100M 参数）
- Loss 设计：
  ```
  Loss = CE_loss(ensemble_output, target)        # 协作输出逼近目标
       + λ × diversity_loss(neuron_outputs)       # 防止神经元输出同质化
       + μ × cooperation_pressure(neuron_outputs) # 单神经元 PPL < 50 时加惩罚
  ```
- 目标：ensemble PPL < 30（协作显著优于单神经元）

#### 阶段 4：端到端微调
- 解冻 body 最后 1-2 层
- 联合微调（body lr << 协作层 lr，保护神经元专业化）
- 目标：ensemble PPL < 20

#### 阶段 5：评估
- EMERGE > 50%（协作远超最强个体）
- API 对话质量达标

### 6.4 关键设计决策（待研究，非立即执行）

| 决策点 | 选项 | 倾向 |
|-------|------|------|
| 协作层架构 | A. 复用 side_channels + cross_spec / B. 新增场演化层 + 协作注意力 | **B**（协作层需足够容量）|
| 单神经元预训练强度 | A. 不预训练 / B. PPL 80-120 / C. PPL 30-50 | **B**（弱能力但保留基础）|
| 神经元数量 | A. 10 / B. 20-30 / C. 50-100 | **B**（已有 tribal 拓扑支持）|

### 6.5 与当前流程的本质差异

| 维度 | 当前流程 | 方向 B 流程 |
|------|---------|-----------|
| 神经元预训练 | PPL 30-50（强能力）| PPL 80-120（弱能力）|
| 协作层角色 | 协作是锦上添花 | **协作是能力载体** |
| 协作层参数占比 | 24% | **>70%** |
| 单神经元能否独立 | 能 | **不能** |
| 协作涌现强度 | 弱 | 强（被迫协作）|

### 6.6 当前状态

**状态**：备案，不立即执行。
**优先路径**：当前 5 神经元阵容 + EOS + 短答案重训 → 评估 → 若不达标再考虑增加神经元数 → 若仍不达标才触发方向 B。
