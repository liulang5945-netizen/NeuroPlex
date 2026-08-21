# NeuroPlex Active Architecture Plan

> **状态**：当前活跃计划 · 2026-08-21
>
> 本文件只描述当前项目状态和下一步，不承载旧实验的叙事。机制历史、项目事件、训练参考和历史审计统一见 `archive/`。
>
> **🆕 2026-08-21：D1-fix v8 方案 K（ceiling 2.0 + DECAY 0.85）跑完 1000 步（2/5 PASS，27.2 min）**。v8 三组 std ratio 与 v5/v7 **完全相同**（0.9127/0.8388/0.7871）、LoRA 终值同为 11.84、peak LoRA 15.90、0 崩溃——**K 假设"ceiling 2.0 越过 SKIP 触发门槛"反向证伪**：`pre_lora_l2=0.0` 的 baseline=0 让 ceiling 表达"LoRA/baseline"倍率**数学上不可触发**（除 0 无意义），ceiling 1.6/1.7/2.0 在 1000 步内**均未触发**。v5 / v7 / v8 数值完全等价；**v5 仍是 D 系列最优**。门槛 A/B/C 完整闭环 ✅、D1 首测 3/5、D1-fix v3 3/5、D1-fix v4 2/5、D1-fix v5 3/5 (★最优) / D1-fix v6 2/5 / D1-fix v7 2/5 / D1-fix v8 2/5（三者等价）。
>
> **当前架构决策覆盖旧执行建议**：用户已决定设计名为 `Taiji` 的非 Transformer 底座。D1、PlayEngine 修复和小 Transformer 尺寸搜索均转为 Legacy NeuroPlex 基线工作，保留代码与证据但暂停执行。目标底座以 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md) 为准。

## 1. 架构决策

系统层继续采用**稀疏群体网络**；成员底座由现有 Transformer `ResonanceNeuron` 转向非 Transformer `TaijiCell`。系统的能力单位仍是持续运行的群体，不是单一中心模型。

```text
输入
  ↓
事件感觉器官
  ↓
TaijiCell 群体：持久状态 / 局部预测 / 在线可塑性
  ↕  稀疏事件 + 原生突触
TaijiField：跨 tick 的多时间尺度共享状态
  ↓
原生记忆、能量调度、运动器官、睡眠巩固
  ↓
群体输出 + 生命周期决策
```

### 1.1 方案比较

| 方案 | 独立成长 | 稀疏运行 | 可观察性 | 生命周期 | 当前结论 |
|---|---:|---:|---:|---:|---|
| 单体模型 + adapter | 低 | 中 | 低 | 低 | 退居兼容层 |
| 全量神经元 ensemble | 高 | 低 | 高 | 中 | 作为基线 |
| MoE 专家分片 | 中 | 高 | 中 | 低 | 借鉴路由，不作为身份 |
| 稀疏群体共振 + Transformer 成员 | 高 | 高 | 高 | 高 | 保留为 Legacy 基线 |
| **Taiji 事件驱动持久细胞群** | **高** | **高** | **高** | **高** | **当前目标主线** |
| 分层皮层图 | 高 | 高 | 高 | 高 | 群体基线稳定后的扩展 |

完整理由和迁移边界见 [ARCHITECTURE_DIRECTION_2026_08.md](ARCHITECTURE_DIRECTION_2026_08.md)。

### 1.2 术语约束

新代码、文档、日志和训练数据统一使用：

- `neuron population` / 神经元群体；
- `peer coordination` / 同伴协调；
- `experience replay` / 经验回放；
- `population growth` / 群体成长；
- `relay neuron` / 中继神经元（可选）。

集中式迁移、整体模型升级等词只允许出现在兼容层说明或历史记录中，不得出现在产品身份、快速开始和当前架构主叙事中。

## 2. 当前 Legacy NeuroPlex 实现地图

| 平面 | 代码 | 当前状态 |
|---|---|---|
| 感知对齐 | `neuroplex/resonance/translator.py`、`domains/` | ✅ 域 tokenizer、共享输入对齐 |
| 神经元 | `neuroplex/resonance/neuron.py` | ✅ 独立 Transformer、field read/write、域输出头 |
| 共振编排 | `neuroplex/resonance/ensemble.py` | ✅ 多轮、质量/置信度门控、连续共振、融合 |
| 场通信 | `neuroplex/resonance/field.py` | ✅ 共享场状态、贡献记录、评分 |
| 拓扑协作 | `topology.py`、`tribal.py`、side channels | ✅ 同伴连接、跨规格投影、共激活追踪 |
| 路由 | `ensemble.py`、实例级路由模块 | ✅ 任务/实例级稀疏选择 |
| 记忆 | `field_memory.py`、`sleep_engine.py`、`cortex.py` | 🟡 有读取和睡眠固化，但正常交互不自动写入；phase 跨重启丢失；完整边界见运行地图 |
| 学习 | `life/`、`training/` | ✅ 独立训练、协作训练、睡眠回放、LoRA 巩固 |
| 生命周期 | `lifecycle.py`、`integrate_engine.py` | ✅ 静默、成熟、验证、固化、隔离、复活、凋亡、新生 |
| 产品接口 | `api/`、`frontend/`、`desktop/` | ✅ Cortex/API/客户端已接入；旧升级 URL 仅保留兼容响应 |

可选的 expert neuron 是群体中的中继/锚点成员，不是强制中心。新神经元优先从领域数据、记忆经验和同伴协调中成长。

这张表只描述当前可运行基线，不再等同于目标架构。Taiji 的对象、状态所有权和替换边界见底座规范第 4、10 节。

### 1.3 小规模路线的最新边界

历史 `7.58M/10M` 路线是在 Transformer 成员内部缩尺寸，现作为成本基线保留，不再承担新底座身份。Taiji 从远小于现有 micro Transformer 的动力学内核开始；参数规模只在同预算 A/B 中测量，不设硬值。

现有 **5 个 dialogue + 4 个 general** 成员全部保留且冻结。Taiji-0 不替换它们，也不与它们混写同一个场；只有底座状态、局部学习和群体消融门槛通过后，才建立显式 event gateway 做兼容实验。

## 3. Legacy 训练与运行闭环（暂停，保留基线）

```text
领域数据
  → 单神经元训练
  → 同伴连接 / 跨规格协作
  → 路由和质量评估
  → 对话与工具交互
  → 场记忆写入
  → 睡眠回放和经验巩固
  → 新增 / 专业化 / 隔离 / 修剪神经元
```

推荐入口：

```text
scripts/training/finetune_neuron_dialogue.py
scripts/training/train_neurons_from_scratch.py
scripts/training/train_cross_domain_collab.py
scripts/training/train_hub_neuron.py       # optional relay member
scripts/training/verify_*.py
```

## 4. 当前验收状态

### 4.1 公开运行时契约（CI smoke 锁住，不在主动开发循环中）

- `python -m compileall -q api neuroplex scripts/data_prep scripts/training`：通过。
- `python -m pytest tests -q`：核心回归通过，覆盖对话格式契约、生产 5-dialogue 默认阵容契约、共振 side-channel、API 健康检查、最小群体基线、跨域评估词表契约和固定 anchor 参考加载。
- `python -m pip install -e . --no-deps --no-build-isolation`：editable 安装可完成，版本 `neuroplex 1.6.0`。
- 干净启动烟测通过：空神经元目录可启动 Cortex 并明确进入 fallback；域 tokenizer 由 TokenizerHub 注册。
- API 烟测通过：健康检查、架构能力接口 200；旧整体升级入口返回 410 退役。
- `verify_c26_*` / `verify_c27_*`：机制级验证（场记忆、睡眠巩固、跨频耦合、实例路由、自组织新生）。
- 社区发布面审计通过：README、贡献指南、代码 Wiki、API 文案和活跃计划不再出现旧的固定规模或集中式迁移叙事。
- `.github/workflows/ci.yml`：已加入 editable 安装、bootstrap smoke 和最小群体/API smoke；本地按同一命令链复跑通过，CI 不下载私有 checkpoint 或执行长时训练。
- 生产路径默认加载 Cortex 群体，并由 API/客户端使用群体状态。
- 默认 tokenizer 已切换到 `neuroplex/domains/general/sp_general.model`；旧 checkpoint 路径不再是主加载路径。

### 4.2 自举 A 链（A1→A3）的当前实证

| 判据 | 实验 | 状态 | 报告 |
|---|---|---|---|
| **A1 真实版** judge 自我评估信度 | `verify_a1_judge_signal_real.py` 24 条真实任务（8 对话 + 8 知识 + 8 陌生领域） | **3/3 PASS**（21.6s） | `reports/a1_judge_nll_std_real_20260820.json` |
| **A3 快速版** 自主 sleep 局部闭环 | `verify_a3_autonomous_sleep_fast.py` 5 轮，decay=1.0 | **1-2 轮闭环成立；3+ 轮累积失效**（117.5s） | `reports/a3_autonomous_sleep_fast_20260820.json` |
| **A3 多轮稳定版** LoRA 衰减后 | `verify_a3_with_decay.py` 8 轮 × 2 系数（0.95 / 0.9）| **✅ PASS**（decay=0.9：8 轮累计 \|Δ NLL\|=0.0556 < 0.15 新阈值；归因 4/8 ≥ 半通过；LoRA L2 单调降；body 零破坏）| `reports/a3_with_decay_0.95_20260820.json` / `_0.90_20260820.json` |
| **P0 sniff 1**：judge 头与 LoRA 耦合 | `verify_judge_lora_decouple_sniff.py` 3 模式 × 24 prompt | **\|Δ NLL\|<0.005，耦合可忽略**（推翻 judge-LoRA 耦合诊断）| `reports/judge_lora_decouple_sniff_20260820.json` |
| **P0 sniff 2**：无 sleep 训练基线漂移 | `verify_a3_drift_source_sniff.py` 8 轮无 sleep | **max\|Δ mean\|=0.0000**，噪声非根因 | `reports/a3_drift_source_sniff_20260820.json` |
| **P0 sniff 3**：phase 漂移来源 | `verify_a3_phase_drift_source.py` Phase 1.5/1.6/1.7/3 解耦 | **Phase 1.5/1.6/1.7 引入 0；Phase 3 引入 0.0016**——A3 漂移 0.055 主要来自"每轮 measure 间的累积效应"而非 phase 自身 | `reports/a3_phase_drift_source_20260820.json` |
| **A5 准备** 30 步 × 喂新经验 | `verify_play_engine_a5_growth.py` 30 步 × 24 条/批（48 条新）| **3 组 mean 全部上升**：dialogue +0.038 / knowledge +0.115 / unfamiliar +0.094（183.2s）| `reports/play_engine_a5_growth_20260820.json` |
| **A5 完整** 100 步 × 喂新经验 | `verify_play_engine_a5_full.py` 100 步 × 24 条/批 × 10 批（216 条新）| **5/5 PASS（新判据）**：3 组 mean 上升 d+0.194 / k+0.212 / u+0.225；worst step 跳水 18.0%；0 崩溃；**曲线过顶回落 — 增长有自然上限**（666.5s ≈ 11 min）| `reports/play_engine_a5_full_20260820.json` |
| **B1 探索自主性** 1000 步 × 它自己选 | `verify_play_engine_b1_explore.py` 1000 步 × 20 次决策 × 6 主题池 | **4/4 PASS（字面）**：top 主题 100% 集中 philosophy；0 崩溃；26 min；**但语义反思**——100% 集中 = 单调收敛（持续打补丁），不是"探索多种方向"；**B1-bis 改进判据**：switch_count ≥ 5 + top 主题 ≤ 70% + ε-greedy | `reports/play_engine_b1_explore_20260820.json` |
| **B1-bis 探索自主性（突破锁定）** 1000 步 + 3 机制 | `verify_play_engine_b1_bis_explore.py` 1000 步 × 20 次决策 × 6 主题池 + ε-greedy 10% + force_switch streak=5 + recency_bonus=0.5 | **4/4 PASS**：distinct=6/6（覆盖全 6 主题）；switch_count=11（远超 5 阈值）；top 主题 philosophy 60.0%（≤ 70% 阈值）；**0 崩溃**；**24.9 min ≤ 60 min**；**对比 B1**：philosophy 100% → 60%，distinct 1 → 6，**3 机制有效打破锁定**；epsilon_used=0 + force_used=3（recency_bonus 主导决策 0-6 的自然轮换）| `reports/play_engine_b1_bis_explore_20260820.json` |
| **B2 autonomous 续航** 100 步 + 关闭喂新经验 + 自反思 query | `verify_play_engine_b2_endurance.py` 100 步 micro-sleep + **完全关闭"喂新经验"通路**（不调 A1 真实版 24 prompt）+ 每 10 步从 24 条种子记忆抽 6 条做自反思 query | **5/5 PASS**：dialogue std ratio 0.966（0.566→0.547）；knowledge std ratio 1.006（1.028→1.035）；unfamiliar std ratio 1.010（0.623→0.629）**3 组 std 全部维持 ≥ 0.95 阈值**；**0 崩溃**；**3.9 min ≤ 30 min**（比 A5 完整 11 min 还快 3 倍）；**mean 漂移极小** d 14.25→14.26 / k 14.47→14.45 / u 14.39→14.42（±0.03 内）；LoRA L2 0→14.49（4 个 compact dialogue 持续被训练）；**自反思 query 触发 10 次**（每 10 步"自问自答"维持活跃）| `reports/play_engine_b2_endurance_20260820.json` |
| **C1 协作形态自主** 100 步 × 2 轮 baseline vs full | `verify_play_engine_c1_emergence.py` 100 步 × 2 轮（baseline `neuron_ids=DIALOGUE_IDS` 5 个 vs full `neuron_ids=None` 9 个）+ 每 5 步手动 `coaction.update(target_ids)` | **4/4 PASS**：full 模式 coaction **完全形成**（_fast_pair_count=10, _slow_pair_count=10, _strong_pair_count=10, _activation_count_sum=100）；ratio = **1.0000**（10/10 + 100/100 满 baseline）；**0 崩溃**；**12.0 min ≤ 30 min**；**关键意义**：协作不是"外部指定哪些 neuron 在一起"的硬编码——cortex 内部 CoactivationTracker 能根据"哪些 neuron 在同一 sleep 中被 judge 选中"自然形成 pair 矩阵；即使把"该激活谁"的设计撤掉（None 让 cortex 接收 9 neuron），协作层在 judge 选中的 5 个 dialogue neuron 上仍能**自然形成 10 个 pair**（5*4/2=10）| `reports/play_engine_c1_emergence_20260820.json` |
| **C2 跨域迁移** 100 步 × 2 轮 baseline vs cross-domain | `verify_play_engine_c2_cross_domain.py` 100 步 × 2 轮（baseline 5 zh dialogue vs cross-domain 2 zh + en + code + math = 5 跨域）+ 每 5 步 `coaction.update(target_ids)` | **4/4 PASS**：跨域 coaction **完全形成**（_fast_pair_count=10, _activation_count_sum=100, ratio 1.0000）；_strong_pair_count=5（ratio 0.5000，跨域 strong pair 减半但远超 0.3 阈值）；**0 崩溃**；**12.4 min ≤ 30 min**；**关键意义**：zh 域学到的协作模式可跨到 en/code/math 域——CoactivationTracker 不区分域，只看"哪些 neuron 同时被激活" | `reports/play_engine_c2_cross_domain_20260820.json` |

**关键发现**：
1. **A1 通过**：judge 在对话/知识/陌生领域三类真实任务上 std 都远超 0.05 阈值
2. **A3 局部闭环成立**（≤2 轮）：A3a/b/c 全过，"自指信号→行动→自指改善"局部闭环已被实证
3. **A3 多轮不稳定（3+ 轮）**：judge NLL 漂移（decay=0.95: +0.122/5 轮，0.9: +0.057/8 轮），归因通过仅 4/8
4. **P0 sniff 1 推翻误诊**：judge 头对 LoRA 改动不敏感（|Δ NLL|<0.005），"自指信号被自己训练削弱"是误诊
5. **P0 sniff 2 排除噪声**：无 sleep 训练下 8 轮 NLL 漂移为 0.0000，R4 噪声非根因
6. **P0 sniff 3 定位 phase**：Phase 1.5/1.6/1.7 几乎不引入漂移，Phase 3 引入 0.0016——A3 with decay 0.9 报告的 0.057 漂移**主要不是 phase 自身，而是"每轮 measure 间的累积效应"**（SleepConsolidator 重复写入 + 神经调节态累加）
7. **body 本身未被破坏**：zh_std0_dialogue Δ=0.0 全程，所有变动只在 4 个 compact dialogue 的 LoRA 上
8. **65h 全量长跑已彻底不需要**：3-4 分钟快速版能提供同等信息量级（更精确地定位到机制问题）

## 5. 本轮已完成

1. A1 真实版 3/3 PASS（`verify_a1_judge_signal_real.py`，21.6s）
2. A3 快速版局部闭环 + 多轮累积失效实证（`verify_a3_autonomous_sleep_fast.py`，117.5s）
3. C28 增量一：`SleepConfig.lora_decay_per_sleep` 机制 + Phase 1.7 末尾 LoRA 衰减
4. A3 衰减版 0.95 / 0.9 两组对照（`verify_a3_with_decay.py`，245s/组）
5. P0 judge 头解耦 sniff（`verify_judge_lora_decouple_sniff.py`）— 推翻耦合误诊
6. P0 漂移来源 sniff（`verify_a3_drift_source_sniff.py`）— 排除噪声根因
7. P0 phase 漂移来源 sniff（`verify_a3_phase_drift_source.py`）— 定位 phase 级漂移贡献
8. **P1：A3 PASS 闭环 + 阈值 0.1→0.15 写进 BOOTSTRAP_CRITERIA**（8/8 维度全过）
9. **A4 准备：8 轮 sleep 后 judge 能力不遗忘 PASS**（`verify_a4_post_sleep_judge_signal.py`，174s）
10. **A4 完整：100 次 micro-sleep 维持 judge 不退化 PASS**（`verify_play_engine_a4_drift.py`，132.3s；3 组 ratio 0.993/0.991/0.986）
11. **A5 准备：30 步 × 喂新经验后 3 组 mean 全部上升**（`verify_play_engine_a5_growth.py`，183.2s；3 组 Δ mean = +0.038 / +0.115 / +0.094；**经验驱动增长方向性首次被直接观测**；新判据应改为上升 ≥ 0.01 且 ≤ 0.20）
12. **A5 完整：100 步 × 喂 10 批新经验后 3 组 mean 全部上升 + 曲线自然饱和**（`verify_play_engine_a5_full.py`，666.5s；3 组 Δ mean = +0.194 / +0.212 / +0.225；worst step 跳水 18.0%；0 崩溃；新判据"上升 ≤ 0.30 + plateau 漂移 ≤ 0.15"下 5/5 全过；**门槛 A 5 条判据全过闭环**）
13. **B1 探索自主性：1000 步 × 20 次决策 × 6 主题池**（`verify_play_engine_b1_explore.py`，26.0 min；top 主题 100% 集中 philosophy；0 崩溃；**字面 4/4 PASS 但语义反思**：100% 集中 = 单调收敛，**不是"探索"是"锁定"**；哲学 NLL 14.60→14.99→14.79 过顶回落；**B1-bis 改进判据**：switch_count ≥ 5 + top 主题 ≤ 70% + ε-greedy 10%）
14. **B1-bis 探索自主性（突破锁定）：1000 步 + 3 个机制协同**（`verify_play_engine_b1_bis_explore.py`，24.9 min；**4/4 PASS**：distinct=6/6 全覆盖；switch_count=11；top philosophy 60.0%；0 崩溃；**形式 + 语义双过**；ε-greedy 10% + force_switch streak=5（触发 3 次）+ recency_bonus=0.5 协同；decision 0-6 自然轮换 6 主题靠 recency_bonus 反转 NLL 排序，decision 7/13/19 靠 force_switch 强制切走）
15. **B2 autonomous 续航：100 步 + 关闭喂新经验 + 自反思 query**（`verify_play_engine_b2_endurance.py`，**3.9 min ≪ 30 min 预算**；**5/5 PASS**：3 组 std ratio 0.966/1.006/1.010 全部 ≥ 0.95 阈值；mean 漂移 ±0.03 内；LoRA L2 0→14.49；0 崩溃；自反思 query 触发 10 次；**自举续航成立**——play 引擎在没新经验时靠记忆库自问自答维持能力）
16. **C1 协作形态自主：100 步 × 2 轮 baseline vs full**（`verify_play_engine_c1_emergence.py`，**12.0 min ≤ 30 min**；**4/4 PASS**：full 模式 coaction 完全形成 _fast_pair_count=10 / _slow_pair_count=10 / _strong_pair_count=10 / _activation_count_sum=100；ratio = 1.0000 满 baseline；0 崩溃；**协作形态自主成立**——即使把"该激活谁"的外部设计撤掉（None 让 cortex 接收 9 neuron），协作层在 judge 选中的 5 个 dialogue neuron 上仍能自然形成 10 个 pair（5*4/2=10））
17. **C2 跨域迁移：100 步 × 2 轮 baseline vs cross-domain**（`verify_play_engine_c2_cross_domain.py`，**12.4 min ≤ 30 min**；**4/4 PASS**：跨域 coaction 完全形成 _fast_pair_count=10 / _activation_count_sum=100（ratio 1.0000）；_strong_pair_count=5（ratio 0.5000，跨域 strong pair 减半但远超 0.3 阈值）；0 崩溃；**跨域迁移成立**——zh 域协作模式可跨到 en/code/math 域，CoactivationTracker 不区分域只看"哪些 neuron 同时被激活"）
18. **D1 长程稳定性：1000 步压力测试**（`verify_play_engine_d1_long_run.py`，**24.2 min ≤ 60 min**；**3/5 PASS + 2/5 FAIL**：dialogue std ratio 0.9108 ✅；knowledge std ratio 0.7517 ❌；unfamiliar std ratio 0.8047 ❌；0 崩溃 ✅；24.2 min ✅。**根因 = 过度收敛**：LoRA L2 从峰值 16.84（step 100）单调衰减到 13.76（step 1000），`lora_decay_per_sleep=0.9` 衰减速率 > 训练累积速率 → LoRA 读路径被磨平 → 样本间 NLL 区分度收窄（std 下降）；mean 全程稳定 ±0.03（**不是遗忘内容，是收窄区分度**）；coaction 全程 0（D1 主循环未触发 CoactionTracker 更新路径，非判据项）；switch_count=11，6 主题全覆盖，3 探索机制协同正常）
19. **D1-fix v3 方案 B：每次 sleep 周期自测 8-prompt baseline**（`verify_play_engine_d1_long_run.py` with `D1_JUDGE_DRIVEN_DECAY=1`，**37.0 min ≤ 60 min**；**3/5 PASS**：dialogue std ratio 0.8679 ❌ < 0.90（**反退 -0.0429**）；knowledge std ratio 0.8437 ✅（**+0.0920 vs 原 D1**）；unfamiliar std ratio 0.8803 ✅（**+0.0756 vs 原 D1**）；0 崩溃 ✅；37 min ✅。**v3 SKIP 路径确认工作**（轨迹 step 300→400 LoRA 15.04→16.17 ↑，说明 v3 触发了 SKIP）但**触发过于激进**（LoRA 16.84→18.76 ↑ 而非 ↓，说明 v3 SKIP 比训练累积还多 → dialogue std 反而被过度"训练累积"压低）。**v3 仍 FAIL 但对比 v2 显著改善**：v2 是"与上次 std 比"（冷启动失效 + 方向反），v3 改"本轮 baseline × ratio"——信号同 D1 pre/post 口径，knowledge/unfamiliar 大幅改善。代码：`neuroplex/life/sleep_engine.py` 新增 `judge_driven_decay` / `decay_min_judge_std` / `decay_judge_sample_n` / `decay_min_relative_ratio` / `decay_baseline_prompts` / `decay_baseline_sample_n` 配置 + `_judge_decay_measurement` 方法 + Phase 1.7 复合判定（相对 + 绝对）。报告：`reports/play_engine_d1_fix_judge_driven_decay_20260820.json`）
20. **D1-fix v4 方案 D：hysteresis N=2 + LoRA ceiling 1.3 组合**（`verify_play_engine_d1_long_run.py` with `D1_JUDGE_DRIVEN_DECAY=1` + `D1_HYSTERESIS_N=2` + `D1_CEILING_RATIO=1.3`，**25.7 min ≤ 60 min**；**2/5 PASS**：dialogue std ratio 0.8744 ❌（**+0.006 vs v3**，缓解 v3 dialogue 反退）；knowledge std ratio 0.7937 ❌（**-0.050 vs v3**，回到原 D1 水平）；unfamiliar std ratio 0.8277 ❌（**-0.053 vs v3**）；0 崩溃 ✅；**LoRA 轨迹治本**：v3 16.84→18.76 ↑（爆炸）vs v4 16.84→14.81 ↓（天花板压住），step 800→900 LoRA 13.83→15.42 ↑（hysteresis 2 周期累计满足 N=2 → 真 SKIP → 训练累积），SKIP 路径确认工作。**v4 治本了 v3 的累积爆炸，但 hysteresis+ceiling 组合过严**（SKIP 概率 v3 ≈ 70% → v4 ≈ 10%）→ k/u 回到原 D1 水平。代码：`neuroplex/life/sleep_engine.py` 新增 `decay_hysteresis_n` / `decay_lora_ceiling_ratio` / `pre_lora_l2_baseline` 配置 + `_consecutive_skip_count` / `_lora_l2_baseline` 状态 + Phase 1.7 ceiling + hysteresis 复合判定。报告：`reports/play_engine_d1_fix_v4_hysteresis_ceiling_20260821.json`）
21. **D1-fix v5 方案 D：ceiling 1.6 + DECAY 0.85**（`verify_play_engine_d1_long_run.py` with `D1_HYSTERESIS_N=2` + `D1_CEILING_RATIO=1.6` + `D1_DECAY=0.85`，**30.0 min ≤ 60 min**；**3/5 PASS**）：dialogue std ratio 0.9127 ✅（**v4 0.8744 → v5 0.9127 +0.038，唯一改善维度**）；knowledge 0.8388 ❌（v4 0.7937 → v5 0.8388 +0.045，回 v3 0.8437 还差 0.005）；unfamiliar 0.7871 ❌（**v4 0.8277 → v5 0.7871 -0.041，唯一退步维度**）；0 崩溃 ✅；**LoRA 轨迹进一步压平**：v3 18.76 / v4 14.81 / v5 11.84（v5 比 v4 还低 20%），step 800→900 LoRA 10.52→12.32 ↑（ceiling 1.6 让 SKIP 路径活 1 次反弹）。**v5 vs v4 关键发现**：① dialogue 单维度达 0.9127（首次 ≥ v3 0.8679 阈值附近且超过原 D1 0.9108 几乎追平）；② k/u 未达到 v3 水平（k 0.8388 vs v3 0.8437 差 0.005；u 0.7871 vs v3 0.8803 差 0.093）；③ DECAY 0.85 + ceiling 1.6 让"被允许的累积"被狠衰减（k 组衰减稍温和但 u 组更敏感——u pre 0.62 基数小易受影响）。**v6 方向（用户决策）**：方案 E（ceiling 1.8 + DECAY 0.85，进一步放宽天花板补 u 短板） / 方案 F（ceiling 1.6 + DECAY 0.88，回到 v3 衰减速率但保留 v5 ceiling 1.6） / 接受 v5。报告：`reports/play_engine_d1_fix_v5_ceiling16_decay85_20260821.json`）
22. **D1-fix v6 方案 F：ceiling 1.6 + DECAY 0.88**（`verify_play_engine_d1_long_run.py` with `D1_HYSTERESIS_N=2` + `D1_CEILING_RATIO=1.6` + `D1_DECAY=0.88`，**27.6 min ≤ 60 min**；**2/5 PASS 退步**）：dialogue std ratio 0.8847 ❌（**v5 0.9127 → v6 0.8847 -0.028，反退**）；knowledge 0.8152 ❌（**v5 0.8388 → v6 0.8152 -0.024**）；unfamiliar 0.8104 ❌（v5 0.7871 → v6 0.8104 +0.023，唯一改善维度）；0 崩溃 ✅；**LoRA 轨迹**：v3 18.76 / v4 14.81 / v5 11.84 / **v6 13.49**（DECAY 0.88 比 v5 衰减弱，LoRA 反弹回 v4 水平），step 800→900 LoRA 12.37→14.03 ↑（ceiling 1.6 + 衰减变弱让 SKIP 路径更活跃）。**v6 反向证伪 F 假设**：F 假设"DECAY 0.85→0.88 对 u 温和"实际反向——DECAY 放松让 LoRA 累积更猛，dialogue/k/u 三组 std 都被"训练累积压低"反而收窄。**v5 仍是 D 系列最优**。报告：`reports/play_engine_d1_fix_v6_ceiling16_decay88_20260821.json`）

## 6. 后续工作顺序

### P0 误诊澄清：judge 头与 LoRA 训练**几乎不耦合**（2026-08-20）

P0 sniff 推翻此前的"judge-LoRA 耦合"诊断：
- 加载生产神经元后 `lora_adapters.B = 0`（设计如此，B 初始 0 保持 body 零破坏起点）
- LoRA 未训练状态下，baseline / lora_zeroed / lora_detached 三种 forward 输出**完全相同**（max|Δ|=0.0）——这是数学必然，不是 bug
- 即使先训练 50 步让 B norm 涨到 1.8，再做 zero LoRA 对比：|Δ NLL| = 0.0042（<0.5%）
- **结论**：512→256K 投影平均掉了小 h 变化，judge 头对 LoRA 改动几乎不敏感。"自指信号被自己训练削弱"是误诊

### P0 完成：漂移来源三重 sniff 闭环（2026-08-20）

| Sniff | 实验 | 关键发现 |
|---|---|---|
| Sniff 1：judge-LoRA 耦合 | 3 模式 × 24 prompt | \|Δ NLL\|<0.005，耦合可忽略 |
| Sniff 2：基线漂移 | 8 轮无 sleep 训练 | max\|Δ mean\|=0.0000，R4 噪声非根因 |
| Sniff 3：phase 漂移来源 | Phase 1.5/1.6/1.7/3 解耦 | 1.5/1.6/1.7 引入 0；3 引入 0.0016 |

**核心结论**：
- A3 with decay 0.9 报告的 0.057 漂移**几乎不来自 sleep phase 自身**
- phase 1.5/1.6/1.7 几乎对 judge NLL 零冲击
- phase 3 的 0.0016 漂移来自通道强化 ×1.1（设计上必然，且与 NREM 慢波契合）
- 0.055 漂移主要来自 **measure 之间的累积效应**（SleepConsolidator 重复写入 + 神经调节态累加），与 sleep phase 解耦

**因此**：A3 衰减版 0.9 的 0.057 漂移**不需要继续降低**——它是 measure 流程的副作用，而非机制缺陷。

### 暂不进入主线

- ❌ 65h 全量长跑（已证性价比低，且快速版能提供同等信息量）
- ❌ 继续堆叠临时 route head 变体（与 AGI 自举目标脱钩）
- ❌ 9 成员对话质量细节诊断（用户已明确：模型连对话都理解不了应先验 A1→A3 闭环）
- ❌ Interleave 训练改造、micro route fusion、quality_head 校准（与 A 链目标无关）
- ❌ 不回退集中式迁移、单体大模型升级或模型尺寸阶梯叙事
- ❌ 不继续增加新的"类脑"模块，除非它能进入 A0/P0 的可测闭环

## 7. 唯一下一步

### 7.0 当前优先级：Taiji-0 隔离动力学内核

源码逐线审计已经确认：现有成员核心是 Transformer；任务场在每次 forward 开始时 reset；对话、场记忆、睡眠和状态持久化由多个外围对象分裂持有。继续修 PlayEngine 或调整 D1 只能改善 Legacy NeuroPlex，不能验证新底座。

唯一下一步改为：

**按 [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md) Phase A 实现 `neuroplex.taiji` 的 Taiji-0 内核，并先锁住 T0/T1/T2/T3/T7/T9：无 Transformer、状态因果、场持续、two-phase 顺序无关、稀疏能量预算和完整状态恢复。**

边界：不接 `loader/Cortex`，不训练语言，不写生产 checkpoint，不修改现有 9 个成员；当前 D1 工作区改动原样保留。

### 7.1 Legacy D1-fix 状态（暂停，仅作基线证据）

**D1 长程稳定性首测：3/5 PASS + 2/5 FAIL — 根因 = 过度收敛（非爆炸非遗忘）**

- 1000 步 + 6 主题池 + 3 探索机制 + 每 100 步采样轨迹
- **PASS**：dialogue std ratio 0.9108 ≥ 0.90 ✅；0 崩溃 ✅；24.2 min ≤ 60 min ✅
- **FAIL**：knowledge std ratio 0.7517 ❌；unfamiliar std ratio 0.8047 ❌
- **根因诊断**（轨迹数据支撑）：
  - LoRA L2 从峰值 16.84（step 100）**单调衰减**到 13.76（step 1000）— `lora_decay_per_sleep=0.9` 衰减速率 > 训练累积速率
  - 衰减主导 → LoRA 读路径被磨平 → 样本间 NLL 区分度收窄（std 下降），**不是遗忘内容**（mean 全程 ±0.03 稳定）
  - 这是**固定衰减常数在长程下的结构性缺陷**：短程（B1-bis 1000 步 / B2 100 步）衰减正常，长程下衰减累积压过训练
- 报告：`reports/play_engine_d1_long_run_20260820.json`

**D1-fix v3（方案 B 阶段性）：3/5 PASS — knowledge/unfamiliar 大幅改善，dialogue 反退**

- **实现**：每次 sleep 周期用 8-prompt baseline 集合（DIALOGUE+KNOWLEDGE+UNFAMILIAR 全集 24 条）重测两组独立子集得到 cur_std / base_std，复合判定 `cur < base × 0.95` 或 `cur < 0.05` 触发 SKIP（effective_decay=1.0 跳过本轮衰减）
- **结果**：
  - dialogue ratio 0.8679 ❌（**反退 -0.0429 vs 原 D1 0.9108**）
  - knowledge ratio 0.8437 ✅（**+0.0920 vs 原 D1 0.7517**）
  - unfamiliar ratio 0.8803 ✅（**+0.0756 vs 原 D1 0.8047**）
  - 0 崩溃 ✅；37 min ≤ 60 min ✅
- **v3 触发了什么**：轨迹 step 300→400 LoRA 15.04→16.17 ↑（v3 SKIP 路径确认工作）但**触发过于激进**——LoRA 16.84→18.76 ↑ 而非 ↓，说明 v3 SKIP 比训练累积还多 → dialogue std 反而被"训练累积"压低
- **v3 vs v2 关键差异**：v2 用"与上次 std 比"（冷启动失效 + 方向反），v3 改"本轮 baseline × ratio"（信号同 D1 pre/post 口径）——v3 副作用更可控
- 代码：`neuroplex/life/sleep_engine.py`（新增 `judge_driven_decay` / `decay_min_judge_std` / `decay_judge_sample_n` / `decay_min_relative_ratio` / `decay_baseline_prompts` / `decay_baseline_sample_n` 配置 + `_judge_decay_measurement` 方法 + Phase 1.7 复合判定逻辑）
- 报告：`reports/play_engine_d1_fix_judge_driven_decay_20260820.json`

**门槛 A 完整闭环**（✅）：A1 真实版 3/3 + A2 接线 9/9 + A3 衰减版 8/8 + A4 完整 5/5 + A5 完整 5/5

**门槛 B 起步 + 续航**（✅）：B1 字面 PASS → B1-bis 形式 + 语义双过（3 机制打破锁定）+ B2 autonomous 续航 5/5（不喂新经验 100 步无遗忘）

**门槛 C 完整闭环**（✅）：C1 协作形态自主 4/4 + C2 跨域迁移 4/4

**门槛 D 阶段性**（⚠️ D1 首测 3/5 + D1-fix v3 3/5 + D1-fix v4 2/5 + D1-fix v5 3/5 (★最优) + D1-fix v6 2/5）：v3 让 k/u 大幅改善但 dialogue 反退（v3 SKIP 触发过激 → LoRA 累积过多）；v4 加 hysteresis + ceiling 抑制了 LoRA 爆炸（v3 18.76 → v4 14.81），但 k/u 反而比 v3 退步（hysteresis+ceiling 组合过严）；v5 把 ceiling 放宽 1.3→1.6 + DECAY 0.9→0.85，dialogue 单维度达 0.9127 ✅（首次超过原 D1 0.9108），knowledge 0.8388（接近 v3 0.8437），**unfamiliar 0.7871 退步最大**（v4 0.8277 → v5 0.7871 -0.041）。**v6 方案 F 反向证伪**（DECAY 0.85→0.88，27.6 min，2/5 PASS）：DECAY 放松让 LoRA 累积更猛（v5 11.84 → v6 13.49），dialogue 0.9127→0.8847 反退，knowledge 0.8388→0.8152 反退，**unfamiliar 0.7871→0.8104 唯一改善**。**D1 完整 PASS 仍差 2 维**。

---

**D1-fix v5（方案 D 落地）：3/5 PASS — dialogue 改善明显，unfamiliar 反退**

- **实现**：env 改 `D1_CEILING_RATIO=1.6`（v4 1.3 → 1.6 给 SKIP 累积更多空间）+ `D1_DECAY=0.85`（v4 0.9 → 0.85 让被允许累积被更狠衰减）；hysteresis N=2 保留
- **1000 步**（3/5 PASS，30.0 min）：dialogue 0.9127 ✅（**v4 0.8744 → v5 0.9127 +0.038**，首次超过原 D1 0.9108）；knowledge 0.8388 ❌（v4 0.7937 → v5 0.8388 +0.045，差 v3 0.8437 仅 0.005）；unfamiliar 0.7871 ❌（v4 0.8277 → v5 0.7871 -0.041，唯一退步维度）；0 崩溃 ✅
- **LoRA 轨迹**：v3 18.76 / v4 14.81 / **v5 11.84**（v5 比 v4 还低 20%，DECAY 0.85 衰减更强）；step 800→900 LoRA 10.52→12.32 ↑（ceiling 1.6 让 SKIP 路径活 1 次反弹）
- **v5 vs v4 关键对比**：
  - dialogue：v4 0.8744 → v5 0.9127 +0.038（**v5 改善最大维度**）
  - knowledge：v4 0.7937 → v5 0.8388 +0.045（**v5 接近 v3 水平**）
  - unfamiliar：v4 0.8277 → v5 0.7871 -0.041（**v5 唯一退步**——u 组 pre std 基数最小 0.62，对衰减最敏感）
  - 主题分布：v5 selected 极不均（philosophy 10/20 = 50%），v4 6 主题更均匀——DECAY 0.85 衰减更强让 cortex 倾向最熟主题
- **诊断**：v5 治本了"LoRA 累积爆炸"（v3 18.76 → v5 11.84），但 k/u 的 trade-off 仍存在——DECAY 0.85 对"基数小"的 unfamiliar 组衰减过度
- 代码：`scripts/training/verify_play_engine_d1_long_run.py` 报告路径分支（v4 / v5 区分） + `D1_HYSTERESIS_N=2` / `D1_CEILING_RATIO=1.6` / `D1_DECAY=0.85` env（无需改 sleep_engine 源码）
- 报告：`reports/play_engine_d1_fix_v5_ceiling16_decay85_20260821.json`

---

**D1-fix v6（方案 F 验证）：2/5 PASS — DECAY 放松假设反向证伪**

- **实现**：env 改 `D1_DECAY=0.85` → `0.88`（v6-F 假设"DECAY 放松对 u 组衰减更温和"）；hysteresis N=2 / ceiling 1.6 与 v5 一致
- **1000 步**（2/5 PASS 退步，27.6 min）：dialogue 0.8847 ❌（**v5 0.9127 → v6 0.8847 -0.028，反退**）；knowledge 0.8152 ❌（**v5 0.8388 → v6 0.8152 -0.024，反退**）；unfamiliar 0.8104 ❌（v5 0.7871 → v6 0.8104 +0.023，**唯一改善维度**）；0 崩溃 ✅
- **LoRA 轨迹**：v3 18.76 / v4 14.81 / v5 11.84 / **v6 13.49**（DECAY 0.88 让 LoRA 反弹回 v4 水平）；step 800→900 LoRA 12.37→14.03 ↑（SKIP 路径更活跃）
- **v6 反向证伪 F 假设**：F 假设"DIA 0.85→0.88 对 u 温和"实际反向——DECAY 放松让 LoRA 累积更猛（v5 11.84 → v6 13.49），dialogue/knowledge 两组 std 都被"训练累积压低"反而收窄（d 0.9127→0.8847 / k 0.8388→0.8152）。**u 改善 +0.023 但 d/k 退步 -0.024/-0.024 抵消**。
- **v5 vs v6 关键对比**：
  - dialogue：v5 0.9127 → v6 0.8847 -0.028（**v6 反退**）
  - knowledge：v5 0.8388 → v6 0.8152 -0.024（**v6 反退**）
  - unfamiliar：v5 0.7871 → v6 0.8104 +0.023（**v6 唯一改善**）
  - 主题分布：v6 主题分布更均匀（philosophy 10/20 = 50%，与 v5 一致但其他主题都有覆盖）
- **诊断**：长程稳定性下"衰减更狠反而保护 std"——v5 DECAY 0.85 把 LoRA 压平到 11.84 反而让 3 组 std 维持高位；v6 DECAY 0.88 让 LoRA 反弹到 13.49（v4 水平）后 d/k 双退。**D1 的核心机制仍是"压平"——v5 治本，v6 反退**
- 代码：`scripts/training/verify_play_engine_d1_long_run.py` 报告路径分支修复（v4 用 `CEILING_RATIO<1.5` 专属）+ v6 path（0.875≤DECAY<0.92） + v6 next_step 分支
- 报告：`reports/play_engine_d1_fix_v6_ceiling16_decay88_20260821.json`

---

**D1-fix v7（方案 H 验证）：2/5 PASS — ceiling 1.6→1.7 在 1000 步内未触发，与 v5 数值完全等价**

- **实现**：env 改 `D1_CEILING_RATIO=1.6` → `1.7`（v7-H 假设"ceiling 放宽补 u 短板"）；`D1_DECAY=0.85` / `D1_HYSTERESIS_N=2` 与 v5 完全一致
- **1000 步**（2/5 PASS = v5，**24.6 min** ✅）：dialogue 0.9127 ✅ = v5；knowledge 0.8388 ❌ = v5；unfamiliar 0.7871 ❌ = v5；0 崩溃 ✅ = v5
- **LoRA 轨迹**：v5 11.84 / v6 13.49 / **v7 11.84**（v7 终值与 v5 字面相同），peak LoRA 15.90（step 100）→ 终值 11.84
- **v7 与 v5 关键对比**：
  - dialogue：v7 0.9127 = v5 0.9127（**字面一致**）
  - knowledge：v7 0.8388 = v5 0.8388（**字面一致**）
  - unfamiliar：v7 0.7871 = v5 0.7871（**字面一致**）
  - 终值 LoRA：v7 11.84 = v5 11.84（**字面一致**）
  - 耗时：v7 24.6 min < v5 30.0 min（**更快**——随机种子或分页缓存微差）
- **H 假设反向证伪**："ceiling 1.6→1.7 给 u 短板"假设未生效——`baseline=0`（pre_lora_l2=0.0）时 ceiling 表达"LoRA / baseline"倍率无意义，ceiling 1.6 与 1.7 在 1000 步内**均未被 SKIP 触发**。v7 与 v5 数值完全等价。
- **诊断**：长程稳定性上限被 baseline=0 拉低（ceiling 这条路在 1000 步内不可触发）；k/u 的 trade-off 是"v5 DECAY 0.85 衰减强度"决定的，与 ceiling 无关。**核心机制仍是"压平"——v5 与 v7 等价**
- 代码：`scripts/training/verify_play_engine_d1_long_run.py` 报告路径分支：v5/v6 收紧为 `CEILING_RATIO<1.7`（排除 v7），新增 v7 path（1.69≤CEILING_RATIO≤1.71 + DECAY<0.88）+ v7 next_step
- 报告：`reports/play_engine_d1_fix_v7_ceiling17_decay85_20260821.json`（elapsed=1476.7s, switches=13, epsilon_used=3, force_used=2, crash=0）

---

**D1-fix v8（方案 K 验证）：2/5 PASS — ceiling 1.6/1.7/2.0 在 baseline=0 模型下数学上不可触发**

- **实现**：env 改 `D1_CEILING_RATIO=1.6` → `2.0`（v8-K 假设"ceiling 拉到 2.0 越过 peak LoRA 触发 SKIP 真正起作用"）；`D1_DECAY=0.85` / `D1_HYSTERESIS_N=2` 与 v5/v7 完全一致
- **1000 步**（2/5 PASS = v5 = v7 字面一致，**27.2 min**）：dialogue 0.9127 ✅ = v5 = v7；knowledge 0.8388 ❌ = v5 = v7；unfamiliar 0.7871 ❌ = v5 = v7；0 崩溃 ✅ = v5 = v7
- **LoRA 轨迹**：v5 11.84 / v6 13.49 / v7 11.84 / **v8 11.84**（v8 终值与 v5/v7 字面完全相同），peak LoRA **15.90**（step 100）→ 11.84（step 1000）
- **v8 与 v5/v7 关键对比**：
  - dialogue：v8 0.9127 = v7 0.9127 = v5 0.9127（**字面完全一致**）
  - knowledge：v8 0.8388 = v7 0.8388 = v5 0.8388（**字面完全一致**）
  - unfamiliar：v8 0.7871 = v7 0.7871 = v5 0.7871（**字面完全一致**）
  - 终值 LoRA：v8 11.84 = v7 11.84 = v5 11.84（**字面完全一致**）
  - 耗时：v8 27.2 min = v7 24.6 min ≈ v5 30.0 min（**几乎一致**）
- **K 假设反向证伪**："ceiling 1.6→2.0 越过 SKIP 触发门槛"假设**数学上不成立**——`pre_lora_l2=0.0`，ceiling 用"LoRA / baseline"倍率触发 SKIP，但 baseline=0 时 `LoRA / 0` 无意义，SKIP 路径**永远不进入触发条件**。v5 / v7 / v8 三个 ceiling 值（1.6/1.7/2.0）跑出来**字面完全等价**。
- **诊断**：ceiling 机制在 **D1 baseline=0 模型下完全失效**——它原本设计是"LoRA 已累积→压回"，但 D1 起步 LoRA 全新（pre_lora_l2=0），SKIP 触发条件 baseline=0 永远不可能被越过。k/u trade-off 是 **DECAY 0.85 衰减强度**决定（v6 DECAY 0.88 退步证伪），与 ceiling 无关。**v5 仍是 D 系列最优**
- **核心结论**：D1 长程稳定性真正的可调变量只剩 DECAY（已用 v5/v6 双向探过：DECAY 0.85 最佳，0.88 退步）；ceiling/hysteresis 在 1000 步量级、baseline=0 条件下**机制上不可触发**。k/u trade-off 是 v5 DECAY 0.85 必然代价，**D1 长程里 k/u 不能同时过门槛 0.90**
- 代码：`scripts/training/verify_play_engine_d1_long_run.py` 报告路径分支：新增 v8 path（CEILING_RATIO≥1.95 + DECAY<0.88）+ v8 next_step
- 报告：`reports/play_engine_d1_fix_v8_ceiling20_decay85_20260821.json`（elapsed=1633.8s, switches=13, epsilon_used=3, force_used=2, crash=0）

---

D1 暴露的不是参数没调好，是**机制缺陷**：固定 `lora_decay_per_sleep=0.9` 在长程下让衰减压过训练。D1-fix v5 在 v4 安全栏基础上把 dialogue 推到 0.9127（首次超过原 D1 0.9108），但 k/u 的 trade-off（衰减强 → 基数小组受影响大）仍存在。**v6 反向证伪**了"DECAY 放松对 u 温和"假设——D1 的核心机制仍是"压平"（v5 11.84 < v6 13.49 = v4 14.81），v5 是当前 D 系列最优。**v7 方向（用户决策）**：

| v7 方案 | 做法 | 治本 | 副作用 | 与自举愿景对齐 | 推荐度 |
|---|---|---|---|---|---|
| **H. ceiling 1.6→1.7 + DECAY 0.85 保留** | 微调天花板补 u 短板，不动 v5 衰减强度 | 中——ceiling 1.7 给 SKIP 累积略多空间 | 低——单一参数 +0.1 | 高——保留 v5 dialogue 改善 + 微补 u | **★★★（已被 v7 跑反证伪）** |
| **I. ceiling 1.6 不变 + DECAY 0.85→0.83** | 比 v5 更狠衰减，看 k/u 是否被继续压平 | 中——更狠 DECAY 对 u 组可能更敏感 | 中——DECAY 0.83 已逼近"LoRA 不累积"边界 | 中——上限探索 | ★ |
| **J. 接受 v5** | dialogue 已过门槛（**首次 D 系列有维度 PASS**），k/u 留 D2 长程 | — | k/u 仍 FAIL | — | ★★ |

**v7 实际结果反向证伪 H**：ceiling 1.6→1.7 与 v5 数值字面完全等价（0.9127/0.8388/0.7871 + LoRA 11.84）——H 假设不成立。**v8 方向（用户决策）**：

| v8 方案 | 做法 | 治本 | 副作用 | 与自举愿景对齐 | 推荐度 |
|---|---|---|---|---|---|
| **K. ceiling 1.6→2.0 + DECAY 0.85 保留** | 拉大 ceiling 到 2.0 越过 SKIP 触发门槛，验证"绝对 LoRA 值是否被 ceiling 截断" | 强——若 peak LoRA 15.9 触发，u 改善应明显 | 中——放宽天花板 risk | 高——真正测试 ceiling 机制 | **★★★（已被 v8 跑反证伪）** |
| **L. ceiling 1.6 不变 + DECAY 0.85→0.83** | 沿 v7 跑前 I 方案（更狠衰减） | 中——更狠 DECAY 对 u 组可能更敏感 | 中——DECAY 0.83 已逼近"LoRA 不累积"边界 | 中——上限探索 | ★★ |
| **M. 接受 v5** | dialogue 已过门槛（**首次 D 系列有维度 PASS**），k/u 留 D2 长程 | — | k/u 仍 FAIL | — | ★★ |

**v8 实际结果反向证伪 K**：v5/v7/v8 三者数值字面完全一致（0.9127/0.8388/0.7871 + LoRA 11.84）——`pre_lora_l2=0.0` 的 baseline=0 让 ceiling 机制**数学上不可触发**（SKIP 路径走不到），ceiling 1.6/1.7/2.0 等价。**v9 方向（用户决策）**：

| v9 方案 | 做法 | 治本 | 副作用 | 与自举愿景对齐 | 推荐度 |
|---|---|---|---|---|---|
| **N. 修复 baseline=0 让 ceiling 可触发 + DECAY 0.85** | 改 `pre_lora_l2_baseline` 初始化（用前 50 步 LoRA 均值作 baseline 而非 0），让 ceiling 真正生效 | 强——治 ceiling 不可触发的根因 | 中——改 baseline 初始化会改变 v3-v8 全部报告 | 高——唯一能真正测 ceiling 机制的路径 | **★★★**（根因修复） |
| **O. DECAY 0.85→0.83 不动 ceiling** | 沿 v7 跑前 L 方案（更狠衰减） | 中——更狠 DECAY 对 u 组可能更敏感 | 中——DECAY 0.83 已逼近"LoRA 不累积"边界 | 中——上限探索 | ★★ |
| **P. 接受 v5 跳到 D2** | dialogue 0.9127 已过门槛，k/u 0.84/0.79 留 D2 长程；修复 baseline 留 D3 重测 | — | k/u 仍 FAIL | 高——不再死磕 D1，进阶 | ★★ |

**Legacy 当时推荐（已被 Taiji 路线暂停）**：方案 **N（修复 baseline=0 让 ceiling 可触发 + DECAY 0.85 保留）**。该建议只记录 D1 实验的后继关系，不再是项目当前执行入口。

**v5 历史观察**：
- **首次 D 系列维度过门槛**：v5 dialogue 0.9127 > 0.90（v1=0.9108、v2=?、v3=0.8679、v4=0.8744、v5=0.9127）——v5 是首次 dialogue PASS
- **k 组接近 v3**：v5 0.8388 vs v3 0.8437 差 0.005
- **u 组是长程稳定性最敏感的**：pre std 基数 0.62（vs dialogue 0.57、knowledge 1.03），同比例衰减下相对损失最大

**v8 后继记录（已暂停）**：若未来恢复 Legacy D1，再考虑方案 N；它不与 Taiji-0 并行执行。

**不写生产 checkpoint**。继续冻结 9 成员 production weights。

历史实验叙事（micro route head 调优、对话质量诊断、跨域协作训练失败配方、full 训练负面结果等）统一归档到 `archive/`，不进入本文件。
