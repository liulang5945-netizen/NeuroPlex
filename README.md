# 态极神经元 (Taiji Neuron) — 共振场架构

> **版本**: v2.0
> **日期**: 2026-07-15
> **状态**: 🏗️ Architecture Ready — 架构就绪，待更多训练验证 1+1>2

---

## 项目概述

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

## 目录结构

```
taiji-neuron/
├── taiji/
│   ├── resonance/          # ★ 共振场引擎（核心）
│   │   ├── field.py        #   共振场（共享通信介质）
│   │   ├── neuron.py       #   共振神经元
│   │   ├── ensemble.py     #   共振循环编排
│   │   ├── gating.py       #   门控机制（置信度+早停+触发）
│   │   ├── quality.py      #   质量过滤
│   │   ├── division.py     #   分工路径（规模分层+集群主导）
│   │   ├── translator.py   #   分词翻译器
│   │   ├── tribal.py       #   部落压缩指标
│   │   └── config.py       #   神经元规格配置
│   ├── brain/
│   │   └── cortex.py       #   意识中心（封装 ResonanceEnsemble）
│   ├── training/           #   训练管线
│   │   ├── distill.py      #   蒸馏训练
│   │   ├── single.py       #   单神经元训练
│   │   ├── joint.py        #   联合训练
│   │   ├── contrastive.py  #   对比学习
│   │   ├── scheduler.py    #   训练调度器
│   │   └── checkpoint_bridge.py  # 一代 checkpoint 桥接
│   ├── layers.py           #   Transformer 基础组件（零改动复用）
│   ├── tokenizer_native_v2.py     # 256K 通用分词器
│   ├── domains/            #   领域专用分词器
│   ├── tools/              #   工具系统（全部复用）
│   ├── agent/              #   Agent 系统（待适配）
│   ├── life/               #   生命系统（待适配）
│   ├── body/               #   身体系统（全部复用）
│   ├── safety/             #   安全系统（全部复用）
│   └── core/               #   核心基础设施
├── api/                    #   FastAPI 路由（待适配）
├── frontend/               #   Vue 3 前端（不变）
├── desktop/                #   PyQt6 桌面端（不变）
├── scripts/
│   ├── training/           #   训练脚本
│   │   ├── prepare_distill_data.py    # 准备蒸馏数据 + 提取教师方向
│   │   ├── distill_neurons.py         # 蒸馏 5 个神经元
│   │   ├── verify_distilled_neurons.py # 质量闸门验证
│   │   ├── run_division_experiments.py # 分工路径实验
│   │   ├── quick_division_test.py     # 快速分工测试
│   │   ├── test_distill_bridge.py     # 蒸馏桥接验证
│   │   └── test_division_path.py      # 分工路径逻辑验证
│   └── data_prep/          #   数据处理（全部复用）
├── data/
│   ├── distill/            #   蒸馏数据 + 教师方向
│   └── neurons/            #   蒸馏后的神经元 checkpoint
└── plans/
    ├── taiji-next-phase-plan.md       # 完整规划
    └── phase4-integration-plan.md     # Phase 4 集成规划
```

---

## 核心架构

### 三层设计

```
Level 0: 通用分词器 (256K)      ← I/O 协议层，可替换
    ↓
Level 1: 领域神经元 (5+个)       ← 独立 Transformer，领域专用
    ↓  field_write / field_read
Level 2: 共振场 (4096-dim)      ← 共享意识，独立于分词器（实际场维度 = 装配神经元规格最大值：2048/3072/4096，跨规格由投影层统一，见 CODE_WIKI）
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

### 神经元规格

| 规格 | 隐藏维度 | 层数 | 参数量 | 用途 |
|------|---------|------|--------|------|
| `compact` | 512 | 6 | ~24M | 辅助执行 |
| `standard` | 768 | 10 | ~59M | 主要执行 |
| `expert` | 1024 | 14 | ~118M | 决策+把关 |

---

## 快速开始

### 1. 验证核心（回归测试）

```bash
# 口径契约 + 共振 side_channels 回归（16 用例）
python -m pytest tests/ -q
# 预期: 16 passed
```

> 注：原 `test_distill_bridge.py` / `test_division_path.py` 已随 2026-08 训练管线重构退役（见 HISTORY 系列文档）；当前训练链路的验证脚本位于 `scripts/training/verify_*.py`（运行日志落盘 `logs/`）。

### 2. 训练管线（当前链路）

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

### 3. 使用 Cortex

```python
from taiji.brain.cortex import Cortex

# 加载神经元
cortex = Cortex(neurons_dir="data/neurons")

# 设置分词器（R18：改为相对路径 + 环境变量覆盖，旧绝对路径已过时）
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(os.path.join(os.environ.get("TAICHI_TEACHER_PATH", "checkpoint-481000"), "sentencepiece.model"))
cortex.set_tokenizer(sp)

# 生成文本
result = cortex.generate("今天天气怎么样？")
```

---

## 实验结论

### 实验 12: 门控机制
- **结论**: 1+1>2 不是默认行为，共振只在不确定时才有帮助
- **ConfidenceGate**: 确定预测应跳过共振（避免场噪声）
- **EarlyStop**: logits 收敛时停止迭代

### 实验 9: 质量过滤
- **结论**: 弱神经元稀释强神经元
- **QualityFilter**: 仅 PPL < 100 的神经元参与共振

### Phase 3: 分工路径
- **结论**: 当专家神经元匹配领域时，规模分层优于等权共识
- **code 领域**: scale_layering PPL 比 consensus 好 2.6×
- **组合策略**: 集群主导 × 内部规模分层

---

## 技术债务和后续工作

> 更新于 2026-08（修复审计 R 系列后）；完整状态见 `plans/BIO_INSPIRED_ARCHITECTURE_PLAN.md` 与 `plans/REMEDIATION_PLAN.md`。

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 🔴 P0 | hub 正式训练 | hub neuron（495M）smoke 链路已通，正式 GPU 训练待执行；随后正式协作层训练（`--hub-anchor-weight --hub-contrastive-weight`）+ 阶段 4 跨域评估 |
| 🔴 P0 | 共振机制 A/B 证据 | W_cond / field_read_layers 已训练闭环（R1/R2），但收益 A/B 报告尚未落盘（N2 规范） |
| 🟡 P1 | 验证硬化 | 关键 verify 脚本转真实 ckpt 加载的 pytest（slow 标记）+ 最小 CI |
| 🟡 P1 | 共享嵌入初始化 | 从 teacher embedding 用 SVD 初始化 512-dim 共享嵌入（低优先，现用正交随机） |
| 🟢 P2 | Agent 适配 | planner/reflector 改用 cortex.think() |
| 🟢 P2 | 工程加固 | ensemble.py 拆分 / 裸 except 加日志 / state_dict 聚合接口（R14 相关） |
| 🟢 P3 | GPU 加速 | 支持 CUDA 推理（loader 已有 device 传播，待实测） |
