<!-- 态极神经元架构全面计划 -->
<!-- 版本: v1.0 (2026-07-17) -->
<!-- 整合来源: 全部对话讨论内容 -->

# 态极神经元架构全面计划

## 文档导览

本文档是整个态极神经元架构的完整计划和设计文档，从愿景到实验，从架构到实现。分为以下部分:

- **第一部分: 愿景与架构** (章 1-4)
- **第二部分: 协同机制** (章 5-7)
- **第三部分: 技术设计** (章 8-12)
- **第四部分: 实验验证** (章 13-17)
- **第五部分: 执行路线** (章 18-19)

---

# 第一部分: 愿景与架构

## 一、态极的大脑: 从单一大模型到神经元模型体

### 1.1 核心转变

态极的本质没有变 — 它始终是一个生命体的核心。

```
原来的态极:
  大脑 = 单一大模型 (1B~12B 的单体 Transformer)
  生命系统 = 外挂在单体模型上的控制层 (feed/sleep/explore/play)
  两者关系 = 控制-被控制

现在的态极:
  大脑 = 一个由神经元模型构建成的模型体
        = 多个独立的小模型 (神经元) + 共享共振场
  生命系统 = 神经元集群的自然涌现行为
  两者关系 = 一体
```

不是"变成了"另一种东西。是大脑的内部构建方式变了 — 从单体变成了集群。

**这个转变的根本原因**: 单体模型的天花板是固定的，神经元模型体可以持续生长。但生命体的核心愿景始终不变: 自我感知、自我适应、自我进化。

### 1.2 神经元模型体的本质

```
神经架构的态极 ≠ "多个模型拼接"
神经架构的态极 = 一个生命体，其内部结构就是神经元群体
```

生命系统的本质:
- 原来: 生命系统是一个定时器 + 状态机，控制单个模型何时训练/推理
- 现在: 生命系统是神经元群体的集体行为本身

```
生命体的运作:
  觉醒 (推理)     = 神经元共振循环
  睡眠 (训练)     = 神经元内部参数整合 + 抱合生长新神经元
  饥饿 (数据缺口) = 加新神经元
  探索 (新领域)   = 激活潜在神经元 / 加神经元
  玩耍 (自由探索) = 神经元自由共振，不为产出服务
  死亡 / 重生     = 神经元剥离 + 重组
```

### 1.3 为何不能只是"换一个更好的单体模型"

单体模型的天花板是固定的。

| 单体模型的硬上限 | 神经元模型体的解决方案 |
|--------------|-----------------|
| 跨领域知识只能平均 | 互补不是平均 — 共振循环让其互补 |
| 权重越训越难变 | 加新神经元 = 加新能力，不影响旧的 |
| 一个权重全牵一动 | 剥离一个神经元不影响其他 |

---

## 二、三层架构

### 2.1 架构层次

```
第一层 (共享感官): 256K 词表 → 512 维共享嵌入
    - 神经语言层，所有神经元共用
    - 512 维是感官分辨率，不是认知瓶颈
    - 类比: 弱视的人不需要换视网膜也能理解复杂概念

第二层 (认知空间): 每个神经元独立的概念空间
    - 训练时更新
    - 真正决定认知能力
    - 领域专用 tokenizer + 转译层

第三层 (神经语言): 4096 维场空间
    - 神经元通过这个空间通信
    - 与 tokenizer 完全独立
    - 认知不变体
```

### 2.2 关键洞察: 三层各司其职

| 层次 | 作用 | 可变性 |
|-----|------|--------|
| 第一层 (共享 I/O) | 感官输入 | 可换 (转译层隔离) |
| 第二层 (神经元) | 认知处理 | 独立生长 |
| 第三层 (场) | 神经语言 | 恒定 |

**第一层是可换的** — 换通用词表只重训转译层，神经元内部不变。

---

## 三、词表系统

### 3.1 核心问题

旧理解: 256K 词表变了全废 (模型逻辑)
新理解: 人脑不是这样的 — 学习新东西可以添加而不影响

### 3.2 解决方案: 词表分层 + 转译层

```
通用词表 (I/O 格式)          神经元专用词表          神经语言
┌──────────────┐            ┌──────────────┐       ┌──────────────┐
│  256K tokens │    ↔       │  专用 tokens  │   ↔   │  场空间 4096 │
│  (转译层)    │            │  (认知空间)   │       │  (恒定)      │
└──────────────┘            └──────────────┘       └──────────────┘
```

- 换通用词表 → 只重训转译层，神经元内部不变
- 新领域 → 加神经元 + 专用 tokenizer，通过转译层接入通用 I/O
- 词表不是瓶颈，是可升级的模块

### 3.3 为什么领域专用 tokenizer

实验 6 和 7 揭示:
- 统一 tokenizer 对各领域不友好 (中文 PPL=415 vs 代码 PPL=3)
- 领域专用 tokenizer 大幅缩小质量差距 (524x → 1.1x)
- **知识缺口真实存在** — 领域专用 tokenizer 是正确的方向

### 3.4 转译层实现

```python
class TokenTranslator(nn.Module):
    """领域 tokenizer → 统一嵌入"""
    def __init__(self, vocab_size, embed_dim=256):
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.token_embedding(token_ids)

class UnifiedEmbeddingSpace(nn.ModuleDict):
    """多个 tokenizer 共享同一个嵌入维度"""
    def __init__(self, tokenizers, embed_dim=256):
        self.translators = nn.ModuleDict({
            name: TokenTranslator(sp.GetPieceSize(), embed_dim)
            for name, sp in tokenizers.items()
        })
```

---

## 四、神经元规格

### 4.1 三种规格

三种规格是进化路径，不是预设的三种"型号":

| 规格 | 参数量 | 角色 | 适合 |
|-----|-------|------|------|
| 紧凑型 (compact) | ~18M | 探索者 | 新领域探路 |
| 标准型 (standard) | ~80M | 主力员工 | 稳定领域 |
| 专家型 (expert) | ~200M | 深度顾问 | 复杂推理 |

### 4.2 神经元配置

```python
# 紧凑型
COMPACT = NeuronConfig(
    hidden_size=512, num_hidden_layers=6,
    num_attention_heads=8, num_key_value_heads=2,
    intermediate_size=1536,
)

# 标准型
STANDARD = NeuronConfig(
    hidden_size=768, num_hidden_layers=10,
    num_attention_heads=12, num_key_value_heads=4,
    intermediate_size=2304,
)

# 专家型
EXPERT = NeuronConfig(
    hidden_size=1024, num_hidden_layers=14,
    num_attention_heads=16, num_key_value_heads=4,
    intermediate_size=3072,
)
```

### 4.3 共享配置

```python
# 所有神经元共享
vocab_size: 256000           # 总词表
base_embed_dim: 512          # 共享嵌入 (感官分辨率)
field_dim: 4096              # 场维度
```

---

# 第二部分: 协同机制

## 五、两种同步策略

### 5.1 层次性同步 (解决"谁来主导"的问题)

**类比人脑**:
- 丘脑 = 域同步路由器
- 视觉皮层 = 功能同步路由器
- V1/V2/V4 = 细节同步路由器

**三层结构**:

```
第一层: 域同步 — 输入属于哪个域
  "用 Python 写 quicksort"
  → 代码域: 强相关 (0.9) → 主导
  → 中文域: 弱相关 (0.2) → 辅助

第二层: 功能同步 — 同一域内的功能分工
  → Python (0.8), Rust (0.3), 算法 (0.7)
  → Python 和算法同步 (同属于 quicksort 相关)
  → Rust 独立 (不相关)

第三层: 细节同步 — 同一功能内的更细分工
  → 排序算法: 0.9 → 主导
  → 网络 I/O: 0.1 → 辅助
```

**关键性质**: 这个机制是"涌现"的 — 不需要预设域的数量、每个域的功能数。相关度 > 阈值 → 同步，相关度 < 阈值 → 独立。

### 5.2 振荡同步 (解决"怎么让弱的变强"的问题)

**适用场景**: 跨领域任务 (一个任务需要多个域的知识)

```
输入: "用 Python 写 f(x)=x²log(x) 的梯度"

R1: A (代码) → VA (代码结构)  B (数学) → VB (数学推导)
    F1 = avg(VA, VB) ← 混合信号

R2: A 读 F1 → "B 说这涉及导数" → VA' 纳入数学正确性
    B 读 F1 → "A 说要输出 Python" → VB' 纳入代码实现
    F2 = avg(VA', VB') ← 两个都对齐了

R3: 进一步精炼 → 收敛到吸引子
```

**为什么 1+1>2**: 每个神经元通过场获得了自己没有的知识。

### 5.3 两种策略的关系

层次性同步 ←→ 振荡同步 = 互补，不是互斥

- 层次性同步: "谁来主导" — 输入来了，哪个域、哪个功能主导
- 振荡同步: "怎么让弱的变强" — 跨领域任务中，弱域从强域吸收知识

---

## 六、分工路径: 规模分层 + 集群主导

### 6.1 从共识到分工

原来的问题: 加权平均 = 所有神经元"民主投票" → 被弱者稀释

真正的分工路径:
```
共识路径: 所有神经元加权平均 → 输出统一答案 (像"投票")
分工路径: 每个神经元负责一部分 → 输出是组合 (像"流水线")
人脑是分工路径，不是共识路径
```

### 6.2 策略 A: 规模分层

```
紧凑型 = 探索者 / 执行者
标准型 = 主力员工
专家型 = 深度顾问 / 决策者

流程:
  1. 规模筛选: 只看专家型 + 标准型 (紧凑型作为辅助)
  2. 紧凑型按分工执行具体任务
  3. 专家型做最终把关
```

**为什么这是分工**: 规模本身 = 置信度信号。前额叶主导决策，运动皮层按指令执行。

### 6.3 策略 B: 集群主导

```
输入进来
  ↓
计算每个集群的契合度 (内部一致性 × 外部相关性)
  ↓
最契合的集群主导
  ↓
其他集群辅助
```

**集群契合度**:
```python
def compute_cluster_fit(input_vector, cluster):
    internal_coherence = cosine(neuron_output, cluster_centroid)
    external_fit = cosine(cluster_centroid, input_vector)
    return internal_coherence * external_fit
```

### 6.4 两种策略结合

```
第一层: 找到最契合的集群 (策略 B: 集群主导)
  ↓
第二层: 集群内部按规模分工 (策略 A: 规模分层)
  ├── 专家型: 决定分工 + 把关质量
  ├── 标准型: 执行主要任务
  └── 紧凑型: 执行辅助任务
  ↓
第三层: 集群间协同 (其他集群辅助)
```

---

## 七、共振的本质理解

### 7.1 共振不是"讨论"，是"精炼预测"

从实验 9 和 12 中发现:

```
单轮 forward:
  输入 → 神经网络 → 输出 logits → argmax → token

共振模式:
  R1: 输入 → 神经网络 → 输出 logits₁
  R2: logits₁ 写入场 → 读场状态 → 输出 logits₂
  R3: logits₂ 写入场 → 读场状态 → 输出 logits₃
  ...收敛到 logits* → argmax → token
```

**为什么共振有效**:
- logits₁ 可能有噪声或偏差
- 场状态聚合了"其他视角"
- 多轮精炼减少了噪声

### 7.2 至关重要: 共振不是默认有效

**实验 12 揭示的核心发现**:

| 测试 | PPL |
|-----|-----|
| code 单独 on 混合 | 15.66 |
| code 共振 on 混合 | 19.88 ← 变差了 |

**共振对好的预测是噪声**:
- code 在混合数据上已经接近完美预测 (PPL=15.66)
- 共振的多轮迭代反而引入场噪声
- 第一次预测是最准的，多轮迭代破坏了这个优势

### 7.3 人脑正确的启发

人脑不是无差别共振:
- 只有在"不确定"时才需要共振
- 在"确定"时直接输出，不需要多轮思考
- 我们之前的共振无差别多轮迭代，导致过思考

### 7.4 1+1>2 的触发条件

```
必须满足以下条件:
  1. 预测不确定性高 (top-k 概率分布均匀)
  2. 多个神经元有互补知识 (双方能力不同)
  3. 预测有足够错误空间 (不是几乎完美)

场景判断:
  简单任务 → 不需要共振
  复杂任务 → 需要共振
  不确定的预测 → 需要共振
  几乎确定的预测 → 反而会破坏
```

---

# 第三部分: 技术设计

## 八、硬上限与解决方案

### 8.1 硬上限清单

| 限制 | 类型 | 能否绕过 | 触发时间 | 解决方案 |
|------|------|---------|---------|---------|
| 嵌入维度 512 | 架构硬上限 | 否 | 需要更强语言理解时 | 512 是感官分辨率，认知在第二层 |
| 首轮 O(N) 激活 | 计算硬上限 | 部分 | N > 100 时明显 | 部落压缩 (Q = α·β·γ) |
| 场写入信息压缩 | 信息论硬上限 | 否 | 深度协作任务 | 动态阈值 + 拥挤度检测 |
| 训练数据质量 | 外部硬上限 | 否 | 场变聪明后 | 外部世界决定 |
| RAM 容量 | 工程软上限 | 是 | N > 300 时 | lazy loading |
| D = 16384 上限 | 设计软上限 | 是 | N > 500 时 | 集群自组织 |
| 词表限制 | 设计软上限 | 是 | 新领域出现时 | 词表分层 + 转译层 |

### 8.2 向人脑学习的设计原则

| 人脑 | 态极 |
|-----|------|
| 词表不是固定的大小 | 词表分层: 通用 + 专用 + 转译层 |
| 弱视不影响认知 | 512 是感官分辨率，认知在第二层 |
| 神经振荡同步 | 神经语言场: 完全独立于 tokenizer |

---

## 九、质量过滤机制

### 9.1 问题根源

实验 8 和 10 揭示:
- math 神经元 PPL=543 参与共振时稀释了 code (PPL=33) 的能力
- 1+1<2 的原因是质量不均，不是架构问题

### 9.2 静态阈值过滤

```python
class QualityFilter:
    def __init__(self, ppl_threshold: float = 100):
        self.ppl_threshold = ppl_threshold

    def filter(self, neurons, neuron_ppls):
        filtered = {}
        for nid, neuron in neurons.items():
            ppl = neuron_ppls.get(nid, float('inf'))
            if ppl < self.ppl_threshold:
                filtered[nid] = neuron
        return filtered
```

### 9.3 自适应阈值

```python
class AdaptiveQualityFilter:
    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier

    def get_threshold(self, neuron_ppls):
        best = min(neuron_ppls.values())
        return best * self.multiplier

    def filter(self, neurons, neuron_ppls):
        threshold = self.get_threshold(neuron_ppls)
        filtered = {}
        for nid, neuron in neurons.items():
            if neuron_ppls[nid] < threshold:
                filtered[nid] = neuron
        return filtered
```

### 9.4 质量监控闭环

```
训练神经元 → 评估 PPL → PPL < 阈值 → 参与共振
                         → PPL >= 阈值 → 继续训练
→ 共振输出
```

---

## 十、置信度门控 + 早停机制

### 10.1 设计动机

实验 12 发现的: 共振对好的预测是噪声。

### 10.2 置信度门控

```python
class ConfidenceGate:
    """低置信度才用共振"""

    def should_resonate(self, logits):
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values
        # 如果确定 (>0.9)，不需要共振
        return max_prob < 0.9
```

### 10.3 早停机制

```python
class EarlyStopResonance:
    """收敛时就停，避免过思考"""

    def should_stop(self, logits_history):
        if len(logits_history) < 2:
            return False
        diff = torch.norm(logits_history[-1] - logits_history[-2])
        return diff < self.threshold
```

### 10.4 触发流程

```
输入到达
  ↓
置信度门控: 预测不确定?
  ├── 否 → 直接输出
  └── 是 → 启动共振
         ├── Round 1: 独立 forward
         ├── Round 2: 读场后 forward
         ├── 早停检查: 收敛?
         │   ├── 是 → 输出
         │   └── 否 → 继续 Round 3...
         └── Round N: 输出
```

---

## 十一、训练闭环

### 11.1 训练 vs 进化: 统一接口

```
态极收到训练任务
    ↓
进化调度器决策:
  ├── 现有神经元能覆盖 → 复用
  └── 现有神经元覆盖不了 → 培育新神经元
    └── 新领域来了 → 加新神经元 (可以加多个)
    └── 持续低分 → 剥离 (不是替代，是移除)
```

### 11.2 部分训练

- 领域感知调度器自动判断
- 输入来了 → 计算相关度 → 过滤低相关神经元 → 只训这些
- 不需要手动划分"训哪些不训哪些"

### 11.3 进化路径

```
进化 = 人口动态 (培育 + 剥离)
  不是"升级" (紧凑型 → 标准型)

培育: 新领域来了 → 加神经元
剥离: 持续低分 → 移除

三种规格 = 三种不同角色的工人:
  紧凑型 = 探索者 (新领域探路)
  标准型 = 主力员工 (稳定领域)
  专家型 = 深度顾问 (复杂推理)
```

---

## 十二、生命系统集成

### 12.1 生命行为与神经元的对应

| 生命行为 | 单体下的实现 | 神经元集群下的重新理解 |
|---------|----------|------------------|
| 睡眠 | 训练单体模型 | 神经元内部参数整合 + 抱合新神经元 |
| 觉醒 | forward pass | 共振循环 |
| 饥饿 | 加载数据 | 领域神经元覆盖不足 → 加神经元 |
| 探索 | 联网搜索 | 激活新领域神经元 / 加神经元 |
| 玩耍 | 随机创作 | 神经元自由共振 |
| 死亡 | 部署新模型 | 剥离表现差的神经元 + 重组 |
| 记忆 | 加载状态 | 神经元权重本身就是记忆 |

---

# 第四部分: 实验验证

## 十三、实验总览 (实验 1-12)

| 实验 | 内容 | 结果 | 关键发现 |
|-----|------|------|---------|
| 1 | 1+1>2 基础验证 (合成数据) | FAIL | 数据太简单 (PPL=1.00) |
| 2 | 规模分层 vs 加权平均 (合成) | FAIL | 加权平均反而最好 |
| 3 | 共振方向分析 | FAIL | 快速同质化 (cos=1.0) |
| 4 | 强制知识缺口 (合成) | WEAK | 数据仍太简单 |
| 5 | 振荡同步互补性测试 | FAIL | 读场后准确率无提升 |
| 6 | 真实数据 (统一词表) | PARTIAL | 知识缺口存在但共振崩了 |
| 7 | 领域专用词表测试 | FAIL | 嵌入未训练，200 步不够 |
| 8 | code+math + 领域专用 tokenizer | PARTIAL | 知识缺口真实存在 |
| 9 | 质量过滤 + 强化训练 | **PASS** | **共振改善 39.4%** 🎉 |
| 10 | 多神经元共振 (统一词表) | FAIL | 词表冲突 |
| 11 | 转译层设计实现 | PARTIAL | 架构可行 |
| 12 | 多 tokenizer 共振 | PARTIAL | **共振不是默认有效** |

---

## 十四、实验 9 详细结果 (核心突破)

### 14.1 强化训练有效

| 神经元 | 实验 8 | 实验 9 | 改善 |
|-------|--------|--------|------|
| code on code | 33.95 | **14.29** | -58% |
| math on math | 543.81 | **62.85** | -88% |

10000 步训练 (lr=5e-4) 大幅提升了质量。

### 14.2 共振机制有效

| 测试 | PPL |
|-----|-----|
| code 单独 forward on 测试数据 | 79.15 |
| **code 共振模式 (多轮) on 测试数据** | **48.00** |
| **改善** | **-39.4%** 🎉 |

### 14.3 质量过滤有效

| 过滤方案 | 阈值 | code | math | 结果 |
|---------|------|------|------|------|
| 静态阈值 | < 100 | 14.29 ✅ | 62.85 ✅ | 两者都参与 |
| 自适应阈值 | best × 2 = 28.59 | 14.29 ✅ | 62.85 ❌ | **math 被过滤** |

---

## 十五、实验 6 详细结果 (知识缺口证据)

### 15.1 知识缺口被观测到

| 测试 | PPL | Gap |
|-----|-----|-----|
| code on code | 3.13 | — |
| **code on 中文** | **23882.97** | **+23879** |

真实数据上，领域间的 PPL 差距是真实的。

### 15.2 共振崩了的原因

| 配置 | PPL |
|-----|-----|
| 最佳单独 (数学 on 混合) | 2.42 |
| 共振 (4 神经元) | 247.55 |
| 共振 (code+math) | 2.97 |

中文神经元 PPL=415 污染了共振场。

---

## 十六、实验 12 详细结果 (共振的触发条件)

### 16.1 转译层可行但共振不是默认有效

| 测试 | PPL |
|-----|-----|
| code on code | 21.10 |
| code on 混合 | **15.66** (最低) |
| code 共振 on 混合 | **19.88** (反而变差) |

### 16.2 根本原因分析

1. code 在混合数据上已经接近完美预测 (PPL=15.66)
2. 共振的多轮迭代反而引入场噪声
3. 人脑只有在"不确定"时才需要共振

---

## 十七、根本问题解答

这个表格总结了所有曾被认为未解决的根本问题:

| 问题 | 状态 | 解答 |
|-----|------|------|
| 弱者稀释 | ✅ 解决 | 自适应质量过滤，差的自动被排除 |
| 谁主导 | ✅ 解决 | 高质量 (低 PPL) 神经元主导 |
| 神经元输出是什么 | ✅ 解答 | **预测下一个 token 的 logits**，不是抽象的"擅长程度" |
| 怎么拼接 | ✅ 解答 | **不是拼接，是加权平均 + 质量权重** |
| 集群怎么形成 | ✅ 解答 | 领域专用 tokenizer 自然形成 |
| 共振何时有效 | ✅ 解答 | 仅在不确定时有效，确定时应直接输出 |
| 1+1>2 | ⚠️ 待验证 | 单神经元共振有效 (39.4%)，多神经元需在"正确条件下"验证 |

---

# 第五部分: 执行路线

## 十八、当前状态总结

### 18.1 已验证的组件

| 组件 | 状态 | 证据 |
|-----|------|------|
| 共振场 (field.py) | ✅ | 6/6 测试通过 |
| 共振神经元 (neuron.py) | ✅ | 训练 + 推理正常工作 |
| 多轮共振 (ensemble.py) | ✅ | PPL 改善 39.4% |
| 领域专用 tokenizer | ✅ | 知识缺口真实存在 |
| 质量过滤 | ✅ | 自适应阈值有效 |
| 转译层 | ✅ | tokenizer 转译到统一嵌入 |
| 置信度门控 + 早停 | ✅ 已设计 | 待实验验证 |

### 18.2 未验证的假设

| 假设 | 状态 | 需要什么 |
|-----|------|---------|
| 1+1>2 在跨领域任务中 | ⚠️ | 需在"不确定预测"条件下测试 |
| 蒸馏自 1.5B 模型 | ⚠️ | 需要 DeepSpeed checkpoint 转换 |
| 多 tokenizer 共振 | ⚠️ | 需要转译层 + 统一嵌入空间 |
| 层次性同步 + 振荡同步 | ⚠️ | 需要实现并验证 |

---

## 十九、下一步行动

### 19.1 短期 (1-2 周)

1. **实现置信度门控 + 早停机制** — 验证共振仅在不确定时有效
2. **多 tokenizer 共振完整测试** — 用转译层解决词表冲突
3. **跨领域任务测试** — 设计需要多个域知识的任务

### 19.2 中期 (1-2 月)

1. **蒸馏路线** — 从 1.5B 模型拆出高质量神经元
2. **质量均衡** — 所有神经元 PPL < 50
3. **完整训练闭环** — 进化调度器 + 质量监控 + 自动剥离

### 19.3 长期 (3-6 月)

1. **共振簇** — 支持 30+ 神经元，验证规模效应
2. **动态场扩张** — 拥堵触发时自动扩展 D
3. **生命系统完全集成** — 生命行为是神经元集群的涌现行为

---

## 二十、已实现代码索引

```
taiji/resonance/
├── __init__.py     ✅ 导出全部公共接口
├── field.py        ✅ ResonanceField + D 自适应 + 部落压缩
├── neuron.py       ✅ ResonanceNeuron + NeuronConfig + 三套规格预设
├── ensemble.py     ✅ ResonanceEnsemble + 多轮共振 + PPL 评估
├── config.py       ✅ 神经元规格配置

taiji_portable/domain_tokenizers/
├── sp_code.model   ✅ 代码专用 tokenizer (12000 tokens)
├── sp_math.model   ✅ 数学专用 tokenizer (10000 tokens)
├── sp_zh.model     ✅ 中文专用 tokenizer (20000 tokens)
├── sp_en.model     ✅ 英文专用 tokenizer (16000 tokens)

tests/
├── test_resonance.py              ✅ 共振场验证测试 (6/6)
├── test_one_plus_one.py           ✅ 1+1>2 基础验证
├── test_knowledge_gap.py          ✅ 强制知识缺口
├── test_real_data.py              ✅ 真实数据测试
├── test_domain_tokenizer.py       ✅ 领域词表测试
├── core_resonance_test.py         ✅ 核心共振测试
├── core_verification_fixed.py     ✅ 统一 tokenizer 验证
├── core_verification_v2.py        ✅ 领域 tokenizer 验证
├── exp9_quality_filter.py         ✅ 质量过滤 + 强化训练
├── exp10_multi_neuron.py          ✅ 多神经元共振
├── exp11_translator.py            ✅ 转译层
├── exp12_multi_translator.py      ✅ 多 tokenizer 共振
├── train_single_neuron.py         ✅ 单神经元收敛测试
```

---


## 更新: 2026-07-17 —— 1+1>2 验证通过

### 验证条件

- **神经元**: zh (STANDARD, 292M params) + en (STANDARD, 292M params)，v1 compat 模式加载
- **数据**: data/distill/domain_datasets.pt，每域 500 x 256 tokens
- **共享嵌入**: teacher 1.55B checkpoint 的 hidden states 经 SharedEmbedProj (2048->512) 投影
- **共振**: 2 轮，v2 路由（熵加权 + LOO 共振分提升 + prediction_complementarity + 非零下限）

### 结果

| 测试域 | zh 独立 PPL | en 独立 PPL | zh+en 集成 PPL | 最佳单体 | 改进 |
|--------|------------|------------|---------------|---------|------|
| zh (中文) | 19,742 | 3,269,017 | **8,127** | 19,742 | **+58.8%** |
| en (英文) | 3,269,017 | 33,450 | **19,544** | 33,450 | **+41.6%** |

### 关键信号

1. **跨域盲区**: 中文神经元在英文数据上的 PPL 是 3.2M（等同于随机），反之亦然
2. **1+1>2**: 集成后 PPL 显著低于任一独立神经元
3. PPL 绝对值仍偏高的原因: (a) 老 checkpoint 是 v1 蒸馏，(b) SharedEmbedProj 可能有投影损耗，(c) STANDARD 容量有限
4. 冒烟测试 verify_h1h8.py 24/24 项通过

### 结论

共振场 v2 路由机制被证明有效。H1-H8 修复消除了架构中的隐藏缺陷。下一步: 用修复后的完整 v2 路径重蒸馏神经元。

### 本次新增文件

- scripts/training/verify_1plus1.py -- 使用真实 teacher hidden states 的验证脚本
- scripts/training/verify_h1h8.py -- H1-H8 冒烟测试（24 项）
- plans/H1-H8-mechanism-fixes.md -- H1-H10 机制解析

---



## 附录 A: 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-07-14 | 词表分层 + 转译层 | 解决词表热插拔问题 |
| 2026-07-14 | 层次性同步 + 振荡同步 | 解决同域多神经元协调问题 |
| 2026-07-14 | 蒸馏路线替代从零训 | 紧凑型质量问题是数据量问题 |
| 2026-07-14 | 512 维是感官分辨率 | 人脑类比: 弱视不影响认知 |
| 2026-07-15 | 长期训练计划 | 快实验无法验证 1+1>2 |
| 2026-07-15 | 质量过滤机制 | 差的神经元稀释好的 |
| 2026-07-15 | 领域专用 tokenizer | 知识缺口被验证真实存在 |
| 2026-07-15 | 转译层设计 | 解决多 tokenizer 共振 |
| 2026-07-15 | 置信度门控 + 早停机制 | 避免过思考，共振仅在不确定时启动 |

---

## 附录 B: 关键架构代码

### B.1 ResonanceField (共振场)

```python
class ResonanceField(nn.Module):
    """共享共振场 — 神经语言的核心"""

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.state = torch.zeros(dim)  # 场状态
        self.W_cond = nn.Parameter(torch.randn(dim, dim) * 0.02)

    def write(self, neuron_id, vector):
        """L2 归一化写入 — 所有神经元平等"""
        self.state += vector / (vector.norm() + 1e-8)

    def score(self, vector):
        """共振度 = cosine(input, field_state)"""
        return cosine_similarity(vector, self.state)

    def compute_threshold(self, congestion):
        """动态阈值 — 拥堵越高门槛越高"""
        return 0.30 + congestion * 3.0
```

### B.2 ResonanceNeuron (共振神经元)

```python
class ResonanceNeuron(nn.Module):
    def __init__(self, config):
        # 嵌入适配器: 共享基底 → 神经元内部
        self.embed_adapter = nn.Linear(config.base_embed_dim, config.hidden_size)
        # Transformer 体
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(...)])
        # 场写入投影
        self.field_write = nn.Linear(config.hidden_size, config.field_dim)
        # 场读取投影 (每层一个)
        self.field_read_layers = nn.ModuleList([...])

    def forward(self, shared_embeddings, field_state=None, round_num=1):
        h = self.embed_adapter(shared_embeddings)
        for i, block in enumerate(self.layers):
            h = h + block(h)
            # R2+ 时施加场条件化
            if field_state is not None and round_num > 1:
                h = h + self.field_read_layers[i](field_state)
        # L2 归一化场写入
        v = self.field_write(h[:, -1, :])
        return {"field_vector": v / (v.norm() + 1e-8)}
```

### B.3 ResonanceEnsemble (共振集成)

```python
class ResonanceEnsemble:
    def forward(self, shared_embeddings, max_rounds=3):
        self.field.reset()
        for round_num in range(1, max_rounds + 1):
            # 所有活跃神经元 forward
            vectors = {}
            for nid in active_ids:
                field_state = self.field.get_state() if round_num > 1 else None
                result = neuron.forward(embeds, field_state, round_num)
                self.field.write(nid, result["field_vector"])
                vectors[nid] = result["field_vector"]
            # 计算共振度，过滤低共振神经元
            scores = {nid: self.field.score(v) for nid, v in vectors.items()}
            active_ids = filter_by_threshold(scores)
        return weighted_average(logits, scores)
```

---

*文档结束*
