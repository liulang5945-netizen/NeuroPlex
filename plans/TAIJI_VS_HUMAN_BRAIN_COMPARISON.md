# NeuroPlex Population Network vs Biological Neural Systems

> 本文是机制类比说明，不是生物学等价声明。当前项目的工程基本单元是神经元群体，而不是单一大模型。

## 1. 核心对应关系

| NeuroPlex | 生物学启发 | 工程含义 |
|---|---|---|
| neuron population | 神经元群体 | 多个独立成员共同承担能力 |
| ResonanceField | 群体活动状态 | 共享通信介质，不承担完整语言模型功能 |
| field read/write | 突触输入/输出 | 神经元读取上下文并贡献部分表征 |
| peer channels | 局部回路 | 邻近成员形成兴奋/抑制关系 |
| sparse routing | 任务相关神经回路 | 每个输入只激活合适的成员 |
| coactivation/topology | 功能连接 | 从共同活动中形成群体结构 |
| field memory | 海马—皮层记忆线索 | 场状态可检索并注入后续推理 |
| sleep replay | 睡眠重放 | 在低干扰阶段巩固经验和连接 |
| neurogenesis | 神经发生 | 能力缺口触发新成员生成 |
| apoptosis/pruning | 凋亡/突触修剪 | 长期无贡献成员和弱连接被隔离或移除 |

## 2. 运行时循环

```text
感知 → 初始反应 → 写入共振场 → 同伴读取
  → 稀疏路由和质量门控 → 条件化再反应
  → 输出融合 → 记录活动/记忆 → 睡眠巩固
```

该循环强调的是群体协作和结构变化。某个 expert/relay neuron 可以承担跨域锚定，但它只是网络成员，不是中心教师。

## 3. 学习机制

| 阶段 | 训练目标 | 主要参数 |
|---|---|---|
| 个体形成 | 让新 neuron 学会自己的领域或功能 | neuron body、域输出头、局部适配器 |
| 同伴协调 | 让成员学会读写场和交换局部信号 | field read/write、side channels、跨规格投影 |
| 路由校准 | 让合适的成员在正确任务上被激活 | quality head、prototype、实例级 router |
| 经验巩固 | 让高价值记忆进入可复用路径 | field memory、LoRA、睡眠 replay |
| 群体成长 | 根据贡献和缺口调整结构 | neurogenesis、maturity、isolation、apoptosis |

这里的“对齐”是同伴协作和场空间对齐，不是把所有成员压缩成一个中心模型。

## 4. 设计边界

- 不把总体参数量当作唯一扩展指标；同时关注活动成员数、路由稀疏度、场贡献、连接健康度和任务覆盖。
- 不让单一 relay neuron 成为隐藏的全局瓶颈。
- 不用一次性的全量重训替代生命周期机制。
- 不把历史迁移脚本当作新神经元的必经路径。
- 所有生物类比都必须落到可观测变量、可回归测试和明确的工程行为。

## 5. 当前可验证指标

- population size / active neuron count;
- route selection and confidence/quality gating;
- field contribution and peer traffic;
- cross-domain alignment and fusion quality;
- memory write/retrieval/conditioned-generation effect;
- sleep consolidation persistence;
- new-neuron maturity, ablation gain, isolation, and revival.
