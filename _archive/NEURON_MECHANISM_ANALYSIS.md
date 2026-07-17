# Taiji Neuron 机制详细解析 + 隐藏问题分析

> 版本: v2.2
> 日期: 2026-07-16
> 范围: 三层架构、神经元内部机制、共振场、共振循环、训练管线，以及阻碍 1+1>2 的隐藏问题

本文分两部分:前半是机制解析(现有代码怎么跑),后半是隐藏问题(H1~H9,按致命/架构性/残留分级)。
后者才是决定 1+1>2 能否成立的关键,每条都带代码出处和后果。

---

## 一、架构全景

### 1.1 三层抽象

```
Level 0: 通用分词器 (256K vocab)
    |  token_id -> shared embedding lookup (base_embed_dim=512)
    v
Level 1: 领域神经元 (N 个独立 Transformer)
    |  embed_adapter -> Transformer blocks -> field_write / field_read
    v
Level 2: 共振场 (4096-dim 共享向量空间)
    |  L2-normalized writes -> cosine similarity scoring -> complementarity
    v
输出: per-position 加权 logits -> 下一 token
```

核心思想:用一个 4096 维共享向量空间替代"单体大模型内部隐状态",
作为知识协作的媒介。每个神经元独立处理输入、向场写入自己的理解,
再从场中读取其他神经元的理解来修正自身输出。

"1+1>2" 的含义:N 个神经元的共振集成 PPL 低于任一单独神经元的 PPL。
这要求场通信承载的信息足够丰富,且加权机制能正确地按位置选择最合适的神经元。

### 1.2 信息流回路

```
输入 token_ids [B, L]
    |
    v
共享嵌入表 Embedding(vocab=256000, dim=512)
    |  shared_embeddings: [B, L, 512]
    |
    +----> neuron_A.forward(emb)
    |           |
    |           v
    |       embed_adapter: 512 -> hidden_A
    |           |
    |           v
    |       TransformerBlocks x N (RoPE + GQA + SwiGLU)
    |           |
    |           +<--- field_read (round 2+): gated per-position
    |           |
    |           v
    |       attention-pooled field_write -> [B, 4096] (L2-normalized)
    |           |
    |           v
    |       lm_head -> [B, L, 256000]
    |
    +----> neuron_B.forward(emb, field_state=...)  读 A 写入的场
    |           ... 同上 ...
    |
    v
集成场(累积所有写入)
    |
    v
per-position 加权融合 -> weighted_logits [B, L, 256000]
```

---

## 二、神经元内部机制 (ResonanceNeuron)

源文件: taiji/resonance/neuron.py

每个神经元 = 独立 Transformer + 场接口,完全复用 layers.py 的 TransformerBlock、RMSNorm,对底层零改动。

### 2.1 embed_adapter

`nn.Linear(base_embed_dim=512, hidden_size, bias=False)`

把共享嵌入(512 维)投影到本神经元的隐维度(512/768/1024)。
这是神经元拥有"概念空间"的唯一入口:所有神经元共享一个词表,
但各自把同样的 512 维嵌入投进自己不同维度的表示空间。

### 2.2 Transformer body

复用 layers.py,每层结构:
- pre-norm (RMSNorm) -> attention (RoPE + GQA) -> 残差
- pre-norm (RMSNorm) -> SwiGLU FFN -> 残差

GQA: num_kv_heads < num_attention_heads,K/V 头共享,省显存。
SwiGLU: feed_forward 由 w_gate(silu) * w1 相乘再 w2 投回。

层数随 spec 变化:compact=6,standard=10,expert=14。

### 2.3 field_write (v2: 注意力池化写入)

旧(v1):取最后一个 token 的隐状态 `h[:, -1, :]` 投到 4096 维。

新(v2,本次改动):
```
attn_scores = matmul(h, field_pool_query) * scale   # [B, L]
attn_weights = softmax(attn_scores, dim=-1)         # [B, L]
pooled = sum(attn_weights * h, dim=1)               # [B, hidden]
v_raw = field_write(pooled)                          # [B, D]
v = v_raw / (||v_raw|| + 1e-8)                       # [B, D] L2 归一
```

`field_pool_query` 是一个可学习参数(`randn * 0.02`),决定序列里哪些位置
进入场写入。意图:用整条序列最显著的概念,而不是末 token。
输出 dict 额外带 `field_attn_weights`,供调试/可视化。

注意隐藏问题 H1: field_pool_query 是 v2 新增参数,旧 checkpoint 里没有。

### 2.4 field_read (v2: 按位置门控读取)

每层一个投影 `field_read_layers[i]: Linear(field_dim, hidden)`,
把 [D] 场状态投回 [hidden]。round 2+ 在每层注入:

旧(v1):`h = h + conditioning.unsqueeze(0).unsqueeze(0)`(广播给所有位置同一向量)

新(v2):per-position 门控
```
projection = conditioning[None, None, :]            # [1, 1, hidden]
gate = sigmoid(field_read_gate(h))                  # [B, L, 1]
h = h + gate * projection                           # 每个位置自行决定吸收量
```

`field_read_gate` 是 `Linear(hidden, 1)`,决定每个位置对场信号的门开度。
意图:不同位置需要不同程度的场协作(就跟不同 token 需要不同上下文一样)。

注意隐藏问题 H1: field_read_gate.weight/.bias 也是 v2 新增,旧 checkpoint 缺失。

### 2.5 NeuronConfig 三档规格

| spec     | hidden | layers | heads | kv_heads | FFN  | 总参数(不含共享嵌入) |
|----------|--------|--------|-------|----------|------|-----------|
| compact  | 512    | 6      | 8     | 2        | 1536 | ~24M      |
| standard | 768    | 10     | 12    | 4        | 2304 | ~59M      |
| expert   | 1024   | 14     | 16    | 4        | 3072 | ~118M     |

另设 TINY_TEST(hidden=256, layers=2, field_dim=512)供烟雾测试。

设计为能在 CPU 上独立训练的最小可理解文本的配置。

### 2.6 lm_head 与 fingerprint

- `lm_head: Linear(hidden, vocab=256000)`,只用于 PPL 评估 / 预训练;
  推理生成时也会用。
- fingerprint:冻结的方向向量 buffer,freeze_fingerprint() 把
  field_write.weight 的行均值归一化存下来,供未来轻量预筛选用(当前未用)。

### 2.7 旁路 channel (side_channels)

为 P1 预留:establish_side_channel(peer_id) 在两个常共激活神经元之间
建立 Linear(field_dim, hidden) 直连,绕过场做点对点通信。当前未启用。

---

## 三、共振场机制 (ResonanceField)

源文件: taiji/resonance/field.py

场是一个 D=4096 维向量空间,所有写入都 L2 归一化,读出累积状态。
它是架构的"神经语言",独立于 tokenizer(Level 0)和神经元概念空间(Level 1)。

### 3.1 L2 归一化写入(均等"音量")

write(neuron_id, vector):
```
v_norm = vector / (||vector|| + 1e-8)   # 不管神经元多大,写入范数=1
state = state + v_norm (B==1) 或 + v_norm.sum(dim=0) (B>1)
```

关键性质:神经元大小不决定"声音大小",只决定"方向"。
compact/expert 写入场后是平等的,靠方向(语义)参与共振,不是靠范数(体量)。

### 3.2 score (对齐度)

`score(vector) = cosine(v, state)`,范围 [-1, 1]。
高=与集体当前方向一致。batch 时先在 dim=0 上取均值。

### 3.3 complementarity_score (v2 新增:正交分量)

```
alignment = cosine(v, f)
orthogonal = v_norm - alignment * f_norm
return ||orthogonal||
```

量的是一个神经元给定之外能为场补充多少"新信息"。
- 高(~1):神经元带来场里还没有的信息(方向正交)
- 低(~0):和场已有的重复

### 3.4 combined_score (对齐 + 互补混合)

`combined_score(v, alpha)`:把对齐映射到 [0,1] 再和互补按 alpha 混。
alpha=0 纯对齐(旧行为),alpha=1 纯互补,0.5 平衡。
当前在 field 模块里提供了,但 ensemble 路由用的是 1+complementarity 的加成,不是 combined。

### 3.5 directional_congestion + compute_threshold (拥堵过滤)

```
congestion_i = 平均 max(0, cosine(v_i, v_j))   对所有活跃神经元 j
threshold_i = 0.30 + congestion_i * 3.0
```

方向拥挤(大家都往一个方向)= 需要更高共振分数才能继续留在活跃集。
低拥堵 0.1 -> T=0.60(容易进);高拥堵 0.85 -> T=2.85(几乎进不来)。
用于 round 2+ 过滤弱神经元。

### 3.6 W_cond (4096 x 4096, ~16.7M 参数)

`self.W_cond = nn.Parameter(randn(dim,dim)*0.02)`。
docstring 说它"学习哪种共激活模式产生好输出",但整个文件里没有任何一处用到它(见隐藏问题 H8)。

---

## 四、共振循环 (ResonanceEnsemble)

源文件: taiji/resonance/ensemble.py

### 4.1 主流程

每条 forward 输入:

1. self.field.reset() 清空场状态。
2. QualityFilter 过滤弱神经元(PPL 太高的不参与)。
3. Round 1:所有活跃神经元独立前向(field_state=None),写入各自场向量,
   计算共振分数。
4. Gating 检查:若已自信(ConfidenceGate max_prob>阈值)或 ResonanceTrigger
   不满足 -> 直接返回 best 门神经元的 round1 logits,跳过共振。
5. Rounds 2+:每个神经元读 field.get_normalised_state() 做条件前向,
   写入场,重算分数;按拥堵动态阈值过滤弱神经元;EarlyStopResonance 检测
   logits 收敛则提前停。
6. 最终融合:return_logits=True 时,按 division_path 有无分流:
   - 有 division_path:分层 + 簇支配权重
   - 默认无 -> 走 v2 per-position 路由(见 4.3)

### 4.2 Gating 三机制(Experiment 12 经验)

- ConfidenceGate:top-1 概率超阈值时跳过共振(自信预测不需场噪声)
- EarlyStopResonance:相邻两轮 logits 相对 L2 差 < 阈值则停
- ResonanceTrigger:综合不确定性 + 互补 + 改进空间三条件决定是否共振

这三者都依赖 round1 分数和场状态。

### 4.3 最终融合 v2:per-position 熵路由 + 互补加成(默认路径)

division_path=None 时默认走这里:
```
for nid: 每个神经元的 logits 算逐位置熵 ent[B,L]
confidence = 1/(ent + 1e-8)                    # [N, B, L]
position_weights = softmax(confidence * 2.0)   # 跨神经元, 每位置独立

comp_boost = 1 + complementarity_score(v_nid)  # [N]
position_weights *= comp_boost                 # 提升"带来新信息"的神经元
position_weights /= 归一

weighted_logits = sum_i w_i * logits_i         # 逐位置加权
```

意图:每个位置独立选最自信的神经元,且给带来新知识的神经元加权。

注意隐藏问题 H5/H6/H7: 这条默认路径不用共振分数,且互补加成存在方向错配。

### 4.4 分层路径 (DivisionPath)

源文件: taiji/resonance/division.py,可选。
- ScaleLayering:expert(118M)决策 / standard(59M)执行 / compact(24M)辅助
- ClusterDominance:最佳簇主导、其余辅助
- 当前实现把每个神经元当独立簇(input_vec 用第一个神经元的场向量代理)

---

## 五、训练管线与 tokenizer

### 5.1 蒸馏数据 + neuron 间训练

- data/distill/domain_datasets.pt:5 领域 x 500 段 x 256 tokens
- data/real/domain_datasets.pt:zh 1468 / en 2000 / code 2000 / math 869 / general 2000,均 256 tokens
- scripts/training/distill_neurons.py:从 teacher 蒸馏领域专一神经元

### 5.2 训练 backend

taiji/training/:single(单神经)、joint(共振联合)、contrastive(对比)、
distill、checkpoint_bridge(teacher 桥接)、scheduler。

### 5.3 tokenizer

taiji/tokenizer_native_v2.py:256K vocab。
scripts/training/build_domain_tokenizers.py 构建领域分词器。
验证脚本用 load_teacher_model 拿 teacher 的真实 2048 维嵌入,
再用正交 Linear(2048,512) 投到 512 维喂给神经元。

### 5.4 验证 1+1>2

scripts/training/_verify_1plus1_real.py:load teacher + load 2 个 v2 neuron,
逐域算 ppl_single / ppl_ensemble,判断 imp = (best_single - p_both)/best_single。

---

## 六、v2 架构改进摘要(本次未提交改动)

三处改动已通过 CPU 烟雾测试(TINY_TEST 端到端 forward 成功),逻辑正确:

1. neuron.py:field_write 从末 token 改注意力池化(加 field_pool_query);
   field_read 从广播改 per-position 门控(加 field_read_gate)。
2. field.py:加 complementarity_score(正交分量)、combined_score(对齐+互补)。
3. ensemble.py:默认融合从全局标量 softmax 改 per-position 熵路由 + 互补加成。

详见下一节:这些改动本身合理,但叠在旧地基上,隐藏问题被放大。

---

## 七、隐藏问题分析(阻碍 1+1>2 的根因)

下面按"致命 > 架构性 > 残留"分级。每条带代码出处和后果。

### 致命级

#### H1. v2 神经元破坏已有 checkpoint 加载

证据(已实证):neuron_zh.pt(mtime 2026/7/15)解析其 data.pkl,
state_dict 共 105 个键,缺 field_pool_query、field_read_gate.weight、
field_read_gate.bias(CHECK 全 False);而 neuron.py 2026/7/16 才加这三个参数。

后果有两层:
- 直接崩:verify_distilled_neurons.py 的 load_neuron 和 _verify_1plus1_real.py 的
  load_v2 都是 load_state_dict(ckpt["state_dict"]),strict 默认 True。
  加载旧 checkpoint 抛 RuntimeError: Missing key(s): "field_pool_query",
  "field_read_gate.weight", "field_read_gate.bias"。
- 即便改 strict=False 也不保真:这三参数随机初始化(field_pool_query=randn*0.02,
  field_read_gate 随机),在未重训的旧 checkpoint 上跑等于给一个调好的前向
  注入未训练噪声——池化聚到错方向、门控随机吸收场信号。你以为测"共振改进",
  实际测"随机噪声扰动"。strict=False 救不了保真度,要么重训,要么显式把这 3 个
  参数的影响置零(如 gate.bias 置大负数让门趋 0、pool_query 初始化成均匀分布退化为 mean-pool)。

#### H2. 共振场把整个 batch 塌缩成一条 [D](跨样本串扰)

证据:field.py 的 write 对 B>1 走 state + v_norm.sum(dim=0),
B==1 走 state + v_norm.squeeze(0),state 是单条 register_buffer("state", zeros(dim));
get_normalised_state() 返回单条 [D]。

后果:round 2 全部 B 条序列读同一条场,样本互相串扰。
关键掩盖点:_verify_1plus1_real.py 用 max_rounds=1,round 2 不发生、
场只写不读,所以此 bug 在当前验证脚本里看不出来。一旦把 max_rounds 提到 >=2
(要真正测"共振")或拿去训练(batch_size=2),串扰立刻生效。
根因:场没有 per-sample 维度 B。修法:场改 [B,D],或硬性 batch_size=1。

### 架构性(决定 1+1>2 能否发生)

#### H3. 验证脚本里"共振"其实是关着的,测的是普通集成

证据:_verify_1plus1_real.py 里 ResonanceEnsemble(neurons, field, max_rounds=1)。
max_rounds=1 只跑 round 1,field_state=None、round_num=1,
neuron.py 的 field_read 分支根本进不去。

后果:所谓"ensemble PPL"只是两套互相独立的 logits 的加权平均,
场写纯开销。v2 的 per-position entropy routing 也只在各自 round-1 logits 上工作。
即你声称验证"共振让 1+1>2",实际验证"加权平均两个领域专一小模型能不能超过
较好的那一个"——对领域专一小模型,这恰是会因稀释失败的经典集成问题。
不是 bug,是实验设置错配了要验证的命题。

#### H4. 场是单向量瓶颈,传不了分位置/分时序的结构化协作

证据:neuron.py 的 field write 用 attention pooling 把 [B,L,hidden] 整条序列
压成 [B,hidden] 再投 [B,4096](再被 H2 塌成 [4096]);
round 2 另一个神经元读这一条 [4096] 经 field_read_layers[i] 投回 [hidden],
门控广播到每个位置。

后果:语言理解本质是分位置(token5 的"理解" != token15 的)。
一条全序列摘要向量只能传"整体氛围",传不了"位置7不确定、位置12很确定"这类
结构化信号。1+1>2(超加性)要发生,得有分位置结构化的知识交换,
单向量场在 write 端就把它压没了。v2 把 read 改 per-position 是对的,
但瓶颈在 write 端:写的就是一条向量,read 端怎么分位置也变不回分位置的结构信息。

#### H5. 共振分数被自贡献污染,且默认路由根本不用它

证据:ensemble.py round1 先(146行)把所有神经写入场,
后(151行)才 score(v_nid)=cosine(v_nid, state)。state 里已含 v_nid 自己。
每个分数都在量"和含自己的集体的对齐",自相关抬高分数,不是干净的跨神经共振。

更隐蔽的割裂:默认走 v2 的 else 分支(已确认 division_path=None 是默认),
那条路由完全不用这些 score,改用 logit entropy + complementarity。
即全场每轮算的共振分数(ConfidenceGate/EarlyStop/division_path 都依赖它)
在默认最终融合路径里"算了白算"。机制内部信号割裂:一套分数喂 gating,
另一套 entropy 才决定融合。

#### H6. complementarity 加成奖励"写法不同"而非"预测更准"

证据:ensemble.py 默认分支 comp_boost = 1 + complementarity_score(v_nid),
量的是该神经场向量相对当前场的正交分量大小(即"写了与众不同的方向")。

后果(接 H7 的均匀化):弱神经元 entropy 都差不多 -> 位置权重塌成 ~1/N,
于是 comp_boost 反客为主。结果路由偏向"写法特别"而非"这个位置预测更准"的神经元,
可能把权重推向更差的神经元。何况 complementarity 也是在含自贡献的场上算的(H5)。

#### H7. per-position 熵路由在当前弱神经元上退化为均匀稀释

证据:ensemble.py confidence=1/(entropy+1e-8),softmax 温度 2.0。
后果:小模型未充分训练时每个位置 entropy 都接近 log(vocab) 的均匀分布,
跨神经差异极小 -> softmax 输出 ~1/N。这正是验证脚本的 dilution 失败模式。
此路要起作用,前提是"神经元已各自充分训练、且真的在分位置 confidence 有差异"。
门建好了,但当前 checkpoint 没有足够的 confidence 信号去驱动它。

### 残留级

#### H8. W_cond(4096x4096 约16.7M 参数)从未使用

证据:field.py 第46行定义 self.W_cond = nn.Parameter(randn(dim,dim)*0.02),
全文件只有第12行 docstring 提到它。write/score/complementarity_score/combined_score
里没有任何一处用到。

后果:死参数,白吃显存、增大 checkpoint;若被 optimizer 收进去还会被无意义更新/正则。
要么用起来(场状态的门控/投影),要么删掉。

#### H9. hidden_before_write 语义变了,diversity loss 行为漂移

证据:v2 把 hidden_before_write 从 last-token 改成 attention-pooled 单条向量。
ensemble.py 的 diversity_lambda=0.01 是基于它做多样性正则。

后果:旧 checkpoint 训练时正则作用在 last-token 表征,新前向作用在池化表征,
两者不一致。不是崩,但旧 checkpoint 落到新代码的训练语义已漂移。

---

## 八、根因小结

前面三个 v2 改动单独看都合理,但它们叠在"场=单条4096向量 + 验证用 max_rounds=1
+ 分数自污染 + 新参数未训练"这个地基上,于是 1+1>2 不会发生:

真正能造成协作的 field_read(round 2+)在验证时是关的(H3);打开后又受 H2 串扰
和 H4 单向量瓶颈限制;最终融合靠的是 H7 均匀化的熵路由 + H6 错向的 complementarity。

GPU 落地前必须先解的顺序:
1. H1:加 checkpoint 兼容层(strict=False + 把 3 个新参数影响置零),否则跑不起来或失真。
2. H3 + 实验设置:验证脚本 max_rounds 提到 >=2,否则测的不是共振。
3. H2:场改 [B,D] 或强制 batch_size=1。
4. H4:场写从单向量升级为分位置/分槽的有结构写入——1+1>2 能否成立的真正瓶颈。
5. H5/H6/H7:共振分数改成"剔除自身后评分";路由用分数替代纯熵,避免弱神经元均匀稀释。
6. H8:删掉或启用 W_cond。

---

## 九、模块依赖图

```
config.py
  |
  +-- NeuronConfig ->--> ResonanceNeuron (neuron.py)
  |                       用 layers.py (TransformerBlock/RMSNorm)
  |
  +-- ResonanceField (field.py)  <-- 由 ResonanceEnsemble 持有
  |
  +-- ResonanceEnsemble (ensemble.py)
        |   持 neurons {id: ResonanceNeuron}
        |   持 field: ResonanceField
        |   可选: confidence_gate / early_stop / resonance_trigger (gating.py)
        |   可选: quality_filter (quality.py)
        |   可选: division_path (division.py: ScaleLayering + ClusterDominance + DivisionPath)
        |
        +-- forward() -> weighted_logits / field_state / scores

训练侧:
  scripts/training/distill_neurons.py
    -> checkpoint_bridge.py (load teacher)
    -> 训练单 neuron -> data/neurons_v2/neuron_<domain>.pt

验证侧:
  scripts/training/_verify_1plus1_real.py
    -> load_v2() --- 当前 strict=True, H1 会崩
    -> ppl_single/ppl_ensemble  --- max_rounds=1, H3 测的不是共振
```

---

## 十、Checkpoint 兼容性操作说明(重要)

旧 checkpoint(2026/7/15 及之前)与新 v2 neuron(2026/7/16)的参数差异:

| 参数                  | 旧 ckpt | 新代码 | 影响                  |
|-----------------------|---------|--------|-----------------------|
| embed_adapter.weight  | 有      | 有     | -                     |
| field_write.weight    | 有      | 有     | -                     |
| field_read_layers.*.weight | 有  | 有     | -                     |
| lm_head.weight        | 有      | 有     | -                     |
| norm.weight / layers.*.* | 有   | 有     | -                     |
| fingerprint           | 有      | 有     | -                     |
| field_pool_query      | 缺      | 新增   | 决定池化聚合点        |
| field_read_gate.weight | 缺     | 新增   | 决定场读取门控        |
| field_read_gate.bias   | 缺     | 新增   | 同上                  |
| W_cond                | 有(场) | 有(场)| 死参数(H8)          |

结论:任何用 load_state_dict(ckpt, strict=True) 加载旧 checkpoint 的路径,
在新代码下都会缺这 3 个键而崩。必须先做兼容处理再上 GPU。
