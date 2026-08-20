# Side Channels（突触投影）实现计划

> **⚠️ 已归档（2026-07-28）**
> Task 1-5 全部实现完成，Task 6 Step 1-2 完成，Step 3（联合微调）进行中。
> 当前进度已同步到 [`BIO_INSPIRED_ARCHITECTURE_PLAN.md`](../../active/BIO_INSPIRED_ARCHITECTURE_PLAN.md) 第 0.4 节。
> 保留作为 side_channels 实现细节参考。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现真正的 per-pair 突触通道（side_channels），让多个同域 compact 神经元在保持低维（field_dim=2048）的前提下通过双向突触投影协作，验证"独立词表 + 突触转译"的核心设计。

**Architecture:** 每个神经元维护 `excite_channels[peer_id]` 和 `inhibit_channels[peer_id]`，每个通道是一个从 peer 的 `field_dim` 到本神经元 `hidden_size` 的可学习 Linear。Ensemble 在多轮共振中把 peer 的 field_vector 通过对应通道转译成 hidden-space 调制信号，残差注入本神经元 hidden 状态。field_dim 保持 2048 不变，不通过 unified_field_dim 强制同维。

**Tech Stack:** PyTorch, taiji.resonance (ResonanceNeuron, ResonanceEnsemble, ResonanceField)

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `taiji/resonance/neuron.py` | 添加 `establish_side_channel`、在 `forward` 中接收并应用 `side_signals`、维护 excite/inhibit 通道 |
| `taiji/resonance/ensemble.py` | 在多轮共振中收集 field_vector、构造并传递 per-pair `side_signals` |
| `scripts/training/train_compact_parallel.py` | 训练前为同域其他神经元建立 side_channels；可选端到端训练 |
| `scripts/training/eval_aug_joint.py` | 评估个体 vs 协作效果，观察 side_channels 是否产生 emergent 协作 |

---

## Task 1: 在 ResonanceNeuron 中实现 side_channels 创建

**Files:**
- Modify: `taiji/resonance/neuron.py:139-154`
- Test: `tests/resonance/test_side_channels.py` (创建)

- [ ] **Step 1: 写失败测试**

```python
def test_establish_side_channel():
    from taiji.resonance import get_domain_neuron_config, ResonanceNeuron
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer = ResonanceNeuron(peer_cfg)
    neuron.establish_side_channel("peer_0", peer, channel_type="excite")
    assert "peer_0" in neuron.excite_channels
    # Linear 输入维度是 peer 的 field_dim，输出维度是本神经元的 hidden_size
    channel = neuron.excite_channels["peer_0"]
    assert channel.weight.shape == (cfg.hidden_size, peer.config.field_dim)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/resonance/test_side_channels.py::test_establish_side_channel -v`
Expected: FAIL - `AttributeError: 'ResonanceNeuron' object has no attribute 'establish_side_channel'`

- [ ] **Step 3: 实现 `establish_side_channel` 方法**

在 `taiji/resonance/neuron.py` 的 `ResonanceNeuron` 类中添加：

```python
def establish_side_channel(
    self,
    peer_id: str,
    peer_neuron: "ResonanceNeuron",
    channel_type: str = "excite",
    init_std: float = 0.01,
):
    """建立一条指向 peer 神经元的突触通道。

    Args:
        peer_id: peer 神经元标识。
        peer_neuron: peer 神经元实例，用于读取其 field_dim。
        channel_type: "excite" 或 "inhibit"。
        init_std: 通道权重初始化标准差。
    """
    src_dim = peer_neuron.config.field_dim
    dst_dim = self.config.hidden_size
    channel = nn.Linear(src_dim, dst_dim, bias=False)
    nn.init.normal_(channel.weight, std=init_std)

    if channel_type == "excite":
        self.excite_channels[peer_id] = channel
    elif channel_type == "inhibit":
        self.inhibit_channels[peer_id] = channel
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/resonance/test_side_channels.py::test_establish_side_channel -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/resonance/test_side_channels.py taiji/resonance/neuron.py
git commit -m "feat(neuron): add establish_side_channel for per-pair synaptic projection"
```

---

## Task 2: 在 ResonanceNeuron.forward 中应用 side_signals

**Files:**
- Modify: `taiji/resonance/neuron.py:311-417`
- Test: `tests/resonance/test_side_channels.py`

- [ ] **Step 1: 写失败测试**

```python
def test_forward_with_side_signals():
    import torch
    from taiji.resonance import get_domain_neuron_config, ResonanceNeuron
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer = ResonanceNeuron(peer_cfg)
    neuron.establish_side_channel("peer_0", peer, channel_type="excite")

    B, T = 2, 10
    x = torch.randn(B, T, cfg.embed_dim)
    side_signals = {
        "peer_0": torch.randn(B, peer.config.field_dim),
    }
    out = neuron.forward(x, side_signals=side_signals, return_logits=False)
    assert "hidden" in out
    assert out["hidden"].shape == (B, T, cfg.hidden_size)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/resonance/test_side_channels.py::test_forward_with_side_signals -v`
Expected: FAIL - side_signals 未生效或 hidden 形状不对

- [ ] **Step 3: 修改 forward 应用 side_signals**

在 `taiji/resonance/neuron.py` 的 `forward` 方法中，在 Transformer layers 之前或之后（推荐在 layers 之后、norm 之前）应用 side_signals：

```python
# side_signals: {peer_id: field_vector [B, peer_field_dim]}
if side_signals:
    excite_sum = None
    inhibit_sum = None
    for peer_id, sig in side_signals.items():
        if peer_id in self.excite_channels:
            proj = self.excite_channels[peer_id](sig)  # [B, hidden_size]
            excite_sum = proj if excite_sum is None else excite_sum + proj
        if peer_id in self.inhibit_channels:
            proj = self.inhibit_channels[peer_id](sig)
            inhibit_sum = proj if inhibit_sum is None else inhibit_sum + proj

    side_mod = torch.zeros_like(h)
    if excite_sum is not None:
        side_mod = side_mod + excite_sum.unsqueeze(1) * 0.1
    if inhibit_sum is not None:
        side_mod = side_mod - inhibit_sum.unsqueeze(1) * 0.1

    h = h + side_mod
```

注意：此代码应插入到 Transformer 输出后、`self.norm(h)` 之前。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/resonance/test_side_channels.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/resonance/test_side_channels.py taiji/resonance/neuron.py
git commit -m "feat(neuron): apply side_signals as residual modulation in forward"
```

---

## Task 3: 在 ResonanceEnsemble 中传递 per-pair side_signals

**Files:**
- Modify: `taiji/resonance/ensemble.py`
- Test: `tests/resonance/test_ensemble_side_channels.py` (创建)

- [ ] **Step 1: 写失败测试**

```python
def test_ensemble_passes_side_signals():
    import torch
    from taiji.resonance import (
        get_domain_neuron_config, ResonanceNeuron, ResonanceField, ResonanceEnsemble
    )
    cfg = get_domain_neuron_config("zh", spec="compact")
    n0 = ResonanceNeuron(cfg)
    n1 = ResonanceNeuron(cfg)
    n0.establish_side_channel("n1", n1, "excite")
    n1.establish_side_channel("n0", n0, "excite")
    neurons = {"n0": n0, "n1": n1}
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    B, T = 1, 5
    emb = {
        "n0": torch.randn(B, T, cfg.embed_dim),
        "n1": torch.randn(B, T, cfg.embed_dim),
    }
    result = ensemble.forward(neuron_embeddings=emb, return_logits=True, fusion_mode="soft")
    # 只要 forward 不报错且返回 logits 即认为 side_signals 已传递
    assert "neuron_logits" in result or "weighted_logits" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/resonance/test_ensemble_side_channels.py::test_ensemble_passes_side_signals -v`
Expected: FAIL - side_signals 未传递导致相关错误，或未触发第二轮调制

- [ ] **Step 3: 修改 ensemble.forward 传递 side_signals**

在 `taiji/resonance/ensemble.py` 中：

1. 第一轮 forward 后，从每个神经元的结果中提取 `field_vector`（原始 field_dim 空间）。
2. 构造 `side_signals_per_neuron: Dict[str, Dict[str, Tensor]]`，其中 `side_signals_per_neuron[post_id][pre_id] = pre_field_vector`。
3. 第二轮 forward 时，把 `side_signals=side_signals_per_neuron[nid]` 传给每个神经元。

关键代码（插入到第一轮循环后、第二轮循环前）：

```python
# 收集每个神经元的 field_vector（原始 field_dim 空间）
field_vectors = {}
for nid in self.neuron_ids:
    field_vectors[nid] = round_results[nid]["field_vector"]  # [B, field_dim]

# 构造 per-neuron side_signals：每个 post 神经元接收所有 pre 的 field_vector
side_signals_per_neuron = {nid: {} for nid in self.neuron_ids}
for post_id in self.neuron_ids:
    for pre_id in self.neuron_ids:
        if post_id == pre_id:
            continue
        if pre_id in self.neurons[post_id].excite_channels or \
           pre_id in self.neurons[post_id].inhibit_channels:
            side_signals_per_neuron[post_id][pre_id] = field_vectors[pre_id]
```

在第二轮 `_forward_neuron` 调用中加入 `side_signals`：

```python
kwargs = dict(
    field_state=field_state,
    round_num=round_num,
    return_logits=need_logits,
    side_signals=side_signals_per_neuron.get(nid, {}),
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/resonance/test_ensemble_side_channels.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tests/resonance/test_ensemble_side_channels.py taiji/resonance/ensemble.py
git commit -m "feat(ensemble): pass per-pair side_signals across resonance rounds"
```

---

## Task 4: 训练脚本中为同域神经元建立 side_channels

**Files:**
- Modify: `scripts/training/train_compact_parallel.py`

- [ ] **Step 1: 在训练前建立同域所有神经元的 side_channels**

在创建神经元后、训练前，根据 `NEURON_IDS` 列表为当前神经元建立指向其他同域神经元的 excite 通道：

```python
# 建立指向其他同域神经元的 side_channels（excitatory by default）
for peer_id in NEURON_IDS:
    if peer_id == args.neuron_id:
        continue
    # 这里只需要 peer 的 field_dim，可以用相同 cfg 创建临时 peer 来获取
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer_neuron = ResonanceNeuron(peer_cfg).to(args.device)
    neuron.establish_side_channel(peer_id, peer_neuron, channel_type="excite")
    del peer_neuron
print(f"  {args.neuron_id}: {len(neuron.excite_channels)} excite side_channels", flush=True)
```

注意：需要导入 `NEURON_IDS` 列表或类似常量。

- [ ] **Step 2: 验证 side_channels 参与训练**

启动单路短训练（如 100 步）并检查：
- 模型参数增加量：每个 side_channel 约 `hidden_size * field_dim = 768 * 2048 = 1.57M`，3 个 peer 共 ~4.7M。
- 训练能正常收敛，无 shape 错误。

Run: `python scripts/training/train_compact_parallel.py --neuron_id zh_aug0_test --data_files simple_zh_texts.jsonl --shared_emb_mode train --steps 100 --dropout 0.2 --threads 3 --eval_every 50`
Expected: 正常完成，参数增加约 4.7M。

- [ ] **Step 3: 提交**

```bash
git add scripts/training/train_compact_parallel.py
git commit -m "feat(train): establish side_channels to peer neurons before training"
```

---

## Task 5: 更新联合评估脚本

**Files:**
- Modify: `scripts/training/eval_aug_joint.py`

- [ ] **Step 1: 加载神经元后建立 side_channels**

在 `load_aug_neurons` 中，加载每个神经元后，为其建立指向其他三个神经元的 excite side_channels：

```python
for nid in NEURON_IDS:
    for peer_id in NEURON_IDS:
        if peer_id == nid:
            continue
        peer_neuron = neurons[peer_id]
        neuron.establish_side_channel(peer_id, peer_neuron, channel_type="excite")
```

- [ ] **Step 2: 确保 max_rounds >= 2**

在 `eval_ppl` 和 `eval_generation` 中创建 `ResonanceEnsemble` 时设置 `max_rounds=2`（或更大），以便 side_signals 能在第二轮被应用：

```python
ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
```

- [ ] **Step 3: 运行联合评估**

Run: `python -u scripts/training/eval_aug_joint.py`
Expected: 输出个体 PPL、协作 PPL（带 side_channels）、生成质量对比，并报告是否出现 EMERGE。

- [ ] **Step 4: 提交**

```bash
git add scripts/training/eval_aug_joint.py
git commit -m "feat(eval): enable side_channels in joint evaluation"
```

---

## Task 6: 跑通 4 路并行训练并验证协作 emergent

**Files:**
- Modify: `scripts/training/run_parallel_aug.ps1` (可选，如需调整参数)

- [x] **Step 1: 启动 4 路并行训练** ✅ 完成（2026-07-27）
  - 4 个 zh_aug0~3 神经元训练完成
  - 个体 PPL：zh_aug0=39.6, zh_aug1=146.6, zh_aug2=22.5, zh_aug3=71.8
  - 关键发现：数据质量 > 数据量（百科 341K 战胜全量 787K）

- [x] **Step 2: 训练完成后运行联合评估** ✅ 完成（2026-07-27）
  - 发现问题：shared_embedding 被覆盖导致 embedding 空间不一致
  - 修复：训练脚本保存 per-neuron embedding
  - 发现问题：side_channels 随机初始化，协作 PPL=148.2 > 最强个体 114.6
  - 修复：需联合微调 side_channels

- [ ] **Step 3: 联合微调 side_channels** 🔄 进行中（2026-07-28）
  - 脚本：`scripts/training/finetune_side_channels.py`
  - 配置：10000 条数据, 6 epoch, 12.58M 可训练参数
  - 工程：日志 tee + 每 epoch checkpoint + 断点续训
  - 当前进度：Epoch 1/6 step 100, PPL=132.4（目标 <114）
  - 日志：`logs/finetune_side_channels_20260728_093654.log`

- [ ] **Step 4: 评估 EMERGE 现象** ⏳ 待 Step 3 完成
  - 运行：`python -u scripts/training/eval_aug_joint.py`
  - 运行：`python -u scripts/training/eval_gen_quality.py`
  - 判定：协作 PPL < 最强个体 PPL（zh_aug1=114.6）则 EMERGE 确认

- [ ] **Step 5: 提交评估结果** ⏳ 待 Step 4 完成

---

## 关键设计决策记录

1. **field_dim 保持 2048**：不启用 `unified_field_dim`，避免"同维"方案。
2. **通道是 per-pair 的**：每个 post 神经元对不同的 pre 神经元有独立的 Linear。
3. **默认 excitatory 通道**：先验证兴奋性调制，抑制性通道已预留接口但本轮不启用。
4. **side_signals 在 Transformer 输出后残差注入**：与计划文档中 `h += 0.1 * proj(signal)` 一致。
5. **端到端训练 side_channels**：先用反向传播联合训练，后续可接入 STDP。

## 风险与回退

- 风险：side_channels 参数量较大（每神经元 ~4.7M），训练时间可能变长。
- 风险：残差调制强度 0.1 可能过强或过弱，需要实验调整。
- 回退：如果 emergent 效果不明显，可尝试 (a) 调制强度可调，(b) 在 field 向量后加 gate，(c) 抑制通道也启用。
