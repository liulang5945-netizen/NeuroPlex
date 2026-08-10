# 态极接口速查与易错点手册（INTERFACE_REFERENCE）

> 目的：解决"经常用错接口"的问题。按**易错点**组织（不是完整 API 参考），每个点给出 错→对 用法。
> 全部条目基于源码实读（2026-08-10），行号随版本可能漂移，以文件为锚。
> 配套架构文档：[CODE_WIKI.md](file:///e:/taiji-neuron/CODE_WIKI.md)、[BIO_INSPIRED_ARCHITECTURE_PLAN.md](file:///e:/taiji-neuron/plans/BIO_INSPIRED_ARCHITECTURE_PLAN.md)

## 一、最易用错的 8 大接口陷阱（速览）

| # | 陷阱 | 一句话正确用法 |
|---|---|---|
| 1 | `side_channels` 只是 `excite_channels` 别名，不含 inhibit | 删/建通道用 `excite_channels`/`inhibit_channels`，别用 `side_channels` |
| 2 | `forward`（推理）与 `forward_train`（训练）返回 key 完全不同 | 推理读 `weighted_logits`，训练读 `fused_logits` |
| 3 | `_parallel_forward` 返回 **6 元组**，docstring 只写 5 个 | 按 6 个解包 `(vecs, logits, conf, scores, ql, judge)` |
| 4 | `judge_lm_head`（general 判定头）vs `lm_head`（域生成头） | 判定信号走 judge_lm_head；扩域词表只 resize `lm_head` |
| 5 | `get_ffn_gain()==get_lr_multiplier()`、`get_attention_temp_gain()==get_field_write_scale()` | 同公式不同语义，改映射会双影响 |
| 6 | `batch_align_and_embed` 返回元数随 `answer_marker` 变化 | 传 `answer_marker` → 4 元组，否则 3 元组 |
| 7 | `SleepConsolidator.consolidate` 参数顺序 `(neurons, coaction, current_step, stdp)` | 位置调用时 `current_step` 与 `stdp_tracker` 极易写反，**用关键字** |
| 8 | `resize_embedding_for_vocab` 文档提到但**不存在** | 用 `resize_linear_for_vocab`（同样适用于 Embedding 权重） |

---

## 二、命名/语义易混

### 2.1 side_channels 别名（neuron.py:271-274）
```python
# ❌ 旧代码遗留：neuron.side_channels[pid] 拿的是 excite 通道（且删不掉 inhibit）
# ✅ 正确：excite_channels（兴奋）/ inhibit_channels（抑制）分开操作
neuron.excite_channels[pid]   # 增删查都走这里
neuron.inhibit_channels[pid]  # 抑制通道必须显式用这个
```
- `side_channels` 是 `@property` → `excite_channels`，**不含 inhibit_channels**。
- 删除通道条目时，还要清理关联 `excite_scale_{pid}` 参数与 `excite_bias_{pid}` buffer，否则 state_dict 残留孤儿（C25-B 的 `apply_structure_updates` 已这样做，手写删除请模仿）。

### 2.2 forward vs forward_train（ensemble.py:1049 vs 1746）
| 维度 | `forward`（推理） | `forward_train`（训练） |
|---|---|---|
| 融合结果 key | `weighted_logits`（跨 vocab 可能缺失！） | `fused_logits` |
| 质量信号 | `quality_logits`（[N]，round1 聚合） | `quality_logits`（[N]）+ `per_neuron_nll` + `contrastive_loss` |
| 判定信号 | `return_judge_logits=True` → `round1_judge_logits`（{nid:[B,L,256K]}） | 无独立 key（nll 内含 judge 空间投影） |
| 默认 fusion_mode | `"per_position"`（**文档与默认值脱节**，实际主路径是 soft/score 共振分融合） | `"soft"` |

**推理要跟训练对齐**：显式传 `fusion_mode="soft"`（cortex.think/generate 已默认 soft；直接调 `ensemble.forward` 要自己传，否则走 entropy 启发式旧路径——PPL 2.2 vs 12.6 的教训，ensemble.py:1628-1633）。

### 2.3 judge_lm_head vs lm_head（双头）
- `lm_head`：域词表生成头（code 12K/math 10K/zh 50K/en 16K）。
- `judge_lm_head`：general 256K 判定头，**初始为 None**，由 loader 从 ckpt `judge_lm_head_state` 注入（维度从权重 shape 推断，非硬编码）。构造的 neuron 默认无判定头，`forward(return_judge_logits=True)` 不返回 `judge_logits`。
- `resize_lm_head_for_vocab(neuron, new_vocab)` 只处理 `lm_head`，judge 头不受影响（正确行为）。

### 2.4 quality_logit 的"三义"
| 名字 | 位置 | 形状 | 语义 |
|---|---|---|---|
| `quality_logit`（单数） | neuron.forward 返回 | [B,1] | **只在 round_num==1 产生** |
| `quality_logits` | forward 推理 result | [N] | round1 聚合质量（C19） |
| `quality_logits` | forward_train result | [N] | 同上（训练侧） |
| `quality_logits`（参数） | `_confidence_routing_fusion` | [N] | trust 校准系数 |

三个 `quality_logits` 语义都是"质量"，但来源/用途不同，勿跨用途混用。

### 2.5 weights vs final_weights（格式不统一）
- `_score_logit_fusion` / `_division_logit_fusion` / `_confidence_routing_fusion` → `result["weights"]`（list）
- `_compute_per_position_weights` / `_residual_logit_fusion` / `_consensus_logit_fusion` → `result["final_weights"]`（dict {nid: float}）
- shared_expert 重加权只检查 `"final_weights"`（ensemble.py:1733）——**soft 融合路径下不生效**。

---

## 三、参数与位置参数（最易踩的坑）

### 3.1 establish_side_channel（neuron.py:276）
```python
# 签名：(peer_id, peer_neuron, channel_type="excite", init_std=0.01, init_scale=50.0)
# ❌ 把 peer 实例当第一参 / 传字符串
# ✅ 正确：第一参是 peer 的 ID 字符串，第二参是 peer 的**实例**（读 field_dim）
neuron.establish_side_channel("math", math_neuron, channel_type="excite")
```

### 3.2 SleepConsolidator.consolidate（neuro_modulation.py:241）
```python
# 签名：(neurons, coactivation_tracker=None, current_step=0, stdp_tracker=None)
# ❌ 位置调用：consolidate(neurons, coaction, self._stdp_tracker, self._current_step)  # 顺序写反
# ✅ 用关键字
consolidate(neurons=..., coactivation_tracker=..., current_step=..., stdp_tracker=...)
```

### 3.3 STDPTracker.apply_updates（stdp.py:343）
```python
# 签名：(post_neuron, pre_neuron_id) —— post 在前！
# ❌ apply_updates(pre_id, post_neuron)
# ✅ apply_updates(post_neuron, pre_id)
```
- **隐蔽坑**：`apply_updates` 用 `post_neuron.config.neuron_id` 查发放历史（stdp.py:361），`config.neuron_id` 为 None 时退化为查 `"self"` → 全部空更新。**训练/装配必须设置 `cfg.neuron_id`**。

### 3.4 field 写入：write vs update vs write_inhibit（field.py:97/141/192）
| 方法 | 语义 | 易错点 |
|---|---|---|
| `write(nid, v, scale=1.0)` | 纯累加（round 1 用） | scale 支持 float 或 [B] tensor |
| `update(nid, v, scale=1.0)` | 替换（round 2+ 用） | 先减旧贡献再加新 |
| `write_inhibit(nid, v, weight=1.0)` | 乘法衰减 | **参数名是 `weight`（抑制强度）不是 `scale`**；用 `v.abs()` 方向 |

### 3.5 batch_align_and_embed（translator.py:792）
```python
# 不传 answer_marker → 3 元组 (shared_emb, domain_targets, attention_mask)
# 传 answer_marker → 4 元组 (..., sft_mask)
# ❌ 固定按 4 元组解包（无 answer_marker 时崩）
# ✅ 按是否传参分支解包
```

### 3.6 装配函数返回元数
- `create_cortex(...)` / `load_cortex(...)` → `(cortex, tokenizer)` 2 元组
- `assemble_cortex(...)` → `(cortex, tokenizer, modules)` 3 元组
- `load_cortex` 与 `create_cortex` **等价**（直接转发），只是命名不同。

---

## 四、返回结构不一致

### 4.1 `_parallel_forward` 返回 6 元组（ensemble.py:1047）
```python
round_vecs, round_logits, round_confidences, round_score_vecs, round_quality_logits, round_judge_logits = \
    self._parallel_forward(...)   # docstring 只写 5 个，按 5 个解包会错位
```

### 4.2 `forward` 的 dict key 是条件性追加的
恒有：`field_state/final_scores/n_rounds/.../round1_logits`
条件：`quality_logits`（quality_head 存在时）、`round1_judge_logits`（`return_judge_logits=True`）、`neuron_logits`+融合结果（`return_logits=True`）
**跨 vocab 融合失败时 `weighted_logits` 缺失**，`fusion_mode="neuron_logits_only"` + `fusion_error`——调用方必须判空。

---

## 五、双语义/同公式映射（neuro_modulation.py）

| 方法 | 公式 | 语义 | 注意 |
|---|---|---|---|
| `get_lr_multiplier` | 0.5 + DA×1.5 | 学习率倍数 | 与 ffn_gain **同公式** |
| `get_ffn_gain` | 0.5 + DA×1.5 | FFN 输出增益 | 改映射双影响 |
| `get_refractory_multiplier` | 0.5 + 5-HT×1.0 | 不应期 | — |
| `get_field_write_scale` | 0.5 + NE×1.0 | 场写入强度 | 与 attention_temp **同公式** |
| `get_attention_temp_gain` | 0.5 + NE×1.0 | attention 温度 | 改映射双影响 |
| `get_attention_focus_gain` | 0.6 + ACh×0.8 | 注意聚焦 | C25-C 新增，与 NE temp 相乘组合 |

- 调质值 `set_targets` 会 clamp [0,1]；**直接改字段不 clamp**。
- 旧 ckpt 无 `acetylcholine` → `load_state_dict` 默认 0.5（中性兼容）。

---

## 六、静态资源与缓存

- `loader.general_vocab_size()`：从 sp_general.model 动态读取，**带进程级缓存**——general 词表重训/扩展后需重启进程才更新。
- `resize_embedding_for_vocab` **不存在**（translator.py:383 docstring 提到，全仓无实现）——用 `resize_linear_for_vocab`（对 Embedding 权重同样适用）。
- `EditableVocabulary` 包装 SentencePiece：`__getattr__` 把未实现方法**透传**给 base SP（translator.py:631）——透传的方法行为可能不同（如 `encode` 不支持 `out_type`），新增能力（`save_ext/load_ext/add_tokens/remove_token`）不受透传影响。`remove_token` 会导致 ext id 重排，**已被 lm_head 引用的 token 勿运行时移除**（建议重训 SP）。
- `general_vocab_size()` 失败回退 256000——所有判定头/共享表维度创建处都应调用它，避免字面量泄漏。

---

## 七、方向性语义（生物语义 vs 工程排序）

- `CoactivationTracker.get_strong_pairs()` 返回 `tuple(sorted([i,j]))`（**无方向**）；`SleepConsolidator.consolidate` 按 `(pre, post)` 解包消费（neuro_modulation.py:291-298）——**只有 sorted 顺序恰好 == (pre,post) 时才语义正确**。nid 命名变化会破坏该隐式约定。
- `CoactivationTracker.update(ids, round_num=1)` 的 `round_num` **未使用**（预留），共激活不分轮次。
- `STDPTracker` 的 `_coactivation_stats` 是**有向** `(pre, post)`（pre 先于 post 发放）；`get_coactivation_stats(A, B)` 与 `(B, A)` 不同。
- `topology.build_topology` 返回语义 = **post 读 pre**（post←pre 有向边）。
- `STDPTracker.get_state_dict` 键序列化为 `"pre|post"`，load 按 `"|"` split——**nid 含 `|` 会解析错**。

---

## 八、兼容层/别名/默认值

| 陷阱 | 说明 |
|---|---|
| `ModelSelfTokenizer = TaijiNativeTokenizerV2` | loader.py:20 向后兼容别名 |
| `_generate_p7` 的 `collab_mode` 默认 `"fusion"` | `generate()` 默认 `"executive"`——直接调 `_generate_p7` 不传会走 token 级融合路径，行为不一致 |
| `forward` 的 `fusion_mode` 默认 `"per_position"` | docstring 与默认值脱节，推理主路径是 soft（见 2.2） |
| `think(active_nids="auto_all")` 字符串模式 | 字符串模式由 `_generate_p7` 展开（cortex.py:1706-1716）；直接调 `think` 字符串会被 ensemble 当普通过滤 → fallback 全量 |
| `cortex.neurons` 与 `ensemble.neurons` 同一引用 | 改一边两边生效（add_neuron 亦然） |
| `TaijiNativeTokenizerV2.encode` 返回 `List[int]` | 与 SP 的 `encode(text, out_type=str)` 不同，混用时注意 out_type 参数 |
| `field.reset` 直接赋值 `self.state`（field.py:81-85） | buffer 对象被替换，外部持旧引用会失效 |

---

## 九、按模块接口速查（核心方法）

| 模块 | 类 | 关键方法 |
|---|---|---|
| resonance/neuron.py | ResonanceNeuron | `forward(e, field_state, round_num, return_logits, return_judge_logits, side_signals, temp_gain, ffn_gain, return_intermediate)`、`establish_side_channel`、`compute_logits`、`enable_lora`、`prune_weak_channels`、`freeze_fingerprint` |
| resonance/ensemble.py | ResonanceEnsemble | `forward`、`forward_train`、`_parallel_forward`、`add_neuron`、`set_tokenizer_hub`、`set_alignment_rules` |
| resonance/translator.py | TokenizerHub / EditableVocabulary | `register_domain/encode/encode_tensor/vocab_size/eos_token_id/load_default_domains/to_editable/add_tokens/unregister_domain`；`add_token(s)/remove_token/encode/decode/GetPieceSize/save_ext/load_ext` |
| resonance/neuro_modulation.py | NeuromodulatorState / SleepConsolidator | `set_targets/step/get_*_multiplier/get_*_gain/should_trigger_neurogenesis`；`record_high_resonance_state/should_consolidate/consolidate` |
| resonance/stdp.py | STDPTracker / STDPRule | `record_firing/accumulate_coactivation/get_coactivation_stats/apply_structure_updates/apply_updates/apply_all_updates/get_state_dict`；`compute_weight_update` |
| resonance/tribal.py | CoactivationTracker | `update/get_coactivation/get_tribe/get_all_tribes/decay/get_strong_pairs/forget_weak` |
| resonance/field.py | ResonanceField | `write/update/write_inhibit/apply_inhibitory_wta/get_effective_state/score/reset` |
| resonance/phasor.py | PhasorDynamics | `register_neurons/assign_phase_by_domain/evolve/kuramoto_step/binding_tensor/task_gradient_step` |
| resonance/topology.py | — | `build_topology/establish_topology_channels/infer_topology_from_state` |
| brain/cortex.py | Cortex | `think/_executive_route/generate/_generate_p7/save_state/load_state/add_neuron/remove_neuron/isolate_neuron/set_*` |
| loader.py | — | `create_cortex/load_cortex/assemble_cortex/general_vocab_size` |
| life/sleep_engine.py | SleepEngine | `sleep/nap/set_brain_interfaces/_update_neuromodulators/_train_cortex_neurons` |

---

*本文档随接口演进维护：新增/变更接口时同步更新对应小节。*
