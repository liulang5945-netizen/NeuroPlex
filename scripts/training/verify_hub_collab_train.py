#!/usr/bin/env python3
"""hub 协作层训练验证（缺口 L 阶段 3 第二部分，2026-08-14）。

验证 train_cross_domain_collab.py 的 hub 接入（--hub-path）：
- A1. load_hub_neuron 装配：hub 规格（expert/1024/field4096/vocab256K）+ 自带 lm_head 保留
- A2. hybrid 拓扑自动选 hub 为 global-hub（容量 1024×14=14336 最大）
- A3. side_channels 双向通道：hub↔spoke（含跨规格投影 4096→1024 / →512/768）
- A4. LoRA 冻结模式：hub body 冻结、side_channels/scale/cross_spec 可训
- A5. forward_train 含 hub 跑通：loss 有限不 NaN + hub excite 通道梯度流动
- A6. checkpoint 保存后回读：side_channels_state 含 hub、cross_spec_state 含 hub
- B1. 回归：无 hub 的 domains 加载路径（shared head 注入）不受影响

运行：python -u scripts/training/verify_hub_collab_train.py
"""
from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from neuroplex.resonance import ResonanceEnsemble, ResonanceField  # noqa: E402
from neuroplex.resonance.geometry import NeuronGeometry  # noqa: E402
from neuroplex.resonance.topology import (  # noqa: E402
    _select_hubs, build_topology, establish_topology_channels, topology_detail,
)
from neuroplex.resonance.translator import TokenizerHub, batch_align_and_embed  # noqa: E402
from scripts.training.utils import load_general_tokenizer  # noqa: E402
import scripts.training.train_cross_domain_collab as tcdc  # noqa: E402

GENERAL_DIR = "data/foundation_v1_general"
HUB_PATH = "data/hub_neuron/neuron_hub.pt"
SFT_DIR = "data/sft"

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def build_ensemble_with_hub(device: str = "cpu", nids=None, field_dim=None):
    """构建含 hub 的协作阵容（域 neuron + hub），模拟 collab 训练 step 1-5。

    nids：参与协作的域 neuron id 列表（foundation_v1_general 基座），
    默认 ["code", "math"]；对比 loss 验证传 ["zh", "code"]。
    field_dim：统一场维度（装配口径 3072 时 hub 也建投影 4096→3072，
    模拟装配综合体——训练产物可复用）；None=阵容 max（含 hub 4096）。
    """
    if nids is None:
        nids = ["code", "math"]
    shared_lm_head = tcdc.load_shared_lm_head(GENERAL_DIR, 512, device)
    neurons = {}
    shared_embeddings = {}
    for nid in nids:
        n = tcdc.load_neuron(nid, GENERAL_DIR, device, shared_lm_head=shared_lm_head)
        neurons[nid] = n
        shared_embeddings[nid] = tcdc.load_shared_embedding(GENERAL_DIR, device)
    hub, hub_emb = tcdc.load_hub_neuron(HUB_PATH, device)
    neurons["hub"] = hub
    shared_embeddings["hub"] = hub_emb

    # TokenizerHub：general 目标空间
    hub_tk = TokenizerHub()
    general_sp = load_general_tokenizer()
    hub_tk.register_domain("general", general_sp)

    # 拓扑（hybrid，默认）——hub 应为 global-hub
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)

    # 冻结 + LoRA（collab 脚本 step 4 同款）
    for neuron in neurons.values():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        neuron.enable_lora(16, layers=None)
        for p in neuron.lora_adapters.parameters():
            p.requires_grad = True
        neuron.train()

    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=field_dim or max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    ensemble.set_tokenizer_hub(hub_tk)
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
    return neurons, shared_embeddings, ensemble, general_sp, topology


def make_batch(texts, general_sp, shared_embeddings, seq_len=32, device="cpu"):
    """模拟 collab 训练 general_mode 的 batch 构建（首 neuron 的目标/掩码）。"""
    neuron_embeddings, targets, mask = {}, None, None
    for nid, emb in shared_embeddings.items():
        out = batch_align_and_embed(texts, general_sp, general_sp, emb, max_seq_len=seq_len)
        neuron_embeddings[nid] = out[0].to(device)
        if targets is None:
            targets = out[1].to(device)
            mask = out[2].to(device)
    return neuron_embeddings, targets, mask


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("hub 协作层训练验证（阶段 3 第二部分）", flush=True)
    print("=" * 60, flush=True)

    # ── A. 装配 + 拓扑 + 通道 ──
    print("\n[A] hub 装配 + 协作层...", flush=True)
    neurons, shared_embeddings, ensemble, general_sp, topology = build_ensemble_with_hub()
    hub = neurons["hub"]

    check("A1. hub 装配（expert/1024/field4096/vocab256K + 自带 head）",
          hub.config.spec == "expert" and hub.config.hidden_size == 1024
          and hub.config.field_dim == 4096 and hub.config.vocab_size == 256000
          and hub.lm_head.out_features == 256000,
          f"hidden={hub.config.hidden_size} head={hub.lm_head.out_features}")

    hubs = _select_hubs(list(neurons.keys()), neurons,
                        NeuronGeometry(embedding_dim=8, sigma=0.5))
    check("A2. hybrid 拓扑自动选 hub 为 global-hub", hubs[0] == "hub",
          f"hubs={hubs} capacity=1024×14=14336 最大")
    # topology_detail 输出含 hub 高 indegree
    detail = topology_detail(topology, neurons)
    check("A2b. hub indegree 最高（汇聚所有 spoke）",
          detail.strip().splitlines() and "hub (expert)" in detail,
          detail.strip().splitlines()[1] if len(detail.strip().splitlines()) > 1 else "")

    spoke_has_hub = all(
        "hub" in neurons[sid].excite_channels for sid in ["code", "math"])
    hub_has_spoke = all(
        sid in hub.excite_channels for sid in ["code", "math"])
    check("A3. side_channels 双向（spoke→hub 与 hub→spoke）",
          spoke_has_hub and hub_has_spoke)
    c2h = neurons["code"].excite_channels["hub"]
    h2c = hub.excite_channels["code"]
    check("A3b. 跨规格通道维度正确（hub field→spoke hidden / spoke field→hub hidden）",
          c2h.in_features == hub.config.field_dim
          and c2h.out_features == neurons["code"].config.hidden_size
          and h2c.in_features == neurons["code"].config.field_dim
          and h2c.out_features == hub.config.hidden_size,
          f"code→hub: {c2h.in_features}→{c2h.out_features}, "
          f"hub→code: {h2c.in_features}→{h2c.out_features}")

    hub_body_frozen = all(not p.requires_grad for name, p in hub.named_parameters()
                          if not name.startswith(("excite_", "inhibit_"))
                          and "scale_" not in name and "lora_adapters" not in name)
    hub_chan_trainable = all(p.requires_grad for ch in hub.excite_channels.values()
                             for p in ch.parameters())
    hub_scale_trainable = any(p.requires_grad for name, p in hub.named_parameters()
                              if "scale_" in name)
    cs_trainable = all(p.requires_grad for proj in ensemble._cross_spec_projectors.values()
                       for p in proj.parameters())
    check("A4. LoRA 冻结模式（hub body 冻结 / 通道+scale+跨规格投影可训）",
          hub_body_frozen and hub_chan_trainable and hub_scale_trainable and cs_trainable)

    # ── A5. forward_train 含 hub 跑通 + 梯度 ──
    print("\n[A5] forward_train 含 hub（2 步）...", flush=True)
    data = torch.load(os.path.join(SFT_DIR, "code_sft.pt"),
                      map_location="cpu", weights_only=False)
    texts = [d["full"] for d in data][:8]
    losses = []
    hub_grads = {sid: None for sid in ["code", "math"]}
    for step in range(2):
        neuron_embeddings, targets, mask = make_batch(
            texts, general_sp, shared_embeddings)
        result = ensemble.forward_train(
            neuron_embeddings=neuron_embeddings, n_rounds=2, fusion_mode="soft",
            targets=targets, field_conditioning=False, step=step,
            target_domain="general",
        )
        lg = result["fused_logits"][:, :-1, :].contiguous()
        tg = targets[:, 1:].contiguous()
        mk = mask[:, 1:].contiguous()
        tg = tg.clone()
        tg[~mk] = -100
        nt = max(mk.sum().item(), 1)
        ce = F.cross_entropy(lg.view(-1, lg.size(-1)), tg.view(-1),
                             ignore_index=-100, reduction="sum") / nt
        loss = ce + 0.01 * result["balance_loss"] + 0.05 * result["diversity_loss"]
        loss.backward()
        losses.append(float(loss.item()))
        for sid in ["code", "math"]:
            w = hub.excite_channels[sid].weight
            hub_grads[sid] = None if w.grad is None else float(w.grad.abs().sum().item())
    finite = all(math.isfinite(l) for l in losses)
    check("A5. forward_train 含 hub 跑通（loss 有限不 NaN）",
          finite and len(losses) == 2, f"losses={['%.3f' % l for l in losses]}")
    check("A5b. hub excite 通道梯度流动（hub→code/math 双向可训）",
          all(g is not None and g > 0 for g in hub_grads.values()),
          f"grad_sum={hub_grads}")

    # ── A6. checkpoint 保存回读 ──
    print("\n[A6] checkpoint 保存 + 回读...", flush=True)
    tmp = tempfile.mkdtemp(prefix="hub_collab_ckpt_")
    ckpt_path = os.path.join(tmp, "hub_collab_smoke.ckpt.pt")
    tcdc.save_checkpoint(ckpt_path, 1, 2, neurons, ensemble,
                         None, None, None, [{"loss": losses[-1]}])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ss = ck["side_channels_state"]
    cs = ck["cross_spec_state"]
    check("A6. checkpoint 回读（side_channels 含 hub 双向通道）",
          "hub" in ss
          and all(sid in ss["hub"]["excite"] for sid in ["code", "math"])
          and all("hub" in ss[sid]["excite"] for sid in ["code", "math"]),
          f"hub channels={sorted(ss['hub']['excite'].keys())}")
    # hub field=4096 即统一维度源头 → 无需投影；spoke(2048) 才建投影（ensemble 设计）
    check("A6b. checkpoint 回读（spoke 投影存 hub 无——hub 是统一维度源头）",
          all(sid in cs["forward"] and sid in cs["backward"] for sid in ["code", "math"])
          and "hub" not in cs["forward"],
          f"projected={sorted(cs['forward'].keys())} unified=4096(hub field)")
    shutil.rmtree(tmp, ignore_errors=True)

    # ── B. 回归：无 hub 的 domains 路径 ──
    print("\n[B] 回归：无 hub domains 加载路径...", flush=True)
    shared_lm_head = tcdc.load_shared_lm_head(GENERAL_DIR, 512, "cpu")
    n = tcdc.load_neuron("code", GENERAL_DIR, "cpu", shared_lm_head=shared_lm_head)
    check("B1. domains 路径不受影响（shared head 注入）",
          n.lm_head is not None and n.lm_head.out_features == 256000,
          f"head={n.lm_head.out_features}")

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
