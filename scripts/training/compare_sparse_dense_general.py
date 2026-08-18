"""General + hub 阵容的稠密/稀疏 Router A/B。

只接受包含 ``sparse_router_state`` 的训练 checkpoint。评估使用各域固定
holdout 的 general 256K 输出空间，比较同一 neuron/collab 权重下：

* holdout PPL / loss；
* 平均激活神经元数；
* Router 归一化熵；
* 有效 token 吞吐。

用法：
    python -X utf8 -u scripts/training/compare_sparse_dense_general.py \
        --ckpt data/neurons/cross_domain_collab_sparse12.ckpt.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.resonance import ResonanceEnsemble, ResonanceField
from neuroplex.resonance.ensemble import SparseRouter
from neuroplex.resonance.geometry import NeuronGeometry
from neuroplex.resonance.topology import build_topology, establish_topology_channels
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.eval_dialogue import load_cross_spec_weights
from scripts.training.train_cross_domain_collab import (
    load_hub_neuron,
    load_neuron,
    load_shared_embedding,
    load_shared_lm_head,
)
from scripts.training.utils import load_general_tokenizer

DEVICE = "cpu"
DOMAINS = ("code", "math", "zh")
GENERAL_DIR = "data/foundation_v1_general"
HUB_PATH = "data/hub_neuron/neuron_hub.pt"
SFT_DIR = "data/sft"


def _result_logits(result: Dict[str, object]) -> torch.Tensor:
    weighted = result.get("weighted_logits")
    if isinstance(weighted, torch.Tensor):
        return weighted
    neuron_logits = result.get("neuron_logits") or result.get("round1_logits")
    if not isinstance(neuron_logits, dict) or not neuron_logits:
        raise RuntimeError("ensemble 没有可评估的 logits")
    return next(iter(neuron_logits.values()))


def _load_holdout_texts(start: int, count: int) -> Dict[str, List[str]]:
    texts: Dict[str, List[str]] = {}
    for domain in DOMAINS:
        path = os.path.join(SFT_DIR, f"{domain}_sft.pt")
        data = torch.load(path, map_location="cpu", weights_only=False)
        selected = [row["full"] for row in data[start:start + count]]
        if len(selected) != count:
            raise ValueError(
                f"{domain} holdout 不足：需要 [{start}:{start + count}]，实际 {len(data)} 条"
            )
        texts[domain] = selected
    return texts


def _build_ensemble(
    ckpt_path: str,
    use_router: bool,
    top_k: int,
    warmup_steps: int,
    load_router: bool = True,
    random_seed: int = 0,
):
    shared_lm_head = load_shared_lm_head(GENERAL_DIR, 512, DEVICE)
    neurons = {}
    shared_embeddings = {}
    for domain in DOMAINS:
        neurons[domain] = load_neuron(
            domain, GENERAL_DIR, DEVICE, shared_lm_head=shared_lm_head
        )
        shared_embeddings[domain] = load_shared_embedding(GENERAL_DIR, DEVICE)
    hub, hub_embedding = load_hub_neuron(HUB_PATH, DEVICE)
    neurons["hub"] = hub
    shared_embeddings["hub"] = hub_embedding

    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    ensemble = ResonanceEnsemble(
        neurons,
        ResonanceField(dim=3072),
        max_rounds=2,
        geometry=geometry,
        use_sparse_router=use_router,
        sparse_router_top_k=top_k,
        sparse_router_warmup_steps=warmup_steps,
    )
    load_cross_spec_weights(ensemble, "dialogue", ckpt_path)
    if use_router and not load_router:
        old_router = ensemble.sparse_router
        assert old_router is not None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            ensemble.sparse_router = SparseRouter(
                field_dim=old_router.field_dim,
                score_dim=old_router.score_dim,
                hidden_dim=old_router.router_mlp[0].out_features,
                top_k=old_router.top_k,
                warmup_steps=old_router.warmup_steps,
                shared_expert_id=old_router.shared_expert_id,
                dynamic_k=old_router.dynamic_k,
                k_min=old_router.k_min,
                k_max=old_router.k_max,
            )
    return neurons, shared_embeddings, ensemble


def _iter_batches(
    texts: Iterable[str],
    general_sp,
    shared_embeddings: Dict[str, torch.nn.Embedding],
    seq_len: int,
):
    for text in texts:
        neuron_embeddings = {}
        targets = None
        mask = None
        for nid, embedding in shared_embeddings.items():
            out = batch_align_and_embed(
                [text], general_sp, general_sp, embedding, max_seq_len=seq_len
            )
            neuron_embeddings[nid] = out[0].to(DEVICE)
            if targets is None:
                targets = out[1].to(DEVICE)
                mask = out[2].to(DEVICE)
        assert targets is not None and mask is not None
        yield neuron_embeddings, targets, mask


def _run_mode(
    ensemble,
    shared_embeddings,
    holdout: Dict[str, List[str]],
    general_sp,
    seq_len: int,
    sparse: bool,
):
    total_loss = 0.0
    total_tokens = 0
    total_active = 0.0
    total_entropy = 0.0
    total_samples = 0
    elapsed = 0.0
    selected_counts = {nid: 0 for nid in shared_embeddings}
    with torch.no_grad():
        for domain in DOMAINS:
            for neuron_embeddings, targets, mask in _iter_batches(
                holdout[domain], general_sp, shared_embeddings, seq_len
            ):
                started = time.perf_counter()
                result = ensemble.forward(
                    neuron_embeddings=neuron_embeddings,
                    return_logits=True,
                    fusion_mode="soft",
                    field_conditioning=True,
                )
                elapsed += time.perf_counter() - started
                logits = _result_logits(result)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous().clone()
                shift_mask = mask[:, 1:].contiguous()
                shift_targets[~shift_mask] = -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_tokens = int(shift_mask.sum().item())
                total_loss += float(loss.item())
                total_tokens += max(n_tokens, 1)
                total_samples += 1

                router_result = ensemble._last_router_result
                if sparse and router_result is not None:
                    k = router_result["k_per_sample"].float()
                    total_active += float(k.mean().item())
                    weights = router_result["soft_weights"]
                    entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
                    entropy = entropy / math.log(max(weights.shape[-1], 2))
                    total_entropy += float(entropy.mean().item())
                    active_ids = list(ensemble._router_active_ids or shared_embeddings)
                    for row in router_result.get("top_k_ids", []):
                        for index in row:
                            if 0 <= index < len(active_ids):
                                selected_counts[active_ids[index]] += 1
                else:
                    total_active += float(len(shared_embeddings))

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20.0)),
        "avg_active": total_active / max(total_samples, 1),
        "router_entropy": total_entropy / max(total_samples, 1) if sparse else None,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
        "selected_counts": selected_counts,
    }


def _route_distance(left: Dict[str, int], right: Dict[str, int]) -> float:
    left_total = max(sum(left.values()), 1)
    right_total = max(sum(right.values()), 1)
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="general+hub 稠密/稀疏 Router A/B")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval-start", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    metadata = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    router_state = metadata.get("sparse_router_state")
    if router_state is None:
        raise RuntimeError(
            "checkpoint 不含 sparse_router_state；拒绝用随机 Router 做真实性 A/B"
        )
    router_cfg = metadata.get("sparse_router_config") or {}
    top_k = args.top_k if args.top_k is not None else int(router_cfg.get("top_k", 2))
    warmup = (
        args.warmup_steps
        if args.warmup_steps is not None
        else int(router_cfg.get("warmup_steps", 0))
    )
    print(f"Router checkpoint: top_k={top_k}, warmup={warmup}", flush=True)

    holdout = _load_holdout_texts(args.eval_start, args.eval_count)
    general_sp = load_general_tokenizer()

    print("\n[稠密]", flush=True)
    dense_neurons, dense_embeddings, dense_ensemble = _build_ensemble(
        args.ckpt, False, top_k, warmup
    )
    dense = _run_mode(
        dense_ensemble, dense_embeddings, holdout, general_sp, args.seq_len, False
    )
    print(dense, flush=True)

    print("\n[稀疏]", flush=True)
    sparse_neurons, sparse_embeddings, sparse_ensemble = _build_ensemble(
        args.ckpt, True, top_k, warmup
    )
    sparse = _run_mode(
        sparse_ensemble, sparse_embeddings, holdout, general_sp, args.seq_len, True
    )
    print(sparse, flush=True)

    print("\n[随机 Router 对照]", flush=True)
    _, random_embeddings, random_ensemble = _build_ensemble(
        args.ckpt, True, top_k, warmup, load_router=False, random_seed=0
    )
    random_sparse = _run_mode(
        random_ensemble, random_embeddings, holdout, general_sp, args.seq_len, True
    )
    print(random_sparse, flush=True)

    ppl_delta_pct = (sparse["ppl"] - dense["ppl"]) / max(dense["ppl"], 1e-9) * 100.0
    active_reduction_pct = (
        1.0 - sparse["avg_active"] / max(dense["avg_active"], 1e-9)
    ) * 100.0
    quality_ok = sparse["ppl"] <= dense["ppl"] * 1.05
    sparsity_ok = sparse["avg_active"] < dense["avg_active"] - 0.01
    route_distance = _route_distance(sparse["selected_counts"], random_sparse["selected_counts"])
    behavior_ok = route_distance >= 0.05
    passed = quality_ok and sparsity_ok and behavior_ok
    print("\n[汇总]", flush=True)
    print(f"  PPL: dense={dense['ppl']:.4f} sparse={sparse['ppl']:.4f} Δ={ppl_delta_pct:+.2f}%", flush=True)
    print(
        f"  激活: dense={dense['avg_active']:.2f} sparse={sparse['avg_active']:.2f} "
        f"减少={active_reduction_pct:+.2f}%",
        flush=True,
    )
    print(
        f"  吞吐: dense={dense['tokens_per_sec']:.2f} sparse={sparse['tokens_per_sec']:.2f} tok/s",
        flush=True,
    )
    print(f"  稀疏 Router 归一化熵: {sparse['router_entropy']:.4f}", flush=True)
    print(f"  随机对照 PPL: {random_sparse['ppl']:.4f}", flush=True)
    print(f"  Router 选择分布距离: {route_distance:.4f}", flush=True)
    print(
        f"  质量门槛={'PASS' if quality_ok else 'FAIL'} "
        f"稀疏门槛={'PASS' if sparsity_ok else 'FAIL'} "
        f"非随机门槛={'PASS' if behavior_ok else 'FAIL'}",
        flush=True,
    )
    print(f"  结论={'PASS' if passed else 'FAIL'}", flush=True)
    if args.strict and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
