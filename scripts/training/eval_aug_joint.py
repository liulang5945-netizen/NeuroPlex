"""评估 zh_aug0~3 四神经元联合效果——保持低维 field_dim=2048，通过 side_channels 协作。

核心设计：
  - 每个神经元用自己的 shared_embedding 副本
  - 每个神经元通过 per-pair side_channels 接收其他神经元的 field_vector
  - 不启用 unified_field_dim，保持 compact 低维（~40M 参数）
  - Ensemble.max_rounds=2，让 side_signals 在第二轮生效

Usage:
    python -u scripts/training/eval_aug_joint.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config,
)
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts, create_shared_embedding,
)
from scripts.training.experiment_config import (
    ZH_COMPACT_NEURON_IDS as NEURON_IDS,
    SHARED_EXPERT_ID,
    ZH_STD_NEURON_ID as STD_NEURON_ID,
    DEFAULT_DOMAIN as DOMAIN,
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_REPETITION_PENALTY, SAMPLING_MAX_TOKENS,
    BASE_PROMPTS,
)

DEVICE = "cpu"


def load_aug_neurons(include_shared_expert=False, include_std=False):
    """加载 aug 神经元，每个用自己的 embedding，并建立 side_channels。

    Args:
        include_shared_expert: 是否加载 general 神经元（Shared Expert 机制）
        include_std: 是否加载 standard 神经元 zh_std0（混合规格协作）
    """
    neurons = {}
    shared_embeddings = {}

    all_ids = list(NEURON_IDS)
    if include_shared_expert:
        all_ids.append(SHARED_EXPERT_ID)
    if include_std:
        all_ids.append(STD_NEURON_ID)

    for nid in all_ids:
        path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
        if not os.path.exists(path):
            print(f"  [{nid}] WARN 未找到 checkpoint: {path}，跳过", flush=True)
            continue
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        # 优先使用 checkpoint 中的 neuron_config，支持混合规格
        if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
            cfg = ckpt["neuron_config"]
        else:
            cfg = get_domain_neuron_config(DOMAIN, spec="compact")
        # 保持低维，不启用 unified_field_dim
        cfg.unified_field_dim = None

        # 加载神经元权重
        neuron = ResonanceNeuron(cfg).to(DEVICE)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        neuron.eval()
        neurons[nid] = neuron

        # 加载该神经元自己的 shared_embedding
        shared_emb = create_shared_embedding(DEVICE)
        if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
            shared_emb.load_state_dict(ckpt["shared_embedding_state"])
            print(f"  [{nid}] OK 加载自己的 shared_embedding (spec={cfg.spec})", flush=True)
        else:
            print(f"  [{nid}] WARN 无 shared_embedding_state，用随机初始化", flush=True)
        shared_emb.to(DEVICE).eval()
        shared_embeddings[nid] = shared_emb

        result = ckpt.get("result", {})
        print(f"  [{nid}] spec={cfg.spec}, best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)

    # 建立 per-pair side_channels：每个 post 神经元对 pre 神经元建立 excite 通道
    # side_channels 天然支持跨规格：src_dim=peer.field_dim, dst_dim=self.hidden_size
    loaded_ids = list(neurons.keys())
    for post_id in loaded_ids:
        for pre_id in loaded_ids:
            if pre_id == post_id:
                continue
            neurons[post_id].establish_side_channel(pre_id, neurons[pre_id], channel_type="excite")
        print(f"  [{post_id}] {len(neurons[post_id].excite_channels)} excite side_channels", flush=True)

    # 加载微调后的 side_channels（如果存在）
    # 注意：Shared Expert 的 side_channels 不在微调权重中，会用随机初始化
    # 跨规格微调权重优先（含 side_channels + cross_spec 投影层）
    cross_spec_path = os.path.join(OUTPUT_DIR, "cross_spec_finetuned.pt")
    finetuned_path = os.path.join(OUTPUT_DIR, "side_channels_finetuned.pt")

    loaded_side_state = None
    if os.path.exists(cross_spec_path):
        ckpt_data = torch.load(cross_spec_path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt_data, dict) and "side_channels" in ckpt_data:
            loaded_side_state = ckpt_data["side_channels"]
            print(f"  [side_channels] 已加载跨规格微调权重: {cross_spec_path}", flush=True)
        else:
            loaded_side_state = ckpt_data
            print(f"  [side_channels] 已加载微调权重: {cross_spec_path}", flush=True)
    elif os.path.exists(finetuned_path):
        loaded_side_state = torch.load(finetuned_path, map_location=DEVICE, weights_only=False)
        print(f"  [side_channels] 已加载微调权重: {finetuned_path}", flush=True)
    else:
        print(f"  [side_channels] 未找到微调权重，使用随机初始化", flush=True)

    if loaded_side_state is not None:
        for nid, neuron in neurons.items():
            if nid in loaded_side_state:
                for pid, ch_state in loaded_side_state[nid].get("excite", {}).items():
                    if pid in neuron.excite_channels:
                        neuron.excite_channels[pid].load_state_dict(ch_state)

    return neurons, shared_embeddings, cfg


def _load_cross_spec_weights(ensemble):
    """加载跨规格投影层微调权重（如果存在）。"""
    cross_spec_path = os.path.join(OUTPUT_DIR, "cross_spec_finetuned.pt")
    if not os.path.exists(cross_spec_path):
        return
    ckpt_data = torch.load(cross_spec_path, map_location=DEVICE, weights_only=False)
    if not isinstance(ckpt_data, dict) or "cross_spec" not in ckpt_data:
        return
    cross_spec_state = ckpt_data["cross_spec"]
    for nid, sd in cross_spec_state.get("forward", {}).items():
        if nid in ensemble._cross_spec_projectors:
            ensemble._cross_spec_projectors[nid].load_state_dict(sd)
    for nid, sd in cross_spec_state.get("backward", {}).items():
        if nid in ensemble._cross_spec_back_projectors:
            ensemble._cross_spec_back_projectors[nid].load_state_dict(sd)
    print(f"  [cross_spec] 已加载跨规格投影层权重: {cross_spec_path}", flush=True)


def eval_ppl(neurons, shared_embeddings, domain_sp, general_sp, n_eval=100,
             shared_expert_id=None, shared_expert_weight=0.3):
    """个体 vs 协作 PPL（side_channels + max_rounds=2）。

    Args:
        shared_expert_id: 如果提供，启用 Shared Expert 机制
        shared_expert_weight: Shared Expert 的固定融合权重
    """
    mode_label = f" + Shared Expert ({shared_expert_id}, w={shared_expert_weight})" if shared_expert_id else ""
    print("\n" + "=" * 70, flush=True)
    print(f"[PPL 评估] 个体 vs 协作 (side_channels, max_rounds=2){mode_label}", flush=True)
    print("=" * 70, flush=True)

    texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=n_eval * 3)
    texts = texts[-n_eval:] if len(texts) > n_eval else texts
    print(f"  评估集(simple_zh): {len(texts)} 条文本", flush=True)

    # 创建 Ensemble（用最大 field_dim，跨规格投影由 ensemble 内部处理）
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(
        neurons, field, max_rounds=2,
        shared_expert_id=shared_expert_id,
        shared_expert_weight=shared_expert_weight,
    )

    # 加载跨规格投影层微调权重（如果存在）
    _load_cross_spec_weights(ensemble)

    # 个体 PPL
    individual_ppls = {}
    for nid, neuron in neurons.items():
        shared_emb = shared_embeddings[nid]
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                shared_emb_out, targets, mask = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                shared_emb_out = shared_emb_out.to(DEVICE)
                targets = targets.to(DEVICE)
                mask = mask.to(DEVICE)
                result = neuron.forward(shared_emb_out, return_logits=True)
                logits = result["logits"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                shift_targets = shift_targets.clone()
                shift_targets[~shift_mask] = -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += shift_mask.sum().item()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        individual_ppls[nid] = ppl
        print(f"  个体 [{nid}]: PPL={ppl:.1f} (loss={avg_loss:.4f})", flush=True)

    # 协作 PPL（side_channels，max_rounds=2）
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            # 每个神经元独立编码
            neuron_embeddings = {}
            targets = None
            mask = None
            for nid, shared_emb in shared_embeddings.items():
                shared_emb_out, tgt, msk = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                neuron_embeddings[nid] = shared_emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)

            # Ensemble forward（max_rounds=2，side_channels 生效）
            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
            )

            if "weighted_logits" in result:
                fused_logits = result["weighted_logits"]
            elif "neuron_logits" in result:
                n_logits = list(result["neuron_logits"].values())
                if len(set(l.shape[-1] for l in n_logits)) == 1:
                    fused_logits = torch.stack(n_logits).mean(dim=0)
                else:
                    best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                    fused_logits = result["neuron_logits"][best_nid]
            else:
                best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                fused_logits = neurons[best_nid].forward(
                    neuron_embeddings[best_nid], return_logits=True
                )["logits"]

            shift_logits = fused_logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets_clone = shift_targets.clone()
            shift_targets_clone[~shift_mask] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets_clone.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    collab_ppl = math.exp(min(avg_loss, 20))
    print(f"  协作 [side]:   PPL={collab_ppl:.1f} (loss={avg_loss:.4f})", flush=True)

    if "final_scores" in result:
        scores_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(result["final_scores"].items(), key=lambda x: -x[1]))
        print(f"  共振分: {scores_str}", flush=True)
    if "final_weights" in result:
        weights_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(result["final_weights"].items(), key=lambda x: -x[1]))
        print(f"  融合权重: {weights_str}", flush=True)

    min_ind = min(individual_ppls.values())
    best_id = min(individual_ppls, key=individual_ppls.get)
    print(f"\n  最强个体: [{best_id}] PPL={min_ind:.1f}", flush=True)
    print(f"  协作 PPL: {collab_ppl:.1f}", flush=True)
    if collab_ppl < min_ind:
        imp = (min_ind - collab_ppl) / min_ind * 100
        print(f"  EMERGE 协作比最强个体好 {imp:.1f}%", flush=True)
    else:
        print(f"  NO_EMERGE 协作 PPL={collab_ppl:.1f} >= 最强个体 {min_ind:.1f}", flush=True)

    return individual_ppls, collab_ppl


def eval_generation(neurons, shared_embeddings, domain_sp, general_sp, cfg,
                    shared_expert_id=None, shared_expert_weight=0.3):
    """生成质量对比：个体 vs 协作（side_channels）。"""
    mode_label = f" + Shared Expert" if shared_expert_id else ""
    print("\n" + "=" * 70, flush=True)
    print(f"[生成质量对比] 个体 vs 协作{mode_label}", flush=True)
    print("=" * 70, flush=True)

    # 用最大 field_dim，跨规格投影由 ensemble 内部处理
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(
        neurons, field, max_rounds=2,
        shared_expert_id=shared_expert_id,
        shared_expert_weight=shared_expert_weight,
    )

    # 加载跨规格投影层微调权重（如果存在）
    _load_cross_spec_weights(ensemble)

    PROMPTS = BASE_PROMPTS

    def generate_individual(prompt, nid, max_tokens=SAMPLING_MAX_TOKENS, temperature=SAMPLING_TEMPERATURE, top_k=SAMPLING_TOP_K, repetition_penalty=SAMPLING_REPETITION_PENALTY):
        neuron = neurons[nid]
        shared_emb = shared_embeddings[nid]
        general_ids = general_sp.EncodeAsIds(prompt)
        if not general_ids:
            return "(empty)"
        ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            generated_domain = []
            for _ in range(max_tokens):
                emb_input = shared_emb(ids)
                result = neuron.forward(emb_input, return_logits=True)
                logits = result["logits"][:, -1, :].float()

                # Repetition penalty: 降低已生成 token 的概率
                if generated_domain:
                    for prev_token in set(generated_domain[-20:]):  # 只惩罚最近 20 个 token
                        logits[0, prev_token] /= repetition_penalty

                # Temperature scaling
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0 and top_k < logits.size(-1):
                    topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                # Softmax + sampling
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if hasattr(domain_sp, 'eos_id'):
                    eos = domain_sp.eos_id() if callable(domain_sp.eos_id) else domain_sp.eos_id
                    if next_token == eos:
                        break
                generated_domain.append(next_token)
                piece = domain_sp.DecodeIds([next_token])
                gen_ids = general_sp.EncodeAsIds(piece)
                if gen_ids:
                    ids = torch.cat([ids, torch.tensor([gen_ids], dtype=torch.long, device=DEVICE)], dim=1)
                else:
                    break
                if ids.shape[1] > 200:
                    break
            text = domain_sp.DecodeIds(generated_domain)
        return text

    def generate_collab(prompt, max_tokens=80, temperature=0.8, top_k=40, repetition_penalty=1.2):
        general_ids = general_sp.EncodeAsIds(prompt)
        if not general_ids:
            return "(empty)"

        with torch.no_grad():
            generated_domain = []
            current_ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)

            for _ in range(max_tokens):
                # 每个神经元独立编码当前输入
                neuron_embeddings = {}
                for nid, shared_emb in shared_embeddings.items():
                    emb_input = shared_emb(current_ids)
                    neuron_embeddings[nid] = emb_input

                # Ensemble forward（max_rounds=2，side_channels 生效）
                result = ensemble.forward(
                    neuron_embeddings=neuron_embeddings,
                    return_logits=True,
                    fusion_mode="soft",
                )

                # 获取融合后的 logits
                if "weighted_logits" in result:
                    logits = result["weighted_logits"][:, -1, :].float()
                elif "neuron_logits" in result:
                    n_logits = list(result["neuron_logits"].values())
                    if len(set(l.shape[-1] for l in n_logits)) == 1:
                        logits = torch.stack(n_logits).mean(dim=0)[:, -1, :].float()
                    else:
                        best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                        logits = result["neuron_logits"][best_nid][:, -1, :].float()
                else:
                    best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                    logits = neurons[best_nid].forward(
                        neuron_embeddings[best_nid], return_logits=True
                    )["logits"][:, -1, :].float()

                # Repetition penalty: 降低已生成 token 的概率
                if generated_domain:
                    for prev_token in set(generated_domain[-20:]):
                        logits[0, prev_token] /= repetition_penalty

                # Temperature scaling
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0 and top_k < logits.size(-1):
                    topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                # Softmax + sampling
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if hasattr(domain_sp, 'eos_id'):
                    eos = domain_sp.eos_id() if callable(domain_sp.eos_id) else domain_sp.eos_id
                    if next_token == eos:
                        break

                generated_domain.append(next_token)

                # 解码并追加
                piece = domain_sp.DecodeIds([next_token])
                gen_ids = general_sp.EncodeAsIds(piece)
                if gen_ids:
                    current_ids = torch.cat([current_ids, torch.tensor([gen_ids], dtype=torch.long, device=DEVICE)], dim=1)
                else:
                    break

                if current_ids.shape[1] > 200:
                    break

            text = domain_sp.DecodeIds(generated_domain)
        return text

    for prompt in PROMPTS:
        print(f"\n  prompt: {prompt}", flush=True)
        collab_out = generate_collab(prompt)
        print(f"  协作[side]: {collab_out[:150] if collab_out else '(empty)'}", flush=True)
        for nid in sorted(neurons.keys()):
            indiv_out = generate_individual(prompt, nid)
            print(f"  [{nid}]: {indiv_out[:120] if indiv_out else '(empty)'}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估 zh_aug 神经元联合效果")
    parser.add_argument("--shared_expert", action="store_true",
                        help="启用 Shared Expert 机制（加载 general 神经元）")
    parser.add_argument("--shared_expert_weight", type=float, default=0.3,
                        help="Shared Expert 的固定融合权重（默认 0.3）")
    parser.add_argument("--include_std", action="store_true",
                        help="加载 standard 神经元 zh_std0（混合规格协作）")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="PPL 评估文本数（默认 100）")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    mode_label = " + Shared Expert" if args.shared_expert else ""
    std_label = " + Standard (zh_std0)" if args.include_std else ""
    print("=" * 60, flush=True)
    print(f"zh_aug0~3 四神经元联合评估（side_channels，低维）{mode_label}{std_label}", flush=True)
    print("=" * 60, flush=True)

    print("\n[1] 加载神经元...", flush=True)
    neurons, shared_embeddings, cfg = load_aug_neurons(
        include_shared_expert=args.shared_expert,
        include_std=args.include_std,
    )
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()

    se_id = SHARED_EXPERT_ID if args.shared_expert else None

    print("\n[2] PPL 评估...", flush=True)
    eval_ppl(neurons, shared_embeddings, domain_sp, general_sp,
             n_eval=args.n_eval,
             shared_expert_id=se_id,
             shared_expert_weight=args.shared_expert_weight)

    print("\n[3] 生成质量...", flush=True)
    eval_generation(neurons, shared_embeddings, domain_sp, general_sp, cfg,
                    shared_expert_id=se_id,
                    shared_expert_weight=args.shared_expert_weight)

    print("\n" + "=" * 60, flush=True)
    print("评估完成", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
