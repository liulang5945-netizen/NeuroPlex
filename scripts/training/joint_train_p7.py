"""P7 Joint Training v3 — shared embedding + field contrastive.

V1 bug: each neuron produces 1 field_vector per step, field_loss degenerates.
V2 fix: per-sample field contrastive (B×B pairwise), but field vectors from
       different embedding spaces — cosine similarity ~0 always.
V3 fix: ALL neurons share a single nn.Embedding(256000, 512). Domain tokenizer
       is used for lm_head targets only; general tokenizer feeds shared embedding.
       Field vectors now come from the SAME embedding basis → comparable!
       Field contrastive loss is NOW meaningful.

Design:
  1. Raw texts sampled from each domain
  2. batch_align_and_embed: domain tokenizer (targets) + general tokenizer (emb)
  3. All neurons receive same shared_emb → [B, D] field vectors comparable
  4. Intra-domain loss: same-neuron field vectors should be diverse
  5. Inter-domain loss: cross-neuron field vectors should separate
  6. LM loss: per-neuron next-token prediction (domain vocab targets)
  7. total = lm_weight * lm + field_weight * field

Usage:
    python scripts/training/joint_train_p7.py --steps 2000 --batch_size 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stub missing modules (same as v2)
for mod_name, attrs in [
    ("taiji.resonance.gating", {
        "ConfidenceGate": type("X", (), {"should_resonate": lambda *a: True}),
        "EarlyStopResonance": type("X", (), {"should_stop": lambda *a: False, "threshold": 1e-3, "min_rounds": 2}),
        "ResonanceTrigger": type("X", (), {"check": lambda *a: True, "should_trigger": lambda *a: False}),
    }),
    ("taiji.resonance.quality", {"QualityFilter": type("X", (), {"filter": lambda s, x: x})}),
    ("taiji.resonance.division", {
        "ScaleLayering": type("X", (), {}), "ClusterDominance": type("X", (), {}), "DivisionPath": type("X", (), {}),
    }),
    ("taiji.resonance.domain_router", {"DomainRouter": type("X", (), {})}),
    ("taiji.resonance.domain_detector", {}),
]:
    m = types.ModuleType(mod_name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[mod_name] = m

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import get_domain_neuron_config, DOMAIN_VOCAB_SIZES
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_domain_texts,
    load_general_tokenizer, load_or_create_shared_embedding, save_shared_embedding,
    DATA_DIR, OUTPUT_DIR as NEURONS_DIR,
    GENERAL_VOCAB_SIZE, SHARED_EMBED_DIM, SHARED_EMBEDDING_PATH,
)

DOMAINS = ["zh", "en", "code", "math", "general"]


def field_contrastive_loss_v3(
    all_field_vectors: Dict[str, torch.Tensor],
    margin: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Pairwise field contrastive loss across neurons.

    V3: field vectors now come from the same shared embedding basis,
    so cosine similarity is meaningful. This loss is fully functional.

    Args:
        all_field_vectors: {neuron_id: [B, D] normalized field vectors}
        margin: cross-neuron cosine should be < margin

    Returns:
        {"intra": ..., "inter": ..., "total": ...}
    """
    nids = list(all_field_vectors.keys())
    N = len(nids)
    if N < 2:
        return {"intra": torch.tensor(0.0), "inter": torch.tensor(0.0), "total": torch.tensor(0.0)}

    # Normalize all vectors
    normed = {}
    for nid, vecs in all_field_vectors.items():
        v = vecs if vecs.dim() == 2 else vecs.unsqueeze(0)
        normed[nid] = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    # Inter-neuron loss: push cross-domain vectors apart
    inter_loss = torch.tensor(0.0)
    inter_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            v_i = normed[nids[i]]  # [Pi, D]
            v_j = normed[nids[j]]  # [Pj, D]
            sim = v_i @ v_j.T  # [Pi, Pj]
            loss = F.relu(sim - margin) ** 2
            inter_loss = inter_loss + loss.mean()
            inter_count += 1
    inter_loss = inter_loss / max(inter_count, 1)

    # Intra-neuron loss: prevent within-neuron collapse
    intra_loss = torch.tensor(0.0)
    intra_count = 0
    for nid in nids:
        v = normed[nid]  # [B, D]
        B = v.shape[0]
        if B < 2:
            continue
        sim = v @ v.T  # [B, B]
        mask = 1.0 - torch.eye(B, device=v.device)
        intra_loss = intra_loss + (sim * mask).abs().mean()
        intra_count += 1
    intra_loss = intra_loss / max(intra_count, 1)

    return {
        "intra": intra_loss,
        "inter": inter_loss,
        "total": intra_loss + inter_loss,
    }


def load_data_as_texts(
    max_texts: int = 4000,
    val_ratio: float = 0.2,
) -> Dict[str, dict]:
    """Load raw texts + split into train/val per domain.

    Returns:
        {domain: {"train_texts": [...], "val_texts": [...], "domain_sp": sp}}
    """
    print("Loading data for all domains...")
    result = {}
    for domain in DOMAINS:
        domain_sp = load_domain_tokenizer(domain)
        texts = load_domain_texts(domain, max_texts=max_texts)
        n = len(texts)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val

        # Shuffle and split
        import random
        random.shuffle(texts)
        train_texts = texts[:n_train]
        val_texts = texts[n_train:]

        result[domain] = {
            "train_texts": train_texts,
            "val_texts": val_texts,
            "domain_sp": domain_sp,
        }
        print(f"  {domain}: {n_train}/{n_val} train/val, vocab={domain_sp.vocab_size()}")
    return result


def load_or_create_neurons(spec: str = "compact", device: str = "cpu") -> Dict[str, ResonanceNeuron]:
    """Load existing P7 neurons or create new ones."""
    neurons = {}
    for domain in DOMAINS:
        ckpt_path = os.path.join(NEURONS_DIR, f"neuron_{domain}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt["neuron_config"]
            neuron = ResonanceNeuron(cfg)
            neuron.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"  Loaded {domain} from {ckpt_path}")
        else:
            cfg = get_domain_neuron_config(domain, spec)
            neuron = ResonanceNeuron(cfg)
            print(f"  Created new {domain} ({spec})")
        neuron.to(device)
        neuron.train()
        neurons[domain] = neuron
    return neurons


def evaluate_ppl(
    neurons: Dict[str, ResonanceNeuron],
    data: Dict[str, dict],
    shared_embedding: nn.Embedding,
    general_sp,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute per-neuron validation PPL using shared embedding + alignment."""
    ppls = {}
    for domain, neuron in neurons.items():
        if domain not in data or not data[domain].get("val_texts"):
            continue
        domain_sp = data[domain]["domain_sp"]
        val_texts = data[domain]["val_texts"][:20]  # limit for speed

        neuron.eval()
        shared_embedding.eval()
        total_loss = 0.0
        n_batches = 0
        batch_size = 2

        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                batch_texts = val_texts[i:i + batch_size]
                if not batch_texts:
                    break

                shared_emb, targets, mask = batch_align_and_embed(
                    batch_texts, domain_sp, general_sp, shared_embedding,
                )
                shared_emb = shared_emb.to(device)
                targets = targets.to(device)
                mask = mask.to(device)

                out = neuron.forward(shared_emb, return_logits=True)
                logits = out["logits"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()

                shift_targets = shift_targets.clone()
                shift_targets[~shift_mask] = -100

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )
                total_loss += loss.item()
                n_batches += 1
                if n_batches >= 10:
                    break

        avg_loss = total_loss / max(n_batches, 1)
        ppls[domain] = math.exp(min(avg_loss, 20))
        neuron.train()
        shared_embedding.train()
    return ppls


def joint_train_v3(
    neurons: Dict[str, ResonanceNeuron],
    data: Dict[str, dict],
    shared_embedding: nn.Embedding,
    general_sp,
    num_steps: int = 2000,
    batch_size: int = 4,
    lr: float = 2e-4,
    lm_weight: float = 1.0,
    field_weight: float = 0.5,
    device: str = "cpu",
    log_every: int = 100,
    eval_every: int = 500,
    save_dir: str = "data/neurons_joint_v3",
) -> dict:
    """Joint training v3: shared embedding + field contrastive.

    ALL neurons receive same shared_emb (from general tokenizer).
    Field vectors NOW come from same embedding basis → comparable.
    Field contrastive loss is fully functional.

    Each step:
      1. Sample B raw texts from each domain
      2. batch_align_and_embed: texts → shared_emb [B, L_g, 512] + domain targets
      3. All neurons forward same shared_emb → [B, D] field vectors
      4. Field contrastive loss (inter-neuron + intra-neuron)
      5. Per-neuron LM loss (domain vocab targets, aligned positions)
      6. total = lm_weight * lm + field_weight * field
    """
    # Validate data availability
    active_domains = [d for d in DOMAINS if d in neurons and d in data and data[d].get("train_texts")]

    # Single optimizer: neuron params + shared embedding
    all_params = list(shared_embedding.parameters())
    for n in neurons.values():
        all_params.extend(n.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr)

    history = {"lm": [], "field": [], "total": [], "val_ppl": []}
    best_val_ppl = float("inf")

    print(f"\nJoint Training v3 (shared embedding): {len(active_domains)} domains, {num_steps} steps")
    print(f"  batch_size={batch_size}/domain, lr={lr}")
    print(f"  lm_weight={lm_weight}, field_weight={field_weight}")
    print(f"  Shared embedding: {GENERAL_VOCAB_SIZE} × {SHARED_EMBED_DIM}")
    print(f"  Field vectors from SAME embedding basis → cosine meaningful!")
    print()

    t_start = time.time()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # 1. Per-domain forward with shared embedding
        all_field_vectors = {}
        lm_losses = []

        for domain in active_domains:
            neuron = neurons[domain]
            domain_sp = data[domain]["domain_sp"]
            train_texts = data[domain]["train_texts"]

            # Sample B raw texts from this domain
            idx = torch.randint(0, len(train_texts), (batch_size,))
            batch_texts = [train_texts[int(i)] for i in idx]

            # Align: domain tokenizer (targets) + general tokenizer (shared emb)
            shared_emb, targets, mask = batch_align_and_embed(
                batch_texts, domain_sp, general_sp, shared_embedding,
            )
            shared_emb = shared_emb.to(device)   # [B, L_g, 512]
            targets = targets.to(device)          # [B, L_g] domain token IDs
            mask = mask.to(device)                # [B, L_g]

            # Forward through neuron
            result = neuron.forward(shared_emb, return_logits=True)

            # LM loss (domain vocab, aligned positions)
            logits = result["logits"]  # [B, L_g, domain_vocab]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()

            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100

            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            lm_losses.append(lm_loss)

            # Field vectors [B, D] — NOW from same embedding basis!
            all_field_vectors[domain] = result["field_vector"]

        # 2. Field contrastive loss (NOW meaningful!)
        field_losses = field_contrastive_loss_v3(all_field_vectors, margin=0.5)

        # 3. Total loss
        avg_lm = torch.stack(lm_losses).mean()
        field_total = field_losses["total"]
        total = lm_weight * avg_lm + field_weight * field_total

        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        history["lm"].append(avg_lm.item())
        history["field"].append(field_total.item())
        history["total"].append(total.item())

        if step % log_every == 0:
            recent_lm = sum(history["lm"][-log_every:]) / log_every
            recent_field = sum(history["field"][-log_every:]) / log_every
            elapsed = time.time() - t_start
            print(
                f"  step {step:>5d} "
                f"lm={recent_lm:.4f} "
                f"field={recent_field:.4f} "
                f"intra={field_losses['intra'].item():.4f} "
                f"inter={field_losses['inter'].item():.4f} "
                f"elapsed={elapsed:.0f}s"
            )

        # Validation
        if step % eval_every == 0 or step == num_steps:
            val_ppls = evaluate_ppl(neurons, data, shared_embedding, general_sp, device)
            avg_val = sum(val_ppls.values()) / max(len(val_ppls), 1)
            history["val_ppl"].append((step, val_ppls))
            best_str = "*" if avg_val < best_val_ppl else ""
            if avg_val < best_val_ppl:
                best_val_ppl = avg_val
            print(f"  [val@{step}] avg_ppl={avg_val:.1f} {best_str}  {val_ppls}")

    # Save neurons
    os.makedirs(save_dir, exist_ok=True)
    for domain, neuron in neurons.items():
        ckpt_path = os.path.join(save_dir, f"neuron_{domain}.pt")
        torch.save({
            "neuron_config": neuron.config,
            "state_dict": neuron.state_dict(),
            "domain": domain,
            "result": {"steps": num_steps, "best_val_ppl": best_val_ppl},
        }, ckpt_path)

    # Save shared embedding
    joint_emb_path = os.path.join(save_dir, "shared_embedding.pt")
    torch.save(shared_embedding.weight.data, joint_emb_path)

    print(f"\nSaved to {save_dir}/")
    print(f"  Shared embedding: {joint_emb_path}")

    return {"best_val_ppl": best_val_ppl}


def main():
    parser = argparse.ArgumentParser(description="P7 Joint Training v3 (shared embedding)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--field_weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--max_texts", type=int, default=3000)
    parser.add_argument("--spec", default="compact")
    parser.add_argument("--save_dir", default="data/neurons_joint_v3")
    parser.add_argument("--general_tokenizer", default=None,
                        help="Path to general 256K tokenizer model (default: auto-detect)")
    args = parser.parse_args()

    print("=" * 60)
    print("P7 Joint Training v3 (shared embedding)")
    print(f"  Steps: {args.steps}, Batch: {args.batch_size}/domain")
    print(f"  Loss: LM*{args.lm_weight} + Field*{args.field_weight}")
    print(f"  Field vectors from SAME embedding basis → cosine meaningful!")
    print("=" * 60)

    # ── Shared resources ──
    shared_embedding = load_or_create_shared_embedding(args.device)
    general_sp = load_general_tokenizer(args.general_tokenizer)
    print()

    # ── Data ──
    data = load_data_as_texts(args.max_texts)

    # ── Neurons ──
    neurons = load_or_create_neurons(spec=args.spec, device=args.device)

    # ── Initial eval ──
    print("\nInitial validation PPL:")
    val_ppls = evaluate_ppl(neurons, data, shared_embedding, general_sp, args.device)
    print(f"  {val_ppls}")

    # ── Train ──
    stats = joint_train_v3(
        neurons=neurons, data=data,
        shared_embedding=shared_embedding,
        general_sp=general_sp,
        num_steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, lm_weight=args.lm_weight, field_weight=args.field_weight,
        device=args.device, log_every=args.log_every,
        eval_every=args.eval_every, save_dir=args.save_dir,
    )

    # Save shared embedding to default location
    save_shared_embedding(shared_embedding)

    print(f"\n{'='*60}")
    print(f"Joint Training v3 Complete")
    print(f"  Best val PPL: {stats['best_val_ppl']:.1f}")
    print(f"  Saved: {args.save_dir}/")
    print(f"  Shared embedding: {SHARED_EMBEDDING_PATH}")


if __name__ == "__main__":
    main()
