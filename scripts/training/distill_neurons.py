"""Phase 2.7: Distill 5 domain neurons from the 1.5B teacher model.

Usage:
    python scripts/training/distill_neurons.py \
        --checkpoint e:/taiji/checkpoint-400000 \
        --data_dir data/distill \
        --output_dir data/neurons \
        --steps 2000 \
        --device cpu
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Embedding projection cache (teacher 2048 → neuron 512) ──
_embed_proj: torch.nn.Linear | None = None


def _project_embedding(teacher_emb: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project teacher embedding [B, L, 2048] → [B, L, target_dim]."""
    global _embed_proj
    src_dim = teacher_emb.shape[-1]
    if src_dim == target_dim:
        return teacher_emb

    device = teacher_emb.device
    if _embed_proj is None or _embed_proj.in_features != src_dim or _embed_proj.out_features != target_dim:
        _embed_proj = torch.nn.Linear(src_dim, target_dim, bias=False).to(device)
        # Initialize with identity-like structure (SVD approximation)
        with torch.no_grad():
            torch.nn.init.orthogonal_(_embed_proj.weight)

    return _embed_proj(teacher_emb)


_distill_proj: torch.nn.Linear | None = None


def _project_teacher_hidden(teacher_hidden: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Project teacher hidden [B, 2048] → [B, target_dim] for distillation."""
    global _distill_proj
    src_dim = teacher_hidden.shape[-1]
    if src_dim == target_dim:
        return teacher_hidden
    device = teacher_hidden.device
    if _distill_proj is None or _distill_proj.in_features != src_dim or _distill_proj.out_features != target_dim:
        _distill_proj = torch.nn.Linear(src_dim, target_dim, bias=False).to(device)
        with torch.no_grad():
            torch.nn.init.orthogonal_(_distill_proj.weight)
    return _distill_proj(teacher_hidden)


from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    NeuronConfig, COMPACT, STANDARD, QualityFilter,
    ConfidenceGate, EarlyStopResonance,
)


def create_neuron(spec: str, device: str = "cpu") -> ResonanceNeuron:
    """Create a neuron of the given spec."""
    if spec == "compact":
        cfg = COMPACT
    elif spec == "expert":
        from taiji.resonance import EXPERT
        cfg = EXPERT
    else:
        cfg = STANDARD

    # Create a copy with required settings
    neuron_cfg = NeuronConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        spec=spec,
        vocab_size=256000,
        base_embed_dim=512,
        field_dim=min(cfg.hidden_size * 4, 4096),
    )
    neuron = ResonanceNeuron(neuron_cfg).to(device)
    return neuron


def distill_one_neuron(
    teacher_model,
    neuron: ResonanceNeuron,
    shared_embedding: torch.nn.Embedding,
    domain_data: torch.Tensor,
    domain_name: str,
    num_steps: int = 2000,
    batch_size: int = 4,
    lm_weight: float = 0.7,
    distill_weight: float = 0.3,
    lr: float = 5e-4,
    device: str = "cpu",
    log_every: int = 200,
) -> Dict[str, float]:
    """Distill a single neuron from the teacher model.

    Args:
        teacher_model: 1.5B teacher (eval mode).
        neuron: student ResonanceNeuron to train.
        shared_embedding: shared embedding from teacher.
        domain_data: [N, L] token IDs for this domain.
        domain_name: domain identifier string.
        num_steps: training steps.
        batch_size: batch size.
        lm_weight: LM loss weight.
        distill_weight: distillation loss weight.
        lr: learning rate.
        device: device.
        log_every: logging interval.

    Returns:
        {"final_loss", "final_ppl", "steps", "domain"}
    """
    dataset = TensorDataset(domain_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(neuron.parameters(), lr=lr)
    neuron.train()
    teacher_model.eval()

    total_lm_loss = 0.0
    total_distill_loss = 0.0
    step = 0

    t_start = time.time()

    for batch in loader:
        if step >= num_steps:
            break

        input_ids = batch[0].to(device)  # [B, L]

        # Shared embeddings (project teacher 2048-dim → neuron 512-dim)
        with torch.no_grad():
            teacher_emb = shared_embedding(input_ids)  # [B, L, 2048]
            # Project down to base_embed_dim (512) for neuron
            shared_emb = _project_embedding(teacher_emb, neuron.config.base_embed_dim)
            # Teacher hidden states (last token only for efficiency)
            teacher_hidden = extract_hidden_states(teacher_model, input_ids)
            teacher_last = teacher_hidden[:, -1, :]  # [B, hidden_dim]

        # Student forward
        result = neuron.forward(shared_emb, return_logits=True)
        student_logits = result["logits"]  # [B, L, vocab]
        student_hidden = result["hidden_before_write"]  # [B, hidden]

        # ── LM loss (next-token prediction) ──
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        # ── Distillation loss (align hidden direction) ──
        teacher_proj = _project_teacher_hidden(teacher_last, neuron.config.hidden_size)
        distill_loss = F.mse_loss(student_hidden, teacher_proj)

        # Combined loss
        loss = lm_weight * lm_loss + distill_weight * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_lm_loss += lm_loss.item()
        total_distill_loss += distill_loss.item()
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t_start
            avg_lm = total_lm_loss / step
            avg_distill = total_distill_loss / step
            ppl = math.exp(avg_lm) if avg_lm < 10 else float('inf')
            print(f"  [{domain_name}] step {step}/{num_steps} | "
                  f"lm={avg_lm:.4f} distill={avg_distill:.4f} | "
                  f"PPL≈{ppl:.1f} | {elapsed:.0f}s")

    avg_lm = total_lm_loss / max(step, 1)
    ppl = math.exp(avg_lm) if avg_lm < 10 else float('inf')

    return {
        "final_loss": avg_lm,
        "final_ppl": ppl,
        "steps": step,
        "domain": domain_name,
    }


def evaluate_neuron(neuron, shared_embedding, data, device="cpu", max_batches=50):
    """Quick PPL evaluation."""
    neuron.eval()
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch[0].to(device)
            teacher_emb = shared_embedding(input_ids)
            shared_emb = _project_embedding(teacher_emb, neuron.config.base_embed_dim)
            result = neuron.forward(shared_emb, return_logits=True)
            logits = result["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * shift_targets.numel()
            total_tokens += shift_targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Distill 5 domain neurons from 1.5B teacher")
    parser.add_argument("--checkpoint", default="e:/taiji/checkpoint-400000")
    parser.add_argument("--data_dir", default="data/distill")
    parser.add_argument("--output_dir", default="data/neurons")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_domains", nargs="*", default=[],
                        help="Domains to skip (e.g., --skip_domains math)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 2.7: Distill 5 Domain Neurons")
    print("=" * 60)

    # ── Load teacher ──
    print("\n[1/4] Loading teacher model...")
    teacher, embedding = load_teacher_model(args.checkpoint, device=args.device)
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B params")

    # ── Load data ──
    print("\n[2/4] Loading domain datasets...")
    data_path = os.path.join(args.data_dir, "domain_datasets.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run prepare_distill_data.py first.")
    datasets = torch.load(data_path, map_location="cpu", weights_only=True)

    # ── Distill each domain ──
    print("\n[3/4] Distilling neurons...")

    # Domain → spec mapping
    DOMAIN_SPECS = {
        "zh": "standard",
        "en": "standard",
        "code": "expert",
        "math": "expert",
        "general": "standard",
    }

    results = {}
    neurons = {}

    for domain in ["zh", "en", "code", "math", "general"]:
        if domain in args.skip_domains:
            print(f"\n  Skipping {domain}")
            continue

        if domain not in datasets:
            print(f"  WARNING: {domain} not in datasets, skipping")
            continue

        spec = DOMAIN_SPECS[domain]
        print(f"\n  --- {domain} ({spec}) ---")

        neuron = create_neuron(spec, device=args.device)
        n_params = sum(p.numel() for p in neuron.parameters())
        print(f"  Neuron params: {n_params:,}")

        result = distill_one_neuron(
            teacher, neuron, embedding,
            datasets[domain], domain,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

        # Quick PPL eval on own domain
        ppl_own = evaluate_neuron(neuron, embedding, datasets[domain], device=args.device)
        result["ppl_own"] = ppl_own
        print(f"\n  {domain}: loss={result['final_loss']:.4f}, "
              f"PPL(own)={ppl_own:.1f}, steps={result['steps']}")

        results[domain] = result
        neurons[domain] = neuron

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"neuron_{domain}.pt")
        torch.save({
            "neuron_config": neuron.config,
            "state_dict": neuron.state_dict(),
            "domain": domain,
            "result": result,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # ── Cross-domain evaluation ──
    print("\n[4/4] Cross-domain evaluation...")
    print("\n  PPL Matrix (row=neuron, col=test domain):")
    print(f"  {'':>10}", end="")
    for domain in datasets:
        print(f"{domain:>10}", end="")
    print()

    ppl_matrix = {}
    for neuron_domain, neuron in neurons.items():
        print(f"  {neuron_domain:>10}", end="")
        for test_domain, test_data in datasets.items():
            ppl = evaluate_neuron(neuron, embedding, test_data, device=args.device)
            print(f"{ppl:>10.1f}", end="")
            ppl_matrix[(neuron_domain, test_domain)] = ppl
        print()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DISTILLATION COMPLETE")
    print("=" * 60)
    for domain, r in results.items():
        ppl_own = ppl_matrix.get((domain, domain), float('inf'))
        other_ppls = [ppl_matrix.get((domain, d), float('inf')) for d in datasets if d != domain]
        min_other = min(other_ppls) if other_ppls else float('inf')
        gap = min_other - ppl_own if ppl_own < float('inf') else 0
        status = "PASS" if ppl_own < 50 and gap > 100 else "NEEDS WORK"
        print(f"  {domain}: PPL(own)={ppl_own:.1f}, gap={gap:.1f} [{status}]")

    print(f"\n  Neurons saved to: {args.output_dir}/")
    print(f"  Next: Phase 2.8 — Quality gate verification + joint fine-tuning")


if __name__ == "__main__":
    main()
