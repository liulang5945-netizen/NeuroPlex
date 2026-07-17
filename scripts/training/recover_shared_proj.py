"""Recover a shared embedding projection (H10 fix).

Problem: distill_neurons.py used a global random orthogonal Linear(2048,512)
that was never saved. Verification creates a different random projection,
so neurons never see the training-time embedding distribution.

Solution: freeze all 5 neurons, train ONLY a shared Linear(2048->512)
using the teacher's real embeddings as input and each neuron's LM loss
as the target. Save to data/shared_proj.pt.

Usage:
    python scripts/training/recover_shared_proj.py [--steps 500] [--batch_size 1]
"""
import argparse, math, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.shared_embed import SharedEmbedProj
from taiji.training.checkpoint_bridge import load_teacher_model


def load_neuron_frozen(ckpt_path, device="cpu"):
    """Load a v2 checkpoint with v1_compat field I/O, all params frozen."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["neuron_config"]
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.v1_compat = True
    neuron.eval()
    for p in neuron.parameters():
        p.requires_grad_(False)
    return neuron


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher_dir", default="E:/taiji/FINAL/checkpoint-481000")
    parser.add_argument("--neurons_dir", default="data/neurons_v2")
    parser.add_argument("--data_path", default="data/distill/domain_datasets.pt")
    parser.add_argument("--output", default="data/shared_proj.pt")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    domains = ["zh", "en", "code", "math", "general"]
    device = "cpu"

    print("=" * 60)
    print("  H10 Fix: Recover Shared Embedding Projection")
    print("=" * 60)

    # Step 1: Load teacher (for the 2048-dim embedding table)
    print(f"\n[1/4] Loading teacher from {args.teacher_dir} ...")
    t0 = time.time()
    teacher, embedding = load_teacher_model(args.teacher_dir, device=device)
    # Freeze embedding (we only need the lookup, no grad for teacher)
    embedding.eval()
    for p in embedding.parameters():
        p.requires_grad_(False)
    print(f"  Teacher loaded in {time.time()-t0:.1f}s | embedding: {embedding.weight.shape}")

    # Step 2: Load all 5 neurons (frozen)
    print(f"\n[2/4] Loading neurons (frozen, v1_compat) ...")
    neurons = {}
    for d in domains:
        path = os.path.join(args.neurons_dir, f"neuron_{d}.pt")
        if not os.path.exists(path):
            print(f"  skip {d}: not found")
            continue
        t0 = time.time()
        n = load_neuron_frozen(path, device=device)
        n.freeze_fingerprint()
        neurons[d] = n
        pcount = sum(p.numel() for p in n.parameters())
        print(f"  {d:8s}: {pcount/1e6:.0f}M params, "
              f"field_dim={n.config.field_dim}, "
              f"{time.time()-t0:.1f}s")

    # Step 3: Load domain data
    print(f"\n[3/4] Loading distill data ...")
    data = torch.load(args.data_path, map_location="cpu", weights_only=True)
    for k, v in data.items():
        print(f"  {k}: {v.shape}")
    avail_domains = [d for d in domains if d in neurons and d in data]
    print(f"  Available: {avail_domains}")

    # Step 4: Train the projection
    print(f"\n[4/4] Training shared projection ({args.steps} steps, bs={args.batch_size}) ...")
    proj = SharedEmbedProj(src_dim=2048, target_dim=512)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=args.lr)

    # Build cycling data iterator
    domain_iters = {}
    for d in avail_domains:
        domain_iters[d] = 0

    t_start = time.time()
    total_loss = 0.0

    for step in range(args.steps):
        # Pick a domain (round-robin for balance)
        domain = avail_domains[step % len(avail_domains)]
        neuron = neurons[domain]
        d_data = data[domain]

        # Get batch (cycle through data)
        idx = domain_iters[domain]
        bs = args.batch_size
        if idx + bs > len(d_data):
            idx = 0
            domain_iters[domain] = 0
        batch_ids = d_data[idx:idx+bs].to(device)
        domain_iters[domain] = idx + bs

        # Forward: teacher embedding -> projection -> neuron -> LM loss
        with torch.no_grad():
            emb_2048 = embedding(batch_ids)  # [B, L, 2048]

        emb_512 = proj(emb_2048.detach())  # [B, L, 512] (grad through proj)
        result = neuron.forward(emb_512, return_logits=True)
        logits = result["logits"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = batch_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % args.log_every == 0:
            avg = total_loss / (step + 1)
            ppl = math.exp(min(avg, 10))
            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (args.steps - step - 1)
            print(f"  step {step+1:4d}/{args.steps} [{domain:8s}] "
                  f"loss={avg:.4f} PPL={ppl:.1f} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    # Save the projection
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    proj.save(args.output)
    print(f"\n  Saved projection to {args.output}")

    # Quick sanity: load it back and test
    loaded = SharedEmbedProj.load(args.output)
    test_emb = torch.randn(1, 8, 2048)
    test_out = loaded(test_emb)
    print(f"  Reload test: {test_emb.shape} -> {test_out.shape} (OK)")

    print(f"\n  Total time: {time.time()-t_start:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
