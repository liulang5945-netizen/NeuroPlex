"""Load field-conditioned neuron checkpoints and run the 1+1>2 verification only.

Salvages the Phase 3 verification step after a full distillation run.

Usage:
    python scripts/training/verify_from_ckpts.py \
        --ckpt_dir data/distill/neurons_out \
        --data_dir data/distill --device cpu
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse, os, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.training import distill_neurons as D
from taiji.resonance import ResonanceNeuron


def load_neuron(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["neuron_config"]
    sd = ckpt["state_dict"]
    # Legacy compat: ckpt trained before low-rank lm_head split.
    has_traditional_lm = any(k == "lm_head.weight" for k in sd)
    if has_traditional_lm and getattr(cfg, "lm_head_rank", 0) > 0:
        cfg.lm_head_rank = 0
        print(f"    [compat] {os.path.basename(ckpt_path)}: lm_head_rank=0 (traditional)")
    neuron = ResonanceNeuron(cfg).to(device)
    missing, unexpected = neuron.load_state_dict(sd, strict=False)
    if missing:
        print(f"    [compat] missing keys (init from scratch): {missing}")
    if unexpected:
        print(f"    [compat] unexpected keys (ignored): {unexpected}")
    neuron.eval()
    return neuron


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="data/distill/neurons_out")
    ap.add_argument("--data_dir", default="data/distill")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--phase1", action="store_true",
                    help="Use Phase 1 neuron_<domain>.pt instead of fieldcond")
    args = ap.parse_args()

    print("=" * 60); print("VERIFY FROM CHECKPOINTS (Phase 3 only)"); print("=" * 60)

    datasets = torch.load(os.path.join(args.data_dir, "domain_datasets.pt"),
                          map_location="cpu", weights_only=True)
    print(f"Loaded {len(datasets)} domain datasets")

    neurons = {}
    for domain in ["zh", "en", "code", "math", "general"]:
        name = f"neuron_{domain}.pt" if args.phase1 else f"neuron_{domain}_fieldcond.pt"
        path = os.path.join(args.ckpt_dir, name)
        if not os.path.exists(path):
            print(f"  [skip] {path} missing"); continue
        neurons[domain] = load_neuron(path, args.device)
        print(f"  loaded {domain}: {name}")

    if len(neurons) < 2:
        raise SystemExit("Need at least 2 neurons")

    cached_embeddings = {}
    cache_dir = os.path.join(args.data_dir, "cache")
    if os.path.isdir(cache_dir):
        for domain in datasets:
            cp = os.path.join(cache_dir, f"{domain}_cached.pt")
            if os.path.exists(cp):
                cached_embeddings[domain] = torch.load(cp, map_location="cpu", weights_only=True)
        print(f"Loaded cache for {len(cached_embeddings)} domains")
    else:
        cached_embeddings = None

    fixed_proj = None
    fp_path = os.path.join(args.data_dir, "fixed_proj.pt")
    if os.path.exists(fp_path):
        fixed_proj = torch.nn.Linear(2048, 512, bias=False).to(args.device)
        fixed_proj.load_state_dict(torch.load(fp_path, map_location=args.device))
        fixed_proj.requires_grad_(False)
        print("Loaded fixed_proj")

    print("\nRunning resonance verification (Round 1 vs Round 2)...\n")
    verification = D.verify_resonance(
        neurons, datasets, device=args.device,
        fixed_proj=fixed_proj, cached_embeddings=cached_embeddings,
        teacher_embedding=None,
    )

    out_json = os.path.join(args.ckpt_dir, "verify_1plus1.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(verification, f, ensure_ascii=False, indent=2, default=float)
        print(f"\nSaved verification results to {out_json}")
    except Exception as exc:
        print(f"[warn] could not write {out_json}: {exc}")


if __name__ == "__main__":
    main()
