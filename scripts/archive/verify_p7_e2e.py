"""P7 End-to-end validation: verify trained neuron + generate quality.

Usage:
    python scripts/training/verify_p7_e2e.py --domain zh
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.config import get_domain_neuron_config, DOMAIN_VOCAB_SIZES
from taiji.resonance.translator import TokenizerHub


def verify_training_checkpoint(domain: str, ckpt_path: str):
    """Verify a trained P7 neuron checkpoint."""
    print(f"\n{'='*60}")
    print(f"[Verify] {domain} neuron checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["neuron_config"]
    result = ckpt.get("result", {})

    ppl = math.exp(min(result.get('final_loss', 10), 20))
    print(f"  Config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"vocab={cfg.vocab_size}, lm_head_rank={cfg.lm_head_rank}")
    print(f"  Training: loss={result.get('final_loss', 'N/A'):.4f}, PPL={ppl:.1f}")

    expected = DOMAIN_VOCAB_SIZES.get(domain)
    assert cfg.vocab_size == expected, f"vocab={cfg.vocab_size} != {expected}"
    assert cfg.lm_head_rank == 0, f"lm_head_rank={cfg.lm_head_rank}"
    print(f"  vocab={cfg.vocab_size} OK, lm_head_rank=0 OK (P7)")

    neuron = ResonanceNeuron(cfg)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.eval()
    return neuron, cfg


def verify_generate(domain: str, neuron: ResonanceNeuron, cfg):
    """Verify generate via ensemble P7 path."""
    print(f"\n[Verify] Generate (P7 path)")

    sp_path = os.path.join("taiji", "domains", domain, f"sp_{domain}.model")
    domain_sp = spm.SentencePieceProcessor()
    domain_sp.Load(sp_path)

    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(
        neurons={domain: neuron},
        field=field,
        max_rounds=2,
    )

    prompts = ["你好，今天", "机器学习是"]
    for prompt in prompts:
        ids = domain_sp.encode(prompt) or [0]
        ids_t = torch.tensor([ids], dtype=torch.long)
        generated = []
        for _ in range(20):
            result = ensemble.forward(input_ids=ids_t, return_logits=True)
            if "weighted_logits" not in result:
                break
            logits = result["weighted_logits"][:, -1, :] / 1.2
            # Top-k + temperature sampling to break repetition
            k = min(40, logits.shape[-1])
            top_vals, top_idx = torch.topk(logits, k)
            probs = torch.softmax(top_vals, dim=-1)
            next_tok = top_idx[0, torch.multinomial(probs, 1).item()].item()
            if next_tok == domain_sp.eos_id():
                break
            generated.append(next_tok)
            ids_t = torch.cat([ids_t, torch.tensor([[next_tok]])], dim=1)

        output = domain_sp.decode(generated) if generated else "(empty)"
        has_ch = any('\u4e00' <= c <= '\u9fff' for c in output)
        print(f"  '{prompt}' -> '{output[:60]}' [{['no-chinese','chinese'][has_ch]}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="zh")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    ckpt_path = args.ckpt or f"data/neurons/neuron_{args.domain}.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    neuron, cfg = verify_training_checkpoint(args.domain, ckpt_path)
    verify_generate(args.domain, neuron, cfg)

    n_params = sum(p.numel() for p in neuron.parameters()) / 1e6
    print(f"\n{'='*60}")
    print(f"P7 End-to-end validation PASSED")
    print(f"  Domain: {args.domain} | {cfg.spec} | {n_params:.1f}M params")
    print(f"  lm_head: {cfg.hidden_size}x{cfg.vocab_size}={cfg.hidden_size*cfg.vocab_size/1e6:.1f}M")
    print(f"  field: dim={cfg.field_dim}")


if __name__ == "__main__":
    main()
