"""1+1>2 verification: single neuron PPL vs two-neuron ensemble PPL.
Uses REAL teacher hidden states (from 1.55B checkpoint) projected through
SharedEmbedProj, so the shared embeddings match the training distribution.

Usage:  python scripts/training/verify_1plus1.py
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance.shared_embed import SharedEmbedProj
from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble

# ---------------------------------------------------------------------------
# one-time globals (expensive to reload)
# ---------------------------------------------------------------------------
_teacher = None
_proj: SharedEmbedProj | None = None  # SharedEmbedProj


def _get_teacher(device: str = "cpu"):
    global _teacher
    if _teacher is None:
        print("  [loading teacher 1.55B ...]")
        _teacher, _emb = load_teacher_model(
            "E:/taiji-neuron/checkpoint-481000", device=device
        )
    return _teacher


def _get_proj() -> SharedEmbedProj:
    global _proj
    if _proj is None:
        _proj = SharedEmbedProj.load("data/shared_proj.pt")
    return _proj


def _make_shared_embeddings(input_ids: torch.Tensor,
                            teacher, proj) -> torch.Tensor:
    """Token IDs -> teacher hidden states -> SharedEmbedProj -> [B,L,512]."""
    hidden = extract_hidden_states(teacher, input_ids)        # [B, L, 2048]
    return proj(hidden)                                        # [B, L, 512]


def load_neuron(domain: str, neuron_dir: str = "data/neurons") -> ResonanceNeuron:
    path = os.path.join(neuron_dir, f"neuron_{domain}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n = ResonanceNeuron(ckpt["neuron_config"])
    n.load_state_dict(ckpt["state_dict"], strict=False)
    n.v1_compat = True       # old ckpt was distilled with v1 paths
    n.eval()
    return n


def ppl_single(neuron: ResonanceNeuron, data: torch.Tensor,
               teacher, proj, max_batches: int = 5) -> float:
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            ids = batch[0]
            emb = _make_shared_embeddings(ids, teacher, proj)
            out = neuron.forward(emb, return_logits=True)
            logits = out["logits"]
            shift = logits[:, :-1, :].contiguous()
            targets = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift.view(-1, shift.size(-1)), targets.view(-1), ignore_index=-100,
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    avg = total_loss / max(total_tokens, 1)
    return math.exp(min(avg, 15.0))   # higher cap than before


def ppl_ensemble(neurons: dict, data: torch.Tensor,
                 teacher, proj, max_batches: int = 5) -> float:
    fd = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(fd)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2)
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            ids = batch[0]
            emb = _make_shared_embeddings(ids, teacher, proj)
            res = ens.forward(emb, return_logits=True, active_filter=False, enable_gating=False)
            if "weighted_logits" not in res:
                continue
            logits = res["weighted_logits"]
            shift = logits[:, :-1, :].contiguous()
            targets = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift.view(-1, shift.size(-1)), targets.view(-1), ignore_index=-100,
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    avg = total_loss / max(total_tokens, 1)
    return math.exp(min(avg, 15.0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("1+1>2 Verification  -  REAL teacher hidden states")
    print("Loading 1.55B teacher + SharedEmbedProj for actual PPL")
    print("=" * 60)

    # Load teacher & proj ---------------------------------------------------
    print("\n[1/4] Loading teacher + SharedEmbedProj ...")
    teacher = _get_teacher()
    proj = _get_proj()
    print(f"  shared_proj: {proj.src_dim}d -> {proj.target_dim}d")

    # Load neurons ----------------------------------------------------------
    print("\n[2/4] Loading neurons (zh + en, both STANDARD 3072)...")
    zh = load_neuron("zh")
    en = load_neuron("en")
    print(f"  zh: {sum(p.numel() for p in zh.parameters())/1e6:.0f}M params")
    print(f"  en: {sum(p.numel() for p in en.parameters())/1e6:.0f}M params")

    # Load data ------------------------------------------------------------
    print("\n[3/4] Loading domain datasets...")
    data_path = "data/distill/domain_datasets.pt"
    datasets = torch.load(data_path, map_location="cpu", weights_only=True)
    for k, v in datasets.items():
        print(f"  {k}: {v.shape}")

    # Run comparison -------------------------------------------------------
    print("\n[4/4] PPL comparison (5 batches each, may be slow due to 1.55B fwd)...")
    neurons = {"zh": zh, "en": en}

    for test_domain in ["zh", "en"]:
        if test_domain not in datasets:
            continue
        d = datasets[test_domain]
        print(f"\n  --- {test_domain} ---")

        print(f"    zh alone ...", end=" ", flush=True)
        p_zh = ppl_single(zh, d, teacher, proj)

        print(f"en alone ...", end=" ", flush=True)
        p_en = ppl_single(en, d, teacher, proj)

        print(f"ensemble ...", end=" ", flush=True)
        p_both = ppl_ensemble(neurons, d, teacher, proj)

        best_single = min(p_zh, p_en)
        improvement = (best_single - p_both) / best_single * 100

        print()
        print(f"    zh alone:      {p_zh:>8.0f}")
        print(f"    en alone:      {p_en:>8.0f}")
        print(f"    zh+en ensemble:{p_both:>8.0f}")
        print(f"    best single:   {best_single:>8.0f}")
        print(f"    delta:         {improvement:>+8.1f}%  "
              f"[{'1+1>2!' if improvement > 0 else 'no gain yet'}]")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()