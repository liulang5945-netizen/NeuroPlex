"""Local 1+1>2 verification with true resonance enabled.

Fixes:
  H1: checkpoint compatibility (strict=False + v1_compat)
  H2: batch_size=1 (avoids cross-sample field contamination)
  H3: max_rounds=3 (actually enables field_read resonance)
  H10: loads trained shared projection from data/shared_proj.pt
  H11: groups neurons by field_dim (can't mix 3072 and 4096)
  H14: includes general neuron in ensemble tests (was wrongly skipped)
"""
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.config import NeuronConfig
from taiji.resonance.shared_embed import SharedEmbedProj
from taiji.training.checkpoint_bridge import load_teacher_model


def load_neuron_compat(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["neuron_config"]
    neuron = ResonanceNeuron(cfg).to(device)
    sd = ckpt["state_dict"]
    # H1: auto-detect v1 vs v2 from the actual parameter keys present.
    # v2 neurons carry field_pool_query + field_read_gate; existing
    # neurons_v2/* checkpoints are v1 and must run in v1-compat mode.
    has_v2 = {"field_pool_query", "field_read_gate.weight"} <= set(sd.keys())
    neuron.load_state_dict(sd, strict=False)
    neuron.v1_compat = not has_v2
    neuron.eval()
    return neuron


_shared_proj = None

def proj_embed(emb, target_dim=512):
    global _shared_proj
    sd = emb.shape[-1]
    if sd == target_dim:
        return emb
    if _shared_proj is None:
        proj_path = "data/shared_proj.pt"
        if os.path.exists(proj_path):
            _shared_proj = SharedEmbedProj.load(proj_path)
            print(f"  [proj] loaded trained projection from {proj_path}")
        else:
            _shared_proj = SharedEmbedProj()
            print(f"  [proj] WARNING: data/shared_proj.pt not found, using random")
    return _shared_proj(emb)


def ppl_single(neuron, embedding, data, max_b=5, device="cpu"):
    tl, tt = 0.0, 0
    for i in range(min(max_b, len(data))):
        ids = data[i:i+1].to(device)
        with torch.no_grad():
            emb = proj_embed(embedding(ids))
            r = neuron.forward(emb, return_logits=True)
            logits = r["logits"]
            sl = logits[:, :-1, :].contiguous()
            st = ids[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item() * st.numel()
            tt += st.numel()
    return math.exp(min(tl / max(tt, 1), 10))


def ppl_ensemble(neurons, embedding, data, max_rounds=3, max_b=5, device="cpu"):
    field_dims = {n.config.field_dim for n in neurons.values()}
    if len(field_dims) > 1:
        raise ValueError(
            f"[verify] neurons disagree on field_dim: {field_dims}. "
            f"H9 unified all v3 specs to 4096. Old 3072 checkpoints must be "
            f"re-distilled before they can co-resonate with v3 neurons."
        )
    fd = field_dims.pop()
    field = ResonanceField(dim=fd, device=torch.device(device))
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=max_rounds)
    tl, tt = 0.0, 0
    for i in range(min(max_b, len(data))):
        ids = data[i:i+1].to(device)
        ensemble.field.reset()
        with torch.no_grad():
            emb = proj_embed(embedding(ids))
            r = ensemble.forward(emb, return_logits=True)
            if "weighted_logits" not in r:
                continue
            logits = r["weighted_logits"]
            sl = logits[:, :-1, :].contiguous()
            st = ids[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item() * st.numel()
            tt += st.numel()
    return math.exp(min(tl / max(tt, 1), 10))


def ppl_ensemble_v1(neurons, embedding, data, max_b=5, device="cpu"):
    return ppl_ensemble(neurons, embedding, data, max_rounds=1, max_b=max_b, device=device)


def main():
    teacher_dir = "E:/taiji/FINAL/checkpoint-481000"
    device = "cpu"
    max_b_single = 5
    max_b_ensem = 5

    print("=" * 60)
    print("  1+1>2 Local Verification (true resonance, v1-compat)")
    print("  H10: shared projection | H11: field_dim grouping | H14: general included")
    print("=" * 60)

    print(f"\n[1/3] Loading teacher from {teacher_dir} ...")
    t0 = time.time()
    teacher, embedding = load_teacher_model(teacher_dir, device=device)
    print(f"  Teacher loaded in {time.time()-t0:.1f}s")

    print(f"\n[2/3] Loading v2 neurons (v1_compat mode) ...")
    domains = ["zh", "code", "en", "math", "general"]
    neurons_all = {}
    for d in domains:
        path = f"data/neurons_v2/neuron_{d}.pt"
        if not os.path.exists(path):
            print(f"  skip {d}: {path} not found")
            continue
        n = load_neuron_compat(path, device=device)
        n.freeze_fingerprint()
        neurons_all[d] = n
        print(f"  {d}: spec={n.config.spec}, field_dim={n.config.field_dim}, "
              f"{sum(p.numel() for p in n.parameters())/1e6:.0f}M")

    groups = {}
    for d, n in neurons_all.items():
        fd = n.config.field_dim
        groups.setdefault(fd, []).append(d)
    print(f"\n  field_dim groups:")
    for fd, members in sorted(groups.items()):
        print(f"    {fd}: {members}")

    print(f"\n[3/3] Loading distill data ...")
    data = torch.load("data/distill/domain_datasets.pt", map_location="cpu", weights_only=True)
    for k, v in data.items():
        print(f"  {k}: {v.shape}")

    for fd, group_domains in sorted(groups.items()):
        group_neurons = {d: neurons_all[d] for d in group_domains if d in neurons_all}
        if len(group_neurons) < 2:
            print(f"\n  Skipping group {fd}: need >=2 neurons, got {len(group_neurons)}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Group field_dim={fd}: {list(group_neurons.keys())}")
        print(f"{'=' * 60}")

        test_domains = [d for d in group_domains if d in data]
        for test_domain in test_domains[:3]:
            d = data[test_domain]
            print(f"\n--- Domain: {test_domain} ---")

            singles = {}
            for nid, n in group_neurons.items():
                t0 = time.time()
                p = ppl_single(n, embedding, d, max_b=max_b_single, device=device)
                singles[nid] = p
                print(f"  {nid:8s} single:  PPL={p:6.1f}  ({time.time()-t0:.1f}s)")

            if len(singles) < 2:
                print("  (need >=2 neurons for ensemble test)")
                continue

            best_nid = min(singles, key=singles.get)
            best_single = singles[best_nid]
            top2 = sorted(singles, key=singles.get)[:2]
            ens_neurons = {nid: group_neurons[nid] for nid in top2}

            t0 = time.time()
            p_v1 = ppl_ensemble_v1(ens_neurons, embedding, d, max_b=max_b_ensem, device=device)
            print(f"  ens({'+'.join(top2)}) r=1: PPL={p_v1:6.1f}  ({time.time()-t0:.1f}s)  [no resonance]")

            for r in [2, 3]:
                t0 = time.time()
                p_r = ppl_ensemble(ens_neurons, embedding, d, max_rounds=r, max_b=max_b_ensem, device=device)
                imp_r = (best_single - p_r) / best_single * 100
                tag = "1+1>2!" if imp_r > 0 else "no gain"
                print(f"  ens({'+'.join(top2)}) r={r}: PPL={p_r:6.1f}  ({time.time()-t0:.1f}s)  "
                      f"delta={imp_r:+.1f}% [{tag}]")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
