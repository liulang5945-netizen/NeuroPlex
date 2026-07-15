"""Phase 3 Quick: Compare division strategies with tiny test.
Uses test_division_path.py logic but with actual distilled neurons (just 2 for speed).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    ScaleLayering, DivisionPath,
)


class PadField(ResonanceField):
    def _pad(self, v):
        if v.dim() == 1: v = v.unsqueeze(0)
        vd = v.shape[-1]
        if vd < self.dim:
            return torch.cat([v, torch.zeros(*v.shape[:-1], self.dim - vd, device=v.device, dtype=v.dtype)], dim=-1)
        return v[..., :self.dim] if vd > self.dim else v
    def write(self, nid, v): return super().write(nid, self._pad(v))
    def score(self, v): return super().score(self._pad(v))


def quick_ppl(ensemble, data, max_b=5):
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    tl, tt = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= max_b: break
            ids = b[0]
            emb = torch.randn(ids.shape[0], ids.shape[1], 512)
            r = ensemble.forward(emb, return_logits=True)
            if "weighted_logits" not in r: continue
            logits = r["weighted_logits"]
            sl = logits[:, :-1, :].contiguous()
            st = ids[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item() * st.numel()
            tt += st.numel()
    return math.exp(min(tl / max(tt, 1), 10))


def main():
    print("Phase 3 Quick: Division Strategy Comparison")
    print("=" * 50)

    # Load 2 neurons (zh + code) for quick test
    neurons, specs = {}, {}
    for d in ["zh", "code"]:
        ckpt = torch.load(f"data/neurons/neuron_{d}.pt", map_location="cpu", weights_only=False)
        cfg = ckpt["neuron_config"]
        n = ResonanceNeuron(cfg)
        n.load_state_dict(ckpt["state_dict"])
        n.eval()
        neurons[d] = n
        specs[d] = cfg.spec
        print(f"  {d}: {cfg.spec}, {sum(p.numel() for p in n.parameters())/1e6:.0f}M params")

    # Load data
    data = torch.load("data/distill/domain_datasets.pt", map_location="cpu", weights_only=True)

    # Field dim = max of neuron field_dims
    fd = max(n.config.field_dim for n in neurons.values())

    # Test on zh and code domains
    for test_domain in ["zh", "code"]:
        if test_domain not in data: continue
        print(f"\n  Test domain: {test_domain}")
        test_data = data[test_domain]

        for strat_name, strat_obj in [
            ("consensus", None),
            ("scale_layering", ScaleLayering()),
        ]:
            field = PadField(dim=fd)
            if strat_obj is not None:
                # For scale_layering: override division_path to just use scale_layering
                dp = DivisionPath()
                dp.compute_final_weights = lambda *a, **kw: strat_obj.compute_weights(specs, {nid: 0.5 for nid in neurons})
            else:
                dp = None
            ensemble = ResonanceEnsemble(neurons, field, max_rounds=1, division_path=dp)
            ppl = quick_ppl(ensemble, test_data)
            print(f"    {strat_name:>20}: PPL={ppl:.1f}")

    print("\n" + "=" * 50)
    print("Done. (Full experiment needs more training steps for meaningful PPL comparison)")
    print("=" * 50)


if __name__ == "__main__":
    main()
