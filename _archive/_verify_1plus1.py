"""Minimal 1+1>2 verification with v2 neurons.

Compares: single neuron PPL vs two-neuron ensemble PPL.
Uses same random embeddings for fair comparison.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble

def load_v2(domain):
    ckpt = torch.load(f"data/neurons_v2/neuron_{domain}.pt", map_location="cpu", weights_only=False)
    n = ResonanceNeuron(ckpt["neuron_config"])
    n.load_state_dict(ckpt["state_dict"])
    n.eval()
    return n

def ppl_single(neuron, data, max_b=10, seed=42):
    """PPL of single neuron on given data."""
    torch.manual_seed(seed)
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    tl, tt = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= max_b: break
            ids = b[0]
            emb = torch.randn(ids.shape[0], ids.shape[1], 512)
            r = neuron.forward(emb, return_logits=True)
            logits = r["logits"]
            sl = logits[:,:-1,:].contiguous()
            st = ids[:,1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item() * st.numel()
            tt += st.numel()
    return math.exp(min(tl/max(tt,1), 10))

class PadField(ResonanceField):
    def _pad(self, v):
        if v.dim()==1: v=v.unsqueeze(0)
        d=v.shape[-1]
        return torch.cat([v, torch.zeros(*v.shape[:-1], self.dim-d, device=v.device, dtype=v.dtype)], dim=-1) if d<self.dim else v[...,:self.dim] if d>self.dim else v
    def write(self, nid, v): return super().write(nid, self._pad(v))
    def score(self, v): return super().score(self._pad(v))

def ppl_ensemble(neurons, data, max_b=10, seed=42):
    """PPL of neuron ensemble on given data."""
    torch.manual_seed(seed)
    fd = max(n.config.field_dim for n in neurons.values())
    field = PadField(dim=fd)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)
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
            sl = logits[:,:-1,:].contiguous()
            st = ids[:,1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item() * st.numel()
            tt += st.numel()
    return math.exp(min(tl/max(tt,1), 10))

print("Loading v2 neurons...")
zh = load_v2("zh")
code = load_v2("code")
print(f"zh: {sum(p.numel() for p in zh.parameters())/1e6:.0f}M, code: {sum(p.numel() for p in code.parameters())/1e6:.0f}M")

data = torch.load("data/distill/domain_datasets.pt", map_location="cpu", weights_only=True)

print("\n=== 1+1>2 Verification ===")
for test_domain in ["zh", "code"]:
    d = data[test_domain]
    p_zh = ppl_single(zh, d)
    p_code = ppl_single(code, d)
    p_both = ppl_ensemble({"zh": zh, "code": code}, d)
    best_single = min(p_zh, p_code)
    improvement = (best_single - p_both) / best_single * 100
    print(f"\n{test_domain}:")
    print(f"  zh alone:    {p_zh:.0f}")
    print(f"  code alone:  {p_code:.0f}")
    print(f"  zh+code:     {p_both:.0f}")
    print(f"  best single: {best_single:.0f}")
    tag = "1+1>2!" if improvement > 0 else "no gain yet"
    print(f"  delta = {improvement:+.1f}% [{tag}]")
