"""Real 1+1>2 test with teacher embeddings (not random).

Loads teacher model for real 2048-dim embeddings → project to 512 → neuron.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.training.checkpoint_bridge import load_teacher_model

def load_v2(domain):
    ckpt = torch.load(f"data/neurons_v2/neuron_{domain}.pt", map_location="cpu", weights_only=False)
    n = ResonanceNeuron(ckpt["neuron_config"])
    n.load_state_dict(ckpt["state_dict"])
    n.eval()
    return n

class PadField(ResonanceField):
    def _pad(self, v):
        if v.dim()==1: v=v.unsqueeze(0)
        d=v.shape[-1]
        return torch.cat([v, torch.zeros(*v.shape[:-1], self.dim-d, device=v.device, dtype=v.dtype)], dim=-1) if d<self.dim else v[...,:self.dim] if d>self.dim else v
    def write(self, nid, v): return super().write(nid, self._pad(v))
    def score(self, v): return super().score(self._pad(v))

# Projection cache
_proj = None
def proj_embed(emb, target_dim=512):
    global _proj
    sd = emb.shape[-1]
    if sd == target_dim: return emb
    if _proj is None or _proj.in_features != sd:
        _proj = torch.nn.Linear(sd, target_dim, bias=False)
        torch.nn.init.orthogonal_(_proj.weight)
    return _proj(emb)

def ppl_single(neuron, embedding, data, max_b=10):
    ds = TensorDataset(data); dl = DataLoader(ds, batch_size=2, shuffle=False)
    tl, tt = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= max_b: break
            ids = b[0]
            emb = proj_embed(embedding(ids))
            r = neuron.forward(emb, return_logits=True)
            logits = r["logits"]
            sl = logits[:,:-1,:].contiguous(); st = ids[:,1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item()*st.numel(); tt += st.numel()
    return math.exp(min(tl/max(tt,1), 10))

def ppl_ensemble(neurons, embedding, data, max_b=10):
    fd = max(n.config.field_dim for n in neurons.values())
    field = PadField(dim=fd)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)
    ds = TensorDataset(data); dl = DataLoader(ds, batch_size=2, shuffle=False)
    tl, tt = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= max_b: break
            ids = b[0]
            emb = proj_embed(embedding(ids))
            r = ensemble.forward(emb, return_logits=True)
            if "weighted_logits" not in r: continue
            logits = r["weighted_logits"]
            sl = logits[:,:-1,:].contiguous(); st = ids[:,1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            tl += loss.item()*st.numel(); tt += st.numel()
    return math.exp(min(tl/max(tt,1), 10))

print("Loading teacher model (1.55B)...")
teacher, embedding = load_teacher_model("e:/taiji/checkpoint-400000", device="cpu")

print("Loading v2 neurons...")
zh = load_v2("zh")
code = load_v2("code")

data = torch.load("data/distill/domain_datasets.pt", map_location="cpu", weights_only=True)

print("\n=== 1+1>2 REAL Test (teacher embeddings) ===")
for test_domain in ["zh", "code"]:
    d = data[test_domain]
    p_zh = ppl_single(zh, embedding, d)
    p_code = ppl_single(code, embedding, d)
    p_both = ppl_ensemble({"zh": zh, "code": code}, embedding, d)
    best_single = min(p_zh, p_code)
    imp = (best_single - p_both) / best_single * 100
    print(f"\n{test_domain}:")
    print(f"  zh alone:     {p_zh:.0f}")
    print(f"  code alone:   {p_code:.0f}")
    print(f"  zh+code:      {p_both:.0f}")
    print(f"  best single:  {best_single:.0f}")
    print(f"  delta = {imp:+.1f}% [{'1+1>2!' if imp > 0 else 'no gain'}]")
