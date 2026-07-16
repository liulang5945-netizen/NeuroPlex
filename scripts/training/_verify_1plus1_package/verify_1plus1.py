#!/usr/bin/env python3
"""1+1>2 resonance field verification script."""
import math, os, sys, argparse, time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble

def to_neuron_config(ckpt_cfg):
    from dataclasses import fields, is_dataclass
    if isinstance(ckpt_cfg, dict):
        return ckpt_cfg
    if is_dataclass(ckpt_cfg):
        return {f.name: getattr(ckpt_cfg, f.name) for f in fields(ckpt_cfg)}
    if hasattr(ckpt_cfg, '__dict__'):
        return ckpt_cfg.__dict__
    return ckpt_cfg

def load_neuron(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from taiji.resonance.config import NeuronConfig
    cfg_dict = to_neuron_config(ckpt.get('neuron_config') or ckpt.get('config'))
    valid_fields = {f.name for f in NeuronConfig.__dataclass_fields__.values()}
    cfg = NeuronConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
    neuron = ResonanceNeuron(cfg).to(device)
    sd = ckpt['state_dict']
    nsd = neuron.state_dict()
    filtered = OrderedDict((k, v) for k, v in sd.items() if k in nsd and nsd[k].shape == v.shape)
    neuron.load_state_dict(filtered, strict=False)
    neuron.eval()
    return neuron

class PadField(ResonanceField):
    def _pad(self, v):
        if v.dim() == 1:
            v = v.unsqueeze(0)
        d = v.shape[-1]
        if d < self.dim:
            return torch.cat([v, torch.zeros(*v.shape[:-1], self.dim - d, device=v.device, dtype=v.dtype)], dim=-1)
        return v[..., :self.dim] if d > self.dim else v
    def write(self, nid, v):
        return super().write(nid, self._pad(v))
    def score(self, v):
        return super().score(self._pad(v))

def compute_ppl_single(neuron, shared_embedding, token_ids, max_batches=20, device='cpu'):
    ds = TensorDataset(token_ids)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            ids = batch[0].to(device)
            emb = shared_embedding(ids)
            result = neuron.forward(emb, return_logits=True)
            logits = result['logits']
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=-100)
            total_loss += loss.item() * shift_targets.numel()
            total_tokens += shift_targets.numel()
    if total_tokens == 0:
        return float('inf')
    return math.exp(min(total_loss / total_tokens, 10))

def compute_ppl_ensemble(neurons, shared_embedding, token_ids, max_batches=20, max_rounds=3, device='cpu'):
    max_dim = max(n.config.field_dim for n in neurons.values())
    field = PadField(dim=max_dim).to(device)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=max_rounds)
    ds = TensorDataset(token_ids)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= max_batches:
                break
            ids = batch[0].to(device)
            emb = shared_embedding(ids)
            result = ensemble.forward(emb, return_logits=True)
            if 'weighted_logits' not in result:
                continue
            logits = result['weighted_logits']
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=-100)
            total_loss += loss.item() * shift_targets.numel()
            total_tokens += shift_targets.numel()
    if total_tokens == 0:
        return float('inf')
    return math.exp(min(total_loss / total_tokens, 10))

def main():
    parser = argparse.ArgumentParser(description='1+1>2 resonance verification')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_batches', type=int, default=20)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--neurons_dir', default='data/neurons_v2')
    parser.add_argument('--data_path', default='data/real/domain_datasets.pt')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    all_data = torch.load(args.data_path, map_location='cpu')
    available = {}
    for domain in ['zh','en','code','math','general']:
        p = os.path.join(args.neurons_dir, f'neuron_{domain}.pt')
        if os.path.exists(p):
            n = load_neuron(p, device=device)
            nparams = sum(p.numel() for p in n.parameters())
            print(f'  [{domain}] {n.config.spec}: {nparams/1e6:.0f}M field_dim={n.config.field_dim}')
            available[domain] = n
    if len(available) < 2:
        print('Need 2+ neurons'); return
    dim = next(iter(available.values())).config.base_embed_dim
    vsz = next(iter(available.values())).config.vocab_size
    emb = nn.Embedding(vsz, dim).to(device)
    domains = list(available.keys())
    results = []
    for a, b in [(a,b) for i,a in enumerate(domains) for b in domains[i+1:]]:
        for td in [a, b, 'general']:
            if td not in all_data: continue
            d = all_data[td]
            n1, n2 = available[a], available[b]
            p1 = compute_ppl_single(n1, emb, d, args.max_batches, device)
            p2 = compute_ppl_single(n2, emb, d, args.max_batches, device)
            ens = {a: n1, b: n2}
            pe = compute_ppl_ensemble(ens, emb, d, args.max_batches, args.max_rounds, device)
            bs = min(p1, p2)
            imp = (bs-pe)/bs*100 if bs>0 else 0
            tag = 'WIN' if imp>0 else 'LOSS'
            print(f'{a}+{b} on {td}: {p1:.0f}/{p2:.0f} -> ens={pe:.0f} best={bs:.0f} delta={imp:+.1f}% [{tag}]')
            results.append({'pair':f'{a}+{b}','td':td,'p1':p1,'p2':p2,'pe':pe,'bs':bs,'imp':imp,'win':imp>0})
    wins = sum(1 for r in results if r['win'])
    print(f'\nWins: {wins}/{len(results)}, avg delta: {sum(r["imp"] for r in results)/len(results):+.1f}%')
    if wins:
        print('1+1>2 detected!')

if __name__ == '__main__':
    main()
