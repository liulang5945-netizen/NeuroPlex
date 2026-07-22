"""V3 Joint Training (5 domains) + Generation test."""
import math, os, sys, time, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print = functools.partial(print, flush=True)

import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.translator import batch_align_and_embed, TokenizerHub
from taiji.brain.cortex import Cortex
from scripts.training.train_neuron import load_domain_tokenizer, load_domain_texts

DOMAINS = ["zh", "en", "code", "math", "general"]
CKPT_DIR = "data/verify_v3"
device = "cpu"

# ── Load shared embedding ──
print("Loading shared embedding...")
shared_embedding = nn.Embedding(256000, 512)
w = torch.load(f"{CKPT_DIR}/shared_embedding.pt", map_location="cpu", weights_only=True)
shared_embedding.weight.data.copy_(w)

# ── Load neurons ──
print("Loading neurons...")
neurons = {}
for d in DOMAINS:
    ckpt = torch.load(f"{CKPT_DIR}/neuron_{d}.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["neuron_config"]
    neuron = ResonanceNeuron(cfg)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    neuron.train()
    neurons[d] = neuron
    print(f"  {d}: hidden={cfg.hidden_size}, vocab={cfg.vocab_size}")

# ── Load tokenizers and data ──
print("Loading tokenizers...")
domain_sps = {d: load_domain_tokenizer(d) for d in DOMAINS}
general_sp = spm.SentencePieceProcessor()
general_sp.Load("taiji/domains/general/sp_general.model")

print("Loading texts...")
texts = {d: load_domain_texts(d, max_texts=2000) for d in DOMAINS}

# ── Joint training ──
print(f"\n{'='*50}")
print("5-Domain Joint Training (300 steps)")
print("=" * 50)

params = list(shared_embedding.parameters())
for n in neurons.values():
    params.extend(n.parameters())
optimizer = torch.optim.AdamW(params, lr=2e-4)
h_lm, h_field, h_intra, h_inter = [], [], [], []
t0 = time.time()

for step in range(1, 301):
    optimizer.zero_grad()
    all_fv, lm_losses = {}, []

    for domain in DOMAINS:
        neuron = neurons[domain]
        domain_sp = domain_sps[domain]
        train_texts = texts[domain]
        idx = torch.randint(0, len(train_texts), (3,))
        batch_texts = [train_texts[int(i)] for i in idx]

        shared_emb, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_embedding)
        targets = targets.clone(); targets[~mask] = -100

        result = neuron.forward(shared_emb, return_logits=True)
        logits = result["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_targets.view(-1), ignore_index=-100)
        lm_losses.append(lm_loss)
        all_fv[domain] = result["field_vector"]

    # Field contrastive loss
    nids = list(all_fv.keys()); N = len(nids)
    normed = {nid: v / (v.norm(dim=-1, keepdim=True) + 1e-8) for nid, v in all_fv.items()}

    inter_loss = torch.tensor(0.0); inter_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            inter_loss = inter_loss + F.relu(normed[nids[i]] @ normed[nids[j]].T - 0.5).pow(2).mean()
            inter_count += 1
    inter_loss = inter_loss / max(inter_count, 1)

    intra_loss = torch.tensor(0.0); intra_count = 0
    for nid in nids:
        v = normed[nid]; B = v.shape[0]
        if B < 2: continue
        sim = v @ v.T
        intra_loss = intra_loss + (sim * (1.0 - torch.eye(B))).abs().mean()
        intra_count += 1
    intra_loss = intra_loss / max(intra_count, 1)

    avg_lm = torch.stack(lm_losses).mean()
    total = avg_lm + 0.3 * (intra_loss + inter_loss)
    total.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()

    h_lm.append(avg_lm.item()); h_field.append((intra_loss + inter_loss).item())
    h_intra.append(intra_loss.item()); h_inter.append(inter_loss.item())

    if step % 100 == 0:
        w = min(step, 50)
        print(f"  step {step:>4d} lm={sum(h_lm[-w:])/w:.4f} "
              f"field={sum(h_field[-w:])/w:.4f} "
              f"intra={sum(h_intra[-w:])/w:.4f} "
              f"inter={sum(h_inter[-w:])/w:.4f}")

elapsed = time.time() - t0
final = {k: sum(v[-50:]) / 50 for k, v in
         [("lm", h_lm), ("field", h_field), ("intra", h_intra), ("inter", h_inter)]}
print(f"\n  Done ({elapsed:.0f}s)")
print(f"  lm={final['lm']:.4f} field={final['field']:.4f} "
      f"intra={final['intra']:.4f} inter={final['inter']:.4f}")

# Pairwise cosine
print("\n  Pairwise field cosine:")
with torch.no_grad():
    fvs = {}
    for d in DOMAINS:
        shared_emb, _, _ = batch_align_and_embed([texts[d][0]], domain_sps[d], general_sp, shared_embedding)
        fvs[d] = neurons[d].forward(shared_emb)["field_vector"]
    for i, d1 in enumerate(DOMAINS):
        for d2 in DOMAINS[i + 1:]:
            cos = (fvs[d1] @ fvs[d2].T).mean().item()
            print(f"    cos({d1:>7}, {d2:>7}) = {cos:.4f}")

# Save
for d, n in neurons.items():
    torch.save({"neuron_config": n.config, "state_dict": n.state_dict(), "domain": d},
               f"{CKPT_DIR}/neuron_{d}_joint.pt")
torch.save(shared_embedding.weight.data, f"{CKPT_DIR}/shared_embedding_joint.pt")

# ── Generation test ──
print(f"\n{'='*50}")
print("E2E Generation Test")
print("=" * 50)

# Override neurons with joint-trained versions
for d in DOMAINS:
    neurons[d].eval()

cortex = Cortex(neurons_dir=CKPT_DIR, shared_embedding=shared_embedding,
                general_tokenizer=general_sp, max_rounds=3)
cortex.neurons = neurons
cortex.is_loaded = True

hub = TokenizerHub()
for d in DOMAINS:
    hub.register_domain(d, domain_sps[d])
hub.register_domain("general", domain_sps["general"])
cortex.set_tokenizer_hub(hub)

tests = [
    ("你好，请介绍一下深度学习。", "zh"),
    ("What is the transformer architecture?", "en"),
    ("def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ", "code"),
    ("The derivative of x^2 is", "math"),
    ("Artificial intelligence is", "general"),
]

for prompt, domain in tests:
    print(f"\n  [{domain}] \"{prompt[:60]}\"")
    try:
        output = cortex.generate(prompt, max_tokens=40, temperature=0.7, domain=domain)
        print(f"    -> \"{output[:120]}\"")
    except Exception as e:
        print(f"    -> ERROR: {e}")

print(f"\n{'='*50}")
print("Complete!")
print("=" * 50)
