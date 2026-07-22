"""V3 Quick Verification — skip tokenizer creation, train + joint verify.
Tokenizers already exist at taiji/domains/.
"""

import math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed

# Unbuffered output
import functools
print = functools.partial(print, flush=True)

DOMAINS = ["zh", "en"]
TOKENIZER_DIR = "taiji/domains"
OUTPUT_DIR = "data/verify_v3"
DATA_DIR = "data/distill"
GENERAL_VOCAB_SIZE = 256000
SHARED_EMBED_DIM = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tokenizers():
    domain_sps = {}
    for domain in DOMAINS:
        path = os.path.join(TOKENIZER_DIR, domain, f"sp_{domain}.model")
        sp = spm.SentencePieceProcessor()
        sp.Load(path)
        domain_sps[domain] = sp
        print(f"Loaded {domain} tokenizer: vocab={sp.vocab_size()}")

    path = os.path.join(TOKENIZER_DIR, "general", "sp_general.model")
    general_sp = spm.SentencePieceProcessor()
    general_sp.Load(path)
    print(f"Loaded general tokenizer: vocab={general_sp.vocab_size()}")
    return domain_sps, general_sp


def load_texts(domain: str, max_texts: int = 2000) -> list:
    cache_path = os.path.join(DATA_DIR, f"{domain}_texts.jsonl")
    texts = []
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_texts:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
    if texts:
        print(f"  {domain}: {len(texts)} texts from cache")
        return texts

    # Download
    sources = {
        "zh": [{"dataset": "shibing624/alpaca-zh", "config": "default", "split": "train",
                 "text_fields": ["instruction", "input", "output"], "max_samples": 3000}],
        "en": [{"dataset": "tatsu-lab/alpaca", "config": "default", "split": "train",
                 "text_fields": ["instruction", "input", "output"], "max_samples": 3000}],
    }
    from datasets import load_dataset
    for src in sources[domain]:
        ds = load_dataset(src["dataset"], src["config"], split=src["split"])
        for example in ds:
            if len(texts) >= max_texts:
                break
            parts = [example.get(f, "").strip() for f in src["text_fields"] if example.get(f, "").strip()]
            if parts:
                texts.append(" ".join(parts))
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    print(f"  {domain}: {len(texts)} texts downloaded")
    return texts


def train_neurons(texts, domain_sps, general_sp, shared_embedding, steps=300, batch_size=4, device="cpu"):
    print(f"\nTraining {DOMAINS} neurons ({steps} steps each)...")
    neurons = {}
    
    for domain in DOMAINS:
        cfg = get_domain_neuron_config(domain, "compact")
        neuron = ResonanceNeuron(cfg).to(device)
        neuron.train()
        
        domain_sp = domain_sps[domain]
        train_texts = texts[domain]
        n_texts = len(train_texts)
        
        params = list(neuron.parameters()) + list(shared_embedding.parameters())
        optimizer = torch.optim.AdamW(params, lr=5e-4)
        
        total_loss = 0.0
        t0 = time.time()
        
        for step in range(1, steps + 1):
            idx = torch.randint(0, n_texts, (batch_size,))
            batch_texts = [train_texts[int(i)] for i in idx]
            
            shared_emb, targets, mask = batch_align_and_embed(
                batch_texts, domain_sp, general_sp, shared_embedding,
            )
            shared_emb = shared_emb.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            
            result = neuron.forward(shared_emb, return_logits=True)
            logits = result["logits"]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1), ignore_index=-100,
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                avg = total_loss / step
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - t0
                print(f"  [{domain}] step {step:>4d} loss={loss.item():.4f} avg={avg:.4f} PPL={ppl:.1f} ({elapsed:.0f}s)")
        
        avg_loss = total_loss / steps
        ppl = math.exp(min(avg_loss, 20))
        print(f"  [{domain}] DONE: avg_loss={avg_loss:.4f} PPL={ppl:.1f}")
        
        save_path = os.path.join(OUTPUT_DIR, f"neuron_{domain}.pt")
        torch.save({"neuron_config": neuron.config, "state_dict": neuron.state_dict(), "domain": domain}, save_path)
        neurons[domain] = neuron
    
    emb_path = os.path.join(OUTPUT_DIR, "shared_embedding.pt")
    torch.save(shared_embedding.weight.data, emb_path)
    print(f"  Shared embedding saved: {emb_path}")
    return neurons


def field_contrastive_loss(all_field_vectors, margin=0.5):
    nids = list(all_field_vectors.keys())
    N = len(nids)
    if N < 2:
        return {"intra": torch.tensor(0.0), "inter": torch.tensor(0.0), "total": torch.tensor(0.0)}
    
    normed = {}
    for nid, vecs in all_field_vectors.items():
        v = vecs if vecs.dim() == 2 else vecs.unsqueeze(0)
        normed[nid] = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    
    inter_loss = torch.tensor(0.0)
    inter_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            sim = normed[nids[i]] @ normed[nids[j]].T
            inter_loss = inter_loss + F.relu(sim - margin).pow(2).mean()
            inter_count += 1
    inter_loss = inter_loss / max(inter_count, 1)
    
    intra_loss = torch.tensor(0.0)
    intra_count = 0
    for nid in nids:
        v = normed[nid]
        B = v.shape[0]
        if B < 2:
            continue
        sim = v @ v.T
        mask = 1.0 - torch.eye(B, device=v.device)
        intra_loss = intra_loss + (sim * mask).abs().mean()
        intra_count += 1
    intra_loss = intra_loss / max(intra_count, 1)
    
    return {"intra": intra_loss, "inter": inter_loss, "total": intra_loss + inter_loss}


def joint_verify(neurons, texts, domain_sps, general_sp, shared_embedding, steps=200, batch_size=4, device="cpu"):
    print(f"\nJoint training ({steps} steps) — verifying field_loss != 0...")
    
    params = list(shared_embedding.parameters())
    for n in neurons.values():
        params.extend(n.parameters())
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    
    h_lm, h_field, h_intra, h_inter = [], [], [], []
    t0 = time.time()
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        all_fv, lm_losses = {}, []
        
        for domain in DOMAINS:
            neuron = neurons[domain]
            domain_sp = domain_sps[domain]
            train_texts = texts[domain]
            
            idx = torch.randint(0, len(train_texts), (batch_size,))
            batch_texts = [train_texts[int(i)] for i in idx]
            
            shared_emb, targets, mask = batch_align_and_embed(
                batch_texts, domain_sp, general_sp, shared_embedding,
            )
            shared_emb = shared_emb.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            
            result = neuron.forward(shared_emb, return_logits=True)
            logits = result["logits"]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~shift_mask] = -100
            
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1), ignore_index=-100,
            )
            lm_losses.append(lm_loss)
            all_fv[domain] = result["field_vector"]
        
        fl = field_contrastive_loss(all_fv, margin=0.5)
        avg_lm = torch.stack(lm_losses).mean()
        total = avg_lm + 0.5 * fl["total"]
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        
        h_lm.append(avg_lm.item())
        h_field.append(fl["total"].item())
        h_intra.append(fl["intra"].item())
        h_inter.append(fl["inter"].item())
        
        if step % 50 == 0:
            r_lm = sum(h_lm[-50:]) / 50
            r_field = sum(h_field[-50:]) / 50
            r_intra = sum(h_intra[-50:]) / 50
            r_inter = sum(h_inter[-50:]) / 50
            print(f"  step {step:>4d} lm={r_lm:.4f} field={r_field:.4f} intra={r_intra:.4f} inter={r_inter:.4f}")
    
    final_field = sum(h_field[-50:]) / 50
    final_intra = sum(h_intra[-50:]) / 50
    final_inter = sum(h_inter[-50:]) / 50
    elapsed = time.time() - t0
    
    print(f"\n  Training done ({elapsed:.0f}s)")
    print(f"  Final field_loss: {final_field:.4f}")
    print(f"    intra (diversity): {final_intra:.4f}")
    print(f"    inter (separation): {final_inter:.4f}")
    
    # Cosine check
    print("\n  Field vector cosine (zh vs en):")
    with torch.no_grad():
        for domain in DOMAINS:
            shared_emb, _, _ = batch_align_and_embed(
                [texts[domain][0]], domain_sps[domain], general_sp, shared_embedding,
            )
            fv = neurons[domain].forward(shared_emb)["field_vector"]
            norm = fv.norm(dim=-1).item()
            print(f"    {domain}: |fv| = {norm:.4f}")
        
        # Cross-domain cosine on first sample
        shared_emb_zh, _, _ = batch_align_and_embed(
            [texts["zh"][0]], domain_sps["zh"], general_sp, shared_embedding,
        )
        fv_zh = neurons["zh"].forward(shared_emb_zh)["field_vector"]
        shared_emb_en, _, _ = batch_align_and_embed(
            [texts["en"][0]], domain_sps["en"], general_sp, shared_embedding,
        )
        fv_en = neurons["en"].forward(shared_emb_en)["field_vector"]
        cos = (fv_zh @ fv_en.T).mean().item()
        status = "GOOD (separated)" if abs(cos) < 0.5 else "HIGH (similar)"
        print(f"    cos(zh, en) = {cos:.4f} — {status}")
    
    passed = final_field > 0.001
    if passed:
        print(f"\n  *** VERIFIED: field_loss = {final_field:.4f} > 0 ***")
    else:
        print(f"\n  *** WARNING: field_loss ≈ 0 ***")
    
    return {"passed": passed, "final_field": final_field, "cos_zh_en": cos}


def main():
    device = "cpu"
    print("V3 Quick Verification")
    print("=" * 40)
    
    # Load tokenizers
    domain_sps, general_sp = load_tokenizers()
    
    # Load texts
    print("\nLoading texts...")
    texts = {d: load_texts(d) for d in DOMAINS}
    
    # Shared embedding
    shared_embedding = nn.Embedding(GENERAL_VOCAB_SIZE, SHARED_EMBED_DIM)
    print(f"\nShared embedding: {GENERAL_VOCAB_SIZE} x {SHARED_EMBED_DIM}")
    
    # Train neurons
    neurons = train_neurons(texts, domain_sps, general_sp, shared_embedding, steps=300, device=device)
    
    # Joint verify
    result = joint_verify(neurons, texts, domain_sps, general_sp, shared_embedding, steps=200, device=device)
    
    print(f"\n{'='*40}")
    print(f"DONE: passed={result['passed']}, cos(zh,en)={result['cos_zh_en']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
