"""V3 Full Pipeline: 5-domain neuron training → joint training → generation.

Prerequisites: tokenizers exist at taiji/domains/ (already built).
Steps:
  1. Load/download texts for all 5 domains
  2. Load zh/en neurons from verify_v3, train code/math/general from scratch
  3. 5-domain joint training with field contrastive loss
  4. End-to-end generation via Cortex
"""

import math, os, sys, time, functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

print = functools.partial(print, flush=True)

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import get_domain_neuron_config, DOMAIN_VOCAB_SIZES
from taiji.resonance.translator import batch_align_and_embed
from taiji.brain.cortex import Cortex
from taiji.resonance.translator import TokenizerHub
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_domain_texts,
    DATA_DIR, OUTPUT_DIR as NEURONS_DIR,
)

DOMAINS = ["zh", "en", "code", "math", "general"]
TOKENIZER_DIR = "taiji/domains"
OUTPUT_DIR = "data/verify_v3_full"
GENERAL_VOCAB_SIZE = 256000
SHARED_EMBED_DIM = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all_texts(max_texts=2000):
    print("Loading texts for all domains...")
    texts = {}
    for d in DOMAINS:
        texts[d] = load_domain_texts(d, max_texts=max_texts)
        print(f"  {d}: {len(texts[d])} texts")
    return texts


# ═══════════════════════════════════════════════════════════════════════════
# Load tokenizers
# ═══════════════════════════════════════════════════════════════════════════

def load_all_tokenizers():
    domain_sps = {}
    for d in DOMAINS:
        sp = load_domain_tokenizer(d)
        domain_sps[d] = sp
        print(f"  {d}: vocab={sp.vocab_size()}")

    path = os.path.join(TOKENIZER_DIR, "general", "sp_general.model")
    general_sp = spm.SentencePieceProcessor()
    general_sp.Load(path)
    print(f"  general: vocab={general_sp.vocab_size()}")
    return domain_sps, general_sp


# ═══════════════════════════════════════════════════════════════════════════
# Train one neuron
# ═══════════════════════════════════════════════════════════════════════════

def train_neuron(domain, texts, domain_sp, general_sp, shared_embedding,
                 steps=300, batch_size=4, lr=5e-4, device="cpu"):
    cfg = get_domain_neuron_config(domain, "compact")
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.train()

    train_texts = texts[domain]
    n_texts = len(train_texts)
    params = list(neuron.parameters()) + list(shared_embedding.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_loss, t0 = 0.0, time.time()

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_texts, (batch_size,))
        batch_texts = [train_texts[int(i)] for i in idx]
        shared_emb, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_embedding)
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

        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_targets.view(-1), ignore_index=-100)
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
    torch.save({"neuron_config": neuron.config, "state_dict": neuron.state_dict(),
                "domain": domain}, save_path)
    return neuron


# ═══════════════════════════════════════════════════════════════════════════
# Joint training (5 domains)
# ═══════════════════════════════════════════════════════════════════════════

def field_contrastive_loss(all_fv, margin=0.5):
    nids = list(all_fv.keys())
    N = len(nids)
    if N < 2:
        return {"intra": torch.tensor(0.0), "inter": torch.tensor(0.0), "total": torch.tensor(0.0)}
    normed = {nid: v / (v.norm(dim=-1, keepdim=True) + 1e-8) if v.dim() == 2
              else v.unsqueeze(0) / (v.unsqueeze(0).norm(dim=-1, keepdim=True) + 1e-8)
              for nid, v in all_fv.items()}

    inter_loss = torch.tensor(0.0); inter_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            sim = normed[nids[i]] @ normed[nids[j]].T
            inter_loss = inter_loss + F.relu(sim - margin).pow(2).mean()
            inter_count += 1
    inter_loss = inter_loss / max(inter_count, 1)

    intra_loss = torch.tensor(0.0); intra_count = 0
    for nid in nids:
        v = normed[nid]; B = v.shape[0]
        if B < 2: continue
        sim = v @ v.T
        mask = 1.0 - torch.eye(B, device=v.device)
        intra_loss = intra_loss + (sim * mask).abs().mean()
        intra_count += 1
    intra_loss = intra_loss / max(intra_count, 1)
    return {"intra": intra_loss, "inter": inter_loss, "total": intra_loss + inter_loss}


def joint_train_all(neurons, texts, domain_sps, general_sp, shared_embedding,
                    steps=300, batch_size=3, device="cpu"):
    print(f"\n5-Domain Joint Training ({steps} steps)...")
    active = [d for d in DOMAINS if d in neurons and d in texts]

    params = list(shared_embedding.parameters())
    for n in neurons.values():
        params.extend(n.parameters())
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    h_lm, h_field, h_intra, h_inter = [], [], [], []
    t0 = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        all_fv, lm_losses = {}, []

        for domain in active:
            neuron = neurons[domain]
            domain_sp = domain_sps[domain]
            train_texts = texts[domain]
            idx = torch.randint(0, len(train_texts), (batch_size,))
            batch_texts = [train_texts[int(i)] for i in idx]

            shared_emb, targets, mask = batch_align_and_embed(
                batch_texts, domain_sp, general_sp, shared_embedding)
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

            lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                      shift_targets.view(-1), ignore_index=-100)
            lm_losses.append(lm_loss)
            all_fv[domain] = result["field_vector"]

        fl = field_contrastive_loss(all_fv)
        avg_lm = torch.stack(lm_losses).mean()
        total = avg_lm + 0.3 * fl["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        h_lm.append(avg_lm.item()); h_field.append(fl["total"].item())
        h_intra.append(fl["intra"].item()); h_inter.append(fl["inter"].item())

        if step % 100 == 0:
            w = min(step, 50)
            print(f"  step {step:>4d} lm={sum(h_lm[-w:])/w:.4f} "
                  f"field={sum(h_field[-w:])/w:.4f} "
                  f"intra={sum(h_intra[-w:])/w:.4f} "
                  f"inter={sum(h_inter[-w:])/w:.4f}")

    final = {k: sum(v[-50:]) / 50 for k, v in
             [("lm", h_lm), ("field", h_field), ("intra", h_intra), ("inter", h_inter)]}
    elapsed = time.time() - t0
    print(f"\n  Joint training done ({elapsed:.0f}s)")
    print(f"  lm={final['lm']:.4f} field={final['field']:.4f} "
          f"intra={final['intra']:.4f} inter={final['inter']:.4f}")

    # Pairwise cosine matrix
    print("\n  Pairwise field cosine:")
    with torch.no_grad():
        fvs = {}
        for d in active:
            shared_emb, _, _ = batch_align_and_embed(
                [texts[d][0]], domain_sps[d], general_sp, shared_embedding)
            fvs[d] = neurons[d].forward(shared_emb)["field_vector"]
        for i, d1 in enumerate(active):
            for d2 in active[i + 1:]:
                cos = (fvs[d1] @ fvs[d2].T).mean().item()
                print(f"    cos({d1}, {d2}) = {cos:.4f}")

    # Save all
    for d, n in neurons.items():
        torch.save({"neuron_config": n.config, "state_dict": n.state_dict(), "domain": d},
                   os.path.join(OUTPUT_DIR, f"neuron_{d}_joint.pt"))
    torch.save(shared_embedding.weight.data, os.path.join(OUTPUT_DIR, "shared_embedding_joint.pt"))
    print(f"\n  Saved to {OUTPUT_DIR}/")
    return final


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end generation test
# ═══════════════════════════════════════════════════════════════════════════

def test_generation(neurons, shared_embedding, general_sp, domain_sps):
    print(f"\n{'='*50}")
    print("E2E Generation Test")
    print("=" * 50)

    # Setup cortex
    cortex = Cortex(
        neurons_dir=OUTPUT_DIR,
        shared_embedding=shared_embedding,
        general_tokenizer=general_sp,
        max_rounds=3,
    )

    # Override neurons with our trained ones
    cortex.neurons = neurons
    cortex.is_loaded = True

    # Setup tokenizer hub
    hub = TokenizerHub()
    for d in DOMAINS:
        hub.register_domain(d, domain_sps[d])
    hub.register_domain("general", domain_sps["general"])
    cortex.set_tokenizer_hub(hub)

    print(f"  Neurons: {list(neurons.keys())}")
    print(f"  Domains: {hub.list_domains()}")

    # Test prompts
    tests = [
        ("你好，请介绍一下人工智能。", "zh"),
        ("What is machine learning?", "en"),
        ("def fibonacci(n):\n    ", "code"),
        ("Solve: 3x + 5 = 20", "math"),
        ("Explain how neural networks work.", "general"),
    ]

    for prompt, domain in tests:
        print(f"\n  [{domain}] Prompt: \"{prompt[:50]}...\"")
        try:
            output = cortex.generate(prompt, max_tokens=30, temperature=0.7, domain=domain)
            output_clean = output[:100].replace("\n", "\\n")
            print(f"  -> \"{output_clean}\"")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print(f"\n{'='*50}")
    print("Generation test complete")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = "cpu"
    print("V3 Full Pipeline: 5-domain neurons + joint training + generation")
    print("=" * 60)

    # 1. Data
    print("\n[1/4] Loading texts...")
    texts = load_all_texts(3000)

    # 2. Tokenizers
    print("\n[2/4] Loading tokenizers...")
    domain_sps, general_sp = load_all_tokenizers()

    # 3. Shared embedding
    shared_embedding = nn.Embedding(GENERAL_VOCAB_SIZE, SHARED_EMBED_DIM)
    # Try loading existing
    existing_path = os.path.join("data/verify_v3", "shared_embedding.pt")
    if os.path.exists(existing_path):
        weight = torch.load(existing_path, map_location="cpu", weights_only=True)
        shared_embedding.weight.data.copy_(weight)
        print(f"\n  Loaded existing shared_embedding from {existing_path}")
    print(f"  Shared embedding: {GENERAL_VOCAB_SIZE} x {SHARED_EMBED_DIM}")

    # 4. Train neurons
    print("\n[3/4] Training neurons (300 steps each)...")
    neurons = {}

    for domain in DOMAINS:
        existing = os.path.join("data/verify_v3", f"neuron_{domain}.pt")
        if os.path.exists(existing):
            ckpt = torch.load(existing, map_location="cpu", weights_only=False)
            cfg = ckpt["neuron_config"]
            neuron = ResonanceNeuron(cfg)
            neuron.load_state_dict(ckpt["state_dict"], strict=False)
            neuron.to(device)
            neuron.train()
            neurons[domain] = neuron
            print(f"  [{domain}] Loaded from {existing} (skipping training)")
            continue

        print(f"\n  [{domain}] Training new neuron...")
        neuron = train_neuron(domain, texts, domain_sps[domain], general_sp,
                              shared_embedding, steps=300, device=device)
        neurons[domain] = neuron

    # 5. Joint training
    print("\n[4/4] Joint training...")
    joint_train_all(neurons, texts, domain_sps, general_sp,
                    shared_embedding, steps=300, device=device)

    # E2E generation test
    test_generation(neurons, shared_embedding, general_sp, domain_sps)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
