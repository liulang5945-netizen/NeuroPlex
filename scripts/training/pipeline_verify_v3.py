"""V3 验证 pipeline：tokenizer 生成 → 单神经元训练 → 联合训练 → 场损失验证。

Steps:
1. Download texts from HuggingFace (zh + en, 2000 each)
2. Train domain tokenizers (zh=20K, en=16K)
3. Train general tokenizer (256K from combined texts)
4. Train zh + en neurons with shared embedding (300 steps each)
5. Joint train (200 steps) — verify field_loss != 0

Usage:
    python scripts/training/pipeline_verify_v3.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed

# ── Config ──────────────────────────────────────────────────────────────────
DOMAINS = ["zh", "en"]
DOMAIN_VOCAB_SIZES = {"zh": 20000, "en": 16000}
GENERAL_VOCAB_SIZE = 256000
SHARED_EMBED_DIM = 512
OUTPUT_DIR = "data/verify_v3"
TOKENIZER_DIR = "taiji/domains"
DATA_DIR = "data/distill"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Download texts
# ═══════════════════════════════════════════════════════════════════════════

def download_texts(max_texts: int = 2000) -> dict:
    """Download texts for zh and en from HuggingFace."""
    print("=" * 50)
    print("Step 1: Downloading texts")
    print("=" * 50)

    DOMAIN_DATA_SOURCES = {
        "zh": [
            {"dataset": "shibing624/alpaca-zh", "config": "default", "split": "train",
             "text_fields": ["instruction", "input", "output"], "max_samples": 3000},
        ],
        "en": [
            {"dataset": "tatsu-lab/alpaca", "config": "default", "split": "train",
             "text_fields": ["instruction", "input", "output"], "max_samples": 3000},
        ],
    }

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("pip install datasets")

    all_texts = {}
    for domain in DOMAINS:
        texts = []
        for src in DOMAIN_DATA_SOURCES[domain]:
            print(f"  Downloading {src['dataset']}...")
            try:
                ds = load_dataset(src["dataset"], src["config"], split=src["split"])
                for example in ds:
                    if len(texts) >= max_texts:
                        break
                    parts = []
                    for field in src["text_fields"]:
                        val = example.get(field, "")
                        if isinstance(val, str) and val.strip():
                            parts.append(val.strip())
                    if parts:
                        texts.append(" ".join(parts))
            except Exception as e:
                print(f"  WARN: {e}")

        all_texts[domain] = texts
        print(f"  {domain}: {len(texts)} texts")
    return all_texts


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Train domain tokenizers
# ═══════════════════════════════════════════════════════════════════════════

def train_domain_tokenizer(domain: str, texts: list[str], vocab_size: int) -> spm.SentencePieceProcessor:
    """Train a domain-specific SentencePiece BPE tokenizer."""
    model_prefix = os.path.join(TOKENIZER_DIR, domain, f"sp_{domain}")
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ").replace("\r", "") + "\n")
        corpus_path = f.name

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        split_digits=True,
        max_sentence_length=16384,
        num_threads=4,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )

    os.unlink(corpus_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    # Verify
    actual = sp.vocab_size()
    print(f"  {domain}: vocab={actual}/{vocab_size}")
    test_text = texts[0][:80] if texts else "test"
    ids = sp.encode(test_text)
    print(f"    '{test_text}' → {len(ids)} tokens")
    return sp


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Train general tokenizer
# ═══════════════════════════════════════════════════════════════════════════

def train_general_tokenizer(all_texts: dict, vocab_size: int = 256000) -> spm.SentencePieceProcessor:
    """Train a 256K general tokenizer from combined domain texts."""
    print("=" * 50)
    print("Step 3: Training general 256K tokenizer")
    print("=" * 50)

    # Combine all texts
    combined = []
    for domain in DOMAINS:
        combined.extend(all_texts[domain])
    print(f"  Combined: {len(combined)} texts")

    model_prefix = os.path.join(TOKENIZER_DIR, "general", "sp_general")
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for t in combined:
            f.write(t.replace("\n", " ").replace("\r", "") + "\n")
        corpus_path = f.name

    print(f"  Training BPE {vocab_size}...")
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        split_digits=True,
        max_sentence_length=16384,
        num_threads=4,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.0f}s")

    os.unlink(corpus_path)
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    print(f"  General tokenizer: vocab={sp.vocab_size()}")
    return sp


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Train neurons with shared embedding
# ═══════════════════════════════════════════════════════════════════════════

def train_neurons(
    texts: dict,
    domain_sps: dict,
    general_sp: spm.SentencePieceProcessor,
    shared_embedding: nn.Embedding,
    steps: int = 300,
    batch_size: int = 4,
    lr: float = 5e-4,
    device: str = "cpu",
) -> dict:
    """Train zh + en neurons with shared embedding."""
    print("=" * 50)
    print(f"Step 4: Training neurons ({steps} steps each)")
    print("=" * 50)

    neurons = {}
    for domain in DOMAINS:
        print(f"\n  [{domain}] Creating neuron...")
        cfg = get_domain_neuron_config(domain, "compact")
        neuron = ResonanceNeuron(cfg).to(device)
        neuron.train()

        domain_sp = domain_sps[domain]
        train_texts = texts[domain]
        n_texts = len(train_texts)

        params = list(neuron.parameters()) + list(shared_embedding.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)

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
                shift_targets.view(-1),
                ignore_index=-100,
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
                print(f"    step {step:>4d} loss={loss.item():.4f} avg_loss={avg:.4f} PPL={ppl:.1f} ({elapsed:.0f}s)")

        avg_loss = total_loss / steps
        ppl = math.exp(min(avg_loss, 20))
        print(f"  [{domain}] Done: avg_loss={avg_loss:.4f} PPL={ppl:.1f}")

        # Save
        save_path = os.path.join(OUTPUT_DIR, f"neuron_{domain}.pt")
        torch.save({
            "neuron_config": neuron.config,
            "state_dict": neuron.state_dict(),
            "domain": domain,
            "result": {"final_loss": avg_loss, "final_ppl": ppl, "steps": steps},
        }, save_path)

        neurons[domain] = neuron

    # Save shared embedding
    emb_path = os.path.join(OUTPUT_DIR, "shared_embedding.pt")
    torch.save(shared_embedding.weight.data, emb_path)
    print(f"\n  Shared embedding saved: {emb_path}")

    return neurons


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Joint training — verify field contrastive loss
# ═══════════════════════════════════════════════════════════════════════════

def field_contrastive_loss(all_field_vectors: dict, margin: float = 0.5) -> dict:
    """Field contrastive loss (inter + intra)."""
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
            v_i = normed[nids[i]]
            v_j = normed[nids[j]]
            sim = v_i @ v_j.T
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


def joint_verify(
    neurons: dict,
    texts: dict,
    domain_sps: dict,
    general_sp,
    shared_embedding: nn.Embedding,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 2e-4,
    device: str = "cpu",
) -> dict:
    """Joint training v3: verify field contrastive loss is non-zero."""
    print("=" * 50)
    print(f"Step 5: Joint training ({steps} steps) — verifying field_loss != 0")
    print("=" * 50)

    params = list(shared_embedding.parameters())
    for n in neurons.values():
        params.extend(n.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    history = {"lm": [], "field": [], "intra": [], "inter": []}

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        all_field_vectors = {}
        lm_losses = []

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
                shift_targets.view(-1),
                ignore_index=-100,
            )
            lm_losses.append(lm_loss)
            all_field_vectors[domain] = result["field_vector"]

        field_losses = field_contrastive_loss(all_field_vectors, margin=0.5)
        avg_lm = torch.stack(lm_losses).mean()
        total = avg_lm + 0.5 * field_losses["total"]

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        history["lm"].append(avg_lm.item())
        history["field"].append(field_losses["total"].item())
        history["intra"].append(field_losses["intra"].item())
        history["inter"].append(field_losses["inter"].item())

        if step % 50 == 0:
            r_lm = sum(history["lm"][-50:]) / 50
            r_field = sum(history["field"][-50:]) / 50
            r_intra = sum(history["intra"][-50:]) / 50
            r_inter = sum(history["inter"][-50:]) / 50
            print(f"  step {step:>4d} lm={r_lm:.4f} field={r_field:.4f} intra={r_intra:.4f} inter={r_inter:.4f}")

    # Final stats
    final_field = sum(history["field"][-50:]) / 50
    final_intra = sum(history["intra"][-50:]) / 50
    final_inter = sum(history["inter"][-50:]) / 50

    print(f"\n  Final field_loss: {final_field:.4f}")
    print(f"    intra (same-neuron diversity): {final_intra:.4f}")
    print(f"    inter (cross-neuron separation): {final_inter:.4f}")

    # Verification
    passed = final_field > 0.001
    if passed:
        print(f"\n  *** VERIFIED: field_loss = {final_field:.4f} > 0 ***")
    else:
        print(f"\n  *** WARNING: field_loss = {final_field:.4f} ≈ 0 ***")

    # Check field vector cosine
    print("\n  Field vector cosine similarity (zh vs en):")
    with torch.no_grad():
        shared_emb, _, _ = batch_align_and_embed(
            [texts["zh"][0]], domain_sps["zh"], general_sp, shared_embedding,
        )
        fv_zh = neurons["zh"].forward(shared_emb)["field_vector"]
        shared_emb, _, _ = batch_align_and_embed(
            [texts["en"][0]], domain_sps["en"], general_sp, shared_embedding,
        )
        fv_en = neurons["en"].forward(shared_emb)["field_vector"]
        cos_sim = (fv_zh @ fv_en.T).mean().item()
        print(f"    cos(zh, en) = {cos_sim:.4f} {'(low — good separation)' if abs(cos_sim) < 0.5 else '(high — may need more training)'}")

    return {"passed": passed, "final_field": final_field, "cos_zh_en": cos_sim}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = "cpu"
    print("V3 Verification Pipeline")
    print(f"  Domains: {DOMAINS}")
    print(f"  Device: {device}")
    print()

    # Step 1: Download texts
    all_texts = download_texts(2000)

    # Step 2: Train domain tokenizers
    print(f"\n{'='*50}")
    print("Step 2: Training domain tokenizers")
    print("=" * 50)
    domain_sps = {}
    for domain in DOMAINS:
        domain_sps[domain] = train_domain_tokenizer(
            domain, all_texts[domain], DOMAIN_VOCAB_SIZES[domain],
        )

    # Step 3: Train general tokenizer
    general_sp = train_general_tokenizer(all_texts, GENERAL_VOCAB_SIZE)

    # Step 4: Create shared embedding + train neurons
    shared_embedding = nn.Embedding(GENERAL_VOCAB_SIZE, SHARED_EMBED_DIM)
    print(f"\n  Shared embedding: {GENERAL_VOCAB_SIZE} x {SHARED_EMBED_DIM}")
    neurons = train_neurons(
        all_texts, domain_sps, general_sp, shared_embedding,
        steps=300, device=device,
    )

    # Step 5: Joint training verification
    result = joint_verify(
        neurons, all_texts, domain_sps, general_sp, shared_embedding,
        steps=200, device=device,
    )

    # Summary
    print(f"\n{'='*50}")
    print("Pipeline complete!")
    print(f"  Tokenizers: {TOKENIZER_DIR}/")
    print(f"  Neurons + shared_embedding: {OUTPUT_DIR}/")
    print(f"  Field contrastive loss working: {result['passed']}")
    print(f"  cos(zh, en): {result['cos_zh_en']:.4f}")
    print("=" * 50)

    return result


if __name__ == "__main__":
    main()
