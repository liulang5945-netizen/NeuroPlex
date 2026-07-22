"""P7: From-scratch neuron training with shared embedding.

三层架构：
- Layer 1 (共享感官): nn.Embedding(256000, 512)，所有 neuron 共用
- Layer 2 (认知空间): per-neuron embed_adapter + Transformer body
- Layer 3 (神经语言): 4096-dim 共振场

TokenTranslator 桥接域 tokenizer → 通用 token → 共享嵌入。
通用词表可热插拔，不影响 neuron 内部。

Data: downloads raw text from HuggingFace datasets for each domain.
At training time, raw texts are aligned via batch_align_and_embed:
domain tokenizer for targets, general tokenizer for shared embedding.

Usage:
    # Train a single zh neuron from scratch
    python scripts/training/train_neuron.py --domain zh --steps 2000

    # Train all 5 domains
    python scripts/training/train_neuron.py --all --steps 5000

    # Resume training
    python scripts/training/train_neuron.py --domain zh --resume data/neurons/neuron_zh.pt --steps 2000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config
from taiji.resonance.config import DOMAIN_VOCAB_SIZES, GENERAL_TOKENIZER_DOMAIN
from taiji.resonance.translator import batch_align_and_embed

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "data/distill"
OUTPUT_DIR = "data/neurons"
DOMAIN_TOKENIZER_DIR = "taiji/domains"
SHARED_EMBEDDING_PATH = "data/shared_embedding.pt"

# General vocab size (hot-swappable: can be upgraded to 512K or 1M without retraining neurons)
GENERAL_VOCAB_SIZE = 256000
SHARED_EMBED_DIM = 512

# ── HuggingFace data source mapping ────────────────────────────────────────
# Each domain can have multiple sources, concatenated for diversity.
# Format: list of {dataset, config, split, text_fields, max_samples}
DOMAIN_DATA_SOURCES = {
    "zh": [
        {"dataset": "shibing624/alpaca-zh", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
        {"dataset": "qingyue2019/wikipedia-zh-cn-20240820", "config": "default", "split": "train",
         "text_fields": ["text"], "max_samples": 5000},
    ],
    "en": [
        {"dataset": "tatsu-lab/alpaca", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
        {"dataset": "wikipedia", "config": "20220301.en", "split": "train",
         "text_fields": ["text"], "max_samples": 5000},
    ],
    "code": [
        {"dataset": "sahil2801/CodeAlpaca-20k", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
    ],
    "math": [
        {"dataset": "openai/gsm8k", "config": "main", "split": "train",
         "text_fields": ["question", "answer"], "max_samples": 8000},
    ],
    "general": [
        {"dataset": "tatsu-lab/alpaca", "config": "default", "split": "train",
         "text_fields": ["instruction", "input", "output"], "max_samples": 8000},
    ],
}


def load_domain_tokenizer(domain: str) -> spm.SentencePieceProcessor:
    """Load domain-specific SentencePiece tokenizer.

    general domain reuses en tokenizer.
    """
    if domain == "general":
        actual_domain = GENERAL_TOKENIZER_DOMAIN  # "en"
    else:
        actual_domain = domain

    model_path = os.path.join(DOMAIN_TOKENIZER_DIR, actual_domain, f"sp_{actual_domain}.model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Domain tokenizer not found: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def load_domain_texts(domain: str, max_texts: int = 5000) -> list[str]:
    """Load raw text data for a domain from multiple HuggingFace datasets.

    Strategy:
    1. Try cached text file (data/distill/{domain}_texts.jsonl)
    2. Download from each source in DOMAIN_DATA_SOURCES[domain]
    3. Concatenate text fields into single strings
    """
    cache_path = os.path.join(DATA_DIR, f"{domain}_texts.jsonl")

    # Strategy 1: cached text
    if os.path.exists(cache_path):
        texts = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_texts:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
        if texts:
            print(f"  Loaded {len(texts)} texts from cache: {cache_path}")
            return texts

    # Strategy 2: download from multiple HuggingFace sources
    sources = DOMAIN_DATA_SOURCES.get(domain)
    if not sources:
        raise ValueError(f"Unknown domain: {domain}")
    if isinstance(sources, dict):
        sources = [sources]  # backward compat: single dict format

    all_texts = []
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets library required: pip install datasets")

    for src in sources:
        src_max = src.get("max_samples", max_texts)
        remaining = max_texts - len(all_texts)
        if remaining <= 0:
            break
        limit = min(remaining, src_max)

        print(f"  Downloading {src['dataset']} (target {limit})...")
        try:
            ds = load_dataset(src["dataset"], src["config"], split=src["split"])
            for example in ds:
                if len(all_texts) >= max_texts:
                    break
                if len(all_texts) >= limit + (max_texts - remaining):
                    # Already got enough from this source
                    pass
                parts = []
                for field in src["text_fields"]:
                    val = example.get(field, "")
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                if parts:
                    all_texts.append(" ".join(parts))

                if len(all_texts) >= limit + (max_texts - remaining):
                    break
        except Exception as e:
            print(f"  WARN: {src['dataset']} failed: {e}, skipping")

    if all_texts:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for t in all_texts:
                f.write(t + "\n")
        print(f"  Downloaded {len(all_texts)} texts total, cached to {cache_path}")

    return all_texts


def load_general_tokenizer(general_model_path: str = None) -> spm.SentencePieceProcessor:
    """Load or create the general 256K tokenizer.

    Priority:
    1. Explicit path (general_model_path)
    2. taiji/domains/general/sp_general.model
    3. Fallback: use en tokenizer as general tokenizer (16K vocab)
       This is temporary; neurons trained this way can accept a larger
       tokenizer later (hot-swap) since shared_embedding has 256K entries.

    Returns:
        SentencePieceProcessor for the general tokenizer.
    """
    if general_model_path and os.path.exists(general_model_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(general_model_path)
        print(f"  General tokenizer loaded: {general_model_path} (vocab={sp.vocab_size()})")
        return sp

    # Try default general tokenizer
    general_path = os.path.join(DOMAIN_TOKENIZER_DIR, "general", "sp_general.model")
    if os.path.exists(general_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(general_path)
        print(f"  General tokenizer loaded: {general_path} (vocab={sp.vocab_size()})")
        return sp

    # Fallback: use en tokenizer as general tokenizer
    en_path = os.path.join(DOMAIN_TOKENIZER_DIR, "en", "sp_en.model")
    if os.path.exists(en_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(en_path)
        print(f"  General tokenizer: FALLBACK to en tokenizer (vocab={sp.vocab_size()}). "
              f"Upgrade to 256K tokenizer later for full vocab coverage.")
        return sp

    raise FileNotFoundError(
        "No general tokenizer found. Place sp_general.model in taiji/domains/general/ "
        "or ensure taiji/domains/en/sp_en.model exists as fallback."
    )


def create_shared_embedding(device: str = "cpu") -> nn.Embedding:
    """Create the shared embedding table (Layer 1: shared sensory).

    All neurons share this single Embedding(256000, 512) table.
    It can be saved/loaded independently and hot-swapped for larger vocabs.

    Args:
        device: torch device.

    Returns:
        nn.Embedding(256000, 512)
    """
    emb = nn.Embedding(GENERAL_VOCAB_SIZE, SHARED_EMBED_DIM)
    emb.to(device)
    print(f"  Shared embedding: {GENERAL_VOCAB_SIZE} × {SHARED_EMBED_DIM} "
          f"({GENERAL_VOCAB_SIZE * SHARED_EMBED_DIM / 1e6:.0f}M params)")
    return emb


def load_or_create_shared_embedding(device: str = "cpu") -> nn.Embedding:
    """Load existing shared embedding or create a new one.

    Args:
        device: torch device.

    Returns:
        nn.Embedding(256000, 512)
    """
    if os.path.exists(SHARED_EMBEDDING_PATH):
        state = torch.load(SHARED_EMBEDDING_PATH, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "weight" in state:
            weight = state["weight"]
        else:
            weight = state
        emb = nn.Embedding(weight.shape[0], weight.shape[1])
        emb.weight.data.copy_(weight)
        emb.to(device)
        print(f"  Loaded shared embedding from {SHARED_EMBEDDING_PATH}: "
              f"{emb.num_embeddings} × {emb.embedding_dim}")
        return emb

    return create_shared_embedding(device)


def save_shared_embedding(emb: nn.Embedding, path: str = None) -> None:
    """Save the shared embedding table for reuse.

    Args:
        emb: the shared nn.Embedding.
        path: save path (default: SHARED_EMBEDDING_PATH).
    """
    path = path or SHARED_EMBEDDING_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(emb.weight.data, path)
    print(f"  Shared embedding saved to {path}")


def tokenize_domain_texts(
    texts: list[str],
    domain_sp: spm.SentencePieceProcessor,
    max_len: int = 256,
) -> torch.Tensor:
    """Tokenize texts with domain-specific tokenizer.

    DEPRECATED in new shared-embedding architecture.
    Kept for backward compatibility with existing checkpoints.
    Use batch_align_and_embed() for new training.

    Returns:
        [N, max_len] long tensor of domain token IDs.
    """
    all_ids = []
    for text in texts:
        ids = domain_sp.encode(text)
        if len(ids) == 0:
            continue
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [domain_sp.pad_id()] * (max_len - len(ids))
        all_ids.append(ids)

    if not all_ids:
        raise ValueError(f"No valid texts after tokenization (max_len={max_len})")

    return torch.tensor(all_ids, dtype=torch.long)


def train_one_neuron(
    neuron: ResonanceNeuron,
    texts: list[str],
    domain: str,
    shared_embedding: nn.Embedding,
    domain_sp: spm.SentencePieceProcessor,
    general_sp: spm.SentencePieceProcessor,
    num_steps: int = 2000,
    batch_size: int = 4,
    lr: float = 5e-4,
    device: str = "cpu",
    log_every: int = 50,
    save_path: Optional[str] = None,
) -> dict:
    """Train a single neuron from scratch on domain data.

    使用共享嵌入表 + token 对齐：
    - 原始文本 → domain tokenizer (targets) + general tokenizer (shared emb)
    - batch_align_and_embed 自动对齐两个 tokenizer 的位置
    - lm_head 在 domain vocab 上计算 loss

    Args:
        neuron: ResonanceNeuron (lm_head_rank=0, per-neuron lm_head).
        texts: list of raw text strings.
        domain: domain name (for logging).
        shared_embedding: nn.Embedding(256000, 512) shared across all neurons.
        domain_sp: domain-specific SentencePieceProcessor.
        general_sp: general SentencePieceProcessor.
        num_steps: training steps.
        batch_size: batch size.
        lr: learning rate.
        device: "cpu" or "cuda".
        log_every: logging interval.
        save_path: checkpoint save path.

    Returns:
        {"final_loss": float, "final_ppl": float, "steps": int, "domain": str}
    """
    # ── Build text index for cycling ──
    n_texts = len(texts)

    def _sample_batch() -> list[str]:
        idx = torch.randint(0, n_texts, (batch_size,))
        return [texts[int(i)] for i in idx]

    optimizer = torch.optim.AdamW(
        list(neuron.parameters()) + list(shared_embedding.parameters()),
        lr=lr,
    )
    neuron.train()
    shared_embedding.train()

    total_loss = 0.0
    step, t_start = 0, time.time()
    best_loss = float("inf")
    best_step = 0

    for _ in range(num_steps):
        batch_texts = _sample_batch()

        # Align domain + general tokenizers → shared_emb + targets
        shared_emb, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_embedding,
        )
        shared_emb = shared_emb.to(device)       # [B, L_g, 512]
        targets = targets.to(device)              # [B, L_g] domain token IDs
        mask = mask.to(device)                    # [B, L_g]

        # Forward through neuron (embed_adapter → Transformer → logits)
        result = neuron.forward(shared_emb, return_logits=True)
        logits = result["logits"]  # [B, L_g, domain_vocab]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        # Apply mask: ignore positions that are padding or unaligned
        shift_targets = shift_targets.clone()
        shift_targets[~shift_mask] = -100

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(neuron.parameters()) + list(shared_embedding.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_step = step

        if step % log_every == 0:
            avg_loss = total_loss / step
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            print(
                f"  [{domain}] step {step}/{num_steps} "
                f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} "
                f"PPL={ppl:.1f} elapsed={elapsed:.0f}s"
            )

    avg_loss = total_loss / max(step, 1)
    ppl = math.exp(min(avg_loss, 20))
    elapsed = time.time() - t_start

    print(
        f"\n[{domain}] Training done. {step} steps, "
        f"avg_loss={avg_loss:.4f}, PPL={ppl:.1f}, "
        f"best_loss={best_loss:.4f}@step{best_step}, "
        f"time={elapsed:.0f}s ({elapsed/60:.1f}min)"
    )

    # Save checkpoint (neuron only; shared_embedding saved separately)
    save_path = save_path or os.path.join(OUTPUT_DIR, f"neuron_{domain}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "neuron_config": neuron.config,
        "state_dict": neuron.state_dict(),
        "domain": domain,
        "result": {
            "final_loss": avg_loss,
            "final_ppl": ppl,
            "steps": step,
            "best_loss": best_loss,
            "best_step": best_step,
        },
    }, save_path)
    print(f"  Saved: {save_path}")

    return {"final_loss": avg_loss, "final_ppl": ppl, "steps": step, "domain": domain}


def main():
    parser = argparse.ArgumentParser(description="P7: From-scratch neuron training (shared embedding)")
    parser.add_argument("--domain", default=None,
                        help="Single domain to train (zh/en/code/math/general)")
    parser.add_argument("--all", action="store_true",
                        help="Train all 5 domains")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps per domain (default: 2000)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Checkpoint save directory")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--spec", default="standard",
                        help="Neuron spec (compact/standard/expert). Default: standard")
    parser.add_argument("--general_tokenizer", default=None,
                        help="Path to general 256K tokenizer model (default: auto-detect)")
    args = parser.parse_args()

    domains = list(DOMAIN_VOCAB_SIZES.keys()) if args.all else [args.domain]
    if args.domain is None and not args.all:
        parser.error("Either --domain or --all is required")

    print(f"P7 From-scratch Training (shared embedding)")
    print(f"  Domains: {domains}")
    print(f"  Steps/domain: {args.steps}")
    print(f"  Spec: {args.spec or 'standard'}")
    print(f"  Device: {args.device}")
    print()

    # ── Create shared resources ──
    shared_embedding = load_or_create_shared_embedding(args.device)
    general_sp = load_general_tokenizer(args.general_tokenizer)
    print()

    all_results = {}

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"[{domain}] Loading data...")

        # 1. Load domain tokenizer
        domain_sp = load_domain_tokenizer(domain)
        print(f"  Domain tokenizer: vocab={domain_sp.vocab_size()}")

        # 2. Load raw texts
        texts = load_domain_texts(domain)
        print(f"  Loaded {len(texts)} raw texts")

        # 3. Create or load neuron
        if args.resume and domain == domains[0]:
            print(f"  Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
            cfg = ckpt["neuron_config"]
            neuron = ResonanceNeuron(cfg).to(args.device)
            neuron.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            cfg = get_domain_neuron_config(domain, args.spec)
            neuron = ResonanceNeuron(cfg).to(args.device)
            print(f"  Created neuron: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
                  f"lm_head_vocab={cfg.vocab_size}, params≈{cfg.approx_params_m:.0f}M")

        # 4. Train
        save_path = os.path.join(args.output_dir, f"neuron_{domain}.pt")
        result = train_one_neuron(
            neuron=neuron,
            texts=texts,
            domain=domain,
            shared_embedding=shared_embedding,
            domain_sp=domain_sp,
            general_sp=general_sp,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            log_every=args.log_every,
            save_path=save_path,
        )
        all_results[domain] = result

    # Save shared embedding
    save_shared_embedding(shared_embedding)

    # Summary
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'domain':<10} {'loss':<10} {'PPL':<10} {'steps':<8} {'lm_head_vocab':<14}")
    print("-" * 55)
    for domain, r in all_results.items():
        print(f"{domain:<10} {r['final_loss']:<10.4f} {r['final_ppl']:<10.1f} "
              f"{r['steps']:<8} {DOMAIN_VOCAB_SIZES[domain]:<14}")
    print(f"\nCheckpoints saved to: {args.output_dir}/")
    print(f"Shared embedding saved to: {SHARED_EMBEDDING_PATH}")


if __name__ == "__main__":
    main()
