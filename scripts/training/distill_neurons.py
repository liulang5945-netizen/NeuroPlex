"""Phase 2: Distill domain neurons + train field conditioning for resonance.

Two-phase pipeline:
  Phase 1: Independent neuron distillation (LM + teacher distillation + field contrastive)
  Phase 2: Joint field conditioning — train field_read_layers so neurons can use
           each other's field_vectors to improve cross-domain predictions.

Usage:
    python scripts/training/distill_neurons.py \
        --checkpoint /path/to/checkpoint \
        --data_dir data/distill \
        --output_dir data/neurons \
        --steps 2000 \
        --field_cond_steps 500 \
        --device cuda
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import argparse, math, os, sys, time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance.shared_embed import SharedEmbedProj
from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    NeuronConfig, COMPACT, STANDARD, EXPERT,
    ConfidenceGate, EarlyStopResonance, QualityFilter,
)

# ── Projection caches (teacher 2048 → student dims) ──
_embed_proj: torch.nn.Linear | None = None  # legacy random proj (unused now, kept for reference)
_distill_proj: torch.nn.Linear | None = None
_field_anchor_proj: torch.nn.Linear | None = None  # H12: field-space anchor projection cache
# Shared, stable teacher->neuron embedding projection (H10): distill trains ONE
# SharedEmbedProj and persists data/shared_proj.pt so what neurons train on
# == what verification evaluates on (no more re-randomised projections).
_shared_embed_proj: SharedEmbedProj | None = None


def _load_shared_embed_proj(src_dim: int, target_dim: int) -> SharedEmbedProj:
    """Return the shared teacher->neuron embedding projection (H10).

    Priority: an in-memory instance, then a persisted data/shared_proj.pt,
    then a fresh orthogonal init (which distillation should save so
    verification can reuse the exact weights the neurons trained on).
    """
    global _shared_embed_proj
    if (_shared_embed_proj is not None
            and _shared_embed_proj.src_dim == src_dim
            and _shared_embed_proj.target_dim == target_dim):
        return _shared_embed_proj
    # H10: prefer data/distill/shared_proj.pt (SVD-initialised from teacher embedding).
    for proj_path in [os.path.join("data", "distill", "shared_proj.pt"),
                      os.path.join("data", "shared_proj.pt")]:
        if os.path.exists(proj_path) and src_dim == 2048 and target_dim == 512:
            try:
                _shared_embed_proj = SharedEmbedProj.load(proj_path, src_dim, target_dim)
                print(f"  [proj] loaded shared_proj from {proj_path}")
                return _shared_embed_proj
            except Exception as exc:
                print(f"  [warn] {proj_path} load failed ({exc}); trying next")
    raise FileNotFoundError(
        "H10: no shared_proj.pt found. Run scripts/training/build_shared_projections.py first."
    )


def _project_embedding(teacher_emb: torch.Tensor, target_dim: int) -> torch.Tensor:
    src_dim = teacher_emb.shape[-1]
    if src_dim == target_dim:
        return teacher_emb
    proj = _load_shared_embed_proj(src_dim, target_dim).to(teacher_emb.device)
    return proj(teacher_emb)


def _project_teacher_hidden(teacher_hidden: torch.Tensor, target_dim: int) -> torch.Tensor:
    """H11: load teacher-hidden SVD projection from disk instead of random init.

    For hidden distill target (target_dim == hidden_size) load distill_hidden_proj_H.pt.
    For field contrastive anchors (target_dim == field_dim) load field_proj_D.pt.
    """
    global _distill_proj, _field_anchor_proj
    src_dim = teacher_hidden.shape[-1]
    if src_dim == target_dim:
        return teacher_hidden
    device = teacher_hidden.device

    # Choose cache slot by target_dim: field_dim (4096) uses field_anchor path,
    # smaller (hidden_size like 384/768/1024) uses hidden distill path.
    is_field = target_dim >= 4096
    cache = _field_anchor_proj if is_field else _distill_proj

    if cache is None or cache.in_features != src_dim or cache.out_features != target_dim:
        # Load persisted SVD proj
        if is_field:
            path = os.path.join("data", "distill", f"field_proj_{target_dim}.pt")
        else:
            path = os.path.join("data", "distill", f"distill_hidden_proj_{target_dim}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"H11/H12: {path} not found. Run scripts/training/build_shared_projections.py first."
            )
        lin = torch.nn.Linear(src_dim, target_dim, bias=False)
        lin.load_state_dict(torch.load(path, map_location="cpu"))
        for p in lin.parameters():
            p.requires_grad_(False)
        lin.eval()
        lin = lin.to(device)
        if is_field:
            _field_anchor_proj = lin
            cache = _field_anchor_proj
        else:
            _distill_proj = lin
            cache = _distill_proj
        print(f"  [proj] loaded {path}")

    return cache(teacher_hidden)


def create_neuron(spec: str, device: str = "cpu") -> ResonanceNeuron:
    from taiji.resonance import FOUNDATION
    if spec == "compact":
        cfg = COMPACT
    elif spec == "expert":
        cfg = EXPERT
    elif spec == "foundation":
        cfg = FOUNDATION
    else:
        cfg = STANDARD
    neuron_cfg = NeuronConfig(
        hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size, spec=spec,
        vocab_size=256000, base_embed_dim=512,
        field_dim=cfg.field_dim,  # H9 unified: all specs use 4096 (TINY_TEST is the only exception)
    )
    return ResonanceNeuron(neuron_cfg).to(device)


# ═══════════════════════════════════════════════════════════
# Phase 1: Independent neuron distillation
# ═══════════════════════════════════════════════════════════

def distill_one_neuron(
    teacher_model, neuron: ResonanceNeuron, shared_embedding: torch.nn.Embedding,
    domain_data: torch.Tensor, domain_name: str,
    num_steps: int = 2000, batch_size: int = 4,
    lm_weight: float = 0.6, distill_weight: float = 0.2,
    field_contrastive_weight: float = 0.2, lr: float = 5e-4,
    device: str = "cpu", log_every: int = 20,
    teacher_directions: dict = None,
) -> Dict[str, float]:
    """Distill a single neuron with field_write contrastive training.

    Three loss terms:
    1. LM loss — learn language modeling on domain data
    2. Distill loss — align hidden states with teacher
    3. Field contrastive — pull field_vector toward own domain, push away from others
    """
    dataset = TensorDataset(domain_data)

    def _cycle_loader():
        while True:
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                yield batch
    loader = _cycle_loader()

    pos_dir, neg_dirs = None, {}
    if teacher_directions and field_contrastive_weight > 0:
        fd = neuron.config.field_dim
        with torch.no_grad():
            for dname, tdir in teacher_directions.items():
                proj = _project_teacher_hidden(tdir.unsqueeze(0).to(device), fd).squeeze(0)
                proj = proj / (proj.norm() + 1e-8)
                if dname == domain_name:
                    pos_dir = proj.detach()
                else:
                    neg_dirs[dname] = proj.detach()
        if pos_dir is not None:
            print(f"  Field contrastive: {len(neg_dirs)} negative domains")

    optimizer = torch.optim.AdamW(neuron.parameters(), lr=lr)
    neuron.train()
    teacher_model.eval()

    total_lm, total_distill, total_field = 0.0, 0.0, 0.0
    step, t_start = 0, time.time()

    for batch in loader:
        if step >= num_steps:
            break
        input_ids = batch[0].to(device)

        with torch.no_grad():
            teacher_emb = shared_embedding(input_ids)
            shared_emb = _project_embedding(teacher_emb, neuron.config.base_embed_dim)
            teacher_hidden = extract_hidden_states(teacher_model, input_ids)
            teacher_last = teacher_hidden[:, -1, :]

        result = neuron.forward(shared_emb, return_logits=True)
        logits, hidden, field_vec = result["logits"], result["hidden_before_write"], result["field_vector"]

        # LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1), ignore_index=-100,
        )

        # Distillation loss
        teacher_proj = _project_teacher_hidden(teacher_last, neuron.config.hidden_size)
        distill_loss = F.mse_loss(hidden, teacher_proj)

        # Field contrastive loss
        field_loss = torch.tensor(0.0, device=device)
        if pos_dir is not None:
            cos_pos = (field_vec * pos_dir.unsqueeze(0)).sum(dim=-1).mean()
            field_loss = field_loss + (1.0 - cos_pos)
            for od_dir in neg_dirs.values():
                cos_neg = (field_vec * od_dir.unsqueeze(0)).sum(dim=-1).mean()
                field_loss = field_loss + torch.clamp(cos_neg - 0.3, min=0.0)
            field_loss = field_loss / (1 + len(neg_dirs))

        loss = lm_weight * lm_loss + distill_weight * distill_loss
        if field_contrastive_weight > 0:
            loss = loss + field_contrastive_weight * field_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_lm += lm_loss.item()
        total_distill += distill_loss.item()
        total_field += field_loss.item()
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t_start
            ppl = math.exp(total_lm / step) if (total_lm / step) < 10 else float('inf')
            print(f"  [{domain_name}] step {step}/{num_steps} | "
                  f"lm={total_lm/step:.4f} distill={total_distill/step:.4f} "
                  f"field={total_field/step:.4f} | PPL={ppl:.1f} | {elapsed:.0f}s")

    avg_lm = total_lm / max(step, 1)
    return {
        "final_loss": avg_lm,
        "final_ppl": math.exp(avg_lm) if avg_lm < 10 else float('inf'),
        "steps": step, "domain": domain_name,
        "field_loss": total_field / max(step, 1),
    }


def evaluate_neuron(neuron, shared_embedding, data, device="cpu", max_batches=50):
    neuron.eval()
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            input_ids = batch[0].to(device)
            teacher_emb = shared_embedding(input_ids)
            shared_emb = _project_embedding(teacher_emb, neuron.config.base_embed_dim)
            logits = neuron.forward(shared_emb, return_logits=True)["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1), ignore_index=-100,
            )
            total_loss += loss.item() * shift_targets.numel()
            total_tokens += shift_targets.numel()
    return math.exp(total_loss / max(total_tokens, 1))


# ═══════════════════════════════════════════════════════════
# Phase 2: Joint field conditioning training
# ═══════════════════════════════════════════════════════════

def _freeze_except(neuron: ResonanceNeuron, unfreeze_patterns: List[str]):
    """Freeze all params except those matching any pattern in unfreeze_patterns."""
    for name, param in neuron.named_parameters():
        param.requires_grad = any(p in name for p in unfreeze_patterns)


def train_field_conditioning_pair(
    neuron_a: ResonanceNeuron, neuron_b: ResonanceNeuron,
    domain_a: str, domain_b: str,
    cross_data: torch.Tensor,
    cached_embeddings: Dict[str, torch.Tensor] | None = None,
    fixed_proj: torch.nn.Linear | None = None,
    num_steps: int = 500, batch_size: int = 4, lr: float = 1e-3,
    device: str = "cpu", log_every: int = 100,
) -> Dict:
    """Train field_read_layers so two neurons can benefit from each other's field vectors.

    Training protocol:
    - Freeze transformer body (embed_adapter, layers, norm, lm_head)
    - Only train: field_read_layers, field_write
    - Round 1: each neuron forwards independently → gets field_vectors
    - Round 2: each neuron forwards conditioned on joint field_state
    - Loss: LM loss on Round 2 (avg of both neurons) + penalty if R2 > R1

    Uses cached teacher embeddings for realistic input (not random noise).
    """
    field_dim = neuron_a.config.field_dim

    # Freeze bodies
    _freeze_except(neuron_a, ["field_read", "field_write"])
    _freeze_except(neuron_b, ["field_read", "field_write"])

    trainable = [p for n in [neuron_a, neuron_b] for p in n.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # Build training dataset: token IDs + pre-computed 512-dim embeddings
    # Use cached embeddings if available, otherwise fall back to random
    use_cached = cached_embeddings is not None and fixed_proj is not None

    if use_cached:
        # Load cached 2048-dim embeddings, project to 512 on the fly
        emb_list = []
        id_list = []
        for domain in [domain_a, domain_b]:
            if domain in cached_embeddings:
                d = cached_embeddings[domain]
                n = min(len(d["input_ids"]), 25)
                id_list.append(d["input_ids"][:n])
                with torch.no_grad():
                    emb_list.append(fixed_proj(d["embeddings"][:n].to(device)).cpu())
        if id_list:
            # Issue 2: different domains may have different seq lengths, so
            # torch.cat on dim=0 would crash. Compute common_len from ALL
            # sequences (ids + embs + cross_data) BEFORE cat, then slice each
            # tensor to common_len first.
            common_len = min(
                cross_data.shape[1],
                *(t.shape[1] for t in id_list),
                *(t.shape[1] for t in emb_list),
            )
            id_list = [t[:, :common_len] for t in id_list]
            emb_list = [t[:, :common_len, :] for t in emb_list]
            combined_ids = torch.cat(id_list, dim=0)
            combined_emb = torch.cat(emb_list, dim=0)
            cross_ids = cross_data[:, :common_len]
            cross_emb = torch.randn(cross_ids.shape[0], common_len, neuron_a.config.base_embed_dim)
            all_ids = torch.cat([combined_ids, cross_ids], dim=0)
            all_emb = torch.cat([combined_emb, cross_emb], dim=0)
        else:
            all_ids = cross_data
            all_emb = torch.randn(cross_data.shape[0], cross_data.shape[1], neuron_a.config.base_embed_dim)
    else:
        all_ids = cross_data
        all_emb = torch.randn(cross_data.shape[0], cross_data.shape[1], neuron_a.config.base_embed_dim)

    perm = torch.randperm(len(all_ids))
    all_ids, all_emb = all_ids[perm], all_emb[perm]

    print(f"  Training samples: {len(all_ids)}")

    dataset = TensorDataset(all_ids, all_emb)
    def _cycle():
        while True:
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                yield batch
    loader = _cycle()

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    neuron_a.train()
    neuron_b.train()

    r1_losses, r2_losses = [], []
    t0 = time.time()

    for step in range(num_steps):
        batch = next(loader)
        input_ids, shared_emb = [b.to(device) for b in batch]

        # ── Round 1: independent forward ──
        r1_a = neuron_a.forward(shared_emb, return_logits=True, round_num=1, field_state=None)
        r1_b = neuron_b.forward(shared_emb, return_logits=True, round_num=1, field_state=None)

        # ── Build joint field state ──
        field = ResonanceField(dim=field_dim).to(device)
        field.reset(batch_size=shared_emb.shape[0])
        field.write(domain_a, r1_a["field_vector"])
        field.write(domain_b, r1_b["field_vector"])
        field_state = field.get_normalised_state()

        # ── Round 2: conditioned forward ──
        r2_a = neuron_a.forward(shared_emb, return_logits=True, round_num=2, field_state=field_state)
        r2_b = neuron_b.forward(shared_emb, return_logits=True, round_num=2, field_state=field_state)

        shift_targets = input_ids[:, 1:].contiguous()

        def lm_loss(logits):
            shift_logits = logits[:, :-1, :].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1), ignore_index=-100,
            )

        l_r1_avg = (lm_loss(r1_a["logits"]) + lm_loss(r1_b["logits"])) / 2
        r2_logits_avg = (r2_a["logits"] + r2_b["logits"]) / 2
        l_r2 = lm_loss(r2_logits_avg)
        l_r2_a = lm_loss(r2_a["logits"])
        l_r2_b = lm_loss(r2_b["logits"])

        # Loss: R2 LM + penalty if R2 worse than R1 + individual neuron R2 losses
        alpha = 2.0
        improvement_penalty = torch.clamp(l_r2 - l_r1_avg, min=0.0)
        loss = l_r2 + alpha * improvement_penalty + 0.3 * (l_r2_a + l_r2_b)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        r1_losses.append(l_r1_avg.item())
        r2_losses.append(l_r2.item())

        if (step + 1) % log_every == 0:
            avg_r1 = sum(r1_losses[-log_every:]) / len(r1_losses[-log_every:])
            avg_r2 = sum(r2_losses[-log_every:]) / len(r2_losses[-log_every:])
            delta = avg_r1 - avg_r2
            print(f"  [{domain_a}+{domain_b}] step {step+1:4d}/{num_steps} | "
                  f"R1_loss={avg_r1:.4f} R2_loss={avg_r2:.4f} delta={delta:+.4f} | "
                  f"R1_PPL={math.exp(avg_r1):.1f} R2_PPL={math.exp(avg_r2):.1f} | "
                  f"{time.time()-t0:.0f}s")

    return {"r1_loss": sum(r1_losses)/len(r1_losses), "r2_loss": sum(r2_losses)/len(r2_losses)}


# ═══════════════════════════════════════════════════════════
# Phase 3: Resonance verification
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def verify_resonance(
    neurons: Dict[str, ResonanceNeuron],
    test_data: Dict[str, torch.Tensor],
    device: str = "cpu",
    fixed_proj: torch.nn.Linear | None = None,
    cached_embeddings: Dict | None = None,
    teacher_embedding=None,
) -> Dict:
    """Verify 1+1>2 resonance effect: Round 1 vs Round 2 with REAL vs RANDOM field.

    Uses cached teacher embeddings when available for realistic PPL measurement.
    """
    # Issue 4: switch all neurons to eval mode. train_field_conditioning_pair
    # leaves them in train() mode; dropout would pollute PPL measurement even
    # under no_grad (no_grad only disables autograd, not dropout).
    for neuron in neurons.values():
        neuron.eval()

    field_dim = next(iter(neurons.values())).config.field_dim
    base_embed_dim = next(iter(neurons.values())).config.base_embed_dim

    def _resolve_embeddings(domain_to_ids: List[Tuple[str, torch.Tensor]]) -> torch.Tensor:
        """Build aligned [N, T, base_embed_dim] embeddings for (domain, ids) chunks.

        Issue 1: real cache lookup instead of always-random embeddings.
        Priority:
          1. cached_embeddings[domain] projected with fixed_proj (real cached
             teacher embeddings — what the neurons actually trained on).
          2. teacher_embedding(input_ids) projected with fixed_proj (run the
             teacher embedding lookup live when cache misses).
          3. random (with warning) — only when neither cache nor teacher is
             available.
        Returns a CPU tensor (DataLoader-friendly); batched_ppl moves each
        batch back to device.
        """
        emb_chunks = []
        warned = False
        for domain, ids in domain_to_ids:
            n = ids.shape[0]
            seq_len = ids.shape[1]
            emb = None
            if cached_embeddings is not None and fixed_proj is not None and domain in cached_embeddings:
                cached = cached_embeddings[domain]
                cached_emb = cached["embeddings"][:n].to(device)
                with torch.no_grad():
                    emb = fixed_proj(cached_emb).cpu()
                emb = emb[:, :seq_len, :]
            elif teacher_embedding is not None and fixed_proj is not None:
                with torch.no_grad():
                    teacher_emb = teacher_embedding(ids.to(device))
                    emb = fixed_proj(teacher_emb).cpu()
                emb = emb[:, :seq_len, :]
            else:
                if not warned:
                    print(f"  [warn] no cache and no teacher_embedding for domain "
                          f"'{domain}'; falling back to random embeddings")
                    warned = True
                emb = torch.randn(n, seq_len, base_embed_dim)
            emb_chunks.append(emb)
        return torch.cat(emb_chunks, dim=0)

    def batched_ppl(model_fn, ids_tensor, emb_tensor, batch_size=4):
        dataset = TensorDataset(ids_tensor, emb_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss, total_tokens = 0.0, 0
        for input_ids, emb in loader:
            input_ids = input_ids.to(device)
            emb = emb.to(device)
            logits = model_fn(emb)
            if isinstance(logits, tuple):
                logits = logits[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1), ignore_index=-100,
            )
            total_loss += loss.item() * shift_targets.numel()
            total_tokens += shift_targets.numel()
        return math.exp(total_loss / max(total_tokens, 1))

    print("\n" + "=" * 60)
    print("RESONANCE VERIFICATION: Round 1 vs Round 2")
    print("=" * 60)

    verification = {}
    domains = list(neurons.keys())

    for nd1, nd2 in combinations(domains, 2):
        n1, n2 = neurons[nd1], neurons[nd2]
        pair_name = f"{nd1}+{nd2}"

        # Build mixed test data from both domains
        d1 = test_data.get(nd1)
        d2 = test_data.get(nd2)
        if d1 is None or d2 is None:
            continue
        n_each = min(len(d1), len(d2), 25)
        mixed_ids = torch.cat([d1[:n_each], d2[:n_each]], dim=0)
        # Issue 1: build REAL embeddings (cached/teacher) aligned with mixed_ids
        # so PPL is measured on the same inputs the neurons trained on.
        mixed_emb = _resolve_embeddings([(nd1, d1[:n_each]), (nd2, d2[:n_each])])
        # Defensively align seq dim — cached seq len may differ from test_data.
        common_len = min(mixed_ids.shape[1], mixed_emb.shape[1])
        mixed_ids = mixed_ids[:, :common_len]
        mixed_emb = mixed_emb[:, :common_len, :]

        # Round 1 functions
        def r1_fn_n1(emb):
            return n1.forward(emb, return_logits=True, round_num=1, field_state=None)["logits"]
        def r1_fn_n2(emb):
            return n2.forward(emb, return_logits=True, round_num=1, field_state=None)["logits"]

        # Round 2 with REAL field
        def make_r2_real(target_neuron, other_neuron, other_name):
            def fn(emb):
                other_r1 = other_neuron.forward(emb, return_logits=True, round_num=1, field_state=None)
                f = ResonanceField(dim=field_dim).to(device)
                f.write(other_name, other_r1["field_vector"])
                return target_neuron.forward(
                    emb, return_logits=True, round_num=2, field_state=f.get_normalised_state(),
                )["logits"]
            return fn

        # Round 2 with RANDOM field (control)
        def make_r2_random(neuron):
            def fn(emb):
                # Issue 5: per-sample random field [B, D] to match the shape of
                # the real field (make_r2_real uses f.get_normalised_state() which
                # is [B, D]); the old [D] shape made the control unfair.
                rand_f = torch.randn(emb.shape[0], field_dim, device=device)
                rand_f = rand_f / (rand_f.norm(dim=-1, keepdim=True) + 1e-8)
                return neuron.forward(emb, return_logits=True, round_num=2, field_state=rand_f)["logits"]
            return fn

        # Round 2 ensemble (both neurons + joint field)
        def make_r2_ens(na, nb, da, db):
            def fn(emb):
                r1_a = na.forward(emb, return_logits=True, round_num=1, field_state=None)
                r1_b = nb.forward(emb, return_logits=True, round_num=1, field_state=None)
                f = ResonanceField(dim=field_dim).to(device)
                f.write(da, r1_a["field_vector"])
                f.write(db, r1_b["field_vector"])
                fs = f.get_normalised_state()
                la = na.forward(emb, return_logits=True, round_num=2, field_state=fs)["logits"]
                lb = nb.forward(emb, return_logits=True, round_num=2, field_state=fs)["logits"]
                return (la + lb) / 2
            return fn

        ppl_r1_n1 = batched_ppl(r1_fn_n1, mixed_ids, mixed_emb)
        ppl_r1_n2 = batched_ppl(r1_fn_n2, mixed_ids, mixed_emb)
        ppl_r2_n1_real = batched_ppl(make_r2_real(n1, n2, nd2), mixed_ids, mixed_emb)
        ppl_r2_n2_real = batched_ppl(make_r2_real(n2, n1, nd1), mixed_ids, mixed_emb)
        ppl_r2_n1_rand = batched_ppl(make_r2_random(n1), mixed_ids, mixed_emb)
        ppl_r2_n2_rand = batched_ppl(make_r2_random(n2), mixed_ids, mixed_emb)
        ppl_r2_ens = batched_ppl(make_r2_ens(n1, n2, nd1, nd2), mixed_ids, mixed_emb)

        best_r1 = min(ppl_r1_n1, ppl_r1_n2)
        best_r2 = min(ppl_r2_n1_real, ppl_r2_n2_real, ppl_r2_ens)
        delta = best_r1 - best_r2

        real_beats_random = (ppl_r2_n1_rand > ppl_r2_n1_real) and (ppl_r2_n2_rand > ppl_r2_n2_real)
        resonance_works = delta > 0

        status = "✅ RESONANCE" if resonance_works else "❌ NO EFFECT"
        control = "✅ REAL>>RANDOM" if real_beats_random else "⚠️ NO DIFF"
        print(f"  {pair_name:>16} | R1_best={best_r1:.1f} R2_best={best_r2:.1f} "
              f"Δ={delta:+.1f} | {status} | {control}")

        verification[pair_name] = {
            "r1_best": best_r1, "r2_best": best_r2, "delta": delta,
            "real_beats_random": real_beats_random, "resonance_works": resonance_works,
            "r2_ensemble": ppl_r2_ens,
        }

    # Summary
    n_pairs = len(verification)
    n_resonance = sum(1 for v in verification.values() if v["resonance_works"])
    n_control = sum(1 for v in verification.values() if v["real_beats_random"])
    avg_delta = sum(v["delta"] for v in verification.values()) / max(n_pairs, 1)

    print(f"\n  ── Summary ──")
    print(f"  Resonance (R2 > R1):   {n_resonance}/{n_pairs}")
    print(f"  REAL >> RANDOM control: {n_control}/{n_pairs}")
    print(f"  Average delta:          {avg_delta:+.1f} PPL")
    if n_resonance >= n_pairs * 0.5:
        print(f"\n  ✅ 1+1>2 RESONANCE VERIFIED")
    else:
        print(f"\n  ⚠️  More training needed")

    return verification


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════


def _safe_save(obj, primary_path):
    """Robust torch.save with fallback locations (Windows sandbox may deny some dirs)."""
    base = os.path.basename(primary_path)
    candidates = [primary_path,
                  os.path.join("data", "distill", "neurons_out", base),
                  os.path.join("checkpoints_v3", base)]
    last_err = None
    for cand in candidates:
        try:
            d = os.path.dirname(cand)
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save(obj, cand)
            if cand != primary_path:
                print(f"    [safe_save] wrote fallback: {cand} (primary {primary_path} denied)")
            return cand
        except (PermissionError, RuntimeError, OSError) as exc:
            last_err = exc
            print(f"    [safe_save] {cand} failed: {exc}; next")
    raise RuntimeError(f"All save fallbacks failed for {primary_path}: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Distill neurons + train field conditioning")
    # Paths
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/checkpoint-481000")
    parser.add_argument("--data_dir", default="data/distill")
    parser.add_argument("--output_dir", default="data/neurons")
    # Phase 1
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--field_contrastive_weight", type=float, default=0.2)
    parser.add_argument("--skip_domains", nargs="*", default=[])
    # Phase 2
    parser.add_argument("--field_cond_steps", type=int, default=500,
                        help="Field conditioning training steps per pair (0 to skip)")
    parser.add_argument("--field_cond_lr", type=float, default=1e-3)
    parser.add_argument("--skip_field_cond", action="store_true",
                        help="Skip Phase 2 entirely")
    # Misc
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        _probe = os.path.join(args.output_dir, ".probe")
        with open(_probe, "w") as _f:
            _f.write("t")
        os.remove(_probe)
    except (PermissionError, OSError) as exc:
        fb = os.path.join("data", "distill", "neurons_out")
        os.makedirs(fb, exist_ok=True)
        print(f"[warn] output_dir {args.output_dir} not writable ({exc}); switching to {fb}")
        args.output_dir = fb

    # ═══════════════════════════════════════════════════════
    # Phase 1: Independent distillation
    # ═══════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Independent Neuron Distillation")
    print("=" * 60)

    print(f"\n[1/4] Loading teacher model...")
    teacher, embedding = load_teacher_model(args.checkpoint, device=args.device)

    print(f"\n[2/4] Loading domain datasets...")
    data_path = os.path.join(args.data_dir, "domain_datasets.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run prepare_distill_data.py first.")
    datasets = torch.load(data_path, map_location="cpu", weights_only=True)

    teacher_dirs = None
    dir_path = os.path.join(args.data_dir, "teacher_directions.pt")
    if os.path.exists(dir_path) and args.field_contrastive_weight > 0:
        teacher_dirs = torch.load(dir_path, map_location="cpu", weights_only=True)
        print(f"  Loaded teacher directions for {len(teacher_dirs)} domains")

    # C3 修复：使用全局 DEFAULT_NEURON_SPEC，避免硬编码 spec 与生产不一致。
    # 历史值: "foundation" (hidden=384)；现默认 "compact" (hidden=512)。
    # 若需蒸馏其他 spec，传入 --spec 参数覆盖。
    from taiji.resonance import DEFAULT_NEURON_SPEC
    DOMAIN_SPECS = {d: DEFAULT_NEURON_SPEC for d in ["zh", "en", "code", "math", "general"]}

    print(f"\n[3/4] Distilling neurons ({args.steps} steps each)...")
    results, neurons = {}, {}

    for domain in ["zh", "en", "code", "math", "general"]:
        if domain in args.skip_domains:
            continue
        if domain not in datasets:
            continue

        spec = DOMAIN_SPECS[domain]
        print(f"\n  --- {domain} ({spec}) ---")
        neuron = create_neuron(spec, device=args.device)
        print(f"  Params: {sum(p.numel() for p in neuron.parameters()):,}")

        result = distill_one_neuron(
            teacher, neuron, embedding, datasets[domain], domain,
            num_steps=args.steps, batch_size=args.batch_size,
            lr=args.lr, device=args.device,
            field_contrastive_weight=args.field_contrastive_weight,
            teacher_directions=teacher_dirs,
        )

        ppl_own = evaluate_neuron(neuron, embedding, datasets[domain], device=args.device)
        result["ppl_own"] = ppl_own
        print(f"  {domain}: PPL(own)={ppl_own:.1f}")

        results[domain], neurons[domain] = result, neuron

        ckpt_path = os.path.join(args.output_dir, f"neuron_{domain}.pt")
        actual_path = _safe_save({
            "neuron_config": neuron.config, "state_dict": neuron.state_dict(),
            "domain": domain, "result": result,
        }, ckpt_path)
        print(f"  Saved: {actual_path}")

    # Cross-domain PPL (Phase 1 only)
    print(f"\n[4/4] Cross-domain PPL matrix...")
    print(f"  {'':>10}", end="")
    for d in datasets:
        print(f"{d:>10}", end="")
    print()

    ppl_matrix = {}
    for nd, neuron in neurons.items():
        print(f"  {nd:>10}", end="")
        for td, tdata in datasets.items():
            ppl = evaluate_neuron(neuron, embedding, tdata, device=args.device)
            ppl_matrix[(nd, td)] = ppl
            print(f"{ppl:>10.1f}", end="")
        print()

    print("\n  Self-PPL:")
    for nd in neurons:
        print(f"    {nd}: {ppl_matrix[(nd, nd)]:.1f}")

    # ═══════════════════════════════════════════════════════
    # Phase 2: Joint field conditioning
    # ═══════════════════════════════════════════════════════
    if args.skip_field_cond or args.field_cond_steps <= 0:
        print("\n" + "=" * 60)
        print("PHASE 2: SKIPPED (--skip_field_cond or --field_cond_steps=0)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"PHASE 2: Joint Field Conditioning ({args.field_cond_steps} steps/pair)")
        print("=" * 60)

        # Load cross-domain data if available
        cross_path = os.path.join(args.data_dir, "cross_domain_data.pt")
        cross_data = torch.load(cross_path, map_location="cpu", weights_only=True) if os.path.exists(cross_path) else {}

        # Load cached teacher embeddings for embedding projection
        cached_embeddings = None
        fixed_proj = None
        cache_dir = os.path.join(args.data_dir, "cache")
        if os.path.exists(cache_dir):
            cached_embeddings = {}
            for domain in datasets:
                cache_path = os.path.join(cache_dir, f"{domain}_cached.pt")
                if os.path.exists(cache_path):
                    cached_embeddings[domain] = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached_embeddings:
                print(f"  Loaded cached teacher embeddings for {len(cached_embeddings)} domains")

        # Load fixed projection
        proj_path = os.path.join(args.data_dir, "fixed_proj.pt")
        if os.path.exists(proj_path):
            fixed_proj = torch.nn.Linear(2048, 512, bias=False).to(args.device)
            fixed_proj.load_state_dict(torch.load(proj_path, map_location=args.device))
            fixed_proj.requires_grad_(False)
            print(f"  Loaded fixed projection")

        domains = list(neurons.keys())
        field_cond_results = {}

        for nd1, nd2 in combinations(domains, 2):
            pair_key = f"{nd1}_{nd2}"
            print(f"\n  --- {nd1} + {nd2} ---")

            # Get training data: prefer cross-domain, fall back to mixed single-domain
            if pair_key in cross_data:
                train_data = cross_data[pair_key]
                print(f"  Using cross-domain data: {train_data.shape}")
            else:
                n_each = min(len(datasets[nd1]), len(datasets[nd2]), 30)
                train_data = torch.cat([datasets[nd1][:n_each], datasets[nd2][:n_each]], dim=0)
                print(f"  Using mixed single-domain data: {train_data.shape}")

            fc_result = train_field_conditioning_pair(
                neurons[nd1], neurons[nd2], nd1, nd2,
                train_data,
                cached_embeddings=cached_embeddings,
                fixed_proj=fixed_proj,
                num_steps=args.field_cond_steps, batch_size=args.batch_size,
                lr=args.field_cond_lr, device=args.device,
            )

            field_cond_results[pair_key] = fc_result

        # Save field-conditioned neurons
        print(f"\n  Saving field-conditioned neurons...")
        for nd in domains:
            neuron = neurons[nd]
            # Mark as field-conditioned
            ckpt_path = os.path.join(args.output_dir, f"neuron_{nd}_fieldcond.pt")
            _safe_save({
                "neuron_config": neuron.config,
                "state_dict": neuron.state_dict(),
                "domain": nd,
                "field_conditioned": True,
                # Issue 3: substring match would let "en" match "general";
                # split on "_" and check exact membership instead.
                "field_cond_pairs": [p for p in field_cond_results if nd in p.split("_")],
            }, ckpt_path)
        print(f"  Saved to {args.output_dir}/neuron_*_fieldcond.pt")

        # ═══════════════════════════════════════════════════════
        # Phase 3: Resonance verification
        # ═══════════════════════════════════════════════════════
        # Issue 1: pass the teacher embedding layer so verify_resonance can run
        # real teacher embeddings on cache misses instead of falling back to random.
        verify_resonance(neurons, datasets, device=args.device,
                         fixed_proj=fixed_proj, cached_embeddings=cached_embeddings,
                         teacher_embedding=embedding)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Phase 1: {len(neurons)} neurons distilled")
    if not args.skip_field_cond and args.field_cond_steps > 0:
        print(f"  Phase 2: {len(field_cond_results)} pairs field-conditioned")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
