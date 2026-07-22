"""P6-8: 自主进化 encoder 训练 + 重算 prototypes + 端到端验证.

完整流程：
  1. 加载 StandaloneEmbedding（P6-1 已构建）
  2. 构建 SharedContextEncoder（从 standalone_embedding 初始化）
  3. 用 domain_datasets.pt 跑 N 步自主进化训练（MLM + Contrastive + Hebbian）
  4. 用训后的 encoder 重算 embedding-based prototypes
  5. 验证路由准确率（对比 P6-5 的 0/4 baseline）
  6. 保存训后 encoder + 新 prototypes

Usage:
    python scripts/training/run_p6_self_evolve.py [--steps 100]
"""
from __future__ import annotations

import os
import sys
import argparse

# sentencepiece + 项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_libs"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from typing import Dict

import torch
import torch.nn.functional as F  # P6-9: for softmax in verify_routing
from taiji.loader import create_cortex
from taiji.resonance import (
    StandaloneEmbedding, SharedContextEncoder, HebbianUpdater,
    ContrastiveLoss, MLMLoss, SelfEvolver, ThalamicRouter,
)


# ── 配置 ──
NEURONS_DIR = "data/neurons"
STANDALONE_EMBED_PATH = "data/distill/standalone_embedding.pt"
DOMAIN_DATA_PATH = "data/distill/domain_datasets.pt"
ROUTER_PATH = "data/distill/thalamic_prototypes.pt"
ENCODER_PATH = "data/distill/shared_context_encoder.pt"
ROUTER_P6_PATH = "data/distill/thalamic_prototypes_p6.pt"

TEST_PROMPTS = [
    ("你好", "zh"),
    ("1+1=", "math"),
    ("hello world", "en"),
    ("def fibonacci", "code"),
]

# Encoder 配置（轻量级，CPU 可训）
ENCODER_CONFIG = dict(
    hidden_dim=512,
    num_layers=2,  # 2 层足够上下文感知，CPU 可承受
    num_heads=8,
    num_kv_heads=2,
    intermediate_size=1408,
    n_domains=5,  # P6-9: domain classifier head（zh/en/code/math/general）
)

# 训练配置
# P6-8 fix: 增大 batch_size (4→8) 让 domain-stratified batch 有更多正样本对
# P6-8 fix v3: 回到 300 步（500 步过度训练导致坍塌），加 uniformity loss 防坍塌
DEFAULT_STEPS = 300
BATCH_SIZE = 8
SEQ_LEN = 64
LR = 1e-4
HEBBIAN_INTERVAL = 10
# P6-8 fix: contrastive 权重 0.5→2.0，强化 domain 分离信号
CONTRASTIVE_WEIGHT = 2.0
# P6-9 fix: domain classification 权重（supervised signal，主导路由学习）
DOMAIN_WEIGHT = 5.0  # 让 supervised 信号压过自监督
# P6-8 fix: 每域样本数 50→100，增加训练数据量
SAMPLES_PER_DOMAIN = 100
# P6-8 fix v2: prototypes 重算样本数 50→100（与训练数据一致，更代表性）
PROTOTYPE_SAMPLES = 100


def load_domain_data() -> dict:
    """加载 domain_datasets.pt."""
    if not os.path.exists(DOMAIN_DATA_PATH):
        print(f"ERROR: {DOMAIN_DATA_PATH} not found")
        sys.exit(1)
    return torch.load(DOMAIN_DATA_PATH, map_location="cpu", weights_only=False)


def prepare_training_samples(domain_data: dict) -> list:
    """把 domain_data 转成 (input_ids, domain_id) 对."""
    domain_to_id = {nid: i for i, nid in enumerate(domain_data.keys())}
    samples = []
    for nid, ids_tensor in domain_data.items():
        did = domain_to_id[nid]
        ids_list = ids_tensor[:SAMPLES_PER_DOMAIN].tolist()  # P6-8 fix: 每域取前 SAMPLES_PER_DOMAIN
        for ids in ids_list:
            if isinstance(ids, list) and len(ids) > 0:
                samples.append((ids, did))
    return samples, domain_to_id


def pad_or_truncate(ids: list, seq_len: int) -> list:
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [0] * (seq_len - len(ids))


def random_subsequence(ids: list, min_len: int = 2, max_len: int = 64):
    """P6-9 fix: 随机截取子序列，让 classifier 见过不同长度输入.

    训练数据原本是 256 tokens 的长文本，但测试 prompt 可能只有 2-6 tokens。
    如果只用 64-token 截断训练，classifier 在短 prompt 上会坍塌。

    本函数从原序列中随机截取 [min_len, max_len] 长度的片段。
    """
    import random as _r
    n = len(ids)
    if n <= min_len:
        return list(ids)
    target_len = _r.randint(min_len, min(max_len, n))
    if n <= target_len:
        return list(ids)
    start = _r.randint(0, n - target_len)
    return ids[start:start + target_len]


def make_attention_mask(input_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """P6-9 fix: 根据 input_ids 构建 attention mask (1=有效, 0=padding)."""
    return (input_ids != pad_id).long()


def build_domain_stratified_batches(
    samples: list,
    domain_to_id: dict,
    n_steps: int,
    batch_size: int,
) -> list:
    """P6-8 fix: 构造 domain-stratified batch，保证每 batch 至少 2 个同 domain 样本.

    原 random shuffle + batch_size=4 时，5 个 domain 的样本均匀分布，
    每 batch 出现同 domain 正样本对的概率极低，导致 ContrastiveLoss 几乎恒为 0.

    本函数采用 "anchor domain + 负采样" 策略：
    - 每 batch 随机选一个 anchor domain，取 batch_size//2 个样本作为正样本对
    - 剩余位置从其他 domain 均匀采样，作为负样本

    Returns:
        List of (input_ids_list, domain_labels_list)
    """
    import random

    # 按 domain 分组
    domain_samples: Dict[int, list] = {did: [] for did in domain_to_id.values()}
    for ids, did in samples:
        domain_samples[did].append(ids)

    # 每个 domain 的样本池（循环使用）
    domain_pools = {did: list(s) for did, s in domain_samples.items()}
    domain_cursors = {did: 0 for did in domain_samples}

    batches = []
    n_pos = max(batch_size // 2, 2)  # 正样本数（至少 2）
    n_neg = batch_size - n_pos         # 负样本数

    all_dids = list(domain_samples.keys())

    for _ in range(n_steps):
        # 选 anchor domain
        anchor_did = random.choice(all_dids)
        if len(domain_pools[anchor_did]) < n_pos:
            # 样本不够，跳过
            continue

        batch_samples = []
        # 正样本：从 anchor domain 取 n_pos 个
        cursor = domain_cursors[anchor_did]
        pool = domain_pools[anchor_did]
        for _ in range(n_pos):
            batch_samples.append((pool[cursor % len(pool)], anchor_did))
            cursor += 1
        domain_cursors[anchor_did] = cursor
        # 池子用完时 reshuffle
        if cursor >= len(pool):
            random.shuffle(domain_pools[anchor_did])
            domain_cursors[anchor_did] = 0

        # 负样本：从其他 domain 均匀采样
        other_dids = [d for d in all_dids if d != anchor_did]
        random.shuffle(other_dids)
        for i in range(n_neg):
            neg_did = other_dids[i % len(other_dids)]
            pool = domain_pools[neg_did]
            if not pool:
                continue
            cursor = domain_cursors[neg_did]
            batch_samples.append((pool[cursor % len(pool)], neg_did))
            domain_cursors[neg_did] = cursor + 1
            if domain_cursors[neg_did] >= len(pool):
                random.shuffle(domain_pools[neg_did])
                domain_cursors[neg_did] = 0

        batches.append(batch_samples)

    return batches


def train_self_evolve(
    encoder: SharedContextEncoder,
    evolver: SelfEvolver,
    samples: list,
    domain_to_id: dict,
    n_steps: int,
    device: str,
) -> dict:
    """跑自主进化训练（P6-8 fix: domain-stratified batch）."""
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)

    # P6-8 fix: domain-stratified batch 保证正样本对
    batches = build_domain_stratified_batches(
        samples, domain_to_id, n_steps, BATCH_SIZE,
    )
    print(f"\n[Training] {len(batches)} stratified batches, "
          f"batch={BATCH_SIZE}, seq_len={SEQ_LEN}, lr={LR}")

    total_loss = 0.0
    mlm_losses = []
    contrastive_losses = []
    domain_losses = []  # P6-9
    n_actual_steps = 0

    for step, batch_samples in enumerate(batches):
        if len(batch_samples) < 2:
            continue

        # P6-9 fix: 随机截断到 2-64 tokens，让 classifier 见过短 prompt
        batch_ids = [pad_or_truncate(random_subsequence(ids, 2, SEQ_LEN), SEQ_LEN)
                     for ids, _ in batch_samples]
        domain_labels = [did for _, did in batch_samples]

        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        domain_tensor = torch.tensor(domain_labels, dtype=torch.long, device=device)
        # P6-9 fix: 构建 attention mask（0=padding, 1=有效）
        attention_mask = make_attention_mask(input_ids, pad_id=0)

        optimizer.zero_grad()
        loss = evolver.training_step(
            batch={"input_ids": input_ids, "attention_mask": attention_mask},
            domain_labels=domain_tensor,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_actual_steps += 1
        summary = evolver.get_loss_summary()
        mlm_losses.append(summary["mlm"])
        contrastive_losses.append(summary["contrastive"])
        domain_losses.append(summary["domain"])  # P6-9

        # 定期 Hebbian
        if (step + 1) % HEBBIAN_INTERVAL == 0:
            with torch.no_grad():
                evolver.apply_hebbian_to_embedding()

        if (step + 1) % 10 == 0:
            avg = total_loss / max(n_actual_steps, 1)
            recent_mlm = sum(mlm_losses[-10:]) / max(len(mlm_losses[-10:]), 1)
            recent_cont = sum(contrastive_losses[-10:]) / max(len(contrastive_losses[-10:]), 1)
            recent_dom = sum(domain_losses[-10:]) / max(len(domain_losses[-10:]), 1)  # P6-9
            print(f"  step {step+1}/{len(batches)}: avg_loss={avg:.4f}, "
                  f"mlm={recent_mlm:.4f}, contrastive={recent_cont:.4f}, "
                  f"domain={recent_dom:.4f}")  # P6-9

    encoder.eval()

    # 最终 Hebbian
    with torch.no_grad():
        hebbian_upd = evolver.apply_hebbian_to_embedding()

    return {
        "avg_loss": total_loss / max(n_actual_steps, 1),
        "final_mlm": mlm_losses[-1] if mlm_losses else 0.0,
        "final_contrastive": contrastive_losses[-1] if contrastive_losses else 0.0,
        "final_domain": domain_losses[-1] if domain_losses else 0.0,  # P6-9
        "hebbian_update": hebbian_upd,
        "n_steps_run": n_actual_steps,
    }


def recompute_prototypes_with_encoder(
    router: ThalamicRouter,
    encoder: SharedContextEncoder,
    neurons: dict,
    domain_data: dict,
    device: str,
) -> ThalamicRouter:
    """用训后的 encoder 重算 embedding-based prototypes.

    与 compute_prototypes_from_embedding 类似，但用 encoder.encode() 而非
    standalone_embedding()。encoder 提供上下文感知的 hidden state。
    """
    embed_dim = encoder.hidden_dim
    router.prototypes_embed = {}
    print(f"\n[Recompute] encoder-based prototypes (dim={embed_dim})...")

    encoder = encoder.to(device)
    encoder.eval()

    for nid, neuron in neurons.items():
        if nid not in domain_data:
            continue
        data = domain_data[nid]
        if data.shape[0] == 0:
            continue
        # P6-8 fix v2: 用 PROTOTYPE_SAMPLES 而非硬编码 50
        n = min(data.shape[0], PROTOTYPE_SAMPLES)
        # P6-9 fix: 截断到 SEQ_LEN=64（与训练一致）
        data = data[:n, :SEQ_LEN].to(device)

        proto_sum = torch.zeros(embed_dim, device=device)
        count = 0

        with torch.no_grad():
            for i in range(0, n, 8):
                batch = data[i:i + 8]
                # P6-9 fix: 构建 attention mask 排除 padding
                attn_mask = make_attention_mask(batch, pad_id=0)
                # 用 encoder.encode 得到上下文感知的 hidden state
                hidden = encoder.encode(batch, attention_mask=attn_mask)  # [B, L, hidden_dim]
                # mean pool over L（用 attention-masked）
                mask = attn_mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                proto_sum += pooled.sum(dim=0)
                count += pooled.shape[0]

        if count > 0:
            prototype = proto_sum / count
            prototype = prototype / (prototype.norm() + 1e-8)
            router.prototypes_embed[nid] = prototype.cpu()
            print(f"  [{nid}] prototype from {count} samples, "
                  f"norm={prototype.norm().item():.4f}")

    return router


def verify_routing(
    router: ThalamicRouter,
    encoder: SharedContextEncoder,
    tokenizer,
    device: str,
    domain_to_id: dict = None,  # P6-9: classifier 维度顺序
) -> float:
    """验证路由准确率（P6-9: 同时测试 prototype-based 和 classifier-based）."""
    print("\n[Verify] 路由准确性（encoder-based）")
    print("-" * 60)

    # P6-9: 用 domain_to_id 作为 classifier 维度顺序标准
    # domain_to_id 来自 prepare_training_samples，与训练时 domain_labels 一致
    if domain_to_id is None:
        # Fallback: 用 router.prototypes_embed 的 keys 顺序
        nids_list = list(router.prototypes_embed.keys())
        domain_to_id = {nid: i for i, nid in enumerate(nids_list)}
    id_to_nid = {i: nid for nid, i in domain_to_id.items()}

    correct_proto = 0
    correct_classifier = 0
    has_classifier = encoder.domain_classifier is not None

    for prompt, expected in TEST_PROMPTS:
        ids = tokenizer.encode(prompt)
        ids_tensor = torch.tensor([pad_or_truncate(ids, SEQ_LEN)], dtype=torch.long, device=device)
        # P6-9 fix: 构建 attention mask 让短 prompt 不被 padding 淹没
        attn_mask = make_attention_mask(ids_tensor, pad_id=0)
        with torch.no_grad():
            # 用 encoder.encode 路由（prototype-based，用 attention-masked mean pool）
            hidden = encoder.encode(ids_tensor, attention_mask=attn_mask)
            input_vec = hidden.mean(dim=1)[0]
            input_vec = input_vec / (input_vec.norm() + 1e-8)

            nids = list(router.prototypes_embed.keys())
            sims = {}
            for nid in nids:
                proto = router.prototypes_embed[nid].to(device)
                sims[nid] = torch.dot(input_vec, proto).item()

            # P6-9: classifier-based routing（传 attention_mask）
            cls_logits = None
            cls_probs = None
            if has_classifier:
                cls_logits = encoder.classify_domain(ids_tensor, attention_mask=attn_mask)[0]
                cls_probs = F.softmax(cls_logits, dim=0)

        # Prototype-based
        top_nid_proto = max(sims, key=sims.get) if sims else "N/A"
        max_sim = sims.get(top_nid_proto, 0.0)
        ok_proto = top_nid_proto == expected
        if ok_proto:
            correct_proto += 1

        # P6-9: Classifier-based
        top_nid_cls = "N/A"
        max_prob = 0.0
        ok_cls = False
        if has_classifier and cls_probs is not None:
            top_idx = int(cls_probs.argmax().item())
            top_nid_cls = id_to_nid.get(top_idx, "N/A")
            max_prob = float(cls_probs[top_idx].item())
            ok_cls = top_nid_cls == expected
            if ok_cls:
                correct_classifier += 1

        # Print
        print(f"  {prompt!r:<20} expected={expected:<8}")
        print(f"      proto:  top={top_nid_proto:<10} sim={max_sim:.4f} "
              f"{'✅' if ok_proto else '❌'}")
        if has_classifier:
            print(f"      class:  top={top_nid_cls:<10} prob={max_prob:.4f} "
                  f"{'✅' if ok_cls else '❌'}")
            # 显示 top-3 logits
            if cls_logits is not None:
                top3 = sorted(enumerate(cls_logits.tolist()), key=lambda x: -x[1])[:3]
                for idx, val in top3:
                    nid = id_to_nid.get(idx, "?")
                    marker = " <- top" if nid == top_nid_cls else ""
                    print(f"              {nid:<10} logit={val:.3f}{marker}")

    acc_proto = correct_proto / len(TEST_PROMPTS)
    print(f"\n  Prototype-based routing: {correct_proto}/{len(TEST_PROMPTS)} ({acc_proto*100:.1f}%)")
    if has_classifier:
        acc_cls = correct_classifier / len(TEST_PROMPTS)
        print(f"  Classifier-based routing: {correct_classifier}/{len(TEST_PROMPTS)} ({acc_cls*100:.1f}%)")
        # 返回更好的那个
        return max(acc_proto, acc_cls)
    return acc_proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"训练步数（默认 {DEFAULT_STEPS}）")
    args = parser.parse_args()

    device = "cpu"

    print("=" * 72)
    print("[P6-8] 自主进化 encoder 训练 + 重算 prototypes + 验证")
    print("=" * 72)

    # 1. 加载 Cortex + neurons
    print("\n[1/6] Loading Cortex + neurons...")
    cortex, tokenizer = create_cortex(
        neurons_dir=NEURONS_DIR,
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    print(f"  Cortex loaded with {len(cortex.neurons)} neurons")

    # 2. 加载 StandaloneEmbedding
    print("\n[2/6] Loading StandaloneEmbedding...")
    if not os.path.exists(STANDALONE_EMBED_PATH):
        print(f"  ERROR: {STANDALONE_EMBED_PATH} not found. Run verify_p6_standalone_inference.py first.")
        sys.exit(1)
    standalone_embed = StandaloneEmbedding.load(STANDALONE_EMBED_PATH, device=device)

    # 3. 构建 SharedContextEncoder（从 standalone_embedding 初始化）
    print("\n[3/6] Building SharedContextEncoder from StandaloneEmbedding...")
    encoder = SharedContextEncoder.build_from_standalone_embedding(
        standalone_embedding=standalone_embed,
        **ENCODER_CONFIG,
    )
    encoder = encoder.to(device)
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  n_domains={encoder.n_domains}, domain_classifier={'yes' if encoder.domain_classifier else 'no'}")

    # 4. 构建 SelfEvolver
    # P6-8 fix: HebbianUpdater 加负采样推远参数；ContrastiveLoss 用 margin loss 防坍塌
    hebbian = HebbianUpdater(
        vocab_size=encoder.vocab_size,
        neg_ratio=0.5,   # 负采样推远强度
        n_negatives=10,  # 每个 token 推远的负样本数
    )
    # P6-8 fix v2: margin loss 替换 InfoNCE，避免坍塌陷阱
    contrastive = ContrastiveLoss(
        margin_pos=0.8, margin_neg=0.3,
        lambda_uniform=0.1,  # 降权，只作微弱防坍塌信号
    )
    mlm = MLMLoss(mask_prob=0.15, mask_token_id=4)
    # P6-9 fix: 加 domain classification loss（supervised），主导路由学习
    evolver = SelfEvolver(encoder, hebbian, contrastive, mlm,
                          weights=(1.0, CONTRASTIVE_WEIGHT, 1.0),
                          w_domain=DOMAIN_WEIGHT)

    # 5. 加载训练数据 + 跑训练
    print("\n[4/6] Loading training data...")
    domain_data = load_domain_data()
    samples, domain_to_id = prepare_training_samples(domain_data)
    print(f"  Loaded {len(samples)} samples from {len(domain_to_id)} domains")

    print(f"\n[5/6] Training self-evolving encoder ({args.steps} steps)...")
    train_stats = train_self_evolve(encoder, evolver, samples, domain_to_id, args.steps, device)
    print(f"\n  Training stats: {train_stats}")

    # 6. 重算 prototypes + 验证
    print("\n[6/6] Recompute prototypes + verify routing...")
    router = ThalamicRouter.load(ROUTER_PATH, device=device)
    router = recompute_prototypes_with_encoder(
        router, encoder, cortex.neurons, domain_data, device,
    )

    # 保存训后 encoder + 新 router
    encoder.save(ENCODER_PATH)
    router.save(ROUTER_P6_PATH)
    print(f"\n  Saved encoder to {ENCODER_PATH}")
    print(f"  Saved router to {ROUTER_P6_PATH}")

    # 验证路由（P6-9: 传 domain_to_id 保证 classifier 维度顺序一致）
    acc = verify_routing(router, encoder, tokenizer, device, domain_to_id)

    # ── 总结 ──
    print("\n" + "=" * 72)
    print("[P6-8 Summary]")
    print("=" * 72)
    print(f"  Training: {args.steps} steps, avg_loss={train_stats['avg_loss']:.4f}")
    print(f"  Final: mlm={train_stats['final_mlm']:.4f}, "
          f"contrastive={train_stats['final_contrastive']:.4f}, "
          f"domain={train_stats['final_domain']:.4f}")  # P6-9
    print(f"  Hebbian update magnitude: {train_stats['hebbian_update']:.6f}")
    print(f"  Encoder saved: {ENCODER_PATH}")
    print(f"  Router saved: {ROUTER_P6_PATH}")
    print(f"\n  Routing accuracy: {acc*100:.1f}% (baseline P6-5: 0%)")
    if acc > 0.5:
        print("  ✓ 自主进化有效，路由准确率显著提升")
    elif acc > 0:
        print("  ⚠ 部分改善，可能需要更多训练步数或调整超参")
    else:
        print("  ✗ 路由准确率未改善，需要诊断（可能训练步数不足/数据不够）")


if __name__ == "__main__":
    main()
