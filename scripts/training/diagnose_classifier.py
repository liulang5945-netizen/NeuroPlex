"""P6-9 诊断：classifier 在短 prompt 上坍塌的根因分析.

假设：
  - 训练数据是 64-token 真实序列（domain_loss=0.0014 极低）
  - 测试 prompt 是 4-5 tokens，padded 到 64
  - mean pool 被 padding 淹没，导致 classifier 看到的是 "padding 表征" 而非 "内容表征"

验证：
  1. 在真实训练数据上测试 classifier（应该准确）
  2. 在短 prompt 上测试 classifier（已知坍塌）
  3. 在更长 prompt 上测试 classifier（验证是否长度问题）
  4. 对比 mean pool vs first-token pool
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_libs"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.nn.functional as F
from taiji.loader import create_cortex
from taiji.resonance import SharedContextEncoder


ENCODER_PATH = "data/distill/shared_context_encoder.pt"
DOMAIN_DATA_PATH = "data/distill/domain_datasets.pt"

DOMAINS = ["zh", "en", "code", "math", "general"]

SHORT_PROMPTS = [
    ("你好", "zh"),
    ("1+1=", "math"),
    ("hello world", "en"),
    ("def fibonacci", "code"),
]

LONGER_PROMPTS = [
    ("你好，今天天气怎么样？我想了解一下北京的情况。", "zh"),
    ("Please calculate 1+1= and explain the mathematical reasoning step by step.", "math"),
    ("Hello world, this is a simple test of the emergency broadcast system.", "en"),
    ("def fibonacci(n): if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2)", "code"),
]


def pad_or_truncate(ids, seq_len):
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [0] * (seq_len - len(ids))


def make_tokenizer():
    """简易 tokenizer：按 UTF-8 bytes 切分（与训练数据一致的 fallback）."""
    def encode_fn(text):
        return list(text.encode("utf-8"))
    obj = type("T", (), {})()
    obj.encode = encode_fn
    return obj


def test_classifier_on_batch(encoder, input_ids, label, device, domain_to_id):
    """测试 classifier 在一个 batch 上的表现，对比 mean pool vs first-token."""
    id_to_nid = {i: nid for nid, i in domain_to_id.items()}
    expected_id = domain_to_id[label]

    with torch.no_grad():
        # mean pool (当前 forward 行为)
        _, hidden, domain_logits_mean = encoder(
            input_ids, return_hidden=True, return_domain_logits=True
        )
        # first-token pool
        h_first = hidden[:, 0]  # [B, hidden_dim]
        domain_logits_first = encoder.domain_classifier(h_first) if encoder.domain_classifier else None
        # last non-padding token pool
        # 找最后一个非 0 的 token
        mask = (input_ids != 0).sum(dim=1) - 1  # [B] last non-pad index
        h_last_list = []
        for b in range(input_ids.shape[0]):
            idx = mask[b].item()
            h_last_list.append(hidden[b, idx])
        h_last = torch.stack(h_last_list)  # [B, hidden_dim]
        domain_logits_last = encoder.domain_classifier(h_last) if encoder.domain_classifier else None

    print(f"  [{label}] expected_id={expected_id} ({id_to_nid[expected_id]})")
    for name, logits in [("mean", domain_logits_mean), ("first", domain_logits_first), ("last", domain_logits_last)]:
        if logits is None:
            continue
        probs = F.softmax(logits[0], dim=0)
        top_idx = int(probs.argmax().item())
        top_nid = id_to_nid.get(top_idx, "?")
        ok = top_idx == expected_id
        # top-3
        top3 = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{id_to_nid.get(i,'?')}={p:.3f}" for i, p in top3)
        print(f"    {name:>5}: top={top_nid:<8} {'✅' if ok else '❌'}  top3: {top3_str}")


def main():
    device = "cpu"
    print(f"Loading encoder from {ENCODER_PATH}...")
    encoder = SharedContextEncoder.load(ENCODER_PATH, device=device)
    encoder.eval()
    print(f"  n_domains={encoder.n_domains}, classifier={'yes' if encoder.domain_classifier else 'no'}")

    print(f"\nLoading domain data from {DOMAIN_DATA_PATH}...")
    domain_data = torch.load(DOMAIN_DATA_PATH, map_location="cpu", weights_only=False)
    print(f"  domains: {list(domain_data.keys())}")
    for nid, data in domain_data.items():
        print(f"    {nid}: {data.shape}")

    # P6-9 fix: 使用 domain_data.keys() 顺序构建 domain_to_id（与训练一致）
    domain_to_id = {nid: i for i, nid in enumerate(domain_data.keys())}
    id_to_nid = {i: nid for nid, i in domain_to_id.items()}
    print(f"  domain_to_id (training order): {domain_to_id}")

    # P6-9 fix: 用 create_cortex 获取正确的 tokenizer（TaijiNativeTokenizerV2）
    print("\nLoading Taiji tokenizer via create_cortex()...")
    _, tokenizer = create_cortex(
        neurons_dir="data/neurons",
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    SEQ_LEN = 64

    # ─── 1. 真实训练数据样本 ───
    print("\n" + "=" * 70)
    print("[1] Classifier on REAL training data samples (should be accurate)")
    print("=" * 70)
    correct = 0
    total = 0
    for nid in DOMAINS:
        data = domain_data[nid]
        # P6-9 fix: 截断到 SEQ_LEN=64（与训练一致）
        samples = data[:10, :SEQ_LEN].to(device)
        with torch.no_grad():
            logits = encoder.classify_domain(samples)
            preds = logits.argmax(dim=1)
        expected_id = domain_to_id[nid]
        n_correct = (preds == expected_id).sum().item()
        correct += n_correct
        total += len(samples)
        print(f"  [{nid}] {n_correct}/{len(samples)} correct (expected_id={expected_id})")
        # 显示第一个样本的 logits
        probs = F.softmax(logits[0], dim=0)
        top3 = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{id_to_nid.get(i,'?')}={p:.3f}" for i, p in top3)
        print(f"    sample[0] top3: {top3_str}")
    print(f"\n  Total: {correct}/{total} ({100*correct/max(total,1):.1f}%)")

    # ─── 2. 短 prompt（已知坍塌）───
    print("\n" + "=" * 70)
    print("[2] Classifier on SHORT prompts (known to collapse)")
    print("=" * 70)
    for prompt, expected in SHORT_PROMPTS:
        ids = tokenizer.encode(prompt)
        ids_tensor = torch.tensor([pad_or_truncate(ids, SEQ_LEN)], dtype=torch.long, device=device)
        print(f"\n  prompt={prompt!r} (len={len(ids)} tokens) expected={expected}")
        test_classifier_on_batch(encoder, ids_tensor, expected, device, domain_to_id)

    # ─── 3. 更长 prompt ───
    print("\n" + "=" * 70)
    print("[3] Classifier on LONGER prompts (test length hypothesis)")
    print("=" * 70)
    for prompt, expected in LONGER_PROMPTS:
        ids = tokenizer.encode(prompt)
        ids_tensor = torch.tensor([pad_or_truncate(ids, SEQ_LEN)], dtype=torch.long, device=device)
        print(f"\n  prompt={prompt[:40]!r}... (len={len(ids)} tokens) expected={expected}")
        test_classifier_on_batch(encoder, ids_tensor, expected, device, domain_to_id)

    # ─── 4. 直接用训练数据的 hidden state 检查 domain 区分度 ───
    print("\n" + "=" * 70)
    print("[4] Hidden state mean per domain (check domain separation)")
    print("=" * 70)
    domain_means = {}
    for nid in DOMAINS:
        data = domain_data[nid][:20, :SEQ_LEN].to(device)
        with torch.no_grad():
            hidden = encoder.encode(data)
            mean = hidden.mean(dim=(0, 1))  # [hidden_dim]
            domain_means[nid] = mean
            print(f"  [{nid}] mean norm={mean.norm().item():.4f}")

    print("\n  Pairwise cosine similarity:")
    for i, n1 in enumerate(DOMAINS):
        for n2 in DOMAINS[i+1:]:
            v1 = domain_means[n1] / (domain_means[n1].norm() + 1e-8)
            v2 = domain_means[n2] / (domain_means[n2].norm() + 1e-8)
            sim = torch.dot(v1, v2).item()
            print(f"    {n1:>8} vs {n2:<8}: {sim:.4f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
