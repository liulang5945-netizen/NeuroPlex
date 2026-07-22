"""路由诊断：对比两个 prototype 文件的路由准确率.

测试三种配置：
1. thalamic_prototypes.pt（旧版，standalone_embedding 算的 prototypes_embed）
2. thalamic_prototypes_p6.pt（P6-8 训练后，encoder.encode 算的 prototypes_embed）
3. 同样 p6 prototypes，但提高 temperature（0.1 → 0.5）

输出每配置的路由准确率 + 每 prompt 的 sim 分布。
"""
from __future__ import annotations

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
from taiji.resonance import StandaloneEmbedding, SharedContextEncoder, ThalamicRouter


TEST_PROMPTS = [
    ("你好", "zh"),
    ("1+1=", "math"),
    ("hello world", "en"),
    ("def fibonacci", "code"),
    ("今天天气", "general"),
]

SEQ_LEN = 64


def pad_or_truncate(ids, n):
    if len(ids) >= n:
        return ids[:n]
    return ids + [0] * (n - len(ids))


def load_tokenizer():
    """加载 tokenizer（用 standalone_embedding 的 v2 tokenizer）."""
    from taiji.loader import TaijiNativeTokenizerV2
    sp_path = "E:/taiji-neuron/checkpoint-481000/sentencepiece.model"
    tok = TaijiNativeTokenizerV2(sp_path)
    return tok


def encode_prompt(prompt, tokenizer, encoder, device, pooling="mean"):
    """用 encoder.encode() 得到 input_vec（支持 mean / last / max pooling）."""
    ids = tokenizer.encode(prompt)
    ids = pad_or_truncate(ids, SEQ_LEN)
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = encoder.encode(ids_tensor)  # [1, L, hidden_dim]
        if pooling == "mean":
            pooled = hidden.mean(dim=1)  # [1, hidden_dim]
        elif pooling == "last":
            pooled = hidden[:, -1, :]  # [1, hidden_dim]
        elif pooling == "max":
            pooled = hidden.max(dim=1)[0]  # [1, hidden_dim]
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        input_vec = pooled[0]  # [hidden_dim]
        input_vec = input_vec / (input_vec.norm() + 1e-8)
    return input_vec


def recompute_prototypes(router, encoder, domain_data, device, pooling="mean"):
    """用指定 pooling 重算 prototypes_embed."""
    embed_dim = encoder.hidden_dim
    new_protos = {}
    for nid, data in domain_data.items():
        if data.shape[0] == 0:
            continue
        n = min(data.shape[0], 100)
        data = data[:n].to(device)
        proto_sum = torch.zeros(embed_dim, device=device)
        count = 0
        with torch.no_grad():
            for i in range(0, n, 8):
                batch = data[i:i + 8]
                hidden = encoder.encode(batch)
                if pooling == "mean":
                    pooled = hidden.mean(dim=1)
                elif pooling == "last":
                    pooled = hidden[:, -1, :]
                elif pooling == "max":
                    pooled = hidden.max(dim=1)[0]
                proto_sum += pooled.sum(dim=0)
                count += pooled.shape[0]
        if count > 0:
            proto = proto_sum / count
            proto = proto / (proto.norm() + 1e-8)
            new_protos[nid] = proto.cpu()
    router.prototypes_embed = new_protos
    return router


def test_routing(router, encoder, tokenizer, device, label, pooling="mean"):
    """测试路由准确率."""
    print(f"\n{'='*70}")
    print(f"[{label}]")
    print(f"{'='*70}")
    print(f"  temperature={router.temperature}, pooling={pooling}")
    print(f"  prototypes_embed keys: {list(router.prototypes_embed.keys())}")

    correct = 0
    for prompt, expected in TEST_PROMPTS:
        input_vec = encode_prompt(prompt, tokenizer, encoder, device, pooling=pooling)
        sims = {}
        for nid, proto in router.prototypes_embed.items():
            proto = proto.to(device)
            sims[nid] = torch.dot(input_vec, proto).item()

        top_nid = max(sims, key=sims.get) if sims else "N/A"
        max_sim = sims.get(top_nid, 0.0)
        ok = top_nid == expected
        if ok:
            correct += 1

        print(f"  {prompt!r:<20} expected={expected:<8} -> top={top_nid:<10} "
              f"sim={max_sim:.4f} {'✅' if ok else '❌'}")
        # 显示 top-3
        for nid, sim in sorted(sims.items(), key=lambda x: -x[1])[:3]:
            marker = " <- top" if nid == top_nid else ""
            print(f"      {nid:<14} sim={sim:.4f}{marker}")

    acc = correct / len(TEST_PROMPTS)
    print(f"\n  Routing accuracy: {correct}/{len(TEST_PROMPTS)} ({acc*100:.1f}%)")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["p6", "standalone"], default="p6",
                        help="用哪个 encoder 路由（p6=SharedContextEncoder, standalone=未训练 embedding）")
    args = parser.parse_args()

    device = "cpu"

    print("=" * 70)
    print("路由诊断：对比 prototype 文件 + encoder 选择")
    print("=" * 70)

    # 加载 tokenizer
    tokenizer = load_tokenizer()
    print(f"Tokenizer loaded")

    # 加载 encoder
    if args.encoder == "p6":
        encoder_path = "data/distill/shared_context_encoder.pt"
        print(f"\nLoading SharedContextEncoder from {encoder_path}...")
        encoder = SharedContextEncoder.load(encoder_path, device=device)
    else:
        standalone_path = "data/distill/standalone_embedding.pt"
        print(f"\nLoading StandaloneEmbedding from {standalone_path}...")
        standalone = StandaloneEmbedding.load(standalone_path, device=device)
        # 用 StandaloneEmbedding 构建 encoder（未训练 transformer）
        encoder = SharedContextEncoder.build_from_standalone_embedding(
            standalone_embedding=standalone,
            hidden_dim=512, num_layers=2, num_heads=8, num_kv_heads=2,
            intermediate_size=1408,
        ).to(device)
        encoder.eval()
        print(f"  Built untrained SharedContextEncoder from StandaloneEmbedding")

    print(f"Encoder: {encoder.__class__.__name__}, hidden_dim={encoder.hidden_dim}")

    # 加载 domain_data 用于重算 prototype
    print("\nLoading domain_data for prototype recompute...")
    domain_data = torch.load("data/distill/domain_datasets.pt",
                            map_location="cpu", weights_only=False)
    print(f"  domains: {list(domain_data.keys())}")

    p6_router_path = "data/distill/thalamic_prototypes_p6.pt"

    # 对每种 pooling 策略测试
    for pooling in ["mean", "last", "max"]:
        # 用 P6 prototypes 作为基础，但重算 prototype 用当前 pooling
        if os.path.exists(p6_router_path):
            router = ThalamicRouter.load(p6_router_path, device=device)
            # 用当前 pooling 重算 prototype（保证 input 和 prototype 用同样 pooling）
            router = recompute_prototypes(router, encoder, domain_data, device, pooling=pooling)
            test_routing(router, encoder, tokenizer, device,
                         f"encoder={args.encoder}, pooling={pooling} (重算 prototype)",
                         pooling=pooling)


if __name__ == "__main__":
    main()
