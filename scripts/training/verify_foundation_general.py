#!/usr/bin/env python3
"""验证 general 统一空间基座训练产物（data/foundation_v1_general）。

确认：
[1] 4 域 checkpoint + shared_lm_head + shared_embedding 完整可加载
[2] 每域 best 权重回读 PPL（collab/eval 口径：最终 shared_embedding）
[3] 配对校验：best 权重 ↔ best 步 embedding 快照（无错配崩溃）
[4] 输出空间验证：logits 在 general 256K 空间（无域词库头）

Usage:
    python scripts/training/verify_foundation_general.py
"""
import os
import sys
import math

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

SAVE_DIR = "data/foundation_v1_general"
DOMAINS = ["code", "math", "zh", "en"]


def main():
    from scripts.training.train_multi_domain_foundation import (
        load_domain_texts, load_general_tokenizer, load_tokenizer_for_vocab,
        verify_checkpoint, GENERAL_VOCAB,
    )
    from taiji.resonance.config import get_domain_neuron_config

    print("=" * 60)
    print("general 统一空间基座产物验证 (data/foundation_v1_general)")
    print("=" * 60)

    # 基础资源
    general_sp = load_general_tokenizer()
    texts = {d: load_domain_texts(d, 3000) for d in DOMAINS}
    for d in DOMAINS:
        print(f"  {d}: {len(texts[d])} 条文本")

    shared_head_path = os.path.join(SAVE_DIR, "shared_lm_head.pt")
    assert os.path.exists(shared_head_path), "[1] shared_lm_head.pt 缺失"

    # [1] + [2] 每域 best 回读（collab/eval 口径：最终 shared_embedding）
    # 注意：general 统一空间模式，输入/目标都 general 编码（与训练一致），
    # verify_checkpoint 的 sp 参数必须传 general_sp，不能传域 sp
    print("\n[1] checkpoint + 最终 embedding 回读（collab/eval 口径）...")
    results = {}
    for d in DOMAINS:
        ckpt_path = os.path.join(SAVE_DIR, f"neuron_{d}.pt")
        assert os.path.exists(ckpt_path), f"[1] {d} checkpoint 缺失"
        avg = verify_checkpoint(
            SAVE_DIR, d, general_sp, general_sp,
            torch.nn.Embedding(GENERAL_VOCAB, 512),
            texts[d], n_check=8,
            lm_head_path=shared_head_path,
        )
        results[d] = math.exp(min(avg, 20))
    print(f"  回读 PPL: { {d: round(v, 1) for d, v in results.items()} }")

    # [3] 配对校验：best 权重 ↔ best 步 embedding 快照
    print("\n[2] 配对校验（best 权重 ↔ best 步 embedding）...")
    for d in DOMAINS:
        best_emb = os.path.join(SAVE_DIR, f"shared_embedding_best_{d}.pt")
        assert os.path.exists(best_emb), f"[2] {d} best embedding 缺失"
        avg = verify_checkpoint(
            SAVE_DIR, d, general_sp, general_sp,
            torch.nn.Embedding(GENERAL_VOCAB, 512),
            texts[d], n_check=8,
            embed_path=best_emb,
            lm_head_path=shared_head_path,
        )
        paired = math.exp(min(avg, 20))
        print(f"  {d}: paired PPL={paired:.1f}")
    print("  ok [2] 配对校验全部可加载运行（无维度错配崩溃）")

    # [4] 输出空间验证：logits 在 general 256K
    print("\n[3] 输出空间验证（general 256K）...")
    import torch.nn.functional as F
    from scripts.training.train_multi_domain_foundation import batch_align_and_embed
    from taiji.resonance.neuron import ResonanceNeuron
    emb = torch.nn.Embedding(GENERAL_VOCAB, 512)
    emb.weight.data.copy_(torch.load(os.path.join(SAVE_DIR, "shared_embedding.pt"),
                                     map_location="cpu", weights_only=False))
    for d in DOMAINS:
        ckpt = torch.load(os.path.join(SAVE_DIR, f"neuron_{d}.pt"),
                          map_location="cpu", weights_only=False)
        cfg = ckpt["neuron_config"]
        cfg.unified_field_dim = None
        head = torch.nn.Linear(cfg.hidden_size, GENERAL_VOCAB, bias=False)
        head.weight.data.copy_(torch.load(shared_head_path, map_location="cpu",
                                          weights_only=False)["weight"])
        n = ResonanceNeuron(cfg, shared_lm_head=head)
        n.load_state_dict(ckpt["state_dict"], strict=False)
        n.eval()
        sp = load_tokenizer_for_vocab(d, get_domain_neuron_config(d, spec="compact").vocab_size)
        out = batch_align_and_embed([texts[d][0]], sp, general_sp, emb, max_seq_len=64)
        with torch.no_grad():
            r = n.forward(out[0], return_logits=True)
        logits = r["logits"]
        assert logits.shape[-1] == GENERAL_VOCAB, \
            f"[3] {d} logits vocab={logits.shape[-1]} != {GENERAL_VOCAB}"
        top_id = int(logits[0, -1].argmax().item())
        top_piece = general_sp.IdToPiece(top_id)
        print(f"  {d}: logits {tuple(logits.shape)}, top token='{top_piece}'")
    print("  ok [3] 所有域输出在 general 256K 空间（共享 lm_head 生效）")

    print(f"\n{'='*60}")
    print(f"产物验证完成: 回读 PPL = { {d: round(v, 1) for d, v in results.items()} }")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
