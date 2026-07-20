"""P2-1.5: 诊断 Cortex.generate 乱码根因。

核心问题：5 个 neuron 蒸馏后 self-PPL=1.1（很好），
但 Cortex.generate 输出乱码。诊断方向：

1. 单 neuron vs ensemble logits 对比
   - 每个 neuron 在测试 prompt 上的 entropy 分布
   - 是否所有 neuron 都不自信（导致加权稀释）？
   - 还是被错误 neuron 主导？

2. Per-position routing 是否有效
   - 每个 position 实际选中哪个 neuron？
   - position_weights 分布是否合理？

3. Token ID 输出范围检查
   - top-k 采样的 token ID 分布
   - 是否落在 sentencepiece vocab 范围内？

4. field_state 演化
   - round 1 vs round 2 的 field_state 变化
   - resonance 真的在发生吗？

运行：
    python scripts/training/diagnose_cortex_garbage.py
"""
from __future__ import annotations

import os
import sys
import math

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


def main():
    print("=" * 72)
    print("P2-1.5: Cortex.generate 乱码根因诊断")
    print("=" * 72)

    # ── 加载 Cortex + teacher ──
    print("\n[Setup] 加载 Cortex + teacher ...")
    from taiji.loader import create_cortex
    from taiji.training.checkpoint_bridge import load_teacher_model
    from taiji.resonance.shared_embed import SharedEmbedProj

    cortex, tokenizer = create_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        enable_gating=False,
    )
    teacher, shared_embedding = load_teacher_model("checkpoint-481000", device="cpu")
    shared_proj = SharedEmbedProj.load("data/distill/shared_proj.pt", 2048, 512)
    cortex.set_teacher_pipeline(teacher_model=teacher, shared_proj=shared_proj)
    cortex.set_tokenizer(tokenizer)

    # ── 加载 DomainRouter 并注入 ensemble ──
    from taiji.resonance import DomainRouter
    router_path = "data/distill/domain_anchors.pt"
    if os.path.exists(router_path):
        router = DomainRouter.load(router_path)
        cortex.ensemble.domain_router = router
        print(f"\n[Info] DomainRouter 已注入 ensemble (temperature={router.temperature})")
    else:
        print(f"\n[Warn] DomainRouter 未找到: {router_path}")
        print(f"       请先运行: python scripts/training/compute_domain_anchors.py")
        router = None

    # ── 测试 prompt ──
    test_prompts = ["你好", "1+1=", "hello world", "def fibonacci"]
    sp_piece_size = tokenizer.sp.GetPieceSize()
    text_offset = tokenizer.text_offset
    print(f"\n[Info] sentencepiece vocab size: {sp_piece_size}")
    print(f"[Info] text_offset: {text_offset}")
    print(f"[Info] contract text_vocab_size: {tokenizer.text_vocab_size}")
    print(f"[Info] valid text token range: [{text_offset}, {text_offset + sp_piece_size})")

    for prompt in test_prompts:
        print("\n" + "=" * 72)
        print(f"[Diagnose] prompt = '{prompt}'")
        print("=" * 72)

        ids = tokenizer.encode(prompt)
        print(f"  encoded: {len(ids)} tokens, IDs = {ids[:10]}{'...' if len(ids) > 10 else ''}")
        input_ids = torch.tensor([ids], dtype=torch.long)

        # ── 1. 每个 neuron 单独 forward ──
        print("\n  ── 1. 单 neuron logits 诊断 ──")
        with torch.no_grad():
            shared_emb = cortex._embed_pipeline(input_ids)
            print(f"  shared_emb shape: {shared_emb.shape}, range [{shared_emb.min():.3f}, {shared_emb.max():.3f}]")

            neuron_logits = {}
            neuron_entropies = {}
            for nid, neuron in cortex.neurons.items():
                result = neuron.forward(shared_emb, return_logits=True)
                logits = result["logits"]  # [1, L, V]
                # 只看最后一个 position（generate 时用的）
                last_logits = logits[0, -1, :]  # [V]
                probs = F.softmax(last_logits, dim=-1)
                log_probs = F.log_softmax(last_logits, dim=-1)
                ent = -(probs * log_probs).sum().item()
                max_prob = probs.max().item()
                top5 = torch.topk(last_logits, 5)

                neuron_logits[nid] = last_logits
                neuron_entropies[nid] = ent

                # top-5 token 是否在 sp vocab 范围内
                top5_ids = top5.indices.tolist()
                top5_in_range = sum(1 for i in top5_ids if text_offset <= i < text_offset + sp_piece_size)

                print(f"  [{nid:>8}] entropy={ent:.2f}, max_prob={max_prob:.4f}, "
                      f"top-5 in sp_vocab: {top5_in_range}/5")
                print(f"             top-5 IDs: {top5_ids}")
                # decode top-5
                top5_decoded = []
                for tid in top5_ids:
                    if text_offset <= tid < text_offset + sp_piece_size:
                        top5_decoded.append(tokenizer.sp.IdToPiece(tid - text_offset))
                    elif tid in tokenizer.special_id_to_text:
                        top5_decoded.append(f"<special:{tokenizer.special_id_to_text[tid]}>")
                    else:
                        top5_decoded.append(f"<oov:{tid}>")
                print(f"             top-5 tokens: {top5_decoded}")

        # ── 2. ensemble forward + 分析 weighted_logits ──
        print("\n  ── 2. ensemble 共振 + 加权 logits ──")
        state = cortex.think(input_ids)
        weighted_logits = state["weighted_logits"]  # [1, L, V]
        final_scores = state["final_scores"]
        final_weights = state.get("final_weights", {})
        n_rounds = state["n_rounds"]
        skipped = state.get("skipped_resonance", False)

        print(f"  n_rounds: {n_rounds}, skipped_resonance: {skipped}")
        print(f"  final_scores (resonance): {dict(final_scores)}")
        print(f"  final_weights (logit 权重): {dict(final_weights)}")

        # 分析 weighted_logits 最后一个 position
        last_weighted = weighted_logits[0, -1, :]  # [V]
        last_probs = F.softmax(last_weighted, dim=-1)
        last_ent = -(last_probs * last_probs.log()).sum().item()
        last_max_prob = last_probs.max().item()
        top10 = torch.topk(last_weighted, 10)
        top10_ids = top10.indices.tolist()

        print(f"\n  weighted_logits 最后 position:")
        print(f"    entropy={last_ent:.2f}, max_prob={last_max_prob:.4f}")
        top10_in_range = sum(1 for i in top10_ids if text_offset <= i < text_offset + sp_piece_size)
        print(f"    top-10 in sp_vocab: {top10_in_range}/10")
        print(f"    top-10 IDs: {top10_ids}")

        # decode top-10
        top10_decoded = []
        for tid in top10_ids:
            if text_offset <= tid < text_offset + sp_piece_size:
                top10_decoded.append(tokenizer.sp.IdToPiece(tid - text_offset))
            elif tid in tokenizer.special_id_to_text:
                top10_decoded.append(f"<special:{tokenizer.special_id_to_text[tid]}>")
            else:
                top10_decoded.append(f"<oov:{tid}>")
        print(f"    top-10 tokens: {top10_decoded}")

        # ── 3. 跨 neuron logits 一致性 ──
        print("\n  ── 3. 跨 neuron 一致性（关键诊断） ──")
        # 看 top-1 prediction 在不同 neuron 之间是否一致
        top1_per_neuron = {nid: lg.argmax().item() for nid, lg in neuron_logits.items()}
        top1_weighted = last_weighted.argmax().item()
        print(f"  各 neuron top-1: {top1_per_neuron}")
        print(f"  weighted top-1: {top1_weighted}")

        # top-1 在各 neuron 中是否一致
        unique_top1 = set(top1_per_neuron.values())
        if len(unique_top1) == 1:
            print(f"  ✅ 所有 neuron top-1 一致: {unique_top1.pop()}")
        else:
            print(f"  ❌ neuron top-1 分歧（{len(unique_top1)} 个不同）")
            # decode 每个
            for nid, tid in top1_per_neuron.items():
                if text_offset <= tid < text_offset + sp_piece_size:
                    piece = tokenizer.sp.IdToPiece(tid - text_offset)
                elif tid in tokenizer.special_id_to_text:
                    piece = f"<special:{tokenizer.special_id_to_text[tid]}>"
                else:
                    piece = f"<oov:{tid}>"
                print(f"     {nid}: {tid} -> '{piece}'")

        # ── 5. 关键结论 ──
        print("\n  ── 5. DomainRouter 路由诊断 ──")
        if router is not None:
            # 计算每个 neuron 当前 field_vector 与自己 anchor 的相似度
            sims_to_own = {}
            for nid in cortex.neurons.keys():
                fv = None
                # 重新跑一次 forward 拿 field_vector
                with torch.no_grad():
                    shared_emb_local = cortex._embed_pipeline(input_ids)
                    result_local = cortex.neurons[nid].forward(shared_emb_local, return_logits=False)
                    fv = result_local["field_vector"]
                sim = router.similarity(nid, fv)
                sims_to_own[nid] = sim
                print(f"  [{nid:>8}] field_vec × own_anchor sim = {sim:+.4f}")

            # 用相似度做 softmax 看权重
            nids = list(sims_to_own.keys())
            sim_tensor = torch.tensor([sims_to_own[n] for n in nids])
            dr_weights = F.softmax(sim_tensor / router.temperature, dim=0)
            print(f"\n  DomainRouter 权重 (temperature={router.temperature}):")
            for i, nid in enumerate(nids):
                print(f"    {nid:>8}: {dr_weights[i].item():.4f}")

            # 对比原 final_weights（entropy-based）
            print(f"\n  原 entropy-based final_weights:")
            for nid, w in final_weights.items():
                print(f"    {nid:>8}: {w:.4f}")

            # 路由是否正确（最相似的 domain 应该匹配 prompt 类型）
            best_dr = nids[dr_weights.argmax().item()]
            best_entropy = max(final_weights, key=final_weights.get)
            print(f"\n  DomainRouter 最佳: {best_dr}")
            print(f"  Entropy 最佳:      {best_entropy}")

            # 简单 prompt-domain 映射（用于验证）
            expected_domain = {
                "你好": "zh", "1+1=": "math",
                "hello world": "en", "def fibonacci": "code",
            }.get(prompt, "general")
            print(f"  期望域:            {expected_domain}")
            print(f"  DomainRouter 正确: {'✅' if best_dr == expected_domain else '❌'}")
            print(f"  Entropy 正确:      {'✅' if best_entropy == expected_domain else '❌'}")

    # ── 5. 关键结论 ──
    print("\n" + "=" * 72)
    print("[Conclusion] DomainRouter vs Entropy-based Routing 对比")
    print("=" * 72)

    # 实际 generate 对比：DomainRouter vs 关闭 DomainRouter
    print("\n[Generate 对比] DomainRouter ON vs OFF")
    test_prompts_short = ["你好", "1+1=", "hello world", "def fibonacci"]
    for prompt in test_prompts_short:
        # DomainRouter ON
        cortex.ensemble.domain_router = router
        out_dr = cortex.generate(prompt, max_tokens=20)

        # DomainRouter OFF（走原 entropy-based）
        cortex.ensemble.domain_router = None
        out_entropy = cortex.generate(prompt, max_tokens=20)

        # 恢复
        cortex.ensemble.domain_router = router

        print(f"\n  prompt: '{prompt}'")
        print(f"    DR ON:     '{out_dr}'")
        print(f"    DR OFF:    '{out_entropy}'")


if __name__ == "__main__":
    main()
