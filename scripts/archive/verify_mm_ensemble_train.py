"""P8: 验证多模态 ensemble 训练闭环。

测试：
1. assemble_cortex 自动注册所有模态的输入投影到所有 neuron
2. _train_multimodal_ensemble 走 ensemble 共振路径执行无误
3. 训练后 loss 正常返回

2026-08-07 收敛后：
- mm_lm_heads 已废弃，输出统一走共享 general lm_head（256K vocab）
- target 映射到 general 词表 codec 段（base + codec_index）
- image/audio 支持 ensemble 训练；video 无 general 词表预留段，v1 不支持

Usage:
    python scripts/training/verify_mm_ensemble_train.py
"""
from __future__ import annotations

import os
import sys
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print = functools.partial(print, flush=True)

import torch


def main():
    print("=== Step 1: assemble_cortex ===")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons", device="cpu", max_rounds=2,
    )
    print(f"Neurons: {list(cortex.neurons.keys())}")
    print(f"Modules: {list(modules.keys())}")

    hub = modules.get("tokenizer_hub")
    if hub is None:
        print("FAIL: tokenizer_hub not in modules")
        return 1
    print(f"Hub modalities: {hub.list_modalities()}")

    print()
    print("=== Step 2: Verify auto-registration (input projections only) ===")
    n_neurons = len(cortex.neurons)
    n_modalities = len(hub.list_modalities())
    expected_registrations = n_neurons * n_modalities
    actual_projections = 0
    for nid, neuron in cortex.neurons.items():
        projs = list(neuron.mm_projections.keys())
        actual_projections += len(projs)
        print(f"  [{nid}] projections={projs}")
    print(f"Expected {expected_registrations} projections, got {actual_projections}")
    assert actual_projections == expected_registrations, "auto_register_projection incomplete"

    print()
    print("=== Step 3: Build SleepEngine + synthetic image sample ===")
    from taiji.life.sleep_engine import SleepEngine
    sleep = SleepEngine()
    sleep.set_brain_interfaces(cortex=cortex)

    # v1 收敛：用 image codec 生成合成样本（image 有 general 词表预留段）
    image_codec = hub.modal_encoders.get("image")
    if image_codec is None:
        print("FAIL: image codec not in hub")
        return 1

    dummy_img = torch.rand(3, 64, 64).clamp(0, 1)
    token_ids = image_codec.encode(dummy_img)
    if not isinstance(token_ids, list):
        token_ids = token_ids.tolist()
    print(f"Image tokens: {len(token_ids)}, range=[{min(token_ids)}, {max(token_ids)}]")

    # 截断 token 数避免内存爆炸：seq_len × 256K vocab × n_neurons 共振
    # 6 neuron × 128 seq × 256K × 4B ≈ 0.8GB（安全）
    MAX_MM_TOKENS = 128
    if len(token_ids) > MAX_MM_TOKENS:
        token_ids = token_ids[:MAX_MM_TOKENS]
        print(f"  truncated to {MAX_MM_TOKENS} tokens (memory safety)")

    split_idx = len(token_ids) // 2
    mm_sample = {
        "type": "multimodal",
        "modality": "image",
        "input_ids": token_ids[:split_idx],
        "target_ids": token_ids[split_idx:],
        "domain": "general",
    }
    n_in = len(mm_sample["input_ids"])
    n_tgt = len(mm_sample["target_ids"])
    print(f"input_ids={n_in}, target_ids={n_tgt}")

    print()
    print("=== Step 4: Call _train_multimodal_ensemble (round 1) ===")
    loss1, ppl1 = sleep._train_multimodal_ensemble("image", mm_sample, tokenizer_hub=hub)
    print(f"Round 1: loss={loss1:.4f}, ppl={ppl1:.1f}")
    assert loss1 is not None, "ensemble training returned None loss"

    print()
    print("=== Step 5: Call _train_multimodal_ensemble (round 2) ===")
    loss2, ppl2 = sleep._train_multimodal_ensemble("image", mm_sample, tokenizer_hub=hub)
    print(f"Round 2: loss={loss2:.4f}, ppl={ppl2:.1f}")

    print()
    print("=== Step 6: Test audio modality ===")
    audio_codec = hub.modal_encoders.get("audio")
    if audio_codec is not None:
        dummy_audio = torch.rand(16000).clamp(-1, 1)
        aud_tokens = audio_codec.encode(dummy_audio)
        if not isinstance(aud_tokens, list):
            aud_tokens = aud_tokens.tolist()
        print(f"Audio tokens: {len(aud_tokens)}")
        # 同样截断保护
        if len(aud_tokens) > MAX_MM_TOKENS:
            aud_tokens = aud_tokens[:MAX_MM_TOKENS]
            print(f"  truncated to {MAX_MM_TOKENS} tokens (memory safety)")
        aud_split = len(aud_tokens) // 2
        aud_sample = {
            "type": "multimodal",
            "modality": "audio",
            "input_ids": aud_tokens[:aud_split],
            "target_ids": aud_tokens[aud_split:],
            "domain": "general",
        }
        aud_loss, aud_ppl = sleep._train_multimodal_ensemble("audio", aud_sample, tokenizer_hub=hub)
        print(f"Audio ensemble training: loss={aud_loss:.4f}, ppl={aud_ppl:.1f}")
        assert aud_loss is not None, "audio ensemble training failed"

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — multimodal ensemble training loop verified")
    print("=" * 60)
    print(f"\nVerified:")
    print(f"  - {n_neurons} neurons auto-registered {n_modalities} modalities (input projections)")
    print(f"  - image ensemble training: 2 rounds OK (loss {loss1:.4f} -> {loss2:.4f})")
    if audio_codec is not None:
        print(f"  - audio ensemble training: 1 round OK (loss={aud_loss:.4f})")
    print(f"  - 2026-08-07 收敛：输出统一走共享 general lm_head（256K vocab）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
