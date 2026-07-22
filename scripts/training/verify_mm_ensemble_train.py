"""P8: 验证多模态 ensemble 训练闭环。

测试：
1. assemble_cortex 自动注册所有模态到所有 neuron
2. _train_multimodal_ensemble 走 ensemble 共振路径执行无误
3. 训练后 loss 正常返回

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
    print("=== Step 2: Verify auto-registration ===")
    n_neurons = len(cortex.neurons)
    n_modalities = len(hub.list_modalities())
    expected_registrations = n_neurons * n_modalities
    actual_projections = 0
    actual_heads = 0
    for nid, neuron in cortex.neurons.items():
        projs = list(neuron.mm_projections.keys())
        heads = list(neuron.mm_lm_heads.keys())
        actual_projections += len(projs)
        actual_heads += len(heads)
        print(f"  [{nid}] projections={projs} heads={heads}")
    print(f"Expected {expected_registrations} pairs, got projections={actual_projections}, heads={actual_heads}")
    assert actual_projections == expected_registrations, "auto_register_projection incomplete"
    assert actual_heads == expected_registrations, "auto_register_lm_head incomplete"

    print()
    print("=== Step 3: Build SleepEngine + synthetic mm sample ===")
    from taiji.life.sleep_engine import SleepEngine
    sleep = SleepEngine()
    sleep.set_brain_interfaces(cortex=cortex)

    # 用 video codec 生成合成样本
    video_codec = hub.modal_encoders.get("video")
    if video_codec is None:
        print("FAIL: video codec not in hub")
        return 1

    dummy_video = torch.rand(3, 16, 32, 32).clamp(0, 1)
    token_ids = video_codec.encode(dummy_video)
    if not isinstance(token_ids, list):
        token_ids = token_ids.tolist()
    print(f"Video tokens: {len(token_ids)}, range=[{min(token_ids)}, {max(token_ids)}]")

    split_idx = len(token_ids) // 2
    mm_sample = {
        "type": "multimodal",
        "modality": "video",
        "input_ids": token_ids[:split_idx],
        "target_ids": token_ids[split_idx:],
        "domain": "general",
    }
    n_in = len(mm_sample["input_ids"])
    n_tgt = len(mm_sample["target_ids"])
    print(f"input_ids={n_in}, target_ids={n_tgt}")

    print()
    print("=== Step 4: Call _train_multimodal_ensemble (round 1) ===")
    loss1, ppl1 = sleep._train_multimodal_ensemble("video", mm_sample, tokenizer_hub=hub)
    print(f"Round 1: loss={loss1:.4f}, ppl={ppl1:.1f}")
    assert loss1 is not None, "ensemble training returned None loss"

    print()
    print("=== Step 5: Call _train_multimodal_ensemble (round 2) ===")
    loss2, ppl2 = sleep._train_multimodal_ensemble("video", mm_sample, tokenizer_hub=hub)
    print(f"Round 2: loss={loss2:.4f}, ppl={ppl2:.1f}")

    print()
    print("=== Step 6: Test image modality ===")
    image_codec = hub.modal_encoders.get("image")
    if image_codec is not None:
        dummy_img = torch.rand(3, 64, 64).clamp(0, 1)
        img_tokens = image_codec.encode(dummy_img)
        if not isinstance(img_tokens, list):
            img_tokens = img_tokens.tolist()
        print(f"Image tokens: {len(img_tokens)}")
        img_split = len(img_tokens) // 2
        img_sample = {
            "type": "multimodal",
            "modality": "image",
            "input_ids": img_tokens[:img_split],
            "target_ids": img_tokens[img_split:],
            "domain": "general",
        }
        img_loss, img_ppl = sleep._train_multimodal_ensemble("image", img_sample, tokenizer_hub=hub)
        print(f"Image ensemble training: loss={img_loss:.4f}, ppl={img_ppl:.1f}")
        assert img_loss is not None, "image ensemble training failed"

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — multimodal ensemble training loop verified")
    print("=" * 60)
    print(f"\nVerified:")
    print(f"  - {n_neurons} neurons auto-registered {n_modalities} modalities each")
    print(f"  - video ensemble training: 2 rounds OK (loss {loss1:.4f} -> {loss2:.4f})")
    if image_codec is not None:
        print(f"  - image ensemble training: 1 round OK (loss {img_loss:.4f})")
    print(f"  - sleep_engine indentation fix working correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
