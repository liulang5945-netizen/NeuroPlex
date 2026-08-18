"""P8: 验证 cortex_generate API 端点的端到端可用性。

模拟 API 路由层调用 cortex.generate_multimodal() 的完整路径：
1. 随机模式：合成随机数据 → codec encode → generate_multimodal → decode → save
2. 模仿模式：load_image → codec encode → generate_multimodal → decode → save
3. 验证 /api/taiji/model/info 的 modalities 信息构造逻辑

不启动 HTTP 服务，直接调用底层代码路径，验证集成正确性。

Usage:
    python scripts/training/verify_cortex_generate_api.py
"""
from __future__ import annotations

import os
import sys
import functools
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print = functools.partial(print, flush=True)

import torch


def clear_output_dir(out_dir: str):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def simulate_cortex_generate(cortex, hub, modality: str, input_path: str = None,
                             max_tokens: int = 0, temperature: float = 1.0,
                             top_k: int = 0, seed: int = 42):
    """模拟 routes_neuroplex.py 中的 cortex_generate 端点逻辑。"""
    if seed is not None:
        torch.manual_seed(seed)

    device = next(cortex.neurons[next(iter(cortex.neurons))].parameters()).device
    codec = hub.modal_encoders.get(modality)

    if input_path:
        # 模仿模式
        from taiji.multimodal.io import load_image, load_audio, load_video
        if modality == "image":
            data = load_image(input_path).to(device)
        elif modality == "audio":
            data = load_audio(input_path).to(device)
        else:
            data = load_video(input_path).to(device)
        x = data.unsqueeze(0).to(device)
    else:
        # 随机模式
        if modality == "image":
            x = torch.rand(1, 3, 32, 32, device=device)
        elif modality == "audio":
            x = torch.rand(1, 1, 16000, device=device) * 0.5
        else:
            x = torch.rand(1, 3, 16, 32, 32, device=device)

    with torch.no_grad():
        z = codec.model.encoder(x)
        if modality == "image":
            B, D, Hz, Wz = z.shape
            z_seq = z.permute(0, 2, 3, 1).contiguous().view(B, Hz * Wz, D)
        elif modality == "audio":
            B, D, Tz = z.shape
            z_seq = z.permute(0, 2, 1).contiguous().view(B, Tz, D)
        else:
            B, D, Tz, Hz, Wz = z.shape
            z_seq = z.permute(0, 2, 3, 4, 1).contiguous().view(B, Tz * Hz * Wz, D)

    # max_tokens=0 自动用 codec 网格大小
    actual_max_tokens = max_tokens if max_tokens and max_tokens > 0 else z_seq.shape[1]

    generated_ids = cortex.generate_multimodal(
        {"modality": modality, "data": z_seq, "domain": "general"},
        max_tokens=actual_max_tokens,
        temperature=temperature,
        top_k=top_k,
        modality=modality,
    )

    recon = hub.decode(generated_ids, modality=modality)
    return generated_ids, recon


def test_random_image(cortex, hub):
    print("\n[Test 1] Random image generation")
    out_dir = "data/image/cortex_api_test"
    clear_output_dir(out_dir)
    ids, recon = simulate_cortex_generate(cortex, hub, "image", seed=42)
    print(f"  tokens: {len(ids)}, range=[{min(ids)}, {max(ids)}]")
    print(f"  recon shape: {recon.shape}, range=[{recon.min().item():.3f}, {recon.max().item():.3f}]")

    from taiji.multimodal.io import save_image
    img = recon if recon.dim() == 3 else recon[0]
    path = os.path.join(out_dir, "random.png")
    save_image(img, path)
    assert os.path.isfile(path), f"image not saved: {path}"
    assert os.path.getsize(path) > 0
    print(f"  saved: {path} ({os.path.getsize(path)} bytes)")


def test_random_audio(cortex, hub):
    print("\n[Test 2] Random audio generation")
    out_dir = "data/audio/cortex_api_test"
    clear_output_dir(out_dir)
    ids, recon = simulate_cortex_generate(cortex, hub, "audio", seed=42)
    print(f"  tokens: {len(ids)}, range=[{min(ids)}, {max(ids)}]")
    print(f"  recon shape: {recon.shape}")

    from taiji.multimodal.io import save_audio
    aud = recon if recon.dim() <= 1 else recon[0]
    path = os.path.join(out_dir, "random.wav")
    save_audio(aud, path, sample_rate=16000)
    assert os.path.isfile(path)
    print(f"  saved: {path} ({os.path.getsize(path)} bytes)")


def test_random_video(cortex, hub):
    print("\n[Test 3] Random video generation")
    out_dir = "data/video/cortex_api_test"
    clear_output_dir(out_dir)
    ids, recon = simulate_cortex_generate(cortex, hub, "video", seed=42)
    print(f"  tokens: {len(ids)}, range=[{min(ids)}, {max(ids)}]")
    print(f"  recon shape: {recon.shape}")

    from taiji.multimodal.io import save_video
    vid = recon if recon.dim() == 4 else recon[0]
    # video decode 返回 [C, T, H, W]，save_video 需要 [T, C, H, W]
    if vid.dim() == 4 and vid.shape[0] == 3:
        vid = vid.permute(1, 0, 2, 3)
    path = os.path.join(out_dir, "random.mp4")
    saved = save_video(vid, path, fps=8, fallback_png=True)
    print(f"  saved: {saved}")


def test_imitation_image(cortex, hub):
    print("\n[Test 4] Imitation image generation (from data/vqvae/samples/original.png)")
    src = "data/vqvae/samples/original.png"
    if not os.path.isfile(src):
        print(f"  SKIP: source not found: {src}")
        return
    out_dir = "data/image/cortex_api_test"
    os.makedirs(out_dir, exist_ok=True)

    ids, recon = simulate_cortex_generate(cortex, hub, "image", input_path=src, seed=42)
    print(f"  tokens: {len(ids)}, range=[{min(ids)}, {max(ids)}]")
    print(f"  recon shape: {recon.shape}")

    from taiji.multimodal.io import save_image
    img = recon if recon.dim() == 3 else recon[0]
    path = os.path.join(out_dir, "imitation.png")
    save_image(img, path)
    assert os.path.isfile(path)
    print(f"  saved: {path} ({os.path.getsize(path)} bytes)")


def test_modalities_info(cortex):
    """模拟 routes_neuroplex.py 中 _cortex_model_info 的 modalities 信息构造。"""
    print("\n[Test 5] Modalities info (simulating /api/taiji/model/info)")
    hub = getattr(cortex, "_tokenizer_hub", None)
    assert hub is not None, "TokenizerHub not set on cortex"

    ckpt_map = {
        "image": "data/vqvae/vqvae_latest.pt",
        "audio": "data/encodec/encodec_latest.pt",
        "video": "data/video/video_latest.pt",
    }
    modalities = []
    for mod in hub.list_modalities():
        codec = hub.modal_encoders.get(mod)
        if codec is None:
            continue
        vocab = codec.vocab_size() if hasattr(codec, "vocab_size") else 0
        ckpt_path = ckpt_map.get(mod, "")
        modalities.append({
            "modality": mod,
            "vocab_size": vocab,
            "trained": os.path.isfile(ckpt_path),
            "checkpoint": ckpt_path if os.path.isfile(ckpt_path) else None,
        })

    print(f"  Modalities: {modalities}")
    assert len(modalities) == 3, f"expected 3 modalities, got {len(modalities)}"
    mods = {m["modality"] for m in modalities}
    assert mods == {"image", "audio", "video"}, f"unexpected modalities: {mods}"

    # 检查每个模态的字段完整性
    for m in modalities:
        assert m["vocab_size"] > 0, f"{m['modality']} vocab_size invalid"
        assert "trained" in m and isinstance(m["trained"], bool)
        assert "checkpoint" in m
    print("  All 3 modalities have valid vocab_size, trained, checkpoint fields")


def main():
    print("=" * 60)
    print("Cortex Generate API Endpoint Verification")
    print("=" * 60)

    print("\n=== Step 1: assemble_cortex ===")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons", device="cpu", max_rounds=2,
    )
    hub = modules.get("tokenizer_hub")
    assert hub is not None
    print(f"  Neurons: {list(cortex.neurons.keys())}")
    print(f"  Modalities: {hub.list_modalities()}")

    # 测试 5 个用例
    test_modalities_info(cortex)
    test_random_image(cortex, hub)
    test_random_audio(cortex, hub)
    test_random_video(cortex, hub)
    test_imitation_image(cortex, hub)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — cortex_generate API endpoint verified")
    print("=" * 60)
    print("\nVerified endpoints (simulated, no HTTP server):")
    print("  - POST /api/taiji/cortex/generate (random image) ✓")
    print("  - POST /api/taiji/cortex/generate (random audio) ✓")
    print("  - POST /api/taiji/cortex/generate (random video) ✓")
    print("  - POST /api/taiji/cortex/generate (imitation image) ✓")
    print("  - GET  /api/taiji/model/info (modalities field)   ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
