"""P8: Multimodal end-to-end verification.

验证完整链路：
  raw input → codec.encode → neuron.forward(mm_logits_modality) → decode → output

即使 codec 未训练，验证 wiring 是否正确连接。

Usage:
    python scripts/training/verify_multimodal.py
"""
from __future__ import annotations

import os
import sys
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print = functools.partial(print, flush=True)

import torch
import torch.nn as nn

from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import get_default_neuron_config
from taiji.multimodal.vqvae import VQVAE, VQVAEImageCodec
from taiji.multimodal.encodec import EnCodec, EnCodecAudioCodec
from taiji.multimodal.video import VideoVQVAE, VideoCodec


def verify_image_pathway():
    print("\n" + "=" * 60)
    print("Step 1: Image pathway (VQ-VAE)")
    print("=" * 60)

    device = "cpu"

    model = VQVAE(in_channels=3, hidden_dim=64, latent_dim=256, num_embeddings=8192, downsample=4)
    codec = VQVAEImageCodec(model=model, image_size=64, device=device)

    dummy_img = torch.rand(3, 64, 64).clamp(0, 1)

    ids = codec.encode(dummy_img)
    print(f"  encode: {len(ids)} tokens")

    recon = codec.decode(ids)
    print(f"  decode: shape={recon.shape}, range=[{recon.min().item():.2f}, {recon.max().item():.2f}]")

    cfg = get_default_neuron_config()
    cfg.spec = "image-fallback"
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.register_modality_projection("image", raw_dim=256)
    neuron.register_modality_lm_head("image", vocab_size=8192)

    features = model.encoder(dummy_img.unsqueeze(0))
    features = features.permute(0, 2, 3, 1).flatten(1, 2)

    shared_emb = neuron.encode_multimodal_input(features, modality="image")
    print(f"  encode_multimodal_input: shape={shared_emb.shape}")

    result = neuron.forward(shared_emb, mm_logits_modality="image")
    print(f"  forward with mm_logits_modality='image':")
    print(f"    field_vector: shape={result['field_vector'].shape}")
    print(f"    logits: shape={result['logits'].shape}")

    print("  ✅ Image pathway verified")


def verify_audio_pathway():
    print("\n" + "=" * 60)
    print("Step 2: Audio pathway (EnCodec)")
    print("=" * 60)

    device = "cpu"

    model = EnCodec(hidden_dim=64, latent_dim=128, num_embeddings=4096)
    codec = EnCodecAudioCodec(model=model, sample_rate=16000, device=device)

    dummy_audio = torch.rand(16000).clamp(-1, 1)

    ids = codec.encode(dummy_audio)
    print(f"  encode: {len(ids)} tokens")

    recon = codec.decode(ids)
    print(f"  decode: shape={recon.shape}, range=[{recon.min().item():.2f}, {recon.max().item():.2f}]")

    cfg = get_default_neuron_config()
    cfg.spec = "audio-fallback"
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.register_modality_projection("audio", raw_dim=128)
    neuron.register_modality_lm_head("audio", vocab_size=4096)

    features = model.encoder(dummy_audio.unsqueeze(0).unsqueeze(0))
    features = features.permute(0, 2, 1)

    shared_emb = neuron.encode_multimodal_input(features, modality="audio")
    print(f"  encode_multimodal_input: shape={shared_emb.shape}")

    result = neuron.forward(shared_emb, mm_logits_modality="audio")
    print(f"  forward with mm_logits_modality='audio':")
    print(f"    field_vector: shape={result['field_vector'].shape}")
    print(f"    logits: shape={result['logits'].shape}")

    print("  ✅ Audio pathway verified")


def verify_video_pathway():
    print("\n" + "=" * 60)
    print("Step 3: Video pathway (VideoVQVAE)")
    print("=" * 60)

    device = "cpu"

    model = VideoVQVAE(hidden_dim=64, latent_dim=256, num_embeddings=256)
    codec = VideoCodec(model=model, frame_size=32, num_frames=16, device=device)

    dummy_video = torch.rand(3, 16, 32, 32).clamp(0, 1)

    ids = codec.encode(dummy_video)
    print(f"  encode: {len(ids)} tokens")

    recon = codec.decode(ids)
    print(f"  decode: shape={recon.shape}, range=[{recon.min().item():.2f}, {recon.max().item():.2f}]")

    cfg = get_default_neuron_config()
    cfg.spec = "video-fallback"
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.register_modality_projection("video", raw_dim=256)
    neuron.register_modality_lm_head("video", vocab_size=256)

    features = model.encoder(dummy_video.unsqueeze(0))
    features = features.permute(0, 2, 3, 4, 1).flatten(1, 3)

    shared_emb = neuron.encode_multimodal_input(features, modality="video")
    print(f"  encode_multimodal_input: shape={shared_emb.shape}")

    result = neuron.forward(shared_emb, mm_logits_modality="video")
    print(f"  forward with mm_logits_modality='video':")
    print(f"    field_vector: shape={result['field_vector'].shape}")
    print(f"    logits: shape={result['logits'].shape}")

    print("  ✅ Video pathway verified")


def verify_cortex_wiring():
    print("\n" + "=" * 60)
    print("Step 4: Cortex multimodal wiring")
    print("=" * 60)

    from taiji.loader import assemble_cortex

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/verify_v3",
        device="cpu",
        max_rounds=2,
    )

    print(f"  Neurons loaded: {list(cortex.neurons.keys())}")
    print(f"  Modules wired: {list(modules.keys())}")

    if cortex.neurons:
        first_nid = list(cortex.neurons.keys())[0]
        neuron = cortex.neurons[first_nid]
        print(f"  First neuron [{first_nid}]:")
        print(f"    mm_projections: {list(neuron.mm_projections.keys())}")
        print(f"    mm_lm_heads: {list(neuron.mm_lm_heads.keys())}")

    if "tokenizer_hub" in modules:
        hub = modules["tokenizer_hub"]
        print(f"  TokenizerHub domains: {hub.list_domains()}")
        print(f"  TokenizerHub modalities: {hub.list_modalities()}")

    print("  ✅ Cortex multimodal wiring verified")


def main():
    print("Multimodal End-to-End Verification")
    print("=" * 60)
    print("Verifying image/audio/video pathways with untrained codecs")

    verify_image_pathway()
    verify_audio_pathway()
    verify_video_pathway()
    verify_cortex_wiring()

    print("\n" + "=" * 60)
    print("ALL PATHWAYS VERIFIED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train VQ-VAE: python scripts/training/train_vqvae.py --steps 2000")
    print("  2. Train EnCodec: python scripts/training/train_encodec.py --steps 2000")
    print("  3. Train VideoVQVAE: python scripts/training/train_video.py --steps 2000")


if __name__ == "__main__":
    main()
