"""P8: 训练 EnCodec 音频编解码器。

让 EnCodecAudioCodec 具备实际重建能力，为后续音频生成 neuron 的 sleep 训练提供预训练 codec。

用法:
    # 合成音频快速验证（无外部依赖）
    python scripts/training/train_encodec.py --steps 2000

    # 加载已有 checkpoint 继续训练
    python scripts/training/train_encodec.py --resume data/encodec/encodec_latest.pt --steps 4000

    # 评估 PSNR（不训练）
    python scripts/training/train_encodec.py --eval-only --resume data/encodec/encodec_latest.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.multimodal.encodec import EnCodec
from neuroplex.multimodal.io import save_audio

# ── 默认超参数 ──
BATCH_SIZE = 8
AUDIO_LEN = 16000          # 1 秒 @ 16kHz
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
DEFAULT_STEPS = 2000
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
NUM_EMBEDDINGS = 4096      # 与 tokenizer_contract.json 对齐
LATENT_DIM = 128
HIDDEN_DIM = 64
COMMITMENT_COST = 0.25
SAMPLE_RATE = 16000

OUTPUT_DIR = "data/encodec"


def synthesize_batch(batch_size: int, audio_len: int, device: torch.device) -> torch.Tensor:
    """合成音频 batch（无外部数据依赖）。

    生成多样化音频信号：
    - 单频正弦波（不同频率/相位）
    - 多频叠加（和弦感）
    - 扫频信号
    - 调幅信号
    - ADSR 包络音符
    """
    t = torch.arange(audio_len, device=device, dtype=torch.float32) / SAMPLE_RATE
    batch = []

    for _ in range(batch_size):
        sig_type = torch.randint(0, 5, (1,)).item()

        if sig_type == 0:
            # 单频正弦波
            freq = torch.rand(1).item() * 800 + 100  # 100-900 Hz
            phase = torch.rand(1).item() * 2 * math.pi
            sig = torch.sin(2 * math.pi * freq * t + phase)
        elif sig_type == 1:
            # 多频叠加（3 个谐波）
            f0 = torch.rand(1).item() * 300 + 100
            sig = (torch.sin(2 * math.pi * f0 * t) +
                   0.5 * torch.sin(2 * math.pi * 2 * f0 * t) +
                   0.3 * torch.sin(2 * math.pi * 3 * f0 * t))
            sig = sig / 1.8
        elif sig_type == 2:
            # 扫频信号（线性 chirp）
            f_start = torch.rand(1).item() * 200 + 100
            f_end = f_start + torch.rand(1).item() * 800 + 200
            freq = f_start + (f_end - f_start) * t / t[-1]
            # 累积相位
            phase = 2 * math.pi * torch.cumsum(freq, dim=0) / SAMPLE_RATE
            sig = torch.sin(phase)
        elif sig_type == 3:
            # 调幅信号
            carrier = torch.rand(1).item() * 500 + 300
            modulator = torch.rand(1).item() * 10 + 2
            sig = (1 + 0.5 * torch.sin(2 * math.pi * modulator * t)) * \
                  torch.sin(2 * math.pi * carrier * t) * 0.5
        else:
            # ADSR 包络音符（多音符序列）
            sig = torch.zeros(audio_len, device=device)
            n_notes = torch.randint(2, 5, (1,)).item()
            note_len = audio_len // n_notes
            for i in range(n_notes):
                freq = torch.rand(1).item() * 600 + 200
                start = i * note_len
                end = start + note_len
                note_t = torch.arange(note_len, device=device) / SAMPLE_RATE
                # ADSR: 10% attack, 40% decay/sustain, 50% release
                env = torch.ones(note_len, device=device)
                a = note_len // 10
                d = note_len // 5
                r = note_len // 2
                env[:a] = torch.linspace(0, 1, a)
                env[a:a+d] = torch.linspace(1, 0.7, d)
                env[-r:] = torch.linspace(0.7, 0, r)
                sig[start:end] = env * torch.sin(2 * math.pi * freq * note_t)

        # 归一化到 [-1, 1]
        if sig.abs().max() > 0:
            sig = sig / sig.abs().max()
        batch.append(sig)

    return torch.stack(batch)  # [B, L]


def compute_psnr(model: EnCodec, device: torch.device, n_samples: int = 100,
                 audio_len: int = AUDIO_LEN) -> float:
    """评估模型重建质量（PSNR dB）。

    PSNR > 20dB = 可听清
    PSNR > 30dB = 良好重建
    PSNR < 15dB = 失真严重
    """
    model.eval()
    total_mse = 0.0
    count = 0
    batch_size = 10

    with torch.no_grad():
        for _ in range(n_samples // batch_size):
            batch = synthesize_batch(batch_size, audio_len, device)
            recon, _, _ = model(batch.unsqueeze(1))  # [B, 1, L]
            # recon 和 batch 都是 [-1, 1]
            mse = F.mse_loss(recon.squeeze(1), batch).item()
            total_mse += mse * batch_size
            count += batch_size

    avg_mse = total_mse / count
    if avg_mse < 1e-10:
        return 100.0
    # 音频归一化到 [-1,1]，最大功率=1，PSNR=10*log10(1/MSE)
    psnr = 10 * math.log10(1.0 / avg_mse)
    model.train()
    return psnr


def train(
    steps: int,
    batch_size: int,
    audio_len: int,
    lr: float,
    output_dir: str,
    resume: str | None,
    device: torch.device,
    log_interval: int = LOG_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
    eval_psnr: bool = True,
):
    """训练 EnCodec。"""
    print(f"EnCodec 训练配置:")
    print(f"  steps={steps}, batch_size={batch_size}, audio_len={audio_len}")
    print(f"  lr={lr}, device={device}, sample_rate={SAMPLE_RATE}")
    print(f"  output={output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建模型
    model = EnCodec(
        in_channels=1,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        commitment_cost=COMMITMENT_COST,
        sample_rate=SAMPLE_RATE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    # 2. 加载 checkpoint
    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"从 checkpoint 恢复: {resume} (step={start_step})")

    model.train()

    # 3. 优化器（排除 codebook 参数，codebook 由 EMA 更新）
    codebook_params = set()
    for mod in model.modules():
        if hasattr(mod, "codebook") and isinstance(mod.codebook, nn.Embedding):
            codebook_params.add(id(mod.codebook.weight))
    trainable = [p for p in model.parameters() if id(p) not in codebook_params]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)

    # 4. 训练循环
    print(f"\n开始训练（step {start_step} → {steps}）...")
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    log_count = 0
    start_time = time.time()

    for step in range(start_step, steps):
        # 4.1 合成 batch
        batch = synthesize_batch(batch_size, audio_len, device)
        x = batch.unsqueeze(1)  # [B, 1, L]

        # 4.2 Forward
        optimizer.zero_grad()
        recon, indices, vq_loss = model(x)

        # 4.3 Loss: reconstruction (MSE) + VQ
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss

        # 4.4 Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        log_count += 1

        # 4.5 日志
        if (step + 1) % log_interval == 0:
            avg_recon = total_recon_loss / log_count
            avg_vq = total_vq_loss / log_count
            elapsed = time.time() - start_time
            steps_per_sec = log_count / elapsed
            unique_codes = len(torch.unique(indices))
            utilization = unique_codes / NUM_EMBEDDINGS * 100

            psnr_str = ""
            if eval_psnr and (step + 1) % (log_interval * 4) == 0:
                psnr = compute_psnr(model, device, n_samples=50, audio_len=audio_len)
                psnr_str = f" | PSNR={psnr:.1f}dB"
                model.train()

            print(f"  step {step + 1}/{steps} | "
                  f"recon={avg_recon:.4f} vq={avg_vq:.4f} | "
                  f"codebook: {unique_codes}/{NUM_EMBEDDINGS} ({utilization:.1f}%){psnr_str} | "
                  f"{steps_per_sec:.1f} steps/s")

            total_recon_loss = 0.0
            total_vq_loss = 0.0
            log_count = 0
            start_time = time.time()

        # 4.6 保存 checkpoint
        if (step + 1) % save_interval == 0 or step + 1 == steps:
            ckpt_path = os.path.join(output_dir, "encodec_latest.pt")
            final_psnr = 0.0
            if eval_psnr:
                final_psnr = compute_psnr(model, device, n_samples=100, audio_len=audio_len)
                model.train()
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "config": {
                    "audio_len": audio_len,
                    "num_embeddings": NUM_EMBEDDINGS,
                    "latent_dim": LATENT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "commitment_cost": COMMITMENT_COST,
                    "sample_rate": SAMPLE_RATE,
                },
                "psnr": final_psnr,
            }, ckpt_path)
            print(f"  💾 checkpoint saved: {ckpt_path} (PSNR={final_psnr:.1f}dB)")

    print(f"\n训练完成！checkpoint: {os.path.join(output_dir, 'encodec_latest.pt')}")


def main():
    parser = argparse.ArgumentParser(description="训练 EnCodec 音频编解码器")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"训练步数（默认 {DEFAULT_STEPS}）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"batch size（默认 {BATCH_SIZE}）")
    parser.add_argument("--audio-len", type=int, default=AUDIO_LEN,
                        help=f"音频长度（采样点数，默认 {AUDIO_LEN}）")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"学习率（默认 {LR}）")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"输出目录（默认 {OUTPUT_DIR}）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备（auto/cpu/cuda）")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                        help=f"日志间隔（默认 {LOG_INTERVAL}）")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                        help=f"保存间隔（默认 {SAVE_INTERVAL}）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估 PSNR（不训练）")
    parser.add_argument("--no-psnr", action="store_true",
                        help="禁用训练中 PSNR 评估（加速）")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 仅评估模式
    if args.eval_only:
        if not args.resume or not os.path.exists(args.resume):
            print("Error: --eval-only 需要 --resume 指定 checkpoint")
            return
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        model = EnCodec(
            hidden_dim=cfg_dict.get("hidden_dim", HIDDEN_DIM),
            latent_dim=cfg_dict.get("latent_dim", LATENT_DIM),
            num_embeddings=cfg_dict.get("num_embeddings", NUM_EMBEDDINGS),
            commitment_cost=cfg_dict.get("commitment_cost", COMMITMENT_COST),
            sample_rate=cfg_dict.get("sample_rate", SAMPLE_RATE),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        psnr = compute_psnr(model, device, n_samples=200, audio_len=args.audio_len)
        print(f"PSNR: {psnr:.2f} dB (checkpoint: {args.resume}, step={ckpt.get('step', '?')})")
        print(f"  {'✅ 可听清 (>20dB)' if psnr > 20 else '⚠ 失真严重 (<15dB)' if psnr < 15 else '🔧 待提升 (15-20dB)'}")
        return

    train(
        steps=args.steps,
        batch_size=args.batch_size,
        audio_len=args.audio_len,
        lr=args.lr,
        output_dir=args.output_dir,
        resume=args.resume,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_psnr=not args.no_psnr,
    )


if __name__ == "__main__":
    main()
