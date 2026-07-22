"""P8: 训练 VideoVQVAE 视频编解码器。

让 VideoCodec 具备实际重建能力，为后续视频生成 neuron 的 sleep 训练提供预训练 codec。

用法:
    # 合成视频快速验证（无外部依赖）
    python scripts/training/train_video.py --steps 2000

    # 加载已有 checkpoint 继续训练
    python scripts/training/train_video.py --resume data/video/video_latest.pt --steps 4000

    # 评估 PSNR（不训练）
    python scripts/training/train_video.py --eval-only --resume data/video/video_latest.pt
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

from taiji.multimodal.video import VideoVQVAE
from taiji.multimodal.io import save_video

# ── 默认超参数 ──
BATCH_SIZE = 4
NUM_FRAMES = 16
FRAME_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
DEFAULT_STEPS = 2000
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
NUM_EMBEDDINGS = 256   # 软分配熵正则防崩塌，用更大 codebook 提升重建质量
LATENT_DIM = 256
HIDDEN_DIM = 64
COMMITMENT_COST = 0.25

OUTPUT_DIR = "data/video"


def synthesize_batch(batch_size: int, num_frames: int, frame_size: int,
                     device: torch.device) -> torch.Tensor:
    """合成视频 batch（简化版，2 种类型便于学习）。

    - 移动色块（不同速度/方向/颜色）
    - 渐变扫光（水平/垂直/对角）
    """
    batch = []

    for _ in range(batch_size):
        vid_type = torch.randint(0, 2, (1,)).item()

        # 随机背景色 [3,1,1,1] → 扩展到 [3,T,H,W]
        bg_color = torch.rand(3, 1, 1, 1, device=device) * 0.4 + 0.3
        video = bg_color.expand(3, num_frames, frame_size, frame_size).clone()

        if vid_type == 0:
            # 移动色块
            color = torch.rand(3, 1, 1, device=device)
            size = torch.randint(4, 10, (1,)).item()
            x0 = torch.randint(0, frame_size - size, (1,)).item()
            y0 = torch.randint(0, frame_size - size, (1,)).item()
            x1 = torch.randint(0, frame_size - size, (1,)).item()
            y1 = torch.randint(0, frame_size - size, (1,)).item()
            for t in range(num_frames):
                alpha = t / max(num_frames - 1, 1)
                cx = int(x0 + (x1 - x0) * alpha)
                cy = int(y0 + (y1 - y0) * alpha)
                video[:, t, cy:cy+size, cx:cx+size] = color

        else:
            # 渐变扫光
            direction = torch.randint(0, 3, (1,)).item()  # 0=水平, 1=垂直, 2=对角
            speed = torch.rand(1).item() * 0.5 + 0.5
            for t in range(num_frames):
                phase = (t / num_frames * speed) % 1.0
                if direction == 0:
                    grad = torch.linspace(0, 1, frame_size, device=device) + phase
                elif direction == 1:
                    grad = torch.linspace(0, 1, frame_size, device=device).view(-1, 1) + phase
                else:
                    g1 = torch.linspace(0, 1, frame_size, device=device)
                    grad = (g1.view(-1, 1) + g1.view(1, -1)) / 2 + phase
                grad = (grad % 1.0).unsqueeze(0).expand(3, -1, -1)
                video[:, t] = video[:, t] * 0.3 + grad * 0.7

        video = video.clamp(0, 1)
        batch.append(video)

    return torch.stack(batch)  # [B, 3, T, H, W]


def compute_psnr(model: VideoVQVAE, device: torch.device, n_samples: int = 20,
                 num_frames: int = NUM_FRAMES, frame_size: int = FRAME_SIZE) -> float:
    """评估模型重建质量（PSNR dB）。"""
    model.eval()
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(n_samples):
            batch = synthesize_batch(1, num_frames, frame_size, device)
            recon, _, _ = model(batch)
            mse = F.mse_loss(recon, batch).item()
            total_mse += mse
            count += 1

    avg_mse = total_mse / count
    if avg_mse < 1e-10:
        return 100.0
    psnr = 10 * math.log10(1.0 / avg_mse)
    model.train()
    return psnr


def train(
    steps: int,
    batch_size: int,
    num_frames: int,
    frame_size: int,
    lr: float,
    output_dir: str,
    resume: str | None,
    device: torch.device,
    log_interval: int = LOG_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
    eval_psnr: bool = True,
    diversity_cost: float = 0.1,
):
    """训练 VideoVQVAE。"""
    print(f"VideoVQVAE 训练配置:")
    print(f"  steps={steps}, batch_size={batch_size}")
    print(f"  num_frames={num_frames}, frame_size={frame_size}")
    print(f"  lr={lr}, device={device}, diversity_cost={diversity_cost}")
    print(f"  output={output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    model = VideoVQVAE(
        in_channels=3,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        commitment_cost=COMMITMENT_COST,
    ).to(device)
    # 覆盖量化器的 diversity_cost（支持训练阶段切换）
    model.quantizer.diversity_cost = diversity_cost

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"从 checkpoint 恢复: {resume} (step={start_step})")

    model.train()

    # 优化器（EMA codebook：排除 codebook 参数，由 EMA 更新）
    codebook_params = set()
    for mod in model.modules():
        if hasattr(mod, "codebook") and isinstance(mod.codebook, nn.Embedding):
            codebook_params.add(id(mod.codebook.weight))
    trainable = [p for p in model.parameters() if id(p) not in codebook_params]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)

    print(f"\n开始训练（step {start_step} → {steps}）...")
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    log_count = 0
    start_time = time.time()

    for step in range(start_step, steps):
        batch = synthesize_batch(batch_size, num_frames, frame_size, device)

        optimizer.zero_grad()
        recon, indices, vq_loss = model(batch)
        recon_loss = F.mse_loss(recon, batch)
        loss = recon_loss + vq_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        log_count += 1

        if (step + 1) % log_interval == 0:
            avg_recon = total_recon_loss / log_count
            avg_vq = total_vq_loss / log_count
            elapsed = time.time() - start_time
            steps_per_sec = log_count / elapsed
            unique_codes = len(torch.unique(indices))
            utilization = unique_codes / NUM_EMBEDDINGS * 100

            psnr_str = ""
            if eval_psnr and (step + 1) % (log_interval * 4) == 0:
                psnr = compute_psnr(model, device, n_samples=10,
                                    num_frames=num_frames, frame_size=frame_size)
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

        if (step + 1) % save_interval == 0 or step + 1 == steps:
            ckpt_path = os.path.join(output_dir, "video_latest.pt")
            final_psnr = 0.0
            if eval_psnr:
                final_psnr = compute_psnr(model, device, n_samples=20,
                                          num_frames=num_frames, frame_size=frame_size)
                model.train()
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_frames": num_frames,
                    "frame_size": frame_size,
                    "num_embeddings": NUM_EMBEDDINGS,
                    "latent_dim": LATENT_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "commitment_cost": COMMITMENT_COST,
                },
                "psnr": final_psnr,
            }, ckpt_path)
            print(f"  💾 checkpoint saved: {ckpt_path} (PSNR={final_psnr:.1f}dB)")

    print(f"\n训练完成！checkpoint: {os.path.join(output_dir, 'video_latest.pt')}")


def main():
    parser = argparse.ArgumentParser(description="训练 VideoVQVAE 视频编解码器")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-psnr", action="store_true")
    parser.add_argument("--diversity-cost", type=float, default=0.1,
                        help="软分配熵正则权重（0=关闭，0.1=默认，第一阶段防崩塌；第二阶段降到0.01专注重建）")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.eval_only:
        if not args.resume or not os.path.exists(args.resume):
            print("Error: --eval-only 需要 --resume 指定 checkpoint")
            return
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        model = VideoVQVAE(
            hidden_dim=cfg_dict.get("hidden_dim", HIDDEN_DIM),
            latent_dim=cfg_dict.get("latent_dim", LATENT_DIM),
            num_embeddings=cfg_dict.get("num_embeddings", NUM_EMBEDDINGS),
            commitment_cost=cfg_dict.get("commitment_cost", COMMITMENT_COST),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        psnr = compute_psnr(model, device, n_samples=50,
                            num_frames=args.num_frames, frame_size=args.frame_size)
        print(f"PSNR: {psnr:.2f} dB (checkpoint: {args.resume}, step={ckpt.get('step', '?')})")
        print(f"  {'✅ 可识别 (>20dB)' if psnr > 20 else '⚠ 失真严重 (<15dB)' if psnr < 15 else '🔧 待提升 (15-20dB)'}")
        return

    train(
        steps=args.steps,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        lr=args.lr,
        output_dir=args.output_dir,
        resume=args.resume,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_psnr=not args.no_psnr,
        diversity_cost=args.diversity_cost,
    )


if __name__ == "__main__":
    main()
